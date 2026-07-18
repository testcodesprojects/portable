/**
 * @file    dpotrf_dense_gpu.cpp
 * @brief   GPU dense Cholesky expansion kernels (4 variants).
 *
 * Mirrors tools/compute/pdpotrf.cpp lines 213-510: the same task switch
 * (POTRF / SYRK / TRSM / GEMM), the same flag-based dependency tracking
 * (ss_/dep_), but kernels run on the GPU with cross-rank ordering enforced
 * by cudaEvent_t paired to every ss_cond_set / dep_set_done.
 */
#ifdef STILES_GPU

#include "dpotrf_dense_gpu.hpp"
#include "gpu_persistent_context.hpp"
#include "gpu_tile_slab.hpp"

#include <omp.h>
#include <array>
#include <vector>
#include <cstdio>

// Same includes pdpotrf.cpp uses so we get the task list, flag macros, and
// the TiledMatrix layout. These compile in the live tree; in this sandbox
// they will resolve only when the surrounding build includes them.
#include "../../control/common.h"           // STILES_RANK, ss_*, ss_init/finalize
#include "../../control/stiles_control.hpp" // dep_*, dep_init/finalize
#include "../../compute/stiles_compute.hpp" // sTiles::get_chol_tasks, get_chol_task_offsets
#include "../../common/stiles_types.hpp"    // TiledMatrix, DenseGpuTile

namespace sTiles { namespace gpu { namespace dpotrf {


// ---------------------------------------------------------------------------
//  slab <-> scheme binding
// ---------------------------------------------------------------------------
void bind_slab_to_scheme(GpuTileSlab& slab, TiledMatrix* scheme) {
    if (!scheme || !scheme->dense_tiles_gpu) {
        std::fprintf(stderr, "[gpu_dpotrf] bind_slab_to_scheme: missing dense_tiles_gpu.\n");
        std::abort();
    }
    const int n = scheme->numActiveTiles;
    for (int i = 0; i < n; ++i) {
        scheme->dense_tiles_gpu[i].x      = slab_tile_ptr(slab, i);
        // Width/height: prefer tileMetaCore when present, else nominal tile_size.
        if (scheme->tileMetaCore) {
            int w = scheme->tileMetaCore[i].width;
            int h = scheme->tileMetaCore[i].height;
            scheme->dense_tiles_gpu[i].width  = (w > 0) ? w : slab.tile_size;
            scheme->dense_tiles_gpu[i].height = (h > 0) ? h : slab.tile_size;
        } else {
            scheme->dense_tiles_gpu[i].width  = slab.tile_size;
            scheme->dense_tiles_gpu[i].height = slab.tile_size;
        }
    }
}


// ---------------------------------------------------------------------------
//  shared helpers (one stream, no events) for the serial variants
// ---------------------------------------------------------------------------
namespace {

inline void enqueue_potrf(GpuPersistentContext& ctx, cudaStream_t stream,
                          int n, double* A, int lda) {
    STILES_CUSOLVER_CHECK(cusolverDnSetStream(ctx.cusolverH, stream));
    STILES_CUSOLVER_CHECK(cusolverDnDpotrf(ctx.cusolverH,
        CUBLAS_FILL_MODE_UPPER, n, A, lda,
        ctx.d_workspace, ctx.workspace_len, ctx.d_devinfo));
}

inline void enqueue_syrk(GpuPersistentContext& ctx, cudaStream_t stream,
                         int n, int k,
                         double alpha, const double* A, int lda,
                         double beta,  double* C, int ldc) {
    STILES_CUBLAS_CHECK(cublasSetStream(ctx.cublasH, stream));
    STILES_CUBLAS_CHECK(cublasDsyrk(ctx.cublasH,
        CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
        n, k, &alpha, A, lda, &beta, C, ldc));
}

inline void enqueue_trsm(GpuPersistentContext& ctx, cudaStream_t stream,
                         int m, int n,
                         double alpha, const double* A, int lda,
                         double* B, int ldb) {
    STILES_CUBLAS_CHECK(cublasSetStream(ctx.cublasH, stream));
    STILES_CUBLAS_CHECK(cublasDtrsm(ctx.cublasH,
        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
        m, n, &alpha, A, lda, B, ldb));
}

inline void enqueue_gemm(GpuPersistentContext& ctx, cudaStream_t stream,
                         int m, int n, int k,
                         double alpha, const double* A, int lda,
                                       const double* B, int ldb,
                         double beta,  double* C, int ldc) {
    STILES_CUBLAS_CHECK(cublasSetStream(ctx.cublasH, stream));
    STILES_CUBLAS_CHECK(cublasDgemm(ctx.cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

// Sync the producer stream and read d_devinfo. Call only when correctness
// requires us to know the factor succeeded (e.g., at end-of-call, or right
// before a host-side ss_cond_set that downstream ranks gate on).
inline int sync_and_read_devinfo(GpuPersistentContext& ctx, cudaStream_t stream) {
    STILES_CUDA_CHECK(cudaStreamSynchronize(stream));
    int h_info = 0;
    STILES_CUDA_CHECK(cudaMemcpy(&h_info, ctx.d_devinfo, sizeof(int),
                                 cudaMemcpyDeviceToHost));
    return h_info;
}

} // namespace


// ---------------------------------------------------------------------------
//  1. Serial pthreads dispatch (single stream)
// ---------------------------------------------------------------------------
void gpu_dpotrf_expansion_dense_serial(TiledMatrix*          scheme,
                                       stiles_context_t*     stile,
                                       GpuPersistentContext& ctx) {
    const int N         = scheme->dim;
    const int tile_size = scheme->tile_size;
    const int nt        = (N == 0) ? 0 : (N - 1) / tile_size + 1;
    const int full_tpd  = (tile_size > 0) ? (N / tile_size) : 0;

    auto BLK = [&](int k) { return (k < full_tpd) ? tile_size : (N % tile_size); };

    const auto& tasks = sTiles::get_chol_tasks(scheme);
    const int ntasks  = static_cast<int>(tasks.size());

    cudaStream_t stream = ctx.streams[0];

    constexpr double zone  =  1.0;
    constexpr double mzone = -1.0;

    for (int idx = 0; idx < ntasks; ++idx) {
        const std::array<int,7>& t = tasks[idx];
        const int rkind  = t[0];
        const int m      = t[1];
        const int k      = t[2];
        const int n      = t[3];
        const int idx1   = t[4];
        const int idx2   = t[5];
        const int idx3   = t[6];

        const int tempkn = (k == nt - 1) ? (N - k * tile_size) : tile_size;
        const int tempmn = (m == nt - 1) ? (N - m * tile_size) : tile_size;
        const int ldak   = BLK(k);
        const int ldan   = BLK(n);

        double* A1 = scheme->dense_tiles_gpu[idx1].x;
        double* A2 = (idx2 >= 0) ? scheme->dense_tiles_gpu[idx2].x : nullptr;
        double* A3 = (idx3 >= 0) ? scheme->dense_tiles_gpu[idx3].x : nullptr;

        switch (rkind) {
            case 1:
                enqueue_potrf(ctx, stream, tempkn, A1, ldak);
                if (sync_and_read_devinfo(ctx, stream) != 0) {
                    std::fprintf(stderr, "sTiles GPU: matrix not PD at tile k=%d\n", k);
                    stile->ss_abort = 1;
                    return;
                }
                break;
            case 2:
                if (A1 && A2)
                    enqueue_syrk(ctx, stream, tempkn, tile_size,
                                 mzone, A1, ldan, zone, A2, ldak);
                break;
            case 3:
                if (A1 && A2)
                    enqueue_trsm(ctx, stream, tile_size, tempmn,
                                 zone, A2, ldak, A1, ldak);
                break;
            case 4:
                if (A1 && A2 && A3)
                    enqueue_gemm(ctx, stream, tile_size, tempmn, tile_size,
                                 mzone, A1, ldan, A2, ldan, zone, A3, ldak);
                break;
            default:
                break;
        }
    }

    STILES_CUDA_CHECK(cudaStreamSynchronize(stream));
}


// ---------------------------------------------------------------------------
//  2. Serial OMP dispatch (single stream)
// ---------------------------------------------------------------------------
void gpu_dpotrf_expansion_dense_serial_omp(TiledMatrix*          scheme,
                                           omp_dep_tracker_t*    dep_tracker,
                                           GpuPersistentContext& ctx) {
    const int N         = scheme->dim;
    const int tile_size = scheme->tile_size;
    const int nt        = (N == 0) ? 0 : (N - 1) / tile_size + 1;
    const int full_tpd  = (tile_size > 0) ? (N / tile_size) : 0;

    auto BLK = [&](int k) { return (k < full_tpd) ? tile_size : (N % tile_size); };

    const auto& tasks = sTiles::get_chol_tasks(scheme);
    const int ntasks  = static_cast<int>(tasks.size());

    cudaStream_t stream = ctx.streams[0];

    constexpr double zone  =  1.0;
    constexpr double mzone = -1.0;

    for (int idx = 0; idx < ntasks; ++idx) {
        const std::array<int,7>& t = tasks[idx];
        const int rkind  = t[0];
        const int m      = t[1];
        const int k      = t[2];
        const int n      = t[3];
        const int idx1   = t[4];
        const int idx2   = t[5];
        const int idx3   = t[6];

        const int tempkn = (k == nt - 1) ? (N - k * tile_size) : tile_size;
        const int tempmn = (m == nt - 1) ? (N - m * tile_size) : tile_size;
        const int ldak   = BLK(k);
        const int ldan   = BLK(n);

        double* A1 = scheme->dense_tiles_gpu[idx1].x;
        double* A2 = (idx2 >= 0) ? scheme->dense_tiles_gpu[idx2].x : nullptr;
        double* A3 = (idx3 >= 0) ? scheme->dense_tiles_gpu[idx3].x : nullptr;

        switch (rkind) {
            case 1:
                enqueue_potrf(ctx, stream, tempkn, A1, ldak);
                if (sync_and_read_devinfo(ctx, stream) != 0) {
                    std::fprintf(stderr, "sTiles GPU: matrix not PD at tile k=%d\n", k);
                    dep_tracker->abort_flag.store(true, std::memory_order_release);
                    return;
                }
                break;
            case 2:
                if (A1 && A2)
                    enqueue_syrk(ctx, stream, tempkn, tile_size,
                                 mzone, A1, ldan, zone, A2, ldak);
                break;
            case 3:
                if (A1 && A2)
                    enqueue_trsm(ctx, stream, tile_size, tempmn,
                                 zone, A2, ldak, A1, ldak);
                break;
            case 4:
                if (A1 && A2 && A3)
                    enqueue_gemm(ctx, stream, tile_size, tempmn, tile_size,
                                 mzone, A1, ldan, A2, ldan, zone, A3, ldak);
                break;
            default:
                break;
        }
    }

    STILES_CUDA_CHECK(cudaStreamSynchronize(stream));
}


// ---------------------------------------------------------------------------
//  3. Parallel pthreads dispatch (per-rank stream + events)
// ---------------------------------------------------------------------------
//  Cross-rank ordering: every ss_cond_set is preceded by cudaEventRecord,
//  every ss_cond_wait is followed by cudaStreamWaitEvent on the consumer
//  stream. SYRK and GEMM are leaf consumers (no ss_cond_set) so they don't
//  record events.
// ---------------------------------------------------------------------------
void gpu_dpotrf_expansion_dense_parallel(TiledMatrix*          scheme,
                                         stiles_context_t*     stile,
                                         GpuPersistentContext& ctx) {
    const int rank      = STILES_RANK;
    const int N         = scheme->dim;
    const int tile_size = scheme->tile_size;
    const int nt        = (N == 0) ? 0 : (N - 1) / tile_size + 1;
    const int full_tpd  = (tile_size > 0) ? (N / tile_size) : 0;

    auto BLK = [&](int k) { return (k < full_tpd) ? tile_size : (N % tile_size); };

    const auto& tasks   = sTiles::get_chol_tasks(scheme);
    const auto& offsets = sTiles::get_chol_task_offsets(scheme);
    const long long start = (rank < (int)offsets.size())     ? offsets[rank]     : 0;
    const long long end   = (rank + 1 < (int)offsets.size()) ? offsets[rank + 1] : (long long)tasks.size();

    cudaStream_t stream = ctx.streams[rank % ctx.num_streams];

    constexpr double zone  =  1.0;
    constexpr double mzone = -1.0;

    ss_init(nt, nt, 0);

    for (long long idx = start; idx < end; ++idx) {
        const std::array<int,7>& t = tasks[idx];
        const int rkind  = t[0];
        const int m      = t[1];
        const int k      = t[2];
        const int n      = t[3];
        const int idx1   = t[4];
        const int idx2   = t[5];
        const int idx3   = t[6];

        const int tempkn = (k == nt - 1) ? (N - k * tile_size) : tile_size;
        const int tempmn = (m == nt - 1) ? (N - m * tile_size) : tile_size;
        const int ldak   = BLK(k);
        const int ldan   = BLK(n);

        double* A1 = scheme->dense_tiles_gpu[idx1].x;
        double* A2 = (idx2 >= 0) ? scheme->dense_tiles_gpu[idx2].x : nullptr;
        double* A3 = (idx3 >= 0) ? scheme->dense_tiles_gpu[idx3].x : nullptr;

        switch (rkind) {
            case 1: { // DPOTRF: produces (k,k); consumers (TRSM cases) gate on event(k,k).
                enqueue_potrf(ctx, stream, tempkn, A1, ldak);
                // PD check requires a sync — bound to a host-visible flag.
                if (sync_and_read_devinfo(ctx, stream) != 0) {
                    std::fprintf(stderr, "sTiles GPU: matrix not PD at tile k=%d\n", k);
                    ss_abort();
                    break;
                }
                STILES_CUDA_CHECK(cudaEventRecord(event_for(ctx, k, k), stream));
                ss_cond_set(k, k, 1);
            } break;

            case 2: { // DSYRK: consumes (k,n) (produced by TRSM), writes (k,k).
                ss_cond_wait(k, n, 1);
                STILES_CUDA_CHECK(cudaStreamWaitEvent(stream, event_for(ctx, k, n), 0));
                if (A1 && A2)
                    enqueue_syrk(ctx, stream, tempkn, tile_size,
                                 mzone, A1, ldan, zone, A2, ldak);
            } break;

            case 3: { // DTRSM: consumes (k,k), writes (m,k).
                ss_cond_wait(k, k, 1);
                STILES_CUDA_CHECK(cudaStreamWaitEvent(stream, event_for(ctx, k, k), 0));
                if (A1 && A2)
                    enqueue_trsm(ctx, stream, tile_size, tempmn,
                                 zone, A2, ldak, A1, ldak);
                STILES_CUDA_CHECK(cudaEventRecord(event_for(ctx, m, k), stream));
                ss_cond_set(m, k, 1);
            } break;

            case 4: { // DGEMM: consumes (k,n) and (m,n), writes (k,m).
                ss_cond_wait(k, n, 1);
                STILES_CUDA_CHECK(cudaStreamWaitEvent(stream, event_for(ctx, k, n), 0));
                ss_cond_wait(m, n, 1);
                STILES_CUDA_CHECK(cudaStreamWaitEvent(stream, event_for(ctx, m, n), 0));
                if (A1 && A2 && A3)
                    enqueue_gemm(ctx, stream, tile_size, tempmn, tile_size,
                                 mzone, A1, ldan, A2, ldan, zone, A3, ldak);
            } break;

            default: break;
        }

        if (ss_aborted()) break;
    }

    STILES_CUDA_CHECK(cudaStreamSynchronize(stream));
    ss_finalize();
}


// ---------------------------------------------------------------------------
//  4. Parallel OMP dispatch (per-rank stream + events)
// ---------------------------------------------------------------------------
void gpu_dpotrf_expansion_dense_parallel_omp(TiledMatrix*          scheme,
                                             omp_dep_tracker_t*    dep_tracker,
                                             GpuPersistentContext& ctx,
                                             int                   worldsize) {
    const int rank      = omp_get_thread_num();
    (void)worldsize;

    const int N         = scheme->dim;
    const int tile_size = scheme->tile_size;
    const int nt        = (N == 0) ? 0 : (N - 1) / tile_size + 1;
    const int full_tpd  = (tile_size > 0) ? (N / tile_size) : 0;

    auto BLK = [&](int k) { return (k < full_tpd) ? tile_size : (N % tile_size); };

    const auto& tasks   = sTiles::get_chol_tasks(scheme);
    const auto& offsets = sTiles::get_chol_task_offsets(scheme);
    const long long start = (rank < (int)offsets.size())     ? offsets[rank]     : 0;
    const long long end   = (rank + 1 < (int)offsets.size()) ? offsets[rank + 1] : (long long)tasks.size();

    cudaStream_t stream = ctx.streams[rank % ctx.num_streams];

    constexpr double zone  =  1.0;
    constexpr double mzone = -1.0;

    dep_init(nt, nt, 0);

    for (long long idx = start; idx < end; ++idx) {
        const std::array<int,7>& t = tasks[idx];
        const int rkind  = t[0];
        const int m      = t[1];
        const int k      = t[2];
        const int n      = t[3];
        const int idx1   = t[4];
        const int idx2   = t[5];
        const int idx3   = t[6];

        const int tempkn = (k == nt - 1) ? (N - k * tile_size) : tile_size;
        const int tempmn = (m == nt - 1) ? (N - m * tile_size) : tile_size;
        const int ldak   = BLK(k);
        const int ldan   = BLK(n);

        double* A1 = scheme->dense_tiles_gpu[idx1].x;
        double* A2 = (idx2 >= 0) ? scheme->dense_tiles_gpu[idx2].x : nullptr;
        double* A3 = (idx3 >= 0) ? scheme->dense_tiles_gpu[idx3].x : nullptr;

        switch (rkind) {
            case 1: {
                enqueue_potrf(ctx, stream, tempkn, A1, ldak);
                if (sync_and_read_devinfo(ctx, stream) != 0) {
                    std::fprintf(stderr, "sTiles GPU: matrix not PD at tile k=%d\n", k);
                    dep_abort_all();
                    break;
                }
                STILES_CUDA_CHECK(cudaEventRecord(event_for(ctx, k, k), stream));
                dep_set_done(k, k, 1);
            } break;

            case 2: {
                dep_wait_for(k, n, 1);
                STILES_CUDA_CHECK(cudaStreamWaitEvent(stream, event_for(ctx, k, n), 0));
                if (A1 && A2)
                    enqueue_syrk(ctx, stream, tempkn, tile_size,
                                 mzone, A1, ldan, zone, A2, ldak);
            } break;

            case 3: {
                dep_wait_for(k, k, 1);
                STILES_CUDA_CHECK(cudaStreamWaitEvent(stream, event_for(ctx, k, k), 0));
                if (A1 && A2)
                    enqueue_trsm(ctx, stream, tile_size, tempmn,
                                 zone, A2, ldak, A1, ldak);
                STILES_CUDA_CHECK(cudaEventRecord(event_for(ctx, m, k), stream));
                dep_set_done(m, k, 1);
            } break;

            case 4: {
                dep_wait_for(k, n, 1);
                STILES_CUDA_CHECK(cudaStreamWaitEvent(stream, event_for(ctx, k, n), 0));
                dep_wait_for(m, n, 1);
                STILES_CUDA_CHECK(cudaStreamWaitEvent(stream, event_for(ctx, m, n), 0));
                if (A1 && A2 && A3)
                    enqueue_gemm(ctx, stream, tile_size, tempmn, tile_size,
                                 mzone, A1, ldan, A2, ldan, zone, A3, ldak);
            } break;

            default: break;
        }

        if (dep_is_aborted()) break;
    }

    STILES_CUDA_CHECK(cudaStreamSynchronize(stream));
    dep_finalize();
}

}}}  // namespace sTiles::gpu::dpotrf

#endif // STILES_GPU
