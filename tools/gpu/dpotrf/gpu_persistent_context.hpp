/**
 * @file    gpu_persistent_context.hpp
 * @brief   Per-call GPU context: cuSOLVER/cuBLAS handles, streams, events, workspace.
 *
 * One context per sTiles call. Created at preprocess time and reused across
 * chol -> trsm -> potri without re-creating handles, streams, events, or the
 * cusolver workspace.
 *
 * Sandbox: not yet wired into the build. Drop-in replacement for the per-call
 * handle/workspace churn in tools/compute/compute_gpu.hpp.
 */
#ifndef STILES_GPU_DPOTRF_PERSISTENT_CONTEXT_HPP
#define STILES_GPU_DPOTRF_PERSISTENT_CONTEXT_HPP

#ifdef STILES_GPU

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace sTiles { namespace gpu { namespace dpotrf {

#define STILES_CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
        std::fprintf(stderr, "[gpu_dpotrf] CUDA error %s:%d: %s\n",           \
                     __FILE__, __LINE__, cudaGetErrorString(_e));             \
        std::abort();                                                         \
    }                                                                         \
} while (0)

#define STILES_CUSOLVER_CHECK(call) do {                                      \
    cusolverStatus_t _s = (call);                                             \
    if (_s != CUSOLVER_STATUS_SUCCESS) {                                      \
        std::fprintf(stderr, "[gpu_dpotrf] cuSOLVER error %s:%d: %d\n",       \
                     __FILE__, __LINE__, (int)_s);                            \
        std::abort();                                                         \
    }                                                                         \
} while (0)

#define STILES_CUBLAS_CHECK(call) do {                                        \
    cublasStatus_t _s = (call);                                               \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                        \
        std::fprintf(stderr, "[gpu_dpotrf] cuBLAS error %s:%d: %d\n",         \
                     __FILE__, __LINE__, (int)_s);                            \
        std::abort();                                                         \
    }                                                                         \
} while (0)


struct GpuPersistentContext {
    int                          gpu_id        = -1;
    int                          tile_size     = 0;     // nb
    int                          num_tiles_dim = 0;     // nt
    int                          num_streams   = 1;

    cusolverDnHandle_t           cusolverH     = nullptr;
    cublasHandle_t               cublasH       = nullptr;

    std::vector<cudaStream_t>    streams;

    // Reusable POTRF workspace. Sized for the full tile_size; trailing
    // boundary tiles need <= this many doubles, so one buffer is enough.
    double*                      d_workspace   = nullptr;
    int                          workspace_len = 0;
    int*                         d_devinfo     = nullptr;

    // 2D event grid: events[m * nt + k] records completion of the kernel
    // that produces tile (m, k). Only POTRF (k,k) and TRSM (m,k) record;
    // SYRK/GEMM are leaf consumers and never produce a tile that another
    // rank consumes through a flag.
    std::vector<cudaEvent_t>     events;

    bool                         initialized   = false;

    // Set this from a kernel that decides to abort (e.g., devInfo != 0).
    // Caller may bridge to stile->ss_abort / dep_tracker->abort_flag.
    int                          abort_flag    = 0;
};


inline void init_persistent_context(GpuPersistentContext& ctx,
                                    int gpu_id,
                                    int tile_size,
                                    int num_tiles_dim,
                                    int num_streams) {
    if (ctx.initialized) return;

    STILES_CUDA_CHECK(cudaSetDevice(gpu_id));

    ctx.gpu_id        = gpu_id;
    ctx.tile_size     = tile_size;
    ctx.num_tiles_dim = num_tiles_dim;
    ctx.num_streams   = (num_streams < 1) ? 1 : num_streams;

    STILES_CUSOLVER_CHECK(cusolverDnCreate(&ctx.cusolverH));
    STILES_CUBLAS_CHECK(cublasCreate(&ctx.cublasH));
    STILES_CUBLAS_CHECK(cublasSetPointerMode(ctx.cublasH, CUBLAS_POINTER_MODE_HOST));

    ctx.streams.resize(ctx.num_streams);
    for (int i = 0; i < ctx.num_streams; ++i) {
        STILES_CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.streams[i], cudaStreamNonBlocking));
    }
    // cuSOLVER/cuBLAS handles are rebound per kernel call to the rank's stream.
    STILES_CUSOLVER_CHECK(cusolverDnSetStream(ctx.cusolverH, ctx.streams[0]));
    STILES_CUBLAS_CHECK(cublasSetStream(ctx.cublasH, ctx.streams[0]));

    // Size the POTRF workspace at the full tile_size. Pass a dummy device
    // pointer; only the leading dimension matters for the buffer-size query.
    double dummy_lda = 0.0;
    double* dummy_ptr = reinterpret_cast<double*>(&dummy_lda); // not dereferenced
    STILES_CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        ctx.cusolverH, CUBLAS_FILL_MODE_UPPER,
        ctx.tile_size, dummy_ptr, ctx.tile_size, &ctx.workspace_len));

    STILES_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.d_workspace),
                                 static_cast<size_t>(ctx.workspace_len) * sizeof(double)));
    STILES_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.d_devinfo), sizeof(int)));

    const size_t nev = static_cast<size_t>(num_tiles_dim) * num_tiles_dim;
    ctx.events.resize(nev);
    for (size_t i = 0; i < nev; ++i) {
        STILES_CUDA_CHECK(cudaEventCreateWithFlags(&ctx.events[i], cudaEventDisableTiming));
    }

    ctx.initialized = true;
}


inline void destroy_persistent_context(GpuPersistentContext& ctx) {
    if (!ctx.initialized) return;

    cudaSetDevice(ctx.gpu_id);

    for (cudaEvent_t& e : ctx.events) {
        if (e) cudaEventDestroy(e);
    }
    ctx.events.clear();

    if (ctx.d_workspace) { cudaFree(ctx.d_workspace); ctx.d_workspace = nullptr; }
    if (ctx.d_devinfo)   { cudaFree(ctx.d_devinfo);   ctx.d_devinfo   = nullptr; }

    for (cudaStream_t& s : ctx.streams) {
        if (s) cudaStreamDestroy(s);
    }
    ctx.streams.clear();

    if (ctx.cublasH)   { cublasDestroy(ctx.cublasH);     ctx.cublasH   = nullptr; }
    if (ctx.cusolverH) { cusolverDnDestroy(ctx.cusolverH); ctx.cusolverH = nullptr; }

    ctx.initialized = false;
}


// Helper: addressable event for tile (m, k).
inline cudaEvent_t event_for(GpuPersistentContext& ctx, int m, int k) {
    return ctx.events[static_cast<size_t>(m) * ctx.num_tiles_dim + k];
}

}}}  // namespace sTiles::gpu::dpotrf

#endif // STILES_GPU
#endif // STILES_GPU_DPOTRF_PERSISTENT_CONTEXT_HPP
