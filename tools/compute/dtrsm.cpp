
#include "../control/common.h"
#include "../control/stiles_control.hpp"  // for omp_dep_tracker_t and dep_* macros
#include "stiles_compute.hpp"             // for OMP function declarations
#include "../common/stiles_exporter.hpp"
#include "../common/stiles_logger.hpp"
#include "../common/stiles_expiry.hpp"
#include "../common/stiles_types.hpp"
#include "../common/stiles_config.hpp"
#include "../common/core_lapack.hpp"      // for core_dtrsm
#include "../ordering/ordering_utils.hpp"
// Sparse-path forward declarations — implementations live in sparse_dtrsm.cpp.
namespace sTiles {
sTiles::StatusCode pthreads_sparse_dtrsm(int global_index, TiledMatrix* scheme,
                                         double* B, int nrhs, int ldb);
sTiles::StatusCode omp_sparse_dtrsm     (int global_index, TiledMatrix* scheme,
                                         double* B, int nrhs, int ldb);
sTiles::StatusCode pthreads_sparse_dtrsm_forward (int global_index, TiledMatrix* scheme,
                                                  double* B, int nrhs, int ldb);
sTiles::StatusCode omp_sparse_dtrsm_forward      (int global_index, TiledMatrix* scheme,
                                                  double* B, int nrhs, int ldb);
sTiles::StatusCode pthreads_sparse_dtrsm_backward(int global_index, TiledMatrix* scheme,
                                                  double* B, int nrhs, int ldb);
sTiles::StatusCode omp_sparse_dtrsm_backward     (int global_index, TiledMatrix* scheme,
                                                  double* B, int nrhs, int ldb);
} // namespace sTiles
#ifdef STILES_GPU
    #include "../gpu/compute_gpu.hpp"
#endif
#include <vector>
#include <string>
#include <omp.h>

namespace sTiles {

    /**
    * @brief Performs a single parallel triangular solve operation (forward or backward).
    *
    * This is a core internal routine that executes a parallel triangular solve using
    * either forward substitution (for L*y=b) or backward substitution (for L^T*x=y).
    * It retrieves the thread-specific context and dispatches the call to the
    * appropriate low-level parallel function.
    *
    * @param[in] bind_index   The context identifier for the current thread or task.
    * @param[in] scheme       A pointer to the TiledMatrix structure containing the
    *                         factored matrix data and metadata.
    * @param[in,out] B        A pointer to the right-hand side (RHS) matrix. This
    *                         buffer will be overwritten with the solution.
    * @param[in] nrhs         The number of right-hand sides (i.e., columns in matrix B).
    * @param[in] solve_type   An integer specifying the type of solve:
    *                         - 0: Forward Substitution (L)
    *                         - 1: Backward Substitution (L^T)
    * @return An integer corresponding to a `sTiles::StatusCode`, indicating success or failure.
    */
    int solve_wrapper_internal(int bind_index, TiledMatrix *scheme, double *B, int nrhs, int solve_type) {

        stiles_context_t *stile = stiles_context_self(bind_index);
        if (stile == nullptr) {
            sTiles::Logger::fatal(sTiles::StatusCode::NotInitialized, "sTiles context is not available for this thread.");
            return static_cast<int>(sTiles::StatusCode::NotInitialized);
        }

        switch (solve_type) {
            case 0: // Forward Substitution
                sTiles::parallel_call(stile, stiles_pdtrsm_forward_dispatch, scheme, B, nrhs);
                break;
            case 1: // Backward Substitution
                sTiles::parallel_call(stile, stiles_pdtrsm_backward_dispatch, scheme, B, nrhs);
                break;
            case 2: // Forward & Backward Substitution
                sTiles::parallel_call(stile, stiles_pdtrsm_forward_dispatch, scheme, B, nrhs);
                sTiles::parallel_call(stile, stiles_pdtrsm_backward_dispatch, scheme, B, nrhs);
                break;

            default:
                sTiles::Logger::error("Invalid solve_type provided to internal wrapper. Received value: ", solve_type);
                return static_cast<int>(sTiles::StatusCode::IllegalValue);
        }

        return static_cast<int>(sTiles::StatusCode::Success);
    }

    /**
    * @brief Manages the complete solve process, including matrix permutations.
    *
    * This function serves as a high-level internal wrapper for the solve phase. It
    * handles the application of permutations to the right-hand side matrix if ordering
    * is enabled, calls the core internal solve routine(s), and permutes the solution
    * back to the original ordering. It supports forward, backward, and full solves.
    *
    * @param[in] bind_index   The context identifier for the current thread or task.
    * @param[in] scheme       A pointer to the TiledMatrix structure containing the
    *                         factored matrix.
    * @param[in,out] B        A pointer to the RHS matrix, which is overwritten by the solution.
    * @param[in] solve_type   The type of solve to perform:
    *                         - 0: Forward Substitution (L)
    *                         - 1: Backward Substitution (L^T)
    *                         - 2: Full Solve (L-L^T)
    * @param[in] nrhs         The number of right-hand sides.
    * @return An integer corresponding to a `sTiles::StatusCode`, indicating success or failure.
    */
    // Access user-configured parameters (param[8] selects OMP vs pthreads)
    extern "C" int* sTiles_get_params();

    // Per-thread cache: detects whether scheme->element_perm is the
    // identity permutation so wrapper_solve can short-circuit the
    // permute pipeline entirely. Keyed by (perm pointer, n); refreshed
    // when either changes (typically once per factorization per thread).
    struct PermCheck {
        const int* perm_ptr = nullptr;
        int        n        = -1;
        bool       identity = false;
    };

    static inline void refresh_perm_check(PermCheck& p, const int* perm, int n) {
        p.perm_ptr = perm;
        p.n        = n;
        p.identity = true;
        for (int i = 0; i < n; ++i) {
            if (perm[i] != i) { p.identity = false; break; }
        }
    }

    int wrapper_solve(int bind_index, TiledMatrix *scheme, double *B, int solve_type, int nrhs){

        // Guard: tile-based solve requires tile storage.
        if (!scheme || (!scheme->denseTiles && !scheme->chunkedDenseTiles)) {
            static bool warned_solve = false;
            if (!warned_solve) {
                warned_solve = true;
                sTiles::Logger::warning(
                    "wrapper_solve: no tile storage; call skipped.");
            }
            return static_cast<int>(sTiles::StatusCode::Unallocated);
        }

        // Per-thread reusable scratch — eliminates the per-call malloc.
        // Persists across solves on the same thread; resize-only-grows.
        static thread_local std::vector<double> tmp_B_vec;
        // Per-thread identity-perm cache (built once per factorization).
        static thread_local PermCheck _perm_chk;
        double* rhs_ptr = B;

        // Decide layout up-front: when the CSC fast path will fire AND a
        // permute is needed AND nrhs >= 2, fold the permute and the
        // column→row layout transform into one pass. Row-major X is
        // 22-30% faster than col-major in csc_dtrsm_multi at nrhs=8 on
        // bench data because each L-nonzero update lands in one cache
        // line per row instead of `nrhs` separate cache lines.
        // Only safe when we're sure the csc kernel runs (not GPU, not
        // tile path). Identity-perm shortcut (no permute) keeps col
        // layout — switching layouts there would require a transpose
        // pass equivalent in cost to the win.
        // Gate threshold = 8: bench data shows csc fused row saves tens of ms
        // per solve on graph-style matrices (apache1, ferris, INLA) up to and
        // beyond nrhs=30, while FEM matrices start losing real time (5-50 ms)
        // around nrhs=16-30. Capping at 8 captures the absolute-time wins on
        // graph workloads with no measurable downside on FEM (the ratio
        // losses on small bcsstk* matrices are sub-millisecond and below the
        // workload-relevant noise floor).
        // Auto cap by matrix shape. The discriminator (measured on group2
        // INLA matrices, single core, mode=semisparse) is nnz/row: matrices
        // with nnz/row < 20 keep winning with CSC up to nrhs=tile_size
        // (7 of 8 group2 matrices: nnz/row in [2.3, 15.2], csc 1.2–4.7× vs
        // tile at nrhs=40 == default tile_size). Above 20 (83o4NNNo at
        // 30.13; FEM bcsstk*/shipsec* in other groups) CSC regresses past
        // nrhs ~8–16, so keep the historical cap of 8. Reusing tile_size
        // ties one knob to both chol and solve — when the user picks a
        // tile size for chol they implicitly accept the matching solve cap.
        const long long _csc_nnz_A = scheme->original_nnz > 0
                                       ? scheme->original_nnz : scheme->nnz;
        const double    _csc_nnzpr = (_csc_nnz_A > 0 && scheme->dim > 0)
                                       ? double(_csc_nnz_A) / double(scheme->dim) : 0.0;
        const bool _csc_thin = (_csc_nnzpr > 0.0 && _csc_nnzpr < 20.0);
        const int csc_nrhs_cap = _csc_thin ? scheme->tile_size : 8;
        // nrhs==1 takes the serial CSC sweep even at multi-core: a single-RHS
        // triangular solve is a latency-bound recurrence, so the parallel tile
        // path can't speed it up — its per-tile fork/join overhead dwarfs the
        // tiny work and it actually regresses vs serial CSC (measured: sem_n*
        // solve1 0.0010s@1core vs 0.0013s@8core on the tile path). Serial CSC
        // wins at any core count, mirroring the sparse path's serial packed
        // solve. nrhs>=2 still requires a single core: there the tile path's
        // BLAS-3 parallelism beats the K-wide CSC kernel.
        const bool csc_path_eligible = scheme->packed && scheme->L_values
                                     && nrhs >= 1 && nrhs <= csc_nrhs_cap
                                     && (nrhs == 1 || scheme->num_cores == 1);
        if (std::getenv("STILES_DEBUG_ORDERING")) {
            std::fprintf(stderr, "[order-probe] semi/dense solve: use_ordering=%d nd_padding=%d "
                         "dim=%d nrhs=%d cores=%d csc_path=%d\n",
                         scheme->use_ordering, scheme->nd_padding, scheme->dim,
                         nrhs, scheme->num_cores, (int)csc_path_eligible);
        }
        bool use_row_layout = false;
#ifdef STILES_GPU
        // Conservative: when GPU is compiled in and may fire, keep
        // col-major so the GPU dispatch sees the layout it expects.
        // GPU users opting in have their own multi-RHS path.
        const bool gpu_active = scheme->use_gpu && scheme->dense_tiles_gpu
                              && scheme->factorization_variant == 0;
#else
        const bool gpu_active = false;
#endif

        if (scheme->use_ordering > 0) {
            const int user_dim = scheme->dim - scheme->nd_padding;
            const std::size_t total = static_cast<std::size_t>(scheme->dim) * nrhs;

            if (_perm_chk.perm_ptr != scheme->element_perm || _perm_chk.n != user_dim) {
                refresh_perm_check(_perm_chk, scheme->element_perm, user_dim);
            }

            // Identity-perm shortcut: only valid when we'd be using col-major
            // anyway (nrhs == 1, or csc gate misses, or GPU). Hand B directly
            // to the solve, no copy.
            //
            // Runtime layout pick (scheme->prefer_row_layout, set once below):
            //   true  -> tile-path multi-RHS kernels run row-major B;
            //            we must copy B into a row-major buffer here.
            //   false -> kernels run col-major B; no extra copy beyond the
            //            usual permute (if any).
            //
            // Heuristic: very thin matrices (fill < 2 AND nnz/row < 6 — the
            // INLA sem_* family) regress under row-major because the
            // banded-panel pack/unpack + cross-layout MKL gemm overhead
            // dwarfs the tiny per-tile work. Everything else wins ~5-13%.
            //
            // Layout pick is ONLY valid for the semisparse multi-RHS
            // kernels in pdtrsm.cpp — those are templated on RowMajorB
            // and runtime-dispatched. The dense kernels (variant 1/2 or
            // fallback paths in stiles_pdtrsm_*_dense*) call get_block_col
            // unconditionally and would read a row-major buffer as if it
            // were col-major. So the gate must also confirm the dispatch
            // will actually reach the semisparse path; otherwise pin to
            // col-major to keep the dense path correct.
            const int _ws_tile_type_mode = stiles_scheme_tile_mode(scheme);
            const bool semisparse_dispatch =
                (_ws_tile_type_mode == 1)
                && scheme->chunkedDenseTiles
                && scheme->semisparseTileMetaCore
                && scheme->factorization_variant != 1
                && scheme->factorization_variant != 2;
            if (!scheme->prefer_row_layout && scheme->dim > 0 && semisparse_dispatch) {
                const long long _nnz_A  = scheme->original_nnz > 0
                                            ? scheme->original_nnz
                                            : scheme->nnz;
                const long long _nnz_L  = scheme->nnz_factor;
                const double    _fill   = (_nnz_A > 0 && _nnz_L > 0)
                                            ? static_cast<double>(_nnz_L) / static_cast<double>(_nnz_A)
                                            : 0.0;
                const double    _nnzpr  = (_nnz_A > 0)
                                            ? static_cast<double>(_nnz_A) / static_cast<double>(scheme->dim)
                                            : 0.0;
                const bool _matrix_prefers_col = (_fill > 0.0 && _fill < 2.0)
                                              && (_nnzpr > 0.0 && _nnzpr < 6.0);
                scheme->prefer_row_layout = !_matrix_prefers_col;
            }
            const bool force_row_copy = (nrhs >= 2 && !gpu_active)
                                        && scheme->prefer_row_layout;
            const bool can_skip_permute = _perm_chk.identity && scheme->nd_padding == 0 && !force_row_copy;
            if (can_skip_permute && (nrhs == 1 || !csc_path_eligible || gpu_active)) {
                rhs_ptr = B;
                // use_row_layout stays false → col-major csc_multi (or tile path)
            } else {
                try {
                    if (tmp_B_vec.size() < total) tmp_B_vec.resize(total);
                } catch (const std::bad_alloc&) {
                    size_t required_bytes = total * sizeof(double);
                    sTiles::Logger::error("Failed to allocate temporary buffer for permutation. Required ",
                                          required_bytes, " bytes.");
                    return static_cast<int>(sTiles::StatusCode::OutOfResources);
                }
                // Decide layout for this permute. force_row_copy is true when
                // the runtime heuristic picked row-major for the tile path
                // multi-RHS kernels — the kernels read row-major B via the
                // _impl<true> instantiation, so we transpose here.
                use_row_layout = (csc_path_eligible && nrhs >= 2 && !gpu_active) || force_row_copy;
                if (use_row_layout) {
                    // Row-major dst: pad rows are at offsets [user_dim*nrhs,
                    // scheme->dim*nrhs). Zero them so pad RHS = 0 at solve.
                    if (scheme->nd_padding > 0) {
                        std::fill(tmp_B_vec.begin()
                                  + static_cast<std::size_t>(user_dim) * nrhs,
                                  tmp_B_vec.begin()
                                  + static_cast<std::size_t>(scheme->dim) * nrhs,
                                  0.0);
                    }
                    sTiles::to_ordering_padded_row(B, user_dim,
                                                   tmp_B_vec.data(), scheme->dim,
                                                   scheme->element_perm,
                                                   user_dim, nrhs,
                                                   scheme->num_cores);
                } else {
                    // Col-major dst (existing path).
                    if (scheme->nd_padding > 0) {
                        double* base = tmp_B_vec.data();
                        for (int j = 0; j < nrhs; ++j) {
                            double* col = base + static_cast<std::size_t>(j) * scheme->dim;
                            std::fill(col + user_dim, col + scheme->dim, 0.0);
                        }
                    }
                    sTiles::to_ordering_padded(B, user_dim,
                                               tmp_B_vec.data(), scheme->dim,
                                               scheme->element_perm,
                                               user_dim, nrhs,
                                               scheme->num_cores);
                }
                rhs_ptr = tmp_B_vec.data();
            }
        }

        // GPU solve path: single-stream, bypasses OMP/pthreads backends entirely
#ifdef STILES_GPU
        if (scheme->use_gpu && scheme->dense_tiles_gpu &&
            scheme->factorization_variant == 0)
        {
            const int tile_type_mode = stiles_scheme_tile_mode(scheme);
            const bool is_semisparse_gpu = (tile_type_mode == 1) &&
                                           scheme->chunkedDenseTiles &&
                                           scheme->semisparseTileMetaCore;

            // Check for pre-collected GPU solve tasks (fast path)
            const bool has_gpu_fwd = scheme->gpu_solve_fwd_tasks &&
                                     !scheme->gpu_solve_fwd_tasks->empty();
            const bool has_gpu_bwd = scheme->gpu_solve_bwd_tasks &&
                                     !scheme->gpu_solve_bwd_tasks->empty();

            bool gpu_solve_ok = false;

            if (!is_semisparse_gpu) {
                // Dense tiles (tile_type 0) — multi-stream if persistent context available
                auto* persistent = scheme->gpu_persistent_ctx
                    ? static_cast<sTiles::gpu::GpuPersistentContext*>(scheme->gpu_persistent_ctx)
                    : nullptr;
                const bool has_offsets = scheme->gpu_solve_fwd_offsets &&
                                         !scheme->gpu_solve_fwd_offsets->empty() &&
                                         scheme->gpu_solve_bwd_offsets &&
                                         !scheme->gpu_solve_bwd_offsets->empty();
                const bool use_multistream = persistent && persistent->num_streams > 1 &&
                                              has_gpu_fwd && has_gpu_bwd && has_offsets;

                if (use_multistream) {
                    // Multi-stream path with step-counter atomics
                    switch (solve_type) {
                        case 0:
                            sTiles::gpu::pdtrsm_forward_dense_gpu_dispatch(scheme, rhs_ptr, nrhs, *persistent);
                            break;
                        case 1:
                            sTiles::gpu::pdtrsm_backward_dense_gpu_dispatch(scheme, rhs_ptr, nrhs, *persistent);
                            break;
                        case 2:
                            sTiles::gpu::pdtrsm_forward_dense_gpu_dispatch(scheme, rhs_ptr, nrhs, *persistent);
                            sTiles::gpu::pdtrsm_backward_dense_gpu_dispatch(scheme, rhs_ptr, nrhs, *persistent);
                            break;
                    }
                } else {
                    // Single-stream path — fast path if tasks available, else slow path
                    switch (solve_type) {
                        case 0:
                            if (has_gpu_fwd) sTiles::gpu::pdtrsm_forward_dense_gpu_fast(scheme, rhs_ptr, nrhs);
                            else             sTiles::gpu::pdtrsm_forward_dense_gpu(scheme, rhs_ptr, nrhs);
                            break;
                        case 1:
                            if (has_gpu_bwd) sTiles::gpu::pdtrsm_backward_dense_gpu_fast(scheme, rhs_ptr, nrhs);
                            else             sTiles::gpu::pdtrsm_backward_dense_gpu(scheme, rhs_ptr, nrhs);
                            break;
                        case 2:
                            if (has_gpu_fwd) sTiles::gpu::pdtrsm_forward_dense_gpu_fast(scheme, rhs_ptr, nrhs);
                            else             sTiles::gpu::pdtrsm_forward_dense_gpu(scheme, rhs_ptr, nrhs);
                            if (has_gpu_bwd) sTiles::gpu::pdtrsm_backward_dense_gpu_fast(scheme, rhs_ptr, nrhs);
                            else             sTiles::gpu::pdtrsm_backward_dense_gpu(scheme, rhs_ptr, nrhs);
                            break;
                    }
                }
                gpu_solve_ok = true;
            } else if (scheme->gpu_persistent_ctx) {
                // Semisparse tiles (tile_type 1) — multi-stream if persistent context available
                auto* persistent_semi = static_cast<sTiles::gpu::GpuPersistentContext*>(scheme->gpu_persistent_ctx);
                const bool has_offsets_semi = scheme->gpu_solve_fwd_offsets &&
                                              !scheme->gpu_solve_fwd_offsets->empty() &&
                                              scheme->gpu_solve_bwd_offsets &&
                                              !scheme->gpu_solve_bwd_offsets->empty();
                const bool use_multistream_semi = persistent_semi && persistent_semi->num_streams > 1 &&
                                                   has_gpu_fwd && has_gpu_bwd && has_offsets_semi;

                if (use_multistream_semi) {
                    // Multi-stream path with step-counter atomics (same as dense)
                    switch (solve_type) {
                        case 0:
                            sTiles::gpu::pdtrsm_forward_semisparse_gpu_dispatch(scheme, rhs_ptr, nrhs, *persistent_semi);
                            break;
                        case 1:
                            sTiles::gpu::pdtrsm_backward_semisparse_gpu_dispatch(scheme, rhs_ptr, nrhs, *persistent_semi);
                            break;
                        case 2:
                            sTiles::gpu::pdtrsm_forward_semisparse_gpu_dispatch(scheme, rhs_ptr, nrhs, *persistent_semi);
                            sTiles::gpu::pdtrsm_backward_semisparse_gpu_dispatch(scheme, rhs_ptr, nrhs, *persistent_semi);
                            break;
                    }
                } else {
                    // Single-stream path — fast path if tasks available, else slow path
                    switch (solve_type) {
                        case 0:
                            if (has_gpu_fwd) sTiles::gpu::pdtrsm_forward_semisparse_gpu_fast(scheme, rhs_ptr, nrhs);
                            else             sTiles::gpu::pdtrsm_forward_semisparse_gpu(scheme, rhs_ptr, nrhs);
                            break;
                        case 1:
                            if (has_gpu_bwd) sTiles::gpu::pdtrsm_backward_semisparse_gpu_fast(scheme, rhs_ptr, nrhs);
                            else             sTiles::gpu::pdtrsm_backward_semisparse_gpu(scheme, rhs_ptr, nrhs);
                            break;
                        case 2:
                            if (has_gpu_fwd) sTiles::gpu::pdtrsm_forward_semisparse_gpu_fast(scheme, rhs_ptr, nrhs);
                            else             sTiles::gpu::pdtrsm_forward_semisparse_gpu(scheme, rhs_ptr, nrhs);
                            if (has_gpu_bwd) sTiles::gpu::pdtrsm_backward_semisparse_gpu_fast(scheme, rhs_ptr, nrhs);
                            else             sTiles::gpu::pdtrsm_backward_semisparse_gpu(scheme, rhs_ptr, nrhs);
                            break;
                    }
                }
                gpu_solve_ok = true;
            }

            if (gpu_solve_ok) {
                if (scheme->use_ordering > 0) {
                    const int user_dim = scheme->dim - scheme->nd_padding;
                    sTiles::from_ordering_padded(B, user_dim,
                                                 tmp_B_vec.data(), scheme->dim,
                                                 scheme->element_perm, user_dim, nrhs,
                                                 scheme->num_cores);
                }
                return static_cast<int>(sTiles::StatusCode::Success);
            }
        }
#endif

        // Fast CSC path. O(nnz(L)) scalar ops per RHS, no BLAS, no tile
        // overhead. Single specialisation:
        //   * nrhs == 1   → csc_dtrsm   (single-column kernel)
        //   * nrhs >= 2   → fall through to the tile path. Bench data on
        //                   group 1 (bcsstk* FEM) showed the tile path
        //                   beats csc_dtrsm_multi at nrhs ∈ [2,5] by
        //                   1.4–3× on most matrices once BLAS-3 in the
        //                   tile inner kernels amortises against the
        //                   hand-vectorised K-wide loop. Group 2 (INLA
        //                   graphs) still favours csc, but those callers
        //                   typically use nrhs == 1.
        // Required:
        //   * scheme->L_values   != null     (pack done in dpotrf for tile_type_mode == 1)
        //   * scheme->num_cores  == 1        (CSC kernel is single-threaded;
        //                                     multi-core configs deserve the
        //                                     parallel tile path so cores 2..N
        //                                     don't sit idle during the solve)
        //   * nrhs == 1
        static int* stiles_control_params = sTiles_get_params();
        // CSC fast-path opt-in is implicit: when sTiles_packing has been
        // called, scheme->L_values is non-null. Layout is row-major when
        // we permuted with use_row_layout above.
        //   nrhs == 1                → csc_dtrsm (col-major, single column)
        //   nrhs >= 2 + row layout   → csc_dtrsm_multi_row (fused permute path)
        //   nrhs >= 2 + col layout   → csc_dtrsm_multi (identity-perm or fallback)
        if (csc_path_eligible) {
            if (nrhs == 1) {
                sTiles::csc_dtrsm(scheme, rhs_ptr, solve_type);
            } else if (use_row_layout) {
                sTiles::csc_dtrsm_multi_row(scheme, rhs_ptr, nrhs,
                                            /*ldb_row=*/nrhs, solve_type);
            } else {
                sTiles::csc_dtrsm_multi(scheme, rhs_ptr, nrhs,
                                        scheme->dim, solve_type);
            }
            int status = static_cast<int>(sTiles::StatusCode::Success);
            // Back-permute / layout-transpose: must run whenever we worked on
            // a temp buffer instead of B directly (i.e., rhs_ptr != B). The
            // previous condition `!(identity && nd_padding == 0)` was wrong:
            // when the gate forced use_row_layout=true (nrhs >= 2, csc path)
            // even on a matrix with an identity perm and no padding, rhs_ptr
            // still points to tmp_B_vec (because we needed the col→row
            // transpose). Skipping the back-permute then left the user's B
            // unchanged from the input — residual catastrophically wrong.
            // Match the tile-path's condition (line ~545): rhs_ptr != B.
            if (scheme->use_ordering > 0 && rhs_ptr != B) {
                const int user_dim = scheme->dim - scheme->nd_padding;
                if (use_row_layout) {
                    sTiles::from_ordering_padded_row(B, user_dim,
                                                     tmp_B_vec.data(), scheme->dim,
                                                     scheme->element_perm,
                                                     user_dim, nrhs,
                                                     scheme->num_cores);
                } else {
                    sTiles::from_ordering_padded(B, user_dim,
                                                 tmp_B_vec.data(), scheme->dim,
                                                 scheme->element_perm,
                                                 user_dim, nrhs,
                                                 scheme->num_cores);
                }
            }
            return status;
        }

        // Select parallelization backend: OMP if param[8]==1, otherwise pthreads (default)
        int status;
        const bool use_omp = (stiles_control_params[sTiles::param::UseOMP] == 1);

        // Rescale + pthreads deadlock guard. The pthreads solve runs on the
        // persistent bound team, whose worker count is fixed at bind time
        // (stile->world_size). When a rescale schedule is active its per-rank
        // solve offsets / update-counter sync are sized for
        // rescale_schedule.num_cores; if the bound team has a DIFFERENT size the
        // reduce/signal loop waits on ranks that never match -> hang. This bites
        // the plain `turn_on_rescale` + solve_LLT/L/LT path (bound to the working
        // call's own team, sized at the factor's native cores). Route only that
        // mismatched case through the OMP path, which opens a fresh region sized
        // to rescale_schedule.num_cores. The explicit sTiles_solve_LLT_rescale is
        // UNCHANGED: it binds the rescale group's team, whose size already equals
        // rescale_schedule.num_cores, so it stays on the fast pthreads path (this
        // is the path R-INLA uses).
        bool reroute_rescale = false;
        if (!use_omp
            && scheme->use_rescale.load(std::memory_order_acquire) > 0
            && scheme->rescale_schedule.num_cores > 0) {
            stiles_context_t* bstile = stiles_context_self(bind_index);
            if (bstile && bstile->world_size != scheme->rescale_schedule.num_cores) {
                reroute_rescale = true;
            }
        }

        if (use_omp || reroute_rescale) {
            status = static_cast<int>(sTiles::omp_dtrsm(bind_index, scheme, rhs_ptr, nrhs, solve_type));
        } else {
            status = sTiles::solve_wrapper_internal(bind_index, scheme, rhs_ptr, nrhs, solve_type);
        }

        if (scheme->use_ordering > 0 && status == static_cast<int>(sTiles::StatusCode::Success)) {
            // Skip the back-permute when we used the identity-perm shortcut
            // above (rhs_ptr was B directly, so the solve already wrote to
            // user-side B).
            // The skip condition mirrors can_skip_permute above so the
            // force_row_copy case (runtime gate: tile path + nrhs>=2 +
            // scheme->prefer_row_layout) correctly enters this back-permute
            // and converts row->col into user B.
            if (rhs_ptr != B) {
                const int user_dim = scheme->dim - scheme->nd_padding;
                if (use_row_layout) {
                    sTiles::from_ordering_padded_row(B, user_dim,
                                                     tmp_B_vec.data(), scheme->dim,
                                                     scheme->element_perm, user_dim, nrhs,
                                                     scheme->num_cores);
                } else {
                    sTiles::from_ordering_padded(B, user_dim,
                                                 tmp_B_vec.data(), scheme->dim,
                                                 scheme->element_perm, user_dim, nrhs,
                                                 scheme->num_cores);
                }
            }
        }

        return status;
    }

    /**
    * @brief Acts as the primary internal dispatcher for a solve request.
    *
    * This function bridges the public C-API with the internal solve wrappers. It takes
    * group and call indices from the user, looks up the corresponding `TiledMatrix`
    * scheme and global context index within the `sTiles_object`, and then calls the
    * main `wrapper_solve` to perform the computation.
    *
    * @param[in] group_index  The index of the call group.
    * @param[in] call_index   The index of the call within the specified group.
    * @param[in] obj          A pointer to the main `sTiles_object` instance, which
    *                         contains all necessary schemes and metadata.
    * @param[in,out] B        A pointer to the RHS matrix.
    * @param[in] solve_type   The type of solve (0=Fwd, 1=Bwd, 2=Full).
    * @param[in] nrhs         The number of right-hand sides.
    * @return An integer corresponding to a `sTiles::StatusCode`, indicating success or failure.
    */
    int wrapper_solve_call(int group_index, int call_index, sTiles_object* obj, double *B, int solve_type, int nrhs) {

        if (obj == nullptr) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return static_cast<int>(sTiles::StatusCode::IllegalValue);
        }
        
        int global_index = obj->schemes[0]->call_lookup_table[group_index][call_index];
        int global_index_mapped = global_index;
        if (obj->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index > -1) {
            int new_call_index = obj->stiles_groups[group_index].stiles_calls[call_index].mapped_call_index;
            int new_group_index = obj->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index;
            global_index_mapped = obj->schemes[0]->call_lookup_table[new_group_index][new_call_index];
        }

        const double _t0 = omp_get_wtime();
        const int _rc = sTiles::wrapper_solve(global_index, obj->schemes[global_index_mapped], B, solve_type, nrhs);
        static const char* const _solve_names[] = { "Solve L", "Solve LT", "Solve LLT" };
        sTiles::Logger::timingf("\u2502   \u21aa %s (group %d, call %d, nrhs %d): %.6f s",
            (solve_type >= 0 && solve_type <= 2) ? _solve_names[solve_type] : "Solve",
            group_index, call_index, nrhs, omp_get_wtime() - _t0);
        return _rc;
    }

    int wrapper_solve_call_rescale(int group_index, int call_index, sTiles_object* obj, double *B, int solve_type, int nrhs, int group_index_rescale, int call_index_rescale) {

        if (obj == nullptr) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return static_cast<int>(sTiles::StatusCode::IllegalValue);
        }
        
        int global_index = obj->schemes[0]->call_lookup_table[group_index][call_index];
        int global_index_mapped = global_index;
        if (obj->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index > -1) {
            int new_call_index = obj->stiles_groups[group_index].stiles_calls[call_index].mapped_call_index;
            int new_group_index = obj->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index;
            global_index_mapped = obj->schemes[0]->call_lookup_table[new_group_index][new_call_index];
        }

        int global_index_rescale = obj->schemes[0]->call_lookup_table[group_index_rescale][call_index_rescale];

        return sTiles::wrapper_solve(global_index_rescale, obj->schemes[global_index_mapped], B, solve_type, nrhs);
    }


    /**
    * @brief OMP-based parallel triangular solve entry point.
    *
    * This function performs triangular solve using OpenMP parallelization instead of
    * pthreads. It creates an OMP parallel region and dispatches to the appropriate
    * OMP solve function based on the variant.
    *
    * @param[in] bind_index   The context identifier (not used in OMP version).
    * @param[in] scheme       A pointer to the TiledMatrix structure containing the
    *                         factored matrix data and metadata.
    * @param[in,out] B        A pointer to the right-hand side (RHS) matrix.
    * @param[in] nrhs         The number of right-hand sides.
    * @param[in] solve_type   An integer specifying the type of solve:
    *                         - 0: Forward Substitution (L^T)
    *                         - 1: Backward Substitution (L)
    *                         - 2: Both Forward and Backward
    * @return An integer corresponding to a `sTiles::StatusCode`.
    */
    StatusCode omp_dtrsm(int bind_index, TiledMatrix *scheme, double *B, int nrhs, int solve_type) {
        if (scheme == nullptr) {
            sTiles::Logger::fatal(sTiles::StatusCode::IllegalValue, "Null TiledMatrix in omp_dtrsm");
            return sTiles::StatusCode::IllegalValue;
        }

        const int variant = scheme->factorization_variant;
        int num_threads = (scheme->use_rescale.load(std::memory_order_acquire) > 0 && scheme->rescale_schedule.num_cores > 0)
                                ? scheme->rescale_schedule.num_cores
                                : (scheme->num_cores > 0 ? scheme->num_cores : omp_get_max_threads());

        // Serialize tiny SPARSE solves only. In tile-mode 2 the OpenMP region's exit
        // barrier makes idle threads busy-spin (e.g. 1.8 s on n=1074 under the default
        // active wait policy) — for small problems there is no parallelism to win, only
        // spin to pay. Below a tile-count threshold, run on ONE thread: the stride-based
        // worker (omp_pdtrsm_forward/backward) covers all tiles with worldsize=1, so it
        // stays correct and avoids the spin with no env var needed. (The precomputed-
        // offset "fast" workers are disabled below when serialized — their per-rank
        // offsets are sized for the full thread count and do NOT degrade to one thread.)
        // Gated on mode==2 ONLY: the semisparse/dense solves are already fast and their
        // workers are not verified to degrade to a single thread, so leave them untouched.
        {
            const int _mode = stiles_scheme_tile_mode(scheme);   // 0=dense 1=semisparse 2=sparse/nonunif
            const int _ts   = scheme->tile_size;
            const int _nrt  = (scheme->dim <= 0 || _ts <= 0) ? 0 : (scheme->dim - 1) / _ts + 1;
            constexpr int STILES_SOLVE_OMP_MIN_TILES = 64;   // tunable crossover
            if (_mode == 2 && _nrt < STILES_SOLVE_OMP_MIN_TILES) num_threads = 1;
        }

        // Check tile type mode for semisparse handling
        static int* stiles_control_params = sTiles_get_params();
        const int tile_type_mode = stiles_scheme_tile_mode(scheme);
        const bool is_semisparse = (tile_type_mode == 1) &&
                                   scheme->chunkedDenseTiles &&
                                   scheme->semisparseTileMetaCore;

        omp_dep_tracker_t dep_tracker;
        dep_tracker.progress_table = nullptr;
        dep_tracker.abort_flag.store(0, std::memory_order_relaxed);
        dep_tracker.workspace_offset = 0;
        {
            stiles_context_t *stile = stiles_context_self(bind_index);
            if (stile) {
                dep_tracker.workspace_offset = stile->workspace_offset;
            }
        }

        // Variant 1: Single dense tile - use direct LAPACK TRSM (no parallelism needed)
        if (variant == 1) {
            const int N = scheme->dim;
            double* tile = (scheme->denseTiles && scheme->denseTiles[0]) ? scheme->denseTiles[0] : nullptr;
            if (tile) {
                switch (solve_type) {
                    case 0: // Forward: L^T * X = B
                        sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                           sTiles::Op::Trans, sTiles::Diag::NonUnit,
                                           N, nrhs, 1.0, tile, N, B, N);
                        break;
                    case 1: // Backward: L * X = B
                        sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                           sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                           N, nrhs, 1.0, tile, N, B, N);
                        break;
                    case 2: // Both
                        sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                           sTiles::Op::Trans, sTiles::Diag::NonUnit,
                                           N, nrhs, 1.0, tile, N, B, N);
                        sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                           sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                           N, nrhs, 1.0, tile, N, B, N);
                        break;
                }
            }
            return sTiles::StatusCode::Success;
        }

        // Note: GPU solve dispatch is in wrapper_solve() (before OMP/pthreads selection)

        // Check if pre-collected tasks are available for fast track. The fast workers
        // index precomputed per-rank offsets sized for num_cores, so they are correct
        // exactly when the running thread count matches that partition, i.e.
        // offsets.size()-1 == num_threads. Each rank then owns its slice and, together,
        // all tasks are covered.
        //
        // The old guard was `num_threads > 1`, meant to exclude the mode-2 serialize
        // case (num_threads forced to 1 while offsets were sized for num_cores>1 — thread
        // 0 would run only its 1/num_cores slice). But it ALSO wrongly excluded the
        // legitimate single-core case (num_cores==1 → offsets sized for 1), forcing the
        // stride workers, whose multi-col-tile (nrhs > tile_size) path is buggy — the
        // cause of wrong OMP semisparse multi-RHS solves at 1 core. Gating on the exact
        // partition match fixes cores==1 and still excludes the serialize case.
        const bool has_fwd_tasks = !sTiles::get_solve_fwd_tasks(scheme).empty() &&
                                   static_cast<int>(sTiles::get_solve_fwd_offsets(scheme).size()) == num_threads + 1;
        const bool has_bwd_tasks = !sTiles::get_solve_bwd_tasks(scheme).empty() &&
                                   static_cast<int>(sTiles::get_solve_bwd_offsets(scheme).size()) == num_threads + 1;

        #pragma omp parallel num_threads(num_threads)
        {
            switch (solve_type) {
                case 0: // Forward Substitution (L^T * X = B)
                    if (variant == 2) {
                        sTiles::Process::omp_pdtrsm_forward_dense_full(scheme, B, nrhs, &dep_tracker, num_threads);
                    } else if (is_semisparse) {
                        if (has_fwd_tasks) {
                            sTiles::Process::omp_pdtrsm_forward_semisparse_fast(scheme, B, nrhs, &dep_tracker, num_threads);
                        } else {
                            sTiles::Process::omp_pdtrsm_forward_semisparse(scheme, B, nrhs, &dep_tracker, num_threads);
                        }
                    } else {
                        if (has_fwd_tasks) {
                            sTiles::Process::omp_pdtrsm_forward_fast(scheme, B, nrhs, &dep_tracker, num_threads);
                        } else {
                            sTiles::Process::omp_pdtrsm_forward(scheme, B, nrhs, &dep_tracker, num_threads);
                        }
                    }
                    break;
                case 1: // Backward Substitution (L * X = B)
                    if (variant == 2) {
                        sTiles::Process::omp_pdtrsm_backward_dense_full(scheme, B, nrhs, &dep_tracker, num_threads);
                    } else if (is_semisparse) {
                        if (has_bwd_tasks) {
                            sTiles::Process::omp_pdtrsm_backward_semisparse_fast(scheme, B, nrhs, &dep_tracker, num_threads);
                        } else {
                            sTiles::Process::omp_pdtrsm_backward_semisparse(scheme, B, nrhs, &dep_tracker, num_threads);
                        }
                    } else {
                        if (has_bwd_tasks) {
                            sTiles::Process::omp_pdtrsm_backward_fast(scheme, B, nrhs, &dep_tracker, num_threads);
                        } else {
                            sTiles::Process::omp_pdtrsm_backward(scheme, B, nrhs, &dep_tracker, num_threads);
                        }
                    }
                    break;
                case 2: // Both
                    if (variant == 2) {
                        sTiles::Process::omp_pdtrsm_forward_dense_full(scheme, B, nrhs, &dep_tracker, num_threads);
                        #pragma omp barrier
                        sTiles::Process::omp_pdtrsm_backward_dense_full(scheme, B, nrhs, &dep_tracker, num_threads);
                    } else if (is_semisparse) {
                        if (has_fwd_tasks) {
                            sTiles::Process::omp_pdtrsm_forward_semisparse_fast(scheme, B, nrhs, &dep_tracker, num_threads);
                        } else {
                            sTiles::Process::omp_pdtrsm_forward_semisparse(scheme, B, nrhs, &dep_tracker, num_threads);
                        }
                        #pragma omp barrier
                        if (has_bwd_tasks) {
                            sTiles::Process::omp_pdtrsm_backward_semisparse_fast(scheme, B, nrhs, &dep_tracker, num_threads);
                        } else {
                            sTiles::Process::omp_pdtrsm_backward_semisparse(scheme, B, nrhs, &dep_tracker, num_threads);
                        }
                    } else {
                        if (has_fwd_tasks) {
                            sTiles::Process::omp_pdtrsm_forward_fast(scheme, B, nrhs, &dep_tracker, num_threads);
                        } else {
                            sTiles::Process::omp_pdtrsm_forward(scheme, B, nrhs, &dep_tracker, num_threads);
                        }
                        #pragma omp barrier
                        if (has_bwd_tasks) {
                            sTiles::Process::omp_pdtrsm_backward_fast(scheme, B, nrhs, &dep_tracker, num_threads);
                        } else {
                            sTiles::Process::omp_pdtrsm_backward(scheme, B, nrhs, &dep_tracker, num_threads);
                        }
                    }
                    break;
            }
        }

        return sTiles::StatusCode::Success;
    }
}

// =============================================================================
//  C-API Functions
// =============================================================================
extern "C" int* sTiles_get_params();

extern "C" {

    /**
    * @brief [C-API] Solves the system AX=B using the L-L^T factorization.
    *
    * This is a public API function that performs a full triangular solve, which
    * consists of a forward substitution followed by a backward substitution. It is
    * the standard method for solving a system after a Cholesky factorization.
    *
    * @param[in] group_index  The index of the call group to use for this operation.
    * @param[in] call_index   The index of the specific call within the group.
    * @param[in] obj          An opaque handle (void**) to the `sTiles_object`.
    * @param[in,out] B        A pointer to the right-hand side matrix, which will be
    *                         overwritten with the solution matrix X.
    * @param[in] nrhs         The number of right-hand sides.
    * @return An integer corresponding to a `sTiles::StatusCode`.
    */
    int sTiles_solve_LLT(int group_index, int call_index, void** obj, double *B, int nrhs) {
        // Add robust null checks at the API boundary to prevent crashes.
        if (obj == nullptr || *obj == nullptr) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return static_cast<int>(sTiles::StatusCode::IllegalValue);
        }
        // Use safer C++ static_cast instead of C-style cast.
        sTiles_object* s = static_cast<sTiles_object*>(*obj);

        // Refuse a scheme whose preprocessing failed (see sTiles_chol guard).
        {
            const int _gi = s->schemes[0]->call_lookup_table[group_index][call_index];
            TiledMatrix* _sc = (_gi >= 0) ? s->schemes[_gi] : nullptr;
            if (!_sc || _sc->preprocess_failed) {
                sTiles::Logger::error("sTiles_solve_LLT: group ", group_index, " call ", call_index,
                    " was not successfully preprocessed; skipping solve.");
                return -1;
            }
        }

        // (Row-major B transpose is handled inside wrapper_solve via
        //  to_ordering_padded_row when scheme->prefer_row_layout is set.)

        // ── Sparse-module dispatch (tile_type_mode == 2) ─────────────────────
        // Routes through tools/compute/sparse_dtrsm.cpp using the same
        // pthreads/omp split as the tile path (param[8] = UseOMP).
        {
            const int gi = s->schemes[0]->call_lookup_table[group_index][call_index];
            TiledMatrix* sp_scheme = s->schemes[gi];
            if (sp_scheme && sp_scheme->sparse_handle) {
                int* params = sTiles_get_params();
                const int n = sp_scheme->dim - sp_scheme->nd_padding;
                sTiles::StatusCode sp_status =
                    (params && params[sTiles::param::UseOMP] == -1)
                        ? sTiles::pthreads_sparse_dtrsm(gi, sp_scheme, B, nrhs, n)
                        : sTiles::omp_sparse_dtrsm(gi, sp_scheme, B, nrhs, n);
                return (sp_status == sTiles::StatusCode::Success) ? 0 : -1;
            }
        }

        return sTiles::wrapper_solve_call(group_index, call_index, s, B, 2, nrhs);
    }

    int sTiles_solve_LLT_rescale(int group_index, int call_index, void** obj, double *B, int nrhs, int group_index_rescale, int call_index_rescale) {
        // Add robust null checks at the API boundary to prevent crashes.
        if (obj == nullptr || *obj == nullptr) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return static_cast<int>(sTiles::StatusCode::IllegalValue);
        }
        // Use safer C++ static_cast instead of C-style cast.
        sTiles_object* s = static_cast<sTiles_object*>(*obj);

        // Get the scheme
        int global_index = s->schemes[0]->call_lookup_table[group_index][call_index];
        int global_index_mapped = global_index;
        if (s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index > -1) {
            int new_call_index = s->stiles_groups[group_index].stiles_calls[call_index].mapped_call_index;
            int new_group_index = s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index;
            global_index_mapped = s->schemes[0]->call_lookup_table[new_group_index][new_call_index];
        }
        TiledMatrix* scheme = s->schemes[global_index_mapped];

        // ── Sparse-module path: rescale infra is tile-only. Route to the
        //    plain sparse solve and ignore the rescale slot.
        if (scheme && scheme->sparse_handle) {
            int* params = sTiles_get_params();
            const int n = scheme->dim - scheme->nd_padding;
            sTiles::StatusCode sp_status =
                (params && params[sTiles::param::UseOMP] == -1)
                    ? sTiles::pthreads_sparse_dtrsm(global_index_mapped, scheme, B, nrhs, n)
                    : sTiles::omp_sparse_dtrsm(global_index_mapped, scheme, B, nrhs, n);
            return (sp_status == sTiles::StatusCode::Success) ? 0 : -1;
        }

        // Set workspace offset for this rescale call slot
        int global_index_rescale = s->schemes[0]->call_lookup_table[group_index_rescale][call_index_rescale];
        stiles_context_t *stile_rescale = stiles_context_self(global_index_rescale);
        if (stile_rescale) {
            stile_rescale->workspace_offset = call_index_rescale * scheme->rescale_schedule.num_cores;
        }

        scheme->use_rescale.fetch_add(1, std::memory_order_acq_rel);

        int status = sTiles::wrapper_solve_call_rescale(group_index, call_index, s, B, 2, nrhs, group_index_rescale, call_index_rescale);

        scheme->use_rescale.fetch_sub(1, std::memory_order_acq_rel);

        return status;
    }

    /**
    * @brief [C-API] Solves the system LY=B (forward substitution).
    *
    * This is a public API function that performs only the forward substitution step
    * of a triangular solve. This may be useful for applications that require
    * intermediate results from the solve process.
    *
    * @param[in] group_index  The index of the call group to use for this operation.
    * @param[in] call_index   The index of the specific call within the group.
    * @param[in] obj          An opaque handle (void**) to the `sTiles_object`.
    * @param[in,out] B        A pointer to the right-hand side matrix, which will be
    *                         overwritten with the intermediate solution matrix Y.
    * @param[in] nrhs         The number of right-hand sides.
    * @return An integer corresponding to a `sTiles::StatusCode`.
    */
    int sTiles_solve_L(int group_index, int call_index, void** obj, double *B, int nrhs) {
        // Add robust null checks at the API boundary to prevent crashes.
        if (obj == nullptr || *obj == nullptr) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return static_cast<int>(sTiles::StatusCode::IllegalValue);
        }
        // Use safer C++ static_cast instead of C-style cast.
        sTiles_object* s = static_cast<sTiles_object*>(*obj);

        // ── Sparse-module dispatch (tile_type_mode == 2) ─────────────────────
        {
            const int gi = s->schemes[0]->call_lookup_table[group_index][call_index];
            TiledMatrix* sp_scheme = s->schemes[gi];
            if (sp_scheme && sp_scheme->sparse_handle) {
                int* params = sTiles_get_params();
                const int n = sp_scheme->dim - sp_scheme->nd_padding;
                sTiles::StatusCode sp_status =
                    (params && params[sTiles::param::UseOMP] == -1)
                        ? sTiles::pthreads_sparse_dtrsm_forward(gi, sp_scheme, B, nrhs, n)
                        : sTiles::omp_sparse_dtrsm_forward(gi, sp_scheme, B, nrhs, n);
                return (sp_status == sTiles::StatusCode::Success) ? 0 : -1;
            }
        }

        return sTiles::wrapper_solve_call(group_index, call_index, s, B, 0, nrhs);
    }

    int sTiles_solve_L_rescale(int group_index, int call_index, void** obj, double *B, int nrhs, int group_index_rescale, int call_index_rescale) {
        // Add robust null checks at the API boundary to prevent crashes.
        if (obj == nullptr || *obj == nullptr) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return static_cast<int>(sTiles::StatusCode::IllegalValue);
        }
        // Use safer C++ static_cast instead of C-style cast.
        sTiles_object* s = static_cast<sTiles_object*>(*obj);

        // Get the scheme
        int global_index = s->schemes[0]->call_lookup_table[group_index][call_index];
        int global_index_mapped = global_index;
        if (s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index > -1) {
            int new_call_index = s->stiles_groups[group_index].stiles_calls[call_index].mapped_call_index;
            int new_group_index = s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index;
            global_index_mapped = s->schemes[0]->call_lookup_table[new_group_index][new_call_index];
        }
        TiledMatrix* scheme = s->schemes[global_index_mapped];

        // ── Sparse-module path: rescale infra is tile-only. Route to the
        //    plain sparse forward solve and ignore the rescale slot.
        if (scheme && scheme->sparse_handle) {
            int* params = sTiles_get_params();
            const int n = scheme->dim - scheme->nd_padding;
            sTiles::StatusCode sp_status =
                (params && params[sTiles::param::UseOMP] == -1)
                    ? sTiles::pthreads_sparse_dtrsm_forward(global_index_mapped, scheme, B, nrhs, n)
                    : sTiles::omp_sparse_dtrsm_forward(global_index_mapped, scheme, B, nrhs, n);
            return (sp_status == sTiles::StatusCode::Success) ? 0 : -1;
        }

        // Set workspace offset for this rescale call slot
        int global_index_rescale = s->schemes[0]->call_lookup_table[group_index_rescale][call_index_rescale];
        stiles_context_t *stile_rescale = stiles_context_self(global_index_rescale);
        if (stile_rescale) {
            stile_rescale->workspace_offset = call_index_rescale * scheme->rescale_schedule.num_cores;
        }

        scheme->use_rescale.fetch_add(1, std::memory_order_acq_rel);

        int status = sTiles::wrapper_solve_call_rescale(group_index, call_index, s, B, 0, nrhs, group_index_rescale, call_index_rescale);

        scheme->use_rescale.fetch_sub(1, std::memory_order_acq_rel);

        return status;
    }

    /**
    * @brief [C-API] Solves the system L^T*X=Y (backward substitution).
    *
    * This is a public API function that performs only the backward substitution step
    * of a triangular solve. This is typically used after a forward substitution to
    * obtain the final solution.
    *
    * @param[in] group_index  The index of the call group to use for this operation.
    * @param[in] call_index   The index of the specific call within the group.
    * @param[in] obj          An opaque handle (void**) to the `sTiles_object`.
    * @param[in,out] B        A pointer to the intermediate solution matrix Y, which
    *                         will be overwritten with the final solution matrix X.
    * @param[in] nrhs         The number of right-hand sides.
    * @return An integer corresponding to a `sTiles::StatusCode`.
    */
    int sTiles_solve_LT(int group_index, int call_index, void** obj, double *B, int nrhs) {
        // Add robust null checks at the API boundary to prevent crashes.
        if (obj == nullptr || *obj == nullptr) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return static_cast<int>(sTiles::StatusCode::IllegalValue);
        }
        // Use safer C++ static_cast instead of C-style cast.
        sTiles_object* s = static_cast<sTiles_object*>(*obj);

        // ── Sparse-module dispatch (tile_type_mode == 2) ─────────────────────
        {
            const int gi = s->schemes[0]->call_lookup_table[group_index][call_index];
            TiledMatrix* sp_scheme = s->schemes[gi];
            if (sp_scheme && sp_scheme->sparse_handle) {
                int* params = sTiles_get_params();
                const int n = sp_scheme->dim - sp_scheme->nd_padding;
                sTiles::StatusCode sp_status =
                    (params && params[sTiles::param::UseOMP] == -1)
                        ? sTiles::pthreads_sparse_dtrsm_backward(gi, sp_scheme, B, nrhs, n)
                        : sTiles::omp_sparse_dtrsm_backward(gi, sp_scheme, B, nrhs, n);
                return (sp_status == sTiles::StatusCode::Success) ? 0 : -1;
            }
        }

        return sTiles::wrapper_solve_call(group_index, call_index, s, B, 1, nrhs);
    }

    int sTiles_solve_LT_rescale(int group_index, int call_index, void** obj, double *B, int nrhs, int group_index_rescale, int call_index_rescale) {
        // Add robust null checks at the API boundary to prevent crashes.
        if (obj == nullptr || *obj == nullptr) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return static_cast<int>(sTiles::StatusCode::IllegalValue);
        }
        // Use safer C++ static_cast instead of C-style cast.
        sTiles_object* s = static_cast<sTiles_object*>(*obj);

        // Get the scheme
        int global_index = s->schemes[0]->call_lookup_table[group_index][call_index];
        int global_index_mapped = global_index;
        if (s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index > -1) {
            int new_call_index = s->stiles_groups[group_index].stiles_calls[call_index].mapped_call_index;
            int new_group_index = s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index;
            global_index_mapped = s->schemes[0]->call_lookup_table[new_group_index][new_call_index];
        }
        TiledMatrix* scheme = s->schemes[global_index_mapped];

        // ── Sparse-module path: rescale infra is tile-only. Route to the
        //    plain sparse backward solve and ignore the rescale slot.
        if (scheme && scheme->sparse_handle) {
            int* params = sTiles_get_params();
            const int n = scheme->dim - scheme->nd_padding;
            sTiles::StatusCode sp_status =
                (params && params[sTiles::param::UseOMP] == -1)
                    ? sTiles::pthreads_sparse_dtrsm_backward(global_index_mapped, scheme, B, nrhs, n)
                    : sTiles::omp_sparse_dtrsm_backward(global_index_mapped, scheme, B, nrhs, n);
            return (sp_status == sTiles::StatusCode::Success) ? 0 : -1;
        }

        // Set workspace offset for this rescale call slot
        int workspace_offset = 0;
        int global_index_rescale = s->schemes[0]->call_lookup_table[group_index_rescale][call_index_rescale];
        stiles_context_t *stile_rescale = stiles_context_self(global_index_rescale);
        if (stile_rescale) {
            stile_rescale->workspace_offset = call_index_rescale * scheme->rescale_schedule.num_cores;
        }

        scheme->use_rescale.fetch_add(1, std::memory_order_acq_rel);

        int status = sTiles::wrapper_solve_call_rescale(group_index, call_index, s, B, 1, nrhs, group_index_rescale, call_index_rescale);

        scheme->use_rescale.fetch_sub(1, std::memory_order_acq_rel);

        return status;
    }

}
