/**
 * @file    pdtrtri.cpp
 * @brief   Parallel (threaded) DTRTRI path; forward from legacy PLASMA to sTiles wrappers.
 *
 * @project sTiles (Sparse Tiles Library)
 * @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @version 3.0.0
 * @date 1 1 2026
 * @license Proprietary
 *
 * @note This file includes code adapted from PLASMA; licensing of third-party
 *       components remains subject to their original terms. sTiles wrappers and
 *       integration code are proprietary as stated below.
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
 */
#include "../control/common.h"
#include "../control/stiles_control.hpp"  // for omp_dep_tracker_t and dep_* macros
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <unordered_set>
#include "../common/core_lapack.hpp"
#include "stiles_xsmm.hpp"
#include "stiles_verify.hpp"
#include "../tile/core_kernels.hpp"
#include "../tile/meta.hpp"

extern "C" int* sTiles_get_params();

//#define STILES_INTERNAL_DEBUGGING
//#define STILES_EXPORT_DAG

// Old GPU code - disabled
// #ifdef STILES_GPU
//     #include "compute_gpu.hpp"
// #endif

#ifdef STILES_INTERNAL_DEBUGGING
    #include "debugging.hpp"
#endif

#ifdef STILES_EXPORT_DAG
    #include "dags.hpp"
#endif

#include <iomanip>

void mirroring(int n, double *A, double *B, int lda) {

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            B[j * lda + i] = A[i * lda + j];
        }
    }
}

// Per-column truncated solves for the diagonal L^{-1} (below). DEFAULT ON;
// set STILES_SEMI_DIAGINV_TRUNC=0 to restore the exact MKL dtbtrs kernel
// (needed only for bit-reproducing a selected inverse against pre-2026-07-05
// builds). Not bit-identical to dtbtrs: diffs at machine epsilon in the
// selected inverse, L/logdet untouched.
//
// MEASURED (2026-07-05): this is the selinv win. ~1.38x on sem_n5000 and
// ~1.29x on sem_n20000 at 1 core (the semisparse GMRF / INLA regime); modest
// on lgm_bw2 (~1.06x); flat on spacetime/bern (phase-2 GEMM bound). Selected
// inverse agrees with the dtbtrs baseline to ~1e-15 rel (backward stable),
// all check_chol_selinv_paths cells PASS. The parallel selinv is already
// non-bit-deterministic run-to-run at ~1e-15 (GEMM reduction order), so the
// truncation loosens no guarantee that was tight.
static inline bool semi_diaginv_trunc_enabled()
{
    static const int on = [] {
        const char* e = std::getenv("STILES_SEMI_DIAGINV_TRUNC");
        return (e && e[0] == '0') ? 0 : 1;   // default ON; explicit "0" disables
    }();
    return on == 1;
}

// Diagonal-block inverse for a banded upper-triangular factor: solve U X = I
// with U n x n, bandwidth kd in LAPACK band storage (ldab = kd+1), X pre-set
// to the identity (dense n x n, column-major). Column j of U^{-1} is
// supported on rows 0..j only, so the gated path solves just the leading
// (j+1)-system per column, skipping the zero tail dtbtrs scans over.
static inline void banded_diag_inverse(int n, int kd, const double* fact, double* inv)
{
    const int ldab = kd + 1;
    if (semi_diaginv_trunc_enabled()) {
        for (int j = 0; j < n; ++j)
            cblas_dtbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        j + 1, kd, fact, ldab, inv + static_cast<std::size_t>(j) * n, 1);
    } else {
        LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N',
                       n, kd, n, fact, ldab, inv, n);
    }
}

// Returns (1-based) index of the first exact zero on the diagonal of a
// semisparse case-1 factor tile, or 0 if the tile is nonsingular. kd==0 stores
// the diagonal contiguously (fact[i]); kd>0 uses LAPACK band storage so U[j,j]
// lives at fact[kd + j*(kd+1)]. Mirrors LAPACKE_dtrtri's exact-zero singularity
// test so the semisparse selinv surfaces a singular tile as ExecutionFailed the
// way chol and the dense selinv path do, instead of silently emitting Inf/NaN.
static inline int semi_diag_singular(int n, int kd, const double* fact)
{
    if (kd == 0) {
        for (int i = 0; i < n; ++i)
            if (fact[i] == 0.0) return i + 1;
    } else {
        const int ldab = kd + 1;
        for (int j = 0; j < n; ++j)
            if (fact[kd + static_cast<std::size_t>(j) * ldab] == 0.0) return j + 1;
    }
    return 0;
}

// STILES_SEMI_TRMM_SOLVE=1: in phase 1, form W = U_ii^{-1} * L_offdiag by a
// banded solve against the factor (~2*m*kd*sa flops) instead of the dense
// triangular multiply by the precomputed U_ii^{-1} (~m^2*sa flops). Not
// bit-identical to the TRMM (a solve rounds differently); off by default.
//
// MEASURED SLOWER (2026-07-05): despite the lower flop count, MKL's dense
// dtrmm against the small triangular tile beats the banded dtbtrs solve
// per-flop on sem_n5000/sem_n20000 (selinv ~0.5-5% slower at both core
// counts). The flop argument does not survive the banded-kernel overhead.
// Kept gated for the record; leave OFF. The selinv win lives in
// STILES_SEMI_DIAGINV_TRUNC (case 1), not here.
static inline bool semi_trmm_solve_enabled()
{
    static const int on = [] {
        const char* e = std::getenv("STILES_SEMI_TRMM_SOLVE");
        return (e && e[0] == '1') ? 1 : 0;
    }();
    return on == 1;
}

// Gated case-2 kernel swap. Returns true if W = U_ii^{-1} * B was formed by
// the banded solve; false leaves B untouched so the caller falls back to the
// TRMM. Only fires when the band is narrow enough for the solve to win.
static inline bool semi_case2_banded_solve(const TiledMatrix* tm, int diag_idx,
                                           int m, int sa, double* B)
{
    if (!semi_trmm_solve_enabled()) return false;
    const int kd = tm->semisparseTileMetaCore[diag_idx].upper_bw;
    if (kd <= 0 || 2 * kd >= m) return false;
    const double* fact_diag = tm->chunkedDenseTiles[diag_idx];
    if (!fact_diag) return false;
    LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', m, kd, sa,
                   fact_diag, kd + 1, B, m);
    return true;
}

// ============================================================================
// Banded Format Helper Functions for Selinv Optimization

namespace sTiles{ 
    
    namespace Process{
        
        //
        //full Dense
        //
        // Forward declarations for helpers defined later in this file.
        inline int variant2_tile_index_inv(int i, int j, int num_tiles) {
            // Formula from build_dense_tile_lookup_variant2 (sparse_dense_tiling.hpp line 2594)
            return i * num_tiles - (i * (i - 1)) / 2 + (j - i);
        }

        static void omp_dpotri_variant2(TiledMatrix *tiledMatrix, omp_dep_tracker_t* dep_tracker, int worldsize)
        {
            const int rank = omp_get_thread_num();
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const double zone  = 1.0;
            const double mzone = -1.0;
            const int nt = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            if (!tiledMatrix->denseTiles || !tiledMatrix->inverseTiles) {
                #pragma omp barrier
                return;
            }

            // Helper to get tile dimension
            auto tile_dim = [&](int idx) -> int {
                return (idx == nt - 1) ? (N - idx * tile_size) : tile_size;
            };

            // Helper to get dense tile index (upper triangular storage)
            auto dense_idx = [&](int i, int j) -> int {
                return variant2_tile_index_inv(i, j, nt);
            };

            // Tile ownership: tile at dense_idx(i,j) is owned by rank (dense_idx(i,j) % worldsize)
            auto owns_tile = [&](int ti, int tj) -> bool {
                return (dense_idx(ti, tj) % worldsize) == rank;
            };

            // Initialize dependency tracker
            dep_init(nt, nt, 0);

            // ============ Region 1: Compute L^{-1} stored in inverseTiles ============
            // type 1: TRSM - inv(i,i) = L(i,i)^{-1} * I
            // type 2: TRMM - L(i,j) = inv(i,i) * L(i,j)  [stores L^{-1}(i,j) in factor tiles]

            int i = nt - 1 - rank;
            while (i >= 0) {
                const int d = dense_idx(i, i);
                const int tempii = tile_dim(i);

                double* L_ii = tiledMatrix->denseTiles[d];
                double* inv_ii = tiledMatrix->inverseTiles[d];

                // type 1: TRSM - compute diagonal inverse
                if (L_ii && inv_ii) {
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                    sTiles::Uplo::Upper,
                                    sTiles::Op::NoTrans,
                                    sTiles::Diag::NonUnit,
                                    tempii, tempii, 1.0,
                                    L_ii, tempii,
                                    inv_ii, tempii);
                }

                // type 2: TRMM for all j > i (apply inverse to off-diagonal tiles)
                for (int j = i + 1; j < nt; ++j) {
                    const int ij = dense_idx(i, j);
                    const int tempjj = tile_dim(j);

                    double* L_ij = tiledMatrix->denseTiles[ij];

                    if (inv_ii && L_ij) {
                        sTiles::core_dtrmm(sTiles::Side::Left,
                                        sTiles::Uplo::Upper,
                                        sTiles::Op::NoTrans,
                                        sTiles::Diag::NonUnit,
                                        tempii, tempjj,
                                        zone,
                                        inv_ii, tempii,
                                        L_ij, tempii);
                    }
                }

                i -= worldsize;  // Move to next row assigned to this thread
            }

            // type 3: Barrier between phases
            #pragma omp barrier

            // ============ Region 2: Compute A^{-1} = L^{-T} * L^{-1} ============
            // All threads iterate over ALL rows, but each thread only processes tiles it owns.
            // After Region 1, denseTiles contain L^{-1}

            for (i = nt - 1; i >= 0; --i) {
                for (int j = nt - 1; j >= i; --j) {
                    const int tempii = tile_dim(i);
                    const int tempjj = tile_dim(j);

                    if (i == j) {
                        // Diagonal tile (i, i)
                        const int d = dense_idx(i, i);

                        // Check if this rank owns the diagonal tile
                        if (!owns_tile(i, i)) continue;

                        double* inv_ii = tiledMatrix->inverseTiles[d];

                        // type 4: LAUUM on diagonal - compute L^{-T}(i,i) * L^{-1}(i,i)
                        if (inv_ii) {
                            sTiles::core_dlauum(sTiles::Uplo::Upper, tempii, inv_ii, tempii);
                            mirroring(tempii, inv_ii, inv_ii, tempii);
                        }

                        // type 5: GEMM - diagonal update from off-diagonal contributions
                        for (int k = i + 1; k < nt; ++k) {
                            // Wait for tile (i, k) to be complete
                            dep_wait_for(i, k, 2);

                            const int ik = dense_idx(i, k);
                            const int tempkk = tile_dim(k);

                            double* Linv_ik = tiledMatrix->denseTiles[ik];  // L^{-1}(i,k)
                            double* inv_ik = tiledMatrix->inverseTiles[ik];

                            if (Linv_ik && inv_ik && inv_ii) {
                                sTiles::core_dgemm(sTiles::Op::NoTrans,
                                                sTiles::Op::Trans,
                                                tempii, tempii, tempkk,
                                                mzone,
                                                Linv_ik, tempii,
                                                inv_ik, tempii,
                                                zone,
                                                inv_ii, tempii);
                            }
                        }

                        // type 6: Signal diagonal (i, i) is complete
                        dep_set_done(i, i, 2);

                    } else {
                        // Off-diagonal tile (i, j) where i < j
                        const int ij = dense_idx(i, j);

                        // Check if this rank owns the off-diagonal tile
                        if (!owns_tile(i, j)) continue;

                        double* inv_ij = tiledMatrix->inverseTiles[ij];

                        // Process contributions from all k in (i+1, ..., nt-1)
                        for (int k = i + 1; k < nt; ++k) {
                            const int tempkk = tile_dim(k);

                            if (k > j) {
                                // type 7: inv(i,j) += -L^{-1}(i,k) * inv(j,k)^T
                                dep_wait_for(j, k, 2);

                                const int ik = dense_idx(i, k);
                                const int jk = dense_idx(j, k);

                                double* Linv_ik = tiledMatrix->denseTiles[ik];
                                double* inv_jk = tiledMatrix->inverseTiles[jk];

                                if (Linv_ik && inv_jk && inv_ij) {
                                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                                    sTiles::Op::Trans,
                                                    tempii, tempjj, tempkk,
                                                    mzone,
                                                    Linv_ik, tempii,
                                                    inv_jk, tempjj,
                                                    zone,
                                                    inv_ij, tempii);
                                }
                            } else {
                                // k <= j (i.e., i < k <= j)
                                // type 8: inv(i,j) += -L^{-1}(i,k) * inv(k,j)
                                dep_wait_for(k, j, 2);

                                const int ik = dense_idx(i, k);
                                const int kj = dense_idx(k, j);

                                double* Linv_ik = tiledMatrix->denseTiles[ik];
                                double* inv_kj = tiledMatrix->inverseTiles[kj];

                                if (Linv_ik && inv_kj && inv_ij) {
                                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                                    sTiles::Op::NoTrans,
                                                    tempii, tempjj, tempkk,
                                                    mzone,
                                                    Linv_ik, tempii,
                                                    inv_kj, tempkk,
                                                    zone,
                                                    inv_ij, tempii);
                                }
                            }
                        }

                        // type 9: Signal tile (i, j) is complete
                        dep_set_done(i, j, 2);
                    }
                }
            }

            dep_finalize();
        }

        static void pthreads_dpotri_variant2(TiledMatrix *tiledMatrix, stiles_context_t *stile)
        {
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const double zone  = 1.0;
            const double mzone = -1.0;
            const int nt = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            // Helper to get tile dimension
            auto tile_dim = [&](int idx) -> int {
                return (idx == nt - 1) ? (N - idx * tile_size) : tile_size;
            };

            // Helper to get dense tile index
            auto dense_idx = [&](int i, int j) -> int {
                return variant2_tile_index_inv(i, j, nt);
            };

            // Debug prints (commented out for production)
            // if (STILES_RANK == 0) {
            //     std::cout << "\n========== pthreads_dpotri_variant2 DEBUG ==========\n";
            //     std::cout << "N=" << N << " tile_size=" << tile_size << " nt=" << nt << "\n";
            //     std::cout << "STILES_SIZE=" << STILES_SIZE << "\n";
            //     std::cout << "Tile index mapping:\n";
            //     for (int i = 0; i < nt; ++i) {
            //         for (int j = i; j < nt; ++j) {
            //             std::cout << "  (" << i << "," << j << ") -> idx=" << dense_idx(i, j) << "\n";
            //         }
            //     }
            //     std::cout << "\n--- Factor tiles (denseTiles) before selinv ---\n";
            //     for (int i = 0; i < nt; ++i) {
            //         for (int j = i; j < nt; ++j) {
            //             const int idx = dense_idx(i, j);
            //             double* tile = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[idx] : nullptr;
            //             const int h = tile_dim(i);
            //             const int w = tile_dim(j);
            //             debug_print_tile(("FACTOR(" + std::to_string(i) + "," + std::to_string(j) + ")").c_str(),
            //                              idx, tile, h, w, 0);
            //         }
            //     }
            // }

            // ============ Initialize inverse tiles (only rank 0 does this) ============
            // Zero ALL inverse tiles and set identity on diagonals
            // This ensures TRSM has proper RHS (identity matrix)
            if (STILES_RANK == 0) {
                for (int i = 0; i < nt; ++i) {
                    for (int j = i; j < nt; ++j) {
                        const int idx = dense_idx(i, j);
                        if (idx < 0 || idx >= tiledMatrix->numActiveTiles) continue;

                        double* inv = tiledMatrix->inverseTiles[idx];
                        if (!inv) continue;

                        const int h = tile_dim(i);
                        const int w = tile_dim(j);

                        // Zero the tile
                        std::memset(inv, 0, static_cast<size_t>(h) * w * sizeof(double));

                        // Set identity on diagonals
                        if (i == j) {
                            for (int ii = 0; ii < h; ++ii) {
                                inv[ii + ii * h] = 1.0;
                            }
                        }
                    }
                }

                // std::cout << "\n--- Inverse tiles after initialization ---\n";
                // for (int i = 0; i < nt; ++i) {
                //     for (int j = i; j < nt; ++j) {
                //         const int idx = dense_idx(i, j);
                //         double* tile = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[idx] : nullptr;
                //         const int h = tile_dim(i);
                //         const int w = tile_dim(j);
                //         debug_print_tile(("INV_INIT(" + std::to_string(i) + "," + std::to_string(j) + ")").c_str(),
                //                          idx, tile, h, w, 0);
                //     }
                // }
            }

            // Barrier to ensure initialization is complete before computation
            sTiles::Control::Barrier(stile);

            in_init(nt, nt, 0);

            // ============ Region 1: Compute L^{-1} stored in inverseTiles ============
            // Following PLASMA parallel pattern: each thread processes rows in cyclic manner
            // type 1: TRSM - inv(i,i) = L(i,i)^{-1} * I
            // type 2: TRMM - L(i,j) = inv(i,i) * L(i,j)  [stores L^{-1}(i,j) in factor tiles]

            // if (STILES_RANK == 0) {
            //     std::cout << "\n========== REGION 1: Compute L^{-1} ==========\n";
            // }

            int i = nt - 1 - STILES_RANK;
            while (i >= 0) {
                const int d = dense_idx(i, i);
                const int tempii = tile_dim(i);

                double* L_ii = tiledMatrix->denseTiles[d];
                double* inv_ii = tiledMatrix->inverseTiles[d];

                // if (STILES_RANK == 0) {
                //     std::cout << "\n[R1] Processing row i=" << i << " (diag idx=" << d << ", dim=" << tempii << ")\n";
                //     debug_print_tile("L_ii before TRSM", d, L_ii, tempii, tempii, 0);
                //     debug_print_tile("inv_ii before TRSM", d, inv_ii, tempii, tempii, 0);
                // }

                // type 1: TRSM - compute diagonal inverse
                if (L_ii && inv_ii) {
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                    sTiles::Uplo::Upper,
                                    sTiles::Op::NoTrans,
                                    sTiles::Diag::NonUnit,
                                    tempii, tempii, 1.0,
                                    L_ii, tempii,
                                    inv_ii, tempii);
                }

                // if (STILES_RANK == 0) {
                //     debug_print_tile("inv_ii after TRSM", d, inv_ii, tempii, tempii, 0);
                // }

                // type 2: TRMM for all j > i (apply inverse to off-diagonal tiles)
                for (int j = i + 1; j < nt; ++j) {
                    const int ij = dense_idx(i, j);
                    const int tempjj = tile_dim(j);

                    double* L_ij = tiledMatrix->denseTiles[ij];

                    // if (STILES_RANK == 0) {
                    //     std::cout << "[R1] TRMM: L(" << i << "," << j << ") = inv_ii * L(" << i << "," << j << ")\n";
                    //     debug_print_tile("L_ij before TRMM", ij, L_ij, tempii, tempjj, 0);
                    // }

                    if (inv_ii && L_ij) {
                        sTiles::core_dtrmm(sTiles::Side::Left,
                                        sTiles::Uplo::Upper,
                                        sTiles::Op::NoTrans,
                                        sTiles::Diag::NonUnit,
                                        tempii, tempjj,
                                        zone,
                                        inv_ii, tempii,
                                        L_ij, tempii);
                    }

                    // if (STILES_RANK == 0) {
                    //     debug_print_tile("L_ij after TRMM", ij, L_ij, tempii, tempjj, 0);
                    // }
                }

                i -= STILES_SIZE;  // Move to next row assigned to this thread
            }

            // type 3: Barrier between phases
            sTiles::Control::Barrier(stile);

            // if (STILES_RANK == 0) {
            //     std::cout << "\n--- State after Region 1 ---\n";
            //     for (int ti = 0; ti < nt; ++ti) {
            //         for (int tj = ti; tj < nt; ++tj) {
            //             const int idx = dense_idx(ti, tj);
            //             const int h = tile_dim(ti);
            //             const int w = tile_dim(tj);
            //             debug_print_tile(("denseTiles(" + std::to_string(ti) + "," + std::to_string(tj) + ")").c_str(),
            //                              idx, tiledMatrix->denseTiles[idx], h, w, 0);
            //             debug_print_tile(("inverseTiles(" + std::to_string(ti) + "," + std::to_string(tj) + ")").c_str(),
            //                              idx, tiledMatrix->inverseTiles[idx], h, w, 0);
            //         }
            //     }
            // }

            // ============ Region 2: Compute A^{-1} = L^{-T} * L^{-1} ============
            // IMPORTANT: ALL threads iterate over ALL rows (i--), but each thread
            // only processes tiles it owns. Ownership is per-TILE, not per-row.
            // This matches the original inv_expansion_phase1_scaled pattern.
            // Note: After Region 1, denseTiles contain L^{-1}

            // Tile ownership: tile at dense_idx(i,j) is owned by rank (dense_idx(i,j) % STILES_SIZE)
            auto owns_tile = [&](int ti, int tj) -> bool {
                return (dense_idx(ti, tj) % STILES_SIZE) == STILES_RANK;
            };

            // Processing order: iterate ALL rows (i from nt-1 down to 0),
            // then ALL columns in that row (j from nt-1 down to i).
            // Each thread only processes tiles it owns, but ALL threads iterate the same order.
            for (i = nt - 1; i >= 0; --i) {
                for (int j = nt - 1; j >= i; --j) {
                    const int tempii = tile_dim(i);
                    const int tempjj = tile_dim(j);

                    if (i == j) {
                        // Diagonal tile (i, i)
                        const int d = dense_idx(i, i);

                        // Check if this rank owns the diagonal tile
                        if (!owns_tile(i, i)) continue;

                        double* inv_ii = tiledMatrix->inverseTiles[d];

                        // type 4: LAUUM on diagonal - compute L^{-T}(i,i) * L^{-1}(i,i)
                        if (inv_ii) {
                            sTiles::core_dlauum(sTiles::Uplo::Upper, tempii, inv_ii, tempii);
                            mirroring(tempii, inv_ii, inv_ii, tempii);
                        }

                        // type 5: GEMM - diagonal update from off-diagonal contributions
                        // inv(i,i) += -L^{-1}(i,k) * inv(i,k)^T for all k > i
                        for (int k = i + 1; k < nt; ++k) {
                            // Wait for tile (i, k) to be complete - signaled by whoever owns (i,k)
                            in_cond_wait(i, k, 2);

                            const int ik = dense_idx(i, k);
                            const int tempkk = tile_dim(k);

                            double* Linv_ik = tiledMatrix->denseTiles[ik];  // L^{-1}(i,k)
                            double* inv_ik = tiledMatrix->inverseTiles[ik];

                            if (Linv_ik && inv_ik && inv_ii) {
                                sTiles::core_dgemm(sTiles::Op::NoTrans,
                                                sTiles::Op::Trans,
                                                tempii, tempii, tempkk,
                                                mzone,
                                                Linv_ik, tempii,
                                                inv_ik, tempii,
                                                zone,
                                                inv_ii, tempii);
                            }
                        }

                        // type 6: Signal diagonal (i, i) is complete
                        in_cond_set(i, i, 2);

                    } else {
                        // Off-diagonal tile (i, j) where i < j
                        const int ij = dense_idx(i, j);

                        // Check if this rank owns the off-diagonal tile
                        if (!owns_tile(i, j)) continue;

                        double* inv_ij = tiledMatrix->inverseTiles[ij];

                        // Process contributions from all k in (i+1, ..., nt-1)
                        for (int k = i + 1; k < nt; ++k) {
                            const int tempkk = tile_dim(k);

                            if (k > j) {
                                // type 7: inv(i,j) += -L^{-1}(i,k) * inv(j,k)^T
                                // Wait for inv(j,k) which is in row j (already processed since j > i)
                                in_cond_wait(j, k, 2);

                                const int ik = dense_idx(i, k);
                                const int jk = dense_idx(j, k);

                                double* Linv_ik = tiledMatrix->denseTiles[ik];
                                double* inv_jk = tiledMatrix->inverseTiles[jk];

                                if (Linv_ik && inv_jk && inv_ij) {
                                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                                    sTiles::Op::Trans,
                                                    tempii, tempjj, tempkk,
                                                    mzone,
                                                    Linv_ik, tempii,
                                                    inv_jk, tempjj,
                                                    zone,
                                                    inv_ij, tempii);
                                }
                            } else {
                                // k <= j (i.e., i < k <= j)
                                // type 8: inv(i,j) += -L^{-1}(i,k) * inv(k,j)
                                // Wait for inv(k,j) which is in row k (already processed since k > i)
                                in_cond_wait(k, j, 2);

                                const int ik = dense_idx(i, k);
                                const int kj = dense_idx(k, j);

                                double* Linv_ik = tiledMatrix->denseTiles[ik];
                                double* inv_kj = tiledMatrix->inverseTiles[kj];

                                if (Linv_ik && inv_kj && inv_ij) {
                                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                                    sTiles::Op::NoTrans,
                                                    tempii, tempjj, tempkk,
                                                    mzone,
                                                    Linv_ik, tempii,
                                                    inv_kj, tempkk,
                                                    zone,
                                                    inv_ij, tempii);
                                }
                            }
                        }

                        // type 9: Signal tile (i, j) is complete
                        in_cond_set(i, j, 2);
                    }
                }
            }

            in_finalize();
        }

        static void omp_dpotri_variant2_e_trick(TiledMatrix *tiledMatrix, omp_dep_tracker_t* dep_tracker, int worldsize)
        {
            const int rank = omp_get_thread_num();
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const double zone  = 1.0;
            const double mzone = -1.0;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            // Check if e_trick_inv arrays are available - if not, use direct computation
            if (!tiledMatrix->e_trick_inv || !tiledMatrix->e_trick_size_inv) {
                omp_dpotri_variant2(tiledMatrix, dep_tracker, worldsize);
                return;
            }

            const int num_tasks = tiledMatrix->e_trick_size_inv[rank];

            auto tile_dims = [&](int dense_idx, int& h, int& w) {
                if (tiledMatrix->tileMetaCore && dense_idx >= 0) {
                    const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[dense_idx];
                    h = (meta.height > 0) ? meta.height : tile_size;
                    w = (meta.width  > 0) ? meta.width  : tile_size;
                } else {
                    h = tile_size;
                    w = tile_size;
                }
            };

            dep_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            int global_in = 0;

            // Phase 1: until routine 3 (barrier)
            for (int in = 0; in < num_tasks; ++in) {
                int myroutine = tiledMatrix->e_trick_inv[rank][0 + 7*in];
                int i = tiledMatrix->e_trick_inv[rank][1 + 7*in];
                int j = tiledMatrix->e_trick_inv[rank][2 + 7*in];
                int k = tiledMatrix->e_trick_inv[rank][3 + 7*in];
                int index1 = tiledMatrix->e_trick_inv[rank][4 + 7*in];
                int index2 = tiledMatrix->e_trick_inv[rank][5 + 7*in];
                int index3 = tiledMatrix->e_trick_inv[rank][6 + 7*in];

                switch (myroutine) {
                    case 1: // TRSM
                    {
                        int h = tile_size, w = tile_size;
                        tile_dims(index1, h, w);
                        double* fact = tiledMatrix->denseTiles[index1];
                        double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (fact && inv) {
                            sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                            sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                            h, w, 1.0, fact, h, inv, h);
                        }
                        break;
                    }
                    case 2: // TRMM
                    {
                        int inv_h = tile_size, inv_w = tile_size;
                        tile_dims(index1, inv_h, inv_w);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index2, mh, mw);
                        double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        double* fact = tiledMatrix->denseTiles[index2];
                        if (inv && fact) {
                            sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                            sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                            mh, mw, zone, inv, inv_h, fact, mh);
                        }
                        break;
                    }
                    case 3: // Barrier
                    {
                        #pragma omp barrier
                        global_in = in + 1;
                        goto exit_phase1_etrick_omp;
                    }
                }
            }

        exit_phase1_etrick_omp:

            // Phase 2: remaining tasks with dependency tracking
            for (int in = global_in; in < num_tasks; ++in) {
                int myroutine = tiledMatrix->e_trick_inv[rank][0 + 7*in];
                int i = tiledMatrix->e_trick_inv[rank][1 + 7*in];
                int j = tiledMatrix->e_trick_inv[rank][2 + 7*in];
                int k = tiledMatrix->e_trick_inv[rank][3 + 7*in];
                int index1 = tiledMatrix->e_trick_inv[rank][4 + 7*in];
                int index2 = tiledMatrix->e_trick_inv[rank][5 + 7*in];
                int index3 = tiledMatrix->e_trick_inv[rank][6 + 7*in];

                switch (myroutine) {
                    case 4: // LAUUM + mirror
                    {
                        int h = tile_size, w = tile_size;
                        tile_dims(index1, h, w);
                        double* inv = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (inv) {
                            sTiles::core_dlauum(sTiles::Uplo::Upper, w, inv, h);
                            mirroring(h, inv, inv, h);
                        }
                        break;
                    }
                    case 5: // GEMM diagonal update
                    {
                        dep_wait_for(i, k, 2);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index2, mh, mw);
                        int inv1_h = tile_size, inv1_w = tile_size;
                        tile_dims(index1, inv1_h, inv1_w);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        double* fact = tiledMatrix->denseTiles[index2];
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv1 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (fact && inv2 && inv1) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                            mh, mh, mw, mzone, fact, mh, inv2, inv2_h, zone, inv1, inv1_h);
                        }
                        break;
                    }
                    case 6:
                        dep_set_done(i, i, 2);
                        break;
                    case 7: // GEMM off-diagonal (Trans)
                    {
                        dep_wait_for(j, k, 2);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index1, mh, mw);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        int inv3_h = tile_size, inv3_w = tile_size;
                        tile_dims(index3, inv3_h, inv3_w);
                        double* fact = tiledMatrix->denseTiles[index1];
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                        if (fact && inv2 && inv3) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                            mh, inv2_h, mw, -1.0, fact, mh, inv2, inv2_h, zone, inv3, inv3_h);
                        }
                        break;
                    }
                    case 8: // GEMM off-diagonal (NoTrans)
                    {
                        dep_wait_for(k, j, 2);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index1, mh, mw);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        int inv3_h = tile_size, inv3_w = tile_size;
                        tile_dims(index3, inv3_h, inv3_w);
                        double* fact = tiledMatrix->denseTiles[index1];
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                        if (fact && inv2 && inv3) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                            mh, inv2_w, mw, mzone, fact, mh, inv2, inv2_h, zone, inv3, inv3_h);
                        }
                        break;
                    }
                    case 9:
                        dep_set_done(i, j, 2);
                        break;
                }
            }

            dep_finalize();
        }

        // Shared variant-1 (single tile) worker for the full inverse.
        // Loads the factor U into inv's upper triangle (expanding LAPACK banded
        // storage on the fly), inverts in place with LAPACKE_dtrtri (n^3/3 flops
        // vs ~n^3 for the previous DTRSM-against-identity), then DLAUUM forms
        // U^{-1} U^{-T} and mirroring symmetrizes the result.
        // Returns 0 on success; nonzero on missing buffers or singular U.
        static int dpotri_full_dense_single_tile(TiledMatrix *tiledMatrix, const char *who) {
            const int N = tiledMatrix->dim;

            // Try denseTiles first, then fall back to chunkedDenseTiles (semisparse mode).
            double* L = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[0] : nullptr;
            double* inv = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[0] : nullptr;
            bool L_is_banded = false;
            int kd = 0;  // bandwidth

            if (!L && tiledMatrix->chunkedDenseTiles) {
                L = tiledMatrix->chunkedDenseTiles[0];
                // Factor may be stored in LAPACK banded format.
                if (tiledMatrix->semisparseTileMetaCore) {
                    kd = tiledMatrix->semisparseTileMetaCore[0].upper_bw;
                    L_is_banded = (kd > 0);
                }
            }
            if (!inv && tiledMatrix->chunkedInverseTiles) inv = tiledMatrix->chunkedInverseTiles[0];

            if (!L || !inv) {
                std::fprintf(stderr, "sTiles error: %s (variant 1) - L or inv is null! L=%s inv=%s\n",
                             who, L ? "valid" : "NULL", inv ? "valid" : "NULL");
                return -1;
            }

            if (L_is_banded && kd > 0) {
                // Expand banded upper triangular ((kd+1) x N, col-major) directly into inv:
                // A(i,j) = AB[kd + i - j + j*ldab] for max(0, j-kd) <= i <= j.
                std::memset(inv, 0, static_cast<size_t>(N) * N * sizeof(double));
                const int ldab = kd + 1;
                for (int j = 0; j < N; ++j)
                    for (int i = std::max(0, j - kd); i <= j; ++i)
                        inv[i + static_cast<size_t>(j) * N] = L[kd + i - j + j * ldab];
            } else {
                // Copy U's upper triangle into inv (the lower half is rewritten by mirroring).
                for (int j = 0; j < N; ++j)
                    std::memcpy(inv + static_cast<size_t>(j) * N,
                                L   + static_cast<size_t>(j) * N,
                                static_cast<size_t>(j + 1) * sizeof(double));
            }

            const lapack_int info = LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', N, inv, N);
            if (info != 0) {
                std::fprintf(stderr, "sTiles error: %s LAPACKE_dtrtri failed with info=%d.\n",
                             who, static_cast<int>(info));
                return static_cast<int>(info);
            }
            sTiles::core_dlauum(sTiles::Uplo::Upper, N, inv, N);
            mirroring(N, inv, inv, N);
            return 0;
        }

        static void pthreads_dpotri_full_dense(TiledMatrix *tiledMatrix, stiles_context_t *stile, int variant) {

            // Variant 1: single tile - only rank 0 does the work.
            if (variant == 1) {
                if (STILES_RANK == 0) {
                    if (dpotri_full_dense_single_tile(tiledMatrix, "pthreads_dpotri_full_dense") != 0) {
                        stile->ss_abort = 1;   // surfaced as ExecutionFailed by pthreads_dpotri
                    }
                }
                return;
            }

            // Variant 2: full triangular tiled dense - on-the-fly (i,j,k) walk.
            pthreads_dpotri_variant2(tiledMatrix, stile);
        }

        static void omp_dpotri_full_dense(TiledMatrix *tiledMatrix, omp_dep_tracker_t* dep_tracker, int variant, int worldsize)
        {
            if (variant == 1 || (tiledMatrix->numActiveTiles == 1 && tiledMatrix->dimTiledMatrix == 1)) {
                // Variant 1: single tile - only thread 0 does the work.
                if (omp_get_thread_num() == 0) {
                    if (dpotri_full_dense_single_tile(tiledMatrix, "omp_dpotri_full_dense") != 0) {
                        dep_tracker->abort_flag.store(true, std::memory_order_release);  // -> ExecutionFailed
                    }
                }
                #pragma omp barrier
            } else if (variant == 2) {
                // Variant 2: full triangular tiled dense - dep_* walk (the e_trick list is
                // never populated in fastmode, so this resolves to omp_dpotri_variant2).
                omp_dpotri_variant2_e_trick(tiledMatrix, dep_tracker, worldsize);
            }
        }


        ////////////////////////////////////////////////////////////////////////////////

        //------------------> Sparse of Dense Tiles      -     (Tile mode 0)
        
        ////////////////////////////////////////////////////////////////////////////////

        static void pthreads_dpotri_dense_serial(TiledMatrix *tiledMatrix, stiles_context_t *stile)
        {
            const int tile_size = tiledMatrix->tile_size;
            const double zone  = 1.0;
            const double mzone = -1.0;

            const auto &tasks = sTiles::get_inv_tasks(tiledMatrix);
            const size_t ntasks = tasks.size();

            auto tile_dims = [&](int dense_idx, int& h, int& w) {
                if (tiledMatrix->tileMetaCore && dense_idx >= 0) {
                    const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[dense_idx];
                    h = (meta.height > 0) ? meta.height : tile_size;
                    w = (meta.width  > 0) ? meta.width  : tile_size;
                } else { h = tile_size; w = tile_size; }
            };

            for (size_t in = 0; in < ntasks; ++in) {
                const auto &t = tasks[in];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                switch (t[0]) {
                    case 1: { // diagonal-tile inversion: copy U_ii + in-place LAPACKE_dtrtri
                        int h = tile_size, w = tile_size;
                        tile_dims(index1, h, w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index1] : nullptr;
                        double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (!fact || !inv) {
                            sTiles::Logger::warning("[pthreads_dpotri_dense_serial case 1] Null tile at index1=", index1);
                            break;
                        }
                        for (int jj = 0; jj < w; ++jj)
                            std::memcpy(inv + static_cast<size_t>(jj) * h,
                                        fact + static_cast<size_t>(jj) * h,
                                        static_cast<size_t>(jj + 1) * sizeof(double));
                        const lapack_int info1 = LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', w, inv, h);
                        if (info1 != 0) {
                            std::fprintf(stderr, "sTiles error: pthreads_dpotri_dense_serial LAPACKE_dtrtri failed info=%d (tile idx=%d).\n",
                                         static_cast<int>(info1), index1);
                            stile->ss_abort = 1;
                            return;
                        }
                        break;
                    }
                    case 2: { // TRMM: fact(index2) = inv(index1) * fact(index2)
                        int inv_h = tile_size, inv_w = tile_size;
                        tile_dims(index1, inv_h, inv_w);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index2, mh, mw);
                        double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index2] : nullptr;
                        if (inv && fact) {
                            sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                               sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                               mh, mw, zone, inv, inv_h, fact, mh);
                        }
                        break;
                    }
                    case 4: { // LAUUM + mirror on inv(index1)
                        int h = tile_size, w = tile_size;
                        tile_dims(index1, h, w);
                        double* inv = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (inv) {
                            sTiles::core_dlauum(sTiles::Uplo::Upper, w, inv, h);
                            mirroring(h, inv, inv, h);
                        }
                        break;
                    }
                    case 5: { // GEMM accumulate into diagonal inv(index1)
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index2, mh, mw);
                        int inv1_h = tile_size, inv1_w = tile_size;
                        tile_dims(index1, inv1_h, inv1_w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index2] : nullptr;
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv1 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (fact && inv2 && inv1) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                               mh, mh, mw, mzone, fact, mh, inv2, mh, zone, inv1, inv1_h);
                        }
                        break;
                    }
                    case 7: { // GEMM update inv(index3), transposed source
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index1, mh, mw);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        int inv3_h = tile_size, inv3_w = tile_size;
                        tile_dims(index3, inv3_h, inv3_w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index1] : nullptr;
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                        if (fact && inv2 && inv3) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                               mh, inv2_h, mw, mzone, fact, mh, inv2, inv2_h, zone, inv3, inv3_h);
                        }
                        break;
                    }
                    case 8: { // GEMM update inv(index3)
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index1, mh, mw);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        int inv3_h = tile_size, inv3_w = tile_size;
                        tile_dims(index3, inv3_h, inv3_w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index1] : nullptr;
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                        if (fact && inv2 && inv3) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                               mh, inv2_w, mw, mzone, fact, mh, inv2, inv2_h, zone, inv3, inv3_h);
                        }
                        break;
                    }
                    case 3:   // phase barrier: no-op in serial
                    case 6:   // dependency flags: no-op in serial
                    case 9:
                    default:
                        break;
                }
            }
        }

        static void omp_dpotri_dense_serial(TiledMatrix *tiledMatrix, omp_dep_tracker_t* dep_tracker)
        {
            const int tile_size = tiledMatrix->tile_size;
            const double zone  = 1.0;
            const double mzone = -1.0;

            const auto &tasks = sTiles::get_inv_tasks(tiledMatrix);
            const size_t ntasks = tasks.size();

            auto tile_dims = [&](int dense_idx, int& h, int& w) {
                if (tiledMatrix->tileMetaCore && dense_idx >= 0) {
                    const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[dense_idx];
                    h = (meta.height > 0) ? meta.height : tile_size;
                    w = (meta.width  > 0) ? meta.width  : tile_size;
                } else { h = tile_size; w = tile_size; }
            };

            for (size_t in = 0; in < ntasks; ++in) {
                const auto &t = tasks[in];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                switch (t[0]) {
                    case 1: { // diagonal-tile inversion: copy U_ii + in-place LAPACKE_dtrtri
                        int h = tile_size, w = tile_size;
                        tile_dims(index1, h, w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index1] : nullptr;
                        double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (!fact || !inv) {
                            sTiles::Logger::warning("[omp_dpotri_dense_serial case 1] Null tile at index1=", index1);
                            break;
                        }
                        for (int jj = 0; jj < w; ++jj)
                            std::memcpy(inv + static_cast<size_t>(jj) * h,
                                        fact + static_cast<size_t>(jj) * h,
                                        static_cast<size_t>(jj + 1) * sizeof(double));
                        const lapack_int info1 = LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', w, inv, h);
                        if (info1 != 0) {
                            std::fprintf(stderr, "sTiles error: omp_dpotri_dense_serial LAPACKE_dtrtri failed info=%d (tile idx=%d).\n",
                                         static_cast<int>(info1), index1);
                            dep_tracker->abort_flag.store(true, std::memory_order_release);
                            return;
                        }
                        break;
                    }
                    case 2: { // TRMM: fact(index2) = inv(index1) * fact(index2)
                        int inv_h = tile_size, inv_w = tile_size;
                        tile_dims(index1, inv_h, inv_w);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index2, mh, mw);
                        double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index2] : nullptr;
                        if (inv && fact) {
                            sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                               sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                               mh, mw, zone, inv, inv_h, fact, mh);
                        }
                        break;
                    }
                    case 4: { // LAUUM + mirror on inv(index1)
                        int h = tile_size, w = tile_size;
                        tile_dims(index1, h, w);
                        double* inv = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (inv) {
                            sTiles::core_dlauum(sTiles::Uplo::Upper, w, inv, h);
                            mirroring(h, inv, inv, h);
                        }
                        break;
                    }
                    case 5: { // GEMM accumulate into diagonal inv(index1)
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index2, mh, mw);
                        int inv1_h = tile_size, inv1_w = tile_size;
                        tile_dims(index1, inv1_h, inv1_w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index2] : nullptr;
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv1 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (fact && inv2 && inv1) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                               mh, mh, mw, mzone, fact, mh, inv2, mh, zone, inv1, inv1_h);
                        }
                        break;
                    }
                    case 7: { // GEMM update inv(index3), transposed source
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index1, mh, mw);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        int inv3_h = tile_size, inv3_w = tile_size;
                        tile_dims(index3, inv3_h, inv3_w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index1] : nullptr;
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                        if (fact && inv2 && inv3) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                               mh, inv2_h, mw, mzone, fact, mh, inv2, inv2_h, zone, inv3, inv3_h);
                        }
                        break;
                    }
                    case 8: { // GEMM update inv(index3)
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index1, mh, mw);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        int inv3_h = tile_size, inv3_w = tile_size;
                        tile_dims(index3, inv3_h, inv3_w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index1] : nullptr;
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                        if (fact && inv2 && inv3) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                               mh, inv2_w, mw, mzone, fact, mh, inv2, inv2_h, zone, inv3, inv3_h);
                        }
                        break;
                    }
                    case 3:   // phase barrier: no-op in serial
                    case 6:   // dependency flags: no-op in serial
                    case 9:
                    default:
                        break;
                }
            }
        }

        static void pthreads_dpotri_dense_parallel(TiledMatrix *tiledMatrix, stiles_context_t *stile)
        {
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const double zone  = 1.0;
            const double mzone = -1.0;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            const auto &tasks   = sTiles::get_inv_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_inv_task_offsets(tiledMatrix);
            size_t start = 0, end = tasks.size();
            if (!offsets.empty()) {
                const size_t r = static_cast<size_t>(STILES_RANK);
                if (r < offsets.size()) start = static_cast<size_t>(offsets[r]);
                if (r + 1 < offsets.size()) end = static_cast<size_t>(offsets[r + 1]);
            }

            auto tile_dims = [&](int dense_idx, int& h, int& w) {
                if (tiledMatrix->tileMetaCore && dense_idx >= 0) {
                    const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[dense_idx];
                    h = (meta.height > 0) ? meta.height : tile_size;
                    w = (meta.width  > 0) ? meta.width  : tile_size;
                } else {
                    h = tile_size;
                    w = tile_size;
                }
            };

            // Debug: Print tile mapping info (commented out for production)
            // if (STILES_RANK == 0) {
            //     std::cout << "\n========== pthreads_dpotri_dense_parallel DEBUG ==========\n";
            //     std::cout << "N=" << N << " tile_size=" << tile_size << " num_tiles=" << num_tiles_per_dim << "\n";
            //     std::cout << "numActiveTiles=" << tiledMatrix->numActiveTiles << "\n";
            //     std::cout << "tasks.size()=" << tasks.size() << " start=" << start << " end=" << end << "\n";
            //     if (tiledMatrix->tileMetaCore) {
            //         std::cout << "TileMetaCore info:\n";
            //         for (int t = 0; t < tiledMatrix->numActiveTiles; ++t) {
            //             const auto& m = tiledMatrix->tileMetaCore[t];
            //             std::cout << "  tile " << t << ": row=" << m.row << " col=" << m.col
            //                       << " h=" << m.height << " w=" << m.width << "\n";
            //         }
            //     }
            //     std::cout << "\n--- Factor tiles (denseTiles) before selinv ---\n";
            //     for (int t = 0; t < tiledMatrix->numActiveTiles; ++t) {
            //         int h = tile_size, w = tile_size;
            //         tile_dims(t, h, w);
            //         debug_print_tile("FACTOR", t, tiledMatrix->denseTiles ? tiledMatrix->denseTiles[t] : nullptr, h, w, 0);
            //     }
            //     std::cout << "\n--- Inverse tiles (inverseTiles) before selinv ---\n";
            //     for (int t = 0; t < tiledMatrix->numActiveTiles; ++t) {
            //         int h = tile_size, w = tile_size;
            //         tile_dims(t, h, w);
            //         debug_print_tile("INV_BEFORE", t, tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[t] : nullptr, h, w, 0);
            //     }
            //     std::cout << "\n--- Task list ---\n";
            //     for (size_t in = start; in < end && in < start + 30; ++in) {
            //         const auto& t = tasks[in];
            //         std::cout << "  task " << in << ": type=" << t[0] << " i=" << t[1] << " j=" << t[2]
            //                   << " k=" << t[3] << " idx1=" << t[4] << " idx2=" << t[5] << " idx3=" << t[6] << "\n";
            //     }
            // }

            in_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            int global_in = -1;

            // Phase 1: until routine 3 (barrier)
            for (size_t in = start; in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int index1 = t[4];
                const int index2 = t[5];

                switch (myroutine) {
                    case 1: // TRSM: inv(index1) = fact(index1)^{-1}
                    {
                        int h = tile_size, w = tile_size;
                        tile_dims(index1, h, w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index1] : nullptr;
                        double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (!fact || !inv) {
                            sTiles::Logger::warning("[pthreads_dpotri_dense_parallel case 1] Null tile at index1=", index1,
                                                    " fact=", (fact ? "valid" : "NULL"),
                                                    " inv=", (inv ? "valid" : "NULL"));
                            break;
                        }

                        // if (STILES_RANK == 0) {
                        //     std::cout << "\n[CASE 1 TRSM] index1=" << index1 << " h=" << h << " w=" << w << "\n";
                        //     debug_print_tile("FACT before TRSM", index1, fact, h, w, 0);
                        //     debug_print_tile("INV before TRSM", index1, inv, h, w, 0);
                        // }

                        // Diagonal-tile inversion: copy U_ii into inv, invert in place with
                        // LAPACKE_dtrtri (w^3/3 flops vs ~w^3 for DTRSM against the identity seed).
                        for (int jj = 0; jj < w; ++jj)
                            std::memcpy(inv + static_cast<size_t>(jj) * h,
                                        fact + static_cast<size_t>(jj) * h,
                                        static_cast<size_t>(jj + 1) * sizeof(double));
                        {
                            const lapack_int info1 = LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', w, inv, h);
                            if (info1 != 0) {
                                std::fprintf(stderr, "sTiles error: pthreads_dpotri_dense_parallel LAPACKE_dtrtri failed info=%d (tile idx=%d).\n",
                                            static_cast<int>(info1), index1);
                                stile->ss_abort = 1;   // surfaced as ExecutionFailed by pthreads_dpotri
                            }
                        }

                        // if (STILES_RANK == 0) {
                        //     debug_print_tile("INV after TRSM", index1, inv, h, w, 0);
                        // }
                        break;
                    }
                    case 2: // TRMM: fact(index2) *= inv(index1)
                    {
                        int inv_h = tile_size, inv_w = tile_size;
                        tile_dims(index1, inv_h, inv_w);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index2, mh, mw);
                        double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index2] : nullptr;
                        if (!inv || !fact) {
                            sTiles::Logger::warning("[pthreads_dpotri_dense_parallel case 2] Null tile at index1=", index1, " index2=", index2,
                                                    " inv=", (inv ? "valid" : "NULL"),
                                                    " fact=", (fact ? "valid" : "NULL"));
                            break;
                        }

                        // if (STILES_RANK == 0) {
                        //     std::cout << "\n[CASE 2 TRMM] index1=" << index1 << " index2=" << index2 << "\n";
                        //     debug_print_tile("INV (A)", index1, inv, inv_h, inv_w, 0);
                        //     debug_print_tile("FACT (B) before TRMM", index2, fact, mh, mw, 0);
                        // }

                        if (inv && fact) {
                            sTiles::core_dtrmm(sTiles::Side::Left,
                                            sTiles::Uplo::Upper,
                                            sTiles::Op::NoTrans,
                                            sTiles::Diag::NonUnit,
                                            mh, mw,
                                            zone,
                                            inv,  inv_h,
                                            fact, mh);
                        }

                        // if (STILES_RANK == 0) {
                        //     debug_print_tile("FACT (B) after TRMM", index2, fact, mh, mw, 0);
                        // }
                        break;
                    }
                    case 3: // Barrier point
                    {
                        // if (STILES_RANK == 0) {
                        //     std::cout << "\n[CASE 3 BARRIER]\n";
                        //     std::cout << "--- State after Phase 1 ---\n";
                        //     for (int t = 0; t < tiledMatrix->numActiveTiles; ++t) {
                        //         int h = tile_size, w = tile_size;
                        //         tile_dims(t, h, w);
                        //         debug_print_tile("INV after phase1", t, tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[t] : nullptr, h, w, 0);
                        //     }
                        // }
                        sTiles::Control::Barrier(stile);
                        global_in = static_cast<int>(in) + 1;
                        goto exit_phase1_dense;
                    }
                    default:
                        break;
                }
            }

        exit_phase1_dense:

            if (global_in < 0) global_in = static_cast<int>(start);

            // if (STILES_RANK == 0) {
            //     std::cout << "\n========== PHASE 2 START ==========\n";
            // }

            // Phase 2: remaining tasks
            for (size_t in = static_cast<size_t>(global_in); in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int i = t[1];
                const int j = t[2];
                const int k = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                switch (myroutine) {
                    case 4: // LAUUM + mirror on inv(index1)
                    {
                        int h = tile_size, w = tile_size;
                        tile_dims(index1, h, w);
                        double* inv = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (!inv) {
                            sTiles::Logger::warning("[pthreads_dpotri_dense_parallel case 4] Null inverse tile at index1=", index1);
                            break;
                        }

                        // if (STILES_RANK == 0) {
                        //     std::cout << "\n[CASE 4 LAUUM] i=" << i << " j=" << j << " index1=" << index1 << "\n";
                        //     debug_print_tile("INV before LAUUM", index1, inv, h, w, 0);
                        // }

                        if (inv) {
                            sTiles::core_dlauum(sTiles::Uplo::Upper, w, inv, h);
                            mirroring(h, inv, inv, h);
                        }

                        // if (STILES_RANK == 0) {
                        //     debug_print_tile("INV after LAUUM+mirror", index1, inv, h, w, 0);
                        // }
                        break;
                    }
                    case 5: // GEMM accumulate into inv(index1)
                    {
                        in_cond_wait(i, k, 2);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index2, mh, mw);
                        int inv1_h = tile_size, inv1_w = tile_size;
                        tile_dims(index1, inv1_h, inv1_w);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index2] : nullptr;
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv1 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (!fact || !inv2 || !inv1) {
                            sTiles::Logger::warning("[pthreads_dpotri_dense_parallel case 5] Null tile at i=", i, " k=", k,
                                                    " index1=", index1, " index2=", index2,
                                                    " fact=", (fact ? "valid" : "NULL"),
                                                    " inv2=", (inv2 ? "valid" : "NULL"),
                                                    " inv1=", (inv1 ? "valid" : "NULL"));
                            break;
                        }

                        // if (STILES_RANK == 0) {
                        //     std::cout << "\n[CASE 5 GEMM diag] i=" << i << " k=" << k << " index1=" << index1 << " index2=" << index2 << "\n";
                        //     std::cout << "  GEMM: inv1 -= fact * inv2^T  (mh=" << mh << " mw=" << mw << ")\n";
                        //     debug_print_tile("FACT (index2)", index2, fact, mh, mw, 0);
                        //     debug_print_tile("INV2 (index2)", index2, inv2, inv2_h, inv2_w, 0);
                        //     debug_print_tile("INV1 before GEMM", index1, inv1, inv1_h, inv1_w, 0);
                        // }

                        if (fact && inv2 && inv1) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans,
                                            sTiles::Op::Trans,
                                            mh, mh, mw,
                                            mzone,
                                            fact, mh,
                                            inv2, inv2_h,
                                            zone,
                                            inv1, inv1_h);
                        }

                        // if (STILES_RANK == 0) {
                        //     debug_print_tile("INV1 after GEMM", index1, inv1, inv1_h, inv1_w, 0);
                        // }
                        break;
                    }
                    case 6:
                        // if (STILES_RANK == 0) {
                        //     std::cout << "\n[CASE 6 SET] i=" << i << " j=" << j << "\n";
                        // }
                        in_cond_set(i, i, 2);
                        break;
                    case 7: // GEMM update inv(index3)
                    {
                        in_cond_wait(j, k, 2);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index1, mh, mw);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        int inv3_h = tile_size, inv3_w = tile_size;
                        tile_dims(index3, inv3_h, inv3_w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index1] : nullptr;
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                        if (!fact || !inv2 || !inv3) {
                            sTiles::Logger::warning("[pthreads_dpotri_dense_parallel case 7] Null tile at j=", j, " k=", k,
                                                    " index1=", index1, " index2=", index2, " index3=", index3,
                                                    " fact=", (fact ? "valid" : "NULL"),
                                                    " inv2=", (inv2 ? "valid" : "NULL"),
                                                    " inv3=", (inv3 ? "valid" : "NULL"));
                            break;
                        }

                        // if (STILES_RANK == 0) {
                        //     std::cout << "\n[CASE 7 GEMM off-diag Trans] i=" << i << " j=" << j << " k=" << k
                        //               << " idx1=" << index1 << " idx2=" << index2 << " idx3=" << index3 << "\n";
                        //     std::cout << "  GEMM: inv3 -= fact * inv2^T\n";
                        //     debug_print_tile("FACT (index1)", index1, fact, mh, mw, 0);
                        //     debug_print_tile("INV2 (index2)", index2, inv2, inv2_h, inv2_w, 0);
                        //     debug_print_tile("INV3 before GEMM", index3, inv3, inv3_h, inv3_w, 0);
                        // }

                        if (fact && inv2 && inv3) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans,
                                            sTiles::Op::Trans,
                                            mh, inv2_h, mw,
                                            -1.0,
                                            fact, mh,
                                            inv2, inv2_h,
                                            zone,
                                            inv3, inv3_h);
                        }

                        // if (STILES_RANK == 0) {
                        //     debug_print_tile("INV3 after GEMM", index3, inv3, inv3_h, inv3_w, 0);
                        // }
                        break;
                    }
                    case 8: // GEMM update inv(index3)
                    {
                        in_cond_wait(k, j, 2);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index1, mh, mw);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        int inv3_h = tile_size, inv3_w = tile_size;
                        tile_dims(index3, inv3_h, inv3_w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index1] : nullptr;
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                        if (!fact || !inv2 || !inv3) {
                            sTiles::Logger::warning("[pthreads_dpotri_dense_parallel case 8] Null tile at k=", k, " j=", j,
                                                    " index1=", index1, " index2=", index2, " index3=", index3,
                                                    " fact=", (fact ? "valid" : "NULL"),
                                                    " inv2=", (inv2 ? "valid" : "NULL"),
                                                    " inv3=", (inv3 ? "valid" : "NULL"));
                            break;
                        }

                        // if (STILES_RANK == 0) {
                        //     std::cout << "\n[CASE 8 GEMM off-diag NoTrans] i=" << i << " j=" << j << " k=" << k
                        //               << " idx1=" << index1 << " idx2=" << index2 << " idx3=" << index3 << "\n";
                        //     std::cout << "  GEMM: inv3 -= fact * inv2\n";
                        //     debug_print_tile("FACT (index1)", index1, fact, mh, mw, 0);
                        //     debug_print_tile("INV2 (index2)", index2, inv2, inv2_h, inv2_w, 0);
                        //     debug_print_tile("INV3 before GEMM", index3, inv3, inv3_h, inv3_w, 0);
                        // }

                        if (fact && inv2 && inv3) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans,
                                            sTiles::Op::NoTrans,
                                            mh, inv2_w, mw,
                                            mzone,
                                            fact, mh,
                                            inv2, inv2_h,
                                            zone,
                                            inv3, inv3_h);
                        }

                        // if (STILES_RANK == 0) {
                        //     debug_print_tile("INV3 after GEMM", index3, inv3, inv3_h, inv3_w, 0);
                        // }
                        break;
                    }
                    case 9:
                        // if (STILES_RANK == 0) {
                        //     std::cout << "\n[CASE 9 SET] i=" << i << " j=" << j << "\n";
                        // }
                        in_cond_set(i, j, 2);
                        break;
                    default:
                        break;
                }
            }

            // if (STILES_RANK == 0) {
            //     std::cout << "\n========== FINAL STATE ==========\n";
            //     for (int t = 0; t < tiledMatrix->numActiveTiles; ++t) {
            //         int h = tile_size, w = tile_size;
            //         tile_dims(t, h, w);
            //         debug_print_tile("FINAL INV", t, tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[t] : nullptr, h, w, 0);
            //     }
            // }

            in_finalize();
        }

        static void omp_dpotri_dense_parallel(TiledMatrix *tiledMatrix, omp_dep_tracker_t* dep_tracker, int worldsize)
        {
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const double zone  = 1.0;
            const double mzone = -1.0;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int rank = omp_get_thread_num();

            const auto &tasks   = sTiles::get_inv_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_inv_task_offsets(tiledMatrix);
            size_t start = 0, end = tasks.size();
            if (!offsets.empty()) {
                const size_t r = static_cast<size_t>(rank);
                if (r < offsets.size()) start = static_cast<size_t>(offsets[r]);
                if (r + 1 < offsets.size()) end = static_cast<size_t>(offsets[r + 1]);
            }

            auto tile_dims = [&](int dense_idx, int& h, int& w) {
                if (tiledMatrix->tileMetaCore && dense_idx >= 0) {
                    const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[dense_idx];
                    h = (meta.height > 0) ? meta.height : tile_size;
                    w = (meta.width  > 0) ? meta.width  : tile_size;
                } else {
                    h = tile_size;
                    w = tile_size;
                }
            };

            dep_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            int global_in = -1;

            // Phase 1: until routine 3 (barrier)
            for (size_t in = start; in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int index1 = t[4];
                const int index2 = t[5];

                switch (myroutine) {
                    case 1: // TRSM: inv(index1) = fact(index1)^{-1}
                    {
                        int h = tile_size, w = tile_size;
                        tile_dims(index1, h, w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index1] : nullptr;
                        double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (fact && inv) {
                            // Diagonal-tile inversion via LAPACKE_dtrtri (w^3/3 vs ~w^3 for DTRSM).
                            for (int jj = 0; jj < w; ++jj)
                                std::memcpy(inv + static_cast<size_t>(jj) * h,
                                            fact + static_cast<size_t>(jj) * h,
                                            static_cast<size_t>(jj + 1) * sizeof(double));
                            const lapack_int info1 = LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', w, inv, h);
                            if (info1 != 0) {
                                std::fprintf(stderr, "sTiles error: omp_dpotri_dense_parallel LAPACKE_dtrtri failed info=%d (tile idx=%d).\n",
                                            static_cast<int>(info1), index1);
                                dep_tracker->abort_flag.store(true, std::memory_order_release);  // -> ExecutionFailed
                            }
                        }
                        break;
                    }
                    case 2: // TRMM: fact(index2) *= inv(index1)
                    {
                        int inv_h = tile_size, inv_w = tile_size;
                        tile_dims(index1, inv_h, inv_w);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index2, mh, mw);
                        double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index2] : nullptr;
                        if (inv && fact) {
                            sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                            sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                            mh, mw, zone, inv, inv_h, fact, mh);
                        }
                        break;
                    }
                    case 3: // Barrier point
                    {
                        #pragma omp barrier
                        global_in = static_cast<int>(in) + 1;
                        goto exit_phase1_dense_omp;
                    }
                    default:
                        break;
                }
            }

        exit_phase1_dense_omp:
            if (global_in < 0) global_in = static_cast<int>(start);

            // Phase 2: remaining tasks with dependency tracking
            for (size_t in = static_cast<size_t>(global_in); in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int i = t[1];
                const int j = t[2];
                const int k = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                switch (myroutine) {
                    case 4: // LAUUM + mirror on inv(index1)
                    {
                        int h = tile_size, w = tile_size;
                        tile_dims(index1, h, w);
                        double* inv = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (inv) {
                            sTiles::core_dlauum(sTiles::Uplo::Upper, w, inv, h);
                            mirroring(h, inv, inv, h);
                        }
                        break;
                    }
                    case 5: // GEMM accumulate into inv(index1)
                    {
                        dep_wait_for(i, k, 2);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index2, mh, mw);
                        int inv1_h = tile_size, inv1_w = tile_size;
                        tile_dims(index1, inv1_h, inv1_w);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index2] : nullptr;
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv1 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                        if (fact && inv2 && inv1) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                            mh, mh, mw, mzone, fact, mh, inv2, inv2_h, zone, inv1, inv1_h);
                        }
                        break;
                    }
                    case 6:
                        dep_set_done(i, i, 2);
                        break;
                    case 7: // GEMM update inv(index3)
                    {
                        dep_wait_for(j, k, 2);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index1, mh, mw);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        int inv3_h = tile_size, inv3_w = tile_size;
                        tile_dims(index3, inv3_h, inv3_w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index1] : nullptr;
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                        if (fact && inv2 && inv3) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                            mh, inv2_h, mw, -1.0, fact, mh, inv2, inv2_h, zone, inv3, inv3_h);
                        }
                        break;
                    }
                    case 8: // GEMM update inv(index3)
                    {
                        dep_wait_for(k, j, 2);
                        int mh = tile_size, mw = tile_size;
                        tile_dims(index1, mh, mw);
                        int inv2_h = tile_size, inv2_w = tile_size;
                        tile_dims(index2, inv2_h, inv2_w);
                        int inv3_h = tile_size, inv3_w = tile_size;
                        tile_dims(index3, inv3_h, inv3_w);
                        double* fact = tiledMatrix->denseTiles ? tiledMatrix->denseTiles[index1] : nullptr;
                        double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                        double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                        if (fact && inv2 && inv3) {
                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                            mh, inv2_w, mw, mzone, fact, mh, inv2, inv2_h, zone, inv3, inv3_h);
                        }
                        break;
                    }
                    case 9:
                        dep_set_done(i, j, 2);
                        break;
                    default:
                        break;
                }
            }

            dep_finalize();
        }

        ////////////////////////////////////////////////////////////////////////////////

        //------------------> Sparse of Semisparse Tiles  -    (Tile mode 1)
        
        ////////////////////////////////////////////////////////////////////////////////

        static void pthreads_dpotri_semi_dense_inv(TiledMatrix *tiledMatrix, stiles_context_t *stile)
        {
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const double zone  = 1.0;
            const double mzone = -1.0;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            // Get workspace for temporary gather operations
            const int rank = STILES_RANK;
            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[trtri] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* workspace = tiledMatrix->workspaces[rank]->aligned_tile();

            const auto &tasks   = sTiles::get_inv_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_inv_task_offsets(tiledMatrix);
            size_t start = 0, end = tasks.size();
            if (!offsets.empty()) {
                const size_t r = static_cast<size_t>(STILES_RANK);
                if (r < offsets.size()) start = static_cast<size_t>(offsets[r]);
                if (r + 1 < offsets.size()) end = static_cast<size_t>(offsets[r + 1]);
            }

            in_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            int global_in = -1;

            // ========== Phase 1: Compute L^{-1} for diagonal tiles, then L^{-1} * L_offdiag ==========
            for (size_t in = start; in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int index1 = t[4];
                const int index2 = t[5];

                switch (myroutine) {
                    case 1: // TRSM: Compute L^{-1} for diagonal tile
                    {
                        // Factor is BANDED, inverse is DENSE (L^{-1} fills in)
                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                        if (!fact || !inv) break;

                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                        const int n = meta.height;
                        const int kd = semi.upper_bw;

                        if (kd >= 0) {
                            // Banded triangular solve: L * X = I  =>  X = L^{-1}
                            // inv starts as identity, becomes L^{-1} (dense)
                            // kd >= 0: banded format with ldab = kd + 1
                            // kd = 0 means pure diagonal matrix (ldab = 1)
                            banded_diag_inverse(n, kd, fact, inv);
                        } else {
                            // Dense triangular solve (kd < 0 indicates dense storage)
                            sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                            sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                            n, n, 1.0, fact, n, inv, n);
                        }
                        break;
                    }
                    case 2: // TRMM: Compute L^{-1} * L_offdiag (stored back in factor tile)
                    {
                        // inv (index1) = diagonal L^{-1} (DENSE)
                        // fact (index2) = off-diagonal L (ACTIVE COLUMNS format, sa columns)
                        double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                        double* fact = tiledMatrix->chunkedDenseTiles[index2];
                        if (!inv || !fact) break;

                        const sTiles::TileMetaCore& meta_inv = tiledMatrix->tileMetaCore[index1];
                        const sTiles::TileMetaCore& meta_fact = tiledMatrix->tileMetaCore[index2];
                        const sTiles::SemisparseTileMetaCore& semi_fact = tiledMatrix->semisparseTileMetaCore[index2];

                        const int m = meta_fact.height;  // rows of off-diag tile
                        const int sa = semi_fact.sa;     // number of active columns

                        if (sa > 0 && !semi_case2_banded_solve(tiledMatrix, index1, m, sa, fact)) {
                            // TRMM: fact = inv * fact (operates on sa columns)
                            sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                            sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                            m, sa, zone, inv, meta_inv.height, fact, m);
                        }
                        break;
                    }
                    case 3: // Barrier
                    {
                        sTiles::Control::Barrier(stile);
                        global_in = static_cast<int>(in) + 1;
                        goto exit_phase1_semisparse;
                    }
                    default:
                        break;
                }
            }

        exit_phase1_semisparse:
            if (global_in < 0) global_in = static_cast<int>(start);

            // ========== Phase 2: Compute selected inverse entries ==========
            for (size_t in = static_cast<size_t>(global_in); in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int i = t[1];
                const int j = t[2];
                const int k = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                switch (myroutine) {
                    case 4: // LAUUM + mirror: Compute L^{-T} * L^{-1} for diagonal tile
                    {
                        double* inv = tiledMatrix->chunkedInverseTiles[index1];
                        if (!inv) break;

                        const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                        const int n = meta.height;

                        sTiles::core_dlauum(sTiles::Uplo::Upper, n, inv, n);
                        mirroring(n, inv, inv, n);
                        break;
                    }
                    case 5: // GEMM: Diagonal tile update: inv1 -= (L^{-1}*L_offdiag) * inv2^T
                    {
                        // index1 = diagonal inverse tile (DENSE)
                        // index2 = off-diagonal tile: factor has ACTIVE COLS, inverse is DENSE
                        in_cond_wait(i, k, 2);

                        double* fact = tiledMatrix->chunkedDenseTiles[index2];   // L^{-1}*L_offdiag in ACTIVE COLS
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2]; // DENSE
                        double* inv1 = tiledMatrix->chunkedInverseTiles[index1]; // DENSE (output)
                        if (!fact || !inv2 || !inv1) break;

                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::TileMetaCore& meta1 = tiledMatrix->tileMetaCore[index1];
                        const sTiles::TileMetaCore& meta2 = tiledMatrix->tileMetaCore[index2];

                        const int m = meta2.height;      // rows
                        const int n = meta2.width;       // cols of dense inverse tile
                        const int sa = semi.sa;          // active columns in factor
                        if (sa <= 0) break;

                        // Gather columns from inv2 at active column indices
                        // gathered = inv2[:, aind[0]], inv2[:, aind[1]], ... (m x sa)
                        for (int jj = 0; jj < sa && jj < (int)semi.aind.size(); ++jj) {
                            const int col = semi.aind[jj];
                            if (col >= 0 && col < n) {
                                for (int ii = 0; ii < m; ++ii) {
                                    workspace[ii + jj * m] = inv2[ii + col * m];
                                }
                            } else {
                                for (int ii = 0; ii < m; ++ii) {
                                    workspace[ii + jj * m] = 0.0;
                                }
                            }
                        }

                        // GEMM: inv1 -= fact * gathered^T  =>  (m x m) -= (m x sa) * (sa x m)
                        sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                        m, m, sa, mzone,
                                        fact, m,
                                        workspace, m,
                                        zone, inv1, meta1.height);
                        break;
                    }
                    case 6:
                        in_cond_set(i, i, 2);
                        break;

                    case 7: // GEMM: Off-diagonal update: inv3 -= fact * inv2^T
                    {
                        // index1 = off-diagonal factor (ACTIVE COLS)
                        // index2 = off-diagonal inverse (DENSE)
                        // index3 = off-diagonal inverse output (DENSE)
                        in_cond_wait(j, k, 2);

                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                        if (!fact || !inv2 || !inv3) break;

                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::TileMetaCore& meta1 = tiledMatrix->tileMetaCore[index1];
                        const sTiles::TileMetaCore& meta2 = tiledMatrix->tileMetaCore[index2];
                        const sTiles::TileMetaCore& meta3 = tiledMatrix->tileMetaCore[index3];

                        const int m_fact = meta1.height;
                        const int m_inv2 = meta2.height;
                        const int n_inv2 = meta2.width;
                        const int sa = semi.sa;
                        if (sa <= 0) break;

                        // Gather columns from inv2 at active column indices
                        // gathered = inv2[:, aind[]] (m_inv2 x sa)
                        for (int jj = 0; jj < sa && jj < (int)semi.aind.size(); ++jj) {
                            const int col = semi.aind[jj];
                            if (col >= 0 && col < n_inv2) {
                                for (int ii = 0; ii < m_inv2; ++ii) {
                                    workspace[ii + jj * m_inv2] = inv2[ii + col * m_inv2];
                                }
                            } else {
                                for (int ii = 0; ii < m_inv2; ++ii) {
                                    workspace[ii + jj * m_inv2] = 0.0;
                                }
                            }
                        }

                        // GEMM: inv3 -= fact * gathered^T  =>  (m_fact x m_inv2) -= (m_fact x sa) * (sa x m_inv2)
                        sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                        m_fact, m_inv2, sa, mzone,
                                        fact, m_fact,
                                        workspace, m_inv2,
                                        zone, inv3, meta3.height);
                        break;
                    }
                    case 8: // GEMM: Off-diagonal update: inv3 -= fact * inv2 (NoTrans)
                    {
                        // index1 = (i,k) off-diagonal factor (ACTIVE COLS)
                        // index2 = (k,j) inverse tile (DENSE)
                        // index3 = (i,j) off-diagonal inverse output (DENSE)
                        in_cond_wait(k, j, 2);

                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                        if (!fact || !inv2 || !inv3) break;

                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::TileMetaCore& meta1 = tiledMatrix->tileMetaCore[index1];
                        const sTiles::TileMetaCore& meta2 = tiledMatrix->tileMetaCore[index2];
                        const sTiles::TileMetaCore& meta3 = tiledMatrix->tileMetaCore[index3];

                        const int m_fact = meta1.height;
                        const int m_inv2 = meta2.height;
                        const int n_inv2 = meta2.width;
                        const int sa = semi.sa;
                        if (sa <= 0) break;

                        // For fact * inv2 with ACTIVE COLS format:
                        // fact column j corresponds to original column aind[j]
                        // So we need rows aind[] of inv2
                        // gathered[i, :] = inv2[aind[i], :] (sa x n_inv2)
                        for (int ii = 0; ii < sa && ii < (int)semi.aind.size(); ++ii) {
                            const int row = semi.aind[ii];
                            if (row >= 0 && row < m_inv2) {
                                for (int jj = 0; jj < n_inv2; ++jj) {
                                    workspace[ii + jj * sa] = inv2[row + jj * m_inv2];
                                }
                            } else {
                                for (int jj = 0; jj < n_inv2; ++jj) {
                                    workspace[ii + jj * sa] = 0.0;
                                }
                            }
                        }

                        // GEMM: inv3 -= fact * gathered  =>  (m_fact x n_inv2) -= (m_fact x sa) * (sa x n_inv2)
                        sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                        m_fact, n_inv2, sa, mzone,
                                        fact, m_fact,
                                        workspace, sa,
                                        zone, inv3, meta3.height);
                        break;
                    }
                    case 9:
                        in_cond_set(i, j, 2);
                        break;

                    default:
                        break;
                }
            }

            in_finalize();
        }

        // OMP mirror of pthreads_dpotri_semi_dense_inv (mode 1, dense inverse storage).
        // Mechanical dep_* swap of the pthreads body -> bit-identical results.
        static void omp_dpotri_semi_dense_inv(TiledMatrix *tiledMatrix, omp_dep_tracker_t* dep_tracker, int worldsize)
        {
            (void)dep_tracker; (void)worldsize;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const double zone  = 1.0;
            const double mzone = -1.0;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            // Get workspace for temporary gather operations
            const int rank = omp_get_thread_num();
            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[omp_dpotri_semi_dense_inv] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* workspace = tiledMatrix->workspaces[rank]->aligned_tile();

            const auto &tasks   = sTiles::get_inv_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_inv_task_offsets(tiledMatrix);
            size_t start = 0, end = tasks.size();
            if (!offsets.empty()) {
                const size_t r = static_cast<size_t>(rank);
                if (r < offsets.size()) start = static_cast<size_t>(offsets[r]);
                if (r + 1 < offsets.size()) end = static_cast<size_t>(offsets[r + 1]);
            }

            dep_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            int global_in = -1;

            // ========== Phase 1: Compute L^{-1} for diagonal tiles, then L^{-1} * L_offdiag ==========
            for (size_t in = start; in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int index1 = t[4];
                const int index2 = t[5];

                switch (myroutine) {
                    case 1: // TRSM: Compute L^{-1} for diagonal tile
                    {
                        // Factor is BANDED, inverse is DENSE (L^{-1} fills in)
                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                        if (!fact || !inv) break;

                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                        const int n = meta.height;
                        const int kd = semi.upper_bw;

                        if (kd >= 0) {
                            // Banded triangular solve: L * X = I  =>  X = L^{-1}
                            // inv starts as identity, becomes L^{-1} (dense)
                            // kd >= 0: banded format with ldab = kd + 1
                            // kd = 0 means pure diagonal matrix (ldab = 1)
                            banded_diag_inverse(n, kd, fact, inv);
                        } else {
                            // Dense triangular solve (kd < 0 indicates dense storage)
                            sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                            sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                            n, n, 1.0, fact, n, inv, n);
                        }
                        break;
                    }
                    case 2: // TRMM: Compute L^{-1} * L_offdiag (stored back in factor tile)
                    {
                        // inv (index1) = diagonal L^{-1} (DENSE)
                        // fact (index2) = off-diagonal L (ACTIVE COLUMNS format, sa columns)
                        double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                        double* fact = tiledMatrix->chunkedDenseTiles[index2];
                        if (!inv || !fact) break;

                        const sTiles::TileMetaCore& meta_inv = tiledMatrix->tileMetaCore[index1];
                        const sTiles::TileMetaCore& meta_fact = tiledMatrix->tileMetaCore[index2];
                        const sTiles::SemisparseTileMetaCore& semi_fact = tiledMatrix->semisparseTileMetaCore[index2];

                        const int m = meta_fact.height;  // rows of off-diag tile
                        const int sa = semi_fact.sa;     // number of active columns

                        if (sa > 0 && !semi_case2_banded_solve(tiledMatrix, index1, m, sa, fact)) {
                            // TRMM: fact = inv * fact (operates on sa columns)
                            sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                            sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                            m, sa, zone, inv, meta_inv.height, fact, m);
                        }
                        break;
                    }
                    case 3: // Barrier
                    {
                        #pragma omp barrier
                        global_in = static_cast<int>(in) + 1;
                        goto exit_phase1_semi_dense_omp;
                    }
                    default:
                        break;
                }
            }

        exit_phase1_semi_dense_omp:
            if (global_in < 0) global_in = static_cast<int>(start);

            // ========== Phase 2: Compute selected inverse entries ==========
            for (size_t in = static_cast<size_t>(global_in); in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int i = t[1];
                const int j = t[2];
                const int k = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                switch (myroutine) {
                    case 4: // LAUUM + mirror: Compute L^{-T} * L^{-1} for diagonal tile
                    {
                        double* inv = tiledMatrix->chunkedInverseTiles[index1];
                        if (!inv) break;

                        const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                        const int n = meta.height;

                        sTiles::core_dlauum(sTiles::Uplo::Upper, n, inv, n);
                        mirroring(n, inv, inv, n);
                        break;
                    }
                    case 5: // GEMM: Diagonal tile update: inv1 -= (L^{-1}*L_offdiag) * inv2^T
                    {
                        // index1 = diagonal inverse tile (DENSE)
                        // index2 = off-diagonal tile: factor has ACTIVE COLS, inverse is DENSE
                        dep_wait_for(i, k, 2);

                        double* fact = tiledMatrix->chunkedDenseTiles[index2];   // L^{-1}*L_offdiag in ACTIVE COLS
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2]; // DENSE
                        double* inv1 = tiledMatrix->chunkedInverseTiles[index1]; // DENSE (output)
                        if (!fact || !inv2 || !inv1) break;

                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::TileMetaCore& meta1 = tiledMatrix->tileMetaCore[index1];
                        const sTiles::TileMetaCore& meta2 = tiledMatrix->tileMetaCore[index2];

                        const int m = meta2.height;      // rows
                        const int n = meta2.width;       // cols of dense inverse tile
                        const int sa = semi.sa;          // active columns in factor
                        if (sa <= 0) break;

                        // Gather columns from inv2 at active column indices
                        // gathered = inv2[:, aind[0]], inv2[:, aind[1]], ... (m x sa)
                        for (int jj = 0; jj < sa && jj < (int)semi.aind.size(); ++jj) {
                            const int col = semi.aind[jj];
                            if (col >= 0 && col < n) {
                                for (int ii = 0; ii < m; ++ii) {
                                    workspace[ii + jj * m] = inv2[ii + col * m];
                                }
                            } else {
                                for (int ii = 0; ii < m; ++ii) {
                                    workspace[ii + jj * m] = 0.0;
                                }
                            }
                        }

                        // GEMM: inv1 -= fact * gathered^T  =>  (m x m) -= (m x sa) * (sa x m)
                        sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                        m, m, sa, mzone,
                                        fact, m,
                                        workspace, m,
                                        zone, inv1, meta1.height);
                        break;
                    }
                    case 6:
                        dep_set_done(i, i, 2);
                        break;

                    case 7: // GEMM: Off-diagonal update: inv3 -= fact * inv2^T
                    {
                        // index1 = off-diagonal factor (ACTIVE COLS)
                        // index2 = off-diagonal inverse (DENSE)
                        // index3 = off-diagonal inverse output (DENSE)
                        dep_wait_for(j, k, 2);

                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                        if (!fact || !inv2 || !inv3) break;

                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::TileMetaCore& meta1 = tiledMatrix->tileMetaCore[index1];
                        const sTiles::TileMetaCore& meta2 = tiledMatrix->tileMetaCore[index2];
                        const sTiles::TileMetaCore& meta3 = tiledMatrix->tileMetaCore[index3];

                        const int m_fact = meta1.height;
                        const int m_inv2 = meta2.height;
                        const int n_inv2 = meta2.width;
                        const int sa = semi.sa;
                        if (sa <= 0) break;

                        // Gather columns from inv2 at active column indices
                        // gathered = inv2[:, aind[]] (m_inv2 x sa)
                        for (int jj = 0; jj < sa && jj < (int)semi.aind.size(); ++jj) {
                            const int col = semi.aind[jj];
                            if (col >= 0 && col < n_inv2) {
                                for (int ii = 0; ii < m_inv2; ++ii) {
                                    workspace[ii + jj * m_inv2] = inv2[ii + col * m_inv2];
                                }
                            } else {
                                for (int ii = 0; ii < m_inv2; ++ii) {
                                    workspace[ii + jj * m_inv2] = 0.0;
                                }
                            }
                        }

                        // GEMM: inv3 -= fact * gathered^T  =>  (m_fact x m_inv2) -= (m_fact x sa) * (sa x m_inv2)
                        sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                        m_fact, m_inv2, sa, mzone,
                                        fact, m_fact,
                                        workspace, m_inv2,
                                        zone, inv3, meta3.height);
                        break;
                    }
                    case 8: // GEMM: Off-diagonal update: inv3 -= fact * inv2 (NoTrans)
                    {
                        // index1 = (i,k) off-diagonal factor (ACTIVE COLS)
                        // index2 = (k,j) inverse tile (DENSE)
                        // index3 = (i,j) off-diagonal inverse output (DENSE)
                        dep_wait_for(k, j, 2);

                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                        if (!fact || !inv2 || !inv3) break;

                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::TileMetaCore& meta1 = tiledMatrix->tileMetaCore[index1];
                        const sTiles::TileMetaCore& meta2 = tiledMatrix->tileMetaCore[index2];
                        const sTiles::TileMetaCore& meta3 = tiledMatrix->tileMetaCore[index3];

                        const int m_fact = meta1.height;
                        const int m_inv2 = meta2.height;
                        const int n_inv2 = meta2.width;
                        const int sa = semi.sa;
                        if (sa <= 0) break;

                        // For fact * inv2 with ACTIVE COLS format:
                        // fact column j corresponds to original column aind[j]
                        // So we need rows aind[] of inv2
                        // gathered[i, :] = inv2[aind[i], :] (sa x n_inv2)
                        for (int ii = 0; ii < sa && ii < (int)semi.aind.size(); ++ii) {
                            const int row = semi.aind[ii];
                            if (row >= 0 && row < m_inv2) {
                                for (int jj = 0; jj < n_inv2; ++jj) {
                                    workspace[ii + jj * sa] = inv2[row + jj * m_inv2];
                                }
                            } else {
                                for (int jj = 0; jj < n_inv2; ++jj) {
                                    workspace[ii + jj * sa] = 0.0;
                                }
                            }
                        }

                        // GEMM: inv3 -= fact * gathered  =>  (m_fact x n_inv2) -= (m_fact x sa) * (sa x n_inv2)
                        sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                        m_fact, n_inv2, sa, mzone,
                                        fact, m_fact,
                                        workspace, sa,
                                        zone, inv3, meta3.height);
                        break;
                    }
                    case 9:
                        dep_set_done(i, j, 2);
                        break;

                    default:
                        break;
                }
            }

            dep_finalize();
        }

        static void pthreads_dpotri_semi_sparse_serial(TiledMatrix *tiledMatrix, stiles_context_t *stile)
        {
            // stile used for the singular-tile abort (case 1) below.
            const int rank = STILES_RANK;
            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[trtri_selinv_imp1_serial] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
            // Dedicated scratch for the no-gather-info fallback's column map (item 1):
            // it used to alias just past the B region of tmp_tile, overflowing the
            // tile_size^2 workspace when sa1==sa3==tile_size (fully-active tiles, the
            // has_gather_info==false giant-matrix regime). sa <= tile_size always.
            std::vector<int> col_map_scratch(static_cast<std::size_t>(tiledMatrix->tile_size));

            // Precomputed gather info (may be null if not built)
            const int32_t* gather_index = (tiledMatrix->inv_gather_index) ? tiledMatrix->inv_gather_index->data() : nullptr;
            const int32_t* gather_packed = (tiledMatrix->inv_gather_packed) ? tiledMatrix->inv_gather_packed->data() : nullptr;
            const bool has_gather_info = (gather_index != nullptr && gather_packed != nullptr
                && tiledMatrix->inv_gather_index && static_cast<int>(tiledMatrix->inv_gather_index->size()) >= static_cast<int>(tiledMatrix->inv_tasks ? tiledMatrix->inv_tasks->size() : 0) * 3);

            const auto &tasks = sTiles::get_inv_tasks(tiledMatrix);
            const size_t start = 0, end = tasks.size();

            // Hoisted base pointers + bounds for the per-iteration prefetch block.
            double** const cdt = tiledMatrix->chunkedDenseTiles;
            double** const cit = tiledMatrix->chunkedInverseTiles;
            const auto* const ssm = tiledMatrix->semisparseTileMetaCore;
            const int ntiles_bounds = tiledMatrix->numActiveTiles;

            // Serial: single pass through all tasks — no barrier, no sync
            for (size_t in = start; in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int i = t[1];
                const int j = t[2];
                const int k = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                // Prefetch next task's tile/meta backing storage. The gather_packed
                // indirection is the unpredictable load that benefits most.
                if (in + 1 < end) {
                    const auto &nt = tasks[in + 1];
                    const int n1 = nt[4], n2 = nt[5], n3 = nt[6];
                    if (n1 >= 0 && n1 < ntiles_bounds) {
                        if (cdt[n1]) __builtin_prefetch(cdt[n1], 0, 2);
                        if (cit[n1]) __builtin_prefetch(cit[n1], 1, 2);
                        __builtin_prefetch(&ssm[n1], 0, 3);
                    }
                    if (n2 >= 0 && n2 < ntiles_bounds) {
                        if (cdt[n2]) __builtin_prefetch(cdt[n2], 0, 2);
                        if (cit[n2]) __builtin_prefetch(cit[n2], 0, 2);
                        __builtin_prefetch(&ssm[n2], 0, 3);
                    }
                    if (n3 >= 0 && n3 < ntiles_bounds && cit[n3]) {
                        __builtin_prefetch(cit[n3], 1, 2);
                    }
                    if (has_gather_info) {
                        const int gi = static_cast<int>(in + 1) * 3;
                        __builtin_prefetch(&gather_index[gi], 0, 3);
                        const int next_off = gather_index[gi];
                        if (next_off >= 0) __builtin_prefetch(gather_packed + next_off, 0, 3);
                    }
                }

                switch (myroutine) {
                    case 1: // TRSM
                    {
                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                        if (!fact || !inv) { break; }

                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const int n = tiledMatrix->tileMetaCore[index1].height;
                        const int kd = semi.upper_bw;

                        if (const int bad = semi_diag_singular(n, kd, fact)) {
                            std::fprintf(stderr, "sTiles error: semi selinv singular diagonal tile idx=%d (local row %d).\n", index1, bad - 1);
                            stile->ss_abort = 1;   // -> ExecutionFailed (dpotri.cpp); break (not return) avoids the phase-1 barrier deadlock
                            break;
                        }
                        if (kd == 0) {
                            std::memset(inv, 0, static_cast<std::size_t>(n) * n * sizeof(double));
                            for (int ii = 0; ii < n; ++ii)
                                inv[ii + ii * n] = 1.0 / fact[ii];
                        } else {
                            banded_diag_inverse(n, kd, fact, inv);
                        }
                        break;
                    }
                    case 2: // TRMM
                    {
                        double* inv = tiledMatrix->chunkedInverseTiles[index1];
                        double* fact = tiledMatrix->chunkedDenseTiles[index2];
                        if (!inv || !fact) { break; }

                        const int h_inv = tiledMatrix->tileMetaCore[index1].height;
                        const sTiles::SemisparseTileMetaCore& semi_inv = tiledMatrix->semisparseTileMetaCore[index1];
                        const int m = tiledMatrix->tileMetaCore[index2].height;
                        const int sa = tiledMatrix->semisparseTileMetaCore[index2].sa;
                        if (sa <= 0) { break; }

                        if (semi_inv.upper_bw == 0) {
                            double* diag = tmp_tile;
                            for (int r = 0; r < m; ++r)
                                diag[r] = inv[r + r * m];
                            for (int cc = 0; cc < sa; ++cc) {
                                double* col = fact + cc * m;
                                #pragma omp simd
                                for (int r = 0; r < m; ++r) {
                                    col[r] *= diag[r];
                                }
                            }
                        } else if (!semi_case2_banded_solve(tiledMatrix, index1, m, sa, fact)) {
                            sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                            sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                            m, sa, 1.0, inv, h_inv, fact, m);
                        }
                        break;
                    }
                    case 3: // Barrier — no-op in serial
                    case 6: // Signal — no-op in serial
                    case 9: // Signal — no-op in serial
                        break;
                    case 4: // LAUUM
                    {
                        double* inv = tiledMatrix->chunkedInverseTiles[index1];
                        if (!inv) { break; }

                        const int n = tiledMatrix->tileMetaCore[index1].height;
                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];

                        if (semi.upper_bw == 0) {
                            for (int ii = 0; ii < n; ++ii)
                                inv[ii + ii * n] *= inv[ii + ii * n];
                        } else {
                            sTiles::core_dlauum(sTiles::Uplo::Upper, n, inv, n);
                            mirroring(n, inv, inv, n);
                        }
                        break;
                    }
                    case 5: // Diagonal tile update
                    {
                        double* fact = tiledMatrix->chunkedDenseTiles[index2];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv1 = tiledMatrix->chunkedInverseTiles[index1];
                        if (!fact || !inv2 || !inv1) { break; }

                        const int m = tiledMatrix->tileMetaCore[index2].height;
                        const int sa2 = tiledMatrix->semisparseTileMetaCore[index2].sa;
                        const int n1 = tiledMatrix->tileMetaCore[index1].height;

                        if (sa2 <= 0) { break; }

                        const int kd1 = tiledMatrix->semisparseTileMetaCore[index1].upper_bw;
                        if (kd1 == 0) {
                            for (int kk = 0; kk < sa2; ++kk) {
                                const double* f_col = fact + kk * m;
                                const double* i_col = inv2 + kk * m;
                                for (int r = 0; r < n1; ++r) {
                                    inv1[r + r * n1] -= f_col[r] * i_col[r];
                                }
                            }
                        } else {
                            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                        n1, n1, sa2, -1.0,
                                        fact, m, inv2, m,
                                        1.0, inv1, n1);
                        }
                        break;
                    }
                    case 7: // GEMM: inv3 -= fact * inv2^T
                    {
                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                        if (!fact || !inv2 || !inv3) { break; }

                        const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                        const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                        const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                        const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                        const int sa1 = semi1.sa;
                        const int sa3 = semi3.sa;

                        const int* aind3 = semi3.aind.data();
                        constexpr int FUSE_THRESH = 8;

                        // ── Fused gather+compute for small tiles ──
                        if (sa1 <= FUSE_THRESH && sa3 <= FUSE_THRESH && has_gather_info) {
                            const int gi_base = static_cast<int>(in) * 3;
                            const int data_off = gather_index[gi_base];
                            const int n_valid  = gather_index[gi_base + 1];
                            const int flags    = gather_index[gi_base + 2];

                            if (flags == 0) { break; }

                            if (flags == 1) {
                                const int32_t* col_offsets = gather_packed + data_off;
                                for (int jj = 0; jj < n_valid; ++jj) {
                                    const int col_off = col_offsets[jj];
                                    const double* fact_col = fact + jj * m_fact;
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const double b_val = inv2[aind3[cc] + col_off];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            } else {
                                const int32_t* pr = gather_packed + data_off;
                                for (int v = 0; v < n_valid; ++v) {
                                    const int jj     = pr[v * 2];
                                    const int offset = pr[v * 2 + 1];
                                    const double* fact_col = fact + jj * m_fact;
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const double b_val = inv2[aind3[cc] + offset];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            }
                            break;
                        }

                        // ── Standard gather + BLAS path ──
                        double* B = tmp_tile;

                        if (has_gather_info) {
                            const int gi_base = static_cast<int>(in) * 3;
                            const int data_off = gather_index[gi_base];
                            const int n_valid  = gather_index[gi_base + 1];
                            const int flags    = gather_index[gi_base + 2];

                            if (flags == 0) {
                                break;
                            }
                            if (flags == 1) {
                                const int32_t* col_offsets = gather_packed + data_off;
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const int c_row = aind3[cc];
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < n_valid; ++jj) {
                                        B_dst[jj] = inv2[c_row + col_offsets[jj]];
                                    }
                                }
                            } else {
                                const int32_t* pairs = gather_packed + data_off;
                                std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const int c_row = aind3[cc];
                                    double* B_dst = B + cc * sa1;
                                    for (int v = 0; v < n_valid; ++v) {
                                        B_dst[pairs[v * 2]] = inv2[c_row + pairs[v * 2 + 1]];
                                    }
                                }
                            }
                        } else {
                            const int* aind1 = semi1.aind.data();
                            const int sa2 = semi2.sa;
                            const int* acol2 = semi2.acol.data();
                            const int acol2_sz = static_cast<int>(semi2.acol.size());

                            int* col_map = col_map_scratch.data();
                            int valid_count = 0;
                            for (int jj = 0; jj < sa1; ++jj) {
                                const int kk = aind1[jj];
                                const int idx = (kk >= 0 && kk < acol2_sz) ? acol2[kk] : -1;
                                col_map[jj] = (idx >= 0 && idx < sa2) ? idx : -1;
                                if (col_map[jj] >= 0) ++valid_count;
                            }
                            if (valid_count == 0) {
                                break;
                            }
                            if (valid_count == sa1) {
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const int c_row = aind3[cc];
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj)
                                        B_dst[jj] = inv2[c_row + col_map[jj] * m_inv2];
                                }
                            } else {
                                std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const int c_row = aind3[cc];
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj) {
                                        const int stored = col_map[jj];
                                        if (stored >= 0) B_dst[jj] = inv2[c_row + stored * m_inv2];
                                    }
                                }
                            }
                        }

                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                        break;
                    }
                    case 8: // GEMM: inv3 -= fact * inv2
                    {
                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                        if (!fact || !inv2 || !inv3) { break; }

                        const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                        const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                        const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                        const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                        const int sa1 = semi1.sa;
                        const int sa3 = semi3.sa;

                        if (sa1 <= 0 || sa3 <= 0) { break; }

                        const int* aind1 = semi1.aind.data();
                        const int* aind3 = semi3.aind.data();
                        constexpr int FUSE_THRESH = 8;

                        // ── Fused gather+compute for small tiles ──
                        if (sa1 <= FUSE_THRESH && sa3 <= FUSE_THRESH && has_gather_info) {
                            const int gi_base = static_cast<int>(in) * 3;
                            const int data_off = gather_index[gi_base];
                            const int n_valid  = gather_index[gi_base + 1];
                            const int flags    = gather_index[gi_base + 2];

                            if (flags == 0) { break; }

                            if (flags == 3) {
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const double* fact_col = fact + jj * m_fact;
                                    const int a1 = aind1[jj];
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const double b_val = inv2[aind3[cc] * m_inv2 + a1];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            } else if (flags == 1) {
                                const int32_t* col_offsets = gather_packed + data_off;
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const double* fact_col = fact + jj * m_fact;
                                    const int a1 = aind1[jj];
                                    for (int cc = 0; cc < n_valid; ++cc) {
                                        const double b_val = inv2[col_offsets[cc] + a1];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            } else {
                                const int32_t* pr = gather_packed + data_off;
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const double* fact_col = fact + jj * m_fact;
                                    const int a1 = aind1[jj];
                                    for (int v = 0; v < n_valid; ++v) {
                                        const int cc     = pr[v * 2];
                                        const double b_val = inv2[pr[v * 2 + 1] + a1];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            }
                            break;
                        }

                        // ── Standard gather + BLAS path ──
                        double* B = tmp_tile;

                        if (has_gather_info) {
                            const int gi_base = static_cast<int>(in) * 3;
                            const int data_off = gather_index[gi_base];
                            const int n_valid  = gather_index[gi_base + 1];
                            const int flags    = gather_index[gi_base + 2];

                            if (flags == 0) {
                                std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                            } else if (flags == 3) {
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const double* inv2_col = inv2 + aind3[cc] * m_inv2;
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj)
                                        B_dst[jj] = inv2_col[aind1[jj]];
                                }
                            } else if (flags == 1) {
                                const int32_t* col_offsets = gather_packed + data_off;
                                for (int cc = 0; cc < n_valid; ++cc) {
                                    const double* inv2_col = inv2 + col_offsets[cc];
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj)
                                        B_dst[jj] = inv2_col[aind1[jj]];
                                }
                            } else {
                                const int32_t* pairs = gather_packed + data_off;
                                std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                for (int v = 0; v < n_valid; ++v) {
                                    const double* inv2_col = inv2 + pairs[v * 2 + 1];
                                    double* B_dst = B + pairs[v * 2] * sa1;
                                    for (int jj = 0; jj < sa1; ++jj)
                                        B_dst[jj] = inv2_col[aind1[jj]];
                                }
                            }
                        } else {
                            const bool inv2_is_diag = (tiledMatrix->tileMetaCore[index2].row == tiledMatrix->tileMetaCore[index2].col);
                            if (inv2_is_diag) {
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const double* inv2_col = inv2 + aind3[cc] * m_inv2;
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj)
                                        B_dst[jj] = inv2_col[aind1[jj]];
                                }
                            } else {
                                const int sa2 = semi2.sa;
                                const int* acol2 = semi2.acol.data();
                                const int acol2_sz = static_cast<int>(semi2.acol.size());
                                int* col_map = col_map_scratch.data();
                                int valid_count = 0;
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const int c = aind3[cc];
                                    const int idx = (c >= 0 && c < acol2_sz) ? acol2[c] : -1;
                                    col_map[cc] = (idx >= 0 && idx < sa2) ? idx : -1;
                                    if (col_map[cc] >= 0) ++valid_count;
                                }
                                if (valid_count == 0) {
                                    std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                } else {
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int stored = col_map[cc];
                                        if (stored >= 0) {
                                            const double* inv2_col = inv2 + stored * m_inv2;
                                            double* B_dst = B + cc * sa1;
                                            for (int jj = 0; jj < sa1; ++jj)
                                                B_dst[jj] = inv2_col[aind1[jj]];
                                        } else {
                                            std::memset(B + cc * sa1, 0, sa1 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        }

                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                        break;
                    }
                    default:
                        break;
                }
            }
        }

        static void pthreads_dpotri_semi_sparse_parallel(TiledMatrix *tiledMatrix, stiles_context_t *stile)
        {
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            // Use pre-allocated workspace for B matrix in cases 7/8, and diagonal extraction in case 2
            const int rank = STILES_RANK;
            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[trtri_selinv_imp1] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
            // Dedicated scratch for the no-gather-info fallback's column map (item 1):
            // it used to alias just past the B region of tmp_tile, overflowing the
            // tile_size^2 workspace when sa1==sa3==tile_size (fully-active tiles, the
            // has_gather_info==false giant-matrix regime). sa <= tile_size always.
            std::vector<int> col_map_scratch(static_cast<std::size_t>(tiledMatrix->tile_size));

            // Precomputed gather info (may be null if not built)
            const int32_t* gather_index = (tiledMatrix->inv_gather_index) ? tiledMatrix->inv_gather_index->data() : nullptr;
            const int32_t* gather_packed = (tiledMatrix->inv_gather_packed) ? tiledMatrix->inv_gather_packed->data() : nullptr;
            const bool has_gather_info = (gather_index != nullptr && gather_packed != nullptr
                && tiledMatrix->inv_gather_index && static_cast<int>(tiledMatrix->inv_gather_index->size()) >= static_cast<int>(tiledMatrix->inv_tasks ? tiledMatrix->inv_tasks->size() : 0) * 3);

            const auto &tasks   = sTiles::get_inv_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_inv_task_offsets(tiledMatrix);
            size_t start = 0, end = tasks.size();
            if (!offsets.empty()) {
                const size_t r = static_cast<size_t>(STILES_RANK);
                if (r < offsets.size()) start = static_cast<size_t>(offsets[r]);
                if (r + 1 < offsets.size()) end = static_cast<size_t>(offsets[r + 1]);
            }

            in_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            int global_in = -1;

            constexpr int FUSE_THRESH = 8;

            // ========== Phase 1: Compute L^{-1} for diagonal tiles ==========
            for (size_t in = start; in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int index1 = t[4];
                const int index2 = t[5];

                switch (myroutine) {
                    case 1: // TRSM: Compute L^{-1} for diagonal tile
                    {
                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                        if (!fact || !inv) { break; }

                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                        const int n = meta.height;
                        const int kd = semi.upper_bw;

                        if (const int bad = semi_diag_singular(n, kd, fact)) {
                            std::fprintf(stderr, "sTiles error: semi selinv singular diagonal tile idx=%d (local row %d).\n", index1, bad - 1);
                            stile->ss_abort = 1;   // -> ExecutionFailed (dpotri.cpp); break (not return) avoids the phase-1 barrier deadlock
                            break;
                        }
                        if (kd == 0) {
                            // Pure diagonal L: L^{-1} = diag(1/L[i])
                            std::memset(inv, 0, static_cast<std::size_t>(n) * n * sizeof(double));
                            for (int ii = 0; ii < n; ++ii)
                                inv[ii + ii * n] = 1.0 / fact[ii];
                        } else {
                            // Banded triangular solve: L * X = I => X = L^{-1}
                            banded_diag_inverse(n, kd, fact, inv);
                        }
                        break;
                    }
                    case 2: // TRMM: Compute L^{-1} * L_offdiag (stored back in factor tile)
                    {
                        double* inv = tiledMatrix->chunkedInverseTiles[index1];
                        double* fact = tiledMatrix->chunkedDenseTiles[index2];
                        if (!inv || !fact) { break; }

                        const int h_inv = tiledMatrix->tileMetaCore[index1].height;
                        const sTiles::SemisparseTileMetaCore& semi_inv = tiledMatrix->semisparseTileMetaCore[index1];
                        const int m = tiledMatrix->tileMetaCore[index2].height;
                        const int sa = tiledMatrix->semisparseTileMetaCore[index2].sa;
                        if (sa <= 0) { break; }

                        if (semi_inv.upper_bw == 0) {
                            // L^{-1} is diagonal: TRMM = row scaling
                            // Extract diagonal into contiguous array to avoid stride-(m+1) access
                            double* diag = tmp_tile;
                            for (int r = 0; r < m; ++r)
                                diag[r] = inv[r + r * m];
                            for (int cc = 0; cc < sa; ++cc) {
                                double* col = fact + cc * m;
                                #pragma omp simd
                                for (int r = 0; r < m; ++r) {
                                    col[r] *= diag[r];
                                }
                            }
                        } else if (!semi_case2_banded_solve(tiledMatrix, index1, m, sa, fact)) {
                            sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                            sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                            m, sa, 1.0, inv, h_inv, fact, m);
                        }
                        break;
                    }
                    case 3: // Barrier
                    {
                        sTiles::Control::Barrier(stile);
                        global_in = static_cast<int>(in) + 1;
                        goto exit_phase1_semisparse_selinv;
                    }
                    default:
                        break;
                }
            }

        exit_phase1_semisparse_selinv:
            if (global_in < 0) global_in = static_cast<int>(start);

            // Hoisted base pointers + bounds for the per-iteration prefetch block.
            double** const cdt = tiledMatrix->chunkedDenseTiles;
            double** const cit = tiledMatrix->chunkedInverseTiles;
            const auto* const ssm = tiledMatrix->semisparseTileMetaCore;
            const int ntiles_bounds = tiledMatrix->numActiveTiles;

            // Cross-iteration B-cache for cases 7/8 BLAS path. When consecutive tasks
            // share (route, index2, index3), the gathered B in tmp_tile is identical
            // and the gather can be skipped. Fused path doesn't write tmp_tile so it
            // doesn't disturb the cache. Only enabled with has_gather_info: the
            // no-gather-info fallback writes col_map past the B region, which would
            // corrupt a cached B from a prior iteration.
            int B_cache_route = -1, B_cache_idx2 = -1, B_cache_idx3 = -1;
            bool B_cached = false;
            long B_cache_hits = 0, B_cache_misses = 0;
            (void)B_cache_hits; (void)B_cache_misses;

            // ========== Phase 2: Compute selective inverse entries ==========
            for (size_t in = static_cast<size_t>(global_in); in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int i = t[1];
                const int j = t[2];
                const int k = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                // Prefetch next task's tile/meta backing storage. The gather_packed
                // indirection is the unpredictable load that benefits most. Prefetch is
                // a hint only — safe under the cond_wait sync protocol below.
                if (in + 1 < end) {
                    const auto &nt = tasks[in + 1];
                    const int n1 = nt[4], n2 = nt[5], n3 = nt[6];
                    if (n1 >= 0 && n1 < ntiles_bounds) {
                        if (cdt[n1]) __builtin_prefetch(cdt[n1], 0, 2);
                        if (cit[n1]) __builtin_prefetch(cit[n1], 1, 2);
                        __builtin_prefetch(&ssm[n1], 0, 3);
                    }
                    if (n2 >= 0 && n2 < ntiles_bounds) {
                        if (cdt[n2]) __builtin_prefetch(cdt[n2], 0, 2);
                        if (cit[n2]) __builtin_prefetch(cit[n2], 0, 2);
                        __builtin_prefetch(&ssm[n2], 0, 3);
                    }
                    if (n3 >= 0 && n3 < ntiles_bounds && cit[n3]) {
                        __builtin_prefetch(cit[n3], 1, 2);
                    }
                    if (has_gather_info) {
                        const int gi = static_cast<int>(in + 1) * 3;
                        __builtin_prefetch(&gather_index[gi], 0, 3);
                        const int next_off = gather_index[gi];
                        if (next_off >= 0) __builtin_prefetch(gather_packed + next_off, 0, 3);
                    }
                }

                switch (myroutine) {
                    case 7: // GEMM: Off-diagonal update: inv3 -= fact * inv2^T
                    {
                        in_cond_wait(j, k, 2);

                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                        if (!fact || !inv2 || !inv3) { break; }

                        const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                        const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                        const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                        const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                        const int sa1 = semi1.sa;
                        const int sa3 = semi3.sa;

                        const int* aind3 = semi3.aind.data();

                        // ── Fused gather+compute for small tiles ──
                        if (sa1 <= FUSE_THRESH && sa3 <= FUSE_THRESH && has_gather_info) {
                            const int gi_base = static_cast<int>(in) * 3;
                            const int data_off = gather_index[gi_base];
                            const int n_valid  = gather_index[gi_base + 1];
                            const int flags    = gather_index[gi_base + 2];

                            if (flags == 0) { break; }

                            if (flags == 1) {
                                const int32_t* col_offsets = gather_packed + data_off;
                                for (int jj = 0; jj < n_valid; ++jj) {
                                    const int col_off = col_offsets[jj];
                                    const double* fact_col = fact + jj * m_fact;
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const double b_val = inv2[aind3[cc] + col_off];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            } else {
                                const int32_t* pr = gather_packed + data_off;
                                for (int v = 0; v < n_valid; ++v) {
                                    const int jj     = pr[v * 2];
                                    const int offset = pr[v * 2 + 1];
                                    const double* fact_col = fact + jj * m_fact;
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const double b_val = inv2[aind3[cc] + offset];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            }
                            break;
                        }

                        // ── Standard gather + BLAS path ──
                        double* B = tmp_tile;

                        const bool can_reuse_B = has_gather_info && B_cached
                            && B_cache_route == 7
                            && B_cache_idx2 == index2
                            && B_cache_idx3 == index3;

                        if (can_reuse_B) {
                            ++B_cache_hits;
                        } else {
                            ++B_cache_misses;
                            if (has_gather_info) {
                                const int gi_base = static_cast<int>(in) * 3;
                                const int data_off = gather_index[gi_base];
                                const int n_valid  = gather_index[gi_base + 1];
                                const int flags    = gather_index[gi_base + 2];

                                if (flags == 0) {
                                    break;
                                }

                                if (flags == 1) {
                                    const int32_t* col_offsets = gather_packed + data_off;
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c_row = aind3[cc];
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < n_valid; ++jj) {
                                            B_dst[jj] = inv2[c_row + col_offsets[jj]];
                                        }
                                    }
                                } else {
                                    const int32_t* pairs = gather_packed + data_off;
                                    std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c_row = aind3[cc];
                                        double* B_dst = B + cc * sa1;
                                        for (int v = 0; v < n_valid; ++v) {
                                            const int jj     = pairs[v * 2];
                                            const int offset = pairs[v * 2 + 1];
                                            B_dst[jj] = inv2[c_row + offset];
                                        }
                                    }
                                }
                            } else {
                                const int* aind1 = semi1.aind.data();
                                const int sa2 = semi2.sa;
                                const int* acol2 = semi2.acol.data();
                                const int acol2_sz = static_cast<int>(semi2.acol.size());

                                int* col_map = col_map_scratch.data();
                                int valid_count = 0;
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const int k = aind1[jj];
                                    const int idx = (k >= 0 && k < acol2_sz) ? acol2[k] : -1;
                                    col_map[jj] = (idx >= 0 && idx < sa2) ? idx : -1;
                                    if (col_map[jj] >= 0) ++valid_count;
                                }
                                if (valid_count == 0) {
                                    B_cached = false; // col_map writes corrupted prior cache
                                    break;
                                }
                                if (valid_count == sa1) {
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c_row = aind3[cc];
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < sa1; ++jj) {
                                            B_dst[jj] = inv2[c_row + col_map[jj] * m_inv2];
                                        }
                                    }
                                } else {
                                    std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c_row = aind3[cc];
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < sa1; ++jj) {
                                            const int stored = col_map[jj];
                                            if (stored >= 0) B_dst[jj] = inv2[c_row + stored * m_inv2];
                                        }
                                    }
                                }
                            }
                        }

                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                        if (has_gather_info) {
                            B_cache_route = 7;
                            B_cache_idx2  = index2;
                            B_cache_idx3  = index3;
                            B_cached      = true;
                        } else {
                            B_cached = false; // col_map fallback aliases tmp_tile region
                        }

                        break;
                    }
                    case 8: // GEMM: Off-diagonal update: inv3 -= fact * inv2
                    {
                        in_cond_wait(k, j, 2);

                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                        if (!fact || !inv2 || !inv3) { break; }

                        const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                        const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                        const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                        const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                        const int sa1 = semi1.sa;
                        const int sa3 = semi3.sa;

                        if (sa1 <= 0 || sa3 <= 0) { break; }

                        const int* aind1 = semi1.aind.data();
                        const int* aind3 = semi3.aind.data();

                        // ── Fused gather+compute for small tiles ──
                        if (sa1 <= FUSE_THRESH && sa3 <= FUSE_THRESH && has_gather_info) {
                            const int gi_base = static_cast<int>(in) * 3;
                            const int data_off = gather_index[gi_base];
                            const int n_valid  = gather_index[gi_base + 1];
                            const int flags    = gather_index[gi_base + 2];

                            if (flags == 0) { break; } // no overlap, inv3 unchanged

                            if (flags == 3) {
                                // diagonal inv2
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const double* fact_col = fact + jj * m_fact;
                                    const int a1 = aind1[jj];
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const double b_val = inv2[aind3[cc] * m_inv2 + a1];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            } else if (flags == 1) {
                                // all_valid
                                const int32_t* col_offsets = gather_packed + data_off;
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const double* fact_col = fact + jj * m_fact;
                                    const int a1 = aind1[jj];
                                    for (int cc = 0; cc < n_valid; ++cc) {
                                        const double b_val = inv2[col_offsets[cc] + a1];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            } else {
                                // partial
                                const int32_t* pr = gather_packed + data_off;
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const double* fact_col = fact + jj * m_fact;
                                    const int a1 = aind1[jj];
                                    for (int v = 0; v < n_valid; ++v) {
                                        const int cc     = pr[v * 2];
                                        const double b_val = inv2[pr[v * 2 + 1] + a1];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            }
                            break;
                        }

                        // ── Standard gather + BLAS path ──
                        double* B = tmp_tile;

                        const bool can_reuse_B = has_gather_info && B_cached
                            && B_cache_route == 8
                            && B_cache_idx2 == index2
                            && B_cache_idx3 == index3;

                        if (can_reuse_B) {
                            ++B_cache_hits;
                        } else {
                            ++B_cache_misses;
                            if (has_gather_info) {
                                const int gi_base = static_cast<int>(in) * 3;
                                const int data_off = gather_index[gi_base];
                                const int n_valid  = gather_index[gi_base + 1];
                                const int flags    = gather_index[gi_base + 2];

                                if (flags == 0) {
                                    std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                } else if (flags == 3) {
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c_col = aind3[cc];
                                        const double* inv2_col = inv2 + c_col * m_inv2;
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < sa1; ++jj) {
                                            B_dst[jj] = inv2_col[aind1[jj]];
                                        }
                                    }
                                } else if (flags == 1) {
                                    const int32_t* col_offsets = gather_packed + data_off;
                                    for (int cc = 0; cc < n_valid; ++cc) {
                                        const double* inv2_col = inv2 + col_offsets[cc];
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < sa1; ++jj) {
                                            B_dst[jj] = inv2_col[aind1[jj]];
                                        }
                                    }
                                } else {
                                    const int32_t* pairs = gather_packed + data_off;
                                    std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                    for (int v = 0; v < n_valid; ++v) {
                                        const int cc     = pairs[v * 2];
                                        const int offset = pairs[v * 2 + 1];
                                        const double* inv2_col = inv2 + offset;
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < sa1; ++jj) {
                                            B_dst[jj] = inv2_col[aind1[jj]];
                                        }
                                    }
                                }
                            } else {
                                const bool inv2_is_diag = (tiledMatrix->tileMetaCore[index2].row == tiledMatrix->tileMetaCore[index2].col);

                                if (inv2_is_diag) {
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c_col = aind3[cc];
                                        const double* inv2_col = inv2 + c_col * m_inv2;
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < sa1; ++jj) {
                                            B_dst[jj] = inv2_col[aind1[jj]];
                                        }
                                    }
                                } else {
                                    const int sa2 = semi2.sa;
                                    const int* acol2 = semi2.acol.data();
                                    const int acol2_sz = static_cast<int>(semi2.acol.size());

                                    int* col_map = col_map_scratch.data();
                                    int valid_count = 0;
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c = aind3[cc];
                                        const int idx = (c >= 0 && c < acol2_sz) ? acol2[c] : -1;
                                        col_map[cc] = (idx >= 0 && idx < sa2) ? idx : -1;
                                        if (col_map[cc] >= 0) ++valid_count;
                                    }
                                    if (valid_count == 0) { std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double)); }
                                    else {
                                        for (int cc = 0; cc < sa3; ++cc) {
                                            const int stored = col_map[cc];
                                            if (stored >= 0) {
                                                const double* inv2_col = inv2 + stored * m_inv2;
                                                double* B_dst = B + cc * sa1;
                                                for (int jj = 0; jj < sa1; ++jj) {
                                                    B_dst[jj] = inv2_col[aind1[jj]];
                                                }
                                            } else {
                                                std::memset(B + cc * sa1, 0, sa1 * sizeof(double));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                        if (has_gather_info) {
                            B_cache_route = 8;
                            B_cache_idx2  = index2;
                            B_cache_idx3  = index3;
                            B_cached      = true;
                        } else {
                            B_cached = false; // col_map fallback aliases tmp_tile region
                        }

                        break;
                    }
                    case 4: // LAUUM + mirror: Compute L^{-T} * L^{-1} for diagonal tile
                    {
                        double* inv = tiledMatrix->chunkedInverseTiles[index1];
                        if (!inv) { break; }

                        const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const int n = meta.height;

                        if (semi.upper_bw == 0) {
                            for (int ii = 0; ii < n; ++ii)
                                inv[ii + ii * n] *= inv[ii + ii * n];
                        } else {
                            sTiles::core_dlauum(sTiles::Uplo::Upper, n, inv, n);
                            mirroring(n, inv, inv, n);
                        }
                        break;
                    }
                    case 5: // Diagonal tile update: inv1 -= fact * inv2^T
                    {
                        in_cond_wait(i, k, 2);

                        double* fact = tiledMatrix->chunkedDenseTiles[index2];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv1 = tiledMatrix->chunkedInverseTiles[index1];
                        if (!fact || !inv2 || !inv1) { break; }

                        const int m = tiledMatrix->tileMetaCore[index2].height;
                        const int sa2 = tiledMatrix->semisparseTileMetaCore[index2].sa;
                        const int n1 = tiledMatrix->tileMetaCore[index1].height;

                        if (sa2 <= 0) { break; }

                        const int kd1 = tiledMatrix->semisparseTileMetaCore[index1].upper_bw;
                        if (kd1 == 0) {
                            for (int kk = 0; kk < sa2; ++kk) {
                                const double* f_col = fact + kk * m;
                                const double* i_col = inv2 + kk * m;
                                for (int r = 0; r < n1; ++r) {
                                    inv1[r + r * n1] -= f_col[r] * i_col[r];
                                }
                            }
                        } else {
                            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                        n1, n1, sa2, -1.0,
                                        fact, m,
                                        inv2, m,
                                        1.0, inv1, n1);
                        }
                        break;
                    }
                    case 6:
                        in_cond_set(i, i, 2);
                        break;
                    case 9:
                        in_cond_set(i, j, 2);
                        break;

                    default:
                        break;
                }
            }

            if (std::getenv("STILES_BCACHE_STATS"))
                std::fprintf(stderr, "[bcache] pthreads rank %d hits=%ld misses=%ld\n",
                             rank, B_cache_hits, B_cache_misses);

            in_finalize();
        }

        static void omp_dpotri_semi_sparse_serial(TiledMatrix *tiledMatrix, omp_dep_tracker_t* dep_tracker)
        {
            (void)dep_tracker; // dep_tracker used for the singular-tile abort below
            const int rank = omp_get_thread_num();
            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[omp_dpotri_semi_sparse_serial] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
            // Dedicated scratch for the no-gather-info fallback's column map (item 1):
            // it used to alias just past the B region of tmp_tile, overflowing the
            // tile_size^2 workspace when sa1==sa3==tile_size (fully-active tiles, the
            // has_gather_info==false giant-matrix regime). sa <= tile_size always.
            std::vector<int> col_map_scratch(static_cast<std::size_t>(tiledMatrix->tile_size));

            // Precomputed gather info (may be null if not built)
            const int32_t* gather_index = (tiledMatrix->inv_gather_index) ? tiledMatrix->inv_gather_index->data() : nullptr;
            const int32_t* gather_packed = (tiledMatrix->inv_gather_packed) ? tiledMatrix->inv_gather_packed->data() : nullptr;
            const bool has_gather_info = (gather_index != nullptr && gather_packed != nullptr
                && tiledMatrix->inv_gather_index && static_cast<int>(tiledMatrix->inv_gather_index->size()) >= static_cast<int>(tiledMatrix->inv_tasks ? tiledMatrix->inv_tasks->size() : 0) * 3);

            const auto &tasks = sTiles::get_inv_tasks(tiledMatrix);
            const size_t start = 0, end = tasks.size();

            // Hoisted base pointers + bounds for the per-iteration prefetch block.
            double** const cdt = tiledMatrix->chunkedDenseTiles;
            double** const cit = tiledMatrix->chunkedInverseTiles;
            const auto* const ssm = tiledMatrix->semisparseTileMetaCore;
            const int ntiles_bounds = tiledMatrix->numActiveTiles;

            // Serial: single pass through all tasks — no barrier, no sync
            for (size_t in = start; in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int i = t[1];
                const int j = t[2];
                const int k = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                // Prefetch next task's tile/meta backing storage. The gather_packed
                // indirection is the unpredictable load that benefits most.
                if (in + 1 < end) {
                    const auto &nt = tasks[in + 1];
                    const int n1 = nt[4], n2 = nt[5], n3 = nt[6];
                    if (n1 >= 0 && n1 < ntiles_bounds) {
                        if (cdt[n1]) __builtin_prefetch(cdt[n1], 0, 2);
                        if (cit[n1]) __builtin_prefetch(cit[n1], 1, 2);
                        __builtin_prefetch(&ssm[n1], 0, 3);
                    }
                    if (n2 >= 0 && n2 < ntiles_bounds) {
                        if (cdt[n2]) __builtin_prefetch(cdt[n2], 0, 2);
                        if (cit[n2]) __builtin_prefetch(cit[n2], 0, 2);
                        __builtin_prefetch(&ssm[n2], 0, 3);
                    }
                    if (n3 >= 0 && n3 < ntiles_bounds && cit[n3]) {
                        __builtin_prefetch(cit[n3], 1, 2);
                    }
                    if (has_gather_info) {
                        const int gi = static_cast<int>(in + 1) * 3;
                        __builtin_prefetch(&gather_index[gi], 0, 3);
                        const int next_off = gather_index[gi];
                        if (next_off >= 0) __builtin_prefetch(gather_packed + next_off, 0, 3);
                    }
                }

                switch (myroutine) {
                    case 1: // TRSM
                    {
                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                        if (!fact || !inv) { break; }

                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const int n = tiledMatrix->tileMetaCore[index1].height;
                        const int kd = semi.upper_bw;

                        if (const int bad = semi_diag_singular(n, kd, fact)) {
                            std::fprintf(stderr, "sTiles error: semi selinv singular diagonal tile idx=%d (local row %d).\n", index1, bad - 1);
                            dep_tracker->abort_flag.store(true, std::memory_order_release);   // -> ExecutionFailed; break (not return) avoids the phase-1 barrier deadlock
                            break;
                        }
                        if (kd == 0) {
                            std::memset(inv, 0, static_cast<std::size_t>(n) * n * sizeof(double));
                            for (int ii = 0; ii < n; ++ii)
                                inv[ii + ii * n] = 1.0 / fact[ii];
                        } else {
                            banded_diag_inverse(n, kd, fact, inv);
                        }
                        break;
                    }
                    case 2: // TRMM
                    {
                        double* inv = tiledMatrix->chunkedInverseTiles[index1];
                        double* fact = tiledMatrix->chunkedDenseTiles[index2];
                        if (!inv || !fact) { break; }

                        const int h_inv = tiledMatrix->tileMetaCore[index1].height;
                        const sTiles::SemisparseTileMetaCore& semi_inv = tiledMatrix->semisparseTileMetaCore[index1];
                        const int m = tiledMatrix->tileMetaCore[index2].height;
                        const int sa = tiledMatrix->semisparseTileMetaCore[index2].sa;
                        if (sa <= 0) { break; }

                        if (semi_inv.upper_bw == 0) {
                            double* diag = tmp_tile;
                            for (int r = 0; r < m; ++r)
                                diag[r] = inv[r + r * m];
                            for (int cc = 0; cc < sa; ++cc) {
                                double* col = fact + cc * m;
                                #pragma omp simd
                                for (int r = 0; r < m; ++r) {
                                    col[r] *= diag[r];
                                }
                            }
                        } else if (!semi_case2_banded_solve(tiledMatrix, index1, m, sa, fact)) {
                            sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                            sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                            m, sa, 1.0, inv, h_inv, fact, m);
                        }
                        break;
                    }
                    case 3: // Barrier — no-op in serial
                    case 6: // Signal — no-op in serial
                    case 9: // Signal — no-op in serial
                        break;
                    case 4: // LAUUM
                    {
                        double* inv = tiledMatrix->chunkedInverseTiles[index1];
                        if (!inv) { break; }

                        const int n = tiledMatrix->tileMetaCore[index1].height;
                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];

                        if (semi.upper_bw == 0) {
                            for (int ii = 0; ii < n; ++ii)
                                inv[ii + ii * n] *= inv[ii + ii * n];
                        } else {
                            sTiles::core_dlauum(sTiles::Uplo::Upper, n, inv, n);
                            mirroring(n, inv, inv, n);
                        }
                        break;
                    }
                    case 5: // Diagonal tile update
                    {
                        double* fact = tiledMatrix->chunkedDenseTiles[index2];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv1 = tiledMatrix->chunkedInverseTiles[index1];
                        if (!fact || !inv2 || !inv1) { break; }

                        const int m = tiledMatrix->tileMetaCore[index2].height;
                        const int sa2 = tiledMatrix->semisparseTileMetaCore[index2].sa;
                        const int n1 = tiledMatrix->tileMetaCore[index1].height;

                        if (sa2 <= 0) { break; }

                        const int kd1 = tiledMatrix->semisparseTileMetaCore[index1].upper_bw;
                        if (kd1 == 0) {
                            for (int kk = 0; kk < sa2; ++kk) {
                                const double* f_col = fact + kk * m;
                                const double* i_col = inv2 + kk * m;
                                for (int r = 0; r < n1; ++r) {
                                    inv1[r + r * n1] -= f_col[r] * i_col[r];
                                }
                            }
                        } else {
                            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                        n1, n1, sa2, -1.0,
                                        fact, m, inv2, m,
                                        1.0, inv1, n1);
                        }
                        break;
                    }
                    case 7: // GEMM: inv3 -= fact * inv2^T
                    {
                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                        if (!fact || !inv2 || !inv3) { break; }

                        const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                        const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                        const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                        const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                        const int sa1 = semi1.sa;
                        const int sa3 = semi3.sa;

                        const int* aind3 = semi3.aind.data();
                        constexpr int FUSE_THRESH = 8;

                        // ── Fused gather+compute for small tiles ──
                        if (sa1 <= FUSE_THRESH && sa3 <= FUSE_THRESH && has_gather_info) {
                            const int gi_base = static_cast<int>(in) * 3;
                            const int data_off = gather_index[gi_base];
                            const int n_valid  = gather_index[gi_base + 1];
                            const int flags    = gather_index[gi_base + 2];

                            if (flags == 0) { break; }

                            if (flags == 1) {
                                const int32_t* col_offsets = gather_packed + data_off;
                                for (int jj = 0; jj < n_valid; ++jj) {
                                    const int col_off = col_offsets[jj];
                                    const double* fact_col = fact + jj * m_fact;
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const double b_val = inv2[aind3[cc] + col_off];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            } else {
                                const int32_t* pr = gather_packed + data_off;
                                for (int v = 0; v < n_valid; ++v) {
                                    const int jj     = pr[v * 2];
                                    const int offset = pr[v * 2 + 1];
                                    const double* fact_col = fact + jj * m_fact;
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const double b_val = inv2[aind3[cc] + offset];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            }
                            break;
                        }

                        // ── Standard gather + BLAS path ──
                        double* B = tmp_tile;

                        if (has_gather_info) {
                            const int gi_base = static_cast<int>(in) * 3;
                            const int data_off = gather_index[gi_base];
                            const int n_valid  = gather_index[gi_base + 1];
                            const int flags    = gather_index[gi_base + 2];

                            if (flags == 0) {
                                break;
                            }
                            if (flags == 1) {
                                const int32_t* col_offsets = gather_packed + data_off;
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const int c_row = aind3[cc];
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < n_valid; ++jj) {
                                        B_dst[jj] = inv2[c_row + col_offsets[jj]];
                                    }
                                }
                            } else {
                                const int32_t* pairs = gather_packed + data_off;
                                std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const int c_row = aind3[cc];
                                    double* B_dst = B + cc * sa1;
                                    for (int v = 0; v < n_valid; ++v) {
                                        B_dst[pairs[v * 2]] = inv2[c_row + pairs[v * 2 + 1]];
                                    }
                                }
                            }
                        } else {
                            const int* aind1 = semi1.aind.data();
                            const int sa2 = semi2.sa;
                            const int* acol2 = semi2.acol.data();
                            const int acol2_sz = static_cast<int>(semi2.acol.size());

                            int* col_map = col_map_scratch.data();
                            int valid_count = 0;
                            for (int jj = 0; jj < sa1; ++jj) {
                                const int kk = aind1[jj];
                                const int idx = (kk >= 0 && kk < acol2_sz) ? acol2[kk] : -1;
                                col_map[jj] = (idx >= 0 && idx < sa2) ? idx : -1;
                                if (col_map[jj] >= 0) ++valid_count;
                            }
                            if (valid_count == 0) {
                                break;
                            }
                            if (valid_count == sa1) {
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const int c_row = aind3[cc];
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj)
                                        B_dst[jj] = inv2[c_row + col_map[jj] * m_inv2];
                                }
                            } else {
                                std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const int c_row = aind3[cc];
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj) {
                                        const int stored = col_map[jj];
                                        if (stored >= 0) B_dst[jj] = inv2[c_row + stored * m_inv2];
                                    }
                                }
                            }
                        }

                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                        break;
                    }
                    case 8: // GEMM: inv3 -= fact * inv2
                    {
                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                        if (!fact || !inv2 || !inv3) { break; }

                        const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                        const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                        const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                        const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                        const int sa1 = semi1.sa;
                        const int sa3 = semi3.sa;

                        if (sa1 <= 0 || sa3 <= 0) { break; }

                        const int* aind1 = semi1.aind.data();
                        const int* aind3 = semi3.aind.data();
                        constexpr int FUSE_THRESH = 8;

                        // ── Fused gather+compute for small tiles ──
                        if (sa1 <= FUSE_THRESH && sa3 <= FUSE_THRESH && has_gather_info) {
                            const int gi_base = static_cast<int>(in) * 3;
                            const int data_off = gather_index[gi_base];
                            const int n_valid  = gather_index[gi_base + 1];
                            const int flags    = gather_index[gi_base + 2];

                            if (flags == 0) { break; }

                            if (flags == 3) {
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const double* fact_col = fact + jj * m_fact;
                                    const int a1 = aind1[jj];
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const double b_val = inv2[aind3[cc] * m_inv2 + a1];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            } else if (flags == 1) {
                                const int32_t* col_offsets = gather_packed + data_off;
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const double* fact_col = fact + jj * m_fact;
                                    const int a1 = aind1[jj];
                                    for (int cc = 0; cc < n_valid; ++cc) {
                                        const double b_val = inv2[col_offsets[cc] + a1];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            } else {
                                const int32_t* pr = gather_packed + data_off;
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const double* fact_col = fact + jj * m_fact;
                                    const int a1 = aind1[jj];
                                    for (int v = 0; v < n_valid; ++v) {
                                        const int cc     = pr[v * 2];
                                        const double b_val = inv2[pr[v * 2 + 1] + a1];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            }
                            break;
                        }

                        // ── Standard gather + BLAS path ──
                        double* B = tmp_tile;

                        if (has_gather_info) {
                            const int gi_base = static_cast<int>(in) * 3;
                            const int data_off = gather_index[gi_base];
                            const int n_valid  = gather_index[gi_base + 1];
                            const int flags    = gather_index[gi_base + 2];

                            if (flags == 0) {
                                std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                            } else if (flags == 3) {
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const double* inv2_col = inv2 + aind3[cc] * m_inv2;
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj)
                                        B_dst[jj] = inv2_col[aind1[jj]];
                                }
                            } else if (flags == 1) {
                                const int32_t* col_offsets = gather_packed + data_off;
                                for (int cc = 0; cc < n_valid; ++cc) {
                                    const double* inv2_col = inv2 + col_offsets[cc];
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj)
                                        B_dst[jj] = inv2_col[aind1[jj]];
                                }
                            } else {
                                const int32_t* pairs = gather_packed + data_off;
                                std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                for (int v = 0; v < n_valid; ++v) {
                                    const double* inv2_col = inv2 + pairs[v * 2 + 1];
                                    double* B_dst = B + pairs[v * 2] * sa1;
                                    for (int jj = 0; jj < sa1; ++jj)
                                        B_dst[jj] = inv2_col[aind1[jj]];
                                }
                            }
                        } else {
                            const bool inv2_is_diag = (tiledMatrix->tileMetaCore[index2].row == tiledMatrix->tileMetaCore[index2].col);
                            if (inv2_is_diag) {
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const double* inv2_col = inv2 + aind3[cc] * m_inv2;
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj)
                                        B_dst[jj] = inv2_col[aind1[jj]];
                                }
                            } else {
                                const int sa2 = semi2.sa;
                                const int* acol2 = semi2.acol.data();
                                const int acol2_sz = static_cast<int>(semi2.acol.size());
                                int* col_map = col_map_scratch.data();
                                int valid_count = 0;
                                for (int cc = 0; cc < sa3; ++cc) {
                                    const int c = aind3[cc];
                                    const int idx = (c >= 0 && c < acol2_sz) ? acol2[c] : -1;
                                    col_map[cc] = (idx >= 0 && idx < sa2) ? idx : -1;
                                    if (col_map[cc] >= 0) ++valid_count;
                                }
                                if (valid_count == 0) {
                                    std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                } else {
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int stored = col_map[cc];
                                        if (stored >= 0) {
                                            const double* inv2_col = inv2 + stored * m_inv2;
                                            double* B_dst = B + cc * sa1;
                                            for (int jj = 0; jj < sa1; ++jj)
                                                B_dst[jj] = inv2_col[aind1[jj]];
                                        } else {
                                            std::memset(B + cc * sa1, 0, sa1 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        }

                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                        break;
                    }
                    default:
                        break;
                }
            }
        }

        static void omp_dpotri_semi_sparse_parallel(TiledMatrix *tiledMatrix, omp_dep_tracker_t* dep_tracker, int worldsize)
        {
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            // Use pre-allocated workspace for B matrix in cases 7/8, and diagonal extraction in case 2
            const int rank = omp_get_thread_num();
            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[omp_dpotri_semi_sparse_parallel] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
            // Dedicated scratch for the no-gather-info fallback's column map (item 1):
            // it used to alias just past the B region of tmp_tile, overflowing the
            // tile_size^2 workspace when sa1==sa3==tile_size (fully-active tiles, the
            // has_gather_info==false giant-matrix regime). sa <= tile_size always.
            std::vector<int> col_map_scratch(static_cast<std::size_t>(tiledMatrix->tile_size));

            // Precomputed gather info (may be null if not built)
            const int32_t* gather_index = (tiledMatrix->inv_gather_index) ? tiledMatrix->inv_gather_index->data() : nullptr;
            const int32_t* gather_packed = (tiledMatrix->inv_gather_packed) ? tiledMatrix->inv_gather_packed->data() : nullptr;
            const bool has_gather_info = (gather_index != nullptr && gather_packed != nullptr
                && tiledMatrix->inv_gather_index && static_cast<int>(tiledMatrix->inv_gather_index->size()) >= static_cast<int>(tiledMatrix->inv_tasks ? tiledMatrix->inv_tasks->size() : 0) * 3);

            const auto &tasks   = sTiles::get_inv_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_inv_task_offsets(tiledMatrix);
            size_t start = 0, end = tasks.size();
            if (!offsets.empty()) {
                const size_t r = static_cast<size_t>(rank);
                if (r < offsets.size()) start = static_cast<size_t>(offsets[r]);
                if (r + 1 < offsets.size()) end = static_cast<size_t>(offsets[r + 1]);
            }

            dep_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            int global_in = -1;

            constexpr int FUSE_THRESH = 8;

            // ========== Phase 1: Compute L^{-1} for diagonal tiles ==========
            for (size_t in = start; in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int index1 = t[4];
                const int index2 = t[5];

                switch (myroutine) {
                    case 1: // TRSM: Compute L^{-1} for diagonal tile
                    {
                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                        if (!fact || !inv) { break; }

                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                        const int n = meta.height;
                        const int kd = semi.upper_bw;

                        if (const int bad = semi_diag_singular(n, kd, fact)) {
                            std::fprintf(stderr, "sTiles error: semi selinv singular diagonal tile idx=%d (local row %d).\n", index1, bad - 1);
                            dep_tracker->abort_flag.store(true, std::memory_order_release);   // -> ExecutionFailed; break (not return) avoids the phase-1 barrier deadlock
                            break;
                        }
                        if (kd == 0) {
                            // Pure diagonal L: L^{-1} = diag(1/L[i])
                            std::memset(inv, 0, static_cast<std::size_t>(n) * n * sizeof(double));
                            for (int ii = 0; ii < n; ++ii)
                                inv[ii + ii * n] = 1.0 / fact[ii];
                        } else {
                            // Banded triangular solve: L * X = I => X = L^{-1}
                            banded_diag_inverse(n, kd, fact, inv);
                        }
                        break;
                    }
                    case 2: // TRMM: Compute L^{-1} * L_offdiag (stored back in factor tile)
                    {
                        double* inv = tiledMatrix->chunkedInverseTiles[index1];
                        double* fact = tiledMatrix->chunkedDenseTiles[index2];
                        if (!inv || !fact) { break; }

                        const int h_inv = tiledMatrix->tileMetaCore[index1].height;
                        const sTiles::SemisparseTileMetaCore& semi_inv = tiledMatrix->semisparseTileMetaCore[index1];
                        const int m = tiledMatrix->tileMetaCore[index2].height;
                        const int sa = tiledMatrix->semisparseTileMetaCore[index2].sa;
                        if (sa <= 0) { break; }

                        if (semi_inv.upper_bw == 0) {
                            // L^{-1} is diagonal: TRMM = row scaling
                            // Extract diagonal into contiguous array to avoid stride-(m+1) access
                            double* diag = tmp_tile;
                            for (int r = 0; r < m; ++r)
                                diag[r] = inv[r + r * m];
                            for (int cc = 0; cc < sa; ++cc) {
                                double* col = fact + cc * m;
                                #pragma omp simd
                                for (int r = 0; r < m; ++r) {
                                    col[r] *= diag[r];
                                }
                            }
                        } else if (!semi_case2_banded_solve(tiledMatrix, index1, m, sa, fact)) {
                            sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                            sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                            m, sa, 1.0, inv, h_inv, fact, m);
                        }
                        break;
                    }
                    case 3: // Barrier
                    {
                        #pragma omp barrier
                        global_in = static_cast<int>(in) + 1;
                        goto exit_phase1_semisparse_selinv;
                    }
                    default:
                        break;
                }
            }

        exit_phase1_semisparse_selinv:
            if (global_in < 0) global_in = static_cast<int>(start);

            // Hoisted base pointers + bounds for the per-iteration prefetch block.
            double** const cdt = tiledMatrix->chunkedDenseTiles;
            double** const cit = tiledMatrix->chunkedInverseTiles;
            const auto* const ssm = tiledMatrix->semisparseTileMetaCore;
            const int ntiles_bounds = tiledMatrix->numActiveTiles;

            // Cross-iteration B-cache for cases 7/8 BLAS path. When consecutive tasks
            // share (route, index2, index3), the gathered B in tmp_tile is identical
            // and the gather can be skipped. Fused path doesn't write tmp_tile so it
            // doesn't disturb the cache. Only enabled with has_gather_info: the
            // no-gather-info fallback writes col_map past the B region, which would
            // corrupt a cached B from a prior iteration.
            int B_cache_route = -1, B_cache_idx2 = -1, B_cache_idx3 = -1;
            bool B_cached = false;
            long B_cache_hits = 0, B_cache_misses = 0;
            (void)B_cache_hits; (void)B_cache_misses;

            // ========== Phase 2: Compute selective inverse entries ==========
            for (size_t in = static_cast<size_t>(global_in); in < end; ++in) {
                const auto &t = tasks[in];
                const int myroutine = t[0];
                const int i = t[1];
                const int j = t[2];
                const int k = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                // Prefetch next task's tile/meta backing storage. The gather_packed
                // indirection is the unpredictable load that benefits most. Prefetch is
                // a hint only — safe under the cond_wait sync protocol below.
                if (in + 1 < end) {
                    const auto &nt = tasks[in + 1];
                    const int n1 = nt[4], n2 = nt[5], n3 = nt[6];
                    if (n1 >= 0 && n1 < ntiles_bounds) {
                        if (cdt[n1]) __builtin_prefetch(cdt[n1], 0, 2);
                        if (cit[n1]) __builtin_prefetch(cit[n1], 1, 2);
                        __builtin_prefetch(&ssm[n1], 0, 3);
                    }
                    if (n2 >= 0 && n2 < ntiles_bounds) {
                        if (cdt[n2]) __builtin_prefetch(cdt[n2], 0, 2);
                        if (cit[n2]) __builtin_prefetch(cit[n2], 0, 2);
                        __builtin_prefetch(&ssm[n2], 0, 3);
                    }
                    if (n3 >= 0 && n3 < ntiles_bounds && cit[n3]) {
                        __builtin_prefetch(cit[n3], 1, 2);
                    }
                    if (has_gather_info) {
                        const int gi = static_cast<int>(in + 1) * 3;
                        __builtin_prefetch(&gather_index[gi], 0, 3);
                        const int next_off = gather_index[gi];
                        if (next_off >= 0) __builtin_prefetch(gather_packed + next_off, 0, 3);
                    }
                }

                switch (myroutine) {
                    case 7: // GEMM: Off-diagonal update: inv3 -= fact * inv2^T
                    {
                        dep_wait_for(j, k, 2);

                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                        if (!fact || !inv2 || !inv3) { break; }

                        const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                        const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                        const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                        const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                        const int sa1 = semi1.sa;
                        const int sa3 = semi3.sa;

                        const int* aind3 = semi3.aind.data();

                        // ── Fused gather+compute for small tiles ──
                        if (sa1 <= FUSE_THRESH && sa3 <= FUSE_THRESH && has_gather_info) {
                            const int gi_base = static_cast<int>(in) * 3;
                            const int data_off = gather_index[gi_base];
                            const int n_valid  = gather_index[gi_base + 1];
                            const int flags    = gather_index[gi_base + 2];

                            if (flags == 0) { break; }

                            if (flags == 1) {
                                const int32_t* col_offsets = gather_packed + data_off;
                                for (int jj = 0; jj < n_valid; ++jj) {
                                    const int col_off = col_offsets[jj];
                                    const double* fact_col = fact + jj * m_fact;
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const double b_val = inv2[aind3[cc] + col_off];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            } else {
                                const int32_t* pr = gather_packed + data_off;
                                for (int v = 0; v < n_valid; ++v) {
                                    const int jj     = pr[v * 2];
                                    const int offset = pr[v * 2 + 1];
                                    const double* fact_col = fact + jj * m_fact;
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const double b_val = inv2[aind3[cc] + offset];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            }
                            break;
                        }

                        // ── Standard gather + BLAS path ──
                        double* B = tmp_tile;

                        const bool can_reuse_B = has_gather_info && B_cached
                            && B_cache_route == 7
                            && B_cache_idx2 == index2
                            && B_cache_idx3 == index3;

                        if (can_reuse_B) {
                            ++B_cache_hits;
                        } else {
                            ++B_cache_misses;
                            if (has_gather_info) {
                                const int gi_base = static_cast<int>(in) * 3;
                                const int data_off = gather_index[gi_base];
                                const int n_valid  = gather_index[gi_base + 1];
                                const int flags    = gather_index[gi_base + 2];

                                if (flags == 0) {
                                    break;
                                }

                                if (flags == 1) {
                                    const int32_t* col_offsets = gather_packed + data_off;
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c_row = aind3[cc];
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < n_valid; ++jj) {
                                            B_dst[jj] = inv2[c_row + col_offsets[jj]];
                                        }
                                    }
                                } else {
                                    const int32_t* pairs = gather_packed + data_off;
                                    std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c_row = aind3[cc];
                                        double* B_dst = B + cc * sa1;
                                        for (int v = 0; v < n_valid; ++v) {
                                            const int jj     = pairs[v * 2];
                                            const int offset = pairs[v * 2 + 1];
                                            B_dst[jj] = inv2[c_row + offset];
                                        }
                                    }
                                }
                            } else {
                                const int* aind1 = semi1.aind.data();
                                const int sa2 = semi2.sa;
                                const int* acol2 = semi2.acol.data();
                                const int acol2_sz = static_cast<int>(semi2.acol.size());

                                int* col_map = col_map_scratch.data();
                                int valid_count = 0;
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const int k = aind1[jj];
                                    const int idx = (k >= 0 && k < acol2_sz) ? acol2[k] : -1;
                                    col_map[jj] = (idx >= 0 && idx < sa2) ? idx : -1;
                                    if (col_map[jj] >= 0) ++valid_count;
                                }
                                if (valid_count == 0) {
                                    B_cached = false; // col_map writes corrupted prior cache
                                    break;
                                }
                                if (valid_count == sa1) {
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c_row = aind3[cc];
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < sa1; ++jj) {
                                            B_dst[jj] = inv2[c_row + col_map[jj] * m_inv2];
                                        }
                                    }
                                } else {
                                    std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c_row = aind3[cc];
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < sa1; ++jj) {
                                            const int stored = col_map[jj];
                                            if (stored >= 0) B_dst[jj] = inv2[c_row + stored * m_inv2];
                                        }
                                    }
                                }
                            }
                        }

                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                        if (has_gather_info) {
                            B_cache_route = 7;
                            B_cache_idx2  = index2;
                            B_cache_idx3  = index3;
                            B_cached      = true;
                        } else {
                            B_cached = false; // col_map fallback aliases tmp_tile region
                        }

                        break;
                    }
                    case 8: // GEMM: Off-diagonal update: inv3 -= fact * inv2
                    {
                        dep_wait_for(k, j, 2);

                        double* fact = tiledMatrix->chunkedDenseTiles[index1];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                        if (!fact || !inv2 || !inv3) { break; }

                        const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                        const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                        const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                        const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                        const int sa1 = semi1.sa;
                        const int sa3 = semi3.sa;

                        if (sa1 <= 0 || sa3 <= 0) { break; }

                        const int* aind1 = semi1.aind.data();
                        const int* aind3 = semi3.aind.data();

                        // ── Fused gather+compute for small tiles ──
                        if (sa1 <= FUSE_THRESH && sa3 <= FUSE_THRESH && has_gather_info) {
                            const int gi_base = static_cast<int>(in) * 3;
                            const int data_off = gather_index[gi_base];
                            const int n_valid  = gather_index[gi_base + 1];
                            const int flags    = gather_index[gi_base + 2];

                            if (flags == 0) { break; } // no overlap, inv3 unchanged

                            if (flags == 3) {
                                // diagonal inv2
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const double* fact_col = fact + jj * m_fact;
                                    const int a1 = aind1[jj];
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const double b_val = inv2[aind3[cc] * m_inv2 + a1];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            } else if (flags == 1) {
                                // all_valid
                                const int32_t* col_offsets = gather_packed + data_off;
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const double* fact_col = fact + jj * m_fact;
                                    const int a1 = aind1[jj];
                                    for (int cc = 0; cc < n_valid; ++cc) {
                                        const double b_val = inv2[col_offsets[cc] + a1];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            } else {
                                // partial
                                const int32_t* pr = gather_packed + data_off;
                                for (int jj = 0; jj < sa1; ++jj) {
                                    const double* fact_col = fact + jj * m_fact;
                                    const int a1 = aind1[jj];
                                    for (int v = 0; v < n_valid; ++v) {
                                        const int cc     = pr[v * 2];
                                        const double b_val = inv2[pr[v * 2 + 1] + a1];
                                        double* inv3_col = inv3 + cc * m_inv3;
                                        #pragma omp simd
                                        for (int r = 0; r < m_inv3; ++r)
                                            inv3_col[r] -= fact_col[r] * b_val;
                                    }
                                }
                            }
                            break;
                        }

                        // ── Standard gather + BLAS path ──
                        double* B = tmp_tile;

                        const bool can_reuse_B = has_gather_info && B_cached
                            && B_cache_route == 8
                            && B_cache_idx2 == index2
                            && B_cache_idx3 == index3;

                        if (can_reuse_B) {
                            ++B_cache_hits;
                        } else {
                            ++B_cache_misses;
                            if (has_gather_info) {
                                const int gi_base = static_cast<int>(in) * 3;
                                const int data_off = gather_index[gi_base];
                                const int n_valid  = gather_index[gi_base + 1];
                                const int flags    = gather_index[gi_base + 2];

                                if (flags == 0) {
                                    std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                } else if (flags == 3) {
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c_col = aind3[cc];
                                        const double* inv2_col = inv2 + c_col * m_inv2;
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < sa1; ++jj) {
                                            B_dst[jj] = inv2_col[aind1[jj]];
                                        }
                                    }
                                } else if (flags == 1) {
                                    const int32_t* col_offsets = gather_packed + data_off;
                                    for (int cc = 0; cc < n_valid; ++cc) {
                                        const double* inv2_col = inv2 + col_offsets[cc];
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < sa1; ++jj) {
                                            B_dst[jj] = inv2_col[aind1[jj]];
                                        }
                                    }
                                } else {
                                    const int32_t* pairs = gather_packed + data_off;
                                    std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                                    for (int v = 0; v < n_valid; ++v) {
                                        const int cc     = pairs[v * 2];
                                        const int offset = pairs[v * 2 + 1];
                                        const double* inv2_col = inv2 + offset;
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < sa1; ++jj) {
                                            B_dst[jj] = inv2_col[aind1[jj]];
                                        }
                                    }
                                }
                            } else {
                                const bool inv2_is_diag = (tiledMatrix->tileMetaCore[index2].row == tiledMatrix->tileMetaCore[index2].col);

                                if (inv2_is_diag) {
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c_col = aind3[cc];
                                        const double* inv2_col = inv2 + c_col * m_inv2;
                                        double* B_dst = B + cc * sa1;
                                        for (int jj = 0; jj < sa1; ++jj) {
                                            B_dst[jj] = inv2_col[aind1[jj]];
                                        }
                                    }
                                } else {
                                    const int sa2 = semi2.sa;
                                    const int* acol2 = semi2.acol.data();
                                    const int acol2_sz = static_cast<int>(semi2.acol.size());

                                    int* col_map = col_map_scratch.data();
                                    int valid_count = 0;
                                    for (int cc = 0; cc < sa3; ++cc) {
                                        const int c = aind3[cc];
                                        const int idx = (c >= 0 && c < acol2_sz) ? acol2[c] : -1;
                                        col_map[cc] = (idx >= 0 && idx < sa2) ? idx : -1;
                                        if (col_map[cc] >= 0) ++valid_count;
                                    }
                                    if (valid_count == 0) { std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double)); }
                                    else {
                                        for (int cc = 0; cc < sa3; ++cc) {
                                            const int stored = col_map[cc];
                                            if (stored >= 0) {
                                                const double* inv2_col = inv2 + stored * m_inv2;
                                                double* B_dst = B + cc * sa1;
                                                for (int jj = 0; jj < sa1; ++jj) {
                                                    B_dst[jj] = inv2_col[aind1[jj]];
                                                }
                                            } else {
                                                std::memset(B + cc * sa1, 0, sa1 * sizeof(double));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                        if (has_gather_info) {
                            B_cache_route = 8;
                            B_cache_idx2  = index2;
                            B_cache_idx3  = index3;
                            B_cached      = true;
                        } else {
                            B_cached = false; // col_map fallback aliases tmp_tile region
                        }

                        break;
                    }
                    case 4: // LAUUM + mirror: Compute L^{-T} * L^{-1} for diagonal tile
                    {
                        double* inv = tiledMatrix->chunkedInverseTiles[index1];
                        if (!inv) { break; }

                        const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const int n = meta.height;

                        if (semi.upper_bw == 0) {
                            for (int ii = 0; ii < n; ++ii)
                                inv[ii + ii * n] *= inv[ii + ii * n];
                        } else {
                            sTiles::core_dlauum(sTiles::Uplo::Upper, n, inv, n);
                            mirroring(n, inv, inv, n);
                        }
                        break;
                    }
                    case 5: // Diagonal tile update: inv1 -= fact * inv2^T
                    {
                        dep_wait_for(i, k, 2);

                        double* fact = tiledMatrix->chunkedDenseTiles[index2];
                        double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                        double* inv1 = tiledMatrix->chunkedInverseTiles[index1];
                        if (!fact || !inv2 || !inv1) { break; }

                        const int m = tiledMatrix->tileMetaCore[index2].height;
                        const int sa2 = tiledMatrix->semisparseTileMetaCore[index2].sa;
                        const int n1 = tiledMatrix->tileMetaCore[index1].height;

                        if (sa2 <= 0) { break; }

                        const int kd1 = tiledMatrix->semisparseTileMetaCore[index1].upper_bw;
                        if (kd1 == 0) {
                            for (int kk = 0; kk < sa2; ++kk) {
                                const double* f_col = fact + kk * m;
                                const double* i_col = inv2 + kk * m;
                                for (int r = 0; r < n1; ++r) {
                                    inv1[r + r * n1] -= f_col[r] * i_col[r];
                                }
                            }
                        } else {
                            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                        n1, n1, sa2, -1.0,
                                        fact, m,
                                        inv2, m,
                                        1.0, inv1, n1);
                        }
                        break;
                    }
                    case 6:
                        dep_set_done(i, i, 2);
                        break;
                    case 9:
                        dep_set_done(i, j, 2);
                        break;

                    default:
                        break;
                }
            }

            if (std::getenv("STILES_BCACHE_STATS"))
                std::fprintf(stderr, "[bcache] omp rank %d hits=%ld misses=%ld\n",
                             rank, B_cache_hits, B_cache_misses);

            dep_finalize();
        }

    }

}


// =============================================================================
// OMP Version of Selected Inversion (omp_pdtrtri)
// =============================================================================
namespace sTiles { namespace Process {

/**
 * @brief Main OMP dispatch function for selected inversion.
 *
 * This is called from omp_dpotri in dpotri.cpp within an OMP parallel region.
 * Routes to appropriate OMP implementation based on tile type and variant.
 */
void omp_pdtrtri(TiledMatrix* tiledMatrix, omp_dep_tracker_t* dep_tracker, int worldsize) {

    static int* stiles_control_params = sTiles_get_params();

    if (!tiledMatrix) {
        #pragma omp single
        std::cout << "Error: null tiledMatrix in omp_pdtrtri" << std::endl;
        return;
    }

    const int tile_type_mode = stiles_control_params[3];
    const int variant = tiledMatrix->factorization_variant;
    const int num_active = tiledMatrix->numActiveTiles;
    const int dim_tiled = tiledMatrix->dimTiledMatrix;

    // Variant 1 or single tile: Direct inversion
    if (variant == 1 || (num_active == 1 && dim_tiled == 1)) {
        omp_dpotri_full_dense(tiledMatrix, dep_tracker, 1, worldsize);
        return;
    }

    // Check if we have pre-collected inversion tasks
    const auto &tasks = sTiles::get_inv_tasks(tiledMatrix);
    if (tasks.empty()) {
        // Fallback to variant 2 dense approach if no tasks
        if (tiledMatrix->denseTiles && tiledMatrix->inverseTiles) {
            omp_dpotri_full_dense(tiledMatrix, dep_tracker, 2, worldsize);
        }
        return;
    }

    // Dispatch based on tile type
    if (tile_type_mode == 0) {
        // Dense mode - serial worker when a single core is used (params[14]==0).
        if (worldsize == 1 && stiles_control_params[14] == 0) {
            #pragma omp single
            omp_dpotri_dense_serial(tiledMatrix, dep_tracker);
        } else {
            omp_dpotri_dense_parallel(tiledMatrix, dep_tracker, worldsize);
        }
    } else if (tile_type_mode == 1 || tile_type_mode == 2) {
        // Semisparse mode (mode 3 routed to semisparse since factor lives there)
        if (tiledMatrix->chunkedDenseTiles && tiledMatrix->semisparseTileMetaCore) {
            // Selective inverse (params[7]==1): serial/parallel split mirrors the
            // pthreads dispatcher (num_cores<=1 -> serial worker).
            if (stiles_control_params[7] == 1) {
                if (worldsize == 1) {
                    #pragma omp single
                    omp_dpotri_semi_sparse_serial(tiledMatrix, dep_tracker);
                } else {
                    omp_dpotri_semi_sparse_parallel(tiledMatrix, dep_tracker, worldsize);
                }
                return;
            }
            omp_dpotri_semi_dense_inv(tiledMatrix, dep_tracker, worldsize);
        } else {
            // Fallback to dense if semisparse structures not available
            omp_dpotri_dense_parallel(tiledMatrix, dep_tracker, worldsize);
        }
    } else {
        #pragma omp single
        std::cout << "OMP inversion not implemented for tile_type=" << tile_type_mode
                  << ". Only dense (0) and semisparse (1) modes are supported." << std::endl;
    }
}

}} // namespace sTiles::Process

void stiles_pdtrtri(stiles_context_t *stile) {

    TiledMatrix *tiledMatrix;
    sTiles::unpack_args(stile, tiledMatrix);

    static int* stiles_control_params = sTiles_get_params();
    const int tile_type_mode = stiles_control_params[3];

    // Get factorization variant from scheme
    const int variant = tiledMatrix->factorization_variant;

    // Debug dispatch info (commented out for production)
    // if (STILES_RANK == 0) {
    //     std::cout << "\n[stiles_pdtrtri] FASTMODE dispatch info:\n";
    //     std::cout << "  variant=" << variant << " tile_type_mode=" << tile_type_mode << "\n";
    //     std::cout << "  numActiveTiles=" << tiledMatrix->numActiveTiles
    //               << " dimTiledMatrix=" << tiledMatrix->dimTiledMatrix << "\n";
    //     std::cout << "  denseTiles=" << (tiledMatrix->denseTiles ? "valid" : "NULL")
    //               << " inverseTiles=" << (tiledMatrix->inverseTiles ? "valid" : "NULL") << "\n";
    //     std::cout << "  chunkedDenseTiles=" << (tiledMatrix->chunkedDenseTiles ? "valid" : "NULL")
    //               << " chunkedInverseTiles=" << (tiledMatrix->chunkedInverseTiles ? "valid" : "NULL") << "\n";
    //     std::cout << "  semisparseTileMetaCore=" << (tiledMatrix->semisparseTileMetaCore ? "valid" : "NULL") << "\n";
    // }

    // ========== Dispatch based on variant and tile type ==========
    //
    // Variants:
    //   0 = Sparse tiled (only active tiles, tile_type can be dense or semisparse)
    //   1 = Single dense tile covering full matrix (direct LAPACK)
    //   2 = All tiles dense (full triangular tiled structure)
    //   3 = Same as variant 0

    // Variant 1 (single tile): direct inversion of one dense/banded tile.
    if (variant == 1 || (tiledMatrix->numActiveTiles == 1 && tiledMatrix->dimTiledMatrix == 1)) {
        sTiles::Process::pthreads_dpotri_full_dense(tiledMatrix, stile, 1);
        return;
    }

    // Variant 2: full triangular tiled dense.
    if (variant == 2) {
        sTiles::Process::pthreads_dpotri_full_dense(tiledMatrix, stile, 2);
        return;
    }

    // Variants 0, 3: Tiled implementation (sparse tiles with task DAG)
    if (tile_type_mode == 1 || tile_type_mode == 2) {
        // tile type 1 (Semisparse) - check if semisparse structures are available
        if (tiledMatrix->chunkedInverseTiles && tiledMatrix->semisparseTileMetaCore) {
            // Check inverse storage mode: params[7]
            //   0 = dense inverse tiles (default)
            //   1 = semisparse inverse tiles (selective inverse elements only)
            const int inverse_storage_mode = stiles_control_params[7];

            if (inverse_storage_mode == 1) {
                // Semisparse selective inverse: store only elements at original sparsity positions
                // if (STILES_RANK == 0) {
                //     std::cout << "  -> [pdtrtri] VARIANT 0/3 + tile_type=1 + inverse_mode=" << inverse_storage_mode << " (pdtrtri_semi_sparse_inv)\n";
                // }
                //off: are active


                // Serial worker only when single-core AND not forced-parallel
                // (params[14]==0), mirroring the dense split below. Without the
                // params[14] guard the 1-core "parallel" harness cells silently
                // ran the serial executor, so the parallel selective path was
                // never exercised at 1 core.
                if (tiledMatrix->num_cores <= 1 && stiles_control_params[14] == 0) {
                    //sTiles::Process::pdtrtri_semi_sparse_inv(tiledMatrix, stile);
                    //sTiles::Process::pdtrtri_semi_sparse_inv_imp1_serial(tiledMatrix, stile);
                    sTiles::Process::pthreads_dpotri_semi_sparse_serial(tiledMatrix, stile);

                }else{

                    //sTiles::Process::pdtrtri_semi_sparse_inv(tiledMatrix, stile);
                    //sTiles::Process::pdtrtri_semi_sparse_inv_imp1(tiledMatrix, stile);


                    //if (stiles_control_params[14] == 3) {

                        //sTiles::Process::pdtrtri_semi_sparse_inv_imp3(tiledMatrix, stile);
                        //sTiles::Process::pdtrtri_semi_sparse_inv_imp2_analysis(tiledMatrix, stile);

                    //} else {
                        sTiles::Process::pthreads_dpotri_semi_sparse_parallel(tiledMatrix, stile);
                    //}

                }

            } else {
                // Dense inverse: full inverse tiles (inverse_storage_mode=0)
                // if (STILES_RANK == 0) {
                //     std::cout << "  -> [pdtrtri] VARIANT 0/3 + tile_type=1 + inverse_mode=" << inverse_storage_mode << " (pthreads_dpotri_semi_dense_inv)\n";
                // }

                //full dense
                sTiles::Process::pthreads_dpotri_semi_dense_inv(tiledMatrix, stile);
            }
        } else {
            // Fallback to dense mode (e.g., for single-tile matrices)
            // if (STILES_RANK == 0) {
            //     std::cout << "  -> [pdtrtri] VARIANT 0/3 + tile_type=1 FALLBACK (pthreads_dpotri_dense_parallel - dense)\n";
            // }
            sTiles::Process::pthreads_dpotri_dense_parallel(tiledMatrix, stile);
        }
    } else {
        // tile type 0 (Dense)
        // if (STILES_RANK == 0) {
        //     std::cout << "  -> [pdtrtri] VARIANT 0/3 + tile_type=0 (pthreads_dpotri_dense_parallel - dense)\n";
        // }
        // Serial worker when a single core is used (params[14]==0), mirroring
        // the chol-side expansion_dense serial/parallel split.
        if (STILES_SIZE == 1 && stiles_control_params[14] == 0) {
            sTiles::Process::pthreads_dpotri_dense_serial(tiledMatrix, stile);
        } else {
            sTiles::Process::pthreads_dpotri_dense_parallel(tiledMatrix, stile);
        }
    }

}


