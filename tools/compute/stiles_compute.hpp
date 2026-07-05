/**
 * @file stiles_compute.hpp
 * @brief Main computation interface for sTiles factorization routines.
 *
 * Declares the primary compute functions including parallel Cholesky factorization
 * (pdpotrf) and related operations that form the computational core of sTiles.
 *
 * @author Esmail Abdul Fattah
 * @affiliation King Abdullah University of Science and Technology (KAUST)
 * @version 3.0.0
 *
 * @copyright Copyright (c) 2026, KAUST. All rights reserved.
 *
 * PROPRIETARY AND CONFIDENTIAL
 * This software is the exclusive property of KAUST. Unauthorized copying, modification,
 * distribution, or use of this software, in whole or in part, is strictly prohibited.
 * This code may not be reproduced, reverse-engineered, or used to create derivative works
 * without explicit written permission from the copyright holder.
 */

#pragma once

#include "../common/stiles_structs.hpp"
#include "../control/stiles_control.hpp"
#include "../control/common.h"


// ─────────────────────────────────────────────────────────────────────────────
// SafeMode OMP entry (uses dep_tracker for spin-wait). Kept outside the sTiles
// namespace because it predates the namespace cleanup and other TUs link to
// the unmangled symbol — moving it would break those callers.
// ─────────────────────────────────────────────────────────────────────────────
void stiles_omp_dpotrf(TiledMatrix*, omp_dep_tracker_t*);


namespace sTiles {

// ─────────────────────────────────────────────────────────────────────────────
// 1. Cholesky factorization (chol)
// ─────────────────────────────────────────────────────────────────────────────

StatusCode pthreads_dpotrf(int bind_index, TiledMatrix *scheme);
StatusCode omp_dpotrf     (int bind_index, TiledMatrix *scheme);


// ─────────────────────────────────────────────────────────────────────────────
// 2. CSC factor packing
//
// Two overloads of pack_L_values:
//   * pack_L_values(stile, scheme) — drives the bound pthreads pool exactly
//                                    like sTiles_chol (parallel_call →
//                                    Process::ppack). Recommended when a
//                                    bound context is available.
//   * pack_L_values(scheme)        — standalone: spawns std::thread workers
//                                    when param[8] == 0, OMP parallel-for
//                                    when param[8] == 1.
// ─────────────────────────────────────────────────────────────────────────────

void pack_L_values(stiles_context_t* stile, TiledMatrix* scheme);
void pack_L_values(TiledMatrix* scheme);

// Runtime tuning for the L_src precomputed-pointer-table memory ceiling
// (default 2 GiB). Set this before the first sTiles_packing call to force
// the per-entry fallback (b=0) or to allow larger tables on memory-rich
// machines.
long long get_pack_cache_threshold_bytes();
void      set_pack_cache_threshold_bytes(long long bytes);


// ─────────────────────────────────────────────────────────────────────────────
// 3. CSC fast-path triangular solve
//
// Serial-CSC solver consuming the flat L_values buffer produced by
// pack_L_values. Used by sTiles_solve_LLT / _L / _LT when packed==true and
// the eligibility gate (num_cores, nrhs, tile_type_mode) lets it run.
//
// solve_type: 0 = forward  L y  = b
//             1 = backward L^T x = y
//             2 = full     L L^T x = b
// ─────────────────────────────────────────────────────────────────────────────

void csc_dtrsm           (const TiledMatrix* scheme, double* x,
                          int solve_type);

// Multi-RHS variant for small nrhs (typically 2..8). One pass over L,
// updates K columns of X in lockstep with SIMD over the K dimension.
// X is column-major with leading dim ldb.
void csc_dtrsm_multi     (const TiledMatrix* scheme, double* X,
                          int nrhs, int ldb, int solve_type);

// Same algorithm, X is row-major (row i starts at X + i*ldb_row, ldb_row >= nrhs).
void csc_dtrsm_multi_row (const TiledMatrix* scheme, double* X, int nrhs,
                          int ldb_row, int solve_type);


// ─────────────────────────────────────────────────────────────────────────────
// 4. Tile-path triangular solve (OMP)
//
// Used when the CSC fast-path isn't eligible (e.g., nrhs > 8, multiple cores
// per call). Walks the tile structure with BLAS dtrsm/dgemm per active tile.
// ─────────────────────────────────────────────────────────────────────────────

StatusCode omp_dtrsm     (int bind_index, TiledMatrix *scheme,
                          double *B, int nrhs, int solve_type);


// ─────────────────────────────────────────────────────────────────────────────
// 5. Selected inversion (after Cholesky)
//
// Computes the entries of A^{-1} corresponding to the sparsity pattern of L,
// using the Takahashi recurrence.
// ─────────────────────────────────────────────────────────────────────────────

StatusCode pthreads_dpotri(int bind_index, TiledMatrix *scheme);
StatusCode omp_dpotri     (int bind_index, TiledMatrix *scheme);


// ─────────────────────────────────────────────────────────────────────────────
// 6. namespace Process — task-graph worker entry points
//
// Worker bodies that execute one logical unit of work (one supernode, one
// tile, one chunk). Driven by parallel_call(stile, Process::xxx, ...) on
// the pthreads pool, or by the OMP-task analogues which take a dep_tracker.
// ─────────────────────────────────────────────────────────────────────────────

namespace Process {

    // ─── Cholesky workers ───
    void pthreads_pdpotrf   (stiles_context_t*);                                       // pthreads
    void omp_pdpotrf        (TiledMatrix*, omp_dep_tracker_t*, int worldsize);         // OMP

    // ─── CSC packing worker ───
    void ppack              (stiles_context_t*);                                       // pthreads

    // ─── Inversion worker (OMP) ───
    void omp_pdtrtri        (TiledMatrix*, omp_dep_tracker_t*, int worldsize);

    // ─── Solve workers (OMP, generic) ───
    // Forward / backward over the full tile grid, no pre-collected schedule.
    void omp_pdtrsm_forward                  (TiledMatrix*, double* B, int nrhs,
                                              omp_dep_tracker_t*, int worldsize);
    void omp_pdtrsm_backward                 (TiledMatrix*, double* B, int nrhs,
                                              omp_dep_tracker_t*, int worldsize);
    void omp_pdtrsm_forward_dense_full       (TiledMatrix*, double* B, int nrhs,
                                              omp_dep_tracker_t*, int worldsize);
    void omp_pdtrsm_backward_dense_full      (TiledMatrix*, double* B, int nrhs,
                                              omp_dep_tracker_t*, int worldsize);
    void omp_pdtrsm_forward_semisparse       (TiledMatrix*, double* B, int nrhs,
                                              omp_dep_tracker_t*, int worldsize);
    void omp_pdtrsm_backward_semisparse      (TiledMatrix*, double* B, int nrhs,
                                              omp_dep_tracker_t*, int worldsize);

    // ─── Solve workers (OMP, fast — pre-collected tasks) ───
    // Use solve_fwd_tasks / solve_bwd_tasks built at preprocess time.
    void omp_pdtrsm_forward_fast             (TiledMatrix*, double* B, int nrhs,
                                              omp_dep_tracker_t*, int worldsize);
    void omp_pdtrsm_backward_fast            (TiledMatrix*, double* B, int nrhs,
                                              omp_dep_tracker_t*, int worldsize);
    void omp_pdtrsm_forward_semisparse_fast  (TiledMatrix*, double* B, int nrhs,
                                              omp_dep_tracker_t*, int worldsize);
    void omp_pdtrsm_backward_semisparse_fast (TiledMatrix*, double* B, int nrhs,
                                              omp_dep_tracker_t*, int worldsize);

} // namespace Process

} // namespace sTiles
