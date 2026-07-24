/**
 * @file lapacke_accel_shim.hpp
 * @brief Minimal LAPACKE-over-Accelerate shim (macOS, ACCELERATE_NEW_LAPACK).
 *
 * Apple's Accelerate framework ships CBLAS and the Fortran LAPACK interface
 * (LAPACK 3.9.1 since macOS 13.3 via ACCELERATE_NEW_LAPACK) but has NO LAPACKE
 * C layer at all — measured by CI's macos-accelerate-bench job: the macOS SDK
 * contains no vecLib/lapacke.h (checked against Xcode 15.4). This header
 * closes that gap by mapping the exact LAPACKE_* surface sTiles uses onto the
 * Fortran symbols Accelerate DOES declare.
 *
 * Scope is deliberately minimal:
 *   - COLUMN-MAJOR ONLY. Every sTiles call site passes LAPACK_COL_MAJOR, so
 *     real LAPACKE's row-major transpose machinery is dead weight here. A
 *     row-major call returns -1 (LAPACKE's "matrix_layout is illegal").
 *   - Only workspace-free routines (potrf/trtri/tbtrs/pbtrf/lauum/laset), so
 *     the *_work variants are identical to the plain ones.
 *
 * Include AFTER <Accelerate/Accelerate.h> with ACCELERATE_NEW_LAPACK defined:
 * the Fortran prototypes (dpotrf_ &c.) must already be in scope, and we call
 * them under whatever declaration Apple ships rather than redeclaring them.
 *
 * @author Esmail Abdul Fattah
 * @affiliation King Abdullah University of Science and Technology (KAUST)
 *
 * @copyright Copyright (c) 2026, KAUST. All rights reserved.
 *
 * PROPRIETARY AND CONFIDENTIAL
 * This software is the exclusive property of KAUST. Unauthorized copying,
 * modification, distribution, or use of this software, in whole or in part,
 * is strictly prohibited without explicit written permission.
 */

#ifndef STILES_LAPACKE_ACCEL_SHIM_HPP
#define STILES_LAPACKE_ACCEL_SHIM_HPP

#if !defined(__APPLE__)
    #error "lapacke_accel_shim.hpp is Accelerate-only (macOS)"
#endif
#if !defined(ACCELERATE_NEW_LAPACK)
    #error "lapacke_accel_shim.hpp needs ACCELERATE_NEW_LAPACK (macOS 13.3+ SDK)"
#endif

#ifndef lapack_int
#define lapack_int int
#endif
#ifndef LAPACK_ROW_MAJOR
#define LAPACK_ROW_MAJOR 101
#endif
#ifndef LAPACK_COL_MAJOR
#define LAPACK_COL_MAJOR 102
#endif

// The Fortran-style LAPACK_* auxiliaries sTiles calls (dlantr/dlascl/dlassq/
// dlangb) are already declared by Accelerate's new-LAPACK headers. Defining
// the macros here makes core_lapack.hpp skip its own guarded redeclarations,
// so every call binds to Apple's prototypes — no signature-clash risk.
#define LAPACK_dlantr dlantr_
#define LAPACK_dlascl dlascl_
#define LAPACK_dlassq dlassq_
#define LAPACK_dlangb dlangb_

// NaN-input checking is an OpenBLAS/MKL LAPACKE extension; with no LAPACKE
// layer there is nothing to switch off.
static inline void LAPACKE_set_nancheck(int /*flag*/) {}

static inline lapack_int LAPACKE_dpotrf_work(int matrix_layout, char uplo,
                                             lapack_int n, double* a,
                                             lapack_int lda)
{
    if (matrix_layout != LAPACK_COL_MAJOR) return -1;
    lapack_int info = 0;
    dpotrf_(&uplo, &n, a, &lda, &info);
    return info;
}

static inline lapack_int LAPACKE_dpotrf(int matrix_layout, char uplo,
                                        lapack_int n, double* a,
                                        lapack_int lda)
{
    return LAPACKE_dpotrf_work(matrix_layout, uplo, n, a, lda);
}

static inline lapack_int LAPACKE_dtrtri_work(int matrix_layout, char uplo,
                                             char diag, lapack_int n,
                                             double* a, lapack_int lda)
{
    if (matrix_layout != LAPACK_COL_MAJOR) return -1;
    lapack_int info = 0;
    dtrtri_(&uplo, &diag, &n, a, &lda, &info);
    return info;
}

static inline lapack_int LAPACKE_dtrtri(int matrix_layout, char uplo,
                                        char diag, lapack_int n, double* a,
                                        lapack_int lda)
{
    return LAPACKE_dtrtri_work(matrix_layout, uplo, diag, n, a, lda);
}

static inline lapack_int LAPACKE_dtbtrs(int matrix_layout, char uplo,
                                        char trans, char diag, lapack_int n,
                                        lapack_int kd, lapack_int nrhs,
                                        const double* ab, lapack_int ldab,
                                        double* b, lapack_int ldb)
{
    if (matrix_layout != LAPACK_COL_MAJOR) return -1;
    lapack_int info = 0;
    // const_cast: AB is read-only for dtbtrs, but Fortran-interface
    // declarations differ on const-qualifying it across SDKs.
    dtbtrs_(&uplo, &trans, &diag, &n, &kd, &nrhs, const_cast<double*>(ab),
            &ldab, b, &ldb, &info);
    return info;
}

static inline lapack_int LAPACKE_dpbtrf(int matrix_layout, char uplo,
                                        lapack_int n, lapack_int kd,
                                        double* ab, lapack_int ldab)
{
    if (matrix_layout != LAPACK_COL_MAJOR) return -1;
    lapack_int info = 0;
    dpbtrf_(&uplo, &n, &kd, ab, &ldab, &info);
    return info;
}

static inline lapack_int LAPACKE_dlauum_work(int matrix_layout, char uplo,
                                             lapack_int n, double* a,
                                             lapack_int lda)
{
    if (matrix_layout != LAPACK_COL_MAJOR) return -1;
    lapack_int info = 0;
    dlauum_(&uplo, &n, a, &lda, &info);
    return info;
}

static inline lapack_int LAPACKE_dlaset_work(int matrix_layout, char uplo,
                                             lapack_int m, lapack_int n,
                                             double alpha, double beta,
                                             double* a, lapack_int lda)
{
    if (matrix_layout != LAPACK_COL_MAJOR) return -1;
    dlaset_(&uplo, &m, &n, &alpha, &beta, a, &lda);   // no INFO argument
    return 0;
}

#endif // STILES_LAPACKE_ACCEL_SHIM_HPP
