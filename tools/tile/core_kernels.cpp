/**
 * @file core_kernels.cpp
 * @brief Implementation of core BLAS/LAPACK kernel wrappers.
 *
 * Implements dense linear algebra operations using CBLAS and LAPACKE interfaces.
 * Includes matrix addition, multiplication, Cholesky factorization, triangular
 * solves, and specialized banded factorization routines.
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

#include <cstdio>
#include "../common/core_lapack.hpp"
#include "../common/stiles_utils.hpp"
#include "stiles_types.hpp"
#include "core_kernels.hpp"
#include "../common/stiles_logger.hpp"


namespace sTiles {

    StatusCode core_dgeadd(Op transa, i32 m, i32 n, f64 alpha, const f64* A, i32 lda, f64 beta, f64* B, i32 ldb)
    {
        if (transa != Op::NoTrans && transa != Op::Trans && transa != Op::ConjTrans) {
            sTiles::Logger::errorf("illegal value of transa");
            return StatusCode::IllegalValue;
        }
        if (m < 0) {
            sTiles::Logger::errorf("illegal value of m");
            return StatusCode::IllegalValue;
        }
        if (n < 0) {
            sTiles::Logger::errorf("illegal value of n");
            return StatusCode::IllegalValue;
        }
        if (A == nullptr) {
            sTiles::Logger::errorf("NULL A");
            return StatusCode::IllegalValue;
        }
        if ((transa == Op::NoTrans && lda < (m > 0 ? m : 1)) ||
            (transa != Op::NoTrans && lda < (n > 0 ? n : 1))) {
            sTiles::Logger::errorf("illegal value of lda");
            return StatusCode::IllegalValue;
        }
        if (B == nullptr) {
            sTiles::Logger::errorf("NULL B");
            return StatusCode::IllegalValue;
        }
        if (ldb < (m > 0 ? m : 1)) {
            sTiles::Logger::errorf("illegal value of ldb");
            return StatusCode::IllegalValue;
        }

        // Quick return
        if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0))
            return StatusCode::Success;

        // Core computation
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < m; i++) {
                i32 a_index, b_index = ldb * j + i;

                if (transa == Op::NoTrans)
                    a_index = lda * j + i;
                else
                    a_index = lda * i + j;

                B[b_index] = beta * B[b_index] + alpha * A[a_index];
            }
        }

        return StatusCode::Success;
    }

     StatusCode core_dgemm(Op transa, Op transb, i32 m, i32 n, i32 k, f64 alpha, const f64* A, i32 lda, const f64* B, i32 ldb, f64 beta, f64* C, i32 ldc)
    {
        // Concise and safe translation from sTiles::Op to CBLAS_TRANSPOSE
        CBLAS_TRANSPOSE cblas_transa = (transa == Op::Trans)     ? CblasTrans
                                     : (transa == Op::ConjTrans) ? CblasConjTrans
                                     :                           CblasNoTrans;

        CBLAS_TRANSPOSE cblas_transb = (transb == Op::Trans)     ? CblasTrans
                                     : (transb == Op::ConjTrans) ? CblasConjTrans
                                     :                           CblasNoTrans;

        // Direct call to CBLAS, assuming all inputs are valid.
        cblas_dgemm(CblasColMajor,
                    cblas_transa, cblas_transb,
                    m, n, k,
                    alpha, A, lda,
                           B, ldb,
                    beta,  C, ldc);

        return StatusCode::Success;
    }

    StatusCode core_dlauum(Uplo uplo, i32 n, f64* A, i32 lda)
    {

        LAPACKE_dlauum_work(LAPACK_COL_MAJOR, (uplo == Uplo::Upper) ? 'U' : 'L', n, A, lda);

        return StatusCode::Success;
    }

    StatusCode core_dpotrf(Uplo uplo, i32 n, f64* A, i32 lda)
    {

        lapack_int info = LAPACKE_dpotrf_work(LAPACK_COL_MAJOR, (uplo == Uplo::Upper) ? 'U' : 'L', n, A, lda);

        // --- 3. Check LAPACK's return code and map to sTiles StatusCode ---
        if (info == 0) {
            // The operation was successful.
            return StatusCode::Success;
        }
        else if (info > 0) {
            // This is the key case: a computational failure.
            // The leading minor of order 'info' is not positive-definite.
            sTiles::Logger::errorf("sTiles::core_dpotrf: matrix not positive-definite. Factorization failed at step %d.", info);
            return StatusCode::ExecutionFailed;
        }
        else { // info < 0
            // An illegal argument was passed to the LAPACK function.
            // Our validation should catch this, but this handles it just in case.
            sTiles::Logger::errorf("sTiles::core_dpotrf: LAPACKE_dpotrf reported illegal argument at position %d", -info);
            return StatusCode::IllegalValue;
        }

    }

    StatusCode core_dsyrk(Uplo uplo, Op trans, i32 n, i32 k, f64 alpha, const f64* A, i32 lda, f64 beta, f64* C, i32 ldc)
    {
        cblas_dsyrk(CblasColMajor, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans, n, k, alpha, A, lda, beta, C, ldc);

        return StatusCode::Success;
    }

    StatusCode core_dtrmm(Side side, Uplo uplo, Op transa, Diag diag, i32 m, i32 n, f64 alpha, const f64* A, i32 lda, f64* B, i32 ldb)
    {

        cblas_dtrmm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)transa, (CBLAS_DIAG)diag, m, n, alpha, A, lda, B, ldb);

        return StatusCode::Success;
        
    }

    StatusCode core_dtrsm(Side side, Uplo uplo, Op transa, Diag diag, i32 m, i32 n, f64 alpha, const f64* A, i32 lda, f64* B, i32 ldb)
    {
        cblas_dtrsm(CblasColMajor, (CBLAS_SIDE)side, (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)transa, (CBLAS_DIAG)diag, m, n, alpha, A, lda, B, ldb);
        return StatusCode::Success;
    }

    StatusCode core_dtrtri(Uplo uplo, Diag diag, i32 n, f64* A, i32 lda)
    {

        lapack_int info = LAPACKE_dtrtri_work(LAPACK_COL_MAJOR, (uplo == Uplo::Upper) ? 'U' : 'L', (diag == Diag::Unit)  ? 'U' : 'N', n, A, lda);

        return (info == 0) ? StatusCode::Success : StatusCode::ExecutionFailed;
    }

    StatusCode corr_dmirr(int n, double *A, double *B, int lda) {

        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                B[j * lda + i] = A[i * lda + j];
            }
        }

        return StatusCode::Success;
    }

    // StatusCode core_dpotrf_upper_banded(i32 n, i32 kd, f64* A, i32 lda)
    // {
    //     // basic argument checks
    //     if (n < 0) {
    //         sTiles::Logger::errorf("core_dpotrf_upper_banded: illegal value of n");
    //         return StatusCode::IllegalValue;
    //     }
    //     if (kd < 0) {
    //         sTiles::Logger::errorf("core_dpotrf_upper_banded: illegal value of kd");
    //         return StatusCode::IllegalValue;
    //     }
    //     if (A == nullptr) {
    //         sTiles::Logger::errorf("core_dpotrf_upper_banded: NULL A");
    //         return StatusCode::IllegalValue;
    //     }
    //     if (lda < (n > 0 ? n : 1)) {
    //         sTiles::Logger::errorf("core_dpotrf_upper_banded: illegal value of lda");
    //         return StatusCode::IllegalValue;
    //     }

    //     if (n == 0) {
    //         return StatusCode::Success;
    //     }

    //     for (i32 j = 0; j < n; ++j) {

    //         // band for column j: rows [j_start .. j]
    //         i32 j_start = j - kd;
    //         if (j_start < 0) {
    //             j_start = 0;
    //         }

    //         f64* colj = A + j * lda;

    //         // 1) update diagonal A(j,j)
    //         f64 ajj = colj[j];
    //         for (i32 krow = j_start; krow < j; ++krow) {
    //             f64 lkj = colj[krow];
    //             ajj -= lkj * lkj;
    //         }

    //         if (ajj <= 0.0) {
    //             std::fprintf(stderr,
    //                         "core_dpotrf_upper_banded: non positive pivot at j = %d (ajj = %g)\n",
    //                         (int)j, (double)ajj);
    //             return StatusCode::ExecutionFailed;
    //         }

    //         ajj = std::sqrt(ajj);
    //         colj[j] = ajj;

    //         // 2) update row j for columns i in (j+1 .. j+kd)
    //         i32 i_max = j + kd;
    //         if (i_max > n - 1) {
    //             i_max = n - 1;
    //         }

    //         for (i32 i = j + 1; i <= i_max; ++i) {

    //             f64* coli = A + i * lda;

    //             // band for column i: rows [i_start .. i]
    //             i32 i_start = i - kd;
    //             if (i_start < 0) {
    //                 i_start = 0;
    //             }

    //             // intersection of bands of column j and i:
    //             // krow in [max(j_start, i_start) .. j - 1]
    //             i32 k_start = (i_start > j_start) ? i_start : j_start;

    //             f64 aji = coli[j];  // element (j, i)

    //             for (i32 krow = k_start; krow < j; ++krow) {
    //                 aji -= colj[krow] * coli[krow];
    //             }

    //             aji /= ajj;
    //             coli[j] = aji;
    //         }
    //     }

    //     return StatusCode::Success;
    // }

    StatusCode core_dpotrf_diag(i32 n, f64* A, i32 lda)
    {
        // basic checks, same style as other cores
        if (n < 0) {
            sTiles::Logger::errorf("core_dpotrf_diag_upper: illegal value of n");
            return StatusCode::IllegalValue;
        }
        if (A == nullptr) {
            sTiles::Logger::errorf("core_dpotrf_diag_upper: NULL A");
            return StatusCode::IllegalValue;
        }
        if (lda < (n > 0 ? n : 1)) {
            sTiles::Logger::errorf("core_dpotrf_diag_upper: illegal value of lda");
            return StatusCode::IllegalValue;
        }

        if (n == 0) {
            return StatusCode::Success;
        }

        for (i32 j = 0; j < n; ++j) {
            f64& ajj = A[j + j * lda];
            if (ajj <= 0.0) {
                sTiles::Logger::errorf("core_dpotrf_diag_upper: non positive diagonal at j = %d (ajj = %g)",
                             (int)j, (double)ajj);
                return StatusCode::ExecutionFailed;
            }
            ajj = std::sqrt(ajj);
        }

        return StatusCode::Success;
    }

    StatusCode core_dpotrf_upper_banded(i32 n, i32 kd, f64* ab)
    {
        const i32 ldab = kd + 1;
        if (n < 0 || kd < 0 || ab == nullptr) {
            sTiles::Logger::errorf("core_dpbtrf_upper_lapack: invalid arguments (n=%d, kd=%d, ab=%p)",
                         (int)n, (int)kd, (void*)ab);
            return StatusCode::IllegalValue;
        }

#ifdef USE_ACCELERATE
        // Apple Accelerate diagonal regularization for numerical stability
        // Accelerate's floating-point accumulation order differs from OpenBLAS,
        // causing near-zero negative diagonals in semisparse mode
        constexpr f64 DIAG_TOL = 1e-10;
        for (i32 j = 0; j < n; ++j) {
            f64& diag = ab[kd + j * ldab];
            if (diag > -DIAG_TOL && diag <= 0.0) {
                diag = 1e-14;
            }
        }
#endif

        lapack_int info = LAPACKE_dpbtrf(LAPACK_COL_MAJOR, 'U', (lapack_int)n,
                                         (lapack_int)kd, (double*)ab,
                                         (lapack_int)ldab);

        if (info != 0) {
            if (info < 0) {
                sTiles::Logger::errorf("core_dpbtrf_upper_lapack: LAPACKE_dpbtrf illegal argument %d", (int)(-info));
                return StatusCode::IllegalValue;
            }
            sTiles::Logger::errorf("core_dpbtrf_upper_lapack: leading minor of order %d is not positive definite", (int)info);
            return StatusCode::ExecutionFailed;
        }

        return StatusCode::Success;
    }

} // namespace sTiles
