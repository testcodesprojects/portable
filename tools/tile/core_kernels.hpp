/**
 * @file core_kernels.hpp
 * @brief Core BLAS/LAPACK kernel interfaces for dense tile operations.
 *
 * Declares wrapper functions for fundamental linear algebra operations including
 * DGEMM, DPOTRF, DTRSM, DSYRK, DTRMM, DTRTRI, and specialized banded/diagonal
 * Cholesky factorization routines.
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

#include "../common/stiles_types.hpp"

namespace sTiles {

    StatusCode core_dgeadd(Op transa, i32 m, i32 n, f64 alpha, const f64* A, i32 lda, f64 beta, f64* B, i32 ldb);
    StatusCode core_dpotrf(Uplo uplo, i32 n, f64* A, i32 lda);
    StatusCode core_dgemm(Op transa, Op transb, i32 m, i32 n, i32 k, f64 alpha, const f64* A, i32 lda, const f64* B, i32 ldb, f64 beta, f64* C, i32 ldc);
    StatusCode core_dlauum(Uplo uplo, i32 n, f64* A, i32 lda);
    StatusCode core_dsyrk(Uplo uplo, Op trans, i32 n, i32 k, f64 alpha, const f64* A, i32 lda, f64 beta, f64* C, i32 ldc);
    StatusCode core_dtrmm(Side side, Uplo uplo, Op transa, Diag diag, i32 m, i32 n, f64 alpha, const f64* A, i32 lda, f64* B, i32 ldb);
    StatusCode core_dtrsm(Side side, Uplo uplo, Op transa, Diag diag, i32 m, i32 n, f64 alpha, const f64* A, i32 lda, f64* B, i32 ldb);
    StatusCode core_dtrtri(Uplo uplo, Diag diag, i32 n, f64* A, i32 lda);
    StatusCode corr_dmirr(int n, double *A, double *B, int lda);

    //semisparse
    StatusCode core_dpotrf_diag(i32 n, f64* A, i32 lda);
    inline StatusCode core_dpotrf_diag_upper(i32 n, f64* A, i32 lda) {
        return core_dpotrf_diag(n, A, lda);
    }
    StatusCode core_dpotrf_upper_banded(i32 n, i32 kd, f64* A);

}
