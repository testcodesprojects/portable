/**
 * @file core_lapack.hpp
 * @brief BLAS/LAPACK backend configuration and include wrapper.
 *
 * Provides a unified interface for including CBLAS and LAPACKE headers
 * across different backends (Intel MKL, Apple Accelerate, OpenBLAS, etc.)
 * with appropriate macro definitions.
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

#ifndef CORE_LAPACK_H
#define CORE_LAPACK_H

#if defined(STILES_WITH_MKL)
    // Intel MKL
    #define MKL_Complex16 double _Complex
    #define MKL_Complex8  float _Complex
    #include <mkl_cblas.h>
    #include <mkl_lapacke.h>

#elif defined(USE_ARMPL)
    // ARM Performance Libraries (Apple Silicon, Graviton)
    #include <armpl.h>

#elif defined(USE_ACCELERATE)
    // macOS Accelerate framework (includes CBLAS and LAPACK)
    #include <Accelerate/Accelerate.h>
    // Accelerate provides lapacke.h since macOS 13.3 (Ventura)
    // For older macOS, users need: brew install lapack
    #ifdef ACCELERATE_NEW_LAPACK
        // macOS 13.3+ has LAPACKE in vecLib
        #include <vecLib/lapacke.h>
    #else
        // Older macOS - need Homebrew lapacke
        #include <lapacke.h>
    #endif

#elif defined(USE_GENERIC_LAPACKE)
    // Generic BLAS/LAPACK (OpenBLAS, reference BLAS, etc.)
    #include <cblas.h>
    #include <lapacke.h>

#else
    #error "LAPACK backend not specified. Define STILES_WITH_MKL, USE_ACCELERATE, or USE_GENERIC_LAPACKE."
#endif


#ifndef lapack_int
#define lapack_int int
#endif

#ifndef LAPACK_GLOBAL
#define LAPACK_GLOBAL(lcname, UCNAME) lcname##_
#endif

// ARMPL already declares all LAPACK symbols via armpl.h
#if !defined(USE_ARMPL)

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LAPACK_dlantr
#define LAPACK_dlantr LAPACK_GLOBAL(dlantr, DLANTR)
double LAPACK_dlantr(const char *norm, const char *uplo, const char *diag,
                     const lapack_int *m, const lapack_int *n,
                     const double *A, const lapack_int *lda,
                     double *work);
#endif

#ifndef LAPACK_dlascl
#define LAPACK_dlascl LAPACK_GLOBAL(dlascl, DLASCL)
void LAPACK_dlascl(const char *type, const lapack_int *kl, const lapack_int *ku,
                   const double *cfrom, const double *cto,
                   const lapack_int *m, const lapack_int *n,
                   double *A, const lapack_int *lda,
                   lapack_int *info);
#endif

#ifndef LAPACK_dlassq
#define LAPACK_dlassq LAPACK_GLOBAL(dlassq, DLASSQ)
void LAPACK_dlassq(const lapack_int *n, const double *x, const lapack_int *incx,
                   double *scale, double *sumsq);
#endif

#ifndef LAPACK_dlangb
#define LAPACK_dlangb LAPACK_GLOBAL(dlangb, DLANGB)
double LAPACK_dlangb(const char *norm,
                     const lapack_int *n, const lapack_int *kl, const lapack_int *ku,
                     const double *A, const lapack_int *lda,
                     double *work);
#endif

#ifdef __cplusplus
}
#endif

#endif // !USE_ARMPL

// Core kernels are now hosted under tools/tile
#include "../tile/core_kernels.hpp"

#ifdef __cplusplus
// Disable LAPACKE NaN input validation to avoid redundant checks on every call
namespace {
    [[maybe_unused]] static const int _lapacke_nancheck_disabled = (LAPACKE_set_nancheck(0), 0);
}
#endif

#endif
