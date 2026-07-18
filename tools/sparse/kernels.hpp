/**
 * @file    kernels.hpp
 * @brief   Dense supernode kernels: thin adapters onto the shared sTiles core kernels.
 *
 * @project sTiles (Sparse Tiles Library)
 * @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @version 3.0.0
 * @date 1 1 2026
 * @license Proprietary
 *
 * @note This file is part of the sTiles library, a proprietary software package.
 *       Redistribution or modification without prior permission is prohibited.
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
 *
 * @license
 * This software is proprietary and confidential. Unauthorized copying, distribution, or modification
 * of this software, via any medium, is strictly prohibited. Permission is granted to use the software
 * in binary form for non-commercial purposes only, provided that this copyright notice and permission
 * notice are included in all copies or substantial portions of the software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef _STILES_SPARSE_KERNELS_HPP_
#define _STILES_SPARSE_KERNELS_HPP_

// Dense supernode kernels for the sparse module.
//
// These are thin adapters that forward to the shared sTiles core kernels
// (tools/tile/core_kernels.*), so the whole framework factorizes through a
// single BLAS/LAPACK backend (LAPACKE/CBLAS, with the MKL / Accelerate /
// reference selection handled centrally in tools/common/core_lapack.hpp).
//
// The char/int signatures below match the historic supernode call sites
// (executor/solve/selinv), so those translation units need no changes; the
// bodies translate the BLAS character flags into the core enum API. `double`
// real only.

#include "../tile/core_kernels.hpp"   // sTiles::core_d* + Uplo/Op/Side/Diag

namespace sTiles { namespace sparse {
namespace kernels {

namespace detail {
inline Uplo to_uplo(char c) { return (c == 'U' || c == 'u') ? Uplo::Upper : Uplo::Lower; }
inline Op   to_op(char c)   { return (c == 'T' || c == 't') ? Op::Trans
                                                                      : (c == 'C' || c == 'c') ? Op::ConjTrans
                                                                      :                          Op::NoTrans; }
inline Side to_side(char c) { return (c == 'R' || c == 'r') ? Side::Right : Side::Left; }
inline Diag to_diag(char c) { return (c == 'U' || c == 'u') ? Diag::Unit : Diag::NonUnit; }
// Map the core status onto the LAPACK-style info the call sites expect:
// 0 on success, nonzero otherwise (call sites only test info != 0).
inline int  info_of(StatusCode s) { return (s == StatusCode::Success) ? 0 : static_cast<int>(s); }
}  // namespace detail

// Cholesky factorization. Returns 0 on success, nonzero if not positive definite.
[[nodiscard]] inline int potrf(char uplo, int n, double* A, int lda) {
    return detail::info_of(core_dpotrf(detail::to_uplo(uplo), n, A, lda));
}

inline void trsm(char side, char uplo, char transa, char diag,
                 int m, int n, double alpha,
                 const double* A, int lda, double* B, int ldb) {
    core_dtrsm(detail::to_side(side), detail::to_uplo(uplo),
               detail::to_op(transa), detail::to_diag(diag),
               m, n, alpha, A, lda, B, ldb);
}

inline void gemm(char transa, char transb, int m, int n, int k,
                 double alpha, const double* A, int lda,
                 const double* B, int ldb,
                 double beta,        double* C, int ldc) {
    core_dgemm(detail::to_op(transa), detail::to_op(transb),
               m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void syrk(char uplo, char trans, int n, int k,
                 double alpha, const double* A, int lda,
                 double beta,        double* C, int ldc) {
    core_dsyrk(detail::to_uplo(uplo), detail::to_op(trans),
               n, k, alpha, A, lda, beta, C, ldc);
}

// Triangular inverse. Returns 0 on success, nonzero on a zero diagonal.
[[nodiscard]] inline int trtri(char uplo, char diag, int n, double* A, int lda) {
    return detail::info_of(core_dtrtri(detail::to_uplo(uplo), detail::to_diag(diag), n, A, lda));
}

// In-place L^T·L (uplo='L') or U·U^T (uplo='U'). Returns 0 on success.
[[nodiscard]] inline int lauum(char uplo, int n, double* A, int lda) {
    return detail::info_of(core_dlauum(detail::to_uplo(uplo), n, A, lda));
}

}  // namespace kernels
}}  // namespace sTiles::sparse

#endif  // _STILES_SPARSE_KERNELS_HPP_
