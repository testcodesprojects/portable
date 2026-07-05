#pragma once

// MKL is the only backend in this tree guaranteed to expose `cblas_ddoti`
// (Sparse BLAS Level 1). Other backends (Accelerate / OpenBLAS / ARMPL /
// generic) fall back to the hand-rolled unroll-8 loop below for all n.
// Detection macro `STILES_WITH_MKL` is set in make.inc when MKL is detected.
#if defined(STILES_WITH_MKL)
#ifdef __cplusplus
extern "C" {
#endif
double cblas_ddoti(int N, const double *X, const int *indx, const double *Y);
#ifdef __cplusplus
}
#endif
#endif

// computes sum_i v[i] * a[idx[i]]
//   MKL build,   n  > 256 : delegate to cblas_ddoti
//   otherwise (any n)     : single SIMD reduction loop
static inline __attribute__((__always_inline__))
double stiles_sparse_ddot(int n,
                          const double * __restrict__ v,
                          const double * __restrict__ a,
                          const int    * __restrict__ idx)
{
#if defined(STILES_WITH_MKL)
    // MKL Sparse BLAS L1 wins for long vectors; short ones stay on the loop.
    if (n > 256) {
        return cblas_ddoti(n, v, idx, a);
    }
#endif
    // All backends (and MKL for n <= 256): SIMD reduction. cblas_ddoti is an
    // MKL-only extension; OpenBLAS / Accelerate / ARMPL do not expose it, so
    // the call above must stay behind STILES_WITH_MKL.
    double res = 0.0;
    #pragma omp simd reduction(+:res)
    for (int i = 0; i < n; ++i) {
        res += v[i] * a[idx[i]];
    }
    return res;
}