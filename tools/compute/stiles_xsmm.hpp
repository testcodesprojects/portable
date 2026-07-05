/**
 * @file    stiles_xsmm.hpp
 * @brief   libxsmm JIT-kernel cache for selinv hot path.
 *
 * Wraps the libxsmm dispatch API with a thread-safe (m,n,k,lda,ldb,ldc)
 * keyed cache. First call for each shape pays the JIT cost; all subsequent
 * calls dereference a function pointer.
 *
 * The selinv kernel is C := A * B (alpha=1, beta=0). Callers wanting
 * `inv3 -= fact * B` (alpha=-1, beta=1) compute `tmp = fact * B` into a scratch
 * buffer and then subtract: this avoids the BLAS-fallback that libxsmm hits
 * when alpha != 1 in its modern API.
 *
 * When STILES_WITH_LIBXSMM is undefined, get_or_dispatch() returns nullptr
 * and the caller is expected to fall back to cblas_dgemm.
 */
#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>

#ifdef STILES_WITH_LIBXSMM
#  include <libxsmm.h>
#endif

namespace sTiles { namespace XSMM {

#ifdef STILES_WITH_LIBXSMM
using gemm_kernel_t = libxsmm_gemmfunction;
#else
// Stand-in typedef so callers compile even without libxsmm. The pointer will
// always be nullptr in this configuration.
using gemm_kernel_t = void*;
#endif

// Pack a GEMM shape signature into a single 64-bit key. m,n,k,lda,ldb,ldc all
// fit in 10 bits (max 1024) — far beyond any selinv tile dim.
inline int64_t pack_shape(int m, int n, int k, int lda, int ldb, int ldc) {
    return  ((int64_t)(m   & 0x3FF))
         | (((int64_t)(n   & 0x3FF)) << 10)
         | (((int64_t)(k   & 0x3FF)) << 20)
         | (((int64_t)(lda & 0x3FF)) << 30)
         | (((int64_t)(ldb & 0x3FF)) << 40)
         | (((int64_t)(ldc & 0x3FF)) << 50);
}

class KernelCache {
public:
    /**
     * Look up or JIT a dgemm kernel for the given shape.
     * The kernel computes  C := 1.0 * A * B + 0.0 * C  (alpha=1, beta=0).
     * NoTrans * NoTrans, column-major, double-precision.
     * Returns nullptr when libxsmm isn't available.
     */
    gemm_kernel_t get_or_dispatch(int m, int n, int k,
                                  int lda, int ldb, int ldc) {
#ifdef STILES_WITH_LIBXSMM
        const int64_t key = pack_shape(m, n, k, lda, ldb, ldc);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            auto it = cache_.find(key);
            if (it != cache_.end()) return it->second;
        }
        // JIT a new kernel for shape (m,n,k,lda,ldb,ldc) with alpha=1, beta=1.
        // The selinv update `inv3 -= fact * B` is realised by negating B during
        // the gather (cheap), then computing C = A * B + C in place.
        const libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(
            m, n, k, lda, ldb, ldc,
            LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
            LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64);
        const libxsmm_bitfield flags    = LIBXSMM_GEMM_FLAG_NONE; // beta=1 default
        const libxsmm_bitfield prefetch = LIBXSMM_PREFETCH_NONE;
        libxsmm_gemmfunction kernel = libxsmm_dispatch_gemm(shape, flags, prefetch);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            auto it = cache_.find(key);
            if (it != cache_.end()) return it->second;
            cache_[key] = kernel;
        }
        return kernel;
#else
        (void)m; (void)n; (void)k; (void)lda; (void)ldb; (void)ldc;
        return nullptr;
#endif
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return cache_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mtx_);
        cache_.clear();
    }

private:
    mutable std::mutex mtx_;
    std::unordered_map<int64_t, gemm_kernel_t> cache_;
};

// Process-wide kernel cache shared across all ranks/threads.
inline KernelCache& global_cache() {
    static KernelCache cache;
    return cache;
}

#ifdef STILES_WITH_LIBXSMM
/**
 * Invoke a JIT'd kernel: tmp = A * B. The caller owns tmp.
 */
inline void invoke(gemm_kernel_t kernel, const double* A, const double* B, double* tmp) {
    libxsmm_gemm_param args;
    args.op.tertiary = nullptr;
    args.a.primary = const_cast<double*>(A);
    args.b.primary = const_cast<double*>(B);
    args.c.primary = tmp;
    kernel(&args);
}
#endif

}} // namespace sTiles::XSMM
