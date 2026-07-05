/**
 * @file    gh200_unified_alloc.hpp
 * @brief   Grace Hopper unified-memory allocator path for sTiles tile slabs.
 *
 * Sits behind STILES_GPU_GRACE_HOPPER (compile-time) AND
 * sTiles::gpu::gh200_unified::enabled() (runtime). When both are true, tile
 * memory is backed by cudaMallocManaged + cudaMemPrefetchAsync hints so the
 * GH200 NVLink-C2C coherence layer can spill across HBM3 + LPDDR5X.
 *
 * Without either: falls back to plain cudaMalloc (the existing path).
 *
 * Why both? The compile-time flag pulls in any GH200-only symbols; the
 * runtime flag lets you A/B compare on the same binary by toggling
 *   STILES_GH200_UNIFIED=0   # disable
 *   STILES_GH200_UNIFIED=1   # enable
 * before launching, or by calling sTiles::gpu::gh200_unified::set(false).
 *
 * Current status: SCAFFOLD. The functions below are correct CUDA calls with
 * TODO_GH200 markers where the real prefetch / access-hint policy goes.
 */
#ifndef STILES_GPU_GH200_UNIFIED_ALLOC_HPP
#define STILES_GPU_GH200_UNIFIED_ALLOC_HPP

#ifdef STILES_GPU

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include "../gpu_dispatch_plan.hpp"   // gh200_unified::enabled()

namespace sTiles { namespace gpu { namespace gh200 {

// Result of an allocation: which path ran, what's the pointer.
struct AllocResult {
    void*  ptr        = nullptr;
    size_t bytes      = 0;
    bool   was_unified = false;   // true if cudaMallocManaged path fired
};

// Allocate `bytes` of GPU-accessible memory for a tile slab on `device_id`.
// Picks unified vs explicit based on:
//   1. STILES_GPU_GRACE_HOPPER must be defined (compile-time)
//   2. sTiles::gpu::gh200_unified::enabled() must be true (runtime)
// If either fails the gate, falls through to plain cudaMalloc on device.
inline AllocResult alloc_slab(int device_id, size_t bytes) {
    AllocResult r{};
    r.bytes = bytes;
    if (cudaSetDevice(device_id) != cudaSuccess) return r;

#ifdef STILES_GPU_GRACE_HOPPER
    if (gh200_unified::enabled()) {
        // ---- GH200 unified-memory path ----
        // Coherent across HBM3 (96 GiB) + LPDDR5X (~480 GiB) via NVLink-C2C.
        cudaError_t e = cudaMallocManaged(&r.ptr, bytes);
        if (e == cudaSuccess) {
            r.was_unified = true;
            // TODO_GH200: pick prefetch + advise hints based on access pattern.
            // For chol tile slabs: GPU is primary accessor, so prefetch onto
            // device; CPU writes value updates before each call so SetAccessedBy
            // host helps the runtime keep CPU-write pages cheap.
            cudaMemPrefetchAsync(r.ptr, bytes, device_id, /*stream=*/0);
            cudaMemAdvise(r.ptr, bytes, cudaMemAdviseSetPreferredLocation, device_id);
            // cudaMemAdvise(r.ptr, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
            std::fprintf(stderr, "[gh200] alloc_slab: managed %.2f GiB on dev %d (unified)\n",
                         (double)bytes / (1024.0*1024.0*1024.0), device_id);
            return r;
        }
        std::fprintf(stderr, "[gh200] cudaMallocManaged failed (%s), falling back to cudaMalloc\n",
                     cudaGetErrorString(e));
    }
#endif

    // ---- Fallback: explicit device alloc (always works on any CUDA GPU) ----
    cudaError_t e = cudaMalloc(&r.ptr, bytes);
    if (e == cudaSuccess) {
        std::fprintf(stderr, "[gh200] alloc_slab: explicit %.2f GiB on dev %d (HBM only)\n",
                     (double)bytes / (1024.0*1024.0*1024.0), device_id);
    } else {
        r.ptr = nullptr;
        std::fprintf(stderr, "[gh200] cudaMalloc failed: %s\n", cudaGetErrorString(e));
    }
    return r;
}

inline void free_slab(const AllocResult& r) {
    if (!r.ptr) return;
    // Both cudaMalloc and cudaMallocManaged buffers free via cudaFree.
    cudaFree(r.ptr);
}

// Optional helper: report the runtime+compile state of the GH200 path.
inline void log_state() {
#ifdef STILES_GPU_GRACE_HOPPER
    const bool compile = true;
#else
    const bool compile = false;
#endif
    const bool runtime = gh200_unified::enabled();
    std::fprintf(stderr, "[gh200] state: compile=%d runtime=%d → active=%d\n",
                 compile ? 1 : 0, runtime ? 1 : 0, (compile && runtime) ? 1 : 0);
}

}}}  // namespace sTiles::gpu::gh200

#endif  // STILES_GPU
#endif  // STILES_GPU_GH200_UNIFIED_ALLOC_HPP
