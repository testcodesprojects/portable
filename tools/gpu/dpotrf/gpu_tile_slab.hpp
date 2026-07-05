/**
 * @file    gpu_tile_slab.hpp
 * @brief   Slab allocator for dense tiles + pinned host staging.
 *
 * One cudaMalloc and (optionally) one cudaMallocHost replace the per-tile
 * allocation loop in tools/process/GpuMemoryManager.hpp. Each
 * scheme->dense_tiles_gpu[i].x is pointed into the slab at
 * base + i * stride. H<->D copies become a single async memcpy (or chunked).
 *
 * Sandbox: independent of the live GpuMemoryManager. Intended for the four
 * GPU dpotrf functions in this folder.
 */
#ifndef STILES_GPU_DPOTRF_TILE_SLAB_HPP
#define STILES_GPU_DPOTRF_TILE_SLAB_HPP

#ifdef STILES_GPU

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "gpu_persistent_context.hpp"

// TiledMatrix is defined as `typedef struct {…} TiledMatrix;` (anonymous-
// struct typedef) in stiles_structs.hpp, so a forward declaration of
// `struct TiledMatrix;` would create a *different* type. Pull in the
// real definition. We also need the full TileMetaCore definition (only
// forward-declared in stiles_structs.hpp).
#include "../../common/stiles_structs.hpp"
#include "../../tile/meta.hpp"             // struct TileMetaCore (full def)

namespace sTiles { namespace gpu { namespace dpotrf {


struct GpuTileSlab {
    int       gpu_id        = -1;
    int       tile_size     = 0;       // nb
    int       num_tiles     = 0;       // numActiveTiles
    size_t    stride_elems  = 0;       // tile_size * tile_size
    size_t    total_bytes   = 0;

    double*   d_base        = nullptr; // device slab
    double*   h_pinned      = nullptr; // optional: pinned host staging mirror
    bool      owns_pinned   = false;
};


inline void slab_allocate(GpuTileSlab& slab,
                          int gpu_id,
                          int tile_size,
                          int num_tiles,
                          bool allocate_pinned_host = false) {
    if (num_tiles <= 0 || tile_size <= 0) {
        std::fprintf(stderr, "[gpu_dpotrf] slab_allocate: invalid sizes "
                             "(num_tiles=%d, tile_size=%d)\n", num_tiles, tile_size);
        std::abort();
    }

    STILES_CUDA_CHECK(cudaSetDevice(gpu_id));

    slab.gpu_id       = gpu_id;
    slab.tile_size    = tile_size;
    slab.num_tiles    = num_tiles;
    slab.stride_elems = static_cast<size_t>(tile_size) * tile_size;
    slab.total_bytes  = slab.stride_elems * num_tiles * sizeof(double);

    STILES_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&slab.d_base),
                                 slab.total_bytes));
    STILES_CUDA_CHECK(cudaMemset(slab.d_base, 0, slab.total_bytes));

    if (allocate_pinned_host) {
        STILES_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&slab.h_pinned),
                                         slab.total_bytes));
        slab.owns_pinned = true;
    }
}


inline void slab_destroy(GpuTileSlab& slab) {
    if (slab.d_base) {
        cudaSetDevice(slab.gpu_id);
        cudaFree(slab.d_base);
        slab.d_base = nullptr;
    }
    if (slab.h_pinned && slab.owns_pinned) {
        cudaFreeHost(slab.h_pinned);
        slab.h_pinned   = nullptr;
        slab.owns_pinned = false;
    }
    slab.total_bytes = 0;
}


// Return the device pointer for tile i (0-indexed within the slab).
inline double* slab_tile_ptr(const GpuTileSlab& slab, int i) {
    return slab.d_base + static_cast<size_t>(i) * slab.stride_elems;
}


// Pinned host pointer for tile i (only valid when slab_allocate was called
// with allocate_pinned_host=true). Returns nullptr otherwise.
inline double* slab_tile_host_ptr(const GpuTileSlab& slab, int i) {
    if (!slab.h_pinned) return nullptr;
    return slab.h_pinned + static_cast<size_t>(i) * slab.stride_elems;
}


// Bulk async copy of the entire slab in one direction. Caller syncs the
// stream when it needs ordering with respect to host work.
inline void slab_copy_h2d_async(const GpuTileSlab& slab, cudaStream_t stream) {
    if (!slab.h_pinned) {
        std::fprintf(stderr, "[gpu_dpotrf] slab_copy_h2d_async: no pinned host buffer.\n");
        std::abort();
    }
    STILES_CUDA_CHECK(cudaMemcpyAsync(slab.d_base, slab.h_pinned,
                                      slab.total_bytes,
                                      cudaMemcpyHostToDevice, stream));
}

inline void slab_copy_d2h_async(const GpuTileSlab& slab, cudaStream_t stream) {
    if (!slab.h_pinned) {
        std::fprintf(stderr, "[gpu_dpotrf] slab_copy_d2h_async: no pinned host buffer.\n");
        std::abort();
    }
    STILES_CUDA_CHECK(cudaMemcpyAsync(slab.h_pinned, slab.d_base,
                                      slab.total_bytes,
                                      cudaMemcpyDeviceToHost, stream));
}

// Copy a single tile H -> D from arbitrary (possibly pageable) host pointer.
// Useful when the caller manages its own host buffers (the live runtime's
// dense_tiles[i].elements).
inline void slab_copy_tile_h2d_async(GpuTileSlab& slab, int i,
                                     const double* h_src, size_t elements,
                                     cudaStream_t stream) {
    STILES_CUDA_CHECK(cudaMemcpyAsync(slab_tile_ptr(slab, i), h_src,
                                      elements * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
}

inline void slab_copy_tile_d2h_async(const GpuTileSlab& slab, int i,
                                     double* h_dst, size_t elements,
                                     cudaStream_t stream) {
    STILES_CUDA_CHECK(cudaMemcpyAsync(h_dst, slab_tile_ptr(slab, i),
                                      elements * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream));
}


// Bind every dense_tiles_gpu[i].x to slab_tile_ptr(slab, i). Implemented in
// the .cpp because TiledMatrix layout pulls in the full common headers.
void bind_slab_to_scheme(GpuTileSlab& slab, TiledMatrix* scheme);

}}}  // namespace sTiles::gpu::dpotrf

#endif // STILES_GPU
#endif // STILES_GPU_DPOTRF_TILE_SLAB_HPP
