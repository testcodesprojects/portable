/**
 * @file stile_threadWorkspace.hpp
 * @brief Thread-local workspace management for parallel computations.
 *
 * Provides per-thread workspace allocation with aligned memory for efficient
 * SIMD operations and cache utilization in multi-threaded tile computations.
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

#ifndef THREAD_WORKSPACE_HPP
#define THREAD_WORKSPACE_HPP

#include "../common/stiles_types.hpp"
#include <memory> // For std::unique_ptr
#include "cpuSmartTileMemoryManager.hpp"
#include "stile_compiler_hints.hpp"

#ifdef STILE_GPU
  #include <cuda_runtime.h>
  #include <cusolverDn.h>
  #include <cublas_v2.h>
  #include "new_gpu_checks.hpp"
  #include "new_gpuMemoryManager.hpp"
#endif

#include <vector>
#include <cstdint>
#include <cstddef>
#include <new>
#include <cstdlib>
#include <cassert>
#include <type_traits>

namespace sTiles {

// -------------------------------
// 64B aligned STL allocator
// -------------------------------
template<class T, std::size_t A>
struct AlignedAllocator {
    static_assert((A & (A - 1)) == 0, "Alignment must be power of two");
    static_assert(A >= alignof(T), "Alignment must be >= alignof(T)");
    using value_type = T;

    AlignedAllocator() noexcept = default;
    template<class U>
    AlignedAllocator(const AlignedAllocator<U, A>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
#if __cpp_aligned_new >= 201606
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(A)));
#else
        void* p = nullptr;
        if (posix_memalign(&p, A, n * sizeof(T))) throw std::bad_alloc();
        return static_cast<T*>(p);
#endif
    }

    void deallocate(T* p, std::size_t) noexcept {
#if __cpp_aligned_new >= 201606
        ::operator delete(p, std::align_val_t(A));
#else
        std::free(p);
#endif
    }

    using is_always_equal = std::true_type;

    template<class U>
    struct rebind { using other = AlignedAllocator<U, A>; };

    template<class U>
    bool operator==(const AlignedAllocator<U, A>&) const noexcept { return true; }
    template<class U>
    bool operator!=(const AlignedAllocator<U, A>&) const noexcept { return false; }
};

using U8Vec64  = std::vector<std::uint8_t, AlignedAllocator<std::uint8_t, 64>>;
using IntVec64 = std::vector<int,          AlignedAllocator<int,          64>>;

// -------------------------------
// Plain data holder used by kernels
// -------------------------------
struct sTile_workspace {
    int uniformTileSize;
    int tile_size; // compatibility alias used by SmartTile compute kernels
    int groupID;
    int gpu_id;

    f64* A_host = nullptr;
    f64* B_host = nullptr;
    f64* C_host = nullptr;

    std::uint8_t* bit_pattern_buffer = nullptr;
    int*          int_buffer = nullptr;

#ifdef STILE_GPU
    f64* A_device = nullptr;
    f64* B_device = nullptr;
    f64* C_device = nullptr;
    int*          info_device = nullptr;
    f64* work_device = nullptr;
    int           lwork = 0;

    cudaStream_t       stream = nullptr;
    cusolverDnHandle_t cusolver_handle = nullptr;
    cublasHandle_t     cublas_handle   = nullptr;
#endif

    inline double* aligned_A_host() {
        ASSUME_ALIGNED_64(A_host);
        return A_host;
    }

    inline double* aligned_B_host() {
        ASSUME_ALIGNED_64(B_host);
        return B_host;
    }

    inline const double* aligned_A_host() const {
        double* ptr = A_host;
        ASSUME_ALIGNED_64(ptr);
        return ptr;
    }

    inline const double* aligned_B_host() const {
        double* ptr = B_host;
        ASSUME_ALIGNED_64(ptr);
        return ptr;
    }

};

#ifdef STILE_GPU
// RAII wrapper for CUDA handles to guarantee they are always destroyed.
struct CudaHandles {
    cudaStream_t stream = nullptr;
    cusolverDnHandle_t cusolver_handle = nullptr;
    cublasHandle_t cublas_handle = nullptr;

    CudaHandles(int gpu_id) {
        CHECK_CUDA(cudaSetDevice(gpu_id));
        CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle));
        CHECK_CUSOLVER(cusolverDnSetStream(cusolver_handle, stream));
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
    }

    ~CudaHandles() {
        if (cublas_handle)   cublasDestroy(cublas_handle);
        if (cusolver_handle) cusolverDnDestroy(cusolver_handle);
        if (stream)          cudaStreamDestroy(stream);
    }
    CudaHandles(const CudaHandles&) = delete;
    CudaHandles& operator=(const CudaHandles&) = delete;
};
#endif

// -------------------------------
// Optional helper: aligned int arena for carving slices at 64B
// -------------------------------
struct IntArena {
    int*        base = nullptr; // 64B aligned
    std::size_t cap  = 0;       // capacity in ints
    std::size_t head = 0;       // offset in ints

    static std::size_t align_up(std::size_t h, std::size_t align_bytes = 64) {
        const std::size_t ints_per_alignment = align_bytes / sizeof(int);
        return (h + ints_per_alignment - 1) & ~(ints_per_alignment - 1);
    }

    int* alloc(std::size_t n, std::size_t align_bytes = 64) {
        head = align_up(head, align_bytes);
        if (head + n > cap) throw std::bad_alloc();
        int* ptr = base + head;
        head += n;
        return ptr;
    }
};

// -------------------------------
// ThreadWorkspace
// -------------------------------
class ThreadWorkspace {
public:
    ThreadWorkspace(int thread_id, int group_id, int gpu_id, int max_dim, bool use_gpu)
        : thread_id_(thread_id), group_id_(group_id), gpu_id_(gpu_id), use_gpu_(use_gpu)
    {
        W_.uniformTileSize = max_dim;
        W_.tile_size       = max_dim;
        W_.groupID         = group_id_;
        W_.gpu_id          = gpu_id_;

        const std::size_t num_elements = static_cast<std::size_t>(max_dim) * max_dim;

        bit_pattern_buffer_owner_.resize(num_elements);        // bytes
        int_buffer_owner_.resize(4ull * num_elements);         // ints, 64B aligned

        W_.bit_pattern_buffer = bit_pattern_buffer_owner_.data();
        W_.int_buffer         = int_buffer_owner_.data();

        assert(((reinterpret_cast<std::uintptr_t>(W_.int_buffer) & 63u) == 0) &&
               "int_buffer not 64B aligned");


#ifdef STILE_GPU
        if (use_gpu_) {

            handles_ = std::make_unique<CudaHandles>(gpu_id_);
            W_.stream = handles_->stream;
            W_.cusolver_handle = handles_->cusolver_handle;
            W_.cublas_handle = handles_->cublas_handle;
            
            W_.A_host = GpuMemoryManager::allocateHostPinned<f64>(num_elements, group_id_, gpu_id_);
            W_.B_host = GpuMemoryManager::allocateHostPinned<f64>(num_elements, group_id_, gpu_id_);
            W_.C_host = GpuMemoryManager::allocateHostPinned<f64>(num_elements, group_id_, gpu_id_);

            W_.A_device    = GpuMemoryManager::allocateDevice<f64>(num_elements, group_id_, gpu_id_);
            W_.B_device    = GpuMemoryManager::allocateDevice<f64>(num_elements, group_id_, gpu_id_);
            W_.C_device    = GpuMemoryManager::allocateDevice<f64>(num_elements, group_id_, gpu_id_);
            W_.info_device = GpuMemoryManager::allocateDevice<int>(1, group_id_, gpu_id_);

            CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(W_.cusolver_handle, CUBLAS_FILL_MODE_LOWER, max_dim, W_.A_device, max_dim, &W_.lwork));

            if (W_.lwork > 0) {
                W_.work_device = GpuMemoryManager::allocateDevice<f64>(W_.lwork, group_id_, gpu_id_);
            }

        } else
#endif
        {
            W_.A_host = CpuMemoryManager::allocate<f64>(num_elements, group_id_);
            W_.B_host = CpuMemoryManager::allocate<f64>(num_elements, group_id_);
            W_.C_host = CpuMemoryManager::allocate<f64>(num_elements, group_id_);

            assert(((reinterpret_cast<std::uintptr_t>(W_.A_host) & 63u) == 0) && "A_host not 64B aligned");
            assert(((reinterpret_cast<std::uintptr_t>(W_.B_host) & 63u) == 0) && "B_host not 64B aligned");
            assert(((reinterpret_cast<std::uintptr_t>(W_.C_host) & 63u) == 0) && "C_host not 64B aligned");

        }

    }

    ~ThreadWorkspace() = default;
    ThreadWorkspace(const ThreadWorkspace&) = delete;
    ThreadWorkspace& operator=(const ThreadWorkspace&) = delete;

    sTile_workspace* getWorkspace() { return &W_; }
    const sTile_workspace* getWorkspace() const { return &W_; }
    int getThreadId() const { return thread_id_; }
    bool isGpuEnabled() const { return use_gpu_; }
    int getGroupId() const { return group_id_; }

private:
    int       thread_id_;
    int       group_id_;
    int       gpu_id_;
    bool      use_gpu_;

    std::vector<u8> bit_pattern_buffer_owner_;
    IntVec64 int_buffer_owner_;
    sTile_workspace W_{}; // Direct member, NO `new`/`delete` needed.
    
#ifdef STILE_GPU
    // Smart pointer to our RAII handle wrapper. Manages lifetime automatically.
    std::unique_ptr<CudaHandles> handles_;
#endif

};

} // namespace sTiles
#endif