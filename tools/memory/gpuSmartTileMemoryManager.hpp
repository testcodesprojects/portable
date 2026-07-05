/**
 * @file gpuSmartTileMemoryManager.hpp
 * @brief GPU memory manager for SmartTile data structures.
 *
 * Provides CUDA device memory allocation and pinned host memory management
 * for GPU-accelerated tile computations.
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

#ifndef STILE_GPU_MEMORY_MANAGER_HPP
#define STILE_GPU_MEMORY_MANAGER_HPP

#include <iostream>
#include <stdexcept>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include "compute/stile_gpu_checks.hpp" // Assuming this contains CHECK_CUDA etc.


enum class GpuMemKind { Device, HostPinned };

struct GpuAllocationRecord {
    void* ptr;
    size_t size;
    int groupID;
    int gpu_id;
    GpuMemKind kind;
};

struct GroupGpuKey {
    int groupID;
    int gpu_id;
    bool operator==(const GroupGpuKey& other) const {
        return groupID == other.groupID && gpu_id == other.gpu_id;
    }
};

struct GroupGpuKeyHash {
    std::size_t operator()(const GroupGpuKey& k) const {
        auto hash1 = std::hash<int>{}(k.groupID);
        auto hash2 = std::hash<int>{}(k.gpu_id);
        return hash1 ^ (hash2 << 1);
    }
};

class GpuMemoryManager {
public:
    template <typename T>
    static T* allocateDevice(size_t count, int groupID, int gpu_id) {
        if (count == 0) return nullptr;
        CHECK_CUDA(cudaSetDevice(gpu_id));
        size_t bytes = count * sizeof(T);
        void* ptr = nullptr;
        CHECK_CUDA(cudaMalloc(&ptr, bytes));
        track(ptr, bytes, groupID, gpu_id, GpuMemKind::Device);
        return static_cast<T*>(ptr);
    }

    template <typename T>
    static T* allocateHostPinned(size_t count, int groupID, int gpu_id) {
        if (count == 0) return nullptr;
        // No cudaSetDevice needed for host memory
        size_t bytes = count * sizeof(T);
        void* ptr = nullptr;
        CHECK_CUDA(cudaMallocHost(&ptr, bytes));
        track(ptr, bytes, groupID, gpu_id, GpuMemKind::HostPinned);
        return static_cast<T*>(ptr);
    }

    // Deallocates a single pointer, automatically using cudaFree or cudaFreeHost.
    template <typename T>
    static void deallocate(T*& ptr) {
        if (!ptr) return;
        GpuAllocationRecord record_copy{};
        bool found = untrack(ptr, record_copy);
        if (found) {
            free_record(record_copy);
        }
        ptr = nullptr;
    }

    // Frees all allocations for a specific group on a specific GPU. O(k).
    static void freeAllGroup(int groupID, int gpu_id) {
        std::vector<GpuAllocationRecord> records_to_free;
        {
            std::lock_guard<std::mutex> lock(globalMutex_);
            GroupGpuKey key = {groupID, gpu_id};
            auto group_it = byGroup_.find(key);

            if (group_it != byGroup_.end()) {
                for (void* ptr : group_it->second) {
                    auto alloc_it = byPtr_.find(ptr);
                    if (alloc_it != byPtr_.end()) {
                        records_to_free.push_back(alloc_it->second);
                        byPtr_.erase(alloc_it);
                    }
                }
                byGroup_.erase(group_it);
            }
        }

        // Free outside the lock
        if (!records_to_free.empty()) {
            CHECK_CUDA(cudaSetDevice(gpu_id));
            for (const auto& rec : records_to_free) {
                free_record(rec);
            }
        }
    }

private:
    static void track(void* ptr, size_t size, int groupID, int gpu_id, GpuMemKind kind) {
        std::lock_guard<std::mutex> lock(globalMutex_);
        byPtr_[ptr] = {ptr, size, groupID, gpu_id, kind};
        byGroup_[{groupID, gpu_id}].push_back(ptr);
    }

    // Fills record_copy with data and returns true if found
    static bool untrack(void* ptr, GpuAllocationRecord& record_copy) {
        std::lock_guard<std::mutex> lock(globalMutex_);
        auto it = byPtr_.find(ptr);
        if (it == byPtr_.end()) return false;
        
        record_copy = it->second;
        
        // Remove from group map
        GroupGpuKey key = {record_copy.groupID, record_copy.gpu_id};
        auto group_it = byGroup_.find(key);
        if (group_it != byGroup_.end()) {
            auto& vec = group_it->second;
            vec.erase(std::remove(vec.begin(), vec.end(), ptr), vec.end());
            if (vec.empty()) {
                byGroup_.erase(group_it);
            }
        }
        byPtr_.erase(it);
        return true;
    }
    
    // Helper to call the correct free function. Must be called outside any lock.
    static void free_record(const GpuAllocationRecord& rec) {
        if (rec.kind == GpuMemKind::Device) {
            CHECK_CUDA(cudaFree(rec.ptr));
        } else {
            CHECK_CUDA(cudaFreeHost(rec.ptr));
        }
    }
    
    inline static std::unordered_map<void*, GpuAllocationRecord> byPtr_;
    inline static std::unordered_map<GroupGpuKey, std::vector<void*>, GroupGpuKeyHash> byGroup_;
    inline static std::mutex globalMutex_;
};

#endif // STILE_GPU_MEMORY_MANAGER_HPP