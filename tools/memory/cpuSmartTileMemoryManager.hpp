/**
 * @file cpuSmartTileMemoryManager.hpp
 * @brief CPU memory manager for SmartTile data structures.
 *
 * Provides aligned memory allocation optimized for CPU cache efficiency
 * and SIMD operations on tile data.
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

#ifndef STILE_CPU_MEMORY_MANAGER_HPP
#define STILE_CPU_MEMORY_MANAGER_HPP

#include <cstddef>
#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <limits>

#include "../common/stiles_types.hpp"
#include "../common/stiles_mem_alloc.hpp"
#include <cstdlib>

namespace sTiles {

struct CpuAllocationInfo {
    size_t size;
    int groupID;
};

class CpuMemoryManager {
public:
    static constexpr int NO_GROUP = -1;

    template <typename T>
    static T* allocate(size_t count, int groupID = NO_GROUP) {
        if (count == 0) return nullptr;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        size_t bytes = count * sizeof(T);
        void* ptr = sTiles_malloc_aligned(bytes, STILES_ALIGN);
        if (!ptr) {
            throw std::bad_alloc();
        }
        track(ptr, bytes, groupID);
        return static_cast<T*>(ptr);
    }

    template <typename T>
    static T* allocateZero(size_t count, int groupID = NO_GROUP) {
        if (count == 0) return nullptr;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        size_t bytes = count * sizeof(T);
        void* ptr = sTiles_calloc_aligned(count, sizeof(T), STILES_ALIGN);
        if (!ptr) {
            throw std::bad_alloc();
        }
        track(ptr, bytes, groupID);
        return static_cast<T*>(ptr);
    }

    template <typename T>
    static void deallocate(T*& ptr) {
        if (ptr) {
            untrack(ptr);
            ptr = nullptr;
        }
    }

    static void freeAllGroup(int groupID) {
        std::vector<void*> pointers_to_free;
        {
            std::lock_guard<std::mutex> lock(globalMutex_);
            auto group_it = groupMap_.find(groupID);
            if (group_it != groupMap_.end()) {
                for (void* ptr : group_it->second) {
                    auto alloc_it = allocationMap_.find(ptr);
                    if (alloc_it != allocationMap_.end()) {
                        totalSize_ -= alloc_it->second.size;
                        allocationMap_.erase(alloc_it);
                    }
                }
                pointers_to_free = std::move(group_it->second);
                groupMap_.erase(group_it);
                if (groupMap_.empty()) groupMap_.rehash(0);
                if (allocationMap_.empty()) allocationMap_.rehash(0);
            }
        }
        for (void* ptr : pointers_to_free) {
            sTiles_free_aligned(ptr);
        }
    }

    static void freeAll() {
        std::vector<void*> pointers_to_free;
        {
            std::lock_guard<std::mutex> lock(globalMutex_);
            pointers_to_free.reserve(allocationMap_.size());
            for (const auto& pair : allocationMap_) {
                pointers_to_free.push_back(pair.first);
            }
            allocationMap_.clear();
            groupMap_.clear();
            totalSize_ = 0;
            allocationMap_.rehash(0);
            groupMap_.rehash(0);
        }
        for (void* ptr : pointers_to_free) {
            sTiles_free_aligned(ptr);
        }
    }

    static size_t getTotalMemoryAllocated() {
        std::lock_guard<std::mutex> lock(globalMutex_);
        return totalSize_;
    }

    static size_t getMemoryAllocatedForGroup(int groupID) {
        std::lock_guard<std::mutex> lock(globalMutex_);
        size_t group_total = 0;
        auto group_it = groupMap_.find(groupID);
        if (group_it != groupMap_.end()) {
            for (void* ptr : group_it->second) {
                auto alloc_it = allocationMap_.find(ptr);
                if (alloc_it != allocationMap_.end()) {
                    group_total += alloc_it->second.size;
                }
            }
        }
        return group_total;
    }

    static void debugDump(std::ostream& out = std::cout) {
        std::lock_guard<std::mutex> lock(globalMutex_);
        out << "===== CPU MemoryManager Allocation Summary =====\n";

        size_t grandTotalBytes = 0;
        size_t grandTotalCount = 0;

        for (const auto& [groupID, ptr_vector] : groupMap_) {
            size_t groupTotalBytes = 0;
            out << "\n--- Group " << groupID << " ---\n";
            for (const void* ptr : ptr_vector) {
                auto it = allocationMap_.find(const_cast<void*>(ptr));
                if (it != allocationMap_.end()) {
                    const auto& info = it->second;
                    out << "  [" << ptr << "] Size: " << info.size << " B\n";
                    groupTotalBytes += info.size;
                }
            }
            out << "  Group Allocations: " << ptr_vector.size() << "\n";
            out << "  Group Total Size:  " << groupTotalBytes << " B\n";

            grandTotalBytes += groupTotalBytes;
            grandTotalCount += ptr_vector.size();
        }

        out << "\n--------------------------------------------\n";
        out << "Grand Total:\n";
        out << "  Total Allocations: " << grandTotalCount << "\n";
        out << "  Total Tracked Memory: " << totalSize_ << " B\n";
        out << "============================================\n";

        if (totalSize_ != grandTotalBytes) {
            out << "WARNING: Mismatch between calculated total (" << grandTotalBytes
                << ") and stored total (" << totalSize_ << ").\n";
        }
    }

private:
    static void track(void* ptr, size_t size, int groupID) {
        std::lock_guard<std::mutex> lock(globalMutex_);
        allocationMap_[ptr] = {size, groupID};
        groupMap_[groupID].push_back(ptr);
        totalSize_ += size;
    }

    static void untrack(void* ptr) {
        bool found = false;
        {
            std::lock_guard<std::mutex> lock(globalMutex_);
            auto it = allocationMap_.find(ptr);
            if (it != allocationMap_.end()) {
                found = true;
                const CpuAllocationInfo info = it->second;
                totalSize_ -= info.size;

                auto group_it = groupMap_.find(info.groupID);
                if (group_it != groupMap_.end()) {
                    auto& vec = group_it->second;
                    vec.erase(std::remove(vec.begin(), vec.end(), ptr), vec.end());
                    if (vec.empty()) {
                        groupMap_.erase(group_it);
                    }
                }
                allocationMap_.erase(it);
            }
        }
        if (found) {
            sTiles_free_aligned(ptr);
        }
    }

    inline static std::unordered_map<void*, CpuAllocationInfo> allocationMap_;
    inline static std::unordered_map<int, std::vector<void*>> groupMap_;
    inline static size_t totalSize_ = 0;
    inline static std::mutex globalMutex_;
};

} // namespace sTiles

#endif // STILE_CPU_MEMORY_MANAGER_HPP
