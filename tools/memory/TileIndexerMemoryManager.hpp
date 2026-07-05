/**
 * @file TileIndexerMemoryManager.hpp
 * @brief Memory manager for tile indexing data structures.
 *
 * Handles memory allocation for tile index maps, adjacency graphs,
 * and lookup tables used in tile-based matrix operations.
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

#ifndef STILES_TILE_INDEXER_MEMORY_MANAGER_HPP
#define STILES_TILE_INDEXER_MEMORY_MANAGER_HPP

#include <cstddef>
#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <type_traits>

#include "../common/stiles_mem_alloc.hpp"
#include <cstdlib>

namespace sTiles {

// Information stored for each tracked allocation.
struct TileIndexerAllocationInfo {
    size_t size;
    int groupID;
};

class TileIndexerMemoryManager {
public:
    enum class Group : int {
        ActiveBool = 0,
        ActiveChar,
        DenseBits,
        SparseSet,
        SparseIds,
        TiledChunkMap,
        TiledChunkWords,
        PagedMap,
        PagedWords,
        FillScratch,
        GraphOffsets,
        GraphEdges,
        SFL,
        Misc
    };

    static constexpr int group_id(Group g) noexcept {
        return static_cast<int>(g);
    }

    static const char* group_name(Group g) {
        switch (g) {
            case Group::ActiveBool:     return "ActiveBool";
            case Group::ActiveChar:     return "ActiveChar";
            case Group::DenseBits:      return "DenseBits";
            case Group::SparseSet:      return "SparseSet";
            case Group::SparseIds:      return "SparseIds";
            case Group::TiledChunkMap:  return "TiledChunkMap";
            case Group::TiledChunkWords:return "TiledChunkWords";
            case Group::PagedMap:       return "PagedMap";
            case Group::PagedWords:     return "PagedWords";
            case Group::FillScratch:    return "FillScratch";
            case Group::GraphOffsets:   return "GraphOffsets";
            case Group::GraphEdges:     return "GraphEdges";
            case Group::SFL:            return "SFL";
            case Group::Misc:           return "Misc";
        }
        return "Unknown";
    }

    template <typename T>
    class Allocator {
    public:
        using value_type = T;
        using pointer = T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using propagate_on_container_copy_assignment = std::true_type;
        using propagate_on_container_move_assignment = std::true_type;
        using propagate_on_container_swap = std::true_type;
        using is_always_equal = std::false_type;

        Allocator(int group = NO_GROUP) noexcept : groupID_(group) {}

        template <typename U>
        Allocator(const Allocator<U>& other) noexcept : groupID_(other.groupID_) {}

        pointer allocate(size_type n) {
            if (n > max_size()) throw std::bad_alloc();
            return TileIndexerMemoryManager::allocate<T>(n, groupID_);
        }

        void deallocate(pointer p, size_type) noexcept {
            TileIndexerMemoryManager::deallocateRaw(static_cast<void*>(p));
        }

        size_type max_size() const noexcept {
            return (std::numeric_limits<size_type>::max)() / (sizeof(T) > 0 ? sizeof(T) : 1);
        }

        int group_id() const noexcept { return groupID_; }

        template <typename U>
        bool operator==(const Allocator<U>& other) const noexcept {
            return groupID_ == other.groupID_;
        }

        template <typename U>
        bool operator!=(const Allocator<U>& other) const noexcept {
            return !(*this == other);
        }

    private:
        int groupID_;
        template <typename> friend class Allocator;
    };

    template <typename T>
    static Allocator<T> makeAllocator(Group g) noexcept {
        return Allocator<T>(group_id(g));
    }

    static constexpr int NO_GROUP = -1;

    template <typename T>
    static T* allocate(size_t count, int groupID = NO_GROUP) {
        if (count == 0) return nullptr;
        const size_t bytes = count * sizeof(T);
        void* ptr = sTiles_malloc(bytes);
        if (!ptr) throw std::bad_alloc();
        track(ptr, bytes, groupID);
        return static_cast<T*>(ptr);
    }

    template <typename T>
    static T* allocateZero(size_t count, int groupID = NO_GROUP) {
        if (count == 0) return nullptr;
        void* ptr = sTiles_calloc(count, sizeof(T));
        if (!ptr) throw std::bad_alloc();
        track(ptr, count * sizeof(T), groupID);
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
            }
        }
        for (void* ptr : pointers_to_free) {
            sTiles_free(ptr);
        }
    }

    static void freeAllGroup(Group group) {
        freeAllGroup(group_id(group));
    }

    static void freeAll() {
        std::vector<void*> pointers_to_free;
        {
            std::lock_guard<std::mutex> lock(globalMutex_);
            pointers_to_free.reserve(allocationMap_.size());
            for (const auto& pair : allocationMap_) pointers_to_free.push_back(pair.first);
            allocationMap_.clear();
            groupMap_.clear();
            totalSize_ = 0;
        }
        for (void* ptr : pointers_to_free) sTiles_free(ptr);
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
                if (alloc_it != allocationMap_.end()) group_total += alloc_it->second.size;
            }
        }
        return group_total;
    }

    static size_t getMemoryAllocatedForGroup(Group group) {
        return getMemoryAllocatedForGroup(group_id(group));
    }

    struct UsageStats {
        size_t total = 0;
        std::vector<size_t> groups;
    };

    static UsageStats snapshotUsage() {
        UsageStats stats;
        std::lock_guard<std::mutex> lock(globalMutex_);
        stats.total = totalSize_;
        stats.groups.resize(static_cast<std::size_t>(Group::Misc) + 1, 0);
        for (std::size_t g = 0; g < stats.groups.size(); ++g) {
            auto it = groupMap_.find(static_cast<int>(g));
            if (it == groupMap_.end()) continue;
            size_t group_total = 0;
            for (void* ptr : it->second) {
                auto alloc_it = allocationMap_.find(ptr);
                if (alloc_it != allocationMap_.end()) group_total += alloc_it->second.size;
            }
            stats.groups[g] = group_total;
        }
        return stats;
    }

    static void debugDump(std::ostream& out = std::cout) {
        std::lock_guard<std::mutex> lock(globalMutex_);
        out << "===== TileIndexerMemoryManager Allocation Summary =====\n";
        size_t grandTotalBytes = 0, grandTotalCount = 0;

        for (const auto& [groupID, ptrs] : groupMap_) {
            size_t groupTotalBytes = 0;
            const char* label = (groupID >= 0 && groupID <= group_id(Group::Misc))
                                ? group_name(static_cast<Group>(groupID))
                                : "Ungrouped";
            out << "\n--- Group " << groupID << " (" << label << ") ---\n";
            for (const void* p : ptrs) {
                auto it = allocationMap_.find(const_cast<void*>(p));
                if (it != allocationMap_.end()) {
                    const auto& info = it->second;
                    out << "  [" << p << "] Size: " << info.size << " B\n";
                    groupTotalBytes += info.size;
                }
            }
            out << "  Group Allocations: " << ptrs.size() << "\n";
            out << "  Group Total Size:  " << groupTotalBytes << " B\n";
            grandTotalBytes += groupTotalBytes;
            grandTotalCount += ptrs.size();
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

    template <typename Func>
    static void for_each_group(Func&& func) {
        std::vector<std::pair<int, size_t>> snapshot;
        {
            std::lock_guard<std::mutex> lock(globalMutex_);
            snapshot.reserve(groupMap_.size());
            for (const auto& [groupID, ptrs] : groupMap_) {
                size_t total = 0;
                for (const void* ptr : ptrs) {
                    auto alloc_it = allocationMap_.find(const_cast<void*>(ptr));
                    if (alloc_it != allocationMap_.end()) total += alloc_it->second.size;
                }
                snapshot.emplace_back(groupID, total);
            }
        }
        for (const auto& entry : snapshot) func(entry.first, entry.second);
    }

    // ---- Helpers for GlobalMemoryManager ----
    static bool owns(const void* ptr) {
        std::lock_guard<std::mutex> lock(globalMutex_);
        return allocationMap_.find(const_cast<void*>(ptr)) != allocationMap_.end();
    }
    static bool deallocateRaw(void* ptr) {
        if (!ptr) return false;
        bool did = false;
        {
            std::lock_guard<std::mutex> lock(globalMutex_);
            auto it = allocationMap_.find(ptr);
            if (it != allocationMap_.end()) did = true;
        }
        if (did) untrack(ptr);
        return did;
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
                const auto info = it->second;
                totalSize_ -= info.size;

                auto group_it = groupMap_.find(info.groupID);
                if (group_it != groupMap_.end()) {
                    auto& vec = group_it->second;
                    vec.erase(std::remove(vec.begin(), vec.end(), ptr), vec.end());
                    if (vec.empty()) groupMap_.erase(group_it);
                }
                allocationMap_.erase(it);
                found = true;
            }
        }
        if (found) sTiles_free(ptr);
    }

    inline static std::unordered_map<void*, TileIndexerAllocationInfo> allocationMap_;
    inline static std::unordered_map<int, std::vector<void*>> groupMap_;
    inline static size_t totalSize_ = 0;
    inline static std::mutex globalMutex_;
};

} // namespace sTiles

#endif // STILES_TILE_INDEXER_MEMORY_MANAGER_HPP
