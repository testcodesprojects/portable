/**
 * @file TreeMemoryManager.hpp
 * @brief Memory manager for tree-based data structures.
 *
 * Manages memory allocation for hierarchical tree structures used in
 * task scheduling and dependency management.
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

#ifndef STILES_TREE_MEMORY_MANAGER_HPP
#define STILES_TREE_MEMORY_MANAGER_HPP

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/stiles_mem_alloc.hpp"

struct TreeAllocationRecord {
    void* ptr;
    size_t size;
    int groupID; // Use -1 when no group is associated.
};

class TreeMemoryManager {
public:
    template <typename T>
    static T* allocate(size_t count, int groupID = -1) {
        size_t bytes = count * sizeof(T);
        T* ptr = static_cast<T*>(sTiles_malloc(bytes));
        if (!ptr) {
            reportError("Memory allocation failed.");
        }
        track(ptr, bytes, groupID);
        return ptr;
    }

    template <typename T>
    static T* allocateZero(size_t count, int groupID = -1) {
        size_t bytes = count * sizeof(T);
        T* ptr = static_cast<T*>(sTiles_calloc(count, sizeof(T)));
        if (!ptr) {
            reportError("Memory allocation failed.");
        }
        track(ptr, bytes, groupID);
        return ptr;
    }

    template <typename T>
    static void deallocate(T*& ptr) {
        if (ptr != nullptr) {
            sTiles_free(ptr);
            untrack(ptr);
            ptr = nullptr;
        }
    }

    static void freeAll() {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        for (const auto& record : globalTracker) {
            sTiles_free(record.ptr);
        }
        globalTracker.clear();
    }

    static void freeAllGroup(int groupID) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        for (int i = static_cast<int>(globalTracker.size()) - 1; i >= 0; --i) {
            if (globalTracker[i].groupID == groupID) {
                sTiles_free(globalTracker[i].ptr);
                globalTracker.erase(globalTracker.begin() + i);
            }
        }
    }

    static size_t getTotalMemoryAllocated() {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        size_t total = 0;
        for (const auto& record : globalTracker) {
            total += record.size;
        }
        return total;
    }

    static double getTotalMemoryAllocatedInGiB() {
        size_t bytes = getTotalMemoryAllocated();
        return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    }

    static double getTotalMemoryAllocatedInGiBForGroup(int groupID) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        size_t total = 0;
        for (const auto& record : globalTracker) {
            if (record.groupID == groupID) {
                total += record.size;
            }
        }
        return static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0);
    }

    static void debugDump(std::ostream& out = std::cout) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);

        out << "===== TreeMemoryManager Allocation Summary =====\n";

        std::map<int, std::vector<TreeAllocationRecord>> grouped;
        for (const auto& record : globalTracker) {
            grouped[record.groupID].push_back(record);
        }

        size_t grandTotalBytes = 0;
        std::size_t grandTotalCount = 0;

        for (const auto& [groupID, records] : grouped) {
            size_t groupTotal = 0;
            out << "\nGroup " << groupID << ":\n";
            for (const auto& record : records) {
                out << "  [" << record.ptr << "] Size: "
                    << std::setw(8) << record.size << " B\n";
                groupTotal += record.size;
            }
            out << "  Total allocations: " << records.size() << "\n";
            out << "  Total size:        " << groupTotal << " B\n";

            grandTotalBytes += groupTotal;
            grandTotalCount += records.size();
        }

        out << "\n--------------------------------------------\n";
        out << "Grand Total:\n";
        out << "  Allocations:  " << grandTotalCount << "\n";
        out << "  Tracked Bytes: " << grandTotalBytes << " B\n";
    }

private:
    static void track(void* ptr, size_t size, int groupID) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        globalTracker.push_back({ptr, size, groupID});
    }

    static void untrack(void* ptr) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        auto it = std::find_if(globalTracker.begin(), globalTracker.end(),
                               [ptr](const TreeAllocationRecord& rec) {
                                   return rec.ptr == ptr;
                               });
        if (it != globalTracker.end()) {
            globalTracker.erase(it);
        }
    }

    static void reportError(const std::string& message,
                            const char* file = __FILE__,
                            int line = __LINE__) {
        std::cerr << "TreeMemoryManager Error: " << message
                  << " (File: " << file << ", Line: " << line << ")\n";
        throw std::runtime_error(message);
    }

    inline static std::vector<TreeAllocationRecord> globalTracker;
    inline static std::mutex globalTrackerMutex;
};

#endif // STILES_Tree_MEMORY_MANAGER_HPP
