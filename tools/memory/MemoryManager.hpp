/**
 * @file MemoryManager.hpp
 * @brief Core memory management utilities for the sTiles framework.
 *
 * Provides centralized memory allocation, tracking, and deallocation services
 * with support for group-based memory organization and leak detection.
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

#ifndef MEMORY_MANAGER_HPP
#define MEMORY_MANAGER_HPP

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <mutex>
#include <algorithm>
#include <string>
#include <cstring>
#include <map>
#include <iomanip>
#include "../common/stiles_mem_alloc.hpp"


// A structure to store an allocation record along with an optional group index.
struct AllocationRecord {
    void* ptr;
    size_t size;
    int groupID; // Use -1 if not assigned.
};

class MemoryManager {
public:
    // Allocate memory for an array of type T.
    // Optional parameter groupID (default -1 means no group assigned).
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

    // Allocate and zero-initialize memory for an array of type T.
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
    
    // Report memory allocation error and throw an exception.
    static void reportError(const std::string& message, const char* file = __FILE__, int line = __LINE__) {
        std::cerr << "MemoryManager Error: " << message 
                  << " (File: " << file << ", Line: " << line << ")" << std::endl;
        throw std::runtime_error(message);
    }
    
    // Deallocate memory and remove it from tracking.
    // The groupID is not needed here because the pointer is unique.
    template<typename T>
    static void deallocate(T*& ptr) {
        if (ptr != nullptr) {
            sTiles_free(ptr);
            untrack(ptr);
            ptr = nullptr;  // Prevent further free attempts.
        }
    }
    
    // Free all allocations registered in the global tracker.
    static void freeAll() {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        for (const auto& record : globalTracker) {
            sTiles_free(record.ptr);
        }
        globalTracker.clear();
    }
    
    // Free all allocations for which record.groupID equals the provided groupID.
    static void freeAllGroup(int groupID) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        // Iterate backwards to safely erase elements from the vector.
        for (int i = static_cast<int>(globalTracker.size()) - 1; i >= 0; i--) {
            if (globalTracker[i].groupID == groupID) {
                sTiles_free(globalTracker[i].ptr);
                globalTracker.erase(globalTracker.begin() + i);
            }
        }
    }
    
    // Compute the total memory (in bytes) allocated in the global tracker.
    static size_t getTotalMemoryAllocated() {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        size_t total = 0;
        for (const auto& record : globalTracker) {
            total += record.size;
        }
        return total;
    }
    
    // Get the total allocated memory in binary gigabytes (GiB) from the global tracker.
    static double getTotalMemoryAllocatedInGiB() {
        size_t totalBytes = getTotalMemoryAllocated();
        return static_cast<double>(totalBytes) / (1024.0 * 1024.0 * 1024.0);
    }

    // Get the total allocated memory in binary gigabytes (GiB) for a specific group.
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
    
    static std::string formatSize(size_t bytes) {
        char buf[32];
        if (bytes >= 1024)
            snprintf(buf, sizeof(buf), "%.2f KiB", bytes / 1024.0);
        else
            snprintf(buf, sizeof(buf), "%zu B", bytes);
        return std::string(buf);
    }

    static void debugDump(std::ostream& out = std::cout) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        out << "===== MemoryManager Allocation Summary =====\n";

        std::map<int, std::vector<AllocationRecord>> grouped;
        for (const auto& record : globalTracker) {
            grouped[record.groupID].push_back(record);
        }

        size_t grandTotalBytes = 0;
        size_t grandTotalCount = 0;

        for (const auto& [groupID, records] : grouped) {
            size_t groupTotal = 0;
            out << "\nGroup " << groupID << ":\n";
            for (const auto& record : records) {
                auto pretty = formatSize(record.size);
                out << "  [" << record.ptr << "] Size: "
                    << std::setw(8) << record.size << " B";
                if (pretty != std::to_string(record.size) + " B")
                    out << " (" << pretty << ")";
                out << "\n";
                groupTotal += record.size;
            }
            out << "  Total allocations: " << records.size() << "\n";
            out << "  Total size:        " << groupTotal << " B";
            auto prettyGroup = formatSize(groupTotal);
            if (prettyGroup != std::to_string(groupTotal) + " B")
                out << " (" << prettyGroup << ")";
            out << "\n";

            grandTotalBytes += groupTotal;
            grandTotalCount += records.size();
        }

        out << "\n--------------------------------------------\n";
        out << "Grand Total:\n";
        out << "  Allocations:  " << grandTotalCount << "\n";
        out << "  Tracked Bytes: " << grandTotalBytes << " B";
        auto prettyGrand = formatSize(grandTotalBytes);
        if (prettyGrand != std::to_string(grandTotalBytes) + " B")
            out << " (" << prettyGrand << ")";
        out << "\n";
    }

    
private:
    // Track the allocated pointer in the global tracker along with its size and group ID.
    static void track(void* ptr, size_t size, int groupID) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        globalTracker.push_back({ptr, size, groupID});
    }
    
    // Remove the pointer from the global tracker.
    static void untrack(void* ptr) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        auto it = std::find_if(globalTracker.begin(), globalTracker.end(), [ptr](const AllocationRecord &rec) {
            return rec.ptr == ptr;
        });
        if (it != globalTracker.end()) {
            globalTracker.erase(it);
        }
    }
    
    // Inline static members for the global tracker (requires C++17)
    inline static std::vector<AllocationRecord> globalTracker;
    inline static std::mutex globalTrackerMutex;
};


#endif // MEMORY_MANAGER_HPP

