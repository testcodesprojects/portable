
#ifndef GPU_MEMORY_MANAGER_HPP
#define GPU_MEMORY_MANAGER_HPP

#include <iostream>
#include <stdexcept>
#include <vector>
#include <mutex>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>       
#include <cusolverDn.h>
#include <cublas_v2.h>
// GPU kernels - include from tools/gpu when available
#ifdef STILES_GPU
// TODO: Add new GPU kernel includes here when ready
// #include "../gpu/gpu_kernels.hpp"
#endif
#include "../memory/MemoryManager.hpp"


// A structure to store an allocation record for GPU memory along with an optional group index.
struct GpuAllocationRecord {
    void* ptr;
    size_t size;
    int groupID; // Use -1 if not assigned.
    int GpuID; // Use -1 if not assigned.
};

class GpuMemoryManager {
public:
    // Allocate GPU memory for an array of type T.
    // Optional parameter groupID (default -1 means no group assigned).
    template <typename T>
    static T* allocate(size_t count, int groupID = -1, int GpuID = 0) {
        size_t bytes = count * sizeof(T);
        T* ptr = nullptr;
        cudaError_t err = cudaSetDevice(GpuID);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed for GPU %d: %s\n", GpuID, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaMalloc((void**)&ptr, bytes);
        if (err != cudaSuccess) {
            reportError("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
        track(ptr, bytes, groupID, GpuID);
        return ptr;
    }

    // Allocate and zero-initialize GPU memory for an array of type T.
    template <typename T>
    static T* allocateZero(size_t count, int groupID = 0, int GpuID = 0) {
        cudaError_t err = cudaSetDevice(GpuID);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed for GPU %d: %s\n", GpuID, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        size_t bytes = count * sizeof(T);
        T* ptr = nullptr;
        err = cudaMalloc((void**)&ptr, bytes);
        if (err != cudaSuccess) {
            reportError("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
        // Initialize GPU memory to zero.
        err = cudaMemset(ptr, 0, bytes);
        if (err != cudaSuccess) {
            reportError("cudaMemset failed: " + std::string(cudaGetErrorString(err)));
        }
        track(ptr, bytes, groupID, GpuID);
        return ptr;
    }
    
    // Report a GPU memory allocation/deallocation error and throw an exception.
    static void reportError(const std::string& message, const char* file = __FILE__, int line = __LINE__) {
        std::cerr << "GpuMemoryManager Error: " << message 
                  << " (File: " << file << ", Line: " << line << ")" << std::endl;
        throw std::runtime_error(message);
    }
    
    // Deallocate GPU memory and remove it from tracking.
    template<typename T>
    static void deallocate(T*& ptr) {
        if (ptr != nullptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
            }
            untrack(ptr);
            ptr = nullptr;  // Prevent further free attempts.
        }
    }
    
    // Free all GPU allocations registered in the global tracker.
    static void freeAll() {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        for (auto& record : globalTracker) {
            //std::cout << "Freeing pointer: " << record.ptr << " on GPU " << record.GpuID << std::endl;
    
            cudaError_t err = cudaSetDevice(record.GpuID);
            if (err != cudaSuccess) {
                std::cerr << "cudaSetDevice failed before freeing memory: " 
                          << cudaGetErrorString(err) << std::endl;
                continue;
            }
    
            err = cudaFree(record.ptr);
            if (err != cudaSuccess) {
                std::cerr << "cudaFree failed during freeAll: " << cudaGetErrorString(err) << std::endl;
            } else {
                //std::cout << "Successfully freed pointer: " << record.ptr << std::endl;
                record.ptr = nullptr;
            }
        }
        globalTracker.clear();
    }

    // Free all GPU allocations for a specific group across all GPUs.
    static void freeAllGroup(int groupID) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        for (int i = static_cast<int>(globalTracker.size()) - 1; i >= 0; --i) {
            if (globalTracker[i].groupID == groupID) {
                // Ensure correct device for the allocation
                cudaError_t err = cudaSetDevice(globalTracker[i].GpuID);
                if (err != cudaSuccess) {
                    std::cerr << "cudaSetDevice failed before freeing memory: "
                              << cudaGetErrorString(err) << std::endl;
                    continue;
                }
                err = cudaFree(globalTracker[i].ptr);
                if (err != cudaSuccess) {
                    std::cerr << "cudaFree failed during freeAllGroup: "
                              << cudaGetErrorString(err) << std::endl;
                }
                globalTracker.erase(globalTracker.begin() + i);
            }
        }
    }
    
    
    
    
    // Free all GPU allocations for which record.groupID equals the provided groupID.
    static void freeAllGroup(int groupID, int GpuID) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        // Iterate backwards to safely erase elements from the vector.
        for (int i = static_cast<int>(globalTracker.size()) - 1; i >= 0; i--) {
            if (globalTracker[i].groupID == groupID && globalTracker[i].GpuID == GpuID) {
                cudaError_t err = cudaFree(globalTracker[i].ptr);
                if (err != cudaSuccess) {
                    std::cerr << "cudaFree failed for group " << groupID 
                              << ": " << cudaGetErrorString(err) << std::endl;
                }
                globalTracker.erase(globalTracker.begin() + i);
            }
        }
    }
    
    // Compute the total GPU memory (in bytes) allocated in the global tracker.
    static size_t getTotalMemoryAllocated() {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        size_t total = 0;
        for (const auto& record : globalTracker) {
            total += record.size;
        }
        return total;
    }
    
    // Get the total allocated GPU memory in binary gigabytes (GiB).
    static double getTotalMemoryAllocatedInGiB() {
        size_t totalBytes = getTotalMemoryAllocated();
        return static_cast<double>(totalBytes) / (1024.0 * 1024.0 * 1024.0);
    }

    // Get total allocated GPU memory in GiB for a specific group across all GPUs.
    static double getTotalMemoryAllocatedInGiBForGroup(int groupID) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        size_t total = 0;
        for (const auto& record : globalTracker) {
            if (record.groupID == groupID) total += record.size;
        }
        return static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0);
    }

    // Get the total allocated GPU memory in binary gigabytes (GiB) for a specific group.
    static double getTotalMemoryAllocatedInGiBForGroup(int groupID, int GpuID) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        size_t total = 0;
        for (const auto& record : globalTracker) {
            if (record.groupID == groupID && record.GpuID == GpuID) {
                total += record.size;
            }
        }
        return static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0);
    }
    
private:
    // Track the allocated GPU pointer in the global tracker along with its size and group ID.
    static void track(void* ptr, size_t size, int groupID, int GpuID) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        globalTracker.push_back({ptr, size, groupID, GpuID});
    }
    
    // Remove the pointer from the global tracker.
    static void untrack(void* ptr) {
        std::lock_guard<std::mutex> lock(globalTrackerMutex);
        auto it = std::find_if(globalTracker.begin(), globalTracker.end(), [ptr](const GpuAllocationRecord &rec) {
            return rec.ptr == ptr;
        });
        if (it != globalTracker.end()) {
            globalTracker.erase(it);
        }
    }
    
    // Inline static members for the global tracker (requires C++17)
    inline static std::vector<GpuAllocationRecord> globalTracker;
    inline static std::mutex globalTrackerMutex;
};

//
// Allocate and initialize GPU memory for a dense tile using GpuMemoryManager.
//
void allocate_dense_tile_elements_gpu(TiledMatrix **scheme, int temp_counter, int width, int height, int group_index) {
    // Set the dimensions of the tile.
    (*scheme)->dense_tiles_gpu[temp_counter].width  = width;
    (*scheme)->dense_tiles_gpu[temp_counter].height = height;

    // Compute the number of elements.
    size_t num_elements = static_cast<size_t>(width) * height;

    (*scheme)->dense_tiles_gpu[temp_counter].x = GpuMemoryManager::allocateZero<double>(num_elements, group_index, (*scheme)->GPU_ID);
    cudaDeviceSynchronize();
    // If allocation fails, the GpuMemoryManager::allocateZero function will throw an error.
}

//
// Allocate and initialize GPU memory for an inverse tile using GpuMemoryManager.
//
void allocate_inverse_tile_elements_gpu(TiledMatrix **scheme, int temp_counter, int width, int height, bool is_diagonal, int group_index) {

    // Set the dimensions for the inverse tile.
    (*scheme)->inverse_tiles_gpu[temp_counter].width  = width;
    (*scheme)->inverse_tiles_gpu[temp_counter].height = height;

    // Compute the total number of elements and allocation size.
    size_t total_elems = static_cast<size_t>(width) * height;
    (*scheme)->inverse_tiles_gpu[temp_counter].x = GpuMemoryManager::allocateZero<double>(total_elems, group_index, (*scheme)->GPU_ID);

    // Allocate and zero-initialize the GPU memory for the inverse tile.
    cudaError_t err = cudaSetDevice((*scheme)->GPU_ID);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed for GPU %d: %s\n", (*scheme)->GPU_ID, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // If the tile is diagonal, launch a kernel to set the diagonal elements to 1.0.
    if (is_diagonal) {
        launchSetDiagonalKernel((*scheme)->inverse_tiles_gpu[temp_counter].x, width, height, (*scheme)->GPU_ID);
    }
    cudaDeviceSynchronize();
}


//
// Allocate GPU memory for tree tiles and CUDA streams using GpuMemoryManager for the GPU allocations.
//
void allocate_for_tree_and_streams_gpu(TiledMatrix **scheme, int group_index, int global_index) {
    // Set the correct GPU device before any GPU operations
    cudaError_t err = cudaSetDevice((*scheme)->GPU_ID);
    if (err != cudaSuccess) {
        printf("Error setting GPU device %d: %s\n", (*scheme)->GPU_ID, cudaGetErrorString(err));
        return;
    }

    // Allocate memory for the CUDA streams (host-side)
    int num_streams = (*scheme)->num_cores;
    (*scheme)->streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));

    if (!(*scheme)->streams) {
        printf("Error allocating memory for CUDA streams\n");
        return;
    }

    for (int i = 0; i < num_streams; i++) {
        err = cudaStreamCreate(&(*scheme)->streams[i]);
        if (err != cudaSuccess) {
            printf("Error creating CUDA stream %d: %s\n", i, cudaGetErrorString(err));
            return;
        }
    }

    // Assuming each tree tile is square with dimensions tile_size x tile_size.
    int tile_size = (*scheme)->tile_size;
    size_t total_elems = static_cast<size_t>(tile_size) * tile_size;

    // Determine the number of tree tiles.
    int num_sep = (*scheme)->red_tree_separator_level;

    if (num_sep > 0) {
        // Allocate memory for the array of DenseGpuTile structures on the host
        (*scheme)->gpu_trees = (DenseGpuTile*)malloc(num_sep * num_streams * sizeof(DenseGpuTile));

        if (!(*scheme)->gpu_trees) {
            printf("Error allocating memory for gpu_trees\n");
            return;
        }

        // For each tree tile, allocate and zero-initialize GPU memory
        for (int i = 0; i < num_sep * num_streams; i++) {
            (*scheme)->gpu_trees[i].x = GpuMemoryManager::allocateZero<double>(total_elems, group_index, (*scheme)->GPU_ID);

            if (!(*scheme)->gpu_trees[i].x) {
                printf("Error allocating GPU memory for DenseGpuTile.x\n");
                return;
            }
        }
    }
}


void INTERNAL_UPDATE_X_APPROACH_2_PHASE_1_CPU_TO_GPU_serial(int global_index, TiledMatrix **scheme) {

    cudaError_t err = cudaSetDevice(scheme[global_index]->GPU_ID);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed for GPU %d: %s\n", scheme[global_index]->GPU_ID, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < scheme[global_index]->numActiveTiles; i++) {
        if (scheme[global_index]->dense_tiles_gpu[i].x == nullptr || scheme[global_index]->dense_tiles[i].elements == nullptr) {
            printf("Error: Null pointer at tile %d\n", i);
            continue;
        }

        cudaMemcpy(
            scheme[global_index]->dense_tiles_gpu[i].x,  
            scheme[global_index]->dense_tiles[i].elements, 
            scheme[global_index]->dense_tiles[i].width * scheme[global_index]->dense_tiles[i].height * sizeof(double), 
            cudaMemcpyHostToDevice
        );
    }

    cudaDeviceSynchronize();
}

void INTERNAL_UPDATE_X_APPROACH_2_PHASE_1_CPU_TO_GPU_parallel(int global_index, TiledMatrix **scheme) {

    cudaError_t err = cudaSetDevice(scheme[global_index]->GPU_ID);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed for GPU %d: %s\n", scheme[global_index]->GPU_ID, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (scheme[global_index]->streams == nullptr) {
        scheme[global_index]->streams = (cudaStream_t*)malloc(scheme[global_index]->num_cores * sizeof(cudaStream_t));
        for (int i = 0; i < scheme[global_index]->num_cores; i++) {
            err = cudaStreamCreate(&scheme[global_index]->streams[i]);
            if (err != cudaSuccess) {
                printf("Error creating CUDA stream %d on GPU %d: %s\n", i, scheme[global_index]->GPU_ID, cudaGetErrorString(err));
                return;
            }
        }
    }

    #pragma omp parallel for num_threads(scheme[global_index]->num_cores)
    for (int i = 0; i < scheme[global_index]->numActiveTiles; i++) {

        int stream_id = i % scheme[global_index]->num_cores;

        if (scheme[global_index]->dense_tiles_gpu[i].x == nullptr || scheme[global_index]->dense_tiles[i].elements == nullptr) {
            printf("Error: Null pointer at tile %d\n", i);
            continue;
        }

        cudaMemcpyAsync(
            scheme[global_index]->dense_tiles_gpu[i].x,  
            scheme[global_index]->dense_tiles[i].elements,  
            scheme[global_index]->dense_tiles[i].width * scheme[global_index]->dense_tiles[i].height * sizeof(double), 
            cudaMemcpyHostToDevice, 
            scheme[global_index]->streams[stream_id]
        );
    }

    for (int i = 0; i < scheme[global_index]->num_cores; i++) {
        cudaStreamSynchronize(scheme[global_index]->streams[i]);
    }
}




#endif // GPU_MEMORY_MANAGER_HPP


/*
void allocate_dense_tile_elements_gpu(TiledMatrix **scheme, int temp_counter, int width, int height){
    
    (*scheme)->inverse_tiles[temp_counter].width = width;
    (*scheme)->inverse_tiles[temp_counter].height = height;

    cudaSetDevice((*scheme)->GPU_ID);
    cudaError_t err = cudaMalloc((void**)&(*scheme)->dense_tiles_gpu[temp_counter].x, width * height * sizeof(double));
    if (err != cudaSuccess) {
        printf("Error allocating GPU memory for tile %d: %s\n", temp_counter, cudaGetErrorString(err));
        return;
    }

    // Set the allocated memory for the tile to zero
    err = cudaMemset((*scheme)->dense_tiles_gpu[temp_counter].x, 0, width * height * sizeof(double));
    if (err != cudaSuccess) {
        printf("Error setting GPU memory to zero for tile %d: %s\n", temp_counter, cudaGetErrorString(err));
        return;
    }

}

void allocate_inverse_tile_elements_gpu(TiledMatrix **scheme, int temp_counter, int width, int height, bool is_diagonal)
{

    (*scheme)->inverse_tiles_gpu[temp_counter].width  = width;
    (*scheme)->inverse_tiles_gpu[temp_counter].height = height;

    // Set the active GPU (if multi-GPU setup)
    cudaSetDevice((*scheme)->GPU_ID);

    size_t total_elems = static_cast<size_t>(width) * height;
    size_t alloc_size = total_elems * sizeof(double);

    // Allocate device memory for the inverse tile
    cudaError_t err = cudaMalloc((void**) &(*scheme)->inverse_tiles_gpu[temp_counter].x, alloc_size);

    if (err != cudaSuccess) {
        printf("Error allocating GPU memory for inv tile %d: %s\n", temp_counter, cudaGetErrorString(err));
        return;
    }

    // Initialize entire tile to 0
    err = cudaMemset((*scheme)->inverse_tiles_gpu[temp_counter].x, 0, alloc_size);
    if (err != cudaSuccess) {
        printf("Error setting GPU memory to zero for inv tile %d: %s\n", temp_counter, cudaGetErrorString(err));
        return;
    }

    // If diagonal, set diagonal elements to 1.0
    if (is_diagonal) {
        launchSetDiagonalKernel((*scheme)->inverse_tiles_gpu[temp_counter].x, width, height);
    }

}

void allocate_for_tree_and_streams_gpu(TiledMatrix **scheme)
{
    // Declare the CUDA error variable.
    cudaError_t err;

    int num_streams = (*scheme)->num_cores;
    (*scheme)->streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    if (!(*scheme)->streams) {
        printf("Error allocating memory for CUDA streams\n");
        return;
    }

    // Assuming each tile is square: tile_size x tile_size.
    size_t total_elems = static_cast<size_t>((*scheme)->tile_size) * (*scheme)->tile_size;
    size_t alloc_size = total_elems * sizeof(double);

    // Define num_sep; assuming red_tree_separator_level holds the needed value.
    int num_sep = (*scheme)->red_tree_separator_level;

    if (num_sep > 0) {
        (*scheme)->gpu_trees = (DenseGpuTile*)malloc(num_sep * num_streams * sizeof(DenseGpuTile));
        if (!(*scheme)->gpu_trees) {
            printf("Error allocating memory for gpu_trees\n");
            return;
        }

        for (int i = 0; i < num_sep * num_streams; i++) {

            err = cudaMalloc((void**)&((*scheme)->gpu_trees[i].x), alloc_size);
            if (err != cudaSuccess) {
                printf("Error allocating GPU memory for tile %d: %s\n", i, cudaGetErrorString(err));
                return;
            }

            // Set the allocated memory for the tile to zero.
            err = cudaMemset((*scheme)->gpu_trees[i].x, 0, alloc_size);
            if (err != cudaSuccess) {
                printf("Error setting GPU memory to zero for tile %d: %s\n", i, cudaGetErrorString(err));
                return;
            }
        }
    }
}
*/
