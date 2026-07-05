/**
 * @file memory_for_compute.hpp
 * @brief GPU memory management utilities for compute operations.
 *
 * Provides GPU memory transfer and synchronization functions for copying
 * tile data between CPU and GPU during hybrid factorization computations.
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

#ifndef GPU_MEMORY_FOR_COMPUTE
#define GPU_MEMORY_FOR_COMPUTE

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


inline void INTERNAL_UPDATE_X_APPROACH_2_PHASE_1_GPU_TO_CPU_serial(int global_index, TiledMatrix **scheme) {

    // Ensure correct GPU is set
    cudaError_t err = cudaSetDevice(scheme[global_index]->GPU_ID);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed for GPU %d: %s\n", scheme[global_index]->GPU_ID, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < scheme[global_index]->numActiveTiles; i++) {
        if (scheme[global_index]->dense_tiles_gpu[i].x == nullptr || 
            scheme[global_index]->dense_tiles[i].elements == nullptr) {
            printf("Error: Null pointer at tile %d\n", i);
            continue;
        }


        cudaMemcpy(
            scheme[global_index]->dense_tiles[i].elements,  // Destination (Host)
            scheme[global_index]->dense_tiles_gpu[i].x,     // Source (Device)
            scheme[global_index]->dense_tiles[i].width * scheme[global_index]->dense_tiles[i].height * sizeof(double), 
            cudaMemcpyDeviceToHost
        );
    }

    // Ensure all memory transfers complete before function exits
    cudaDeviceSynchronize();
}

//INTERNAL_UPDATE_X_APPROACH_2_PHASE_1_INV_GPU_TO_CPU_serial
inline void SyncInverseResult_GpuToCpu_Serial(int global_index, TiledMatrix **scheme) {

    // Ensure correct GPU is set
    cudaError_t err = cudaSetDevice(scheme[global_index]->GPU_ID);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed for GPU %d: %s\n", scheme[global_index]->GPU_ID, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < scheme[global_index]->numActiveTiles; i++) {
        if (scheme[global_index]->inverse_tiles_gpu[i].x == nullptr || 
            scheme[global_index]->inverse_tiles[i].elements == nullptr) {
            printf("Error: Null pointer at tile %d\n", i);
            continue;
        }


        cudaMemcpy(
            scheme[global_index]->inverse_tiles[i].elements,  // Destination (Host)
            scheme[global_index]->inverse_tiles_gpu[i].x,     // Source (Device)
            scheme[global_index]->inverse_tiles[i].width * scheme[global_index]->inverse_tiles[i].height * sizeof(double), 
            cudaMemcpyDeviceToHost
        );
    }

    // Ensure all memory transfers complete before function exits
    cudaDeviceSynchronize();
}

inline void INTERNAL_UPDATE_X_APPROACH_2_PHASE_1_GPU_TO_CPU_parallel(int global_index, TiledMatrix **scheme) {

    // Set device for the main thread
    cudaError_t err = cudaSetDevice(scheme[global_index]->GPU_ID);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed for GPU %d: %s\n", scheme[global_index]->GPU_ID, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for num_threads(scheme[global_index]->num_cores)
    for (int i = 0; i < scheme[global_index]->numActiveTiles; i++) {
        // Set GPU for each OpenMP thread

        if (scheme[global_index]->dense_tiles_gpu[i].x == nullptr || 
            scheme[global_index]->dense_tiles[i].elements == nullptr) {
            printf("Error: Null pointer at tile %d\n", i);
            continue;
        }

        // Choose a stream using round-robin
        int stream_id = i % scheme[global_index]->num_cores;

        // Perform asynchronous memory copy
        cudaError_t err = cudaMemcpyAsync(
            scheme[global_index]->dense_tiles[i].elements,  // Destination (Host)
            scheme[global_index]->dense_tiles_gpu[i].x,     // Source (Device)
            scheme[global_index]->dense_tiles[i].width * scheme[global_index]->dense_tiles[i].height * sizeof(double), 
            cudaMemcpyDeviceToHost, 
            scheme[global_index]->streams[stream_id]
        );

        if (err != cudaSuccess) {
            printf("cudaMemcpyAsync failed for tile %d: %s\n", i, cudaGetErrorString(err));
        }
    }

    // Wait for all asynchronous memory transfers to complete
    for (int i = 0; i < scheme[global_index]->num_cores; i++) {
        cudaStreamSynchronize(scheme[global_index]->streams[i]);
    }

    // Final device-wide synchronization to ensure consistency
    cudaDeviceSynchronize();
}

#endif // GPU_MEMORY_FOR_COMPUTE


