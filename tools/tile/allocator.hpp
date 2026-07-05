/**
 * @file cpu_allocator.hpp
 * @brief CPU memory allocation utilities for dense tiles.
 *
 * Provides CPU-side memory allocation functions for dense tile elements
 * and inverse tile storage with support for group-based memory tracking.
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

#pragma once

#include <cstdio>
#include <cstdlib>

#include "../common/stiles_structs.hpp"
#include "../memory/MemoryManager.hpp"

#ifdef STILES_GPU
#include "../gpu/GpuMemoryManager.hpp"
#endif

namespace sTiles{ 


// void allocate_dense_tile_elements_cpu(TiledMatrix **scheme, int temp_counter, int width, int height, int group_index){
    
//     (*scheme)->dense_tiles[temp_counter].row = temp_counter;
//     (*scheme)->dense_tiles[temp_counter].col = temp_counter;
//     (*scheme)->dense_tiles[temp_counter].width = width;
//     (*scheme)->dense_tiles[temp_counter].height = height;

//     (*scheme)->dense_tiles[temp_counter].elements = MemoryManager::allocate<double>(width * height);
// }

// void allocate_inverse_dense_tile_elements_cpu(TiledMatrix **scheme, int temp_counter, int width, int height, bool is_diagonal, int group_index){

//     (*scheme)->inverse_tiles[temp_counter].row = temp_counter;
//     (*scheme)->inverse_tiles[temp_counter].col = temp_counter;
//     (*scheme)->inverse_tiles[temp_counter].width = width;
//     (*scheme)->inverse_tiles[temp_counter].height = height;
//     (*scheme)->inverse_tiles[temp_counter].elements = MemoryManager::allocate<double>(width * height);

//     if ((*scheme)->inverse_tiles[temp_counter].elements == NULL) {
//         fprintf(stderr, "Memory allocation failed for inverse_tiles[%d].elements.\n", temp_counter);
//         return;
//     }

//     if (is_diagonal) {
//         for (int ii = 0; ii < height; ii++) {
//             for (int jj = 0; jj < width; jj++) {
//                 (*scheme)->inverse_tiles[temp_counter].elements[ii * width + jj] =
//                     (ii == jj) ? 1.0 : 0.0;
//             }
//         }
//     } else {
//         LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', height, width, 0.0, 0.0, (*scheme)->inverse_tiles[temp_counter].elements, height);
//     }


// }


// int preprocess_sparse_dense_tiles(sTiles_call **call_info, TiledMatrix **scheme, int call_index, int group_index) { 


//     if((*scheme)->use_ordering==4) (*call_info)->use_nested_dissection = true;

//     //-----------------------------------------------------------------------

//     (*scheme)->inverse_tiles = nullptr;  // Initialize to null
//     (*scheme)->saved_tiles = nullptr;  // Initialize to null
//     (*scheme)->dense_tiles = TileMemoryManager::allocate<DenseTileSafeMode>((*scheme)->numActiveTiles, group_index);
//     (*scheme)->rhs_tiles = NULL;

//     if ((*scheme)->compute_inverse) {
//         (*scheme)->inverse_tiles = TileMemoryManager::allocate<DenseTileSafeMode>((*scheme)->numActiveTiles, group_index);
//         (*scheme)->saved_tiles = TileMemoryManager::allocate<DenseTileSafeMode>((*scheme)->numActiveTiles, group_index);
//     }

//     if ((*scheme)->dense_tiles == NULL) {
//         fprintf(stderr, "ERROR: Memory allocation failed for dense_tiles.\n");
//         exit(1);
//     }

//     if ((*scheme)->compute_inverse && (*scheme)->inverse_tiles == NULL) {
//         fprintf(stderr, "ERROR: Memory allocation failed for inverse_tiles.\n");
//         exit(1);
//     }

//     if(call_index==0){
//         (*scheme)->tileIndexMapper = TileMemoryManager::allocateZero<int>((*scheme)->triangular_size, group_index);
//         if ((*scheme)->tileIndexMapper == NULL) {
//             fprintf(stderr, "ERROR: Memory allocation failed for tileIndexMapper.\n");
//             exit(1);
//         }
//     }

//     (*scheme)->remainderTileSize = ((*scheme)->original_order % (*scheme)->tile_size == 0) ? (*scheme)->tile_size : (*scheme)->original_order % (*scheme)->tile_size;


//     #ifdef STILES_GPU

//         if((*scheme)->use_gpu){
//             // Set the active GPU.
//             cudaSetDevice((*scheme)->GPU_ID);

//             (*scheme)->inverse_tiles_gpu = nullptr;  // Initialize to null
//             (*scheme)->dense_tiles_gpu = TileMemoryManager::allocate<DenseGpuTile>((*scheme)->numActiveTiles, group_index);
//             if ((*scheme)->dense_tiles_gpu == NULL) {
//                 fprintf(stderr, "ERROR: Memory allocation failed for dense_tiles_gpu.\n");
//                 exit(1);
//             }

//             if ((*scheme)->compute_inverse_on_gpu) {
//                 (*scheme)->inverse_tiles_gpu = TileMemoryManager::allocate<DenseGpuTile>((*scheme)->numActiveTiles, group_index);
//             }

//             if ((*scheme)->compute_inverse_on_gpu && (*scheme)->inverse_tiles_gpu == NULL) {
//                 fprintf(stderr, "ERROR: Memory allocation failed for inverse_tiles_gpu.\n");
//                 exit(1);
//             }

//         }

//     #endif


//     int temp_counter = 0, index = 0;

//     for (int j = 0; j < (*scheme)->dimTiledMatrix; j++) {
//         for (int i = 0; i <= j; i++) {

//             index = i * (2 * ((*scheme)->dimTiledMatrix) - i - 1) / 2 + j;

//             // Ensure index is within bounds of tileIndexMapper
//             if (index >= ((*scheme)->triangular_size)) {

//                 fprintf(stderr, "ERROR: Out-of-bounds access in tileIndexMapper at index %d (max: %d)\n",
//                         index, (*scheme)->triangular_size - 1);
//                 exit(1);
//             }

//             if ((*scheme)->permutation_flags[index]) {

//                 int width = (j == (*scheme)->dimTiledMatrix - 1) ? (*scheme)->remainderTileSize : (*scheme)->tile_size;
//                 int height = (i == (*scheme)->dimTiledMatrix - 1) ? (*scheme)->remainderTileSize : (*scheme)->tile_size;

//                 // Ensure temp_counter is within bounds
//                 if (temp_counter >= (*scheme)->numActiveTiles) {
//                     fprintf(stderr, "ERROR: Out-of-bounds access in dense_tiles at temp_counter %d (max: %d)\n", temp_counter, (*scheme)->numActiveTiles - 1);
//                     exit(1);
//                 }

//                 allocate_dense_tile_elements(scheme, temp_counter, width, height, group_index);
//                 #ifdef STILES_GPU
//                     if((*scheme)->use_gpu) {
//                         allocate_dense_tile_elements_gpu(scheme, temp_counter, width, height, group_index);  
//                     }                 
//                 #endif

//                 if ((*scheme)->dense_tiles[temp_counter].elements == NULL) {
//                     fprintf(stderr, "Memory allocation failed for dense_tiles[%d].elements.\n", temp_counter);
//                     return -1;
//                 }

//                 if ((*call_info)->compute_inverse) {
//                     allocate_inverse_tile_elements(scheme, temp_counter, width, height, i == j, group_index);
//                     #ifdef STILES_GPU
//                         if((*scheme)->compute_inverse_on_gpu) allocate_inverse_tile_elements_gpu(scheme, temp_counter, width, height, i == j, group_index);
//                     #endif

//                 }

//                 if (call_index == 0) {
//                     (*scheme)->tileIndexMapper[index] = temp_counter;
//                 }

//                 temp_counter++;
//             }
//         }
//     }


//     return 0;

// }


}
