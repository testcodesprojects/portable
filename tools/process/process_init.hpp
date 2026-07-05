#ifndef STILES_PROCESS_INIT_H
#define STILES_PROCESS_INIT_H


#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <cassert>
#include <stdio.h>
#include <unistd.h>

#ifdef __APPLE__
#include <sys/sysctl.h> // Required for sysctlbyname on macOS
#endif

#include "../common/stiles_config.hpp"
#include "../common/stiles_utils.hpp"
#include "../common/stiles_logger.hpp"
#include "../tree/tree.hpp"
#include "binding.hpp"
#include "../memory/MemoryManager.hpp"
#include "../common/stiles_exporter.hpp"
#include "stiles_copying.hpp"
#include "../tile/sparse_dense_tiling.hpp"
#include "../memory/cpuSmartTileMemoryManager.hpp"
#include <stiles_process.h>
#include <stdio.h>


// int sTiles_preprocess_group_org(int group_index, sTiles_object **obj, TiledMatrix **stiles_schemes) {

//     if (obj == NULL || *obj == NULL) {
//         fprintf(stderr, "Error: obj is not properly initialized.\n");
//         return EXIT_FAILURE;
//     }

//     // Number of calls in the current group
//     int num_calls = (*obj)->stiles_groups[group_index].num_calls;
//     int save_global_index = (*obj)->stiles_groups[group_index].stiles_calls[0].global_index;

//     if (num_calls <= 0) {
//         fprintf(stderr, "Error: No calls to preprocess in the group.\n");
//         return EXIT_FAILURE;
//     }

//     // Loop through each call in the group
//     for (int call_index = 0; call_index < num_calls; call_index++) {

//         int alloc_status = -1, global_index = -1, copy_status = -1;

//         // Get the pointer to sTiles_call for the current call
//         sTiles_call *call_info = &(*obj)->stiles_groups[group_index].stiles_calls[call_index];
//         global_index = (*obj)->stiles_groups[group_index].stiles_calls[call_index].global_index;

//         // Synchronize fields for calls after the first
//         if (call_index != 0) {
//             (*obj)->stiles_groups[group_index].stiles_calls[call_index].order = (*obj)->stiles_groups[group_index].stiles_calls[0].order;
//             (*obj)->stiles_groups[group_index].stiles_calls[call_index].nnz = (*obj)->stiles_groups[group_index].stiles_calls[0].nnz;
//             (*obj)->stiles_groups[group_index].stiles_calls[call_index].row_indices = (*obj)->stiles_groups[group_index].stiles_calls[0].row_indices;
//             (*obj)->stiles_groups[group_index].stiles_calls[call_index].col_indices = (*obj)->stiles_groups[group_index].stiles_calls[0].col_indices;
//         }

//         #pragma omp barrier 

//         // Initialize a new scheme for the current call
//         TiledMatrix *scheme = NULL;
//         int status = sTiles_preprocess_initialization(&call_info, &scheme, call_index, group_index);
//         if (status != 0) {
//             fprintf(stderr, "Error: Failed to preprocess parameters for call %d.\n", call_index);
//             return EXIT_FAILURE;
//         }

//         // Store the scheme in the stiles_schemes array
//         stiles_schemes[global_index] = scheme;
//         stiles_schemes[global_index]->call_lookup_table = (*obj)->call_matrix;

//         // Perform symbolic factorization and memory allocation for the first call
//         if (call_index == 0) {

//         #ifdef STILES_SAFEMODE
//             auto sym_fact_status = sTiles::preprocess_symbolic_factorization_main_safe_mode(&call_info, &stiles_schemes[global_index], group_index);
//             if (sym_fact_status != sTiles::StatusCode::Success) {
//                 fprintf(stderr, "Error: Symbolic factorization failed for the first call.\n");
//                 return EXIT_FAILURE;
//             }
//         #elif defined(STILES_VERIFY_FASTMODE)
//             // TODO: implement fast-mode verification path.
//         #else
//             // TODO: implement default fast-mode path.
//         #endif

//         }

//         #pragma omp barrier 

//         #ifdef STILES_GPU

//             GPUInfo *gpuInfo = getGPUInfo(
//                                         stiles_schemes[global_index]->tile_size,
//                                         stiles_schemes[global_index]->num_cores,
//                                         stiles_schemes[global_index]->red_tree_separator_level,
//                                         stiles_schemes[global_index]->numActiveTiles
//                                     );
//             if(call_index==0) printGPUInfo(gpuInfo);


//             stiles_schemes[global_index]->use_gpu = configureGpuAllocation(gpuInfo, (*obj)->num_call_groups, (*obj)->num_calls_per_group, &stiles_schemes[global_index]->GPU_ID, call_index, &stiles_schemes[global_index]->compute_inverse_on_gpu);
        
//                 #ifdef DEBUG
//                     if(stiles_schemes[global_index]->use_gpu){
//                         std::cout << "GPU setup successful. Selected GPU_ID: " << stiles_schemes[global_index]->GPU_ID << std::endl;
//                     } else {
//                         std::cout << "GPU setup failed." << std::endl;
//                     }
//                 #endif

//             freeGPUInfo(gpuInfo);

//         #endif

//         if (call_index == 0){

//             alloc_status = sTiles_preprocess_sparse_dense_allocation_using_bol(&call_info, &stiles_schemes[global_index], call_index, group_index);
//             if (alloc_status != 0) {
//                 fprintf(stderr, "Error: Memory allocation failed for the first call.\n");
//                 return EXIT_FAILURE;
//             }

//         }else{

//             copy_status = sTiles_copy_configuration_1(&stiles_schemes[save_global_index], &stiles_schemes[global_index]);
//             if (copy_status != 0) {
//                 fprintf(stderr, "Error: Failed to copy configuration for call %d.\n", call_index);
//                 return EXIT_FAILURE;
//             }

//             alloc_status = sTiles_preprocess_sparse_dense_allocation_using_bol(&call_info, &stiles_schemes[global_index], call_index, group_index);
//             if (alloc_status != 0) {
//                 fprintf(stderr, "Error: Memory allocation failed for the first call.\n");
//                 return EXIT_FAILURE;
//             }

//         }

//         #pragma omp barrier 

//         if (call_index == 0){
        
//             sTiles_preprocess_sparse_dense_tree_phase_0_using_bool_safe_mode(&call_info, &stiles_schemes[global_index], group_index);

//         }else{

//             copy_status = sTiles_copy_configuration_2(&stiles_schemes[save_global_index], &stiles_schemes[global_index]);
//             if (copy_status != 0) {
//                 fprintf(stderr, "Error: Failed to copy configuration for call %d.\n", call_index);
//                 return EXIT_FAILURE;
//             }
            
//         }

//         #pragma omp barrier 

//         if (call_index > 0){

//             copy_status = sTiles_copy_configuration_3(&stiles_schemes[save_global_index], &stiles_schemes[global_index]);
//             if (copy_status != 0) {
//                 fprintf(stderr, "Error: Failed to copy configuration for call %d.\n", call_index);
//                 return EXIT_FAILURE;
//             }

//         }

//         sTiles_preprocess_sparse_dense_tree_phase_1_using_bool_safe_mode(&call_info, &stiles_schemes[global_index], group_index);
//         collect_tasks(&call_info, &stiles_schemes[global_index], group_index, num_threads_level1);
//         sTiles_preprocess_sparse_dense_allocation_copy_using_bol(&call_info, &stiles_schemes[global_index]);

//         if (call_index == 0){
        
//             sTiles_update_x_sparse_dense_tiles_phase_0(&call_info, stiles_schemes[global_index], group_index);

//         }

//         #pragma omp barrier 

//         if (call_index > 0){

//             copy_status = sTiles_copy_configuration_4(&stiles_schemes[save_global_index], &stiles_schemes[global_index]);
//             if (copy_status != 0) {
//                 fprintf(stderr, "Error: Failed to copy configuration for call %d.\n", call_index);
//                 return EXIT_FAILURE;
//             }

//         }

//         #ifdef STILES_GPU
//             if(stiles_schemes[global_index]->use_gpu) allocate_for_tree_and_streams_gpu(&stiles_schemes[global_index], group_index, global_index);
//         #endif


//     }
//     return EXIT_SUCCESS;

// }

//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//___________________________________________________________________________ONE_CALL___________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________

// int sTiles_preprocess_call(int group_index, int call_index, sTiles_object **obj, TiledMatrix **stiles_schemes) {

//     if (obj == NULL || *obj == NULL) {
//         fprintf(stderr, "Error: obj is not properly initialized.\n");
//         return EXIT_FAILURE;
//     }

//     // Number of calls in the current group
//     int num_calls = (*obj)->stiles_groups[group_index].num_calls;
//     if (num_calls <= 0) {
//         fprintf(stderr, "Error: No calls to preprocess in the group.\n");
//         return EXIT_FAILURE;
//     }

//     sTiles_call *call_info = &(*obj)->stiles_groups[group_index].stiles_calls[call_index];
//     int global_index = (*obj)->stiles_groups[group_index].stiles_calls[call_index].global_index;

//     // Initialize a new scheme for the current call
//     TiledMatrix *scheme = NULL;

//     int status = sTiles_preprocess_initialization(&call_info, &scheme, call_index, group_index);
//     if (status != 0) {
//         fprintf(stderr, "Error: Failed to preprocess parameters for call %d.\n", call_index);
//         return EXIT_FAILURE;
//     }

//     // Store the scheme in the stiles_schemes array
//     stiles_schemes[global_index] = scheme;
//     stiles_schemes[global_index]->call_lookup_table = (*obj)->call_matrix;

//     #ifdef STILES_SAFEMODE
//     auto sym_fact_status = sTiles::preprocess_symbolic_factorization_main_safe_mode(&call_info, &stiles_schemes[global_index], group_index);
//     if (sym_fact_status != sTiles::StatusCode::Success) {
//         fprintf(stderr, "Error: Symbolic factorization failed for the first call.\n");
//         return EXIT_FAILURE;
//     }
//     #elif defined(STILES_VERIFY_FASTMODE)
//         // TODO: implement fast-mode verification path.
//     #else
//         // TODO: implement default fast-mode path.
//     #endif

//     #ifdef STILES_GPU

//         GPUInfo *gpuInfo = getGPUInfo(
//             stiles_schemes[global_index]->tile_size,
//             stiles_schemes[global_index]->num_cores,
//             stiles_schemes[global_index]->red_tree_separator_level,
//             stiles_schemes[global_index]->numActiveTiles
//         );
//         if(call_index==0) printGPUInfo(gpuInfo);


//         stiles_schemes[global_index]->use_gpu = configureGpuAllocation(gpuInfo, (*obj)->num_call_groups, (*obj)->num_calls_per_group, &stiles_schemes[global_index]->GPU_ID, call_index, &stiles_schemes[global_index]->compute_inverse_on_gpu);
    
//         #ifdef DEBUG
//             if(stiles_schemes[global_index]->use_gpu){
//                 std::cout << "GPU setup successful. Selected GPU_ID: " << stiles_schemes[global_index]->GPU_ID << std::endl;
//             } else {
//                 std::cout << "GPU setup failed." << std::endl;
//             }
//         #endif

//         freeGPUInfo(gpuInfo);

//     #endif

//     int alloc_status = sTiles_preprocess_sparse_dense_allocation_using_bol(&call_info, &stiles_schemes[global_index], 0, group_index);
//     if (alloc_status != 0) {
//         fprintf(stderr, "Error: Memory allocation failed for the first call.\n");
//         return EXIT_FAILURE;
//     }

//         sTiles_preprocess_sparse_dense_tree_phase_0_using_bool_safe_mode(&call_info, &stiles_schemes[global_index], group_index);
//         sTiles_preprocess_sparse_dense_tree_phase_1_using_bool_safe_mode(&call_info, &stiles_schemes[global_index], group_index);
//         collect_tasks(&call_info, &stiles_schemes[global_index], group_index, num_threads_level1);
//         sTiles_preprocess_sparse_dense_allocation_copy_using_bol(&call_info, &stiles_schemes[global_index]);


//     sTiles_update_x_sparse_dense_tiles_phase_0(&call_info, stiles_schemes[global_index], group_index);

//     #ifdef STILES_GPU
//         if(stiles_schemes[global_index]->use_gpu) allocate_for_tree_and_streams_gpu(&stiles_schemes[global_index], group_index, global_index);
//     #endif
    
//     return EXIT_SUCCESS;

// }

#endif
