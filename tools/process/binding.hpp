/**
 * @file binding.cpp
 *
 * @brief Implementation of core and NUMA node binding strategies for parallel execution in the sTiles framework.
 *
 * Designed and developed by:
 * - Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST).
 *
 * @version 1.0.0
 * @date 2025-01-30
 * @license Proprietary
 *
 * @note This file is part of the sTiles framework, a proprietary software package. 
 *       Redistribution or modification without prior permission is prohibited.
 *
 * @contact esmail.abdulfattah@kaust.edu.sa
 *
 * Copyright (c) 2025, Esmail Abdul Fattah, KAUST.
 * All rights reserved.
 *
 * @license
 * This software is proprietary and confidential. Unauthorized copying, distribution, or modification 
 * of this software, via any medium, is strictly prohibited. Permission is granted to use the software 
 * in binary form for non-commercial purposes only, provided that this copyright notice and permission 
 * notice are included in all copies or substantial portions of the software.
 *
 * DISCLAIMER:
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. 
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, 
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE 
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

 #include "stiles_process.h"
 #include "../common/core_lapack.hpp"
 #include "../memory/MemoryManager.hpp"
 
 /**
  * @brief Assigns cores to sTiles calls based on NUMA binding and the specified strategy.
  *
  * @param coresbind Pointer to the array of core bindings for the current call.
  * @param offset Starting index for core assignment.
  * @param total_threads Total available threads for binding.
  * @param call_index Index of the current sTiles call.
  * @param num_cores Number of cores to assign to the current call.
  * @param num_calls Total number of calls in the group.
  * @param strategy Core binding strategy (0 = default sequential binding).
  * @param numaenabled Whether NUMA binding is enabled.
  * @param count_numa_id Array containing the number of cores per NUMA node.
  * @param numa_matrix Matrix containing the cores per NUMA node.
  * @param numa_array Array of all cores across NUMA nodes.
  *
  * @return 1 on successful core binding.
  */
 int preprocessing_binding(int** coresbind, int offset, int total_threads, int call_index, int num_cores, int num_calls, int strategy, int numaenabled, int* count_numa_id, int** numa_matrix, int* numa_array) {
 
     if(numaenabled){
 
         if(strategy == 0) {
 
             if(num_calls==1){
 
                 if(strategy == 0) {
 
                     *coresbind = MemoryManager::allocateZero<int>(num_cores);
                     //offset = call_index * num_cores;
 
                     if(false && num_cores > count_numa_id[0] && count_numa_id[0]==count_numa_id[1] && num_cores < (count_numa_id[0]+count_numa_id[1])){
 
                         int half_cores1 = num_cores - num_cores/2;
                         int half_cores2 = num_cores/2;
                         int k =0;
 
                         for(int j = 0; j < half_cores1; j++) {
                             (*coresbind)[k++] = numa_matrix[0][j]; // Ensure k is incremented
                         }
 
                         for(int j = 0; j < half_cores2; j++) {
                             (*coresbind)[k++] = numa_matrix[1][j]; // Ensure k is incremented
                         }
 
 
                     }else{
                         for(int i = 0; i < num_cores; i++) 
                             (*coresbind)[i] = numa_array[(i + offset) % total_threads];
                     }
 
                     return 1;
 
                 }
 
             }else if(num_calls>1){
 
                     *coresbind = MemoryManager::allocateZero<int>(num_cores); 
                     //offset = call_index * num_cores;
                     for(int i = 0; i < num_cores; i++) 
                         (*coresbind)[i] = numa_array[(i + offset) % total_threads];
 
             }
 
 
             return 1;
         }
 
 
     }else{
 
         if(strategy == 0) {
 
             //*coresbind = (int *)calloc(num_cores, sizeof(int)); 
             //*offset = call_index * num_cores;
 
             if(num_cores>0) for(int i = 0; i < num_cores; i++) 
                 (*coresbind)[i] = (i + offset) % total_threads;
             
             return 1;
         }
 
     }
 
     return 1;
 
 }
 
 /**
  * @brief Determines how many core groups can fit within a NUMA node.
  *
  * @param core_groups Array of core group sizes.
  * @param num_groups Total number of core groups.
  * @param numa_size Number of cores available in the NUMA node.
  *
  * @return Number of core groups that can fit in the NUMA node.
  */
 int how_many_fit(int* core_groups, int num_groups, int numa_size) {
 
     int total_cores = 0;
     int groups_fit = 0;
 
     // Iterate over each core group and add the num_cores
     for (int i = 0; i < num_groups; i++) {
         total_cores += core_groups[i];
 
         // If the total exceeds the NUMA size, stop
         if (total_cores > numa_size) {
             break;
         }
 
         // Count how many stiles_groups can fit
         groups_fit++;
     }
 
     return groups_fit;
 
 }
 
 /**
  * @brief Distributes cores across multiple sTiles calls, supporting NUMA-enabled configurations.
  *
  * @param num_cores Total number of cores available for distribution.
  * @param num_calls Total number of calls to distribute cores across.
  * @param stiles_calls Array of sTiles_call structures, each representing a call.
  * @param max_cores_sys Maximum number of cores allowed.
  * @param numaenabled Whether NUMA binding is enabled.
  */
 void distribute_cores_to_calls(int num_cores, int num_calls, sTiles_call *stiles_calls, int max_cores_sys, int numaenabled) {
  
     if (num_calls == 0) {
         return;  // No stiles_calls to assign num_cores to
     }
 
     int cores_per_call = num_cores / num_calls;  // Base num_cores per call
     int remainder_cores = num_cores % num_calls; // Remaining num_cores to distribute
 
     // If we have fewer num_cores than stiles_calls, some stiles_calls will get zero num_cores
     if (num_cores < num_calls) {
         for (int i = 0; i < num_cores; i++) {
             stiles_calls[i].num_cores = 1;  // Assign one core to as many stiles_calls as possible
         }
         for (int i = num_cores; i < num_calls; i++) {
             stiles_calls[i].num_cores = 1;  // Remaining stiles_calls get zero num_cores
         }
     } else {
         // Assign the base number of num_cores to all stiles_calls
         for (int i = 0; i < num_calls; i++) {
             stiles_calls[i].num_cores = cores_per_call;
         }
 
         // Distribute the remainder num_cores, giving one extra core to some stiles_calls
         for (int i = 0; i < remainder_cores; i++) {
             stiles_calls[i].num_cores += 1;  // Distribute extra num_cores to the first few stiles_calls
         }
     }
 
     if (num_cores < num_calls) {
         
         int counter = 0;
         for (int i = 0; i < num_calls; i++) {
 
             stiles_calls[i].core_bind_ids = MemoryManager::allocateZero<int>(1);
             stiles_calls[i].core_bind_ids[0] = counter%num_cores;
             counter++;
 
         }
 
     }else{
 
         
         if(numaenabled){
 
             #ifdef NUMA_ENABLED 
                 int max_threads = std::max(omp_get_max_threads(), omp_get_num_procs());
                 int* numa_id = MemoryManager::allocateZero<int>(max_threads); 
                 int i, j;
                 int max_numa_id = 0;
 
                 for(i = 0; i < max_threads; i++) {
                 
                     numa_id[i] = numa_node_of_cpu(i);
                     //printf("numa_id[i] %d \n", numa_id[i]);
                     if(numa_id[i] > max_numa_id) max_numa_id = numa_id[i];
                 }
 
                 max_numa_id++;
 
                 int* count_numa_id = MemoryManager::allocateZero<int>(max_numa_id); 
                 for(i = 0; i < max_threads; i++) 
                     for(j = 0; j < max_numa_id; j++) 
                         if(numa_id[i] == j) 
                             count_numa_id[j]++;
                 int rows = max_numa_id; 
 
                 int** numa_matrix = (int **)malloc(rows * sizeof(int*));
                 for(int i = 0; i < rows; i++) {
                     numa_matrix[i] = (int *)malloc(count_numa_id[i] * sizeof(int));
                 }
                 for(i = 0; i < rows; i++) {
                     int counter = 0, ind = 0;
                     for(j = 0; j < max_threads; j++) {
                         if(numa_id[j] == i) {
                             numa_matrix[i][ind] = counter; 
                             ind++;
                         }
                         counter++;
                     }
                 }
 
                 /*printf("rows %d \n", rows);
                 for(i = 0; i < rows; i++) {
                     for(j = 0; j < count_numa_id[i]; j++) {
                         printf("%d, ", numa_matrix[i][j]);
                     }
                     printf("\n");
                 }*/
 
                 int total_elements = 0;
                 for (i = 0; i < max_numa_id; i++) {
                     total_elements += count_numa_id[i];
                 }
 
 
                 int* core_groups = MemoryManager::allocateZero<int>(num_calls);
                 for(i=0; i <num_calls;i++){
                     core_groups[i] = stiles_calls[i].num_cores;
                 }
 
                 int* groups_that_fit_per_row = MemoryManager::allocateZero<int>(rows);
                 int s_num_calls = num_calls;
                 for(i = 0; i < rows; i++) {
                     groups_that_fit_per_row[i] = how_many_fit(core_groups, s_num_calls, count_numa_id[i]);
                     //printf("%d core group(s) can fit on the NUMA node of size %d num_cores.\n", groups_that_fit_per_row[i], count_numa_id[i]);
                     s_num_calls -= groups_that_fit_per_row[i];
                     
                 }
 
                 bool split = false;
                 s_num_calls = 0;
                 for(i = 0; i < rows; i++) {
                     s_num_calls += groups_that_fit_per_row[i];
                 }
 
                 if(s_num_calls==num_calls) split = true;
 
                 int* numa_array = MemoryManager::allocateZero<int>(total_elements);
                 if(split){
 
                     bool** bool_numa_m = (bool **)calloc(rows, sizeof(bool*)); 
                     for (int i = 0; i < rows; i++) {
                         bool_numa_m[i] = (bool *)calloc(count_numa_id[i], sizeof(bool));  // Automatically initializes to false
                     }
 
                     int* cores_per_row = (int *)calloc(rows, sizeof(int)); 
                     int r = 0;
                     for(i = 0; i < rows; i++) {
                         
                         for(j =0; j<groups_that_fit_per_row[i];j++){
                             cores_per_row[i] += core_groups[r];
                             r++;
                         }
                         //printf("cores_per_row[%d] %d \n", i, cores_per_row[i]);
                     }
                     
                     r = 0;
                     for(i = 0; i < rows; i++) {
                         for(j = 0; j < cores_per_row[i]; j++) {
                             numa_array[r++] = numa_matrix[i][j]; // Ensure k is incremented
                             bool_numa_m[i][j] = true;
                         }
                     }
 
                     // Second pass: append each NUMA node's remaining (unassigned) CPUs.
                     // Bound by count_numa_id[i] (the size of numa_matrix[i]/bool_numa_m[i]).
                     // Using core_groups[i] here read OUT OF BOUNDS: core_groups is sized
                     // num_calls, but i ranges over NUMA nodes (rows). On machines with more
                     // NUMA nodes than calls (e.g. 8-NUMA Rome, single call) the garbage bound
                     // overflowed bool_numa_m[i]/numa_matrix[i] -> "free(): invalid next size".
                     for(i = 0; i < rows; i++) {
                         for(j = 0; j < count_numa_id[i]; j++) {
                             if(!bool_numa_m[i][j]) numa_array[r++] = numa_matrix[i][j];
                             bool_numa_m[i][j] = true;
                         }
                     }
 
 
 
                     for (int i = 0; i < rows; i++) {
                         free(bool_numa_m[i]);
                     }
                     free(bool_numa_m);
                     free(cores_per_row);
 
             }else{
 
                 int k=0;
                 for(i = 0; i < rows; i++) {
                     for(j = 0; j < count_numa_id[i]; j++) {
                         numa_array[k++] = numa_matrix[i][j]; // Ensure k is incremented
                     }
                 }
 
             }
 
             /*printf("NUMA Array:\n");
             for (int i = 0; i < total_elements; i++) {
                 printf("%d, ", numa_array[i]);
             }
             printf("\n");*/
 
             int offset = 0;
             for (i = 0; i < num_calls; i++) {
                 
                 stiles_calls[i].core_bind_ids = MemoryManager::allocateZero<int>(stiles_calls[i].num_cores);
                 preprocessing_binding(&stiles_calls[i].core_bind_ids, offset, num_cores, stiles_calls[i].call_index, stiles_calls[i].num_cores, num_calls, 0, numaenabled, count_numa_id, numa_matrix, numa_array);
                 offset += stiles_calls[i].num_cores;
             }
 
             // Free numa_matrix
             for (int i = 0; i < rows; i++) {
                 if (numa_matrix[i] != NULL) {
                     free(numa_matrix[i]);
                 }
             }
             free(numa_matrix);
             numa_matrix = NULL;
 
 
             // Free other allocated memory (use MemoryManager to untrack)
             MemoryManager::deallocate(numa_id);
             MemoryManager::deallocate(count_numa_id);
             MemoryManager::deallocate(core_groups);
             MemoryManager::deallocate(groups_that_fit_per_row);
             MemoryManager::deallocate(numa_array);
             #endif
         }else{
 
             int offset = 0;
             for (int i = 0; i < num_calls; i++) {
                 
                 stiles_calls[i].core_bind_ids = MemoryManager::allocateZero<int>(stiles_calls[i].num_cores);
                 preprocessing_binding(&stiles_calls[i].core_bind_ids, offset, num_cores, stiles_calls[i].call_index, stiles_calls[i].num_cores, num_calls, 0, numaenabled, NULL, NULL, NULL);
                 offset += stiles_calls[i].num_cores;
             }
         }
 
     } 
  
 }
 
 
 
 
