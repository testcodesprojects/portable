#pragma once

/**
 * @file tree.hpp
 *
 * @brief Implementation of tree structures and utility routines for the sTiles framework.
 *
 * This file defines and implements the construction, initialization, and management of tree structures 
 * used in the sTiles framework for hierarchical computations. These tree structures are designed to 
 * optimize matrix factorization processes, such as Cholesky factorization, by leveraging tree-based 
 * dependency management and task scheduling. The primary functionalities include:
 * - Creation of hierarchical binary trees for computational nodes.
 * - Support for reduced and specialized tree configurations to handle sparsity and custom tile setups.
 * - Utility functions for memory management, tree traversal, and GEMM operation counting.
 *
 * Key Functions:
 * - `createTree`: Constructs a binary tree for handling specified GEMM operations.
 * - `createRedTree`: Constructs a reduced tree structure optimized for fewer GEMM operations.
 * - `freeTree`: Releases memory allocated for a tree structure.
 * - `printTreeInfo`: Displays information about a tree's structure and nodes.
 * - `count_gemms_*`: Counts and tracks GEMM operations based on tree configurations.
 * - `TREE_SETUP_STG2_STILES_*`: Advanced routines for tree setup and processing tailored for specific 
 *   sparsity and tile-based configurations.
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


// #include "stiles_process.h"
// #include "core_lapack.hpp"
// #include "sparse_utils.hpp"
#include "../memory/TreeMemoryManager.hpp"
#include "../common/stiles_logger.hpp"
#include "../tree/tree_structs.hpp"


/**
 * @brief Creates a binary tree structure for managing GEMM operations.
 * @param num_gemm_operations Number of GEMM operations the tree should handle.
 * @param leafheight Height of each leaf node in the tree.
 * @param leafwidth Width of each leaf node in the tree.
 * @return Pointer to the created `TreeLeaf` structure.
 */
TreeLeaf* createOptimizedTree(int num_splits, int num_tasks, int leafheight, int leafwidth, int group_index) {
    if (num_splits <= 0 || num_tasks <= 0) {
        fprintf(stderr, "Invalid input: num_splits=%d, num_tasks=%d\n", num_splits, num_tasks);
        return nullptr;
    }

    TreeLeaf *tree = TreeMemoryManager::allocate<TreeLeaf>(1, group_index);
    if (!tree) {
        fprintf(stderr, "Failed to allocate memory for TreeLeaf.\n");
        return nullptr;
    }

    tree->num_splits = num_splits;
    tree->num_tasks = num_tasks;
    
    // Allocate nodes
    tree->nodes = TreeMemoryManager::allocate<NodeLeaf>(num_splits, group_index);
    if (!tree->nodes) {
        fprintf(stderr, "Failed to allocate memory for nodes.\n");
        TreeMemoryManager::deallocate(tree);  // Free tree before returning
        return nullptr;
    }

    // Allocate dependency array
    tree->dependency = TreeMemoryManager::allocateZero<int>(num_splits, group_index);
    if (!tree->dependency) {
        fprintf(stderr, "Failed to allocate memory for dependency.\n");
        TreeMemoryManager::deallocate(tree->nodes);
        TreeMemoryManager::deallocate(tree);
        return nullptr;
    }

    // Allocate `x` inside each node
    for (int i = 0; i < num_splits; ++i) {
        tree->nodes[i].index = i;
        tree->nodes[i].x = TreeMemoryManager::allocateZero<double>(leafheight * leafwidth, group_index);
        if (!tree->nodes[i].x) {
            fprintf(stderr, "Failed to allocate memory for x at index %d.\n", i);
            
            // Free already allocated memory before returning
            for (int j = 0; j < i; ++j) {
                TreeMemoryManager::deallocate(tree->nodes[j].x);
            }
            TreeMemoryManager::deallocate(tree->dependency);
            TreeMemoryManager::deallocate(tree->nodes);
            TreeMemoryManager::deallocate(tree);
            return nullptr;
        }

        tree->nodes[i].leafwidth = leafwidth;
        tree->nodes[i].leafheight = leafheight;
    }

    return tree;
}


/**
 * @brief Creates a binary tree structure for managing GEMM operations.
 * @param num_gemm_operations Number of GEMM operations the tree should handle.
 * @param leafheight Height of each leaf node in the tree.
 * @param leafwidth Width of each leaf node in the tree.
 * @return Pointer to the created `TreeLeaf` structure.
 */
TreeLeaf* createTree(int num_gemm_operations, int leafheight, int leafwidth, int group_index) {

    // Calculate max_levels assuming a binary tree structure
    int levels = 0;
    int temp = num_gemm_operations;
    while (temp > 0) {
        levels++;
        temp /= 2;
    }

    TreeLeaf *tree = (TreeLeaf*)malloc(sizeof(TreeLeaf));
    if (!tree) {
        fprintf(stderr, "Failed to allocate memory for tree.\n");
        exit(EXIT_FAILURE);
    }

    tree->max_levels = levels;
    tree->gold_number = (int*)calloc(levels, sizeof(int));
    tree->half_gold = (int*)calloc(levels, sizeof(int));
    if (!tree->gold_number || !tree->half_gold) {
        fprintf(stderr, "Failed to allocate memory for gold_number and half_gold.\n");
        exit(EXIT_FAILURE);
    }

    tree->gold_number[0] = 2;
    tree->half_gold[0] = tree->gold_number[0]/2;  
    for (int i = 1; i < tree->max_levels; ++i) {
        tree->gold_number[i] = tree->gold_number[i-1]*2;
        tree->half_gold[i] = tree->gold_number[i]/2;
    }

    tree->max_nodes = (int*)malloc(levels * sizeof(int));
    if (!tree->max_nodes) {
        fprintf(stderr, "Failed to allocate memory for max_nodes.\n");
        exit(EXIT_FAILURE);
    }

    tree->num_nodes = num_gemm_operations;
    tree->counter_nodes = 0;

    tree->silver_number = 0;
    temp = num_gemm_operations;
    for (int i = 0; i < levels; ++i) {
        tree->max_nodes[i] = temp;
        tree->silver_number += temp;
        temp = (temp + 1) / 2;
    }

    tree->nodes = (NodeLeaf*)malloc(num_gemm_operations * sizeof(NodeLeaf));
    if (!tree->nodes) {
        fprintf(stderr, "Failed to allocate memory for nodes.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_gemm_operations; ++i) {
        tree->nodes[i].index = i;
        tree->nodes[i].level = 0;
        tree->nodes[i].x = (double*)calloc(leafheight * leafwidth, sizeof(double));
        tree->nodes[i].leafwidth = leafwidth;
        tree->nodes[i].leafheight = leafheight;

        if (!tree->nodes[i].x) {
            fprintf(stderr, "Failed to allocate memory for x at index %d.\n", i);
            exit(EXIT_FAILURE);
        }
    }

    tree->dependency = (int*)calloc(tree->max_nodes[0], sizeof(int));
    if (!tree->dependency) {
        fprintf(stderr, "Failed to allocate memory for dependency.\n");
        exit(EXIT_FAILURE);
    }

    return tree;
}


/**
 * @brief Creates a reduced tree structure optimized for fewer GEMM operations.
 * @param new_tiles Number of new tiles to initialize in the tree.
 * @param num_gemm_operations Number of GEMM operations the reduced tree should handle.
 * @param leafheight Height of each leaf node in the tree.
 * @param leafwidth Width of each leaf node in the tree.
 * @return Pointer to the created reduced `TreeLeaf` structure.
 */
TreeLeaf* createRedTree(int new_tiles, int num_gemm_operations, int leafheight, int leafwidth) {

    TreeLeaf *tree = (TreeLeaf*)malloc(sizeof(TreeLeaf));
    if (!tree) {
        fprintf(stderr, "Failed to allocate memory for tree.\n");
        exit(EXIT_FAILURE);
    }

    tree->max_levels = 0;
    tree->gold_number = NULL;
    tree->half_gold = NULL;
 
    tree->num_nodes = 0;
    tree->counter_nodes = 0;
    tree->silver_number = 0;
    tree->max_nodes = (int*)malloc(1 * sizeof(int));
    tree->max_nodes[0] = num_gemm_operations;


    tree->nodes = (NodeLeaf*)malloc(new_tiles * sizeof(NodeLeaf));
    if (!tree->nodes) {
        fprintf(stderr, "Failed to allocate memory for nodes.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < new_tiles; ++i) {
        tree->nodes[i].x = (double*)calloc(leafheight * leafwidth, sizeof(double));
        if (!tree->nodes[i].x) {
            fprintf(stderr, "Failed to allocate memory for x at index %d.\n", i);
            exit(EXIT_FAILURE);
        }
    }

    tree->dependency = (int*)calloc(new_tiles, sizeof(int));
    if (!tree->dependency) {
        fprintf(stderr, "Failed to allocate memory for dependency.\n");
        exit(EXIT_FAILURE);
    }

    return tree;
}

/**
 * @brief Frees memory allocated for a tree structure.
 * @param tree Pointer to the `TreeLeaf` structure to be freed.
 */
void freeTree(TreeLeaf *tree) {

    if (!tree) {
        return; // Nothing to free
    }

    // Free memory for nodes
    if (tree->nodes) {
        for (int i = 0; i < tree->num_nodes; ++i) {
            if (tree->nodes[i].x) {
                free(tree->nodes[i].x); // Free memory for x in each node
                tree->nodes[i].x = NULL;
            }
        }
        free(tree->nodes); // Free the array of nodes
        tree->nodes = NULL;
    }

    // Free memory for gold_number and half_gold
    if (tree->gold_number) {
        free(tree->gold_number);
        tree->gold_number = NULL;
    }

    if (tree->half_gold) {
        free(tree->half_gold);
        tree->half_gold = NULL;
    }

    // Free memory for max_nodes
    if (tree->max_nodes) {
        free(tree->max_nodes);
        tree->max_nodes = NULL;
    }

    // Free memory for dependency
    if (tree->dependency) {
        free(tree->dependency);
        tree->dependency = NULL;
    }

    // Finally, free the TreeLeaf structure itself
    free(tree);
}


/**
 * @brief Prints detailed information about a tree structure.
 * @param tree Pointer to the `TreeLeaf` structure to inspect.
 */
void printTreeInfo(const TreeLeaf *tree) {

    if (tree == NULL) {
        printf("Tree is NULL.\n");
        return;
    }

    printf("Total Levels in the Tree: %d\n", tree->max_levels);
    for (int i = 0; i < tree->max_levels; ++i) {
        printf("Level %d has %d node(s).\n", i, tree->max_nodes[i]);
    }
}

/**
 * @brief Counts GEMM operations in a reduced tree structure.
 * @param tiles Array to store GEMM counts.
 * @param on_off_tiles 2D array indicating active tiles.
 * @param total_num_used_tiles Total number of tiles used in the computation.
 * @param sep Separation parameter for tile indexing.
 */
void count_gemms_redtree(int* tiles, bool **on_off_tiles, int total_num_used_tiles, int sep){
    
    int k, m, n;
    int next_k;
    int next_m;
    int next_n;
    int tm1, tm2;

    n = 0;
    m = total_num_used_tiles-sep;
    k = total_num_used_tiles-sep;

    while (k < total_num_used_tiles && m < total_num_used_tiles) {

        next_n = n;
        next_m = m;
        next_k = k;

        next_n++;
        if (next_n > next_k) {
            next_m += 1;
            while (next_m >= total_num_used_tiles && next_k < total_num_used_tiles) {
                next_k++;
                next_m = next_m-total_num_used_tiles+next_k;
            }
            next_n = 0;
        }
    
        if (m == k) {
            if (n != k){
                
                if(on_off_tiles[n][k]){

                    tm1 = k - total_num_used_tiles + sep;
                    tiles[tm1*(2*sep-tm1-1)/2 + tm1] += 1;

                }
            }

        }else {

            if (n != k) {

                if(on_off_tiles[n][k] && on_off_tiles[n][m]){

                    tm1 = k - total_num_used_tiles + sep;
                    tm2 = m - total_num_used_tiles + sep;
                    tiles[tm1*(2*sep-tm1-1)/2 + tm2] += 1;
                            
                }
            }
        }

        n = next_n;
        m = next_m;
        k = next_k;

    }


}

/**
 * @brief Counts GEMM operations using a single-dimensional tile array.
 * @param tiles Array to store GEMM counts.
 * @param on_off_tiles Array indicating active tiles.
 * @param total_num_used_tiles Total number of tiles used in the computation.
 * @param sep Separation parameter for tile indexing.
 */
void count_gemms_redtree_stiles(int* tiles, bool *on_off_tiles, int total_num_used_tiles, int sep){
    
    int k, m, n;
    int next_k;
    int next_m;
    int next_n;
    int tm1, tm2;

    n = 0;
    m = total_num_used_tiles-sep;
    k = total_num_used_tiles-sep;

    while (k < total_num_used_tiles && m < total_num_used_tiles) {

        next_n = n;
        next_m = m;
        next_k = k;

        next_n++;
        if (next_n > next_k) {
            next_m += 1;
            while (next_m >= total_num_used_tiles && next_k < total_num_used_tiles) {
                next_k++;
                next_m = next_m-total_num_used_tiles+next_k;
            }
            next_n = 0;
        }
    
        if (m == k) {
            if (n != k) {
                
                if(on_off_tiles[n*(2*total_num_used_tiles-n-1)/2 + k]){   

                    tm1 = k - total_num_used_tiles + sep;
                    tiles[tm1*(2*sep-tm1-1)/2 + tm1] += 1;

                }
            }
        
        }else {

            if (n != k) {

                if(on_off_tiles[n*(2*total_num_used_tiles-n-1)/2 + k] && on_off_tiles[n*(2*total_num_used_tiles-n-1)/2 + m]){ 

                    tm1 = k - total_num_used_tiles + sep;
                    tm2 = m - total_num_used_tiles + sep;
                    tiles[tm1*(2*sep-tm1-1)/2 + tm2] += 1;
                            
                }
               
            }
        }

        
        n = next_n;
        m = next_m;
        k = next_k;

    }

}

void count_gemms_redtree_stiles_for_group_call(int** tiles, bool *on_off_tiles, int total_num_used_tiles, int sep){
    
    int k, m, n;
    int next_k;
    int next_m;
    int next_n;
    int tm1, tm2;

    n = 0;
    m = total_num_used_tiles-sep;
    k = total_num_used_tiles-sep;

    while (k < total_num_used_tiles && m < total_num_used_tiles) {

        next_n = n;
        next_m = m;
        next_k = k;

        next_n++;
        if (next_n > next_k) {
            next_m += 1;
            while (next_m >= total_num_used_tiles && next_k < total_num_used_tiles) {
                next_k++;
                next_m = next_m-total_num_used_tiles+next_k;
            }
            next_n = 0;
        }
    
        if (m == k) {
            if (n != k) {
                
                if(on_off_tiles[n*(2*total_num_used_tiles-n-1)/2 + k]){   

                    tm1 = k - total_num_used_tiles + sep;
                    (*tiles)[tm1*(2*sep-tm1-1)/2 + tm1] += 1;

                }
            }
        
        }else {

            if (n != k) {

                if(on_off_tiles[n*(2*total_num_used_tiles-n-1)/2 + k] && on_off_tiles[n*(2*total_num_used_tiles-n-1)/2 + m]){ 

                    tm1 = k - total_num_used_tiles + sep;
                    tm2 = m - total_num_used_tiles + sep;
                    (*tiles)[tm1*(2*sep-tm1-1)/2 + tm2] += 1;
                            
                }
               
            }
        }

        
        n = next_n;
        m = next_m;
        k = next_k;

    }

}


// Fast-mode variant: pass a predicate that answers whether tile (i,j)
// is active (neighbors) in the upper triangle (i<=j) for a grid of size N.
// Example predicate for TileIndexer fast path:
//   auto is_on = [&](int a, int b){
//       if (a > b) std::swap(a,b);
//       return scheme_fast->state.isActive(a, b, scheme_fast->dimTiledMatrix);
//   };
//   count_gemms_redtree_stiles_for_group_call_fast(&tiles, is_on, scheme_fast->dimTiledMatrix, sep);
template <typename IsOn>
inline void count_gemms_redtree_stiles_for_group_call_fast(int** tiles, IsOn is_on, int total_num_used_tiles, int sep){
    int k, m, n;
    int next_k;
    int next_m;
    int next_n;
    int tm1, tm2;

    n = 0;
    m = total_num_used_tiles - sep;
    k = total_num_used_tiles - sep;

    while (k < total_num_used_tiles && m < total_num_used_tiles) {
        next_n = n;
        next_m = m;
        next_k = k;

        next_n++;
        if (next_n > next_k) {
            next_m += 1;
            while (next_m >= total_num_used_tiles && next_k < total_num_used_tiles) {
                next_k++;
                next_m = next_m - total_num_used_tiles + next_k;
            }
            next_n = 0;
        }

        if (m == k) {
            if (n != k) {
                if (is_on(n, k)) {
                    tm1 = k - total_num_used_tiles + sep;
                    (*tiles)[tm1 * (2 * sep - tm1 - 1) / 2 + tm1] += 1;
                }
            }
        } else {
            if (n != k) {
                if (is_on(n, k) && is_on(n, m)) {
                    tm1 = k - total_num_used_tiles + sep;
                    tm2 = m - total_num_used_tiles + sep;
                    (*tiles)[tm1 * (2 * sep - tm1 - 1) / 2 + tm2] += 1;
                }
            }
        }

        n = next_n;
        m = next_m;
        k = next_k;
    }
}


/**
 * @brief Counts GEMM operations using hash-bit optimizations.
 * @param tiles Array to store GEMM counts.
 * @param bit_array Bit-array representing tile dependencies.
 * @param total_num_used_tiles Total number of tiles used in the computation.
 * @param sep Separation parameter for tile indexing.
 */
void count_gemms_redtree_stiles_hash_bit(int* tiles, unsigned char *bit_array, int total_num_used_tiles, int sep){
    
    int k, m, n;
    int next_k;
    int next_m;
    int next_n;
    int tm1, tm2;

    n = 0;
    m = total_num_used_tiles-sep;
    k = total_num_used_tiles-sep;

    while (k < total_num_used_tiles && m < total_num_used_tiles) {

        next_n = n;
        next_m = m;
        next_k = k;

        next_n++;
        if (next_n > next_k) {
            next_m += 1;
            while (next_m >= total_num_used_tiles && next_k < total_num_used_tiles) {
                next_k++;
                next_m = next_m-total_num_used_tiles+next_k;
            }
            next_n = 0;
        }
    
        if (m == k) {
            if (n != k) {

                if (stiles_checkBitNeighbor(bit_array, n, k, total_num_used_tiles)){

                    tm1 = k - total_num_used_tiles + sep;
                    tiles[tm1*(2*sep-tm1-1)/2 + tm1] += 1;

                }
            }
        }
        else {
            if (n != k) {

                if(stiles_checkBitNeighbor(bit_array, n, k, total_num_used_tiles) && stiles_checkBitNeighbor(bit_array, n, m, total_num_used_tiles)){ 

                    tm1 = k - total_num_used_tiles + sep;
                    tm2 = m - total_num_used_tiles + sep;
                    tiles[tm1*(2*sep-tm1-1)/2 + tm2] += 1;
                            
                }
               
            }
        }

        n = next_n;
        m = next_m;
        k = next_k;

    }


}

/**
 * @brief Advanced tree setup routine for stage 2 processing with hash optimization.
 * @param num_tiles Number of tiles to initialize.
 * @param tile_size Size of each tile.
 * @param tree_sep Pointer to tree separation parameter.
 * @param neighbors Array of neighbors for each tile.
 * @param neighbors_sizes Size of the neighbors array.
 * @param acc Threshold for GEMM operation acceptance.
 * @param scheme_trees Double pointer to store the created tree structures.
 * @return Number of trees created.
 */ 
void count_gemms_redtree_stiles_hash(int* tiles, int* neighbors, int* neighbors_sizes, int total_num_used_tiles, int sep){
    

    int k, m, n;
    int next_k;
    int next_m;
    int next_n;
    int tm1, tm2;

    n = 0;
    m = total_num_used_tiles-sep;
    k = total_num_used_tiles-sep;

    while (k < total_num_used_tiles && m < total_num_used_tiles) {

        next_n = n;
        next_m = m;
        next_k = k;

        next_n++;
        if (next_n > next_k) {
            next_m += 1;
            while (next_m >= total_num_used_tiles && next_k < total_num_used_tiles) {
                next_k++;
                next_m = next_m-total_num_used_tiles+next_k;
            }
            next_n = 0;
        }
    
        if (m == k) {
            if (n != k) {

                if(stiles_isNeighbor(n, k, neighbors, neighbors_sizes)){ 

                    tm1 = k - total_num_used_tiles + sep;
                    tiles[tm1*(2*sep-tm1-1)/2 + tm1] += 1;

                }
            }
        }
        else {
            if (n != k) {

                if(stiles_isNeighbor(n, k, neighbors, neighbors_sizes) && stiles_isNeighbor(n, m, neighbors, neighbors_sizes)){ 

                    tm1 = k - total_num_used_tiles + sep;
                    tm2 = m - total_num_used_tiles + sep;
                    tiles[tm1*(2*sep-tm1-1)/2 + tm2] += 1;
                            
                }
               
            }
        }

        n = next_n;
        m = next_m;
        k = next_k;

    }


}

/**
 * @brief Sets up tree structures for stage 2 processing in the sTiles framework.
 *
 * @param num_tiles Total number of tiles to process.
 * @param tile_size The size of each tile (assumed square: `tile_size x tile_size`).
 * @param tree_sep Pointer to the separator value determining tree configurations.
 * @param on_off_tiles 2D array indicating which tiles are active (on/off state).
 * @param acc Threshold for GEMM operation acceptance in tree creation.
 * @param scheme_trees Pointer to store the array of created tree structures.
 * @return Number of trees created or 0 if no trees meet the acceptance threshold.
 *
 * This function creates hierarchical tree structures optimized for matrix factorization tasks.
 * It calculates GEMM operation requirements based on active tiles and separates operations 
 * into multiple tree structures. Each tree is created using `createTree` and stored in 
 * `scheme_trees` if it satisfies the acceptance threshold.
 */
int TREE_SETUP_STG2_STILES(int num_tiles, int tile_size, int *tree_sep, bool *on_off_tiles, int acc, TreeLeaf ***scheme_trees, int CORE_SIZE, int group_index){

    int sep = *tree_sep;
    int num_sep = ((sep*sep) - sep)/2 + sep;
    int* gemmcounter_redtree = (int*)calloc(num_sep, sizeof(int));

    if (gemmcounter_redtree == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    count_gemms_redtree_stiles(gemmcounter_redtree, on_off_tiles, num_tiles, sep);

    int index_redtree = 0;
    int count_new_tiles = 0;

#ifdef DEBUG
    printf("                  o Indices of trees.\n");
#endif

        int num_trees = 0;
        TreeLeaf **trees = (TreeLeaf**)calloc(num_sep, sizeof(TreeLeaf*));
        if (trees == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            free(gemmcounter_redtree); // Free previously allocated memory
            return 0;
        }

        for (int k = 0; k < sep; k++) {
            for (int j = 0; j <= k; j++) {
                index_redtree = j*(2*sep-j-1)/2 + k;

                if (gemmcounter_redtree[index_redtree] > acc) {

#ifdef DEBUG
                    printf("                  (%d %d) <=> (%d %d) and number of gemms is %2d.\n", j + num_tiles - sep, k + num_tiles - sep, j, k, gemmcounter_redtree[index_redtree]);
                    printf("                  index of tree is %d, separator is %d and tile size: %d.\n", index_redtree, num_sep, tile_size);
#endif
                    //trees[index_redtree] = createTree(gemmcounter_redtree[index_redtree], tile_size, tile_size);
                    //trees[index_redtree] = createTree(CORE_SIZE, tile_size, tile_size, group_index);
                    trees[index_redtree] = createOptimizedTree(CORE_SIZE, gemmcounter_redtree[index_redtree], tile_size, tile_size, group_index);

                    num_trees = k + 1;
                    count_new_tiles += CORE_SIZE;
                }
        }
    }

    if(num_trees==0){

        *tree_sep = 0;

    }else{

        *scheme_trees = trees;

    }

    #ifdef DEBUG
        size_t element_size = sizeof(double); 
        size_t total_memory_bytes = (size_t)count_new_tiles * tile_size * tile_size* element_size;
        double total_memory_gb = (double)total_memory_bytes / (1024 * 1024 * 1024);
        printf("                   o Memory needed for %d new tiles (all trees): %.6f GB\n", count_new_tiles, total_memory_gb);
    #endif

    free(gemmcounter_redtree); // Free the memory allocated for gemmcounter_redtree

    return num_trees;
    
}

int TREE_SETUP_STG2_STILES_PHASE_0(int num_tiles, int tile_size, int *tree_sep, bool *on_off_tiles, int** gemmcounter_redtree, int group_index){

    int sep = *tree_sep;
    int num_sep = ((sep*sep) - sep)/2 + sep;

    // Allocate memory for gemmcounter_redtree
    *gemmcounter_redtree = TreeMemoryManager::allocateZero<int>(num_sep, group_index);
    if (*gemmcounter_redtree == NULL) { // Check the dereferenced pointer
        fprintf(stderr, "Memory allocation failed\n");
        return -1; // Return error code
    }

    count_gemms_redtree_stiles_for_group_call(gemmcounter_redtree, on_off_tiles, num_tiles, sep);

    return 1;

}

int TREE_SETUP_STG2_STILES_PHASE_1(int num_tiles, int tile_size, int *tree_sep, bool *on_off_tiles, int** gemmcounter_redtree, int acc, TreeLeaf ***scheme_trees, int CORE_SIZE, int group_index){

    int sep = *tree_sep;
    int num_sep = ((sep*sep) - sep)/2 + sep;
    int index_redtree = 0;
    int count_new_tiles = 0;

    int num_trees = 0;
    TreeLeaf **trees = TreeMemoryManager::allocateZero<TreeLeaf*>(num_sep, group_index);

    sTiles::Logger::info("│ ");
    sTiles::Logger::info("│ ↳ Preprocessing phase 3 ...                                ");
    sTiles::Logger::info("│   [✔] Allocating tree reduction structures        ");
    sTiles::Logger::debug("│     • tree_sep     = " + std::to_string(sep));
    sTiles::Logger::debug("│     • num_tiles    = " + std::to_string(num_tiles));
    sTiles::Logger::debug("│     • acc threshold= " + std::to_string(acc));
    sTiles::Logger::debug("│     • tile_size    = " + std::to_string(tile_size));
    sTiles::Logger::debug("│     • num_sep      = " + std::to_string(num_sep));
    sTiles::Logger::debug("│     • CORE_SIZE    = " + std::to_string(CORE_SIZE));

    if (trees == NULL) {
        sTiles::Logger::error("Memory allocation failed for red-tree array.");
        TreeMemoryManager::deallocate(*gemmcounter_redtree); 
        *gemmcounter_redtree = NULL; // Avoid dangling pointer
        return -1; // Return error code
    }

    for (int k = 0; k < sep; k++) {
        for (int j = 0; j <= k; j++) {
            index_redtree = j*(2*sep-j-1)/2 + k;
            int gemm_count = (*gemmcounter_redtree)[index_redtree];

            if ((*gemmcounter_redtree)[index_redtree] > acc) {

                sTiles::Logger::trace("│     • Tree[" + std::to_string(index_redtree) + "] : from tiles (" + 
                             std::to_string(j + num_tiles - sep) + ", " + 
                             std::to_string(k + num_tiles - sep) + "), gemms = " + 
                             std::to_string(gemm_count));
                //trees[index_redtree] = createTree((*gemmcounter_redtree)[index_redtree], tile_size, tile_size);
                //trees[index_redtree] = createTree(CORE_SIZE, tile_size, tile_size, group_index);
                trees[index_redtree] = createOptimizedTree(CORE_SIZE, (*gemmcounter_redtree)[index_redtree], tile_size, tile_size, group_index);

                num_trees = k + 1;
                count_new_tiles += CORE_SIZE;

            }
    }
}

    if(num_trees==0){

        sTiles::Logger::info("│     • No trees created → Deallocating structure            ");
        TreeMemoryManager::deallocate(trees);  // Free the allocated array if no trees were created.
        *tree_sep = 0;

    }else{

        sTiles::Logger::info("│     • Trees created successfully                           ");
        *scheme_trees = trees;

    }

    if (num_trees > 0 && trees[0] == nullptr) {
        sTiles::Logger::warning("trees[0] is null even though num_trees > 0");
    }

    
    if (num_trees > 0) {
        size_t total_bytes = (size_t)count_new_tiles * tile_size * tile_size * sizeof(double);
        double total_gb = static_cast<double>(total_bytes) / (1024.0 * 1024 * 1024);
        sTiles::Logger::info("│     • Memory allocated for trees = " + std::to_string(total_gb) + " GB");
    }

    sTiles::Logger::info("│   [✔] Tree reduction setup complete.           ");
    return num_trees;
    
}



/**
 * @brief Sets up tree structures using hash-based neighbor optimizations for stage 2 processing.
 *
 * @param num_tiles Total number of tiles to process.
 * @param tile_size The size of each tile (assumed square: `tile_size x tile_size`).
 * @param tree_sep Pointer to the separator value determining tree configurations.
 * @param neighbors Array of neighbors for each tile.
 * @param neighbors_sizes Array indicating the size of the neighbor list for each tile.
 * @param acc Threshold for GEMM operation acceptance in tree creation.
 * @param scheme_trees Pointer to store the array of created tree structures.
 * @return Number of trees created or 0 if no trees meet the acceptance threshold.
 *
 * This function uses neighbor lists to identify dependencies and optimize tree setup for 
 * hierarchical computations. It calculates GEMM operation counts and dynamically creates 
 * trees that satisfy the given acceptance threshold.
 */
int TREE_SETUP_STG2_STILES_HASH(int num_tiles, int tile_size, int *tree_sep, int *neighbors, int* neighbors_sizes, int acc, TreeLeaf ***scheme_trees, int CORE_SIZE, int group_index){

    int sep = *tree_sep;
    int num_sep = ((sep*sep) - sep)/2 + sep;
    int* gemmcounter_redtree = (int*)calloc(num_sep, sizeof(int));

    if (gemmcounter_redtree == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    count_gemms_redtree_stiles_hash(gemmcounter_redtree, neighbors, neighbors_sizes, num_tiles, sep);

    int index_redtree = 0;
    int count_new_tiles = 0;

#ifdef DEBUG
    printf("                  o Indices of trees.\n");
#endif

    int num_trees = 0;
    TreeLeaf **trees = (TreeLeaf**)calloc(num_sep, sizeof(TreeLeaf*));
    if (trees == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(gemmcounter_redtree); // Free previously allocated memory
        return 0;
    }

    for (int k = 0; k < sep; k++) {
        for (int j = 0; j <= k; j++) {
            index_redtree = j*(2*sep-j-1)/2 + k;

                if (gemmcounter_redtree[index_redtree] > acc) {

#ifdef DEBUG
                    printf("                  (%d %d) <=> (%d %d) and number of gemms is %2d.\n", j + num_tiles - sep, k + num_tiles - sep, j, k, gemmcounter_redtree[index_redtree]);
                    printf("                  index of tree is %d, separator is %d and tile size: %d.\n", index_redtree, num_sep, tile_size);
#endif
                    //trees[index_redtree] = createTree(gemmcounter_redtree[index_redtree], tile_size, tile_size);
                    trees[index_redtree] = createOptimizedTree(CORE_SIZE, gemmcounter_redtree[index_redtree], tile_size, tile_size, group_index);

                    num_trees = k + 1;
                    count_new_tiles += CORE_SIZE;

            }
        }
    }

    if(num_trees==0){

        *tree_sep = 0;

    }else{

        *scheme_trees = trees;

    }

#ifdef DEBUG
    size_t element_size = sizeof(double); 
    size_t total_memory_bytes = (size_t)count_new_tiles * tile_size * tile_size* element_size;
    double total_memory_gb = (double)total_memory_bytes / (1024 * 1024 * 1024);
    printf("                   o Memory needed for %d new tiles (all trees): %.6f GB\n", count_new_tiles, total_memory_gb);
#endif

    free(gemmcounter_redtree); // Free the memory allocated for gemmcounter_redtree

    return num_trees;
    
}

/**
 * @brief Sets up tree structures using bit-array optimizations for stage 2 processing.
 *
 * @param num_tiles Total number of tiles to process.
 * @param tile_size The size of each tile (assumed square: `tile_size x tile_size`).
 * @param tree_sep Pointer to the separator value determining tree configurations.
 * @param bit_array Bit array representing dependency relationships between tiles.
 * @param acc Threshold for GEMM operation acceptance in tree creation.
 * @param scheme_trees Pointer to store the array of created tree structures.
 * @return Number of trees created or 0 if no trees meet the acceptance threshold.
 *
 * This function leverages a bit-array representation to efficiently identify dependencies 
 * between tiles. It calculates GEMM operation counts and dynamically creates trees that 
 * satisfy the given acceptance threshold, storing them in the `scheme_trees` array.
 */
int TREE_SETUP_STG2_STILES_HASH_BIT(int num_tiles, int tile_size, int *tree_sep, unsigned char *bit_array, int acc, TreeLeaf ***scheme_trees, int CORE_SIZE, int group_index){

    int sep = *tree_sep;
    int num_sep = ((sep*sep) - sep)/2 + sep;
    int* gemmcounter_redtree = (int*)calloc(num_sep, sizeof(int));

    if (gemmcounter_redtree == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    count_gemms_redtree_stiles_hash_bit(gemmcounter_redtree, bit_array, num_tiles, sep);

    int index_redtree = 0;
    int count_new_tiles = 0;

#ifdef DEBUG
    printf("                  o Indices of trees.\n");
#endif


    int num_trees = 0;
    TreeLeaf **trees = (TreeLeaf**)calloc(num_sep, sizeof(TreeLeaf*));
    if (trees == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(gemmcounter_redtree); // Free previously allocated memory
        return 0;
    }

    for (int k = 0; k < sep; k++) {
        for (int j = 0; j <= k; j++) {
            index_redtree = j*(2*sep-j-1)/2 + k;

                if (gemmcounter_redtree[index_redtree] > acc) {

#ifdef DEBUG
                    printf("                  (%d %d) <=> (%d %d) and number of gemms is %2d.\n", j + num_tiles - sep, k + num_tiles - sep, j, k, gemmcounter_redtree[index_redtree]);
                    printf("                  index of tree is %d, separator is %d and tile size: %d.\n", index_redtree, num_sep, tile_size);
#endif
                    //trees[index_redtree] = createTree(gemmcounter_redtree[index_redtree], tile_size, tile_size);
                    //trees[index_redtree] = createTree(CORE_SIZE, tile_size, tile_size, group_index);
                    trees[index_redtree] = createOptimizedTree(CORE_SIZE, gemmcounter_redtree[index_redtree], tile_size, tile_size, group_index);

                    num_trees = k + 1;
                    count_new_tiles += CORE_SIZE;
            }
        }
    }

    if(num_trees==0){

        *tree_sep = 0;

    }else{

        *scheme_trees = trees;

    }

#ifdef DEBUG
    size_t element_size = sizeof(double); 
    size_t total_memory_bytes = (size_t)count_new_tiles * tile_size * tile_size* element_size;
    double total_memory_gb = (double)total_memory_bytes / (1024 * 1024 * 1024);
    printf("                   o Memory needed for %d new tiles (all trees): %.6f GB\n", count_new_tiles, total_memory_gb);
#endif

    free(gemmcounter_redtree); // Free the memory allocated for gemmcounter_redtree

    return num_trees;
    
}

/**
 * @brief Sets up tree structures for stage 2 processing using boolean on/off tile states.
 *
 * @param num_tiles Total number of tiles to process.
 * @param tile_size The size of each tile (assumed square: `tile_size x tile_size`).
 * @param tree_sep Pointer to the separator value determining tree configurations.
 * @param on_off_tiles 2D boolean array representing the on/off state of tiles.
 * @param acc Threshold for GEMM operation acceptance in tree creation.
 * @param scheme_trees Pointer to store the array of created tree structures.
 * @return Number of trees created or 0 if no trees meet the acceptance threshold.
 *
 * This function calculates GEMM operation requirements based on the on/off states of tiles 
 * and separates them into hierarchical tree structures. Trees are created dynamically and 
 * stored in the provided `scheme_trees` array.
 */
int STILES_TREE_SETUP_STG2(int num_tiles, int tile_size, int *tree_sep, bool **on_off_tiles, int acc, TreeLeaf ***scheme_trees, int CORE_SIZE, int group_index){

    int sep = *tree_sep;
    int num_sep = ((sep*sep) - sep)/2 + sep;
    int* gemmcounter_redtree = (int*)calloc(num_sep, sizeof(int));

    if (gemmcounter_redtree == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    count_gemms_redtree(gemmcounter_redtree, on_off_tiles, num_tiles, sep);

    int index_redtree = 0;
    int count_new_tiles = 0;

#ifdef DEBUG
    printf("                  o Indices of trees.\n");
#endif


    int num_trees = 0;
    TreeLeaf **trees = (TreeLeaf**)calloc(num_sep, sizeof(TreeLeaf*));
    if (trees == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(gemmcounter_redtree); // Free previously allocated memory
        return 0;
    }

    for (int k = 0; k < sep; k++) {
        for (int j = 0; j <= k; j++) {
            index_redtree = j*(2*sep-j-1)/2 + k;

            if (gemmcounter_redtree[index_redtree] > acc) {

#ifdef DEBUG
                printf("                  (%d %d) <=> (%d %d) and number of gemms is %2d.\n", j + num_tiles - sep, k + num_tiles - sep, j, k, gemmcounter_redtree[index_redtree]);
                printf("                  index of tree is %d, separator is %d and tile size: %d.\n", index_redtree, num_sep, tile_size);
#endif
                //trees[index_redtree] = createTree(gemmcounter_redtree[index_redtree], tile_size, tile_size);
                trees[index_redtree] = createOptimizedTree(CORE_SIZE, gemmcounter_redtree[index_redtree], tile_size, tile_size, group_index);

                num_trees = k + 1;
                count_new_tiles += CORE_SIZE;
                
            }
        }
    }

    if(num_trees==0){

        *tree_sep = 0;

    }else{

        *scheme_trees = trees;

    }

#ifdef DEBUG
    size_t element_size = sizeof(double); 
    size_t total_memory_bytes = (size_t)count_new_tiles * tile_size * tile_size* element_size;
    double total_memory_gb = (double)total_memory_bytes / (1024 * 1024 * 1024);
    printf("                   o Memory needed for %d new tiles (all trees): %.6f GB\n", count_new_tiles, total_memory_gb);
#endif

    free(gemmcounter_redtree); // Free the memory allocated for gemmcounter_redtree

    return num_trees;
    
}
