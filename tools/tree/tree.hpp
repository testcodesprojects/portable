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


#include "tree_structs.hpp"
#include "../memory/TreeMemoryManager.hpp"
#include "../common/stiles_logger.hpp"
#include "../algorithms/tree.hpp"

namespace sTiles{ namespace Tree{

// Fast-mode variant: uses a predicate to test activity (neighbors) for (i,j) in upper triangle.
// Example usage with TileIndexer fast path after constructMapper/bindActive:
//   const int N = scheme_fast->dimTiledMatrix;
//   auto is_on = [&](int a, int b){
//       if (a > b) std::swap(a,b);
//       return scheme_fast->state.isActive(a, b, N);
//   };
//   sTiles::Tree::count_gemms_redtree_stiles_for_group_call_fast(&tiles, is_on, N, sep);
template <typename IsOn>
inline void count_gemms(int** tiles, IsOn is_on, int total_num_used_tiles, int sep){
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

inline int create_trees(int num_tiles, int tile_size, int *tree_sep, int** gemmcounter_redtree, int acc, TreeLeaf ***scheme_trees, int CORE_SIZE, int group_index){

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

}}