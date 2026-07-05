/**
 * @file inv_algorithms.hpp
 * @brief Selective inversion algorithms for tiled sparse matrices.
 *
 * This file implements task collection and scheduling algorithms for parallel
 * selective inversion (computing selected elements of the inverse) on tiled
 * sparse matrices. Features include:
 * - Multiple inversion algorithm variants optimized for different sparsity patterns
 * - Task generation for TRTRI, TRMM, GEMM, and SYRK operations
 * - Support for computing diagonal and off-diagonal inverse elements
 * - Integration with the Cholesky factor structure
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

#ifndef STILES_INV_ALGORITHMS_HPP
#define STILES_INV_ALGORITHMS_HPP

#include <vector>
#include <array>
#include <functional>
#include <iostream> // For error reporting, as in the original


#include "../TileIndexer/TileIndexer.hpp"
#include "../TileIndexer/TileIndexerGraphBuilder.hpp"
#include "../TileIndexer/TileIndexerMemoryUtils.hpp"
#include "../common/stiles_structs.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <vector>

namespace alg {

inline int inv_variant4(const TileIndexer::State& state_in,
                        TileIndexer::Method method,
                        int myrank,
                        int worldsize,
                        int num_tiles,
                        const std::function<int(int,int)>& map_id,
                        std::vector<std::array<int,7>>& out_tasks)
{
    (void)method;

    out_tasks.clear();
    if (num_tiles <= 0 || worldsize <= 0) return 0;

    // Graph-based neighbor lists (upper CSR)
    const int* up_off = state_in.graph_off_up.data();
    const int* up_edg = state_in.graph_edges_up.data();

    auto dense_idx = [&](int a, int b) -> int {
        return map_id(a, b);
    };

    auto has_tile = [&](int a, int b) -> bool {
        return dense_idx(a, b) >= 0;
    };

    auto emit = [&](int type, int m, int k, int n,
                    int idx1, int idx2, int idx3) {
        std::array<int,7> row{};
        row[0] = type;
        row[1] = m;
        row[2] = k;
        row[3] = n;
        row[4] = idx1;
        row[5] = idx2;
        row[6] = idx3;
        out_tasks.push_back(row);
    };

    // Region 1 (ascending j per row i assigned to this rank)
    {
        int i = num_tiles - 1 - myrank;
        while (i >= 0) {
            const int d = dense_idx(i, i);
            if (d >= 0) {
                emit(1, i, i, 0, d, d, 0);
            }

            if (true) {
                // GRAPH: iterate j over upper neighbors of i
                const int i_up_start = up_off[i];
                const int i_up_deg   = up_off[i + 1] - i_up_start;
                for (int ji = 0; ji < i_up_deg; ++ji) {
                    const int j = up_edg[i_up_start + ji];
                    if (j <= i) continue;
                    const int ij = dense_idx(i, j);
                    if (ij < 0 || d < 0) continue;
                    emit(2, i, j, 0, d, ij, 0);
                }
            } else {
                // BRUTE-FORCE: scan all j > i
                for (int j = i + 1; j < num_tiles; ++j) {
                    if (!has_tile(i, j)) continue;
                    const int ij = dense_idx(i, j);
                    if (ij < 0 || d < 0) continue;
                    emit(2, i, j, 0, d, ij, 0);
                }
            }

            i -= worldsize;
        }

        emit(3, 0, 0, 0, 0, 0, 0);
    }

    // Region 2 (descending j per row i)
    {
        if (false) {
            // ── NEW: round-robin (i,j) pair distribution across all rows ──
            int pair_counter = 0;
            int i = num_tiles - 1;
            while (i >= 0) {
                const int d = dense_idx(i, i);
                const int i_up_start = up_off[i];
                const int i_up_deg   = up_off[i + 1] - i_up_start;

                // Diagonal case
                if (d >= 0) {
                    const int owner = pair_counter % worldsize;
                    ++pair_counter;
                    if (myrank == owner) {
                        emit(4, i, i, 0, d, 0, 0);

                        for (int ki = 0; ki < i_up_deg; ++ki) {
                            const int k = up_edg[i_up_start + ki];
                            if (k <= i) continue;
                            const int ik = dense_idx(i, k);
                            if (ik < 0) continue;
                            emit(5, i, i, k, d, ik, 0);
                        }

                        emit(6, i, i, 0, d, 0, 0);
                    }
                }

                // Off-diagonal cases: j in upper neighbors of i, descending
                for (int ji = i_up_deg - 1; ji >= 0; --ji) {
                    const int j = up_edg[i_up_start + ji];
                    if (j <= i) continue;
                    const int ij = dense_idx(i, j);
                    if (ij < 0) continue;
                    const int owner = pair_counter % worldsize;
                    ++pair_counter;
                    if (myrank != owner) continue;

                    int ii_last = i, jj_last = j;

                    for (int ki = 0; ki < i_up_deg; ++ki) {
                        const int k = up_edg[i_up_start + ki];
                        if (k <= i) continue;
                        if (k > j) {
                            const int ik = dense_idx(i, k);
                            const int jk = dense_idx(j, k);
                            if (ik >= 0 && jk >= 0) {
                                emit(7, i, j, k, ik, jk, ij);
                                ii_last = i; jj_last = j;
                            }
                        } else {
                            const int ik = dense_idx(i, k);
                            const int kj = dense_idx(k, j);
                            if (ik >= 0 && kj >= 0) {
                                emit(8, i, j, k, ik, kj, ij);
                                ii_last = i; jj_last = j;
                            }
                        }
                    }

                    emit(9, ii_last, jj_last, 0, 0, 0, 0);
                }

                --i;
            }
        } else {
            // ── OLD: row-striped distribution (i -= worldsize) ──
            int i = num_tiles - 1 - myrank;
            while (i >= 0) {
                const int d = dense_idx(i, i);
                const int i_up_start = up_off[i];
                const int i_up_deg   = up_off[i + 1] - i_up_start;

                if (true) {
                    // GRAPH: iterate j over upper neighbors of i (descending), plus i itself
                    // Diagonal case (i == i)
                    if (d >= 0) {
                        emit(4, i, i, 0, d, 0, 0);

                        if (true) {
                            // GRAPH: iterate k over upper neighbors of i for type 5
                            for (int ki = 0; ki < i_up_deg; ++ki) {
                                const int k = up_edg[i_up_start + ki];
                                if (k <= i) continue;
                                const int ik = dense_idx(i, k);
                                if (ik < 0) continue;
                                emit(5, i, i, k, d, ik, 0);
                            }
                        } else {
                            // BRUTE-FORCE: scan all k > i for type 5
                            for (int k = i + 1; k <= num_tiles - 1; ++k) {
                                if (!has_tile(i, k)) continue;
                                const int ik = dense_idx(i, k);
                                if (ik < 0) continue;
                                emit(5, i, i, k, d, ik, 0);
                            }
                        }

                        emit(6, i, i, 0, d, 0, 0);
                    }

                    // Off-diagonal cases: j in upper neighbors of i, descending
                    for (int ji = i_up_deg - 1; ji >= 0; --ji) {
                        const int j = up_edg[i_up_start + ji];
                        if (j <= i) continue;
                        const int ij = dense_idx(i, j);
                        if (ij < 0) continue;

                        int ii_last = i, jj_last = j;

                        if (true) {
                            // GRAPH: iterate k over upper neighbors of i for types 7/8
                            for (int ki = 0; ki < i_up_deg; ++ki) {
                                const int k = up_edg[i_up_start + ki];
                                if (k <= i) continue;
                                if (k > j) {
                                    const int ik = dense_idx(i, k);
                                    const int jk = dense_idx(j, k);
                                    if (ik >= 0 && jk >= 0) {
                                        emit(7, i, j, k, ik, jk, ij);
                                        ii_last = i; jj_last = j;
                                    }
                                } else {
                                    const int ik = dense_idx(i, k);
                                    const int kj = dense_idx(k, j);
                                    if (ik >= 0 && kj >= 0) {
                                        emit(8, i, j, k, ik, kj, ij);
                                        ii_last = i; jj_last = j;
                                    }
                                }
                            }
                        } else {
                            // BRUTE-FORCE: scan all k for types 7/8
                            for (int k = i + 1; k <= num_tiles - 1; ++k) {
                                if (k > j) {
                                    if (has_tile(i, j) &&
                                        has_tile(j, k) &&
                                        has_tile(i, k)) {
                                        const int ik = dense_idx(i, k);
                                        const int jk = dense_idx(j, k);
                                        if (ik < 0 || jk < 0 || ij < 0) continue;
                                        emit(7, i, j, k, ik, jk, ij);
                                        ii_last = i; jj_last = j;
                                    }
                                } else {
                                    if (has_tile(i, j) &&
                                        has_tile(k, j) &&
                                        has_tile(i, k)) {
                                        const int ik = dense_idx(i, k);
                                        const int kj = dense_idx(k, j);
                                        if (ik < 0 || kj < 0 || ij < 0) continue;
                                        emit(8, i, j, k, ik, kj, ij);
                                        ii_last = i; jj_last = j;
                                    }
                                }
                            }
                        }

                        emit(9, ii_last, jj_last, 0, 0, 0, 0);
                    }
                } else {
                    // BRUTE-FORCE: original Region 2 with full j scan
                    for (int j = num_tiles - 1; j >= i; --j) {
                        if (i == j) {
                            if (d < 0) continue;
                            emit(4, i, j, 0, d, 0, 0);
                            for (int k = i + 1; k <= num_tiles - 1; ++k) {
                                if (!has_tile(i, k)) continue;
                                const int ik = dense_idx(i, k);
                                if (ik < 0) continue;
                                emit(5, i, j, k, d, ik, 0);
                            }
                            emit(6, i, j, 0, d, 0, 0);
                        } else {
                            int ii_last = i;
                            int jj_last = j;
                            for (int k = i + 1; k <= num_tiles - 1; ++k) {
                                if (k > j) {
                                    if (has_tile(i, j) &&
                                        has_tile(j, k) &&
                                        has_tile(i, k)) {
                                        const int ik = dense_idx(i, k);
                                        const int jk = dense_idx(j, k);
                                        const int ij = dense_idx(i, j);
                                        if (ik < 0 || jk < 0 || ij < 0) continue;
                                        emit(7, i, j, k, ik, jk, ij);
                                        ii_last = i; jj_last = j;
                                    }
                                } else {
                                    if (has_tile(i, j) &&
                                        has_tile(k, j) &&
                                        has_tile(i, k)) {
                                        const int ik = dense_idx(i, k);
                                        const int kj = dense_idx(k, j);
                                        const int ij = dense_idx(i, j);
                                        if (ik < 0 || kj < 0 || ij < 0) continue;
                                        emit(8, i, j, k, ik, kj, ij);
                                        ii_last = i; jj_last = j;
                                    }
                                }
                            }
                            emit(9, ii_last, jj_last, 0, 0, 0, 0);
                        }
                    }
                }

                i -= worldsize;
            }
        }
    }

    return static_cast<int>(out_tasks.size());
}


inline int inv_scaled_variant4(const TileIndexer::State& state_in,
                               TileIndexer::Method method,
                               int myrank,
                               int worldsize,
                               int num_tiles,
                               const std::function<int(int,int)>& map_id,
                               std::vector<std::array<int,7>>& out_tasks)
{
    (void)method;

    out_tasks.clear();
    if (num_tiles <= 0 || worldsize <= 0) return 0;

    // Graph-based neighbor lists (upper CSR)
    const int* up_off = state_in.graph_off_up.data();
    const int* up_edg = state_in.graph_edges_up.data();

    auto dense_idx = [&](int a, int b) -> int {
        return map_id(a, b);
    };

    auto has_tile = [&](int a, int b) -> bool {
        return dense_idx(a, b) >= 0;
    };

    auto emit = [&](int type, int m, int k, int n,
                    int idx1, int idx2, int idx3) {
        std::array<int,7> row{};
        row[0] = type;
        row[1] = m;
        row[2] = k;
        row[3] = n;
        row[4] = idx1;
        row[5] = idx2;
        row[6] = idx3;
        out_tasks.push_back(row);
    };

    // Region 1 (same as unscaled)
    {
        int i = num_tiles - 1 - myrank;
        while (i >= 0) {
            const int d = dense_idx(i, i);
            if (d >= 0) {
                emit(1, i, i, 0, d, d, 0);
            }

            if (false) {
                // GRAPH: iterate j over upper neighbors of i
                const int i_up_start = up_off[i];
                const int i_up_deg   = up_off[i + 1] - i_up_start;
                for (int ji = 0; ji < i_up_deg; ++ji) {
                    const int j = up_edg[i_up_start + ji];
                    if (j <= i) continue;
                    const int ij = dense_idx(i, j);
                    if (ij < 0 || d < 0) continue;
                    emit(2, i, j, 0, d, ij, 0);
                }
            } else {
                // BRUTE-FORCE: scan all j > i
                for (int j = i + 1; j < num_tiles; ++j) {
                    if (!has_tile(i, j)) continue;
                    const int ij = dense_idx(i, j);
                    if (ij < 0 || d < 0) continue;
                    emit(2, i, j, 0, d, ij, 0);
                }
            }

            i -= worldsize;
        }

        emit(3, 0, 0, 0, 0, 0, 0);
    }

    // Region 2 (work scaled by dense index modulo worldsize)
    {
        int i = num_tiles - 1;
        while (i >= 0) {
            const int i_up_start = up_off[i];
            const int i_up_deg   = up_off[i + 1] - i_up_start;

            if (false) {
                // GRAPH: iterate j over upper neighbors of i (descending), plus diagonal
                // Diagonal case
                const int d = dense_idx(i, i);
                if (d >= 0 && myrank == (d % worldsize)) {
                    emit(4, i, i, 0, d, 0, 0);

                    if (false) {
                        // GRAPH: iterate k over upper neighbors of i for type 5
                        for (int ki = 0; ki < i_up_deg; ++ki) {
                            const int k = up_edg[i_up_start + ki];
                            if (k <= i) continue;
                            const int ik = dense_idx(i, k);
                            if (ik < 0) continue;
                            emit(5, i, i, k, d, ik, 0);
                        }
                    } else {
                        // BRUTE-FORCE: scan all k > i for type 5
                        for (int k = i + 1; k <= num_tiles - 1; ++k) {
                            if (!has_tile(i, k)) continue;
                            const int ik = dense_idx(i, k);
                            if (ik < 0) continue;
                            emit(5, i, i, k, d, ik, 0);
                        }
                    }

                    emit(6, i, i, 0, d, 0, 0);
                }

                // Off-diagonal cases: j in upper neighbors of i, descending
                for (int ji = i_up_deg - 1; ji >= 0; --ji) {
                    const int j = up_edg[i_up_start + ji];
                    if (j <= i) continue;
                    const int ij = dense_idx(i, j);
                    if (ij < 0) continue;
                    if (myrank != (ij % worldsize)) continue;

                    bool setting = false;
                    int ii_last = i;
                    int jj_last = j;

                    if (i + 1 <= num_tiles - 1) {
                        setting = true;
                    }

                    if (false) {
                        // GRAPH: iterate k over upper neighbors of i for types 7/8
                        for (int ki = 0; ki < i_up_deg; ++ki) {
                            const int k = up_edg[i_up_start + ki];
                            if (k <= i) continue;
                            if (k > j) {
                                const int ik = dense_idx(i, k);
                                const int jk = dense_idx(j, k);
                                if (ik >= 0 && jk >= 0) {
                                    emit(7, i, j, k, ik, jk, ij);
                                    ii_last = i; jj_last = j;
                                }
                            } else {
                                const int ik = dense_idx(i, k);
                                const int kj = dense_idx(k, j);
                                if (ik >= 0 && kj >= 0) {
                                    emit(8, i, j, k, ik, kj, ij);
                                    ii_last = i; jj_last = j;
                                }
                            }
                        }
                    } else {
                        // BRUTE-FORCE: scan all k for types 7/8
                        for (int k = i + 1; k <= num_tiles - 1; ++k) {
                            if (k > j) {
                                if (has_tile(i, j) &&
                                    has_tile(j, k) &&
                                    has_tile(i, k)) {
                                    const int ik = dense_idx(i, k);
                                    const int jk = dense_idx(j, k);
                                    if (ik < 0 || jk < 0) continue;
                                    emit(7, i, j, k, ik, jk, ij);
                                    ii_last = i; jj_last = j;
                                }
                            } else {
                                if (has_tile(i, j) &&
                                    has_tile(k, j) &&
                                    has_tile(i, k)) {
                                    const int ik = dense_idx(i, k);
                                    const int kj = dense_idx(k, j);
                                    if (ik < 0 || kj < 0) continue;
                                    emit(8, i, j, k, ik, kj, ij);
                                    ii_last = i; jj_last = j;
                                }
                            }
                        }
                    }

                    if (setting) {
                        const int tmp_index = dense_idx(ii_last, jj_last);
                        if (tmp_index >= 0 &&
                            myrank == (tmp_index % worldsize)) {
                            emit(9, ii_last, jj_last, 0, 0, 0, 0);
                        }
                    }
                }
            } else {
                // BRUTE-FORCE: original Region 2 with full j scan
                for (int j = num_tiles - 1; j >= i; --j) {
                    if (i == j) {
                        const int d = dense_idx(i, i);
                        if (d < 0) continue;
                        if (myrank != (d % worldsize)) continue;
                        emit(4, i, j, 0, d, 0, 0);
                        for (int k = i + 1; k <= num_tiles - 1; ++k) {
                            if (!has_tile(i, k)) continue;
                            const int ik = dense_idx(i, k);
                            if (ik < 0) continue;
                            emit(5, i, j, k, d, ik, 0);
                        }
                        emit(6, i, j, 0, d, 0, 0);
                    } else {
                        const int ij = dense_idx(i, j);
                        if (ij < 0) continue;
                        if (myrank != (ij % worldsize)) continue;
                        bool setting = false;
                        int ii_last = i;
                        int jj_last = j;
                        if (i + 1 <= num_tiles - 1) {
                            setting = true;
                        }
                        for (int k = i + 1; k <= num_tiles - 1; ++k) {
                            if (k > j) {
                                if (has_tile(i, j) &&
                                    has_tile(j, k) &&
                                    has_tile(i, k)) {
                                    const int ik = dense_idx(i, k);
                                    const int jk = dense_idx(j, k);
                                    if (ik < 0 || jk < 0) continue;
                                    emit(7, i, j, k, ik, jk, ij);
                                    ii_last = i; jj_last = j;
                                }
                            } else {
                                if (has_tile(i, j) &&
                                    has_tile(k, j) &&
                                    has_tile(i, k)) {
                                    const int ik = dense_idx(i, k);
                                    const int kj = dense_idx(k, j);
                                    if (ik < 0 || kj < 0) continue;
                                    emit(8, i, j, k, ik, kj, ij);
                                    ii_last = i; jj_last = j;
                                }
                            }
                        }
                        if (setting) {
                            const int tmp_index = dense_idx(ii_last, jj_last);
                            if (tmp_index >= 0 &&
                                myrank == (tmp_index % worldsize)) {
                                emit(9, ii_last, jj_last, 0, 0, 0, 0);
                            }
                        }
                    }
                }
            }

            --i;
        }
    }

    return static_cast<int>(out_tasks.size());
}

} // namespace alg


#endif // STILES_INV_ALGORITHMS_HPP


// #ifndef STILES_INV_ALGORITHMS_HPP
// #define STILES_INV_ALGORITHMS_HPP

// #include "../TileIndexer/TileIndexer.hpp"
// #include "../TileIndexer/TileIndexerGraphBuilder.hpp"
// #include "../TileIndexer/TileIndexerMemoryUtils.hpp"
// #include "../common/stiles_structs.hpp"

// #include <algorithm>
// #include <array>
// #include <functional>
// #include <iterator>
// #include <vector>

// namespace alg {

// // small helper (same as in chol variants): binary-search with tiny-linear fallback
// constexpr int kTinySpan = 16;
// #ifndef STILES_CONTAINS_SORTED_SPAN_HELPER
// inline bool contains_sorted_span(const int* base, int len, int key) {
//     if (len <= kTinySpan) {
//         for (int i = 0; i < len; ++i) if (base[i] == key) return true;
//         return false;
//     }
//     return std::binary_search(base, base + len, key);
// }
// #define STILES_CONTAINS_SORTED_SPAN_HELPER 1
// #endif

// inline int inv_variant1(const TileIndexer::State& state_in,
//                         TileIndexer::Method /*method*/,
//                         int myrank,
//                         int worldsize,
//                         int num_tiles,
//                         const std::function<int(int,int)>& map_id,
//                         std::vector<std::array<int,7>>& out_tasks)
// {
//     out_tasks.clear();
//     if (num_tiles <= 0 || worldsize <= 0) return 0;

//     // Ensure graphs with include_self = true
//     TileIndexer::State& state = const_cast<TileIndexer::State&>(state_in);
//     if (!state.graphs_built || state.graph_N != num_tiles || !state.graph_include_self) {
//     #if defined(_OPENMP)
//         #pragma omp critical(stiles_graph_build_once)
//         {
//             if (!state.graphs_built || state.graph_N != num_tiles || !state.graph_include_self) {
//                 TileIndexer::build_graphs_up_lo(state, num_tiles, /*include_self=*/true);
//             }
//         }
//     #else
//         TileIndexer::build_graphs_up_lo(state, num_tiles, /*include_self=*/true);
//     #endif
//     }

//     // CSR (upper) with self included
//     const int* __restrict off_up   = state.graph_off_up.data();
//     const int* __restrict edges_up = state.graph_edges_up.data();

//     const auto deg_up = [&](int i) -> int {
//         return off_up[static_cast<std::size_t>(i + 1)] - off_up[static_cast<std::size_t>(i)];
//     };
//     const auto has_up = [&](int a, int b) -> bool {
//         if (a > b) std::swap(a, b);
//         const int s = off_up[static_cast<std::size_t>(a)];
//         const int d = deg_up(a);
//         return contains_sorted_span(edges_up + s, d, b);
//     };
//     const auto idx_of = [&](int a, int b) -> int { return map_id(std::min(a,b), std::max(a,b)); };
//     const auto owner_of = [&](int a, int b) -> int {
//         const int id = idx_of(a,b);
//         return (id >= 0) ? (id % worldsize) : -1;
//     };

//     // -------------------------
//     // Pass 1: count all rows
//     // -------------------------
//     auto count_rows = [&]() -> int {
//         int cnt = 0;

//         // Region 1 (row-sliced by rank): iterate only neighbors in the upper graph
//         {
//             int i = num_tiles - 1 - myrank;
//             while (i >= 0) {
//                 // type 1 for (i,i)
//                 ++cnt;
//                 const int su = off_up[static_cast<std::size_t>(i)];
//                 const int eu = off_up[static_cast<std::size_t>(i + 1)];
//                 for (int p = su; p < eu; ++p) {
//                     const int j = edges_up[p];
//                     if (j == i) continue; // already counted diag as type 1
//                     // type 2 for each active (i,j)
//                     ++cnt;
//                 }
//                 i -= worldsize;
//             }
//             // barrier row: type 3 (exactly one per rank)
//             ++cnt;
//         }

//         // Region 2 (row-sliced by rank): emit per i-row handled by this rank
//         {
//             int i = num_tiles - 1 - myrank;
//             while (i >= 0) {
//                 // Diagonal-owned work (types 4,5,6) for this i
//                 // type 4
//                 ++cnt;
//                 // type 5 for neighbors k>i
//                 const int su = off_up[static_cast<std::size_t>(i)];
//                 const int eu = off_up[static_cast<std::size_t>(i + 1)];
//                 for (int p = su; p < eu; ++p) {
//                     const int k = edges_up[p];
//                     if (k > i) ++cnt;
//                 }
//                 // type 6
//                 ++cnt;

//                 // Off-diagonal j across the full row; type 7/8 over neighbor k, then unconditional type 9
//                 for (int j = num_tiles - 1; j >= i; --j) {
//                     if (j == i) continue;
//                     for (int q = su; q < eu; ++q) {
//                         const int k = edges_up[q];
//                         if (k <= i) continue;
//                         if (k > j) {
//                             if (has_up(i, j) && has_up(j, k)) ++cnt; // type 7
//                         } else { // k <= j
//                             if (has_up(i, j) && has_up(k, j)) ++cnt; // type 8
//                         }
//                     }
//                     // type 9 finalize (i,j)
//                     ++cnt;
//                 }

//                 i -= worldsize;
//             }
//         }

//         return cnt;
//     };

//     const int total = count_rows();
//     if (total == 0) return 0;
//     out_tasks.resize(static_cast<std::size_t>(total));

//     // -------------------------
//     // Pass 2: emit rows
//     // -------------------------
//     int w = 0;
//     auto emit = [&](int t,int i,int j,int k,int i1,int i2,int i3){
//         auto& r = out_tasks[static_cast<std::size_t>(w++)];
//         r[0]=t; r[1]=i; r[2]=j; r[3]=k; r[4]=i1; r[5]=i2; r[6]=i3;
//     };

//     // Region 1: neighbors only
//     {
//         int i = num_tiles - 1 - myrank;
//         while (i >= 0) {
//             const int d = idx_of(i,i);
//             emit(1, i,i,0, d,d,0);

//             const int su = off_up[static_cast<std::size_t>(i)];
//             const int eu = off_up[static_cast<std::size_t>(i + 1)];
//             for (int p = su; p < eu; ++p) {
//                 const int j = edges_up[p];
//                 if (j == i) continue;
//                 const int ii = d;
//                 const int ij = idx_of(i,j);
//                 emit(2, i,j,0, ii,ij,0);
//             }
//             i -= worldsize;
//         }
//         // barrier (type 3)
//         emit(3, 0,0,0, 0,0,0);
//     }

//     // Region 2: row-sliced by rank; iterate full j range and neighbor k, then finalize
//     {
//         int i = num_tiles - 1 - myrank;
//         while (i >= 0) {
//             const int d = idx_of(i,i);
//             emit(4, i,i,0, d,0,0);
//             const int su = off_up[static_cast<std::size_t>(i)];
//             const int eu = off_up[static_cast<std::size_t>(i + 1)];
//             for (int p = su; p < eu; ++p) {
//                 const int k = edges_up[p];
//                 if (k > i) {
//                     const int ik = idx_of(i,k);
//                     emit(5, i,i,k, d,ik,0);
//                 }
//             }
//             emit(6, i,i,0, d,0,0);

//             for (int j = num_tiles - 1; j >= i; --j) {
//                 if (j == i) continue;
//                 for (int q = su; q < eu; ++q) {
//                     const int k = edges_up[q];
//                     if (k <= i) continue;
//                     if (k > j) {
//                         if (has_up(i,j) && has_up(j,k)) {
//                             const int ik = idx_of(i,k);
//                             const int jk = idx_of(j,k);
//                             const int ij = idx_of(i,j);
//                             emit(7, i,j,k, ik,jk,ij);
//                         }
//                     } else { // k <= j
//                         if (has_up(i,j) && has_up(k,j)) {
//                             const int ik = idx_of(i,k);
//                             const int kj = idx_of(k,j);
//                             const int ij = idx_of(i,j);
//                             emit(8, i,j,k, ik,kj,ij);
//                         }
//                     }
//                 }
//                 // finalize (type 9) for (i,j)
//                 emit(9, i,j,0, 0,0,0);
//             }

//             i -= worldsize;
//         }
//     }

//     return total;
// }

// inline int inv_variant4(const TileIndexer::State& state_in,
//                         TileIndexer::Method method,
//                         int myrank,
//                         int worldsize,
//                         int num_tiles,
//                         const std::function<int(int,int)>& map_id,
//                         std::vector<std::array<int,7>>& out_tasks,
//                         int panel_height = 64)
// {
//     out_tasks.clear();
//     if (num_tiles <= 0 || worldsize <= 0) return 0;

//     std::vector<std::array<int,7>> staging;
//     const int produced = inv_variant1(state_in,
//                                       method,
//                                       myrank,
//                                       worldsize,
//                                       num_tiles,
//                                       map_id,
//                                       staging);
//     if (produced <= 0) {
//         return produced;
//     }

//     const int panel = std::max(1, panel_height);
//     const int num_panels = std::max(1, (num_tiles + panel - 1) / panel);

//     auto barrier_it = std::find_if(staging.begin(), staging.end(),
//                                    [](const std::array<int,7>& row) { return row[0] == 3; });
//     auto stage2_begin = (barrier_it != staging.end()) ? std::next(barrier_it) : staging.end();

//     std::vector<std::vector<std::array<int,7>>> stage1_bins(static_cast<std::size_t>(num_panels));
//     for (auto it = staging.begin(); it != barrier_it; ++it) {
//         const int i = (*it)[1];
//         const int panel_id = (i >= 0) ? std::min(num_panels - 1, i / panel) : 0;
//         stage1_bins[static_cast<std::size_t>(panel_id)].push_back(std::move(*it));
//     }

//     std::vector<std::vector<std::array<int,7>>> stage2_bins(static_cast<std::size_t>(num_panels));
//     for (auto it = stage2_begin; it != staging.end(); ++it) {
//         const int i = (*it)[1];
//         const int panel_id = (i >= 0) ? std::min(num_panels - 1, i / panel) : 0;
//         stage2_bins[static_cast<std::size_t>(panel_id)].push_back(std::move(*it));
//     }

//     auto stage1_cmp = [](const std::array<int,7>& a, const std::array<int,7>& b) {
//         if (a[1] != b[1]) return a[1] > b[1];
//         if (a[0] != b[0]) return a[0] < b[0];
//         if (a[2] != b[2]) return a[2] < b[2];
//         return a[3] < b[3];
//     };
//     auto stage2_cmp = [](const std::array<int,7>& a, const std::array<int,7>& b) {
//         if (a[1] != b[1]) return a[1] > b[1];
//         if (a[2] != b[2]) return a[2] > b[2];
//         if (a[0] != b[0]) return a[0] < b[0];
//         if (a[3] != b[3]) return a[3] < b[3];
//         if (a[4] != b[4]) return a[4] < b[4];
//         return a[5] < b[5];
//     };

//     out_tasks.reserve(staging.size());
//     for (auto& bin : stage1_bins) {
//         if (bin.empty()) continue;
//         std::sort(bin.begin(), bin.end(), stage1_cmp);
//         out_tasks.insert(out_tasks.end(),
//                          std::make_move_iterator(bin.begin()),
//                          std::make_move_iterator(bin.end()));
//     }
//     if (barrier_it != staging.end()) {
//         out_tasks.push_back(std::move(*barrier_it));
//     }
//     for (auto& bin : stage2_bins) {
//         if (bin.empty()) continue;
//         std::sort(bin.begin(), bin.end(), stage2_cmp);
//         out_tasks.insert(out_tasks.end(),
//                          std::make_move_iterator(bin.begin()),
//                          std::make_move_iterator(bin.end()));
//     }

//     return static_cast<int>(out_tasks.size());
// }

// inline int inv_scaled_variant1(const TileIndexer::State& state_in,
//                                TileIndexer::Method /*method*/,
//                                int myrank,
//                                int worldsize,
//                                int num_tiles,
//                                const std::function<int(int,int)>& map_id,
//                                std::vector<std::array<int,7>>& out_tasks)
// {
//     out_tasks.clear();
//     if (num_tiles <= 0 || worldsize <= 0) return 0;

//     // Ensure graphs with include_self = true
//     TileIndexer::State& state = const_cast<TileIndexer::State&>(state_in);
//     if (!state.graphs_built || state.graph_N != num_tiles || !state.graph_include_self) {
//     #if defined(_OPENMP)
//         #pragma omp critical(stiles_graph_build_once)
//         {
//             if (!state.graphs_built || state.graph_N != num_tiles || !state.graph_include_self) {
//                 TileIndexer::build_graphs_up_lo(state, num_tiles, /*include_self=*/true);
//             }
//         }
//     #else
//         TileIndexer::build_graphs_up_lo(state, num_tiles, /*include_self=*/true);
//     #endif
//     }

//     // CSR (upper) with self included
//     const int* __restrict off_up   = state.graph_off_up.data();
//     const int* __restrict edges_up = state.graph_edges_up.data();

//     const auto deg_up = [&](int i) -> int {
//         return off_up[static_cast<std::size_t>(i + 1)] - off_up[static_cast<std::size_t>(i)];
//     };
//     const auto has_up = [&](int a, int b) -> bool {
//         if (a > b) std::swap(a, b);
//         const int s = off_up[static_cast<std::size_t>(a)];
//         const int d = deg_up(a);
//         return contains_sorted_span(edges_up + s, d, b);
//     };
//     const auto idx_of   = [&](int a, int b) -> int { return map_id(std::min(a,b), std::max(a,b)); };
//     const auto owner_of = [&](int a, int b) -> int {
//         const int id = idx_of(a,b);
//         return (id >= 0) ? (id % worldsize) : -1;
//     };

//     // -------------------------
//     // Pass 1: count (adjacency-driven)
//     // -------------------------
//     auto count_rows = [&]() -> int {
//         int cnt = 0;

//         // Region 1 (neighbors only)
//         {
//             int i = num_tiles - 1 - myrank;
//             while (i >= 0) {
//                 // type 1
//                 ++cnt;
//                 const int su = off_up[static_cast<std::size_t>(i)];
//                 const int eu = off_up[static_cast<std::size_t>(i + 1)];
//                 for (int p = su; p < eu; ++p) {
//                     const int j = edges_up[p];
//                     if (j == i) continue;
//                     // type 2 for each active (i,j)
//                     ++cnt;
//                 }
//                 i -= worldsize;
//             }
//             // barrier (type 3)
//             ++cnt;
//         }

//         // Region 2 (owner-gated; full j range with neighbor-k scan)
//         {
//             for (int i = num_tiles - 1; i >= 0; --i) {
//                 const int d = idx_of(i,i);
//                 const int owner_ii = (d >= 0) ? (d % worldsize) : -1;
//                 if (myrank == owner_ii) {
//                     // type 4
//                     ++cnt;
//                     // type 5 for neighbors k>i
//                     const int su = off_up[static_cast<std::size_t>(i)];
//                     const int eu = off_up[static_cast<std::size_t>(i + 1)];
//                     for (int p = su; p < eu; ++p) {
//                         const int k = edges_up[p];
//                         if (k > i) ++cnt;
//                     }
//                     // type 6
//                     ++cnt;
//                 }

//                 const int su = off_up[static_cast<std::size_t>(i)];
//                 const int eu = off_up[static_cast<std::size_t>(i + 1)];
//                 for (int j = num_tiles - 1; j >= i; --j) {
//                     if (j == i) continue;
//                     const int owner = owner_of(i,j);
//                     if (myrank != owner) continue;

//                     for (int q = su; q < eu; ++q) {
//                         const int k = edges_up[q];
//                         if (k <= i) continue;
//                         if (k > j) {
//                             if (has_up(i,j) && has_up(j,k)) ++cnt; // type 7
//                         } else {
//                             if (has_up(i,j) && has_up(k,j)) ++cnt; // type 8
//                         }
//                     }
//                     // finalize (type 9)
//                     ++cnt;
//                 }
//             }
//         }

//         return cnt;
//     };

//     const int total = count_rows();
//     if (total == 0) return 0;
//     out_tasks.resize(static_cast<std::size_t>(total));

//     // -------------------------
//     // Pass 2: emit
//     // -------------------------
//     int w = 0;
//     auto emit = [&](int t,int i,int j,int k,int i1,int i2,int i3){
//         auto& r = out_tasks[static_cast<std::size_t>(w++)];
//         r[0]=t; r[1]=i; r[2]=j; r[3]=k; r[4]=i1; r[5]=i2; r[6]=i3;
//     };

//     // Region 1 (neighbors only)
//     {
//         int i = num_tiles - 1 - myrank;
//         while (i >= 0) {
//             const int d = idx_of(i,i);
//             emit(1, i,i,0, d,d,0);
//             const int su = off_up[static_cast<std::size_t>(i)];
//             const int eu = off_up[static_cast<std::size_t>(i + 1)];
//             for (int p = su; p < eu; ++p) {
//                 const int j = edges_up[p];
//                 if (j == i) continue;
//                 const int ij = idx_of(i,j);
//                 emit(2, i,j,0, d,ij,0);
//             }
//             i -= worldsize;
//         }
//         // barrier (type 3)
//         emit(3, 0,0,0, 0,0,0);
//     }

//     // Region 2 (owner-gated; full j range)
//     {
//         for (int i = num_tiles - 1; i >= 0; --i) {
//             const int d = idx_of(i,i);
//             const int owner_ii = (d >= 0) ? (d % worldsize) : -1;
//             if (myrank == owner_ii) {
//                 emit(4, i,i,0, d,0,0);
//                 const int su = off_up[static_cast<std::size_t>(i)];
//                 const int eu = off_up[static_cast<std::size_t>(i + 1)];
//                 for (int p = su; p < eu; ++p) {
//                     const int k = edges_up[p];
//                     if (k > i) {
//                         const int ik = idx_of(i,k);
//                         emit(5, i,i,k, d,ik,0);
//                     }
//                 }
//                 emit(6, i,i,0, d,0,0);
//             }

//             const int su = off_up[static_cast<std::size_t>(i)];
//             const int eu = off_up[static_cast<std::size_t>(i + 1)];
//             for (int j = num_tiles - 1; j >= i; --j) {
//                 if (j == i) continue;
//                 const int owner = owner_of(i,j);
//                 if (myrank != owner) continue;

//                 for (int q = su; q < eu; ++q) {
//                     const int k = edges_up[q];
//                     if (k <= i) continue;
//                     if (k > j) {
//                         if (has_up(i,j) && has_up(j,k)) {
//                             const int ik = idx_of(i,k);
//                             const int jk = idx_of(j,k);
//                             const int ij = idx_of(i,j);
//                             emit(7, i,j,k, ik,jk,ij);
//                         }
//                     } else {
//                         if (has_up(i,j) && has_up(k,j)) {
//                             const int ik = idx_of(i,k);
//                             const int kj = idx_of(k,j);
//                             const int ij = idx_of(i,j);
//                             emit(8, i,j,k, ik,kj,ij);
//                         }
//                     }
//                 }
//                 // finalize on owner
//                 emit(9, i,j,0, 0,0,0);
//             }
//         }
//     }

//     return total;
// }

// inline int inv_scaled_variant4(const TileIndexer::State& state_in,
//                                TileIndexer::Method method,
//                                int myrank,
//                                int worldsize,
//                                int num_tiles,
//                                const std::function<int(int,int)>& map_id,
//                                std::vector<std::array<int,7>>& out_tasks,
//                                int panel_height = 64)
// {
//     out_tasks.clear();
//     if (num_tiles <= 0 || worldsize <= 0) return 0;

//     std::vector<std::array<int,7>> staging;
//     const int produced = inv_scaled_variant1(state_in,
//                                              method,
//                                              myrank,
//                                              worldsize,
//                                              num_tiles,
//                                              map_id,
//                                              staging);
//     if (produced <= 0) {
//         return produced;
//     }

//     const int panel = std::max(1, panel_height);
//     const int num_panels = std::max(1, (num_tiles + panel - 1) / panel);

//     auto barrier_it = std::find_if(staging.begin(), staging.end(),
//                                    [](const std::array<int,7>& row) { return row[0] == 3; });
//     auto stage2_begin = (barrier_it != staging.end()) ? std::next(barrier_it) : staging.end();

//     std::vector<std::vector<std::array<int,7>>> stage1_bins(static_cast<std::size_t>(num_panels));
//     for (auto it = staging.begin(); it != barrier_it; ++it) {
//         const int i = (*it)[1];
//         const int panel_id = (i >= 0) ? std::min(num_panels - 1, i / panel) : 0;
//         stage1_bins[static_cast<std::size_t>(panel_id)].push_back(std::move(*it));
//     }

//     std::vector<std::vector<std::array<int,7>>> stage2_bins(static_cast<std::size_t>(num_panels));
//     for (auto it = stage2_begin; it != staging.end(); ++it) {
//         const int i = (*it)[1];
//         const int panel_id = (i >= 0) ? std::min(num_panels - 1, i / panel) : 0;
//         stage2_bins[static_cast<std::size_t>(panel_id)].push_back(std::move(*it));
//     }

//     auto stage1_cmp = [](const std::array<int,7>& a, const std::array<int,7>& b) {
//         if (a[1] != b[1]) return a[1] > b[1];
//         if (a[0] != b[0]) return a[0] < b[0];
//         if (a[2] != b[2]) return a[2] < b[2];
//         return a[3] < b[3];
//     };
//     auto stage2_cmp = [](const std::array<int,7>& a, const std::array<int,7>& b) {
//         if (a[1] != b[1]) return a[1] > b[1];
//         if (a[2] != b[2]) return a[2] > b[2];
//         if (a[0] != b[0]) return a[0] < b[0];
//         if (a[3] != b[3]) return a[3] < b[3];
//         if (a[4] != b[4]) return a[4] < b[4];
//         return a[5] < b[5];
//     };

//     out_tasks.reserve(staging.size());
//     for (auto& bin : stage1_bins) {
//         if (bin.empty()) continue;
//         std::sort(bin.begin(), bin.end(), stage1_cmp);
//         out_tasks.insert(out_tasks.end(),
//                          std::make_move_iterator(bin.begin()),
//                          std::make_move_iterator(bin.end()));
//     }
//     if (barrier_it != staging.end()) {
//         out_tasks.push_back(std::move(*barrier_it));
//     }
//     for (auto& bin : stage2_bins) {
//         if (bin.empty()) continue;
//         std::sort(bin.begin(), bin.end(), stage2_cmp);
//         out_tasks.insert(out_tasks.end(),
//                          std::make_move_iterator(bin.begin()),
//                          std::make_move_iterator(bin.end()));
//     }

//     return static_cast<int>(out_tasks.size());
// }


// } // namespace alg

// #endif
