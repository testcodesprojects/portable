/**
 * @file chol_algorithms.hpp
 * @brief Cholesky factorization algorithms for tiled sparse matrices.
 *
 * This file implements task collection and scheduling algorithms for parallel
 * Cholesky factorization on tiled sparse matrices. Key components include:
 * - CSR adjacency view for efficient graph traversal
 * - Task generation for POTRF, TRSM, SYRK, and GEMM operations
 * - Load-balanced task distribution across multiple cores
 * - Support for both dense and sparse tile configurations
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

#ifndef STILES_CHOL_ALGORITHMS_HPP
#define STILES_CHOL_ALGORITHMS_HPP

#include "../TileIndexer/TileIndexer.hpp"
#include "../TileIndexer/TileIndexerGraphBuilder.hpp"
#include "../TileIndexer/TileIndexerMemoryUtils.hpp"
#include "../common/stiles_structs.hpp"
#include "../common/stiles_utils.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <vector>

namespace alg {

    // CSRAdjacencyView: helper to navigate CSR upper/lower graphs with small caches.
    struct CSRAdjacencyView {
        const int* up_offsets = nullptr;
        const int* up_edges   = nullptr;
        const int* lo_offsets = nullptr;
        const int* lo_edges   = nullptr;
        int N = 0;

        int cached_m = -1;      // cached column for lower graph lookups
        int lo_cache_start = 0; // start index into lo_edges for cached_m
        int lo_cache_len = 0;   // number of neighbors for cached_m

        CSRAdjacencyView() = default;
        CSRAdjacencyView(const int* up_off,
                         const int* up_edg,
                         const int* lo_off,
                         const int* lo_edg,
                         int num_tiles)
            : up_offsets(up_off), up_edges(up_edg),
              lo_offsets(lo_off), lo_edges(lo_edg),
              N(num_tiles) {}

        inline int rowLengthForK(int k) const { return N - k; }
        inline int degreeUpper(int i)   const { return up_offsets[i + 1] - up_offsets[i]; }
        inline int degreeLower(int i)   const { return lo_offsets[i + 1] - lo_offsets[i]; }

        static inline bool containsSorted(const int* base, int len, int key) {
            constexpr int kTiny = 16;
            if (len <= kTiny) { for (int i = 0; i < len; ++i) if (base[i] == key) return true; return false; }
            return std::binary_search(base, base + len, key);
        }

        inline void cacheLowerOf(int m) {
            if (m != cached_m) {
                cached_m = m;
                lo_cache_start = lo_offsets[m];
                lo_cache_len   = lo_offsets[m + 1] - lo_cache_start;
            }
        }
        inline bool lowerHasCached(int x) const {
            return containsSorted(lo_edges + lo_cache_start, lo_cache_len, x);
        }
        inline bool upperHasEdge(int n, int m) const {
            const int su = up_offsets[n];
            const int du = up_offsets[n + 1] - su;
            return containsSorted(up_edges + su, du, m);
        }
        // Adjacency test between n and m; pick the cheaper side to probe.
        inline bool areAdjacentNM(int n, int m) {
            if (n > m) std::swap(n, m);
            cacheLowerOf(m);
            const int dl = lo_cache_len;
            const int du = degreeUpper(n);
            if (du + du < dl) return upperHasEdge(n, m);
            return lowerHasCached(n);
        }
        // Edge (k,m) via lower graph (uses cached lower for m)
        inline bool hasEdgeKMInLower(int k, int m) {
            cacheLowerOf(m);
            return lowerHasCached(k);
        }
    };

    inline int chol_expansion_variant1(TileIndexer::State& state,
                                       TileIndexer::Method method,
                                       int myrank,
                                       int worldsize,
                                       int num_tiles,
                                       const std::function<int(int,int)>& map_id,
                                       std::vector<std::array<int, 7>>& out_tasks)
{
    out_tasks.clear();

    if (num_tiles <= 0 || worldsize <= 0) return 0;

    tilecounter_utils::bind_is_active(state, method);
    auto index_of = [&](int a, int b) -> int {
        return map_id(a, b);
    };

    auto is_active_tile = [&](int a, int b) -> bool {
        if (state.isActive(a, b, num_tiles)) return true;
        if (a > b) std::swap(a, b);
        const std::size_t idx = TileIndexer::upper_tile_index(a, b, num_tiles);
        return (!state.lazy_index_map.empty() && idx < state.lazy_index_map.size())
                 ? state.lazy_index_map[idx] >= 0
                 : false;
    };

    auto push_task = [&](int type, int m, int k, int n, int idx1, int idx2, int idx3) {
        std::array<int, 7> row{};
        row[0] = type;
        row[1] = m;
        row[2] = k;
        row[3] = n;
        row[4] = idx1;
        row[5] = idx2;
        row[6] = idx3;
        out_tasks.push_back(row);
    };

    int k = 0;
    int m = myrank;
    while (m >= num_tiles) {
        ++k;
        m = m - num_tiles + k;
    }
    int n = 0;

    while (k < num_tiles && m < num_tiles) {
        int next_n = n + 1;
        int next_m = m;
        int next_k = k;

        if (next_n > next_k) {
            next_m += worldsize;
            while (next_m >= num_tiles && next_k < num_tiles) {
                ++next_k;
                next_m = next_m - num_tiles + next_k;
            }
            next_n = 0;
        }

        if (m == k) {
            if (n == k) {
                const int idx1 = index_of(k, k);
                push_task(1, m, k, n, idx1, idx1, 0);
            } else if (is_active_tile(n, k)) {
                const int idx1 = index_of(n, k);
                const int idx2 = index_of(k, k);
                push_task(2, m, k, n, idx1, idx2, 0);
            }
        } else {
            if (n == k) {
                if (is_active_tile(k, m)) {
                    const int idx1 = index_of(k, m);
                    const int idx2 = index_of(k, k);
                    push_task(3, m, k, n, idx1, idx2, idx1);
                }
            } else {
                if (is_active_tile(n, k) && is_active_tile(n, m)) {
                    const int idx1 = index_of(n, k);
                    const int idx2 = index_of(n, m);
                    const int idx3 = index_of(k, m);
                    push_task(4, m, k, n, idx1, idx2, idx3);
                }
            }
        }

        n = next_n;
        m = next_m;
        k = next_k;
    }

    return static_cast<int>(out_tasks.size());
}

    inline int chol_expansion_variant2(const TileIndexer::State& state,
                                    TileIndexer::Method method,
                                    int myrank,
                                    int worldsize,
                                    int num_tiles,
                                    const std::function<int(int,int)>& map_id,
                                    std::vector<std::array<int, 7>>& out_tasks)
    {
        out_tasks.clear();

        if (num_tiles <= 0 || worldsize <= 0) return 0;

        (void)method;

        // Precondition: caller must ensure graphs are built with include_self=true
        // Requires: state.graphs_built && state.graph_N == num_tiles && state.graph_include_self

        const auto& off_up   = state.graph_off_up;
        const auto& edges_up = state.graph_edges_up;
        const auto& off_lo   = state.graph_off_lo;
        const auto& edges_lo = state.graph_edges_lo;

        const auto row_len = [&](int k) -> int { return num_tiles - k; };
        const auto deg_up = [&](int i) -> int { return off_up[static_cast<std::size_t>(i + 1)] - off_up[static_cast<std::size_t>(i)]; };
        const auto deg_lo = [&](int i) -> int { return off_lo[static_cast<std::size_t>(i + 1)] - off_lo[static_cast<std::size_t>(i)]; };

        constexpr int kTiny = 16;
        const auto contains_sorted_span = [&](const int* base, int len, int key) -> bool {
            if (len <= kTiny) {
                for (int i = 0; i < len; ++i) {
                    if (base[i] == key) return true;
                }
                return false;
            }
            return std::binary_search(base, base + len, key);
        };

        int cached_m  = -1;
        int cached_Ls =  0;
        int cached_Ln =  0;
        const auto ensure_lo_cache = [&](int m) {
            if (m != cached_m) {
                cached_m  = m;
                cached_Ls = off_lo[static_cast<std::size_t>(m)];
                const int cached_Le = off_lo[static_cast<std::size_t>(m + 1)];
                cached_Ln = cached_Le - cached_Ls;
            }
        };

        const auto has_in_lo_cached = [&](int x) -> bool {
            return contains_sorted_span(edges_lo.data() + cached_Ls, cached_Ln, x);
        };

        const auto has_in_up = [&](int n, int m) -> bool {
            const int su = off_up[static_cast<std::size_t>(n)];
            const int eu = off_up[static_cast<std::size_t>(n + 1)];
            const int du = eu - su;
            return contains_sorted_span(edges_up.data() + su, du, m);
        };

        const auto adj_nm = [&](int n, int m) -> bool {
            if (n > m) std::swap(n, m);
            ensure_lo_cache(m);
            const int dl = cached_Ln;
            const int du = deg_up(n);
            if (du + du < dl) {
                return has_in_up(n, m);
            }
            return has_in_lo_cached(n);
        };

        const auto has_edge_km_via_lo = [&](int k, int m) -> bool {
            ensure_lo_cache(m);
            return has_in_lo_cached(k);
        };

        const auto index_of = [&](int a, int b) -> int {
            return map_id(a, b);
        };

        auto push_task = [&](int type, int m, int k, int n, int idx1, int idx2, int idx3) {
            std::array<int, 7> row{};
            row[0] = type;
            row[1] = m;
            row[2] = k;
            row[3] = n;
            row[4] = idx1;
            row[5] = idx2;
            row[6] = idx3;
            out_tasks.push_back(row);
        };

        int tmp_k = 0;
        int tmp_m = myrank;
        int tmp_n = 0;

        while (tmp_k < num_tiles && tmp_m >= row_len(tmp_k)) {
            tmp_m -= row_len(tmp_k);
            ++tmp_k;
        }

        while (tmp_k < num_tiles) {
            const int ind_k = tmp_k;
            const int ind_m = ind_k + tmp_m;
            const int k_lo_deg = deg_lo(ind_k);

            if (tmp_n < k_lo_deg) {
                const int base = off_lo[static_cast<std::size_t>(ind_k)];
                const int ind_n = edges_lo[static_cast<std::size_t>(base + tmp_n)];
                bool accept = false;

                if (ind_m == ind_k) {
                    accept = true;
                } else if (ind_n == ind_k) {
                    accept = has_edge_km_via_lo(ind_k, ind_m);
                } else {
                    accept = adj_nm(ind_n, ind_m);
                }

                if (accept) {
                    if (ind_m == ind_k) {
                        if (ind_n == ind_k) {
                            const int idx1 = index_of(ind_k, ind_k);
                            push_task(1, ind_m, ind_k, ind_n, idx1, idx1, 0);
                        } else {
                            const int idx1 = index_of(ind_n, ind_k);
                            const int idx2 = index_of(ind_k, ind_k);
                            push_task(2, ind_m, ind_k, ind_n, idx1, idx2, 0);
                        }
                    } else {
                        if (ind_n == ind_k) {
                            const int idx1 = index_of(ind_k, ind_m);
                            const int idx2 = index_of(ind_k, ind_k);
                            push_task(3, ind_m, ind_k, ind_n, idx1, idx2, idx1);
                        } else {
                            const int idx1 = index_of(ind_n, ind_k);
                            const int idx2 = index_of(ind_n, ind_m);
                            const int idx3 = index_of(ind_k, ind_m);
                            push_task(4, ind_m, ind_k, ind_n, idx1, idx2, idx3);
                        }
                    }
                }
            }

            int tmp_next_n = tmp_n + 1;
            int tmp_next_m = tmp_m;
            int tmp_next_k = tmp_k;

            if (tmp_next_n > tmp_next_k) {
                tmp_next_m += worldsize;
                while (tmp_next_k < num_tiles && tmp_next_m >= row_len(tmp_next_k)) {
                    tmp_next_m -= row_len(tmp_next_k);
                    ++tmp_next_k;
                }
                tmp_next_n = 0;
            }

            tmp_n = tmp_next_n;
            tmp_m = tmp_next_m;
            tmp_k = tmp_next_k;
        }

        return static_cast<int>(out_tasks.size());
    }

    inline int chol_expansion_variant3(const TileIndexer::State& state_in,
                                    TileIndexer::Method method,
                                    int myrank,
                                    int worldsize,
                                    int num_tiles,
                                    const std::function<int(int,int)>& map_id,
                                    std::vector<std::array<int, 7>>& out_tasks)
    {
        out_tasks.clear();
        if (num_tiles <= 0 || worldsize <= 0) return 0;

        (void)method;

        // Precondition: caller must ensure graphs are built with include_self=true
        // Requires: state_in.graphs_built && state_in.graph_N == num_tiles && state_in.graph_include_self
        const TileIndexer::State& state = state_in;

        // CSR-like graph views (upper and lower)
        const int* __restrict up_off  = state.graph_off_up.data();
        const int* __restrict up_edg  = state.graph_edges_up.data();
        const int* __restrict lo_off  = state.graph_off_lo.data();
        const int* __restrict lo_edg  = state.graph_edges_lo.data();

        CSRAdjacencyView adj(up_off, up_edg, lo_off, lo_edg, num_tiles);

        // -------------------------
        // Pass 1: count exact tasks
        // -------------------------
        int total = 0;
        {
            int cur_k = 0;
            int cur_m = myrank;
            int cur_n = 0;

            // Position (k,m) to the starting slot for this rank
            while (cur_k < num_tiles && cur_m >= adj.rowLengthForK(cur_k)) {
                cur_m -= adj.rowLengthForK(cur_k);
                ++cur_k;
            }

            while (cur_k < num_tiles) {
                const int k_idx = cur_k;
                const int m_idx = k_idx + cur_m;

                // Iterate n over neighbors of k in the lower graph (include_self=true ensures n==k appears)
                const int k_lo_start = lo_off[k_idx];
                const int k_lo_deg   = adj.degreeLower(k_idx);

                if (cur_n < k_lo_deg) {
                    const int n_idx = lo_edg[k_lo_start + cur_n];

                    bool accept = false;
                    if (m_idx == k_idx) {
                        // m == k: accept all n in N(k) (self included)
                        accept = true;
                    } else if (n_idx == k_idx) {
                        // n == k: accept iff (k,m) edge exists
                        accept = adj.hasEdgeKMInLower(k_idx, m_idx);
                    } else {
                        // general case: n adjacent to both k and m
                        accept = adj.areAdjacentNM(n_idx, m_idx);
                    }

                    if (accept) ++total;

                    // ---- Skip "empty n" iterations (no more neighbors for this k) ----
                    if (cur_n + 1 >= k_lo_deg) {
                        int next_k = cur_k;
                        int next_m = cur_m + worldsize;
                        while (next_k < num_tiles && next_m >= adj.rowLengthForK(next_k)) {
                            next_m -= adj.rowLengthForK(next_k);
                            ++next_k;
                        }
                        cur_k = next_k;
                        cur_m = next_m;
                        cur_n = 0;
                        continue; // move to next (k,m)
                    }
                }

                // advance schedule cursor one step in n
                int next_n = cur_n + 1;
                int next_m = cur_m;
                int next_k = cur_k;

                if (next_n > next_k) {
                    next_m += worldsize;
                    while (next_k < num_tiles && next_m >= adj.rowLengthForK(next_k)) {
                        next_m -= adj.rowLengthForK(next_k);
                        ++next_k;
                    }
                    next_n = 0;
                }

                cur_n = next_n;
                cur_m = next_m;
                cur_k = next_k;
            }
        }
        if (total == 0) return 0;

        // Pre-size the output exactly; we'll assign by index (no push_back).
        out_tasks.resize(static_cast<std::size_t>(total));

        // --------------------------------------------
        // Pass 2: emit rows with light-weight caching
        // --------------------------------------------
        int write_pos = 0;

        // Index caches to avoid redundant map_id lookups
        struct IndexCaches {
            const std::function<int(int,int)>& map;
            int diag_k = -1;
            int diag_idx = -1;
            int km_k = -1;
            int km_m = -1;
            int km_idx = -1;
            inline int diagIndexFor(int k) {
                if (k != diag_k) { diag_k = k; diag_idx = map(k,k); km_k = -1; km_m = -1; }
                return diag_idx;
            }
            inline int kmIndexFor(int k, int m) {
                if (k != km_k || m != km_m) { km_k = k; km_m = m; km_idx = map(k,m); }
                return km_idx;
            }
        } caches{map_id};

        // Re-run the same schedule, now emitting rows
        {
            int cur_k = 0;
            int cur_m = myrank;
            int cur_n = 0;

            while (cur_k < num_tiles && cur_m >= adj.rowLengthForK(cur_k)) {
                cur_m -= adj.rowLengthForK(cur_k);
                ++cur_k;
            }

            while (cur_k < num_tiles) {
                const int k_idx = cur_k;
                const int m_idx = k_idx + cur_m;

                const int k_lo_start = lo_off[k_idx];
                const int k_lo_deg   = adj.degreeLower(k_idx);

                if (cur_n < k_lo_deg) {
                    const int n_idx = lo_edg[k_lo_start + cur_n];

                    bool accept = false;
                    if (m_idx == k_idx) {
                        accept = true;
                    } else if (n_idx == k_idx) {
                        accept = adj.hasEdgeKMInLower(k_idx, m_idx);
                    } else {
                        accept = adj.areAdjacentNM(n_idx, m_idx);
                    }

                    if (accept) {
                        if (m_idx == k_idx) {
                            if (n_idx == k_idx) {
                                const int d = caches.diagIndexFor(k_idx);
                                auto& row = out_tasks[static_cast<std::size_t>(write_pos++)];
                                row[0]=1; row[1]=m_idx; row[2]=k_idx; row[3]=n_idx; row[4]=d; row[5]=d; row[6]=0;
                            } else {
                                const int i1 = map_id(n_idx, k_idx);
                                const int d  = caches.diagIndexFor(k_idx);
                                auto& row = out_tasks[static_cast<std::size_t>(write_pos++)];
                                row[0]=2; row[1]=m_idx; row[2]=k_idx; row[3]=n_idx; row[4]=i1; row[5]=d; row[6]=0;
                            }
                        } else {
                            if (n_idx == k_idx) {
                                const int km = caches.kmIndexFor(k_idx, m_idx);
                                const int d  = caches.diagIndexFor(k_idx);
                                auto& row = out_tasks[static_cast<std::size_t>(write_pos++)];
                                row[0]=3; row[1]=m_idx; row[2]=k_idx; row[3]=n_idx; row[4]=km; row[5]=d; row[6]=km;
                            } else {
                                const int i1 = map_id(n_idx, k_idx);
                                const int i2 = map_id(n_idx, m_idx);
                                const int i3 = caches.kmIndexFor(k_idx, m_idx);
                                auto& row = out_tasks[static_cast<std::size_t>(write_pos++)];
                                row[0]=4; row[1]=m_idx; row[2]=k_idx; row[3]=n_idx; row[4]=i1; row[5]=i2; row[6]=i3;
                            }
                        }
                    }

                    // ---- Skip "empty n" iterations (no more neighbors for this k) ----
                    if (cur_n + 1 >= k_lo_deg) {
                        int next_k = cur_k;
                        int next_m = cur_m + worldsize;
                        while (next_k < num_tiles && next_m >= adj.rowLengthForK(next_k)) {
                            next_m -= adj.rowLengthForK(next_k);
                            ++next_k;
                        }
                        cur_k = next_k;
                        cur_m = next_m;
                        cur_n = 0;
                        continue; // move to next (k,m)
                    }
                }

                int next_n = cur_n + 1;
                int next_m = cur_m;
                int next_k = cur_k;

                if (next_n > next_k) {
                    next_m += worldsize;
                    while (next_k < num_tiles && next_m >= adj.rowLengthForK(next_k)) {
                        next_m -= adj.rowLengthForK(next_k);
                        ++next_k;
                    }
                    next_n = 0;
                }

                cur_n = next_n;
                cur_m = next_m;
                cur_k = next_k;
            }
        }

        return total;
    }

    inline int chol_expansion_variant4(const TileIndexer::State& state_in,
                                       TileIndexer::Method method,
                                       int myrank,
                                       int worldsize,
                                       int num_tiles,
                                       const std::function<int(int,int)>& map_id,
                                       std::vector<std::array<int, 7>>& out_tasks,
                                       int panel_height = 64)
    {
        out_tasks.clear();
        if (num_tiles <= 0 || worldsize <= 0) return 0;

        (void)method;

        const TileIndexer::State& state = state_in;
        const int* __restrict up_off  = state.graph_off_up.data();
        const int* __restrict up_edg  = state.graph_edges_up.data();
        const int* __restrict lo_off  = state.graph_off_lo.data();
        const int* __restrict lo_edg  = state.graph_edges_lo.data();

        CSRAdjacencyView adj(up_off, up_edg, lo_off, lo_edg, num_tiles);

        const int panel = std::max(1, panel_height);
        const int num_panels = std::max(1, (num_tiles + panel - 1) / panel);
        std::vector<std::vector<std::array<int, 7>>> panel_bins(static_cast<std::size_t>(num_panels));

        struct IndexCaches {
            const std::function<int(int,int)>& map;
            int diag_k = -1;
            int diag_idx = -1;
            int km_k = -1;
            int km_m = -1;
            int km_idx = -1;
            inline int diagIndexFor(int k) {
                if (k != diag_k) {
                    diag_k = k;
                    diag_idx = map(k, k);
                    km_k = -1;
                    km_m = -1;
                }
                return diag_idx;
            }
            inline int kmIndexFor(int k, int m) {
                if (k != km_k || m != km_m) {
                    km_k = k;
                    km_m = m;
                    km_idx = map(k, m);
                }
                return km_idx;
            }
        } caches{map_id};

        int produced = 0;

        int cur_k = 0;
        int cur_m = myrank;
        int cur_n = 0;

        while (cur_k < num_tiles && cur_m >= adj.rowLengthForK(cur_k)) {
            cur_m -= adj.rowLengthForK(cur_k);
            ++cur_k;
        }

        while (cur_k < num_tiles) {
            const int k_idx = cur_k;
            const int m_idx = k_idx + cur_m;

            const int k_lo_start = lo_off[k_idx];
            const int k_lo_deg   = adj.degreeLower(k_idx);

            if (cur_n < k_lo_deg) {
                const int n_idx = lo_edg[k_lo_start + cur_n];

                bool accept = false;
                if (m_idx == k_idx) {
                    accept = true;
                } else if (n_idx == k_idx) {
                    accept = adj.hasEdgeKMInLower(k_idx, m_idx);
                } else {
                    // DGEMM: C(k,m) -= A(n,k)^T * B(n,m)
                    // Check adjacency AND that output tile C(k,m) exists in mapper.
                    // Tile adjacency over-approximates: A and B may both exist but
                    // their active columns don't overlap, so C has no L entries.
                    accept = adj.areAdjacentNM(n_idx, m_idx) && (map_id(k_idx, m_idx) >= 0);
                }

                if (accept) {
                    std::array<int, 7> row{};
                    row[1] = m_idx;
                    row[2] = k_idx;
                    row[3] = n_idx;

                    if (m_idx == k_idx) {
                        if (n_idx == k_idx) {
                            const int d = caches.diagIndexFor(k_idx);
                            row[0] = 1;
                            row[4] = d;
                            row[5] = d;
                            row[6] = 0;
                        } else {
                            const int i1 = map_id(n_idx, k_idx);
                            const int d  = caches.diagIndexFor(k_idx);
                            row[0] = 2;
                            row[4] = i1;
                            row[5] = d;
                            row[6] = 0;
                        }
                    } else {
                        if (n_idx == k_idx) {
                            const int km = caches.kmIndexFor(k_idx, m_idx);
                            const int d  = caches.diagIndexFor(k_idx);
                            row[0] = 3;
                            row[4] = km;
                            row[5] = d;
                            row[6] = km;
                        } else {
                            const int i1 = map_id(n_idx, k_idx);
                            const int i2 = map_id(n_idx, m_idx);
                            const int i3 = caches.kmIndexFor(k_idx, m_idx);
                            row[0] = 4;
                            row[4] = i1;
                            row[5] = i2;
                            row[6] = i3;
                        }
                    }

                    const int panel_id = (k_idx >= 0) ? std::min(num_panels - 1, k_idx / panel) : 0;
                    panel_bins[static_cast<std::size_t>(panel_id)].push_back(row);
                    ++produced;
                }

                if (cur_n + 1 >= k_lo_deg) {
                    int next_k = cur_k;
                    int next_m = cur_m + worldsize;
                    while (next_k < num_tiles && next_m >= adj.rowLengthForK(next_k)) {
                        next_m -= adj.rowLengthForK(next_k);
                        ++next_k;
                    }
                    cur_k = next_k;
                    cur_m = next_m;
                    cur_n = 0;
                    continue;
                }
            }

            int next_n = cur_n + 1;
            int next_m = cur_m;
            int next_k = cur_k;

            if (next_n > next_k) {
                next_m += worldsize;
                while (next_k < num_tiles && next_m >= adj.rowLengthForK(next_k)) {
                    next_m -= adj.rowLengthForK(next_k);
                    ++next_k;
                }
                next_n = 0;
            }

            cur_n = next_n;
            cur_m = next_m;
            cur_k = next_k;
        }

        out_tasks.reserve(static_cast<std::size_t>(produced));
        for (auto& bin : panel_bins) {
            if (bin.empty()) continue;
            out_tasks.insert(out_tasks.end(),
                             std::make_move_iterator(bin.begin()),
                             std::make_move_iterator(bin.end()));
        }

        return produced;
    }

    // Adjacency helpers (same pattern you used in variant2/3)
    #ifndef STILES_CONTAINS_SORTED_SPAN_HELPER
    inline bool contains_sorted_span(const int* base, int len, int key) {
        constexpr int kTiny = 16;
        if (len <= kTiny) { for (int i=0;i<len;++i) if (base[i]==key) return true; return false; }
        return std::binary_search(base, base+len, key);
    }
    #define STILES_CONTAINS_SORTED_SPAN_HELPER 1
    #endif

    // Local index of a red tree (diagonal or off-diagonal) inside the bottom-right sep×sep block.
    inline int redtree_local_index(int k, int m, int N, int sep) {
        const int base = N - sep;
        const int kk = k - base;
        const int mm = m - base;
        if (kk < 0 || mm < kk || kk >= sep || mm >= sep) return -1;
        return kk * (2*sep - kk - 1) / 2 + mm;
    }

    // Region-A partition view for the composed ND+tree path. Each entry is one
    // NON-root partition (leaf or intermediate separator) lying in [0, cutoff).
    // When supplied, Region A is collected PER-PARTITION — rank_in_region =
    // (myrank - first_core) strides within the partition's [first_core,
    // first_core+num_cores) core group — instead of by a single global
    // m-stride. That gives ND leaf parallelism while Region B (the root
    // separator corner) stays the tree reduction. Null ⇒ flat global-stride
    // Region A (original behaviour). The emitted task SET is identical either
    // way: cross-partition tiles are inactive (ND independence), so a walk over
    // column k only emits tiles owned by k's partition — no gaps, no overlap.
    struct RegionAPartition { int start_tile; int end_tile; int first_core; int num_cores; };

    // Region A is collected by `worldsize` threads (= P, the collection-thread
    // count) via m-striding. Region B's GEMM slicing is per-execution-rank, so it
    // uses `worldsize_b` (= a, the execution-rank count) and is only emitted by
    // collection ranks myrank < worldsize_b. This decouples collection
    // parallelism (Region A wants all cores) from execution-rank slicing
    // (Region B must match the executor's `a` ranks exactly).
    inline int chol_reduction_variant(const TileIndexer::State& state_in,
                                    TileIndexer::Method /*method*/,
                                    int myrank,
                                    int worldsize,
                                    int worldsize_b,
                                    int num_tiles,
                                    int sep,
                                    TreeLeaf** trees,
                                    const std::function<int(int,int)>& map_id,
                                    std::vector<std::array<int,7>>& out_tasks,
                                    const RegionAPartition* regionA_parts = nullptr,
                                    int regionA_num = 0)
    {
        out_tasks.clear();
        if (num_tiles <= 0 || worldsize <= 0 || sep <= 0) return 0;
        if (worldsize_b <= 0) worldsize_b = worldsize;  // backward-compatible fallback

        // Precondition: caller must ensure graphs are built with include_self=true
        // Requires: state_in.graphs_built && state_in.graph_N == num_tiles && state_in.graph_include_self
        const TileIndexer::State& state = state_in;

        const int* __restrict off_up   = state.graph_off_up.data();
        const int* __restrict edges_up = state.graph_edges_up.data();
        const int* __restrict off_lo   = state.graph_off_lo.data();
        const int* __restrict edges_lo = state.graph_edges_lo.data();

        auto deg_up = [&](int i){ return off_up[i+1] - off_up[i]; };
        auto deg_lo = [&](int i){ return off_lo[i+1] - off_lo[i]; };

        auto adj_nm = [&](int n, int m)->bool{
            if (n > m) std::swap(n,m);
            const int su = off_up[n], eu = off_up[n+1], du = eu - su;
            const int sl = off_lo[m], el = off_lo[m+1], dl = el - sl;
            if (du + du < dl) return contains_sorted_span(edges_up + su, du, m);
            return contains_sorted_span(edges_lo + sl, dl, n);
        };
        auto has_edge_km_via_lo = [&](int k, int m)->bool{
            const int sl = off_lo[m], dl = off_lo[m+1] - sl;
            return contains_sorted_span(edges_lo + sl, dl, k);
        };

        auto index_of = [&](int a, int b){ return map_id(a,b); };

        const int cutoff = num_tiles - sep;
        const int num_sep = (sep*sep - sep)/2 + sep;

        // Per-tree distributions are sliced across `worldsize_b` execution ranks
        // (not `worldsize` collection threads) so that each execution rank gets
        // exactly its share of the slot's GEMMs. Collection ranks with
        // myrank >= worldsize_b skip Region B entirely (see gates below).
        std::vector<sTiles::Utils::TaskDistribution> distr;
        distr.assign(static_cast<std::size_t>(num_sep), sTiles::Utils::TaskDistribution{0,0});
        std::vector<int>              total_tasks(num_sep, 0);
        if (trees && myrank < worldsize_b) {
            for (int kk = 0; kk < sep; ++kk) {
                for (int mm = kk; mm < sep; ++mm) {
                    const int t = kk*(2*sep - kk - 1)/2 + mm;
                    if (trees[t]) {
                        total_tasks[t] = trees[t]->num_tasks;
                        distr[t] = sTiles::Utils::calculateTaskDistribution(myrank, worldsize_b, total_tasks[t]);
                    }
                }
            }
        }

        auto in_rank_slice = [&](int t, int v)->bool{
            if (!trees || !trees[t]) return false;
            const int s = distr[t].start_index, e = s + distr[t].num_tasks;
            return (v >= s && v < e);
        };
        auto at_slice_tail = [&](int t, int v)->bool{
            if (!trees || !trees[t]) return false;
            const int s = distr[t].start_index, e = s + distr[t].num_tasks;
            return (v == e - 1);
        };

        // ---------
        // Pass 1: count
        // ---------
        int total_rows = 0;

        // Region A (green) — variant4-style n iteration over N(k).
        // The CSR edges_lo[off_lo[k]..off_lo[k+1]) are sorted ascending and
        // include k itself (precondition: include_self=true), so iterating
        // them yields the same n order as the variant3 walk but skips inactive
        // positions. Saves the n ∈ N(k) membership test on every (k,m,n).
        {
            // Region A count over column range [k_start,k_end), rank `r` of `ws`
            // striding the (k,m) walk. Driven once (flat) or once per non-root
            // partition (composed ND path) — see regionA_parts.
            auto countA = [&](int k_start, int k_end, int r, int ws) {
                int k = k_start, m = k_start + r;  // diagonal + rank offset (m>=k); flat: k_start=0 → m=myrank
                while (m >= num_tiles && k < k_end) { ++k; m = m - num_tiles + k; }
                while (k < k_end && m < num_tiles) {
                    const int sk = off_lo[k], dk = deg_lo(k);
                    if (m == k) {
                        // Every n ∈ N(k) contributes (n==k → POTRF; n<k → SYRK).
                        for (int ni = 0; ni < dk; ++ni) {
                            const int n = edges_lo[sk + ni];
                            if (n > k) break;
                            ++total_rows;
                        }
                    } else {
                        // Off-diagonal column: TRSM (one, when n==k and (k,m)∈E),
                        // plus GEMM for each n ∈ N(k) ∩ N(m) with n<k.
                        // Tile adjacency over-approximates — emit only when the
                        // output tile (k, m) actually exists in the mapper.
                        const int km_existing = index_of(k, m);
                        if (km_existing >= 0) {
                            bool km_known = false, km_active = false;
                            for (int ni = 0; ni < dk; ++ni) {
                                const int n = edges_lo[sk + ni];
                                if (n > k) break;
                                if (n == k) {
                                    if (!km_known) { km_active = has_edge_km_via_lo(k, m); km_known = true; }
                                    if (km_active) ++total_rows;
                                } else if (adj_nm(n, m)) {
                                    ++total_rows;
                                }
                            }
                        }
                    }
                    // advance to next (k, m)
                    m += ws;
                    while (m >= num_tiles && k < k_end) { ++k; m = m - num_tiles + k; }
                }
            };
            if (regionA_parts) {
                for (int p = 0; p < regionA_num; ++p) {
                    const RegionAPartition& pv = regionA_parts[p];
                    if (myrank >= pv.first_core && myrank < pv.first_core + pv.num_cores)
                        // Cap at cutoff: the root separator [cutoff,num_tiles) is Region B's
                        // exclusive domain. An uncapped end_tile>cutoff makes Region A ALSO emit
                        // the separator-column TRSMs Region B emits -> same tile produced by two
                        // ranks -> 2nd write races the corner reduction read (stcov ts120 >=8 cores).
                        countA(pv.start_tile, (pv.end_tile < cutoff ? pv.end_tile : cutoff), myrank - pv.first_core, pv.num_cores);
                }
            } else {
                countA(0, cutoff, myrank, worldsize);
            }
        }

        // Region B (red) — variant4-style n iteration. Outer loop walks every
        // (k, m) in the separator block (m strides by 1, no rank-striping here);
        // inner loop iterates n over N(k) ∩ [0, k]. gemm_ctr[t] only increments
        // on active triples, so skipping inactive n is safe — counter stays
        // deterministic across collection ranks.
        // Skip entirely on collection ranks beyond the execution-rank count;
        // those ranks have no Region B work to emit.
        if (myrank < worldsize_b) {
            std::vector<int> gemm_ctr(num_sep, 0);
            for (int k = cutoff; k < num_tiles; ++k) {
                const int sk = off_lo[k], dk = deg_lo(k);
                for (int m = k; m < num_tiles; ++m) {
                    if (m == k) {
                        for (int ni = 0; ni < dk; ++ni) {
                            const int n = edges_lo[sk + ni];
                            if (n > k) break;
                            if (n == k) {
                                if (myrank == 0) ++total_rows;   // type 1
                            } else {
                                const int t = redtree_local_index(k, m, num_tiles, sep);
                                if (trees && t >= 0 && trees[t]) {
                                    if (in_rank_slice(t, gemm_ctr[t])) ++total_rows;               // type 5
                                    if (at_slice_tail(t, gemm_ctr[t])) {                            // type 6 (+ type 7 on rank 0)
                                        ++total_rows;                                              // type 6
                                        if (myrank == 0) {
                                            ++total_rows;                                          // type 7
                                            ++gemm_ctr[t];                                        // extra bump for type-7 slot (rank 0 only)
                                        }
                                    }
                                    ++gemm_ctr[t];
                                } else {
                                    if (myrank == 0) ++total_rows;                                  // type 2 (no tree)
                                }
                            }
                        }
                    } else {
                        // Output-tile existence check (mirrors Region A).
                        // The tree path uses tree scratch (not a tile) so it
                        // doesn't need the check, but for the non-tree fallback
                        // and the case-3 DTRSM, the destination must exist.
                        const int km_existing = index_of(k, m);
                        bool km_known = false, km_active = false;
                        for (int ni = 0; ni < dk; ++ni) {
                            const int n = edges_lo[sk + ni];
                            if (n > k) break;
                            if (n == k) {
                                if (!km_known) { km_active = has_edge_km_via_lo(k, m); km_known = true; }
                                if (km_active && myrank == 0 && km_existing >= 0) ++total_rows;     // type 3
                            } else if (adj_nm(n, m)) {
                                const int t = redtree_local_index(k, m, num_tiles, sep);
                                if (trees && t >= 0 && trees[t]) {
                                    if (in_rank_slice(t, gemm_ctr[t])) ++total_rows;               // type 8
                                    if (at_slice_tail(t, gemm_ctr[t])) {
                                        ++total_rows;                                              // type 9
                                        if (myrank == 0) {
                                            ++total_rows;                                          // type 10
                                            ++gemm_ctr[t];                                        // extra bump for type-10 slot (rank 0 only)
                                        }
                                    }
                                    ++gemm_ctr[t];
                                } else {
                                    if (myrank == 0 && km_existing >= 0) ++total_rows;             // type 4 (no tree)
                                }
                            }
                        }
                    }
                }
            }
        }

        if (total_rows == 0) return 0;
        out_tasks.resize(static_cast<std::size_t>(total_rows));

        // ---------
        // Pass 2: emit (with caches)
        // ---------
        int write_pos = 0;
        auto emit = [&](int t,int m,int k,int n,int i1,int i2,int i3){
            auto& r = out_tasks[static_cast<std::size_t>(write_pos++)];
            r[0]=t; r[1]=m; r[2]=k; r[3]=n; r[4]=i1; r[5]=i2; r[6]=i3;
        };

        // caches
        int cached_k = -1, cached_diag = -1;
        auto diag_idx = [&](int k)->int{
            if (k != cached_k) { cached_k = k; cached_diag = index_of(k,k); }
            return cached_diag;
        };
        int cached_km_k = -1, cached_km_m = -1, cached_km = -1;
        auto km_idx = [&](int k,int m)->int{
            if (k != cached_km_k || m != cached_km_m) { cached_km_k=k; cached_km_m=m; cached_km=index_of(k,m); }
            return cached_km;
        };

        // Region A emit — same n-over-N(k) iteration as the count pass.
        {
            auto emitA = [&](int k_start, int k_end, int r, int ws) {
                int k = k_start, m = k_start + r;  // diagonal + rank offset (m>=k); flat: k_start=0 → m=myrank
                while (m >= num_tiles && k < k_end) { ++k; m = m - num_tiles + k; }
                while (k < k_end && m < num_tiles) {
                    const int sk = off_lo[k], dk = deg_lo(k);
                    if (m == k) {
                        for (int ni = 0; ni < dk; ++ni) {
                            const int n = edges_lo[sk + ni];
                            if (n > k) break;
                            if (n == k) {
                                const int d = diag_idx(k);
                                emit(1, m,k,n, d,d,0);
                            } else {
                                const int i1 = index_of(n,k);
                                const int d  = diag_idx(k);
                                emit(2, m,k,n, i1,d,0);
                            }
                        }
                    } else {
                        // Gate on output-tile existence (mirrors count pass).
                        const int km_existing = km_idx(k, m);
                        if (km_existing >= 0) {
                            bool km_known = false, km_active = false;
                            for (int ni = 0; ni < dk; ++ni) {
                                const int n = edges_lo[sk + ni];
                                if (n > k) break;
                                if (n == k) {
                                    if (!km_known) { km_active = has_edge_km_via_lo(k, m); km_known = true; }
                                    if (km_active) {
                                        const int d  = diag_idx(k);
                                        emit(3, m,k,n, km_existing,d,km_existing);
                                    }
                                } else if (adj_nm(n, m)) {
                                    const int i1 = index_of(n,k);
                                    const int i2 = index_of(n,m);
                                    emit(4, m,k,n, i1,i2,km_existing);
                                }
                            }
                        }
                    }
                    m += ws;
                    while (m >= num_tiles && k < k_end) { ++k; m = m - num_tiles + k; }
                }
            };
            if (regionA_parts) {
                for (int p = 0; p < regionA_num; ++p) {
                    const RegionAPartition& pv = regionA_parts[p];
                    if (myrank >= pv.first_core && myrank < pv.first_core + pv.num_cores)
                        // Cap at cutoff (see matching countA above): separator columns are Region B's,
                        // so Region A must not also emit their TRSMs (duplicate-write race).
                        emitA(pv.start_tile, (pv.end_tile < cutoff ? pv.end_tile : cutoff), myrank - pv.first_core, pv.num_cores);
                }
            } else {
                emitA(0, cutoff, myrank, worldsize);
            }
        }

        // Region B emit — same n-over-N(k) iteration as the count pass.
        if (myrank < worldsize_b) {
            std::vector<int> gemm_ctr(num_sep, 0);
            for (int k = cutoff; k < num_tiles; ++k) {
                const int sk = off_lo[k], dk = deg_lo(k);
                for (int m = k; m < num_tiles; ++m) {
                    if (m == k) {
                        for (int ni = 0; ni < dk; ++ni) {
                            const int n = edges_lo[sk + ni];
                            if (n > k) break;
                            if (n == k) {
                                if (myrank == 0) {
                                    const int d = diag_idx(k);
                                    emit(1, m,k,n, d,d,0);
                                }
                            } else {
                                const int t = redtree_local_index(k, m, num_tiles, sep);
                                if (trees && t >= 0 && trees[t]) {
                                    if (in_rank_slice(t, gemm_ctr[t])) {
                                        const int i1 = index_of(n,k);
                                        emit(5, m,k,n, i1, t, 0);
                                    }
                                    if (at_slice_tail(t, gemm_ctr[t])) {
                                        emit(6, m,k,n, 0, t, 165715);
                                        if (myrank == 0) {
                                            const int d = diag_idx(k);
                                            emit(7, m,k,n, t, d, 165715);
                                            ++gemm_ctr[t];
                                        }
                                    }
                                    ++gemm_ctr[t];
                                } else {
                                    if (myrank == 0) {
                                        const int i1 = index_of(n,k);
                                        const int d  = diag_idx(k);
                                        emit(2, m,k,n, i1,d,0);
                                    }
                                }
                            }
                        }
                    } else {
                        // Output-tile existence check for non-tree paths (mirrors count).
                        const int km_existing = km_idx(k, m);
                        bool km_known = false, km_active = false;
                        for (int ni = 0; ni < dk; ++ni) {
                            const int n = edges_lo[sk + ni];
                            if (n > k) break;
                            if (n == k) {
                                if (!km_known) { km_active = has_edge_km_via_lo(k, m); km_known = true; }
                                if (km_active && myrank == 0 && km_existing >= 0) {
                                    const int d  = diag_idx(k);
                                    emit(3, m,k,n, km_existing,d,0);
                                }
                            } else if (adj_nm(n, m)) {
                                const int t = redtree_local_index(k, m, num_tiles, sep);
                                if (trees && t >= 0 && trees[t]) {
                                    if (in_rank_slice(t, gemm_ctr[t])) {
                                        const int i1 = index_of(n,k);
                                        const int i2 = index_of(n,m);
                                        emit(8, m,k,n, i1,i2,t);
                                    }
                                    if (at_slice_tail(t, gemm_ctr[t])) {
                                        emit(9, m,k,n, 0,t,621075);
                                        if (myrank == 0) {
                                            emit(10, m,k,n, t,621075, km_existing);
                                            ++gemm_ctr[t];
                                        }
                                    }
                                    ++gemm_ctr[t];
                                } else {
                                    if (myrank == 0 && km_existing >= 0) {
                                        const int i1 = index_of(n,k);
                                        const int i2 = index_of(n,m);
                                        emit(4, m,k,n, i1,i2,km_existing);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return total_rows;
    }

    inline int chol_reduction_variant4(const TileIndexer::State& state_in,
                                       TileIndexer::Method method,
                                       int myrank,
                                       int worldsize,
                                       int worldsize_b,
                                       int num_tiles,
                                       int sep,
                                       TreeLeaf** trees,
                                       const std::function<int(int,int)>& map_id,
                                       std::vector<std::array<int,7>>& out_tasks,
                                       int panel_height = 64,
                                       const RegionAPartition* regionA_parts = nullptr,
                                       int regionA_num = 0)
    {
        out_tasks.clear();
        if (num_tiles <= 0 || worldsize <= 0 || sep <= 0) return 0;

        std::vector<std::array<int,7>> staging;
        const int produced = chol_reduction_variant(state_in,
                                                    method,
                                                    myrank,
                                                    worldsize,
                                                    worldsize_b,
                                                    num_tiles,
                                                    sep,
                                                    trees,
                                                    map_id,
                                                    staging,
                                                    regionA_parts,
                                                    regionA_num);
        if (produced <= 0) {
            return produced;
        }

        const int panel = std::max(1, panel_height);
        const int num_panels = std::max(1, (num_tiles + panel - 1) / panel);
        std::vector<std::vector<std::array<int,7>>> panel_bins(static_cast<std::size_t>(num_panels));

        for (auto& row : staging) {
            const int k = row[2];
            const int clamped = (k >= 0) ? std::min(num_panels - 1, k / panel) : 0;
            panel_bins[static_cast<std::size_t>(clamped)].push_back(std::move(row));
        }

        auto cmp = [](const std::array<int,7>& a, const std::array<int,7>& b) {
            if (a[2] != b[2]) return a[2] < b[2];
            if (a[1] != b[1]) return a[1] < b[1];
            if (a[3] != b[3]) return a[3] < b[3];
            return a[0] < b[0];
        };

        out_tasks.reserve(staging.size());
        for (auto& bin : panel_bins) {
            if (bin.empty()) continue;
            std::sort(bin.begin(), bin.end(), cmp);
            out_tasks.insert(out_tasks.end(),
                             std::make_move_iterator(bin.begin()),
                             std::make_move_iterator(bin.end()));
        }

        return static_cast<int>(out_tasks.size());
    }

    inline int chol_expansion_variant4_ND(const TileIndexer::State& state_in,
                                          TileIndexer::Method method,
                                          int myrank,
                                          int worldsize,
                                          int num_tiles,
                                          const std::function<int(int,int)>& map_id,
                                          std::vector<std::array<int, 7>>& out_tasks,
                                          int start_tile,
                                          int end_tile,
                                          int panel_height = 64)
    {
        out_tasks.clear();
        if (num_tiles <= 0 || worldsize <= 0) return 0;
        if (start_tile >= end_tile || start_tile < 0 || end_tile > num_tiles) return 0;

        (void)method;

        const TileIndexer::State& state = state_in;
        const int* __restrict up_off  = state.graph_off_up.data();
        const int* __restrict up_edg  = state.graph_edges_up.data();
        const int* __restrict lo_off  = state.graph_off_lo.data();
        const int* __restrict lo_edg  = state.graph_edges_lo.data();

        CSRAdjacencyView adj(up_off, up_edg, lo_off, lo_edg, num_tiles);

        const int panel = std::max(1, panel_height);
        const int num_panels = std::max(1, (num_tiles + panel - 1) / panel);
        std::vector<std::vector<std::array<int, 7>>> panel_bins(static_cast<std::size_t>(num_panels));

        struct IndexCaches {
            const std::function<int(int,int)>& map;
            int diag_k = -1;
            int diag_idx = -1;
            int km_k = -1;
            int km_m = -1;
            int km_idx = -1;
            inline int diagIndexFor(int k) {
                if (k != diag_k) {
                    diag_k = k;
                    diag_idx = map(k, k);
                    km_k = -1;
                    km_m = -1;
                }
                return diag_idx;
            }
            inline int kmIndexFor(int k, int m) {
                if (k != km_k || m != km_m) {
                    km_k = k;
                    km_m = m;
                    km_idx = map(k, m);
                }
                return km_idx;
            }
        } caches{map_id};

        int produced = 0;

        // Start from start_tile instead of 0
        int cur_k = start_tile;
        int cur_m = myrank;
        int cur_n = 0;

        // Align the starting (k, m) pair within the [start_tile, end_tile) window
        while (cur_k < end_tile && cur_m >= adj.rowLengthForK(cur_k)) {
            cur_m -= adj.rowLengthForK(cur_k);
            ++cur_k;
        }

        // Main loop: process tiles in range [start_tile, end_tile)
        while (cur_k < end_tile) {
            const int k_idx = cur_k;
            const int m_idx = k_idx + cur_m;

            const int k_lo_start = lo_off[k_idx];
            const int k_lo_deg   = adj.degreeLower(k_idx);

            if (cur_n < k_lo_deg) {
                const int n_idx = lo_edg[k_lo_start + cur_n];

                bool accept = false;
                if (m_idx == k_idx) {
                    accept = true;
                } else if (n_idx == k_idx) {
                    accept = adj.hasEdgeKMInLower(k_idx, m_idx);
                } else {
                    // DGEMM: C(k,m) -= A(n,k)^T * B(n,m)
                    // Check adjacency AND that output tile C(k,m) exists in mapper.
                    // Tile adjacency over-approximates: A and B may both exist but
                    // their active columns don't overlap, so C has no L entries.
                    accept = adj.areAdjacentNM(n_idx, m_idx) && (map_id(k_idx, m_idx) >= 0);
                }

                if (accept) {
                    std::array<int, 7> row{};
                    row[1] = m_idx;
                    row[2] = k_idx;
                    row[3] = n_idx;

                    if (m_idx == k_idx) {
                        if (n_idx == k_idx) {
                            const int d = caches.diagIndexFor(k_idx);
                            row[0] = 1;
                            row[4] = d;
                            row[5] = d;
                            row[6] = 0;
                        } else {
                            const int i1 = map_id(n_idx, k_idx);
                            const int d  = caches.diagIndexFor(k_idx);
                            row[0] = 2;
                            row[4] = i1;
                            row[5] = d;
                            row[6] = 0;
                        }
                    } else {
                        if (n_idx == k_idx) {
                            const int km = caches.kmIndexFor(k_idx, m_idx);
                            const int d  = caches.diagIndexFor(k_idx);
                            row[0] = 3;
                            row[4] = km;
                            row[5] = d;
                            row[6] = km;
                        } else {
                            const int i1 = map_id(n_idx, k_idx);
                            const int i2 = map_id(n_idx, m_idx);
                            const int i3 = caches.kmIndexFor(k_idx, m_idx);
                            row[0] = 4;
                            row[4] = i1;
                            row[5] = i2;
                            row[6] = i3;
                        }
                    }

                    const int panel_id = (k_idx >= 0) ? std::min(num_panels - 1, k_idx / panel) : 0;
                    panel_bins[static_cast<std::size_t>(panel_id)].push_back(row);
                    ++produced;
                }

                if (cur_n + 1 >= k_lo_deg) {
                    int next_k = cur_k;
                    int next_m = cur_m + worldsize;
                    while (next_k < end_tile && next_m >= adj.rowLengthForK(next_k)) {
                        next_m -= adj.rowLengthForK(next_k);
                        ++next_k;
                    }
                    cur_k = next_k;
                    cur_m = next_m;
                    cur_n = 0;
                    continue;
                }
            }

            int next_n = cur_n + 1;
            int next_m = cur_m;
            int next_k = cur_k;

            if (next_n > next_k) {
                next_m += worldsize;
                while (next_k < end_tile && next_m >= adj.rowLengthForK(next_k)) {
                    next_m -= adj.rowLengthForK(next_k);
                    ++next_k;
                }
                next_n = 0;
            }

            cur_n = next_n;
            cur_m = next_m;
            cur_k = next_k;
        }

        out_tasks.reserve(static_cast<std::size_t>(produced));
        for (auto& bin : panel_bins) {
            if (bin.empty()) continue;
            out_tasks.insert(out_tasks.end(),
                             std::make_move_iterator(bin.begin()),
                             std::make_move_iterator(bin.end()));
        }

        return produced;
    }

} // namespace alg

#endif
