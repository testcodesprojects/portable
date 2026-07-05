/**
 * @file    TileIndexerGraphBuilder.hpp
 * @brief   Build CSR adjacency (upper/lower) from TileIndexer state for
 *          algorithms that benefit from explicit neighborhood lists.
 *
 * @project sTiles (Sparse Tiles Library)
 * @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @version 3.0.0
 * @date 1 1 2026
 * @license Proprietary
 *
 * @note This file is part of the sTiles library, a proprietary software package.
 *       Redistribution or modification without prior permission is prohibited.
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
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

#pragma once

#include "TileIndexer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <unordered_set>
#include <vector>

namespace tilecounter {

/*
Function: build_graph_upper_generic / build_graph_lower_generic
Purpose:  Build flattened CSR adjacency lists using a provided has_edge
          callback. Upper lists neighbors j>=i; lower lists neighbors j<=i.
*/
// Build flattened adjacency lists. Two modes:
//  - Upper: neighbors j >= i (optionally including self).
//  - Lower: neighbors j <= i (optionally including self).

template <class HasEdgeFn>
static inline void build_graph_upper_generic(int N,
                                             HasEdgeFn has_edge,
                                             std::vector<int>& offsets,
                                             std::vector<int>& edges,
                                             bool include_self)
{
    offsets.assign(static_cast<std::size_t>(N) + 1, 0);

    for (int i = 0; i < N; ++i) {
        const int start = include_self ? i : i + 1;
        int deg = 0;
        for (int j = start; j < N; ++j) {
            if (has_edge(i, j)) ++deg;
        }
        offsets[static_cast<std::size_t>(i + 1)] = offsets[static_cast<std::size_t>(i)] + deg;
    }

    edges.assign(static_cast<std::size_t>(offsets.back()), 0);
    for (int i = 0; i < N; ++i) {
        const int start = include_self ? i : i + 1;
        int write = offsets[static_cast<std::size_t>(i)];
        for (int j = start; j < N; ++j) {
            if (has_edge(i, j)) edges[static_cast<std::size_t>(write++)] = j;
        }
    }
}

template <class HasEdgeFn>
static inline void build_graph_lower_generic(int N,
                                             HasEdgeFn has_edge,
                                             std::vector<int>& offsets,
                                             std::vector<int>& edges,
                                             bool include_self)
{
    offsets.assign(static_cast<std::size_t>(N) + 1, 0);

    for (int i = 0; i < N; ++i) {
        const int limit = include_self ? i : i - 1;
        int deg = 0;
        for (int j = 0; j <= limit; ++j) {
            if (has_edge(j, i)) ++deg;
        }
        offsets[static_cast<std::size_t>(i + 1)] = offsets[static_cast<std::size_t>(i)] + deg;
    }

    edges.assign(static_cast<std::size_t>(offsets.back()), 0);
    for (int i = 0; i < N; ++i) {
        const int limit = include_self ? i : i - 1;
        int write = offsets[static_cast<std::size_t>(i)];
        for (int j = 0; j <= limit; ++j) {
            if (has_edge(j, i)) edges[static_cast<std::size_t>(write++)] = j;
        }
    }
}

// Parallel version: build upper CSR using num_cores threads
// Phase 1: count degrees per row in parallel
// Phase 2: prefix sum (sequential, O(N))
// Phase 3: fill edges per row in parallel (each row writes to non-overlapping range)
template <class HasEdgeFn>
static inline void build_graph_upper_generic_parallel(int N,
                                                      HasEdgeFn has_edge,
                                                      std::vector<int>& offsets,
                                                      std::vector<int>& edges,
                                                      bool include_self,
                                                      int num_cores)
{
    offsets.assign(static_cast<std::size_t>(N) + 1, 0);

    // Phase 1: count degrees per row in parallel
    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_cores)
    for (int i = 0; i < N; ++i) {
        const int start = include_self ? i : i + 1;
        int deg = 0;
        for (int j = start; j < N; ++j) {
            if (has_edge(i, j)) ++deg;
        }
        offsets[static_cast<std::size_t>(i) + 1] = deg;
    }

    // Phase 2: prefix sum (sequential)
    for (int i = 0; i < N; ++i) {
        offsets[static_cast<std::size_t>(i) + 1] += offsets[static_cast<std::size_t>(i)];
    }

    // Phase 3: fill edges per row in parallel
    edges.assign(static_cast<std::size_t>(offsets.back()), 0);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_cores)
    for (int i = 0; i < N; ++i) {
        const int start = include_self ? i : i + 1;
        int write = offsets[static_cast<std::size_t>(i)];
        for (int j = start; j < N; ++j) {
            if (has_edge(i, j)) edges[static_cast<std::size_t>(write++)] = j;
        }
    }
}

// Parallel version: build lower CSR using num_cores threads
template <class HasEdgeFn>
static inline void build_graph_lower_generic_parallel(int N,
                                                      HasEdgeFn has_edge,
                                                      std::vector<int>& offsets,
                                                      std::vector<int>& edges,
                                                      bool include_self,
                                                      int num_cores)
{
    offsets.assign(static_cast<std::size_t>(N) + 1, 0);

    // Phase 1: count degrees per row in parallel
    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_cores)
    for (int i = 0; i < N; ++i) {
        const int limit = include_self ? i : i - 1;
        int deg = 0;
        for (int j = 0; j <= limit; ++j) {
            if (has_edge(j, i)) ++deg;
        }
        offsets[static_cast<std::size_t>(i) + 1] = deg;
    }

    // Phase 2: prefix sum (sequential)
    for (int i = 0; i < N; ++i) {
        offsets[static_cast<std::size_t>(i) + 1] += offsets[static_cast<std::size_t>(i)];
    }

    // Phase 3: fill edges per row in parallel
    edges.assign(static_cast<std::size_t>(offsets.back()), 0);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_cores)
    for (int i = 0; i < N; ++i) {
        const int limit = include_self ? i : i - 1;
        int write = offsets[static_cast<std::size_t>(i)];
        for (int j = 0; j <= limit; ++j) {
            if (has_edge(j, i)) edges[static_cast<std::size_t>(write++)] = j;
        }
    }
}

// Helper: Invert upper_tile_index to recover (i, j) from packed index u
// Formula: u = i*(2*N - i + 1)/2 + (j - i), solve for i then j
static inline void unpack_upper_index(std::size_t u, int N, int& i_out, int& j_out) {
    // Solve i from: u >= i*(2*N - i + 1)/2
    // Rearranged: i^2 - (2*N+1)*i + 2*u <= 0
    // i = floor((2*N+1 - sqrt((2*N+1)^2 - 8*u)) / 2)
    const double dN = static_cast<double>(N);
    const double du = static_cast<double>(u);
    const double disc = (2.0*dN + 1.0)*(2.0*dN + 1.0) - 8.0*du;
    int i = static_cast<int>((2.0*dN + 1.0 - std::sqrt(disc)) / 2.0);
    // Clamp and verify
    if (i < 0) i = 0;
    if (i >= N) i = N - 1;
    // Row offset for row i
    const std::size_t row_off = static_cast<std::size_t>(i) * (2ULL * static_cast<std::size_t>(N) - static_cast<std::size_t>(i) + 1ULL) / 2ULL;
    // If u < row_off, we overshot - back up one row
    if (u < row_off && i > 0) {
        --i;
    }
    const std::size_t final_off = static_cast<std::size_t>(i) * (2ULL * static_cast<std::size_t>(N) - static_cast<std::size_t>(i) + 1ULL) / 2ULL;
    int j = i + static_cast<int>(u - final_off);
    i_out = i;
    j_out = j;
}

// Fast O(E) graph builder from sorted ids vector
static inline void build_graphs_from_ids(State& state, int N, bool include_self) {
    const auto& ids = state.ids;
    const std::size_t E = ids.size();

    // Count degrees for upper graph (neighbors j >= i)
    state.graph_off_up.assign(static_cast<std::size_t>(N) + 1, 0);
    for (std::size_t k = 0; k < E; ++k) {
        int i, j;
        unpack_upper_index(ids[k], N, i, j);
        if (i == j) {
            if (include_self) state.graph_off_up[static_cast<std::size_t>(i) + 1]++;
        } else {
            // (i,j) with i < j: upper[i] gets j, lower[j] gets i
            state.graph_off_up[static_cast<std::size_t>(i) + 1]++;
        }
    }
    // Prefix sum for upper
    for (int i = 0; i < N; ++i) {
        state.graph_off_up[static_cast<std::size_t>(i) + 1] += state.graph_off_up[static_cast<std::size_t>(i)];
    }

    // Count degrees for lower graph (neighbors j <= i)
    state.graph_off_lo.assign(static_cast<std::size_t>(N) + 1, 0);
    for (std::size_t k = 0; k < E; ++k) {
        int i, j;
        unpack_upper_index(ids[k], N, i, j);
        if (i == j) {
            if (include_self) state.graph_off_lo[static_cast<std::size_t>(i) + 1]++;
        } else {
            // (i,j) with i < j: lower[j] gets i
            state.graph_off_lo[static_cast<std::size_t>(j) + 1]++;
        }
    }
    // Prefix sum for lower
    for (int i = 0; i < N; ++i) {
        state.graph_off_lo[static_cast<std::size_t>(i) + 1] += state.graph_off_lo[static_cast<std::size_t>(i)];
    }

    // Fill edges
    state.graph_edges_up.resize(static_cast<std::size_t>(state.graph_off_up[static_cast<std::size_t>(N)]));
    state.graph_edges_lo.resize(static_cast<std::size_t>(state.graph_off_lo[static_cast<std::size_t>(N)]));

    std::vector<int> write_up(static_cast<std::size_t>(N), 0);
    std::vector<int> write_lo(static_cast<std::size_t>(N), 0);
    for (int i = 0; i < N; ++i) {
        write_up[static_cast<std::size_t>(i)] = state.graph_off_up[static_cast<std::size_t>(i)];
        write_lo[static_cast<std::size_t>(i)] = state.graph_off_lo[static_cast<std::size_t>(i)];
    }

    for (std::size_t k = 0; k < E; ++k) {
        int i, j;
        unpack_upper_index(ids[k], N, i, j);
        if (i == j) {
            if (include_self) {
                state.graph_edges_up[static_cast<std::size_t>(write_up[static_cast<std::size_t>(i)]++)] = j;
                state.graph_edges_lo[static_cast<std::size_t>(write_lo[static_cast<std::size_t>(i)]++)] = i;
            }
        } else {
            // Upper: row i gets neighbor j
            state.graph_edges_up[static_cast<std::size_t>(write_up[static_cast<std::size_t>(i)]++)] = j;
            // Lower: row j gets neighbor i
            state.graph_edges_lo[static_cast<std::size_t>(write_lo[static_cast<std::size_t>(j)]++)] = i;
        }
    }

    state.graph_N = N;
    state.graph_include_self = include_self;
    state.graphs_built = true;
}

// Parallel O(E) graph builder from sorted ids vector
// Upper and lower graphs are built concurrently via omp sections,
// and within each half the degree-count and edge-fill loops are parallelised.
static inline void build_graphs_from_ids_parallel(State& state, int N, bool include_self, int num_cores) {
    const auto& ids = state.ids;
    const std::size_t E = ids.size();
    const std::size_t Ns = static_cast<std::size_t>(N);

    // Pre-unpack all (i,j) pairs once (shared read-only)
    std::vector<int> id_i(E);
    std::vector<int> id_j(E);
    #pragma omp parallel for schedule(static) num_threads(num_cores)
    for (std::size_t k = 0; k < E; ++k) {
        unpack_upper_index(ids[k], N, id_i[k], id_j[k]);
    }

    // Allocate degree arrays (one per graph, indexed by row)
    std::vector<int> deg_up(Ns, 0);
    std::vector<int> deg_lo(Ns, 0);

    // Count degrees — upper[i]++ and lower[j]++ (or lower[i]++ on diagonal)
    // Each id maps to a unique (i,j). For off-diagonal: upper row=i, lower row=j.
    // Since multiple ids may map to the same row, we need atomic increments.
    #pragma omp parallel for schedule(static) num_threads(num_cores)
    for (std::size_t k = 0; k < E; ++k) {
        const int i = id_i[k];
        const int j = id_j[k];
        if (i == j) {
            if (include_self) {
                #pragma omp atomic
                deg_up[static_cast<std::size_t>(i)]++;
                #pragma omp atomic
                deg_lo[static_cast<std::size_t>(i)]++;
            }
        } else {
            #pragma omp atomic
            deg_up[static_cast<std::size_t>(i)]++;
            #pragma omp atomic
            deg_lo[static_cast<std::size_t>(j)]++;
        }
    }

    // Build offsets via prefix sum (sequential, O(N))
    state.graph_off_up.assign(Ns + 1, 0);
    state.graph_off_lo.assign(Ns + 1, 0);
    for (int r = 0; r < N; ++r) {
        state.graph_off_up[static_cast<std::size_t>(r) + 1] = state.graph_off_up[static_cast<std::size_t>(r)] + deg_up[static_cast<std::size_t>(r)];
        state.graph_off_lo[static_cast<std::size_t>(r) + 1] = state.graph_off_lo[static_cast<std::size_t>(r)] + deg_lo[static_cast<std::size_t>(r)];
    }

    // Allocate edge arrays
    state.graph_edges_up.resize(static_cast<std::size_t>(state.graph_off_up[Ns]));
    state.graph_edges_lo.resize(static_cast<std::size_t>(state.graph_off_lo[Ns]));

    // Write pointers (one per row, initialised to row start)
    std::vector<int> write_up(Ns);
    std::vector<int> write_lo(Ns);
    #pragma omp parallel for schedule(static) num_threads(num_cores)
    for (int r = 0; r < N; ++r) {
        write_up[static_cast<std::size_t>(r)] = state.graph_off_up[static_cast<std::size_t>(r)];
        write_lo[static_cast<std::size_t>(r)] = state.graph_off_lo[static_cast<std::size_t>(r)];
    }

    // Fill edges — same atomic pattern as degree counting
    #pragma omp parallel for schedule(static) num_threads(num_cores)
    for (std::size_t k = 0; k < E; ++k) {
        const int i = id_i[k];
        const int j = id_j[k];
        if (i == j) {
            if (include_self) {
                int pos_up, pos_lo;
                #pragma omp atomic capture
                pos_up = write_up[static_cast<std::size_t>(i)]++;
                #pragma omp atomic capture
                pos_lo = write_lo[static_cast<std::size_t>(i)]++;
                state.graph_edges_up[static_cast<std::size_t>(pos_up)] = j;
                state.graph_edges_lo[static_cast<std::size_t>(pos_lo)] = i;
            }
        } else {
            int pos_up, pos_lo;
            #pragma omp atomic capture
            pos_up = write_up[static_cast<std::size_t>(i)]++;
            #pragma omp atomic capture
            pos_lo = write_lo[static_cast<std::size_t>(j)]++;
            state.graph_edges_up[static_cast<std::size_t>(pos_up)] = j;
            state.graph_edges_lo[static_cast<std::size_t>(pos_lo)] = i;
        }
    }

    // Sort each row's edges to match the deterministic order of the serial version.
    // The parallel fill leaves edges in non-deterministic order within each row,
    // which breaks the Cholesky task generator's dependency assumptions.
    #pragma omp parallel for schedule(static) num_threads(num_cores)
    for (int r = 0; r < N; ++r) {
        const int beg_up = state.graph_off_up[static_cast<std::size_t>(r)];
        const int end_up = state.graph_off_up[static_cast<std::size_t>(r) + 1];
        if (end_up - beg_up > 1) {
            std::sort(state.graph_edges_up.begin() + beg_up,
                      state.graph_edges_up.begin() + end_up);
        }
        const int beg_lo = state.graph_off_lo[static_cast<std::size_t>(r)];
        const int end_lo = state.graph_off_lo[static_cast<std::size_t>(r) + 1];
        if (end_lo - beg_lo > 1) {
            std::sort(state.graph_edges_lo.begin() + beg_lo,
                      state.graph_edges_lo.begin() + end_lo);
        }
    }

    state.graph_N = N;
    state.graph_include_self = include_self;
    state.graphs_built = true;
}

// Helper: extract ids from active_bool storage (O(T) where T = N*(N+1)/2)
static inline void extract_ids_from_active_bool(State& state, int N) {
    const std::size_t tri_size = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
    if (state.active_bool.size() != tri_size) return;

    state.ids.clear();
    // Pre-count to reserve (avoids reallocations)
    std::size_t count = 0;
    for (std::size_t u = 0; u < tri_size; ++u) {
        if (state.active_bool[u]) ++count;
    }
    state.ids.reserve(count);
    for (std::size_t u = 0; u < tri_size; ++u) {
        if (state.active_bool[u]) state.ids.push_back(u);
    }
}

// Helper: extract ids from bits storage (bitset mask, O(T/64) + O(E) where E = active edges)
static inline void extract_ids_from_bits(State& state, int N) {
    const std::size_t tri_size = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
    const std::size_t words = (tri_size + 63ULL) / 64ULL;
    if (state.bits.size() < words) return;

    state.ids.clear();
    // Pre-count using popcount
    std::size_t count = 0;
    for (std::size_t w = 0; w < words; ++w) {
#if defined(__GNUC__) || defined(__clang__)
        count += static_cast<std::size_t>(__builtin_popcountll(state.bits[w]));
#else
        std::uint64_t x = state.bits[w];
        x = x - ((x >> 1) & 0x5555555555555555ULL);
        x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
        count += static_cast<std::size_t>((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
#endif
    }
    state.ids.reserve(count);

    for (std::size_t w = 0; w < words; ++w) {
        std::uint64_t bits = state.bits[w];
        while (bits != 0ULL) {
#if defined(__GNUC__) || defined(__clang__)
            const int bit = __builtin_ctzll(bits);
#else
            int bit = 0;
            std::uint64_t tmp = bits;
            if (!(tmp & 0xFFFFFFFFULL)) { bit += 32; tmp >>= 32; }
            if (!(tmp & 0xFFFFULL)) { bit += 16; tmp >>= 16; }
            if (!(tmp & 0xFFULL)) { bit += 8; tmp >>= 8; }
            if (!(tmp & 0xFULL)) { bit += 4; tmp >>= 4; }
            if (!(tmp & 0x3ULL)) { bit += 2; tmp >>= 2; }
            if (!(tmp & 0x1ULL)) { bit += 1; }
#endif
            const std::size_t u = w * 64ULL + static_cast<std::size_t>(bit);
            if (u < tri_size) state.ids.push_back(u);
            bits &= bits - 1ULL; // Clear lowest set bit
        }
    }
}

// Convenience: build both upper and lower CSR graphs into State
/*
Function: build_graphs_up_lo
Purpose:  Convenience to build both upper and lower graphs into State using
          the bound isActive predicate. Uses fast O(E) path when ids available.
*/
static inline void build_graphs_up_lo(State& state,
                                          int N,
                                          bool include_self = true)
{
    // Fast path 1: if ids vector is populated and sorted, build directly in O(E)
    if (!state.ids.empty() && std::is_sorted(state.ids.begin(), state.ids.end())) {
        build_graphs_from_ids(state, N, include_self);
        return;
    }

    // Fast path 2: extract ids from bits (bitset mask) - O(T/64) + O(E)
    const std::size_t tri_size = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
    const std::size_t words = (tri_size + 63ULL) / 64ULL;
    if (!state.bits.empty() && state.bits.size() >= words) {
        extract_ids_from_bits(state, N);
        if (!state.ids.empty()) {
            build_graphs_from_ids(state, N, include_self);
            return;
        }
    }

    // Fast path 3: extract ids from active_bool - O(T) but single pass
    if (!state.active_bool.empty() && state.active_bool.size() == tri_size) {
        extract_ids_from_active_bool(state, N);
        if (!state.ids.empty()) {
            build_graphs_from_ids(state, N, include_self);
            return;
        }
    }

    // Slow path: O(N²) probing of all pairs
    const auto has_edge = [&](int a, int b) -> bool {
        return state.isActive(a, b, N);
    };
    build_graph_upper_generic(N, has_edge, state.graph_off_up, state.graph_edges_up, include_self);
    build_graph_lower_generic(N, has_edge, state.graph_off_lo, state.graph_edges_lo, include_self);
    state.graph_N = N;
    state.graph_include_self = include_self;
    state.graphs_built = true;
}

static inline void build_graphs_up_lo_parallel(State& state,
                                               int N,
                                               bool include_self,
                                               int num_cores)
{
    if (num_cores <= 1) {
        build_graphs_up_lo(state, N, include_self);
        return;
    }

    // Fast path 1: if ids vector is populated and sorted, build directly in O(E)
    if (!state.ids.empty() && std::is_sorted(state.ids.begin(), state.ids.end())) {
        build_graphs_from_ids_parallel(state, N, include_self, num_cores);
        return;
    }

    // Fast path 2: extract ids from bits (bitset mask) - O(T/64) + O(E)
    const std::size_t tri_size = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
    const std::size_t words = (tri_size + 63ULL) / 64ULL;
    if (!state.bits.empty() && state.bits.size() >= words) {
        extract_ids_from_bits(state, N);
        if (!state.ids.empty()) {
            build_graphs_from_ids_parallel(state, N, include_self, num_cores);
            return;
        }
    }

    // Fast path 3: extract ids from active_bool - O(T) but single pass
    if (!state.active_bool.empty() && state.active_bool.size() == tri_size) {
        extract_ids_from_active_bool(state, N);
        if (!state.ids.empty()) {
            build_graphs_from_ids_parallel(state, N, include_self, num_cores);
            return;
        }
    }

    // Slow path: O(N²) probing of all pairs — upper and lower in parallel
    const auto has_edge = [&](int a, int b) -> bool {
        return state.isActive(a, b, N);
    };
    build_graph_upper_generic_parallel(N, has_edge, state.graph_off_up, state.graph_edges_up, include_self, num_cores);
    build_graph_lower_generic_parallel(N, has_edge, state.graph_off_lo, state.graph_edges_lo, include_self, num_cores);
    state.graph_N = N;
    state.graph_include_self = include_self;
    state.graphs_built = true;
}


// 1) CharMask
static inline void build_graph_char(const State& idx,
                                    int N,
                                    std::vector<int>& offsets,
                                    std::vector<int>& edges,
                                    bool include_self = false)
{
    const auto has_edge = [&](int a, int b) -> bool {
        if (a > b) std::swap(a, b);
        const std::size_t u = upper_tile_index(a, b, N);
        return (u < idx.active_char.size()) && (idx.active_char[u] != 0);
    };
    build_graph_upper_generic(N, has_edge, offsets, edges, include_self);
}

static inline void build_graph_char_lower(const State& idx,
                                          int N,
                                          std::vector<int>& offsets,
                                          std::vector<int>& edges,
                                          bool include_self = false)
{
    const auto has_edge = [&](int a, int b) -> bool {
        if (a > b) std::swap(a, b);
        const std::size_t u = upper_tile_index(a, b, N);
        return (u < idx.active_char.size()) && (idx.active_char[u] != 0);
    };
    build_graph_lower_generic(N, has_edge, offsets, edges, include_self);
}

// 2) BoolMask
static inline void build_graph_bool(const State& idx,
                                    int N,
                                    std::vector<int>& offsets,
                                    std::vector<int>& edges,
                                    bool include_self = false)
{
    const auto has_edge = [&](int a, int b) -> bool {
        if (a > b) std::swap(a, b);
        const std::size_t u = upper_tile_index(a, b, N);
        return (u < idx.active_bool.size()) && idx.active_bool[u];
    };
    build_graph_upper_generic(N, has_edge, offsets, edges, include_self);
}

static inline void build_graph_bool_lower(const State& idx,
                                          int N,
                                          std::vector<int>& offsets,
                                          std::vector<int>& edges,
                                          bool include_self = false)
{
    const auto has_edge = [&](int a, int b) -> bool {
        if (a > b) std::swap(a, b);
        const std::size_t u = upper_tile_index(a, b, N);
        return (u < idx.active_bool.size()) && idx.active_bool[u];
    };
    build_graph_lower_generic(N, has_edge, offsets, edges, include_self);
}

// 3) BitsetMask
static inline void build_graph_bitset(const State& idx,
                                      int N,
                                      std::vector<int>& offsets,
                                      std::vector<int>& edges,
                                      bool include_self = false)
{
    const auto has_edge = [&](int a, int b) -> bool {
        if (a > b) std::swap(a, b);
        const std::size_t u = upper_tile_index(a, b, N);
        const std::size_t w = u >> 6;
        return (w < idx.bits.size()) && ((idx.bits[w] & (1ULL << (u & 63ULL))) != 0ULL);
    };
    build_graph_upper_generic(N, has_edge, offsets, edges, include_self);
}

static inline void build_graph_bitset_lower(const State& idx,
                                            int N,
                                            std::vector<int>& offsets,
                                            std::vector<int>& edges,
                                            bool include_self = false)
{
    const auto has_edge = [&](int a, int b) -> bool {
        if (a > b) std::swap(a, b);
        const std::size_t u = upper_tile_index(a, b, N);
        const std::size_t w = u >> 6;
        return (w < idx.bits.size()) && ((idx.bits[w] & (1ULL << (u & 63ULL))) != 0ULL);
    };
    build_graph_lower_generic(N, has_edge, offsets, edges, include_self);
}

// 4) TiledBoolMask (chunks)
static inline void build_graph_tiled_bool(const State& idx,
                                          int N,
                                          std::vector<int>& offsets,
                                          std::vector<int>& edges,
                                          bool include_self = false)
{
    const int B = idx.tiled_block_dim;
    const int Bdim = idx.tiled_blocks_per_dim;
    const auto has_edge = [&](int a, int b) -> bool {
        if (B <= 0 || Bdim <= 0) return false;
        if (a > b) std::swap(a, b);
        const int bi = a / B, bj = b / B;
        const int li = a - bi * B, lj = b - bj * B;
        const std::size_t key = static_cast<std::size_t>(bi) * static_cast<std::size_t>(Bdim) + static_cast<std::size_t>(bj);
        auto it = idx.tiled_chunks.find(key);
        if (it == idx.tiled_chunks.end()) return false;
        const std::size_t k = static_cast<std::size_t>(li) * static_cast<std::size_t>(B) + static_cast<std::size_t>(lj);
        const std::size_t w = k >> 6; const std::uint64_t m = 1ULL << (k & 63ULL);
        return (w < it->second.size()) && ((it->second[w] & m) != 0ULL);
    };
    build_graph_upper_generic(N, has_edge, offsets, edges, include_self);
}

// 5) TiledBitsetMask (same backing as tiled bool in this design)
static inline void build_graph_tiled_bitset(const State& idx,
                                            int N,
                                            std::vector<int>& offsets,
                                            std::vector<int>& edges,
                                            bool include_self = false)
{
    build_graph_tiled_bool(idx, N, offsets, edges, include_self);
}

static inline void build_graph_tiled_bool_lower(const State& idx,
                                                int N,
                                                std::vector<int>& offsets,
                                                std::vector<int>& edges,
                                                bool include_self = false)
{
    const int B = idx.tiled_block_dim;
    const int Bdim = idx.tiled_blocks_per_dim;
    const auto has_edge = [&](int a, int b) -> bool {
        if (B <= 0 || Bdim <= 0) return false;
        if (a > b) std::swap(a, b);
        const int bi = a / B, bj = b / B;
        const int li = a - bi * B, lj = b - bj * B;
        const std::size_t key = static_cast<std::size_t>(bi) * static_cast<std::size_t>(Bdim) + static_cast<std::size_t>(bj);
        auto it = idx.tiled_chunks.find(key);
        if (it == idx.tiled_chunks.end()) return false;
        const std::size_t k = static_cast<std::size_t>(li) * static_cast<std::size_t>(B) + static_cast<std::size_t>(lj);
        const std::size_t w = k >> 6; const std::uint64_t m = 1ULL << (k & 63ULL);
        return (w < it->second.size()) && ((it->second[w] & m) != 0ULL);
    };
    build_graph_lower_generic(N, has_edge, offsets, edges, include_self);
}

static inline void build_graph_tiled_bitset_lower(const State& idx,
                                                  int N,
                                                  std::vector<int>& offsets,
                                                  std::vector<int>& edges,
                                                  bool include_self = false)
{
    build_graph_tiled_bool_lower(idx, N, offsets, edges, include_self);
}

// 6) HashSet
static inline void build_graph_hashset(const State& idx,
                                       int N,
                                       std::vector<int>& offsets,
                                       std::vector<int>& edges,
                                       bool include_self = false)
{
    const auto has_edge = [&](int a, int b) -> bool {
        if (a > b) std::swap(a, b);
        return idx.S.find(upper_tile_index(a, b, N)) != idx.S.end();
    };
    build_graph_upper_generic(N, has_edge, offsets, edges, include_self);
}

static inline void build_graph_hashset_lower(const State& idx,
                                             int N,
                                             std::vector<int>& offsets,
                                             std::vector<int>& edges,
                                             bool include_self = false)
{
    const auto has_edge = [&](int a, int b) -> bool {
        if (a > b) std::swap(a, b);
        return idx.S.find(upper_tile_index(a, b, N)) != idx.S.end();
    };
    build_graph_lower_generic(N, has_edge, offsets, edges, include_self);
}

// 7) SortUnique (ids vector; use binary_search if sorted)
static inline void build_graph_sortunique(const State& idx,
                                          int N,
                                          std::vector<int>& offsets,
                                          std::vector<int>& edges,
                                          bool include_self = false)
{
    // Ensure we have fast membership: prefer binary_search on a sorted view
    TileIndexer::State::IdVector sorted_ids(idx.makeAllocator<std::size_t>(TileIndexer::State::Group::GraphOffsets));
    const TileIndexer::State::IdVector* src = &idx.ids;
    if (!std::is_sorted(idx.ids.begin(), idx.ids.end())) {
        sorted_ids.assign(idx.ids.begin(), idx.ids.end());
        std::sort(sorted_ids.begin(), sorted_ids.end());
        sorted_ids.erase(std::unique(sorted_ids.begin(), sorted_ids.end()), sorted_ids.end());
        src = &sorted_ids;
    }
    const auto has_edge = [&](int a, int b) -> bool {
        if (a > b) std::swap(a, b);
        const std::size_t u = upper_tile_index(a, b, N);
        return std::binary_search(src->begin(), src->end(), u);
    };
    build_graph_upper_generic(N, has_edge, offsets, edges, include_self);
}

static inline void build_graph_sortunique_lower(const State& idx,
                                                int N,
                                                std::vector<int>& offsets,
                                                std::vector<int>& edges,
                                                bool include_self = false)
{
    TileIndexer::State::IdVector sorted_ids(idx.makeAllocator<std::size_t>(TileIndexer::State::Group::GraphOffsets));
    const TileIndexer::State::IdVector* src = &idx.ids;
    if (!std::is_sorted(idx.ids.begin(), idx.ids.end())) {
        sorted_ids.assign(idx.ids.begin(), idx.ids.end());
        std::sort(sorted_ids.begin(), sorted_ids.end());
        sorted_ids.erase(std::unique(sorted_ids.begin(), sorted_ids.end()), sorted_ids.end());
        src = &sorted_ids;
    }
    const auto has_edge = [&](int a, int b) -> bool {
        if (a > b) std::swap(a, b);
        const std::size_t u = upper_tile_index(a, b, N);
        return std::binary_search(src->begin(), src->end(), u);
    };
    build_graph_lower_generic(N, has_edge, offsets, edges, include_self);
}

// 8) PagedMask (paged over packed IDs)
static inline void build_graph_paged(const State& idx,
                                     int N,
                                     std::vector<int>& offsets,
                                     std::vector<int>& edges,
                                     bool include_self = false)
{
    const std::size_t PB = idx.paged.page_bits;
    const auto has_edge = [&](int a, int b) -> bool {
        if (PB == 0) return false;
        if (a > b) std::swap(a, b);
        const std::size_t u = upper_tile_index(a, b, N);
        const std::size_t page = u / PB; const std::size_t bit = u % PB;
        auto it = idx.paged.pages.find(page);
        if (it == idx.paged.pages.end()) return false;
        const std::size_t w = bit >> 6; const std::uint64_t m = 1ULL << (bit & 63ULL);
        return (w < it->second.size()) && ((it->second[w] & m) != 0ULL);
    };
    build_graph_upper_generic(N, has_edge, offsets, edges, include_self);
}

static inline void build_graph_paged_lower(const State& idx,
                                           int N,
                                           std::vector<int>& offsets,
                                           std::vector<int>& edges,
                                           bool include_self = false)
{
    const std::size_t PB = idx.paged.page_bits;
    const auto has_edge = [&](int a, int b) -> bool {
        if (PB == 0) return false;
        if (a > b) std::swap(a, b);
        const std::size_t u = upper_tile_index(a, b, N);
        const std::size_t page = u / PB; const std::size_t bit = u % PB;
        auto it = idx.paged.pages.find(page);
        if (it == idx.paged.pages.end()) return false;
        const std::size_t w = bit >> 6; const std::uint64_t m = 1ULL << (bit & 63ULL);
        return (w < it->second.size()) && ((it->second[w] & m) != 0ULL);
    };
    build_graph_lower_generic(N, has_edge, offsets, edges, include_self);
}

// Unified dispatcher: picks the correct builder for the given Method
inline void build_graph(const State& idx,
                 Method m,
                 int N,
                 std::vector<int>& offsets,
                 std::vector<int>& edges,
                 bool include_self = false)
{
    Method resolved = resolve_auto_method_for_fill(idx, m);
    switch (resolved) {
        case Method::CharMask:
            build_graph_char(idx, N, offsets, edges, include_self);
            break;
        case Method::BoolMask:
        case Method::LazyLookUp:
            build_graph_bool(idx, N, offsets, edges, include_self);
            break;
        case Method::BitsetMask:
            build_graph_bitset(idx, N, offsets, edges, include_self);
            break;
        case Method::TiledBoolMask:
            build_graph_tiled_bool(idx, N, offsets, edges, include_self);
            break;
        case Method::TiledBitsetMask:
            build_graph_tiled_bitset(idx, N, offsets, edges, include_self);
            break;
        case Method::PagedMask:
            build_graph_paged(idx, N, offsets, edges, include_self);
            break;
        case Method::HashSet:
            build_graph_hashset(idx, N, offsets, edges, include_self);
            break;
        case Method::SortUnique:
            build_graph_sortunique(idx, N, offsets, edges, include_self);
            break;
        case Method::Auto:
            build_graph_bool(idx, N, offsets, edges, include_self);
            break;
    }
}

inline void build_graph_lower(const State& idx,
                              Method m,
                              int N,
                              std::vector<int>& offsets,
                              std::vector<int>& edges,
                              bool include_self = false)
{
    Method resolved = resolve_auto_method_for_fill(idx, m);
    switch (resolved) {
        case Method::CharMask:
            build_graph_char_lower(idx, N, offsets, edges, include_self);
            break;
        case Method::BoolMask:
        case Method::LazyLookUp:
            build_graph_bool_lower(idx, N, offsets, edges, include_self);
            break;
        case Method::BitsetMask:
            build_graph_bitset_lower(idx, N, offsets, edges, include_self);
            break;
        case Method::TiledBoolMask:
            build_graph_tiled_bool_lower(idx, N, offsets, edges, include_self);
            break;
        case Method::TiledBitsetMask:
            build_graph_tiled_bitset_lower(idx, N, offsets, edges, include_self);
            break;
        case Method::PagedMask:
            build_graph_paged_lower(idx, N, offsets, edges, include_self);
            break;
        case Method::HashSet:
            build_graph_hashset_lower(idx, N, offsets, edges, include_self);
            break;
        case Method::SortUnique:
            build_graph_sortunique_lower(idx, N, offsets, edges, include_self);
            break;
        case Method::Auto:
            build_graph_bool_lower(idx, N, offsets, edges, include_self);
            break;
    }
}

// Note: CSR builders also exist in TileIndexerFill.hpp for fill paths.

} // namespace tilecounter
