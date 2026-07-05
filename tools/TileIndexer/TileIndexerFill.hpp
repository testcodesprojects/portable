/**
 * @file    TileIndexerFill.hpp
 * @brief   Fill/closure algorithms for TileIndexer. Dense/sparse/tiled
 *          implementations (and parallel variants) that add implied tiles; CSR helpers.
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
#include <cstdint>
#include <unordered_set>
#include <cstddef>
#include <vector>
#include <thread>
#include <atomic>
#include <utility>

#include "../sort/stiles_sort_dispatch.hpp"

namespace tilecounter {

/*
Struct: CSR
Purpose: Minimal CSR adjacency for upper/lower neighborhoods used by
         closure routines to accelerate neighbor checks.
*/

// Lightweight CSR for upper-tri adjacency
struct CSR {
    std::vector<int> out_off, out_e; // row lo -> neighbors hi>=lo
    std::vector<int> in_off,  in_e;  // column hi -> neighbors lo<=hi
};

/*
Function: make_upper_row_offsets_diag
Purpose:  Compute packed indices of diagonal entries (i,i) for all i, enabling
          fast slicing of row-major upper-triangle segments.
*/
static inline std::vector<std::size_t> make_upper_row_offsets_diag(int N) {
    // off[i] = packed index of (i,i)
    std::vector<std::size_t> off(static_cast<std::size_t>(N), 0);
    for (int i = 1; i < N; ++i) {
        off[static_cast<std::size_t>(i)] = off[static_cast<std::size_t>(i - 1)]
                                         + static_cast<std::size_t>(N - (i - 1));
    }
    return off;
}

// Forward declarations for tiled helpers referenced by dense fallbacks.
static inline int fill_closure_tiled_bool(State& idx, int N, int counted);
static inline int fill_closure_tiled_bitset(State& idx, int N, int counted);

// Build CSR from dense bitset with upper-tri packing
/*
Function: build_csr_from_bitset
Purpose:  Build CSR adjacency from a dense bitset mask laid out over the
          packed upper triangle.
*/
template <typename Alloc>
inline CSR build_csr_from_bitset(const State& owner,
                                 const std::vector<std::uint64_t, Alloc>& bits,
                                 int N) {
    CSR G; G.out_off.assign(static_cast<std::size_t>(N) + 1, 0); G.in_off.assign(static_cast<std::size_t>(N) + 1, 0);
    if (N <= 0) return G;
    const auto off = make_upper_row_offsets_diag(N);
    const std::uint64_t* B = bits.data();

    // 1) Count out-degrees per row lo (lo..N-1 range contiguous)
    auto count_row = [&](int lo) {
        const std::size_t base = off[static_cast<std::size_t>(lo)];
        const std::size_t end  = base + static_cast<std::size_t>(N - 1 - lo);
        const std::size_t w0   = base >> 6, b0 = base & 63u;
        const std::size_t w1   = end  >> 6, b1 = end & 63u;

        if (w0 == w1) {
            const std::uint64_t mask = (~0ULL << b0) & ((b1 == 63u) ? ~0ULL : ((1ULL << (b1 + 1)) - 1ULL));
            G.out_off[static_cast<std::size_t>(lo + 1)] += static_cast<int>(__builtin_popcountll(B[w0] & mask));
            return;
        }
        // first word
        G.out_off[static_cast<std::size_t>(lo + 1)] += static_cast<int>(__builtin_popcountll(B[w0] & (~0ULL << b0)));
        // middle words
        for (std::size_t w = w0 + 1; w < w1; ++w)
            G.out_off[static_cast<std::size_t>(lo + 1)] += static_cast<int>(__builtin_popcountll(B[w]));
        // last word
        G.out_off[static_cast<std::size_t>(lo + 1)] += static_cast<int>(__builtin_popcountll(B[w1] & ((b1 == 63u) ? ~0ULL : ((1ULL << (b1 + 1)) - 1ULL))));
    };
    for (int lo = 0; lo < N; ++lo) count_row(lo);

    // 2) Prefix-sum and emit out edges, while accumulating indegrees
    for (int i = 0; i < N; ++i) G.out_off[static_cast<std::size_t>(i + 1)] += G.out_off[static_cast<std::size_t>(i)];
    G.out_e.resize(G.out_off.back());
    auto int_alloc = owner.makeAllocator<int>(TileIndexer::State::Group::SFL);
    std::vector<int, TileIndexer::State::Allocator<int>> indeg(static_cast<std::size_t>(N), 0, int_alloc);

    auto emit_row = [&](int lo) {
        const std::size_t base = off[static_cast<std::size_t>(lo)];
        const std::size_t end  = base + static_cast<std::size_t>(N - 1 - lo);
        const std::size_t w0   = base >> 6, b0 = base & 63u;
        const std::size_t w1   = end  >> 6, b1 = end & 63u;
        int write = G.out_off[static_cast<std::size_t>(lo)];

        auto emit_word = [&](std::uint64_t word, std::size_t base_u) {
            while (word) {
                const std::uint64_t t = word & -word;
                const unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
                const std::size_t u = base_u + bit;
                const int hi = lo + static_cast<int>(u - base);
                G.out_e[static_cast<std::size_t>(write++)] = hi;
                ++indeg[static_cast<std::size_t>(hi)];
                word ^= t;
            }
        };

        if (w0 == w1) {
            const std::uint64_t mask = (~0ULL << b0) & ((b1 == 63u) ? ~0ULL : ((1ULL << (b1 + 1)) - 1ULL));
            emit_word(B[w0] & mask, w0 << 6);
            return;
        }
        emit_word(B[w0] & (~0ULL << b0), w0 << 6);
        for (std::size_t w = w0 + 1; w < w1; ++w) emit_word(B[w], w << 6);
        emit_word(B[w1] & ((b1 == 63u) ? ~0ULL : ((1ULL << (b1 + 1)) - 1ULL)), w1 << 6);
    };
    for (int lo = 0; lo < N; ++lo) emit_row(lo);

    // 3) Build in CSR from indeg + out edges
    for (int v = 0; v < N; ++v) G.in_off[static_cast<std::size_t>(v + 1)] = G.in_off[static_cast<std::size_t>(v)] + indeg[static_cast<std::size_t>(v)];
    G.in_e.resize(G.in_off.back());
    std::vector<int, TileIndexer::State::Allocator<int>> cursor(G.in_off.begin(), G.in_off.end(), int_alloc);
    for (int lo = 0; lo < N; ++lo) {
        for (int p = G.out_off[static_cast<std::size_t>(lo)]; p < G.out_off[static_cast<std::size_t>(lo + 1)]; ++p) {
            int hi = G.out_e[static_cast<std::size_t>(p)];
            G.in_e[static_cast<std::size_t>(cursor[static_cast<std::size_t>(hi)]++)] = lo;
        }
    }
    return G;
}

// Internal helper: generic triple-loop fill using provided test/set callbacks.
// - N: number of tile-grid rows/cols
// - counted: current count of active tiles (upper-tri)
// Returns updated count after setting missing (j,i) when exists k<j with (k,j) and (k,i).
/*
Function: fill_generic_version1
Purpose:  Baseline triple-loop closure using provided membership and setter
          callbacks. Useful for correctness and sparse backends.
*/
template <class TestFn, class SetFn>
static inline int fill_generic_version1(int N, int counted, TestFn has_edge, SetFn add_edge) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < i; ++j) {

            bool sumT = false;
            for (int k = 0; k < j; ++k) {
                if (has_edge(k, j) && has_edge(k, i)) { sumT = true; break; }
            }
            if (sumT && !has_edge(j, i)) {
                if (add_edge(j, i)) ++counted;
            }

        }
    }
    return counted;
}

/*
Function: fill_generic
Purpose:  Optimized closure that reuses per-row seed lists to cut redundant
          checks, reducing complexity on many inputs.
*/
template <class TestFn, class SetFn>
static inline int fill_generic(State& owner, int N, int counted, TestFn has_edge, SetFn add_edge) {
    // For each i, keep two lists:
    //  - init: neighbors k<i already connected to i at the start of this i
    //  - small: seeds that are <= current j (maintained in ascending order)
    std::vector<int, TileIndexer::State::Allocator<int>> init(owner.makeAllocator<int>(TileIndexer::State::Group::SFL));
    std::vector<int, TileIndexer::State::Allocator<int>> small(owner.makeAllocator<int>(TileIndexer::State::Group::SFL));

    for (int i = 0; i < N; ++i) {
        init.clear();
        small.clear();

        // Heuristic reserve: at most i neighbors exist for row i.
        // Cap the hint to avoid over-reserving on dense rows.
        const int reserve_hint = std::min(i, 256);
        if (init.capacity() < static_cast<std::size_t>(reserve_hint)) init.reserve(reserve_hint);
        if (small.capacity() < static_cast<std::size_t>(reserve_hint)) small.reserve(reserve_hint);

        // Build initial seeds: all k < i with (k,i) present (ascending by construction).
        for (int k = 0; k < i; ++k) {
            if (has_edge(k, i)) init.push_back(k);
        }

        // Pointer into 'init' for seeds that have become <= current j.
        std::size_t p = 0;

        // 'small' will mostly grow by moving seeds from 'init' and occasionally by adding 'j'.
        // If 'init' turned out large, make sure 'small' has room as well (bounded by the same cap).
        if (small.capacity() < init.size()) {
            const std::size_t target = std::min<std::size_t>(init.size(), static_cast<std::size_t>(256));
            small.reserve(target);
        }

        for (int j = 0; j < i; ++j) {

            // Move all initial seeds <= j into 'small' (kept ascending).
            while (p < init.size() && init[p] <= j) small.push_back(init[p++]);

            // If (j,i) already exists (i.e., j was in 'init'), skip work.
            // Avoids an extra has_edge(j,i) call by checking we just moved it.
            if (p > 0 && init[p - 1] == j) continue;

            // Number of seeds strictly less than j (exclude trailing == j if present).
            std::size_t small_lt = small.size();
            if (small_lt && small.back() == j) --small_lt;
            if (small_lt == 0) continue; // no possible k < j with (k,i)

            // Check only seeds k < j; scan from larger k down (often hits earlier).
            bool found = false;
            for (std::size_t t = small_lt; t-- > 0; ) {
                const int k = small[t];
                if (has_edge(k, j)) { found = true; break; }
            }

            if (found) {
                // Add (j,i); then 'j' becomes a seed for future j' > j.
                if (add_edge(j, i)) {
                    ++counted;
                    small.push_back(j); // keeps 'small' ascending since j increases
                }
            }
        }
    }
    return counted;
}

// CSR (Compressed Sparse Row) with strictly increasing neighbors.
// out_offsets[u]..out_offsets[u+1]-1   : neighbors v > u
// in_offsets[v]..in_offsets[v+1]-1     : neighbors u < v
// add_edge(j,i) should set (j,i) if missing and return true iff it was newly added.
/*
Function: fill_monotone_closure_csr
Purpose:  Closure using prebuilt CSR: mark In(i) and test intersection with
          In(j) to determine whether to add (j,i).
*/
template <class AddEdgeFn>
inline int fill_monotone_closure_csr(
    int N,
    const std::vector<int>& /*out_offsets*/, const std::vector<int>& /*out_edges*/, // unused in this variant
    const std::vector<int>&  in_offsets, const std::vector<int>&  in_edges,
    const State& owner,
    AddEdgeFn add_edge,
    int counted)
{
    // For each i, mark all k in In(i). Then for each j<i, check if any k in In(j) is marked.
    std::vector<char, TileIndexer::State::Allocator<char>> mark(static_cast<std::size_t>(N), 0,
        owner.makeAllocator<char>(TileIndexer::State::Group::SFL));
    for (int i = 0; i < N; ++i) {
        // mark initial in-neighbors of i (k<i)
        for (int p = in_offsets[static_cast<std::size_t>(i)]; p < in_offsets[static_cast<std::size_t>(i + 1)]; ++p)
            mark[static_cast<std::size_t>(in_edges[static_cast<std::size_t>(p)])] = 1;

        for (int j = 0; j < i; ++j) {
            // if already present, skip adding but update mark after to reflect (j,i)
            bool found = false;
            for (int p = in_offsets[static_cast<std::size_t>(j)]; p < in_offsets[static_cast<std::size_t>(j + 1)]; ++p) {
                int k = in_edges[static_cast<std::size_t>(p)];
                if (k >= j) break; // in CSR, in-neighbors are < j and sorted; early stop not guaranteed but harmless
                if (mark[static_cast<std::size_t>(k)]) { found = true; break; }
            }
            if (found) {
                if (add_edge(j, i)) ++counted;
                mark[static_cast<std::size_t>(j)] = 1; // newly added becomes in-neighbor for later j'
            }
        }

        // clear marks for this i
        for (int p = in_offsets[static_cast<std::size_t>(i)]; p < in_offsets[static_cast<std::size_t>(i + 1)]; ++p)
            mark[static_cast<std::size_t>(in_edges[static_cast<std::size_t>(p)])] = 0;
        // also clear any j we set; cheapest is to reset whole array when i increases moderately small
        // but to keep O(E), we can lazily leave marks and overwrite on next i; here we clear the entries we set via inner loop
        for (int j = 0; j < i; ++j) mark[static_cast<std::size_t>(j)] = 0;
    }
    return counted;
}

/*
Function: fill_closure_from_bitset
Purpose:  Run closure by converting a bitset mask to CSR once, applying the
          CSR-based algorithm, and writing results back into the bitset.
*/
/*
Function: fill_closure_from_bitset
Purpose:  Convert the bitset mask into CSR, apply CSR-based closure, then
          write updates back into the bitset structure.
*/
int fill_closure_from_bitset(State& idx, int N, int counted) {
    // 1) Build CSR from the current bitset mask:
    auto G = build_csr_from_bitset(idx, idx.bits, N);

    // 2) Define how to set edges back into the bitset:
    auto add_edge = [&](int a, int b) -> bool {
        // a<b guaranteed by the algorithm
        const std::size_t u = tilecounter::upper_tile_index(a, b, N);
        std::size_t w = u >> 6; uint64_t m = 1ull << (u & 63ull);
        uint64_t old = idx.bits[w];
        if ((old & m) == 0) { idx.bits[w] = old | m; return true; }
        return false;
    };

    // 3) Run the closure:
    return fill_monotone_closure_csr(
        N, G.out_off, G.out_e, G.in_off, G.in_e, idx, add_edge, counted);
}


// 1) CharMask (dense char vector)
/*
Function: fill_closure_char
Purpose:  Closure directly on a dense char mask representation.
*/
static inline int fill_closure_char(State& idx, int N, int counted) {
    const std::size_t tri = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
    if (idx.active_char.size() < tri) idx.active_char.resize(tri, 0);

    auto get_u = [&](int a, int b) -> std::size_t {
        return upper_tile_index(a, b, N);
    };
    auto has_edge = [&](int a, int b) -> bool {
        return idx.active_char[get_u(a, b)] != 0;
    };
    auto add_edge = [&](int a, int b) -> bool {
        const std::size_t u = get_u(a, b);
        if (!idx.active_char[u]) {
            idx.active_char[u] = 1;
            return true;
        }
        return false;
    };
    return fill_generic(idx, N, counted, has_edge, add_edge);
}

// 2) BoolMask (dense bool vector)
/*
Function: fill_closure_bool
Purpose:  Closure on bool mask by converting to/from a char mask for parity.
*/
static inline int fill_closure_bool(State& idx, int N, int counted) {
    const std::size_t tri = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
    if (idx.active_bool.size() < tri) idx.active_bool.resize(tri, false);
    if (idx.active_char.size() < tri) idx.active_char.resize(tri, 0);
    for (std::size_t u = 0; u < tri; ++u) {
        idx.active_char[u] = idx.active_bool[u] ? 1 : 0;
    }

    const int filled = fill_closure_char(idx, N, counted);

    for (std::size_t u = 0; u < tri; ++u) {
        idx.active_bool[u] = (idx.active_char[u] != 0);
    }
    return filled;
}

/*
Function: fill_closure_lazy_lookup
Purpose:  Closure with lazy index map maintenance for LazyLookUp method.
*/
static inline int fill_closure_lazy_lookup(State& idx, int N, int counted) {
    const std::size_t tri = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
    if (idx.active_bool.size() < tri) idx.active_bool.assign(tri, false);
    if (!idx.active_char.empty()) {
        const std::size_t copy = std::min<std::size_t>(idx.active_char.size(), tri);
        for (std::size_t u = 0; u < copy; ++u) {
            if (idx.active_char[u]) idx.active_bool[u] = true;
        }
        idx.active_char.clear();
    }

    const int filled = fill_closure_bool(idx, N, counted);

#ifdef TILEINDEXER_DEBUG_LAZY_LOOKUP
    std::cerr << "[lazylookup] counted=" << counted << " filled=" << filled << "\n";
#endif

    idx.lazy_index_map.assign(idx.active_bool.size(), -1);
    int next = 0;
    for (std::size_t u = 0; u < idx.active_bool.size(); ++u) {
        if (idx.active_bool[u]) idx.lazy_index_map[u] = next++;
    }
    return filled;
}

// 3) BitsetMask (dense bitset in 64-bit words)
/*
Function: fill_closure_bitset
Purpose:  Closure by round-tripping through a char mask for simplicity and
          correctness parity with the CharMask implementation.
*/
static inline int fill_closure_bitset(State& idx, int N, int counted) {
    const std::size_t tri = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
    const std::size_t words = (tri + 63ULL) >> 6;
    if (idx.bits.size() < words) idx.bits.resize(words, 0ULL);

    // Convert bitset to char mask for correctness parity with CharMask path
    std::vector<char, TileIndexer::State::Allocator<char>> saved_char(idx.makeAllocator<char>(TileIndexer::State::Group::SFL));
    saved_char.swap(idx.active_char);
    idx.active_char.assign(tri, 0);
    for (std::size_t w = 0, u = 0; w < words; ++w) {
        std::uint64_t word = idx.bits[w];
        while (word) {
            std::uint64_t t = word & -word;
            unsigned b = static_cast<unsigned>(__builtin_ctzll(word));
            idx.active_char[u + b] = 1;
            word ^= t;
        }
        u += 64;
    }
    int out = fill_closure_char(idx, N, counted);
    // Repack char mask back into bitset
    std::fill(idx.bits.begin(), idx.bits.end(), 0ULL);
    for (std::size_t u = 0; u < idx.active_char.size(); ++u) {
        if (idx.active_char[u]) idx.bits[u >> 6] |= (1ULL << (u & 63ULL));
    }
    idx.active_char.swap(saved_char);
    return out;
}

// 4) TiledBoolMask (chunked BxB blocks stored as packed bits)
/*
Function: fill_closure_tiled_bool
Purpose:  Closure over sparse BxB tile blocks packed into 64-bit words.
*/
static inline int fill_closure_tiled_bool(State& idx, int N, int counted) {
    const int B = idx.tiled_block_dim;
    const int blocks_per_dim = idx.tiled_blocks_per_dim;
    // Assumption: B and blocks_per_dim have been set consistently when the mask was built.

    auto has_edge = [&](int a, int b) -> bool {
        const int bi = a / B, bj = b / B;
        const int li = a - bi * B, lj = b - bj * B;
        const std::size_t key = static_cast<std::size_t>(bi) * static_cast<std::size_t>(blocks_per_dim) + static_cast<std::size_t>(bj);
        auto it = idx.tiled_chunks.find(key);
        if (it == idx.tiled_chunks.end()) return false;
        const std::size_t k = static_cast<std::size_t>(li) * static_cast<std::size_t>(B) + static_cast<std::size_t>(lj);
        return (it->second[k >> 6] & (1ULL << (k & 63ULL))) != 0ULL;
    };
    auto add_edge = [&](int a, int b) -> bool {
        const int bi = a / B, bj = b / B;
        const int li = a - bi * B, lj = b - bj * B;
        const std::size_t key = static_cast<std::size_t>(bi) * static_cast<std::size_t>(blocks_per_dim) + static_cast<std::size_t>(bj);
        const std::size_t block_bits = static_cast<std::size_t>(B) * static_cast<std::size_t>(B);
        const std::size_t words = (block_bits + 63ULL) >> 6;
        auto& vec = idx.tiled_chunks.try_emplace(key, idx.make_chunk_vector(words, 0ULL)).first->second;
        const std::size_t k = static_cast<std::size_t>(li) * static_cast<std::size_t>(B) + static_cast<std::size_t>(lj);
        const std::size_t w = k >> 6; const std::uint64_t m = 1ULL << (k & 63ULL);
        if ((vec[w] & m) == 0ULL) { vec[w] |= m; return true; }
        return false;
    };
    return fill_generic(idx, N, counted, has_edge, add_edge);
}

// 5) TiledBitsetMask (same storage as tiled bool in this design)
/*
Function: fill_closure_tiled_bitset
Purpose:  Alias of tiled-bool closure (shared storage design).
*/
static inline int fill_closure_tiled_bitset(State& idx, int N, int counted) {
    return fill_closure_tiled_bool(idx, N, counted);
}

// 6) HashSet (unordered_set of packed IDs)
/*
Function: fill_closure_hashset
Purpose:  Closure over a sparse set of packed tile IDs (unordered_set).
*/
static inline int fill_closure_hashset(State& idx, int N, int counted) {
    auto has_edge = [&](int a, int b) -> bool {
        return idx.S.find(upper_tile_index(a, b, N)) != idx.S.end();
    };
    auto add_edge = [&](int a, int b) -> bool {
        auto [it, inserted] = idx.S.insert(upper_tile_index(a, b, N));
        return inserted;
    };
    return fill_generic(idx, N, counted, has_edge, add_edge);
}

// 7) SortUnique (vector of packed IDs; ensure uniqueness on writeback)
/*
Function: fill_closure_sortunique
Purpose:  Closure over a vector of packed IDs that is uniqued on writeback.
*/
static inline int fill_closure_sortunique(State& idx, int N, int counted) {
    std::unordered_set<std::size_t> set(idx.ids.begin(), idx.ids.end());
    auto has_edge = [&](int a, int b) -> bool {
        return set.find(upper_tile_index(a, b, N)) != set.end();
    };
    auto add_edge = [&](int a, int b) -> bool {
        auto [it, inserted] = set.insert(upper_tile_index(a, b, N));
        return inserted;
    };
    const int out = fill_generic(idx, N, counted, has_edge, add_edge);
    idx.ids.assign(set.begin(), set.end());
    sTiles::sort(idx.ids.begin(), idx.ids.end());
    idx.ids.erase(std::unique(idx.ids.begin(), idx.ids.end()), idx.ids.end());
    return out;
}

// 8) PagedMask (paged over packed IDs; test-and-set per bit)
/*
Function: fill_closure_paged
Purpose:  Closure over a paged bitset where each page holds packed 64-bit
          words and page/word prefixes are used for efficient access.
*/
static inline int fill_closure_paged(State& idx, int N, int counted) {
    const std::size_t PB = idx.paged.page_bits;
    std::size_t PW = idx.paged.page_words;
    if (PW == 0) {
        PW = std::max<std::size_t>(1, (PB + 63ULL) >> 6ULL);
        idx.paged.page_words = PW;
    }

    struct PageAccess {
        TileIndexer::State::Paged::PageVector* vec;
        std::size_t word;
        std::uint64_t mask;
    };

    auto locate_bit = [&](std::size_t u, bool create) -> PageAccess {
        const std::size_t page = PB ? (u / PB) : 0;
        const std::size_t bit  = PB ? (u % PB) : u;
        const std::size_t word = bit >> 6;
        const std::uint64_t mask = 1ULL << (bit & 63ULL);

        auto it = idx.paged.pages.find(page);
        if (it == idx.paged.pages.end()) {
            if (!create) return {nullptr, word, mask};
            auto vec = idx.paged.make_page_vector(PW ? PW : (word + 1), 0ULL);
            if (word >= vec.size()) vec.resize(word + 1, 0ULL);
            it = idx.paged.pages.emplace(page, std::move(vec)).first;
        } else if (word >= it->second.size()) {
            if (!create) return {nullptr, word, mask};
            it->second.resize(word + 1, 0ULL);
        }

        return {&(it->second), word, mask};
    };

    auto has_edge = [&](int a, int b) -> bool {
        const std::size_t u = upper_tile_index(a, b, N);
        auto access = locate_bit(u, false);
        if (!access.vec) return false;
        return ((*access.vec)[access.word] & access.mask) != 0ULL;
    };

    auto add_edge = [&](int a, int b) -> bool {
        const std::size_t u = upper_tile_index(a, b, N);
        auto access = locate_bit(u, true);
        std::uint64_t& word = (*access.vec)[access.word];
        if ((word & access.mask) == 0ULL) {
            word |= access.mask;
            return true;
        }
        return false;
    };

    return fill_generic(idx, N, counted, has_edge, add_edge);
}

// ---------------- Parallel variants for selected strategies -----------------

namespace detail {

template <class HasEdge, class Alloc>
static inline void build_seeds(int i, HasEdge has_edge, std::vector<int, Alloc>& buf) {
    buf.clear();
    const int hint = std::min(i, 256);
    if (buf.capacity() < static_cast<std::size_t>(hint)) buf.reserve(hint);
    for (int k = 0; k < i; ++k) {
        if (has_edge(k, i)) buf.push_back(k);
    }
}

template <class MakeHasEdgeFn, class CommitFn>
static inline int fill_fixpoint_parallel(State& owner, int N, int counted, int num_threads,
                                         MakeHasEdgeFn make_has_edge,
                                         CommitFn commit_edge)
{
    if (N <= 1 || num_threads <= 1) return counted;

    const int threads = std::max(1, std::min(num_threads, N));
    int total_added = 0;
    const int max_passes = std::max(1, N);

    using EdgePair = std::pair<int, int>;
    using EdgeVector = std::vector<EdgePair, TileIndexer::State::Allocator<EdgePair>>;
    using IntVector = std::vector<int, TileIndexer::State::Allocator<int>>;
    std::vector<EdgeVector> per_thread_edges;
    std::vector<IntVector> per_thread_seeds;
    per_thread_edges.reserve(static_cast<std::size_t>(threads));
    per_thread_seeds.reserve(static_cast<std::size_t>(threads));
    for (int i = 0; i < threads; ++i) {
        per_thread_edges.emplace_back(owner.makeAllocator<EdgePair>(TileIndexer::State::Group::SFL));
        per_thread_seeds.emplace_back(owner.makeAllocator<int>(TileIndexer::State::Group::SFL));
    }

    for (int pass = 0; pass < max_passes; ++pass) {
        for (auto& v : per_thread_edges) v.clear();
        auto has_edge = make_has_edge();

        std::atomic<int> next_i{0};
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(threads));

        for (int tid = 0; tid < threads; ++tid) {
            workers.emplace_back([&, tid]() {
                auto& edges = per_thread_edges[static_cast<std::size_t>(tid)];
                auto& seeds = per_thread_seeds[static_cast<std::size_t>(tid)];
                while (true) {
                    int i = next_i.fetch_add(1, std::memory_order_relaxed);
                    if (i >= N) break;

                    detail::build_seeds(i, has_edge, seeds);
                    if (seeds.empty()) continue;

                    for (int j = 0; j < i; ++j) {
                        if (has_edge(j, i)) continue;
                        auto it = std::lower_bound(seeds.begin(), seeds.end(), j);
                        bool found = false;
                        for (auto t = it; t != seeds.begin();) {
                            --t;
                            if (has_edge(*t, j)) { found = true; break; }
                        }
                        if (found) edges.emplace_back(j, i);
                    }
                }
            });
        }
        for (auto& t : workers) t.join();

        int added_this_pass = 0;
        for (const auto& edges : per_thread_edges) {
            for (const auto& e : edges) {
                if (commit_edge(e.first, e.second)) {
                    ++added_this_pass;
                }
            }
        }

        if (added_this_pass == 0) break;
        total_added += added_this_pass;
    }

    return counted + total_added;
}

} // namespace detail

static inline int fill_closure_bitset_parallel(State& idx, int N, int counted, int threads) {
    const std::size_t tri = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
    const std::size_t words = (tri + 63ULL) >> 6;
    if (idx.bits.size() < words) idx.bits.resize(words, 0ULL);

    auto make_has = [&]() {
        const std::uint64_t* data = idx.bits.data();
        return [=](int a, int b) -> bool {
            const std::size_t u = upper_tile_index(a, b, N);
            return (data[u >> 6] & (1ULL << (u & 63ULL))) != 0ULL;
        };
    };
    auto commit = [&](int a, int b) -> bool {
        const std::size_t u = upper_tile_index(a, b, N);
        std::uint64_t& word = idx.bits[u >> 6];
        const std::uint64_t mask = 1ULL << (u & 63ULL);
        if ((word & mask) == 0ULL) { word |= mask; return true; }
        return false;
    };

    return detail::fill_fixpoint_parallel(idx, N, counted, threads, make_has, commit);
}

static inline int fill_closure_tiled_bool_parallel(State& idx, int N, int counted, int threads) {
    const int B = idx.tiled_block_dim;
    const int blocks_per_dim = idx.tiled_blocks_per_dim;
    if (B <= 0 || blocks_per_dim <= 0) return counted;

    auto make_has = [&]() {
        const auto& chunks = idx.tiled_chunks;
        return [&chunks, B, blocks_per_dim](int a, int b) -> bool {
            const int bi = a / B, bj = b / B;
            const int li = a - bi * B, lj = b - bj * B;
            const std::size_t key = static_cast<std::size_t>(bi) * static_cast<std::size_t>(blocks_per_dim) + static_cast<std::size_t>(bj);
            auto it = chunks.find(key);
            if (it == chunks.end()) return false;
            const std::size_t k = static_cast<std::size_t>(li) * static_cast<std::size_t>(B) + static_cast<std::size_t>(lj);
            return (it->second[k >> 6] & (1ULL << (k & 63ULL))) != 0ULL;
        };
    };

    auto commit = [&](int a, int b) -> bool {
        const int bi = a / B, bj = b / B;
        const int li = a - bi * B, lj = b - bj * B;
        const std::size_t key = static_cast<std::size_t>(bi) * static_cast<std::size_t>(blocks_per_dim) + static_cast<std::size_t>(bj);
        const std::size_t block_bits = static_cast<std::size_t>(B) * static_cast<std::size_t>(B);
        const std::size_t words = (block_bits + 63ULL) >> 6;
        auto& vec = idx.tiled_chunks.try_emplace(key, idx.make_chunk_vector(words, 0ULL)).first->second;
        const std::size_t k = static_cast<std::size_t>(li) * static_cast<std::size_t>(B) + static_cast<std::size_t>(lj);
        const std::size_t w = k >> 6;
        const std::uint64_t m = 1ULL << (k & 63ULL);
        if ((vec[w] & m) == 0ULL) { vec[w] |= m; return true; }
        return false;
    };

    return detail::fill_fixpoint_parallel(idx, N, counted, threads, make_has, commit);
}

static inline int fill_closure_tiled_bitset_parallel(State& idx, int N, int counted, int threads) {
    return fill_closure_tiled_bool_parallel(idx, N, counted, threads);
}

static inline int fill_closure_paged_parallel(State& idx, int N, int counted, int threads) {
    (void)threads;
    return fill_closure_paged(idx, N, counted);
}

} // namespace tilecounter

// ----------------------- Heuristic method auto-selection -----------------------
namespace tilecounter {

// Return an estimated memory footprint for dense masks on the upper triangle
// - CharMask: 1 byte per entry
// - BitsetMask: 1 bit per entry (packed into 64-bit words)
static inline void estimate_dense_bytes(std::uint64_t N,
                                        std::uint64_t& char_bytes,
                                        std::uint64_t& bitset_bytes)
{
    // Compute tri = N*(N+1)/2 without 128-bit extensions to satisfy -pedantic.
    // Use divide-first to avoid overflow in the product.
    const std::uint64_t a = N;
    const std::uint64_t b = N + 1ULL;
    const std::uint64_t tri = ((a & 1ULL) == 0ULL) ? ((a >> 1) * b)
                                                   : (a * (b >> 1));
    char_bytes   = tri;                 // 1 byte per entry
    bitset_bytes = (tri + 7ULL) >> 3;   // 1 bit per entry (rounded up)
}

// Heuristic choice of storage method given tile-grid size N, optional memory budget,
// and optional expected density p in (0,1). The goal is to pick a fast-but-feasible
// layout without exhausting memory.
//
// - If mem_budget_bytes > 0 and BitsetMask fits within ~40% of budget, choose BitsetMask.
// - If there is no budget provided but BitsetMask size is modest (< 512 MiB), choose BitsetMask.
// - If expected density p is provided and extremely small (< 1e-4), prefer PagedMask.
// - Otherwise prefer TiledBitsetMask as sparse fallback when dense does not fit.
//
// prefer_char_dense = true lets you pick CharMask when memory allows and you want max per-op speed.
// Auto-selection helpers removed: require explicit Method selection by caller.

} // namespace tilecounter

// Dispatcher to fill based on the strategy used.
namespace tilecounter {
inline int FillTiles(State& idx, Method m, int N, int counted, int num_threads) {
    if (m == Method::Auto) {
        Method resolved = resolve_auto_method_for_fill(idx, m);
        int threads = resolve_auto_threads_for_fill(idx, num_threads > 0 ? num_threads : 1);
        return FillTiles(idx, resolved, N, counted, threads);
    }
    int threads = (num_threads > 1) ? num_threads : 1;
    if (threads > 1) {
    switch (m) {
        case Method::BitsetMask:      return fill_closure_bitset_parallel(idx, N, counted, threads);
        case Method::TiledBoolMask:   return fill_closure_tiled_bool_parallel(idx, N, counted, threads);
        case Method::TiledBitsetMask: return fill_closure_tiled_bitset_parallel(idx, N, counted, threads);
        case Method::PagedMask:       return fill_closure_paged_parallel(idx, N, counted, threads);
        case Method::Auto:            return FillTiles(idx, resolve_auto_method_for_fill(idx, Method::Auto), N, counted, threads);
        default: break;
    }
    }

    switch (m) {
        case Method::Auto:
            return fill_closure_bool(idx, N, counted);
        case Method::CharMask:        return fill_closure_char(idx, N, counted);
        case Method::BoolMask:        return fill_closure_bool(idx, N, counted);
        case Method::LazyLookUp:
            return fill_closure_lazy_lookup(idx, N, counted);
        case Method::BitsetMask:      return fill_closure_bitset(idx, N, counted);
        case Method::TiledBoolMask:   return fill_closure_tiled_bool(idx, N, counted);
        case Method::TiledBitsetMask: return fill_closure_tiled_bitset(idx, N, counted);
        case Method::HashSet:         return fill_closure_hashset(idx, N, counted);
        case Method::SortUnique:      return fill_closure_sortunique(idx, N, counted);
        case Method::PagedMask:       return fill_closure_paged(idx, N, counted);
    }
    return counted;
}

inline int FillTiles(State& idx, Method m, int N, int counted) {
    return FillTiles(idx, m, N, counted, 1);
}

/**
 * @brief Ensure all diagonal tiles are marked as active in the TileIndexer state.
 *        This is critical for Cholesky factorization, which requires all diagonal
 *        tiles to exist even if they have no non-zero entries.
 *
 * @param idx The TileIndexer state to modify
 * @param m The storage method being used
 * @param N Number of tiles per dimension
 * @param counted Current count of active tiles
 * @return Updated count of active tiles after ensuring diagonals
 */
inline int ensure_diagonal_tiles_active(State& idx, Method m, int N, int& counted) {
    if (N <= 0) return counted;

    // Determine which storage method we're using
    Method effective_method = (m == Method::Auto) ? resolve_auto_method_for_fill(idx, m) : m;

    switch (effective_method) {
        case Method::BoolMask:
        case Method::LazyLookUp: {
            if (idx.active_bool.empty()) {
                const std::size_t tri = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
                idx.active_bool.assign(tri, false);
            }
            for (int d = 0; d < N; ++d) {
                const std::size_t diag_idx = upper_tile_index(d, d, N);
                if (diag_idx < idx.active_bool.size() && !idx.active_bool[diag_idx]) {
                    idx.active_bool[diag_idx] = true;
                    ++counted;
                }
            }
            break;
        }

        case Method::CharMask: {
            if (idx.active_char.empty()) {
                const std::size_t tri = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
                idx.active_char.assign(tri, 0);
            }
            for (int d = 0; d < N; ++d) {
                const std::size_t diag_idx = upper_tile_index(d, d, N);
                if (diag_idx < idx.active_char.size() && !idx.active_char[diag_idx]) {
                    idx.active_char[diag_idx] = 1;
                    ++counted;
                }
            }
            break;
        }

        case Method::BitsetMask: {
            if (idx.bits.empty()) {
                const std::size_t tri = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
                const std::size_t words = (tri + 63ULL) >> 6;
                idx.bits.assign(words, 0ULL);
            }
            for (int d = 0; d < N; ++d) {
                const std::size_t diag_idx = upper_tile_index(d, d, N);
                const std::size_t w = diag_idx >> 6;
                const unsigned b = static_cast<unsigned>(diag_idx & 63ULL);
                if (w < idx.bits.size()) {
                    const std::uint64_t mask = (1ULL << b);
                    if (!(idx.bits[w] & mask)) {
                        idx.bits[w] |= mask;
                        ++counted;
                    }
                }
            }
            break;
        }

        case Method::HashSet: {
            for (int d = 0; d < N; ++d) {
                const std::size_t diag_idx = upper_tile_index(d, d, N);
                if (idx.S.insert(diag_idx).second) {
                    ++counted;
                }
            }
            break;
        }

        case Method::SortUnique: {
            // For SortUnique, we need to add to ids and then sort/unique
            std::unordered_set<std::size_t> existing(idx.ids.begin(), idx.ids.end());
            int added = 0;
            for (int d = 0; d < N; ++d) {
                const std::size_t diag_idx = upper_tile_index(d, d, N);
                if (existing.insert(diag_idx).second) {
                    idx.ids.push_back(diag_idx);
                    ++added;
                }
            }
            if (added > 0) {
                sTiles::sort(idx.ids.begin(), idx.ids.end());
                idx.ids.erase(std::unique(idx.ids.begin(), idx.ids.end()), idx.ids.end());
                counted += added;
            }
            break;
        }

        case Method::TiledBoolMask:
        case Method::TiledBitsetMask: {
            // Diagonal tiles are stored in tiled_chunks — must check there, not active_bool
            const int B = idx.tiled_block_dim;
            const int bpd = idx.tiled_blocks_per_dim;
            if (B <= 0 || bpd <= 0) break;
            const std::size_t block_bits = static_cast<std::size_t>(B) * static_cast<std::size_t>(B);
            const std::size_t words = (block_bits + 63ULL) >> 6;
            for (int d = 0; d < N; ++d) {
                const int bl = d / B;
                const int li = d - bl * B;   // li == lj for diagonal
                const std::size_t key = static_cast<std::size_t>(bl) * static_cast<std::size_t>(bpd)
                                      + static_cast<std::size_t>(bl);
                const std::size_t k = static_cast<std::size_t>(li) * static_cast<std::size_t>(B)
                                    + static_cast<std::size_t>(li);
                auto& vec = idx.tiled_chunks.try_emplace(
                    key, idx.make_chunk_vector(words, 0ULL)).first->second;
                const std::uint64_t m = 1ULL << (k & 63ULL);
                if ((vec[k >> 6] & m) == 0ULL) { vec[k >> 6] |= m; ++counted; }
            }
            break;
        }

        case Method::PagedMask: {
            // For paged mask, we need to ensure diagonal bits are set
            const std::size_t page_bits = idx.paged.page_bits ? idx.paged.page_bits : (64ULL * 1024ULL);
            for (int d = 0; d < N; ++d) {
                const std::size_t diag_idx = upper_tile_index(d, d, N);
                const std::size_t page = diag_idx / page_bits;
                const std::size_t bit = diag_idx % page_bits;
                const std::size_t w = bit >> 6;
                const unsigned b = static_cast<unsigned>(bit & 63ULL);

                auto& page_words = idx.paged.pages[page];
                if (w >= page_words.size()) {
                    page_words.resize(w + 1, 0ULL);
                }
                const std::uint64_t mask = (1ULL << b);
                if (!(page_words[w] & mask)) {
                    page_words[w] |= mask;
                    ++counted;
                }
            }
            break;
        }

        default:
            // For unknown methods, try active_bool as fallback
            if (idx.active_bool.empty()) {
                const std::size_t tri = static_cast<std::size_t>(N) * (static_cast<std::size_t>(N) + 1ULL) / 2ULL;
                idx.active_bool.assign(tri, false);
            }
            for (int d = 0; d < N; ++d) {
                const std::size_t diag_idx = upper_tile_index(d, d, N);
                if (diag_idx < idx.active_bool.size() && !idx.active_bool[diag_idx]) {
                    idx.active_bool[diag_idx] = true;
                    ++counted;
                }
            }
            break;
    }

    return counted;
}

} // namespace tilecounter
