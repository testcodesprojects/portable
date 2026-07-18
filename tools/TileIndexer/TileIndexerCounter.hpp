/**
 * @file    TileIndexerCounter.hpp
 * @brief   Header-only helpers and counting routines for TileIndexer including
 *          fast division utilities, OpenMP variants, and strict planning hooks.
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
#include "TileIndexerAutoStrict.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <utility>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <cstddef>
#include "../sort/stiles_sort_dispatch.hpp"

namespace tilecounter {

/*
Group: Inline helpers
Purpose: Common predicates and utilities used by multiple counting paths.
*/
static inline bool invalid_inputs(const int* r, const int* c, int64_t nnz, int n, int tile) {
    return (!r || !c || nnz <= 0 || n <= 0 || tile <= 0);
}
static inline int tiles_from_n(int n, int tile_size) {
    return (n + tile_size - 1) / tile_size;
}
// Row offsets for packed upper triangle (row-major order)
// off[i] = i*(2*N - i - 1)/2 using O(n) adds
/*
Function: make_upper_row_offsets
Purpose:  Precompute row-major offsets for the packed upper triangle to allow
          O(1) mapping from (i,j) to u when i<=j.
*/
static inline std::vector<std::size_t> make_upper_row_offsets(int num_tiles) {
    std::vector<std::size_t> off(static_cast<std::size_t>(num_tiles), 0);
    for (int i = 1; i < num_tiles; ++i) {
        off[static_cast<std::size_t>(i)] = off[static_cast<std::size_t>(i - 1)]
                                         + static_cast<std::size_t>(num_tiles - (i - 1));
    }
    return off;
}
static inline bool out_of_range(int x, int n) {
    return static_cast<unsigned>(x) >= static_cast<unsigned>(n);
}
static inline int fast_div_40_u32(std::uint32_t x) {
    return static_cast<int>((static_cast<std::uint64_t>(x >> 3) * 0xCCCCCCCDull) >> 34);
}

// Additional fast divisions for common tile sizes
static inline int fast_div_80_u32(std::uint32_t x) {
    return fast_div_40_u32(x) >> 1; // 80 = 2 * 40
}
static inline int fast_div_3_u32(std::uint32_t x) {
    // floor(x/3) using 64-bit reciprocal multiply
    return static_cast<int>((static_cast<std::uint64_t>(x) * 0xAAAAAAABull) >> 33);
}
static inline int fast_div_120_u32(std::uint32_t x) {
    // 120 = 3 * 40
    return fast_div_3_u32(static_cast<std::uint32_t>(fast_div_40_u32(x)));
}
static inline int fast_div_5_u32(std::uint32_t x) {
    // floor(x/5)
    return static_cast<int>((static_cast<std::uint64_t>(x) * 0xCCCCCCCDull) >> 34);
}
static inline int fast_div_160_u32(std::uint32_t x) {
    // 160 = 32 * 5
    return fast_div_5_u32(static_cast<std::uint32_t>(x >> 5));
}

// Generic helper for tile division with common fast paths
static inline int fast_div_tile_u32(std::uint32_t x, int tile_size) {
    switch (tile_size) {
        case 40:  return fast_div_40_u32(x);
        case 80:  return fast_div_80_u32(x);
        case 120: return fast_div_120_u32(x);
        case 160: return fast_div_160_u32(x);
        default:  return static_cast<int>(x / static_cast<std::uint32_t>(tile_size));
    }
}

// Heuristic: choose paged bits per page based on matrix size and tile size
static inline std::size_t choose_page_bits_from_target_pages(int num_tiles, int tile_size) {
    const std::size_t tri = static_cast<std::size_t>(num_tiles) * (static_cast<std::size_t>(num_tiles) + 1ULL) / 2ULL;
    // Aim for a practical number of pages across sizes (clamp to [256, 4096])
    std::size_t target_pages = (tri > 0) ? (tri / 200000ULL) : 256ULL; // ~200k entries per page initially
    if (target_pages < 256ULL)  target_pages = 256ULL;
    if (target_pages > 4096ULL) target_pages = 4096ULL;

    // Bits per page so that pages ≈ target_pages, aligned to 64
    std::size_t pb = (tri + target_pages - 1ULL) / target_pages; // ceil
    pb = (pb + 63ULL) & ~63ULL;

    // Minimum: one block of tile_size^2 64-bit words (your original choice)
    const std::size_t min_pb = 64ULL * static_cast<std::size_t>(tile_size) * static_cast<std::size_t>(tile_size);
    if (pb < min_pb) pb = min_pb;

    // Cap: avoid very large pages (e.g., <= 8 MiB/page)
    const std::size_t max_page_bytes = 8ULL * 1024ULL * 1024ULL; // 8 MiB
    const std::size_t max_pb = max_page_bytes * 8ULL;
    if (pb > max_pb) pb = max_pb;

    return pb;
}

// Cache-targeted page sizing for PagedMask.
// - target_page_bytes: cache-friendly size, e.g., 128 KiB or 256 KiB.
// - Enforces a minimum of 64 * tile_size^2 bits and aligns to 64 bits.
// - Caps to a safety maximum to avoid huge per-page allocations.
static inline std::size_t choose_page_bits_from_cache_target(
    int tile_size,
    std::size_t target_page_bytes = 128ULL * 1024ULL,   // 128 KiB default
    std::size_t max_page_bytes    = 1ULL   * 1024ULL * 1024ULL // 1 MiB cap
) {
    const std::size_t min_pb = 64ULL * static_cast<std::size_t>(tile_size) * static_cast<std::size_t>(tile_size);
    std::size_t pb = target_page_bytes * 8ULL;          // bytes -> bits
    pb = (pb + 63ULL) & ~63ULL;                         // align to 64-bit boundary
    if (pb < min_pb) pb = min_pb;
    const std::size_t max_pb = (max_page_bytes * 8ULL) & ~63ULL;
    if (pb > max_pb) pb = max_pb;
    return pb;
}

// Hybrid: cache-targeted first, then gently clamp to limit total page count on huge matrices.
static inline std::size_t choose_page_bits_cache_hybrid(
    int num_tiles, int tile_size,
    std::size_t target_page_bytes = 128ULL * 1024ULL,
    std::size_t max_page_bytes    = 1ULL   * 1024ULL * 1024ULL
) {
    const std::size_t tri = static_cast<std::size_t>(num_tiles) * (static_cast<std::size_t>(num_tiles) + 1ULL) / 2ULL;
    std::size_t pb = choose_page_bits_from_cache_target(tile_size, target_page_bytes, max_page_bytes);
    if (pb == 0) return pb;
    std::size_t pages = (tri + pb - 1ULL) / pb; // ceil(tri/pb)
    if (pages > 4096ULL) {
        // Grow pages to bring count down into a practical band
        pb = (tri + 4096ULL - 1ULL) / 4096ULL; // ceil(tri / 4096)
        pb = (pb + 63ULL) & ~63ULL;
        const std::size_t min_pb = 64ULL * static_cast<std::size_t>(tile_size) * static_cast<std::size_t>(tile_size);
        const std::size_t max_pb = (max_page_bytes * 8ULL) & ~63ULL;
        if (pb < min_pb) pb = min_pb;
        if (pb > max_pb) pb = max_pb;
    }
    return pb;
}

// Upper triangular index (inline definition)
inline std::size_t upper_tile_index(int row_tile, int col_tile, int num_tiles) {
    const std::size_t i = static_cast<std::size_t>(row_tile);
    const std::size_t j = static_cast<std::size_t>(col_tile);
    const std::size_t n = static_cast<std::size_t>(num_tiles);
    // Row-major for upper triangle: off_row[i] + (j - i)
    return (i * (2ULL * n - i + 1ULL)) / 2ULL + (j - i);
}

// Human-readable names (inline definition)
inline const char* to_string(Method m) {
    switch (m) {
        case Method::Auto:           return "Auto";
        case Method::CharMask:        return "CharMask";
        case Method::BoolMask:        return "BoolMask";
        case Method::BitsetMask:      return "BitsetMask";
        case Method::TiledBoolMask:   return "TiledBoolMask";
        case Method::TiledBitsetMask: return "TiledBitsetMask";
        case Method::PagedMask:       return "PagedMask";
        case Method::LazyLookUp:      return "LazyLookUp";
        case Method::HashSet:         return "HashSet";
        case Method::SortUnique:      return "SortUnique";
        default:                      return "Unknown";
    }
}

// Per-strategy counting (header-only)
inline int count_char(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const std::size_t tri = static_cast<std::size_t>(num_tiles) * (num_tiles + 1ULL) / 2ULL;
    st.active_char.assign(tri, 0);
    const auto off = make_upper_row_offsets(num_tiles);
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif
    // non power-of-two handled by fast_div_tile_u32
    int count = 0;
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        if (ti > tj) std::swap(ti, tj);
        const std::size_t u = off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti);
        if (!st.active_char[u]) { st.active_char[u] = 1; ++count; }
    }
    return count;
}

// OpenMP variant: builds thread-local sets of u, then merges and materializes
inline int count_char_omp(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st, int num_cores) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const std::size_t tri = static_cast<std::size_t>(num_tiles) * (num_tiles + 1ULL) / 2ULL;
    const std::size_t words = (tri + 63ULL) >> 6;
    const auto off = make_upper_row_offsets(num_tiles);
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif
    const int T = std::max(1, num_cores);
    std::vector<TileIndexer::State::WordVector> locals;
    locals.reserve(static_cast<std::size_t>(T));
    for (int i = 0; i < T; ++i) {
        TileIndexer::State::WordVector vec(st.makeAllocator<std::uint64_t>(TileIndexer::State::Group::FillScratch));
        vec.assign(words, 0ULL);
        locals.emplace_back(std::move(vec));
    }

#if defined(_OPENMP)
    const int chunk = std::max(1024, nnz / (T * 16));
    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        auto& B = locals[static_cast<std::size_t>(tid)];
        #pragma omp for schedule(static, chunk)
        for (int64_t k = 0; k < nnz; ++k) {
            const int r = row_indices[k], c = col_indices[k];
            if (out_of_range(r, n) || out_of_range(c, n)) continue;
            int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
            int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
            if (ti > tj) std::swap(ti, tj);
            const std::size_t u = off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti);
            B[u >> 6] |= (1ULL << (u & 63ULL));
        }
    }
#else
    (void)threads;
    auto& B = locals[0];
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        if (ti > tj) std::swap(ti, tj);
        const std::size_t u = off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti);
        B[u >> 6] |= (1ULL << (u & 63ULL));
    }
#endif

    // OR-reduce and expand to active_char
    TileIndexer::State::WordVector tmp(st.makeAllocator<std::uint64_t>(TileIndexer::State::Group::FillScratch));
    tmp.assign(words, 0ULL);
    for (const auto& Bv : locals)
        for (std::size_t w = 0; w < words; ++w) tmp[w] |= Bv[w];

    st.active_char.assign(tri, 0);
    int count = 0;
    for (std::size_t w = 0; w < words; ++w) {
        std::uint64_t x = tmp[w];
        while (x) {
#if defined(__GNUC__) || defined(__clang__)
            unsigned tz = static_cast<unsigned>(__builtin_ctzll(x));
#else
            unsigned tz = 0; { std::uint64_t y = x; while ((y & 1ULL) == 0ULL) { ++tz; y >>= 1; } }
#endif
            const std::size_t u = (w << 6) + tz;
            st.active_char[u] = 1; ++count;
            x &= (x - 1);
        }
    }
    return count;
}

inline int count_bool(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const std::size_t tri = static_cast<std::size_t>(num_tiles) * (num_tiles + 1ULL) / 2ULL;
    st.active_bool.assign(tri, false);
    const auto off = make_upper_row_offsets(num_tiles);
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif
    int count = 0;
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        if (ti > tj) std::swap(ti, tj);
        const std::size_t u = off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti);
        if (!st.active_bool[u]) { st.active_bool[u] = true; ++count; }
    }
    return count;
}

inline int count_bool_omp(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st, int num_cores) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const std::size_t tri = static_cast<std::size_t>(num_tiles) * (num_tiles + 1ULL) / 2ULL;
    const std::size_t words = (tri + 63ULL) >> 6;
    const auto off = make_upper_row_offsets(num_tiles);
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif

    const int T = std::max(1, num_cores);
    std::vector<TileIndexer::State::WordVector> locals;
    locals.reserve(static_cast<std::size_t>(T));
    for (int i = 0; i < T; ++i) {
        TileIndexer::State::WordVector vec(st.makeAllocator<std::uint64_t>(TileIndexer::State::Group::FillScratch));
        vec.assign(words, 0ULL);
        locals.emplace_back(std::move(vec));
    }

#if defined(_OPENMP)
    const int chunk = std::max(1024, nnz / (T * 16));
    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        auto& B = locals[static_cast<std::size_t>(tid)];
        #pragma omp for schedule(static, chunk)
        for (int64_t k = 0; k < nnz; ++k) {
            const int r = row_indices[k], c = col_indices[k];
            if (out_of_range(r, n) || out_of_range(c, n)) continue;
            int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
            int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
            const int lo = (ti <= tj) ? ti : tj;
            const int hi = ti ^ tj ^ lo;
            const std::size_t u = off[static_cast<std::size_t>(lo)] + static_cast<std::size_t>(hi - lo);
            B[u >> 6] |= (1ULL << (u & 63ULL));
        }
    }
#else
    (void)threads;
    auto& B = locals[0];
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        const int lo = (ti <= tj) ? ti : tj;
        const int hi = ti ^ tj ^ lo;
        const std::size_t u = off[static_cast<std::size_t>(lo)] + static_cast<std::size_t>(hi - lo);
        B[u >> 6] |= (1ULL << (u & 63ULL));
    }
#endif

    TileIndexer::State::WordVector tmp(st.makeAllocator<std::uint64_t>(TileIndexer::State::Group::FillScratch));
    tmp.assign(words, 0ULL);
    for (const auto& Bv : locals)
        for (std::size_t w = 0; w < words; ++w) tmp[w] |= Bv[w];

    int count = 0;
#if defined(__GNUC__) || defined(__clang__)
    for (std::size_t w = 0; w < words; ++w) count += static_cast<int>(__builtin_popcountll(tmp[w]));
#else
    for (std::size_t w = 0; w < words; ++w) {
        std::uint64_t x = tmp[w];
        x = x - ((x >> 1) & 0x5555555555555555ULL);
        x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
        count += static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
    }
#endif

    st.active_bool.assign(tri, false);
    for (std::size_t w = 0; w < words; ++w) {
        std::uint64_t x = tmp[w];
        while (x) {
#if defined(__GNUC__) || defined(__clang__)
            unsigned tz = static_cast<unsigned>(__builtin_ctzll(x));
#else
            unsigned tz = 0; { std::uint64_t y = x; while ((y & 1ULL) == 0ULL) { ++tz; y >>= 1; } }
#endif
            const std::size_t u = (w << 6) + tz;
            st.active_bool[u] = true;
            x &= (x - 1);
        }
    }
    return count;
}

inline int count_bitset(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const std::size_t tri = static_cast<std::size_t>(num_tiles) * (num_tiles + 1ULL) / 2ULL;
    const std::size_t W = 64;
    const std::size_t words = (tri + W - 1) / W;
    st.bits.assign(words, 0ULL);
    const auto off = make_upper_row_offsets(num_tiles);
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        const int lo = (ti <= tj) ? ti : tj;
        const int hi = ti ^ tj ^ lo;
        const std::size_t u = off[static_cast<std::size_t>(lo)] + static_cast<std::size_t>(hi - lo);
        st.bits[u >> 6] |= (1ULL << (u & 63ULL));
    }
    int count = 0;
#if defined(__GNUC__) || defined(__clang__)
    for (std::size_t w = 0; w < words; ++w) count += static_cast<int>(__builtin_popcountll(st.bits[w]));
#elif defined(_MSC_VER) && defined(_M_X64)
    for (std::size_t w = 0; w < words; ++w) count += static_cast<int>(__popcnt64(st.bits[w]));
#else
    for (std::size_t w = 0; w < words; ++w) {
        std::uint64_t x = st.bits[w];
        x = x - ((x >> 1) & 0x5555555555555555ULL);
        x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
        count += static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
    }
#endif
    return count;
}

// OpenMP variant: thread-local bit-vectors, then OR-reduce
inline int count_bitset_omp(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st, int num_cores) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const std::size_t tri = static_cast<std::size_t>(num_tiles) * (num_tiles + 1ULL) / 2ULL;
    const std::size_t W = 64;
    const std::size_t words = (tri + W - 1) / W;
    const auto off = make_upper_row_offsets(num_tiles);
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif

    const int threads = std::max(1, num_cores);
    std::vector<TileIndexer::State::WordVector> locals;
    locals.reserve(static_cast<std::size_t>(threads));
    for (int i = 0; i < threads; ++i) {
        TileIndexer::State::WordVector vec(st.makeAllocator<std::uint64_t>(TileIndexer::State::Group::FillScratch));
        vec.assign(words, 0ULL);
        locals.emplace_back(std::move(vec));
    }

#if defined(_OPENMP)
    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        auto& B = locals[static_cast<std::size_t>(tid)];
        #pragma omp for schedule(static)
        for (int64_t k = 0; k < nnz; ++k) {
            const int r = row_indices[k], c = col_indices[k];
            if (out_of_range(r, n) || out_of_range(c, n)) continue;
            int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
            int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
            const int lo = (ti <= tj) ? ti : tj;
            const int hi = ti ^ tj ^ lo;
            const std::size_t u = off[static_cast<std::size_t>(lo)] + static_cast<std::size_t>(hi - lo);
            B[u >> 6] |= (1ULL << (u & 63ULL));
        }
    }
#else
    (void)threads;
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        const int lo = (ti <= tj) ? ti : tj;
        const int hi = ti ^ tj ^ lo;
        const std::size_t u = off[static_cast<std::size_t>(lo)] + static_cast<std::size_t>(hi - lo);
        locals[0][u >> 6] |= (1ULL << (u & 63ULL));
    }
#endif

    st.bits.assign(words, 0ULL);
    for (const auto& B : locals) {
        for (std::size_t w = 0; w < words; ++w) st.bits[w] |= B[w];
    }
    int count = 0;
#if defined(__GNUC__) || defined(__clang__)
    for (std::size_t w = 0; w < words; ++w) count += static_cast<int>(__builtin_popcountll(st.bits[w]));
#elif defined(_MSC_VER) && defined(_M_X64)
    for (std::size_t w = 0; w < words; ++w) count += static_cast<int>(__popcnt64(st.bits[w]));
#else
    for (std::size_t w = 0; w < words; ++w) {
        std::uint64_t x = st.bits[w];
        x = x - ((x >> 1) & 0x5555555555555555ULL);
        x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
        count += static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
    }
#endif
    return count;
}

inline int count_tiled_chunks(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const int B = std::max(1, std::min(num_tiles, 8 * tile_size));
    const int Bdim = (num_tiles + B - 1) / B;
    st.tiled_block_dim = B;
    st.tiled_blocks_per_dim = Bdim;
    const std::size_t block_bits = static_cast<std::size_t>(B) * static_cast<std::size_t>(B);
    const std::size_t words_per_block = (block_bits + 63ULL) >> 6;
    const std::size_t num_blocks = static_cast<std::size_t>(Bdim) * static_cast<std::size_t>(Bdim);
    std::vector<TileIndexer::ChunkVector> blocks;
    blocks.reserve(num_blocks);
    for (std::size_t i = 0; i < num_blocks; ++i) {
        blocks.emplace_back(st.make_chunk_vector(0));
    }
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        if (ti > tj) std::swap(ti, tj);
        const int bi = ti / B, bj = tj / B;
        const int li = ti - bi * B, lj = tj - bj * B;
        const std::size_t idx = static_cast<std::size_t>(li) * static_cast<std::size_t>(B) + static_cast<std::size_t>(lj);
        auto& vec = blocks[static_cast<std::size_t>(bi) * static_cast<std::size_t>(Bdim) + static_cast<std::size_t>(bj)];
        if (vec.empty()) vec.assign(words_per_block, 0ULL);
        vec[idx >> 6] |= (1ULL << (idx & 63ULL));
    }
    int count = 0;
    st.tiled_chunks.clear();
    for (std::size_t bi = 0; bi < static_cast<std::size_t>(Bdim); ++bi) {
        for (std::size_t bj = 0; bj < static_cast<std::size_t>(Bdim); ++bj) {
            const std::size_t key = bi * static_cast<std::size_t>(Bdim) + bj;
            auto& vec = blocks[key];
            if (vec.empty()) continue;
#if defined(__GNUC__) || defined(__clang__)
            for (std::size_t w = 0; w < vec.size(); ++w) count += static_cast<int>(__builtin_popcountll(vec[w]));
#elif defined(_MSC_VER) && defined(_M_X64)
            for (std::size_t w = 0; w < vec.size(); ++w) count += static_cast<int>(__popcnt64(vec[w]));
#else
            for (std::size_t w = 0; w < vec.size(); ++w) {
                std::uint64_t x = vec[w];
                x = x - ((x >> 1) & 0x5555555555555555ULL);
                x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
                count += static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
            }
#endif
            st.tiled_chunks.emplace(key, std::move(vec));
        }
    }
    return count;
}

// Parallel variant for tiled chunks: thread-local block maps then OR-merge and popcount
inline int count_tiled_chunks_omp(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st, int num_cores) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const int B = std::max(1, std::min(num_tiles, 8 * tile_size));
    const int Bdim = (num_tiles + B - 1) / B;
    const std::size_t block_bits = static_cast<std::size_t>(B) * static_cast<std::size_t>(B);
    const std::size_t words_per_block = (block_bits + 63ULL) >> 6;

    st.tiled_block_dim = B;
    st.tiled_blocks_per_dim = Bdim;
    st.tiled_chunks.clear();

    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif

    const int T = std::max(1, num_cores);
    using ChunkVector = TileIndexer::ChunkVector;
    using BlockMapAlloc = TileIndexer::State::Allocator<std::pair<const std::size_t, ChunkVector>>;
    using BlockMap = std::unordered_map<std::size_t, ChunkVector, std::hash<std::size_t>, std::equal_to<std::size_t>, BlockMapAlloc>;
    std::vector<BlockMap> locals;
    locals.reserve(static_cast<std::size_t>(T));
    for (int i = 0; i < T; ++i) {
        locals.emplace_back(0, std::hash<std::size_t>{}, std::equal_to<std::size_t>{},
                            st.makeAllocator<std::pair<const std::size_t, ChunkVector>>(TileIndexer::State::Group::TiledChunkMap));
    }

#if defined(_OPENMP)
    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        auto& map = locals[static_cast<std::size_t>(tid)];
        #pragma omp for schedule(static)
        for (int64_t k = 0; k < nnz; ++k) {
            const int r = row_indices[k], c = col_indices[k];
            if (out_of_range(r, n) || out_of_range(c, n)) continue;
            int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
            int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
            if (ti > tj) std::swap(ti, tj);
            const int bi = ti / B, bj = tj / B;
            const int li = ti - bi * B, lj = tj - bj * B;
            const std::size_t key = static_cast<std::size_t>(bi) * static_cast<std::size_t>(Bdim) + static_cast<std::size_t>(bj);
            auto [it, inserted] = map.try_emplace(key, st.make_chunk_vector(words_per_block, 0ULL));
            auto& vec = it->second;
            if (!inserted && vec.empty()) vec.assign(words_per_block, 0ULL);
            const std::size_t idx = static_cast<std::size_t>(li) * static_cast<std::size_t>(B) + static_cast<std::size_t>(lj);
            vec[idx >> 6] |= (1ULL << (idx & 63ULL));
        }
    }
#else
    (void)num_cores;
    auto& map = locals[0];
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        if (ti > tj) std::swap(ti, tj);
        const int bi = ti / B, bj = tj / B;
        const int li = ti - bi * B, lj = tj - bj * B;
        const std::size_t key = static_cast<std::size_t>(bi) * static_cast<std::size_t>(Bdim) + static_cast<std::size_t>(bj);
        auto [it, inserted] = map.try_emplace(key, st.make_chunk_vector(words_per_block, 0ULL));
        auto& vec = it->second;
        if (!inserted && vec.empty()) vec.assign(words_per_block, 0ULL);
        const std::size_t idx = static_cast<std::size_t>(li) * static_cast<std::size_t>(B) + static_cast<std::size_t>(lj);
        vec[idx >> 6] |= (1ULL << (idx & 63ULL));
    }
#endif

    // Merge locals into global map
    for (auto& mp : locals) {
        for (auto& kv : mp) {
            auto it = st.tiled_chunks.find(kv.first);
            if (it == st.tiled_chunks.end()) {
                st.tiled_chunks.emplace(kv.first, std::move(kv.second));
            } else {
                auto& dst = it->second; auto& src = kv.second;
                if (dst.size() < src.size()) dst.resize(src.size(), 0ULL);
                for (std::size_t w = 0; w < src.size(); ++w) dst[w] |= src[w];
            }
        }
    }

    // Count set bits across all blocks
    int count = 0;
#if defined(_OPENMP)
    #pragma omp parallel for reduction(+:count) schedule(static)
#endif
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(st.tiled_chunks.size()); ++idx) {
        auto it = st.tiled_chunks.begin();
        std::advance(it, idx);
        const auto& vec = it->second;
#if defined(__GNUC__) || defined(__clang__)
        for (std::size_t w = 0; w < vec.size(); ++w) count += static_cast<int>(__builtin_popcountll(vec[w]));
#else
        for (std::size_t w = 0; w < vec.size(); ++w) {
            std::uint64_t x = vec[w];
            x = x - ((x >> 1) & 0x5555555555555555ULL);
            x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
            count += static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
        }
#endif
    }
    return count;
}

inline int count_hashset(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const auto off = make_upper_row_offsets(num_tiles);
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif
    st.S.clear();
    st.S.reserve(static_cast<std::size_t>(nnz));
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        if (ti > tj) std::swap(ti, tj);
        st.S.insert(off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti));
    }
    return static_cast<int>(st.S.size());
}

// OpenMP variant: thread-local sets, then merge
inline int count_hashset_omp(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st, int num_cores) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const auto off = make_upper_row_offsets(num_tiles);
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif
    const int threads = std::max(1, num_cores);
    std::vector<std::unordered_set<std::size_t>> locals(static_cast<std::size_t>(threads));

#if defined(_OPENMP)
    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        auto& S = locals[static_cast<std::size_t>(tid)];
        S.reserve(static_cast<std::size_t>(nnz / threads + 64));
        #pragma omp for schedule(static)
        for (int64_t k = 0; k < nnz; ++k) {
            const int r = row_indices[k], c = col_indices[k];
            if (out_of_range(r, n) || out_of_range(c, n)) continue;
            int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
            int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
            if (ti > tj) std::swap(ti, tj);
            S.insert(off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti));
        }
    }
#else
    (void)num_cores;
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        if (ti > tj) std::swap(ti, tj);
        locals[0].insert(off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti));
    }
#endif

    std::size_t total = 0; for (const auto& S : locals) total += S.size();
    st.S.clear(); st.S.reserve(total);
    for (auto& L : locals) { for (auto u : L) st.S.insert(u); }
    return static_cast<int>(st.S.size());
}

inline int count_sortunique(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const auto off = make_upper_row_offsets(num_tiles);
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif
    st.ids.clear(); st.ids.reserve(static_cast<std::size_t>(nnz));
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        if (ti > tj) std::swap(ti, tj);
        st.ids.push_back(off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti));
    }
    sTiles::sort(st.ids.begin(), st.ids.end());
    st.ids.erase(std::unique(st.ids.begin(), st.ids.end()), st.ids.end());
    return static_cast<int>(st.ids.size());
}

// OpenMP variant: thread-local vectors, then concatenate and unique
inline int count_sortunique_omp(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st, int num_cores) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const auto off = make_upper_row_offsets(num_tiles);
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif

    const int threads = std::max(1, num_cores);
    std::vector<std::vector<std::size_t>> locals(static_cast<std::size_t>(threads));
    for (auto& v : locals) v.reserve(static_cast<std::size_t>(nnz / threads + 64));

#if defined(_OPENMP)
    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        auto& V = locals[static_cast<std::size_t>(tid)];
        #pragma omp for schedule(static)
        for (int64_t k = 0; k < nnz; ++k) {
            const int r = row_indices[k], c = col_indices[k];
            if (out_of_range(r, n) || out_of_range(c, n)) continue;
            int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
            int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
            if (ti > tj) std::swap(ti, tj);
            V.push_back(off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti));
        }
    }
#else
    (void)num_cores;
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        if (ti > tj) std::swap(ti, tj);
        locals[0].push_back(off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti));
    }
#endif

    st.ids.clear();
    std::size_t total = 0; for (const auto& V : locals) total += V.size();
    st.ids.reserve(total);
    for (auto& V : locals) { st.ids.insert(st.ids.end(), V.begin(), V.end()); }
    sTiles::sort(st.ids.begin(), st.ids.end());
    st.ids.erase(std::unique(st.ids.begin(), st.ids.end()), st.ids.end());
    return static_cast<int>(st.ids.size());
}

inline int count_paged(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const auto off = make_upper_row_offsets(num_tiles);
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif
    // Choose page geometry: cache-friendly with gentle global limit
    const std::size_t PB = choose_page_bits_cache_hybrid(num_tiles, tile_size,
                                                         128ULL * 1024ULL,  // target page ~128 KiB
                                                         1ULL   * 1024ULL * 1024ULL); // cap 1 MiB
    st.paged.page_bits  = PB;
    st.paged.page_words = (PB >> 6);
    st.paged.pages.clear();
    auto test_and_set = [&](std::size_t u, int& cnt) {
        const std::size_t page = st.paged.page_bits ? (u / st.paged.page_bits) : 0;
        const std::size_t bit  = st.paged.page_bits ? (u % st.paged.page_bits) : u;
        const std::size_t w    = bit >> 6;
        const std::uint64_t m  = 1ULL << (bit & 63ULL);
        auto it = st.paged.pages.find(page);
        if (it == st.paged.pages.end()) it = st.paged.pages.emplace(page, st.paged.make_page_vector(st.paged.page_words, 0ULL)).first;
        std::uint64_t& word = it->second[w];
        if ((word & m) == 0ULL) { word |= m; ++cnt; }
    };
    int count = 0;
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        const int lo = (ti <= tj) ? ti : tj;
        const int hi = ti ^ tj ^ lo;
        const std::size_t u = off[static_cast<std::size_t>(lo)] + static_cast<std::size_t>(hi - lo);
        test_and_set(u, count);
    }
    return count;
}

// OpenMP variant: thread-local page maps, then merge OR into st.paged.pages
inline int count_paged_omp(const int* row_indices, const int* col_indices, int64_t nnz, int n, int tile_size, State& st, int num_cores) {
    const int num_tiles = tiles_from_n(n, tile_size);
    const auto off = make_upper_row_offsets(num_tiles);
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#else
    unsigned shift = 0; if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif
    const std::size_t PB = choose_page_bits_cache_hybrid(num_tiles, tile_size,
                                                         128ULL * 1024ULL,
                                                         1ULL   * 1024ULL * 1024ULL);
    st.paged.page_bits  = PB;
    st.paged.page_words = (PB >> 6);
    st.paged.pages.clear();

    using PageVector = TileIndexer::State::Paged::PageVector;
    using PageMapAlloc = TileIndexer::State::Allocator<std::pair<const std::size_t, PageVector>>;
    using PageMap = std::unordered_map<std::size_t, PageVector, std::hash<std::size_t>, std::equal_to<std::size_t>, PageMapAlloc>;
    const int threads = std::max(1, num_cores);
    std::vector<PageMap> locals;
    locals.reserve(static_cast<std::size_t>(threads));
    for (int i = 0; i < threads; ++i) {
        locals.emplace_back(0, std::hash<std::size_t>{}, std::equal_to<std::size_t>{},
                             st.makeAllocator<std::pair<const std::size_t, PageVector>>(TileIndexer::State::Group::PagedMap));
    }

#if defined(_OPENMP)
    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        auto& P = locals[static_cast<std::size_t>(tid)];
        #pragma omp for schedule(static)
        for (int64_t k = 0; k < nnz; ++k) {
            const int r = row_indices[k], c = col_indices[k];
            if (out_of_range(r, n) || out_of_range(c, n)) continue;
            int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
            int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                          : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
            const int lo = (ti <= tj) ? ti : tj;
            const int hi = ti ^ tj ^ lo;
            const std::size_t u = off[static_cast<std::size_t>(lo)] + static_cast<std::size_t>(hi - lo);
            const std::size_t page = st.paged.page_bits ? (u / st.paged.page_bits) : 0;
            const std::size_t bit  = st.paged.page_bits ? (u % st.paged.page_bits) : u;
            const std::size_t w    = bit >> 6;
            const std::uint64_t m  = 1ULL << (bit & 63ULL);
            auto it = P.find(page);
            if (it == P.end()) it = P.emplace(page, st.paged.make_page_vector(st.paged.page_words, 0ULL)).first;
            it->second[w] |= m;
        }
    }
#else
    (void)threads;
    auto& P = locals[0];
    for (int64_t k = 0; k < nnz; ++k) {
        const int r = row_indices[k], c = col_indices[k];
        if (out_of_range(r, n) || out_of_range(c, n)) continue;
        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(r), tile_size);
        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift)
                      : fast_div_tile_u32(static_cast<std::uint32_t>(c), tile_size);
        const int lo = (ti <= tj) ? ti : tj;
        const int hi = ti ^ tj ^ lo;
        const std::size_t u = off[static_cast<std::size_t>(lo)] + static_cast<std::size_t>(hi - lo);
        const std::size_t page = st.paged.page_bits ? (u / st.paged.page_bits) : 0;
        const std::size_t bit  = st.paged.page_bits ? (u % st.paged.page_bits) : u;
        const std::size_t w    = bit >> 6;
        const std::uint64_t m  = 1ULL << (bit & 63ULL);
        auto it = P.find(page);
        if (it == P.end()) it = P.emplace(page, st.paged.make_page_vector(st.paged.page_words, 0ULL)).first;
        it->second[w] |= m;
    }
#endif

    int count = 0;
    for (auto& Pm : locals) {
        for (auto& kv : Pm) {
            auto it = st.paged.pages.find(kv.first);
            if (it == st.paged.pages.end()) it = st.paged.pages.emplace(kv.first, st.paged.make_page_vector(st.paged.page_words, 0ULL)).first;
            auto& dst = it->second; auto& src = kv.second;
            for (std::size_t w = 0; w < src.size(); ++w) {
                const std::uint64_t before = dst[w];
                dst[w] |= src[w];
#if defined(__GNUC__) || defined(__clang__)
                count += static_cast<int>(__builtin_popcountll(dst[w] & ~before));
#elif defined(_MSC_VER) && defined(_M_X64)
                count += static_cast<int>(__popcnt64(dst[w] & ~before));
#else
                std::uint64_t x = (dst[w] & ~before);
                x = x - ((x >> 1) & 0x5555555555555555ULL);
                x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
                count += static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
#endif
            }
        }
    }
    return count;
}

// Dispatcher: header-only switch that calls per-strategy functions
inline int countActiveTiles(const int* row_indices,
                            const int* col_indices,
                            int64_t nnz, int n, int tile_size,
                            Method method,
                            State* state,
                            int num_cores,
                            int group_id)
{
    if (invalid_inputs(row_indices, col_indices, nnz, n, tile_size)) return 0;
    State local(group_id); State& st = state ? *state : local;

    Method resolved_method = method;
    int resolved_threads = (num_cores > 0) ? num_cores : 1;
    if (method == Method::Auto) {
        auto plan = make_universal_plan(static_cast<std::uint64_t>(n),
                                        static_cast<std::uint64_t>(nnz),
                                        tile_size);
        resolved_method = plan.count_method;
        if (plan.count_threads > 0) resolved_threads = plan.count_threads;
        st.auto_plan_valid = true;
        st.auto_count_method = plan.count_method;
        st.auto_fill_method = plan.fill_method;
        st.auto_count_threads = plan.count_threads;
        st.auto_fill_threads = plan.fill_threads;
    } else {
        st.auto_plan_valid = false;
        st.auto_count_method = method;
        st.auto_fill_method = method;
        st.auto_count_threads = resolved_threads;
        st.auto_fill_threads = resolved_threads;
    }

    const int threads = (resolved_threads > 1) ? resolved_threads : 1;
    int ret = 0;
    switch (resolved_method) {
        case Method::CharMask:
            ret = (threads > 1 ? count_char_omp(row_indices, col_indices, nnz, n, tile_size, st, threads)
                                 : count_char(row_indices, col_indices, nnz, n, tile_size, st));
            break;
        case Method::BoolMask:
            ret = (threads > 1 ? count_bool_omp(row_indices, col_indices, nnz, n, tile_size, st, threads)
                                 : count_bool(row_indices, col_indices, nnz, n, tile_size, st));
            break;
        case Method::BitsetMask:
            ret = (threads > 1 ? count_bitset_omp(row_indices, col_indices, nnz, n, tile_size, st, threads)
                                 : count_bitset(row_indices, col_indices, nnz, n, tile_size, st));
            break;
        case Method::TiledBoolMask:
            ret = (threads > 1 ? count_tiled_chunks_omp(row_indices, col_indices, nnz, n, tile_size, st, threads)
                                 : count_tiled_chunks(row_indices, col_indices, nnz, n, tile_size, st));
            break;
        case Method::TiledBitsetMask:
            ret = (threads > 1 ? count_tiled_chunks_omp(row_indices, col_indices, nnz, n, tile_size, st, threads)
                                 : count_tiled_chunks(row_indices, col_indices, nnz, n, tile_size, st));
            break;
        case Method::PagedMask:
            ret = (threads > 1 ? count_paged_omp(row_indices, col_indices, nnz, n, tile_size, st, threads)
                                 : count_paged(row_indices, col_indices, nnz, n, tile_size, st));
            break;
        case Method::HashSet:
            ret = (threads > 1 ? count_hashset_omp(row_indices, col_indices, nnz, n, tile_size, st, threads)
                                 : count_hashset(row_indices, col_indices, nnz, n, tile_size, st));
            break;
        case Method::SortUnique:
            ret = (threads > 1 ? count_sortunique_omp(row_indices, col_indices, nnz, n, tile_size, st, threads)
                                 : count_sortunique(row_indices, col_indices, nnz, n, tile_size, st));
            break;
        case Method::Auto:
            ret = (threads > 1 ? count_bool_omp(row_indices, col_indices, nnz, n, tile_size, st, threads)
                               : count_bool(row_indices, col_indices, nnz, n, tile_size, st));
            break;
        default:
            ret = count_char(row_indices, col_indices, nnz, n, tile_size, st);
            break;
    }
    return ret;
}

} // namespace tilecounter
