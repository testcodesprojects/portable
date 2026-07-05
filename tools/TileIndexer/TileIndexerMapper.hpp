/**
 * @file    TileIndexerMapper.hpp
 * @brief   Map active tile coordinates or packed indices to compact ranks for
 *          dense, paged, and ID-based representations with a unified façade.
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

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <variant>
#include <cassert>

namespace tilecounter {

inline TileIndexer::State::WordVector& acquire_scratch(const TileIndexer::State& owner,
                                                       std::size_t words) {
    auto& scratch = owner.scratch_bits;
    scratch.assign(words, 0ULL);
    return scratch;
}

// Canonical packed index helper (assumes any i,j; reorders to i<=j)
inline std::size_t packed_u(int i, int j, int n) {
    if (i > j) std::swap(i, j);
    return upper_tile_index(i, j, n);
}

// ----------------------
// Bitset rank mapper
// ----------------------
// Computes compact rank k for a set bit at u in O(1), using word-prefix popcounts.
/*
Struct: BitsetRankMapper
Purpose: O(1) rank lookup for dense packed bitsets using per-word prefix sums.
*/
struct BitsetRankMapper {
    const std::uint64_t* words{nullptr};
    std::size_t word_count{0};
    std::vector<std::uint32_t> word_prefix; // prefix[w] = popcount of words [0..w-1]

    // Returns -1 if u not active; else returns compact index k in [0..#active)
    inline int map_u(std::size_t u) const {
        if (!words) return -1;
        const std::size_t w = u >> 6; const unsigned b = static_cast<unsigned>(u & 63ULL);
        if (w >= word_count) return -1;
        const std::uint64_t word = words[w];
        const std::uint64_t mask = (b == 0 ? 0ULL : ((1ULL << b) - 1ULL));
        if (((word >> b) & 1ULL) == 0ULL) return -1;
        return static_cast<int>(word_prefix[w]
                              + static_cast<std::uint32_t>(__builtin_popcountll(word & mask)));
    }
    inline int map_ij(int i, int j, int n) const { return map_u(packed_u(i,j,n)); }
};

// Build from a bitset directly
template <typename Alloc>
inline BitsetRankMapper make_bitset_rank_mapper(const std::vector<std::uint64_t, Alloc>& bits) {
    BitsetRankMapper m; m.words = bits.data(); m.word_count = bits.size(); m.word_prefix.resize(bits.size());
    std::uint32_t acc = 0;
    for (std::size_t w = 0; w < bits.size(); ++w) {
        m.word_prefix[w] = acc;
        acc += static_cast<std::uint32_t>(__builtin_popcountll(bits[w]));
    }
    return m;
}

// Build from bool/char masks by packing into a temporary bitset
template <typename Alloc>
inline BitsetRankMapper make_rank_from_dense_mask(const TileIndexer::State& owner,
                                                  const std::vector<char, Alloc>& active_char) {
    const std::size_t T = active_char.size();
    const std::size_t words = (T + 63ULL) >> 6;
    auto& tmp_bits = acquire_scratch(owner, words);
    for (std::size_t u = 0; u < T; ++u) {
        if (active_char[u]) {
            tmp_bits[u >> 6] |= (1ULL << (u & 63ULL));
        }
    }
    return make_bitset_rank_mapper(tmp_bits);
}

template <typename Alloc>
inline BitsetRankMapper make_rank_from_dense_mask(const TileIndexer::State& owner,
                                                  const std::vector<bool, Alloc>& active_bool) {
    const std::size_t T = active_bool.size();
    const std::size_t words = (T + 63ULL) >> 6;
    auto& tmp_bits = acquire_scratch(owner, words);
    for (std::size_t u = 0; u < T; ++u) {
        if (active_bool[u]) {
            tmp_bits[u >> 6] |= (1ULL << (u & 63ULL));
        }
    }
    return make_bitset_rank_mapper(tmp_bits);
}

// ----------------------
// Paged rank mapper
// ----------------------
// For PagedMask: O(1) rank by page-prefix + per-page word-prefix.
/*
Struct: PagedRankMapper
Purpose: O(1) rank lookup for paged bitsets, combining page and word prefixes.
*/
struct PagedRankMapper {
    const TileIndexer::PagedState* paged{nullptr};
    // Cumulative popcount before each page
    std::unordered_map<std::size_t, std::size_t> page_prefix;
    // Per-page word-prefix arrays (index by page id)
    std::unordered_map<std::size_t, std::vector<std::uint32_t>> page_word_prefix;

    inline int map_u(std::size_t u) const {
        const std::size_t PB = paged->page_bits; if (!PB) return -1;
        const std::size_t page = u / PB; const std::size_t bit = u % PB;
        const std::size_t w = bit >> 6; const unsigned b = static_cast<unsigned>(bit & 63ULL);
        auto itP = paged->pages.find(page); if (itP == paged->pages.end()) return -1;
        const auto& words = itP->second; if (w >= words.size()) return -1;
        const std::uint64_t word = words[w];
        if (((word >> b) & 1ULL) == 0ULL) return -1;
        const std::size_t base = page_prefix.at(page);
        const std::uint32_t wpref = page_word_prefix.at(page)[w];
        const std::uint64_t mask = (b == 0 ? 0ULL : ((1ULL << b) - 1ULL));
        return static_cast<int>(base + wpref + static_cast<std::uint32_t>(__builtin_popcountll(word & mask)));
    }
    inline int map_ij(int i, int j, int n) const { return map_u(packed_u(i,j,n)); }
};

inline PagedRankMapper make_paged_rank_mapper(const TileIndexer::PagedState& paged) {
    PagedRankMapper m; m.paged = &paged;
    // Build page order and prefix
    std::vector<std::size_t> keys; keys.reserve(paged.pages.size());
    for (const auto& kv : paged.pages) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    std::size_t acc = 0;
    for (auto p : keys) {
        m.page_prefix.emplace(p, acc);
        // per-page word prefix
        const auto& words = paged.pages.at(p);
        auto& wpref = m.page_word_prefix[p]; wpref.resize(words.size());
        std::uint32_t wacc = 0;
        for (std::size_t w = 0; w < words.size(); ++w) { wpref[w] = wacc; wacc += static_cast<std::uint32_t>(__builtin_popcountll(words[w])); }
        acc += wacc;
    }
    return m;
}

// ----------------------
// ID-based mapping (HashSet / SortUnique)
// ----------------------
/*
Struct: IdRankMapper
Purpose: Rank mapping for sparse ID sets/vectors.
*/
struct IdRankMapper {
    std::unordered_map<std::size_t,int> rank; // u -> k
    inline int map_u(std::size_t u) const {
        auto it = rank.find(u); return (it == rank.end() ? -1 : it->second);
    }
    inline int map_ij(int i, int j, int n) const { return map_u(packed_u(i,j,n)); }
};

template <typename Alloc>
inline IdRankMapper make_rank_from_ids(const std::vector<std::size_t, Alloc>& ids) {
    IdRankMapper m; m.rank.reserve(ids.size());
    int t = 0; for (auto u : ids) m.rank.emplace(u, t++);
    return m;
}

template <typename Hash, typename Eq, typename Alloc>
inline IdRankMapper make_rank_from_set(const std::unordered_set<std::size_t, Hash, Eq, Alloc>& S) {
    // Build sorted ids first to ensure stable compact order if desired
    std::vector<std::size_t> ids;
    ids.reserve(S.size());
    for (auto u : S) {
        ids.push_back(u);
    }
    std::sort(ids.begin(), ids.end());
    return make_rank_from_ids(ids);
}

// ----------------------
// Convenience per-strategy builders
// ----------------------
inline BitsetRankMapper build_mapper_bitset(const TileIndexer::State& idx) {
    return make_bitset_rank_mapper(idx.bits);
}

inline BitsetRankMapper build_mapper_char(const TileIndexer::State& idx) {
    return make_rank_from_dense_mask(idx, idx.active_char);
}

inline BitsetRankMapper build_mapper_bool(const TileIndexer::State& idx) {
    return make_rank_from_dense_mask(idx, idx.active_bool);
}

inline PagedRankMapper build_mapper_paged(const TileIndexer::State& idx) {
    return make_paged_rank_mapper(idx.paged);
}

inline IdRankMapper build_mapper_hashset(const TileIndexer::State& idx) {
    return make_rank_from_set(idx.S);
}

inline IdRankMapper build_mapper_sortunique(const TileIndexer::State& idx) {
    // idx.ids is already sort+unique after counting; keep its order
    return make_rank_from_ids(idx.ids);
}

// Tiled: either convert to a temporary bitset or page and reuse above
inline BitsetRankMapper build_mapper_tiled_as_bitset(const TileIndexer::State& idx, std::size_t triangular_size) {
    const std::size_t words = (triangular_size + 63ULL) >> 6;
    auto& tmp_bits = acquire_scratch(idx, words);
    const int B = idx.tiled_block_dim; const int Bdim = idx.tiled_blocks_per_dim;
    if (B > 0 && Bdim > 0) {
        for (const auto& kv : idx.tiled_chunks) {
            const auto& vec = kv.second;
            for (std::size_t w = 0; w < vec.size(); ++w) tmp_bits[w] |= vec[w];
        }
    }
    return make_bitset_rank_mapper(tmp_bits);
}

static inline void order_pair(int& i, int& j) noexcept {
    const int lo = (i <= j) ? i : j;
    j = (i ^ j ^ lo);
    i = lo;
}

enum class MapperKind : unsigned { None, Bitset, Paged, Id };

/*
Struct: Mapper
Purpose: Strategy-aware façade that performs rank mapping without exposing the
         underlying representation. Supports Bitset, Paged, and Id variants.
*/
struct Mapper {
    MapperKind kind{MapperKind::None};
    std::variant<std::monostate, BitsetRankMapper, PagedRankMapper, IdRankMapper> var;

    inline int map_u(std::size_t u) const noexcept {
        switch (kind) {
            case MapperKind::Bitset: return std::get<BitsetRankMapper>(var).map_u(u);
            case MapperKind::Paged:  return std::get<PagedRankMapper>(var).map_u(u);
            case MapperKind::Id:     return std::get<IdRankMapper>(var).map_u(u);
            default: return -1;
        }
    }

    inline int map_ij(int i, int j, int n) const noexcept {
        order_pair(i, j);
        return map_u(upper_tile_index(i, j, n));
    }

    inline bool valid()  const noexcept { return kind != MapperKind::None; }
    inline bool dense()  const noexcept { return kind == MapperKind::Bitset; }
    inline bool paged()  const noexcept { return kind == MapperKind::Paged;  }
    inline bool by_ids() const noexcept { return kind == MapperKind::Id;     }
};

inline Method resolve_auto_method_for_mapper(const TileIndexer::State& st) {
    if (!st.bits.empty())                                  return Method::BitsetMask;
    if (!st.active_bool.empty())                           return Method::BoolMask;
    if (!st.active_char.empty())                           return Method::CharMask;
    if (st.paged.page_bits && !st.paged.pages.empty())     return Method::PagedMask;
    if (!st.ids.empty())                                   return Method::SortUnique;
    if (!st.S.empty())                                     return Method::HashSet;
    if (!st.tiled_chunks.empty())                          return Method::TiledBitsetMask;
    return Method::BoolMask;
}

inline Mapper build_mapper(const TileIndexer::State& st, Method method, int num_tiles) {
    
    Mapper mm;

    if (method == Method::Auto) {
        method = resolve_auto_method_for_mapper(st);
    }

    switch (method) {
        case Method::CharMask:
            mm.kind = MapperKind::Bitset;
            mm.var  = build_mapper_char(st);
            break;
        case Method::BoolMask:
            mm.kind = MapperKind::Bitset;
            mm.var  = build_mapper_bool(st);
            break;
        case Method::BitsetMask:
            mm.kind = MapperKind::Bitset;
            mm.var  = build_mapper_bitset(st);
            break;

        case Method::PagedMask:
            mm.kind = MapperKind::Paged;
            mm.var  = build_mapper_paged(st);
            break;

        case Method::HashSet:
            mm.kind = MapperKind::Id;
            mm.var  = build_mapper_hashset(st);
            break;
        case Method::SortUnique:
            mm.kind = MapperKind::Id;
            mm.var  = build_mapper_sortunique(st);
            break;

        case Method::TiledBoolMask:
        case Method::TiledBitsetMask: {
            assert(num_tiles > 0 && "build_mapper(..., Tiled*, num_tiles) requires num_tiles > 0");
            const std::size_t tri = static_cast<std::size_t>(num_tiles) * (static_cast<std::size_t>(num_tiles) + 1ULL) / 2ULL;
            mm.kind = MapperKind::Bitset;
            mm.var  = build_mapper_tiled_as_bitset(st, tri);
            break;
        }

        case Method::LazyLookUp:
        default:
            mm.kind = MapperKind::Bitset;
            mm.var  = build_mapper_bool(st);
            break;
    }

    return mm;
}

inline Mapper build_mapper_from_matrix(const TileIndexer::State& st,
                                       Method method,
                                       int n,
                                       int tile_size)
{
    const int num_tiles = (tile_size > 0) ? ((n + tile_size - 1) / tile_size) : 0;
    return build_mapper(st, method, num_tiles);
}

} // namespace tilecounter
