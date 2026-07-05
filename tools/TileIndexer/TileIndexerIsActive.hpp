/**
 * @file    TileIndexerIsActive.hpp
 * @brief   Per-method isActive predicates that test whether a tile (i,j) is
 *          active in the current State representation.
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
#include <cstddef>
#include <cstdint>

namespace tilecounter {

// All functions assume any (i,j); will normalize to i<=j internally.

/*
Function: is_active_char
Purpose:  Check activity using the dense char mask.
*/
inline bool is_active_char(const TileIndexer::State& idx, int i, int j, int N) {
    if (i > j) std::swap(i, j);
    const std::size_t u = upper_tile_index(i, j, N);
    return (u < idx.active_char.size()) && (idx.active_char[u] != 0);
}

/*
Function: is_active_bool
Purpose:  Check activity using the dense bool mask.
*/
inline bool is_active_bool(const TileIndexer::State& idx, int i, int j, int N) {
    if (i > j) std::swap(i, j);
    const std::size_t u = upper_tile_index(i, j, N);
    return (u < idx.active_bool.size()) && idx.active_bool[u];
}

/*
Function: is_active_bitset
Purpose:  Check activity using the packed 64-bit bitset mask.
*/
inline bool is_active_bitset(const TileIndexer::State& idx, int i, int j, int N) {
    if (i > j) std::swap(i, j);
    const std::size_t u = upper_tile_index(i, j, N);
    const std::size_t w = u >> 6; const unsigned b = static_cast<unsigned>(u & 63ULL);
    return (w < idx.bits.size()) && (((idx.bits[w] >> b) & 1ULL) != 0ULL);
}

/*
Function: is_active_tiled_bool
Purpose:  Check activity within a sparse BxB tile block.
*/
inline bool is_active_tiled_bool(const TileIndexer::State& idx, int i, int j, int /*N*/) {
    if (i > j) std::swap(i, j);
    const int B = idx.tiled_block_dim, Bdim = idx.tiled_blocks_per_dim;
    if (B <= 0 || Bdim <= 0) return false;
    const int bi = i / B, bj = j / B;
    const int li = i - bi * B, lj = j - bj * B;
    const std::size_t key = static_cast<std::size_t>(bi) * static_cast<std::size_t>(Bdim) + static_cast<std::size_t>(bj);
    auto it = idx.tiled_chunks.find(key);
    if (it == idx.tiled_chunks.end()) return false;
    const std::size_t k = static_cast<std::size_t>(li) * static_cast<std::size_t>(B) + static_cast<std::size_t>(lj);
    const std::size_t w = k >> 6; const std::uint64_t m = 1ULL << (k & 63ULL);
    return (w < it->second.size()) && ((it->second[w] & m) != 0ULL);
}

/*
Function: is_active_tiled_bitset
Purpose:  Alias of tiled-bool checker (shared storage design).
*/
inline bool is_active_tiled_bitset(const TileIndexer::State& idx, int i, int j, int N) {
    // Same storage as tiled_bool in current design
    return is_active_tiled_bool(idx, i, j, N);
}

/*
Function: is_active_paged
Purpose:  Check activity in a paged bitset using page/word addressing.
*/
inline bool is_active_paged(const TileIndexer::State& idx, int i, int j, int N) {
    if (i > j) std::swap(i, j);
    const std::size_t u = upper_tile_index(i, j, N);
    const std::size_t PB = idx.paged.page_bits; if (!PB) return false;
    const std::size_t page = u / PB; const std::size_t bit = u % PB; const std::size_t w = bit >> 6;
    auto it = idx.paged.pages.find(page);
    if (it == idx.paged.pages.end()) return false;
    const std::uint64_t word = it->second[w]; const unsigned b = static_cast<unsigned>(bit & 63ULL);
    return ((word >> b) & 1ULL) != 0ULL;
}

/*
Function: is_active_hashset
Purpose:  Check activity using an unordered_set of packed IDs.
*/
inline bool is_active_hashset(const TileIndexer::State& idx, int i, int j, int N) {
    if (i > j) std::swap(i, j);
    const std::size_t u = upper_tile_index(i, j, N);
    return idx.S.find(u) != idx.S.end();
}

/*
Function: is_active_sortunique
Purpose:  Check activity using a sorted vector of packed IDs (binary search).
*/
inline bool is_active_sortunique(const TileIndexer::State& idx, int i, int j, int N) {
    if (i > j) std::swap(i, j);
    const std::size_t u = upper_tile_index(i, j, N);
    // ids is kept sorted & unique after counting; binary search is O(log A)
    return std::binary_search(idx.ids.begin(), idx.ids.end(), u);
}

} // namespace tilecounter
