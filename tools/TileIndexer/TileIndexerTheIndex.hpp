/**
 * @file    TileIndexerTheIndex.hpp
 * @brief   Unified helpers to compute the canonical packed index for (i,j) in
 *          the upper triangle across TileIndexer methods.
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

#include <cstddef>

namespace tilecounter {

// Canonical helper: packed upper-tri index for tile(i,j), i<=j
/*
Function: the_index_upper
Purpose:  Compute canonical packed index for tile (i,j) with i<=j.
*/
inline std::size_t the_index_upper(int i, int j, int N) {
    const int lo = (i <= j) ? i : j;
    const int hi = i ^ j ^ lo;
    return upper_tile_index(lo, hi, N);
}

// 1) CharMask
/*
Function: the_index_char
Purpose:  Packed index helper for CharMask method.
*/
inline std::size_t the_index_char(const TileIndexer::State& /*idx*/, int N, int i, int j) {
    return the_index_upper(i, j, N);
}

// 2) BoolMask
/*
Function: the_index_bool
Purpose:  Packed index helper for BoolMask method.
*/
inline std::size_t the_index_bool(const TileIndexer::State& /*idx*/, int N, int i, int j) {
    return the_index_upper(i, j, N);
}

// 3) BitsetMask
/*
Function: the_index_bitset
Purpose:  Packed index helper for BitsetMask method.
*/
inline std::size_t the_index_bitset(const TileIndexer::State& /*idx*/, int N, int i, int j) {
    return the_index_upper(i, j, N);
}

// 4) TiledBoolMask
/*
Function: the_index_tiled_bool
Purpose:  Packed index helper for TiledBoolMask method.
*/
inline std::size_t the_index_tiled_bool(const TileIndexer::State& /*idx*/, int N, int i, int j) {
    return the_index_upper(i, j, N);
}

// 5) TiledBitsetMask
/*
Function: the_index_tiled_bitset
Purpose:  Packed index helper for TiledBitsetMask method.
*/
inline std::size_t the_index_tiled_bitset(const TileIndexer::State& /*idx*/, int N, int i, int j) {
    return the_index_upper(i, j, N);
}

// 6) PagedMask
/*
Function: the_index_paged
Purpose:  Packed index helper for PagedMask method.
*/
inline std::size_t the_index_paged(const TileIndexer::State& /*idx*/, int N, int i, int j) {
    return the_index_upper(i, j, N);
}

// 7) HashSet
/*
Function: the_index_hashset
Purpose:  Packed index helper for HashSet method.
*/
inline std::size_t the_index_hashset(const TileIndexer::State& /*idx*/, int N, int i, int j) {
    return the_index_upper(i, j, N);
}

// 8) SortUnique
/*
Function: the_index_sortunique
Purpose:  Packed index helper for SortUnique method.
*/
inline std::size_t the_index_sortunique(const TileIndexer::State& /*idx*/, int N, int i, int j) {
    return the_index_upper(i, j, N);
}

// Dispatcher by Method
/*
Function: TheIndex
Purpose:  Dispatch helper to compute packed index for any Method.
*/
inline std::size_t TheIndex(const TileIndexer::State& idx, Method m, int N, int i, int j) {
    switch (m) {
        case Method::CharMask:        return the_index_char(idx, N, i, j);
        case Method::BoolMask:        return the_index_bool(idx, N, i, j);
        case Method::BitsetMask:      return the_index_bitset(idx, N, i, j);
        case Method::TiledBoolMask:   return the_index_tiled_bool(idx, N, i, j);
        case Method::TiledBitsetMask: return the_index_tiled_bitset(idx, N, i, j);
        case Method::PagedMask:       return the_index_paged(idx, N, i, j);
        case Method::HashSet:         return the_index_hashset(idx, N, i, j);
        case Method::SortUnique:      return the_index_sortunique(idx, N, i, j);
        default:                      return the_index_upper(i, j, N);
    }
}

} // namespace tilecounter
