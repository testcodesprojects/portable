/// @file    preprocess_helpers.hpp
/// @brief   Preprocessing utilities to construct compact mappers for active
///          tiles and to bind isActive according to the chosen method.
///
/// @project sTiles (Sparse Tiles Library)
/// @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
/// @contact esmail.abdulfattah@kaust.edu.sa
/// @version 3.0.0
/// @date 1 1 2026
/// @license Proprietary
///
/// @note This file is part of the sTiles library, a proprietary software package.
///       Redistribution or modification without prior permission is prohibited.
///
/// Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
///
/// @license
/// This software is proprietary and confidential. Unauthorized copying, distribution, or modification
/// of this software, via any medium, is strictly prohibited. Permission is granted to use the software
/// in binary form for non-commercial purposes only, provided that this copyright notice and permission
/// notice are included in all copies or substantial portions of the software.
///
/// DISCLAIMER:
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
/// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
/// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY,
/// WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE
/// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <cstdio>
#include <new>

#include "../common/stiles_structs.hpp"
#include "../common/stiles_logger.hpp"
#include "../TileIndexer/TileIndexerMapper.hpp"
#include "../TileIndexer/TileIndexerMemoryUtils.hpp"

namespace sTiles {
namespace preprocess {

inline sTiles::StatusCode constructMapper(TiledMatrix **scheme) {
    /*
    Purpose: Build a stable, compact rank mapping for all active tiles in the
             upper triangle using the same ordering as safemode.
    Params:  scheme  Input/output scheme carrying TileIndexer state.
    Returns: StatusCode indicating success or failure.
    */
    const int num_tiles = (*scheme)->dimTiledMatrix;

    // Build a stable, safe-compatible compact ID ordering for active tiles
    // without using safe-mode structures. We derive the active set from the
    // fast-mode state (dense masks, bitsets, or sparse ids) and populate
    // state.ids in the SAME scan order used by safe mode: column-major over
    // the upper triangle: for (j=0..N-1) for (i=0..j).

    TileIndexer::State &st = (*scheme)->state;

    // Ensure ids exists and is empty before we fill it deterministically.
    st.ids.clear();

    const std::size_t tri = static_cast<std::size_t>(num_tiles) * (static_cast<std::size_t>(num_tiles) + 1ULL) / 2ULL;
    st.ids.reserve(tri); // upper bound; we will push only active positions

    auto is_active_u = [&](std::size_t u) -> bool {
        if (!st.active_bool.empty()) {
            return (u < st.active_bool.size()) && st.active_bool[u];
        }
        if (!st.active_char.empty()) {
            return (u < st.active_char.size()) && (st.active_char[u] != 0);
        }
        if (!st.bits.empty()) {
            const std::size_t w = u >> 6; const unsigned b = static_cast<unsigned>(u & 63ULL);
            return (w < st.bits.size()) && (((st.bits[w] >> b) & 1ULL) != 0ULL);
        }
        if (!st.S.empty()) {
            return st.S.find(u) != st.S.end();
        }
        return false;
    };

    if (!st.active_bool.empty() || !st.active_char.empty() || !st.bits.empty() || !st.S.empty()) {
        // Populate ids using safe-mode scan order (column-major upper triangle)
        for (int j = 0; j < num_tiles; ++j) {
            for (int i = 0; i <= j; ++i) {
                const std::size_t u = TileIndexer::upper_tile_index(i, j, num_tiles);
                if (is_active_u(u)) st.ids.push_back(u);
            }
        }
    } else if (!st.tiled_chunks.empty()) {
        // Expand tiled chunks into bool mask for enumeration
        st.active_bool.assign(tri, false);

        const int B = st.tiled_block_dim;
        const int blocks_per_dim = st.tiled_blocks_per_dim;
        if (B > 0 && blocks_per_dim > 0) {
            for (const auto& kv : st.tiled_chunks) {
                const std::size_t key = kv.first;
                const int block_row = static_cast<int>(key / blocks_per_dim);
                const int block_col = static_cast<int>(key % blocks_per_dim);
                const int row_base = block_row * B;
                const int col_base = block_col * B;
                const auto& words = kv.second;

                for (std::size_t word_idx = 0; word_idx < words.size(); ++word_idx) {
                    std::uint64_t word = words[word_idx];
                    if (!word) continue;

                    const unsigned base = static_cast<unsigned>(word_idx * 64);
                    for (unsigned bit = 0; bit < 64; ++bit) {
                        if (!((word >> bit) & 1ULL)) continue;
                        const unsigned local = base + bit;
                        const int local_row = static_cast<int>(local / static_cast<unsigned>(B));
                        const int local_col = static_cast<int>(local % static_cast<unsigned>(B));
                        const int i = row_base + local_row;
                        const int j = col_base + local_col;
                        if (i >= num_tiles || j >= num_tiles) continue;
                        const int lo = (i <= j) ? i : j;
                        const int hi = i ^ j ^ lo;
                        const std::size_t u = TileIndexer::upper_tile_index(lo, hi, num_tiles);
                        st.active_bool[u] = true;
                    }
                }
            }

            for (int j = 0; j < num_tiles; ++j) {
                for (int i = 0; i <= j; ++i) {
                    const std::size_t u = TileIndexer::upper_tile_index(i, j, num_tiles);
                    if (st.active_bool[u]) st.ids.push_back(u);
                }
            }
        }
    } else if (!st.paged.pages.empty()) {
        // Expand paged structure into bool mask
        st.active_bool.assign(tri, false);
        const std::size_t page_bits = st.paged.page_bits ? st.paged.page_bits : 0;
        if (page_bits > 0) {
            for (const auto& page_kv : st.paged.pages) {
                const std::size_t page = page_kv.first;
                const auto& words = page_kv.second;
                const std::size_t base_u = page * page_bits;
                for (std::size_t word_idx = 0; word_idx < words.size(); ++word_idx) {
                    std::uint64_t word = words[word_idx];
                    if (!word) continue;
                    for (unsigned bit = 0; bit < 64; ++bit) {
                        if (!((word >> bit) & 1ULL)) continue;
                        const std::size_t u = base_u + word_idx * 64 + bit;
                        if (u < tri) st.active_bool[u] = true;
                    }
                }
            }
            for (int j = 0; j < num_tiles; ++j) {
                for (int i = 0; i <= j; ++i) {
                    const std::size_t u = TileIndexer::upper_tile_index(i, j, num_tiles);
                    if (st.active_bool[u]) st.ids.push_back(u);
                }
            }
        }
    } else {
        // Fallback: if ids were already prepared by the chosen strategy, keep order
        if (!st.ids.empty()) {
            // no-op; the existing order is preserved
        }
    }

    (*scheme)->mapper = TileIndexer::build_mapper((*scheme)->state,
                                                  TileIndexer::Method::SortUnique,
                                                  num_tiles);
    if (!(*scheme)->mapper.valid()) {
        std::fprintf(stderr,
                     "ERROR: TileIndexer mapper is not available for dense tile allocation.\n");
        return sTiles::StatusCode::Failure;
    }

    if ((*scheme)->diagonal_mapper) {
        TileIndexerMemoryManager::deallocate((*scheme)->diagonal_mapper);
        (*scheme)->diagonal_mapper = nullptr;
    }

    int *diag_map = nullptr;
    try {
        diag_map = TileIndexerMemoryManager::allocate<int>(num_tiles);
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr,
                     "ERROR: Failed to allocate diagonal mapper (tiles=%d).\n",
                     num_tiles);
        return sTiles::StatusCode::OutOfResources;
    }

    for (int i = 0; i < num_tiles; ++i) {
        const int dense_id = (*scheme)->mapper.map_ij(i, i, num_tiles);
        if (dense_id < 0) {
            std::fprintf(stderr,
                         "ERROR: TileIndexer mapper does not contain diagonal tile (%d,%d).\n",
                         i, i);
            TileIndexerMemoryManager::deallocate(diag_map);
            return sTiles::StatusCode::Failure;
        }
        diag_map[i] = dense_id;
    }

    (*scheme)->diagonal_mapper = diag_map;

    return sTiles::StatusCode::Success;
}

inline sTiles::StatusCode bindActive(TiledMatrix **scheme) {
    /*
    Purpose: Bind the state.isActive function pointer to the correct checker
             based on the configured neighbor lookup method.
    Returns: StatusCode::Success when binding succeeds; Failure otherwise.
    */
    TileIndexer::State &state = (*scheme)->state;
    TileIndexer::Method method = (*scheme)->neighbor_lookup_method;

    tilecounter_utils::bind_is_active(state, method);
    if (!state.is_active) {
        sTiles::Logger::error("ERROR: Failed to bind TileIndexer isActive checker for method ",
                              TileIndexer::to_string(method));
        return sTiles::StatusCode::Failure;
    }

    sTiles::Logger::debug("│     • TileIndexer active-check bound (method=",
                          TileIndexer::to_string(method), ")");
    return sTiles::StatusCode::Success;
}

}  // namespace preprocess
}  // namespace sTiles

// Element-level symbolic fill-in (implementation in pruned_fillin.cpp)
//#include "../ordering/pruned_fillin.hpp"
