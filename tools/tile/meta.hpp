/**
 * @file MetaTile.hpp
 * @brief Metadata structures for dense and sparse tile representations.
 *
 * Defines core metadata structures including TileMetaCore, SparseTileMetaCore,
 * SemisparseTileMetaCore, and SparseTileMetaData for managing tile properties,
 * sparsity patterns, and CSC format construction.
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

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

#include "../common/stiles_structs.hpp"
#ifdef SMART_TILES
#endif

namespace sTiles {

// COO element for sparse tile construction (used by SparseTileMetaData)
#ifndef SMART_TILES
struct TempElementCOO {
    int row;
    int col;
    int index;

    bool operator<(const TempElementCOO& other) const {
        if (col != other.col) {
            return col < other.col;
        }
        return row < other.row;
    }
};
#endif

struct TileMetaCore {
    int32_t index{0};
    int32_t row{0};
    int32_t col{0};
    int32_t width{0};
    int32_t height{0};
};
struct SemisparseTileMetaCore {
    int32_t fa{-1};
    int32_t la{-1};
    int32_t upper_bw{0};
    int32_t sa{0}; //sum of active
    bool is_contiguous{false};  // true if aind values are consecutive (la - fa + 1 == sa)
    bool is_full_width{false};  // true if sa == tile width (all columns active)
    std::vector<std::int32_t> aind;
    std::vector<std::int32_t> acol;
    // Per-tile CSC from exact L fill-in
    std::vector<std::int32_t> csc_colptr;   // size = n_local_cols + 1
    std::vector<std::int32_t> csc_rowind;   // size = csc_nnz (local row indices)
    int64_t csc_nnz{0};
};
struct SparseTileMetaCore {
    int32_t upper_bw{0};  // 0 means non diagonal, >0 will be final bandwidth for diagonals
    std::vector<int32_t> colptr;          // size = width + 1
    std::vector<int32_t> rowind;          // size = nnz
    std::vector<int32_t> tmp_columns;     // size = width + 1
    int64_t nnz{0};
};
struct SparseTileMetaData {
    std::vector<int32_t> indices_sorted;      
    std::vector<TempElementCOO> coo_elements_;
    std::atomic<int> current_index_{0};

    SparseTileMetaData() = default;
    SparseTileMetaData(const SparseTileMetaData&) = delete;
    SparseTileMetaData& operator=(const SparseTileMetaData&) = delete;
    SparseTileMetaData(SparseTileMetaData&&) = delete;
    SparseTileMetaData& operator=(SparseTileMetaData&&) = delete;

    inline void appendElement(int r, int c, int idx) {
        int insertion_index = current_index_.fetch_add(1, std::memory_order_relaxed);
        if (insertion_index < static_cast<int>(coo_elements_.size())) {
            coo_elements_[static_cast<std::size_t>(insertion_index)] = TempElementCOO{r, c, idx};
        } else {
            throw std::out_of_range("Attempted to append more elements than the tile meta was sized for.");
        }
    }

    inline void finalizeConstructionSparse(SparseTileMetaCore& meta, int ncols) {
        if (!meta.colptr.empty()) {
            return;
        }

        if (ncols <= 0) {
            meta.nnz = 0;
            meta.colptr.assign(1, 0);
            meta.rowind.clear();
            indices_sorted.clear();
            coo_elements_.clear();
            coo_elements_.shrink_to_fit();
            return;
        }

        if (coo_elements_.empty()) {
            meta.nnz = 0;
            meta.colptr.assign(static_cast<std::size_t>(ncols) + 1, 0);
            meta.rowind.clear();
            indices_sorted.clear();
            return;
        }

        std::sort(coo_elements_.begin(), coo_elements_.end());

        const std::int64_t nnz_int = static_cast<std::int64_t>(coo_elements_.size());
        meta.nnz = nnz_int;

        meta.colptr.assign(static_cast<std::size_t>(ncols) + 1, 0);
        meta.rowind.resize(static_cast<std::size_t>(nnz_int));
        indices_sorted.resize(static_cast<std::size_t>(nnz_int));

        int current_col = 0;
        meta.colptr[0] = 0;

        for (std::int64_t k = 0; k < nnz_int; ++k) {
            const TempElementCOO& elem = coo_elements_[static_cast<std::size_t>(k)];
            meta.rowind[static_cast<std::size_t>(k)] = elem.row;
            indices_sorted[static_cast<std::size_t>(k)] = elem.index;

            while (current_col < elem.col) {
                ++current_col;
                meta.colptr[static_cast<std::size_t>(current_col)] = static_cast<int32_t>(k);
            }
        }

        while (current_col < ncols) {
            ++current_col;
            meta.colptr[static_cast<std::size_t>(current_col)] = static_cast<int32_t>(nnz_int);
        }

        if (meta.upper_bw == 1) {
            computeUpperKdLocal(meta, ncols);
        }

        coo_elements_.clear();
        coo_elements_.shrink_to_fit();
    }

    inline void computeUpperKdLocal(SparseTileMetaCore& meta, int ncols) const {
        if (meta.colptr.empty() || meta.nnz == 0 || ncols <= 0) {
            return;
        }

        int32_t kd = 0;

        for (int j = 0; j < ncols; ++j) {
            int col_start = meta.colptr[static_cast<std::size_t>(j)];
            int col_end = meta.colptr[static_cast<std::size_t>(j + 1)];

            for (int idx = col_start; idx < col_end; ++idx) {
                int row = meta.rowind[static_cast<std::size_t>(idx)];
                int bw = j - row;
                if (bw > kd) {
                    kd = bw;
                }
            }
        }

        meta.upper_bw = kd;
    }
};

#ifdef SPARSE_STILES
struct SparseTileCSC {
    int32_t nnz{0};
    std::vector<int32_t> colptr;       // size = ncols + 1
    std::vector<int32_t> rowind;       // size = nnz
    std::vector<double>  values;       // size = nnz
    std::vector<int32_t> active_cols;  // sorted indices of non-empty columns
};
#endif

struct SymbolicTileBitmaskCore {
    int32_t words_per_col{0};
    std::vector<std::uint64_t> bitmaps; // size = sa * words_per_col

    inline void reset() {
        words_per_col = 0;
        bitmaps.clear();
    }

    inline void init(int32_t sa, int32_t height) {
        reset();
        if (sa <= 0 || height <= 0) return;

        const std::size_t h = static_cast<std::size_t>(height);
        words_per_col = static_cast<int32_t>((h + 63u) / 64u);
        const std::size_t total_words = static_cast<std::size_t>(sa) * static_cast<std::size_t>(words_per_col);
        bitmaps.assign(total_words, std::uint64_t(0));
    }

    inline void set_bit(int32_t active_col, int32_t row) {
        const std::size_t w = static_cast<std::size_t>(row >> 6);
        const int bit = row & 63;
        const std::size_t idx = static_cast<std::size_t>(active_col) * static_cast<std::size_t>(words_per_col) + w;
        bitmaps[idx] |= (std::uint64_t(1) << bit);
    }

    inline bool test_bit(int32_t active_col, int32_t row) const {
        const std::size_t w = static_cast<std::size_t>(row >> 6);
        const int bit = row & 63;
        const std::size_t idx = static_cast<std::size_t>(active_col) * static_cast<std::size_t>(words_per_col) + w;
        return (bitmaps[idx] & (std::uint64_t(1) << bit)) != 0;
    }

    inline std::uint64_t *column_bits(int32_t active_col) {
        return bitmaps.data() + static_cast<std::size_t>(active_col) * static_cast<std::size_t>(words_per_col);
    }

    inline const std::uint64_t *column_bits(int32_t active_col) const {
        return bitmaps.data() + static_cast<std::size_t>(active_col) * static_cast<std::size_t>(words_per_col);
    }
};

using DenseTile = double*;




} // namespace sTiles


