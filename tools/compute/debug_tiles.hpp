/**
 * @file debug_tiles.hpp
 * @brief Helpers to export dense tile buffers for offline inspection.
 *
 * Dumps fast-mode or safe-mode tile buffers to text files for debugging
 * and validation, enabling comparison between different execution modes.
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

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>

#include "../tile/meta.hpp"  // brings in TileMetaCore + TiledMatrix definitions

namespace sTiles { namespace debug {

/**
 * @brief Export dense tile values to a text file.
 *
 * @param matrix               Pointer to the matrix descriptor whose tiles should be exported.
 * @param file_path            Destination file; parent directories are created on demand.
 * @param prefer_fast_layout   When both fast and safe buffers are present, choose fast buffers.
 * @param include_empty_tiles  If true, tiles without storage are annotated instead of skipped.
 *
 * @return true on success, false if the matrix/stream is invalid or no tiles are available.
 */
[[nodiscard]] inline bool export_dense_tiles(const TiledMatrix* matrix,
                                             const std::string& file_path,
                                             bool prefer_fast_layout = true,
                                             bool include_empty_tiles = false)
{
    if (!matrix) return false;

    const std::filesystem::path path(file_path);
    const std::filesystem::path parent = path.parent_path();
    std::error_code ec;
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
    }

    std::ofstream out(file_path);
    if (!out.is_open()) return false;

    out.setf(std::ios::scientific);
    out << std::setprecision(17);

    const bool fast_available = (matrix->denseTiles != nullptr) && (matrix->tileMetaCore != nullptr);
    const bool safe_available = (matrix->dense_tiles != nullptr);
    const bool use_fast = fast_available && (prefer_fast_layout || !safe_available);
    const bool use_safe = safe_available && !use_fast;
    if (!use_fast && !use_safe) return false;

    const int num_tiles = matrix->numActiveTiles;
    const int tile_size = matrix->tile_size;

    // Recover tile (row,col) coordinates for each dense index.
    std::vector<std::pair<int, int>> tile_coords(static_cast<std::size_t>(num_tiles), {-1, -1});
    if (fast_available) {
        for (int idx = 0; idx < num_tiles; ++idx) {
            const TileMetaCore& meta = matrix->tileMetaCore[idx];
            if (meta.width > 0 && meta.height > 0) {
                tile_coords[static_cast<std::size_t>(idx)] = {meta.row, meta.col};
            }
        }
    }
    if (matrix->tileIndexMapper) {
        const int tiles_per_dim = matrix->dimTiledMatrix;
        const int packed_limit = matrix->triangular_size;
        for (int col = 0; col < tiles_per_dim; ++col) {
            for (int row = 0; row <= col; ++row) {
                const int packed = row * (2 * tiles_per_dim - row - 1) / 2 + col;
                if (packed >= packed_limit) continue;
                const int dense_idx = matrix->tileIndexMapper[packed];
                if (dense_idx < 0 || dense_idx >= num_tiles) continue;
                auto& slot = tile_coords[static_cast<std::size_t>(dense_idx)];
                if (slot.first < 0 || slot.second < 0) slot = {row, col};
            }
        }
    }

    out << "# matrix_dim=" << matrix->dim
        << " tile_size=" << tile_size
        << " tiles_per_dim=" << matrix->dimTiledMatrix
        << " layout=" << (use_fast ? "fast" : "safe")
        << '\n';

    out << "# tile_index tile_row tile_col local_row local_col global_row global_col value\n";

    for (int idx = 0; idx < num_tiles; ++idx) {
        const auto [tile_row, tile_col] = tile_coords[static_cast<std::size_t>(idx)];

        int width = 0;
        int height = 0;
        const double* data = nullptr;

        if (use_fast) {
            const TileMetaCore& meta = matrix->tileMetaCore[idx];
            width = meta.width;
            height = meta.height;
            data = matrix->denseTiles[idx];
        } else {
            const DenseTileSafeMode& tile = matrix->dense_tiles[idx];
            width = tile.width;
            height = tile.height;
            data = tile.elements;
        }

        if (width <= 0 || height <= 0 || data == nullptr) {
            if (include_empty_tiles) {
                out << "# tile " << idx << " row " << tile_row
                    << " col " << tile_col << " width " << width
                    << " height " << height << " (no data)\n\n";
            }
            continue;
        }

        const int base_row = (tile_row >= 0) ? tile_row * tile_size : 0;
        const int base_col = (tile_col >= 0) ? tile_col * tile_size : 0;

        out << "# tile " << idx << " row " << tile_row
            << " col " << tile_col << " width " << width
            << " height " << height << '\n';

        if (use_fast) {
            const int ld = height;
            for (int local_col = 0; local_col < width; ++local_col) {
                for (int local_row = 0; local_row < height; ++local_row) {
                    const double value = data[static_cast<std::size_t>(local_col) * ld + local_row];
                    out << idx << ' ' << tile_row << ' ' << tile_col << ' '
                        << local_row << ' ' << local_col << ' '
                        << (base_row + local_row) << ' ' << (base_col + local_col) << ' '
                        << value << '\n';
                }
            }
        } else {
            const int ld = width;
            for (int local_row = 0; local_row < height; ++local_row) {
                for (int local_col = 0; local_col < width; ++local_col) {
                    const double value = data[static_cast<std::size_t>(local_row) * ld + local_col];
                    out << idx << ' ' << tile_row << ' ' << tile_col << ' '
                        << local_row << ' ' << local_col << ' '
                        << (base_row + local_row) << ' ' << (base_col + local_col) << ' '
                        << value << '\n';
                }
            }
        }

        out << '\n';
    }

    return true;
}

}} // namespace sTiles::debug
