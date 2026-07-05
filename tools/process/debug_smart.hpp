// debug_smart.hpp
// Lightweight comparison utilities between SmartTile (sparse) and a reconstructed
// dense view produced from (tile_index_lookup, element_offset_lookup) and x.

#ifndef STILES_PROCESS_DEBUG_SMART_HPP
#define STILES_PROCESS_DEBUG_SMART_HPP

#ifdef SMART_TILES

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <exception>
#include <iostream>
#include <algorithm>

#include "../common/stiles_structs.hpp"
#include "../common/stiles_logger.hpp"
#include "../tile/meta.hpp"  // TileMetaCore definition

namespace sTiles { namespace debug {

// Compare SmartTiles (facTiles) against the nonzero pattern of dense tiles.
// Preferred source is S->denseTiles after update; if not present, reconstruct
// a dense view from x using tile_index_lookup/element_offset_lookup and compare
// the nonzero patterns column-by-column for each tile. Prints a brief summary.
inline void compare_smart_vs_dense_tiles(int global_index,
                                         TiledMatrix **schemes,
                                         const double *x,
                                         double tol = 1e-9,
                                         int max_reports = 10,
                                         bool print_ok = true,
                                         bool stop_on_mismatch = false)
{
    if (!schemes || !x) {
        std::cout << "[SMART-CHECK] Null input (schemes or x)." << std::endl;
        return;
    }

    TiledMatrix* S = schemes[global_index];
    if (!S) {
        std::cout << "[SMART-CHECK] Null scheme at global_index=" << global_index << std::endl;
        return;
    }

    if (!S->facTiles) {
        std::cout << "[SMART-CHECK] facTiles not present; nothing to compare." << std::endl;
        return;
    }
    if (!S->tile_index_lookup || !S->element_offset_lookup) {
        std::cout << "[SMART-CHECK] Lookup arrays missing; cannot reconstruct dense view." << std::endl;
        return;
    }
    const int num_tiles = S->numActiveTiles;
    const int nnz       = S->original_nnz;
    if (num_tiles <= 0 || nnz <= 0) {
        std::cout << "[SMART-CHECK] Nothing to compare (num_tiles=" << num_tiles
                  << ", nnz=" << nnz << ")." << std::endl;
        return;
    }

    // Prepare tile extents. Prefer scheme->tileMetaCore; if missing, derive from mapper.
    const bool have_meta = (S->tileMetaCore != nullptr);
    std::vector<sTiles::TileMetaCore> meta_fallback;
    if (!have_meta) {
        std::cout << "[SMART-CHECK] tileMetaCore missing; deriving tile dimensions from mapper." << std::endl;
        const int tdim = S->dimTiledMatrix;
        const int remainder = (S->remainderTileSize > 0) ? S->remainderTileSize : S->tile_size;
        meta_fallback.resize((std::size_t)num_tiles);
        for (int j = 0; j < tdim; ++j) {
            const int w = (j == tdim - 1) ? remainder : S->tile_size;
            for (int i = 0; i <= j; ++i) {
                const int dense_idx = S->mapper.map_ij(i, j, tdim);
                if (dense_idx < 0 || dense_idx >= num_tiles) continue;
                const int h = (i == tdim - 1) ? remainder : S->tile_size;
                auto& m = meta_fallback[(std::size_t)dense_idx];
                m.index  = dense_idx;
                m.row    = i;
                m.col    = j;
                m.width  = w;
                m.height = h;
            }
        }
    }

    // Always reconstruct dense tiles from x using lookup tables to avoid relying
    // on whether S->denseTiles were populated in this code path.
    std::vector<std::vector<double>> dense_tiles_tmp;
    dense_tiles_tmp.resize((std::size_t)num_tiles);
    for (int t = 0; t < num_tiles; ++t) {
        const sTiles::TileMetaCore& meta = have_meta ? S->tileMetaCore[t] : meta_fallback[(std::size_t)t];
        const int h = (meta.height > 0) ? meta.height : S->tile_size;
        const int w = (meta.width  > 0) ? meta.width  : S->tile_size;
        const std::size_t elems = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
        dense_tiles_tmp[(std::size_t)t].assign(elems, 0.0);
    }
    for (int idx = 0; idx < nnz; ++idx) {
        const int tile = S->tile_index_lookup[idx];
        if (tile < 0 || tile >= num_tiles) continue;
        const int off = S->element_offset_lookup[idx];
        auto& buf = dense_tiles_tmp[(std::size_t)tile];
        if (off < 0 || static_cast<std::size_t>(off) >= buf.size()) continue;
        buf[(std::size_t)off] = x[idx];
    }

    // Compare SmartTile values vs reconstructed dense tiles
    int mismatched_tiles = 0;
    int tested_tiles = 0;
    int reported = 0;
    for (int t = 0; t < num_tiles; ++t) {
        const sTiles::SmartTile* smart = S->facTiles[t];
        if (!smart) {
            continue; // not allocated/active; skip
        }
        ++tested_tiles;

        const sTiles::TileMetaCore& meta = have_meta ? S->tileMetaCore[t] : meta_fallback[(std::size_t)t];
        const int h = (meta.height > 0) ? meta.height : S->tile_size;
        const int w = (meta.width  > 0) ? meta.width  : S->tile_size;
        const int ld = h;

        // Build nonzero row sets per column from dense buffer
        std::vector<std::vector<int>> dense_nz_rows;
        dense_nz_rows.resize((std::size_t)w);
        const double* dense_ptr = dense_tiles_tmp[(std::size_t)t].data();
        for (int col = 0; col < w; ++col) {
            auto& rows = dense_nz_rows[(std::size_t)col];
            for (int row = 0; row < h; ++row) {
                const double v = dense_ptr[(std::size_t)col * (std::size_t)ld + (std::size_t)row];
                if (std::fabs(v) > tol) {
                    rows.push_back(row);
                }
            }
        }

        // Build nonzero row sets per column from SmartTile CSC
        std::vector<std::vector<int>> smart_nz_rows;
        smart_nz_rows.resize((std::size_t)w);
        const auto* data = smart->getData();
        if (!data || !data->original_colptr) {
            if (reported < max_reports) {
                std::cout << "[SMART-CHECK] Missing original CSC arrays on SmartTile t=" << t << std::endl;
            }
            ++mismatched_tiles;
            ++reported;
            continue;
        }
        const int* cp = data->original_colptr;
        const int* ri = data->original_rowind; // may be null if nnz == 0
        for (int col = 0; col < w; ++col) {
            const int start = cp[col];
            const int end   = cp[col+1];
            auto& rows = smart_nz_rows[(std::size_t)col];
            for (int k = start; k < end; ++k) {
                rows.push_back(ri ? ri[k] : 0);
            }
            std::sort(rows.begin(), rows.end());
        }

        // Compare patterns. For diagonal tiles, SmartTile stores lower-triangular
        // indices locally, while the reconstructed dense buffer uses upper. Check
        // both orientations against each other without reorienting either tile.
        bool ok = true;
        int mismatch_col = -1;
        int mismatch_row = -1;
        bool mismatch_dense_to_smart = false;
        if (meta.row == meta.col) {
            for (int col = 0; col < w && ok; ++col) {
                const auto& dense_rows = dense_nz_rows[(std::size_t)col];
                for (int row : dense_rows) {
                    if (row < 0 || row >= w) { ok = false; break; }
                    const auto& smart_rows = smart_nz_rows[(std::size_t)row];
                    if (!std::binary_search(smart_rows.begin(), smart_rows.end(), col)) {
                        ok = false;
                        mismatch_col = col;
                        mismatch_row = row;
                        mismatch_dense_to_smart = true;
                        break;
                    }
                }
            }
            for (int col = 0; col < w && ok; ++col) {
                const auto& smart_rows = smart_nz_rows[(std::size_t)col];
                for (int row : smart_rows) {
                    if (row < 0 || row >= w) { ok = false; break; }
                    const auto& dense_rows = dense_nz_rows[(std::size_t)row];
                    if (!std::binary_search(dense_rows.begin(), dense_rows.end(), col)) {
                        ok = false;
                        mismatch_col = col;
                        mismatch_row = row;
                        mismatch_dense_to_smart = false;
                        break;
                    }
                }
            }
        } else {
            for (int col = 0; col < w && ok; ++col) {
                const auto& a = dense_nz_rows[(std::size_t)col];
                const auto& b = smart_nz_rows[(std::size_t)col];
                if (a.size() != b.size()) { ok = false; break; }
                for (std::size_t i = 0; i < a.size(); ++i) {
                    if (a[i] != b[i]) { ok = false; break; }
                }
            }
        }
        if (!ok) {
            ++mismatched_tiles;
            if (reported < max_reports) {
                std::cout << "[SMART-CHECK] Pattern mismatch at tile " << t
                          << " (row=" << meta.row << ", col=" << meta.col
                          << ", h=" << h << ", w=" << w << ")" << std::endl;
                if (meta.row == meta.col && mismatch_col >= 0 && mismatch_row >= 0) {
                    if (mismatch_dense_to_smart) {
                        std::cout << "    missing in SmartTile: dense upper entry at (row="
                                  << mismatch_row << ", col=" << mismatch_col
                                  << ") has no lower-triangular counterpart." << std::endl;
                    } else {
                        std::cout << "    missing in dense reconstruction: SmartTile lower entry at (row="
                                  << mismatch_row << ", col=" << mismatch_col
                                  << ") has no upper-triangular counterpart." << std::endl;
                    }
                }
                // Print first few differing columns
                int printed_cols = 0;
                for (int col = 0; col < w && printed_cols < 3; ++col) {
                    const auto& a = dense_nz_rows[(std::size_t)col];
                    const auto& b = smart_nz_rows[(std::size_t)col];
                    if (a != b) {
                        auto print_vec = [](const std::vector<int>& v){
                            std::cout << "[";
                            for (std::size_t i = 0; i < v.size(); ++i) {
                                if (i) std::cout << ",";
                                std::cout << v[i];
                            }
                            std::cout << "]";
                        };
                        std::cout << "    col " << col << ": dense rows="; print_vec(a);
                        std::cout << " vs smart rows="; print_vec(b); std::cout << std::endl;
                        ++printed_cols;
                    }
                }
                ++reported;
            }
        }
    }

    if (mismatched_tiles == 0) {
        if (print_ok) {
            std::cout << "[SMART-CHECK] OK: All " << tested_tiles
                      << " tile(s) match element-wise with dense reconstruction (tol="
                      << tol << ")" << std::endl;
        }
    } else {
        if (reported >= max_reports) {
            std::cout << "[SMART-CHECK] Additional tile mismatches suppressed (total mismatched tiles="
                      << mismatched_tiles << ")" << std::endl;
        }
        std::cout << "[SMART-CHECK] Summary: mismatched tiles = " << mismatched_tiles
                  << " out of tested = " << tested_tiles << std::endl;
        if (stop_on_mismatch) {
            std::cout << "[SMART-CHECK] Fatal: mismatches detected, aborting." << std::endl;
            std::abort();
        }
    }
}

// Verify that every diagonal SmartTile stores only lower-triangular entries in its CSC
// structure. Reports the first few violations and optionally aborts when a mismatch is
// encountered.
inline void check_smart_diagonal_tiles_lower(int global_index,
                                             TiledMatrix **schemes,
                                             int max_reports = 10,
                                             bool print_ok = true,
                                             bool stop_on_violation = false)
{
    if (!schemes) {
        std::cout << "[SMART-LOWER] Null input (schemes)." << std::endl;
        return;
    }

    TiledMatrix* S = schemes[global_index];
    if (!S) {
        std::cout << "[SMART-LOWER] Null scheme at global_index=" << global_index << std::endl;
        return;
    }

    if (!S->facTiles) {
        std::cout << "[SMART-LOWER] facTiles not present; nothing to inspect." << std::endl;
        return;
    }

    const int num_tiles = S->numActiveTiles;
    if (num_tiles <= 0) {
        std::cout << "[SMART-LOWER] Nothing to inspect (num_tiles=" << num_tiles << ")." << std::endl;
        return;
    }

    const bool have_meta = (S->tileMetaCore != nullptr);
    std::vector<sTiles::TileMetaCore> meta_fallback;
    if (!have_meta) {
        std::cout << "[SMART-LOWER] tileMetaCore missing; deriving tile dimensions from mapper." << std::endl;
        const int tdim = S->dimTiledMatrix;
        const int remainder = (S->remainderTileSize > 0) ? S->remainderTileSize : S->tile_size;
        meta_fallback.resize((std::size_t)num_tiles);
        for (int j = 0; j < tdim; ++j) {
            const int w = (j == tdim - 1) ? remainder : S->tile_size;
            for (int i = 0; i <= j; ++i) {
                const int dense_idx = S->mapper.map_ij(i, j, tdim);
                if (dense_idx < 0 || dense_idx >= num_tiles) continue;
                const int h = (i == tdim - 1) ? remainder : S->tile_size;
                auto& m = meta_fallback[(std::size_t)dense_idx];
                m.index  = dense_idx;
                m.row    = i;
                m.col    = j;
                m.width  = w;
                m.height = h;
            }
        }
    }

    int inspected_tiles = 0;
    int violations = 0;
    int reported = 0;

    for (int t = 0; t < num_tiles; ++t) {
        const sTiles::SmartTile* smart = S->facTiles[t];
        if (!smart) continue;

        const sTiles::TileMetaCore& meta = have_meta ? S->tileMetaCore[t] : meta_fallback[(std::size_t)t];
        if (meta.row != meta.col) continue; // only diagonal tiles

        ++inspected_tiles;
        const auto* data = smart->getData();
        if (!data) {
            if (reported < max_reports) {
                std::cout << "[SMART-LOWER] Tile " << t << " (row=" << meta.row
                          << ", col=" << meta.col << ") has null SmartTileData." << std::endl;
            }
            ++violations;
            ++reported;
            continue;
        }

        const int rows = data->rows;
        const int cols = data->cols;
        bool tile_ok = true;
        const char* failing_label = nullptr;
        int failing_row = -1;
        int failing_col = -1;

        if (rows != cols) {
            tile_ok = false;
            failing_label = "geometry";
            failing_row = -8;
            failing_col = -8;
        }

        auto validate_csc = [&](const char* label,
                                 const int* colptr,
                                 const int* rowind,
                                 int nnz) {
            if (!tile_ok) return;
            if (!colptr) {
                tile_ok = false;
                failing_label = label;
                failing_row = -2;
                failing_col = -2;
                return;
            }
            if (cols <= 0) {
                tile_ok = false;
                failing_label = label;
                failing_row = -3;
                failing_col = -3;
                return;
            }

            const int expected_entries = (nnz >= 0) ? nnz : 0;
            if (nnz > 0 && !rowind) {
                tile_ok = false;
                failing_label = label;
                failing_row = -4;
                failing_col = -4;
                return;
            }

            for (int col = 0; col < cols; ++col) {
                const int start = colptr[col];
                const int end   = colptr[col + 1];
                if (start > end) {
                    tile_ok = false;
                    failing_label = label;
                    failing_row = -5;
                    failing_col = col;
                    return;
                }
                if (start < 0 || end < 0) {
                    tile_ok = false;
                    failing_label = label;
                    failing_row = -6;
                    failing_col = col;
                    return;
                }
                if (expected_entries > 0 && end > expected_entries) {
                    tile_ok = false;
                    failing_label = label;
                    failing_row = -7;
                    failing_col = col;
                    return;
                }
                if (!rowind) continue; // nnz == 0 case

                for (int idx = start; idx < end; ++idx) {
                    const int row = rowind[idx];
                    if (row < 0 || row >= rows) {
                        tile_ok = false;
                        failing_label = label;
                        failing_row = row;
                        failing_col = col;
                        return;
                    }
                    if (row < col) {
                        tile_ok = false;
                        failing_label = label;
                        failing_row = row;
                        failing_col = col;
                        return;
                    }
                }
            }
        };

        validate_csc("original", data->original_colptr, data->original_rowind, data->original_nnz);
        validate_csc("factor",   data->colptr,         data->rowind,         data->nnz);

        if (!tile_ok) {
            ++violations;
            if (reported < max_reports) {
                std::cout << "[SMART-LOWER] Violation in tile " << t
                          << " (row=" << meta.row << ", col=" << meta.col
                          << ", rows=" << rows << ", cols=" << cols << ")";
                if (failing_label) {
                    std::cout << " [" << failing_label << "]";
                }
                if (failing_row >= 0 && failing_col >= 0) {
                    std::cout << " entry at (row=" << failing_row << ", col=" << failing_col
                              << ") is above diagonal.";
                } else if (failing_row == -2) {
                    std::cout << " missing column pointer array.";
                } else if (failing_row == -3) {
                    std::cout << " invalid tile dimensions.";
                } else if (failing_row == -4) {
                    std::cout << " missing row index array for non-empty tile.";
                } else if (failing_row == -5) {
                    std::cout << " column pointer is not monotonic at col=" << failing_col << ".";
                } else if (failing_row == -6) {
                    std::cout << " negative column pointer entry at col=" << failing_col << ".";
                } else if (failing_row == -7) {
                    std::cout << " column pointer exceeds nnz bounds at col=" << failing_col << ".";
                } else if (failing_row == -8) {
                    std::cout << " tile is not square.";
                }
                std::cout << std::endl;
                ++reported;
            }
        }
    }

    if (violations == 0) {
        if (print_ok) {
            std::cout << "[SMART-LOWER] OK: " << inspected_tiles
                      << " diagonal tile(s) verified as lower-triangular." << std::endl;
        }
    } else {
        if (reported >= max_reports) {
            std::cout << "[SMART-LOWER] Additional violations suppressed (total="
                      << violations << ")." << std::endl;
        }
        std::cout << "[SMART-LOWER] Summary: violations=" << violations
                  << " inspected=" << inspected_tiles << std::endl;
        if (stop_on_violation) {
            std::cout << "[SMART-LOWER] Fatal: lower-triangular check failed, aborting." << std::endl;
            std::abort();
        }
    }
}

}} // namespace sTiles::debug

#endif // SMART_TILES

#endif // STILES_PROCESS_DEBUG_SMART_HPP
