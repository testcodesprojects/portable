/**
 * @file stiles_utils.hpp
 * @brief Lightweight helpers shared across the sTiles codebase.
 *
 * Provides header-only utility functions for matrix operations, debugging,
 * export capabilities, and common operations shared between C and C++ modules.
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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <set>
#include <string>
#include <vector>

#include "stiles_structs.hpp"
#include "../tile/meta.hpp"

#ifdef _OPENMP
  #include <omp.h>
#else
  inline int omp_get_max_threads() { return 1; }
  inline int omp_get_num_procs() { return 1; }
#endif

namespace sTiles { namespace Utils {

namespace detail {

enum class DumpLayout { Vertical, Horizontal };

inline DumpLayout current_dump_layout() {
    static DumpLayout layout = [] {
        const char* env = std::getenv("STILES_DUMP_LAYOUT");
        if (!env) {
            return DumpLayout::Vertical;
        }
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (value == "horizontal" || value == "h" || value == "wide") {
            return DumpLayout::Horizontal;
        }
        return DumpLayout::Vertical;
    }();
    return layout;
}

inline std::string format_entry(int row, double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "row " << std::setw(3) << row << " = " << value;
    return oss.str();
}

} // namespace detail

/**
 * @brief Simple container describing the slice of work assigned to a rank.
 */
struct TaskDistribution {
  int start_index; ///< Index of the first task assigned to the rank.
  int num_tasks;   ///< Number of tasks the rank is expected to process.
};

inline int imax(int a, int b) {
    return (a > b) ? a : b;
}

/**
 * @brief Determine how many cores should be used for a computation.
 *
 * The function consults the OpenMP runtime (if available) and clamps the result
 * to the number of cores requested by the caller.
 *
 * @param needed_cores Maximum number of cores the caller is willing to use.
 * @return Number of cores that should actually be used.
 */
inline int get_cores(int needed_cores) {
    int max_cores_sys = std::max(omp_get_max_threads(), omp_get_num_procs());
    if (max_cores_sys > needed_cores) {
        max_cores_sys = needed_cores;
    }
    return max_cores_sys;
}

inline int get_max_cores() {
    return std::max(omp_get_max_threads(), omp_get_num_procs());
}

inline int effective_threads(int problem_size, int min_work_per_thread = 16) {
    const int available = omp_get_max_threads();
    if (problem_size <= 0) return 1;
    const int useful = std::max(1, problem_size / min_work_per_thread);
    return std::min(available, useful);
}

inline std::size_t lapack_band_offset(int diag_cols, int row, int col) {
    return static_cast<std::size_t>(diag_cols - 1 - (col - row))
         + static_cast<std::size_t>(diag_cols) * static_cast<std::size_t>(col);
}

inline bool store_in_lapack_band(double* band_tile,
                                 int diag_cols,
                                 int row,
                                 int col,
                                 double value,
                                 bool accumulate = true) {
    if (!band_tile || diag_cols <= 0 || row < 0 || col < row) {
        return false;
    }
    const int band = col - row;
    if (band < 0 || band >= diag_cols) {
        return false;
    }
    const std::size_t offset = lapack_band_offset(diag_cols, row, col);
    if (accumulate) {
        band_tile[offset] += value;
    } else {
        band_tile[offset] = value;
    }
    return true;
}

inline void pack_dense_upper_to_lapack_band(const double* dense_tile,
                                            int ld_dense,
                                            int rows,
                                            int cols,
                                            double* band_tile,
                                            int diag_cols) {
    if (!dense_tile || !band_tile || ld_dense <= 0 || rows <= 0 || cols <= 0 || diag_cols <= 0) {
        return;
    }
    const std::size_t total = static_cast<std::size_t>(diag_cols)
                            * static_cast<std::size_t>(cols);
    std::fill(band_tile, band_tile + total, 0.0);
    for (int col = 0; col < cols; ++col) {
        const int max_row = std::min(rows - 1, col);
        for (int row = 0; row <= max_row; ++row) {
            store_in_lapack_band(band_tile, diag_cols, row, col,
                                 dense_tile[row + col * ld_dense], false);
        }
    }
}

/**
 * @brief Evenly distribute @p total_tasks among @p total_ranks.
 *
 * Lower ranks receive one extra task if the work cannot be split evenly, which
 * keeps the total number of tasks balanced.
 */
inline TaskDistribution calculateTaskDistribution(int rank, int total_ranks, int total_tasks) {
    int base_tasks = total_tasks / total_ranks;
    int extra_tasks = total_tasks % total_ranks;

    TaskDistribution distribution;
    distribution.start_index = rank * base_tasks + (rank < extra_tasks ? rank : extra_tasks);
    distribution.num_tasks = base_tasks + (rank < extra_tasks ? 1 : 0);

    return distribution;
}


}} // namespace sTiles::Utils

namespace sTiles {

inline void export_dense_tiled_matrix(const TiledMatrix* scheme,
                                      const std::string& filepath,
                                      int rank,
                                      bool filter_upper,
                                      bool export_acol = false) {
    if (!scheme || rank != 0) return;

    try {
        const std::filesystem::path path_obj(filepath);
        if (path_obj.has_parent_path()) {
            std::filesystem::create_directories(path_obj.parent_path());
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "ERROR: Could not create directory for " << filepath
                  << ": " << e.what() << std::endl;
        return;
    }

    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        std::cerr << "ERROR: Could not open " << filepath << " for writing." << std::endl;
        return;
    }

    const int tile_size = scheme->tile_size;

    const bool has_semisparse_meta = (scheme->semisparseTileMetaCore != nullptr);
    const bool export_semisparse = export_acol && has_semisparse_meta;
    const bool export_bitmaps = (scheme->symbolicTileBitmaskCore != nullptr);

    outfile << "# tile_size=" << tile_size
            << " filter_upper=" << (filter_upper ? 1 : 0)
            << " export_acol=" << (export_semisparse ? 1 : 0)
            << " export_bitmask=" << (export_bitmaps ? 1 : 0)
            << '\n';

    for (int t = 0; t < scheme->numActiveTiles; ++t) {
        const double* tile_data = (scheme->denseTiles) ? scheme->denseTiles[t] : nullptr;
        if (!tile_data) continue;

        const TileMetaCore& meta = scheme->tileMetaCore[t];
        const int tile_row_idx = meta.row;
        const int tile_col_idx = meta.col;

        const int h = (meta.height > 0) ? meta.height : tile_size;
        const int w = (meta.width > 0) ? meta.width : tile_size;
        const int ld = h;

        outfile << "# tile row=" << tile_row_idx
                << " col=" << tile_col_idx
                << " height=" << h
                << " width=" << w << '\n';

        for (int j = 0; j < w; ++j) {
            for (int i = 0; i < h; ++i) {
                const double val = tile_data[i + j * ld];
                if (val == 0.0) continue;

                const int global_row = tile_row_idx * tile_size + i;
                const int global_col = tile_col_idx * tile_size + j;
                if (filter_upper && global_row > global_col) continue;

                outfile << i << ' '
                        << j << ' '
                        << std::scientific << std::setprecision(16) << val
                        << '\n';
            }
        }

        const SemisparseTileMetaCore* semisparse_meta = has_semisparse_meta
                                                        ? &scheme->semisparseTileMetaCore[t]
                                                        : nullptr;
        const SparseTileMetaCore* sparse_meta = (scheme->sparseTileMetaCore)
                                                ? &scheme->sparseTileMetaCore[t]
                                                : nullptr;
        const bool is_diagonal_tile = (tile_row_idx == tile_col_idx);

        // ═══════════════════════════════════════════════════════════════
        // Export semisparse metadata (acol, fa, la, sa, bw)
        // ═══════════════════════════════════════════════════════════════
        if (export_semisparse && semisparse_meta) {
            const SemisparseTileMetaCore& semi = *semisparse_meta;
            
            // Count active columns from acol array
            int active_cols = 0;
            for (std::int32_t val : semi.acol) {
                if (val >= 0) ++active_cols;
            }
            const int inactive_cols = static_cast<int>(semi.acol.size()) - active_cols;
            
            outfile << "# acol tile=" << t
                    << " row=" << tile_row_idx
                    << " col=" << tile_col_idx
                    << " count=" << semi.acol.size()
                    << " active=" << active_cols
                    << " inactive=" << inactive_cols
                    << " bw=" << semi.upper_bw
                    << " fa=" << semi.fa
                    << " la=" << semi.la
                    << " sa=" << semi.sa
                    << '\n';
            
            // Write acol array values
            if (!semi.acol.empty()) {
                for (std::int32_t col : semi.acol) {
                    outfile << col << ' ';
                }
            }
            outfile << '\n';
        }

        // ═══════════════════════════════════════════════════════════════
        // Export bitmask data (row indices per active column)
        // ═══════════════════════════════════════════════════════════════
        if (export_bitmaps) {
            const SymbolicTileBitmaskCore& bitmap = scheme->symbolicTileBitmaskCore[t];
            const int words_per_col = bitmap.words_per_col;
            
            if (words_per_col > 0 && !bitmap.bitmaps.empty()) {
                const std::size_t words_per_col_sz = static_cast<std::size_t>(words_per_col);
                
                int active_columns = (semisparse_meta && semisparse_meta->sa > 0)
                                        ? semisparse_meta->sa
                                        : static_cast<int>(bitmap.bitmaps.size() / words_per_col_sz);
                
                if (active_columns > 0) {
                    const std::size_t required_words = static_cast<std::size_t>(active_columns) * words_per_col_sz;
                    if (required_words > bitmap.bitmaps.size()) {
                        active_columns = static_cast<int>(bitmap.bitmaps.size() / words_per_col_sz);
                    }
                }

                if (active_columns > 0) {
                    // Lambda to extract row indices from bitmask
                    auto gather_rows = [&](int active_idx, std::set<int> &rows_out) {
                        rows_out.clear();
                        if (active_idx < 0) return;
                        for (int word = 0; word < words_per_col; ++word) {
                            const std::size_t offset = static_cast<std::size_t>(active_idx) * words_per_col_sz
                                                     + static_cast<std::size_t>(word);
                            if (offset >= bitmap.bitmaps.size()) {
                                break;
                            }
                            std::uint64_t bits_word = bitmap.bitmaps[offset];
                            int base_row = word * 64;
                            while (bits_word) {
                                const int bit = __builtin_ctzll(bits_word);
                                const int row = base_row + bit;
                                if (row >= h) break;
                                rows_out.insert(row);
                                bits_word &= (bits_word - 1);
                            }
                        }
                    };

                    // For diagonal tiles, filter to upper triangular (row <= col)
                    std::vector<std::set<int>> diag_upper_rows;
                    if (is_diagonal_tile && semisparse_meta) {
                        diag_upper_rows.resize(w);
                        for (int col = 0; col < w; ++col) {
                            if (col >= static_cast<int>(semisparse_meta->acol.size())) continue;
                            const int active_idx = semisparse_meta->acol[static_cast<std::size_t>(col)];
                            if (active_idx < 0) continue;
                            std::set<int> rows_tmp;
                            gather_rows(active_idx, rows_tmp);
                            for (int row : rows_tmp) {
                                if (row <= col) {
                                    diag_upper_rows[col].insert(row);
                                }
                            }
                            if (diag_upper_rows[col].empty()) {
                                diag_upper_rows[col].insert(col);
                            }
                        }
                    }

                    outfile << "# bitmask tile=" << t
                            << " row=" << tile_row_idx
                            << " col=" << tile_col_idx
                            << " sa=" << active_columns
                            << " words_per_col=" << words_per_col
                            << '\n';

                    for (int active = 0; active < active_columns; ++active) {
                        const int column_idx = (semisparse_meta && active < static_cast<int>(semisparse_meta->aind.size()))
                                                ? semisparse_meta->aind[active]
                                                : active;
                        outfile << column_idx << ':';

                        std::set<int> rows_to_print;
                        if (!diag_upper_rows.empty() && column_idx >= 0 && column_idx < static_cast<int>(diag_upper_rows.size())) {
                            rows_to_print = diag_upper_rows[static_cast<std::size_t>(column_idx)];
                        } else {
                            gather_rows(active, rows_to_print);
                        }

                        if (rows_to_print.empty()) {
                            outfile << " -";
                        } else {
                            bool first = true;
                            for (int row : rows_to_print) {
                                outfile << (first ? ' ' : ',') << row;
                                first = false;
                            }
                        }
                        outfile << '\n';
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // Export sparse diagonal structure (CSC format) if no bitmask
        // ═══════════════════════════════════════════════════════════════
        if (!export_bitmaps && is_diagonal_tile && sparse_meta) {
            const auto &diag = *sparse_meta;
            const auto &colptr = diag.colptr;
            const auto &rowind = diag.rowind;
            const int width_cols = static_cast<int>(colptr.size()) - 1;
            
            if (width_cols > 0 && !colptr.empty()) {
                outfile << "# diag_structure tile=" << t
                        << " row=" << tile_row_idx
                        << " col=" << tile_col_idx
                        << " width=" << width_cols
                        << " nnz=" << diag.nnz
                        << " upper_bw=" << diag.upper_bw
                        << '\n';

                for (int col = 0; col < width_cols; ++col) {
                    outfile << col << ':';
                    const int start = colptr[static_cast<std::size_t>(col)];
                    const int end   = colptr[static_cast<std::size_t>(col + 1)];
                    if (start >= end || rowind.empty()) {
                        outfile << " -";
                    } else {
                        bool first = true;
                        for (int idx = start; idx < end; ++idx) {
                            int row_val = rowind[static_cast<std::size_t>(idx)];
                            if (row_val < col) {
                                outfile << (first ? ' ' : ',') << row_val;
                                first = false;
                            }
                        }
                        if (first) {
                            outfile << " -";
                        }
                    }
                    outfile << '\n';
                }
            }
        }

        outfile << '\n';
    }

    outfile.close();
    std::cout << "[sTiles] Exported matrix to " << filepath << std::endl;
}

} // namespace sTiles
namespace sTiles {
namespace Utils {

inline void dump_semisparse_tile_state(const char *label,
                                       int tile_index,
                                       const TileMetaCore &meta,
                                       const SemisparseTileMetaCore &semi,
                                       const SymbolicTileBitmaskCore &bits) {
    std::cout << "[SymbolicDump] " << (label ? label : "")
              << " tile=" << tile_index
              << " row=" << meta.row
              << " col=" << meta.col
              << " width=" << meta.width
              << " height=" << meta.height
              << " sa=" << semi.sa
              << " fa=" << semi.fa
              << " la=" << semi.la
              << " upper_bw=" << semi.upper_bw
              << '\n';

    std::cout << "  acol:";
    if (semi.acol.empty()) {
        std::cout << " <empty>";
    } else {
        for (std::size_t i = 0; i < semi.acol.size(); ++i) {
            std::cout << ' ' << semi.acol[i];
        }
    }
    std::cout << '\n';

    std::cout << "  aind:";
    if (semi.aind.empty()) {
        std::cout << " <empty>";
    } else {
        for (std::size_t i = 0; i < semi.aind.size(); ++i) {
            std::cout << ' ' << semi.aind[i];
        }
    }
    std::cout << '\n';

    const int words_per_col = bits.words_per_col;
    if (words_per_col > 0 && !bits.bitmaps.empty()) {
        const std::size_t words_per_col_sz = static_cast<std::size_t>(words_per_col);
        int active_columns = (semi.sa > 0) ? semi.sa
                                           : static_cast<int>(bits.bitmaps.size() / words_per_col_sz);
        if (active_columns > 0) {
            const std::size_t required_words = static_cast<std::size_t>(active_columns) * words_per_col_sz;
            if (required_words > bits.bitmaps.size()) {
                active_columns = static_cast<int>(bits.bitmaps.size() / words_per_col_sz);
            }
        }

        std::cout << "  bitmask:" << '\n';
        for (int active = 0; active < active_columns; ++active) {
            const int column_idx = (active < static_cast<int>(semi.aind.size())) ? semi.aind[active]
                                                                                : active;
            std::cout << "    col " << column_idx << ':';

            bool has_rows = false;
            for (int word = 0; word < words_per_col; ++word) {
                const std::size_t offset = static_cast<std::size_t>(active) * words_per_col_sz
                                         + static_cast<std::size_t>(word);
                if (offset >= bits.bitmaps.size()) {
                    break;
                }

                std::uint64_t mask = bits.bitmaps[offset];
                int bit_index = 0;
                while (mask) {
                    if (mask & std::uint64_t(1)) {
                        const int row = word * 64 + bit_index;
                        std::cout << (has_rows ? ',' : ' ') << row;
                        has_rows = true;
                    }
                    mask >>= 1;
                    ++bit_index;
                }
            }

            if (!has_rows) {
                std::cout << " -";
            }
            std::cout << '\n';
        }
    } else {
        std::cout << "  bitmask: <empty>" << '\n';
    }
}

} // namespace Utils
} // namespace sTiles

namespace sTiles {
namespace Utils {

inline void dump_dense_tile_values(const char *label,
                                   int tile_index,
                                   const TileMetaCore &meta,
                                   const double *tile,
                                   int rows,
                                   int cols,
                                   int ld,
                                   double eps = 1e-12) {
    std::cout << "[DenseDump] " << (label ? label : "")
              << " tile=" << tile_index
              << " row=" << meta.row
              << " col=" << meta.col
              << " height=" << rows
              << " width=" << cols
              << '\n';

    if (!tile) {
        std::cout << "  <null tile>" << std::endl;
        return;
    }

    bool any = false;
    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);
    std::cout << std::setprecision(6) << std::fixed;

    const bool is_diagonal = (meta.row == meta.col);
    const detail::DumpLayout layout = detail::current_dump_layout();

    for (int j = 0; j < cols; ++j) {
        if (layout == detail::DumpLayout::Horizontal) {
            std::vector<std::string> entries;
            for (int i = 0; i < rows; ++i) {
                if (is_diagonal && i > j) continue;
                const double val = tile[i + j * ld];
                if (std::fabs(val) > eps) {
                    entries.push_back(detail::format_entry(i, val));
                }
            }
            if (!entries.empty()) {
                any = true;
                std::cout << "  col " << std::setw(3) << j << " : ";
                for (std::size_t idx = 0; idx < entries.size(); ++idx) {
                    if (idx > 0) std::cout << " | ";
                    std::cout << entries[idx];
                }
                std::cout << '\n';
            }
            continue;
        }

        bool printed_column = false;
        for (int i = 0; i < rows; ++i) {
            if (is_diagonal && i > j) continue;
            const double val = tile[i + j * ld];
            if (std::fabs(val) > eps) {
                if (!printed_column) {
                    std::cout << "  col " << std::setw(3) << j << " |";
                    printed_column = true;
                    any = true;
                } else {
                    std::cout << "           |";
                }
                std::cout << " row " << std::setw(3) << i << " = " << val << '\n';
            }
        }
        if (printed_column) {
            std::cout << '\n';
        }
    }

    if (!any) {
        std::cout << "  <all zeros within eps=" << eps << ">" << std::endl;
    }

    std::cout.copyfmt(old_state);
}

inline void print_dense_tile_bitmap(const char *stage,
                                    int tile_index,
                                    const TileMetaCore *tile_meta,
                                    int fallback_tile_size,
                                    const double *data) {
    if (!data || fallback_tile_size <= 0) {
        return;
    }

    int rows = fallback_tile_size;
    int cols = fallback_tile_size;

    if (tile_meta) {
        const TileMetaCore &meta = tile_meta[tile_index];
        if (meta.height > 0) rows = meta.height;
        if (meta.width > 0) cols = meta.width;
    }

    if (rows <= 0 || cols <= 0) {
        return;
    }

    constexpr double kThreshold = 1e-12;
    const int ld_tile = rows;

    std::cout << "[" << stage << "] tile=" << tile_index
              << " structure (" << rows << "x" << cols << ")" << std::endl;

    for (int row = 0; row < rows; ++row) {
        std::cout << "  |";
        for (int col = 0; col < cols; ++col) {
            const double value = data[col * ld_tile + row];
            std::cout << (std::fabs(value) > kThreshold ? '1' : ' ');
        }
        std::cout << "|" << std::endl;
    }
}

inline void dump_chunked_tile_values(const char *label,
                                     int tile_index,
                                     const TileMetaCore &meta,
                                     const SemisparseTileMetaCore &semi,
                                     const double *chunk,
                                     int tile_size,
                                     bool lapack_layout,
                                     double eps = 1e-12) {
    std::cout << "[ChunkDump] " << (label ? label : "")
              << " tile=" << tile_index
              << " row=" << meta.row
              << " col=" << meta.col
              << " height=" << ((meta.height > 0) ? meta.height : tile_size)
              << " width=" << ((meta.width  > 0) ? meta.width  : tile_size)
              << " sa=" << semi.sa
              << '\n';

    if (!chunk) {
        std::cout << "  <null chunk>" << std::endl;
        return;
    }

    const int rows = (meta.height > 0) ? meta.height : tile_size;
    const int cols = (meta.width  > 0) ? meta.width  : tile_size;
    if (rows <= 0 || cols <= 0) {
        std::cout << "  <no active data>" << std::endl;
        return;
    }

    const bool is_diag_tile = (meta.row == meta.col);
    int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
    if ((lapack_layout || is_diag_tile) && diag_cols <= 0) {
        diag_cols = cols;
    }
    const int active_cols = lapack_layout ? cols : (is_diag_tile && diag_cols > 0 ? cols : semi.sa);

    if (active_cols <= 0) {
        std::cout << "  <no active data>" << std::endl;
        return;
    }

    bool any = false;
    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);
    std::cout << std::setprecision(6) << std::fixed;

    const int aind_size = static_cast<int>(semi.aind.size());
    const detail::DumpLayout layout = detail::current_dump_layout();

    if (lapack_layout && diag_cols > 0) {
        for (int global_col = 0; global_col < cols; ++global_col) {
            std::vector<std::string> entries;
            const int max_row = std::min(global_col, rows - 1);
            for (int row = 0; row <= max_row; ++row) {
                const int band = global_col - row;
                if (band < 0 || band >= diag_cols) continue;
                const int lapack_row = diag_cols - 1 - band;
                const double val = chunk[static_cast<std::size_t>(lapack_row)
                                         + static_cast<std::size_t>(diag_cols)
                                         * static_cast<std::size_t>(global_col)];
                if (std::fabs(val) > eps) {
                    entries.push_back(detail::format_entry(row, val));
                }
            }
            if (!entries.empty()) {
                any = true;
                if (layout == detail::DumpLayout::Horizontal) {
                    std::cout << "  col " << std::setw(3) << global_col << " : ";
                    for (std::size_t idx = 0; idx < entries.size(); ++idx) {
                        if (idx > 0) std::cout << " | ";
                        std::cout << entries[idx];
                    }
                    std::cout << '\n';
                } else {
                    for (std::size_t idx = 0; idx < entries.size(); ++idx) {
                        if (idx == 0) {
                            std::cout << "  col " << std::setw(3) << global_col << " |";
                        } else {
                            std::cout << "           |";
                        }
                        std::cout << ' ' << entries[idx] << '\n';
                    }
                    std::cout << '\n';
                }
            }
        }
    } else if (is_diag_tile && diag_cols > 0) {
        for (int global_col = 0; global_col < cols; ++global_col) {
            std::vector<std::string> entries;
            const int max_row = std::min(global_col, rows - 1);
            for (int row = 0; row <= max_row; ++row) {
                const int band = global_col - row;
                if (band >= diag_cols) continue;
                const double val = chunk[static_cast<std::size_t>(band) * rows + row];
                if (std::fabs(val) > eps) {
                    entries.push_back(detail::format_entry(row, val));
                }
            }
            if (!entries.empty()) {
                any = true;
                if (layout == detail::DumpLayout::Horizontal) {
                    std::cout << "  col " << std::setw(3) << global_col << " : ";
                    for (std::size_t idx = 0; idx < entries.size(); ++idx) {
                        if (idx > 0) std::cout << " | ";
                        std::cout << entries[idx];
                    }
                    std::cout << '\n';
                } else {
                    bool printed_column = false;
                    for (const auto& entry : entries) {
                        if (!printed_column) {
                            std::cout << "  col " << std::setw(3) << global_col << " |";
                            printed_column = true;
                        } else {
                            std::cout << "           |";
                        }
                        std::cout << ' ' << entry << '\n';
                    }
                    std::cout << '\n';
                }
            }
        }
    } else {
        for (int j = 0; j < active_cols; ++j) {
            const int local_col = (j < aind_size) ? semi.aind[j] : -1;
            if (local_col < 0 || local_col >= cols) {
                continue;
            }
            const double* col_ptr = chunk + static_cast<std::size_t>(j) * rows;

            if (layout == detail::DumpLayout::Horizontal) {
                std::vector<std::string> entries;
                for (int i = 0; i < rows; ++i) {
                    const double val = col_ptr[i];
                    if (std::fabs(val) > eps) {
                        entries.push_back(detail::format_entry(i, val));
                    }
                }
                if (!entries.empty()) {
                    any = true;
                    std::cout << "  col " << std::setw(3) << local_col << " : ";
                    for (std::size_t idx = 0; idx < entries.size(); ++idx) {
                        if (idx > 0) std::cout << " | ";
                        std::cout << entries[idx];
                    }
                    std::cout << '\n';
                }
                continue;
            }

            bool printed_column = false;
            for (int i = 0; i < rows; ++i) {
                const double val = col_ptr[i];
                if (std::fabs(val) > eps) {
                    if (!printed_column) {
                        std::cout << "  col " << std::setw(3) << local_col << " |";
                        printed_column = true;
                        any = true;
                    } else {
                        std::cout << "           |";
                    }
                    std::cout << " row " << std::setw(3) << i << " = " << val << '\n';
                }
            }
            if (printed_column) {
                std::cout << '\n';
            }
        }
    }

    if (!any) {
        std::cout << "  <all zeros within eps=" << eps << ">" << std::endl;
    }

    std::cout.copyfmt(old_state);
}

/**
 * @brief Print dense tile, optionally filtering out all-zero columns
 *
 * @param label Optional label for the output
 * @param A Pointer to the dense tile data
 * @param rows Number of rows in the tile
 * @param cols Number of columns in the tile
 * @param ld Leading dimension of the tile
 * @param max_rows Maximum number of rows to print (default 8)
 * @param max_cols Maximum number of columns to print (default 8)
 * @param precision Precision for floating point output (default 6)
 * @param zero_tol Tolerance for considering a value as zero (default 1e-15)
 */
inline void print_tile_filtered(const char* label, const double* A, int rows, int cols, int ld,
                                int max_rows = 8, int max_cols = 8, int precision = 6, double zero_tol = 1e-15) {
    if (!A) {
        std::cout << "[SMART-CHECK] " << (label ? label : "dense") << ": null pointer" << std::endl;
        return;
    }
    if (ld < rows) {
        std::cout << "[SMART-CHECK] " << (label ? label : "dense")
                  << ": invalid leading dimension ld=" << ld << " < rows=" << rows << std::endl;
        return;
    }

    // Identify non-zero columns
    std::vector<int> nonzero_cols;
    for (int j = 0; j < cols; ++j) {
        bool has_nonzero = false;
        for (int i = 0; i < rows; ++i) {
            if (std::fabs(A[i + j * ld]) > zero_tol) {
                has_nonzero = true;
                break;
            }
        }
        if (has_nonzero) {
            nonzero_cols.push_back(j);
        }
    }

    if (nonzero_cols.empty()) {
        std::cout << "[SMART-CHECK] " << (label ? label : "dense")
                  << " (" << rows << "x" << cols << ", ld=" << ld << "): All zeros\n";
        return;
    }

    const int pr = std::min(rows, max_rows);
    const int pc = std::min(static_cast<int>(nonzero_cols.size()), max_cols);

    std::cout.setf(std::ios::fixed);
    std::cout << "[SMART-CHECK] Dense tile " << (label ? label : "")
              << " (" << rows << "x" << cols << ", ld=" << ld << ") showing " << pr << " rows x "
              << nonzero_cols.size() << " non-zero cols";
    if (pc < static_cast<int>(nonzero_cols.size())) {
        std::cout << " (displaying first " << pc << ")";
    }
    std::cout << ":\n";

    std::streamsize oldp = std::cout.precision();
    std::cout.precision(precision);

    for (int i = 0; i < pr; ++i) {
        for (int jj = 0; jj < pc; ++jj) {
            const int j = nonzero_cols[jj];
            const double v = A[i + j * ld];
            std::cout << v;
            if (jj + 1 < pc) std::cout << ", ";
        }
        if (pc < static_cast<int>(nonzero_cols.size())) std::cout << ", ...";
        std::cout << std::endl;
    }
    if (pr < rows) {
        std::cout << "..." << std::endl;
    }
    std::cout.precision(oldp);
}

/**
 * @brief Statistics for comparing dense vs semisparse tile elements
 */
struct ExportDiffStats {
    int nnz_checked;         ///< Number of non-zero elements checked
    int count_bad;           ///< Number of elements that failed comparison
    int first_bad_idx;       ///< Index of first element that failed
    double first_dense;      ///< Dense value at first failure
    double first_semisparse; ///< Semisparse value at first failure
    double max_abs;          ///< Maximum absolute difference
    double max_rel;          ///< Maximum relative difference
};

/**
 * @brief Compare dense vs semisparse sparse elements serially
 *
 * @param S Pointer to the tiled matrix
 * @param atol Absolute tolerance for comparison
 * @param rtol Relative tolerance for comparison
 * @param max_print Maximum number of differences to print
 * @return Statistics about the comparison
 */
inline ExportDiffStats compare_dense_vs_semisparse_sparse_elements_serial(TiledMatrix* S, double atol, double rtol, int max_print) {
    ExportDiffStats st;
    st.nnz_checked = 0;
    st.count_bad = 0;
    st.first_bad_idx = -1;
    st.first_dense = 0.0;
    st.first_semisparse = 0.0;
    st.max_abs = 0.0;
    st.max_rel = 0.0;

    if (!S) return st;

    const int nnz = S->original_nnz;
    const bool *diag_map = S->diagonal_bmapper;

    if (!S->denseTiles || !S->tileMetaCore || !S->tile_index_lookup || !S->element_offset_lookup) {
        std::cerr << "[compare_dense_vs_semisparse] Missing dense requirements" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (!S->chunkedDenseTiles || !S->semisparseTileMetaCore || !S->withinTileRow || !S->withinTileCol) {
        std::cerr << "[compare_dense_vs_semisparse] Missing semisparse requirements" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    auto is_diagonal_tile = [&](int tile_idx) -> bool {
        const TileMetaCore& meta = S->tileMetaCore[tile_idx];
        return meta.row == meta.col;
    };

    auto uses_lapack_diagonal = [&](int tile_idx) -> bool {
        return diag_map && diag_map[tile_idx];
    };

    auto read_dense = [&](int idx, bool &ok) -> double {
        ok = false;
        const int tile = S->tile_index_lookup[idx];
        if (tile < 0 || tile >= S->numActiveTiles) return 0.0;
        double* A = S->denseTiles[tile];
        if (!A) return 0.0;
        const int off = S->element_offset_lookup[idx];
        ok = true;
        return A[off];
    };

    auto read_semisparse = [&](int idx, bool &ok) -> double {
        ok = false;
        const int tile = S->tile_index_lookup[idx];
        if (tile < 0 || tile >= S->numActiveTiles) return 0.0;
        double* chunk = S->chunkedDenseTiles[tile];
        if (!chunk) return 0.0;

        const SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[tile];
        const TileMetaCore& meta = S->tileMetaCore[tile];

        const int local_row = S->withinTileRow[idx];
        const int local_col = S->withinTileCol[idx];
        if (local_row < 0 || local_col < 0) return 0.0;

        const int h = (meta.height > 0) ? meta.height : S->tile_size;
        const int w = (meta.width  > 0) ? meta.width  : S->tile_size;
        if (h <= 0 || w <= 0) return 0.0;

        const bool lapack_diag = uses_lapack_diagonal(tile);
        const bool is_diag = is_diagonal_tile(tile);

        int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
        if ((lapack_diag || is_diag) && diag_cols <= 0) diag_cols = w;

        if (lapack_diag) {
            if (local_col >= w || local_row >= h) return 0.0;
            if (diag_cols <= 0) return 0.0;
            const int band = local_col - local_row;
            if (band < 0 || band >= diag_cols) return 0.0;
            const int lapack_row = diag_cols - 1 - band;
            const std::size_t off = static_cast<std::size_t>(lapack_row) + static_cast<std::size_t>(diag_cols) * static_cast<std::size_t>(local_col);
            ok = true;
            return chunk[off];
        }

        if (local_row >= h) return 0.0;

        if (is_diag && diag_cols > 0) {
            const int band = local_col - local_row;
            if (band < 0 || band >= diag_cols) return 0.0;
            const std::size_t off = static_cast<std::size_t>(local_row) + static_cast<std::size_t>(band) * static_cast<std::size_t>(h);
            ok = true;
            return chunk[off];
        }

        if (semi.acol.empty() || semi.sa <= 0) return 0.0;
        if (local_col >= static_cast<int>(semi.acol.size())) return 0.0;

        const int active_col = semi.acol[static_cast<std::size_t>(local_col)];
        if (active_col < 0 || active_col >= semi.sa) return 0.0;

        const std::size_t off = static_cast<std::size_t>(local_row) + static_cast<std::size_t>(active_col) * static_cast<std::size_t>(h);
        ok = true;
        return chunk[off];
    };

    int printed = 0;

    for (int idx = 0; idx < nnz; ++idx) {
        bool okd = false;
        bool oks = false;
        const double vd = read_dense(idx, okd);
        const double vs = read_semisparse(idx, oks);

        const double diff = std::fabs(vd - vs);
        const double denom = std::max(std::fabs(vd), std::fabs(vs));
        const double rel = (denom > 0.0) ? (diff / denom) : diff;

        st.max_abs = std::max(st.max_abs, diff);
        st.max_rel = std::max(st.max_rel, rel);

        const bool bad = (!okd || !oks || (diff > atol && rel > rtol));
        if (bad) {
            if (st.first_bad_idx < 0) { st.first_bad_idx = idx; st.first_dense = vd; st.first_semisparse = vs; }
            ++st.count_bad;
            if (printed < max_print) {
                std::cout << "[COMPARE] idx=" << idx << " dense=" << vd << " semisparse=" << vs << " abs=" << diff << " rel=" << rel << "\n";
                ++printed;
            }
        }

        ++st.nnz_checked;
    }

    return st;
}

} // namespace Utils
} // namespace sTiles
