/**
 * @file pdpotrf.cpp
 * @brief Parallel tiled Cholesky factorization (PDPOTRF) implementation.
 *
 * Implements the main parallel Cholesky factorization algorithm with support
 * for dense, semisparse, and sparse tile variants using OpenMP task parallelism.
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

#include <omp.h>
#include <cmath>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <functional>
#include <array>
#include <vector>
#include <set>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <chrono>

#include "../common/core_lapack.hpp"
// Per-thread scratch for SmartTile and dense conversions
#include "../memory/stile_threadWorkspace.hpp"

#include "../control/stiles_control.hpp"
#include "../compute/stiles_compute.hpp"
#include "../control/common.h" //remove

// Debug flag for OMP - define this to enable debug output
// #define STILES_DEBUG_OMP
#include "../tile/meta.hpp"

#include "../common/stiles_utils.hpp"
#include "../common/stiles_expiry.hpp"
#include "../tile/sparse_dense_tiling.hpp"
#include "debug_tiles.hpp"
#include "tile_compare.hpp"
#ifdef SMART_TILES
// SmartTile CHOL compute (symbolic + numeric)
#endif
#include "../symbolic/core_semisparse_kernels.hpp"
#include "../symbolic/core_sparse_kernels.hpp"
#ifdef SPARSE_STILES
#include "../sparse/sparse_kernels.hpp"
#endif

// Forward declaration of global control parameter accessor
extern "C" int* sTiles_get_params();
extern "C" const char* sTiles_get_dense_export_file(void);

// Mark rarely-taken helpers so the compiler outlines them into a cold text
// section, keeping the hot dense dispatch's i-cache footprint tight.
#if defined(__GNUC__) || defined(__clang__)
  #define STILES_COLD __attribute__((cold, noinline))
#else
  #define STILES_COLD
#endif

static inline void print_tasks(int rank,
                               int routine,
                               int m,
                               int k,
                               int n,
                               int index1,
                               int index2,
                               int index3) {
    static std::mutex task_print_mutex;

    auto tile_label = [](int row, int col) {
        std::ostringstream tmp;
        tmp << "T(" << row << "," << col << ")";
        return tmp.str();
    };

    std::ostringstream oss;
    switch (routine) {
        case 1:
            oss << "chol(" << tile_label(k, k) << ")";
            break;
        case 2:
            oss << tile_label(n, n) << " -= "
                << tile_label(n, k) << "^T * " << tile_label(n, k);
            break;
        case 3:
            oss << "solve(" << tile_label(k, k) << ", " << tile_label(k, m) << ")";
            break;
        case 4:
            oss << tile_label(n, m) << " -= "
                << tile_label(n, k) << "^T * " << tile_label(m, k);
            break;
        default:
            oss << "routine " << routine
                << " (m=" << m << ", k=" << k << ", n=" << n
                << ") idxs (" << index1 << "," << index2 << "," << index3 << ")";
            break;
    }

    std::lock_guard<std::mutex> guard(task_print_mutex);
    std::cout << "[task] core " << rank << " | " << oss.str() << '\n';
}

namespace {

bool write_dense_values_to_txt(const std::string& filepath,
                               const std::vector<double>& values) {
    if (filepath.empty() || values.empty()) {
        return false;
    }

    try {
        const std::filesystem::path path_obj(filepath);
        if (path_obj.has_parent_path()) {
            std::filesystem::create_directories(path_obj.parent_path());
        }
    } catch (const std::filesystem::filesystem_error& e) {
        sTiles::Logger::error("[dense-export] Failed to prepare path ", filepath,
                              ": ", e.what());
        return false;
    }

    std::ofstream outfile(filepath, std::ios::out | std::ios::trunc);
    if (!outfile.is_open()) {
        sTiles::Logger::error("[dense-export] Could not open ", filepath, " for writing.");
        return false;
    }

    outfile.setf(std::ios::scientific);
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);

    for (double value : values) {
        outfile << value << '\n';
    }

    return true;
}

/// Print a semisparse diagonal tile stored in LAPACK upper-banded format.
/// @param label   A caller-supplied tag (e.g. "BEFORE POTRF", "AFTER POTRF")
/// @param tile    Pointer to the banded data buffer
/// @param meta    TileMetaCore for this tile (height/width, grid position)
/// @param semi    SemisparseTileMetaCore (upper_bw, etc.)
/// @param index   Linear tile index in chunkedDenseTiles
static void print_semisparse_diag_tile(const char* label,
                                       const double* tile,
                                       const sTiles::TileMetaCore& meta,
                                       const sTiles::SemisparseTileMetaCore& semi,
                                       int index)
{
    if (!tile) {
        std::fprintf(stderr, "[SS-DIAG %s] tile_idx=%d  (null pointer)\n", label, index);
        return;
    }
    const int rows = meta.height;
    const int cols = meta.width;
    const int kd   = semi.upper_bw;          // number of super-diagonals
    const int ldab = kd + 1;                  // leading dim of banded storage

    std::fprintf(stderr,
        "[SS-DIAG %s] tile_idx=%d  grid=(%d,%d)  rows=%d  cols=%d  upper_bw=%d\n",
        label, index, meta.row, meta.col, rows, cols, kd);

    // Reconstruct the full upper triangle from banded format and print
    // row i, col j  (j >= i, j-i <= kd):  A(kd + i - j, j)  i.e. buf[(kd - (j-i)) + j*ldab]
    for (int i = 0; i < rows; ++i) {
        std::fprintf(stderr, "  row %3d: ", i);
        for (int j = 0; j < cols; ++j) {
            const int band = j - i;
            if (band < 0 || band > kd) {
                std::fprintf(stderr, " %12s", ".");         // lower triangle / out-of-band
            } else {
                const int brow = kd - band;
                const double val = tile[brow + j * ldab];
                std::fprintf(stderr, " %12.4f", val);
            }
        }
        std::fprintf(stderr, "\n");
    }
}

/// Print a semisparse off-diagonal tile stored in compressed active-column format.
/// @param label   A caller-supplied tag (e.g. "BEFORE TRSM", "AFTER GEMM")
/// @param tile    Pointer to the column-major active-columns buffer
/// @param meta    TileMetaCore for this tile (height/width, grid position)
/// @param semi    SemisparseTileMetaCore (sa, aind, acol)
/// @param index   Linear tile index in chunkedDenseTiles
static void print_semisparse_offdiag_tile(const char* label,
                                          const double* tile,
                                          const sTiles::TileMetaCore& meta,
                                          const sTiles::SemisparseTileMetaCore& semi,
                                          int index)
{
    if (!tile) {
        std::fprintf(stderr, "[SS-OFFDIAG %s] tile_idx=%d  (null pointer)\n", label, index);
        return;
    }
    const int rows = meta.height;
    const int sa   = semi.sa;                 // number of active (stored) columns
    const int aind_size = static_cast<int>(semi.aind.size());

    std::fprintf(stderr,
        "[SS-OFFDIAG %s] tile_idx=%d  grid=(%d,%d)  rows=%d  width=%d  sa=%d  fa=%d  la=%d\n",
        label, index, meta.row, meta.col, rows, meta.width, sa,
        semi.fa, semi.la);

    // Header: show which original columns are stored in each slot
    std::fprintf(stderr, "  %10s", "");
    for (int s = 0; s < sa; ++s) {
        const int orig_col = (s < aind_size) ? semi.aind[s] : -1;
        std::fprintf(stderr, " col%-4d(s%d)", orig_col, s);
    }
    std::fprintf(stderr, "\n");

    // Print rows
    for (int i = 0; i < rows; ++i) {
        std::fprintf(stderr, "  row %3d: ", i);
        for (int s = 0; s < sa; ++s) {
            const double val = tile[i + static_cast<std::size_t>(s) * rows];
            std::fprintf(stderr, " %12.4f", val);
        }
        std::fprintf(stderr, "\n");
    }
}

} // namespace


namespace sTiles{ namespace Process{

        //numeric

        // Direct Cholesky for dense variants 1 and 2 (no task pre-collection)
        // Variant 1: Single tile covering entire matrix
        // Variant 2: Scaled dense with upper triangular tiles
        void dpotrf_dense_variant_direct(TiledMatrix *tiledMatrix, stiles_context_t *stile, int variant) {
            const int rank = STILES_RANK;
            const int worldsize = STILES_SIZE;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles = tiledMatrix->dimTiledMatrix;

            const double zone = 1.0;
            const double mzone = -1.0;

            // Variant 1: Single tile - just one POTRF call
            if (variant == 1) {
                if (rank == 0 && tiledMatrix->denseTiles && tiledMatrix->denseTiles[0]) {
                    double* tile = tiledMatrix->denseTiles[0];
                    sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, N, tile, N);
                    if (status != sTiles::StatusCode::Success) {
                        std::fprintf(stderr, "sTiles error: matrix is not positive definite.\n");
                        ss_abort();
                    }
                }
                return;
            }

            // Variant 2: Scaled dense with all upper triangular tiles
            // Upper triangular tile index: upper_idx(i,j) = i * num_tiles - (i * (i - 1)) / 2 + (j - i)
            auto upper_idx = [num_tiles](int i, int j) -> int {
                return i * num_tiles - (i * (i - 1)) / 2 + (j - i);
            };

            // Calculate tile dimensions
            auto tile_dim = [N, tile_size, num_tiles](int t) -> int {
                return (t == num_tiles - 1) ? (N - t * tile_size) : tile_size;
            };

            ss_init(num_tiles, num_tiles, 0);

            // Iterate through k, m, n similar to STYLE_GREEN_TREE_PHASE_1
            int k = 0;
            int m = rank;
            while (m >= num_tiles) {
                k++;
                m = m - num_tiles + k;
            }
            int n = 0;

            while (k < num_tiles && m < num_tiles) {
                // Compute next iteration values
                int next_n = n;
                int next_m = m;
                int next_k = k;

                next_n++;
                if (next_n > next_k) {
                    next_m += worldsize;
                    while (next_m >= num_tiles && next_k < num_tiles) {
                        next_k++;
                        next_m = next_m - num_tiles + next_k;
                    }
                    next_n = 0;
                }

                // Get tile dimensions
                const int tempkn = tile_dim(k);
                const int tempmn = tile_dim(m);
                const int ldak = tile_dim(k);
                const int ldan = tile_dim(n);

                // For dense matrix, all tiles exist - no on_off_tiles check needed
                if (m == k) {
                    if (n == k) {
                        // Case 1: POTRF on diagonal tile (k, k)
                        const int idx_kk = upper_idx(k, k);
                        double* tile_out = tiledMatrix->denseTiles[idx_kk];
                        if (tile_out) {
                            sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
                            if (status != sTiles::StatusCode::Success) {
                                std::fprintf(stderr, "sTiles error: matrix is not positive definite (tile k=%d).\n", k);
                                ss_abort();
                            }
                        }
                        ss_cond_set(k, k, 1);
                    } else {
                        // Case 2: SYRK - update diagonal tile (k, k) with tile (n, k)
                        const int idx_nk = upper_idx(n, k);
                        const int idx_kk = upper_idx(k, k);

                        ss_cond_wait(k, n, 1);
                        double* tile_in = tiledMatrix->denseTiles[idx_nk];
                        double* tile_out = tiledMatrix->denseTiles[idx_kk];
                        if (tile_in && tile_out) {
                            sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone, tile_in, ldan, zone, tile_out, ldak);
                        }
                    }
                } else {
                    if (n == k) {
                        // Case 3: TRSM - solve for tile (k, m)
                        const int idx_km = upper_idx(k, m);
                        const int idx_kk = upper_idx(k, k);

                        ss_cond_wait(k, k, 1);
                        double* tile_rhs = tiledMatrix->denseTiles[idx_kk];
                        double* tile_out = tiledMatrix->denseTiles[idx_km];
                        if (tile_rhs && tile_out) {
                            sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit, tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
                        }
                        ss_cond_set(m, k, 1);
                    } else {
                        // Case 4: GEMM - update tile (k, m) with tiles (n, k) and (n, m)
                        const int idx_nk = upper_idx(n, k);
                        const int idx_nm = upper_idx(n, m);
                        const int idx_km = upper_idx(k, m);

                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);
                        double* tile_a = tiledMatrix->denseTiles[idx_nk];
                        double* tile_b = tiledMatrix->denseTiles[idx_nm];
                        double* tile_out = tiledMatrix->denseTiles[idx_km];
                        if (tile_out && tile_a && tile_b) {
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone, tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
                        }
                    }
                }

                if (ss_aborted()) break;

                // Move to next iteration
                n = next_n;
                m = next_m;
                k = next_k;
            }

            ss_finalize();
        }

        // #define STILES_USE_METACORE
        //numeric
        void dpotrf_expansion_from_chol_tasks_dense_use_metacore(TiledMatrix *tiledMatrix, stiles_context_t *stile) {
            const int rank = STILES_RANK;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

            //export_dense_tiled_matrix(tiledMatrix, "fillin/matrix_before_factorization.txt", rank, false, true);

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

#ifdef STILES_USE_METACORE
                // Use tile metadata for dimensions
                // Compute tile indices for diagonal tiles to get dimensions
                const int tile_kk = k * num_tiles_per_dim + k;
                const int tile_mm = m * num_tiles_per_dim + m;
                const int tile_nn = n * num_tiles_per_dim + n;

                const int height_k = tiledMatrix->tileMetaCore[tile_kk].height;
                const int height_m = tiledMatrix->tileMetaCore[tile_mm].height;
                const int width_k = tiledMatrix->tileMetaCore[tile_kk].width;
                const int width_n = tiledMatrix->tileMetaCore[tile_nn].width;
#else
                // Traditional computation based on tile_size
                const int height_k = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int height_m = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int width_k = BLKADD_CT(k);
                const int width_n = BLKADD_CT(n);
#endif

                const double zone = 1.0;
                const double mzone = -1.0;

                switch (myroutine) {
                    case 1: { // DPOTRF
                        double *tile_out = tiledMatrix->denseTiles[index1];
                        sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, height_k, tile_out, width_k);
                        if (status != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (tile k=%d).\n", k);
                            ss_abort(); break;
                        }
                        ss_cond_set(k, k, 1);
                        break;
                    }
                    case 2: { // DSYRK
                        ss_cond_wait(m, k, 1);
                        double *tile_in = tiledMatrix->denseTiles[index1];
                        double *tile_out = tiledMatrix->denseTiles[index2];
                        if (tile_in && tile_out) sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, height_k, tile_size, mzone, tile_in, width_n, zone, tile_out, width_k);
                        break;
                    }
                    case 3: { // DTRSM
                        ss_cond_wait(k, k, 1);
                        double *tile_rhs = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index1];
                        if (tile_rhs && tile_out) sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit, tile_size, height_m, zone, tile_rhs, width_k, tile_out, width_k);
                        ss_cond_set(m, k, 1);
                        break;
                    }
                    case 4: { // DGEMM
                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);
                        double *tile_a = tiledMatrix->denseTiles[index1];
                        double *tile_b = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index3];
                        if (tile_out && tile_a && tile_b) sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, height_m, tile_size, mzone, tile_a, width_n, tile_b, width_n, zone, tile_out, width_k);
                        break;
                    }
                    default:
                        break;
                }
                if (ss_aborted()) break;
            }
            ss_finalize();
            //export_dense_tiled_matrix(tiledMatrix, "fillin/matrix_after_factorization.txt", rank, false, true);
            #undef BLKADD_CT
        }

        

        /**
         * @brief Debug function: Compare dense vs semisparse task-by-task
         *
         * For each task, executes both the dense and semisparse operation,
         * then compares the output tile to detect discrepancies.
         */
        void dpotrf_expansion_compare_per_task(TiledMatrix *tiledMatrix, stiles_context_t *stile) {
            // This debug function runs per-rank but each rank uses its own workspace
            // and processes only its assigned tasks (same as the normal parallel versions)
            const int rank = STILES_RANK;

            // Mutex for thread-safe printing
            static std::mutex print_mutex;
            // Atomic flag to stop all cores when failure detected
            static std::atomic<bool> failure_detected{false};

            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int ntasks = static_cast<int>(tasks.size());
            const bool *diag_map = tiledMatrix->diagonal_bmapper;

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[compare_per_task] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            // Each rank uses its own workspace
            double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();

            // Each rank processes only its assigned tasks (same partitioning as parallel versions)
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : ntasks;

            {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << "[DEBUG] dpotrf_expansion_compare_per_task RANK=" << rank
                          << " STILES_SIZE=" << STILES_SIZE << " start=" << start << " end=" << end
                          << " ntasks=" << ntasks << "\n";
            }

            const double zone = 1.0;
            const double mzone = -1.0;

            int total_bad = 0;
            int task_with_diff = -1;

            int total_comparisons = 0;

            auto compare_tile_values = [&](int tile_idx, const char* op_name, int task_idx) -> bool {
                if (tile_idx < 0 || tile_idx >= tiledMatrix->numActiveTiles) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout << "[CORE " << rank << " TASK " << task_idx << " " << op_name << "] SKIP: invalid tile_idx=" << tile_idx << "\n";
                    return true;
                }

                double* dense = tiledMatrix->denseTiles[tile_idx];
                double* chunk = tiledMatrix->chunkedDenseTiles[tile_idx];
                if (!dense || !chunk) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout << "[CORE " << rank << " TASK " << task_idx << " " << op_name << "] SKIP: null pointer (dense="
                              << (dense ? "ok" : "NULL") << ", chunk=" << (chunk ? "ok" : "NULL") << ")\n";
                    return true;
                }

                const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[tile_idx];
                const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[tile_idx];
                const int h = (meta.height > 0) ? meta.height : tile_size;
                const int w = (meta.width > 0) ? meta.width : tile_size;
                const bool is_diag = (meta.row == meta.col);

                // For semisparse, diagonal tiles always use LAPACK upper banded format
                const int upper_bw = semi.upper_bw;
                const int ldab = upper_bw + 1;  // leading dimension of banded storage

                double max_abs = 0.0;
                int bad_count = 0;
                int num_compared = 0;

                // Use ostringstream to buffer output, then print atomically
                std::ostringstream oss;

                if (is_diag) {
                    // Diagonal tile: semisparse uses LAPACK upper banded format
                    // LAPACK stores A(i,j) at AB(kd+i-j, j) where kd = upper_bw
                    // So AB(upper_bw - band, j) where band = j - i
                    for (int j = 0; j < w; ++j) {
                        for (int i = 0; i <= j && i < h; ++i) {
                            const int band = j - i;
                            if (band > upper_bw) continue;  // outside bandwidth

                            const double vd = dense[i + j * BLKADD_CT(meta.col)];
                            const int lapack_row = upper_bw - band;
                            const double vs = chunk[lapack_row + ldab * j];

                            ++num_compared;
                            const double diff = std::fabs(vd - vs);
                            max_abs = std::max(max_abs, diff);
                            if (diff > 1e-10) {
                                ++bad_count;
                                if (bad_count <= 3) {
                                    oss << "[CORE " << rank << " TASK " << task_idx << " " << op_name << "] diag_tile=" << tile_idx
                                        << " (" << i << "," << j << ") band=" << band
                                        << ": dense=" << vd << " sparse=" << vs
                                        << " diff=" << diff << "\n";
                                }
                            }
                        }
                    }
                } else {
                    // Off-diagonal tile: semisparse stores only active columns
                    // aind[ac] = original column index for active column ac
                    // chunk layout: chunk[row + ac * height] for ac in [0, sa)
                    const int sa = semi.sa;
                    if (sa <= 0) {
                        std::lock_guard<std::mutex> lock(print_mutex);
                        std::cout << "[CORE " << rank << " TASK " << task_idx << " " << op_name << "] SKIP: off-diag tile=" << tile_idx << " sa=0\n";
                        return true;
                    }

                    const int aind_size = static_cast<int>(semi.aind.size());
                    for (int ac = 0; ac < sa && ac < aind_size; ++ac) {
                        const int orig_col = semi.aind[ac];
                        if (orig_col < 0 || orig_col >= w) continue;

                        for (int i = 0; i < h; ++i) {
                            const double vd = dense[i + orig_col * BLKADD_CT(meta.col)];
                            const double vs = chunk[i + ac * h];

                            ++num_compared;
                            const double diff = std::fabs(vd - vs);
                            max_abs = std::max(max_abs, diff);
                            if (diff > 1e-10) {
                                ++bad_count;
                                if (bad_count <= 3) {
                                    oss << "[CORE " << rank << " TASK " << task_idx << " " << op_name << "] tile=" << tile_idx
                                        << " (" << i << "," << orig_col << ") ac=" << ac
                                        << ": dense=" << vd << " sparse=" << vs
                                        << " diff=" << diff << "\n";
                                }
                            }
                        }
                    }
                }

                ++total_comparisons;
                if (bad_count > 0) {
                    oss << "[CORE " << rank << " TASK " << task_idx << " " << op_name << "] tile=" << tile_idx
                        << " FAIL: compared=" << num_compared << " bad=" << bad_count << " max_abs=" << max_abs << "\n";

                    // Print the tiles for debugging
                    const int print_max = 12;  // limit output size
                    const int ph = std::min(h, print_max);
                    const int pw = std::min(w, print_max);

                    oss << "--- DENSE tile " << tile_idx << " (row=" << meta.row << ", col=" << meta.col
                        << ", h=" << h << ", w=" << w << ") ---\n";
                    for (int ii = 0; ii < ph; ++ii) {
                        for (int jj = 0; jj < pw; ++jj) {
                            oss << std::setw(12) << std::setprecision(5) << dense[ii + jj * BLKADD_CT(meta.col)] << " ";
                        }
                        if (pw < w) oss << "...";
                        oss << "\n";
                    }
                    if (ph < h) oss << "...\n";

                    if (is_diag) {
                        oss << "--- SEMISPARSE BANDED tile " << tile_idx << " (upper_bw=" << upper_bw
                            << ", ldab=" << ldab << ") ---\n";
                        oss << "    LAPACK banded format (row 0 = superdiagonal " << upper_bw << ", row " << upper_bw << " = main diag):\n";
                        for (int band_row = 0; band_row < ldab && band_row < print_max; ++band_row) {
                            oss << "    row " << band_row << ": ";
                            for (int jj = 0; jj < pw; ++jj) {
                                oss << std::setw(12) << std::setprecision(5) << chunk[band_row + ldab * jj] << " ";
                            }
                            if (pw < w) oss << "...";
                            oss << "\n";
                        }
                        // Also print as reconstructed upper triangular for easier comparison
                        oss << "    Reconstructed upper triangular:\n";
                        for (int ii = 0; ii < ph; ++ii) {
                            for (int jj = 0; jj < pw; ++jj) {
                                const int band = jj - ii;
                                if (band >= 0 && band <= upper_bw) {
                                    const int lapack_row = upper_bw - band;
                                    oss << std::setw(12) << std::setprecision(5) << chunk[lapack_row + ldab * jj] << " ";
                                } else {
                                    oss << std::setw(12) << "." << " ";
                                }
                            }
                            if (pw < w) oss << "...";
                            oss << "\n";
                        }
                        if (ph < h) oss << "...\n";
                    } else {
                        const int sa = semi.sa;
                        const int aind_size = static_cast<int>(semi.aind.size());
                        oss << "--- SEMISPARSE ACTIVE-COL tile " << tile_idx << " (sa=" << sa << ") ---\n";
                        oss << "    Active columns (aind): ";
                        for (int ac = 0; ac < std::min(sa, print_max) && ac < aind_size; ++ac) {
                            oss << semi.aind[ac] << " ";
                        }
                        if (sa > print_max) oss << "...";
                        oss << "\n";
                        oss << "    Chunk data (rows x active_cols):\n";
                        for (int ii = 0; ii < ph; ++ii) {
                            for (int ac = 0; ac < std::min(sa, print_max); ++ac) {
                                oss << std::setw(12) << std::setprecision(5) << chunk[ii + ac * h] << " ";
                            }
                            if (sa > print_max) oss << "...";
                            oss << "\n";
                        }
                        if (ph < h) oss << "...\n";
                    }
                    oss << "--- END tile " << tile_idx << " ---\n\n";

                    // Print atomically
                    {
                        std::lock_guard<std::mutex> lock(print_mutex);
                        std::cout << oss.str();
                    }
                    return false;
                }
                oss << "[CORE " << rank << " TASK " << task_idx << " " << op_name << "] tile=" << tile_idx
                    << " OK: compared=" << num_compared << " max_abs=" << max_abs << "\n";
                // Print atomically
                {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout << oss.str();
                }
                return true;
            };

            // Helper to print tile contents (only when N < 500)
            auto print_tile = [&](const char* label, int tile_idx, int task_idx, const char* op_name) {
                if (failure_detected.load()) return;  // Stop printing if failure detected
                if (N >= 500) return;
                if (tile_idx < 0 || tile_idx >= tiledMatrix->numActiveTiles) return;

                double* dense = tiledMatrix->denseTiles[tile_idx];
                double* chunk = tiledMatrix->chunkedDenseTiles[tile_idx];
                const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[tile_idx];
                const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[tile_idx];
                const int h = (meta.height > 0) ? meta.height : tile_size;
                const int w = (meta.width > 0) ? meta.width : tile_size;
                const bool is_diag = (meta.row == meta.col);
                const int ld = BLKADD_CT(meta.col);

                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << "[CORE " << rank << " " << label << "] task=" << task_idx << " " << op_name
                          << " tile=" << tile_idx << " (row=" << meta.row << ", col=" << meta.col
                          << ", h=" << h << ", w=" << w << ")\n";

                // Print dense tile (compact: max 8x8)
                if (dense) {
                    std::cout << "  DENSE (" << h << "x" << w << "):\n";
                    const int ph = std::min(h, 8);
                    const int pw = std::min(w, 8);
                    for (int i = 0; i < ph; ++i) {
                        std::cout << "    ";
                        for (int j = 0; j < pw; ++j) {
                            std::cout << std::setw(12) << std::setprecision(4) << dense[i + j * ld] << " ";
                        }
                        if (pw < w) std::cout << "...";
                        std::cout << "\n";
                    }
                    if (ph < h) std::cout << "    ...\n";
                }

                // Print semisparse tile
                if (chunk) {
                    if (is_diag) {
                        const int kd = semi.upper_bw;
                        const int ldab = kd + 1;
                        std::cout << "  SEMISPARSE BANDED (kd=" << kd << ", ldab=" << ldab << "):\n";
                        const int prows = std::min(ldab, 8);
                        const int pcols = std::min(w, 8);
                        for (int r = 0; r < prows; ++r) {
                            std::cout << "    ";
                            for (int c = 0; c < pcols; ++c) {
                                std::cout << std::setw(12) << std::setprecision(4) << chunk[r + c * ldab] << " ";
                            }
                            if (pcols < w) std::cout << "...";
                            std::cout << "\n";
                        }
                        if (prows < ldab) std::cout << "    ...\n";
                    } else {
                        const int sa = semi.sa;
                        std::cout << "  SEMISPARSE ACOL (sa=" << sa << ") aind: ";
                        for (int i = 0; i < std::min(sa, 10); ++i) {
                            std::cout << semi.aind[i] << " ";
                        }
                        if (sa > 10) std::cout << "...";
                        std::cout << "\n";
                        if (sa > 0) {
                            const int prows = std::min(h, 8);
                            const int pcols = std::min(sa, 8);
                            for (int r = 0; r < prows; ++r) {
                                std::cout << "    ";
                                for (int ac = 0; ac < pcols; ++ac) {
                                    std::cout << std::setw(12) << std::setprecision(4) << chunk[r + ac * h] << " ";
                                }
                                if (pcols < sa) std::cout << "...";
                                std::cout << "\n";
                            }
                            if (prows < h) std::cout << "    ...\n";
                        }
                    }
                }
            };

            // Initialize dependency tracking for parallel execution
            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);

            for (int idx = start; idx < end; ++idx) {
                // Stop processing if another core detected a failure
                if (failure_detected.load()) break;

                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADD_CT(k);
                const int ldan = BLKADD_CT(n);

                const char* op_names[] = {"", "DPOTRF", "DSYRK", "DTRSM", "DGEMM"};
                const char* op_name = (myroutine >= 1 && myroutine <= 4) ? op_names[myroutine] : "UNKNOWN";

                switch (myroutine) {
                    case 1: { // DPOTRF - dense
                        // Print BEFORE
                        print_tile("BEFORE", index1, idx, op_name);

                        double *tile_out = tiledMatrix->denseTiles[index1];
                        sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);

                        // DPOTRF - semisparse
                        double *ssstile_out = tiledMatrix->chunkedDenseTiles[index1];
                        if (ssstile_out) {
                            const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                            const int kd = semi.upper_bw;
                            const int rows = meta.height;
                            sTiles::core_dpotrf_upper_banded(rows, kd, ssstile_out);
                        }

                        // Print AFTER
                        print_tile("AFTER", index1, idx, op_name);

                        if (!compare_tile_values(index1, op_name, idx)) {
                            if (task_with_diff < 0) task_with_diff = idx;
                            ++total_bad;
                            failure_detected.store(true);
                            ss_abort(); break;
                        }
                        // Signal DPOTRF completion for dependent tasks
                        {
                            std::lock_guard<std::mutex> lock(print_mutex);
                            std::cout << "[CORE " << rank << " TASK " << idx << " DPOTRF] Setting ss_cond_set(" << k << "," << k << ",1)\n";
                        }
                        ss_cond_set(k, k, 1);
                        break;
                    }
                    case 2: { // DSYRK - dense
                        // Wait for DTRSM on tile (k,n) to complete
                        ss_cond_wait(k, n, 1);

                        // Print BEFORE (input tile and output tile)
                        print_tile("BEFORE_IN", index1, idx, op_name);
                        print_tile("BEFORE_OUT", index2, idx, op_name);

                        double *tile_in = tiledMatrix->denseTiles[index1];
                        double *tile_out = tiledMatrix->denseTiles[index2];
                        if (tile_in && tile_out) {
                            sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone, tile_in, ldan, zone, tile_out, ldak);
                        }

                        // DSYRK - semisparse
                        double *ssstile_in = tiledMatrix->chunkedDenseTiles[index1];
                        double *ssstile_out = tiledMatrix->chunkedDenseTiles[index2];
                        if (ssstile_in && ssstile_out) {
                            const sTiles::SemisparseTileMetaCore& semi_in = tiledMatrix->semisparseTileMetaCore[index1];
                            const int active_cols = semi_in.sa;
                            if (active_cols > 0) {
                                const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index2];
                                const sTiles::TileMetaCore& meta_in = tiledMatrix->tileMetaCore[index1];
                                const int rows_in = (meta_in.height > 0) ? meta_in.height : tile_size;

                                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, active_cols, rows_in, mzone, ssstile_in, rows_in, 0.0, tmp_tile, active_cols);

                                const int aind_size = static_cast<int>(semi_in.aind.size());
                                const int cols_out = (tiledMatrix->tileMetaCore[index2].width > 0) ? tiledMatrix->tileMetaCore[index2].width : tile_size;
                                const int diag_cols = (semi_out.upper_bw >= 0) ? (semi_out.upper_bw + 1) : cols_out;

                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = (j < aind_size) ? semi_in.aind[j] : -1;
                                    if (col_idx < 0 || col_idx >= cols_out) continue;
                                    for (int i = 0; i <= j; ++i) {
                                        const int row_idx = (i < aind_size) ? semi_in.aind[i] : -1;
                                        if (row_idx < 0 || row_idx > col_idx) continue;
                                        const int band = col_idx - row_idx;
                                        if (band >= diag_cols) continue;
                                        const int lapack_row = diag_cols - 1 - band;
                                        const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                                                + static_cast<std::size_t>(diag_cols)
                                                                * static_cast<std::size_t>(col_idx);
                                        ssstile_out[offset] += tmp_tile[i + j * active_cols];
                                    }
                                }
                            }
                        }

                        // Print AFTER
                        print_tile("AFTER_OUT", index2, idx, op_name);

                        if (!compare_tile_values(index2, op_name, idx)) {
                            if (task_with_diff < 0) task_with_diff = idx;
                            ++total_bad;
                            failure_detected.store(true);
                            ss_abort(); break;
                        }
                        break;
                    }
                    case 3: { // DTRSM - dense
                        // Wait for DPOTRF on diagonal tile (k,k) to complete
                        ss_cond_wait(k, k, 1);

                        // Print BEFORE (input/output tile and diagonal tile)
                        print_tile("BEFORE_RHS", index1, idx, op_name);
                        print_tile("BEFORE_DIAG", index2, idx, op_name);

                        double *tile_rhs = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index1];
                        if (tile_rhs && tile_out) {
                            sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit, tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
                        }

                        // DTRSM - semisparse
                        double* ss_rhs = tiledMatrix->chunkedDenseTiles[index1];
                        double* ss_diag = tiledMatrix->chunkedDenseTiles[index2];
                        if (ss_rhs && ss_diag) {
                            const sTiles::SemisparseTileMetaCore& semi_rhs = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_diag = tiledMatrix->semisparseTileMetaCore[index2];
                            const sTiles::TileMetaCore& meta_rhs = tiledMatrix->tileMetaCore[index1];
                            const sTiles::TileMetaCore& meta_diag = tiledMatrix->tileMetaCore[index2];
                            const int active_cols = semi_rhs.sa;
                            const int rows_rhs = (meta_rhs.height > 0) ? meta_rhs.height : tile_size;
                            const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tile_size;
                            const int diag_bw = (semi_diag.upper_bw >= 0) ? semi_diag.upper_bw : 0;

                            if (active_cols > 0) {
                                const int kd = diag_bw;
                                const int ldab = diag_bw + 1;
                                LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', diag_dim, kd, active_cols, ss_diag, ldab, ss_rhs, rows_rhs);
                            }
                        }

                        // Print AFTER
                        print_tile("AFTER_RHS", index1, idx, op_name);

                        if (!compare_tile_values(index1, op_name, idx)) {
                            if (task_with_diff < 0) task_with_diff = idx;
                            ++total_bad;
                            failure_detected.store(true);
                            ss_abort(); break;
                        }
                        // Signal DTRSM completion for dependent tasks
                        ss_cond_set(m, k, 1);
                        break;
                    }
                    case 4: { // DGEMM - dense
                        // Wait for DTRSM on tiles (k,n) and (m,n) to complete
                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);

                        // Print BEFORE (A, B, and C tiles)
                        print_tile("BEFORE_A", index1, idx, op_name);
                        print_tile("BEFORE_B", index2, idx, op_name);
                        print_tile("BEFORE_C", index3, idx, op_name);

                        double *tile_a = tiledMatrix->denseTiles[index1];
                        double *tile_b = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index3];
                        if (tile_out && tile_a && tile_b) {
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone, tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
                        }

                        // DGEMM - semisparse
                        double* chunk_a = tiledMatrix->chunkedDenseTiles[index1];
                        double* chunk_b = tiledMatrix->chunkedDenseTiles[index2];
                        double* chunk_out = tiledMatrix->chunkedDenseTiles[index3];
                        if (chunk_a && chunk_b && chunk_out) {
                            const sTiles::SemisparseTileMetaCore& semi_a = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_b = tiledMatrix->semisparseTileMetaCore[index2];
                            const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index3];
                            const sTiles::TileMetaCore& meta_out = tiledMatrix->tileMetaCore[index3];

                            const int cols_a = semi_a.sa;
                            const int cols_b = semi_b.sa;
                            const int rows_a = tiledMatrix->tileMetaCore[index1].height;
                            const int rows_b = tiledMatrix->tileMetaCore[index2].height;
                            const int rows_out = meta_out.height;

                            if (cols_a > 0 && cols_b > 0 && rows_a > 0) {
                                const int ld_tmp = cols_a;
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, cols_a, cols_b, rows_a, mzone, chunk_a, rows_a, chunk_b, rows_b, 0.0, tmp_tile, ld_tmp);

                                const int cols_out = (meta_out.width > 0) ? meta_out.width : tile_size;
                                const int actual_rows_out = (rows_out > 0) ? rows_out : tile_size;
                                const bool lapack_diag = diag_map && diag_map[index3];
                                const bool is_diag = (meta_out.row == meta_out.col);

                                int diag_cols = (semi_out.upper_bw >= 0) ? (semi_out.upper_bw + 1) : 0;
                                if ((lapack_diag || is_diag) && diag_cols <= 0) diag_cols = cols_out;

                                const int aind_a_size = static_cast<int>(semi_a.aind.size());
                                const int aind_b_size = static_cast<int>(semi_b.aind.size());

                                for (int jj = 0; jj < cols_b; ++jj) {
                                    const int col_idx = (jj < aind_b_size) ? semi_b.aind[jj] : -1;
                                    if (col_idx < 0 || col_idx >= cols_out) continue;
                                    for (int ii = 0; ii < cols_a; ++ii) {
                                        const int row_idx = (ii < aind_a_size) ? semi_a.aind[ii] : -1;
                                        if (row_idx < 0 || row_idx >= actual_rows_out) continue;

                                        if (lapack_diag && diag_cols > 0) {
                                            const int band = col_idx - row_idx;
                                            if (band >= 0 && band < diag_cols) {
                                                const int lapack_row = diag_cols - 1 - band;
                                                chunk_out[lapack_row + diag_cols * col_idx] += tmp_tile[ii + jj * ld_tmp];
                                            }
                                        } else if (is_diag && diag_cols > 0 && row_idx <= col_idx) {
                                            const int band = col_idx - row_idx;
                                            if (band < diag_cols) {
                                                chunk_out[row_idx + band * actual_rows_out] += tmp_tile[ii + jj * ld_tmp];
                                            }
                                        } else if (!semi_out.acol.empty()) {
                                            if (col_idx < static_cast<int>(semi_out.acol.size())) {
                                                const int active_col = semi_out.acol[col_idx];
                                                if (active_col >= 0 && active_col < semi_out.sa) {
                                                    chunk_out[row_idx + active_col * actual_rows_out] += tmp_tile[ii + jj * ld_tmp];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Print AFTER
                        print_tile("AFTER_C", index3, idx, op_name);

                        if (!compare_tile_values(index3, op_name, idx)) {
                            // Debug: print active columns for small matrices
                            if (N < 500) {
                                std::lock_guard<std::mutex> lock(print_mutex);
                                const sTiles::SemisparseTileMetaCore& semi_a = tiledMatrix->semisparseTileMetaCore[index1];
                                const sTiles::SemisparseTileMetaCore& semi_b = tiledMatrix->semisparseTileMetaCore[index2];
                                const sTiles::TileMetaCore& meta_a = tiledMatrix->tileMetaCore[index1];
                                const sTiles::TileMetaCore& meta_b = tiledMatrix->tileMetaCore[index2];

                                std::cout << "[CORE " << rank << " DEBUG DGEMM FAIL] task=" << idx << " A=tile" << index1
                                          << " B=tile" << index2 << " C=tile" << index3 << "\n";

                                // Print semi_a.aind
                                std::cout << "  semi_a.aind (sa=" << semi_a.sa << "): ";
                                for (int i = 0; i < std::min(static_cast<int>(semi_a.aind.size()), 20); ++i) {
                                    std::cout << semi_a.aind[i] << " ";
                                }
                                if (semi_a.aind.size() > 20) std::cout << "...";
                                std::cout << "\n";

                                // Print semi_b.aind
                                std::cout << "  semi_b.aind (sa=" << semi_b.sa << "): ";
                                for (int i = 0; i < std::min(static_cast<int>(semi_b.aind.size()), 20); ++i) {
                                    std::cout << semi_b.aind[i] << " ";
                                }
                                if (semi_b.aind.size() > 20) std::cout << "...";
                                std::cout << "\n";

                                // Analyze dense tile A: find columns with non-zero values
                                double* dense_a = tiledMatrix->denseTiles[index1];
                                const int rows_a = meta_a.height > 0 ? meta_a.height : tile_size;
                                const int cols_a_dense = meta_a.width > 0 ? meta_a.width : tile_size;
                                std::cout << "  Dense tile A (index" << index1 << ", " << rows_a << "x" << cols_a_dense << ") non-zero cols: ";
                                for (int c = 0; c < cols_a_dense; ++c) {
                                    double col_max = 0.0;
                                    for (int r = 0; r < rows_a; ++r) {
                                        col_max = std::max(col_max, std::fabs(dense_a[r + c * ldan]));
                                    }
                                    if (col_max > 1e-15) {
                                        std::cout << c << "(max=" << col_max << ") ";
                                    }
                                }
                                std::cout << "\n";

                                // Analyze dense tile B: find columns with non-zero values
                                double* dense_b = tiledMatrix->denseTiles[index2];
                                const int rows_b = meta_b.height > 0 ? meta_b.height : tile_size;
                                const int cols_b_dense = meta_b.width > 0 ? meta_b.width : tile_size;
                                std::cout << "  Dense tile B (index" << index2 << ", " << rows_b << "x" << cols_b_dense << ") non-zero cols: ";
                                for (int c = 0; c < cols_b_dense; ++c) {
                                    double col_max = 0.0;
                                    for (int r = 0; r < rows_b; ++r) {
                                        col_max = std::max(col_max, std::fabs(dense_b[r + c * ldan]));
                                    }
                                    if (col_max > 1e-15) {
                                        std::cout << c << "(max=" << col_max << ") ";
                                    }
                                }
                                std::cout << "\n";

                                // Find columns in dense A that are NOT in semi_a.aind
                                std::cout << "  MISSING in semi_a.aind: ";
                                std::set<int> aind_set(semi_a.aind.begin(), semi_a.aind.end());
                                for (int c = 0; c < cols_a_dense; ++c) {
                                    double col_max = 0.0;
                                    for (int r = 0; r < rows_a; ++r) {
                                        col_max = std::max(col_max, std::fabs(dense_a[r + c * ldan]));
                                    }
                                    if (col_max > 1e-15 && aind_set.find(c) == aind_set.end()) {
                                        std::cout << c << "(max=" << col_max << ") ";
                                    }
                                }
                                std::cout << "\n";
                            }

                            if (task_with_diff < 0) task_with_diff = idx;
                            ++total_bad;
                            failure_detected.store(true);
                            ss_abort(); break;
                        }
                        break;
                    }
                    default:
                        break;
                }
            }

            // Finalize dependency tracking
            ss_finalize();

            {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << "[COMPARE_PER_TASK] rank=" << rank << " start=" << start << " end=" << end
                          << " total_tasks=" << ntasks
                          << " tile_comparisons=" << total_comparisons
                          << " tasks_with_diff=" << total_bad;
                if (task_with_diff >= 0) {
                    std::cout << " first_diff_task=" << task_with_diff;
                }
                std::cout << "\n";
            }

            #undef BLKADD_CT
        }

        void dpotrf_expansion_from_chol_tasks_semisparse(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int rank = STILES_RANK;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());
            const bool *diag_map = tiledMatrix->diagonal_bmapper;

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[semisparse] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADD_CT(k);
                const int ldan = BLKADD_CT(n);

                const double zone = 1.0;
                const double mzone = -1.0;
                
                switch (myroutine) {
                    case 1: { // DPOTRF

                        double *ssstile_out = tiledMatrix->chunkedDenseTiles[index1];

                        if (ssstile_out) {
                            const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                            const int kd = semi.upper_bw;
                            const int rows = meta.height;

                            sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(rows, kd, ssstile_out);
                            if (ssstatus != sTiles::StatusCode::Success) {
                                std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                                ss_abort(); break;
                            }
                        }
                        ss_cond_set(k, k, 1);
                        break;

                    }
                    case 2: { // DSYRK

                        ss_cond_wait(k, n, 1);

                        double *ssstile_in = tiledMatrix->chunkedDenseTiles[index1];
                        double *ssstile_out = tiledMatrix->chunkedDenseTiles[index2];

                        if (ssstile_in && ssstile_out) {

                            const sTiles::SemisparseTileMetaCore& semi_in = tiledMatrix->semisparseTileMetaCore[index1];
                            const int active_cols = semi_in.sa;

                            if (active_cols > 0) {

                                const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index2];
                                const sTiles::TileMetaCore& meta_in = tiledMatrix->tileMetaCore[index1];
                                const sTiles::TileMetaCore& meta_out = tiledMatrix->tileMetaCore[index2];
                                const bool lapack_diag_tile = diag_map && diag_map[index2];
                                // Use actual tile height, not nominal tile_size (fixes buffer overrun for boundary tiles)
                                const int rows_in = (meta_in.height > 0) ? meta_in.height : tile_size;

                                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, active_cols, rows_in, mzone, ssstile_in, rows_in, 0.0, tmp_tile, active_cols);

                                const int aind_size = static_cast<int>(semi_in.aind.size());
                                const int cols_out = (tiledMatrix->tileMetaCore[index2].width > 0) ? tiledMatrix->tileMetaCore[index2].width : tile_size;
                                const int rows_out = meta_out.height;
                                const int ld_tmp = active_cols;

                                // DSYRK output is always a diagonal tile in LAPACK banded format (for dpbtrf/dtbtrs)
                                const int diag_cols = (semi_out.upper_bw >= 0) ? (semi_out.upper_bw + 1) : cols_out;

                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = (j < aind_size) ? semi_in.aind[j] : -1;
                                    if (col_idx < 0 || col_idx >= cols_out) {
                                        continue;
                                    }
                                    for (int i = 0; i <= j; ++i) {
                                        const int row_idx = (i < aind_size) ? semi_in.aind[i] : -1;
                                        if (row_idx < 0 || row_idx > col_idx) {
                                            continue;
                                        }
                                        const int band = col_idx - row_idx;
                                        if (band >= diag_cols) {
                                            const double dropped = tmp_tile[i + j * ld_tmp];
                                            if (std::fabs(dropped) > 1e-15) {
                                                std::fprintf(stderr, "sTiles warning: DSYRK bandwidth overflow in tile k=%d: band=%d >= diag_cols=%d, dropped=%.6e\n",
                                                             k, band, diag_cols, dropped);
                                            }
                                            continue;
                                        }
                                        const int lapack_row = diag_cols - 1 - band;
                                        const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                                                + static_cast<std::size_t>(diag_cols)
                                                                * static_cast<std::size_t>(col_idx);
                                        ssstile_out[offset] += tmp_tile[i + j * ld_tmp];
                                    }
                                }
                            }
                        }

                        break;
                    }
                    case 3: { // DTRSM
                        ss_cond_wait(k, k, 1);

                        double* ss_rhs  = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index1] : nullptr;
                        double* ss_diag = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index2] : nullptr;
                        if (ss_rhs && ss_diag) {

                            const sTiles::SemisparseTileMetaCore& semi_rhs = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_diag = tiledMatrix->semisparseTileMetaCore[index2];
                            const sTiles::TileMetaCore& meta_rhs = tiledMatrix->tileMetaCore[index1];
                            const sTiles::TileMetaCore& meta_diag = tiledMatrix->tileMetaCore[index2];
                            const int active_cols = semi_rhs.sa;
                            const int rows_rhs = (meta_rhs.height > 0) ? meta_rhs.height : tile_size;
                            const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tile_size;
                            const int diag_bw = (semi_diag.upper_bw >= 0) ? semi_diag.upper_bw : 0;

                            if (active_cols > 0) {
                                const int kd = diag_bw;
                                const int ldab = diag_bw + 1;
                                LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', diag_dim, kd, active_cols, ss_diag, ldab, ss_rhs, rows_rhs);

                            }
                        }


                        ss_cond_set(m, k, 1);
                        break;
                    }
                    case 4: { // DGEMM
                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);

                        double* chunk_a   = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index1] : nullptr;
                        double* chunk_b   = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index2] : nullptr;
                        double* chunk_out = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index3] : nullptr;

                        if (chunk_a && chunk_b && chunk_out) {

                            const sTiles::SemisparseTileMetaCore& semi_a = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_b = tiledMatrix->semisparseTileMetaCore[index2];
                            const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index3];
                            const sTiles::TileMetaCore& meta_out = tiledMatrix->tileMetaCore[index3];
                            const bool lapack_diag_tile = diag_map && diag_map[index3];

                            const int cols_a = semi_a.sa;
                            const int cols_b = semi_b.sa;
                            const int rows_a = tiledMatrix->tileMetaCore[index1].height;
                            const int rows_b = tiledMatrix->tileMetaCore[index2].height;
                            const int rows_out = meta_out.height;
                            if (cols_a > 0 && cols_b > 0 && rows_a > 0) {
                                const int ld_tmp = cols_a;
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                cols_a, cols_b, rows_a,
                                                mzone, chunk_a, rows_a,
                                                chunk_b, rows_a,
                                                0.0, tmp_tile, ld_tmp);

                                // Map tmp_tile (cols_a x cols_b) to semisparse output
                                const int aind_a_size = static_cast<int>(semi_a.aind.size());
                                const int aind_b_size = static_cast<int>(semi_b.aind.size());
                                const auto& acol_map = semi_out.acol;
                                const int cols_out = meta_out.width;

                                for (int j = 0; j < cols_b && j < aind_b_size; ++j) {
                                    const int global_col = semi_b.aind[j];
                                    if (global_col < 0 || global_col >= cols_out) continue;

                                    // Find which slot in chunk_out corresponds to this global column
                                    const bool has_mapping = (global_col >= 0 && global_col < static_cast<int>(acol_map.size()));
                                    const int slot = has_mapping ? acol_map[global_col] : -1;
                                    if (slot < 0 || slot >= semi_out.sa) continue;

                                    double* out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;

                                    for (int i = 0; i < cols_a && i < aind_a_size; ++i) {
                                        const int global_row = semi_a.aind[i];
                                        if (global_row < 0 || global_row >= rows_out) continue;

                                        out_col[global_row] += tmp_tile[i + ld_tmp * j];
                                    }
                                }
                            }

                        }


                        break;
                    }
                    default:
                        break;
                }
                if (ss_aborted()) break;
            }


            // Check for fully empty tiles (rank 0 only) and export to file
            // if (rank == 0) {
            //     const int num_active = tiledMatrix->numActiveTiles;
            //     std::vector<int> empty_tile_indices;

            //     for (int t = 0; t < num_active; ++t) {
            //         double* chunk = tiledMatrix->chunkedDenseTiles[t];
            //         if (!chunk) continue;

            //         const SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[t];
            //         const TileMetaCore& meta = tiledMatrix->tileMetaCore[t];

            //         // Skip diagonal tiles
            //         const bool is_diag = (meta.row == meta.col) || (diag_map && diag_map[t]);
            //         if (is_diag) continue;

            //         const int sa = semi.sa;
            //         if (sa <= 0) continue;

            //         const int rows = (meta.height > 0) ? meta.height : tile_size;
            //         if (rows <= 0) continue;

            //         // Check if tile is fully zero
            //         bool any_nonzero = false;
            //         const int slots = sa;
            //         for (int slot = 0; slot < slots && !any_nonzero; ++slot) {
            //             const std::size_t base = static_cast<std::size_t>(slot) * static_cast<std::size_t>(rows);
            //             for (int r = 0; r < rows; ++r) {
            //                 if (std::fabs(chunk[base + r]) > 1e-15) {
            //                     any_nonzero = true;
            //                     break;
            //                 }
            //             }
            //         }

            //         if (!any_nonzero) {
            //             empty_tile_indices.push_back(t);
            //         }
            //     }

            //     if (!empty_tile_indices.empty()) {
            //         std::fprintf(stderr, "[SEMISPARSE] Found %zu fully empty off-diagonal tiles after factorization\n", empty_tile_indices.size());

            //         // Export to file
            //         FILE* fp = std::fopen("empty_tiles.txt", "w");
            //         if (fp) {
            //             std::fprintf(fp, "# Empty off-diagonal tiles after Cholesky factorization\n");
            //             std::fprintf(fp, "# Total: %zu tiles\n", empty_tile_indices.size());
            //             std::fprintf(fp, "# Format: tile_index tile_row tile_col height width sa\n");
            //             for (int t : empty_tile_indices) {
            //                 const TileMetaCore& meta = tiledMatrix->tileMetaCore[t];
            //                 const SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[t];
            //                 std::fprintf(fp, "%d %d %d %d %d %d\n",
            //                              t, meta.row, meta.col, meta.height, meta.width, semi.sa);
            //             }
            //             std::fclose(fp);
            //             std::fprintf(stderr, "[SEMISPARSE] Exported empty tile list to empty_tiles.txt\n");
            //         }

            //         // Deactivate empty tiles
            //         for (int t : empty_tile_indices) {
            //             SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[t];

            //             // Mark all active columns as inactive in acol
            //             const int old_sa = semi.sa;
            //             const int aind_size = static_cast<int>(semi.aind.size());
            //             const int limit = (old_sa < aind_size) ? old_sa : aind_size;
            //             for (int slot = 0; slot < limit; ++slot) {
            //                 const int local_col = semi.aind[static_cast<std::size_t>(slot)];
            //                 if (local_col >= 0 && local_col < static_cast<int>(semi.acol.size())) {
            //                     semi.acol[static_cast<std::size_t>(local_col)] = -1;
            //                 }
            //             }

            //             // Reset tile metadata
            //             semi.sa = 0;
            //             semi.fa = -1;
            //             semi.la = -1;
            //             semi.aind.clear();
            //         }
            //         std::fprintf(stderr, "[SEMISPARSE] Deactivated %zu empty tiles\n", empty_tile_indices.size());
            //     }
            // }

            ss_finalize();
            #undef BLKADD_CT
        }

        void dpotrf_expansion_from_chol_tasks_semisparse_debug(TiledMatrix *tiledMatrix, stiles_context_t *stile) {
            
            const int rank = STILES_RANK;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());
            const bool *diag_map = tiledMatrix->diagonal_bmapper;

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[semisparse] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADD_CT(k);
                const int ldan = BLKADD_CT(n);

                const double zone = 1.0;
                const double mzone = -1.0;
                
                switch (myroutine) {
                    case 1: { // DPOTRF

                        double *ssstile_out = tiledMatrix->chunkedDenseTiles[index1];

                        if (ssstile_out) {
                            const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                            const int kd = semi.upper_bw;
                            const int rows = meta.height;

                            sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(rows, kd, ssstile_out);
                            if (ssstatus != sTiles::StatusCode::Success) {
                                std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                                ss_abort(); break;
                            }
                        }
                        ss_cond_set(k, k, 1);
                        break;

                    }
                    case 2: { // DSYRK

                        ss_cond_wait(k, n, 1);

                        double *ssstile_in = tiledMatrix->chunkedDenseTiles[index1];
                        double *ssstile_out = tiledMatrix->chunkedDenseTiles[index2];

                        if (ssstile_in && ssstile_out) {

                            const sTiles::SemisparseTileMetaCore& semi_in = tiledMatrix->semisparseTileMetaCore[index1];
                            const int active_cols = semi_in.sa;

                            if (active_cols > 0) {

                                const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index2];
                                const sTiles::TileMetaCore& meta_in = tiledMatrix->tileMetaCore[index1];
                                const sTiles::TileMetaCore& meta_out = tiledMatrix->tileMetaCore[index2];
                                const bool lapack_diag_tile = diag_map && diag_map[index2];
                                // Use actual tile height, not nominal tile_size (fixes buffer overrun for boundary tiles)
                                const int rows_in = (meta_in.height > 0) ? meta_in.height : tile_size;

                                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, active_cols, rows_in, mzone, ssstile_in, rows_in, 0.0, tmp_tile, active_cols);

                                const int aind_size = static_cast<int>(semi_in.aind.size());
                                const int cols_out = (tiledMatrix->tileMetaCore[index2].width > 0) ? tiledMatrix->tileMetaCore[index2].width : tile_size;
                                const int rows_out = meta_out.height;
                                const int ld_tmp = active_cols;

                                // DSYRK output is always a diagonal tile in LAPACK banded format (for dpbtrf/dtbtrs)
                                const int diag_cols = (semi_out.upper_bw >= 0) ? (semi_out.upper_bw + 1) : cols_out;

                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = (j < aind_size) ? semi_in.aind[j] : -1;
                                    if (col_idx < 0 || col_idx >= cols_out) {
                                        continue;
                                    }
                                    for (int i = 0; i <= j; ++i) {
                                        const int row_idx = (i < aind_size) ? semi_in.aind[i] : -1;
                                        if (row_idx < 0 || row_idx > col_idx) {
                                            continue;
                                        }
                                        const int band = col_idx - row_idx;
                                        if (band >= diag_cols) {
                                            const double dropped = tmp_tile[i + j * ld_tmp];
                                            if (std::fabs(dropped) > 1e-15) {
                                                std::fprintf(stderr, "sTiles warning: DSYRK bandwidth overflow in tile k=%d: band=%d >= diag_cols=%d, dropped=%.6e\n",
                                                             k, band, diag_cols, dropped);
                                            }
                                            continue;
                                        }
                                        const int lapack_row = diag_cols - 1 - band;
                                        const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                                                + static_cast<std::size_t>(diag_cols)
                                                                * static_cast<std::size_t>(col_idx);
                                        ssstile_out[offset] += tmp_tile[i + j * ld_tmp];
                                    }
                                }
                            }
                        }

                        break;
                    }
                    case 3: { // DTRSM
                        ss_cond_wait(k, k, 1);

                        double* ss_rhs  = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index1] : nullptr;
                        double* ss_diag = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index2] : nullptr;
                        if (ss_rhs && ss_diag) {

                            const sTiles::SemisparseTileMetaCore& semi_rhs = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_diag = tiledMatrix->semisparseTileMetaCore[index2];
                            const sTiles::TileMetaCore& meta_rhs = tiledMatrix->tileMetaCore[index1];
                            const sTiles::TileMetaCore& meta_diag = tiledMatrix->tileMetaCore[index2];
                            const int active_cols = semi_rhs.sa;
                            const int rows_rhs = (meta_rhs.height > 0) ? meta_rhs.height : tile_size;
                            const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tile_size;
                            const int diag_bw = (semi_diag.upper_bw >= 0) ? semi_diag.upper_bw : 0;

                            if (active_cols > 0) {
                                const int kd = diag_bw;
                                const int ldab = diag_bw + 1;
                                LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', diag_dim, kd, active_cols, ss_diag, ldab, ss_rhs, rows_rhs);

                            }
                        }


                        ss_cond_set(m, k, 1);
                        break;
                    }
                    case 4: { // DGEMM
                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);

                        double* chunk_a   = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index1] : nullptr;
                        double* chunk_b   = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index2] : nullptr;
                        double* chunk_out = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index3] : nullptr;

                        if (chunk_a && chunk_b && chunk_out) {

                            const sTiles::SemisparseTileMetaCore& semi_a = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_b = tiledMatrix->semisparseTileMetaCore[index2];
                            const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index3];
                            const sTiles::TileMetaCore& meta_out = tiledMatrix->tileMetaCore[index3];
                            const bool lapack_diag_tile = diag_map && diag_map[index3];

                            const int cols_a = semi_a.sa;
                            const int cols_b = semi_b.sa;
                            const int rows_a = tiledMatrix->tileMetaCore[index1].height;
                            const int rows_b = tiledMatrix->tileMetaCore[index2].height;
                            const int rows_out = meta_out.height;
                            if (cols_a > 0 && cols_b > 0 && rows_a > 0) {
                                const int ld_tmp = cols_a;
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                cols_a, cols_b, rows_a,
                                                mzone, chunk_a, rows_a,
                                                chunk_b, rows_a,
                                                0.0, tmp_tile, ld_tmp);

                                // Map tmp_tile (cols_a x cols_b) to semisparse output
                                const int aind_a_size = static_cast<int>(semi_a.aind.size());
                                const int aind_b_size = static_cast<int>(semi_b.aind.size());
                                const auto& acol_map = semi_out.acol;
                                const int cols_out = meta_out.width;

                                for (int j = 0; j < cols_b && j < aind_b_size; ++j) {
                                    const int global_col = semi_b.aind[j];
                                    if (global_col < 0 || global_col >= cols_out) continue;

                                    // Find which slot in chunk_out corresponds to this global column
                                    const bool has_mapping = (global_col >= 0 && global_col < static_cast<int>(acol_map.size()));
                                    const int slot = has_mapping ? acol_map[global_col] : -1;
                                    if (slot < 0 || slot >= semi_out.sa) continue;

                                    double* out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;

                                    for (int i = 0; i < cols_a && i < aind_a_size; ++i) {
                                        const int global_row = semi_a.aind[i];
                                        if (global_row < 0 || global_row >= rows_out) continue;

                                        out_col[global_row] += tmp_tile[i + ld_tmp * j];
                                    }
                                }
                            }

                        }


                        break;
                    }
                    default:
                        break;
                }
                if (ss_aborted()) break;
            }


            // Check for fully empty tiles (rank 0 only) and export to file
            // if (rank == 0) {
            //     const int num_active = tiledMatrix->numActiveTiles;
            //     std::vector<int> empty_tile_indices;

            //     for (int t = 0; t < num_active; ++t) {
            //         double* chunk = tiledMatrix->chunkedDenseTiles[t];
            //         if (!chunk) continue;

            //         const SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[t];
            //         const TileMetaCore& meta = tiledMatrix->tileMetaCore[t];

            //         // Skip diagonal tiles
            //         const bool is_diag = (meta.row == meta.col) || (diag_map && diag_map[t]);
            //         if (is_diag) continue;

            //         const int sa = semi.sa;
            //         if (sa <= 0) continue;

            //         const int rows = (meta.height > 0) ? meta.height : tile_size;
            //         if (rows <= 0) continue;

            //         // Check if tile is fully zero
            //         bool any_nonzero = false;
            //         const int slots = sa;
            //         for (int slot = 0; slot < slots && !any_nonzero; ++slot) {
            //             const std::size_t base = static_cast<std::size_t>(slot) * static_cast<std::size_t>(rows);
            //             for (int r = 0; r < rows; ++r) {
            //                 if (std::fabs(chunk[base + r]) > 1e-15) {
            //                     any_nonzero = true;
            //                     break;
            //                 }
            //             }
            //         }

            //         if (!any_nonzero) {
            //             empty_tile_indices.push_back(t);
            //         }
            //     }

            //     if (!empty_tile_indices.empty()) {
            //         std::fprintf(stderr, "[SEMISPARSE] Found %zu fully empty off-diagonal tiles after factorization\n", empty_tile_indices.size());

            //         // Export to file
            //         FILE* fp = std::fopen("empty_tiles.txt", "w");
            //         if (fp) {
            //             std::fprintf(fp, "# Empty off-diagonal tiles after Cholesky factorization\n");
            //             std::fprintf(fp, "# Total: %zu tiles\n", empty_tile_indices.size());
            //             std::fprintf(fp, "# Format: tile_index tile_row tile_col height width sa\n");
            //             for (int t : empty_tile_indices) {
            //                 const TileMetaCore& meta = tiledMatrix->tileMetaCore[t];
            //                 const SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[t];
            //                 std::fprintf(fp, "%d %d %d %d %d %d\n",
            //                              t, meta.row, meta.col, meta.height, meta.width, semi.sa);
            //             }
            //             std::fclose(fp);
            //             std::fprintf(stderr, "[SEMISPARSE] Exported empty tile list to empty_tiles.txt\n");
            //         }

            //         // Deactivate empty tiles
            //         for (int t : empty_tile_indices) {
            //             SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[t];

            //             // Mark all active columns as inactive in acol
            //             const int old_sa = semi.sa;
            //             const int aind_size = static_cast<int>(semi.aind.size());
            //             const int limit = (old_sa < aind_size) ? old_sa : aind_size;
            //             for (int slot = 0; slot < limit; ++slot) {
            //                 const int local_col = semi.aind[static_cast<std::size_t>(slot)];
            //                 if (local_col >= 0 && local_col < static_cast<int>(semi.acol.size())) {
            //                     semi.acol[static_cast<std::size_t>(local_col)] = -1;
            //                 }
            //             }

            //             // Reset tile metadata
            //             semi.sa = 0;
            //             semi.fa = -1;
            //             semi.la = -1;
            //             semi.aind.clear();
            //         }
            //         std::fprintf(stderr, "[SEMISPARSE] Deactivated %zu empty tiles\n", empty_tile_indices.size());
            //     }
            // }

            ss_finalize();
            #undef BLKADD_CT
        }

        // Debug version of dpotrf_expansion_from_chol_tasks_semisparse — prints diagnostics in every case
        void dpotrf_expansion_from_chol_tasks_semisparse_debug_sparse(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int rank = STILES_RANK;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());
            const bool *diag_map = tiledMatrix->diagonal_bmapper;

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[semisparse_debug] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
#ifdef SPARSE_STILES
            double* sparse_work = tiledMatrix->workspaces[rank]->aligned_sparse_work();
#endif

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADD_CT(k);
                const int ldan = BLKADD_CT(n);

                const double zone = 1.0;
                const double mzone = -1.0;

                switch (myroutine) {
                    case 1: { // DPOTRF
                        std::fprintf(stderr, "\n======== CASE 1 (DPOTRF) ========  idx=%d, k=%d, index1=%d\n",
                                     idx, k, index1);

                        double *ssstile_out = tiledMatrix->chunkedDenseTiles[index1];
                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];

                        // ---- BEFORE ----
                        std::fprintf(stderr, "---- BEFORE DPOTRF ----\n");
                        print_semisparse_diag_tile("BEFORE POTRF", ssstile_out, meta, semi, index1);
#ifdef SPARSE_STILES
                        if (tiledMatrix->sparseTileCSC)
                            print_sparse_diag_tile("BEFORE POTRF", tiledMatrix->sparseTileCSC[index1], meta, semi, index1);
#endif

                        if (ssstile_out) {
                            const int kd = semi.upper_bw;
                            const int rows = meta.height;

                            sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(rows, kd, ssstile_out);
                            if (ssstatus != sTiles::StatusCode::Success) {
                                std::fprintf(stderr, "[SS_DEBUG]   DPOTRF FAILED: not positive definite (banded tile k=%d)\n", k);
                                std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                                ss_abort(); break;
                            }

#ifdef SPARSE_STILES
                            // Cholesky on sparse tile (same banded layout)
                            if (tiledMatrix->sparseTileCSC && !tiledMatrix->sparseTileCSC[index1].values.empty()) {
                                sTiles::StatusCode sp_status = sTiles::core_dpotrf_upper_banded(rows, kd, tiledMatrix->sparseTileCSC[index1].values.data());
                                if (sp_status != sTiles::StatusCode::Success) {
                                    std::fprintf(stderr, "[SS_DEBUG]   DPOTRF on sparse tile FAILED (tile k=%d)\n", k);
                                }
                            }
#endif
                        }

                        // ---- AFTER ----
                        std::fprintf(stderr, "---- AFTER DPOTRF ----\n");
                        print_semisparse_diag_tile("AFTER POTRF", ssstile_out, meta, semi, index1);
#ifdef SPARSE_STILES
                        if (tiledMatrix->sparseTileCSC)
                            print_sparse_diag_tile("AFTER POTRF", tiledMatrix->sparseTileCSC[index1], meta, semi, index1);
                        if (tiledMatrix->sparseTileCSC)
                            if (!compare_diag_tile("POTRF", ssstile_out, tiledMatrix->sparseTileCSC[index1], meta, semi, index1))
                                std::exit(EXIT_FAILURE);
#endif
                        std::fprintf(stderr, "======== END CASE 1 (DPOTRF) ========\n\n");

                        ss_cond_set(k, k, 1);
                        break;

                    }
                    case 2: { // DSYRK
                        std::fprintf(stderr, "\n======== CASE 2 (DSYRK) ========  idx=%d, m=%d, k=%d, n=%d, index1=%d (in), index2=%d (out)\n",
                                     idx, m, k, n, index1, index2);

                        ss_cond_wait(k, n, 1);

                        double *ssstile_in = tiledMatrix->chunkedDenseTiles[index1];
                        double *ssstile_out = tiledMatrix->chunkedDenseTiles[index2];
                        const sTiles::SemisparseTileMetaCore& semi_in = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::TileMetaCore& meta_in = tiledMatrix->tileMetaCore[index1];
                        const sTiles::TileMetaCore& meta_out = tiledMatrix->tileMetaCore[index2];

                        // ---- BEFORE ----
                        std::fprintf(stderr, "---- BEFORE DSYRK ----\n");
                        print_semisparse_offdiag_tile("BEFORE DSYRK (input)", ssstile_in, meta_in, semi_in, index1);
                        std::fprintf(stderr, "\n");
                        print_semisparse_diag_tile("BEFORE DSYRK (output)", ssstile_out, meta_out, semi_out, index2);
#ifdef SPARSE_STILES
                        if (tiledMatrix->sparseTileCSC) {
                            print_sparse_offdiag_tile("BEFORE DSYRK (input)", tiledMatrix->sparseTileCSC[index1], meta_in, index1);
                            print_sparse_diag_tile("BEFORE DSYRK (output)", tiledMatrix->sparseTileCSC[index2], meta_out, semi_out, index2);
                        }
#endif

                        if (ssstile_in && ssstile_out) {

                            const int active_cols = semi_in.sa;

                            if (active_cols > 0) {

                                const bool lapack_diag_tile = diag_map && diag_map[index2];
                                const int rows_in = (meta_in.height > 0) ? meta_in.height : tile_size;

                                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, active_cols, rows_in, mzone, ssstile_in, rows_in, 0.0, tmp_tile, active_cols);

                                const int aind_size = static_cast<int>(semi_in.aind.size());
                                const int cols_out = (tiledMatrix->tileMetaCore[index2].width > 0) ? tiledMatrix->tileMetaCore[index2].width : tile_size;
                                const int rows_out = meta_out.height;
                                const int ld_tmp = active_cols;

                                const int diag_cols = (semi_out.upper_bw >= 0) ? (semi_out.upper_bw + 1) : cols_out;

                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = (j < aind_size) ? semi_in.aind[j] : -1;
                                    if (col_idx < 0 || col_idx >= cols_out) {
                                        continue;
                                    }
                                    for (int i = 0; i <= j; ++i) {
                                        const int row_idx = (i < aind_size) ? semi_in.aind[i] : -1;
                                        if (row_idx < 0 || row_idx > col_idx) {
                                            continue;
                                        }
                                        const int band = col_idx - row_idx;
                                        if (band >= diag_cols) {
                                            const double dropped = tmp_tile[i + j * ld_tmp];
                                            if (std::fabs(dropped) > 1e-15) {
                                                std::fprintf(stderr, "sTiles warning: DSYRK bandwidth overflow in tile k=%d: band=%d >= diag_cols=%d, dropped=%.6e\n",
                                                             k, band, diag_cols, dropped);
                                            }
                                            continue;
                                        }
                                        const int lapack_row = diag_cols - 1 - band;
                                        const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                                                + static_cast<std::size_t>(diag_cols)
                                                                * static_cast<std::size_t>(col_idx);
                                        ssstile_out[offset] += tmp_tile[i + j * ld_tmp];
                                    }
                                }
                            }
                        }

#ifdef SPARSE_STILES
                        // Sparse DSYRK: diag -= offdiag^T * offdiag
                        if (tiledMatrix->sparseTileCSC) {
                            sTiles::SparseTileCSC& sp_in  = tiledMatrix->sparseTileCSC[index1];
                            sTiles::SparseTileCSC& sp_out = tiledMatrix->sparseTileCSC[index2];
                            if (!sp_in.values.empty() && !sp_out.values.empty()) {
                                const int sp_n_out = (meta_out.height > 0) ? meta_out.height : tile_size;
                                const int sp_kd    = (semi_out.upper_bw >= 0) ? semi_out.upper_bw : 0;
                                const int sp_nrows = (meta_in.height > 0) ? meta_in.height : tile_size;
                                sTiles::sparse_dsyrk_csc_to_banded(sp_in, sp_out.values.data(), sp_n_out, sp_kd,
                                                                   tmp_tile, sparse_work, sp_nrows);
                            }
                        }
#endif

                        // ---- AFTER ----
                        std::fprintf(stderr, "---- AFTER DSYRK ----\n");
                        print_semisparse_offdiag_tile("AFTER DSYRK (input)", ssstile_in, meta_in, semi_in, index1);
                        std::fprintf(stderr, "\n");
                        print_semisparse_diag_tile("AFTER DSYRK (output)", ssstile_out, meta_out, semi_out, index2);
#ifdef SPARSE_STILES
                        if (tiledMatrix->sparseTileCSC) {
                            print_sparse_offdiag_tile("AFTER DSYRK (input)", tiledMatrix->sparseTileCSC[index1], meta_in, index1);
                            print_sparse_diag_tile("AFTER DSYRK (output)", tiledMatrix->sparseTileCSC[index2], meta_out, semi_out, index2);
                            if (!compare_diag_tile("DSYRK", ssstile_out, tiledMatrix->sparseTileCSC[index2], meta_out, semi_out, index2))
                                std::exit(EXIT_FAILURE);
                        }
#endif
                        std::fprintf(stderr, "======== END CASE 2 (DSYRK) ========\n\n");

                        break;
                    }
                    case 3: { // DTRSM
                        std::fprintf(stderr, "\n======== CASE 3 (DTRSM) ========  idx=%d, m=%d, k=%d, index1=%d (rhs), index2=%d (diag)\n",
                                     idx, m, k, index1, index2);

                        ss_cond_wait(k, k, 1);

                        double* ss_rhs  = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index1] : nullptr;
                        double* ss_diag = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index2] : nullptr;
                        const sTiles::SemisparseTileMetaCore& semi_rhs = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi_diag = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::TileMetaCore& meta_rhs = tiledMatrix->tileMetaCore[index1];
                        const sTiles::TileMetaCore& meta_diag = tiledMatrix->tileMetaCore[index2];

                        // ---- BEFORE ----
                        std::fprintf(stderr, "---- BEFORE DTRSM ----\n");
                        print_semisparse_offdiag_tile("BEFORE DTRSM (rhs)", ss_rhs, meta_rhs, semi_rhs, index1);
                        std::fprintf(stderr, "\n");
                        print_semisparse_diag_tile("BEFORE DTRSM (diag)", ss_diag, meta_diag, semi_diag, index2);
#ifdef SPARSE_STILES
                        if (tiledMatrix->sparseTileCSC) {
                            print_sparse_offdiag_tile("BEFORE DTRSM (rhs)", tiledMatrix->sparseTileCSC[index1], meta_rhs, index1);
                            print_sparse_diag_tile("BEFORE DTRSM (diag)", tiledMatrix->sparseTileCSC[index2], meta_diag, semi_diag, index2);
                        }
#endif

                        if (ss_rhs && ss_diag) {

                            const int active_cols = semi_rhs.sa;
                            const int rows_rhs = (meta_rhs.height > 0) ? meta_rhs.height : tile_size;
                            const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tile_size;
                            const int diag_bw = (semi_diag.upper_bw >= 0) ? semi_diag.upper_bw : 0;

                            if (active_cols > 0) {
                                const int kd = diag_bw;
                                const int ldab = diag_bw + 1;
                                LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', diag_dim, kd, active_cols, ss_diag, ldab, ss_rhs, rows_rhs);
                            }
                        }

#ifdef SPARSE_STILES
                        // Sparse DTRSM: offdiag = diag^{-T} * offdiag
                        if (tiledMatrix->sparseTileCSC) {
                            sTiles::SparseTileCSC& sp_rhs  = tiledMatrix->sparseTileCSC[index1];
                            sTiles::SparseTileCSC& sp_diag = tiledMatrix->sparseTileCSC[index2];
                            if (!sp_rhs.values.empty() && !sp_diag.values.empty()) {
                                const int n_L = (meta_diag.height > 0) ? meta_diag.height : tile_size;
                                const int kd  = (semi_diag.upper_bw >= 0) ? semi_diag.upper_bw : 0;
                                sTiles::sparse_dtrsm_banded_csc(sp_diag.values.data(), n_L, kd, sp_rhs, tmp_tile);
                            }
                        }
#endif

                        // ---- AFTER ----
                        std::fprintf(stderr, "---- AFTER DTRSM ----\n");
                        print_semisparse_offdiag_tile("AFTER DTRSM (rhs)", ss_rhs, meta_rhs, semi_rhs, index1);
                        std::fprintf(stderr, "\n");
                        print_semisparse_diag_tile("AFTER DTRSM (diag)", ss_diag, meta_diag, semi_diag, index2);
#ifdef SPARSE_STILES
                        if (tiledMatrix->sparseTileCSC) {
                            print_sparse_offdiag_tile("AFTER DTRSM (rhs)", tiledMatrix->sparseTileCSC[index1], meta_rhs, index1);
                            print_sparse_diag_tile("AFTER DTRSM (diag)", tiledMatrix->sparseTileCSC[index2], meta_diag, semi_diag, index2);
                            if (!compare_offdiag_tile("DTRSM", ss_rhs, tiledMatrix->sparseTileCSC[index1], meta_rhs, semi_rhs, index1))
                                std::exit(EXIT_FAILURE);
                        }
#endif
                        std::fprintf(stderr, "======== END CASE 3 (DTRSM) ========\n\n");

                        ss_cond_set(m, k, 1);
                        break;
                    }
                    case 4: { // DGEMM
                        std::fprintf(stderr, "\n======== CASE 4 (DGEMM) ========  idx=%d, m=%d, k=%d, n=%d, index1=%d (A), index2=%d (B), index3=%d (out)\n",
                                     idx, m, k, n, index1, index2, index3);

                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);

                        double* chunk_a   = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index1] : nullptr;
                        double* chunk_b   = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index2] : nullptr;
                        double* chunk_out = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index3] : nullptr;
                        const sTiles::SemisparseTileMetaCore& semi_a = tiledMatrix->semisparseTileMetaCore[index1];
                        const sTiles::SemisparseTileMetaCore& semi_b = tiledMatrix->semisparseTileMetaCore[index2];
                        const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index3];
                        const sTiles::TileMetaCore& meta_a = tiledMatrix->tileMetaCore[index1];
                        const sTiles::TileMetaCore& meta_b = tiledMatrix->tileMetaCore[index2];
                        const sTiles::TileMetaCore& meta_out = tiledMatrix->tileMetaCore[index3];
                        const bool out_is_diag = diag_map && diag_map[index3];

                        // ---- BEFORE ----
                        std::fprintf(stderr, "---- BEFORE DGEMM ----\n");
                        print_semisparse_offdiag_tile("BEFORE DGEMM (A)", chunk_a, meta_a, semi_a, index1);
                        std::fprintf(stderr, "\n");
                        print_semisparse_offdiag_tile("BEFORE DGEMM (B)", chunk_b, meta_b, semi_b, index2);
                        std::fprintf(stderr, "\n");
                        if (out_is_diag) {
                            print_semisparse_diag_tile("BEFORE DGEMM (out-diag)", chunk_out, meta_out, semi_out, index3);
                        } else {
                            print_semisparse_offdiag_tile("BEFORE DGEMM (out-offdiag)", chunk_out, meta_out, semi_out, index3);
                        }
#ifdef SPARSE_STILES
                        if (tiledMatrix->sparseTileCSC) {
                            print_sparse_offdiag_tile("BEFORE DGEMM (A)", tiledMatrix->sparseTileCSC[index1], meta_a, index1);
                            print_sparse_offdiag_tile("BEFORE DGEMM (B)", tiledMatrix->sparseTileCSC[index2], meta_b, index2);
                            print_sparse_offdiag_tile("BEFORE DGEMM (out)", tiledMatrix->sparseTileCSC[index3], meta_out, index3);
                        }
#endif

                        if (chunk_a && chunk_b && chunk_out) {

                            const int cols_a = semi_a.sa;
                            const int cols_b = semi_b.sa;
                            const int rows_a = meta_a.height;
                            const int rows_out = meta_out.height;

                            if (cols_a > 0 && cols_b > 0 && rows_a > 0) {
                                const int ld_tmp = cols_a;
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                cols_a, cols_b, rows_a,
                                                mzone, chunk_a, rows_a,
                                                chunk_b, rows_a,
                                                0.0, tmp_tile, ld_tmp);

                                const int aind_a_size = static_cast<int>(semi_a.aind.size());
                                const int aind_b_size = static_cast<int>(semi_b.aind.size());
                                const auto& acol_map = semi_out.acol;
                                const int cols_out = meta_out.width;

                                for (int j = 0; j < cols_b && j < aind_b_size; ++j) {
                                    const int global_col = semi_b.aind[j];
                                    if (global_col < 0 || global_col >= cols_out) continue;

                                    const bool has_mapping = (global_col >= 0 && global_col < static_cast<int>(acol_map.size()));
                                    const int slot = has_mapping ? acol_map[global_col] : -1;
                                    if (slot < 0 || slot >= semi_out.sa) continue;

                                    double* out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;

                                    for (int i = 0; i < cols_a && i < aind_a_size; ++i) {
                                        const int global_row = semi_a.aind[i];
                                        if (global_row < 0 || global_row >= rows_out) continue;

                                        out_col[global_row] += tmp_tile[i + ld_tmp * j];
                                    }
                                }
                            }
                        }

#ifdef SPARSE_STILES
                        // Sparse DGEMM: out -= A^T * B
                        if (tiledMatrix->sparseTileCSC) {
                            sTiles::SparseTileCSC& sp_a   = tiledMatrix->sparseTileCSC[index1];
                            sTiles::SparseTileCSC& sp_b   = tiledMatrix->sparseTileCSC[index2];
                            sTiles::SparseTileCSC& sp_out = tiledMatrix->sparseTileCSC[index3];
                            if (!sp_a.values.empty() && !sp_b.values.empty() && !sp_out.values.empty()) {
                                const int sp_nrows = (meta_a.height > 0) ? meta_a.height : tile_size;
                                sTiles::sparse_dgemm_AtB_csc(sp_a, sp_b, sp_out,
                                                             sparse_work, tmp_tile, sp_nrows);
                            }
                        }
#endif

                        // ---- AFTER ----
                        std::fprintf(stderr, "---- AFTER DGEMM ----\n");
                        print_semisparse_offdiag_tile("AFTER DGEMM (A)", chunk_a, meta_a, semi_a, index1);
                        std::fprintf(stderr, "\n");
                        print_semisparse_offdiag_tile("AFTER DGEMM (B)", chunk_b, meta_b, semi_b, index2);
                        std::fprintf(stderr, "\n");
                        if (out_is_diag) {
                            print_semisparse_diag_tile("AFTER DGEMM (out-diag)", chunk_out, meta_out, semi_out, index3);
                        } else {
                            print_semisparse_offdiag_tile("AFTER DGEMM (out-offdiag)", chunk_out, meta_out, semi_out, index3);
                        }
#ifdef SPARSE_STILES
                        if (tiledMatrix->sparseTileCSC) {
                            print_sparse_offdiag_tile("AFTER DGEMM (A)", tiledMatrix->sparseTileCSC[index1], meta_a, index1);
                            print_sparse_offdiag_tile("AFTER DGEMM (B)", tiledMatrix->sparseTileCSC[index2], meta_b, index2);
                            print_sparse_offdiag_tile("AFTER DGEMM (out)", tiledMatrix->sparseTileCSC[index3], meta_out, index3);
                            if (!compare_offdiag_tile("DGEMM", chunk_out, tiledMatrix->sparseTileCSC[index3], meta_out, semi_out, index3))
                                std::exit(EXIT_FAILURE);
                        }
#endif
                        std::fprintf(stderr, "======== END CASE 4 (DGEMM) ========\n\n");

                        break;
                    }
                    default:
                        std::fprintf(stderr, "[SS_DEBUG] case DEFAULT: idx=%d, myroutine=%d (unexpected!)\n",
                                     idx, myroutine);
                        break;
                }
                if (ss_aborted()) {
                    break;
                }
            }

            ss_finalize();
            #undef BLKADD_CT
        }

        void dpotrf_expansion_from_chol_tasks_semisparse_test_sparse(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int rank = STILES_RANK;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[sparse] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
#ifdef SPARSE_STILES
            double* sparse_work = tiledMatrix->workspaces[rank]->aligned_sparse_work();
            double* dtrsm_work  = tiledMatrix->workspaces[rank]->aligned_tile();
#endif

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                switch (myroutine) {
                    case 1: { // DPOTRF
#ifdef SPARSE_STILES
                        if (tiledMatrix->sparseTileCSC && !tiledMatrix->sparseTileCSC[index1].values.empty()) {
                            const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                            const int rows = meta.height;
                            const int kd = semi.upper_bw;
                            sTiles::StatusCode sp_status = sTiles::core_dpotrf_upper_banded(rows, kd, tiledMatrix->sparseTileCSC[index1].values.data());
                            if (sp_status != sTiles::StatusCode::Success) {
                                sTiles::Logger::error("matrix is not positive definite (sparse tile k=", k, ")");
                                ss_abort(); break;
                            }
                        }
#endif
                        ss_cond_set(k, k, 1);
                        break;
                    }
                    case 2: { // DSYRK
                        ss_cond_wait(k, n, 1);
#ifdef SPARSE_STILES
                        if (tiledMatrix->sparseTileCSC) {
                            sTiles::SparseTileCSC& sp_in  = tiledMatrix->sparseTileCSC[index1];
                            sTiles::SparseTileCSC& sp_out = tiledMatrix->sparseTileCSC[index2];
                            if (!sp_in.values.empty() && !sp_out.values.empty()) {
                                const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index2];
                                const sTiles::TileMetaCore& meta_out = tiledMatrix->tileMetaCore[index2];
                                const sTiles::TileMetaCore& meta_in  = tiledMatrix->tileMetaCore[index1];
                                const int n_out = (meta_out.height > 0) ? meta_out.height : tile_size;
                                const int kd    = (semi_out.upper_bw >= 0) ? semi_out.upper_bw : 0;
                                const int nrows = (meta_in.height > 0) ? meta_in.height : tile_size;
                                sTiles::sparse_dsyrk_csc_to_banded(sp_in, sp_out.values.data(), n_out, kd,
                                                                   dtrsm_work, sparse_work, nrows);
                            }
                        }
#endif
                        break;
                    }
                    case 3: { // DTRSM
                        ss_cond_wait(k, k, 1);
#ifdef SPARSE_STILES
                        if (tiledMatrix->sparseTileCSC) {
                            sTiles::SparseTileCSC& sp_rhs  = tiledMatrix->sparseTileCSC[index1];
                            sTiles::SparseTileCSC& sp_diag = tiledMatrix->sparseTileCSC[index2];
                            if (!sp_rhs.values.empty() && !sp_diag.values.empty()) {
                                const sTiles::SemisparseTileMetaCore& semi_diag = tiledMatrix->semisparseTileMetaCore[index2];
                                const sTiles::TileMetaCore& meta_diag = tiledMatrix->tileMetaCore[index2];
                                const int n_L = (meta_diag.height > 0) ? meta_diag.height : tile_size;
                                const int kd  = (semi_diag.upper_bw >= 0) ? semi_diag.upper_bw : 0;
                                sTiles::sparse_dtrsm_banded_csc(sp_diag.values.data(), n_L, kd, sp_rhs, dtrsm_work);
                            }
                        }
#endif
                        ss_cond_set(m, k, 1);
                        break;
                    }
                    case 4: { // DGEMM
                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);
#ifdef SPARSE_STILES
                        if (tiledMatrix->sparseTileCSC) {
                            sTiles::SparseTileCSC& sp_a   = tiledMatrix->sparseTileCSC[index1];
                            sTiles::SparseTileCSC& sp_b   = tiledMatrix->sparseTileCSC[index2];
                            sTiles::SparseTileCSC& sp_out = tiledMatrix->sparseTileCSC[index3];
                            if (!sp_a.values.empty() && !sp_b.values.empty() && !sp_out.values.empty()) {
                                const int nrows = (tiledMatrix->tileMetaCore[index1].height > 0)
                                                ? tiledMatrix->tileMetaCore[index1].height : tile_size;
                                sTiles::sparse_dgemm_AtB_csc(sp_a, sp_b, sp_out,
                                                             sparse_work, dtrsm_work, nrows);
                            }
                        }
#endif
                        break;
                    }
                    default:
                        break;
                }
                if (ss_aborted()) {
                    break;
                }
            }

            ss_finalize();

#ifdef SPARSE_STILES
            // Compute sparse logdet from CSC diagonal tiles for cross-validation
            {
                double sparse_logdet = 0.0;
                const bool* diag_map = tiledMatrix->diagonal_bmapper;
                const int ntiles = tiledMatrix->dimTiledMatrix;
                for (int t = 0; t < ntiles; ++t) {
                    if (!diag_map || !diag_map[t]) continue;
                    const sTiles::SparseTileCSC& sp = tiledMatrix->sparseTileCSC[t];
                    if (sp.values.empty()) continue;
                    const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[t];
                    const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[t];
                    const int ts = (meta.height > 0) ? meta.height : tile_size;
                    const int kd = (semi.upper_bw >= 0) ? semi.upper_bw : 0;
                    const int ldab = kd + 1;
                    const double* vals = sp.values.data();
                    for (int j = 0; j < ts; ++j) {
                        double d = vals[kd + static_cast<std::size_t>(j) * ldab];
                        if (d > 0.0) sparse_logdet += std::log(d);
                    }
                }
                sparse_logdet *= 2.0;
                std::cout << "[sparse-logdet] " << std::setprecision(10) << sparse_logdet << std::endl;
            }
#endif
        }


        // Single-core semisparse Cholesky — no dependency tracking overhead
        void dpotrf_expansion_from_chol_tasks_semisparse_imp1_serial(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const bool *diag_map = tiledMatrix->diagonal_bmapper;

            double* tmp_tile = tiledMatrix->workspaces[0]->aligned_tile();

            const int ntasks = static_cast<int>(tasks.size());
            for (int idx = 0; idx < ntasks; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADD_CT(k);
                const int ldan = BLKADD_CT(n);

                const double mzone = -1.0;

                switch (myroutine) {
                    case 1: { // DPOTRF
                        double *ssstile_out = tiledMatrix->chunkedDenseTiles[index1];
                        if (ssstile_out) {
                            const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                            const int kd = semi.upper_bw;
                            const int rows = meta.height;
                            sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(rows, kd, ssstile_out);
                            if (ssstatus != sTiles::StatusCode::Success) {
                                std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                                stile->ss_abort = 1;
                                return;
                            }
                        }
                        break;
                    }
                    case 2: { // DSYRK
                        double *ssstile_in = tiledMatrix->chunkedDenseTiles[index1];
                        double *ssstile_out = tiledMatrix->chunkedDenseTiles[index2];

                        if (ssstile_in && ssstile_out) {
                            const sTiles::SemisparseTileMetaCore& semi_in = tiledMatrix->semisparseTileMetaCore[index1];
                            const int active_cols = semi_in.sa;

                            if (active_cols > 0) {
                                const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index2];
                                const sTiles::TileMetaCore& meta_in = tiledMatrix->tileMetaCore[index1];
                                const sTiles::TileMetaCore& meta_out = tiledMatrix->tileMetaCore[index2];
                                const int rows_in = (meta_in.height > 0) ? meta_in.height : tile_size;

                                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, active_cols, rows_in, mzone, ssstile_in, rows_in, 0.0, tmp_tile, active_cols);

                                const int aind_size = static_cast<int>(semi_in.aind.size());
                                const int cols_out = (tiledMatrix->tileMetaCore[index2].width > 0) ? tiledMatrix->tileMetaCore[index2].width : tile_size;
                                const int rows_out = meta_out.height;
                                const int ld_tmp = active_cols;
                                const int diag_cols = (semi_out.upper_bw >= 0) ? (semi_out.upper_bw + 1) : cols_out;

                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = (j < aind_size) ? semi_in.aind[j] : -1;
                                    if (col_idx < 0 || col_idx >= cols_out) continue;
                                    for (int i = 0; i <= j; ++i) {
                                        const int row_idx = (i < aind_size) ? semi_in.aind[i] : -1;
                                        if (row_idx < 0 || row_idx > col_idx) continue;
                                        const int band = col_idx - row_idx;
                                        if (band >= diag_cols) {
                                            const double dropped = tmp_tile[i + j * ld_tmp];
                                            if (std::fabs(dropped) > 1e-15) {
                                                std::fprintf(stderr, "sTiles warning: DSYRK bandwidth overflow in tile k=%d: band=%d >= diag_cols=%d, dropped=%.6e\n",
                                                             k, band, diag_cols, dropped);
                                            }
                                            continue;
                                        }
                                        const int lapack_row = diag_cols - 1 - band;
                                        const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                                                + static_cast<std::size_t>(diag_cols)
                                                                * static_cast<std::size_t>(col_idx);
                                        ssstile_out[offset] += tmp_tile[i + j * ld_tmp];
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case 3: { // DTRSM
                        double* ss_rhs  = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index1] : nullptr;
                        double* ss_diag = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index2] : nullptr;
                        if (ss_rhs && ss_diag) {
                            const sTiles::SemisparseTileMetaCore& semi_rhs = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_diag = tiledMatrix->semisparseTileMetaCore[index2];
                            const sTiles::TileMetaCore& meta_rhs = tiledMatrix->tileMetaCore[index1];
                            const sTiles::TileMetaCore& meta_diag = tiledMatrix->tileMetaCore[index2];
                            const int active_cols = semi_rhs.sa;
                            const int rows_rhs = (meta_rhs.height > 0) ? meta_rhs.height : tile_size;
                            const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tile_size;
                            const int diag_bw = (semi_diag.upper_bw >= 0) ? semi_diag.upper_bw : 0;

                            if (active_cols > 0) {
                                const int kd = diag_bw;
                                const int ldab = diag_bw + 1;
                                LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', diag_dim, kd, active_cols, ss_diag, ldab, ss_rhs, rows_rhs);
                            }
                        }
                        break;
                    }
                    case 4: { // DGEMM
                        double* chunk_a   = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index1] : nullptr;
                        double* chunk_b   = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index2] : nullptr;
                        double* chunk_out = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index3] : nullptr;

                        if (chunk_a && chunk_b && chunk_out) {
                            const sTiles::SemisparseTileMetaCore& semi_a = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_b = tiledMatrix->semisparseTileMetaCore[index2];
                            const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index3];
                            const sTiles::TileMetaCore& meta_out = tiledMatrix->tileMetaCore[index3];

                            const int cols_a = semi_a.sa;
                            const int cols_b = semi_b.sa;
                            const int rows_a = tiledMatrix->tileMetaCore[index1].height;
                            const int rows_b = tiledMatrix->tileMetaCore[index2].height;
                            const int rows_out = meta_out.height;
                            if (cols_a > 0 && cols_b > 0 && rows_a > 0) {
                                const int ld_tmp = cols_a;
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                cols_a, cols_b, rows_a,
                                                mzone, chunk_a, rows_a,
                                                chunk_b, rows_a,
                                                0.0, tmp_tile, ld_tmp);

                                const int aind_a_size = static_cast<int>(semi_a.aind.size());
                                const int aind_b_size = static_cast<int>(semi_b.aind.size());
                                const auto& acol_map = semi_out.acol;
                                const int cols_out = meta_out.width;

                                for (int j = 0; j < cols_b && j < aind_b_size; ++j) {
                                    const int global_col = semi_b.aind[j];
                                    if (global_col < 0 || global_col >= cols_out) continue;

                                    const bool has_mapping = (global_col >= 0 && global_col < static_cast<int>(acol_map.size()));
                                    const int slot = has_mapping ? acol_map[global_col] : -1;
                                    if (slot < 0 || slot >= semi_out.sa) continue;

                                    double* out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;

                                    for (int i = 0; i < cols_a && i < aind_a_size; ++i) {
                                        const int global_row = semi_a.aind[i];
                                        if (global_row < 0 || global_row >= rows_out) continue;
                                        out_col[global_row] += tmp_tile[i + ld_tmp * j];
                                    }
                                }
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
            #undef BLKADD_CT
        }

        void dpotrf_expansion_from_chol_tasks_semisparse_imp2_serial(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;

            // Cache commonly accessed pointers with restrict + alignment hints
            double** __restrict__ const chunked_tiles = tiledMatrix->chunkedDenseTiles;
            const sTiles::SemisparseTileMetaCore* __restrict__ const semisparse_meta = tiledMatrix->semisparseTileMetaCore;
            const sTiles::TileMetaCore* __restrict__ const tile_meta = tiledMatrix->tileMetaCore;

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const int ntasks = static_cast<int>(tasks.size());

            double* __restrict__ tmp_tile = tiledMatrix->workspaces[0]->aligned_tile();
            const double mzone = -1.0;

            // Fuse threshold: below this, bypass BLAS and compute dot products inline
            constexpr int FUSE_THRESHOLD = 16;

            // O(1) contiguity check: since aind is always sorted ascending,
            // contiguous iff last - first + 1 == count
            auto is_contiguous = [](const sTiles::SemisparseTileMetaCore& s) -> bool {
                return s.sa > 0 && (s.la - s.fa + 1 == s.sa);
            };

            const int ntiles_bounds = tiledMatrix->numActiveTiles;

            for (int idx = 0; idx < ntasks; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int k = t[2];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                // Prefetch next task's tiles and metadata
                if (idx + 1 < ntasks) {
                    const std::array<int,7> &next_t = tasks[idx + 1];
                    const int next_idx1 = next_t[4];
                    const int next_idx2 = next_t[5];
                    if (next_idx1 >= 0 && next_idx1 < ntiles_bounds && chunked_tiles[next_idx1]) {
                        const char* p1 = reinterpret_cast<const char*>(chunked_tiles[next_idx1]);
                        __builtin_prefetch(p1,       0, 3);
                        __builtin_prefetch(p1 + 64,  0, 3);
                        __builtin_prefetch(p1 + 128, 0, 3);
                        __builtin_prefetch(p1 + 192, 0, 3);
                    }
                    if (next_idx2 >= 0 && next_idx2 < ntiles_bounds && chunked_tiles[next_idx2]) {
                        const char* p2 = reinterpret_cast<const char*>(chunked_tiles[next_idx2]);
                        __builtin_prefetch(p2,       1, 3);
                        __builtin_prefetch(p2 + 64,  1, 3);
                        __builtin_prefetch(p2 + 128, 1, 3);
                        __builtin_prefetch(p2 + 192, 1, 3);
                    }
                    if (next_t[0] == 4) {
                        const int next_idx3 = next_t[6];
                        if (next_idx3 >= 0 && next_idx3 < ntiles_bounds && chunked_tiles[next_idx3]) {
                            const char* p3 = reinterpret_cast<const char*>(chunked_tiles[next_idx3]);
                            __builtin_prefetch(p3,       1, 3);
                            __builtin_prefetch(p3 + 64,  1, 3);
                        }
                    }
                    __builtin_prefetch(&semisparse_meta[next_idx1], 0, 3);
                    __builtin_prefetch(&semisparse_meta[next_idx2], 0, 3);
                }

                switch (myroutine) {

                    case 4: { // DGEMM — C(k,m) -= A(n,k)^T * B(n,m)
                        // Load output tile once for all consecutive case-4 tasks with same index3
                        double* __restrict__ chunk_out = chunked_tiles[index3];
                        const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index3];
                        const int rows_out = tile_meta[index3].height;
                        const int* __restrict__ acol_map = semi_out.acol.data();
                        const bool out_full = semi_out.is_full_width;

                        // Find last consecutive case-4 task with same output tile
                        int fused_end = idx;
                        while (fused_end + 1 < ntasks && tasks[fused_end + 1][0] == 4 && tasks[fused_end + 1][6] == index3)
                            ++fused_end;

                        for (int fi = idx; fi <= fused_end; ++fi) {
                            const auto& ft = tasks[fi];
                            const int fi1 = ft[4];
                            const int fi2 = ft[5];

                            double* __restrict__ chunk_a = chunked_tiles[fi1];
                            double* __restrict__ chunk_b = chunked_tiles[fi2];

                            const sTiles::SemisparseTileMetaCore& semi_a = semisparse_meta[fi1];
                            const sTiles::SemisparseTileMetaCore& semi_b = semisparse_meta[fi2];

                            const int cols_a = semi_a.sa;
                            const int cols_b = semi_b.sa;
                            const int rows_a = tile_meta[fi1].height;

                            const int* __restrict__ aind_a = semi_a.aind.data();
                            const int* __restrict__ aind_b = semi_b.aind.data();

                            const bool a_contig = semi_a.is_contiguous;
                            const bool b_contig = semi_b.is_contiguous;

                            if (a_contig && b_contig && out_full) {
                                // === FAST PATH: direct DGEMM into output, no tmp_tile ===
                                const int a_start = semi_a.fa;
                                const int b_start = semi_b.fa;
                                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                            cols_a, cols_b, rows_a, mzone,
                                            chunk_a, rows_a,
                                            chunk_b, rows_a,
                                            1.0, chunk_out + a_start + static_cast<std::size_t>(b_start) * rows_out, rows_out);

                            } else if (cols_a <= FUSE_THRESHOLD && cols_b <= FUSE_THRESHOLD) {
                                // === FUSED compute-scatter: bypass BLAS entirely ===
                                if (a_contig) {
                                    const int a_start = semi_a.fa;
                                    for (int j = 0; j < cols_b; ++j) {
                                        double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(acol_map[aind_b[j]]) * rows_out;
                                        const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                        for (int i = 0; i < cols_a; ++i) {
                                            const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                            double dot = 0.0;
                                            for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                            out_col[a_start + i] -= dot;
                                        }
                                    }
                                } else {
                                    for (int j = 0; j < cols_b; ++j) {
                                        double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(acol_map[aind_b[j]]) * rows_out;
                                        const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                        for (int i = 0; i < cols_a; ++i) {
                                            const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                            double dot = 0.0;
                                            for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                            out_col[aind_a[i]] -= dot;
                                        }
                                    }
                                }

                            } else {
                                // === BLAS DGEMM + scatter ===
                                const int ld_tmp = cols_a;
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                   cols_a, cols_b, rows_a,
                                                   mzone, chunk_a, rows_a,
                                                   chunk_b, rows_a,
                                                   0.0, tmp_tile, ld_tmp);

                                if (a_contig) {
                                    const int a_start = semi_a.fa;
                                    for (int j = 0; j < cols_b; ++j) {
                                        cblas_daxpy(cols_a, 1.0, tmp_tile + j * ld_tmp, 1,
                                                    chunk_out + static_cast<std::size_t>(acol_map[aind_b[j]]) * rows_out + a_start, 1);
                                    }
                                } else {
                                    for (int j = 0; j < cols_b; ++j) {
                                        double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(acol_map[aind_b[j]]) * rows_out;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        int i = 0;
                                        for (; i + 3 < cols_a; i += 4) {
                                            out_col[aind_a[i]]     += tmp_col[i];
                                            out_col[aind_a[i + 1]] += tmp_col[i + 1];
                                            out_col[aind_a[i + 2]] += tmp_col[i + 2];
                                            out_col[aind_a[i + 3]] += tmp_col[i + 3];
                                        }
                                        for (; i < cols_a; ++i) out_col[aind_a[i]] += tmp_col[i];
                                    }
                                }
                            }
                        } // end fused loop

                        idx = fused_end;
                        break;
                    }

                    case 1: { // DPOTRF — banded Cholesky on diagonal tile
                        double* __restrict__ tile_out = chunked_tiles[index1];
                        if (tile_out) {
                            const sTiles::SemisparseTileMetaCore& semi = semisparse_meta[index1];
                            const sTiles::TileMetaCore& meta = tile_meta[index1];
                            sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(meta.height, semi.upper_bw, tile_out);
                            if (ssstatus != sTiles::StatusCode::Success) {
                                std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                                stile->ss_abort = 1;
                                return;
                            }
                        }
                        break;
                    }

                    case 2: { // DSYRK — symmetric rank-k update into banded diagonal
                        double* __restrict__ tile_in = chunked_tiles[index1];
                        double* __restrict__ tile_out = chunked_tiles[index2];

                        if (tile_in && tile_out) {
                            const sTiles::SemisparseTileMetaCore& semi_in = semisparse_meta[index1];
                            const int active_cols = semi_in.sa;

                            if (active_cols > 0) {
                                const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index2];
                                const int diag_cols = semi_out.upper_bw + 1;
                                const int rows_in = tile_meta[index1].height;
                                const int base_row = diag_cols - 1;

                                const bool aind_contig = semi_in.is_contiguous;

                                if (active_cols <= FUSE_THRESHOLD && aind_contig && active_cols <= diag_cols) {
                                    // === FUSED contiguous: dot products directly into banded output ===
                                    const int aind_start = semi_in.fa;
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_start + j;
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                        for (int i = 0; i <= j; ++i) {
                                            const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                            double dot = 0.0;
                                            for (int r = 0; r < rows_in; ++r) dot += col_i[r] * col_j[r];
                                            tile_out[base_row - (j - i) + col_base] -= dot;
                                        }
                                    }

                                } else if (active_cols <= FUSE_THRESHOLD) {
                                    // === FUSED scattered: dot products with aind lookups ===
                                    const int* __restrict__ aind = semi_in.aind.data();
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind[j];
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                        for (int i = 0; i <= j; ++i) {
                                            const int band = col_idx - aind[i];
                                            if (band < diag_cols) {
                                                const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                                double dot = 0.0;
                                                for (int r = 0; r < rows_in; ++r) dot += col_i[r] * col_j[r];
                                                tile_out[base_row - band + col_base] -= dot;
                                            }
                                        }
                                    }

                                } else if (aind_contig && active_cols <= diag_cols) {
                                    // === BLAS dsyrk + contiguous scatter ===
                                    const int ld_tmp = active_cols;
                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                       active_cols, rows_in, mzone,
                                                       tile_in, rows_in, 0.0, tmp_tile, ld_tmp);

                                    const int aind_start = semi_in.fa;
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_start + j;
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        for (int i = 0; i <= j; ++i) {
                                            tile_out[base_row - (j - i) + col_base] += tmp_col[i];
                                        }
                                    }

                                } else {
                                    // === BLAS dsyrk + general scatter ===
                                    const int ld_tmp = active_cols;
                                    const int* __restrict__ aind = semi_in.aind.data();
                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                       active_cols, rows_in, mzone,
                                                       tile_in, rows_in, 0.0, tmp_tile, ld_tmp);

                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind[j];
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        for (int i = 0; i <= j; ++i) {
                                            const int band = col_idx - aind[i];
                                            if (band < diag_cols) {
                                                tile_out[base_row - band + col_base] += tmp_col[i];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }

                    case 3: { // DTRSM — banded triangular solve
                        double* __restrict__ ss_rhs = chunked_tiles[index1];
                        double* __restrict__ ss_diag = chunked_tiles[index2];

                        if (ss_rhs && ss_diag) {
                            const sTiles::SemisparseTileMetaCore& semi_rhs = semisparse_meta[index1];
                            const sTiles::TileMetaCore& meta_rhs = tile_meta[index1];
                            const sTiles::TileMetaCore& meta_diag = tile_meta[index2];

                            const int active_cols = semi_rhs.sa;
                            if (active_cols > 0) {
                                const int rows_rhs = meta_rhs.height;
                                const int diag_dim = meta_diag.height;
                                const int kd = semisparse_meta[index2].upper_bw;

                                LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                               diag_dim, kd, active_cols,
                                               ss_diag, kd + 1, ss_rhs, rows_rhs);
                            }
                        }
                        break;
                    }

                    default:
                        break;
                }
            }
        }

        void dpotrf_expansion_from_chol_tasks_semisparse_imp3_serial(TiledMatrix *tiledMatrix, stiles_context_t *stile) {
            (void)stile; // unused in serial version

            const int rank = STILES_RANK;

            double** __restrict__ const chunked_tiles = tiledMatrix->chunkedDenseTiles;
            const sTiles::SemisparseTileMetaCore* __restrict__ const semisparse_meta = tiledMatrix->semisparseTileMetaCore;
            const sTiles::TileMetaCore* __restrict__ const tile_meta = tiledMatrix->tileMetaCore;

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const int start = 0;
            const int end = static_cast<int>(tasks.size());

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[semisparse_imp3_serial] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* __restrict__ tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
            const double mzone = -1.0;

            constexpr int FUSE_THRESHOLD = 16;

            // Precomputed scatter info (may be null)
            const int64_t* scatter_index = (tiledMatrix->chol_scatter_index) ? tiledMatrix->chol_scatter_index->data() : nullptr;
            const int32_t* scatter_packed = (tiledMatrix->chol_scatter_packed) ? tiledMatrix->chol_scatter_packed->data() : nullptr;
            const bool has_scatter_info = (scatter_index != nullptr && scatter_packed != nullptr);

            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int ntiles_bounds = tiledMatrix->numActiveTiles;

                // Prefetching
                if (idx + 1 < end) {
                    const std::array<int,7> &next_t = tasks[idx + 1];
                    const int next_idx1 = next_t[4];
                    const int next_idx2 = next_t[5];
                    if (next_idx1 >= 0 && next_idx1 < ntiles_bounds && chunked_tiles[next_idx1]) {
                        const char* p1 = reinterpret_cast<const char*>(chunked_tiles[next_idx1]);
                        __builtin_prefetch(p1, 0, 3); __builtin_prefetch(p1+64, 0, 3);
                        __builtin_prefetch(p1+128, 0, 3); __builtin_prefetch(p1+192, 0, 3);
                    }
                    if (next_idx2 >= 0 && next_idx2 < ntiles_bounds && chunked_tiles[next_idx2]) {
                        const char* p2 = reinterpret_cast<const char*>(chunked_tiles[next_idx2]);
                        __builtin_prefetch(p2, 1, 3); __builtin_prefetch(p2+64, 1, 3);
                        __builtin_prefetch(p2+128, 1, 3); __builtin_prefetch(p2+192, 1, 3);
                    }
                    if (next_t[0] == 4) {
                        const int next_idx3 = next_t[6];
                        if (next_idx3 >= 0 && next_idx3 < ntiles_bounds && chunked_tiles[next_idx3]) {
                            const char* p3 = reinterpret_cast<const char*>(chunked_tiles[next_idx3]);
                            __builtin_prefetch(p3, 1, 3); __builtin_prefetch(p3+64, 1, 3);
                        }
                    }
                    __builtin_prefetch(&semisparse_meta[next_idx1], 0, 3);
                    __builtin_prefetch(&semisparse_meta[next_idx2], 0, 3);
                }

                switch (myroutine) {

                    case 4: { // DGEMM — C(k,m) -= A(n,k)^T * B(n,m)
                        double* __restrict__ chunk_a = chunked_tiles[index1];
                        double* __restrict__ chunk_b = chunked_tiles[index2];
                        double* __restrict__ chunk_out = chunked_tiles[index3];

                        const sTiles::SemisparseTileMetaCore& semi_a = semisparse_meta[index1];
                        const sTiles::SemisparseTileMetaCore& semi_b = semisparse_meta[index2];

                        const int cols_a = semi_a.sa;
                        const int cols_b = semi_b.sa;
                        const int rows_a = tile_meta[index1].height;
                        const int rows_out = tile_meta[index3].height;

                        // Use precomputed path flag + slot_map if available
                        if (has_scatter_info) {
                            const int si_base = idx * 2;
                            const int64_t data_off  = scatter_index[si_base];
                            const int path_flag = static_cast<int>(scatter_index[si_base + 1]);

                            if (path_flag == 0) {
                                // Direct GEMM: a_contig && b_contig && out_full
                                const int a_start = semi_a.fa;
                                const int b_start = semi_b.fa;
                                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                            cols_a, cols_b, rows_a, mzone,
                                            chunk_a, rows_a, chunk_b, rows_a,
                                            1.0, chunk_out + a_start + static_cast<std::size_t>(b_start) * rows_out, rows_out);

                            } else if (path_flag == 1) {
                                // Fused contiguous: a_contig, small tiles
                                const int a_start = semi_a.fa;
                                const int32_t* slots = scatter_packed + data_off;
                                for (int j = 0; j < cols_b; ++j) {
                                    const int slot = slots[j];
                                    if (slot < 0) continue;
                                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                    const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                    for (int i = 0; i < cols_a; ++i) {
                                        const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                        double dot = 0.0;
                                        for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                        out_col[a_start + i] -= dot;
                                    }
                                }

                            } else if (path_flag == 2) {
                                // Fused scatter: non-contiguous a, small tiles
                                const int* __restrict__ aind_a = semi_a.aind.data();
                                const int32_t* slots = scatter_packed + data_off;
                                for (int j = 0; j < cols_b; ++j) {
                                    const int slot = slots[j];
                                    if (slot < 0) continue;
                                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                    const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                    for (int i = 0; i < cols_a; ++i) {
                                        const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                        double dot = 0.0;
                                        for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                        out_col[aind_a[i]] -= dot;
                                    }
                                }

                            } else if (path_flag == 3) {
                                // BLAS + contiguous scatter
                                const int ld_tmp = cols_a;
                                const int a_start = semi_a.fa;
                                const int32_t* slots = scatter_packed + data_off;
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                   cols_a, cols_b, rows_a,
                                                   mzone, chunk_a, rows_a, chunk_b, rows_a,
                                                   0.0, tmp_tile, ld_tmp);
                                for (int j = 0; j < cols_b; ++j) {
                                    const int slot = slots[j];
                                    if (slot < 0) continue;
                                    cblas_daxpy(cols_a, 1.0, tmp_tile + j * ld_tmp, 1,
                                                chunk_out + static_cast<std::size_t>(slot) * rows_out + a_start, 1);
                                }

                            } else if (path_flag == 4) {
                                // BLAS + indexed scatter
                                const int ld_tmp = cols_a;
                                const int* __restrict__ aind_a = semi_a.aind.data();
                                const int32_t* slots = scatter_packed + data_off;
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                   cols_a, cols_b, rows_a,
                                                   mzone, chunk_a, rows_a, chunk_b, rows_a,
                                                   0.0, tmp_tile, ld_tmp);
                                for (int j = 0; j < cols_b; ++j) {
                                    const int slot = slots[j];
                                    if (slot < 0) continue;
                                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                    const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                    int i = 0;
                                    for (; i + 3 < cols_a; i += 4) {
                                        out_col[aind_a[i]]     += tmp_col[i];
                                        out_col[aind_a[i + 1]] += tmp_col[i + 1];
                                        out_col[aind_a[i + 2]] += tmp_col[i + 2];
                                        out_col[aind_a[i + 3]] += tmp_col[i + 3];
                                    }
                                    for (; i < cols_a; ++i) out_col[aind_a[i]] += tmp_col[i];
                                }
                            }
                        } else {
                            // Fallback: original runtime path selection
                            const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index3];
                            const int* __restrict__ aind_a = semi_a.aind.data();
                            const int* __restrict__ aind_b = semi_b.aind.data();
                            const int* __restrict__ acol_map = semi_out.acol.data();
                            const bool a_contig = semi_a.is_contiguous;
                            const bool b_contig = semi_b.is_contiguous;
                            const bool out_full = semi_out.is_full_width;

                            if (a_contig && b_contig && out_full) {
                                const int a_start = semi_a.fa;
                                const int b_start = semi_b.fa;
                                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                            cols_a, cols_b, rows_a, mzone,
                                            chunk_a, rows_a, chunk_b, rows_a,
                                            1.0, chunk_out + a_start + static_cast<std::size_t>(b_start) * rows_out, rows_out);
                            } else if (cols_a <= FUSE_THRESHOLD && cols_b <= FUSE_THRESHOLD) {
                                if (a_contig) {
                                    const int a_start = semi_a.fa;
                                    for (int j = 0; j < cols_b; ++j) {
                                        const int slot = acol_map[aind_b[j]];
                                        if (slot < 0) continue;
                                        double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                        const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                        for (int i = 0; i < cols_a; ++i) {
                                            const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                            double dot = 0.0;
                                            for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                            out_col[a_start + i] -= dot;
                                        }
                                    }
                                } else {
                                    for (int j = 0; j < cols_b; ++j) {
                                        const int slot = acol_map[aind_b[j]];
                                        if (slot < 0) continue;
                                        double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                        const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                        for (int i = 0; i < cols_a; ++i) {
                                            const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                            double dot = 0.0;
                                            for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                            out_col[aind_a[i]] -= dot;
                                        }
                                    }
                                }
                            } else {
                                const int ld_tmp = cols_a;
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                   cols_a, cols_b, rows_a,
                                                   mzone, chunk_a, rows_a, chunk_b, rows_a,
                                                   0.0, tmp_tile, ld_tmp);
                                if (a_contig) {
                                    const int a_start = semi_a.fa;
                                    // Unrolled scatter (contiguous) — avoids daxpy call overhead
                                    for (int j = 0; j < cols_b; ++j) {
                                        const int slot = acol_map[aind_b[j]];
                                        if (slot < 0) continue;
                                        double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out + a_start;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        int i = 0;
                                        for (; i + 3 < cols_a; i += 4) {
                                            out_col[i]     += tmp_col[i];
                                            out_col[i + 1] += tmp_col[i + 1];
                                            out_col[i + 2] += tmp_col[i + 2];
                                            out_col[i + 3] += tmp_col[i + 3];
                                        }
                                        for (; i < cols_a; ++i) out_col[i] += tmp_col[i];
                                    }
                                } else {
                                    for (int j = 0; j < cols_b; ++j) {
                                        const int slot = acol_map[aind_b[j]];
                                        if (slot < 0) continue;
                                        double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        int i = 0;
                                        for (; i + 3 < cols_a; i += 4) {
                                            out_col[aind_a[i]]     += tmp_col[i];
                                            out_col[aind_a[i + 1]] += tmp_col[i + 1];
                                            out_col[aind_a[i + 2]] += tmp_col[i + 2];
                                            out_col[aind_a[i + 3]] += tmp_col[i + 3];
                                        }
                                        for (; i < cols_a; ++i) out_col[aind_a[i]] += tmp_col[i];
                                    }
                                }
                            }
                        }
                        break;
                    }

                    case 1: { // DPOTRF — banded Cholesky on diagonal tile
                        double* __restrict__ tile_out = chunked_tiles[index1];
                        if (tile_out) {
                            const sTiles::SemisparseTileMetaCore& semi = semisparse_meta[index1];
                            const sTiles::TileMetaCore& meta = tile_meta[index1];
                            sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(meta.height, semi.upper_bw, tile_out);
                            if (ssstatus != sTiles::StatusCode::Success) {
                                std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                                return;
                            }
                        }
                        break;
                    }

                    case 2: { // DSYRK — symmetric rank-k update into banded diagonal
                        double* __restrict__ tile_in = chunked_tiles[index1];
                        double* __restrict__ tile_out = chunked_tiles[index2];

                        if (tile_in && tile_out) {
                            const sTiles::SemisparseTileMetaCore& semi_in = semisparse_meta[index1];
                            const int active_cols = semi_in.sa;

                            if (active_cols > 0) {
                                const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index2];
                                const int diag_cols = semi_out.upper_bw + 1;
                                const int tile_size = tiledMatrix->tile_size;
                                const int rows_in = (tile_meta[index1].height > 0) ? tile_meta[index1].height : tile_size;
                                const int base_row = diag_cols - 1;

                                const bool aind_contig = semi_in.is_contiguous;

                                if (active_cols <= FUSE_THRESHOLD && aind_contig && active_cols <= diag_cols) {
                                    const int aind_start = semi_in.fa;
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_start + j;
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                        for (int i = 0; i <= j; ++i) {
                                            const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                            double dot = 0.0;
                                            for (int r = 0; r < rows_in; ++r) dot += col_i[r] * col_j[r];
                                            tile_out[base_row - (j - i) + col_base] -= dot;
                                        }
                                    }
                                } else if (active_cols <= FUSE_THRESHOLD) {
                                    const int* __restrict__ aind = semi_in.aind.data();
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind[j];
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                        for (int i = 0; i <= j; ++i) {
                                            const int band = col_idx - aind[i];
                                            if (band < diag_cols) {
                                                const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                                double dot = 0.0;
                                                for (int r = 0; r < rows_in; ++r) dot += col_i[r] * col_j[r];
                                                tile_out[base_row - band + col_base] -= dot;
                                            }
                                        }
                                    }
                                } else if (aind_contig && active_cols <= diag_cols) {
                                    const int ld_tmp = active_cols;
                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                       active_cols, rows_in, mzone,
                                                       tile_in, rows_in, 0.0, tmp_tile, ld_tmp);
                                    const int aind_start = semi_in.fa;
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_start + j;
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        for (int i = 0; i <= j; ++i)
                                            tile_out[base_row - (j - i) + col_base] += tmp_col[i];
                                    }
                                } else {
                                    const int ld_tmp = active_cols;
                                    const int* __restrict__ aind = semi_in.aind.data();
                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                       active_cols, rows_in, mzone,
                                                       tile_in, rows_in, 0.0, tmp_tile, ld_tmp);
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind[j];
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        for (int i = 0; i <= j; ++i) {
                                            const int band = col_idx - aind[i];
                                            if (band < diag_cols)
                                                tile_out[base_row - band + col_base] += tmp_col[i];
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }

                    case 3: { // DTRSM — banded triangular solve
                        double* __restrict__ ss_rhs = chunked_tiles[index1];
                        double* __restrict__ ss_diag = chunked_tiles[index2];

                        if (ss_rhs && ss_diag) {
                            const sTiles::SemisparseTileMetaCore& semi_rhs = semisparse_meta[index1];
                            const sTiles::SemisparseTileMetaCore& semi_diag = semisparse_meta[index2];
                            const sTiles::TileMetaCore& meta_rhs = tile_meta[index1];
                            const sTiles::TileMetaCore& meta_diag = tile_meta[index2];

                            const int active_cols = semi_rhs.sa;
                            if (active_cols > 0) {
                                const int rows_rhs = meta_rhs.height;
                                const int diag_dim = meta_diag.height;
                                const int kd = semi_diag.upper_bw;

                                lapack_int info = LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                                                 diag_dim, kd, active_cols,
                                                                 ss_diag, kd + 1, ss_rhs, rows_rhs);
                                if (info != 0) {
                                    std::fprintf(stderr, "sTiles error: LAPACKE_dtbtrs failed with info=%d (tile m=%d, k=%d)\n",
                                                 info, m, k);
                                    return;
                                }
                            }
                        }
                        break;
                    }

                    default:
                        break;
                }
            }
        }

        void dpotrf_expansion_from_chol_tasks_semisparse_imp4_serial(TiledMatrix *tiledMatrix, stiles_context_t *stile) {
            (void)stile; // unused in serial version

            const int rank = STILES_RANK;

            double** __restrict__ const chunked_tiles = tiledMatrix->chunkedDenseTiles;
            const sTiles::SemisparseTileMetaCore* __restrict__ const semisparse_meta = tiledMatrix->semisparseTileMetaCore;
            const sTiles::TileMetaCore* __restrict__ const tile_meta = tiledMatrix->tileMetaCore;

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const int start = 0;
            const int end = static_cast<int>(tasks.size());

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[semisparse_imp3_serial] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* __restrict__ tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
            const double mzone = -1.0;

            constexpr int FUSE_THRESHOLD = 16;

            // Loop-invariant fields hoisted out of the per-task body.
            const int ntiles_bounds = tiledMatrix->numActiveTiles;
            const int tile_size = tiledMatrix->tile_size;

            // Precomputed scatter info — required (param[7]==1, default in fast mode).
            // Case 4's dispatch reads scatter_index/scatter_packed unconditionally; the
            // runtime fallback was removed because build_chol_scatter_info is always
            // called by sTiles_preprocess_group_fastmode when param[7]==1.
            const int64_t* scatter_index = tiledMatrix->chol_scatter_index->data();
            const int32_t* scatter_packed = tiledMatrix->chol_scatter_packed->data();

            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                // Prefetch: focus on first cache line of each tile (metadata is in regs by the
                // time we touch the body), plus indirection arrays (scatter_packed, aind) which
                // are unpredictable loads and benefit most from prefetch. _MM_HINT_T1 (==2) lands
                // in L2 to avoid L1 thrash with the current task's working set.
                if (idx + 1 < end) {
                    const std::array<int,7> &next_t = tasks[idx + 1];
                    const int next_routine = next_t[0];
                    const int next_idx1 = next_t[4];
                    const int next_idx2 = next_t[5];
                    if (next_idx1 >= 0 && next_idx1 < ntiles_bounds && chunked_tiles[next_idx1]) {
                        __builtin_prefetch(chunked_tiles[next_idx1], 0, 2);
                    }
                    if (next_idx2 >= 0 && next_idx2 < ntiles_bounds && chunked_tiles[next_idx2]) {
                        __builtin_prefetch(chunked_tiles[next_idx2], 1, 2);
                    }
                    if (next_routine == 4) {
                        const int next_idx3 = next_t[6];
                        if (next_idx3 >= 0 && next_idx3 < ntiles_bounds && chunked_tiles[next_idx3]) {
                            __builtin_prefetch(chunked_tiles[next_idx3], 1, 2);
                        }
                        const int64_t next_si = static_cast<int64_t>(idx + 1) * 2;
                        __builtin_prefetch(&scatter_index[next_si], 0, 3);
                        const int64_t next_data_off = scatter_index[next_si];
                        if (next_data_off >= 0) {
                            __builtin_prefetch(scatter_packed + next_data_off, 0, 3);
                        }
                    }
                    __builtin_prefetch(&semisparse_meta[next_idx1], 0, 3);
                    __builtin_prefetch(&semisparse_meta[next_idx2], 0, 3);
                    if (next_idx1 >= 0 && next_idx1 < ntiles_bounds) {
                        const int* aind_ptr = semisparse_meta[next_idx1].aind.data();
                        if (aind_ptr) __builtin_prefetch(aind_ptr, 0, 2);
                    }
                }

                switch (myroutine) {

                    case 4: { // DGEMM — C(k,m) -= A(n,k)^T * B(n,m)
                        double* __restrict__ chunk_a = chunked_tiles[index1];
                        double* __restrict__ chunk_b = chunked_tiles[index2];
                        double* __restrict__ chunk_out = chunked_tiles[index3];

                        const sTiles::SemisparseTileMetaCore& semi_a = semisparse_meta[index1];
                        const sTiles::SemisparseTileMetaCore& semi_b = semisparse_meta[index2];

                        const int cols_a = semi_a.sa;
                        const int cols_b = semi_b.sa;
                        const int rows_a = tile_meta[index1].height;
                        const int rows_out = tile_meta[index3].height;

                        // ── Dense path: precomputed path_flag dispatch ──
                        // Precondition: param[7]==1 (default in fast mode), which guarantees
                        // build_chol_scatter_info has populated scatter_index/scatter_packed.
                        const int si_base = idx * 2;
                        const int64_t data_off  = scatter_index[si_base];
                        const int path_flag = static_cast<int>(scatter_index[si_base + 1]);

                        if (path_flag == 0) {
                            // Direct GEMM: a_contig && b_contig && out_full
                            const int a_start = semi_a.fa;
                            const int b_start = semi_b.fa;
                            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                        cols_a, cols_b, rows_a, mzone,
                                        chunk_a, rows_a, chunk_b, rows_a,
                                        1.0, chunk_out + a_start + static_cast<std::size_t>(b_start) * rows_out, rows_out);

                        } else if (path_flag == 1) {
                            // Fused contiguous: vectorized dot + 2-i blocking (reuses col_b_j register)
                            const int a_start = semi_a.fa;
                            const int32_t* slots = scatter_packed + data_off;
                            const int cols_a_pair = cols_a & ~1;
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out + a_start;
                                const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                int i = 0;
                                for (; i < cols_a_pair; i += 2) {
                                    const double* __restrict__ col_a_i0 = chunk_a + static_cast<std::size_t>(i)     * rows_a;
                                    const double* __restrict__ col_a_i1 = chunk_a + static_cast<std::size_t>(i + 1) * rows_a;
                                    double dot0 = 0.0, dot1 = 0.0;
                                    #pragma omp simd reduction(+:dot0,dot1)
                                    for (int r = 0; r < rows_a; ++r) {
                                        const double bv = col_b_j[r];
                                        dot0 += col_a_i0[r] * bv;
                                        dot1 += col_a_i1[r] * bv;
                                    }
                                    out_col[i]     -= dot0;
                                    out_col[i + 1] -= dot1;
                                }
                                if (i < cols_a) {
                                    const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                    double dot = 0.0;
                                    #pragma omp simd reduction(+:dot)
                                    for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                    out_col[i] -= dot;
                                }
                            }

                        } else if (path_flag == 2) {
                            // Fused scatter: vectorized dot + 2-i blocking
                            const int* __restrict__ aind_a = semi_a.aind.data();
                            const int32_t* slots = scatter_packed + data_off;
                            const int cols_a_pair = cols_a & ~1;
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                int i = 0;
                                for (; i < cols_a_pair; i += 2) {
                                    const double* __restrict__ col_a_i0 = chunk_a + static_cast<std::size_t>(i)     * rows_a;
                                    const double* __restrict__ col_a_i1 = chunk_a + static_cast<std::size_t>(i + 1) * rows_a;
                                    double dot0 = 0.0, dot1 = 0.0;
                                    #pragma omp simd reduction(+:dot0,dot1)
                                    for (int r = 0; r < rows_a; ++r) {
                                        const double bv = col_b_j[r];
                                        dot0 += col_a_i0[r] * bv;
                                        dot1 += col_a_i1[r] * bv;
                                    }
                                    out_col[aind_a[i]]     -= dot0;
                                    out_col[aind_a[i + 1]] -= dot1;
                                }
                                if (i < cols_a) {
                                    const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                    double dot = 0.0;
                                    #pragma omp simd reduction(+:dot)
                                    for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                    out_col[aind_a[i]] -= dot;
                                }
                            }

                        } else if (path_flag == 3) {
                            // BLAS + contiguous scatter — inline SIMD axpy avoids cblas_daxpy call overhead
                            const int ld_tmp = cols_a;
                            const int a_start = semi_a.fa;
                            const int32_t* slots = scatter_packed + data_off;
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                               cols_a, cols_b, rows_a,
                                               mzone, chunk_a, rows_a, chunk_b, rows_a,
                                               0.0, tmp_tile, ld_tmp);
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out + a_start;
                                const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                #pragma omp simd
                                for (int i = 0; i < cols_a; ++i) out_col[i] += tmp_col[i];
                            }

                        } else if (path_flag == 4) {
                            // BLAS + indexed scatter
                            const int ld_tmp = cols_a;
                            const int* __restrict__ aind_a = semi_a.aind.data();
                            const int32_t* slots = scatter_packed + data_off;
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                               cols_a, cols_b, rows_a,
                                               mzone, chunk_a, rows_a, chunk_b, rows_a,
                                               0.0, tmp_tile, ld_tmp);
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                int i = 0;
                                for (; i + 3 < cols_a; i += 4) {
                                    out_col[aind_a[i]]     += tmp_col[i];
                                    out_col[aind_a[i + 1]] += tmp_col[i + 1];
                                    out_col[aind_a[i + 2]] += tmp_col[i + 2];
                                    out_col[aind_a[i + 3]] += tmp_col[i + 3];
                                }
                                for (; i < cols_a; ++i) out_col[aind_a[i]] += tmp_col[i];
                            }
                        }
                        break;
                    }

                    case 1: { // DPOTRF — banded Cholesky on diagonal tile
                        // Diagonal tile: chunked_tiles[index1] is allocated with elems = height*(upper_bw+1),
                        // so it's never null for an active diagonal — no null guard needed.
                        double* __restrict__ tile_out = chunked_tiles[index1];
                        const sTiles::SemisparseTileMetaCore& semi = semisparse_meta[index1];
                        const sTiles::TileMetaCore& meta = tile_meta[index1];
                        sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(meta.height, semi.upper_bw, tile_out);
                        if (ssstatus != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", t[2]);
                            return;
                        }
                        break;
                    }

                    case 2: { // DSYRK — symmetric rank-k update into banded diagonal
                        // chunked_tiles[idx]==nullptr ⇔ semi.sa==0 (see allocate_semisparse_tiles),
                        // so the active_cols>0 check subsumes a tile_in null guard. tile_out is the
                        // diagonal, never null.
                        const sTiles::SemisparseTileMetaCore& semi_in = semisparse_meta[index1];
                        const int active_cols = semi_in.sa;
                        if (active_cols > 0) {
                            double* __restrict__ tile_in  = chunked_tiles[index1];
                            double* __restrict__ tile_out = chunked_tiles[index2];
                            const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index2];
                            const int diag_cols = semi_out.upper_bw + 1;
                            const int rows_in = (tile_meta[index1].height > 0) ? tile_meta[index1].height : tile_size;
                            const int base_row = diag_cols - 1;

                            const bool aind_contig = semi_in.is_contiguous;

                            if (active_cols <= FUSE_THRESHOLD && aind_contig && active_cols <= diag_cols) {
                                const int aind_start = semi_in.fa;
                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = aind_start + j;
                                    const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                    const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                    for (int i = 0; i <= j; ++i) {
                                        const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                        double dot = 0.0;
                                        #pragma omp simd reduction(+:dot)
                                        for (int r = 0; r < rows_in; ++r) dot += col_i[r] * col_j[r];
                                        tile_out[base_row - (j - i) + col_base] -= dot;
                                    }
                                }
                            } else if (active_cols <= FUSE_THRESHOLD) {
                                const int* __restrict__ aind = semi_in.aind.data();
                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = aind[j];
                                    const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                    const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                    for (int i = 0; i <= j; ++i) {
                                        const int band = col_idx - aind[i];
                                        if (band < diag_cols) {
                                            const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                            double dot = 0.0;
                                            #pragma omp simd reduction(+:dot)
                                            for (int r = 0; r < rows_in; ++r) dot += col_i[r] * col_j[r];
                                            tile_out[base_row - band + col_base] -= dot;
                                        }
                                    }
                                }
                            } else if (aind_contig && active_cols <= diag_cols) {
                                const int ld_tmp = active_cols;
                                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                   active_cols, rows_in, mzone,
                                                   tile_in, rows_in, 0.0, tmp_tile, ld_tmp);
                                const int aind_start = semi_in.fa;
                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = aind_start + j;
                                    const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                    const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                    for (int i = 0; i <= j; ++i)
                                        tile_out[base_row - (j - i) + col_base] += tmp_col[i];
                                }
                            } else {
                                const int ld_tmp = active_cols;
                                const int* __restrict__ aind = semi_in.aind.data();
                                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                   active_cols, rows_in, mzone,
                                                   tile_in, rows_in, 0.0, tmp_tile, ld_tmp);
                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = aind[j];
                                    const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                    const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                    for (int i = 0; i <= j; ++i) {
                                        const int band = col_idx - aind[i];
                                        if (band < diag_cols)
                                            tile_out[base_row - band + col_base] += tmp_col[i];
                                    }
                                }
                            }
                        }
                        break;
                    }

                    case 3: { // DTRSM — banded triangular solve
                        // Same invariant as case 2: ss_rhs is null iff active_cols==0.
                        // ss_diag is the diagonal, never null. Single guard suffices.
                        const sTiles::SemisparseTileMetaCore& semi_rhs = semisparse_meta[index1];
                        const int active_cols = semi_rhs.sa;
                        if (active_cols > 0) {
                            double* __restrict__ ss_rhs  = chunked_tiles[index1];
                            double* __restrict__ ss_diag = chunked_tiles[index2];
                            const sTiles::SemisparseTileMetaCore& semi_diag = semisparse_meta[index2];
                            const int rows_rhs = tile_meta[index1].height;
                            const int diag_dim = tile_meta[index2].height;
                            const int kd = semi_diag.upper_bw;

                            lapack_int info = LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                                             diag_dim, kd, active_cols,
                                                             ss_diag, kd + 1, ss_rhs, rows_rhs);
                            if (info != 0) {
                                std::fprintf(stderr, "sTiles error: LAPACKE_dtbtrs failed with info=%d (tile m=%d, k=%d)\n",
                                             info, t[1], t[2]);
                                return;
                            }
                        }
                        break;
                    }

                    default:
                        break;
                }
            }
        }

        // ── Cold helpers for the sparse fast paths in imp3_serial_and_sparse ──
        // These are taken only when a tile's L-fill density is below the gate
        // (case 4: ~10%, case 2: ~5%). On dense-dominated workloads they never
        // fire; outlining them keeps the hot dispatch's i-cache footprint tight.

        STILES_COLD
        static void dgemm_sparse_csc_fastpath(
                double* __restrict__ chunk_a,
                double* __restrict__ chunk_b,
                double* __restrict__ chunk_out,
                const int* __restrict__ a_cp,
                const int* __restrict__ a_ri,
                const int* __restrict__ aind_a,
                const int* __restrict__ aind_b,
                const int* __restrict__ acol_map,
                int cols_a, int cols_b, int rows_a, int rows_out,
                int a_start, int b_start,
                bool a_contig, bool b_contig, bool out_full)
        {
            if (a_contig && b_contig && out_full) {
                // Variant matching path 0: contiguous block write into chunk_out.
                for (int j = 0; j < cols_b; ++j) {
                    const double* __restrict__ col_b = chunk_b + static_cast<std::size_t>(j) * rows_a;
                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(b_start + j) * rows_out;
                    for (int i = 0; i < cols_a; ++i) {
                        const double* __restrict__ col_a = chunk_a + static_cast<std::size_t>(i) * rows_a;
                        const int p_lo = a_cp[i];
                        const int p_hi = a_cp[i + 1];
                        double dot = 0.0;
                        for (int p = p_lo; p < p_hi; ++p) {
                            const int r = a_ri[p];
                            dot += col_a[r] * col_b[r];
                        }
                        out_col[a_start + i] -= dot;
                    }
                }
            } else if (a_contig) {
                // Variant matching paths 1/3: contiguous A rows, indexed B slot.
                for (int j = 0; j < cols_b; ++j) {
                    const int slot = acol_map[aind_b[j]];
                    if (slot < 0) continue;
                    const double* __restrict__ col_b = chunk_b + static_cast<std::size_t>(j) * rows_a;
                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                    for (int i = 0; i < cols_a; ++i) {
                        const double* __restrict__ col_a = chunk_a + static_cast<std::size_t>(i) * rows_a;
                        const int p_lo = a_cp[i];
                        const int p_hi = a_cp[i + 1];
                        double dot = 0.0;
                        for (int p = p_lo; p < p_hi; ++p) {
                            const int r = a_ri[p];
                            dot += col_a[r] * col_b[r];
                        }
                        out_col[a_start + i] -= dot;
                    }
                }
            } else {
                // Variant matching paths 2/4: indexed A rows, indexed B slot.
                for (int j = 0; j < cols_b; ++j) {
                    const int slot = acol_map[aind_b[j]];
                    if (slot < 0) continue;
                    const double* __restrict__ col_b = chunk_b + static_cast<std::size_t>(j) * rows_a;
                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                    for (int i = 0; i < cols_a; ++i) {
                        const double* __restrict__ col_a = chunk_a + static_cast<std::size_t>(i) * rows_a;
                        const int p_lo = a_cp[i];
                        const int p_hi = a_cp[i + 1];
                        double dot = 0.0;
                        for (int p = p_lo; p < p_hi; ++p) {
                            const int r = a_ri[p];
                            dot += col_a[r] * col_b[r];
                        }
                        out_col[aind_a[i]] -= dot;
                    }
                }
            }
            (void)b_contig; // currently only used to select the first variant
        }

        STILES_COLD
        static void dsyrk_sparse_csc_fastpath(
                double* __restrict__ tile_in,
                double* __restrict__ tile_out,
                const int* __restrict__ a_cp,
                const int* __restrict__ a_ri,
                const int* __restrict__ aind,
                int active_cols, int rows_in, int diag_cols, int base_row,
                int aind_start,
                bool aind_contig)
        {
            if (aind_contig && active_cols <= diag_cols) {
                // Variant matching the contiguous SYRK paths (fused or BLAS+scatter).
                // active_cols <= diag_cols is required so that band = (j - i) stays
                // inside the diagonal tile's upper-banded layout.
                for (int j = 0; j < active_cols; ++j) {
                    const int col_idx = aind_start + j;
                    const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                    const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                    for (int i = 0; i <= j; ++i) {
                        const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                        const int p_lo = a_cp[i];
                        const int p_hi = a_cp[i + 1];
                        double dot = 0.0;
                        for (int p = p_lo; p < p_hi; ++p) {
                            const int r = a_ri[p];
                            dot += col_i[r] * col_j[r];
                        }
                        tile_out[base_row - (j - i) + col_base] -= dot;
                    }
                }
            } else {
                // Variant matching the indexed SYRK paths (band check required).
                for (int j = 0; j < active_cols; ++j) {
                    const int col_idx = aind[j];
                    const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                    const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                    for (int i = 0; i <= j; ++i) {
                        const int band = col_idx - aind[i];
                        if (band >= diag_cols) continue;
                        const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                        const int p_lo = a_cp[i];
                        const int p_hi = a_cp[i + 1];
                        double dot = 0.0;
                        for (int p = p_lo; p < p_hi; ++p) {
                            const int r = a_ri[p];
                            dot += col_i[r] * col_j[r];
                        }
                        tile_out[base_row - band + col_base] -= dot;
                    }
                }
            }
        }

        void dpotrf_expansion_from_chol_tasks_semisparse_imp3_serial_and_sparse(TiledMatrix *tiledMatrix, stiles_context_t *stile) {
            (void)stile; // unused in serial version

            const int rank = STILES_RANK;

            double** __restrict__ const chunked_tiles = tiledMatrix->chunkedDenseTiles;
            const sTiles::SemisparseTileMetaCore* __restrict__ const semisparse_meta = tiledMatrix->semisparseTileMetaCore;
            const sTiles::TileMetaCore* __restrict__ const tile_meta = tiledMatrix->tileMetaCore;

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const int start = 0;
            const int end = static_cast<int>(tasks.size());

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[semisparse_imp3_serial] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* __restrict__ tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
            const double mzone = -1.0;

            constexpr int FUSE_THRESHOLD = 16;

            // Loop-invariant fields hoisted out of the per-task body.
            const int ntiles_bounds = tiledMatrix->numActiveTiles;
            const int tile_size = tiledMatrix->tile_size;

            // Precomputed scatter info — required (param[7]==1, default in fast mode).
            // Case 4's dispatch reads scatter_index/scatter_packed unconditionally; the
            // runtime fallback was removed because build_chol_scatter_info is always
            // called by sTiles_preprocess_group_fastmode when param[7]==1.
            const int64_t* scatter_index = tiledMatrix->chol_scatter_index->data();
            const int32_t* scatter_packed = tiledMatrix->chol_scatter_packed->data();

            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                // Prefetching
                if (idx + 1 < end) {
                    const std::array<int,7> &next_t = tasks[idx + 1];
                    const int next_idx1 = next_t[4];
                    const int next_idx2 = next_t[5];
                    if (next_idx1 >= 0 && next_idx1 < ntiles_bounds && chunked_tiles[next_idx1]) {
                        const char* p1 = reinterpret_cast<const char*>(chunked_tiles[next_idx1]);
                        __builtin_prefetch(p1, 0, 3); __builtin_prefetch(p1+64, 0, 3);
                        __builtin_prefetch(p1+128, 0, 3); __builtin_prefetch(p1+192, 0, 3);
                    }
                    if (next_idx2 >= 0 && next_idx2 < ntiles_bounds && chunked_tiles[next_idx2]) {
                        const char* p2 = reinterpret_cast<const char*>(chunked_tiles[next_idx2]);
                        __builtin_prefetch(p2, 1, 3); __builtin_prefetch(p2+64, 1, 3);
                        __builtin_prefetch(p2+128, 1, 3); __builtin_prefetch(p2+192, 1, 3);
                    }
                    if (next_t[0] == 4) {
                        const int next_idx3 = next_t[6];
                        if (next_idx3 >= 0 && next_idx3 < ntiles_bounds && chunked_tiles[next_idx3]) {
                            const char* p3 = reinterpret_cast<const char*>(chunked_tiles[next_idx3]);
                            __builtin_prefetch(p3, 1, 3); __builtin_prefetch(p3+64, 1, 3);
                        }
                    }
                    __builtin_prefetch(&semisparse_meta[next_idx1], 0, 3);
                    __builtin_prefetch(&semisparse_meta[next_idx2], 0, 3);
                }

                switch (myroutine) {

                    case 4: { // DGEMM — C(k,m) -= A(n,k)^T * B(n,m)
                        // Hoisted: loaded once, used by both sparse fast path and dense dispatch.
                        double* __restrict__ chunk_a = chunked_tiles[index1];
                        double* __restrict__ chunk_b = chunked_tiles[index2];
                        double* __restrict__ chunk_out = chunked_tiles[index3];

                        const sTiles::SemisparseTileMetaCore& semi_a = semisparse_meta[index1];
                        const sTiles::SemisparseTileMetaCore& semi_b = semisparse_meta[index2];

                        const int cols_a = semi_a.sa;
                        const int cols_b = semi_b.sa;
                        const int rows_a = tile_meta[index1].height;
                        const int rows_out = tile_meta[index3].height;

                        // ── Sparse fast path (cold): drive inner reduction off A's L-derived CSC ──
                        // Self-disabling when CSC is absent (gate evaluates false). The actual
                        // reduction lives in dgemm_sparse_csc_fastpath() — marked STILES_COLD so
                        // the compiler outlines it into .text.unlikely, keeping the hot dispatch
                        // i-cache footprint tight on dense-dominated workloads.
                        // __builtin_expect(..., 0): rare branch on most matrices.
                        if (__builtin_expect(
                                rows_a > 0 && cols_a > 0 && cols_b > 0 &&
                                !semi_a.csc_colptr.empty() && semi_a.csc_nnz > 0 &&
                                static_cast<long long>(semi_a.csc_nnz) * 10 <
                                    static_cast<long long>(rows_a) * cols_a, 0))
                        {
                            const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index3];
                            dgemm_sparse_csc_fastpath(
                                chunk_a, chunk_b, chunk_out,
                                semi_a.csc_colptr.data(), semi_a.csc_rowind.data(),
                                semi_a.aind.data(), semi_b.aind.data(), semi_out.acol.data(),
                                cols_a, cols_b, rows_a, rows_out,
                                semi_a.fa, semi_b.fa,
                                semi_a.is_contiguous, semi_b.is_contiguous, semi_out.is_full_width);
                            break;
                        }

                        // ── Dense path: precomputed path_flag dispatch ──
                        // The runtime fallback (if has_scatter_info==false) was removed.
                        // Precondition: param[7]==1 (default in fast mode), which guarantees
                        // build_chol_scatter_info has populated scatter_index/scatter_packed.
                        const int si_base = idx * 2;
                        const int64_t data_off  = scatter_index[si_base];
                        const int path_flag = static_cast<int>(scatter_index[si_base + 1]);

                        if (path_flag == 0) {
                            // Direct GEMM: a_contig && b_contig && out_full
                            const int a_start = semi_a.fa;
                            const int b_start = semi_b.fa;
                            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                        cols_a, cols_b, rows_a, mzone,
                                        chunk_a, rows_a, chunk_b, rows_a,
                                        1.0, chunk_out + a_start + static_cast<std::size_t>(b_start) * rows_out, rows_out);

                        } else if (path_flag == 1) {
                            // Fused contiguous: a_contig, small tiles
                            const int a_start = semi_a.fa;
                            const int32_t* slots = scatter_packed + data_off;
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                for (int i = 0; i < cols_a; ++i) {
                                    const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                    double dot = 0.0;
                                    for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                    out_col[a_start + i] -= dot;
                                }
                            }

                        } else if (path_flag == 2) {
                            // Fused scatter: non-contiguous a, small tiles
                            const int* __restrict__ aind_a = semi_a.aind.data();
                            const int32_t* slots = scatter_packed + data_off;
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                for (int i = 0; i < cols_a; ++i) {
                                    const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                    double dot = 0.0;
                                    for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                    out_col[aind_a[i]] -= dot;
                                }
                            }

                        } else if (path_flag == 3) {
                            // BLAS + contiguous scatter
                            const int ld_tmp = cols_a;
                            const int a_start = semi_a.fa;
                            const int32_t* slots = scatter_packed + data_off;
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                               cols_a, cols_b, rows_a,
                                               mzone, chunk_a, rows_a, chunk_b, rows_a,
                                               0.0, tmp_tile, ld_tmp);
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                cblas_daxpy(cols_a, 1.0, tmp_tile + j * ld_tmp, 1,
                                            chunk_out + static_cast<std::size_t>(slot) * rows_out + a_start, 1);
                            }

                        } else if (path_flag == 4) {
                            // BLAS + indexed scatter
                            const int ld_tmp = cols_a;
                            const int* __restrict__ aind_a = semi_a.aind.data();
                            const int32_t* slots = scatter_packed + data_off;
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                               cols_a, cols_b, rows_a,
                                               mzone, chunk_a, rows_a, chunk_b, rows_a,
                                               0.0, tmp_tile, ld_tmp);
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                int i = 0;
                                for (; i + 3 < cols_a; i += 4) {
                                    out_col[aind_a[i]]     += tmp_col[i];
                                    out_col[aind_a[i + 1]] += tmp_col[i + 1];
                                    out_col[aind_a[i + 2]] += tmp_col[i + 2];
                                    out_col[aind_a[i + 3]] += tmp_col[i + 3];
                                }
                                for (; i < cols_a; ++i) out_col[aind_a[i]] += tmp_col[i];
                            }
                        }
                        break;
                    }

                    case 1: { // DPOTRF — banded Cholesky on diagonal tile
                        // Diagonal tile: chunked_tiles[index1] is allocated with elems = height*(upper_bw+1),
                        // so it's never null for an active diagonal — no null guard needed.
                        double* __restrict__ tile_out = chunked_tiles[index1];
                        const sTiles::SemisparseTileMetaCore& semi = semisparse_meta[index1];
                        const sTiles::TileMetaCore& meta = tile_meta[index1];
                        sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(meta.height, semi.upper_bw, tile_out);
                        if (ssstatus != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", t[2]);
                            return;
                        }
                        break;
                    }

                    case 2: { // DSYRK — symmetric rank-k update into banded diagonal
                        // chunked_tiles[idx]==nullptr ⇔ semi.sa==0 (see allocate_semisparse_tiles),
                        // so the active_cols>0 check subsumes a tile_in null guard. tile_out is the
                        // diagonal, never null.
                        const sTiles::SemisparseTileMetaCore& semi_in = semisparse_meta[index1];
                        const int active_cols = semi_in.sa;
                        if (active_cols > 0) {
                            double* __restrict__ tile_in  = chunked_tiles[index1];
                            double* __restrict__ tile_out = chunked_tiles[index2];
                            const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index2];
                            const int diag_cols = semi_out.upper_bw + 1;
                            const int rows_in = (tile_meta[index1].height > 0) ? tile_meta[index1].height : tile_size;
                            const int base_row = diag_cols - 1;

                            const bool aind_contig = semi_in.is_contiguous;
                            // ── Sparse fast path (cold): drive inner reduction off the L-derived CSC ──
                            // Same idea as case 4: replace the dense reduction over rows with a walk
                            // over the exact L-fill row pattern of each active column. The actual
                            // reduction lives in dsyrk_sparse_csc_fastpath() — marked STILES_COLD.
                            // Gate: density < ~5% (half of case 4's 10% threshold). SYRK only
                            // computes the upper triangle, so it does ~half the FLOPs of a full
                            // GEMM at the same dimensions — crossover sits at half the density.
                            // __builtin_expect(..., 0): rare branch.
                            if (__builtin_expect(
                                    rows_in > 0 && active_cols > 0 &&
                                    !semi_in.csc_colptr.empty() && semi_in.csc_nnz > 0 &&
                                    static_cast<long long>(semi_in.csc_nnz) * 20 <
                                        static_cast<long long>(rows_in) * active_cols, 0))
                            {
                                dsyrk_sparse_csc_fastpath(
                                    tile_in, tile_out,
                                    semi_in.csc_colptr.data(), semi_in.csc_rowind.data(),
                                    semi_in.aind.data(),
                                    active_cols, rows_in, diag_cols, base_row,
                                    semi_in.fa,
                                    aind_contig);
                            } else if (active_cols <= FUSE_THRESHOLD && aind_contig && active_cols <= diag_cols) {
                                const int aind_start = semi_in.fa;
                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = aind_start + j;
                                    const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                    const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                    for (int i = 0; i <= j; ++i) {
                                        const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                        double dot = 0.0;
                                        for (int r = 0; r < rows_in; ++r) dot += col_i[r] * col_j[r];
                                        tile_out[base_row - (j - i) + col_base] -= dot;
                                    }
                                }
                            } else if (active_cols <= FUSE_THRESHOLD) {
                                const int* __restrict__ aind = semi_in.aind.data();
                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = aind[j];
                                    const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                    const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                    for (int i = 0; i <= j; ++i) {
                                        const int band = col_idx - aind[i];
                                        if (band < diag_cols) {
                                            const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                            double dot = 0.0;
                                            for (int r = 0; r < rows_in; ++r) dot += col_i[r] * col_j[r];
                                            tile_out[base_row - band + col_base] -= dot;
                                        }
                                    }
                                }
                            } else if (aind_contig && active_cols <= diag_cols) {
                                const int ld_tmp = active_cols;
                                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                   active_cols, rows_in, mzone,
                                                   tile_in, rows_in, 0.0, tmp_tile, ld_tmp);
                                const int aind_start = semi_in.fa;
                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = aind_start + j;
                                    const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                    const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                    for (int i = 0; i <= j; ++i)
                                        tile_out[base_row - (j - i) + col_base] += tmp_col[i];
                                }
                            } else {
                                const int ld_tmp = active_cols;
                                const int* __restrict__ aind = semi_in.aind.data();
                                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                   active_cols, rows_in, mzone,
                                                   tile_in, rows_in, 0.0, tmp_tile, ld_tmp);
                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = aind[j];
                                    const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                    const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                    for (int i = 0; i <= j; ++i) {
                                        const int band = col_idx - aind[i];
                                        if (band < diag_cols)
                                            tile_out[base_row - band + col_base] += tmp_col[i];
                                    }
                                }
                            }
                        }
                        break;
                    }

                    case 3: { // DTRSM — banded triangular solve
                        // Same invariant as case 2: ss_rhs is null iff active_cols==0.
                        // ss_diag is the diagonal, never null. Single guard suffices.
                        const sTiles::SemisparseTileMetaCore& semi_rhs = semisparse_meta[index1];
                        const int active_cols = semi_rhs.sa;
                        if (active_cols > 0) {
                            double* __restrict__ ss_rhs  = chunked_tiles[index1];
                            double* __restrict__ ss_diag = chunked_tiles[index2];
                            const sTiles::SemisparseTileMetaCore& semi_diag = semisparse_meta[index2];
                            const int rows_rhs = tile_meta[index1].height;
                            const int diag_dim = tile_meta[index2].height;
                            const int kd = semi_diag.upper_bw;

                            lapack_int info = LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                                             diag_dim, kd, active_cols,
                                                             ss_diag, kd + 1, ss_rhs, rows_rhs);
                            if (info != 0) {
                                std::fprintf(stderr, "sTiles error: LAPACKE_dtbtrs failed with info=%d (tile m=%d, k=%d)\n",
                                             info, t[1], t[2]);
                                return;
                            }
                        }
                        break;
                    }

                    default:
                        break;
                }
            }

        }

        void dpotrf_expansion_from_chol_tasks_semisparse_imp1(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int rank = STILES_RANK;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            // Cache commonly accessed pointers
            double** __restrict__ const chunked_tiles = tiledMatrix->chunkedDenseTiles;
            const sTiles::SemisparseTileMetaCore* __restrict__ const semisparse_meta = tiledMatrix->semisparseTileMetaCore;
            const sTiles::TileMetaCore* __restrict__ const tile_meta = tiledMatrix->tileMetaCore;

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[semisparse_serial] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* __restrict__ tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
            const double mzone = -1.0;

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);

            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                // Prefetch next task's tiles (if not last task)
                if (idx + 1 < end) {
                    const std::array<int,7> &next_t = tasks[idx + 1];
                    const int next_idx1 = next_t[4];
                    const int next_idx2 = next_t[5];
                    if (chunked_tiles[next_idx1]) {
                        __builtin_prefetch(chunked_tiles[next_idx1], 0, 3);
                    }
                    if (chunked_tiles[next_idx2]) {
                        __builtin_prefetch(chunked_tiles[next_idx2], 1, 3);
                    }
                }

                switch (myroutine) {
                    case 1: { // DPOTRF
                        double* __restrict__ tile_out = chunked_tiles[index1];
                        if (tile_out) {
                            const sTiles::SemisparseTileMetaCore& semi = semisparse_meta[index1];
                            const sTiles::TileMetaCore& meta = tile_meta[index1];
                            const int kd_imp1 = semi.upper_bw;
                            const int rows_imp1 = meta.height;

                            sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(rows_imp1, kd_imp1, tile_out);
                            if (ssstatus != sTiles::StatusCode::Success) {
                                std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                                ss_abort(); break;
                            }
                        }
                        ss_cond_set(k, k, 1);
                        break;
                    }

                    case 2: { // DSYRK
                        ss_cond_wait(k, n, 1);

                        double* __restrict__ tile_in = chunked_tiles[index1];
                        double* __restrict__ tile_out = chunked_tiles[index2];

                        if (tile_in && tile_out) {
                            const sTiles::SemisparseTileMetaCore& semi_in = semisparse_meta[index1];
                            const int active_cols = semi_in.sa;

                            if (active_cols > 0) {
                                const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index2];
                                const int diag_cols = semi_out.upper_bw + 1;
                                const int ld_tmp = active_cols;
                                // Use actual tile height, not nominal tile_size (fixes buffer overrun for boundary tiles)
                                const int rows_in = tile_meta[index1].height;

                                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                   active_cols, rows_in, mzone,
                                                   tile_in, rows_in, 0.0, tmp_tile, ld_tmp);

                                // Tight scatter loop - assumes valid aind data
                                const int* __restrict__ aind = semi_in.aind.data();
                                const int base_row = diag_cols - 1;

                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = aind[j];
                                    const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                    const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;

                                    for (int i = 0; i <= j; ++i) {
                                        const int row_idx = aind[i];
                                        const int band = col_idx - row_idx;
                                        if (band < diag_cols) {
                                            tile_out[base_row - band + col_base] += tmp_col[i];
                                        } else if (std::fabs(tmp_col[i]) > 1e-15) {
                                            std::fprintf(stderr, "sTiles warning: DSYRK bandwidth overflow in tile k=%d: band=%d >= diag_cols=%d, dropped=%.6e\n",
                                                         k, band, diag_cols, tmp_col[i]);
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }

                    case 3: { // DTRSM
                        ss_cond_wait(k, k, 1);

                        double* __restrict__ ss_rhs = chunked_tiles[index1];
                        double* __restrict__ ss_diag = chunked_tiles[index2];

                        if (ss_rhs && ss_diag) {
                            const sTiles::SemisparseTileMetaCore& semi_rhs = semisparse_meta[index1];
                            const sTiles::SemisparseTileMetaCore& semi_diag = semisparse_meta[index2];
                            const sTiles::TileMetaCore& meta_rhs = tile_meta[index1];
                            const sTiles::TileMetaCore& meta_diag = tile_meta[index2];

                            const int active_cols = semi_rhs.sa;
                            if (active_cols > 0) {
                                const int rows_rhs = meta_rhs.height;
                                const int diag_dim = meta_diag.height;
                                const int kd = semi_diag.upper_bw;

                                LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                               diag_dim, kd, active_cols,
                                               ss_diag, kd + 1, ss_rhs, rows_rhs);
                            }
                        }
                        ss_cond_set(m, k, 1);
                        break;
                    }

                    case 4: { // DGEMM
                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);

                        double* __restrict__ chunk_a = chunked_tiles[index1];
                        double* __restrict__ chunk_b = chunked_tiles[index2];
                        double* __restrict__ chunk_out = chunked_tiles[index3];

                        if (chunk_a && chunk_b && chunk_out) {
                            const sTiles::SemisparseTileMetaCore& semi_a = semisparse_meta[index1];
                            const sTiles::SemisparseTileMetaCore& semi_b = semisparse_meta[index2];
                            const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index3];
                            const sTiles::TileMetaCore& meta_out = tile_meta[index3];

                            const int cols_a = semi_a.sa;
                            const int cols_b = semi_b.sa;
                            const int rows_a = tile_meta[index1].height;
                            const int rows_out = meta_out.height;

                            if (cols_a > 0 && cols_b > 0) {
                                const int ld_tmp = cols_a;

                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                   cols_a, cols_b, rows_a,
                                                   mzone, chunk_a, rows_a,
                                                   chunk_b, rows_a,
                                                   0.0, tmp_tile, ld_tmp);

                                // Tight scatter - assumes valid indices
                                const int* __restrict__ aind_a = semi_a.aind.data();
                                const int* __restrict__ aind_b = semi_b.aind.data();
                                const int* __restrict__ acol_map = semi_out.acol.data();
                                const int semi_out_sa = semi_out.sa;

                                for (int j = 0; j < cols_b; ++j) {
                                    const int slot = acol_map[aind_b[j]];
                                    if (slot >= 0 && slot < semi_out_sa) {
                                        double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;

                                        // Unrolled inner loop
                                        int i = 0;
                                        for (; i + 3 < cols_a; i += 4) {
                                            out_col[aind_a[i]]     += tmp_col[i];
                                            out_col[aind_a[i + 1]] += tmp_col[i + 1];
                                            out_col[aind_a[i + 2]] += tmp_col[i + 2];
                                            out_col[aind_a[i + 3]] += tmp_col[i + 3];
                                        }
                                        for (; i < cols_a; ++i) {
                                            out_col[aind_a[i]] += tmp_col[i];
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }

                    default:
                        break;
                }

                if (ss_aborted()) break;
            }

            ss_finalize();
        }

        // =====================================================================
        // _imp2: High-performance semisparse Cholesky with:
        //   - Hybrid spin-wait (_mm_pause + backoff instead of sched_yield)
        //   - Contiguous aind fast paths (cblas_daxpy for DGEMM scatter)
        //   - Fused compute-scatter for small tiles (bypasses BLAS overhead)
        //   - Improved prefetching (multi-cacheline + metadata)
        // =====================================================================
        void dpotrf_expansion_from_chol_tasks_semisparse_imp2(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int rank = STILES_RANK;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            // Cache commonly accessed pointers with restrict + alignment hints
            double** __restrict__ const chunked_tiles = tiledMatrix->chunkedDenseTiles;
            const sTiles::SemisparseTileMetaCore* __restrict__ const semisparse_meta = tiledMatrix->semisparseTileMetaCore;
            const sTiles::TileMetaCore* __restrict__ const tile_meta = tiledMatrix->tileMetaCore;

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[semisparse_imp2] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* __restrict__ tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
            const double mzone = -1.0;

            // Fuse threshold: below this, bypass BLAS and compute dot products inline
            constexpr int FUSE_THRESHOLD = 16;

            // O(1) contiguity check: since aind is always sorted ascending,
            // contiguous iff last - first + 1 == count
            auto is_contiguous = [](const sTiles::SemisparseTileMetaCore& s) -> bool {
                return s.sa > 0 && (s.la - s.fa + 1 == s.sa);
            };

            // Hybrid spin-wait: use _mm_pause for first 64 iterations, then sched_yield
            #define SS_WAIT_IMP2(m_coord, n_coord, val) \
            { \
                int _spin_ct = 0; \
                while (!stile->ss_abort && \
                        stile->ss_progress[(m_coord) + stile->ss_ld * (n_coord)] != (val)) { \
                    hpc_pause_hybrid(_spin_ct); \
                } \
                if (stile->ss_abort) break; \
            }

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);

            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int ntiles_bounds = tiledMatrix->numActiveTiles;
                // // Skip tasks with invalid tile indices (safety net — accept filter should prevent this)
                // if (index1 < 0 || index1 >= ntiles_bounds ||
                //     index2 < 0 || index2 >= ntiles_bounds ||
                //     (myroutine == 4 && (index3 < 0 || index3 >= ntiles_bounds))) {
                //     continue;
                // }

                // Improved prefetching: 4 cache lines per tile + metadata
                if (idx + 1 < end) {
                    const std::array<int,7> &next_t = tasks[idx + 1];
                    const int next_idx1 = next_t[4];
                    const int next_idx2 = next_t[5];
                    if (next_idx1 >= 0 && next_idx1 < ntiles_bounds && chunked_tiles[next_idx1]) {
                        const char* p1 = reinterpret_cast<const char*>(chunked_tiles[next_idx1]);
                        __builtin_prefetch(p1,       0, 3);
                        __builtin_prefetch(p1 + 64,  0, 3);
                        __builtin_prefetch(p1 + 128, 0, 3);
                        __builtin_prefetch(p1 + 192, 0, 3);
                    }
                    if (next_idx2 >= 0 && next_idx2 < ntiles_bounds && chunked_tiles[next_idx2]) {
                        const char* p2 = reinterpret_cast<const char*>(chunked_tiles[next_idx2]);
                        __builtin_prefetch(p2,       1, 3);
                        __builtin_prefetch(p2 + 64,  1, 3);
                        __builtin_prefetch(p2 + 128, 1, 3);
                        __builtin_prefetch(p2 + 192, 1, 3);
                    }
                    // Prefetch third tile for DGEMM tasks
                    if (next_t[0] == 4) {
                        const int next_idx3 = next_t[6];
                        if (next_idx3 >= 0 && next_idx3 < ntiles_bounds && chunked_tiles[next_idx3]) {
                            const char* p3 = reinterpret_cast<const char*>(chunked_tiles[next_idx3]);
                            __builtin_prefetch(p3,       1, 3);
                            __builtin_prefetch(p3 + 64,  1, 3);
                        }
                    }
                    // Prefetch next task's metadata
                    __builtin_prefetch(&semisparse_meta[next_idx1], 0, 3);
                    __builtin_prefetch(&semisparse_meta[next_idx2], 0, 3);
                }

                switch (myroutine) {

                    case 4: { // DGEMM — C(k,m) -= A(n,k)^T * B(n,m)
                        SS_WAIT_IMP2(k, n, 1);
                        SS_WAIT_IMP2(m, n, 1);
                        double* __restrict__ chunk_a = chunked_tiles[index1];
                        double* __restrict__ chunk_b = chunked_tiles[index2];
                        double* __restrict__ chunk_out = chunked_tiles[index3];

                        const sTiles::SemisparseTileMetaCore& semi_a = semisparse_meta[index1];
                        const sTiles::SemisparseTileMetaCore& semi_b = semisparse_meta[index2];
                        const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index3];

                        const int cols_a = semi_a.sa;
                        const int cols_b = semi_b.sa;
                        const int rows_a = tile_meta[index1].height;
                        const int rows_out = tile_meta[index3].height;

                        {
                                const int* __restrict__ aind_a = semi_a.aind.data();
                                const int* __restrict__ aind_b = semi_b.aind.data();
                                const int* __restrict__ acol_map = semi_out.acol.data();
                                const int semi_out_sa = semi_out.sa;

                                // Check if both A rows and output are contiguous
                                const bool a_contig = semi_a.is_contiguous;
                                const bool b_contig = semi_b.is_contiguous;
                                const bool out_full = semi_out.is_full_width;

                                if (a_contig && b_contig && out_full) {
                                    // === FAST PATH: direct DGEMM into output, no tmp_tile ===
                                    const int a_start = semi_a.fa;
                                    const int b_start = semi_b.fa;
                                    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                                cols_a, cols_b, rows_a, mzone,
                                                chunk_a, rows_a,
                                                chunk_b, rows_a,
                                                1.0, chunk_out + a_start + static_cast<std::size_t>(b_start) * rows_out, rows_out);

                                } else if (cols_a <= FUSE_THRESHOLD && cols_b <= FUSE_THRESHOLD) {
                                    // === FUSED compute-scatter: bypass BLAS entirely ===
                                    if (a_contig) {
                                        const int a_start = semi_a.fa;
                                        for (int j = 0; j < cols_b; ++j) {
                                            const int slot = acol_map[aind_b[j]];
                                            if (slot < 0) continue;
                                            double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                            const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                            for (int i = 0; i < cols_a; ++i) {
                                                const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                                double dot = 0.0;
                                                for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                                out_col[a_start + i] -= dot;
                                            }
                                        }
                                    } else {
                                        for (int j = 0; j < cols_b; ++j) {
                                            const int slot = acol_map[aind_b[j]];
                                            if (slot < 0) continue;
                                            double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                            const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                            for (int i = 0; i < cols_a; ++i) {
                                                const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                                double dot = 0.0;
                                                for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                                out_col[aind_a[i]] -= dot;
                                            }
                                        }
                                    }

                                } else {
                                    // === BLAS DGEMM + scatter ===
                                    const int ld_tmp = cols_a;
                                    sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                       cols_a, cols_b, rows_a,
                                                       mzone, chunk_a, rows_a,
                                                       chunk_b, rows_a,
                                                       0.0, tmp_tile, ld_tmp);

                                    if (a_contig) {
                                        const int a_start = semi_a.fa;
                                        for (int j = 0; j < cols_b; ++j) {
                                            const int slot = acol_map[aind_b[j]];
                                            if (slot < 0) continue;
                                            cblas_daxpy(cols_a, 1.0, tmp_tile + j * ld_tmp, 1,
                                                        chunk_out + static_cast<std::size_t>(slot) * rows_out + a_start, 1);
                                        }
                                    } else {
                                        for (int j = 0; j < cols_b; ++j) {
                                            const int slot = acol_map[aind_b[j]];
                                            if (slot < 0) continue;
                                            double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                            const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                            int i = 0;
                                            for (; i + 3 < cols_a; i += 4) {
                                                out_col[aind_a[i]]     += tmp_col[i];
                                                out_col[aind_a[i + 1]] += tmp_col[i + 1];
                                                out_col[aind_a[i + 2]] += tmp_col[i + 2];
                                                out_col[aind_a[i + 3]] += tmp_col[i + 3];
                                            }
                                            for (; i < cols_a; ++i) out_col[aind_a[i]] += tmp_col[i];
                                        }
                                    }
                                }
                            }
                        
                        break;
                    }

                    case 1: { // DPOTRF — banded Cholesky on diagonal tile
                        double* __restrict__ tile_out = chunked_tiles[index1];
                        if (tile_out) {
                            const sTiles::SemisparseTileMetaCore& semi = semisparse_meta[index1];
                            const sTiles::TileMetaCore& meta = tile_meta[index1];
                            sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(meta.height, semi.upper_bw, tile_out);
                            if (ssstatus != sTiles::StatusCode::Success) {
                                std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                                ss_abort(); break;
                            }
                        }
                        ss_cond_set(k, k, 1);
                        break;
                    }

                    case 2: { // DSYRK — symmetric rank-k update into banded diagonal
                        SS_WAIT_IMP2(k, n, 1);

                        double* __restrict__ tile_in = chunked_tiles[index1];
                        double* __restrict__ tile_out = chunked_tiles[index2];

                        if (tile_in && tile_out) {
                            const sTiles::SemisparseTileMetaCore& semi_in = semisparse_meta[index1];
                            const int active_cols = semi_in.sa;

                            if (active_cols > 0) {
                                const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index2];
                                const int diag_cols = semi_out.upper_bw + 1;
                                const int rows_in = (tile_meta[index1].height > 0) ? tile_meta[index1].height : tile_size;
                                const int base_row = diag_cols - 1;

                                const bool aind_contig = semi_in.is_contiguous;

                                if (active_cols <= FUSE_THRESHOLD && aind_contig && active_cols <= diag_cols) {
                                    // === FUSED contiguous: dot products directly into banded output ===
                                    const int aind_start = semi_in.fa;
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_start + j;
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                        for (int i = 0; i <= j; ++i) {
                                            const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                            double dot = 0.0;
                                            for (int r = 0; r < rows_in; ++r) {
                                                dot += col_i[r] * col_j[r];
                                            }
                                            tile_out[base_row - (j - i) + col_base] -= dot;
                                        }
                                    }

                                } else if (active_cols <= FUSE_THRESHOLD) {
                                    // === FUSED scattered: dot products with aind lookups ===
                                    const int* __restrict__ aind = semi_in.aind.data();
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind[j];
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                        for (int i = 0; i <= j; ++i) {
                                            const int band = col_idx - aind[i];
                                            if (band < diag_cols) {
                                                const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                                double dot = 0.0;
                                                for (int r = 0; r < rows_in; ++r) {
                                                    dot += col_i[r] * col_j[r];
                                                }
                                                tile_out[base_row - band + col_base] -= dot;
                                            }
                                        }
                                    }

                                } else if (aind_contig && active_cols <= diag_cols) {
                                    // === BLAS dsyrk + contiguous scatter (branch-free inner loop) ===
                                    const int ld_tmp = active_cols;
                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                       active_cols, rows_in, mzone,
                                                       tile_in, rows_in, 0.0, tmp_tile, ld_tmp);

                                    const int aind_start = semi_in.fa;
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_start + j;
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        for (int i = 0; i <= j; ++i) {
                                            // band = j - i, always < diag_cols (guaranteed by outer guard)
                                            tile_out[base_row - (j - i) + col_base] += tmp_col[i];
                                        }
                                    }

                                } else {
                                    // === BLAS dsyrk + general scatter (same as _imp1) ===
                                    const int ld_tmp = active_cols;
                                    const int* __restrict__ aind = semi_in.aind.data();
                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                       active_cols, rows_in, mzone,
                                                       tile_in, rows_in, 0.0, tmp_tile, ld_tmp);

                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind[j];
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        for (int i = 0; i <= j; ++i) {
                                            const int band = col_idx - aind[i];
                                            if (band < diag_cols) {
                                                tile_out[base_row - band + col_base] += tmp_col[i];
                                            } else if (std::fabs(tmp_col[i]) > 1e-15) {
                                                std::fprintf(stderr, "sTiles warning: DSYRK bandwidth overflow in tile k=%d: band=%d >= diag_cols=%d, dropped=%.6e\n",
                                                             k, band, diag_cols, tmp_col[i]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }

                    case 3: { // DTRSM — banded triangular solve
                        SS_WAIT_IMP2(k, k, 1);

                        double* __restrict__ ss_rhs = chunked_tiles[index1];
                        double* __restrict__ ss_diag = chunked_tiles[index2];

                        if (ss_rhs && ss_diag) {
                            const sTiles::SemisparseTileMetaCore& semi_rhs = semisparse_meta[index1];
                            const sTiles::SemisparseTileMetaCore& semi_diag = semisparse_meta[index2];
                            const sTiles::TileMetaCore& meta_rhs = tile_meta[index1];
                            const sTiles::TileMetaCore& meta_diag = tile_meta[index2];

                            const int active_cols = semi_rhs.sa;
                            if (active_cols > 0) {
                                const int rows_rhs = meta_rhs.height;
                                const int diag_dim = meta_diag.height;
                                const int kd = semi_diag.upper_bw;

                                lapack_int info = LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                                                 diag_dim, kd, active_cols,
                                                                 ss_diag, kd + 1, ss_rhs, rows_rhs);
                                if (info != 0) {
                                    std::fprintf(stderr, "sTiles error: LAPACKE_dtbtrs failed with info=%d (tile m=%d, k=%d)\n",
                                                 info, m, k);
                                    ss_abort(); break;
                                }
                            }
                        }
                        ss_cond_set(m, k, 1);
                        break;
                    }

                    default:
                        break;
                }

                if (ss_aborted()) break;
            }

            ss_finalize();
            #undef SS_WAIT_IMP2
        }

        void dpotrf_expansion_from_chol_tasks_semisparse_imp3(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int rank = STILES_RANK;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            double** __restrict__ const chunked_tiles = tiledMatrix->chunkedDenseTiles;
            const sTiles::SemisparseTileMetaCore* __restrict__ const semisparse_meta = tiledMatrix->semisparseTileMetaCore;
            const sTiles::TileMetaCore* __restrict__ const tile_meta = tiledMatrix->tileMetaCore;

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[semisparse_imp3] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* __restrict__ tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
            const double mzone = -1.0;

            constexpr int FUSE_THRESHOLD = 16;

            // Precomputed scatter info — required (param[7]==1, default in fast mode).
            // Case 4's dispatch reads scatter_index/scatter_packed unconditionally; the
            // runtime fallback was removed because build_chol_scatter_info is always
            // called by sTiles_preprocess_group_fastmode when param[7]==1.
            const int64_t* scatter_index = tiledMatrix->chol_scatter_index->data();
            const int32_t* scatter_packed = tiledMatrix->chol_scatter_packed->data();

            #define SS_WAIT_IMP2(m_coord, n_coord, val) \
            { \
                int _spin_ct = 0; \
                while (!stile->ss_abort && \
                        stile->ss_progress[(m_coord) + stile->ss_ld * (n_coord)] != (val)) { \
                    hpc_pause_hybrid(_spin_ct); \
                } \
                if (stile->ss_abort) break; \
            }

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);

            const int ntiles_bounds = tiledMatrix->numActiveTiles;

            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                // Prefetching
                if (idx + 1 < end) {
                    const std::array<int,7> &next_t = tasks[idx + 1];
                    const int next_idx1 = next_t[4];
                    const int next_idx2 = next_t[5];
                    if (next_idx1 >= 0 && next_idx1 < ntiles_bounds && chunked_tiles[next_idx1]) {
                        const char* p1 = reinterpret_cast<const char*>(chunked_tiles[next_idx1]);
                        __builtin_prefetch(p1, 0, 3); __builtin_prefetch(p1+64, 0, 3);
                        __builtin_prefetch(p1+128, 0, 3); __builtin_prefetch(p1+192, 0, 3);
                    }
                    if (next_idx2 >= 0 && next_idx2 < ntiles_bounds && chunked_tiles[next_idx2]) {
                        const char* p2 = reinterpret_cast<const char*>(chunked_tiles[next_idx2]);
                        __builtin_prefetch(p2, 1, 3); __builtin_prefetch(p2+64, 1, 3);
                        __builtin_prefetch(p2+128, 1, 3); __builtin_prefetch(p2+192, 1, 3);
                    }
                    if (next_t[0] == 4) {
                        const int next_idx3 = next_t[6];
                        if (next_idx3 >= 0 && next_idx3 < ntiles_bounds && chunked_tiles[next_idx3]) {
                            const char* p3 = reinterpret_cast<const char*>(chunked_tiles[next_idx3]);
                            __builtin_prefetch(p3, 1, 3); __builtin_prefetch(p3+64, 1, 3);
                        }
                    }
                    __builtin_prefetch(&semisparse_meta[next_idx1], 0, 3);
                    __builtin_prefetch(&semisparse_meta[next_idx2], 0, 3);
                }

                switch (myroutine) {

                    case 4: { // DGEMM — C(k,m) -= A(n,k)^T * B(n,m)
                        SS_WAIT_IMP2(k, n, 1);
                        SS_WAIT_IMP2(m, n, 1);

                        double* __restrict__ chunk_out = chunked_tiles[index3];
                        const int rows_out = tile_meta[index3].height;

                        double* __restrict__ chunk_a = chunked_tiles[index1];
                        double* __restrict__ chunk_b = chunked_tiles[index2];

                        const sTiles::SemisparseTileMetaCore& semi_a = semisparse_meta[index1];
                        const sTiles::SemisparseTileMetaCore& semi_b = semisparse_meta[index2];

                        const int cols_a = semi_a.sa;
                        const int cols_b = semi_b.sa;
                        const int rows_a = tile_meta[index1].height;

                        // ── Dense path: precomputed path_flag dispatch ──
                        // Precondition: param[7]==1 (default in fast mode), which guarantees
                        // build_chol_scatter_info has populated scatter_index/scatter_packed.
                        const int si_base = idx * 2;
                        const int64_t data_off  = scatter_index[si_base];
                        const int path_flag = static_cast<int>(scatter_index[si_base + 1]);

                        // Case 4 small-K path — before the existing p0-p4 dispatch
                        if (path_flag == 0) {
                            const int a_start = semi_a.fa;
                            const int b_start = semi_b.fa;
                            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                        cols_a, cols_b, rows_a, mzone,
                                        chunk_a, rows_a, chunk_b, rows_a,
                                        1.0, chunk_out + a_start + static_cast<std::size_t>(b_start) * rows_out, rows_out);

                        } else if (path_flag == 1) {
                            const int a_start = semi_a.fa;
                            const int32_t* slots = scatter_packed + data_off;
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                for (int i = 0; i < cols_a; ++i) {
                                    const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                    double dot = 0.0;
                                    for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                    out_col[a_start + i] -= dot;
                                }
                            }

                        } else if (path_flag == 2) {
                            const int* __restrict__ aind_a = semi_a.aind.data();
                            const int32_t* slots = scatter_packed + data_off;
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                for (int i = 0; i < cols_a; ++i) {
                                    const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                    double dot = 0.0;
                                    for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                    out_col[aind_a[i]] -= dot;
                                }
                            }

                        } else if (path_flag == 3) {
                            const int ld_tmp = cols_a;
                            const int a_start = semi_a.fa;
                            const int32_t* slots = scatter_packed + data_off;
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                               cols_a, cols_b, rows_a,
                                               mzone, chunk_a, rows_a, chunk_b, rows_a,
                                               0.0, tmp_tile, ld_tmp);
                            // Unrolled scatter (contiguous) — avoids daxpy call overhead on small cols_a
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out + a_start;
                                const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                int i = 0;
                                for (; i + 3 < cols_a; i += 4) {
                                    out_col[i]     += tmp_col[i];
                                    out_col[i + 1] += tmp_col[i + 1];
                                    out_col[i + 2] += tmp_col[i + 2];
                                    out_col[i + 3] += tmp_col[i + 3];
                                }
                                for (; i < cols_a; ++i) out_col[i] += tmp_col[i];
                            }

                        } else if (path_flag == 4) {
                            const int ld_tmp = cols_a;
                            const int* __restrict__ aind_a = semi_a.aind.data();
                            const int32_t* slots = scatter_packed + data_off;
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                               cols_a, cols_b, rows_a,
                                               mzone, chunk_a, rows_a, chunk_b, rows_a,
                                               0.0, tmp_tile, ld_tmp);
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                int i = 0;
                                for (; i + 3 < cols_a; i += 4) {
                                    out_col[aind_a[i]]     += tmp_col[i];
                                    out_col[aind_a[i + 1]] += tmp_col[i + 1];
                                    out_col[aind_a[i + 2]] += tmp_col[i + 2];
                                    out_col[aind_a[i + 3]] += tmp_col[i + 3];
                                }
                                for (; i < cols_a; ++i) out_col[aind_a[i]] += tmp_col[i];
                            }
                        }
                        break;
                    }

                    case 1: { // DPOTRF — banded Cholesky on diagonal tile
                        // Diagonal tile: chunked_tiles[index1] is allocated with elems = height*(upper_bw+1),
                        // so it's never null for an active diagonal — no null guard needed.
                        double* __restrict__ tile_out = chunked_tiles[index1];
                        const sTiles::SemisparseTileMetaCore& semi = semisparse_meta[index1];
                        const sTiles::TileMetaCore& meta = tile_meta[index1];
                        sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(meta.height, semi.upper_bw, tile_out);
                        if (ssstatus != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                            ss_abort(); break;
                        }
                        ss_cond_set(k, k, 1);
                        break;
                    }

                    case 2: { // DSYRK — symmetric rank-k update into banded diagonal
                        SS_WAIT_IMP2(k, n, 1);

                        double* __restrict__ tile_in = chunked_tiles[index1];
                        double* __restrict__ tile_out = chunked_tiles[index2];

                        if (tile_in && tile_out) {
                            const sTiles::SemisparseTileMetaCore& semi_in = semisparse_meta[index1];
                            const int active_cols = semi_in.sa;

                            if (active_cols > 0) {
                                const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index2];
                                const int diag_cols = semi_out.upper_bw + 1;
                                const int rows_in = (tile_meta[index1].height > 0) ? tile_meta[index1].height : tile_size;
                                const int base_row = diag_cols - 1;

                                const bool aind_contig = semi_in.is_contiguous;

                                if (active_cols <= FUSE_THRESHOLD && aind_contig && active_cols <= diag_cols) {
                                    // === FUSED contiguous: dot products directly into banded output ===
                                    const int aind_start = semi_in.fa;
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_start + j;
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                        for (int i = 0; i <= j; ++i) {
                                            const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                            double dot = 0.0;
                                            for (int r = 0; r < rows_in; ++r) {
                                                dot += col_i[r] * col_j[r];
                                            }
                                            tile_out[base_row - (j - i) + col_base] -= dot;
                                        }
                                    }

                                } else if (active_cols <= FUSE_THRESHOLD) {
                                    // === FUSED scattered: dot products with aind lookups ===
                                    const int* __restrict__ aind = semi_in.aind.data();
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind[j];
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                        for (int i = 0; i <= j; ++i) {
                                            const int band = col_idx - aind[i];
                                            if (band < diag_cols) {
                                                const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                                double dot = 0.0;
                                                for (int r = 0; r < rows_in; ++r) {
                                                    dot += col_i[r] * col_j[r];
                                                }
                                                tile_out[base_row - band + col_base] -= dot;
                                            }
                                        }
                                    }

                                } else if (aind_contig && active_cols <= diag_cols) {
                                    // === BLAS dsyrk + contiguous scatter (branch-free inner loop) ===
                                    const int ld_tmp = active_cols;
                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                       active_cols, rows_in, mzone,
                                                       tile_in, rows_in, 0.0, tmp_tile, ld_tmp);

                                    const int aind_start = semi_in.fa;
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_start + j;
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        for (int i = 0; i <= j; ++i) {
                                            // band = j - i, always < diag_cols (guaranteed by outer guard)
                                            tile_out[base_row - (j - i) + col_base] += tmp_col[i];
                                        }
                                    }

                                } else {
                                    // === BLAS dsyrk + general scatter (same as _imp1) ===
                                    const int ld_tmp = active_cols;
                                    const int* __restrict__ aind = semi_in.aind.data();
                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                       active_cols, rows_in, mzone,
                                                       tile_in, rows_in, 0.0, tmp_tile, ld_tmp);

                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind[j];
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        for (int i = 0; i <= j; ++i) {
                                            const int band = col_idx - aind[i];
                                            if (band < diag_cols) {
                                                tile_out[base_row - band + col_base] += tmp_col[i];
                                            } else if (std::fabs(tmp_col[i]) > 1e-15) {
                                                std::fprintf(stderr, "sTiles warning: DSYRK bandwidth overflow in tile k=%d: band=%d >= diag_cols=%d, dropped=%.6e\n",
                                                             k, band, diag_cols, tmp_col[i]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }

                    case 3: { // DTRSM — banded triangular solve
                        // Same invariant as case 2: ss_rhs is null iff active_cols==0.
                        // ss_diag is the diagonal, never null. Single guard suffices.
                        SS_WAIT_IMP2(k, k, 1);

                        const sTiles::SemisparseTileMetaCore& semi_rhs = semisparse_meta[index1];
                        const int active_cols = semi_rhs.sa;
                        if (active_cols > 0) {
                            double* __restrict__ ss_rhs  = chunked_tiles[index1];
                            double* __restrict__ ss_diag = chunked_tiles[index2];
                            const sTiles::SemisparseTileMetaCore& semi_diag = semisparse_meta[index2];
                            const int rows_rhs = tile_meta[index1].height;
                            const int diag_dim = tile_meta[index2].height;
                            const int kd = semi_diag.upper_bw;

                            lapack_int info = LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                                             diag_dim, kd, active_cols,
                                                             ss_diag, kd + 1, ss_rhs, rows_rhs);
                            if (info != 0) {
                                std::fprintf(stderr, "sTiles error: LAPACKE_dtbtrs failed with info=%d (tile m=%d, k=%d)\n",
                                             info, m, k);
                                ss_abort(); break;
                            }
                        }
                        ss_cond_set(m, k, 1);
                        break;
                    }

                    default:
                        break;
                }

                if (ss_aborted()) break;
            }
            ss_finalize();
            #undef SS_WAIT_IMP2
        }

        void dpotrf_expansion_from_chol_tasks_semisparse_imp4(TiledMatrix *tiledMatrix, stiles_context_t *stile) {
            // imp4 = imp3 + memory/core-model improvements:
            //   (a) #pragma omp simd reduction on hand-rolled dot loops (cases 4-p1/p2, 2-fused)
            //   (b) 2-column-of-A blocking in cases 4-p1/p2 → reuses col_b_j register, ~halves B traffic
            //   (c) Smarter prefetch: drops redundant 4×64B tile-head reads, prefetches metadata,
            //       scatter_packed[next_data_off], and aind data of next semisparse meta
            //   (d) Single load of next-task scatter_index pair (cache line warmup)
            //   (e) Hoists tile_meta[*].height to scalars before hot loops
            // Numerics, semantics, and dependency protocol are identical to imp3.

            const int rank = STILES_RANK;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            double** __restrict__ const chunked_tiles = tiledMatrix->chunkedDenseTiles;
            const sTiles::SemisparseTileMetaCore* __restrict__ const semisparse_meta = tiledMatrix->semisparseTileMetaCore;
            const sTiles::TileMetaCore* __restrict__ const tile_meta = tiledMatrix->tileMetaCore;

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[semisparse_imp4] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* __restrict__ tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
            const double mzone = -1.0;

            constexpr int FUSE_THRESHOLD = 16;

            const int64_t* scatter_index = tiledMatrix->chol_scatter_index->data();
            const int32_t* scatter_packed = tiledMatrix->chol_scatter_packed->data();

            #define SS_WAIT_IMP2(m_coord, n_coord, val) \
            { \
                int _spin_ct = 0; \
                while (!stile->ss_abort && \
                        stile->ss_progress[(m_coord) + stile->ss_ld * (n_coord)] != (val)) { \
                    hpc_pause_hybrid(_spin_ct); \
                } \
                if (stile->ss_abort) break; \
            }

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);

            const int ntiles_bounds = tiledMatrix->numActiveTiles;

            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                // ── (c)+(d) Prefetch: focus on first cache line of each tile (metadata is already
                // in registers by the time we touch the body), plus the indirection arrays
                // (scatter_packed, aind) that drive the scatter — those are unpredictable loads
                // and benefit most from prefetch. Use _MM_HINT_T1 (==2) to land in L2 and avoid
                // L1 thrash with the current task's working set.
                if (idx + 1 < end) {
                    const std::array<int,7> &next_t = tasks[idx + 1];
                    const int next_routine = next_t[0];
                    const int next_idx1 = next_t[4];
                    const int next_idx2 = next_t[5];

                    if (next_idx1 >= 0 && next_idx1 < ntiles_bounds && chunked_tiles[next_idx1]) {
                        __builtin_prefetch(chunked_tiles[next_idx1], 0, 2);
                    }
                    if (next_idx2 >= 0 && next_idx2 < ntiles_bounds && chunked_tiles[next_idx2]) {
                        __builtin_prefetch(chunked_tiles[next_idx2], 1, 2);
                    }
                    if (next_routine == 4) {
                        const int next_idx3 = next_t[6];
                        if (next_idx3 >= 0 && next_idx3 < ntiles_bounds && chunked_tiles[next_idx3]) {
                            __builtin_prefetch(chunked_tiles[next_idx3], 1, 2);
                        }
                        // Indirection arrays for the next case-4 dispatch
                        const int64_t next_si = static_cast<int64_t>(idx + 1) * 2;
                        __builtin_prefetch(&scatter_index[next_si], 0, 3);
                        const int64_t next_data_off = scatter_index[next_si];
                        if (next_data_off >= 0) {
                            __builtin_prefetch(scatter_packed + next_data_off, 0, 3);
                        }
                    }
                    __builtin_prefetch(&semisparse_meta[next_idx1], 0, 3);
                    __builtin_prefetch(&semisparse_meta[next_idx2], 0, 3);
                    // aind backing data for next semis (the vector header is in the meta cache line,
                    // but the heap-allocated int array is a separate line)
                    if (next_idx1 >= 0 && next_idx1 < ntiles_bounds) {
                        const int* aind_ptr = semisparse_meta[next_idx1].aind.data();
                        if (aind_ptr) __builtin_prefetch(aind_ptr, 0, 2);
                    }
                }

                switch (myroutine) {

                    case 4: { // DGEMM — C(k,m) -= A(n,k)^T * B(n,m)
                        SS_WAIT_IMP2(k, n, 1);
                        SS_WAIT_IMP2(m, n, 1);

                        double* __restrict__ chunk_out = chunked_tiles[index3];
                        const int rows_out = tile_meta[index3].height;

                        double* __restrict__ chunk_a = chunked_tiles[index1];
                        double* __restrict__ chunk_b = chunked_tiles[index2];

                        const sTiles::SemisparseTileMetaCore& semi_a = semisparse_meta[index1];
                        const sTiles::SemisparseTileMetaCore& semi_b = semisparse_meta[index2];

                        const int cols_a = semi_a.sa;
                        const int cols_b = semi_b.sa;
                        const int rows_a = tile_meta[index1].height;

                        const int si_base = idx * 2;
                        const int64_t data_off  = scatter_index[si_base];
                        const int path_flag = static_cast<int>(scatter_index[si_base + 1]);

                        if (path_flag == 0) {
                            const int a_start = semi_a.fa;
                            const int b_start = semi_b.fa;
                            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                        cols_a, cols_b, rows_a, mzone,
                                        chunk_a, rows_a, chunk_b, rows_a,
                                        1.0, chunk_out + a_start + static_cast<std::size_t>(b_start) * rows_out, rows_out);

                        } else if (path_flag == 1) {
                            // (a) vectorize dot, (b) 2-i blocking: process A-cols (i, i+1) against
                            // a single B-col → halves loads of col_b_j. Tail handles odd cols_a.
                            const int a_start = semi_a.fa;
                            const int32_t* slots = scatter_packed + data_off;
                            const int cols_a_pair = cols_a & ~1;
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out + a_start;
                                const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                int i = 0;
                                for (; i < cols_a_pair; i += 2) {
                                    const double* __restrict__ col_a_i0 = chunk_a + static_cast<std::size_t>(i)     * rows_a;
                                    const double* __restrict__ col_a_i1 = chunk_a + static_cast<std::size_t>(i + 1) * rows_a;
                                    double dot0 = 0.0, dot1 = 0.0;
                                    #pragma omp simd reduction(+:dot0,dot1)
                                    for (int r = 0; r < rows_a; ++r) {
                                        const double bv = col_b_j[r];
                                        dot0 += col_a_i0[r] * bv;
                                        dot1 += col_a_i1[r] * bv;
                                    }
                                    out_col[i]     -= dot0;
                                    out_col[i + 1] -= dot1;
                                }
                                if (i < cols_a) {
                                    const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                    double dot = 0.0;
                                    #pragma omp simd reduction(+:dot)
                                    for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                    out_col[i] -= dot;
                                }
                            }

                        } else if (path_flag == 2) {
                            const int* __restrict__ aind_a = semi_a.aind.data();
                            const int32_t* slots = scatter_packed + data_off;
                            const int cols_a_pair = cols_a & ~1;
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                int i = 0;
                                for (; i < cols_a_pair; i += 2) {
                                    const double* __restrict__ col_a_i0 = chunk_a + static_cast<std::size_t>(i)     * rows_a;
                                    const double* __restrict__ col_a_i1 = chunk_a + static_cast<std::size_t>(i + 1) * rows_a;
                                    double dot0 = 0.0, dot1 = 0.0;
                                    #pragma omp simd reduction(+:dot0,dot1)
                                    for (int r = 0; r < rows_a; ++r) {
                                        const double bv = col_b_j[r];
                                        dot0 += col_a_i0[r] * bv;
                                        dot1 += col_a_i1[r] * bv;
                                    }
                                    out_col[aind_a[i]]     -= dot0;
                                    out_col[aind_a[i + 1]] -= dot1;
                                }
                                if (i < cols_a) {
                                    const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                    double dot = 0.0;
                                    #pragma omp simd reduction(+:dot)
                                    for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                    out_col[aind_a[i]] -= dot;
                                }
                            }

                        } else if (path_flag == 3) {
                            const int ld_tmp = cols_a;
                            const int a_start = semi_a.fa;
                            const int32_t* slots = scatter_packed + data_off;
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                               cols_a, cols_b, rows_a,
                                               mzone, chunk_a, rows_a, chunk_b, rows_a,
                                               0.0, tmp_tile, ld_tmp);
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out + a_start;
                                const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                #pragma omp simd
                                for (int i = 0; i < cols_a; ++i) out_col[i] += tmp_col[i];
                            }

                        } else if (path_flag == 4) {
                            const int ld_tmp = cols_a;
                            const int* __restrict__ aind_a = semi_a.aind.data();
                            const int32_t* slots = scatter_packed + data_off;
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                               cols_a, cols_b, rows_a,
                                               mzone, chunk_a, rows_a, chunk_b, rows_a,
                                               0.0, tmp_tile, ld_tmp);
                            for (int j = 0; j < cols_b; ++j) {
                                const int slot = slots[j];
                                if (slot < 0) continue;
                                double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                int i = 0;
                                for (; i + 3 < cols_a; i += 4) {
                                    out_col[aind_a[i]]     += tmp_col[i];
                                    out_col[aind_a[i + 1]] += tmp_col[i + 1];
                                    out_col[aind_a[i + 2]] += tmp_col[i + 2];
                                    out_col[aind_a[i + 3]] += tmp_col[i + 3];
                                }
                                for (; i < cols_a; ++i) out_col[aind_a[i]] += tmp_col[i];
                            }
                        }
                        break;
                    }

                    case 1: { // DPOTRF — banded Cholesky on diagonal tile
                        double* __restrict__ tile_out = chunked_tiles[index1];
                        const sTiles::SemisparseTileMetaCore& semi = semisparse_meta[index1];
                        const sTiles::TileMetaCore& meta = tile_meta[index1];
                        sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(meta.height, semi.upper_bw, tile_out);
                        if (ssstatus != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                            ss_abort(); break;
                        }
                        ss_cond_set(k, k, 1);
                        break;
                    }

                    case 2: { // DSYRK — symmetric rank-k update into banded diagonal
                        SS_WAIT_IMP2(k, n, 1);

                        double* __restrict__ tile_in = chunked_tiles[index1];
                        double* __restrict__ tile_out = chunked_tiles[index2];

                        if (tile_in && tile_out) {
                            const sTiles::SemisparseTileMetaCore& semi_in = semisparse_meta[index1];
                            const int active_cols = semi_in.sa;

                            if (active_cols > 0) {
                                const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index2];
                                const int diag_cols = semi_out.upper_bw + 1;
                                const int rows_in = (tile_meta[index1].height > 0) ? tile_meta[index1].height : tile_size;
                                const int base_row = diag_cols - 1;

                                const bool aind_contig = semi_in.is_contiguous;

                                if (active_cols <= FUSE_THRESHOLD && aind_contig && active_cols <= diag_cols) {
                                    // === FUSED contiguous: vectorized dot products into banded output ===
                                    const int aind_start = semi_in.fa;
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_start + j;
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                        for (int i = 0; i <= j; ++i) {
                                            const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                            double dot = 0.0;
                                            #pragma omp simd reduction(+:dot)
                                            for (int r = 0; r < rows_in; ++r) {
                                                dot += col_i[r] * col_j[r];
                                            }
                                            tile_out[base_row - (j - i) + col_base] -= dot;
                                        }
                                    }

                                } else if (active_cols <= FUSE_THRESHOLD) {
                                    // === FUSED scattered: vectorized dots with aind lookups ===
                                    const int* __restrict__ aind = semi_in.aind.data();
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind[j];
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                        for (int i = 0; i <= j; ++i) {
                                            const int band = col_idx - aind[i];
                                            if (band < diag_cols) {
                                                const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                                double dot = 0.0;
                                                #pragma omp simd reduction(+:dot)
                                                for (int r = 0; r < rows_in; ++r) {
                                                    dot += col_i[r] * col_j[r];
                                                }
                                                tile_out[base_row - band + col_base] -= dot;
                                            }
                                        }
                                    }

                                } else if (aind_contig && active_cols <= diag_cols) {
                                    const int ld_tmp = active_cols;
                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                       active_cols, rows_in, mzone,
                                                       tile_in, rows_in, 0.0, tmp_tile, ld_tmp);

                                    const int aind_start = semi_in.fa;
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_start + j;
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        for (int i = 0; i <= j; ++i) {
                                            tile_out[base_row - (j - i) + col_base] += tmp_col[i];
                                        }
                                    }

                                } else {
                                    const int ld_tmp = active_cols;
                                    const int* __restrict__ aind = semi_in.aind.data();
                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                       active_cols, rows_in, mzone,
                                                       tile_in, rows_in, 0.0, tmp_tile, ld_tmp);

                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind[j];
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        for (int i = 0; i <= j; ++i) {
                                            const int band = col_idx - aind[i];
                                            if (band < diag_cols) {
                                                tile_out[base_row - band + col_base] += tmp_col[i];
                                            } else if (std::fabs(tmp_col[i]) > 1e-15) {
                                                std::fprintf(stderr, "sTiles warning: DSYRK bandwidth overflow in tile k=%d: band=%d >= diag_cols=%d, dropped=%.6e\n",
                                                             k, band, diag_cols, tmp_col[i]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }

                    case 3: { // DTRSM — banded triangular solve
                        SS_WAIT_IMP2(k, k, 1);

                        const sTiles::SemisparseTileMetaCore& semi_rhs = semisparse_meta[index1];
                        const int active_cols = semi_rhs.sa;
                        if (active_cols > 0) {
                            double* __restrict__ ss_rhs  = chunked_tiles[index1];
                            double* __restrict__ ss_diag = chunked_tiles[index2];
                            const sTiles::SemisparseTileMetaCore& semi_diag = semisparse_meta[index2];
                            const int rows_rhs = tile_meta[index1].height;
                            const int diag_dim = tile_meta[index2].height;
                            const int kd = semi_diag.upper_bw;

                            lapack_int info = LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                                             diag_dim, kd, active_cols,
                                                             ss_diag, kd + 1, ss_rhs, rows_rhs);
                            if (info != 0) {
                                std::fprintf(stderr, "sTiles error: LAPACKE_dtbtrs failed with info=%d (tile m=%d, k=%d)\n",
                                             info, m, k);
                                ss_abort(); break;
                            }
                        }
                        ss_cond_set(m, k, 1);
                        break;
                    }

                    default:
                        break;
                }

                if (ss_aborted()) break;
            }
            ss_finalize();
            #undef SS_WAIT_IMP2
        }

        void dpotrf_expansion_from_chol_tasks_semisparse_imp3_analysis(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int rank = STILES_RANK;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

            double** __restrict__ const chunked_tiles = tiledMatrix->chunkedDenseTiles;
            const sTiles::SemisparseTileMetaCore* __restrict__ const semisparse_meta = tiledMatrix->semisparseTileMetaCore;
            const sTiles::TileMetaCore* __restrict__ const tile_meta = tiledMatrix->tileMetaCore;

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[semisparse_imp3_analysis] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* __restrict__ tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();
            const double mzone = -1.0;

            constexpr int FUSE_THRESHOLD = 16;

            // Precomputed scatter info — required (param[7]==1, default in fast mode).
            // Case 4's dispatch reads scatter_index/scatter_packed unconditionally; the
            // runtime fallback was removed because build_chol_scatter_info is always
            // called by sTiles_preprocess_group_fastmode when param[7]==1.
            const int64_t* scatter_index = tiledMatrix->chol_scatter_index->data();
            const int32_t* scatter_packed = tiledMatrix->chol_scatter_packed->data();

            // ───────────── ANALYSIS COUNTERS (per-rank, stack-local) ─────────────
            std::uint64_t n_case1 = 0, n_case2 = 0, n_case3 = 0, n_case4 = 0;
            std::uint64_t n_case2_fused_contig = 0, n_case2_fused_scat = 0;
            std::uint64_t n_case2_blas_contig  = 0, n_case2_blas_general = 0;
            std::uint64_t n_case2_empty = 0;
            std::uint64_t n_case3_processed = 0, n_case3_empty = 0;
            std::uint64_t n_case4_path[6] = {0, 0, 0, 0, 0, 0};
            std::uint64_t n_p5_eligible = 0;  // cols_a <= FUSE_THRESHOLD
            std::uint64_t n_waits = 0, n_waits_hit = 0;    // hit == ready on first check (zero wait)
            std::uint64_t wait_ns_total = 0;

            // active_cols / dims histogram: 0, 1, 2-4, 5-8, 9-16, 17-32, 33-64, 65+
            auto _bucket = [](int v) -> int {
                if (v <= 0) return 0;
                if (v == 1) return 1;
                if (v <= 4) return 2;
                if (v <= 8) return 3;
                if (v <= 16) return 4;
                if (v <= 32) return 5;
                if (v <= 64) return 6;
                return 7;
            };
            std::uint64_t hist_c2_active[8] = {0};   // DSYRK active_cols
            std::uint64_t hist_c3_active[8] = {0};   // DTRSM active_cols
            std::uint64_t hist_c4_colsA[8]  = {0};   // DGEMM cols_a
            std::uint64_t hist_c4_colsB[8]  = {0};   // DGEMM cols_b
            std::uint64_t hist_c4_rowsA[8]  = {0};   // DGEMM rows_a

            // Timed wait macro — fast path avoids chrono when dep already satisfied.
            #define SS_WAIT_IMP2(m_coord, n_coord, val) \
            { \
                ++n_waits; \
                if (stile->ss_progress[(m_coord) + stile->ss_ld * (n_coord)] == (val)) { \
                    ++n_waits_hit; \
                } else { \
                    int _spin_ct = 0; \
                    auto _w_start = std::chrono::steady_clock::now(); \
                    while (!stile->ss_abort && \
                            stile->ss_progress[(m_coord) + stile->ss_ld * (n_coord)] != (val)) { \
                        hpc_pause_hybrid(_spin_ct); \
                    } \
                    auto _w_end = std::chrono::steady_clock::now(); \
                    wait_ns_total += std::chrono::duration_cast<std::chrono::nanoseconds>(_w_end - _w_start).count(); \
                    if (stile->ss_abort) break; \
                } \
            }

            const auto _fn_start = std::chrono::steady_clock::now();

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);

            const int ntiles_bounds = tiledMatrix->numActiveTiles;

            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                // Prefetching
                if (idx + 1 < end) {
                    const std::array<int,7> &next_t = tasks[idx + 1];
                    const int next_idx1 = next_t[4];
                    const int next_idx2 = next_t[5];
                    if (next_idx1 >= 0 && next_idx1 < ntiles_bounds && chunked_tiles[next_idx1]) {
                        const char* p1 = reinterpret_cast<const char*>(chunked_tiles[next_idx1]);
                        __builtin_prefetch(p1, 0, 3); __builtin_prefetch(p1+64, 0, 3);
                        __builtin_prefetch(p1+128, 0, 3); __builtin_prefetch(p1+192, 0, 3);
                    }
                    if (next_idx2 >= 0 && next_idx2 < ntiles_bounds && chunked_tiles[next_idx2]) {
                        const char* p2 = reinterpret_cast<const char*>(chunked_tiles[next_idx2]);
                        __builtin_prefetch(p2, 1, 3); __builtin_prefetch(p2+64, 1, 3);
                        __builtin_prefetch(p2+128, 1, 3); __builtin_prefetch(p2+192, 1, 3);
                    }
                    if (next_t[0] == 4) {
                        const int next_idx3 = next_t[6];
                        if (next_idx3 >= 0 && next_idx3 < ntiles_bounds && chunked_tiles[next_idx3]) {
                            const char* p3 = reinterpret_cast<const char*>(chunked_tiles[next_idx3]);
                            __builtin_prefetch(p3, 1, 3); __builtin_prefetch(p3+64, 1, 3);
                        }
                    }
                    __builtin_prefetch(&semisparse_meta[next_idx1], 0, 3);
                    __builtin_prefetch(&semisparse_meta[next_idx2], 0, 3);
                }

                switch (myroutine) {

                    case 4: { // DGEMM — C(k,m) -= A(n,k)^T * B(n,m)
                        ++n_case4;
                        SS_WAIT_IMP2(k, n, 1);
                        SS_WAIT_IMP2(m, n, 1);

                        double* __restrict__ chunk_out = chunked_tiles[index3];
                        const int rows_out = tile_meta[index3].height;

                        double* __restrict__ chunk_a = chunked_tiles[index1];
                        double* __restrict__ chunk_b = chunked_tiles[index2];

                        const sTiles::SemisparseTileMetaCore& semi_a = semisparse_meta[index1];
                        const sTiles::SemisparseTileMetaCore& semi_b = semisparse_meta[index2];

                        const int cols_a = semi_a.sa;
                        const int cols_b = semi_b.sa;
                        const int rows_a = tile_meta[index1].height;

                        ++hist_c4_colsA[_bucket(cols_a)];
                        ++hist_c4_colsB[_bucket(cols_b)];
                        ++hist_c4_rowsA[_bucket(rows_a)];

                        // ── Dense path: precomputed path_flag dispatch ──
                        // Precondition: param[7]==1 (default in fast mode), which guarantees
                        // build_chol_scatter_info has populated scatter_index/scatter_packed.
                        const int si_base = idx * 2;
                        const int64_t data_off  = scatter_index[si_base];
                        const int path_flag = static_cast<int>(scatter_index[si_base + 1]);
                        // p5 — small-cols_a sparse update using per-tile CSC.
                        // Mirrors dgemm_sparse_csc_fastpath but gated on cols_a <= FUSE_THRESHOLD
                        // instead of the 10%-density threshold.  Uses semi_a.csc_colptr/csc_rowind
                        // (local row indices, indexed by packed active-column number) to skip
                        // structural zeros in the inner dot-product loop.
                        const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index3];
                        if (cols_a > 0 && cols_a <= FUSE_THRESHOLD) ++n_p5_eligible;
                        if (cols_a > 0 && cols_a <= FUSE_THRESHOLD &&
                            static_cast<int>(semi_a.csc_colptr.size()) >= cols_a + 1 &&
                            static_cast<long long>(semi_a.csc_nnz) * 10 <
                                static_cast<long long>(rows_a) * cols_a) {
                            ++n_case4_path[5];
                            const int* __restrict__ a_cp   = semi_a.csc_colptr.data();
                            const int* __restrict__ a_ri   = semi_a.csc_rowind.data();

                            if (semi_a.is_contiguous && semi_b.is_contiguous && semi_out.is_full_width) {
                                for (int j = 0; j < cols_b; ++j) {
                                    const double* __restrict__ col_b = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(semi_b.fa + j) * rows_out;
                                    for (int i = 0; i < cols_a; ++i) {
                                        const double* __restrict__ col_a = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                        double dot = 0.0;
                                        for (int p = a_cp[i]; p < a_cp[i + 1]; ++p) {
                                            const int r = a_ri[p];
                                            dot += col_a[r] * col_b[r];
                                        }
                                        out_col[semi_a.fa + i] -= dot;
                                    }
                                }
                            } else if (semi_a.is_contiguous) {
                                const int* __restrict__ acol_out = semi_out.acol.data();
                                const int* __restrict__ aind_b   = semi_b.aind.data();
                                for (int j = 0; j < cols_b; ++j) {
                                    const int slot = acol_out[aind_b[j]];
                                    if (slot < 0) continue;
                                    const double* __restrict__ col_b = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                    for (int i = 0; i < cols_a; ++i) {
                                        const double* __restrict__ col_a = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                        double dot = 0.0;
                                        for (int p = a_cp[i]; p < a_cp[i + 1]; ++p) {
                                            const int r = a_ri[p];
                                            dot += col_a[r] * col_b[r];
                                        }
                                        out_col[semi_a.fa + i] -= dot;
                                    }
                                }
                            } else {
                                const int* __restrict__ acol_out = semi_out.acol.data();
                                const int* __restrict__ aind_a   = semi_a.aind.data();
                                const int* __restrict__ aind_b   = semi_b.aind.data();
                                for (int j = 0; j < cols_b; ++j) {
                                    const int slot = acol_out[aind_b[j]];
                                    if (slot < 0) continue;
                                    const double* __restrict__ col_b = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                    for (int i = 0; i < cols_a; ++i) {
                                        const double* __restrict__ col_a = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                        double dot = 0.0;
                                        for (int p = a_cp[i]; p < a_cp[i + 1]; ++p) {
                                            const int r = a_ri[p];
                                            dot += col_a[r] * col_b[r];
                                        }
                                        out_col[aind_a[i]] -= dot;
                                    }
                                }
                            }
                        } else {
                            if (path_flag >= 0 && path_flag <= 4) ++n_case4_path[path_flag];

                            if (path_flag == 0) {
                                const int a_start = semi_a.fa;
                                const int b_start = semi_b.fa;
                                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                            cols_a, cols_b, rows_a, mzone,
                                            chunk_a, rows_a, chunk_b, rows_a,
                                            1.0, chunk_out + a_start + static_cast<std::size_t>(b_start) * rows_out, rows_out);

                            } else if (path_flag == 1) {
                                const int a_start = semi_a.fa;
                                const int32_t* slots = scatter_packed + data_off;
                                for (int j = 0; j < cols_b; ++j) {
                                    const int slot = slots[j];
                                    if (slot < 0) continue;
                                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                    const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                    for (int i = 0; i < cols_a; ++i) {
                                        const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                        double dot = 0.0;
                                        for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                        out_col[a_start + i] -= dot;
                                    }
                                }

                            } else if (path_flag == 2) {
                                const int* __restrict__ aind_a = semi_a.aind.data();
                                const int32_t* slots = scatter_packed + data_off;
                                for (int j = 0; j < cols_b; ++j) {
                                    const int slot = slots[j];
                                    if (slot < 0) continue;
                                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                    const double* __restrict__ col_b_j = chunk_b + static_cast<std::size_t>(j) * rows_a;
                                    for (int i = 0; i < cols_a; ++i) {
                                        const double* __restrict__ col_a_i = chunk_a + static_cast<std::size_t>(i) * rows_a;
                                        double dot = 0.0;
                                        for (int r = 0; r < rows_a; ++r) dot += col_a_i[r] * col_b_j[r];
                                        out_col[aind_a[i]] -= dot;
                                    }
                                }

                            } else if (path_flag == 3) {
                                const int ld_tmp = cols_a;
                                const int a_start = semi_a.fa;
                                const int32_t* slots = scatter_packed + data_off;
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                   cols_a, cols_b, rows_a,
                                                   mzone, chunk_a, rows_a, chunk_b, rows_a,
                                                   0.0, tmp_tile, ld_tmp);
                                // Unrolled scatter (contiguous) — avoids daxpy call overhead on small cols_a
                                for (int j = 0; j < cols_b; ++j) {
                                    const int slot = slots[j];
                                    if (slot < 0) continue;
                                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out + a_start;
                                    const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                    int i = 0;
                                    for (; i + 3 < cols_a; i += 4) {
                                        out_col[i]     += tmp_col[i];
                                        out_col[i + 1] += tmp_col[i + 1];
                                        out_col[i + 2] += tmp_col[i + 2];
                                        out_col[i + 3] += tmp_col[i + 3];
                                    }
                                    for (; i < cols_a; ++i) out_col[i] += tmp_col[i];
                                }

                            } else if (path_flag == 4) {
                                const int ld_tmp = cols_a;
                                const int* __restrict__ aind_a = semi_a.aind.data();
                                const int32_t* slots = scatter_packed + data_off;
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                   cols_a, cols_b, rows_a,
                                                   mzone, chunk_a, rows_a, chunk_b, rows_a,
                                                   0.0, tmp_tile, ld_tmp);
                                for (int j = 0; j < cols_b; ++j) {
                                    const int slot = slots[j];
                                    if (slot < 0) continue;
                                    double* __restrict__ out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                    const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                    int i = 0;
                                    for (; i + 3 < cols_a; i += 4) {
                                        out_col[aind_a[i]]     += tmp_col[i];
                                        out_col[aind_a[i + 1]] += tmp_col[i + 1];
                                        out_col[aind_a[i + 2]] += tmp_col[i + 2];
                                        out_col[aind_a[i + 3]] += tmp_col[i + 3];
                                    }
                                    for (; i < cols_a; ++i) out_col[aind_a[i]] += tmp_col[i];
                                }
                            }
                        }
                        break;
                    }

                    case 1: { // DPOTRF — banded Cholesky on diagonal tile
                        ++n_case1;
                        // Diagonal tile: chunked_tiles[index1] is allocated with elems = height*(upper_bw+1),
                        // so it's never null for an active diagonal — no null guard needed.
                        double* __restrict__ tile_out = chunked_tiles[index1];
                        const sTiles::SemisparseTileMetaCore& semi = semisparse_meta[index1];
                        const sTiles::TileMetaCore& meta = tile_meta[index1];
                        sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(meta.height, semi.upper_bw, tile_out);
                        if (ssstatus != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                            ss_abort(); break;
                        }
                        ss_cond_set(k, k, 1);
                        break;
                    }

                    case 2: { // DSYRK — symmetric rank-k update into banded diagonal
                        ++n_case2;
                        SS_WAIT_IMP2(k, n, 1);

                        double* __restrict__ tile_in = chunked_tiles[index1];
                        double* __restrict__ tile_out = chunked_tiles[index2];

                        if (tile_in && tile_out) {
                            const sTiles::SemisparseTileMetaCore& semi_in = semisparse_meta[index1];
                            const int active_cols = semi_in.sa;
                            ++hist_c2_active[_bucket(active_cols)];

                            if (active_cols > 0) {
                                const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index2];
                                const int diag_cols = semi_out.upper_bw + 1;
                                const int rows_in = (tile_meta[index1].height > 0) ? tile_meta[index1].height : tile_size;
                                const int base_row = diag_cols - 1;

                                const bool aind_contig = semi_in.is_contiguous;

                                if (active_cols <= FUSE_THRESHOLD && aind_contig && active_cols <= diag_cols) {
                                    ++n_case2_fused_contig;
                                    // === FUSED contiguous: dot products directly into banded output ===
                                    const int aind_start = semi_in.fa;
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_start + j;
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                        for (int i = 0; i <= j; ++i) {
                                            const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                            double dot = 0.0;
                                            for (int r = 0; r < rows_in; ++r) {
                                                dot += col_i[r] * col_j[r];
                                            }
                                            tile_out[base_row - (j - i) + col_base] -= dot;
                                        }
                                    }

                                } else if (active_cols <= FUSE_THRESHOLD) {
                                    ++n_case2_fused_scat;
                                    // === FUSED scattered: dot products with aind lookups ===
                                    const int* __restrict__ aind = semi_in.aind.data();
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind[j];
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ col_j = tile_in + static_cast<std::size_t>(j) * rows_in;
                                        for (int i = 0; i <= j; ++i) {
                                            const int band = col_idx - aind[i];
                                            if (band < diag_cols) {
                                                const double* __restrict__ col_i = tile_in + static_cast<std::size_t>(i) * rows_in;
                                                double dot = 0.0;
                                                for (int r = 0; r < rows_in; ++r) {
                                                    dot += col_i[r] * col_j[r];
                                                }
                                                tile_out[base_row - band + col_base] -= dot;
                                            }
                                        }
                                    }

                                } else if (aind_contig && active_cols <= diag_cols) {
                                    ++n_case2_blas_contig;
                                    // === BLAS dsyrk + contiguous scatter (branch-free inner loop) ===
                                    const int ld_tmp = active_cols;
                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                       active_cols, rows_in, mzone,
                                                       tile_in, rows_in, 0.0, tmp_tile, ld_tmp);

                                    const int aind_start = semi_in.fa;
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_start + j;
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        for (int i = 0; i <= j; ++i) {
                                            // band = j - i, always < diag_cols (guaranteed by outer guard)
                                            tile_out[base_row - (j - i) + col_base] += tmp_col[i];
                                        }
                                    }

                                } else {
                                    ++n_case2_blas_general;
                                    // === BLAS dsyrk + general scatter (same as _imp1) ===
                                    const int ld_tmp = active_cols;
                                    const int* __restrict__ aind = semi_in.aind.data();
                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                       active_cols, rows_in, mzone,
                                                       tile_in, rows_in, 0.0, tmp_tile, ld_tmp);

                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind[j];
                                        const std::size_t col_base = static_cast<std::size_t>(diag_cols) * col_idx;
                                        const double* __restrict__ tmp_col = tmp_tile + j * ld_tmp;
                                        for (int i = 0; i <= j; ++i) {
                                            const int band = col_idx - aind[i];
                                            if (band < diag_cols) {
                                                tile_out[base_row - band + col_base] += tmp_col[i];
                                            } else if (std::fabs(tmp_col[i]) > 1e-15) {
                                                std::fprintf(stderr, "sTiles warning: DSYRK bandwidth overflow in tile k=%d: band=%d >= diag_cols=%d, dropped=%.6e\n",
                                                             k, band, diag_cols, tmp_col[i]);
                                            }
                                        }
                                    }
                                }
                            } else {
                                ++n_case2_empty;
                            }
                        }
                        break;
                    }

                    case 3: { // DTRSM — banded triangular solve
                        ++n_case3;
                        // Same invariant as case 2: ss_rhs is null iff active_cols==0.
                        // ss_diag is the diagonal, never null. Single guard suffices.
                        SS_WAIT_IMP2(k, k, 1);

                        const sTiles::SemisparseTileMetaCore& semi_rhs = semisparse_meta[index1];
                        const int active_cols = semi_rhs.sa;
                        ++hist_c3_active[_bucket(active_cols)];
                        if (active_cols > 0) {
                            ++n_case3_processed;
                            double* __restrict__ ss_rhs  = chunked_tiles[index1];
                            double* __restrict__ ss_diag = chunked_tiles[index2];
                            const sTiles::SemisparseTileMetaCore& semi_diag = semisparse_meta[index2];
                            const int rows_rhs = tile_meta[index1].height;
                            const int diag_dim = tile_meta[index2].height;
                            const int kd = semi_diag.upper_bw;

                            lapack_int info = LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                                             diag_dim, kd, active_cols,
                                                             ss_diag, kd + 1, ss_rhs, rows_rhs);
                            if (info != 0) {
                                std::fprintf(stderr, "sTiles error: LAPACKE_dtbtrs failed with info=%d (tile m=%d, k=%d)\n",
                                             info, m, k);
                                ss_abort(); break;
                            }
                        } else {
                            ++n_case3_empty;
                        }
                        ss_cond_set(m, k, 1);
                        break;
                    }

                    default:
                        break;
                }

                if (ss_aborted()) break;
            }
            ss_finalize();
            #undef SS_WAIT_IMP2

            // ───────────── ANALYSIS SUMMARY ─────────────
            const auto _fn_end = std::chrono::steady_clock::now();
            const std::uint64_t total_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(_fn_end - _fn_start).count();
            const std::uint64_t compute_ns = (total_ns > wait_ns_total) ? (total_ns - wait_ns_total) : 0;
            const double total_ms   = total_ns   / 1.0e6;
            const double wait_ms    = wait_ns_total / 1.0e6;
            const double compute_ms = compute_ns / 1.0e6;
            const double wait_pct   = (total_ns > 0) ? (100.0 * wait_ns_total / total_ns) : 0.0;

            const std::uint64_t n_tasks = static_cast<std::uint64_t>(end - start);
            const std::uint64_t n_case2_paths_sum =
                n_case2_fused_contig + n_case2_fused_scat +
                n_case2_blas_contig  + n_case2_blas_general;

            // Build the summary into a single buffer so prints from different ranks don't interleave.
            std::ostringstream os;
            os << "\n===== [semisparse_imp3_analysis] rank " << rank
               << " / " << STILES_SIZE << " =====\n";
            os << "  tasks:           " << n_tasks
               << "  (range [" << start << ", " << end << "))\n";
            os << "  time:            total=" << std::fixed << std::setprecision(3) << total_ms
               << " ms   compute=" << compute_ms
               << " ms   wait=" << wait_ms << " ms (" << std::setprecision(1) << wait_pct
               << "% of total)\n";
            os << "  waits:           total=" << n_waits
               << "   zero-wait=" << n_waits_hit
               << "   blocked=" << (n_waits - n_waits_hit);
            if (n_waits > n_waits_hit) {
                const double avg_wait_us = wait_ns_total / 1000.0 / (n_waits - n_waits_hit);
                os << "   avg-blocked=" << std::setprecision(2) << avg_wait_us << " us";
            }
            os << "\n";
            os << "  case counts:     DPOTRF(1)=" << n_case1
               << "  DSYRK(2)=" << n_case2
               << "  DTRSM(3)=" << n_case3
               << "  DGEMM(4)=" << n_case4 << "\n";

            os << "  DSYRK paths:     fused_contig="  << n_case2_fused_contig
               << "  fused_scat="     << n_case2_fused_scat
               << "  blas_contig="    << n_case2_blas_contig
               << "  blas_general="   << n_case2_blas_general
               << "  empty="          << n_case2_empty;
            if (n_case2_paths_sum > 0) {
                os << "\n                   (fused%="
                   << std::setprecision(1)
                   << (100.0 * (n_case2_fused_contig + n_case2_fused_scat) / n_case2_paths_sum)
                   << ", blas%="
                   << (100.0 * (n_case2_blas_contig + n_case2_blas_general) / n_case2_paths_sum)
                   << ")";
            }
            os << "\n";

            os << "  DTRSM paths:     processed=" << n_case3_processed
               << "  empty="     << n_case3_empty << "\n";

            os << "  DGEMM paths:     p0(full dense BLAS)=" << n_case4_path[0]
               << "  p1(no-BLAS contig)=" << n_case4_path[1]
               << "  p2(no-BLAS scat)="   << n_case4_path[2]
               << "  p3(BLAS+contig)="    << n_case4_path[3]
               << "  p4(BLAS+scat)="      << n_case4_path[4]
               << "  p5(exact-L sparse)=" << n_case4_path[5]
               << "  (eligible=" << n_p5_eligible << ")\n";

            // Histograms (buckets: 0, 1, 2-4, 5-8, 9-16, 17-32, 33-64, 65+)
            const char* bucket_labels[8] = {
                "0", "1", "2-4", "5-8", "9-16", "17-32", "33-64", "65+"
            };
            auto dump_hist = [&](const char* name, const std::uint64_t* h) {
                os << "  " << name;
                std::size_t pad = std::strlen(name);
                while (pad++ < 15) os << ' ';
                os << ":";
                for (int b = 0; b < 8; ++b) {
                    os << "  " << bucket_labels[b] << "=" << h[b];
                }
                os << "\n";
            };
            dump_hist("hist DSYRK.ac",  hist_c2_active);
            dump_hist("hist DTRSM.ac",  hist_c3_active);
            dump_hist("hist DGEMM.cA",  hist_c4_colsA);
            dump_hist("hist DGEMM.cB",  hist_c4_colsB);
            dump_hist("hist DGEMM.rA",  hist_c4_rowsA);
            os << "=====================================================\n";

            std::fprintf(stderr, "%s", os.str().c_str());
            std::fflush(stderr);
        }

        void dpotrf_expansion_from_chol_tasks_semisparse_draft(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int rank = STILES_RANK;

            if(STILES_SIZE==1){

                const int N = tiledMatrix->dim;
                const int tile_size = tiledMatrix->tile_size;
                const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

                const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
                const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
                const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
                const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());
                const bool *diag_map = tiledMatrix->diagonal_bmapper;

                if (rank >= tiledMatrix->num_workspaces) {
                    sTiles::Logger::error("[semisparse_omp_serial] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                    return;
                }
                // Use restrict to help compiler optimize
                double* __restrict__ tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();

                // Cache commonly accessed pointers
                double** const chunked_tiles = tiledMatrix->chunkedDenseTiles;
                const sTiles::SemisparseTileMetaCore* const semisparse_meta = tiledMatrix->semisparseTileMetaCore;
                const sTiles::TileMetaCore* const tile_meta = tiledMatrix->tileMetaCore;

                const double mzone = -1.0;

                for (int idx = start; idx < end; ++idx) {
                    const std::array<int,7> &t = tasks[idx];
                    const int myroutine = t[0];
                    const int m = t[1];
                    const int k = t[2];
                    const int n = t[3];
                    const int index1 = t[4];
                    const int index2 = t[5];
                    const int index3 = t[6];


                    //if(rank==1) print_tasks(rank, myroutine, m, k, n, index1, index2, index3);
                    
                    switch (myroutine) {
                        case 1: { // DPOTRF
                            double *ssstile_out = chunked_tiles[index1];

                            if (ssstile_out) {
                                const sTiles::SemisparseTileMetaCore& semi = semisparse_meta[index1];
                                const sTiles::TileMetaCore& meta = tile_meta[index1];
                                const int kd = semi.upper_bw;
                                const int rows = meta.height;

                                sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(rows, kd, ssstile_out);
                                if (ssstatus != sTiles::StatusCode::Success) {
                                    std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                                    stile->ss_abort = 1;
                                    return;
                                }
                            }
                            break;
                        }
                        case 2: { // DSYRK - OPTIMIZED

                            double *ssstile_in = chunked_tiles[index1];
                            double *ssstile_out = chunked_tiles[index2];

                            if (ssstile_in && ssstile_out) {
                                const sTiles::SemisparseTileMetaCore& semi_in = semisparse_meta[index1];
                                const int active_cols = semi_in.sa;

                                if (active_cols > 0) {
                                    const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index2];
                                    const sTiles::TileMetaCore& meta_in = tile_meta[index1];
                                    const sTiles::TileMetaCore& meta_out = tile_meta[index2];
                                    // Use actual tile height, not nominal tile_size (fixes buffer overrun for boundary tiles)
                                    const int rows_in = (meta_in.height > 0) ? meta_in.height : tile_size;

                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                    active_cols, rows_in, mzone, ssstile_in,
                                                    rows_in, 0.0, tmp_tile, active_cols);

                                    // Cache frequently accessed values
                                    const int* const aind_ptr = semi_in.aind.data();
                                    const int aind_size = static_cast<int>(semi_in.aind.size());
                                    const int cols_out = (meta_out.width > 0) ? meta_out.width : tile_size;
                                    const int diag_cols = (semi_out.upper_bw >= 0) ? (semi_out.upper_bw + 1) : cols_out;
                                    const int ld_tmp = active_cols;

                                    // Optimized scatter: reduced branching, pointer arithmetic
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_ptr[j];
                                        // Single bounds check per column
                                        if (col_idx < 0 || col_idx >= cols_out) continue;

                                        const double* const tmp_col = tmp_tile + j * ld_tmp;

                                        // Inner loop with minimal branching
                                        const int max_i = (j < aind_size) ? j : aind_size - 1;
                                        for (int i = 0; i <= max_i; ++i) {
                                            const int row_idx = aind_ptr[i];
                                            // Only check upper triangular bound
                                            if (row_idx > col_idx) continue;

                                            const int band = col_idx - row_idx;
                                            if (band >= diag_cols) {
                                                if (std::fabs(tmp_col[i]) > 1e-15) {
                                                    std::fprintf(stderr, "sTiles warning: DSYRK bandwidth overflow in tile k=%d: band=%d >= diag_cols=%d, dropped=%.6e\n",
                                                                 k, band, diag_cols, tmp_col[i]);
                                                }
                                                continue;
                                            }
                                            const int lapack_row = diag_cols - 1 - band;
                                            const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                                                    + static_cast<std::size_t>(diag_cols)
                                                                    * static_cast<std::size_t>(col_idx);
                                            ssstile_out[offset] += tmp_col[i];
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case 3: { // DTRSM

                            double* ss_rhs  = chunked_tiles[index1];
                            double* ss_diag = chunked_tiles[index2];

                            if (ss_rhs && ss_diag) {
                                const sTiles::SemisparseTileMetaCore& semi_rhs = semisparse_meta[index1];
                                const sTiles::SemisparseTileMetaCore& semi_diag = semisparse_meta[index2];
                                const sTiles::TileMetaCore& meta_rhs = tile_meta[index1];
                                const sTiles::TileMetaCore& meta_diag = tile_meta[index2];
                                const int active_cols = semi_rhs.sa;

                                if (active_cols > 0) {
                                    const int rows_rhs = (meta_rhs.height > 0) ? meta_rhs.height : tile_size;
                                    const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tile_size;
                                    const int diag_bw = (semi_diag.upper_bw >= 0) ? semi_diag.upper_bw : 0;
                                    const int kd = diag_bw;
                                    const int ldab = diag_bw + 1;

                                    LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                                   diag_dim, kd, active_cols,
                                                   ss_diag, ldab, ss_rhs, rows_rhs);
                                }
                            }

                            break;
                        }
                        case 4: { // DGEMM - OPTIMIZED

                            double* chunk_a   = chunked_tiles[index1];
                            double* chunk_b   = chunked_tiles[index2];
                            double* chunk_out = chunked_tiles[index3];

                            if (chunk_a && chunk_b && chunk_out) {
                                const sTiles::SemisparseTileMetaCore& semi_a = semisparse_meta[index1];
                                const sTiles::SemisparseTileMetaCore& semi_b = semisparse_meta[index2];
                                const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index3];
                                const sTiles::TileMetaCore& meta_out = tile_meta[index3];

                                const int cols_a = semi_a.sa;
                                const int cols_b = semi_b.sa;
                                const int rows_a = tile_meta[index1].height;
                                const int rows_out = meta_out.height;
                                const int cols_out = meta_out.width;

                                if (cols_a > 0 && cols_b > 0 && rows_a > 0) {
                                    const int ld_tmp = cols_a;

                                    sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                    cols_a, cols_b, rows_a,
                                                    mzone, chunk_a, rows_a,
                                                    chunk_b, rows_a,
                                                    0.0, tmp_tile, ld_tmp);

                                    // Cache pointers and sizes to reduce indirection
                                    const int* const aind_a_ptr = semi_a.aind.data();
                                    const int* const aind_b_ptr = semi_b.aind.data();
                                    const int* const acol_map_ptr = semi_out.acol.data();
                                    const int acol_map_size = static_cast<int>(semi_out.acol.size());
                                    const int aind_a_size = static_cast<int>(semi_a.aind.size());
                                    const int semi_out_sa = semi_out.sa;

                                    // Optimized scatter loop
                                    for (int j = 0; j < cols_b; ++j) {
                                        const int global_col = aind_b_ptr[j];

                                        // Combined bounds check
                                        if (global_col < 0 || global_col >= cols_out || global_col >= acol_map_size)
                                            continue;

                                        const int slot = acol_map_ptr[global_col];
                                        if (slot < 0 || slot >= semi_out_sa) continue;

                                        // Cache output column pointer and tmp column pointer
                                        double* const out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                        const double* const tmp_col = tmp_tile + ld_tmp * j;

                                        // Inner loop with vectorization potential
                                        for (int i = 0; i < cols_a; ++i) {
                                            const int global_row = aind_a_ptr[i];
                                            // Compiler can potentially vectorize with simple bounds check
                                            if (global_row >= 0 && global_row < rows_out) {
                                                out_col[global_row] += tmp_col[i];
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        default:
                            break;
                    }
                }

                #ifdef ESMAIL_DEBUG
                if (rank == 0) {
                    const int nnz = tiledMatrix->original_nnz;
                    const char* export_path = sTiles_get_dense_export_file();
                    if (export_path && export_path[0] != '\0' && nnz > 0) {
                        std::vector<double> y_dense(static_cast<std::size_t>(nnz));
                        sTiles::preprocess::export_x_semisparse_tiles_serial(tiledMatrix, y_dense.data());
                        sTiles::Logger::info("[dense-export] Writing ", nnz, " values to ", export_path);
                        const auto export_begin = std::chrono::steady_clock::now();
                        if (!write_dense_values_to_txt(export_path, y_dense)) {
                            sTiles::Logger::warning("[dense-export] Failed to write nnz vector to ", export_path);
                        } else {
                            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - export_begin).count();
                            sTiles::Logger::info("[dense-export] Completed in ", secs, " s");
                        }
                    }
                }
                #endif

            }else{

                const int N = tiledMatrix->dim;
                const int tile_size = tiledMatrix->tile_size;
                const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

                // Early exit if no chunked tiles
                if (!tiledMatrix->chunkedDenseTiles) {
                    ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
                    ss_finalize();
                    return;
                }

                const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
                const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
                const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
                const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());
                const bool *diag_map = tiledMatrix->diagonal_bmapper;

                if (rank >= tiledMatrix->num_workspaces) {
                    sTiles::Logger::error("[semisparse_omp] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                    return;
                }
                // Use restrict to help compiler optimize
                double* __restrict__ tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();

                // Cache commonly accessed pointers
                double** const chunked_tiles = tiledMatrix->chunkedDenseTiles;
                const sTiles::SemisparseTileMetaCore* const semisparse_meta = tiledMatrix->semisparseTileMetaCore;
                const sTiles::TileMetaCore* const tile_meta = tiledMatrix->tileMetaCore;

                const double mzone = -1.0;

                // constexpr int kMaxTasksToPrint = 10;
                // int printed_tasks = 0;

                ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
                for (int idx = start; idx < end; ++idx) {
                    const std::array<int,7> &t = tasks[idx];
                    const int myroutine = t[0];
                    const int m = t[1];
                    const int k = t[2];
                    const int n = t[3];
                    const int index1 = t[4];
                    const int index2 = t[5];
                    const int index3 = t[6];

                    // if (printed_tasks < kMaxTasksToPrint) {
                    //     print_tasks(rank, myroutine, m, k, n, index1, index2, index3);
                    //     ++printed_tasks;
                    // }

                    switch (myroutine) {
                        case 1: { // DPOTRF
                            double *ssstile_out = chunked_tiles[index1];

                            if (ssstile_out) {
                                const sTiles::SemisparseTileMetaCore& semi = semisparse_meta[index1];
                                const sTiles::TileMetaCore& meta = tile_meta[index1];
                                const int kd = semi.upper_bw;
                                const int rows = meta.height;

                                sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(rows, kd, ssstile_out);
                                if (ssstatus != sTiles::StatusCode::Success) {
                                    std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                                    ss_abort(); break;
                                }
                            }
                            ss_cond_set(k, k, 1);
                            break;
                        }
                        case 2: { // DSYRK - OPTIMIZED
                            ss_cond_wait(k, n, 1);

                            double *ssstile_in = chunked_tiles[index1];
                            double *ssstile_out = chunked_tiles[index2];

                            if (ssstile_in && ssstile_out) {
                                const sTiles::SemisparseTileMetaCore& semi_in = semisparse_meta[index1];
                                const int active_cols = semi_in.sa;

                                if (active_cols > 0) {
                                    const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index2];
                                    const sTiles::TileMetaCore& meta_in = tile_meta[index1];
                                    const sTiles::TileMetaCore& meta_out = tile_meta[index2];
                                    // Use actual tile height, not nominal tile_size (fixes buffer overrun for boundary tiles)
                                    const int rows_in = (meta_in.height > 0) ? meta_in.height : tile_size;

                                    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                                    active_cols, rows_in, mzone, ssstile_in,
                                                    rows_in, 0.0, tmp_tile, active_cols);

                                    // Cache frequently accessed values
                                    const int* const aind_ptr = semi_in.aind.data();
                                    const int aind_size = static_cast<int>(semi_in.aind.size());
                                    const int cols_out = (meta_out.width > 0) ? meta_out.width : tile_size;
                                    const int diag_cols = (semi_out.upper_bw >= 0) ? (semi_out.upper_bw + 1) : cols_out;
                                    const int ld_tmp = active_cols;

                                    // Optimized scatter: reduced branching, pointer arithmetic
                                    for (int j = 0; j < active_cols; ++j) {
                                        const int col_idx = aind_ptr[j];
                                        // Single bounds check per column
                                        if (col_idx < 0 || col_idx >= cols_out) continue;

                                        const double* const tmp_col = tmp_tile + j * ld_tmp;

                                        // Inner loop with minimal branching
                                        const int max_i = (j < aind_size) ? j : aind_size - 1;
                                        for (int i = 0; i <= max_i; ++i) {
                                            const int row_idx = aind_ptr[i];
                                            // Only check upper triangular bound
                                            if (row_idx > col_idx) continue;

                                            const int band = col_idx - row_idx;
                                            if (band >= diag_cols) {
                                                if (std::fabs(tmp_col[i]) > 1e-15) {
                                                    std::fprintf(stderr, "sTiles warning: DSYRK bandwidth overflow in tile k=%d: band=%d >= diag_cols=%d, dropped=%.6e\n",
                                                                 k, band, diag_cols, tmp_col[i]);
                                                }
                                                continue;
                                            }
                                            const int lapack_row = diag_cols - 1 - band;
                                            const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                                                    + static_cast<std::size_t>(diag_cols)
                                                                    * static_cast<std::size_t>(col_idx);
                                            ssstile_out[offset] += tmp_col[i];
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case 3: { // DTRSM
                            ss_cond_wait(k, k, 1);

                            double* ss_rhs  = chunked_tiles[index1];
                            double* ss_diag = chunked_tiles[index2];

                            if (ss_rhs && ss_diag) {
                                const sTiles::SemisparseTileMetaCore& semi_rhs = semisparse_meta[index1];
                                const sTiles::SemisparseTileMetaCore& semi_diag = semisparse_meta[index2];
                                const sTiles::TileMetaCore& meta_rhs = tile_meta[index1];
                                const sTiles::TileMetaCore& meta_diag = tile_meta[index2];
                                const int active_cols = semi_rhs.sa;

                                if (active_cols > 0) {
                                    const int rows_rhs = (meta_rhs.height > 0) ? meta_rhs.height : tile_size;
                                    const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tile_size;
                                    const int diag_bw = (semi_diag.upper_bw >= 0) ? semi_diag.upper_bw : 0;
                                    const int kd = diag_bw;
                                    const int ldab = diag_bw + 1;

                                    LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                                diag_dim, kd, active_cols,
                                                ss_diag, ldab, ss_rhs, rows_rhs);
                                }
                            }

                            ss_cond_set(m, k, 1);
                            break;
                        }
                        case 4: { // DGEMM - OPTIMIZED
                            ss_cond_wait(k, n, 1);
                            ss_cond_wait(m, n, 1);

                            double* chunk_a   = chunked_tiles[index1];
                            double* chunk_b   = chunked_tiles[index2];
                            double* chunk_out = chunked_tiles[index3];

                            if (chunk_a && chunk_b && chunk_out) {
                                const sTiles::SemisparseTileMetaCore& semi_a = semisparse_meta[index1];
                                const sTiles::SemisparseTileMetaCore& semi_b = semisparse_meta[index2];
                                const sTiles::SemisparseTileMetaCore& semi_out = semisparse_meta[index3];
                                const sTiles::TileMetaCore& meta_out = tile_meta[index3];

                                const int cols_a = semi_a.sa;
                                const int cols_b = semi_b.sa;
                                const int rows_a = tile_meta[index1].height;
                                const int rows_out = meta_out.height;
                                const int cols_out = meta_out.width;

                                if (cols_a > 0 && cols_b > 0 && rows_a > 0) {
                                    const int ld_tmp = cols_a;

                                    sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                    cols_a, cols_b, rows_a,
                                                    mzone, chunk_a, rows_a,
                                                    chunk_b, rows_a,
                                                    0.0, tmp_tile, ld_tmp);

                                    // Cache pointers and sizes to reduce indirection
                                    const int* const aind_a_ptr = semi_a.aind.data();
                                    const int* const aind_b_ptr = semi_b.aind.data();
                                    const int* const acol_map_ptr = semi_out.acol.data();
                                    const int acol_map_size = static_cast<int>(semi_out.acol.size());
                                    const int aind_a_size = static_cast<int>(semi_a.aind.size());
                                    const int semi_out_sa = semi_out.sa;

                                    // Optimized scatter loop
                                    for (int j = 0; j < cols_b; ++j) {
                                        const int global_col = aind_b_ptr[j];

                                        // Combined bounds check
                                        if (global_col < 0 || global_col >= cols_out || global_col >= acol_map_size)
                                            continue;

                                        const int slot = acol_map_ptr[global_col];
                                        if (slot < 0 || slot >= semi_out_sa) continue;

                                        // Cache output column pointer and tmp column pointer
                                        double* const out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                        const double* const tmp_col = tmp_tile + ld_tmp * j;

                                        // Inner loop with vectorization potential
                                        for (int i = 0; i < cols_a; ++i) {
                                            const int global_row = aind_a_ptr[i];
                                            // Compiler can potentially vectorize with simple bounds check
                                            if (global_row >= 0 && global_row < rows_out) {
                                                out_col[global_row] += tmp_col[i];
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        default:
                            break;
                    }
                    if (ss_aborted()) break;
                }

                ss_finalize();

                #ifdef ESMAIL_DEBUG
                if (rank == 0) {
                    const int nnz = tiledMatrix->original_nnz;
                    const char* export_path = sTiles_get_dense_export_file();
                    if (export_path && export_path[0] != '\0' && nnz > 0) {
                        std::vector<double> y_dense(static_cast<std::size_t>(nnz));
                        sTiles::preprocess::export_x_semisparse_tiles_serial(tiledMatrix, y_dense.data());
                        sTiles::Logger::info("[dense-export] Writing ", nnz, " values to ", export_path);
                        const auto export_begin = std::chrono::steady_clock::now();
                        if (!write_dense_values_to_txt(export_path, y_dense)) {
                            sTiles::Logger::warning("[dense-export] Failed to write nnz vector to ", export_path);
                        } else {
                            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - export_begin).count();
                            sTiles::Logger::info("[dense-export] Completed in ", secs, " s");
                        }
                    }
                }
                #endif

                // Print full matrix: tile numbers and their elements
                if (false && rank == 0) {  // Only print from rank 0 to avoid duplicates
                    std::cout << "\n=== FULL MATRIX AFTER CHOLESKY ===" << std::endl;
                    const int total_tiles = num_tiles_per_dim * num_tiles_per_dim;

                    for (int tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
                        if (!tiledMatrix->chunkedDenseTiles || !tiledMatrix->chunkedDenseTiles[tile_idx]) {
                            continue;  // Skip inactive tiles
                        }

                        const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[tile_idx];
                        const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[tile_idx];
                        const double* tile_data = tiledMatrix->chunkedDenseTiles[tile_idx];

                        const int tile_row = tile_idx / num_tiles_per_dim;
                        const int tile_col = tile_idx % num_tiles_per_dim;
                        const int rows = meta.height;
                        const int cols = semi.sa;  // Active columns in semisparse format
                        const int upper_bw = semi.upper_bw;

                        std::cout << "\nTile [" << tile_row << "," << tile_col << "] (index=" << tile_idx
                                  << ", rows=" << rows << ", active_cols=" << cols << ", bw=" << upper_bw << "):" << std::endl;

                        if (upper_bw >= 0) {
                            // Banded LAPACK format (diagonal tile)
                            const int ldab = upper_bw + 1;
                            for (int i = 0; i < ldab && i < rows; ++i) {
                                std::cout << "  Band " << i << ": ";
                                // For banded diagonal tiles, print all rows (ignore semi.sa which is 0 for diagonals)
                                for (int j = 0; j < rows; ++j) {
                                    std::cout << std::setw(12) << std::setprecision(6) << tile_data[i + j * ldab] << " ";
                                }
                                std::cout << std::endl;
                            }
                        } else {
                            // Regular semisparse format (off-diagonal tile)
                            for (int i = 0; i < rows; ++i) {
                                std::cout << "  Row " << i << ": ";
                                for (int j = 0; j < cols; ++j) {
                                    std::cout << std::setw(12) << std::setprecision(6) << tile_data[i + j * rows] << " ";
                                }
                                std::cout << std::endl;
                            }
                        }
                    }
                    std::cout << "\n=== END FULL MATRIX ===" << std::endl;

                    std::cout << "\n=== DIAGONAL TILE SUMMARY ===" << std::endl;
                    for (int tile = 0; tile < num_tiles_per_dim; ++tile) {
                        const int diag_idx = tile * num_tiles_per_dim + tile;
                        std::cout << "Diagonal tile [" << tile << "," << tile << "] (index="
                                  << diag_idx << ")";

                        const bool has_diag_flag = tiledMatrix->diagonal_bmapper &&
                                                    tiledMatrix->diagonal_bmapper[diag_idx];
                        if (has_diag_flag) {
                            std::cout << " uses diagonal_bmapper";
                        }

                        double* diag_data = (tiledMatrix->chunkedDenseTiles &&
                                             tiledMatrix->chunkedDenseTiles[diag_idx])
                                               ? tiledMatrix->chunkedDenseTiles[diag_idx]
                                               : nullptr;

                        if (!diag_data) {
                            std::cout << " -> NO DATA" << std::endl;
                            continue;
                        }

                        const auto& meta = tiledMatrix->semisparseTileMetaCore[diag_idx];
                        std::cout << " -> bw=" << meta.upper_bw
                                  << ", active_cols=" << meta.sa;
                        if (meta.upper_bw >= 0 && meta.sa > 0) {
                            std::cout << ", first=" << diag_data[0];
                        }
                        std::cout << std::endl;
                    }
                    std::cout << "=== END DIAGONAL SUMMARY ===" << std::endl;
                }
            }
        }

        void dpotrf_expansion_from_chol_tasks_semisparse_numeric(TiledMatrix *tiledMatrix, stiles_context_t *stile){
            // ==============================================================================
            // Testing function: Compare dense vs semisparse for ALL cases (1-4)
            // ==============================================================================
            // Purpose: Validate that semisparse implementation produces correct results
            // Note: Diagonal tiles in semisparse are stored in banded format (LAPACK-style)
            // ==============================================================================

            const int rank = STILES_RANK;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());
            const bool *diag_map = tiledMatrix->diagonal_bmapper;

            if (rank >= tiledMatrix->num_workspaces) {
                sTiles::Logger::error("[semisparse_test] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
                return;
            }
            double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();

            std::cout << "\n╔═══════════════════════════════════════════════════════════════════════╗\n";
            std::cout <<   "║   SEMISPARSE NUMERIC TESTING: Dense vs Semisparse (ALL CASES)        ║\n";
            std::cout <<   "╚═══════════════════════════════════════════════════════════════════════╝\n";
            std::cout << "Rank " << rank << ": Processing tasks [" << start << ", " << end << ")\n";
            std::cout << "Matrix dimension: " << N << ", Tile size: " << tile_size << "\n";
            std::cout << std::string(75, '-') << "\n\n";

            int case_counts[5] = {0}; // counters for cases 1-4
            const double zone = 1.0;
            const double mzone = -1.0;
            const double tolerance = 1e-9;

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);

            // Iterate through tasks and process ALL cases
            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADD_CT(k);
                const int ldan = BLKADD_CT(n);

                switch (myroutine) {
                    case 1: { // DPOTRF
                        ++case_counts[1];
                        std::cout << "┌─────────────────────────────────────────────────────────────────────┐\n";
                        std::cout << "│ CASE 1: DPOTRF #" << case_counts[1] << " - Tile(" << k << "," << k << ") index=" << index1 << std::string(20, ' ') << "│\n";
                        std::cout << "└─────────────────────────────────────────────────────────────────────┘\n";

                        // Dense version
                        std::cout << "  ┏━ DENSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n";
                        double *tile_dense = tiledMatrix->denseTiles[index1];
                        if (tile_dense) {
                            std::cout << "  → BEFORE:\n";
                            sTiles::Utils::print_tile_filtered("    ", tile_dense, tempkn, tempkn, ldak, 6, 6, 4);
                            sTiles::StatusCode status_dense = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_dense, ldak);
                            std::cout << "  → AFTER: " << (status_dense == sTiles::StatusCode::Success ? "✓" : "✗") << "\n";
                            sTiles::Utils::print_tile_filtered("    ", tile_dense, tempkn, tempkn, ldak, 6, 6, 4);
                        }

                        // Semisparse version
                        std::cout << "  ┏━ SEMISPARSE (banded) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n";
                        double *tile_semisparse = tiledMatrix->chunkedDenseTiles[index1];
                        if (tile_semisparse) {
                            const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                            const int kd = semi.upper_bw;
                            const int rows = tiledMatrix->tileMetaCore[index1].height;
                            const int ldab = kd + 1;
                            std::cout << "  Metadata: kd=" << kd << ", rows=" << rows << ", ldab=" << ldab << "\n";
                            std::cout << "  → BEFORE:\n";
                            sTiles::Utils::print_tile_filtered("    ", tile_semisparse, ldab, rows, ldab, std::min(ldab,6), std::min(rows,6), 4);
                            sTiles::StatusCode status_ss = sTiles::core_dpotrf_upper_banded(rows, kd, tile_semisparse);
                            std::cout << "  → AFTER: " << (status_ss == sTiles::StatusCode::Success ? "✓" : "✗") << "\n";
                            sTiles::Utils::print_tile_filtered("    ", tile_semisparse, ldab, rows, ldab, std::min(ldab,6), std::min(rows,6), 4);

                            // Validation
                            if (tile_dense) {
                                std::cout << "  ┏━ VALIDATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n";
                                int mismatches = 0;
                                for (int j = 0; j < tempkn; ++j) {
                                    for (int i = 0; i <= j && i < tempkn; ++i) {
                                        const int band = j - i;
                                        if (band > kd) continue;
                                        const int lapack_row = kd - band;
                                        const double banded_val = tile_semisparse[lapack_row + j * ldab];
                                        const double dense_val = tile_dense[i + j * ldak];
                                        if (std::fabs(dense_val - banded_val) > tolerance) {
                                            if (mismatches < 5) std::cout << "  ✗ (" << i << "," << j << "): dense=" << dense_val << " vs banded=" << banded_val << "\n";
                                            ++mismatches;
                                        }
                                    }
                                }
                                std::cout << "  " << (mismatches == 0 ? "✓ PERFECT MATCH" : "✗ FAILED: " + std::to_string(mismatches) + " mismatches") << "\n";
                            }
                        }
                        ss_cond_set(k, k, 1);
                        std::cout << std::string(75, '=') << "\n\n";
                        break;
                    }

                    case 2: { // DSYRK
                        ss_cond_wait(k, n, 1);
                        ++case_counts[2];
                        std::cout << "┌─────────────────────────────────────────────────────────────────────┐\n";
                        std::cout << "│ CASE 2: DSYRK #" << case_counts[2] << " - in=Tile(" << k << "," << n << "), out=Tile(" << k << "," << k << ")      │\n";
                        std::cout << "└─────────────────────────────────────────────────────────────────────┘\n";

                        // Dense version
                        std::cout << "  ┏━ DENSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n";
                        double *tile_in_dense = tiledMatrix->denseTiles[index1];
                        double *tile_out_dense = tiledMatrix->denseTiles[index2];
                        if (tile_in_dense && tile_out_dense) {
                            std::cout << "  → Input tile(" << k << "," << n << "):\n";
                            sTiles::Utils::print_tile_filtered("    ", tile_in_dense, tile_size, tempmn, ldan, 6, 6, 4);
                            std::cout << "  → Output tile(" << k << "," << k << ") BEFORE:\n";
                            sTiles::Utils::print_tile_filtered("    ", tile_out_dense, tempkn, tempkn, ldak, 6, 6, 4);
                            sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone, tile_in_dense, ldan, zone, tile_out_dense, ldak);
                            std::cout << "  → Output AFTER:\n";
                            sTiles::Utils::print_tile_filtered("    ", tile_out_dense, tempkn, tempkn, ldak, 6, 6, 4);
                        }

                        // Semisparse version
                        std::cout << "  ┏━ SEMISPARSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n";
                        double *tile_in_ss = tiledMatrix->chunkedDenseTiles[index1];
                        double *tile_out_ss = tiledMatrix->chunkedDenseTiles[index2];
                        if (tile_in_ss && tile_out_ss) {
                            const sTiles::SemisparseTileMetaCore& semi_in = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index2];
                            const int active_cols = semi_in.sa;
                            // Use actual tile height, not nominal tile_size (fixes buffer overrun for boundary tiles)
                            const int rows_in = tiledMatrix->tileMetaCore[index1].height;
                            std::cout << "  Input: active_cols=" << active_cols << ", rows_in=" << rows_in << "\n";
                            sTiles::Utils::print_tile_filtered("    ", tile_in_ss, rows_in, active_cols, rows_in, 6, std::min(active_cols,6), 4);

                            if (active_cols > 0) {
                                const int kd = semi_out.upper_bw;
                                const int ldab = kd + 1;
                                const int rows_out = tiledMatrix->tileMetaCore[index2].height;
                                std::cout << "  → Output BEFORE (banded, kd=" << kd << "):\n";
                                sTiles::Utils::print_tile_filtered("    ", tile_out_ss, ldab, rows_out, ldab, std::min(ldab,6), std::min(rows_out,6), 4);

                                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, active_cols, rows_in, mzone, tile_in_ss, rows_in, 0.0, tmp_tile, active_cols);

                                const int aind_size = static_cast<int>(semi_in.aind.size());
                                const int cols_out = tiledMatrix->tileMetaCore[index2].width;
                                const int diag_cols = ldab;
                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = (j < aind_size) ? semi_in.aind[j] : -1;
                                    if (col_idx < 0 || col_idx >= cols_out) continue;
                                    for (int i = 0; i <= j; ++i) {
                                        const int row_idx = (i < aind_size) ? semi_in.aind[i] : -1;
                                        if (row_idx < 0 || row_idx > col_idx) continue;
                                        const int band = col_idx - row_idx;
                                        const int lapack_row = diag_cols - 1 - band;
                                        const std::size_t offset = static_cast<std::size_t>(lapack_row) + static_cast<std::size_t>(diag_cols) * static_cast<std::size_t>(col_idx);
                                        tile_out_ss[offset] += tmp_tile[i + j * active_cols];
                                    }
                                }
                                std::cout << "  → Output AFTER:\n";
                                sTiles::Utils::print_tile_filtered("    ", tile_out_ss, ldab, rows_out, ldab, std::min(ldab,6), std::min(rows_out,6), 4);
                            }
                        }
                        std::cout << std::string(75, '=') << "\n\n";
                        break;
                    }

                    case 3: { // DTRSM
                        ss_cond_wait(k, k, 1);
                        ++case_counts[3];
                        std::cout << "┌─────────────────────────────────────────────────────────────────────┐\n";
                        std::cout << "│ CASE 3: DTRSM #" << case_counts[3] << " - diag=Tile(" << k << "," << k << "), rhs=Tile(" << m << "," << k << ")  │\n";
                        std::cout << "└─────────────────────────────────────────────────────────────────────┘\n";

                        // Dense version
                        std::cout << "  ┏━ DENSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n";
                        double *tile_rhs_dense = tiledMatrix->denseTiles[index2];
                        double *tile_out_dense_trsm = tiledMatrix->denseTiles[index1];
                        if (tile_rhs_dense && tile_out_dense_trsm) {
                            std::cout << "  → RHS BEFORE:\n";
                            sTiles::Utils::print_tile_filtered("    ", tile_out_dense_trsm, tile_size, tempmn, ldak, 6, 6, 4);
                            sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit, tile_size, tempmn, zone, tile_rhs_dense, ldak, tile_out_dense_trsm, ldak);
                            std::cout << "  → RHS AFTER:\n";
                            sTiles::Utils::print_tile_filtered("    ", tile_out_dense_trsm, tile_size, tempmn, ldak, 6, 6, 4);
                        }

                        // Semisparse version
                        std::cout << "  ┏━ SEMISPARSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n";
                        double* ss_rhs = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index1] : nullptr;
                        double* ss_diag = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index2] : nullptr;
                        if (ss_rhs && ss_diag) {
                            const sTiles::SemisparseTileMetaCore& semi_rhs = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_diag = tiledMatrix->semisparseTileMetaCore[index2];
                            const int active_cols = semi_rhs.sa;
                            const int rows_rhs = tiledMatrix->tileMetaCore[index1].height;
                            const int diag_bw = semi_diag.upper_bw;
                            std::cout << "  Metadata: active_cols=" << active_cols << ", diag_bw=" << diag_bw << "\n";
                            std::cout << "  → RHS BEFORE:\n";
                            sTiles::Utils::print_tile_filtered("    ", ss_rhs, rows_rhs, active_cols, rows_rhs, 6, std::min(active_cols,6), 4);

                            if (active_cols > 0) {
                                const int kd = diag_bw;
                                const int ldab = diag_bw + 1;
                                const int diag_dim = tiledMatrix->tileMetaCore[index2].height;
                                LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', diag_dim, kd, active_cols, ss_diag, ldab, ss_rhs, rows_rhs);
                            }
                            std::cout << "  → RHS AFTER:\n";
                            sTiles::Utils::print_tile_filtered("    ", ss_rhs, rows_rhs, active_cols, rows_rhs, 6, std::min(active_cols,6), 4);
                        }
                        ss_cond_set(m, k, 1);
                        std::cout << std::string(75, '=') << "\n\n";
                        break;
                    }

                    case 4: { // DGEMM
                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);
                        ++case_counts[4];
                        std::cout << "┌─────────────────────────────────────────────────────────────────────┐\n";
                        std::cout << "│ CASE 4: DGEMM #" << case_counts[4] << " - A=(" << k << "," << n << "), B=(" << m << "," << n << "), C=(" << m << "," << k << ")   │\n";
                        std::cout << "└─────────────────────────────────────────────────────────────────────┘\n";

                        // Dense version
                        std::cout << "  ┏━ DENSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n";
                        double *tile_a_dense = tiledMatrix->denseTiles[index1];
                        double *tile_b_dense = tiledMatrix->denseTiles[index2];
                        double *tile_out_gemm_dense = tiledMatrix->denseTiles[index3];
                        if (tile_a_dense && tile_b_dense && tile_out_gemm_dense) {
                            std::cout << "  → C BEFORE:\n";
                            sTiles::Utils::print_tile_filtered("    ", tile_out_gemm_dense, tile_size, tempmn, ldak, 5, 5, 4);
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone, tile_a_dense, ldan, tile_b_dense, ldan, zone, tile_out_gemm_dense, ldak);
                            std::cout << "  → C AFTER:\n";
                            sTiles::Utils::print_tile_filtered("    ", tile_out_gemm_dense, tile_size, tempmn, ldak, 5, 5, 4);
                        }

                        // Semisparse version
                        std::cout << "  ┏━ SEMISPARSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n";
                        double* chunk_a = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index1] : nullptr;
                        double* chunk_b = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index2] : nullptr;
                        double* chunk_out = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index3] : nullptr;
                        if (chunk_a && chunk_b && chunk_out) {
                            const sTiles::SemisparseTileMetaCore& semi_a = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_b = tiledMatrix->semisparseTileMetaCore[index2];
                            const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index3];
                            const int cols_a = semi_a.sa;
                            const int cols_b = semi_b.sa;
                            const int rows_a = tiledMatrix->tileMetaCore[index1].height;
                            const int rows_out = tiledMatrix->tileMetaCore[index3].height;
                            std::cout << "  Metadata: cols_a=" << cols_a << ", cols_b=" << cols_b << ", sa_out=" << semi_out.sa << "\n";
                            std::cout << "  → C BEFORE:\n";
                            sTiles::Utils::print_tile_filtered("    ", chunk_out, rows_out, semi_out.sa, rows_out, 5, std::min(semi_out.sa,5), 4);

                            if (cols_a > 0 && cols_b > 0 && rows_a > 0) {
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, cols_a, cols_b, rows_a, mzone, chunk_a, rows_a, chunk_b, rows_a, 0.0, tmp_tile, cols_a);

                                const int aind_a_size = static_cast<int>(semi_a.aind.size());
                                const int aind_b_size = static_cast<int>(semi_b.aind.size());
                                const auto& acol_map = semi_out.acol;
                                const int cols_out = tiledMatrix->tileMetaCore[index3].width;

                                for (int j = 0; j < cols_b && j < aind_b_size; ++j) {
                                    const int global_col = semi_b.aind[j];
                                    if (global_col < 0 || global_col >= cols_out) continue;
                                    const bool has_mapping = (global_col >= 0 && global_col < static_cast<int>(acol_map.size()));
                                    const int slot = has_mapping ? acol_map[global_col] : -1;
                                    if (slot < 0 || slot >= semi_out.sa) continue;
                                    double* out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;
                                    for (int i = 0; i < cols_a && i < aind_a_size; ++i) {
                                        const int global_row = semi_a.aind[i];
                                        if (global_row < 0 || global_row >= rows_out) continue;
                                        out_col[global_row] += tmp_tile[i + cols_a * j];
                                    }
                                }
                            }
                            std::cout << "  → C AFTER:\n";
                            sTiles::Utils::print_tile_filtered("    ", chunk_out, rows_out, semi_out.sa, rows_out, 5, std::min(semi_out.sa,5), 4);
                        }
                        std::cout << std::string(75, '=') << "\n\n";
                        break;
                    }

                    default:
                        break;
                }
                if (ss_aborted()) break;
            }

            ss_finalize();

            std::cout << "\n╔═══════════════════════════════════════════════════════════════════════╗\n";
            std::cout <<   "║  TESTING COMPLETE - SUMMARY                                           ║\n";
            std::cout <<   "╚═══════════════════════════════════════════════════════════════════════╝\n";
            std::cout << "  DPOTRF operations: " << case_counts[1] << "\n";
            std::cout << "  DSYRK operations:  " << case_counts[2] << "\n";
            std::cout << "  DTRSM operations:  " << case_counts[3] << "\n";
            std::cout << "  DGEMM operations:  " << case_counts[4] << "\n";
            std::cout << "  Total operations:  " << (case_counts[1] + case_counts[2] + case_counts[3] + case_counts[4]) << "\n\n";

            #undef BLKADD_CT
        }

        void pthreads_dpotrf_reduction_from_chol_tasks(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int rank = STILES_RANK;
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int sep = tiledMatrix->red_tree_separator_level;
            const int num_sep = ((sep * sep) - sep) / 2 + sep;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADDI_CT(k) ( (k) < full_tiles_per_dim ? tile_size : (N % tile_size) )

            sTiles::Utils::TaskDistribution *distribution = (sTiles::Utils::TaskDistribution *)malloc(num_sep * sizeof(sTiles::Utils::TaskDistribution));
            for (int ind = 0; ind < num_sep; ind++) {
                if (tiledMatrix->trees[ind]) {
                    distribution[ind] = sTiles::Utils::calculateTaskDistribution(rank, STILES_SIZE, tiledMatrix->trees[ind]->num_tasks);
                }
            }

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end   = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());
            sTiles::Logger::info("│   [ESMAIL_CHECK] dpotrf_reduction_from_chol_tasks: rank=" + std::to_string(rank) +
                                ", tasks=[" + std::to_string(start) + "," + std::to_string(end) + ")");

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADDI_CT(k);
                const int ldan = BLKADDI_CT(n);

                const double zone  = 1.0;
                const double mzone = -1.0;

                switch (myroutine) {
                    case 1: // DPOTRF
                    {
                        double* tile_out = tiledMatrix->denseTiles[index1];
                        if (!tile_out) {
                            sTiles::Logger::warning("│   [ESMAIL_CHECK] null tile_out in potrf (index1=", index1, ")");
                            break;
                        }
                        sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
                        if (status != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (tile k=%d).\n", k);
                            ss_abort(); break;
                        }
                        ss_cond_set(k, k, 1);
                        break;
                    }
                    case 2: // DSYRK
                    {
                        ss_cond_wait(k, n, 1);
                        double* tile_in  = tiledMatrix->denseTiles[index1];
                        double* tile_out = tiledMatrix->denseTiles[index2];
                        if (!tile_out) {
                            sTiles::Logger::warning("│   [ESMAIL_CHECK] null tile_out in dsyrk (index2=", index2, ")");
                            break;
                        }
                        sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone,
                                        tile_in, ldan, zone, tile_out, ldak);
                        break;
                    }
                    case 3: // DTRSM
                    {
                        ss_cond_wait(k, k, 1);
                        double* tile_rhs = tiledMatrix->denseTiles[index2];
                        double* tile_out = tiledMatrix->denseTiles[index1];
                        if (!tile_out) {
                            sTiles::Logger::warning("│   [ESMAIL_CHECK] null tile_out in dtrsm (index1=", index1, ")");
                            break;
                        }
                        sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit,
                                        tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
                        ss_cond_set(m, k, 1);
                        break;
                    }
                    case 4: // DGEMM
                    {
                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);
                        double* tile_a   = tiledMatrix->denseTiles[index1];
                        double* tile_b   = tiledMatrix->denseTiles[index2];
                        double* tile_out = tiledMatrix->denseTiles[index3];
                        if (!tile_out) {
                            sTiles::Logger::warning("│   [ESMAIL_CHECK] null tile_out in dgemm (index3=", index3, ")");
                            break;
                        }
                        sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone,
                                        tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
                        break;
                    }
                    case 5: // DSYRK (reduction tree)
                    {
                        ss_cond_wait(k, n, 1);
                        double* tile_out = tiledMatrix->trees[index2]->nodes[STILES_RANK].x;
                        double* tile_in  = tiledMatrix->denseTiles[index1];
                        sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, zone,
                                        tile_in, ldan, zone, tile_out, ldak);
                        break;
                    }
                    case 6: // Set dependency flag
                    {
                        tiledMatrix->trees[index2]->dependency[STILES_RANK] = index3;
                        break;
                    }
                    case 7: // DGEADD (reduction tree accumulate across ranks into tile index2)
                    {
                        const sTiles::TileMetaCore* meta_out_ptr = tiledMatrix->tileMetaCore ? &tiledMatrix->tileMetaCore[index2] : nullptr;
                        const int tile_height = (meta_out_ptr && meta_out_ptr->height > 0) ? meta_out_ptr->height : tile_size;
                        const int tile_width  = (meta_out_ptr && meta_out_ptr->width  > 0) ? meta_out_ptr->width  : tile_size;
                        double* tile_out = tiledMatrix->denseTiles[index2];
                        for (int rank_idx = 0; rank_idx < STILES_SIZE; rank_idx++) {
                            if (rank_idx > 0) ss_cond_wait_tree_e_s_t_y_l_e(rank_idx, index3, tiledMatrix->trees[index1]->dependency);
                            double* tree_tile = tiledMatrix->trees[index1]->nodes[rank_idx].x;
                            sTiles::core_dgeadd(sTiles::Op::NoTrans, tile_height, tile_width, mzone,
                                                tree_tile, tile_height, zone, tile_out, tile_height);
                            tiledMatrix->trees[index1]->dependency[rank_idx] = 0;
                        }
                        break;
                    }
                    case 8: // DGEMM (reduction tree)
                    {
                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);
                        double* tile_out = tiledMatrix->trees[index3]->nodes[STILES_RANK].x;
                        double* tile_a   = tiledMatrix->denseTiles[index1];
                        double* tile_b   = tiledMatrix->denseTiles[index2];
                        sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, zone,
                                        tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
                        break;
                    }
                    case 9: // Set dependency flag (fixed token)
                    {
                        tiledMatrix->trees[index2]->dependency[STILES_RANK] = 165715;
                        break;
                    }
                    case 10: // DGEADD (reduction tree from index1 into tile index3)
                    {
                        const sTiles::TileMetaCore* meta_out_ptr2 = tiledMatrix->tileMetaCore ? &tiledMatrix->tileMetaCore[index3] : nullptr;
                        const int tile_height2 = (meta_out_ptr2 && meta_out_ptr2->height > 0) ? meta_out_ptr2->height : tile_size;
                        const int tile_width2  = (meta_out_ptr2 && meta_out_ptr2->width  > 0) ? meta_out_ptr2->width  : tile_size;
                        double* tile_out2 = tiledMatrix->denseTiles[index3];
                        for (int rank_idx = 0; rank_idx < STILES_SIZE; rank_idx++) {
                            if (rank_idx > 0) ss_cond_wait_tree_e_s_t_y_l_e(rank_idx, 165715, tiledMatrix->trees[index1]->dependency);
                            double* tree_tile = tiledMatrix->trees[index1]->nodes[rank_idx].x;
                            sTiles::core_dgeadd(sTiles::Op::NoTrans, tile_height2, tile_width2, mzone,
                                                tree_tile, tile_height2, zone, tile_out2, tile_height2);
                            tiledMatrix->trees[index1]->dependency[rank_idx] = 0;
                        }
                        break;
                    }
                    default:
                        sTiles::Logger::warning("│   [ESMAIL_CHECK] Unexpected routine id ", myroutine, " in reduction tasks");
                        break;
                }
                if (ss_aborted()) break;
            }
            ss_finalize();
            free(distribution);
            #undef BLKADDI_CT
        }

        void pthreads_dpotrf_expansion_from_chol_tasks_dense_serial_baseline(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);

            const int ntasks = static_cast<int>(tasks.size());
            for (int idx = 0; idx < ntasks; ++idx) {

                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADD_CT(k);
                const int ldan = BLKADD_CT(n);

                const double zone = 1.0;
                const double mzone = -1.0;

                switch (myroutine) {
                    case 1: { // DPOTRF
                        double *tile_out = tiledMatrix->denseTiles[index1];
                        sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
                        if (status != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (tile k=%d).\n", k);
                            stile->ss_abort = 1;
                            return;
                        }
                        break;
                    }
                    case 2: { // DSYRK
                        double *tile_in = tiledMatrix->denseTiles[index1];
                        double *tile_out = tiledMatrix->denseTiles[index2];
                        if (tile_in && tile_out) sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone, tile_in, ldan, zone, tile_out, ldak);
                        break;
                    }
                    case 3: { // DTRSM
                        double *tile_rhs = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index1];
                        if (tile_rhs && tile_out) sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit, tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
                        break;
                    }
                    case 4: { // DGEMM
                        double *tile_a = tiledMatrix->denseTiles[index1];
                        double *tile_b = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index3];
                        if (tile_out && tile_a && tile_b) sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone, tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
                        break;
                    }
                    default:
                        break;
                }
            }

            #ifdef ESMAIL_DEBUG
            {
                const int nnz = tiledMatrix->original_nnz;
                const char* export_path = sTiles_get_dense_export_file();
                if (export_path && export_path[0] != '\0' && nnz > 0) {
                    std::vector<double> y_dense(static_cast<std::size_t>(nnz));
                    sTiles::preprocess::export_x_dense_tiles_serial(tiledMatrix, y_dense.data());
                    sTiles::Logger::info("[dense-export] Writing ", nnz, " values to ", export_path);
                    const auto export_begin = std::chrono::steady_clock::now();
                    if (!write_dense_values_to_txt(export_path, y_dense)) {
                        sTiles::Logger::warning("[dense-export] Failed to write nnz vector to ", export_path);
                    } else {
                        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - export_begin).count();
                        sTiles::Logger::info("[dense-export] Completed in ", secs, " s");
                    }
                }
            }
            #endif

            #undef BLKADD_CT
        }

        void pthreads_dpotrf_expansion_from_chol_tasks_dense_serial(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);

            const int ntasks = static_cast<int>(tasks.size());
            for (int idx = 0; idx < ntasks; ++idx) {

                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADD_CT(k);
                const int ldan = BLKADD_CT(n);

                const double zone = 1.0;
                const double mzone = -1.0;

                switch (myroutine) {
                    case 1: { // DPOTRF
                        double *tile_out = tiledMatrix->denseTiles[index1];
                        sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
                        if (status != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (tile k=%d).\n", k);
                            stile->ss_abort = 1;
                            return;
                        }
                        break;
                    }
                    case 2: { // DSYRK
                        double *tile_in = tiledMatrix->denseTiles[index1];
                        double *tile_out = tiledMatrix->denseTiles[index2];
                        if (tile_in && tile_out) sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone, tile_in, ldan, zone, tile_out, ldak);
                        break;
                    }
                    case 3: { // DTRSM
                        double *tile_rhs = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index1];
                        if (tile_rhs && tile_out) sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit, tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
                        break;
                    }
                    case 4: { // DGEMM
                        double *tile_a = tiledMatrix->denseTiles[index1];
                        double *tile_b = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index3];
                        if (tile_out && tile_a && tile_b) sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone, tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
                        break;
                    }
                    default:
                        break;
                }
            }

            #undef BLKADD_CT
        }

        void pthreads_dpotrf_expansion_from_chol_tasks_dense(TiledMatrix *tiledMatrix, stiles_context_t *stile) {
            const int rank = STILES_RANK;

            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

            //export_dense_tiled_matrix(tiledMatrix, "fillin/matrix_before_factorization.txt", rank, false, true);

            // constexpr int kMaxTasksToPrint = 10;
            // int printed_tasks = 0;
            
            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            for (int idx = start; idx < end; ++idx) {

                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADD_CT(k);
                const int ldan = BLKADD_CT(n);

                const double zone = 1.0;
                const double mzone = -1.0;

                // if (printed_tasks < kMaxTasksToPrint) {
                //     print_tasks(rank, myroutine, m, k, n, index1, index2, index3);
                //     ++printed_tasks;
                // }

                switch (myroutine) {
                    case 1: { // DPOTRF
                        double *tile_out = tiledMatrix->denseTiles[index1];
                        sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
                        if (status != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (tile k=%d).\n", k);
                            ss_abort(); break;
                        }
                        ss_cond_set(k, k, 1);
                        break;
                    }
                    case 2: { // DSYRK
                        ss_cond_wait(k, n, 1);
                        double *tile_in = tiledMatrix->denseTiles[index1];
                        double *tile_out = tiledMatrix->denseTiles[index2];
                        if (tile_in && tile_out) sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone, tile_in, ldan, zone, tile_out, ldak);
                        break;
                    }
                    case 3: { // DTRSM
                        ss_cond_wait(k, k, 1);
                        double *tile_rhs = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index1];
                        if (tile_rhs && tile_out) sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit, tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
                        ss_cond_set(m, k, 1);
                        break;
                    }
                    case 4: { // DGEMM

                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);
                        double *tile_a = tiledMatrix->denseTiles[index1];
                        double *tile_b = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index3];
                        if (tile_out && tile_a && tile_b) sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone, tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
                        break;
                    }
                    default:
                        break;
                }
                if (ss_aborted()) break;
            }
            ss_finalize();
            //export_dense_tiled_matrix(tiledMatrix, "fillin/matrix_after_factorization.txt", rank, false, true);
            
            
            //use this to compare ND (with and without padding), with parallism of partitions of ND, with same ordering strategy but different tile size, without tile layer of ordering..
            
            #ifdef ESMAIL_DEBUG
            if (rank == 0) {
                const int nnz = tiledMatrix->original_nnz;
                const char* export_path = sTiles_get_dense_export_file();
                if (export_path && export_path[0] != '\0' && nnz > 0) {
                    std::vector<double> y_dense(static_cast<std::size_t>(nnz));
                    sTiles::preprocess::export_x_dense_tiles_serial(tiledMatrix, y_dense.data());
                    sTiles::Logger::info("[dense-export] Writing ", nnz, " values to ", export_path);
                    const auto export_begin = std::chrono::steady_clock::now();
                    if (!write_dense_values_to_txt(export_path, y_dense)) {
                        sTiles::Logger::warning("[dense-export] Failed to write nnz vector to ", export_path);
                    } else {
                        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - export_begin).count();
                        sTiles::Logger::info("[dense-export] Completed in ", secs, " s");
                    }
                }
            }
            #endif

            #undef BLKADD_CT


        }
        
        void pthreads_dpotrf_expansion_from_chol_tasks_dense_prints(TiledMatrix *tiledMatrix, stiles_context_t *stile) {
            const int rank = STILES_RANK;

            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end   = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1]
                                                                            : static_cast<int>(tasks.size());

            // ── PRE-SCAN: count how many GEMMs/SYRKs/TRSMs share the same panel n ──────
            // This tells us the *theoretical* batch size if we could collect them all.
            // Key: n = t[3] is the "panel" that gates the dependency (ss_cond_wait uses n).
            std::unordered_map<int, int> gemm_per_panel;   // panel n → GEMM count
            std::unordered_map<int, int> syrk_per_panel;   // panel n → SYRK count
            std::unordered_map<int, int> trsm_per_panel;   // panel k → TRSM count

            for (int i = start; i < end; ++i) {
                const int routine = tasks[i][0];
                const int n_val   = tasks[i][3];
                const int k_val   = tasks[i][2];
                if      (routine == 4) gemm_per_panel[n_val]++;
                else if (routine == 2) syrk_per_panel[n_val]++;
                else if (routine == 3) trsm_per_panel[k_val]++;
            }

            // Print pre-scan summary once (rank 0 only to avoid interleaving)
            if (rank == 0) {
                printf("\n══ BATCH-POTENTIAL ANALYSIS (rank 0 slice) ══\n");
                printf("   tile_size=%d  num_tiles=%d  total_tasks=[%d,%d)\n\n",
                    tile_size, num_tiles_per_dim, start, end);

                printf("   %-10s  %-8s  %-10s  %-10s  %-10s\n",
                    "panel_n", "DTRSMs", "DGEMMs", "DSYRKs", "batch_ok?");
                printf("   %-10s  %-8s  %-10s  %-10s  %-10s\n",
                    "-------", "------", "------", "------", "---------");

                // Collect all panels seen
                std::set<int> all_panels;
                for (auto &[p,_] : gemm_per_panel) all_panels.insert(p);
                for (auto &[p,_] : syrk_per_panel) all_panels.insert(p);
                for (auto &[p,_] : trsm_per_panel) all_panels.insert(p);

                for (int p : all_panels) {
                    int ng = gemm_per_panel.count(p) ? gemm_per_panel[p] : 0;
                    int ns = syrk_per_panel.count(p) ? syrk_per_panel[p] : 0;
                    int nt = trsm_per_panel.count(p) ? trsm_per_panel[p] : 0;
                    // "batch_ok" = more than 1 GEMM shares same panel → worth batching
                    const char* verdict = (ng > 1) ? "YES" : (ng == 1) ? "single" : "none";
                    printf("   %-10d  %-8d  %-10d  %-10d  %-10s\n", p, nt, ng, ns, verdict);
                }
                printf("\n");
                fflush(stdout);
            }

            // ── RUNTIME PRINTS: track when each task executes and how long it waits ─────
            using Clock = std::chrono::steady_clock;
            const auto t0 = Clock::now();   // reference start for all timestamps

            auto elapsed_us = [&]() -> long {
                return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
            };

            // Per-panel GEMM sequence counter: lets us see if GEMMs at same panel
            // are serialized (count goes 1,2,3... far apart in time) or clustered.
            std::unordered_map<int, int> gemm_seq;

            ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);

            for (int idx = start; idx < end; ++idx) {

                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADD_CT(k);
                const int ldan = BLKADD_CT(n);

                const double zone  =  1.0;
                const double mzone = -1.0;

                switch (myroutine) {

                    case 1: { // DPOTRF — diagonal, serial bottleneck
                        long t_start = elapsed_us();
                        double *tile_out = tiledMatrix->denseTiles[index1];
                        sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
                        long t_end = elapsed_us();

                        printf("[POTRF ] rank=%d  k=%d  dim=%d  time=%ldus\n",
                            rank, k, tempkn, t_end - t_start);
                        fflush(stdout);

                        if (status != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (tile k=%d).\n", k);
                            ss_abort(); break;
                        }
                        ss_cond_set(k, k, 1);
                        break;
                    }

                    case 2: { // DSYRK
                        long wait_start = elapsed_us();
                        ss_cond_wait(k, n, 1);
                        long wait_end = elapsed_us();

                        double *tile_in  = tiledMatrix->denseTiles[index1];
                        double *tile_out = tiledMatrix->denseTiles[index2];

                        long t_start = elapsed_us();
                        if (tile_in && tile_out)
                            sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans,
                                            tempkn, tile_size, mzone, tile_in, ldan, zone, tile_out, ldak);
                        long t_end = elapsed_us();

                        printf("[DSYRK ] rank=%d  panel_n=%d  k=%d  waited=%ldus  compute=%ldus\n",
                            rank, n, k, wait_end - wait_start, t_end - t_start);
                        fflush(stdout);
                        break;
                    }

                    case 3: { // DTRSM
                        long wait_start = elapsed_us();
                        ss_cond_wait(k, k, 1);
                        long wait_end = elapsed_us();

                        double *tile_rhs = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index1];

                        long t_start = elapsed_us();
                        if (tile_rhs && tile_out)
                            sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans,
                                            sTiles::Diag::NonUnit,
                                            tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
                        long t_end = elapsed_us();

                        printf("[DTRSM ] rank=%d  panel_k=%d  m=%d  waited=%ldus  compute=%ldus\n",
                            rank, k, m, wait_end - wait_start, t_end - t_start);
                        fflush(stdout);

                        ss_cond_set(m, k, 1);
                        break;
                    }

                    case 4: { // DGEMM — the main batching target
                        long wait_start = elapsed_us();
                        ss_cond_wait(k, n, 1);
                        ss_cond_wait(m, n, 1);
                        long wait_end = elapsed_us();

                        double *tile_a   = tiledMatrix->denseTiles[index1];
                        double *tile_b   = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index3];

                        long t_start = elapsed_us();
                        if (tile_out && tile_a && tile_b)
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                            tile_size, tempmn, tile_size,
                                            mzone, tile_a, ldan, tile_b, ldan,
                                            zone,  tile_out, ldak);
                        long t_end = elapsed_us();

                        int seq = ++gemm_seq[n];
                        int total_at_panel = gemm_per_panel.count(n) ? gemm_per_panel[n] : 1;

                        // KEY PRINT: seq/total tells you if GEMMs at same panel are serialized.
                        // If compute times overlap across ranks → batching would help.
                        // If waited >> compute → cores are starved waiting, not compute-bound.
                        printf("[DGEMM ] rank=%d  panel_n=%d  update=(%d,%d)  seq=%d/%d"
                            "  waited=%ldus  compute=%ldus  M=%d K=%d N=%d\n",
                            rank, n, k, m, seq, total_at_panel,
                            wait_end - wait_start, t_end - t_start,
                            tile_size, tile_size, tempmn);
                        fflush(stdout);
                        break;
                    }

                    default:
                        break;
                }
                if (ss_aborted()) break;
            }

            ss_finalize();

            // ── POST-RUN SUMMARY ─────────────────────────────────────────────────────────
            if (rank == 0) {
                printf("\n══ BATCHING VERDICT ══\n");
                printf("   Look at [DGEMM] lines above and ask:\n");
                printf("   (1) Are waited= times large vs compute= ?  → cores are starved\n");
                printf("   (2) Do multiple ranks show same panel_n at similar timestamps?\n");
                printf("       → those could be a batch (they're already running in parallel)\n");
                printf("   (3) Is seq=1/1 everywhere?  → no batching opportunity per rank\n");
                printf("   (4) Is seq=N/N with N>4?    → worth restructuring into batch\n");
                printf("\n");
                fflush(stdout);
            }

            #undef BLKADD_CT
        }
        
        void pthreads_pdpotrf(stiles_context_t *stile) {

            static int* stiles_control_params = sTiles_get_params();

            TiledMatrix* tiledMatrix = nullptr;
            sTiles::unpack_args(stile, tiledMatrix);

            if (!tiledMatrix) {
                std::cout << "Error: null tiledMatrix in stiles_pdpotrf" << std::endl;
                return;
            }

            // Reset chol_info before factorization (rank 0 only to avoid races)
            if (STILES_RANK == 0) {
                tiledMatrix->chol_info.store(0, std::memory_order_relaxed);
            }

            if (tiledMatrix->use_boosted_e_trick) {

                if (tiledMatrix->red_tree_separator_level > 0) {
                    sTiles::Process::pthreads_dpotrf_reduction_from_chol_tasks(tiledMatrix, stile);
                } else {
                    // params[3]  = tile_type_mode (0 dense, 1 semisparse, 2 nonunif)
                    // params[14] = parallel knob (0 → serial path when STILES_SIZE==1;
                    //              any other value → multi-core path).
                    const int  tile_type_mode = stiles_control_params[3];
                    const bool serial_path    = (STILES_SIZE == 1 && stiles_control_params[14] == 0);
                    const bool has_semisparse = tiledMatrix->chunkedDenseTiles && tiledMatrix->semisparseTileMetaCore;

                    if (tile_type_mode == 0) {
                        // Dense mode
                        if (serial_path) {
                            sTiles::Process::pthreads_dpotrf_expansion_from_chol_tasks_dense_serial(tiledMatrix, stile);
                        } else {
                            sTiles::Process::pthreads_dpotrf_expansion_from_chol_tasks_dense(tiledMatrix, stile);
                        }

                    } else if (tile_type_mode == 1) {
                        // Semisparse mode
                        if (has_semisparse) {
                            if (serial_path) {
                                sTiles::Process::dpotrf_expansion_from_chol_tasks_semisparse_imp4_serial(tiledMatrix, stile);
                            } else {
                                sTiles::Process::dpotrf_expansion_from_chol_tasks_semisparse_imp4(tiledMatrix, stile);
                            }
                        } else {
                            // Fallback to dense mode (e.g., for single-tile matrices)
                            sTiles::Process::pthreads_dpotrf_expansion_from_chol_tasks_dense(tiledMatrix, stile);
                        }

                    } else if (tile_type_mode == 2) {
                        // Nonunif mode — semisparse storage with no-pruning policy.
                        if (has_semisparse) {
                            if (tiledMatrix->tile_size <= 2) {
                                sTiles::Process::dpotrf_expansion_from_chol_tasks_semisparse_numeric(tiledMatrix, stile);
                            } else {
                                sTiles::Process::dpotrf_expansion_from_chol_tasks_semisparse(tiledMatrix, stile);
                            }
                        } else {
                            // Fallback to dense mode (e.g., for single-tile matrices)
                            sTiles::Process::pthreads_dpotrf_expansion_from_chol_tasks_dense(tiledMatrix, stile);
                        }
                    }
                }
            } else {
                // Dense variants 1 and 2 without pre-collected tasks.
                // Variant 1: single tile (numActiveTiles == 1, dimTiledMatrix == 1)
                // Variant 2: scaled dense (numActiveTiles == triangular_size)
                const int num_active = tiledMatrix->numActiveTiles;
                const int dim_tiled  = tiledMatrix->dimTiledMatrix;
                const int tri_size   = tiledMatrix->triangular_size;

                if (num_active == 1 && dim_tiled == 1) {
                    sTiles::Process::dpotrf_dense_variant_direct(tiledMatrix, stile, 1);
                } else if (num_active == tri_size && tri_size > 0) {
                    sTiles::Process::dpotrf_dense_variant_direct(tiledMatrix, stile, 2);
                }
            }

            // Record Cholesky status: 0 = success, 1 = not positive definite
            if (STILES_RANK == 0) {
                tiledMatrix->chol_info.store(stile->ss_abort ? 1 : 0, std::memory_order_release);
            }
        }

        // OMP worker functions - use dep_* macros instead of ss_* macros
        void dpotrf_expansion_from_chol_tasks_dense_omp(TiledMatrix* tiledMatrix, omp_dep_tracker_t* dep_tracker, int worldsize) {
            const int rank = omp_get_thread_num();
            (void)worldsize; // Used implicitly - offsets array was computed for worldsize threads

            #ifdef STILES_DEBUG_OMP
            #pragma omp single
            std::cerr << "[OMP_DEBUG] dense_omp: entering, worldsize=" << worldsize << std::endl;
            #endif

            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT_OMP(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            #ifdef STILES_DEBUG_OMP
            #pragma omp single
            std::cerr << "[OMP_DEBUG] dense_omp: N=" << N << ", tile_size=" << tile_size
                      << ", num_tiles_per_dim=" << num_tiles_per_dim << std::endl;
            #endif

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);

            #ifdef STILES_DEBUG_OMP
            #pragma omp single
            std::cerr << "[OMP_DEBUG] dense_omp: tasks.size()=" << tasks.size()
                      << ", offsets.size()=" << offsets.size() << std::endl;
            #endif

            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

            #ifdef STILES_DEBUG_OMP
            #pragma omp single
            std::cerr << "[OMP_DEBUG] dense_omp: about to call dep_init(" << num_tiles_per_dim << ", " << num_tiles_per_dim << ", 0)" << std::endl;
            #endif

            dep_init(num_tiles_per_dim, num_tiles_per_dim, 0);

            #ifdef STILES_DEBUG_OMP
            #pragma omp single
            std::cerr << "[OMP_DEBUG] dense_omp: dep_init complete, progress_table=" << (void*)dep_tracker->progress_table
                      << ", ld=" << dep_tracker->ld << ", abort_flag=" << dep_tracker->abort_flag.load() << std::endl;

            #pragma omp single
            std::cerr << "[OMP_DEBUG] dense_omp: starting task loop" << std::endl;
            #endif

            for (int idx = start; idx < end; ++idx) {

                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADD_CT_OMP(k);
                const int ldan = BLKADD_CT_OMP(n);

                const double zone = 1.0;
                const double mzone = -1.0;

                switch (myroutine) {
                    case 1: { // DPOTRF
                        double *tile_out = tiledMatrix->denseTiles[index1];
                        sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
                        if (status != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (tile k=%d).\n", k);
                            dep_abort_all(); break;
                        }
                        dep_set_done(k, k, 1);
                        break;
                    }
                    case 2: { // DSYRK
                        dep_wait_for(k, n, 1);
                        double *tile_in = tiledMatrix->denseTiles[index1];
                        double *tile_out = tiledMatrix->denseTiles[index2];
                        if (tile_in && tile_out) sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone, tile_in, ldan, zone, tile_out, ldak);
                        break;
                    }
                    case 3: { // DTRSM
                        dep_wait_for(k, k, 1);
                        double *tile_rhs = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index1];
                        if (tile_rhs && tile_out) sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit, tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
                        dep_set_done(m, k, 1);
                        break;
                    }
                    case 4: { // DGEMM
                        dep_wait_for(k, n, 1);
                        dep_wait_for(m, n, 1);
                        double *tile_a = tiledMatrix->denseTiles[index1];
                        double *tile_b = tiledMatrix->denseTiles[index2];
                        double *tile_out = tiledMatrix->denseTiles[index3];
                        if (tile_out && tile_a && tile_b) sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone, tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
                        break;
                    }
                    default:
                        break;
                }
                if (dep_is_aborted()) break;
            }

            #ifdef STILES_DEBUG_OMP
            #pragma omp single
            std::cerr << "[OMP_DEBUG] dense_omp: task loop complete, calling dep_finalize" << std::endl;
            #endif

            dep_finalize();

            #ifdef STILES_DEBUG_OMP
            #pragma omp single
            std::cerr << "[OMP_DEBUG] dense_omp: dep_finalize complete, exiting" << std::endl;
            #endif

            #undef BLKADD_CT_OMP
        }

        void dpotrf_expansion_from_chol_tasks_semisparse_omp(TiledMatrix* tiledMatrix, omp_dep_tracker_t* dep_tracker, int worldsize) {

            const int rank = omp_get_thread_num();
            const int actual_threads = omp_get_num_threads();  // How many threads are actually in this region
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADD_CT_OMP(k) ((k) < full_tiles_per_dim ? tile_size : (N % tile_size))

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());
            const bool *diag_map = tiledMatrix->diagonal_bmapper;

            #ifdef STILES_DEBUG_OMP
            #pragma omp critical
            {
                std::cerr << "[OMP_DEBUG] semisparse_omp: rank=" << rank
                          << ", actual_threads=" << actual_threads
                          << ", worldsize=" << worldsize
                          << ", num_workspaces=" << tiledMatrix->num_workspaces
                          << ", offsets.size()=" << offsets.size()
                          << ", tasks.size()=" << tasks.size()
                          << ", start=" << start << ", end=" << end
                          << ", num_tasks=" << (end - start) << std::endl;
            }
            #endif

            // Verify thread count matches expected - warn if mismatch
            if (rank == 0 && actual_threads != worldsize) {
                std::cerr << "[sTiles WARNING] OMP thread count mismatch: expected=" << worldsize
                          << " actual=" << actual_threads
                          << ". This may indicate nested OMP issues." << std::endl;
            }

            // Safety check: ensure workspace is available for this thread
            if (rank >= tiledMatrix->num_workspaces || !tiledMatrix->workspaces || !tiledMatrix->workspaces[rank]) {
                #pragma omp critical
                std::cerr << "[sTiles ERROR] semisparse_omp: no workspace for rank " << rank
                          << " (num_workspaces=" << tiledMatrix->num_workspaces << ")" << std::endl;
                return;
            }

            double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();

            dep_init(num_tiles_per_dim, num_tiles_per_dim, 0);
            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADD_CT_OMP(k);
                const int ldan = BLKADD_CT_OMP(n);

                const double zone = 1.0;
                const double mzone = -1.0;

                switch (myroutine) {
                    case 1: { // DPOTRF
                        double *ssstile_out = tiledMatrix->chunkedDenseTiles[index1];

                        if (ssstile_out) {
                            const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                            const int kd = semi.upper_bw;
                            const int rows = meta.height;

                            sTiles::StatusCode ssstatus = sTiles::core_dpotrf_upper_banded(rows, kd, ssstile_out);
                            if (ssstatus != sTiles::StatusCode::Success) {
                                std::fprintf(stderr, "sTiles error: matrix is not positive definite (banded tile k=%d).\n", k);
                                dep_abort_all(); break;
                            }
                        }
                        dep_set_done(k, k, 1);
                        break;
                    }
                    case 2: { // DSYRK
                        dep_wait_for(k, n, 1);

                        double *ssstile_in = tiledMatrix->chunkedDenseTiles[index1];
                        double *ssstile_out = tiledMatrix->chunkedDenseTiles[index2];

                        if (ssstile_in && ssstile_out) {
                            const sTiles::SemisparseTileMetaCore& semi_in = tiledMatrix->semisparseTileMetaCore[index1];
                            const int active_cols = semi_in.sa;

                            if (active_cols > 0) {
                                const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index2];
                                const sTiles::TileMetaCore& meta_in = tiledMatrix->tileMetaCore[index1];
                                const sTiles::TileMetaCore& meta_out = tiledMatrix->tileMetaCore[index2];
                                const bool lapack_diag_tile = diag_map && diag_map[index2];
                                // Use actual tile height, not nominal tile_size (fixes buffer overrun for boundary tiles)
                                const int rows_in = (meta_in.height > 0) ? meta_in.height : tile_size;

                                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, active_cols, rows_in, mzone, ssstile_in, rows_in, 0.0, tmp_tile, active_cols);

                                const int aind_size = static_cast<int>(semi_in.aind.size());
                                const int cols_out = (tiledMatrix->tileMetaCore[index2].width > 0) ? tiledMatrix->tileMetaCore[index2].width : tile_size;
                                const int rows_out = meta_out.height;
                                const int ld_tmp = active_cols;
                                const int diag_cols = (semi_out.upper_bw >= 0) ? (semi_out.upper_bw + 1) : cols_out;

                                for (int j = 0; j < active_cols; ++j) {
                                    const int col_idx = (j < aind_size) ? semi_in.aind[j] : -1;
                                    if (col_idx < 0 || col_idx >= cols_out) continue;
                                    for (int i = 0; i <= j; ++i) {
                                        const int row_idx = (i < aind_size) ? semi_in.aind[i] : -1;
                                        if (row_idx < 0 || row_idx > col_idx) continue;
                                        const int band = col_idx - row_idx;
                                        if (band >= diag_cols) {
                                            const double dropped = tmp_tile[i + j * ld_tmp];
                                            if (std::fabs(dropped) > 1e-15) {
                                                std::fprintf(stderr, "sTiles warning: DSYRK bandwidth overflow in tile k=%d: band=%d >= diag_cols=%d, dropped=%.6e\n",
                                                             k, band, diag_cols, dropped);
                                            }
                                            continue;
                                        }
                                        const int lapack_row = diag_cols - 1 - band;
                                        const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                                                + static_cast<std::size_t>(diag_cols)
                                                                * static_cast<std::size_t>(col_idx);
                                        ssstile_out[offset] += tmp_tile[i + j * ld_tmp];
                                    }
                                }
                            }
                        }
                        break;
                    }
                    case 3: { // DTRSM
                        dep_wait_for(k, k, 1);

                        double* ss_rhs  = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index1] : nullptr;
                        double* ss_diag = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index2] : nullptr;
                        if (ss_rhs && ss_diag) {
                            const sTiles::SemisparseTileMetaCore& semi_rhs = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_diag = tiledMatrix->semisparseTileMetaCore[index2];
                            const sTiles::TileMetaCore& meta_rhs = tiledMatrix->tileMetaCore[index1];
                            const sTiles::TileMetaCore& meta_diag = tiledMatrix->tileMetaCore[index2];
                            const int active_cols = semi_rhs.sa;
                            const int rows_rhs = (meta_rhs.height > 0) ? meta_rhs.height : tile_size;
                            const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tile_size;
                            const int diag_bw = (semi_diag.upper_bw >= 0) ? semi_diag.upper_bw : 0;

                            if (active_cols > 0) {
                                const int kd = diag_bw;
                                const int ldab = diag_bw + 1;
                                LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', diag_dim, kd, active_cols, ss_diag, ldab, ss_rhs, rows_rhs);
                            }
                        }
                        dep_set_done(m, k, 1);
                        break;
                    }
                    case 4: { // DGEMM
                        dep_wait_for(k, n, 1);
                        dep_wait_for(m, n, 1);

                        double* chunk_a   = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index1] : nullptr;
                        double* chunk_b   = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index2] : nullptr;
                        double* chunk_out = tiledMatrix->chunkedDenseTiles ? tiledMatrix->chunkedDenseTiles[index3] : nullptr;

                        if (chunk_a && chunk_b && chunk_out) {
                            const sTiles::SemisparseTileMetaCore& semi_a = tiledMatrix->semisparseTileMetaCore[index1];
                            const sTiles::SemisparseTileMetaCore& semi_b = tiledMatrix->semisparseTileMetaCore[index2];
                            const sTiles::SemisparseTileMetaCore& semi_out = tiledMatrix->semisparseTileMetaCore[index3];
                            const sTiles::TileMetaCore& meta_out = tiledMatrix->tileMetaCore[index3];
                            const bool lapack_diag_tile = diag_map && diag_map[index3];

                            const int cols_a = semi_a.sa;
                            const int cols_b = semi_b.sa;
                            const int rows_a = tiledMatrix->tileMetaCore[index1].height;
                            const int rows_b = tiledMatrix->tileMetaCore[index2].height;
                            const int rows_out = meta_out.height;
                            if (cols_a > 0 && cols_b > 0 && rows_a > 0) {
                                const int ld_tmp = cols_a;
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                cols_a, cols_b, rows_a,
                                                mzone, chunk_a, rows_a,
                                                chunk_b, rows_a,
                                                0.0, tmp_tile, ld_tmp);

                                const int aind_a_size = static_cast<int>(semi_a.aind.size());
                                const int aind_b_size = static_cast<int>(semi_b.aind.size());
                                const auto& acol_map = semi_out.acol;
                                const int cols_out = meta_out.width;

                                for (int j = 0; j < cols_b && j < aind_b_size; ++j) {
                                    const int global_col = semi_b.aind[j];
                                    if (global_col < 0 || global_col >= cols_out) continue;

                                    const bool has_mapping = (global_col >= 0 && global_col < static_cast<int>(acol_map.size()));
                                    const int slot = has_mapping ? acol_map[global_col] : -1;
                                    if (slot < 0 || slot >= semi_out.sa) continue;

                                    double* out_col = chunk_out + static_cast<std::size_t>(slot) * rows_out;

                                    for (int i = 0; i < cols_a && i < aind_a_size; ++i) {
                                        const int global_row = semi_a.aind[i];
                                        if (global_row < 0 || global_row >= rows_out) continue;
                                        out_col[global_row] += tmp_tile[i + ld_tmp * j];
                                    }
                                }
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }
                if (dep_is_aborted()) break;
            }

            dep_finalize();
            #undef BLKADD_CT_OMP
        }

        /**
         * @brief OMP version of dense variant direct Cholesky (variants 1 and 2).
         *
         * Variant 1: Single tile factorization
         * Variant 2: Scaled dense with all upper triangular tiles
         */
        void dpotrf_dense_variant_direct_omp(TiledMatrix *tiledMatrix, omp_dep_tracker_t* dep_tracker, int variant, int worldsize) {
            const int rank = omp_get_thread_num();
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles = tiledMatrix->dimTiledMatrix;

            const double zone = 1.0;
            const double mzone = -1.0;

            // Variant 1: Single tile - just one POTRF call
            if (variant == 1) {
                if (rank == 0 && tiledMatrix->denseTiles && tiledMatrix->denseTiles[0]) {
                    double* tile = tiledMatrix->denseTiles[0];
                    sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, N, tile, N);
                    if (status != sTiles::StatusCode::Success) {
                        std::fprintf(stderr, "sTiles error: matrix is not positive definite.\n");
                        dep_abort_all();
                    }
                }
                return;
            }

            // Variant 2: Scaled dense with all upper triangular tiles
            // Upper triangular tile index: upper_idx(i,j) = i * num_tiles - (i * (i - 1)) / 2 + (j - i)
            auto upper_idx = [num_tiles](int i, int j) -> int {
                return i * num_tiles - (i * (i - 1)) / 2 + (j - i);
            };

            // Calculate tile dimensions
            auto tile_dim = [N, tile_size, num_tiles](int t) -> int {
                return (t == num_tiles - 1) ? (N - t * tile_size) : tile_size;
            };

            dep_init(num_tiles, num_tiles, 0);

            // Iterate through k, m, n similar to STYLE_GREEN_TREE_PHASE_1
            int k = 0;
            int m = rank;
            while (m >= num_tiles) {
                k++;
                m = m - num_tiles + k;
            }
            int n = 0;

            while (k < num_tiles && m < num_tiles) {
                // Compute next iteration values
                int next_n = n;
                int next_m = m;
                int next_k = k;

                next_n++;
                if (next_n > next_k) {
                    next_m += worldsize;
                    while (next_m >= num_tiles && next_k < num_tiles) {
                        next_k++;
                        next_m = next_m - num_tiles + next_k;
                    }
                    next_n = 0;
                }

                // Get tile dimensions
                const int tempkn = tile_dim(k);
                const int tempmn = tile_dim(m);
                const int ldak = tile_dim(k);
                const int ldan = tile_dim(n);

                // For dense matrix, all tiles exist - no on_off_tiles check needed
                if (m == k) {
                    if (n == k) {
                        // Case 1: POTRF on diagonal tile (k, k)
                        const int idx_kk = upper_idx(k, k);
                        double* tile_out = tiledMatrix->denseTiles[idx_kk];
                        if (tile_out) {
                            sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
                            if (status != sTiles::StatusCode::Success) {
                                std::fprintf(stderr, "sTiles error: matrix is not positive definite (tile k=%d).\n", k);
                                dep_abort_all();
                            }
                        }
                        dep_set_done(k, k, 1);
                    } else {
                        // Case 2: SYRK - update diagonal tile (k, k) with tile (n, k)
                        const int idx_nk = upper_idx(n, k);
                        const int idx_kk = upper_idx(k, k);

                        dep_wait_for(k, n, 1);
                        double* tile_in = tiledMatrix->denseTiles[idx_nk];
                        double* tile_out = tiledMatrix->denseTiles[idx_kk];
                        if (tile_in && tile_out) {
                            sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone, tile_in, ldan, zone, tile_out, ldak);
                        }
                    }
                } else {
                    if (n == k) {
                        // Case 3: TRSM - solve for tile (k, m)
                        const int idx_km = upper_idx(k, m);
                        const int idx_kk = upper_idx(k, k);

                        dep_wait_for(k, k, 1);
                        double* tile_rhs = tiledMatrix->denseTiles[idx_kk];
                        double* tile_out = tiledMatrix->denseTiles[idx_km];
                        if (tile_rhs && tile_out) {
                            sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit, tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
                        }
                        dep_set_done(m, k, 1);
                    } else {
                        // Case 4: GEMM - update tile (k, m) with tiles (n, k) and (n, m)
                        const int idx_nk = upper_idx(n, k);
                        const int idx_nm = upper_idx(n, m);
                        const int idx_km = upper_idx(k, m);

                        dep_wait_for(k, n, 1);
                        dep_wait_for(m, n, 1);
                        double* tile_a = tiledMatrix->denseTiles[idx_nk];
                        double* tile_b = tiledMatrix->denseTiles[idx_nm];
                        double* tile_out = tiledMatrix->denseTiles[idx_km];
                        if (tile_out && tile_a && tile_b) {
                            sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone, tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
                        }
                    }
                }

                if (dep_is_aborted()) break;

                // Move to next iteration
                n = next_n;
                m = next_m;
                k = next_k;
            }

            dep_finalize();
        }

        /**
         * @brief OMP version of reduction tree Cholesky factorization.
         *
         * Handles tree reduction when red_tree_separator_level > 0.
         */
        void dpotrf_reduction_from_chol_tasks_omp(TiledMatrix *tiledMatrix, omp_dep_tracker_t* dep_tracker, int worldsize) {

            const int rank = omp_get_thread_num();
            const int N = tiledMatrix->dim;
            const int tile_size = tiledMatrix->tile_size;
            const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
            const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
            #define BLKADDI_CT_OMP2(k) ( (k) < full_tiles_per_dim ? tile_size : (N % tile_size) )

            const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
            const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
            const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
            const int end   = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

            dep_init(num_tiles_per_dim, num_tiles_per_dim, 0);

            for (int idx = start; idx < end; ++idx) {
                const std::array<int,7> &t = tasks[idx];
                const int myroutine = t[0];
                const int m = t[1];
                const int k = t[2];
                const int n = t[3];
                const int index1 = t[4];
                const int index2 = t[5];
                const int index3 = t[6];

                const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
                const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
                const int ldak = BLKADDI_CT_OMP2(k);
                const int ldan = BLKADDI_CT_OMP2(n);

                const double zone  = 1.0;
                const double mzone = -1.0;

                switch (myroutine) {
                    case 1: // DPOTRF
                    {
                        double* tile_out = tiledMatrix->denseTiles[index1];
                        if (!tile_out) {
                            break;
                        }
                        sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
                        if (status != sTiles::StatusCode::Success) {
                            std::fprintf(stderr, "sTiles error: matrix is not positive definite (tile k=%d).\n", k);
                            dep_abort_all(); break;
                        }
                        dep_set_done(k, k, 1);
                        break;
                    }
                    case 2: // DSYRK
                    {
                        dep_wait_for(k, n, 1);
                        double* tile_in  = tiledMatrix->denseTiles[index1];
                        double* tile_out = tiledMatrix->denseTiles[index2];
                        if (!tile_out) {
                            break;
                        }
                        sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone,
                                        tile_in, ldan, zone, tile_out, ldak);
                        break;
                    }
                    case 3: // DTRSM
                    {
                        dep_wait_for(k, k, 1);
                        double* tile_rhs = tiledMatrix->denseTiles[index2];
                        double* tile_out = tiledMatrix->denseTiles[index1];
                        if (!tile_out) {
                            break;
                        }
                        sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit,
                                        tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
                        dep_set_done(m, k, 1);
                        break;
                    }
                    case 4: // DGEMM
                    {
                        dep_wait_for(k, n, 1);
                        dep_wait_for(m, n, 1);
                        double* tile_a   = tiledMatrix->denseTiles[index1];
                        double* tile_b   = tiledMatrix->denseTiles[index2];
                        double* tile_out = tiledMatrix->denseTiles[index3];
                        if (!tile_out) {
                            break;
                        }
                        sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone,
                                        tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
                        break;
                    }
                    case 5: // DSYRK (reduction tree)
                    {
                        dep_wait_for(k, n, 1);
                        double* tile_out = tiledMatrix->trees[index2]->nodes[rank].x;
                        double* tile_in  = tiledMatrix->denseTiles[index1];
                        sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, zone,
                                        tile_in, ldan, zone, tile_out, ldak);
                        break;
                    }
                    case 6: // Set dependency flag
                    {
                        tiledMatrix->trees[index2]->dependency[rank] = index3;
                        break;
                    }
                    case 7: // DGEADD (reduction tree accumulate across ranks into tile index2)
                    {
                        const sTiles::TileMetaCore* meta_out_ptr = tiledMatrix->tileMetaCore ? &tiledMatrix->tileMetaCore[index2] : nullptr;
                        const int tile_height = (meta_out_ptr && meta_out_ptr->height > 0) ? meta_out_ptr->height : tile_size;
                        const int tile_width  = (meta_out_ptr && meta_out_ptr->width  > 0) ? meta_out_ptr->width  : tile_size;
                        double* tile_out = tiledMatrix->denseTiles[index2];
                        for (int rank_idx = 0; rank_idx < worldsize; rank_idx++) {
                            if (rank_idx > 0) dep_wait_for_tree(rank_idx, index3, tiledMatrix->trees[index1]->dependency);
                            double* tree_tile = tiledMatrix->trees[index1]->nodes[rank_idx].x;
                            sTiles::core_dgeadd(sTiles::Op::NoTrans, tile_height, tile_width, mzone,
                                                tree_tile, tile_height, zone, tile_out, tile_height);
                            tiledMatrix->trees[index1]->dependency[rank_idx] = 0;
                        }
                        break;
                    }
                    case 8: // DGEMM (reduction tree)
                    {
                        dep_wait_for(k, n, 1);
                        dep_wait_for(m, n, 1);
                        double* tile_out = tiledMatrix->trees[index3]->nodes[rank].x;
                        double* tile_a   = tiledMatrix->denseTiles[index1];
                        double* tile_b   = tiledMatrix->denseTiles[index2];
                        sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, zone,
                                        tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
                        break;
                    }
                    case 9: // Set dependency flag (fixed token)
                    {
                        tiledMatrix->trees[index2]->dependency[rank] = 165715;
                        break;
                    }
                    case 10: // DGEADD (reduction tree from index1 into tile index3)
                    {
                        const sTiles::TileMetaCore* meta_out_ptr2 = tiledMatrix->tileMetaCore ? &tiledMatrix->tileMetaCore[index3] : nullptr;
                        const int tile_height2 = (meta_out_ptr2 && meta_out_ptr2->height > 0) ? meta_out_ptr2->height : tile_size;
                        const int tile_width2  = (meta_out_ptr2 && meta_out_ptr2->width  > 0) ? meta_out_ptr2->width  : tile_size;
                        double* tile_out2 = tiledMatrix->denseTiles[index3];
                        for (int rank_idx = 0; rank_idx < worldsize; rank_idx++) {
                            if (rank_idx > 0) dep_wait_for_tree(rank_idx, 165715, tiledMatrix->trees[index1]->dependency);
                            double* tree_tile = tiledMatrix->trees[index1]->nodes[rank_idx].x;
                            sTiles::core_dgeadd(sTiles::Op::NoTrans, tile_height2, tile_width2, mzone,
                                                tree_tile, tile_height2, zone, tile_out2, tile_height2);
                            tiledMatrix->trees[index1]->dependency[rank_idx] = 0;
                        }
                        break;
                    }
                    default:
                        break;
                }
                if (dep_is_aborted()) break;
            }
            dep_finalize();
            #undef BLKADDI_CT_OMP2
        }

        /**
         * @brief OMP version of pdpotrf - parallel Cholesky factorization using OpenMP.
         *
         * This is the OMP equivalent of pdpotrf. Each OMP thread calls this function
         * and uses omp_get_thread_num() for thread identification and dep_* macros
         * for synchronization.
         *
         * @param[in] tiledMatrix The matrix structure containing tiles and metadata.
         * @param[in,out] dep_tracker The OMP dependency tracker for synchronization.
         * @param[in] worldsize The total number of OMP threads (must match offsets array size).
         */
        void omp_pdpotrf(TiledMatrix* tiledMatrix, omp_dep_tracker_t* dep_tracker, int worldsize) {

            static int* stiles_control_params = sTiles_get_params();
            const int rank = omp_get_thread_num();

            #ifdef STILES_DEBUG_OMP
            #pragma omp single
            std::cerr << "[OMP_DEBUG] omp_pdpotrf: thread 0 entering, worldsize=" << worldsize << std::endl;
            #endif

            if (!tiledMatrix) {
                #pragma omp single
                std::cerr << "Error: null tiledMatrix in omp_pdpotrf" << std::endl;
                return;
            }

            // Reset chol_info before factorization (single thread)
            #pragma omp single
            { tiledMatrix->chol_info.store(0, std::memory_order_relaxed); }

            #ifdef STILES_DEBUG_OMP
            #pragma omp single
            std::cerr << "[OMP_DEBUG] omp_pdpotrf: use_boosted_e_trick=" << tiledMatrix->use_boosted_e_trick
                      << ", red_tree_separator_level=" << tiledMatrix->red_tree_separator_level
                      << ", tile_type_mode=" << stiles_control_params[3] << std::endl;
            #endif

            if (tiledMatrix->use_boosted_e_trick) {
                if (tiledMatrix->red_tree_separator_level > 0) {
                    #ifdef STILES_DEBUG_OMP
                    #pragma omp single
                    std::cerr << "[OMP_DEBUG] omp_pdpotrf: calling dpotrf_reduction_from_chol_tasks_omp" << std::endl;
                    #endif
                    dpotrf_reduction_from_chol_tasks_omp(tiledMatrix, dep_tracker, worldsize);
                } else {
                    // Non-reduction mode with boosted e_trick
                    if (stiles_control_params[3] == 0) {
                        #ifdef STILES_DEBUG_OMP
                        #pragma omp single
                        std::cerr << "[OMP_DEBUG] omp_pdpotrf: calling dpotrf_expansion_from_chol_tasks_dense_omp" << std::endl;
                        #endif
                        dpotrf_expansion_from_chol_tasks_dense_omp(tiledMatrix, dep_tracker, worldsize);

                    } else if (stiles_control_params[3] == 1) {
                        // Semisparse mode - check if semisparse structures are available
                        if (tiledMatrix->chunkedDenseTiles && tiledMatrix->semisparseTileMetaCore) {
                            // CPU semisparse: use OMP version
                            #ifdef STILES_DEBUG_OMP
                            #pragma omp single
                            std::cerr << "[OMP_DEBUG] omp_pdpotrf: calling dpotrf_expansion_from_chol_tasks_semisparse_omp" << std::endl;
                            #endif
                            dpotrf_expansion_from_chol_tasks_semisparse_omp(tiledMatrix, dep_tracker, worldsize);
                        } else {
                            #ifdef STILES_DEBUG_OMP
                            #pragma omp single
                            std::cerr << "[OMP_DEBUG] omp_pdpotrf: semisparse fallback to dense" << std::endl;
                            #endif
                            dpotrf_expansion_from_chol_tasks_dense_omp(tiledMatrix, dep_tracker, worldsize);
                        }

                    } else {
                        #pragma omp single
                        std::cout << "OMP mode not implemented for tile_type=" << stiles_control_params[3]
                                  << ". Only dense (0) and semisparse (1) modes are supported." << std::endl;
                    }
                }
            } else {
                // Dense variants 1 and 2 without pre-collected tasks
                const int num_active = tiledMatrix->numActiveTiles;
                const int dim_tiled = tiledMatrix->dimTiledMatrix;
                const int tri_size = tiledMatrix->triangular_size;

                #ifdef STILES_DEBUG_OMP
                #pragma omp single
                std::cerr << "[OMP_DEBUG] omp_pdpotrf: non-boosted path, num_active=" << num_active
                          << ", dim_tiled=" << dim_tiled << ", tri_size=" << tri_size << std::endl;
                #endif

                if (num_active == 1 && dim_tiled == 1) {
                    #ifdef STILES_DEBUG_OMP
                    #pragma omp single
                    std::cerr << "[OMP_DEBUG] omp_pdpotrf: calling dpotrf_dense_variant_direct_omp variant=1" << std::endl;
                    #endif
                    dpotrf_dense_variant_direct_omp(tiledMatrix, dep_tracker, 1, worldsize);
                } else if (num_active == tri_size && tri_size > 0) {
                    #ifdef STILES_DEBUG_OMP
                    #pragma omp single
                    std::cerr << "[OMP_DEBUG] omp_pdpotrf: calling dpotrf_dense_variant_direct_omp variant=2" << std::endl;
                    #endif
                    dpotrf_dense_variant_direct_omp(tiledMatrix, dep_tracker, 2, worldsize);
                }
            }

            // Record Cholesky status: 0 = success, 1 = not positive definite
            #pragma omp single
            {
                tiledMatrix->chol_info.store(
                    dep_tracker->abort_flag.load(std::memory_order_acquire) ? 1 : 0,
                    std::memory_order_release);
            }

            #ifdef STILES_DEBUG_OMP
            #pragma omp barrier
            #pragma omp single
            std::cerr << "[OMP_DEBUG] omp_pdpotrf: completed successfully" << std::endl;
            #endif
        }


}}
#
