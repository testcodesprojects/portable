/**
 * @file semisparsepdpotrf.cpp
 * @brief Semisparse variant of parallel Cholesky factorization.
 *
 * Implements Cholesky factorization for semisparse matrices that track
 * active columns and bandwidth for efficient banded factorization.
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
#include <string>
#include <functional>
#include <array>
#include <vector>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <iomanip>

#include "../common/core_lapack.hpp"
// Per-thread scratch for SmartTile and dense conversions
#include "../memory/stile_threadWorkspace.hpp"

#include "../control/stiles_control.hpp"
#include "../compute/stiles_compute.hpp"
#include "../control/common.h" //remove
#include "../tile/meta.hpp"

#include "../common/stiles_utils.hpp"
#include "debug_tiles.hpp"
#include "tile_compare.hpp"
#ifdef SMART_TILES
// SmartTile CHOL compute (symbolic + numeric)
#endif

#ifdef STILES_SAFEMODE
namespace sTiles{ namespace Process{
void pdpotrf(stiles_context_t *stile) {

    TiledMatrix* tiledMatrix = nullptr;
    sTiles::unpack_args(stile, tiledMatrix);

    if (!tiledMatrix) {
        std::cout << "Error: null tiledMatrix in stiles_pdpotrf" << std::endl;
        return;
    }

    if (tiledMatrix->use_boosted_e_trick) {
        if (tiledMatrix->red_tree_separator_level > 0) {
            sTiles::SafeMode::dpotrf_reduction(tiledMatrix, stile);
        } else {
            sTiles::SafeMode::dpotrf_expansion(tiledMatrix, stile);
        }
    } else {
        std::cout << "FIXME: SafeMode without boosted_e_trick is not implemented." << std::endl;
    }

}}}
#endif


#if defined(STILES_FASTMODE) || defined(STILES_SEMISPARSEMODE)
namespace sTiles{ namespace Process{

namespace {

static inline void print_tile_debug(const char* when,
                                    const char* routine_name,
                                    int myroutine,
                                    int m, int k, int n,
                                    int index,
                                    const double* tile,
                                    int rows, int cols, int ld,
                                    const sTiles::TileMetaCore* meta,
                                    const sTiles::SparseTileMetaCore* sparse_meta)
{
    sTiles::Logger::info("[ESMAIL_CHECK] ", routine_name, " ", when,
                         " r=", myroutine,
                         " m=", m, " k=", k, " n=", n,
                         " idx=", index,
                         " tile=", (meta ? meta->row : -1), ",", (meta ? meta->col : -1),
                         " rows=", rows, " cols=", cols,
                         " ld=", ld);

    if (meta) {
        sTiles::Logger::info("    meta row=", meta->row,
                             " col=", meta->col,
                             " width=", meta->width,
                             " height=", meta->height);
    } else {
        sTiles::Logger::info("    meta: <none>");
    }

    if (sparse_meta && meta && meta->row == meta->col) {
        sTiles::Logger::info("    diag upper_bw=", sparse_meta->upper_bw);
    }

    if (!tile) {
        sTiles::Logger::warning("[ESMAIL_CHECK] null tile pointer in ", routine_name,
                                " ", when, " idx=", index);
        return;
    }

    for (int i = 0; i < rows; ++i) {
        std::ostringstream oss;
        oss.setf(std::ios::fixed, std::ios::floatfield);
        oss << std::setprecision(6);
        oss << "    row " << i << ": ";
        for (int j = 0; j < cols; ++j) {
            const double val = tile[i + j * ld];
            if (std::abs(val) < 1e-12) {
                oss << std::setw(12) << ' ';
            } else {
                oss << std::setw(12) << val;
            }
        }
        sTiles::Logger::info(oss.str());
    }

    if (sparse_meta) {
        if (!sparse_meta->colptr.empty()) {
            std::ostringstream oss;
            oss << "    colptr:";
            for (const auto val : sparse_meta->colptr) {
                oss << ' ' << val;
            }
            sTiles::Logger::info(oss.str());
        }
        if (!sparse_meta->rowind.empty()) {
            std::ostringstream oss;
            oss << "    rowind:";
            for (const auto val : sparse_meta->rowind) {
                oss << ' ' << val;
            }
            sTiles::Logger::info(oss.str());
        }
        // acol/temp_acol no longer tracked in SparseTileMetaCore
    }
}

static inline void refresh_acol(const double* tile,
                                int rows,
                                int cols,
                                int ld,
                                sTiles::SparseTileMetaCore* sparse_meta)
{
    (void)tile;
    (void)rows;
    (void)cols;
    (void)ld;
    (void)sparse_meta;
}

static inline void log_acol(int tile_index,
                            const sTiles::SparseTileMetaCore* sparse_meta)
{
    (void)tile_index;
    (void)sparse_meta;
}

} // namespace

void dpotrf_expansion_from_chol_tasks(TiledMatrix *tiledMatrix, stiles_context_t *stile) {
    const int rank = STILES_RANK;
    const int N = tiledMatrix->dim;
    const int tile_size = tiledMatrix->tile_size;
    const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
    const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
    #define BLKADD_CT(k) ( (k) < full_tiles_per_dim ? tile_size : (N % tile_size) )

    const auto &tasks = sTiles::get_chol_tasks(tiledMatrix);
    const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
    const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
    const int end   = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

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

        const double zone  = 1.0;
        const double mzone = -1.0;

    const auto get_meta = [&](int tile_index) -> const sTiles::TileMetaCore* {
        if (!tiledMatrix->tileMetaCore) return nullptr;
        if (tile_index < 0 || tile_index >= tiledMatrix->numActiveTiles) return nullptr;
        return &tiledMatrix->tileMetaCore[tile_index];
    };
    const auto get_sparse_meta = [&](int tile_index) -> sTiles::SparseTileMetaCore* {
        if (!tiledMatrix->sparseTileMetaCore) return nullptr;
        if (tile_index < 0 || tile_index >= tiledMatrix->numActiveTiles) return nullptr;
        return &tiledMatrix->sparseTileMetaCore[tile_index];
    };

        switch (myroutine) {
            case 1: // DPOTRF
            {
                double* tile_out = tiledMatrix->denseTiles[index1];
                const sTiles::TileMetaCore* meta_out = get_meta(index1);
                auto* sparse_meta = get_sparse_meta(index1);

                print_tile_debug("before", "DPOTRF",
                                 myroutine, m, k, n,
                                 index1,
                                 tile_out,
                                 tempkn, tempkn, ldak,
                                 meta_out,
                                 sparse_meta);

                sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
                (void)status;

                refresh_acol(tile_out, tempkn, tempkn, ldak, sparse_meta);
                print_tile_debug("after", "DPOTRF",
                                 myroutine, m, k, n,
                                 index1,
                                 tile_out,
                                 tempkn, tempkn, ldak,
                                 meta_out,
                                 sparse_meta);
                log_acol(index1, sparse_meta);
                std::cout << std::endl;
                ss_cond_set(k, k, 1);
                break;
            }
            case 2: // DSYRK
            {
                ss_cond_wait(k, n, 1);
                double* tile_in  = tiledMatrix->denseTiles[index1];
                double* tile_out = tiledMatrix->denseTiles[index2];
                const sTiles::TileMetaCore* meta_in  = get_meta(index1);
                const sTiles::TileMetaCore* meta_out = get_meta(index2);
                auto* sparse_in  = get_sparse_meta(index1);
                auto* sparse_out = get_sparse_meta(index2);

                print_tile_debug("before A", "DSYRK",
                                 myroutine, m, k, n,
                                 index1,
                                 tile_in,
                                 tile_size, tempkn, ldan,
                                 meta_in,
                                 sparse_in);

                print_tile_debug("before C", "DSYRK",
                                 myroutine, m, k, n,
                                 index2,
                                 tile_out,
                                 tempkn, tempkn, ldak,
                                 meta_out,
                                 sparse_out);

                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone,
                                   tile_in, ldan, zone, tile_out, ldak);

                refresh_acol(tile_out, tempkn, tempkn, ldak, sparse_out);
                print_tile_debug("after C", "DSYRK",
                                 myroutine, m, k, n,
                                 index2,
                                 tile_out,
                                 tempkn, tempkn, ldak,
                                 meta_out,
                                 sparse_out);
                log_acol(index2, sparse_out);
                std::cout << std::endl;
                break;
            }
            case 3: // DTRSM
            {
                ss_cond_wait(k, k, 1);
                double* tile_rhs = tiledMatrix->denseTiles[index2];
                double* tile_out = tiledMatrix->denseTiles[index1];
                const sTiles::TileMetaCore* meta_rhs = get_meta(index2);
                const sTiles::TileMetaCore* meta_out = get_meta(index1);
                auto* sparse_rhs = get_sparse_meta(index2);
                auto* sparse_out = get_sparse_meta(index1);

                print_tile_debug("before A", "DTRSM",
                                 myroutine, m, k, n,
                                 index2,
                                 tile_rhs,
                                 tile_size, tile_size, ldak,
                                 meta_rhs,
                                 sparse_rhs);

                print_tile_debug("before B", "DTRSM",
                                 myroutine, m, k, n,
                                 index1,
                                 tile_out,
                                 tile_size, tempmn, ldak,
                                 meta_out,
                                 sparse_out);

                sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit,
                                   tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);

                refresh_acol(tile_out, tile_size, tempmn, ldak, sparse_out);
                print_tile_debug("after B", "DTRSM",
                                 myroutine, m, k, n,
                                 index1,
                                 tile_out,
                                 tile_size, tempmn, ldak,
                                 meta_out,
                                 sparse_out);
                log_acol(index1, sparse_out);
                std::cout << std::endl;
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
                const sTiles::TileMetaCore* meta_a   = get_meta(index1);
                const sTiles::TileMetaCore* meta_b   = get_meta(index2);
                const sTiles::TileMetaCore* meta_out = get_meta(index3);
                auto* sparse_a   = get_sparse_meta(index1);
                auto* sparse_b   = get_sparse_meta(index2);
                auto* sparse_out = get_sparse_meta(index3);

                print_tile_debug("before A", "DGEMM",
                                 myroutine, m, k, n,
                                 index1,
                                 tile_a,
                                 tile_size, tile_size, ldan,
                                 meta_a,
                                 sparse_a);

                print_tile_debug("before B", "DGEMM",
                                 myroutine, m, k, n,
                                 index2,
                                 tile_b,
                                 tile_size, tempmn, ldan,
                                 meta_b,
                                 sparse_b);

                print_tile_debug("before C", "DGEMM",
                                 myroutine, m, k, n,
                                 index3,
                                 tile_out,
                                 tile_size, tempmn, ldak,
                                 meta_out,
                                 sparse_out);

                // Dense path only
                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone,
                                   tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);

                refresh_acol(tile_out, tile_size, tempmn, ldak, sparse_out);
                print_tile_debug("after C", "DGEMM",
                                 myroutine, m, k, n,
                                 index3,
                                 tile_out,
                                 tile_size, tempmn, ldak,
                                 meta_out,
                                 sparse_out);
                log_acol(index3, sparse_out);
                std::cout << std::endl;
                break;
            }
            default:
                sTiles::Logger::warning("│   [ESMAIL_CHECK] Unexpected routine id ", myroutine, " in expansion tasks");
                break;
        }
        if (ss_aborted()) break;
    }
    ss_finalize();

    #undef BLKADD_CT
}

void dpotrf_reduction_from_chol_tasks(TiledMatrix *tiledMatrix, stiles_context_t *stile) {
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
                sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
                (void)status;
                ss_cond_set(k, k, 1);
                break;
            }
            case 2: // DSYRK
            {
                ss_cond_wait(k, n, 1);
                double* tile_in  = tiledMatrix->denseTiles[index1];
                double* tile_out = tiledMatrix->denseTiles[index2];
                sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone,
                                   tile_in, ldan, zone, tile_out, ldak);
                break;
            }
            case 3: // DTRSM
            {
                ss_cond_wait(k, k, 1);
                double* tile_rhs = tiledMatrix->denseTiles[index2];
                double* tile_out = tiledMatrix->denseTiles[index1];
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

void pdpotrf(stiles_context_t *stile) {

    TiledMatrix* tiledMatrix = nullptr;
    sTiles::unpack_args(stile, tiledMatrix);

    if (!tiledMatrix) {
        std::cout << "Error: null tiledMatrix in stiles_pdpotrf" << std::endl;
        return;
    }

    if (tiledMatrix->use_boosted_e_trick) {
        if (tiledMatrix->red_tree_separator_level > 0) {
            sTiles::Process::dpotrf_reduction_from_chol_tasks(tiledMatrix, stile);
        } else {
            sTiles::Process::dpotrf_expansion_from_chol_tasks(tiledMatrix, stile);
        }
    } else {
        std::cout << "FIXME: SafeMode without boosted_e_trick is not implemented." << std::endl;
    }

}}}
#endif











// namespace {

// inline void smart_debug_print_dense_tile(const char* case_name,
//                                          int tile_index,
//                                          const double* data,
//                                          int rows,
//                                          int cols,
//                                          int ld,
//                                          int max_show = 6) {
//     if (!case_name) case_name = "UNKNOWN";
//     if (!data) {
//         std::cout << "[SMART-CHECK] " << case_name
//                   << ": tile_out index=" << tile_index << " is null." << std::endl;
//         return;
//     }
//     std::cout << "[SMART-CHECK] " << case_name
//               << ": tile_out index=" << tile_index
//               << " (rows=" << rows << ", cols=" << cols
//               << ", ld=" << ld << ")" << std::endl;
//     sTiles::debug::print_dense_tile(case_name, data, rows, cols, ld,
//                                     std::min(rows, max_show),
//                                     std::min(cols, max_show), 6);
// }

// } // anonymous namespace


// namespace sTiles{ namespace Process
// {

// void dpotrf_expansion(TiledMatrix *tiledMatrix, stiles_context_t *stile){

//     const int rank = STILES_RANK;
//     const int N = tiledMatrix->dim;
//     const int tile_size = tiledMatrix->tile_size;
//     const int num_tiles_per_dim = (N == 0) ? 0 : (N-1)/tile_size + 1;
//     const int full_tiles_per_dim = N/tile_size;
//     #define BLKADD(k) ( (k) < full_tiles_per_dim ? tile_size : N % tile_size )

//     int k, m, n;
//     int index1, index2, index3, myroutine;
//     int ldak, ldan;
//     int tempkn, tempmn;
//     const double zone  = (double) 1.0;
//     const double mzone = (double)-1.0;
//     const int num_tasks = tiledMatrix->e_trick_size[rank];

//     ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
//     for(int i=0; i<num_tasks;i++){

//         myroutine = tiledMatrix->e_trick[rank][0+7*i];
//         m = tiledMatrix->e_trick[rank][1+i*7];
//         k = tiledMatrix->e_trick[rank][2+i*7];
//         n = tiledMatrix->e_trick[rank][3+i*7];
//         index1 = tiledMatrix->e_trick[rank][4+i*7];
//         index2 = tiledMatrix->e_trick[rank][5+i*7];
//         index3 = tiledMatrix->e_trick[rank][6+i*7];

//         tempkn = k == (num_tiles_per_dim-1) ? N-k*tile_size : tile_size;
//         tempmn = m == (num_tiles_per_dim-1) ? N-m*tile_size : tile_size;
//         ldak = BLKADD(k);
//         ldan = BLKADD(n);

//         switch (myroutine) {
//             case 1:
//             {

//                 // double* tile_out = tiledMatrix->dense_tiles[index1].elements;
//                 // sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);

//                 double* tile_out1 = tiledMatrix->denseTiles[index1];
//                 if (STILES_RANK == 0 && tile_out1) {
//                     sTiles::debug::print_tile_dense("case1: pre-dpotrf U (dense)", tile_out1, ldak, tempkn, ldak, 6);
//                 }
//                 sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out1, ldak);
//                 if (status != sTiles::StatusCode::Success) std::cout << "fix me" << std::endl;
//                 if (STILES_RANK == 0 && tile_out1) {
//                     sTiles::debug::print_tile_dense("case1: post-dpotrf U (dense)", tile_out1, ldak, tempkn, ldak, 6);
//                 }

//                 ss_cond_set(k, k, 1);
//                 break;
//             }
        
//             case 2: // DSYRK
//             {
//                 ss_cond_wait(k, n, 1);
//                 // double* tile_in = tiledMatrix->dense_tiles[index1].elements;
//                 // double* tile_out = tiledMatrix->dense_tiles[index2].elements;
//                 // sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone,
//                 //                    tile_in, ldan, zone, tile_out, ldak);

//                 double* tile_in1 = tiledMatrix->denseTiles[index1];
//                 double* tile_out1 = tiledMatrix->denseTiles[index2];
//                 if (!tile_in1 || !tile_out1) {
//                     std::cout << "fix me" << std::endl;
//                     break;
//                 }
//                 if (STILES_RANK == 0) {
//                     sTiles::debug::print_tile_dense("case2: DSYRK before: A (dense)", tile_in1, ldan, tile_size, ldan, 6);
//                     sTiles::debug::print_tile_dense("case2: DSYRK before: C (dense)", tile_out1, ldak, tile_size, ldak, 6);
//                 }
//                 sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone,
//                                    tile_in1, ldan, zone, tile_out1, ldak);
//                 if (STILES_RANK == 0) {
//                     sTiles::debug::print_tile_dense("case2: DSYRK after: C (dense)", tile_out1, ldak, tile_size, ldak, 6);
//                 }

//                 break;
//             }

//             case 3: // DTRSM
//             {
//                 ss_cond_wait(k, k, 1);
//                 // double* tile_rhs = tiledMatrix->dense_tiles[index2].elements;
//                 // double* tile_out = tiledMatrix->dense_tiles[index1].elements;
//                 // sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit,
//                 //                    tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
                
//                 double* tile_rhs1 = tiledMatrix->denseTiles[index2];
//                 double* tile_out1 = tiledMatrix->denseTiles[index1];
//                 if (STILES_RANK == 0) {
//                     if (tile_out1) sTiles::debug::print_tile_dense("case3: DTRSM before: rhs (dense)", tile_out1, ldak, tempmn, ldak, 6);
//                     if (tile_rhs1) sTiles::debug::print_tile_dense("case3: DTRSM before: diag (dense)", tile_rhs1, ldak, ldak, ldak, 6);
//                 }
//                 sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit,
//                                    tile_size, tempmn, zone, tile_rhs1, ldak, tile_out1, ldak);
//                 if (STILES_RANK == 0) {
//                     if (tile_out1) sTiles::debug::print_tile_dense("case3: DTRSM after: rhs (dense)", tile_out1, ldak, tempmn, ldak, 6);
//                     if (tile_rhs1) sTiles::debug::print_tile_dense("case3: DTRSM after: diag (dense)", tile_rhs1, ldak, ldak, ldak, 6);
//                 }

//                 ss_cond_set(m, k, 1);
//                 break;
//             }
            
//             case 4: // DGEMM
//             {
//                 ss_cond_wait(k, n, 1);
//                 ss_cond_wait(m, n, 1);
//                 // double* tile_a = tiledMatrix->dense_tiles[index1].elements;
//                 // double* tile_b = tiledMatrix->dense_tiles[index2].elements;
//                 // double* tile_out = tiledMatrix->dense_tiles[index3].elements;
//                 // sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone,
//                 //                    tile_a, ldan,
//                 //                    tile_b, ldan, zone, tile_out, ldak);

//                 double* tile_a1 = tiledMatrix->denseTiles[index1];
//                 double* tile_b1 = tiledMatrix->denseTiles[index2];
//                 double* tile_out1 = tiledMatrix->denseTiles[index3];
//                 if (STILES_RANK == 0) {
//                     if (tile_a1)  sTiles::debug::print_tile_dense("case4: DGEMM before: A (dense)", tile_a1, ldan, tile_size, ldan, 6);
//                     if (tile_b1)  sTiles::debug::print_tile_dense("case4: DGEMM before: B (dense)", tile_b1, ldan, tempmn, ldan, 6);
//                     if (tile_out1) sTiles::debug::print_tile_dense("case4: DGEMM before: C (dense)", tile_out1, ldak, tempmn, ldak, 6);
//                 }
//                 sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone,
//                                    tile_a1, ldan,
//                                    tile_b1, ldan, zone, tile_out1, ldak);

//                 if (STILES_RANK == 0) {
//                     if (tile_out1) sTiles::debug::print_tile_dense("case4: DGEMM after: C (dense)", tile_out1, ldak, tempmn, ldak, 6);
//                 }
                                   
//                 break;
//             }
//         }
        
//         #ifdef COMPUTE_FLOPS
//             if (myroutine >= 1 && myroutine <= 4) {
//                  tiledMatrix->flops_mat[(rank * 5) + (myroutine - 1)] += 1;
//             }
//         #endif

//         if(ss_aborted()) break;
//     }
//     ss_finalize();

// #undef BLKADD
// }

// void dpotrf_reduction(TiledMatrix *tiledMatrix, stiles_context_t *stile) {


//     const int rank = STILES_RANK;
//     const int N = tiledMatrix->dim;
//     const int tile_size = tiledMatrix->tile_size;
//     const int num_tiles_per_dim = (N == 0) ? 0 : (N-1)/tile_size + 1;
//     const int sep = tiledMatrix->red_tree_separator_level;
//     const int num_sep = ((sep * sep) - sep) / 2 + sep;
//     const int full_tiles_per_dim = N/tile_size;
//     #define BLKADDI(k) ( (k) < full_tiles_per_dim ? tile_size : N % tile_size )

//     int k, m, n;
//     int ldak, ldan;
//     int tempkn, tempmn;
//     int index1, index2, index3;
//     int myroutine;
//     double zone = 1.0, mzone = -1.0;

//     sTiles::Utils::TaskDistribution *distribution = (sTiles::Utils::TaskDistribution *)malloc(num_sep * sizeof(sTiles::Utils::TaskDistribution));
//     for (int ind = 0; ind < num_sep; ind++) {
//         if (tiledMatrix->trees[ind]) {
//             distribution[ind] = sTiles::Utils::calculateTaskDistribution(rank, STILES_SIZE, tiledMatrix->trees[ind]->num_tasks);
//         }
//     }

//     ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
//     for (int i = 0; i < tiledMatrix->e_trick_size[rank]; i++) {

//         myroutine = tiledMatrix->e_trick[rank][0 + 7 * i];
//         m = tiledMatrix->e_trick[rank][1 + i * 7];
//         k = tiledMatrix->e_trick[rank][2 + i * 7];
//         n = tiledMatrix->e_trick[rank][3 + i * 7];
//         index1 = tiledMatrix->e_trick[rank][4 + i * 7];
//         index2 = tiledMatrix->e_trick[rank][5 + i * 7];
//         index3 = tiledMatrix->e_trick[rank][6 + i * 7];

//         tempkn = (k == (num_tiles_per_dim - 1)) ? N - k * tile_size : tile_size;
//         tempmn = (m == (num_tiles_per_dim - 1)) ? N - m * tile_size : tile_size;
//         ldak = BLKADDI(k);
//         ldan = BLKADDI(n);

//         switch (myroutine) {
//             case 1: // DPOTRF
//             {
//                 double* tile_out = tiledMatrix->denseTiles[index1];
//                 if (!tile_out) {
//                     std::cout << "fix me" << std::endl;
//                     break;
//                 }
//                 sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
//                 if (status != sTiles::StatusCode::Success) std::cout << "fix me" << std::endl;
//                 ss_cond_set(k, k, 1);
//                 break;
//             }
        
//             case 2: // DSYRK
//             {
//                 ss_cond_wait(k, n, 1);
//                 double* tile_in = tiledMatrix->denseTiles[index1];
//                 double* tile_out = tiledMatrix->denseTiles[index2];
//                 if (!tile_in || !tile_out) {
//                     std::cout << "fix me" << std::endl;
//                     break;
//                 }
//                 sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone,
//                                    tile_in, ldan, zone, tile_out, ldak);
//                 break;
//             }

//             case 3: // DTRSM
//             {
//                 ss_cond_wait(k, k, 1);
//                 double* tile_rhs = tiledMatrix->denseTiles[index2];
//                 double* tile_out = tiledMatrix->denseTiles[index1];
//                 if (!tile_rhs || !tile_out) {
//                     std::cout << "fix me" << std::endl;
//                     break;
//                 }
//                 sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit,
//                                    tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
//                 ss_cond_set(m, k, 1);
//                 break;
//             }
            
//             case 4: // DGEMM
//             {
//                 ss_cond_wait(k, n, 1);
//                 ss_cond_wait(m, n, 1);
//                 double* tile_a = tiledMatrix->denseTiles[index1];
//                 double* tile_b = tiledMatrix->denseTiles[index2];
//                 double* tile_out = tiledMatrix->denseTiles[index3];
//                 if (!tile_a || !tile_b || !tile_out) {
//                     std::cout << "fix me" << std::endl;
//                     break;
//                 }
//                 sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone,
//                                    tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
//                 break;
//             }

//             case 5: // DSYRK (reduction tree)
//             {
//                 ss_cond_wait(k, n, 1);
//                 double* tile_out = tiledMatrix->trees[index2]->nodes[STILES_RANK].x;
//                 double* tile_in = tiledMatrix->denseTiles[index1];
//                 if (!tile_out || !tile_in) {
//                     std::cout << "fix me" << std::endl;
//                     break;
//                 }
//                 sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, zone,
//                                    tile_in, ldan, zone, tile_out, ldak);
//                 break;
//             }

//             case 6: // Set dependency flag
//                 tiledMatrix->trees[index2]->dependency[STILES_RANK] = index3;
//                 break;

//             case 7: // DGEADD (reduction tree)
//             {
//                 const sTiles::TileMetaCore* meta_out_ptr = tiledMatrix->tileMetaCore ? &tiledMatrix->tileMetaCore[index2] : nullptr;
//                 const int tile_height = (meta_out_ptr && meta_out_ptr->height > 0) ? meta_out_ptr->height : tile_size;
//                 const int tile_width  = (meta_out_ptr && meta_out_ptr->width  > 0) ? meta_out_ptr->width  : tile_size;
//                 double* tile_out = tiledMatrix->denseTiles[index2];
//                 for (int rank_idx = 0; rank_idx < STILES_SIZE; rank_idx++) {
//                     if (rank_idx > 0) {
//                         ss_cond_wait_tree_e_s_t_y_l_e(rank_idx, index3, tiledMatrix->trees[index1]->dependency);
//                     }
//                     double* tree_tile = tiledMatrix->trees[index1]->nodes[rank_idx].x;
//                     if (!tile_out || !tree_tile) {
//                         std::cout << "fix me" << std::endl;
//                         continue;
//                     }
//                     sTiles::core_dgeadd(sTiles::Op::NoTrans, tile_height, tile_width, mzone,
//                                         tree_tile, tile_height, zone, tile_out, tile_height);
//                     tiledMatrix->trees[index1]->dependency[rank_idx] = 0;
//                 }
//                 break;
//             }

//             case 8: // DGEMM (reduction tree)
//             {
//                 ss_cond_wait(k, n, 1);
//                 ss_cond_wait(m, n, 1);
//                 double* tile_out = tiledMatrix->trees[index3]->nodes[STILES_RANK].x;
//                 double* tile_a   = tiledMatrix->denseTiles[index1];
//                 double* tile_b   = tiledMatrix->denseTiles[index2];
//                 if (!tile_out || !tile_a || !tile_b) {
//                     std::cout << "fix me" << std::endl;
//                     break;
//                 }
//                 sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, zone,
//                                    tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
//                 break;
//             }

//             case 9: // Set dependency flag
//                 tiledMatrix->trees[index2]->dependency[STILES_RANK] = 165715;
//                 break;

//             case 10: // DGEADD (reduction tree)
//             {
//                 const sTiles::TileMetaCore* meta_out_ptr2 = tiledMatrix->tileMetaCore ? &tiledMatrix->tileMetaCore[index3] : nullptr;
//                 const int tile_height2 = (meta_out_ptr2 && meta_out_ptr2->height > 0) ? meta_out_ptr2->height : tile_size;
//                 const int tile_width2  = (meta_out_ptr2 && meta_out_ptr2->width  > 0) ? meta_out_ptr2->width  : tile_size;
//                 double* tile_out2 = tiledMatrix->denseTiles[index3];
//                 if (!tile_out2) {
//                     std::cout << "fix me" << std::endl;
//                     break;
//                 }
//                 for (int rank_idx = 0; rank_idx < STILES_SIZE; rank_idx++) {
//                     if (rank_idx > 0) {
//                         ss_cond_wait_tree_e_s_t_y_l_e(rank_idx, 165715, tiledMatrix->trees[index1]->dependency);
//                     }
//                     double* tree_tile = tiledMatrix->trees[index1]->nodes[rank_idx].x;
//                     if (!tile_out2 || !tree_tile) {
//                         std::cout << "fix me" << std::endl;
//                         continue;
//                     }
//                     sTiles::core_dgeadd(sTiles::Op::NoTrans, tile_height2, tile_width2, mzone,
//                                         tree_tile, tile_height2, zone, tile_out2, tile_height2);
//                     tiledMatrix->trees[index1]->dependency[rank_idx] = 0;
//                 }
//                 break;
//             }
//         }
//         if (ss_aborted()) break;
//     }

//     ss_finalize();
//     free(distribution);

// #undef BLKADDI
// }

// //
// // Alternative execution using collected chol_tasks/chol_task_offsets
// //




















// void symbolic(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

//     const int world = STILES_SIZE;
//     const int rank  = STILES_RANK;

//     // Lazily allocate one sTiles::ThreadWorkspace per core and cache on the context.
//     // Keeps setup cost out of the inner loops and allows reuse across calls.
//     {
//         // Step 1: rank 0 owns the shared pointer-array (allocate/resize as needed)
//         if (rank == 0) {
//             const bool need_realloc = (!stile->thread_workspaces) || (stile->thread_workspaces_count != world);
//             if (need_realloc) {
//                 // Cleanup stale set, if any
//                 if (stile->thread_workspaces) {
//                     for (int r = 0; r < stile->thread_workspaces_count; ++r) {
//                         if (stile->thread_workspaces[r]) {
//                             auto *tw_old = reinterpret_cast<sTiles::ThreadWorkspace*>(stile->thread_workspaces[r]);
//                             delete tw_old;
//                             stile->thread_workspaces[r] = nullptr;
//                         }
//                     }
//                     free(stile->thread_workspaces);
//                     stile->thread_workspaces = nullptr;
//                     stile->thread_workspaces_count = 0;
//                 }
//                 stile->thread_workspaces = static_cast<void**>(calloc((size_t)world, sizeof(void*)));
//                 stile->thread_workspaces_count = world;
//             }
//         }

//         // Ensure all threads see the pointer array before allocating their own entries
//         sTiles::Control::Barrier(stile);

//         // Step 2: Each rank allocates its own workspace lazily
//         if (stile->thread_workspaces && stile->thread_workspaces_count == world && !stile->thread_workspaces[rank]) {
//             const int group_id = 0; // Grouping for workspace allocator (0 = default pool)
//             const int gpu_id   = tiledMatrix ? tiledMatrix->GPU_ID : -1;
//             const int max_dim  = tiledMatrix ? tiledMatrix->tile_size : 0;
//             const bool use_gpu = tiledMatrix ? tiledMatrix->use_gpu : false;

//             auto *tw = new sTiles::ThreadWorkspace(rank, group_id, gpu_id, max_dim, use_gpu);
//             stile->thread_workspaces[rank] = reinterpret_cast<void*>(tw);
//         }

//         // Optional: sync after allocation so every rank has a valid workspace
//         sTiles::Control::Barrier(stile);
//     }

//     //text 
//     /*
//         each core will enter this function 
//         each core should have it own workspace  

    
//     */

//     // sTiles::SmartTile* st = tiledMatrix->facTiles[2];
//     // st->printSymbolicSparse();
//     // exit(0);

//     //std::cout <<"start"<<std::endl;
//     const int N = tiledMatrix->dim;
//     const int tile_size = tiledMatrix->tile_size;
//     const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
//     const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
//     #define BLKADD_CT(k) ( (k) < full_tiles_per_dim ? tile_size : (N % tile_size) )

//     const auto &tasks = tiledMatrix->chol_tasks;
//     const auto &offsets = tiledMatrix->chol_task_offsets;
//     const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
//     const int end   = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

//     ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
//     for (int idx = start; idx < end; ++idx) {
//         const std::array<int,7> &t = tasks[idx];
//         const int myroutine = t[0];
//         const int m = t[1];
//         const int k = t[2];
//         const int n = t[3];
//         const int index1 = t[4];
//         const int index2 = t[5];
//         const int index3 = t[6];

//         const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
//         const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
//         const int ldak = BLKADD_CT(k);
//         const int ldan = BLKADD_CT(n);

//         const double zone  = 1.0;
//         const double mzone = -1.0;

//         //std::cout << "pdpppppppppppppppppppppppppppppppotrf" << std::endl;
//         //sTiles::SmartTile* st = tiledMatrix->facTiles[2];
//         //st->printSymbolicSparse();

//         //exit(0);
//         switch (myroutine) {
//             case 1: // DPOTRF
//             {   
//                 {
//                     // SmartTile experimental path (guarded by workspace availability)
//                     auto* tw = reinterpret_cast<sTiles::ThreadWorkspace*>(stile->thread_workspaces[STILES_RANK]);
//                     sTiles::sTile_workspace* W = tw->getWorkspace();
//                     sTiles::SmartTile* st = tiledMatrix->facTiles[index1];
//                     if (st) {
//                         //std::cout << "tile index is: " << index1 << std::endl;
//                         st->bindWorkspace(W);

                            
//                         //if(index1==2) st->printSymbolicSparse();
//                         st->symbolicChol_fillIn(sTiles::SymbolicMethod::Auto);
//                         //if(index1==2) std::cout << "Afffter" << std::endl;
//                         //if(index1==2) st->printSymbolicSparse();

//                         st->symbolicChol();
//                         //if(index1==2) exit(0);

//                     }
//                 }
//                 ss_cond_set(k, k, 1);
//                 break;
//             }
//             case 2: // DSYRK
//             {
//                 ss_cond_wait(k, n, 1);
//                 double* tile_in  = tiledMatrix->denseTiles[index1];
//                 double* tile_out = tiledMatrix->denseTiles[index2];

//                 // SmartTile mirror: predict -A^T*A into C_s (index2), then compare to dense delta (after - before)
                
//                 if (tiledMatrix->facTiles && stile && stile->thread_workspaces) {
//                     auto* A_st = tiledMatrix->facTiles[index1];
//                     auto* C_st = tiledMatrix->facTiles[index2];

//                     //if(index2==2) std::cout << " ---------------before--------------- case 2" << std::endl;
//                     //if(index2==2) C_st->printSymbolicSparse();

//                     //if(index2==4) std::cout << "hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh" << std::endl;
//                     auto* tw   = reinterpret_cast<sTiles::ThreadWorkspace*>(stile->thread_workspaces[STILES_RANK]);
//                     auto* W    = tw ? tw->getWorkspace() : nullptr;
//                     C_st->bindWorkspace(W);
//                     C_st->symbolicAtA(A_st, A_st, sTiles::SymbolicMethod::Auto);
                    
//                     // if(index2==2) std::cout << " ---------------after--------------- case 2" << std::endl;
//                     // if(index2==2) C_st->printSymbolicSparse();
//                     // if(index2==2) exit(0);


//                 }

//                 break;
//             }
//             case 3: // DTRSM
//             {
//                 ss_cond_wait(k, k, 1);
        
//                 // SmartTile solve to mirror dense TRSM
//                 if (tiledMatrix->facTiles && stile && stile->thread_workspaces) {
//                     auto* rhs  = tiledMatrix->facTiles[index1];
//                     auto* diag = tiledMatrix->facTiles[index2];
//                     //if(index1==4) std::cout << "hhhhhhhhhhhhhhhhhhhhhhhhhhhhh333333333333hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh" << std::endl;
//                     if (rhs && diag) {
//                         auto* tw = reinterpret_cast<sTiles::ThreadWorkspace*>(stile->thread_workspaces[STILES_RANK]);
//                         auto* W  = tw ? tw->getWorkspace() : nullptr;
//                         if (W) rhs->bindWorkspace(W);
//                         rhs->symbolicForSolve(diag, sTiles::SymbolicMethod::Auto);
//                     }

//                     //if(index2==2) std::cout << " ------------------------------ case 3" << std::endl;
//                     //if(index2==2) rhs->printSymbolicSparse();

//                 }

//                 ss_cond_set(m, k, 1);
//                 break;
//             }
//             case 4: // DGEMM
//             {
//                 ss_cond_wait(k, n, 1);
//                 ss_cond_wait(m, n, 1);
             
//                 // SmartTile mirror: C(index3) := C - A(index1)^T * B(index2) starting from zero (to match delta)
//                 if (tiledMatrix->facTiles && stile && stile->thread_workspaces) {
//                     auto* A_st = tiledMatrix->facTiles[index1];
//                     auto* B_st = tiledMatrix->facTiles[index2];
//                     auto* C_st = tiledMatrix->facTiles[index3];
//                     //if(index3==4) std::cout << "hhhhhhhhhhhhhhhhhhhhhhhhhhhhh4444444444444444444444hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh" << std::endl;

//                     auto* tw   = reinterpret_cast<sTiles::ThreadWorkspace*>(stile->thread_workspaces[STILES_RANK]);
//                     auto* W    = tw ? tw->getWorkspace() : nullptr;
//                     C_st->bindWorkspace(W);
//                     C_st->symbolicAtB(A_st, B_st, sTiles::SymbolicMethod::Auto);

//                     //if(index3==2) std::cout << " ------------------------------ case 3" << std::endl;
//                     //if(index3==2) C_st->printSymbolicSparse();

//                 }

//                 break;
//             }
//             default:
//                 sTiles::Logger::warning("│   [ESMAIL_CHECK] Unexpected routine id ", myroutine, " in expansion tasks");
//                 break;
//         }
//         if (ss_aborted()) break;
//     }

//     // After building symbolic structure for all referenced tiles, initialize
//     // SmartTile values from their original values once. Do this on rank 0 to
//     // avoid redundant multi-thread updates.
//     if (tiledMatrix->facTiles && STILES_RANK==0) {
//         const int world = 1;//std::max(1, STILES_SIZE);
//         const int rank  = 0;//std::max(0, STILES_RANK);

//         sTiles::sTile_workspace* W_local = nullptr;
//         if (stile && stile->thread_workspaces &&
//             stile->thread_workspaces_count > rank &&
//             stile->thread_workspaces[rank]) {
//             auto* tw_local = reinterpret_cast<sTiles::ThreadWorkspace*>(stile->thread_workspaces[rank]);
//             if (tw_local) W_local = tw_local->getWorkspace();
//         }

//         for (int t = rank; t < tiledMatrix->numActiveTiles; t += world) {
//             sTiles::SmartTile* st = tiledMatrix->facTiles[t];
//             if (!st) continue;

//             if (W_local) st->bindWorkspace(W_local);

//             auto* td = st->getTile();
//             if (!td) continue;

//             const bool have_original  = (td->original_colptr && td->original_rowind);
//             const bool have_structure = (td->colptr && td->rowind);
//             if (have_original && have_structure) {
//                 st->buildBitmapperFromOriginal();
//             }

//             st->allocateValues();

//             if (td->original_values || td->values) {
//                 st->updateValuesFromOriginal();
//             }
//         }
//     }
//         std::cout << "ending" << std::endl;


//     ss_finalize();

//     #undef BLKADD_CT
//     //exit(0);
// }


// void dpotrf_expansion_from_chol_tasks_smart_compare(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

//     const int world = STILES_SIZE;
//     const int rank  = STILES_RANK;
//     int win_AtB = 0;
//     //std::cout <<"start compute"<<std::endl;
//     const int N = tiledMatrix->dim;
//     const int tile_size = tiledMatrix->tile_size;
//     const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
//     const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
//     #define BLKADD_CT(k) ( (k) < full_tiles_per_dim ? tile_size : (N % tile_size) )

//     const auto &tasks = tiledMatrix->chol_tasks;
//     const auto &offsets = tiledMatrix->chol_task_offsets;
//     const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
//     const int end   = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

//     ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
//     for (int idx = start; idx < end; ++idx) {
//         const std::array<int,7> &t = tasks[idx];
//         const int myroutine = t[0];
//         const int m = t[1];
//         const int k = t[2];
//         const int n = t[3];
//         const int index1 = t[4];
//         const int index2 = t[5];
//         const int index3 = t[6];

//         const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
//         const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
//         const int ldak = BLKADD_CT(k);
//         const int ldan = BLKADD_CT(n);

//         const double zone  = 1.0;
//         const double mzone = -1.0;

//         switch (myroutine) {
//             case 1: // DPOTRF
//             {   
//                 //std::cout << "case1" << std::endl;
//                 double* tile_out = tiledMatrix->denseTiles[index1];
//                 if (!tile_out) sTiles::Logger::warning("│  null tile_out in potrf (index1=", index1, ")");

//                 auto* tw = (stile && stile->thread_workspaces && STILES_RANK < stile->thread_workspaces_count)
//                                ? reinterpret_cast<sTiles::ThreadWorkspace*>(stile->thread_workspaces[STILES_RANK])
//                                : nullptr;
//                 auto* W  = tw ? tw->getWorkspace() : nullptr;

//                 auto* A_st = tiledMatrix->facTiles[index1];




//                 // smart_debug_print_dense_tile("DPOTRF", index1, tile_out, tempkn, tempkn, ldak);
//                 // A_st->printAsDense();

//                 double t1 = omp_get_wtime();
//                 sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
//                 std::cout << "time of chol(naive): " << omp_get_wtime() - t1 << std::endl;
//                 (void)status;

//                 if (!A_st) {
//                     //std::cout << "[SMART-CHECK] FAIL: missing SmartTile for DPOTRF index " << index1 << std::endl;
//                     ss_abort();
//                     break;
//                 }



//                 if (W) A_st->bindWorkspace(W);

//                 double t0 = omp_get_wtime();
//                 A_st->numericChol();
//                 std::cout << "time of chol(smart): " << omp_get_wtime() - t0 << std::endl;


//                 // smart_debug_print_dense_tile("DPOTRF", index1, tile_out, tempkn, tempkn, ldak);
//                 // A_st->printAsDense();

//                 // std::cout << "[SMART-CHECK] Case DPOTRF (rank " << STILES_RANK
//                 //           << "): dense tile index=" << index1 << std::endl;
//                 bool ok_compare = true;
//                 if (!tile_out) {
//                     //std::cout << "[SMART-CHECK] FAIL: missing dense tile for DPOTRF index " << index1 << std::endl;
//                     ok_compare = false;
//                 } else if (!A_st->getTile()) {
//                     //std::cout << "[SMART-CHECK] FAIL: SmartTile has no data for DPOTRF index " << index1 << std::endl;
//                     ok_compare = false;
//                 } else {
//                     ok_compare = sTiles::debug::compare_smart_L_to_dense_U(A_st->getTile(), tile_out, ldak);
//                 }
//                 if (!ok_compare) {
//                     ss_abort();
//                     break;
//                 }
//                 std::cout << std::endl;
//                 ss_cond_set(k, k, 1);

//                 break;
//             }
//             case 2: // DSYRK
//             {

//                 ss_cond_wait(k, n, 1);
//                 double* tile_in  = tiledMatrix->denseTiles[index1];
//                 double* tile_out = tiledMatrix->denseTiles[index2];
//                 //std::cout << "case2" << std::endl;

//                 auto* tw = (stile && stile->thread_workspaces && STILES_RANK < stile->thread_workspaces_count)
//                                ? reinterpret_cast<sTiles::ThreadWorkspace*>(stile->thread_workspaces[STILES_RANK])
//                                : nullptr;
//                 auto* W  = tw ? tw->getWorkspace() : nullptr;

//                 auto* A_st = tiledMatrix->facTiles[index1];
//                 auto* C_st = tiledMatrix->facTiles[index2];
//                 if (!A_st || !C_st) {
//                     // std::cout << "[SMART-CHECK] FAIL: missing SmartTile(s) for DSYRK indices index1="
//                     //           << index1 << " index2=" << index2 << std::endl;
//                     ss_abort();
//                     break;
//                 }
//                 if (W) {
//                     A_st->bindWorkspace(W);
//                     C_st->bindWorkspace(W);
//                 }
//                 //smart_debug_print_dense_tile("DSYRK-input", index1, tile_in, ldan, tile_size, ldan);
                
//                 double t1 = omp_get_wtime();
//                 sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone,
//                                    tile_in, ldan, zone, tile_out, ldak);
//                 std::cout << "time of syrk(naive): " << omp_get_wtime() - t1 << std::endl;

//                 double t0 = omp_get_wtime();
//                 C_st->numericAtA(A_st, A_st);
//                 std::cout << "time of syrk(smart): " << omp_get_wtime() - t0 << std::endl;

//                 //smart_debug_print_dense_tile("DSYRK", index2, tile_out, tempkn, tempkn, ldak);

//                 // std::cout << "[SMART-CHECK] Case DSYRK (rank " << STILES_RANK
//                 //           << "): dense tile index2=" << index2
//                 //           << ", input tile index1=" << index1 << std::endl;
//                 bool ok_compare = true;
//                 if (!tile_out) {
//                     //std::cout << "[SMART-CHECK] FAIL: missing dense tile for DSYRK index " << index2 << std::endl;
//                     ok_compare = false;
//                 } else if (!C_st->getTile()) {
//                     //std::cout << "[SMART-CHECK] FAIL: SmartTile has no data for DSYRK index " << index2 << std::endl;
//                     ok_compare = false;
//                 } else {
//                     ok_compare = sTiles::debug::compare_smart_L_to_dense_U(C_st->getTile(), tile_out, ldak);
//                 }
//                 if (!ok_compare) {
//                     ss_abort();
//                     break;
//                 }
//                 break;
//             }
//             case 3: // DTRSM
//             {


//                 ss_cond_wait(k, k, 1);
//                 double* tile_rhs = tiledMatrix->denseTiles[index2];
//                 double* tile_out = tiledMatrix->denseTiles[index1];
//                 //std::cout << "case3" << std::endl;

//                 auto* tw = (stile && stile->thread_workspaces && STILES_RANK < stile->thread_workspaces_count)
//                                ? reinterpret_cast<sTiles::ThreadWorkspace*>(stile->thread_workspaces[STILES_RANK])
//                                : nullptr;
//                 auto* W  = tw ? tw->getWorkspace() : nullptr;

//                 // Print BEFORE (dense + smart) on rank 0
//                 // if (STILES_RANK == 0) {
//                 //     sTiles::debug::print_tile_dense("DTRSM before: tile_out (dense)", tile_out, ldak, tempmn, ldak, 6);
//                 //     sTiles::debug::print_tile_dense("DTRSM before: tile_rhs/diag (dense)", tile_rhs, ldak, ldak, ldak, 6);
//                 //     sTiles::SmartTile* rhs_st_before  = tiledMatrix->facTiles[index1];
//                 //     sTiles::SmartTile* diag_st_before = tiledMatrix->facTiles[index2];
//                 //     if (rhs_st_before && rhs_st_before->getTile())
//                 //         sTiles::debug::print_tile_smart("DTRSM before: rhs (smart)", rhs_st_before->getTile());
//                 //     if (diag_st_before && diag_st_before->getTile())
//                 //         sTiles::debug::print_tile_smart("DTRSM before: diag (smart)", diag_st_before->getTile());
//                 // }

//                 // // Compare inputs (Smart vs Dense) succinctly before modification (single rank)
//                 // if (tiledMatrix->facTiles && STILES_RANK == 0) {
//                 //     sTiles::debug::compare_trsm_inputs(
//                 //         tiledMatrix->facTiles[index1], tile_out, ldak, tempmn, ldak,
//                 //         tiledMatrix->facTiles[index2], tile_rhs, ldak,
//                 //         1e-8, 10);
//                 // }

//                 auto* rhs  = tiledMatrix->facTiles[index1];
//                 auto* diag = tiledMatrix->facTiles[index2];
//                 if (!rhs || !diag) {
//                     // std::cout << "[SMART-CHECK] FAIL: missing SmartTile(s) for DTRSM indices index1="
//                     //           << index1 << " index2=" << index2 << std::endl;
//                     ss_abort();
//                     break;
//                 }
//                 if (W) {
//                     rhs->bindWorkspace(W);
//                     diag->bindWorkspace(W);
//                 }

//                 double t0 = omp_get_wtime();
//                 sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit,
//                                    tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
//                 std::cout << "time of trsm(naive): " << omp_get_wtime() - t0 << std::endl;

//                 double t1 = omp_get_wtime();
//                 rhs->numericForSolve(diag);
//                 std::cout << "time of trsm(smart): " << omp_get_wtime() - t1 << std::endl;

//                 //smart_debug_print_dense_tile("DTRSM-diag", index2, tile_rhs, ldak, ldak, ldak);
//                 //smart_debug_print_dense_tile("DTRSM", index1, tile_out, ldak, tempmn, ldak);

//                 // std::cout << "[SMART-CHECK] Case DTRSM (rank " << STILES_RANK
//                 //           << "): output tile index1=" << index1
//                 //           << ", diag tile index2=" << index2 << std::endl;
//                 bool ok_compare = true;
//                 if (!tile_out) {
//                     //std::cout << "[SMART-CHECK] FAIL: missing dense tile for DTRSM index " << index1 << std::endl;
//                     ok_compare = false;
//                 } else if (!rhs->getTile()) {
//                     //std::cout << "[SMART-CHECK] FAIL: SmartTile has no data for DTRSM index " << index1 << std::endl;
//                     ok_compare = false;
//                 } else {
//                     ok_compare = sTiles::debug::compare_smart_values_to_dense_tile(rhs->getTile(), tile_out, ldak);
//                 }
//                 if (!ok_compare) {
//                     ss_abort();
//                     break;
//                 }

//                 ss_cond_set(m, k, 1);
//                 break;
//             }
//             case 4: // DGEMM
//             {

//                 ss_cond_wait(k, n, 1);
//                 ss_cond_wait(m, n, 1);
//                 double* tile_a   = tiledMatrix->denseTiles[index1];
//                 double* tile_b   = tiledMatrix->denseTiles[index2];
//                 double* tile_out = tiledMatrix->denseTiles[index3];
//                 //std::cout << "case4" << std::endl;

//                 bool ran_smart = false;
//                 sTiles::SmartTile* C_st = nullptr;

//                 double t0 = omp_get_wtime();
//                 sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone,
//                                    tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);
//                 double f_nsmart = omp_get_wtime() - t0;
//                 std::cout << "time of gemm(naive): " << f_nsmart << std::endl;

//                 // SmartTile mirror: C(index3) := C - A(index1)^T * B(index2)
//                 if (tiledMatrix->facTiles && stile && stile->thread_workspaces) {
//                     auto* A_st = tiledMatrix->facTiles[index1];
//                     auto* B_st = tiledMatrix->facTiles[index2];
//                     C_st = tiledMatrix->facTiles[index3];

//                     auto* tw = (STILES_RANK < stile->thread_workspaces_count)
//                                    ? reinterpret_cast<sTiles::ThreadWorkspace*>(stile->thread_workspaces[STILES_RANK])
//                                    : nullptr;
//                     auto* W  = tw ? tw->getWorkspace() : nullptr;
//                     if (W) {
//                         if (A_st) A_st->bindWorkspace(W);
//                         if (B_st) B_st->bindWorkspace(W);
//                         if (C_st) C_st->bindWorkspace(W);
//                     }

//                     if (A_st && B_st && C_st) {

//                         double t1 = omp_get_wtime();
//                         C_st->numericAtB(A_st, B_st);
//                         double f_smart = omp_get_wtime() - t1;
//                         std::cout << "time of gemm(smart): " << f_smart << std::endl;
//                         if(f_smart > f_nsmart) win_AtB++;
//                         ran_smart = true;
//                     } else {
//                         // std::cout << "[SMART-CHECK] FAIL: missing SmartTile(s) for DGEMM indices index1="
//                         //           << index1 << " index2=" << index2 << " index3=" << index3 << std::endl;
//                         ss_abort();
//                         break;
//                     }
//                 }

//                 //smart_debug_print_dense_tile("DGEMM-A", index1, tile_a, ldan, tile_size, ldan);
//                 //smart_debug_print_dense_tile("DGEMM-B", index2, tile_b, ldan, tempmn, ldan);
//                 // Dense update: C := -A^T*B + 1*C


//                 if (ran_smart) {
//                     //smart_debug_print_dense_tile("DGEMM", index3, tile_out, ldak, tempmn, ldak);

//                     // std::cout << "[SMART-CHECK] Case DGEMM (rank " << STILES_RANK
//                     //           << "): output tile index3=" << index3
//                     //           << ", A index1=" << index1
//                     //           << ", B index2=" << index2 << std::endl;
//                     bool ok_compare = true;
//                     if (!tile_out) {
//                         //std::cout << "[SMART-CHECK] FAIL: missing dense tile for DGEMM index " << index3 << std::endl;
//                         ok_compare = false;
//                     } else if (!C_st || !C_st->getTile()) {
//                         //std::cout << "[SMART-CHECK] FAIL: SmartTile has no data for DGEMM index " << index3 << std::endl;
//                         ok_compare = false;
//                     } else {
//                         ok_compare = sTiles::debug::compare_smart_values_to_dense_tile(C_st->getTile(), tile_out, ldak);
//                     }
//                     if (!ok_compare) {
//                         ss_abort();
//                         break;
//                     }
//                 }
//                 break;
//             }
//             default:
//                 sTiles::Logger::warning("│   [ESMAIL_CHECK] Unexpected routine id ", myroutine, " in expansion tasks");
//                 break;
//         }
//         if (ss_aborted()) break;
//     }
//     ss_finalize();

//     for (int t = 0; t < tiledMatrix->numActiveTiles; ++t) {
//         sTiles::SmartTile* st = tiledMatrix->facTiles[t];
//         if (!st) continue;
//         st->export_tile(t, "../checks/set11/results_new_matrix_tiling");
//         //st->printAsDense();
//     }
//     std::cout << "success with " << win_AtB << "wins" << std::endl;
    
//     //exit(0);

//     #undef BLKADD_CT
// }

// void numeric(TiledMatrix *tiledMatrix, stiles_context_t *stile) {

//     const int world = STILES_SIZE;
//     const int rank  = STILES_RANK;

//     //std::cout <<"start compute"<<std::endl;
//     const int N = tiledMatrix->dim;
//     const int tile_size = tiledMatrix->tile_size;
//     const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
//     const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
//     #define BLKADD_CT(k) ( (k) < full_tiles_per_dim ? tile_size : (N % tile_size) )

//     const auto &tasks = tiledMatrix->chol_tasks;
//     const auto &offsets = tiledMatrix->chol_task_offsets;
//     const int start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
//     const int end   = (rank + 1 < static_cast<int>(offsets.size())) ? offsets[rank + 1] : static_cast<int>(tasks.size());

//     ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
//     for (int idx = start; idx < end; ++idx) {
//         const std::array<int,7> &t = tasks[idx];
//         const int myroutine = t[0];
//         const int m = t[1];
//         const int k = t[2];
//         const int n = t[3];
//         const int index1 = t[4];
//         const int index2 = t[5];
//         const int index3 = t[6];

//         const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
//         const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
//         const int ldak = BLKADD_CT(k);
//         const int ldan = BLKADD_CT(n);

//         const double zone  = 1.0;
//         const double mzone = -1.0;

//         switch (myroutine) {
//             case 1: // DPOTRF
//             {   
//                 auto* A_st = tiledMatrix->facTiles[index1];
//                 A_st->numericChol();
//                 ss_cond_set(k, k, 1);
//                 break;
//             }
//             case 2: // DSYRK
//             {

//                 ss_cond_wait(k, n, 1);
//                 auto* A_st = tiledMatrix->facTiles[index1];
//                 auto* C_st = tiledMatrix->facTiles[index2];
//                 C_st->numericAtA(A_st, A_st);

//                 break;
//             }
//             case 3: // DTRSM
//             {


//                 ss_cond_wait(k, k, 1);

//                 auto* rhs  = tiledMatrix->facTiles[index1];
//                 auto* diag = tiledMatrix->facTiles[index2];
//                 rhs->numericForSolve(diag);
                
//                 ss_cond_set(m, k, 1);
//                 break;
//             }
//             case 4: // DGEMM
//             {

//                 ss_cond_wait(k, n, 1);
//                 ss_cond_wait(m, n, 1);

//                 if (tiledMatrix->facTiles && stile && stile->thread_workspaces && STILES_RANK == 0) {
//                     auto* A_st = tiledMatrix->facTiles[index1];
//                     auto* B_st = tiledMatrix->facTiles[index2];
//                     auto* C_st = tiledMatrix->facTiles[index3];
//                     C_st->numericAtB(A_st, B_st);
//                 }

//                 break;
//             }
//             default:
//                 sTiles::Logger::warning("│   [ESMAIL_CHECK] Unexpected routine id ", myroutine, " in expansion tasks");
//                 break;
//         }
//         if (ss_aborted()) break;
//     }
//     ss_finalize();

//     std::cout << "numeric_smart_success" << std::endl;

//     #undef BLKADD_CT
// }


// void pdpotrf(stiles_context_t *stile) {

//     TiledMatrix* tiledMatrix = nullptr;
//     sTiles::unpack_args(stile, tiledMatrix);

//     if (!tiledMatrix) {
//         std::cout << "Error: null tiledMatrix in stiles_pdpotrf" << std::endl;
//         return;
//     }

// #ifdef STILES_SAFEMODE
//     if (tiledMatrix->use_boosted_e_trick) {
//         if (tiledMatrix->red_tree_separator_level > 0) {
//             sTiles::SafeMode::dpotrf_reduction(tiledMatrix, stile);
//         } else {
//             sTiles::SafeMode::dpotrf_expansion(tiledMatrix, stile);
//         }
//     } else {
//         std::cout << "FIXME: SafeMode without boosted_e_trick is not implemented." << std::endl;
//     }
// #elif STILES_STABLEMODE
//     if (tiledMatrix->use_boosted_e_trick) {
//         if (tiledMatrix->red_tree_separator_level > 0) {
//             sTiles::Process::dpotrf_reduction(tiledMatrix, stile);
//         } else {
//             sTiles::Process::dpotrf_expansion(tiledMatrix, stile);
//         }
//     } else {
//         std::cout << "FIXME: SafeMode without boosted_e_trick is not implemented." << std::endl;
//     }
// #elif STILES_DEVELMODE

//     #ifdef STILES_SMART_COMPARE

//         symbolic(tiledMatrix, stile);
        
//         if(true){

//             dpotrf_expansion_from_chol_tasks_smart_compare(tiledMatrix, stile);
//             exit(0);

//         }else{

//             double t0 = omp_get_wtime();
//             dpotrf_expansion_from_chol_tasks(tiledMatrix, stile);
//             double t1 = omp_get_wtime();

//             double t2 = omp_get_wtime();
//             numeric(tiledMatrix, stile);
//             double t3 = omp_get_wtime();

//             if(STILES_RANK==0){

//                 std::printf("naive: %.3f s\n", (t1 - t0));
//                 std::printf("smart: %.3f s\n", (t3 - t2));
//                 exit(0);

//             }

//         }




//     #else
//         if (tiledMatrix->red_tree_separator_level > 0) {
//             #ifdef ESMAIL_CHECK
//                 dpotrf_reduction(tiledMatrix, stile);
//             #else
//                 dpotrf_reduction_from_chol_tasks(tiledMatrix, stile);
//             #endif
//         } else {
//             #ifdef ESMAIL_CHECK
//                 dpotrf_expansion(tiledMatrix, stile);
//             #else
//                 dpotrf_expansion_from_chol_tasks(tiledMatrix, stile);

//                 // symbolic(tiledMatrix, stile);
//                 // numeric(tiledMatrix, stile);

//             #endif
//         }
//     #endif
// #endif

// #ifdef STILES_DUMP_TILES
//     const std::string dump_path = "tools/compute/debug_dumps/pdpotrf_n" +
//                                   std::to_string(tiledMatrix->dim) + "_ts" +
//                                   std::to_string(tiledMatrix->tile_size) + "_rank" +
//                                   std::to_string(STILES_RANK) + ".txt";
//     sTiles::debug::export_dense_tiles(tiledMatrix, dump_path);
// #endif

// }

// }} // namespace name


// // void dpotrf_expansion_original(TiledMatrix *tiledMatrix, stiles_context_t *stile){

// //     const int rank = STILES_RANK;
// //     const int N = tiledMatrix->dim;
// //     const int tile_size = tiledMatrix->tile_size;
// //     const int num_tiles_per_dim = (N == 0) ? 0 : (N-1)/tile_size + 1;
// //     const int full_tiles_per_dim = N/tile_size;
// //     #define BLKADD(k) ( (k) < full_tiles_per_dim ? tile_size : N % tile_size )

// //     int k, m, n;
// //     int index1, index2, index3, myroutine;
// //     int ldak, ldan;
// //     int tempkn, tempmn;
// //     const double zone  = (double) 1.0;
// //     const double mzone = (double)-1.0;
// //     const int num_tasks = tiledMatrix->e_trick_size[rank];

// //     ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
// //     for(int i=0; i<num_tasks;i++){

// //         myroutine = tiledMatrix->e_trick[rank][0+7*i];
// //         m = tiledMatrix->e_trick[rank][1+i*7];
// //         k = tiledMatrix->e_trick[rank][2+i*7];
// //         n = tiledMatrix->e_trick[rank][3+i*7];
// //         index1 = tiledMatrix->e_trick[rank][4+i*7];
// //         index2 = tiledMatrix->e_trick[rank][5+i*7];
// //         index3 = tiledMatrix->e_trick[rank][6+i*7];

// //         tempkn = k == (num_tiles_per_dim-1) ? N-k*tile_size : tile_size;
// //         tempmn = m == (num_tiles_per_dim-1) ? N-m*tile_size : tile_size;
// //         ldak = BLKADD(k);
// //         ldan = BLKADD(n);

// //         switch (myroutine) {
// //             case 1:
// //             {

// //                 double* tile_out = tiledMatrix->dense_tiles[index1].elements;
// //                 sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);

// //                 double* tile_out1 = tiledMatrix->denseTiles[index1];
// //                 sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out1, ldak);
// //                 if (status != sTiles::StatusCode::Success) std::cout << "fix me" << std::endl;


// //                 ss_cond_set(k, k, 1);
// //                 break;
// //             }
        
// //             case 2: // DSYRK
// //             {
// //                 ss_cond_wait(k, n, 1);
// //                 double* tile_in = tiledMatrix->dense_tiles[index1].elements;
// //                 double* tile_out = tiledMatrix->dense_tiles[index2].elements;
// //                 if (!tile_in || !tile_out) {
// //                     std::cout << "fix me" << std::endl;
// //                     break;
// //                 }
// //                 sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone,
// //                                    tile_in, ldan, zone, tile_out, ldak);

// //                 double* tile_in1 = tiledMatrix->denseTiles[index1];
// //                 double* tile_out1 = tiledMatrix->denseTiles[index2];
// //                 sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone,
// //                                    tile_in1, ldan, zone, tile_out1, ldak);



// //                 break;
// //             }

// //             case 3: // DTRSM
// //             {
// //                 ss_cond_wait(k, k, 1);
// //                 double* tile_rhs = tiledMatrix->dense_tiles[index2].elements;
// //                 double* tile_out = tiledMatrix->dense_tiles[index1].elements;
// //                 sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit,
// //                                    tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);
                
// //                 double* tile_rhs1 = tiledMatrix->denseTiles[index2];
// //                 double* tile_out1 = tiledMatrix->denseTiles[index1];
// //                 sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit,
// //                                    tile_size, tempmn, zone, tile_rhs1, ldak, tile_out1, ldak);

// //                 ss_cond_set(m, k, 1);
// //                 break;
// //             }
            
// //             case 4: // DGEMM
// //             {
// //                 ss_cond_wait(k, n, 1);
// //                 ss_cond_wait(m, n, 1);
// //                 double* tile_a = tiledMatrix->dense_tiles[index1].elements;
// //                 double* tile_b = tiledMatrix->dense_tiles[index2].elements;
// //                 double* tile_out = tiledMatrix->dense_tiles[index3].elements;
// //                 sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone,
// //                                    tile_a, ldan,
// //                                    tile_b, ldan, zone, tile_out, ldak);

// //                 double* tile_a1 = tiledMatrix->denseTiles[index1];
// //                 double* tile_b1 = tiledMatrix->denseTiles[index2];
// //                 double* tile_out1 = tiledMatrix->denseTiles[index3];
// //                 sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone,
// //                                    tile_a1, ldan,
// //                                    tile_b1, ldan, zone, tile_out1, ldak);
                                   
// //                 break;
// //             }
// //         }
        
// //         #ifdef COMPUTE_FLOPS
// //             if (myroutine >= 1 && myroutine <= 4) {
// //                  tiledMatrix->flops_mat[(rank * 5) + (myroutine - 1)] += 1;
// //             }
// //         #endif

// //         if(ss_aborted()) break;
// //     }
// //     ss_finalize();

// // #undef BLKADD
// // }
