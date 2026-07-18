/**
 * @file pdpotrf_symbolic.cpp
 * @brief Parallel symbolic Cholesky factorization implementation.
 *
 * Implements the parallel symbolic factorization phase using OpenMP for
 * multi-threaded execution. Computes sparsity patterns and fill-in structure
 * for tiled sparse Cholesky decomposition.
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
#include <cstdlib>
#include <new>
#include <mutex>

#include "../common/core_lapack.hpp"
// Per-thread scratch for SmartTile and dense conversions
#include "../memory/stile_threadWorkspace.hpp"
#include "../memory/workspace.hpp"

#include "../common/stiles_logger.hpp"
#include "../control/stiles_control.hpp"
#include "../compute/stiles_compute.hpp"
#include "../control/common.h" // ss_* macros
#include "../tile/meta.hpp"
#include "core_semisparse_kernels.hpp"
#include "core_sparse_kernels.hpp"

#include "../common/stiles_utils.hpp"
#include "../compute/debug_tiles.hpp"
#include "../compute/tile_compare.hpp"
#ifdef SMART_TILES
// SmartTile CHOL compute (symbolic + numeric)
#endif

// Forward declaration of global control parameter accessor
extern "C" int* sTiles_get_params();

namespace sTiles { namespace preprocess {

#ifdef SPARSE_STILES
#endif // SPARSE_STILES

    





    void dpotrf_expansion_from_chol_tasks_symbolic_semisparse_no_debug(TiledMatrix *tiledMatrix, stiles_context_t *stile){

        //std::cout << " STILES_SIZE " << STILES_SIZE << std::endl;
        const int bw_mode = sTiles_get_params()[sTiles::param::BandwidthMode];
        const int rank = STILES_RANK;
        const int world = STILES_SIZE;

        if (world <= 0) {
            sTiles::Logger::error("[symbolic_semisparse] Invalid STILES_SIZE=", world);
            std::exit(EXIT_FAILURE);
        }

        if (!tiledMatrix->workspaces) {
            static std::mutex workspace_init_mutex;
            std::lock_guard<std::mutex> lock(workspace_init_mutex);
            if (!tiledMatrix->workspaces) {
                std::size_t slots = static_cast<std::size_t>(world);
                sTiles::Workspace** tmp = static_cast<sTiles::Workspace**>(std::calloc(slots, sizeof(sTiles::Workspace*)));
                if (!tmp) {
                    sTiles::Logger::error("[symbolic_semisparse] Failed to allocate workspace slots (world=", world, ")");
                    std::exit(EXIT_FAILURE);
                }
                tiledMatrix->workspaces = tmp;
                tiledMatrix->num_workspaces = world;  // Track the workspace count
            }
        }
        
         sTiles::Control::Barrier(stile);
         
        if (rank >= world) {
            sTiles::Logger::error("[symbolic_semisparse] Rank ", rank, " exceeds workspace slots ", world);
            std::exit(EXIT_FAILURE);
        }

        if (!tiledMatrix->workspaces[rank]) {
            const int workspace_dim = (tiledMatrix->tile_size > 0) ? tiledMatrix->tile_size : 1;
            try {
                tiledMatrix->workspaces[rank] = new sTiles::Workspace(/*group_id*/0, workspace_dim);
            } catch (const std::bad_alloc&) {
                sTiles::Logger::error("[symbolic_semisparse] Failed to allocate workspace for rank ", rank);
                std::exit(EXIT_FAILURE);
            }
        }

        const int N = tiledMatrix->dim;
        const int tile_size = tiledMatrix->tile_size;
        const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
        const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
        #define BLKADD_CT(k) ( (k) < full_tiles_per_dim ? tile_size : (N % tile_size) )

        const auto &tasks   = sTiles::get_chol_tasks(tiledMatrix);
        const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
        const long long start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
        const long long end   = (rank + 1 < static_cast<int>(offsets.size()))
                            ? offsets[rank + 1]
                            : static_cast<int>(tasks.size());

        ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
        // std::cout << " start " << start << std::endl;
        // std::cout << " end " << end << std::endl;

        for (long long idx = start; idx < end; ++idx) {
            const std::array<int,7> &t = tasks[idx];
            const int myroutine = t[0];
            const int m        = t[1];
            const int k        = t[2];
            const int n        = t[3];
            const int index1   = t[4];
            const int index2   = t[5];
            const int index3   = t[6];

            const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
            const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
            const int ldak   = BLKADD_CT(k);
            const int ldan   = BLKADD_CT(n);

            (void)tempkn;
            (void)tempmn;
            (void)ldak;
            (void)ldan;

            switch (myroutine) {
                case 1:  // POTRF symbolic
                {
                    SemisparseTileMetaCore &Ameta = tiledMatrix->semisparseTileMetaCore[index1];

                    sTiles::core_sspotrf(Ameta);

                    ss_cond_set(k, k, 1);
                    break;
                }

                case 2:  // SYRK symbolic: A(index2) = A(index2) - B(index1)^T * B(index1)
                {
                    ss_cond_wait(k, n, 1);

                    SemisparseTileMetaCore       &Ameta = tiledMatrix->semisparseTileMetaCore[index2];
                    const SemisparseTileMetaCore &Bmeta = tiledMatrix->semisparseTileMetaCore[index1];

                    sTiles::core_ssdsyrk(Ameta, Bmeta, bw_mode);
                    break;
                }

                case 3:  // TRSM symbolic (no fill in, just propagate state if you ever need)
                {
                    ss_cond_wait(k, k, 1);

                    SemisparseTileMetaCore &Ameta = tiledMatrix->semisparseTileMetaCore[index1];
                    const SemisparseTileMetaCore &Bmeta = tiledMatrix->semisparseTileMetaCore[index2];

                    sTiles::core_ssdtrsm(Ameta, Bmeta);

                    ss_cond_set(m, k, 1);
                    break;
                }

                case 4:  // GEMM symbolic: C(index3) = C(index3) - A(index1)^T * B(index2)
                {
                    ss_cond_wait(k, n, 1);
                    ss_cond_wait(m, n, 1);

                    // Match numeric mapping:
                    //   A = index1
                    //   B = index2
                    //   C = index3

                    if (index3 < 0) break;  // C tile doesn't exist (no fill-in at this position), skip

                    const SemisparseTileMetaCore &Ameta = tiledMatrix->semisparseTileMetaCore[index1];
                    const SemisparseTileMetaCore &Bmeta = tiledMatrix->semisparseTileMetaCore[index2];

                    SemisparseTileMetaCore &Cmeta = tiledMatrix->semisparseTileMetaCore[index3];

                    sTiles::core_ssdgemm(Ameta, Bmeta, Cmeta);
                    break;
                }


                default:
                    sTiles::Logger::warning("│   [ESMAIL_CHECK] Unexpected routine id ",
                                            myroutine, " in symbolic expansion tasks");
                    break;
            }

            if (ss_aborted()) break;
        }

        ss_finalize();
        #undef BLKADD_CT
    }

    void dpotrf_expansion_from_chol_tasks_symbolic_semisparse(TiledMatrix *tiledMatrix, stiles_context_t *stile){

        //std::cout << " STILES_SIZE " << STILES_SIZE << std::endl;
        const int bw_mode = sTiles_get_params()[sTiles::param::BandwidthMode];
        if (STILES_RANK == 0) {
            //sTiles::Logger::errorf("[symbolic_semisparse] bw_mode=%d (%s)",
            //             bw_mode, bw_mode == 1 ? "tight" : "conservative");
        }
        const int rank = STILES_RANK;
        const int world = STILES_SIZE;

        if (world <= 0) {
            sTiles::Logger::error("[symbolic_semisparse] Invalid STILES_SIZE=", world);
            std::exit(EXIT_FAILURE);
        }

        if (!tiledMatrix->workspaces) {
            static std::mutex workspace_init_mutex;
            std::lock_guard<std::mutex> lock(workspace_init_mutex);
            if (!tiledMatrix->workspaces) {
                std::size_t slots = static_cast<std::size_t>(world);
                sTiles::Workspace** tmp = static_cast<sTiles::Workspace**>(std::calloc(slots, sizeof(sTiles::Workspace*)));
                if (!tmp) {
                    sTiles::Logger::error("[symbolic_semisparse] Failed to allocate workspace slots (world=", world, ")");
                    std::exit(EXIT_FAILURE);
                }
                tiledMatrix->workspaces = tmp;
                tiledMatrix->num_workspaces = world;  // Track the workspace count
            }
        }
        
         sTiles::Control::Barrier(stile);
         
        if (rank >= world) {
            sTiles::Logger::error("[symbolic_semisparse] Rank ", rank, " exceeds workspace slots ", world);
            std::exit(EXIT_FAILURE);
        }

        if (!tiledMatrix->workspaces[rank]) {
            const int workspace_dim = (tiledMatrix->tile_size > 0) ? tiledMatrix->tile_size : 1;
            try {
                tiledMatrix->workspaces[rank] = new sTiles::Workspace(/*group_id*/0, workspace_dim);
            } catch (const std::bad_alloc&) {
                sTiles::Logger::error("[symbolic_semisparse] Failed to allocate workspace for rank ", rank);
                std::exit(EXIT_FAILURE);
            }
        }

        const int N = tiledMatrix->dim;
        const int tile_size = tiledMatrix->tile_size;
        const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
        const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
        #define BLKADD_CT(k) ( (k) < full_tiles_per_dim ? tile_size : (N % tile_size) )

        const auto &tasks   = sTiles::get_chol_tasks(tiledMatrix);
        const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
        const long long start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
        const long long end   = (rank + 1 < static_cast<int>(offsets.size()))
                            ? offsets[rank + 1]
                            : static_cast<int>(tasks.size());

        ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
        // std::cout << " start " << start << std::endl;
        // std::cout << " end " << end << std::endl;

        for (long long idx = start; idx < end; ++idx) {
            const std::array<int,7> &t = tasks[idx];
            const int myroutine = t[0];
            const int m        = t[1];
            const int k        = t[2];
            const int n        = t[3];
            const int index1   = t[4];
            const int index2   = t[5];
            const int index3   = t[6];

            const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
            const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
            const int ldak   = BLKADD_CT(k);
            const int ldan   = BLKADD_CT(n);

            (void)tempkn;
            (void)tempmn;
            (void)ldak;
            (void)ldan;

            // // DEBUG: mutex for synchronized printing
            // static std::mutex debug_print_mutex;

            // // DEBUG helper lambda to print tile state
            // auto print_tile_state = [](const char* label, int tile_idx, const SemisparseTileMetaCore& meta) {
            //     sTiles::Logger::errorf("  %s Tile[%d]: fa=%d, la=%d, sa=%d, upper_bw=%d",
            //                  label, tile_idx, meta.fa, meta.la, meta.sa, meta.upper_bw);
            //     sTiles::Logger::errorf("    acol[%zu]: ", meta.acol.size());
            //     for (std::size_t c = 0; c < meta.acol.size() && c < 15; ++c) {
            //         sTiles::Logger::errorf("%d ", meta.acol[c]);
            //     }
            //     if (meta.acol.size() > 15) sTiles::Logger::errorf("...");
            //     sTiles::Logger::errorf("");
            //     sTiles::Logger::errorf("    aind[%zu]: ", meta.aind.size());
            //     for (std::size_t i = 0; i < meta.aind.size() && i < 15; ++i) {
            //         sTiles::Logger::errorf("%d ", meta.aind[i]);
            //     }
            //     if (meta.aind.size() > 15) sTiles::Logger::errorf("...");
            //     sTiles::Logger::errorf("");
            // };

            switch (myroutine) {
                case 1:  // POTRF symbolic
                {
                    SemisparseTileMetaCore &Ameta = tiledMatrix->semisparseTileMetaCore[index1];

                    // {
                    //     std::lock_guard<std::mutex> lock(debug_print_mutex);
                    //     sTiles::Logger::errorf("\n--- Task[%d] POTRF: tile=%d (k=%d) ---", idx, index1, k);
                    //     print_tile_state("BEFORE", index1, Ameta);
                    // }

                    sTiles::core_sspotrf(Ameta);

                    // {
                    //     std::lock_guard<std::mutex> lock(debug_print_mutex);
                    //     print_tile_state("AFTER ", index1, Ameta);
                    // }

                    ss_cond_set(k, k, 1);
                    break;
                }

                case 2:  // SYRK symbolic: A(index2) = A(index2) - B(index1)^T * B(index1)
                {
                    ss_cond_wait(k, n, 1);

                    SemisparseTileMetaCore       &Ameta = tiledMatrix->semisparseTileMetaCore[index2];
                    const SemisparseTileMetaCore &Bmeta = tiledMatrix->semisparseTileMetaCore[index1];

                    // {
                    //     std::lock_guard<std::mutex> lock(debug_print_mutex);
                    //     sTiles::Logger::errorf("\n--- Task[%d] SYRK: A=%d, B=%d (m=%d,k=%d,n=%d) ---", idx, index2, index1, m, k, n);
                    //     print_tile_state("BEFORE A", index2, Ameta);
                    //     print_tile_state("INPUT  B", index1, Bmeta);
                    // }

                    sTiles::core_ssdsyrk(Ameta, Bmeta, bw_mode);

                    // {
                    //     std::lock_guard<std::mutex> lock(debug_print_mutex);
                    //     print_tile_state("AFTER  A", index2, Ameta);
                    // }
                    break;
                }

                case 3:  // TRSM symbolic (no fill in, just propagate state if you ever need)
                {
                    ss_cond_wait(k, k, 1);

                    SemisparseTileMetaCore &Ameta = tiledMatrix->semisparseTileMetaCore[index1];
                    const SemisparseTileMetaCore &Bmeta = tiledMatrix->semisparseTileMetaCore[index2];

                    // {
                    //     std::lock_guard<std::mutex> lock(debug_print_mutex);
                    //     sTiles::Logger::errorf("\n--- Task[%d] TRSM: A=%d, B=%d (m=%d,k=%d) ---", idx, index1, index2, m, k);
                    //     print_tile_state("BEFORE A", index1, Ameta);
                    //     print_tile_state("INPUT  B", index2, Bmeta);
                    // }

                    sTiles::core_ssdtrsm(Ameta, Bmeta);

                    // {
                    //     std::lock_guard<std::mutex> lock(debug_print_mutex);
                    //     print_tile_state("AFTER  A", index1, Ameta);
                    // }

                    ss_cond_set(m, k, 1);
                    break;
                }

                case 4:  // GEMM symbolic: C(index3) = C(index3) - A(index1)^T * B(index2)
                {
                    ss_cond_wait(k, n, 1);
                    ss_cond_wait(m, n, 1);

                    // Match numeric mapping:
                    //   A = index1
                    //   B = index2
                    //   C = index3

                    if (index3 < 0) break;  // C tile doesn't exist (no fill-in at this position), skip

                    const SemisparseTileMetaCore &Ameta = tiledMatrix->semisparseTileMetaCore[index1];
                    const SemisparseTileMetaCore &Bmeta = tiledMatrix->semisparseTileMetaCore[index2];

                    SemisparseTileMetaCore &Cmeta = tiledMatrix->semisparseTileMetaCore[index3];

                    // {
                    //     std::lock_guard<std::mutex> lock(debug_print_mutex);
                    //     sTiles::Logger::errorf("\n--- Task[%d] GEMM: A=%d, B=%d, C=%d (m=%d,k=%d,n=%d) ---", idx, index1, index2, index3, m, k, n);
                    //     print_tile_state("INPUT  A", index1, Ameta);
                    //     print_tile_state("INPUT  B", index2, Bmeta);
                    //     print_tile_state("BEFORE C", index3, Cmeta);
                    // }

                    sTiles::core_ssdgemm(Ameta, Bmeta, Cmeta);

                    // {
                    //     std::lock_guard<std::mutex> lock(debug_print_mutex);
                    //     print_tile_state("AFTER  C", index3, Cmeta);
                    // }
                    break;
                }


                default:
                    sTiles::Logger::warning("│   [ESMAIL_CHECK] Unexpected routine id ",
                                            myroutine, " in symbolic expansion tasks");
                    break;
            }

            if (ss_aborted()) break;
        }

        ss_finalize();

        sTiles::Control::Barrier(stile);
        if (STILES_RANK == 0) {
            int count_zero = 0, count_nonzero = 0;
            int max_bw = 0, sum_bw = 0;
            const int num_active = tiledMatrix->numActiveTiles;
            //sTiles::Logger::errorf("[symbolic_semisparse] diagonal tile upper_bw (bw_mode=%d, active=%d):", bw_mode, num_active);
            for (int idx = 0; idx < num_active; ++idx) {
                const auto &meta = tiledMatrix->tileMetaCore[idx];
                if (meta.row != meta.col) continue;
                const auto &s = tiledMatrix->semisparseTileMetaCore[idx];
                const int bw = s.upper_bw;
                if (bw == 0) ++count_zero; else ++count_nonzero;
                if (bw > max_bw) max_bw = bw;
                sum_bw += bw;
            }
            // sTiles::Logger::errorf("  summary: %d diag tiles with bw>0, %d with bw=0, max_bw=%d, avg_bw=%.1f",
            //              count_nonzero, count_zero, max_bw,
            //              (count_nonzero + count_zero > 0) ? (double)sum_bw / (count_nonzero + count_zero) : 0.0);
        }

        #undef BLKADD_CT
    }


    void dpotrf_expansion_from_chol_tasks_symbolic_sparse(TiledMatrix *tiledMatrix, stiles_context_t *stile){

        //std::cout << " STILES_SIZE " << STILES_SIZE << std::endl;
        const int rank = STILES_RANK;
        const int N = tiledMatrix->dim;
        const int tile_size = tiledMatrix->tile_size;
        const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;
        const int full_tiles_per_dim = (tile_size > 0) ? (N / tile_size) : 0;
        #define BLKADD_CT(k) ( (k) < full_tiles_per_dim ? tile_size : (N % tile_size) )

        TileMetaCore *tileMeta = tiledMatrix->tileMetaCore;
        SemisparseTileMetaCore *semiMeta = tiledMatrix->semisparseTileMetaCore;
        SymbolicTileBitmaskCore *symbolicBits = tiledMatrix->symbolicTileBitmaskCore;
        SparseTileMetaCore *sparseMeta = tiledMatrix->sparseTileMetaCore;

        if (!tileMeta || !semiMeta || !symbolicBits) {
            sTiles::Logger::error("[symbolic_sparse] Missing tile metadata/bitmasks for symbolic expansion.");
            return;
        }

        const auto &tasks   = sTiles::get_chol_tasks(tiledMatrix);
        const auto &offsets = sTiles::get_chol_task_offsets(tiledMatrix);
        const long long start = (rank < static_cast<int>(offsets.size())) ? offsets[rank] : 0;
        const long long end   = (rank + 1 < static_cast<int>(offsets.size()))
                            ? offsets[rank + 1]
                            : static_cast<int>(tasks.size());

        ss_init(num_tiles_per_dim, num_tiles_per_dim, 0);
        // std::cout << " start " << start << std::endl;
        // std::cout << " end " << end << std::endl;

        for (long long idx = start; idx < end; ++idx) {
            const std::array<int,7> &t = tasks[idx];
            const int myroutine = t[0];
            const int m        = t[1];
            const int k        = t[2];
            const int n        = t[3];
            const int index1   = t[4];
            const int index2   = t[5];
            const int index3   = t[6];

            const int tempkn = (k == (num_tiles_per_dim - 1)) ? (N - k * tile_size) : tile_size;
            const int tempmn = (m == (num_tiles_per_dim - 1)) ? (N - m * tile_size) : tile_size;
            const int ldak   = BLKADD_CT(k);
            const int ldan   = BLKADD_CT(n);

            (void)tempkn;
            (void)tempmn;
            (void)ldak;
            (void)ldan;

            switch (myroutine) {
                case 1:  // POTRF symbolic
                {
                    TileMetaCore &Atile = tileMeta[index1];
                    SemisparseTileMetaCore &Ameta = semiMeta[index1];
                    SymbolicTileBitmaskCore &Abits = symbolicBits[index1];
                    sTiles::core_spotrf(sTiles::Uplo::Upper, Atile, Ameta, Abits);

                    ss_cond_set(k, k, 1);
                    break;
                }

                case 2:  // SYRK symbolic: A(index2) = A(index2) - B(index1)^T * B(index1)
                {
                    ss_cond_wait(k, n, 1);

                    SemisparseTileMetaCore       &Ameta = semiMeta[index2];
                    const SemisparseTileMetaCore &Bmeta = semiMeta[index1];
                    SymbolicTileBitmaskCore      &Abits = symbolicBits[index2];
                    const SymbolicTileBitmaskCore &Bbits = symbolicBits[index1];

                    sTiles::core_ssyrk(sTiles::Uplo::Upper, Ameta, Bmeta, Abits, Bbits);
                    break;
                }

                case 3:  // TRSM symbolic (no fill in, just propagate state if you ever need)
                {
                    ss_cond_wait(k, k, 1);

                    const TileMetaCore           &Atile = tileMeta[index2];
                    const SemisparseTileMetaCore &Ameta = semiMeta[index2];
                    SemisparseTileMetaCore       &Bmeta = semiMeta[index1];
                    SymbolicTileBitmaskCore      &Bbits = symbolicBits[index1];

                    const SymbolicTileBitmaskCore &Abits_diag = symbolicBits[index2];
                    sTiles::core_strsm(sTiles::Op::Trans, Atile, Ameta, Abits_diag, Bmeta, Bbits);

                    ss_cond_set(m, k, 1);
                    break;
                }

                case 4:  // GEMM symbolic: C(index3) = C(index3) - A(index1)^T * B(index2)
                {
                    ss_cond_wait(k, n, 1);
                    ss_cond_wait(m, n, 1);

                    // Match numeric mapping:
                    //   A = index1
                    //   B = index2
                    //   C = index3

                    if (index3 < 0) break;  // C tile doesn't exist (no fill-in at this position), skip

                    const SemisparseTileMetaCore &Ameta = semiMeta[index1];
                    const SemisparseTileMetaCore &Bmeta = semiMeta[index2];

                    SemisparseTileMetaCore &Cmeta = semiMeta[index3];

                    const SymbolicTileBitmaskCore &Abits = symbolicBits[index1];
                    const SymbolicTileBitmaskCore &Bbits = symbolicBits[index2];
                    SymbolicTileBitmaskCore      &Cbits = symbolicBits[index3];

                    sTiles::core_sgemm(Ameta, Bmeta, Cmeta, Abits, Bbits, Cbits);
                    break;
                }


                default:
                    sTiles::Logger::warning("│   [ESMAIL_CHECK] Unexpected routine id ",
                                            myroutine, " in symbolic expansion tasks");
                    break;
            }

            if (ss_aborted()) break;
        }

        ss_finalize();
        #undef BLKADD_CT
    }

    void symbolic_pdpotrf_semisparse(stiles_context_t *stile) {

        TiledMatrix* tiledMatrix = nullptr;
        sTiles::unpack_args(stile, tiledMatrix);

        if (!tiledMatrix) {
            sTiles::Logger::error("null tiledMatrix in symbolic pdpotrf");
            return;
        }

        if (tiledMatrix->use_boosted_e_trick) {
            if (tiledMatrix->red_tree_separator_level > 0) {
                sTiles::Logger::error("SafeMode symbolic tree-reduction path is not implemented.");
                exit(0);
            } else {
                sTiles::preprocess::dpotrf_expansion_from_chol_tasks_symbolic_semisparse(tiledMatrix, stile);
            }
        } else {
            sTiles::Logger::error("SafeMode without boosted_e_trick is not implemented.");
        }

    }

    void symbolic_pdpotrf_sparse(stiles_context_t *stile) {

        TiledMatrix* tiledMatrix = nullptr;
        sTiles::unpack_args(stile, tiledMatrix);

        if (!tiledMatrix) {
            sTiles::Logger::error("null tiledMatrix in symbolic pdpotrf");
            return;
        }

        if (tiledMatrix->use_boosted_e_trick) {
            if (tiledMatrix->red_tree_separator_level > 0) {
                sTiles::Logger::error("SafeMode symbolic tree-reduction path is not implemented.");
                exit(0);
            } else {
                sTiles::preprocess::dpotrf_expansion_from_chol_tasks_symbolic_sparse(tiledMatrix, stile);
            }
        } else {
            sTiles::Logger::error("SafeMode without boosted_e_trick is not implemented.");
        }

    }

} // namespace Process
} // namespace sTiles
