/**
 * @file dpotrf_symbolic.cpp
 * @brief Symbolic Cholesky factorization driver.
 *
 * Implements the symbolic phase of Cholesky factorization for sparse matrices,
 * computing fill-in patterns and allocating structures for the numeric phase.
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

#include <stiles_process.h>
#include "../common/stiles_types.hpp"
#include "../common/stiles_logger.hpp"
#include "../control/common.h"
#include "../compute/stiles_compute.hpp"
#include "../tile/meta.hpp"
#include "../memory/TileMemoryManager.hpp"
#include "symbolic_semisparse.hpp"
#include "../tile/sparse_dense_tiling.hpp"

// Forward declaration of global control parameter accessor
extern "C" int* sTiles_get_params();

namespace sTiles {
namespace preprocess {
    void symbolic_pdpotrf_semisparse(stiles_context_t *stile);
    void symbolic_pdpotrf_sparse(stiles_context_t *stile);
}
}

#ifdef STILES_GPU
    #include "../compute/memory_for_compute.hpp"
#endif

#include <algorithm>
#include <cstdlib>
#include <iostream>
#ifdef STILES_COLLECT_CHOL_TILES
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include "../tile/meta.hpp"   // TileMetaCore
#endif

extern "C" int* sTiles_get_params();
namespace {

inline bool _validate_scheme(const TiledMatrix* scheme) {
    if (!scheme) {
        sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "TiledMatrix scheme cannot be null.");
        return false;
    }
    if (scheme->dim <= 0) {
        sTiles::Logger::info("│   [symbolic] Skipping factorization for N=", scheme->dim, ".");
        return false;
    }
    return true;
}

} // namespace

namespace sTiles { namespace preprocess {

    static sTiles::StatusCode pthreads_semi_sparse_symbolic_dpotrf(int bind_index, TiledMatrix *scheme) {

        if (!_validate_scheme(scheme)) {
            return scheme ? sTiles::StatusCode::Success : sTiles::StatusCode::IllegalValue;
        }

        stiles_context_t *stile = stiles_context_self(bind_index);
        if (!stile) {
            sTiles::Logger::fatal(sTiles::StatusCode::NotInitialized, "sTiles context is not initialized.");
            return sTiles::StatusCode::NotInitialized;
        }

        sTiles::parallel_call(stile, sTiles::preprocess::symbolic_pdpotrf_semisparse, scheme);

        return sTiles::StatusCode::Success;
    }

    static sTiles::StatusCode pthreads_sparse_symbolic_dpotrf(int bind_index, TiledMatrix *scheme) {

        if (!_validate_scheme(scheme)) {
            return scheme ? sTiles::StatusCode::Success : sTiles::StatusCode::IllegalValue;
        }

        stiles_context_t *stile = stiles_context_self(bind_index);
        if (!stile) {
            sTiles::Logger::fatal(sTiles::StatusCode::NotInitialized, "sTiles context is not initialized.");
            return sTiles::StatusCode::NotInitialized;
        }

        sTiles::parallel_call(stile, sTiles::preprocess::symbolic_pdpotrf_sparse, scheme);

        return sTiles::StatusCode::Success;
    }

    sTiles::StatusCode symbolic_semisparse(int group_index, int call_index, TiledMatrix *scheme) {

        int* params = sTiles_get_params();
        const int tile_type_mode = params[sTiles::param::TileTypeMode];
        const int correction_mode = params[sTiles::param::SemisparsePruningMode];

        // Only proceed if:
        // - tile_type_mode == 1 (semisparse tiles only), OR
        // - tile_type_mode == 2 (dense and semisparse together), OR
        // - tile_type_mode == 0 (dense tiles) AND correction_mode > 0 (correction enabled)
        if (!(tile_type_mode == 1 || tile_type_mode == 2 || (tile_type_mode == 0 && correction_mode > 0))) {
            // Don't proceed with semisparse tile lookup
            return StatusCode::Success;
        }

        if (!scheme) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue,
                                "Cannot run semisparse symbolic phase on a null scheme.");
            return sTiles::StatusCode::IllegalValue;
        }

#if 0
        // -----------------------------------------------------------------------
        // OLD: task-DAG simulation via pthreads persistent team.
        // Kept for reference/comparison. Switch #if 0 -> #if 1 to restore.
        // -----------------------------------------------------------------------
        if (!scheme->call_lookup_table) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue,
                                "Call lookup table is not initialized for the provided scheme.");
            return sTiles::StatusCode::IllegalValue;
        }

        int bind_index = scheme->call_lookup_table[group_index][call_index];

        const int rc = sTiles::Control::ActivatePersistentTeam(bind_index);
        if (rc != EXIT_SUCCESS) {
            sTiles::Logger::error(sTiles::StatusCode::Failure,
                                "Failed to activate persistent team for semisparse symbolic phase.");
            return sTiles::StatusCode::Failure;
        }

        const sTiles::StatusCode status = pthreads_semi_sparse_symbolic_dpotrf(bind_index, scheme);
        //const sTiles::StatusCode status = pthreads_semi_sparse_symbolic_dpotri(bind_index, scheme);
        sTiles::Control::DeactivatePersistentTeam(bind_index);

        return status;
#endif

        // -----------------------------------------------------------------------
        // NEW: direct scan of exact etree fill-in (L_colptr / L_rowind).
        // Diagonal tiles  → exact half-bandwidth (upper_bw) for banded storage.
        // Off-diagonal tiles → mark active columns from L pattern; recompute fa/la/sa.
        // -----------------------------------------------------------------------
        if (!scheme->L_colptr || !scheme->L_rowind) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue,
                                  "L factor pattern (L_colptr/L_rowind) not available in scheme.");
            return sTiles::StatusCode::IllegalValue;
        }

        SemisparseTileMetaCore* semi      = scheme->semisparseTileMetaCore;
        TileMetaCore*           tile_meta = scheme->tileMetaCore;
        const bool*             bmap      = scheme->diagonal_bmapper;
        const int               num_tiles = scheme->numActiveTiles;
        const int               tile_size = scheme->tile_size;
        const int64_t*              L_colptr  = scheme->L_colptr;
        const int*              L_rowind  = scheme->L_rowind;

        if (!semi || !tile_meta || num_tiles <= 0) {
            return StatusCode::Success;
        }

        const int num_cores = (scheme->num_cores > 0) ? scheme->num_cores : 1;

        auto process_tile = [&](int t) {
            const int tile_col       = tile_meta[t].col;
            const int tile_row       = tile_meta[t].row;
            const int width          = tile_meta[t].width;
            const int height         = tile_meta[t].height;
            const int elem_col_start = tile_col * tile_size;
            const int elem_row_start = tile_row * tile_size;
            const int row_hi         = elem_row_start + height;

            SemisparseTileMetaCore& c = semi[t];
            const bool is_diag = (bmap && bmap[t]) || (tile_row == tile_col);

            if (is_diag) {
                // Diagonal tile: compute exact half-bandwidth from L entries within the tile.
                // The banded storage size is (upper_bw+1)*height — must be correct before
                // allocate_semisparse_tiles calls chunked_tile_element_count.
                // acol/sa for diagonal tiles are handled by compress_semisparse_columns.
                int upper_bw = 0;
                for (int j = 0; j < width; ++j) {
                    const int g = elem_col_start + j;
                    for (int64_t p = L_colptr[g]; p < L_colptr[g + 1]; ++p) {
                        const int r = L_rowind[p];
                        if (r < elem_col_start) continue;
                        if (r >= elem_col_start + width) break;  // entries are sorted
                        const int bw = r - g;                    // lower-tri: r >= g
                        if (bw > upper_bw) upper_bw = bw;
                    }
                }
                c.upper_bw = upper_bw;

            } else {
                // Off-diagonal tile: mark fill-in columns from the exact L pattern,
                // then recompute fa/la/sa from the full acol state.
                for (int j = 0; j < width; ++j) {
                    if (c.acol[static_cast<std::size_t>(j)] != -1) continue;  // already marked by phase1
                    const int g   = elem_col_start + j;
                    const int64_t p_lo = L_colptr[g];
                    const int64_t p_hi = L_colptr[g + 1];
                    for (int64_t p = p_lo; p < p_hi; ++p) {
                        const int r = L_rowind[p];
                        if (r >= row_hi) break;
                        if (r >= elem_row_start) {
                            c.acol[static_cast<std::size_t>(j)] = 1;
                            break;
                        }
                    }
                }

                int fa = -1, la = -1, sa = 0;
                for (int j = 0; j < width; ++j) {
                    if (c.acol[static_cast<std::size_t>(j)] != -1) {
                        if (fa < 0) fa = j;
                        la = j;
                        ++sa;
                    }
                }
                c.fa = fa;
                c.la = la;
                c.sa = sa;
            }
        };

    #ifdef _OPENMP
        const bool can_spawn = (!omp_in_parallel()) || (omp_get_max_active_levels() > 1);
        if (can_spawn && num_cores > 1) {
            #pragma omp parallel for schedule(static) num_threads(num_cores)
            for (int t = 0; t < num_tiles; ++t) {
                process_tile(t);
            }
            return StatusCode::Success;
        }
    #endif

        for (int t = 0; t < num_tiles; ++t) {
            process_tile(t);
        }
        return StatusCode::Success;
    }

    static inline void build_semisparse_column_map_for_tile(SemisparseTileMetaCore &c) {
        const int width = static_cast<int>(c.acol.size());
        const int sa = c.sa;

        if (width <= 0 || sa <= 0) {
            c.aind.clear();
            return;
        }

        c.aind.resize(static_cast<std::size_t>(sa));

        int local = 0;
        int first_active = -1;
        int last_active = -1;
        for (int j = 0; j < width; ++j) {
            if (c.acol[static_cast<std::size_t>(j)] != -1) {
                c.aind[static_cast<std::size_t>(local)] = j;   // cind[k] = column index
                c.acol[static_cast<std::size_t>(j)] = local;   // acol[j] = local index
                ++local;

                if (first_active < 0 || j < first_active) {
                    first_active = j;
                }
                if (j > last_active) {
                    last_active = j;
                }
            }
        }

        if (local > 0) {
            c.fa = first_active;
            c.la = last_active;
        } else {
            c.fa = -1;
            c.la = -1;
        }
    }

    static inline void build_semisparse_column_maps(TiledMatrix *scheme, int num_cores) {
        SemisparseTileMetaCore *core = scheme->semisparseTileMetaCore;
        const int num_tiles_active = scheme->numActiveTiles;
        if (!core || num_tiles_active <= 0) return;

        const bool *bmap = scheme->diagonal_bmapper;
        if (bmap) {
            for (int t = 0; t < num_tiles_active; ++t) {
                if (!bmap[t]) continue;

                SemisparseTileMetaCore &c = core[t];
                const int width = static_cast<int>(c.acol.size());
                if (width <= 0) {
                    continue;
                }

                c.fa = 0;
                c.la = width - 1;
                c.sa = width;
                std::fill(c.acol.begin(), c.acol.end(), 1);
            }
        }

    #ifndef _OPENMP
        (void)num_cores;
        for (int t = 0; t < num_tiles_active; ++t) {
            build_semisparse_column_map_for_tile(core[t]);
        }
    #else
        const bool nested_enabled = (omp_get_max_active_levels() > 1);
        const bool can_spawn = (!omp_in_parallel()) || nested_enabled;
        const int threads = (num_cores > 0) ? num_cores : omp_get_max_threads();
        const int nnz = scheme->original_nnz;

        if (!can_spawn || threads < 2 || nnz <= 5000) {
            for (int t = 0; t < num_tiles_active; ++t) {
                build_semisparse_column_map_for_tile(core[t]);
            }
        } else {
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (int t = 0; t < num_tiles_active; ++t) {
                build_semisparse_column_map_for_tile(core[t]);
            }
        }
    #endif
    }

    static inline void build_symbolic_bitmasks_for_semisparse_tiles_serial(TiledMatrix *scheme) {
        const int num_tiles_active = scheme->numActiveTiles;
        const int original_nnz = scheme->original_nnz;

        if (num_tiles_active <= 0 || original_nnz <= 0) return;

        SemisparseTileMetaCore *core = scheme->semisparseTileMetaCore;
        TileMetaCore *tile_meta = scheme->tileMetaCore;
        SymbolicTileBitmaskCore *symb = scheme->symbolicTileBitmaskCore;
        const bool *bmap = scheme->diagonal_bmapper;

        if (!core || !tile_meta || !symb || !scheme->withinTileRow || !scheme->withinTileCol || !scheme->tile_index_lookup) {
            sTiles::Logger::error("[SymbolicBitmasks] Missing metadata to build symbolic bitmasks.");
            return;
        }

        // 1) init bitmaps per tile using sa and tile height
        for (int t = 0; t < num_tiles_active; ++t) {
            SemisparseTileMetaCore &c = core[t];
            TileMetaCore &tm = tile_meta[t];
            SymbolicTileBitmaskCore &s = symb[t];

            // if (bmap && bmap[t]) {
            //     // diagonal tiles are dense or banded, usually you do not need bitmaps
            //     s.reset();
            //     continue;
            // }

            const int sa = c.sa;
            const int height = tm.height;
            s.init(sa, height);
        }

        // 2) fill bitmaps using original nnz, tile index and within tile coords
        for (int idx = 0; idx < original_nnz; ++idx) {
            const int tile = scheme->tile_index_lookup[idx];
            if (tile < 0 || tile >= num_tiles_active) continue;

            // if (bmap && bmap[tile]) {
            //     // skip diagonal tiles if they are not semisparse
            //     continue;
            // }

            SemisparseTileMetaCore &c = core[tile];
            SymbolicTileBitmaskCore &s = symb[tile];
            if (s.words_per_col == 0) continue;

            const int width = static_cast<int>(c.acol.size());
            const int local_col = scheme->withinTileCol[idx];
            const int local_row = scheme->withinTileRow[idx];

            if (local_col < 0 || local_col >= width || local_row < 0) {
                continue;
            }

            const int active_col = c.acol[static_cast<std::size_t>(local_col)];
            if (active_col < 0) {
                // column not active in this semisparse tile
                continue;
            }

            s.set_bit(active_col, local_row);
        }
    }

    static inline void build_symbolic_bitmasks_for_semisparse_tiles(TiledMatrix *scheme, int num_cores) {
        const int num_tiles_active = scheme->numActiveTiles;
        const int original_nnz = scheme->original_nnz;

        if (num_tiles_active <= 0 || original_nnz <= 0) return;

    #ifndef _OPENMP
        (void)num_cores;
        build_symbolic_bitmasks_for_semisparse_tiles_serial(scheme);
    #else
        SemisparseTileMetaCore *core = scheme->semisparseTileMetaCore;
        TileMetaCore *tile_meta = scheme->tileMetaCore;
        SymbolicTileBitmaskCore *symb = scheme->symbolicTileBitmaskCore;
        const bool *bmap = scheme->diagonal_bmapper;

        if (!core || !tile_meta || !symb || !scheme->withinTileRow || !scheme->withinTileCol || !scheme->tile_index_lookup) {
            sTiles::Logger::error("[SymbolicBitmasks] Missing metadata to build symbolic bitmasks.");
            return;
        }

        const bool nested_enabled = (omp_get_max_active_levels() > 1);
        const bool can_spawn = (!omp_in_parallel()) || nested_enabled;
        const int threads = (num_cores > 0) ? num_cores : omp_get_max_threads();

        if (!can_spawn || threads < 2 || original_nnz <= 5000) {
            build_symbolic_bitmasks_for_semisparse_tiles_serial(scheme);
            return;
        }

        // 1) init bitmaps per tile in parallel
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int t = 0; t < num_tiles_active; ++t) {
            SemisparseTileMetaCore &c = core[t];
            TileMetaCore &tm = tile_meta[t];
            SymbolicTileBitmaskCore &s = symb[t];

            // if (bmap && bmap[t]) {
            //     s.reset();
            //     continue;
            // }

            const int sa = c.sa;
            const int height = tm.height;
            s.init(sa, height);
        }

        // 2) fill bitmaps in parallel, protect per tile updates with locks
        std::vector<omp_lock_t> tile_locks(static_cast<std::size_t>(num_tiles_active));
        for (int t = 0; t < num_tiles_active; ++t) {
            omp_init_lock(&tile_locks[static_cast<std::size_t>(t)]);
        }

        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int idx = 0; idx < original_nnz; ++idx) {
            const int tile = scheme->tile_index_lookup[idx];
            if (tile < 0 || tile >= num_tiles_active) continue;

            // if (bmap && bmap[tile]) {
            //     continue;
            // }

            SemisparseTileMetaCore &c = core[tile];
            SymbolicTileBitmaskCore &s = symb[tile];
            if (s.words_per_col == 0) continue;

            const int width = static_cast<int>(c.acol.size());
            const int local_col = scheme->withinTileCol[idx];
            const int local_row = scheme->withinTileRow[idx];

            if (local_col < 0 || local_col >= width || local_row < 0) {
                continue;
            }

            const int active_col = c.acol[static_cast<std::size_t>(local_col)];
            if (active_col < 0) {
                continue;
            }

            omp_set_lock(&tile_locks[static_cast<std::size_t>(tile)]);
            s.set_bit(active_col, local_row);
            omp_unset_lock(&tile_locks[static_cast<std::size_t>(tile)]);
        }

        for (int t = 0; t < num_tiles_active; ++t) {
            omp_destroy_lock(&tile_locks[static_cast<std::size_t>(t)]);
        }
    #endif
    }

    inline StatusCode build_symbolic_sparse(TiledMatrix *scheme, int group_index, int num_cores) {
        if (!scheme) return StatusCode::InvalidArgument;

        const int num_active = scheme->numActiveTiles;
        if (num_active <= 0) return StatusCode::InvalidArgument;

        if (!scheme->symbolicTileBitmaskCore) {
            scheme->symbolicTileBitmaskCore = TileMemoryManager::allocate<SymbolicTileBitmaskCore>(num_active, group_index);
            if (!scheme->symbolicTileBitmaskCore) {
                sTiles::Logger::errorf("Memory allocation failed for symbolicTileBitmaskCore.");
                return StatusCode::OutOfResources;
            }
        }

        // assumes build_semisparse_tile_lookup_phase1/2 already ran and filled semisparseTileMetaCore etc
        build_semisparse_column_maps(scheme, num_cores);
        build_symbolic_bitmasks_for_semisparse_tiles(scheme, num_cores);

        // SemisparseTileMetaCore *core = scheme->semisparseTileMetaCore;
        // if (core) {
        //     for (int t = 0; t < num_active; ++t) {
        //         SemisparseTileMetaCore &c = core[t];

        //         sTiles::Logger::errorf("Tile %d:", t);
        //         sTiles::Logger::errorf("  sa = %d", c.sa);

        //         sTiles::Logger::errorf("  acol (size = %zu): ", c.acol.size());
        //         for (std::size_t j = 0; j < c.acol.size(); ++j) {
        //             sTiles::Logger::errorf("%d ", c.acol[j]);
        //         }
        //         sTiles::Logger::errorf("");

        //         sTiles::Logger::errorf("  aind (size = %zu): ", c.aind.size());
        //         for (std::size_t k = 0; k < c.aind.size(); ++k) {
        //             sTiles::Logger::errorf("%d ", c.aind[k]);
        //         }
        //         sTiles::Logger::errorf("");
        //     }
        // }
        // exit(0);


        return StatusCode::Ok;
    }

    sTiles::StatusCode symbolic_sparse(int group_index, int call_index, TiledMatrix *scheme, int num_cores) {

        if (!scheme) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue,
                                "Cannot run semisparse symbolic phase on a null scheme.");
            return sTiles::StatusCode::IllegalValue;
        }

        if (!scheme->call_lookup_table) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue,
                                "Call lookup table is not initialized for the provided scheme.");
            return sTiles::StatusCode::IllegalValue;
        }

        int bind_index = scheme->call_lookup_table[group_index][call_index];

        // build symbolic metadata (column maps + bitmaps)
        const StatusCode sym_status = build_symbolic_sparse(scheme, group_index, num_cores);
        if (sym_status != StatusCode::Ok) {
            sTiles::Logger::error(sym_status, "Failed to build symbolic sparse metadata.");
            return sym_status;
        }

        const int rc = sTiles::Control::ActivatePersistentTeam(bind_index);
        if (rc != EXIT_SUCCESS) {
            sTiles::Logger::error(sTiles::StatusCode::Failure,
                                "Failed to activate persistent team for semisparse symbolic phase.");
            return sTiles::StatusCode::Failure;
        }

        const sTiles::StatusCode status = pthreads_sparse_symbolic_dpotrf(bind_index, scheme);
        // const sTiles::StatusCode status = sTiles::pthreads_semi_sparse_symbolic_dpotri(bind_index, scheme);
        sTiles::Control::DeactivatePersistentTeam(bind_index);

        return sym_status;
    }


} // namespace preprocess

} // namespace sTiles
