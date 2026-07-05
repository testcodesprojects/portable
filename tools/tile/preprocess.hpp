/**
 * @file preprocess_dense_tiles.hpp
 * @brief Preprocessing and allocation routines for dense tile structures.
 *
 * Provides functions for allocating and initializing dense tile metadata,
 * sparse tile construction, and tile memory management during the preprocessing
 * phase of tiled matrix factorization.
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

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstddef>   // for std::size_t
#include <atomic>
#include <exception>

#include "meta.hpp"
#include "../common/stiles_structs.hpp"
#include "../memory/TileMemoryManager.hpp"
#include "../memory/MemoryManager.hpp"
#include "../TileIndexer/TileIndexerMapper.hpp"
#include "../TileIndexer/TileIndexerMemoryUtils.hpp"
#include "../common/stiles_global.hpp"

// #ifdef STILES_GPU
// #include "../process/GpuMemoryManager.hpp"
// #endif

namespace sTiles {

    static inline void set_tile_extents(TiledMatrix **scheme) {
        if (!scheme) return;
        TiledMatrix* matrix = *scheme;
        if (!matrix || !matrix->tileMetaCore) return;

        const int num_tiles  = matrix->dimTiledMatrix;
        const int max_active = matrix->numActiveTiles;
        if (num_tiles <= 0 || max_active <= 0) return;
        if (!matrix->mapper.valid()) return;

        const int tile_size = matrix->tile_size;
        if (tile_size <= 0) return;

        const int n = matrix->original_order;
        const int remainder = (n % tile_size == 0) ? tile_size : (n % tile_size);

        for (int j = 0; j < num_tiles; ++j) {
            const int width = (j == num_tiles - 1) ? remainder : tile_size;
            for (int i = 0; i <= j; ++i) {
                const int dense_idx = matrix->mapper.map_ij(i, j, num_tiles);
                if (dense_idx < 0 || dense_idx >= max_active) continue;

                const int height = (i == num_tiles - 1) ? remainder : tile_size;

                TileMetaCore& meta = matrix->tileMetaCore[dense_idx];
                meta.index  = dense_idx;
                meta.row    = i;
                meta.col    = j;
                meta.width  = width;
                meta.height = height;
            }
        }
    }
}

namespace sTiles { namespace preprocess{

    static inline void compute_remainder(TiledMatrix* scheme) {
        const int ts = scheme->tile_size;
        const int n  = scheme->original_order;
        scheme->remainderTileSize = (n % ts == 0) ? ts : (n % ts);
    }

    // Allocates the top-level pointer arrays (denseTiles, tileMetaCore, and optionally inverse/saved).
    static inline StatusCode allocate_top_level_arrays(TiledMatrix* scheme, int group_index) {
        // ---------- allocate pointer arrays + metadata ----------
        scheme->inverseTiles = nullptr;
        scheme->savedTiles   = nullptr;
#ifdef SMART_TILES
        scheme->rhsTiles     = nullptr;
#endif

        scheme->denseTiles = TileMemoryManager::allocate<DenseTile>(scheme->numActiveTiles, group_index);
        scheme->tileMetaCore   = TileMemoryManager::allocate<TileMetaCore>(scheme->numActiveTiles, group_index);

        if (!scheme->denseTiles || !scheme->tileMetaCore) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for denseTiles or tileMetaCore.\n");
            return StatusCode::OutOfResources;
        }

        if (scheme->compute_inverse) {
            scheme->inverseTiles = TileMemoryManager::allocate<DenseTile>(scheme->numActiveTiles, group_index);
            scheme->savedTiles   = TileMemoryManager::allocate<DenseTile>(scheme->numActiveTiles, group_index);
            if (!scheme->inverseTiles || !scheme->savedTiles) {
                std::fprintf(stderr, "ERROR: Memory allocation failed for inverseTiles/savedTiles.\n");
                return StatusCode::OutOfResources;
            }
        }

        return StatusCode::Success;
    }

    static StatusCode allocate_buffers_for_tile_org(TiledMatrix* scheme, int idx, int group_index) {
        // For variants 1 and 2, tileMetaCore may not be allocated
        // Use matrix dim directly in that case
        int h, w;
        if (scheme->tileMetaCore) {
            const TileMetaCore& m = scheme->tileMetaCore[idx];
            h = (m.height > 0) ? m.height : scheme->tile_size;
            w = (m.width  > 0) ? m.width  : scheme->tile_size;
        } else {
            // Variant 1: single tile covering entire matrix
            h = scheme->dim;
            w = scheme->dim;
        }

        const std::size_t elems = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);

        // Dense (factor) tile
        scheme->denseTiles[idx] = TileMemoryManager::allocate<double>(elems, group_index);
        if (!scheme->denseTiles[idx]) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for denseTiles[%d].\n", idx);
            return StatusCode::OutOfResources;
        }

        if (scheme->compute_inverse) {
            // Inverse tile
            scheme->inverseTiles[idx] = TileMemoryManager::allocate<double>(elems, group_index);
            if (!scheme->inverseTiles[idx]) {
                std::fprintf(stderr, "ERROR: Memory allocation failed for inverseTiles[%d].\n", idx);
                return StatusCode::OutOfResources;
            }
            // Zero initialize (column-major leading dimension = h)
            LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, scheme->inverseTiles[idx], h);

            // Saved tile
            scheme->savedTiles[idx] = TileMemoryManager::allocate<double>(elems, group_index);
            if (!scheme->savedTiles[idx]) {
                std::fprintf(stderr, "ERROR: Memory allocation failed for savedTiles[%d].\n", idx);
                return StatusCode::OutOfResources;
            }
        }

        return StatusCode::Success;
    }


    static StatusCode allocate_buffers_for_all_tiles_org(TiledMatrix* scheme, int group_index, int num_cores = 1) {
        #ifdef _OPENMP
        if (num_cores > 1) {
            std::atomic<int> error_code{0};

            #pragma omp parallel for schedule(static) num_threads(num_cores)
            for (int idx = 0; idx < scheme->numActiveTiles; ++idx) {
                if (error_code.load(std::memory_order_relaxed) != 0) {
                    continue;
                }

                const StatusCode sc = allocate_buffers_for_tile_org(scheme, idx, group_index);
                if (sc != StatusCode::Success) {
                    int expected = 0;
                    const int cast_code = static_cast<int>(sc);
                    error_code.compare_exchange_strong(expected, cast_code,
                                                        std::memory_order_relaxed,
                                                        std::memory_order_relaxed);
                }
            }

            const int final_code = error_code.load(std::memory_order_relaxed);
            if (final_code != 0) {
                std::fprintf(stderr, "ERROR: Parallel tile allocation failed with code %d. Exiting.\n", final_code);
                std::exit(EXIT_FAILURE);
            }
            return StatusCode::Success;
        }
        #else
        (void)num_cores; // suppress unused warning when OpenMP is disabled
        #endif

        for (int idx = 0; idx < scheme->numActiveTiles; ++idx) {
            const StatusCode sc = allocate_buffers_for_tile_org(scheme, idx, group_index);
            if (sc != StatusCode::Success) {
                std::fprintf(stderr, "ERROR: Tile allocation failed at index %d. Exiting.\n", idx);
                std::exit(EXIT_FAILURE);
            }
        }

        return StatusCode::Success;
    }

    // Forward declaration for dense variant
    static inline StatusCode init_inverse_identity_on_diagonals_dense(TiledMatrix* scheme);

    static inline StatusCode init_inverse_identity_on_diagonals_org(TiledMatrix* scheme) {
        if (!scheme->compute_inverse || !scheme->inverseTiles) return StatusCode::Success;
        if (!scheme->tileMetaCore) return StatusCode::Success;

        // Use dense variant if diagonal_mapper is not available (dense variants 1 and 2)
        if (!scheme->diagonal_mapper) {
            return init_inverse_identity_on_diagonals_dense(scheme);
        }

        for (int tile_idx = 0; tile_idx < scheme->dimTiledMatrix; ++tile_idx) {
            const int dense_idx = scheme->diagonal_mapper[tile_idx];
            if (dense_idx < 0) continue; // safety

            const TileMetaCore& m = scheme->tileMetaCore[dense_idx];
            const int h = (m.height > 0) ? m.height : scheme->tile_size;
            const int w = (m.width  > 0) ? m.width  : scheme->tile_size;

            double* inv = scheme->inverseTiles[dense_idx];
            if (!inv) continue; // should already be allocated

            const int ld  = h;                  // column-major leading dimension
            const int dsz = std::min(h, w);     // <-- handle rectangular tiles safely
            for (int ii = 0; ii < dsz; ++ii) {
                inv[ii * ld + ii] = 1.0;
            }
        }
        return StatusCode::Success;
    }

    // Dense variant: computes diagonal indices directly (no diagonal_bmapper needed)
    static inline StatusCode init_inverse_identity_on_diagonals_dense(TiledMatrix* scheme) {
        if (!scheme->compute_inverse || !scheme->inverseTiles) return StatusCode::Success;
        if (!scheme->tileMetaCore) return StatusCode::Success;

        const int num_active = scheme->numActiveTiles;

        if (num_active == 1) {
            // Variant 1: Single tile covering entire matrix - always diagonal
            const TileMetaCore& m = scheme->tileMetaCore[0];
            const int h = (m.height > 0) ? m.height : scheme->tile_size;
            double* inv = scheme->inverseTiles[0];
            if (inv) {
                for (int ii = 0; ii < h; ++ii) {
                    inv[ii * h + ii] = 1.0;
                }
            }
        } else {
            // Variant 2: Diagonal tiles at computed indices
            // Diagonal tile (i,i) is at upper triangular index: i * N - (i * (i - 1)) / 2
            const int N = scheme->dimTiledMatrix;
            for (int i = 0; i < N; ++i) {
                const int idx = i * N - (i * (i - 1)) / 2;
                if (idx < 0 || idx >= num_active) continue;

                const TileMetaCore& m = scheme->tileMetaCore[idx];
                const int h = (m.height > 0) ? m.height : scheme->tile_size;
                double* inv = scheme->inverseTiles[idx];
                if (inv) {
                    for (int ii = 0; ii < h; ++ii) {
                        inv[ii * h + ii] = 1.0;
                    }
                }
            }
        }
        return StatusCode::Success;
    }

    inline StatusCode structering_dense_tiles(sTiles_call **call_info, TiledMatrix **scheme, int call_index, int group_index) {

        if ((*scheme)->use_ordering == 4) {
            (*call_info)->use_nested_dissection = true;
        }

        // sparse_dense
        if ((*call_info)->factorization_variant == 0) {

            // 1) Top-level arrays (pointers + metadata)
            {
                StatusCode sc = allocate_top_level_arrays(*scheme, group_index);
                if (sc != StatusCode::Success) return sc;
            }

            // 2) Tile extents + remainder size (extents might fill tileMetaCore)
            compute_remainder(*scheme);
            if (call_index == 0) sTiles::set_tile_extents(scheme);


            #ifdef STILES_FASTMODE
            // 3) Per-tile buffer allocations (dense + optional inverse/saved)
            {
                const int threads = (*call_info)->num_cores; 
                StatusCode sc = allocate_buffers_for_all_tiles_org(*scheme, group_index, threads);
                if (sc != StatusCode::Success) return sc;
            }
            #elif STILES_SEMISPARSEMODE

            // 3) Per-tile buffer allocations (dense + optional inverse/saved)
            {
                //remove or change the position of this
                const int threads = (*call_info)->num_cores; 
                StatusCode sc = allocate_buffers_for_all_tiles_org(*scheme, group_index, threads);
                if (sc != StatusCode::Success) return sc;
            }
            #endif


            // 4) Initialize inverse tiles’ diagonals to identity on the diagonal tiles
            {
                StatusCode sc = init_inverse_identity_on_diagonals_org(*scheme);
                if (sc != StatusCode::Success) return sc;
            }

        // one chunk
        } else if ((*call_info)->factorization_variant == 1) {
            
            std::cout << "TODO: add one-chunk variant helpers here as needed." << std::endl;
            exit(0);
        // dense_dense
        } else if ((*call_info)->factorization_variant == 2) {

            std::cout << "TODO: add dense-dense variant helpers here as needed." << std::endl;
            exit(0);
        } else if ((*call_info)->factorization_variant == 3) {

            StatusCode sc = StatusCode::Success;
    #ifdef STILES_SMART_COMPARE
            sc = allocate_top_level_arrays(*scheme, group_index);
            if (sc != StatusCode::Success) {
                return sc;
            }
    #else
            (*scheme)->tileMetaCore = TileMemoryManager::allocate<TileMetaCore>((*scheme)->numActiveTiles, group_index);
            if (!(*scheme)->tileMetaCore) {
                std::fprintf(stderr, "ERROR: Memory allocation failed for denseTiles or tileMetaCore.\n");
                return StatusCode::OutOfResources;
            }
    #endif
            compute_remainder(*scheme);
            if (call_index == 0) sTiles::set_tile_extents(scheme);

    #ifdef STILES_SMART_COMPARE
            const int threads = (*call_info)->num_cores;
            sc = allocate_buffers_for_all_tiles_org(*scheme, group_index, threads);
            if (sc != StatusCode::Success) {
                return sc;
            }
    #endif
        }

        return StatusCode::Success;
    }

    inline StatusCode allocate_dense_buffers_from_primary(const TiledMatrix* primary, TiledMatrix* clone, int group_index, int num_threads)
    {
        if (!primary || !clone) {
            return StatusCode::Failure;
        }

        const int num_tiles = primary->numActiveTiles;
        if (num_tiles <= 0) {
            clone->denseTiles   = nullptr;
            clone->inverseTiles = nullptr;
            clone->savedTiles   = nullptr;
#ifdef SMART_TILES
            clone->rhsTiles     = nullptr;
#endif
            return StatusCode::Success;
        }

        try {
            clone->denseTiles = TileMemoryManager::allocateZero<DenseTile>(num_tiles, group_index);
            if (primary->compute_inverse) {
                clone->inverseTiles = TileMemoryManager::allocateZero<DenseTile>(num_tiles, group_index);
                clone->savedTiles   = TileMemoryManager::allocateZero<DenseTile>(num_tiles, group_index);
            } else {
                clone->inverseTiles = nullptr;
                clone->savedTiles   = nullptr;
            }
#ifdef SMART_TILES
            clone->rhsTiles = nullptr;
#endif
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "ERROR: Failed to allocate dense tile pointer arrays while cloning fast-mode scheme (%s).\n", ex.what());
            return StatusCode::OutOfResources;
        }

        const auto allocate_tile = [&](int idx) -> StatusCode {
            return allocate_buffers_for_tile_org(clone, idx, group_index);
        };

    #ifdef _OPENMP
        const int threads = (num_threads > 0) ? num_threads : 1;
        if (threads > 1) {
            std::atomic<int> err_code{0};
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (int idx = 0; idx < num_tiles; ++idx) {
                if (err_code.load(std::memory_order_relaxed) != 0) {
                    continue;
                }
                const StatusCode sc = allocate_tile(idx);
                if (sc != StatusCode::Success) {
                    int expected = 0;
                    err_code.compare_exchange_strong(expected,
                                                    static_cast<int>(sc),
                                                    std::memory_order_relaxed,
                                                    std::memory_order_relaxed);
                }
            }
            const int final_code = err_code.load(std::memory_order_relaxed);
            if (final_code != 0) {
                return static_cast<StatusCode>(final_code);
            }
        } else
    #endif
        {
            for (int idx = 0; idx < num_tiles; ++idx) {
                const StatusCode sc = allocate_tile(idx);
                if (sc != StatusCode::Success) {
                    return sc;
                }
            }
        }

        if (clone->compute_inverse) {
            const StatusCode sc = init_inverse_identity_on_diagonals_org(clone);
            if (sc != StatusCode::Success) {
                return sc;
            }
        }

        return StatusCode::Success;
    }

}}




// ==============================
// small functions (helpers)
// ==============================
// namespace detail {
    


//     // Allocates buffers for a single tile (dense, and optionally inverse/saved).
//     static inline StatusCode allocate_buffers_for_tile_org(TiledMatrix* scheme, int idx, int group_index) {
//         const TileMetaCore& m = scheme->tileMetaCore[idx];
//         const int h = (m.height > 0) ? m.height : scheme->tile_size;
//         const int w = (m.width  > 0) ? m.width  : scheme->tile_size;

//         const std::size_t elems = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);

//         // Dense (factor) tile
//         scheme->denseTiles[idx] = TileMemoryManager::allocate<double>(elems, group_index);
//         if (!scheme->denseTiles[idx]) {
//             std::fprintf(stderr, "ERROR: Memory allocation failed for denseTiles[%d].\n", idx);
//             return StatusCode::OutOfResources;
//         }

//         if (scheme->compute_inverse) {
//             // Inverse tile
//             scheme->inverseTiles[idx] = TileMemoryManager::allocate<double>(elems, group_index);
//             if (!scheme->inverseTiles[idx]) {
//                 std::fprintf(stderr, "ERROR: Memory allocation failed for inverseTiles[%d].\n", idx);
//                 return StatusCode::OutOfResources;
//             }
//             // Zero initialize (column-major leading dimension = h)
//             LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, scheme->inverseTiles[idx], h);

//             // Saved tile
//             scheme->savedTiles[idx] = TileMemoryManager::allocate<double>(elems, group_index);
//             if (!scheme->savedTiles[idx]) {
//                 std::fprintf(stderr, "ERROR: Memory allocation failed for savedTiles[%d].\n", idx);
//                 return StatusCode::OutOfResources;
//             }
//         }

//         return StatusCode::Success;
//     }

//     // Allocates buffers for all active tiles using their own extents.
//     static inline StatusCode allocate_buffers_for_all_tiles_org(TiledMatrix* scheme,
//                                                             int group_index,
//                                                             int num_threads = 1) {
// #ifdef _OPENMP
//         if (num_threads > 1) {
//             std::atomic<int> error_code{0};

//             #pragma omp parallel for schedule(static) num_threads(num_threads)
//             for (int idx = 0; idx < scheme->numActiveTiles; ++idx) {
//                 if (error_code.load(std::memory_order_relaxed) != 0) {
//                     continue;
//                 }

//                 const StatusCode sc = allocate_buffers_for_tile_org(scheme, idx, group_index);
//                 if (sc != StatusCode::Success) {
//                     int expected = 0;
//                     const int cast_code = static_cast<int>(sc);
//                     error_code.compare_exchange_strong(expected, cast_code,
//                                                         std::memory_order_relaxed,
//                                                         std::memory_order_relaxed);
//                 }
//             }

//             const int final_code = error_code.load(std::memory_order_relaxed);
//             if (final_code != 0) {
//                 return static_cast<StatusCode>(final_code);
//             }
//             return StatusCode::Success;
//         }
// #else
//         (void)num_threads; // suppress unused warning when OpenMP is disabled
// #endif

//         for (int idx = 0; idx < scheme->numActiveTiles; ++idx) {
//             const StatusCode sc = allocate_buffers_for_tile_org(scheme, idx, group_index);
//             if (sc != StatusCode::Success) return sc;
//         }
//         return StatusCode::Success;
//     }

//     // Puts 1s on the diagonal of each *diagonal* inverse tile, if we’re computing an inverse.
//     static inline StatusCode init_inverse_identity_on_diagonals_org(TiledMatrix* scheme) {
//         if (!scheme->compute_inverse || !scheme->inverseTiles) return StatusCode::Success;

//         for (int tile_idx = 0; tile_idx < scheme->dimTiledMatrix; ++tile_idx) {
//             const int dense_idx = scheme->diagonal_mapper[tile_idx];
//             if (dense_idx < 0) continue; // safety

//             const TileMetaCore& m = scheme->tileMetaCore[dense_idx]; // <-- bugfix: index with dense_idx
//             const int h = (m.height > 0) ? m.height : scheme->tile_size;
//             const int w = (m.width  > 0) ? m.width  : scheme->tile_size;

//             double* inv = scheme->inverseTiles[dense_idx];
//             if (!inv) continue; // should already be allocated

//             const int ld  = h;                  // column-major leading dimension
//             const int dsz = std::min(h, w);     // <-- handle rectangular tiles safely
//             for (int ii = 0; ii < dsz; ++ii) {
//                 inv[ii * ld + ii] = 1.0;
//             }
//         }
//         return StatusCode::Success;
//     }
// } // namespace detail

// namespace preprocess {

// // Allocates per-tile dense buffers for a clone scheme using the primary's metadata.
// // Assumes clone->tileMetaCore is already set to the primary's tileMetaCore before calling.
// inline StatusCode allocate_dense_buffers_from_primary(const TiledMatrix* primary, TiledMatrix* clone, int group_index, int num_threads)
// {
//     if (!primary || !clone) {
//         return StatusCode::Failure;
//     }

//     const int num_tiles = primary->numActiveTiles;
//     if (num_tiles <= 0) {
//         clone->denseTiles   = nullptr;
//         clone->inverseTiles = nullptr;
//         clone->savedTiles   = nullptr;
//         clone->rhsTiles     = nullptr;
//         return StatusCode::Success;
//     }

//     try {
//         clone->denseTiles = TileMemoryManager::allocateZero<DenseTile>(num_tiles, group_index);
//         if (primary->compute_inverse) {
//             clone->inverseTiles = TileMemoryManager::allocateZero<DenseTile>(num_tiles, group_index);
//             clone->savedTiles   = TileMemoryManager::allocateZero<DenseTile>(num_tiles, group_index);
//         } else {
//             clone->inverseTiles = nullptr;
//             clone->savedTiles   = nullptr;
//         }
//         clone->rhsTiles = nullptr;
//     } catch (const std::exception& ex) {
//         std::fprintf(stderr, "ERROR: Failed to allocate dense tile pointer arrays while cloning fast-mode scheme (%s).\n", ex.what());
//         return StatusCode::OutOfResources;
//     }

//     const auto allocate_tile = [&](int idx) -> StatusCode {
//         return allocate_buffers_for_tile_org(clone, idx, group_index);
//     };

// #ifdef _OPENMP
//     const int threads = (num_threads > 0) ? num_threads : 1;
//     if (threads > 1) {
//         std::atomic<int> err_code{0};
//         #pragma omp parallel for schedule(static) num_threads(threads)
//         for (int idx = 0; idx < num_tiles; ++idx) {
//             if (err_code.load(std::memory_order_relaxed) != 0) {
//                 continue;
//             }
//             const StatusCode sc = allocate_tile(idx);
//             if (sc != StatusCode::Success) {
//                 int expected = 0;
//                 err_code.compare_exchange_strong(expected,
//                                                  static_cast<int>(sc),
//                                                  std::memory_order_relaxed,
//                                                  std::memory_order_relaxed);
//             }
//         }
//         const int final_code = err_code.load(std::memory_order_relaxed);
//         if (final_code != 0) {
//             return static_cast<StatusCode>(final_code);
//         }
//     } else
// #endif
//     {
//         for (int idx = 0; idx < num_tiles; ++idx) {
//             const StatusCode sc = allocate_tile(idx);
//             if (sc != StatusCode::Success) {
//                 return sc;
//             }
//         }
//     }

//     if (clone->compute_inverse) {
//         const StatusCode sc = init_inverse_identity_on_diagonals_org(clone);
//         if (sc != StatusCode::Success) {
//             return sc;
//         }
//     }

//     return StatusCode::Success;
// }


// inline StatusCode structering_semisparse_tiles(sTiles_call **call_info, TiledMatrix **scheme, int call_index, int group_index) {

//     if ((*scheme)->use_ordering == 4) {
//         (*call_info)->use_nested_dissection = true;
//     }

//     // sparse_dense
//     if ((*call_info)->factorization_variant == 0) {

//         // 1) Top-level arrays (pointers + metadata)
//         {
//             StatusCode sc = allocate_top_level_chunked_arrays(*scheme, group_index);
//             if (sc != StatusCode::Success) return sc;
//         }

//         // 2) Tile extents + remainder size (extents might fill tileMetaCore)
//         compute_remainder(*scheme);
//         if (call_index == 0) sTiles::set_tile_extents(scheme);


//         #ifdef STILES_FASTMODE
//         // 3) Per-tile buffer allocations (dense + optional inverse/saved)
//         {
//             const int threads = (*call_info)->num_cores; 
//             StatusCode sc = allocate_buffers_for_all_tiles_org(*scheme, group_index, threads);
//             if (sc != StatusCode::Success) return sc;
//         }
//         #elif STILES_SEMISPARSEMODE

//         // 3) Per-tile buffer allocations (dense + optional inverse/saved)
//         {
//             //remove or change the position of this
//             const int threads = (*call_info)->num_cores; 
//             StatusCode sc = allocate_buffers_for_all_tiles_org(*scheme, group_index, threads);
//             if (sc != StatusCode::Success) return sc;
//         }
//         #endif


//         // 4) Initialize inverse tiles’ diagonals to identity on the diagonal tiles
//         {
//             StatusCode sc = init_inverse_identity_on_diagonals_org(*scheme);
//             if (sc != StatusCode::Success) return sc;
//         }

//     // one chunk
//     } else if ((*call_info)->factorization_variant == 1) {
        
//         std::cout << "TODO: add one-chunk variant helpers here as needed." << std::endl;
//         exit(0);
//     // dense_dense
//     } else if ((*call_info)->factorization_variant == 2) {

//         std::cout << "TODO: add dense-dense variant helpers here as needed." << std::endl;
//         exit(0);
//     } else if ((*call_info)->factorization_variant == 3) {

//         StatusCode sc = StatusCode::Success;
// #ifdef STILES_SMART_COMPARE
//         sc = allocate_top_level_arrays(*scheme, group_index);
//         if (sc != StatusCode::Success) {
//             return sc;
//         }
// #else
//         (*scheme)->tileMetaCore = TileMemoryManager::allocate<TileMetaCore>((*scheme)->numActiveTiles, group_index);
//         if (!(*scheme)->tileMetaCore) {
//             std::fprintf(stderr, "ERROR: Memory allocation failed for denseTiles or tileMetaCore.\n");
//             return StatusCode::OutOfResources;
//         }
// #endif
//         compute_remainder(*scheme);
//         if (call_index == 0) sTiles::set_tile_extents(scheme);

// #ifdef STILES_SMART_COMPARE
//         const int threads = (*call_info)->num_cores;
//         sc = allocate_buffers_for_all_tiles_org(*scheme, group_index, threads);
//         if (sc != StatusCode::Success) {
//             return sc;
//         }
// #endif
//     }

//     return StatusCode::Success;
// }

// inline StatusCode structering_sparse_tiles(sTiles_call **call_info, TiledMatrix **scheme, int call_index, int group_index) {

//     if ((*scheme)->use_ordering == 4) {
//         (*call_info)->use_nested_dissection = true;
//     }

//     // sparse_dense
//     if ((*call_info)->factorization_variant == 0) {

//         // 1) Top-level arrays (pointers + metadata)
//         {
//             StatusCode sc = allocate_top_level_chunked_arrays(*scheme, group_index);
//             if (sc != StatusCode::Success) return sc;
//         }

//         // 2) Tile extents + remainder size (extents might fill tileMetaCore)
//         compute_remainder(*scheme);
//         if (call_index == 0) sTiles::set_tile_extents(scheme);


//         #ifdef STILES_FASTMODE
//         // 3) Per-tile buffer allocations (dense + optional inverse/saved)
//         {
//             const int threads = (*call_info)->num_cores; 
//             StatusCode sc = allocate_buffers_for_all_tiles_org(*scheme, group_index, threads);
//             if (sc != StatusCode::Success) return sc;
//         }
//         #elif STILES_SEMISPARSEMODE

//         // 3) Per-tile buffer allocations (dense + optional inverse/saved)
//         {
//             //remove or change the position of this
//             const int threads = (*call_info)->num_cores; 
//             StatusCode sc = allocate_buffers_for_all_tiles_org(*scheme, group_index, threads);
//             if (sc != StatusCode::Success) return sc;
//         }
//         #endif


//         // 4) Initialize inverse tiles’ diagonals to identity on the diagonal tiles
//         {
//             StatusCode sc = init_inverse_identity_on_diagonals_org(*scheme);
//             if (sc != StatusCode::Success) return sc;
//         }

//     // one chunk
//     } else if ((*call_info)->factorization_variant == 1) {
        
//         std::cout << "TODO: add one-chunk variant helpers here as needed." << std::endl;
//         exit(0);
//     // dense_dense
//     } else if ((*call_info)->factorization_variant == 2) {

//         std::cout << "TODO: add dense-dense variant helpers here as needed." << std::endl;
//         exit(0);
//     } else if ((*call_info)->factorization_variant == 3) {

//         StatusCode sc = StatusCode::Success;
// #ifdef STILES_SMART_COMPARE
//         sc = allocate_top_level_arrays(*scheme, group_index);
//         if (sc != StatusCode::Success) {
//             return sc;
//         }
// #else
//         (*scheme)->tileMetaCore = TileMemoryManager::allocate<TileMetaCore>((*scheme)->numActiveTiles, group_index);
//         if (!(*scheme)->tileMetaCore) {
//             std::fprintf(stderr, "ERROR: Memory allocation failed for denseTiles or tileMetaCore.\n");
//             return StatusCode::OutOfResources;
//         }
// #endif
//         compute_remainder(*scheme);
//         if (call_index == 0) sTiles::set_tile_extents(scheme);

// #ifdef STILES_SMART_COMPARE
//         const int threads = (*call_info)->num_cores;
//         sc = allocate_buffers_for_all_tiles_org(*scheme, group_index, threads);
//         if (sc != StatusCode::Success) {
//             return sc;
//         }
// #endif
//     }

//     return StatusCode::Success;
// }

//} // namespace preprocess
//} // namespace sTiles
