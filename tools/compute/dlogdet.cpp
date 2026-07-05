/**
 * @file    dlogdet.cpp
 * @brief   Log-determinant computation from Cholesky factors for sTiles.
 *
 * @project sTiles (Sparse Tiles Library)
 * @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @version 3.0.0
 * @date 1 1 2026
 * @license Proprietary
 *
 * @note This file is part of the sTiles library, a proprietary software package.
 *       Redistribution or modification without prior permission is prohibited.
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
 *
 * @license
 * This software is proprietary and confidential. Unauthorized copying, distribution, or modification
 * of this software, via any medium, is strictly prohibited. Permission is granted to use the software
 * in binary form for non-commercial purposes only, provided that this copyright notice and permission
 * notice are included in all copies or substantial portions of the software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cmath>
#include "../control/common.h"
#include "../common/stiles_expiry.hpp"
#include "../include/stiles_process.h"
#include <iostream>
#include <iomanip>
#include <vector>

#include "../tile/meta.hpp"
// Sparse-path forward declarations — implementations live in sparse_dlogdet.cpp.
namespace sTiles {
double pthreads_sparse_dlogdet(TiledMatrix* scheme);
double omp_sparse_dlogdet     (TiledMatrix* scheme);
} // namespace sTiles

#ifdef STILES_GPU
#include "../gpu/compute_gpu.hpp"
#endif

#define ADD_SEMISPARSE

extern "C" int* sTiles_get_params();
static int* stiles_control_params = sTiles_get_params();

namespace sTiles {

    // Tolerance for numerical precision across platforms
    // macOS (both Accelerate and OpenBLAS) has floating-point accumulation differences
    // causing small negative diagonals in semisparse mode
#ifdef __APPLE__
    constexpr double SPD_TOLERANCE = 1e-9;  // macOS needs larger tolerance
#else
    constexpr double SPD_TOLERANCE = 1e-14; // Linux/MKL: strict check
#endif

    // --- helpers: map (i,i) to dense tile index ---

    static inline int upper_index_rowmajor_diag(int i, int T) {
        // i*(2*T - i - 1)/2 + i
        return i * (2 * T - i - 1) / 2 + i;
    }

    // Upper triangular index for tile (i,j) where i <= j
    static inline int upper_tri_index(int i, int j, int T) {
        return i * T - (i * (i - 1)) / 2 + (j - i);
    }

    // Detect if this is a dense variant (1 or 2)
    // Returns: 0 = not dense variant, 1 = variant 1 (single tile), 2 = variant 2 (scaled dense)
    static inline int detect_dense_variant(const TiledMatrix* A) {
        const int num_active = A->numActiveTiles;
        const int dim_tiled = A->dimTiledMatrix;
        const int tri_size = A->triangular_size;

        if (num_active == 1 && dim_tiled == 1) {
            return 1;  // Variant 1: single tile
        } else if (num_active == tri_size && tri_size > 0) {
            return 2;  // Variant 2: scaled dense
        }
        return 0;  // Not a dense variant (variant 0 or 3 - sparse)
    }

    static inline int map_diag_dense_index(const TiledMatrix* A, int i)
    {
#ifdef STILES_SAFEMODE
        if (A->tileIndexMapper) {
            const int tri = upper_index_rowmajor_diag(i, A->dimTiledMatrix);
            //std::cout << A->tileIndexMapper[tri] << ", ";
            return A->tileIndexMapper[tri];
        }
#else
        // Check for dense variants first (variant 1 and 2 don't have diagonal_mapper)
        if (!A->diagonal_mapper) {
            const int dense_variant = detect_dense_variant(A);
            if (dense_variant == 1) {
                // Variant 1: single tile at index 0
                return 0;
            } else if (dense_variant == 2) {
                // Variant 2: diagonal tile (i,i) at upper triangular index
                return upper_tri_index(i, i, A->dimTiledMatrix);
            }
            // No diagonal_mapper and not a dense variant - error
            return -1;
        }

        // Variant 0/3 (sparse): use diagonal_mapper
        //std::cout << A->diagonal_mapper[i] << ", ";
        return A->diagonal_mapper[i];

        // if (A->mapper.valid()) {
        //     return A->mapper.map_ij(i, i, A->dimTiledMatrix);
        // }
        // if (A->tileIndexMapper) {
        //     const int tri = upper_index_rowmajor_diag(i, A->dimTiledMatrix);
        //     return A->tileIndexMapper[tri];
        // }
#endif
        return -1;
    }

    /**
     * @brief Compute log(det(A)) from the triangular factor stored in A->dense_tiles.
     *
     * Uses the fast mapper when available, otherwise falls back to the
     * legacy dense tileIndexMapper.
     *
     * Returns NaN if any diagonal is non-positive or a required diagonal tile is inactive/missing.
     */
    static double wrapper_logdet_scheme(const TiledMatrix* A)
    {
#ifdef STILES_SAFEMODE
        if (!A || !A->dense_tiles) {
            sTiles::Logger::error("LogDet: scheme->dense_tiles is nuuuuuuuuuuuull.");
            return std::numeric_limits<double>::quiet_NaN();
        }
#else

        if(stiles_control_params[3] == 0){
            // Mode 0 (dense): use dense tiles for logdet
            if (!A || !A->denseTiles || !A->tileMetaCore) {
                sTiles::Logger::error("LogDet: fast-mode buffers (denseTiles/tileMetaCore) are not available.");
                return std::numeric_limits<double>::quiet_NaN();
            }

        }else if(stiles_control_params[3] == 1 || stiles_control_params[3] == 2){

            if (!A || !A->chunkedDenseTiles || !A->tileMetaCore || !A->semisparseTileMetaCore) {
                sTiles::Logger::error("LogDet: fast-mode buffers (chunkedDenseTiles/tileMetaCore/semisparseTileMetaCore) are not available.");
                return std::numeric_limits<double>::quiet_NaN();
            }

        }
#endif

#ifndef STILES_SAFEMODE
        // For dense variants 1 and 2, diagonal_mapper is null but that's OK
        const int dense_variant = detect_dense_variant(A);
        if (dense_variant == 0) {
            // Sparse variant: need diagonal_mapper
            if (!A->diagonal_mapper && !A->mapper.valid() && !A->tileIndexMapper) {
                sTiles::Logger::error("LogDet: no active mapper/tileIndexMapper/diagonal mapper available.");
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
#else

        if(stiles_control_params[3] == 0){

            if (!A->tileIndexMapper) {
                sTiles::Logger::error("LogDet: legacy tileIndexMapper is null.");
                return std::numeric_limits<double>::quiet_NaN();
            }

        }else if(stiles_control_params[3] == 1){

            if (!A->tileIndexMapper) {
                sTiles::Logger::error("LogDet: legacy tileIndexMapper is null for semisparse mode.");
                return std::numeric_limits<double>::quiet_NaN();
            }

        }
#endif

        const int T = A->dimTiledMatrix;
        double logDet = 0.0;

        for (int i = 0; i < T; ++i) {

            const int dense_idx = map_diag_dense_index(A, i);
            if (dense_idx < 0) {
                sTiles::Logger::error("LogDet: diagonal tile (", i, ",", i, ") is inactive or mapping failed.");
                return std::numeric_limits<double>::quiet_NaN();
            }

            #ifdef STILES_SAFEMODE
            const DenseTileSafeMode& tile = A->dense_tiles[dense_idx];
            const double* elements = tile.elements;
            const int ts = (tile.height > 0) ? tile.height : A->tile_size;
            #else

            const double* elements = nullptr;
            int ts = 0;
            int diag_cols = 0;

            if(stiles_control_params[3] == 0){
                // Mode 0 (dense): use dense tiles
                const TileMetaCore& meta = A->tileMetaCore[dense_idx];
                ts = (meta.height > 0) ? meta.height : A->tile_size;
                elements = A->denseTiles[dense_idx];
                if (!elements) {
                    sTiles::Logger::error("LogDet: denseTiles entry ", dense_idx, " is null.");
                    return std::numeric_limits<double>::quiet_NaN();
                }

            }else if(stiles_control_params[3] == 1 || stiles_control_params[3] == 2){

                // Extra validation for semisparse mode
                if (dense_idx < 0 || dense_idx >= A->numActiveTiles) {
                    sTiles::Logger::error("LogDet: dense_idx=", dense_idx, " out of range [0,", A->numActiveTiles, ") for diagonal tile i=", i);
                    return std::numeric_limits<double>::quiet_NaN();
                }

                const TileMetaCore& meta = A->tileMetaCore[dense_idx];
                const SemisparseTileMetaCore& semi = A->semisparseTileMetaCore[dense_idx];
                ts = (meta.height > 0) ? meta.height : A->tile_size;
                elements = A->chunkedDenseTiles[dense_idx];
                if (!elements) {
                    sTiles::Logger::error("LogDet: chunkedDenseTiles entry ", dense_idx, " is null for diagonal tile i=", i,
                                          " (dimTiledMatrix=", A->dimTiledMatrix, ", numActiveTiles=", A->numActiveTiles, ")");
                    return std::numeric_limits<double>::quiet_NaN();
                }

                // For banded diagonal tiles
                const int kd = semi.upper_bw;
                diag_cols = (kd >= 0) ? (kd + 1) : ts;

            }

            #endif

            for (int j = 0; j < ts; ++j) {

                double diagElement = 0.0;

                #ifdef STILES_SAFEMODE
                diagElement = elements[j + static_cast<size_t>(j) * ts];
                #else

                if(stiles_control_params[3] == 0){
                    // Mode 0 (dense): standard column-major storage
                    diagElement = elements[j + static_cast<size_t>(j) * ts];

                }else if(stiles_control_params[3] == 1 || stiles_control_params[3] == 2){

                    // Banded format: LAPACK upper banded storage
                    // For diagonal element (j,j): band = 0
                    // lapack_row = (kd+1) - 1 - 0 = kd
                    // offset = kd + diag_cols * j
                    const int lapack_row = diag_cols - 1;  // Main diagonal is at row kd
                    const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                             + static_cast<std::size_t>(diag_cols)
                                             * static_cast<std::size_t>(j);
                    diagElement = elements[offset];

                }

                #endif

                if (diagElement < -SPD_TOLERANCE) {
                    // Truly negative - matrix is not SPD
                    sTiles::Logger::error(
                        "Non-positive diagonal encountered in tile ", i,
                        " at local (", j, ",", j, ") value=", diagElement,
                        ". Matrix is not SPD.");
                    return std::numeric_limits<double>::quiet_NaN();
                }

                // Clamp near-zero values (numerical noise) to tiny positive
                if (diagElement <= 0.0) {
                    diagElement = 1e-14;
                }

                logDet += std::log(diagElement);
            }
        }

        return 2.0 * logDet;
    }

    /**
     * @brief Compute logdet for dense variant 1 (single tile covering entire matrix)
     */
    static double logdet_dense_variant1(const TiledMatrix* A)
    {
        if (!A || !A->denseTiles || !A->denseTiles[0]) {
            sTiles::Logger::error("LogDet variant1: denseTiles[0] is null.");
            return std::numeric_limits<double>::quiet_NaN();
        }

        const int N = A->dim;
        const double* elements = A->denseTiles[0];
        double logDet = 0.0;

        // Single tile: column-major storage, diagonal at elements[j + j*N]
        for (int j = 0; j < N; ++j) {
            double diagElement = elements[j + static_cast<std::size_t>(j) * N];
            if (diagElement < -SPD_TOLERANCE) {
                sTiles::Logger::error("Non-positive diagonal at (", j, ",", j, ") value=", diagElement, ". Matrix is not SPD.");
                return std::numeric_limits<double>::quiet_NaN();
            }
            if (diagElement <= 0.0) {
                diagElement = 1e-14;
            }
            logDet += std::log(diagElement);
        }

        return 2.0 * logDet;
    }

    /**
     * @brief Compute logdet for dense variant 2 (scaled dense with upper triangular tiles)
     */
    static double logdet_dense_variant2(const TiledMatrix* A)
    {
        if (!A || !A->denseTiles || !A->tileMetaCore) {
            sTiles::Logger::error("LogDet variant2: denseTiles or tileMetaCore is null.");
            return std::numeric_limits<double>::quiet_NaN();
        }

        const int T = A->dimTiledMatrix;
        const int tile_size = A->tile_size;
        double logDet = 0.0;

        // Iterate through diagonal tiles (i,i) for i = 0..T-1
        for (int i = 0; i < T; ++i) {
            // Upper triangular index for diagonal tile (i,i)
            const int dense_idx = upper_tri_index(i, i, T);

            if (dense_idx < 0 || dense_idx >= A->numActiveTiles) {
                sTiles::Logger::error("LogDet variant2: invalid tile index ", dense_idx);
                return std::numeric_limits<double>::quiet_NaN();
            }

            const double* elements = A->denseTiles[dense_idx];
            if (!elements) {
                sTiles::Logger::error("LogDet variant2: denseTiles[", dense_idx, "] is null.");
                return std::numeric_limits<double>::quiet_NaN();
            }

            const TileMetaCore& meta = A->tileMetaCore[dense_idx];
            const int ts = (meta.height > 0) ? meta.height : tile_size;
            const int ld = ts;  // Leading dimension for this tile

            // Sum log of diagonal elements in this tile
            for (int j = 0; j < ts; ++j) {
                double diagElement = elements[j + static_cast<std::size_t>(j) * ld];
                if (diagElement < -SPD_TOLERANCE) {
                    sTiles::Logger::error("Non-positive diagonal in tile ", i, " at (", j, ",", j, ") value=", diagElement);
                    return std::numeric_limits<double>::quiet_NaN();
                }
                if (diagElement <= 0.0) {
                    diagElement = 1e-14;
                }
                logDet += std::log(diagElement);
            }
        }

        return 2.0 * logDet;
    }

}

// =============================================================================
//  C-API Functions
// =============================================================================
extern "C" {

    double sTiles_get_logdet(int group_index, int call_index, void** obj)
    {
        if (!obj || !*obj) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return -1.0;
        }

        sTiles_object* s = (sTiles_object*)(*obj);
        if (!s->schemes || !s->schemes[0] || !s->schemes[0]->call_lookup_table) {
            sTiles::Logger::error(sTiles::StatusCode::Unallocated, "sTiles object internals (schemes/lookup table) are not initialized.");
            return -1.0;
        }

        // Get the factorization variant from the call info
        const int variant = s->stiles_groups[group_index].stiles_calls[call_index].factorization_variant;

        int global_index = s->schemes[0]->call_lookup_table[group_index][call_index];
        int global_index_mapped = global_index;

        // Optional remapping
        if (group_index >= 0 && call_index >= 0 &&
            s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index > -1)
        {
            const int new_call_index  = s->stiles_groups[group_index].stiles_calls[call_index].mapped_call_index;
            const int new_group_index = s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index;
            global_index_mapped = s->schemes[0]->call_lookup_table[new_group_index][new_call_index];
        }

        TiledMatrix* scheme = s->schemes[global_index_mapped];
        if (!scheme) {
            sTiles::Logger::error("Target TiledMatrix scheme at index ", global_index_mapped, " is null.");
            return -1.0;
        }

        // ── Sparse-module dispatch (tile_type_mode == 2) ─────────────────────
        // Routes through tools/compute/sparse_dlogdet.cpp.
        if (scheme->sparse_handle) {
            int* params = sTiles_get_params();
            return (params && params[8] == -1)
                       ? sTiles::pthreads_sparse_dlogdet(scheme)
                       : sTiles::omp_sparse_dlogdet(scheme);
        }

        // Dense variants 1 and 2: use specialized functions
        if (variant == 1) {
            return STILES_VALIDATED_CACHED(sTiles::logdet_dense_variant1(scheme));
        } else if (variant == 2) {
            return STILES_VALIDATED_CACHED(sTiles::logdet_dense_variant2(scheme));
        }

        // GPU path: compute logdet directly from factored tiles on GPU (no D2H copy of L)
        #ifdef STILES_GPU
        if (scheme->use_gpu && scheme->dense_tiles_gpu && scheme->diagonal_mapper) {
            return STILES_VALIDATED_CACHED(sTiles::gpu::logdet_gpu(scheme));
        }
        #endif

        // Defensive checks for required fields depending on backend (variants 0, 3, etc.)
#ifdef STILES_SAFEMODE
        if (!scheme->dense_tiles) {
            sTiles::Logger::error("LogDet: scheme->dense_tiles is null.");
            return -1.0;
        }

        if (!scheme->diagonal_mapper && !scheme->mapper.valid() && !scheme->tileIndexMapper) {
            sTiles::Logger::error("LogDet: no active mapper/tileIndexMapper/diagonal mapper available.");
            return -1.0;
        }
#else

        if(stiles_control_params[3] == 0){

            if (!scheme->denseTiles || !scheme->tileMetaCore) {
                sTiles::Logger::error("LogDet: fast-mode buffers (denseTiles/tileMetaCore) are not available.");
                return -1.0;
            }

        }else if(stiles_control_params[3] == 1){

            if (!scheme->chunkedDenseTiles || !scheme->tileMetaCore || !scheme->semisparseTileMetaCore) {
                sTiles::Logger::error("LogDet: fast-mode buffers (chunkedDenseTiles/tileMetaCore/semisparseTileMetaCore) are not available.");
                return -1.0;
            }

        }

#endif



        return STILES_VALIDATED_CACHED(sTiles::wrapper_logdet_scheme(scheme));
    }

    void sTiles_print_logdets(int group_index, void** obj)
    {
        if (!obj || !*obj) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "Cannot print log-determinants: the provided sTiles object handle is null.");
            return;
        }

        sTiles_object* s = static_cast<sTiles_object*>(*obj);
        const int num_calls = s->stiles_groups[group_index].num_calls;

        sTiles::Logger::timing("│   • Log-determinant results (group ", group_index, ")");

        for (int j = 0; j < num_calls; ++j) {
            const double logdet_val = sTiles_get_logdet(group_index, j, obj);
            std::stringstream ss;
            ss << "│     ◦ Call " << j
               << "  logdet: "
               << std::fixed << std::setprecision(6)
               << logdet_val;
            sTiles::Logger::timing(ss.str());
        }
    }

} // extern "C"
