/**
 * @file    dpotri.cpp
 * @brief   Parallel selected inversion (DPOTRI) driver for sTiles.
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

#include <stdbool.h>
#include <omp.h>
#include <cstring>
#include <cmath>
#include <sstream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include "../common/stiles_types.hpp"
#include "../common/stiles_logger.hpp"
#include "../tile/meta.hpp" // for TileMetaCore and fast dense tile layout
// Sparse-path forward declarations — implementations live in sparse_dpotri.cpp.
namespace sTiles {
sTiles::StatusCode pthreads_sparse_dpotri(int global_index, TiledMatrix* scheme);
sTiles::StatusCode omp_sparse_dpotri     (int global_index, TiledMatrix* scheme);
} // namespace sTiles
#include "../control/common.h"
#include "../control/stiles_control.hpp"   // for omp_dep_tracker_t
#include "../compute/stiles_compute.hpp"
#ifdef STILES_GPU
    #include "memory_for_compute.hpp"
    #include "../gpu/compute_gpu.hpp"
#endif

// Forward declaration for control parameter access
extern "C" int* sTiles_get_params();

namespace { // Anonymous namespace to keep the helper private to this file
inline bool _validate_scheme(const TiledMatrix* scheme) {
    if (!scheme) {
        sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "TiledMatrix scheme cannot be null.");
        return false;
    }
    if (scheme->dim <= 0) {
        sTiles::Logger::info("Skipping inversion for zero or negative dimension matrix (N=", scheme->dim, ").");
        return false; // Return false to indicate no work should be done
    }
    return true;
}
}

namespace sTiles {

    /**
    * @brief Performs the core parallel triangular matrix inversion.
    *
    * This routine executes the parallel inversion of a Cholesky-factored matrix
    * using the pthreads runtime. It retrieves the thread-specific context and
    * dispatches the call to the low-level parallel function `stiles_pdtrtri`.
    *
    * @param[in] bind_index The context identifier for the current task.
    * @param[in] scheme A pointer to the TiledMatrix structure containing the factored matrix.
    * @return A `sTiles::StatusCode` indicating success or failure.
    */
    sTiles::StatusCode pthreads_dpotri(int bind_index, TiledMatrix *scheme) {

        if (!_validate_scheme(scheme)) {
            return scheme ? sTiles::StatusCode::Success : sTiles::StatusCode::IllegalValue;
        }

        stiles_context_t *stile = stiles_context_self(bind_index);

        if (!stile) {
            sTiles::Logger::fatal(sTiles::StatusCode::NotInitialized, "sTiles context is not initialized for this thread.");
            return sTiles::StatusCode::NotInitialized;
        }

        sTiles::parallel_call(stile, stiles_pdtrtri, scheme);

        return stile->ss_abort ? sTiles::StatusCode::ExecutionFailed
                               : sTiles::StatusCode::Success;
    }

    /**
    * @brief Performs the core parallel triangular matrix inversion using OpenMP.
    *
    * This is the OMP equivalent of pthreads_dpotri. It creates a parallel region
    * and each thread executes omp_pdtrtri which contains the non-SafeMode algorithm.
    *
    * @param[in] bind_index The context identifier (unused for OMP, kept for API consistency).
    * @param[in] scheme A pointer to the TiledMatrix structure containing the factored matrix.
    * @return A `sTiles::StatusCode` indicating success or failure.
    */
    sTiles::StatusCode omp_dpotri(int bind_index, TiledMatrix *scheme) {

        (void)bind_index; // unused for OMP, kept for API consistency

        if (!_validate_scheme(scheme)) {
            return scheme ? sTiles::StatusCode::Success : sTiles::StatusCode::IllegalValue;
        }

        omp_dep_tracker_t tracker_object;
        tracker_object.progress_table = nullptr;
        tracker_object.abort_flag.store(0, std::memory_order_relaxed);
        const int worldsize = scheme->num_cores;

        // NOTE: num_threads() clause overrides omp_set_num_threads().
        // omp_set_dynamic(0) freezes the team size at worldsize (no runtime
        // shrinkage). proc_bind(close) pins threads to the parent's place
        // list, matching the pthreads backend and the chol driver (omp_dpotrf).
        omp_set_dynamic(0);
        #pragma omp parallel num_threads(worldsize) proc_bind(close)
        {
            const int actual_threads = omp_get_num_threads();
            #pragma omp single
            {
                if (actual_threads != worldsize) {
                    sTiles::Logger::warning("[omp_dpotri] OMP parallel region: requested ", worldsize,
                                            " threads but got ", actual_threads,
                                            ". Check omp_set_max_active_levels() for nested parallelism.");
                }
            }

            sTiles::Process::omp_pdtrtri(scheme, &tracker_object, worldsize);
        }

        return tracker_object.abort_flag.load(std::memory_order_acquire)
                   ? sTiles::StatusCode::ExecutionFailed
                   : sTiles::StatusCode::Success;
    }

    /**
    * @brief Copies the Cholesky factor `L` to a backup buffer.
    *
    * This function iterates through all active tiles of the matrix and copies the
    * element data from the primary buffer (`dense_tiles`) to a backup buffer
    * (`saved_tiles`). This is necessary because the inversion process is destructive.
    *
    * @param[in] scheme A pointer to the TiledMatrix structure.
    */
    void copy_L_for_selinv(TiledMatrix **scheme) {
        if (!scheme || !(*scheme)) return;
#ifdef STILES_SAFEMODE
        // Safe-mode: use DenseTileSafeMode buffers
        for (int i = 0; i < (*scheme)->numActiveTiles; ++i) {
            const int width  = (*scheme)->dense_tiles[i].width;
            const int height = (*scheme)->dense_tiles[i].height;
            double* src  = (*scheme)->dense_tiles[i].elements;
            double* dest = (*scheme)->saved_tiles[i].elements;
            if (!src || !dest) {
                sTiles::Logger::error("Null tile buffer detected in copy_L_for_selinv (safe) at active tile index ", i);
                return;
            }
            std::memcpy(dest, src, static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(double));
        }
#else
        // Fast mode: prefer SmartTiles path if present; otherwise use raw DenseTile arrays
        TiledMatrix* S = *scheme;
#ifdef SMART_TILES
        if (S->facTiles) {
            // TODO: implement SmartTile backup/restore for selinv.
            // For now we skip copying and rely on downstream logic.
            sTiles::Logger::debug("copy_L_for_selinv: SmartTiles path not implemented; skipping copy.");
            return;
        }
#endif

        // Check for semisparse mode (chunkedDenseTiles present but denseTiles is NULL)
        if (!S->denseTiles && S->chunkedDenseTiles) {
            // Semisparse mode: backup chunkedDenseTiles to chunkedSavedTiles
            // Format: banded (kd+1)×h for diagonal (L factor), h×sa for off-diagonal
            // When inverse_storage_mode=1, chunkedSavedTiles matches chunkedDenseTiles format
            if (!S->chunkedSavedTiles || !S->tileMetaCore || !S->semisparseTileMetaCore) {
                sTiles::Logger::warning("copy_L_for_selinv: Semisparse backup buffers not available.");
                return;
            }
            for (int t = 0; t < S->numActiveTiles; ++t) {
                double* src = S->chunkedDenseTiles[t];
                double* dest = S->chunkedSavedTiles[t];
                // Skip null tiles (can happen for inactive tiles or sparse patterns)
                if (!src || !dest) continue;
                const sTiles::TileMetaCore& meta = S->tileMetaCore[t];
                const sTiles::SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[t];
                const int h = (meta.height > 0) ? meta.height : S->tile_size;
                const int w = (meta.width > 0) ? meta.width : S->tile_size;
                // Match allocation logic from chunked_tile_element_count exactly
                // to avoid buffer overflow when copying
                const bool meta_diag = (meta.row == meta.col);
                const bool diag_hint = (S->diagonal_bmapper && S->diagonal_bmapper[t]);
                const bool is_diag = diag_hint || meta_diag;
                // Compute size using same logic as chunked_tile_element_count
                int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
                if (is_diag && diag_cols <= 0) {
                    diag_cols = w;  // Fallback to full width for diagonal tiles
                }
                const int raw_cols = is_diag ? diag_cols : semi.sa;
                const int active_cols = (raw_cols > 0) ? raw_cols : 0;
                // Skip if no valid data to copy
                if (h <= 0 || active_cols <= 0) continue;
                const int copy_size = h * active_cols;
                std::memcpy(dest, src, static_cast<size_t>(copy_size) * sizeof(double));
            }
            return;
        }

        if (!S->denseTiles || !S->savedTiles) {
            sTiles::Logger::debug("copy_L_for_selinv: denseTiles or savedTiles not allocated; skipping copy.");
            return;
        }

        for (int t = 0; t < S->numActiveTiles; ++t) {
            const sTiles::TileMetaCore& m = S->tileMetaCore[t];
            const int h = (m.height > 0) ? m.height : S->tile_size;
            const int w = (m.width  > 0) ? m.width  : S->tile_size;
            double* src  = S->denseTiles[t];
            double* dest = S->savedTiles[t];
            if (!src || !dest) {
                // In sparse tiled mode, some tiles may legitimately be null - skip them
                continue;
            }
            std::memcpy(dest, src, static_cast<size_t>(h) * static_cast<size_t>(w) * sizeof(double));
        }
#endif
    }

    /**
    * @brief Re-initializes inverse tiles with identity matrix before inversion.
    *
    * This function must be called before each selinv to ensure inverse tiles
    * start from the identity matrix. Without this, subsequent calls would
    * start from previously computed values instead of identity.
    *
    * @param[in] scheme A pointer to the TiledMatrix structure.
    */
    void reinit_inverse_identity(TiledMatrix **scheme) {
        if (!scheme || !(*scheme)) {
            sTiles::Logger::warning("[reinit_inverse_identity] scheme is null");
            return;
        }
        TiledMatrix* S = *scheme;

        sTiles::Logger::debug("[reinit_inverse_identity] Called for scheme with numActiveTiles=", S->numActiveTiles,
                              " chunkedInverseTiles=", (S->chunkedInverseTiles ? "valid" : "NULL"),
                              " inverseTiles=", (S->inverseTiles ? "valid" : "NULL"),
                              " diagonal_bmapper=", (S->diagonal_bmapper ? "valid" : "NULL"));

#ifdef STILES_SAFEMODE
        // Safe-mode: re-init DenseTileSafeMode inverse tiles
        if (!S->inverse_tiles || !S->diagonal_bmapper) return;
        for (int t = 0; t < S->numActiveTiles; ++t) {
            DenseTileSafeMode& inv_tile = S->inverse_tiles[t];
            if (!inv_tile.elements) continue;
            const int h = inv_tile.height;
            const int w = inv_tile.width;
            if (h <= 0 || w <= 0) continue;
            // Zero out entire tile
            std::memset(inv_tile.elements, 0, static_cast<size_t>(h) * w * sizeof(double));
            // Set identity on diagonal tiles
            if (S->diagonal_bmapper[t]) {
                const int dsz = std::min(h, w);
                for (int i = 0; i < dsz; ++i) {
                    inv_tile.elements[i + i * h] = 1.0;
                }
            }
        }
#else
        // Fast mode: check for semisparse tiles first
        if (S->chunkedInverseTiles && S->tileMetaCore && S->diagonal_bmapper) {
            // Check inverse storage mode: params[7]
            //   0 = dense: all inverse tiles are full h×w
            //   1 = semisparse: diagonal tiles are DENSE, off-diagonal use active-cols format
            int* params = sTiles_get_params();
            const int inverse_storage_mode = params ? params[7] : 0;
            const bool semisparse_offdiag = (inverse_storage_mode == 1);

            int diag_count = 0;
            int total_zeroed = 0;
            for (int t = 0; t < S->numActiveTiles; ++t) {
                double* inv = S->chunkedInverseTiles[t];
                if (!inv) continue;
                const sTiles::TileMetaCore& meta = S->tileMetaCore[t];
                const int h = (meta.height > 0) ? meta.height : S->tile_size;
                const int w = (meta.width > 0) ? meta.width : S->tile_size;
                if (h <= 0 || w <= 0) continue;

                const bool meta_diag = (meta.row == meta.col);
                const bool lapack_diag = S->diagonal_bmapper[t];
                const bool is_diag = meta_diag || lapack_diag;

                if (is_diag) {
                    // Diagonal tiles are always dense (h × w)
                    std::memset(inv, 0, static_cast<size_t>(h) * w * sizeof(double));
                    const int dsz = std::min(h, w);
                    for (int i = 0; i < dsz; ++i) {
                        inv[i + i * h] = 1.0;
                    }
                    diag_count++;
                } else if (semisparse_offdiag && S->semisparseTileMetaCore) {
                    // Off-diagonal tile: active-columns format h × sa
                    const sTiles::SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[t];
                    const int sa = semi.sa;
                    if (sa > 0) {
                        const std::size_t elems = static_cast<std::size_t>(h) * static_cast<std::size_t>(sa);
                        std::memset(inv, 0, elems * sizeof(double));
                    }
                } else {
                    // Dense off-diagonal tile (inverse_storage_mode=0)
                    std::memset(inv, 0, static_cast<size_t>(h) * w * sizeof(double));
                }
                total_zeroed++;
            }
            sTiles::Logger::debug("[reinit_inverse_identity] Semisparse: zeroed ", total_zeroed,
                                  " tiles, set identity on ", diag_count, " diagonal tiles",
                                  " (inverse_storage_mode=", inverse_storage_mode, ")");
            return;
        }

        // Handle semisparse with missing diagonal_bmapper (e.g., single-tile fallback case)
        if (S->chunkedInverseTiles && S->tileMetaCore && !S->diagonal_bmapper) {
            // For single tile case, the tile is always diagonal
            if (S->numActiveTiles == 1 && S->dimTiledMatrix == 1) {
                double* inv = S->chunkedInverseTiles[0];
                if (inv) {
                    const sTiles::TileMetaCore& meta = S->tileMetaCore[0];
                    const int h = (meta.height > 0) ? meta.height : S->tile_size;
                    const int w = (meta.width > 0) ? meta.width : S->tile_size;
                    // Zero out and set identity
                    std::memset(inv, 0, static_cast<size_t>(h) * w * sizeof(double));
                    const int dsz = std::min(h, w);
                    for (int i = 0; i < dsz; ++i) {
                        inv[i + i * h] = 1.0;
                    }
                    sTiles::Logger::debug("[reinit_inverse_identity] Single-tile fallback: initialized identity on tile 0");
                    return;
                }
            }
            sTiles::Logger::warning("[reinit_inverse_identity] chunkedInverseTiles present but diagonal_bmapper is NULL!");
        }

        // Dense mode: re-init inverseTiles
        if (S->inverseTiles && S->tileMetaCore) {
            const int variant = S->factorization_variant;
            const int N = S->dimTiledMatrix;

            // Zero all tiles first (common to all variants)
            for (int t = 0; t < S->numActiveTiles; ++t) {
                double* inv = S->inverseTiles[t];
                if (!inv) continue;
                const sTiles::TileMetaCore& meta = S->tileMetaCore[t];
                const int h = (meta.height > 0) ? meta.height : S->tile_size;
                const int w = (meta.width > 0) ? meta.width : S->tile_size;
                if (h <= 0 || w <= 0) continue;
                std::memset(inv, 0, static_cast<size_t>(h) * w * sizeof(double));
            }

            // Variant 2: use upper triangular packed formula for diagonal tiles
            if (variant == 2 && N > 1) {
                // Set identity on diagonal tiles using variant 2 formula
                // Formula: i * N - (i * (i - 1)) / 2 + (j - i), for diagonal j = i
                for (int i = 0; i < N; ++i) {
                    const int idx = i * N - (i * (i - 1)) / 2;
                    if (idx < 0 || idx >= S->numActiveTiles) continue;
                    double* inv = S->inverseTiles[idx];
                    if (!inv) continue;
                    const sTiles::TileMetaCore& meta = S->tileMetaCore[idx];
                    const int h = (meta.height > 0) ? meta.height : S->tile_size;
                    for (int ii = 0; ii < h; ++ii) {
                        inv[ii + ii * h] = 1.0;
                    }
                }
                sTiles::Logger::debug("[reinit_inverse_identity] Variant 2: reinitialized ", N, " diagonal tiles");
                return;
            }

            // Variant 0/3: use mapper or diagonal_mapper for diagonal tiles
            if ((variant == 0 || variant == 3) && N > 1) {
                // Try mapper first (for fast mode)
                if (S->mapper.valid()) {
                    for (int i = 0; i < N; ++i) {
                        const int idx = S->mapper.map_ij(i, i, N);
                        if (idx < 0 || idx >= S->numActiveTiles) continue;
                        double* inv = S->inverseTiles[idx];
                        if (!inv) continue;
                        const sTiles::TileMetaCore& meta = S->tileMetaCore[idx];
                        const int h = (meta.height > 0) ? meta.height : S->tile_size;
                        for (int ii = 0; ii < h; ++ii) {
                            inv[ii + ii * h] = 1.0;
                        }
                    }
                    sTiles::Logger::debug("[reinit_inverse_identity] Variant 0: reinitialized diagonal tiles using mapper");
                    return;
                }
                // Fallback to diagonal_mapper if available
                if (S->diagonal_mapper) {
                    for (int i = 0; i < N; ++i) {
                        const int idx = S->diagonal_mapper[i];
                        if (idx < 0 || idx >= S->numActiveTiles) continue;
                        double* inv = S->inverseTiles[idx];
                        if (!inv) continue;
                        const sTiles::TileMetaCore& meta = S->tileMetaCore[idx];
                        const int h = (meta.height > 0) ? meta.height : S->tile_size;
                        for (int ii = 0; ii < h; ++ii) {
                            inv[ii + ii * h] = 1.0;
                        }
                    }
                    sTiles::Logger::debug("[reinit_inverse_identity] Variant 0: reinitialized diagonal tiles using diagonal_mapper");
                    return;
                }
            }

            // Original code path for sparse variants with diagonal_bmapper
            if (S->diagonal_bmapper) {
                for (int t = 0; t < S->numActiveTiles; ++t) {
                    double* inv = S->inverseTiles[t];
                    if (!inv) continue;
                    const sTiles::TileMetaCore& meta = S->tileMetaCore[t];
                    const int h = (meta.height > 0) ? meta.height : S->tile_size;
                    const int w = (meta.width > 0) ? meta.width : S->tile_size;
                    if (h <= 0 || w <= 0) continue;
                    // Zero out entire tile
                    std::memset(inv, 0, static_cast<size_t>(h) * w * sizeof(double));
                    // Set identity on diagonal tiles
                    const bool meta_diag = (meta.row == meta.col);
                    const bool lapack_diag = S->diagonal_bmapper[t];
                    if (meta_diag || lapack_diag) {
                        const int dsz = std::min(h, w);
                        for (int i = 0; i < dsz; ++i) {
                            inv[i + i * h] = 1.0;
                        }
                    }
                }
                return;
            }
        }

        // If we get here, no inverse tiles were found to reinitialize
        sTiles::Logger::warning("[reinit_inverse_identity] No inverse tiles found to reinitialize!");
#endif
    }

    /**
    * @brief Restores the original Cholesky factor `L` from the backup buffer.
    *
    * This function performs the reverse operation of `copy_L_for_selinv`, copying
    * the saved tile data from the backup buffer back to the primary buffer.
    * This restores the matrix to its state before the destructive inversion.
    *
    * @param[in] scheme A pointer to the TiledMatrix structure.
    */
    void copy_saved_L_back_to_L(TiledMatrix **scheme) {
        if (!scheme || !(*scheme)) return;
#ifdef STILES_SAFEMODE
        // Safe-mode: restore DenseTileSafeMode
        for (int i = 0; i < (*scheme)->numActiveTiles; ++i) {
            const int width  = (*scheme)->dense_tiles[i].width;
            const int height = (*scheme)->dense_tiles[i].height;
            double* src  = (*scheme)->saved_tiles[i].elements;
            double* dest = (*scheme)->dense_tiles[i].elements;
            if (!src || !dest) {
                sTiles::Logger::error("Null tile buffer detected in copy_saved_L_back_to_L (safe) at active tile index ", i);
                return;
            }
            std::memcpy(dest, src, static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(double));
        }
#else
        // Fast mode: prefer SmartTiles path if present; otherwise restore DenseTile arrays
        TiledMatrix* S = *scheme;
#ifdef SMART_TILES
        if (S->facTiles) {
            // TODO: implement SmartTile backup/restore for selinv.
            // For now we skip restore and rely on downstream logic.
            sTiles::Logger::debug("copy_saved_L_back_to_L: SmartTiles path not implemented; skipping restore.");
            return;
        }
#endif

        // Check for semisparse mode (chunkedDenseTiles present but denseTiles is NULL)
        if (!S->denseTiles && S->chunkedDenseTiles) {
            // Semisparse mode: restore chunkedDenseTiles from chunkedSavedTiles
            // Note: restore the same data that was backed up (matching copy_L_for_selinv)
            if (!S->chunkedSavedTiles || !S->tileMetaCore || !S->semisparseTileMetaCore) {
                sTiles::Logger::warning("copy_saved_L_back_to_L: Semisparse restore buffers not available.");
                return;
            }
            for (int t = 0; t < S->numActiveTiles; ++t) {
                double* src = S->chunkedSavedTiles[t];
                double* dest = S->chunkedDenseTiles[t];
                // Skip null tiles (can happen for inactive tiles or sparse patterns)
                if (!src || !dest) continue;
                const sTiles::TileMetaCore& meta = S->tileMetaCore[t];
                const sTiles::SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[t];
                const int h = (meta.height > 0) ? meta.height : S->tile_size;
                const int w = (meta.width > 0) ? meta.width : S->tile_size;
                // Match allocation logic from chunked_tile_element_count exactly
                // to avoid buffer overflow when copying
                const bool meta_diag = (meta.row == meta.col);
                const bool diag_hint = (S->diagonal_bmapper && S->diagonal_bmapper[t]);
                const bool is_diag = diag_hint || meta_diag;
                // Compute size using same logic as chunked_tile_element_count
                int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
                if (is_diag && diag_cols <= 0) {
                    diag_cols = w;  // Fallback to full width for diagonal tiles
                }
                const int raw_cols = is_diag ? diag_cols : semi.sa;
                const int active_cols = (raw_cols > 0) ? raw_cols : 0;
                // Skip if no valid data to copy
                if (h <= 0 || active_cols <= 0) continue;
                const int copy_size = h * active_cols;
                std::memcpy(dest, src, static_cast<size_t>(copy_size) * sizeof(double));
            }
            return;
        }

        if (!S->denseTiles || !S->savedTiles) {
            sTiles::Logger::debug("copy_saved_L_back_to_L: denseTiles or savedTiles not allocated; skipping restore.");
            return;
        }

        for (int t = 0; t < S->numActiveTiles; ++t) {
            const sTiles::TileMetaCore& m = S->tileMetaCore[t];
            const int h = (m.height > 0) ? m.height : S->tile_size;
            const int w = (m.width  > 0) ? m.width  : S->tile_size;
            double* src  = S->savedTiles[t];
            double* dest = S->denseTiles[t];
            if (!src || !dest) {
                // In sparse tiled mode, some tiles may legitimately be null - skip them
                continue;
            }
            std::memcpy(dest, src, static_cast<size_t>(h) * static_cast<size_t>(w) * sizeof(double));
        }
#endif
    }

}

// =============================================================================
//  C-API Functions
// =============================================================================
extern "C" {

    /**
    * @brief [C-API] Executes a selected inversion on a Cholesky-factored matrix.
    *
    * This function performs an in-place inversion of the triangular factor L,
    * computes the inverse, and then restores the original L factor. It is the
    * primary entry point for selected inversion tasks.
    *
    * @param[in] group_index The index of the call group for this operation.
    * @param[in] call_index The index of the specific call within the group.
    * @param[in] obj An opaque handle (void**) to the `sTiles_object`.
    * @return An integer corresponding to a `sTiles::StatusCode`.
    */
    int sTiles_selinv(int group_index, int call_index, void **obj) {

        if (obj == nullptr || *obj == nullptr) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return static_cast<int>(sTiles::StatusCode::IllegalValue);
        }

        sTiles_object* s = static_cast<sTiles_object*>(*obj);

        double etime = omp_get_wtime();
        int global_index = s->schemes[0]->call_lookup_table[group_index][call_index];
        int global_index_mapped = global_index;

        sTiles_call *call = &s->stiles_groups[group_index].stiles_calls[call_index];

        if (call->mapped_group_index > -1) {
            int new_call_index  = call->mapped_call_index;
            int new_group_index = call->mapped_group_index;
            global_index_mapped = s->schemes[0]->call_lookup_table[new_group_index][new_call_index];
        }

        s->schemes[global_index_mapped]->timings[1] = 0.0;

        static int* stiles_control_params = sTiles_get_params();

        // ── Sparse-module dispatch (tile_type_mode == 2) ─────────────────────
        // Routes through tools/compute/sparse_dpotri.cpp using the same
        // pthreads/omp split as the tile path (param[8] = UseOMP).
        {
            TiledMatrix* sp_scheme = s->schemes[global_index_mapped];
            if (sp_scheme && sp_scheme->sparse_handle) {
                sTiles::StatusCode sp_status =
                    (stiles_control_params[8] == -1)
                        ? sTiles::pthreads_sparse_dpotri(global_index_mapped, sp_scheme)
                        : sTiles::omp_sparse_dpotri(global_index_mapped, sp_scheme);
                s->schemes[global_index_mapped]->timings[1] = omp_get_wtime() - etime;
                return (sp_status == sTiles::StatusCode::Success) ? 0 : -1;
            }
        }

        sTiles::StatusCode status;

        #ifdef STILES_GPU
        {
            TiledMatrix* scheme = s->schemes[global_index_mapped];
            const int tile_type_mode = stiles_control_params[3];  // 0=dense, 1=semisparse
            const int gpu_comparison_mode = stiles_control_params[10]; // 0=GPU-only, 1=GPU with CPU validation

            bool gpu_selinv_ready = scheme->use_gpu &&
                                    scheme->factorization_variant == 0 &&
                                    (tile_type_mode == 0 || tile_type_mode == 1) &&
                                    scheme->dense_tiles_gpu != nullptr &&
                                    scheme->inverse_tiles_gpu != nullptr &&
                                    scheme->tileMetaCore != nullptr &&
                                    scheme->numActiveTiles > 0;

            if (gpu_selinv_ready) {
                const bool semisparse_gpu = (tile_type_mode == 1);
                const char* mode_str = semisparse_gpu ? "Semisparse" : "Dense";

                // CPU validation path: run CPU selinv first, save results for comparison
                double cpu_time = 0.0;
                std::vector<std::vector<double>> cpu_inv_results;
                if (gpu_comparison_mode != 0) {
                    {
                        std::stringstream ss;
                        ss << "│   ↪ GPU Selinv Validation (" << mode_str << "): Running CPU selinv (reference)";
                        sTiles::Logger::timing(ss.str());
                    }

                    double cpu_start = omp_get_wtime();
                    sTiles::copy_L_for_selinv(&s->schemes[global_index_mapped]);
                    sTiles::reinit_inverse_identity(&s->schemes[global_index_mapped]);
                    if (stiles_control_params[8] == 1) {
                        sTiles::omp_dpotri(global_index, scheme);
                    } else {
                        sTiles::pthreads_dpotri(global_index, scheme);
                    }
                    cpu_time = omp_get_wtime() - cpu_start;

                    sTiles::Logger::timing("│   ↪ GPU Selinv Validation: Saving CPU inverse results");
                    cpu_inv_results.resize(scheme->numActiveTiles);
                    for (int i = 0; i < scheme->numActiveTiles; i++) {
                        double* cpu_tile = semisparse_gpu ? scheme->chunkedInverseTiles[i]
                                                          : scheme->inverseTiles[i];
                        if (!cpu_tile) continue;

                        int width, height;
                        if (semisparse_gpu) {
                            width = scheme->inverse_tiles_gpu[i].width;
                            height = scheme->inverse_tiles_gpu[i].height;
                        } else {
                            width = scheme->tileMetaCore[i].width;
                            height = scheme->tileMetaCore[i].height;
                            if (width <= 0) width = scheme->tile_size;
                            if (height <= 0) height = scheme->tile_size;
                        }
                        if (width <= 0 || height <= 0) continue;

                        size_t num_elements = static_cast<size_t>(width) * height;
                        cpu_inv_results[i].resize(num_elements);
                        std::memcpy(cpu_inv_results[i].data(), cpu_tile,
                                   num_elements * sizeof(double));
                    }
                    sTiles::copy_saved_L_back_to_L(&s->schemes[global_index_mapped]);
                }

                // Run GPU selinv (handles save/restore L and init identity internally on GPU)
                {
                    std::stringstream ss;
                    ss << "│   ↪ GPU Selinv (" << mode_str << "): Running GPU selinv";
                    sTiles::Logger::timing(ss.str());
                }
                double gpu_start = omp_get_wtime();
                sTiles::gpu::selinv_gpu(scheme);
                double gpu_time = omp_get_wtime() - gpu_start;

                // Copy GPU inverse results to CPU
                double copy_from_gpu_start = omp_get_wtime();
                sTiles::gpu::copy_inverse_to_cpu(scheme);
                double copy_from_gpu_time = omp_get_wtime() - copy_from_gpu_start;

                // Report timing
                double gpu_total_time = gpu_time + copy_from_gpu_time;
                {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(6);
                    ss << "│   ↪ GPU Selinv :: GPU compute = " << gpu_time << " s";
                    sTiles::Logger::timing(ss.str());
                }
                {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(6);
                    ss << "│   ↪ GPU Selinv :: copy_from   = " << copy_from_gpu_time << " s";
                    sTiles::Logger::timing(ss.str());
                }
                {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(6);
                    ss << "│   ↪ GPU Selinv :: GPU total   = " << gpu_total_time << " s";
                    sTiles::Logger::timing(ss.str());
                }

                // CPU validation comparison (only when gpu_comparison_mode != 0)
                if (gpu_comparison_mode != 0) {
                    sTiles::Logger::timing("│   ↪ GPU Selinv Validation: Comparing results");
                    double max_abs_err = 0.0;
                    double max_rel_err = 0.0;
                    int bad_count = 0;
                    const double tol = 1e-8;

                    for (int i = 0; i < scheme->numActiveTiles; i++) {
                        double* gpu_tile = semisparse_gpu ? scheme->chunkedInverseTiles[i]
                                                          : scheme->inverseTiles[i];
                        if (gpu_tile && !cpu_inv_results[i].empty()) {
                            size_t num_elements = cpu_inv_results[i].size();
                            double* gpu_data = gpu_tile;
                            double* cpu_data = cpu_inv_results[i].data();

                            for (size_t j = 0; j < num_elements; j++) {
                                double abs_err = std::fabs(gpu_data[j] - cpu_data[j]);
                                double rel_err = (std::fabs(cpu_data[j]) > tol)
                                                 ? abs_err / std::fabs(cpu_data[j]) : abs_err;

                                if (abs_err > max_abs_err) max_abs_err = abs_err;
                                if (rel_err > max_rel_err) max_rel_err = rel_err;
                                if (abs_err > tol) bad_count++;
                            }
                        }
                    }

                    {
                        std::stringstream ss;
                        ss << std::fixed << std::setprecision(6);
                        ss << "│   ↪ GPU Selinv Validation :: CPU time    = " << cpu_time << " s";
                        sTiles::Logger::timing(ss.str());
                    }
                    {
                        std::stringstream ss;
                        ss << std::fixed << std::setprecision(6);
                        double speedup = (gpu_total_time > 0) ? cpu_time / gpu_total_time : 0.0;
                        ss << "│   ↪ GPU Selinv Validation :: Speedup     = " << std::setprecision(2) << speedup << "x";
                        if (speedup > 1.0) ss << " (GPU faster)";
                        else if (speedup < 1.0) ss << " (CPU faster)";
                        sTiles::Logger::timing(ss.str());
                    }

                    if (bad_count == 0) {
                        if (max_abs_err == 0.0) {
                            sTiles::Logger::timing("│   ↪ GPU Selinv Validation: CPU and GPU results [EXACT MATCH]");
                        } else {
                            std::stringstream ss_pass;
                            ss_pass << "│   ↪ GPU Selinv Validation: Max abs error = " << std::scientific << max_abs_err
                                   << ", Max rel error = " << max_rel_err << " [PASS]";
                            sTiles::Logger::timing(ss_pass.str());
                        }
                    } else {
                        std::stringstream ss_fail;
                        ss_fail << "│   ↪ GPU Selinv Validation: Max abs error = " << std::scientific << max_abs_err
                               << ", Bad elements = " << bad_count << " [FAIL]";
                        sTiles::Logger::timing(ss_fail.str());
                    }
                }

                s->schemes[global_index_mapped]->timings[1] = omp_get_wtime() - etime;
                return static_cast<int>(sTiles::StatusCode::Success);
            }
        }
        #endif

        // CPU path: save L, init identity, compute, restore
        sTiles::copy_L_for_selinv(&s->schemes[global_index_mapped]);
        sTiles::reinit_inverse_identity(&s->schemes[global_index_mapped]);

        if (stiles_control_params[8] == 1) {
            status = sTiles::omp_dpotri(global_index, s->schemes[global_index_mapped]);
        } else {
            status = sTiles::pthreads_dpotri(global_index, s->schemes[global_index_mapped]);
        }

        #ifdef STILES_GPU
        // Only sync GPU→CPU if GPU was actually used for inverse tiles
        if (s->schemes[global_index_mapped]->use_gpu &&
            s->schemes[global_index_mapped]->inverse_tiles_gpu != nullptr) {
            SyncInverseResult_GpuToCpu_Serial(global_index_mapped, s->schemes);
        }
        #endif

        sTiles::copy_saved_L_back_to_L(&s->schemes[global_index_mapped]);
        s->schemes[global_index_mapped]->timings[1] = omp_get_wtime() - etime;

        return static_cast<int>(status);
    }

    /**
    * @brief [C-API] Retrieves the execution time of a specific selected inversion.
    *
    * After an inversion is performed via `sTiles_selinv`, this function can be
    * used to get the wall-clock time it took to complete.
    *
    * @param[in] group_index The index of the call group.
    * @param[in] call_index The index of the specific call within the group.
    * @param[in] obj An opaque handle (void**) to the `sTiles_object`.
    * @return The execution time in seconds, or a negative value on error.
    */
    double sTiles_get_selinv_timing(int group_index, int call_index, void** obj) {

        if (obj == nullptr || *obj == nullptr) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return -1.0;
        }
        sTiles_object* s = (sTiles_object*)(*obj);
        if (!s->schemes || !s->schemes[0] || !s->schemes[0]->call_lookup_table) {
            sTiles::Logger::error(sTiles::StatusCode::Unallocated, "sTiles object internals are not initialized.");
            return -1.0;
        }

        int global_index = s->schemes[0]->call_lookup_table[group_index][call_index];
        int global_index_mapped = global_index;

        if (group_index >= 0 && call_index >= 0 &&  s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index > -1) {
            
            int new_call_index  = s->stiles_groups[group_index].stiles_calls[call_index].mapped_call_index;
            int new_group_index = s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index;
            global_index_mapped = s->schemes[0]->call_lookup_table[new_group_index][new_call_index];
        }

        return s->schemes[global_index_mapped]->timings[1];
    }   

    /**
    * @brief [C-API] Prints the timing results for all selected inversions in a group.
    *
    * A utility function to print a formatted summary of the selinv
    * times for every call within a given group.
    *
    * @param[in] group_index The group to print timings for.
    * @param[in] obj An opaque handle (void**) to the `sTiles_object`.
    */
    void sTiles_print_selinv_timings(int group_index, void** obj) {

        if (obj == nullptr || *obj == nullptr) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return;
        }

        sTiles_object* s = *((sTiles_object**)obj);
        int num_calls = s->stiles_groups[group_index].num_calls;
        
        sTiles::Logger::timing("--- Selected Inversion Timings for Group ", group_index, " ---");
        for (int j = 0; j < num_calls; ++j) {
            double selinv_time = sTiles_get_selinv_timing(group_index, j, obj);
            if (selinv_time >= 0.0) {
                std::stringstream ss;
                ss << "Call " << j
                   << "  Time: " << std::fixed << std::setprecision(6)
                   << std::setw(12) << selinv_time << " s";
                sTiles::Logger::timing(ss.str());
            }
        }

    }

}
