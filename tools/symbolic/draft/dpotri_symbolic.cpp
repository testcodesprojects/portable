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
#include "../common/stiles_types.hpp"
#include "../common/stiles_logger.hpp"
#include "../tile/meta.hpp" // for TileMetaCore and fast dense tile layout
#include "../control/common.h"
#ifdef STILES_GPU
    #include "memory_for_compute.hpp"
#endif

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

        if (!scheme) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "TiledMatrix scheme cannot be null.");
            return sTiles::StatusCode::IllegalValue;
        }

        if (scheme->dim < 0) {
            sTiles::Logger::error("Matrix dimension (N) cannot be negative. Received value: ", scheme->dim);
            return sTiles::StatusCode::IllegalValue;
        }

        stiles_context_t *stile = stiles_context_self(bind_index);

        if (!stile) {
            sTiles::Logger::fatal(sTiles::StatusCode::NotInitialized, "sTiles context is not initialized for this thread.");
            return sTiles::StatusCode::NotInitialized;
        }

        sTiles::parallel_call(stile, stiles_pdtrtri, scheme);

        return sTiles::StatusCode::Success;
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
        if (S->facTiles) {
            // TODO: implement SmartTile backup/restore for selinv.
            // For now we skip copying and rely on downstream logic.
            sTiles::Logger::debug("copy_L_for_selinv: SmartTiles path not implemented; skipping copy.");
            return;
        }

        for (int t = 0; t < S->numActiveTiles; ++t) {
            const sTiles::TileMetaCore& m = S->tileMetaCore[t];
            const int h = (m.height > 0) ? m.height : S->tile_size;
            const int w = (m.width  > 0) ? m.width  : S->tile_size;
            double* src  = S->denseTiles[t];
            double* dest = S->savedTiles[t];
            if (!src || !dest) {
                sTiles::Logger::error("Null tile buffer detected in copy_L_for_selinv (fast) at tile ", t);
                return;
            }
            std::memcpy(dest, src, static_cast<size_t>(h) * static_cast<size_t>(w) * sizeof(double));
        }
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
        if (S->facTiles) {
            // TODO: implement SmartTile backup/restore for selinv.
            // For now we skip restore and rely on downstream logic.
            sTiles::Logger::debug("copy_saved_L_back_to_L: SmartTiles path not implemented; skipping restore.");
            return;
        }

        for (int t = 0; t < S->numActiveTiles; ++t) {
            const sTiles::TileMetaCore& m = S->tileMetaCore[t];
            const int h = (m.height > 0) ? m.height : S->tile_size;
            const int w = (m.width  > 0) ? m.width  : S->tile_size;
            double* src  = S->savedTiles[t];
            double* dest = S->denseTiles[t];
            if (!src || !dest) {
                sTiles::Logger::error("Null tile buffer detected in copy_saved_L_back_to_L (fast) at tile ", t);
                return;
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

        sTiles::copy_L_for_selinv(&s->schemes[global_index_mapped]);

        s->schemes[global_index_mapped]->timings[1] = 0.0;
        sTiles::StatusCode status = sTiles::pthreads_dpotri(global_index, s->schemes[global_index_mapped]);

        #ifdef STILES_GPU
            SyncInverseResult_GpuToCpu_Serial(global_index_mapped, s->schemes);
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
