/**
 * @file    dpotrf.cpp
 * @brief   Tile-based Cholesky factorization (DPOTRF) orchestration for sTiles.
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


#include <stiles_process.h>
#include "../common/stiles_types.hpp"
#include "../common/stiles_logger.hpp"
#include "../control/common.h"
#include "../compute/stiles_compute.hpp"

#ifdef STILES_GPU
    #include "../gpu/gpu_tile_manager.hpp"
    #include "../gpu/compute_gpu.hpp"
#endif

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sstream>

// Debug flag for OMP - define this to enable debug output
// #define STILES_DEBUG_OMP

//==============================================================================
// sTiles_chol Execution Path Summary
//==============================================================================
//
// CPU (all variants, all tile types):
//   Variant 0, Tile Type 0 (dense)       — DONE (pthreads + OMP)
//   Variant 0, Tile Type 1 (semisparse)   — DONE (pthreads + OMP)
//   Variant 1 (single dense tile)         — DONE (pthreads + OMP)
//   Variant 2 (scaled dense / upper tri)  — DONE (pthreads + OMP)
//
// GPU (variant 0 only):
//   Tile Type 0 (dense), serial           — DONE (single-stream cuSOLVER/cuBLAS)
//   Tile Type 0 (dense), parallel         — DONE (multi-stream with CUDA events)
//   Tile Type 1 (semisparse), dense diag, serial   — DONE (single-stream cuSOLVER/cuBLAS)
//   Tile Type 1 (semisparse), dense diag, parallel  — DONE (multi-stream with CUDA events)
//   Tile Type 1 (semisparse), banded diag, serial   — DONE (single-stream, custom kernels <<<1,1>>>)
//   Tile Type 1 (semisparse), banded diag, parallel  — DONE (multi-stream with CUDA events)
//   Variant 1 on GPU                      — TODO
//   Variant 2 on GPU                      — TODO
//
// GPU Solve (in dtrsm.cpp):
//   Tile Type 0 (dense), forward/backward — DONE (single-stream)
//   Tile Type 1 (semisparse), forward/backward — DONE (single-stream)
//   Tile Type 0/1, fast path (task-based)  — DONE (single-stream, pre-collected tasks)
//   Tile Type 0 (dense), parallel          — DONE (multi-stream with step-counter atomics)
//   Tile Type 1 (semisparse), parallel     — DONE (multi-stream with step-counter atomics)
//
// GPU Inversion (in dpotri.cpp):
//   Tile Type 0 (dense), serial           — DONE (single-stream, variant 0)
//   Tile Type 0 (dense), parallel         — DONE (multi-stream with 2D CUDA events)
//   Tile Type 1 (semisparse), serial      — DONE (single-stream, variant 0)
//   Tile Type 1 (semisparse), parallel    — DONE (multi-stream with 2D CUDA events + scatter/gather)
//
//==============================================================================

#ifdef STILES_DEBUG_OMP
    #define OMP_DEBUG(msg) std::cerr << "[OMP_DEBUG] " << msg << std::endl
    #define OMP_DEBUG_SINGLE(msg) _Pragma("omp single") std::cerr << "[OMP_DEBUG] " << msg << std::endl
#else
    #define OMP_DEBUG(msg) ((void)0)
    #define OMP_DEBUG_SINGLE(msg) ((void)0)
#endif

// Access user-configured parameters (tile type is at index 3)
extern "C" int* sTiles_get_params();

// Sparse-path forward declarations — implementations live in sparse_dpotrf.cpp.
namespace sTiles {
sTiles::StatusCode pthreads_sparse_dpotrf(int global_index, TiledMatrix* scheme);
sTiles::StatusCode omp_sparse_dpotrf     (int global_index, TiledMatrix* scheme);
} // namespace sTiles

#ifdef STILES_COLLECT_CHOL_TILES
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include "../tile/meta.hpp"   // TileMetaCore
#endif

namespace { // Anonymous namespace to keep the helper private to this file
inline bool _validate_scheme(const TiledMatrix* scheme) {
    if (!scheme) {
        sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "TiledMatrix scheme cannot be null.");
        return false;
    }
    if (scheme->dim <= 0) {
        sTiles::Logger::info("Skipping factorization for zero or negative dimension matrix (N=", scheme->dim, ").");
        return false; // Return false to indicate no work should be done
    }
    return true;
}
}

namespace sTiles {

    /**
    * @brief Performs the core parallel Cholesky factorization using the pthreads runtime.
    *
    * This is the primary internal routine that executes the parallel factorization.
    * It retrieves the thread-specific context and dispatches the call to the
    * low-level parallel function `stiles_pdpotrf`.
    *
    * @param[in] bind_index The context identifier for the current task.
    * @param[in] scheme A pointer to the TiledMatrix structure containing the matrix
    *                   data and metadata for factorization.
    * @return A `sTiles::StatusCode` indicating success or failure of the operation.
    */
    sTiles::StatusCode pthreads_dpotrf(int bind_index, TiledMatrix *scheme) {
    
        if (!_validate_scheme(scheme)) {
            return scheme ? sTiles::StatusCode::Success : sTiles::StatusCode::IllegalValue;
        }

        stiles_context_t *stile = stiles_context_self(bind_index);
        if (!stile) {
            sTiles::Logger::fatal(sTiles::StatusCode::NotInitialized, "sTiles context is not initialized.");
            return sTiles::StatusCode::NotInitialized; 
        }

        sTiles::parallel_call(stile, sTiles::Process::pthreads_pdpotrf, scheme);

        return stile->ss_abort ? sTiles::StatusCode::ExecutionFailed
                               : sTiles::StatusCode::Success;
    }

    /**
    * @brief Acts as the primary internal dispatcher for a Cholesky factorization request.
    *
    * This function bridges the public C-API with the internal factorization wrappers. It
    * takes group and call indices, looks up the corresponding `TiledMatrix` scheme and
    * global context index, records timing, and then calls the main factorization routine.
    *
    * @param[in] group_index The index of the call group.
    * @param[in] call_index The index of the call within the specified group.
    * @param[in] s A pointer to the main `sTiles_object` instance, which contains
    *              all necessary schemes and metadata.
    * @return An integer corresponding to a `sTiles::StatusCode`.
    */
    /**
    * @brief Performs the core parallel Cholesky factorization using OpenMP.
    *
    * This is the OMP equivalent of pthreads_dpotrf. It creates a parallel region
    * and each thread executes omp_pdpotrf which contains the non-SafeMode algorithm.
    *
    * @param[in] bind_index The context identifier (unused for OMP, kept for API consistency).
    * @param[in] scheme A pointer to the TiledMatrix structure containing the matrix
    *                   data and metadata for factorization.
    * @return A `sTiles::StatusCode` indicating success or failure of the operation.
    */
    sTiles::StatusCode omp_dpotrf(int bind_index, TiledMatrix *scheme) {

        #ifdef STILES_DEBUG_OMP
        std::cerr << "[OMP_DEBUG] omp_dpotrf: entering" << std::endl;
        #endif

        if (!_validate_scheme(scheme)) {
            #ifdef STILES_DEBUG_OMP
            std::cerr << "[OMP_DEBUG] omp_dpotrf: validation failed" << std::endl;
            #endif
            return scheme ? sTiles::StatusCode::Success : sTiles::StatusCode::IllegalValue;
        }

        #ifdef STILES_DEBUG_OMP
        std::cerr << "[OMP_DEBUG] omp_dpotrf: scheme validated, dim=" << scheme->dim
                  << ", num_cores=" << scheme->num_cores
                  << ", variant=" << scheme->factorization_variant << std::endl;
        #endif

        omp_dep_tracker_t tracker_object;
        tracker_object.progress_table = nullptr;
        tracker_object.abort_flag.store(0, std::memory_order_relaxed);
        const int worldsize = scheme->num_cores;

        #ifdef STILES_DEBUG_OMP
        std::cerr << "[OMP_DEBUG] omp_dpotrf: tracker initialized, entering parallel region with "
                  << worldsize << " threads" << std::endl;
        #endif

        // NOTE: num_threads() clause overrides omp_set_num_threads().
        // omp_set_dynamic(0) freezes the team size at worldsize (no runtime
        // shrinkage). proc_bind(close) pins threads to the parent's place
        // list, which closes the gap with the pthreads backend — but only
        // when OMP_PLACES is set to physical cores (see make.inc / run_with_omp.sh).
        omp_set_dynamic(0);
        #pragma omp parallel num_threads(worldsize) proc_bind(close)
        {
            const int actual_threads = omp_get_num_threads();
            #pragma omp single
            {
                if (actual_threads != worldsize) {
                    std::cerr << "[sTiles WARNING] OMP parallel region: requested " << worldsize
                              << " threads but got " << actual_threads
                              << ". Check omp_set_max_active_levels() for nested parallelism." << std::endl;
                }
                #ifdef STILES_DEBUG_OMP
                std::cerr << "[OMP_DEBUG] omp_dpotrf: inside parallel region with " << actual_threads
                          << " threads, calling omp_pdpotrf" << std::endl;
                #endif
            }

            sTiles::Process::omp_pdpotrf(scheme, &tracker_object, worldsize);
        }

        #ifdef STILES_DEBUG_OMP
        std::cerr << "[OMP_DEBUG] omp_dpotrf: parallel region complete, returning" << std::endl;
        #endif
        return tracker_object.abort_flag.load(std::memory_order_acquire)
                   ? sTiles::StatusCode::ExecutionFailed
                   : sTiles::StatusCode::Success;
    }
}

//#define fixing

// =============================================================================
//  C-API Functions
// =============================================================================
extern "C" {

    /**
    * @brief [C-API] Executes a pre-configured Cholesky factorization.
    *
    * This is the public API function to trigger a Cholesky factorization. It takes a handle
    * to the sTiles object and indices that specify which pre-configured factorization
    * to perform.
    *
    * @param[in] group_index The index of the call group for this operation.
    * @param[in] call_index The index of the specific call within the group.
    * @param[in] obj An opaque handle (void**) to the `sTiles_object`.
    * @return An integer corresponding to a `sTiles::StatusCode`.
    */
    int sTiles_chol(int group_index, int call_index, void **obj) {

        if (!obj || !*obj) { // Also checks the dereferenced pointer
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return static_cast<int>(sTiles::StatusCode::IllegalValue);
        }

        sTiles_object* s = (sTiles_object*)(*obj);  // Cast from void* to internal type

        double etime = omp_get_wtime();
        int global_index = s->schemes[0]->call_lookup_table[group_index][call_index];

        int global_index_mapped = global_index;
        if (s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index > -1) {
            int new_call_index  = s->stiles_groups[group_index].stiles_calls[call_index].mapped_call_index;
            int new_group_index = s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index;
            global_index_mapped = s->schemes[0]->call_lookup_table[new_group_index][new_call_index];
        }

        s->schemes[global_index_mapped]->timings[0] = 0.0;


        // Get control parameters for variant and tile type
        static int* stiles_control_params = sTiles_get_params();
        TiledMatrix* scheme = s->schemes[global_index_mapped];

        // ── Sparse-module dispatch (tile_type_mode == 2) ─────────────────────
        // Routes through tools/compute/sparse_dpotrf.cpp. Defaults to OMP
        // (persistent thread pool); user can force pthreads by setting
        // param[8] = -1. The default param[8] = 0 routes to OMP because the
        // pthreads variant spawns std::thread per call and anti-scales on
        // fine-grained per-rank work.
        if (scheme->sparse_handle) {
            sTiles::StatusCode sp_status =
                (stiles_control_params[8] == -1)
                    ? sTiles::pthreads_sparse_dpotrf(global_index_mapped, scheme)
                    : sTiles::omp_sparse_dpotrf(global_index_mapped, scheme);
            s->schemes[global_index_mapped]->timings[0] = omp_get_wtime() - etime;
            return (sp_status == sTiles::StatusCode::Success) ? 0 : -1;
        }

        const int factorization_variant = scheme->factorization_variant;
        const int tile_type_mode = stiles_control_params[3];  // 0=dense, 1=semisparse
        const int gpu_comparison_mode = stiles_control_params[10]; // 0=GPU-only, 1=GPU with CPU validation

        // Select parallelization backend: OMP if param[8]==1, otherwise pthreads (default)
        sTiles::StatusCode status;

        #ifdef STILES_GPU
        // GPU factorization path for variant 0, tile_type 0 (dense)
        // Fast-mode only: uses denseTiles + tileMetaCore
        bool gpu_tiles_ready = scheme->use_gpu &&
                               factorization_variant == 0 &&
                               tile_type_mode == 0 &&  // dense tiles only
                               scheme->dense_tiles_gpu != nullptr &&
                               scheme->denseTiles != nullptr &&
                               scheme->tileMetaCore != nullptr &&
                               scheme->numActiveTiles > 0 &&
                               scheme->dense_tiles_gpu[0].x != nullptr;

        // Debug: print condition values (use timing so it always shows)
        sTiles::Logger::timing("│   ↪ GPU Check: use_gpu=", scheme->use_gpu,
                             ", variant=", factorization_variant,
                             ", tile_type=", tile_type_mode,
                             ", dense_tiles_gpu=", (scheme->dense_tiles_gpu != nullptr),
                             ", denseTiles=", (scheme->denseTiles != nullptr),
                             ", numActiveTiles=", scheme->numActiveTiles,
                             ", gpu[0].x=", (scheme->dense_tiles_gpu ? (scheme->dense_tiles_gpu[0].x != nullptr) : false));

        if (gpu_tiles_ready) {
            auto* persistent = static_cast<sTiles::gpu::GpuPersistentContext*>(scheme->gpu_persistent_ctx);

            // Copy tiles to GPU
            sTiles::Logger::timing("│   ↪ GPU Dense: Copying tiles to GPU");
            double copy_to_gpu_start = omp_get_wtime();
            sTiles::gpu::copy_tiles_to_gpu(scheme);
            double copy_to_gpu_time = omp_get_wtime() - copy_to_gpu_start;

            // CPU validation path: run CPU first, save results for comparison
            double cpu_time = 0.0;
            std::vector<std::vector<double>> cpu_results;
            if (gpu_comparison_mode != 0) {
                sTiles::Logger::timing("│   ↪ GPU Validation: Running CPU factorization");
                double cpu_start = omp_get_wtime();
                if (stiles_control_params[8] == 1) {
                    status = sTiles::omp_dpotrf(global_index, scheme);
                } else {
                    status = sTiles::pthreads_dpotrf(global_index, scheme);
                }
                cpu_time = omp_get_wtime() - cpu_start;

                sTiles::Logger::timing("│   ↪ GPU Validation: Saving CPU results");
                cpu_results.resize(scheme->numActiveTiles);
                for (int i = 0; i < scheme->numActiveTiles; i++) {
                    if (scheme->denseTiles[i]) {
                        int width = scheme->tileMetaCore[i].width;
                        int height = scheme->tileMetaCore[i].height;
                        if (width <= 0) width = scheme->tile_size;
                        if (height <= 0) height = scheme->tile_size;
                        size_t num_elements = static_cast<size_t>(width) * height;
                        cpu_results[i].resize(num_elements);
                        std::memcpy(cpu_results[i].data(), scheme->denseTiles[i],
                                   num_elements * sizeof(double));
                    }
                }
            }

            // Run GPU factorization
            sTiles::Logger::timing("│   ↪ GPU Dense: Running GPU factorization");
            double gpu_start = omp_get_wtime();
            sTiles::gpu::dpotrf_dense_gpu(scheme, *persistent);
            double gpu_time = omp_get_wtime() - gpu_start;

            // Copy GPU results back to CPU
            double copy_from_gpu_start = omp_get_wtime();
            sTiles::gpu::copy_factorization_to_cpu(scheme);
            double copy_from_gpu_time = omp_get_wtime() - copy_from_gpu_start;

            // Report timing
            double gpu_total_time = copy_to_gpu_time + gpu_time + copy_from_gpu_time;
            {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(6);
                ss << "│   ↪ GPU Dense :: GPU compute = " << gpu_time << " s";
                sTiles::Logger::timing(ss.str());
            }
            {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(6);
                ss << "│   ↪ GPU Dense :: copy_to     = " << copy_to_gpu_time << " s";
                sTiles::Logger::timing(ss.str());
            }
            {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(6);
                ss << "│   ↪ GPU Dense :: copy_from   = " << copy_from_gpu_time << " s";
                sTiles::Logger::timing(ss.str());
            }
            {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(6);
                ss << "│   ↪ GPU Dense :: GPU total   = " << gpu_total_time << " s";
                sTiles::Logger::timing(ss.str());
            }

            // CPU validation comparison (only when gpu_comparison_mode != 0)
            if (gpu_comparison_mode != 0) {
                sTiles::Logger::timing("│   ↪ GPU Validation: Comparing results");
                double max_abs_err = 0.0;
                double max_rel_err = 0.0;
                int bad_count = 0;
                const double tol = 1e-8;

                for (int i = 0; i < scheme->numActiveTiles; i++) {
                    if (scheme->denseTiles[i] && !cpu_results[i].empty()) {
                        size_t num_elements = cpu_results[i].size();
                        double* gpu_data = scheme->denseTiles[i];
                        double* cpu_data = cpu_results[i].data();

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
                    ss << "│   ↪ GPU Validation :: CPU time    = " << cpu_time << " s";
                    sTiles::Logger::timing(ss.str());
                }
                {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(6);
                    double speedup = (gpu_total_time > 0) ? cpu_time / gpu_total_time : 0.0;
                    ss << "│   ↪ GPU Validation :: Speedup     = " << std::setprecision(2) << speedup << "x";
                    if (speedup > 1.0) ss << " (GPU faster)";
                    else if (speedup < 1.0) ss << " (CPU faster)";
                    sTiles::Logger::timing(ss.str());
                }

                if (bad_count == 0) {
                    if (max_abs_err == 0.0) {
                        sTiles::Logger::timing("│   ↪ GPU Validation: CPU and GPU results [EXACT MATCH]");
                    } else {
                        std::stringstream ss_pass;
                        ss_pass << "│   ↪ GPU Validation: Max abs error = " << std::scientific << max_abs_err
                               << ", Max rel error = " << max_rel_err << " [PASS]";
                        sTiles::Logger::timing(ss_pass.str());
                    }
                } else {
                    std::stringstream ss_fail;
                    ss_fail << "│   ↪ GPU Validation: Max abs error = " << std::scientific << max_abs_err
                           << ", Bad elements = " << bad_count << " [FAIL]";
                    sTiles::Logger::timing(ss_fail.str());
                }
            }

            status = sTiles::StatusCode::Success;
        } else {
            // GPU factorization path for variant 0, tile_type 1 (semisparse)
            bool gpu_semisparse_ready = scheme->use_gpu &&
                                        factorization_variant == 0 &&
                                        tile_type_mode == 1 &&
                                        scheme->dense_tiles_gpu != nullptr &&
                                        scheme->chunkedDenseTiles != nullptr &&
                                        scheme->semisparseTileMetaCore != nullptr &&
                                        scheme->diagonal_bmapper != nullptr &&
                                        scheme->numActiveTiles > 0 &&
                                        scheme->dense_tiles_gpu[0].x != nullptr;

            if (gpu_semisparse_ready) {
                // GPU semisparse: uses multi-stream when GpuPersistentContext is initialized,
                // otherwise falls back to single-stream.
                {
                    bool is_multi = false;
                    if (scheme->gpu_persistent_ctx) {
                        auto* pctx = static_cast<sTiles::gpu::GpuPersistentContext*>(scheme->gpu_persistent_ctx);
                        is_multi = pctx->initialized && pctx->num_streams > 1 && pctx->semisparse_mode;
                    }
                    if (is_multi) {
                        sTiles::Logger::timing("│   ↪ GPU Semisparse: multi-stream parallel (" +
                            std::to_string(static_cast<sTiles::gpu::GpuPersistentContext*>(scheme->gpu_persistent_ctx)->num_streams) +
                            " streams)");
                    } else {
                        sTiles::Logger::timing("│   ↪ GPU Semisparse: single-stream");
                    }
                }

                // Copy semisparse tiles to GPU
                sTiles::Logger::timing("│   ↪ GPU Semisparse: Copying tiles to GPU");
                double copy_to_gpu_start = omp_get_wtime();
                sTiles::gpu::GpuTileManager::copy_tiles_to_gpu_semisparse(scheme);
                double copy_to_gpu_time = omp_get_wtime() - copy_to_gpu_start;

                // CPU validation path: run CPU first, save results for comparison
                double cpu_time = 0.0;
                std::vector<std::vector<double>> cpu_results;
                if (gpu_comparison_mode != 0) {
                    sTiles::Logger::timing("│   ↪ GPU Validation: Running CPU factorization");
                    double cpu_start = omp_get_wtime();
                    if (stiles_control_params[8] == 1) {
                        status = sTiles::omp_dpotrf(global_index, scheme);
                    } else {
                        status = sTiles::pthreads_dpotrf(global_index, scheme);
                    }
                    cpu_time = omp_get_wtime() - cpu_start;

                    sTiles::Logger::timing("│   ↪ GPU Validation: Saving CPU results");
                    cpu_results.resize(scheme->numActiveTiles);
                    for (int i = 0; i < scheme->numActiveTiles; i++) {
                        if (!scheme->chunkedDenseTiles[i]) continue;
                        int height = scheme->tileMetaCore[i].height;
                        if (height <= 0) height = scheme->tile_size;

                        if (scheme->diagonal_bmapper[i]) {
                            int kd = scheme->semisparseTileMetaCore[i].upper_bw;
                            int n = height;
                            int ld_banded = kd + 1;
                            size_t dense_elements = static_cast<size_t>(n) * n;
                            cpu_results[i].resize(dense_elements, 0.0);
                            const double* banded = scheme->chunkedDenseTiles[i];
                            for (int j = 0; j < n; ++j) {
                                for (int ii = std::max(0, j - kd); ii <= j; ++ii) {
                                    cpu_results[i][ii + j * n] = banded[(kd + ii - j) + j * ld_banded];
                                }
                            }
                        } else {
                            int sa = scheme->semisparseTileMetaCore[i].sa;
                            size_t num_elements = static_cast<size_t>(height) * sa;
                            if (num_elements > 0) {
                                cpu_results[i].resize(num_elements);
                                std::memcpy(cpu_results[i].data(), scheme->chunkedDenseTiles[i],
                                           num_elements * sizeof(double));
                            }
                        }
                    }
                }

                // Run GPU factorization
                sTiles::Logger::timing("│   ↪ GPU Semisparse: Running GPU factorization");
                double gpu_start = omp_get_wtime();
                sTiles::gpu::dpotrf_semisparse_gpu(scheme, nullptr);
                double gpu_time = omp_get_wtime() - gpu_start;

                // Copy GPU results back to CPU (chunkedDenseTiles)
                double copy_from_gpu_start = omp_get_wtime();
                sTiles::gpu::GpuTileManager::copy_tiles_from_gpu_semisparse(scheme);
                double copy_from_gpu_time = omp_get_wtime() - copy_from_gpu_start;

                // Report timing
                double gpu_total_time = copy_to_gpu_time + gpu_time + copy_from_gpu_time;
                {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(6);
                    ss << "│   ↪ GPU Semisparse :: GPU compute = " << gpu_time << " s";
                    sTiles::Logger::timing(ss.str());
                }
                {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(6);
                    ss << "│   ↪ GPU Semisparse :: copy_to     = " << copy_to_gpu_time << " s";
                    sTiles::Logger::timing(ss.str());
                }
                {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(6);
                    ss << "│   ↪ GPU Semisparse :: copy_from   = " << copy_from_gpu_time << " s";
                    sTiles::Logger::timing(ss.str());
                }
                {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(6);
                    ss << "│   ↪ GPU Semisparse :: GPU total   = " << gpu_total_time << " s";
                    sTiles::Logger::timing(ss.str());
                }

                // CPU validation comparison (only when gpu_comparison_mode != 0)
                if (gpu_comparison_mode != 0) {
                    sTiles::Logger::timing("│   ↪ GPU Validation: Comparing results");
                    double max_abs_err = 0.0;
                    double max_rel_err = 0.0;
                    int bad_count = 0;
                    const double tol = 1e-8;

                    for (int i = 0; i < scheme->numActiveTiles; i++) {
                        if (!scheme->chunkedDenseTiles[i] || cpu_results[i].empty()) continue;

                        int height = scheme->tileMetaCore[i].height;
                        if (height <= 0) height = scheme->tile_size;

                        if (scheme->diagonal_bmapper[i]) {
                            int kd = scheme->semisparseTileMetaCore[i].upper_bw;
                            int n = height;
                            int ld_banded = kd + 1;
                            double* tmp = scheme->workspaces[0]->aligned_tile();
                            std::memset(tmp, 0, static_cast<size_t>(n) * n * sizeof(double));
                            const double* banded = scheme->chunkedDenseTiles[i];
                            for (int j = 0; j < n; ++j) {
                                for (int ii = std::max(0, j - kd); ii <= j; ++ii) {
                                    tmp[ii + j * n] = banded[(kd + ii - j) + j * ld_banded];
                                }
                            }

                            size_t num_elements = cpu_results[i].size();
                            for (size_t j = 0; j < num_elements; j++) {
                                double abs_err = std::fabs(tmp[j] - cpu_results[i][j]);
                                double rel_err = (std::fabs(cpu_results[i][j]) > tol)
                                                 ? abs_err / std::fabs(cpu_results[i][j]) : abs_err;
                                if (abs_err > max_abs_err) max_abs_err = abs_err;
                                if (rel_err > max_rel_err) max_rel_err = rel_err;
                                if (abs_err > tol) bad_count++;
                            }
                        } else {
                            size_t num_elements = cpu_results[i].size();
                            double* gpu_data = scheme->chunkedDenseTiles[i];
                            for (size_t j = 0; j < num_elements; j++) {
                                double abs_err = std::fabs(gpu_data[j] - cpu_results[i][j]);
                                double rel_err = (std::fabs(cpu_results[i][j]) > tol)
                                                 ? abs_err / std::fabs(cpu_results[i][j]) : abs_err;
                                if (abs_err > max_abs_err) max_abs_err = abs_err;
                                if (rel_err > max_rel_err) max_rel_err = rel_err;
                                if (abs_err > tol) bad_count++;
                            }
                        }
                    }

                    {
                        std::stringstream ss;
                        ss << std::fixed << std::setprecision(6);
                        ss << "│   ↪ GPU Validation :: CPU time    = " << cpu_time << " s";
                        sTiles::Logger::timing(ss.str());
                    }
                    {
                        std::stringstream ss;
                        ss << std::fixed << std::setprecision(6);
                        double speedup = (gpu_total_time > 0) ? cpu_time / gpu_total_time : 0.0;
                        ss << "│   ↪ GPU Validation :: Speedup     = " << std::setprecision(2) << speedup << "x";
                        if (speedup > 1.0) ss << " (GPU faster)";
                        else if (speedup < 1.0) ss << " (CPU faster)";
                        sTiles::Logger::timing(ss.str());
                    }

                    if (bad_count == 0) {
                        if (max_abs_err == 0.0) {
                            sTiles::Logger::timing("│   ↪ GPU Semisparse: CPU and GPU results [EXACT MATCH]");
                        } else {
                            std::stringstream ss_pass;
                            ss_pass << "│   ↪ GPU Semisparse: Max abs error = " << std::scientific << max_abs_err
                                   << ", Max rel error = " << max_rel_err << " [PASS]";
                            sTiles::Logger::timing(ss_pass.str());
                        }
                    } else {
                        std::stringstream ss_fail;
                        ss_fail << "│   ↪ GPU Semisparse: Max abs error = " << std::scientific << max_abs_err
                               << ", Bad elements = " << bad_count << " [FAIL]";
                        sTiles::Logger::timing(ss_fail.str());
                    }
                }

                status = sTiles::StatusCode::Success;
            } else {
                // GPU was requested but this path is not implemented yet — print TODO
                if (scheme->use_gpu) {
                    if (factorization_variant != 0) {
                        sTiles::Logger::timing("│   ↪ [TODO] GPU variant ", factorization_variant,
                                             " not implemented. Falling back to CPU.");
                    } else if (tile_type_mode != 0 && tile_type_mode != 1) {
                        sTiles::Logger::timing("│   ↪ [TODO] GPU tile_type ", tile_type_mode,
                                             " not implemented. Falling back to CPU.");
                    } else {
                        sTiles::Logger::timing("│   ↪ GPU requested but tiles not ready (missing allocation?). Falling back to CPU.");
                    }
                }
                // Standard CPU path
                if (stiles_control_params[8] == 1) {
                    status = sTiles::omp_dpotrf(global_index, scheme);
                } else {
                    status = sTiles::pthreads_dpotrf(global_index, scheme);
                }
            }
        }
        #else
        // Non-GPU build: standard CPU path
        {
            if (stiles_control_params[8] == 1) {
                status = sTiles::omp_dpotrf(global_index, scheme);
            } else {
                status = sTiles::pthreads_dpotrf(global_index, scheme);
            }
        }
        #endif
        // Packing the factor into a flat CSC layout (scheme->L_values) has
        // been moved out of chol — see sTiles_packing(group, obj) for the
        // explicit phase. Decoupling lets the caller time the pack
        // separately from chol and call it (or not) based on whether
        // subsequent solves will benefit from the csc_dtrsm fast path.

        s->schemes[global_index_mapped]->timings[0] = omp_get_wtime() - etime;

        // Any previously packed L_values now mirrors the *old* factor. Mark
        // the CSC buffer stale so the solve gate misses until sTiles_packing
        // is called again. The buffer itself is kept allocated for reuse.
        s->schemes[global_index_mapped]->packed = false;

        return static_cast<int>(status);
    }

    /**
    * @brief [C-API] Retrieves the execution time of a specific Cholesky factorization.
    *
    * After a factorization is performed via `sTiles_chol`, this function can be
    * used to get the wall-clock time it took to complete.
    *
    * @param[in] group_index The index of the call group.
    * @param[in] call_index The index of the specific call within the group.
    * @param[in] obj An opaque handle (void**) to the `sTiles_object`.
    * @return The execution time in seconds, or a negative value on error.
    */
    double sTiles_get_chol_timing(int group_index, int call_index, void** obj) {

        if (!obj) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return -1.0;
        }

        sTiles_object* s = (sTiles_object*)(*obj);
        if (!s->schemes) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "TiledMatrix scheme cannot be null.");
            return -1.0;
        }

        int global_index = s->schemes[0]->call_lookup_table[group_index][call_index];
        int global_index_mapped = global_index;

        if (group_index >= 0 && call_index >= 0 && s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index > -1) {
            
            int new_call_index  = s->stiles_groups[group_index].stiles_calls[call_index].mapped_call_index;
            int new_group_index = s->stiles_groups[group_index].stiles_calls[call_index].mapped_group_index;
            global_index_mapped = s->schemes[0]->call_lookup_table[new_group_index][new_call_index];
        }

        return s->schemes[global_index_mapped]->timings[0];
    }   

    /**
    * @brief [C-API] Prints the timing results for all calls within a specific group.
    *
    * A utility function to print a formatted summary of the Cholesky factorization
    * times for every call within a given group.
    *
    * @param[in] group_index The group to print timings for.
    * @param[in] obj An opaque handle (void**) to the `sTiles_object`.
    */
    void sTiles_print_chol_timings(int group_index, void** obj) {

        if (!obj) {
            sTiles::Logger::error(sTiles::StatusCode::IllegalValue, "The provided sTiles object handle is null.");
            return;
        }

        sTiles_object* s = *((sTiles_object**)obj);
        int num_calls = s->stiles_groups[group_index].num_calls;

        sTiles::Logger::timing("│   • Cholesky timings (group ", group_index, ")");
        for (int j = 0; j < num_calls; ++j) {
            sTiles::f64 chol_time = sTiles_get_chol_timing(group_index, j, obj);
            if (chol_time >= 0.0) {
                // Use a stringstream to preserve complex formatting
                std::stringstream ss;
                ss << "│     ◦ Call " << j
                   << "  time: " << std::fixed << std::setprecision(6)
                   << chol_time << " s";
                sTiles::Logger::timing(ss.str());
            }
        }

    }
}
