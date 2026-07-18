
/**
 * @file process.cpp
 *
 * @brief Implementation of sTiles process initialization, preprocessing, and execution routines.
 *
 * This file contains the core implementation of various initialization and preprocessing routines 
 * used in the sTiles framework. These routines support hierarchical matrix factorization tasks 
 * (e.g., Cholesky factorization) by dividing computations into tiled structures and optimizing 
 * execution for multi-core architectures. The primary functions handle:
 * - Initialization of sTiles objects and groups for hierarchical processing.
 * - Setup and management of computational phases (e.g., tree reduction, symbolic factorization).
 * - Preprocessing and configuration of matrix properties for efficient execution.
 * - Advanced routines for load balancing, task distribution, and memory allocation.
 *
 * Designed and developed by:
 * - Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST).
 *
 * @version 3.0.0
 * @date 2026
 * @license Proprietary
 *
 * @note This file is part of the sTiles framework, a proprietary software package.
 *       Redistribution or modification without prior permission is prohibited.
 *
 * @contact esmail.abdulfattah@kaust.edu.sa
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST.
 * All rights reserved.
 *
 * @license
 * This software is proprietary and confidential. Unauthorized copying, distribution, or modification 
 * of this software, via any medium, is strictly prohibited. Permission is granted to use the software 
 * in binary form for non-commercial purposes only, provided that this copyright notice and permission 
 * notice are included in all copies or substantial portions of the software.
 *
 * DISCLAIMER:
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. 
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, 
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE 
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// =============================================================================
// File-local compile-time constants
// =============================================================================

#ifdef STILES_GPU
    #define DEFAULT_TILE_SIZE 600
#else
    #define DEFAULT_TILE_SIZE 40
#endif

// One-level partitions = 2, two-level = 6. Disabled by default.
#ifndef STILES_USE_PARTITIONS
#define STILES_USE_PARTITIONS 0
#endif

// The control-parameter array size (STILES_NUM_PARAMS), the per-slot
// documentation, and the default table all live in
// tools/common/stiles_params.hpp — the single source of truth. Included
// here (before the project headers) because the array must be defined
// before tools/ordering/stiles_ordering.hpp, which reads it inline.
#include "../common/stiles_params.hpp"

// =============================================================================
// Standard-library headers
// =============================================================================

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>   // std::getenv / std::atoi (adaptive_snode_cap)
#include <cstring>   // std::strcmp (sTiles_get_version git fallback)
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef EXPORT_MATRIX
    #include <sys/stat.h>
    #include <sys/types.h>
#endif

// =============================================================================
// Control-parameter arrays
// =============================================================================
//
// Defined here (before the project headers) because tools/ordering/stiles_ordering.hpp
// inlines code that reads stiles_control_params[] and is included below.
//
// Per-slot semantics, named indices (sTiles::param::*) and the default table
// are documented ONCE in tools/common/stiles_params.hpp. Setters/getters live
// in tools/common/stiles_params.cpp.
//
// Live storage. Initialized from the same STILES_DEFAULT_PARAMS_INIT macro as
// sTiles::kDefaultParams, so the reset_* APIs and the load-time state can
// never diverge.
int stiles_control_params[STILES_NUM_PARAMS] = STILES_DEFAULT_PARAMS_INIT;

// User-configuration shadow of the live array. Holds what the USER asked for
// (defaults + explicit setter calls) and is never touched by preprocessing.
// The auto-resolvable slots ([1] tile size, [3] tile type, [4] tile ordering,
// [5] tile-ordering block size) get RESOLVED values written into the live
// array during preprocessing; sTiles_preprocess_group restores those slots
// from this shadow before each group so every group re-resolves from the
// user's configuration instead of inheriting the previous group's resolution.
// Maintained by the setters in tools/common/stiles_params.cpp.
int stiles_user_params[STILES_NUM_PARAMS] = STILES_DEFAULT_PARAMS_INIT;

// Expert-mode flag. Must call sTiles_expert_user() before any configuration
// setter takes effect. Non-static because tools/ordering/ordering_overrides.cpp
// and tools/common/stiles_params.cpp both read it via `extern`.
bool expert_mode_enabled = false;

// Typo scan for STILES_* environment variables (tools/common/stiles_params.cpp).
extern "C" void stiles_env_validate(void);

// =============================================================================
// Project headers
// =============================================================================

#include "process_init.hpp"
#include "debug.hpp"
#include "../compute/stiles_compute.hpp"  // sTiles::pack_L_values
#include "debug_smart.hpp"
#ifdef STILES_GPU
#include "../gpu/gpu_dispatch_plan.hpp"   // sTiles::gpu::plan_dispatch (multi-GPU)
#include "../gpu/dpotrf/gh200_unified_alloc.hpp"  // sTiles::gpu::gh200::alloc_slab
#endif

// ------------------------------------------------------------------
// Force the BLAS to a SINGLE thread on non-MKL (OpenBLAS / generic) builds.
//
// sTiles supplies its own parallelism through the tile schedule, so the BLAS must
// be SEQUENTIAL — exactly like the Linux build, which links sequential MKL. macOS
// links OpenBLAS, which is multi-threaded by DEFAULT; under sTiles' nested tile
// parallelism that both oversubscribes and, worse, makes CONCURRENT factorizations
// non-deterministic: each concurrent call gets a different OpenBLAS reduction
// order, so their log-determinants disagree (observed ~1.8e-4 on ferris, 2 calls
// x 2 cores). Pinning OpenBLAS to one thread restores the sequential-BLAS design
// and makes concurrent chols bit-consistent (as they already are with sequential
// MKL on Linux). Multi-threaded BLAS buys nothing here anyway — the hot GEMMs are
// small per-tile blocks. Runs once at library load. MKL builds skip this (their
// embedded MKL is already sequential).
#if !defined(STILES_WITH_MKL)
extern "C" void openblas_set_num_threads(int) __attribute__((weak));
namespace {
struct sTilesBlasSingleThreadInit {
    sTilesBlasSingleThreadInit() {
        // Fallback for BLAS libs that read the env but export no setter symbol
        // (respect an explicit user setting), then force it authoritatively.
        stiles_setenv("OPENBLAS_NUM_THREADS", "1", /*overwrite=*/0);
        if (openblas_set_num_threads) openblas_set_num_threads(1);
    }
};
static sTilesBlasSingleThreadInit s_blas_single_thread_init;
}  // namespace
#endif

// ------------------------------------------------------------------
// CPU-path file-local helpers (used by the sparse/auto path — NOT GPU-specific,
// so they must live OUTSIDE the STILES_GPU guard or CPU builds fail to link).
namespace {

// Supernode-width cap for the sparse module's CHOLMOD analyze. Default = legacy
// 2*tile_size (raising it was tested on af_0_k101/audikw and did NOT help — 80 is
// best). STILES_SNODE_CAP env var overrides, for experiments.
inline int adaptive_snode_cap(int tile_size, int num_cores, int n) {
    (void)num_cores; (void)n;
    if (const char* e = std::getenv("STILES_SNODE_CAP")) { int v = std::atoi(e); if (v > 0) return v; }
    return std::max(64, 2 * tile_size);
}

// Mean tile column-occupancy from L's CSC pattern (L_colptr/L_rowind), available right
// after symbolic_phase. Per active tile-block: (#active columns)/tile_width, averaged.
// High (clustered, e.g. sem_n*) -> semisparse; low (scattered, irregular) -> sparse.
// One O(nnz(L)) pass, O(ntr) scratch. Assumes L_rowind sorted ascending within columns.
inline double mean_occupancy_from_L(const TiledMatrix* scheme, int tile_size) {
    const int64_t* colptr = scheme->L_colptr;
    const int* rowind = scheme->L_rowind;
    const int n  = scheme->dim;
    const int ts = tile_size;
    if (!colptr || !rowind || n <= 0 || ts <= 0) return 0.0;
    const int ntr = (n - 1) / ts + 1;
    // 2D block density (hub-robust): mean over off-diagonal tile-blocks of
    // nnz(block)/(height*width). A high-degree hub column spreads thin across many
    // blocks -> low density (correctly favors sparse), unlike a raw active-column count
    // which a hub inflates to ~1 everywhere.
    std::vector<long long> nnzR(static_cast<std::size_t>(ntr), 0);
    std::vector<int> touched;
    double dens_sum = 0.0; long long nblk = 0;
    for (int C = 0; C < ntr; ++C) {
        const int c0 = C * ts;
        const int w  = std::min(n, c0 + ts) - c0;
        touched.clear();
        for (int c = c0; c < c0 + w; ++c) {
            for (long long p = colptr[c]; p < colptr[c + 1]; ++p) {
                const int R = static_cast<int>(rowind[p] / ts);
                if (R <= C) continue;                   // off-diagonal blocks only
                if (nnzR[R] == 0) touched.push_back(R);
                nnzR[R] += 1;
            }
        }
        nblk += static_cast<long long>(touched.size());
        for (int R : touched) {
            const int r0 = R * ts;
            const int h  = std::min(n, r0 + ts) - r0;
            dens_sum += static_cast<double>(nnzR[R]) / (static_cast<double>(h) * w);
            nnzR[R] = 0;
        }
    }
    return nblk ? dens_sum / static_cast<double>(nblk) : 0.0;
}

// Arrowhead / hub signal: max column nnz divided by mean column nnz of L. A
// bordered (arrowhead) matrix has a few hub columns (the dense border) whose
// degree is orders of magnitude above the mean -> large skew; a uniform mesh or
// graph runs skew ~1.5. This separates bordered INLA precisions (semisparse-ideal:
// dense border + sparse interior) from uniformly-sparse matrices (supernodal-best),
// which the MEAN block density alone confuses, because a hub column spreads thin
// across many blocks and reads as low density. Measured: bern 1025x, pedigree 526x,
// sem_n* ~556x vs thermal1/thermomech/net* ~1.4--1.5x, so a >=50 cutoff has a wide
// margin on both sides.
inline double degree_skew_from_L(const TiledMatrix* scheme) {
    const int64_t* colptr = scheme->L_colptr;
    const int n = scheme->dim;
    if (!colptr || n <= 0) return 1.0;
    long long total = 0, mx = 0;
    for (int c = 0; c < n; ++c) {
        const long long d = colptr[c + 1] - colptr[c];
        total += d;
        if (d > mx) mx = d;
    }
    const double mean = static_cast<double>(total) / n;
    return mean > 0.0 ? static_cast<double>(mx) / mean : 1.0;
}

// Peak tile fan-out: max over tile-columns of the number of off-diagonal active
// tile-rows below the diagonal. ~ the peak available parallelism (step k spawns
// ~(neighbors of k)^2 independent GEMM updates). Used to diagnose/decide the
// work-aware core cap (a column with few neighbors can't fill many cores).
inline int max_neighbor_tiles_from_L(const TiledMatrix* scheme, int tile_size) {
    const int64_t* colptr = scheme->L_colptr;
    const int* rowind = scheme->L_rowind;
    const int n  = scheme->dim;
    const int ts = tile_size;
    if (!colptr || !rowind || n <= 0 || ts <= 0) return 0;
    const int ntr = (n - 1) / ts + 1;
    std::vector<char> seen(static_cast<std::size_t>(ntr), 0);
    std::vector<int> touched;
    int maxnb = 0;
    for (int C = 0; C < ntr; ++C) {
        const int c0 = C * ts;
        const int w  = std::min(n, c0 + ts) - c0;
        touched.clear();
        for (int c = c0; c < c0 + w; ++c)
            for (long long p = colptr[c]; p < colptr[c + 1]; ++p) {
                const int R = static_cast<int>(rowind[p] / ts);
                if (R <= C) continue;                 // off-diagonal, below diagonal
                if (!seen[static_cast<std::size_t>(R)]) { seen[static_cast<std::size_t>(R)] = 1; touched.push_back(R); }
            }
        if (static_cast<int>(touched.size()) > maxnb) maxnb = static_cast<int>(touched.size());
        for (int R : touched) seen[static_cast<std::size_t>(R)] = 0;
    }
    return maxnb;
}

}  // namespace (CPU-path helpers)

#ifdef STILES_GPU
namespace {

// ------------------------------------------------------------------
// GPU planner helper (file-local, used by sTiles_preprocess_group).
//
// Runs plan_dispatch for `num_calls` × `bytes_per_call`, prints the
// diagnostic, and writes the outcome onto the primary scheme. Returns
// true when any GPU path (case A or B) was selected.
//
// `label` distinguishes "dense" vs "semisparse" in the log line.
// ------------------------------------------------------------------
inline bool gpu_plan_and_apply_to_scheme(TiledMatrix*  primary_scheme,
                                         int           num_calls,
                                         size_t        bytes_per_call,
                                         double        safety_margin,
                                         bool          compute_inverse,
                                         const char*   label)
{
    sTiles::gpu::DispatchPlan plan;
    const bool fit = sTiles::gpu::plan_dispatch(
        num_calls, bytes_per_call, safety_margin, plan);
    sTiles::gpu::print_plan(plan, num_calls);

    if (fit) {
        primary_scheme->use_gpu               = true;
        primary_scheme->GPU_ID                = plan.gpu_id[0];
        primary_scheme->compute_inverse_on_gpu = compute_inverse;
        primary_scheme->gpu_dispatch_plan     = plan;
        sTiles::Logger::info("│   ✓ GPU MODE (", label, "): case ", plan.strategy,
            " — ", num_calls, " call(s) across ",
            static_cast<int>(plan.devices_used.size()), " device(s)");
    } else {
        primary_scheme->use_gpu               = false;
        primary_scheme->GPU_ID                = -1;
        primary_scheme->compute_inverse_on_gpu = false;
        primary_scheme->gpu_dispatch_plan     = sTiles::gpu::DispatchPlan{};
        sTiles::Logger::warning("│   ✗ CPU MODE: case C — ", label, " ", num_calls,
            " call(s) exceed total GPU capacity even spread across all devices.");
    }
    return fit;
}

// ------------------------------------------------------------------
// Per-device allocator helper. Walks the plan, groups schemes by their
// assigned device, and invokes `alloc_fn(per_device_schemes, count,
// group_id, device_id)` once per distinct device. Returns false on
// the first allocation failure.
//
// The two existing tile-allocator entry points (dense and semisparse)
// share the same call shape, so this helper is templated on the
// allocator function pointer.
//
// Note: the per-device cudaMalloc happens *inside* the allocator
// callee (today: GpuTileManager). The GH200 unified-memory path lives
// at that layer — see sTiles::gpu::gh200::alloc_slab — which is
// runtime-gated and takes effect once GpuTileManager calls into it.
// ------------------------------------------------------------------
template <typename AllocFn>
inline bool gpu_allocate_per_device(const sTiles::gpu::DispatchPlan& plan,
                                    const std::vector<TiledMatrix*>& group_schemes,
                                    int       group_index,
                                    AllocFn   allocator)
{
    const int num_calls = static_cast<int>(group_schemes.size());
    for (int dev_id : plan.devices_used) {
        std::vector<TiledMatrix*> dev_schemes;
        dev_schemes.reserve(num_calls);
        for (int c = 0; c < num_calls; ++c) {
            if (plan.gpu_id[c] == dev_id) dev_schemes.push_back(group_schemes[c]);
        }
        if (dev_schemes.empty()) continue;
        const bool ok = allocator(dev_schemes.data(),
                                  static_cast<int>(dev_schemes.size()),
                                  group_index, dev_id);
        if (!ok) return false;
    }
    return true;
}

// Disable GPU for every scheme in the group (used on allocation failure
// to fall back to CPU end-to-end).
inline void gpu_disable_group(std::vector<TiledMatrix*>& group_schemes)
{
    for (auto* s : group_schemes) {
        s->use_gpu = false;
        s->GPU_ID  = -1;
        s->compute_inverse_on_gpu = false;
        s->gpu_persistent_ctx = nullptr;
    }
}

}  // namespace (file-local GPU helpers)
#endif  // STILES_GPU

#include "../algorithms/tasks.hpp"
#include "../common/stiles_config.hpp"
#include "../common/stiles_expiry.hpp"
#include "../common/stiles_policy.hpp"
#include "../tile/preprocess.hpp"
#include "../free/free.hpp"
#include "../sparse/api.hpp"
#include "../memory/cpuSmartTileMemoryManager.hpp"
#include "../memory/memory_estimate.hpp"
#include "../memory/memory_namespace.hpp"
#include "../memory/TileMemoryManager.hpp"
#include "../memory/workspace.hpp"
#include "../ordering/ordering_utils.hpp"
#include "../ordering/stiles_ordering.hpp"
#include "../symbolic/symbolic_semisparse.hpp"
#include "../tree/tree_wrappers.hpp"
#include "../TileIndexer/preprocess_helpers.hpp"
#include "../TileIndexer/TileIndexer.hpp"
#include "../TileIndexer/TileIndexerMapper.hpp"


// =============================================================================
// Forward declarations to symbols defined in other translation units
// =============================================================================

// User-permutation / forced-partition state and the matching sTiles_set_*/clear_*
// entry points live in tools/ordering/ordering_overrides.cpp. Only the
// "clear all" entry is called from this file (sTiles_quit), so we forward-
// declare it here and skip the rest.
extern "C" void sTiles_clear_all_user_permutations(void);

// =============================================================================
// File-local globals (process-side state)
// =============================================================================

// Per-group rescale core count: 0 = no rescale, >0 = target core count.
// Populated before sTiles_create by sTiles_set_rescale_cores.
static std::vector<int> g_rescale_per_group;
static int g_rescale_num_groups = 0;

// Registered sTiles_object instances; iterated by sTiles_freeGroup / sTiles_quit.
static std::vector<sTiles_object*> g_registered_objects;

// Default neighbor-lookup method. CharMask is auto-promoted to BitsetMask at
// runtime when the matrix is large enough (see sTiles::policy::pick_indexer_method).
// An explicit sTiles_set_neighbor_lookup_method() call overrides this default.
static TileIndexer::Method user_neighbor_lookup_method = TileIndexer::Method::CharMask;


//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//___________________________________________________________________________MEMORY MANAGEMENT_________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________

#ifdef __cplusplus
extern "C" {
#endif

// Memory introspection / estimate C-ABI entry points
// (sTiles_GetGroupMemoryUsage, sTiles_GetGroupsMemoryUsage, sTiles_estimate_memory,
// sTiles_estimate_memory_verbose, sTiles_memory_fits) live in
// tools/memory/memory_api.cpp.

void sTiles_freeGroup(int group_ID) {
    // Allow freeing group 0 as well; negative group IDs remain a no-op.
    if (group_ID >= 0) {
        for (sTiles_object* s : g_registered_objects) {
            sTiles::destroy_all_schemes_for_group(s, group_ID);
        }
        sTiles::Memory::MemoryManager::freeAllGroup(group_ID);
        sTiles::Memory::TileMemoryManager::freeAllGroup(group_ID);
        sTiles::Memory::TreeMemoryManager::freeAllGroup(group_ID);
        sTiles::Memory::AlgorithmsMemoryManager::freeAllGroup(group_ID);
        sTiles::Memory::OrderingMemoryManager::freeAllGroup(group_ID);
        sTiles::Memory::CpuSmartTileMemoryManager::freeAllGroup(group_ID);
        sTiles::Memory::TileIndexerMemoryManager::freeAllGroup(group_ID);
#ifdef STILES_GPU
        sTiles::Memory::GpuMemoryManager::freeAllGroup(group_ID);
#endif
    }
}

void sTiles_quit() {
#ifdef STILES_LEAK_CHECK
    // Snapshot pre-free usage across all managers
    size_t pre_remaining = 0;
    pre_remaining += sTiles::Memory::MemoryManager::getTotalMemoryAllocated();
    pre_remaining += sTiles::Memory::TileMemoryManager::getTotalMemoryAllocated();
    pre_remaining += sTiles::Memory::TreeMemoryManager::getTotalMemoryAllocated();
    pre_remaining += sTiles::Memory::AlgorithmsMemoryManager::getTotalMemoryAllocated();
    pre_remaining += sTiles::Memory::OrderingMemoryManager::getTotalMemoryAllocated();
    pre_remaining += sTiles::Memory::CpuSmartTileMemoryManager::getTotalMemoryAllocated();
    pre_remaining += sTiles::Memory::TileIndexerMemoryManager::getTotalMemoryAllocated();
#ifdef STILES_GPU
    const size_t pre_gpu = static_cast<size_t>(sTiles::Memory::GpuMemoryManager::getTotalMemoryAllocatedInGiB() * 1024.0 * 1024.0 * 1024.0);
    // Don't include GPU bytes in the aggregated CPU "remaining"; track separately for clarity
#else
    const size_t pre_gpu = 0;
#endif
    sTiles::Logger::info("│   [LeakCheck] Pre-free summary: CPU=" + std::to_string(pre_remaining) + " bytes, GPU=" + std::to_string(pre_gpu) + " bytes");
#endif

    // IMPORTANT: Destroy schemes FIRST to clean up GpuPersistentContext
    // (CUDA handles, streams, events, semisparse indices) before freeing tile memory.
    // This ensures proper synchronization and resource release order.
    for (sTiles_object* s : g_registered_objects) {
        sTiles::destroy_all_schemes(s);
    }
    g_registered_objects.clear();

    // Now free GPU tile memory (after handles/streams are destroyed)
#ifdef STILES_GPU
    sTiles::Memory::GpuMemoryManager::freeAll();
#endif

    // Free user-provided permutations
    sTiles_clear_all_user_permutations();

    // Free all tracked CPU allocations across managers
    sTiles::Memory::MemoryManager::freeAll();
    sTiles::Memory::TileMemoryManager::freeAll();
    sTiles::Memory::TreeMemoryManager::freeAll();

    sTiles::Memory::AlgorithmsMemoryManager::freeAll();
    sTiles::Memory::OrderingMemoryManager::freeAll();
    sTiles::Memory::CpuSmartTileMemoryManager::freeAll();
    sTiles::Memory::TileIndexerMemoryManager::freeAll();

    // Summarize any remaining tracked bytes across managers
    size_t remaining = 0;
    remaining += sTiles::Memory::MemoryManager::getTotalMemoryAllocated();
    remaining += sTiles::Memory::TileMemoryManager::getTotalMemoryAllocated();
    remaining += sTiles::Memory::TreeMemoryManager::getTotalMemoryAllocated();
    remaining += sTiles::Memory::AlgorithmsMemoryManager::getTotalMemoryAllocated();
    remaining += sTiles::Memory::OrderingMemoryManager::getTotalMemoryAllocated();
    remaining += sTiles::Memory::CpuSmartTileMemoryManager::getTotalMemoryAllocated();
    remaining += sTiles::Memory::TileIndexerMemoryManager::getTotalMemoryAllocated();

    if (remaining > 0) {
        sTiles::Logger::warning("│   [LeakCheck] " + std::to_string(remaining) + " bytes not freed across managers.");
#ifdef STILES_LEAK_CHECK
        {
            const struct { const char* name; size_t bytes; } mgrs[] = {
                {"MemoryManager",            sTiles::Memory::MemoryManager::getTotalMemoryAllocated()},
                {"TileMemoryManager",        sTiles::Memory::TileMemoryManager::getTotalMemoryAllocated()},
                {"TreeMemoryManager",        sTiles::Memory::TreeMemoryManager::getTotalMemoryAllocated()},
                {"AlgorithmsMemoryManager",  sTiles::Memory::AlgorithmsMemoryManager::getTotalMemoryAllocated()},
                {"OrderingMemoryManager",    sTiles::Memory::OrderingMemoryManager::getTotalMemoryAllocated()},
                {"CpuSmartTileMemoryManager", sTiles::Memory::CpuSmartTileMemoryManager::getTotalMemoryAllocated()},
                {"TileIndexerMemoryManager", sTiles::Memory::TileIndexerMemoryManager::getTotalMemoryAllocated()},
            };
            for (const auto& m : mgrs) {
                if (m.bytes == 0) continue;
                sTiles::Logger::warning(std::string("│   [LeakCheck] ") + m.name + ": " + std::to_string(m.bytes) + " bytes");
            }
        }

#endif
        if (static_cast<int>(sTiles::getLevel()) >= static_cast<int>(sTiles::Level::Trace)) {
            std::stringstream ss;
            sTiles::Memory::MemoryManager::debugDump(ss);
            std::string line;
            while (std::getline(ss, line)) {
                sTiles::Logger::debug(std::string(4, ' ') + "│ " + line);
            }

#ifdef STILES_LEAK_CHECK
            auto dump_mgr = [&](auto& fn, const char* tag) {
                std::stringstream s2;
                fn(s2);
                std::string ln;
                sTiles::Logger::debug(std::string(4, ' ') + "│ === " + tag + " ===");
                while (std::getline(s2, ln)) {
                    sTiles::Logger::debug(std::string(4, ' ') + "│ " + ln);
                }
            };
            if (sTiles::Memory::TileMemoryManager::getTotalMemoryAllocated() > 0)
                dump_mgr(sTiles::Memory::TileMemoryManager::debugDump, "TileMemoryManager");
            if (sTiles::Memory::TreeMemoryManager::getTotalMemoryAllocated() > 0)
                dump_mgr(sTiles::Memory::TreeMemoryManager::debugDump, "TreeMemoryManager");
            if (sTiles::Memory::AlgorithmsMemoryManager::getTotalMemoryAllocated() > 0)
                dump_mgr(sTiles::Memory::AlgorithmsMemoryManager::debugDump, "AlgorithmsMemoryManager");
            if (sTiles::Memory::OrderingMemoryManager::getTotalMemoryAllocated() > 0)
                dump_mgr(sTiles::Memory::OrderingMemoryManager::debugDump, "OrderingMemoryManager");
            if (sTiles::Memory::TileIndexerMemoryManager::getTotalMemoryAllocated() > 0)
                dump_mgr(sTiles::Memory::TileIndexerMemoryManager::debugDump, "TileIndexerMemoryManager");
            if (sTiles::Memory::CpuSmartTileMemoryManager::getTotalMemoryAllocated() > 0)
                dump_mgr(sTiles::Memory::CpuSmartTileMemoryManager::debugDump, "CpuSmartTileMemoryManager");
#ifdef STILES_GPU
            size_t gpu_bytes = static_cast<size_t>(sTiles::Memory::GpuMemoryManager::getTotalMemoryAllocatedInGiB() * 1024.0 * 1024.0 * 1024.0);
            if (gpu_bytes > 0) {
                sTiles::Logger::warning("│   [LeakCheck] GpuMemoryManager: " + std::to_string(gpu_bytes) + " bytes (summary)");
            }
#endif
#endif // STILES_LEAK_CHECK

            printf("\n");
        }
    }
#ifdef STILES_LEAK_CHECK
    else {
        // Explicitly report success when leak check is enabled
        sTiles::Logger::info("│   [LeakCheck] All tracked allocations freed successfully (CPU=" + std::to_string(pre_remaining) + 
                            ", GPU=" + std::to_string(pre_gpu) + " before free)");
    }
#endif

    // Destroy persistent teams so future runs can change core counts
    sTiles::Control::FinalizeGlobal();

}

#ifdef __cplusplus
} // extern "C"
#endif



//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//___________________________________________________________________________USER To sTiles Object____(Sparse & Dense)__________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________

/**
 * @brief Returns the compile-time version string of the sTiles library.
 *
 * This function provides a stable, C-style API to retrieve the library's
 * version and build date, defined at compile time.
 *
 * @return A constant C-style string in the format "MAJOR.MINOR.MICRO (built YYYY-MM-DD)".
 *         Example: "2.0.0 (built 2026-01-21)"
 */
const char* sTiles_get_version() {
    // Build the full version string with build date and build identity at
    // first call.
    static char full_version[160] = {0};
    if (full_version[0] == '\0') {
        // STILES_GIT_VERSION is the build identity baked in by make.inc, in
        // order of preference: an explicit BUILD_VERSION=<...> make argument,
        // "git <date> [<hash>]" from a git checkout, "ci <ref> [<sha>]" from
        // the GitHub Actions environment (tarball checkouts), or
        // "src [<hash>]" — a content hash of the sTiles sources — for any
        // other tree. Spaces arrive as underscores (make -D quoting); undo
        // that for display. "unknown" (no identity at all) is omitted.
        const char* id_raw = "";
#ifdef STILES_GIT_VERSION
        id_raw = STILES_GIT_VERSION;
#endif
        if (id_raw[0] != '\0' && std::strcmp(id_raw, "unknown") != 0) {
            char id_pretty[96] = {0};
            snprintf(id_pretty, sizeof(id_pretty), "%s", id_raw);
            for (char* p = id_pretty; *p; ++p)
                if (*p == '_') *p = ' ';
            snprintf(full_version, sizeof(full_version), "%s (built %04d-%02d-%02d, %s)",
                     sTiles::config::versionString,
                     sTiles::internal::BUILD_Y,
                     sTiles::internal::BUILD_M,
                     sTiles::internal::BUILD_D,
                     id_pretty);
        } else {
            snprintf(full_version, sizeof(full_version), "%s (built %04d-%02d-%02d)",
                     sTiles::config::versionString,
                     sTiles::internal::BUILD_Y,
                     sTiles::internal::BUILD_M,
                     sTiles::internal::BUILD_D);
        }
    }
    return full_version;
}

/**
 * @brief Prints the sTiles library's ASCII art logo and version information to stdout.
 *
 * This function serves as a C-style API wrapper for the C++ sTiles::show_logo()
 * function, providing a consistent and branded way to display version details.
 */
void sTiles_print_version() {
    // Leverage the new, better-looking function from the config header.
    // This avoids code duplication and provides a better user experience.
    sTiles::show_logo();
}


void sTiles_set_rescale_cores(const int* rescale_per_group, int num_groups) {
    g_rescale_per_group.clear();
    g_rescale_num_groups = 0;
    if (rescale_per_group && num_groups > 0) {
        // Validate: each value must be 0 or 1
        for (int i = 0; i < num_groups; i++) {
            if (rescale_per_group[i] != 0 && rescale_per_group[i] != 1) {
                sTiles::Logger::error("sTiles_set_rescale_cores: rescale_per_group[", i, "] = ",
                                      rescale_per_group[i], " is invalid. Must be 0 (not rescale) or 1 (rescale).");
                return;
            }
        }
        g_rescale_per_group.assign(rescale_per_group, rescale_per_group + num_groups);
        g_rescale_num_groups = num_groups;
    }
}

/**
 * @brief Sets the runtime logging verbosity for the sTiles library.
 *
 * This function allows the user to control which log messages are displayed at runtime.
 * Note that this is limited by the compile-time level set by STILES_LOG_LEVEL.
 * For example, if the library was compiled with STILES_LOG_LEVEL=1 (Info), calling
 * sTiles_set_log_level(2) will not enable Debug messages, as they were compiled out.
 *
 * @param level An integer representing the desired log level:
 *              - -2: TimingOnly (hide everything except `[TIME]` logs)
 *              - -1: None (only errors and fatal messages are shown)
 *              -  0: Time (timing messages)
 *              -  1: Info (informational messages, warnings)
 *              -  2: Debug (detailed debugging information)
 *              -  3: Trace (most verbose, function entry/exit, etc.)
 */
void sTiles_set_log_level(int level) {

    sTiles::Level new_level;
    bool is_valid = true;

    switch (level) {
        case -2:
            new_level = sTiles::Level::TimingOnly;
            break;
        case STILES_LOG_LEVEL_NONE:
            new_level = sTiles::Level::None;
            break;
        case STILES_LOG_LEVEL_TIME:
            new_level = sTiles::Level::Time;
            break;
        case STILES_LOG_LEVEL_INFO:
            new_level = sTiles::Level::Info;
            break;
        case STILES_LOG_LEVEL_DEBUG:
            new_level = sTiles::Level::Debug;
            break;
        case STILES_LOG_LEVEL_TRACE:
            new_level = sTiles::Level::Trace;
            break;
        default:
            sTiles::Logger::warning("Invalid log level provided: ", level,
                                    ". Valid levels are -2 to 3. Log level remains unchanged.");
            is_valid = false;
            break;
    }

    if (is_valid) {
        sTiles::setLevel(new_level);
        // Provide feedback on the change, if the new level allows it.
        // E.g., if you set the level to None, this confirmation message won't print, which is consistent.
        sTiles::Logger::info("sTiles runtime log level set to: ", sTiles::detail::levelToString(new_level));
    }
}

// sTiles_set_*/sTiles_get_* parameter setters/getters live in tools/common/stiles_params.cpp.
// Storage (stiles_control_params[]) stays here; that file reaches it via extern.

/**
 * @brief Set the neighbor lookup method (TileIndexer::Method).
 *
 * Controls which tile-counting strategy is used during symbolic factorization:
 *   - 0 = PagedMask      (default) — sparse, best for large matrices
 *   - 1 = BitsetMask     — dense uint64_t bitset, O(n²) memory (small matrices only)
 *   - 2 = BoolMask       — dense std::vector<bool>, O(n²) memory (small matrices only)
 *   - 3 = CharMask       — dense std::vector<char>, O(n²) memory (small matrices only)
 *   - 4 = TiledBoolMask  — chunked bool mask in tiled blocks
 *   - 5 = TiledBitsetMask— chunked bitset mask in tiled blocks
 *   - 6 = HashSet        — sparse unordered_set of active tile IDs
 *   - 7 = SortUnique     — sparse: gather IDs, sort, unique
 *   - 8 = Auto           — auto-select via planner
 *   - 9 = LazyLookUp     — dense bool mask with compact index map + lazy closure
 *
 * @param value 0..9 as above.
 */
void sTiles_set_neighbor_lookup_method(int value) {
    switch (value) {
        case 0: user_neighbor_lookup_method = TileIndexer::Method::PagedMask;       break;
        case 1: user_neighbor_lookup_method = TileIndexer::Method::BitsetMask;      break;
        case 2: user_neighbor_lookup_method = TileIndexer::Method::BoolMask;        break;
        case 3: user_neighbor_lookup_method = TileIndexer::Method::CharMask;        break;
        case 4: user_neighbor_lookup_method = TileIndexer::Method::TiledBoolMask;   break;
        case 5: user_neighbor_lookup_method = TileIndexer::Method::TiledBitsetMask; break;
        case 6: user_neighbor_lookup_method = TileIndexer::Method::HashSet;         break;
        case 7: user_neighbor_lookup_method = TileIndexer::Method::SortUnique;      break;
        case 8: user_neighbor_lookup_method = TileIndexer::Method::Auto;            break;
        case 9: user_neighbor_lookup_method = TileIndexer::Method::LazyLookUp;      break;
        default:
            sTiles::Logger::warning("sTiles_set_neighbor_lookup_method: invalid value ", value,
                                    " (0=PagedMask,1=BitsetMask,2=BoolMask,3=CharMask,"
                                    "4=TiledBoolMask,5=TiledBitsetMask,6=HashSet,7=SortUnique,"
                                    "8=Auto,9=LazyLookUp)");
    }
}

/**
 * @brief Get the current neighbor lookup method as an integer.
 *
 * @return 0=PagedMask, 1=BitsetMask, 2=BoolMask, 3=CharMask,
 *         4=TiledBoolMask, 5=TiledBitsetMask, 6=HashSet, 7=SortUnique,
 *         8=Auto, 9=LazyLookUp.
 */
int sTiles_get_neighbor_lookup_method() {
    switch (user_neighbor_lookup_method) {
        case TileIndexer::Method::PagedMask:       return 0;
        case TileIndexer::Method::BitsetMask:      return 1;
        case TileIndexer::Method::BoolMask:        return 2;
        case TileIndexer::Method::CharMask:        return 3;
        case TileIndexer::Method::TiledBoolMask:   return 4;
        case TileIndexer::Method::TiledBitsetMask: return 5;
        case TileIndexer::Method::HashSet:         return 6;
        case TileIndexer::Method::SortUnique:      return 7;
        case TileIndexer::Method::Auto:            return 8;
        case TileIndexer::Method::LazyLookUp:      return 9;
        default:                                   return -1;
    }
}


/**
 * @brief Returns the cache-derived tile_size first guess.
 *
 * Inspects the machine's total L3 cache and picks via a fixed staircase
 * (40/80/120). Pure read-only; does not mutate library state. This is the
 * same routine the library itself uses when `tile_size` is left at -1 (auto).
 *
 * @return Suggested tile size.
 */
int sTiles_get_auto_tile_size(void){

    return sTiles::get::auto_tile_size();
}

//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________


void sTiles_map_group_call_to_group_call(void** obj, int group_index1, int call_index1, int group_index2, int call_index2) {

    if (!obj || !(*obj)) {
        sTiles::Logger::error("[✘] obj is null.");
        return;
    }
        
    sTiles_object* s = (sTiles_object*)(*obj);

    s->stiles_groups[group_index1].stiles_calls[call_index1].mapped_group_index = group_index2;
    s->stiles_groups[group_index1].stiles_calls[call_index1].mapped_call_index  = call_index2;
}

int sTiles_bind(int group_index, int call_index, void** obj) {
    if (!obj || !(*obj)) {
        sTiles::Logger::error("sTiles_bind: obj is null.");
        return -1; // Or a specific error code like -104
    }

    sTiles_object* s = (sTiles_object*)(*obj);

    //if (s->factorization_type_per_group[group_index] < CPU_BIND_FACTOR_TYPE_MAX) {
        
        sTiles_call* call = &s->stiles_groups[group_index].stiles_calls[call_index];
        // Prefer namespaced persistent team activation (no thread creation here)
        return sTiles::Control::ActivatePersistentTeam(call->global_index);
    //}

    return 0;
}

int sTiles_unbind(int group_index, int call_index, void** obj) {
    if (!obj || !(*obj)) {
        return -1; // Or a specific error code
    }

    sTiles_object* s = (sTiles_object*)(*obj);

    // This conditional logic remains the same.
    //if (s->factorization_type_per_group[group_index] < CPU_BIND_FACTOR_TYPE_MAX) {
        
        // Broadcast the new deactivation routine the same way as the legacy finalize.
        // We just need the global index, which is the handle to the context.
        int global_index = s->stiles_groups[group_index].stiles_calls[call_index].global_index;
        return sTiles::Control::DeactivatePersistentTeam(global_index);

    //}

    // If the condition is false, it means no team was bound, so nothing to unbind.
    return 0;
}

//==============================================================================
// Rescale Schedule Switching
//==============================================================================

int sTiles_turn_on_rescale(int group_index, void** obj) {
    if (!obj || !(*obj)) {
        sTiles::Logger::error("sTiles_turn_on_rescale: obj is null.");
        return -1;
    }

    sTiles_object* s = (sTiles_object*)(*obj);
    if (group_index < 0 || group_index >= s->num_call_groups) {
        sTiles::Logger::error("sTiles_turn_on_rescale: invalid group_index.");
        return -1;
    }

    const int num_calls = s->stiles_groups[group_index].num_calls;
    for (int call_index = 0; call_index < num_calls; ++call_index) {
        const int global_index = s->stiles_groups[group_index].stiles_calls[call_index].global_index;
        TiledMatrix* scheme = s->schemes[global_index];
        if (scheme) {
            scheme->use_rescale.store(1, std::memory_order_release);
        }
    }

    return 0;
}

int sTiles_turn_off_rescale(int group_index, void** obj) {
    if (!obj || !(*obj)) {
        sTiles::Logger::error("sTiles_turn_off_rescale: obj is null.");
        return -1;
    }

    sTiles_object* s = (sTiles_object*)(*obj);
    if (group_index < 0 || group_index >= s->num_call_groups) {
        sTiles::Logger::error("sTiles_turn_off_rescale: invalid group_index.");
        return -1;
    }

    const int num_calls = s->stiles_groups[group_index].num_calls;
    for (int call_index = 0; call_index < num_calls; ++call_index) {
        const int global_index = s->stiles_groups[group_index].stiles_calls[call_index].global_index;
        TiledMatrix* scheme = s->schemes[global_index];
        if (scheme) {
            scheme->use_rescale.store(0, std::memory_order_release);
        }
    }

    return 0;
}


//==============================================================================
// User Parallel Execution
//==============================================================================

int sTiles_parallel_exec(int group_index, int call_index, void** obj, sTiles_user_func_t user_func, void* user_data) {
    if (!obj || !(*obj)) {
        sTiles::Logger::error("sTiles_parallel_exec: obj is null.");
        return -1;
    }
    if (!user_func) {
        sTiles::Logger::error("sTiles_parallel_exec: user_func is null.");
        return -1;
    }

    sTiles_object* s = (sTiles_object*)(*obj);
    int global_index = s->stiles_groups[group_index].stiles_calls[call_index].global_index;

    return sTiles::Control::DispatchUserFunction(global_index, user_func, user_data);
}


namespace sTiles { namespace Binding {

static int Setup_teams_for_call(sTiles_object* s, sTiles_call* call)
{
    if (!s || !call) {
        sTiles::Logger::error("Setup_teams_for_call: null pointer.");
        return -1;
    }

    const int ncores    = call->num_cores;
    const int* core_ids = call->core_bind_ids;
    const int base_idx  = call->global_index;
    const int total_idx = s->num_total_indices;

    if (!core_ids || ncores <= 0) {
        sTiles::Logger::error("Setup_teams_for_call: invalid core binding info.");
        return -1;
    }

    int status = 0;

    status = sTiles::Control::CreatePersistentTeam(ncores, core_ids, base_idx);
    if (status != 0) return status;

#if STILES_USE_PARTITIONS == 2

    if (ncores > 1) {
        int first_count  = ncores / 2;
        int second_count = ncores - first_count;

        const int* first_cores  = core_ids;
        const int* second_cores = core_ids + first_count;

        status = sTiles::Control::CreatePersistentTeam(first_count, first_cores, base_idx + 1 * total_idx);
        if (status != 0) return status;

        status = sTiles::Control::CreatePersistentTeam(second_count, second_cores, base_idx + 2 * total_idx);
        if (status != 0) return status;
    }

#elif STILES_USE_PARTITIONS == 6

    if (ncores > 1) {
        int first_count  = ncores / 2;
        int second_count = ncores - first_count;

        const int* first_cores  = core_ids;
        const int* second_cores = core_ids + first_count;

        if (ncores >= 4) {
            int first_count_partition1  = first_count / 2;
            int second_count_partition1 = first_count - first_count_partition1;
            const int* first_cores_partition1  = first_cores;
            const int* second_cores_partition1 = first_cores + first_count_partition1;

            int first_count_partition2  = second_count / 2;
            int second_count_partition2 = second_count - first_count_partition2;
            const int* first_cores_partition2  = second_cores;
            const int* second_cores_partition2 = second_cores + first_count_partition2;

            status = sTiles::Control::CreatePersistentTeam(first_count_partition1, first_cores_partition1, base_idx + 1 * total_idx);
            if (status != 0) return status;

            status = sTiles::Control::CreatePersistentTeam(first_count_partition2, first_cores_partition2, base_idx + 2 * total_idx);
            if (status != 0) return status;

            status = sTiles::Control::CreatePersistentTeam(second_count_partition1, second_cores_partition1, base_idx + 3 * total_idx);
            if (status != 0) return status;

            status = sTiles::Control::CreatePersistentTeam(second_count_partition2, second_cores_partition2, base_idx + 4 * total_idx);
            if (status != 0) return status;
        }

        status = sTiles::Control::CreatePersistentTeam(first_count, first_cores, base_idx + 5 * total_idx);
        if (status != 0) return status;

        status = sTiles::Control::CreatePersistentTeam(second_count, second_cores, base_idx + 6 * total_idx);
        if (status != 0) return status;
    }

#endif

    return 0;
}

int Setup_all_teams(void** obj)
{
    if (!obj || !(*obj)) {
        sTiles::Logger::error("Setup_all_teams: obj is null.");
        return -1;
    }

    sTiles_object* s = (sTiles_object*)(*obj);

    for (int group_index = 0; group_index < s->num_call_groups; ++group_index) {
        int num_calls_in_group = s->stiles_groups[group_index].num_calls;

        for (int call_index = 0; call_index < num_calls_in_group; ++call_index) {
            sTiles_call* call = &s->stiles_groups[group_index].stiles_calls[call_index];

            int status = Setup_teams_for_call(s, call);
            if (status != 0) {
                sTiles::Logger::error("Failed to create teams for global index ", call->global_index);
                return status;
            }
        }
    }

    // group teams: one team per group that combines all cores of that group
    const int group_team_base = s->num_total_indices * (1 + STILES_USE_PARTITIONS);

    for (int group_index = 0; group_index < s->num_call_groups; ++group_index) {
        std::vector<int> group_cores;

        int num_calls_in_group = s->stiles_groups[group_index].num_calls;
        for (int call_index = 0; call_index < num_calls_in_group; ++call_index) {
            sTiles_call* call = &s->stiles_groups[group_index].stiles_calls[call_index];
            const int* core_ids = call->core_bind_ids;
            int ncores = call->num_cores;
            for (int k = 0; k < ncores; ++k) group_cores.push_back(core_ids[k]);
        }

        if (group_cores.empty()) continue;

        std::sort(group_cores.begin(), group_cores.end());
        group_cores.erase(std::unique(group_cores.begin(), group_cores.end()), group_cores.end());

        int group_global_index = group_team_base + group_index;

        int status = sTiles::Control::CreatePersistentTeam((int)group_cores.size(), group_cores.data(), group_global_index);
        if (status != 0) {
            sTiles::Logger::error("Failed to create group team for group ", group_index);
            return status;
        }
    }

    return 0;
}

} } // namespace sTiles::Binding


//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//___________________________________________________________________________Preprocess____(Sparse & Dense)_____________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________

int sTiles_preprocess_initialization(sTiles_call **call_info, TiledMatrix **scheme, int debug, int group_index) {


    sTiles::Logger::info("│                                                            ");
    sTiles::Logger::info("│           ─────────   Call Index "+ std::to_string(debug) +"  ─────────              ");
    sTiles::Logger::info("│ ↳ Preprocessing phase 1 ...                                ");

    // Validate input pointers
    if (call_info == NULL || *call_info == NULL) {
        sTiles::Logger::error("Invalid pointer: sTiles_call is null.");
        return -1;
    }

    sTiles::Logger::info("│   [✔] sTiles configuration allocation successful.          ");

    // Allocate memory for the scheme using memory manager
    *scheme = MemoryManager::allocateZero<TiledMatrix>(1, group_index);
    if (*scheme == NULL) {
        sTiles::Logger::error("Memory allocation failed for TiledMatrix.");
        return -1;
    }
    // TiledMatrix has non-trivial C++ members (std::string selected_ordering,
    // std::vector partitions, shared_ptr task lists); calloc'd memory is NOT a
    // constructed object. A zeroed std::string has _M_p == NULL, and assigning
    // an EMPTY string to it writes the '\0' terminator through that null
    // pointer (SEGV) — hit by every single-tile model (order <= tile_size,
    // variant 1), whose primary skips the ordering tournament and leaves
    // selected_ordering empty for the secondary-call copy. Placement
    // value-init keeps all POD fields zero (same as calloc) but constructs
    // the C++ members validly. Must run BEFORE any field writes below.
    new (*scheme) TiledMatrix();
    // Raw zeroed allocation: mark the per-scheme tile mode as unresolved.
    // sTiles_preprocess_group snapshots the resolved mode at the end of
    // preprocessing; compute-phase code reads it via stiles_scheme_tile_mode.
    (*scheme)->tile_type_mode = -1;

    // Check if preprocessing is required
    if ((*call_info)->sequence_id < 1) {

        // Initialize basic parameters
        (*call_info)->tile_size = stiles_control_params[sTiles::param::UserTileSize]; //DEFAULT_TILE_SIZE;
        // Slot 2 (ordering mode) is deprecated: the ordering bake-off selects
        // its candidates adaptively. ordering_strategy is kept only for
        // logging in the variant-2 path.
        (*call_info)->ordering_strategy = 0;
        if((*call_info)->order < 1000){
            stiles_control_params[sTiles::param::TileOrderingStrategy] = 0;
            sTiles::Logger::debug("│   ↪ Matrix size < 1000 → Disabling tile-level ordering     ");
        }

        sTiles::Logger::debug("│     • Initial tile size = " + std::to_string((*call_info)->tile_size));
        sTiles::Logger::debug("│     • Initial ordering startegy = " + std::to_string((*call_info)->ordering_strategy));

        // Handle special ordering cases
        // if ((*call_info)->ordering_strategy == 4) {
        //     int min_cores = (*call_info)->parameters[static_cast<int>(sTiles::Parameter::AndMinCores)];
        //     if ((*call_info)->num_cores < min_cores) {
        //         (*call_info)->ordering_strategy = 1;
        //         sTiles::Logger::debug("│   ↪ Ordering 4 downgraded to 1 (number of cores < " + std::to_string(min_cores) + ")");
        //     } else {
        //         (*call_info)->use_nested_dissection = true;
        //         sTiles::Logger::debug("│   ↪ Ordering 4 accepted → Enabling Nested Dissection (ND)  ");
        //     }
        // }

        if ((*call_info)->factorization_variant == 1) {
            (*call_info)->tile_size = (*call_info)->order;
            (*call_info)->ordering_strategy = 0;
            stiles_control_params[sTiles::param::TileOrderingStrategy] = 0;
            sTiles::Logger::debug("│   ↪ Factorization Type = 1 → tile size set to N = " + std::to_string((*call_info)->tile_size));
        }

        if((*call_info)->factorization_variant==2){
            (*call_info)->ordering_strategy = 0;
            stiles_control_params[sTiles::param::TileOrderingStrategy] = 0;
            sTiles::Logger::debug("│   ↪ Factorization Type = 2 → ordering forced to 0          ");
        }
        

        // Configure based on matrix type
        // Variant 1: small dense matrix (order <= tile_size means single tile)
        if ((*call_info)->order <= (*call_info)->tile_size) {

            // Dense matrix configuration
            (*call_info)->factorization_variant = 1;
            (*call_info)->ordering_strategy = 0;
            stiles_control_params[sTiles::param::TileOrderingStrategy] = 0;
            (*call_info)->use_nested_dissection = false;
            (*call_info)->tile_size = (*call_info)->order;

            sTiles::Logger::info("│   ↪ Detected dense matrix configuration.                   ");
            sTiles::Logger::debug("│     • N <= tile size → tile size set to N = " + std::to_string((*call_info)->order));
            sTiles::Logger::debug("│     • Factorization Type set to 1                          ");
            sTiles::Logger::debug("│     • Ordering  set to 0                                   ");
            sTiles::Logger::debug("│     • Nested Dissection (ND) disabled.                     ");

        } else {

            // Sparse matrix configuration
            sTiles::Logger::info("│   ↪ Detected sparse matrix configuration.                  ");

            // Sparse matrix configuration
            if ((*call_info)->num_cores < 2) {
                (*call_info)->red_tree_separator_level = 0;
                sTiles::Logger::debug("│     • Number of cores < 2 → Red Tree Separator set to 0    ");

            } else {
                int thick = (*call_info)->arrowhead_thickness;
                int tile = (*call_info)->tile_size;
                int n = (*call_info)->order;
                int num_cores = (*call_info)->num_cores;

                (*call_info)->red_tree_separator_level = (thick + tile - 1) / tile;
                int num_tiles = n / tile;
                if (n % tile != 0) num_tiles += 1;

                if ((num_tiles - 1) < 2 * num_cores) {
                    (*call_info)->red_tree_separator_level = 0;
                    sTiles::Logger::debug("│     • (N / Tile size - 1) < 2 * Number of cores → Red Tree Separator set to 0");
                } else {
                    sTiles::Logger::debug("│     • Red Tree Separator = (" + std::to_string(thick) + " + " + std::to_string(tile) + " - 1) / " + std::to_string(tile) + " = " + std::to_string((*call_info)->red_tree_separator_level));
                }
            }
        }

        // Set preprocessing level
        (*call_info)->preprocess_level = 1;
        sTiles::Logger::info("│   ↪ Setting preprocessing level to 1 ");

        // Initialize scheme parameters based on call_info
        (*scheme)->dim = (*call_info)->order;
        (*scheme)->nnz = (*call_info)->nnz;
        (*scheme)->original_order = (*call_info)->order;
        (*scheme)->original_nnz = (*call_info)->nnz;
        (*scheme)->tile_size = (*call_info)->tile_size;

        // Ensure tile_size does not exceed matrix dimension
        if ((*scheme)->tile_size > (*scheme)->dim) {
            (*scheme)->tile_size = (*scheme)->dim;
        }

        sTiles::Logger::debug("│     • N              = " + std::to_string((*scheme)->dim));
        sTiles::Logger::debug("│     • nnz            = " + std::to_string((*scheme)->nnz));
        sTiles::Logger::debug("│     • tile size used = " + std::to_string((*scheme)->tile_size));

        // Calculate total tiles and size parameters
        (*scheme)->dimTiledMatrix = ((*scheme)->dim % (*scheme)->tile_size == 0) ? ((*scheme)->dim / (*scheme)->tile_size) : ((*scheme)->dim / (*scheme)->tile_size + 1);
        (*scheme)->triangular_size = (((*scheme)->dimTiledMatrix * (*scheme)->dimTiledMatrix) - (*scheme)->dimTiledMatrix) / 2 + (*scheme)->dimTiledMatrix;
        (*scheme)->fixed_column_size = (*call_info)->arrowhead_thickness;

        sTiles::Logger::debug("│     • tiles dim      = " + std::to_string((*scheme)->dimTiledMatrix));
        sTiles::Logger::debug("│     • off_on_size    = " + std::to_string((*scheme)->triangular_size));
        sTiles::Logger::debug("│     • fixed columns  = " + std::to_string((*scheme)->fixed_column_size));
        sTiles::Logger::debug("│   [✔] Permutation vector allocation succeeded ");


        (*scheme)->permutation_flags = NULL;
        (*scheme)->partition_sizes = NULL;
        (*scheme)->new_partition_sizes = NULL;
        (*scheme)->nd_nnz = 0;
        (*scheme)->nd_order = 0;
        (*scheme)->nd_padding = 0;
        (*scheme)->scotch_partition_collection = false;
        (*scheme)->scotch_root_sep_tiles = 0;
        (*scheme)->element_perm = NULL;
        (*scheme)->element_iperm = NULL;
        (*scheme)->red_tree_separator_level = (*call_info)->red_tree_separator_level;
        // Variant 1 and 2 don't use boosted_e_trick (they use direct DPOTRF calls)
        (*scheme)->use_boosted_e_trick = ((*call_info)->factorization_variant == 0 || (*call_info)->factorization_variant == 3);
        (*scheme)->internal_version = 101;
        (*scheme)->num_cores = (*call_info)->num_cores;
        (*scheme)->num_gpu_streams = (*call_info)->num_cores;
        (*scheme)->copy_on_preprocess = true;
        (*scheme)->compute_inverse = (*call_info)->compute_inverse;
        (*scheme)->neighbor_lookup_method =
            (user_neighbor_lookup_method == TileIndexer::Method::CharMask)
                ? sTiles::policy::pick_indexer_method((*call_info)->order, (*call_info)->tile_size)
                : user_neighbor_lookup_method;
        (*scheme)->compute_inverse_on_gpu = false;
        (*scheme)->use_gpu = false;
        (*scheme)->use_banded_gpu = false;

        #if defined(STILES_USE_CSC_INDEX)
            (*scheme)->cscIndex = nullptr;
            (*scheme)->cscIndex_built = false;
        #endif

        // Set null pointers for structures not yet initialized
        (*scheme)->trees = NULL;
        (*scheme)->solve_trick_type0 = NULL;
        (*scheme)->solve_trick_type1 = NULL;
        (*scheme)->tree_counter = NULL;
        (*scheme)->timings = MemoryManager::allocateZero<double>(2, group_index);
        (*scheme)->nd_padding = 0;

    #ifdef STILES_GPU
        (*scheme)->GPU_ID = 0;
        (*scheme)->gpu_trees = NULL;
        (*scheme)->gpu_persistent_ctx = nullptr;  // Initialized during GPU tile allocation
    #endif

        // Increment preprocessing tick
        (*call_info)->sequence_id++;
    }

return 0;
}


// Symbolic-only preprocessing: when set, sTiles_init / sTiles_init_group run the
// ordering + symbolic phase (populating scheme->element_perm) but STOP before
// committing the big numeric factor arena. Lets a caller (e.g. an MPI rank) get
// the permutation cheaply and ship it, deferring the numeric allocation. Gated
// at the two arena entry points so it covers every numeric mode.
static int g_symbolic_only = 0;

// Symbolic-ready callback: if set, fired from sTiles_preprocess_group the instant
// group 0's ordering+symbolic is done (element_perm ready) -- BEFORE the numeric
// arena. Lets an MPI rank ship the perm mid-init and overlap the rest. NULL =
// no-op (default), so this is inert unless a caller registers it.
static void (*g_symbolic_done_cb)(int group, int call, const int *perm, int n) = NULL;

// =============================================================================
// Primary-call helpers split out of sTiles_preprocess_group by factorization
// variant. The orchestrator (sTiles_preprocess_group) handles shared scaffolding
// — argument validation, primary-call init, memory estimate + GPU dispatch,
// secondary-call cloning — and delegates the variant-specific work to these.
//
//   variants 0, 3 (sparse path)        → preprocess_primary_sparse
//   variants 1, 2 (dense / scaled)     → preprocess_primary_dense
//   tile allocation (all variants)     → preprocess_primary_alloc_tiles
// =============================================================================

// Variants 1 (full dense) and 2 (scaled dense): no ordering, no ND, no symbolic
// phase. Initialize scheme fields, allocate workspaces, then build the
// variant-specific dense tile lookup. Called BEFORE the GPU dispatch block.
static int preprocess_primary_dense(TiledMatrix* scheme,
                                   sTiles_call* call_info,
                                   int group_index,
                                   int rescale_total_cores) {
    // Dense/scaled-dense variants do not require ordering or ND.
    call_info->ordering_strategy = 0;
    call_info->use_nested_dissection = false;
    scheme->use_ordering = 0;
    scheme->nd_padding = 0;
    scheme->nd_nnz = 0;
    scheme->nd_order = 0;
    scheme->red_tree_separator_level = 0;
    scheme->scotch_partition_collection = false;
    scheme->scotch_root_sep_tiles = 0;
    scheme->element_perm = nullptr;
    scheme->element_iperm = nullptr;
    scheme->permutation_flags = nullptr;

    // Set numActiveTiles and dimTiledMatrix based on variant
    if (call_info->factorization_variant == 1) {
        // Variant 1: Entire matrix as one dense tile
        scheme->numActiveTiles = 1;
        scheme->dimTiledMatrix = 1;
        scheme->triangular_size = 1;
    } else if (call_info->factorization_variant == 2){
        // Variant 2: Scaled-dense uses upper triangular tiles
        scheme->numActiveTiles = scheme->triangular_size;
    }

    // Dense variants don't use trees or sparse structures
    scheme->trees = nullptr;
    scheme->tree_counter = nullptr;

    scheme->sparseTileMetaCore = nullptr;
    scheme->invSparseTileMetaCore = nullptr;
    scheme->sparseTileMetaData = nullptr;
    scheme->nnz_tile_counter = nullptr;

    scheme->semisparseTileMetaCore = nullptr;

    scheme->e_trick = nullptr;
    scheme->e_trick_size = nullptr;
    scheme->e_trick_inv = nullptr;
    scheme->e_trick_copy_ind = nullptr;
    scheme->e_trick_size_inv = nullptr;

    scheme->symbolicTileBitmaskCore = nullptr;
    scheme->workspace_bit_array = nullptr;
    scheme->on_off_tiles = nullptr;

    scheme->solve_trick_type0 = nullptr;
    scheme->solve_trick_type1 = nullptr;

    scheme->t_indicies = nullptr;
    scheme->e_indicies = nullptr;
    scheme->withinTileRow = nullptr;
    scheme->withinTileCol = nullptr;
    scheme->tileIndexMapper = nullptr;
    scheme->tileIndexMapper2 = nullptr;
    scheme->partition_sizes = nullptr;
    scheme->new_partition_sizes = nullptr;
    scheme->diagonal_mapper = nullptr;

    // Workspace allocation (dense variants skip the symbolic phase that would
    // normally allocate them).
    {
        const int world =   std::max(rescale_total_cores, call_info->num_cores);
        const int workspace_dim = (scheme->tile_size > 0) ? scheme->tile_size : 1;
        std::size_t slots = static_cast<std::size_t>(world);
        scheme->workspaces = static_cast<sTiles::Workspace**>(std::calloc(slots, sizeof(sTiles::Workspace*)));
        if (!scheme->workspaces) {
            sTiles::Logger::error("Failed to allocate workspace slots for dense variant.");
            return EXIT_FAILURE;
        }
        scheme->num_workspaces = world;
        for (int r = 0; r < world; ++r) {
            scheme->workspaces[r] = new sTiles::Workspace(group_index, workspace_dim);
        }
        sTiles::Logger::debug("│   ↪ Allocated ", world, " workspaces for dense variant (dim=", workspace_dim, ")");
    }

    // Dense tile arrays (allocated later by preprocess_primary_alloc_tiles)
    scheme->denseTiles = nullptr;
    scheme->inverseTiles = nullptr;
    scheme->savedTiles = nullptr;
    scheme->rhTiles = nullptr;
    scheme->tileMetaCore = nullptr;
    scheme->tile_index_lookup = nullptr;
    scheme->element_offset_lookup = nullptr;
    scheme->diagonal_bmapper = nullptr;

    scheme->chunkedDenseTiles = nullptr;
    scheme->chunkedRhsTiles = nullptr;
    scheme->chunkedSavedTiles = nullptr;
    scheme->chunkedInverseTiles = nullptr;

    scheme->mapper = TileIndexer::Mapper{};
    scheme->state = TileIndexer::State{};

    sTiles::Logger::debug("│   ↪ Dense factorization variant detected → skipping ordering and ND.");

    // Build the variant-specific dense tile lookup
    {
        const double t0 = omp_get_wtime();
        if (call_info->factorization_variant == 1) {
            if (sTiles::preprocess::build_dense_tile_lookup_variant1(&call_info, scheme, group_index) != sTiles::StatusCode::Success) {
                sTiles::Logger::error("Error: Dense tile lookup failed (variant 1).");
                return EXIT_FAILURE;
            }
            sTiles::Logger::timing("│   ↪ Build dense tile lookup (variant 1): ", (omp_get_wtime() - t0), " s");
        } else if (call_info->factorization_variant == 2) {
            if (sTiles::preprocess::build_dense_tile_lookup_variant2(&call_info, scheme, group_index) != sTiles::StatusCode::Success) {
                sTiles::Logger::error("Error: Dense tile lookup failed (variant 2).");
                return EXIT_FAILURE;
            }
            sTiles::Logger::timing("│   ↪ Build dense tile lookup (variant 2): ", (omp_get_wtime() - t0), " s");
        }
    }

    return EXIT_SUCCESS;
}

// Variants 0 and 3 (sparse path): symbolic factorization + ordering, then the
// full sparse tile-lookup pipeline (mapper, bind-active, leaves, tree, dense+
// semisparse lookup, task collection, semisparse tile alloc, workspaces, and
// optional gather/scatter info). Called BEFORE the GPU dispatch block.
static int preprocess_primary_sparse(sTiles_call*& call_info,
                                    TiledMatrix*& scheme,
                                    TiledMatrix** stiles_schemes,
                                    int global_index,
                                    int group_index,
                                    int eff_cores,
                                    int num_threads_level1,
                                    int rescale_cores,
                                    int rescale_total_cores,
                                    bool force_nd) {
    // ── Auto-mode (mode 3) resolution ─────────────────────────────────────
    // tile_type_mode == 3 means "pick the best mode based on the symbolic
    // factorization's fill ratio". We run symbolic now so we can read
    // nnz(L), then rewrite stiles_control_params[3] to either 1 (semisparse)
    // for low fill or 2 (non-uniform) for high fill. Slaves and downstream
    // dispatch read param[3] and follow the resolved mode.
    int tile_type_mode = stiles_control_params[sTiles::param::TileTypeMode];
    bool symbolic_already_run = false;
    if (tile_type_mode == 3) {
        const double t0 = omp_get_wtime();
        if (force_nd) {
            if (sTiles::preprocess::symbolic_ND_phase(&call_info, &scheme, group_index, eff_cores) != sTiles::StatusCode::Success) {
                sTiles::Logger::error("Error: ND symbolic factorization failed (auto-mode resolution).");
                return EXIT_FAILURE;
            }
        } else {
            if (sTiles::preprocess::symbolic_phase(&call_info, &scheme, group_index, eff_cores) != sTiles::StatusCode::Success) {
                sTiles::Logger::error("Error: Symbolic factorization failed (auto-mode resolution).");
                return EXIT_FAILURE;
            }
        }
        sTiles::Logger::timing("│   ↪ Symbolic phase (auto-mode resolution): ", (omp_get_wtime() - t0), " s");
        symbolic_already_run = true;

        // Decide mode from fill ratio. Empirical crossover ~3.5× across the
        // matrix suite tested (FEM / INLA / network). Below → semisparse
        // wins on numeric+inverse; above → non-uniform wins.
        const double kAutoFillThreshold = 3.5;
        const long long input_nnz = scheme->nnz > 0 ? scheme->nnz : 1;
        const double fill = static_cast<double>(scheme->nnz_factor) /
                            static_cast<double>(input_nnz);
        // Block occupancy (mean tile density) drives both the dense and the
        // semisparse branches, so compute it once up front.
        const double occ = mean_occupancy_from_L(scheme, call_info->tile_size);
        const double skew = degree_skew_from_L(scheme);   // arrowhead/hub signal
        // Dense branch (NEW): a NEAR-DENSE matrix (occ >= kDenseOcc) belongs in the
        // regular tiled-dense path (mode 0), NOT sparse supernodal. The auto selector
        // never picked dense before, so near-dense matrices were mis-routed to mode 2
        // and COLLAPSED at high cores (validated: animal1 occ=0.97 -> sparse 8x slower
        // at 22c, dense stays flat + logdet exact). Default 0.90 catches the animal1
        // class only; the nd-family (occ<=0.79, sparse-best at their optimum) stays out.
        // Env-tunable (STILES_DENSE_OCC) for calibration (e.g. testing animal2 @0.867);
        // keep it above ~0.82 so the nd-family is never pulled in.
        double kDenseOcc = 0.90;
        if (const char* e = std::getenv("STILES_DENSE_OCC")) { const double v = std::atof(e); if (v > 0.0) kDenseOcc = v; }
        int resolved;
        if (occ >= kDenseOcc) {
            resolved = 0;                              // near-dense -> tiled dense (scales; avoids supernodal collapse)
        } else if (fill >= kAutoFillThreshold) {
            resolved = 2;                              // high fill -> sparse (unambiguous)
        } else if (std::getenv("STILES_LEGACY_FILL")) {
            resolved = 1;                              // opt-out: old fill-only rule (low fill -> semi)
        } else {
            // Low-fill routing (DEFAULT): semisparse only when tiles are DENSE (block-density
            // >= kOcc, clustered/structured) AND fill is LOW (<= kFillMax). That isolates the
            // INLA-precision class (sem_n*); scattered/hub graphs (low density) and dense-but-
            // high-fill FEM (gyro/msc/nasa) both go to sparse. Validated 17/18 on the suite.
            const double kOccThreshold = 0.15;
            const double kFillMaxSemi  = 2.0;
            // Arrowhead/bordered matrices (INLA precisions like bern, pedigree) are
            // globally sparse -> low mean block density (occ < kOcc), so the occ floor
            // alone mis-routes them to supernodal even though their dense border makes
            // them semisparse-ideal. Admit them by degree skew: arrowheads run 500--1000x,
            // uniform meshes/graphs ~1.5x (validated bern/pedigree vs thermal1/net*), so a
            // >=50 cutoff catches the bordered class without pulling in the uniformly
            // sparse, supernodal-best matrices.
            double kArrowheadSkew = 20.0;
            if (const char* e = std::getenv("STILES_ARROWHEAD_SKEW")) { const double v = std::atof(e); if (v > 0.0) kArrowheadSkew = v; }
            const bool has_border = (skew >= kArrowheadSkew);
            resolved = (fill <= kFillMaxSemi && (occ >= kOccThreshold || has_border)) ? 1 : 2;
        }
        // Routing decision (always-on): which numeric mode auto-mode resolved to,
        // and the signals behind it. Visible in every run so a mis-route is obvious.
        {
            const char* mode_name = (resolved == 0 ? "dense"
                                   : resolved == 1 ? "semisparse" : "sparse");
            char sig[96];
            std::snprintf(sig, sizeof(sig), "occ=%.3f fill=%.2f skew=%.1f", occ, fill, skew);
            sTiles::Logger::timing_always("│   ↪ Mode auto-select: ", sig, " -> ",
                mode_name, " (mode ", resolved, ")");
        }
        stiles_control_params[sTiles::param::TileTypeMode] = resolved;
        tile_type_mode = resolved;

        // ── Diagnostic: dump the work/parallelism signals and EXIT (env-gated).
        // STILES_DUMP_NEIGHBOR=1 prints per-matrix signals after symbolic (no
        // factorization) so a quick sweep over ALL matrices calibrates the
        // work-aware core cap (max_neighbor = peak fan-out ~ peak parallelism).
        if (std::getenv("STILES_DUMP_NEIGHBOR")) {
            const int ts_now  = call_info->tile_size;
            const int ntiles  = (scheme->dim + ts_now - 1) / ts_now;
            const int maxnb   = max_neighbor_tiles_from_L(scheme, ts_now);
            const double occ2 = mean_occupancy_from_L(scheme, ts_now);
            fprintf(stderr,
                "[NEIGHBOR-DUMP] n=%d tiles_dim=%d ts=%d nnz_factor=%lld fill=%.2f occ=%.4f mode=%d "
                "max_neighbor=%d peak_par=%lld\n",
                scheme->dim, ntiles, ts_now, static_cast<long long>(scheme->nnz_factor),
                fill, occ2, resolved, maxnb, static_cast<long long>(maxnb) * maxnb);
            fflush(stderr);
            std::exit(0);   // skip the factorization -> cover all matrices fast
        }

        // ── Adaptive tile size (tile-size sweep, 2026-06) ─────────────────
        // fill + mode are known, so pick the tile size that the sweep showed
        // wins for this regime. The ordering AND the fill (L_colptr/L_rowind)
        // are tile-size-INDEPENDENT, so changing the tile size needs only the
        // tile-dependent state re-counted (numActiveTiles + the active mask),
        // NOT a re-run of the (expensive) ordering search. The downstream
        // tiling chain (constructMapper -> build_*_lookup) rebuilds the rest.
        // Skipped under force_nd (tile-aligned ND padding would need re-padding).
        if (!force_nd) {
            int new_ts = call_info->tile_size;
            if (resolved == 1)                              new_ts = 120; // semisparse / structured INLA (sem_n*: ~3x)
            else if (resolved == 0)                                       // near-dense tiled (animal2): 40x40 tiles starve BLAS-3
                new_ts = std::max(call_info->tile_size, sTiles_get_auto_tile_size()); // cache-driven size (80/120); validated ts120 -> animal2 1.7-1.9x
            else if (scheme->dim > 500000 && fill >= 25.0)  new_ts = 120; // large dense FEM (audikw/bone/Fault: 1.3-1.5x)
            if (new_ts != call_info->tile_size &&
                scheme->L_colptr && scheme->L_rowind && scheme->nnz_factor > 0) {
                sTiles::Logger::timing("│   ↪ Adaptive tile size: ", call_info->tile_size,
                                       " -> ", new_ts, " (re-tile, no re-order)");
                call_info->tile_size   = new_ts;
                scheme->tile_size      = new_ts;
                scheme->dimTiledMatrix = (scheme->dim + new_ts - 1) / new_ts;
                // triangular_size is derived from dimTiledMatrix and MUST be recomputed
                // at the new tile size -- the dense path (mode 0) sets numActiveTiles from
                // it, so a stale ts40 value against a ts120 grid corrupts the factor.
                scheme->triangular_size = (scheme->dimTiledMatrix * scheme->dimTiledMatrix
                                           - scheme->dimTiledMatrix) / 2 + scheme->dimTiledMatrix;
                // The supernodal-sparse path (mode 2) builds its own structure and
                // only reads tile_size for the snode_cap, so setting tile_size above is
                // all it needs. BOTH the semisparse (mode 1) and the dense (mode 0)
                // routes consume the TileIndexer active-tile state downstream
                // (constructMapper, build_dense_tile_lookup, collect_tasks), so the
                // state must be re-counted at the new size from the (tile-independent) L
                // -- otherwise the dense lookup is built against the stale ts grid and
                // the factor is garbage (factor=0, logdet=-1). For a dense matrix every
                // tile comes back active, so the same rebuild yields numActiveTiles ==
                // triangular_size. L_row = L_rowind directly; build L_col.
                if (resolved == 1 || resolved == 0) {
                    const int        n2 = scheme->dim;
                    const long long  nL = scheme->nnz_factor;
                    std::vector<int> L_col(static_cast<std::size_t>(nL));
                    for (int j = 0; j < n2; ++j)
                        for (long long p = scheme->L_colptr[j]; p < scheme->L_colptr[j + 1]; ++p)
                            L_col[static_cast<std::size_t>(p)] = j;
                    TileIndexer::release_state_resources(scheme->state);
                    scheme->numActiveTiles = TileIndexer::countActiveTiles(
                        scheme->L_rowind, L_col.data(), static_cast<int>(nL),
                        n2, new_ts, scheme->neighbor_lookup_method, &scheme->state, eff_cores);
                    if (scheme->numActiveTiles >= 0)
                        TileIndexer::ensure_diagonal_tiles_active(
                            scheme->state, scheme->neighbor_lookup_method,
                            scheme->dimTiledMatrix, scheme->numActiveTiles);
                }
            }
        }
        sTiles::Logger::timing("│   ↪ Auto mode: fill=", fill, "× → using ",
                               (resolved == 2 ? "non-uniform tiles (mode 2)"
                                              : "semisparse tiles (mode 1)"));
    }

    // ── Non-uniform tile path (mode 2) ────────────────────────────────────
    // Reached either because the user picked mode 2 directly, or because
    // mode-3 auto resolved to it above. The non-uniform path requires a
    // user permutation, so we run sTiles' ordering first (unless the
    // auto-resolution block already did) to produce scheme->element_perm,
    // then hand it over to sTiles::sparse.
    if (tile_type_mode == 2) {
        sTiles::Logger::info("│ ↪ Routing to sTiles::sparse (variant=0, tile_type_mode=2)");

        // Run the symbolic phase first so element_perm + L_colptr are
        // populated — unless auto-mode already ran it above.
        if (!symbolic_already_run) {
            const double t0 = omp_get_wtime();
            if (force_nd) {
                if (sTiles::preprocess::symbolic_ND_phase(&call_info, &scheme, group_index, eff_cores) != sTiles::StatusCode::Success) {
                    sTiles::Logger::error("Error: ND symbolic factorization failed (sparse module path).");
                    return EXIT_FAILURE;
                }
            } else {
                if (sTiles::preprocess::symbolic_phase(&call_info, &scheme, group_index, eff_cores) != sTiles::StatusCode::Success) {
                    sTiles::Logger::error("Error: Symbolic factorization failed (sparse module path).");
                    return EXIT_FAILURE;
                }
            }
            sTiles::Logger::timing("│   ↪ Symbolic phase (for sparse module): ", (omp_get_wtime() - t0), " s");
        }

        if (g_symbolic_only) {
            // symbolic-only: scheme->element_perm is ready; defer the numeric
            // arena (api::create/assign_graph -> CellStore::allocate).
            return EXIT_SUCCESS;
        }

        void* h = nullptr;
        if (sTiles::sparse::api::create(&h, call_info->num_cores) != 0) {
            sTiles::Logger::error("sTiles::sparse::api::create failed.");
            return EXIT_FAILURE;
        }
        sTiles::sparse::api::set_group_id(&h, group_index);
        // Adaptive supernode-width cap (per-CPU ceiling, STILES_SNODE_CAP override,
        // parallelism-bounded). Replaces the fixed 2*tile_size that fragmented wide
        // natural supernodes on dense-FEM into many small GEMMs. RelaxSupernodes still
        // merges only up to each matrix's natural width, so narrow cases are unchanged.
        {
            const int snode_cap = adaptive_snode_cap(call_info->tile_size, call_info->num_cores, scheme->dim);
            sTiles::sparse::api::set_max_supernode(&h, snode_cap);
        }
        if (sTiles::sparse::api::set_user_permutation(&h, scheme->element_perm, scheme->dim) != 0) {
            sTiles::Logger::error("sTiles::sparse::api::set_user_permutation failed.");
            sTiles::sparse::api::freeGroup(&h);
            return EXIT_FAILURE;
        }
        if (sTiles::sparse::api::assign_graph(&h, scheme->dim, scheme->nnz,
                                        call_info->row_indices, call_info->col_indices) != 0) {
            sTiles::Logger::error("sTiles::sparse::api::assign_graph failed.");
            sTiles::sparse::api::freeGroup(&h);
            return EXIT_FAILURE;
        }
        scheme->sparse_handle = h;
        return EXIT_SUCCESS;
    }

    // Symbolic phase (skipped if auto-mode resolution already ran it).
    if (!symbolic_already_run) {
        const double t0 = omp_get_wtime();
        if (force_nd) {
            if (sTiles::preprocess::symbolic_ND_phase(&call_info, &scheme, group_index, eff_cores) != sTiles::StatusCode::Success) {
                sTiles::Logger::error("Error: ND symbolic factorization failed.");
                return EXIT_FAILURE;
            }
            sTiles::Logger::timing("│   ↪ Symbolic ND phase total: ", (omp_get_wtime() - t0), " s");
        } else {
            if (sTiles::preprocess::symbolic_phase(&call_info, &scheme, group_index, eff_cores) != sTiles::StatusCode::Success) {
                sTiles::Logger::error("Error: Symbolic factorization failed.");
                return EXIT_FAILURE;
            }
            sTiles::Logger::timing("│   ↪ Symbolic phase total: ", (omp_get_wtime() - t0), " s");
        }
    }

    // Sparse variant preprocessing pipeline
    {
        const double t0 = omp_get_wtime();
        if (sTiles::preprocess::constructMapper(&stiles_schemes[global_index]) != sTiles::StatusCode::Success) {
            sTiles::Logger::error("Error: Failed to construct TileIndexer mapper.");
            return EXIT_FAILURE;
        }
        sTiles::Logger::timing("│   ↪ Construct mapper: ", (omp_get_wtime() - t0), " s");
    }

    {
        const double t0 = omp_get_wtime();
        if (sTiles::preprocess::bindActive(&stiles_schemes[global_index]) != sTiles::StatusCode::Success) {
            sTiles::Logger::error("Error: Failed to bind TileIndexer isActive.");
            return EXIT_FAILURE;
        }
        sTiles::Logger::timing("│   ↪ Bind active predicates: ", (omp_get_wtime() - t0), " s");
    }

    {
        const double t0 = omp_get_wtime();
        if (sTiles::preprocess::corner_probe(&call_info, &stiles_schemes[global_index], group_index) != sTiles::StatusCode::Success) {
            sTiles::Logger::error("Error: Corner probe failed.");
            return EXIT_FAILURE;
        }
        sTiles::Logger::timing("│   ↪ Corner probe: ", (omp_get_wtime() - t0), " s");
    }

    {
        const double t0 = omp_get_wtime();
        if (sTiles::preprocess::leaves_counter(&call_info, &stiles_schemes[global_index], group_index) != sTiles::StatusCode::Success) {
            sTiles::Logger::error("Error: Tree leaves counting failed.");
            return EXIT_FAILURE;
        }
        sTiles::Logger::timing("│   ↪ Count tree leaves: ", (omp_get_wtime() - t0), " s");
    }

    {
        const double t0 = omp_get_wtime();
        if (sTiles::preprocess::tree_creation(&call_info, &stiles_schemes[global_index], group_index) != sTiles::StatusCode::Success) {
            sTiles::Logger::error("Error: Tree creation failed.");
            return EXIT_FAILURE;
        }
        sTiles::Logger::timing("│   ↪ Build elimination tree: ", (omp_get_wtime() - t0), " s");
    }

    {
        const double t0 = omp_get_wtime();
        sTiles::preprocess::build_dense_tile_lookup(&call_info, stiles_schemes[global_index], group_index, eff_cores);
        sTiles::Logger::timing("│   ↪ Dense tile lookup: ", (omp_get_wtime() - t0), " s");
    }

    {
        const double t0 = omp_get_wtime();
        sTiles::preprocess::build_semisparse_tile_lookup(&call_info, stiles_schemes[global_index], group_index, eff_cores);
        sTiles::Logger::timing("│   ↪ Semisparse tile lookup: ", (omp_get_wtime() - t0), " s");
    }

    {
        const double t0 = omp_get_wtime();
        if (sTiles::preprocess::collect_tasks(&call_info, &stiles_schemes[global_index], group_index, eff_cores, num_threads_level1, rescale_cores) != sTiles::StatusCode::Success) {
            sTiles::Logger::error("Error: Task collection failed.");
            return EXIT_FAILURE;
        }
        sTiles::Logger::timing("│   ↪ Collect numeric tasks: ", (omp_get_wtime() - t0), " s");
    }

    {
        const double t0 = omp_get_wtime();
        sTiles::preprocess::allocate_semisparse_tiles(stiles_schemes[global_index], group_index, eff_cores);
        sTiles::Logger::timing("│   ↪ Allocate semisparse tiles: ", (omp_get_wtime() - t0), " s");
    }

    // Allocate per-thread workspaces for the sparse variant.
    {
        TiledMatrix* sparse_scheme = stiles_schemes[global_index];
        const int world = std::max(rescale_total_cores, call_info->num_cores);
        if (!sparse_scheme->workspaces || sparse_scheme->num_workspaces < world) {
            if (sparse_scheme->workspaces) {
                for (int r = 0; r < sparse_scheme->num_workspaces; ++r) {
                    delete sparse_scheme->workspaces[r];
                }
                std::free(sparse_scheme->workspaces);
            }
            const int workspace_dim = (sparse_scheme->tile_size > 0) ? sparse_scheme->tile_size : 1;
            std::size_t slots = static_cast<std::size_t>(world);
            sparse_scheme->workspaces = static_cast<sTiles::Workspace**>(std::calloc(slots, sizeof(sTiles::Workspace*)));
            if (sparse_scheme->workspaces) {
                sparse_scheme->num_workspaces = world;
                for (int r = 0; r < world; ++r) {
                    sparse_scheme->workspaces[r] = new sTiles::Workspace(group_index, workspace_dim);
                }
            } else {
                sparse_scheme->num_workspaces = 0;
            }
        }
    }

    // Build precomputed gather/scatter info for semisparse tasks (after
    // allocate_semisparse_tiles populates aind/acol).
    // NOTE: params[7] (inverse storage mode) only governs the INVERSE gather
    // tables. The chol scatter tables are REQUIRED by the semisparse chol
    // executors regardless (their runtime fallback was removed), so gating
    // them on params[7] made tile_mode 1 + dense-inverse-storage segfault in
    // pdpotrf (null chol_scatter_index). Build them for any semisparse mode.
    {
        const double t_pre = omp_get_wtime();
        bool built = false;
        if (stiles_control_params[sTiles::param::InverseStorageMode] == 1 && stiles_schemes[global_index]->compute_inverse) {
            sTiles::preprocess::build_inv_gather_info(stiles_schemes[global_index]);
            built = true;
        }
        if (stiles_control_params[sTiles::param::InverseStorageMode] == 1 ||
            stiles_control_params[sTiles::param::TileTypeMode] == 1 || stiles_control_params[sTiles::param::TileTypeMode] == 2) {
            sTiles::preprocess::build_chol_scatter_info(stiles_schemes[global_index]);
            built = true;
        }
        if (built)
            sTiles::Logger::timing("│   ↪ Build scatter/gather info: ", (omp_get_wtime() - t_pre), " s");
    }

    return EXIT_SUCCESS;
}

// Per-iteration body for secondary calls (call_index >= 1). Clones the primary
// scheme's symbolic structure into the secondary, copies variant-specific metadata
// so dense (1) and scaled-dense (2) work as well as sparse (0/3), allocates
// per-call workspaces and tile buffers. Called from inside the omp parallel-for
// in sTiles_preprocess_group; returns EXIT_SUCCESS or EXIT_FAILURE.
static int preprocess_secondary_call(int call_index,
                                 int group_index,
                                 sTiles_object** obj,
                                 TiledMatrix** stiles_schemes,
                                 int save_global_index,
                                 int eff_cores,
                                 int num_threads_level1,
                                 int rescale_cores,
                                 int rescale_total_cores) {
    sTiles_call *call_info = &(*obj)->stiles_groups[group_index].stiles_calls[call_index];
    sTiles_call *primary_call = &(*obj)->stiles_groups[group_index].stiles_calls[0];
    const int global_index = call_info->global_index;

    call_info->order = primary_call->order;
    call_info->nnz = primary_call->nnz;
    call_info->row_indices = primary_call->row_indices;
    call_info->col_indices = primary_call->col_indices;
    call_info->factorization_variant = primary_call->factorization_variant;
    call_info->tile_size = primary_call->tile_size;
    call_info->arrowhead_thickness = primary_call->arrowhead_thickness;
    call_info->parameters = primary_call->parameters;
    call_info->use_nested_dissection = primary_call->use_nested_dissection;
    call_info->red_tree_separator_level = primary_call->red_tree_separator_level;
    call_info->bandwidth = primary_call->bandwidth;
    call_info->num_right_hand_sides = primary_call->num_right_hand_sides;
    call_info->compute_inverse = primary_call->compute_inverse;
    call_info->compute_log_determinant = primary_call->compute_log_determinant;
    // Dense/scaled-dense variants run no ordering; mirror the primary's
    // ordering_strategy=0 so init doesn't pick up a stale value.
    call_info->ordering_strategy = primary_call->ordering_strategy;

    TiledMatrix *scheme = NULL;
    if (::sTiles_preprocess_initialization(&call_info, &scheme, call_index, group_index) != 0) {
        return EXIT_FAILURE;
    }

    stiles_schemes[global_index] = scheme;
    stiles_schemes[global_index]->call_lookup_table = (*obj)->call_matrix;

    TiledMatrix* primary_scheme = stiles_schemes[save_global_index];
    TiledMatrix* current_scheme = stiles_schemes[global_index];

    if (sTiles::copy_configuration_1(&primary_scheme, &current_scheme) != 0) return EXIT_FAILURE;
    if (sTiles::copy_configuration_2(&primary_scheme, &current_scheme) != 0) return EXIT_FAILURE;
    if (sTiles::copy_configuration_3(&primary_scheme, &current_scheme) != 0) return EXIT_FAILURE;
    if (sTiles::copy_configuration_4(&primary_scheme, &current_scheme) != 0) return EXIT_FAILURE;

    current_scheme->original_order = primary_scheme->original_order;
    current_scheme->original_nnz = primary_scheme->original_nnz;
    current_scheme->remainderTileSize = primary_scheme->remainderTileSize;
    current_scheme->neighbor_lookup_method = primary_scheme->neighbor_lookup_method;
    current_scheme->use_boosted_e_trick = primary_scheme->use_boosted_e_trick;
    current_scheme->compute_inverse = primary_scheme->compute_inverse;
    current_scheme->storage = primary_scheme->storage;

    current_scheme->selected_ordering = primary_scheme->selected_ordering;

    // Variant-specific scheme metadata. Primary sets these in
    // preprocess_primary_dense (variant 1: numActiveTiles/dimTiledMatrix/triangular_size=1;
    // variant 2: numActiveTiles=triangular_size). Copy them so secondary allocations
    // and tile-element accesses use the same dimensions as the primary.
    current_scheme->numActiveTiles = primary_scheme->numActiveTiles;
    current_scheme->dimTiledMatrix = primary_scheme->dimTiledMatrix;
    current_scheme->triangular_size = primary_scheme->triangular_size;
    current_scheme->tile_size = primary_scheme->tile_size;
    current_scheme->use_ordering = primary_scheme->use_ordering;
    current_scheme->nd_padding = primary_scheme->nd_padding;
    current_scheme->nd_nnz = primary_scheme->nd_nnz;
    current_scheme->nd_order = primary_scheme->nd_order;
    current_scheme->red_tree_separator_level = primary_scheme->red_tree_separator_level;
    current_scheme->element_perm = primary_scheme->element_perm;
    current_scheme->element_iperm = primary_scheme->element_iperm;
    current_scheme->permutation_flags = primary_scheme->permutation_flags;

    // Copy GPU settings from primary call
    current_scheme->use_gpu = primary_scheme->use_gpu;
    current_scheme->use_banded_gpu = primary_scheme->use_banded_gpu;
    current_scheme->GPU_ID = primary_scheme->GPU_ID;
    current_scheme->compute_inverse_on_gpu = primary_scheme->compute_inverse_on_gpu;

    current_scheme->tile_indexer_state = primary_scheme->tile_indexer_state;
    current_scheme->state = primary_scheme->state;
    current_scheme->tile_indexer_graph = primary_scheme->tile_indexer_graph;
    current_scheme->mapper = primary_scheme->mapper;
    current_scheme->diagonal_mapper = primary_scheme->diagonal_mapper;
    current_scheme->tileMetaCore = primary_scheme->tileMetaCore;
    current_scheme->sparseTileMetaCore = primary_scheme->sparseTileMetaCore;
    current_scheme->invSparseTileMetaCore = primary_scheme->invSparseTileMetaCore;
    current_scheme->sparseTileMetaData = primary_scheme->sparseTileMetaData;
    current_scheme->semisparseTileMetaCore = primary_scheme->semisparseTileMetaCore;
    current_scheme->tile_index_lookup = primary_scheme->tile_index_lookup;
    current_scheme->element_offset_lookup = primary_scheme->element_offset_lookup;
    // Tree-reduction scratch (NodeLeaf.x/.dirty, TreeLeaf.dependency) is WRITTEN
    // during the reduction chol, so it MUST be per-call: aliasing the primary's
    // trees races when calls in a multi-call group factor concurrently (nested
    // parallelism) and corrupts the factor nondeterministically. Deep-copy the
    // topology; only the mutable scratch is freshly allocated. `red_tree_separator_level`
    // was copied above, so it is available here.
    current_scheme->trees =
        (primary_scheme->trees && primary_scheme->red_tree_separator_level > 0)
            ? cloneRedTrees(primary_scheme->trees, primary_scheme->red_tree_separator_level, group_index)
            : primary_scheme->trees;

    // Copy shared within-tile indexing arrays (CRITICAL for tile element access)
    current_scheme->withinTileRow = primary_scheme->withinTileRow;
    current_scheme->withinTileCol = primary_scheme->withinTileCol;
    current_scheme->diagonal_bmapper = primary_scheme->diagonal_bmapper;

    // Copy shared symbolic and workspace arrays
    current_scheme->symbolicTileBitmaskCore = primary_scheme->symbolicTileBitmaskCore;
    current_scheme->workspace_bit_array = primary_scheme->workspace_bit_array;
    current_scheme->on_off_tiles = primary_scheme->on_off_tiles;

    // Alias the CSC symbolic factor from primary onto every secondary. The pattern
    // (L_colptr/L_rowind/nnz_factor) is structural and identical across the
    // group, so we share the primary's buffers — they're owned by
    // OrderingMemoryManager and freed in bulk at shutdown, not per-scheme,
    // so aliasing is safe. Without this, sTiles_packing(group, call_index>=1)
    // silently no-ops because _pack_prep_once guards on these three fields.
    // L_values and L_src remain per-call (each secondary runs its own chol on
    // its own tile data) and are built lazily inside _pack_prep_once.
    current_scheme->L_colptr   = primary_scheme->L_colptr;
    current_scheme->L_rowind   = primary_scheme->L_rowind;
    current_scheme->nnz_factor = primary_scheme->nnz_factor;

    // Copy shared e_trick arrays
    current_scheme->e_trick = primary_scheme->e_trick;
    current_scheme->e_trick_size = primary_scheme->e_trick_size;
    current_scheme->e_trick_inv = primary_scheme->e_trick_inv;
    current_scheme->e_trick_copy_ind = primary_scheme->e_trick_copy_ind;
    current_scheme->e_trick_size_inv = primary_scheme->e_trick_size_inv;

    if (call_info->num_cores == primary_call->num_cores) {
        current_scheme->chol_tasks = primary_scheme->chol_tasks;
        current_scheme->chol_task_offsets = primary_scheme->chol_task_offsets;
        current_scheme->inv_tasks = primary_scheme->inv_tasks;
        current_scheme->inv_task_offsets = primary_scheme->inv_task_offsets;
        current_scheme->inv_gather_packed = primary_scheme->inv_gather_packed;
        current_scheme->inv_gather_index = primary_scheme->inv_gather_index;
        current_scheme->chol_scatter_packed = primary_scheme->chol_scatter_packed;
        current_scheme->chol_scatter_index = primary_scheme->chol_scatter_index;
        current_scheme->solve_fwd_tasks = primary_scheme->solve_fwd_tasks;
        current_scheme->solve_fwd_offsets = primary_scheme->solve_fwd_offsets;
        current_scheme->solve_bwd_tasks = primary_scheme->solve_bwd_tasks;
        current_scheme->solve_bwd_offsets = primary_scheme->solve_bwd_offsets;
        // The semisparse multi-RHS tasked solve reads the update-counter arrays
        // (solve_fwd/bwd_expected). They are structural (same graph -> same
        // counts across the group), so alias them from the primary too. Without
        // this, secondary calls have empty `expected` and the tasked kernel
        // falls back to stiles_pdtrsm_*_semisparse — which is correct for
        // col-major B but WRONG for row-major B (prefer_row_layout matrices,
        // nnz/row >= 6), so every call_index >= 1 returned a bad multi-RHS solve.
        current_scheme->solve_fwd_expected = primary_scheme->solve_fwd_expected;
        current_scheme->solve_bwd_expected = primary_scheme->solve_bwd_expected;
        current_scheme->rescale_schedule = primary_scheme->rescale_schedule;
    } else {
        current_scheme->chol_tasks.reset();
        current_scheme->chol_task_offsets.reset();
        current_scheme->inv_tasks.reset();
        current_scheme->inv_task_offsets.reset();
        current_scheme->inv_gather_packed.reset();
        current_scheme->inv_gather_index.reset();
        current_scheme->chol_scatter_packed.reset();
        current_scheme->chol_scatter_index.reset();

        if (sTiles::preprocess::collect_tasks(&call_info, &stiles_schemes[global_index], group_index, eff_cores, num_threads_level1, rescale_cores) != sTiles::StatusCode::Success) {
            return EXIT_FAILURE;
        }
        // Tasks were rebuilt under a different num_cores layout, so the
        // per-task scatter info must be rebuilt too. Same decoupling as the
        // primary build: inverse gather follows params[7]; chol scatter is
        // required for any semisparse mode.
        if (stiles_control_params[sTiles::param::InverseStorageMode] == 1 && current_scheme->compute_inverse)
            sTiles::preprocess::build_inv_gather_info(current_scheme);
        if (stiles_control_params[sTiles::param::InverseStorageMode] == 1 ||
            stiles_control_params[sTiles::param::TileTypeMode] == 1 || stiles_control_params[sTiles::param::TileTypeMode] == 2) {
            sTiles::preprocess::build_chol_scatter_info(current_scheme);
        }
    }

    current_scheme->tileIndexMapper2 = primary_scheme->tileIndexMapper2;
    current_scheme->nnz_tile_counter = nullptr;
    current_scheme->solve_trick_type0 = primary_scheme->solve_trick_type0;
    current_scheme->solve_trick_type1 = primary_scheme->solve_trick_type1;

    // Per-call tile buffers are nullptr until the variant-specific allocator below.
    current_scheme->dense_tiles = nullptr;
    current_scheme->inverse_tiles = nullptr;
    current_scheme->saved_tiles = nullptr;
    current_scheme->rhs_tiles = nullptr;
    current_scheme->denseTiles = nullptr;
    current_scheme->rhTiles = nullptr;
    current_scheme->savedTiles = nullptr;
    current_scheme->inverseTiles = nullptr;
    current_scheme->chunkedDenseTiles = nullptr;
    current_scheme->chunkedRhsTiles = nullptr;
    current_scheme->chunkedSavedTiles = nullptr;
    current_scheme->chunkedInverseTiles = nullptr;

    // Allocate per-call workspaces
    {
        const int world = std::max(rescale_total_cores, call_info->num_cores);
        const int workspace_dim = (current_scheme->tile_size > 0) ? current_scheme->tile_size : 1;
        std::size_t slots = static_cast<std::size_t>(world);
        current_scheme->workspaces = static_cast<sTiles::Workspace**>(std::calloc(slots, sizeof(sTiles::Workspace*)));
        if (current_scheme->workspaces) {
            current_scheme->num_workspaces = world;
            for (int r = 0; r < world; ++r) {
                current_scheme->workspaces[r] = new sTiles::Workspace(group_index, workspace_dim);
            }
        } else {
            current_scheme->num_workspaces = 0;
        }
    }

    // Copy shared configuration flags
    current_scheme->internal_version = primary_scheme->internal_version;
    current_scheme->copy_on_preprocess = primary_scheme->copy_on_preprocess;

    // Per-call tile buffer allocation, dispatched by variant + tile_type_mode.
    int* params = sTiles_get_params();
    const int tile_type_mode = params[sTiles::param::TileTypeMode];

    if (call_info->factorization_variant == 0 || call_info->factorization_variant == 3) {
        // Sparse variants: allocate based on tile_type_mode
        if (tile_type_mode == 0) {
            if (sTiles::preprocess::allocate_dense_buffers_from_primary(primary_scheme, current_scheme, group_index, call_info->num_cores) != sTiles::StatusCode::Success)
                return EXIT_FAILURE;
        } else if (tile_type_mode == 1) {
            if (sTiles::preprocess::allocate_semisparse_buffers_from_primary(primary_scheme, current_scheme, group_index, call_info->num_cores) != sTiles::StatusCode::Success)
                return EXIT_FAILURE;
        } else if (tile_type_mode == 2) {
            // Supernodal sparse module: each secondary gets its own per-call handle
            // built from the same user permutation as the primary.
            void* h = nullptr;
            if (sTiles::sparse::api::create(&h, call_info->num_cores) != 0) {
                sTiles::Logger::error("sTiles::sparse::api::create failed (secondary call).");
                return EXIT_FAILURE;
            }
            sTiles::sparse::api::set_group_id(&h, group_index);
            {
                const int snode_cap = adaptive_snode_cap(call_info->tile_size, call_info->num_cores, primary_scheme->dim);
                sTiles::sparse::api::set_max_supernode(&h, snode_cap);
            }
            if (sTiles::sparse::api::set_user_permutation(&h, primary_scheme->element_perm, primary_scheme->dim) != 0) {
                sTiles::sparse::api::freeGroup(&h);
                return EXIT_FAILURE;
            }
            if (sTiles::sparse::api::assign_graph(&h, primary_scheme->dim, primary_scheme->nnz,
                                            call_info->row_indices, call_info->col_indices) != 0) {
                sTiles::sparse::api::freeGroup(&h);
                return EXIT_FAILURE;
            }
            current_scheme->sparse_handle = h;
        }
    } else {
        // Dense variants (1 and 2): always use dense tiles
        if (sTiles::preprocess::allocate_dense_buffers_from_primary(primary_scheme, current_scheme, group_index, call_info->num_cores) != sTiles::StatusCode::Success)
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// Allocates per-call tile buffers based on factorization variant. Called AFTER
// the GPU dispatch block (so use_gpu is decided) and AFTER both pre-GPU paths.
static int preprocess_primary_alloc_tiles(sTiles_call* call_info,
                                         TiledMatrix* scheme,
                                         int group_index,
                                         int eff_cores) {
    const double t0 = omp_get_wtime();
    if (call_info->factorization_variant == 0 || call_info->factorization_variant == 3) {
        // Supernodal sparse module owns its own storage — skip tile allocation.
        if (scheme->sparse_handle != nullptr) {
            return EXIT_SUCCESS;
        }
        if (sTiles::preprocess::allocate_dense_tiles(scheme, group_index, eff_cores) != sTiles::StatusCode::Success) {
            sTiles::Logger::error("Error: Dense tiles allocation failed (sparse variant).");
            return EXIT_FAILURE;
        }
        sTiles::Logger::timing("│   ↪ Allocate dense tiles (sparse): ", (omp_get_wtime() - t0), " s");
    } else if (call_info->factorization_variant == 1) {
        if (sTiles::preprocess::allocate_dense_tiles_variant1(scheme, group_index, eff_cores) != sTiles::StatusCode::Success) {
            sTiles::Logger::error("Error: Full dense tile allocation failed (variant 1).");
            return EXIT_FAILURE;
        }
        sTiles::Logger::timing("│   ↪ Allocate full dense tile: ", (omp_get_wtime() - t0), " s");
    } else if (call_info->factorization_variant == 2) {
        if (sTiles::preprocess::allocate_dense_tiles_variant2(scheme, group_index, eff_cores) != sTiles::StatusCode::Success) {
            sTiles::Logger::error("Error: Scaled dense tiles allocation failed (variant 2).");
            return EXIT_FAILURE;
        }
        sTiles::Logger::timing("│   ↪ Allocate scaled dense tiles: ", (omp_get_wtime() - t0), " s");
    }
    return EXIT_SUCCESS;
}

//groups
int sTiles_preprocess_group(int group_index, sTiles_object **obj, TiledMatrix **stiles_schemes) {

    //section A
    if (obj == NULL || *obj == NULL) {
        sTiles::Logger::errorf("obj is not properly initialized.");
        return EXIT_FAILURE;
    }

    // Re-resolution: preprocessing writes RESOLVED values (auto tile size,
    // auto tile-type routing, small-matrix tile-ordering disable) into the
    // live control-parameter slots below. Restore those slots from the
    // user-config shadow first so THIS group resolves from what the user
    // configured, not from whatever the previous group resolved to.
    {
        const int kResolvableSlots[] = { 1 /*tile size*/, 3 /*tile type*/,
                                         4 /*tile ordering*/, 5 /*tile-ordering size*/ };
        for (int slot : kResolvableSlots)
            stiles_control_params[slot] = stiles_user_params[slot];
        if (stiles_control_params[sTiles::param::UserTileSize] < 0) {
            // Hardware-derived, stable within a process: detect (and log) once.
            static const int auto_ts = sTiles::get::auto_tile_size();
            stiles_control_params[sTiles::param::UserTileSize] = auto_ts;
        }
        if (stiles_control_params[sTiles::param::TileOrderingThreshold] < 0)
            stiles_control_params[sTiles::param::TileOrderingThreshold] = stiles_control_params[sTiles::param::UserTileSize] / 2;
    }

    sTiles_object *s = *obj;

    int num_calls = s->stiles_groups[group_index].num_calls;
    int save_global_index = s->stiles_groups[group_index].stiles_calls[0].global_index;
    int num_threads_level1 = s->num_cores_per_group[group_index];

    int rescale_cores = 0;
    int rescale_total_cores = 0;
    if (g_rescale_num_groups > 0) {
        for(int g = 0; g < static_cast<int>(g_rescale_per_group.size()); g++){
            if(g_rescale_per_group[g] == 1){
                int num_calls_g = s->stiles_groups[g].num_calls;
                int num_cores_g = s->num_cores_per_group[g];
                rescale_cores = num_cores_g / num_calls_g;  // Cores per call
                rescale_total_cores = num_cores_g;           // Total across all call slots
                break;
            }
        }
    }

    int eff_cores = 1;

    if (num_calls <= 0) {
        sTiles::Logger::errorf("No calls to preprocess in the group.");
        return EXIT_FAILURE;
    }

    
    // Primary call (call_index 0)
    {

        //section B
        sTiles_call *call_info = &s->stiles_groups[group_index].stiles_calls[0];
        int global_index = call_info->global_index;

        TiledMatrix *scheme = NULL;
        if (sTiles_preprocess_initialization(&call_info, &scheme, 0, group_index) != 0) {
            sTiles::Logger::errorf("Failed to preprocess parameters for call %d.", 0);
            return EXIT_FAILURE;
        }

        // Check dense_variant AFTER sTiles_preprocess_initialization, which may
        // set factorization_variant=1 for small matrices (N < tile_size)
        const bool dense_variant = (call_info->factorization_variant == 1 || call_info->factorization_variant == 2);

        stiles_schemes[global_index] = scheme;
        scheme->call_lookup_table = s->call_matrix;
        eff_cores = sTiles::Utils::effective_threads(scheme->dimTiledMatrix);

        // Variant-specific symbolic + lookup work (pre-GPU dispatch)
        if (dense_variant) {
            if (preprocess_primary_dense(scheme, call_info, group_index, rescale_total_cores) != EXIT_SUCCESS) {
                scheme->preprocess_failed = true;
                return EXIT_FAILURE;
            }
        } else {
            const bool force_nd = (stiles_control_params[sTiles::param::ForceNDOrdering] == 1);
            if (preprocess_primary_sparse(call_info, scheme, stiles_schemes, global_index,
                                         group_index, eff_cores, num_threads_level1,
                                         rescale_cores, rescale_total_cores, force_nd) != EXIT_SUCCESS) {
                scheme->preprocess_failed = true;
                return EXIT_FAILURE;
            }
        }

        // Perm is ready (element_perm) but the numeric arena is not yet committed:
        // fire the symbolic-ready callback for group 0 so a caller can ship the
        // ordering now and overlap the remaining allocation.
        if (g_symbolic_done_cb && group_index == 0 && scheme->element_perm) {
            g_symbolic_done_cb(group_index, 0, scheme->element_perm, scheme->dim);
        }

        if (g_symbolic_only) {
            // symbolic-only: the primary call's element_perm is populated; skip
            // the memory estimate, GPU dispatch, tile allocation, and ALL
            // secondary calls -- everything that commits numeric storage. This
            // covers dense/semisparse (arena in preprocess_primary_alloc_tiles)
            // as well as sparse (which already returned early above).
            return EXIT_SUCCESS;
        }

        //section C
        // Calculate exact memory requirements before allocation
        {
            // Get tile_type_mode for accurate GPU memory calculation
            int* params = sTiles_get_params();
            const int tile_type_mode = params[sTiles::param::TileTypeMode];

            // Param 12 (MemoryEstimateMode) gates BOTH the computation and the
            // printing. Exception: when the GPU path is enabled (param 11 != 0),
            // the estimate is FORCED on — the per-device planner below sizes GPU
            // memory from mem_est.gpu_total_gb, so it cannot be skipped there.
            bool want_mem_est = (stiles_control_params[sTiles::param::MemoryEstimateMode] != 0);
            #ifdef STILES_GPU
            if (stiles_control_params[sTiles::param::GpuEnable] != 0) want_mem_est = true;
            #endif

            sTiles::MemoryEstimate mem_est{};
            if (want_mem_est) {
                mem_est = sTiles::memory::calculate_memory_exact(
                    stiles_schemes[global_index], call_info, tile_type_mode);
            }

            // Only print memory breakdown if explicitly requested
            if (stiles_control_params[sTiles::param::MemoryEstimateMode]) {
                sTiles::memory::print_memory_estimate(mem_est);

                // Warn if memory seems high (> 32 GB)
                if (mem_est.total_gb > 32.0) {
                    sTiles::Logger::warning("│   ⚠ High memory usage expected: ",
                        std::fixed, std::setprecision(1), mem_est.total_gb, " GB");
                }
            }

            // GPU availability and memory check (variant 0, tile_type 0 or 1)
            #ifdef STILES_GPU
            // Check if GPU is globally enabled via sTiles_use_gpu()
            if (stiles_control_params[sTiles::param::GpuEnable] == 0) {
                // GPU disabled by user
                stiles_schemes[global_index]->use_gpu = false;
                stiles_schemes[global_index]->GPU_ID = -1;
                stiles_schemes[global_index]->compute_inverse_on_gpu = false;
                sTiles::Logger::info("│   ✗ CPU MODE: GPU disabled by user (sTiles_use_gpu(0))");
            } else {
            int* gpu_params = sTiles_get_params();
            const int gpu_tile_type_mode = gpu_params[sTiles::param::TileTypeMode];  // 0=dense, 1=semisparse
            sTiles::Logger::info("│   GPU Early Check: variant=", call_info->factorization_variant,
                                 ", tile_type_mode=", gpu_tile_type_mode);

            // GPU supported for: variant 0 with dense tiles (0) OR semisparse tiles (1, 3)
            const bool gpu_supported = (call_info->factorization_variant == 0) &&
                                       (gpu_tile_type_mode == 0 || gpu_tile_type_mode == 1 || gpu_tile_type_mode == 2);

            constexpr double GPU_SAFETY_MARGIN = 0.15;   // keep 15% per-device free

            if (gpu_supported && gpu_tile_type_mode == 0) {
                // Dense tile mode (tile_type 0) — multi-GPU dispatch
                sTiles::gpu::printGpuInfo(mem_est.gpu_total_gb, num_calls, call_info->compute_inverse);

                const size_t bytes_per_call = static_cast<size_t>(
                    mem_est.gpu_total_gb * 1024.0 * 1024.0 * 1024.0);
                gpu_plan_and_apply_to_scheme(stiles_schemes[global_index],
                                             num_calls, bytes_per_call,
                                             GPU_SAFETY_MARGIN,
                                             call_info->compute_inverse,
                                             "dense");
            } else if (gpu_supported && (gpu_tile_type_mode == 1 || gpu_tile_type_mode == 2)) {
                // Semisparse tile mode (tile_type 1 or 3)
                // Calculate compact memory needed (only sa columns per tile)
                double compact_gb = 0.0;
                if (stiles_schemes[global_index]->semisparseTileMetaCore) {
                    size_t total_elements = 0;
                    const int num_tiles = stiles_schemes[global_index]->numActiveTiles;
                    const int tile_size = stiles_schemes[global_index]->tile_size;
                    for (int t = 0; t < num_tiles; ++t) {
                        int sa = stiles_schemes[global_index]->semisparseTileMetaCore[t].sa;
                        int height = tile_size;
                        if (stiles_schemes[global_index]->tileMetaCore) {
                            height = stiles_schemes[global_index]->tileMetaCore[t].height;
                            if (height <= 0) height = tile_size;
                        }
                        total_elements += static_cast<size_t>(sa) * height;
                    }
                    compact_gb = (total_elements * sizeof(double)) / (1024.0 * 1024.0 * 1024.0);
                    if (call_info->compute_inverse) compact_gb *= 2.0;  // Factor + inverse
                }

                sTiles::Logger::info("│   GPU Early Check (semisparse): compact_memory=",
                                     std::fixed, std::setprecision(2), compact_gb, " GB per call");
                sTiles::gpu::printGpuInfo(compact_gb, num_calls, call_info->compute_inverse);

                const size_t bytes_per_call = static_cast<size_t>(
                    compact_gb * 1024.0 * 1024.0 * 1024.0);
                gpu_plan_and_apply_to_scheme(stiles_schemes[global_index],
                                             num_calls, bytes_per_call,
                                             GPU_SAFETY_MARGIN,
                                             call_info->compute_inverse,
                                             "semisparse");
            } else {
                // GPU not used: either non-sparse variant (1, 2) or unsupported tile type
                stiles_schemes[global_index]->use_gpu = false;
                stiles_schemes[global_index]->GPU_ID = -1;
                stiles_schemes[global_index]->compute_inverse_on_gpu = false;
                sTiles::Logger::info("│   ✗ CPU MODE: GPU disabled (variant=", call_info->factorization_variant,
                                     ", tile_type=", gpu_tile_type_mode, ")");
            }
            }  // end else (GPU enabled)
            #else
            // No GPU support compiled in
            stiles_schemes[global_index]->use_gpu = false;
            stiles_schemes[global_index]->GPU_ID = -1;
            stiles_schemes[global_index]->compute_inverse_on_gpu = false;
            #endif
        }

        // Variant-specific tile allocation
        if (preprocess_primary_alloc_tiles(call_info, stiles_schemes[global_index],
                                          group_index, eff_cores) != EXIT_SUCCESS) {
            return EXIT_FAILURE;
        }

    }

    // Remaining calls (call_index >= 1)
    bool an_error_occurred = false;

    //section D
    if (num_calls > 1) {
        const int outer_iterations = num_calls - 1;
        const int threads_for_outer = std::min(eff_cores, outer_iterations);

        #pragma omp parallel for schedule(static,1) if(threads_for_outer > 1) num_threads(threads_for_outer)
        for (int call_index = 1; call_index < num_calls; ++call_index) {

            #pragma omp flush(an_error_occurred)
            if (an_error_occurred) continue;

            if (preprocess_secondary_call(call_index, group_index, obj, stiles_schemes,
                                          save_global_index, eff_cores, num_threads_level1,
                                          rescale_cores, rescale_total_cores) != EXIT_SUCCESS) {
                #pragma omp critical
                { an_error_occurred = true; }
            }
        } // for call_index

        if (an_error_occurred) return EXIT_FAILURE;
    }

    // GPU tile allocation for all calls (after CPU tiles are allocated)
    #ifdef STILES_GPU
    {
        TiledMatrix* primary_scheme = stiles_schemes[(*obj)->call_matrix[group_index][0]];
        int* params = sTiles_get_params();
        const int tile_type_mode = params[sTiles::param::TileTypeMode];

        // Allocate GPU tiles if: use_gpu=true, variant 0, tile_type 0 (dense)
        if (primary_scheme->use_gpu &&
            primary_scheme->factorization_variant == 0 &&
            tile_type_mode == 0) {

            const double t_gpu = omp_get_wtime();
            const sTiles::gpu::DispatchPlan& plan = primary_scheme->gpu_dispatch_plan;

            // Per-call (gpu_id, slot) assignment from the plan. Secondary calls
            // inherit use_gpu / compute_inverse_on_gpu from primary but may land
            // on a different device when case B fires.
            std::vector<TiledMatrix*> group_schemes(num_calls);
            for (int c = 0; c < num_calls; ++c) {
                const int gi = (*obj)->call_matrix[group_index][c];
                group_schemes[c] = stiles_schemes[gi];
                group_schemes[c]->use_gpu               = true;
                group_schemes[c]->GPU_ID                = plan.gpu_id[c];
                group_schemes[c]->compute_inverse_on_gpu = primary_scheme->compute_inverse_on_gpu;
            }

            const bool gpu_alloc_ok = gpu_allocate_per_device(
                plan, group_schemes, group_index,
                &sTiles::gpu::GpuTileManager::allocate_for_group);

            if (!gpu_alloc_ok) {
                sTiles::Logger::error("│   ✗ GPU allocation failed - falling back to CPU for all calls");
                gpu_disable_group(group_schemes);
            } else {
                sTiles::Logger::timing("│   ↪ GPU tile allocation (",
                    static_cast<int>(plan.devices_used.size()), " device",
                    plan.devices_used.size()==1?"":"s",
                    "): ", (omp_get_wtime() - t_gpu), " s");

                // Persistent GPU contexts: one per call, bound to its assigned device.
                const double t_ctx = omp_get_wtime();
                for (int c = 0; c < num_calls; ++c) {
                    auto* persistent = new sTiles::gpu::GpuPersistentContext();
                    sTiles::gpu::init_gpu_persistent_context(group_schemes[c], *persistent);
                    group_schemes[c]->gpu_persistent_ctx = persistent;
                }
                sTiles::Logger::timing("│   ↪ GPU persistent context init: ", (omp_get_wtime() - t_ctx), " s");
            }
        }
        // Allocate GPU tiles for semisparse mode (tile_type 1 or 3)
        else if (primary_scheme->use_gpu &&
                 primary_scheme->factorization_variant == 0 &&
                 (tile_type_mode == 1 || tile_type_mode == 2)) {

            const double t_gpu = omp_get_wtime();

            // Calculate max_sa for context initialization
            int max_sa = 0;
            for (int t = 0; t < primary_scheme->numActiveTiles; ++t) {
                if (primary_scheme->semisparseTileMetaCore &&
                    primary_scheme->semisparseTileMetaCore[t].sa > max_sa) {
                    max_sa = primary_scheme->semisparseTileMetaCore[t].sa;
                }
            }

            const sTiles::gpu::DispatchPlan& plan_ss = primary_scheme->gpu_dispatch_plan;

            std::vector<TiledMatrix*> group_schemes(num_calls);
            for (int c = 0; c < num_calls; ++c) {
                const int gi = (*obj)->call_matrix[group_index][c];
                group_schemes[c] = stiles_schemes[gi];
                group_schemes[c]->use_gpu               = true;
                group_schemes[c]->GPU_ID                = plan_ss.gpu_id[c];
                group_schemes[c]->compute_inverse_on_gpu = primary_scheme->compute_inverse_on_gpu;
            }

            const bool gpu_alloc_ok = gpu_allocate_per_device(
                plan_ss, group_schemes, group_index,
                &sTiles::gpu::GpuTileManager::allocate_for_group_semisparse);

            if (!gpu_alloc_ok) {
                sTiles::Logger::error("│   ✗ GPU semisparse allocation failed - falling back to CPU for all calls");
                gpu_disable_group(group_schemes);
            } else {
                sTiles::Logger::timing("│   ↪ GPU semisparse tile allocation (",
                    static_cast<int>(plan_ss.devices_used.size()), " device",
                    plan_ss.devices_used.size()==1?"":"s",
                    "): ", (omp_get_wtime() - t_gpu), " s");

                // Persistent contexts: one per call, bound to its device.
                const double t_ctx = omp_get_wtime();
                for (int c = 0; c < num_calls; ++c) {
                    auto* persistent = new sTiles::gpu::GpuPersistentContext();
                    sTiles::gpu::init_gpu_persistent_context_semisparse(
                        group_schemes[c], *persistent, max_sa);
                    group_schemes[c]->gpu_persistent_ctx = persistent;
                }
                sTiles::Logger::timing("│   ↪ GPU semisparse context init: ", (omp_get_wtime() - t_ctx), " s");
            }
        }
    }
    #endif

    // Snapshot the RESOLVED tile mode onto every scheme in this group.
    // Auto mode (3) was resolved to 0/1/2 during symbolic preprocessing (the
    // resolution is written into the live slot [3] for the preprocess-time
    // readers); compute-phase code must use the per-scheme copy so groups
    // with different resolutions can coexist and run concurrently.
    for (int c = 0; c < num_calls; ++c) {
        const int gi = s->call_matrix[group_index][c];
        if (gi >= 0 && stiles_schemes[gi])
            stiles_schemes[gi]->tile_type_mode = stiles_control_params[sTiles::param::TileTypeMode];
    }

    return EXIT_SUCCESS;
}

//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//___________________________________________________________________________UPDATE_X_VALUES____________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________

int sTiles_assign_values(int group_index, int call_index, void **obj, double *x, bool nested) {

    if (!obj || !(*obj)) {
        sTiles::Logger::error("[✘] Error: obj is null..");
        return -1;
    }

    sTiles_object *s = (sTiles_object *)(*obj);
    int global_index = s->schemes[0]->call_lookup_table[group_index][call_index];

    // ── Sparse-module dispatch (tile_type_mode == 2) ─────────────────────────
    // When sparse_handle is set, hand A's numeric values straight to the
    // sTiles::sparse module (so its CellStore has values for chol/solve/selinv).
    {
        TiledMatrix* sp_scheme = s->schemes[global_index];
        if (sp_scheme && sp_scheme->sparse_handle) {
            return sTiles::sparse::api::assign_values(&sp_scheme->sparse_handle, x);
        }
    }

#ifdef EXPORT_MATRIX
    if (!getenv("STILES_NO_EXPORT")) {
        sTiles_call* call = &s->stiles_groups[group_index].stiles_calls[call_index];
        TiledMatrix* export_scheme = s->schemes[global_index];
        const int pad = export_scheme ? export_scheme->nd_padding : 0;
        int N   = call->order - pad;
        // Export only the user-facing portion of the COO (drops pad-diagonal tail).
        int NNZ = call->nnz - pad;

        // Create folder name with matrix size and tile mode (only overwrites same size+mode)
        const int tile_mode = stiles_control_params[sTiles::param::TileTypeMode];
        char export_dir[256];
        snprintf(export_dir, sizeof(export_dir), "stiles_exported_N%d_mode%d", N, tile_mode);
        #ifdef _WIN32
            _mkdir(export_dir);
        #else
            mkdir(export_dir, 0755);
        #endif

        char path[512];

        snprintf(path, sizeof(path), "%s/exported_meta.txt", export_dir);
        FILE* f_meta = fopen(path, "w");
        if (f_meta) {
            fprintf(f_meta, "%d\n%d\n", N, NNZ);
            fclose(f_meta);
        }

        snprintf(path, sizeof(path), "%s/exported_row_indices.txt", export_dir);
        FILE* f_row = fopen(path, "w");
        if (f_row) {
            for (int i = 0; i < NNZ; i++) fprintf(f_row, "%d\n", call->row_indices[i]);
            fclose(f_row);
        }

        snprintf(path, sizeof(path), "%s/exported_col_indices.txt", export_dir);
        FILE* f_col = fopen(path, "w");
        if (f_col) {
            for (int i = 0; i < NNZ; i++) fprintf(f_col, "%d\n", call->col_indices[i]);
            fclose(f_col);
        }

        snprintf(path, sizeof(path), "%s/exported_values.txt", export_dir);
        FILE* f_val = fopen(path, "w");
        if (f_val) {
            for (int i = 0; i < NNZ; i++) fprintf(f_val, "%.17g\n", x[i]);
            fclose(f_val);
        }

        TiledMatrix* scheme = s->schemes[global_index];
        bool perm_exported = false;
        if (scheme && scheme->element_perm) {
            snprintf(path, sizeof(path), "%s/exported_perm.txt", export_dir);
            FILE* f_perm = fopen(path, "w");
            if (f_perm) {
                for (int i = 0; i < N; i++) fprintf(f_perm, "%d\n", scheme->element_perm[i]);
                fclose(f_perm);
                perm_exported = true;
            }
        }

        printf("EXPORT_MATRIX: exported N=%d, NNZ=%d (%d files) to %s/\n", N, NNZ, perm_exported ? 5 : 4, export_dir);
    }
#endif

    // Determine the actual variant for this specific scheme
    // Small matrices may be converted to variant 1 during preprocessing
    TiledMatrix* scheme = s->schemes[global_index];
    int variant = s->factorization_type_per_group[group_index];

    // Check if this scheme was converted to variant 1 (single tile)
    // Must have: single tile, single dimension, AND element_offset_lookup (set by build_dense_tile_lookup_variant1)
    // For single-tile matrices, always use variant 1 regardless of tile_type_mode since semisparse
    // doesn't make sense for a single tile - the entire matrix fits in one dense tile.
    bool is_single_tile = (scheme && scheme->numActiveTiles == 1 && scheme->dimTiledMatrix == 1);
    if (is_single_tile && scheme->element_offset_lookup) {
        variant = 1;
    }
    // Check for variant 2 (triangular tiles - all upper triangle tiles active)
    // Must have: all triangular tiles active AND more than 1 tile
    // IMPORTANT: Skip variant 2 for semisparse mode (tile_type_mode=1 or 3) - semisparse needs variant 0
    // DISABLED: pdtrtri_variant2 has a bug - use variant 0 instead which works correctly
    // else if (scheme && scheme->numActiveTiles == scheme->triangular_size && scheme->triangular_size > 1
    //          && stiles_control_params[3] == 0) {  // Only for dense tiles mode
    //     variant = 2;
    // }

    // Store the determined variant in the scheme for later use by solve/inverse functions
    if (scheme) {
        scheme->factorization_variant = variant;
    }

    if (variant == 1) {
        // Variant 1: single tile covering entire matrix (optimized)
        sTiles::preprocess::update_x_dense_tiles_variant1(global_index, s->schemes, x, nested);
    } else if (variant == 2) {
        // Variant 2: triangular tiles
        sTiles::preprocess::update_x_dense_tiles_variant2(global_index, s->schemes, x, nested);
    } else if (variant == 0 || variant == 3) {
        // Sparse variants
        // Populate semisparse/chunked tiles and mirror the values into dense tiles so the
        // MKL kernels never see an all-zero block from an otherwise SPD matrix.
        if (stiles_control_params[sTiles::param::TileTypeMode] == 0) {
            sTiles::preprocess::update_x_dense_tiles(global_index, s->schemes, x, nested);
        } else if (stiles_control_params[sTiles::param::TileTypeMode] == 1) {
            sTiles::preprocess::update_x_semisparse_tiles(global_index, s->schemes, x, nested);
        } else if (stiles_control_params[sTiles::param::TileTypeMode] == 2) {
            sTiles::preprocess::update_x_semisparse_tiles(global_index, s->schemes, x, nested);
            sTiles::preprocess::update_x_dense_tiles(global_index, s->schemes, x, nested);
        }

    } else {
        sTiles::Logger::errorf("Unsupported factorization variant %d in group %d", variant, group_index);
        std::abort();
    }

    return 0;
}

int sTiles_assign_values_ones(int group_index, int call_index, void **obj, bool nested) {

    if (!obj || !(*obj)) {
        sTiles::Logger::error("[✘] Error: obj is null..");
        return -1;
    }

    sTiles_object *s = (sTiles_object *)(*obj);
    int global_index = s->schemes[0]->call_lookup_table[group_index][call_index];

    // Determine the actual variant for this specific scheme
    // Small matrices may be converted to variant 1 during preprocessing
    TiledMatrix* scheme = s->schemes[global_index];
    int variant = s->factorization_type_per_group[group_index];

    // Check if this scheme was converted to variant 1 (single tile)
    // Must have: single tile, single dimension, AND element_offset_lookup (set by build_dense_tile_lookup_variant1)
    // For single-tile matrices, always use variant 1 regardless of tile_type_mode since semisparse
    // doesn't make sense for a single tile - the entire matrix fits in one dense tile.
    bool is_single_tile = (scheme && scheme->numActiveTiles == 1 && scheme->dimTiledMatrix == 1);
    if (is_single_tile && scheme->element_offset_lookup) {
        variant = 1;
    }
    // Check for variant 2 (triangular tiles - all upper triangle tiles active)
    // Must have: all triangular tiles active AND more than 1 tile
    // IMPORTANT: Skip variant 2 for semisparse mode (tile_type_mode=1 or 3) - semisparse needs variant 0
    // DISABLED: pdtrtri_variant2 has a bug - use variant 0 instead which works correctly
    // else if (scheme && scheme->numActiveTiles == scheme->triangular_size && scheme->triangular_size > 1
    //          && stiles_control_params[3] == 0) {  // Only for dense tiles mode
    //     variant = 2;
    // }

    // Store the determined variant in the scheme for later use by solve/inverse functions
    if (scheme) {
        scheme->factorization_variant = variant;
    }


    if (variant == 0 || variant == 3) {

        if (stiles_control_params[sTiles::param::TileTypeMode] == 0) {
            sTiles::preprocess::update_x_dense_tiles_ones(global_index, s->schemes, nested);
        } else if (stiles_control_params[sTiles::param::TileTypeMode] == 1) {
            sTiles::preprocess::update_x_semisparse_tiles_ones(global_index, s->schemes, nested);
        } else if (stiles_control_params[sTiles::param::TileTypeMode] == 2) {
            sTiles::preprocess::update_x_semisparse_tiles_ones(global_index, s->schemes, nested);
            sTiles::preprocess::update_x_dense_tiles_ones(global_index, s->schemes, nested);
        }

    } else {
        sTiles::Logger::errorf("Unsupported factorization variant %d in group %d", variant, group_index);
        std::abort();
    }


    return 0;
}

// Backward-compatible overload (no nested argument)
int sTiles_assign_values(int group_index, int call_index, void **obj, double *x) {
    return sTiles_assign_values(group_index, call_index, obj, x, false);
}


//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//___________________________________________________________________________GET/SET_VALUE__________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________

double sTiles_get_selinv_elm(int group_index, int call_index, int irow, int icol, void **obj) {

    if (!obj || !(*obj)) {
        sTiles::Logger::error("sTiles_get_selinv_elm: obj is null.");
        return 0.0;
    }

    sTiles_object *s = (sTiles_object *)(*obj);

    int global_index = s->schemes[0]->call_lookup_table[group_index][call_index];
    sTiles_call *call_info = &s->stiles_groups[group_index].stiles_calls[call_index];

    TiledMatrix* S = s->schemes[global_index];

    // Sparse-module dispatch — when sparse_handle is set, route element queries
    // to sTiles::sparse instead of the tile-storage accessors below.
    if (S && S->sparse_handle) {
        return sTiles::sparse::api::get_selinv_elm(&S->sparse_handle, irow, icol);
    }

    if (!S) {
        sTiles::Logger::error("sTiles_get_selinv_elm: scheme is null.");
        return 0.0;
    }

    // Check tile type mode for semisparse
    const int tile_type_mode = stiles_control_params[sTiles::param::TileTypeMode];
    const bool use_semisparse = (tile_type_mode == 1 || tile_type_mode == 2);
    // For selinv, check chunkedInverseTiles which stores the inverse (not chunkedDenseTiles which stores L)
    const bool has_semisparse_inv = (S->chunkedInverseTiles != nullptr && S->semisparseTileMetaCore != nullptr && S->tileMetaCore != nullptr);

    // Variants 1 (full dense) and 2 (scaled dense) always store the inverse in
    // S->inverseTiles regardless of tile_type_mode. Skip the semisparse path for
    // them — has_semisparse_inv is false anyway, but the user might pin
    // tile_type_mode=1 alongside variant=1, in which case we'd otherwise drop
    // through to the warning instead of using the dense accessor below.
    const bool use_dense_variant = (S->factorization_variant == 1 ||
                                    S->factorization_variant == 2);

    // Handle semisparse mode: inverse values are stored in chunkedInverseTiles after inversion
    // Mode 3 allocates both dense and semisparse inverse tiles, but the semisparse ones
    // are the authoritative output of pdtrtri_semi_dense_inv — so read those.
    if (use_semisparse && has_semisparse_inv && !use_dense_variant && (!S->inverseTiles || tile_type_mode == 2)) {
        // Normalize to upper triangle (row <= col) for symmetric matrix
        int row = irow;
        int col = icol;
        if (row > col) std::swap(row, col);

        // Apply element permutation if ordering is active
        if (S->use_ordering == 1 && S->element_perm) {
            if (S->element_perm[col] < S->element_perm[row]) {
                int tmp = S->element_perm[row];
                row = S->element_perm[col];
                col = tmp;
            } else {
                row = S->element_perm[row];
                col = S->element_perm[col];
            }
        }

        const int ts = S->tile_size;
        const int N = S->dim;
        const int num_tiles = S->dimTiledMatrix;
        const int tileRow = row / ts;
        const int tileCol = col / ts;

        // Calculate tile height (last tile might be smaller)
        auto tile_dim = [N, ts, num_tiles](int t) -> int {
            return (t == num_tiles - 1) ? (N - t * ts) : ts;
        };

        int tile_idx = -1;
        int withinRow = 0;
        int withinCol = 0;

        // Ensure we're in upper triangle for tile lookup
        int tr = tileRow, tc = tileCol;
        if (tr > tc) std::swap(tr, tc);

        // Use the same tile index lookup as pdtrtri/compute routines
        // Priority: 1) mapper.map_ij, 2) tileIndexMapper, 3) direct formula
        if (S->mapper.valid()) {
            tile_idx = S->mapper.map_ij(tr, tc, num_tiles);
        } else if (S->tileIndexMapper) {
            const int tri = tr * (2 * num_tiles - tr - 1) / 2 + tc;
            tile_idx = S->tileIndexMapper[tri];
        } else {
            // Fallback: direct triangular index (all tiles present)
            tile_idx = tr * num_tiles - (tr * (tr - 1)) / 2 + (tc - tr);
        }

        if (tileRow <= tileCol) {
            withinRow = row % ts;
            withinCol = col % ts;
        } else {
            // Transpose position within tile for lower triangle access
            withinRow = col % ts;
            withinCol = row % ts;
        }

        if (tile_idx < 0 || tile_idx >= S->numActiveTiles) {
            return 0.0;
        }

        // Read from chunkedInverseTiles (where semisparse inverse is stored)
        double* tile_data = S->chunkedInverseTiles[tile_idx];
        if (!tile_data) {
            return 0.0;
        }

        const sTiles::TileMetaCore& meta = S->tileMetaCore[tile_idx];
        const int tile_height = (meta.height > 0) ? meta.height : ts;
        const int tile_width = (meta.width > 0) ? meta.width : ts;

        if (withinRow < 0 || withinRow >= tile_height ||
            withinCol < 0 || withinCol >= tile_width) {
            return 0.0;
        }

        // Check inverse storage mode: params[7]
        //   0 = dense: inverse tiles are full h×w (default)
        //   1 = semisparse: diagonal dense, off-diagonal active-columns
        const int inverse_storage_mode = stiles_control_params[sTiles::param::InverseStorageMode];

        if (inverse_storage_mode == 1 && S->semisparseTileMetaCore) {
            // Semisparse inverse storage mode:
            // - Diagonal tiles are dense, off-diagonal use active-columns
            const sTiles::SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[tile_idx];
            const bool is_diag = (meta.row == meta.col) || (S->diagonal_bmapper && S->diagonal_bmapper[tile_idx]);

            if (is_diag) {
                // Diagonal inverse tile is DENSE (h × w)
                const std::size_t offset = static_cast<std::size_t>(withinRow)
                                         + static_cast<std::size_t>(withinCol)
                                         * static_cast<std::size_t>(tile_height);
                return tile_data[offset];
            } else {
                // Off-diagonal tile: active-columns format h × sa
                const int sa = semi.sa;
                if (sa <= 0 || semi.acol.empty()) {
                    return 0.0;  // No active columns
                }
                // Check if withinCol is an active column
                if (withinCol >= static_cast<int>(semi.acol.size())) {
                    return 0.0;
                }
                const int active_col = semi.acol[withinCol];
                if (active_col < 0 || active_col >= sa) {
                    return 0.0;  // This column is not active
                }
                const std::size_t offset = static_cast<std::size_t>(withinRow)
                                         + static_cast<std::size_t>(active_col)
                                         * static_cast<std::size_t>(tile_height);
                return tile_data[offset];
            }
        } else {
            // Dense inverse storage mode (default, mode 0)
            // Dense column-major format: element at inv[row + col * height]
            const std::size_t offset = static_cast<std::size_t>(withinRow)
                                     + static_cast<std::size_t>(withinCol)
                                     * static_cast<std::size_t>(tile_height);
            return tile_data[offset];
        }
    }

    // Prefer fast-mode accessor when TileIndexer mapper is available
    if (S && S->mapper.valid() && S->inverseTiles) {
        return sTiles_get_selinv_elm_fast_dense_wrapper(global_index, irow, icol, s->schemes);
    }

    // Direct accessor for dense mode when inverseTiles exists (small matrices,
    // single tile, etc.). Also taken by variants 1/2 even if the user pinned
    // tile_type_mode=1 — those variants always store the inverse densely.
    if ((use_dense_variant || !use_semisparse) && S->inverseTiles) {
        int row = irow;
        int col = icol;
        if (row > col) std::swap(row, col);  // Upper triangle

        // Apply element permutation if ordering is active
        if (S->use_ordering == 1 && S->element_perm) {
            if (S->element_perm[col] < S->element_perm[row]) {
                int tmp = S->element_perm[row];
                row = S->element_perm[col];
                col = tmp;
            } else {
                row = S->element_perm[row];
                col = S->element_perm[col];
            }
        }

        const int ts = S->tile_size;
        const int num_tiles = S->dimTiledMatrix;
        const int tileRow = row / ts;
        const int tileCol = col / ts;

        // Determine tile index based on available mapper
        int tile_idx = -1;
        if (num_tiles == 1) {
            tile_idx = 0;  // Single tile, direct access
        } else if (S->tileIndexMapper) {
            // Variant 0 (sparse): use tileIndexMapper for tile lookup
            // tileIndexMapper uses upper triangular packed index
            const int tri = tileRow * (2 * num_tiles - tileRow - 1) / 2 + tileCol;
            tile_idx = S->tileIndexMapper[tri];
        } else if (S->factorization_variant == 2 || S->factorization_variant == 1) {
            // Variant 1/2: Upper triangular packed index (matches build_dense_tile_lookup_variant2)
            // Formula: i * num_tiles - (i * (i - 1)) / 2 + (j - i)
            tile_idx = tileRow * num_tiles - (tileRow * (tileRow - 1)) / 2 + (tileCol - tileRow);
        } else {
            // Fallback for other cases: try the upper triangular formula
            tile_idx = tileRow * num_tiles - (tileRow * (tileRow - 1)) / 2 + (tileCol - tileRow);
        }

        if (tile_idx >= 0 && tile_idx < S->numActiveTiles && S->inverseTiles[tile_idx]) {
            const int withinRow = row % ts;
            const int withinCol = col % ts;
            // Get tile dimensions from tileMetaCore
            const int tile_height = S->tileMetaCore ? S->tileMetaCore[tile_idx].height : ts;
            double* inv_tile = S->inverseTiles[tile_idx];
            return inv_tile[withinRow + withinCol * tile_height];
        }
    }

    // No valid accessor available
    static bool warned_selinv = false;
    if (!warned_selinv) {
        sTiles::Logger::warning("sTiles_get_selinv_elm: no valid accessor available for this mode");
        warned_selinv = true;
    }
    return 0.0;
}

double* sTiles_get_selinv_row(int group_index, int call_index, int node, int* node_neighbors, int size, void** obj) {

    if (!obj || !(*obj)) {
        sTiles::Logger::error("sTiles_get_selinv_row: obj is null.");
        return NULL;
    }

    double* values = (double*)malloc(size * sizeof(double));
    if (!values) {
        sTiles::Logger::error("sTiles_get_selinv_row: memory allocation failed.");
        return NULL;
    }

    for (int i = 0; i < size; i++) {
        values[i] = ::sTiles_get_selinv_elm(group_index, call_index, node, node_neighbors[i], obj);
    }

    return values;
}

double sTiles_get_chol_elm(int group_index, int call_index, int irow, int icol, void **obj) {

    if (!obj || !(*obj)) {
        sTiles::Logger::error("sTiles_get_chol_elm: obj is null.");
        return 0.0;
    }

    sTiles_object *s = (sTiles_object *)(*obj);

    int global_index = s->schemes[0]->call_lookup_table[group_index][call_index];
    TiledMatrix* S = s->schemes[global_index];

    if (!S) {
        sTiles::Logger::error("sTiles_get_chol_elm: scheme is null.");
        return 0.0;
    }

    // Sparse-module dispatch — when sparse_handle is set, route element queries
    // to sTiles::sparse instead of the tile-storage accessors below.
    if (S->sparse_handle) {
        return sTiles::sparse::api::get_chol_elm(&S->sparse_handle, irow, icol);
    }

    // Check tile type mode
    const int tile_type_mode = stiles_control_params[sTiles::param::TileTypeMode];
    const bool use_semisparse = (tile_type_mode == 1 || tile_type_mode == 2);

    // Determine which tile storage to use
    const bool has_dense = (S->denseTiles != nullptr);
    const bool has_semisparse = (S->chunkedDenseTiles != nullptr && S->semisparseTileMetaCore != nullptr && S->tileMetaCore != nullptr);

    if (!has_dense && !has_semisparse) {
        static bool warned_tiles = false;
        if (!warned_tiles) { sTiles::Logger::info("sTiles_get_chol_elm: no tile storage available"); warned_tiles = true; }
        return 0.0;
    }

    // Normalize to upper triangle (row <= col)
    int row = irow;
    int col = icol;
    if (row > col) std::swap(row, col);

    // Apply element permutation if ordering is active
    if (S->use_ordering == 1 && S->element_perm) {
        if (S->element_perm[col] < S->element_perm[row]) {
            int tmp = S->element_perm[row];
            row = S->element_perm[col];
            col = tmp;
        } else {
            row = S->element_perm[row];
            col = S->element_perm[col];
        }
    }

    const int ts = S->tile_size;
    const int N = S->dim;
    const int num_tiles = S->dimTiledMatrix;
    const int tileRow = row / ts;
    const int tileCol = col / ts;

    // Calculate tile height (last tile might be smaller)
    auto tile_dim = [N, ts, num_tiles](int t) -> int {
        return (t == num_tiles - 1) ? (N - t * ts) : ts;
    };

    // Calculate tile index using mapper if available
    int tile_idx = -1;
    int withinRow = 0;
    int withinCol = 0;

    // Ensure we're in upper triangle for tile lookup
    int tr = tileRow, tc = tileCol;
    if (tr > tc) std::swap(tr, tc);

    // Use the same tile index lookup as pdtrtri/compute routines
    // Priority: 1) mapper.map_ij, 2) tileIndexMapper, 3) direct formula
    if (S->mapper.valid()) {
        tile_idx = S->mapper.map_ij(tr, tc, num_tiles);
    } else if (S->tileIndexMapper) {
        const int tri = tr * (2 * num_tiles - tr - 1) / 2 + tc;
        tile_idx = S->tileIndexMapper[tri];
    } else {
        // Fallback: direct triangular index (all tiles present)
        tile_idx = tr * num_tiles - (tr * (tr - 1)) / 2 + (tc - tr);
    }

    if (tileRow <= tileCol) {
        withinRow = row % ts;
        withinCol = col % ts;
    } else {
        // Transpose position within tile for lower triangle access
        withinRow = col % ts;
        withinCol = row % ts;
    }

    // Try semisparse storage first if available and in semisparse mode
    if (use_semisparse && has_semisparse) {
        // Bounds check for semisparse
        if (tile_idx < 0 || tile_idx >= S->numActiveTiles) {
            return 0.0;
        }
        double* tile_data = S->chunkedDenseTiles[tile_idx];
        if (tile_data) {
            const sTiles::TileMetaCore& meta = S->tileMetaCore[tile_idx];
            const sTiles::SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[tile_idx];
            const int tile_height = (meta.height > 0) ? meta.height : ts;
            const int tile_width = (meta.width > 0) ? meta.width : ts;
            const bool is_diagonal = (tileRow == tileCol);
            const bool uses_lapack_diagonal = (S->diagonal_bmapper && S->diagonal_bmapper[tile_idx]);

            if (is_diagonal) {
                int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
                if (diag_cols <= 0) diag_cols = tile_width;
                const int band = withinCol - withinRow;
                if (band < 0 || band >= diag_cols) return 0.0;  // Outside bandwidth

                if (uses_lapack_diagonal) {
                    // LAPACK upper banded format: element (i,j) stored at row (kd+1-1-(j-i)), col j
                    const int lapack_row = diag_cols - 1 - band;
                    const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                             + static_cast<std::size_t>(diag_cols)
                                             * static_cast<std::size_t>(withinCol);
                    return tile_data[offset];
                } else {
                    // Non-LAPACK diagonal format: band-by-band column-major storage
                    // offset = local_row + band * tile_height
                    const std::size_t offset = static_cast<std::size_t>(withinRow)
                                             + static_cast<std::size_t>(band)
                                             * static_cast<std::size_t>(tile_height);
                    return tile_data[offset];
                }
            } else {
                // Off-diagonal tile: column-compressed format
                if (withinCol >= 0 && withinCol < static_cast<int>(semi.acol.size())) {
                    const int compressed_col = semi.acol[withinCol];
                    if (compressed_col >= 0) {
                        const std::size_t offset = static_cast<std::size_t>(withinRow)
                                                 + static_cast<std::size_t>(compressed_col)
                                                 * static_cast<std::size_t>(tile_height);
                        return tile_data[offset];
                    }
                }
                return 0.0;  // Column not active
            }
        }
    }

    // Fall back to dense storage
    if (has_dense) {
        // Bounds check for dense tiles
        if (tile_idx < 0 || tile_idx >= S->numActiveTiles) {
            return 0.0;  // Tile not active (zero in sparse matrix)
        }
        double* tile_data = S->denseTiles[tile_idx];
        if (!tile_data) {
            static bool warned_elm = false;
            if (!warned_elm) { sTiles::Logger::info("sTiles_get_chol_elm: denseTiles[%d] is null", tile_idx); warned_elm = true; }
            return 0.0;
        }

        // Get tile dimensions (column-major storage)
        const int tile_height = tile_dim(tileRow <= tileCol ? tileRow : tileCol);
        const int off = withinRow + (withinCol * tile_height);
        return tile_data[off];
    }

    return 0.0;
}

int* sTiles_return_perm_vec(int group_index, void **obj) {

    if (!obj || !(*obj)) {
        sTiles::Logger::error("sTiles_return_perm_vec: obj is null.");
        return NULL;
    }

    sTiles_object *s = (sTiles_object *)(*obj);
    int global_index = s->stiles_groups[group_index].stiles_calls[0].global_index;

    if (s->schemes[global_index]->use_ordering > 0) {
        return s->schemes[global_index]->element_perm;
    } else {
        int N = s->schemes[global_index]->dim;
        int *default_perm = MemoryManager::allocate<int>(N, group_index);
        for (int i = 0; i < N; ++i) {
            default_perm[i] = i;
        }
        return default_perm;
    }
}

int* sTiles_return_iperm_vec(int group_index, void **obj) {

    if (!obj || !(*obj)) {
        sTiles::Logger::error("sTiles_return_iperm_vec: obj is null.");
        return NULL;
    }

    sTiles_object *s = (sTiles_object *)(*obj);
    int global_index = s->stiles_groups[group_index].stiles_calls[0].global_index;

    if (s->schemes[global_index]->use_ordering > 0) {
        return s->schemes[global_index]->element_iperm;
    } else {
        int N = s->schemes[global_index]->dim;
        int *default_iperm = MemoryManager::allocate<int>(N, group_index);
        for (int i = 0; i < N; ++i) {
            default_iperm[i] = i;
        }
        return default_iperm;
    }
}

int sTiles_get_num_calls(void** obj, int group_index) {

    if (!obj || !(*obj)) {
        sTiles::Logger::error("sTiles_return_iperm_vec: obj is null.");
        return -1;
    }

    sTiles_object *s = (sTiles_object *)(*obj);
    return s->stiles_groups[group_index].num_calls;
}

long long sTiles_get_nnz_factor(int group_index, int call_index, void** obj) {
    if (!obj || !(*obj)) return -1LL;
    sTiles_object *s = (sTiles_object *)(*obj);
    const int global_index = s->stiles_groups[group_index].stiles_calls[call_index].global_index;
    TiledMatrix* scheme = s->schemes[global_index];
    return scheme ? scheme->nnz_factor : -1LL;
}

// Read the flat CSC L_values buffer for one call. Returns nullptr if the
// scheme isn't allocated or pack hasn't run yet. Intended for tests/diagnostics
// (e.g. cross-kernel correctness checks); production code should go through
// sTiles_dtrsm rather than touching the buffer directly.
double* sTiles_get_L_values(int group_index, int call_index, void** obj) {
    if (!obj || !(*obj)) return nullptr;
    sTiles_object *s = (sTiles_object *)(*obj);
    const int global_index = s->stiles_groups[group_index].stiles_calls[call_index].global_index;
    TiledMatrix* scheme = s->schemes[global_index];
    return scheme ? scheme->L_values : nullptr;
}

// Runtime tuning of the L_src precomputed-pointer-table memory ceiling.
// Set 0 to force the per-entry fallback kernel (useful for A/B and tests).
// Set larger to allow bigger tables on memory-rich machines. Default 2 GiB.
void sTiles_set_pack_cache_threshold_bytes(long long bytes) {
    sTiles::set_pack_cache_threshold_bytes(bytes);
}
long long sTiles_get_pack_cache_threshold_bytes(void) {
    return sTiles::get_pack_cache_threshold_bytes();
}

const int* sTiles_get_element_perm(int group_index, int call_index, void** obj) {
    if (!obj || !(*obj)) return nullptr;
    sTiles_object *s = (sTiles_object *)(*obj);
    const int global_index = s->stiles_groups[group_index].stiles_calls[call_index].global_index;
    TiledMatrix* scheme = s->schemes[global_index];
    return scheme ? scheme->element_perm : nullptr;
}

int sTiles_get_element_perm_length(int group_index, int call_index, void** obj) {
    if (!obj || !(*obj)) return -1;
    sTiles_object *s = (sTiles_object *)(*obj);
    const int global_index = s->stiles_groups[group_index].stiles_calls[call_index].global_index;
    TiledMatrix* scheme = s->schemes[global_index];
    return (scheme && scheme->element_perm) ? scheme->dim : -1;
}

int sTiles_get_logical_element_perm(int group_index, int call_index, void** obj, int* out_perm) {
    if (!obj || !(*obj) || !out_perm) return -1;
    sTiles_object *s = (sTiles_object *)(*obj);
    const int global_index = s->stiles_groups[group_index].stiles_calls[call_index].global_index;
    TiledMatrix* scheme = s->schemes[global_index];
    if (!scheme || !scheme->element_perm) return -1;
    // Logical (unpadded) dimension. nd_padding counts the identity pad rows ND
    // block padding appended; element_perm has scheme->dim entries where the first
    // n_logical index the ORIGINAL nodes (their positions live in [0, scheme->dim)).
    const int n_logical = scheme->dim - scheme->nd_padding;
    if (n_logical <= 0) return -1;
    // Rank-compress the original nodes' positions to [0, n_logical): sort original
    // node indices by their permuted position, then assign ranks. This removes the
    // pad-slot gaps and yields a valid permutation of the original matrix.
    std::vector<int> order(n_logical);
    for (int i = 0; i < n_logical; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return scheme->element_perm[a] < scheme->element_perm[b]; });
    for (int rank = 0; rank < n_logical; ++rank) out_perm[order[rank]] = rank;
    return n_logical;
}

const int* sTiles_get_partition_sizes(int group_index, void** obj) {
    if (!obj || !*obj) {
        sTiles::Logger::warning("sTiles_get_partition_sizes: invalid obj pointer");
        return nullptr;
    }

    sTiles_object *s = (sTiles_object *)(*obj);

    if (group_index < 0 || group_index >= s->num_call_groups) {
        sTiles::Logger::warning("sTiles_get_partition_sizes: invalid group_index=", group_index);
        return nullptr;
    }

    int global_index = s->stiles_groups[group_index].stiles_calls[0].global_index;
    TiledMatrix* scheme = s->schemes[global_index];

    if (!scheme || !scheme->partition_sizes) {
        return nullptr;
    }

    return scheme->partition_sizes;
}

int sTiles_clear_selinv(int group_index, int call_index, void **obj) {

    if (!obj || !(*obj)) {
        sTiles::Logger::error("sTiles_clear_selinv: obj is null.");
        return -1;
    }

    sTiles_object *s = (sTiles_object *)(*obj);
    const int global_index = s->stiles_groups[group_index].stiles_calls[call_index].global_index;
    TiledMatrix *scheme = s->schemes[global_index];

    if (!scheme->compute_inverse) {
        return 0;
    }

    // ── Sparse-module path (tile_type_mode == 2) ─────────────────────────
    // Z_cs is allocated lazily by selinv; zero its arena and drop the
    // selinved flag so the next get_selinv_elm forces a fresh selinv pass.
    if (scheme->sparse_handle) {
        return sTiles::sparse::api::clear_selinv(&scheme->sparse_handle);
    }

    const int variant = scheme->factorization_variant;

    // Variant 1: full dense, single tile
    if (variant == 1) {
        double* inv = scheme->inverseTiles ? scheme->inverseTiles[0] : nullptr;
        if (inv) {
            const int N = scheme->dim;
            LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', N, N, 0.0, 1.0, inv, N);
        }
        return 0;
    }

    // Variant 2: scaled dense, upper-triangular tiles
    if (variant == 2) {
        if (!scheme->inverseTiles) return 0;

        const int N_tiles  = scheme->dimTiledMatrix;
        const int ts       = scheme->tile_size;
        const int mat_size = scheme->dim;

        int dense_idx = 0;
        for (int i = 0; i < N_tiles; ++i) {
            for (int j = i; j < N_tiles; ++j) {
                if (dense_idx >= scheme->numActiveTiles) break;

                double* inv = scheme->inverseTiles[dense_idx];
                if (inv) {
                    const int h = (i == N_tiles - 1) ? (mat_size - i * ts) : ts;
                    const int w = (j == N_tiles - 1) ? (mat_size - j * ts) : ts;

                    if (i == j) {
                        LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 1.0, inv, h);
                    } else {
                        LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, inv, h);
                    }
                }
                dense_idx++;
            }
        }
        return 0;
    }

    // Variant 0: tile path (mode 0 = dense tiles, mode 1 = semisparse).
    // Mode 2 (supernodal sparse module) is handled above via sparse_handle.
    int* params = sTiles_get_params();
    const int tile_type_mode = params[sTiles::param::TileTypeMode];
    const int inverse_storage_mode = params[sTiles::param::InverseStorageMode];

    // Clear dense inverseTiles (mode 0)
    if (tile_type_mode == 0 && scheme->inverseTiles && scheme->tileMetaCore) {
        const int num_active = scheme->numActiveTiles;
        const bool *diag_map = scheme->diagonal_bmapper;

        for (int t = 0; t < num_active; ++t) {
            double* inv = scheme->inverseTiles[t];
            if (!inv) continue;

            const sTiles::TileMetaCore& meta = scheme->tileMetaCore[t];
            const int h = (meta.height > 0) ? meta.height : scheme->tile_size;
            const int w = (meta.width  > 0) ? meta.width  : scheme->tile_size;

            const bool is_diag = (diag_map && diag_map[t]) || (meta.row == meta.col);

            if (is_diag) {
                LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 1.0, inv, h);
            } else {
                LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, inv, h);
            }
        }
    }

    // Clear chunked (semisparse) inverseTiles (mode 1)
    if (tile_type_mode == 1 && scheme->chunkedInverseTiles) {
        const int num_active = scheme->numActiveTiles;
        const bool *diag_map = scheme->diagonal_bmapper;
        const bool semisparse_offdiag = (inverse_storage_mode == 1);

        for (int t = 0; t < num_active; ++t) {
            double* inv = scheme->chunkedInverseTiles[t];
            if (!inv) continue;

            const sTiles::TileMetaCore& meta = scheme->tileMetaCore[t];
            const int h = (meta.height > 0) ? meta.height : scheme->tile_size;
            const int w = (meta.width  > 0) ? meta.width  : scheme->tile_size;

            const bool is_diag = (diag_map && diag_map[t]) || (meta.row == meta.col);

            if (is_diag) {
                // Diagonal tiles always use dense h × w format
                LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 1.0, inv, h);
            } else if (semisparse_offdiag && scheme->semisparseTileMetaCore) {
                // Off-diagonal in semisparse inverse mode: h × sa
                const int sa = scheme->semisparseTileMetaCore[t].sa;
                if (sa > 0) {
                    const std::size_t elems = static_cast<std::size_t>(h) * static_cast<std::size_t>(sa);
                    std::fill(inv, inv + elems, 0.0);
                }
            } else {
                // Off-diagonal in dense inverse mode: h × w
                LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, inv, h);
            }
        }
    }

    return 0;
}

//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//___________________________________________________________________________ASSIGNING GRAPH____________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________

int sTiles_assign_graph(int group_index, void** stile, int N, int NNZ, int* row_indices, int* col_indices)
{   
    // Cast the opaque pointer back to its real type
    if (!stile || !(*stile)) {
        sTiles::Logger::error("sTiles_return_iperm_vec: obj is null.");
        return -1;
    }

    sTiles_object *s = (sTiles_object *)(*stile);

    // Assign matrix dimensions
    for (int call_index = 0; call_index < s->num_calls_per_group[group_index]; call_index++) {
        s->stiles_groups[group_index].stiles_calls[call_index].order = N;
        s->stiles_groups[group_index].stiles_calls[call_index].nnz = NNZ;
    }

    // Instead of allocating new memory, simply assign the provided pointers
    s->stiles_groups[group_index].stiles_calls[0].row_indices = row_indices;
    s->stiles_groups[group_index].stiles_calls[0].col_indices = col_indices;
    s->stiles_groups[group_index].same_group = true;

    return 0;
}

int sTiles_assign_graph_one_call(int group_index, int call_index, void** stile, int N, int NNZ, int* row_indices, int* col_indices)
{
    // Cast the opaque pointer back to its real type
    if (!stile || !(*stile)) {
        sTiles::Logger::error("sTiles_return_iperm_vec: obj is null.");
        return -1;
    }

    sTiles_object *s = (sTiles_object *)(*stile);

    sTiles_call* call = &s->stiles_groups[group_index].stiles_calls[call_index];

    call->order   = N;
    call->nnz = NNZ;

    // Free existing pointers if needed
    if (call->row_indices) {
        free(call->row_indices);
        call->row_indices = NULL;
    }

    if (call->col_indices) {
        free(call->col_indices);
        call->col_indices = NULL;
    }

    // Assign new pointers
    call->row_indices = row_indices;
    call->col_indices = col_indices;

    return 0;
}

//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//___________________________________________________________________________INIT Call ________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________
//______________________________________________________________________________________________________________________________________________________________________________________________________

int sTiles_init(void **obj) {

    if (!obj || !(*obj)) {
        sTiles::Logger::error("[✘] Error: obj is null..");
        return -1;
    }

    sTiles::Logger::info("╭────────────────── sTiles: Preprocessing ───────────────────╮");

    sTiles_object *s = (sTiles_object *)(*obj);
    if (s->total_calls <= 0) {
        sTiles::Logger::error("Invalid total_calls value: must be > 0.");
        return -1;
    }

    if (!s->schemes) {
        sTiles::Logger::error("Schemes array not allocated. Did sTiles_create() fail?");
        return -1;
    }
    
    sTiles::Logger::info("│  - Using pre-allocated scheme array for " + std::to_string(s->total_calls) + " total calls");

    int total_groups = s->num_call_groups;
    sTiles::Binding::Setup_all_teams(obj);
    for (int group_index = 0; group_index < total_groups; group_index++) {
        // Skip groups that are rescale targets (g_rescale_per_group[i] > 0).
        // They will use the rescale schedule from the primary group at runtime.
        if (g_rescale_num_groups > 0
            && group_index < static_cast<int>(g_rescale_per_group.size())
            && g_rescale_per_group[group_index] > 0) {
            sTiles::Logger::info("│  - Skipping preprocessing for group " + std::to_string(group_index) + " (rescale target, " + std::to_string(g_rescale_per_group[group_index]) + " cores)");
            continue;
        }
        // Skip groups whose first call has no graph assigned
        if (s->stiles_groups[group_index].num_calls > 0
            && s->stiles_groups[group_index].stiles_calls[0].row_indices == nullptr) {
            sTiles::Logger::info("│  - Skipping preprocessing for group " + std::to_string(group_index) + " (no graph assigned)");
            continue;
        }
        sTiles_preprocess_group(group_index, (sTiles_object**)obj, s->schemes);
    }
    sTiles::Logger::info("╰────────────────────────────────────────────────────────────╯");
    if (static_cast<int>(sTiles::getLevel()) >= static_cast<int>(sTiles::Level::Info)) { printf("\n");  printf("\n"); }
    //exit(0);

    return 0;
}

// Toggle symbolic-only preprocessing (see g_symbolic_only). Persistent until
// cleared; sTiles_init_symbolic wraps it for a single call.
void sTiles_set_symbolic_only(int on) { g_symbolic_only = (on ? 1 : 0); }
int  sTiles_get_symbolic_only(void)   { return g_symbolic_only; }

// Register (or clear with NULL) the symbolic-ready callback (see g_symbolic_done_cb).
void sTiles_set_symbolic_done_callback(void (*cb)(int, int, const int *, int)) { g_symbolic_done_cb = cb; }

// Ordering + symbolic only: populates scheme->element_perm for every group but
// commits NO numeric factor arena. Read the perm via sTiles_return_perm_vec.
// A later sTiles_init (numeric) with the perm forced completes the factor.
int sTiles_init_symbolic(void **obj) {
    g_symbolic_only = 1;
    int rc = sTiles_init(obj);
    g_symbolic_only = 0;
    return rc;
}

int sTiles_init_group(int group_index, void **obj) {

    if (!obj || !(*obj)) {
        sTiles::Logger::error("[✘] Error: obj is null..");
        return -1;
    }

    if (group_index==0) sTiles::Logger::info("╭────────────────── sTiles: Preprocessing ───────────────────╮");
    
    sTiles_object *s = (sTiles_object *)(*obj);
    if (s->total_calls <= 0) {
        sTiles::Logger::error("Invalid total_calls value: must be > 0.");
        return -1;
    }

    if (!s->schemes) {
        sTiles::Logger::error("Schemes array not allocated. Did sTiles_create() fail?");
        return -1;
    }

    sTiles::Binding::Setup_all_teams(obj);

    // Skip groups that are rescale targets (g_rescale_per_group[i] > 0).
    // They will use the rescale schedule from the primary group at runtime.
    if (g_rescale_num_groups > 0
        && group_index < static_cast<int>(g_rescale_per_group.size())
        && g_rescale_per_group[group_index] > 0) {
        sTiles::Logger::info("│  - Skipping preprocessing for group " + std::to_string(group_index) + " (rescale target, " + std::to_string(g_rescale_per_group[group_index]) + " cores)");
    } else if (s->stiles_groups[group_index].num_calls > 0
               && s->stiles_groups[group_index].stiles_calls[0].row_indices == nullptr) {
        sTiles::Logger::info("│  - Skipping preprocessing for group " + std::to_string(group_index) + " (no graph assigned)");
    } else {
        if (sTiles_preprocess_group(group_index, (sTiles_object**)obj, s->schemes) != EXIT_SUCCESS) {
            // Preprocessing (symbolic/ordering) failed for this group. Propagate
            // the failure so the caller aborts instead of proceeding into chol
            // with an unresolved scheme (which would cascade through pdpotrf).
            // The failing scheme(s) are flagged preprocess_failed, so chol/solve/
            // selinv/logdet also refuse to run if the caller ignores this return.
            sTiles::Logger::error("sTiles_init_group: preprocessing failed for group ", group_index,
                                  "; factorization will not be attempted (see preceding error).");
            return -1;
        }
    }

    // After preprocessing, every scheme's numActiveTiles is finalized.
    // Allocate the persistent byte-progress buffers (one per scheme) so
    // the chol executor doesn't malloc/free them per call. Two separate
    // buffers — pthreads (ss_slots) and OMP (dep_tracker->slots) workers
    // run independently and may differ in cell-level state mid-chol.
    {
        const int num_calls = s->stiles_groups[group_index].num_calls;
        for (int c = 0; c < num_calls; ++c) {
            if (!s->schemes || !s->schemes[0] || !s->schemes[0]->call_lookup_table) break;
            const int gi = s->schemes[0]->call_lookup_table[group_index][c];
            if (gi < 0 || gi >= s->total_calls) continue;
            TiledMatrix* scheme = s->schemes[gi];
            if (!scheme) continue;
            const std::size_t n = (scheme->numActiveTiles > 0)
                                  ? static_cast<std::size_t>(scheme->numActiveTiles)
                                  : 0;
            if (n == 0) continue;
            if (!scheme->byte_progress_buf) {
                scheme->byte_progress_buf =
                    static_cast<unsigned char*>(std::malloc(n));
            }
            if (!scheme->byte_progress_buf_omp) {
                // std::atomic<unsigned char> is layout-compatible with
                // unsigned char on every supported platform; default-init is
                // fine — the executor stores the actual init value per call.
                scheme->byte_progress_buf_omp =
                    new std::atomic<unsigned char>[n];
            }
        }
    }

    if (group_index == s->num_call_groups - 1) {
        sTiles::Logger::info("╰────────────────────────────────────────────────────────────╯");
        if (static_cast<int>(sTiles::getLevel()) >= static_cast<int>(sTiles::Level::Info)) { printf("\n");  printf("\n"); }
    }
   // exit(0);
    return 0;
}


// ── Explicit CSC packing phase ──────────────────────────────────────────────
// Pack L (the Cholesky factor) into a flat CSC layout (scheme->L_values).
// After this returns, single-core nrhs==1 solves take the csc_dtrsm shortcut;
// multi-core / multi-RHS solves keep using the parallel tile path.
//
// API contract — symmetric with sTiles_chol:
//   sTiles_bind(group, call, &obj);
//   sTiles_chol(group, call, &obj);
//   sTiles_packing(group, call, &obj);   // ← uses chol's bound thread pool
//   sTiles_unbind(group, call, &obj);
//
// Per-call so the bind_index / context resolution is exact: the pack runs
// on the same persistent thread pool sTiles_chol was bound to, via
// parallel_call → Process::ppack. This is what makes parallel calls
// (multiple cholesky+pack+solve on different (group, call) slots) safe —
// each pack uses only its own bound team.
//
// Backend dispatch follows param[8] inside pack_L_values:
//   * param[8] == 1 → OMP parallel-for (matches chol's omp_dpotrf path)
//   * param[8] == 0 → pthreads parallel_call on the bound pool
//                     (matches chol's pthreads_dpotrf path)
//
// Returns 0 on success, -1 on null/invalid handle or unbound context,
// -2 on out-of-range index.
int sTiles_packing(int group_index, int call_index, void** obj) {
    if (!obj || !*obj) {
        sTiles::Logger::error("sTiles_packing: null sTiles object handle");
        return -1;
    }
    sTiles_object* s = static_cast<sTiles_object*>(*obj);
    if (group_index < 0 || group_index >= s->num_call_groups) {
        sTiles::Logger::error("sTiles_packing: group_index ", group_index,
                              " out of range [0, ", s->num_call_groups, ")");
        return -2;
    }
    if (call_index < 0 || call_index >= s->stiles_groups[group_index].num_calls) {
        sTiles::Logger::error("sTiles_packing: call_index ", call_index,
                              " out of range [0, ",
                              s->stiles_groups[group_index].num_calls, ")");
        return -2;
    }
    if (!s->schemes || !s->schemes[0] || !s->schemes[0]->call_lookup_table) {
        sTiles::Logger::error("sTiles_packing: schemes not initialised; call sTiles_init_group first");
        return -1;
    }

    const int gi = s->schemes[0]->call_lookup_table[group_index][call_index];
    if (gi < 0 || !s->schemes[gi]) return -1;
    TiledMatrix* scheme = s->schemes[gi];
    if (!scheme->L_colptr || !scheme->L_rowind || scheme->nnz_factor <= 0) {
        // No symbolic factor → nothing to pack. Common before sTiles_chol.
        return 0;
    }

    // Idempotent: if the buffer already mirrors the current factor, skip the
    // work. sTiles_chol clears `packed` so a fresh factor always re-packs;
    // callers wanting to force a re-pack should sTiles_unpacking first.
    if (scheme->packed) return 0;

    // Resolve the bound context for this call — same lookup chol uses.
    // If null, the user hasn't called sTiles_bind; pack_L_values's
    // standalone overload will fall back to std::thread workers.
    stiles_context_t* stile = stiles_context_self(gi);
    sTiles::pack_L_values(stile, scheme);
    // Mark fresh only if pack actually produced a buffer (it can no-op when
    // dense/semisparse tile arrays aren't set up — see _pack_prep_once).
    scheme->packed = (scheme->L_values != nullptr);
    return 0;
}

// ── Bench-only: time csc_dtrsm_multi (col-major) vs csc_dtrsm_multi_row
// (row-major) on the already-packed scheme for `repeats` iterations each.
// Bypasses the solve permute pipeline so the comparison is purely
// kernel-vs-kernel on the same L_values. Skips correctness — fills X
// with random values per iteration; the goal is timing.
//
// Returns 0 on success. Best wall-clock per iteration in seconds is
// written to *out_col_s and *out_row_s.
int sTiles_bench_csc_layouts(int group_index, int call_index, void** obj,
                             int nrhs, int repeats,
                             double* out_col_s, double* out_row_s) {
    if (out_col_s) *out_col_s = 0.0;
    if (out_row_s) *out_row_s = 0.0;
    if (!obj || !*obj || nrhs <= 0 || repeats <= 0) return -1;
    sTiles_object* s = static_cast<sTiles_object*>(*obj);
    if (group_index < 0 || group_index >= s->num_call_groups) return -2;
    if (call_index < 0 || call_index >= s->stiles_groups[group_index].num_calls) return -2;
    if (!s->schemes || !s->schemes[0] || !s->schemes[0]->call_lookup_table) return -1;
    const int gi = s->schemes[0]->call_lookup_table[group_index][call_index];
    if (gi < 0 || !s->schemes[gi]) return -1;
    TiledMatrix* scheme = s->schemes[gi];
    if (!scheme->packed || !scheme->L_values || scheme->dim <= 0) {
        sTiles::Logger::error("sTiles_bench_csc_layouts: scheme not packed; call sTiles_packing first");
        return -1;
    }

    const int N = scheme->dim;
    std::vector<double> X(static_cast<std::size_t>(N) * nrhs);
    std::vector<double> X_init(X.size());
    for (auto& v : X_init) v = 0.5 + 0.001 * (rand() & 0xFFF) / 4096.0;

    double best_col = 1e30;
    double best_row = 1e30;
    for (int r = 0; r < repeats; ++r) {
        // Column-major: ldb = N
        std::copy(X_init.begin(), X_init.end(), X.begin());
        const double t0 = omp_get_wtime();
        sTiles::csc_dtrsm_multi(scheme, X.data(), nrhs, N, /*solve_type=*/2);
        const double dt = omp_get_wtime() - t0;
        if (dt < best_col) best_col = dt;
    }
    for (int r = 0; r < repeats; ++r) {
        // Row-major: ldb_row = nrhs
        std::copy(X_init.begin(), X_init.end(), X.begin());
        const double t0 = omp_get_wtime();
        sTiles::csc_dtrsm_multi_row(scheme, X.data(), nrhs, nrhs, /*solve_type=*/2);
        const double dt = omp_get_wtime() - t0;
        if (dt < best_row) best_row = dt;
    }

    if (out_col_s) *out_col_s = best_col;
    if (out_row_s) *out_row_s = best_row;
    return 0;
}

// Free L_values for one call. Symmetric counterpart to sTiles_packing.
// After this, the solve gate misses (packed == false) and solves revert
// to the parallel tile path. Use to release memory between workload
// phases (e.g. transitioning from nrhs=1 batch to nrhs >= 2 batch).
int sTiles_unpacking(int group_index, int call_index, void** obj) {
    if (!obj || !*obj) return -1;
    sTiles_object* s = static_cast<sTiles_object*>(*obj);
    if (group_index < 0 || group_index >= s->num_call_groups) return -2;
    if (call_index < 0 || call_index >= s->stiles_groups[group_index].num_calls) return -2;
    if (!s->schemes || !s->schemes[0] || !s->schemes[0]->call_lookup_table) return -1;
    const int gi = s->schemes[0]->call_lookup_table[group_index][call_index];
    if (gi >= 0 && s->schemes[gi]) {
        if (s->schemes[gi]->L_values) {
            delete[] s->schemes[gi]->L_values;
            s->schemes[gi]->L_values = nullptr;
        }
        if (s->schemes[gi]->L_src) {
            delete[] s->schemes[gi]->L_src;
            s->schemes[gi]->L_src = nullptr;
        }
        s->schemes[gi]->packed = false;
        // Structural prep just got torn down — next sTiles_packing must
        // re-allocate L_values, rebuild L_src, and re-derive cached fields.
        s->schemes[gi]->pack_prep_done = false;
    }
    return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace sTiles {

    void sTiles_setup_groups_wrapper(sTiles_object **obj,
                                     int num_call_groups,
                                     const int *num_calls_per_group,
                                     const int *num_cores_per_group,
                                     const int *arrowhead_size_per_group,
                                     const int *factorization_type_per_group,
                                     const int *arrowhead_size,
                                     const bool *get_inverse,
                                     const int* user_params) {

    sTiles::Logger::info("│ ↳ Allocating and initializing group structures...          ");
    sTiles::Logger::info("│   • num_call_groups = " + std::to_string(num_call_groups) + std::string(30 - std::to_string(num_call_groups).size(), ' '));

    // Allocate memory for the sTiles_object
    *obj = nullptr;  // Ensure it's null before allocation
    *obj = MemoryManager::allocate<sTiles_object>(1);
    (*obj)->num_call_groups = num_call_groups;
    (*obj)->max_cores_sys = sTiles::Utils::get_max_cores();

    #ifdef NUMA_ENABLED
        (*obj)->numa_enabled = true;
    #else
        (*obj)->numa_enabled = false;
    #endif

    sTiles::Logger::debug("│   ↪ max_cores_sys = " + std::to_string((*obj)->max_cores_sys));
    sTiles::Logger::debug("│   ↪ NUMA enabled   = " + std::string((*obj)->numa_enabled ? "true" : "false"));

    // Allocate memory for num_calls_per_group array
    (*obj)->num_calls_per_group = MemoryManager::allocate<int>(num_call_groups);
    (*obj)->num_cores_per_group = MemoryManager::allocate<int>(num_call_groups);
    (*obj)->factorization_type_per_group = MemoryManager::allocate<int>(num_call_groups);
    (*obj)->stiles_groups = MemoryManager::allocate<sTiles_group>(num_call_groups);

    (*obj)->num_total_indices = 0;
    int r = 0, offset = 0;



    for (int i = 0; i < num_call_groups; i++) {

        // Assign num_calls_per_group
        (*obj)->num_calls_per_group[i] = num_calls_per_group[i];

        // Validate the number of cores per group
        if (num_cores_per_group[i] > (*obj)->max_cores_sys) {
            sTiles::Logger::warning("Group " + std::to_string(i) + ": requested " + std::to_string(num_cores_per_group[i]) + " cores exceeds system max (" + std::to_string((*obj)->max_cores_sys) + "). Using max.");
            (*obj)->num_cores_per_group[i] = (*obj)->max_cores_sys;
        }else{
            (*obj)->num_cores_per_group[i] = num_cores_per_group[i];
        }   
      
        // Assign factorization_type_per_group and increment num_total_indices
        (*obj)->factorization_type_per_group[i] = factorization_type_per_group[i];
        (*obj)->num_total_indices += (*obj)->num_calls_per_group[i];

        // Variant 1 (single dense tile): Each call needs its own context, so cores >= calls
        // This ensures each call can bind to a unique worker context
        if (factorization_type_per_group[i] == 1 &&
            (*obj)->num_cores_per_group[i] < num_calls_per_group[i]) {
            int target_cores = num_calls_per_group[i];
            int max_cores = (*obj)->max_cores_sys;

            if (target_cores > max_cores) {
                // System doesn't have enough cores for all calls to run in parallel
                sTiles::Logger::warning("Group " + std::to_string(i) +
                    ": Variant 1 needs " + std::to_string(target_cores) +
                    " cores for " + std::to_string(num_calls_per_group[i]) +
                    " calls, but system only has " + std::to_string(max_cores) +
                    ". Some calls must run sequentially.");
                (*obj)->num_cores_per_group[i] = max_cores;
            } else {
                sTiles::Logger::warning("Group " + std::to_string(i) +
                    ": Variant 1 requires cores >= calls. Adjusting cores from " +
                    std::to_string((*obj)->num_cores_per_group[i]) + " to " +
                    std::to_string(target_cores) + ".");
                (*obj)->num_cores_per_group[i] = target_cores;
            }
        }

        // Initialize sTiles_group
        (*obj)->stiles_groups[i].group_index = i;
        (*obj)->stiles_groups[i].num_calls = num_calls_per_group[i];
        (*obj)->stiles_groups[i].group_offset = offset;
        (*obj)->stiles_groups[i].same_group = false;
        (*obj)->stiles_groups[i].arrowhead_size_per_group = arrowhead_size_per_group ? arrowhead_size_per_group[i] : 0; 
        offset += num_calls_per_group[i];

        // Allocate stiles_calls array
        (*obj)->stiles_groups[i].stiles_calls = MemoryManager::allocate<sTiles_call>(num_calls_per_group[i]);

        // Initialize stiles_calls and distribute cores
        for (int j = 0; j < num_calls_per_group[i]; j++) {
            (*obj)->stiles_groups[i].stiles_calls[j].call_index = j;
            (*obj)->stiles_groups[i].stiles_calls[j].global_index = r;
            (*obj)->stiles_groups[i].stiles_calls[j].mapped_call_index = -1;
            (*obj)->stiles_groups[i].stiles_calls[j].mapped_group_index = -1;
            r++;
        }

        distribute_cores_to_calls((*obj)->num_cores_per_group[i], 
                                  num_calls_per_group[i],
                                  (*obj)->stiles_groups[i].stiles_calls, 
                                  (*obj)->max_cores_sys, 
                                  (*obj)->numa_enabled);

    }

    // Determine the maximum number of calls in any group
    int max_num_calls = 0;
    for (int i = 0; i < num_call_groups; i++) {
        if (num_calls_per_group[i] > max_num_calls) {
        max_num_calls = num_calls_per_group[i];
        }
    }

    // Allocate call_matrix (array of int pointers)
    (*obj)->call_matrix = MemoryManager::allocate<int*>(num_call_groups);

    // Allocate and initialize rows of call_matrix
    for (int i = 0; i < num_call_groups; i++) {
        (*obj)->call_matrix[i] = MemoryManager::allocate<int>(max_num_calls);
    }

    // Populate call_matrix with example values
    int counter = 0;
    for (int j = 0; j < num_call_groups; j++) {
        for (int i = 0; i < num_calls_per_group[j]; i++) {
            (*obj)->call_matrix[j][i] = counter++;  // Example initialization logic
        }
    }

    sTiles::Logger::info("│   → Total number of calls across all groups: " + std::to_string((*obj)->num_total_indices));

    //for (int j = 0; j < num_call_groups; j++) {
    //    for (int i=0; i < num_calls_per_group[j]; i++) {
    //        printf("%d-%d ---> %d \n", j, i, (*obj)->call_matrix[j][i]);
    //    } 
    //    printf("\n");
    //}

    int con = 0;
    for (int i = 0; i < num_call_groups; i++) {

        offset = 0;
        con = 0;
        for (int j = 0; j < num_calls_per_group[i]; j++) {

            // Set local_offset and increment offset
            (*obj)->stiles_groups[i].stiles_calls[j].local_offset = offset;
            offset += (*obj)->stiles_groups[i].stiles_calls[j].num_cores;

            // Set default values for STILES_CALL_INFO fields
            (*obj)->stiles_groups[i].stiles_calls[j].ordering_strategy = 0;
            (*obj)->stiles_groups[i].stiles_calls[j].compute_inverse = get_inverse[i];
            (*obj)->stiles_groups[i].stiles_calls[j].compute_log_determinant = false;
            (*obj)->stiles_groups[i].stiles_calls[j].save_factor = false;
            (*obj)->stiles_groups[i].stiles_calls[j].sequence_id = 0;

            // Initialize numerical and pointer fields to default values
            (*obj)->stiles_groups[i].stiles_calls[j].order = 0;
            (*obj)->stiles_groups[i].stiles_calls[j].nnz = 0;
            (*obj)->stiles_groups[i].stiles_calls[j].row_indices = nullptr;
            (*obj)->stiles_groups[i].stiles_calls[j].col_indices = nullptr;
            (*obj)->stiles_groups[i].stiles_calls[j].matrix_values = nullptr;
            (*obj)->stiles_groups[i].stiles_calls[j].red_tree_separator_level = 0;
            (*obj)->stiles_groups[i].stiles_calls[j].tile_size = 120;
            (*obj)->stiles_groups[i].stiles_calls[j].factorization_variant = factorization_type_per_group[i];
            (*obj)->stiles_groups[i].stiles_calls[j].use_nested_dissection = false;
            (*obj)->stiles_groups[i].stiles_calls[j].arrowhead_thickness = 0;
            (*obj)->stiles_groups[i].stiles_calls[j].preprocess_level = 0;
            (*obj)->stiles_groups[i].stiles_calls[j].bandwidth = 0;

            // Allocate and initialize parameters array
            (*obj)->stiles_groups[i].stiles_calls[j].parameters = MemoryManager::allocateZero<int>(20);

            // Set default values for parameters
            auto& params = (*obj)->stiles_groups[i].stiles_calls[j].parameters;
            params[static_cast<int>(sTiles::Parameter::TreeReductionAcc)] = 100;
            params[static_cast<int>(sTiles::Parameter::TreeReductionMaxCores)] = 32;
            params[static_cast<int>(sTiles::Parameter::TreeReductionStrategy)] = 2;
            params[static_cast<int>(sTiles::Parameter::AndMinCores)] = 3;
            params[static_cast<int>(sTiles::Parameter::AndMinMatrixSize)] = 10000;
            params[static_cast<int>(sTiles::Parameter::AndMultipleNestedDissection)] = 0;
            params[static_cast<int>(sTiles::Parameter::AndTileFillInRef)] = 15;
            params[static_cast<int>(sTiles::Parameter::AndAllowedBan)] = 800;
            params[static_cast<int>(sTiles::Parameter::CoresSplitStrategy)] = 2;
            params[static_cast<int>(sTiles::Parameter::BoostedETrick)] = 1;

            // Handle arrowhead thickness
            if (arrowhead_size && !arrowhead_size_per_group) {
                (*obj)->stiles_groups[i].stiles_calls[j].arrowhead_thickness = arrowhead_size[con++];
            } else if (arrowhead_size_per_group) {
                (*obj)->stiles_groups[i].stiles_calls[j].arrowhead_thickness = arrowhead_size_per_group[i];
            } else {
                (*obj)->stiles_groups[i].stiles_calls[j].arrowhead_thickness = 0;
            }

            const auto& call = (*obj)->stiles_groups[i].stiles_calls[j];
            sTiles::Logger::trace("│   → Call " + std::to_string(j)
                                   + ": global_index = " + std::to_string(call.global_index)
                                   + ", arrowhead = "   + std::to_string(call.arrowhead_thickness)
                                   + ", tile_size = "   + std::to_string(call.tile_size));
            sTiles::Logger::debug("│   ↪ Parameters for call " + std::to_string(call.global_index));
            sTiles::Logger::debug("│       • TREE_REDUCTION_ACC             = " + std::to_string(params[static_cast<int>(sTiles::Parameter::TreeReductionAcc)]));
            sTiles::Logger::debug("│       • TREE_REDUCTION_MAX_CORES       = " + std::to_string(params[static_cast<int>(sTiles::Parameter::TreeReductionMaxCores)]));
            sTiles::Logger::debug("│       • TREE_REDUCTION_STRATEGY        = " + std::to_string(params[static_cast<int>(sTiles::Parameter::TreeReductionStrategy)]));
            sTiles::Logger::debug("│       • AND_MIN_CORES                  = " + std::to_string(params[static_cast<int>(sTiles::Parameter::AndMinCores)]));
            sTiles::Logger::debug("│       • AND_MIN_MATRIX_SIZE            = " + std::to_string(params[static_cast<int>(sTiles::Parameter::AndMinMatrixSize)]));
            sTiles::Logger::debug("│       • AND_MULTIPLE_NESTED_DISSECTION = " + std::to_string(params[static_cast<int>(sTiles::Parameter::AndMultipleNestedDissection)]));
            sTiles::Logger::debug("│       • AND_TILE_FILL_IN_REF           = " + std::to_string(params[static_cast<int>(sTiles::Parameter::AndTileFillInRef)]));
            sTiles::Logger::debug("│       • AND_ALLOWED_BAN                = " + std::to_string(params[static_cast<int>(sTiles::Parameter::AndAllowedBan)]));
            sTiles::Logger::debug("│       • CORES_SPLIT_STRATEGY           = " + std::to_string(params[static_cast<int>(sTiles::Parameter::CoresSplitStrategy)]));
            sTiles::Logger::debug("│       • BOOSTED_E_TRICK                = " + std::to_string(params[static_cast<int>(sTiles::Parameter::BoostedETrick)]));

        }

        sTiles::Logger::info("│   ─ Group " + std::to_string(i) + " ──────────────────────────────");
        sTiles::Logger::info("│     • num_calls:        " + std::to_string(num_calls_per_group[i]));
        sTiles::Logger::info("│     • num_cores:        " + std::to_string((*obj)->num_cores_per_group[i]));
        sTiles::Logger::info("│     • factor_type:      " + std::to_string(factorization_type_per_group[i]));
        if (arrowhead_size_per_group) {
            sTiles::Logger::info("│     • arrowhead size:   " + std::to_string(arrowhead_size_per_group[i]));
        }
    }

    sTiles::Logger::info("│   [✔] Group wrapper initialization complete.               ");

}

sTiles::StatusCode sTiles_setup_groups(sTiles_object **obj,
                                       int num_call_groups,
                                       const int *num_calls_per_group,
                                       const int *num_cores_per_group,
                                       const int *factorization_type_per_group,
                                       const bool *get_inverse,
                                       const int *arrowhead_size,
                                       const int *arrowhead_size_per_group,
                                       const int *user_params) {

    sTiles::Logger::info("│ ↳ Starting sTiles group initialization...                  ");
    sTiles::Logger::debug("│   ↪ Pointer to obj: 0x" + std::to_string(reinterpret_cast<std::uintptr_t>(*obj)));

    try {
        // Initialize sTiles_object (allocates and sets up groups, calls, etc.)

        int sum_indices = 0;
        for(int i=0; i<num_call_groups;i++) sum_indices += num_calls_per_group[i];

        #ifdef STILES_USE_PARTITIONS
            int new_indices = sum_indices * (1 + STILES_USE_PARTITIONS) + num_call_groups;
            sTiles::Control::InitializeGlobal(new_indices);
        #else
            sTiles::Control::InitializeGlobal(sum_indices);
        #endif

        sTiles_setup_groups_wrapper(obj,
                                    num_call_groups,
                                    num_calls_per_group,
                                    num_cores_per_group,
                                    arrowhead_size_per_group,
                                    factorization_type_per_group,
                                    arrowhead_size,
                                    get_inverse,
                                    user_params);

        // Perform any additional setup tasks
        //STYLES((*obj)->num_total_indices);
        

        // #ifdef STILES_USE_PARTITIONS
        //     int new_indices = (*obj)->num_total_indices * (1 + STILES_USE_PARTITIONS) + num_call_groups;
        //     sTiles::Control::InitializeGlobal(new_indices);
        // #else
        //     sTiles::Control::InitializeGlobal((*obj)->num_total_indices);
        // #endif

        // Calculate total number of calls and set it in the object
        int num_total_calls = 0;
        for (int i = 0; i < num_call_groups; i++) {
            num_total_calls += num_calls_per_group[i];
        }
        (*obj)->total_calls = num_total_calls;

        sTiles::Logger::trace("│   ↪ Total number of calls to initialize: " + std::to_string(num_total_calls));

        // Allocate the schemes array of pointers directly inside the sTiles object
        (*obj)->schemes = MemoryManager::allocateZero<TiledMatrix*>(num_total_calls);
        if (!(*obj)->schemes) {
            throw std::bad_alloc();
        }

        sTiles::Logger::info("│   [✔] Group setup and memory allocation succeeded.         ");
        return sTiles::StatusCode::Success;

    } catch (const std::exception &e) {
        
        sTiles::Logger::error("Error during sTiles setup: " + std::string(e.what()));
        if (*obj) {
            MemoryManager::freeAll();
            *obj = nullptr;
        }
        return sTiles::StatusCode::Failure; 
    }
}

void configure_tile_size() {

    if(stiles_control_params[sTiles::param::UserTileSize] < 0) {

        stiles_control_params[sTiles::param::UserTileSize] = sTiles::get::auto_tile_size();

    }else{

        sTiles::Logger::info("│ ✓ User-specified tile size selected: ", stiles_control_params[sTiles::param::UserTileSize]);

    }

    if(stiles_control_params[sTiles::param::TileOrderingThreshold] < 0) {

        stiles_control_params[sTiles::param::TileOrderingThreshold] = stiles_control_params[sTiles::param::UserTileSize] / 2;

    }

}

/**
 * @brief Internal implementation for sTiles object creation.
 *
 * This function contains the shared logic for both the simple and expert
 * creation routines, preventing code duplication and ensuring consistent
*  error handling and logging.
 */
static int create_internal(void **obj,
                           int num_call_groups,
                           const int *calls_per_group,
                           const int *cores_per_group,
                           const int *factor_type_per_group,
                           const bool *get_inverse,
                           const int *arrowhead_size,
                           const int *arrowhead_size_per_group,
                           const int *user_params) {

    // 0. Verify expiry before any other operation
    sTiles::internal::verify_expiry();
    sTiles::internal::init_time_tracking();
    sTiles::internal::cache_validity_state();  // Cache for zero-overhead checks later

    // Warn once about set-but-unrecognized STILES_* environment variables
    // (a misspelled override is silently inert otherwise).
    stiles_env_validate();

    // 1. Validate input pointer
    if (!obj) {
        sTiles::Logger::error("sTiles_create received a null output pointer for the object.");
        return static_cast<int>(sTiles::StatusCode::IllegalValue);
    }
    *obj = nullptr; // Initialize output to null for safety

    sTiles::show_logo();
    if (static_cast<int>(sTiles::getLevel()) >= static_cast<int>(sTiles::Level::Info)) printf("\n");
    sTiles::Logger::info("╭──────────────────── sTiles: Initialization ────────────────╮");

    // 2. Allocate and configure the call groups (allocation happens inside setup)
    sTiles::Logger::info("│ ↳ Configuring call groups...");


    // 3. Set up the call groups using the parameters provided
    sTiles::StatusCode setup_status = sTiles_setup_groups((sTiles_object **)obj,
                                                          num_call_groups, calls_per_group, cores_per_group,
                                                          factor_type_per_group, get_inverse,
                                                          arrowhead_size, arrowhead_size_per_group, user_params);
    if (setup_status != sTiles::StatusCode::Success) {
        sTiles::Logger::error("│ ✗ sTiles_setup_groups failed with status code: ", static_cast<int>(setup_status));
        *obj = nullptr;
        sTiles::Logger::info("╰────────────────────────────────────────────────────────────╯");
        return static_cast<int>(setup_status);
    }
    sTiles::Logger::info("│ ✓ Call groups configured successfully.\n[INFO]    │ ↳ Configuring tile size...");

    // 4. Configure tile size
    sTiles::configure_tile_size();
    sTiles::Logger::info("│ ✓ Tile size configured.\n[INFO]    │ ✓ sTiles object created successfully.");
    sTiles::Logger::info("╰────────────────────────────────────────────────────────────╯");
    if (static_cast<int>(sTiles::getLevel()) >= static_cast<int>(sTiles::Level::Info)) printf("\n");

    sTiles_object *internal = static_cast<sTiles_object*>(*obj);
    if (internal) {
        g_registered_objects.push_back(internal);
    }

    return static_cast<int>(sTiles::StatusCode::Success);
}

}
/**
 * @brief Creates and initializes a standard sTiles object.
 */
int sTiles_create(void **obj,
                  int num_call_groups,
                  const int *calls_per_group,
                  const int *cores_per_group,
                  const int *factor_type_per_group,
                  const bool *get_inverse) {

    return sTiles::create_internal(obj,
                                   num_call_groups,
                                   calls_per_group,
                                   cores_per_group,
                                   factor_type_per_group,
                                   get_inverse,
                                   nullptr, // arrowhead_size
                                   nullptr, // arrowhead_size_per_group
                                   nullptr  // user_params
                                   );
}

/**
 * @brief Creates and initializes an sTiles object with expert parameters.
 */
int sTiles_create_expert(void **obj,
                         int num_call_groups,
                         const int *calls_per_group,
                         const int *cores_per_group,
                         const int *factor_type_per_group,
                         const bool *get_inverse,
                         const int *arrowhead_size,
                         const int *arrowhead_size_per_group,
                         const int *user_params) {

    return sTiles::create_internal(obj,
                                   num_call_groups,
                                   calls_per_group,
                                   cores_per_group,
                                   factor_type_per_group,
                                   get_inverse,
                                   arrowhead_size,
                                   arrowhead_size_per_group,
                                   user_params);

}


/*

user input: sTiles_create
--------------------------------------------------------------------
sTiles_create
create_internal
sTiles_setup_groups
sTiles_setup_groups_wrapper
--------------------------------------------------------------------

user input: sTiles_init
--------------------------------------------------------------------
sTiles_init/sTiles_init_group
sTiles_preprocess_group:
    sTiles_preprocess_initialization: initilize scheme
sTiles_preprocess_solve_call_using_bool (maybe update or delete)
*/
