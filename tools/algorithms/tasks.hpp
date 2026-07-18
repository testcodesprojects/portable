/**
 * @file tasks.hpp
 * @brief Task collection and distribution utilities for parallel sparse matrix operations.
 *
 * This file provides functions for collecting, distributing, and managing computational
 * tasks across multiple cores. It includes utilities for:
 * - Forward and backward solve task collection
 * - Cholesky factorization task distribution
 * - Selective inversion task scheduling
 * - Load balancing across processing cores
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

#include "../common/stiles_structs.hpp"             // sTiles_call, TiledMatrix, enums
#include "chol_algorithms.hpp"
#include "inv_algorithms.hpp"
#include "../TileIndexer/TileIndexerGraphBuilder.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <functional>
#include <vector>
#include <array>
#include <new>
#include <atomic>

// Access to global control parameters (defined in process.cpp).
//   params[9]  : semisparse implementation selector
//                  0 = original semisparse, 1 = imp1, 2 = imp2 (==imp3_serial here),
//                  3 = imp3_serial_and_sparse (sparse-aware fast path)
//   params[14] : serial mode
//                  0 = auto (use serial kernels when num_cores == 1)
//                  1 = always parallel
extern "C" int* sTiles_get_params();

namespace sTiles{ namespace preprocess{

/**
 * @brief Collect solve tasks for a single core (forward direction).
 *
 * Pre-computes (k, m) tile pairs with their indices for triangular solve.
 * Task format: [type, k, m, tile_idx, lda, temp_dim]
 *   type=1: diagonal TRSM (m==k)
 *   type=2: off-diagonal GEMM (m!=k)
 *
 * @param state TileIndexer state for adjacency checks
 * @param method Lookup method
 * @param myrank Core rank (0 to worldsize-1)
 * @param worldsize Total number of cores
 * @param num_tiles Number of tile rows/cols
 * @param tile_size Size of each tile
 * @param N Matrix dimension
 * @param map_id Function to map (i,j) to tile index
 * @param out_tasks Output vector for collected tasks
 * @return Number of tasks collected
 */
inline int solve_collect_forward(const TileIndexer::State& state,
                                  TileIndexer::Method method,
                                  int myrank,
                                  int worldsize,
                                  int num_tiles,
                                  int tile_size,
                                  int N,
                                  const std::function<int(int,int)>& map_id,
                                  std::vector<std::array<int,6>>& out_tasks)
{
    if (num_tiles <= 0 || worldsize <= 0) return 0;

    const int Adesc_lm1 = N / tile_size;
    auto BLKLDD = [&](int k) { return (k < Adesc_lm1) ? tile_size : (N % tile_size); };

    int k = 0;
    int m = myrank;

    // Find starting position for this core
    while (m >= num_tiles) {
        k++;
        m = m - num_tiles + k;
    }

    while (k < num_tiles && m < num_tiles) {
        // Compute next position
        int next_k = k;
        int next_m = m + worldsize;
        while (next_m >= num_tiles && next_k < num_tiles) {
            next_k++;
            next_m = next_m - num_tiles + next_k;
        }

        const int tempkm = (k == num_tiles - 1) ? (N - k * tile_size) : tile_size;
        const int tempmm = (m == num_tiles - 1) ? (N - m * tile_size) : tile_size;

        if (m == k) {
            // Diagonal tile: TRSM
            const int ldak = BLKLDD(k);
            const int idx = map_id(k, k);
            if (idx >= 0) {
                out_tasks.push_back({1, k, m, idx, ldak, tempkm});
            }
        } else {
            // Off-diagonal tile: GEMM (if tile exists)
            if (state.isActive(k, m, num_tiles)) {
                const int ldak = BLKLDD(k);
                const int idx = map_id(k, m);
                if (idx >= 0) {
                    out_tasks.push_back({2, k, m, idx, ldak, tempmm});
                }
            }
        }

        k = next_k;
        m = next_m;
    }

    return static_cast<int>(out_tasks.size());
}

/**
 * @brief Collect solve tasks for a single core (backward direction).
 *
 * Same as forward but with reversed tile ordering for backward substitution.
 * The actual tile indices are computed at runtime based on num_tiles.
 */
inline int solve_collect_backward(const TileIndexer::State& state,
                                   TileIndexer::Method method,
                                   int myrank,
                                   int worldsize,
                                   int num_tiles,
                                   int tile_size,
                                   int N,
                                   const std::function<int(int,int)>& map_id,
                                   std::vector<std::array<int,6>>& out_tasks)
{
    if (num_tiles <= 0 || worldsize <= 0) return 0;

    const int Adesc_lm1 = N / tile_size;
    auto BLKLDD = [&](int k) { return (k < Adesc_lm1) ? tile_size : (N % tile_size); };

    int k = 0;
    int m = myrank;

    // Find starting position for this core
    while (m >= num_tiles) {
        k++;
        m = m - num_tiles + k;
    }

    while (k < num_tiles && m < num_tiles) {
        // Compute next position
        int next_k = k;
        int next_m = m + worldsize;
        while (next_m >= num_tiles && next_k < num_tiles) {
            next_k++;
            next_m = next_m - num_tiles + next_k;
        }

        // For backward solve, actual tiles are at (num_tiles-1-k, num_tiles-1-m)
        const int actual_k = num_tiles - 1 - k;
        const int actual_m = num_tiles - 1 - m;

        const int tempkm = (k == 0) ? (N - (num_tiles - 1) * tile_size) : tile_size;
        const int tempmm = tile_size;  // For backward, m != last row in most cases

        if (m == k) {
            // Diagonal tile: TRSM
            const int ldak = BLKLDD(actual_k);
            const int idx = map_id(actual_k, actual_k);
            if (idx >= 0) {
                out_tasks.push_back({1, k, m, idx, ldak, tempkm});
            }
        } else {
            // Off-diagonal tile: GEMM (if tile exists)
            if (state.isActive(actual_m, actual_k, num_tiles)) {
                const int ldam = BLKLDD(actual_m);
                const int idx = map_id(actual_m, actual_k);
                if (idx >= 0) {
                    out_tasks.push_back({2, k, m, idx, ldam, tempkm});
                }
            }
        }

        k = next_k;
        m = next_m;
    }

    return static_cast<int>(out_tasks.size());
}

/**
 * @brief Collect GPU solve tasks for backward direction using actual coordinates.
 *
 * Unlike the CPU backward collection (which uses virtual/reversed coordinates),
 * this emits tasks in the exact order the GPU backward loop executes:
 *   for k = num_tiles-1 down to 0:
 *     off-diagonal GEMMs first (m = k+1..num_tiles-1)
 *     then diagonal TRSM
 *
 * Task format: [type, k, m, tile_idx, lda, dim] (same as CPU solve tasks)
 */
inline int gpu_solve_collect_backward(
    const TileIndexer::State& state,
    int num_tiles,
    int tile_size,
    int N,
    const std::function<int(int,int)>& map_id,
    std::vector<std::array<int,6>>& out_tasks)
{
    if (num_tiles <= 0) return 0;

    const int Adesc_lm1 = N / tile_size;
    auto BLKLDD = [&](int k) { return (k < Adesc_lm1) ? tile_size : (N % tile_size); };

    for (int k = num_tiles - 1; k >= 0; --k) {
        const int tempkm = (k < Adesc_lm1) ? tile_size : (N - k * tile_size);
        const int ldak = BLKLDD(k);

        // Off-diagonal updates first (backward: B[k] -= U[k,m] * B[m])
        for (int m = k + 1; m < num_tiles; ++m) {
            if (state.isActive(k, m, num_tiles)) {
                const int idx = map_id(k, m);
                if (idx >= 0) {
                    const int tempmm = (m < Adesc_lm1) ? tile_size : (N - m * tile_size);
                    out_tasks.push_back({2, k, m, idx, ldak, tempmm});
                }
            }
        }

        // Diagonal TRSM: B[k] = U[k,k]^{-1} * B[k]
        const int diag_idx = map_id(k, k);
        if (diag_idx >= 0) {
            out_tasks.push_back({1, k, k, diag_idx, ldak, tempkm});
        }
    }

    return static_cast<int>(out_tasks.size());
}

    inline void export_core_tasks_to_txt(const std::vector<std::array<int,7>>& tasks,
                                     int core_id,
                                     int group_index,
                                     const sTiles_call* call,
                                     const std::string& phase_label,
                                     bool canonicalize_inverse = false) {
    const std::string label = phase_label.empty() ? std::string("tasks") : phase_label;
    const int call_idx = call ? call->call_index : -1;
    const int global_idx = call ? call->global_index : -1;

    std::ostringstream oss;
    oss << "tasks_" << label;
    if (global_idx >= 0) oss << "_global" << global_idx;
    if (group_index >= 0) oss << "_group" << group_index;
    if (call_idx >= 0) oss << "_call" << call_idx;
    oss << "_core" << core_id << ".txt";

    const std::filesystem::path out_dir("tasks");
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    const std::filesystem::path out_path = out_dir / oss.str();

    std::ofstream out(out_path, std::ios::trunc);
    if (!out.is_open()) {
        return;
    }

    const std::vector<std::array<int,7>>* data_ptr = &tasks;
    std::vector<std::array<int,7>> canonical;
    if (canonicalize_inverse && !tasks.empty()) {
        canonical = tasks;
        data_ptr = &canonical;
    }

    out << "# phase=" << label
        << " coreslot=" << core_id
        << " tasks=" << data_ptr->size()
        << '\n';
    for (const auto& row : *data_ptr) {
        for (int idx = 0; idx < 7; ++idx) {
            if (idx > 0) out << ' ';
            out << row[static_cast<std::size_t>(idx)];
        }
        out << '\n';
    }
}


struct MapIdFunctor {
    TiledMatrix* scheme_local;
    int num_tiles;

    int operator()(int i, int j) const {
        int ii = i;
        int jj = j;
        if (ii > jj) std::swap(ii, jj);
        if (scheme_local->mapper.valid()) {
            const int idx = scheme_local->mapper.map_ij(ii, jj, num_tiles);
            if (idx >= 0) return idx;
        }
        if (scheme_local->tileIndexMapper) {
            const int tri = ii * (2 * num_tiles - ii - 1) / 2 + jj;
            return scheme_local->tileIndexMapper[tri];
        }
        return -1;
    }
};

struct ProcessCore {

    TileIndexer::State& state;
    TileIndexer::Method method;
    int a;            // collection-thread count (P): Region A m-stride / standard collection
    int a_exec;       // execution-rank count: Region B per-slot slicing
    int num_tiles;
    int group_index;
    sTiles_call** call_info;
    TiledMatrix* scheme_ptr;
    std::function<int(int,int)> map_id;
    bool has_red_tree;

    sTiles::StatusCode operator()(int corea, std::vector<std::array<int,7>>& chol_out, std::vector<std::array<int,7>>* inv_out) const {

        std::vector<std::array<int,7>> local_tasks;
        int produced = 0;

        // Confound-free scheduling comparison (STILES_FLAT_SCHED=1): force the
        // STANDARD global-stride expansion even when the matrix is SCOTCH-ordered
        // and padded with partitions present. This holds the matrix (ordering +
        // padding + fill + task SET) IDENTICAL while toggling only the schedule,
        // so flat-vs-ND-vs-composed timings isolate the scheduler, not the fill.
        static const bool flat_sched = [](){
            const char* e = std::getenv("STILES_FLAT_SCHED"); return e && e[0] == '1';
        }();

        if (!flat_sched && has_red_tree) {
            // ===== COMPOSED ND + TREE (Option A) =====
            // When the SCOTCH ND-partition collection is active, Region A is
            // walked PER NON-ROOT PARTITION (leaf/intermediate-sep parallelism)
            // instead of flat; Region B (the root separator corner) stays the
            // tree reduction. corner_probe set red_tree_separator_level =
            // scotch_root_sep_tiles (Step 3), so cutoff = num_tiles - sep equals
            // the root partition boundary and the non-root partitions tile
            // [0, cutoff) exactly. Without partition collection (arrowhead /
            // unpartitioned), regionA_parts stays null → original flat Region A.
            std::vector<alg::RegionAPartition> ra_parts;
            if (scheme_ptr->scotch_partition_collection) {
                if (scheme_ptr->num_partitions > 0 && !scheme_ptr->partitions.empty()) {
                    // N-way: non-root = every partition except the last (S_root).
                    const int np = scheme_ptr->num_partitions;
                    ra_parts.reserve(static_cast<std::size_t>(std::max(0, np - 1)));
                    for (int p = 0; p < np - 1; ++p) {
                        const auto& pd = scheme_ptr->partitions[static_cast<std::size_t>(p)];
                        ra_parts.push_back({pd.start_tile, pd.end_tile, pd.first_core, pd.num_cores});
                    }
                } else if (scheme_ptr->partition_sizes) {
                    // Legacy 3-way: non-root = P1, P2 (exclude Sep = root). Mirror
                    // the ND-branch core split (cores 0..a/2 → P1, a/2..a → P2).
                    const int tsz = scheme_ptr->tile_size;
                    if (tsz > 0) {
                        const int p1_tiles = scheme_ptr->partition_sizes[0] / tsz;
                        const int p2_tiles = scheme_ptr->partition_sizes[1] / tsz;
                        const int half = (a_exec >= 2) ? (a_exec / 2) : 1;
                        const int rem  = (a_exec >= 2) ? (a_exec - half) : 1;
                        ra_parts.push_back({0,        p1_tiles,            0,                    half});
                        ra_parts.push_back({p1_tiles, p1_tiles + p2_tiles, (a_exec >= 2) ? half : 0, rem});
                    }
                }
            }
            const alg::RegionAPartition* rap = ra_parts.empty() ? nullptr : ra_parts.data();
            const int rapn = static_cast<int>(ra_parts.size());
            // ---- Region B (root-separator tree) core cap ----
            // worldsize_b used to be a_exec (= ALL cores): slicing the root
            // separator across every core over-subscribes a SMALL root (each
            // rank gets a sliver, but the cross-rank barrier+DGEADD reduce cost
            // scales with rank count). Measured at c40: shipsec8 tree 1.99x
            // SLOWER, s3dkq4m2 1.44x, ship_003 1.34x. Conversely a WIDE root
            // (consph/audikw — where the tree genuinely wins, 0.85-0.91) wants
            // full parallelism. cap = clamp(beta * sep_tiles, 2, a_exec) does
            // both: narrow root -> few ranks (tree ~ pure-ND, no over-sub),
            // wide root -> stays near a_exec. red_tree_separator_level is the
            // active sep (root_sep for composed, chosen_sep for arrowhead).
            // PROVISIONAL beta=2 — calibrate from the wide-sep giants' root_sep.
            // Env: STILES_TREE_ALPHA (beta), STILES_TREE_MAX_CORES (hard ceiling).
            int tree_cores = a_exec;
            // CAP ONLY THE COMPOSED PATH. The arrowhead tree (no partitions) is
            // where the tree's MEASURED wins live (sem_n*/INLA +12-17%, bcsstk13/15,
            // 8rtKSK) and its parallelism is the GEMM ACCUMULATION into a NARROW
            // corner (chosen_sep=1), not the separator width — a beta*sep cap
            // would throttle sem_n* to 2 ranks and kill that win. So leave
            // arrowhead at worldsize_b=a_exec (exactly as those wins were
            // measured) and cap only the composed root separator, where the
            // over-subscription was measured (shipsec8 c40 1.99x).
            if (scheme_ptr->scotch_partition_collection) {
                const int rs = scheme_ptr->red_tree_separator_level;  // composed root_sep
                if (rs > 0) {
                    int beta = 2;
                    if (const char* e = std::getenv("STILES_TREE_ALPHA")) { const int v = std::atoi(e); if (v > 0) beta = v; }
                    long long cap = static_cast<long long>(beta) * rs;
                    if (cap < 2)      cap = 2;
                    if (cap > a_exec) cap = a_exec;
                    tree_cores = static_cast<int>(cap);
                }
            }
            // Hard ceiling (manual override) applies to either path if set.
            if (const char* e = std::getenv("STILES_TREE_MAX_CORES")) { const int v = std::atoi(e); if (v > 0 && v < tree_cores) tree_cores = v; }
            // Publish the tree worker count to the executor: the reduce-wait loops in
            // pthreads_dpotrf_reduction_{semi,dense} must bound by tree_cores, not
            // STILES_SIZE — ranks >= tree_cores never run Region B (myrank < worldsize_b
            // gate) so they never signal their tree dependency. Single writer (corea==0);
            // identical value across collectors, so no meaningful race.
            if (corea == 0) scheme_ptr->red_tree_cores = tree_cores;
            produced = alg::chol_reduction_variant4(state, method, corea, a, tree_cores, num_tiles,
                                                    scheme_ptr->red_tree_separator_level, scheme_ptr->trees,
                                                    map_id, local_tasks, 64, rap, rapn);
        } else if (!flat_sched && scheme_ptr->nd_padding > 0 &&
                   (scheme_ptr->num_partitions > 0 || scheme_ptr->partition_sizes)) {
            // ========== NESTED DISSECTION MODE (N-way scheduling) ==========
            // Generalized from the original 3-way (P1 | P2 | Sep) layout to any
            // postorder list of tile-aligned regions described by scheme_ptr->
            // partitions. Each PartitionDesc says:
            //   - [start_tile, end_tile): tile range in the permuted matrix
            //   - [first_core, first_core + num_cores): consecutive cores
            //     assigned to emit tasks for this region
            //   - label: short tag used in the per-core log
            //
            // Backward compatibility: when num_partitions == 0 but
            // partition_sizes[3] is populated (legacy symbolic_ND_phase / older
            // SCOTCH-padding path), we synthesize a 3-entry partition list on
            // the fly that reproduces the original behavior exactly
            // (cores 0..a/2 → P1, cores a/2..a → P2, all → Sep).
            //
            // Dependencies are carried at the tile level via ss_cond_wait/set,
            // same as before — partition assignment only affects which core
            // EMITS the task for a given (k, m, n). Within each core's stream,
            // tasks remain in k-ascending order after the outer sort pass.

            const int tile_size = scheme_ptr->tile_size;

            // Build a temporary partition view so the executor loop below
            // doesn't have to special-case legacy vs new layouts.
            struct PartitionView {
                int start_tile, end_tile;
                int first_core, num_cores;
                const char* label;
            };
            std::vector<PartitionView> parts;
            if (scheme_ptr->num_partitions > 0) {
                parts.reserve(static_cast<std::size_t>(scheme_ptr->num_partitions));
                for (int p = 0; p < scheme_ptr->num_partitions; ++p) {
                    const auto& pd = scheme_ptr->partitions[p];
                    parts.push_back({pd.start_tile, pd.end_tile,
                                     pd.first_core, pd.num_cores, pd.label});
                }
            } else {
                // Legacy 3-way synthesis (matches the original hardcoded logic).
                const int p1_tiles  = scheme_ptr->partition_sizes[0] / tile_size;
                const int p2_tiles  = scheme_ptr->partition_sizes[1] / tile_size;
                const int sep_tiles = scheme_ptr->partition_sizes[2] / tile_size;
                const int half_cores = (a >= 2) ? (a / 2) : 1;
                const int remaining  = (a >= 2) ? (a - half_cores) : 1;
                const int p1_first   = 0;
                const int p2_first   = (a >= 2) ? half_cores : 0;
                parts.push_back({0,                     p1_tiles,
                                 p1_first,              half_cores,           "P1"});
                parts.push_back({p1_tiles,              p1_tiles + p2_tiles,
                                 p2_first,              remaining,            "P2"});
                parts.push_back({p1_tiles + p2_tiles,   p1_tiles + p2_tiles + sep_tiles,
                                 0,                     a,                    "Sep"});
            }

            // Per-region task collection — each core only emits for partitions
            // that include its rank.
            std::vector<int> per_region_produced(parts.size(), 0);
            int total_produced = 0;
            local_tasks.reserve(256);
            for (std::size_t p = 0; p < parts.size(); ++p) {
                const auto& pv = parts[p];
                if (pv.num_cores <= 0 || pv.end_tile <= pv.start_tile) continue;
                if (corea < pv.first_core || corea >= pv.first_core + pv.num_cores) continue;

                const int rank_in_region = corea - pv.first_core;
                std::vector<std::array<int,7>> region_tasks;
                int prod = alg::chol_expansion_variant4_ND(
                    state, method,
                    rank_in_region, pv.num_cores,
                    num_tiles, map_id, region_tasks,
                    pv.start_tile, pv.end_tile);

                if (prod < 0) { produced = prod; break; }
                per_region_produced[p] = prod;
                total_produced += prod;
                local_tasks.insert(local_tasks.end(),
                    std::make_move_iterator(region_tasks.begin()),
                    std::make_move_iterator(region_tasks.end()));
            }
            if (produced >= 0) produced = total_produced;

            // Per-core summary — one line per core so you can see the binding
            // (cores assigned to leaf k will have non-zero counts only for
            // leaf k and the separators up the ancestor chain).
            {
                std::string regions_summary;
                for (std::size_t p = 0; p < parts.size(); ++p) {
                    if (!regions_summary.empty()) regions_summary += ", ";
                    regions_summary += parts[p].label
                                     + std::string("=")
                                     + std::to_string(per_region_produced[p]);
                }
                sTiles::Logger::timing("│   ↪ [Path 2 ND chol] core " + std::to_string(corea) +
                                       "/" + std::to_string(a) +
                                       " parts=" + std::to_string(parts.size()) +
                                       ": " + regions_summary);
            }
        } else {
            // ========== STANDARD MODE ==========
            // [Path 3 standard chol] log gated `if (false)` — pure diagnostic.
            if (false && corea == 0) {
                sTiles::Logger::timing("│   ↪ [Path 3 standard chol] num_tiles=" + std::to_string(num_tiles)
                                       + ", cores=" + std::to_string(a));
            }
            produced = alg::chol_expansion_variant4(state, method, corea, a, num_tiles, map_id, local_tasks);
        }

        #ifdef STILES_COLLECT_TASKS
        export_core_tasks_to_txt(local_tasks,
                                 corea,
                                 group_index,
                                 *call_info,
                                 has_red_tree ? "chol_reduction" : "chol_expansion",
                                 false);
        #endif

        if (produced < 0) {
            return sTiles::StatusCode::ExecutionFailed;
        } else if (produced > 0) {
            chol_out.insert(chol_out.end(), std::make_move_iterator(local_tasks.begin()), std::make_move_iterator(local_tasks.end()));
        }

        bool scaled_inv = false;
        if (scheme_ptr->compute_inverse && inv_out) {
            std::vector<std::array<int,7>> inv_local;
            int produced_inv = 0;
            if ((*call_info)->factorization_variant == 2) {
                produced_inv = alg::inv_scaled_variant4(state, method, corea, a, num_tiles, map_id, inv_local);
                scaled_inv = true;
            } else {
                produced_inv = alg::inv_variant4(state, method, corea, a, num_tiles, map_id, inv_local);
            }

            #ifdef STILES_COLLECT_TASKS
            export_core_tasks_to_txt(inv_local,
                                     corea,
                                     group_index,
                                     *call_info,
                                     scaled_inv ? "inv_scaled" : "inv",
                                     true);
            #endif


            if (produced_inv < 0) {
                return sTiles::StatusCode::ExecutionFailed;
            } else if (produced_inv > 0) {
                inv_out->insert(inv_out->end(), std::make_move_iterator(inv_local.begin()), std::make_move_iterator(inv_local.end()));
            }
        }

        return sTiles::StatusCode::Success;
    }
};

// Build precomputed gather offsets for semisparse inv tasks (cases 7 and 8).
// Must be called after inv_tasks are fully built. Populates inv_gather_packed and inv_gather_index.
inline void build_inv_gather_info(TiledMatrix* scheme) {
    if (!scheme || !scheme->inv_tasks || scheme->inv_tasks->empty()) return;
    if (!scheme->semisparseTileMetaCore || !scheme->tileMetaCore) return;

    // Test hook: STILES_FORCE_NO_GATHER=1 skips precomputation so selinv takes
    // the on-the-fly gather fallback (cases 7/8) that giant matrices hit when
    // the packed buffer would exceed INT32_MAX. Lets the small-matrix harness
    // exercise + ASan-check that path without a multi-GB matrix.
    {
        static const bool force_no_gather = [] {
            const char* e = std::getenv("STILES_FORCE_NO_GATHER");
            return e && e[0] == '1';
        }();
        if (force_no_gather) {
            scheme->inv_gather_index.reset();
            scheme->inv_gather_packed.reset();
            return;
        }
    }

    const auto& tasks = *scheme->inv_tasks;
    // 64-bit: on huge matrices the task count exceeds INT32_MAX. A truncating
    // int cast wraps negative, and make_shared<vector<>>(negative*stride) then
    // throws std::length_error before the INT32 packed-size guard below runs.
    const int64_t n_tasks = static_cast<int64_t>(tasks.size());

    // Pre-flight: estimate the maximum size of the packed buffer.
    // data_offset is stored as int32 (see lines below), so if packed.size()
    // exceeds INT32_MAX the offset wraps to negative and the consumer in
    // pdtrtri.cpp dereferences a pointer ~8 GB before gather_packed (SEGV).
    // In that case, skip precomputation entirely; the consumers
    // (pdtrtri.cpp cases 7/8) have an on-the-fly fallback gated on
    // has_gather_info==false.
    {
        int64_t estimated_packed = 0;
        for (int64_t t = 0; t < n_tasks; ++t) {
            const auto& task = tasks[static_cast<std::size_t>(t)];
            const int myroutine = task[0];
            if (myroutine == 7) {
                // worst case: partial branch pushes 2 ints per active column of fact (sa1)
                const int index1 = task[4];
                estimated_packed += 2LL * scheme->semisparseTileMetaCore[index1].sa;
            } else if (myroutine == 8) {
                const int index2 = task[5];
                const bool inv2_is_diag =
                    (scheme->tileMetaCore[index2].row == scheme->tileMetaCore[index2].col);
                if (!inv2_is_diag) {
                    // worst case: partial branch pushes 2 ints per active column of inv3 (sa3)
                    const int index3 = task[6];
                    estimated_packed += 2LL * scheme->semisparseTileMetaCore[index3].sa;
                }
            }
        }
        constexpr int64_t SAFETY_MARGIN = 1024;
        constexpr int64_t INT32_LIMIT = static_cast<int64_t>(INT32_MAX) - SAFETY_MARGIN;
        if (estimated_packed > INT32_LIMIT) {
            sTiles::Logger::warning(std::string("│   ↪ inv_gather precomputation skipped: estimated packed size ")
                + std::to_string(estimated_packed)
                + " > INT32_MAX (" + std::to_string(INT32_LIMIT)
                + "); falling back to on-the-fly gather in selinv");
            scheme->inv_gather_index.reset();
            scheme->inv_gather_packed.reset();
            return;
        }
    }

    auto& index = *(scheme->inv_gather_index = std::make_shared<std::vector<int32_t>>(
        static_cast<std::size_t>(n_tasks) * 3, 0));
    auto& packed = *(scheme->inv_gather_packed = std::make_shared<std::vector<int32_t>>());
    packed.reserve(static_cast<std::size_t>(n_tasks) * 4); // rough estimate

    for (int64_t t = 0; t < n_tasks; ++t) {
        const auto& task = tasks[static_cast<std::size_t>(t)];
        const int myroutine = task[0];
        const int64_t idx_base = t * 3; // [data_offset, n_valid, flags]

        if (myroutine == 7) {
            // Case 7: col_map maps fact columns (aind1) to inv2 stored columns (acol2)
            const int index1 = task[4]; // fact (i,k)
            const int index2 = task[5]; // inv2 (j,k)
            const auto& semi1 = scheme->semisparseTileMetaCore[index1];
            const auto& semi2 = scheme->semisparseTileMetaCore[index2];
            const int sa1 = semi1.sa;
            const int sa2 = semi2.sa;
            const int m_inv2 = scheme->tileMetaCore[index2].height;
            const int* aind1 = semi1.aind.data();
            const int* acol2 = semi2.acol.data();
            const int acol2_sz = static_cast<int>(semi2.acol.size());

            // Build col_map and count valid entries
            int n_valid = 0;
            // Temporary: store col_map results to decide all_valid vs partial
            std::vector<std::pair<int,int>> valid_entries; // (jj, offset)
            valid_entries.reserve(static_cast<std::size_t>(sa1));
            for (int jj = 0; jj < sa1; ++jj) {
                const int k = aind1[jj];
                const int idx = (k >= 0 && k < acol2_sz) ? acol2[k] : -1;
                const int stored = (idx >= 0 && idx < sa2) ? idx : -1;
                if (stored >= 0) {
                    valid_entries.push_back({jj, stored * m_inv2});
                    ++n_valid;
                }
            }

            if (n_valid == 0) {
                index[static_cast<std::size_t>(idx_base)]     = 0;
                index[static_cast<std::size_t>(idx_base) + 1] = 0;
                index[static_cast<std::size_t>(idx_base) + 2] = 0; // skip
            } else if (n_valid == sa1) {
                // All valid: just store offsets sequentially
                const int data_offset = static_cast<int>(packed.size());
                for (const auto& ve : valid_entries)
                    packed.push_back(ve.second); // col_offset
                index[static_cast<std::size_t>(idx_base)]     = data_offset;
                index[static_cast<std::size_t>(idx_base) + 1] = n_valid;
                index[static_cast<std::size_t>(idx_base) + 2] = 1; // all_valid
            } else {
                // Partial: interleave [valid_jj, col_offset, ...]
                const int data_offset = static_cast<int>(packed.size());
                for (const auto& ve : valid_entries) {
                    packed.push_back(ve.first);  // valid_jj
                    packed.push_back(ve.second); // col_offset
                }
                index[static_cast<std::size_t>(idx_base)]     = data_offset;
                index[static_cast<std::size_t>(idx_base) + 1] = n_valid;
                index[static_cast<std::size_t>(idx_base) + 2] = 2; // partial
            }
        } else if (myroutine == 8) {
            // Case 8: col_map maps inv3 columns (aind3) to inv2 stored columns (acol2)
            const int index2 = task[5]; // inv2 (k,j)
            const int index3 = task[6]; // inv3 (i,j)
            const bool inv2_is_diag = (scheme->tileMetaCore[index2].row == scheme->tileMetaCore[index2].col);

            if (inv2_is_diag) {
                // Diagonal inv2: no col_map needed, direct access
                index[static_cast<std::size_t>(idx_base)]     = 0;
                index[static_cast<std::size_t>(idx_base) + 1] = 0;
                index[static_cast<std::size_t>(idx_base) + 2] = 3; // diagonal, use original code
            } else {
                const auto& semi2 = scheme->semisparseTileMetaCore[index2];
                const auto& semi3 = scheme->semisparseTileMetaCore[index3];
                const int sa2 = semi2.sa;
                const int sa3 = semi3.sa;
                const int m_inv2 = scheme->tileMetaCore[index2].height;
                const int* aind3 = semi3.aind.data();
                const int* acol2 = semi2.acol.data();
                const int acol2_sz = static_cast<int>(semi2.acol.size());

                int n_valid = 0;
                std::vector<std::pair<int,int>> valid_entries;
                valid_entries.reserve(static_cast<std::size_t>(sa3));
                for (int cc = 0; cc < sa3; ++cc) {
                    const int c = aind3[cc];
                    const int idx = (c >= 0 && c < acol2_sz) ? acol2[c] : -1;
                    const int stored = (idx >= 0 && idx < sa2) ? idx : -1;
                    if (stored >= 0) {
                        valid_entries.push_back({cc, stored * m_inv2});
                        ++n_valid;
                    }
                }

                if (n_valid == 0) {
                    index[static_cast<std::size_t>(idx_base)]     = 0;
                    index[static_cast<std::size_t>(idx_base) + 1] = 0;
                    index[static_cast<std::size_t>(idx_base) + 2] = 0; // skip
                } else if (n_valid == sa3) {
                    const int data_offset = static_cast<int>(packed.size());
                    for (const auto& ve : valid_entries)
                        packed.push_back(ve.second);
                    index[static_cast<std::size_t>(idx_base)]     = data_offset;
                    index[static_cast<std::size_t>(idx_base) + 1] = n_valid;
                    index[static_cast<std::size_t>(idx_base) + 2] = 1; // all_valid
                } else {
                    const int data_offset = static_cast<int>(packed.size());
                    for (const auto& ve : valid_entries) {
                        packed.push_back(ve.first);
                        packed.push_back(ve.second);
                    }
                    index[static_cast<std::size_t>(idx_base)]     = data_offset;
                    index[static_cast<std::size_t>(idx_base) + 1] = n_valid;
                    index[static_cast<std::size_t>(idx_base) + 2] = 2; // partial
                }
            }
        } else {
            // Cases 1-6, 9: not applicable
            index[static_cast<std::size_t>(idx_base) + 2] = 0xFF;
        }
    }

    packed.shrink_to_fit();
}

// Build precomputed scatter info for semisparse chol tasks (case 4 DGEMM).
//
// Two enum modes are supported, picked based on which factorization function
// will consume this data (mirrors the dispatch in pdpotrf.cpp at line ~6135):
//
//   ORIGINAL enum (dense-only) — used by imp3_serial / imp3 / imp1 / semisparse:
//     0 = direct GEMM             (a_contig && b_contig && out_full)
//     1 = fused contiguous        (a_contig, small tiles)
//     2 = fused indexed scatter   (small tiles)
//     3 = BLAS + contiguous scatter   (a_contig, large tiles)
//     4 = BLAS + indexed scatter      (large tiles)
//
//   SPARSE-AWARE enum — used by imp3_serial_and_sparse only:
//     Same 0-4 values, BUT when the per-tile density is below the sparse
//     threshold the function reads paths 3/4 differently (sparse fast path).
//     The sparse-aware path is decided at runtime via __builtin_expect inside
//     imp3_serial_and_sparse, NOT here, so this function still writes the
//     same 0-4 enum but the density-eligibility flag is unused.
//
// We pick which enum to write based on stiles_control_params[9] and [14] AND
// scheme->num_cores (proxy for STILES_SIZE in single-process mode), so the
// scatter info matches whichever function will consume it at factor time.
inline void build_chol_scatter_info(TiledMatrix* scheme) {
    if (!scheme || !scheme->chol_tasks || scheme->chol_tasks->empty()) return;
    if (!scheme->semisparseTileMetaCore || !scheme->tileMetaCore) return;

    // Look up the dispatch hint so the enum we write matches the consumer.
    // Mirrors the dispatch chain in pdpotrf.cpp:
    //   if (STILES_SIZE == 1 && stiles_control_params[14] == 0) {
    //       if (params[9] == 2) → imp3_serial                   (ORIGINAL enum)
    //       else                → imp3_serial_and_sparse        (SPARSE-AWARE enum)
    //   } else { ... parallel functions, all read ORIGINAL enum }
    //
    // We use scheme->num_cores as a stand-in for STILES_SIZE since the world
    // size is propagated into scheme->num_cores at preprocessing time.
    const int* params = sTiles_get_params();
    const bool serial_path = (scheme->num_cores <= 1) && (params[sTiles::param::SerialMode] == 0);
    const bool target_is_sparse_aware = serial_path && (params[sTiles::param::SemisparseImpl] != 2);
    (void)target_is_sparse_aware; // currently both enum modes share the same
                                  // 0-4 layout; the sparse fast path inside
                                  // imp3_serial_and_sparse decides at runtime.
                                  // Keep the hint plumbed so future changes can
                                  // emit a sparse-aware enum here without
                                  // breaking imp3_serial.

    const auto& tasks = *scheme->chol_tasks;
    // 64-bit: on huge matrices the chol task count exceeds INT32_MAX. A
    // truncating int cast wraps negative, so make_shared<vector<int64_t>>(
    // static_cast<size_t>(n_tasks)*2) below allocates a ~10^19-element vector
    // and throws std::length_error (the crash on bern_spd, n=6.2M).
    const int64_t n_tasks = static_cast<int64_t>(tasks.size());
    constexpr int FUSE_THRESHOLD = 16;

    // chol_scatter_index stores 64-bit data_offset so packed.size() can exceed
    // INT32_MAX on huge matrices (the previous int32 overflow guard would
    // null-out the shared_ptr and trigger a deref crash in imp3_serial which
    // assumes the info is always populated when param[7]==1).
    auto& index = *(scheme->chol_scatter_index = std::make_shared<std::vector<int64_t>>(
        static_cast<std::size_t>(n_tasks) * 2, 0));
    auto& packed = *(scheme->chol_scatter_packed = std::make_shared<std::vector<int32_t>>());
    packed.reserve(static_cast<std::size_t>(n_tasks) * 2);

    for (int64_t t = 0; t < n_tasks; ++t) {
        const auto& task = tasks[static_cast<std::size_t>(t)];
        const int myroutine = task[0];
        const int64_t idx_base = t * 2;

        if (myroutine != 4) {
            index[static_cast<std::size_t>(idx_base) + 1] = 0xFF; // not case 4
            continue;
        }

        const int index1 = task[4]; // A tile (n,k)
        const int index2 = task[5]; // B tile (n,m)
        const int index3 = task[6]; // output tile (k,m)

        // Defensive bound-check: routine-4 tasks emitted by the chol collectors
        // should always have valid tile indices for the active mapper, but a
        // mismatch between the symbolic graph and the mapper has historically
        // produced -1 indices here. Mark such tasks as "skip" so the executor's
        // null-tile guards do the rest, rather than dereferencing OOB metadata.
        const int nat = scheme->numActiveTiles;
        if (index1 < 0 || index1 >= nat || index2 < 0 || index2 >= nat || index3 < 0 || index3 >= nat) {
            index[static_cast<std::size_t>(idx_base) + 1] = 0xFF;
            continue;
        }

        const auto& semi_a = scheme->semisparseTileMetaCore[index1];
        const auto& semi_b = scheme->semisparseTileMetaCore[index2];
        const auto& semi_out = scheme->semisparseTileMetaCore[index3];

        const int cols_a = semi_a.sa;
        const int cols_b = semi_b.sa;
        const bool a_contig = semi_a.is_contiguous;
        const bool b_contig = semi_b.is_contiguous;
        const bool out_full = semi_out.is_full_width;

        if (a_contig && b_contig && out_full) {
            // Path 0: direct GEMM into output — no scatter needed
            index[static_cast<std::size_t>(idx_base)]     = 0;
            index[static_cast<std::size_t>(idx_base) + 1] = 0;
        } else if (cols_a <= FUSE_THRESHOLD && cols_b <= FUSE_THRESHOLD) {
            // Path 1 or 2: fused compute-scatter
            // Precompute slot_map for B columns
            const int64_t data_offset = static_cast<int64_t>(packed.size());
            const int* aind_b = semi_b.aind.data();
            const int* acol_map = semi_out.acol.data();
            for (int j = 0; j < cols_b; ++j) {
                packed.push_back(acol_map[aind_b[j]]);
            }
            index[static_cast<std::size_t>(idx_base)]     = data_offset;
            index[static_cast<std::size_t>(idx_base) + 1] = a_contig ? 1 : 2;
        } else {
            // Path 3 or 4: BLAS DGEMM + scatter
            // Precompute slot_map for B columns
            const int64_t data_offset = static_cast<int64_t>(packed.size());
            const int* aind_b = semi_b.aind.data();
            const int* acol_map = semi_out.acol.data();
            for (int j = 0; j < cols_b; ++j) {
                packed.push_back(acol_map[aind_b[j]]);
            }
            index[static_cast<std::size_t>(idx_base)]     = data_offset;
            index[static_cast<std::size_t>(idx_base) + 1] = a_contig ? 3 : 4;
        }
    }

    packed.shrink_to_fit();
}

sTiles::StatusCode collect_tasks(sTiles_call **call_info, TiledMatrix **scheme, int group_index, int num_cores, int num_threads_level1, int rescale_cores, bool collect_chol_inv = true) {
    // This function is only called for sparse variants (0 or 3) now
    if (!call_info || !*call_info || !scheme || !*scheme) {
        return sTiles::StatusCode::IllegalValue;
    }

    TiledMatrix* scheme_ptr = *scheme;
    const int num_tiles = scheme_ptr->dimTiledMatrix;
    const TileIndexer::Method method = scheme_ptr->neighbor_lookup_method;
    TileIndexer::State& state = scheme_ptr->state;

    // Preprocessing-time-measurement gate. When STILES_SKIP_SOLVE_INV_PREP is set,
    // skip the selected-inverse AND triangular-solve task collection so init_group
    // reflects the chol-only preprocessing cost. Intended for symbolic/preprocess
    // benchmarking (e.g. bench_ordering, which stops after init and never solves).
    // Do NOT set it for a workload that then solves or computes the inverse — the
    // tile solve/selinv paths need these task lists.
    const bool skip_solve_inv_prep = (std::getenv("STILES_SKIP_SOLVE_INV_PREP") != nullptr);

    // Cap num_cores to problem size — no point forking more threads than tile rows
    const int available = std::min(num_cores, omp_get_max_threads());
    const int eff_cores = (num_tiles <= 0) ? 1 : std::min(available, std::max(1, num_tiles / 16));

    try {
        if (state.graph_N != num_tiles || !state.graph_include_self || state.graph_edges_up.empty() || state.graph_edges_lo.empty()) {
            TileIndexer::build_graphs_up_lo_parallel(state, num_tiles, true, eff_cores);
        }

        TiledMatrix* scheme_local = *scheme;
        MapIdFunctor map_functor{scheme_local, num_tiles};
        std::function<int(int,int)> map_id = map_functor;

        if (collect_chol_inv) {
            scheme_ptr->chol_tasks.reset();
            scheme_ptr->chol_task_offsets.reset();
            scheme_ptr->inv_tasks.reset();
            scheme_ptr->inv_task_offsets.reset();
        }
        scheme_ptr->solve_fwd_tasks.reset();
        scheme_ptr->solve_fwd_offsets.reset();
        scheme_ptr->solve_bwd_tasks.reset();
        scheme_ptr->solve_bwd_offsets.reset();
        scheme_ptr->gpu_solve_fwd_tasks.reset();
        scheme_ptr->gpu_solve_bwd_tasks.reset();

        if((*call_info)->parameters[static_cast<int>(sTiles::Parameter::BoostedETrick)]==1){
                const int a = scheme_ptr->use_gpu ? scheme_ptr->num_gpu_streams : (*call_info)->num_cores;
                const bool has_red_tree = (scheme_ptr->red_tree_separator_level > 0);
                sTiles::StatusCode status = sTiles::StatusCode::Success;

                if (collect_chol_inv) {
                    // ── CHOL: collect with P threads, remap P→a, sort ──
                    // Normally P=max(a, omp_max) for faster parallel collection,
                    // then `p % a` remaps to execution bins. That remap is fine
                    // for Path 3 (homogeneous task pool) but FATALLY breaks
                    // Path 2 (ND partitioned): P1 tasks from collection-rank k
                    // and P2 tasks from collection-rank k+half land in the same
                    // execution bin via `p % a`, interleaving the partitions
                    // and completely destroying the P1|P2 binding that
                    // ProcessCore just carefully arranged. To preserve partition
                    // segregation, fix P=a when Path 2 is active — then
                    // `p % a == p` is identity and each execution bin holds
                    // exactly the tasks its ProcessCore rank collected.
                    //
                    // For the tree path (has_red_tree): keep P=max(a, omp_max)
                    // so Region A still benefits from full collection parallelism.
                    // Region B's per-execution-rank slicing is handled inside
                    // chol_reduction_variant via a separate `worldsize_b=a`
                    // parameter, with collection ranks myrank >= a skipping
                    // Region B entirely. The `p % a` remap is then identity for
                    // Region B (only ranks 0..a-1 emit it) and standard
                    // interleaving for Region A (post-sort restores order).
                    const bool nd_partition_path =
                        (scheme_ptr->nd_padding > 0 && scheme_ptr->partition_sizes);
                    const int P = nd_partition_path
                        ? a
                        : std::max(a, omp_get_max_threads());
                    const double _ct_t0 = omp_get_wtime();

                    std::vector<std::vector<std::array<int,7>>> tmp_chol_bins(static_cast<std::size_t>(P));
                    ProcessCore chol_collector{state, method, P, a, num_tiles, group_index, call_info, scheme_ptr, map_id, has_red_tree};

                    if (P > 1) {
                        std::atomic<int> error_flag{0};
#ifdef _OPENMP
                        #pragma omp parallel num_threads(P)
                        {
#else
                        {
#endif
#ifdef _OPENMP
                            #pragma omp for schedule(dynamic,1) nowait
#endif
                            for (int corea = 0; corea < P; ++corea) {
                                if (error_flag.load(std::memory_order_relaxed) != 0) continue;
                                auto& chol_bucket = tmp_chol_bins[static_cast<std::size_t>(corea)];
                                sTiles::StatusCode local_status = chol_collector(corea, chol_bucket, nullptr);
                                if (local_status != sTiles::StatusCode::Success) {
                                    error_flag.store(static_cast<int>(local_status), std::memory_order_relaxed);
                                }
                            }
                        }

                        const int err = error_flag.load(std::memory_order_relaxed);
                        if (err != 0) {
                            return static_cast<sTiles::StatusCode>(err);
                        }
                    } else {
                        for (int corea = 0; corea < P; ++corea) {
                            auto& chol_bucket = tmp_chol_bins[static_cast<std::size_t>(corea)];
                            sTiles::StatusCode sc_local = chol_collector(corea, chol_bucket, nullptr);
                            if (sc_local != sTiles::StatusCode::Success) {
                                return sc_local;
                            }
                        }
                    }

                    const double _ct_t1 = omp_get_wtime();
                    // std::fprintf(stderr, "[collect_tasks] chol generate: %.4f s (P=%d, a=%d, tiles=%d)\n", _ct_t1 - _ct_t0, P, a, num_tiles);

                    // Remap P collection bins → a execution bins via p % a
                    std::vector<std::vector<std::array<int,7>>> chol_bins(static_cast<std::size_t>(a));
                    for (int p = 0; p < P; ++p) {
                        const int tgt = p % a;
                        auto& src = tmp_chol_bins[static_cast<std::size_t>(p)];
                        if (!src.empty()) {
                            chol_bins[static_cast<std::size_t>(tgt)].insert(
                                chol_bins[static_cast<std::size_t>(tgt)].end(),
                                std::make_move_iterator(src.begin()),
                                std::make_move_iterator(src.end()));
                        }
                    }

                    const double _ct_t2 = omp_get_wtime();
                    // std::fprintf(stderr, "[collect_tasks] chol remap: %.4f s\n", _ct_t2 - _ct_t1);

                    // Sort each chol bin by (k, m, n) to restore panel order.
                    // row[2]=k, row[1]=m, row[3]=n.
                    for (int b = 0; b < a; ++b) {
                        auto& cb = chol_bins[static_cast<std::size_t>(b)];
                        if (P > 1 && cb.size() > 1) {
                            sTiles::sort(cb.begin(), cb.end(), [](const std::array<int,7>& x, const std::array<int,7>& y) {
                                if (x[2] != y[2]) return x[2] < y[2]; // k ascending
                                if (x[1] != y[1]) return x[1] < y[1]; // m ascending
                                return x[3] < y[3];                    // n ascending
                            });
                        }
                    }

                    const double _ct_t3 = omp_get_wtime();
                    // std::fprintf(stderr, "[collect_tasks] chol sort: %.4f s\n", _ct_t3 - _ct_t2);

                    // Flatten chol bins into final task array
                    std::size_t chol_total = 0;
                    for (const auto& bin : chol_bins) chol_total += bin.size();
                    auto& chol_tasks = sTiles::ensure_chol_tasks(scheme_ptr);
                    auto& chol_offsets = sTiles::ensure_chol_task_offsets(scheme_ptr);
                    chol_tasks.clear();
                    chol_tasks.reserve(chol_total);
                    chol_offsets.assign(static_cast<std::size_t>(a) + 1, 0);
                    for (int core = 0; core < a; ++core) {
                        chol_offsets[static_cast<std::size_t>(core)] = static_cast<long long>(chol_tasks.size());
                        auto& bin = chol_bins[static_cast<std::size_t>(core)];
                        if (!bin.empty()) {
                            chol_tasks.insert(chol_tasks.end(), std::make_move_iterator(bin.begin()), std::make_move_iterator(bin.end()));
                        }
                    }
                    chol_offsets[static_cast<std::size_t>(a)] = static_cast<long long>(chol_tasks.size());

                    const double _ct_t4 = omp_get_wtime();
                    {
                        std::size_t n_chol = scheme_ptr->chol_tasks ? scheme_ptr->chol_tasks->size() : 0;
                        // std::fprintf(stderr, "[collect_tasks] chol flatten: %.4f s  chol=%zu\n", _ct_t4 - _ct_t3, n_chol);
                    }

                    // ── INV: collect with P threads, remap P→a, stable sort, dedup barriers ──
                    if (scheme_ptr->compute_inverse && !skip_solve_inv_prep) {
                        const double _inv_t0 = omp_get_wtime();
                        std::vector<std::vector<std::array<int,7>>> tmp_inv_bins(static_cast<std::size_t>(P));
                        ProcessCore inv_collector{state, method, P, a, num_tiles, group_index, call_info, scheme_ptr, map_id, has_red_tree};

                        if (P > 1) {
                            std::atomic<int> inv_error_flag{0};
#ifdef _OPENMP
                            #pragma omp parallel num_threads(P)
                            {
#else
                            {
#endif
#ifdef _OPENMP
                                #pragma omp for schedule(dynamic,1) nowait
#endif
                                for (int corea = 0; corea < P; ++corea) {
                                    if (inv_error_flag.load(std::memory_order_relaxed) != 0) continue;
                                    std::vector<std::array<int,7>> dummy_chol;
                                    auto* inv_bucket = &tmp_inv_bins[static_cast<std::size_t>(corea)];
                                    sTiles::StatusCode sc_local = inv_collector(corea, dummy_chol, inv_bucket);
                                    if (sc_local != sTiles::StatusCode::Success) {
                                        inv_error_flag.store(static_cast<int>(sc_local), std::memory_order_relaxed);
                                    }
                                }
                            }

                            const int inv_err = inv_error_flag.load(std::memory_order_relaxed);
                            if (inv_err != 0) {
                                return static_cast<sTiles::StatusCode>(inv_err);
                            }
                        } else {
                            for (int corea = 0; corea < P; ++corea) {
                                std::vector<std::array<int,7>> dummy_chol;
                                auto* inv_bucket = &tmp_inv_bins[static_cast<std::size_t>(corea)];
                                sTiles::StatusCode sc_local = inv_collector(corea, dummy_chol, inv_bucket);
                                if (sc_local != sTiles::StatusCode::Success) {
                                    return sc_local;
                                }
                            }
                        }

                        const double _inv_t1 = omp_get_wtime();
                        // std::fprintf(stderr, "[collect_tasks] inv generate: %.4f s\n", _inv_t1 - _inv_t0);

                        // Remap P → a
                        std::vector<std::vector<std::array<int,7>>> inv_bins(static_cast<std::size_t>(a));
                        for (int p = 0; p < P; ++p) {
                            const int tgt = p % a;
                            auto& src = tmp_inv_bins[static_cast<std::size_t>(p)];
                            if (!src.empty()) {
                                inv_bins[static_cast<std::size_t>(tgt)].insert(
                                    inv_bins[static_cast<std::size_t>(tgt)].end(),
                                    std::make_move_iterator(src.begin()),
                                    std::make_move_iterator(src.end()));
                            }
                        }

                        const double _inv_t2 = omp_get_wtime();
                        // std::fprintf(stderr, "[collect_tasks] inv remap: %.4f s\n", _inv_t2 - _inv_t1);

                        // Stable sort + deduplicate barriers in each bin
                        for (int b = 0; b < a; ++b) {
                            auto& ib = inv_bins[static_cast<std::size_t>(b)];
                            if (ib.size() <= 1) continue;

                            // Stable sort: R1 < barrier < R2.
                            // Within R2: i desc, j desc.
                            // Stable preserves per-(i,j) type sequence (4→5→6, 7/8→9).
                            std::stable_sort(ib.begin(), ib.end(), [](const std::array<int,7>& x, const std::array<int,7>& y) {
                                auto region = [](int t) { return (t <= 2) ? 0 : (t == 3) ? 1 : 2; };
                                const int rx = region(x[0]);
                                const int ry = region(y[0]);
                                if (rx != ry) return rx < ry;           // R1 < barrier < R2
                                if (rx != 2) return false;              // R1/barrier: preserve order
                                if (x[1] != y[1]) return x[1] > y[1];  // R2: row i descending
                                return x[2] > y[2];                     // R2 same row: j descending
                            });

                            // Deduplicate barriers: keep exactly one type 3
                            bool seen_barrier = false;
                            ib.erase(std::remove_if(ib.begin(), ib.end(), [&seen_barrier](const std::array<int,7>& t) {
                                if (t[0] == 3) {
                                    if (seen_barrier) return true; // remove duplicate
                                    seen_barrier = true;
                                }
                                return false;
                            }), ib.end());
                        }

                        const double _inv_t3 = omp_get_wtime();
                        // std::fprintf(stderr, "[collect_tasks] inv sort+dedup: %.4f s\n", _inv_t3 - _inv_t2);

                        // Flatten inv bins
                        std::size_t inv_total = 0;
                        for (const auto& bin : inv_bins) inv_total += bin.size();
                        auto& inv_tasks = sTiles::ensure_inv_tasks(scheme_ptr);
                        auto& inv_offsets = sTiles::ensure_inv_task_offsets(scheme_ptr);
                        inv_tasks.clear();
                        inv_tasks.reserve(inv_total);
                        inv_offsets.assign(static_cast<std::size_t>(a) + 1, 0);

                        for (int core = 0; core < a; ++core) {
                            inv_offsets[static_cast<std::size_t>(core)] = static_cast<long long>(inv_tasks.size());
                            auto& bin = inv_bins[static_cast<std::size_t>(core)];
                            if (!bin.empty()) {
                                inv_tasks.insert(inv_tasks.end(), std::make_move_iterator(bin.begin()), std::make_move_iterator(bin.end()));
                            }
                        }
                        inv_offsets[static_cast<std::size_t>(a)] = static_cast<long long>(inv_tasks.size());

                        const double _inv_t4 = omp_get_wtime();
                        std::size_t n_inv = scheme_ptr->inv_tasks ? scheme_ptr->inv_tasks->size() : 0;
                        // std::fprintf(stderr, "[collect_tasks] inv flatten: %.4f s  inv=%zu  inv_total: %.4f s\n", _inv_t4 - _inv_t3, n_inv, _inv_t4 - _inv_t0);
                    }

                    {
                        std::size_t n_chol = scheme_ptr->chol_tasks ? scheme_ptr->chol_tasks->size() : 0;
                        std::size_t n_inv  = scheme_ptr->inv_tasks  ? scheme_ptr->inv_tasks->size()  : 0;
                        // std::fprintf(stderr, "[collect_tasks] total: %.4f s  chol=%zu inv=%zu\n", omp_get_wtime() - _ct_t0, n_chol, n_inv);
                    }
                } // collect_chol_inv

                // ========== SOLVE TASK COLLECTION ==========
                // Pre-collect (k, m) tile pairs for triangular solve
                // This eliminates isActive() and map_ij() calls at solve time
                // (skipped under STILES_SKIP_SOLVE_INV_PREP — preprocess timing only)
                if (!skip_solve_inv_prep)
                {
                    const int tile_size = scheme_ptr->tile_size;
                    const int N = scheme_ptr->dim;
                    const int num_row_tiles = (tile_size == 0)
                        ? 0
                        : (N + tile_size - 1) / tile_size;

                    std::vector<std::vector<std::array<int,6>>> solve_fwd_bins(static_cast<std::size_t>(a));
                    std::vector<std::vector<std::array<int,6>>> solve_bwd_bins(static_cast<std::size_t>(a));

                    // Collect solve tasks for each core (striped binning)
                    for (int corea = 0; corea < a; ++corea) {
                        solve_collect_forward(state, method, corea, a, num_tiles, tile_size, N, map_id, solve_fwd_bins[static_cast<std::size_t>(corea)]);
                        solve_collect_backward(state, method, corea, a, num_tiles, tile_size, N, map_id, solve_bwd_bins[static_cast<std::size_t>(corea)]);
                    }

                    // Combine forward solve tasks
                    std::size_t fwd_total = 0;
                    for (const auto& bin : solve_fwd_bins) fwd_total += bin.size();
                    auto& fwd_tasks = sTiles::ensure_solve_fwd_tasks(scheme_ptr);
                    auto& fwd_offsets = sTiles::ensure_solve_fwd_offsets(scheme_ptr);
                    fwd_tasks.clear();
                    fwd_tasks.reserve(fwd_total);
                    fwd_offsets.assign(static_cast<std::size_t>(a) + 1, 0);

                    for (int core = 0; core < a; ++core) {
                        fwd_offsets[static_cast<std::size_t>(core)] = static_cast<int>(fwd_tasks.size());
                        auto& bin = solve_fwd_bins[static_cast<std::size_t>(core)];
                        if (!bin.empty()) {
                            fwd_tasks.insert(fwd_tasks.end(), std::make_move_iterator(bin.begin()), std::make_move_iterator(bin.end()));
                        }
                    }
                    fwd_offsets[static_cast<std::size_t>(a)] = static_cast<int>(fwd_tasks.size());

                    // Combine backward solve tasks
                    std::size_t bwd_total = 0;
                    for (const auto& bin : solve_bwd_bins) bwd_total += bin.size();
                    auto& bwd_tasks = sTiles::ensure_solve_bwd_tasks(scheme_ptr);
                    auto& bwd_offsets = sTiles::ensure_solve_bwd_offsets(scheme_ptr);
                    bwd_tasks.clear();
                    bwd_tasks.reserve(bwd_total);
                    bwd_offsets.assign(static_cast<std::size_t>(a) + 1, 0);

                    for (int core = 0; core < a; ++core) {
                        bwd_offsets[static_cast<std::size_t>(core)] = static_cast<int>(bwd_tasks.size());
                        auto& bin = solve_bwd_bins[static_cast<std::size_t>(core)];
                        if (!bin.empty()) {
                            bwd_tasks.insert(bwd_tasks.end(), std::make_move_iterator(bin.begin()), std::make_move_iterator(bin.end()));
                        }
                    }
                    bwd_offsets[static_cast<std::size_t>(a)] = static_cast<int>(bwd_tasks.size());

                    // ========== UPDATE-COUNTER METADATA ==========
                    // expected[m] = number of off-diagonal updates targeting row m.
                    // Used by sparse-iteration kernels (see pdtrsm.cpp).
                    {
                        auto& fwd_expected = sTiles::ensure_solve_fwd_expected(scheme_ptr);
                        auto& bwd_expected = sTiles::ensure_solve_bwd_expected(scheme_ptr);
                        fwd_expected.assign(static_cast<std::size_t>(num_row_tiles), 0);
                        bwd_expected.assign(static_cast<std::size_t>(num_row_tiles), 0);
                        for (const auto& t : fwd_tasks) {
                            if (t[0] == 2 && t[2] >= 0 && t[2] < num_row_tiles)
                                ++fwd_expected[static_cast<std::size_t>(t[2])];
                        }
                        for (const auto& t : bwd_tasks) {
                            if (t[0] == 2 && t[2] >= 0 && t[2] < num_row_tiles)
                                ++bwd_expected[static_cast<std::size_t>(t[2])];
                        }
                    }

                    // ========== ROW-AFFINITY REBIN OF SOLVE TASKS ==========
                    // Semi-sparse kernels iterate the task list directly, with
                    // update-counter sync on B[m]. They REQUIRE row-affinity
                    // (each m's writes go to one rank: m % a) to avoid B_m
                    // write races. Sort each rank's bin by (k ASC, m ASC) so
                    // diagonals (type=1, m == k) for any given m come AFTER
                    // all off-diagonals (type=2, k < m) that contribute to it.
                    //
                    // The DENSE kernel uses a different model: it walks (k, m)
                    // positionally (rank, +STILES_SIZE, wrap on num_row_tiles)
                    // and POSITION-MATCHES tasks at each step. The initial
                    // striped binning above (solve_collect_forward per `corea`)
                    // already produces tasks in that walk order. Rebinning
                    // shuffles tasks out of the walk's order → the positional
                    // match fails → active tiles look inactive → wrong x.
                    //
                    // Gate the rebin on tile_type_mode: only semi (mode 1) gets
                    // the row-affinity rebin. Dense (mode 0) keeps the natural
                    // striped order from solve_collect_forward.
                    {
                        const int  _tile_type_mode = stiles_scheme_tile_mode(scheme_ptr);
                        if (_tile_type_mode == 1) {
                            auto rebin_by_row_and_sort = [&](
                                std::vector<std::array<int,6>>& tasks_inout,
                                std::vector<int>& offsets_inout)
                            {
                                std::vector<std::vector<std::array<int,6>>>
                                    by_owner(static_cast<std::size_t>(a));
                                for (const auto& t : tasks_inout) {
                                    const int m = t[2];
                                    if (m < 0) continue;
                                    const std::size_t owner = (a > 0)
                                        ? static_cast<std::size_t>(m % a)
                                        : 0;
                                    by_owner[owner].push_back(t);
                                }
                                for (auto& bin : by_owner) {
                                    std::stable_sort(bin.begin(), bin.end(),
                                        [](const std::array<int,6>& a_, const std::array<int,6>& b_) {
                                            if (a_[1] != b_[1]) return a_[1] < b_[1];   // k
                                            return a_[2] < b_[2];                        // m
                                        });
                                }
                                tasks_inout.clear();
                                offsets_inout.assign(static_cast<std::size_t>(a) + 1, 0);
                                for (int core = 0; core < a; ++core) {
                                    offsets_inout[static_cast<std::size_t>(core)] = static_cast<int>(tasks_inout.size());
                                    auto& bin = by_owner[static_cast<std::size_t>(core)];
                                    tasks_inout.insert(tasks_inout.end(), bin.begin(), bin.end());
                                }
                                offsets_inout[static_cast<std::size_t>(a)] = static_cast<int>(tasks_inout.size());
                            };
                            rebin_by_row_and_sort(fwd_tasks, fwd_offsets);
                            rebin_by_row_and_sort(bwd_tasks, bwd_offsets);
                        }
                    }

                    // ========== GPU SOLVE TASK COLLECTION ==========
                    // Per-stream bins with offsets for multi-stream GPU execution.
                    // Forward: round-robin across streams (same as CPU solve pattern).
                    // Backward: all tasks at step k go to stream k % num_streams
                    //           (serializes GEMMs that write to same B[k]).
                    {
                        const int num_streams = scheme_ptr->num_cores;

                        // --- Forward: collect all, then re-bin by m % num_streams ---
                        // All GEMMs writing to B[m] must be on the same stream to
                        // avoid write-write races (mirrors backward's k % num_streams
                        // pattern that serializes writes to B[k]).
                        {
                            std::vector<std::array<int,6>> all_fwd;
                            solve_collect_forward(state, method, 0, 1,
                                                  num_tiles, tile_size, N, map_id,
                                                  all_fwd);

                            std::vector<std::vector<std::array<int,6>>> gpu_fwd_bins(
                                static_cast<std::size_t>(num_streams));
                            for (auto& task : all_fwd) {
                                const int m_val = task[2];  // m is the write target in forward solve
                                const int stream = m_val % num_streams;
                                gpu_fwd_bins[static_cast<std::size_t>(stream)].push_back(task);
                            }

                            std::size_t fwd_total = 0;
                            for (const auto& bin : gpu_fwd_bins) fwd_total += bin.size();
                            auto& gpu_fwd = sTiles::ensure_gpu_solve_fwd_tasks(scheme_ptr);
                            auto& gpu_fwd_off = sTiles::ensure_gpu_solve_fwd_offsets(scheme_ptr);
                            gpu_fwd.clear();
                            gpu_fwd.reserve(fwd_total);
                            gpu_fwd_off.assign(static_cast<std::size_t>(num_streams) + 1, 0);
                            for (int s = 0; s < num_streams; ++s) {
                                gpu_fwd_off[static_cast<std::size_t>(s)] = static_cast<int>(gpu_fwd.size());
                                auto& bin = gpu_fwd_bins[static_cast<std::size_t>(s)];
                                if (!bin.empty()) {
                                    gpu_fwd.insert(gpu_fwd.end(),
                                        std::make_move_iterator(bin.begin()),
                                        std::make_move_iterator(bin.end()));
                                }
                            }
                            gpu_fwd_off[static_cast<std::size_t>(num_streams)] =
                                static_cast<int>(gpu_fwd.size());
                        }

                        // --- Backward: collect all, then re-bin by k % num_streams ---
                        {
                            std::vector<std::array<int,6>> all_bwd;
                            gpu_solve_collect_backward(state, num_tiles, tile_size, N, map_id, all_bwd);

                            std::vector<std::vector<std::array<int,6>>> gpu_bwd_bins(
                                static_cast<std::size_t>(num_streams));
                            for (auto& task : all_bwd) {
                                const int k = task[1];
                                const int stream = k % num_streams;
                                gpu_bwd_bins[static_cast<std::size_t>(stream)].push_back(task);
                            }

                            std::size_t bwd_total = 0;
                            for (const auto& bin : gpu_bwd_bins) bwd_total += bin.size();
                            auto& gpu_bwd = sTiles::ensure_gpu_solve_bwd_tasks(scheme_ptr);
                            auto& gpu_bwd_off = sTiles::ensure_gpu_solve_bwd_offsets(scheme_ptr);
                            gpu_bwd.clear();
                            gpu_bwd.reserve(bwd_total);
                            gpu_bwd_off.assign(static_cast<std::size_t>(num_streams) + 1, 0);
                            for (int s = 0; s < num_streams; ++s) {
                                gpu_bwd_off[static_cast<std::size_t>(s)] = static_cast<int>(gpu_bwd.size());
                                auto& bin = gpu_bwd_bins[static_cast<std::size_t>(s)];
                                if (!bin.empty()) {
                                    gpu_bwd.insert(gpu_bwd.end(),
                                        std::make_move_iterator(bin.begin()),
                                        std::make_move_iterator(bin.end()));
                                }
                            }
                            gpu_bwd_off[static_cast<std::size_t>(num_streams)] =
                                static_cast<int>(gpu_bwd.size());
                        }
                    }
                }

            if (!has_red_tree) {
                scheme_ptr->trees = NULL;
            }
        }
    
    
        // ========== RESCALE SCHEDULE ==========
        // When rescale_cores > 0, pre-compute a second task schedule for that
        // core count so the runtime can switch without recomputing.
        if (rescale_cores > 0) {
            const int r = rescale_cores;
            const bool r_has_red_tree = (scheme_ptr->red_tree_separator_level > 0);

            auto& sched = scheme_ptr->rescale_schedule;
            sched = sTiles::TaskSchedule{};
            sched.num_cores = r;

            // --- Cholesky / inverse tasks (disabled for rescale — only solve is rescaled) ---
            // if (collect_chol_inv) {
            //     std::vector<std::vector<std::array<int,7>>> r_chol_bins(static_cast<std::size_t>(r));
            //     std::vector<std::vector<std::array<int,7>>> r_inv_bins;
            //     if (scheme_ptr->compute_inverse) r_inv_bins.resize(static_cast<std::size_t>(r));
            //
            //     ProcessCore r_proc{state, method, r, num_tiles, group_index,
            //                        call_info, scheme_ptr, map_id, r_has_red_tree};
            //
            //     if (r > 1) {
            //         std::atomic<int> r_error{0};
            // #ifdef _OPENMP
            //         #pragma omp parallel num_threads(r)
            //         {
            // #else
            //         {
            // #endif
            // #ifdef _OPENMP
            //             #pragma omp for schedule(dynamic,1) nowait
            // #endif
            //             for (int corea = 0; corea < r; ++corea) {
            //                 if (r_error.load(std::memory_order_relaxed) != 0) continue;
            //                 auto& cb = r_chol_bins[static_cast<std::size_t>(corea)];
            //                 auto* ib = scheme_ptr->compute_inverse
            //                            ? &r_inv_bins[static_cast<std::size_t>(corea)] : nullptr;
            //                 sTiles::StatusCode ls = r_proc(corea, cb, ib);
            //                 if (ls != sTiles::StatusCode::Success)
            //                     r_error.store(static_cast<int>(ls), std::memory_order_relaxed);
            //             }
            //         }
            //         const int err = r_error.load(std::memory_order_relaxed);
            //         if (err != 0) return static_cast<sTiles::StatusCode>(err);
            //     } else {
            //         for (int corea = 0; corea < r; ++corea) {
            //             auto& cb = r_chol_bins[static_cast<std::size_t>(corea)];
            //             auto* ib = scheme_ptr->compute_inverse
            //                        ? &r_inv_bins[static_cast<std::size_t>(corea)] : nullptr;
            //             sTiles::StatusCode ls = r_proc(corea, cb, ib);
            //             if (ls != sTiles::StatusCode::Success) return ls;
            //         }
            //     }
            //
            //     // Combine chol bins
            //     sched.chol_tasks = std::make_shared<std::vector<std::array<int,7>>>();
            //     sched.chol_task_offsets = std::make_shared<std::vector<int>>();
            //     {
            //         std::size_t total = 0;
            //         for (const auto& bin : r_chol_bins) total += bin.size();
            //         sched.chol_tasks->reserve(total);
            //         sched.chol_task_offsets->assign(static_cast<std::size_t>(r) + 1, 0);
            //         for (int c = 0; c < r; ++c) {
            //             (*sched.chol_task_offsets)[static_cast<std::size_t>(c)] =
            //                 static_cast<int>(sched.chol_tasks->size());
            //             auto& bin = r_chol_bins[static_cast<std::size_t>(c)];
            //             if (!bin.empty())
            //                 sched.chol_tasks->insert(sched.chol_tasks->end(),
            //                     std::make_move_iterator(bin.begin()),
            //                     std::make_move_iterator(bin.end()));
            //         }
            //         (*sched.chol_task_offsets)[static_cast<std::size_t>(r)] =
            //             static_cast<int>(sched.chol_tasks->size());
            //     }
            //
            //     // Combine inv bins
            //     if (scheme_ptr->compute_inverse) {
            //         sched.inv_tasks = std::make_shared<std::vector<std::array<int,7>>>();
            //         sched.inv_task_offsets = std::make_shared<std::vector<int>>();
            //         std::size_t total = 0;
            //         for (const auto& bin : r_inv_bins) total += bin.size();
            //         sched.inv_tasks->reserve(total);
            //         sched.inv_task_offsets->assign(static_cast<std::size_t>(r) + 1, 0);
            //         for (int c = 0; c < r; ++c) {
            //             (*sched.inv_task_offsets)[static_cast<std::size_t>(c)] =
            //                 static_cast<int>(sched.inv_tasks->size());
            //             auto& bin = r_inv_bins[static_cast<std::size_t>(c)];
            //             if (!bin.empty())
            //                 sched.inv_tasks->insert(sched.inv_tasks->end(),
            //                     std::make_move_iterator(bin.begin()),
            //                     std::make_move_iterator(bin.end()));
            //         }
            //         (*sched.inv_task_offsets)[static_cast<std::size_t>(r)] =
            //             static_cast<int>(sched.inv_tasks->size());
            //     }
            // } // collect_chol_inv

            // --- Solve tasks (forward + backward) ---
            {
                const int tile_size = scheme_ptr->tile_size;
                const int N = scheme_ptr->dim;

                std::vector<std::vector<std::array<int,6>>> r_fwd_bins(static_cast<std::size_t>(r));
                std::vector<std::vector<std::array<int,6>>> r_bwd_bins(static_cast<std::size_t>(r));
                for (int corea = 0; corea < r; ++corea) {
                    solve_collect_forward(state, method, corea, r, num_tiles,
                                          tile_size, N, map_id,
                                          r_fwd_bins[static_cast<std::size_t>(corea)]);
                    solve_collect_backward(state, method, corea, r, num_tiles,
                                           tile_size, N, map_id,
                                           r_bwd_bins[static_cast<std::size_t>(corea)]);
                }

                // Forward
                sched.solve_fwd_tasks = std::make_shared<std::vector<std::array<int,6>>>();
                sched.solve_fwd_offsets = std::make_shared<std::vector<int>>();
                {
                    std::size_t total = 0;
                    for (const auto& bin : r_fwd_bins) total += bin.size();
                    sched.solve_fwd_tasks->reserve(total);
                    sched.solve_fwd_offsets->assign(static_cast<std::size_t>(r) + 1, 0);
                    for (int c = 0; c < r; ++c) {
                        (*sched.solve_fwd_offsets)[static_cast<std::size_t>(c)] =
                            static_cast<int>(sched.solve_fwd_tasks->size());
                        auto& bin = r_fwd_bins[static_cast<std::size_t>(c)];
                        if (!bin.empty())
                            sched.solve_fwd_tasks->insert(sched.solve_fwd_tasks->end(),
                                std::make_move_iterator(bin.begin()),
                                std::make_move_iterator(bin.end()));
                    }
                    (*sched.solve_fwd_offsets)[static_cast<std::size_t>(r)] =
                        static_cast<int>(sched.solve_fwd_tasks->size());
                }

                // Backward
                sched.solve_bwd_tasks = std::make_shared<std::vector<std::array<int,6>>>();
                sched.solve_bwd_offsets = std::make_shared<std::vector<int>>();
                {
                    std::size_t total = 0;
                    for (const auto& bin : r_bwd_bins) total += bin.size();
                    sched.solve_bwd_tasks->reserve(total);
                    sched.solve_bwd_offsets->assign(static_cast<std::size_t>(r) + 1, 0);
                    for (int c = 0; c < r; ++c) {
                        (*sched.solve_bwd_offsets)[static_cast<std::size_t>(c)] =
                            static_cast<int>(sched.solve_bwd_tasks->size());
                        auto& bin = r_bwd_bins[static_cast<std::size_t>(c)];
                        if (!bin.empty())
                            sched.solve_bwd_tasks->insert(sched.solve_bwd_tasks->end(),
                                std::make_move_iterator(bin.begin()),
                                std::make_move_iterator(bin.end()));
                    }
                    (*sched.solve_bwd_offsets)[static_cast<std::size_t>(r)] =
                        static_cast<int>(sched.solve_bwd_tasks->size());
                }

                // ========== ROW-AFFINITY REBIN (semi only) ==========
                // Mirror the primary solve schedule (see the mode==1 rebin
                // above): the semisparse solve kernel syncs on B[m] via update
                // counters and REQUIRES each m's writes to land on a single
                // rank (m % r). The striped binning above does NOT guarantee
                // that, so without this rebin the rescaled semi solve races on
                // B[m] once r >= 3 -> wrong x (dense walks positionally and
                // must keep the striped order, so it is left untouched).
                {
                    const int  _tile_type_mode = stiles_scheme_tile_mode(scheme_ptr);
                    if (_tile_type_mode == 1) {
                        auto rebin_by_row_and_sort = [&](
                            std::vector<std::array<int,6>>& tasks_inout,
                            std::vector<int>& offsets_inout)
                        {
                            std::vector<std::vector<std::array<int,6>>>
                                by_owner(static_cast<std::size_t>(r));
                            for (const auto& t : tasks_inout) {
                                const int m = t[2];
                                if (m < 0) continue;
                                const std::size_t owner = (r > 0)
                                    ? static_cast<std::size_t>(m % r)
                                    : 0;
                                by_owner[owner].push_back(t);
                            }
                            for (auto& bin : by_owner) {
                                std::stable_sort(bin.begin(), bin.end(),
                                    [](const std::array<int,6>& a_, const std::array<int,6>& b_) {
                                        if (a_[1] != b_[1]) return a_[1] < b_[1];   // k
                                        return a_[2] < b_[2];                        // m
                                    });
                            }
                            tasks_inout.clear();
                            offsets_inout.assign(static_cast<std::size_t>(r) + 1, 0);
                            for (int core = 0; core < r; ++core) {
                                offsets_inout[static_cast<std::size_t>(core)] = static_cast<int>(tasks_inout.size());
                                auto& bin = by_owner[static_cast<std::size_t>(core)];
                                tasks_inout.insert(tasks_inout.end(), bin.begin(), bin.end());
                            }
                            offsets_inout[static_cast<std::size_t>(r)] = static_cast<int>(tasks_inout.size());
                        };
                        rebin_by_row_and_sort(*sched.solve_fwd_tasks, *sched.solve_fwd_offsets);
                        rebin_by_row_and_sort(*sched.solve_bwd_tasks, *sched.solve_bwd_offsets);
                    }
                }
            }
        }







    } catch (const std::bad_alloc&) {
        return sTiles::StatusCode::OutOfResources;
    } catch (...) {
        return sTiles::StatusCode::Unexpected;
    }

    return sTiles::StatusCode::Success;
}

}}
