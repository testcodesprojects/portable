
#ifndef STILES_TREE_WRAPPERS_SAFEMODE_HPP
#define STILES_TREE_WRAPPERS_SAFEMODE_HPP

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/stiles_structs.hpp"
#include "../memory/TreeMemoryManager.hpp"
#include "../common/stiles_logger.hpp"
#include "tree.hpp"

extern "C" int* sTiles_get_params(void);

namespace sTiles{ namespace preprocess{


// Adaptive corner-probe: counts GEMMs in the bottom-right sep×sep block and
// sets scheme->red_tree_separator_level if the corner is heavy enough to
// warrant the tree-reduction path. Skips work if the caller already set sep>0
// (treated as a structural hint from arrowhead_thickness).
//
// Activation gates (calibrated from group1+group2 measurements):
//   - num_cores  >= 4   : tree's parallelism gain doesn't beat overhead with
//                         only 2 cores per rank-slice. Empirically cores=2/3
//                         shows mixed results; cores=1 short-circuits anyway.
//   - num_tiles  >  50  : tiny matrices factorize in microseconds — tree's
//                         per-slot scratch+reduce overhead dominates. Group1
//                         (bcsstk09/10/14/15) shows up to −100% slowdown.
//   - aggregate-work    : Σ(heavy slot GEMM counts) >= 8 · num_cores².
//                         Crude break-even estimate: each tree slot has ~20µs
//                         overhead (scratch alloc + barrier + reduce), each
//                         GEMM is ~1µs at tile_size=40, parallel saving per
//                         GEMM is ~(cores-1)/cores. So total GEMMs must
//                         exceed ~(overhead_per_slot · num_slots · cores /
//                         ((cores-1)·gemm_time)). The constant 8·c² is a
//                         single-knob proxy that matches the data well.
inline sTiles::StatusCode corner_probe(sTiles_call **call_info,
                                       TiledMatrix **scheme,
                                       int /*group_index*/) {
    TiledMatrix* sc = *scheme;
    const int N = sc->dimTiledMatrix;
    if (N <= 1) return sTiles::StatusCode::Success;

    // STILES_FLAT_SCHED: force standard expansion (no tree) for the confound-free
    // scheduling comparison — leave red_tree_separator_level at 0.
    { const char* e = std::getenv("STILES_FLAT_SCHED"); if (e && e[0] == '1') return sTiles::StatusCode::Success; }

    // Respect user-provided arrowhead hint — if sep is already > 0, the caller
    // declared structure; just leave it.
    if (sc->red_tree_separator_level > 0) return sTiles::StatusCode::Success;

    const int num_cores = (*call_info)->num_cores;

    // Gate 1: minimum core count. cores < 2 has no parallelism to exploit;
    // cores = 2 is allowed only for arrowhead-narrow patterns (see Gate 5
    // below — heavy_slots cap is tightened to 2 when cores < 4).
    // Empirical: at cores=2 with a 4·cores² heavy_slots cap, mid-sep matrices
    // slowed down (diff −35%, 83o4NNNo −9%, ferris −9%). Tightening to
    // heavy_slots ≤ 2 captures the arrowhead wins (sem_n5000 +16%, sem_n2000
    // +12%, 8rtKSK +16%, bcsstk13 +17%, bcsstk15 +11%) without those losses.
    constexpr int kMinCores = 2;
    if (num_cores < kMinCores) {
        sTiles::Logger::timing_always("│   ↪ Corner-probe skipped: num_cores=", num_cores,
            " < ", kMinCores, " (not enough parallelism)");
        return sTiles::StatusCode::Success;
    }

    // Gate 2: minimum tile count. Tiny matrices have fixed-cost overhead
    // (per-tile workspace, scatter info, tree-task emission) that exceeds
    // any parallelism gain at this scale.
    constexpr int kMinNumTiles = 50;
    if (N < kMinNumTiles) {
        sTiles::Logger::timing_always("│   ↪ Corner-probe skipped: num_tiles=", N,
            " < ", kMinNumTiles, " (matrix too small)");
        return sTiles::StatusCode::Success;
    }

    const int probe_sep   = std::min({N - 1, std::max(1, N / 10), 32});
    const int activate_if = 2 * num_cores;

    auto is_on = [sc, N](int a, int b) {
        if (a > b) std::swap(a, b);
        if (sc->state.is_active) return sc->state.isActive(a, b, N);
        if (sc->mapper.valid())  return sc->mapper.map_ij(a, b, N) >= 0;
        if (sc->permutation_flags) {
            const int tri = a * (2 * N - a - 1) / 2 + b;
            return sc->permutation_flags[tri];
        }
        return false;
    };

    const int num_sep = probe_sep * (probe_sep + 1) / 2;
    std::vector<int> ctr(num_sep, 0);
    int* ctr_ptr = ctr.data();
    sTiles::Tree::count_gemms(&ctr_ptr, is_on, N, probe_sep);

    int peak = 0;
    long long sum = 0;
    long long heavy_sum = 0;
    int nonzero_slots = 0;
    int heavy_slots = 0;
    // chosen_sep = smallest sep that still contains every "heavy" slot.
    // A heavy slot at local (kk, mm) sits at absolute tile-row (N - probe_sep + kk),
    // so a separator of size s contains it iff s >= probe_sep - kk. We need the
    // max of those across all heavy slots, which is determined by the smallest kk
    // (= the topmost heavy row in the probed block).
    int min_heavy_kk = probe_sep;  // sentinel: "no heavy row found yet"
    for (int kk = 0; kk < probe_sep; ++kk) {
        for (int mm = kk; mm < probe_sep; ++mm) {
            const int idx = kk * (2*probe_sep - kk - 1) / 2 + mm;
            const int v = ctr[idx];
            peak = std::max(peak, v);
            sum += v;
            if (v > 0) ++nonzero_slots;
            if (v > activate_if) {
                ++heavy_slots;
                heavy_sum += v;
                if (kk < min_heavy_kk) min_heavy_kk = kk;
            }
        }
    }
    const double mean = num_sep > 0 ? double(sum) / double(num_sep) : 0.0;
    const int chosen_sep = (min_heavy_kk < probe_sep)
                           ? (probe_sep - min_heavy_kk)  // tight fit around heavy slots
                           : 0;                          // no heavy slots — no tree

    // Option A (composed ND + tree): when the SCOTCH ND-partition collection is
    // active, the tree must reduce EXACTLY the root separator corner so that
    // Region B = [N - root_sep_tiles, N) and Region A = the non-root partitions
    // tile [0, cutoff) with no gap/overlap. So the activation separator is the
    // STRUCTURAL root_sep_tiles, not the geometric chosen_sep. The heavy/
    // aggregate gates below still decide WHETHER to activate (using the corner
    // GEMM scan as the "enough accumulation" test); we only swap which sep we
    // activate with. If root_sep_tiles is too wide (> max_sep) the tree is
    // rejected and the root separator stays a normal ND partition (pure ND).
    const bool composed_nd = (sc->scotch_partition_collection && sc->scotch_root_sep_tiles > 0);
    const int  activ_sep   = composed_nd ? sc->scotch_root_sep_tiles : chosen_sep;

    // Gate 3: aggregate-work check. The amortization threshold scales with
    // num_cores² — more cores means more per-slot overhead AND less work
    // per rank-slice, so we need quadratic-in-cores total GEMMs to break even.
    const long long aggregate_threshold =
        8LL * static_cast<long long>(num_cores) * static_cast<long long>(num_cores);
    const bool aggregate_ok = (heavy_sum >= aggregate_threshold);

    // Gate 4 (mode + backend aware): cap chosen_sep per (backend, mode).
    // After the case-7/10 memset relocation (memsets moved out of the
    // reduce path into first-use beta=0 / first-use memset), both backends
    // show the same crossover behavior on semisparse group2 at 8 cores:
    //   - sep ≤ 5  : win (sem_n5000, ayaLRw, 8rtKSK, 83o4NNNo, net814381)
    //   - sep ≥ 9  : loss (diff −14/−17%, ferris −19/−24% on pthreads)
    // Dense (mode 0) parallel worker is already well-parallelized so tree
    // only helps for true arrowhead (sep ≤ 4).
    const int* probe_params = sTiles_get_params();
    const int tile_type_mode = (probe_params ? probe_params[sTiles::param::TileTypeMode] : 0);
    const int use_omp        = (probe_params ? probe_params[sTiles::param::UseOMP] : 0);
    (void)use_omp;
    constexpr int kMode0MaxSep   = 4;   // dense always conservative
    constexpr int kModeSemiMaxSep = 5;  // semisparse (both backends): cap at 5
    int max_sep = (tile_type_mode == 0) ? kMode0MaxSep : kModeSemiMaxSep;
    // Composed ND path: the leaves are factored in parallel, so the tree's job
    // is to attack the otherwise-serial ROOT SEPARATOR — wide heavy separators
    // (big FEM: 20-30 tiles, heavy_sum >> threshold) are exactly where it should
    // help. The <=5 cap was measured on the OLD flat-Region-A model (sep>=9 lost
    // there) and does not apply once the leaves are parallel. Lift it for the
    // composed path (default effectively uncapped); env-tunable for the sweep.
    if (composed_nd) {
        max_sep = (1 << 20);
        if (const char* e = std::getenv("STILES_COMPOSED_MAX_SEP")) {
            const int v = std::atoi(e); if (v > 0) max_sep = v;
        }
    }
    const bool mode_ok = (activ_sep <= max_sep);

    // Gate 5 (heavy-slot cap): one tree per heavy slot means per-tree fixed
    // overhead (scratch alloc + cross-rank barrier + DGEADD reduce) scales
    // linearly with heavy_slots. When the corner is dense (528 slots all
    // heavy) and core count is modest, per-slot overhead × heavy_slots can
    // exceed parallel savings even when total work is large.
    //   - cores ≥ 4: cap at 4·cores² (filters spacetime/ferris dense-corner
    //                slowdowns while keeping real wins).
    //   - cores < 4: cap at 2 (arrowhead-only). At 2 cores the parallel
    //                saving is only 2×, so we can only afford 1-2 trees
    //                worth of fixed cost. This captures the arrowhead wins
    //                (sem_n5000, 8rtKSK, etc.) without the mid-sep losses.
    int max_heavy_slots = (num_cores >= 4)
        ? 4 * num_cores * num_cores
        : 2;
    // Composed ND path: the slot-count cap was a flat-model profitability guard
    // (per-tree overhead × heavy_slots). In the composed model the dense root
    // separator IS the serial bottleneck after the leaves go parallel, so a
    // dense corner (all slots heavy) is exactly where the tree should reduce.
    // Lift the cap for composed (env-tunable); the aggregate "enough GEMM
    // accumulation" gate (heavy_sum >= 8·cores²) still governs activation.
    if (composed_nd) {
        max_heavy_slots = (1 << 30);
        if (const char* e = std::getenv("STILES_COMPOSED_MAX_HEAVY_SLOTS")) {
            const int v = std::atoi(e); if (v > 0) max_heavy_slots = v;
        }
    }
    const bool slot_count_ok = (heavy_slots <= max_heavy_slots);

    // Gate 6 (composed width FLOOR): for the composed ND+tree path, a NARROW
    // root separator is not worth tree-reducing — the leaves already
    // parallelize (Region A), and a small root just adds reduce overhead while
    // the worldsize_b cap shrinks it toward pure-ND anyway. Decline composed
    // trees with root_sep < W. Default 0 (disabled) until W is calibrated from
    // the wide-sep giants' root_sep; env STILES_COMPOSED_MIN_SEP sets W. This
    // is the inverse of the old flat-model ≤5 *cap* — here we want WIDE roots.
    int composed_min_sep = 0;
    if (composed_nd) {
        if (const char* e = std::getenv("STILES_COMPOSED_MIN_SEP")) { const int v = std::atoi(e); if (v > 0) composed_min_sep = v; }
    }
    const bool composed_floor_ok = (activ_sep >= composed_min_sep);

    // Effective activation decision = all gates pass:
    //   - chosen_sep > 0    : at least one slot crossed per-slot threshold
    //   - aggregate_ok      : total heavy GEMMs >= 8·cores²
    //   - mode_ok           : dense mode rejects chosen_sep > 4
    //   - slot_count_ok     : heavy_slots <= 4·cores² (too many trees → overhead)
    //   - composed_floor_ok : composed root_sep >= W (narrow root not worth it)
    const bool work_qualifies = (activ_sep > 0) && aggregate_ok && mode_ok && slot_count_ok && composed_floor_ok;

    // Single diagnostic line carries the decision and the reason it was rejected.
    const char* decision_label =
        work_qualifies               ? "would ACTIVATE" :
        (chosen_sep == 0)            ? "off (no heavy slot)" :
        !aggregate_ok                ? "off (work below aggregate threshold)" :
        !mode_ok                     ? "off (chosen_sep > backend cap)" :
        !slot_count_ok               ? "off (too many heavy slots)" :
        !composed_floor_ok           ? "off (composed root_sep below floor)" :
                                       "off";

    char mean_buf[32];
    std::snprintf(mean_buf, sizeof(mean_buf), "%.1f", mean);
    sTiles::Logger::timing_always("│   ↪ Corner-probe: N=", N, " mode=", tile_type_mode,
        " probe_sep=", probe_sep, " slots=", num_sep, " nonzero=", nonzero_slots,
        " peak=", peak, " mean=", mean_buf, " heavy=", heavy_slots, "/", max_heavy_slots,
        " heavy_sum=", heavy_sum, " chosen_sep=", chosen_sep,
        " agg_thr=", aggregate_threshold, " -> ", decision_label);

    // Test-only bypass: STILES_CORNER_PROBE_FORCE=1 (env var) OR
    // sTiles_set_tree_path_force(1) (param slot 26) skips ALL six gates and
    // activates the tree path unconditionally. Used by kernel-correctness
    // tests to guarantee the reduction code path actually executes (the
    // normal gate would silently fall back to the standard expansion path
    // on matrices it deems unprofitable, hiding any tree-kernel bugs).
    // When chosen_sep would have been 0 (no heavy slot), we use probe_sep
    // so the tree spans the full scanned corner.
    {
        const bool param_force = (probe_params && probe_params[sTiles::param::TreePathForce] != 0);
        const char* env_force = std::getenv("STILES_CORNER_PROBE_FORCE");
        const bool env_force_enabled = (env_force && env_force[0] == '1');
        const bool force_enabled = param_force || env_force_enabled;
        if (force_enabled) {
            // Composed ND path forces the STRUCTURAL root-sep so Region A/B stay
            // aligned; otherwise force the geometric corner (or full probe).
            int forced_sep = composed_nd ? sc->scotch_root_sep_tiles
                                         : ((chosen_sep > 0) ? chosen_sep : probe_sep);
            if (forced_sep <= 0) forced_sep = 1;
            // Safety clamp: force bypasses the PROFITABILITY gates (aggregate work,
            // slot count) so the reduction kernel runs even where the gate would
            // decline it — but it must NOT bypass the kernel's validated separator
            // WIDTH. The reduction setup/scatter (create_trees + case 5/7/8/10) is
            // only correct for a narrow separator; a wider one drives the reduce
            // loops past their per-tree scratch allocations and segfaults (e.g.
            // pedigree, chosen_sep=17, crashes in the DGEADD-reduce). max_sep is the
            // same bound the gate uses (4 dense / 5 semi; effectively uncapped for
            // the composed ND path, whose structural root-sep is valid at any
            // width). Clamp so forcing on a gate-rejected wide-sep matrix degrades
            // to a safe capped tree instead of crashing.
            if (forced_sep > max_sep) {
                sTiles::Logger::timing_always("│   ↪ Corner-probe force: clamping sep ",
                    forced_sep, " → ", max_sep, " (kernel safe-width bound)");
                forced_sep = max_sep;
            }
            sc->red_tree_separator_level = forced_sep;
            (*call_info)->red_tree_separator_level = forced_sep;
            sTiles::Logger::timing_always("│   ↪ Corner-probe forced (",
                param_force ? "param" : "env", "): sep=", forced_sep,
                " (gate decision was: ", decision_label, ")");
            return sTiles::StatusCode::Success;
        }
    }

    // Activation gate: global param (slot 25 = TREE_PATH_ENABLE) OR
    // env var STILES_CORNER_PROBE_ACTIVATE=1 as a fallback for shell-driven testing.
    // Default is OFF — callers opt in via sTiles_set_tree_path_enable(1).
    if (work_qualifies) {
        const bool param_enabled = (probe_params && probe_params[sTiles::param::TreePathEnable] != 0);
        const char* env = std::getenv("STILES_CORNER_PROBE_ACTIVATE");
        const bool env_enabled = (env && env[0] == '1');
        if (param_enabled || env_enabled) {
            sc->red_tree_separator_level = activ_sep;
            (*call_info)->red_tree_separator_level = activ_sep;
            sTiles::Logger::timing_always("│   ↪ Corner-probe activated (",
                param_enabled ? "param" : "env", "): sep=", activ_sep,
                composed_nd ? " [composed: root-sep]" : "");
        }
    }
    return sTiles::StatusCode::Success;
}


inline sTiles::StatusCode leaves_counter(sTiles_call **call_info, TiledMatrix **scheme, int group_index) {
    // This function is only called for sparse variants (0 or 3) now

    if ((*scheme)->red_tree_separator_level > 0) {
        sTiles::Logger::info("│     • Tree reduction is activated.");
        sTiles::Logger::info("│     • Number of separators is " + std::to_string((*scheme)->red_tree_separator_level) + ".");

        int new_sep = (*scheme)->red_tree_separator_level;
        if ((*call_info)->parameters[static_cast<int>(sTiles::Parameter::CoresSplitStrategy)] == 1) {
            sTiles::Logger::debug("│     • Skipping TREE_SETUP_STG1 (stubbed).");
        } else {
            const int N = (*scheme)->dimTiledMatrix;
            const int sep = (*scheme)->red_tree_separator_level;
            const int num_sep = sep * (sep + 1) / 2;

            if ((*scheme)->tree_counter == nullptr) {
                (*scheme)->tree_counter = TreeMemoryManager::allocateZero<int>(num_sep, group_index);
            } else {
                std::fill((*scheme)->tree_counter, (*scheme)->tree_counter + num_sep, 0);
            }

            auto is_on = [scheme = *scheme, N](int a, int b) {
                if (a > b) std::swap(a, b);
                if (scheme->state.is_active) return scheme->state.isActive(a, b, N);
                if (scheme->mapper.valid())  return scheme->mapper.map_ij(a, b, N) >= 0;
                // Fallback to safe flags if fast state/mapper are unavailable
                if (scheme->permutation_flags) {
                    const int tri = a * (2 * N - a - 1) / 2 + b;
                    return scheme->permutation_flags[tri];
                }
                return false;
            };
            sTiles::Tree::count_gemms(&(*scheme)->tree_counter, is_on, N, (*scheme)->red_tree_separator_level);
            new_sep = (*scheme)->red_tree_separator_level;
        }
        (void)new_sep;  // reserved for future use
    } else {
        (*scheme)->red_tree_separator_level = 0;
    }

    return sTiles::StatusCode::Success;
}

inline sTiles::StatusCode tree_creation(sTiles_call **call_info, TiledMatrix **scheme, int group_index) {
    // This function is only called for sparse variants (0 or 3) now

    if ((*scheme)->red_tree_separator_level > 0) {
        sTiles::Logger::info("│     • Tree reduction is activated.");
        sTiles::Logger::info("│     • Number of separators is " + std::to_string((*scheme)->red_tree_separator_level) + ".");

        int new_sep = (*scheme)->red_tree_separator_level;
        if ((*call_info)->parameters[static_cast<int>(sTiles::Parameter::CoresSplitStrategy)] == 1) {
            // new_sep = TREE_SETUP_STG1(STILES_CORES_PER_CHOLESKY_VECTOR, schemes, config);
            sTiles::Logger::debug("│     • Skipping TREE_SETUP_STG1 (stubbed).");
        } else {
            const int N = (*scheme)->dimTiledMatrix;
            new_sep = sTiles::Tree::create_trees(N,
                                                (*scheme)->tile_size,
                                                &(*scheme)->red_tree_separator_level,
                                                &(*scheme)->tree_counter,
                                                (*call_info)->parameters[static_cast<int>(sTiles::Parameter::TreeReductionAcc)],
                                                &(*scheme)->trees,
                                                (*scheme)->num_cores,
                                                group_index);
        }
        (void)new_sep;  // reserved for future use
    } else {
        (*scheme)->red_tree_separator_level = 0;
    }

    return sTiles::StatusCode::Success;
}

}}

#endif  // STILES_TREE_WRAPPERS_HPP
