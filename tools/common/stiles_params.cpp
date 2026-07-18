/**
 * @file stiles_params.cpp
 * @brief C-ABI setters/getters for the sTiles control-parameter array.
 *
 * Owns every `sTiles_set_*` / `sTiles_get_*` entry point that wraps
 * `stiles_control_params[N]`. The live array itself stays in
 * tools/process/process.cpp because preprocessing code reads it directly in
 * many places; we reach it here via `extern`.
 *
 * Two arrays exist (both defined in process.cpp, both initialized from
 * STILES_DEFAULT_PARAMS_INIT):
 *   - stiles_control_params[] : the LIVE array everything reads. Slots 1/3/4/5
 *     get auto-RESOLVED values written into them during preprocessing.
 *   - stiles_user_params[]    : what the user configured. Written only here
 *     (setters + resets). sTiles_preprocess_group restores the resolvable
 *     slots from it before each group, so "auto" is re-evaluated per group
 *     instead of sticking to the first group's resolution.
 *
 * See stiles_params.hpp for the named slot indices, per-slot documentation,
 * and the immutable defaults (`sTiles::kDefaultParams`).
 */

#include "stiles_params.hpp"
#include "stiles_logger.hpp"
#ifdef STILES_GPU
#include "../gpu/gpu_dispatch_plan.hpp"   // gh200_unified::{set,enabled}
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Live storage + user-config shadow (defined in tools/process/process.cpp).
extern bool expert_mode_enabled;
extern int  stiles_control_params[STILES_NUM_PARAMS];
extern int  stiles_user_params[STILES_NUM_PARAMS];

// Forward declaration: defined in tools/ordering/ordering_nd.cpp.
// Selects METIS_NodeNDP (use_ndp=1) vs METIS_NodeND (use_ndp=0).
extern "C" void stiles_set_nd_use_ndp(int v);

namespace {

// Every user-facing write goes through here so the live array and the
// user-config shadow stay in sync. Preprocessing writes RESOLVED values to
// the live array directly and never calls this.
inline void set_param_both(int index, int value) {
    stiles_control_params[index] = value;
    stiles_user_params[index]    = value;
}

inline bool expert_gate(const char* fn) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning(fn, ": call sTiles_expert_user() first to enable configuration");
        return false;
    }
    return true;
}

// Shared digit validation for the digit-list parameters (slot 4).
// Returns true and stores iff `value` is 0 or consists of digits 1..9 only.
inline bool set_digit_list_param(const char* fn, int index, int value) {
    if (value < 0) {
        sTiles::Logger::warning(fn, " ignored negative value ", value);
        return false;
    }
    int digits = value;
    while (digits > 0) {
        const int d = digits % 10;
        if (d <= 0 || d > 9) {
            sTiles::Logger::warning(fn, " expects digits 1..9 only; got ", value);
            return false;
        }
        digits /= 10;
    }
    set_param_both(index, value);
    return true;
}

} // namespace

extern "C" {

// Forward declarations so the dispatcher below can call typed setters.
void sTiles_set_correction_mode(int value);
void sTiles_set_tile_size(int tile_size);
void sTiles_set_ordering_mode(int value);
void sTiles_set_tile_type_mode(int value);
void sTiles_set_tile_ordering_mode(int value);
void sTiles_set_tile_ordering_size(int value);
void sTiles_force_ND(int enable);
void sTiles_set_inverse_storage_mode(int mode);
void sTiles_set_use_omp(int mode);
void sTiles_set_semisparse_impl(int impl);
void sTiles_set_memory_estimate(int enable);
void sTiles_set_tile_ordering_min_dim(int value);
void sTiles_set_serial_mode(int mode);
void sTiles_set_bw_mode(int mode);
void sTiles_set_tile_first_ordering_mode(int value);
void sTiles_set_force_scotch_ordering(int on);
void sTiles_set_scotch_padding(int on);
void sTiles_set_path2_depth(int depth);
void sTiles_set_tree_path_enable(int on);
void sTiles_set_tree_path_force(int on);
void sTiles_use_gpu(int enable);

void sTiles_expert_user() {
    expert_mode_enabled = true;
    sTiles::Logger::info("Expert mode enabled - configuration functions are now available");
}

void sTiles_set_tile_size(int tile_size) {
    if (!expert_gate("sTiles_set_tile_size")) return;
    if (tile_size > 0 || tile_size == -1) {
        set_param_both(sTiles::param::UserTileSize, tile_size);
    }
}

int sTiles_return_tile_size() {
    return stiles_control_params[sTiles::param::UserTileSize];
}

void sTiles_set_bw_mode(int mode) {
    if (!expert_gate("sTiles_set_bw_mode")) return;
    set_param_both(sTiles::param::BandwidthMode, (mode != 0) ? 1 : 0);
}

int sTiles_get_bw_mode() {
    return stiles_control_params[sTiles::param::BandwidthMode];
}

int* sTiles_get_params() {
    return stiles_control_params;
}

// Slot 25: TREE_PATH_ENABLE — global enable for the tree-reduction path
// triggered by corner_probe. 0 = disabled, 1 = enabled (default; the
// 6-gate predicate still decides per matrix). Env var
// STILES_CORNER_PROBE_ACTIVATE=1 remains supported as an override for
// shell-driven testing.
void sTiles_set_tree_path_enable(int on) {
    if (!expert_gate("sTiles_set_tree_path_enable")) return;
    set_param_both(sTiles::param::TreePathEnable, (on != 0) ? 1 : 0);
}

int sTiles_get_tree_path_enable() {
    return stiles_control_params[sTiles::param::TreePathEnable];
}

// Slot 26: TREE_PATH_FORCE — bypass the 6-gate predicate and activate the
// tree-reduction path unconditionally. Used by kernel-correctness tests to
// guarantee the reduction code path executes on matrices where the gate
// would otherwise reject. 0 = honor gate (default), 1 = force activation.
// Env var STILES_CORNER_PROBE_FORCE=1 is also honored (either trigger activates).
void sTiles_set_tree_path_force(int on) {
    if (!expert_gate("sTiles_set_tree_path_force")) return;
    set_param_both(sTiles::param::TreePathForce, (on != 0) ? 1 : 0);
}

int sTiles_get_tree_path_force() {
    return stiles_control_params[sTiles::param::TreePathForce];
}

void sTiles_set_params(const int* params, int n) {
    if (!expert_gate("sTiles_set_params")) return;
    if (n > STILES_NUM_PARAMS) n = STILES_NUM_PARAMS;
    for (int i = 0; i < n; ++i) {
        set_param_both(i, params[i]);
    }
}

void sTiles_get_all_params(int* params, int* size) {
    for (int i = 0; i < STILES_NUM_PARAMS; ++i) {
        params[i] = stiles_control_params[i];
    }
    if (size) *size = STILES_NUM_PARAMS;
}

} // extern "C"

static const char* stiles_value_name(int index, int value) {
    switch (index) {
        case  0: switch (value) {
                     case 0: return "STILES_CORRECTION_NONE";
                     case 1: return "STILES_CORRECTION_SINGLE";
                     case 2: return "STILES_CORRECTION_TILES";
                     case 3: return "STILES_CORRECTION_COLUMNS";
                 } break;
        case  1: if (value == -1) return "STILES_TILE_AUTO";
                 break;
        case  3: switch (value) {
                     case 0: return "STILES_TILE_DENSE";
                     case 1: return "STILES_TILE_SEMISPARSE";
                     case 2: return "STILES_TILE_SPARSE";
                     case 3: return "STILES_TILE_AUTO_SELECT";
                 } break;
        case  5: if (value == -1) return "auto: tile_size / 2";
                 break;
        case  6: switch (value) {
                     case 0: return "STILES_ND_AUTO";
                     case 1: return "STILES_ND_FORCE";
                 } break;
        case  7: switch (value) {
                     case 0: return "STILES_INV_OVERWRITE";
                     case 1: return "STILES_INV_SEPARATE";
                 } break;
        case  8: switch (value) {
                     case 0: return "STILES_THREAD_PTHREADS";
                     case 1: return "STILES_THREAD_OMP";
                 } break;
        case  9: switch (value) {
                     case 0: return "STILES_SEMI_ORIGINAL";
                     case 1: return "STILES_SEMI_IMPROVED";
                     case 2: return "STILES_SEMI_VECTORIZED";
                     case 3: return "STILES_SEMI_SERIAL_SPARSE";
                 } break;
        case 10: switch (value) {
                     case 0: return "STILES_GPU_ONLY";
                     case 1: return "STILES_GPU_WITH_CPU";
                 } break;
        case 11: switch (value) {
                     case 0: return "STILES_GPU_DISABLED";
                     case 1: return "STILES_GPU_ENABLED";
                 } break;
        case 12: switch (value) {
                     case 0: return "STILES_MEMEST_SKIP";
                     case 1: return "STILES_MEMEST_PRINT";
                 } break;
        case 14: switch (value) {
                     case 0: return "STILES_SERIAL_AUTO";
                     case 1: return "STILES_SERIAL_ALWAYS_PARALLEL";
                 } break;
        case 15: switch (value) {
                     case 0: return "STILES_BW_CONSERVATIVE";
                     case 1: return "STILES_BW_TIGHT";
                 } break;
    }
    return nullptr;
}

extern "C" {

void sTiles_print_params() {
    static const char* names[sTiles::kNumDefinedParams] = {
        "CORRECTION_MODE", "TILE_SIZE", "ORDERING_MODE", "TILE_TYPE",
        "TILE_ORDERING_MODE", "TILE_ORDERING_SIZE", "FORCE_ND",
        "INVERSE_STORAGE", "USE_OMP", "SEMISPARSE_IMPL",
        "GPU_COMPARE", "GPU_ENABLE", "MEMORY_ESTIMATE",
        "RESERVED_13", "SERIAL_MODE", "BW_MODE",
        "RESERVED_16", "RESERVED_17", "RESERVED_18",
        "RESERVED_19", "FORCE_SCOTCH_ORDERING", "SCOTCH_PADDING",
        "PATH2_DEPTH", "RESERVED_23", "RESERVED_24",
        "TREE_PATH_ENABLE", "TREE_PATH_FORCE"
    };

    std::fprintf(stdout,
        "sTiles Global Parameters\n"
        "──────────────────────────────────────────────────────────────────\n");
    // Iterate the named slots; reserved 27..49 don't need to print.
    for (int i = 0; i < sTiles::kNumDefinedParams; ++i) {
        int val = stiles_control_params[i];
        const char* vname = stiles_value_name(i, val);
        if (vname) {
            std::fprintf(stdout, " [%2d] %-23s = %-6d  (%s)\n",
                         i, names[i], val, vname);
        } else {
            std::fprintf(stdout, " [%2d] %-23s = %d\n",
                         i, names[i], val);
        }
    }
    std::fprintf(stdout,
        "──────────────────────────────────────────────────────────────────\n");
}

void sTiles_set_correction_mode(int value) {
    if (!expert_gate("sTiles_set_correction_mode")) return;
    if (value < 0 || value > 3) {
        sTiles::Logger::warning("sTiles_set_correction_mode: invalid value=", value,
                                " (must be 0..3)");
        return;
    }
    set_param_both(sTiles::param::SemisparsePruningMode, value);
}

// Ordering candidate list for the bake-off. Digits in the PUBLIC numbering
// (STILES_ORD_* in stiles.h): 1=RCM, 2=METIS, 3=SCOTCH, 4=ASCOTCH,
// 5=FSCOTCH, 6=AMD, 7=CAMD, 8=COLAMD. E.g. 167 = evaluate RCM, AMD, CAMD
// and pick the best by fill; a single digit pins one ordering.
// 0 (default) = adaptive selection by matrix size/class.
void sTiles_set_ordering_mode(int value) {
    if (!expert_gate("sTiles_set_ordering_mode")) return;
    if (value < 0) {
        sTiles::Logger::warning("sTiles_set_ordering_mode ignored negative value ", value);
        return;
    }
    int digits = value;
    while (digits > 0) {
        const int d = digits % 10;
        if (d < 1 || d > 8) {
            sTiles::Logger::warning("sTiles_set_ordering_mode expects digits 1..8 "
                                    "(1=RCM 2=METIS 3=SCOTCH 4=ASCOTCH 5=FSCOTCH "
                                    "6=AMD 7=CAMD 8=COLAMD); got ", value);
            return;
        }
        digits /= 10;
    }
    set_param_both(sTiles::param::OrderingMode, value);
}

void sTiles_set_tile_ordering_mode(int value) {
    if (!expert_gate("sTiles_set_tile_ordering_mode")) return;
    set_digit_list_param("sTiles_set_tile_ordering_mode",
                         sTiles::param::TileOrderingStrategy, value);
}

// DEPRECATED. Slot 16 (tile-first ordering) is not consumed anywhere; the
// value is stored but has no effect. Kept for ABI compatibility.
void sTiles_set_tile_first_ordering_mode(int value) {
    if (!expert_gate("sTiles_set_tile_first_ordering_mode")) return;
    sTiles::Logger::warning("sTiles_set_tile_first_ordering_mode is deprecated and has no effect.");
    set_param_both(16, value);
}

void sTiles_set_tile_ordering_size(int value) {
    if (!expert_gate("sTiles_set_tile_ordering_size")) return;
    if (value < -1) {
        sTiles::Logger::warning("sTiles_set_tile_ordering_size ignored negative value ", value);
        return;
    }
    set_param_both(sTiles::param::TileOrderingThreshold, value);
}

// DEPRECATED. Slot 13 (tile-ordering min dim) is not consumed anywhere; the
// value is stored but has no effect. Kept for ABI compatibility.
void sTiles_set_tile_ordering_min_dim(int value) {
    if (!expert_gate("sTiles_set_tile_ordering_min_dim")) return;
    sTiles::Logger::warning("sTiles_set_tile_ordering_min_dim is deprecated and has no effect.");
    if (value < 0) return;
    set_param_both(13, value);
}

void sTiles_set_tile_type_mode(int value) {
    if (!expert_gate("sTiles_set_tile_type_mode")) return;
    // 0 = dense, 1 = semisparse, 2 = non-uniform tiles, 3 = auto (pick based
    // on occupancy / fill / degree skew after symbolic factorization).
    if (value == 0 || value == 1 || value == 2 || value == 3) {
        set_param_both(sTiles::param::TileTypeMode, value);
    } else {
        sTiles::Logger::warning("sTiles_set_tile_type_mode: invalid value=", value,
                                " (must be 0=dense, 1=semisparse, 2=non-uniform, or 3=auto)");
    }
}

void sTiles_force_ND(int enable) {
    if (!expert_gate("sTiles_force_ND")) return;
    stiles_set_nd_use_ndp(enable);
    set_param_both(sTiles::param::ForceNDOrdering, (enable != 0) ? 1 : 0);
}

void sTiles_set_inverse_storage_mode(int mode) {
    if (!expert_gate("sTiles_set_inverse_storage_mode")) return;
    set_param_both(sTiles::param::InverseStorageMode, (mode == 0 || mode == 1) ? mode : 1);
}

int sTiles_get_inverse_storage_mode() {
    return stiles_control_params[sTiles::param::InverseStorageMode];
}

void sTiles_set_use_omp(int mode) {
    if (!expert_gate("sTiles_set_use_omp")) return;
    set_param_both(sTiles::param::UseOMP, (mode != 0) ? 1 : 0);
}

int sTiles_get_use_omp() {
    return stiles_control_params[sTiles::param::UseOMP];
}

void sTiles_set_semisparse_impl(int impl) {
    if (!expert_gate("sTiles_set_semisparse_impl")) return;
    set_param_both(sTiles::param::SemisparseImpl, (impl >= 0 && impl <= 3) ? impl : 2);
}

int sTiles_get_semisparse_impl() {
    return stiles_control_params[sTiles::param::SemisparseImpl];
}

void sTiles_set_force_scotch_ordering(int on) {
    if (!expert_gate("sTiles_set_force_scotch_ordering")) return;
    set_param_both(sTiles::param::ForceScotchOrdering, (on != 0) ? 1 : 0);
}

int sTiles_get_force_scotch_ordering() {
    return stiles_control_params[sTiles::param::ForceScotchOrdering];
}

void sTiles_set_scotch_padding(int on) {
    if (!expert_gate("sTiles_set_scotch_padding")) return;
    set_param_both(sTiles::param::ScotchPadding, (on != 0) ? 1 : 0);
}

int sTiles_get_scotch_padding() {
    return stiles_control_params[sTiles::param::ScotchPadding];
}

void sTiles_set_path2_depth(int depth) {
    if (!expert_gate("sTiles_set_path2_depth")) return;
    set_param_both(sTiles::param::Path2Depth, (depth >= 0 && depth <= 12) ? depth : 0);
}

int sTiles_get_path2_depth() {
    return stiles_control_params[sTiles::param::Path2Depth];
}

void sTiles_set_serial_mode(int mode) {
    if (!expert_gate("sTiles_set_serial_mode")) return;
    // 0 = auto serial when num_cores==1, anything else = always parallel.
    // (Legacy values 3/4 were once accepted here; no consumer ever
    // distinguished them, so they collapse to 1.)
    set_param_both(sTiles::param::SerialMode, (mode != 0) ? 1 : 0);
}

int sTiles_get_serial_mode() {
    return stiles_control_params[sTiles::param::SerialMode];
}

// Deliberately NOT gated by expert mode (like sTiles_use_gpu): it only
// toggles a diagnostic printout, never numerics.
void sTiles_set_memory_estimate(int enable) {
    set_param_both(sTiles::param::MemoryEstimateMode, (enable != 0) ? 1 : 0);
}

int sTiles_get_memory_estimate() {
    return stiles_control_params[sTiles::param::MemoryEstimateMode];
}

void sTiles_use_gpu(int enable) {
    // GPU enable doesn't require expert mode - it's a runtime toggle.
    set_param_both(sTiles::param::GpuEnable, (enable != 0) ? 1 : 0);
    if (enable) {
        sTiles::Logger::info("GPU acceleration: ENABLED");
    } else {
        sTiles::Logger::info("GPU acceleration: DISABLED (CPU only)");
    }
}

int sTiles_get_gpu_enabled() {
    return stiles_control_params[sTiles::param::GpuEnable];
}

// Grace Hopper unified-memory path toggle (runtime).
// Compile-time gate: STILES_GPU_GRACE_HOPPER must also be defined.
// When STILES_GPU_GRACE_HOPPER is NOT defined, this call is a no-op accepted
// by the API so calling code can stay agnostic to the build flavor.
void sTiles_use_gh200_unified(int enable) {
#ifdef STILES_GPU_GRACE_HOPPER
    sTiles::gpu::gh200_unified::set(enable != 0);
    sTiles::Logger::info(enable
        ? "GH200 unified memory: ENABLED (runtime)"
        : "GH200 unified memory: DISABLED (runtime)");
#else
    (void)enable;
    sTiles::Logger::info("GH200 unified memory: build was not compiled with "
                         "STILES_GPU_GRACE_HOPPER; call ignored");
#endif
}

int sTiles_get_gh200_unified_enabled() {
#ifdef STILES_GPU_GRACE_HOPPER
    return sTiles::gpu::gh200_unified::enabled() ? 1 : 0;
#else
    return 0;
#endif
}

int sTiles_get_control_param(int index) {
    if (index >= 0 && index < STILES_NUM_PARAMS) {
        return stiles_control_params[index];
    }
    sTiles::Logger::warning("sTiles_get_control_param: index ", index,
                            " out of range [0, ", STILES_NUM_PARAMS, ")");
    return -1;
}

void sTiles_set_control_param(int index, int value) {
    if (!expert_gate("sTiles_set_control_param")) return;
    if (index < 0 || index >= STILES_NUM_PARAMS) {
        sTiles::Logger::warning("sTiles_set_control_param: index ", index,
                                " out of range [0, ", STILES_NUM_PARAMS, ")");
        return;
    }

    // Legacy reset convenience: -1 restores the platform default,
    // EXCEPT for slots where -1 is itself a valid auto sentinel
    // (1=TILE_SIZE auto, 5=TILE_ORDERING_SIZE auto).
    if (value == -1 && index != 1 && index != 5) {
        set_param_both(index, sTiles::kDefaultParams[index]);
        return;
    }

    switch (index) {
        case  0: sTiles_set_correction_mode(value);          return;
        case  1: sTiles_set_tile_size(value);                return;
        case  2: sTiles_set_ordering_mode(value);            return;
        case  3: sTiles_set_tile_type_mode(value);           return;
        case  4: sTiles_set_tile_ordering_mode(value);       return;
        case  5: sTiles_set_tile_ordering_size(value);       return;
        case  6: sTiles_force_ND(value);                     return;
        case  7: sTiles_set_inverse_storage_mode(value);     return;
        case  8: sTiles_set_use_omp(value);                  return;
        case  9: sTiles_set_semisparse_impl(value);          return;
        case 10:                                             // GPU_COMPARE: 0/1
                 set_param_both(index, (value != 0) ? 1 : 0); return;
        case 11: sTiles_use_gpu(value);                      return;
        case 12: sTiles_set_memory_estimate(value);          return;
        case 13: sTiles_set_tile_ordering_min_dim(value);    return;  // deprecated (warns)
        case 14: sTiles_set_serial_mode(value);              return;
        case 15: sTiles_set_bw_mode(value);                  return;
        case 16: sTiles_set_tile_first_ordering_mode(value); return;  // deprecated (warns)
        case 17:
        case 18:
        case 19:
        case 23:
        case 24:
                 sTiles::Logger::warning("sTiles_set_control_param: slot ", index,
                                         " is reserved; value stored but has no effect");
                 set_param_both(index, value);               return;
        case 20: sTiles_set_force_scotch_ordering(value);    return;
        case 21: sTiles_set_scotch_padding(value);           return;
        case 22: sTiles_set_path2_depth(value);              return;
        case 25: sTiles_set_tree_path_enable(value);         return;
        case 26: sTiles_set_tree_path_force(value);          return;
        default:
                 sTiles::Logger::warning("sTiles_set_control_param: slot ", index,
                                         " is reserved; value stored but has no effect");
                 set_param_both(index, value);               return;
    }
}

void sTiles_reset_control_param(int index) {
    if (index >= 0 && index < STILES_NUM_PARAMS) {
        set_param_both(index, sTiles::kDefaultParams[index]);
    }
}

void sTiles_reset_all_params(void) {
    for (int i = 0; i < STILES_NUM_PARAMS; ++i) {
        set_param_both(i, sTiles::kDefaultParams[i]);
    }
}

const char* sTiles_get_param_description(int index) {
    switch (index) {
        case  0: return "Correction mode [sTiles_set_correction_mode]\n"
                        "    0 = No pruning (default)\n"
                        "    1 = Prune single zero active column range\n"
                        "    2 = Prune zero semisparse tiles\n"
                        "    3 = Prune zero semisparse columns";
        case  1: return "Tile size [sTiles_set_tile_size]\n"
                        "   -1 = Auto-detect tile size (default)\n"
                        "    N = Use tile size N (typical: 40 for CPU)";
        case  2: return "Ordering candidates [sTiles_set_ordering_mode]\n"
                        "    0      = Adaptive selection by matrix size/class (default)\n"
                        "    digits = Exact candidate set, best fill wins:\n"
                        "             1 = RCM       2 = METIS (ND)  3 = SCOTCH\n"
                        "             4 = ASCOTCH   5 = FSCOTCH     6 = AMD\n"
                        "             7 = CAMD      8 = COLAMD\n"
                        "    e.g. 167 = evaluate RCM, AMD, CAMD; 3 = force SCOTCH only";
        case  3: return "Tile type [sTiles_set_tile_type_mode]\n"
                        "    0 = Dense tiles (uniform tile_size, fully populated)\n"
                        "    1 = Semisparse tiles (uniform tile_size, per-column bitmap)\n"
                        "    2 = Non-uniform tiles (etree-driven cell sizes, sTiles::sparse)\n"
                        "    3 = Auto (resolves to 0/1/2 after symbolic, by occupancy/fill/skew)";
        case  4: return "Tile ordering mode [sTiles_set_tile_ordering_mode]\n"
                        "    0      = No tile-level reordering\n"
                        "    >0     = Digits select strategies for per-partition tile ordering:\n"
                        "             1 = RCM, 2 = ND, 3 = ND+RCM, 4 = SCOTCH\n"
                        "             5 = AMD, 6 = CAMD, 7 = COLAMD, 8 = CCOLAMD, 9 = SYMAMD\n"
                        "    Default 14569 = RCM, SCOTCH, AMD, CAMD, SYMAMD.\n"
                        "    Only engages when partition_sizes are present (SCOTCH padding).";
        case  5: return "Tile ordering size [sTiles_set_tile_ordering_size]\n"
                        "   -1 = Auto: tile_size / 2 (default)\n"
                        "    N = Block size for tile-level ordering";
        case  6: return "Force nested dissection [sTiles_force_ND]\n"
                        "    0 = Auto-detect ordering (default)\n"
                        "    1 = Force nested dissection";
        case  7: return "Inverse storage [sTiles_set_inverse_storage_mode]\n"
                        "    0 = Overwrite factor in-place\n"
                        "    1 = Separate inverse storage (default)";
        case  8: return "Parallelization backend [sTiles_set_use_omp]\n"
                        "    0 = pthreads (Linux default)\n"
                        "    1 = OpenMP (macOS default)";
        case  9: return "Semisparse implementation [sTiles_set_semisparse_impl]\n"
                        "    0 = Original implementation\n"
                        "    1 = Improved implementation\n"
                        "    2 = Vectorized implementation (default)\n"
                        "    3 = Vectorized + sparse-aware 1-core fast path";
        case 10: return "GPU comparison mode [sTiles_set_control_param]\n"
                        "    0 = GPU results only (default)\n"
                        "    1 = GPU with CPU validation";
        case 11: return "GPU enable [sTiles_use_gpu]\n"
                        "    0 = GPU disabled\n"
                        "    1 = GPU enabled (default)";
        case 12: return "Memory estimate [sTiles_set_memory_estimate]\n"
                        "    0 = Skip memory estimation (default)\n"
                        "    1 = Compute & print estimate";
        case 13: return "Reserved (was tile-ordering min dim) -- DEPRECATED, no effect.";
        case 14: return "Serial mode [sTiles_set_serial_mode]\n"
                        "    0 = Serial when 1 core, parallel otherwise (default)\n"
                        "    1 = Always use parallel kernels";
        case 15: return "Bandwidth mode [sTiles_set_bw_mode]\n"
                        "    0 = Conservative: bandwidth = tile_width - 1 (default)\n"
                        "    1 = Tight: bandwidth = la - fa";
        case 16: return "Reserved (was tile-first ordering mode) -- DEPRECATED, no effect.";
        case 17:
        case 18:
        case 19: return "Reserved (formerly blocksparse subsystem; removed).";
        case 20: return "Force SCOTCH ordering [sTiles_set_force_scotch_ordering]\n"
                        "    0 = Adaptive candidate selection (default)\n"
                        "    1 = Force SCOTCH (id=4), skip the bake-off";
        case 21: return "SCOTCH padding [sTiles_set_scotch_padding]\n"
                        "    0 = Off (default)\n"
                        "    1 = Pad P1|P2|Sep so collect_tasks Path 2 engages";
        case 22: return "Path 2 ND scheduling depth [sTiles_set_path2_depth]\n"
                        "    0/1 = Legacy 3-way (default); D>=2 = 2^(D+1)-1 regions";
        case 23:
        case 24: return "Reserved.";
        case 25: return "Tree-reduction path enable [sTiles_set_tree_path_enable]\n"
                        "    0 = Off\n"
                        "    1 = On (default; corner_probe's gate still decides per matrix)";
        case 26: return "Tree-reduction path force [sTiles_set_tree_path_force]\n"
                        "    0 = Honor the activation gate (default)\n"
                        "    1 = Force activation (kernel-correctness tests)";
        default: return "Unknown/reserved";
    }
}

} // extern "C"

// =============================================================================
// Environment-variable registry
// =============================================================================
//
// Central catalog of every STILES_* environment variable the library reads.
// These are runtime OVERRIDES and debug hooks, not the primary configuration
// interface (that is the control-parameter array above). Keep this table in
// sync when adding a getenv() call; sTiles_print_env() is the user-visible
// index of them.

namespace {

struct EnvEntry {
    const char* name;
    const char* def;    // default / inert state
    const char* what;
};

// Grouped: routing & auto-mode, ordering, tree-reduction & partitions,
// kernels, GPU, diagnostics.
static const EnvEntry stiles_env_registry[] = {
    // --- auto-mode routing (tile type 3) ---
    { "STILES_DENSE_OCC",         "0.90",  "auto-mode: occupancy threshold that routes to dense tiles (mode 0)" },
    { "STILES_ARROWHEAD_SKEW",    "20",    "auto-mode: degree-skew threshold admitting bordered/arrowhead matrices to semisparse" },
    { "STILES_LEGACY_FILL",       "unset", "auto-mode: use the old fill-only routing rule (low fill -> semisparse)" },
    { "STILES_SNODE_CAP",         "max(64, 2*tile_size)", "sparse mode: supernode-width cap for the sparse module's analyze" },
    { "STILES_SPARSE_CORE_CAP",   "auto",  "sparse mode: override the work-aware core cap (0 disables the cap)" },
    { "STILES_SPARSE_MIN_FLOPS_PER_RANK", "auto", "sparse mode: min flops per rank used by the core cap" },
    { "STILES_SPARSE_SYM_VERBOSE", "unset", "sparse mode: verbose symbolic-factorization tracing" },
    // --- ordering / bake-off ---
    { "STILES_FORCE_ORDERING",    "unset", "pin the ordering bake-off to a single id (1=RCM, 4=SCOTCH, 21=METIS, 5=AMD, 6=CAMD, ...)" },
    { "STILES_DISABLE_SCOTCH",    "0",     "drop the SCOTCH family (4/41/42) from the bake-off" },
    { "STILES_SCOTCH_CTX",        "1",     "per-call SCOTCH_Context so SCOTCH orderings run concurrently (0 = old global mutex)" },
    { "STILES_SHARED_CSR",        "auto",  "share one adjacency CSR across bake-off candidates (auto: on for sparse-class); 1/0 override" },
    { "STILES_PARALLEL_EVAL",     "1",     "evaluate ordering candidates concurrently (std::async)" },
    { "STILES_EVAL_INNER",        "1",     "nested parallelism inside each candidate's count/sort/COO evaluation" },
    { "STILES_METIS_SEED",        "METIS default", "METIS ordering: random seed" },
    { "STILES_METIS_NITER",       "METIS default", "METIS ordering: refinement iterations" },
    { "STILES_METIS_PFACTOR",     "METIS default", "METIS ordering: pruning factor" },
    { "STILES_METIS_NSEPS",       "METIS default", "METIS ordering: separators tried per level" },
    // --- tree reduction / composed ND partitions ---
    { "STILES_CORNER_PROBE_ACTIVATE", "unset", "enable the tree-reduction corner probe (shell override of param 25)" },
    { "STILES_CORNER_PROBE_FORCE",    "unset", "force tree-reduction activation, bypassing the 6-gate predicate (param 26)" },
    { "STILES_TREE_MAX_CORES",    "auto",  "cap the tree-reduction worker count" },
    { "STILES_TREE_ALPHA",        "auto",  "tree-reduction load-balance exponent" },
    { "STILES_COMPOSED_MIN_SEP",  "auto",  "composed ND+tree: min separator width" },
    { "STILES_COMPOSED_MAX_SEP",  "auto",  "composed ND+tree: max separator width (force clamp)" },
    { "STILES_COMPOSED_MAX_HEAVY_SLOTS", "auto", "composed ND+tree: heavy-slot cap in the corner probe" },
    { "STILES_PART_MIN_CORES",    "4",     "ND partitions: min cores before leaf partitioning engages" },
    { "STILES_PART_MIN_TILES_PER_LEAF", "8", "ND partitions: min tiles per leaf" },
    { "STILES_AUTO_PART_MIN_CORES", "auto", "auto-partition gate: min cores" },
    // --- kernels / execution ---
    { "STILES_FLAT_SCHED",        "0",     "flatten the task schedule (disable region nesting)" },
    { "STILES_FORCE_NO_GATHER",   "0",     "disable the gather fast path in task generation" },
    { "STILES_FUSE_THRESHOLD",    "auto",  "chol executor: task-fusion threshold" },
    { "STILES_SPARSE_FUSE_W",     "auto",  "sparse solve: fused-column width" },
    { "STILES_SPARSE_CSC_W",      "auto",  "sparse solve: packed-CSC supernode-width gate" },
    { "STILES_SPARSE_CSC_PAR",    "0",     "sparse solve: parallel packed-CSC path (measured slower; keep 0)" },
    { "STILES_SPARSE_CSC_NRHS",   "auto",  "sparse solve: max nrhs for the packed-CSC path" },
    { "STILES_SOLVE_KERNEL",      "auto",  "CSC solve: kernel override (see compute/csc_solve.cpp)" },
    { "STILES_SEMI_DIAGINV_TRUNC","1",     "semisparse selinv: truncated diagonal-inverse (0 restores dtbtrs)" },
    { "STILES_SEMI_TRMM_SOLVE",   "0",     "semisparse selinv: banded case-2 TRMM solve (measured slower; keep 0)" },
    { "STILES_SKIP_SOLVE_INV_PREP","0",    "skip solve/inverse prep during task generation" },
    // --- GPU ---
    { "STILES_GH200_UNIFIED",     "unset", "GH200 unified-memory slab allocator (needs STILES_GPU_GRACE_HOPPER build)" },
    // --- diagnostics / dumps (may print or exit) ---
    { "STILES_DUMP_TASKS",        "unset", "dump the generated task schedule" },
    { "STILES_DUMP_NEIGHBOR",     "unset", "print per-matrix routing signals after symbolic, then exit(0)" },
    { "STILES_DUMP_LAYOUT",       "unset", "dump the tile layout" },
    { "STILES_PROFILE_SYM",       "unset", "profile the symbolic phase (per-candidate timings)" },
    { "STILES_DEBUG_ORDERING",    "unset", "verbose ordering/solve debug output" },
    { "STILES_PRINT_OCC",         "unset", "print per-tile occupancy statistics (semisparse)" },
    { "STILES_PAD_DUMP",          "unset", "dump SCOTCH padding regions" },
    { "STILES_BCACHE_STATS",      "unset", "print selinv B-cache statistics" },
    { "STILES_EXPORT_PERM",       "unset", "export the winning permutation to file" },
    { "STILES_NO_EXPORT",         "unset", "suppress EXPORT_MATRIX file output" },
    { "STILES_NO_BANNER",         "unset", "suppress the startup logo/banner" },
};

// Recognized STILES_* names that are NOT runtime overrides read by the
// library: build-time make variables and test-suite knobs a user may
// legitimately have exported. They must not trigger the typo warning.
static const char* stiles_env_known_other[] = {
    "STILES_WAIT_MODE",       // make variable (compile-time spin policy)
    "STILES_LOG_LEVEL",       // make variable (compile-time default log level)
    "STILES_TEST_PTHREADS",   // run/accuracy test-suite backend toggle
    "STILES_PATH",            // run/ Makefile variable
    // INLA/GMRFLib MPI-layer vars: consumed by r-inla's gmrflib/stiles-mpi.c
    // and the inla-stiles-mpi-call.sh launcher, NOT by sTiles itself. Listed
    // here so sTiles' env validator does not flag them as typos.
    "STILES_MPI_LOG",
    "STILES_RANK0_LIGHT",
    "STILES_WORKER_CORES",
    "STILES_MPI_NP",
    "STILES_MPI_OVERSUBSCRIBE",
    "STILES_MPI_EVAL",
    "STILES_MPI_STATS",
};

} // namespace

// Portable access to the process environment (macOS dylibs cannot link the
// flat `environ` symbol; Windows/MinGW spells it `_environ`).
#if defined(__APPLE__)
  #include <crt_externs.h>
  #define STILES_ENVIRON (*_NSGetEnviron())
#elif defined(_WIN32)
  #define STILES_ENVIRON _environ
#else
  extern char **environ;
  #define STILES_ENVIRON environ
#endif

extern "C" {

/**
 * Scan the process environment for STILES_*-prefixed variables the library
 * does not recognize and warn about each (a set-but-misspelled override is
 * silently inert otherwise — e.g. STILES_DENSE_OC=0.8 would do nothing).
 * Called once from sTiles_create; safe to call repeatedly.
 */
void stiles_env_validate(void) {
    static bool done = false;
    if (done) return;
    done = true;
    char** env = STILES_ENVIRON;
    if (!env) return;
    for (char** e = env; *e; ++e) {
        const char* entry = *e;
        if (std::strncmp(entry, "STILES_", 7) != 0) continue;
        const char* eq = std::strchr(entry, '=');
        const std::size_t len = eq ? static_cast<std::size_t>(eq - entry) : std::strlen(entry);
        bool known = false;
        for (const EnvEntry& k : stiles_env_registry)
            if (std::strlen(k.name) == len && std::strncmp(k.name, entry, len) == 0) { known = true; break; }
        if (!known)
            for (const char* k : stiles_env_known_other)
                if (std::strlen(k) == len && std::strncmp(k, entry, len) == 0) { known = true; break; }
        if (!known) {
            sTiles::Logger::warning("unrecognized environment variable ",
                                    std::string(entry, len),
                                    " (possible typo? it has no effect; see sTiles_print_env() for the known set)");
        }
    }
}

/**
 * Print every STILES_* environment variable the library reads: current value
 * (or "unset"), the default/inert state, and a one-line description. These
 * are runtime overrides and debug hooks; prefer the sTiles_set_* API for
 * stable configuration.
 */
void sTiles_print_env(void) {
    std::fprintf(stdout,
        "sTiles Environment Overrides (unset = built-in default)\n"
        "──────────────────────────────────────────────────────────────────\n");
    for (const EnvEntry& e : stiles_env_registry) {
        const char* v = std::getenv(e.name);
        std::fprintf(stdout, " %-33s = %-10s [default: %s]\n    %s\n",
                     e.name, v ? v : "(unset)", e.def, e.what);
    }
    std::fprintf(stdout,
        "──────────────────────────────────────────────────────────────────\n");
}

} // extern "C"
