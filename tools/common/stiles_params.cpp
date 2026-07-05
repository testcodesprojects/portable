/**
 * @file stiles_params.cpp
 * @brief C-ABI setters/getters for the sTiles control-parameter array.
 *
 * Owns the platform-default table and every `sTiles_set_*` / `sTiles_get_*`
 * entry point that wraps `stiles_control_params[N]`. The live array itself
 * stays in tools/process/process.cpp because preprocessing code reads it
 * directly in many places; we reach it here via `extern`.
 *
 * See stiles_params.hpp for the named slot indices and the immutable
 * defaults (`sTiles::kDefaultParams`).
 */

#include "stiles_params.hpp"
#include "stiles_logger.hpp"
#ifdef STILES_GPU
#include "../gpu/gpu_dispatch_plan.hpp"   // gh200_unified::{set,enabled}
#endif

#include <cstdio>

// Live storage (defined in tools/process/process.cpp).
extern bool expert_mode_enabled;
extern int  stiles_control_params[STILES_NUM_PARAMS];

// Forward declaration: defined in tools/ordering/ordering_nd.cpp.
// Selects METIS_NodeNDP (use_ndp=1) vs METIS_NodeND (use_ndp=0).
extern "C" void stiles_set_nd_use_ndp(int v);

// Platform-default snapshot used by reset_*. Must match the live array's
// initialiser in process.cpp; both derive from sTiles::kDefaultParams.
#ifdef __APPLE__
static const int stiles_default_params[STILES_NUM_PARAMS] = {
    0, -1, 14569, 0, 14569, -1, 0, 1, 0, 2,
    0,  1, 0, 100, 0, 0, 0, 0, 0, 0,
    0,  0, 0, 0, 0,
    // 25..49: reserved for future params
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
};
#else
static const int stiles_default_params[STILES_NUM_PARAMS] = {
    0, -1, 14569, 1, 14569, -1, 0, 1, 0, 2,
    0,  1, 0, 100, 0, 0, 0, 0, 0, 0,
    0,  0, 0, 0, 0,
    // 25..49: reserved for future params
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
};
#endif

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

void sTiles_expert_user() {
    expert_mode_enabled = true;
    sTiles::Logger::info("Expert mode enabled - configuration functions are now available");
}

void sTiles_set_tile_size(int tile_size) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_tile_size: call sTiles_expert_user() first to enable configuration");
        return;
    }
    if (tile_size > 0) {
        stiles_control_params[1] = tile_size;
    }
}

int sTiles_return_tile_size() {
    return stiles_control_params[1];
}

void sTiles_set_bw_mode(int mode) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_bw_mode: call sTiles_expert_user() first to enable configuration");
        return;
    }
    stiles_control_params[15] = (mode != 0) ? 1 : 0;
}

int sTiles_get_bw_mode() {
    return stiles_control_params[15];
}

int* sTiles_get_params() {
    return stiles_control_params;
}

// Slot 25: TREE_PATH_ENABLE — global enable for the tree-reduction path
// triggered by corner_probe. 0 = disabled (default), 1 = enabled.
// Read by sTiles::preprocess::corner_probe; env var STILES_CORNER_PROBE_ACTIVATE=1
// remains supported as an override for shell-driven testing.
void sTiles_set_tree_path_enable(int on) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_tree_path_enable: call sTiles_expert_user() first to enable configuration");
        return;
    }
    stiles_control_params[25] = (on != 0) ? 1 : 0;
}

int sTiles_get_tree_path_enable() {
    return stiles_control_params[25];
}

// Slot 26: TREE_PATH_FORCE — bypass the 6-gate predicate and activate the
// tree-reduction path unconditionally. Used by kernel-correctness tests to
// guarantee the reduction code path executes on matrices where the gate
// would otherwise reject. 0 = honor gate (default), 1 = force activation.
// Env var STILES_CORNER_PROBE_FORCE=1 is also honored (either trigger activates).
void sTiles_set_tree_path_force(int on) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_tree_path_force: call sTiles_expert_user() first to enable configuration");
        return;
    }
    stiles_control_params[26] = (on != 0) ? 1 : 0;
}

int sTiles_get_tree_path_force() {
    return stiles_control_params[26];
}

void sTiles_set_params(const int* params, int n) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_params: call sTiles_expert_user() first to enable configuration");
        return;
    }
    if (n > 20) n = 20;
    for (int i = 0; i < n; ++i) {
        stiles_control_params[i] = params[i];
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
                 } break;
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
    static const char* names[STILES_NUM_PARAMS] = {
        "CORRECTION_MODE", "TILE_SIZE", "ORDERING_MODE", "TILE_TYPE",
        "TILE_ORDERING_MODE", "TILE_ORDERING_SIZE", "FORCE_ND",
        "INVERSE_STORAGE", "USE_OMP", "SEMISPARSE_IMPL",
        "GPU_COMPARE", "GPU_ENABLE", "MEMORY_ESTIMATE",
        "TILE_ORD_MIN_DIM", "SERIAL_MODE", "BW_MODE",
        "TILE_FIRST_ORDERING_MODE", "RESERVED_17", "RESERVED_18",
        "RESERVED_19", "FORCE_SCOTCH_ORDERING", "SCOTCH_PADDING",
        "PATH2_DEPTH", "SOLVE_REFINEMENT", "RESERVED_24",
        "TREE_PATH_ENABLE", "TREE_PATH_FORCE"
    };

    std::fprintf(stdout,
        "sTiles Global Parameters\n"
        "──────────────────────────────────────────────────────────────────\n");
    // Iterate the named slots; reserved 26..49 don't need to print.
    for (int i = 0; i < 26; ++i) {
        int val = stiles_control_params[i];
        const char* vname = stiles_value_name(i, val);
        if (vname) {
            std::fprintf(stdout, " [%2d] %-22s = %-6d  (%s)\n",
                         i, names[i], val, vname);
        } else {
            std::fprintf(stdout, " [%2d] %-22s = %d\n",
                         i, names[i], val);
        }
    }
    std::fprintf(stdout,
        "──────────────────────────────────────────────────────────────────\n");
}

void sTiles_set_correction_mode(int value) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_correction_mode: call sTiles_expert_user() first to enable configuration");
        return;
    }
    stiles_control_params[0] = value;
}

void sTiles_set_ordering_mode(int value) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_ordering_mode: call sTiles_expert_user() first to enable configuration");
        return;
    }
    if (value < 0) {
        sTiles::Logger::warning("sTiles_set_ordering_mode ignored negative value ", value);
        return;
    }

    if (value == 0) {
        stiles_control_params[2] = 0;
        return;
    }

    int digits = value;
    while (digits > 0) {
        const int d = digits % 10;
        if (d <= 0 || d > 9) {
            sTiles::Logger::warning("sTiles_set_ordering_mode expects digits 1..9 only; got ", value);
            return;
        }
        digits /= 10;
    }

    stiles_control_params[2] = value;
}

void sTiles_set_tile_ordering_mode(int value) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_tile_ordering_mode: call sTiles_expert_user() first to enable configuration");
        return;
    }
    if (value < 0) {
        sTiles::Logger::warning("sTiles_set_tile_ordering_mode ignored negative value ", value);
        return;
    }

    if (value == 0) {
        stiles_control_params[4] = 0;
        return;
    }

    int digits = value;
    while (digits > 0) {
        const int d = digits % 10;
        if (d <= 0 || d > 9) {
            sTiles::Logger::warning("sTiles_set_tile_ordering_mode expects digits 1..9 only; got ", value);
            return;
        }
        digits /= 10;
    }

    stiles_control_params[4] = value;
}

void sTiles_set_tile_first_ordering_mode(int value) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_tile_first_ordering_mode: call sTiles_expert_user() first to enable configuration");
        return;
    }
    if (value < 0) {
        sTiles::Logger::warning("sTiles_set_tile_first_ordering_mode ignored negative value ", value);
        return;
    }

    if (value == 0) {
        stiles_control_params[16] = 0;
        return;
    }

    int digits = value;
    while (digits > 0) {
        const int d = digits % 10;
        if (d <= 0 || d > 9) {
            sTiles::Logger::warning("sTiles_set_tile_first_ordering_mode expects digits 1..9 only; got ", value);
            return;
        }
        digits /= 10;
    }

    stiles_control_params[16] = value;
}

void sTiles_set_tile_ordering_size(int value) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_tile_ordering_size: call sTiles_expert_user() first to enable configuration");
        return;
    }
    if (value < -1) {
        sTiles::Logger::warning("sTiles_set_tile_ordering_size ignored negative value ", value);
        return;
    }
    stiles_control_params[5] = value;
}

void sTiles_set_tile_ordering_min_dim(int value) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_tile_ordering_min_dim: call sTiles_expert_user() first to enable configuration");
        return;
    }
    if (value < 0) {
        sTiles::Logger::warning("sTiles_set_tile_ordering_min_dim ignored negative value ", value);
        return;
    }
    stiles_control_params[13] = value;
}

void sTiles_set_tile_type_mode(int value) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_tile_type_mode: call sTiles_expert_user() first to enable configuration");
        return;
    }
    // 0 = dense, 1 = semisparse, 2 = non-uniform tiles, 3 = auto (pick based
    // on fill ratio after symbolic factorization: low fill → semisparse,
    // high fill → non-uniform).
    if (value == 0 || value == 1 || value == 2 || value == 3) {
        stiles_control_params[3] = value;
    } else {
        sTiles::Logger::warning("sTiles_set_tile_type_mode: invalid value=", value,
                                " (must be 0=dense, 1=semisparse, 2=non-uniform, or 3=auto)");
    }
}

void sTiles_force_ND(int enable) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_force_ND: call sTiles_expert_user() first to enable configuration");
        return;
    }
    stiles_set_nd_use_ndp(enable);
    stiles_control_params[6] = (enable != 0) ? 1 : 0;
}

void sTiles_set_inverse_storage_mode(int mode) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_inverse_storage_mode: call sTiles_expert_user() first to enable configuration");
        return;
    }
    stiles_control_params[7] = (mode == 0 || mode == 1) ? mode : 1;
}

int sTiles_get_inverse_storage_mode() {
    return stiles_control_params[7];
}

void sTiles_set_use_omp(int mode) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_use_omp: call sTiles_expert_user() first to enable configuration");
        return;
    }
    stiles_control_params[8] = (mode != 0) ? 1 : 0;
}

int sTiles_get_use_omp() {
    return stiles_control_params[8];
}

void sTiles_set_semisparse_impl(int impl) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_semisparse_impl: call sTiles_expert_user() first to enable configuration");
        return;
    }
    stiles_control_params[9] = (impl >= 0 && impl <= 3) ? impl : 2;
}

int sTiles_get_semisparse_impl() {
    return stiles_control_params[9];
}

void sTiles_set_force_scotch_ordering(int on) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_force_scotch_ordering: call sTiles_expert_user() first to enable configuration");
        return;
    }
    stiles_control_params[20] = (on != 0) ? 1 : 0;
}

int sTiles_get_force_scotch_ordering() {
    return stiles_control_params[20];
}

void sTiles_set_scotch_padding(int on) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_scotch_padding: call sTiles_expert_user() first to enable configuration");
        return;
    }
    stiles_control_params[21] = (on != 0) ? 1 : 0;
}

int sTiles_get_scotch_padding() {
    return stiles_control_params[21];
}

void sTiles_set_path2_depth(int depth) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_path2_depth: call sTiles_expert_user() first to enable configuration");
        return;
    }
    stiles_control_params[22] = (depth >= 0 && depth <= 12) ? depth : 0;
}

int sTiles_get_path2_depth() {
    return stiles_control_params[22];
}

void sTiles_set_serial_mode(int mode) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_serial_mode: call sTiles_expert_user() first to enable configuration");
        return;
    }
    if (mode == 3 || mode == 4) {
        stiles_control_params[14] = mode;
    } else {
        stiles_control_params[14] = (mode != 0) ? 1 : 0;
    }
}

int sTiles_get_serial_mode() {
    return stiles_control_params[14];
}

void sTiles_set_memory_estimate(int enable) {
    stiles_control_params[12] = (enable != 0) ? 1 : 0;
}

int sTiles_get_memory_estimate() {
    return stiles_control_params[12];
}

void sTiles_use_gpu(int enable) {
    // GPU enable doesn't require expert mode - it's a runtime toggle.
    stiles_control_params[11] = (enable != 0) ? 1 : 0;
    if (enable) {
        sTiles::Logger::info("GPU acceleration: ENABLED");
    } else {
        sTiles::Logger::info("GPU acceleration: DISABLED (CPU only)");
    }
}

int sTiles_get_gpu_enabled() {
    return stiles_control_params[11];
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
    return -1;
}

void sTiles_set_control_param(int index, int value) {
    if (!expert_mode_enabled) {
        sTiles::Logger::warning("sTiles_set_control_param: call sTiles_expert_user() first to enable configuration");
        return;
    }
    if (index < 0 || index >= STILES_NUM_PARAMS) {
        sTiles::Logger::warning("sTiles_set_control_param: index ", index,
                                " out of range [0, ", STILES_NUM_PARAMS, ")");
        return;
    }

    // Legacy reset convenience: -1 restores the platform default,
    // EXCEPT for slots where -1 is itself a valid auto sentinel
    // (1=TILE_SIZE auto, 5=TILE_ORDERING_SIZE auto).
    if (value == -1 && index != 1 && index != 5) {
        stiles_control_params[index] = stiles_default_params[index];
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
        case 11: stiles_control_params[index] = (value != 0) ? 1 : 0; return;
        case 12: sTiles_set_memory_estimate(value);          return;
        case 13: sTiles_set_tile_ordering_min_dim(value);    return;
        case 14: sTiles_set_serial_mode(value);              return;
        case 15: sTiles_set_bw_mode(value);                  return;
        case 16: sTiles_set_tile_first_ordering_mode(value); return;
        case 17:                                             // reserved (was BLOCKSPARSE_MODE)
        case 18:                                             // reserved (was BLOCKSPARSE_THRESHOLD)
        case 19: stiles_control_params[index] = value;       return;  // reserved (was INV_BACKEND)
        case 20: sTiles_set_force_scotch_ordering(value);    return;
        case 21: sTiles_set_scotch_padding(value);           return;
        case 22: sTiles_set_path2_depth(value);              return;
        default:
            sTiles::Logger::warning("sTiles_set_control_param: slot ", index,
                                    " has no typed setter wired");
            return;
    }
}

void sTiles_reset_control_param(int index) {
    if (index >= 0 && index < STILES_NUM_PARAMS) {
        stiles_control_params[index] = stiles_default_params[index];
    }
}

void sTiles_reset_all_params(void) {
    for (int i = 0; i < STILES_NUM_PARAMS; ++i) {
        stiles_control_params[i] = stiles_default_params[i];
    }
}

const char* sTiles_get_param_description(int index) {
    switch (index) {
        case  0: return "Correction mode [sTiles_set_correction_mode]\n"
                        "    0 = No pruning (skip semisparse checks)\n"
                        "    1 = Prune single zero active column range\n"
                        "    2 = Prune zero semisparse tiles (default)\n"
                        "    3 = Prune zero semisparse columns";
        case  1: return "Tile size [sTiles_set_tile_size]\n"
                        "   -1 = Auto-detect tile size\n"
                        "    N = Use tile size N (typical: 40 for CPU)";
        case  2: return "Ordering mode [sTiles_set_ordering_mode]\n"
                        "    0      = Use single ordering from call.ordering_strategy\n"
                        "    >0     = Digits select orderings to evaluate (best wins):\n"
                        "             1 = RCM (Reverse Cuthill-McKee)\n"
                        "             2 = ND (METIS nested dissection)\n"
                        "             3 = ND+RCM (nested dissection + RCM refinement)\n"
                        "             4 = SCOTCH\n"
                        "             5 = AMD (SuiteSparse approximate minimum degree)\n"
                        "             6 = CAMD (SuiteSparse constrained AMD)\n"
                        "             7 = COLAMD (SuiteSparse column AMD)\n"
                        "             8 = CCOLAMD (SuiteSparse constrained column AMD)\n"
                        "             9 = SYMAMD (SuiteSparse symmetric AMD)\n"
                        "            10 = User-provided permutation\n"
                        "    e.g. 145689 = evaluate RCM, SCOTCH, AMD, CAMD, CCOLAMD, SYMAMD";
        case  3: return "Tile type [sTiles_set_tile_type_mode]\n"
                        "    0 = Dense tiles (uniform tile_size, fully populated)\n"
                        "    1 = Semisparse tiles (uniform tile_size, per-column bitmap)\n"
                        "    2 = Non-uniform tiles (etree-driven cell sizes, sTiles::sparse)\n"
                        "    3 = Auto (resolves to 1 or 2 after symbolic, by fill ratio)";
        case  4: return "Tile ordering mode [sTiles_set_tile_ordering_mode]\n"
                        "    0      = No tile-level reordering\n"
                        "    >0     = Digits select orderings (same scheme as param[2]):\n"
                        "             1 = RCM, 2 = ND, 3 = ND+RCM, 4 = SCOTCH\n"
                        "             5 = AMD, 6 = CAMD, 7 = COLAMD, 8 = CCOLAMD, 9 = SYMAMD\n"
                        "    e.g. 145689 = evaluate RCM, SCOTCH, AMD, CAMD, CCOLAMD, SYMAMD";
        case  5: return "Tile ordering size [sTiles_set_tile_ordering_size]\n"
                        "    N = Block size for tile-level ordering (default: 20)";
        case  6: return "Force nested dissection [sTiles_force_ND]\n"
                        "    0 = Auto-detect ordering\n"
                        "    1 = Force nested dissection";
        case  7: return "Inverse storage [sTiles_set_inverse_storage_mode]\n"
                        "    0 = Overwrite factor in-place\n"
                        "    1 = Separate inverse storage";
        case  8: return "Parallelization backend [sTiles_set_use_omp]\n"
                        "    0 = pthreads (default)\n"
                        "    1 = OpenMP";
        case  9: return "Semisparse implementation [sTiles_set_semisparse_impl]\n"
                        "    0 = Original implementation\n"
                        "    1 = Improved implementation\n"
                        "    2 = Vectorized implementation (default)";
        case 10: return "GPU comparison mode [sTiles_set_control_param]\n"
                        "    0 = GPU results only\n"
                        "    1 = GPU with CPU validation";
        case 11: return "GPU enable [sTiles_use_gpu]\n"
                        "    0 = GPU disabled\n"
                        "    1 = GPU enabled (default)";
        case 12: return "Memory estimate [sTiles_set_memory_estimate]\n"
                        "    0 = Skip memory estimation\n"
                        "    1 = Compute & print estimate";
        case 13: return "Tile ordering min dim threshold [sTiles_set_tile_ordering_min_dim]\n"
                        "    0 = Always run tile ordering\n"
                        "    N = Skip tile ordering when tiles_dim < N (default: 100)";
        case 14: return "Serial mode [sTiles_set_serial_mode]\n"
                        "    0 = Serial when 1 core, parallel otherwise\n"
                        "    1 = Always use parallel kernels";
        case 15: return "Bandwidth mode [sTiles_set_bw_mode]\n"
                        "    0 = Conservative: bandwidth = tile_width - 1\n"
                        "    1 = Tight: bandwidth = la - fa";
        default: return "Unknown/reserved";
    }
}

} // extern "C"
