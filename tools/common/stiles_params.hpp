/**
 * @file stiles_params.hpp
 * @brief Single source of truth for the sTiles control-parameter array.
 *
 * Replaces the magic indices `params[3]`, `params[19]`, etc. scattered across
 * the codebase with named constants in `sTiles::param`. Adding a new
 * parameter is a one-file edit:
 *   1. Bump kNumDefinedParams (and kNumParams if 27..49 are exhausted).
 *   2. Add a name in `sTiles::param`.
 *   3. Add its default to STILES_DEFAULT_PARAMS_INIT (both platform variants
 *      if they differ).
 *   4. Document the slot in the comment table below.
 *
 * STILES_DEFAULT_PARAMS_INIT is the ONE textual definition of the default
 * table. It initializes, in this order:
 *   - sTiles::kDefaultParams            (constexpr reference copy, this file)
 *   - stiles_control_params[]           (live array, tools/process/process.cpp)
 *   - stiles_user_params[]              (user-config shadow, process.cpp)
 * Never hand-edit one of those initializers independently — that is exactly
 * how the slot-25 reset bug happened (live default ON, reset table OFF).
 *
 * IMPORTANT: the *order* of slots is part of the ABI exposed via
 * sTiles_get_all_params / sTiles_set_params. Do not reorder existing
 * entries — only append new ones at the end.
 */

#ifndef _STILES_PARAMS_HPP_
#define _STILES_PARAMS_HPP_

namespace sTiles {

/// Size of the control-parameter array.
/// Slots 0..26 are defined; slots 27..49 are reserved for future use
/// (default 0). Reserve a slot by giving it a name in `sTiles::param`.
inline constexpr int kNumParams = 50;
inline constexpr int kNumDefinedParams = 27;

/// Named indices for the control-parameter array.
///
/// Use `params[sTiles::param::TileTypeMode]` instead of `params[3]`.
namespace param {

// ---------------------------------------------------------------------------
// Slot index | Setter                              | Meaning
// ---------------------------------------------------------------------------
inline constexpr int SemisparsePruningMode   =  0; // sTiles_set_correction_mode
                                                   //   0 = no pruning (default)
                                                   //   1 = prune_single_zero_active_column_range
                                                   //   2 = prune_zero_semisparse_tiles_range
                                                   //   3 = prune_zero_semisparse_columns_range
inline constexpr int UserTileSize            =  1; // -1 = auto-detect (default)
inline constexpr int OrderingMode            =  2; // sTiles_set_ordering_mode
                                                   //   0 = adaptive candidate selection by matrix
                                                   //       size/class (default)
                                                   //   digits 1..8 = exact bake-off candidate set in
                                                   //       the PUBLIC numbering (STILES_ORD_*):
                                                   //       1=RCM 2=METIS 3=SCOTCH 4=ASCOTCH
                                                   //       5=FSCOTCH 6=AMD 7=CAMD 8=COLAMD
                                                   //   e.g. 167 = RCM+AMD+CAMD; 3 = SCOTCH only.
                                                   //   Consumed by symbolic_phase (candidate pool);
                                                   //   an explicit list bypasses the sparse-class
                                                   //   prune and STILES_DISABLE_SCOTCH.
inline constexpr int TileTypeMode            =  3; // sTiles_set_tile_type_mode
                                                   //   0 = dense tiles only (uniform tile_size × tile_size,
                                                   //       fully populated)
                                                   //   1 = semisparse tiles (uniform tile_size × tile_size,
                                                   //       per-column active bitmap)
                                                   //   2 = non-uniform tiles (variable cell sizes driven by
                                                   //       the elimination tree; cell width = column-supernode
                                                   //       width, cell rows = row-supernode run length;
                                                   //       routes variant 0 through tools/sparse/)
                                                   //   3 = auto: run symbolic, then pick 0/1/2 from
                                                   //       occupancy, fill and degree skew.
                                                   //   Default: 3 (auto) on all platforms.
                                                   //   The per-scheme resolution is snapshotted into
                                                   //   TiledMatrix::tile_type_mode at preprocess time;
                                                   //   compute-phase code reads the scheme field, not
                                                   //   this slot.
inline constexpr int TileOrderingStrategy    =  4; // sTiles_set_tile_ordering_mode
                                                   //   0 = no tile-level reordering
                                                   //   >0 = digits 1..9 select strategies (per-partition
                                                   //        tile ordering; needs partition_sizes)
                                                   //   Default 14569 = RCM, SCOTCH, AMD, CAMD, SYMAMD.
inline constexpr int TileOrderingThreshold   =  5; // tile-ordering block size
                                                   //   -1 = auto (resolves to tile_size / 2, default)
inline constexpr int ForceNDOrdering         =  6; // 0 = auto (default), 1 = force ND
inline constexpr int InverseStorageMode      =  7; // sTiles_set_inverse_storage_mode
                                                   //   0 = dense full h×w
                                                   //   1 = diagonal dense, off-diagonal h×sa (default)
inline constexpr int UseOMP                  =  8; // sTiles_set_use_omp
                                                   //   0 = pthreads (Linux default)
                                                   //   1 = OpenMP   (macOS default)
inline constexpr int SemisparseImpl          =  9; // sTiles_set_semisparse_impl
                                                   //   0 = original
                                                   //   1 = imp1
                                                   //   2 = imp3 / imp3_serial (default)
                                                   //   3 = imp3_serial_and_sparse (1-core only)
inline constexpr int GpuCompareMode          = 10; // 0 = GPU only (default), 1 = GPU + CPU compare
inline constexpr int GpuEnable               = 11; // 0 = disabled, 1 = enabled (default)
inline constexpr int MemoryEstimateMode      = 12; // sTiles_set_memory_estimate
                                                   //   0 = skip (default), 1 = compute & print
// Slot 13 DEPRECATED (was TileOrderingMinDim; never consumed). Kept for ABI.
inline constexpr int SerialMode              = 14; // sTiles_set_serial_mode
                                                   //   0 = auto serial when num_cores==1 (default)
                                                   //   1 = always parallel kernels
inline constexpr int BandwidthMode           = 15; // sTiles_set_bw_mode
                                                   //   0 = conservative tile-width-1 (default)
                                                   //   1 = tight la-fa
// Slot 16 DEPRECATED (was TileFirstOrderingMode; never consumed). Kept for ABI.
// Slots 17, 18, 19 are reserved (formerly blocksparse mode/threshold/inv-backend;
// the blocksparse backend was removed). Default value 0; do not reuse without
// bumping the ABI version to avoid silent misinterpretation by older callers.
inline constexpr int ForceScotchOrdering     = 20; // sTiles_set_force_scotch_ordering
                                                   //   0 = adaptive selection (default)
                                                   //   1 = force SCOTCH (id=4), skip benchmark
inline constexpr int ScotchPadding           = 21; // sTiles_set_scotch_padding
                                                   //   0 = off (default; Path 2 never fires)
                                                   //   1 = on  (pads P1|P2|Sep, Path 2 engages)
inline constexpr int Path2Depth              = 22; // sTiles_set_path2_depth
                                                   //   0/1 = legacy 3-way (default), 2 = 7 regions, 3 = 15, ...
// Slot 23 reserved (formerly SolveRefinementSteps; iterative refinement was
// removed). Default 0; do not reuse without bumping the ABI version.
// Slot 24 reserved (formerly InlaPickerMode; the INLA auto-picker was removed
// along with the supernode/blocksparse subsystem). Default 0; do not reuse
// without bumping the ABI version.
inline constexpr int TreePathEnable          = 25; // sTiles_set_tree_path_enable
                                                   //   0 = off, 1 = on (default; corner_probe's 6-gate
                                                   //   predicate still decides per matrix)
inline constexpr int TreePathForce           = 26; // sTiles_set_tree_path_force
                                                   //   0 = honor gate (default), 1 = force activation
                                                   //   (kernel-correctness tests)

} // namespace param
} // namespace sTiles

/// The ONE default table. The only platform difference is UseOMP (slot 8):
/// OMP (1) on macOS (no sched-affinity for the pthreads backend there),
/// pthreads (0) on Linux. TileTypeMode (slot 3) defaults to auto (3)
/// and TreePathEnable (slot 25) to ON on all platforms.
#ifdef __APPLE__
#define STILES_DEFAULT_PARAMS_INIT {                                          \
    /* [ 0] SemisparsePruningMode   */ 0,                                     \
    /* [ 1] UserTileSize            */ -1,                                    \
    /* [ 2] OrderingMode (deprec.)  */ 0,                                     \
    /* [ 3] TileTypeMode            */ 3,                                     \
    /* [ 4] TileOrderingStrategy    */ 14569,                                 \
    /* [ 5] TileOrderingThreshold   */ -1,                                    \
    /* [ 6] ForceNDOrdering         */ 0,                                     \
    /* [ 7] InverseStorageMode      */ 1,                                     \
    /* [ 8] UseOMP                  */ 1,                                     \
    /* [ 9] SemisparseImpl          */ 2,                                     \
    /* [10] GpuCompareMode          */ 0,                                     \
    /* [11] GpuEnable               */ 1,                                     \
    /* [12] MemoryEstimateMode      */ 0,                                     \
    /* [13] reserved (deprecated)   */ 0,                                     \
    /* [14] SerialMode              */ 0,                                     \
    /* [15] BandwidthMode           */ 0,                                     \
    /* [16] reserved (deprecated)   */ 0,                                     \
    /* [17] reserved                */ 0,                                     \
    /* [18] reserved                */ 0,                                     \
    /* [19] reserved                */ 0,                                     \
    /* [20] ForceScotchOrdering     */ 0,                                     \
    /* [21] ScotchPadding           */ 0,                                     \
    /* [22] Path2Depth              */ 0,                                     \
    /* [23] reserved                */ 0,                                     \
    /* [24] reserved                */ 0,                                     \
    /* [25] TreePathEnable          */ 1,                                     \
    /* [26] TreePathForce           */ 0,                                     \
    /* 27..49: reserved             */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,          \
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,          \
                                       0, 0, 0 }
#else
#define STILES_DEFAULT_PARAMS_INIT {                                          \
    /* [ 0] SemisparsePruningMode   */ 0,                                     \
    /* [ 1] UserTileSize            */ -1,                                    \
    /* [ 2] OrderingMode (deprec.)  */ 0,                                     \
    /* [ 3] TileTypeMode            */ 3,                                     \
    /* [ 4] TileOrderingStrategy    */ 14569,                                 \
    /* [ 5] TileOrderingThreshold   */ -1,                                    \
    /* [ 6] ForceNDOrdering         */ 0,                                     \
    /* [ 7] InverseStorageMode      */ 1,                                     \
    /* [ 8] UseOMP                  */ 0,                                     \
    /* [ 9] SemisparseImpl          */ 2,                                     \
    /* [10] GpuCompareMode          */ 0,                                     \
    /* [11] GpuEnable               */ 1,                                     \
    /* [12] MemoryEstimateMode      */ 0,                                     \
    /* [13] reserved (deprecated)   */ 0,                                     \
    /* [14] SerialMode              */ 0,                                     \
    /* [15] BandwidthMode           */ 0,                                     \
    /* [16] reserved (deprecated)   */ 0,                                     \
    /* [17] reserved                */ 0,                                     \
    /* [18] reserved                */ 0,                                     \
    /* [19] reserved                */ 0,                                     \
    /* [20] ForceScotchOrdering     */ 0,                                     \
    /* [21] ScotchPadding           */ 0,                                     \
    /* [22] Path2Depth              */ 0,                                     \
    /* [23] reserved                */ 0,                                     \
    /* [24] reserved                */ 0,                                     \
    /* [25] TreePathEnable          */ 1,                                     \
    /* [26] TreePathForce           */ 0,                                     \
    /* 27..49: reserved             */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,          \
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,          \
                                       0, 0, 0 }
#endif

namespace sTiles {

/// Constexpr reference copy of the defaults (usable in static_asserts and by
/// the reset_* APIs). Same textual source as the live array's initializer.
inline constexpr int kDefaultParams[kNumParams] = STILES_DEFAULT_PARAMS_INIT;

static_assert(kDefaultParams[param::UserTileSize]  == -1, "slot 1 default is auto");
static_assert(kDefaultParams[param::TreePathEnable] == 1, "tree path defaults ON");
static_assert(kDefaultParams[kNumParams - 1] == 0, "default table fills all 50 slots");

} // namespace sTiles

// Legacy macro retained for code that still uses #if STILES_NUM_PARAMS, etc.
// New code should prefer sTiles::kNumParams.
#ifndef STILES_NUM_PARAMS
#define STILES_NUM_PARAMS 50
#endif

#endif // _STILES_PARAMS_HPP_
