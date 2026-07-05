/**
 * @file stiles_params.hpp
 * @brief Single source of truth for the sTiles control-parameter array.
 *
 * Replaces the magic indices `params[3]`, `params[19]`, etc. scattered across
 * the codebase with named constants in `sTiles::param`. Adding a new
 * parameter is a one-file edit:
 *   1. Bump kNumParams.
 *   2. Add a name in `sTiles::param`.
 *   3. Append a default value to kDefaultParams (and the macOS variant if
 *      it differs).
 *   4. Document the slot in the comment table below.
 *
 * IMPORTANT: the *order* of slots is part of the ABI exposed via
 * sTiles_get_control_params / sTiles_set_control_params. Do not reorder
 * existing entries — only append new ones at the end and bump kNumParams.
 */

#ifndef _STILES_PARAMS_HPP_
#define _STILES_PARAMS_HPP_

namespace sTiles {

/// Size of the control-parameter array. Bump when appending a new slot.
/// Slots 0..24 are defined; slots 25..49 are reserved for future use
/// (default 0). Reserve a slot by giving it a name in `sTiles::param`.
inline constexpr int kNumParams = 50;
inline constexpr int kNumDefinedParams = 25;

/// Named indices for the control-parameter array.
///
/// Use `params[sTiles::param::TileTypeMode]` instead of `params[3]`.
namespace param {

// ---------------------------------------------------------------------------
// Slot index | Setter                              | Meaning
// ---------------------------------------------------------------------------
inline constexpr int SemisparsePruningMode   =  0; // sTiles_set_semisparse_param
                                                   //   0 = no pruning
                                                   //   1 = prune_single_zero_active_column_range
                                                   //   2 = prune_zero_semisparse_tiles_range
                                                   //   3 = prune_zero_semisparse_columns_range
inline constexpr int UserTileSize            =  1; // -1 = auto-detect
inline constexpr int OrderingMode            =  2; // sTiles_set_ordering_mode
                                                   //   0 = single ordering from call.ordering_strategy
                                                   //   1 = parallel RCM + ND, pick best
                                                   //   2 = parallel RCM + ND + SCOTCH, pick best
                                                   // (sentinel 14569 = "not configured by user")
inline constexpr int TileTypeMode            =  3; // sTiles_set_tile_type_mode
                                                   //   0 = dense tiles only (uniform tile_size × tile_size,
                                                   //       fully populated)
                                                   //   1 = semisparse tiles (uniform tile_size × tile_size,
                                                   //       per-column active bitmap)
                                                   //   2 = non-uniform tiles (variable cell sizes driven by
                                                   //       the elimination tree; cell width = column-supernode
                                                   //       width, cell rows = row-supernode run length;
                                                   //       routes variant 0 through tools/sparse/)
                                                   //   3 = auto: run symbolic, then pick mode 1 or 2 based on
                                                   //       fill = nnz(L) / nnz(A). Low fill → semisparse;
                                                   //       high fill → non-uniform. Threshold ≈ 3.5×.
inline constexpr int TileOrderingStrategy    =  4; // (sentinel 14569 = unset)
inline constexpr int TileOrderingThreshold   =  5; // tile-ordering size threshold
inline constexpr int ForceNDOrdering         =  6; // 0 = auto, 1 = force ND
inline constexpr int InverseStorageMode      =  7; // sTiles_set_inverse_storage_mode
                                                   //   0 = dense full h×w
                                                   //   1 = diagonal dense, off-diagonal h×sa
inline constexpr int UseOMP                  =  8; // sTiles_set_use_omp
                                                   //   0 = pthreads (Linux default)
                                                   //   1 = OpenMP   (macOS default)
inline constexpr int SemisparseImpl          =  9; // sTiles_set_semisparse_impl
                                                   //   0 = original
                                                   //   1 = imp1
                                                   //   2 = imp3 / imp3_serial (default)
                                                   //   3 = imp3_serial_and_sparse (1-core only)
inline constexpr int GpuCompareMode          = 10; // 0 = GPU only, 1 = GPU + CPU compare
inline constexpr int GpuEnable               = 11; // 0 = disabled, 1 = enabled
inline constexpr int MemoryEstimateMode      = 12; // sTiles_set_memory_estimate
                                                   //   0 = skip, 1 = compute & print
inline constexpr int TileOrderingMinDim      = 13; // skip tile-level ordering when tiles_dim < this
inline constexpr int SerialMode              = 14; // sTiles_set_serial_mode
                                                   //   0 = auto serial when num_cores==1
                                                   //   1 = always parallel kernels
inline constexpr int BandwidthMode           = 15; // sTiles_set_bw_mode
                                                   //   0 = conservative tile-width-1
                                                   //   1 = tight la-fa
inline constexpr int TileFirstOrderingMode   = 16; // sTiles_set_tile_first_ordering_mode
// Slots 17, 18, 19 are reserved (formerly blocksparse mode/threshold/inv-backend;
// the blocksparse backend was removed). Default value 0; do not reuse without
// bumping the ABI version to avoid silent misinterpretation by older callers.
inline constexpr int ForceScotchOrdering     = 20; // sTiles_set_force_scotch_ordering
                                                   //   0 = adaptive selection (default)
                                                   //   1 = force SCOTCH (id=4), skip benchmark
inline constexpr int ScotchPadding           = 21; // sTiles_set_scotch_padding
                                                   //   0 = off (Path 2 never fires)
                                                   //   1 = on  (pads P1|P2|Sep, Path 2 engages)
inline constexpr int Path2Depth              = 22; // sTiles_set_path2_depth
                                                   //   0 = legacy 3-way, 2 = 7 regions, 3 = 15, 4 = 31, ...
// Slot 23 reserved (formerly SolveRefinementSteps; iterative refinement was
// removed). Default 0; do not reuse without bumping the ABI version.
// Slot 24 reserved (formerly InlaPickerMode; the INLA auto-picker was removed
// along with the supernode/blocksparse subsystem). Default 0; do not reuse
// without bumping the ABI version.

} // namespace param

/// Default values, indexed by `sTiles::param::*`. The macOS variant differs in
/// TileTypeMode (slot 3): macOS = 0 (dense), Linux = 1 (semisparse); and in
/// UseOMP (slot 8): macOS = 1 (OMP, no sched-affinity for pthreads on macOS),
/// Linux = 0 (pthreads).
#ifdef __APPLE__
inline constexpr int kDefaultParams[kNumParams] = {
    /* SemisparsePruningMode   */ 0,
    /* UserTileSize            */ -1,
    /* OrderingMode            */ 14569,
    /* TileTypeMode            */ 0,
    /* TileOrderingStrategy    */ 14569,
    /* TileOrderingThreshold   */ -1,
    /* ForceNDOrdering         */ 0,
    /* InverseStorageMode      */ 1,
    /* UseOMP                  */ 1,
    /* SemisparseImpl          */ 2,
    /* GpuCompareMode          */ 0,
    /* GpuEnable               */ 1,
    /* MemoryEstimateMode      */ 0,
    /* TileOrderingMinDim      */ 100,
    /* SerialMode              */ 0,
    /* BandwidthMode           */ 0,
    /* TileFirstOrderingMode   */ 0,
    /* reserved (slot 17)      */ 0,
    /* reserved (slot 18)      */ 0,
    /* reserved (slot 19)      */ 0,
    /* ForceScotchOrdering     */ 0,
    /* ScotchPadding           */ 0,
    /* Path2Depth              */ 0,
    /* reserved (slot 23)      */ 0,
    /* reserved (slot 24)      */ 0,
    /* 25..49: reserved        */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0,
};
#else
inline constexpr int kDefaultParams[kNumParams] = {
    /* SemisparsePruningMode   */ 0,
    /* UserTileSize            */ -1,
    /* OrderingMode            */ 14569,
    /* TileTypeMode            */ 1,
    /* TileOrderingStrategy    */ 14569,
    /* TileOrderingThreshold   */ -1,
    /* ForceNDOrdering         */ 0,
    /* InverseStorageMode      */ 1,
    /* UseOMP                  */ 0,
    /* SemisparseImpl          */ 2,
    /* GpuCompareMode          */ 0,
    /* GpuEnable               */ 1,
    /* MemoryEstimateMode      */ 0,
    /* TileOrderingMinDim      */ 100,
    /* SerialMode              */ 0,
    /* BandwidthMode           */ 0,
    /* TileFirstOrderingMode   */ 0,
    /* reserved (slot 17)      */ 0,
    /* reserved (slot 18)      */ 0,
    /* reserved (slot 19)      */ 0,
    /* ForceScotchOrdering     */ 0,
    /* ScotchPadding           */ 0,
    /* Path2Depth              */ 0,
    /* reserved (slot 23)      */ 0,
    /* reserved (slot 24)      */ 0,
    /* 25..49: reserved        */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0,
};
#endif

} // namespace sTiles

// Legacy macro retained for code that still uses #if STILES_NUM_PARAMS, etc.
// New code should prefer sTiles::kNumParams.
#ifndef STILES_NUM_PARAMS
#define STILES_NUM_PARAMS 50
#endif

#endif // _STILES_PARAMS_HPP_
