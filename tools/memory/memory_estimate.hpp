/**
 * @file memory_estimate.hpp
 * @brief Memory estimation utilities for sTiles computations.
 *
 * Provides functions to estimate memory requirements before and after
 * symbolic factorization to help users determine if their matrix fits in RAM.
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

#ifndef STILES_MEMORY_ESTIMATE_HPP
#define STILES_MEMORY_ESTIMATE_HPP

#include "../common/stiles_structs.hpp"
#include "../common/stiles_logger.hpp"
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace sTiles {

/**
 * @brief Memory estimation utilities for sTiles
 *
 * Provides both early estimates (before preprocessing) and exact calculations
 * (after symbolic factorization) to help users determine if their matrix
 * will fit in available RAM.
 */
struct MemoryEstimate {
    // Factorization memory (diagonal tiles use banded format)
    double banded_diagonal_gb;    // Memory for banded diagonal tiles (factorization)
    double offdiag_semisparse_gb; // Memory for off-diagonal tiles (semisparse format)
    double semisparse_indices_gb; // Memory for aind/acol arrays (GPU)

    // Inverse memory (diagonal tiles use full dense format)
    double dense_diagonal_gb;     // Memory for dense diagonal tiles (inverse only)
    double inverse_tiles_gb;      // Memory for inverse tiles (if compute_inverse)
    double saved_tiles_gb;        // Memory for saved tiles (if compute_inverse)

    // Legacy fields (for compatibility)
    double dense_tiles_gb;        // Memory for all tiles as dense (legacy estimate)
    double semisparse_tiles_gb;   // Memory for semisparse tiles (legacy)

    double metadata_gb;           // Memory for tile metadata, lookup tables
    double workspace_gb;          // Memory for thread workspaces
    double symbolic_pattern_gb;   // L_rowind (nnz_factor*4) + L_colptr ((dim+1)*8), ordering-layer
    double solve_pack_gb;         // Packed-CSC solve buffers: L_values (nnz_factor*8) [+ L_src if <= ceiling]
    double total_gb;              // Steady factor resident (chol only, no packed solve)
    double total_solve_gb;        // Factor + packed solve resident (total_gb + solve_pack_gb)
    double total_peak_gb;         // Max RSS to provision (steady + larger of {analyze COO transient, solve pack})
    double gpu_total_gb;          // GPU-specific memory requirement
    double gpu_fac_only_gb;       // GPU memory for factorization only (no inverse)

    int num_active_tiles;         // Number of active tiles (exact or estimated)
    int num_diagonal_tiles;       // Number of diagonal tiles
    int num_offdiag_tiles;        // Number of off-diagonal tiles
    int tile_size;                // Tile size used
    int matrix_dim;               // Matrix dimension
    bool is_estimate;             // true = early estimate, false = exact calculation

    // Diagnostic info (for debugging semisparse vs dense comparison)
    double avg_upper_bw;          // Average bandwidth for diagonal tiles
    double avg_sa;                // Average active columns for off-diagonal tiles
};

namespace memory {

/**
 * @brief Early memory estimate based on matrix parameters (before preprocessing)
 *
 * Call this BEFORE sTiles_preprocess_group() to check if the matrix will fit.
 * This is a conservative estimate based on typical fill-in ratios.
 *
 * @param n Matrix dimension (number of rows/columns)
 * @param nnz Number of non-zeros in the input matrix
 * @param tile_size Tile size (use 0 for auto-detection, typically 40 for CPU)
 * @param variant Factorization variant (0=semisparse, 1=dense, 2=scaled)
 * @param compute_inverse Whether inverse will be computed (multiplies by 3)
 * @param use_nested_dissection Whether ND ordering will be used (affects fill-in estimate)
 * @param tile_type_mode Tile type mode (0=dense, 1=semisparse, 3=both). Default 0.
 *                       This affects GPU memory calculation:
 *                       - Mode 0 (dense): GPU uses only dense tiles
 *                       - Mode 1 (semisparse): GPU uses dense diagonals + semisparse off-diagonals
 * @return MemoryEstimate structure with estimated memory usage
 */
inline MemoryEstimate estimate_memory_early(
    int n,
    int nnz,
    int tile_size,
    int variant,
    bool compute_inverse,
    bool use_nested_dissection = true,
    int tile_type_mode = 0
) {
    MemoryEstimate est = {};
    est.matrix_dim = n;
    est.is_estimate = true;

    // Auto-detect tile size if not specified
    if (tile_size <= 0) {
        #ifdef STILES_GPU
        tile_size = 600;
        #else
        tile_size = 40;
        #endif
    }
    est.tile_size = tile_size;

    // Number of tiles per side
    const int num_tiles = (n + tile_size - 1) / tile_size;

    // Estimate fill-in ratio based on ordering and matrix characteristics
    // These are empirical estimates for typical sparse matrices
    double fill_in_ratio;
    if (use_nested_dissection) {
        // ND typically gives 10-30x fill-in for 3D problems
        // Use density-based heuristic
        const double density = static_cast<double>(nnz) / (static_cast<double>(n) * n);
        if (density > 0.01) {
            fill_in_ratio = 5.0;  // Already fairly dense
        } else if (density > 0.001) {
            fill_in_ratio = 15.0; // Moderate sparsity
        } else {
            fill_in_ratio = 25.0; // Very sparse (typical for large 3D problems)
        }
    } else {
        // RCM ordering typically gives higher fill-in
        fill_in_ratio = 40.0;
    }

    const double bytes_per_double = 8.0;
    const double gb_divisor = 1024.0 * 1024.0 * 1024.0;

    if (variant == 1) {
        // Variant 1: Full dense - single tile covering entire matrix
        // Memory = n^2 * 8 bytes
        const double n_sq = static_cast<double>(n) * static_cast<double>(n);
        est.num_active_tiles = 1;
        est.dense_tiles_gb = (n_sq * bytes_per_double) / gb_divisor;

    } else if (variant == 2) {
        // Variant 2: Scaled dense - upper triangular tiles
        // All tiles in upper triangle are allocated
        const int triangular_tiles = (num_tiles * (num_tiles + 1)) / 2;
        est.num_active_tiles = triangular_tiles;

        const double tile_elements = static_cast<double>(tile_size) * static_cast<double>(tile_size);
        est.dense_tiles_gb = (static_cast<double>(triangular_tiles) * tile_elements * bytes_per_double) / gb_divisor;

    } else {
        // Variant 0 (and 3): Semisparse - only active tiles allocated
        // Estimate active tiles based on fill-in
        const double estimated_factor_nnz = static_cast<double>(nnz) * fill_in_ratio;
        const double tile_elements = static_cast<double>(tile_size) * static_cast<double>(tile_size);

        // Estimate number of active tiles
        // Each tile can hold tile_size^2 elements, but fill varies
        // Use a coverage factor (tiles are typically 30-70% full on average)
        const double avg_tile_fill = 0.5;
        const int estimated_active_tiles = static_cast<int>(
            std::ceil(estimated_factor_nnz / (tile_elements * avg_tile_fill))
        );

        // Cap at maximum possible (upper triangular)
        const int max_tiles = (num_tiles * (num_tiles + 1)) / 2;
        est.num_active_tiles = std::min(estimated_active_tiles, max_tiles);

        est.dense_tiles_gb = (static_cast<double>(est.num_active_tiles) * tile_elements * bytes_per_double) / gb_divisor;

        // Semisparse tiles (roughly 20-50% of dense in typical cases)
        est.semisparse_tiles_gb = est.dense_tiles_gb * 0.3;

        // Semisparse index arrays (aind, acol) - estimate based on active columns
        // Assume average sa = tile_size * 0.3, each tile needs 2*sa ints + 2 metadata ints
        const double avg_sa = static_cast<double>(tile_size) * 0.3;
        const double indices_per_tile = 2.0 * avg_sa + 2.0; // aind + acol + (sa, upper_bw)
        est.semisparse_indices_gb = (static_cast<double>(est.num_active_tiles) * indices_per_tile * 4.0) / gb_divisor;
    }

    // Inverse and saved tiles (if computing inverse)
    if (compute_inverse) {
        est.inverse_tiles_gb = est.dense_tiles_gb;
        est.saved_tiles_gb = est.dense_tiles_gb;
    }

    // Metadata overhead (lookup tables, tile metadata, etc.)
    // Roughly: 3 * nnz * 4 bytes (tile_index_lookup, withinTileRow, withinTileCol)
    // Plus metadata per tile
    const double lookup_bytes = 3.0 * static_cast<double>(nnz) * 4.0;
    const double metadata_per_tile = 100.0; // approximate bytes per tile for metadata
    est.metadata_gb = (lookup_bytes + static_cast<double>(est.num_active_tiles) * metadata_per_tile) / gb_divisor;

    // Workspace memory (thread-local buffers)
    // Typically tile_size^2 * num_threads * some factor
    const int assumed_threads = 8;
    const double workspace_per_thread = static_cast<double>(tile_size) * static_cast<double>(tile_size) * bytes_per_double * 4.0;
    est.workspace_gb = (static_cast<double>(assumed_threads) * workspace_per_thread) / gb_divisor;

    // Total CPU memory
    // Note: semisparse is always allocated on CPU for preprocessing, regardless of tile_type_mode
    est.total_gb = est.dense_tiles_gb + est.inverse_tiles_gb + est.saved_tiles_gb +
                   est.semisparse_tiles_gb + est.semisparse_indices_gb +
                   est.metadata_gb + est.workspace_gb;

    // GPU memory calculation depends on tile_type_mode:
    // - Mode 0 (dense): GPU uses only dense tiles - semisparse is NOT used for GPU computations
    // - Mode 1 (semisparse): GPU uses dense diagonal tiles + semisparse off-diagonal tiles
    // - Mode 3 (both): same as semisparse for GPU purposes
    const bool use_semisparse_on_gpu = (tile_type_mode == 1 || tile_type_mode == 2);

    if (use_semisparse_on_gpu) {
        // Semisparse GPU mode: diagonal tiles (dense) + off-diagonal (semisparse format)
        est.gpu_total_gb = est.dense_tiles_gb + est.semisparse_tiles_gb + est.semisparse_indices_gb;
    } else {
        // Dense GPU mode: only dense tiles needed - semisparse is for CPU preprocessing only
        est.gpu_total_gb = est.dense_tiles_gb;
    }

    // Add inverse tiles if computing inverse on GPU
    if (compute_inverse) {
        est.gpu_total_gb += est.inverse_tiles_gb;
    }

    return est;
}

/**
 * @brief Exact memory calculation after symbolic factorization
 *
 * Call this AFTER symbolic factorization when numActiveTiles is known exactly.
 * This is called automatically inside sTiles_preprocess_group() before allocation.
 *
 * @param scheme The TiledMatrix after symbolic factorization
 * @param call_info The sTiles_call with factorization parameters
 * @param tile_type_mode Tile type mode (0=dense, 1=semisparse, 3=both). Default -1 uses dense mode.
 *                       This affects GPU memory calculation:
 *                       - Mode 0 (dense): GPU uses only dense tiles
 *                       - Mode 1 (semisparse): GPU uses dense diagonals + semisparse off-diagonals
 * @return MemoryEstimate structure with exact memory usage
 */
inline MemoryEstimate calculate_memory_exact(const TiledMatrix* scheme, const sTiles_call* call_info, int tile_type_mode = -1) {
    MemoryEstimate est = {};

    if (!scheme || !call_info) {
        return est;
    }

    est.matrix_dim = scheme->dim;
    est.tile_size = scheme->tile_size;
    est.num_active_tiles = scheme->numActiveTiles;
    est.is_estimate = false;

    const double bytes_per_double = 8.0;
    const double gb_divisor = 1024.0 * 1024.0 * 1024.0;
    const int variant = call_info->factorization_variant;

    if (variant == 1) {
        // Variant 1: Full dense - single tile = n^2
        const double n_sq = static_cast<double>(scheme->dim) * static_cast<double>(scheme->dim);
        est.dense_tiles_gb = (n_sq * bytes_per_double) / gb_divisor;

    } else if (variant == 2) {
        // Variant 2: Scaled dense - all upper triangular tiles
        const double tile_elements = static_cast<double>(scheme->tile_size) * static_cast<double>(scheme->tile_size);
        est.dense_tiles_gb = (static_cast<double>(scheme->numActiveTiles) * tile_elements * bytes_per_double) / gb_divisor;

    } else {
        // Variant 0/3: Semisparse
        // Calculate exact memory based on actual tile dimensions
        // Separate diagonal tiles (banded for fac, dense for inv) from off-diagonal (semisparse)
        double total_dense_elements = 0.0;
        double banded_diagonal_elements = 0.0;
        double dense_diagonal_elements = 0.0;
        double offdiag_semisparse_elements = 0.0;
        double total_aind_acol_ints = 0.0;
        int num_diagonal = 0;
        int num_offdiag = 0;
        double sum_upper_bw = 0.0;
        double sum_sa = 0.0;

        if (scheme->tileMetaCore && scheme->semisparseTileMetaCore) {
            for (int idx = 0; idx < scheme->numActiveTiles; ++idx) {
                const TileMetaCore& meta = scheme->tileMetaCore[idx];
                const SemisparseTileMetaCore& semi = scheme->semisparseTileMetaCore[idx];
                const int h = (meta.height > 0) ? meta.height : scheme->tile_size;
                const int w = (meta.width > 0) ? meta.width : scheme->tile_size;

                // Use diagonal_bmapper if available (authoritative), fallback to meta.row == meta.col
                const bool is_diagonal = (scheme->diagonal_bmapper && scheme->diagonal_bmapper[idx])
                                       || (meta.row == meta.col);

                // Legacy: total dense elements (for dense mode comparison)
                total_dense_elements += static_cast<double>(h) * static_cast<double>(w);

                if (is_diagonal) {
                    // Diagonal tile: banded for factorization, dense for inverse
                    num_diagonal++;
                    // Banded storage: (upper_bw + 1) × height for LAPACK banded format
                    // upper_bw is the number of superdiagonals
                    const int bw = (semi.upper_bw > 0) ? (semi.upper_bw + 1) : w; // +1 for LAPACK format
                    sum_upper_bw += static_cast<double>(semi.upper_bw);
                    banded_diagonal_elements += static_cast<double>(bw) * static_cast<double>(h);
                    // Dense storage: full h × w (for inverse)
                    dense_diagonal_elements += static_cast<double>(h) * static_cast<double>(w);
                } else {
                    // Off-diagonal tile: semisparse format (sa active columns × height)
                    // sa is the number of active (non-zero) columns in the tile
                    num_offdiag++;
                    const int active_cols = (semi.sa > 0) ? semi.sa : w; // fallback to full width
                    sum_sa += static_cast<double>(active_cols);
                    offdiag_semisparse_elements += static_cast<double>(active_cols) * static_cast<double>(h);
                }

                // GPU index arrays: aind + acol per tile (only for off-diagonal in semisparse mode)
                // Diagonal tiles don't need aind/acol - they use banded format
                if (!is_diagonal) {
                    total_aind_acol_ints += static_cast<double>(semi.aind.size());
                    total_aind_acol_ints += static_cast<double>(semi.acol.size());
                }
            }
            // Calculate averages
            est.avg_upper_bw = (num_diagonal > 0) ? (sum_upper_bw / num_diagonal) : 0.0;
            est.avg_sa = (num_offdiag > 0) ? (sum_sa / num_offdiag) : 0.0;
            est.num_offdiag_tiles = num_offdiag;
        } else if (scheme->tileMetaCore) {
            // Fallback without semisparse metadata
            for (int idx = 0; idx < scheme->numActiveTiles; ++idx) {
                const TileMetaCore& meta = scheme->tileMetaCore[idx];
                const int h = (meta.height > 0) ? meta.height : scheme->tile_size;
                const int w = (meta.width > 0) ? meta.width : scheme->tile_size;
                total_dense_elements += static_cast<double>(h) * static_cast<double>(w);
            }
        } else {
            // Fallback: assume all tiles are tile_size x tile_size
            const double tile_elements = static_cast<double>(scheme->tile_size) * static_cast<double>(scheme->tile_size);
            total_dense_elements = static_cast<double>(scheme->numActiveTiles) * tile_elements;
        }

        est.num_diagonal_tiles = num_diagonal;
        est.dense_tiles_gb = (total_dense_elements * bytes_per_double) / gb_divisor;
        est.banded_diagonal_gb = (banded_diagonal_elements * bytes_per_double) / gb_divisor;
        est.dense_diagonal_gb = (dense_diagonal_elements * bytes_per_double) / gb_divisor;
        est.offdiag_semisparse_gb = (offdiag_semisparse_elements * bytes_per_double) / gb_divisor;
        est.semisparse_tiles_gb = est.offdiag_semisparse_gb; // Legacy compatibility
        est.semisparse_indices_gb = (total_aind_acol_ints * 4.0) / gb_divisor; // int = 4 bytes
    }

    // Inverse and saved tiles
    if (scheme->compute_inverse) {
        est.inverse_tiles_gb = est.dense_tiles_gb;
        est.saved_tiles_gb = est.dense_tiles_gb;
    }

    // Metadata
    const double lookup_bytes = 3.0 * static_cast<double>(scheme->original_nnz) * 4.0;
    const double metadata_per_tile = 100.0;
    est.metadata_gb = (lookup_bytes + static_cast<double>(scheme->numActiveTiles) * metadata_per_tile) / gb_divisor;

    // Workspace
    const int num_threads = call_info->num_cores > 0 ? call_info->num_cores : 1;
    const double workspace_per_thread = static_cast<double>(scheme->tile_size) * static_cast<double>(scheme->tile_size) * bytes_per_double * 4.0;
    est.workspace_gb = (static_cast<double>(num_threads) * workspace_per_thread) / gb_divisor;

    // Symbolic pattern (L_rowind int + L_colptr int64): allocated by the ordering
    // layer, NOT in any tile array, so it must be added explicitly. Sparse-tiled
    // variants (0/3) only; dense variants (1/2) carry no CSC pattern.
    if (variant != 1 && variant != 2 && scheme->nnz_factor > 0) {
        est.symbolic_pattern_gb =
            (static_cast<double>(scheme->nnz_factor) * 4.0
             + static_cast<double>(scheme->dim + 1) * 8.0) / gb_divisor;
    }

    // Actual factor tile storage. For the sparse-tiled variants the tiles are
    // allocated by chunked_tile_element_count as banded (diagonal, (upper_bw+1)*h)
    // + semisparse (off-diagonal, sa*h) — i.e. banded_diagonal_gb + offdiag_semisparse_gb.
    // dense_tiles_gb (all tiles priced as h*w) is NOT allocated in these modes; it is
    // only the legacy dense-mode comparison and the basis for the (dense) inverse
    // tiles, so it must NOT be summed into the factor total. Variants 1/2 do use the
    // full dense tiles.
    const double factor_tiles_gb = (variant == 1 || variant == 2)
        ? est.dense_tiles_gb
        : (est.banded_diagonal_gb + est.offdiag_semisparse_gb);

    // Packed-CSC solve buffers, allocated by sTiles_packing for the fast solve:
    //   L_values = nnz_factor doubles (always packed for semisparse/dense),
    //   L_src    = nnz_factor pointers, ONLY when it fits the pack-cache ceiling
    //              (csc_solve.cpp g_l_src_max_bytes, default 2 GiB); above that it
    //              falls back to the per-entry kernel and L_src is NOT allocated
    //              (e.g. bern's 48 GB exceeds 2 GiB, so L_src is skipped).
    if (variant != 1 && variant != 2 && scheme->nnz_factor > 0) {
        const double l_values_bytes = static_cast<double>(scheme->nnz_factor) * 8.0;
        const long long l_src_bytes = static_cast<long long>(scheme->nnz_factor) * 8LL;
        const long long l_src_ceiling = 2LL << 30;   // matches g_l_src_max_bytes default
        const double l_src_gb = (l_src_bytes <= l_src_ceiling)
                              ? static_cast<double>(scheme->nnz_factor) * 8.0 : 0.0;
        est.solve_pack_gb = (l_values_bytes + l_src_gb) / gb_divisor;
    }

    // STEADY factor resident (chol only, no packed solve). inverse/saved are 0
    // unless compute_inverse (selinv), where they correctly use the dense basis.
    est.total_gb = factor_tiles_gb
                 + est.inverse_tiles_gb + est.saved_tiles_gb
                 + est.semisparse_indices_gb
                 + est.symbolic_pattern_gb
                 + est.metadata_gb + est.workspace_gb;

    // Factor + packed solve resident (what INLA holds for repeated chol+solve).
    est.total_solve_gb = est.total_gb + est.solve_pack_gb;

    // Max RSS to provision. The two large auxiliary allocations are SEQUENTIAL and
    // never coexist: the ANALYZE-time COO transient (result.ri + L_row + L_col =
    // 3*nnz_factor ints) is freed before sTiles_packing allocates the solve buffers.
    // So the peak is the steady factor plus the LARGER of the two.
    const double analyze_transient_gb = (variant != 1 && variant != 2)
        ? (3.0 * static_cast<double>(scheme->nnz_factor) * 4.0) / gb_divisor
        : 0.0;
    est.total_peak_gb = est.total_gb + std::max(analyze_transient_gb, est.solve_pack_gb);

    // GPU memory calculation depends on tile_type_mode:
    // - Mode 0 (dense): GPU uses only dense tiles - semisparse is NOT used for GPU computations
    // - Mode 1 (semisparse): GPU uses banded diagonal + semisparse off-diagonal tiles
    // - Mode 3 (both): same as semisparse for GPU purposes
    // - Mode -1 (default/unspecified): assume dense mode for backwards compatibility
    const bool use_semisparse_on_gpu = (tile_type_mode == 1 || tile_type_mode == 2);

    if (use_semisparse_on_gpu) {
        // Semisparse GPU mode:
        // Factorization: banded diagonal + semisparse off-diagonal + indices
        est.gpu_fac_only_gb = est.banded_diagonal_gb + est.offdiag_semisparse_gb + est.semisparse_indices_gb;
        est.gpu_total_gb = est.gpu_fac_only_gb;

        // If computing inverse, add dense diagonal tiles (inverse uses full dense for diagonals)
        if (scheme->compute_inverse) {
            est.gpu_total_gb += est.dense_diagonal_gb;
        }
    } else {
        // Dense GPU mode: all tiles stored as full dense
        est.gpu_fac_only_gb = est.dense_tiles_gb;
        est.gpu_total_gb = est.dense_tiles_gb;

        // Add inverse tiles if computing inverse
        if (scheme->compute_inverse) {
            est.gpu_total_gb += est.inverse_tiles_gb;
        }
    }

    return est;
}

/**
 * @brief Format memory size with appropriate unit (GB, MB, KB)
 */
inline std::string format_memory_size(double gb) {
    std::ostringstream ss;
    ss << std::fixed;
    if (gb >= 1.0) {
        ss << std::setprecision(2) << gb << " GB";
    } else if (gb >= 0.001) {
        ss << std::setprecision(2) << (gb * 1024.0) << " MB";
    } else if (gb > 0.0) {
        ss << std::setprecision(2) << (gb * 1024.0 * 1024.0) << " KB";
    } else {
        ss << "0 KB";
    }
    return ss.str();
}

/**
 * @brief Print memory estimate to console using timing log format
 */
inline void print_memory_estimate(const MemoryEstimate& est) {
    const char* type_label = est.is_estimate ? "Memory estimate" : "Memory exact";

    Logger::timing("│   ↪ ", type_label, " (Primary Call): n=", est.matrix_dim,
                   ", tile_size=", est.tile_size, ", active_tiles=", est.num_active_tiles,
                   " (", est.num_diagonal_tiles, " diag, ", est.num_offdiag_tiles, " off-diag)");

    // Diagnostic: show avg bandwidth and active columns
    if (est.avg_upper_bw > 0 || est.avg_sa > 0) {
        Logger::timing("│       avg_upper_bw=", static_cast<int>(est.avg_upper_bw),
                       ", avg_sa=", static_cast<int>(est.avg_sa),
                       " (tile_size=", est.tile_size, ")");
    }

    // Factorization memory breakdown
    if (est.banded_diagonal_gb > 0 || est.offdiag_semisparse_gb > 0) {
        Logger::timing("│       [Factorization - Semisparse mode]");
        Logger::timing("│         banded_diagonal=", format_memory_size(est.banded_diagonal_gb));
        Logger::timing("│         offdiag_semisparse=", format_memory_size(est.offdiag_semisparse_gb));
        if (est.semisparse_indices_gb > 0) {
            Logger::timing("│         semisparse_indices=", format_memory_size(est.semisparse_indices_gb));
        }
        // Show comparison: semisparse total vs dense
        double semisparse_fac = est.banded_diagonal_gb + est.offdiag_semisparse_gb;
        Logger::timing("│         → semisparse factorization=", format_memory_size(semisparse_fac),
                       " vs dense=", format_memory_size(est.dense_tiles_gb));
    } else {
        // Legacy output for dense mode
        Logger::timing("│       dense_tiles=", format_memory_size(est.dense_tiles_gb));
    }

    // Inverse memory breakdown
    if (est.inverse_tiles_gb > 0) {
        Logger::timing("│       [Inverse]");
        Logger::timing("│         dense_diagonal=", format_memory_size(est.dense_diagonal_gb));
        Logger::timing("│         inverse_tiles=", format_memory_size(est.inverse_tiles_gb));
        Logger::timing("│         saved_tiles=", format_memory_size(est.saved_tiles_gb));
    }

    Logger::timing("│       metadata=", format_memory_size(est.metadata_gb),
                   ", workspaces=", format_memory_size(est.workspace_gb));
    if (est.symbolic_pattern_gb > 0) {
        Logger::timing("│       symbolic_pattern (L_rowind+L_colptr)=",
                       format_memory_size(est.symbolic_pattern_gb));
    }
    if (est.solve_pack_gb > 0) {
        Logger::timing("│       solve_pack (L_values[+L_src])=",
                       format_memory_size(est.solve_pack_gb));
    }
    Logger::timing("│       TOTAL (factor only, steady)=", format_memory_size(est.total_gb));
    if (est.total_solve_gb > est.total_gb) {
        Logger::timing("│       TOTAL (factor+solve, steady)=", format_memory_size(est.total_solve_gb));
    }
    Logger::timing("│       TOTAL (PEAK, provision --mem)=", format_memory_size(est.total_peak_gb));

    // GPU memory comparison
    if (est.gpu_fac_only_gb > 0) {
        Logger::timing("│       GPU (fac only)=", format_memory_size(est.gpu_fac_only_gb),
                       " [dense would be: ", format_memory_size(est.dense_tiles_gb), "]");
    }
    if (est.gpu_total_gb > 0 && est.gpu_total_gb != est.gpu_fac_only_gb) {
        Logger::timing("│       GPU (fac+inv)=", format_memory_size(est.gpu_total_gb));
    } else if (est.gpu_total_gb > 0 && est.gpu_fac_only_gb == 0) {
        Logger::timing("│       GPU required=", format_memory_size(est.gpu_total_gb));
    }
}

/**
 * @brief Check if estimated memory fits in available RAM
 *
 * @param est Memory estimate
 * @param available_gb Available RAM in GB
 * @param safety_margin Fraction of RAM to keep free (default 0.1 = 10%)
 * @return true if memory fits, false otherwise
 */
inline bool memory_fits(const MemoryEstimate& est, double available_gb, double safety_margin = 0.1) {
    const double usable_gb = available_gb * (1.0 - safety_margin);
    // Must fit the PEAK (ANALYZE-time transient), not just the steady factor.
    // Early estimates leave total_peak_gb=0, so fall back to total_gb via max().
    return std::max(est.total_gb, est.total_peak_gb) <= usable_gb;
}

} // namespace memory
} // namespace sTiles

#endif // STILES_MEMORY_ESTIMATE_HPP
