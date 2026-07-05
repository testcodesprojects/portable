/**
 * @file workspace.hpp
 * @brief Per-thread workspace for temporary tile computations.
 *
 * Provides aligned temporary storage for intermediate results during
 * tile-based matrix operations like GEMM and TRSM.
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

#ifndef WORKSPACE_HPP
#define WORKSPACE_HPP

#include "../common/stiles_types.hpp"
#include "cpuSmartTileMemoryManager.hpp"
#include "stile_compiler_hints.hpp"

#include <cstddef>
#include <cassert>
#include <cstdint>

namespace sTiles {

struct Workspace {
    int uniformTileSize = 0;
    int tile_size = 0;
    int groupID = 0;

    f64* tile = nullptr;

#ifdef SPARSE_STILES
    f64* sparse_work = nullptr;  ///< Scratch vector for sparse kernels (size = tile_size, zero-initialized)
#endif

    Workspace(int group_id, int max_dim)
        : uniformTileSize(max_dim), tile_size(max_dim), groupID(group_id), tile(nullptr), group_id_(group_id) {
        const std::size_t num_elements = static_cast<std::size_t>(max_dim) * max_dim;
        tile = CpuMemoryManager::allocate<f64>(num_elements, group_id_);
        assert(((reinterpret_cast<std::uintptr_t>(tile) & 63u) == 0) && "tile not 64B aligned");

#ifdef SPARSE_STILES
        // Sparse kernels use a scratch vector of size tile_size for scatter-dot operations.
        // Allocate zero-initialized so kernels can rely on the zero invariant from the start.
        sparse_work = CpuMemoryManager::allocateZero<f64>(static_cast<size_t>(max_dim), group_id_);
        assert(((reinterpret_cast<std::uintptr_t>(sparse_work) & 63u) == 0) && "sparse_work not 64B aligned");
#endif
    }

    ~Workspace() = default;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    inline double* aligned_tile() {
        ASSUME_ALIGNED_64(tile);
        return tile;
    }

    inline const double* aligned_tile() const {
        double* ptr = tile;
        ASSUME_ALIGNED_64(ptr);
        return ptr;
    }

#ifdef SPARSE_STILES
    /// Returns the sparse scratch vector (tile_size, zero-initialized).
    /// Used by sparse kernels for per-column scatter-dot operations.
    inline double* aligned_sparse_work() {
        ASSUME_ALIGNED_64(sparse_work);
        return sparse_work;
    }
#endif

private:
    int group_id_;
};

} // namespace sTiles

#endif
