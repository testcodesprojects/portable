/**
 * @file memory_api.cpp
 * @brief C-ABI memory query / estimate entry points.
 *
 * Thin wrappers around `sTiles::Memory::*` (manager totals) and
 * `sTiles::memory::*` (early estimate / fits-in-RAM). Lives in the memory
 * module so the public memory-introspection API is co-located with the
 * managers it queries; previously inline in tools/process/process.cpp.
 *
 * Public declarations live in tools/include/stiles.h.
 */

#include "../tile/meta.hpp"  // Full defs of TileMetaCore / SemisparseTileMetaCore (memory_estimate.hpp uses sizes)
#include "../common/stiles_logger.hpp"
#include "memory_namespace.hpp"
#include "memory_estimate.hpp"

namespace {

// Public-facing factorization variants: 0 = Sparse, 1 = full dense, 2 = scaled dense.
// Anything else gets clamped to 0 with a warning so callers don't silently slip
// into an internal-only code path.
int validate_variant(int variant, const char* fn) {
    if (variant >= 0 && variant <= 2) return variant;
    sTiles::Logger::warning(fn, ": invalid variant=", variant,
                            " — must be 0 (Sparse), 1 (full dense), or 2 (scaled dense). "
                            "Falling back to 0.");
    return 0;
}

} // namespace

extern "C" {

double sTiles_GetGroupMemoryUsage(int group_ID) {
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    double total = 0.0;

    // Core/general allocations
    total += sTiles::Memory::MemoryManager::getTotalMemoryAllocatedInGiBForGroup(group_ID);
    total += sTiles::Memory::TileMemoryManager::getTotalMemoryAllocatedInGiBForGroup(group_ID);
    total += sTiles::Memory::TreeMemoryManager::getTotalMemoryAllocatedInGiBForGroup(group_ID);
    total += sTiles::Memory::AlgorithmsMemoryManager::getTotalMemoryAllocatedInGiBForGroup(group_ID);
    total += sTiles::Memory::OrderingMemoryManager::getTotalMemoryAllocatedInGiBForGroup(group_ID);
    // CPU manager + TileIndexer manager return bytes; convert to GiB
    total += static_cast<double>(sTiles::Memory::CpuSmartTileMemoryManager::getMemoryAllocatedForGroup(group_ID)) / GiB;
    total += static_cast<double>(sTiles::Memory::TileIndexerMemoryManager::getMemoryAllocatedForGroup(group_ID)) / GiB;
#ifdef STILES_GPU
    total += sTiles::Memory::GpuMemoryManager::getTotalMemoryAllocatedInGiBForGroup(group_ID);
#endif
    return total;
}

double sTiles_GetGroupsMemoryUsage() {
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    double total = 0.0;
    total += sTiles::Memory::MemoryManager::getTotalMemoryAllocatedInGiB();
    total += sTiles::Memory::TileMemoryManager::getTotalMemoryAllocatedInGiB();
    total += sTiles::Memory::TreeMemoryManager::getTotalMemoryAllocatedInGiB();
    total += sTiles::Memory::AlgorithmsMemoryManager::getTotalMemoryAllocatedInGiB();
    total += sTiles::Memory::OrderingMemoryManager::getTotalMemoryAllocatedInGiB();
    total += static_cast<double>(sTiles::Memory::CpuSmartTileMemoryManager::getTotalMemoryAllocated()) / GiB;
    total += static_cast<double>(sTiles::Memory::TileIndexerMemoryManager::getTotalMemoryAllocated()) / GiB;
#ifdef STILES_GPU
    total += sTiles::Memory::GpuMemoryManager::getTotalMemoryAllocatedInGiB();
#endif
    return total;
}

/**
 * @brief Early memory estimate before preprocessing.
 *
 * Call BEFORE sTiles_preprocess_group() to check whether the matrix will fit
 * in RAM. Returns estimated memory in GB.
 */
double sTiles_estimate_memory(int n, int nnz, int tile_size, int variant,
                              int compute_inverse, int use_nested_dissection) {
    variant = validate_variant(variant, "sTiles_estimate_memory");
    const sTiles::MemoryEstimate est = sTiles::memory::estimate_memory_early(
        n, nnz, tile_size, variant,
        compute_inverse != 0,
        use_nested_dissection != 0
    );
    return est.total_gb;
}

/**
 * @brief Same as sTiles_estimate_memory but also prints a detailed breakdown.
 */
double sTiles_estimate_memory_verbose(int n, int nnz, int tile_size, int variant,
                                      int compute_inverse, int use_nested_dissection) {
    variant = validate_variant(variant, "sTiles_estimate_memory_verbose");
    const sTiles::MemoryEstimate est = sTiles::memory::estimate_memory_early(
        n, nnz, tile_size, variant,
        compute_inverse != 0,
        use_nested_dissection != 0
    );
    sTiles::memory::print_memory_estimate(est);
    return est.total_gb;
}

/**
 * @brief Check if the matrix will fit in available RAM.
 * @return 1 if it fits, 0 otherwise.
 */
int sTiles_memory_fits(int n, int nnz, int tile_size, int variant,
                       int compute_inverse, int use_nested_dissection,
                       double available_ram_gb) {
    variant = validate_variant(variant, "sTiles_memory_fits");
    const sTiles::MemoryEstimate est = sTiles::memory::estimate_memory_early(
        n, nnz, tile_size, variant,
        compute_inverse != 0,
        use_nested_dissection != 0
    );
    return sTiles::memory::memory_fits(est, available_ram_gb) ? 1 : 0;
}

} // extern "C"
