/**
 * @file stiles_policy.hpp
 * @brief Runtime auto-selection rules ("pickers") for sTiles parameters.
 *
 * Static defaults live in stiles_params.hpp. This file holds functions that
 * decide a value at runtime based on inputs (matrix size, hardware, etc.)
 * — i.e. what to do when a slot is set to "auto" or left unset.
 *
 * Conventions:
 *   - All pickers live in `namespace sTiles::policy`.
 *   - Each picker decides exactly one parameter / one decision.
 *   - Pickers are pure-ish: they may read hardware/runtime state and may
 *     log, but must not mutate global state.
 *   - Name pickers `pick_<thing>(...)` for grep-ability.
 */

#ifndef _STILES_POLICY_HPP_
#define _STILES_POLICY_HPP_

#include <string>

#include "stiles_logger.hpp"
#include "../TileIndexer/TileIndexer.hpp"

namespace sTiles {
namespace policy {

/// Pick the TileIndexer neighbor-lookup method for a matrix of dimension `n`
/// with the given `tile_size`.
///
/// Memory footprint of dense masks (upper triangle, N = ceil(n/tile_size)):
///   CharMask   : N*(N+1)/2 bytes      (1 byte per tile entry)
///   BitsetMask : N*(N+1)/2 / 8 bytes  (1 bit  per tile entry, 8x smaller)
///
/// CharMask is fastest for small/medium matrices but grows quadratically.
/// At N > ~20 000 tiles (~n > 800 K with tile_size = 40) the dense char
/// array exceeds ~200 MB; switch to BitsetMask which stays at ~25 MB and
/// is equally fast on those sizes.
inline TileIndexer::Method pick_indexer_method(int n, int tile_size) {
    if (tile_size <= 0) tile_size = 40;
    const long long N = (static_cast<long long>(n) + tile_size - 1) / tile_size;
    const long long upper_tri_bytes = N * (N + 1) / 2;
    constexpr long long threshold_bytes = 200LL * 1024 * 1024; // 200 MB
    if (upper_tri_bytes > threshold_bytes) {
        sTiles::Logger::info("│   [TileIndexer] n=" + std::to_string(n) +
                             ", N=" + std::to_string(N) +
                             " → CharMask would use " +
                             std::to_string(upper_tri_bytes / (1024*1024)) +
                             " MB → switching to BitsetMask (8x smaller)");
        return TileIndexer::Method::BitsetMask;
    }
    return TileIndexer::Method::CharMask;
}

} // namespace policy
} // namespace sTiles

#endif // _STILES_POLICY_HPP_
