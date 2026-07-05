/**
 * @file    TileIndexerMemoryUtils.hpp
 * @brief   Helpers to track/format TileIndexer memory usage by group, bind the
 *          appropriate isActive checker, and refresh lazy lookup maps.
 *
 * @project sTiles (Sparse Tiles Library)
 * @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @version 3.0.0
 * @date 1 1 2026
 * @license Proprietary
 *
 * @note This file is part of the sTiles library, a proprietary software package.
 *       Redistribution or modification without prior permission is prohibited.
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
 *
 * @license
 * This software is proprietary and confidential. Unauthorized copying, distribution, or modification
 * of this software, via any medium, is strictly prohibited. Permission is granted to use the software
 * in binary form for non-commercial purposes only, provided that this copyright notice and permission
 * notice are included in all copies or substantial portions of the software.
 *
 * DISCLAIMER:
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "TileIndexer.hpp"
#include "TileIndexerIsActive.hpp"

#include <array>
#include <unordered_map>

namespace tilecounter_utils {

using Manager = sTiles::TileIndexerMemoryManager;

inline constexpr std::array<std::pair<Manager::Group, const char*>, 14> kGroupInfo = {{
    {Manager::Group::ActiveBool,     "ActiveBool"},
    {Manager::Group::ActiveChar,     "ActiveChar"},
    {Manager::Group::DenseBits,      "DenseBits"},
    {Manager::Group::SparseSet,      "SparseSet"},
    {Manager::Group::SparseIds,      "SparseIds"},
    {Manager::Group::TiledChunkMap,  "TiledChunkMap"},
    {Manager::Group::TiledChunkWords,"TiledChunkWords"},
    {Manager::Group::PagedMap,       "PagedMap"},
    {Manager::Group::PagedWords,     "PagedWords"},
    {Manager::Group::FillScratch,    "FillScratch"},
    {Manager::Group::GraphOffsets,   "GraphOffsets"},
    {Manager::Group::GraphEdges,     "GraphEdges"},
    {Manager::Group::SFL,            "SFL"},
    {Manager::Group::Misc,           "Misc"}
}};

static_assert(TileIndexer::State::kGroupCount == static_cast<int>(kGroupInfo.size()),
              "TileIndexer group info mismatch");

/*
Struct: MemorySnapshot
Purpose: Aggregate view of total bytes and per-group usage for diagnostics.
*/
struct MemorySnapshot {
    std::size_t total = 0;
    std::array<std::size_t, kGroupInfo.size()> groups{};
};

/*
Function: take_snapshot
Purpose:  Capture current total allocator usage and per-group counters.
*/
inline MemorySnapshot take_snapshot() {
    MemorySnapshot snap;
    snap.total = Manager::getTotalMemoryAllocated();
    snap.groups.fill(0);
    Manager::for_each_group([&](int group_id, std::size_t bytes) {
        if (group_id < 0) return;
        const int canonical = group_id % TileIndexer::State::kGroupCount;
        if (canonical >= 0 && static_cast<std::size_t>(canonical) < snap.groups.size()) {
            snap.groups[static_cast<std::size_t>(canonical)] += bytes;
        }
    });
    return snap;
}

/*
Function: diff_snapshot
Purpose:  Compute the delta between two snapshots.
*/
inline MemorySnapshot diff_snapshot(const MemorySnapshot& after, const MemorySnapshot& before) {
    MemorySnapshot diff;
    diff.total = (after.total >= before.total) ? (after.total - before.total) : 0;
    for (std::size_t i = 0; i < kGroupInfo.size(); ++i) {
        diff.groups[i] = (after.groups[i] >= before.groups[i]) ? (after.groups[i] - before.groups[i]) : 0;
    }
    return diff;
}

/*
Function: bind_is_active
Purpose:  Assign the State.isActive pointer based on the resolved method.
*/
inline void bind_is_active(TileIndexer::State& state, TileIndexer::Method method) {
    TileIndexer::Method resolved = TileIndexer::resolve_auto_method_for_fill(state, method);
    using tilecounter::is_active_bitset;
    using tilecounter::is_active_bool;
    using tilecounter::is_active_char;
    using tilecounter::is_active_hashset;
    using tilecounter::is_active_paged;
    using tilecounter::is_active_sortunique;
    using tilecounter::is_active_tiled_bitset;
    using tilecounter::is_active_tiled_bool;

    switch (resolved) {
        case TileIndexer::Method::BitsetMask:      state.setIsActive(is_active_bitset); break;
        case TileIndexer::Method::BoolMask:        state.setIsActive(is_active_bool); break;
        case TileIndexer::Method::CharMask:        state.setIsActive(is_active_char); break;
        case TileIndexer::Method::TiledBoolMask:   state.setIsActive(is_active_tiled_bool); break;
        case TileIndexer::Method::TiledBitsetMask: state.setIsActive(is_active_tiled_bitset); break;
        case TileIndexer::Method::PagedMask:       state.setIsActive(is_active_paged); break;
        case TileIndexer::Method::HashSet:         state.setIsActive(is_active_hashset); break;
        case TileIndexer::Method::SortUnique:      state.setIsActive(is_active_sortunique); break;
        default:
            state.setIsActive(nullptr);
            break;
    }
}

} // namespace tilecounter_utils

namespace tilecounter {

/*
Function: refresh_lazy_index_map_if_needed
Purpose:  Rebuild the lazy (u -> rank) index map when using LazyLookUp.
*/
inline void refresh_lazy_index_map_if_needed(State& state, Method method) {
    Method resolved = resolve_auto_method_for_fill(state, method);
    if (resolved != Method::LazyLookUp) return;
    state.lazy_index_map.assign(state.active_bool.size(), -1);
    int next = 0;
    for (std::size_t u = 0; u < state.active_bool.size(); ++u) {
        if (state.active_bool[u]) state.lazy_index_map[u] = next++;
    }
}

} // namespace tilecounter
