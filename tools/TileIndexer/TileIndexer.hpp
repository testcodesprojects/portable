/**
 * @file    TileIndexer.hpp
 * @brief   Public API and core state for TileIndexer: storage strategies,
 *          shared State (custom allocators), helpers, and top-level APIs.
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
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <iostream>

#include "../memory/TileIndexerMemoryManager.hpp"

namespace tilecounter {

/**
 * @brief Select the internal representation used to record active tiles.
 *        Dense masks (Char/Bool/Bitset), chunked/tiled masks, paged masks,
 *        and sparse sets/vectors are available. Auto defers to a planner.
 */
// ---- Public strategy choice ----
enum class Method {
    Auto,        // auto-select strategy (resolved via planner)
    CharMask,     // dense upper-tri mask in std::vector<char>
    BoolMask,     // dense upper-tri mask in std::vector<bool>
    BitsetMask,    // dense upper-tri bitset in uint64_t words
    TiledBoolMask, // chunked bool mask: tiled blocks (stored as bits)
    TiledBitsetMask, // chunked bitset mask: tiled blocks of bits
    PagedMask,     // paged 1D mask over packed upper-tri indices (64*n^2 bits per page)
    LazyLookUp,    // dense bool mask with compact index map + lazy closure
    HashSet,      // sparse: unordered_set of active tile IDs
    SortUnique    // sparse: gather IDs, sort, unique
};

/** @brief Human-readable name for the given Method. */
const char* to_string(Method m);

/**
 * @brief Packed row-major index for address (row_tile, col_tile) in the
 *        upper triangle (assumes row_tile <= col_tile).
 */
std::size_t upper_tile_index(int row_tile, int col_tile, int num_tiles);


/**
 * @brief Owns storage and allocators for all TileIndexer strategies including
 *        dense masks, sparse sets, chunked/paged layouts, graphs, and the
 *        generic isActive checker. Also tracks auto-planning hints.
 */
struct State {
    using Manager = sTiles::TileIndexerMemoryManager;
    using Group = Manager::Group;

    template <typename T>
    using Allocator = Manager::Allocator<T>;

    static constexpr int kGroupCount = static_cast<int>(Group::Misc) + 1;

    static constexpr int compute_offset(int group_id) noexcept {
        return group_id * kGroupCount;
    }

    static constexpr int group_index(int base, Group g) noexcept {
        return base + static_cast<int>(g);
    }

    template <typename T>
    static Allocator<T> makeAllocatorFor(int base, Group g) {
        return Allocator<T>(group_index(base, g));
    }

    template <typename T>
    Allocator<T> makeAllocator(Group g) const {
        return makeAllocatorFor<T>(group_offset_, g);
    }

    using BoolVector = std::vector<bool, Allocator<bool>>;
    using CharVector = std::vector<char, Allocator<char>>;
    using WordVector = std::vector<std::uint64_t, Allocator<std::uint64_t>>;
    using IdVector   = std::vector<std::size_t, Allocator<std::size_t>>;
    using SparseSet  = std::unordered_set<std::size_t, std::hash<std::size_t>, std::equal_to<std::size_t>, Allocator<std::size_t>>;
    using ChunkVector = std::vector<std::uint64_t, Allocator<std::uint64_t>>;
    using ChunkMap = std::unordered_map<
        std::size_t,
        ChunkVector,
        std::hash<std::size_t>,
        std::equal_to<std::size_t>,
        Allocator<std::pair<const std::size_t, ChunkVector>>>;

    struct Paged {
        using PageVector = std::vector<std::uint64_t, Allocator<std::uint64_t>>;
        using PageMap = std::unordered_map<
            std::size_t,
            PageVector,
            std::hash<std::size_t>,
            std::equal_to<std::size_t>,
            Allocator<std::pair<const std::size_t, PageVector>>>;

        Paged(Allocator<std::uint64_t> vecAlloc = Allocator<std::uint64_t>(),
              Allocator<std::pair<const std::size_t, PageVector>> mapAlloc = Allocator<std::pair<const std::size_t, PageVector>>())
            : page_bits(0),
              page_words(0),
              pages(0, std::hash<std::size_t>{}, std::equal_to<std::size_t>{}, mapAlloc),
              page_vector_alloc(vecAlloc)
        {}

        std::size_t page_bits;   // 64 * tile_size^2
        std::size_t page_words;  // page_bits / 64 == tile_size^2
        PageMap pages;           // key = page index, value = words
        Allocator<std::uint64_t> page_vector_alloc;

        void clear() {
            PageMap empty(0, std::hash<std::size_t>{}, std::equal_to<std::size_t>{}, pages.get_allocator());
            pages.swap(empty);
            page_bits = 0;
            page_words = 0;
        }

        bool empty() const { return pages.empty(); }

        PageVector make_page_vector(std::size_t words, std::uint64_t value = 0ULL) const {
            PageVector vec(page_vector_alloc);
            vec.assign(words, value);
            return vec;
        }
    };

    explicit State(int group_id = -1)
        : group_offset_(group_id >= 0 ? compute_offset(group_id)
                                      : allocate_group_offset()),
          active_bool(makeAllocator<bool>(Group::ActiveBool)),
          active_char(makeAllocator<char>(Group::ActiveChar)),
          bits(makeAllocator<std::uint64_t>(Group::DenseBits)),
          S(0, std::hash<std::size_t>{}, std::equal_to<std::size_t>{}, makeAllocator<std::size_t>(Group::SparseSet)),
          ids(makeAllocator<std::size_t>(Group::SparseIds)),
          lazy_index_map(makeAllocator<int>(Group::Misc)),
          auto_count_method(Method::BoolMask),
          auto_fill_method(Method::BoolMask),
          auto_count_threads(1),
          auto_fill_threads(1),
          auto_plan_valid(false),
          tiled_block_dim(0),
          tiled_blocks_per_dim(0),
          tiled_chunks(0,
                       std::hash<std::size_t>{},
                       std::equal_to<std::size_t>{},
                       makeAllocator<std::pair<const std::size_t, ChunkVector>>(Group::TiledChunkMap)),
          scratch_bits(makeAllocator<std::uint64_t>(Group::FillScratch)),
          paged(makeAllocator<std::uint64_t>(Group::PagedWords),
                makeAllocator<std::pair<const std::size_t, Paged::PageVector>>(Group::PagedMap))
    {}

    int group_offset() const noexcept { return group_offset_; }
    int group_base() const noexcept { return group_offset_ / kGroupCount; }
    int group_for(Group g) const noexcept { return group_index(group_offset_, g); }

private:
    int group_offset_;
    static int allocate_group_offset() {
        const int base = next_group_base_.fetch_add(1, std::memory_order_relaxed);
        return base * kGroupCount;
    }
    inline static std::atomic<int> next_group_base_{0};

public:
    // Dense
    BoolVector active_bool;
    CharVector active_char;
    WordVector bits;

    // Sparse
    SparseSet S;
    IdVector  ids; // for sort+unique
    std::vector<int, Allocator<int>> lazy_index_map;
    Method auto_count_method;
    Method auto_fill_method;
    int auto_count_threads;
    int auto_fill_threads;
    bool auto_plan_valid;

    // Chunked bool mask storage (tiled upper-tri mask)
    // - The tile grid (num_tiles x num_tiles, upper-tri used) is partitioned
    //   into blocks of size B x B in tile coordinates.
    // - Only blocks that receive at least one set bit are materialized.
    // - Each block stores a dense bitset of B*B bits packed into uint64_t words;
    //   bit k corresponds to local (li, lj) where k = li*B + lj.
    // - Key encodes block coordinates: key = bi*blocks_per_dim + bj.
    int tiled_block_dim;        // B: block side length in tile-grid units
    int tiled_blocks_per_dim;   // blocks_per_dim: number of blocks per dimension
    ChunkMap tiled_chunks;      // sparse map of blocks

    // Paged 1D mask over packed upper-tri indices (u). Pages are 64*n^2 bits each.
    Paged paged;
    mutable WordVector scratch_bits;

    ChunkVector make_chunk_vector(std::size_t words, std::uint64_t value = 0ULL) const {
        ChunkVector vec(makeAllocator<std::uint64_t>(Group::TiledChunkWords));
        vec.assign(words, value);
        return vec;
    }

    // Generic active-predicate function pointer (strategy-specific checker)
    using IsActiveFn = bool(*)(const State&, int /*i*/, int /*j*/, int /*N*/);
    IsActiveFn is_active = nullptr;

    inline void setIsActive(IsActiveFn fn) { is_active = fn; }
    inline bool isActive(int i, int j, int N) const {
        return is_active ? is_active(*this, i, j, N) : false;
    }

    void reset() {
        BoolVector(makeAllocator<bool>(Group::ActiveBool)).swap(active_bool);
        CharVector(makeAllocator<char>(Group::ActiveChar)).swap(active_char);
        WordVector(makeAllocator<std::uint64_t>(Group::DenseBits)).swap(bits);
        SparseSet(0, std::hash<std::size_t>{}, std::equal_to<std::size_t>{}, makeAllocator<std::size_t>(Group::SparseSet)).swap(S);
        IdVector(makeAllocator<std::size_t>(Group::SparseIds)).swap(ids);
        std::vector<int, Allocator<int>>(makeAllocator<int>(Group::Misc)).swap(lazy_index_map);
        tiled_block_dim = 0;
        tiled_blocks_per_dim = 0;
        ChunkMap(0,
                 std::hash<std::size_t>{},
                 std::equal_to<std::size_t>{},
                 makeAllocator<std::pair<const std::size_t, ChunkVector>>(Group::TiledChunkMap)).swap(tiled_chunks);
        WordVector(makeAllocator<std::uint64_t>(Group::FillScratch)).swap(scratch_bits);
        paged.clear();
        auto_count_method = Method::BoolMask;
        auto_fill_method = Method::BoolMask;
        auto_count_threads = 1;
        auto_fill_threads = 1;
        auto_plan_valid = false;

        // Clear any prebuilt graphs
        graph_off_up.clear(); graph_off_up.shrink_to_fit();
        graph_edges_up.clear(); graph_edges_up.shrink_to_fit();
        graph_off_lo.clear(); graph_off_lo.shrink_to_fit();
        graph_edges_lo.clear(); graph_edges_lo.shrink_to_fit();
        graph_N = 0; graph_include_self = false; graphs_built = false;
    }

    // Optional prebuilt adjacency graphs (CSR) for upper/lower neighborhoods
    // Built over tile indices [0..N). When present, algorithms can reuse them
    // instead of calling isActive repeatedly.
    std::vector<int> graph_off_up;
    std::vector<int> graph_edges_up;
    std::vector<int> graph_off_lo;
    std::vector<int> graph_edges_lo;
    int  graph_N = 0;
    bool graph_include_self = false;
    bool graphs_built = false;
};

using Manager = State::Manager;
using Group = State::Group;

template <typename T>
using Allocator = State::Allocator<T>;

using BoolVector = State::BoolVector;
using CharVector = State::CharVector;
using WordVector = State::WordVector;
using IdVector   = State::IdVector;
using SparseSet  = State::SparseSet;
using ChunkVector = State::ChunkVector;
using ChunkMap = State::ChunkMap;
using PagedState = State::Paged;

/**
 * @brief Build the active-tile structure from matrix coordinates and return
 *        the number of unique active tiles discovered. Uses the requested
 *        Method to choose storage and may populate State if provided.
 */
int countActiveTiles(const int* row_indices,
                     const int* col_indices,
                     int64_t nnz,
                     int n,
                     int tile_size,
                     Method method,
                     State* state = nullptr,
                     int num_cores = 1,
                     int group_id = 0);

/**
 * @brief Add tiles implied by the transitive closure rule using the strategy
 *        encoded in State/Method. Returns the updated active count.
 * @param N tiles-per-dimension (num_tiles)
 * @param counted pre-fill active count
 */
int FillTiles(State& state, Method method, int N, int counted, int num_threads);
int FillTiles(State& state, Method method, int N, int counted);

/**
 * @brief Ensure all diagonal tiles are marked as active in the TileIndexer state.
 *        This is critical for Cholesky factorization, which requires all diagonal
 *        tiles to exist even if they have no non-zero entries.
 * @param state The TileIndexer state to modify
 * @param method The storage method being used
 * @param N Number of tiles per dimension
 * @param counted Current count of active tiles (updated in place)
 * @return Updated count of active tiles after ensuring diagonals
 */
int ensure_diagonal_tiles_active(State& state, Method method, int N, int& counted);

/**
 * @brief Clear all containers and free memory recorded against State's
 *        internal allocation groups.
 */
inline void release_state_resources(State& state) {
    state.reset();
    using G = State::Group;
    constexpr G groups[] = {
        G::ActiveBool,
        G::ActiveChar,
        G::DenseBits,
        G::SparseSet,
        G::SparseIds,
        G::TiledChunkMap,
        G::TiledChunkWords,
        G::PagedMap,
        G::PagedWords,
        G::FillScratch,
        G::GraphOffsets,
        G::GraphEdges,
        G::SFL,
        G::Misc
    };
    for (G g : groups) {
        sTiles::TileIndexerMemoryManager::freeAllGroup(state.group_for(g));
    }
}

inline Method resolve_auto_method_for_count(const State& state, Method requested) {
    if (requested == Method::Auto && state.auto_plan_valid) {
        return state.auto_count_method;
    }
    return (requested == Method::Auto) ? Method::BoolMask : requested;
}

inline Method resolve_auto_method_for_fill(const State& state, Method requested) {
    if (requested == Method::Auto && state.auto_plan_valid) {
        return state.auto_fill_method;
    }
    return (requested == Method::Auto) ? Method::BoolMask : requested;
}

inline int resolve_auto_threads_for_count(const State& state, int fallback) {
    if (state.auto_plan_valid && state.auto_count_threads > 0) {
        return state.auto_count_threads;
    }
    return fallback;
}

inline int resolve_auto_threads_for_fill(const State& state, int fallback) {
    if (state.auto_plan_valid && state.auto_fill_threads > 0) {
        return state.auto_fill_threads;
    }
    return fallback;
}

/**
 * @brief Build a CSR adjacency of active neighbors per tile for the chosen
 *        representation. Implemented in TileIndexerGraphBuilder.hpp.
 */
void build_graph(const State& idx,
                 Method m,
                 int N,
                 std::vector<int>& offsets,
                 std::vector<int>& edges,
                 bool include_self);

struct GraphBuffers {
    std::vector<int> offsets;
    std::vector<int> edges;
};

/**
 * @brief End-to-end construction: build State by counting, perform a fill,
 *        optionally prepare lazy index, and build the upper-triangle graph.
 * @return 0 on success (active/filled are output parameters), non-zero on bad inputs.
 */
inline int build_state_and_graph(const int* rows,
                                 const int* cols,
                                 int nnz,
                                 int n,
                                 int tile_size,
                                 Method method,
                                 State& state,
                                 GraphBuffers& graph,
                                 int& active_out,
                                 int& filled_out,
                                 int num_threads = 1)
{
    if (tile_size <= 0) {
        std::cerr << "Error: tile_size must be positive.\n";
        return 1;
    }

    const int num_tiles = (n + tile_size - 1) / tile_size;
    const int active = countActiveTiles(rows, cols, nnz, n,
                                        tile_size, method,
                                        &state, num_threads);
    const int filled = FillTiles(state, method, num_tiles, active, num_threads);
    if (resolve_auto_method_for_fill(state, method) == Method::LazyLookUp) {
        state.lazy_index_map.assign(state.active_bool.size(), -1);
        int next = 0;
        for (std::size_t u = 0; u < state.active_bool.size(); ++u) {
            if (state.active_bool[u]) state.lazy_index_map[u] = next++;
        }
    }

    graph.offsets.clear();
    graph.edges.clear();
    build_graph(state, method, num_tiles, graph.offsets, graph.edges, false);

    active_out = active;
    filled_out = filled;
    return 0;
}

} // namespace tilecounter

// Public namespace alias to present the API as TileIndexer::
namespace TileIndexer = tilecounter;
