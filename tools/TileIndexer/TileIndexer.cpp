/**
 * @file    TileIndexer.cpp
 * @brief   Core implementations for TileIndexer: method naming, upper-triangle
 *          indexing helper, and countActiveTiles dispatcher.
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

#include "TileIndexer.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace tilecounter {

/** @brief Human-readable string for a Method value. */
const char* to_string(Method m) {
    switch (m) {
        case Method::Auto:      return "Auto";
        case Method::CharMask:   return "CharMask";
        case Method::BoolMask:   return "BoolMask";
        case Method::BitsetMask: return "BitsetMask";
        case Method::TiledBoolMask:   return "TiledBoolMask";
        case Method::TiledBitsetMask: return "TiledBitsetMask";
        case Method::PagedMask:       return "PagedMask";
        case Method::LazyLookUp:      return "LazyLookUp";
        case Method::HashSet:    return "HashSet";
        case Method::SortUnique: return "SortUnique";
        default:                 return "Unknown";
    }
}

// ----- detail -----
namespace detail {

// Upper-triangular packed index (i <= j) in row-major over upper triangle
inline std::size_t upper_tile_index_u(std::size_t i, std::size_t j, std::size_t n) {
    // Row-major for upper triangle: off_row[i] + (j - i), where off_row[i] = i*(2*n - i + 1)/2
    return (i * (2ULL * n - i + 1ULL)) / 2ULL + (j - i);
}

} // namespace detail

/**
 * @brief Compute the packed row-major index for (row_tile, col_tile) assuming
 *        row_tile <= col_tile over an N x N tile grid.
 */
std::size_t upper_tile_index(int row_tile, int col_tile, int num_tiles) {
    const std::size_t i = static_cast<std::size_t>(row_tile);
    const std::size_t j = static_cast<std::size_t>(col_tile);
    const std::size_t n = static_cast<std::size_t>(num_tiles);
    return detail::upper_tile_index_u(i, j, n);
}

// ---- Common pre-checks ----
static inline bool invalid_inputs(const int* r, const int* c, int64_t nnz, int n, int tile) {
    return (!r || !c || nnz <= 0 || n <= 0 || tile <= 0);
}

static inline int tiles_from_n(int n, int tile_size) {
    return (n + tile_size - 1) / tile_size;
}

// Precompute starting offsets for each row in the packed upper triangle (row-major order).
// off[i] = i*(2*N - i - 1)/2 computed with O(n) adds.
static inline std::vector<std::size_t> make_upper_row_offsets(int num_tiles) {
    std::vector<std::size_t> off(static_cast<std::size_t>(num_tiles), 0);
    for (int i = 1; i < num_tiles; ++i) {
        off[static_cast<std::size_t>(i)] = off[static_cast<std::size_t>(i - 1)]
                                         + static_cast<std::size_t>(num_tiles - (i - 1));
    }
    return off;
}

// Small helper: fast bounds check that catches x<0 || x>=n in one branch.
static inline bool out_of_range(int x, int n) {
    return static_cast<unsigned>(x) >= static_cast<unsigned>(n);
}

// ---- Dispatcher with external state ----
/**
 * @brief Build the selected representation of active tiles by scanning the
 *        coordinate arrays (rows, cols) and counting unique tiles.
 *        Chooses specialized paths per Method and supports both power-of-two
 *        and generic tile sizes.
 */
int countActiveTiles(const int* row_indices,
                     const int* col_indices,
                     int64_t nnz, int n, int tile_size,
                     Method method,
                     State* state,
                     int num_cores,
                     int group_id)
{
    (void)num_cores;
    if (invalid_inputs(row_indices, col_indices, nnz, n, tile_size)) return 0;

    State local(group_id);
    State& st = state ? *state : local;

    const int num_tiles = tiles_from_n(n, tile_size);
    const std::size_t tri_size = static_cast<std::size_t>(num_tiles) * (num_tiles + 1ULL) / 2ULL;
    const auto off = make_upper_row_offsets(num_tiles);

    // Detect power-of-two tile size for fast division via shifts
    const bool pow2 = (tile_size & (tile_size - 1)) == 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned shift = pow2 ? __builtin_ctz(static_cast<unsigned>(tile_size)) : 0U;
#elif defined(_MSC_VER)
    unsigned shift = 0;
    if (pow2) { unsigned long s; _BitScanForward(&s, static_cast<unsigned>(tile_size)); shift = s; }
#else
    unsigned shift = 0;
    if (pow2) { while ((1u << shift) != static_cast<unsigned>(tile_size)) ++shift; }
#endif

    switch (method) {
        case Method::CharMask: {
            st.active_char.assign(tri_size, 0);
#if defined(_OPENMP)
            // Parallel path: per-thread 64-bit word masks, OR-merged into the same
            // active_char + count as the serial loop below (bit-identical result).
            // Gated on num_cores>1 AND a worthwhile nnz so small counts stay serial.
            // Callers that must not oversubscribe (e.g. concurrent bake-off counts)
            // pass num_cores=1 and skip this entirely.
            if (num_cores > 1 && nnz > (1 << 16)) {
                const int T = num_cores;
                const std::size_t words = (tri_size + 63ULL) >> 6;
                std::vector<std::vector<std::uint64_t>> locals(
                    static_cast<std::size_t>(T), std::vector<std::uint64_t>(words, 0ULL));
                const int64_t chunk = std::max<int64_t>(1024, nnz / (T * 16));
                #pragma omp parallel num_threads(T)
                {
                    std::vector<std::uint64_t>& B = locals[static_cast<std::size_t>(omp_get_thread_num())];
                    #pragma omp for schedule(static, chunk)
                    for (int64_t k = 0; k < nnz; ++k) {
                        const int r = row_indices[k], c = col_indices[k];
                        if (out_of_range(r, n) || out_of_range(c, n)) continue;
                        int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift) : r / tile_size;
                        int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift) : c / tile_size;
                        if (ti > tj) std::swap(ti, tj);
                        const std::size_t u = off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti);
                        B[u >> 6] |= (1ULL << (u & 63ULL));
                    }
                }
                std::vector<std::uint64_t> merged(words, 0ULL);
                for (const auto& B : locals)
                    for (std::size_t w = 0; w < words; ++w) merged[w] |= B[w];
                int pcount = 0;
                for (std::size_t w = 0; w < words; ++w) {
                    std::uint64_t x = merged[w];
                    while (x) {
                        const std::size_t u = (w << 6) + static_cast<unsigned>(__builtin_ctzll(x));
                        st.active_char[u] = 1; ++pcount;
                        x &= (x - 1);
                    }
                }
                return pcount;
            }
#endif
            int count = 0;
            if (pow2) {
                for (int64_t k = 0; k < nnz; ++k) {
                    const int r = row_indices[k], c = col_indices[k];
                    if (out_of_range(r, n) || out_of_range(c, n)) continue;
                    int ti = static_cast<int>(static_cast<unsigned>(r) >> shift);
                    int tj = static_cast<int>(static_cast<unsigned>(c) >> shift);
                    if (ti > tj) std::swap(ti, tj);
                    const std::size_t u = off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti);
                    if (!st.active_char[u]) { st.active_char[u] = 1; ++count; }
                }
            } else {
                for (int64_t k = 0; k < nnz; ++k) {
                    const int r = row_indices[k], c = col_indices[k];
                    if (out_of_range(r, n) || out_of_range(c, n)) continue;
                    int ti = r / tile_size, tj = c / tile_size;
                    if (ti > tj) std::swap(ti, tj);
                    const std::size_t u = off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti);
                    if (!st.active_char[u]) { st.active_char[u] = 1; ++count; }
                }
            }
            return count;
        }
        case Method::Auto:
        case Method::BoolMask:
        case Method::LazyLookUp: {
            st.active_bool.assign(tri_size, false);
            if (method == Method::LazyLookUp) st.lazy_index_map.assign(tri_size, -1);
            int count = 0;
            if (pow2) {
                for (int64_t k = 0; k < nnz; ++k) {
                    const int r = row_indices[k], c = col_indices[k];
                    if (out_of_range(r, n) || out_of_range(c, n)) continue;
                    int ti = static_cast<int>(static_cast<unsigned>(r) >> shift);
                    int tj = static_cast<int>(static_cast<unsigned>(c) >> shift);
                    if (ti > tj) std::swap(ti, tj);
                    const std::size_t u = off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti);
                    if (!st.active_bool[u]) {
                        st.active_bool[u] = true;
                        ++count;
                    }
                }
            } else {
                for (int64_t k = 0; k < nnz; ++k) {
                    const int r = row_indices[k], c = col_indices[k];
                    if (out_of_range(r, n) || out_of_range(c, n)) continue;
                    int ti = r / tile_size, tj = c / tile_size;
                    if (ti > tj) std::swap(ti, tj);
                    const std::size_t u = off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti);
                    if (!st.active_bool[u]) {
                        st.active_bool[u] = true;
                        ++count;
                    }
                }
            }
            if (method == Method::LazyLookUp) {
                for (std::size_t u = 0, next = 0; u < st.active_bool.size(); ++u) {
                    if (st.active_bool[u]) st.lazy_index_map[u] = next++;
                }
            }
            return count;
        }
        case Method::BitsetMask: {
            constexpr std::size_t W = 64;
            const std::size_t words = (tri_size + W - 1) / W;
            st.bits.assign(words, 0ULL);
            if (pow2) {
                for (int64_t k = 0; k < nnz; ++k) {
                    const int r = row_indices[k], c = col_indices[k];
                    if (out_of_range(r, n) || out_of_range(c, n)) continue;
                    int ti = static_cast<int>(static_cast<unsigned>(r) >> shift);
                    int tj = static_cast<int>(static_cast<unsigned>(c) >> shift);
                    const int lo = (ti <= tj) ? ti : tj;
                    const int hi = ti ^ tj ^ lo;
                    const std::size_t u = off[static_cast<std::size_t>(lo)] + static_cast<std::size_t>(hi - lo);
                    st.bits[u >> 6] |= (1ULL << (u & 63ULL));
                }
            } else {
                for (int64_t k = 0; k < nnz; ++k) {
                    const int r = row_indices[k], c = col_indices[k];
                    if (out_of_range(r, n) || out_of_range(c, n)) continue;
                    int ti = r / tile_size, tj = c / tile_size;
                    const int lo = (ti <= tj) ? ti : tj;
                    const int hi = ti ^ tj ^ lo;
                    const std::size_t u = off[static_cast<std::size_t>(lo)] + static_cast<std::size_t>(hi - lo);
                    st.bits[u >> 6] |= (1ULL << (u & 63ULL));
                }
            }
            // popcount all words
            int count = 0;
#if defined(_MSC_VER) && defined(_M_X64)
            for (std::size_t w = 0; w < words; ++w) count += static_cast<int>(__popcnt64(st.bits[w]));
#elif defined(__GNUC__) || defined(__clang__)
            for (std::size_t w = 0; w < words; ++w) count += static_cast<int>(__builtin_popcountll(st.bits[w]));
#else
            for (std::size_t w = 0; w < words; ++w) {
                std::uint64_t x = st.bits[w];
                x = x - ((x >> 1) & 0x5555555555555555ULL);
                x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
                count += static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
            }
#endif
            return count;
        }
        case Method::TiledBoolMask: {
            // Choose block dimension: 8 * tile_size (so each block has 64*n^2 bits)
            const int B = std::max(1, std::min(num_tiles, 8 * tile_size));
            const int blocks_per_dim = (num_tiles + B - 1) / B;
            st.tiled_block_dim = B;
            st.tiled_blocks_per_dim = blocks_per_dim;
            const std::size_t block_bits = static_cast<std::size_t>(B) * static_cast<std::size_t>(B);
            const std::size_t words_per_block = (block_bits + 63ULL) >> 6;

            for (int64_t k = 0; k < nnz; ++k) {
                const int r = row_indices[k], c = col_indices[k];
                if (out_of_range(r, n) || out_of_range(c, n)) continue;
                int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift) : (r / tile_size);
                int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift) : (c / tile_size);
                if (ti > tj) std::swap(ti, tj);
                const int bi = ti / B, bj = tj / B;
                const int li = ti - bi * B, lj = tj - bj * B; // local within block
                // ti<=tj globally, so within block li<=lj when bi==bj; fine for rectangular blocks too
                const std::size_t idx = static_cast<std::size_t>(li) * static_cast<std::size_t>(B) + static_cast<std::size_t>(lj);
                const std::size_t key = static_cast<std::size_t>(bi) * static_cast<std::size_t>(blocks_per_dim) + static_cast<std::size_t>(bj);
                auto& vec = st.tiled_chunks.try_emplace(key, st.make_chunk_vector(words_per_block, 0ULL)).first->second;
                vec[idx >> 6] |= (1ULL << (idx & 63ULL));
            }

            // Count set bits across all blocks
            int count = 0;
#if defined(_MSC_VER) && defined(_M_X64)
            for (const auto& kv : st.tiled_chunks) {
                const auto& vec = kv.second;
                for (std::size_t w = 0; w < vec.size(); ++w) count += static_cast<int>(__popcnt64(vec[w]));
            }
#elif defined(__GNUC__) || defined(__clang__)
            for (const auto& kv : st.tiled_chunks) {
                const auto& vec = kv.second;
                for (std::size_t w = 0; w < vec.size(); ++w) count += static_cast<int>(__builtin_popcountll(vec[w]));
            }
#else
            for (const auto& kv : st.tiled_chunks) {
                const auto& vec = kv.second;
                for (std::size_t w = 0; w < vec.size(); ++w) {
                    std::uint64_t x = vec[w];
                    x = x - ((x >> 1) & 0x5555555555555555ULL);
                    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
                    count += static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
                }
            }
#endif
            return count;
        }
        case Method::TiledBitsetMask: {
            // Identical storage to TiledBoolMask but tracked under a separate method
            const int B = std::max(1, std::min(num_tiles, 8 * tile_size));
            const int blocks_per_dim = (num_tiles + B - 1) / B;
            st.tiled_block_dim = B;
            st.tiled_blocks_per_dim = blocks_per_dim;
            const std::size_t block_bits = static_cast<std::size_t>(B) * static_cast<std::size_t>(B);
            const std::size_t words_per_block = (block_bits + 63ULL) >> 6;

            for (int64_t k = 0; k < nnz; ++k) {
                const int r = row_indices[k], c = col_indices[k];
                if (out_of_range(r, n) || out_of_range(c, n)) continue;
                int ti = pow2 ? static_cast<int>(static_cast<unsigned>(r) >> shift) : (r / tile_size);
                int tj = pow2 ? static_cast<int>(static_cast<unsigned>(c) >> shift) : (c / tile_size);
                if (ti > tj) std::swap(ti, tj);
                const int bi = ti / B, bj = tj / B;
                const int li = ti - bi * B, lj = tj - bj * B;
                const std::size_t idx = static_cast<std::size_t>(li) * static_cast<std::size_t>(B) + static_cast<std::size_t>(lj);
                const std::size_t key = static_cast<std::size_t>(bi) * static_cast<std::size_t>(blocks_per_dim) + static_cast<std::size_t>(bj);
                auto& vec = st.tiled_chunks.try_emplace(key, st.make_chunk_vector(words_per_block, 0ULL)).first->second;
                vec[idx >> 6] |= (1ULL << (idx & 63ULL));
            }

            int count = 0;
#if defined(_MSC_VER) && defined(_M_X64)
            for (const auto& kv : st.tiled_chunks) {
                const auto& vec = kv.second;
                for (std::size_t w = 0; w < vec.size(); ++w) count += static_cast<int>(__popcnt64(vec[w]));
            }
#elif defined(__GNUC__) || defined(__clang__)
            for (const auto& kv : st.tiled_chunks) {
                const auto& vec = kv.second;
                for (std::size_t w = 0; w < vec.size(); ++w) count += static_cast<int>(__builtin_popcountll(vec[w]));
            }
#else
            for (const auto& kv : st.tiled_chunks) {
                const auto& vec = kv.second;
                for (std::size_t w = 0; w < vec.size(); ++w) {
                    std::uint64_t x = vec[w];
                    x = x - ((x >> 1) & 0x5555555555555555ULL);
                    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
                    count += static_cast<int>((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56);
                }
            }
#endif
            return count;
        }
        case Method::PagedMask: {
            // Pages over packed upper-tri indices (u): each page = 64 * tile_size^2 bits
            const std::size_t N  = static_cast<std::size_t>(tile_size);
            const std::size_t PB = 64ULL * N * N;   // bits per page
            const std::size_t PW = (PB >> 6);       // words per page (== N*N)

            st.paged.page_bits  = PB;
            st.paged.page_words = PW;
            st.paged.pages.clear();

            auto test_and_set = [&](std::size_t u, int& cnt) {
                const std::size_t page = PB ? (u / PB) : 0;
                const std::size_t bit  = PB ? (u % PB) : u;
                const std::size_t w    = bit >> 6;
                const std::uint64_t m  = 1ULL << (bit & 63ULL);
                auto it = st.paged.pages.find(page);
                if (it == st.paged.pages.end()) {
                    it = st.paged.pages.emplace(page, st.paged.make_page_vector(PW, 0ULL)).first;
                }
                std::uint64_t& word = it->second[w];
                if ((word & m) == 0ULL) { word |= m; ++cnt; }
            };

            int count = 0;
            if (pow2) {
                for (int64_t k = 0; k < nnz; ++k) {
                    const int r = row_indices[k], c = col_indices[k];
                    if (out_of_range(r, n) || out_of_range(c, n)) continue;
                    int ti = static_cast<int>(static_cast<unsigned>(r) >> shift);
                    int tj = static_cast<int>(static_cast<unsigned>(c) >> shift);
                    const int lo = (ti <= tj) ? ti : tj;
                    const int hi = ti ^ tj ^ lo;
                    const std::size_t u = off[static_cast<std::size_t>(lo)] + static_cast<std::size_t>(hi - lo);
                    test_and_set(u, count);
                }
            } else {
                for (int64_t k = 0; k < nnz; ++k) {
                    const int r = row_indices[k], c = col_indices[k];
                    if (out_of_range(r, n) || out_of_range(c, n)) continue;
                    int ti = r / tile_size, tj = c / tile_size;
                    if (ti > tj) std::swap(ti, tj);
                    const std::size_t u = off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti);
                    test_and_set(u, count);
                }
            }
            return count;
        }
        case Method::HashSet: {
            st.S.clear();
            st.S.reserve(static_cast<std::size_t>(std::min<std::size_t>(nnz, tri_size)));
            if (pow2) {
                for (int64_t k = 0; k < nnz; ++k) {
                    const int r = row_indices[k], c = col_indices[k];
                    if (out_of_range(r, n) || out_of_range(c, n)) continue;
                    int ti = static_cast<int>(static_cast<unsigned>(r) >> shift);
                    int tj = static_cast<int>(static_cast<unsigned>(c) >> shift);
                    if (ti > tj) std::swap(ti, tj);
                    st.S.insert(off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti));
                }
            } else {
                for (int64_t k = 0; k < nnz; ++k) {
                    const int r = row_indices[k], c = col_indices[k];
                    if (out_of_range(r, n) || out_of_range(c, n)) continue;
                    int ti = r / tile_size, tj = c / tile_size;
                    if (ti > tj) std::swap(ti, tj);
                    st.S.insert(off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti));
                }
            }
            return static_cast<int>(st.S.size());
        }
        case Method::SortUnique: {
            st.ids.clear();
            st.ids.reserve(static_cast<std::size_t>(nnz));
            if (pow2) {
                for (int64_t k = 0; k < nnz; ++k) {
                    const int r = row_indices[k], c = col_indices[k];
                    if (out_of_range(r, n) || out_of_range(c, n)) continue;
                    int ti = static_cast<int>(static_cast<unsigned>(r) >> shift);
                    int tj = static_cast<int>(static_cast<unsigned>(c) >> shift);
                    if (ti > tj) std::swap(ti, tj);
                    st.ids.push_back(off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti));
                }
            } else {
                for (int64_t k = 0; k < nnz; ++k) {
                    const int r = row_indices[k], c = col_indices[k];
                    if (out_of_range(r, n) || out_of_range(c, n)) continue;
                    int ti = r / tile_size, tj = c / tile_size;
                    if (ti > tj) std::swap(ti, tj);
                    st.ids.push_back(off[static_cast<std::size_t>(ti)] + static_cast<std::size_t>(tj - ti));
                }
            }
            std::sort(st.ids.begin(), st.ids.end());
            st.ids.erase(std::unique(st.ids.begin(), st.ids.end()), st.ids.end());
            return static_cast<int>(st.ids.size());
        }
        default: {
            // Fallback to CharMask
            st.active_char.assign(tri_size, 0);
            int count = 0;
            for (int64_t k = 0; k < nnz; ++k) {
                int r = row_indices[k], c = col_indices[k];
                if (r < 0 || c < 0 || r >= n || c >= n) continue;
                int ti = r / tile_size, tj = c / tile_size;
                if (ti > tj) std::swap(ti, tj);
                const std::size_t u = upper_tile_index(ti, tj, num_tiles);
                if (!st.active_char[u]) { st.active_char[u] = 1; ++count; }
            }
            return count;
        }
    }
}

} // namespace tilecounter
