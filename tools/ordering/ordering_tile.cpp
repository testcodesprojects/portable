// Tile-level ordering wrapper.
//
// Compresses the element-level COO graph into a tile graph, runs a chosen
// permutation algorithm on that smaller graph, then expands the tile
// permutation back to an element permutation via block-aligned mapping.
//
// Convention (same as all other orderings in this codebase):
//   perm[new_pos]  = old_element   (0-based)
//   iperm[old_element] = new_pos   (0-based)
//
// Memory: *perm and *iperm are allocated with new[] inside this file and
//         must be freed by the caller with delete[].

#include <vector>
#include <unordered_set>
#include <cstdint>
#include <cstdlib>   // malloc / free (external sub-orderings use malloc internally)
#include <cstdio>
#include <algorithm>
#include <numeric>
#include "ordering_shared_csr.hpp"   // sTiles::SharedAdjCSR (added to runSCOTCH/runSuiteSparse sigs)

// ── Sub-ordering forward declarations ────────────────────────────────────────
// Cases 1-2: C-ABI (extern "C") — actual symbols in ordering_rcm.cpp / ordering_nd.cpp
extern "C" {
    int  stiles_runRCM(int** csr_i, int** csr_j, int N, int nnz, int m,
                       int** perm, int** iperm, bool safe);
    void stiles_runND(int** csr_i, int** csr_j, int N, int nnz, int m,
                      int** perm, int** iperm, int num_sep, int** sizes);
}
// Case 3: stiles_runNDRCM — C-ABI (extern "C") due to ordering_utils.hpp declaration
// Case 4: runSCOTCH    — C-ABI (extern "C") due to ordering_utils.hpp declaration
extern "C" {
    void stiles_runNDRCM(int** csr_i, int** csr_j, int N, int nnz, int m,
                         int** perm, int** iperm, int num_sep, int** sizes);
    int  runSCOTCH(int** csr_i, int** csr_j, int N, int nnz, int m,
                   int** perm, int** iperm, int num_cores, int seed,
                   const sTiles::SharedAdjCSR* shared = nullptr);
}
// Cases 5-9: runSuiteSparse — C++ in namespace sTiles (NOT in extern "C" in ordering_utils.hpp)
namespace sTiles {
    int  runSuiteSparse(int** csr_i, int** csr_j, int N, int nnz, int m,
                        int** perm, int** iperm, int num_cores, int strategy_num,
                        const SharedAdjCSR* shared = nullptr);
}
// Cases 12-17: external orderings — allocate *perm / *iperm internally with malloc
namespace sTiles {
void stiles_runAMD(int* csr_i, int* csr_j, int N, int nnz,
                   int** perm, int** iperm);
void stiles_runAMDBAR(int* csr_i, int* csr_j, int N, int nnz,
                      int** perm, int** iperm);
void stiles_runAMDBARNEW(int* csr_i, int* csr_j, int N, int nnz,
                         int** perm, int** iperm);
void stiles_runAMDNEW(int* csr_i, int* csr_j, int N, int nnz,
                      int** perm, int** iperm);
void stiles_runGenMMD(int* csr_i, int* csr_j, int N, int nnz,
                      int** perm, int** iperm, int delta = 0);
void stiles_runGPS(int* csr_i, int* csr_j, int N, int nnz,
                   int** perm, int** iperm, int mode = 0);
} // namespace sTiles
// ─────────────────────────────────────────────────────────────────────────────

namespace sTiles {

// ── Step 1: build tile graph ─────────────────────────────────────────────────
// Returns the tile COO in tile_i / tile_j (both vectors sized tile_nnz).
// Layout: tiles_dim diagonal self-loops first, then unique off-diagonal edges
// (lo < hi, sorted for determinism).  This matches the format expected by all
// run_permutation cases (symmetric, one direction per off-diagonal edge, plus
// explicit diagonal entries so METIS sees every vertex).
static void build_tile_graph(
    const int* elem_i, const int* elem_j, int elem_nnz,
    int N, int tile_size,
    int tiles_dim,
    std::vector<int>& tile_i, std::vector<int>& tile_j)
{
    // Collect unique undirected off-diagonal tile edges
    std::unordered_set<std::uint64_t> edge_set;
    edge_set.reserve(static_cast<std::size_t>(elem_nnz));

    for (int k = 0; k < elem_nnz; ++k) {
        int r = elem_i[k], c = elem_j[k];
        if (r < 0 || r >= N || c < 0 || c >= N) continue;
        int tR = r / tile_size;
        int tC = c / tile_size;
        if (tR == tC) continue;
        if (tR > tC) std::swap(tR, tC);
        std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(tR)) << 32)
                          |  static_cast<std::uint32_t>(tC);
        edge_set.insert(key);
    }

    const int n_edges = static_cast<int>(edge_set.size());
    tile_i.reserve(tiles_dim + n_edges);
    tile_j.reserve(tiles_dim + n_edges);

    // Diagonal (one per tile vertex)
    for (int t = 0; t < tiles_dim; ++t) {
        tile_i.push_back(t);
        tile_j.push_back(t);
    }

    // Off-diagonal edges, sorted for determinism
    std::vector<std::uint64_t> sorted_keys(edge_set.begin(), edge_set.end());
    std::sort(sorted_keys.begin(), sorted_keys.end());
    for (std::uint64_t key : sorted_keys) {
        tile_i.push_back(static_cast<int>(key >> 32));
        tile_j.push_back(static_cast<int>(key & 0xFFFFFFFFu));
    }
}

// ── Step 2: fix partial-last-tile alignment ───────────────────────────────────
// When N is not a multiple of tile_size the last tile is smaller than the rest.
// All ordering algorithms may send it anywhere.  Force it to the last slot so
// that the block-aligned expansion ep[e] = tp[e/ts]*ts + e%ts is valid for
// every element without an out-of-bounds access.
static void fix_partial_tile(std::vector<int>& tp, int tiles_dim, int N, int tile_size)
{
    const int last_tile  = tiles_dim - 1;
    const int partial_sz = N - last_tile * tile_size;

    if (partial_sz == tile_size) return;  // N is a multiple of tile_size — nothing to do
    if (tp[last_tile] == last_tile) return;  // already in the last slot

    // Find which tile currently occupies the last slot and swap
    std::vector<int> inv(tiles_dim);
    for (int t = 0; t < tiles_dim; ++t) inv[tp[t]] = t;

    const int q = inv[last_tile];   // tile currently going to last slot
    const int p = tp[last_tile];    // slot the partial tile was sent to
    tp[last_tile] = last_tile;
    tp[q]         = p;
}

// ── Step 3: expand tile perm → element perm ──────────────────────────────────
// ep[e] = tp[e / tile_size] * tile_size + (e % tile_size)
// Allocates *perm and *iperm with new int[].
static void expand_tile_perm(
    const std::vector<int>& tp, int tiles_dim, int N, int tile_size,
    int** perm, int** iperm)
{
    (void)tiles_dim;
    *perm  = new int[N];
    *iperm = new int[N];

    for (int e = 0; e < N; ++e)
        (*perm)[e] = tp[e / tile_size] * tile_size + (e % tile_size);

    for (int e = 0; e < N; ++e)
        (*iperm)[(*perm)[e]] = e;
}

// ── Public entry point ────────────────────────────────────────────────────────
//
// stiles_runTileOrdering
//
//   elem_i, elem_j : element COO (0-based, size nnz); not modified
//   N               : number of elements (matrix dimension)
//   nnz             : number of entries in the COO
//   tile_size       : tile size used to build the tile graph
//   sub_ordering    : algorithm to run on the tile graph
//                       1  = RCM
//                       2  = ND
//                       3  = ND+RCM
//                      12  = AMD
//                      13  = AMDBAR
//                      14  = AMDBARNEW
//                      15  = AMDNEW
//                      16  = GenMMD
//                      17  = GPS
//                      18  = SMTP-Band
//   perm, iperm     : output element permutations (allocated with new[], caller
//                     must delete[]); set to nullptr on failure
//
void stiles_runTileOrdering(
    int* elem_i, int* elem_j,
    int N, int nnz,
    int sub_ordering,
    int** perm, int** iperm,
    int tile_size = 2)
{
    *perm  = nullptr;
    *iperm = nullptr;

    if (N <= 0 || nnz < 0 || tile_size <= 0) {
        fprintf(stderr, "stiles_runTileOrdering: invalid arguments "
                        "(N=%d, nnz=%d, tile_size=%d)\n", N, nnz, tile_size);
        return;
    }

    const int tiles_dim = (N + tile_size - 1) / tile_size;

    // ── Step 1: build tile graph ──────────────────────────────────────────────
    std::vector<int> tile_i_vec, tile_j_vec;
    build_tile_graph(elem_i, elem_j, nnz, N, tile_size, tiles_dim,
                     tile_i_vec, tile_j_vec);
    const int tile_nnz = static_cast<int>(tile_i_vec.size());

    // Mutable working copies (some ordering routines modify the arrays)
    int* ti = new int[tile_nnz];
    int* tj = new int[tile_nnz];
    std::copy(tile_i_vec.begin(), tile_i_vec.end(), ti);
    std::copy(tile_j_vec.begin(), tile_j_vec.end(), tj);
    tile_i_vec.clear();
    tile_j_vec.clear();

    // ── Step 2: run ordering on tile graph ────────────────────────────────────
    // tile_perm / tile_iperm ownership rules:
    //   cases 1-3  : pre-allocated here with malloc; ordering fills in-place
    //   cases 12+  : ordering mallocs internally; we free with free() after use
    int* tile_perm  = nullptr;
    int* tile_iperm = nullptr;
    bool extern_owned = false;  // true → free() required, false → delete[] required

    switch (sub_ordering) {

        case 1: {  // RCM
            tile_perm  = static_cast<int*>(malloc(tiles_dim * sizeof(int)));
            tile_iperm = static_cast<int*>(malloc(tiles_dim * sizeof(int)));
            int* ti2 = ti, *tj2 = tj;
            stiles_runRCM(&ti2, &tj2, tiles_dim, tile_nnz, 0, &tile_iperm, &tile_perm, false);
            break;
        }

        case 2: {  // ND
            tile_perm  = static_cast<int*>(malloc(tiles_dim * sizeof(int)));
            tile_iperm = static_cast<int*>(malloc(tiles_dim * sizeof(int)));
            int* ti2 = ti, *tj2 = tj;
            int* sizes = nullptr;
            stiles_runND(&ti2, &tj2, tiles_dim, tile_nnz, 0, &tile_iperm, &tile_perm, 2, &sizes);
            if (sizes) delete[] sizes;
            break;
        }

        case 3: {  // ND+RCM
            tile_perm  = static_cast<int*>(malloc(tiles_dim * sizeof(int)));
            tile_iperm = static_cast<int*>(malloc(tiles_dim * sizeof(int)));
            int* ti2 = ti, *tj2 = tj;
            int* sizes = nullptr;
            ::stiles_runNDRCM(&ti2, &tj2, tiles_dim, tile_nnz, 0, &tile_perm, &tile_iperm, 2, &sizes);
            if (sizes) delete[] sizes;
            break;
        }

        case 4: {  // SCOTCH
            tile_perm  = static_cast<int*>(malloc(tiles_dim * sizeof(int)));
            tile_iperm = static_cast<int*>(malloc(tiles_dim * sizeof(int)));
            int* ti2 = ti, *tj2 = tj;
            ::runSCOTCH(&ti2, &tj2, tiles_dim, tile_nnz, 0, &tile_iperm, &tile_perm, 1, 40);
            break;
        }

        case 5:   // SuiteSparse AMD
        case 6:   // SuiteSparse CAMD
        case 7:   // SuiteSparse COLAMD
        case 8:   // SuiteSparse CCOLAMD
        case 9: { // SuiteSparse SYMAMD
            tile_perm  = static_cast<int*>(malloc(tiles_dim * sizeof(int)));
            tile_iperm = static_cast<int*>(malloc(tiles_dim * sizeof(int)));
            int* ti2 = ti, *tj2 = tj;
            runSuiteSparse(&ti2, &tj2, tiles_dim, tile_nnz, 0, &tile_iperm, &tile_perm, 1,
                           sub_ordering - 5);
            break;
        }

        // External and bandwidth orderings — allocate perm/iperm internally with malloc
        case 12:
            sTiles::stiles_runAMD(ti, tj, tiles_dim, tile_nnz, &tile_perm, &tile_iperm);
            extern_owned = true;
            break;

        case 13:
            sTiles::stiles_runAMDBAR(ti, tj, tiles_dim, tile_nnz, &tile_perm, &tile_iperm);
            extern_owned = true;
            break;

        case 14:
            sTiles::stiles_runAMDBARNEW(ti, tj, tiles_dim, tile_nnz, &tile_perm, &tile_iperm);
            extern_owned = true;
            break;

        case 15:
            sTiles::stiles_runAMDNEW(ti, tj, tiles_dim, tile_nnz, &tile_perm, &tile_iperm);
            extern_owned = true;
            break;

        case 16:
            sTiles::stiles_runGenMMD(ti, tj, tiles_dim, tile_nnz, &tile_perm, &tile_iperm);
            extern_owned = true;
            break;

        case 17:
            sTiles::stiles_runGPS(ti, tj, tiles_dim, tile_nnz, &tile_perm, &tile_iperm);
            extern_owned = true;
            break;

        default:
            fprintf(stderr, "stiles_runTileOrdering: unknown sub_ordering=%d, "
                            "falling back to identity\n", sub_ordering);
            tile_perm  = static_cast<int*>(malloc(tiles_dim * sizeof(int)));
            tile_iperm = static_cast<int*>(malloc(tiles_dim * sizeof(int)));
            std::iota(tile_perm,  tile_perm  + tiles_dim, 0);
            std::iota(tile_iperm, tile_iperm + tiles_dim, 0);
            break;
    }

    delete[] ti;
    delete[] tj;

    if (!tile_perm || !tile_iperm) {
        fprintf(stderr, "stiles_runTileOrdering: sub-ordering %d returned null perm\n",
                sub_ordering);
        if (extern_owned) { free(tile_perm); free(tile_iperm); }
        else            { free(tile_perm); free(tile_iperm); }
        return;
    }

    // ── Step 3: fix partial-last-tile, then expand ────────────────────────────
    std::vector<int> tp(tile_perm, tile_perm + tiles_dim);
    if (extern_owned) { free(tile_perm);  free(tile_iperm); }
    else            { free(tile_perm);  free(tile_iperm); }

    fix_partial_tile(tp, tiles_dim, N, tile_size);
    expand_tile_perm(tp, tiles_dim, N, tile_size, perm, iperm);
}

} // namespace sTiles
