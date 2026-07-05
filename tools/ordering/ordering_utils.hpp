#pragma once

// Set to true by METIS hub-detection guard; reset at start of each runND call.
// Defined at global scope in ordering_nd.cpp — declared here outside any namespace.
extern thread_local bool g_nd_hub_skipped;

#include "../TileIndexer/TileIndexer.hpp"
#include "../common/stiles_logger.hpp"
#include "../common/stiles_types.hpp"
#include "../memory/OrderingMemoryManager.hpp"
#include "ordering_shared_csr.hpp"   // sTiles::SharedAdjCSR (shared bake-off graph)
#include <cstddef>

#include <iomanip>
#include <sstream>
#include <vector>
#include <omp.h>

#ifdef STILES_FAST_RCM
#include "stiles_runRCM_fast.hpp"
#endif

namespace sTiles {

// SCOTCH nested-dissection separator tree, optionally returned by runSCOTCH_with_tree.
// Indices in `rangtab` reference positions in SCOTCH's permuted space [0, dim) where
// dim = N - m (the number of vertices SCOTCH actually orders; the trailing m "fixed"
// rows are appended as identity outside any block). For block k:
//   - rangtab[k] .. rangtab[k+1]-1  are the permuted-space positions in that block
//   - treetab[k] is the parent block index in the separator tree (-1 = root)
struct ScotchTree {
    int cblknbr = 0;
    std::vector<int> rangtab;  // size cblknbr+1
    std::vector<int> treetab;  // size cblknbr
};

// Forward C-ABI functions without including stiles_ordering.h
extern "C" {
    int  stiles_runRCM(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, bool safe);
    int  stiles_runRCM_scipy(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, bool safe);
    void stiles_runND(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, int num_sep, int** sizes);
    void stiles_runNDRCM(int** indices_i, int** indices_j, int N, int nnz, int m, int** perm, int** iperm, int num_sep, int** sizes);
    int runSCOTCH(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, int num_cores, int seed, const SharedAdjCSR* shared = nullptr);
    int runASCOTCH(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, int num_cores, int seed);

}

// C++ entry point: same as runSCOTCH but additionally returns the ND separator tree
// when tree_out != nullptr. Defined in ordering_scotch.cpp.
int runSCOTCH_with_tree(int** csr_i, int** csr_j, int N, int nnz, int m,
                        int** perm, int** iperm, int num_cores, int seed,
                        ScotchTree* tree_out, const SharedAdjCSR* shared = nullptr);

inline int runRCM(int** csr_i, int** csr_j, int N, int nnz, int m,
                  int** perm, int** iperm, bool safe)
{
    return stiles_runRCM(csr_i, csr_j, N, nnz, m, perm, iperm, safe);
}

inline int runRCM_scipy(int** csr_i, int** csr_j, int N, int nnz, int m,
                        int** perm, int** iperm, bool safe)
{
    return stiles_runRCM_scipy(csr_i, csr_j, N, nnz, m, perm, iperm, safe);
}

inline void runND(int** csr_i, int** csr_j, int N, int nnz, int m,
                  int** perm, int** iperm, int num_sep, int** sizes)
{
    stiles_runND(csr_i, csr_j, N, nnz, m, perm, iperm, num_sep, sizes);
}

inline void runNDRCM(int** indices_i, int** indices_j, int N, int nnz, int m,
                     int** perm, int** iperm, int num_sep, int** sizes)
{
    stiles_runNDRCM(indices_i, indices_j, N, nnz, m, perm, iperm, num_sep, sizes);
}

inline std::string format_seconds(double seconds)
{
    std::ostringstream oss;
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss << std::setprecision(3) << seconds;
    return oss.str();
}

// Validate that iperm is the inverse of perm (iperm[perm[i]] == i)
inline bool check_inverse_permutation(int* perm, int* iperm, int N)
{
    if (!perm || !iperm || N <= 0) {
        sTiles::Logger::error("check_inverse_permutation: invalid arguments (perm=",
                              static_cast<const void*>(perm), ", iperm=",
                              static_cast<const void*>(iperm), ", N=", N, ")");
        return false;
    }

    std::vector<char> seen_perm(static_cast<std::size_t>(N), 0);
    std::vector<char> seen_iperm(static_cast<std::size_t>(N), 0);

    for (int i = 0; i < N; ++i) {
        const int p = perm[i];
        if (p < 0 || p >= N) {
            sTiles::Logger::error("check_inverse_permutation: perm[", i,
                                  "] = ", p, " is out of range [0, ", N, ")");
            return false;
        }
        if (seen_perm[p]) {
            sTiles::Logger::error("check_inverse_permutation: duplicate permutation value detected (value=",
                                  p, ")");
            return false;
        }
        seen_perm[p] = 1;

        const int inv = iperm[p];
        if (inv < 0 || inv >= N) {
            sTiles::Logger::error("check_inverse_permutation: iperm[", p,
                                  "] = ", inv, " is out of range [0, ", N, ")");
            return false;
        }
        if (inv != i) {
            sTiles::Logger::error("check_inverse_permutation: mismatch detected (iperm[", p,
                                  "] = ", inv, ", expected ", i, ")");
            return false;
        }
        seen_iperm[inv] = 1;
    }

    // Verify that every slot in iperm was hit exactly once.
    for (int i = 0; i < N; ++i) {
        if (!seen_perm[i]) {
            sTiles::Logger::error("check_inverse_permutation: missing permutation value ", i);
            return false;
        }
        if (!seen_iperm[i]) {
            sTiles::Logger::error("check_inverse_permutation: missing inverse permutation value ", i);
            return false;
        }
    }

    return true;
}

// Compute inverse permutation: iperm[perm[i]] = i
inline void compute_inverse_permutation(const int* perm, int* iperm, int N)
{
    for (int i = 0; i < N; ++i) {
        iperm[perm[i]] = i;
    }
}

// Build permuted (row,col) arrays applying a vertex permutation.
// Mirrors original convention: row came from indices_j, col from indices_i.
inline void make_permuted_rc(const int* indices_i,
                             const int* indices_j,
                             int nnz,
                             const int* perm, // perm[v] = new index of vertex v
                             std::vector<int>& row_out,
                             std::vector<int>& col_out)
{
    row_out.resize(static_cast<std::size_t>(nnz));
    col_out.resize(static_cast<std::size_t>(nnz));
    for (int k = 0; k < nnz; ++k) {
        row_out[static_cast<std::size_t>(k)] = perm[indices_j[k]];
        col_out[static_cast<std::size_t>(k)] = perm[indices_i[k]];
    }
}

// Convenience wrapper: permute then count active tiles.
inline int count_active_tiles_with_perm(const int* indices_i,
                                        const int* indices_j,
                                        int nnz,
                                        int n,
                                        int tile_size,
                                        const int* perm,
                                        tilecounter::Method method,
                                        tilecounter::State* state = nullptr,
                                        int num_cores = 1,
                                        int group_id  = 0)
{
    std::vector<int> r, c;
    make_permuted_rc(indices_i, indices_j, nnz, perm, r, c);
    return tilecounter::countActiveTiles(r.data(), c.data(), nnz, n, tile_size,
                                         method, state, num_cores, group_id);
}

// Namespaced implementation of row/col permutation and triangular swap.
inline void permute_and_swap(int** perm,
                             int** row_indices,
                             int** col_indices,
                             int nnz)
{
    std::vector<int> rows(static_cast<std::size_t>(nnz));
    std::vector<int> cols(static_cast<std::size_t>(nnz));
    for (int index = 0; index < nnz; ++index) {
        if ((*perm)[(*col_indices)[index]] < (*perm)[(*row_indices)[index]]) {
            cols[static_cast<std::size_t>(index)] = (*perm)[(*row_indices)[index]];
            rows[static_cast<std::size_t>(index)] = (*perm)[(*col_indices)[index]];
        } else {
            rows[static_cast<std::size_t>(index)] = (*perm)[(*row_indices)[index]];
            cols[static_cast<std::size_t>(index)] = (*perm)[(*col_indices)[index]];
        }
    }
    for (int index = 0; index < nnz; ++index) {
        (*row_indices)[index] = cols[static_cast<std::size_t>(index)];
        (*col_indices)[index] = rows[static_cast<std::size_t>(index)];
    }
}

inline void to_ordering(double *x, double *ordered_x, int *perm, int n, int m) {

    for (int j = 0; j < m; j++) {
        // Reorder the j-th vector
        for (int i = 0; i < n; i++) {
            ordered_x[(j * n) + perm[i]] = x[(j * n) + i];
        }
    }
}

inline void from_ordering(double *x, double *ordered_x, int *perm, int n, int m) {

    for (int j = 0; j < m; j++) {
        // Reorder the j-th vector
        for (int i = 0; i < n; i++) {
            x[(j * n) + i] = ordered_x[(j * n) + perm[i]];
        }
    }
}

// Padded variants: user's x has stride x_stride; the reordered buffer has
// stride ox_stride (typically scheme->dim, which includes pad rows from SCOTCH
// block padding or ND padding). Iterate n = x_stride (number of user entries)
// per RHS; positions in [n, ox_stride) are not written and must be
// pre-zeroed by the caller (solve treats pad RHS = 0 since pad diag = 1).
// Default num_threads = 1 keeps callers that don't pass it on the serial path
// (matches pre-existing behaviour). dtrsm.cpp passes scheme->num_cores so the
// permute pass uses the same thread budget the chol/solve already paid for.
// RHS columns are independent → embarrassingly parallel along j.
//
// We branch at the C++ level (not via the omp `if(...)` clause) so the serial
// path has *zero* OMP runtime touch — no parallel-region setup, no team
// teardown. The omp `if` clause still costs a fork/join attempt; an
// if/else around the directive keeps the single-thread case truly bare.
inline void to_ordering_padded(const double *x, int x_stride,
                               double *ordered_x, int ox_stride,
                               const int *perm, int n, int m,
                               int num_threads = 1) {
    if (num_threads > 1 && m > 1) {
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int j = 0; j < m; ++j) {
            const double* src_col = x + static_cast<std::size_t>(j) * x_stride;
            double* dst_col = ordered_x + static_cast<std::size_t>(j) * ox_stride;
            for (int i = 0; i < n; ++i) {
                dst_col[perm[i]] = src_col[i];
            }
        }
    } else {
        for (int j = 0; j < m; ++j) {
            const double* src_col = x + static_cast<std::size_t>(j) * x_stride;
            double* dst_col = ordered_x + static_cast<std::size_t>(j) * ox_stride;
            for (int i = 0; i < n; ++i) {
                dst_col[perm[i]] = src_col[i];
            }
        }
    }
}

inline void from_ordering_padded(double *x, int x_stride,
                                 const double *ordered_x, int ox_stride,
                                 const int *perm, int n, int m,
                                 int num_threads = 1) {
    if (num_threads > 1 && m > 1) {
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int j = 0; j < m; ++j) {
            double* dst_col = x + static_cast<std::size_t>(j) * x_stride;
            const double* src_col = ordered_x + static_cast<std::size_t>(j) * ox_stride;
            for (int i = 0; i < n; ++i) {
                dst_col[i] = src_col[perm[i]];
            }
        }
    } else {
        for (int j = 0; j < m; ++j) {
            double* dst_col = x + static_cast<std::size_t>(j) * x_stride;
            const double* src_col = ordered_x + static_cast<std::size_t>(j) * ox_stride;
            for (int i = 0; i < n; ++i) {
                dst_col[i] = src_col[perm[i]];
            }
        }
    }
}

// Row-major counterparts: fuse the permutation with a layout transform so
// the downstream csc_dtrsm_multi_row kernel reads the permuted RHS in
// row-major form (rows are m doubles, contiguous; row stride = m).
//
// to_ordering_padded_row: source is column-major user B (n × m, ldb=x_stride),
// destination is row-major (rows in [0, ox_stride), each row m doubles).
// Pad rows [n, ox_stride) must be pre-zeroed by the caller.
inline void to_ordering_padded_row(const double *x, int x_stride,
                                   double *ordered_x_row, int /*ox_stride*/,
                                   const int *perm, int n, int m,
                                   int num_threads = 1) {
    auto body = [&](int i) {
        const int p = perm[i];
        double* dst_row = ordered_x_row + static_cast<std::size_t>(p) * m;
        for (int j = 0; j < m; ++j) {
            dst_row[j] = x[i + static_cast<std::size_t>(j) * x_stride];
        }
    };
    if (num_threads > 1 && n > 1) {
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int i = 0; i < n; ++i) body(i);
    } else {
        for (int i = 0; i < n; ++i) body(i);
    }
}

// from_ordering_padded_row: source is row-major (ox_stride × m), destination
// is column-major user B (n × m, ldb=x_stride). Inverse permutation +
// row-major-read + col-major-write fused into one pass.
inline void from_ordering_padded_row(double *x, int x_stride,
                                     const double *ordered_x_row, int /*ox_stride*/,
                                     const int *perm, int n, int m,
                                     int num_threads = 1) {
    auto body = [&](int i) {
        const int p = perm[i];
        const double* src_row = ordered_x_row + static_cast<std::size_t>(p) * m;
        for (int j = 0; j < m; ++j) {
            x[i + static_cast<std::size_t>(j) * x_stride] = src_row[j];
        }
    };
    if (num_threads > 1 && n > 1) {
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int i = 0; i < n; ++i) body(i);
    } else {
        for (int i = 0; i < n; ++i) body(i);
    }
}

inline void process_tiles(int** indices_i, int** indices_j, int** my_perm, int tile_size, int* total_num_used_tiles, int* nnz, bool* of_perm, int* counted_tiles, int* fix_counted_tiles) {

    int tileRow, tileCol;
    *counted_tiles = 0;
    
    for (int index = 0; index < (*nnz); index++) {
        tileRow = (*my_perm)[(*indices_j)[index]] / tile_size;
        tileCol = (*my_perm)[(*indices_i)[index]] / tile_size;

        if (tileRow <= tileCol) {
            if (!of_perm[tileRow * (2 * (*total_num_used_tiles) - tileRow - 1) / 2 + tileCol]) {
                of_perm[tileRow * (2 * (*total_num_used_tiles) - tileRow - 1) / 2 + tileCol] = true;
                (*counted_tiles)++;
            }
        } else {
            if (!of_perm[tileCol * (2 * (*total_num_used_tiles) - tileCol - 1) / 2 + tileRow]) {
                of_perm[tileCol * (2 * (*total_num_used_tiles) - tileCol - 1) / 2 + tileRow] = true;
                (*counted_tiles)++;
            }
        }
    }

    *fix_counted_tiles = *counted_tiles;
    bool sumT = true, res = true;
    for (int i = 0; i < (*total_num_used_tiles); i++) {
        for (int j = 0; j < i; j++) {

            sumT = false;
            for (int k = 0; k < j; k++){
                res = of_perm[k*(2*(*total_num_used_tiles)-k-1)/2 + i] * of_perm[k*(2*(*total_num_used_tiles)-k-1)/2 + j];
                if(res) {sumT = true; break;}
            }

            if(sumT && !(of_perm[j*(2*(*total_num_used_tiles)-j-1)/2 + i])){

                of_perm[j*(2*(*total_num_used_tiles)-j-1)/2 + i] = true;
                (*counted_tiles)++;
            }
        }
    }

}

// run_permutation moved to stiles_ordering.hpp for better cohesion

/**
 * @brief Convert CSR tile graph to COO format for ordering algorithms
 *
 * The tile graph is stored in CSR format (offsets + edges). Ordering algorithms
 * expect COO format (separate row/col arrays). This function extracts edges
 * from CSR and creates symmetric COO representation.
 *
 * @param graph_off_up CSR row offsets (size N+1)
 * @param graph_edges_up CSR column indices
 * @param graph_N Number of tiles
 * @param csr_i Output: row indices (allocated by function)
 * @param csr_j Output: column indices (allocated by function)
 * @param nnz Output: number of edges (doubled for symmetry)
 * @param group_index Memory allocation group
 * @return Number of edges (positive on success, negative on error)
 */
inline int convert_tile_graph_csr_to_coo(
    const std::vector<int>& graph_off_up,
    const std::vector<int>& graph_edges_up,
    int graph_N,
    int** csr_i,
    int** csr_j,
    int* nnz,
    int group_index)
{
    // Count edges (exclude diagonal if present)
    int edge_count = 0;
    for (int i = 0; i < graph_N; ++i) {
        const int start = graph_off_up[i];
        const int end = graph_off_up[i + 1];
        for (int k = start; k < end; ++k) {
            const int j = graph_edges_up[k];
            if (i != j) {  // Exclude self-loops
                edge_count++;
            }
        }
    }

    // Symmetric: count both (i,j) and (j,i)
    const int total_edges = 2 * edge_count;
    *nnz = total_edges;

    // Allocate COO arrays
    *csr_i = OrderingMemoryManager::allocate<int>(total_edges, group_index);
    *csr_j = OrderingMemoryManager::allocate<int>(total_edges, group_index);

    if (!*csr_i || !*csr_j) {
        sTiles::Logger::error("Failed to allocate COO arrays for tile graph");
        return -1;
    }

    // Populate COO: for each edge (i,j) in upper triangle, add both (i,j) and (j,i)
    int write_idx = 0;
    for (int i = 0; i < graph_N; ++i) {
        const int start = graph_off_up[i];
        const int end = graph_off_up[i + 1];
        for (int k = start; k < end; ++k) {
            const int j = graph_edges_up[k];
            if (i != j) {
                // Add edge (i,j) - note: ordering functions use swapped convention
                (*csr_j)[write_idx] = i;  // row in ordering convention
                (*csr_i)[write_idx] = j;  // col in ordering convention
                write_idx++;

                // Add symmetric edge (j,i)
                (*csr_j)[write_idx] = j;
                (*csr_i)[write_idx] = i;
                write_idx++;
            }
        }
    }

    return edge_count;
}

/**
 * @brief Compose tile-level permutation with element-level permutation
 *
 * Given:
 * - element_perm: old_element -> new_element (size = dim)
 * - tile_perm: old_tile -> new_tile (size = num_tiles)
 * - tile_size: elements per tile
 *
 * Compute composed permutation that applies tile reordering then element reordering.
 *
 * Strategy:
 * 1. For each element e in [0, dim):
 *    - old_tile = e / tile_size
 *    - within_tile_pos = e % tile_size
 *    - new_tile = tile_perm[old_tile]
 *    - new_base = new_tile * tile_size
 *    - intermediate_pos = new_base + within_tile_pos
 *    - final_pos = element_perm[intermediate_pos]
 * 2. composed_perm[e] = final_pos
 *
 * @param element_perm Existing element permutation (may be identity)
 * @param tile_perm Tile permutation to compose
 * @param dim Matrix dimension (number of elements)
 * @param num_tiles Number of tiles
 * @param tile_size Tile size
 * @param composed_perm Output: composed permutation (caller allocates)
 * @param composed_iperm Output: inverse of composed permutation (caller allocates)
 */
inline void compose_tile_and_element_permutation(
    const int* element_perm,
    const int* tile_perm,
    int dim,
    int num_tiles,
    int tile_size,
    int* composed_perm,
    int* composed_iperm)
{
    // First, create element-level permutation induced by tile permutation
    int* tile_induced_perm = new int[dim];

    for (int old_elem = 0; old_elem < dim; ++old_elem) {
        const int old_tile = old_elem / tile_size;
        const int within_tile = old_elem % tile_size;
        const int new_tile = tile_perm[old_tile];
        tile_induced_perm[old_elem] = new_tile * tile_size + within_tile;
    }

    // Compose: first apply tile permutation, then element permutation
    // composed_perm[e] = element_perm[tile_induced_perm[e]]
    for (int e = 0; e < dim; ++e) {
        const int intermediate = tile_induced_perm[e];
        composed_perm[e] = element_perm[intermediate];
    }

    // Compute inverse
    compute_inverse_permutation(composed_perm, composed_iperm, dim);

    delete[] tile_induced_perm;
}

/**
 * @brief Apply tile permutation to elements (simplified version without composition)
 *
 * Given a tile permutation, create the induced element permutation.
 * This is used when there's no existing element permutation to compose with.
 *
 * @param tile_perm Tile permutation
 * @param dim Matrix dimension (number of elements)
 * @param num_tiles Number of tiles
 * @param tile_size Tile size
 * @param element_perm Output: element permutation induced by tile permutation
 * @param element_iperm Output: inverse permutation
 */
inline void apply_tile_permutation_to_elements(
    const int* tile_perm,
    int dim,
    int num_tiles,
    int tile_size,
    int* element_perm,
    int* element_iperm)
{
    for (int old_elem = 0; old_elem < dim; ++old_elem) {
        const int old_tile = old_elem / tile_size;
        const int within_tile = old_elem % tile_size;
        const int new_tile = tile_perm[old_tile];
        element_perm[old_elem] = new_tile * tile_size + within_tile;
    }
    compute_inverse_permutation(element_perm, element_iperm, dim);
}


} // namespace sTiles

// Note: process_tiles remains provided via the C API (see stiles_ordering.h) and is
// implemented in a translation unit to preserve C linkage. Use that symbol if needed.
