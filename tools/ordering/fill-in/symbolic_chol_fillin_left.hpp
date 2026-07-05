/**
 * symbolic_chol_fillin_left.hpp
 *
 * Left-looking sparse symbolic Cholesky fill-in — element-wise, full matrix.
 * No sTiles structs, no custom memory managers, no external dependencies.
 *
 * Algorithm: left-looking column-by-column
 *   For each column j:
 *     1. Seed from A[:,j]  (rows >= j)
 *     2. For each k < j where L(j,k) != 0: merge struct(L[:,k]) rows > j
 *     3. Deduplicate via marker array, sort for CSC order
 *
 *   L_T[i] tracks which columns j < i have L(i,j) != 0
 *   (replaces SparseRowHandler)
 *
 * Input : lower-triangular CSC, 0-based indices, symmetric matrix
 * Output: nnz(L), colptr_L, rowind_L
 *
 * Compare with symbolic_chol_fillin.hpp (etree-based) for correctness/speed.
 */

#pragma once

#include <vector>
#include <algorithm>

namespace sTiles {

inline int symbolic_chol_fillin_left(
    int                      n,
    const std::vector<int>&  colptr_A,
    const std::vector<int>&  rowind_A,
    std::vector<int>&        colptr_L,
    std::vector<int>&        rowind_L,
    size_t                   max_nnz = 0)
{
    // L_T[i] = columns j < i where L(i,j) != 0  (row i of L^T)
    // Replaces SparseRowHandler — plain flat vectors, no pointer chasing.
    std::vector<std::vector<int>> L_T(n);

    colptr_L.resize(n + 1);
    rowind_L.clear();
    size_t reserve_sz = static_cast<size_t>(colptr_A.back()) * 4;
    if (max_nnz > 0) reserve_sz = std::min(reserve_sz, max_nnz);
    rowind_L.reserve(reserve_sz);
    colptr_L[0] = 0;

    // marker[i] = j  →  node i already queued for column j (no reset needed)
    std::vector<int> marker(n, -1);
    std::vector<int> col_j;
    col_j.reserve(256);

    for (int j = 0; j < n; ++j) {
        col_j.clear();

        // 1. Seed from A[:,j]  (lower triangle: rows >= j)
        for (int p = colptr_A[j]; p < colptr_A[j + 1]; ++p) {
            int i = rowind_A[p];
            if (i >= j && marker[i] != j) {
                marker[i] = j;
                col_j.push_back(i);
            }
        }

        // 2. Left-looking update: for each k < j where L(j,k) != 0
        //    merge struct(L[:,k]) restricted to rows > j
        for (int k : L_T[j]) {
            for (int p = colptr_L[k]; p < colptr_L[k + 1]; ++p) {
                int i = rowind_L[p];
                if (i > j && marker[i] != j) {
                    marker[i] = j;
                    col_j.push_back(i);
                }
            }
        }

        // 3. Sort (no duplicates — marker guarantees uniqueness)
        std::sort(col_j.begin(), col_j.end());

        // 4. Budget check before appending
        if (max_nnz > 0 && rowind_L.size() + col_j.size() > max_nnz) {
            colptr_L.clear();
            rowind_L.clear();
            return -1;   // fill-in too large — apply fill-reducing ordering first
        }

        // 5. Append to L and update L_T for future columns
        for (int row : col_j) {
            rowind_L.push_back(row);
            if (row > j)
                L_T[row].push_back(j);  // L(row, j) != 0
        }

        colptr_L[j + 1] = static_cast<int>(rowind_L.size());
    }

    rowind_L.shrink_to_fit();
    return static_cast<int>(rowind_L.size());
}

inline int symbolic_chol_fillin_left_count(
    int                      n,
    const std::vector<int>&  colptr_A,
    const std::vector<int>&  rowind_A,
    size_t                   max_nnz = 0)
{
    std::vector<int> colptr_L, rowind_L;
    return symbolic_chol_fillin_left(n, colptr_A, rowind_A, colptr_L, rowind_L, max_nnz);
}

} // namespace sTiles
