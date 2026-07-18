/**
 * symbolic_chol_fillin.hpp
 *
 * Sparse symbolic Cholesky fill-in — element-wise, full matrix.
 *
 * Algorithm: elimination tree (etree) + column merge
 *   1. Compute etree via Liu's algorithm with path compression  O(n · α(n))
 *   2. Build children list as flat CSR                          O(n)
 *   3. Column-merge along etree, single pass                    O(nnz(L) · log k)
 *
 * Input : lower-triangular CSC, 0-based indices, symmetric matrix
 * Output: nnz(L) >= 0, or -1 if nnz(L) exceeds max_nnz budget
 *
 * IMPORTANT: apply a fill-reducing ordering (METIS/AMD) before calling.
 * Natural ordering on large matrices gives O(n^1.5) fill — intractable.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <cstdint>
#include "../../sort/stiles_sort_dispatch.hpp"

namespace sTiles {

// ---------------------------------------------------------------------------
// Elimination tree — O(n · α(n))
// ---------------------------------------------------------------------------
inline void compute_etree(int                      n,
                           const std::vector<int>&  colptr,
                           const std::vector<int>&  rowind,
                           std::vector<int>&        parent)
{
    parent.assign(n, -1);
    std::vector<int> ancestor(n, -1);

    // Liu's algorithm requires entries processed in order of their
    // UPPER-tri column (= max(row,col)).  For lower-tri A(i,k) with i>k,
    // that column is i, not k.  So we must iterate by ROW i, not column k.
    //
    // Build: for each row i, the list of columns k < i where A(i,k) != 0.
    // This is the upper-tri view of the symmetric matrix.

    std::vector<int> row_ptr(n + 1, 0);
    for (int k = 0; k < n; ++k)
        for (int p = colptr[k]; p < colptr[k + 1]; ++p)
            if (rowind[p] > k) ++row_ptr[rowind[p] + 1];
    for (int i = 0; i < n; ++i) row_ptr[i + 1] += row_ptr[i];

    std::vector<int> col_buf(row_ptr[n]);
    {
        std::vector<int> pos(row_ptr.begin(), row_ptr.end());
        for (int k = 0; k < n; ++k)
            for (int p = colptr[k]; p < colptr[k + 1]; ++p) {
                int i = rowind[p];
                if (i > k) col_buf[pos[i]++] = k;
            }
    }

    // Now iterate row i = upper-tri column i, processing columns k < i
    for (int i = 0; i < n; ++i) {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
            int t = col_buf[p];                 // t = k < i
            while (ancestor[t] != -1 && ancestor[t] != i) {
                int next    = ancestor[t];
                ancestor[t] = i;                // path compression
                t           = next;
            }
            if (ancestor[t] == -1) {
                ancestor[t] = i;
                parent[t]   = i;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Symbolic Cholesky fill-in
//
// max_nnz : abort and return -1 if nnz(L) would exceed this.
//           Pass 0 to disable the check (not recommended for large matrices
//           without a fill-reducing ordering).
// ---------------------------------------------------------------------------
// colptr_L holds 64-bit cumulative offsets (can exceed INT_MAX); rowind_L
// elements stay 32-bit (row indices < n). Returns nnz(L) as int64, or -1 if it
// exceeds max_nnz.
inline int64_t symbolic_chol_fillin(
    int                          n,
    const std::vector<int>&      colptr_A,
    const std::vector<int>&      rowind_A,
    std::vector<int64_t>&        colptr_L,
    std::vector<int>&            rowind_L,
    size_t                       max_nnz = 0)
{
    // ---- Step 1: elimination tree ----
    std::vector<int> parent;
    compute_etree(n, colptr_A, rowind_A, parent);

    // ---- Step 2: children list as flat CSR ----
    std::vector<int> child_ptr(n + 1, 0);
    for (int k = 0; k < n; ++k)
        if (parent[k] != -1) ++child_ptr[parent[k] + 1];
    for (int j = 0; j < n; ++j)
        child_ptr[j + 1] += child_ptr[j];

    std::vector<int> child_list(child_ptr[n]);
    {
        std::vector<int> pos(child_ptr.begin(), child_ptr.end());
        for (int k = 0; k < n; ++k)
            if (parent[k] != -1)
                child_list[pos[parent[k]]++] = k;
    }

    // ---- Step 3: single-pass column-merge ----
    const int nnz_A = colptr_A.back();

    // Safe initial reservation: 4× nnz(A), capped at max_nnz if set.
    // If natural ordering produces more fill we grow via push_back — no crash.
    size_t reserve_sz = static_cast<size_t>(nnz_A) * 4;
    if (max_nnz > 0) reserve_sz = std::min(reserve_sz, max_nnz);

    colptr_L.resize(n + 1);
    rowind_L.clear();
    rowind_L.reserve(reserve_sz);
    colptr_L[0] = 0;

    std::vector<int> marker(n, -1);
    std::vector<int> col_j;
    col_j.reserve(std::max(256, (int)(colptr_A[n] / n) * 8));  // ~8× avg col density

    for (int j = 0; j < n; ++j) {
        col_j.clear();

        // Seed from A[:,j]
        for (int p = colptr_A[j]; p < colptr_A[j + 1]; ++p) {
            int i = rowind_A[p];
            if (i >= j && marker[i] != j) {
                marker[i] = j;
                col_j.push_back(i);
            }
        }

        // Merge each direct etree child k
        for (int ci = child_ptr[j]; ci < child_ptr[j + 1]; ++ci) {
            int k = child_list[ci];
            for (int64_t p = colptr_L[k]; p < colptr_L[k + 1]; ++p) {
                int i = rowind_L[p];
                if (i >= j && marker[i] != j) {
                    marker[i] = j;
                    col_j.push_back(i);
                }
            }
        }

        sTiles::sort(col_j.begin(), col_j.end());

        // Budget check before appending
        if (max_nnz > 0 && rowind_L.size() + col_j.size() > max_nnz) {
            colptr_L.clear();
            rowind_L.clear();
            return -1;   // fill-in too large — apply fill-reducing ordering first
        }

        for (int row : col_j)
            rowind_L.push_back(row);

        colptr_L[j + 1] = static_cast<int64_t>(rowind_L.size());
    }

    rowind_L.shrink_to_fit();
    return static_cast<int64_t>(rowind_L.size());
}

// Count only — no pattern stored, same budget check.
inline int64_t symbolic_chol_fillin_count(
    int                      n,
    const std::vector<int>&  colptr_A,
    const std::vector<int>&  rowind_A,
    size_t                   max_nnz = 0)
{
    std::vector<int64_t> colptr_L;
    std::vector<int>     rowind_L;
    return symbolic_chol_fillin(n, colptr_A, rowind_A, colptr_L, rowind_L, max_nnz);
}

} // namespace sTiles
