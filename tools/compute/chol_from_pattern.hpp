/**
 * chol_from_pattern.hpp
 *
 * Reference sparse Cholesky A = L L^T using a PRE-COMPUTED L sparsity pattern.
 *
 * Two entry points:
 *   - factor_from_pattern          : scalar, single-threaded reference
 *   - factor_from_pattern_parallel : elimination-tree level-set parallel
 *
 * Both consume:
 *   - A in lower-triangular CSC: (Ap, Ai, Ax). Row indices within each column
 *     must be sorted ascending, include the diagonal, exclude upper triangle.
 *   - L pattern in CSC: (Lp, Li). Must be a superset of A's lower triangle.
 *     Diagonal of L is the first entry of each column (Li[Lp[j]] == j).
 *
 * Both produce:
 *   - Lx: values aligned 1:1 with Li positions (caller allocates nnz(L) doubles).
 *   - logdet: 2 * sum(log(L(j,j))).
 *
 * Return:
 *   0 on success; (column_index + 1) of the first SPD breakdown.
 *
 * Algorithm: left-looking sparse Cholesky. For each column j:
 *   1. Scatter A(:,j) into workspace w.
 *   2. For each k<j with L(j,k)!=0 (via precomputed row-pattern of L),
 *      subtract L(:,k) * L(j,k) from w's positions.
 *   3. L(j,j) = sqrt(w[j]); L(i,j) = w[i] / L(j,j) for i>j with L(i,j)!=0.
 *   4. Clear the touched positions in w.
 *
 * Parallel variant adds an etree level-set schedule: columns at the same
 * elimination-tree depth have disjoint dependency sets and factor in parallel.
 * Each thread holds its own length-n dense workspace.
 *
 * Complexity: O(sum_k nnzcol(k) * nnzrow_below(k)). Scalar per column.
 * No BLAS-3, no tiling, no vectorization hand-tuning.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sTiles {

// ----------------------------------------------------------------------------
// Internal helpers shared by both entry points.
// ----------------------------------------------------------------------------
namespace detail_chol_pat {

struct RowPattern {
    std::vector<int> Lrp;      // size n+1, strict-lower row pointers
    std::vector<int> Lri;      // column index k for each row entry
    std::vector<int> Lrx_pos;  // position in Lx of the (i,k) entry
};

inline void build_row_pattern(int n,
                              const int* Lp, const int* Li,
                              RowPattern& rp)
{
    rp.Lrp.assign(static_cast<std::size_t>(n + 1), 0);
    for (int k = 0; k < n; ++k) {
        const int pbeg = Lp[k];
        const int pend = Lp[k + 1];
        for (int p = pbeg + 1; p < pend; ++p)
            ++rp.Lrp[Li[p] + 1];
    }
    for (int i = 0; i < n; ++i) rp.Lrp[i + 1] += rp.Lrp[i];

    rp.Lri.assign(static_cast<std::size_t>(rp.Lrp[n]), 0);
    rp.Lrx_pos.assign(static_cast<std::size_t>(rp.Lrp[n]), 0);
    std::vector<int> cursor(rp.Lrp.begin(), rp.Lrp.begin() + n);
    for (int k = 0; k < n; ++k) {
        const int pbeg = Lp[k];
        const int pend = Lp[k + 1];
        for (int p = pbeg + 1; p < pend; ++p) {
            const int i     = Li[p];
            const int w_pos = cursor[i]++;
            rp.Lri[w_pos]     = k;
            rp.Lrx_pos[w_pos] = p;
        }
    }
}

// Factor a single column j using already-finalized columns k < j.
// Workspace w is length-n and must be zero on entry; on exit, entries at
// L's column-j positions are zeroed by this function (clean for next use).
inline int factor_one_column(
    int j,
    const int* Ap, const int* Ai, const double* Ax,
    const int* Lp, const int* Li, double* Lx,
    const RowPattern& rp,
    double* w,
    double& logdet_contrib)
{
    // 1. Scatter A(:,j).
    for (int p = Ap[j]; p < Ap[j + 1]; ++p) w[Ai[p]] = Ax[p];

    // 2. Apply updates from every k < j with L(j,k) != 0.
    for (int q = rp.Lrp[j]; q < rp.Lrp[j + 1]; ++q) {
        const int    k   = rp.Lri[q];
        const double Ljk = Lx[rp.Lrx_pos[q]];
        const int    col_end = Lp[k + 1];
        for (int p = rp.Lrx_pos[q]; p < col_end; ++p)
            w[Li[p]] -= Lx[p] * Ljk;
    }

    // 3. Finalize column j.
    const int col_j_beg = Lp[j];
    const int col_j_end = Lp[j + 1];
    const double diag = w[j];
    if (!(diag > 0.0)) return j + 1;   // SPD breakdown
    const double Ljj     = std::sqrt(diag);
    const double inv_Ljj = 1.0 / Ljj;
    Lx[col_j_beg]     = Ljj;
    logdet_contrib   += 2.0 * std::log(Ljj);
    for (int p = col_j_beg + 1; p < col_j_end; ++p)
        Lx[p] = w[Li[p]] * inv_Ljj;

    // 4. Clear workspace at touched positions.
    for (int p = col_j_beg; p < col_j_end; ++p) w[Li[p]] = 0.0;

    return 0;
}

} // namespace detail_chol_pat

// ----------------------------------------------------------------------------
// Single-threaded scalar reference.
// ----------------------------------------------------------------------------
inline int factor_from_pattern(
    int                  n,
    const int*           Ap,
    const int*           Ai,
    const double*        Ax,
    const int*           Lp,
    const int*           Li,
    double*              Lx,
    double&              logdet)
{
    detail_chol_pat::RowPattern rp;
    detail_chol_pat::build_row_pattern(n, Lp, Li, rp);
    std::vector<double> w(static_cast<std::size_t>(n), 0.0);
    logdet = 0.0;
    for (int j = 0; j < n; ++j) {
        int rc = detail_chol_pat::factor_one_column(
            j, Ap, Ai, Ax, Lp, Li, Lx, rp, w.data(), logdet);
        if (rc != 0) return rc;
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Etree level-set parallel variant.
//
// Each column is assigned a level = its depth in the elimination tree
// (leaves = 0, each parent one higher than its deepest child). Columns at the
// same level live in disjoint etree subtrees, so their L-pattern dependencies
// are disjoint — they factor in parallel safely.
//
// Per-thread state: a length-n dense workspace. Workspaces stay clean between
// that thread's own iterations because factor_one_column zeros the positions
// it touches.
// ----------------------------------------------------------------------------
inline int factor_from_pattern_parallel(
    int                  n,
    const int*           Ap,
    const int*           Ai,
    const double*        Ax,
    const int*           Lp,
    const int*           Li,
    double*              Lx,
    double&              logdet,
    int                  num_threads = 0)
{
#ifndef _OPENMP
    (void)num_threads;
    return factor_from_pattern(n, Ap, Ai, Ax, Lp, Li, Lx, logdet);
#else
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    if (num_threads <= 1 || n < 1024) {
        return factor_from_pattern(n, Ap, Ai, Ax, Lp, Li, Lx, logdet);
    }

    detail_chol_pat::RowPattern rp;
    detail_chol_pat::build_row_pattern(n, Lp, Li, rp);

    // --- Etree parent: parent[j] = first strict-lower row in column j.
    std::vector<int> parent(static_cast<std::size_t>(n), -1);
    for (int j = 0; j < n; ++j) {
        if (Lp[j + 1] > Lp[j] + 1) parent[j] = Li[Lp[j] + 1];
    }

    // --- Level: leaves at 0, parent at max(children)+1. Process j ascending.
    std::vector<int> level(static_cast<std::size_t>(n), 0);
    int max_lvl = 0;
    for (int j = 0; j < n; ++j) {
        const int p = parent[j];
        if (p >= 0) {
            const int cand = level[j] + 1;
            if (cand > level[p]) level[p] = cand;
        }
        if (level[j] > max_lvl) max_lvl = level[j];
    }

    // --- Bucket columns by level (CSR-style for fast iteration).
    std::vector<int> level_head(static_cast<std::size_t>(max_lvl + 2), 0);
    for (int j = 0; j < n; ++j) ++level_head[level[j] + 1];
    for (int l = 0; l <= max_lvl; ++l) level_head[l + 1] += level_head[l];
    std::vector<int> level_cols(static_cast<std::size_t>(n));
    {
        std::vector<int> cursor(level_head.begin(), level_head.begin() + (max_lvl + 1));
        for (int j = 0; j < n; ++j) level_cols[cursor[level[j]]++] = j;
    }

    // --- Per-thread workspaces + per-thread logdet accumulators.
    std::vector<std::vector<double>> tl_w(static_cast<std::size_t>(num_threads),
        std::vector<double>(static_cast<std::size_t>(n), 0.0));
    std::vector<double> tl_logdet(static_cast<std::size_t>(num_threads), 0.0);
    std::vector<int>    tl_err(static_cast<std::size_t>(num_threads), 0);

    for (int l = 0; l <= max_lvl; ++l) {
        const int beg = level_head[l];
        const int end = level_head[l + 1];
        const int mcols = end - beg;
        if (mcols <= 0) continue;

        // Small levels run serially on thread 0 to avoid spawn overhead.
        if (mcols < 8) {
            for (int idx = beg; idx < end; ++idx) {
                const int j = level_cols[idx];
                int rc = detail_chol_pat::factor_one_column(
                    j, Ap, Ai, Ax, Lp, Li, Lx, rp,
                    tl_w[0].data(), tl_logdet[0]);
                if (rc != 0) return rc;
            }
            continue;
        }

        int level_err = 0;
        #pragma omp parallel num_threads(num_threads) shared(level_err)
        {
            const int tid = omp_get_thread_num();
            double* w = tl_w[tid].data();

            #pragma omp for schedule(dynamic, 16) nowait
            for (int idx = beg; idx < end; ++idx) {
                if (level_err) continue;
                const int j = level_cols[idx];
                int rc = detail_chol_pat::factor_one_column(
                    j, Ap, Ai, Ax, Lp, Li, Lx, rp, w, tl_logdet[tid]);
                if (rc != 0) {
                    #pragma omp atomic write
                    level_err = rc;
                    tl_err[tid] = rc;
                }
            }
        }
        if (level_err != 0) return level_err;
    }

    logdet = 0.0;
    for (int t = 0; t < num_threads; ++t) logdet += tl_logdet[t];
    return 0;
#endif
}

} // namespace sTiles
