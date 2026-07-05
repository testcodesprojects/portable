// SciPy-derived Reverse Cuthill-McKee implementation in plain C++.
//
// Algorithm ported from:
//   scipy/sparse/csgraph/_reordering.pyx
//   Author: Paul Nation <nonhermitian@gmail.com>
//   Original source: QuTiP (qutip.org)
//   License: New BSD (C) 2014
//
// Key difference from Burkardt: seed node is always the lowest-degree
// unvisited node in each connected component (no pseudo-peripheral search),
// combined with insertion-sort by degree within each BFS level.

#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

#include "ordering_utils.hpp"

// ---------------------------------------------------------------------------
// Internal helpers (anonymous namespace)
// ---------------------------------------------------------------------------
namespace {

// Build a 0-based symmetric CSR from COO input.
// Convention matches the rest of sTiles: csr_i = col indices, csr_j = row indices.
// Only nodes 0..dim-1 are included (pinned nodes >= dim are excluded).
void build_csr(int** csr_i, int** csr_j, int nnz, int dim,
               std::vector<int>& ptr, std::vector<int>& ind)
{
    ptr.assign(dim + 1, 0);

    for (int k = 0; k < nnz; k++) {
        int row = (*csr_j)[k];
        int col = (*csr_i)[k];
        if (row < dim && col < dim && row != col) {
            ptr[row + 1]++;
            ptr[col + 1]++;
        }
    }

    for (int i = 1; i <= dim; i++) ptr[i] += ptr[i-1];

    ind.resize(ptr[dim]);
    std::vector<int> pos(ptr.begin(), ptr.begin() + dim);

    for (int k = 0; k < nnz; k++) {
        int row = (*csr_j)[k];
        int col = (*csr_i)[k];
        if (row < dim && col < dim && row != col) {
            ind[pos[row]++] = col;
            ind[pos[col]++] = row;
        }
    }
}

// Node degree = number of neighbours (self-loops excluded by build_csr).
std::vector<int> compute_degrees(const std::vector<int>& ptr, int n)
{
    std::vector<int> deg(n);
    for (int i = 0; i < n; i++) deg[i] = ptr[i+1] - ptr[i];
    return deg;
}

// BFS + insertion-sort RCM (direct C++ port of SciPy _reverse_cuthill_mckee).
// Returns 0-based permutation: order[new_position] = old_node.
std::vector<int> scipy_rcm_core(const std::vector<int>& ind,
                                  const std::vector<int>& ptr, int n)
{
    auto deg = compute_degrees(ptr, n);

    // inds = nodes sorted by ascending degree (argsort)
    std::vector<int> inds(n);
    std::iota(inds.begin(), inds.end(), 0);
    std::sort(inds.begin(), inds.end(),
              [&](int a, int b){ return deg[a] < deg[b]; });

    // rev_inds[node] = position of that node in inds
    std::vector<int> rev_inds(n);
    for (int i = 0; i < n; i++) rev_inds[inds[i]] = i;

    // flat visited array: single lookup instead of inds[rev_inds[nb]]
    std::vector<bool> visited(n, false);

    int max_deg = *std::max_element(deg.begin(), deg.end());
    std::vector<int> tmp_deg(std::max(max_deg + 1, 1));
    std::vector<int> order(n);
    int N = 0;

    for (int zz = 0; zz < n; zz++) {
        if (inds[zz] == -1) continue;   // already visited

        // Seed = next unvisited lowest-degree node
        int seed = inds[zz];
        order[N++] = seed;
        inds[rev_inds[seed]] = -1;
        visited[seed] = true;

        int level_start = N - 1;
        int level_end   = N;

        while (level_start < level_end) {
            for (int ii = level_start; ii < level_end; ii++) {
                int node  = order[ii];
                int N_old = N;

                // Enqueue unvisited neighbours
                for (int jj = ptr[node]; jj < ptr[node+1]; jj++) {
                    int nb = ind[jj];
                    if (!visited[nb]) {
                        visited[nb] = true;
                        inds[rev_inds[nb]] = -1;
                        order[N++] = nb;
                    }
                }

                // Insertion-sort the newly added neighbours by degree
                int level_len = N - N_old;
                for (int kk = 0; kk < level_len; kk++)
                    tmp_deg[kk] = deg[order[N_old + kk]];

                for (int kk = 1; kk < level_len; kk++) {
                    int td    = tmp_deg[kk];
                    int node2 = order[N_old + kk];
                    int ll    = kk;
                    while (ll > 0 && td < tmp_deg[ll-1]) {
                        tmp_deg[ll]       = tmp_deg[ll-1];
                        order[N_old + ll] = order[N_old + ll - 1];
                        ll--;
                    }
                    tmp_deg[ll]       = td;
                    order[N_old + ll] = node2;
                }
            }
            level_start = level_end;
            level_end   = N;
        }

        if (N == n) break;
    }

    // Reverse to get RCM ordering
    std::reverse(order.begin(), order.end());
    return order;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public C-ABI entry point
// ---------------------------------------------------------------------------
namespace sTiles {

int stiles_runRCM_scipy(int** csr_i, int** csr_j, int N, int nnz, int m,
                         int** perm, int** iperm, bool safe)
{
    if (!csr_i || !csr_j || !perm || !iperm || N <= 0 || nnz < 0 || m < 0 || m >= N) {
        fprintf(stderr, "stiles_runRCM_scipy: invalid arguments (N=%d, nnz=%d, m=%d)\n",
                N, nnz, m);
        return -1;
    }

    int dim = N - m;  // number of nodes to reorder; pinned nodes [dim, N) stay fixed

    // Build symmetric 0-based CSR for the active subgraph
    std::vector<int> ptr, ind;
    build_csr(csr_i, csr_j, nnz, dim, ptr, ind);

    // Run the SciPy BFS+RCM on the active subgraph
    std::vector<int> order = scipy_rcm_core(ind, ptr, dim);

    // Populate perm / iperm (0-based).
    // Convention (matches stiles_wrapRCM):
    //   perm[new_pos]  = old_node   (what node is at this new position)
    //   iperm[old_node] = new_pos   (where does this old node go)
    for (int k = 0; k < dim; k++) {
        (*perm)[k]         = order[k];
        (*iperm)[order[k]] = k;
    }
    // Pinned nodes map to themselves
    for (int i = dim; i < N; i++) {
        (*perm)[i]  = i;
        (*iperm)[i] = i;
    }

    if (safe) {
        std::vector<bool> seen(N, false);
        for (int i = 0; i < N; i++) {
            if ((*perm)[i] < 0 || (*perm)[i] >= N || seen[(*perm)[i]])
                throw std::runtime_error(
                    "stiles_runRCM_scipy: perm contains invalid or duplicate values");
            seen[(*perm)[i]] = true;
        }
    }

    return 1;
}

} // namespace sTiles
