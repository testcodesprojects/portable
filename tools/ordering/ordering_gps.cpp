// Wrapper for GPSKCA (Gibbs-Poole-Stockmeyer / Gibbs-King bandwidth & profile reduction)
// Fortran source: external/acm582.F
// Subroutine: GPSKCA(N, DEGREE, RSTART, CONNEC, IOPTPRO, WRKLEN, PERMUT, WORK, BANDWD, PROFIL, ERROR, SPACE)
//
// Input graph format (1-based adjacency list):
//   DEGREE(i)  = number of neighbors of node i
//   RSTART(i)  = start index in CONNEC for node i  (1-based)
//   CONNEC(*)  = concatenated neighbor lists       (1-based node numbers)
//
// ioptpro = 0 → bandwidth reduction (Gibbs-Poole-Stockmeyer)
// ioptpro = 1 → profile reduction   (Gibbs-King)
//
// Output:
//   PERMUT(i) = j (1-based): node i gets new index j → this is iperm.

#include <vector>
#include <algorithm>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <numeric>

extern "C" {
    void gpskca_(int* N, int* DEGREE, int* RSTART, int* CONNEC,
                 int* IOPTPRO, int* WRKLEN, int* PERMUT, int* WORK,
                 int* BANDWD, int* PROFIL, int* ERROR, int* SPACE);
}

// Build adjacency list format required by GPSKCA from COO input.
// Returns degree[], rstart[], connec[] (all 1-based).
static void coo_to_gps_format(int* row, int* col, int nnz, int n,
                               std::vector<int>& degree,
                               std::vector<int>& rstart,
                               std::vector<int>& connec)
{
    // Count degrees (off-diagonal only, symmetric)
    degree.assign(n, 0);
    for (int k = 0; k < nnz; k++) {
        int i = row[k], j = col[k];
        if (i != j) { degree[i]++; degree[j]++; }
    }

    // Build rstart (1-based): rstart[i] = start position in connec for node i+1
    rstart.resize(n);
    rstart[0] = 1;
    for (int i = 1; i < n; i++)
        rstart[i] = rstart[i - 1] + degree[i - 1];

    int total = rstart[n - 1] + degree[n - 1] - 1;  // 1-based: last slot = rstart[n-1]+degree[n-1]-1
    connec.assign(total, 0);

    // Fill connec (1-based neighbor indices)
    std::vector<int> pos(rstart.begin(), rstart.end());
    for (int k = 0; k < nnz; k++) {
        int i = row[k], j = col[k];
        if (i != j) {
            connec[pos[i] - 1] = j + 1;  // 1-based node number
            connec[pos[j] - 1] = i + 1;
            pos[i]++;
            pos[j]++;
        }
    }
}

namespace sTiles {

// mode: 0 = bandwidth reduction (GPS), 1 = profile reduction (Gibbs-King)
void stiles_runGPS(int* csr_i, int* csr_j, int N, int nnz,
                   int** perm, int** iperm,
                   int mode = 0)
{
    // coo_to_gps_format expects each undirected edge exactly once (lower triangle,
    // i > j for off-diagonal). The input may be fully symmetric, lower-tri-only,
    // upper-tri-only, or asymmetric (after preprocessing). Canonicalize by
    // converting each (i,j) off-diagonal pair to (max,min) and deduplicating.
    std::vector<std::pair<int,int>> edge_set;
    edge_set.reserve(nnz);
    for (int k = 0; k < nnz; ++k) {
        int i = csr_i[k], j = csr_j[k];
        if (i != j)
            edge_set.push_back({std::max(i,j), std::min(i,j)});  // canonical lower-tri
    }
    std::sort(edge_set.begin(), edge_set.end());
    edge_set.erase(std::unique(edge_set.begin(), edge_set.end()), edge_set.end());

    // Build diagonal + deduplicated lower-tri edges for coo_to_gps_format
    std::vector<int> row_lt, col_lt;
    row_lt.reserve(N + (int)edge_set.size());
    col_lt.reserve(N + (int)edge_set.size());
    for (int i = 0; i < N; ++i) { row_lt.push_back(i); col_lt.push_back(i); }
    for (auto& [r, c] : edge_set) { row_lt.push_back(r); col_lt.push_back(c); }
    int nnz_lt = (int)row_lt.size();

    std::vector<int> degree, rstart, connec;
    coo_to_gps_format(row_lt.data(), col_lt.data(), nnz_lt, N, degree, rstart, connec);

    int wrklen = 6 * N + 3;
    std::vector<int> PERMUT(N), WORK(wrklen);

    // Initialize PERMUT to identity (1-based), as expected by GPSKCA
    for (int i = 0; i < N; i++) PERMUT[i] = i + 1;

    int IOPTPRO = mode;
    int BANDWD = 0, PROFIL = 0, ERROR = 0, SPACE = 0;

    gpskca_(&N, degree.data(), rstart.data(), connec.data(),
            &IOPTPRO, &wrklen, PERMUT.data(), WORK.data(),
            &BANDWD, &PROFIL, &ERROR, &SPACE);

    if (ERROR)
        fprintf(stderr, "[GPS] gpskca_ returned error=%d (N=%d)\n", ERROR, N);

    // PERMUT(i) = j (1-based): node i → new position j → iperm
    *iperm = (int*)malloc(N * sizeof(int));
    *perm  = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        (*iperm)[i] = PERMUT[i] - 1;
    for (int i = 0; i < N; i++)
        (*perm)[(*iperm)[i]] = i;

}

} // namespace sTiles
