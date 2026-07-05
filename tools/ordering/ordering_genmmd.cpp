// Wrapper for GENMMD (Generalized Multiple Minimum Degree)
// Fortran source: external/genmmd.f
// Subroutine: GENMMD(NEQNS, XADJ, ADJNCY, INVP, PERM, DELTA, DHEAD, QSIZE, LLIST, MARKER, MAXINT, NOFSUB)
//
// Takes standard CSR (1-based) and produces:
//   PERM(k) = i  — the k-th node eliminated is node i  (1-based)
//   INVP(i) = k  — node i is the k-th to be eliminated (1-based)
//
// DELTA: tolerance for multiple elimination. Use 0 for pure minimum degree,
//        or a small positive value (e.g. 1) to allow multiple elimination.
// MAXINT: largest representable integer (used internally for marking). Use INT_MAX.

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <climits>

extern "C" {
    void genmmd_(int* NEQNS, int* XADJ, int* ADJNCY,
                 int* INVP, int* PERM,
                 int* DELTA, int* DHEAD, int* QSIZE,
                 int* LLIST, int* MARKER,
                 int* MAXINT, int* NOFSUB);
}

static void coo_to_sym_csr_1based(int* row, int* col, int nnz, int n,
                                   std::vector<int>& xadj, std::vector<int>& adjncy)
{
    // Build symmetric CSR (1-based, no diagonal), as expected by GENMMD.
    std::vector<int> xadj0(n + 1, 0);
    for (int k = 0; k < nnz; k++) {
        int i = row[k], j = col[k];
        if (i != j) { xadj0[i + 1]++; xadj0[j + 1]++; }
    }
    for (int i = 1; i <= n; i++) xadj0[i] += xadj0[i - 1];
    std::vector<int> adj0(xadj0[n]);
    std::vector<int> pos(xadj0.begin(), xadj0.begin() + n);
    for (int k = 0; k < nnz; k++) {
        int i = row[k], j = col[k];
        if (i != j) { adj0[pos[i]++] = j + 1; adj0[pos[j]++] = i + 1; }
    }

    // Convert xadj to 1-based
    xadj.resize(n + 1);
    for (int i = 0; i <= n; i++) xadj[i] = xadj0[i] + 1;
    adjncy = adj0;
}

namespace sTiles {

// delta: multiple-elimination tolerance. 0 = pure min-degree.
void stiles_runGenMMD(int* csr_i, int* csr_j, int N, int nnz,
                      int** perm, int** iperm,
                      int delta = 0)
{
    std::vector<int> xadj, adjncy;
    coo_to_sym_csr_1based(csr_i, csr_j, nnz, N, xadj, adjncy);

    std::vector<int> PERM(N, 0), INVP(N, 0);
    std::vector<int> DHEAD(N, 0), QSIZE(N, 0), LLIST(N, 0), MARKER(N, 0);
    int MAXINT = INT_MAX;
    int NOFSUB = 0;

    genmmd_(&N, xadj.data(), adjncy.data(),
            INVP.data(), PERM.data(),
            &delta, DHEAD.data(), QSIZE.data(),
            LLIST.data(), MARKER.data(),
            &MAXINT, &NOFSUB);

    // PERM(k) = i (1-based): k-th eliminated node is i → perm[k-1] = i-1 (0-based)
    // INVP(i) = k (1-based): node i at position k   → iperm[i-1] = k-1 (0-based)
    *perm  = (int*)malloc(N * sizeof(int));
    *iperm = (int*)malloc(N * sizeof(int));
    for (int k = 0; k < N; k++)
        (*perm)[k] = PERM[k] - 1;
    for (int i = 0; i < N; i++)
        (*iperm)[i] = INVP[i] - 1;

}

} // namespace sTiles
