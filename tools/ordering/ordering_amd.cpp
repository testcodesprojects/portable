// Wrapper for AMD (Approximate Minimum Degree, with aggressive absorption)
// Fortran source: external/amd.f
// Subroutine: AMD(N, PE, IW, LEN, IWLEN, PFREE, NV, NEXT, LAST, HEAD, ELEN, DEGREE, NCMPA, W)
// Output: LAST(i) = j means node i is the j-th in the new ordering (1-based)

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include "../common/stiles_logger.hpp"

// Fortran AMD subroutine (compiled with gfortran, appends underscore)
extern "C" {
    void amd_(int* N, int* PE, int* IW, int* LEN, int* IWLEN, int* PFREE,
              int* NV, int* NEXT, int* LAST, int* HEAD, int* ELEN,
              int* DEGREE, int* NCMPA, int* W);
}

// Build symmetric CSR (0-based, no diagonal) from COO input.
// Input COO may be upper/lower/full; diagonal entries are skipped.
static void coo_to_sym_csr(int* row, int* col, int nnz, int n,
                            std::vector<int>& xadj, std::vector<int>& adjncy)
{
    xadj.assign(n + 1, 0);
    for (int k = 0; k < nnz; k++) {
        int i = row[k], j = col[k];
        if (i != j) { xadj[i + 1]++; xadj[j + 1]++; }
    }
    for (int i = 1; i <= n; i++) xadj[i] += xadj[i - 1];
    adjncy.resize(xadj[n]);
    std::vector<int> pos(xadj.begin(), xadj.begin() + n);
    for (int k = 0; k < nnz; k++) {
        int i = row[k], j = col[k];
        if (i != j) { adjncy[pos[i]++] = j; adjncy[pos[j]++] = i; }
    }
}

// Convert symmetric CSR (0-based) to AMD internal format (1-based).
// After call: PE, IW, LEN are filled; IWLEN and PFREE are set.
static void csr_to_amd_format(const std::vector<int>& xadj,
                               const std::vector<int>& adjncy, int n,
                               std::vector<int>& PE, std::vector<int>& IW,
                               std::vector<int>& LEN, int& IWLEN, int& PFREE)
{
    PE.resize(n);
    LEN.resize(n);
    for (int i = 0; i < n; i++) {
        PE[i]  = xadj[i] + 1;                   // 1-based pointer into IW
        LEN[i] = xadj[i + 1] - xadj[i];         // number of off-diag neighbors
    }
    PFREE  = (int)adjncy.size() + 1;             // first free position
    // AMD needs up to 2*|E| + n workspace during elimination.
    // Using PFREE + n (minimum) causes heap corruption for larger matrices.
    IWLEN  = (int)adjncy.size() * 2 + n * 2 + 1; // safe upper bound
    IW.assign(IWLEN, 0);
    for (int k = 0; k < (int)adjncy.size(); k++)
        IW[k] = adjncy[k] + 1;                  // 1-based neighbor indices
}

namespace sTiles {

// stiles_runAMD
//   csr_i, csr_j : COO arrays (0-based, size nnz)
//   N             : number of nodes
//   nnz           : number of nonzero entries (including diagonal)
//   perm          : output permutation  (perm[new_pos]  = old_node, 0-based)
//   iperm         : output inv-perm     (iperm[old_node] = new_pos,  0-based)
void stiles_runAMD(int* csr_i, int* csr_j, int N, int nnz,
                   int** perm, int** iperm)
{
    // Build symmetric CSR
    std::vector<int> xadj, adjncy;
    coo_to_sym_csr(csr_i, csr_j, nnz, N, xadj, adjncy);

    // Convert to AMD internal format
    std::vector<int> PE, IW, LEN;
    int IWLEN, PFREE;
    csr_to_amd_format(xadj, adjncy, N, PE, IW, LEN, IWLEN, PFREE);

    // Workspace arrays
    std::vector<int> NV(N, 0), NEXT(N, 0), LAST(N, 0);
    std::vector<int> HEAD(N, 0), ELEN(N, 0), DEGREE(N, 0), W(N, 0);
    int NCMPA = 0;

    // Call Fortran AMD
    amd_(&N, PE.data(), IW.data(), LEN.data(), &IWLEN, &PFREE,
         NV.data(), NEXT.data(), LAST.data(), HEAD.data(),
         ELEN.data(), DEGREE.data(), &NCMPA, W.data());

    // Extract permutation: LAST(i) = j means node i (1-based) → position j (1-based)
    *iperm = (int*)malloc(N * sizeof(int));
    *perm  = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        (*iperm)[i] = LAST[i] - 1;              // convert to 0-based
    for (int i = 0; i < N; i++)
        (*perm)[(*iperm)[i]] = i;

    if (NCMPA > 0)
        sTiles::Logger::errorf("[AMD] %d garbage collections during ordering (N=%d)", NCMPA, N);
}

} // namespace sTiles
