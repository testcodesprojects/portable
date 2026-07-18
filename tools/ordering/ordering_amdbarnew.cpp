// Wrapper for AMDBARNEW (AMD without aggressive absorption — newer variant)
// Fortran source: external/amdbarNEW.f
// Subroutine: AMDBARNEW(N, PE, IW, LEN, IWLEN, PFREE, NV, NEXT, LAST, HEAD, ELEN, DEGREE, NCMPA, W)
// Newer version of AMDBAR from the MC47/AMD suite — no aggressive absorption.

#include <vector>
#include <cstdio>
#include <cstdlib>
#include "../common/stiles_logger.hpp"

extern "C" {
    void amdbarnew_(int* N, int* PE, int* IW, int* LEN, int* IWLEN, int* PFREE,
                    int* NV, int* NEXT, int* LAST, int* HEAD, int* ELEN,
                    int* DEGREE, int* NCMPA, int* W);
}

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

static void csr_to_amd_format(const std::vector<int>& xadj,
                               const std::vector<int>& adjncy, int n,
                               std::vector<int>& PE, std::vector<int>& IW,
                               std::vector<int>& LEN, int& IWLEN, int& PFREE)
{
    PE.resize(n);
    LEN.resize(n);
    for (int i = 0; i < n; i++) {
        PE[i]  = xadj[i] + 1;
        LEN[i] = xadj[i + 1] - xadj[i];
    }
    PFREE  = (int)adjncy.size() + 1;
    IWLEN  = (int)adjncy.size() * 2 + n * 2 + 1; // safe upper bound
    IW.assign(IWLEN, 0);
    for (int k = 0; k < (int)adjncy.size(); k++)
        IW[k] = adjncy[k] + 1;
}

namespace sTiles {

void stiles_runAMDBARNEW(int* csr_i, int* csr_j, int N, int nnz,
                         int** perm, int** iperm)
{
    std::vector<int> xadj, adjncy;
    coo_to_sym_csr(csr_i, csr_j, nnz, N, xadj, adjncy);

    std::vector<int> PE, IW, LEN;
    int IWLEN, PFREE;
    csr_to_amd_format(xadj, adjncy, N, PE, IW, LEN, IWLEN, PFREE);

    std::vector<int> NV(N, 0), NEXT(N, 0), LAST(N, 0);
    std::vector<int> HEAD(N, 0), ELEN(N, 0), DEGREE(N, 0), W(N, 0);
    int NCMPA = 0;

    amdbarnew_(&N, PE.data(), IW.data(), LEN.data(), &IWLEN, &PFREE,
               NV.data(), NEXT.data(), LAST.data(), HEAD.data(),
               ELEN.data(), DEGREE.data(), &NCMPA, W.data());

    *iperm = (int*)malloc(N * sizeof(int));
    *perm  = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        (*iperm)[i] = LAST[i] - 1;
    for (int i = 0; i < N; i++)
        (*perm)[(*iperm)[i]] = i;

    if (NCMPA > 0)
        sTiles::Logger::errorf("[AMDBARNEW] %d garbage collections (N=%d)", NCMPA, N);
}

} // namespace sTiles
