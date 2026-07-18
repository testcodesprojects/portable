#include "csc_mapper.hpp"

#include <algorithm>
#include <cstring>

namespace sTiles { namespace csc_sparse {

bool build_mapper(int           n,
                  int           nnz_Q,
                  const int*    row,
                  const int*    col,
                  const int*    element_perm,
                  const int64_t*    L_colptr,
                  const int*    L_rowind,
                  long long     nnz_L,
                  Mapper&       out)
{
    out.nnz_L = nnz_L;
    out.qmap.assign(nnz_Q, -1);

    for (int k = 0; k < nnz_Q; ++k) {
        int r = element_perm[row[k]];
        int c = element_perm[col[k]];
        // Keep lower triangle only (r >= c). The other half is the
        // symmetric duplicate and must not double-contribute to L.
        if (r < c) {
            // upper-triangular entry — skip, leave qmap[k] = -1
            continue;
        }
        // Binary search r in L_rowind[L_colptr[c] .. L_colptr[c+1]).
        // L_rowind is sorted within each column (symbolic_phase sorts it).
        const int64_t beg = L_colptr[c];
        const int64_t end = L_colptr[c + 1];
        const int* lo = L_rowind + beg;
        const int* hi = L_rowind + end;
        const int* it = std::lower_bound(lo, hi, r);
        if (it == hi || *it != r) {
            // r,c is not in L's symbolic pattern — incompatible factor
            return false;
        }
        out.qmap[k] = static_cast<long long>(it - L_rowind);
    }
    return true;
}

void scatter_q_to_L(const Mapper&  m,
                    const double*  q_values,
                    double*        L_values)
{
    std::memset(L_values, 0, static_cast<std::size_t>(m.nnz_L) * sizeof(double));
    const long long nnz_Q = static_cast<long long>(m.qmap.size());
    for (long long k = 0; k < nnz_Q; ++k) {
        const long long pos = m.qmap[k];
        if (pos < 0) continue;          // upper-triangular Q entry
        // '+=' so duplicate COO entries on the same (r,c) accumulate;
        // matches the convention used elsewhere in sTiles when building
        // the symmetric Q from user COO.
        L_values[pos] += q_values[k];
    }
}

}} // namespace sTiles::csc_sparse
