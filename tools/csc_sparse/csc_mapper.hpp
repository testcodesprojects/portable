#ifndef STILES_CSC_SPARSE_MAPPER_HPP
#define STILES_CSC_SPARSE_MAPPER_HPP

#include <cstddef>
#include <vector>

namespace sTiles { namespace csc_sparse {

// Mapper: precomputed once after symbolic_phase. Holds the position in
// L_values where each Q nonzero (after permutation, lower-triangle only)
// must land. Q positions that fall on the strict-upper triangle of the
// permuted matrix are dropped (symmetric duplicate).
//
//   qmap[k] = -1   means Q(row[k], col[k]) is upper-triangular after perm
//                  and should be ignored by scatter_q_to_L.
//   qmap[k] >= 0   means scatter target is L_values[qmap[k]].
struct Mapper {
    std::vector<long long> qmap;     // size = nnz_Q
    long long              nnz_L{0}; // matches scheme.nnz_factor
};

// Build the Q -> L_values scatter map.
//
// Inputs (all from sTiles state after symbolic_phase has completed):
//   n           = matrix dimension
//   nnz_Q       = number of input COO entries from the user
//   row, col    = user's COO row/col indices, length nnz_Q (0-based)
//   element_perm = permutation chosen by symbolic_phase, length n.
//                  Row r in the original matrix becomes row element_perm[r]
//                  in the permuted (factorised) matrix.
//   L_colptr    = CSC column pointers for L, length n+1
//   L_rowind    = CSC row indices for L, length nnz_L
//   nnz_L       = scheme.nnz_factor
//
// On exit:
//   out.qmap is sized to nnz_Q, with qmap[k] = position in L_values for
//   the lower-triangular permuted (row, col) of Q[k], or -1 if upper.
//
// Returns true on success; false if any permuted lower entry is missing
// from the symbolic L pattern (would indicate a stale/incompatible
// symbolic factor for this Q).
bool build_mapper(int                       n,
                  int                       nnz_Q,
                  const int*                row,
                  const int*                col,
                  const int*                element_perm,
                  const int64_t*                L_colptr,
                  const int*                L_rowind,
                  long long                 nnz_L,
                  Mapper&                   out);

// Scatter user-supplied Q values into L_values using a prebuilt mapper.
// L_values must point to nnz_L doubles; the buffer is zeroed first so
// pure-fill positions in L start at 0 (then the numeric chol fills them).
void scatter_q_to_L(const Mapper&        m,
                    const double*        q_values,   // length = m.qmap.size()
                    double*              L_values);  // length = m.nnz_L

}} // namespace sTiles::csc_sparse

#endif // STILES_CSC_SPARSE_MAPPER_HPP
