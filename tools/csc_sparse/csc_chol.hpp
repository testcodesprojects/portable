#ifndef STILES_CSC_SPARSE_CHOL_HPP
#define STILES_CSC_SPARSE_CHOL_HPP

namespace sTiles { namespace csc_sparse {

enum class CholStatus {
    Success      = 0,
    NotPosDef    = 1,   // negative or zero pivot encountered
    BadStructure = 2,   // L pattern is missing a required entry
};

// In-place numeric sparse Cholesky on a precomputed lower-triangular CSC
// pattern.
//
// Input pre-conditions:
//   * L_colptr / L_rowind hold the *exact symbolic* Cholesky pattern of a
//     symmetric positive definite A (i.e. the pattern returned by
//     symbolic_chol_fillin in symbolic_phase).
//   * L_values is sized to L_colptr[n] and has been populated with the
//     lower triangle of the permuted A by scatter_q_to_L. Entries that
//     belong to L's structural fill (not present in A) are 0.
//   * Within each column j, L_rowind[L_colptr[j] .. L_colptr[j+1]-1] is
//     sorted ascending, and the very first entry of every column is the
//     diagonal (row == j). This is what symbolic_chol_fillin produces.
//
// On exit (Success):
//   * L_values holds the numeric lower-triangular Cholesky factor:
//     L_values[L_colptr[j]] is L(j,j) (positive), and the off-diagonals
//     follow in the same slots as the pattern.
//
// Algorithm:
//   Left-looking sparse Cholesky in the style of Davis (CSparse, cs_chol).
//   For each column j:
//     1. Gather A(:, j) (already in L_values for the lower part) into a
//        dense workspace x of length n.
//     2. For every previous column k that has a nonzero in row j
//        (i.e. L(j,k) != 0), apply x -= L(k+1:, k) * L(j,k).
//        We discover the set of such k by walking up the elimination tree
//        rooted at j, but here we use the simpler approach: scan the
//        L pattern column-by-column and use a per-row "first nz below
//        diagonal" index to know which columns contribute to j.
//     3. Take L(j,j) = sqrt(x[j]); scale L(j+1:, j) = x(rows in pattern) / L(j,j).
//
// Complexity matches the FLOP count of the symbolic factor (sum over
// columns of (col nnz)^2-ish, exact value depends on the pattern).
CholStatus chol_numeric(int           n,
                        const int64_t*    L_colptr,
                        const int*    L_rowind,
                        double*       L_values);

}} // namespace sTiles::csc_sparse

#endif // STILES_CSC_SPARSE_CHOL_HPP
