#include "csc_chol.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace sTiles { namespace csc_sparse {

// Left-looking sparse Cholesky on a lower-CSC pattern (column-major).
//
// Workspaces:
//   x[0..n-1]           — dense rhs/accumulator for the current column.
//   next_idx[k]         — next position in column k to read while applying
//                         it to a future column. Initialised to L_colptr[k]+1
//                         (skip k's diagonal); advanced as k contributes
//                         to columns j > k.
//   col_head[j]         — linked list of column indices k whose next
//                         contribution row is j. We pop the whole list
//                         when processing column j (left-looking update).
//   col_next[k]         — next pointer in the linked list (head/next arrays).
//
// Why the linked list: a left-looking step at column j needs all columns
// k < j with L(j,k) != 0. Maintaining a per-row bucket of "columns whose
// current head is in row j" makes that lookup O(updates_to_j) without any
// search. Standard trick from CSparse / Eisenstat-Liu.
CholStatus chol_numeric(int          n,
                        const int64_t*   L_colptr,
                        const int*   L_rowind,
                        double*      L_values)
{
    std::vector<double> x(n, 0.0);
    std::vector<int64_t> next_idx(n, 0);   // holds CSC offsets (can exceed INT_MAX)
    std::vector<int>    col_head(n, -1);
    std::vector<int>    col_next(n, -1);

    // Seed: every column k contributes first to its first sub-diagonal row.
    // If column k has no sub-diagonal entry (e.g. isolated node), it never
    // updates anyone and stays out of the lists.
    for (int k = 0; k < n; ++k) {
        const int64_t p = L_colptr[k];
        const int64_t q = L_colptr[k + 1];
        // First entry must be the diagonal (row == k). The first sub-diag,
        // if any, is at p+1.
        if (q - p >= 2) {
            const int first_sub_row = L_rowind[p + 1];
            next_idx[k] = p + 1;
            col_next[k] = col_head[first_sub_row];
            col_head[first_sub_row] = k;
        } else {
            next_idx[k] = q;        // exhausted
        }
    }

    for (int j = 0; j < n; ++j) {
        const int64_t pj  = L_colptr[j];
        const int64_t pj1 = L_colptr[j + 1];

        // 1. Gather column j of A (currently sitting in L_values at the
        //    pattern slots) into dense x. Then zero those slots so the
        //    accumulating updates land cleanly; we'll write final L back
        //    into them at the end.
        if (pj1 == pj || L_rowind[pj] != j) {
            // Diagonal missing — pattern is broken.
            return CholStatus::BadStructure;
        }
        for (int64_t p = pj; p < pj1; ++p) {
            x[L_rowind[p]] = L_values[p];
            L_values[p]    = 0.0;
        }

        // 2. Apply contributions from all columns k < j with L(j,k) != 0.
        //    These are exactly the columns currently chained at col_head[j].
        int k = col_head[j];
        col_head[j] = -1;
        while (k != -1) {
            const int kk_next  = col_next[k];
            const int64_t p_jk     = next_idx[k];        // points at L(j,k)
            const double L_jk  = L_values[p_jk];
            const int64_t kend     = L_colptr[k + 1];

            // x(rows in col k, from j onward) -= L(rows, k) * L(j, k)
            for (int64_t p = p_jk; p < kend; ++p) {
                x[L_rowind[p]] -= L_values[p] * L_jk;
            }

            // Advance column k past row j; re-bucket it under its next row.
            const int64_t p_next = p_jk + 1;
            next_idx[k] = p_next;
            if (p_next < kend) {
                const int next_row = L_rowind[p_next];
                col_next[k] = col_head[next_row];
                col_head[next_row] = k;
            }
            k = kk_next;
        }

        // 3. Pivot + scale.
        const double djj = x[j];
        if (!(djj > 0.0)) {
            return CholStatus::NotPosDef;
        }
        const double Ljj = std::sqrt(djj);
        L_values[pj]    = Ljj;
        x[j]            = 0.0;       // clean workspace for next column

        const double invLjj = 1.0 / Ljj;
        for (int64_t p = pj + 1; p < pj1; ++p) {
            const int    r = L_rowind[p];
            L_values[p]    = x[r] * invLjj;
            x[r]           = 0.0;    // clean workspace
        }
    }

    return CholStatus::Success;
}

}} // namespace sTiles::csc_sparse
