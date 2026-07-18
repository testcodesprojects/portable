/**
 * @file    element_access.cpp
 * @brief   Element/row accessors for the factor L and the selected inverse Z.
 *
 * @project sTiles (Sparse Tiles Library)
 * @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @version 3.0.0
 * @date 1 1 2026
 * @license Proprietary
 *
 * @note This file is part of the sTiles library, a proprietary software package.
 *       Redistribution or modification without prior permission is prohibited.
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
 *
 * @license
 * This software is proprietary and confidential. Unauthorized copying, distribution, or modification
 * of this software, via any medium, is strictly prohibited. Permission is granted to use the software
 * in binary form for non-commercial purposes only, provided that this copyright notice and permission
 * notice are included in all copies or substantial portions of the software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "element_access.hpp"

#include <algorithm>

namespace sTiles { namespace sparse {

namespace {

// Single (r, c) lookup in factor coords (0-based, r >= c required for the
// caller — we don't swap inside; the public element APIs handle that).
// Returns 0 if the cell isn't stored or the row isn't in its pattern.
inline double get_at(const Symbolic& sym, const CellStore& cs, int r, int c) {
    const Int  supJ = sym.supernode_of_col[r];
    const Int  supI = sym.supernode_of_col[c];
    const Cell* cell = cs.find(supJ, supI);
    if (!cell) return 0.0;
    const int      col_off = c - (sym.supernode_first_col[supI - 1] - 1);
    const Idx*     base    = &sym.row_pattern[cell->lx_offset - 1];
    const Idx*     end_    = base + cell->rows;
    const Idx      target  = static_cast<Idx>(r + 1);
    const Idx*     it      = std::lower_bound(base, end_, target);
    if (it == end_ || *it != target) return 0.0;
    const int      row_off = static_cast<int>(it - base);
    return cell->nzval[row_off + cell->rows * col_off];
}

}  // namespace

double sps_get_chol_elm(const Symbolic& sym, const CellStore& L_cs, int i, int j) {
    int r = sym.ordering.invp[i] - 1;
    int c = sym.ordering.invp[j] - 1;
    if (r < c) std::swap(r, c);     // factor stores J >= I only — fetch the
                                                                      // symmetric counterpart for upper-tri queries
    return get_at(sym, L_cs, r, c);
}

double sps_get_selinv_elm(const Symbolic& sym, const CellStore& Z_cs, int i, int j) {
    int r = sym.ordering.invp[i] - 1;
    int c = sym.ordering.invp[j] - 1;
    if (r < c) std::swap(r, c);
    return get_at(sym, Z_cs, r, c);
}

void sps_get_chol_column(const Symbolic&      sym,
                         const CellStore&     L_cs,
                         int                  j_orig,
                         std::vector<int>&    rows,
                         std::vector<double>& vals) {
    rows.clear();
    vals.clear();
    const int  c       = sym.ordering.invp[j_orig] - 1;     // factor col, 0-based
    const Int  supI    = sym.supernode_of_col[c];              // 1-based
    const int  col_off = c - (sym.supernode_first_col[supI - 1] - 1);

    // Cells of column-supernode supI are listed contiguously by walking
    // cells_[] (CellStore lists them grouped by I). We don't have a public
    // group iterator, so use cs.find(J, supI) for J = supI .. n_super and
    // skip nullptrs — cheap because cs.find is a hashmap probe.
    for (Int J = supI; J <= sym.n_super; ++J) {
        const Cell* cell = L_cs.find(J, supI);
        if (!cell) continue;
        const Idx* base = &sym.row_pattern[cell->lx_offset - 1];
        for (Int rr = 0; rr < cell->rows; ++rr) {
            const int  r_factor = base[rr] - 1;                 // 0-based factor row
            const int  i_orig   = sym.ordering.perm[r_factor] - 1;
            // Diagonal supernode (J == supI): only the lower triangle of the
            // diag block is part of L (strict lower + diagonal). Entries with
            // r_factor < c are upper-tri of the diag block — drop them.
            if (J == supI && r_factor < c) continue;
            rows.push_back(i_orig);
            vals.push_back(cell->nzval[rr + cell->rows * col_off]);
        }
    }
}

void sps_get_selinv_neighbors(const Symbolic&      sym,
                              const CellStore&     Z_cs,
                              int                  j_orig,
                              std::vector<int>&    rows,
                              std::vector<double>& vals) {
    rows.clear();
    vals.clear();
    const int  c       = sym.ordering.invp[j_orig] - 1;
    const Int  supI    = sym.supernode_of_col[c];
    const int  col_off = c - (sym.supernode_first_col[supI - 1] - 1);

    // (1) Lower-triangle column of factor coords: cells (J, supI) for J >= supI.
    for (Int J = supI; J <= sym.n_super; ++J) {
        const Cell* cell = Z_cs.find(J, supI);
        if (!cell) continue;
        const Idx* base = &sym.row_pattern[cell->lx_offset - 1];
        for (Int rr = 0; rr < cell->rows; ++rr) {
            const int r_factor = base[rr] - 1;
            const int i_orig   = sym.ordering.perm[r_factor] - 1;
            rows.push_back(i_orig);
            vals.push_back(cell->nzval[rr + cell->rows * col_off]);
        }
    }

    // (2) Upper-triangle column of factor coords = lower-triangle row, by Z's
    //     symmetry. Z[r, c] for r < c is stored as Z[c, r] in the cell whose
    //     COLUMN supernode is memb[r] and whose ROW supernode contains c. We
    //     enumerate by walking column-supernodes I = 1 .. supI - 1 and asking:
    //     does I's row pattern (row_pattern[row_pattern_ptr[I-1]-1 .. row_pattern_ptr[I]-2]) contain
    //     `c` ? If yes, the cell (supI, I) holds Z[c, k] for every k in I's
    //     column range — pick the row offset corresponding to `c` and gather.
    for (Int I = 1; I < supI; ++I) {
        const Cell* cell = Z_cs.find(supI, I);
        if (!cell) continue;
        const Idx* base   = &sym.row_pattern[cell->lx_offset - 1];
        const Idx* end_   = base + cell->rows;
        const Idx  target = static_cast<Idx>(c + 1);
        const Idx* it     = std::lower_bound(base, end_, target);
        if (it == end_ || *it != target) continue;            // c not in this cell
        const int  row_off = static_cast<int>(it - base);
        const int  Icol_b  = sym.supernode_first_col[I - 1] - 1;           // 0-based first col of I
        const int  Icol_e  = sym.supernode_first_col[I]     - 1;           // 0-based one-past-last
        for (int kk = Icol_b; kk < Icol_e; ++kk) {
            const int  i_orig = sym.ordering.perm[kk] - 1;
            const int  cco    = kk - Icol_b;                    // col offset within cell
            rows.push_back(i_orig);
            vals.push_back(cell->nzval[row_off + cell->rows * cco]);
        }
    }
}

}}  // namespace sTiles::sparse
