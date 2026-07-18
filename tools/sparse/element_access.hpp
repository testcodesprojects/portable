/**
 * @file    element_access.hpp
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

#ifndef _STILES_SPARSE_ELEMENT_ACCESS_HPP_
#define _STILES_SPARSE_ELEMENT_ACCESS_HPP_

// Element/row accessors for L (Cholesky factor) and Z (selected inverse).
//
// Both L_cs and Z_cs use the same supernodal cell layout (Z_cs is built via
// `CellStore::allocate_like(L_cs)`), so the same lookup logic works for
// either store. Indexing convention is **0-based original** (the indexing
// the caller's input matrix used) — these functions internally permute
// through `sym.ordering.invp` to reach the factor coordinates that the
// CellStore actually stores in.
//
// Mirrors sTiles' `sTiles_get_chol_elm` / `sTiles_get_selinv_elm` and
// `sTiles_get_selinv_row` so side-by-side regression checks call into
// the same shape of API on both sides.

#include "symbolic.hpp"
#include "supernode.hpp"

#include <vector>

namespace sTiles { namespace sparse {

// Single element: return the value the factor stores at original-index (i, j).
// Returns 0.0 if the position is structurally zero (not in pattern(L+L^T))
// or if it lives in the upper triangle of factor coords AND the caller
// asks for it directly (we mirror via swap so this never happens here —
// upper-triangle requests fetch the symmetric stored counterpart).
//
// Intended uses:
//   - sps_get_chol_elm   : `cs` is the L CellStore filled by factorize.
//   - sps_get_selinv_elm : `cs` is the Z CellStore filled by sTiles::sparse::selinv.
//
// Both names alias the same lookup; they exist so call-sites read clearly
// and so future divergence (e.g. clamping for L's strict-lower convention)
// has a place to live.
double sps_get_chol_elm  (const Symbolic& sym, const CellStore& L_cs, int i, int j);
double sps_get_selinv_elm(const Symbolic& sym, const CellStore& Z_cs, int i, int j);

// Lower-triangle column extraction.
//
// For an original column index `j_orig`, returns every stored entry whose
// **factor column** equals `invp[j_orig]` — i.e. the rows of the supernode
// J ⊇ j_orig that own this column, gathered across all cells (K, J) with
// K ≥ J. The output is in original coords:
//
//   rows[k] = original row index i (0-based)
//   vals[k] = stored value at (i, j_orig)
//
// For L this is exactly column j_orig of the lower-triangular factor in
// original coords. For Z = A^{-1}|pattern(L+L^T), this is the lower
// triangle of column j_orig; the upper triangle of the same column equals
// row j_orig of Z by symmetry — use `sps_get_selinv_neighbors` below to
// get both halves at once (the natural "node + its neighbors" view).
void sps_get_chol_column(const Symbolic&      sym,
                         const CellStore&     L_cs,
                         int                  j_orig,
                         std::vector<int>&    rows,
                         std::vector<double>& vals);

// All neighbors of node `j_orig` in Z, with values.
//
// Z is symmetric on pattern(L+L^T), so the full set of i with Z[i, j_orig]
// != 0 is: { i : (i, j_orig) ∈ pattern(L+L^T) }. We assemble it by
//   - taking the lower-triangle column (factor sense) of j_orig, then
//   - taking the lower-triangle row of j_orig (= upper-triangle column),
// translating both to original coords. The diagonal entry is included
// once. `rows` and `vals` are not sorted — caller can `std::sort` if a
// canonical order is needed.
//
// O(width(supernode-of-j) + |cells touching j|) — bounded by the
// supernodal pattern, NOT by n.
void sps_get_selinv_neighbors(const Symbolic&      sym,
                              const CellStore&     Z_cs,
                              int                  j_orig,
                              std::vector<int>&    rows,
                              std::vector<double>& vals);

}}  // namespace sTiles::sparse

#endif  // _STILES_SPARSE_ELEMENT_ACCESS_HPP_
