#include "selinv.hpp"

#include "kernels.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace sTiles { namespace sparse {

namespace {

struct ColIndex {
  std::vector<Int>              diag_cell;
  std::vector<std::vector<Int>> off_cells;
};

ColIndex build_col_index(const Symbolic& s, const CellStore& cs) {
  ColIndex ci;
  ci.diag_cell.assign(s.n_super + 1, -1);
  ci.off_cells.assign(s.n_super + 1, {});
  Int next = 0;
  for (Int I = 1; I <= s.n_super; ++I) {
    if (next >= cs.cell_count() || cs.at(next).I != I || cs.at(next).J != I) {
      throw std::logic_error("selinv: missing diagonal cell at column I");
    }
    ci.diag_cell[I] = next;
    ++next;
    while (next < cs.cell_count() && cs.at(next).I == I) {
      ci.off_cells[I].push_back(next);
      ++next;
    }
  }
  return ci;
}

// After dlauum('L'), the lower triangle of A holds L^T·L; mirror to make it
// fully symmetric so subsequent GEMMs read a consistent matrix.
void mirror_low_to_up(int n, double* A, int lda) {
  for (int j = 0; j < n; ++j) {
    for (int i = j + 1; i < n; ++i) {
      A[j + lda * i] = A[i + lda * j];
    }
  }
}

// Find target's position within sorted row_pattern slice [base, base+n). Returns
// -1 if not present. Used to align row sets between cells.
Int find_pos(Idx target, const Idx* base, Int n) {
  // n is small in practice; linear scan keeps the inner loop branchless.
  for (Int i = 0; i < n; ++i) if (base[i] == target) return i;
  return -1;
}

}  // namespace

void selinv(const Symbolic& s, const CellStore& L_cs, CellStore& Z_cs) {
  Z_cs.allocate_like(L_cs);
  ColIndex ci = build_col_index(s, Z_cs);

  // ─────────── Phase 1 ───────────
  // Z's cells are populated with M = L^{-1}:
  //   Z[I, I] := L[I, I]^{-1}                    (TRTRI)
  //   Z[J, I] := L[J, I] · L[I, I]^{-1}          (right TRSM)
  for (Int I = 1; I <= s.n_super; ++I) {
    Cell&       Z_diag = Z_cs.at(ci.diag_cell[I]);
    const Cell& L_diag = L_cs.at(ci.diag_cell[I]);
    const Int   w      = Z_diag.cols;

    std::copy(L_diag.nzval, L_diag.nzval + (size_t)w * w, Z_diag.nzval);
    int info = kernels::trtri('L', 'N', w, Z_diag.nzval, Z_diag.rows);
    if (info != 0) {
      throw std::runtime_error("selinv: dtrtri failed (matrix not invertible)");
    }

    for (Int off_idx : ci.off_cells[I]) {
      Cell&       Z_off = Z_cs.at(off_idx);
      const Cell& L_off = L_cs.at(off_idx);
      std::copy(L_off.nzval, L_off.nzval + (size_t)L_off.rows * L_off.cols,
                Z_off.nzval);
      kernels::trsm('R', 'L', 'N', 'N',
                    Z_off.rows, Z_off.cols,
                    1.0,
                    L_diag.nzval, L_diag.rows,
                    Z_off.nzval,  Z_off.rows);
    }
  }

  // ─────────── Phase 2 ───────────
  // Reverse-order Takahashi sweep with row-subset gather over the cell's
  // active rows. For each K-contribution we build a tiny aligned temp so a
  // single dgemm directly produces the contribution in acc's coordinate
  // system.
  std::vector<double> M_diag_buf;
  std::vector<double> M_off_buf;        // concatenated off-diag M-cells

  for (Int I = s.n_super; I >= 1; --I) {
    Cell& Z_diag = Z_cs.at(ci.diag_cell[I]);
    const Int w_I = Z_diag.cols;
    auto& off_list = ci.off_cells[I];

    // Save M's for column I before we overwrite.
    M_diag_buf.assign(Z_diag.nzval, Z_diag.nzval + (size_t)w_I * w_I);
    {
      size_t total_off = 0;
      for (Int idx : off_list) total_off += (size_t)Z_cs.at(idx).rows * w_I;
      M_off_buf.assign(total_off, 0.0);
      size_t cur = 0;
      for (Int idx : off_list) {
        const Cell& c = Z_cs.at(idx);
        std::copy(c.nzval, c.nzval + (size_t)c.rows * w_I,
                  M_off_buf.data() + cur);
        cur += (size_t)c.rows * w_I;
      }
    }
    std::vector<size_t> M_off_starts(off_list.size() + 1, 0);
    for (size_t i = 0; i < off_list.size(); ++i) {
      M_off_starts[i + 1] =
          M_off_starts[i] + (size_t)Z_cs.at(off_list[i]).rows * w_I;
    }
    auto M_off_ptr = [&](size_t a) -> const double* {
      return M_off_buf.data() + M_off_starts[a];
    };

    // Off-diagonal Z[J, I] for each J in pattern (DESCENDING).
    for (size_t b_idx = off_list.size(); b_idx-- > 0; ) {
      Int idx_JI = off_list[b_idx];
      Cell& Z_JI = Z_cs.at(idx_JI);
      const Int   J        = Z_JI.J;
      const Int   rJ_in_I  = Z_JI.rows;
      const Int   J_super0 = s.supernode_first_col[J - 1];
      const Idx*  JI_rows  = &s.row_pattern[Z_JI.lx_offset - 1];

      // Local-in-J positions for cell (J, I)'s rows.
      std::vector<Int> p_J(rJ_in_I);
      for (Int a = 0; a < rJ_in_I; ++a) p_J[a] = JI_rows[a] - J_super0;

      std::vector<double> acc((size_t)rJ_in_I * w_I, 0.0);

      for (size_t a_idx = 0; a_idx < off_list.size(); ++a_idx) {
        const Int     idx_KI    = off_list[a_idx];
        const Cell&   M_KI_cell = Z_cs.at(idx_KI);
        const double* M_KI      = M_off_ptr(a_idx);
        const Int     K         = M_KI_cell.J;
        const Int     rK_in_I   = M_KI_cell.rows;
        const Int     K_super0  = s.supernode_first_col[K - 1];
        const Idx*    KI_rows   = &s.row_pattern[M_KI_cell.lx_offset - 1];

        if (K == J) {
          // Z[J, J] is wJ × wJ symmetric (post-LAUUM/mirror, but at this
          // point in the loop Z[J, J] hasn't been finalized yet — Z[J, J]
          // gets finalized AFTER this column-I iteration completes
          // because J was processed in an EARLIER iteration of the
          // outer descending loop. ✓ already finalized.)
          //
          // Build Z_sub[a, b] = Z[J, J][p_J[a], p_J[b]]:  shape rJ × rJ.
          const Cell& Z_JJ = Z_cs.at(ci.diag_cell[J]);
          std::vector<double> Z_sub((size_t)rJ_in_I * rJ_in_I);
          for (Int b = 0; b < rJ_in_I; ++b) {
            for (Int a = 0; a < rJ_in_I; ++a) {
              Z_sub[a + (size_t)rJ_in_I * b] =
                  Z_JJ.nzval[p_J[a] + (size_t)Z_JJ.rows * p_J[b]];
            }
          }
          // acc += Z_sub · M[J, I]
          kernels::gemm('N', 'N',
                        rJ_in_I, w_I, rJ_in_I,
                        1.0,
                        Z_sub.data(), rJ_in_I,
                        M_KI,         rK_in_I,
                        1.0,
                        acc.data(), rJ_in_I);
        } else if (K > J) {
          // Z[K, J] is rK_in_J × wJ. We need Z[K, J] subm at rows aligned
          // to M[K, I]'s K-rows AND cols aligned to cell (J, I)'s J-rows.
          // Build Z_aligned[k, a] = Z[K, J][KI_to_KJ_pos[k], p_J[a]]:
          //   shape rK_in_I × rJ_in_I.
          // Then acc += Z_aligned^T · M[K, I] (rJ × wI).
          const Cell* Z_KJ = Z_cs.find(K, J);
          if (!Z_KJ) continue;
          const Idx* KJ_rows = &s.row_pattern[Z_KJ->lx_offset - 1];
          const Int  rK_in_J = Z_KJ->rows;

          std::vector<double> Z_aligned((size_t)rK_in_I * rJ_in_I, 0.0);
          // KI_rows ⊆ KJ_rows by Liu's theorem (factorize_run UPDATE relies
          // on this; same property here since both K appearances come from
          // the same etree-ancestor relationship).
          Int kj_cursor = 0;
          for (Int k_in_I = 0; k_in_I < rK_in_I; ++k_in_I) {
            Idx target = KI_rows[k_in_I];
            while (kj_cursor < rK_in_J && KJ_rows[kj_cursor] != target)
              ++kj_cursor;
            if (kj_cursor == rK_in_J) {
              throw std::logic_error(
                  "selinv: KI rows not subset of KJ rows (case K > J)");
            }
            for (Int a = 0; a < rJ_in_I; ++a) {
              Z_aligned[k_in_I + (size_t)rK_in_I * a] =
                  Z_KJ->nzval[kj_cursor + (size_t)Z_KJ->rows * p_J[a]];
            }
          }
          // acc += Z_aligned^T · M[K, I]
          kernels::gemm('T', 'N',
                        rJ_in_I, w_I, rK_in_I,
                        1.0,
                        Z_aligned.data(), rK_in_I,
                        M_KI,             rK_in_I,
                        1.0,
                        acc.data(), rJ_in_I);
        } else {  // I < K < J
          // Z[J, K] is rJ_in_K × wK. We need:
          //   contribution[a, c] = sum_k Z[J, K][JK_pos(JI_rows[a]), k_local]
          //                                 · M[K, I][k, c]
          //   where k_local = KI_rows[k] - K_super0 (K-local position).
          // Build Z_aligned[a, k] = Z[J, K][JK_pos(JI_rows[a]), KI_rows[k] - K_super0]:
          //   shape rJ_in_I × rK_in_I.
          // Rows of acc that aren't in Z[J, K]'s row set get a zero row
          // (Z[J, K][r, *] = 0 by symbolic absence, no contribution).
          const Cell* Z_JK = Z_cs.find(J, K);
          if (!Z_JK) continue;
          const Idx* JK_rows = &s.row_pattern[Z_JK->lx_offset - 1];
          const Int  rJ_in_K = Z_JK->rows;

          std::vector<double> Z_aligned((size_t)rJ_in_I * rK_in_I, 0.0);
          for (Int a = 0; a < rJ_in_I; ++a) {
            Int jk_pos = find_pos(JI_rows[a], JK_rows, rJ_in_K);
            if (jk_pos < 0) continue;   // contribution row is zero
            for (Int k_in_I = 0; k_in_I < rK_in_I; ++k_in_I) {
              Int k_local = KI_rows[k_in_I] - K_super0;
              Z_aligned[a + (size_t)rJ_in_I * k_in_I] =
                  Z_JK->nzval[jk_pos + (size_t)Z_JK->rows * k_local];
            }
          }
          // acc += Z_aligned · M[K, I]
          kernels::gemm('N', 'N',
                        rJ_in_I, w_I, rK_in_I,
                        1.0,
                        Z_aligned.data(), rJ_in_I,
                        M_KI,             rK_in_I,
                        1.0,
                        acc.data(), rJ_in_I);
        }
      }

      // Z[J, I] := -acc.
      for (size_t k = 0; k < acc.size(); ++k) Z_JI.nzval[k] = -acc[k];
    }

    // Diagonal Z[I, I] = M[I,I]^T · M[I,I] − sum_{K > I} M[K,I]^T · Z[K,I].
    std::copy(M_diag_buf.begin(), M_diag_buf.end(), Z_diag.nzval);
    {
      int info = kernels::lauum('L', w_I, Z_diag.nzval, Z_diag.rows);
      if (info != 0) throw std::runtime_error("selinv: dlauum failed");
      mirror_low_to_up(w_I, Z_diag.nzval, Z_diag.rows);
    }
    for (size_t a = 0; a < off_list.size(); ++a) {
      const Cell&   Z_KI  = Z_cs.at(off_list[a]);
      const double* M_KI  = M_off_ptr(a);
      kernels::gemm('T', 'N',
                    w_I, w_I, Z_KI.rows,
                    -1.0,
                    M_KI,        Z_KI.rows,
                    Z_KI.nzval,  Z_KI.rows,
                    1.0,
                    Z_diag.nzval, Z_diag.rows);
    }
  }
}

}}  // namespace sTiles::sparse
