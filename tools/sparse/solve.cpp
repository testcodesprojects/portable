#include "solve.hpp"

#include "kernels.hpp"
#include "../common/core_lapack.hpp"  // cblas_dtrsv/dgemv (+ Cblas* enums) for the nrhs==1 fast path
#include <cstdlib>                    // getenv/atoi for the STILES_SPARSE_* knobs

#include <algorithm>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace sTiles { namespace sparse {

ColIndex build_col_index(const Symbolic& s, const CellStore& cs) {
  ColIndex ci;
  ci.diag_cell.assign(s.n_super + 1, -1);
  ci.off_cells.assign(s.n_super + 1, {});
  Int next = 0;
  for (Int I = 1; I <= s.n_super; ++I) {
    if (next >= cs.cell_count() || cs.at(next).I != I || cs.at(next).J != I) {
      throw std::logic_error("solve: missing diagonal cell at column I");
    }
    ci.diag_cell[I] = next;
    ++next;
    while (next < cs.cell_count() && cs.at(next).I == I) {
      Int r = cs.at(next).rows;
      if (r > ci.max_off_rows) ci.max_off_rows = r;
      ci.off_cells[I].push_back(next);
      ++next;
    }
  }
  return ci;
}

// Build level-set schedule from the supernode etree. level_of_super[I] is
// the longest distance (in supernode-etree edges) from any leaf descendant
// of I up to I. Leaves have level 0; the root has the highest level.
//
// Since sn_etree is post-ordered, all descendants of I have indices < I, so
// a single forward pass over I = 1..n_super suffices: each supernode I has
// already had its `level_of_super[I]` finalized by the time we read it,
// and we then bump its parent's level if I's level + 1 exceeds it.
EtreeSchedule build_etree_schedule(const Symbolic& s) {
  EtreeSchedule es;
  if (s.n_super <= 0) {
    es.num_levels = 0;
    return es;
  }
  es.level_of_super.assign(s.n_super + 1, 0);
  for (Int I = 1; I <= s.n_super; ++I) {
    const Int parent = s.sn_etree.post_parent_of(I - 1);  // 0-based index → 1-based parent (0 = root)
    if (parent > 0) {
      const Int candidate = es.level_of_super[I] + 1;
      if (candidate > es.level_of_super[parent]) {
        es.level_of_super[parent] = candidate;
      }
    }
  }
  Int max_level = 0;
  for (Int I = 1; I <= s.n_super; ++I) {
    if (es.level_of_super[I] > max_level) max_level = es.level_of_super[I];
  }
  es.num_levels = max_level + 1;
  es.supers_in_level.assign(static_cast<std::size_t>(es.num_levels), {});
  for (Int I = 1; I <= s.n_super; ++I) {
    es.supers_in_level[es.level_of_super[I]].push_back(I);
  }
  return es;
}

namespace {

// y[invp[k] - 1] = bx[k - 1]   (apply P).
void permute_in(const Symbolic& s, const double* bx, int nrhs, int64_t ldb,
                std::vector<double>& y) {
  const Int n = s.n;
  // No zero-fill: invp is a full permutation, so the scatter below writes every
  // element of y — pre-zeroing is pure redundant memory traffic (an extra n*nrhs
  // writes per solve). Just ensure capacity (ensure_scratch already sizes it).
  const std::size_t need = static_cast<std::size_t>(n) * nrhs;
  if (y.size() < need) y.resize(need);
  for (int j = 0; j < nrhs; ++j) {
    const double* bcol = bx + static_cast<size_t>(j) * ldb;
    double*       ycol = y.data() + static_cast<size_t>(j) * n;
    for (Int k = 1; k <= n; ++k) {
      ycol[s.ordering.invp[k - 1] - 1] = bcol[k - 1];
    }
  }
}

// bx[k - 1] = y[invp[k] - 1]   (apply P^T).
void permute_out(const Symbolic& s, const std::vector<double>& y,
                 int nrhs, double* bx, int64_t ldb) {
  const Int n = s.n;
  for (int j = 0; j < nrhs; ++j) {
    double*       bcol = bx + static_cast<size_t>(j) * ldb;
    const double* ycol = y.data() + static_cast<size_t>(j) * n;
    for (Int k = 1; k <= n; ++k) {
      bcol[k - 1] = ycol[s.ordering.invp[k - 1] - 1];
    }
  }
}

// Generic scatter:  y[rows[ai]-1, j] -= scratch[ai, j]   for ai in 0..c_rows.
// y is column-major with leading dim n, scratch is column-major with leading
// dim c_rows.
//
// Two specializations:
//   nrhs == 1 — single contiguous walk, no outer loop.
//   nrhs == 2 — interleaved: rows[] walked ONCE, both RHS columns updated
//               at each scatter target. Halves the rows[] traffic and lets
//               the prefetcher pull the two ycol cache lines together.
//   else      — fall through to the general j-major loop.
inline void scatter_subtract(double* y, std::size_t n,
                             const double* scratch, Int c_rows,
                             const Idx* rows, int nrhs) {
    if (nrhs == 1) {
        for (Int ai = 0; ai < c_rows; ++ai) {
            y[rows[ai] - 1] -= scratch[ai];
        }
    } else if (nrhs == 2) {
        double*       y0 = y;
        double*       y1 = y + n;
        const double* t0 = scratch;
        const double* t1 = scratch + c_rows;
        for (Int ai = 0; ai < c_rows; ++ai) {
            const Int idx = rows[ai] - 1;
            y0[idx] -= t0[ai];
            y1[idx] -= t1[ai];
        }
    } else {
        for (int j = 0; j < nrhs; ++j) {
            double*       ycol = y       + static_cast<std::size_t>(j) * n;
            const double* tcol = scratch + static_cast<std::size_t>(j) * c_rows;
            for (Int ai = 0; ai < c_rows; ++ai) {
                ycol[rows[ai] - 1] -= tcol[ai];
            }
        }
    }
}

// Inverse of scatter_subtract for backward_subst's prelude:
//   scratch[ai, j] = y[rows[ai]-1, j]
inline void gather_into(const double* y, std::size_t n,
                        double* scratch, Int c_rows,
                        const Idx* rows, int nrhs) {
    if (nrhs == 1) {
        for (Int ai = 0; ai < c_rows; ++ai) {
            scratch[ai] = y[rows[ai] - 1];
        }
    } else if (nrhs == 2) {
        const double* y0 = y;
        const double* y1 = y + n;
        double*       t0 = scratch;
        double*       t1 = scratch + c_rows;
        for (Int ai = 0; ai < c_rows; ++ai) {
            const Int idx = rows[ai] - 1;
            t0[ai] = y0[idx];
            t1[ai] = y1[idx];
        }
    } else {
        for (int j = 0; j < nrhs; ++j) {
            const double* ycol = y       + static_cast<std::size_t>(j) * n;
            double*       tcol = scratch + static_cast<std::size_t>(j) * c_rows;
            for (Int ai = 0; ai < c_rows; ++ai) {
                tcol[ai] = ycol[rows[ai] - 1];
            }
        }
    }
}

// In-place forward substitution L y = y on a permuted vector.
void forward_subst(const Symbolic& s, const CellStore& cs,
                   const ColIndex& ci, std::vector<double>& y, int nrhs,
                   std::vector<double>& scratch) {
  const Int n = s.n;

  // nrhs == 1 fast path, per-supernode dispatch on width w:
  //   w <= FUSE_W → FUSED SCALAR matvec-scatter: no cblas call, no scratch
  //                 round-trip. For the width-1-ish supernodes of ultra-sparse
  //                 GMRF factors the cblas dispatch overhead (a call per node to
  //                 do ~1 flop) dominated; this is what makes csc_dtrsm fast.
  //   w >  FUSE_W → BLAS-2 (dtrsv/dgemv): chunky supernodes vectorize/block
  //                 better in the library. STILES_SPARSE_FUSE_W overrides (0=all
  //                 BLAS-2). Numerically identical either way.
  if (nrhs == 1) {
    static const int FUSE_W = [](){
      const char* e = std::getenv("STILES_SPARSE_FUSE_W");
      const int v = e ? std::atoi(e) : 8;
      return v < 0 ? 0 : v;
    }();
    for (Int I = 1; I <= s.n_super; ++I) {
      const Int fc = s.supernode_first_col[I - 1];
      const Int w  = s.supernode_first_col[I] - fc;
      const Cell& diag = cs.at(ci.diag_cell[I]);
      double* __restrict__ xv = y.data() + (fc - 1);
      if (w <= FUSE_W) {
        // scalar lower-triangular solve  L xv = xv  (col-major, non-unit)
        const double* __restrict__ Ld = diag.nzval; const Int ld = diag.rows;
        for (Int col = 0; col < w; ++col) {
          const double xc = (xv[col] /= Ld[(std::size_t)col + (std::size_t)col * ld]);
          for (Int r = col + 1; r < w; ++r)
            xv[r] -= Ld[(std::size_t)r + (std::size_t)col * ld] * xc;
        }
        // fused off-diagonal:  y[rows[r]] -= sum_k C[r,k] * xv[k]
        for (Int idx : ci.off_cells[I]) {
          const Cell& c = cs.at(idx);
          const Idx* rows = &s.row_pattern[c.lx_offset - 1];
          const double* __restrict__ C = c.nzval; const Int lc = c.rows;
          for (Int r = 0; r < c.rows; ++r) {
            double acc = 0.0;
            for (Int k = 0; k < w; ++k) acc += C[(std::size_t)r + (std::size_t)k * lc] * xv[k];
            y[rows[r] - 1] -= acc;
          }
        }
      } else {
        cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    w, diag.nzval, diag.rows, xv, 1);
        for (Int idx : ci.off_cells[I]) {
          const Cell& c = cs.at(idx);
          cblas_dgemv(CblasColMajor, CblasNoTrans, c.rows, w,
                      1.0, c.nzval, c.rows, xv, 1, 0.0, scratch.data(), 1);
          const Idx* rows = &s.row_pattern[c.lx_offset - 1];
          scatter_subtract(y.data(), static_cast<std::size_t>(n),
                           scratch.data(), c.rows, rows, 1);
        }
      }
    }
    return;
  }

  for (Int I = 1; I <= s.n_super; ++I) {
    Int fc = s.supernode_first_col[I - 1];
    Int w  = s.supernode_first_col[I] - fc;
    const Cell& diag = cs.at(ci.diag_cell[I]);

    kernels::trsm('L', 'L', 'N', 'N',
                  w, nrhs,
                  1.0,
                  diag.nzval, diag.rows,
                  y.data() + (fc - 1), n);

    for (Int idx : ci.off_cells[I]) {
      const Cell& c = cs.at(idx);
      kernels::gemm('N', 'N',
                    c.rows, nrhs, w,
                    1.0,
                    c.nzval, c.rows,
                    y.data() + (fc - 1), n,
                    0.0,
                    scratch.data(), c.rows);
      const Idx* rows = &s.row_pattern[c.lx_offset - 1];
      scatter_subtract(y.data(), static_cast<std::size_t>(n),
                       scratch.data(), c.rows, rows, nrhs);
    }
  }
}

// In-place backward substitution L^T y = y on a permuted vector.
void backward_subst(const Symbolic& s, const CellStore& cs,
                    const ColIndex& ci, std::vector<double>& y, int nrhs,
                    std::vector<double>& scratch) {
  const Int n = s.n;

  // nrhs == 1 fast path — fused-scalar / BLAS-2 mirror of forward_subst.
  if (nrhs == 1) {
    static const int FUSE_W = [](){
      const char* e = std::getenv("STILES_SPARSE_FUSE_W");
      const int v = e ? std::atoi(e) : 8;
      return v < 0 ? 0 : v;
    }();
    for (Int I = s.n_super; I >= 1; --I) {
      const Int fc = s.supernode_first_col[I - 1];
      const Int w  = s.supernode_first_col[I] - fc;
      const Cell& diag = cs.at(ci.diag_cell[I]);
      double* __restrict__ xv = y.data() + (fc - 1);
      if (w <= FUSE_W) {
        // fused off-diagonal:  xv[k] -= sum_r C[r,k] * y[rows[r]]
        for (Int idx : ci.off_cells[I]) {
          const Cell& c = cs.at(idx);
          const Idx* rows = &s.row_pattern[c.lx_offset - 1];
          const double* __restrict__ C = c.nzval; const Int lc = c.rows;
          for (Int k = 0; k < w; ++k) {
            double acc = 0.0;
            for (Int r = 0; r < c.rows; ++r) acc += C[(std::size_t)r + (std::size_t)k * lc] * y[rows[r] - 1];
            xv[k] -= acc;
          }
        }
        // scalar back-substitution  L^T xv = xv  (col-major, non-unit)
        const double* __restrict__ Ld = diag.nzval; const Int ld = diag.rows;
        for (Int col = w - 1; col >= 0; --col) {
          double acc = xv[col];
          for (Int r = col + 1; r < w; ++r) acc -= Ld[(std::size_t)r + (std::size_t)col * ld] * xv[r];
          xv[col] = acc / Ld[(std::size_t)col + (std::size_t)col * ld];
        }
      } else {
        for (Int idx : ci.off_cells[I]) {
          const Cell& c = cs.at(idx);
          const Idx* rows = &s.row_pattern[c.lx_offset - 1];
          gather_into(y.data(), static_cast<std::size_t>(n),
                      scratch.data(), c.rows, rows, 1);
          cblas_dgemv(CblasColMajor, CblasTrans, c.rows, w,
                      -1.0, c.nzval, c.rows, scratch.data(), 1, 1.0, xv, 1);
        }
        cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                    w, diag.nzval, diag.rows, xv, 1);
      }
    }
    return;
  }

  for (Int I = s.n_super; I >= 1; --I) {
    Int fc = s.supernode_first_col[I - 1];
    Int w  = s.supernode_first_col[I] - fc;
    const Cell& diag = cs.at(ci.diag_cell[I]);

    for (Int idx : ci.off_cells[I]) {
      const Cell& c = cs.at(idx);
      const Idx* rows = &s.row_pattern[c.lx_offset - 1];
      gather_into(y.data(), static_cast<std::size_t>(n),
                  scratch.data(), c.rows, rows, nrhs);
      kernels::gemm('T', 'N',
                    w, nrhs, c.rows,
                    -1.0,
                    c.nzval, c.rows,
                    scratch.data(), c.rows,
                    1.0,
                    y.data() + (fc - 1), n);
    }

    kernels::trsm('L', 'L', 'T', 'N',
                  w, nrhs,
                  1.0,
                  diag.nzval, diag.rows,
                  y.data() + (fc - 1), n);
  }
}

void check_args(const Symbolic& s, int nrhs, int64_t ldb) {
  if (nrhs <= 0) return;
  if (ldb < s.n) throw std::logic_error("solve: ldb < n");
}

// Common driver shared by the cached-state overloads. `ss` is grown lazily
// to fit (n*nrhs, max_off_rows*nrhs); the buffers persist across calls so
// subsequent solves at the same nrhs do zero heap work.
inline void ensure_scratch(SolveScratch& ss, Int n, Int max_off_rows, int nrhs) {
    const std::size_t y_sz = static_cast<std::size_t>(n) * nrhs;
    const std::size_t s_sz = static_cast<std::size_t>(max_off_rows) * nrhs;
    if (ss.y.size()       < y_sz) ss.y.resize(y_sz);
    if (ss.scatter.size() < s_sz) ss.scatter.resize(s_sz);
}

// Parallel-scratch variant: the off-diagonal GEMM output is partitioned per
// thread so threads don't stomp on each other's GEMM buffers. Slot for
// thread t starts at `scatter.data() + t * max_off_rows * nrhs`.
inline void ensure_scratch_par(SolveScratch& ss, Int n, Int max_off_rows,
                               int nrhs, int num_threads) {
    const std::size_t y_sz = static_cast<std::size_t>(n) * nrhs;
    const std::size_t s_sz = static_cast<std::size_t>(max_off_rows) * nrhs
                           * static_cast<std::size_t>(std::max(num_threads, 1));
    if (ss.y.size()       < y_sz) ss.y.resize(y_sz);
    if (ss.scatter.size() < s_sz) ss.scatter.resize(s_sz);
}

// Parallel forward substitution. Within each level, supernodes are
// independent in their diagonal-block solve, but the off-diagonal scatter
// writes to ancestor rows that can be shared across supernodes at the
// same level → atomic update required.
void forward_subst_par(const Symbolic& s, const CellStore& cs,
                       const ColIndex& ci, const EtreeSchedule& es,
                       std::vector<double>& y, int nrhs,
                       std::vector<double>& scatter, int num_threads) {
  const Int n = s.n;
  const std::size_t per_tid = static_cast<std::size_t>(ci.max_off_rows) * nrhs;

  for (Int level = 0; level < es.num_levels; ++level) {
    const auto& supers = es.supers_in_level[static_cast<std::size_t>(level)];
    const int   m      = static_cast<int>(supers.size());
    if (m == 0) continue;

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int k = 0; k < m; ++k) {
      const int    tid     = omp_get_thread_num();
      double*      my_scr  = scatter.data() + static_cast<std::size_t>(tid) * per_tid;
      const Int    I       = supers[static_cast<std::size_t>(k)];
      const Int    fc      = s.supernode_first_col[I - 1];
      const Int    w       = s.supernode_first_col[I] - fc;
      const Cell&  diag    = cs.at(ci.diag_cell[I]);

      kernels::trsm('L', 'L', 'N', 'N',
                    w, nrhs,
                    1.0,
                    diag.nzval, diag.rows,
                    y.data() + (fc - 1), n);

      for (Int idx : ci.off_cells[I]) {
        const Cell& c = cs.at(idx);
        kernels::gemm('N', 'N',
                      c.rows, nrhs, w,
                      1.0,
                      c.nzval, c.rows,
                      y.data() + (fc - 1), n,
                      0.0,
                      my_scr, c.rows);
        const Idx* rows = &s.row_pattern[c.lx_offset - 1];
        // Atomic scatter: ancestor rows can be written by sibling supernodes.
        for (int j = 0; j < nrhs; ++j) {
          double*       ycol = y.data() + static_cast<std::size_t>(j) * n;
          const double* tcol = my_scr   + static_cast<std::size_t>(j) * c.rows;
          for (Int ai = 0; ai < c.rows; ++ai) {
            const double val = tcol[ai];
            const Int    row = rows[ai] - 1;
            #pragma omp atomic update
            ycol[row] -= val;
          }
        }
      }
    }
  }
}

// Parallel backward substitution. Each supernode writes only to its own
// disjoint column range y[fc..fc+w-1], so supernodes within a level have
// no write contention — no atomics needed. Process levels root → leaves.
void backward_subst_par(const Symbolic& s, const CellStore& cs,
                        const ColIndex& ci, const EtreeSchedule& es,
                        std::vector<double>& y, int nrhs,
                        std::vector<double>& scatter, int num_threads) {
  const Int n = s.n;
  const std::size_t per_tid = static_cast<std::size_t>(ci.max_off_rows) * nrhs;

  for (Int level = es.num_levels - 1; level >= 0; --level) {
    const auto& supers = es.supers_in_level[static_cast<std::size_t>(level)];
    const int   m      = static_cast<int>(supers.size());
    if (m == 0) continue;

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int k = 0; k < m; ++k) {
      const int    tid     = omp_get_thread_num();
      double*      my_scr  = scatter.data() + static_cast<std::size_t>(tid) * per_tid;
      const Int    I       = supers[static_cast<std::size_t>(k)];
      const Int    fc      = s.supernode_first_col[I - 1];
      const Int    w       = s.supernode_first_col[I] - fc;
      const Cell&  diag    = cs.at(ci.diag_cell[I]);

      for (Int idx : ci.off_cells[I]) {
        const Cell& c = cs.at(idx);
        const Idx* rows = &s.row_pattern[c.lx_offset - 1];
        gather_into(y.data(), static_cast<std::size_t>(n),
                    my_scr, c.rows, rows, nrhs);
        kernels::gemm('T', 'N',
                      w, nrhs, c.rows,
                      -1.0,
                      c.nzval, c.rows,
                      my_scr, c.rows,
                      1.0,
                      y.data() + (fc - 1), n);
      }

      kernels::trsm('L', 'L', 'T', 'N',
                    w, nrhs,
                    1.0,
                    diag.nzval, diag.rows,
                    y.data() + (fc - 1), n);
    }
  }
}

}  // namespace

// ── Cached-state entry points (preferred) ────────────────────────────────────

void solve_forward(const Symbolic& s, const CellStore& cs,
                   const ColIndex& ci, SolveScratch& ss,
                   double* bx, int nrhs, int64_t ldb) {
  check_args(s, nrhs, ldb);
  if (nrhs <= 0) return;

  ensure_scratch(ss, s.n, ci.max_off_rows, nrhs);
  permute_in(s, bx, nrhs, ldb, ss.y);
  forward_subst(s, cs, ci, ss.y, nrhs, ss.scatter);
  permute_out(s, ss.y, nrhs, bx, ldb);
}

void solve_backward(const Symbolic& s, const CellStore& cs,
                    const ColIndex& ci, SolveScratch& ss,
                    double* bx, int nrhs, int64_t ldb) {
  check_args(s, nrhs, ldb);
  if (nrhs <= 0) return;

  ensure_scratch(ss, s.n, ci.max_off_rows, nrhs);
  permute_in(s, bx, nrhs, ldb, ss.y);
  backward_subst(s, cs, ci, ss.y, nrhs, ss.scatter);
  permute_out(s, ss.y, nrhs, bx, ldb);
}

void solve(const Symbolic& s, const CellStore& cs,
           const ColIndex& ci, SolveScratch& ss,
           double* bx, int nrhs, int64_t ldb) {
  check_args(s, nrhs, ldb);
  if (nrhs <= 0) return;

  ensure_scratch(ss, s.n, ci.max_off_rows, nrhs);
  permute_in(s, bx, nrhs, ldb, ss.y);
  forward_subst (s, cs, ci, ss.y, nrhs, ss.scatter);
  backward_subst(s, cs, ci, ss.y, nrhs, ss.scatter);
  permute_out(s, ss.y, nrhs, bx, ldb);
}

// ── Parallel overloads (level-set schedule, OMP) ─────────────────────────────
//
// Single-thread fast path delegates to the serial implementation to avoid
// OMP setup overhead. Multi-thread path uses the level schedule: per-level
// parallel-for with atomic scatter for forward, contention-free writes for
// backward.

void solve_forward(const Symbolic& s, const CellStore& cs,
                   const ColIndex& ci, const EtreeSchedule& es,
                   SolveScratch& ss,
                   double* bx, int nrhs, int64_t ldb, int num_threads) {
  check_args(s, nrhs, ldb);
  if (nrhs <= 0) return;

  if (num_threads <= 1) {
    solve_forward(s, cs, ci, ss, bx, nrhs, ldb);
    return;
  }
  ensure_scratch_par(ss, s.n, ci.max_off_rows, nrhs, num_threads);
  permute_in(s, bx, nrhs, ldb, ss.y);
  forward_subst_par(s, cs, ci, es, ss.y, nrhs, ss.scatter, num_threads);
  permute_out(s, ss.y, nrhs, bx, ldb);
}

void solve_backward(const Symbolic& s, const CellStore& cs,
                    const ColIndex& ci, const EtreeSchedule& es,
                    SolveScratch& ss,
                    double* bx, int nrhs, int64_t ldb, int num_threads) {
  check_args(s, nrhs, ldb);
  if (nrhs <= 0) return;

  if (num_threads <= 1) {
    solve_backward(s, cs, ci, ss, bx, nrhs, ldb);
    return;
  }
  ensure_scratch_par(ss, s.n, ci.max_off_rows, nrhs, num_threads);
  permute_in(s, bx, nrhs, ldb, ss.y);
  backward_subst_par(s, cs, ci, es, ss.y, nrhs, ss.scatter, num_threads);
  permute_out(s, ss.y, nrhs, bx, ldb);
}

void solve(const Symbolic& s, const CellStore& cs,
           const ColIndex& ci, const EtreeSchedule& es,
           SolveScratch& ss,
           double* bx, int nrhs, int64_t ldb, int num_threads) {
  check_args(s, nrhs, ldb);
  if (nrhs <= 0) return;

  if (num_threads <= 1) {
    solve(s, cs, ci, ss, bx, nrhs, ldb);
    return;
  }
  ensure_scratch_par(ss, s.n, ci.max_off_rows, nrhs, num_threads);
  permute_in(s, bx, nrhs, ldb, ss.y);
  forward_subst_par (s, cs, ci, es, ss.y, nrhs, ss.scatter, num_threads);
  backward_subst_par(s, cs, ci, es, ss.y, nrhs, ss.scatter, num_threads);
  permute_out(s, ss.y, nrhs, bx, ldb);
}

// ── Convenience overloads (per-call alloc, kept for legacy callers) ──────────

void solve_forward(const Symbolic& s, const CellStore& cs,
                   double* bx, int nrhs, int64_t ldb) {
  const ColIndex ci = build_col_index(s, cs);
  SolveScratch ss;
  solve_forward(s, cs, ci, ss, bx, nrhs, ldb);
}

void solve_backward(const Symbolic& s, const CellStore& cs,
                    double* bx, int nrhs, int64_t ldb) {
  const ColIndex ci = build_col_index(s, cs);
  SolveScratch ss;
  solve_backward(s, cs, ci, ss, bx, nrhs, ldb);
}

void solve(const Symbolic& s, const CellStore& cs,
           double* bx, int nrhs, int64_t ldb) {
  const ColIndex ci = build_col_index(s, cs);
  SolveScratch ss;
  solve(s, cs, ci, ss, bx, nrhs, ldb);
}

// ── Packed flat-CSC path (auto-switch for thin supernodes) ───────────────────
// Extract L into a flat lower-triangular CSC in FACTOR (permuted) coords. This
// mirrors sps_get_chol_column's cell walk (diag cell = w×w block, off cells =
// below-diagonal), but keeps factor coords (no permute-back) and lays columns
// out contiguously with the diagonal first — exactly what solve_packed's flat
// sweep needs.
void build_packed_csc(const Symbolic& s, const CellStore& cs,
                      const ColIndex& ci, PackedCsc& out) {
  const Int n = s.n;
  out.colptr.assign(static_cast<std::size_t>(n) + 1, 0);
  out.rowind.clear();
  out.values.clear();
  out.src.clear();
  std::vector<std::pair<Idx, const double*>> colbuf;         // (row, source nzval slot)
  for (Int I = 1; I <= s.n_super; ++I) {
    const Int fc = s.supernode_first_col[I - 1];            // 1-based
    const Int w  = s.supernode_first_col[I] - fc;
    const Cell& diag = cs.at(ci.diag_cell[I]);
    const Idx* dbase = &s.row_pattern[diag.lx_offset - 1];
    for (Int co = 0; co < w; ++co) {
      const Int c = (fc - 1) + co;                           // 0-based factor column
      colbuf.clear();
      // diagonal w×w block: lower triangle of this column (rows >= c).
      // Skip explicit-zero fill from relaxed supernode amalgamation (STRUCTURAL
      // zeros — always zero for this pattern, so safe to drop once); keep the
      // diagonal unconditionally. Record the nzval slot, not the value.
      for (Int rr = 0; rr < diag.rows; ++rr) {
        const Int rf = dbase[rr] - 1;
        if (rf < c) continue;                                // drop upper-tri of the block
        const double* sp = &diag.nzval[rr + diag.rows * co];
        if (*sp == 0.0 && rf != c) continue;                 // drop amalgamation zero
        colbuf.emplace_back(static_cast<Idx>(rf), sp);
      }
      // off-diagonal cells: rows all below the supernode.
      for (Int idx : ci.off_cells[I]) {
        const Cell& oc = cs.at(idx);
        const Idx* obase = &s.row_pattern[oc.lx_offset - 1];
        for (Int rr = 0; rr < oc.rows; ++rr) {
          const double* sp = &oc.nzval[rr + oc.rows * co];
          if (*sp == 0.0) continue;                          // drop amalgamation zero
          colbuf.emplace_back(static_cast<Idx>(obase[rr] - 1), sp);
        }
      }
      // Sort by row (diagonal is smallest → stays first, as the solve requires);
      // ascending rows give the sequential locality the kernels need.
      std::sort(colbuf.begin(), colbuf.end(),
                [](const std::pair<Idx,const double*>& a, const std::pair<Idx,const double*>& b){
                  return a.first < b.first; });
      for (const auto& pr : colbuf) {
        out.rowind.push_back(pr.first);
        out.src.push_back(pr.second);
        out.values.push_back(*pr.second);
      }
      out.colptr[c + 1] = static_cast<Ptr>(out.rowind.size());
    }
  }
  out.structure_built = true;
  out.values_built    = true;
}

// Value-only refresh (same pattern, new numeric factor): gather through src[].
void refresh_packed_csc_values(PackedCsc& out) {
  const std::size_t          nnz = out.src.size();
  const double* const* __restrict__ src = out.src.data();
  double*              __restrict__ v   = out.values.data();
  for (std::size_t p = 0; p < nnz; ++p) v[p] = *src[p];
  out.values_built = true;
}

void solve_packed(const Symbolic& s, const PackedCsc& csc,
                  SolveScratch& ss, double* bx, int64_t ldb, int solve_type) {
  const Int n = s.n;
  if (n <= 0) return;
  permute_in(s, bx, 1, ldb, ss.y);                           // y = P b
  double* __restrict__       y  = ss.y.data();
  const Ptr* __restrict__    cp = csc.colptr.data();
  const Idx* __restrict__    ri = csc.rowind.data();
  const double* __restrict__ Lv = csc.values.data();
  if (solve_type == 0 || solve_type == 2) {                  // forward  L y = y
    for (Int j = 0; j < n; ++j) {
      const double yj = (y[j] /= Lv[cp[j]]);
      for (Ptr p = cp[j] + 1; p < cp[j + 1]; ++p) y[ri[p]] -= Lv[p] * yj;
    }
  }
  if (solve_type == 1 || solve_type == 2) {                  // backward  L^T x = y
    for (Int j = n - 1; j >= 0; --j) {
      double acc = y[j];
      for (Ptr p = cp[j] + 1; p < cp[j + 1]; ++p) acc -= Lv[p] * y[ri[p]];
      y[j] = acc / Lv[cp[j]];
    }
  }
  permute_out(s, ss.y, 1, bx, ldb);                          // b = P^T y
}

void solve_packed_par(const Symbolic& s, const PackedCsc& csc,
                      const EtreeSchedule& es, SolveScratch& ss,
                      double* bx, int64_t ldb, int num_threads, int solve_type) {
  const Int n = s.n;
  if (n <= 0) return;
  if (num_threads <= 1 || es.num_levels <= 0) { solve_packed(s, csc, ss, bx, ldb, solve_type); return; }
  permute_in(s, bx, 1, ldb, ss.y);
  double* __restrict__       y  = ss.y.data();
  const Ptr* __restrict__    cp = csc.colptr.data();
  const Idx* __restrict__    ri = csc.rowind.data();
  const double* __restrict__ Lv = csc.values.data();

  // forward  L y = y : leaves → root. Supernodes within a level are independent
  // (no data dependence), so thread over them. The below-diagonal scatter can
  // target ancestor rows shared by sibling supernodes → atomic update.
  if (solve_type == 0 || solve_type == 2)
  for (Int lvl = 0; lvl < es.num_levels; ++lvl) {
    const std::vector<Int>& sup = es.supers_in_level[static_cast<std::size_t>(lvl)];
    const int m = static_cast<int>(sup.size());
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 16)
    for (int k = 0; k < m; ++k) {
      const Int I  = sup[static_cast<std::size_t>(k)];
      const Int fc = s.supernode_first_col[I - 1];           // 1-based
      const Int lc = s.supernode_first_col[I];
      for (Int c = fc - 1; c < lc - 1; ++c) {                // columns of this supernode, serial
        const double yj = (y[c] /= Lv[cp[c]]);
        for (Ptr p = cp[c] + 1; p < cp[c + 1]; ++p) {
          const Idx    r = ri[p];
          const double v = Lv[p] * yj;
          #pragma omp atomic
          y[r] -= v;
        }
      }
    }
  }

  // backward  L^T x = y : root → leaves. Each supernode writes only its own
  // columns; sibling supernodes in a level are disjoint → no atomics needed.
  if (solve_type == 1 || solve_type == 2)
  for (Int lvl = es.num_levels - 1; lvl >= 0; --lvl) {
    const std::vector<Int>& sup = es.supers_in_level[static_cast<std::size_t>(lvl)];
    const int m = static_cast<int>(sup.size());
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 16)
    for (int k = 0; k < m; ++k) {
      const Int I  = sup[static_cast<std::size_t>(k)];
      const Int fc = s.supernode_first_col[I - 1];
      const Int lc = s.supernode_first_col[I];
      for (Int c = lc - 2; c >= fc - 1; --c) {               // columns reverse, serial
        double acc = y[c];
        for (Ptr p = cp[c] + 1; p < cp[c + 1]; ++p) acc -= Lv[p] * y[ri[p]];
        y[c] = acc / Lv[cp[c]];
      }
    }
  }
  permute_out(s, ss.y, 1, bx, ldb);
}

void solve_packed_multi(const Symbolic& s, const PackedCsc& csc,
                        SolveScratch& ss, double* bx, int nrhs, int64_t ldb, int solve_type) {
  const Int n = s.n;
  if (n <= 0 || nrhs <= 0) return;
  if (nrhs == 1) { solve_packed(s, csc, ss, bx, ldb, solve_type); return; }

  const std::size_t need = static_cast<std::size_t>(n) * nrhs;
  if (ss.y.size() < need) ss.y.resize(need);
  double* __restrict__       Y    = ss.y.data();          // ROW-major: Y[i*nrhs + c]
  const Int*  __restrict__   invp = s.ordering.invp.data();
  const std::size_t          R    = static_cast<std::size_t>(nrhs);

  // permute in (row-major): k-outer so each (random) row block Y[row*nrhs .. ]
  // is written ONCE, contiguously — avoids revisiting each row nrhs times.
  for (Int k = 0; k < n; ++k) {
    double* __restrict__ Yrow = Y + static_cast<std::size_t>(invp[k] - 1) * R;
    for (int c = 0; c < nrhs; ++c) Yrow[c] = bx[static_cast<std::size_t>(k) + static_cast<std::size_t>(c) * ldb];
  }

  const Ptr* __restrict__    cp = csc.colptr.data();
  const Idx* __restrict__    ri = csc.rowind.data();
  const double* __restrict__ Lv = csc.values.data();

  if (nrhs <= 64) {
    // Fast K-wide SIMD kernel with a hot local column buffer (ported from the
    // semi/dense csc_dtrsm_multi_row): the current column's K values live in a
    // stack array so the scatter/gather inner loops vectorize and don't re-read
    // the big Y array. This is what closes the gap to the semi kernel.
    if (solve_type == 0 || solve_type == 2)
    for (Int j = 0; j < n; ++j) {                          // forward  L Y = B
      const double inv = 1.0 / Lv[cp[j]];
      double* __restrict__ Yj = Y + static_cast<std::size_t>(j) * R;
      double xj[64];
      #pragma omp simd
      for (int c = 0; c < nrhs; ++c) { const double v = Yj[c] * inv; Yj[c] = v; xj[c] = v; }
      for (Ptr p = cp[j] + 1; p < cp[j + 1]; ++p) {
        const double Lij = Lv[p];
        double* __restrict__ Yi = Y + static_cast<std::size_t>(ri[p]) * R;
        #pragma omp simd
        for (int c = 0; c < nrhs; ++c) Yi[c] -= Lij * xj[c];
      }
    }
    if (solve_type == 1 || solve_type == 2)
    for (Int j = n - 1; j >= 0; --j) {                     // backward  L^T X = Y
      double* __restrict__ Yj = Y + static_cast<std::size_t>(j) * R;
      double acc[64];
      #pragma omp simd
      for (int c = 0; c < nrhs; ++c) acc[c] = Yj[c];
      for (Ptr p = cp[j] + 1; p < cp[j + 1]; ++p) {
        const double Lij = Lv[p];
        const double* __restrict__ Yi = Y + static_cast<std::size_t>(ri[p]) * R;
        #pragma omp simd
        for (int c = 0; c < nrhs; ++c) acc[c] -= Lij * Yi[c];
      }
      const double inv = 1.0 / Lv[cp[j]];
      #pragma omp simd
      for (int c = 0; c < nrhs; ++c) Yj[c] = acc[c] * inv;
    }
  } else {
    // Fallback for nrhs > 64 (no fixed stack buffer): simple in-place.
    if (solve_type == 0 || solve_type == 2)
    for (Int j = 0; j < n; ++j) {
      const double inv = 1.0 / Lv[cp[j]];
      double* __restrict__ Yj = Y + static_cast<std::size_t>(j) * R;
      for (int c = 0; c < nrhs; ++c) Yj[c] *= inv;
      for (Ptr p = cp[j] + 1; p < cp[j + 1]; ++p) {
        const double v = Lv[p];
        double* __restrict__ Yi = Y + static_cast<std::size_t>(ri[p]) * R;
        for (int c = 0; c < nrhs; ++c) Yi[c] -= v * Yj[c];
      }
    }
    if (solve_type == 1 || solve_type == 2)
    for (Int j = n - 1; j >= 0; --j) {
      const double inv = 1.0 / Lv[cp[j]];
      double* __restrict__ Yj = Y + static_cast<std::size_t>(j) * R;
      for (Ptr p = cp[j] + 1; p < cp[j + 1]; ++p) {
        const double v = Lv[p];
        const double* __restrict__ Yi = Y + static_cast<std::size_t>(ri[p]) * R;
        for (int c = 0; c < nrhs; ++c) Yj[c] -= v * Yi[c];
      }
      for (int c = 0; c < nrhs; ++c) Yj[c] *= inv;
    }
  }

  // permute out (row-major): k-outer, each row block read once.
  for (Int k = 0; k < n; ++k) {
    const double* __restrict__ Yrow = Y + static_cast<std::size_t>(invp[k] - 1) * R;
    for (int c = 0; c < nrhs; ++c) bx[static_cast<std::size_t>(k) + static_cast<std::size_t>(c) * ldb] = Yrow[c];
  }
}

}}  // namespace sTiles::sparse
