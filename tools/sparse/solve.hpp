/**
 * @file    solve.hpp
 * @brief   Triangular solves for the sparse module, including the packed-CSC fast path.
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

#ifndef _STILES_SPARSE_SOLVE_HPP_
#define _STILES_SPARSE_SOLVE_HPP_

#include "symbolic.hpp"
#include "supernode.hpp"

#include <vector>

namespace sTiles { namespace sparse {

// Per-column-supernode metadata consumed by the solve loops. Built once
// via build_col_index() from a Symbolic + CellStore; valid as long as the
// symbolic structure (s.n_super, cells_, J/I assignments) doesn't change.
// Numerical re-factorizations on the same pattern keep ColIndex valid.
struct ColIndex {
        std::vector<Int>              diag_cell;     // 1-based: diag_cell[I] = index into cs of (I,I)
        std::vector<std::vector<Int>> off_cells;     // 1-based: off_cells[I] = indices of (I,J>I)
        Int                           max_off_rows = 0;
};

// Build the per-supernode cell index. O(cs.cell_count()); typically called
// once at factor time and cached on the sparse handle.
ColIndex build_col_index(const Symbolic& s, const CellStore& cs);

// Persistent per-solve scratch buffers. Reused across repeated solves on
// the same factor to amortize heap allocations. When num_threads > 1, the
// `scatter` buffer is partitioned: thread t owns the slice starting at
// max_off_rows * nrhs * t.
struct SolveScratch {
        std::vector<double> y;        // permuted RHS / running solution (n × nrhs)
        std::vector<double> scatter;  // (max_off_rows × nrhs × num_threads)
};

// Elimination-tree level schedule. All supernodes in the same level have no
// ancestor/descendant relationship, so their diagonal block solves are
// independent. Built once per symbolic factor.
//
// Forward substitution: process levels 0, 1, ..., num_levels-1 in order
// (leaves → root). Within a level, supernodes run in parallel. The
// off-diagonal scatter targets ancestor rows (higher levels) and may have
// inter-supernode contention within the level — requires atomic updates.
//
// Backward substitution: process levels num_levels-1, ..., 1, 0
// (root → leaves). Each supernode writes only to its own column range, so
// supernodes within a level are embarrassingly parallel (no atomics).
struct EtreeSchedule {
        std::vector<Int>              level_of_super;    // size n_super+1, 1-indexed
        std::vector<std::vector<Int>> supers_in_level;   // [lvl] → list of supernode IDs
        Int                           num_levels = 0;
};

// Build the schedule from the supernode etree. O(n_super).
EtreeSchedule build_etree_schedule(const Symbolic& s);

// Three solve entry points. In all three, `bx` is in NATURAL (original)
// ordering both on entry and on return — the routines apply the permutation
// at the edges. `bx` is column-major of shape (s.n, nrhs) with leading
// dimension `ldb >= s.n`.
//
//   solve_forward (s, cs, bx, ...) :  bx ← P^T · L^{-1} · P · bx
//   solve_backward(s, cs, bx, ...) :  bx ← P^T · L^{-T} · P · bx
//   solve         (s, cs, bx, ...) :  bx ← P^T · L^{-T} · L^{-1} · P · bx
//
// The "forward" / "backward" terminology matches sTiles' `pdtrsm_forward` /
// `pdtrsm_backward` — split apart so callers that only need one half (e.g.
// computing z = L^{-1} P b for a log-likelihood, or sampling x ~ N(0, A^{-1})
// via L^T x = y on a random y) can skip the other.
//
// Default overloads allocate ColIndex + scratch per call. The state-taking
// overloads below reuse them — strongly preferred for the chol → solve loop.
void solve_forward (const Symbolic& s, const CellStore& cs,
                    double* bx, int nrhs, int64_t ldb);

void solve_backward(const Symbolic& s, const CellStore& cs,
                    double* bx, int nrhs, int64_t ldb);

void solve         (const Symbolic& s, const CellStore& cs,
                    double* bx, int nrhs, int64_t ldb);

// Fast overloads. Caller owns the ColIndex (built once per factor pattern)
// and the SolveScratch (sized lazily on first call, retained across calls).
void solve_forward (const Symbolic& s, const CellStore& cs,
                    const ColIndex& ci, SolveScratch& ss,
                    double* bx, int nrhs, int64_t ldb);

void solve_backward(const Symbolic& s, const CellStore& cs,
                    const ColIndex& ci, SolveScratch& ss,
                    double* bx, int nrhs, int64_t ldb);

void solve         (const Symbolic& s, const CellStore& cs,
                    const ColIndex& ci, SolveScratch& ss,
                    double* bx, int nrhs, int64_t ldb);

// Parallel overloads. When num_threads <= 1, fall back to the serial path
// above (avoids OMP-region setup overhead). Otherwise dispatches the
// per-level supernodes across `num_threads` OMP threads, using atomic
// updates on the forward-substitution scatter to handle inter-supernode
// contention. SolveScratch grows lazily to fit num_threads × max_off_rows
// × nrhs and persists for reuse on the next call.
void solve_forward (const Symbolic& s, const CellStore& cs,
                    const ColIndex& ci, const EtreeSchedule& es,
                    SolveScratch& ss,
                    double* bx, int nrhs, int64_t ldb, int num_threads);

void solve_backward(const Symbolic& s, const CellStore& cs,
                    const ColIndex& ci, const EtreeSchedule& es,
                    SolveScratch& ss,
                    double* bx, int nrhs, int64_t ldb, int num_threads);

void solve         (const Symbolic& s, const CellStore& cs,
                    const ColIndex& ci, const EtreeSchedule& es,
                    SolveScratch& ss,
                    double* bx, int nrhs, int64_t ldb, int num_threads);

// ── Packed flat-CSC nrhs==1 solve (auto-switch for thin supernodes) ──────────
// The supernodal solve walks a cell structure per supernode; on thin-supernode /
// ultra-sparse factors that per-node overhead dominates and a flat scalar sweep
// (csc_dtrsm-style) is much faster. PackedCsc is the lower-triangular factor L in
// FACTOR (permuted) coords — 0-based colptr/rowind, diagonal first per column —
// built once from the supernodal CellStore and reused across solves on the same
// factor. Rebuilt when the factor changes.
struct PackedCsc {
        std::vector<Ptr>            colptr;   // size n+1                    — STRUCTURE (per pattern)
        std::vector<Idx>            rowind;   // 0-based rows, diag first    — STRUCTURE (per pattern)
        std::vector<const double*>  src;      // src[p] → supernodal nzval slot for entry p — STRUCTURE
        std::vector<double>         values;   // refreshed each factorization via src (mirrors semi L_src)
        bool structure_built = false;         // colptr/rowind/src — built ONCE per symbolic pattern
        bool values_built    = false;         // values — refreshed on each new factorization
};

// Full build from the supernodal factor: colptr/rowind + the src pointer map +
// initial values. O(nnz(L)) incl. per-column sort. Sets both flags. Structure is
// pattern-only (amalgamation zeros are structural), so it need not be rebuilt
// across re-factorizations of the same pattern.
void build_packed_csc(const Symbolic& s, const CellStore& cs,
                      const ColIndex& ci, PackedCsc& out);

// Refresh ONLY the values of an already-built PackedCsc from the (same-pattern)
// supernodal factor — O(nnz(L)) gather through src[], no walk/sort. Call after a
// re-factorization; mirrors the semi path's `L_values[ptr] = *L_src[ptr]`.
void refresh_packed_csc_values(PackedCsc& out);

// nrhs==1 flat-CSC solve: bx ← P^T L^{-T} L^{-1} P bx. Serial; same
// permute-at-edges contract as solve(). Reuses ss.y as the work vector.
// solve_type: 0 = forward (L y = P b), 1 = backward (L^T x = P b), 2 = full LL^T.
void solve_packed(const Symbolic& s, const PackedCsc& csc,
                  SolveScratch& ss, double* bx, int64_t ldb, int solve_type = 2);

// Parallel (level-scheduled) variant of solve_packed: threads over the
// independent supernodes within each etree level (forward scatter uses atomics;
// backward is contention-free). Numerically identical to solve_packed. Falls
// back to serial when num_threads <= 1. NOTE: for ultra-sparse factors the
// serial sweep usually wins — a sparse triangular solve pays level-barrier +
// atomic overhead that exceeds the parallel gain — but this exposes the option.
void solve_packed_par(const Symbolic& s, const PackedCsc& csc,
                      const EtreeSchedule& es, SolveScratch& ss,
                      double* bx, int64_t ldb, int num_threads, int solve_type = 2);

// Multi-RHS (nrhs>=2) flat-CSC solve — the K-wide analogue of solve_packed and
// the sparse-path mirror of csc_dtrsm_multi_row. Works internally in ROW-major
// (Y[i*nrhs + c]) so the per-entry update over the nrhs columns is contiguous
// and vectorizes; reads each L value once and updates all columns in one pass.
// Serial; same permute-at-edges contract as solve(). Falls back to solve_packed
// for nrhs==1.
void solve_packed_multi(const Symbolic& s, const PackedCsc& csc,
                        SolveScratch& ss, double* bx, int nrhs, int64_t ldb, int solve_type = 2);

}}  // namespace sTiles::sparse

#endif  // _STILES_SPARSE_SOLVE_HPP_
