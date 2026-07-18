/**
 * @file    collect_selinv.cpp
 * @brief   Selected-inversion task collection for the sparse module.
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

#include "collect_selinv.hpp"

#include <algorithm>
#include <stdexcept>

namespace sTiles { namespace sparse {

namespace {

struct ColumnIndex {
    std::vector<uint32_t>                           diag_cell;
    std::vector<std::vector<std::pair<Int, uint32_t>>> off_diag;  // (J, cell_idx)
};

ColumnIndex build_column_index(const Symbolic& s, const CellStore& cs) {
    ColumnIndex ci;
    ci.diag_cell.assign(s.n_super + 1, 0);
    ci.off_diag.assign(s.n_super + 1, {});
    Int next = 0;
    for (Int I = 1; I <= s.n_super; ++I) {
        if (next >= cs.cell_count() || cs.at(next).I != I || cs.at(next).J != I) {
            throw std::logic_error(
                "collect_selinv_tasks: missing diagonal cell at column I");
        }
        ci.diag_cell[I] = static_cast<uint32_t>(next);
        ++next;
        while (next < cs.cell_count() && cs.at(next).I == I) {
            ci.off_diag[I].emplace_back(cs.at(next).J,
                                         static_cast<uint32_t>(next));
            ++next;
        }
    }
    return ci;
}

// Estimate phase-1 flops for supernode I: TRTRI + sum of TRSMs.
double phase1_flops(const Symbolic& s, const ColumnIndex&, Int I) {
    Int width_I = s.supernode_first_col[I] - s.supernode_first_col[I - 1];
    Int len     = s.col_count[s.supernode_first_col[I - 1] - 1];
    Int off     = std::max<Int>(0, len - width_I);
    double f = (double)width_I * width_I * width_I / 6.0;       // TRTRI
    f       += (double)width_I * width_I * off;                  // TRSMs
    return f;
}

// Estimate phase-2 flops for supernode I. Heuristic: per-cell GEMM cost
// summed over all (J, K) pairs in I's pattern, plus diag LAUUM.
double phase2_flops(const Symbolic& s, const ColumnIndex& ci, Int I) {
    Int width_I = s.supernode_first_col[I] - s.supernode_first_col[I - 1];
    Int len     = s.col_count[s.supernode_first_col[I - 1] - 1];
    Int off     = std::max<Int>(0, len - width_I);
    // Phase 2 off-diag: for each J in off_list, ~|off_list| GEMMs of size
    // (rJ, wI, rK) with rJ, rK averaging off / |off_list|. Total ~ off^2 * wI.
    double f = (double)off * off * width_I;
    // Phase 2 diag: LAUUM(width) + |off_list| GEMMs of (wI, wI, rK).
    f += (double)width_I * width_I * width_I / 6.0;
    f += (double)width_I * width_I * off;
    (void)ci;
    return f;
}

// Supernodal-etree level (leaf = 0, root = max).
std::vector<Int> compute_levels(const Symbolic& s) {
    std::vector<Int> level(s.n_super + 1, 0);
    std::vector<Int> max_child(s.n_super + 1, -1);
    const auto& parent = s.sn_etree.parents();
    for (Int I = 1; I <= s.n_super; ++I) {
        level[I] = max_child[I] + 1;
        Int p = parent[I - 1];
        if (p != 0 && level[I] > max_child[p]) max_child[p] = level[I];
    }
    return level;
}

// Build consumers_of[K] = list of I < K with K in I's row pattern.
// Walks each supernode's row_pattern slice once; total work O(row_pattern_len).
void build_consumers(const Symbolic& s,
                     std::vector<std::vector<uint32_t>>& consumers_of,
                     std::vector<int>& contrib_remaining) {
    consumers_of.assign(s.n_super + 1, {});
    contrib_remaining.assign(s.n_super + 1, 0);
    for (Int I = 1; I <= s.n_super; ++I) {
        Ptr lb = s.row_pattern_ptr[I - 1];
        Ptr le = s.row_pattern_ptr[I] - 1;
        // Skip diag rows (the first entries of row_pattern[I] are I's own columns).
        // supernode_of_col lookup gives the supernode index for each row.
        for (Ptr p = lb; p <= le; ++p) {
            Int row = s.row_pattern[p - 1];
            Int K   = s.supernode_of_col[row - 1];
            if (K <= I) continue;  // diag rows or smaller-index entries
            // Avoid duplicates: a supernode K may span multiple rows in row_pattern[I];
            // we want to count K only once per I.
            if (consumers_of[K].empty() ||
                consumers_of[K].back() != static_cast<uint32_t>(I)) {
                consumers_of[K].push_back(static_cast<uint32_t>(I));
                contrib_remaining[I]++;
            }
        }
    }
}

// Emit phase 1 tasks for supernode I (TRTRI + per-off TRSMs). Phase 1 has no
// cross-column dependencies, so order within I is just: TRTRI, then TRSMs.
void emit_phase1(Int I, const ColumnIndex& ci,
                 std::vector<SelinvTask>& out_tasks) {
    uint32_t cell_II = ci.diag_cell[I];

    SelinvTask t{};
    t.op        = SelinvOp::PHASE1_TRTRI;
    t.I         = static_cast<uint32_t>(I);
    t.cell_diag = cell_II;
    out_tasks.push_back(t);

    for (const auto& off : ci.off_diag[I]) {
        SelinvTask u{};
        u.op        = SelinvOp::PHASE1_TRSM;
        u.I         = static_cast<uint32_t>(I);
        u.J         = static_cast<uint32_t>(off.first);
        u.cell_diag = cell_II;
        u.cell_off  = off.second;
        out_tasks.push_back(u);
    }
}

// Emit phase 2 tasks for supernode I. Order within I: all PHASE2_OFF first
// (descending J, mirroring the serial code's order), then PHASE2_DIAG last.
void emit_phase2(Int I, const ColumnIndex& ci,
                 std::vector<SelinvTask>& out_tasks,
                 std::vector<int>& n_off_in_col) {
    uint32_t cell_II = ci.diag_cell[I];
    const auto& off  = ci.off_diag[I];

    for (size_t b_idx = off.size(); b_idx-- > 0; ) {
        SelinvTask t{};
        t.op        = SelinvOp::PHASE2_OFF;
        t.I         = static_cast<uint32_t>(I);
        t.J         = static_cast<uint32_t>(off[b_idx].first);
        t.cell_diag = cell_II;
        t.cell_off  = off[b_idx].second;
        out_tasks.push_back(t);
    }
    n_off_in_col[I] = static_cast<int>(off.size());

    SelinvTask d{};
    d.op        = SelinvOp::PHASE2_DIAG;
    d.I         = static_cast<uint32_t>(I);
    d.cell_diag = cell_II;
    out_tasks.push_back(d);
}

}  // namespace

void collect_selinv_tasks(const Symbolic&        s,
                          const CellStore&       cs,
                          int                    n_threads,
                          CollectedSelinvTasks&  out) {
    out.tasks.clear();
    out.offsets.clear();
    out.n_off_in_col.assign(s.n_super + 1, 0);

    ColumnIndex ci = build_column_index(s, cs);
    build_consumers(s, out.consumers_of, out.contrib_remaining);

    const int N = std::max(1, n_threads);

    // Single-rank fast path: emit phase 1 forward, phase 2 reverse — matches
    // the serial selinv exactly. Used for validation in step 2.
    if (N == 1) {
        for (Int I = 1; I <= s.n_super; ++I) {
            emit_phase1(I, ci, out.tasks);
        }
        for (Int I = s.n_super; I >= 1; --I) {
            emit_phase2(I, ci, out.tasks, out.n_off_in_col);
        }
        out.offsets = {0, static_cast<int>(out.tasks.size())};
        return;
    }

    // Multi-rank: phase 1 LPT-spread (one big level), phase 2 level-set
    // root → leaves. Within each level, LPT by phase-2 flops.
    std::vector<Int> level = compute_levels(s);
    Int max_level = 0;
    for (Int I = 1; I <= s.n_super; ++I) max_level = std::max(max_level, level[I]);

    std::vector<std::vector<SelinvTask>> rank_tasks(N);
    std::vector<double>                  rank_load(N, 0.0);

    // Phase 1: all supernodes are independent.
    std::vector<Int> all_sns;
    all_sns.reserve(s.n_super);
    for (Int I = 1; I <= s.n_super; ++I) all_sns.push_back(I);
    std::sort(all_sns.begin(), all_sns.end(),
              [&](Int a, Int b) {
                return phase1_flops(s, ci, a) > phase1_flops(s, ci, b);
              });
    for (Int I : all_sns) {
        int min_r = 0;
        for (int r = 1; r < N; ++r)
            if (rank_load[r] < rank_load[min_r]) min_r = r;
        rank_load[min_r] += phase1_flops(s, ci, I);
        emit_phase1(I, ci, rank_tasks[min_r]);
    }

    // Reset load for phase 2's separate balancing.
    std::fill(rank_load.begin(), rank_load.end(), 0.0);

    // Phase 2: bucket by level, iterate root → leaves. Within each level,
    // distribute INDIVIDUAL tasks across ranks (not whole supernodes), so
    // matrices with linear-chain etrees (one supernode per level — common
    // for dense matrices) still get parallelism: PHASE2_OFF tasks of the
    // same column run on different ranks. The atomic counters
    // (n_off_remaining, contrib_remaining) enforce ordering regardless of
    // which rank executes which task.
    std::vector<std::vector<Int>> by_level(max_level + 1);
    for (Int I = 1; I <= s.n_super; ++I) by_level[level[I]].push_back(I);

    for (Int L = max_level; L >= 0; --L) {
        auto& sns = by_level[L];

        // First, set n_off_in_col[I] for every supernode in this level (used
        // by the executor's wait_zero on n_off_remaining[I]).
        for (Int I : sns) {
            out.n_off_in_col[I] = static_cast<int>(ci.off_diag[I].size());
        }

        struct LevelTask {
            SelinvTask t;
            double     flops;
        };
        std::vector<LevelTask> off_tasks;
        std::vector<LevelTask> diag_tasks;

        for (Int I : sns) {
            uint32_t cell_II = ci.diag_cell[I];
            const auto& off  = ci.off_diag[I];
            const Int   width_I = s.supernode_first_col[I] - s.supernode_first_col[I - 1];

            for (size_t b_idx = off.size(); b_idx-- > 0; ) {
                SelinvTask t{};
                t.op        = SelinvOp::PHASE2_OFF;
                t.I         = static_cast<uint32_t>(I);
                t.J         = static_cast<uint32_t>(off[b_idx].first);
                t.cell_diag = cell_II;
                t.cell_off  = off[b_idx].second;
                const Cell& c = cs.at(off[b_idx].second);
                const double f = static_cast<double>(off.size()) *
                                                  static_cast<double>(c.rows) *
                                                  static_cast<double>(width_I);
                off_tasks.push_back({t, f});
            }
            SelinvTask d{};
            d.op        = SelinvOp::PHASE2_DIAG;
            d.I         = static_cast<uint32_t>(I);
            d.cell_diag = cell_II;
            const double diag_f = (double)width_I * width_I * width_I / 6.0
                                                    + (double)off.size() * width_I * width_I;
            diag_tasks.push_back({d, diag_f});
        }

        // Critical: emit OFFs first (LPT), then DIAGs (LPT). Within any rank's
        // slice, every OFF of this level appears before any DIAG of this level.
        // This prevents the deadlock where a rank's DIAG(I) spins on
        // n_off_remaining[I] waiting for an OFF(I, *) queued behind it in the
        // same rank.
        auto distribute = [&](std::vector<LevelTask>& bucket) {
            std::sort(bucket.begin(), bucket.end(),
                      [](const LevelTask& a, const LevelTask& b) {
                        return a.flops > b.flops;
                      });
            for (auto& lt : bucket) {
                int min_r = 0;
                for (int r = 1; r < N; ++r)
                    if (rank_load[r] < rank_load[min_r]) min_r = r;
                rank_load[min_r] += lt.flops;
                rank_tasks[min_r].push_back(lt.t);
            }
        };
        distribute(off_tasks);
        distribute(diag_tasks);
    }

    // Concatenate per-rank buckets into the global task list.
    out.offsets.assign(N + 1, 0);
    for (int r = 0; r < N; ++r) {
        out.offsets[r] = static_cast<int>(out.tasks.size());
        out.tasks.insert(out.tasks.end(),
                         rank_tasks[r].begin(), rank_tasks[r].end());
    }
    out.offsets[N] = static_cast<int>(out.tasks.size());
}

}}  // namespace sTiles::sparse
