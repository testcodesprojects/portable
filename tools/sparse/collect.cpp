/**
 * @file    collect.cpp
 * @brief   Numeric task collection and the work-aware core cap for the sparse executor.
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

#include "collect.hpp"
#include "kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace sTiles { namespace sparse {

namespace {

// Pre-Phase-8 linear flop-balanced partition removed; level-set + LPT
// inside collect_tasks supersedes it.

uint32_t require_cell_idx(const CellStore& cs, Int J, Int I,
                          const char* what) {
    const Cell* c = cs.find(J, I);
    if (c == nullptr) {
        throw std::logic_error(
            std::string("collect_tasks: missing cell while emitting ") + what);
    }
    return static_cast<uint32_t>(c - &cs.at(0));
}

}  // namespace

// Walk `cs.cells_` linearly once and for each column-supernode I record:
//   - diag_cell[I]    : index of the diagonal cell (I, I)
//   - off_diag[I]     : list of (J, cell_idx) for J > I in I's pattern
// Both built in O(total cells). Used by the level-set emitter so it can
// fetch a supernode's task ingredients in any order without paying the
// O(n_super²) cost of repeated `cs.find(J, I)`.
namespace {
struct ColumnIndex {
    std::vector<uint32_t>                                diag_cell;   // size n_super+1
    std::vector<std::vector<std::pair<Int, uint32_t>>>   off_diag;    // size n_super+1
};

ColumnIndex build_column_index(const Symbolic& s, const CellStore& cs) {
    ColumnIndex ci;
    ci.diag_cell.assign(s.n_super + 1, 0);
    ci.off_diag.assign(s.n_super + 1, {});
    Int next = 0;
    for (Int I = 1; I <= s.n_super; ++I) {
        if (next >= cs.cell_count() || cs.at(next).I != I || cs.at(next).J != I) {
            throw std::logic_error(
                "collect_tasks: missing diagonal cell — CellStore emission order "
                "does not match elimination-order assumption");
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

// Estimate the total flops of supernode I's task group (FACTOR + TRSMs +
// UPDATEs). Used for LPT load balancing; only ratios matter.
double supernode_flops(const Symbolic& s, const ColumnIndex&, Int I) {
    Int width_I = s.supernode_first_col[I] - s.supernode_first_col[I - 1];
    // col_count[supernode_first_col[I-1]-1] is the length of I's row pattern including the diag
    // block; subtracting width_I gives the total off-diagonal row count.
    Int len = s.col_count[s.supernode_first_col[I - 1] - 1];
    Int off = std::max<Int>(0, len - width_I);
    double f = (double)width_I * width_I * width_I / 3.0;   // FACTOR
    f += (double)width_I * width_I * off;                   // TRSMs
    f += (double)off * off * width_I;                       // UPDATEs (rough)
    return f;
}

// One-time single-core BLAS-3 flop-rate calibration (cached, thread-safe). Times a
// small dense DGEMM through the SAME backend the factorization uses, so the rate is
// in the same flop units as supernode_flops(). The work-floor's per-rank threshold
// is then derived as min_fpr = T_MIN_S * flop_rate, which auto-scales to the host
// CPU: on a faster core the same work-TIME corresponds to more flops, so a single
// time floor travels across Intel/AMD instead of a machine-specific flop constant.
// The 128-block stays single-threaded in the BLAS backend, matching the per-rank
// (single-core) work the cap is reasoning about.
inline double sparse_blas3_flop_rate() {
    static std::atomic<double> cached{0.0};
    double r = cached.load(std::memory_order_relaxed);
    if (r > 0.0) return r;
    const int n = 128;
    std::vector<double> A(static_cast<size_t>(n) * n, 1.0),
                                            B(static_cast<size_t>(n) * n, 1.0),
                                            C(static_cast<size_t>(n) * n, 0.0);
    kernels::gemm('N', 'N', n, n, n, 1.0, A.data(), n, B.data(), n, 0.0, C.data(), n);  // warm up
    const int reps = 64;
    const double t0 = omp_get_wtime();
    for (int i = 0; i < reps; ++i)
        kernels::gemm('N', 'N', n, n, n, 1.0, A.data(), n, B.data(), n, 1.0, C.data(), n);
    const double dt = omp_get_wtime() - t0;
    const double flops = 2.0 * static_cast<double>(n) * n * n * reps;
    r = (dt > 1e-9) ? flops / dt : 3.0e10;   // fallback ~30 GF/s if the timer is unusable
    cached.store(r, std::memory_order_relaxed);
    return r;
}

// Emit FACTOR + TRSMs + UPDATEs for one supernode I, in elimination order.
// Returns nothing; appends to out_tasks and bumps out_target_count.
void emit_supernode_tasks(Int                       I,
                          const ColumnIndex&        ci,
                          const CellStore&          cs,
                          std::vector<SpsTask>&     out_tasks,
                          std::vector<int>&         out_target_count) {
    uint32_t cell_II = ci.diag_cell[I];

    SpsTask t{};
    t.op     = TaskOp::FACTOR;
    t.I      = static_cast<uint32_t>(I);
    t.cell_a = cell_II;
    out_tasks.push_back(t);

    const auto& off = ci.off_diag[I];

    for (size_t a = 0; a < off.size(); ++a) {
        SpsTask u{};
        u.op     = TaskOp::TRSM;
        u.I      = static_cast<uint32_t>(I);
        u.J      = static_cast<uint32_t>(off[a].first);
        u.cell_a = cell_II;
        u.cell_b = off[a].second;
        out_tasks.push_back(u);
    }

    for (size_t a = 0; a < off.size(); ++a) {
        for (size_t b = a; b < off.size(); ++b) {
            Int J = off[a].first;
            Int K = off[b].first;
            SpsTask u{};
            u.op     = TaskOp::UPDATE;
            u.I      = static_cast<uint32_t>(I);
            u.J      = static_cast<uint32_t>(J);
            u.K      = static_cast<uint32_t>(K);
            u.cell_a = off[b].second;     // (K, I)
            u.cell_b = off[a].second;     // (J, I)
            u.cell_c = require_cell_idx(cs, K, J, "UPDATE dest cell (K,J)");
            out_tasks.push_back(u);
            out_target_count[u.cell_c]++;
        }
    }
}

// Compute supernodal-etree level (leaf = 0, root = max). Children always
// appear before their parent in 1..n_super by post-order, so a single
// linear pass suffices.
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
}  // namespace

void collect_tasks(const Symbolic&  s,
                   const CellStore& cs,
                   int              n_threads,
                   CollectedTasks&  out) {
    out.tasks.clear();
    out.offsets.clear();
    out.update_target_count.assign(cs.cell_count(), 0);

    int N = std::max(1, n_threads);
    ColumnIndex ci = build_column_index(s, cs);

    // Single-thread fast path: emit in elimination order, no level-set.
    // Same behaviour as the pre-Phase-8 collector — keeps single-thread perf
    // identical to the validated baseline.
    if (N == 1) {
        for (Int I = 1; I <= s.n_super; ++I) {
            emit_supernode_tasks(I, ci, cs, out.tasks, out.update_target_count);
        }
        out.offsets = {0, static_cast<int>(out.tasks.size())};
        return;
    }

    // Multi-thread: level-set distribution.
    // 1. Compute supernodal level (leaf = 0).
    std::vector<Int> level = compute_levels(s);

    // 2. Bucket supernodes by level.
    Int max_level = 0;
    for (Int I = 1; I <= s.n_super; ++I) max_level = std::max(max_level, level[I]);
    std::vector<std::vector<Int>> by_level(max_level + 1);
    for (Int I = 1; I <= s.n_super; ++I) by_level[level[I]].push_back(I);

    // 2b. Structural core cap. With a barrier between levels, a level cannot
    //     finish faster than its heaviest single supernode, so cores beyond
    //     ceil(level_total_flops / level_max_flops) sit idle in that level.
    //     The peak useful rank count over the whole factorization is therefore
    //     max_L ceil(level_total[L] / level_max[L]); allocating more than this
    //     leaves at least one rank idle in EVERY level (pure waste). We clamp N
    //     down to it. This is an upper bound on usable parallelism, so it can
    //     only trim dead cores, never slow the factor. Disable with
    //     STILES_SPARSE_CORE_CAP=0.
    {
        const char* e = std::getenv("STILES_SPARSE_CORE_CAP");
        const bool  cap_on = !(e && e[0] == '0');
        if (cap_on && N > 1) {
            double total_flops = 0.0, crit_flops = 0.0;
            int    cap = 1;
            // Dense-block dimension below which BLAS-3 gains little from extra cores.
            const int CAP_BLAS_DIM = 128;
            for (Int L = 0; L <= max_level; ++L) {
                double lt = 0.0, lm = 0.0;
                Int    lm_len = 1;                 // front size of the heaviest supernode
                for (Int I : by_level[L]) {
                    double f = supernode_flops(s, ci, I);
                    lt += f;
                    if (f > lm) { lm = f; lm_len = s.col_count[s.supernode_first_col[I - 1] - 1]; }
                }
                // The heaviest supernode of a level is a DENSE block; its POTRF/TRSM/GEMM
                // parallelize across ~min(N, front/CAP_BLAS_DIM) cores, so its critical
                // time is lm/p, not lm. The original model used lm (treating the supernode
                // as serial), which made a level dominated by ONE large dense supernode
                // look like a serial wall (speedup_ceiling=1.0x) and throttled large-dense
                // GMRFs such as lgm_50400 from ~18x scaling down to 2 cores. Crediting the
                // within-supernode parallelism fixes that; for small supernodes p==1 and
                // this is identical to the original cap (so other matrices are unchanged).
                const int    p      = std::min<int>(N, std::max(1, static_cast<int>(lm_len / CAP_BLAS_DIM)));
                const double lm_eff = lm / static_cast<double>(p);
                total_flops += lt;
                crit_flops  += lm_eff;
                int u = (lm_eff > 0.0) ? static_cast<int>(std::ceil(lt / lm_eff)) : 1;
                if (u > cap) cap = u;
            }
            // Absolute-work floor. The structural cap above measures how many tasks
            // CAN run concurrently per level, but it is blind to absolute work: on a
            // tiny factorization each task is so small that the per-level barrier +
            // task-dispatch + spin-wait overhead exceeds the math, giving NEGATIVE
            // scaling (bcsstk* class: 7ms@1core -> 55ms@10, structural_cap=10 yet a
            // 4.4x ceiling never materializes). Require each rank to receive at least
            // MIN_FLOPS_PER_RANK of work so the overhead is amortized; for large
            // factorizations work_cap >> N so nothing changes (lgm_50400 keeps ~18x).
            // Tunable via STILES_SPARSE_MIN_FLOPS_PER_RANK (set 0 to disable the floor).
            // Self-calibrating per-rank work floor (see sparse_blas3_flop_rate above):
            // require each rank to do at least T_MIN_S of real single-core BLAS-3 work,
            // so min_fpr = T_MIN_S * flop_rate adapts to the host. On this Intel node the
            // measured rate (~30 GF/s) yields ~3e6, the value an explicit sweep landed on
            // (2e7 over-capped parallel-but-modest-flop matrices like gyro_m by 7x; 1e6
            // let bcsstk* re-collapse); AMD derives its own equivalent automatically.
            // STILES_SPARSE_MIN_FLOPS_PER_RANK overrides outright; =0 disables the floor.
            const double T_MIN_S = 5.0e-5;   // 0.05 ms of single-core work per rank;
            // chosen so this node's measured ~60 GF/s yields ~3e6 (the swept sweet spot).
            double min_fpr = T_MIN_S * sparse_blas3_flop_rate();
            if (const char* mf = std::getenv("STILES_SPARSE_MIN_FLOPS_PER_RANK"))
                min_fpr = std::atof(mf);
            const int work_cap = (min_fpr > 0.0)
                                                  ? std::max(1, static_cast<int>(total_flops / min_fpr))
                                                  : N;
            int N_eff = std::min(std::min(N, cap), work_cap);
            if (N_eff < 1) N_eff = 1;
            if (N_eff < N) {
                if (std::getenv("STILES_SPARSE_CORE_CAP_LOG")) {
                    double ceil_sp = (crit_flops > 0.0) ? total_flops / crit_flops : 1.0;
                    std::fprintf(stderr,
                        "[sparse/core-cap] requested=%d structural_cap=%d work_cap=%d "
                        "(total_flops=%.3g min_fpr=%.3g flop_rate=%.1fGF/s) "
                        "speedup_ceiling=%.1fx -> using %d ranks\n",
                        N, cap, work_cap, total_flops, min_fpr,
                        sparse_blas3_flop_rate() / 1e9, ceil_sp, N_eff);
                }
                N = N_eff;
            }
        }
    }

    // 3. Per-rank task buckets, built in level order. Within each level we
    //    sort supernodes by descending flop cost and assign each to the
    //    least-loaded rank (LPT — Longest Processing Time first heuristic).
    std::vector<std::vector<SpsTask>> rank_tasks(N);
    std::vector<double>               rank_load(N, 0.0);

    for (Int L = 0; L <= max_level; ++L) {
        auto& sns = by_level[L];
        std::sort(sns.begin(), sns.end(),
                  [&](Int a, Int b) {
                    return supernode_flops(s, ci, a) > supernode_flops(s, ci, b);
                  });
        for (Int I : sns) {
            int min_r = 0;
            for (int r = 1; r < N; ++r)
                if (rank_load[r] < rank_load[min_r]) min_r = r;
            double f = supernode_flops(s, ci, I);
            rank_load[min_r] += f;
            // Append this supernode's tasks to rank min_r's bucket.
            emit_supernode_tasks(I, ci, cs,
                                 rank_tasks[min_r], out.update_target_count);
        }
    }

    // 4. Concatenate per-rank buckets into the global task array.
    out.offsets.assign(N + 1, 0);
    for (int r = 0; r < N; ++r) {
        out.offsets[r] = static_cast<int>(out.tasks.size());
        out.tasks.insert(out.tasks.end(),
                         rank_tasks[r].begin(), rank_tasks[r].end());
    }
    out.offsets[N] = static_cast<int>(out.tasks.size());
}

}}  // namespace sTiles::sparse
