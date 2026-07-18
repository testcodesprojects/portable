/**
 * @file    executor.hpp
 * @brief   pthreads/OMP task executors driving the sparse module’s numeric phases.
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

#ifndef _STILES_SPARSE_EXECUTOR_HPP_
#define _STILES_SPARSE_EXECUTOR_HPP_

#include "symbolic.hpp"
#include "supernode.hpp"
#include "collect.hpp"
#include "collect_selinv.hpp"
#include "progress.hpp"

#include <atomic>
#include <memory>

namespace sTiles { namespace sparse {

// Persistent state carried across `factorize_run` calls. After wiring the
// `symbolic`, `cells`, `tasks` pointers (Phase 4 outputs), call
// `prepare(state)` once to size the progress matrix and the
// `updates_remaining` atomic counters. Then call `factorize_run(rank, &state)`
// from each rank (rank 0 only for Phase 5).
//
// `abort_flag` is set when a Potrf reports the matrix isn't PD; the executor
// short-circuits all downstream tasks. After a run, callers inspect this
// flag to decide return status.
struct SpsState {
    const Symbolic*       symbolic = nullptr;
    CellStore*            cells    = nullptr;
    const CollectedTasks* tasks    = nullptr;

    ProgressMatrix                       progress;
    std::atomic<bool>                    abort_flag{false};
    std::unique_ptr<std::atomic<int>[]>  updates_remaining;
    // Per-cell spinlock used by UPDATE's scatter to serialize concurrent
    // contributions to the same destination cell. 0 = unlocked, 1 = locked.
    std::unique_ptr<std::atomic<int>[]>  cell_locks;
    Int                                  n_cells = 0;

    // Per-rank UPDATE scratch (T = A * B^T, A.rows × B.rows). One buffer per
    // rank so threads don't race. `prepare(state)` reads `tasks->offsets` to
    // size each rank's buffer to the largest UPDATE in that rank's slice.
    std::vector<std::unique_ptr<double[]>> update_scratch;
};

// Launches `factorize_run(rank, &state)` from each rank using
// std::thread (one fresh thread per call, no pool). Mirrors sTiles'
// pthreads_dpotrf path.
void factorize_run_parallel_pthreads(SpsState& state);

// Same, but uses OpenMP's runtime thread pool. Mirrors sTiles'
// omp_dpotrf path.
void factorize_run_parallel_omp(SpsState& state);

// Allocate progress matrix and updates_remaining atomics. Resets all
// progress entries to 0, abort_flag to false, and per-cell counters to
// `tasks->update_target_count[i]`.
void prepare(SpsState& state);

// Reset just the per-call state (progress + abort + counters). The arena and
// task array stay untouched. Useful if the caller wants to re-factorize the
// same matrix multiple times (benchmarks, perturbation studies).
void reset_for_factorize(SpsState& state);

// Execute one rank's slice of the task array. The slice is
// `tasks[offsets[rank] .. offsets[rank+1] - 1]`. Phase 5 calls this with
// rank = 0 (single-threaded); Phase 7 will dispatch from N OpenMP /
// pthreads threads.
void factorize_run(int rank, SpsState* state);

// ─────────────────────────── Selected inverse ─────────────────────────────
// Per-call selinv state. Holds the M-snapshot arena (a copy of Z's arena
// taken after phase 1 finishes; phase 2 reads M from here, writes Z into
// the regular CellStore). Built by `selinv_prepare`, freed in the
// destructor.
struct SelinvState {
    SelinvState() = default;
    SelinvState(const SelinvState&)            = delete;
    SelinvState& operator=(const SelinvState&) = delete;
    SelinvState(SelinvState&& other) noexcept;
    SelinvState& operator=(SelinvState&& other) noexcept;
    ~SelinvState();

    CellStore*                       L_cs   = nullptr;
    CellStore*                       Z_cs   = nullptr;
    const Symbolic*                  sym    = nullptr;
    const CollectedSelinvTasks*      tasks  = nullptr;

    // M-snapshot: parallel copy of Z's arena after phase 1. Allocated via
    // sTiles::Memory::MemoryManager (tagged with `group_id`) so its bytes
    // appear in the per-group memory totals and get reaped by
    // sTiles::Memory::MemoryManager::freeAllGroup. Same size + layout as
    // Z's arena — index by `Z_cs->cells_[i].nzval - Z_cs->arena_data()` to
    // find the corresponding M slot.
    double*                          M_arena      = nullptr;  // owned via MemoryManager
    Ptr                              M_arena_size = 0;
    int                              group_id     = -1;

    // Per-supernode atomic counters for cross-column dependency tracking.
    // contrib_remaining[I]: K > I in I's pattern, decremented by PHASE2_DIAG(K).
    // n_off_remaining[I]: PHASE2_OFF tasks of column I, decremented when each
    //                     completes; PHASE2_DIAG(I) waits until 0.
    // phase1_done[I]: 1 once PHASE1_TRTRI(I) finishes; PHASE1_TRSM(I,*) waits.
    std::unique_ptr<std::atomic<int>[]>  contrib_remaining;
    std::unique_ptr<std::atomic<int>[]>  n_off_remaining;
    std::unique_ptr<std::atomic<uint8_t>[]> phase1_done;

    // Per-column cell layout, built once in selinv_prepare. Indexed by
    // supernode I (1-based). col_diag[I] = index of diag cell (I, I).
    // col_off_first[I] / col_off_last[I] bound the contiguous range of off-
    // diagonal cells for column I in the CellStore (cells are emitted
    // grouped by I). This replaces an O(n_cells) scan per task.
    std::vector<uint32_t> col_diag;
    std::vector<uint32_t> col_off_first;
    std::vector<uint32_t> col_off_last;   // one-past-the-end

    std::atomic<bool>  abort_flag{false};
};

// Allocate the M-snapshot arena and the per-supernode atomic counters.
// Initial state: phase1_done[*] = 0, n_off_remaining[I] = tasks->n_off_in_col[I],
// contrib_remaining[I] = tasks->contrib_remaining[I].
void selinv_prepare(SelinvState& state);

// Execute one rank's slice of the selinv task list.
void selinv_run(int rank, SelinvState* state);

// Run all ranks in parallel, then return.
void selinv_run_parallel_pthreads(SelinvState& state);
void selinv_run_parallel_omp     (SelinvState& state);

}}  // namespace sTiles::sparse

#endif  // _STILES_SPARSE_EXECUTOR_HPP_
