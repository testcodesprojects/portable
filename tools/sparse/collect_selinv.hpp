#ifndef SPS_SCHED_COLLECT_SELINV_HPP
#define SPS_SCHED_COLLECT_SELINV_HPP

#include "selinv_task.hpp"
#include "supernode.hpp"
#include "symbolic.hpp"

#include <cstdint>
#include <vector>

namespace sTiles { namespace sparse {

// Output of collect_selinv_tasks. Per-rank slice is
//   tasks[offsets[r] .. offsets[r+1]-1].
//
// `contrib_remaining[I]` (size n_super+1, 1-based) is the count of supernodes
// K > I appearing in I's row pattern that contribute to column I in phase 2.
// Initialized at collect time, decremented by PHASE2_DIAG(K) when column K
// finishes. Column I's phase-2 tasks spin until `contrib_remaining[I] == 0`.
//
// `consumers_of[K]` (size n_super+1) is the inverse adjacency: list of
// supernodes I < K that have K in their row pattern. PHASE2_DIAG(K) walks
// this list and atomic-decrements each I's `contrib_remaining`.
//
// `n_off_in_col[I]` (size n_super+1) is the count of PHASE2_OFF tasks emitted
// for column I (= |off_list(I)|). PHASE2_DIAG(I) waits for this many off
// tasks of its own column to finish before reading their Z output.
struct CollectedSelinvTasks {
  std::vector<SelinvTask>            tasks;
  std::vector<int>                   offsets;
  std::vector<int>                   contrib_remaining;
  std::vector<std::vector<uint32_t>> consumers_of;
  std::vector<int>                   n_off_in_col;
};

// Walk the supernodal structure and emit phase 1 and phase 2 selinv tasks.
//   - Phase 1 (PHASE1_TRTRI + PHASE1_TRSM) is embarrassingly parallel across
//     supernodes; we LPT-distribute one big level.
//   - Phase 2 (PHASE2_OFF + PHASE2_DIAG) runs root → leaves; we level-set
//     by supernodal etree level and LPT-distribute within each level.
//
// `n_threads` controls the per-rank slicing. n_threads <= 1 emits a single
// rank in elimination order (phase 1 forward, phase 2 reverse) — same shape
// as the serial selinv, useful for validation.
void collect_selinv_tasks(const Symbolic&        s,
                          const CellStore&       cs,
                          int                    n_threads,
                          CollectedSelinvTasks&  out);

}}  // namespace sTiles::sparse

#endif  // SPS_SCHED_COLLECT_SELINV_HPP
