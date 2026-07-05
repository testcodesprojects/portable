#ifndef SPS_SCHED_COLLECT_HPP
#define SPS_SCHED_COLLECT_HPP

#include "symbolic.hpp"
#include "supernode.hpp"
#include "task.hpp"

#include <cstdint>
#include <vector>

namespace sTiles { namespace sparse {

// Output of `collect_tasks`. `offsets` has size n_threads + 1 and gives
// each rank's task slice as `tasks[offsets[r] .. offsets[r+1]-1]`.
//
// `update_target_count[cell_idx]` is the number of UPDATE tasks whose
// destination is `cell_idx`. The executor (Phase 7) initializes one
// `std::atomic<int>` per cell from these counts and decrements on each
// UPDATE; when it reaches zero, the producer sets the (K, J) progress bit.
struct CollectedTasks {
  std::vector<SpsTask> tasks;
  std::vector<int>     offsets;
  std::vector<int>     update_target_count;
};

// Walk the supernodal etree in elimination order, emitting FACTOR /
// TRSM / UPDATE tasks per plan §3 / §6. Inputs:
//   - `s`        : symbolic factorization output
//   - `cs`       : cell store (already allocated against `s`)
//   - `n_threads`: how many ranks the equal-flop partitioner targets
//
// On output, `out.tasks` is in elimination order, `out.offsets` has
// n_threads + 1 entries, and `out.update_target_count` has `cs.cell_count()`
// entries.
//
// Throws std::logic_error if a cell referenced by the symbolic structure
// can't be found in `cs` (would indicate symbolic/cell-store mismatch).
void collect_tasks(const Symbolic&  s,
                   const CellStore& cs,
                   int              n_threads,
                   CollectedTasks&  out);

}}  // namespace sTiles::sparse

#endif  // SPS_SCHED_COLLECT_HPP
