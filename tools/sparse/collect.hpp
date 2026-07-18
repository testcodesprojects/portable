/**
 * @file    collect.hpp
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

#ifndef _STILES_SPARSE_COLLECT_HPP_
#define _STILES_SPARSE_COLLECT_HPP_

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

#endif  // _STILES_SPARSE_COLLECT_HPP_
