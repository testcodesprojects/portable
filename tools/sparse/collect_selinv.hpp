/**
 * @file    collect_selinv.hpp
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

#ifndef _STILES_SPARSE_COLLECT_SELINV_HPP_
#define _STILES_SPARSE_COLLECT_SELINV_HPP_

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

#endif  // _STILES_SPARSE_COLLECT_SELINV_HPP_
