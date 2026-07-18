/**
 * @file    selinv_task.hpp
 * @brief   Selected-inversion task descriptor for the sparse executor.
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

#ifndef _STILES_SPARSE_SELINV_TASK_HPP_
#define _STILES_SPARSE_SELINV_TASK_HPP_

#include <cstdint>

namespace sTiles { namespace sparse {

// Task kinds emitted by collect_selinv_tasks. Phase 1 inverts L into M = L^{-1}
// (stored in Z); phase 2 builds Z from M via Takahashi reverse sweep.
enum class SelinvOp : uint8_t {
    PHASE1_TRTRI = 1,   // Z[I,I] := L[I,I]^{-1}
    PHASE1_TRSM  = 2,   // Z[J,I] := L[J,I] * Z[I,I]
    PHASE2_OFF   = 3,   // Z[J,I] := -sum over K of contributions
    PHASE2_DIAG  = 4,   // Z[I,I] := M[I,I]^T * M[I,I] - sum_K M[K,I]^T * Z[K,I]
};

// 32-byte POD selinv task record. Mirrors SpsTask shape so the executor's
// per-rank slice is uniform.
struct alignas(32) SelinvTask {
    SelinvOp op;
    uint8_t  pad0[3];
    uint32_t I;           // pivot column supernode (1-based)
    uint32_t J;           // row supernode for PHASE1_TRSM/PHASE2_OFF; 0 otherwise
    uint32_t cell_diag;   // index of (I, I) — saves a find() in the hot path
    uint32_t cell_off;    // index of (J, I) for PHASE1_TRSM/PHASE2_OFF; 0 otherwise
    uint32_t pad1;
    uint32_t flags;       // reserved
};
static_assert(sizeof(SelinvTask) == 32, "SelinvTask must be cache-line friendly");

}}  // namespace sTiles::sparse

#endif  // _STILES_SPARSE_SELINV_TASK_HPP_
