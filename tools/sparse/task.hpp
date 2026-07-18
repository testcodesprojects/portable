/**
 * @file    task.hpp
 * @brief   Numeric task descriptor for the sparse executor.
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

#ifndef _STILES_SPARSE_TASK_HPP_
#define _STILES_SPARSE_TASK_HPP_

#include <cstdint>

namespace sTiles { namespace sparse {

enum class TaskOp : uint8_t {
    FACTOR = 1,   // POTRF on diagonal cell (I, I)
    TRSM   = 2,   // TRSM solving cell (J, I) using diagonal cell (I, I)
    UPDATE = 3,   // C(K, J) -= A(K, I) * B(J, I)^T
};

// 32-byte POD task record. The fields name the supernode triple plus the
// three cell indices the executor needs to look up dense blocks. The
// `flags` slot is reserved for future use (e.g. LAST_OF_CELL bit, debug
// info); keep it at 0 for v1 — atomic update_target_count handles
// last-update detection.
struct alignas(32) SpsTask {
    TaskOp   op;
    uint8_t  pad0[3];
    uint32_t I;       // pivot column supernode (1-based)
    uint32_t J;       // row supernode for TRSM/UPDATE; 0 for FACTOR
    uint32_t K;       // facing row supernode for UPDATE; 0 otherwise
    uint32_t cell_a;  // index into CellStore::cells_, the "A" input
    uint32_t cell_b;  // index into CellStore::cells_, the "B" input (UPDATE/TRSM)
    uint32_t cell_c;  // index into CellStore::cells_, the destination (UPDATE only)
    uint32_t flags;   // reserved; 0 for v1
};
static_assert(sizeof(SpsTask) == 32, "SpsTask must be cache-line friendly");

}}  // namespace sTiles::sparse

#endif  // _STILES_SPARSE_TASK_HPP_
