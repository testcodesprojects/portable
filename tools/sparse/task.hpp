#ifndef SPS_SCHED_TASK_HPP
#define SPS_SCHED_TASK_HPP

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

#endif  // SPS_SCHED_TASK_HPP
