#ifndef SPS_SCHED_SELINV_TASK_HPP
#define SPS_SCHED_SELINV_TASK_HPP

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

#endif  // SPS_SCHED_SELINV_TASK_HPP
