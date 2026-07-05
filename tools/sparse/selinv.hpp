#ifndef SPS_SCHED_SELINV_HPP
#define SPS_SCHED_SELINV_HPP

#include "symbolic.hpp"
#include "supernode.hpp"

namespace sTiles { namespace sparse {

// Selected (sparse) inverse of A on the pattern of L+L^T.
//
// After `factorize_run` produced L in `L_cs`, this routine allocates `Z_cs`
// with the same cell layout as `L_cs` and computes Z = A^{-1} restricted to
// pattern(L+L^T). Cell coordinates and storage are bit-identical between
// L_cs and Z_cs, so a caller that already indexed L by (J, I) can index Z
// the same way. Entries outside pattern(L+L^T) are not computed.
//
// Algorithm: supernodal Takahashi (Erisman-Tinney) recursion. Two phases,
// matching the 9-op encoding in sTiles' `inv_variant4`:
//
//   Phase 1 (forward, I = 1..n_super):
//     - TRTRI on diagonal cell (I, I): M[I,I] = L[I,I]^{-1} (overwrites Z's
//       diag cell).
//     - TRSM (right) on each off-diag cell (J, I): M[J,I] = L[J,I] · L[I,I]^{-1}
//       (stored in Z's off-diag cells; the original L is left intact in
//       L_cs so the caller can still TRSM-solve afterwards if desired).
//
//   Phase 2 (reverse, I = n_super..1):
//     - LAUUM + mirror on (I, I): seeds Z[I,I] := M[I,I]^T · M[I,I].
//     - For each K > I in I's pattern (descending): Z[I,I] -= M[K,I]^T · Z[K,I].
//     - For each (J, I) with J > I in I's pattern (descending), and each K:
//         K > J: Z[J,I] -= M[K,I]^T · Z[K,J]
//         K ≤ J: Z[J,I] -= M[K,I]^T · Z[J,K]^T
//
// Single-thread for v1. Reads/writes only happen at cells that exist in
// `L_cs` — automatic restriction to pattern(L+L^T), no work on structurally
// zero blocks.
//
// V1 LIMITATION: requires that Z[J, K]'s row count equals Z[J, I]'s row
// count for all (I, J, K) triples touched by the recursion, i.e. that
// J's rows in K's pattern equal J's rows in I's pattern. This holds for
// fundamental supernodes (where J's rows are always exactly width(J) when
// J appears) but can fail under relaxation when amalgamated supernodes
// don't have full-width row coverage. In that case `selinv` throws
// `std::runtime_error("selinv: shape mismatch ...")`. The production fix
// is a row-subset gather analogous to UPDATE's scatter — same machinery.
// Smoke matrices pass; ferris-scale relaxation does not — pending Phase 12.1.
void selinv(const Symbolic& s, const CellStore& L_cs, CellStore& Z_cs);

}}  // namespace sTiles::sparse

#endif  // SPS_SCHED_SELINV_HPP
