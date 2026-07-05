# Parallel selinv plan for `tools/sparse/`

Goal: make `sTiles::sparse::api::selinv` scale on multiple cores using the same
architecture as `factorize_run_parallel_{omp,pthreads}`. Reuse the existing
SpsState, ProgressMatrix, atomic-counter, per-rank-scratch, level-set+LPT
infrastructure already validated on the chol path.

## 1. Algorithmic recap (today, serial)

Current implementation: [tools/sparse/selinv.cpp:57](selinv.cpp#L57).

Two phases. Z_cs is allocated `allocate_like(L_cs)` so cell layout (I,J,
rows, cols, lx_offset) is identical to L_cs. All reads/writes are inside
`Z_cs` (off-diag cells start as scratch, get overwritten in-place).

### Phase 1 — invert L into Z (forward, `I = 1..n_super`)
For each supernode `I`:
- `Z[I,I] := L[I,I]⁻¹`  (TRTRI in place on a copy of L's diag block)
- For each off-diag cell `(J,I)` (J > I):
  - `Z[J,I] := L[J,I] · L[I,I]⁻¹`  (right TRSM)

Inputs: `L_cs[I,I]` and `L_cs[J,I]` (read-only).
Outputs: `Z_cs[I,I]` and `Z_cs[J,I]`.

**Dependencies inside phase 1:**
- TRSM(J,I) needs Z[I,I] (= TRTRI result) finished.
- Across different `I`, work is fully independent (different output cells,
  different L input cells). **Embarrassingly parallel across supernodes.**

### Phase 2 — Takahashi reverse sweep (`I = n_super..1`)
Per supernode I:
1. **Save M.** Snapshot `M_diag = Z[I,I]` (currently holding `M[I,I] = L[I,I]⁻¹`)
   and `M_off[J] = Z[J,I]` (each currently `M[J,I] = L[J,I]·L[I,I]⁻¹`).
   These get overwritten below.
2. **Compute off-diag `Z[J,I]`** (descending J in I's pattern):
   ```
   acc = 0
   for K in off_list(I):                           # all K > I in I's pattern
     if K == J:    acc += Z[J,J]_subm · M[J,I]
     elif K > J:   acc += Z[K,J]_alignedᵀ · M[K,I]
     else:         acc += Z[J,K]_aligned   · M[K,I]   # I < K < J
   Z[J,I] := -acc
   ```
3. **Compute diag `Z[I,I]`:**
   ```
   Z[I,I] := M[I,I]ᵀ · M[I,I]                    # LAUUM + mirror
   for K in off_list(I):
     Z[I,I] -= M[K,I]ᵀ · Z[K,I]                  # uses just-written Z[K,I]
   ```

**Dependencies inside phase 2:**

For column I to start, every column K > I that contributes to I must already be
fully done — meaning Z[K,K] and all Z[K',K] (K' > K) are finalized.

Concretely for cell `(J,I)` (J > I) to compute, we need:
- For each K in off_list(I), the Z values listed above:
  - `K == J` → `Z[J,J]` finalized
  - `K > J`  → `Z[K,J]` finalized
  - `I < K < J` → `Z[J,K]` finalized (= `Z[K,J]ᵀ` by symmetry — *but cell
    `(J,K)` may not exist!* the code does `Z_cs.find(J,K)` and skips if
    null, since the symbolic pattern is lower-triangular)
- Plus `M[I,I]` (saved into M_diag_buf) and `M[K,I]` (saved into M_off_buf
  for all K) — those are local snapshots, no cross-column dependency.

For cell `(I,I)` to compute we need every `Z[K,I]` (K > I) already finalized
*in this iteration*, since the diag sum reads them.

**Crucial dependency rule:** A column I in phase 2 depends on every supernode K
*whose subtree in the supernodal etree contains I*, restricted to K's that
appear in I's row pattern. Equivalently: I depends on K iff K is an ancestor
of I in `s.sn_etree` AND K ∈ off_list(I). The supernodal etree gives a clean
DAG.

But there's a second class of dependency — `Z[J,K]` when `I < K < J`. Here K
is *between* I and J in the etree, and we need cell `(J,K)`. That cell was
populated when column K was processed (column K's off-diag write to (J,K) iff
J is in K's pattern). So this dep simplifies to: column I depends on column K
having finished, for every K in I's row pattern with K > I. The two classes
collapse into one: **I depends on every K > I that appears in I's row pattern.**

This is a strict subset of "K is an ancestor of I in the etree" — Liu's row-
pattern theorem. So the dependency DAG is *tighter than* the supernodal etree.
Practically, taking `parent(I) -> I` from `s.sn_etree.parent_array()` plus
maybe the pattern is enough — but the cleanest spec is: **I waits for every
supernode index appearing in `Lindx[Xlindx[I-1] .. Xlindx[I]-2]` to be done.**

## 2. What we keep from the chol path

Already in [tools/sparse/executor.{hpp,cpp}](executor.hpp) and used by chol —
all directly reusable, no rewrites needed:

- `SpsState` struct: holds `symbolic`, `cells`, `tasks`, `progress`,
  `abort_flag`, `updates_remaining[]`, `cell_locks[]`, `update_scratch[]`.
  We extend it (or add a sibling state) to also hold a selinv scratch and a
  selinv task list.
- `ProgressMatrix` (one byte per cell): selinv signals "this cell's Z value
  is final" the same way chol signals "this cell's L value is final".
- `std::atomic<int> updates_remaining[i]`: chol uses this to count remaining
  UPDATEs into cell `i`. Selinv uses an analogous counter for "remaining
  Z-contributors into this cell" (see §4).
- `std::atomic<int> cell_locks[i]`: per-cell spinlock for serializing
  concurrent writers. Selinv has fewer write conflicts (each cell is written
  only inside its own column's iteration), but `(I,I)` accumulates from
  `K > I` and `(J,I)` accumulates from many K — locks still useful unless
  we keep per-task accumulator buffers (we will — see §4).
- `factorize_run_parallel_pthreads` / `_omp`: literal copy with the
  `factorize_run` body replaced by a `selinv_run` body.
- LPT level-set partitioning in [collect.cpp](collect.cpp): same algorithm
  works — supernodes at the same etree level run in parallel; LPT balances
  load across ranks; the same `compute_levels` helper is reused.

## 3. New code surface

### 3.1 New task kinds (extend `task.hpp` or add `selinv_task.hpp`)

```cpp
enum class SelinvOp : uint8_t {
  PHASE1_TRTRI = 1,   // Z[I,I] := L[I,I]^{-1}   (independent across I)
  PHASE1_TRSM  = 2,   // Z[J,I] := L[J,I] · Z[I,I]   (depends on PHASE1_TRTRI(I))
  PHASE2_OFF   = 3,   // compute Z[J,I] in phase 2 (descending order)
  PHASE2_DIAG  = 4,   // compute Z[I,I] in phase 2
};

struct alignas(32) SelinvTask {
  SelinvOp op;
  uint8_t  pad0[3];
  uint32_t I;           // pivot column
  uint32_t J;           // for PHASE1_TRSM and PHASE2_OFF; 0 otherwise
  uint32_t cell_diag;   // index of (I,I) — saves a find() in the hot path
  uint32_t cell_off;    // index of (J,I) for PHASE1_TRSM / PHASE2_OFF
  uint32_t flags;       // reserved
};
static_assert(sizeof(SelinvTask) == 32);
```

Decision: keep `SelinvTask` separate from `SpsTask` (chol task). They have
different op spaces and we don't want runtime branching in the executor.

### 3.2 New `CollectedSelinvTasks` (mirror `CollectedTasks`)

```cpp
struct CollectedSelinvTasks {
  std::vector<SelinvTask> tasks;
  std::vector<int>        offsets;             // n_threads+1
  std::vector<int>        contrib_remaining;   // size = n_super+1; per-supernode
                                               // count of K > I whose cells
                                               // exist and contribute to I
                                               // in phase 2.
};
```

Why `contrib_remaining` per-supernode (not per-cell)? In phase 2, column I
starts only when **all** ancestor supernodes K in I's pattern are done.
That's a per-column notion. Per-cell `updates_remaining` (chol-style) doesn't
quite map because phase 2 has nested loops where we read many cells of K
before writing any of I.

### 3.3 New entry in `executor.hpp`

```cpp
// Same SpsState; just additional members or a sibling state.
void selinv_prepare(SpsState& state, const CollectedSelinvTasks& tasks);
void selinv_run(int rank, SpsState* state, const CollectedSelinvTasks* tasks);
void selinv_run_parallel_pthreads(SpsState& state, const CollectedSelinvTasks& tasks);
void selinv_run_parallel_omp     (SpsState& state, const CollectedSelinvTasks& tasks);
```

Or — simpler — add a parallel `selinv` flavor that owns its own state inside
[selinv.cpp](selinv.cpp), and keep the existing `selinv(...)` serial entry
alive (so callers that want serial still have it).

### 3.4 New collector `collect_selinv_tasks` (analog of `collect_tasks`)

Inputs:
- `Symbolic s`
- `const CellStore& cs` (Z layout, identical to L layout)
- `int n_threads`

Algorithm:
1. **Phase 1 emit** — embarrassingly parallel:
   - For each supernode I, emit `PHASE1_TRTRI(I)` then one `PHASE1_TRSM(I,J)`
     per off-diag cell. The TRSM depends only on the same column's TRTRI; no
     cross-column deps.
   - LPT-distribute supernodes across ranks (single big level — they are all
     independent). Reuse the LPT helper from `collect_tasks`.
2. **Phase 2 emit** — level-set on supernodal etree, **iterated reverse**:
   - Compute level using `compute_levels` (already exists in collect.cpp).
   - For phase 2, the dependency runs root → leaves. So iterate
     `L = max_level..0` (root first), and within each level, LPT-distribute
     supernodes by their flop estimate.
   - For each supernode I, emit:
     - `PHASE2_OFF(I, J)` for each J > I in I's off_list **in descending J
       order** (matches the serial code's order; needed because (J,I) reads
       cells produced by smaller-K iterations of the same column — actually
       no, (J,I) reads only cells from columns K > I, so within column I the
       descending-J order is just convention, all (J,I) for fixed I are
       independent of each other once column I is "ready").
     - `PHASE2_DIAG(I)` last in column I.
   - Set `contrib_remaining[I] = number of K > I in Lindx(I) such that K
     <= n_super and K's column was visited in phase 2`. This is the count of
     ancestor supernodes that must finish before I can run.

Key subtlety: within a column, all PHASE2_OFF(I, J*) and the final
PHASE2_DIAG(I) can run in parallel **only if** they don't write to overlapping
cells. They don't — PHASE2_OFF(I, J) writes only to `Z[J,I]`, PHASE2_DIAG(I)
writes only to `Z[I,I]`. They all *read* from `Z[K,I]` (M-snapshots) and
`Z[K,*]` for K > I, both already finalized. So phase 2 within a column is
actually **independent across J's, plus the diag**.

But: PHASE2_DIAG(I) reads `Z[K,I]` for K > I — these are the M-snapshots
*saved before phase 2 of column I started*. Those snapshots live in
M_diag_buf and M_off_buf today, which are stack-allocated per outer-loop
iteration. To parallelize, we must save the M snapshots into per-cell buffers
that survive long enough.

### 3.5 M-snapshot storage

Today, M-snapshots are local variables. The serial code overwrites Z[J,I]
with the new value in the same loop iteration that read M[K,I]. To
parallelize, M[K,I] readers must read from a stable snapshot, not from Z[K,I]
(which gets overwritten in K's own iteration earlier).

Two clean options:

**Option A: Pre-snapshot all M values up front.** Before phase 2, copy the
entire `Z_cs` arena (which holds M = L⁻¹) into a parallel `M_arena` of the
same size. Phase 2 reads from `M_arena`, writes to `Z_cs`. Cost: one extra
arena allocation (already sized by `arena_size_`), one memcpy. For our
shipsec1 case that's 49M doubles = 376 MB. Acceptable for the parallel win.

**Option B: Per-column M-snapshot at task emission.** Allocate `M_diag` and
`M_off` for each column lazily, freed when its consumers all finish. More
complex, more memory plumbing. Skip.

**Recommendation: Option A.** Add a sibling `arena_M` in `SpsState` (or a
small `MArena` field), copy from `Z_cs.arena_data()` to `arena_M` between
phase 1 and phase 2 in `selinv_prepare`. Provide cell pointers `M_cell(idx)`
that index into `arena_M` using the same offsets as `Z_cs.cells_[idx].nzval -
Z_cs.arena_data()`.

### 3.6 Cross-column dependency tracking

Per supernode I, `contrib_remaining[I]` starts at the count of ancestor
supernodes K > I in I's pattern. When supernode K's PHASE2_DIAG(K) finishes
(K's column fully done), it decrements `contrib_remaining[J]` for every J < K
that has K in their pattern.

To avoid the O(n_super²) reverse-pattern scan, build once at `selinv_prepare`
time: a `consumers_of[K]` adjacency list of all I < K with K in I's pattern.
Then PHASE2_DIAG(K) just walks `consumers_of[K]` and atomic-decrements each.

A column I is "ready" when `contrib_remaining[I] == 0`. PHASE2_OFF(I, *) and
PHASE2_DIAG(I) all spin on this one counter (or on a cheap progress flag
"column I ready" derived from it).

## 4. Memory & races

### 4.1 Per-rank scratch

Phase 2 needs per-(I,J) acc accumulator (size `rJ × wI`) and per-(I,J,K)
Z_aligned buffer (size up to `rK × rJ`). Reuse `update_scratch[rank]` slot
*sized for selinv's largest combined buffer* — `selinv_prepare` walks the
selinv task list to find the max.

Allocate two per-rank scratches:
- `selinv_acc_scratch[rank]` — accumulator for phase 2 off-diag.
- `selinv_align_scratch[rank]` — Z_aligned and Z_sub buffer (variable size).

Sizing: max over the rank's PHASE2_OFF tasks of `rJ_in_I × wI` for acc, and
`max(rK_in_I × rJ_in_I, rJ_in_I × rJ_in_I)` for align. Easy to bound from
the task list.

### 4.2 Lock-free design

Each Z cell `(J,I)` has exactly **one writer task** in phase 2: PHASE2_OFF(I, J).
Each `Z[I,I]` has exactly one writer: PHASE2_DIAG(I). So no per-cell locks
needed in selinv — `cell_locks` from chol is unused here (or repurposed for
the contrib_remaining counters).

### 4.3 ProgressMatrix usage

selinv's `progress[cell_idx]` semantic: "this Z cell is finalized". Each
PHASE2_OFF / PHASE2_DIAG sets progress on its output cell at the end. Other
tasks `wait_done` on input cells before reading.

But wait — phase 2 reads M values (snapshots), not Z. The cells whose
*progress* matters are:
- `Z[K,J]` (K > J > I) — written by PHASE2_OFF(J, K) — but cells (K, J) are
  produced when column J is processed, which (since I < J) happens *after* I
  in the reverse iteration. So the descending order of column processing
  guarantees this. The `contrib_remaining[I]` mechanism enforces it.
- `Z[J,K]` (I < K < J) — written by PHASE2_OFF(K, J).

Easier to use **column-level readiness** rather than cell-level: column I is
"ready" when `contrib_remaining[I] == 0`, meaning all ancestor columns are
done. Once ready, all PHASE2_OFF(I, *) tasks for column I can fire.

Within a column, all PHASE2_OFF(I, J*) tasks run in parallel, then
PHASE2_DIAG(I) waits for them (it reads the just-written Z[J,I] for all J's).

### 4.4 Phase 2 task ordering inside a rank

A single rank's slice may contain tasks from many columns (LPT-spread). Each
task spins on `contrib_remaining[task.I]`. So the rank's executor loop is:

```cpp
for each SelinvTask t in this rank's slice:
    if (t.op == PHASE2_OFF || t.op == PHASE2_DIAG) {
        spin until contrib_remaining[t.I] == 0;
        if (t.op == PHASE2_DIAG) {
            spin until all PHASE2_OFF(t.I, *) of this column are done;
            // — track via a per-column "n_off_remaining" atomic.
        }
        execute t;
        if (t.op == PHASE2_OFF) decrement n_off_remaining[t.I];
        if (t.op == PHASE2_DIAG) {
            // column I is now fully done — decrement consumers' counters.
            for each I' in consumers_of[t.I]:
                contrib_remaining[I'].fetch_sub(1);
        }
    } else {
        // PHASE1_TRTRI / PHASE1_TRSM
        if (t.op == PHASE1_TRSM) wait for PHASE1_TRTRI of same I;
        execute;
    }
```

We need: `phase1_trtri_done[I]` (one-bit progress per supernode), and
`n_off_remaining[I]` (per supernode count), and `contrib_remaining[I]`.

Not minimal — but small (3 atomics per supernode). For 6068 supernodes
on shipsec1 that's 72 KB. Trivial.

## 5. Step-by-step implementation order

This is the build sequence I'd follow.

### Step 1 — collect_selinv_tasks (no executor yet)
- Add `selinv_task.hpp` with `SelinvOp` and `SelinvTask`.
- Add `collect_selinv.{hpp,cpp}` with `CollectedSelinvTasks` and the
  collector. Implement single-thread fast path first (n_threads == 1), no
  level-set: emit phase 1 in elimination order, phase 2 in reverse
  elimination order.
- Add `consumers_of` build (one pass over supernodal Lindx).
- Unit test: print task counts; verify that phase 1 has `n_super +
  Σ off_list` tasks and phase 2 has same shape.

### Step 2 — selinv_run (single-rank executor)
- Add `selinv_run(0, &state, &tasks)` that walks the task list serially and
  executes each task with the same kernel calls as the current
  `selinv(...)`. **Validate** that single-rank `selinv_run_parallel_omp`
  gives the same Z as the current serial `selinv` on bcsstk16/shipsec1
  (compare logdet, tr(A⁻¹), off_abs).
- This step is a literal lift-and-shift of the serial body — no scaling
  yet, but proves the task encoding is correct.

### Step 3 — M-snapshot
- Add `selinv_state` extension for `arena_M` (separate `unique_ptr<double[]>`
  + size).
- In `selinv_prepare`, after L_cs is loaded, copy
  `Z_cs.arena_data() → arena_M` once between phase 1 and phase 2.
- Refactor PHASE2_OFF / PHASE2_DIAG to read M values from `arena_M` instead
  of from Z_cs. Re-validate.

### Step 4 — multi-rank (LPT level-set partitioning)
- Implement multi-rank `collect_selinv_tasks`:
  - Phase 1: one big bucket, LPT-spread across ranks.
  - Phase 2: level-set with `level[I]` from `compute_levels`. Iterate
    `L = max_level..0` (root → leaves). LPT within each level.
- Implement atomic-counter waits in `selinv_run`.
- Implement `consumers_of[K]` walk inside PHASE2_DIAG to decrement.
- Implement `selinv_run_parallel_pthreads` and `_omp` (literal copies of
  chol's parallel wrappers).

### Step 5 — wire in api.cpp
- Split `api::selinv` into `api::selinv_pthreads` and `api::selinv_omp`
  (mirror `chol_pthreads`/`chol_omp`).
- Update `tools/compute/sparse_dpotri.cpp` to dispatch via param[8]
  (default OMP, same convention as `sparse_dpotrf.cpp`).

### Step 6 — verify scaling
- Bench shipsec1 with `--cores=1,2,4 --modes=3 --variant=0`. Expect:
  - 1 core: roughly same as current serial selinv (~2.4s), within 5%
    overhead.
  - 4 cores: significant speedup. Honest target: 2× (selinv has more
    cross-column deps than chol, so 4× is unlikely without further
    work).
- Re-validate fingerprints (logdet, tr(A⁻¹), off_abs) match between mode 1
  and mode 3 across all core counts.

## 6. Risks & open questions

### 6.1 Phase 2's critical path is long
Even with perfect parallelism within a level, phase 2's level-set forces
serialization across levels. For matrices with deep narrow trees (like
`pedigree`), levels = O(n) and parallelism is low. Same problem as chol —
LPT mitigates it but doesn't fix it. Honest expectation: 2–3× on 4 cores
for shipsec1, less on pedigree-shaped trees.

### 6.2 M-arena memory cost
On large matrices (net1628760: 175M doubles = 1.4 GB), the M-arena doubles
the sparse path's working set. Documented, gate behind `selinv_prepare`,
free immediately after `selinv_run` completes. For small matrices it's a
non-issue.

### 6.3 V1 selinv shape assumption
The current serial code already has the row-subset gather (find_pos +
KI_to_KJ alignment) — see [selinv.cpp:184-201](selinv.cpp#L184). So we don't
need to revisit that; the parallel version inherits the gather logic
verbatim per-task.

### 6.4 PHASE1 vs PHASE2 bridge
Phase 1 and phase 2 have a hard barrier between them (phase 2 reads M which
phase 1 wrote). Two options:
- **Single global barrier** between the two phases (simple; one OMP
  `#pragma omp barrier` if everyone is in the same parallel region, or
  separate `parallel` regions for pthreads).
- **Soft barrier per column** (PHASE2(I) waits for PHASE1(K) for K in I's
  pattern). More complex, marginal win.

Recommendation: global barrier. Phase 1 is fast already (TRTRI + TRSM only,
embarrassingly parallel); the win is in phase 2.

## 7. File touch list

New files:
- `tools/sparse/selinv_task.hpp`
- `tools/sparse/collect_selinv.hpp` + `.cpp`

Modified files:
- `tools/sparse/executor.hpp` — add `selinv_run_parallel_*` declarations.
- `tools/sparse/executor.cpp` — add `selinv_prepare`, `selinv_run`,
  `selinv_run_parallel_pthreads`, `selinv_run_parallel_omp`.
- `tools/sparse/selinv.hpp` — keep `selinv(...)` serial entry; add no public
  surface (parallel path goes through api).
- `tools/sparse/api.hpp` + `.cpp` — split `selinv` into `selinv_pthreads`
  and `selinv_omp`. The serial body stays callable from both for now (we
  switch them to parallel in step 5).
- `tools/compute/sparse_dpotri.cpp` — switch from a single `selinv(...)`
  call to dispatching `chol_pthreads`/`chol_omp` style helpers.
- `tools/sparse/Makefile` — append new sources.

No changes to:
- `tools/sparse/supernode.{hpp,cpp}`, `tools/sparse/symbolic.{hpp,cpp}`,
  `tools/sparse/etree.{hpp,cpp}`, `tools/sparse/kernels.hpp`,
  `tools/sparse/solve.cpp`, `tools/sparse/element_access.{hpp,cpp}`,
  `tools/sparse/progress.hpp`, `tools/sparse/task.hpp`.

## 8. Effort estimate

- Step 1 (collect_selinv_tasks single-thread): ~2 hours
- Step 2 (selinv_run serial via task list): ~3 hours
- Step 3 (M-snapshot arena): ~2 hours
- Step 4 (multi-rank LPT + atomics): ~4 hours
- Step 5 (api wiring): ~1 hour
- Step 6 (validation + bench): ~2 hours

**Total: ~14 hours of focused work.** Most of it is Step 4 (correct atomic
ordering on contrib_remaining and column readiness signals). The rest is
mechanical reuse of patterns already validated in chol.
