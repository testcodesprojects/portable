# pdpotrf.cpp — Execution Path Map

Parallel tiled Cholesky factorization (`PDPOTRF`, upper `U^T·U`). All code lives
in `namespace sTiles::Process`. File: `tools/compute/pdpotrf.cpp` (~3613 lines).

Every executor is mirrored across **two synchronization backends**:

| Backend  | Rank / size            | Sync primitives                                                        | Abort                                    |
|----------|------------------------|------------------------------------------------------------------------|------------------------------------------|
| pthreads | `STILES_RANK` / `STILES_SIZE` | `ss_init_dp` / `SS_(COND_)WAIT_DP` / `ss_set_done_dp` / `ss_aborted` / `ss_finalize_dp` | `ss_abort()` / `stile->ss_abort = 1`     |
| OMP      | `omp_get_thread_num()` / `worldsize` | `dep_init_dp` / `dep_wait_for_dp` / `dep_set_done_dp` / `dep_is_aborted` / `dep_finalize_dp` | `dep_abort_all()` / `abort_flag.store()` |

The two backends are line-for-line equivalent in numerics and task handling;
only the sync layer differs. Tree reduce adds `ss_cond_wait_tree_e_s_t_y_l_e`
(pthreads) / `dep_wait_for_tree` (OMP).

---

## Upstream activation chain (dpotrf.cpp → pdpotrf.cpp)

The public C-API `sTiles_chol(group, call, obj)` (`dpotrf.cpp` L263) resolves the
`TiledMatrix scheme` + `sTiles_get_params()`, then selects a path in **three layers**.
Only Layer 3 reaches this file.

```
sTiles_chol()  [dpotrf.cpp]
├── Layer 1 — MODULE
│   └── scheme->sparse_handle ?                         (tile_type_mode == 2)
│       └── param[8]==-1 ? pthreads_sparse_dpotrf
│                        : omp_sparse_dpotrf            → sparse_dpotrf.cpp, RETURN
│           (this is the mode-2 "short-circuit upstream"; never enters pdpotrf.cpp)
│
├── Layer 2 — DEVICE   (only in #ifdef STILES_GPU builds)
│   ├── use_gpu && variant==0 && mode==0 && dense ready  → dpotrf_dense_gpu, RETURN
│   ├── use_gpu && variant==0 && mode==1 && semi  ready  → dpotrf_semisparse_gpu, RETURN
│   └── else (not ready / not implemented)               → fall through to CPU
│       (param[10]=gpu_comparison_mode !=0 → also run CPU path first and diff)
│
└── Layer 3 — CPU BACKEND   (the entry into THIS file)
    ├── param[8]==1 → sTiles::omp_dpotrf(L181)
    │                  → omp parallel num_threads(worldsize) proc_bind(close)
    │                  → Process::omp_pdpotrf(scheme, &tracker, worldsize)
    └── else (dflt) → sTiles::pthreads_dpotrf(L139)
                       → parallel_call(stile, Process::pthreads_pdpotrf, scheme)
```

**Knobs, end to end:**

| Signal | Set where | Selects |
|--------|-----------|---------|
| `scheme->sparse_handle` | preprocess | Layer 1: sparse module (mode 2) vs. continue |
| `param[8]` | user | `-1` pthreads-sparse, `0` pthreads (default), `1` OMP |
| `scheme->use_gpu`+`variant`+`param[3]` | user/preprocess | Layer 2: GPU dense / GPU semi / CPU fallback |
| `param[10]` | user | GPU-only vs. GPU+CPU validation diff |
| `param[3]` tile_type_mode | user | 0 dense / 1 semisparse / (2 → Layer 1) |
| `param[14]` | user | serial vs. parallel executor (when size==1) |
| `red_tree_separator_level` | preprocess/ordering | tree-reduction vs. expansion executor |
| `use_boosted_e_trick` | preprocess | boosted (task-list) vs. direct full_dense |

`param[8]` picks the **backend** (which mirror); the fields below pick the
**executor within that mirror**.

---

## Top-level dispatch

`pthreads_pdpotrf(stile)` (L3401) and `omp_pdpotrf(tiledMatrix, dep_tracker, worldsize)` (L3504)
select the executor. Decision inputs:

- `tiledMatrix->use_boosted_e_trick` — boosted (pre-collected task list) vs. direct.
- `stiles_control_params[3]` = **`tile_type_mode`**: `0` dense, `1` semisparse, `2` nonunif/sparse.
- `stiles_control_params[14]` = parallel knob: `0` + size==1 ⇒ **serial** path.
- `tiledMatrix->red_tree_separator_level > 0` ⇒ **tree-reduction** path (`tree_active`).
- `has_semisparse` = `chunkedDenseTiles && semisparseTileMetaCore` (else falls back to dense).

Sets `tiledMatrix->chol_info` = `0` success / `1` not-positive-definite at the end.

### Dispatch decision tree (boosted)

```
use_boosted_e_trick == true
├── tree_active (red_tree_separator_level > 0)
│   ├── mode 0 → *_dpotrf_reduction_dense
│   ├── mode 1 → has_semisparse ? *_dpotrf_reduction_semi : *_dpotrf_reduction_dense
│   └── mode 2 → ERROR (unsupported with tree path) → abort
├── mode 0 (dense)
│   ├── serial_path → *_dpotrf_expansion_dense_serial   (OMP: wrapped in #pragma omp single)
│   └── else        → *_dpotrf_expansion_dense_parallel
├── mode 1 (semisparse)
│   ├── has_semisparse
│   │   ├── serial_path → *_dpotrf_expansion_semi_serial_imp4   (OMP: omp single)
│   │   └── else        → *_dpotrf_expansion_semi_parallel_imp4
│   └── !has_semisparse → *_dpotrf_expansion_dense_parallel  (fallback, e.g. single-tile)
└── mode 2 (nonunif) → ERROR: expected sTiles::sparse to short-circuit upstream → abort

use_boosted_e_trick == false  (direct dense, no pre-collected tasks)
├── numActiveTiles==1 && dimTiledMatrix==1 → *_dpotrf_full_dense(variant 1)
└── numActiveTiles==triangular_size        → *_dpotrf_full_dense(variant 2)
```

---

## The 16 executors

| # | Function | Line | Mode | Sync | Notes |
|---|----------|------|------|------|-------|
| 1 | `pthreads_dpotrf_expansion_dense_serial`      | 108  | 0 dense | pthreads | Whole task list, one worker, no sync |
| 2 | `omp_dpotrf_expansion_dense_serial`           | 175  | 0 dense | OMP      | mirror of #1 |
| 3 | `pthreads_dpotrf_expansion_dense_parallel`    | 242  | 0 dense | pthreads | Per-rank task slice via offsets |
| 4 | `omp_dpotrf_expansion_dense_parallel`         | 326  | 0 dense | OMP      | mirror of #3 |
| 5 | `pthreads_dpotrf_expansion_semi_serial_imp4`  | 422  | 1 semi  | pthreads | Serial; full path_flag dispatch |
| 6 | `omp_dpotrf_expansion_semi_serial_imp4`       | 765  | 1 semi  | OMP      | mirror of #5 |
| 7 | `pthreads_dpotrf_expansion_semi_parallel_imp4`| 1089 | 1 semi  | pthreads | **Main semisparse hot path**; auto FUSE_THRESHOLD |
| 8 | `omp_dpotrf_expansion_semi_parallel_imp4`     | 1512 | 1 semi  | OMP      | mirror of #7 (FUSE_THRESHOLD=16 const) |
| 9 | `pthreads_dpotrf_reduction_dense`             | 1930 | 0 dense | pthreads | Tree reduce, routines 1–10 |
| 10| `omp_dpotrf_reduction_dense`                  | 2095 | 0 dense | OMP      | mirror of #9 |
| 11| `pthreads_dpotrf_reduction_semi`              | 2273 | 1 semi  | pthreads | Tree reduce + semi↔dense scratch bridge |
| 12| `omp_dpotrf_reduction_semi`                   | 2702 | 1 semi  | OMP      | mirror of #11 |
| 13| `pthreads_dpotrf_full_dense`                  | 3122 | direct  | pthreads | Variants 1 & 2, on-the-fly (k,m,n) walk |
| 14| `omp_dpotrf_full_dense`                       | 3262 | direct  | OMP      | mirror of #13 |
| 15| `pthreads_pdpotrf`                             | 3401 | dispatch| pthreads | Entry point |
| 16| `omp_pdpotrf`                                 | 3504 | dispatch| OMP      | Entry point |

---

## Routine codes (task[0])

Tasks are `std::array<int,7>` = `{routine, m, k, n, index1, index2, index3}`,
pulled from `get_chol_tasks(tiledMatrix)` and sliced by `get_chol_task_offsets`.

### Standard routines (expansion + tree paths)
| Code | Op     | Meaning |
|------|--------|---------|
| 1    | DPOTRF | Cholesky on diagonal tile (k,k). Dense: `core_dpotrf`. Semi: `core_dpotrf_upper_banded`. |
| 2    | DSYRK  | Rank-k update of diagonal (k,k) from off-diag (n,k). |
| 3    | DTRSM  | Triangular solve for off-diag (m,k). Semi: `LAPACKE_dtbtrs` banded. |
| 4    | DGEMM  | Trailing update (k,m) -= (n,k)^T·(n,m). Semi: 5-way path_flag dispatch. |

### Tree-reduction routines (paths 9–12 only, red_tree_separator_level > 0)
| Code | Meaning |
|------|---------|
| 5    | Tree DSYRK — accumulate partial `A^T·A` into per-rank scratch `trees[t]->nodes[rank].x` (beta=0 first use via `dirty` flag). |
| 6    | Set diag-reduce dependency token (`= index3`). |
| 7    | DGEADD-reduce — rank 0 folds all ranks' scratch into diagonal (skips `!dirty`). |
| 8    | Tree DGEMM — accumulate partial `A^T·B` into per-rank scratch. |
| 9    | Set off-diag-reduce dependency token (magic value **`165715`**). |
| 10   | DGEADD-reduce — fold scratches into off-diagonal tile. |

---

## Semisparse DGEMM `path_flag` dispatch (case 4)

For semisparse mode-1 executors, case 4 reads a precomputed `path_flag` from
`chol_scatter_index[idx*2 + 1]` and the scatter slot table at
`chol_scatter_packed + chol_scatter_index[idx*2]`. Required by `param[7]==1`
(default in fast mode; `build_chol_scatter_info` populates the arrays).

| path_flag | Strategy | Where |
|-----------|----------|-------|
| 0 | Direct `cblas_dgemm` into contiguous output block (a_contig && b_contig && out_full). |
| 1 | Fused contiguous: hand-rolled SIMD dot + 2-column-of-A blocking (reuses `col_b_j`). |
| 2 | Fused scattered: same SIMD dot, output via `aind_a[i]` indirection. |
| 3 | BLAS into `tmp_tile` + contiguous scatter (inline SIMD axpy). |
| 4 | BLAS into `tmp_tile` + indexed (`aind_a`) scatter, 4-wide unrolled. |

**Note:** the two `expansion_semi` paths (#5–8) use the fully specialized
1/2 SIMD kernels. The two `reduction_semi` paths (#11–12) collapse `path_flag
1||2` into the general BLAS+scatter form (correctness over specialization).

---

## Key knobs & gotchas

- **`FUSE_THRESHOLD`** (semi DSYRK/DGEMM small-update crossover):
  - pthreads parallel imp4 (#7): auto — AMD ⇒ `1<<20` (always fuse), Intel ⇒ `16`;
    override with env `STILES_FUSE_THRESHOLD`.
  - serial imp4 (#5) and OMP imp4 (#6, #8): fixed `constexpr 16`.
- **`red_tree_cores` / `reduce_workers`** (tree paths #9–12): the DGEADD-reduce
  loops bound rank iteration by `red_tree_cores` (falls back to `STILES_SIZE`/
  `worldsize` if 0 or >= size). Surplus threads (`tree_cores < a_exec`) get an
  **empty** task range (`start = tasks.size()`, not `0`) — the tree-reduction
  deadlock fix. Do not revert `": tasks.size()"` back to `": 0"`.
- **`165715`** — magic off-diagonal reduce dependency token (routines 9/10).
- **Banded diagonal storage** (semi): diag tile is `height × (upper_bw+1)`,
  `diag_cols = upper_bw+1`, `base_row = upper_bw`; column-major upper band.
- **`STILES_DUMP_TASKS=1`** — stderr dump of per-rank task slice, per-routine
  counts, and wallclock (present in imp4 parallel + reduction_semi paths).
- **`STILES_MERGE_TILES`** (off by default) — folds trailing `N % tile_size`
  fragment into the last tile via `sTiles_layout_num_tiles`/`_tile_dim`.
  Requires symbolic-side allocation changes before enabling.
- Prefetch: imp4 paths prefetch next task's tiles (`_MM_HINT_T1`), metadata,
  `scatter_packed[next_data_off]`, and `aind` backing arrays.

---

## Cross-references

- `tile_type_mode == 2` (nonunif/sparse) is **not** handled in this file — it
  is expected to short-circuit upstream to `sTiles::sparse` (`dpotrf.cpp`,
  `scheme->sparse_handle`). Reaching mode 2 here is an error → abort.
- Task list + offsets: `get_chol_tasks` / `get_chol_task_offsets`.
- Scatter build: `build_chol_scatter_info` (fast-mode preprocess, `param[7]==1`).
- Core kernels: `core_dpotrf`, `core_dsyrk`, `core_dtrsm`, `core_dgemm`,
  `core_dgeadd`, `core_dpotrf_upper_banded` (`common/core_lapack.hpp`,
  `symbolic/core_semisparse_kernels.hpp`).
