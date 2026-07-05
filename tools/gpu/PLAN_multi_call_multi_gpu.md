# GPU Multi-Call Multi-GPU Plan — sTiles modes 0 & 1

Author intent: extend `sTiles_preprocess_group` so that the symbolic
phase decides, based on numActiveTiles, mode, N_calls and the node's
GPUs, how to spread chol calls across available GPU memory.

Scope (initial): tile modes 0 (dense) and 1 (semisparse) only.

## Decision logic (inside sTiles_preprocess_group)

After symbolic finishes we know: numActiveTiles, tile_size, mode, N_calls.

1. discover_gpus() → G, M_g per device
   - cudaGetDeviceCount + cudaMemGetInfo
   - fall back to CPU path if G == 0

2. Memory estimator (analytical, mode-specific):
   - mode 0 (dense):   B_tiles ≈ numActiveTiles · ts² · 8
   - mode 1 (semi):    B_tiles ≈ Σ (height_i · (upper_bw_i + 1)) · 8
   - + cuSOLVER workspace (cusolverDnDpotrf_bufferSize)
   - + input value buffer
   - + DTRSM/DSYRK/DGEMM scratch
   - + 10% safety
   → B_call

3. Total demand vs available:
   - T = N_calls · B_call
   - C = G · M_g · 0.85
   - if T > C → error: too many calls; suggest batching

4. K_per_gpu = floor(M_g · 0.85 / B_call)

5. Dispatch strategy:
   case A:  N_calls ≤ K_per_gpu        → 1 GPU, all calls concurrent
   case B:  K_per_gpu < N_calls ≤ K·G  → spread across ⌈N_calls/K⌉ GPUs
   case C:  N_calls > K·G              → use all G, queue ⌈N_calls/G⌉ per GPU

6. Record plan in scheme: (call_id) → (gpu_id, slot)

## Execution (single process, threads)

- One worker thread per used GPU.
- Each worker creates a persistent context: cuBLAS + cuSOLVER handles,
  K cuStreams, K tile-slabs preallocated.
- Workers consume work items (call_id, values_ptr): upload → chol →
  logdet/selinv → copy result back.

## First-call probe (per GPU)

- Wrap first chol with cudaMemGetInfo before/after to get peak measured.
- If measured > 1.2 · B_estimated:
    - case A/B: log warning, continue (headroom present).
    - case C: reduce K, replan remaining work to avoid OOM.

## Implementation phases (deferred)

| phase | scope |
|---|---|
| 1 | Discovery + analytical estimator inside sTiles_preprocess_group; planner returns dispatch plan |
| 2 | Wire tools/gpu/dpotrf/ into build (currently sandbox); single-GPU single-call works |
| 3 | K-stream concurrent calls on one GPU |
| 4 | Multi-GPU worker pool + dispatch from planner |
| 5 | First-call probe + replan-on-mismatch |
| 6 | Bench on ibex multi-GPU node |

Estimated ~2 weeks of focused work.

## Existing assets to use

- tools/gpu/dpotrf/gpu_persistent_context.hpp — per-call context skeleton
- tools/gpu/dpotrf/gpu_tile_slab.hpp — preallocated slab pattern
- tools/gpu/dpotrf/dpotrf_dense_gpu.{cpp,hpp} — dense chol primitive
- run/multipl_calls/bench_mc.cpp — multi-call benchmark (CPU only, to extend)
