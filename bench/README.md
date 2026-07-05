# sTiles speed test

A tiny, self-contained smoke test: does the built library **work**, and how
**fast** is it on a few representative matrices?

## Run it

```bash
# from the repo root, build the library once:
make all

# then:
cd bench
./run.sh
```

Output is one row per matrix (phase timings + solve residuals at two RHS
widths) and a PASS/FAIL verdict:

```
matrix                n        nnz   fill  chol(s)   selinv  slv1(s)  slvT(s)          logdet     res(1)   res(T+1)  status
diff.mtx          21529     200017   3.60   0.0172   0.0510   0.0028   0.0281     148738.4035   3.22e-16   6.10e-16  PASS
ferris.mtx        42961     650694   5.39   0.0403   0.1337   0.0106   0.0735     296806.3956   1.94e-15   9.13e-15  PASS
spacetime.mtx     24931     644273  26.27   0.4318   0.7163   0.0137   0.1124     172241.5553   3.96e-16   4.16e-16  PASS
```

`run.sh` exits non-zero if **any** matrix fails (either solve residual ≥ 1e-6
or a non-finite log-determinant), so CI can gate on it.

## Options

```bash
CORES=8 TILE=40 ./run.sh          # override thread count / tile size
./run.sh ../run/mtx/inla_graph_ferris.mtx   # run a custom matrix set
```

## What it measures

Each matrix runs the full INLA path in **auto mode** (the library's selector
picks dense / semisparse / sparse):

- **chol** — numeric Cholesky factorization time
- **selinv** — selected-inverse (Takahashi) time
- **slv1 / slvT** — triangular solve time at rhs = 1 and rhs = tile+1 (41 by
  default); best-of-3 (the first solve pays one-time lazy setup)
- **res(1) / res(T+1)** — worst-column relative residual `‖x − x_true‖/‖x_true‖`
  for a manufactured solution `A·x_true = b`; the correctness check

## Files

- `speedtest.cpp` — ~150-line driver; also a minimal example of the public API
- `Makefile` — links against `../lib/libstiles.so` via the repo's `bench.inc`
  (auto-detects MKL/OpenBLAS on Linux, Accelerate on macOS)
- `mats/` — three matrices (diff = diffusion GMRF, ferris, spacetime)
- `run.sh` — build + run all matrices, print the table, gate on PASS/FAIL
