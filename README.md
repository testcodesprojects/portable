# sTiles (portable)

A self-contained, cross-platform build of the **sTiles** sparse tiled linear
algebra library. Builds an optimized, portable shared library on Linux and
macOS (Intel and Apple Silicon).

## Build

```bash
make all
```

Produces `lib/libstiles.so` (Linux) or `lib/libstiles.dylib` (macOS), optimized
for portability (AVX2 baseline on x86-64; Apple Silicon native on arm64).

- After editing any file, plain `make` rebuilds only what changed.
- `make MARCH=native` tunes for the current CPU; `make MARCH=avx512` targets
  AVX-512 servers; `make MARCH=generic` runs on any x86-64.
- `make show-config` prints the detected platform, compiler, and BLAS.
- `make help` lists all targets.

### Prerequisites

- A C/C++/Fortran toolchain (`g++`/`gfortran` on Linux, or Homebrew `gcc` on macOS)
- `hwloc` (topology detection) and a BLAS/LAPACK:
  - **Linux**: Intel MKL *or* OpenBLAS (`sudo apt install libopenblas-dev liblapacke-dev libhwloc-dev`)
  - **macOS**: `brew install gcc libomp hwloc openblas lapack`
    (OpenBLAS provides the LAPACKE interface that Apple's Accelerate lacks)
- Network on the first build (fetches libxsmm; cached afterwards)

The graph-ordering dependencies (SCOTCH, METIS + GKlib, and the AMD/CAMD/
COLAMD/CCOLAMD/BTF part of SuiteSparse) are **vendored** and compiled directly
into the library — nothing to install. libxsmm is cloned and built on first use.

## Test it

```bash
cd bench && ./run.sh
```

Factorizes a few matrices, checks correctness (solve residual), and prints
timings. Exits non-zero if anything fails. See [bench/README.md](bench/README.md).

## Continuous integration

[.github/workflows/build.yml](.github/workflows/build.yml) builds the library
natively on Linux, Intel macOS, and Apple Silicon, runs the speed test on each,
and uploads the resulting libraries as artifacts.

## Install

```bash
make install PREFIX=$HOME/.local   # or sudo make install  (default /usr/local)
```

Installs `libstiles`, the `stiles.h` header, and a `pkg-config` file.
