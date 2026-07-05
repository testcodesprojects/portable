#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One command to (re)build the portable, redistributable sТiles bundle.
#
# This CANNOT be a plain host `make all`: the glibc floor of a binary equals the
# glibc of the machine it is compiled on, so a build on this host (glibc 2.39)
# would only run on the newest distros. The portable bundle is therefore built
# INSIDE an old-glibc (2.28) manylinux container, rootless, via udocker/proot.
#
# What it does, every run (so it always reflects your current source):
#   1. sync a fresh lean copy of THIS tree (minus the huge tests/ matrices)
#   2. ensure udocker + the manylinux_2_28 image + the build container exist
#   3. run a CLEAN build in the container (portable codegen, MKL embedded)
#   4. package -> dist/stiles-linux-x86_64.tar.gz   (glibc 2.27 floor, any x86-64)
#
# Usage:   bash tools/package/build_portable_bundle.sh
# Or:      make portable-bundle
# Knobs (env):  STILES_BUILD_COPY  STILES_CONTAINER  STILES_CXXRT_LIBDIR  MKLROOT_HOST
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE=$(cd -- "$(dirname -- "$0")" && pwd -P)
ROOT=$(cd -- "$HERE/../.." && pwd -P)                 # adv_sTiles root
COPY=${STILES_BUILD_COPY:-/tmp/stiles-mlx}
IMAGE=quay.io/pypa/manylinux_2_28_x86_64
CNAME=${STILES_CONTAINER:-mlxbuild}
MKL_HOST=${MKLROOT_HOST:-/opt/intel}
CXXRT=${STILES_CXXRT_LIBDIR:-$HOME/miniconda3/lib}
export PATH="$HOME/.local/bin:$PATH"

say(){ printf '\n\033[1m== %s\033[0m\n' "$*"; }
die(){ printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# ---- preconditions ----------------------------------------------------------
command -v udocker >/dev/null 2>&1 || die "udocker not found. Install once:  pip install --user udocker && udocker install"
[ -d "$MKL_HOST/oneapi/mkl" ] || die "Intel oneAPI MKL not found under $MKL_HOST (set MKLROOT_HOST). The static MKL is baked into libstiles.so."
[ -e "$CXXRT/libstdc++.so.6" ] || die "No libstdc++.so.6 in $CXXRT (set STILES_CXXRT_LIBDIR). Needs a NEW-GLIBCXX + LOW-glibc libstdc++, e.g. conda-forge libstdcxx-ng."

say "image + container ($IMAGE -> $CNAME)"
udocker images 2>/dev/null | grep -q manylinux_2_28 || udocker pull "$IMAGE"
udocker create --name="$CNAME" "$IMAGE" >/dev/null 2>&1 || true   # idempotent: ok if it exists
udocker setup --execmode=P1 "$CNAME" >/dev/null 2>&1 || true

say "sync fresh source copy -> $COPY (excluding matrices/results/build artifacts)"
rm -rf "$COPY"; mkdir -p "$COPY/tests"
rsync -a \
  --exclude='tests/' --exclude='dist/' --exclude='results/' --exclude='ibex/' \
  --exclude='docs/' --exclude='pardiso/' --exclude='improve/' \
  --exclude='*.o' --exclude='*.so' --exclude='*.a' \
  "$ROOT/"  "$COPY/"
# one small sample matrix so the bundle ships a runnable example
s=$(find "$ROOT/tests" -name '*.mtx' -printf '%s %p\n' 2>/dev/null | sort -n | awk 'NR==1{print $2}')
[ -n "${s:-}" ] && cp -L "$s" "$COPY/tests/"

say "clean build + package inside the container (glibc 2.28, portable codegen)"
udocker run \
  -v "$COPY":/work \
  -v "$MKL_HOST":/opt/intel \
  -v "$CXXRT":/opt/cxxrt \
  "$CNAME" bash /work/tools/package/container_build.sh

OUT="$COPY/dist/stiles-linux-x86_64.tar.gz"
[ -f "$OUT" ] || die "container build did not produce $OUT"
mkdir -p "$ROOT/dist"
cp -f "$OUT" "$ROOT/dist/stiles-linux-x86_64.tar.gz"

say "DONE"
echo "  -> $ROOT/dist/stiles-linux-x86_64.tar.gz   ($(du -h "$ROOT/dist/stiles-linux-x86_64.tar.gz" | cut -f1))"
echo "  glibc floor / verify:  run  bash tools/package/build_portable_bundle.sh  again any time after source changes."
