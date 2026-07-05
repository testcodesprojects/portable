#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Runs INSIDE a manylinux_2_28 container (glibc 2.28) to produce a maximally
# portable sТiles bundle:
#   * low glibc floor      <- the container's old glibc baseline
#   * CPU-portable codegen <- strip -march=native (MKL still runtime-dispatches)
#   * MKL embedded         <- host's static MKL, bind-mounted at /opt/intel
#
# Expects:  /work       = copy of the sТiles source tree (rw bind mount)
#           /opt/intel  = host Intel oneAPI (read-only bind mount, for static MKL)
# Produces: /work/dist/stiles-linux-x86_64.tar.gz
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
echo "== container: $(. /etc/os-release; echo "$PRETTY_NAME")  | glibc $(ldd --version | awk 'NR==1{print $NF}') =="

# toolchain (manylinux puts gcc-toolset on PATH)
gcc --version | head -1
gfortran --version | head -1 || { echo "FATAL: no gfortran in container"; exit 1; }

echo "== install build deps (numa/hwloc/compression + flex/bison for ordering codegen) =="
yum install -y -q numactl-devel hwloc-devel zlib-devel bzip2-devel xz-devel flex bison >/dev/null 2>&1 \
  || echo "WARN: yum had issues (continuing — deps may already be present)"

export MKLROOT=/opt/intel/oneapi/mkl/2026.0
[ -f "$MKLROOT/lib/libmkl_core.a" ] || { echo "FATAL: static MKL not mounted at $MKLROOT"; exit 1; }
echo "== MKLROOT=$MKLROOT (static MKL present) =="

cd /work
# --- portable codegen: remove -march=native from BOTH build paths -------------
sed -i 's/-march=native//g' make.inc run/Makefile
echo "== stripped -march=native (lib + run) =="

# --- the CLI must NOT link host MKL directly ---------------------------------
# run/Makefile's LDFLAGS pulls $(BLAS_LIBS) = host oneAPI MKL .so, which drags in
# newer-glibc symbols (__isoc23_strtol@2.38, __libc_start_main@2.34) and a newer
# GLIBCXX. bench_stiles needs no BLAS of its own (it calls the sТiles API); BLAS
# lives inside libstiles.so as the static MKL .a (clean, GLIBC_2.27). Drop it.
sed -i 's/\$(BLAS_LIBS)//g' run/Makefile
echo "== dropped host MKL from CLI link (BLAS comes from libstiles.so) =="

PORT='-O3 -funroll-loops -ftree-vectorize'        # == DIST_OPT_GENERIC (no -march)
J=$(( $(nproc) > 8 ? 8 : $(nproc) ))

echo "== build deps (ordering libs, inline) =="
make -j$J OPT_FLAGS="$PORT" deps               >/tmp/b_deps.log 2>&1 || { echo "DEPS FAILED:"; tail -40 /tmp/b_deps.log; exit 1; }
echo "== build libstiles + shared (.so) =="
make -j$J OPT_FLAGS="$PORT" libstiles libstiles_shared >/tmp/b_lib.log 2>&1 || { echo "LIB FAILED:"; tail -50 /tmp/b_lib.log; exit 1; }
echo "== build bench_stiles (CLI) =="
# rsync copies the host-native bench_stiles binary (no extension -> not caught by
# the *.o/*.so/*.a excludes); make would then see it up-to-date and skip it,
# shipping the host glibc/-march=native binary. Force a fresh portable build.
rm -f run/bench_stiles
make -C run -j$J bench_stiles                  >/tmp/b_run.log 2>&1 || { echo "RUN FAILED:"; tail -50 /tmp/b_run.log; exit 1; }

echo "== guard: glibc floor must not exceed the container glibc (catches host-lib leaks) =="
for art in run/bench_stiles lib/libstiles.so; do
  minor=$(objdump -T "$art" 2>/dev/null | grep -oE 'GLIBC_2\.[0-9]+' | sed 's/GLIBC_2\.//' | sort -n | tail -1)
  echo "   $art -> max GLIBC_2.${minor:-?}"
  if [ "${minor:-0}" -gt 28 ]; then
    echo "FATAL: $art needs GLIBC_2.$minor (> 2.28) — a host library leaked into the build."; exit 1
  fi
done

echo "== verify the produced binaries are CPU-portable (no AVX-512/AVX2 in sТiles' OWN code) =="
own_zmm=$(objdump -d /work/run/bench_stiles 2>/dev/null | grep -c 'zmm' || true)
echo "   bench_stiles zmm(AVX-512) insns: $own_zmm  (MKL is in the .so and runtime-gated; this is sТiles' own code)"

echo "== package the bundle =="
# gcc-toolset-14 is newer than AlmaLinux 8's base libstdc++ (3.4.25), which
# corrupts the heap at runtime. Ship a new-GLIBCXX + low-glibc libstdc++ instead.
# Mount such a dir (e.g. conda-forge libstdcxx-ng) at /opt/cxxrt:
#   udocker run -v <conda>/lib:/opt/cxxrt:ro ...
CXXRT=""
[ -e /opt/cxxrt/libstdc++.so.6 ] && CXXRT=/opt/cxxrt
[ -z "$CXXRT" ] && echo "WARN: no /opt/cxxrt libstdc++ mounted — bundle may crash on its base libstdc++"
ROOT=/work STILES_CXXRT_LIBDIR="$CXXRT" bash tools/package/make_bundle.sh

echo "== GLIBC floor of the produced artifacts (must be <= container glibc) =="
objdump -T /work/run/bench_stiles /work/lib/libstiles.so 2>/dev/null \
  | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sort -uV | tail -3
echo "== DONE =="
