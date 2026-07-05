#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build a relocatable, self-contained sТiles CLI bundle for redistribution.
#
# MKL (BLAS/LAPACK) and the ordering libs (METIS/SCOTCH/SuiteSparse) are already
# statically baked INTO libstiles.so, so the only remaining external libraries
# are a handful of standard system .so. This script gathers that shared-library
# closure (minus the host glibc, which must come from the target system), drops
# in a launcher that pins LD_LIBRARY_PATH to the bundle, adds a sample matrix +
# docs, and tars it.
#
# The result runs on any x86-64 Linux whose glibc >= this build host's glibc.
# (Build on an older-glibc machine/container to lower that floor.)
#
# Usage:  bash tools/package/make_bundle.sh
# Output: dist/stiles-linux-x86_64/  and  dist/stiles-linux-x86_64.tar.gz
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT=${ROOT:-/home/abdulfe/Documents/ideas/adv_sTiles}
LIB=$ROOT/lib/libstiles.so
BIN=$ROOT/run/bench_stiles
NAME=stiles-linux-x86_64
OUT=$ROOT/dist
B=$OUT/$NAME

[ -f "$LIB" ] || { echo "missing $LIB (run: make all)"; exit 1; }
[ -f "$BIN" ] || { echo "missing $BIN (run: make all / build run/)"; exit 1; }

rm -rf "$B"; mkdir -p "$B/bin" "$B/lib" "$B/examples"
cp -L "$LIB" "$B/lib/libstiles.so"
cp -L "$BIN" "$B/bin/stiles-bin"

# --- recursive shared-library closure of BOTH the binary and libstiles.so, ----
# --- excluding the host glibc core (never bundle libc/ld-linux: it must match  -
# --- the target kernel/loader). Iterate to a fixpoint so deps-of-deps (e.g.    -
# --- hwloc -> xml2) are caught.                                                 -
GLIBC_CORE='/(ld-linux-x86-64|libc|libm|libdl|libpthread|librt|libresolv)\.so'
collect() { ldd "$1" 2>/dev/null | awk '/=>/ && $3 ~ /^\//{print $3}'; }

deps=$OUT/_deps.txt; work=$OUT/_work.txt; new=$OUT/_new.txt; merged=$OUT/_merged.txt
: > "$deps"; { echo "$BIN"; echo "$LIB"; } > "$work"
for pass in 1 2 3 4 5; do
  while read -r f; do collect "$f"; done < "$work" | sort -u | grep -vE "$GLIBC_CORE" > "$new" || true
  cat "$deps" "$new" | sort -u > "$merged"
  cmp -s "$deps" "$merged" && break
  cp "$merged" "$deps"; cp "$new" "$work"
done

# copy each dep (deref symlink, keep SONAME basename) except libstiles.so itself
while read -r d; do
  case "$d" in */libstiles.so) continue;; esac
  cp -Lf "$d" "$B/lib/$(basename "$d")"
done < "$deps"
rm -f "$deps" "$work" "$new" "$merged"

# --- launcher: no external commands, no patchelf; LD_LIBRARY_PATH beats the ----
# --- binary's RUNPATH so the bundled libs win on any host. ---------------------
cat > "$B/stiles" <<'EOF'
#!/bin/sh
# sТiles launcher — pins the bundled libraries (libstiles.so with MKL + ordering
# baked in, plus the system .so closure). Works from any location.
HERE=$(cd -- "${0%/*}" 2>/dev/null && pwd -P) || HERE=.
export LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HERE/bin/stiles-bin" "$@"
EOF
chmod +x "$B/stiles"

# --- sample matrix (smallest local .mtx) so users can test immediately --------
SAMPLE=$(find "$ROOT/tests" -name '*.mtx' -printf '%s %p\n' 2>/dev/null | sort -n | awk 'NR==1{print $2}')
[ -n "${SAMPLE:-}" ] && cp -L "$SAMPLE" "$B/examples/$(basename "$SAMPLE")" || true
SNAME=$(basename "${SAMPLE:-sample.mtx}")

# --- docs -----------------------------------------------------------------
cat > "$B/README.txt" <<EOF
sТiles — tile-based sparse/dense direct Cholesky solver (Linux x86-64)
=====================================================================

Self-contained bundle. BLAS/LAPACK (Intel MKL) and the fill-reducing ordering
libraries (METIS, SCOTCH, SuiteSparse) are baked into libstiles.so — nothing to
install. The only requirement is a reasonably recent glibc (see REQUIREMENTS).

QUICK START
-----------
  ./stiles examples/$SNAME                       # factor + solve + log-determinant
  ./stiles examples/$SNAME --threads=8           # use 8 threads
  ./stiles examples/$SNAME --tile-mode=3         # 3 = auto (default); 0 dense / 1 semisparse / 2 sparse
  ./stiles your_matrix.mtx --threads=16 --csv    # machine-readable CSV row

Matrix input is symmetric positive-definite, Matrix Market (.mtx) format,
lower/upper triangle.

COMMON OPTIONS
--------------
  --threads=N        number of cores (default: all)
  --tile-mode=M      0 dense | 1 semisparse | 2 sparse | 3 auto (default)
  --tile-size=T      override tile size (default: auto, L3-cache driven)
  --fact-reps=R      repeat the factorization R times (timing)
  --csv              print one CSV row instead of the human report

REQUIREMENTS
------------
  - Linux x86-64
  - glibc >= $(ldd --version | awk 'NR==1{print $NF}')  (this is the build host's glibc; older hosts will not load it)
  Everything else (MKL, OpenMP, gfortran/stdc++ runtimes, hwloc, numa) ships in lib/.

LICENSE
-------
  See THIRD_PARTY.txt. Includes Intel oneMKL, redistributed under Intel's
  oneAPI redistribution terms.
EOF

cat > "$B/THIRD_PARTY.txt" <<'EOF'
This bundle statically/​dynamically includes third-party components:

  - Intel oneMKL (BLAS/LAPACK), statically linked into libstiles.so.
    Redistributed under the Intel Simplified Software License / oneAPI
    redistribution terms: https://www.intel.com/content/www/us/en/developer/articles/license/onemkl-license-faq.html
  - METIS (Apache-2.0), SCOTCH (CeCILL-C), SuiteSparse AMD/CAMD/COLAMD
    (BSD-3 / LGPL), compiled into libstiles.so.
  - GCC runtime libraries (libstdc++, libgcc_s, libgfortran, libgomp) under
    GCC Runtime Library Exception; hwloc (BSD), libnuma (LGPL-2.1),
    zlib/bzip2/xz (permissive), bundled in lib/.
EOF

# --- C++ runtime override (IMPORTANT for cross-distro) ------------------------
# When the build compiler (e.g. gcc-toolset-14) is newer than the base system
# libstdc++, the binaries are compiled against a newer GLIBCXX inline ABI. The
# base libstdc++ then corrupts the heap at runtime (malloc_consolidate) even
# though the *versioned* symbols it needs are old. Fix: ship a libstdc++ that
# has a NEW GLIBCXX (>= the compiler's) AND a LOW glibc floor. conda-forge's
# libstdcxx-ng is built exactly that way (glibc 2.17 baseline, latest GLIBCXX).
# Point STILES_CXXRT_LIBDIR at such a dir (e.g. <conda>/lib).
if [ -n "${STILES_CXXRT_LIBDIR:-}" ] && [ -e "$STILES_CXXRT_LIBDIR/libstdc++.so.6" ]; then
  cp -Lf "$STILES_CXXRT_LIBDIR/libstdc++.so.6" "$B/lib/libstdc++.so.6"
  [ -e "$STILES_CXXRT_LIBDIR/libgcc_s.so.1" ] && cp -Lf "$STILES_CXXRT_LIBDIR/libgcc_s.so.1" "$B/lib/libgcc_s.so.1"
  echo "C++ runtime overridden from $STILES_CXXRT_LIBDIR (GLIBCXX $(objdump -T "$B/lib/libstdc++.so.6" 2>/dev/null | grep -oE 'GLIBCXX_[0-9.]+' | sort -uV | tail -1))"
fi

# --- pack -----------------------------------------------------------------
( cd "$OUT" && tar czf "$NAME.tar.gz" "$NAME" )
echo "bundle:  $B"
echo "tarball: $OUT/$NAME.tar.gz"
echo "size:    $(du -sh "$B" | cut -f1)  ($(ls "$B/lib" | wc -l) libs in lib/)"
