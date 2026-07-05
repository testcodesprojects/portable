#!/usr/bin/env bash
###############################################################################
# sTiles speed + correctness smoke test.
#
#   ./run.sh                 # build, then run every matrix in mats/
#   CORES=8 TILE=40 ./run.sh # override thread count / tile size
#   ./run.sh path/to/other/*.mtx   # run a custom set of matrices
#
# Prints one row per matrix (phase timings + solve residual) and a PASS/FAIL
# verdict. Exits non-zero if ANY matrix fails, so CI can gate on it.
###############################################################################
set -u
cd "$(dirname "$0")"

CORES="${CORES:-4}"
TILE="${TILE:-40}"

# macOS gives OMP worker threads a small stack (the OMP backend is the macOS
# default), which can overflow in the numeric factorization of large/high-fill
# matrices. A big OMP stack is harmless on Linux (pthreads backend ignores it).
export OMP_STACKSIZE="${OMP_STACKSIZE:-256M}"

echo "Building speedtest..."
if ! make -s speedtest; then
    echo "BUILD FAILED — is the library built? (run 'make all' in the repo root)" >&2
    exit 1
fi

# Matrices: CLI args if given, else everything in mats/.
if [ "$#" -gt 0 ]; then
    mats=("$@")
else
    mats=(mats/*.mtx)
fi
if [ ! -e "${mats[0]}" ]; then
    echo "No matrices found (looked in mats/). Put some .mtx files there." >&2
    exit 1
fi

echo ""
echo "sTiles speed test   cores=$CORES  tile=$TILE   (solve at rhs=1 and rhs=$((TILE + 1)))"
printf '%-14s %8s %10s %6s %8s %8s %8s %8s %15s %10s %10s  %s\n' \
       matrix n nnz fill "chol(s)" "selinv" "slv1(s)" "slvT(s)" logdet "res(1)" "res(T+1)" status
printf '%.0s-' $(seq 1 122); echo

fail=0
# Explicit template: bare `mktemp` errors on macOS/BSD (needs one).
errlog="$(mktemp "${TMPDIR:-/tmp}/stiles_smoke.XXXXXX")"
for mtx in "${mats[@]}"; do
    [ -e "$mtx" ] || continue
    # Row (stdout) always shown; library stderr hidden unless the matrix fails.
    if ! ./speedtest "$mtx" "$CORES" "$TILE" 2>"$errlog"; then
        fail=$((fail + 1))
        echo "  --- stderr for $(basename "$mtx") ---"
        sed 's/^/  /' "$errlog"
        # On a crash (segfault/abort with no useful message), re-run under a
        # debugger to capture where it died — invaluable for platform-specific
        # crashes that can't be reproduced on the dev machine.
        if command -v lldb >/dev/null 2>&1; then
            echo "  --- lldb backtrace for $(basename "$mtx") ---"
            lldb --batch -o run -o "thread backtrace all" -o quit \
                 -- ./speedtest "$mtx" "$CORES" "$TILE" 2>&1 | sed 's/^/  /' | tail -80
        elif command -v gdb >/dev/null 2>&1; then
            echo "  --- gdb backtrace for $(basename "$mtx") ---"
            gdb -batch -ex run -ex "thread apply all bt" \
                --args ./speedtest "$mtx" "$CORES" "$TILE" 2>&1 | sed 's/^/  /' | tail -80
        fi
    fi
done
rm -f "$errlog"

echo ""
if [ "$fail" -eq 0 ]; then
    echo "All matrices PASSED."
else
    echo "$fail matrix/matrices FAILED."
fi
exit "$fail"
