###############################################################################
# sTiles speed + correctness smoke test
#
#   make            # build ./speedtest against ../lib/libstiles.so
#   ./run.sh        # run it over every matrix in mats/ (table + PASS/FAIL)
#
# Reuses the repo's shared platform/BLAS/OpenMP detection (../bench.inc), so it
# links correctly on Linux (MKL/OpenBLAS) and macOS (Accelerate) with no edits.
###############################################################################

# Repo root is one level up (bench/ sits directly under it).
STILES_PATH ?= $(shell cd .. && pwd)
include $(STILES_PATH)/bench.inc

# -g -fno-omit-frame-pointer: run.sh re-runs a crashed speedtest under
# lldb/gdb; without frame pointers that backtrace is just "PC=0" (as in the
# macos-intel dual-OpenMP crash). Negligible cost for a bench binary.
CXXFLAGS = -std=c++20 -O2 -g -fno-omit-frame-pointer -Wall -Wextra $(OMP_CXXFLAGS)

all: speedtest parallel_chol

speedtest: speedtest.cpp $(STILES_PATH)/lib/libstiles.so
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $< $(LDFLAGS)

# Demo: N concurrent Cholesky factorizations, each on its own cores.
#   ./parallel_chol mats/ferris.mtx 4 2
parallel_chol: parallel_chol.cpp $(STILES_PATH)/lib/libstiles.so
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $< $(LDFLAGS)

# macOS produces libstiles.dylib; the .so symlink also exists, but depend on the
# real file so a stale binary rebuilds after the library changes.
$(STILES_PATH)/lib/libstiles.so:
	@echo "libstiles not built yet — run 'make all' in the repo root first." >&2
	@exit 1

clean:
	rm -f speedtest parallel_chol

.PHONY: all clean
