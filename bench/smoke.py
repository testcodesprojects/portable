#!/usr/bin/env python3
# Windows smoke test: load libstiles.dll via ctypes (LoadLibrary — no
# compile-time import, so no MinGW auto-import/pseudo-reloc can ever happen)
# and factor a real matrix. This is exactly how pysTiles uses the DLL.
#
#   python smoke.py <matrix.mtx>
#
# The DLL and all its runtime deps (openblas/gomp/hwloc/winpthread/gfortran)
# must sit in the current directory (the CI step copies them there) so
# os.add_dll_directory(cwd) resolves everything.
import ctypes as C
import math
import os
import sys

here = os.getcwd()
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(here)              # let libstiles.dll find its deps here
lib = C.CDLL(os.path.join(here, "libstiles.dll"))

lib.sTiles_get_version.restype = C.c_char_p
lib.sTiles_get_logdet.restype = C.c_double
print("loaded libstiles.dll — sTiles version:", lib.sTiles_get_version().decode())

# ---- parse Matrix Market (symmetric coordinate real) -> lower COO 0-based ----
path = sys.argv[1] if len(sys.argv) > 1 else "mats/ferris.mtx"
n = 0
ent = []
with open(path) as f:
    got_dims = False
    for line in f:
        if not line or line[0] == "%":
            continue
        p = line.split()
        if not got_dims:
            n = int(p[0]); got_dims = True; continue
        i = int(p[0]) - 1; j = int(p[1]) - 1; v = float(p[2])
        if i < j:
            i, j = j, i                     # keep lower triangle
        ent.append((i, j, v))
ent.sort(key=lambda t: (t[0], t[1]))        # (row, col) order
nnz = len(ent)
Row = (C.c_int * nnz)(*[t[0] for t in ent])
Col = (C.c_int * nnz)(*[t[1] for t in ent])
Val = (C.c_double * nnz)(*[t[2] for t in ent])
print("matrix:", path, "n =", n, "nnz(lower) =", nnz)

# ---- factor it (mirrors the validated pysTiles call sequence) ----
h = C.c_void_p(0)
calls = (C.c_int * 1)(1)
cores = (C.c_int * 1)(2)
ctype = (C.c_int * 1)(0)
getinv = (C.c_bool * 1)(False)
lib.sTiles_set_log_level(-1)
lib.sTiles_expert_user()
lib.sTiles_set_tile_size(40)
lib.sTiles_set_tile_type_mode(3)            # auto
if lib.sTiles_create(C.byref(h), 1, calls, cores, ctype, getinv) != 0:
    sys.exit("sTiles_create failed")
if lib.sTiles_assign_graph_one_call(0, 0, C.byref(h), n, nnz, Row, Col) != 0:
    sys.exit("sTiles_assign_graph_one_call failed")
if lib.sTiles_init_group(0, C.byref(h)) != 0:
    sys.exit("sTiles_init_group failed")
lib.sTiles_assign_values(0, 0, C.byref(h), Val)
lib.sTiles_bind(0, 0, C.byref(h))
rc = lib.sTiles_chol(0, 0, C.byref(h))
lib.sTiles_unbind(0, 0, C.byref(h))
if rc != 0:
    sys.exit("sTiles_chol failed")
ld = lib.sTiles_get_logdet(0, 0, C.byref(h))
lib.sTiles_freeGroup(0)
print("logdet =", ld)
if not (math.isfinite(ld) and ld > 0):
    sys.exit("logdet not finite/positive")
print("OK: libstiles.dll loads and factors on Windows")
