###############################################################################
# sTiles Compute Module
###############################################################################

STILES_DIR := ../..
SRC := dpotrf.cpp pdpotrf.cpp dpotri.cpp dlogdet.cpp dtrsm.cpp pdtrsm.cpp pdtrtri.cpp \
       sparse_dpotrf.cpp sparse_dpotri.cpp sparse_dtrsm.cpp sparse_dlogdet.cpp \
       csc_solve.cpp
EXTRA_INC := -I../control

# Intel-runtime stub: libxsmm references _intel_fast_memset/_intel_fast_memcpy
# even when built with gcc; the stub aliases them to libc memset/memcpy.
# MKLMPI stub: libmkl_core.a references MKLMPI_Get_wrappers (MKL's MPI hook);
# when MKL is statically embedded via --whole-archive that reference must be
# satisfied. sTiles never uses MKL's distributed paths, so it asserts if called.
# ('=' on purpose: LIBXSMM_DEF/LAPACKE_DEF come from make.inc, parsed later.)
EXTRA_OBJ = $(if $(strip $(LIBXSMM_DEF)),stiles_xsmm_intel_stubs.o) \
            $(if $(findstring STILES_WITH_MKL,$(LAPACKE_DEF)),stiles_mkl_mpi_stub.o)

include $(STILES_DIR)/tools/module.mk
