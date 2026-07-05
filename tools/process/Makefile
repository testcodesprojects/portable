###############################################################################
# sTiles Process Module
###############################################################################

STILES_DIR := ../..
include $(STILES_DIR)/Makefile.internal

# Pin the default goal: inline-ordering blocks below define pattern rules, and we
# must never let one of those become make's default target (a bare `make` here,
# as invoked by the top-level libstiles recipe, must build `all`).
.DEFAULT_GOAL := all

# Additional include paths (METIS/SCOTCH/SuiteSparse includes come from make.inc,
# which orders METIS_INC before SCOTCH_INC).
INC += -I. -I$(STILES_DIR)/lib

# Sources (ordering modules + main process)
SRC := ../ordering/ordering.cpp \
       ../ordering/ordering_overrides.cpp \
       ../ordering/ordering_rcm.cpp \
       ../ordering/ordering_rcm_scipy.cpp \
       ../ordering/ordering_nd.cpp \
       ../ordering/ordering_scotch.cpp \
       ../ordering/ordering_suitesparse.cpp \
       ../ordering/ordering_mixed.cpp \
       ../ordering/ordering_amd.cpp \
       ../ordering/ordering_amdbar.cpp \
       ../ordering/ordering_amdbarnew.cpp \
       ../ordering/ordering_amdnew.cpp \
       ../ordering/ordering_genmmd.cpp \
       ../ordering/ordering_gps.cpp \
       ../ordering/ordering_tile.cpp \
       ../common/stiles_params.cpp \
       process.cpp

OBJ := $(SRC:.cpp=.o)

# Fortran/C sources from external/ needed by the ordering wrappers
EXTERN_DIR := ../ordering/external
# Output objects to ../ordering/ so the top-level find -maxdepth 1 picks them up.
EXTERN_OBJS := ../ordering/extern_amdbarnew.o \
               ../ordering/extern_amdnew.o \
               ../ordering/extern_amdpre.o \
               ../ordering/extern_genmmd.o \
               ../ordering/extern_acm582.o

# ----------------------------------------------------------------------------
# SuiteSparse: compile the AMD/CAMD/COLAMD/CCOLAMD/BTF/SuiteSparse_config sources
# straight into ../ordering/ss_*.o so the top-level archiver folds them into
# libstiles.a (no cmake sub-build, no static archives). The compiled .c set
# matches the members of libamd.a/.../libsuitesparseconfig.a exactly; amd.f and
# amdbar.f supply the amd_/amdbar_ Fortran entry points that ordering_amd.cpp /
# ordering_amdbar.cpp call.
# ----------------------------------------------------------------------------
ifeq ($(USE_SUITESPARSE),1)
SS_ROOT  := $(SUITESPARSE_SRC_ROOT)
SS_C_SRC := $(SS_ROOT)/SuiteSparse_config/SuiteSparse_config.c \
            $(wildcard $(SS_ROOT)/AMD/Source/*.c) \
            $(wildcard $(SS_ROOT)/CAMD/Source/*.c) \
            $(wildcard $(SS_ROOT)/COLAMD/Source/*.c) \
            $(wildcard $(SS_ROOT)/CCOLAMD/Source/*.c) \
            $(wildcard $(SS_ROOT)/BTF/Source/*.c)
SS_C_OBJS := $(addprefix ../ordering/ss_,$(notdir $(SS_C_SRC:.c=.o)))
SS_F_OBJS := ../ordering/ss_amd.o ../ordering/ss_amdbar.o
EXTERN_OBJS += $(SS_C_OBJS) $(SS_F_OBJS)

# Same arch/opt/LTO as every other sTiles object; AMD/COLAMD are
# integer-deterministic so opt level cannot change the ordering.
# OPENMP_INC is needed on macOS: SuiteSparse_config.h pulls in <omp.h> under
# -fopenmp, and clang's -Xpreprocessor -fopenmp does NOT add the libomp include
# path (Homebrew /opt/homebrew/opt/libomp/include). Empty on Linux.
SS_CFLAGS := $(OPT_FLAGS) -fPIC -std=gnu11 $(OPENMP_FLAGS) $(OPENMP_INC) $(SUITESPARSE_INC)

vpath %.c $(SS_ROOT)/SuiteSparse_config $(SS_ROOT)/AMD/Source \
          $(SS_ROOT)/CAMD/Source $(SS_ROOT)/COLAMD/Source \
          $(SS_ROOT)/CCOLAMD/Source $(SS_ROOT)/BTF/Source
endif

# ----------------------------------------------------------------------------
# METIS + GKlib: compile libmetis/*.c and GKlib/src/*.c into ../ordering/mt_*.o /
# gk_*.o (archived into libstiles.a) — no cmake/make-config sub-build. Skipped
# only when a system METIS is used (METIS_SYS_PREFIX set; then -lmetis is linked).
# Static pattern rules (not vpath) because GKlib and libmetis both ship a
# graph.c; the gk_/mt_ prefixes + explicit source dirs keep them distinct.
# Unlike AMD/COLAMD, METIS uses floating point, so we strip -flto/-funroll-loops
# to match the cmake build's codegen profile (-O3 -march=native) and keep the
# ordering bit-identical; defines mirror the cmake flags.make exactly.
# ----------------------------------------------------------------------------
ifeq ($(strip $(METIS_SYS_PREFIX)),)
GK_ROOT := $(GKLIB_DIR)/src
MT_ROOT := $(METIS_ABS_DIR)/libmetis
GK_SRC  := $(wildcard $(GK_ROOT)/*.c)
MT_SRC  := $(wildcard $(MT_ROOT)/*.c)
GK_OBJS := $(patsubst $(GK_ROOT)/%.c,../ordering/gk_%.o,$(GK_SRC))
MT_OBJS := $(patsubst $(MT_ROOT)/%.c,../ordering/mt_%.o,$(MT_SRC))
EXTERN_OBJS += $(GK_OBJS) $(MT_OBJS)

# Match cmake's codegen (no LTO/unroll) but keep sTiles' arch (-march via OPT_FLAGS).
METIS_INLINE_OPT := $(filter-out -flto% -funroll-loops,$(OPT_FLAGS))
# OS defines: mirror what GKlib/METIS cmake sets per platform (-DLINUX on
# Linux, -DMACOS on Apple; _POSIX_C_SOURCE is Linux-only there too). The
# Linux flag strings below stay byte-identical to the validated build.
ifeq ($(PLATFORM),macos)
    GKMT_OS_DEF  := -DMACOS
    GK_POSIX_DEF :=
else
    GKMT_OS_DEF  := -DLINUX
    GK_POSIX_DEF := -D_POSIX_C_SOURCE=200809L
endif
GK_INC := $(GKLIB_DIR)/include
GK_CFLAGS := $(METIS_INLINE_OPT) -fPIC -fno-strict-aliasing -I$(GK_INC) \
             -DHAVE_EXEC_INFO_H -DHAVE_GETLINE -DHAVE_REGEX_H $(GKMT_OS_DEF) -DNDEBUG2 \
             $(GK_POSIX_DEF) -D__thread=__thread
MT_CFLAGS := $(METIS_INLINE_OPT) -fPIC -fno-strict-aliasing -std=c99 \
             -I$(MT_ROOT) -I$(GK_INC) -I$(METIS_ABS_DIR)/include \
             $(METIS_DEF) $(GKMT_OS_DEF) -D_FILE_OFFSET_BITS=64 -DNDEBUG2 \
             -DHAVE_EXECINFO_H -DHAVE_GETLINE
# NOTE: the gk_/mt_ compile RULES live after the `all:` target (below) so they
# do not become make's default goal — keep it that way.
endif

# ----------------------------------------------------------------------------
# SCOTCH: compile the 226 libscotch objects (cmake ground-truth list in
# scotch_inline/scotch_src_bases.txt) + library_error (libscotcherr) into
# ../ordering/sco_*.o. parser_ll.c/parser_yy.c are the vendored flex/bison
# output; all other sources come from src/libscotch. Flags mirror SCOTCH's cmake
# flags.make EXACTLY (no -march/-flto) so the ordering is bit-identical. SCOTCH
# still needs -lz -lbz2 -llzma at link (set in make.inc).
# Compile RULES live after `all:` (below) so they never become the default goal.
# ----------------------------------------------------------------------------
SCO_SRC_DIR  := $(SCOTCH_SRC_ROOT)/src/libscotch
SCO_GEN_DIR  := $(SCOTCH_INLINE_DIR)
SCO_SRC_BASES := $(shell cat $(SCOTCH_INLINE_DIR)/scotch_src_bases.txt 2>/dev/null)
SCO_SRC_OBJS := $(addprefix ../ordering/sco_,$(addsuffix .o,$(SCO_SRC_BASES)))
SCO_GEN_OBJS := ../ordering/sco_parser_ll.o ../ordering/sco_parser_yy.o
EXTERN_OBJS  += $(SCO_SRC_OBJS) $(SCO_GEN_OBJS)

# Thread affinity uses Linux-only cpu_set_t/CPU_SET; must be off on macOS.
# macOS also lacks _POSIX_TIMERS, so SCOTCH's clockGet() falls back to the
# gettimeofday() branch, which needs <sys/time.h>/<sys/resource.h> — SCOTCH
# only includes those when HAVE_SYS_TIME_H/HAVE_SYS_RESOURCE_H are set (its
# configure would probe them). Linux takes the clock_gettime path, so these
# are a no-op there; keep them macOS-only to leave the Linux build untouched.
ifeq ($(PLATFORM),linux)
    SCO_AFFINITY_DEF := -DCOMMON_PTHREAD_AFFINITY_LINUX
    SCO_TIME_DEF     :=
else
    SCO_AFFINITY_DEF :=
    SCO_TIME_DEF     := -DHAVE_SYS_TIME_H -DHAVE_SYS_RESOURCE_H
endif
SCO_DEFINES := -DCOMMON_FILE_COMPRESS_BZ2 -DCOMMON_FILE_COMPRESS_GZ -DCOMMON_FILE_COMPRESS_LZMA \
               -DCOMMON_PTHREAD $(SCO_AFFINITY_DEF) $(SCO_TIME_DEF) -DCOMMON_RANDOM_FIXED_SEED \
               -DSCOTCH_DETERMINISTIC -DSCOTCH_PATCHLEVEL_NUM=10 -DSCOTCH_PTHREAD \
               -DSCOTCH_RELEASE_NUM=0 -DSCOTCH_RENAME -DSCOTCH_VERSION_NUM=7 -Drestrict=__restrict
SCO_CFLAGS := -U_FORTIFY_SOURCE -O3 -fPIC -DNDEBUG -std=gnu99 $(SCO_DEFINES) \
              -I$(SCO_SRC_DIR) -I$(SCO_GEN_DIR)

# GPU support (gpu_utils.cu moved to old_house, no longer needed here)

# Target
all: $(OBJ) $(EXTERN_OBJS)

# Header dependency files for sTiles' own C++ objects (third-party C sources
# below change rarely and keep their validated one-shot compile lines).
DEP := $(OBJ:.o=.d)

# Compilation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -MMD -MP -c $< -o $@

# Inline SuiteSparse C sources (resolved via vpath above)
../ordering/ss_%.o: %.c
	$(CC) $(SS_CFLAGS) -c $< -o $@

# Inline SuiteSparse Fortran (amd_/amdbar_ entry points)
../ordering/ss_amd.o: $(SUITESPARSE_SRC_ROOT)/AMD/Source/amd.f
	$(FC) $(FFLAGS) -c $< -o $@
../ordering/ss_amdbar.o: $(SUITESPARSE_SRC_ROOT)/AMD/Source/amdbar.f
	$(FC) $(FFLAGS) -c $< -o $@

# METIS + GKlib compile rules (static patterns; kept AFTER `all:` so they never
# become the default goal). gk_/mt_ prefixes disambiguate the two graph.c files.
ifeq ($(strip $(METIS_SYS_PREFIX)),)
$(GK_OBJS): ../ordering/gk_%.o: $(GK_ROOT)/%.c
	$(CC) $(GK_CFLAGS) -c $< -o $@
$(MT_OBJS): ../ordering/mt_%.o: $(MT_ROOT)/%.c
	$(CC) $(MT_CFLAGS) -c $< -o $@
endif

# SCOTCH compile rules (static patterns; kept AFTER `all:` so they never become
# the default goal). src/libscotch sources + the two vendored generated parsers.
$(SCO_SRC_OBJS): ../ordering/sco_%.o: $(SCO_SRC_DIR)/%.c
	$(CC) $(SCO_CFLAGS) -c $< -o $@
$(SCO_GEN_OBJS): ../ordering/sco_%.o: $(SCO_GEN_DIR)/%.c
	$(CC) $(SCO_CFLAGS) -c $< -o $@

# External Fortran/C sources → objects in ../ordering/ for archiver pickup
# ($(FC) from make.inc, version-matched to CXX; FFLAGS = -O2 -fPIC, no
# -march/-flto so orderings stay stable across builds)
../ordering/extern_amdbarnew.o:    $(EXTERN_DIR)/amdbarNEW.f
	$(FC) $(FFLAGS) -c $< -o $@
../ordering/extern_amdnew.o:       $(EXTERN_DIR)/amdNEW.f
	$(FC) $(FFLAGS) -c $< -o $@
../ordering/extern_amdpre.o:       $(EXTERN_DIR)/amdpre.f
	$(FC) $(FFLAGS) -c $< -o $@
../ordering/extern_genmmd.o:       $(EXTERN_DIR)/genmmd.f
	$(FC) $(FFLAGS) -c $< -o $@
../ordering/extern_acm582.o:       $(EXTERN_DIR)/acm582.F
	$(FC) $(FFLAGS) -c $< -o $@

# Clean (also ../common/stiles_params.o, compiled by this module)
clean:
	rm -f *.o *.d ../ordering/*.o ../ordering/*.d ../common/*.o ../common/*.d *~

cleanall: clean

-include $(DEP)

.PHONY: all clean cleanall
