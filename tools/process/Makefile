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
    GKMT_OS_DEF    := -DMACOS
    GK_POSIX_DEF   :=
    GK_FEATURE_DEF := -DHAVE_EXEC_INFO_H -DHAVE_GETLINE -DHAVE_REGEX_H
    MT_FEATURE_DEF := -DHAVE_EXECINFO_H -DHAVE_GETLINE
else ifeq ($(PLATFORM),windows)
    # MinGW: no regex.h / execinfo.h and no _POSIX_C_SOURCE. GKlib detects
    # Windows via WIN32 (gk_arch.h). Do NOT assert the HAVE_* features (so
    # GKlib.h uses its bundled gkregex.h header), and DO define USE_GKREGEX so
    # gkregex.c actually COMPILES its regcomp/regexec/regfree/regerror bodies
    # (they're the whole file behind #ifdef USE_GKREGEX; Windows has no libc
    # regex to fall back to, unlike Linux/macOS).
    GKMT_OS_DEF    := -DWIN32
    GK_POSIX_DEF   :=
    GK_FEATURE_DEF := -DUSE_GKREGEX
    MT_FEATURE_DEF :=
else
    GKMT_OS_DEF    := -DLINUX
    GK_POSIX_DEF   := -D_POSIX_C_SOURCE=200809L
    GK_FEATURE_DEF := -DHAVE_EXEC_INFO_H -DHAVE_GETLINE -DHAVE_REGEX_H
    MT_FEATURE_DEF := -DHAVE_EXECINFO_H -DHAVE_GETLINE
endif
GK_INC := $(GKLIB_DIR)/include
# gcc 16 (MinGW UCRT) defaults to C23, where bool/true/false are keywords and
# gkregex.c's old-style `enum {false,true} bool` is illegal. Pin GKlib to gnu11
# on Windows (pre-C23, still has C11 features). Linux/macOS keep their default.
GK_STD := $(if $(filter windows,$(PLATFORM)),-std=gnu11,)
GK_CFLAGS := $(METIS_INLINE_OPT) -fPIC -fno-strict-aliasing -I$(GK_INC) $(GK_STD) \
             $(GK_FEATURE_DEF) $(GKMT_OS_DEF) -DNDEBUG2 \
             $(GK_POSIX_DEF) -D__thread=__thread
MT_CFLAGS := $(METIS_INLINE_OPT) -fPIC -fno-strict-aliasing -std=c99 \
             -I$(MT_ROOT) -I$(GK_INC) -I$(METIS_ABS_DIR)/include \
             $(METIS_DEF) $(GKMT_OS_DEF) -D_FILE_OFFSET_BITS=64 -DNDEBUG2 \
             $(MT_FEATURE_DEF)
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
# Compressed graph-FILE support (gz/bz2/lzma) is UNUSED by sTiles — it feeds
# SCOTCH in-memory graphs via the API, never reads .grf files. Kept on Linux to
# preserve the bit-identical ordering build; dropped on macOS, where lzma.h is
# not in the SDK (and it drops the -lz/-lbz2/-llzma link deps there too).
#
# SCO_OS_DEF: SCOTCH's own platform flag. On macOS, -DCOMMON_OS_MACOS makes
# common.h define _DARWIN_C_SOURCE + HAVE_SYS_SYSCTL_H + COMMON_TIMING_OLD, so
# SCOTCH uses the sysctl() core-count path and getrusage() timing (both macOS-
# correct) instead of the Linux sysconf()/gettimeofday() paths. Linux passes
# no OS flag (its default branch is the Linux one).
ifeq ($(PLATFORM),linux)
    SCO_AFFINITY_DEF := -DCOMMON_PTHREAD_AFFINITY_LINUX
    SCO_OS_DEF       :=
    # Compressed graph-FILE support is unused by sTiles (see make.inc note); the
    # portable artifact builds with STILES_SCOTCH_COMPRESS=0 to drop it and its
    # -lz/-lbz2/-llzma deps. Default ON preserves the bit-identical Linux build.
    STILES_SCOTCH_COMPRESS ?= 1
    ifeq ($(STILES_SCOTCH_COMPRESS),1)
        SCO_COMPRESS_DEF := -DCOMMON_FILE_COMPRESS_BZ2 -DCOMMON_FILE_COMPRESS_GZ -DCOMMON_FILE_COMPRESS_LZMA
    else
        SCO_COMPRESS_DEF :=
    endif
else ifeq ($(PLATFORM),windows)
    # -DCOMMON_OS_WINDOWS makes SCOTCH's common.h take its Windows path
    # (io.h/windows.h, skip unistd.h, no sys/resource.h|sys/time.h — those are
    # gated on macOS's COMMON_TIMING_OLD, which must NOT be set here). No
    # compression libs (parity with SCOTCH_EXTRA_LIBS being Linux-only).
    SCO_AFFINITY_DEF :=
    SCO_OS_DEF       := -DCOMMON_OS_WINDOWS
    SCO_COMPRESS_DEF :=
else
    SCO_AFFINITY_DEF :=
    SCO_OS_DEF       := -DCOMMON_OS_MACOS
    SCO_COMPRESS_DEF :=
endif
SCO_DEFINES := $(SCO_COMPRESS_DEF) $(SCO_OS_DEF) \
               -DCOMMON_PTHREAD $(SCO_AFFINITY_DEF) -DCOMMON_RANDOM_FIXED_SEED \
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

# syntax : compile-check this module's sTiles-own SRC (ordering wrappers +
# process.cpp + stiles_params.cpp) WITHOUT codegen/link — for the top-level
# `cross-syntax` Windows/MinGW probe. Vendored SuiteSparse/Fortran excluded.
SYNTAX_ERR ?= /dev/null
syntax:
	@for s in $(SRC); do \
	    if $(CXX) $(CXXFLAGS) $(INC) -fsyntax-only $$s 2>>$(SYNTAX_ERR); then \
	        echo "  ok    $$s"; \
	    else echo "  FAIL  $$s"; fi; \
	done

-include $(DEP)

.PHONY: all clean cleanall syntax
