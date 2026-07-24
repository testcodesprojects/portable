###############################################################################
# Shared platform / compiler / BLAS / OpenMP detection for the benchmark
# Makefiles (run/Makefile, tests/run/Makefile).
#
# Include AFTER setting STILES_PATH (path to the sTiles project root).
# Provides:
#   PLATFORM            linux | macos
#   CXX                 g++ (Linux) or Homebrew g++ / clang++ (macOS)
#   BENCH_NATIVE_FLAGS  native-arch + LTO flags valid for the chosen compiler
#   OMP_CXXFLAGS        OpenMP compile flags (add to CXXFLAGS)
#   OMP_LIBS            OpenMP link libs
#   BLAS_LIBS           auto-detected BLAS link line (MKL > OpenBLAS > generic)
#   OPTIONAL_LIBS       numa / hwloc when present
#   INC, LIBPATH, LIBS  sTiles include/lib paths
#   LDFLAGS             full link line (recursive: honors LDFLAGS_EXTRA set
#                       later, e.g. by a sanitize build)
###############################################################################

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
else
    PLATFORM := linux
endif

###############################################################################
# Compiler
###############################################################################
ifeq ($(PLATFORM),macos)
    HOMEBREW_GCC := $(shell which g++-15 2>/dev/null || which g++-14 2>/dev/null || which g++-13 2>/dev/null || which g++-12 2>/dev/null)
    ifneq ($(HOMEBREW_GCC),)
        CXX = $(HOMEBREW_GCC)
    else
        CXX = clang++
    endif
else
    CXX = g++
endif

# Native-arch + LTO spellings the selected compiler accepts (mirrors make.inc):
# clang rejects -flto=auto (use thin LTO), arm64 wants -mcpu=native.
BENCH_IS_CLANG := $(findstring clang,$(shell $(CXX) --version 2>/dev/null))
ifneq ($(filter arm64 aarch64,$(UNAME_M)),)
    BENCH_NATIVE_ARCH := -mcpu=native
else
    BENCH_NATIVE_ARCH := -march=native
endif
BENCH_LTO_FLAG := $(if $(BENCH_IS_CLANG),-flto=thin,-flto=auto)
BENCH_NATIVE_FLAGS := $(BENCH_NATIVE_ARCH) $(BENCH_LTO_FLAG)

###############################################################################
# OpenMP
###############################################################################
ifeq ($(PLATFORM),macos)
    ifneq ($(BENCH_IS_CLANG),)
        LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
        ifneq ($(LIBOMP_PREFIX),)
            OMP_CXXFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
            OMP_LIBS := -L$(LIBOMP_PREFIX)/lib -lomp
        endif
    else
        OMP_CXXFLAGS := -fopenmp
        OMP_LIBS := -lgomp
    endif
else
    OMP_CXXFLAGS := -fopenmp
    OMP_LIBS := -lgomp
endif

###############################################################################
# Include and Library Paths
###############################################################################
INC = -I$(STILES_PATH)/tools/include
LIBPATH = -L$(STILES_PATH)/lib -Wl,-rpath,$(STILES_PATH)/lib
LIBS = -lstiles

###############################################################################
# BLAS Auto-Detection
###############################################################################
ifeq ($(PLATFORM),macos)
    MKL_AVAILABLE := $(shell test -d "/opt/intel/oneapi/mkl" && echo 1 || echo 0)
    ifeq ($(MKL_AVAILABLE),1)
        BLAS_LIBS = -L/opt/intel/oneapi/mkl/latest/lib -lmkl_intel_lp64 -lmkl_core -lmkl_sequential
    else ifneq ($(wildcard /opt/homebrew/opt/openblas/lib/libopenblas.dylib),)
        # Apple Silicon Homebrew OpenBLAS — match the library's BLAS (the lib is
        # built against OpenBLAS since Accelerate lacks LAPACKE).
        BLAS_LIBS = -L/opt/homebrew/opt/openblas/lib -L/opt/homebrew/opt/lapack/lib -lopenblas -llapacke
    else ifneq ($(wildcard /usr/local/opt/openblas/lib/libopenblas.dylib),)
        # Intel-mac Homebrew OpenBLAS
        BLAS_LIBS = -L/usr/local/opt/openblas/lib -L/usr/local/opt/lapack/lib -lopenblas -llapacke
    else
        BLAS_LIBS = -framework Accelerate
    endif
else ifneq ($(wildcard $(OPENBLAS_STATIC_ROOT)/lib/libopenblas.a),)
    # Same knob as make.inc: the portable arm64 lane builds a pinned static
    # OpenBLAS from source (no system BLAS exists in the manylinux container,
    # so the -lblas/-llapack fallback below would fail the bench link).
    BLAS_LIBS = $(OPENBLAS_STATIC_ROOT)/lib/libopenblas.a
else
    # MKLROOT is set by Intel oneAPI's setvars.sh; otherwise probe a couple of
    # well-known install locations. We bake -L and -Wl,-rpath into the binary
    # so it can find libmkl_*.so at runtime without needing LD_LIBRARY_PATH.
    MKL_ROOT := $(or $(MKLROOT),$(wildcard /opt/intel/oneapi/mkl/latest),$(wildcard /opt/intel/oneapi/mkl/2026.0))
    ifneq ($(strip $(MKL_ROOT)),)
        ifneq ($(wildcard $(MKL_ROOT)/lib/libmkl_core.so),)
            MKL_LIBDIR := $(MKL_ROOT)/lib
        else ifneq ($(wildcard $(MKL_ROOT)/lib/intel64/libmkl_core.so),)
            MKL_LIBDIR := $(MKL_ROOT)/lib/intel64
        endif
    endif
    ifneq ($(strip $(MKL_LIBDIR)),)
        BLAS_LIBS = -L$(MKL_LIBDIR) -Wl,-rpath,$(MKL_LIBDIR) -lmkl_gf_lp64 -lmkl_core -lmkl_sequential
    else
        MKL_AVAILABLE := $(shell ldconfig -p 2>/dev/null | grep -q libmkl_core && echo 1 || echo 0)
        ifeq ($(MKL_AVAILABLE),1)
            BLAS_LIBS = -lmkl_gf_lp64 -lmkl_core -lmkl_sequential
        else
            OPENBLAS_AVAILABLE := $(shell ldconfig -p 2>/dev/null | grep -q libopenblas && echo 1 || echo 0)
            ifeq ($(OPENBLAS_AVAILABLE),1)
                BLAS_LIBS = -lopenblas
            else
                BLAS_LIBS = -lblas -llapack
            endif
        endif
    endif
endif

###############################################################################
# Optional Libraries
###############################################################################
OPTIONAL_LIBS =

ifeq ($(PLATFORM),linux)
    NUMA_AVAILABLE := $(shell ldconfig -p 2>/dev/null | grep -q libnuma && echo 1 || echo 0)
    ifeq ($(NUMA_AVAILABLE),1)
        OPTIONAL_LIBS += -lnuma
    endif
endif

HWLOC_LIBS := $(shell pkg-config --libs hwloc 2>/dev/null)
ifneq ($(HWLOC_LIBS),)
    OPTIONAL_LIBS += $(HWLOC_LIBS)
endif

###############################################################################
# gfortran runtime (needed by the embedded ordering Fortran objects). On macOS
# clang++ has no gfortran libdir in its search path; locate it via gfortran.
###############################################################################
GFORTRAN_BENCH_LINK := -lgfortran
ifeq ($(PLATFORM),macos)
    GFORTRAN_BENCH_DYLIB := $(shell gfortran -print-file-name=libgfortran.dylib 2>/dev/null)
    GFORTRAN_BENCH_LIBDIR := $(dir $(GFORTRAN_BENCH_DYLIB))
    ifneq ($(strip $(GFORTRAN_BENCH_DYLIB)),)
    ifneq ($(GFORTRAN_BENCH_LIBDIR),./)
        GFORTRAN_BENCH_LINK := -L$(GFORTRAN_BENCH_LIBDIR) -Wl,-rpath,$(GFORTRAN_BENCH_LIBDIR) -lgfortran
    endif
    endif
endif

###############################################################################
# Link Flags
###############################################################################
# Note: SCOTCH and SuiteSparse symbols are embedded in libstiles via
# --whole-archive; SCOTCH's compression libs are only needed on Linux (macOS
# builds SCOTCH without gz/bz2/lzma file support — see tools/process/Makefile).
# Recursive on purpose: LDFLAGS_EXTRA may be appended after this include.
# Gated by STILES_SCOTCH_COMPRESS (same knob as the library build): the portable
# artifact builds the .so with =0, so speedtest must not pull -lbz2/-llzma either
# (they are absent in the manylinux container and unused by the self-contained .so).
STILES_SCOTCH_COMPRESS ?= 1
ifeq ($(PLATFORM)$(STILES_SCOTCH_COMPRESS),linux1)
    SCOTCH_COMPRESS_LIBS := -lz -lbz2 -llzma
else
    SCOTCH_COMPRESS_LIBS :=
endif
LDFLAGS = $(LIBPATH) $(LIBS) $(BLAS_LIBS) $(OMP_LIBS) $(OPTIONAL_LIBS) $(LDFLAGS_EXTRA) $(GFORTRAN_BENCH_LINK) -lpthread -lm -ldl $(SCOTCH_COMPRESS_LIBS)
