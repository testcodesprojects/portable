###############################################################################
#                           sTiles Build Configuration
###############################################################################
#
# This file contains all configurable build settings for sTiles.
# Users should only need to modify this file to build on different systems.
#
###############################################################################
#                              QUICK REFERENCE
###############################################################################
#
# ======================== BUILDING THE LIBRARY ==============================
#
#   cd /path/to/stiles
#   make all                    # Build libstiles.so (portable-optimized, AVX2)
#   make standalone             # Self-contained libstiles_standalone.a (slow)
#   make clean                  # Remove object files
#   make cleanall               # Remove all build artifacts
#
# After editing a file (source OR header), plain `make` rebuilds only what
# changed (.d dependency tracking) — no cleanall needed.
#
# CPU target (sTiles' own code; MKL/libxsmm dispatch at runtime anyway):
#   make MARCH=avx2             # Default: AVX2/FMA, Intel 2013+/AMD 2017+
#   make MARCH=avx512           # AVX-512 servers (NOT Meteor Lake laptops)
#   make MARCH=native           # Tune for THIS machine (benchmark nodes)
#   make MARCH=generic          # Any x86-64 CPU
#
# Build Modes:
#   make BUILD=release          # Optimized build (default) -O3
#   make BUILD=debug            # Debug symbols -O0 -g3 -DDEBUG
#   make BUILD=sanitize         # AddressSanitizer for memory debugging //LD_PRELOAD=/lib/x86_64-linux-gnu/libasan.so.8
#
# ======================== BINARY DISTRIBUTION ===============================
#
# Create distribution packages with embedded ordering libraries. The CPU
# target is the MARCH knob (see above), not a per-arch target:
#
#   make dist                   # Package the default build (MARCH=avx2)
#   make dist MARCH=avx512      # AVX-512 servers (from a clean tree)
#   make dist MARCH=native      # tune for THIS machine
#   make dist-all               # Linux: build all 3 x86_64 variants at once
#   make dist-clean             # Remove distribution directory
#
# GPU Distribution (requires CUDA toolkit):
#   make STILES_GPU=1 dist                  # GPU build for YOUR GPU (auto-detect arch)
#   make STILES_GPU=1 CUDA_MULTI_ARCH=1 dist  # Fat binary (all GPUs: sm_60-sm_90)
#
# Output: dist/lib/libstiles.so (stripped, ~3MB)
#         dist/include/stiles.h
#         dist/lib/pkgconfig/stiles.pc
#
# EMBEDDED LIBRARIES (users don't need to install):
#   - SCOTCH (graph ordering)
#   - METIS (nested dissection)
#   - SuiteSparse (AMD, COLAMD, CAMD, CCOLAMD, BTF)
#
# REQUIRED BY USERS:
#   - BLAS/LAPACK (Intel MKL or OpenBLAS)
#   - Standard system libraries (libc, libm, libstdc++)
#
# ======================== BENCHMARKS / CHECKS ===============================
#
#   cd run/
#   make                        # Build bench + check programs (see run/Makefile)
#   make help                   # List available targets
#
# ======================== AUTO-DETECTION ====================================
#
# Automatically detected:
#   - Platform:      Linux / macOS
#   - Architecture:  x86_64 / arm64 (Apple Silicon, Graviton)
#   - SIMD:          AVX-512 / AVX2 / AVX / SSE (x86) or NEON (ARM)
#   - BLAS/LAPACK:   MKL → OpenBLAS → Accelerate (macOS) → Generic
#   - OpenMP:        gomp (GCC) / libomp (Clang on macOS)
#   - Optional:      NUMA, hwloc (enabled if available)
#
# ======================== KEY CONFIGURATION =================================
#
#   LAPACKE_DIR      - Path to lapacke.h (optional override; auto-detected)
#   MKLROOT          - Intel MKL installation (optional, improves performance)
#   USE_SUITESPARSE  - SuiteSparse orderings (default: 1)
#   USE_SMART_TILES  - SmartTile support (default: 0, for internal use)
#   STILES_GPU       - CUDA GPU support (default: 0, auto-detects GPU architecture)
#   STILES_LOG_LEVEL - Logging: 0=none, 1=error, 2=warn, 3=info, 4=debug
#
# ======================== QUICK SETUP EXAMPLES ==============================
#
# Intel MKL (Linux):
#   source /opt/intel/oneapi/setvars.sh   # Set up environment
#   make all && make dist                 # portable AVX2 package
#
# OpenBLAS (Ubuntu/Debian):
#   sudo apt install libopenblas-dev liblapacke-dev
#   make all && make dist
#
# OpenBLAS (Fedora/RHEL):
#   sudo dnf install openblas-devel lapack-devel
#   make all && make dist
#
# macOS (Apple Silicon or Intel):
#   # Option 1: Use built-in Accelerate (no install needed, macOS 13.3+)
#   make all
#
#   # Option 2: Use OpenBLAS (recommended for best compatibility)
#   brew install openblas
#   make all
#
###############################################################################

#==============================================================================
# Build Mode Configuration
#==============================================================================
BUILD ?= release

#==============================================================================
# Platform Detection (automatic - do not modify)
#==============================================================================
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
    SHARED_EXT := dylib
    # No baked -install_name here: SHARED_LDFLAGS is reused by other shared
    # libs (e.g. libtileindexer); libstiles' install_name is set at its own link.
    SHARED_LDFLAGS := -dynamiclib
else ifneq ($(filter MINGW% MSYS% CYGWIN% Windows_NT,$(UNAME_S)),)
    # Windows via MinGW-w64 (an MSYS2 UCRT64 shell reports uname -s = MINGW64_*).
    # Target toolchain: the UCRT64 MinGW-w64 (x86_64-w64-mingw32ucrt-*), i.e. the
    # standard Rtools44 toolchain. Kept minimal: it exists so the compile PROBE
    # (`make cross-syntax`) can retarget sTiles' own sources for Windows and
    # report what still assumes Linux/POSIX. It is NOT a finished Windows port.
    PLATFORM := windows
    SHARED_EXT := dll
    SHARED_LDFLAGS := -shared
else
    PLATFORM := linux
    SHARED_EXT := so
    SHARED_LDFLAGS := -shared
endif
RPATH_FLAG = -Wl,-rpath,

#==============================================================================
# Directory Layout (automatic - do not modify)
#==============================================================================
MAKE_INC_DIR   := $(dir $(lastword $(MAKEFILE_LIST)))
PROJECT_ROOT   ?= $(abspath $(MAKE_INC_DIR))
TOOLS_DIR      ?= $(PROJECT_ROOT)/tools
INCLUDE_DIR    ?= $(TOOLS_DIR)/include
LIB_DIR        ?= $(PROJECT_ROOT)/lib
LIB_OUTPUT_DIR ?= $(TOOLS_DIR)/libs
METIS_DIR      ?= $(TOOLS_DIR)/ordering/metis-5.1.0

#==============================================================================
# Compiler Selection
#==============================================================================
# Set USE_CCACHE=0 to disable ccache
USE_CCACHE ?= 1

ifeq ($(USE_CCACHE),1)
    CCACHE := $(shell command -v ccache 2>/dev/null)
endif

ifeq ($(PLATFORM),macos)
    CC_BASE  ?= clang
    CXX_BASE ?= clang++
else
    CC_BASE  ?= gcc
    CXX_BASE ?= g++
endif

ifdef CCACHE
    CC  := $(CCACHE) $(CC_BASE)
    CXX := $(CCACHE) $(CXX_BASE)
else
    CC  := $(CC_BASE)
    CXX := $(CXX_BASE)
endif

# Auto-detect GCC major version so FC/AR/RANLIB match CXX (prevents LTO mismatch)
# Note: Make has a built-in default FC=f77, so ?= won't work — we override it explicitly
ifeq ($(PLATFORM),macos)
  # Apple toolchain: gcc-ar/gcc-ranlib do not exist; plain ar/ranlib handle
  # LLVM LTO objects via libLTO. gfortran comes from Homebrew gcc when present.
  FC      := gfortran
  ARCH    := ar
  RANLIB  := ranlib
else
  _GCC_VER := $(shell $(CXX_BASE) -dumpversion 2>/dev/null | cut -d. -f1)
  ifneq ($(strip $(_GCC_VER)),)
    FC      := $(if $(shell command -v gfortran-$(_GCC_VER) 2>/dev/null),gfortran-$(_GCC_VER),gfortran)
    ARCH    := $(if $(shell command -v gcc-ar-$(_GCC_VER) 2>/dev/null),gcc-ar-$(_GCC_VER),gcc-ar)
    RANLIB  := $(if $(shell command -v gcc-ranlib-$(_GCC_VER) 2>/dev/null),gcc-ranlib-$(_GCC_VER),gcc-ranlib)
  else
    FC      := gfortran
    ARCH    := gcc-ar
    RANLIB  := gcc-ranlib
  endif
endif
NVCC      ?= nvcc
ARCHFLAGS ?= rcs
MKDIR_P   ?= mkdir -p

#==============================================================================
# C++ Standard
#==============================================================================
CXX_STD ?= -std=c++20

#==============================================================================
# Build Mode Flags
#==============================================================================
ifeq ($(BUILD),debug)
    OPT_FLAGS := -O0 -g3 -DDEBUG
    SANITIZE_LDFLAGS :=
else ifeq ($(BUILD),sanitize)
    # AddressSanitizer build for memory debugging
    # Run with: LD_PRELOAD=$(gcc -print-file-name=libasan.so) ./program
    OPT_FLAGS := -O1 -g3 -fsanitize=address -fno-omit-frame-pointer -DDEBUG
    SANITIZE_LDFLAGS := -fsanitize=address
else
    # Release build (default).
    #
    # MARCH selects the CPU target for sTiles' OWN code only — the BLAS
    # backend (MKL/OpenBLAS/ARMPL/Accelerate) and libxsmm pick their kernels
    # at RUNTIME, so the bulk of the FLOPs is unaffected by this flag.
    #   MARCH=avx2    (default) optimized AND portable: AVX2/FMA baseline,
    #                 runs on any Intel 2013+ / AMD 2017+ (x86-64-v3 class)
    #   MARCH=avx512  AVX-512 servers (Intel Skylake-X 2017+, AMD Zen4 2022+);
    #                 SIGILLs on chips without it, e.g. this Meteor Lake laptop
    #   MARCH=native  tune for THIS machine (node-pinned benchmarking runs)
    #   MARCH=generic any x86-64 CPU (maximum compatibility)
    # arm64/aarch64: native → -mcpu=native; otherwise the portable baseline
    # (-mcpu=apple-m1 on macOS covers every Apple Silicon Mac).
    MARCH ?= avx2
    ifneq ($(filter arm64 aarch64,$(UNAME_M)),)
        ifeq ($(MARCH),native)
            ARCH_TARGET_FLAG := -mcpu=native
        else ifeq ($(PLATFORM),macos)
            ARCH_TARGET_FLAG := -mcpu=apple-m1
        else
            ARCH_TARGET_FLAG :=
        endif
    else
        ifeq ($(MARCH),native)
            ARCH_TARGET_FLAG := -march=native
        else ifeq ($(MARCH),avx512)
            # -march=skylake-avx512 ALONE keeps GCC's default 256-bit vector
            # width (EVEX-encoded ymm, ~AVX2 in practice) because 512-bit ops
            # downclock older Skylake-X/Cascade Lake cores. Add
            # -mprefer-vector-width=512 so this genuinely emits 512-bit zmm
            # (a win on Zen4 / Ice Lake+ / Sapphire Rapids where the downclock
            # is minor/gone). Drop that flag if targeting old Skylake-X.
            ARCH_TARGET_FLAG := -march=skylake-avx512 -mtune=generic -mprefer-vector-width=512
        else ifeq ($(MARCH),generic)
            ARCH_TARGET_FLAG := -mtune=generic
        else
            ARCH_TARGET_FLAG := -march=haswell -mtune=generic
        endif
    endif
    # LTO spelling the compiler actually accepts: clang rejects -flto=auto
    # (GCC-only); use thin LTO there instead.
    IS_CLANG := $(findstring clang,$(shell $(CXX_BASE) --version 2>/dev/null))
    LTO_FLAG := $(if $(IS_CLANG),-flto=thin,-flto=auto)
    OPT_FLAGS := -O3 -g $(ARCH_TARGET_FLAG) $(LTO_FLAG) -funroll-loops -DNDEBUG
    SANITIZE_LDFLAGS :=
endif

#==============================================================================
# OpenMP Configuration
#==============================================================================
ifeq ($(PLATFORM),macos)
    # macOS: requires Homebrew libomp
    BREW_PREFIX := $(shell brew --prefix 2>/dev/null || echo /usr/local)
    OPENMP_FLAGS := -Xpreprocessor -fopenmp
    OPENMP_INC   := -I$(BREW_PREFIX)/opt/libomp/include
    OPENMP_LIBS  := -L$(BREW_PREFIX)/opt/libomp/lib -lomp
else
    OPENMP_FLAGS := -fopenmp
    OPENMP_INC   :=
    OPENMP_LIBS  :=
endif

#==============================================================================
# LAPACK/BLAS Configuration (AUTO-DETECTION)
#==============================================================================
# The build system automatically detects BLAS/LAPACK in this order:
#
# macOS ARM64 (Apple Silicon):
#   1. ARM Performance Libraries (ARMPL) - fastest on Apple Silicon
#   2. Intel MKL
#   3. OpenBLAS
#   4. macOS Accelerate framework (fallback)
#
# macOS x86_64 / Linux:
#   1. Intel MKL (best performance on Intel/AMD)
#   2. OpenBLAS (good performance, open source)
#   3. System BLAS/LAPACK (fallback)
#   4. macOS Accelerate framework (macOS only)
#
# To force a specific BLAS, set one of these:
#   MKLROOT=/path/to/mkl        # Force Intel MKL
#   USE_OPENBLAS=1              # Skip ARMPL/MKL detection so OpenBLAS is picked
#   USE_ARMPL=0                 # Disable ARM Performance Libraries (macOS ARM64)
#   ARMPL_DIR=/path/to/armpl    # Specify ARMPL installation path
#==============================================================================

# User overrides (leave empty for auto-detection)
MKLROOT ?=
LAPACKE_DIR ?=
USE_ARMPL ?= 1
ARMPL_DIR ?=
USE_OPENBLAS ?= 0

# Auto-detect BLAS/LAPACK
BLAS_DETECTED := none

# Step 1: ARM Performance Libraries (ARMPL) - macOS ARM64 FIRST
# ARMPL is the fastest option on Apple Silicon Macs.
# Skipped with USE_ARMPL=0 or USE_OPENBLAS=1.
ifeq ($(PLATFORM),macos)
  ifeq ($(UNAME_M),arm64)
    ifneq ($(USE_ARMPL),0)
      ifneq ($(USE_OPENBLAS),1)
        ifneq ($(strip $(ARMPL_DIR)),)
            ARMPL_PREFIX := $(ARMPL_DIR)
        else ifneq ($(wildcard /opt/homebrew/opt/arm-performance-libraries/include/armpl.h),)
            ARMPL_PREFIX := /opt/homebrew/opt/arm-performance-libraries
        else ifneq ($(wildcard /opt/arm/armpl/include/armpl.h),)
            ARMPL_PREFIX := /opt/arm/armpl
        else
            # Check for versioned ARMPL installs (e.g., armpl_25.07_flang-20)
            ARMPL_PREFIX := $(shell ls -d /opt/arm/armpl_*/include/armpl.h 2>/dev/null | head -1 | sed 's|/include/armpl.h||')
        endif
        ifneq ($(strip $(ARMPL_PREFIX)),)
            BLAS_DETECTED := armpl
            LAPACKE_INC := -I$(ARMPL_PREFIX)/include
            LAPACKE_LIB := -L$(ARMPL_PREFIX)/lib -larmpl -lamath -lastring
            LAPACKE_DEF := -DUSE_ARMPL
            $(info BLAS: ARM Performance Libraries detected at $(ARMPL_PREFIX))
        endif
      endif
    endif
  endif
endif

# Step 2: Check for Intel MKL (skipped when USE_OPENBLAS=1 forces OpenBLAS)
ifeq ($(BLAS_DETECTED)$(USE_OPENBLAS),none0)
    # Check common MKL locations
    ifneq ($(strip $(MKLROOT)),)
        # User specified MKLROOT
        MKL_FOUND := $(MKLROOT)
    else ifneq ($(wildcard /opt/intel/oneapi/mkl/latest/include/mkl.h),)
        MKL_FOUND := /opt/intel/oneapi/mkl/latest
    else ifneq ($(wildcard /opt/intel/mkl/include/mkl.h),)
        MKL_FOUND := /opt/intel/mkl
    else
        MKL_FOUND :=
    endif

    ifneq ($(strip $(MKL_FOUND)),)
        BLAS_DETECTED := mkl
        MKLROOT := $(MKL_FOUND)
        # Pick the dir that actually holds the MKL libraries: oneAPI <=2023 uses
        # lib/intel64, oneAPI >=2024 (e.g. ibex's /sw 2025.3) uses lib/. Detect by
        # presence rather than by arch so no MKL_LIB_PATH override is needed.
        ifneq ($(wildcard $(MKLROOT)/lib/intel64/libmkl_core.*),)
            MKL_LIB_PATH := $(MKLROOT)/lib/intel64
        else
            MKL_LIB_PATH := $(MKLROOT)/lib
        endif
        LAPACKE_INC := -I$(MKLROOT)/include
        LAPACKE_LIB := -L$(MKL_LIB_PATH) -lmkl_gf_lp64 -lmkl_core -lmkl_sequential
        LAPACKE_DEF := -DSTILES_WITH_MKL
        $(info BLAS: Intel MKL detected at $(MKLROOT))
        # Static MKL archives for embedding into libstiles.so.
        # Embedding isolates sTiles' MKL (sequential) from any MKL the consumer
        # process may load with a different threading layer (e.g. a host program
        # that loads MKL with the GNU threading layer, mkl_gnu_thread). Combined
        # with the version script, MKL symbols stay local to libstiles.so and
        # cannot clash at runtime.
        ifeq ($(PLATFORM),linux)
            ifneq ($(wildcard $(MKL_LIB_PATH)/libmkl_core.a),)
                LAPACKE_STATIC_LIBS := -Wl,--start-group \
                    $(MKL_LIB_PATH)/libmkl_gf_lp64.a \
                    $(MKL_LIB_PATH)/libmkl_sequential.a \
                    $(MKL_LIB_PATH)/libmkl_core.a \
                    -Wl,--end-group
                $(info MKL: static archives found - will be embedded into libstiles.so)
            endif
        endif
    endif
endif

# Step 3: Check for OpenBLAS (USE_OPENBLAS=1 skips ARMPL/MKL above, so
# detection lands here; both Homebrew prefixes and pkg-config are probed)
ifeq ($(BLAS_DETECTED),none)
    OPENBLAS_PKG := $(shell pkg-config --exists openblas 2>/dev/null && echo 1)
    ifeq ($(OPENBLAS_PKG),1)
        BLAS_DETECTED := openblas
        LAPACKE_INC := $(shell pkg-config --cflags openblas 2>/dev/null)
        # MinGW/MSYS2 OpenBLAS bundles LAPACKE inside libopenblas (no separate
        # liblapacke); Linux ships a standalone liblapacke, so append it there.
        LAPACKE_LIB := $(shell pkg-config --libs openblas 2>/dev/null) $(if $(filter windows,$(PLATFORM)),,-llapacke)
        LAPACKE_DEF := -DUSE_GENERIC_LAPACKE
        $(info BLAS: OpenBLAS detected via pkg-config)
    else ifneq ($(wildcard /usr/include/openblas/cblas.h),)
        BLAS_DETECTED := openblas
        LAPACKE_INC := -I/usr/include/openblas
        LAPACKE_LIB := -lopenblas -llapacke
        LAPACKE_DEF := -DUSE_GENERIC_LAPACKE
        $(info BLAS: OpenBLAS detected in /usr/include/openblas)
    else ifneq ($(wildcard /usr/local/opt/openblas/include/cblas.h),)
        # Homebrew OpenBLAS on macOS (Intel)
        BLAS_DETECTED := openblas
        OPENBLAS_PREFIX := /usr/local/opt/openblas
        LAPACKE_INC := -I$(OPENBLAS_PREFIX)/include -I/usr/local/opt/lapack/include
        LAPACKE_LIB := -L$(OPENBLAS_PREFIX)/lib -L/usr/local/opt/lapack/lib -lopenblas -llapacke
        LAPACKE_DEF := -DUSE_GENERIC_LAPACKE
        $(info BLAS: OpenBLAS detected via Homebrew (Intel Mac))
    else ifneq ($(wildcard /opt/homebrew/opt/openblas/include/cblas.h),)
        # Homebrew OpenBLAS on macOS (Apple Silicon)
        BLAS_DETECTED := openblas
        OPENBLAS_PREFIX := /opt/homebrew/opt/openblas
        LAPACKE_INC := -I$(OPENBLAS_PREFIX)/include -I/opt/homebrew/opt/lapack/include
        LAPACKE_LIB := -L$(OPENBLAS_PREFIX)/lib -L/opt/homebrew/opt/lapack/lib -lopenblas -llapacke
        LAPACKE_DEF := -DUSE_GENERIC_LAPACKE
        $(info BLAS: OpenBLAS detected via Homebrew (Apple Silicon))
    endif
endif

# Step 4: Check for system BLAS/LAPACK
ifeq ($(BLAS_DETECTED),none)
    LAPACKE_PKG := $(shell pkg-config --exists lapacke 2>/dev/null && echo 1)
    ifeq ($(LAPACKE_PKG),1)
        BLAS_DETECTED := system
        LAPACKE_INC := $(shell pkg-config --cflags lapacke 2>/dev/null)
        LAPACKE_LIB := $(shell pkg-config --libs lapacke 2>/dev/null)
        LAPACKE_DEF := -DUSE_GENERIC_LAPACKE
        $(info BLAS: System LAPACK detected via pkg-config)
    else ifneq ($(wildcard /usr/include/lapacke.h),)
        BLAS_DETECTED := system
        LAPACKE_INC := -I/usr/include
        LAPACKE_LIB := -llapacke -llapack -lblas
        LAPACKE_DEF := -DUSE_GENERIC_LAPACKE
        $(info BLAS: System LAPACK detected in /usr/include)
    endif
endif

# Step 5: macOS Accelerate framework (fallback)
ifeq ($(BLAS_DETECTED),none)
    ifeq ($(PLATFORM),macos)
        BLAS_DETECTED := accelerate
        LAPACKE_INC :=
        LAPACKE_LIB := -framework Accelerate
        # macOS 13.3+ (Ventura) has LAPACKE in Accelerate
        # Define both USE_ACCELERATE and ACCELERATE_NEW_LAPACK
        LAPACKE_DEF := -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK
        $(info BLAS: macOS Accelerate framework (built-in, no installation needed))
    endif
endif

# Final check - no BLAS found
ifeq ($(BLAS_DETECTED),none)
    $(warning ============================================================)
    $(warning No BLAS/LAPACK found! Install one of:)
    $(warning   Ubuntu/Debian: sudo apt install libopenblas-dev liblapacke-dev)
    $(warning   Fedora/RHEL:   sudo dnf install openblas-devel lapack-devel)
    $(warning   macOS:         brew install openblas)
    $(warning   Intel MKL:     source /opt/intel/oneapi/setvars.sh)
    $(warning ============================================================)
    LAPACKE_LIB := -llapacke -llapack -lblas
    LAPACKE_DEF := -DUSE_GENERIC_LAPACKE
endif

# Allow user to override LAPACKE header location
ifneq ($(strip $(LAPACKE_DIR)),)
    LAPACKE_INC := -I$(LAPACKE_DIR)
endif

# When ARMPL is detected on macOS, use ARMPL's bundled libomp instead of
# Homebrew's libomp to avoid loading two copies of libomp at runtime.
# Two libomp copies cause segfaults in multi-threaded execution.
ifeq ($(BLAS_DETECTED),armpl)
    ifeq ($(PLATFORM),macos)
        ifneq ($(wildcard $(ARMPL_PREFIX)/lib/libomp.dylib),)
            OPENMP_LIBS := -L$(ARMPL_PREFIX)/lib -Wl,-rpath,$(ARMPL_PREFIX)/lib -lomp
            $(info OpenMP: using ARMPL's bundled libomp (avoids duplicate libomp))
        endif
    endif
endif

# -ldl (dlopen) does not exist on Windows; POSIX platforms need it.
LIBLAPACK := $(LAPACKE_LIB) -lpthread -lm $(if $(filter windows,$(PLATFORM)),,-ldl)

#==============================================================================
# Optional Features
#==============================================================================

# mimalloc was removed. It was previously embedded into libstiles.so as the
# operator new/delete override, but the version script (stiles.map, `local: *`)
# localizes those symbols, so libstiles freed with mimalloc what libstdc++
# allocated with glibc (std::string/stringstream) — a cross-allocator free that
# crashed in mi_free_size at any visible log level. sTiles now uses the standard
# C allocator everywhere (one allocator, consistent with libstdc++).

# NUMA support (Linux only)
ifeq ($(PLATFORM),linux)
    NUMA_AVAILABLE := $(shell ldconfig -p 2>/dev/null | grep -q libnuma && echo 1)
    ifeq ($(NUMA_AVAILABLE),1)
        NUMA_DEF := -DNUMA_ENABLED
        NUMA_LIB := -lnuma
    endif
endif

# hwloc support (required for topology detection)
HWLOC_AVAILABLE := $(shell pkg-config --exists hwloc 2>/dev/null && echo 1)
ifeq ($(HWLOC_AVAILABLE),1)
    HWLOC_INC := $(shell pkg-config --cflags hwloc)
    HWLOC_LIB := $(shell pkg-config --libs hwloc)
    HWLOC_DEF := -DSTILES_HWLOC
else ifeq ($(PLATFORM),macos)
    # Homebrew hwloc (pkg-config may not find it without PKG_CONFIG_PATH)
    HWLOC_BREW := $(shell brew --prefix hwloc 2>/dev/null)
    ifneq ($(strip $(HWLOC_BREW)),)
        ifneq ($(wildcard $(HWLOC_BREW)/include/hwloc.h),)
            HWLOC_AVAILABLE := 1
            HWLOC_INC := -I$(HWLOC_BREW)/include
            HWLOC_LIB := -L$(HWLOC_BREW)/lib -lhwloc
            HWLOC_DEF := -DSTILES_HWLOC
            $(info hwloc: detected via Homebrew at $(HWLOC_BREW))
        endif
    endif
endif
ifeq ($(HWLOC_AVAILABLE),1)
    $(info hwloc: enabled)
else
    $(warning ============================================================)
    $(warning hwloc NOT found! hwloc is required for topology detection.)
    $(warning   Ubuntu/Debian: sudo apt install libhwloc-dev)
    $(warning   Fedora/RHEL:   sudo dnf install hwloc-devel)
    $(warning   macOS:         brew install hwloc)
    $(warning ============================================================)
endif

#==============================================================================
# Standalone .so: statically embed numa / hwloc when possible
#==============================================================================
# Goal: libstiles.so should load on a machine that lacks these shared libs
# (libnuma.so.1 / libhwloc.so.15). Only a PIC static archive
# can be linked into a .so, so each candidate .a is force-linked into a
# throwaway .so at parse time and kept ONLY if that succeeds. Ubuntu's
# libnuma.a, for example, is built without -fPIC and is rejected here, so numa
# stays dynamic until a PIC libnuma.a is available. The universal runtime libs
# (libc/stdc++/gomp/m/z/bz2/lzma) intentionally stay dynamic. gfortran is
# embedded separately via -static-libgfortran in the Makefile's .so link.
# Set STANDALONE_SO=0 to keep the historical all-dynamic .so.
STANDALONE_SO ?= 1
# PROJECT_ROOT (not STILES_DIR): make.inc is also included directly by
# Makefiles that never define STILES_DIR (e.g. tools/TileIndexer), and an
# empty STILES_DIR would put the cache at /.standalone_so.cache.
STANDALONE_SO_CACHE := $(PROJECT_ROOT)/.standalone_so.cache
ifeq ($(STANDALONE_SO),1)
ifeq ($(PLATFORM),linux)
  # The PIC test below force-links each candidate .a into a throwaway .so; this
  # is moderately expensive, and make.inc is re-parsed by every recursive
  # sub-make, so cache the resolved paths and reuse them. Delete the cache (or
  # `make clean`) to re-detect after installing/removing a static lib.
  -include $(STANDALONE_SO_CACHE)
ifndef STANDALONE_SO_CACHED
  # pic_static: echo $(1) if that archive can be embedded in a .so, else empty.
  # -shared allows undefined symbols, so only a real PIC-relocation failure (not
  # a missing transitive dep) rejects the archive. No source input is needed —
  # the compiler links the archive straight into a throwaway test .so (mktemp:
  # ld cannot reliably write a shared object to /dev/null, and a fixed /tmp
  # name would collide between users).
  pic_static = $(if $(strip $(1)),$(if $(shell t=$$(mktemp) && $(CC_BASE) -fPIC -shared -o $$t -Wl,--whole-archive $(1) -Wl,--no-whole-archive >/dev/null 2>&1 && echo ok; rm -f $$t),$(1),))

  ifneq ($(strip $(NUMA_DEF)),)
    NUMA_A_CAND := $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libnuma.a /usr/lib64/libnuma.a /usr/lib/libnuma.a))
    NUMA_STATIC_LIB := $(call pic_static,$(NUMA_A_CAND))
  endif

  ifeq ($(HWLOC_AVAILABLE),1)
    HWLOC_A_CAND := $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libhwloc.a /usr/lib64/libhwloc.a /usr/lib/libhwloc.a))
    HWLOC_STATIC_LIB := $(call pic_static,$(HWLOC_A_CAND))
    # hwloc.a pulls these; libudev stays dynamic (present on all systemd Linux).
    HWLOC_STATIC_EXTRA := -ludev -lm
  endif

  # Persist the decision so recursive sub-makes skip the link tests.
  $(shell { echo 'STANDALONE_SO_CACHED := 1'; \
            echo 'NUMA_STATIC_LIB := $(NUMA_STATIC_LIB)'; \
            echo 'HWLOC_STATIC_LIB := $(HWLOC_STATIC_LIB)'; \
            echo 'HWLOC_STATIC_EXTRA := $(HWLOC_STATIC_EXTRA)'; } > $(STANDALONE_SO_CACHE))
  $(if $(HWLOC_STATIC_LIB),$(info hwloc: PIC static archive found, embedding into libstiles.so))
  $(if $(NUMA_STATIC_LIB),$(info numa: PIC static archive found, embedding into libstiles.so),$(if $(NUMA_A_CAND),$(info numa: libnuma.a is non-PIC, keeping libnuma dynamic)))
endif
endif
endif

#==============================================================================
# sTiles Build Options
#==============================================================================
# Log level: 0=none, 1=error, 2=warning, 3=info, 4=debug
STILES_LOG_LEVEL ?= 3

# Build mode: FASTMODE (default), SAFEMODE, or SEMISPARSEMODE
STILES_MODE ?= FASTMODE

# Memory leak checking
STILES_LEAK_CHECK ?= 0

STILES_DEFS := -DSTILES_LOG_LEVEL=$(STILES_LOG_LEVEL) -DSTILES_$(STILES_MODE)
ifeq ($(STILES_LEAK_CHECK),1)
    STILES_DEFS += -DSTILES_LEAK_CHECK
endif

# Export matrix for debugging (set EXPORT_MATRIX=1 on command line to enable)
EXPORT_MATRIX ?= 0
ifeq ($(EXPORT_MATRIX),1)
    STILES_DEFS += -DEXPORT_MATRIX
endif

# Sparse tile CSC support (set SPARSE_STILES=1 on command line to enable)
SPARSE_STILES ?= 0
ifeq ($(SPARSE_STILES),1)
    STILES_DEFS += -DSPARSE_STILES
    $(info SPARSE_STILES: enabled - SparseTileCSC support active)
endif

# Wait-strategy selection for in_cond_wait spin loop:
#   0 = LEGACY    sched_yield() per spin (default — matches original behavior)
#   1 = PAUSE     CPU pause hint only, no syscalls (best when oversubscription
#                 is impossible and producer is always on a different core)
#   2 = ADAPTIVE  pause N times, then sched_yield() (good on >=8 cores)
# Override on command line: make STILES_WAIT_MODE=2 ...
STILES_WAIT_MODE ?= 0
STILES_DEFS += -DSTILES_WAIT_MODE=$(STILES_WAIT_MODE)
ifeq ($(STILES_WAIT_MODE),1)
    $(info STILES_WAIT_MODE: PAUSE (cpu-pause spin, no syscalls))
else ifeq ($(STILES_WAIT_MODE),2)
    $(info STILES_WAIT_MODE: ADAPTIVE (pause then yield))
else
    $(info STILES_WAIT_MODE: LEGACY (sched_yield per spin))
endif

#==============================================================================
# OMP runtime affinity (read by libgomp at LD time; can't be set from C code)
# ─────────────────────────────────────────────────────────────────────────────
# For best OMP scaling export the following before launching your binary
# (e.g. in ~/.bashrc, in a job script, or via run/run_with_omp.sh):
#
#     export OMP_PLACES=cores
#     export OMP_DYNAMIC=false
#
# OMP_PLACES=cores pins OMP threads to physical cores, closing the ~2×
# scaling gap vs the pthreads backend at high core counts. These vars affect
# ONLY the OMP backend; the pthreads backend is untouched (we deliberately
# do NOT recommend OMP_WAIT_POLICY=ACTIVE, which would make idle OMP threads
# busy-spin and steal cycles from pthreads workers in mixed-backend runs).
#==============================================================================

#==============================================================================
# GPU (CUDA) Support
#==============================================================================
STILES_GPU       ?= 0
CUDA_PATH        ?= /usr/local/cuda

# CUDA Multi-Architecture Build (for distribution)
# Set CUDA_MULTI_ARCH=1 to build a fat binary that works on multiple GPUs
CUDA_MULTI_ARCH  ?= 0

# Auto-detect CUDA architecture if not specified
# Uses nvidia-smi to query the GPU compute capability (e.g., 8.9 -> sm_89)
ifndef STILES_CUDA_ARCH
    # Get compute capability (e.g., "8.9") and convert to sm_XX (e.g., "sm_89")
    CUDA_COMPUTE_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    ifneq ($(strip $(CUDA_COMPUTE_CAP)),)
        STILES_CUDA_ARCH := sm_$(CUDA_COMPUTE_CAP)
    else
        # Fallback to sm_80 (Ampere) if detection fails
        STILES_CUDA_ARCH := sm_80
    endif
endif

ifeq ($(STILES_GPU),1)
    CUDA_INC := -I$(CUDA_PATH)/include
    CUDA_LIB := -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcusolver $(RPATH_FLAG)$(CUDA_PATH)/lib64
    CUDA_DEF := -DSTILES_GPU

    ifeq ($(CUDA_MULTI_ARCH),1)
        # Fat binary: supports multiple GPU architectures (for distribution)
        # Includes: Pascal(60), Volta(70), Turing(75), Ampere(80,86), Ada(89), Hopper(90)
        # Plus PTX for forward compatibility with future GPUs
        CUDA_GENCODE := \
            -gencode arch=compute_60,code=sm_60 \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_89,code=sm_89 \
            -gencode arch=compute_90,code=sm_90 \
            -gencode arch=compute_90,code=compute_90
        # Note: -D__PRFCHW__ and -D_AMXTILEINTRIN_H_INCLUDED prevent nvcc from choking on GCC intrinsic headers
        NVCCFLAGS := -std=c++17 \
                     -Xcompiler "$(OPT_FLAGS) -fPIC $(OPENMP_FLAGS) -DADD_ $(LAPACKE_DEF)" \
                     $(CUDA_GENCODE) \
                     -DSTILES_GPU $(LAPACKE_DEF) \
                     -D__PRFCHW__ -D_AMXTILEINTRIN_H_INCLUDED -D_AMXFP16INTRIN_H_INCLUDED -D_AMXBF16INTRIN_H_INCLUDED -D_AMXINT8INTRIN_H_INCLUDED
        $(info GPU: CUDA enabled - MULTI-ARCH fat binary (sm_60 to sm_90 + PTX))
    else
        # Single architecture: optimized for current/specified GPU
        # Note: -D__PRFCHW__ and -D_AMXTILEINTRIN_H_INCLUDED prevent nvcc from choking on GCC intrinsic headers
        NVCCFLAGS := -std=c++17 \
                     -Xcompiler "$(OPT_FLAGS) -fPIC $(OPENMP_FLAGS) -DADD_ $(LAPACKE_DEF)" \
                     -arch=$(STILES_CUDA_ARCH) \
                     -DSTILES_GPU $(LAPACKE_DEF) \
                     -D__PRFCHW__ -D_AMXTILEINTRIN_H_INCLUDED -D_AMXFP16INTRIN_H_INCLUDED -D_AMXBF16INTRIN_H_INCLUDED -D_AMXINT8INTRIN_H_INCLUDED
        $(info GPU: CUDA enabled with architecture $(STILES_CUDA_ARCH))
    endif
endif

#==============================================================================
# Bundled Dependencies (SCOTCH, METIS, SuiteSparse)
#==============================================================================
# These libraries are STATICALLY EMBEDDED into libstiles.so for distribution.
# Users of the distributed library do NOT need to install these.
#
# The vendored sources are compiled directly into libstiles (no sub-builds);
# `make clean && make all` recompiles them. `make clean-ordering-subbuilds`
# removes only the regenerated _local/cmake artifacts of the offline
# install_*_local.sh regeneration scripts.
#==============================================================================

# SCOTCH - Graph ordering library, compiled directly into libstiles.
# The libscotch sources are compiled into ../ordering/sco_*.o by the process
# Makefile and archived into libstiles.a; scotch.h and the flex/bison/dummysizes
# outputs are vendored under scotch_inline/ (no per-system codegen). Only the
# compression libs SCOTCH depends on are linked. There is no cmake sub-build.
SCOTCH_SRC_ROOT   := $(TOOLS_DIR)/ordering/scotch/scotch
SCOTCH_INLINE_DIR := $(TOOLS_DIR)/ordering/scotch/scotch_inline
SCOTCH_INC        := -I$(SCOTCH_INLINE_DIR)
SCOTCH_LIB        :=
SCOTCH_STATIC_LIBS :=
# SCOTCH's compressed-file support (and thus its -lz/-lbz2/-llzma deps) is only
# compiled on Linux (see tools/process/Makefile). macOS and Windows build SCOTCH
# without it, so no compression libs are needed — and -llzma would fail to link
# there.
ifeq ($(PLATFORM),linux)
    SCOTCH_EXTRA_LIBS := -lz -lbz2 -llzma
else
    SCOTCH_EXTRA_LIBS :=
endif

# SuiteSparse - AMD/CAMD/COLAMD/CCOLAMD/BTF, compiled directly into libstiles.
# The sources are compiled into ../ordering/ss_*.o by tools/process/Makefile and
# archived into libstiles.a (no cmake sub-build, no archives linked). Headers
# come straight from the in-tree source modules; include_inline/ supplies the
# <suitesparse/...> prefix the sTiles wrappers use. Set USE_SUITESPARSE=0 to
# drop SuiteSparse ordering support entirely.
USE_SUITESPARSE ?= 1
SUITESPARSE_SRC_ROOT := $(TOOLS_DIR)/suitesparse/SuiteSparse
ifeq ($(USE_SUITESPARSE),1)
    SUITESPARSE_INC := -I$(TOOLS_DIR)/suitesparse/include_inline \
                       -I$(SUITESPARSE_SRC_ROOT)/SuiteSparse_config \
                       -I$(SUITESPARSE_SRC_ROOT)/AMD/Include \
                       -I$(SUITESPARSE_SRC_ROOT)/AMD/Source \
                       -I$(SUITESPARSE_SRC_ROOT)/CAMD/Include \
                       -I$(SUITESPARSE_SRC_ROOT)/CAMD/Source \
                       -I$(SUITESPARSE_SRC_ROOT)/COLAMD/Include \
                       -I$(SUITESPARSE_SRC_ROOT)/COLAMD/Source \
                       -I$(SUITESPARSE_SRC_ROOT)/CCOLAMD/Include \
                       -I$(SUITESPARSE_SRC_ROOT)/CCOLAMD/Source \
                       -I$(SUITESPARSE_SRC_ROOT)/BTF/Include \
                       -I$(SUITESPARSE_SRC_ROOT)/BTF/Source
    SUITESPARSE_DEF := -DSTILES_WITH_SUITESPARSE
    SUITESPARSE_LIB :=
    SUITESPARSE_STATIC_LIBS :=
    SUITESPARSE_EXTRA_LIBS :=
endif

# libxsmm - JIT'd small-matrix GEMM kernels for selinv hot path (statically embedded)
USE_LIBXSMM ?= 1
LIBXSMM_ROOT ?= $(TOOLS_DIR)/libxsmm/libxsmm_local
ifeq ($(USE_LIBXSMM),1)
    ifneq ($(wildcard $(LIBXSMM_ROOT)/include/libxsmm.h),)
        LIBXSMM_INC := -I$(LIBXSMM_ROOT)/include
        LIBXSMM_DEF := -DSTILES_WITH_LIBXSMM
        # Detect lib dir (always lib for libxsmm's own Makefile install)
        ifneq ($(wildcard $(LIBXSMM_ROOT)/lib/libxsmm.a),)
            LIBXSMM_LIB_DIR := $(LIBXSMM_ROOT)/lib
        else ifneq ($(wildcard $(LIBXSMM_ROOT)/lib64/libxsmm.a),)
            LIBXSMM_LIB_DIR := $(LIBXSMM_ROOT)/lib64
        else
            LIBXSMM_LIB_DIR := $(LIBXSMM_ROOT)/lib
        endif
        LIBXSMM_LIB := -L$(LIBXSMM_LIB_DIR) $(RPATH_FLAG)$(LIBXSMM_LIB_DIR) -lxsmm -lxsmmgen -ldl
        # Static archive for embedding into libstiles.so (--whole-archive ensures
        # libxsmm symbols are exported and not stripped by --gc-sections).
        ifneq ($(wildcard $(LIBXSMM_LIB_DIR)/libxsmm.a),)
            ifeq ($(PLATFORM),macos)
                LIBXSMM_STATIC_LIBS := \
                    -Wl,-force_load,$(LIBXSMM_LIB_DIR)/libxsmm.a
            else
                LIBXSMM_STATIC_LIBS := -Wl,--whole-archive \
                    $(LIBXSMM_LIB_DIR)/libxsmm.a \
                    -Wl,--no-whole-archive
            endif
        endif
    endif
endif

#==============================================================================
# METIS — try system install first, fall back to bundled build
#==============================================================================
# Detection order (first hit wins):
#   1. $METIS_ROOT env var pointing to <prefix>/include/metis.h + <prefix>/lib*/libmetis.*
#   2. pkg-config --exists metis
#   3. /usr/include/metis.h, /usr/local/include/metis.h, /opt/homebrew/include/metis.h, $HOME/.local/include/metis.h
#
# Override:
#   make USE_SYSTEM_METIS=0 all     → always use bundled static build (default)
#   make USE_SYSTEM_METIS=auto all  → detect a system install, else bundled
#   make USE_SYSTEM_METIS=1 all     → require system install, error if not found
#
# Detection (auto/1) rejects metis.h that:
#   - hard-codes IDXTYPEWIDTH 64 (we compile with 32 → ABI mismatch), OR
#   - lacks METIS_NodeNDP (METIS 5.2 dropped it; sTiles requires it)
#
# Default is 0 (bundled static), NOT auto: a system METIS is linked DYNAMICALLY
# (-lmetis), which adds a libmetis.so.N NEEDED entry to libstiles.so. That
# dependency is resolved against whatever METIS sits on the *consumer's* machine
# at load time, and the version script cannot localize a dynamic NEEDED symbol —
# so a different/incompatible system METIS leaks in and can silently change
# behaviour or slow things down. Embedding the bundled static libmetis.a keeps
# libstiles self-contained: its METIS symbols become local and nothing external
# can rebind them. Opt back into detection with USE_SYSTEM_METIS=auto.
# Unless a system METIS is used, METIS + GKlib are compiled directly into
# libstiles (no cmake/`make config` sub-build) — see the inline branch below.
USE_SYSTEM_METIS ?= 0

METIS_VENDOR_DIR := $(METIS_DIR)
METIS_ABS_DIR    := $(abspath $(METIS_VENDOR_DIR))
GKLIB_DIR        := $(abspath $(TOOLS_DIR)/ordering/GKlib)

METIS_SYS_PREFIX :=
METIS_SYS_INC    :=
METIS_SYS_LIB    :=

ifneq ($(USE_SYSTEM_METIS),0)
  METIS_CANDIDATE_PREFIXES :=
  ifneq ($(strip $(METIS_ROOT)),)
    METIS_CANDIDATE_PREFIXES := $(METIS_ROOT)
  endif
  METIS_PKGCONFIG_PREFIX := $(shell pkg-config --variable=prefix metis 2>/dev/null)
  ifneq ($(strip $(METIS_PKGCONFIG_PREFIX)),)
    METIS_CANDIDATE_PREFIXES += $(METIS_PKGCONFIG_PREFIX)
  endif
  METIS_CANDIDATE_PREFIXES += /usr /usr/local /opt/homebrew $(HOME)/.local

  # A system METIS install is usable when:
  #   1. <prefix>/include/metis.h exists
  #   2. The header does NOT hard-code IDXTYPEWIDTH 64
  #   3. METIS_NodeNDP is declared in the header (METIS 5.1.x; 5.2 dropped it)
  define probe_metis_prefix
    $(if $(METIS_SYS_PREFIX),,\
      $(if $(wildcard $(1)/include/metis.h),\
        $(if $(shell awk '/^[[:space:]]*#[[:space:]]*define[[:space:]]+IDXTYPEWIDTH[[:space:]]+64/{print "bad"; exit}' $(1)/include/metis.h 2>/dev/null),\
          ,\
          $(if $(shell grep -q METIS_NodeNDP $(1)/include/metis.h 2>/dev/null && echo ok),\
            $(eval METIS_SYS_PREFIX := $(1)),\
            $(info METIS: skipping $(1) — header lacks METIS_NodeNDP (METIS >= 5.2 dropped it; need 5.1.x))))))
  endef
  $(foreach p,$(METIS_CANDIDATE_PREFIXES),$(call probe_metis_prefix,$(p)))

  ifneq ($(strip $(METIS_SYS_PREFIX)),)
    METIS_SYS_INC := -I$(METIS_SYS_PREFIX)/include
    # Detect lib dir: standard /lib, /lib64, or Debian/Ubuntu multiarch /lib/<tuple>
    METIS_MULTIARCH := $(shell $(CXX) -print-multiarch 2>/dev/null)
    ifneq ($(wildcard $(METIS_SYS_PREFIX)/lib/libmetis.*),)
      METIS_SYS_LIB := -L$(METIS_SYS_PREFIX)/lib -lmetis
    else ifneq ($(wildcard $(METIS_SYS_PREFIX)/lib64/libmetis.*),)
      METIS_SYS_LIB := -L$(METIS_SYS_PREFIX)/lib64 -lmetis
    else ifneq ($(strip $(METIS_MULTIARCH)),)
      ifneq ($(wildcard $(METIS_SYS_PREFIX)/lib/$(METIS_MULTIARCH)/libmetis.*),)
        METIS_SYS_LIB := -L$(METIS_SYS_PREFIX)/lib/$(METIS_MULTIARCH) -lmetis
      endif
    endif
    ifeq ($(strip $(METIS_SYS_LIB)),)
      METIS_SYS_PREFIX :=
      METIS_SYS_INC    :=
    endif
  endif
endif

ifeq ($(USE_SYSTEM_METIS),1)
  ifeq ($(strip $(METIS_SYS_PREFIX)),)
    $(error USE_SYSTEM_METIS=1 but no usable system METIS found (set METIS_ROOT=<prefix>))
  endif
endif

ifneq ($(strip $(METIS_SYS_PREFIX)),)
  # ===== Use system METIS =====
  METIS_HEADER_FILE      := $(METIS_SYS_PREFIX)/include/metis.h
  METIS_INC              := $(METIS_SYS_INC) -DSTILES_METIS_HEADER='"$(METIS_HEADER_FILE)"'
  METIS_DEF              := -DIDXTYPEWIDTH=32 -DREALTYPEWIDTH=32 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64
  METIS_LINK_FLAGS_LINUX := $(METIS_SYS_LIB)
  METIS_LINK_FLAGS_MACOS := $(METIS_SYS_LIB)
else
  # ===== Inline METIS + GKlib (sources compiled into libstiles) =====
  # No sub-build, no archives: tools/process/Makefile compiles libmetis/*.c and
  # GKlib/src/*.c into ../ordering/mt_*.o / gk_*.o, archived into libstiles.a.
  # metis_inline/metis.h bakes in IDXTYPEWIDTH/REALTYPEWIDTH (like the cmake
  # build's generated header) so sTiles' own C++ sources compile without relying
  # on METIS_DEF, which the final CXXFLAGS assembly does not propagate.
  METIS_HEADER_FILE      := $(abspath $(TOOLS_DIR)/ordering/metis_inline)/metis.h
  METIS_INC              := -I$(abspath $(TOOLS_DIR)/ordering/metis_inline) -I$(GKLIB_DIR)/include -DSTILES_METIS_HEADER='"$(METIS_HEADER_FILE)"'
  METIS_DEF              := -DIDXTYPEWIDTH=32 -DREALTYPEWIDTH=32 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64
  METIS_LINK_FLAGS_LINUX :=
  METIS_LINK_FLAGS_MACOS :=
endif

#==============================================================================
# Final Flag Assembly (do not modify below)
#==============================================================================
# METIS_INC must come BEFORE SCOTCH_INC (SCOTCH ships its own metis.h stub).
INC := -I$(INCLUDE_DIR) $(LAPACKE_INC) $(OPENMP_INC) \
       $(HWLOC_INC) $(METIS_INC) $(SCOTCH_INC) $(SUITESPARSE_INC) $(LIBXSMM_INC) $(CUDA_INC)

# Build date and expiry configuration (auto-set at compile time).
# Leading zeros are stripped portably (GNU date's %-m is not available in
# BSD date on macOS); an unstripped 08/09 would be an invalid octal literal.
BUILD_YEAR  := $(shell date +%Y)
BUILD_MONTH := $(shell date +%m | sed 's/^0//')
BUILD_DAY   := $(shell date +%d | sed 's/^0//')

# How many years the library is valid after build (change this to adjust expiry)
VALID_YEARS := 5

# Whole pipeline silenced and defaulted: in a non-git checkout git must not
# spam stderr on every parse, and the fallback must actually apply.
GIT_VERSION := $(shell { git -C "$(PROJECT_ROOT)" log -1 --format='%ai [%h]' 2>/dev/null || echo "unknown"; } | tr ' ' _)

BUILD_DATE_DEFS := -DSTILES_BUILD_YEAR=$(BUILD_YEAR) \
                   -DSTILES_BUILD_MONTH=$(BUILD_MONTH) \
                   -DSTILES_BUILD_DAY=$(BUILD_DAY) \
                   -DSTILES_VALID_YEARS=$(VALID_YEARS) \
                   -DSTILES_GIT_VERSION=\"$(GIT_VERSION)\"

COMMON_DEFS := -DADD_ $(LAPACKE_DEF) $(NUMA_DEF) \
               $(HWLOC_DEF) $(SUITESPARSE_DEF) $(LIBXSMM_DEF) $(CUDA_DEF) $(STILES_DEFS) \
               $(BUILD_DATE_DEFS)

CFLAGS   := $(OPT_FLAGS) -fPIC $(OPENMP_FLAGS) $(COMMON_DEFS) $(INC)
CXXFLAGS := $(OPT_FLAGS) -fPIC $(OPENMP_FLAGS) $(CXX_STD) $(COMMON_DEFS) $(INC)
# Fortran flags for the fixed-form ordering sources (AMD/GENMMD/...):
# deliberately -O2 without -march/-flto so orderings stay stable across builds.
FFLAGS   := -O2 -fPIC

LDFLAGS := $(SANITIZE_LDFLAGS) $(OPENMP_LIBS) $(NUMA_LIB) \
           $(HWLOC_LIB) $(SCOTCH_LIB) $(SUITESPARSE_LIB) $(LIBXSMM_LIB) $(CUDA_LIB) \
           $(LIBLAPACK)

# Note: Run 'make show-config' from the project root to display build configuration
