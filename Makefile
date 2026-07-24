###############################################################################
# sTiles Makefile
#
# This Makefile automates the build, testing, cleaning, and installation 
# of the sTiles library and its associated components.
#
# Features:
# - Builds libstiles (static and shared libraries)
# - Generates pkgconfig files for configuration management
# - Installs libraries and headers to a specified directory
# - Cleans up intermediate and final build artifacts
#
# Targets:
# - all:         Builds all required libraries and generates pkgconfig files.
# - clean:       Removes temporary build files.
# - cleanall:    Removes all generated binaries and libraries.
# - install:     Installs libraries, headers, and configuration files.
###############################################################################

# Set project root directory
STILES_DIR := .
include ./Makefile.internal

# Dead-code elimination: compile with per-function sections, let the linker
# strip unreferenced ones (EXPORT_MATRIX etc. are handled in make.inc).
CXXFLAGS += -ffunction-sections -fdata-sections
ifeq ($(PLATFORM),macos)
    LDFLAGS += -Wl,-dead_strip
else
    LDFLAGS += -Wl,--gc-sections
endif

# SCOTCH - sources compiled into libstiles.a; only the compression libs it
# depends on are linked (no -lscotch/-lscotcherr).
##########################################################################################################
SCOTCH_LINK_LIBS := $(SCOTCH_EXTRA_LIBS)
##########################################################################################################

# SuiteSparse - sources compiled into libstiles.a by tools/process/Makefile
# (no cmake sub-build, no archive to link).
##########################################################################################################
SUITESPARSE_LIB :=
SUITESPARSE_LINK_LIBS :=

##########################################################################################################
# libxsmm (optional)
##########################################################################################################
LIBXSMM_LIB_DYN := $(LIBXSMM_LIB)
ifeq ($(USE_LIBXSMM),1)
LIBXSMM_INSTALL_DIR := $(TOOLS_DIR)/libxsmm/libxsmm_local
LIBXSMM_INSTALL_SCRIPT := $(TOOLS_DIR)/libxsmm/install_libxsmm_local.sh
# Real-file target so missing libxsmm triggers a rebuild via the install script.
ifneq ($(wildcard $(LIBXSMM_INSTALL_DIR)/lib/libxsmm.a),)
    LIBXSMM_LIB := $(LIBXSMM_INSTALL_DIR)/lib/libxsmm.a
else ifneq ($(wildcard $(LIBXSMM_INSTALL_DIR)/lib64/libxsmm.a),)
    LIBXSMM_LIB := $(LIBXSMM_INSTALL_DIR)/lib64/libxsmm.a
else
    LIBXSMM_LIB := $(LIBXSMM_INSTALL_DIR)/lib/libxsmm.a
endif
ifneq ($(strip $(LIBXSMM_STATIC_LIBS)),)
    LIBXSMM_LINK_LIBS := $(LIBXSMM_STATIC_LIBS) -ldl
    $(info libxsmm: embedding static libraries)
else ifneq ($(strip $(LIBXSMM_LIB_DYN)),)
    LIBXSMM_LINK_LIBS := $(LIBXSMM_LIB_DYN)
    $(info libxsmm: static libs not found, using dynamic linking)
else
    LIBXSMM_LINK_LIBS :=
    $(info libxsmm: not found — selinv will use cblas_dgemm fallback)
endif

# Build libxsmm if not present
$(LIBXSMM_LIB):
	@echo ""
	@echo "================================================================================"
	@echo "libxsmm not found. Building from GitHub source..."
	@echo "This is a one-time build and may take a few minutes."
	@echo "================================================================================"
	@echo ""
	@chmod +x $(LIBXSMM_INSTALL_SCRIPT) && $(LIBXSMM_INSTALL_SCRIPT)
else
LIBXSMM_LIB :=
LIBXSMM_LINK_LIBS :=
endif

###############################################################################
# LDFLAGS for shared library (excludes dynamic SCOTCH/SuiteSparse to allow
# static embedding via --whole-archive)
###############################################################################
# Prefer static MKL embedding when archives are available — keeps sTiles'
# MKL (sequential) isolated from any consumer-loaded MKL with a different
# threading layer. Falls back to dynamic LIBLAPACK otherwise.
ifneq ($(strip $(LAPACKE_STATIC_LIBS)),)
    LIBLAPACK_LINK := $(LAPACKE_STATIC_LIBS) -lpthread -lm -ldl
    $(info LAPACK: embedding static BLAS archives ($(BLAS_DETECTED)) into libstiles.so)
else
    LIBLAPACK_LINK := $(LIBLAPACK)
endif

# Prefer PIC static archives for numa/hwloc so the .so embeds them and
# stays loadable on machines that lack the shared libs. Each *_STATIC_LIB is set
# in make.inc only when the archive is PIC; otherwise fall back to dynamic.
# After the version script localizes them, the embedded symbols can't clash.
ifneq ($(strip $(NUMA_STATIC_LIB)),)
    NUMA_LINK := $(NUMA_STATIC_LIB)
else
    NUMA_LINK := $(NUMA_LIB)
endif
ifneq ($(strip $(HWLOC_STATIC_LIB)),)
    HWLOC_LINK := $(HWLOC_STATIC_LIB) $(HWLOC_STATIC_EXTRA)
else
    HWLOC_LINK := $(HWLOC_LIB)
endif
# gfortran (pulled in by embedded MKL/AMD Fortran): link its archive statically
# on Linux standalone builds so libgfortran.so.5 is not a runtime dependency.
# -static-libgfortran is unreliable with g++ + explicit -lgfortran, so force it
# with -Bstatic around -lgfortran. This only works when libgfortran.a is PIC:
# some toolchains (e.g. gcc-14 on Ubuntu 24.04) ship a non-PIC archive whose
# TLS relocations cannot go into a -shared object, so -Bstatic would fail the
# libstiles.so link. Probe by ACTUALLY test-linking the archive into a
# throwaway .so (arch-neutral: the bad reloc names differ between x86_64
# TPOFF32 and aarch64 TLSLE, so grepping one name misses the other; -shared
# permits undefined symbols, so only a genuine PIC failure rejects it). Fall
# back to dynamic -lgfortran (a libgfortran.so.5 runtime dep) when the archive
# is not embeddable. The version script localizes the embedded gfortran
# symbols when the static path is used.
ifeq ($(STANDALONE_SO)$(PLATFORM),1linux)
    GFORTRAN_LINK := $(shell \
        a=$$($(CXX_BASE) -print-file-name=libgfortran.a 2>/dev/null); \
        t=$$(mktemp); \
        if [ -f "$$a" ] && $(CXX_BASE) -fPIC -shared -o "$$t" \
            -Wl,--whole-archive "$$a" -Wl,--no-whole-archive >/dev/null 2>&1; then \
            echo '-Wl,-Bstatic -lgfortran -Wl,-Bdynamic'; \
        else \
            echo '-lgfortran'; \
        fi; rm -f "$$t")
else ifeq ($(PLATFORM),macos)
    # clang++ has no gfortran runtime in its default search path; locate it via
    # gfortran itself (Homebrew gcc) and bake an rpath. If gfortran is missing,
    # -print-file-name echoes the bare name back ($(dir ...) = ./) and we fall
    # through to plain -lgfortran.
    GFORTRAN_DYLIB := $(shell $(FC) -print-file-name=libgfortran.dylib 2>/dev/null)
    GFORTRAN_LIBDIR := $(dir $(GFORTRAN_DYLIB))
    GFORTRAN_LINK := -lgfortran
    ifneq ($(strip $(GFORTRAN_DYLIB)),)
    ifneq ($(GFORTRAN_LIBDIR),./)
        GFORTRAN_LINK := -L$(GFORTRAN_LIBDIR) -Wl,-rpath,$(GFORTRAN_LIBDIR) -lgfortran
    endif
    endif
else
    GFORTRAN_LINK := -lgfortran
endif

# Full-static C++ runtime: embed libstdc++ and libgcc so libstiles.so carries no
# GLIBCXX_* / libgcc_s runtime floor. After the glibc floor itself (set by the
# build container), this is the biggest portability lever: it removes the exact
# `GLIBCXX_3.4.3x' / libstdc++ mismatch that breaks cross-distro deployment.
# Safe for libstiles specifically: its public ABI is extern "C" and stiles.map
# localizes every embedded C++ symbol, so no C++ object or exception crosses the
# boundary into a host that already has its own libstdc++. Default ON for Linux
# standalone builds; STILES_STATIC_CXX=0 keeps libstdc++.so.6 a dynamic dep.
# (libgomp stays dynamic: its .a is commonly non-PIC, and libgomp.so.1 is a
# universal, low-glibc gcc-runtime lib, so it is not a portability concern.)
STILES_STATIC_CXX ?= $(if $(filter linux,$(PLATFORM)),$(STANDALONE_SO),0)
ifeq ($(STILES_STATIC_CXX),1)
    STATIC_CXX_LINK := -static-libstdc++ -static-libgcc
endif

LDFLAGS_SHARED := $(SANITIZE_LDFLAGS) $(OPENMP_LIBS) $(NUMA_LINK) \
                  $(HWLOC_LINK) $(CUDA_LIB) $(LIBLAPACK_LINK) $(GFORTRAN_LINK) \
                  $(STATIC_CXX_LINK)

# macOS: fold OpenBLAS + LAPACKE + libomp + hwloc + gfortran INTO the single
# libstiles.dylib (the macOS analog of the Linux .so embedding MKL). Each dep's
# STATIC archive is force-loaded, so the resulting ONE file references only
# system libraries and runs on any Apple-Silicon Mac with no Homebrew. Homebrew
# ships these .a on the build machine; the shipped dylib needs none of them.
# Only engages when the full set of archives is present; otherwise the dynamic
# LDFLAGS_SHARED above is kept and the CI dylibbundler step bundles the deps as
# siblings instead (still Homebrew-free, just not a single file).
# Only when the build actually resolved to OpenBLAS: force-loading
# libopenblas.a into a dylib whose code calls a DIFFERENT backend (Accelerate/
# ARMPL) would embed a dead second BLAS and invite duplicate-symbol clashes.
ifeq ($(PLATFORM)$(BLAS_DETECTED),macosopenblas)
    MACOS_BREW  := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
    # Wildcards catch the versioned/plain archive names Homebrew uses
    # (e.g. libopenblas.a or libopenblasp-r0.3.33.a). LAPACKE is taken from
    # OpenBLAS's OWN static archive (OpenBLAS builds the LAPACKE C interface in),
    # so we do NOT depend on Homebrew's `lapack` keg, which ships dylibs only
    # (no liblapacke.a — that missing archive is what defeated the first embed).
    _OMP_A      := $(firstword $(wildcard $(MACOS_BREW)/opt/libomp/lib/libomp*.a))
    _OPENBLAS_A := $(firstword $(wildcard $(MACOS_BREW)/opt/openblas/lib/libopenblas*.a))
    _HWLOC_A    := $(firstword $(wildcard $(MACOS_BREW)/opt/hwloc/lib/libhwloc*.a))
    _GFORTRAN_A := $(wildcard $(shell $(FC) -print-file-name=libgfortran.a 2>/dev/null))
    _QUADMATH_A := $(wildcard $(shell $(FC) -print-file-name=libquadmath.a 2>/dev/null))
    _GCC_A      := $(wildcard $(shell $(FC) -print-file-name=libgcc.a 2>/dev/null))
    # One line so the CI log records exactly what was / wasn't found.
    $(info macOS embed probe: omp=[$(_OMP_A)] openblas=[$(_OPENBLAS_A)] hwloc=[$(_HWLOC_A)] gfortran=[$(_GFORTRAN_A)] quadmath=[$(_QUADMATH_A)] gcc=[$(_GCC_A)])
    # $(and ...) is empty unless every critical archive was found.
    ifneq ($(and $(_OMP_A),$(_OPENBLAS_A),$(_HWLOC_A),$(_GFORTRAN_A)),)
        MACOS_EMBED_ARCHIVES := $(_OMP_A) $(_OPENBLAS_A) $(_HWLOC_A) \
                                $(_GFORTRAN_A) $(_QUADMATH_A)
        MACOS_EMBED := $(foreach a,$(MACOS_EMBED_ARCHIVES),-Wl,-force_load,$(a))
        # Transitive static-link deps of the force-loaded archives:
        #  * libgcc.a as a PLAIN archive (not force_load): resolves the soft-float
        #    long-double helpers gfortran/quadmath reference (__unordtf2, __*tf2)
        #    that clang's compiler-rt lacks on arm64. Plain = pull only undefined
        #    symbols, so no duplicate-symbol clash with compiler-rt.
        #  * -lxml2: hwloc.a needs it. Frameworks: hwloc's macOS topology backends
        #    use IOKit + CoreFoundation, and its GPU discovery uses OpenCL.
        LDFLAGS_SHARED := $(SANITIZE_LDFLAGS) $(NUMA_LINK) $(CUDA_LIB) \
                          $(MACOS_EMBED) $(_GCC_A) -lxml2 -lpthread -lm \
                          -framework OpenCL -framework CoreFoundation -framework IOKit
        $(info macOS: embedding static archives into libstiles.dylib (single self-contained file): $(notdir $(MACOS_EMBED_ARCHIVES)))
    else
        $(info macOS: not all static archives present; keeping dynamic deps (dylibbundler will bundle them as siblings))
    endif
endif

# Version script restricts the libstiles.so dynamic symbol table to the public
# sTiles_*/STILES_* API. Internal symbols (and embedded MKL/SCOTCH/etc.
# symbols) become local, preventing runtime clashes when sTiles is loaded
# into a process that already has its own MKL.
STILES_VERSION_SCRIPT := $(STILES_DIR)/stiles.map
# binutils honors --version-script for PE too: symbols under `local:` are not
# exported. On Windows this is REQUIRED, not just hygiene — without it the DLL
# auto-exports all the vendored guts, and a consumer .exe auto-imports a stray
# DATA symbol from it, hitting MinGW's 32-bit pseudo-reloc overflow at load.
ifneq ($(filter linux windows,$(PLATFORM)),)
    ifneq ($(wildcard $(STILES_VERSION_SCRIPT)),)
        SHARED_VERSION_FLAGS := -Wl,--version-script=$(STILES_VERSION_SCRIPT)
    endif
endif
##########################################################################################################



# METIS detection & flags live in make.inc. METIS + GKlib are compiled directly
# into libstiles (tools/process/Makefile) unless a system METIS is detected, in
# which case -lmetis is linked instead.
ifneq ($(strip $(METIS_SYS_PREFIX)),)
  $(info METIS: using system install at $(METIS_SYS_PREFIX))
else
  $(info METIS: compiled into libstiles (vendored sources))
endif

# Detect optional fast RCM build (set via make.inc CXXFLAGS)
FAST_RCM_ENABLED := $(findstring -DSTILES_FAST_RCM,$(CXXFLAGS))

# SCOTCH is compiled directly into libstiles (no sub-build, no library to link).
###############################################################################
SCOTCH_LIB :=
###############################################################################

# GPU support (conditionally enabled)
ifeq ($(STILES_GPU),1)
  GPU_BUILD_CMD := (cd tools/gpu && $(MAKE))
  GPU_CLEAN_CMD := (cd tools/gpu && $(MAKE) cleanall)
else
  GPU_BUILD_CMD :=
  GPU_CLEAN_CMD :=
endif

# Define libraries
LIBS := libstiles tileindexer

# Single source of truth for the module layout:
#   BUILD_MODULES – sub-makes run by the `libstiles` target (tile is built via
#                   densetile, gpu only when STILES_GPU=1)
#   ARCHIVE_DIRS  – directories whose top-level *.o go into lib/libstiles.a
BUILD_MODULES := tools/control tools/compute tools/memory tools/symbolic \
                 tools/sort tools/free tools/sparse tools/process
ARCHIVE_DIRS  := tools/control tools/process tools/free tools/common tools/sparse \
                 tools/compute tools/memory tools/ordering tools/tile \
                 tools/symbolic tools/sort
ifeq ($(STILES_GPU),1)
    ARCHIVE_DIRS += tools/gpu
endif
# Superset used by the clean targets (always includes gpu/TileIndexer)
CLEAN_OBJ_DIRS := $(sort $(ARCHIVE_DIRS) tools/gpu tools/TileIndexer)

# Archive every module object into lib/libstiles.a (shared by `libstiles`
# and `relink` so the object list cannot drift between them).
define ARCHIVE_LIBSTILES
@rm -f lib/libstiles.a; \
objs=`find $(ARCHIVE_DIRS) -maxdepth 1 -type f -name '*.o' 2>/dev/null`; \
if [ -z "$${objs}" ]; then \
  echo "[!] No object files found - run 'make all' first"; exit 1; \
else \
  ar rcs lib/libstiles.a $${objs}; \
  ranlib lib/libstiles.a; \
fi
endef

.DEFAULT_GOAL := all

# Two-phase build: phase 1 builds libxsmm if missing (the one remaining
# sub-built dependency; SCOTCH/METIS/SuiteSparse are compiled inline), phase 2
# re-invokes make so make.inc re-parses with the installed archive visible and
# LIBXSMM_STATIC_LIBS gets embedded into libstiles.so.
all:
	@$(MAKE) --no-print-directory deps
	@$(MAKE) --no-print-directory build

deps: $(LIBXSMM_LIB)

# Fast edit loop: after a source/header edit, `make` recompiles only what
# changed (.d dependency tracking) and relinks libstiles.{a,so}. The
# standalone archive is a distribution artifact whose LTO partial-link takes
# minutes — build it on demand with `make standalone`.
build: lib libstiles_shared

.PHONY: all deps build

# Fully portable, redistributable CLI bundle (any x86-64 Linux, glibc 2.27 floor).
# NOT a host build: a binary's glibc floor equals the glibc of the machine it is
# compiled on, so this builds INSIDE an old-glibc (2.28) manylinux container,
# rootless, via udocker. One command; rebuild any time after source changes.
# Prereqs: udocker (pip install --user udocker), Intel oneAPI MKL, and a
# new-GLIBCXX + low-glibc libstdc++ (conda-forge libstdcxx-ng). See tools/package/.
.PHONY: portable-bundle
portable-bundle:
	@bash tools/package/build_portable_bundle.sh

###############################################################################
# Standalone static library — a single, self-contained libstiles_standalone.a
#
# The normal lib/libstiles.a holds ONLY sTiles' own objects, so a consumer must
# also supply METIS/SCOTCH/SuiteSparse/libxsmm, and those dependency symbols
# stay global (the stiles.map version script only protects the .so, never a
# .a). A consumer that *also* links a different METIS then clashes — the exact
# problem this target removes.
#
# How it works: a partial link (-r) merges sTiles + tileindexer + the bundled
# dependency archives into ONE relocatable object, with the LTO plugin/libLTO
# compiling everything down to real machine code, then we localize every symbol
# except the public sTiles_*/STILES_* API — the same contract stiles.map
# enforces for the .so. Linux uses g++ -r + objcopy --keep-global-symbol;
# macOS uses ld -r + -exported_symbols_list (see the recipe below). The result:
#   * embeds METIS/GKlib/SCOTCH/SuiteSparse/libxsmm, so NO -lmetis/-lscotch/...
#     is needed by the consumer, and a different copy on the machine cannot bind;
#   * carries NO LTO bytecode, so a consumer does not need a matching LTO
#     toolchain (more portable than libstiles.a itself).
# It still relies on the system runtime: libstdc++/gfortran/gomp/m/dl/pthread,
# z/bz2/lzma (SCOTCH), and numa/hwloc, plus a BLAS (see MKL note).
#
# Two LTO details that make this work and are easy to get wrong:
#   * -flto-partition=none avoids an lto1 "add_symbol_to_partition_1" ICE that
#     g++ -r otherwise hits on this object set.
#   * objcopy --keep-global-symbol localizes within a single object, where the
#     internal references are already resolved by the -r link. Doing it on a
#     plain ar-merged archive instead would make the symbols local *per member*
#     and break cross-object references (and corrupt the slim-LTO members).
#
# MKL is the BLAS backend and is NOT embedded by default: it is ~610 MB static
# and consumers normally link their own BLAS. Set STANDALONE_EMBED_MKL=1 to bake
# the static MKL archives in too (yields a ~670 MB fully self-contained .a).
#
# Pull the bare *.a paths out of the *_STATIC_LIBS lists (which also carry
# -Wl,--whole-archive / --start-group tokens) with $(filter %.a,...).
STANDALONE_EMBED_MKL ?= 0
STANDALONE_DEP_ARCHIVES := \
    $(filter %.a,$(METIS_LINK_FLAGS_LINUX)) \
    $(filter %.a,$(SUITESPARSE_STATIC_LIBS)) \
    $(filter %.a,$(SCOTCH_STATIC_LIBS)) \
    $(filter %.a,$(LIBXSMM_STATIC_LIBS))
ifeq ($(STANDALONE_EMBED_MKL),1)
    STANDALONE_DEP_ARCHIVES += $(filter %.a,$(LAPACKE_STATIC_LIBS))
endif

STANDALONE_LIB := lib/libstiles_standalone.a
STANDALONE_OBJ := lib/libstiles_standalone.o
STANDALONE_EXP_LIST := lib/.standalone_exported.syms

.PHONY: standalone
standalone: lib libstiles tileindexer $(LIBXSMM_LIB)
ifeq ($(PLATFORM),macos)
	@echo "Building $(STANDALONE_LIB) (ld -r merge + exported-symbols-list localize, macOS)"
	@rm -f $(STANDALONE_LIB) $(STANDALONE_OBJ) $(STANDALONE_EXP_LIST)
	@# macOS/Mach-O analogue of the Linux recipe. ld64 has no objcopy; instead a
	@# relocatable (-r) link with -exported_symbols_list keeps ONLY the matching
	@# symbols global and demotes everything else to "private extern" (local) —
	@# the classic single-object-prelink trick. C symbols carry a leading '_' on
	@# Mach-O, so the API patterns are _sTiles_* / _STILES_*. -force_load is the
	@# -Wl,--whole-archive equivalent and pulls the full sTiles API in. The -r
	@# link also runs libLTO so the slim-LTO sTiles objects become real code.
	@printf '_sTiles_*\n_STILES_*\n' > $(STANDALONE_EXP_LIST)
	$(CXX) -nostdlib -Wl,-r $(OPT_FLAGS) $(OPENMP_FLAGS) \
	    -o $(STANDALONE_OBJ) \
	    -Wl,-exported_symbols_list,$(STANDALONE_EXP_LIST) \
	    -Wl,-force_load,lib/libstiles.a -Wl,-force_load,tools/libs/libtileindexer.a \
	    $(STANDALONE_DEP_ARCHIVES)
	@libtool -static -o $(STANDALONE_LIB) $(STANDALONE_OBJ)
	@rm -f $(STANDALONE_OBJ) $(STANDALONE_EXP_LIST)
	@leaks=$$(nm $(STANDALONE_LIB) 2>/dev/null | grep -E ' T ' | grep -vcE ' T _(sTiles_|STILES_)'); \
	 echo "  -> $(STANDALONE_LIB) ($$(du -h $(STANDALONE_LIB) | cut -f1)); non-API global symbols: $$leaks; embeds:$(if $(STANDALONE_DEP_ARCHIVES), $(notdir $(STANDALONE_DEP_ARCHIVES)),none)"
else
	@echo "Building $(STANDALONE_LIB) (partial-link merge + objcopy localize)"
	@rm -f $(STANDALONE_LIB) $(STANDALONE_OBJ)
	@# 1. Merge sTiles + tileindexer + bundled deps into one real-code object.
	@#    --whole-archive keeps the full sTiles API; deps are pulled on demand.
	@#    --start-group resolves circular refs (matters when MKL is embedded).
	$(CXX) -r $(OPT_FLAGS) -flto-partition=none -fno-fat-lto-objects $(OPENMP_FLAGS) \
	    -o $(STANDALONE_OBJ) \
	    -Wl,--whole-archive lib/libstiles.a tools/libs/libtileindexer.a -Wl,--no-whole-archive \
	    -Wl,--start-group $(STANDALONE_DEP_ARCHIVES) -Wl,--end-group
	@# 2. Localize everything except the public API (same contract as stiles.map).
	@objcopy --wildcard \
	    --keep-global-symbol 'sTiles_*' \
	    --keep-global-symbol 'STILES_*' \
	    $(STANDALONE_OBJ)
	@# 3. Archive the single hidden object.
	@ar rcs $(STANDALONE_LIB) $(STANDALONE_OBJ)
	@ranlib $(STANDALONE_LIB)
	@rm -f $(STANDALONE_OBJ)
	@leaks=$$(nm $(STANDALONE_LIB) 2>/dev/null | grep -E ' T ' | grep -vcE ' T (sTiles_|STILES_)'); \
	 echo "  -> $(STANDALONE_LIB) ($$(du -h $(STANDALONE_LIB) | cut -f1)); non-API global symbols: $$leaks; embeds:$(if $(STANDALONE_DEP_ARCHIVES), $(notdir $(STANDALONE_DEP_ARCHIVES)),none)"
endif

lib: ${LIBS} pkgconfig

libstiles_shared: lib-dir libstiles tileindexer
ifeq ($(PLATFORM),macos)
	@echo "Creating shared library libstiles.dylib (macOS)"
	$(CXX) -dynamiclib -install_name @rpath/libstiles.dylib \
	    -o lib/libstiles.dylib \
	    -Wl,-force_load,lib/libstiles.a \
	    -Wl,-force_load,tools/libs/libtileindexer.a \
	    $(METIS_LINK_FLAGS_MACOS) \
	    $(SUITESPARSE_LINK_LIBS) $(SCOTCH_LINK_LIBS) $(LIBXSMM_LINK_LIBS) $(CXXFLAGS) $(LDFLAGS_SHARED)
	@# Create .so symlink for compatibility
	@ln -sf libstiles.dylib lib/libstiles.so
else ifeq ($(PLATFORM),windows)
	@echo "Creating shared library libstiles.dll (Windows/MinGW)"
	$(CXX) -shared -o lib/libstiles.dll \
	    -Wl,--out-implib,lib/libstiles.dll.a \
	    $(SHARED_VERSION_FLAGS) \
	    -Wl,--whole-archive lib/libstiles.a -Wl,--no-whole-archive \
	    tools/libs/libtileindexer.a $(METIS_LINK_FLAGS_LINUX) \
	    $(SUITESPARSE_LINK_LIBS) $(SCOTCH_LINK_LIBS) $(LIBXSMM_LINK_LIBS) $(CXXFLAGS) $(LDFLAGS_SHARED)
else
	@echo "Creating shared library libstiles.so (Linux)"
	$(CXX) -shared -o lib/libstiles.so \
	    $(SHARED_VERSION_FLAGS) \
	    -Wl,--whole-archive lib/libstiles.a -Wl,--no-whole-archive \
	    tools/libs/libtileindexer.a $(METIS_LINK_FLAGS_LINUX) \
	    $(SUITESPARSE_LINK_LIBS) $(SCOTCH_LINK_LIBS) $(LIBXSMM_LINK_LIBS) $(CXXFLAGS) $(LDFLAGS_SHARED)
endif

# tile (core BLAS wrappers + metadata + preprocess) - objects included in libstiles.a
densetile:
	(cd tools/tile && $(MAKE))
	$(GPU_BUILD_CMD)

lib-dir:
	mkdir -p lib

# Building libstiles static library (includes all core BLAS wrappers)
libstiles: lib-dir densetile tileindexer $(LIBXSMM_LIB)
	@for d in $(BUILD_MODULES); do $(MAKE) -C $$d || exit 1; done
	@# Build fast RCM object (tools path) only when requested
	@if [ -n "$(FAST_RCM_ENABLED)" ] && [ -f "tools/ordering/ordering_rcm_fast.cpp" ]; then \
	  echo "Compiling tools/ordering/ordering_rcm_fast.cpp"; \
	  $(CXX) $(CXXFLAGS) -c tools/ordering/ordering_rcm_fast.cpp -o tools/ordering/ordering_rcm_fast.o; \
	fi
	@echo "Archiving objects into lib/libstiles.a"
	$(ARCHIVE_LIBSTILES)

# Fast relink - just rebuild libraries from existing .o files (use after editing source)
# Usage: make relink  OR  make relink STILES_GPU=1
.PHONY: relink
relink: lib-dir
	@echo "Relinking libraries from existing .o files..."
	@rm -f lib/libstiles.a lib/libstiles.so lib/libstiles.dylib
	@echo "Archiving objects into lib/libstiles.a"
	$(ARCHIVE_LIBSTILES)
	@$(MAKE) --no-print-directory libstiles_shared
	@echo "Relink complete!"

# Cleanup targets
.PHONY: cleanup quick-clean

# Quick clean - just remove .o/.d files from main folders (fastest way to force recompile)
# Usage: make quick-clean && make STILES_GPU=1 all
quick-clean:
	@echo "Removing .o files from main folders..."
	@rm -f $(addsuffix /*.o,$(CLEAN_OBJ_DIRS)) $(addsuffix /*.d,$(CLEAN_OBJ_DIRS))
	@rm -f lib/libstiles.a lib/libstiles.so lib/libstiles.dylib lib/libstiles_standalone.a lib/libstiles_standalone.o lib/.standalone_exported.syms .standalone_so.cache
	@rm -f tools/libs/*.a tools/libs/*.so
	@echo "Done. Run 'make STILES_GPU=1 all' to rebuild."

clean:
	@for d in $(BUILD_MODULES) tools/tile; do $(MAKE) -C $$d clean || exit 1; done
	(cd tools/TileIndexer && $(MAKE) LIB_DIR=../libs clean)
	@rm -f lib/libstiles.so lib/libstiles.dylib lib/libstiles.a lib/libstiles_standalone.a lib/libstiles_standalone.o lib/.standalone_exported.syms .standalone_so.cache
	@rm -f lib/pkgconfig/*.pc

cleanup: clean
	@echo "Removing exported fill-in artifacts"
	@rm -rf fillin
	@rm -f matrix_before_factorization.txt matrix_after_factorization.txt

cleanall:
	@for d in $(BUILD_MODULES) tools/tile; do $(MAKE) -C $$d cleanall || exit 1; done
	(cd tools/TileIndexer && $(MAKE) LIB_DIR=../libs clean)
	@rm -f lib/*.a lib/*.so lib/*.dylib
	@rm -f lib/pkgconfig/*.pc
	@rm -f .standalone_so.cache
	$(GPU_CLEAN_CMD)
	@# Remove ALL object/dep files (catch any missed by sub-Makefiles)
	@find tools -name "*.o" -type f -delete 2>/dev/null || true
	@rm -f $(addsuffix /*.d,$(CLEAN_OBJ_DIRS))
	@# Remove tools/libs archives
	@rm -f tools/libs/*.a tools/libs/*.so tools/libs/*.dylib
	@# NOTE: ordering sources are vendored in-tree and compiled into libstiles;
	@# there are no external ordering libs to remove (see clean-ordering-subbuilds).
	@echo "cleanall complete."

# Remove regenerated ordering sub-build artifacts (the _local installs + cmake
# build dirs produced by the tools/.../install_*_local.sh regeneration scripts).
# The vendored ordering SOURCE trees are NEVER touched -- they are required by
# the inline build that compiles them into libstiles.
.PHONY: clean-ordering-subbuilds
clean-ordering-subbuilds:
	@echo "Removing regenerated ordering sub-build artifacts (sources preserved)..."
	@rm -rf $(TOOLS_DIR)/suitesparse/suitesparse_local $(TOOLS_DIR)/suitesparse/SuiteSparse/build
	@rm -rf $(TOOLS_DIR)/ordering/metis-5.1.0/local $(TOOLS_DIR)/ordering/metis-5.1.0/build
	@rm -rf $(TOOLS_DIR)/ordering/GKlib/local $(TOOLS_DIR)/ordering/GKlib/build
	@rm -rf $(TOOLS_DIR)/ordering/scotch/scotch_local $(TOOLS_DIR)/ordering/scotch/scotch/build

# Nuclear option: remove objects, libs, dist (ordering sources are vendored).
.PHONY: full-clean
full-clean: cleanall dist-clean
	@echo ""
	@echo "full-clean complete (objects, libraries, dist)."


tileindexer:
	(cd tools/TileIndexer && \
		$(MAKE) CXXFLAGS="$(CXXFLAGS)" LDFLAGS="$(LDFLAGS)" SHLDFLAGS="$(SHARED_LDFLAGS)" \
		LIB_DIR=../libs ../libs/libtileindexer.a ../libs/libtileindexer.so)

###############################################################################
# Installation & pkg-config

# $(call write_pc,<prefix>,<includedir>,<outfile>,<description>)
# Writes a complete pkg-config file; $${...} stays a literal ${...} in the file.
ifeq ($(PLATFORM),macos)
    PC_BLAS_EXTRA := -framework Accelerate
else
    PC_BLAS_EXTRA :=
endif
define write_pc
{ echo 'prefix=$(1)'; \
  echo 'libdir=$${prefix}/lib'; \
  echo 'includedir=$(2)'; \
  echo ''; \
  echo 'Name: stiles'; \
  echo 'Description: $(4)'; \
  echo 'Version: $(DIST_VERSION)'; \
  echo 'Libs: -L$${libdir} -lstiles $(PC_BLAS_EXTRA)'; \
  echo 'Cflags: -I$${includedir}'; } > $(3)
endef

# In-tree pkg-config file pointing at this checkout, for local consumers:
#   PKG_CONFIG_PATH=<repo>/lib/pkgconfig pkg-config --libs stiles
pkgconfig:
	@mkdir -p lib/pkgconfig
	@$(call write_pc,$(PROJECT_ROOT),$${prefix}/tools/include,lib/pkgconfig/stiles.pc,Sparse Tiled Linear Algebra Library)

# Install to PREFIX (default /usr/local; DESTDIR honored):
#   make install PREFIX=$$HOME/.local
PREFIX ?= /usr/local
.PHONY: pkgconfig install
install: all
	install -d $(DESTDIR)$(PREFIX)/lib/pkgconfig $(DESTDIR)$(PREFIX)/include
	install -m 644 tools/include/stiles.h $(DESTDIR)$(PREFIX)/include/
ifeq ($(PLATFORM),macos)
	install -m 755 lib/libstiles.dylib $(DESTDIR)$(PREFIX)/lib/
	ln -sf libstiles.dylib $(DESTDIR)$(PREFIX)/lib/libstiles.so
else
	install -m 755 lib/libstiles.so $(DESTDIR)$(PREFIX)/lib/
endif
	install -m 644 lib/libstiles.a $(DESTDIR)$(PREFIX)/lib/
	@$(call write_pc,$(PREFIX),$${prefix}/include,$(DESTDIR)$(PREFIX)/lib/pkgconfig/stiles.pc,Sparse Tiled Linear Algebra Library)
	@echo "Installed sTiles to $(DESTDIR)$(PREFIX)"

###############################################################################
# Distribution target - creates a self-contained binary distribution
#
# The CPU target is the MARCH knob (make.inc), NOT a per-arch dist target:
#   make dist                 # package the current build (portable AVX2)
#   make dist MARCH=avx512    # AVX-512 servers        \  run from a clean tree
#   make dist MARCH=native    # this-CPU tuned          }  (MARCH change must
#   make dist MARCH=generic   # any x86-64             /   recompile sTiles)
#   make dist-all             # Linux: build all 3 x86_64 variants at once
###############################################################################
DIST_DIR ?= dist
DIST_VERSION ?= $(shell date +%Y%m%d)

.PHONY: dist dist-all dist-clean

# Build (honoring MARCH) then package. DIST_ARCH labels the .pc/description.
dist: all
	@$(MAKE) --no-print-directory _dist_package DIST_ARCH="$(MARCH)"

# Build every portable x86_64 variant into dist/<variant>/, cleaning between
# each (a MARCH change requires recompiling sTiles' own objects). On macOS
# there is only the one native arch, so this just packages it — use CI
# (.github/workflows/build.yml) for Intel + Apple-Silicon Mac binaries.
dist-all:
ifeq ($(PLATFORM),macos)
	@$(MAKE) --no-print-directory dist DIST_DIR=dist/native
	@echo "macOS: packaged native build in dist/native/ (cross-arch → use CI)."
else
	@for m in generic avx2 avx512; do \
	  echo "=== dist variant: $$m ==="; \
	  $(MAKE) --no-print-directory clean; \
	  $(MAKE) --no-print-directory dist MARCH=$$m DIST_DIR=dist/$$m || exit 1; \
	done
	@echo "Built dist/{generic,avx2,avx512}/"
endif

# Internal target to package the distribution (platform-aware)
_dist_package:
	@echo "================================================================================"
	@echo "Creating $(DIST_ARCH) distribution in $(DIST_DIR)/ ($(PLATFORM))"
	@echo "================================================================================"
	@mkdir -p $(DIST_DIR)/lib $(DIST_DIR)/include $(DIST_DIR)/lib/pkgconfig
ifeq ($(PLATFORM),macos)
	@# macOS: Copy dylib and strip debug symbols
	@cp lib/libstiles.dylib $(DIST_DIR)/lib/
	@strip -x $(DIST_DIR)/lib/libstiles.dylib
	@# Fix install name for redistribution
	@install_name_tool -id @rpath/libstiles.dylib $(DIST_DIR)/lib/libstiles.dylib
	@# Remove build-specific rpaths
	@# Create .so symlink for compatibility
	@ln -sf libstiles.dylib $(DIST_DIR)/lib/libstiles.so
else
	@# Linux: Copy .so and strip debug symbols
	@cp lib/libstiles.so $(DIST_DIR)/lib/
	@strip --strip-unneeded $(DIST_DIR)/lib/libstiles.so
	@# Remove build-specific rpaths (SCOTCH/SuiteSparse are now embedded)
	@if command -v patchelf >/dev/null 2>&1; then \
		patchelf --remove-rpath $(DIST_DIR)/lib/libstiles.so 2>/dev/null || true; \
		echo "Removed build rpaths with patchelf"; \
	elif command -v chrpath >/dev/null 2>&1; then \
		chrpath -d $(DIST_DIR)/lib/libstiles.so 2>/dev/null || true; \
		echo "Removed build rpaths with chrpath"; \
	else \
		echo "WARNING: patchelf/chrpath not found - rpaths not removed"; \
		echo "Install with: sudo apt install patchelf  OR  sudo apt install chrpath"; \
	fi
endif
	@# Copy headers
	@cp tools/include/stiles.h $(DIST_DIR)/include/
	@# Copy examples if the directory exists (optional)
	@if [ -d examples ]; then \
		mkdir -p $(DIST_DIR)/examples; \
		cp examples/*.c examples/*.cpp $(DIST_DIR)/examples/ 2>/dev/null || true; \
		cp examples/Makefile examples/README.md $(DIST_DIR)/examples/ 2>/dev/null || true; \
	fi
	@# Create distribution pkg-config file (no build paths)
	@$(call write_pc,/usr/local,$${prefix}/include,$(DIST_DIR)/lib/pkgconfig/stiles.pc,Sparse Tiled Linear Algebra Library ($(DIST_ARCH)))
	@echo ""
	@echo "================================================================================"
	@echo "Distribution ($(DIST_ARCH)) created in $(DIST_DIR)/"
	@echo "================================================================================"
	@echo "Contents:"
	@echo "  $(DIST_DIR)/"
	@echo "  ├── include/stiles.h"
ifeq ($(PLATFORM),macos)
	@echo "  ├── lib/libstiles.dylib"
	@echo "  ├── lib/libstiles.so -> libstiles.dylib"
else
	@echo "  ├── lib/libstiles.so"
endif
	@echo "  └── lib/pkgconfig/stiles.pc"
	@echo ""
ifeq ($(PLATFORM),macos)
	@ls -lh $(DIST_DIR)/lib/libstiles.dylib
else
	@ls -lh $(DIST_DIR)/lib/libstiles.so
endif
	@echo ""
	@echo "Dependencies:"
ifeq ($(PLATFORM),macos)
	@otool -L $(DIST_DIR)/lib/libstiles.dylib | head -15
else
	@ldd $(DIST_DIR)/lib/libstiles.so | grep -v "linux-vdso" | head -10
endif
	@echo "================================================================================"

dist-clean:
	@rm -rf dist
	@echo "Distribution directory removed"

###############################################################################
# Help target
###############################################################################
.PHONY: help
help:
	@echo "================================================================================"
	@echo "sTiles Makefile - Build System for Sparse Tiled Linear Algebra Library"
	@echo "================================================================================"
	@echo ""
	@echo "Platform: $(PLATFORM) ($(UNAME_M))"
	@echo ""
	@echo "Main Targets:"
	@echo "  all              - Build libstiles.a + shared lib (portable-optimized AVX2)"
	@echo "                     MARCH=avx512  → AVX-512 servers (not this laptop)"
	@echo "                     MARCH=native  → tune for THIS CPU (benchmark runs)"
	@echo "                     MARCH=generic → any x86-64 (max compatibility)"
	@echo "  standalone       - Self-contained libstiles_standalone.a (slow LTO link)"
	@echo "  lib              - Build static libraries only"
ifeq ($(PLATFORM),macos)
	@echo "  libstiles_shared - Build shared library (libstiles.dylib)"
else
	@echo "  libstiles_shared - Build shared library (libstiles.so)"
endif
	@echo "  install          - Install libraries and headers"
	@echo ""
	@echo "Distribution Targets (CPU target = MARCH, see Main Targets):"
	@echo "  dist             - Package the current build into dist/"
	@echo "  dist MARCH=avx512 - AVX-512 variant (from a clean tree)"
ifneq ($(PLATFORM),macos)
	@echo "  dist-all         - Build all 3 x86_64 variants (generic/avx2/avx512)"
endif
	@echo "  portable-bundle  - Redistributable CLI tarball (Linux, glibc 2.27 floor)"
	@echo ""
	@echo "Vendored Dependencies (compiled into libstiles automatically):"
	@echo "  SCOTCH           - Graph ordering library"
	@echo "  METIS + GKlib    - Nested dissection ordering"
	@echo "  SuiteSparse      - AMD/COLAMD family (disable with USE_SUITESPARSE=0)"
	@echo "  libxsmm          - JIT small-GEMM kernels (built once on first make)"
	@echo ""
	@echo "Testing / benchmarks:"
	@echo "  cd run/ && make  - Build bench + check_* programs (see run/Makefile help)"
	@echo ""
	@echo "Rebuild helpers:"
	@echo "  relink           - Re-archive + relink libraries from existing .o files"
	@echo "  quick-clean      - Remove module .o/.d files only (fast forced recompile)"
	@echo ""
	@echo "Cleaning:"
	@echo "  clean            - Remove object files and libraries"
	@echo "  cleanall         - Remove all generated files"
	@echo "  cleanup          - Clean + remove test artifacts"
	@echo "  clean-ordering-subbuilds - Remove regenerated _local ordering artifacts"
	@echo "  dist-clean       - Remove distribution directory"
	@echo "  full-clean       - Remove absolutely everything (objects, libs, dist)"
	@echo ""
	@echo "Install:"
	@echo "  install          - Install to PREFIX (default /usr/local; DESTDIR honored)"
	@echo ""
	@echo "Configuration:"
	@echo "  Edit make.inc to configure:"
	@echo "    - Compilers (CC, CXX, FC)"
	@echo "    - Optimization flags (CXXFLAGS, CFLAGS)"
ifeq ($(PLATFORM),macos)
	@echo "    - BLAS: Uses Accelerate (built-in) or OpenBLAS (brew install openblas)"
else
	@echo "    - LAPACK/BLAS paths (MKLROOT, LAPACKE_DIR)"
endif
	@echo "    - Optional features (STILES_GPU)"
	@echo ""
	@echo "Cross-Platform Support:"
	@echo "  - Linux (Ubuntu, Fedora, RHEL, Debian)"
	@echo "  - macOS (Intel and Apple Silicon)"
	@echo ""
	@echo "================================================================================"

# Show build configuration
.PHONY: show-config
show-config:
	@echo "=============================================="
	@echo "sTiles Build Configuration"
	@echo "=============================================="
	@echo "Platform:      $(PLATFORM) ($(UNAME_M))"
	@echo "Build mode:    $(BUILD)"
	@echo "CPU target:    $(MARCH) ($(ARCH_TARGET_FLAG))"
	@echo "C Compiler:    $(CC)"
	@echo "C++ Compiler:  $(CXX)"
	@echo "CUDA:          $(if $(filter 1,$(STILES_GPU)),enabled (arch=$(STILES_CUDA_ARCH)),disabled)"
	@echo "SuiteSparse:   $(if $(filter 1,$(USE_SUITESPARSE)),enabled,disabled)"
	@echo "STILES_MODE:   $(STILES_MODE)"
	@echo "=============================================="

###############################################################################
# cross-syntax : Windows / MinGW portability PROBE  (diagnostic, non-gating)
# ---------------------------------------------------------------------------
# Compile-checks sTiles' OWN C++ sources — the REAL per-module SRC lists (with
# each module's EXTRA_INC), NOT the vendored SCOTCH / METIS / GKlib /
# SuiteSparse / libxsmm which carry their own upstream Windows story — with the
# active compiler + real CXXFLAGS, WITHOUT linking. Run it from an MSYS2 UCRT64
# shell, where CXX is the UCRT64 MinGW-w64 g++ (x86_64-w64-mingw32ucrt-*, the
# standard Rtools44 toolchain), and it reports exactly which sTiles sources
# still assume Linux/POSIX (ungated sched_setaffinity, numa, mmap, dlopen, ...)
# and so need porting for Windows. It does NOT build a .dll and does NOT fix
# anything; it just measures the gap. Errors land in cross-syntax.err; the
# recipe prints a pass/fail tally and never fails the build (so CI can surface
# the whole list rather than stop at the first error).
#
# Coverage = the module.mk-driven modules that hold sTiles' OS-sensitive code
# (control/compute/memory/symbolic/sort/free/sparse/process + tile). GPU
# (STILES_GPU-gated) and TileIndexer (separate lib driver) are out of scope.
###############################################################################
CROSS_MODULES := $(BUILD_MODULES) tools/tile

.PHONY: cross-syntax
cross-syntax:
	@echo "=================================================================="
	@echo "sTiles Windows / MinGW compile probe (syntax-only, no link)"
	@echo "  platform : $(PLATFORM) ($(UNAME_M))"
	@echo "  CXX      : $(CXX)"
	@echo "  modules  : $(CROSS_MODULES)"
	@echo "=================================================================="
	@rm -f cross-syntax.err cross-syntax.log
	@for d in $(CROSS_MODULES); do \
	    echo "-- $$d"; \
	    $(MAKE) --no-print-directory -C $$d syntax SYNTAX_ERR=$(CURDIR)/cross-syntax.err; \
	done | tee cross-syntax.log
	@total=$$(grep -cE '^  (ok|FAIL) ' cross-syntax.log 2>/dev/null); \
	 fail=$$(grep -cE '^  FAIL ' cross-syntax.log 2>/dev/null); \
	 : "$${total:=0}" "$${fail:=0}"; \
	 echo "=================================================================="; \
	 echo "  $$((total-fail))/$$total compiled clean;  $$fail failed"; \
	 echo "  full compiler output: cross-syntax.err"; \
	 echo "=================================================================="
