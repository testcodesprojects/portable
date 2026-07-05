###############################################################################
# sTiles GPU module
#
# Compiles tools/gpu/**/*.cpp into objects archived into libstiles.a.
# Builds ONLY when STILES_GPU=1 is set (the parent Makefile drives this).
#
# Notes:
#   - dpotrf_dense_gpu.cpp uses CUDA Runtime + cuBLAS + cuSOLVER APIs ONLY
#     (no __global__/__device__ kernels, no <<<>>> launch syntax), so it
#     compiles with g++ + CUDA headers; nvcc is NOT required.
#   - gpu_persistent_context.hpp, gpu_tile_slab.hpp, gh200_unified_alloc.hpp,
#     and gpu_dispatch_plan.hpp are header-only — no .o files to produce.
#   - Each source file is itself guarded by `#ifdef STILES_GPU`, so even
#     when invoked without that flag the result is empty objects (safe).
###############################################################################

STILES_DIR := ../..
include $(STILES_DIR)/Makefile.internal

# Extra includes: this dir for the headers, parent for sibling modules.
INC += -I. -I.. -I../common -I../control -I../compute -I../include

# Sources rooted in subdirectories. Add more here as the GPU module grows.
SRC := dpotrf/dpotrf_dense_gpu.cpp

OBJ := $(SRC:.cpp=.o)

# Default target: build object files. The parent Makefile archives
# tools/gpu/*.o via its ARCHIVE_DIRS find (`-maxdepth 1`), so we surface
# objects at THIS level by copying them up after building from subdirs —
# see the post-build rule.
all: $(OBJ) bubble_up

# Compile each .cpp with the project's flags. CUDA_DEF/CUDA_INC are added
# by Makefile.internal when STILES_GPU=1 is in effect.
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -MMD -MP -c $< -o $@

DEP := $(OBJ:.o=.d)
-include $(DEP)

# Parent Makefile globs `tools/gpu/*.o` at maxdepth 1. Our .o files live
# under dpotrf/, so symlink them up to this directory after compilation.
bubble_up: $(OBJ)
	@# Copy (not symlink) because the parent Makefile's archive step uses
	@# `find … -maxdepth 1 -type f -name '*.o'` which skips symlinks.
	@for o in $(OBJ); do \
	    base=$$(basename $$o); \
	    if [ ! -e $$base ] || [ $$o -nt $$base ]; then \
	        cp -f $$o $$base; \
	    fi; \
	done

clean:
	rm -f $(OBJ) $(DEP) *.o *.d *~

cleanall: clean

.PHONY: all clean cleanall bubble_up
