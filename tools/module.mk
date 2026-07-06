###############################################################################
# Shared rules for sTiles modules (tools/<module>/Makefile)
#
# A module Makefile defines, in this order, then includes this file:
#   STILES_DIR – relative path to the project root (usually ../..)
#   SRC        – .cpp sources (objects land next to them)
#   EXTRA_OBJ  – optional extra objects (e.g. conditional C stubs); define it
#                with '=' when it references make.inc variables — it is only
#                expanded after make.inc has been parsed below
#   EXTRA_INC  – optional module-local include paths (base includes are
#                already embedded in CXXFLAGS/CFLAGS by make.inc)
#
# Compilation emits .d dependency files (-MMD -MP), so header edits trigger
# exactly the rebuilds they need — no cleanall-after-header-edit required.
###############################################################################

include $(STILES_DIR)/Makefile.internal

OBJ := $(SRC:.cpp=.o) $(EXTRA_OBJ)
DEP := $(OBJ:.o=.d)

all: $(OBJ)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(EXTRA_INC) -MMD -MP -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(EXTRA_INC) -MMD -MP -c $< -o $@

clean:
	rm -f $(OBJ) $(DEP) *~

cleanall: clean

# syntax : compile-check this module's real SRC (with its EXTRA_INC) WITHOUT
# codegen or linking. Drives the top-level `cross-syntax` Windows/MinGW
# portability probe, so it sees exactly the sources + include paths the real
# build compiles. Appends compiler stderr to $(SYNTAX_ERR) and prints one
# ok/FAIL line per source; never fails the make (the driver tallies).
SYNTAX_ERR ?= /dev/null
syntax:
	@for s in $(SRC); do \
	    if $(CXX) $(CXXFLAGS) $(EXTRA_INC) -fsyntax-only $$s 2>>$(SYNTAX_ERR); then \
	        echo "  ok    $$s"; \
	    else echo "  FAIL  $$s"; fi; \
	done

-include $(DEP)

.PHONY: all clean cleanall syntax
