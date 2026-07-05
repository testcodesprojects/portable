###############################################################################
# sTiles sparse module
#
# Supernodal sparse Cholesky under namespace sTiles::sparse. The internal
# C++ helper surface (sTiles::sparse::api::*) lives in api.{hpp,cpp};
# everything else is internal.
#
# Objects land at tools/sparse/*.o and are archived into libstiles.a by the
# top-level Makefile.
###############################################################################

STILES_DIR := ../..
SRC := etree.cpp symbolic.cpp supernode.cpp \
       collect.cpp collect_selinv.cpp executor.cpp solve.cpp selinv.cpp \
       element_access.cpp api.cpp
EXTRA_INC := -I.

include $(STILES_DIR)/tools/module.mk
