include ../../make.inc

CXXFLAGS += -Wall -Wextra -pedantic
SHLDFLAGS ?= $(SHARED_LDFLAGS)

APP_SRC = main.cpp MatrixIO.cpp
APP_OBJ = $(APP_SRC:.cpp=.o)
TARGET = TileIndexerCounter

LIB_DIR ?= $(LIB_OUTPUT_DIR)
LIB_STATIC = $(LIB_DIR)/libtileindexer.a
LIB_SHARED = $(LIB_DIR)/libtileindexer.so
LIB_SRC = TileIndexer.cpp MatrixIO.cpp
LIB_OBJ = $(LIB_SRC:.cpp=.lib.o)

all: $(TARGET) $(LIB_STATIC) $(LIB_SHARED)

$(TARGET): $(APP_OBJ)
	$(CXX) $(APP_OBJ) -o $@ $(LDFLAGS)

$(LIB_DIR):
	$(MKDIR_P) $(LIB_DIR)

$(LIB_STATIC): $(LIB_OBJ) | $(LIB_DIR)
	$(ARCH) $(ARCHFLAGS) $@ $(LIB_OBJ)
	$(RANLIB) $@

$(LIB_SHARED): $(LIB_OBJ) | $(LIB_DIR)
	$(CXX) $(SHLDFLAGS) $(LIB_OBJ) -o $@ $(OPENMP_FLAGS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

%.lib.o: %.cpp
	$(CXX) $(CXXFLAGS) -DTILEINDEXER_LIBRARY_ONLY -MMD -MP -c $< -o $@

DEP = $(APP_OBJ:.o=.d) $(LIB_OBJ:.o=.d)
-include $(DEP)

clean:
	rm -f $(APP_OBJ) $(LIB_OBJ) $(DEP) $(TARGET) tile_counter
	rm -f $(LIB_STATIC) $(LIB_SHARED)

.PHONY: all clean asan

asan:
	$(MAKE) clean
	$(MAKE) CXXFLAGS="$(CXXFLAGS) -fsanitize=address -fno-omit-frame-pointer" \
	       LDFLAGS="$(LDFLAGS) -fsanitize=address" \
	       SHLDFLAGS="$(SHLDFLAGS) -fsanitize=address"
