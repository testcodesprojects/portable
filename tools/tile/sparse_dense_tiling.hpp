#pragma once

// C++ headers first (prefer <c...> forms in C++):
#include <cstdlib>
#include <cstddef>
#include <ctime>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <exception>
#include <new>


// Only include OpenMP if enabled:
#ifdef _OPENMP
  #include <omp.h>
  #include <atomic>
#endif

// POSIX only:
#if defined(__unix__) || defined(__APPLE__)
  #include <unistd.h>
#endif

#ifdef __APPLE__
  #include <sys/sysctl.h>  // sysctlbyname on macOS
#endif

// Project headers last
#include "../common/core_lapack.hpp"
#include "../memory/TileMemoryManager.hpp"
#include "../tile/meta.hpp"
#include "../common/stiles_logger.hpp"
#include "helpers.hpp"
#ifdef SMART_TILES
#endif

// Forward declaration of global control parameter accessor
extern "C" int* sTiles_get_params();

namespace sTiles {
namespace preprocess {

    static inline void compress_semisparse_columns(SemisparseTileMetaCore& semi);
    StatusCode build_semisparse_tile_lookup(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores);
    StatusCode allocate_semisparse_tiles(TiledMatrix *scheme, int group_index, int num_cores = 1);
    StatusCode allocate_semisparse_buffers_from_primary(const TiledMatrix* primary, TiledMatrix* clone, int group_index, int num_threads);
    void update_symbolic_semisparse_tiles(TiledMatrix *S);

    StatusCode build_sparse_tile_lookup(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores);
    StatusCode allocate_sparse_tiles(TiledMatrix *scheme, int group_index, int num_cores = 1);

    StatusCode build_dense_tile_lookup(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores);
    StatusCode build_dense_tile_lookup_variant1(sTiles_call **call_info, TiledMatrix *scheme, int group_index);
    StatusCode build_dense_tile_lookup_variant2(sTiles_call **call_info, TiledMatrix *scheme, int group_index);
    StatusCode allocate_dense_tiles(TiledMatrix *scheme, int group_index, int num_cores);
    StatusCode allocate_dense_tiles_variant1(TiledMatrix *scheme, int group_index, int num_cores);
    StatusCode allocate_dense_tiles_variant2(TiledMatrix *scheme, int group_index, int num_cores);

    StatusCode build_all_tile_lookup(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores);
    StatusCode allocate_all_tiles(TiledMatrix *scheme, int group_index, int num_cores);
    
}  // namespace preprocess
}  // namespace sTiles
namespace sTiles{ 
    
namespace preprocess {
// Function bodies are split across the 4 .inc.hpp files below by tile-type.
// They are textually inserted into the namespace opened above; together they
// reproduce the original sparse_dense_tiling.hpp byte-for-byte.
#include "tile_semisparse.inc.hpp"
#include "tile_sparse.inc.hpp"
#include "tile_dense.inc.hpp"
#include "tile_all.inc.hpp"
