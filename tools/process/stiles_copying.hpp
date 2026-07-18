#ifndef STILES_COPYING_H
#define STILES_COPYING_H

#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <cassert>
#include <stdio.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h> // Required for sysctlbyname on macOS
#endif
#include "../memory/MemoryManager.hpp"

#if defined(STILES_USE_BITRANK_INDEX)
  #include <stiles_index.hpp>  // declares StilesBitRankIndex
#endif

namespace sTiles {namespace SafeMode{
int copy_configuration_1(TiledMatrix **scheme_0, TiledMatrix **scheme_i) {
    
    if (!scheme_0 || !*scheme_0) {
        sTiles::Logger::errorf("Source scheme is null.");
        return -1; // Failure
    }

    if (!scheme_i || !*scheme_i) {
        sTiles::Logger::errorf("Source scheme is null.");
        return -1; // Failure
    }

    if ((*scheme_i)->permutation_flags != NULL) {
        MemoryManager::deallocate((*scheme_i)->permutation_flags);
        (*scheme_i)->permutation_flags = NULL;
    }

    (*scheme_i)->permutation_flags = (*scheme_0)->permutation_flags;
    (*scheme_i)->dim = (*scheme_0)->dim;
    (*scheme_i)->nnz = (*scheme_0)->nnz;
    (*scheme_i)->fixed_column_size = (*scheme_0)->fixed_column_size;
    (*scheme_i)->tile_size = (*scheme_0)->tile_size;
    (*scheme_i)->dimTiledMatrix = (*scheme_0)->dimTiledMatrix;
    (*scheme_i)->numActiveTiles = (*scheme_0)->numActiveTiles;
    (*scheme_i)->nd_nnz = (*scheme_0)->nd_nnz;
    (*scheme_i)->nd_order = (*scheme_0)->nd_order;
    (*scheme_i)->nd_padding = (*scheme_0)->nd_padding;
    (*scheme_i)->triangular_size = (*scheme_0)->triangular_size;
    (*scheme_i)->element_perm = (*scheme_0)->element_perm;
    (*scheme_i)->element_iperm = (*scheme_0)->element_iperm;
    (*scheme_i)->partition_sizes = (*scheme_0)->partition_sizes;
    (*scheme_i)->new_partition_sizes = (*scheme_0)->new_partition_sizes;
    (*scheme_i)->use_ordering = (*scheme_0)->use_ordering;
    (*scheme_i)->red_tree_separator_level = (*scheme_0)->red_tree_separator_level;
    (*scheme_i)->scotch_partition_collection = (*scheme_0)->scotch_partition_collection;
    (*scheme_i)->scotch_root_sep_tiles = (*scheme_0)->scotch_root_sep_tiles;

#if defined(STILES_USE_BITRANK_INDEX)
    (*scheme_i)->bitIndex          = nullptr;
    (*scheme_i)->bitIndex_built    = false;
#endif

    return 0;

}

int copy_configuration_2(TiledMatrix **scheme_0, TiledMatrix **scheme_i) {
    
    if (!scheme_0 || !*scheme_0) {
        sTiles::Logger::errorf("Source scheme is null.");
        return -1; // Failure
    }

    if (!scheme_i || !*scheme_i) {
        sTiles::Logger::errorf("Source scheme is null.");
        return -1; // Failure
    }

#if defined(STILES_USE_BITRANK_INDEX)
  #if defined(STILES_INDEX_TEST)
    // In test builds we still share the legacy mapper to cross-check indices.
    (*scheme_i)->tileIndexMapper = (*scheme_0)->tileIndexMapper;
  #else
    // In pure BitRank builds we don't need the dense mapper.
    (*scheme_i)->tileIndexMapper = nullptr;
  #endif
#else
    // Legacy path unchanged
    (*scheme_i)->tileIndexMapper = (*scheme_0)->tileIndexMapper;
#endif

    return 0;

}

int copy_configuration_3(TiledMatrix **scheme_0, TiledMatrix **scheme_i) {
    
    if (!scheme_0 || !*scheme_0 || !scheme_i || !*scheme_i) {
        sTiles::Logger::errorf("sTiles_copy_configuration_3 received a null scheme pointer.");
        return -1;
    }
    
    (*scheme_i)->tree_counter = (*scheme_0)->tree_counter;

    return 0;

}

int copy_configuration_4(TiledMatrix **scheme_0, TiledMatrix **scheme_i) {
    
    (*scheme_i)->t_indicies = (*scheme_0)->t_indicies;
    (*scheme_i)->e_indicies = (*scheme_0)->e_indicies;

    return 0;

}

SolveTrickConfig* sTiles_copy_configuration_5(const SolveTrickConfig* src) {
    return const_cast<SolveTrickConfig*>(src);  // drop const, since you're reusing it
}
}}

// Fast-mode/general copy helpers and legacy wrappers
namespace sTiles {

inline int copy_configuration_1(TiledMatrix **scheme_0, TiledMatrix **scheme_i) {
    if (!scheme_0 || !*scheme_0) {
        sTiles::Logger::errorf("Source scheme is null.");
        return -1;
    }
    if (!scheme_i || !*scheme_i) {
        sTiles::Logger::errorf("Destination scheme is null.");
        return -1;
    }

    if ((*scheme_i)->permutation_flags != nullptr) {
        MemoryManager::deallocate((*scheme_i)->permutation_flags);
        (*scheme_i)->permutation_flags = nullptr;
    }

    (*scheme_i)->permutation_flags      = (*scheme_0)->permutation_flags;
    (*scheme_i)->dim                    = (*scheme_0)->dim;
    (*scheme_i)->nnz                    = (*scheme_0)->nnz;
    (*scheme_i)->fixed_column_size      = (*scheme_0)->fixed_column_size;
    (*scheme_i)->tile_size              = (*scheme_0)->tile_size;
    (*scheme_i)->dimTiledMatrix         = (*scheme_0)->dimTiledMatrix;
    (*scheme_i)->numActiveTiles         = (*scheme_0)->numActiveTiles;
    (*scheme_i)->nd_nnz                 = (*scheme_0)->nd_nnz;
    (*scheme_i)->nd_order               = (*scheme_0)->nd_order;
    (*scheme_i)->nd_padding             = (*scheme_0)->nd_padding;
    (*scheme_i)->triangular_size        = (*scheme_0)->triangular_size;
    (*scheme_i)->element_perm           = (*scheme_0)->element_perm;
    (*scheme_i)->element_iperm          = (*scheme_0)->element_iperm;
    (*scheme_i)->partition_sizes        = (*scheme_0)->partition_sizes;
    (*scheme_i)->new_partition_sizes    = (*scheme_0)->new_partition_sizes;
    (*scheme_i)->use_ordering           = (*scheme_0)->use_ordering;
    (*scheme_i)->red_tree_separator_level = (*scheme_0)->red_tree_separator_level;
    (*scheme_i)->scotch_partition_collection = (*scheme_0)->scotch_partition_collection;
    (*scheme_i)->scotch_root_sep_tiles  = (*scheme_0)->scotch_root_sep_tiles;

#if defined(STILES_USE_BITRANK_INDEX)
    (*scheme_i)->bitIndex          = nullptr;
    (*scheme_i)->bitIndex_built    = false;
#endif

    return 0;
}

inline int copy_configuration_2(TiledMatrix **scheme_0, TiledMatrix **scheme_i) {
    if (!scheme_0 || !*scheme_0) {
        sTiles::Logger::errorf("Source scheme is null.");
        return -1;
    }
    if (!scheme_i || !*scheme_i) {
        sTiles::Logger::errorf("Destination scheme is null.");
        return -1;
    }

#if defined(STILES_USE_BITRANK_INDEX)
  #if defined(STILES_INDEX_TEST)
    (*scheme_i)->tileIndexMapper = (*scheme_0)->tileIndexMapper;
  #else
    (*scheme_i)->tileIndexMapper = nullptr;
  #endif
#else
    (*scheme_i)->tileIndexMapper = (*scheme_0)->tileIndexMapper;
#endif

    return 0;
}

inline int copy_configuration_3(TiledMatrix **scheme_0, TiledMatrix **scheme_i) {
    if (!scheme_0 || !*scheme_0 || !scheme_i || !*scheme_i) {
        sTiles::Logger::errorf("copy_configuration_3 received a null scheme pointer.");
        return -1;
    }
    (*scheme_i)->tree_counter = (*scheme_0)->tree_counter;
    return 0;
}

inline int copy_configuration_4(TiledMatrix **scheme_0, TiledMatrix **scheme_i) {
    if (!scheme_0 || !*scheme_0 || !scheme_i || !*scheme_i) {
        sTiles::Logger::errorf("copy_configuration_4 received a null scheme pointer.");
        return -1;
    }
    (*scheme_i)->t_indicies = (*scheme_0)->t_indicies;
    (*scheme_i)->e_indicies = (*scheme_0)->e_indicies;
    return 0;
}

// Backward-compatible C-style wrappers used across the codebase
inline int sTiles_copy_configuration_1(TiledMatrix **scheme_0, TiledMatrix **scheme_i) {
    return copy_configuration_1(scheme_0, scheme_i);
}
inline int sTiles_copy_configuration_2(TiledMatrix **scheme_0, TiledMatrix **scheme_i) {
    return copy_configuration_2(scheme_0, scheme_i);
}
inline int sTiles_copy_configuration_3(TiledMatrix **scheme_0, TiledMatrix **scheme_i) {
    return copy_configuration_3(scheme_0, scheme_i);
}
inline int sTiles_copy_configuration_4(TiledMatrix **scheme_0, TiledMatrix **scheme_i) {
    return copy_configuration_4(scheme_0, scheme_i);
}

// Optional: namespaced variant mirroring existing usage for SolveTrickConfig
inline SolveTrickConfig* copy_configuration_5(const SolveTrickConfig* src) {
    return const_cast<SolveTrickConfig*>(src);
}

} // namespace sTiles

#endif // STILES_COPYING_H
