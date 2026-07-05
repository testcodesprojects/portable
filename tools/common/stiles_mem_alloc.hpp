/**
 * @file stiles_mem_alloc.hpp
 * @brief Low-level memory allocation primitives with alignment support.
 *
 * Provides portable memory allocation macros and functions with configurable
 * alignment, using the standard C allocator and platform-specific aligned
 * allocation interfaces.
 *
 * @author Esmail Abdul Fattah
 * @affiliation King Abdullah University of Science and Technology (KAUST)
 * @version 3.0.0
 *
 * @copyright Copyright (c) 2026, KAUST. All rights reserved.
 *
 * PROPRIETARY AND CONFIDENTIAL
 * This software is the exclusive property of KAUST. Unauthorized copying, modification,
 * distribution, or use of this software, in whole or in part, is strictly prohibited.
 * This code may not be reproduced, reverse-engineered, or used to create derivative works
 * without explicit written permission from the copyright holder.
 */

#ifndef MEM_ALLOC_H
#define MEM_ALLOC_H

#include <cstddef>   // alignof

// Pick a numeric default based on ISA. All are plain integers.
#ifndef STILES_ALIGN
  // Always request 64B alignment so code that relies on cache-line alignment,
  // such as Workspace tiles, keeps working even when the compiler does not
  // emit ISA specific defines (e.g. when building without -mavx2).
  #define STILES_ALIGN 64
#endif

// C++ compile-time checks (not preprocessor). These allow -DSTILES_ALIGN=...
static_assert((STILES_ALIGN & (STILES_ALIGN - 1)) == 0,
              "STILES_ALIGN must be a power of two");
static_assert(STILES_ALIGN % alignof(void*) == 0,
              "STILES_ALIGN must be a multiple of pointer alignment");

#include <stddef.h>
#include <string.h>   // memset

#ifdef _WIN32
  #include <malloc.h> // _aligned_malloc, _aligned_free
#endif

#include <stdlib.h>
#define sTiles_malloc(size)       malloc(size)
#define sTiles_free(ptr)          free(ptr)
#define sTiles_calloc(n, sz)      calloc(n, sz)
#define sTiles_realloc(p, new_sz) realloc(p, new_sz)

// aligned
static inline void* sTiles_malloc_aligned(size_t size, size_t alignment = STILES_ALIGN) {
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#else
  void* p = NULL;
  if (posix_memalign(&p, alignment, size) != 0) return NULL;
  return p;
#endif
}
static inline void  sTiles_free_aligned(void* p) {
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}
static inline void* sTiles_calloc_aligned(size_t n, size_t sz, size_t alignment = STILES_ALIGN) {
  void* p = sTiles_malloc_aligned(n * sz, alignment);
  if (p) memset(p, 0, n * sz);
  return p;
}

#endif // MEM_ALLOC_H
