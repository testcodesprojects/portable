/**
 * @file stile_compiler_hints.hpp
 * @brief Compiler-specific optimization hints and intrinsics.
 *
 * Provides portable macros for restrict qualifiers, alignment hints,
 * and other compiler-specific optimizations across different platforms.
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

#ifndef STILES_COMPILER_HINTS_HPP
#define STILES_COMPILER_HINTS_HPP

#include <cstddef>
#include <cstdint>

#if defined(_MSC_VER)
  #define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
  #define RESTRICT __restrict__
#else
  #define RESTRICT
#endif

// Prefer C++20 std::assume_aligned when available
#if __has_include(<memory>)
  #include <memory>
#endif

#if defined(__cpp_lib_assume_aligned) && __cpp_lib_assume_aligned >= 201811L
  template<class T>
  inline T* assume_aligned_64(T* p) noexcept {
      return std::assume_aligned<64>(p);
  }
  #define ASSUME_ALIGNED_64(p) (p = assume_aligned_64(p))
#elif defined(__GNUC__) || defined(__clang__)
  #define ASSUME_ALIGNED_64(p) (p = (decltype(p))__builtin_assume_aligned((p), 64))
#elif defined(_MSC_VER)
// MSVC has no assume_aligned builtin. Use a no-op in release and keep a debug assert.
  #define ASSUME_ALIGNED_64(p) ((void)0)
#else
  #define ASSUME_ALIGNED_64(p) ((void)0)
#endif

// Optional: branch hints
#if defined(__GNUC__) || defined(__clang__)
  #define LIKELY(x)   __builtin_expect(!!(x), 1)
  #define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
  #define LIKELY(x)   (x)
  #define UNLIKELY(x) (x)
#endif

// Optional: debug-only runtime alignment check to protect the assume
#ifndef NDEBUG
  #include <cassert>
  #define DEBUG_ASSERT_ALIGNED_64(p) \
      assert(((reinterpret_cast<std::uintptr_t>(p) & 63u) == 0) && "pointer must be 64B aligned")
#else
  #define DEBUG_ASSERT_ALIGNED_64(p) ((void)0)
#endif

#endif // STILES_COMPILER_HINTS_HPP
