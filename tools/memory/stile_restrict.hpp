/**
 * @file stile_restrict.hpp
 * @brief Portable restrict keyword definition for C and C++.
 *
 * Defines STILE_RESTRICT as a portable alias for compiler-specific restrict
 * qualifiers to enable pointer aliasing optimizations.
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

#pragma once

#ifndef STILE_RESTRICT

  // C mode: prefer the C99 keyword if available
  #if !defined(__cplusplus)
    #if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
      #define STILE_RESTRICT restrict
    #else
      #define STILE_RESTRICT
    #endif

  // C++ mode: use compiler extensions
  #else
    #if defined(__GNUC__) || defined(__clang__)
      #define STILE_RESTRICT __restrict__
    #elif defined(_MSC_VER)
      #define STILE_RESTRICT __restrict
    #else
      #define STILE_RESTRICT
    #endif
  #endif

#endif  // STILE_RESTRICT


// Prefetch hints
#if defined(__GNUC__) || defined(__clang__)
  #define STILE_PREFETCH_R(addr) __builtin_prefetch((addr), 0, 1)
  #define STILE_PREFETCH_W(addr) __builtin_prefetch((addr), 1, 1)
#else
  #define STILE_PREFETCH_R(addr) ((void)0)
  #define STILE_PREFETCH_W(addr) ((void)0)
#endif

#ifndef STILE_PFDIST
  #define STILE_PFDIST 4
#endif

