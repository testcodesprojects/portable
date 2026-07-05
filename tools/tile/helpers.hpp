#pragma once

// C++ headers first (prefer <c...> forms in C++):
#include <cstdlib>
#include <ctime>
#include <cassert>

// Only include OpenMP if enabled:
#ifdef _OPENMP
  #include <omp.h>
#endif

// POSIX only:
#if defined(__unix__) || defined(__APPLE__)
  #include <unistd.h>
#endif

#ifdef __APPLE__
  #include <sys/sysctl.h>  // sysctlbyname on macOS
#endif

// Project headers last
#include "../memory/TileMemoryManager.hpp"

