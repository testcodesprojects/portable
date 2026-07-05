/*
 * Stubs for Intel runtime symbols referenced by libxsmm even when built with
 * gcc. _intel_fast_memset and _intel_fast_memcpy are provided by Intel's
 * compiler runtime (libirc); on plain gcc/clang we alias them to libc.
 */
#include <string.h>

void* _intel_fast_memset(void* s, int c, size_t n) { return memset(s, c, n); }
void* _intel_fast_memcpy(void* dst, const void* src, size_t n) { return memcpy(dst, src, n); }
