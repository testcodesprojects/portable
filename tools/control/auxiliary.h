/**
 * @file auxiliary.hpp
 * @brief Inline auxiliary utilities for sTiles
 *
 * @version 3.0.0
 * @date 2026-01-01
 */

#pragma once

#include "../common/stiles_logger.hpp"
#include <algorithm>
#include <cstddef>
#include <string>

namespace sTiles::aux {

//---------------------------------------------
// Memory helpers
//---------------------------------------------

template <typename T>
inline void mem_copy(T* dst, const T* src, std::size_t count) {
    std::copy(src, src + count, dst);
}

template <typename T>
inline void mem_zero(T* mem, std::size_t count) {
    std::fill(mem, mem + count, T{});
}

inline void mem_set_int(int* mem, std::size_t count, int value) {
    std::fill_n(mem, count, value);
}

inline void mem_set_int(volatile int* mem, std::size_t count, int value) {
    for (std::size_t i = 0; i < count; i++) {
        mem[i] = value;
    }
}

} // namespace sTiles::aux
