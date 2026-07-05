/**
 * @file auxiliary.cpp
 * @brief Implementation of sTiles auxiliary routines and utilities (C++ version).
 *
 * @version 3.0.0
 * @author 
 * @date 2026-01-01
 */

#include "stiles_logger.hpp"
#include <vector>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <cstring> // for legacy compatibility

namespace sTiles::aux {

//---------------------------------------------
// Logging wrappers
//---------------------------------------------

inline void Warning(const std::string& message) {
    sTiles::Warning(message);
}

inline void Error(const std::string& message) {
    sTiles::Error(message);
}

inline void Fatal(const std::string& message) {
    sTiles::Fatal(message);
}

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
