#pragma once

#include <algorithm>   // std::sort
#include <chrono>
#include <cstddef>
#include <cstdlib>     // std::malloc, std::free
#include <cstring>     // std::memcpy
#include <fstream>
#include <mutex>
#include <vector>



extern "C" {
void quadsort(void *array, std::size_t nmemb, std::size_t size,
              int (*cmp)(const void *, const void *));
void fluxsort(void *array, std::size_t nmemb, std::size_t size,
              int (*cmp)(const void *, const void *));
}

namespace sTile {

using SortCompare = int (*)(const void *, const void *);

inline bool is_sorted_qstyle(void* array,
                                std::size_t nmemb,
                                std::size_t size,
                                SortCompare cmp)
{
    if (nmemb <= 1 || size == 0 || array == nullptr || cmp == nullptr) {
        return true;
    }

    unsigned char* data = static_cast<unsigned char*>(array);
    for (std::size_t i = 1; i < nmemb; ++i) {
        void* prev = data + (i - 1) * size;
        void* curr = data + i * size;
        if (cmp(prev, curr) > 0) {
            return false;
        }
    }
    return true;
}


// generic dispatcher
inline void quadfluxsort(void *array,
                         std::size_t nmemb,
                         std::size_t size,
                         SortCompare cmp)
{
    // if (nmemb > 128u) {
    //     quadsort(array, nmemb, size, cmp);
    // } else {
    //     fluxsort(array, nmemb, size, cmp);
    // }

    if (!is_sorted_qstyle(array, nmemb, size, cmp)) {
        if (nmemb > 128u) {
            quadsort(array, nmemb, size, cmp);
        } else {
            fluxsort(array, nmemb, size, cmp);
        }
    }

}


//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

namespace sort_test1{
using SortCompare = int (*)(const void*, const void*);

namespace {
    std::mutex &quadflux_log_mutex() {
        static std::mutex m;
        return m;
    }

    std::ofstream &quadflux_log_stream() {
        static std::ofstream out("quadfluxsort_log.txt",
                                 std::ios::out | std::ios::app);
        return out;
    }

    inline void quadflux_log(const char* tag,
                             std::size_t nmemb,
                             long long ns)
    {
        std::ofstream &log = quadflux_log_stream();
        if (!log) return;

        log << tag << " " << nmemb << " " << ns << "\n";
        // comment out if you want less flushing
        log.flush();
    }

    // adapter: use std::sort with a qsort style comparator
    inline void std_qsort_compatible(void* base,
                                     std::size_t nmemb,
                                     std::size_t size,
                                     SortCompare cmp)
    {
        if (nmemb <= 1 || size == 0 || base == nullptr || cmp == nullptr) {
            return;
        }

        unsigned char* data = static_cast<unsigned char*>(base);

        // permutation indices 0..nmemb-1
        std::vector<std::size_t> idx(nmemb);
        for (std::size_t i = 0; i < nmemb; ++i) {
            idx[i] = i;
        }

        auto cmp_index = [data, size, cmp](std::size_t a, std::size_t b) {
            const void* pa = data + a * size;
            const void* pb = data + b * size;
            return cmp(pa, pb) < 0;
        };

        // std::sort works on the index array
        std::sort(idx.begin(), idx.end(), cmp_index);

        // apply the permutation in place to data
        std::vector<bool> visited(nmemb, false);
        std::vector<unsigned char> tmp(size);

        for (std::size_t i = 0; i < nmemb; ++i) {
            if (visited[i] || idx[i] == i) {
                continue;
            }

            std::size_t j = i;
            std::memcpy(tmp.data(), data + i * size, size);

            while (!visited[j]) {
                visited[j] = true;
                std::size_t k = idx[j];
                if (k == i) {
                    std::memcpy(data + j * size, tmp.data(), size);
                    break;
                }
                std::memcpy(data + j * size, data + k * size, size);
                j = k;
            }
        }
    }
}

//quadfluxsort_log.txt: rename it to quadfluxsort
inline void quadfluxsort(void *array,
                         std::size_t nmemb,
                         std::size_t size,
                         SortCompare cmp)
{
    if (nmemb <= 1 || size == 0 || array == nullptr) {
        return;
    }

    const std::size_t bytes = nmemb * size;

    // three independent copies of the original data
    unsigned char *buf_q = static_cast<unsigned char*>(std::malloc(bytes));
    unsigned char *buf_f = static_cast<unsigned char*>(std::malloc(bytes));
    unsigned char *buf_s = static_cast<unsigned char*>(std::malloc(bytes));

    if (!buf_q || !buf_f || !buf_s) {
        std::free(buf_q);
        std::free(buf_f);
        std::free(buf_s);
        // fall back to just calling one sort without logging
        quadsort(array, nmemb, size, cmp);
        return;
    }

    std::memcpy(buf_q, array, bytes);
    std::memcpy(buf_f, array, bytes);
    std::memcpy(buf_s, array, bytes);

    using clock = std::chrono::steady_clock;

    // quadsort on its own copy
    auto t0 = clock::now();
    quadsort(buf_q, nmemb, size, cmp);
    auto t1 = clock::now();

    // fluxsort on its own copy
    auto t2 = clock::now();
    fluxsort(buf_f, nmemb, size, cmp);
    auto t3 = clock::now();

    // std::sort (through the qsort compatible adapter) on its own copy
    auto t4 = clock::now();
    std_qsort_compatible(buf_s, nmemb, size, cmp);
    auto t5 = clock::now();

    using namespace std::chrono;
    const long long dt_quadsort_ns = duration_cast<nanoseconds>(t1 - t0).count();
    const long long dt_fluxsort_ns = duration_cast<nanoseconds>(t3 - t2).count();
    const long long dt_std_ns      = duration_cast<nanoseconds>(t5 - t4).count();

    {
        std::lock_guard<std::mutex> lock(quadflux_log_mutex());
        quadflux_log("q", nmemb, dt_quadsort_ns);
        quadflux_log("f", nmemb, dt_fluxsort_ns);
        quadflux_log("s", nmemb, dt_std_ns);
    }

    // choose which result you want to expose to the caller
    // here we pick fluxsort result by default
    std::memcpy(array, buf_f, bytes);

    std::free(buf_q);
    std::free(buf_f);
    std::free(buf_s);
}

}



namespace sort_test2{

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

using SortCompare = int (*)(const void*, const void*);

namespace {
    std::mutex &quadflux_log_mutex() {
        static std::mutex m;
        return m;
    }

    // renamed file: from "quadfluxsort_log.txt" to "quadfluxsort"
    std::ofstream &quadflux_log_stream() {
    static std::ofstream out("quadfluxsort_test2.txt",
                            std::ios::out | std::ios::app);

        return out;
    }

    inline void quadflux_log(const char* tag,
                             std::size_t nmemb,
                             long long ns)
    {
        std::ofstream &log = quadflux_log_stream();
        if (!log) return;

        log << tag << " " << nmemb << " " << ns << "\n";
        log.flush();  // optional, comment out if you want less flushing
    }

    // the hybrid we want to test: threshold 128
    inline void hybrid_128(void* array,
                           std::size_t nmemb,
                           std::size_t size,
                           SortCompare cmp)
    {
        if (nmemb > 128u) {
            quadsort(array, nmemb, size, cmp);
        } else {
            fluxsort(array, nmemb, size, cmp);
        }
    }
}

// benchmark: direct hybrid vs "check sorted then hybrid"
inline void quadfluxsort(void *array,
                              std::size_t nmemb,
                              std::size_t size,
                              SortCompare cmp)
{
    if (nmemb <= 1 || size == 0 || array == nullptr) {
        return;
    }

    const std::size_t bytes = nmemb * size;

    // two independent copies of the original data
    unsigned char *buf_direct = static_cast<unsigned char*>(std::malloc(bytes));
    unsigned char *buf_check  = static_cast<unsigned char*>(std::malloc(bytes));

    if (!buf_direct || !buf_check) {
        std::free(buf_direct);
        std::free(buf_check);
        // fall back to just calling hybrid once without logging
        hybrid_128(array, nmemb, size, cmp);
        return;
    }

    std::memcpy(buf_direct, array, bytes);
    std::memcpy(buf_check,  array, bytes);

    using clock = std::chrono::steady_clock;

    // first test: direct hybrid_128
    auto t0 = clock::now();
    hybrid_128(buf_direct, nmemb, size, cmp);
    auto t1 = clock::now();

    // second test: check sorted, then hybrid_128 only if needed
    auto t2 = clock::now();
    if (!is_sorted_qstyle(buf_check, nmemb, size, cmp)) {
        hybrid_128(buf_check, nmemb, size, cmp);
    }
    auto t3 = clock::now();

    using namespace std::chrono;
    const long long dt_direct_ns = duration_cast<nanoseconds>(t1 - t0).count();
    const long long dt_check_ns  = duration_cast<nanoseconds>(t3 - t2).count();

    {
        std::lock_guard<std::mutex> lock(quadflux_log_mutex());
        quadflux_log("h", nmemb, dt_direct_ns);  // h = hybrid direct
        quadflux_log("c", nmemb, dt_check_ns);   // c = check + hybrid
    }

    // expose one sorted result to caller; pick the direct hybrid result
    std::memcpy(array, buf_direct, bytes);

    std::free(buf_direct);
    std::free(buf_check);
}


}
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------


// comparators
// inline int cmp_int_asc(const void *a, const void *b)
// {
//     const int *ia = static_cast<const int *>(a);
//     const int *ib = static_cast<const int *>(b);
//     return (*ia > *ib) - (*ia < *ib);
// }

inline int cmp_int_asc(const void *a, const void *b)
{
    const int ia = *static_cast<const int *>(a);
    const int ib = *static_cast<const int *>(b);
    return ia - ib;  
}


inline int cmp_double_asc(const void *a, const void *b)
{
    const double *da = static_cast<const double *>(a);
    const double *db = static_cast<const double *>(b);
    return (*da > *db) - (*da < *db);
}

// integer convenience: iquadfluxsort(begin, end)
inline void iquadfluxsort(int *first, int *last)
{
    std::size_t nmemb = static_cast<std::size_t>(last - first);
    if (nmemb == 0) return;

    quadfluxsort(static_cast<void *>(first),
                 nmemb,
                 sizeof(int),
                 cmp_int_asc);


}

// double convenience: dquadfluxsort(begin, end)
inline void dquadfluxsort(double *first, double *last)
{
    std::size_t nmemb = static_cast<std::size_t>(last - first);
    if (nmemb == 0) return;

    quadfluxsort(static_cast<void *>(first),
                 nmemb,
                 sizeof(double),
                 cmp_double_asc);

}

// std::sort variants

inline void istdsort(int *first, int *last)
{
    std::sort(first, last);
}

inline void dstdsort(double *first, double *last)
{
    std::sort(first, last);
}

} // namespace sTile

#ifdef STILES_SORT_IMPLEMENTATION
extern "C" {
    #include "fsort/quadsort.h"
    #include "fsort/fluxsort.h"
}
#endif
