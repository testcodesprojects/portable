#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include "stiles_sort.hpp"

#ifndef STILES_SORT_USE_QUADFLUX
#define STILES_SORT_USE_QUADFLUX 1
#endif

namespace sTiles {

namespace detail {

template<class T, class Compare> inline const Compare*& cmp_slot() { static thread_local const Compare* p = nullptr; return p; }

template<class T, class Compare> inline int cmp_qsort(const void* a, const void* b) {
    const T& x = *static_cast<const T*>(a);
    const T& y = *static_cast<const T*>(b);
    const Compare* c = cmp_slot<T, Compare>();
    if ((*c)(x, y)) return -1;
    if ((*c)(y, x)) return 1;
    return 0;
}

template<class RandomIt> inline bool is_contiguous_range(RandomIt first, RandomIt last) {
    using T = typename std::iterator_traits<RandomIt>::value_type;
    if (first == last) return true;
    if constexpr (!std::is_same_v<decltype(std::addressof(*first)), T*>) return false;
    const std::size_t n = static_cast<std::size_t>(last - first);
    if (n <= 1) return true;
    T* p0 = std::addressof(*first);
    T* pN = std::addressof(*(last - 1));
    return pN == p0 + (n - 1);
}

} // namespace detail

template<class RandomIt, class Compare> inline void sort(RandomIt first, RandomIt last, Compare comp) {
    static_assert(std::is_base_of_v<std::random_access_iterator_tag, typename std::iterator_traits<RandomIt>::iterator_category>, "sTiles::sort requires random access iterators");
    using T = typename std::iterator_traits<RandomIt>::value_type;
    const std::size_t n = static_cast<std::size_t>(last - first);
    if (n <= 1) return;

#if STILES_SORT_USE_QUADFLUX
    if constexpr (std::is_trivially_copyable_v<T>) {
        if (detail::is_contiguous_range(first, last)) {
            auto* prev = detail::cmp_slot<T, Compare>();
            detail::cmp_slot<T, Compare>() = std::addressof(comp);
            sTile::quadfluxsort(static_cast<void*>(std::addressof(*first)), n, sizeof(T), &detail::cmp_qsort<T, Compare>);
            detail::cmp_slot<T, Compare>() = prev;
            return;
        }
    }
#endif

    std::sort(first, last, comp);
}

template<class RandomIt> inline void sort(RandomIt first, RandomIt last) {
    using T = typename std::iterator_traits<RandomIt>::value_type;
    sTiles::sort(first, last, std::less<T>{});
}

} // namespace sTiles
