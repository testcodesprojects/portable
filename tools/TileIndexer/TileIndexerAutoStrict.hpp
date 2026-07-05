/**
 * @file    TileIndexerAutoStrict.hpp
 * @brief   Heuristic planners that select TileIndexer methods/threads based on
 *          matrix size and sparsity estimates. Defines Plan and helpers.
 *
 * @project sTiles (Sparse Tiles Library)
 * @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @version 3.0.0
 * @date 1 1 2026
 * @license Proprietary
 *
 * @note This file is part of the sTiles library, a proprietary software package.
 *       Redistribution or modification without prior permission is prohibited.
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
 *
 * @license
 * This software is proprietary and confidential. Unauthorized copying, distribution, or modification
 * of this software, via any medium, is strictly prohibited. Permission is granted to use the software
 * in binary form for non-commercial purposes only, provided that this copyright notice and permission
 * notice are included in all copies or substantial portions of the software.
 *
 * DISCLAIMER:
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <thread>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <string>
#include <sstream>
#include <limits>
#include <array>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "TileIndexer.hpp"

namespace tilecounter {

enum class DensityClass { VerySparse, Sparse, Moderate, Dense, VeryDense };

/*
Struct: Plan
Purpose: Selected methods and threads for a run, with rationale and density
         classification.
*/
struct Plan {
    Method  count_method;     // counting strategy
    Method  fill_method;      // fill strategy
    int     count_threads;    // threads for counting
    int     fill_threads;     // threads for filling
    int     N;                // tiles per dimension
    double  p_est;            // estimated tile density [0,1], -1 if unknown
    DensityClass density_class;
    bool    repack;           // if true, two-phase (count->repack->fill)
    std::string reason;       // brief rationale
};

// -------------- helpers ----------------
static inline int hw_threads() {
#if defined(_OPENMP)
    return std::max(1, omp_get_max_threads());
#else
    unsigned hc = std::thread::hardware_concurrency();
    return static_cast<int>(std::max(1u, hc));
#endif
}

static inline std::uint64_t tri(std::uint64_t N) { return N * (N + 1ull) / 2ull; }

// Estimate number of active tiles from nnz using 1 - exp(-nnz/T)
static inline std::uint64_t est_active_from_nnz(std::uint64_t nnz, std::uint64_t T) {
    if (T == 0 || nnz == 0) return 0;
    long double x = -(long double)nnz / (long double)T;
    long double e = std::expl(x);
    long double val = (long double)T * (1.0L - e);
    if (val < 0) val = 0;
    std::uint64_t out = (std::uint64_t)(val + 0.5L);
    return std::min<std::uint64_t>(out, T);
}

static inline DensityClass classify(double p) {
    if (p < 5e-4)  return DensityClass::VerySparse;   // <0.05%
    if (p < 0.02)  return DensityClass::Sparse;       // 0.05%–2%
    if (p < 0.09)  return DensityClass::Moderate;     // 2%–9%
    if (p < 0.15)  return DensityClass::Dense;        // 9%–15%
    return DensityClass::VeryDense;                   // ≥15%
}

static inline void dense_bytes(std::uint64_t N, std::uint64_t& char_b, std::uint64_t& bit_b) {
    std::uint64_t T = tri(N);
    char_b = T; bit_b = (T + 7ull) >> 3;
}

static inline Method choose_dense_variant(int N, double p, bool have_p) {
    if (!have_p) return Method::BoolMask; // safe default
    if (p >= 0.15) return Method::BitsetMask;                // very dense
    if (p >= 0.02) return (N <= 2048 ? Method::CharMask : Method::BoolMask);
    return Method::BoolMask;                                  // low density but dense still wins in fill
}

// STRICT counting threads
static inline int strict_count_threads(Method m, int N) {
    switch (m) {
        case Method::CharMask:
        case Method::BoolMask:
        case Method::BitsetMask:
            if (N < 256)   return 2;
            if (N < 2048)  return 8;
            return 4;
        case Method::TiledBoolMask:
        case Method::TiledBitsetMask:
        case Method::PagedMask:
            if (N < 1024)  return 4;
            return 8; // good scaling but stop at 8 to avoid regressions
        default:
            return 4;
    }
}

// STRICT fill threads
static inline int strict_fill_threads(Method m, int /*N*/) {
    switch (m) {
        case Method::CharMask:
        case Method::BoolMask:
        case Method::BitsetMask:
        case Method::TiledBoolMask:
        case Method::TiledBitsetMask:
        case Method::PagedMask:
            return 1; // safest default from your data
        default:
            return 1;
    }
}

// Build a strict single-phase plan (same method both stages)
/*
Function: make_strict_single_phase
Purpose:  Choose one method for both count and fill given basic constraints.
*/
static inline Plan make_strict_single_phase(std::uint64_t n, std::uint64_t nnz,
                                            int tile_size, std::uint64_t mem_budget_bytes = 0)
{
    Plan P{}; P.repack = false;
    P.N = (int)((n + tile_size - 1ull) / (std::uint64_t)tile_size);

    const std::uint64_t T = tri(P.N);
    const std::uint64_t act = (nnz>0)? est_active_from_nnz(nnz, T) : 0;
    P.p_est = (T>0)? double(act)/double(T) : -1.0;
    bool have_p = (P.p_est >= 0.0);

    std::uint64_t char_b=0, bit_b=0; dense_bytes(P.N, char_b, bit_b);
    auto fits = [&](std::uint64_t need){
        if (mem_budget_bytes == 0) return true;            // unknown → assume ok
        return need <= (mem_budget_bytes * 8) / 10;        // leave 20% headroom
    };

    Method chosen;
    if (fits(char_b))          chosen = choose_dense_variant(P.N, P.p_est, have_p);
    else if (fits(bit_b))      chosen = Method::BitsetMask;  // dense bitset as second-best
    else { // low-memory fallback
        if (have_p && P.p_est < 5e-4) chosen = Method::PagedMask;
        else                          chosen = Method::TiledBitsetMask;
    }

    P.density_class = have_p ? classify(P.p_est) : DensityClass::Sparse;
    P.count_method  = chosen;
    P.fill_method   = chosen;
    P.count_threads = strict_count_threads(chosen, P.N);
    P.fill_threads  = strict_fill_threads(chosen, P.N);

    P.reason = std::string("strict single-phase: method=") + to_string(chosen)
               + ", N=" + std::to_string(P.N)
               + ", p_est=" + (have_p? std::to_string(P.p_est) : std::string("unknown"))
               + ", count_t=" + std::to_string(P.count_threads)
               + ", fill_t="  + std::to_string(P.fill_threads);
    return P;
}

// Build a strict two-phase plan (fast count → best fill)
/*
Function: make_strict_two_phase
Purpose:  Two-phase plan: count, then repack/select optimal fill method.
*/
static inline Plan make_strict_two_phase(std::uint64_t n, std::uint64_t nnz,
                                         int tile_size, std::uint64_t mem_budget_bytes = 0)
{
    // Phase A: count
    Plan P = make_strict_single_phase(n, nnz, tile_size, mem_budget_bytes);
    P.repack = true;

    // Phase B: choose best dense for fill if memory allows
    std::uint64_t char_b=0, bit_b=0; dense_bytes(P.N, char_b, bit_b);
    auto fits = [&](std::uint64_t need){
        if (mem_budget_bytes == 0) return true;
        return need <= (mem_budget_bytes * 8) / 10;
    };

    if (fits(char_b) || fits(bit_b)) {
        Method dense_for_fill = choose_dense_variant(P.N, P.p_est, (P.p_est>=0.0));
        P.fill_method  = dense_for_fill;
        P.fill_threads = strict_fill_threads(dense_for_fill, P.N);
    }

    P.reason += std::string(" | two-phase: fill=") + to_string(P.fill_method)
                + ", fill_t=" + std::to_string(P.fill_threads);
    return P;
}

struct UniversalParams {
    int expected_fills_k = 1;
    std::uint64_t mask_mem_cap_bytes = 0;
    bool bit_pack_bool = false;
    bool allow_fill_fusion = true;
};

constexpr std::array<int,2> S3_ACTIVE_THREAD_CANDIDATES = {1, 4};
constexpr double S3_SYMMETRY_FACTOR = 2.0;
[[maybe_unused]] constexpr double S3_TIE_BREAK_WITHIN = 0.05;
constexpr double S3_HUGE_EDGE_THRESHOLD = 200e6;

/*
Function: make_universal_plan
Purpose:  More flexible planner that balances memory caps, estimated density,
          and expected fill costs to pick a robust strategy.
*/
inline Plan make_universal_plan(std::uint64_t n,
                                std::uint64_t nnz,
                                int tile_size,
                                const UniversalParams& params = {})
{
    Plan P{};
    P.repack = false;
    P.N = static_cast<int>((n + static_cast<std::uint64_t>(tile_size) - 1ull) / static_cast<std::uint64_t>(tile_size));

    const std::uint64_t T = tri(static_cast<std::uint64_t>(P.N));
    const std::uint64_t active_pairs_est = est_active_from_nnz(nnz, T);
    P.p_est = (T > 0) ? static_cast<double>(active_pairs_est) / static_cast<double>(T) : -1.0;
    P.density_class = (P.p_est >= 0.0) ? classify(P.p_est) : DensityClass::Sparse;

    auto safe_mul = [](std::uint64_t a, std::uint64_t b) -> std::uint64_t {
        long double prod = static_cast<long double>(a) * static_cast<long double>(b);
        if (prod >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
            return std::numeric_limits<std::uint64_t>::max();
        return static_cast<std::uint64_t>(prod + 0.5L);
    };

    const std::uint64_t cells_per_pair = safe_mul(static_cast<std::uint64_t>(tile_size), static_cast<std::uint64_t>(tile_size));
    const std::uint64_t bytes_char   = safe_mul(active_pairs_est, cells_per_pair);
    const std::uint64_t bytes_bitset = safe_mul(active_pairs_est, (cells_per_pair + 7ull) >> 3);
    std::uint64_t bytes_bool = bytes_char;
    if (params.bit_pack_bool) bytes_bool = safe_mul(active_pairs_est, (cells_per_pair + 7ull) >> 3);

    auto within_cap = [&](std::uint64_t bytes) {
        if (params.mask_mem_cap_bytes == 0) return true;
        return bytes <= params.mask_mem_cap_bytes;
    };

    const bool allow_char   = within_cap(bytes_char);
    const bool allow_bitset = within_cap(bytes_bitset);
    const bool allow_bool   = within_cap(bytes_bool);

    const std::uint64_t filled_cells_est = safe_mul(active_pairs_est, cells_per_pair);
    const long double edges_est_ld = static_cast<long double>(filled_cells_est) * static_cast<long double>(S3_SYMMETRY_FACTOR);
    const double edges_est = static_cast<double>(edges_est_ld);

    auto pick_fallback_mask = [&]() -> Method {
        if (params.expected_fills_k >= 5 && allow_char) return Method::CharMask;
        if (edges_est <= 5e6) {
            if (allow_char) return Method::CharMask;
            if (allow_bitset) return Method::BitsetMask;
            if (allow_bool) return Method::BoolMask;
        } else if (edges_est <= 150e6) {
            if (allow_bitset) return Method::BitsetMask;
            if (allow_char) return Method::CharMask;
            if (allow_bool) return Method::BoolMask;
        } else {
            if (allow_bool) return Method::BoolMask;
            if (allow_bitset) return Method::BitsetMask;
            if (allow_char) return Method::CharMask;
        }
        if (allow_bitset) return Method::BitsetMask;
        if (allow_char) return Method::CharMask;
        return Method::BoolMask;
    };

    Method chosen_mask = pick_fallback_mask();
    if (!allow_char && !allow_bitset && !allow_bool) {
        chosen_mask = Method::BitsetMask;
    }

    int hw = hw_threads();
    int active_threads = 1;
    for (int cand : S3_ACTIVE_THREAD_CANDIDATES) {
        if (P.N >= 1024 && cand == 4) active_threads = 4;
    }
    active_threads = std::min(active_threads, std::max(1, hw));

    P.count_method = chosen_mask;
    P.fill_method  = chosen_mask;
    P.count_threads = active_threads;
    P.fill_threads  = 1;

    if (chosen_mask == Method::BoolMask && edges_est > S3_HUGE_EDGE_THRESHOLD && hw >= 2) {
        P.fill_threads = std::min(2, hw);
    }

    std::ostringstream oss;
    oss << "S3: mask=" << TileIndexer::to_string(chosen_mask)
        << ", ActivePairs≈" << active_pairs_est
        << ", FilledCells≈" << filled_cells_est
        << ", Edges≈" << static_cast<std::uint64_t>(edges_est);
    if (params.mask_mem_cap_bytes) {
        oss << ", cap=" << params.mask_mem_cap_bytes << "B";
    }
    oss << ", k=" << params.expected_fills_k;
    P.reason = oss.str();

    return P;
}

} // namespace tilecounter
