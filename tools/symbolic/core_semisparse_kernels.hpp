/**
 * @file core_semisparse_kernels.hpp
 * @brief Core symbolic kernels for semisparse tile operations.
 *
 * Implements symbolic computation kernels for semisparse matrices including
 * POTRF, TRSM, SYRK, and GEMM operations that track active column indices
 * and bandwidth for efficient factorization.
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

#ifndef STILES_SYMBOLIC_CORE_SEMISPARSE_KERNELS_HPP
#define STILES_SYMBOLIC_CORE_SEMISPARSE_KERNELS_HPP

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../tile/meta.hpp"

namespace sTiles {

inline void core_sspotrf(SemisparseTileMetaCore &core) {
    (void)core;
    // POTRF itself doesn't increase bandwidth beyond what SYRK contributed
    // (Cholesky preserves bandwidth for banded matrices)
}

inline void core_ssdtrsm(SemisparseTileMetaCore &Acore,      // off-diagonal tile (to solve)
                         const SemisparseTileMetaCore &Bcore) // diagonal tile (factored)
{
    (void)Bcore;
    // In semisparse (columns-only) metadata, TRSM does not introduce
    // new column indices; keep Acore.acol/fa/la/sa unchanged here.
    (void)Acore;
}

inline void core_ssdsyrk(SemisparseTileMetaCore &Acore,
                         const SemisparseTileMetaCore &Bcore,
                         int bw_mode = 0) {
    // Bandwidth contribution from SYRK: A -= B^T * B
    // Max bandwidth = span of active columns in B
    //
    // bw_mode 0: conservative (tile width - 1)
    // bw_mode 1: tight (la - fa)

    // DEBUG: trace SYRK calls on diagonal tiles
    //static std::atomic<int> syrk_call_count{0};
    //const int call_id = syrk_call_count.fetch_add(1);
    //if (call_id < 50) {
    //    std::fprintf(stderr, "  [ssdsyrk #%d] Bcore: acol.size=%zu, fa=%d, la=%d, sa=%d | Acore: upper_bw=%d\n",
    //                 call_id, Bcore.acol.size(), Bcore.fa, Bcore.la, Bcore.sa, Acore.upper_bw);
    //}

    std::int32_t span;

    if (bw_mode == 1) {
        // Tight estimate: actual span from la - fa.
        // By the time SYRK executes, the input tile Bcore is fully settled:
        // task dependencies (ss_cond_wait) ensure all prior GEMMs that update
        // Bcore have completed before the TRSM that produces it, and SYRK
        // waits for that TRSM. So fa/la are final and safe to read.
        std::int32_t fa = __sync_add_and_fetch(const_cast<std::int32_t*>(&Bcore.fa), 0);
        std::int32_t la = __sync_add_and_fetch(const_cast<std::int32_t*>(&Bcore.la), 0);

        // fa/la may not be set yet (e.g. first column tiles before any GEMM).
        // Derive from acol if needed.
        if (fa < 0 || la < 0) {
            const auto &acol = Bcore.acol;
            const std::int32_t width = static_cast<std::int32_t>(acol.size());
            fa = -1;
            la = -1;
            for (std::int32_t j = 0; j < width; ++j) {
                if (acol[j] > 0) {
                    if (fa < 0) fa = j;
                    la = j;
                }
            }
        }

        if (fa < 0 || la < 0 || la <= fa) {
            return;
        }
        span = la - fa;
    } else {
        // Conservative (worst-case) estimate: tile width - 1.
        const std::int32_t width = static_cast<std::int32_t>(Bcore.acol.size());
        if (width <= 1) {
            return;
        }
        span = width - 1;
    }

    // Use atomic compare-and-swap to safely update upper_bw in parallel.
    // Multiple threads may update the same diagonal tile's bandwidth concurrently.
    std::int32_t old_bw = Acore.upper_bw;
    while (span > old_bw) {
        if (__sync_bool_compare_and_swap(&Acore.upper_bw, old_bw, span)) {
            break;  // Successfully updated
        }
        old_bw = Acore.upper_bw;  // Re-read and retry
    }
}

inline void core_ssdgemm(const SemisparseTileMetaCore &Acore,
                         const SemisparseTileMetaCore &Bcore,
                         SemisparseTileMetaCore       &Ccore) {
    auto       &Ccol = Ccore.acol;
    const auto &Bcol = Bcore.acol;

    const std::size_t nC = Ccol.size();
    const std::size_t nB = Bcol.size();
    const std::size_t n  = std::min(nC, nB);

    if (n == 0) {
        return;
    }

    // Check if A has any active columns (use atomic load for thread safety)
    // If A has no active columns, GEMM C -= A^T * B contributes nothing
    const std::int32_t faA = __sync_add_and_fetch(const_cast<std::int32_t*>(&Acore.fa), 0);
    const std::int32_t laA = __sync_add_and_fetch(const_cast<std::int32_t*>(&Acore.la), 0);

    if (faA < 0 && laA < 0) return;

    // Thread-safe GEMM symbolic update.
    // Multiple threads may update the same Ccore concurrently.
    for (std::size_t j = 0; j < n; ++j) {
        const bool b_active = (Bcol[j] > 0);

        if (!b_active) {
            continue;
        }

        // Atomically try to mark this column as active.
        // Use compare-and-swap: only succeed if currently inactive (<=0)
        std::int32_t old_val = Ccol[j];
        while (old_val <= 0) {
            if (__sync_bool_compare_and_swap(&Ccol[j], old_val, 1)) {
                // Successfully marked column as active - update fa/la/sa atomically
                const std::int32_t jj = static_cast<std::int32_t>(j);

                // Atomically update fa (minimum)
                std::int32_t old_fa = Ccore.fa;
                while (old_fa < 0 || jj < old_fa) {
                    if (__sync_bool_compare_and_swap(&Ccore.fa, old_fa, jj)) {
                        break;
                    }
                    old_fa = Ccore.fa;
                }

                // Atomically update la (maximum)
                std::int32_t old_la = Ccore.la;
                while (jj > old_la) {
                    if (__sync_bool_compare_and_swap(&Ccore.la, old_la, jj)) {
                        break;
                    }
                    old_la = Ccore.la;
                }

                // Atomically increment sa
                __sync_fetch_and_add(&Ccore.sa, 1);

                break;  // Column marked, done with this j
            }
            old_val = Ccol[j];  // Re-read and retry
        }
        // If old_val > 0, column was already active - nothing to do
    }
}

} // namespace sTiles

#endif // STILES_SYMBOLIC_CORE_SEMISPARSE_KERNELS_HPP
