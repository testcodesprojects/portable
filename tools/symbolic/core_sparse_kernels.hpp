/**
 * @file core_sparse_kernels.hpp
 * @brief Core symbolic kernels for sparse tile operations.
 *
 * Provides low-level symbolic computation kernels for sparse matrix operations
 * including bitmask-based sparsity pattern manipulation and portable bit-counting
 * utilities for cross-platform support.
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

#ifndef STILES_SYMBOLIC_CORE_SPARSE_KERNELS_HPP
#define STILES_SYMBOLIC_CORE_SPARSE_KERNELS_HPP

#include <cstdint>
#include <vector>
#include <algorithm>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "../tile/meta.hpp"

// Restrict portability helper
#if defined(__clang__) || defined(__GNUC__) || defined(__INTEL_COMPILER)
    #define STILES_RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define STILES_RESTRICT __restrict
#else
    #define STILES_RESTRICT
#endif

namespace sTiles {

// Portable count trailing zeros for 64 bit, precondition x != 0
inline int stiles_ctz64(std::uint64_t x) {
#if defined(__clang__) || defined(__GNUC__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return static_cast<int>(idx);
#else
    int c = 0;
    while ((x & 1u) == 0u) {
        x >>= 1;
        ++c;
    }
    return c;
#endif
}

// Extract set row indices from a bitmap column
static inline int extract_rows_from_bitmap_column(const std::uint64_t *STILES_RESTRICT bits, int height, int words_per_col, int *STILES_RESTRICT rows_out) {
    int q = 0;
    for (int w = 0; w < words_per_col; ++w) {
        std::uint64_t x = bits[w];
        while (x) {
            const int bit = stiles_ctz64(x);
            const int row = (w << 6) + bit;
            if (row >= height) break;
            rows_out[q++] = row;
            x &= x - 1;
        }
    }
    return q;
}

// Check if two bitmap columns share any nonzero row
static inline bool bitmap_columns_intersect(const std::uint64_t *STILES_RESTRICT x, const std::uint64_t *STILES_RESTRICT y, int words_per_col) {
    for (int w = 0; w < words_per_col; ++w) {
        if (x[w] & y[w]) return true;
    }
    return false;
}

// =============================================================================
// core_strsm: symbolic TRSM fill pattern
// =============================================================================
inline void core_strsm(sTiles::Op transa, const TileMetaCore &Atile, const SemisparseTileMetaCore &Acore, const SymbolicTileBitmaskCore &AbitsDiag, SemisparseTileMetaCore &Bcore, SymbolicTileBitmaskCore &Bbits) {
    const int sa_B = Bcore.sa;
    const int height = Atile.height;
    const int words = Bbits.words_per_col;
    if (sa_B <= 0 || height <= 0 || words <= 0) return;

    // Process each active column of B independently
    for (int b_idx = 0; b_idx < sa_B; ++b_idx) {
        const int phys_col_b = Bcore.aind[static_cast<std::size_t>(b_idx)];
        if (phys_col_b < 0) continue;

        std::uint64_t *B_col_bits = Bbits.column_bits(b_idx);
        if (!B_col_bits) continue;

        if (transa == sTiles::Op::Trans) {
            // Solving U^T * X = B, forward substitution
            for (int i = 0; i < height; ++i) {
                const int word_i = i >> 6;
                const std::uint64_t mask_i = (1ULL << (i & 63));

                // Skip if already nonzero
                if (B_col_bits[word_i] & mask_i) continue;

                // Column index in A, skip early if not active
                const int active_col_i = Acore.acol[static_cast<std::size_t>(i)];
                if (active_col_i < 0) continue;

                bool fill_in = false;
                for (int j = 0; j < i && !fill_in; ++j) {
                    if (!(B_col_bits[j >> 6] & (1ULL << (j & 63)))) continue;
                    if (AbitsDiag.test_bit(active_col_i, j)) {
                        fill_in = true;
                    }
                }

                if (fill_in) {
                    B_col_bits[word_i] |= mask_i;
                }
            }
        } else {
            // Solving U * X = B, backward substitution
            for (int i = height - 1; i >= 0; --i) {
                const int word_i = i >> 6;
                const std::uint64_t mask_i = (1ULL << (i & 63));

                // Skip if already nonzero
                if (B_col_bits[word_i] & mask_i) continue;

                bool fill_in = false;
                for (int j = i + 1; j < height && !fill_in; ++j) {
                    if (!(B_col_bits[j >> 6] & (1ULL << (j & 63)))) continue;

                    const int active_col_j = Acore.acol[static_cast<std::size_t>(j)];
                    if (active_col_j >= 0 && AbitsDiag.test_bit(active_col_j, i)) {
                        fill_in = true;
                    }
                }

                if (fill_in) {
                    B_col_bits[word_i] |= mask_i;
                }
            }
        }
    }
}

// =============================================================================
 // core_sgemm: C += A^T * B (symbolic)
// =============================================================================
inline void core_sgemm(const SemisparseTileMetaCore &Acore, const SemisparseTileMetaCore &Bcore, SemisparseTileMetaCore &Ccore, const SymbolicTileBitmaskCore &Abits, const SymbolicTileBitmaskCore &Bbits, SymbolicTileBitmaskCore &Cbits) {
    const int saA = Acore.sa;
    const int saB = Bcore.sa;
    const int words = Abits.words_per_col;

    if (saA <= 0 || saB <= 0 || words <= 0) return;

    for (int kA = 0; kA < saA; ++kA) {
        const int phys_col_a = Acore.aind[static_cast<std::size_t>(kA)];
        if (phys_col_a < 0) continue;

        const std::uint64_t *colA = Abits.column_bits(kA);
        if (!colA) continue;

        for (int kB = 0; kB < saB; ++kB) {
            const int phys_col_b = Bcore.aind[static_cast<std::size_t>(kB)];
            if (phys_col_b < 0) continue;

            const int active_c = Ccore.acol[static_cast<std::size_t>(phys_col_b)];
            if (active_c < 0) continue;

            const std::uint64_t *colB = Bbits.column_bits(kB);
            if (!colB) continue;

            // Check if columns of A and B share any nonzero row
            if (!bitmap_columns_intersect(colA, colB, words)) continue;

            // C[phys_col_a, phys_col_b] is nonzero
            Cbits.set_bit(active_c, phys_col_a);
        }
    }
}

// =============================================================================
// core_ssyrk: A += B^T * B (symmetric rank k update)
// =============================================================================
inline void core_ssyrk(sTiles::Uplo uplo, SemisparseTileMetaCore &Acore, const SemisparseTileMetaCore &Bcore, SymbolicTileBitmaskCore &Abits, const SymbolicTileBitmaskCore &Bbits) {
    const int saB = Bcore.sa;
    const int wordsB = Bbits.words_per_col;
    const int widthA = static_cast<int>(Acore.acol.size());

    if (saB <= 0 || wordsB <= 0 || widthA <= 0) return;

    for (int k = 0; k < saB; ++k) {
        const int idx_k = Bcore.aind[static_cast<std::size_t>(k)];
        if (idx_k < 0 || idx_k >= widthA) continue;

        const std::uint64_t *bits_k = Bbits.column_bits(k);
        if (!bits_k) continue;

        // Exploit symmetry: only compute m <= k
        for (int m = 0; m <= k; ++m) {
            const int idx_m = Bcore.aind[static_cast<std::size_t>(m)];
            if (idx_m < 0 || idx_m >= widthA) continue;

            const std::uint64_t *bits_m = Bbits.column_bits(m);
            if (!bits_m) continue;

            if (!bitmap_columns_intersect(bits_k, bits_m, wordsB)) continue;

            int row_tgt;
            int col_tgt;
            if (uplo == sTiles::Uplo::Upper) {
                row_tgt = std::min(idx_k, idx_m);
                col_tgt = std::max(idx_k, idx_m);
            } else {
                row_tgt = std::max(idx_k, idx_m);
                col_tgt = std::min(idx_k, idx_m);
            }

            const int active_col_a = Acore.acol[static_cast<std::size_t>(col_tgt)];
            if (active_col_a >= 0) {
                Abits.set_bit(active_col_a, row_tgt);
            }
        }
    }
}

// =============================================================================
// core_spotrf: Cholesky factorization fill in prediction
// =============================================================================
inline void core_spotrf(sTiles::Uplo uplo, const TileMetaCore &Atile, SemisparseTileMetaCore &core, SymbolicTileBitmaskCore &Abits) {
    const int height = Atile.height;
    const int sa = core.sa;
    const int words = Abits.words_per_col;

    if (sa <= 0 || height <= 0 || words <= 0) return;

    std::vector<int> queue(static_cast<std::size_t>(height));

    if (uplo == sTiles::Uplo::Upper) {
        const int acol_size = static_cast<int>(core.acol.size());

        for (int j = 0; j < height; ++j) {
            if (j >= acol_size) break;

            const int active_j = core.acol[static_cast<std::size_t>(j)];
            if (active_j < 0) continue;

            std::uint64_t *j_bits = Abits.column_bits(active_j);
            if (!j_bits) continue;

            for (int k = 0; k < j; ++k) {
                const int word_k = k >> 6;
                const std::uint64_t pivot_mask = (1ULL << (k & 63));
                if (!(j_bits[word_k] & pivot_mask)) continue;

                for (int i = k + 1; i <= j; ++i) {
                    if (i >= acol_size) break;

                    const int active_i = core.acol[static_cast<std::size_t>(i)];
                    if (active_i < 0) continue;

                    const std::uint64_t *col_i_bits = Abits.column_bits(active_i);
                    if (!col_i_bits || !(col_i_bits[word_k] & pivot_mask)) continue;

                    j_bits[i >> 6] |= (1ULL << (i & 63));
                }
            }

            int count = extract_rows_from_bitmap_column(j_bits, height, words, queue.data());
            for (int idx = 0; idx < count; ++idx) {
                const int row = queue[idx];
                if (row <= j) {
                    const int bw = j - row;
                    if (bw > core.upper_bw) {
                        core.upper_bw = bw;
                    }
                }
            }
        }
    }
}

} // namespace sTiles

#endif // STILES_SYMBOLIC_CORE_SPARSE_KERNELS_HPP




































// #ifndef STILES_SYMBOLIC_CORE_SPARSE_KERNELS_HPP
// #define STILES_SYMBOLIC_CORE_SPARSE_KERNELS_HPP

// #include <cstdint>
// #include <vector>
// #include <algorithm>

// #include "../tile/meta.hpp"

// namespace sTiles {


// static inline int extract_rows_from_bitmap_column_old(const std::uint64_t * __restrict__ bits, int height, int words_per_col, int * __restrict__ rows_out) {
//     int q = 0;
//     for (int w = 0; w < words_per_col; ++w) {
//         std::uint64_t x = bits[w];
//         while (x) {
//             const int bit = __builtin_ctzll(x);
//             const int row = (w << 6) + bit;
//             if (row >= height) break;
//             rows_out[q++] = row;
//             x &= x - 1;
//         }
//     }
//     return q;
// }

// static inline bool bitmap_columns_intersect_old(const std::uint64_t * __restrict__ x, const std::uint64_t * __restrict__ y, int words_per_col) {
//     int w = 0;
//     // Unroll by 4 to allow better pipelining and reduce branch overhead
//     for (; w + 3 < words_per_col; w += 4) {
//         if ((x[w] & y[w]) | (x[w+1] & y[w+1]) |
//             (x[w+2] & y[w+2]) | (x[w+3] & y[w+3])) {
//             return true;
//         }
//     }
//     // Handle remaining words
//     for (; w < words_per_col; ++w) {
//         if (x[w] & y[w]) return true;
//     }
//     return false;
// }

// inline void core_strsm_old(sTiles::Op transa,
//                        const TileMetaCore &Atile,
//                        const SemisparseTileMetaCore &Acore,
//                        const SymbolicTileBitmaskCore &AbitsDiag,
//                        SemisparseTileMetaCore &Bcore,
//                        SymbolicTileBitmaskCore &Bbits)
// {
//     const int sa_B = Bcore.sa;
//     const int height = Atile.height;
//     const int words = Bbits.words_per_col;

//     if (sa_B <= 0 || height <= 0 || words <= 0) return;

//     // Process each active column of B independently
//     for (int b_idx = 0; b_idx < sa_B; ++b_idx) {
//         const int phys_col_b = Bcore.aind[static_cast<std::size_t>(b_idx)];
//         if (phys_col_b < 0) continue;

//         std::uint64_t *B_col_bits = Bbits.column_bits(b_idx);
//         if (!B_col_bits) continue;

//         if (transa == sTiles::Op::Trans) {
//             // =================================================================
//             // Solving U^T * X = B where U is Upper triangular
//             // Equivalent to L * X = B where L = U^T (Lower triangular)
//             // FORWARD substitution: compute X[0], X[1], ..., X[n-1] in order
//             //
//             // X[i] = (B[i] - sum_{j<i} L[i,j]*X[j]) / L[i,i]
//             //
//             // Fill-in at row i occurs if:
//             //   - B[i] was already nonzero, OR
//             //   - exists j < i where X[j] != 0 AND L[i,j] != 0
//             //
//             // Since L = U^T: L[i,j] = U[j,i] (for j < i, this is above diagonal in U)
//             // U[j,i] is stored in column i at row j
//             // =================================================================
//             for (int i = 0; i < height; ++i) {
//                 // Skip if already nonzero
//                 if (B_col_bits[i / 64] & (1ULL << (i % 64))) continue;

//                 // Check if any previous row j < i causes fill-in
//                 bool fill_in = false;
//                 for (int j = 0; j < i && !fill_in; ++j) {
//                     // Is X[j] nonzero?
//                     if (!(B_col_bits[j / 64] & (1ULL << (j % 64)))) continue;

//                     // Is L[i,j] = U[j,i] nonzero?
//                     // U[j,i] is in column i of U, at row j
//                     const int active_col_i = Acore.acol[static_cast<std::size_t>(i)];
//                     if (active_col_i >= 0 && AbitsDiag.test_bit(active_col_i, j)) {
//                         fill_in = true;
//                     }
//                 }

//                 if (fill_in) {
//                     B_col_bits[i / 64] |= (1ULL << (i % 64));
//                 }
//             }
//         } else {
//             // =================================================================
//             // Solving U * X = B where U is Upper triangular
//             // BACKWARD substitution: compute X[n-1], X[n-2], ..., X[0] in order
//             //
//             // X[i] = (B[i] - sum_{j>i} U[i,j]*X[j]) / U[i,i]
//             //
//             // Fill-in at row i occurs if:
//             //   - B[i] was already nonzero, OR
//             //   - exists j > i where X[j] != 0 AND U[i,j] != 0
//             //
//             // U[i,j] is stored in column j at row i (for j > i)
//             // =================================================================
//             for (int i = height - 1; i >= 0; --i) {
//                 // Skip if already nonzero
//                 if (B_col_bits[i / 64] & (1ULL << (i % 64))) continue;

//                 // Check if any later row j > i causes fill-in
//                 bool fill_in = false;
//                 for (int j = i + 1; j < height && !fill_in; ++j) {
//                     // Is X[j] nonzero?
//                     if (!(B_col_bits[j / 64] & (1ULL << (j % 64)))) continue;

//                     // Is U[i,j] nonzero?
//                     // U[i,j] is in column j of U, at row i
//                     const int active_col_j = Acore.acol[static_cast<std::size_t>(j)];
//                     if (active_col_j >= 0 && AbitsDiag.test_bit(active_col_j, i)) {
//                         fill_in = true;
//                     }
//                 }

//                 if (fill_in) {
//                     B_col_bits[i / 64] |= (1ULL << (i % 64));
//                 }
//             }
//         }
//     }
// }

// // =============================================================================
// // core_sgemm: C += A^T * B (symbolic)
// // =============================================================================
// inline void core_sgemm_old(const SemisparseTileMetaCore &Acore,
//                        const SemisparseTileMetaCore &Bcore,
//                        SemisparseTileMetaCore &Ccore,
//                        const SymbolicTileBitmaskCore &Abits,
//                        const SymbolicTileBitmaskCore &Bbits,
//                        SymbolicTileBitmaskCore &Cbits)
// {
//     const int saA = Acore.sa;
//     const int saB = Bcore.sa;
//     const int words = Abits.words_per_col;

//     if (saA <= 0 || saB <= 0 || words <= 0) return;

//     for (int kA = 0; kA < saA; ++kA) {
//         const int phys_col_a = Acore.aind[static_cast<std::size_t>(kA)];
//         if (phys_col_a < 0) continue;

//         const std::uint64_t *colA = Abits.column_bits(kA);
//         if (!colA) continue;

//         for (int kB = 0; kB < saB; ++kB) {
//             const int phys_col_b = Bcore.aind[static_cast<std::size_t>(kB)];
//             if (phys_col_b < 0) continue;

//             // Find active index for column phys_col_b in C
//             const int active_c = Ccore.acol[static_cast<std::size_t>(phys_col_b)];
//             if (active_c < 0) continue;

//             const std::uint64_t *colB = Bbits.column_bits(kB);
//             if (!colB) continue;

//             // Check if columns of A and B share any nonzero row
//             if (!bitmap_columns_intersect(colA, colB, words)) continue;

//             // C[phys_col_a, phys_col_b] is nonzero
//             Cbits.set_bit(active_c, phys_col_a);
//         }
//     }

// }

// // =============================================================================
// // core_ssyrk: A += B^T * B (symmetric rank-k update)
// // =============================================================================
// inline void core_ssyrk_old(sTiles::Uplo uplo,
//                        SemisparseTileMetaCore &Acore,
//                        const SemisparseTileMetaCore &Bcore,
//                        SymbolicTileBitmaskCore &Abits,
//                        const SymbolicTileBitmaskCore &Bbits)
// {
//     const int saB = Bcore.sa;
//     const int wordsB = Bbits.words_per_col;
//     const int widthA = static_cast<int>(Acore.acol.size());

//     if (saB <= 0 || wordsB <= 0 || widthA <= 0) return;

//     for (int k = 0; k < saB; ++k) {
//         const int idx_k = Bcore.aind[static_cast<std::size_t>(k)];
//         if (idx_k < 0 || idx_k >= widthA) continue;

//         const std::uint64_t *bits_k = Bbits.column_bits(k);
//         if (!bits_k) continue;

//         // Exploit symmetry: only compute m <= k
//         for (int m = 0; m <= k; ++m) {
//             const int idx_m = Bcore.aind[static_cast<std::size_t>(m)];
//             if (idx_m < 0 || idx_m >= widthA) continue;

//             const std::uint64_t *bits_m = Bbits.column_bits(m);
//             if (!bits_m) continue;

//             // Check if columns k and m of B share any nonzero row
//             if (!bitmap_columns_intersect(bits_k, bits_m, wordsB)) continue;

//             // Result at (idx_k, idx_m) and (idx_m, idx_k) - store according to uplo
//             int row_tgt, col_tgt;
//             if (uplo == sTiles::Uplo::Upper) {
//                 row_tgt = std::min(idx_k, idx_m);
//                 col_tgt = std::max(idx_k, idx_m);
//             } else {
//                 row_tgt = std::max(idx_k, idx_m);
//                 col_tgt = std::min(idx_k, idx_m);
//             }

//             const int active_col_a = Acore.acol[static_cast<std::size_t>(col_tgt)];
//             if (active_col_a >= 0) {
//                 Abits.set_bit(active_col_a, row_tgt);
//             }
//         }
//     }

// }

// // =============================================================================
// // core_spotrf: Cholesky factorization fill-in prediction
// // =============================================================================
// inline void core_spotrf_old(sTiles::Uplo uplo,
//                         const TileMetaCore &Atile,
//                         SemisparseTileMetaCore &core,
//                         SymbolicTileBitmaskCore &Abits)
// {
//     const int height = Atile.height;
//     const int sa = core.sa;
//     const int words = Abits.words_per_col;

//     if (sa <= 0 || height <= 0 || words <= 0) return;

//     std::vector<int> queue(static_cast<std::size_t>(height));

//     if (uplo == sTiles::Uplo::Upper) {
//         for (int j = 0; j < height; ++j) {
//             if (j >= static_cast<int>(core.acol.size())) break;
//             const int active_j = core.acol[static_cast<std::size_t>(j)];
//             if (active_j < 0) continue;

//             std::uint64_t* j_bits = Abits.column_bits(active_j);
//             if (!j_bits) continue;

//             for (int k = 0; k < j; ++k) {
//                 const std::uint64_t pivot_mask = (1ULL << (k % 64));
//                 if (!(j_bits[k / 64] & pivot_mask)) continue;

//                 for (int i = k + 1; i <= j; ++i) {
//                     if (i >= static_cast<int>(core.acol.size())) break;
//                     const int active_i = core.acol[static_cast<std::size_t>(i)];
//                     if (active_i < 0) continue;

//                     const std::uint64_t* col_i_bits = Abits.column_bits(active_i);
//                     if (!col_i_bits) continue;

//                     if (!(col_i_bits[k / 64] & pivot_mask)) continue;

//                     const int word = i / 64;
//                     const std::uint64_t row_mask = (1ULL << (i % 64));
//                     j_bits[word] |= row_mask;
//                 }
//             }

//             int count = extract_rows_from_bitmap_column(j_bits, height, words, queue.data());
//             for (int idx = 0; idx < count; ++idx) {
//                 const int row = queue[idx];
//                 if (row <= j) {
//                     const int bw = j - row;
//                     if (bw > core.upper_bw) {
//                         core.upper_bw = bw;
//                     }
//                 }
//             }
//         }
//     }
// }


// }
// #endif  // STILES_SYMBOLIC_CORE_SPARSE_KERNELS_HPP
