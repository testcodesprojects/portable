/**
 * @file tile_compare.hpp
 * @brief Tile comparison utilities for validation and debugging.
 *
 * Provides functions to compare sparse SmartTile factors against dense
 * tile representations for numerical validation and correctness checking.
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

#ifndef STILES_TILE_COMPARE_HPP
#define STILES_TILE_COMPARE_HPP

#ifdef SMART_TILES

#include <cmath>
#include <iostream>
#include <iomanip>


namespace sTiles { namespace debug {

// Compares a lower-triangular sparse factor L (in CSC: SmartTileData)
// against a dense upper-triangular factor U (column-major, leading dim ld),
// by checking U(j,i) ~= L(i,j) for all i >= j within tolerance.
// Returns true if all mapped entries match within tol.
inline bool compare_smart_L_to_dense_U(const sTiles::SmartTileData* L,
                                       const double* U,
                                       int ld,
                                       double tol = 1e-9,
                                       int max_reports = 10)
{
    if (!L || !U) {
        std::cout << "[SMART-CHECK] compare: null inputs L or U" << std::endl;
        return false;
    }
    if (!L->colptr || !L->rowind || !L->values) {
        std::cout << "[SMART-CHECK] compare: missing L structure/values (colptr/rowind/values)." << std::endl;
        return false;
    }
    const int n = L->cols;
    if (ld < n) {
        std::cout << "[SMART-CHECK] compare: invalid leading dimension (ld < n)." << std::endl;
        return false;
    }

    bool ok = true;
    int reported = 0;

    for (int j = 0; j < n; ++j) {
        const int p0 = L->colptr[j];
        const int p1 = L->colptr[j + 1];
        for (int p = p0; p < p1; ++p) {
            const int i = L->rowind[p];
            if (i < j) continue; // strictly lower/diagonal entries only
            const double lij = L->values[p];

            // Compare with U(j, i) (upper-triangular, so column = i, row = j)
            const double uji = U[j + i * ld];
            if (std::fabs(uji - lij) > tol) {
                if (reported < max_reports) {
                    std::cout << "[SMART-CHECK] Mismatch at L(" << i << "," << j << ") vs U(" << j << "," << i
                              << ") : L=" << lij << ", U=" << uji << ", diff=" << (uji - lij) << std::endl;
                    ++reported;
                }
                ok = false;
            }
        }
    }

    if (ok) {
        std::cout << "[SMART-CHECK] OK: SmartTile L matches dense U within tol=" << tol << std::endl;
    } else if (reported >= max_reports) {
        std::cout << "[SMART-CHECK] Additional mismatches suppressed." << std::endl;
    }

    return ok;
}

// Pretty-print a dense tile (column-major) for quick diagnostics.
// Prints up to max_rows x max_cols elements.
inline void print_dense_tile(const char* label,
                             const double* A,
                             int rows,
                             int cols,
                             int ld,
                             int max_rows = 8,
                             int max_cols = 8,
                             int precision = 6)
{
    if (!A) {
        std::cout << "[SMART-CHECK] " << (label ? label : "dense") << ": null pointer" << std::endl;
        return;
    }
    if (ld < rows) {
        std::cout << "[SMART-CHECK] " << (label ? label : "dense")
                  << ": invalid leading dimension ld=" << ld << " < rows=" << rows << std::endl;
        return;
    }
    const int pr = std::min(rows, max_rows);
    const int pc = std::min(cols, max_cols);
    std::cout.setf(std::ios::fixed);
    std::cout << "[SMART-CHECK] Dense tile " << (label ? label : "")
              << " (" << rows << "x" << cols << ", ld=" << ld << ") showing "
              << pr << "x" << pc << ":" << std::endl;
    std::streamsize oldp = std::cout.precision();
    std::cout.precision(precision);
    for (int i = 0; i < pr; ++i) {
        for (int j = 0; j < pc; ++j) {
            const double v = A[i + j * ld];
            std::cout << v;
            if (j + 1 < pc) std::cout << ", ";
        }
        if (pc < cols) std::cout << ", ...";
        std::cout << std::endl;
    }
    if (pr < rows) {
        std::cout << "..." << std::endl;
    }
    std::cout.precision(oldp);
}

// Compare a generic SmartTileData (CSC) against a dense tile D (column-major)
// by expanding only the SmartTile nonzeros and checking D(row, col) ~= val.
// rows, cols specify the sub-dimensions to check; ld is leading dimension of D.
inline bool compare_smart_to_dense(const sTiles::SmartTileData* S,
                                   const double* D,
                                   int rows,
                                   int cols,
                                   int ld,
                                   double tol = 1e-9,
                                   int max_reports = 10)
{
    if (!S || !D) {
        std::cout << "[SMART-CHECK] compare_smart_to_dense: null inputs." << std::endl;
        return false;
    }
    const int* colptr = S->colptr ? S->colptr : S->original_colptr;
    const int* rowind = S->rowind ? S->rowind : S->original_rowind;
    const double* vals = S->values ? S->values : S->original_values;
    if (!colptr || !rowind || !vals) {
        std::cout << "[SMART-CHECK] compare_smart_to_dense: missing structure/values." << std::endl;
        return false;
    }
    if (ld < rows) {
        std::cout << "[SMART-CHECK] compare_smart_to_dense: invalid ld (" << ld << ") < rows (" << rows << ")." << std::endl;
        return false;
    }

    const int use_cols = std::min(cols, S->cols);
    const int use_rows = std::min(rows, S->rows);
    bool ok = true;
    int reported = 0;

    for (int c = 0; c < use_cols; ++c) {
        const int p0 = colptr[c];
        const int p1 = colptr[c + 1];
        for (int p = p0; p < p1; ++p) {
            const int r = rowind[p];
            if (r >= use_rows) continue;
            const double sv = vals[p];
            const double dv = D[r + c * ld];
            if (std::fabs(dv - sv) > tol) {
                if (reported < max_reports) {
                    std::cout << "[SMART-CHECK] Mismatch at (r,c)=(" << r << "," << c
                              << ") : smart=" << sv << ", dense=" << dv
                              << ", diff=" << (dv - sv) << std::endl;
                    ++reported;
                }
                ok = false;
            }
        }
    }

    if (ok) {
        std::cout << "[SMART-CHECK] OK: SmartTile matches dense within tol=" << tol << std::endl;
    } else if (reported >= max_reports) {
        std::cout << "[SMART-CHECK] Additional mismatches suppressed." << std::endl;
    }
    return ok;
}

// Strict comparison using the SmartTile's active CSC structure (colptr/rowind/values)
// against a dense tile. Returns true on success, false on the first mismatch
// and prints a FAIL/SUCCESS diagnostic.
inline bool compare_smart_values_to_dense_tile(const sTiles::SmartTileData* smart,
                                               const double* dense,
                                               int ld,
                                               double tol = 1e-9)
{
    if (!smart || !dense) {
        std::cout << "[SMART-CHECK] FAIL: compare_smart_values_to_dense_tile received null inputs." << std::endl;
        return false;
    }
    if (!smart->colptr || !smart->rowind || !smart->values) {
        std::cout << "[SMART-CHECK] FAIL: SmartTile missing colptr/rowind/values for comparison." << std::endl;
        return false;
    }
    const int rows = smart->rows;
    const int cols = smart->cols;
    if (ld < rows) {
        std::cout << "[SMART-CHECK] FAIL: dense leading dimension " << ld
                  << " is smaller than SmartTile rows " << rows << "." << std::endl;
        return false;
    }

    if (smart->is_diagonal) {
        for (int col = 0; col < cols; ++col) {
            const int start = smart->colptr[col];
            const int end   = smart->colptr[col + 1];
            for (int p = start; p < end; ++p) {
                const int row = smart->rowind[p];
                if (row < col) {
                    std::cout << "[SMART-CHECK] FAIL: diagonal SmartTile stores entry above diagonal at ("
                              << row << "," << col << ")." << std::endl;
                    return false;
                }
                if (row >= rows) {
                    std::cout << "[SMART-CHECK] FAIL: SmartTile row index " << row
                              << " out of bounds for rows=" << rows << "." << std::endl;
                    return false;
                }
                const double smart_val = smart->values[p];
                const double dense_val = dense[col + row * ld]; // U(col,row) in dense upper storage
                if (std::fabs(dense_val - smart_val) > tol) {
                    std::cout << "[SMART-CHECK] FAIL: mismatch at L(" << row << "," << col
                              << ") vs U(" << col << "," << row << ") smart=" << smart_val
                              << " dense=" << dense_val
                              << " diff=" << (dense_val - smart_val) << std::endl;
                    return false;
                }
            }
        }
        std::cout << "[SMART-CHECK] success: diagonal SmartTile matches dense upper tile within tol="
                  << tol << std::endl;
        return true;
    } else {
        for (int col = 0; col < cols; ++col) {
            const int start = smart->colptr[col];
            const int end   = smart->colptr[col + 1];
            for (int p = start; p < end; ++p) {
                const int row = smart->rowind[p];
                if (row < 0 || row >= rows) {
                    std::cout << "[SMART-CHECK] FAIL: SmartTile row index " << row
                              << " out of bounds for rows=" << rows << "." << std::endl;
                    return false;
                }
                const double smart_val = smart->values[p];
                const double dense_val = dense[row + col * ld];
                if (std::fabs(dense_val - smart_val) > tol) {
                    std::cout << "[SMART-CHECK] FAIL: mismatch at (row,col)=(" << row << "," << col
                              << ") smart=" << smart_val << " dense=" << dense_val
                              << " diff=" << (dense_val - smart_val) << std::endl;
                    return false;
                }
            }
        }
        std::cout << "[SMART-CHECK] success: SmartTile matches dense tile within tol=" << tol << std::endl;
        return true;
    }
}

// Convenience: compare TRSM inputs succinctly.
// - rhs_tile (SmartTile, arbitrary pattern) vs rhs_dense (ld x cols)
// - diag_tile (SmartTile stores L) vs diag_dense_U (dense upper U)
inline void compare_trsm_inputs(const sTiles::SmartTile* rhs_tile,
                                const double* rhs_dense,
                                int rhs_rows,
                                int rhs_cols,
                                int rhs_ld,
                                const sTiles::SmartTile* diag_tile,
                                const double* diag_dense_U,
                                int diag_ld,
                                double tol = 1e-8,
                                int max_reports = 10)
{
    if (rhs_tile && rhs_tile->getTile() && rhs_dense) {
        compare_smart_to_dense(rhs_tile->getTile(), rhs_dense, rhs_rows, rhs_cols, rhs_ld, tol, max_reports);
    }
    if (diag_tile && diag_tile->getTile() && diag_dense_U) {
        compare_smart_L_to_dense_U(diag_tile->getTile(), diag_dense_U, diag_ld, tol, max_reports);
    }
}

// Convenience wrapper: print the entire dense tile (all rows/cols)
inline void print_tile_dense(const char* label,
                             const double* A,
                             int rows,
                             int cols,
                             int ld,
                             int precision = 6)
{
    print_dense_tile(label, A, rows, cols, ld, rows, cols, precision);
}

// Print a SmartTileData (CSC) with its (row, val) pairs per column.
inline void print_tile_smart(const char* label,
                             const sTiles::SmartTileData* S,
                             int max_cols = -1,
                             int precision = 6)
{
    if (!S) {
        std::cout << "[SMART-CHECK] " << (label ? label : "smart") << ": null tile" << std::endl;
        return;
    }
    const int* colptr = S->colptr ? S->colptr : S->original_colptr;
    const int* rowind = S->rowind ? S->rowind : S->original_rowind;
    const double* vals = S->values ? S->values : S->original_values;
    if (!colptr || !rowind) {
        std::cout << "[SMART-CHECK] " << (label ? label : "smart")
                  << ": missing structure (colptr/rowind)" << std::endl;
        return;
    }

    const int rows = S->rows;
    const int cols = S->cols;
    const int nnz  = S->nnz;

    std::cout << "[SMART-CHECK] Smart tile " << (label ? label : "")
              << " (" << rows << "x" << cols << ", nnz=" << nnz << ")" << std::endl;
    std::streamsize oldp = std::cout.precision();
    std::cout.setf(std::ios::fixed);
    std::cout.precision(precision);

    const int show_cols = (max_cols < 0) ? cols : std::min(cols, max_cols);
    for (int c = 0; c < show_cols; ++c) {
        const int p0 = colptr[c];
        const int p1 = colptr[c + 1];
        std::cout << "  col " << c << " (len=" << (p1 - p0) << "):";
        for (int p = p0; p < p1; ++p) {
            const int r = rowind[p];
            double v = 0.0;
            if (vals) v = vals[p];
            std::cout << " (" << r << ", " << v << ")";
        }
        std::cout << std::endl;
    }
    if (show_cols < cols) {
        std::cout << "  ..." << std::endl;
    }
    std::cout.precision(oldp);
}

}} // namespace sTiles::debug

#endif // SMART_TILES

#endif // STILES_TILE_COMPARE_HPP
