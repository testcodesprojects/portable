

#include <vector>
#include "../control/common.h"
#include "../control/stiles_control.hpp"  // for omp_dep_tracker_t and dep_* macros
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "../common/core_lapack.hpp"
#include "stiles_verify.hpp"
#include "../tile/core_kernels.hpp"
#include "../tile/meta.hpp"
#include "../common/stiles_config.hpp"

#include <cmath>
#include <omp.h>
// cblas is already included via core_lapack.hpp
#include <cstring>


// Address of tile (m, n) inside B.
//
//   get_block_col — B is col-major,  ldb = N    (row stride 1, col stride N)
//   get_block_row — B is row-major,  ldb = nrhs (row stride nrhs, col stride 1)
//
// Both return a pointer to B[m*tile_size, n*tile_size] interpreted in their
// respective layout. The arithmetic is mirrored so the two are easy to
// keep in sync.
inline static double* get_block_col(double* B, int N, int tile_size, int m, int n)
{
    size_t offset_in_elements = (size_t)(m * tile_size) + (size_t)(n * tile_size) * (size_t)N;
    return B + offset_in_elements;
}

inline static double* get_block_row(double* B, int nrhs, int tile_size, int m, int n)
{
    size_t offset_in_elements = (size_t)(m * tile_size) * (size_t)nrhs + (size_t)(n * tile_size);
    return B + offset_in_elements;
}

// B-layout helpers — templated on RowMajorB so the dispatcher chooses
// at runtime which instantiation runs (no compile-time #ifdef).
template<bool RowMajorB>
inline double* get_block_B(double* B, int N, int nrhs,
                           int tile_size, int row_tile, int col_tile)
{
    if constexpr (RowMajorB) {
        (void)N;
        // Row-major: block at (row_tile, col_tile) starts at
        //   B + row_tile * tile_size * nrhs + col_tile * tile_size
        return B
             + static_cast<std::size_t>(row_tile) * tile_size * nrhs
             + static_cast<std::size_t>(col_tile) * tile_size;
    } else {
        (void)nrhs;
        return get_block_col(B, N, tile_size, row_tile, col_tile);
    }
}

template<bool RowMajorB>
inline int ldb_B(int N, int nrhs)
{
    if constexpr (RowMajorB) { (void)N; return nrhs; }
    else                     { (void)nrhs; return N; }
}

// Diagonal banded solve wrapper. LAPACKE_dtbtrs with LAPACK_ROW_MAJOR
// internally transposes the banded factor + RHS via LAPACKE_malloc on
// every call — measured ~20-100% slowdown at (n=40, kd=5, nrhs=40).
// Cheaper to pack B into a tile_size x nrhs col-major panel, call the
// col-major LAPACKE_dtbtrs, then unpack back.
template<bool RowMajorB>
inline int banded_solve_via_panel(
    double* B_block, int diag_dim, int kd, int tempnn,
    const double* ss_diag, int ldab, int ldb_in)
{
    if constexpr (RowMajorB) {
        // Scratch panel is diag_dim x tempnn (col-major). A fixed [40*64]
        // stack buffer overflowed once tile_size>64 or nrhs>40 (e.g. ts120
        // nrhs=40 needs 120*40=4800 > 2560 -> stack smash / SIGBUS). Size it
        // dynamically via a reused thread-local buffer.
        thread_local std::vector<double> panel_buf;
        const std::size_t _need = static_cast<std::size_t>(diag_dim) * static_cast<std::size_t>(tempnn);
        if (panel_buf.size() < _need) panel_buf.resize(_need);
        double* panel = panel_buf.data();
        const int panel_ld = diag_dim;
        for (int j = 0; j < tempnn; ++j)
            for (int i = 0; i < diag_dim; ++i)
                panel[i + j * panel_ld] = B_block[i * ldb_in + j];

        lapack_int info = LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                         diag_dim, kd, tempnn,
                                         ss_diag, ldab, panel, panel_ld);

        for (int j = 0; j < tempnn; ++j)
            for (int i = 0; i < diag_dim; ++i)
                B_block[i * ldb_in + j] = panel[i + j * panel_ld];

        return static_cast<int>(info);
    } else {
        return static_cast<int>(LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                               diag_dim, kd, tempnn,
                                               ss_diag, ldab, B_block, ldb_in));
    }
}

template<bool RowMajorB>
inline int banded_solve_via_panel_notrans(
    double* B_block, int diag_dim, int kd, int tempnn,
    const double* ss_diag, int ldab, int ldb_in)
{
    if constexpr (RowMajorB) {
        // Scratch panel is diag_dim x tempnn (col-major). A fixed [40*64]
        // stack buffer overflowed once tile_size>64 or nrhs>40 (e.g. ts120
        // nrhs=40 needs 120*40=4800 > 2560 -> stack smash / SIGBUS). Size it
        // dynamically via a reused thread-local buffer.
        thread_local std::vector<double> panel_buf;
        const std::size_t _need = static_cast<std::size_t>(diag_dim) * static_cast<std::size_t>(tempnn);
        if (panel_buf.size() < _need) panel_buf.resize(_need);
        double* panel = panel_buf.data();
        const int panel_ld = diag_dim;
        for (int j = 0; j < tempnn; ++j)
            for (int i = 0; i < diag_dim; ++i)
                panel[i + j * panel_ld] = B_block[i * ldb_in + j];

        lapack_int info = LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N',
                                         diag_dim, kd, tempnn,
                                         ss_diag, ldab, panel, panel_ld);

        for (int j = 0; j < tempnn; ++j)
            for (int i = 0; i < diag_dim; ++i)
                B_block[i * ldb_in + j] = panel[i + j * panel_ld];

        return static_cast<int>(info);
    } else {
        return static_cast<int>(LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N',
                                               diag_dim, kd, tempnn,
                                               ss_diag, ldab, B_block, ldb_in));
    }
}

namespace sTiles{ namespace Process{

// Direct index for upper triangular tile (i, j) where i <= j
inline int dense_full_tile_index(int i, int j, int num_tiles) {
    return i * num_tiles - (i * (i - 1)) / 2 + (j - i);
}

void stiles_pdtrsm_forward_original(stiles_context_t *stile)
{
    // Unpack arguments from the sTiles context
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    double alpha = 1.0;
    double zone  = (double) 1.0;
    double mzone = (double)-1.0;
    // Use the unpacked arguments
    int tile_size = tiledMatrixA->tile_size;
    int N = tiledMatrixA->dim;

    int k, m, n;
    int next_k, next_m, next_n;
    int ldak;
    int tempkm, tempmm, tempnn;
    double lalpha;
    int index1;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size  + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size  + 1;

    int Adesc_lm1 = (N/tile_size);
    #define BLKLDD1(k) ( (k) < Adesc_lm1 ? tile_size : N % tile_size)

    ss_init(num_row_tiles, num_col_tiles, -1);
    k = 0;
    m = STILES_RANK;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;
    
    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += STILES_SIZE;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;
        tempmm = m == num_row_tiles-1 ? N-m*tile_size : tile_size;
        tempkm = k == num_row_tiles-1 ? N-k*tile_size : tile_size;
        lalpha = k == 0 ? alpha : zone;

        if (m == k) {
            ss_cond_wait(m, n, k-1);
            tempkm = k == num_row_tiles-1 ? N-k*tile_size : tile_size;
            ldak = BLKLDD1(k);
            index1 = tiledMatrixA->mapper.map_ij(k, k, tiledMatrixA->dimTiledMatrix);

            sTiles::core_dtrsm(sTiles::Side::Left,
                               sTiles::Uplo::Upper,
                               sTiles::Op::Trans,
                               sTiles::Diag::NonUnit,
                               tempkm, tempnn, lalpha,
                               tiledMatrixA->denseTiles[index1], ldak,
                               get_block_col(B, N, tile_size, k, n), N);
            ss_cond_set(k, n, k);
        } else {
            ss_cond_wait(k, n, k);
            ss_cond_wait(m, n, k-1);
            //if(tiledMatrixA->permutation_flags[k*(2*num_row_tiles-k-1)/2 + m]){
            if(tiledMatrixA->state.isActive(k, m, tiledMatrixA->dimTiledMatrix)){
                ldak = BLKLDD1(k);
                index1 = tiledMatrixA->mapper.map_ij(k, m, tiledMatrixA->dimTiledMatrix);

                sTiles::core_dgemm(sTiles::Op::Trans,
                                   sTiles::Op::NoTrans,
                                   tempmm, tempnn, tempkm,  mzone,
                                   tiledMatrixA->denseTiles[index1], ldak,
                                   get_block_col(B, N, tile_size, k, n), N,
                                   lalpha, get_block_col(B, N, tile_size, m, n), N);
            }
            ss_cond_set(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    ss_finalize();
}

void stiles_pdtrsm_backward_original(stiles_context_t *stile)
{
    // Unpack arguments from the sTiles context
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    int side = sTilesLeft;
    int diag = sTilesNonUnit;
    double alpha = 1.0;
    double zone  = (double) 1.0;
    double mzone = (double)-1.0;
    // Use the unpacked arguments
    int tile_size = tiledMatrixA->tile_size;
    int N = tiledMatrixA->dim;
    int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size  + 1;
    int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size  + 1;

    int Adesc_lm1 = (N/tile_size);
    // Note: BLKLDD1 was already defined in the forward function scope, so it's fine here.
    #define BLKLDD1(k) ( (k) < Adesc_lm1 ? tile_size : N % tile_size)

    int k, m, n;
    int next_k, next_m, next_n;
    int ldam, ldak;
    int tempkm, tempnn;
    double lalpha;
    int index1;

    ss_init(num_row_tiles, num_col_tiles, -1);
    k = 0;
    m = STILES_RANK;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += STILES_SIZE;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;
        lalpha = k == 0 ? alpha : zone;

        if (m == k) {
            ss_cond_wait(m, n, k-1);
            tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
            ldak = BLKLDD1(num_row_tiles-1-k);
            index1 = tiledMatrixA->mapper.map_ij((num_row_tiles-1-k), (num_row_tiles-1-k), tiledMatrixA->dimTiledMatrix);

            sTiles::core_dtrsm(sTiles::Side::Left,
                               sTiles::Uplo::Upper,
                               sTiles::Op::NoTrans,
                               sTiles::Diag::NonUnit,
                               tempkm, tempnn, lalpha,
                               tiledMatrixA->denseTiles[index1], ldak,
                               get_block_col(B, N, tile_size, num_row_tiles-1-k, n), N);
                              
            ss_cond_set(k, n, k);
        } else {
            ss_cond_wait(k, n, k);
            ss_cond_wait(m, n, k-1);
            //if(tiledMatrixA->permutation_flags[(num_row_tiles-1-m)*(2*num_row_tiles-(num_row_tiles-1-m)-1)/2 + (num_row_tiles-1-k)]){
            if(tiledMatrixA->state.isActive((num_row_tiles-1-m), (num_row_tiles-1-k), tiledMatrixA->dimTiledMatrix)){

                tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
                ldam = BLKLDD1(num_row_tiles-1-m);
                index1 = tiledMatrixA->mapper.map_ij((num_row_tiles-1-m), (num_row_tiles-1-k), tiledMatrixA->dimTiledMatrix);

                sTiles::core_dgemm(sTiles::Op::NoTrans,
                                   sTiles::Op::NoTrans,
                                   tile_size, tempnn, tempkm, mzone,
                                   tiledMatrixA->denseTiles[index1], ldam,
                                   get_block_col(B, N, tile_size, num_row_tiles-1-k, n),  N, lalpha,
                                   get_block_col(B, N, tile_size, num_row_tiles-1-m, n), N);
            }
            ss_cond_set(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    ss_finalize();
}

void stiles_pdtrsm_forward_dense(stiles_context_t *stile){
    // Unpack arguments from the sTiles context
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    double alpha = 1.0;
    double zone  = (double) 1.0;
    double mzone = (double)-1.0;
    // Use the unpacked arguments
    int tile_size = tiledMatrixA->tile_size;
    int N = tiledMatrixA->dim;

    int k, m, n;
    int next_k, next_m, next_n;
    int ldak;
    int tempkm, tempmm, tempnn;
    double lalpha;
    int index1;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size  + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size  + 1;

    int Adesc_lm1 = (N/tile_size);
    #define BLKLDD1(k) ( (k) < Adesc_lm1 ? tile_size : N % tile_size)

    ss_init(num_row_tiles, num_col_tiles, -1);
    k = 0;
    m = STILES_RANK;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += STILES_SIZE;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;
        tempmm = m == num_row_tiles-1 ? N-m*tile_size : tile_size;
        tempkm = k == num_row_tiles-1 ? N-k*tile_size : tile_size;
        lalpha = k == 0 ? alpha : zone;

        if (m == k) {
            ss_cond_wait(m, n, k-1);
            tempkm = k == num_row_tiles-1 ? N-k*tile_size : tile_size;
            ldak = BLKLDD1(k);
            index1 = tiledMatrixA->mapper.map_ij(k, k, tiledMatrixA->dimTiledMatrix);

            double* diag_tile = (index1 >= 0 && tiledMatrixA->denseTiles) ? tiledMatrixA->denseTiles[index1] : nullptr;
            if (diag_tile) {
                sTiles::core_dtrsm(sTiles::Side::Left,
                                   sTiles::Uplo::Upper,
                                   sTiles::Op::Trans,
                                   sTiles::Diag::NonUnit,
                                   tempkm, tempnn, lalpha,
                                   diag_tile, ldak,
                                   get_block_col(B, N, tile_size, k, n), N);
            }
            ss_cond_set(k, n, k);
        } else {
            ss_cond_wait(k, n, k);
            ss_cond_wait(m, n, k-1);
            //if(tiledMatrixA->permutation_flags[k*(2*num_row_tiles-k-1)/2 + m]){
            if(tiledMatrixA->state.isActive(k, m, tiledMatrixA->dimTiledMatrix)){
                ldak = BLKLDD1(k);
                index1 = tiledMatrixA->mapper.map_ij(k, m, tiledMatrixA->dimTiledMatrix);

                double* offdiag_tile = (index1 >= 0 && tiledMatrixA->denseTiles) ? tiledMatrixA->denseTiles[index1] : nullptr;
                if (offdiag_tile) {
                    sTiles::core_dgemm(sTiles::Op::Trans,
                                       sTiles::Op::NoTrans,
                                       tempmm, tempnn, tempkm,  mzone,
                                       offdiag_tile, ldak,
                                       get_block_col(B, N, tile_size, k, n), N,
                                       lalpha, get_block_col(B, N, tile_size, m, n), N);
                }
            }
            ss_cond_set(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    ss_finalize();
}

void stiles_pdtrsm_backward_dense(stiles_context_t *stile){
    // Unpack arguments from the sTiles context
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    int side = sTilesLeft;
    int diag = sTilesNonUnit;
    double alpha = 1.0;
    double zone  = (double) 1.0;
    double mzone = (double)-1.0;
    // Use the unpacked arguments
    int tile_size = tiledMatrixA->tile_size;
    int N = tiledMatrixA->dim;
    int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size  + 1;
    int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size  + 1;

    int Adesc_lm1 = (N/tile_size);
    // Note: BLKLDD1 was already defined in the forward function scope, so it's fine here.
    #define BLKLDD1(k) ( (k) < Adesc_lm1 ? tile_size : N % tile_size)

    int k, m, n;
    int next_k, next_m, next_n;
    int ldam, ldak;
    int tempkm, tempnn;
    double lalpha;
    int index1;

    ss_init(num_row_tiles, num_col_tiles, -1);
    k = 0;
    m = STILES_RANK;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += STILES_SIZE;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;
        lalpha = k == 0 ? alpha : zone;

        if (m == k) {
            ss_cond_wait(m, n, k-1);
            tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
            ldak = BLKLDD1(num_row_tiles-1-k);
            index1 = tiledMatrixA->mapper.map_ij((num_row_tiles-1-k), (num_row_tiles-1-k), tiledMatrixA->dimTiledMatrix);

            double* diag_tile = (index1 >= 0 && tiledMatrixA->denseTiles) ? tiledMatrixA->denseTiles[index1] : nullptr;
            if (diag_tile) {
                sTiles::core_dtrsm(sTiles::Side::Left,
                                   sTiles::Uplo::Upper,
                                   sTiles::Op::NoTrans,
                                   sTiles::Diag::NonUnit,
                                   tempkm, tempnn, lalpha,
                                   diag_tile, ldak,
                                   get_block_col(B, N, tile_size, num_row_tiles-1-k, n), N);
            }
            ss_cond_set(k, n, k);
        } else {
            ss_cond_wait(k, n, k);
            ss_cond_wait(m, n, k-1);
            //if(tiledMatrixA->permutation_flags[(num_row_tiles-1-m)*(2*num_row_tiles-(num_row_tiles-1-m)-1)/2 + (num_row_tiles-1-k)]){
            if(tiledMatrixA->state.isActive((num_row_tiles-1-m), (num_row_tiles-1-k), tiledMatrixA->dimTiledMatrix)){

                tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
                ldam = BLKLDD1(num_row_tiles-1-m);
                index1 = tiledMatrixA->mapper.map_ij((num_row_tiles-1-m), (num_row_tiles-1-k), tiledMatrixA->dimTiledMatrix);

                double* offdiag_tile = (index1 >= 0 && tiledMatrixA->denseTiles) ? tiledMatrixA->denseTiles[index1] : nullptr;
                if (offdiag_tile) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::NoTrans,
                                       tile_size, tempnn, tempkm, mzone,
                                       offdiag_tile, ldam,
                                       get_block_col(B, N, tile_size, num_row_tiles-1-k, n),  N, lalpha,
                                       get_block_col(B, N, tile_size, num_row_tiles-1-m, n), N);
                }
            }
            ss_cond_set(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    ss_finalize();
}

void stiles_pdtrsm_forward_dense_single(stiles_context_t *stile, int variant)
{
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    const int N = tiledMatrixA->dim;

    // Variant 1: Single tile - just one TRSM call
    if (variant == 1) {
        if (STILES_RANK == 0 && tiledMatrixA->denseTiles && tiledMatrixA->denseTiles[0]) {
            double* tile = tiledMatrixA->denseTiles[0];
            // Solve L^T * X = B (forward substitution with upper triangular transpose)
            sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper,
                               sTiles::Op::Trans, sTiles::Diag::NonUnit,
                               N, nrhs, 1.0, tile, N, B, N);
        }
        return;
    }

    // Variant 2: Use the tiled implementation
    sTiles::Process::stiles_pdtrsm_forward_dense(stile);
}

void stiles_pdtrsm_backward_dense_single(stiles_context_t *stile, int variant)
{
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    const int N = tiledMatrixA->dim;

    // Variant 1: Single tile - just one TRSM call
    if (variant == 1) {
        if (STILES_RANK == 0 && tiledMatrixA->denseTiles && tiledMatrixA->denseTiles[0]) {
            double* tile = tiledMatrixA->denseTiles[0];
            // Solve L * X = B (backward substitution with upper triangular no-transpose)
            sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper,
                               sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                               N, nrhs, 1.0, tile, N, B, N);
        }
        return;
    }

    // Variant 2: Use the tiled implementation
    sTiles::Process::stiles_pdtrsm_backward_dense(stile);
}

void stiles_pdtrsm_forward_semisparse(stiles_context_t *stile)
{
    // Unpack arguments from the sTiles context
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    const double zone  = 1.0;
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    // Get workspace for temporary GEMM results
    const int rank = STILES_RANK;
    const int workspace_rank = rank + stile->workspace_offset;
    if (workspace_rank >= tiledMatrixA->num_workspaces) {
        sTiles::Logger::error("[trsm] workspace_rank ", workspace_rank, " >= num_workspaces ", tiledMatrixA->num_workspaces);
        return;
    }
    double* tmp_tile = tiledMatrixA->workspaces[workspace_rank]->aligned_tile();

    int k, m, n;
    int next_k, next_m, next_n;
    int tempkm, tempmm, tempnn;
    int index1;

    ss_init(num_row_tiles, num_col_tiles, -1);
    k = 0;
    m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += STILES_SIZE;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;
        tempmm = m == num_row_tiles-1 ? N-m*tile_size : tile_size;
        tempkm = k == num_row_tiles-1 ? N-k*tile_size : tile_size;

        if (m == k) {
            // Diagonal tile: solve L^T * X = B using triangular solver
            ss_cond_wait(m, n, k-1);
            index1 = tiledMatrixA->mapper.map_ij(k, k, tiledMatrixA->dimTiledMatrix);

            double* ss_diag = tiledMatrixA->chunkedDenseTiles ? tiledMatrixA->chunkedDenseTiles[index1] : nullptr;
            if (ss_diag) {
                const sTiles::SemisparseTileMetaCore& semi_diag = tiledMatrixA->semisparseTileMetaCore[index1];
                const sTiles::TileMetaCore& meta_diag = tiledMatrixA->tileMetaCore[index1];
                const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tempkm;
                const int kd = semi_diag.upper_bw;

                double* B_block = get_block_col(B, N, tile_size, k, n);

                if (kd >= 0) {
                    // Banded format: use dtbtrs
                    const int ldab = kd + 1;
                    LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                   diag_dim, kd, tempnn, ss_diag, ldab, B_block, N);
                } else {
                    // Dense format: use dtrsm
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::Trans,
                                       sTiles::Diag::NonUnit,
                                       diag_dim, tempnn, 1.0,
                                       ss_diag, diag_dim,
                                       B_block, N);
                }
            }
            ss_cond_set(k, n, k);
        } else {
            // Off-diagonal tile: B[m] -= A[k,m]^T * B[k]
            ss_cond_wait(k, n, k);
            ss_cond_wait(m, n, k-1);

            if(tiledMatrixA->state.isActive(k, m, tiledMatrixA->dimTiledMatrix)){
                index1 = tiledMatrixA->mapper.map_ij(k, m, tiledMatrixA->dimTiledMatrix);

                double* ss_tile = tiledMatrixA->chunkedDenseTiles ? tiledMatrixA->chunkedDenseTiles[index1] : nullptr;
                if (ss_tile) {
                    const sTiles::SemisparseTileMetaCore& semi = tiledMatrixA->semisparseTileMetaCore[index1];
                    const int active_cols = semi.sa;

                    if (active_cols > 0) {
                        double* B_k = get_block_col(B, N, tile_size, k, n);
                        double* B_m = get_block_col(B, N, tile_size, m, n);

                        // Compute GEMM: tmp = A[k,m]^T * B_k
                        // A[k,m] is tempkm x active_cols (column-compressed)
                        // A[k,m]^T is active_cols x tempkm
                        // B_k is tempkm x tempnn (leading dim N)
                        // Result tmp is active_cols x tempnn
                        const int ld_tmp = active_cols;
                        sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                          active_cols, tempnn, tempkm,
                                          mzone, ss_tile, tempkm,
                                          B_k, N,
                                          0.0, tmp_tile, ld_tmp);

                        // Scatter tmp to B_m using aind mapping
                        // tmp[i, j] goes to B_m[aind[i], j]
                        const int aind_size = static_cast<int>(semi.aind.size());
                        for (int j = 0; j < tempnn; ++j) {
                            for (int i = 0; i < active_cols && i < aind_size; ++i) {
                                const int out_row = semi.aind[i];
                                if (out_row >= 0 && out_row < tempmm) {
                                    B_m[out_row + j * N] += tmp_tile[i + j * ld_tmp];
                                }
                            }
                        }
                    }
                }
            }
            ss_cond_set(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    ss_finalize();
}

void stiles_pdtrsm_backward_semisparse(stiles_context_t *stile)
{
    // Unpack arguments from the sTiles context
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    // Get workspace for temporary gather buffer
    // Note: We only need one buffer at a time (gather then GEMM), so reuse tmp_tile
    const int rank = STILES_RANK;
    const int workspace_rank = rank + stile->workspace_offset;
    if (workspace_rank >= tiledMatrixA->num_workspaces) {
        sTiles::Logger::error("[trsm_semi] workspace_rank ", workspace_rank, " >= num_workspaces ", tiledMatrixA->num_workspaces);
        return;
    }
    double* tmp_gather = tiledMatrixA->workspaces[workspace_rank]->aligned_tile();

    int k, m, n;
    int next_k, next_m, next_n;
    int tempkm, tempnn, tempmm;
    int index1;

    ss_init(num_row_tiles, num_col_tiles, -1);
    k = 0;
    m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += STILES_SIZE;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;

        const int actual_k = num_row_tiles - 1 - k;
        const int actual_m = num_row_tiles - 1 - m;

        if (m == k) {
            // Diagonal tile: solve L * X = B (no transpose) using triangular solver
            ss_cond_wait(m, n, k-1);
            tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
            index1 = tiledMatrixA->mapper.map_ij(actual_k, actual_k, tiledMatrixA->dimTiledMatrix);

            double* ss_diag = tiledMatrixA->chunkedDenseTiles ? tiledMatrixA->chunkedDenseTiles[index1] : nullptr;
            if (ss_diag) {
                const sTiles::SemisparseTileMetaCore& semi_diag = tiledMatrixA->semisparseTileMetaCore[index1];
                const sTiles::TileMetaCore& meta_diag = tiledMatrixA->tileMetaCore[index1];
                const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tempkm;
                const int kd = semi_diag.upper_bw;

                double* B_block = get_block_col(B, N, tile_size, actual_k, n);

                if (kd >= 0) {
                    // Banded format: use dtbtrs
                    const int ldab = kd + 1;
                    LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N',
                                   diag_dim, kd, tempnn, ss_diag, ldab, B_block, N);
                } else {
                    // Dense format: use dtrsm
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans,
                                       sTiles::Diag::NonUnit,
                                       diag_dim, tempnn, 1.0,
                                       ss_diag, diag_dim,
                                       B_block, N);
                }
            }
            ss_cond_set(k, n, k);
        } else {
            // Off-diagonal tile: B[actual_m] -= A[actual_m, actual_k] * B[actual_k]
            ss_cond_wait(k, n, k);
            ss_cond_wait(m, n, k-1);

            if(tiledMatrixA->state.isActive(actual_m, actual_k, tiledMatrixA->dimTiledMatrix)){
                tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
                tempmm = actual_m == num_row_tiles-1 ? N-actual_m*tile_size : tile_size;
                index1 = tiledMatrixA->mapper.map_ij(actual_m, actual_k, tiledMatrixA->dimTiledMatrix);

                double* ss_tile = tiledMatrixA->chunkedDenseTiles ? tiledMatrixA->chunkedDenseTiles[index1] : nullptr;
                if (ss_tile) {
                    const sTiles::SemisparseTileMetaCore& semi = tiledMatrixA->semisparseTileMetaCore[index1];
                    const int active_cols = semi.sa;

                    if (active_cols > 0) {
                        double* B_k = get_block_col(B, N, tile_size, actual_k, n);
                        double* B_m = get_block_col(B, N, tile_size, actual_m, n);

                        // Gather B_k values at positions given by aind
                        // tmp_gather is active_cols x tempnn
                        const int aind_size = static_cast<int>(semi.aind.size());
                        const int ld_gather = active_cols;
                        for (int j = 0; j < tempnn; ++j) {
                            for (int i = 0; i < active_cols && i < aind_size; ++i) {
                                const int b_row = semi.aind[i];
                                if (b_row >= 0 && b_row < tempkm) {
                                    tmp_gather[i + j * ld_gather] = B_k[b_row + j * N];
                                } else {
                                    tmp_gather[i + j * ld_gather] = 0.0;
                                }
                            }
                        }

                        // Compute GEMM: B_m -= A * tmp_gather
                        // A is tempmm x active_cols
                        // tmp_gather is active_cols x tempnn
                        // Result is tempmm x tempnn, goes directly to B_m
                        sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                          tempmm, tempnn, active_cols,
                                          mzone, ss_tile, tempmm,
                                          tmp_gather, ld_gather,
                                          1.0, B_m, N);
                    }
                }
            }
            ss_cond_set(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    ss_finalize();
}

template<bool RowMajorB>
static void stiles_pdtrsm_forward_semisparse_tasked_impl(stiles_context_t *stile)
{
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    const auto& tasks    = sTiles::get_solve_fwd_tasks(tiledMatrixA);
    const auto& offsets  = sTiles::get_solve_fwd_offsets(tiledMatrixA);
    const auto& expected = sTiles::get_solve_fwd_expected(tiledMatrixA);

    const int rank = STILES_RANK;

    if (offsets.empty() ||
        static_cast<std::size_t>(rank) + 1 >= offsets.size() ||
        expected.empty())
    {
        stiles_pdtrsm_forward_semisparse(stile);
        return;
    }

    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int workspace_rank = rank + stile->workspace_offset;
    if (workspace_rank >= tiledMatrixA->num_workspaces) {
        sTiles::Logger::error("[trsm_tasks] workspace_rank ", workspace_rank,
                              " >= num_workspaces ", tiledMatrixA->num_workspaces);
        return;
    }
    double* tmp_tile = tiledMatrixA->workspaces[workspace_rank]->aligned_tile();

    const int task_start = offsets[static_cast<std::size_t>(rank)];
    const int task_end   = offsets[static_cast<std::size_t>(rank) + 1];

    // ──────────── Sparse iteration + update-counter sync (multi-RHS) ────────────
    // Same protocol as _tasked_1rhs (sparse iteration, ss[m] update counter
    // with expected[m]+1 = "row solved" sentinel). Per task we process ALL
    // RHS column-blocks in a single BLAS-3 call — no n-axis in the sync.
    //
    // ss[m] is shared across all RHS columns: when row m is solved (after
    // all expected updates land + diagonal trsm done), the sentinel
    // publishes B[m, :] for ALL nrhs columns at once.
    ss_init(num_row_tiles, 1, 0);
    volatile int* __restrict__ ss = stile->ss_progress;
    const int* __restrict__ exp_arr = expected.data();

    for (int idx = task_start; idx < task_end; ++idx) {
        if (stile->ss_abort) break;
        const auto& task = tasks[static_cast<std::size_t>(idx)];
        const int type     = task[0];
        const int k        = task[1];
        const int m        = task[2];
        const int tile_idx = task[3];

        const int tempmm = (m == num_row_tiles - 1) ? (N - m * tile_size) : tile_size;
        const int tempkm = (k == num_row_tiles - 1) ? (N - k * tile_size) : tile_size;

        if (type == 1) {
            // ── Diagonal: solve U^T X = B on this block, all RHS columns ──
            const int exp_m = exp_arr[m];
            {
                int _spin_ct = 0;
                while (!stile->ss_abort && ss[m] < exp_m) {
                    hpc_pause_hybrid(_spin_ct);
                }
                if (stile->ss_abort) break;
            }

            if (tile_idx >= 0) {
                double* ss_diag = tiledMatrixA->chunkedDenseTiles
                    ? tiledMatrixA->chunkedDenseTiles[tile_idx] : nullptr;
                if (ss_diag) {
                    const sTiles::SemisparseTileMetaCore& semi_diag =
                        tiledMatrixA->semisparseTileMetaCore[tile_idx];
                    const sTiles::TileMetaCore& meta_diag =
                        tiledMatrixA->tileMetaCore[tile_idx];
                    const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tempkm;
                    const int kd = semi_diag.upper_bw;

                    // One BLAS call covers ALL RHS columns (n = 0..num_col_tiles-1).
                    for (int n = 0; n < num_col_tiles; ++n) {
                        const int tempnn = (n == num_col_tiles - 1)
                                              ? (nrhs - n * tile_size) : tile_size;
                        double* B_block = get_block_B<RowMajorB>(B, N, nrhs, tile_size, k, n);
                        if (kd >= 0) {
                            const int ldab = kd + 1;
                            banded_solve_via_panel<RowMajorB>(B_block, diag_dim, kd, tempnn,
                                                              ss_diag, ldab, ldb_B<RowMajorB>(N, nrhs));
                        } else {
                            sTiles::core_dtrsm(sTiles::Side::Left,
                                               sTiles::Uplo::Upper,
                                               sTiles::Op::Trans,
                                               sTiles::Diag::NonUnit,
                                               diag_dim, tempnn, 1.0,
                                               ss_diag, diag_dim,
                                               B_block, ldb_B<RowMajorB>(N, nrhs));
                        }
                    }
                }
            }
            __atomic_store_n(&ss[m], exp_m + 1, __ATOMIC_RELEASE);

        } else if (type == 2) {
            // ── Off-diagonal: B[m] -= A[k,m]^T * B[k], all RHS columns ──
            const int exp_k = exp_arr[k];
            {
                int _spin_ct = 0;
                while (!stile->ss_abort && ss[k] < exp_k + 1) {
                    hpc_pause_hybrid(_spin_ct);
                }
                if (stile->ss_abort) break;
            }

            if (tile_idx >= 0) {
                double* ss_tile = tiledMatrixA->chunkedDenseTiles
                    ? tiledMatrixA->chunkedDenseTiles[tile_idx] : nullptr;
                if (ss_tile) {
                    const sTiles::SemisparseTileMetaCore& semi =
                        tiledMatrixA->semisparseTileMetaCore[tile_idx];
                    const int active_cols = semi.sa;
                    if (active_cols > 0) {
                        const int ld_tmp = active_cols;
                        const int aind_size = static_cast<int>(semi.aind.size());

                        for (int n = 0; n < num_col_tiles; ++n) {
                            const int tempnn = (n == num_col_tiles - 1)
                                                  ? (nrhs - n * tile_size) : tile_size;
                            double* B_k = get_block_B<RowMajorB>(B, N, nrhs, tile_size, k, n);
                            double* B_m = get_block_B<RowMajorB>(B, N, nrhs, tile_size, m, n);

                            if constexpr (RowMajorB) {
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::Trans,
                                                  active_cols, tempnn, tempkm,
                                                  mzone, ss_tile, tempkm,
                                                  B_k, ldb_B<RowMajorB>(N, nrhs),
                                                  0.0, tmp_tile, ld_tmp);
                            } else {
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                  active_cols, tempnn, tempkm,
                                                  mzone, ss_tile, tempkm,
                                                  B_k, N,
                                                  0.0, tmp_tile, ld_tmp);
                            }

                            for (int j = 0; j < tempnn; ++j) {
                                for (int i = 0; i < active_cols && i < aind_size; ++i) {
                                    const int out_row = semi.aind[i];
                                    if (out_row >= 0 && out_row < tempmm) {
                                        if constexpr (RowMajorB) {
                                            B_m[static_cast<std::size_t>(out_row) * nrhs + j] += tmp_tile[i + j * ld_tmp];
                                        } else {
                                            B_m[out_row + static_cast<std::size_t>(j) * N] += tmp_tile[i + j * ld_tmp];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            __atomic_fetch_add(&ss[m], 1, __ATOMIC_RELEASE);
        }
    }

    ss_finalize();
}

// Runtime dispatcher: picks row/col instantiation based on scheme flag.
void stiles_pdtrsm_forward_semisparse_tasked(stiles_context_t *stile)
{
    TiledMatrix *tiledMatrixA; double *B; int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);
    if (tiledMatrixA && tiledMatrixA->prefer_row_layout)
        stiles_pdtrsm_forward_semisparse_tasked_impl<true>(stile);
    else
        stiles_pdtrsm_forward_semisparse_tasked_impl<false>(stile);
}

void stiles_pdtrsm_forward_semisparse_tasked_1rhs(stiles_context_t *stile)
{
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);
    (void)nrhs;  // single-RHS contract

    const auto& tasks    = sTiles::get_solve_fwd_tasks(tiledMatrixA);
    const auto& offsets  = sTiles::get_solve_fwd_offsets(tiledMatrixA);
    const auto& expected = sTiles::get_solve_fwd_expected(tiledMatrixA);

    const int rank = STILES_RANK;

    // Fall back to legacy if precomputed data isn't available.
    if (offsets.empty() ||
        static_cast<std::size_t>(rank) + 1 >= offsets.size() ||
        expected.empty())
    {
        stiles_pdtrsm_forward_semisparse(stile);
        return;
    }

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N - 1) / tile_size + 1;

    const int workspace_rank = rank + stile->workspace_offset;
    if (workspace_rank >= tiledMatrixA->num_workspaces) {
        sTiles::Logger::error("[trsm_tasks] workspace_rank ", workspace_rank,
                              " >= num_workspaces ", tiledMatrixA->num_workspaces);
        return;
    }

    const int task_start = offsets[static_cast<std::size_t>(rank)];
    const int task_end   = offsets[static_cast<std::size_t>(rank) + 1];

    // ──────────── Sparse iteration + update-counter sync ────────────
    // Drop the dense (k, m) walk. Iterate ONLY the task list, which has
    // been row-affinity-rebinned in symbolic (every (*, m) and (m, m)
    // lands on rank m % a), so each rank exclusively owns each row m's
    // writes — no B_m race.
    //
    // ss[m] semantics:
    //   value v in [0, expected[m]]  →  v off-diagonal updates landed
    //   value expected[m] + 1        →  row m diagonal-solved (sentinel)
    //
    // Off-diag at (k, m): wait ss[k] >= expected[k]+1, BLAS, ss[m]++.
    // Diagonal at (m, m): wait ss[m] >= expected[m],  BLAS, ss[m] = expected[m]+1.
    //
    // Pre-fix this kernel walked num_row_tiles² (k, m) slots (~700 ms
    // on sem_n100000); post-fix it runs ~numActiveTiles BLAS calls.
    ss_init(num_row_tiles, 1, 0);
    volatile int* __restrict__ ss = stile->ss_progress;
    const int* __restrict__ exp_arr = expected.data();

    for (int idx = task_start; idx < task_end; ++idx) {
        if (stile->ss_abort) break;
        const auto& task = tasks[static_cast<std::size_t>(idx)];
        const int type     = task[0];  // 1=diagonal, 2=off-diagonal
        const int k        = task[1];
        const int m        = task[2];
        const int tile_idx = task[3];

        const int tempmm = (m == num_row_tiles - 1) ? (N - m * tile_size) : tile_size;
        const int tempkm = (k == num_row_tiles - 1) ? (N - k * tile_size) : tile_size;

        if (type == 1) {
            // ── Diagonal: solve L^T x = b on this block ──
            const int exp_m = exp_arr[m];
            {
                int _spin_ct = 0;
                while (!stile->ss_abort && ss[m] < exp_m) {
                    hpc_pause_hybrid(_spin_ct);
                }
                if (stile->ss_abort) break;
            }

            if (tile_idx >= 0) {
                double* ss_diag = tiledMatrixA->chunkedDenseTiles
                    ? tiledMatrixA->chunkedDenseTiles[tile_idx] : nullptr;
                if (ss_diag) {
                    const sTiles::SemisparseTileMetaCore& semi_diag =
                        tiledMatrixA->semisparseTileMetaCore[tile_idx];
                    const sTiles::TileMetaCore& meta_diag =
                        tiledMatrixA->tileMetaCore[tile_idx];
                    const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tempkm;
                    const int kd = semi_diag.upper_bw;

                    double* B_block = B + k * tile_size;
                    if (kd >= 0) {
                        const int ldab = kd + 1;
                        cblas_dtbsv(CblasColMajor,
                                    CblasUpper, CblasTrans, CblasNonUnit,
                                    diag_dim, kd, ss_diag, ldab,
                                    B_block, /*incx=*/1);
                    } else {
                        cblas_dtrsv(CblasColMajor,
                                    CblasUpper, CblasTrans, CblasNonUnit,
                                    diag_dim, ss_diag, diag_dim,
                                    B_block, /*incx=*/1);
                    }
                }
            }
            // Publish "row m solved" sentinel; release ensures B_m is visible.
            __atomic_store_n(&ss[m], exp_m + 1, __ATOMIC_RELEASE);

        } else if (type == 2) {
            // ── Off-diagonal: B_m -= A^T · B_k ──
            const int exp_k = exp_arr[k];
            {
                int _spin_ct = 0;
                while (!stile->ss_abort && ss[k] < exp_k + 1) {
                    hpc_pause_hybrid(_spin_ct);
                }
                if (stile->ss_abort) break;
            }

            if (tile_idx >= 0) {
                double* ss_tile = tiledMatrixA->chunkedDenseTiles
                    ? tiledMatrixA->chunkedDenseTiles[tile_idx] : nullptr;
                if (ss_tile) {
                    const sTiles::SemisparseTileMetaCore& semi =
                        tiledMatrixA->semisparseTileMetaCore[tile_idx];
                    const int active_cols = semi.sa;
                    if (active_cols > 0) {
                        const double* __restrict__ B_k = B + k * tile_size;
                        double*       __restrict__ B_m = B + m * tile_size;
                        const int* __restrict__ aind = semi.aind.data();
                        const int aind_size = static_cast<int>(semi.aind.size());

                        const int active_pair = active_cols & ~1;
                        int i = 0;
                        for (; i < active_pair; i += 2) {
                            const int row0 = aind[i];
                            const int row1 = aind[i + 1];
                            const bool in0 = (row0 >= 0 && row0 < tempmm);
                            const bool in1 = (row1 >= 0 && row1 < tempmm);
                            if (!in0 && !in1) continue;
                            const double* __restrict__ col_a_0 = ss_tile + static_cast<std::size_t>(i)     * tempkm;
                            const double* __restrict__ col_a_1 = ss_tile + static_cast<std::size_t>(i + 1) * tempkm;
                            double dot0 = 0.0, dot1 = 0.0;
                            #pragma omp simd reduction(+:dot0,dot1)
                            for (int r = 0; r < tempkm; ++r) {
                                const double bv = B_k[r];
                                dot0 += col_a_0[r] * bv;
                                dot1 += col_a_1[r] * bv;
                            }
                            if (in0) B_m[row0] -= dot0;
                            if (in1) B_m[row1] -= dot1;
                        }
                        for (; i < active_cols && i < aind_size; ++i) {
                            const int row = aind[i];
                            if (row < 0 || row >= tempmm) continue;
                            const double* __restrict__ col_a = ss_tile + static_cast<std::size_t>(i) * tempkm;
                            double dot = 0.0;
                            #pragma omp simd reduction(+:dot)
                            for (int r = 0; r < tempkm; ++r) dot += col_a[r] * B_k[r];
                            B_m[row] -= dot;
                        }
                    }
                }
            }
            // Atomic increment in case another rank also targets row m
            // (shouldn't happen with row-affinity rebin, but cheap safety).
            __atomic_fetch_add(&ss[m], 1, __ATOMIC_RELEASE);
        }
    }

    ss_finalize();
}

template<bool RowMajorB>
static void stiles_pdtrsm_backward_semisparse_tasked_impl(stiles_context_t *stile)
{
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    const auto& tasks    = sTiles::get_solve_bwd_tasks(tiledMatrixA);
    const auto& offsets  = sTiles::get_solve_bwd_offsets(tiledMatrixA);
    const auto& expected = sTiles::get_solve_bwd_expected(tiledMatrixA);

    const int rank = STILES_RANK;

    if (offsets.empty() ||
        static_cast<std::size_t>(rank) + 1 >= offsets.size() ||
        expected.empty())
    {
        stiles_pdtrsm_backward_semisparse(stile);
        return;
    }

    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int workspace_rank = rank + stile->workspace_offset;
    if (workspace_rank >= tiledMatrixA->num_workspaces) {
        sTiles::Logger::error("[trsm_semi_tasks] workspace_rank ", workspace_rank,
                              " >= num_workspaces ", tiledMatrixA->num_workspaces);
        return;
    }
    double* tmp_gather = tiledMatrixA->workspaces[workspace_rank]->aligned_tile();

    const int task_start = offsets[static_cast<std::size_t>(rank)];
    const int task_end   = offsets[static_cast<std::size_t>(rank) + 1];

    // ──────────── Sparse iteration + update-counter sync (multi-RHS backward) ────────────
    // Mirror of forward multi-RHS: ss[m] 1D counter, expected[m]+1 = sentinel.
    // Per task we loop over all num_col_tiles RHS column-blocks; the BLAS-3
    // dtrsm / dgemm handles tempnn columns at a time.
    ss_init(num_row_tiles, 1, 0);
    volatile int* __restrict__ ss = stile->ss_progress;
    const int* __restrict__ exp_arr = expected.data();

    for (int idx = task_start; idx < task_end; ++idx) {
        if (stile->ss_abort) break;
        const auto& task = tasks[static_cast<std::size_t>(idx)];
        const int type     = task[0];
        const int k        = task[1];
        const int m        = task[2];
        const int tile_idx = task[3];

        const int actual_k = num_row_tiles - 1 - k;
        const int actual_m = num_row_tiles - 1 - m;

        if (type == 1) {
            // ── Diagonal: solve U X = B (NoTrans), all RHS columns ──
            const int exp_m = exp_arr[m];
            {
                int _spin_ct = 0;
                while (!stile->ss_abort && ss[m] < exp_m) {
                    hpc_pause_hybrid(_spin_ct);
                }
                if (stile->ss_abort) break;
            }

            const int tempkm = (k == 0) ? (N - (num_row_tiles - 1) * tile_size) : tile_size;
            if (tile_idx >= 0) {
                double* ss_diag = tiledMatrixA->chunkedDenseTiles
                    ? tiledMatrixA->chunkedDenseTiles[tile_idx] : nullptr;
                if (ss_diag) {
                    const sTiles::SemisparseTileMetaCore& semi_diag =
                        tiledMatrixA->semisparseTileMetaCore[tile_idx];
                    const sTiles::TileMetaCore& meta_diag =
                        tiledMatrixA->tileMetaCore[tile_idx];
                    const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tempkm;
                    const int kd = semi_diag.upper_bw;

                    for (int n = 0; n < num_col_tiles; ++n) {
                        const int tempnn = (n == num_col_tiles - 1)
                                              ? (nrhs - n * tile_size) : tile_size;
                        double* B_block = get_block_B<RowMajorB>(B, N, nrhs, tile_size, actual_k, n);
                        if (kd >= 0) {
                            const int ldab = kd + 1;
                            banded_solve_via_panel_notrans<RowMajorB>(B_block, diag_dim, kd, tempnn,
                                                                       ss_diag, ldab, ldb_B<RowMajorB>(N, nrhs));
                        } else {
                            sTiles::core_dtrsm(sTiles::Side::Left,
                                               sTiles::Uplo::Upper,
                                               sTiles::Op::NoTrans,
                                               sTiles::Diag::NonUnit,
                                               diag_dim, tempnn, 1.0,
                                               ss_diag, diag_dim,
                                               B_block, ldb_B<RowMajorB>(N, nrhs));
                        }
                    }
                }
            }
            __atomic_store_n(&ss[m], exp_m + 1, __ATOMIC_RELEASE);

        } else if (type == 2) {
            // ── Off-diagonal: B[actual_m] -= A * B[actual_k], all RHS columns ──
            const int exp_k = exp_arr[k];
            {
                int _spin_ct = 0;
                while (!stile->ss_abort && ss[k] < exp_k + 1) {
                    hpc_pause_hybrid(_spin_ct);
                }
                if (stile->ss_abort) break;
            }

            if (tile_idx >= 0) {
                const int tempkm = (k == 0) ? (N - (num_row_tiles - 1) * tile_size) : tile_size;
                const int tempmm = (actual_m == num_row_tiles - 1)
                                     ? (N - actual_m * tile_size) : tile_size;
                double* ss_tile = tiledMatrixA->chunkedDenseTiles
                    ? tiledMatrixA->chunkedDenseTiles[tile_idx] : nullptr;
                if (ss_tile) {
                    const sTiles::SemisparseTileMetaCore& semi =
                        tiledMatrixA->semisparseTileMetaCore[tile_idx];
                    const int active_cols = semi.sa;
                    if (active_cols > 0) {
                        const int aind_size = static_cast<int>(semi.aind.size());
                        const int ld_gather = active_cols;

                        for (int n = 0; n < num_col_tiles; ++n) {
                            const int tempnn = (n == num_col_tiles - 1)
                                                  ? (nrhs - n * tile_size) : tile_size;
                            double* B_k = get_block_B<RowMajorB>(B, N, nrhs, tile_size, actual_k, n);
                            double* B_m = get_block_B<RowMajorB>(B, N, nrhs, tile_size, actual_m, n);

                            for (int j = 0; j < tempnn; ++j) {
                                for (int i = 0; i < active_cols && i < aind_size; ++i) {
                                    const int b_row = semi.aind[i];
                                    if (b_row >= 0 && b_row < tempkm) {
                                        if constexpr (RowMajorB) {
                                            tmp_gather[i + j * ld_gather] = B_k[static_cast<std::size_t>(b_row) * nrhs + j];
                                        } else {
                                            tmp_gather[i + j * ld_gather] = B_k[b_row + j * N];
                                        }
                                    } else {
                                        tmp_gather[i + j * ld_gather] = 0.0;
                                    }
                                }
                            }

                            if constexpr (RowMajorB) {
                                // Row-major B_m (tempmm x tempnn, row-stride nrhs) appears to
                                // col-major MKL as (tempnn x tempmm) with ld=nrhs.
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::Trans,
                                                  tempnn, tempmm, active_cols,
                                                  mzone, tmp_gather, ld_gather,
                                                  ss_tile, tempmm,
                                                  1.0, B_m, ldb_B<RowMajorB>(N, nrhs));
                            } else {
                                sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                                  tempmm, tempnn, active_cols,
                                                  mzone, ss_tile, tempmm,
                                                  tmp_gather, ld_gather,
                                                  1.0, B_m, N);
                            }
                        }
                    }
                }
            }
            __atomic_fetch_add(&ss[m], 1, __ATOMIC_RELEASE);
        }
    }

    ss_finalize();
}

// Runtime dispatcher: picks row/col instantiation based on scheme flag.
void stiles_pdtrsm_backward_semisparse_tasked(stiles_context_t *stile)
{
    TiledMatrix *tiledMatrixA; double *B; int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);
    if (tiledMatrixA && tiledMatrixA->prefer_row_layout)
        stiles_pdtrsm_backward_semisparse_tasked_impl<true>(stile);
    else
        stiles_pdtrsm_backward_semisparse_tasked_impl<false>(stile);
}

void stiles_pdtrsm_backward_semisparse_tasked_1rhs(stiles_context_t *stile)
{
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);
    (void)nrhs;  // single-RHS contract

    const auto& tasks    = sTiles::get_solve_bwd_tasks(tiledMatrixA);
    const auto& offsets  = sTiles::get_solve_bwd_offsets(tiledMatrixA);
    const auto& expected = sTiles::get_solve_bwd_expected(tiledMatrixA);

    const int rank = STILES_RANK;

    // Fall back to non-fast version if tasks not available.
    if (offsets.empty() || static_cast<std::size_t>(rank) + 1 >= offsets.size()) {
        stiles_pdtrsm_backward_semisparse(stile);
        return;
    }

    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N - 1) / tile_size + 1;

    // No gather buffer needed — the off-diagonal path now does inline
    // axpy-style column updates with B_k indirected reads (no temp).
    // Keep the workspace_rank check as a structural invariant matching
    // the parent _fast contract.
    const int workspace_rank = rank + stile->workspace_offset;
    if (workspace_rank >= tiledMatrixA->num_workspaces) {
        sTiles::Logger::error("[trsm_semi_tasks] workspace_rank ", workspace_rank,
                              " >= num_workspaces ", tiledMatrixA->num_workspaces);
        return;
    }

    const int task_start = offsets[static_cast<std::size_t>(rank)];
    const int task_end   = offsets[static_cast<std::size_t>(rank) + 1];

    // ──────────── Sparse iteration + update-counter sync (mirror of forward) ────────────
    // Same as forward: skip the dense (k, m) walk, iterate only the
    // row-affinity-rebinned task list, use ss[m] as an off-diagonal
    // update counter with sentinel expected[m]+1 = "row solved".
    //
    // The backward kernel works in TASK-INDEX space (k, m); the actual
    // matrix rows are N-1-k and N-1-m, mapped at use site.
    //
    // expected[m] semantics: number of off-diagonal (k, m) entries
    // contributing to row m's iteration counter. Same dependency
    // structure as forward — diagonal at (m, m) needs all its
    // off-diagonals first.
    if (expected.empty()) {
        // Fallback to legacy if expected[] not populated yet.
        stiles_pdtrsm_backward_semisparse(stile);
        return;
    }
    ss_init(num_row_tiles, 1, 0);
    volatile int* __restrict__ ss = stile->ss_progress;
    const int* __restrict__ exp_arr = expected.data();

    for (int idx = task_start; idx < task_end; ++idx) {
        if (stile->ss_abort) break;
        const auto& task = tasks[static_cast<std::size_t>(idx)];
        const int type     = task[0];  // 1=diagonal, 2=off-diagonal
        const int k        = task[1];
        const int m        = task[2];
        const int tile_idx = task[3];

        const int actual_k = num_row_tiles - 1 - k;
        const int actual_m = num_row_tiles - 1 - m;

        if (type == 1) {
            // ── Diagonal: solve U · x = b on this block (NoTrans) ──
            const int exp_m = exp_arr[m];
            {
                int _spin_ct = 0;
                while (!stile->ss_abort && ss[m] < exp_m) {
                    hpc_pause_hybrid(_spin_ct);
                }
                if (stile->ss_abort) break;
            }

            const int tempkm = (k == 0) ? (N - (num_row_tiles - 1) * tile_size)
                                        : tile_size;
            if (tile_idx >= 0) {
                double* ss_diag = tiledMatrixA->chunkedDenseTiles
                    ? tiledMatrixA->chunkedDenseTiles[tile_idx] : nullptr;
                if (ss_diag) {
                    const sTiles::SemisparseTileMetaCore& semi_diag =
                        tiledMatrixA->semisparseTileMetaCore[tile_idx];
                    const sTiles::TileMetaCore& meta_diag =
                        tiledMatrixA->tileMetaCore[tile_idx];
                    const int diag_dim = (meta_diag.height > 0) ? meta_diag.height
                                                                : tempkm;
                    const int kd = semi_diag.upper_bw;

                    double* B_block = B + actual_k * tile_size;

                    if (kd >= 0) {
                        const int ldab = kd + 1;
                        cblas_dtbsv(CblasColMajor,
                                    CblasUpper, CblasNoTrans, CblasNonUnit,
                                    diag_dim, kd, ss_diag, ldab,
                                    B_block, /*incx=*/1);
                    } else {
                        cblas_dtrsv(CblasColMajor,
                                    CblasUpper, CblasNoTrans, CblasNonUnit,
                                    diag_dim, ss_diag, diag_dim,
                                    B_block, /*incx=*/1);
                    }
                }
            }
            // Publish "row m solved" sentinel.
            __atomic_store_n(&ss[m], exp_m + 1, __ATOMIC_RELEASE);

        } else if (type == 2) {
            // ── Off-diagonal: B[actual_m] -= A · B[actual_k] ──
            const int exp_k = exp_arr[k];
            {
                int _spin_ct = 0;
                while (!stile->ss_abort && ss[k] < exp_k + 1) {
                    hpc_pause_hybrid(_spin_ct);
                }
                if (stile->ss_abort) break;
            }

            if (tile_idx >= 0) {
                const int tempkm = (k == 0) ? (N - (num_row_tiles - 1) * tile_size)
                                            : tile_size;
                const int tempmm = (actual_m == num_row_tiles - 1)
                                     ? (N - actual_m * tile_size) : tile_size;

                double* ss_tile = tiledMatrixA->chunkedDenseTiles
                    ? tiledMatrixA->chunkedDenseTiles[tile_idx] : nullptr;
                if (ss_tile) {
                    const sTiles::SemisparseTileMetaCore& semi =
                        tiledMatrixA->semisparseTileMetaCore[tile_idx];
                    const int active_cols = semi.sa;
                    if (active_cols > 0) {
                        const double* __restrict__ B_k = B + actual_k * tile_size;
                        double*       __restrict__ B_m = B + actual_m * tile_size;
                        const int*    __restrict__ aind = semi.aind.data();
                        const int aind_size = static_cast<int>(semi.aind.size());

                        const int active_pair = active_cols & ~1;
                        int j = 0;
                        for (; j < active_pair; j += 2) {
                            const int r0 = (j     < aind_size) ? aind[j]     : -1;
                            const int r1 = (j + 1 < aind_size) ? aind[j + 1] : -1;
                            const double s0 = (r0 >= 0 && r0 < tempkm) ? -B_k[r0] : 0.0;
                            const double s1 = (r1 >= 0 && r1 < tempkm) ? -B_k[r1] : 0.0;
                            const double* __restrict__ col0 = ss_tile + static_cast<std::size_t>(j)     * tempmm;
                            const double* __restrict__ col1 = ss_tile + static_cast<std::size_t>(j + 1) * tempmm;
                            #pragma omp simd
                            for (int i = 0; i < tempmm; ++i) {
                                B_m[i] += s0 * col0[i] + s1 * col1[i];
                            }
                        }
                        for (; j < active_cols && j < aind_size; ++j) {
                            const int r = aind[j];
                            if (r < 0 || r >= tempkm) continue;
                            const double s = -B_k[r];
                            const double* __restrict__ col = ss_tile + static_cast<std::size_t>(j) * tempmm;
                            #pragma omp simd
                            for (int i = 0; i < tempmm; ++i) B_m[i] += s * col[i];
                        }
                        (void)mzone;
                    }
                }
            }
            __atomic_fetch_add(&ss[m], 1, __ATOMIC_RELEASE);
        }
    }

    ss_finalize();
}

void stiles_pdtrsm_forward_dense_full(stiles_context_t *stile)
{
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    const double alpha = 1.0;
    const double zone  = 1.0;
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int Adesc_lm1 = (N/tile_size);
    #define BLKLDD_V2(k) ( (k) < Adesc_lm1 ? tile_size : N % tile_size)

    int k, m, n;
    int next_k, next_m, next_n;

    ss_init(num_row_tiles, num_col_tiles, -1);
    k = 0;
    m = STILES_RANK;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += STILES_SIZE;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        const int tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;
        const int tempmm = m == num_row_tiles-1 ? N-m*tile_size : tile_size;
        const int tempkm = k == num_row_tiles-1 ? N-k*tile_size : tile_size;
        const double lalpha = k == 0 ? alpha : zone;

        if (m == k) {
            ss_cond_wait(m, n, k-1);
            const int ldak = BLKLDD_V2(k);
            const int index1 = dense_full_tile_index(k, k, num_row_tiles);

            double* diag_tile = tiledMatrixA->denseTiles[index1];
            if (diag_tile) {
                sTiles::core_dtrsm(sTiles::Side::Left,
                                   sTiles::Uplo::Upper,
                                   sTiles::Op::Trans,
                                   sTiles::Diag::NonUnit,
                                   tempkm, tempnn, lalpha,
                                   diag_tile, ldak,
                                   get_block_col(B, N, tile_size, k, n), N);
            }
            ss_cond_set(k, n, k);
        } else {
            ss_cond_wait(k, n, k);
            ss_cond_wait(m, n, k-1);
            // All tiles present in this path, no need to check isActive
            const int ldak = BLKLDD_V2(k);
            const int index1 = dense_full_tile_index(k, m, num_row_tiles);

            double* offdiag_tile = tiledMatrixA->denseTiles[index1];
            if (offdiag_tile) {
                sTiles::core_dgemm(sTiles::Op::Trans,
                                   sTiles::Op::NoTrans,
                                   tempmm, tempnn, tempkm, mzone,
                                   offdiag_tile, ldak,
                                   get_block_col(B, N, tile_size, k, n), N,
                                   lalpha, get_block_col(B, N, tile_size, m, n), N);
            }
            ss_cond_set(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    ss_finalize();
    #undef BLKLDD_V2
}

void stiles_pdtrsm_backward_dense_full(stiles_context_t *stile)
{
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    const double alpha = 1.0;
    const double zone  = 1.0;
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int Adesc_lm1 = (N/tile_size);
    #define BLKLDD_V2(k) ( (k) < Adesc_lm1 ? tile_size : N % tile_size)

    int k, m, n;
    int next_k, next_m, next_n;

    ss_init(num_row_tiles, num_col_tiles, -1);
    k = 0;
    m = STILES_RANK;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += STILES_SIZE;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        const int tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;
        const double lalpha = k == 0 ? alpha : zone;

        if (m == k) {
            ss_cond_wait(m, n, k-1);
            const int tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
            const int ldak = BLKLDD_V2(num_row_tiles-1-k);
            const int actual_k = num_row_tiles - 1 - k;
            const int index1 = dense_full_tile_index(actual_k, actual_k, num_row_tiles);

            double* diag_tile = tiledMatrixA->denseTiles[index1];
            if (diag_tile) {
                sTiles::core_dtrsm(sTiles::Side::Left,
                                   sTiles::Uplo::Upper,
                                   sTiles::Op::NoTrans,
                                   sTiles::Diag::NonUnit,
                                   tempkm, tempnn, lalpha,
                                   diag_tile, ldak,
                                   get_block_col(B, N, tile_size, actual_k, n), N);
            }
            ss_cond_set(k, n, k);
        } else {
            ss_cond_wait(k, n, k);
            ss_cond_wait(m, n, k-1);
            // All tiles present in this path, no need to check isActive
            const int tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
            const int actual_k = num_row_tiles - 1 - k;
            const int actual_m = num_row_tiles - 1 - m;
            const int ldam = BLKLDD_V2(actual_m);
            const int index1 = dense_full_tile_index(actual_m, actual_k, num_row_tiles);

            double* offdiag_tile = tiledMatrixA->denseTiles[index1];
            if (offdiag_tile) {
                sTiles::core_dgemm(sTiles::Op::NoTrans,
                                   sTiles::Op::NoTrans,
                                   tile_size, tempnn, tempkm, mzone,
                                   offdiag_tile, ldam,
                                   get_block_col(B, N, tile_size, actual_k, n), N,
                                   lalpha, get_block_col(B, N, tile_size, actual_m, n), N);
            }
            ss_cond_set(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    ss_finalize();
    #undef BLKLDD_V2
}

void omp_pdtrsm_forward(TiledMatrix *tiledMatrixA, double *B, int nrhs, omp_dep_tracker_t* dep_tracker, int worldsize)
{
    const double alpha = 1.0;
    const double zone  = 1.0;
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int Adesc_lm1 = (N/tile_size);
    #define BLKLDD_OMP(k) ( (k) < Adesc_lm1 ? tile_size : N % tile_size)

    const int rank = omp_get_thread_num();

    int k, m, n;
    int next_k, next_m, next_n;

    dep_init(num_row_tiles, num_col_tiles, -1);

    k = 0;
    m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += worldsize;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        const int tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;
        const int tempmm = m == num_row_tiles-1 ? N-m*tile_size : tile_size;
        const int tempkm = k == num_row_tiles-1 ? N-k*tile_size : tile_size;
        const double lalpha = k == 0 ? alpha : zone;

        if (m == k) {
            dep_wait_for(m, n, k-1);
            const int ldak = BLKLDD_OMP(k);
            const int index1 = tiledMatrixA->mapper.map_ij(k, k, tiledMatrixA->dimTiledMatrix);

            double* diag_tile = (index1 >= 0 && tiledMatrixA->denseTiles) ? tiledMatrixA->denseTiles[index1] : nullptr;
            if (diag_tile) {
                sTiles::core_dtrsm(sTiles::Side::Left,
                                   sTiles::Uplo::Upper,
                                   sTiles::Op::Trans,
                                   sTiles::Diag::NonUnit,
                                   tempkm, tempnn, lalpha,
                                   diag_tile, ldak,
                                   get_block_col(B, N, tile_size, k, n), N);
            }
            dep_set_done(k, n, k);
        } else {
            dep_wait_for(k, n, k);
            dep_wait_for(m, n, k-1);
            if(tiledMatrixA->state.isActive(k, m, tiledMatrixA->dimTiledMatrix)){
                const int ldak = BLKLDD_OMP(k);
                const int index1 = tiledMatrixA->mapper.map_ij(k, m, tiledMatrixA->dimTiledMatrix);

                double* offdiag_tile = (index1 >= 0 && tiledMatrixA->denseTiles) ? tiledMatrixA->denseTiles[index1] : nullptr;
                if (offdiag_tile) {
                    sTiles::core_dgemm(sTiles::Op::Trans,
                                       sTiles::Op::NoTrans,
                                       tempmm, tempnn, tempkm, mzone,
                                       offdiag_tile, ldak,
                                       get_block_col(B, N, tile_size, k, n), N,
                                       lalpha, get_block_col(B, N, tile_size, m, n), N);
                }
            }
            dep_set_done(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    dep_finalize();
    #undef BLKLDD_OMP
}

void omp_pdtrsm_backward(TiledMatrix *tiledMatrixA, double *B, int nrhs, omp_dep_tracker_t* dep_tracker, int worldsize)
{
    const double alpha = 1.0;
    const double zone  = 1.0;
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int Adesc_lm1 = (N/tile_size);
    #define BLKLDD_OMP(k) ( (k) < Adesc_lm1 ? tile_size : N % tile_size)

    const int rank = omp_get_thread_num();

    int k, m, n;
    int next_k, next_m, next_n;

    dep_init(num_row_tiles, num_col_tiles, -1);

    k = 0;
    m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += worldsize;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        const int tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;
        const double lalpha = k == 0 ? alpha : zone;

        if (m == k) {
            dep_wait_for(m, n, k-1);
            const int tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
            const int ldak = BLKLDD_OMP(num_row_tiles-1-k);
            const int actual_k = num_row_tiles - 1 - k;
            const int index1 = tiledMatrixA->mapper.map_ij(actual_k, actual_k, tiledMatrixA->dimTiledMatrix);

            double* diag_tile = (index1 >= 0 && tiledMatrixA->denseTiles) ? tiledMatrixA->denseTiles[index1] : nullptr;
            if (diag_tile) {
                sTiles::core_dtrsm(sTiles::Side::Left,
                                   sTiles::Uplo::Upper,
                                   sTiles::Op::NoTrans,
                                   sTiles::Diag::NonUnit,
                                   tempkm, tempnn, lalpha,
                                   diag_tile, ldak,
                                   get_block_col(B, N, tile_size, actual_k, n), N);
            }
            dep_set_done(k, n, k);
        } else {
            dep_wait_for(k, n, k);
            dep_wait_for(m, n, k-1);
            const int actual_k = num_row_tiles - 1 - k;
            const int actual_m = num_row_tiles - 1 - m;
            if(tiledMatrixA->state.isActive(actual_m, actual_k, tiledMatrixA->dimTiledMatrix)){
                const int tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
                const int ldam = BLKLDD_OMP(actual_m);
                const int index1 = tiledMatrixA->mapper.map_ij(actual_m, actual_k, tiledMatrixA->dimTiledMatrix);

                double* offdiag_tile = (index1 >= 0 && tiledMatrixA->denseTiles) ? tiledMatrixA->denseTiles[index1] : nullptr;
                if (offdiag_tile) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::NoTrans,
                                       tile_size, tempnn, tempkm, mzone,
                                       offdiag_tile, ldam,
                                       get_block_col(B, N, tile_size, actual_k, n), N,
                                       lalpha, get_block_col(B, N, tile_size, actual_m, n), N);
                }
            }
            dep_set_done(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    dep_finalize();
    #undef BLKLDD_OMP
}

void omp_pdtrsm_forward_dense_full(TiledMatrix *tiledMatrixA, double *B, int nrhs, omp_dep_tracker_t* dep_tracker, int worldsize)
{
    const double alpha = 1.0;
    const double zone  = 1.0;
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int Adesc_lm1 = (N/tile_size);
    #define BLKLDD_V2_OMP(k) ( (k) < Adesc_lm1 ? tile_size : N % tile_size)

    const int rank = omp_get_thread_num();

    int k, m, n;
    int next_k, next_m, next_n;

    dep_init(num_row_tiles, num_col_tiles, -1);

    k = 0;
    m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += worldsize;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        const int tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;
        const int tempmm = m == num_row_tiles-1 ? N-m*tile_size : tile_size;
        const int tempkm = k == num_row_tiles-1 ? N-k*tile_size : tile_size;
        const double lalpha = k == 0 ? alpha : zone;

        if (m == k) {
            dep_wait_for(m, n, k-1);
            const int ldak = BLKLDD_V2_OMP(k);
            const int index1 = dense_full_tile_index(k, k, num_row_tiles);

            double* diag_tile = tiledMatrixA->denseTiles[index1];
            if (diag_tile) {
                sTiles::core_dtrsm(sTiles::Side::Left,
                                   sTiles::Uplo::Upper,
                                   sTiles::Op::Trans,
                                   sTiles::Diag::NonUnit,
                                   tempkm, tempnn, lalpha,
                                   diag_tile, ldak,
                                   get_block_col(B, N, tile_size, k, n), N);
            }
            dep_set_done(k, n, k);
        } else {
            dep_wait_for(k, n, k);
            dep_wait_for(m, n, k-1);
            // All tiles present in this path
            const int ldak = BLKLDD_V2_OMP(k);
            const int index1 = dense_full_tile_index(k, m, num_row_tiles);

            double* offdiag_tile = tiledMatrixA->denseTiles[index1];
            if (offdiag_tile) {
                sTiles::core_dgemm(sTiles::Op::Trans,
                                   sTiles::Op::NoTrans,
                                   tempmm, tempnn, tempkm, mzone,
                                   offdiag_tile, ldak,
                                   get_block_col(B, N, tile_size, k, n), N,
                                   lalpha, get_block_col(B, N, tile_size, m, n), N);
            }
            dep_set_done(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    dep_finalize();
    #undef BLKLDD_V2_OMP
}

void omp_pdtrsm_backward_dense_full(TiledMatrix *tiledMatrixA, double *B, int nrhs, omp_dep_tracker_t* dep_tracker, int worldsize)
{
    const double alpha = 1.0;
    const double zone  = 1.0;
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int Adesc_lm1 = (N/tile_size);
    #define BLKLDD_V2_OMP(k) ( (k) < Adesc_lm1 ? tile_size : N % tile_size)

    const int rank = omp_get_thread_num();

    int k, m, n;
    int next_k, next_m, next_n;

    dep_init(num_row_tiles, num_col_tiles, -1);

    k = 0;
    m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += worldsize;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        const int tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;
        const double lalpha = k == 0 ? alpha : zone;

        if (m == k) {
            dep_wait_for(m, n, k-1);
            const int tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
            const int ldak = BLKLDD_V2_OMP(num_row_tiles-1-k);
            const int actual_k = num_row_tiles - 1 - k;
            const int index1 = dense_full_tile_index(actual_k, actual_k, num_row_tiles);

            double* diag_tile = tiledMatrixA->denseTiles[index1];
            if (diag_tile) {
                sTiles::core_dtrsm(sTiles::Side::Left,
                                   sTiles::Uplo::Upper,
                                   sTiles::Op::NoTrans,
                                   sTiles::Diag::NonUnit,
                                   tempkm, tempnn, lalpha,
                                   diag_tile, ldak,
                                   get_block_col(B, N, tile_size, actual_k, n), N);
            }
            dep_set_done(k, n, k);
        } else {
            dep_wait_for(k, n, k);
            dep_wait_for(m, n, k-1);
            // All tiles present in this path
            const int tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
            const int actual_k = num_row_tiles - 1 - k;
            const int actual_m = num_row_tiles - 1 - m;
            const int ldam = BLKLDD_V2_OMP(actual_m);
            const int index1 = dense_full_tile_index(actual_m, actual_k, num_row_tiles);

            double* offdiag_tile = tiledMatrixA->denseTiles[index1];
            if (offdiag_tile) {
                sTiles::core_dgemm(sTiles::Op::NoTrans,
                                   sTiles::Op::NoTrans,
                                   tile_size, tempnn, tempkm, mzone,
                                   offdiag_tile, ldam,
                                   get_block_col(B, N, tile_size, actual_k, n), N,
                                   lalpha, get_block_col(B, N, tile_size, actual_m, n), N);
            }
            dep_set_done(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    dep_finalize();
    #undef BLKLDD_V2_OMP
}

void omp_pdtrsm_forward_semisparse(TiledMatrix *tiledMatrixA, double *B, int nrhs, omp_dep_tracker_t* dep_tracker, int worldsize)
{
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    // Get workspace for temporary GEMM results
    const int rank = omp_get_thread_num();
    const int workspace_rank = rank + dep_tracker->workspace_offset;
    if (workspace_rank >= tiledMatrixA->num_workspaces) {
        sTiles::Logger::error("[trsm_omp] workspace_rank ", workspace_rank, " >= num_workspaces ", tiledMatrixA->num_workspaces);
        return;
    }
    double* tmp_tile = tiledMatrixA->workspaces[workspace_rank]->aligned_tile();

    int k, m, n;
    int next_k, next_m, next_n;
    int tempkm, tempmm, tempnn;
    int index1;

    dep_init(num_row_tiles, num_col_tiles, -1);
    k = 0;
    m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += worldsize;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;
        tempmm = m == num_row_tiles-1 ? N-m*tile_size : tile_size;
        tempkm = k == num_row_tiles-1 ? N-k*tile_size : tile_size;

        if (m == k) {
            // Diagonal tile: solve L^T * X = B using triangular solver
            dep_wait_for(m, n, k-1);
            index1 = tiledMatrixA->mapper.map_ij(k, k, tiledMatrixA->dimTiledMatrix);

            double* ss_diag = tiledMatrixA->chunkedDenseTiles ? tiledMatrixA->chunkedDenseTiles[index1] : nullptr;
            if (ss_diag) {
                const sTiles::SemisparseTileMetaCore& semi_diag = tiledMatrixA->semisparseTileMetaCore[index1];
                const sTiles::TileMetaCore& meta_diag = tiledMatrixA->tileMetaCore[index1];
                const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tempkm;
                const int kd = semi_diag.upper_bw;

                double* B_block = get_block_col(B, N, tile_size, k, n);

                if (kd >= 0) {
                    // Banded format: use dtbtrs
                    const int ldab = kd + 1;
                    LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
                                   diag_dim, kd, tempnn, ss_diag, ldab, B_block, N);
                } else {
                    // Dense format: use dtrsm
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::Trans,
                                       sTiles::Diag::NonUnit,
                                       diag_dim, tempnn, 1.0,
                                       ss_diag, diag_dim,
                                       B_block, N);
                }
            }
            dep_set_done(k, n, k);
        } else {
            // Off-diagonal tile: B[m] -= A[k,m]^T * B[k]
            dep_wait_for(k, n, k);
            dep_wait_for(m, n, k-1);

            if(tiledMatrixA->state.isActive(k, m, tiledMatrixA->dimTiledMatrix)){
                index1 = tiledMatrixA->mapper.map_ij(k, m, tiledMatrixA->dimTiledMatrix);

                double* ss_tile = tiledMatrixA->chunkedDenseTiles ? tiledMatrixA->chunkedDenseTiles[index1] : nullptr;
                if (ss_tile) {
                    const sTiles::SemisparseTileMetaCore& semi = tiledMatrixA->semisparseTileMetaCore[index1];
                    const int active_cols = semi.sa;

                    if (active_cols > 0) {
                        double* B_k = get_block_col(B, N, tile_size, k, n);
                        double* B_m = get_block_col(B, N, tile_size, m, n);

                        // Compute GEMM: tmp = A[k,m]^T * B_k
                        const int ld_tmp = active_cols;
                        sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                          active_cols, tempnn, tempkm,
                                          mzone, ss_tile, tempkm,
                                          B_k, N,
                                          0.0, tmp_tile, ld_tmp);

                        // Scatter tmp to B_m using aind mapping
                        const int aind_size = static_cast<int>(semi.aind.size());
                        for (int j = 0; j < tempnn; ++j) {
                            for (int i = 0; i < active_cols && i < aind_size; ++i) {
                                const int out_row = semi.aind[i];
                                if (out_row >= 0 && out_row < tempmm) {
                                    B_m[out_row + j * N] += tmp_tile[i + j * ld_tmp];
                                }
                            }
                        }
                    }
                }
            }
            dep_set_done(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    dep_finalize();
}

void omp_pdtrsm_backward_semisparse(TiledMatrix *tiledMatrixA, double *B, int nrhs, omp_dep_tracker_t* dep_tracker, int worldsize)
{
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    // Get workspace for temporary gather buffer
    const int rank = omp_get_thread_num();
    const int workspace_rank = rank + dep_tracker->workspace_offset;
    if (workspace_rank >= tiledMatrixA->num_workspaces) {
        sTiles::Logger::error("[trsm_omp_semi] workspace_rank ", workspace_rank, " >= num_workspaces ", tiledMatrixA->num_workspaces);
        return;
    }
    double* tmp_gather = tiledMatrixA->workspaces[workspace_rank]->aligned_tile();

    int k, m, n;
    int next_k, next_m, next_n;
    int tempkm, tempnn, tempmm;
    int index1;

    dep_init(num_row_tiles, num_col_tiles, -1);
    k = 0;
    m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        next_k = k;
        next_m = m;
        next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += worldsize;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        tempnn = n == num_col_tiles-1 ? nrhs-n*tile_size : tile_size;

        const int actual_k = num_row_tiles - 1 - k;
        const int actual_m = num_row_tiles - 1 - m;

        if (m == k) {
            // Diagonal tile: solve L * X = B (no transpose) using triangular solver
            dep_wait_for(m, n, k-1);
            tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
            index1 = tiledMatrixA->mapper.map_ij(actual_k, actual_k, tiledMatrixA->dimTiledMatrix);

            double* ss_diag = tiledMatrixA->chunkedDenseTiles ? tiledMatrixA->chunkedDenseTiles[index1] : nullptr;
            if (ss_diag) {
                const sTiles::SemisparseTileMetaCore& semi_diag = tiledMatrixA->semisparseTileMetaCore[index1];
                const sTiles::TileMetaCore& meta_diag = tiledMatrixA->tileMetaCore[index1];
                const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tempkm;
                const int kd = semi_diag.upper_bw;

                double* B_block = get_block_col(B, N, tile_size, actual_k, n);

                if (kd >= 0) {
                    // Banded format: use dtbtrs
                    const int ldab = kd + 1;
                    LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N',
                                   diag_dim, kd, tempnn, ss_diag, ldab, B_block, N);
                } else {
                    // Dense format: use dtrsm
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans,
                                       sTiles::Diag::NonUnit,
                                       diag_dim, tempnn, 1.0,
                                       ss_diag, diag_dim,
                                       B_block, N);
                }
            }
            dep_set_done(k, n, k);
        } else {
            // Off-diagonal tile: B[actual_m] -= A[actual_m, actual_k] * B[actual_k]
            dep_wait_for(k, n, k);
            dep_wait_for(m, n, k-1);

            if(tiledMatrixA->state.isActive(actual_m, actual_k, tiledMatrixA->dimTiledMatrix)){
                tempkm = k == 0 ? N-(num_row_tiles-1)*tile_size : tile_size;
                tempmm = actual_m == num_row_tiles-1 ? N-actual_m*tile_size : tile_size;
                index1 = tiledMatrixA->mapper.map_ij(actual_m, actual_k, tiledMatrixA->dimTiledMatrix);

                double* ss_tile = tiledMatrixA->chunkedDenseTiles ? tiledMatrixA->chunkedDenseTiles[index1] : nullptr;
                if (ss_tile) {
                    const sTiles::SemisparseTileMetaCore& semi = tiledMatrixA->semisparseTileMetaCore[index1];
                    const int active_cols = semi.sa;

                    if (active_cols > 0) {
                        double* B_k = get_block_col(B, N, tile_size, actual_k, n);
                        double* B_m = get_block_col(B, N, tile_size, actual_m, n);

                        // Gather B_k values at aind positions into tmp_gather
                        const int aind_size = static_cast<int>(semi.aind.size());
                        for (int j = 0; j < tempnn; ++j) {
                            for (int i = 0; i < active_cols && i < aind_size; ++i) {
                                const int src_row = semi.aind[i];
                                if (src_row >= 0 && src_row < tempkm) {
                                    tmp_gather[i + j * active_cols] = B_k[src_row + j * N];
                                } else {
                                    tmp_gather[i + j * active_cols] = 0.0;
                                }
                            }
                        }

                        // Compute GEMM: B_m -= A * tmp_gather
                        sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                          tempmm, tempnn, active_cols,
                                          mzone, ss_tile, tempmm,
                                          tmp_gather, active_cols,
                                          1.0, B_m, N);
                    }
                }
            }
            dep_set_done(m, n, k);
        }
        n = next_n;
        m = next_m;
        k = next_k;
    }
    dep_finalize();
}

void omp_pdtrsm_forward_fast(TiledMatrix *tiledMatrixA, double *B, int nrhs, omp_dep_tracker_t* dep_tracker, int worldsize)
{
    // Get pre-collected tasks for this core
    const auto& tasks = sTiles::get_solve_fwd_tasks(tiledMatrixA);
    const auto& offsets = sTiles::get_solve_fwd_offsets(tiledMatrixA);

    const int rank = omp_get_thread_num();

    // Safety check: fall back to non-fast version if tasks not available
    if (offsets.empty() || static_cast<std::size_t>(rank) + 1 >= offsets.size()) {
        omp_pdtrsm_forward(tiledMatrixA, B, nrhs, dep_tracker, worldsize);
        return;
    }

    const double alpha = 1.0;
    const double zone  = 1.0;
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int Adesc_lm1 = N / tile_size;
    #define BLKLDD_OMP_FAST(kk) ( (kk) < Adesc_lm1 ? tile_size : N % tile_size)

    // Get this rank's task range
    const int task_start = offsets[static_cast<std::size_t>(rank)];
    const int task_end = offsets[static_cast<std::size_t>(rank) + 1];

    // Pointer into task list - advances when we find matching (k,m) position
    int task_ptr = task_start;

    dep_init(num_row_tiles, num_col_tiles, -1);

    // Use SAME iteration structure as original code
    int k = 0;
    int m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    int n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        // Compute next position (same as original)
        int next_k = k;
        int next_m = m;
        int next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += worldsize;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        const int tempnn = (n == num_col_tiles - 1) ? (nrhs - n * tile_size) : tile_size;
        const int tempmm = (m == num_row_tiles - 1) ? (N - m * tile_size) : tile_size;
        const int tempkm = (k == num_row_tiles - 1) ? (N - k * tile_size) : tile_size;
        const double lalpha = (k == 0) ? alpha : zone;

        // Check if current position (k, m) matches next pre-collected task
        int tile_idx = -1;
        if (task_ptr < task_end) {
            const auto& task = tasks[static_cast<std::size_t>(task_ptr)];
            if (task[1] == k && task[2] == m) {
                tile_idx = task[3];
                if (next_n == 0 || next_k != k || next_m != m) {
                    task_ptr++;
                }
            }
        }

        if (m == k) {
            // Diagonal tile: TRSM
            dep_wait_for(m, n, k - 1);

            if (tile_idx >= 0) {
                const int lda = BLKLDD_OMP_FAST(k);
                double* diag_tile = tiledMatrixA->denseTiles[tile_idx];
                if (diag_tile) {
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::Trans,
                                       sTiles::Diag::NonUnit,
                                       tempkm, tempnn, lalpha,
                                       diag_tile, lda,
                                       get_block_col(B, N, tile_size, k, n), N);
                }
            }
            dep_set_done(k, n, k);
        } else {
            // Off-diagonal tile: GEMM
            dep_wait_for(k, n, k);
            dep_wait_for(m, n, k - 1);

            if (tile_idx >= 0) {
                const int lda = BLKLDD_OMP_FAST(k);
                double* offdiag_tile = tiledMatrixA->denseTiles[tile_idx];
                if (offdiag_tile) {
                    sTiles::core_dgemm(sTiles::Op::Trans,
                                       sTiles::Op::NoTrans,
                                       tempmm, tempnn, tempkm, mzone,
                                       offdiag_tile, lda,
                                       get_block_col(B, N, tile_size, k, n), N,
                                       lalpha, get_block_col(B, N, tile_size, m, n), N);
                }
            }
            // ALWAYS signal - crucial for inactive tiles to maintain sync chain
            dep_set_done(m, n, k);
        }

        n = next_n;
        m = next_m;
        k = next_k;
    }

    dep_finalize();
    #undef BLKLDD_OMP_FAST
}

void omp_pdtrsm_backward_fast(TiledMatrix *tiledMatrixA, double *B, int nrhs, omp_dep_tracker_t* dep_tracker, int worldsize)
{
    // Get pre-collected tasks for this core
    const auto& tasks = sTiles::get_solve_bwd_tasks(tiledMatrixA);
    const auto& offsets = sTiles::get_solve_bwd_offsets(tiledMatrixA);

    const int rank = omp_get_thread_num();

    // Safety check: fall back to non-fast version if tasks not available
    if (offsets.empty() || static_cast<std::size_t>(rank) + 1 >= offsets.size()) {
        omp_pdtrsm_backward(tiledMatrixA, B, nrhs, dep_tracker, worldsize);
        return;
    }

    const double alpha = 1.0;
    const double zone  = 1.0;
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int Adesc_lm1 = N / tile_size;
    #define BLKLDD_OMP_FAST(kk) ( (kk) < Adesc_lm1 ? tile_size : N % tile_size)

    // Get this rank's task range
    const int task_start = offsets[static_cast<std::size_t>(rank)];
    const int task_end = offsets[static_cast<std::size_t>(rank) + 1];

    // Pointer into task list
    int task_ptr = task_start;

    dep_init(num_row_tiles, num_col_tiles, -1);

    // Same iteration structure as original
    int k = 0;
    int m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    int n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        // Compute next position
        int next_k = k;
        int next_m = m;
        int next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += worldsize;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        const int tempnn = (n == num_col_tiles - 1) ? (nrhs - n * tile_size) : tile_size;
        const double lalpha = (k == 0) ? alpha : zone;

        // For backward, actual tiles are at (num_tiles-1-k, num_tiles-1-m)
        const int actual_k = num_row_tiles - 1 - k;
        const int actual_m = num_row_tiles - 1 - m;

        // Check if current position matches next pre-collected task
        int tile_idx = -1;
        if (task_ptr < task_end) {
            const auto& task = tasks[static_cast<std::size_t>(task_ptr)];
            if (task[1] == k && task[2] == m) {
                tile_idx = task[3];
                if (next_n == 0 || next_k != k || next_m != m) {
                    task_ptr++;
                }
            }
        }

        if (m == k) {
            // Diagonal tile: TRSM (no transpose for backward)
            dep_wait_for(m, n, k - 1);

            const int tempkm = (k == 0) ? (N - (num_row_tiles - 1) * tile_size) : tile_size;

            if (tile_idx >= 0) {
                const int ldak = BLKLDD_OMP_FAST(actual_k);
                double* diag_tile = tiledMatrixA->denseTiles[tile_idx];
                if (diag_tile) {
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans,
                                       sTiles::Diag::NonUnit,
                                       tempkm, tempnn, lalpha,
                                       diag_tile, ldak,
                                       get_block_col(B, N, tile_size, actual_k, n), N);
                }
            }
            dep_set_done(k, n, k);
        } else {
            // Off-diagonal tile: GEMM
            dep_wait_for(k, n, k);
            dep_wait_for(m, n, k - 1);

            const int tempkm = (k == 0) ? (N - (num_row_tiles - 1) * tile_size) : tile_size;

            if (tile_idx >= 0) {
                const int ldam = BLKLDD_OMP_FAST(actual_m);
                double* offdiag_tile = tiledMatrixA->denseTiles[tile_idx];
                if (offdiag_tile) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::NoTrans,
                                       tile_size, tempnn, tempkm, mzone,
                                       offdiag_tile, ldam,
                                       get_block_col(B, N, tile_size, actual_k, n), N,
                                       lalpha, get_block_col(B, N, tile_size, actual_m, n), N);
                }
            }
            // ALWAYS signal
            dep_set_done(m, n, k);
        }

        n = next_n;
        m = next_m;
        k = next_k;
    }

    dep_finalize();
    #undef BLKLDD_OMP_FAST
}

template<bool RowMajorB>
static void omp_pdtrsm_forward_semisparse_fast_impl(TiledMatrix *tiledMatrixA, double *B, int nrhs, omp_dep_tracker_t* dep_tracker, int worldsize)
{
    const auto& tasks    = sTiles::get_solve_fwd_tasks(tiledMatrixA);
    const auto& offsets  = sTiles::get_solve_fwd_offsets(tiledMatrixA);
    const auto& expected = sTiles::get_solve_fwd_expected(tiledMatrixA);

    const int rank = omp_get_thread_num();

    if (offsets.empty() ||
        static_cast<std::size_t>(rank) + 1 >= offsets.size() ||
        expected.empty())
    {
        omp_pdtrsm_forward_semisparse(tiledMatrixA, B, nrhs, dep_tracker, worldsize);
        return;
    }

    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int workspace_rank = rank + dep_tracker->workspace_offset;
    if (workspace_rank >= tiledMatrixA->num_workspaces) {
        sTiles::Logger::error("[trsm_omp_tasks] workspace_rank ", workspace_rank,
                              " >= num_workspaces ", tiledMatrixA->num_workspaces);
        return;
    }
    double* tmp_tile = tiledMatrixA->workspaces[workspace_rank]->aligned_tile();

    const int task_start = offsets[static_cast<std::size_t>(rank)];
    const int task_end   = offsets[static_cast<std::size_t>(rank) + 1];

    // OMP sparse iteration + update-counter sync (mirror of pthreads).
    // Per-row atomic counter (num_row_tiles entries, init 0). sentinel
    // value expected[m]+1 = "row m solved".
    dep_init(num_row_tiles, 1, 0);
    std::atomic<int>* __restrict__ ss = dep_tracker->progress_table;
    const int* __restrict__ exp_arr = expected.data();

    for (int idx = task_start; idx < task_end; ++idx) {
        if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break;
        const auto& task = tasks[static_cast<std::size_t>(idx)];
        const int type     = task[0];
        const int k        = task[1];
        const int m        = task[2];
        const int tile_idx = task[3];

        const int tempmm = (m == num_row_tiles - 1) ? (N - m * tile_size) : tile_size;
        const int tempkm = (k == num_row_tiles - 1) ? (N - k * tile_size) : tile_size;

        if (type == 1) {
            const int exp_m = exp_arr[m];
            {
                int _spin_ct = 0;
                while (ss[m].load(std::memory_order_relaxed) < exp_m) {
                    if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break;
                    hpc_pause_hybrid(_spin_ct);
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break;
            }

            if (tile_idx >= 0) {
                double* ss_diag = tiledMatrixA->chunkedDenseTiles
                    ? tiledMatrixA->chunkedDenseTiles[tile_idx] : nullptr;
                if (ss_diag) {
                    const sTiles::SemisparseTileMetaCore& semi_diag =
                        tiledMatrixA->semisparseTileMetaCore[tile_idx];
                    const sTiles::TileMetaCore& meta_diag =
                        tiledMatrixA->tileMetaCore[tile_idx];
                    const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tempkm;
                    const int kd = semi_diag.upper_bw;

                    for (int n = 0; n < num_col_tiles; ++n) {
                        const int tempnn = (n == num_col_tiles - 1)
                                              ? (nrhs - n * tile_size) : tile_size;
                        double* B_block = get_block_B<RowMajorB>(B, N, nrhs, tile_size, k, n);
                        if (kd >= 0) {
                            const int ldab = kd + 1;
                            banded_solve_via_panel<RowMajorB>(B_block, diag_dim, kd, tempnn,
                                                              ss_diag, ldab, ldb_B<RowMajorB>(N, nrhs));
                        } else {
                            sTiles::core_dtrsm(sTiles::Side::Left,
                                               sTiles::Uplo::Upper,
                                               sTiles::Op::Trans,
                                               sTiles::Diag::NonUnit,
                                               diag_dim, tempnn, 1.0,
                                               ss_diag, diag_dim,
                                               B_block, ldb_B<RowMajorB>(N, nrhs));
                        }
                    }
                }
            }
            ss[m].store(exp_m + 1, std::memory_order_release);

        } else if (type == 2) {
            const int exp_k = exp_arr[k];
            {
                int _spin_ct = 0;
                while (ss[k].load(std::memory_order_relaxed) < exp_k + 1) {
                    if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break;
                    hpc_pause_hybrid(_spin_ct);
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break;
            }

            if (tile_idx >= 0) {
                double* ss_tile = tiledMatrixA->chunkedDenseTiles
                    ? tiledMatrixA->chunkedDenseTiles[tile_idx] : nullptr;
                if (ss_tile) {
                    const sTiles::SemisparseTileMetaCore& semi =
                        tiledMatrixA->semisparseTileMetaCore[tile_idx];
                    const int active_cols = semi.sa;
                    if (active_cols > 0) {
                        const int ld_tmp = active_cols;
                        const int aind_size = static_cast<int>(semi.aind.size());

                        for (int n = 0; n < num_col_tiles; ++n) {
                            const int tempnn = (n == num_col_tiles - 1)
                                                  ? (nrhs - n * tile_size) : tile_size;
                            double* B_k = get_block_B<RowMajorB>(B, N, nrhs, tile_size, k, n);
                            double* B_m = get_block_B<RowMajorB>(B, N, nrhs, tile_size, m, n);

                            if constexpr (RowMajorB) {
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::Trans,
                                                  active_cols, tempnn, tempkm,
                                                  mzone, ss_tile, tempkm,
                                                  B_k, ldb_B<RowMajorB>(N, nrhs),
                                                  0.0, tmp_tile, ld_tmp);
                            } else {
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans,
                                                  active_cols, tempnn, tempkm,
                                                  mzone, ss_tile, tempkm,
                                                  B_k, N,
                                                  0.0, tmp_tile, ld_tmp);
                            }

                            for (int j = 0; j < tempnn; ++j) {
                                for (int i = 0; i < active_cols && i < aind_size; ++i) {
                                    const int out_row = semi.aind[i];
                                    if (out_row >= 0 && out_row < tempmm) {
                                        if constexpr (RowMajorB) {
                                            B_m[static_cast<std::size_t>(out_row) * nrhs + j] += tmp_tile[i + j * ld_tmp];
                                        } else {
                                            B_m[out_row + static_cast<std::size_t>(j) * N] += tmp_tile[i + j * ld_tmp];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            ss[m].fetch_add(1, std::memory_order_release);
        }
    }

    dep_finalize();
}

// Runtime dispatcher: picks row/col instantiation based on scheme flag.
void omp_pdtrsm_forward_semisparse_fast(TiledMatrix *tiledMatrixA, double *B, int nrhs, omp_dep_tracker_t* dep_tracker, int worldsize)
{
    if (tiledMatrixA && tiledMatrixA->prefer_row_layout)
        omp_pdtrsm_forward_semisparse_fast_impl<true>(tiledMatrixA, B, nrhs, dep_tracker, worldsize);
    else
        omp_pdtrsm_forward_semisparse_fast_impl<false>(tiledMatrixA, B, nrhs, dep_tracker, worldsize);
}

template<bool RowMajorB>
static void omp_pdtrsm_backward_semisparse_fast_impl(TiledMatrix *tiledMatrixA, double *B, int nrhs, omp_dep_tracker_t* dep_tracker, int worldsize)
{
    // Get pre-collected tasks for this core
    const auto& tasks = sTiles::get_solve_bwd_tasks(tiledMatrixA);
    const auto& offsets = sTiles::get_solve_bwd_offsets(tiledMatrixA);

    const int rank = omp_get_thread_num();

    // Safety check: fall back to non-fast version if tasks not available
    const auto& expected = sTiles::get_solve_bwd_expected(tiledMatrixA);
    if (offsets.empty() ||
        static_cast<std::size_t>(rank) + 1 >= offsets.size() ||
        expected.empty())
    {
        omp_pdtrsm_backward_semisparse(tiledMatrixA, B, nrhs, dep_tracker, worldsize);
        return;
    }

    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int workspace_rank = rank + dep_tracker->workspace_offset;
    if (workspace_rank >= tiledMatrixA->num_workspaces) {
        sTiles::Logger::error("[trsm_omp_semi_tasks] workspace_rank ", workspace_rank,
                              " >= num_workspaces ", tiledMatrixA->num_workspaces);
        return;
    }
    double* tmp_gather = tiledMatrixA->workspaces[workspace_rank]->aligned_tile();

    const int task_start = offsets[static_cast<std::size_t>(rank)];
    const int task_end   = offsets[static_cast<std::size_t>(rank) + 1];

    // OMP sparse iteration + update-counter sync (mirror of pthreads backward).
    dep_init(num_row_tiles, 1, 0);
    std::atomic<int>* __restrict__ ss = dep_tracker->progress_table;
    const int* __restrict__ exp_arr = expected.data();

    for (int idx = task_start; idx < task_end; ++idx) {
        if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break;
        const auto& task = tasks[static_cast<std::size_t>(idx)];
        const int type     = task[0];
        const int k        = task[1];
        const int m        = task[2];
        const int tile_idx = task[3];

        const int actual_k = num_row_tiles - 1 - k;
        const int actual_m = num_row_tiles - 1 - m;
        const int tempmm = (actual_m == num_row_tiles - 1)
                             ? (N - actual_m * tile_size) : tile_size;
        const int tempkm = (actual_k == num_row_tiles - 1)
                             ? (N - actual_k * tile_size) : tile_size;

        if (type == 1) {
            const int exp_m = exp_arr[m];
            {
                int _spin_ct = 0;
                while (ss[m].load(std::memory_order_relaxed) < exp_m) {
                    if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break;
                    hpc_pause_hybrid(_spin_ct);
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break;
            }

            if (tile_idx >= 0) {
                double* ss_diag = tiledMatrixA->chunkedDenseTiles
                    ? tiledMatrixA->chunkedDenseTiles[tile_idx] : nullptr;
                if (ss_diag) {
                    const sTiles::SemisparseTileMetaCore& semi_diag =
                        tiledMatrixA->semisparseTileMetaCore[tile_idx];
                    const sTiles::TileMetaCore& meta_diag =
                        tiledMatrixA->tileMetaCore[tile_idx];
                    const int diag_dim = (meta_diag.height > 0) ? meta_diag.height : tempkm;
                    const int kd = semi_diag.upper_bw;

                    for (int n = 0; n < num_col_tiles; ++n) {
                        const int tempnn = (n == num_col_tiles - 1)
                                              ? (nrhs - n * tile_size) : tile_size;
                        double* B_block = get_block_B<RowMajorB>(B, N, nrhs, tile_size, actual_k, n);
                        if (kd >= 0) {
                            const int ldab = kd + 1;
                            banded_solve_via_panel_notrans<RowMajorB>(B_block, diag_dim, kd, tempnn,
                                                                       ss_diag, ldab, ldb_B<RowMajorB>(N, nrhs));
                        } else {
                            sTiles::core_dtrsm(sTiles::Side::Left,
                                               sTiles::Uplo::Upper,
                                               sTiles::Op::NoTrans,
                                               sTiles::Diag::NonUnit,
                                               diag_dim, tempnn, 1.0,
                                               ss_diag, diag_dim,
                                               B_block, ldb_B<RowMajorB>(N, nrhs));
                        }
                    }
                }
            }
            ss[m].store(exp_m + 1, std::memory_order_release);

        } else if (type == 2) {
            const int exp_k = exp_arr[k];
            {
                int _spin_ct = 0;
                while (ss[k].load(std::memory_order_relaxed) < exp_k + 1) {
                    if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break;
                    hpc_pause_hybrid(_spin_ct);
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break;
            }

            if (tile_idx >= 0) {
                double* ss_tile = tiledMatrixA->chunkedDenseTiles
                    ? tiledMatrixA->chunkedDenseTiles[tile_idx] : nullptr;
                if (ss_tile) {
                    const sTiles::SemisparseTileMetaCore& semi =
                        tiledMatrixA->semisparseTileMetaCore[tile_idx];
                    const int active_cols = semi.sa;
                    if (active_cols > 0) {
                        const int aind_size = static_cast<int>(semi.aind.size());
                        const int ld_gather = active_cols;

                        for (int n = 0; n < num_col_tiles; ++n) {
                            const int tempnn = (n == num_col_tiles - 1)
                                                  ? (nrhs - n * tile_size) : tile_size;
                            double* B_k = get_block_B<RowMajorB>(B, N, nrhs, tile_size, actual_k, n);
                            double* B_m = get_block_B<RowMajorB>(B, N, nrhs, tile_size, actual_m, n);

                            for (int j = 0; j < tempnn; ++j) {
                                for (int i = 0; i < active_cols && i < aind_size; ++i) {
                                    const int src_row = semi.aind[i];
                                    if (src_row >= 0 && src_row < tempkm) {
                                        if constexpr (RowMajorB) {
                                            tmp_gather[i + j * ld_gather] = B_k[static_cast<std::size_t>(src_row) * nrhs + j];
                                        } else {
                                            tmp_gather[i + j * ld_gather] = B_k[src_row + j * N];
                                        }
                                    } else {
                                        tmp_gather[i + j * ld_gather] = 0.0;
                                    }
                                }
                            }

                            if constexpr (RowMajorB) {
                                sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::Trans,
                                                  tempnn, tempmm, active_cols,
                                                  mzone, tmp_gather, ld_gather,
                                                  ss_tile, tempmm,
                                                  1.0, B_m, ldb_B<RowMajorB>(N, nrhs));
                            } else {
                                sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                                  tempmm, tempnn, active_cols,
                                                  mzone, ss_tile, tempmm,
                                                  tmp_gather, ld_gather,
                                                  1.0, B_m, N);
                            }
                        }
                    }
                }
            }
            ss[m].fetch_add(1, std::memory_order_release);
        }
    }

    dep_finalize();
}

// Runtime dispatcher: picks row/col instantiation based on scheme flag.
void omp_pdtrsm_backward_semisparse_fast(TiledMatrix *tiledMatrixA, double *B, int nrhs, omp_dep_tracker_t* dep_tracker, int worldsize)
{
    if (tiledMatrixA && tiledMatrixA->prefer_row_layout)
        omp_pdtrsm_backward_semisparse_fast_impl<true>(tiledMatrixA, B, nrhs, dep_tracker, worldsize);
    else
        omp_pdtrsm_backward_semisparse_fast_impl<false>(tiledMatrixA, B, nrhs, dep_tracker, worldsize);
}

void stiles_pdtrsm_forward_dense_tasked(stiles_context_t *stile)
{
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    // Get pre-collected tasks for this core
    const auto& tasks = sTiles::get_solve_fwd_tasks(tiledMatrixA);
    const auto& offsets = sTiles::get_solve_fwd_offsets(tiledMatrixA);

    const int rank = STILES_RANK;

    // Safety check: ensure offsets array has enough entries for this rank
    if (offsets.empty() || static_cast<std::size_t>(rank) + 1 >= offsets.size()) {
        sTiles::Process::stiles_pdtrsm_forward_dense(stile);
        return;
    }

    const double alpha = 1.0;
    const double zone  = 1.0;
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int Adesc_lm1 = N / tile_size;
    #define BLKLDD_FAST(kk) ( (kk) < Adesc_lm1 ? tile_size : N % tile_size)

    // Get this rank's task range
    const int task_start = offsets[static_cast<std::size_t>(rank)];
    const int task_end = offsets[static_cast<std::size_t>(rank) + 1];

    // Pointer into task list - advances when we find matching (k,m) position
    int task_ptr = task_start;

    ss_init(num_row_tiles, num_col_tiles, -1);

    // Use SAME iteration structure as original code
    // This ensures correct synchronization - we signal for ALL positions
    int k = 0;
    int m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    int n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        // Compute next position (same as original)
        int next_k = k;
        int next_m = m;
        int next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += STILES_SIZE;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        const int tempnn = (n == num_col_tiles - 1) ? (nrhs - n * tile_size) : tile_size;
        const int tempmm = (m == num_row_tiles - 1) ? (N - m * tile_size) : tile_size;
        const int tempkm = (k == num_row_tiles - 1) ? (N - k * tile_size) : tile_size;
        const double lalpha = (k == 0) ? alpha : zone;

        // Check if current position (k, m) matches next pre-collected task
        // Tasks are collected in same k-major order, so we can use sequential matching
        int tile_idx = -1;
        if (task_ptr < task_end) {
            const auto& task = tasks[static_cast<std::size_t>(task_ptr)];
            if (task[1] == k && task[2] == m) {
                // Match! Get the pre-computed tile index
                tile_idx = task[3];
                // Only advance pointer when n wraps (we'll see this k,m again for next n)
                if (next_n == 0 || next_k != k || next_m != m) {
                    task_ptr++;
                }
            }
        }

        if (m == k) {
            // Diagonal tile: TRSM
            ss_cond_wait(m, n, k - 1);

            // Only do computation if tile is active (tile_idx >= 0)
            if (tile_idx >= 0) {
                const int lda = BLKLDD_FAST(k);
                double* diag_tile = tiledMatrixA->denseTiles[tile_idx];
                if (diag_tile) {
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::Trans,
                                       sTiles::Diag::NonUnit,
                                       tempkm, tempnn, lalpha,
                                       diag_tile, lda,
                                       get_block_col(B, N, tile_size, k, n), N);
                }
            }
            ss_cond_set(k, n, k);
        } else {
            // Off-diagonal tile: GEMM
            ss_cond_wait(k, n, k);
            ss_cond_wait(m, n, k - 1);

            // Only do computation if tile is active (tile_idx >= 0)
            if (tile_idx >= 0) {
                const int lda = BLKLDD_FAST(k);
                double* offdiag_tile = tiledMatrixA->denseTiles[tile_idx];
                if (offdiag_tile) {
                    sTiles::core_dgemm(sTiles::Op::Trans,
                                       sTiles::Op::NoTrans,
                                       tempmm, tempnn, tempkm, mzone,
                                       offdiag_tile, lda,
                                       get_block_col(B, N, tile_size, k, n), N,
                                       lalpha, get_block_col(B, N, tile_size, m, n), N);
                }
            }
            // ALWAYS signal - crucial for inactive tiles to maintain sync chain
            ss_cond_set(m, n, k);
        }

        n = next_n;
        m = next_m;
        k = next_k;
    }

    ss_finalize();
    #undef BLKLDD_FAST
}

void stiles_pdtrsm_forward_dense_tasked_1rhs(stiles_context_t *stile)
{
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);
    (void)nrhs;  // single-RHS contract — caller must pass nrhs == 1

    const auto& tasks   = sTiles::get_solve_fwd_tasks(tiledMatrixA);
    const auto& offsets = sTiles::get_solve_fwd_offsets(tiledMatrixA);

    const int rank = STILES_RANK;

    if (offsets.empty() || static_cast<std::size_t>(rank) + 1 >= offsets.size()) {
        sTiles::Process::stiles_pdtrsm_forward_dense(stile);
        return;
    }

    const double alpha = 1.0;
    const double zone  = 1.0;
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N - 1) / tile_size + 1;

    const int Adesc_lm1 = N / tile_size;
    #define BLKLDD_FAST_ONE(kk) ( (kk) < Adesc_lm1 ? tile_size : N % tile_size )

    const int task_start = offsets[static_cast<std::size_t>(rank)];
    const int task_end   = offsets[static_cast<std::size_t>(rank) + 1];

    int task_ptr = task_start;

    // Single column-tile only.
    ss_init(num_row_tiles, 1, -1);

    int k = 0;
    int m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }

    while (k < num_row_tiles && m < num_row_tiles) {
        // n-axis collapsed: every step advances m by STILES_SIZE.
        int next_k = k;
        int next_m = m + STILES_SIZE;
        while (next_m >= num_row_tiles && next_k < num_row_tiles) {
            next_k++;
            next_m = next_m - num_row_tiles + next_k;
        }

        const int tempmm = (m == num_row_tiles - 1) ? (N - m * tile_size) : tile_size;
        const int tempkm = (k == num_row_tiles - 1) ? (N - k * tile_size) : tile_size;
        const double lalpha = (k == 0) ? alpha : zone;

        // Sequential task-list match — no n-axis means each (k, m) advances the
        // pointer exactly once on a hit.
        int tile_idx = -1;
        if (task_ptr < task_end) {
            const auto& task = tasks[static_cast<std::size_t>(task_ptr)];
            if (task[1] == k && task[2] == m) {
                tile_idx = task[3];
                task_ptr++;
            }
        }

        if (m == k) {
            // Diagonal tile: TRSM (single column).
            ss_cond_wait(m, 0, k - 1);
            if (tile_idx >= 0) {
                const int lda = BLKLDD_FAST_ONE(k);
                double* diag_tile = tiledMatrixA->denseTiles[tile_idx];
                if (diag_tile) {
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::Trans,
                                       sTiles::Diag::NonUnit,
                                       tempkm, 1, lalpha,
                                       diag_tile, lda,
                                       B + k * tile_size, N);
                }
            }
            ss_cond_set(k, 0, k);
        } else {
            // Off-diagonal tile: GEMM (single column).
            ss_cond_wait(k, 0, k);
            ss_cond_wait(m, 0, k - 1);
            if (tile_idx >= 0) {
                const int lda = BLKLDD_FAST_ONE(k);
                double* offdiag_tile = tiledMatrixA->denseTiles[tile_idx];
                if (offdiag_tile) {
                    sTiles::core_dgemm(sTiles::Op::Trans,
                                       sTiles::Op::NoTrans,
                                       tempmm, 1, tempkm, mzone,
                                       offdiag_tile, lda,
                                       B + k * tile_size, N,
                                       lalpha, B + m * tile_size, N);
                }
            }
            // Always signal — keeps the sync chain intact for inactive tiles.
            ss_cond_set(m, 0, k);
        }

        m = next_m;
        k = next_k;
    }

    ss_finalize();
    #undef BLKLDD_FAST_ONE
}

void stiles_pdtrsm_backward_dense_tasked(stiles_context_t *stile)
{
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    // Get pre-collected tasks for this core
    const auto& tasks = sTiles::get_solve_bwd_tasks(tiledMatrixA);
    const auto& offsets = sTiles::get_solve_bwd_offsets(tiledMatrixA);

    const int rank = STILES_RANK;

    // Safety check: ensure offsets array has enough entries for this rank
    if (offsets.empty() || static_cast<std::size_t>(rank) + 1 >= offsets.size()) {
        sTiles::Process::stiles_pdtrsm_backward_dense(stile);
        return;
    }

    const double alpha = 1.0;
    const double zone  = 1.0;
    const double mzone = -1.0;

    const int tile_size = tiledMatrixA->tile_size;
    const int N = tiledMatrixA->dim;
    const int num_row_tiles = (N == 0) ? 0 : (N-1)/tile_size + 1;
    const int num_col_tiles = (nrhs == 0) ? 0 : (nrhs-1)/tile_size + 1;

    const int Adesc_lm1 = N / tile_size;
    #define BLKLDD_FAST(kk) ( (kk) < Adesc_lm1 ? tile_size : N % tile_size)

    // Get this rank's task range
    const int task_start = offsets[static_cast<std::size_t>(rank)];
    const int task_end = offsets[static_cast<std::size_t>(rank) + 1];

    // Pointer into task list - advances when we find matching (k,m) position
    int task_ptr = task_start;

    ss_init(num_row_tiles, num_col_tiles, -1);

    // Use SAME iteration structure as original code
    // This ensures correct synchronization - we signal for ALL positions
    int k = 0;
    int m = rank;
    while (m >= num_row_tiles) {
        k++;
        m = m - num_row_tiles + k;
    }
    int n = 0;

    while (k < num_row_tiles && m < num_row_tiles) {
        // Compute next position (same as original)
        int next_k = k;
        int next_m = m;
        int next_n = n;
        next_n++;
        if (next_n >= num_col_tiles) {
            next_m += STILES_SIZE;
            while (next_m >= num_row_tiles && next_k < num_row_tiles) {
                next_k++;
                next_m = next_m - num_row_tiles + next_k;
            }
            next_n = 0;
        }

        const int tempnn = (n == num_col_tiles - 1) ? (nrhs - n * tile_size) : tile_size;
        const double lalpha = (k == 0) ? alpha : zone;

        // For backward, actual tiles are at (num_tiles-1-k, num_tiles-1-m)
        const int actual_k = num_row_tiles - 1 - k;
        const int actual_m = num_row_tiles - 1 - m;

        // Check if current position (k, m) matches next pre-collected task
        // Tasks are collected in same k-major order, so we can use sequential matching
        int tile_idx = -1;
        if (task_ptr < task_end) {
            const auto& task = tasks[static_cast<std::size_t>(task_ptr)];
            if (task[1] == k && task[2] == m) {
                // Match! Get the pre-computed tile index
                tile_idx = task[3];
                // Only advance pointer when n wraps (we'll see this k,m again for next n)
                if (next_n == 0 || next_k != k || next_m != m) {
                    task_ptr++;
                }
            }
        }

        if (m == k) {
            // Diagonal tile: TRSM (no transpose for backward)
            ss_cond_wait(m, n, k - 1);

            const int tempkm = (k == 0) ? (N - (num_row_tiles - 1) * tile_size) : tile_size;

            // Only do computation if tile is active (tile_idx >= 0)
            if (tile_idx >= 0) {
                const int ldak = BLKLDD_FAST(actual_k);
                double* diag_tile = tiledMatrixA->denseTiles[tile_idx];
                if (diag_tile) {
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans,
                                       sTiles::Diag::NonUnit,
                                       tempkm, tempnn, lalpha,
                                       diag_tile, ldak,
                                       get_block_col(B, N, tile_size, actual_k, n), N);
                }
            }
            ss_cond_set(k, n, k);
        } else {
            // Off-diagonal tile: GEMM
            ss_cond_wait(k, n, k);
            ss_cond_wait(m, n, k - 1);

            const int tempkm = (k == 0) ? (N - (num_row_tiles - 1) * tile_size) : tile_size;

            // Only do computation if tile is active (tile_idx >= 0)
            if (tile_idx >= 0) {
                const int ldam = BLKLDD_FAST(actual_m);
                double* offdiag_tile = tiledMatrixA->denseTiles[tile_idx];
                if (offdiag_tile) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::NoTrans,
                                       tile_size, tempnn, tempkm, mzone,
                                       offdiag_tile, ldam,
                                       get_block_col(B, N, tile_size, actual_k, n), N,
                                       lalpha, get_block_col(B, N, tile_size, actual_m, n), N);
                }
            }
            // ALWAYS signal - crucial for inactive tiles to maintain sync chain
            ss_cond_set(m, n, k);
        }

        n = next_n;
        m = next_m;
        k = next_k;
    }

    ss_finalize();
    #undef BLKLDD_FAST
}

}} // namespace sTiles::Process

// Forward declaration of global control parameter accessor
extern "C" int* sTiles_get_params();

// Top-level dispatchers exposed via stiles_pcompute.h. They pick the right
// leaf kernel based on factorization variant, tile_type_mode, nrhs, and
// whether pre-collected solve task lists are available. The global names
// have the _dispatch suffix to make the routing role explicit and to avoid
// shadowing the sTiles::Process::stiles_pdtrsm_forward/_backward leaf
// kernels (which are the catch-all fallbacks at the end of these
// dispatchers).
void stiles_pdtrsm_forward_dispatch(stiles_context_t *stile) {

    static int* stiles_control_params = sTiles_get_params();
    const int tile_type_mode = stiles_control_params[3];

    // Unpack arguments to determine variant
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    // Get factorization variant directly from scheme
    const int variant = tiledMatrixA->factorization_variant;

    // ========== Dispatch based on variant and tile type ==========
    //
    // Variants:
    //   0 = Sparse tiled (only active tiles, tile_type can be dense or semisparse)
    //   1 = Single dense tile covering full matrix (direct LAPACK)
    //   2 = All tiles dense (full triangular tiled structure)
    //   3 = Same as variant 0
    //
    // For solve: variant 1 uses direct single-tile TRSM
    //            variant 2 uses dedicated all-dense-tiles implementation
    //            variants 0, 3 use tiled implementation with mapper

    // Variant 1: Single dense tile - direct LAPACK TRSM
    if (variant == 1) {
        sTiles::Process::stiles_pdtrsm_forward_dense_single(stile, 1);
    }
    // Variant 2: All tiles dense - dedicated implementation with direct indexing
    else if (variant == 2) {
        sTiles::Process::stiles_pdtrsm_forward_dense_full(stile);
    }
    // Semisparse tile type (tile_type_mode == 1) with available structures
    else if (tile_type_mode == 1 && tiledMatrixA->chunkedDenseTiles && tiledMatrixA->semisparseTileMetaCore) {

        // Use fast version if pre-collected tasks are available
        const auto& fwd_tasks = sTiles::get_solve_fwd_tasks(tiledMatrixA);
        const auto& fwd_offsets = sTiles::get_solve_fwd_offsets(tiledMatrixA);
        if (!fwd_tasks.empty() && fwd_offsets.size() > 1) {
            if (nrhs == 1) {
                sTiles::Process::stiles_pdtrsm_forward_semisparse_tasked_1rhs(stile);
            } else {
                sTiles::Process::stiles_pdtrsm_forward_semisparse_tasked(stile);
            }
        } else {
            sTiles::Process::stiles_pdtrsm_forward_semisparse(stile);
        }


    }
    // Variants 0, 3: Use fast solve if pre-collected tasks available (dense tiles)
    else if (!sTiles::get_solve_fwd_tasks(tiledMatrixA).empty() && sTiles::get_solve_fwd_offsets(tiledMatrixA).size() > 1) {
        if (nrhs == 1) {
            sTiles::Process::stiles_pdtrsm_forward_dense_tasked_1rhs(stile);
        } else {
            sTiles::Process::stiles_pdtrsm_forward_dense_tasked(stile);
        }
    }
    else {
        sTiles::Process::stiles_pdtrsm_forward_dense(stile);
    }

}

void stiles_pdtrsm_backward_dispatch(stiles_context_t *stile) {

    static int* stiles_control_params = sTiles_get_params();
    const int tile_type_mode = stiles_control_params[3];

    // Unpack arguments to determine variant
    TiledMatrix *tiledMatrixA;
    double *B;
    int nrhs;
    sTiles::unpack_args(stile, tiledMatrixA, B, nrhs);

    // Get factorization variant directly from scheme
    const int variant = tiledMatrixA->factorization_variant;

    // ========== Dispatch based on variant and tile type ==========
    //
    // Variants:
    //   0 = Sparse tiled (only active tiles, tile_type can be dense or semisparse)
    //   1 = Single dense tile covering full matrix (direct LAPACK)
    //   2 = All tiles dense (full triangular tiled structure)
    //   3 = Same as variant 0
    //
    // For solve: variant 1 uses direct single-tile TRSM
    //            variant 2 uses dedicated all-dense-tiles implementation
    //            variants 0, 3 use tiled implementation with mapper

    // Variant 1: Single dense tile - direct LAPACK TRSM
    if (variant == 1) {
        sTiles::Process::stiles_pdtrsm_backward_dense_single(stile, 1);
    }
    // Variant 2: All tiles dense - dedicated implementation with direct indexing
    else if (variant == 2) {
        sTiles::Process::stiles_pdtrsm_backward_dense_full(stile);
    }
    // Semisparse tile type (tile_type_mode == 1) with available structures
    else if (tile_type_mode == 1 && tiledMatrixA->chunkedDenseTiles && tiledMatrixA->semisparseTileMetaCore) {
        // Use fast version if pre-collected tasks are available
        const auto& bwd_tasks = sTiles::get_solve_bwd_tasks(tiledMatrixA);
        const auto& bwd_offsets = sTiles::get_solve_bwd_offsets(tiledMatrixA);
        if (!bwd_tasks.empty() && bwd_offsets.size() > 1) {
            if (nrhs == 1) {
                sTiles::Process::stiles_pdtrsm_backward_semisparse_tasked_1rhs(stile);
            } else {
                sTiles::Process::stiles_pdtrsm_backward_semisparse_tasked(stile);
            }
        } else {
            sTiles::Process::stiles_pdtrsm_backward_semisparse(stile);
        }
    }
    // Variants 0, 3: Use fast solve if pre-collected tasks available (dense tiles)
    else if (!sTiles::get_solve_bwd_tasks(tiledMatrixA).empty() &&
             sTiles::get_solve_bwd_offsets(tiledMatrixA).size() > 1) {
        sTiles::Process::stiles_pdtrsm_backward_dense_tasked(stile);
    }
    else {
        sTiles::Process::stiles_pdtrsm_backward_dense(stile);
    }
}


