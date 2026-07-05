/**
 * @file    pdtrtri.cpp
 * @brief   Parallel (threaded) DTRTRI path; forward from legacy PLASMA to sTiles wrappers.
 *
 * @project sTiles (Sparse Tiles Library)
 * @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @version 3.0.0
 * @date 1 1 2026
 * @license Proprietary
 *
 * @note This file includes code adapted from PLASMA; licensing of third-party
 *       components remains subject to their original terms. sTiles wrappers and
 *       integration code are proprietary as stated below.
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
 */
#include "../../control/common.h"
#include <stdlib.h> 
#include <math.h>
#include <omp.h>
#include <iostream>
#include "../common/core_lapack.hpp"
#include "stiles_verify.hpp" 
#include "../tile/core_kernels.hpp"
#include "../tile/meta.hpp"

//#define STILES_INTERNAL_DEBUGGING
//#define STILES_EXPORT_DAG

// Old GPU code - disabled
// #ifdef STILES_GPU
//     #include "compute_gpu.hpp"
// #endif

#ifdef STILES_INTERNAL_DEBUGGING
    #include "debugging.hpp"
#endif

#ifdef STILES_EXPORT_DAG
    #include "dags.hpp"
#endif

#include <iomanip>

void mirroring(int n, double *A, double *B, int lda) {

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            B[j * lda + i] = A[i * lda + j];
        }
    }
}


namespace sTiles{ namespace SafeMode{
void pdtrtri(TiledMatrix *tiledMatrix, stiles_context_t *stile)
{

    const int rank = STILES_RANK;
    const int N = tiledMatrix->dim;
    const int tile_size = tiledMatrix->tile_size;
    const double zone  = (double) 1.0;
    const double mzone = (double)-1.0;
    const int num_tasks = tiledMatrix->e_trick_size_inv[STILES_RANK];
    const int num_tiles_per_dim = (N == 0) ? 0 : (N-1)/tile_size + 1;

    int info;
    int myroutine, i, j, k;
    int index1, index2, index3; 

    in_init(num_tiles_per_dim, num_tiles_per_dim, 0);
    int global_in = 0;

    for(int in=0; in<num_tasks;in++){

        myroutine = tiledMatrix->e_trick_inv[STILES_RANK][0+7*in];
        i = tiledMatrix->e_trick_inv[STILES_RANK][1+in*7];
        j = tiledMatrix->e_trick_inv[STILES_RANK][2+in*7];
        k = tiledMatrix->e_trick_inv[STILES_RANK][3+in*7];
        index1 = tiledMatrix->e_trick_inv[STILES_RANK][4+in*7];
        index2 = tiledMatrix->e_trick_inv[STILES_RANK][5+in*7];
        index3 = tiledMatrix->e_trick_inv[STILES_RANK][6+in*7];

        switch (myroutine) {

            case 1:
                
                sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit, tiledMatrix->dense_tiles[index1].width, tiledMatrix->dense_tiles[index1].width, 1.0,
                        tiledMatrix->dense_tiles[index1].elements, tiledMatrix->dense_tiles[index1].width,
                        tiledMatrix->inverse_tiles[index1].elements, tiledMatrix->inverse_tiles[index1].width); 


                break;

            case 2:

                sTiles::core_dtrmm(
                    sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                    tiledMatrix->dense_tiles[index2].height, tiledMatrix->dense_tiles[index2].width,  
                    zone, tiledMatrix->inverse_tiles[index1].elements, tiledMatrix->inverse_tiles[index1].height,
                    tiledMatrix->dense_tiles[index2].elements, tiledMatrix->dense_tiles[index2].height); 

                break;

            case 3:
                sTiles::Control::Barrier(stile);
                global_in = in + 1;
                goto exit_first_loop;

        }

    }

    exit_first_loop:  

    for(int in=global_in; in<num_tasks;in++){

        myroutine = tiledMatrix->e_trick_inv[STILES_RANK][0+7*in];
        i = tiledMatrix->e_trick_inv[STILES_RANK][1+in*7];
        j = tiledMatrix->e_trick_inv[STILES_RANK][2+in*7];
        k = tiledMatrix->e_trick_inv[STILES_RANK][3+in*7];
        index1 = tiledMatrix->e_trick_inv[STILES_RANK][4+in*7];
        index2 = tiledMatrix->e_trick_inv[STILES_RANK][5+in*7];
        index3 = tiledMatrix->e_trick_inv[STILES_RANK][6+in*7];

        switch (myroutine) {

            case 4:

                sTiles::core_dlauum(sTiles::Uplo::Upper, tiledMatrix->inverse_tiles[index1].width, tiledMatrix->inverse_tiles[index1].elements, tiledMatrix->inverse_tiles[index1].height); //L * L ^T
                mirroring( tiledMatrix->inverse_tiles[index1].height, tiledMatrix->inverse_tiles[index1].elements, tiledMatrix->inverse_tiles[index1].elements,  tiledMatrix->inverse_tiles[index1].height); //copy the upper of dense_tiles tile to the upper & lower inverse_tiles

                break;

            case 5:
                
                in_cond_wait(i, k, 2);
                sTiles::core_dgemm(
                    sTiles::Op::NoTrans, sTiles::Op::Trans,
                    tiledMatrix->dense_tiles[index2].height, tiledMatrix->dense_tiles[index2].height, tiledMatrix->dense_tiles[index2].width,
                    mzone, tiledMatrix->dense_tiles[index2].elements, tiledMatrix->dense_tiles[index2].height,
                            tiledMatrix->inverse_tiles[index2].elements, tiledMatrix->inverse_tiles[index2].height,
                        zone, tiledMatrix->inverse_tiles[index1].elements, tiledMatrix->inverse_tiles[index1].height);

                break;

            case 6:

                in_cond_set(i, i, 2);

                break;

            case 7:

                in_cond_wait(j, k, 2);
                sTiles::core_dgemm(
                    sTiles::Op::NoTrans, sTiles::Op::Trans,
                    tiledMatrix->dense_tiles[index1].height, tiledMatrix->inverse_tiles[index2].height, tiledMatrix->dense_tiles[index1].width,
                    -1, tiledMatrix->dense_tiles[index1].elements, tiledMatrix->dense_tiles[index1].height, // i is less than j
                            tiledMatrix->inverse_tiles[index2].elements, tiledMatrix->inverse_tiles[index2].height,
                        zone, tiledMatrix->inverse_tiles[index3].elements, tiledMatrix->inverse_tiles[index3].height);


                break;

            case 8:

                in_cond_wait(k, j, 2);
                sTiles::core_dgemm(
                    sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                    tiledMatrix->dense_tiles[index1].height, tiledMatrix->inverse_tiles[index2].width, tiledMatrix->dense_tiles[index1].width,
                    mzone, tiledMatrix->dense_tiles[index1].elements, tiledMatrix->dense_tiles[index1].height, // i is less than j
                            tiledMatrix->inverse_tiles[index2].elements, tiledMatrix->inverse_tiles[index2].height,
                        zone, tiledMatrix->inverse_tiles[index3].elements, tiledMatrix->inverse_tiles[index3].height);


                break;

            case 9:

                in_cond_set(i, j, 2);
                break;

        }


    }

    in_finalize();

}

}}

namespace sTiles{ namespace Process{
void pdtrtri(TiledMatrix *tiledMatrix, stiles_context_t *stile)
{

    const int rank = STILES_RANK;
    const int N = tiledMatrix->dim;
    const int tile_size = tiledMatrix->tile_size;
    const double zone  = (double) 1.0;
    const double mzone = (double)-1.0;
    const int num_tasks = tiledMatrix->e_trick_size_inv[STILES_RANK];
    const int num_tiles_per_dim = (N == 0) ? 0 : (N-1)/tile_size + 1;

    int info;
    int myroutine, i, j, k;
    int index1, index2, index3; 

    in_init(num_tiles_per_dim, num_tiles_per_dim, 0);
    int global_in = 0;

    auto tile_dims = [&](int dense_idx, int& h, int& w) {
        if (tiledMatrix->tileMetaCore && dense_idx >= 0) {
            const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[dense_idx];
            h = (meta.height > 0) ? meta.height : tile_size;
            w = (meta.width  > 0) ? meta.width  : tile_size;
        } else {
            h = tile_size;
            w = tile_size;
        }
    };

    for (int in = 0; in < num_tasks; ++in) {

        myroutine = tiledMatrix->e_trick_inv[STILES_RANK][0+7*in];
        i = tiledMatrix->e_trick_inv[STILES_RANK][1+in*7];
        j = tiledMatrix->e_trick_inv[STILES_RANK][2+in*7];
        k = tiledMatrix->e_trick_inv[STILES_RANK][3+in*7];
        index1 = tiledMatrix->e_trick_inv[STILES_RANK][4+in*7];
        index2 = tiledMatrix->e_trick_inv[STILES_RANK][5+in*7];
        index3 = tiledMatrix->e_trick_inv[STILES_RANK][6+in*7];

        switch (myroutine) {

            case 1:
            {
                int h = tile_size, w = tile_size;
                tile_dims(index1, h, w);
                double* fact = tiledMatrix->denseTiles[index1];
                double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                if (fact && inv) {
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans,
                                       sTiles::Diag::NonUnit,
                                       h, w, 1.0,
                                       fact, h,
                                       inv,  h);
                }
                break;
            }


            case 2:
            {
                int inv_h = tile_size, inv_w = tile_size;
                tile_dims(index1, inv_h, inv_w);
                int mh = tile_size, mw = tile_size;
                tile_dims(index2, mh, mw);
                double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                double* fact = tiledMatrix->denseTiles[index2];
                if (inv && fact) {
                    sTiles::core_dtrmm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans,
                                       sTiles::Diag::NonUnit,
                                       mh, mw,
                                       zone,
                                       inv,  inv_h,
                                       fact, mh);
                }

                break;
            }

            case 3:
                sTiles::Control::Barrier(stile);
                global_in = in + 1;
                goto exit_first_loop;

        }

    }

    exit_first_loop:  

    for (int in = global_in; in < num_tasks; ++in) {

        myroutine = tiledMatrix->e_trick_inv[STILES_RANK][0+7*in];
        i = tiledMatrix->e_trick_inv[STILES_RANK][1+in*7];
        j = tiledMatrix->e_trick_inv[STILES_RANK][2+in*7];
        k = tiledMatrix->e_trick_inv[STILES_RANK][3+in*7];
        index1 = tiledMatrix->e_trick_inv[STILES_RANK][4+in*7];
        index2 = tiledMatrix->e_trick_inv[STILES_RANK][5+in*7];
        index3 = tiledMatrix->e_trick_inv[STILES_RANK][6+in*7];

        switch (myroutine) {

            case 4:
            {
                int h = tile_size, w = tile_size;
                tile_dims(index1, h, w);
                double* inv = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                if (inv) {
                    sTiles::core_dlauum(sTiles::Uplo::Upper, w, inv, h);
                    mirroring(h, inv, inv, h);
                }

                break;
            }

            case 5:
            {
                in_cond_wait(i, k, 2);
                int mh = tile_size, mw = tile_size;
                tile_dims(index2, mh, mw);
                int inv1_h = tile_size, inv1_w = tile_size;
                tile_dims(index1, inv1_h, inv1_w);
                int inv2_h = tile_size, inv2_w = tile_size;
                tile_dims(index2, inv2_h, inv2_w);
                double* fact = tiledMatrix->denseTiles[index2];
                double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                double* inv1 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                if (fact && inv2 && inv1) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::Trans,
                                       mh, mh, mw,
                                       mzone,
                                       fact, mh,
                                       inv2, inv2_h,
                                       zone,
                                       inv1, inv1_h);
                }

                break;
            }

            case 6:

                in_cond_set(i, i, 2);

                break;

            case 7:
            {
                in_cond_wait(j, k, 2);
                int mh = tile_size, mw = tile_size;
                tile_dims(index1, mh, mw);
                int inv2_h = tile_size, inv2_w = tile_size;
                tile_dims(index2, inv2_h, inv2_w);
                int inv3_h = tile_size, inv3_w = tile_size;
                tile_dims(index3, inv3_h, inv3_w);
                double* fact = tiledMatrix->denseTiles[index1];
                double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                if (fact && inv2 && inv3) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::Trans,
                                       mh, inv2_h, mw,
                                       -1.0,
                                       fact, mh,
                                       inv2, inv2_h,
                                       zone,
                                       inv3, inv3_h);
                }


                break;
            }

            case 8:
            {
                in_cond_wait(k, j, 2);
                int mh = tile_size, mw = tile_size;
                tile_dims(index1, mh, mw);
                int inv2_h = tile_size, inv2_w = tile_size;
                tile_dims(index2, inv2_h, inv2_w);
                int inv3_h = tile_size, inv3_w = tile_size;
                tile_dims(index3, inv3_h, inv3_w);
                double* fact = tiledMatrix->denseTiles[index1];
                double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                if (fact && inv2 && inv3) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::NoTrans,
                                       mh, inv2_w, mw,
                                       mzone,
                                       fact, mh,
                                       inv2, inv2_h,
                                       zone,
                                       inv3, inv3_h);
                }


                break;
            }

            case 9:

                in_cond_set(i, j, 2);
                break;

        }


    }

    in_finalize();

}

void pdtrtri_from_inv_tasks(TiledMatrix *tiledMatrix, stiles_context_t *stile)
{
    const int N = tiledMatrix->dim;
    const int tile_size = tiledMatrix->tile_size;
    const double zone  = 1.0;
    const double mzone = -1.0;
    const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

    auto tile_dims = [&](int dense_idx, int& h, int& w) {
        if (tiledMatrix->tileMetaCore && dense_idx >= 0) {
            const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[dense_idx];
            h = (meta.height > 0) ? meta.height : tile_size;
            w = (meta.width  > 0) ? meta.width  : tile_size;
        } else {
            h = tile_size;
            w = tile_size;
        }
    };

    const auto &tasks   = tiledMatrix->inv_tasks;
    const auto &offsets = tiledMatrix->inv_task_offsets;
    size_t start = 0, end = tasks.size();
    if (!offsets.empty()) {
        const size_t r = static_cast<size_t>(STILES_RANK);
        if (r < offsets.size()) start = static_cast<size_t>(offsets[r]);
        if (r + 1 < offsets.size()) end = static_cast<size_t>(offsets[r + 1]);
    }

    in_init(num_tiles_per_dim, num_tiles_per_dim, 0);
    int global_in = -1;

    // Phase 1: until routine 3 (barrier)
    for (size_t in = start; in < end; ++in) {
        const auto &t = tasks[in];
        const int myroutine = t[0];
        const int index1 = t[4];
        const int index2 = t[5];

        switch (myroutine) {
            case 1: // TRSM: inv(index1) = fact(index1)^{-1}
            {
                int h = tile_size, w = tile_size;
                tile_dims(index1, h, w);
                double* fact = tiledMatrix->denseTiles[index1];
                double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                if (fact && inv) {
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans,
                                       sTiles::Diag::NonUnit,
                                       h, w, 1.0,
                                       fact, h,
                                       inv,  h);
                }
                break;
            }
            case 2: // TRMM: fact(index2) *= inv(index1)
            {
                int inv_h = tile_size, inv_w = tile_size;
                tile_dims(index1, inv_h, inv_w);
                int mh = tile_size, mw = tile_size;
                tile_dims(index2, mh, mw);
                double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                double* fact = tiledMatrix->denseTiles[index2];
                if (inv && fact) {
                    sTiles::core_dtrmm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans,
                                       sTiles::Diag::NonUnit,
                                       mh, mw,
                                       zone,
                                       inv,  inv_h,
                                       fact, mh);
                }
                break;
            }
            case 3: // Barrier point
                sTiles::Control::Barrier(stile);
                global_in = static_cast<int>(in + 1);
                in = end; // break slice loop
                break;
            default:
                break;
        }
    }

    if (global_in < 0) global_in = static_cast<int>(start);

    // Phase 2: remaining tasks
    for (size_t in = static_cast<size_t>(global_in); in < end; ++in) {
        const auto &t = tasks[in];
        const int myroutine = t[0];
        const int i = t[1];
        const int j = t[2];
        const int k = t[3];
        const int index1 = t[4];
        const int index2 = t[5];
        const int index3 = t[6];

        switch (myroutine) {
            case 4: // LAUUM + mirror on inv(index1)
            {
                int h = tile_size, w = tile_size;
                tile_dims(index1, h, w);
                double* inv = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                if (inv) {
                    sTiles::core_dlauum(sTiles::Uplo::Upper, w, inv, h);
                    mirroring(h, inv, inv, h);
                }
                break;
            }
            case 5: // GEMM accumulate into inv(index1)
            {
                in_cond_wait(i, k, 2);
                int mh = tile_size, mw = tile_size;
                tile_dims(index2, mh, mw);
                int inv1_h = tile_size, inv1_w = tile_size;
                tile_dims(index1, inv1_h, inv1_w);
                int inv2_h = tile_size, inv2_w = tile_size;
                tile_dims(index2, inv2_h, inv2_w);
                double* fact = tiledMatrix->denseTiles[index2];
                double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                double* inv1 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                if (fact && inv2 && inv1) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::Trans,
                                       mh, mh, mw,
                                       mzone,
                                       fact, mh,
                                       inv2, inv2_h,
                                       zone,
                                       inv1, inv1_h);
                }
                break;
            }
            case 6:
                in_cond_set(i, i, 2);
                break;
            case 7: // GEMM update inv(index3)
            {
                in_cond_wait(j, k, 2);
                int mh = tile_size, mw = tile_size;
                tile_dims(index1, mh, mw);
                int inv2_h = tile_size, inv2_w = tile_size;
                tile_dims(index2, inv2_h, inv2_w);
                int inv3_h = tile_size, inv3_w = tile_size;
                tile_dims(index3, inv3_h, inv3_w);
                double* fact = tiledMatrix->denseTiles[index1];
                double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                if (fact && inv2 && inv3) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::Trans,
                                       mh, inv2_h, mw,
                                       -1.0,
                                       fact, mh,
                                       inv2, inv2_h,
                                       zone,
                                       inv3, inv3_h);
                }
                break;
            }
            case 8: // GEMM update inv(index3)
            {
                in_cond_wait(k, j, 2);
                int mh = tile_size, mw = tile_size;
                tile_dims(index1, mh, mw);
                int inv2_h = tile_size, inv2_w = tile_size;
                tile_dims(index2, inv2_h, inv2_w);
                int inv3_h = tile_size, inv3_w = tile_size;
                tile_dims(index3, inv3_h, inv3_w);
                double* fact = tiledMatrix->denseTiles[index1];
                double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                if (fact && inv2 && inv3) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::NoTrans,
                                       mh, inv2_w, mw,
                                       mzone,
                                       fact, mh,
                                       inv2, inv2_h,
                                       zone,
                                       inv3, inv3_h);
                }
                break;
            }
            case 9:
                in_cond_set(i, j, 2);
                break;
            default:
                break;
        }
    }

    in_finalize();
}

}}

void stiles_pdtrtri(stiles_context_t *stile) {

    TiledMatrix *tiledMatrix;
    sTiles::unpack_args(stile, tiledMatrix);

    // #ifdef STILES_GPU
    //     //stiles_pdtrtri_gpu(tiledMatrix, stile);
    //     std::cout << "fix me" << std::endl;
    // #else
    //     #ifdef STILES_INTERNAL_DEBUGGING
    //         //stiles_pdtrtri_cpu_debug(tiledMatrix, stile);
    //         std::cout << "fix me" << std::endl;
    //     #elif defined(STILES_EXPORT_DAG)  
    //         //export_dag_tree_inverse(tiledMatrix, stile);
    //         std::cout << "fix me" << std::endl;
    //     #else
    //         stiles_pdtrtri_cpu(tiledMatrix, stile);
    //     #endif    
    // #endif

#ifdef STILES_SAFEMODE
    sTiles::SafeMode::pdtrtri(tiledMatrix, stile);
#elif defined(STILES_FASTMODE)
    sTiles::Process::pdtrtri_from_inv_tasks(tiledMatrix, stile);
#endif


// #ifdef ESMAIL_CHECK
//     sTiles::Process::pdtrtri(tiledMatrix, stile);
// #else
//     
// #endif

}


void stiles_pdtrtri_cpu(stiles_context_t *stile)
{
    int uplo;
    TilesDescriptor A;
    STILES_sequence *sequence;
    STILES_request *request;

    int info;
    int myroutine, i, j, k, task_reached;
    int index1, index2, index3; 

    double zone  = (double) 1.0;
    double mzone = (double)-1.0;

    sTiles::unpack_args(stile, uplo, A, sequence, request);
    if (sequence->status != 0) printf("Error! \n");
        //return 1;

    in_init(A.nt, A.nt, 0);

    int global_in = 0;
    for(int in=0; in<A.e_trick_size_inv[STILES_RANK];in++){

        myroutine = A.e_trick_inv[STILES_RANK][0+7*in];
        i = A.e_trick_inv[STILES_RANK][1+in*7];
        j = A.e_trick_inv[STILES_RANK][2+in*7];
        k = A.e_trick_inv[STILES_RANK][3+in*7];
        index1 = A.e_trick_inv[STILES_RANK][4+in*7];
        index2 = A.e_trick_inv[STILES_RANK][5+in*7];
        index3 = A.e_trick_inv[STILES_RANK][6+in*7];

        switch (myroutine) {

            case 1:
                
                sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit, A.dense_tiles[index1].width, A.dense_tiles[index1].width, 1.0,
                        A.dense_tiles[index1].elements, A.dense_tiles[index1].width,  // A: triangular matrix
                        A.inverse_tiles[index1].elements, A.inverse_tiles[index1].width); // B: identity matrix (right-hand side)


                break;

            case 2:

                sTiles::core_dtrmm(
                    sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                    A.dense_tiles[index2].height, A.dense_tiles[index2].width,  // M and N (rows and columns of B)
                    zone, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].height,  // A and its leading dimension (LDA)
                    A.dense_tiles[index2].elements, A.dense_tiles[index2].height); 

                break;

            case 3:
                sTiles::Control::Barrier(stile);
                global_in = in + 1;  // Save the next index for the second loop
                goto exit_first_loop;  // Break out of the first loop

        }

    }

    exit_first_loop:  

    for(int in=global_in; in<A.e_trick_size_inv[STILES_RANK];in++){

        myroutine = A.e_trick_inv[STILES_RANK][0+7*in];
        i = A.e_trick_inv[STILES_RANK][1+in*7];
        j = A.e_trick_inv[STILES_RANK][2+in*7];
        k = A.e_trick_inv[STILES_RANK][3+in*7];
        index1 = A.e_trick_inv[STILES_RANK][4+in*7];
        index2 = A.e_trick_inv[STILES_RANK][5+in*7];
        index3 = A.e_trick_inv[STILES_RANK][6+in*7];

        switch (myroutine) {

            case 4:

                sTiles::core_dlauum(sTiles::Uplo::Upper, A.inverse_tiles[index1].width, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].height); //L * L ^T
                mirroring( A.inverse_tiles[index1].height, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].elements,  A.inverse_tiles[index1].height); //copy the upper of dense_tiles tile to the upper & lower inverse_tiles

                break;

            case 5:
                
                in_cond_wait(i, k, 2);
                sTiles::core_dgemm(
                    sTiles::Op::NoTrans, sTiles::Op::Trans,
                    A.dense_tiles[index2].height, A.dense_tiles[index2].height, A.dense_tiles[index2].width,
                    mzone, A.dense_tiles[index2].elements, A.dense_tiles[index2].height,
                            A.inverse_tiles[index2].elements, A.inverse_tiles[index2].height,
                        zone, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].height);

                break;

            case 6:

                in_cond_set(i, i, 2);

                break;

            case 7:

                in_cond_wait(j, k, 2);
                sTiles::core_dgemm(
                    sTiles::Op::NoTrans, sTiles::Op::Trans,
                    A.dense_tiles[index1].height, A.inverse_tiles[index2].height, A.dense_tiles[index1].width,
                    -1, A.dense_tiles[index1].elements, A.dense_tiles[index1].height, // i is less than j
                            A.inverse_tiles[index2].elements, A.inverse_tiles[index2].height,
                        zone, A.inverse_tiles[index3].elements, A.inverse_tiles[index3].height);


                break;

            case 8:

                in_cond_wait(k, j, 2);
                sTiles::core_dgemm(
                    sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                    A.dense_tiles[index1].height, A.inverse_tiles[index2].width, A.dense_tiles[index1].width,
                    mzone, A.dense_tiles[index1].elements, A.dense_tiles[index1].height, // i is less than j
                            A.inverse_tiles[index2].elements, A.inverse_tiles[index2].height,
                        zone, A.inverse_tiles[index3].elements, A.inverse_tiles[index3].height);


                break;

            case 9:

                in_cond_set(i, j, 2);
                break;

        }


    }

    in_finalize();

}


void stiles_pdtrtri_cpu_no(stiles_context_t *stile)
{
    // --- Argument Unpacking & Initial Setup ---
    int uplo_enum;
    TilesDescriptor A;
    STILES_sequence *sequence;
    STILES_request *request;
    sTiles::unpack_args(stile, uplo_enum, A, sequence, request);

    if (sequence->status != 0) {
        printf("Error! Sequence status not success in stiles_pdtrtri_cpu.\n");
        return;
    }

    // --- Enum Conversions (for new API) ---
    sTiles::Uplo stiles_uplo = sTiles::Uplo::Upper;

    // --- Variable Declarations ---
    int myroutine, i, j, k;
    int index1, index2, index3;
    int global_in = 0;
    double zone = 1.0;
    double mzone = -1.0;
    
    in_init(A.nt, A.nt, 0);

    // --- Phase 1: Triangular Solve & Updates ---
    for (int in = 0; in < A.e_trick_size_inv[STILES_RANK]; in++) {
        myroutine = A.e_trick_inv[STILES_RANK][0 + 7 * in];
        i = A.e_trick_inv[STILES_RANK][1 + 7 * in];
        j = A.e_trick_inv[STILES_RANK][2 + 7 * in];
        k = A.e_trick_inv[STILES_RANK][3 + 7 * in];
        index1 = A.e_trick_inv[STILES_RANK][4 + 7 * in];
        index2 = A.e_trick_inv[STILES_RANK][5 + 7 * in];
        index3 = A.e_trick_inv[STILES_RANK][6 + 7 * in];

        switch (myroutine) {
            case 1: // DTRSM
            {
                auto& tile_A = A.dense_tiles[index1];
                auto& tile_invA = A.inverse_tiles[index1];
                
                auto new_op = [&]() {
                    sTiles::core_dtrsm(sTiles::Side::Left, stiles_uplo, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                       tile_A.width, tile_A.width, zone,
                                       tile_A.elements, tile_A.width,
                                       tile_invA.elements, tile_invA.width);
                };
                auto old_op = [&](double* tile, int lda) { // Corrected signature
                    sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                      tile_A.width, tile_A.width, zone,
                                      tile_A.elements, tile_A.width,
                                      tile, lda); // Use provided lda
                };
                
                #if defined(STILE_VERIFY_AGAINST_OLD)
                    STILE_VERIFY_EXEC("TRTRI_DTRSM", tile_A.width, tile_A.width, tile_invA.elements, tile_invA.width, new_op, old_op);
                #elif defined(STILE_USE_NEW_API)
                    new_op();
                #else
                    old_op(tile_invA.elements, tile_invA.width);
                #endif
                break;
            }
            case 2: // DTRMM
            {
                auto& tile_invA_diag = A.inverse_tiles[index1];
                auto& tile_B = A.dense_tiles[index2];
                
                auto new_op = [&]() {
                    sTiles::core_dtrmm(sTiles::Side::Left, stiles_uplo, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                       tile_B.height, tile_B.width, zone,
                                       tile_invA_diag.elements, tile_invA_diag.height,
                                       tile_B.elements, tile_B.height);
                };
                auto old_op = [&](double* tile, int lda) { // Corrected signature
                    sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                      tile_B.height, tile_B.width, zone,
                                      tile_invA_diag.elements, tile_invA_diag.height,
                                      tile, lda); // Use provided lda
                };

                #if defined(STILE_VERIFY_AGAINST_OLD)
                    STILE_VERIFY_EXEC("TRTRI_DTRMM", tile_B.height, tile_B.width, tile_B.elements, tile_B.height, new_op, old_op);
                #elif defined(STILE_USE_NEW_API)
                    new_op();
                #else
                    old_op(tile_B.elements, tile_B.height);
                #endif
                break;
            }
            case 3: // Synchronization point
                sTiles::Control::Barrier(stile);
                global_in = in + 1;
                goto exit_first_loop;
        }
    }

exit_first_loop:

    // --- Phase 2: Form the full inverse matrix ---
    for (int in = global_in; in < A.e_trick_size_inv[STILES_RANK]; in++) {
        myroutine = A.e_trick_inv[STILES_RANK][0 + 7 * in];
        i = A.e_trick_inv[STILES_RANK][1 + 7 * in];
        j = A.e_trick_inv[STILES_RANK][2 + 7 * in];
        k = A.e_trick_inv[STILES_RANK][3 + 7 * in];
        index1 = A.e_trick_inv[STILES_RANK][4 + 7 * in];
        index2 = A.e_trick_inv[STILES_RANK][5 + 7 * in];
        index3 = A.e_trick_inv[STILES_RANK][6 + 7 * in];

        switch (myroutine) {
            case 4: // DLAUUM
            {
                auto& tile_invA = A.inverse_tiles[index1];
                auto new_op = [&]() {
                    sTiles::core_dlauum(stiles_uplo, tile_invA.width, tile_invA.elements, tile_invA.height);
                    sTiles::corr_dmirr(tile_invA.height, tile_invA.elements, tile_invA.elements, tile_invA.height);
                };
                auto old_op = [&](double* tile, int lda) { // Corrected signature
                    sTiles::core_dlauum(sTiles::Uplo::Upper, tile_invA.width, tile, lda);
                    mirroring(tile_invA.height, tile, tile, lda);
                };
                
                #if defined(STILE_VERIFY_AGAINST_OLD)
                    STILE_VERIFY_EXEC("TRTRI_DLAUUM", tile_invA.width, tile_invA.width, tile_invA.elements, tile_invA.height, new_op, old_op);
                #elif defined(STILE_USE_NEW_API)
                    new_op();
                #else
                    old_op(tile_invA.elements, tile_invA.height);
                #endif
                break;
            }
            case 5: // DGEMM for diagonal update
            {
                auto& tile_A_offdiag = A.dense_tiles[index2];
                auto& tile_invA_offdiag = A.inverse_tiles[index2];
                auto& tile_invA_diag = A.inverse_tiles[index1];
                auto new_op = [&]() { /* ... */ };
                auto old_op = [&](double* tile, int lda) { // Corrected signature
                    sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                      tile_A_offdiag.height, tile_A_offdiag.height, tile_A_offdiag.width,
                                      mzone, tile_A_offdiag.elements, tile_A_offdiag.height,
                                      tile_invA_offdiag.elements, tile_invA_offdiag.height,
                                      zone, tile, lda); // Use provided lda
                };
                
                in_cond_wait(i, k, 2);
                #if defined(STILE_VERIFY_AGAINST_OLD)
                    STILE_VERIFY_EXEC("TRTRI_DGEMM_DIAG", tile_A_offdiag.height, tile_A_offdiag.height, tile_invA_diag.elements, tile_invA_diag.height, new_op, old_op);
                #elif defined(STILE_USE_NEW_API)
                    new_op();
                #else
                    old_op(tile_invA_diag.elements, tile_invA_diag.height);
                #endif
                break;
            }
            case 6: // Set dependency flag
                in_cond_set(i, i, 2);
                break;
            case 7: // DGEMM for off-diagonal update
            {
                auto& tile_A = A.dense_tiles[index1];
                auto& tile_invA1 = A.inverse_tiles[index2];
                auto& tile_invA2 = A.inverse_tiles[index3];
                auto new_op = [&]() { /* ... */ };
                auto old_op = [&](double* tile, int lda) { // Corrected signature
                    sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                      tile_A.height, tile_invA1.height, tile_A.width,
                                      -1.0, tile_A.elements, tile_A.height,
                                      tile_invA1.elements, tile_invA1.height,
                                      zone, tile, lda); // Use provided lda
                };
                
                in_cond_wait(j, k, 2);
                #if defined(STILE_VERIFY_AGAINST_OLD)
                    STILE_VERIFY_EXEC("TRTRI_DGEMM_OFF1", tile_A.height, tile_invA1.height, tile_invA2.elements, tile_invA2.height, new_op, old_op);
                #elif defined(STILE_USE_NEW_API)
                    new_op();
                #else
                    old_op(tile_invA2.elements, tile_invA2.height);
                #endif
                break;
            }
            case 8: // DGEMM for off-diagonal update
            {
                auto& tile_A = A.dense_tiles[index1];
                auto& tile_invA1 = A.inverse_tiles[index2];
                auto& tile_invA2 = A.inverse_tiles[index3];
                auto new_op = [&]() { /* ... */ };
                auto old_op = [&](double* tile, int lda) { // Corrected signature
                    sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                      tile_A.height, tile_invA1.width, tile_A.width,
                                      mzone, tile_A.elements, tile_A.height,
                                      tile_invA1.elements, tile_invA1.height,
                                      zone, tile, lda); // Use provided lda
                };

                in_cond_wait(k, j, 2);
                #if defined(STILE_VERIFY_AGAINST_OLD)
                    STILE_VERIFY_EXEC("TRTRI_DGEMM_OFF2", tile_A.height, tile_invA1.width, tile_invA2.elements, tile_invA2.height, new_op, old_op);
                #elif defined(STILE_USE_NEW_API)
                    new_op();
                #else
                    old_op(tile_invA2.elements, tile_invA2.height);
                #endif
                break;
            }
            case 9: // Set dependency flag
                in_cond_set(i, j, 2);
                break;
        }
    }

    in_finalize();
}
