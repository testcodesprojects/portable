/**
 * @file compute_gpu.hpp
 * @brief GPU-accelerated computation kernels for tiled Cholesky factorization.
 *
 * Implements CUDA-based parallel Cholesky factorization using cuSOLVER and cuBLAS
 * for GPU-accelerated dense tile operations with hybrid CPU-GPU execution support.
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

#ifndef H_COMPUTE_GPU
#define H_COMPUTE_GPU

#include "../control/common.h" // Include the header with C function declarations
#include <stiles_process.h> // Include the header with C function declarations
#include "../include/sparse_utils.hpp"
#include <omp.h>
#include "../common/core_lapack.hpp"
#include <math.h>
#include <iostream>
// GPU kernels - include from tools/gpu when available
#ifdef STILES_GPU
// TODO: Add new GPU kernel includes here when ready
// #include "../gpu/gpu_kernels.hpp"
#endif

#define CHECK_CUDA(call) \
    if (call != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUSOLVER(call) \
    if (call != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "CUSOLVER Error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

inline void dpotrf_redtree_boosted_101_gpu(stiles_context_t *stile){

    int uplo;
    TilesDescriptor A;
    STILES_sequence *sequence;
    STILES_request *request;

    stiles_unpack_args_4(uplo, A, sequence, request);
    cudaSetDevice(A.GPU_ID);
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH;

    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
    cublasCreate(&cublasH);

    CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, A.streams[STILES_RANK]));  // Set the stream for cusolver
    cublasSetStream(cublasH, A.streams[STILES_RANK]);

    int workspace_size = 0;
    double* workspace;
    int* devInfo;

    CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_UPPER, A.nb, A.dense_tiles_gpu[0].x, A.nb, &workspace_size));
    CHECK_CUDA(cudaMalloc((void**)&workspace, workspace_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));

    int k, m, n;
    int next_k;
    int next_m;
    int next_n;
    int ldak, ldam, ldan;
    int tempkn, tempmn;
    int index1, index2, index3;
    int myroutine = 0;

    double zone  = (double) 1.0;
    double mzone = (double)-1.0;
    int cpuntk = 0;
    if (sequence->status != 0)
        return;

    int sep = A.tree_sep;
    int indeprint = A.nt-2;
    int num_sep = ((sep*sep) - sep)/2 + sep;
    int* gemmcounter_redtree = (int*)calloc(num_sep, sizeof(int));
    int* gemmcounter_tasks = (int*)calloc(num_sep, sizeof(int));

    TaskDistribution *distribution;
    distribution = (TaskDistribution *)malloc(num_sep * sizeof(TaskDistribution));


    for (int ind = 0; ind < num_sep; ind++) {

        if(A.trees[ind]){
            int total_tasks = A.trees[ind]->num_tasks; //A.trees[ind]->max_nodes[0]; // total number of tasks
            distribution[ind] = stiles_calculateTaskDistribution(STILES_RANK, STILES_SIZE, total_tasks);
            gemmcounter_redtree[ind] =0;// distribution[ind].start_index;
            gemmcounter_tasks[ind] = distribution[ind].start_index;

        }
    }

    int index_survival = 0;
    ss_init(A.nt, A.nt, 0);
    int spec = 0;

    for(int i=0; i<A.e_trick_size[STILES_RANK];i++){

        myroutine = A.e_trick[STILES_RANK][0+7*i];
        m = A.e_trick[STILES_RANK][1+i*7];
        k = A.e_trick[STILES_RANK][2+i*7];
        n = A.e_trick[STILES_RANK][3+i*7];
        index1 = A.e_trick[STILES_RANK][4+i*7];
        index2 = A.e_trick[STILES_RANK][5+i*7];
        index3 = A.e_trick[STILES_RANK][6+i*7];

        tempkn = k == (A.nt-1) ? A.n-k*A.nb : A.nb;
        tempmn = m == (A.nt-1) ? A.n-m*A.nb : A.nb;

        ldak = BLKLDD(A, k);
        ldan = BLKLDD(A, n);
        ldam = BLKLDD(A, m);

        switch (myroutine) {

            case 1:

                cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_UPPER, tempkn, A.dense_tiles_gpu[index1].x, ldak, workspace, workspace_size, devInfo);
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                ss_cond_set(k, k, 1);

                break;
        
            case 2:

                ss_cond_wait(k, n, 1);
                cublasDsyrk(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, tempkn, A.nb, &mzone, A.dense_tiles_gpu[index1].x, ldan, &zone, A.dense_tiles_gpu[index2].x, ldak);
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                break;

            case 3:

                ss_cond_wait(k, k, 1);
                cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    A.nb, tempmn, &zone,
                    A.dense_tiles_gpu[index2].x, ldak,
                    A.dense_tiles_gpu[index1].x, ldak);
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                ss_cond_set(m, k, 1);

                break;
            
            case 4:

                ss_cond_wait(k, n, 1);
                ss_cond_wait(m, n, 1);
                cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                    A.nb, tempmn, A.nb, &mzone,
                    A.dense_tiles_gpu[index1].x, ldan,
                    A.dense_tiles_gpu[index2].x, ldan,
                    &zone,
                    A.dense_tiles_gpu[index3].x, ldak);
                cudaStreamSynchronize(A.streams[STILES_RANK]);

                break;

            case 5:

                ss_cond_wait(k, n, 1);
                cublasDsyrk(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                    tempkn, A.nb, &zone,
                    A.dense_tiles_gpu[index1].x, ldan,
                    &zone,
                    A.gpu_trees[STILES_RANK].x, ldak);
                cudaStreamSynchronize(A.streams[STILES_RANK]);

                break;

            case 6:

                A.trees[index2]->dependency[STILES_RANK] = index3;

                break;

            case 7:

                for(int i=0; i < STILES_SIZE; i++){
                    
                    if (i > 0) {
                        ss_cond_wait_tree_e_s_t_y_l_e(i, index3, A.trees[index1]->dependency);
                    }
                    cublasDgeam(cublasH,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        A.dense_tiles[index2].width, A.dense_tiles[index2].height,
                        &mzone,
                        A.gpu_trees[i].x, A.dense_tiles[index2].width,
                        &zone,
                        A.dense_tiles_gpu[index2].x, A.dense_tiles[index2].width,
                        A.dense_tiles_gpu[index2].x, A.dense_tiles[index2].width);     
                    
                    A.trees[index1]->dependency[i] = 0;

                }
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                break;

            case 8:

                ss_cond_wait(k, n, 1);
                ss_cond_wait(m, n, 1);
                cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                    A.nb, tempmn, A.nb, &zone,
                    A.dense_tiles_gpu[index1].x, ldan,
                    A.dense_tiles_gpu[index2].x, ldan,
                    &zone,
                    A.gpu_trees[STILES_RANK].x, ldak);
                cudaStreamSynchronize(A.streams[STILES_RANK]);

                break;

            case 9:

                A.trees[index2]->dependency[STILES_RANK] = 621075;
                break;

            case 10:

                for(int i=0; i < STILES_SIZE; i++){
                    
                    if (i > 0) {
                        ss_cond_wait_tree_e_s_t_y_l_e(i, 621075, A.trees[index1]->dependency);
                    }

                    cublasDgeam(cublasH,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        A.dense_tiles[index3].width, A.dense_tiles[index3].height,
                        &mzone,
                        A.gpu_trees[i].x, A.dense_tiles[index3].width,
                        &zone,
                        A.dense_tiles_gpu[index3].x, A.dense_tiles[index3].width,
                        A.dense_tiles_gpu[index3].x, A.dense_tiles[index3].width);

                    A.trees[index1]->dependency[i] = 0;

                }
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                break;
                
        }


        if(ss_aborted()) break;
    }

    ss_finalize();
    free(gemmcounter_redtree);
    free(gemmcounter_tasks);
    free(distribution);


}

inline void dpotrf_greentree_boosted_101_gpu(stiles_context_t *stile){

    int uplo;
    TilesDescriptor A;
    STILES_sequence *sequence;
    STILES_request *request;

    stiles_unpack_args_4(uplo, A, sequence, request);
    cudaSetDevice(A.GPU_ID);
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH;

    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
    cublasCreate(&cublasH);

    CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, A.streams[STILES_RANK]));  // Set the stream for cusolver
    cublasSetStream(cublasH, A.streams[STILES_RANK]);

    int workspace_size = 0;
    double* workspace;
    int* devInfo;

    CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_UPPER, A.nb, A.dense_tiles_gpu[0].x, A.nb, &workspace_size));
    CHECK_CUDA(cudaMalloc((void**)&workspace, workspace_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));
    
    int k, m, n;
    int index1, index2, index3, myroutine;
    int ldak, ldam, ldan;
    int tempkn, tempmn;
    double zone  = (double) 1.0;
    double mzone = (double)-1.0;

    if (sequence->status != 0)
        return;

    ss_init(A.nt, A.nt, 0);
    for(int i=0; i<A.e_trick_size[STILES_RANK];i++){

        myroutine = A.e_trick[STILES_RANK][0+7*i];
        m = A.e_trick[STILES_RANK][1+i*7];
        k = A.e_trick[STILES_RANK][2+i*7];
        n = A.e_trick[STILES_RANK][3+i*7];
        index1 = A.e_trick[STILES_RANK][4+i*7];
        index2 = A.e_trick[STILES_RANK][5+i*7];
        index3 = A.e_trick[STILES_RANK][6+i*7];

        tempkn = k == (A.nt-1) ? A.n-k*A.nb : A.nb;
        tempmn = m == (A.nt-1) ? A.n-m*A.nb : A.nb;

        ldak = BLKLDD(A, k);
        ldan = BLKLDD(A, n);
        ldam = BLKLDD(A, m);

        switch (myroutine) {

            case 1:

                cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_UPPER, tempkn, A.dense_tiles_gpu[index1].x, ldak, workspace, workspace_size, devInfo);                
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                ss_cond_set(k, k, 1);

                break;
            
            case 2:

                ss_cond_wait(k, n, 1);

                cublasDsyrk(cublasH, 
                    CUBLAS_FILL_MODE_UPPER, 
                    CUBLAS_OP_T,  // A^T
                    tempkn, A.nb, 
                    &mzone, A.dense_tiles_gpu[index1].x, ldan, 
                    &zone, A.dense_tiles_gpu[index2].x, ldak);
                cudaStreamSynchronize(A.streams[STILES_RANK]);

                break;

            case 3:

                ss_cond_wait(k, k, 1);
                cublasDtrsm(cublasH, 
                    CUBLAS_SIDE_LEFT, 
                    CUBLAS_FILL_MODE_UPPER, 
                    CUBLAS_OP_T,   // A^T
                    CUBLAS_DIAG_NON_UNIT, 
                    A.nb, tempmn, 
                    &zone, A.dense_tiles_gpu[index2].x, ldak, 
                            A.dense_tiles_gpu[index1].x, ldak);
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                ss_cond_set(m, k, 1);

                break;
            
            case 4:

                ss_cond_wait(k, n, 1);
                ss_cond_wait(m, n, 1);
                cublasDgemm(cublasH, 
                    CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose first matrix, no transpose second
                    A.nb, tempmn, A.nb, 
                    &mzone, A.dense_tiles_gpu[index1].x, ldan, 
                            A.dense_tiles_gpu[index2].x, ldan, 
                    &zone,  A.dense_tiles_gpu[index3].x, ldak);
                cudaStreamSynchronize(A.streams[STILES_RANK]);

                break;
    
        }

        if(ss_aborted()) break;
    }

    ss_finalize();
    cudaFree(workspace);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);
}

inline void stiles_pdtrtri_gpu(stiles_context_t *stile)
{
    int uplo;
    TilesDescriptor A;
    STILES_sequence *sequence;
    STILES_request *request;

    int info;
    int myroutine, i, j, k;
    int index1, index2, index3; 

    double zone  = 1.0;
    double mzone = -1.0;
    double zerzone = 0.0;

    // Unpack arguments
    stiles_unpack_args_4(uplo, A, sequence, request);
    if (sequence->status != 0) {
        printf("Error! \n");
        // return or handle error as needed.
    }
    
    // Set the GPU device and create a cuBLAS handle on the proper stream
    cudaSetDevice(A.GPU_ID);
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    cublasSetStream(cublasH, A.streams[STILES_RANK]);

    int* devInfo;
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));

    in_init(A.nt, A.nt, 0);
    int global_in = 0;

    for (int in = 0; in < A.e_trick_size_inv[STILES_RANK]; in++) {
        myroutine = A.e_trick_inv[STILES_RANK][0 + 7 * in];
        i = A.e_trick_inv[STILES_RANK][1 + in * 7];
        j = A.e_trick_inv[STILES_RANK][2 + in * 7];
        k = A.e_trick_inv[STILES_RANK][3 + in * 7];
        index1 = A.e_trick_inv[STILES_RANK][4 + in * 7];
        index2 = A.e_trick_inv[STILES_RANK][5 + in * 7];
        index3 = A.e_trick_inv[STILES_RANK][6 + in * 7];

        switch (myroutine) {
            case 1:
                // Case 1: Triangular solve (equivalent to plasma_core_dtrsm)
                cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                            A.dense_tiles[index1].width, A.dense_tiles[index1].width,
                            &zone,
                            A.dense_tiles_gpu[index1].x, A.dense_tiles[index1].width,
                            A.inverse_tiles_gpu[index1].x, A.inverse_tiles[index1].width);
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                break;

            case 2:
                // Case 2: Triangular matrix multiply (equivalent to plasma_core_dtrmm)
                cublasDtrmm_v2(cublasH,
                               CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                               CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                               A.dense_tiles[index2].height, A.dense_tiles[index2].width,
                               &zone,
                               A.inverse_tiles_gpu[index1].x, A.inverse_tiles_gpu[index1].height,
                               A.dense_tiles_gpu[index2].x, A.dense_tiles_gpu[index2].height,
                               A.dense_tiles_gpu[index2].x, A.dense_tiles_gpu[index2].height);
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                break;

            case 3:
                sTiles::Control::Barrier(stile);
                global_in = in + 1;  // Save the next index for the second loop
                goto exit_first_loop;  // Break out of the first loop

        }
    }

    exit_first_loop:  
    
    for(int in=global_in; in<A.e_trick_size_inv[STILES_RANK];in++){
        
        myroutine = A.e_trick_inv[STILES_RANK][0 + 7 * in];
        i = A.e_trick_inv[STILES_RANK][1 + in * 7];
        j = A.e_trick_inv[STILES_RANK][2 + in * 7];
        k = A.e_trick_inv[STILES_RANK][3 + in * 7];
        index1 = A.e_trick_inv[STILES_RANK][4 + in * 7];
        index2 = A.e_trick_inv[STILES_RANK][5 + in * 7];
        index3 = A.e_trick_inv[STILES_RANK][6 + in * 7];

        switch (myroutine) {

            case 4:
                // Case 4: Compute U * U^T (emulating plasma_core_dlauum) and mirror the result.
                // Compute U * U^T using GEMM:
                cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                            A.inverse_tiles[index1].width, A.inverse_tiles[index1].width,
                            A.inverse_tiles[index1].width,
                            &zone,
                            A.inverse_tiles_gpu[index1].x, A.inverse_tiles[index1].height,
                            A.inverse_tiles_gpu[index1].x, A.inverse_tiles[index1].height,
                            &zerzone,
                            A.inverse_tiles_gpu[index1].x, A.inverse_tiles[index1].height);
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                // Mirror the computed upper triangle to the lower part
                mirroring_gpu(A.inverse_tiles[index1].height,
                              A.inverse_tiles_gpu[index1].x,
                              A.inverse_tiles_gpu[index1].x,
                              A.inverse_tiles[index1].height,
                              A.streams[STILES_RANK]);
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                break;

            case 5:
                // Case 5: GEMM with NoTrans for A and Trans for B (emulating plasma_core_dgemm)
                in_cond_wait(i, k, 2);
                cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                            A.dense_tiles[index2].height, A.dense_tiles[index2].height, A.dense_tiles[index2].width,
                            &mzone,
                            A.dense_tiles_gpu[index2].x, A.dense_tiles[index2].height,
                            A.inverse_tiles_gpu[index2].x, A.inverse_tiles[index2].height,
                            &zone,
                            A.inverse_tiles_gpu[index1].x, A.inverse_tiles[index1].height);
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                break;

            case 6:
                // Case 6: Condition set (host-side)
                in_cond_set(i, i, 2);
                break;

            case 7:
                // Case 7: Wait on condition, then perform GEMM with NoTrans and Trans
                in_cond_wait(j, k, 2);     
                cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                            A.dense_tiles[index1].height, A.inverse_tiles[index2].height, A.dense_tiles[index1].width,
                            &mzone,
                            A.dense_tiles_gpu[index1].x, A.dense_tiles[index1].height,
                            A.inverse_tiles_gpu[index2].x, A.inverse_tiles[index2].height,
                            &zone,
                            A.inverse_tiles_gpu[index3].x, A.inverse_tiles[index3].height);
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                

                break;

            case 8:
                // Case 8: Wait on condition, then perform GEMM with NoTrans for both operands.
                in_cond_wait(k, j, 2);
                cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                            A.dense_tiles[index1].height, A.inverse_tiles[index2].width, A.dense_tiles[index1].width,
                            &mzone,
                            A.dense_tiles_gpu[index1].x, A.dense_tiles[index1].height,
                            A.inverse_tiles_gpu[index2].x, A.inverse_tiles[index2].height,
                            &zone,
                            A.inverse_tiles_gpu[index3].x, A.inverse_tiles[index3].height);
                cudaStreamSynchronize(A.streams[STILES_RANK]);
                break;

            case 9:

                in_cond_set(i, j, 2);
                break;

            default:
                // Optional: handle unexpected routine codes
                printf("Unknown routine: %d\n", myroutine);
                break;

        }
    }

    in_finalize();
    cudaFree(devInfo);
    cublasDestroy(cublasH);
}

#endif






