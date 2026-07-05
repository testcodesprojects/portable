/**
 * @file debugging.hpp
 * @brief Debugging utilities for matrix operations.
 *
 * Provides debugging helper functions including matrix mirroring and
 * verification routines for validating factorization results.
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

#ifndef H_INTERNAL_DEBUGGING
#define H_INTERNAL_DEBUGGING

#include "../control/common.h"
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <iomanip>

void mirroring_debug(int n, double *A, double *B, int lda) {

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            B[j * lda + i] = A[i * lda + j];
        }
    }
}

void stiles_pdtrtri_cpu_debug(stiles_context_t *stile)
{
    int uplo;
    TilesDescriptor A;
    STILES_sequence *sequence;
    STILES_request *request;

    int info;
    int myroutine, task_reached;
    int index1, index2, index3; 

    double zone  = (double) 1.0;
    double mzone = (double)-1.0;

    stiles_unpack_args_4(uplo, A, sequence, request);
    if (sequence->status != 0) printf("Error! \n");
        //return 1;

    in_init(A.nt, A.nt, 0);
    int myrank = STILES_RANK;
    int worldsize = STILES_SIZE;
    bool* on_off_tiles = A.of_perm;
    int* magic_perm1 = A.magic_perm1;
    bool setting = false;
    int ii=0, jj=0;
    int j, k=0;
    int i = A.nt - 1 - myrank;
    int num_tiles = A.nt;

    while (i >= 0) {

        for(j = i; j < num_tiles; j++) {

            index1 = magic_perm1[i*(2*num_tiles-i-1)/2 + i];
            index2 = magic_perm1[i*(2*num_tiles-i-1)/2 + j]; // i switch it!

            if(i==j){

                sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit, 
                                  A.dense_tiles[index1].width, A.dense_tiles[index1].width, 1.0,
                                  A.dense_tiles[index1].elements, A.dense_tiles[index1].width,  // A: triangular matrix
                                  A.inverse_tiles[index1].elements, A.dense_tiles[index1].width); // B: identity matrix (right-hand side)

               //plasma_core_dtrtri(sTilesUpper, sTilesNonUnit, A.dense_tiles[index1].width, A.dense_tiles[index1].elements, A.dense_tiles[index1].width); //inverse of the upper A(k,k) and it is put in the lower A(k,k)

            }else{
                
                if(on_off_tiles[i*(2*num_tiles-i-1)/2 + j]){
                    
                    sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                      A.dense_tiles[index2].height, A.dense_tiles[index2].width, zone,   // M and N (rows and columns of B)
                                      A.inverse_tiles[index1].elements, A.inverse_tiles[index1].height,  // A and its leading dimension (LDA)
                                      A.dense_tiles[index2].elements, A.dense_tiles[index2].height); 
    
                }
            }

        }

        i = i - worldsize; 
    }
    
    sTiles::Control::Barrier(stile);

    i = num_tiles - 1 - myrank;

    while (i >= 0) {

        for(j = num_tiles - 1; j >= i; j--) {

            if(i==j){

                index1 = magic_perm1[i*(2*num_tiles-i-1)/2 + i];

                sTiles::core_dlauum(sTiles::Uplo::Upper, A.inverse_tiles[index1].width, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].height); //L * L ^T
                mirroring_debug( A.inverse_tiles[index1].height, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].elements,  A.inverse_tiles[index1].height); //copy the upper of dense_tiles tile to the upper & lower inverse_tiles
    
                // std::cout << "--------------------------------------------------------" << std::endl;
               //  std::cout << "RANK: " << STILES_RANK << " dlauum and mirroring: " << index1 << std::endl;
                // std::cout << "--------------------------------------------------------" << std::endl;

                // sTiles::core_dlauum(sTiles::Uplo::Upper, A.inverse_tiles[index1].width,  A.dense_tiles[index1].elements, A.inverse_tiles[index1].width); //LT x L
                // copy_lower_to_upper(A.inverse_tiles[index1].width, A.dense_tiles[index1].elements, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].width);

                if((i+1)<=(num_tiles-1)){


                    for(k = i+1; k<=(num_tiles-1); k++){

                        index2 = magic_perm1[i*(2*num_tiles-i-1)/2 + k];

                        if(on_off_tiles[i*(2*num_tiles-i-1)/2 + k]){
                            in_cond_wait(i, k, 2);
                            // std::cout << "1WWWWAIT" << STILES_RANK << ": (" <<  magic_perm1[i*(2*num_tiles-i-1)/2 + k] << ")" << std::endl;

                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                A.dense_tiles[index2].height, A.dense_tiles[index2].height, A.dense_tiles[index2].width,
                                mzone, A.dense_tiles[index2].elements, A.dense_tiles[index2].height,
                                        A.inverse_tiles[index2].elements, A.inverse_tiles[index2].height,
                                    zone, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].height);
            
                    //        std::cout << "GEMM1 by: " << STILES_RANK << " INV["<<index1<<"] = INV["<<index1<<"] - M["<<index2<<"]*INV["<<index2<<"] , waited: "  << index2 << std::endl;

                        }

                    }

                    in_cond_set(i, i, 2);
                   // std::cout << "1set" << STILES_RANK << ": (" << magic_perm1[i*(2*num_tiles-i-1)/2 + i] << ")" << std::endl;
    
                }else{
                    in_cond_set(i, i, 2);
                  //  std::cout << "majset" << STILES_RANK << ": (" << magic_perm1[i*(2*num_tiles-i-1)/2 + i] << ")" << std::endl;
               
                }


            }else{

                if((i+1)<=(num_tiles-1)) {setting = true; ii =0; jj =0;}
                for(k = i+1; k<=(num_tiles-1); k++){

                    if(k > j){

                        index1 = magic_perm1[i*(2*num_tiles-i-1)/2 + k];
                        index2 = magic_perm1[j*(2*num_tiles-j-1)/2 + k];
                        index3 = magic_perm1[i*(2*num_tiles-i-1)/2 + j];

                        if(on_off_tiles[i*(2*num_tiles-i-1)/2 + j] && on_off_tiles[j*(2*num_tiles-j-1)/2 + k] && on_off_tiles[i*(2*num_tiles-i-1)/2 + k]){
                            in_cond_wait(j, k, 2);
                         //   std::cout << "2WWWWAIT" << STILES_RANK << ": (" <<  magic_perm1[j*(2*num_tiles-j-1)/2 + k] << ")" << std::endl;

                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                A.dense_tiles[index1].height, A.inverse_tiles[index2].height, A.dense_tiles[index1].width,
                                -1, A.dense_tiles[index1].elements, A.dense_tiles[index1].height, // i is less than j
                                        A.inverse_tiles[index2].elements, A.inverse_tiles[index2].height,
                                    zone, A.inverse_tiles[index3].elements, A.inverse_tiles[index3].height);
                              //      std::cout << "GEMM2 by: " << STILES_RANK << " INV["<<index3<<"] = INV["<<index3<<"] - M["<<index1<<"]*INV["<<index2<<"], waited: "  << index1 << std::endl;

                                    //in_cond_set(i, k, 2);
                                    //in_cond_set(i, j, 2);
                                  //  std::cout << "2set" << STILES_RANK << ": (" << magic_perm1[i*(2*num_tiles-i-1)/2 + k] << ")" << std::endl;
                                  //  std::cout << "2set" << STILES_RANK << ": (" << magic_perm1[i*(2*num_tiles-i-1)/2 + j] << ")" << std::endl;
                                    ii = i;
                                    jj = j;
        
                        }

                    }else{

                        index1 = magic_perm1[i*(2*num_tiles-i-1)/2 + k];
                        index2 = magic_perm1[k*(2*num_tiles-k-1)/2 + j];
                        index3 = magic_perm1[i*(2*num_tiles-i-1)/2 + j];

                        if(on_off_tiles[i*(2*num_tiles-i-1)/2 + j] && on_off_tiles[k*(2*num_tiles-k-1)/2 + j] && on_off_tiles[i*(2*num_tiles-i-1)/2 + k]){

                            in_cond_wait(k, j, 2);
                           // std::cout << "3WWWWAIT" << STILES_RANK << ": (" << magic_perm1[k*(2*num_tiles-k-1)/2 + j] << ")" << std::endl;

                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                A.dense_tiles[index1].height, A.inverse_tiles[index2].width, A.dense_tiles[index1].width,
                                mzone, A.dense_tiles[index1].elements, A.dense_tiles[index1].height, // i is less than j
                                        A.inverse_tiles[index2].elements, A.inverse_tiles[index2].height,
                                    zone, A.inverse_tiles[index3].elements, A.inverse_tiles[index3].height);

                            //in_cond_set(i, k, 2);
                            //in_cond_set(i, j, 2);
                         //   std::cout << "GEMM3 by: " << STILES_RANK << " INV["<<index3<<"] = INV["<<index3<<"] - M["<<index1<<"]*INV["<<index2<<"] , waited: "  << index2 << std::endl;

                           // std::cout << "3set" << STILES_RANK << ": (" << magic_perm1[i*(2*num_tiles-i-1)/2 + k] << ")" << std::endl;
                            //std::cout << "3set" << STILES_RANK << ": (" << magic_perm1[i*(2*num_tiles-i-1)/2 + j] << ")" << std::endl;
                                ii = i;
                                jj = j;

                        }
                        

                    }

                    
                }
                if(setting) {in_cond_set(ii, jj, 2);

               // std::cout << "23set" << STILES_RANK << ": (" << magic_perm1[ii*(2*num_tiles-ii-1)/2 + jj] << ")" << std::endl;

                }
                setting = false;


            }
        }

        i = i - worldsize; 
    }

    in_finalize();

}

void stiles_pdtrtri_cpu_debug_2(stiles_context_t *stile)
{
    int uplo;
    TilesDescriptor A;
    STILES_sequence *sequence;
    STILES_request *request;

    int info;
    int myroutine, task_reached;
    int index1, index2, index3; 

    double zone  = (double) 1.0;
    double mzone = (double)-1.0;

    stiles_unpack_args_4(uplo, A, sequence, request);
    if (sequence->status != 0) printf("Error! \n");
        //return 1;

    in_init(A.nt, A.nt, 0);
    int myrank = STILES_RANK;
    int worldsize = STILES_SIZE;
    bool* on_off_tiles = A.of_perm;
    int* magic_perm1 = A.magic_perm1;
    bool setting = false;
    int ii=0, jj=0;
    int j, k=0;
    int i = A.nt - 1 - myrank;
    int num_tiles = A.nt;

    while (i >= 0) {

        for(j = i; j < num_tiles; j++) {

            index1 = magic_perm1[i*(2*num_tiles-i-1)/2 + i];
            index2 = magic_perm1[i*(2*num_tiles-i-1)/2 + j]; // i switch it!

            if(i==j){

                sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit, 
                                  A.dense_tiles[index1].width, A.dense_tiles[index1].width, 1.0,
                                  A.dense_tiles[index1].elements, A.dense_tiles[index1].width,  // A: triangular matrix
                                  A.inverse_tiles[index1].elements, A.dense_tiles[index1].width); // B: identity matrix (right-hand side)

               //plasma_core_dtrtri(sTilesUpper, sTilesNonUnit, A.dense_tiles[index1].width, A.dense_tiles[index1].elements, A.dense_tiles[index1].width); //inverse of the upper A(k,k) and it is put in the lower A(k,k)

            }else{
                
                if(on_off_tiles[i*(2*num_tiles-i-1)/2 + j]){
                    
                    sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                      A.dense_tiles[index2].height, A.dense_tiles[index2].width, zone,   // M and N (rows and columns of B)
                                      A.inverse_tiles[index1].elements, A.inverse_tiles[index1].height,  // A and its leading dimension (LDA)
                                      A.dense_tiles[index2].elements, A.dense_tiles[index2].height); 
    
                }
            }

        }

        i = i - worldsize; 
    }
    
    sTiles::Control::Barrier(stile);

    i = num_tiles - 1;

    while (i >= 0) {

        for(j = num_tiles - 1; j >= i; j--) {

            if(i==j){

                index1 = magic_perm1[i*(2*num_tiles-i-1)/2 + i];
               if(myrank==(index1 % worldsize)){

                sTiles::core_dlauum(sTiles::Uplo::Upper, A.inverse_tiles[index1].width, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].height); //L * L ^T
                mirroring_debug( A.inverse_tiles[index1].height, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].elements,  A.inverse_tiles[index1].height); //copy the upper of dense_tiles tile to the upper & lower inverse_tiles
    
                // std::cout << "--------------------------------------------------------" << std::endl;
               //  std::cout << "RANK: " << STILES_RANK << " dlauum and mirroring: " << index1 << std::endl;
                // std::cout << "--------------------------------------------------------" << std::endl;

                // sTiles::core_dlauum(sTiles::Uplo::Upper, A.inverse_tiles[index1].width,  A.dense_tiles[index1].elements, A.inverse_tiles[index1].width); //LT x L
                // copy_lower_to_upper(A.inverse_tiles[index1].width, A.dense_tiles[index1].elements, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].width);

                if((i+1)<=(num_tiles-1)){


                    //for(k = i+1; k<=(num_tiles-1); k++){
                    for(k = (num_tiles - 1); k >= (i + 1); k--) {

                        index2 = magic_perm1[i*(2*num_tiles-i-1)/2 + k];

                        if(on_off_tiles[i*(2*num_tiles-i-1)/2 + k]){
                            in_cond_wait(i, k, 2);
                            // std::cout << "1WWWWAIT" << STILES_RANK << ": (" <<  magic_perm1[i*(2*num_tiles-i-1)/2 + k] << ")" << std::endl;

                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                A.dense_tiles[index2].height, A.dense_tiles[index2].height, A.dense_tiles[index2].width,
                                mzone, A.dense_tiles[index2].elements, A.dense_tiles[index2].height,
                                        A.inverse_tiles[index2].elements, A.inverse_tiles[index2].height,
                                    zone, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].height);
            
                //            std::cout << "GEMM1 by: " << STILES_RANK << " INV["<<index1<<"] = INV["<<index1<<"] - M["<<index2<<"]*INV["<<index2<<"] , waited: "  << index2 << std::endl;

                        }

                    }

                    in_cond_set(i, i, 2);
                   // std::cout << "1set" << STILES_RANK << ": (" << magic_perm1[i*(2*num_tiles-i-1)/2 + i] << ")" << std::endl;
    
                }else{
                    in_cond_set(i, i, 2);
                  //  std::cout << "majset" << STILES_RANK << ": (" << magic_perm1[i*(2*num_tiles-i-1)/2 + i] << ")" << std::endl;
               
                }
                }

            }else{

                index3 = magic_perm1[i*(2*num_tiles-i-1)/2 + j];
                if(myrank==(index3 % worldsize)){ 

                if((i+1)<=(num_tiles-1)) {setting = true; ii =0; jj =0;}
                //for(k = i+1; k<=(num_tiles-1); k++){
                for(k = (num_tiles-1); k >= (i+1); k--) {

                    if(k > j){

                        index1 = magic_perm1[i*(2*num_tiles-i-1)/2 + k];
                        index2 = magic_perm1[j*(2*num_tiles-j-1)/2 + k];

                        if(on_off_tiles[i*(2*num_tiles-i-1)/2 + j] && on_off_tiles[j*(2*num_tiles-j-1)/2 + k] && on_off_tiles[i*(2*num_tiles-i-1)/2 + k]){
                            in_cond_wait(j, k, 2);
                         //   std::cout << "2WWWWAIT" << STILES_RANK << ": (" <<  magic_perm1[j*(2*num_tiles-j-1)/2 + k] << ")" << std::endl;

                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                A.dense_tiles[index1].height, A.inverse_tiles[index2].height, A.dense_tiles[index1].width,
                                -1, A.dense_tiles[index1].elements, A.dense_tiles[index1].height, // i is less than j
                                        A.inverse_tiles[index2].elements, A.inverse_tiles[index2].height,
                                    zone, A.inverse_tiles[index3].elements, A.inverse_tiles[index3].height);
                              //      std::cout << "GEMM2 by: " << STILES_RANK << " INV["<<index3<<"] = INV["<<index3<<"] - M["<<index1<<"]*INV["<<index2<<"], waited: "  << index1 << std::endl;

                                    //in_cond_set(i, k, 2);
                                    //in_cond_set(i, j, 2);
                                  //  std::cout << "2set" << STILES_RANK << ": (" << magic_perm1[i*(2*num_tiles-i-1)/2 + k] << ")" << std::endl;
                                  //  std::cout << "2set" << STILES_RANK << ": (" << magic_perm1[i*(2*num_tiles-i-1)/2 + j] << ")" << std::endl;
                                    ii = i;
                                    jj = j;
        
                        }

                    }else{

                        index1 = magic_perm1[i*(2*num_tiles-i-1)/2 + k];
                        index2 = magic_perm1[k*(2*num_tiles-k-1)/2 + j];
                        index3 = magic_perm1[i*(2*num_tiles-i-1)/2 + j];

                        if(on_off_tiles[i*(2*num_tiles-i-1)/2 + j] && on_off_tiles[k*(2*num_tiles-k-1)/2 + j] && on_off_tiles[i*(2*num_tiles-i-1)/2 + k]){

                            in_cond_wait(k, j, 2);
                           // std::cout << "3WWWWAIT" << STILES_RANK << ": (" << magic_perm1[k*(2*num_tiles-k-1)/2 + j] << ")" << std::endl;

                            sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                A.dense_tiles[index1].height, A.inverse_tiles[index2].width, A.dense_tiles[index1].width,
                                mzone, A.dense_tiles[index1].elements, A.dense_tiles[index1].height, // i is less than j
                                        A.inverse_tiles[index2].elements, A.inverse_tiles[index2].height,
                                    zone, A.inverse_tiles[index3].elements, A.inverse_tiles[index3].height);

                            //in_cond_set(i, k, 2);
                            //in_cond_set(i, j, 2);
                          //  std::cout << "GEMM3 by: " << STILES_RANK << " INV["<<index3<<"] = INV["<<index3<<"] - M["<<index1<<"]*INV["<<index2<<"] , waited: "  << index2 << std::endl;

                           // std::cout << "3set" << STILES_RANK << ": (" << magic_perm1[i*(2*num_tiles-i-1)/2 + k] << ")" << std::endl;
                            //std::cout << "3set" << STILES_RANK << ": (" << magic_perm1[i*(2*num_tiles-i-1)/2 + j] << ")" << std::endl;
                                ii = i;
                                jj = j;

                        }
                        

                    }

                    
                }
                if(setting) {in_cond_set(ii, jj, 2);

               // std::cout << "23set" << STILES_RANK << ": (" << magic_perm1[ii*(2*num_tiles-ii-1)/2 + jj] << ")" << std::endl;

                }
                setting = false;

            }
            }
        }

        i = i - 1; 
    }

    in_finalize();

}

#endif




