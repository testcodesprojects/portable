/**
 * @file dags.hpp
 * @brief DAG (Directed Acyclic Graph) utilities for task dependency tracking.
 *
 * Provides helper functions for constructing and exporting task dependency
 * graphs used in parallel tiled matrix factorization scheduling.
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

#ifndef H_DAGD_GPU
#define H_DAGS_GPU

#include "../control/common.h"
#include <stdlib.h> 
#include <math.h>
#include <omp.h>
#include <iostream>
#include <iomanip>

int** helper_for_exporting_dag(int num_tiles) {
    int** matrix = (int**)malloc(num_tiles * sizeof(int*));
    if (!matrix) { // Check for successful allocation
        return NULL;
    }

    for (int i = 0; i < num_tiles; i++) {
        matrix[i] = (int*)malloc(num_tiles * sizeof(int)); // Corrected 'cols' to 'num_tiles'
        if (!matrix[i]) { // Check for successful allocation
            for (int j = 0; j < i; j++) {
                free(matrix[j]); // Free previously allocated memory
            }
            free(matrix);
            return NULL;
        }

        for (int j = 0; j < num_tiles; j++) {
            matrix[i][j] = 0; // Initialize all tiles as 'off'
        }
    }
    return matrix;
}

int export_dag_tree_inverse(stiles_context_t *stile)
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

    //size of the matrix: 6 by 6
    A.nt = 6;

    in_init(A.nt, A.nt, 0);
    int myrank = STILES_RANK;
    int worldsize = STILES_SIZE;
    bool* on_off_tiles = A.of_perm;
    int* magic_perm1 = A.magic_perm1;
    bool setting = false;
    int num_tiles = A.nt;
    int i, j, k, ii, jj;

    FILE *file1 = fopen("/home/abdulfe/Documents/postdoc/STiles/dev/sTiles_25.04.01/exported_txts/edges.txt", "w");
    FILE *file2 = fopen("/home/abdulfe/Documents/postdoc/STiles/dev/sTiles_25.04.01/exported_txts/nodes.txt", "w");

    if (file1 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    if (file2 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }


    for(int i=0; i<num_tiles; i++){
        for(int j=0; j<i; j++){
            on_off_tiles[j*(2*A.nt-j-1)/2 + i] = false;
        }
    }


    if(true){

        for(int i=0; i<num_tiles; i++){
            on_off_tiles[i*(2*A.nt-i-1)/2 + num_tiles-1] = true;
            on_off_tiles[i*(2*A.nt-i-1)/2 + i] = true;

        }

    }

    on_off_tiles[0*(2*A.nt-0-1)/2 + 1] = true;
    on_off_tiles[1*(2*A.nt-1-1)/2 + 2] = true;
    on_off_tiles[2*(2*A.nt-2-1)/2 + 3] = true;
    on_off_tiles[3*(2*A.nt-3-1)/2 + 4] = true;
    on_off_tiles[4*(2*A.nt-4-1)/2 + 5] = true;

    for(int i=0; i<num_tiles; i++){
        for(int j=0; j<i; j++){
            on_off_tiles[j*(2*A.nt-j-1)/2 + i] = true;
        }
    }

    on_off_tiles[i*(2*A.nt-i-1)/2 + i] = true;

    /*on_off_tiles[0*(2*A.nt-0-1)/2 + 2] = false;
    on_off_tiles[0*(2*A.nt-0-1)/2 + 3] = false;
    on_off_tiles[0*(2*A.nt-0-1)/2 + 4] = false;

    on_off_tiles[1*(2*A.nt-1-1)/2 + 3] = false;
    on_off_tiles[1*(2*A.nt-1-1)/2 + 4] = false;

    on_off_tiles[2*(2*A.nt-2-1)/2 + 4] = false;*/


    // Print the on_off_tiles matrix in a readable format
    printf("On-Off Tiles Matrix:\n");
        for (int j = 0; j < num_tiles; j++) {
            for (int i = 0; i < num_tiles; i++) {

                if (j <=i) { // Ensuring the lower-triangular structure is accessed properly
                    printf("%d ", on_off_tiles[j*(2*A.nt-j-1)/2 + i] ? 1 : 0);
                } else {
                    printf("  "); // Align for readability
                }
        }
        printf("\n");
    }


    int** optype = helper_for_exporting_dag(num_tiles);
    int** opnum = helper_for_exporting_dag(num_tiles);
    int** invopnum = helper_for_exporting_dag(num_tiles);
    int trtri = 10000, trsm = 20000, syrk =30000, gemms =40000, lacpy = 50000, lauum = 60000;


    i = A.nt - 1 - STILES_RANK;

    while (i >= 0) {

        for(j = i; j < num_tiles; j++) {

            index1 = magic_perm1[i*(2*num_tiles-i-1)/2 + i];
            index2 = magic_perm1[i*(2*num_tiles-i-1)/2 + j]; // i switch it!

            if(i==j){

                trtri +=1;
                opnum[i][i] = trtri;
                fprintf(file2, " '%d': ('TRSM', '#EFB717'), \n", opnum[i][i]); //opnum[i][i]  is the number of the unique node.
                printf("plasma_core_dtrsm on Tile: %d %d \n", i, i);

            }else{
                
                if(on_off_tiles[i*(2*num_tiles-i-1)/2 + j]){
                    
                    trsm +=1;
                    opnum[i][j] = trsm;
                    fprintf(file1, "('%d' , '%d'), \n", opnum[i][i], opnum[i][j]); //making the connection
                    fprintf(file2, " '%d': ('TRMM', '#F0911B'), \n", opnum[i][j]); //opnum[i][j] is the number of the unique node.
                    printf("plasma_core_dtrmm on: %d %d  = upper(%d %d) x (%d %d) \n", i, j, i, i, i, j);

                }
            }

        }

        i = i - STILES_SIZE; 
    }

    sTiles::Control::Barrier(stile);
    printf("\n");
    printf("\n");
    printf("\n");
    printf("------------------------------------- \n");
    printf("\n");
    printf("\n");
    printf("\n");

    i = num_tiles - 1 - myrank;

    while (i >= 0) {

        for(j = num_tiles - 1; j >= i; j--) {

            if(i==j){

                lauum +=1;
                trtri +=1;

                fprintf(file1, "('%d' , '%d'), \n", opnum[i][i], trtri); //relation between diagonal node of chol and diagonal node of inverse
                fprintf(file2, " '%d': ('LAUUM', '#CDCE27'), \n", trtri); //unique number to the diagonal node, again
                invopnum[i][i] = trtri; //is the LAUUM
                optype[i][i] = 1;

                if((i+1)<=(num_tiles-1)){

                    for(k = i+1; k<=(num_tiles-1); k++){

                        if(on_off_tiles[i*(2*num_tiles-i-1)/2 + k]){

                            in_cond_wait(i, k, 2);

                            printf("((WAIT)): %d %d \n", i, k);
                            // Print the computation step
                            std::cout << "1: INV[" << i << ", " << i << "] = INV[" << i << ", " << i 
                            << "] - M[" << i << ", " << k << "]*INV[" << i << ", " << k << "]" 
                            << std::endl;

                            if(optype[i][i]==1){
                            
                                gemms +=1;
                                fprintf(file1, "('%d' , '%d'), \n", invopnum[i][i], gemms);
                                fprintf(file1, "('%d' , '%d'), \n", invopnum[i][k], gemms);
                                invopnum[i][i] = gemms;
                                optype[i][i] = 2;
                                fprintf(file2, " '%d': ('GEMM', '#2596BE'), \n", invopnum[i][i]);

                            }

                        }

                    }

                    in_cond_set(i, i, 2);
                    printf("((SETT)): %d %d \n", i, i);

                }else{
                
                    in_cond_set(i, i, 2);
                    printf("((SETT)): %d %d \n", i, i);

                }

            }else{

                if((i+1)<=(num_tiles-1)) {setting = true; ii =0; jj =0;}
                for(k = i+1; k<=(num_tiles-1); k++){
                    if(k > j){

                        if(on_off_tiles[i*(2*num_tiles-i-1)/2 + j] && on_off_tiles[j*(2*num_tiles-j-1)/2 + k] && on_off_tiles[i*(2*num_tiles-i-1)/2 + k]){
                            
                            in_cond_wait(j, k, 2);

                            printf("((WAIT)): %d %d \n", j, k);
                            // Print the computation step
                            std::cout << "2: INV[" << i << ", " << j << "] = INV[" << i << ", " << j << "] - M[" << i << ", " << k << "]*INV[" << j << ", " << k << "]" << std::endl;
                            ii = i;
                            jj = j;
                            
                            //if(optype[i][j]==0){

                                gemms +=1;
                                invopnum[i][j] = gemms;
                                fprintf(file2, " '%d': ('GEMM', '#2596BE'), \n", invopnum[i][j]);
                                optype[i][j] = 2;

                            //}
                            fprintf(file1, "('%d' , '%d'), \n", opnum[i][k],    invopnum[i][j]);
                            fprintf(file1, "('%d' , '%d'), \n", invopnum[j][k], invopnum[i][j]);
                        }

                    }else{

                        index1 = magic_perm1[i*(2*num_tiles-i-1)/2 + k];
                        index2 = magic_perm1[k*(2*num_tiles-k-1)/2 + j];
                        index3 = magic_perm1[i*(2*num_tiles-i-1)/2 + j];

                        if(on_off_tiles[i*(2*num_tiles-i-1)/2 + j] && on_off_tiles[k*(2*num_tiles-k-1)/2 + j] && on_off_tiles[i*(2*num_tiles-i-1)/2 + k]){

                            in_cond_wait(k, j, 2);
                            printf("((WAIT)): %d %d \n", k, j);
                            // Print the computation step
                            std::cout << "3: INV[" << i << ", " << j << "] = INV[" << i << ", " << j 
                            << "] - M[" << i << ", " << k << "]*INV[" << k << ", " << j << "]" 
                            << std::endl;

                            ii = i;
                            jj = j;

                            //if(optype[i][j]==0){
                                gemms +=1;
                                invopnum[i][j] = gemms;
                                fprintf(file2, " '%d': ('GEMM', '#2596BE'), \n", invopnum[i][j]);
                                optype[i][j] = 2;
                            //}
                            fprintf(file1, "('%d' , '%d'), \n", opnum[i][k], invopnum[i][j]);
                            fprintf(file1, "('%d' , '%d'), \n", invopnum[k][j], invopnum[i][j]);
                        }
                        

                    }

                    
                }
                if(setting) {
                    
                    in_cond_set(ii, jj, 2);
                    printf("((SETT)): %d %d \n", ii, jj);

                }
                setting = false;


            }
        }

        i = i - worldsize; 

    }




/*
    while (i >= 0) {

        for(j = i; j < A.nt; j++) {

            index1 = A.magic_perm1[i*(2*A.mt-i-1)/2 + i];
            index2 = A.magic_perm1[i*(2*A.mt-i-1)/2 + j]; // i switch it!

            if(i==j){

                //CORE_dtrtri(PlasmaUpper, PlasmaNonUnit, ldak, A.style[index1].elements, ldak, &info); //inverse of the upper A(k,k) and it is put in the lower A(k,k)
                //CORE_dtrtri(PlasmaUpper, PlasmaNonUnit, A.style[index1].width, A.style[index1].elements, A.style[index1].width, &info); //inverse of the upper A(k,k) and it is put in the lower A(k,k)
                //printf("CORE_dtrtri on Tile: %d %d \n", i, i);

                trtri +=1;
                opnum[i][i] = trtri;

                //fprintf(file1, "('%d' , '%d'), \n", trtri, opnum[i][i]);
                fprintf(file2, " '%d': ('TRSM', '#EFB717'), \n", opnum[i][i]);



            }else{
                
                if(A.of_perm[i*(2*A.nt-i-1)/2 + j]){

                    //CORE_dtrmm(
                     //   PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit,
                      //  A.style[index2].height, A.style[index2].width,  // M and N (rows and columns of B)
                      //  zone, A.style[index1].elements, A.style[index1].height,  // A and its leading dimension (LDA)
                       // A.style[index2].elements, A.style[index2].height);       // B and its leading dimension (LDB)
                        
                        //printf("CORE_dtrmm on: %d %d  = upper(%d %d) x (%d %d) \n", i, j, i, i, i, j);

                        trsm +=1;
                        opnum[i][j] = trsm;

                        fprintf(file1, "('%d' , '%d'), \n", opnum[i][i], opnum[i][j]);
                        fprintf(file2, " '%d': ('TRMM', '#F0911B'), \n", opnum[i][j]);

                }
            }

        }

        i = i - PLASMA_SIZE; 
    }
    
    plasma_barrier(plasma);
    printf("\n");
    printf("\n");
    printf("\n");
    printf("------------------------------------- \n");
    printf("\n");
    printf("\n");
    printf("\n");

    i = A.nt - 1 - PLASMA_RANK;

    while (i >= 0) {

        for(j = A.nt - 1; j >= i; j--) {

            if(i==j){

                index1 = A.magic_perm1[i*(2*A.mt-i-1)/2 + i];

                //CORE_dlauum(PlasmaUpper, A.style[index1].width, A.style[index1].elements, A.style[index1].height); //L * L ^T
                //copy_lower_to_upper( A.style[index1].height, A.style[index1].elements, A.inv_style[index1].elements,  A.style[index1].height); //copy the upper of style tile to the upper & lower inv_style
                //printf("fixing CORE_dlauum and copy_lower_to_upper at indicies: %d \n", index1);

                printf("LAUUM(%d %d)  ---->  TRTRI(%d %d)^T \n", i, i, i, i);
                printf("INV(%d %d)    ---->  LAUUM(%d %d)^T \n", i, i, i, i);


                lauum +=1;
                invopnum[i][i] = lauum;
                trtri +=1;


                fprintf(file1, "('%d' , '%d'), \n", opnum[i][i], trtri); //trtri is LAUUM and opnum[i][i] is trsm
                fprintf(file2, " '%d': ('LAUUM', '#CDCE27'), \n", trtri);
                invopnum[i][i] = trtri;
               // fprintf(file1, "('%d' , '%d'), \n", opnum[i][i], invopnum[i][i]);
               // fprintf(file2, " '%d': ('LACPY', '#CDCE27'), \n", invopnum[i][i]); //invopnum[i][i] is 
                optype[i][i] = 1;


                for(k = i+1; k<=(A.nt-1); k++){

                    index2 = A.magic_perm1[i*(2*A.mt-i-1)/2 + k];
                    if(A.of_perm[i*(2*A.nt-i-1)/2 + k]){


                        //printf("11 CORE_dgemm on: INV(%d %d)  = INV(%d %d) - TRMM(%d %d) x INV(%d %d)^T \n", i, i, i, i , i, k, i, k);

                        printf("\n");
                        printf("11 CORE_GEMM \n");
                        printf("INV(%d %d)  ---->  TRMM(%d %d) \n", i, i, i, k);
                        printf("INV(%d %d)  ---->  INV(%d %d) \n", i, i, i,k);
                        printf("\n");


                        if(optype[i][i]==1){
                            
                            gemms +=1;
                            fprintf(file1, "('%d' , '%d'), \n", invopnum[i][i], gemms);

                            invopnum[i][i] = gemms;
                            optype[i][i] = 2;
                            fprintf(file2, " '%d': ('GEMM', '#2596BE'), \n", invopnum[i][i]);

                        }
                       
                        fprintf(file1, "('%d' , '%d'), \n", opnum[i][k], invopnum[i][i]);
                        fprintf(file1, "('%d' , '%d'), \n", invopnum[i][k], invopnum[i][i]);

                    }
                    

                }
                in_cond_set(i, i, 2);
                

                
            }else{

                for(k = i+1; k<=A.nt-1; k++){

                    if(k > j){

                        index1 = A.magic_perm1[i*(2*A.mt-i-1)/2 + k];
                        index2 = A.magic_perm1[j*(2*A.mt-j-1)/2 + k];
                        index3 = A.magic_perm1[i*(2*A.mt-i-1)/2 + j];

                        if(A.of_perm[i*(2*A.nt-i-1)/2 + j] && A.of_perm[j*(2*A.nt-j-1)/2 + k] && A.of_perm[i*(2*A.nt-i-1)/2 + k]){

                            in_cond_wait(j, k, 2);


                        //printf("22 CORE_dgemm on: INV(%d %d)  = INV(%d %d) - S(%d %d) x INV(%d %d)^T \n", i, j, i, j , i, k, j, k);

                        printf("\n");
                        printf("22 CORE_GEMM \n");
                        printf("INV(%d %d)  ---->  TRMM(%d %d) \n", i, j, i, i);
                        printf("INV(%d %d)  ---->  INV(%d %d) \n", i, j, j, k);
                        printf("\n");

                       // if(optype[i][j]==0){

                            gemms +=1;
                            invopnum[i][j] = gemms;
                            fprintf(file2, " '%d': ('GEMM', '#2596BE'), \n", invopnum[i][j]);
                            optype[i][j] = 2;

                       // }



                        fprintf(file1, "('%d' , '%d'), \n", opnum[i][k], invopnum[i][j]);
                        fprintf(file1, "('%d' , '%d'), \n", invopnum[j][k], invopnum[i][j]);

                            in_cond_set(i, k, 2);
                            in_cond_set(i, j, 2);
                        }

                    }else{

                        index1 = A.magic_perm1[i*(2*A.mt-i-1)/2 + k];
                        index2 = A.magic_perm1[k*(2*A.mt-k-1)/2 + j];
                        index3 = A.magic_perm1[i*(2*A.mt-i-1)/2 + j];

                        if(A.of_perm[i*(2*A.nt-i-1)/2 + j] && A.of_perm[k*(2*A.nt-k-1)/2 + j] && A.of_perm[i*(2*A.nt-i-1)/2 + k]){

                            in_cond_wait(k, j, 2);

                        printf("\n");
                        printf("33 CORE_GEMM \n");
                        printf("INV(%d %d)  ---->  TRMM(%d %d) \n", i, j, i, k);
                        printf("INV(%d %d)  ---->  INV(%d %d) \n", i, j, k, j);
                        printf("\n");

                        
                       //  on: INV(%d %d)  = INV(%d %d) - TRMM(%d %d) x INV(%d %d)^T \n", i, j, i, j, i, k, k, j);

                       // if(optype[i][j]==0){

                            gemms +=1;
                            invopnum[i][j] = gemms;
                            fprintf(file2, " '%d': ('GEMM', '#2596BE'), \n", invopnum[i][j]);
                            optype[i][j] = 2;

                        //}

                        fprintf(file1, "('%d' , '%d'), \n", opnum[i][k], invopnum[i][j]);
                        fprintf(file1, "('%d' , '%d'), \n", invopnum[k][j], invopnum[i][j]);


                            in_cond_set(i, k, 2);
                            in_cond_set(i, j, 2);

                        }
                        

                    }
                    
                }
            }
        }

        i = i - PLASMA_SIZE; 
    }
*/


    in_finalize();
    fclose(file1);
    fclose(file2);

    return 1;
}



#endif




