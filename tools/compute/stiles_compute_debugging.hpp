/**
 * @file stiles_compute_debugging.hpp
 * @brief Internal debugging routines for compute operations.
 *
 * Provides debugging and validation functions for Cholesky factorization
 * including reference implementations and comparison utilities.
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



void compute_chol_plasma_lower(double* A, int dim, SparseTile *tile_obj) {

    int k, m, n;
    int next_k;
    int next_m;
    int next_n;
    int counter = 0;
    double logdet = 0.0;

    sTile_sparse_tile* tile = tile_obj->getTile();  // <-- Correct access
    int* adjacency_flat = tile->adjacency_flat;
    int* adjacency_index = tile->adjacency_index;
    int* adjacency_flat_lower = tile->adjacency_flat_lower;
    int* adjacency_index_lower = tile->adjacency_index_lower;
    int neighbor_index_k, num_neighbors_k, start_k;
    int neighbor_index_n, num_neighbors_n, start_n;



    k = 0;
    m = 0;
    while (m >= dim) {
        k++;
        m = m-dim+k;
    }

    start_k = adjacency_index[k];
    num_neighbors_k = adjacency_flat[start_k-1];  
    neighbor_index_k = start_k;

    start_n = adjacency_index_lower[k];
    num_neighbors_n = adjacency_flat_lower[start_n-1];  
    neighbor_index_n = start_n;

    n = adjacency_flat_lower[neighbor_index_n];
    while (k < dim && m < dim) {

        next_n = n;
        next_m = m;
        next_k = k;

        bool good_k = true;

        neighbor_index_n++;
        if(neighbor_index_n==(num_neighbors_n+start_n)) next_n = next_k+1;
        else next_n = adjacency_flat_lower[neighbor_index_n];
        
        if (next_n > next_k) {
            neighbor_index_k++;
            if(neighbor_index_k>=(num_neighbors_k+start_k)) {good_k = false; neighbor_index_k = start_k;}
            else{
                next_m = adjacency_flat[neighbor_index_k]; //neighbors_k[neighbor_index_k]; neighbor_index_k++
            }

            while (!good_k && next_k < dim) {                
                next_k++;
                if(next_k < dim){
                    start_k = adjacency_index[next_k];
                    num_neighbors_k = adjacency_flat[start_k-1];  
                    neighbor_index_k = start_k;
                    next_m = adjacency_flat[neighbor_index_k]; //next_m - dim + next_k;
                }
                good_k = true;
            }

            if(next_k < dim){
                start_n = adjacency_index_lower[next_k];
                num_neighbors_n = adjacency_flat_lower[start_n-1];  
                neighbor_index_n = start_n;
                next_n = adjacency_flat_lower[neighbor_index_n];
            }
        }

        //std::cout << "                                                   k = " << k << ", m = " << m << ", n = " << n << "\n";

        if (m == k) {
            if (n == k) {
                
                //std::cout << "[DPOTRF] Computing sqrt for A(" << k << "," << k << ") = " << A[k * dim + k] << std::endl;
                if (A[k * dim + k] <= 0.0) {
                    std::cerr << "Cholesky failed: non-positive pivot at (" << k << "," << k << ") = " << A[k * dim + k] << "\n";
                    A[k * dim + k] = 0.0; // or return/fail
                } else {
                    A[k * dim + k] = sqrt(A[k * dim + k]);
                }
                logdet += std::log(A[k * dim + k]);

            }
            else {

                //std::cout << "[DSYRK] Updating A(" << k << "," << k << ") -= A(" << n << "," << k << ")^2 = " << A[n * dim + k] << "^2" << std::endl;
                //if(on_off_tiles[n*(2*dim-n-1)/2 + k])
                A[k * dim + k] -= A[(n * dim) + k] * A[n * dim + k];

                
            }
        }
        else {
            if (n == k) {

                //std::cout << "[DTRSM] Dividing A(" << k << "," << m << ") /= A(" << k << "," << k << ") = " << A[m * dim + k] << " / " << A[k * dim + k] << std::endl;
                A[m + (dim*k)] /= A[k * dim + k];
               // if(on_off_tiles[k*(2*dim-k-1)/2 + m])

                
            }
            else {

                //std::cout << "[DGEMM] Updating A(" << k << "," << m << ") -= A(" << n << "," << m << ") * A(" << n << "," << k << ")" << " = " << A[m * dim + n] << " * " << A[k * dim + n] << std::endl;
                A[m + (dim*k)] -= A[m + (dim*n)] * A[k + (dim*n)];
               // if(on_off_tiles[n*(2*dim-n-1)/2 + k] && on_off_tiles[n*(2*dim-n-1)/2 + m])


                
            }
        }

        
        n = next_n;
        m = next_m;
        k = next_k;

    }
    std::cout << "\nLog determinant = " << std::fixed << std::setprecision(6) << 2.0 * logdet << std::endl;

}


#endif




