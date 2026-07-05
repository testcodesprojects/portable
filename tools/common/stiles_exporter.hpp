/**
 * @file stiles_exporter.hpp
 * @brief File export utilities for debugging and data output.
 *
 * Provides helper functions for exporting vectors and matrices to files
 * for debugging, validation, and external analysis purposes.
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

#ifndef STILES_EXPORTER_H
#define STILES_EXPORTER_H

#include <stdio.h>
#include <stdlib.h>

static inline void export_to_file(int** vector, int size, const char* filename) {
    
    // stiles_permute_and_swap(&newperm, csr_i, csr_j, nnz);
    // export_to_file(csr_i, nnz, "/home/abdulfe/sTiles/stiles_1.0.0/csr_i.txt");
    // export_to_file(csr_j, nnz, "/home/abdulfe/sTiles/stiles_1.0.0/csr_j.txt");

    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Failed to open file for writing");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < size; i++) {
        fprintf(file, "%d\n", (*vector)[i]);  // Write each element on a new line
    }
    
    fclose(file);
    printf("Vector exported to %s successfully.\n", filename);
}


#endif // STILES_PROCESS_H
