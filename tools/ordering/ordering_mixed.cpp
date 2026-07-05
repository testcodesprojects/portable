
#include <vector>
#include <iostream>
#include "ordering_utils.hpp"
#include "../common/stiles_exporter.hpp"

#include <iostream>
#include <stdexcept>

void checkPermutation(int* perm, size_t N) {
    if (!perm) {
        throw std::invalid_argument("Null pointer provided for permutation array.");
    }

    bool* found = new bool[N](); // Initialize all values to false

    for (size_t i = 0; i < N; ++i) {
        int value = perm[i];
        if (value < 0 || value >= static_cast<int>(N)) {
            delete[] found; // Clean up memory
            throw std::out_of_range("Element out of range in permutation");
        }
        found[value] = true;
    }

    for (size_t i = 0; i < N; ++i) {
        if (!found[i]) {
            delete[] found; // Clean up memory
            std::cout << "-----------------> i: " << i << std::endl;
            throw std::logic_error("Missing element in permutation");
        }
    }

    delete[] found; // Clean up memory
    std::cout << "Permutation is valid.\n";
}


namespace sTiles {

void stiles_runNDRCM(int** indices_i, int** indices_j, int N, int nnz, int m, int** perm, int** iperm, int num_sep, int** sizes) {

    int* save_rows = (int*)malloc(nnz * sizeof(int));
    int* save_cols = (int*)malloc(nnz * sizeof(int));

    for(int i=0; i<nnz;i++){

        save_rows[i] = (*indices_i)[i];
        save_cols[i] = (*indices_j)[i];

    }

    int* perm_nd = (int *)malloc(N * sizeof(int));
    int* iperm_nd = (int *)malloc(N * sizeof(int));

    sTiles::runND(indices_i, indices_j, N, nnz, 0, &perm_nd, &iperm_nd, num_sep, sizes);

    int* my_perm1 = (int *)malloc((*sizes)[0] * sizeof(int));
    int* my_iperm1 = (int *)malloc((*sizes)[0] * sizeof(int));
    int* my_perm2 = (int *)malloc((*sizes)[1] * sizeof(int));
    int* my_iperm2 = (int *)malloc((*sizes)[1] * sizeof(int));

    sTiles::permute_and_swap(&iperm_nd, indices_i, indices_j, nnz);

    std::vector<int> row_indices1;
    std::vector<int> col_indices1;
    std::vector<int> row_indices2;
    std::vector<int> col_indices2;

    row_indices1.reserve(nnz); 
    col_indices1.reserve(nnz); 
    row_indices2.reserve(nnz); 
    col_indices2.reserve(nnz); 

    int n_nnz1 = 0, n_nnz2 = 0, a = 0;
    for (int i = 0; i < nnz; i++) {
        int row = (*indices_i)[i];
        int col = (*indices_j)[i];

        // Check if the entry is within the bounds of the submatrix
        if (row >= 0 && row < (*sizes)[0] && col >= 0 && col < (*sizes)[0]) {
            row_indices1.push_back(row-a);
            col_indices1.push_back(col-a);
            n_nnz1++;
        }else if(row >= (*sizes)[0] && row < ((*sizes)[0]+(*sizes)[1]) && col >= (*sizes)[0] && col < ((*sizes)[0]+(*sizes)[1])) {
            row_indices2.push_back(row-(*sizes)[0]);
            col_indices2.push_back(col-(*sizes)[0]);
            n_nnz2++;
        }

    }

    int* csr_i_data1 = row_indices1.data();
    int* csr_j_data1 = col_indices1.data();
    sTiles::runRCM(&csr_i_data1, &csr_j_data1, (*sizes)[0], n_nnz1, 0, &my_iperm1, &my_perm1, false);

    int counter = 0;
    for(int i=0; i<(*sizes)[0];i++){
        (*perm)[counter] = my_iperm1[i];
        counter++;
    }
    
    int* csr_i_data2 = row_indices2.data();
    int* csr_j_data2 = col_indices2.data();
    sTiles::runRCM(&csr_i_data2, &csr_j_data2, (*sizes)[1], n_nnz2, 0, &my_iperm2, &my_perm2, false);

    for(int i=0; i<(*sizes)[1];i++){
        (*perm)[counter] = my_iperm2[i] + (*sizes)[0];
        counter++;
    }

    for(int i=counter; i< N;i++){
        (*perm)[counter] = i;
        counter++;
    }

    int* newiperm = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        newiperm[i] = perm_nd[(*perm)[i]]; // or (*iperm)[iperm_nd[i]];
    }

    for (size_t i = 0; i < N; ++i) {
        (*perm)[i] = newiperm[i];
    }

    for (size_t i = 0; i < N; ++i) {
        (*iperm)[(*perm)[i]] = i;
    }  

    /*try {
        checkPermutation(*perm, N);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }*/

    for(int i=0; i<nnz;i++){
        (*indices_i)[i] = save_rows[i];
        (*indices_j)[i] = save_cols[i];
    }
    
    free(save_rows);
    free(save_cols);
    free(perm_nd);
    free(iperm_nd);
    free(my_perm1);
    free(my_iperm1);
    free(my_perm2);
    free(my_iperm2);
    free(newiperm);
    
}
}
