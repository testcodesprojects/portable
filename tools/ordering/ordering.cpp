// C ABI implementations and thin wrappers for ordering utilities
#include "ordering_utils.hpp"

#include <cstdlib>

extern "C" bool stiles_check_inverse_permutation(int* perm, int* iperm, int N)
{
    return sTiles::check_inverse_permutation(perm, iperm, N);
}

extern "C" void sTiles_to_ordering(double* x, double* ordered_x, int* perm, int n, int m)
{
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            ordered_x[(j * n) + perm[i]] = x[(j * n) + i];
        }
    }
}

extern "C" void sTiles_from_ordering(double* x, double* ordered_x, int* perm, int n, int m)
{
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            x[(j * n) + i] = ordered_x[(j * n) + perm[i]];
        }
    }
}

extern "C" void stiles_permute_and_swap(int** perm, int** row_indices, int** col_indices, int nnz)
{
    sTiles::permute_and_swap(perm, row_indices, col_indices, nnz);
}

extern "C" void process_tiles(int** indices_i, int** indices_j, int** my_perm,
                               int tile_size, int* total_num_used_tiles, int* nnz,
                               bool* of_perm, int* counted_tiles, int* fix_counted_tiles)
{
    int tileRow, tileCol;
    *counted_tiles = 0;

    for (int index = 0; index < (*nnz); ++index) {
        tileRow = (*my_perm)[(*indices_j)[index]] / tile_size;
        tileCol = (*my_perm)[(*indices_i)[index]] / tile_size;

        if (tileRow <= tileCol) {
            const int idx = tileRow * (2 * (*total_num_used_tiles) - tileRow - 1) / 2 + tileCol;
            if (!of_perm[idx]) { of_perm[idx] = true; ++(*counted_tiles); }
        } else {
            const int idx = tileCol * (2 * (*total_num_used_tiles) - tileCol - 1) / 2 + tileRow;
            if (!of_perm[idx]) { of_perm[idx] = true; ++(*counted_tiles); }
        }
    }

    *fix_counted_tiles = *counted_tiles;
    bool sumT = true;
    for (int i = 0; i < (*total_num_used_tiles); ++i) {
        for (int j = 0; j < i; ++j) {
            sumT = false;
            for (int k = 0; k < j; ++k) {
                const int idx_ki = k * (2 * (*total_num_used_tiles) - k - 1) / 2 + i;
                const int idx_kj = k * (2 * (*total_num_used_tiles) - k - 1) / 2 + j;
                if (of_perm[idx_ki] && of_perm[idx_kj]) { sumT = true; break; }
            }
            const int idx_ji = j * (2 * (*total_num_used_tiles) - j - 1) / 2 + i;
            if (sumT && !of_perm[idx_ji]) { of_perm[idx_ji] = true; ++(*counted_tiles); }
        }
    }
}
