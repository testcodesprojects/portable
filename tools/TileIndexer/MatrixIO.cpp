/**
 * @file    MatrixIO.cpp
 * @brief   Implementation of loaders for compact CSR-like binaries into
 *          coordinate arrays used by TileIndexer; includes deallocation.
 *
 * @project sTiles (Sparse Tiles Library)
 * @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @version 3.0.0
 * @date 1 1 2026
 * @license Proprietary
 *
 * @note This file is part of the sTiles library, a proprietary software package.
 *       Redistribution or modification without prior permission is prohibited.
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
 *
 * @license
 * This software is proprietary and confidential. Unauthorized copying, distribution, or modification
 * of this software, via any medium, is strictly prohibited. Permission is granted to use the software
 * in binary form for non-commercial purposes only, provided that this copyright notice and permission
 * notice are included in all copies or substantial portions of the software.
 *
 * DISCLAIMER:
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "MatrixIO.hpp"

#include <cstdio>
#include <vector>

namespace tilecounter {

/*
Function: loadMatrixIndices
Purpose:  Load a matrix in CSR-like binary format and convert it into
          coordinate arrays. The function handles headers encoded as doubles
          or ints for robustness with legacy files.
Returns:  true on success; false on failure.
*/
bool loadMatrixIndices(const std::filesystem::path& filename,
                       int& n,
                       int& nnz,
                       int*& row_indices,
                       int*& col_indices)
{
    FILE* file = std::fopen(filename.string().c_str(), "rb");
    if (!file) {
        return false;
    }

    double tmp1 = 0.0;
    double tmp2 = 0.0;
    if (std::fread(&tmp1, sizeof(double), 1, file) != 1 ||
        std::fread(&tmp2, sizeof(double), 1, file) != 1) {
        std::fclose(file);
        return false;
    }

    n   = static_cast<int>(tmp1);
    nnz = static_cast<int>(tmp2);

    if (n == 0) {
        std::rewind(file);
        int tmp1_int = 0;
        int tmp2_int = 0;
        if (std::fread(&tmp1_int, sizeof(int), 1, file) != 1 ||
            std::fread(&tmp2_int, sizeof(int), 1, file) != 1) {
            std::fclose(file);
            return false;
        }
        n   = tmp1_int;
        nnz = tmp2_int;
    }

    if (n <= 0 || nnz < 0) {
        std::fclose(file);
        return false;
    }

    row_indices = new int[static_cast<std::size_t>(nnz)];
    col_indices = new int[static_cast<std::size_t>(nnz)];
    std::vector<int>    csr_p(static_cast<std::size_t>(n) + 1);
    std::vector<double> values(static_cast<std::size_t>(nnz));

    if (std::fread(row_indices, sizeof(int), static_cast<std::size_t>(nnz), file) != static_cast<std::size_t>(nnz) ||
        std::fread(csr_p.data(), sizeof(int), static_cast<std::size_t>(n) + 1, file) != static_cast<std::size_t>(n) + 1 ||
        std::fread(values.data(), sizeof(double), static_cast<std::size_t>(nnz), file) != static_cast<std::size_t>(nnz)) {
        std::fclose(file);
        delete[] row_indices;
        delete[] col_indices;
        row_indices = col_indices = nullptr;
        return false;
    }

    std::fclose(file);

    std::size_t csr_index = 0;
    for (int col = 0; col < n; ++col) {
        const int start = csr_p[static_cast<std::size_t>(col)];
        const int end   = csr_p[static_cast<std::size_t>(col + 1)];
        for (int idx = start; idx < end; ++idx, ++csr_index) {
            row_indices[csr_index] -= 1;
            col_indices[csr_index]  = col;
        }
    }

    return true;
}

/*
Function: freeMatrixIndices
Purpose:  Delete the arrays produced by loadMatrixIndices and set pointers to
          nullptr to avoid dangling references.
*/
void freeMatrixIndices(int*& row_indices, int*& col_indices) {
    delete[] row_indices;
    delete[] col_indices;
    row_indices = col_indices = nullptr;
}

} // namespace tilecounter
