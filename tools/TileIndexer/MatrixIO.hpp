/**
 * @file    MatrixIO.hpp
 * @brief   Declarations for loading sparse matrices used by TileIndexer tests
 *          and tools (coordinate arrays from a compact CSR-like binary format).
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

#pragma once

#include <filesystem>

namespace tilecounter {

/**
 * @brief  Load a matrix from a binary file into coordinate form (row/col index arrays).
 *         The file may encode sizes as double or int, followed by CSR-style row indices,
 *         column pointers, and values.
 * @param[in]  filename     Path to the .bin file.
 * @param[out] n            Dimension (rows/cols).
 * @param[out] nnz          Number of nonzeros.
 * @param[out] row_indices  Allocated array of row indices (size nnz).
 * @param[out] col_indices  Allocated array of column indices (size nnz).
 * @return true on success; false on I/O or format errors.
 */
bool loadMatrixIndices(const std::filesystem::path& filename,
                       int& n,
                       int& nnz,
                       int*& row_indices,
                       int*& col_indices);

/**
 * @brief Free arrays allocated by loadMatrixIndices and null the pointers.
 * @param[in,out] row_indices  Row index buffer to delete and null.
 * @param[in,out] col_indices  Column index buffer to delete and null.
 */
void freeMatrixIndices(int*& row_indices, int*& col_indices);

} // namespace tilecounter
