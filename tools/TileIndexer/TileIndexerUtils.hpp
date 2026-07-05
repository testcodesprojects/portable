/**
 * @file    TileIndexerUtils.hpp
 * @brief   Lightweight utilities used by TileIndexer for neighbor lookup and
 *          compact bitset operations on 2D layouts.
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
 *
 * These helpers provide:
 * - A simple binary search over a half‑open range [start, end) of a sorted int array.
 * - Neighbor checks using a CSR‑style adjacency (neighbors + neighbor_offsets).
 * - Bit get/set helpers for a 2D boolean grid stored as a packed bit array.
 *
 * Notes on conventions:
 * - All index ranges use half‑open intervals [start, end).
 * - The adjacency offsets array must have length N+1, where the slice for node i is
 *   neighbors[ neighbor_offsets[i] .. neighbor_offsets[i+1] ).
 */

#pragma once

#include <cstddef> // size_t

namespace tilecounter {


/**
 * @brief Binary search in a sorted array over [start, end).
 * @param array  Pointer to a sorted ascending array of ints.
 * @param start  Inclusive start index (0-based).
 * @param end    Exclusive end index (0-based).
 * @param target Value to search for.
 * @return true if found in the range, false otherwise.
 */
inline bool binarySearch(const int* array, int start, int end, int target) {
    while (start < end) {
        int mid = start + (end - start) / 2;
        if (array[mid] == target) return true;
        else if (array[mid] < target) start = mid + 1;
        else end = mid;
    }
    return false;
}

/**
 * @brief Check adjacency using CSR-like neighbor arrays.
 * @param node1             Row/node id (0-based).
 * @param node2             Column/node id (0-based).
 * @param neighbors         Concatenated neighbor list.
 * @param neighbor_offsets  CSR offsets of length N+1. Range for node i is
 *                          neighbors[off[i]..off[i+1]).
 * @return true if node2 appears in node1's neighbor slice.
 */
inline bool isNeighbor(int node1, int node2, const int* neighbors, const int* neighbor_offsets) {
    int start = neighbor_offsets[node1];
    int end = neighbor_offsets[node1 + 1];
    return binarySearch(neighbors, start, end, node2);
}

/**
 * @brief Test a bit in a packed 2D bit array.
 * @param bit_array Packed bit storage (row-major, 1 bit per entry).
 * @param row       Row index (0-based).
 * @param col       Column index (0-based).
 * @param num_cols  Number of columns in the logical 2D grid.
 * @return true if the bit is set, false otherwise.
 */
inline bool isBitSet(const unsigned char* bit_array, size_t row, size_t col, size_t num_cols) {
    size_t index = row * num_cols + col; // Calculate 1D index
    return ((bit_array[index / 8] >> (index % 8)) & 1u) != 0u; // Check the specific bit
}

/**
 * @brief Set a bit in a packed 2D bit array.
 * @param bit_array Packed bit storage (row-major, 1 bit per entry).
 * @param row       Row index (0-based).
 * @param col       Column index (0-based).
 * @param num_cols  Number of columns in the logical 2D grid.
 */
inline void setBit(unsigned char* bit_array, size_t row, size_t col, size_t num_cols) {
    size_t index = row * num_cols + col; // Calculate 1D index
    bit_array[index / 8] |= static_cast<unsigned char>(1u << (index % 8)); // Set the specific bit
}

/**
 * @brief Test neighbor relation encoded by a packed 2D bit array.
 * @param bit_array Packed bit storage (row-major, 1 bit per entry).
 * @param row       Row index (0-based).
 * @param col       Column index (0-based).
 * @param num_cols  Number of columns in the logical 2D grid.
 * @return 1 if set, 0 otherwise.
 */
inline int checkBitNeighbor(const unsigned char *bit_array, size_t row, size_t col, size_t num_cols) {

    size_t index = row * num_cols + col; // Calculate 1D index
    return static_cast<int>((bit_array[index / 8] >> (index % 8)) & 1u); // Check the bit

}

/**
 * @brief Set neighbor relation in a packed 2D bit array.
 * @param bit_array Packed bit storage (row-major, 1 bit per entry).
 * @param row       Row index (0-based).
 * @param col       Column index (0-based).
 * @param num_cols  Number of columns in the logical 2D grid.
 */
inline void setBitNeighbor(unsigned char *bit_array, size_t row, size_t col, size_t num_cols) { //, int is_neighbor) {

    size_t index = row * num_cols + col; // Calculate 1D index
    bit_array[index / 8] |= static_cast<unsigned char>(1u << (index % 8));

    /*if (is_neighbor) { // 1
        bit_array[index / 8] |= (1 << (index % 8));
    } else { // 0
        bit_array[index / 8] &= ~(1 << (index % 8));
    }*/

}

} 
