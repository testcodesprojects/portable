/**
 * @file    supernode.cpp
 * @brief   Supernode detection, amalgamation and the non-uniform cell layout.
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "supernode.hpp"

#include "../common/stiles_logger.hpp"
#include "../memory/MemoryManager.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

#include <omp.h>

namespace sTiles { namespace sparse {

CellStore::~CellStore() {
    release_arena_();
}

CellStore::CellStore(CellStore&& other) noexcept
        : n_super_   (other.n_super_),
            arena_size_(other.arena_size_),
            arena_     (other.arena_),
            group_id_  (other.group_id_),
            cells_     (std::move(other.cells_)),
            cell_idx_  (std::move(other.cell_idx_)) {
    other.arena_      = nullptr;
    other.arena_size_ = 0;
    other.n_super_    = 0;
}

CellStore& CellStore::operator=(CellStore&& other) noexcept {
    if (this != &other) {
        release_arena_();
        n_super_    = other.n_super_;
        arena_size_ = other.arena_size_;
        arena_      = other.arena_;
        group_id_   = other.group_id_;
        cells_      = std::move(other.cells_);
        cell_idx_   = std::move(other.cell_idx_);
        other.arena_      = nullptr;
        other.arena_size_ = 0;
        other.n_super_    = 0;
    }
    return *this;
}

void CellStore::release_arena_() {
    if (arena_ != nullptr) {
        ::MemoryManager::deallocate(arena_);  // sets arena_ to nullptr
    }
    arena_size_ = 0;
}

void CellStore::allocate(const Symbolic& s, int group_id) {
    n_super_  = s.n_super;
    group_id_ = group_id;
    cells_.clear();
    cell_idx_.clear();

    const double t0 = omp_get_wtime();

    // Pass 1: arena size = sum over column-supernodes I of (row_pattern[I] length) *
    // width(I). row_pattern[I] holds I's row pattern, packed; cells split it by
    // supernode_of_col grouping.
    Ptr new_size = 0;
    for (Int I = 1; I <= n_super_; ++I) {
        Int width_I = s.supernode_first_col[I] - s.supernode_first_col[I - 1];
        Ptr lb      = s.row_pattern_ptr[I - 1];
        Ptr le      = s.row_pattern_ptr[I] - 1;
        new_size += static_cast<Ptr>(le - lb + 1) * width_I;
    }
    const double t1 = omp_get_wtime();

    // Guard: the supernodal layout materializes each (column-supernode I,
    // row-supernode J) block as a dense rows×cols slab. Refuse to allocate
    // beyond a 16 GiB envelope so we get a clean error instead of an OOM.
    constexpr Ptr kMaxArenaDoubles = static_cast<Ptr>(2) * 1024 * 1024 * 1024; // 16 GiB / 8B
    if (new_size > kMaxArenaDoubles) {
        throw std::runtime_error(
            "CellStore::allocate: supernodal arena too large for this matrix "
            "(likely caused by dense rows pushed to end of ordering — this "
            "matrix is unsuitable for tile_type_mode=3)");
    }

    // Free any previous arena, then allocate (zero-initialized) through
    // sTiles::Memory::MemoryManager so the bytes show up in the per-group
    // memory totals and get reaped by sTiles_freeGroup.
    release_arena_();
    arena_size_ = new_size;
    arena_      = ::MemoryManager::allocateZero<double>(static_cast<size_t>(arena_size_), group_id_);
    const double t2 = omp_get_wtime();

    // Pass 2: walk row_pattern[I] and group consecutive entries by their row-
    // supernode (supernode_of_col). Each group becomes one cell with the
    // run-length as its row count. The cell's global row indices are
    // row_pattern[lx_offset .. lx_offset + rows - 1] (sorted ascending).
    Ptr cursor = 0;
    Int total_cells = 0;
    for (Int I = 1; I <= n_super_; ++I) {
        Int width_I = s.supernode_first_col[I] - s.supernode_first_col[I - 1];
        Ptr lb      = s.row_pattern_ptr[I - 1];
        Ptr le      = s.row_pattern_ptr[I] - 1;
        Ptr p       = lb;
        while (p <= le) {
            Int J = s.supernode_of_col[s.row_pattern[p - 1] - 1];
            Ptr q = p;
            while (q <= le && s.supernode_of_col[s.row_pattern[q - 1] - 1] == J) ++q;
            Int run = static_cast<Int>(q - p);

            Cell c;
            c.I         = I;
            c.J         = J;
            c.rows      = run;
            c.cols      = width_I;
            c.lx_offset = p;
            c.nzval     = arena_ + cursor;
            cursor     += static_cast<Ptr>(run) * width_I;

            Int idx = static_cast<Int>(cells_.size());
            cells_.push_back(c);
            int64_t key = static_cast<int64_t>(J - 1)
                                    + static_cast<int64_t>(n_super_) * (I - 1);
            cell_idx_.emplace(key, idx);
            ++total_cells;

            p = q;
        }
    }
    (void)t0;
    (void)t1;
    (void)t2;
    (void)total_cells;

    assert(cursor == arena_size_);
}

void CellStore::load_from_csc(const CscLower& A, const Symbolic& s) {
    if (A.size != s.n) {
        throw std::logic_error(
            "CellStore::load_from_csc: matrix size does not match symbolic.n");
    }
    if (A.nzval.empty()) {
        throw std::logic_error(
            "CellStore::load_from_csc: input CscLower has no nzval");
    }

    for (Int j = 1; j <= A.size; ++j) {
        Ptr jb = A.colptr[j - 1];
        Ptr je = A.colptr[j] - 1;
        for (Ptr k = jb; k <= je; ++k) {
            Idx    i = A.rowind[k - 1];
            double v = A.nzval [k - 1];

            Int new_i   = s.ordering.invp[i - 1];
            Int new_j   = s.ordering.invp[j - 1];
            Int new_row = std::max(new_i, new_j);
            Int new_col = std::min(new_i, new_j);

            if (A.expanded && i < j) continue;  // skip upper-tri duplicate

            Int I = s.supernode_of_col[new_col - 1];
            Int J = s.supernode_of_col[new_row - 1];

            Cell* c = find(J, I);
            if (c == nullptr) {
                throw std::logic_error(
                    "CellStore::load_from_csc: A entry falls outside symbolic pattern");
            }

            Int col_off = new_col - s.supernode_first_col[I - 1];

            // Diagonal cells span their supernode's full range; off-diagonal cells
            // store an arbitrary sorted subset of J's range — look up in row_pattern.
            Int row_off;
            if (c->I == c->J) {
                row_off = new_row - s.supernode_first_col[I - 1];
            } else {
                const Idx* lo = &s.row_pattern[c->lx_offset - 1];
                const Idx* hi = lo + c->rows;
                const Idx* it = std::lower_bound(lo, hi, static_cast<Idx>(new_row));
                if (it == hi || *it != static_cast<Idx>(new_row)) {
                    throw std::logic_error(
                        "CellStore::load_from_csc: row not present in cell's row_pattern slice");
                }
                row_off = static_cast<Int>(it - lo);
            }

            c->nzval[row_off + c->rows * col_off] = v;
        }
    }
}

void CellStore::build_load_map(const CscLower& A, const Symbolic& s,
                               std::vector<Ptr>& map_out) const {
    if (A.size != s.n) {
        throw std::logic_error(
            "CellStore::build_load_map: matrix size does not match symbolic.n");
    }
    // One slot per CSC nonzero position; -1 means "skip" (upper-tri duplicate).
    map_out.assign(A.rowind.size(), static_cast<Ptr>(-1));

    for (Int j = 1; j <= A.size; ++j) {
        Ptr jb = A.colptr[j - 1];
        Ptr je = A.colptr[j] - 1;
        for (Ptr k = jb; k <= je; ++k) {
            Idx i = A.rowind[k - 1];

            Int new_i   = s.ordering.invp[i - 1];
            Int new_j   = s.ordering.invp[j - 1];
            Int new_row = std::max(new_i, new_j);
            Int new_col = std::min(new_i, new_j);

            if (A.expanded && i < j) continue;  // skip upper-tri duplicate (stays -1)

            Int I = s.supernode_of_col[new_col - 1];
            Int J = s.supernode_of_col[new_row - 1];

            const Cell* c = find(J, I);
            if (c == nullptr) {
                throw std::logic_error(
                    "CellStore::build_load_map: A entry falls outside symbolic pattern");
            }

            Int col_off = new_col - s.supernode_first_col[I - 1];
            Int row_off;
            if (c->I == c->J) {
                row_off = new_row - s.supernode_first_col[I - 1];
            } else {
                const Idx* lo = &s.row_pattern[c->lx_offset - 1];
                const Idx* hi = lo + c->rows;
                const Idx* it = std::lower_bound(lo, hi, static_cast<Idx>(new_row));
                if (it == hi || *it != static_cast<Idx>(new_row)) {
                    throw std::logic_error(
                        "CellStore::build_load_map: row not present in cell's row_pattern slice");
                }
                row_off = static_cast<Int>(it - lo);
            }

            // Absolute offset into the arena: (cell base) + (r + rows*c). c->nzval
            // points inside arena_, so the difference is the cell's arena offset.
            map_out[static_cast<std::size_t>(k - 1)] =
                    static_cast<Ptr>(c->nzval - arena_) +
                    static_cast<Ptr>(row_off) +
                    static_cast<Ptr>(c->rows) * static_cast<Ptr>(col_off);
        }
    }
}

void CellStore::load_from_csc_mapped(const CscLower& A,
                                     const std::vector<Ptr>& map) {
    double* base = arena_;
    const std::size_t total = A.nzval.size();
    for (std::size_t p = 0; p < total; ++p) {
        const Ptr d = map[p];
        if (d >= 0) base[static_cast<std::size_t>(d)] = A.nzval[p];
    }
}

void CellStore::allocate_like(const CellStore& other, int group_id) {
    release_arena_();
    n_super_     = other.n_super_;
    arena_size_  = other.arena_size_;
    group_id_    = group_id;
    arena_       = arena_size_ > 0
            ? ::MemoryManager::allocateZero<double>(static_cast<size_t>(arena_size_), group_id_)
            : nullptr;

    cells_       = other.cells_;          // copy I, J, rows, cols, lx_offset
    cell_idx_    = other.cell_idx_;

    // Rebind every nzval pointer to the new arena, preserving the same offset
    // each cell had into `other`'s arena.
    for (auto& c : cells_) {
        Ptr off = c.nzval - other.arena_;
        c.nzval = arena_ + off;
    }
}

Cell* CellStore::find(Int J, Int I) {
    if (J < 1 || J > n_super_ || I < 1 || I > n_super_) return nullptr;
    int64_t key = static_cast<int64_t>(J - 1)
                            + static_cast<int64_t>(n_super_) * (I - 1);
    auto it = cell_idx_.find(key);
    return it == cell_idx_.end() ? nullptr : &cells_[it->second];
}

const Cell* CellStore::find(Int J, Int I) const {
    if (J < 1 || J > n_super_ || I < 1 || I > n_super_) return nullptr;
    int64_t key = static_cast<int64_t>(J - 1)
                            + static_cast<int64_t>(n_super_) * (I - 1);
    auto it = cell_idx_.find(key);
    return it == cell_idx_.end() ? nullptr : &cells_[it->second];
}

}}  // namespace sTiles::sparse
