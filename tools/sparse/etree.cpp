/**
 * @file    etree.cpp
 * @brief   Elimination tree construction and postordering for the non-uniform tile (sparse) module.
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

#include "etree.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace sTiles { namespace sparse {

// ---------------------------------------------------------------------------
// CscLower::expand
// ---------------------------------------------------------------------------
void CscLower::expand() {
    if (expanded) return;
    const Int n = size;
    const bool with_values = !nzval.empty();

    // Pass 1: per-column degree of the full symmetric pattern. Each stored
    // (row i, col j) entry contributes to column j, and its mirror to column i
    // when i != j.
    std::vector<Ptr> degree(n, 0);
    for (Int j = 1; j <= n; ++j) {
        for (Ptr k = colptr[j - 1]; k < colptr[j]; ++k) {
            Idx i = rowind[k - 1];
            ++degree[j - 1];
            if (i != j) ++degree[i - 1];
        }
    }

    // Prefix-sum the degrees into the expanded column pointers.
    std::vector<Ptr> new_colptr(n + 1);
    new_colptr[0] = 1;
    for (Int j = 1; j <= n; ++j) new_colptr[j] = new_colptr[j - 1] + degree[j - 1];

    std::vector<Idx>    new_rowind(new_colptr[n] - 1);
    std::vector<double> new_nzval;
    if (with_values) new_nzval.assign(new_colptr[n] - 1, 0.0);

    // Pass 2: scatter each entry and its mirror, advancing a per-column cursor.
    std::vector<Ptr> cursor = new_colptr;
    for (Int j = 1; j <= n; ++j) {
        for (Ptr k = colptr[j - 1]; k < colptr[j]; ++k) {
            Idx    i = rowind[k - 1];
            double v = with_values ? nzval[k - 1] : 0.0;
            Ptr&   cj = cursor[j - 1];
            new_rowind[cj - 1] = i;
            if (with_values) new_nzval[cj - 1] = v;
            ++cj;
            if (i != j) {
                Ptr& ci = cursor[i - 1];
                new_rowind[ci - 1] = static_cast<Idx>(j);
                if (with_values) new_nzval[ci - 1] = v;
                ++ci;
            }
        }
    }

    // Sort each column's rows ascending, carrying values along if present.
    std::vector<std::pair<Idx, double>> buf;
    for (Int j = 1; j <= n; ++j) {
        Ptr b = new_colptr[j - 1];
        Ptr e = new_colptr[j];
        if (with_values) {
            buf.clear();
            buf.reserve(e - b);
            for (Ptr k = b; k < e; ++k)
                buf.emplace_back(new_rowind[k - 1], new_nzval[k - 1]);
            std::sort(buf.begin(), buf.end(),
                      [](const auto& a, const auto& c) { return a.first < c.first; });
            for (Ptr k = b; k < e; ++k) {
                new_rowind[k - 1] = buf[k - b].first;
                new_nzval [k - 1] = buf[k - b].second;
            }
        } else {
            std::sort(new_rowind.begin() + (b - 1), new_rowind.begin() + (e - 1));
        }
    }

    colptr = std::move(new_colptr);
    rowind = std::move(new_rowind);
    if (with_values) nzval = std::move(new_nzval);
    expanded = true;
}

// ---------------------------------------------------------------------------
// Permutation
// ---------------------------------------------------------------------------
void Permutation::set_identity(Int n) {
    perm.resize(n);
    invp.resize(n);
    for (Int i = 1; i <= n; ++i) { perm[i - 1] = i; invp[i - 1] = i; }
}

bool Permutation::validate() const {
    if (perm.size() != invp.size() || perm.empty()) return false;
    const Int n = static_cast<Int>(perm.size());
    for (Int i = 1; i <= n; ++i) {
        Int p = perm[i - 1];
        if (p < 1 || p > n) return false;
        if (invp[p - 1] != i) return false;
    }
    return true;
}

void Permutation::relabel(const std::vector<Int>& invp2) {
    const Int n = static_cast<Int>(invp.size());
    assert(static_cast<Int>(invp2.size()) == n);
    for (Int i = 1; i <= n; ++i) invp[i - 1] = invp2[invp[i - 1] - 1];
    for (Int i = 1; i <= n; ++i) perm[invp[i - 1] - 1] = i;
}

// ---------------------------------------------------------------------------
// UnionFind
// ---------------------------------------------------------------------------
void UnionFind::reset(Int n) {
    link_.assign(n, 0);
    rep_.assign(n, 0);
}

void UnionFind::clear() {
    link_.clear();
    rep_.clear();
}

// ---------------------------------------------------------------------------
// EliminationTree
// ---------------------------------------------------------------------------
void EliminationTree::build(const CscLower& g, const Permutation& perm) {
    if (!g.expanded) {
        throw std::logic_error(
            "EliminationTree::build: graph must be expanded (full symmetric pattern)");
    }
    if (!perm.validate()) {
        throw std::logic_error(
            "EliminationTree::build: permutation perm/invp invalid");
    }
    if (static_cast<Int>(perm.perm.size()) != g.size) {
        throw std::logic_error(
            "EliminationTree::build: permutation size does not match graph size");
    }

    postordered_ = false;
    n_           = g.size;
    parent_idx_.assign(n_, 0);

    // Liu's elimination-tree algorithm with path compression. Process vertices
    // in the new order 1..n. For vertex i = invp(node), inspect every neighbour
    // of `node` in the unpermuted graph; map it to the new order; for neighbours
    // strictly below i, walk the virtual-ancestor chain up to vertex i, marking
    // the chain as we go, and hang the chain's top off i.
    std::vector<Int> ancestor(n_, 0);

    for (Int i = 1; i <= n_; ++i) {
        parent_idx_[i - 1] = 0;
        ancestor[i - 1]    = 0;

        const Int node  = perm.perm[i - 1];          // 1-based original column
        const Ptr begin = g.colptr[node - 1];
        const Ptr end   = g.colptr[node];

        for (Ptr e = begin; e < end; ++e) {
            Int v = perm.invp[g.rowind[e - 1] - 1];    // neighbour in new order
            if (v >= i) continue;                       // only earlier vertices (skips diag)

            // Climb the ancestor chain, compressing it onto i, until we reach a
            // vertex already tied to i or an unattached root.
            while (ancestor[v - 1] != 0 && ancestor[v - 1] != i) {
                Int up        = ancestor[v - 1];
                ancestor[v - 1] = i;
                v             = up;
            }
            if (ancestor[v - 1] == 0) {                 // unattached root -> child of i
                ancestor[v - 1]   = i;
                parent_idx_[v - 1] = i;
            }
        }
    }
}

void EliminationTree::dfs_postorder(const std::vector<Int>& first_child,
                                    const std::vector<Int>& next_sibling,
                                    std::vector<Int>& new_pos) const {
    new_pos.assign(n_, 0);
    std::vector<Int> stack(n_);

    Int top   = 0;
    Int v     = n_;          // highest-numbered node is always a tree root
    Int count = 0;

    while (count < n_) {
        // Descend to the deepest first-child, stacking the path.
        while (v > 0) {
            stack[top++] = v;
            v = first_child[v - 1];
        }
        // Unwind, numbering nodes in post-order and stepping to siblings.
        while (v == 0) {
            if (top <= 0) return;
            v = stack[--top];
            new_pos[v - 1] = ++count;
            v = next_sibling[v - 1];
        }
    }
}

void EliminationTree::postorder(Permutation& perm, Int* relinvp) {
    if (n_ == 0 || postordered_) return;

    // Build the first-child / next-sibling forest. Iterating vertices top-down
    // and pushing each onto the front of its parent's child list leaves the
    // children in ascending order; roots are chained together likewise.
    std::vector<Int> first_child(n_, 0);
    std::vector<Int> next_sibling(n_, 0);

    Int root_head = n_;
    for (Int v = n_ - 1; v >= 1; --v) {
        Int p = parent_idx_[v - 1];
        if (p == 0 || p == v) {
            next_sibling[root_head - 1] = v;            // extend the root chain
            root_head = v;
        } else {
            next_sibling[v - 1]  = first_child[p - 1];  // push v to front of p's kids
            first_child[p - 1]   = v;
        }
    }

    std::vector<Int> new_pos;
    dfs_postorder(first_child, next_sibling, new_pos);

    // Recompute parents in the post-ordered numbering.
    post_parent_idx_.assign(n_, 0);
    for (Int i = 1; i <= n_; ++i) {
        Int p = parent_idx_[i - 1];
        if (p > 0) p = new_pos[p - 1];
        post_parent_idx_[new_pos[i - 1] - 1] = p;
    }

    perm.relabel(new_pos);
    if (relinvp != nullptr) std::copy(new_pos.begin(), new_pos.end(), relinvp);

    postordered_ = true;
}

void EliminationTree::reorder_children(std::vector<Int>& weight,
                                       Permutation& perm) {
    if (!postordered_) postorder(perm);

    // Build child lists keyed on the post-ordered parents, inserting each child
    // so that heavier subtrees (larger weight) precede lighter ones.
    std::vector<Int> first_child(n_, 0);
    std::vector<Int> next_sibling(n_, 0);
    std::vector<Int> last_child(n_, 0);

    Int root_head = n_;
    for (Int v = n_ - 1; v >= 1; --v) {
        Int p = post_parent_idx_[v - 1];
        if (p == 0 || p == v) {
            next_sibling[root_head - 1] = v;
            root_head = v;
            continue;
        }
        Int tail = last_child[p - 1];
        if (tail == 0) {                              // first child seen for p
            first_child[p - 1] = v;
            last_child[p - 1]  = v;
        } else if (weight[v - 1] >= weight[tail - 1]) {
            next_sibling[v - 1] = first_child[p - 1];   // heavy -> front
            first_child[p - 1]  = v;
        } else {
            next_sibling[tail - 1] = v;                 // light -> back
            last_child[p - 1]      = v;
        }
    }
    next_sibling[root_head - 1] = 0;

    std::vector<Int> new_pos;
    dfs_postorder(first_child, next_sibling, new_pos);

    // Re-express the post-ordered parents under the new numbering.
    std::vector<Int> new_post_parent(n_, 0);
    for (Int i = 1; i <= n_; ++i) {
        Int p = post_parent_idx_[i - 1];
        if (p > 0) p = new_pos[p - 1];
        new_post_parent[new_pos[i - 1] - 1] = p;
    }
    post_parent_idx_ = std::move(new_post_parent);

    // Permute the weight array to match.
    std::vector<Int> new_weight(n_);
    for (Int node = 1; node <= n_; ++node) new_weight[new_pos[node - 1] - 1] = weight[node - 1];
    weight = std::move(new_weight);

    perm.relabel(new_pos);
}

EliminationTree EliminationTree::supernodal_tree(
    const std::vector<Int>& first_col,
    const std::vector<Int>& col_super) const {
    if (!postordered_) {
        throw std::logic_error(
            "EliminationTree::supernodal_tree requires a post-ordered tree");
    }
    EliminationTree out;
    out.n_           = static_cast<Int>(first_col.size()) - 1;
    out.postordered_ = true;
    out.parent_idx_.assign(out.n_, 0);
    for (Int s = 1; s <= out.n_; ++s) {
        Int last_col    = first_col[s] - 1;           // last column of supernode s
        Int parent_col  = post_parent_of(last_col - 1);
        out.parent_idx_[s - 1] = (parent_col == 0) ? 0 : col_super[parent_col - 1];
    }
    out.post_parent_idx_ = out.parent_idx_;
    return out;
}

}}  // namespace sTiles::sparse
