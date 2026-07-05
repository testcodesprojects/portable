#ifndef SPS_ALGO_ETREE_HPP
#define SPS_ALGO_ETREE_HPP

#include <cstdint>
#include <vector>

namespace sTiles { namespace sparse {

using Int = int32_t;
using Idx = int32_t;
using Ptr = int64_t;

// Lower-triangular CSC of a symmetric matrix (1-based colptr/rowind). The
// graph is *not* expanded — each off-diagonal nonzero appears once, in the
// column with the smaller index.
//
// The elimination-tree routines need the *expanded* (full symmetric) graph,
// so callers either supply expanded = true or call CscLower::expand() before
// passing it in.
struct CscLower {
  Int             size = 0;       // n
  bool            expanded = false;
  std::vector<Ptr> colptr;        // size n+1, 1-based
  std::vector<Idx> rowind;        // 1-based row indices
  std::vector<double> nzval;      // optional; aligns with rowind. Empty if
                                  // only the symbolic pattern is needed.

  // Turn lower-only structure into a full symmetric pattern.
  // If `nzval` is populated, mirrored entries get the same value.
  void expand();
};

// Permutation pair, 1-based:
//   perm[i-1] = original index of the i-th node in the new ordering
//   invp[k-1] = new index of original node k
// perm and invp must be inverses of each other.
struct Permutation {
  std::vector<Int> perm;
  std::vector<Int> invp;

  void set_identity(Int n);

  // Validate perm/invp are non-empty, same length, and mutual inverses.
  // Returns true if valid. Used by tests.
  bool validate() const;

  // Relabel this permutation by a second one given as invp2 (1-based):
  //   new invp[i-1] = invp2[ old_invp[i-1] - 1 ]
  // perm is rebuilt to stay the inverse of invp.
  void relabel(const std::vector<Int>& invp2);
};

// Union-find forest with path compression, used while building the
// elimination tree and while relaxing supernodes. Sets are identified by a
// stored representative (Rep) that the caller assigns.
class UnionFind {
 public:
  void reset(Int n);
  void clear();
  Int  make(Int i)            { link_[i - 1] = i; return i; }
  Int  attach(Int s, Int t)   { link_[s - 1] = t; return t; }
  Int  find_root(Int i) {
    Int p  = link_[i - 1];
    Int gp = link_[p - 1];
    while (gp != p) {
      link_[gp - 1] = gp;      // compress: promote grandparent to a local root
      i  = gp;
      p  = link_[i - 1];
      gp = link_[p - 1];
    }
    return p;
  }
  void unite(Int s, Int t, Int rep = -1) {
    Int tRoot = find_root(t);
    Int sRoot = find_root(s);
    sRoot = attach(sRoot, tRoot);
    rep_[sRoot - 1] = (rep == -1) ? t : rep;
  }
  Int& rep(Int i) { return rep_[i - 1]; }

 private:
  std::vector<Int> link_;
  std::vector<Int> rep_;
};

// Elimination tree of the *permuted* matrix P A P^T.
// Single-process implementation.
class EliminationTree {
 public:
  EliminationTree() = default;

  // Build the parent array for the elimination tree of P A P^T, where the
  // symmetric expanded graph is `g` and P is given by `perm`. `g.expanded`
  // must be true.
  void build(const CscLower& g, const Permutation& perm);

  // Post-order the tree, folding the post-order permutation into `perm`.
  // Marks the tree post-ordered and fills the post-ordered parent array.
  // If `relinvp` is non-null, it receives a copy of the post-order invp
  // (n entries, 1-based).
  void postorder(Permutation& perm, Int* relinvp = nullptr);

  // Reorder the children of each node by descending weight `weight`
  // (column count). Folds the resulting permutation into `perm` and updates
  // the post-ordered parent array and `weight`. Post-orders the tree first if
  // it is not yet post-ordered.
  void reorder_children(std::vector<Int>& weight, Permutation& perm);

  // Lift the tree to supernode granularity.
  //   first_col[ksup-1..ksup] defines the column range of supernode ksup.
  //   col_super[c-1]          is the supernode that owns column c.
  // Requires *this is post-ordered.
  EliminationTree supernodal_tree(const std::vector<Int>& first_col,
                                  const std::vector<Int>& col_super) const;

  bool is_postordered() const  { return postordered_; }
  Int  n() const               { return n_; }
  Int  size() const            { return static_cast<Int>(parent_idx_.size()); }
  Int  parent_of(Int i) const      { return parent_idx_[i]; }      // 0-based
  Int  post_parent_of(Int i) const { return post_parent_idx_[i]; } // index in
  const std::vector<Int>& parents() const      { return parent_idx_; }
  const std::vector<Int>& post_parents() const { return post_parent_idx_; }

 private:
  // DFS post-order of the first-child / next-sibling forest. Fills
  // new_pos[0..n-1] with 1-based post-order positions.
  void dfs_postorder(const std::vector<Int>& first_child,
                     const std::vector<Int>& next_sibling,
                     std::vector<Int>& new_pos) const;

  Int              n_ = 0;
  bool             postordered_ = false;
  std::vector<Int> parent_idx_;       // 1-based parents, 0 = root
  std::vector<Int> post_parent_idx_;  // post-ordered parents (1-based, 0 = root)
};

}}  // namespace sTiles::sparse

#endif  // SPS_ALGO_ETREE_HPP
