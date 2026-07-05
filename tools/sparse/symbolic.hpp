#ifndef SPS_ALGO_SYMBOLIC_HPP
#define SPS_ALGO_SYMBOLIC_HPP

#include "etree.hpp"

#include <cstdint>
#include <vector>

namespace sTiles { namespace sparse {

// Knobs controlling supernode formation and relaxation.
struct SymbolicOptions {
  // Hard cap on supernode width passed to the supernode detector. 0 = unbounded.
  Int relax_max_size = 0;

  // Relaxation thresholds. Setting nrelax0 = 0 disables all merging.
  Int    nrelax0 = 8;
  Int    nrelax1 = 32;
  Int    nrelax2 = 64;
  double zrelax0 = 0.8;
  double zrelax1 = 0.1;
  double zrelax2 = 0.05;
};

// Output of compute_symbolic. All index arrays are 1-based.
//
//   ordering : caller's permutation composed with post-order and
//              child-reordering composition. Use this when permuting nzval.
//   etree    : column-level etree of the permuted matrix, post-ordered.
//   sn_etree : supernode-level etree (also post-ordered).
//   supernode_first_col   : supernode column ranges, length n_super + 1.
//              Supernode k owns columns [supernode_first_col[k-1] .. supernode_first_col[k]-1].
//   supernode_of_col : 1-based supernode index for each column (length n).
//   row_pattern_ptr, row_pattern : packed row pattern of L by supernode.
//                   Supernode k's pattern = row_pattern[row_pattern_ptr[k-1]-1 .. row_pattern_ptr[k]-2].
//                   The first entry is always supernode_first_col[k-1] (the supernode's
//                   first column).
//   col_count       : post-relaxation column counts, length n. col_count[supernode_first_col[k-1]-1] is
//              the row count of supernode k.
//   nnz_l    : sum of col_count[i-1] over i = 1..n. The total nnz in L (counting
//              every supernode column, not just the leading column).
//   row_pattern_len   : row_pattern.size(), i.e. the supernode-packed row count.
struct Symbolic {
  Int n       = 0;
  Int n_super = 0;
  Permutation      ordering;
  EliminationTree  etree;
  EliminationTree  sn_etree;
  std::vector<Int> supernode_first_col;
  std::vector<Int> supernode_of_col;
  std::vector<Ptr> row_pattern_ptr;
  std::vector<Idx> row_pattern;
  std::vector<Int> col_count;
  Ptr              nnz_l  = 0;
  Ptr              row_pattern_len = 0;
};

// Run the full symbolic factorization. `A_lower` is the lower-triangular CSC
// of the symmetric input (`A_lower.expanded` may be true or false; if false,
// it is expanded internally). `user_perm` is the user-supplied ordering;
// after this call, `out.ordering` is `user_perm` further composed with
// internal post-order + child-reordering permutations.
//
// Throws std::logic_error on contract violations (non-square, perm size
// mismatch, invalid perm/invp).
void compute_symbolic(const CscLower&        A_lower,
                      const Permutation&     user_perm,
                      const SymbolicOptions& opts,
                      Symbolic&              out);

}}  // namespace sTiles::sparse

#endif  // SPS_ALGO_SYMBOLIC_HPP
