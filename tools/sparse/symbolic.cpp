#include "symbolic.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace {
inline bool sps_verbose() {
  const char* e = std::getenv("SPS_SYMBOLIC_VERBOSE");
  return e && e[0] != '0';
}
}

namespace sTiles { namespace sparse {

// ---------------------------------------------------------------------------
// Helpers — all internal to this translation unit.
// ---------------------------------------------------------------------------
namespace {

// Permute an *expanded* (full symmetric) graph by `p`:
//   new column k = original column p.perm[k-1],
//   new row idx  = p.invp[old row idx - 1].
// Each new column's row indices are sorted ascending.
void permute_expanded(const CscLower& g, const Permutation& p, CscLower& out) {
  assert(g.expanded);
  const Int n = g.size;
  out.size     = n;
  out.expanded = true;

  out.colptr.assign(n + 1, 1);
  for (Int k = 1; k <= n; ++k) {
    Int src = p.perm[k - 1];
    out.colptr[k] = out.colptr[k - 1] + (g.colptr[src] - g.colptr[src - 1]);
  }

  out.rowind.resize(out.colptr[n] - 1);
  for (Int k = 1; k <= n; ++k) {
    Int src = p.perm[k - 1];
    Ptr dst = out.colptr[k - 1];
    for (Ptr e = g.colptr[src - 1]; e < g.colptr[src]; ++e)
      out.rowind[dst++ - 1] = p.invp[g.rowind[e - 1] - 1];
    std::sort(out.rowind.begin() + (out.colptr[k - 1] - 1),
              out.rowind.begin() + (out.colptr[k]     - 1));
  }
}

// Column counts of L via the elimination-tree / least-common-ancestor method
// (Gilbert, Ng & Peyton 1994; see George & Liu 1981). For each column u in
// post-order we scan its higher-numbered neighbours in A; every such neighbour
// w whose row-subtree u newly enters contributes to the counts, and the skeleton
// of the subtree is tracked with a path-compressing LCA forest `anc`.
//
// Caller guarantees: `g` is the original (unpermuted) expanded graph, `tree`
// is post-ordered, and `p.perm[k-1]` is the original column with new index k.
void column_counts(const CscLower&        g,
                   const Permutation&     p,
                   const EliminationTree& tree,
                   std::vector<Int>&      col_count,
                   std::vector<Int>&      rc) {
  assert(g.expanded);
  assert(tree.is_postordered());
  const Int n = g.size;

  col_count.assign(n, 0);
  rc.assign(n, 0);

  std::vector<Int> depth(n + 1, 0);    // depth in the etree (root at depth 1)
  std::vector<Int> wt(n + 1, 1);       // running partial column weights
  std::vector<Int> first_desc(n + 1);  // lowest post-order descendant of a node
  std::vector<Int> nchild(n + 1, 0);
  std::vector<Int> anc(n, 0);          // LCA forest (1-based logical)
  std::vector<Int> prev_leaf(n, 0);    // previous leaf seen in each row-subtree
  std::vector<Int> prev_nbr(n, 0);     // previous lower-neighbour in each row

  first_desc[0] = 0;
  for (Int u = n; u >= 1; --u) {
    rc[u - 1]     = 1;
    anc[u - 1]    = u;
    depth[u]      = depth[tree.post_parent_of(u - 1)] + 1;
    first_desc[u] = u;
  }
  // A node with any child gets weight 0; leaves keep weight 1. Also propagate
  // the first-descendant of each child up to its parent.
  for (Int u = 1; u < n; ++u) {
    Int par = tree.post_parent_of(u - 1);
    wt[par] = 0;
    ++nchild[par];
    if (first_desc[u] < first_desc[par]) first_desc[par] = first_desc[u];
  }

  Int anchor = 1;
  for (Int u = 1; u <= n; ++u) {
    bool is_leaf_somewhere = false;
    const Int fd   = first_desc[u];
    const Int orig = p.perm[u - 1];

    for (Ptr e = g.colptr[orig - 1]; e < g.colptr[orig]; ++e) {
      Int w = p.invp[g.rowind[e - 1] - 1];
      if (w <= u) continue;                       // only higher neighbours

      if (fd > prev_nbr[w - 1]) {                 // u is a leaf of w's subtree
        ++wt[u];
        Int pl = prev_leaf[w - 1];
        if (pl == 0) {
          rc[w - 1] += depth[u] - depth[w];
        } else {
          // Path-compressing climb to the LCA of pl and u in the forest.
          Int q   = pl;
          Int qa  = anc[q - 1];
          Int lca = anc[qa - 1];
          while (lca != qa) {
            anc[q - 1] = lca;
            q   = lca;
            qa  = anc[q - 1];
            lca = anc[qa - 1];
          }
          rc[w - 1] += depth[u] - depth[lca];
          --wt[lca];
        }
        prev_leaf[w - 1]  = u;
        is_leaf_somewhere = true;
      }
      prev_nbr[w - 1] = u;
    }

    Int par = tree.post_parent_of(u - 1);
    --wt[par];
    if (is_leaf_somewhere || nchild[u] >= 2) anchor = u;
    anc[anchor - 1] = par;                         // splice the chain to parent
  }

  // Accumulate partial weights up the tree into final column counts.
  for (Int u = 1; u <= n; ++u) {
    Int s = col_count[u - 1] + wt[u];
    col_count[u - 1] = s;
    Int par = tree.post_parent_of(u - 1);
    if (par != 0) col_count[par - 1] += s;
  }
}

// Fundamental supernode detection from the post-ordered elimination tree and
// column counts (Liu, Ng & Peyton 1993). Column j joins the preceding column's
// supernode iff j-1 is j's tree-child and the counts step down by exactly one,
// optionally bounded by a maximum width.
void find_supernodes(const EliminationTree&  tree,
                     const std::vector<Int>& col_count,
                     std::vector<Int>&       col_super,
                     std::vector<Int>&       first_col,
                     Int                     max_width) {
  const Int n = tree.n();
  col_super.assign(n, 0);

  Int num_super = 1;
  Int width     = 1;
  col_super[0]  = 1;

  for (Int j = 2; j <= n; ++j) {
    const bool extend = (tree.post_parent_of(j - 2) == j)
                     && (col_count[j - 2] == col_count[j - 1] + 1)
                     && (max_width == 0 || width < max_width);
    if (extend) {
      ++width;
      col_super[j - 1] = num_super;
    } else {
      ++num_super;
      width = 1;
      col_super[j - 1] = num_super;
    }
  }

  // Derive the first-column boundary of every supernode from membership.
  first_col.assign(num_super + 1, 0);
  Int prev = num_super + 1;
  for (Int j = n; j >= 1; --j) {
    Int s = col_super[j - 1];
    if (s != prev) first_col[prev - 1] = j + 1;
    prev = s;
  }
  first_col[0] = 1;
}

// Decide whether a child supernode (width `cw`, fill `child_fill`) should merge
// into its parent (width `pw`, fill `parent_fill`), accumulating the extra
// explicit zeros introduced by the merge into `zeros`. Mirrors the relaxed
// supernode-amalgamation heuristic of Ashcraft & Grimes (1989).
bool amalgamate(Int cw, Int pw, Int child_fill, Int parent_fill,
                Int& zeros, const SymbolicOptions& opt) {
  const Int merged = cw + pw;
  if (!(merged <= opt.relax_max_size || opt.relax_max_size == 0)) return false;
  if (opt.nrelax0 <= 0) return false;

  if (merged <= opt.nrelax0) return true;

  const double new_zeros = static_cast<double>(cw)
                         * (static_cast<double>(parent_fill) + cw - child_fill);
  if (new_zeros == 0) return true;

  const double tot_zeros = static_cast<double>(zeros) + new_zeros;
  const double m         = static_cast<double>(merged);
  const double tot_size  = (m * (m + 1) / 2) + m * (parent_fill - pw);
  const double density   = tot_zeros / tot_size;

  zeros += static_cast<Int>(new_zeros);

  const Ptr ptr_max = std::numeric_limits<Ptr>::max();
  if (tot_size * sizeof(double) >= static_cast<double>(ptr_max)) return false;

  if (merged <= opt.nrelax1 && density < opt.zrelax0) return true;
  if (merged <= opt.nrelax2 && density < opt.zrelax1) return true;
  if (density < opt.zrelax2)                          return true;
  return false;
}

// Supernode relaxation: walk supernodes from the leaves up, amalgamating a
// supernode with its immediate parent supernode while the fill heuristic
// allows it. Updates `col_super`, `first_col`, and `col_count` in place.
void relax_supernodes(const EliminationTree& tree,
                      std::vector<Int>&      col_count,
                      std::vector<Int>&      col_super,
                      std::vector<Int>&      first_col,
                      const SymbolicOptions& opt) {
  const Int num_super = static_cast<Int>(first_col.size()) - 1;
  if (opt.nrelax0 <= 0) return;  // merging disabled

  UnionFind groups;
  groups.reset(num_super);
  std::vector<Int> swidth(num_super);   // current (merged) width per group
  std::vector<Int> szeros(num_super);   // accumulated explicit zeros per group
  std::vector<Int> sfill(num_super);    // current (merged) leading-column fill

  for (Int s = num_super; s >= 1; --s) {
    groups.rep(groups.make(s)) = s;
    swidth[s - 1] = first_col[s] - first_col[s - 1];
    szeros[s - 1] = 0;
    sfill[s - 1]  = col_count[first_col[s - 1] - 1];
  }

  for (Int s = num_super; s >= 1; --s) {
    const Int fcol = first_col[s - 1];
    const Int lcol = first_col[s] - 1;
    const Int cw   = swidth[s - 1];

    const Int par_col = tree.post_parent_of(lcol - 1);
    if (par_col == 0) continue;

    const Int par = groups.rep(groups.find_root(col_super[par_col - 1]));
    if (par != s + 1) continue;                   // parent must be contiguous

    Int tot_zeros = szeros[par - 1];
    const bool merge = amalgamate(cw, swidth[par - 1],
                                  col_count[fcol - 1], col_count[first_col[par - 1] - 1],
                                  tot_zeros, opt);
    if (merge) {
      swidth[s - 1] += swidth[par - 1];
      szeros[s - 1]  = tot_zeros;
      sfill[s - 1]   = cw + sfill[par - 1];
      groups.unite(s, par, s);
    }
  }

  // Compact the surviving group leaders into a new supernode partition.
  std::vector<Int> new_first_col(num_super + 1);
  Int kept = 0;
  for (Int s = 1; s <= num_super; ++s) {
    if (s == groups.rep(groups.find_root(s))) {
      new_first_col[kept] = first_col[s - 1];
      sfill[kept]         = sfill[s - 1];
      ++kept;
    }
  }
  new_first_col[kept] = first_col[num_super];
  new_first_col.resize(kept + 1);

  for (Int s = 1; s <= kept; ++s) {
    Int fcol = new_first_col[s - 1];
    Int lcol = new_first_col[s] - 1;
    for (Int col = fcol; col <= lcol; ++col) {
      col_super[col - 1] = s;
      col_count[col - 1]        = sfill[s - 1] - (col - fcol);
    }
  }
  first_col = std::move(new_first_col);
}

// Symbolic factorization over relaxed supernodes: produces row_pattern_ptr / row_pattern (the
// per-supernode row pattern of L). Each supernode's pattern is assembled by
// merging the (already computed) patterns of the supernodes that update it,
// plus the structure of its own columns in the permuted graph, into a sorted
// reach list. `pgraph` is the permuted expanded graph.
void build_lindx(const CscLower&         pgraph,
                 const EliminationTree&  tree,
                 const std::vector<Int>& first_col,
                 const std::vector<Int>& col_super,
                 const std::vector<Int>& col_count,
                 std::vector<Ptr>&       row_pattern_ptr,
                 std::vector<Idx>&       row_pattern) {
  const Int size      = pgraph.size;
  const Int num_super = static_cast<Int>(first_col.size()) - 1;

  // Lay out the output: supernode s owns col_count[first_col[s]-1] slots.
  row_pattern_ptr.assign(num_super + 1, 0);
  Ptr cursor = 1;
  for (Int s = 1; s <= num_super; ++s) {
    row_pattern_ptr[s - 1] = cursor;
    cursor += col_count[first_col[s - 1] - 1];
  }
  row_pattern_ptr[num_super] = cursor;
  row_pattern.assign(cursor - 1, 0);

  const Idx head = 0;
  const Idx tail = size + 1;
  std::vector<Idx> reach(size + 1);     // singly-linked sorted reach list
  std::vector<Idx> merge_chain(num_super, 0);  // supernodes updating a parent
  std::vector<Int> seen(size, 0);       // membership stamp per row

  Int cnt   = 0;                        // rows collected for the current snode
  Int ksup  = 0;

  // Insert row r into the sorted reach list if not already present.
  auto insert_sorted = [&](Idx r) {
    Idx i = head, nx = reach[head];
    while (r > nx) { i = nx; nx = reach[i]; }
    if (r < nx) { reach[i] = r; reach[r] = nx; seen[r - 1] = ksup; ++cnt; }
  };

  const bool verbose = sps_verbose();
  auto t0 = std::chrono::steady_clock::now();
  const Int report_every = std::max<Int>(1, num_super / 20);

  Ptr written = 0;
  for (ksup = 1; ksup <= num_super; ++ksup) {
    if (verbose && (ksup % report_every == 0 || ksup == num_super)) {
      double el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
      std::printf("    [build_lindx] ksup=%d/%d  rows=%lld  elapsed=%.2fs\n",
                  (int)ksup, (int)num_super, (long long)written, el);
      std::fflush(stdout);
    }

    const Int fcol = first_col[ksup - 1];
    const Int lcol = first_col[ksup] - 1;
    const Int wdt  = lcol - fcol + 1;
    const Int len  = col_count[fcol - 1];
    cnt = 0;
    reach[head] = tail;

    // Merge the row patterns of the supernodes that update ksup.
    Int child = merge_chain[ksup - 1];
    bool first_child = true;
    while (child > 0 && cnt < len) {
      const Int cw   = first_col[child] - first_col[child - 1];
      const Ptr cb   = row_pattern_ptr[child - 1] + cw;     // skip the child's own block
      const Ptr ce   = row_pattern_ptr[child]     - 1;
      if (first_child) {
        // Nothing in the list yet: splice the child's (sorted) rows in reverse
        // so the list comes out ascending.
        for (Ptr e = ce; e >= cb; --e) {
          Idx r = row_pattern[e - 1];
          seen[r - 1] = ksup;
          reach[r]    = reach[head];
          reach[head] = r;
          ++cnt;
        }
      } else {
        for (Ptr e = cb; e <= ce; ++e) insert_sorted(row_pattern[e - 1]);
      }
      first_child = false;
      const Int next = merge_chain[child - 1];
      child = next;
    }

    // Add ksup's own structure if the merge did not already cover it.
    if (cnt < len) {
      for (Int row = fcol; row <= lcol; ++row)
        if (row > fcol && seen[row - 1] != ksup) insert_sorted(row);

      for (Int col = fcol; col <= lcol; ++col)
        for (Ptr e = pgraph.colptr[col - 1]; e < pgraph.colptr[col]; ++e) {
          Idx r = pgraph.rowind[e - 1];
          if (r > fcol && seen[r - 1] != ksup) insert_sorted(r);
        }
    }

    // The supernode's own first column heads the pattern.
    if (reach[head] != fcol) {
      reach[fcol] = reach[head];
      reach[head] = fcol;
      ++cnt;
    }

    if (cnt != len) {
      throw std::logic_error(
          "build_lindx: row-pattern count mismatch (cnt != col_count[first_col-1])");
    }

    // Flush the sorted reach list into row_pattern.
    Idx node = head;
    for (Int t = 0; t < cnt; ++t) {
      node = reach[node];
      row_pattern[written + t] = node;
    }
    written += cnt;

    // Register ksup with its parent supernode's merge chain (if it has rows
    // below its own block, i.e. it updates something).
    if (len > wdt) {
      Idx par_col = tree.post_parent_of(fcol + wdt - 2);
      Int par     = col_super[par_col - 1];
      merge_chain[ksup - 1] = merge_chain[par - 1];
      merge_chain[par - 1]  = ksup;
    }
  }

  row_pattern.resize(written);
}

}  // namespace (anonymous)

// ---------------------------------------------------------------------------
// compute_symbolic — top-level orchestration
// ---------------------------------------------------------------------------
void compute_symbolic(const CscLower&        A_lower,
                      const Permutation&     user_perm,
                      const SymbolicOptions& opts,
                      Symbolic&              out) {
  if (A_lower.size <= 0) {
    throw std::logic_error("compute_symbolic: empty matrix");
  }
  if (static_cast<Int>(user_perm.perm.size()) != A_lower.size) {
    throw std::logic_error(
        "compute_symbolic: ordering size does not match matrix size");
  }
  if (!user_perm.validate()) {
    throw std::logic_error("compute_symbolic: invalid ordering");
  }

  using clock_t = std::chrono::steady_clock;
  const bool verbose = sps_verbose();
  auto stamp = [&](const char* label, clock_t::time_point t0) {
    if (verbose) {
      double ms = std::chrono::duration<double, std::milli>(
                       clock_t::now() - t0).count();
      std::printf("  [symbolic] %-20s %.2f ms\n", label, ms);
      std::fflush(stdout);
    }
  };

  auto T = clock_t::now();

  CscLower g = A_lower;
  if (!g.expanded) g.expand();
  stamp("expand", T); T = clock_t::now();

  out = Symbolic{};
  out.n        = A_lower.size;
  out.ordering = user_perm;

  out.etree.build(g, out.ordering);
  stamp("etree_build", T); T = clock_t::now();

  out.etree.postorder(out.ordering);
  stamp("postorder", T); T = clock_t::now();

  std::vector<Int> rc;
  column_counts(g, out.ordering, out.etree, out.col_count, rc);
  stamp("column_counts", T); T = clock_t::now();

  out.etree.reorder_children(out.col_count, out.ordering);
  stamp("reorder_children", T); T = clock_t::now();

  find_supernodes(out.etree, out.col_count,
                  out.supernode_of_col, out.supernode_first_col, opts.relax_max_size);
  stamp("find_supernodes", T); T = clock_t::now();

  relax_supernodes(out.etree, out.col_count,
                   out.supernode_of_col, out.supernode_first_col, opts);
  stamp("relax_supernodes", T); T = clock_t::now();

  out.n_super = static_cast<Int>(out.supernode_first_col.size()) - 1;

  CscLower pgraph;
  permute_expanded(g, out.ordering, pgraph);
  stamp("permute_expanded", T); T = clock_t::now();

  build_lindx(pgraph, out.etree,
              out.supernode_first_col, out.supernode_of_col, out.col_count,
              out.row_pattern_ptr, out.row_pattern);
  stamp("build_lindx", T); T = clock_t::now();

  out.row_pattern_len = static_cast<Ptr>(out.row_pattern.size());
  out.nnz_l  = 0;
  for (Int j = 1; j <= out.n; ++j) out.nnz_l += out.col_count[j - 1];

  out.sn_etree = out.etree.supernodal_tree(out.supernode_first_col, out.supernode_of_col);
  stamp("sn_etree", T);
}

}}  // namespace sTiles::sparse
