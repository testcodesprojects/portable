#ifndef SPS_CORE_SUPERNODE_HPP
#define SPS_CORE_SUPERNODE_HPP

#include "etree.hpp"
#include "symbolic.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sTiles { namespace sparse {

// One dense block of L belonging to supernode pair (J, I), where
//   I is the column ("pivot") supernode,
//   J is the row supernode (J >= I; J == I is the diagonal block).
//
// Storage is column-major in the cell's local frame:
//   nzval[r + rows * c],  0 <= r < rows,  0 <= c < cols.
//
// `cols` equals the width of supernode I.
// `rows` equals the run-length of column-supernode I's row pattern that
// belongs to supernode J. For a fundamental supernode J this is exactly
// `width(J)`, but after relaxation it can be any sorted subset of J's
// range. The diagonal cell (J == I) always has rows = cols = width(I) and
// covers global rows [supernode_first_col[I-1], supernode_first_col[I]-1].
//
// `lx_offset` (1-based) is the absolute index into Symbolic::row_pattern of this
// cell's first row entry. The cell's row indices in *global* numbering are
// `row_pattern[lx_offset - 1 .. lx_offset + rows - 2]`, sorted ascending.
struct Cell {
  Int     I        = 0;
  Int     J        = 0;
  Int     rows     = 0;
  Int     cols     = 0;
  Ptr     lx_offset = 0;
  double* nzval    = nullptr;
};

// Owning storage of all cells. One contiguous arena of doubles backs every
// cell's `nzval`, so iterating cells of the same column-supernode is
// cache-friendly. Cells are listed grouped by column-supernode I, in
// increasing J within each group; the diagonal cell of supernode I is the
// first cell of its group.
class CellStore {
 public:
  CellStore() = default;
  CellStore(const CellStore&)            = delete;
  CellStore& operator=(const CellStore&) = delete;
  CellStore(CellStore&& other) noexcept;
  CellStore& operator=(CellStore&& other) noexcept;
  ~CellStore();

  // Allocate the arena (via sTiles::Memory::MemoryManager, tagged with
  // `group_id`) and emit the cells_ + cell_idx_ layout. The arena is
  // zero-initialized. Existing arena (if any) is freed first.
  void allocate(const Symbolic& s, int group_id = -1);

  // Walk the user's lower-triangular CSC of A, permute each entry through
  // `s.ordering`, and write into the right cell at the right (r, c) offset.
  // Pre-existing arena values are not cleared (caller can zero the arena
  // first if needed).
  void load_from_csc(const CscLower& A_lower, const Symbolic& s);

  // Build a flat "scatter plan": for each nonzero position p of A_lower (index
  // into A_lower.nzval / A_lower.rowind), record the ABSOLUTE arena offset where
  // that value must land, or -1 to skip (upper-tri duplicate). The plan depends
  // ONLY on the symbolic pattern + ordering + cell geometry — none of which
  // change across re-factorizations — so it is built once and reused. This is
  // exactly the per-entry destination that load_from_csc() recomputes every
  // call (two invp lookups + two supernode lookups + find() + a lower_bound).
  // Arena must be allocated (arena_size() > 0) before calling.
  void build_load_map(const CscLower& A_lower, const Symbolic& s,
                      std::vector<Ptr>& map_out) const;

  // Apply a plan from build_load_map(): arena[map[p]] = A_lower.nzval[p] for
  // every p with map[p] >= 0. Assumes the arena is already zeroed. Equivalent
  // in result to load_from_csc() but with no per-entry index computation.
  void load_from_csc_mapped(const CscLower& A_lower, const std::vector<Ptr>& map);

  // Lookup. Returns nullptr if cell (J, I) is not present.
  Cell*       find(Int J, Int I);
  const Cell* find(Int J, Int I) const;

  Int         cell_count() const { return static_cast<Int>(cells_.size()); }
  Cell&       at(Int idx)        { return cells_[idx]; }
  const Cell& at(Int idx) const  { return cells_[idx]; }

  Ptr           arena_size()  const { return arena_size_; }
  const double* arena_data()  const { return arena_; }
  double*       arena_data()        { return arena_; }

  // Allocate a fresh CellStore that mirrors `other`'s layout. The arena is
  // zero-initialized via sTiles::Memory::MemoryManager (tagged with
  // `group_id`), and nzval pointers are rebound to the new arena.
  void allocate_like(const CellStore& other, int group_id = -1);

 private:
  void release_arena_();

  Int                       n_super_    = 0;
  Ptr                       arena_size_ = 0;
  double*                   arena_      = nullptr;   // owned via MemoryManager
  int                       group_id_   = -1;
  std::vector<Cell>         cells_;
  std::unordered_map<int64_t, Int> cell_idx_;
};

}}  // namespace sTiles::sparse

#endif  // SPS_CORE_SUPERNODE_HPP
