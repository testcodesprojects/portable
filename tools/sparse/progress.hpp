#ifndef SPS_SCHED_PROGRESS_HPP
#define SPS_SCHED_PROGRESS_HPP

#include <atomic>
#include <cstdint>
#include <memory>

namespace sTiles { namespace sparse {

// Atomic byte array indexed by cell index. Producers call
// `mark_done(cell_idx)`; consumers call `wait_done(cell_idx, abort)`. One
// byte per cell — never more.
//
// Used to be n_super × n_super (sized for `(row_sn, col_sn)` lookup), but
// that exploded at scale: yU0G1u with 250k supernodes → 62 GB just for
// the progress array. Cell indexing is bounded by `cell_count()` (~1 M
// cells in the same case → 1 MB).
//
// Phase 7 will replace the simple `pause` with sTiles' `cpu_pause_hybrid`
// adaptive backoff (plan §4); for v1 it's just an x86 PAUSE.
class ProgressMatrix {
 public:
  void resize(size_t n_cells) {
    n_ = n_cells;
    flat_.reset(new std::atomic<uint8_t>[n_cells]);
    clear();
  }

  void clear() {
    for (size_t i = 0; i < n_; ++i) {
      flat_[i].store(0, std::memory_order_relaxed);
    }
  }

  void mark_done(size_t cell_idx) {
    flat_[cell_idx].store(1, std::memory_order_release);
  }

  bool is_done(size_t cell_idx) const {
    return flat_[cell_idx].load(std::memory_order_acquire) == 1;
  }

  // Spin until the cell flips to 1 or the abort flag is set.
  void wait_done(size_t cell_idx, const std::atomic<bool>& abort) const {
    auto& cell = flat_[cell_idx];
    while (cell.load(std::memory_order_acquire) != 1) {
      if (abort.load(std::memory_order_relaxed)) return;
#if defined(__x86_64__) || defined(__i386__)
      __builtin_ia32_pause();
#endif
    }
  }

  size_t size() const { return n_; }

 private:
  size_t n_ = 0;
  std::unique_ptr<std::atomic<uint8_t>[]> flat_;
};

}}  // namespace sTiles::sparse

#endif  // SPS_SCHED_PROGRESS_HPP
