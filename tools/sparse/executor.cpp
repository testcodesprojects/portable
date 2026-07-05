#include "executor.hpp"

#include "kernels.hpp"

#include "../common/stiles_logger.hpp"
#include "../memory/MemoryManager.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

#include <omp.h>
#include <pthread.h>
#include <sched.h>

namespace sTiles { namespace sparse {

namespace {
#if defined(__linux__)
// Affinity mask snapshot taken at library load, before sTiles' static pool
// binds the calling thread to a single core (tools/control/stilesos.cpp).
// std::threads inherit the creator's mask, so the pthreads runners below
// would otherwise spawn every rank onto main's one core and time-slice it
// (observed: 8 ranks, cpu/wall ~= 1, slower than serial). Each spawned rank
// resets itself to this snapshot, which still honors any taskset/cgroup
// restriction present at process start.
//
// Linux-only: cpu_set_t / sched_getaffinity / pthread_setaffinity_np have no
// macOS equivalent (no sched-affinity API). The pthreads backend still runs
// there without pinning, but OMP is the default backend on macOS (param 8).
struct LoadTimeAffinity {
  cpu_set_t mask;
  bool      valid = false;
  LoadTimeAffinity() { valid = (sched_getaffinity(0, sizeof(mask), &mask) == 0); }
};
const LoadTimeAffinity g_load_affinity;

inline void unbind_from_creator() {
  if (g_load_affinity.valid)
    pthread_setaffinity_np(pthread_self(), sizeof(g_load_affinity.mask),
                           &g_load_affinity.mask);
}
#else
inline void unbind_from_creator() {}   // no sched-affinity on this OS (macOS)
#endif
}  // namespace

void prepare(SpsState& s) {
  if (!s.symbolic || !s.cells || !s.tasks) {
    throw std::logic_error("SpsState::prepare: symbolic/cells/tasks must be wired");
  }
  s.n_cells = s.cells->cell_count();
  s.progress.resize(static_cast<size_t>(s.n_cells));
  s.updates_remaining.reset(new std::atomic<int>[s.n_cells]);
  s.cell_locks.reset(new std::atomic<int>[s.n_cells]);
  for (Int i = 0; i < s.n_cells; ++i)
    s.cell_locks[i].store(0, std::memory_order_relaxed);

  // Per-rank UPDATE scratch (T = A * B^T). One buffer per rank so threads
  // don't race; sized to the largest UPDATE in that rank's slice.
  const auto& tasks   = s.tasks->tasks;
  const auto& offsets = s.tasks->offsets;
  const int   n_ranks = std::max<int>(1,
                          static_cast<int>(offsets.size()) - 1);
  s.update_scratch.clear();
  s.update_scratch.resize(n_ranks);
  for (int r = 0; r < n_ranks; ++r) {
    Ptr max_t = 0;
    for (int idx = offsets[r]; idx < offsets[r + 1]; ++idx) {
      const SpsTask& t = tasks[idx];
      if (t.op != TaskOp::UPDATE) continue;
      const Cell& A = s.cells->at(t.cell_a);
      const Cell& B = s.cells->at(t.cell_b);
      Ptr sz = static_cast<Ptr>(A.rows) * B.rows;
      if (sz > max_t) max_t = sz;
    }
    s.update_scratch[r].reset(max_t > 0 ? new double[max_t] : nullptr);
  }

  reset_for_factorize(s);
}

void reset_for_factorize(SpsState& s) {
  s.progress.clear();
  s.abort_flag.store(false, std::memory_order_relaxed);
  for (Int i = 0; i < s.n_cells; ++i) {
    s.updates_remaining[i].store(s.tasks->update_target_count[i],
                                  std::memory_order_relaxed);
  }
}

void factorize_run(int rank, SpsState* s) {
  const auto& tasks   = s->tasks->tasks;
  const auto& offsets = s->tasks->offsets;
  const int   start   = offsets[rank];
  const int   end     = offsets[rank + 1];

  ProgressMatrix& prog = s->progress;
  std::atomic<bool>& abort = s->abort_flag;

  for (int idx = start; idx < end; ++idx) {
    if (abort.load(std::memory_order_relaxed)) return;
    const SpsTask& t = tasks[idx];

    switch (t.op) {
      case TaskOp::FACTOR: {
        // Wait for all UPDATEs that target cell (I, I) to finish their
        // scatter. updates_remaining[t.cell_a] hits 0 after the last
        // scatter completes (fetch_sub is acq_rel, so the scatter
        // happens-before the decrement).
        auto& urem_diag = s->updates_remaining[t.cell_a];
        while (urem_diag.load(std::memory_order_acquire) > 0) {
          if (abort.load(std::memory_order_relaxed)) return;
#if defined(__x86_64__) || defined(__i386__)
          __builtin_ia32_pause();
#endif
        }

        Cell& c = s->cells->at(t.cell_a);
        int info = kernels::potrf('L', c.cols, c.nzval, c.rows);
        if (info != 0) {
          abort.store(true, std::memory_order_release);
          return;
        }
        prog.mark_done(t.cell_a);   // cell_a == (I, I)
        break;
      }

      case TaskOp::TRSM: {
        // Wait for FACTOR(I) (cell (I, I) = cell_a is final L11) AND for
        // all UPDATEs to cell (J, I) = cell_b to scatter.
        prog.wait_done(t.cell_a, abort);
        auto& urem_off = s->updates_remaining[t.cell_b];
        while (urem_off.load(std::memory_order_acquire) > 0) {
          if (abort.load(std::memory_order_relaxed)) return;
#if defined(__x86_64__) || defined(__i386__)
          __builtin_ia32_pause();
#endif
        }
        if (abort.load(std::memory_order_relaxed)) return;

        Cell& diag = s->cells->at(t.cell_a);
        Cell& off  = s->cells->at(t.cell_b);
        kernels::trsm('R', 'L', 'T', 'N',
                      off.rows, off.cols,
                      1.0,
                      diag.nzval, diag.rows,
                      off.nzval,  off.rows);
        prog.mark_done(t.cell_b);   // cell_b == (J, I)
        break;
      }

      case TaskOp::UPDATE: {
        // Wait for cell (K, I) = cell_a and cell (J, I) = cell_b to be
        // produced (by their respective TRSM/FACTOR producers).
        prog.wait_done(t.cell_a, abort);
        prog.wait_done(t.cell_b, abort);
        if (abort.load(std::memory_order_relaxed)) return;

        Cell& A = s->cells->at(t.cell_a);   // (K, I)
        Cell& B = s->cells->at(t.cell_b);   // (J, I)
        Cell& C = s->cells->at(t.cell_c);   // (K, J)

        // T = A * B^T into per-rank scratch, then scatter into C with row
        // remapping (A's K-rows are a sorted subset of C's K-rows). Per-
        // cell spinlock serializes concurrent UPDATEs to the same C.
        double* T = s->update_scratch[rank].get();
        kernels::gemm('N', 'T',
                      A.rows, B.rows, A.cols,
                      1.0,
                      A.nzval, A.rows,
                      B.nzval, B.rows,
                      0.0,
                      T, A.rows);

        const Symbolic& sym = *s->symbolic;
        const Idx* A_rows = &sym.row_pattern[A.lx_offset - 1];
        const Idx* C_rows = &sym.row_pattern[C.lx_offset - 1];
        const Idx* B_rows = &sym.row_pattern[B.lx_offset - 1];
        const Int  J_xsuper_lo = sym.supernode_first_col[t.J - 1];

        auto& lk = s->cell_locks[t.cell_c];
        int   expected;
        do {
          expected = 0;
          if (lk.compare_exchange_weak(expected, 1,
                                       std::memory_order_acquire,
                                       std::memory_order_relaxed)) break;
#if defined(__x86_64__) || defined(__i386__)
          __builtin_ia32_pause();
#endif
        } while (true);

        Int ci = 0;
        for (Int ai = 0; ai < A.rows; ++ai) {
          const Idx ra = A_rows[ai];
          while (ci < C.rows && C_rows[ci] != ra) ++ci;
          if (ci == C.rows) {
            lk.store(0, std::memory_order_release);
            abort.store(true, std::memory_order_release);
            return;
          }
          for (Int bj = 0; bj < B.rows; ++bj) {
            const Int col_off = static_cast<Int>(B_rows[bj]) - J_xsuper_lo;
            C.nzval[ci + C.rows * col_off] -= T[ai + A.rows * bj];
          }
        }
        lk.store(0, std::memory_order_release);

        s->updates_remaining[t.cell_c].fetch_sub(1, std::memory_order_acq_rel);
        break;
      }
    }
  }
}

void factorize_run_parallel_pthreads(SpsState& state) {
  const int n_ranks = static_cast<int>(state.tasks->offsets.size()) - 1;
  if (n_ranks <= 1) {
    factorize_run(0, &state);
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(n_ranks);
  for (int r = 0; r < n_ranks; ++r) {
    threads.emplace_back([r, &state]{
      unbind_from_creator();
      factorize_run(r, &state);
    });
  }
  for (auto& t : threads) t.join();
}

void factorize_run_parallel_omp(SpsState& state) {
  const int n_ranks = static_cast<int>(state.tasks->offsets.size()) - 1;
  const bool _dump = []() {
    const char* v = std::getenv("STILES_DUMP_TASKS");
    return v && std::atoi(v) != 0;
  }();
  if (_dump) {
    std::fprintf(stderr, "[STILES_DUMP/sparse] n_ranks=%d  total_tasks=%d  offsets=[",
                 n_ranks,
                 (state.tasks->offsets.empty() ? 0 :
                  static_cast<int>(state.tasks->offsets.back())));
    for (std::size_t i = 0; i < state.tasks->offsets.size(); ++i) {
      std::fprintf(stderr, "%s%d", (i ? "," : ""),
                   static_cast<int>(state.tasks->offsets[i]));
    }
    std::fprintf(stderr, "]\n");
  }
  if (n_ranks <= 1) {
    const double _t0 = _dump ? omp_get_wtime() : 0.0;
    factorize_run(0, &state);
    if (_dump) {
      std::fprintf(stderr, "[STILES_DUMP/sparse] SERIAL rank=0 dt=%.4fs\n",
                   omp_get_wtime() - _t0);
    }
    return;
  }
  // Nested-OMP guard. If we are already inside an outer OMP parallel
  // region (e.g. user code dispatching multiple chol calls in parallel)
  // and OMP nested parallelism is disabled (the default), the inner
  // `#pragma omp parallel num_threads(n_ranks)` collapses to 1 thread.
  // n_ranks-way task partitioning then leaves rank 1..n_ranks-1's tasks
  // unexecuted and FACTOR/TRSM cross-rank dependencies hang forever.
  // Fall through to the std::thread runner — OS-level threads are
  // unaffected by OMP's nesting setting.
  if (omp_in_parallel() &&
      omp_get_max_active_levels() <= omp_get_active_level()) {
    factorize_run_parallel_pthreads(state);
    return;
  }
  #pragma omp parallel num_threads(n_ranks)
  {
    const int _rank = omp_get_thread_num();
    const int _start = (_rank < static_cast<int>(state.tasks->offsets.size()))
                          ? static_cast<int>(state.tasks->offsets[_rank]) : 0;
    const int _end   = (_rank + 1 < static_cast<int>(state.tasks->offsets.size()))
                          ? static_cast<int>(state.tasks->offsets[_rank + 1]) : 0;
    const double _t0 = _dump ? omp_get_wtime() : 0.0;
    factorize_run(_rank, &state);
    if (_dump) {
      const double _dt = omp_get_wtime() - _t0;
      #pragma omp critical
      std::fprintf(stderr, "[STILES_DUMP/sparse] rank=%d  tasks=[%d,%d)  count=%d  dt=%.4fs\n",
                   _rank, _start, _end, _end - _start, _dt);
    }
  }
}

// ─────────────────────────── Selected inverse ─────────────────────────────

namespace {

// Mirror lower triangle to upper for the (I,I) diag block after LAUUM.
inline void mirror_low_to_up(int n, double* A, int lda) {
  for (int j = 0; j < n; ++j) {
    for (int i = j + 1; i < n; ++i) {
      A[j + lda * i] = A[i + lda * j];
    }
  }
}

inline Int find_pos(Idx target, const Idx* base, Int n) {
  for (Int i = 0; i < n; ++i) if (base[i] == target) return i;
  return -1;
}

// Pointer into the M arena that corresponds to a given Z cell. Same offset
// the Z cell has into Z's arena, but pointing into M_arena.
inline const double* M_for(const SelinvState& st, const Cell& zc) {
  const Ptr off = static_cast<Ptr>(zc.nzval - st.Z_cs->arena_data());
  return st.M_arena + off;
}

}  // namespace

SelinvState::~SelinvState() {
  if (M_arena != nullptr) {
    ::MemoryManager::deallocate(M_arena);  // sets M_arena to nullptr
  }
  M_arena_size = 0;
}

SelinvState::SelinvState(SelinvState&& other) noexcept
    : L_cs            (other.L_cs),
      Z_cs            (other.Z_cs),
      sym             (other.sym),
      tasks           (other.tasks),
      M_arena         (other.M_arena),
      M_arena_size    (other.M_arena_size),
      group_id        (other.group_id),
      contrib_remaining(std::move(other.contrib_remaining)),
      n_off_remaining (std::move(other.n_off_remaining)),
      phase1_done     (std::move(other.phase1_done)),
      col_diag        (std::move(other.col_diag)),
      col_off_first   (std::move(other.col_off_first)),
      col_off_last    (std::move(other.col_off_last))
{
  abort_flag.store(other.abort_flag.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
  other.M_arena      = nullptr;
  other.M_arena_size = 0;
}

SelinvState& SelinvState::operator=(SelinvState&& other) noexcept {
  if (this != &other) {
    if (M_arena) ::MemoryManager::deallocate(M_arena);
    L_cs              = other.L_cs;
    Z_cs              = other.Z_cs;
    sym               = other.sym;
    tasks             = other.tasks;
    M_arena           = other.M_arena;
    M_arena_size      = other.M_arena_size;
    group_id          = other.group_id;
    contrib_remaining = std::move(other.contrib_remaining);
    n_off_remaining   = std::move(other.n_off_remaining);
    phase1_done       = std::move(other.phase1_done);
    col_diag          = std::move(other.col_diag);
    col_off_first     = std::move(other.col_off_first);
    col_off_last      = std::move(other.col_off_last);
    abort_flag.store(other.abort_flag.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    other.M_arena      = nullptr;
    other.M_arena_size = 0;
  }
  return *this;
}

void selinv_prepare(SelinvState& st) {
  if (!st.L_cs || !st.Z_cs || !st.sym || !st.tasks) {
    throw std::logic_error("selinv_prepare: L_cs/Z_cs/sym/tasks must be wired");
  }

  // Z layout mirrors L (allocated upstream by api::selinv via allocate_like).
  // Free any previous M arena, then allocate (uninitialized; snapshot_M will
  // overwrite every byte) through MemoryManager so the bytes show up in the
  // per-group memory totals and get reaped on freeGroup.
  if (st.M_arena != nullptr) {
    ::MemoryManager::deallocate(st.M_arena);
  }
  st.M_arena_size = st.Z_cs->arena_size();
  st.M_arena      = (st.M_arena_size > 0)
      ? ::MemoryManager::allocate<double>(static_cast<size_t>(st.M_arena_size), st.group_id)
      : nullptr;

  const Int n_super = st.sym->n_super;
  st.contrib_remaining.reset(new std::atomic<int>[n_super + 1]);
  st.n_off_remaining .reset(new std::atomic<int>[n_super + 1]);
  st.phase1_done     .reset(new std::atomic<uint8_t>[n_super + 1]);
  for (Int I = 0; I <= n_super; ++I) {
    st.contrib_remaining[I].store(I == 0 ? 0 : st.tasks->contrib_remaining[I],
                                   std::memory_order_relaxed);
    st.n_off_remaining[I].store(I == 0 ? 0 : st.tasks->n_off_in_col[I],
                                 std::memory_order_relaxed);
    st.phase1_done[I].store(0, std::memory_order_relaxed);
  }
  st.abort_flag.store(false, std::memory_order_relaxed);

  // Precompute per-column cell layout (linear scan over CellStore once).
  // Replaces the O(n_cells) build_col_view that fired on every task.
  const Int n_cells = st.Z_cs->cell_count();
  st.col_diag     .assign(n_super + 1, 0);
  st.col_off_first.assign(n_super + 1, 0);
  st.col_off_last .assign(n_super + 1, 0);
  Int idx = 0;
  for (Int I = 1; I <= n_super; ++I) {
    if (idx >= n_cells || st.Z_cs->at(idx).I != I || st.Z_cs->at(idx).J != I) {
      throw std::logic_error("selinv_prepare: missing diagonal cell at column I");
    }
    st.col_diag[I] = static_cast<uint32_t>(idx);
    ++idx;
    st.col_off_first[I] = static_cast<uint32_t>(idx);
    while (idx < n_cells && st.Z_cs->at(idx).I == I) ++idx;
    st.col_off_last[I]  = static_cast<uint32_t>(idx);
  }
}

namespace {

// Spin until `cell` flips to 1 or abort fires. Used for phase1_done.
inline void wait_byte(std::atomic<uint8_t>& cell,
                      const std::atomic<bool>& abort) {
  while (cell.load(std::memory_order_acquire) == 0) {
    if (abort.load(std::memory_order_relaxed)) return;
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
  }
}

inline void wait_zero(std::atomic<int>& cell,
                      const std::atomic<bool>& abort) {
  while (cell.load(std::memory_order_acquire) > 0) {
    if (abort.load(std::memory_order_relaxed)) return;
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
  }
}

void exec_phase1_trtri(const SelinvTask& t, SelinvState* st) {
  Cell& Z_diag       = st->Z_cs->at(t.cell_diag);
  const Cell& L_diag = st->L_cs->at(t.cell_diag);
  const Int   w      = Z_diag.cols;
  std::copy(L_diag.nzval, L_diag.nzval + (size_t)w * w, Z_diag.nzval);
  int info = kernels::trtri('L', 'N', w, Z_diag.nzval, Z_diag.rows);
  if (info != 0) {
    st->abort_flag.store(true, std::memory_order_release);
    return;
  }
  st->phase1_done[t.I].store(1, std::memory_order_release);
}

void exec_phase1_trsm(const SelinvTask& t, SelinvState* st) {
  // PHASE1_TRTRI(I) only matters as a barrier with PHASE1_TRSM in the
  // serial code's ordering; the TRSM itself uses the *original* L diag
  // (TRSM solves X*L = B, so passing L gives X = B*L^{-1}). Passing Z's
  // diag (which holds L^{-1} after TRTRI) would compute X = B*L, the
  // wrong direction.
  wait_byte(st->phase1_done[t.I], st->abort_flag);
  if (st->abort_flag.load(std::memory_order_relaxed)) return;

  Cell&       Z_off  = st->Z_cs->at(t.cell_off);
  const Cell& L_off  = st->L_cs->at(t.cell_off);
  const Cell& L_diag = st->L_cs->at(t.cell_diag);
  std::copy(L_off.nzval, L_off.nzval + (size_t)L_off.rows * L_off.cols,
            Z_off.nzval);
  kernels::trsm('R', 'L', 'N', 'N',
                Z_off.rows, Z_off.cols,
                1.0,
                L_diag.nzval, L_diag.rows,
                Z_off.nzval,  Z_off.rows);
}

// Per-column off-cell range, precomputed once in selinv_prepare. Iterating
// `for (uint32_t i = first; i < last; ++i)` enumerates the off cells of
// column I in CellStore order (ascending J).
struct OffRange {
  uint32_t first;
  uint32_t last;
};

inline OffRange off_range(const SelinvState& st, Int I) {
  return { st.col_off_first[I], st.col_off_last[I] };
}

void exec_phase2_off(const SelinvTask& t, SelinvState* st) {
  // Wait for column I to be ready (all ancestor supernodes done).
  wait_zero(st->contrib_remaining[t.I], st->abort_flag);
  if (st->abort_flag.load(std::memory_order_relaxed)) return;

  const Symbolic& s = *st->sym;
  CellStore&      Z_cs = *st->Z_cs;

  Cell&       Z_JI = Z_cs.at(t.cell_off);
  const Int   w_I        = Z_cs.at(t.cell_diag).cols;
  const Int   J          = Z_JI.J;
  const Int   rJ_in_I    = Z_JI.rows;
  const Int   J_super0   = s.supernode_first_col[J - 1];
  const Idx*  JI_rows    = &s.row_pattern[Z_JI.lx_offset - 1];

  // Local-in-J positions for cell (J, I)'s rows.
  std::vector<Int> p_J(rJ_in_I);
  for (Int a = 0; a < rJ_in_I; ++a) p_J[a] = JI_rows[a] - J_super0;

  std::vector<double> acc((size_t)rJ_in_I * w_I, 0.0);

  // Walk all K in I's off_list using the precomputed range.
  const OffRange off = off_range(*st, t.I);

  for (uint32_t idx_KI = off.first; idx_KI < off.last; ++idx_KI) {
    const Cell&   M_KI_cell = Z_cs.at(idx_KI);
    const double* M_KI      = M_for(*st, M_KI_cell);
    const Int     K         = M_KI_cell.J;
    const Int     rK_in_I   = M_KI_cell.rows;
    const Int     K_super0  = s.supernode_first_col[K - 1];
    const Idx*    KI_rows   = &s.row_pattern[M_KI_cell.lx_offset - 1];

    if (K == J) {
      const Cell* Z_JJ_p = Z_cs.find(J, J);
      if (!Z_JJ_p) continue;
      std::vector<double> Z_sub((size_t)rJ_in_I * rJ_in_I);
      for (Int b = 0; b < rJ_in_I; ++b) {
        for (Int a = 0; a < rJ_in_I; ++a) {
          Z_sub[a + (size_t)rJ_in_I * b] =
              Z_JJ_p->nzval[p_J[a] + (size_t)Z_JJ_p->rows * p_J[b]];
        }
      }
      kernels::gemm('N', 'N',
                    rJ_in_I, w_I, rJ_in_I,
                    1.0,
                    Z_sub.data(), rJ_in_I,
                    M_KI,         rK_in_I,
                    1.0,
                    acc.data(), rJ_in_I);
    } else if (K > J) {
      const Cell* Z_KJ = Z_cs.find(K, J);
      if (!Z_KJ) continue;
      const Idx* KJ_rows = &s.row_pattern[Z_KJ->lx_offset - 1];
      const Int  rK_in_J = Z_KJ->rows;

      std::vector<double> Z_aligned((size_t)rK_in_I * rJ_in_I, 0.0);
      Int kj_cursor = 0;
      for (Int k_in_I = 0; k_in_I < rK_in_I; ++k_in_I) {
        Idx target = KI_rows[k_in_I];
        while (kj_cursor < rK_in_J && KJ_rows[kj_cursor] != target)
          ++kj_cursor;
        if (kj_cursor == rK_in_J) {
          st->abort_flag.store(true, std::memory_order_release);
          return;
        }
        for (Int a = 0; a < rJ_in_I; ++a) {
          Z_aligned[k_in_I + (size_t)rK_in_I * a] =
              Z_KJ->nzval[kj_cursor + (size_t)Z_KJ->rows * p_J[a]];
        }
      }
      kernels::gemm('T', 'N',
                    rJ_in_I, w_I, rK_in_I,
                    1.0,
                    Z_aligned.data(), rK_in_I,
                    M_KI,             rK_in_I,
                    1.0,
                    acc.data(), rJ_in_I);
    } else {  // I < K < J
      const Cell* Z_JK = Z_cs.find(J, K);
      if (!Z_JK) continue;
      const Idx* JK_rows = &s.row_pattern[Z_JK->lx_offset - 1];
      const Int  rJ_in_K = Z_JK->rows;

      std::vector<double> Z_aligned((size_t)rJ_in_I * rK_in_I, 0.0);
      for (Int a = 0; a < rJ_in_I; ++a) {
        Int jk_pos = find_pos(JI_rows[a], JK_rows, rJ_in_K);
        if (jk_pos < 0) continue;
        for (Int k_in_I = 0; k_in_I < rK_in_I; ++k_in_I) {
          Int k_local = KI_rows[k_in_I] - K_super0;
          Z_aligned[a + (size_t)rJ_in_I * k_in_I] =
              Z_JK->nzval[jk_pos + (size_t)Z_JK->rows * k_local];
        }
      }
      kernels::gemm('N', 'N',
                    rJ_in_I, w_I, rK_in_I,
                    1.0,
                    Z_aligned.data(), rJ_in_I,
                    M_KI,             rK_in_I,
                    1.0,
                    acc.data(), rJ_in_I);
    }
  }

  for (size_t k = 0; k < acc.size(); ++k) Z_JI.nzval[k] = -acc[k];

  // Mark this off-cell complete; PHASE2_DIAG(I) waits for n_off_remaining=0.
  st->n_off_remaining[t.I].fetch_sub(1, std::memory_order_acq_rel);
}

void exec_phase2_diag(const SelinvTask& t, SelinvState* st) {
  // Column I must be ready, AND all PHASE2_OFF(I, *) of this column must be
  // done (we read their freshly-written Z[K,I] values).
  wait_zero(st->contrib_remaining[t.I], st->abort_flag);
  if (st->abort_flag.load(std::memory_order_relaxed)) return;
  wait_zero(st->n_off_remaining[t.I], st->abort_flag);
  if (st->abort_flag.load(std::memory_order_relaxed)) return;

  Cell& Z_diag        = st->Z_cs->at(t.cell_diag);
  const Int  w_I      = Z_diag.cols;
  const double* M_diag = M_for(*st, Z_diag);

  // Z[I,I] := M[I,I]^T * M[I,I] via LAUUM on the M snapshot.
  // We need a writable copy since LAUUM is in-place; reuse Z_diag.nzval
  // (which currently holds M[I,I] = L[I,I]^{-1} from phase 1).
  std::copy(M_diag, M_diag + (size_t)w_I * w_I, Z_diag.nzval);
  int info = kernels::lauum('L', w_I, Z_diag.nzval, Z_diag.rows);
  if (info != 0) {
    st->abort_flag.store(true, std::memory_order_release);
    return;
  }
  mirror_low_to_up(w_I, Z_diag.nzval, Z_diag.rows);

  // Subtract sum_K M[K,I]^T * Z[K,I] for each K in I's pattern (precomputed range).
  const OffRange off = off_range(*st, t.I);
  for (uint32_t idx_KI = off.first; idx_KI < off.last; ++idx_KI) {
    const Cell&   Z_KI  = st->Z_cs->at(idx_KI);
    const double* M_KI  = M_for(*st, Z_KI);
    kernels::gemm('T', 'N',
                  w_I, w_I, Z_KI.rows,
                  -1.0,
                  M_KI,        Z_KI.rows,
                  Z_KI.nzval,  Z_KI.rows,
                  1.0,
                  Z_diag.nzval, Z_diag.rows);
  }

  // Column I is now fully done. Decrement consumers' contrib_remaining.
  const auto& consumers = st->tasks->consumers_of[t.I];
  for (uint32_t I_consumer : consumers) {
    st->contrib_remaining[I_consumer].fetch_sub(1, std::memory_order_acq_rel);
  }
}

}  // namespace

void selinv_run(int rank, SelinvState* st) {
  // Phase boundary: phase 1 fills Z with M = L^{-1}. Phase 2 reads M from
  // M_arena. We snapshot the arena once between the two phases. The
  // snapshot is performed by rank 0; everyone else waits on a phase-1
  // barrier inside the parallel wrapper.

  const auto& tasks   = st->tasks->tasks;
  const auto& offsets = st->tasks->offsets;
  const int   start   = offsets[rank];
  const int   end     = offsets[rank + 1];

  for (int idx = start; idx < end; ++idx) {
    if (st->abort_flag.load(std::memory_order_relaxed)) return;
    const SelinvTask& t = tasks[idx];
    switch (t.op) {
      case SelinvOp::PHASE1_TRTRI: exec_phase1_trtri(t, st); break;
      case SelinvOp::PHASE1_TRSM:  exec_phase1_trsm (t, st); break;
      case SelinvOp::PHASE2_OFF:   exec_phase2_off  (t, st); break;
      case SelinvOp::PHASE2_DIAG:  exec_phase2_diag (t, st); break;
    }
  }
}

namespace {

// Snapshot Z's arena into M_arena. Called once between phase 1 and phase 2.
void snapshot_M(SelinvState& st) {
  if (st.M_arena_size == 0 || !st.M_arena || !st.Z_cs->arena_data()) return;
  std::copy(st.Z_cs->arena_data(),
            st.Z_cs->arena_data() + st.M_arena_size,
            st.M_arena);
}

// Index of the first PHASE2_* task in this rank's slice; used as a barrier
// landmark when running with separate phase-1 / phase-2 parallel regions.
int find_phase2_start(const SelinvState& st, int rank) {
  const auto& tasks   = st.tasks->tasks;
  const auto& offsets = st.tasks->offsets;
  for (int idx = offsets[rank]; idx < offsets[rank + 1]; ++idx) {
    const SelinvOp op = tasks[idx].op;
    if (op == SelinvOp::PHASE2_OFF || op == SelinvOp::PHASE2_DIAG) return idx;
  }
  return offsets[rank + 1];
}

void selinv_run_phase1_only(int rank, SelinvState* st) {
  const auto& tasks   = st->tasks->tasks;
  const auto& offsets = st->tasks->offsets;
  const int   start   = offsets[rank];
  const int   end     = find_phase2_start(*st, rank);
  for (int idx = start; idx < end; ++idx) {
    if (st->abort_flag.load(std::memory_order_relaxed)) return;
    const SelinvTask& t = tasks[idx];
    switch (t.op) {
      case SelinvOp::PHASE1_TRTRI: exec_phase1_trtri(t, st); break;
      case SelinvOp::PHASE1_TRSM:  exec_phase1_trsm (t, st); break;
      default: break;
    }
  }
}

void selinv_run_phase2_only(int rank, SelinvState* st) {
  const auto& tasks   = st->tasks->tasks;
  const auto& offsets = st->tasks->offsets;
  const int   start   = find_phase2_start(*st, rank);
  const int   end     = offsets[rank + 1];
  for (int idx = start; idx < end; ++idx) {
    if (st->abort_flag.load(std::memory_order_relaxed)) return;
    const SelinvTask& t = tasks[idx];
    switch (t.op) {
      case SelinvOp::PHASE2_OFF:  exec_phase2_off (t, st); break;
      case SelinvOp::PHASE2_DIAG: exec_phase2_diag(t, st); break;
      default: break;
    }
  }
}

}  // namespace

void selinv_run_parallel_pthreads(SelinvState& state) {
  const int n_ranks = static_cast<int>(state.tasks->offsets.size()) - 1;
  if (n_ranks <= 1) {
    selinv_run_phase1_only(0, &state);
    snapshot_M(state);
    selinv_run_phase2_only(0, &state);
    return;
  }
  // Phase 1 in parallel.
  std::vector<std::thread> threads;
  threads.reserve(n_ranks);
  for (int r = 0; r < n_ranks; ++r) {
    threads.emplace_back([r, &state]{
      unbind_from_creator();
      selinv_run_phase1_only(r, &state);
    });
  }
  for (auto& t : threads) t.join();
  threads.clear();

  // Hard barrier — copy M.
  if (state.abort_flag.load()) return;
  snapshot_M(state);

  // Phase 2 in parallel.
  for (int r = 0; r < n_ranks; ++r) {
    threads.emplace_back([r, &state]{
      unbind_from_creator();
      selinv_run_phase2_only(r, &state);
    });
  }
  for (auto& t : threads) t.join();
}

void selinv_run_parallel_omp(SelinvState& state) {
  const int n_ranks = static_cast<int>(state.tasks->offsets.size()) - 1;
  if (n_ranks <= 1) {
    selinv_run_phase1_only(0, &state);
    snapshot_M(state);
    selinv_run_phase2_only(0, &state);
    return;
  }
  // Nested-OMP guard — see factorize_run_parallel_omp for the same case.
  // If we are nested inside an outer OMP region with nested parallelism
  // disabled, the inner team collapses to 1 and the n_ranks partitioning
  // deadlocks. Route to the std::thread runner instead.
  if (omp_in_parallel() &&
      omp_get_max_active_levels() <= omp_get_active_level()) {
    selinv_run_parallel_pthreads(state);
    return;
  }
  #pragma omp parallel num_threads(n_ranks)
  {
    selinv_run_phase1_only(omp_get_thread_num(), &state);
    #pragma omp barrier
    #pragma omp single
    {
      snapshot_M(state);
    }
    // implicit barrier at end of single
    selinv_run_phase2_only(omp_get_thread_num(), &state);
  }
}

}}  // namespace sTiles::sparse
