/**
 * @file api.cpp
 * @brief Implementation of the sTiles_sparse_* C-ABI surface.
 *
 * Wraps the in-namespace `sTiles::sparse` library (etree / symbolic /
 * supernode / executor / solve / selinv) behind opaque `void*` handles so
 * sTiles' tile-DAG pipeline can route to it via tile_type_mode == 2.
 *
 * Conventions (from the underlying module headers):
 *   - 1-based internal indices (elimination tree, permutation, symbolic, row_pattern, …).
 *   - Element accessors (`sps_get_chol_elm` / `sps_get_selinv_elm`) are
 *     0-based at the public surface — sTiles' bench passes 0-based indices.
 *   - `CscLower` stores the lower triangle, 1-based colptr/rowind. We
 *     accept 0-based COO from sTiles' `assign_graph` and convert here.
 *
 * Permutation handoff: the permutation always comes from sTiles' ordering
 * competition in tools/ordering/stiles_ordering.hpp (run by
 * `preprocess_primary_sparse`). The `element_perm` it produces is what we
 * receive in `sTiles_sparse_set_user_permutation`; the sparse module never
 * runs its own ordering.
 *
 * @project sTiles (Sparse Tiles Library)
 * @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @version 3.0.0
 * @date 1 1 2026
 * @license Proprietary
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
 */

#include "api.hpp"

#include "collect.hpp"
#include "element_access.hpp"
#include "etree.hpp"
#include "executor.hpp"
#include "selinv.hpp"
#include "solve.hpp"
#include "supernode.hpp"
#include "symbolic.hpp"

#include "../common/stiles_logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>   // getenv/atoi for STILES_SPARSE_CSC_W
#include <new>
#include <vector>

#include <omp.h>

namespace {

using sTiles::sparse::CellStore;
using sTiles::sparse::CollectedTasks;
using sTiles::sparse::CscLower;
using sTiles::sparse::Idx;
using sTiles::sparse::Int;
using sTiles::sparse::Permutation;
using sTiles::sparse::Ptr;
using sTiles::sparse::SpsState;
using sTiles::sparse::Symbolic;
using sTiles::sparse::SymbolicOptions;

// Per-handle state. Allocated by sTiles_sparse_create, freed by freeGroup.
struct Handle {
    Int  n          = 0;
    Int  nnz_input  = 0;        // user-provided COO nnz (one triangle)
    int  num_cores  = 1;
    int  max_snode  = -1;       // -1 = library default
    int  group_id   = -1;       // sTiles group_index for memory accounting

    bool have_perm  = false;
    bool have_graph = false;
    bool have_vals  = false;
    bool factored   = false;
    bool selinved   = false;

    // Permutation supplied by sTiles' ordering pipeline (1-based, as the
    // sparse module expects). Never produced by the module itself — sTiles
    // runs the ordering competition (RCM/ND/SCOTCH/AMD/...) in
    // tools/ordering/stiles_ordering.hpp and hands the winner to us.
    Permutation     ordering;

    // Lower-triangular CSC built from the user's COO. Pattern is built once
    // at assign_graph; nzval is refreshed at every assign_values.
    CscLower        A_lower;

    // Cached COO arrays (pointers owned by the caller — sTiles_call holds
    // them for the lifetime of the scheme). Used by assign_values to
    // re-scatter values into A_lower.nzval in CSC order.
    const int*      coo_row = nullptr;
    const int*      coo_col = nullptr;
    int             coo_nnz = 0;

    // Position of each COO entry in the lower-triangular CSC arena
    // (lex sorted). -1 marks entries we drop (strict-upper or out-of-range).
    // Built once at assign_graph alongside A_lower; re-applied at every
    // assign_values to splat values without re-sorting.
    std::vector<Ptr> coo_to_csc_pos;

    // Flat scatter plan for the CSC -> supernodal-arena load. Built once (first
    // assign_values after the arena exists) and reused across every
    // re-factorization: geometry is fixed, only the values change. Replaces the
    // per-entry index math in CellStore::load_from_csc(). Invalidated whenever
    // the symbolic pattern is rebuilt (assign_graph).
    std::vector<Ptr> load_map;
    bool            load_map_built = false;

    // Symbolic factorization (etree + supernodes + row_pattern).
    Symbolic        sym;

    // Cell stores. L_cs holds the factor; Z_cs is allocated lazily by selinv.
    CellStore       L_cs;
    CellStore       Z_cs;

    // Executor state + task list (built once after symbolic).
    SpsState        state;
    CollectedTasks  tasks;
    bool            tasks_collected = false;

    // Selinv parallel state (built lazily on first selinv call).
    sTiles::sparse::SelinvState           selinv_state;
    sTiles::sparse::CollectedSelinvTasks  selinv_tasks;
    bool            selinv_tasks_collected = false;

    // Solve cache: ColIndex is built once per symbolic pattern; scratch
    // buffers persist across solve calls. Invalidated when the symbolic
    // pattern changes (e.g., assign_graph rebuilds), reused as-is across
    // numerical re-factorizations (same supernode geometry → same indices).
    sTiles::sparse::ColIndex     solve_col_index;
    sTiles::sparse::SolveScratch solve_scratch;
    bool            solve_col_index_built = false;

    // Etree-level schedule for parallel solve. Same lifecycle as solve_col_index
    // (built once per symbolic pattern). Drives the level-set parallel
    // forward/backward substitution when num_cores > 1.
    sTiles::sparse::EtreeSchedule solve_schedule;
    bool            solve_schedule_built = false;

    // Packed flat-CSC path for nrhs==1: auto-selected when supernodes are thin
    // (mean width small), where the supernodal cell-walk overhead dominates and
    // the flat scalar sweep wins. `packed_csc` is rebuilt whenever the factor
    // changes (invalidated at factorize); `packed_use` is decided once.
    sTiles::sparse::PackedCsc packed_csc;
    int             packed_use = -1;   // -1 undecided, 0 supernodal, 1 packed-csc
};

inline Handle* as_handle(void** obj) {
    return (obj && *obj) ? static_cast<Handle*>(*obj) : nullptr;
}

constexpr int kErrNullHandle = -1;
constexpr int kErrInvalidArg = -2;
constexpr int kErrInternal   = -3;

// Build a 1-based lower-triangular CSC from a 0-based COO triangle. Records
// the per-COO-entry position in `pos_out` (length `nnz`, -1 for dropped
// entries) so a subsequent assign_values pass can splat numeric values
// without re-sorting.
//
// Pattern only: pass val=nullptr; nzval is left empty and sized by the
// caller before splatting.
void coo_to_csc_lower(int n, int nnz,
                      const int* row, const int* col,
                      CscLower& out,
                      std::vector<Ptr>& pos_out) {
    pos_out.assign(static_cast<std::size_t>(nnz), -1);

    // Count entries per column (lower triangle only).
    std::vector<Ptr> col_count(static_cast<std::size_t>(n) + 1, 0);
    for (int k = 0; k < nnz; ++k) {
        const int r = row[k];
        const int c = col[k];
        if (r < 0 || c < 0 || r >= n || c >= n) continue;
        if (r < c) continue;          // skip strict-upper
        col_count[static_cast<std::size_t>(c) + 1]++;
    }
    // Prefix-sum into colptr (1-based).
    out.size     = static_cast<Int>(n);
    out.expanded = false;
    out.colptr.assign(static_cast<std::size_t>(n) + 1, 1);
    for (int j = 0; j < n; ++j) {
        out.colptr[static_cast<std::size_t>(j) + 1] =
            out.colptr[static_cast<std::size_t>(j)] +
            col_count[static_cast<std::size_t>(j) + 1];
    }
    const Ptr total = out.colptr[static_cast<std::size_t>(n)] - 1;
    out.rowind.assign(static_cast<std::size_t>(total), 0);
    out.nzval.clear();

    // Bucket-fill rowind, recording the per-COO position and its inverse
    // (slot -> originating COO index, used by the per-column sort below).
    std::vector<Ptr> pos(static_cast<std::size_t>(n), 0);
    for (int j = 0; j < n; ++j) pos[static_cast<std::size_t>(j)] = out.colptr[static_cast<std::size_t>(j)] - 1;
    std::vector<int> inv_pos(static_cast<std::size_t>(total), -1);
    for (int k = 0; k < nnz; ++k) {
        const int r = row[k];
        const int c = col[k];
        if (r < 0 || c < 0 || r >= n || c >= n) continue;
        if (r < c) continue;
        const Ptr p = pos[static_cast<std::size_t>(c)]++;
        out.rowind[static_cast<std::size_t>(p)] = static_cast<Idx>(r + 1);  // 1-based
        pos_out[static_cast<std::size_t>(k)] = p;
        inv_pos[static_cast<std::size_t>(p)] = k;
    }

    // Sort row indices within each column ascending. Use inv_pos[p] to find
    // the originating COO index, so the pos_out remap is O(nnz_in_column)
    // per column instead of O(nnz) per column.
    for (int j = 0; j < n; ++j) {
        const Ptr s = out.colptr[static_cast<std::size_t>(j)] - 1;
        const Ptr e = out.colptr[static_cast<std::size_t>(j) + 1] - 1;
        if (e - s <= 1) continue;
        const std::size_t len = static_cast<std::size_t>(e - s);
        // Indirect sort: build new order, then permute rowind and remap pos_out.
        std::vector<Ptr> idx(len);
        for (Ptr i = 0; i < e - s; ++i) idx[static_cast<std::size_t>(i)] = s + i;
        std::sort(idx.begin(), idx.end(),
                  [&](Ptr a, Ptr b){ return out.rowind[a] < out.rowind[b]; });
        // Apply: new_rowind[s+i] = old_rowind[idx[i]]; remap is old → new.
        std::vector<Idx> rs(len);
        std::vector<Ptr> remap(len);
        for (Ptr i = 0; i < e - s; ++i) {
            rs[static_cast<std::size_t>(i)] = out.rowind[idx[static_cast<std::size_t>(i)]];
            remap[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)] - s)] = s + i;
        }
        for (Ptr i = 0; i < e - s; ++i) out.rowind[s + i] = rs[static_cast<std::size_t>(i)];
        // Re-target only the pos_out entries that live in this column.
        for (Ptr p = s; p < e; ++p) {
            const int k = inv_pos[static_cast<std::size_t>(p)];
            if (k >= 0) pos_out[static_cast<std::size_t>(k)] = remap[static_cast<std::size_t>(p - s)];
        }
    }
}

} // namespace

namespace sTiles { namespace sparse { namespace api {

// -----------------------------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------------------------

int  init()  { return 0; }
void quit()  {}

int create(void** obj, int num_cores) {
    if (!obj) return kErrInvalidArg;
    auto* h = new (std::nothrow) Handle();
    if (!h) return kErrInvalidArg;
    h->num_cores = (num_cores > 0) ? num_cores : 1;
    *obj = h;
    return 0;
}

void freeGroup(void** obj) {
    if (!obj || !*obj) return;
    delete static_cast<Handle*>(*obj);
    *obj = nullptr;
}

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

int set_user_permutation(void** obj, const int* perm, int n) {
    Handle* h = as_handle(obj);
    if (!h)              return kErrNullHandle;
    if (!perm || n <= 0) return kErrInvalidArg;

    // sTiles' element_perm comes from stiles_ordering.hpp's competition;
    // its convention is `perm[old_index] = new_index` (0-based). The sparse
    // module wants 1-based:
    //   invp[k-1] = new index of original node k       == perm[old]+1
    //   perm[i-1] = original index of i-th new node    == inverse of above
    h->ordering.invp.assign(static_cast<std::size_t>(n), 0);
    h->ordering.perm.assign(static_cast<std::size_t>(n), 0);
    for (int k = 0; k < n; ++k) {
        const int new_idx = perm[k];
        if (new_idx < 0 || new_idx >= n) return kErrInvalidArg;
        h->ordering.invp[static_cast<std::size_t>(k)]       = static_cast<Int>(new_idx + 1);
        h->ordering.perm[static_cast<std::size_t>(new_idx)] = static_cast<Int>(k + 1);
    }
    if (!h->ordering.validate()) {
        sTiles::Logger::error("sTiles_sparse_set_user_permutation: not a valid permutation");
        return kErrInvalidArg;
    }

    h->n         = static_cast<Int>(n);
    h->have_perm = true;
    return 0;
}

void set_max_supernode(void** obj, int max_snode) {
    if (Handle* h = as_handle(obj)) h->max_snode = max_snode;
}

void set_group_id(void** obj, int group_id) {
    if (Handle* h = as_handle(obj)) h->group_id = group_id;
}

// -----------------------------------------------------------------------------
// Symbolic + numeric phases
// -----------------------------------------------------------------------------

int assign_graph(void** obj, int n, int nnz,
                 const int* row, const int* col) {
    Handle* h = as_handle(obj);
    if (!h)                                    return kErrNullHandle;
    if (n <= 0 || nnz <= 0 || !row || !col)    return kErrInvalidArg;
    if (!h->have_perm)                         return kErrInvalidArg;
    if (n != h->n) {
        sTiles::Logger::error("sTiles_sparse_assign_graph: n=", n,
                              " disagrees with permutation size=", h->n);
        return kErrInvalidArg;
    }

    // Cache COO pointers for assign_values.
    h->coo_row = row;
    h->coo_col = col;
    h->coo_nnz = nnz;

    // Structure is being (re)built -> any cached CSC->arena scatter plan is stale.
    h->load_map_built = false;
    h->load_map.clear();

    const double t_coo = omp_get_wtime();
    // Build pattern-only CSC and record per-COO target positions.
    coo_to_csc_lower(n, nnz, row, col, h->A_lower, h->coo_to_csc_pos);
    sTiles::Logger::timing("│   ↪ Build sparse pattern (CSC): ", (omp_get_wtime() - t_coo), " s");

    const double t_sym = omp_get_wtime();
    // Symbolic factorization: composes user perm with post-order + sort.
    SymbolicOptions opts;
    if (h->max_snode > 0) opts.relax_max_size = static_cast<Int>(h->max_snode);
    try {
        sTiles::sparse::compute_symbolic(h->A_lower, h->ordering, opts, h->sym);
    } catch (const std::exception& e) {
        sTiles::Logger::error("sTiles_sparse_assign_graph: compute_symbolic threw: ", e.what());
        return kErrInternal;
    }
    sTiles::Logger::timing("│   ↪ Compute supernodal symbolic: ", (omp_get_wtime() - t_sym), " s");

    const double t_alloc = omp_get_wtime();
    // Allocate the L cell store and build the executor task list (the task
    // partition depends on num_cores, which is fixed at create time).
    h->L_cs.allocate(h->sym, h->group_id);
    sTiles::Logger::timing("│   ↪ Allocate L cell store: ", (omp_get_wtime() - t_alloc), " s");

    // Task-partition width: clamp the rank count by problem size before ANY
    // task collection (numeric here, selinv/solve reuse h->num_cores later).
    // Partitioning a tiny factor across many ranks makes cross-rank
    // dependency spin-waits dwarf the numeric work (a 577-var / ~30k-nnz
    // factor took ~1.5 s per chol at 8 ranks vs ~0.01 s serial). Require
    // enough work per rank: >= STILES_SPARSE_NNZ_PER_RANK nnz(L) (default
    // 200k) and >= 4 supernodes per rank; big factors keep the full width.
    {
        long long per_rank = 200000;
        if (const char* e = std::getenv("STILES_SPARSE_NNZ_PER_RANK")) {
            const long long v = std::atoll(e);
            if (v > 0) per_rank = v;
        }
        const long long nnzL = static_cast<long long>(h->sym.nnz_l);
        int cap = static_cast<int>(std::min(nnzL / per_rank,
                                            static_cast<long long>(h->sym.n_super / 4)));
        if (cap < 1) cap = 1;
        if (h->num_cores > cap) {
            sTiles::Logger::timing("│   ↪ Sparse ranks clamped ", h->num_cores, " → ", cap,
                                   " (nnz(L)=", nnzL, ", supernodes=", h->sym.n_super, ")");
            h->num_cores = cap;
        }
    }

    const double t_tasks = omp_get_wtime();
    sTiles::sparse::collect_tasks(h->sym, h->L_cs, h->num_cores, h->tasks);
    h->tasks_collected = true;
    sTiles::Logger::timing("│   ↪ Collect numeric tasks: ", (omp_get_wtime() - t_tasks), " s");

    h->nnz_input  = static_cast<Int>(nnz);
    h->have_graph = true;
    return 0;
}

int assign_values(void** obj, const double* values) {
    Handle* h = as_handle(obj);
    if (!h)             return kErrNullHandle;
    if (!values)        return kErrInvalidArg;
    if (!h->have_graph) return kErrInvalidArg;

    const double t_assign = omp_get_wtime();
    // Splat values into A_lower.nzval using the per-COO position map built
    // at assign_graph. Entries that were dropped (strict-upper or out-of-
    // range) have pos == -1; we ignore them.
    const Ptr total = static_cast<Ptr>(h->A_lower.rowind.size());
    h->A_lower.nzval.assign(static_cast<std::size_t>(total), 0.0);
    for (int k = 0; k < h->coo_nnz; ++k) {
        const Ptr p = h->coo_to_csc_pos[static_cast<std::size_t>(k)];
        if (p >= 0) h->A_lower.nzval[static_cast<std::size_t>(p)] = values[k];
    }

    // Zero the existing arena instead of re-allocating — the cell layout
    // built at assign_graph is still valid, only nzval needs to be reset.
    if (h->L_cs.arena_size() > 0 && h->L_cs.arena_data() != nullptr) {
        std::fill_n(h->L_cs.arena_data(), h->L_cs.arena_size(), 0.0);
        // Cached CSC->arena scatter: build the destination plan once (the
        // geometry never changes across re-factorizations), then splat. This
        // replaces load_from_csc()'s per-entry invp/supernode/find/lower_bound
        // work with a single indexed store per nonzero.
        if (!h->load_map_built) {
            h->L_cs.build_load_map(h->A_lower, h->sym, h->load_map);
            h->load_map_built = true;
        }
        h->L_cs.load_from_csc_mapped(h->A_lower, h->load_map);
    } else {
        // Degenerate/empty arena — fall back to the direct path.
        h->L_cs.load_from_csc(h->A_lower, h->sym);
    }
    (void)t_assign;

    h->have_vals = true;
    h->factored  = false;
    h->selinved  = false;
    return 0;
}

namespace {
int chol_common(Handle* h, bool use_omp) {
    if (!h)             return kErrNullHandle;
    if (!h->have_vals)  return kErrInvalidArg;

    // Wire executor state pointers and (re)set per-call counters.
    h->state.symbolic = &h->sym;
    h->state.cells    = &h->L_cs;
    h->state.tasks    = &h->tasks;

    sTiles::sparse::prepare(h->state);
    if (use_omp) {
        sTiles::sparse::factorize_run_parallel_omp(h->state);
    } else {
        sTiles::sparse::factorize_run_parallel_pthreads(h->state);
    }

    if (h->state.abort_flag.load()) {
        sTiles::Logger::error("sTiles::sparse::api::chol: matrix not positive-definite (abort_flag set).");
        return kErrInternal;
    }
    h->factored = true;
    h->packed_csc.values_built = false;   // factor changed → refresh CSC values (structure kept)
    return 0;
}
} // namespace

int chol_pthreads(void** obj) { return chol_common(as_handle(obj), false); }
int chol_omp     (void** obj) { return chol_common(as_handle(obj), true);  }

namespace {
int selinv_common(Handle* h, bool use_omp) {
    if (!h)            return kErrNullHandle;
    if (!h->factored)  return kErrInvalidArg;

    // Allocate Z to mirror L's cell layout (zero-initialized arena).
    h->Z_cs.allocate_like(h->L_cs, h->group_id);

    // Collect parallel selinv tasks (one-shot for this call: dependencies
    // and the M arena are call-local).
    sTiles::sparse::collect_selinv_tasks(h->sym, h->Z_cs, h->num_cores,
                                         h->selinv_tasks);
    h->selinv_tasks_collected = true;

    h->selinv_state.L_cs     = &h->L_cs;
    h->selinv_state.Z_cs     = &h->Z_cs;
    h->selinv_state.sym      = &h->sym;
    h->selinv_state.tasks    = &h->selinv_tasks;
    h->selinv_state.group_id = h->group_id;

    try {
        sTiles::sparse::selinv_prepare(h->selinv_state);
        if (use_omp) {
            sTiles::sparse::selinv_run_parallel_omp(h->selinv_state);
        } else {
            sTiles::sparse::selinv_run_parallel_pthreads(h->selinv_state);
        }
    } catch (const std::exception& e) {
        sTiles::Logger::error("sTiles::sparse::api::selinv threw: ", e.what());
        return kErrInternal;
    }
    if (h->selinv_state.abort_flag.load()) {
        sTiles::Logger::error("sTiles::sparse::api::selinv: abort flag set "
                              "(numerical failure inside selinv kernel).");
        return kErrInternal;
    }
    h->selinved = true;
    return 0;
}
} // namespace

int selinv_pthreads(void** obj) { return selinv_common(as_handle(obj), false); }
int selinv_omp     (void** obj) { return selinv_common(as_handle(obj), true);  }

// Lazily build the per-symbolic solve caches (ColIndex + EtreeSchedule).
// Idempotent: once built, both are reused across all subsequent solves
// until the symbolic pattern changes.
static inline void ensure_solve_cache(Handle* h) {
    if (!h->solve_col_index_built) {
        h->solve_col_index = sTiles::sparse::build_col_index(h->sym, h->L_cs);
        h->solve_col_index_built = true;
    }
    if (!h->solve_schedule_built) {
        h->solve_schedule = sTiles::sparse::build_etree_schedule(h->sym);
        h->solve_schedule_built = true;
    }
}

// Route the sparse solve through the packed flat-CSC path when the factor is
// thin (mean supernode width ≤ gate) and 1 ≤ nrhs ≤ cap. solve_type: 0=forward,
// 1=backward, 2=full. Returns true if handled; false → caller uses supernodal.
// Auto-switches: thin factors → packed (the flat sweep beats the supernodal
// cell walk); chunky → false. Cap = tile_size (STILES_SPARSE_CSC_NRHS overrides;
// STILES_SPARSE_CSC_W the width gate, 0 disables). Shared by solve_LLT/L/LT so
// forward-only and backward-only get the packed speedup too.
static bool packed_solve_if_eligible(Handle* h, double* b, int nrhs, int ldb,
                                     int tile_size, int solve_type) {
    if (h->packed_use < 0) {
        static const int W_GATE = [](){
            const char* e = std::getenv("STILES_SPARSE_CSC_W");
            return e ? std::atoi(e) : 16;
        }();
        const double mean_w = (h->sym.n_super > 0)
            ? static_cast<double>(h->sym.n) / static_cast<double>(h->sym.n_super)
            : 1e18;
        h->packed_use = (W_GATE > 0 && mean_w <= static_cast<double>(W_GATE)) ? 1 : 0;
    }
    if (h->packed_use != 1) return false;
    static const int env_cap = [](){
        const char* e = std::getenv("STILES_SPARSE_CSC_NRHS");
        return e ? std::atoi(e) : -1;
    }();
    const int nrhs_cap = (env_cap >= 0) ? env_cap : (tile_size > 0 ? tile_size : 64);
    if (nrhs < 1 || nrhs > nrhs_cap) return false;

    if (std::getenv("STILES_DEBUG_ORDERING")) {
        const auto& invp = h->sym.ordering.invp;
        long moved = 0;
        for (std::size_t k = 0; k < invp.size(); ++k)
            if (invp[k] != static_cast<long>(k) + 1) ++moved;
        std::fprintf(stderr, "[order-probe] sparse packed solve: n=%d perm_moved=%ld/%zu "
                     "(0=identity) nrhs=%d solve_type=%d\n",
                     h->sym.n, moved, invp.size(), nrhs, solve_type);
    }

    if (!h->packed_csc.structure_built)
        sTiles::sparse::build_packed_csc(h->sym, h->L_cs, h->solve_col_index, h->packed_csc);
    else if (!h->packed_csc.values_built)
        sTiles::sparse::refresh_packed_csc_values(h->packed_csc);   // cheap per-factor gather

    if (nrhs >= 2) {
        sTiles::sparse::solve_packed_multi(h->sym, h->packed_csc, h->solve_scratch,
                                           b, nrhs, static_cast<int64_t>(ldb), solve_type);
    } else {
        static const bool csc_par = [](){
            const char* e = std::getenv("STILES_SPARSE_CSC_PAR");
            return e && std::atoi(e) != 0;
        }();
        if (csc_par && h->num_cores > 1)
            sTiles::sparse::solve_packed_par(h->sym, h->packed_csc, h->solve_schedule,
                                             h->solve_scratch, b, static_cast<int64_t>(ldb),
                                             h->num_cores, solve_type);
        else
            sTiles::sparse::solve_packed(h->sym, h->packed_csc, h->solve_scratch,
                                         b, static_cast<int64_t>(ldb), solve_type);
    }
    return true;
}

int solve_LLT(void** obj, double* b, int nrhs, int ldb, int tile_size) {
    Handle* h = as_handle(obj);
    if (!h)                          return kErrNullHandle;
    if (!b || nrhs <= 0 || ldb <= 0) return kErrInvalidArg;
    if (!h->factored)                return kErrInvalidArg;
    ensure_solve_cache(h);
    if (packed_solve_if_eligible(h, b, nrhs, ldb, tile_size, 2)) return 0;
    sTiles::sparse::solve(h->sym, h->L_cs, h->solve_col_index,
                          h->solve_schedule, h->solve_scratch,
                          b, nrhs, static_cast<int64_t>(ldb), h->num_cores);
    return 0;
}

int solve_L(void** obj, double* b, int nrhs, int ldb, int tile_size) {
    Handle* h = as_handle(obj);
    if (!h)                          return kErrNullHandle;
    if (!b || nrhs <= 0 || ldb <= 0) return kErrInvalidArg;
    if (!h->factored)                return kErrInvalidArg;
    ensure_solve_cache(h);
    if (packed_solve_if_eligible(h, b, nrhs, ldb, tile_size, 0)) return 0;
    sTiles::sparse::solve_forward(h->sym, h->L_cs, h->solve_col_index,
                                  h->solve_schedule, h->solve_scratch,
                                  b, nrhs, static_cast<int64_t>(ldb), h->num_cores);
    return 0;
}

int solve_LT(void** obj, double* b, int nrhs, int ldb, int tile_size) {
    Handle* h = as_handle(obj);
    if (!h)                          return kErrNullHandle;
    if (!b || nrhs <= 0 || ldb <= 0) return kErrInvalidArg;
    if (!h->factored)                return kErrInvalidArg;
    ensure_solve_cache(h);
    if (packed_solve_if_eligible(h, b, nrhs, ldb, tile_size, 1)) return 0;
    sTiles::sparse::solve_backward(h->sym, h->L_cs, h->solve_col_index,
                                   h->solve_schedule, h->solve_scratch,
                                   b, nrhs, static_cast<int64_t>(ldb), h->num_cores);
    return 0;
}

int clear_selinv(void** obj) {
    Handle* h = as_handle(obj);
    if (!h) return kErrNullHandle;
    if (h->Z_cs.arena_size() > 0 && h->Z_cs.arena_data() != nullptr) {
        std::fill_n(h->Z_cs.arena_data(), h->Z_cs.arena_size(), 0.0);
    }
    h->selinved = false;
    return 0;
}

// -----------------------------------------------------------------------------
// Element access
// -----------------------------------------------------------------------------

double get_chol_elm(void** obj, int i, int j) {
    Handle* h = as_handle(obj);
    if (!h || !h->factored) return 0.0;
    return sTiles::sparse::sps_get_chol_elm(h->sym, h->L_cs, i, j);
}

double get_selinv_elm(void** obj, int i, int j) {
    Handle* h = as_handle(obj);
    if (!h || !h->selinved) return 0.0;
    return sTiles::sparse::sps_get_selinv_elm(h->sym, h->Z_cs, i, j);
}

double get_logdet(void** obj) {
    Handle* h = as_handle(obj);
    if (!h || !h->factored) return 0.0;
    // 2 * Σ log(L_ii) over the diagonal cells of every supernode.
    double s = 0.0;
    const Symbolic& sym = h->sym;
    const Int n_super = sym.n_super;
    for (Int I = 1; I <= n_super; ++I) {
        const auto* cell = h->L_cs.find(I, I);
        if (!cell) continue;
        const Int rows = cell->rows;
        const Int cols = cell->cols;
        for (Int r = 0; r < cols; ++r) {
            const std::size_t off = static_cast<std::size_t>(r)
                                  + static_cast<std::size_t>(rows)
                                  * static_cast<std::size_t>(r);
            const double d = cell->nzval[off];
            if (d > 0.0) s += std::log(d);
        }
    }
    return 2.0 * s;
}

long long get_nnz_factor(void** obj) {
    Handle* h = as_handle(obj);
    if (!h || !h->factored) return 0;
    return static_cast<long long>(h->sym.nnz_l);
}

}}} // namespace sTiles::sparse::api
