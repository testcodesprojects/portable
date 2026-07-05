#pragma once
/**
 * @file chol_sync_policy.hpp
 * @brief Sync-primitive policies for sharing chol worker bodies between the
 *        pthreads and OMP backends.
 *
 * Each policy struct wraps the backend-specific sync macros (ss_* for
 * pthreads, dep_* for OMP) behind a uniform interface. A templated worker
 * body uses the policy methods instead of macros directly, so a single
 * implementation supports both backends.
 *
 * The policy methods alias the held sync-state pointer to the name the
 * underlying macro expects (`stile` or `dep_tracker`), so the macros find
 * their state through normal name lookup. With -O2/-O3 the methods inline
 * away — generated assembly is identical to direct macro use.
 *
 * Adding a new sync primitive: implement it on BOTH policies (compile error
 * if you forget). No worker needs editing.
 */

#include "../control/common.h"          // ss_init / ss_cond_wait / ss_cond_set / ss_abort / ss_aborted / ss_finalize
#include "../control/stiles_control.hpp" // dep_init / dep_wait_for / dep_set_done / dep_abort_all / dep_is_aborted / dep_finalize

namespace sTiles { namespace detail {

// ─── pthreads backend ─────────────────────────────────────────────────────────
// State is `stiles_context_t* stile` from the bound thread pool. The ss_*
// macros (defined in common.h) reference a local variable named `stile`,
// so each method aliases its member to that name before invoking the macro.
//
// NOTE: `wait` returns bool because the underlying ss_cond_wait macro ends
// with `if (stile->ss_abort) break;` — that `break` only makes sense inside
// a loop/switch at the call site. The policy method instead inlines the
// wait loop and returns false on abort, so callers can `if (!sync.wait(...)) break;`.
struct PthreadsSync {
    stiles_context_t* ctx;

    int  rank()       const { stiles_context_t* stile = ctx; return STILES_RANK; }
    bool is_aborted() const { stiles_context_t* stile = ctx; return ss_aborted(); }

    void init(int m, int n, int init_val) {
        stiles_context_t* stile = ctx;
        ss_init(m, n, init_val);
    }
    // Returns true if the wait completed (predicate satisfied); false if
    // aborted while spinning. Callers should `break` on false to mirror the
    // original macro's behaviour of bailing out of the enclosing switch case.
    bool wait(int m, int n, int val) {
        stiles_context_t* stile = ctx;
        STILES_WAIT_LOOP(stile->ss_progress[(m) + stile->ss_ld * (n)] != (val))
        return !stile->ss_abort;
    }
    void set_done(int m, int n, int val) {
        stiles_context_t* stile = ctx;
        ss_cond_set(m, n, val);
    }
    void abort_all() {
        stiles_context_t* stile = ctx;
        ss_abort();
    }
    void finalize() {
        stiles_context_t* stile = ctx;
        ss_finalize();
    }
};

// ─── OMP backend ──────────────────────────────────────────────────────────────
// State is `omp_dep_tracker_t* dep_tracker` set up by the OMP parallel region.
// `worldsize` is captured but unused inside the body — kept so it's available
// to callers that need it for OMP-region setup.
struct OmpSync {
    omp_dep_tracker_t* ctx;
    int                worldsize;

    int  rank()       const { return omp_get_thread_num(); }
    bool is_aborted() const { omp_dep_tracker_t* dep_tracker = ctx; return dep_is_aborted(); }

    void init(int m, int n, int init_val) {
        omp_dep_tracker_t* dep_tracker = ctx;
        dep_init(m, n, init_val);
    }
    // Inlined wait loop (same shape as dep_wait_for but returning bool so the
    // policy doesn't need a `break` that would escape its function scope).
    bool wait(int m, int n, int val) {
        omp_dep_tracker_t* dep_tracker = ctx;
        int _spin_count = 0;
        while (dep_tracker->progress_table[(m) + dep_tracker->ld * (n)].load(std::memory_order_acquire) != (val)) {
            if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break;
            hpc_pause_hybrid(_spin_count);
        }
        return !dep_tracker->abort_flag.load(std::memory_order_relaxed);
    }
    void set_done(int m, int n, int val) {
        omp_dep_tracker_t* dep_tracker = ctx;
        dep_set_done(m, n, val);
    }
    void abort_all() {
        omp_dep_tracker_t* dep_tracker = ctx;
        dep_abort_all();
    }
    void finalize() {
        omp_dep_tracker_t* dep_tracker = ctx;
        dep_finalize();
    }
};

}} // namespace sTiles::detail
