/**
 * @file  csc_solve.cpp
 * @brief Pack Cholesky factor values into CSC layout and serial CSC solve.
 *
 * After sTiles_chol completes, pack_L_values() extracts tile values into the
 * flat L_values[] array that matches L_colptr/L_rowind.  csc_dtrsm() then
 * performs the full L * L^T solve in O(nnz(L)) scalar operations — no BLAS,
 * no tile overhead — which dominates for nrhs==1.
 *
 * Supported tile storage modes:
 *   Dense (denseTiles):      column-major tiles
 *   Semisparse (chunkedDenseTiles):
 *     diagonal tiles  — banded column-major with bandwidth kd = upper_bw
 *     off-diagonal    — column-compressed: height × active_cols, ld = height
 */

#include "../common/stiles_structs.hpp"
#include "../control/common.h"          // STILES_RANK, STILES_SIZE
#include "../control/stiles_dispatch.h" // sTiles::parallel_call, unpack_args
#include "../dot/sparse-dot.h"          // stiles_sparse_ddot (gather-dot)
#include "../tile/meta.hpp"
#include "../common/stiles_multiversion.hpp"   // STILES_MULTIVERSION (FMV: AVX-512 clone)
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <new>
#include <omp.h>
#include <thread>
#include <vector>

extern "C" int* sTiles_get_params();

namespace sTiles {

// Runtime-tunable memory ceiling for the L_src pointer table. Above this many
// bytes, _pack_prep_once skips the build and pack falls back to the per-entry
// kernel. Public via sTiles_get/set_pack_cache_threshold_bytes — exposed for
// (a) memory-tight nodes that need a lower threshold and (b) correctness tests
// that need to force-disable the fast path by setting it to 0.
static long long g_l_src_max_bytes = 2LL << 30;  // 2 GiB default

long long get_pack_cache_threshold_bytes() { return g_l_src_max_bytes; }
void      set_pack_cache_threshold_bytes(long long b) { g_l_src_max_bytes = b; }

namespace {

// Per-column body, shared by all backends. Reads tile data for
// L_colptr[j..j+1) and writes into scheme->L_values at the same span.
// All inputs are passed in to keep the worker free of TiledMatrix lookups
// that the compiler would otherwise emit per-iteration.
inline void pack_L_columns_range_org(TiledMatrix* scheme,
                                 const bool has_dense, const bool has_semi,
                                 const int j_lo, const int j_hi,
                                 const int N, const int ts, const int nt,
                                 const int Adesc_lm1) {
    double*    Lv  = scheme->L_values;
    const int64_t* Lcp = scheme->L_colptr;
    const int* Lri = scheme->L_rowind;

    for (int j = j_lo; j < j_hi; ++j) {
        const int tr = j / ts;
        const int lj = j % ts;
        const int ld = (tr < Adesc_lm1) ? ts : (N % ts);

        for (int64_t ptr = Lcp[j]; ptr < Lcp[j + 1]; ++ptr) {
            const int i        = Lri[ptr];
            const int tc       = i / ts;
            const int li       = i % ts;
            const int tile_idx = scheme->mapper.map_ij(tr, tc, nt);

            if (tile_idx < 0) { Lv[ptr] = 0.0; continue; }

            if (has_semi && scheme->chunkedDenseTiles[tile_idx]) {
                const double* tile = scheme->chunkedDenseTiles[tile_idx];
                if (tr == tc) {
                    const int kd = scheme->semisparseTileMetaCore[tile_idx].upper_bw;
                    Lv[ptr] = tile[(kd + lj - li) + li * (kd + 1)];
                } else {
                    const SemisparseTileMetaCore& semi = scheme->semisparseTileMetaCore[tile_idx];
                    const int ki = (li < (int)semi.acol.size()) ? semi.acol[li] : -1;
                    if (ki < 0) { Lv[ptr] = 0.0; continue; }
                    const int h = scheme->tileMetaCore[tile_idx].height;
                    Lv[ptr] = tile[lj + ki * h];
                }
            } else if (has_dense && scheme->denseTiles[tile_idx]) {
                Lv[ptr] = scheme->denseTiles[tile_idx][lj + li * ld];
            } else {
                Lv[ptr] = 0.0;
            }
        }
    }
}


// Hot pack body. All per-iteration decision-making (tile-kind dispatch,
// index math, mapper lookup) is collapsed into scheme->L_src at build
// time; this function is the steady-state path for every pack after the
// first. The unused signature args (has_dense/has_semi/N/ts/nt/Adesc_lm1)
// are retained so the callers in ppack/pack_L_omp/pack_L_pthreads_stdthread
// keep their existing parallel_call argument packing.
inline void pack_L_columns_range(TiledMatrix* scheme,
                                 const bool /*has_dense*/, const bool /*has_semi*/,
                                 const int j_lo, const int j_hi,
                                 const int /*N*/, const int /*ts*/, const int /*nt*/,
                                 const int /*Adesc_lm1*/) {
    double* __restrict__         Lv  = scheme->L_values;
    const double* const* __restrict__ src = scheme->L_src;
    const int64_t*                   Lcp = scheme->L_colptr;
    const int64_t lo  = Lcp[j_lo];
    const int64_t hi  = Lcp[j_hi];

    for (int64_t ptr = lo; ptr < hi; ++ptr) Lv[ptr] = *src[ptr];
}

// Dispatch helper: pick the fast L_src kernel when the precomputed table
// exists, otherwise fall back to the per-entry kernel. The decision is taken
// once per worker chunk, not per CSC entry — free at runtime, lets pack
// gracefully degrade for matrices whose L_src would have exceeded the memory
// ceiling in _pack_prep_once.
inline void pack_L_columns_range_dispatch(TiledMatrix* scheme,
                                          const bool has_dense, const bool has_semi,
                                          const int j_lo, const int j_hi,
                                          const int N, const int ts, const int nt,
                                          const int Adesc_lm1) {
    if (scheme->L_src) {
        pack_L_columns_range(scheme, has_dense, has_semi, j_lo, j_hi,
                             N, ts, nt, Adesc_lm1);
    } else {
        pack_L_columns_range_org(scheme, has_dense, has_semi, j_lo, j_hi,
                                 N, ts, nt, Adesc_lm1);
    }
}

// OMP backend: parallel-for over columns, dynamic scheduling for fan-out
// imbalance (etree leaves are cheap, root columns are heavy).
inline void pack_L_omp(TiledMatrix* scheme,
                       int pack_threads, bool has_dense, bool has_semi,
                       int N, int ts, int nt, int Adesc_lm1) {
    #pragma omp parallel for num_threads(pack_threads) schedule(dynamic, 64)
    for (int j = 0; j < N; ++j) {
        pack_L_columns_range_dispatch(scheme, has_dense, has_semi, j, j + 1,
                                      N, ts, nt, Adesc_lm1);
    }
}

// pthreads backend, fallback path for callers without a bound stile
// context. Spawns std::thread workers (libstdc++'s std::thread wraps
// pthreads). Used by the standalone pack_L_values(scheme) entry point
// when no stile is available; the bound-pool variant below is preferred.
inline void pack_L_pthreads_stdthread(TiledMatrix* scheme,
                                      int pack_threads, bool has_dense, bool has_semi,
                                      int N, int ts, int nt, int Adesc_lm1) {
    if (pack_threads <= 1 || N <= 0) {
        pack_L_columns_range_dispatch(scheme, has_dense, has_semi, 0, N,
                                      N, ts, nt, Adesc_lm1);
        return;
    }
    const int chunk = (N + pack_threads - 1) / pack_threads;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(pack_threads - 1));
    for (int t = 1; t < pack_threads; ++t) {
        const int j_lo = t * chunk;
        if (j_lo >= N) break;
        const int j_hi = std::min(N, j_lo + chunk);
        workers.emplace_back([=]() {
            pack_L_columns_range_dispatch(scheme, has_dense, has_semi, j_lo, j_hi,
                                          N, ts, nt, Adesc_lm1);
        });
    }
    pack_L_columns_range_dispatch(scheme, has_dense, has_semi,
                                  0, std::min(N, chunk),
                                  N, ts, nt, Adesc_lm1);
    for (auto& w : workers) w.join();
}

} // anonymous namespace

// pthreads-pool backend: drives pack_L_columns_range across the workers
// already bound by sTiles_bind / sTiles_create, exactly the same pool
// sTiles_chol uses through pthreads_dpotrf.
//
// Dynamic scheduling — each worker atomically grabs chunks of PPACK_CHUNK
// columns from a shared counter passed via the args buffer. This mirrors
// what OMP's `schedule(dynamic, N)` does internally and is essential here
// because column work is wildly imbalanced (etree-leaf columns have a
// few nonzeros each, root columns can have thousands). Static chunking
// stalls all but the worker that drew the root range; dynamic chunking
// keeps everyone busy until the work runs out.
//
// PPACK_CHUNK = 64 columns matches OMP's typical default for this pattern
// and keeps the atomic-fetch overhead amortised over enough work.
namespace Process {

static constexpr int PPACK_CHUNK = 64;

void ppack(stiles_context_t* stile) {
    TiledMatrix* scheme = nullptr;
    bool has_dense = false, has_semi = false;
    int N = 0, ts = 0, nt = 0, Adesc_lm1 = 0;
    std::atomic<int>* next_chunk = nullptr;
    sTiles::unpack_args(stile, scheme, has_dense, has_semi,
                        N, ts, nt, Adesc_lm1, next_chunk);

    if (!scheme || N <= 0 || !next_chunk) return;

    while (true) {
        const int start = next_chunk->fetch_add(PPACK_CHUNK, std::memory_order_relaxed);
        if (start >= N) break;
        const int end = std::min(start + PPACK_CHUNK, N);
        pack_L_columns_range_dispatch(scheme, has_dense, has_semi,
                                      start, end, N, ts, nt, Adesc_lm1);
    }
}

} // namespace Process

// ---------------------------------------------------------------------------
// pack_L_values
//
// Walks L_colptr / L_rowind (lower triangular CSC of L) and for each (i, j)
// reads L[i][j] = U[j][i] from the upper-triangular tile storage. Backend
// (OMP vs pthreads) follows the global UseOMP setting (param[8]) so the
// pack mirrors sTiles_chol's parallelisation strategy:
//
//   * param[8] == 1 → OMP parallel-for (dynamic schedule for fan-out balance)
//   * param[8] == 0 → pthreads:
//       - if `stile` is non-null (bound context available): drives the same
//         persistent pool sTiles_chol uses, via parallel_call → Process::ppack.
//         This is the production path — guarantees the pack runs on exactly
//         the cores chol was bound to, with no extra thread spawn.
//       - if `stile` is null: falls back to std::thread workers (which are
//         libstdc++'s pthreads wrappers). For ad-hoc callers that don't bind.
//
// Tile coordinate convention (upper-tri tiles):
//   tr = j / ts  (tile-row in U = tile-col of L column j)
//   tc = i / ts  (tile-col in U = tile-row of L row i; tc >= tr since i >= j)
//   lj = j % ts  (local row in U tile)
//   li = i % ts  (local col in U tile)
//
// Dense tile element   :  denseTiles[t][lj + li * ld]   ld = height of tile-row tr
// Semisparse diag tile :  chunkedDenseTiles[t][(kd + lj - li) + li * (kd+1)]
// Semisparse off-diag  :  chunkedDenseTiles[t][lj + acol[li] * h]
// ---------------------------------------------------------------------------
// One-shot structural prepare. Runs the O(nnz_factor) L_src build and the
// L_values allocation exactly once per scheme — gated by pack_prep_done.
// Subsequent calls (after numeric re-chols) are a single bool check and
// return immediately. Caller (pack_L_values) reads the cached has_dense /
// has_semi / Adesc_lm1 off the scheme; dim / tile_size / dimTiledMatrix
// are read directly (already on the struct).
//
// Invalidated by sTiles_unpacking (frees L_values/L_src) and any path that
// reallocates tiles or rebuilds the symbolic factor — those sites must clear
// pack_prep_done explicitly. The gate is only set after every allocation
// and the L_src build have succeeded, so a partially-prepared scheme stays
// retryable on the next call.
static inline bool _pack_prep_once(TiledMatrix* scheme) {
    if (scheme && scheme->pack_prep_done) return true;
    if (!scheme || !scheme->L_colptr || !scheme->L_rowind || scheme->nnz_factor <= 0)
        return false;
    const bool has_dense = (scheme->denseTiles != nullptr);
    const bool has_semi  = (scheme->chunkedDenseTiles != nullptr &&
                            scheme->semisparseTileMetaCore != nullptr &&
                            scheme->tileMetaCore != nullptr &&
                            scheme->diagonal_bmapper != nullptr);
    if (!has_dense && !has_semi) return false;
    const int N         = scheme->dim;
    const int ts        = scheme->tile_size;
    const int nt        = scheme->dimTiledMatrix;
    const int Adesc_lm1 = N / ts;
    if (!scheme->L_values) {
        scheme->L_values = new (std::nothrow) double[scheme->nnz_factor];
        if (!scheme->L_values) return false;
    }

    // Build the precomputed source-pointer table once. After this, every
    // subsequent pack reduces to `L_values[ptr] = *L_src[ptr]` — no index
    // math, no mapper lookup, no tile-kind dispatch. Valid as long as the
    // symbolic factor (L_colptr/L_rowind, mapper) and tile base pointers
    // (denseTiles[k], chunkedDenseTiles[k]) don't change — chol updates
    // tile *contents* in place, so the pointers stay live across re-chols.
    //
    // Memory ceiling: skip the build for very large factors so we don't
    // double the per-call pack memory on huge problems. Above the ceiling
    // the pack hot loop falls back to the per-entry kernel automatically
    // (gated on `scheme->L_src != nullptr`). Runtime-tunable via
    // sTiles_set_pack_cache_threshold_bytes(); default 2 GiB ≈ 250M nnz_factor.
    const long long bytes_needed =
        scheme->nnz_factor * (long long)sizeof(const double*);
    if (!scheme->L_src && bytes_needed <= g_l_src_max_bytes) {
        scheme->L_src = new (std::nothrow) const double*[scheme->nnz_factor];
        if (!scheme->L_src) return false;
        scheme->pack_zero = 0.0;

        const int64_t*  Lcp = scheme->L_colptr;
        const int*  Lri = scheme->L_rowind;
        const double* const zero = &scheme->pack_zero;

        for (int j = 0; j < N; ++j) {
            const int tr = j / ts;
            const int lj = j % ts;
            const int ld = (tr < Adesc_lm1) ? ts : (N % ts);
            for (int64_t ptr = Lcp[j]; ptr < Lcp[j + 1]; ++ptr) {
                const int i        = Lri[ptr];
                const int tc       = i / ts;
                const int li       = i % ts;
                const int tile_idx = scheme->mapper.map_ij(tr, tc, nt);
                const double* p    = zero;

                if (tile_idx >= 0) {
                    if (has_semi && scheme->chunkedDenseTiles[tile_idx]) {
                        const double* tile = scheme->chunkedDenseTiles[tile_idx];
                        if (tr == tc) {
                            const int kd = scheme->semisparseTileMetaCore[tile_idx].upper_bw;
                            p = &tile[(kd + lj - li) + li * (kd + 1)];
                        } else {
                            const SemisparseTileMetaCore& semi =
                                scheme->semisparseTileMetaCore[tile_idx];
                            const int ki = (li < (int)semi.acol.size()) ? semi.acol[li] : -1;
                            if (ki >= 0) {
                                const int h = scheme->tileMetaCore[tile_idx].height;
                                p = &tile[lj + ki * h];
                            }
                        }
                    } else if (has_dense && scheme->denseTiles[tile_idx]) {
                        p = &scheme->denseTiles[tile_idx][lj + li * ld];
                    }
                }
                scheme->L_src[ptr] = p;
            }
        }
    }

    // Commit cached state — only after every allocation/build succeeded.
    // Setting pack_prep_done last makes the gate a strict success witness:
    // any failure path above leaves it false so the next call retries.
    scheme->has_dense_cached = has_dense;
    scheme->has_semi_cached  = has_semi;
    scheme->Adesc_lm1_cached = Adesc_lm1;
    scheme->pack_prep_done   = true;
    return true;
}

// Bound-pool entry point. Mirrors sTiles_chol's pthreads_dpotrf shape:
// hand the work to parallel_call(stile, Process::ppack, ...) so the same
// thread pool, same bindings, same barriers are used. Falls back to OMP
// when param[8] == 1, identically to chol.
void pack_L_values(stiles_context_t* stile, TiledMatrix* scheme) {
    if (!_pack_prep_once(scheme)) return;
    const bool has_dense = scheme->has_dense_cached;
    const bool has_semi  = scheme->has_semi_cached;
    const int  N         = scheme->dim;
    const int  ts        = scheme->tile_size;
    const int  nt        = scheme->dimTiledMatrix;
    const int  Adesc_lm1 = scheme->Adesc_lm1_cached;

    int* params = sTiles_get_params();
    const bool use_omp = (params && params[sTiles::param::UseOMP] == 1);

    if (use_omp) {
        const int pack_threads = (scheme->num_cores > 0) ? scheme->num_cores : 1;
        pack_L_omp(scheme, pack_threads, has_dense, has_semi,
                   N, ts, nt, Adesc_lm1);
        return;
    }

    if (stile) {
        // Production pthreads path: drive the bound pool, exactly like chol.
        // Atomic counter for dynamic-schedule chunk dispatch lives on the
        // caller's stack; parallel_call's exit barrier guarantees no worker
        // outlives this stack frame so the address remains valid throughout.
        std::atomic<int> next_chunk{0};
        sTiles::parallel_call(stile, sTiles::Process::ppack,
                              scheme, has_dense, has_semi,
                              N, ts, nt, Adesc_lm1, &next_chunk);
    } else {
        // Standalone fallback (no bound context): spawn std::thread workers.
        const int pack_threads = (scheme->num_cores > 0) ? scheme->num_cores : 1;
        pack_L_pthreads_stdthread(scheme, pack_threads, has_dense, has_semi,
                                  N, ts, nt, Adesc_lm1);
    }
}

// Standalone overload — kept for callers that don't have a bound stile.
void pack_L_values(TiledMatrix* scheme) {
    pack_L_values(nullptr, scheme);
}

// ---------------------------------------------------------------------------
// csc_dtrsm  —  serial CSC solve using the packed CSC factor
//
// solve_type:
//   0 = forward    L y = b     (column sweep, top-down)
//   1 = backward   L^T x = y   (column sweep, bottom-up)
//   2 = full       forward then backward
//
// Diagonal is the first entry in each column (L_rowind[L_colptr[j]] == j).
// ---------------------------------------------------------------------------
STILES_MULTIVERSION
void csc_dtrsm(const TiledMatrix* scheme, double* x, int solve_type) {
    const int                       N   = scheme->dim;
    const int64_t*    __restrict__      Lcp = scheme->L_colptr;
    const int*    __restrict__      Lri = scheme->L_rowind;
    const double* __restrict__      Lv  = scheme->L_values;
    double*       __restrict__      x_  = x;  // local restrict-qualified alias

    if (solve_type == 0 || solve_type == 2) {
        for (int j = 0; j < N; ++j) {
            const double xj = (x_[j] /= Lv[Lcp[j]]);
            for (int64_t ptr = Lcp[j] + 1; ptr < Lcp[j + 1]; ++ptr)
                x_[Lri[ptr]] -= Lv[ptr] * xj;
        }
    }

    if (solve_type == 1 || solve_type == 2) {
        // Kernel selection for the L^T x = y back-sweep:
        //
        //   avg_col_len = (nnzL - N) / N    // off-diagonal nnz per column
        //   if avg_col_len >= 30  →  K1 (stiles_sparse_ddot, MKL cblas_ddoti when n>256)
        //   else                  →  K0 (inline scalar gather)
        //
        // Rationale (measured on 16 matrices, group2 INLA + group3 FEM):
        //   long L columns (FEM, INLA with heavy fill) win 3-11% with K1.
        //   short L columns (INLA sem_* family, ~10-30 nnz/col) lose 30-60%
        //   with K1 due to call overhead. Threshold 50 cleanly separates the
        //   two regimes in the measured data.
        //
        // Override at runtime:
        //   STILES_SOLVE_KERNEL=0 → force scalar  (debug / diagnostic)
        //   STILES_SOLVE_KERNEL=1 → force ddot    (debug / diagnostic)
        //   unset                  → auto via heuristic above
        const long long nnzL = scheme->nnz_factor;
        const int kernel = [N, nnzL](){
            const char* env = std::getenv("STILES_SOLVE_KERNEL");
            int k;
            const char* how;
            if (env && (env[0] == '0' || env[0] == '1')) {
                k = env[0] - '0';
                how = "env override";
            } else {
                const long long avg = (nnzL > N) ? (nnzL - N) / (long long)N : 0;
                k = (avg >= 30) ? 1 : 0;
                how = "auto (avg_col_len)";
            }
            (void)how;
            return k;
        }();

        if (kernel == 0) {
            for (int j = N - 1; j >= 0; --j) {
                double acc = x_[j];
                for (int64_t ptr = Lcp[j] + 1; ptr < Lcp[j + 1]; ++ptr)
                    acc -= Lv[ptr] * x_[Lri[ptr]];
                x_[j] = acc / Lv[Lcp[j]];
            }
        } else {
            for (int j = N - 1; j >= 0; --j) {
                const int64_t ifrom = Lcp[j] + 1;
                const int64_t ito   = Lcp[j + 1];
                const double acc = x_[j] - stiles_sparse_ddot(ito - ifrom, Lv + ifrom, x_, Lri + ifrom);
                x_[j] = acc / Lv[Lcp[j]];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// csc_dtrsm_multi  —  serial CSC solve for small nrhs (2..8).
//
// Same algorithm as csc_dtrsm but the inner update is vectorised across K
// columns. Reads L_values once and updates K columns of X in one pass —
// trades one cache line of L for K independent FMAs, which is a clear win
// for small K where the BLAS-3 tile path can't amortise dispatch yet.
//
// X is column-major: column k starts at X + k*ldb. ldb is typically
// scheme->dim (matches the buffer wrapper_solve passes in).
//
// solve_type:  0 = fwd,  1 = bwd,  2 = full LL^T
// ---------------------------------------------------------------------------
STILES_MULTIVERSION
void csc_dtrsm_multi(const TiledMatrix* scheme, double* X, int nrhs, int ldb, int solve_type) {
    const int                       N   = scheme->dim;
    const int64_t*    __restrict__      Lcp = scheme->L_colptr;
    const int*    __restrict__      Lri = scheme->L_rowind;
    const double* __restrict__      Lv  = scheme->L_values;
    double*       __restrict__      X_  = X;   // local restrict alias for the SIMD hot loop

    if (solve_type == 0 || solve_type == 2) {
        for (int j = 0; j < N; ++j) {
            const double inv_diag = 1.0 / Lv[Lcp[j]];
            double xj[64];
            #pragma omp simd
            for (int k = 0; k < nrhs; ++k) {
                const double v = X_[static_cast<std::size_t>(k) * ldb + j] * inv_diag;
                X_[static_cast<std::size_t>(k) * ldb + j] = v;
                xj[k] = v;
            }
            for (int64_t ptr = Lcp[j] + 1; ptr < Lcp[j + 1]; ++ptr) {
                const int    i   = Lri[ptr];
                const double Lij = Lv[ptr];
                #pragma omp simd
                for (int k = 0; k < nrhs; ++k) {
                    X_[static_cast<std::size_t>(k) * ldb + i] -= Lij * xj[k];
                }
            }
        }
    }

    if (solve_type == 1 || solve_type == 2) {
        // Same kernel-selection scheme as csc_dtrsm.  See its comment block
        // for the rationale on the avg_col_len heuristic and the env override.
        //   K0: K-wide SIMD inner loop (reads each L entry once, amortises K)
        //   K1: K separate stiles_sparse_ddot calls per column
        //       (loses L amortisation but routes the X gather through MKL
        //        cblas_ddoti when n>256 — wins on long L columns)
        const long long nnzL = scheme->nnz_factor;
        const int kernel = [N, nnzL](){
            const char* env = std::getenv("STILES_SOLVE_KERNEL");
            int k;
            const char* how;
            if (env && (env[0] == '0' || env[0] == '1')) {
                k = env[0] - '0';
                how = "env override";
            } else {
                const long long avg = (nnzL > N) ? (nnzL - N) / (long long)N : 0;
                k = (avg >= 30) ? 1 : 0;
                how = "auto (avg_col_len)";
            }
            static bool announced = false;
            if (!announced) {
                announced = true;
                const long long avg = (nnzL > N) ? (nnzL - N) / (long long)N : 0;
                sTiles::Logger::debugf(
                    "[csc_dtrsm_multi] N=%d nnzL=%lld avg_col_len=%lld -> kernel=%d (%s; %s)",
                    N, nnzL, avg, k,
                    k == 0 ? "K-wide SIMD" : "K x stiles_sparse_ddot", how);
            }
            return k;
        }();

        if (kernel == 0) {
            for (int j = N - 1; j >= 0; --j) {
                double acc[64];
                #pragma omp simd
                for (int k = 0; k < nrhs; ++k) {
                    acc[k] = X_[static_cast<std::size_t>(k) * ldb + j];
                }
                for (int64_t ptr = Lcp[j] + 1; ptr < Lcp[j + 1]; ++ptr) {
                    const int    i   = Lri[ptr];
                    const double Lij = Lv[ptr];
                    #pragma omp simd
                    for (int k = 0; k < nrhs; ++k) {
                        acc[k] -= Lij * X_[static_cast<std::size_t>(k) * ldb + i];
                    }
                }
                const double inv_diag = 1.0 / Lv[Lcp[j]];
                #pragma omp simd
                for (int k = 0; k < nrhs; ++k) {
                    X_[static_cast<std::size_t>(k) * ldb + j] = acc[k] * inv_diag;
                }
            }
        } else {
            for (int j = N - 1; j >= 0; --j) {
                const int64_t ifrom    = Lcp[j] + 1;
                const int64_t ito      = Lcp[j + 1];
                const int    n        = ito - ifrom;
                const double inv_diag = 1.0 / Lv[Lcp[j]];
                for (int k = 0; k < nrhs; ++k) {
                    double* __restrict__ Xk = X_ + static_cast<std::size_t>(k) * ldb;
                    const double acc = Xk[j] - stiles_sparse_ddot(n, Lv + ifrom, Xk, Lri + ifrom);
                    Xk[j] = acc * inv_diag;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// csc_dtrsm_multi_row  —  same algorithm as csc_dtrsm_multi but X is
// row-major: row i of X starts at X + i*ldb_row, with ldb_row >= nrhs.
// At small nrhs (e.g. 2-4) the K-wide updates land within one cache line
// per row, which can be friendlier to the cache hierarchy than the
// column-major layout where the K columns are scheme->dim apart.
// ---------------------------------------------------------------------------
STILES_MULTIVERSION
void csc_dtrsm_multi_row(const TiledMatrix* scheme, double* X, int nrhs,
                         int ldb_row, int solve_type) {
    const int                       N   = scheme->dim;
    const int64_t*    __restrict__      Lcp = scheme->L_colptr;
    const int*    __restrict__      Lri = scheme->L_rowind;
    const double* __restrict__      Lv  = scheme->L_values;
    double*       __restrict__      X_  = X;

    if (solve_type == 0 || solve_type == 2) {
        for (int j = 0; j < N; ++j) {
            const double inv_diag = 1.0 / Lv[Lcp[j]];
            double* __restrict__ Xj = X_ + static_cast<std::size_t>(j) * ldb_row;
            double xj[64];
            #pragma omp simd
            for (int k = 0; k < nrhs; ++k) {
                const double v = Xj[k] * inv_diag;
                Xj[k]  = v;
                xj[k]  = v;
            }
            for (int64_t ptr = Lcp[j] + 1; ptr < Lcp[j + 1]; ++ptr) {
                const int    i   = Lri[ptr];
                const double Lij = Lv[ptr];
                double* __restrict__ Xi = X_ + static_cast<std::size_t>(i) * ldb_row;
                #pragma omp simd
                for (int k = 0; k < nrhs; ++k) {
                    Xi[k] -= Lij * xj[k];
                }
            }
        }
    }

    if (solve_type == 1 || solve_type == 2) {
        // Kernel selection — row-major variant.
        //   K0: K-wide SIMD over contiguous Xi[k] (very cache-friendly)
        //   K1: K separate stiles_sparse_ddot calls per column, using a
        //       per-column scaled index buffer so the gather hits the
        //       row-major X at offsets (Lri[ptr]*ldb_row + k).
        //
        // Same avg_col_len >= 30 split as csc_dtrsm, plus a second axis:
        // at nrhs >= 3, K1's K-fold L re-reads start costing real bandwidth
        // when L doesn't fit in L3. Empirically (22-matrix sweep at nrhs=4)
        // K0 wins by 4-10% on row-major when nnzL > ~15M — spacetime (18M),
        // shipsec1 (43M), shipsec8 (41M), ship_003 (66M). Below 15M
        // (oilpan 8.7M, ferris 3.5M, 83o4NNNo 4.7M, bcsstk*) K1 still
        // wins. Col-major and nrhs<=2 don't need this axis (K1 wins
        // almost universally there).
        const long long nnzL = scheme->nnz_factor;
        const int kernel = [N, nnzL, nrhs](){
            const char* env = std::getenv("STILES_SOLVE_KERNEL");
            int k;
            const char* how;
            if (env && (env[0] == '0' || env[0] == '1')) {
                k = env[0] - '0';
                how = "env override";
            } else {
                const long long avg = (nnzL > N) ? (nnzL - N) / (long long)N : 0;
                const bool large_L_at_wide_nrhs = (nrhs >= 3) && (nnzL > 15'000'000LL);
                k = (avg >= 30 && !large_L_at_wide_nrhs) ? 1 : 0;
                how = "auto (avg_col_len + nnzL)";
            }
            static bool announced = false;
            if (!announced) {
                announced = true;
                const long long avg = (nnzL > N) ? (nnzL - N) / (long long)N : 0;
                sTiles::Logger::debugf(
                    "[csc_dtrsm_multi_row] N=%d nnzL=%lld avg_col_len=%lld nrhs=%d -> kernel=%d (%s; %s)",
                    N, nnzL, avg, nrhs, k,
                    k == 0 ? "K-wide SIMD" : "K x stiles_sparse_ddot", how);
            }
            return k;
        }();

        if (kernel == 0) {
            for (int j = N - 1; j >= 0; --j) {
                double* __restrict__ Xj = X_ + static_cast<std::size_t>(j) * ldb_row;
                double acc[64];
                #pragma omp simd
                for (int k = 0; k < nrhs; ++k) acc[k] = Xj[k];
                for (int64_t ptr = Lcp[j] + 1; ptr < Lcp[j + 1]; ++ptr) {
                    const int    i   = Lri[ptr];
                    const double Lij = Lv[ptr];
                    const double* __restrict__ Xi = X_ + static_cast<std::size_t>(i) * ldb_row;
                    #pragma omp simd
                    for (int k = 0; k < nrhs; ++k) acc[k] -= Lij * Xi[k];
                }
                const double inv_diag = 1.0 / Lv[Lcp[j]];
                #pragma omp simd
                for (int k = 0; k < nrhs; ++k) Xj[k] = acc[k] * inv_diag;
            }
        } else {
            // Scratch buffer of pre-scaled row indices so cblas_ddoti can
            // gather the row-major X at stride ldb_row per RHS.  Upper-bounded
            // by N (no column has more than N nonzeros).
            std::vector<int> idx_scaled(N);
            for (int j = N - 1; j >= 0; --j) {
                const int64_t ifrom = Lcp[j] + 1;
                const int64_t ito   = Lcp[j + 1];
                const int n     = ito - ifrom;
                for (int t = 0; t < n; ++t) {
                    idx_scaled[t] = Lri[ifrom + t] * ldb_row;
                }
                double* __restrict__ Xj = X_ + static_cast<std::size_t>(j) * ldb_row;
                const double inv_diag = 1.0 / Lv[Lcp[j]];
                for (int k = 0; k < nrhs; ++k) {
                    const double acc = Xj[k] - stiles_sparse_ddot(n, Lv + ifrom, X_ + k, idx_scaled.data());
                    Xj[k] = acc * inv_diag;
                }
            }
        }
    }
}

} // namespace sTiles
