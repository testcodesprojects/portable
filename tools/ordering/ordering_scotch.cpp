#include "ordering_utils.hpp"
#include <scotch.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mutex>
#include "../sort/stiles_sort_dispatch.hpp"

extern "C" int stiles_createSmartPermutation(int** row_indices, int** col_indices, int nnz, int node_num, int** perm);

namespace sTiles {

// Build symmetric deduplicated CSR (verttab/edgetab) from COO input using flat allocation.
// Two-pass: count degrees, fill raw edges, sort+unique in parallel, compact.
static void build_scotch_graph(const int* rows, const int* cols, int nnz, int dim,
                               std::vector<SCOTCH_Num>& verttab,
                               std::vector<SCOTCH_Num>& edgetab,
                               SCOTCH_Num& edgenbr)
{
    // Pass 1: raw degree count
    std::vector<SCOTCH_Num> deg(dim, 0);
    for (int k = 0; k < nnz; ++k) {
        int r = rows[k], c = cols[k];
        if (r >= 0 && c >= 0 && r < dim && c < dim && r != c) { deg[r]++; deg[c]++; }
    }

    // Prefix sum -> raw offset table
    std::vector<SCOTCH_Num> raw_vtab(dim + 1, 0);
    for (int v = 0; v < dim; ++v) raw_vtab[v+1] = raw_vtab[v] + deg[v];

    // Flat raw edge buffer (pre-dedup, symmetric so 2x entries)
    std::vector<SCOTCH_Num> raw_etab(raw_vtab[dim]);
    std::vector<SCOTCH_Num> pos(dim, 0);

    // Pass 2: fill edges
    for (int k = 0; k < nnz; ++k) {
        int r = rows[k], c = cols[k];
        if (r >= 0 && c >= 0 && r < dim && c < dim && r != c) {
            raw_etab[raw_vtab[r] + pos[r]++] = c;
            raw_etab[raw_vtab[c] + pos[c]++] = r;
        }
    }

    // Sort + unique per vertex
    std::vector<SCOTCH_Num> new_deg(dim);
    for (int v = 0; v < dim; ++v) {
        auto* beg = raw_etab.data() + raw_vtab[v];
        auto* end = beg + deg[v];
        std::sort(beg, end);
        new_deg[v] = (SCOTCH_Num)(std::unique(beg, end) - beg);
    }

    // Build final compact verttab
    verttab.resize(dim + 1);
    verttab[0] = 0;
    for (int v = 0; v < dim; ++v) verttab[v+1] = verttab[v] + new_deg[v];
    edgenbr = verttab[dim];

    // Compact edgetab (copy deduplicated slices into contiguous buffer)
    edgetab.resize(edgenbr);
    for (int v = 0; v < dim; ++v) {
        std::memcpy(edgetab.data() + verttab[v],
                    raw_etab.data() + raw_vtab[v],
                    new_deg[v] * sizeof(SCOTCH_Num));
    }
}

// perm[iperm[i]] = i  for i in [0, N). Validates that iperm is a permutation of
// [0,N): a corrupt ordering from the backend (out-of-range index) would otherwise
// scatter-write perm[garbage] to a wild address (seen as an arm64 EXC_BAD_ACCESS).
// Returns false on a bad index so the caller can reject the ordering and fall back
// to another bake-off candidate instead of crashing.
static bool build_perm_from_iperm(int* perm, const int* iperm, int N) {
    for (int i = 0; i < N; ++i) {
        const int j = iperm[i];
        if (j < 0 || j >= N) {
            std::fprintf(stderr,
                "build_perm_from_iperm: iperm[%d]=%d out of range [0,%d) — rejecting ordering\n",
                i, j, N);
            return false;
        }
        perm[j] = i;
    }
    return true;
}

// Gather: dst[i] = src[idx[i]]
static void gather_int(int* dst, const int* src, const int* idx, int N) {
    for (int i = 0; i < N; ++i) dst[i] = src[idx[i]];
}



// Internal worker shared by runSCOTCH (extern "C") and runSCOTCH_with_tree (C++).
// When tree_out != nullptr, the SCOTCH ND separator tree (cblknbr/rangtab/treetab) is
// captured into *tree_out. Otherwise the tree is discarded.
// SCOTCH uses GLOBAL random state (SCOTCH_randomSeed/SCOTCH_randomReset). When
// the symbolic phase launches SCOTCH/SCOTCH2/FSCOTCH concurrently via std::async
// (see stiles_ordering.hpp), their SCOTCH_randomSeed calls race and produce
// nondeterministic orderings. Serialize all SCOTCH calls with this mutex so
// each variant gets its seed installed atomically. The cost is small: SCOTCH
// runs are short and were going to share a single CPU's worth of work anyway.
static std::mutex g_scotch_global_mutex;

static int runSCOTCH_impl(int** csr_i, int** csr_j, int N, int nnz, int m,
                          int** perm, int** iperm, int num_cores, int seed,
                          ScotchTree* tree_out, const SharedAdjCSR* shared) {
    // SCOTCH parallelization (DEFAULT ON): use a per-call SCOTCH_Context (SCOTCH 7)
    // that isolates the random state + deterministic option per call, so concurrent
    // SCOTCH orderings are thread-safe -> NO global SCOTCH_DETERMINISTIC env, NO
    // serializing mutex (the 3 SCOTCH variants run in parallel). Validated bit-exact
    // on 87/89 matrices (2 differ by ~1e-6, alternative valid orderings).
    // Set STILES_SCOTCH_CTX=0 to fall back to the original global-determinism +
    // mutex path (serial, exact-reproducible).
    const char* _e_ctx = std::getenv("STILES_SCOTCH_CTX");
    // DEFAULT ON (per-call SCOTCH_Context) on ALL platforms; STILES_SCOTCH_CTX=0
    // forces the classic global+mutex path.
    //
    // macOS/arm64 REQUIRES the context path. The classic non-context path uses
    // SCOTCH's global state and, on arm64, corrupts memory in SCOTCH's own graph
    // recursion (a garbage-pointer SEGV in SCOTCH_graphExit / an abort under ASan)
    // for the spacetime matrix. The context path binds an isolated managed graph
    // and initializes threading through the context (honoring SCOTCH_PTHREAD_NUMBER=1,
    // set on macOS above/at load), and is CLEAN there: the macos-asan-scotch CI
    // probe aborts with CTX=0 but PASSES with CTX=1. (An earlier build defaulted
    // this OFF on macOS, before the tree-skip removed the context path's own
    // treetab over-free; with that fixed, context-on is the good path.)
    const bool use_ctx = !(_e_ctx && std::atoi(_e_ctx) == 0);
    std::unique_lock<std::mutex> scotch_lock(g_scotch_global_mutex, std::defer_lock);
    if (!use_ctx) {
        // Force SCOTCH into deterministic mode (global) + serialize callers.
        setenv("SCOTCH_DETERMINISTIC", "1", 1);
        scotch_lock.lock();
    }

    if (tree_out) {
        tree_out->cblknbr = 0;
        tree_out->rangtab.clear();
        tree_out->treetab.clear();
    }

    const int dim = N - m;
    if (dim <= 0) {
        std::fprintf(stderr, "runSCOTCH: dim = N - m <= 0 (N = %d, m = %d)\n", N, m);
        return 1;
    }

    std::vector<SCOTCH_Num> verttab, edgetab;
    SCOTCH_Num edgenbr = 0;
    if (shared && shared->valid_for(dim)) {
        // Reuse the prebuilt canonical graph. It is, by construction, identical to
        // what build_scotch_graph produces (0-based, symmetric, dedup, no diagonal);
        // here we only widen int -> SCOTCH_Num (a no-op copy when they match).
        const int ne = shared->xadj[dim];
        verttab.resize(static_cast<std::size_t>(dim) + 1);
        edgetab.resize(static_cast<std::size_t>(ne));
        for (int v = 0; v <= dim; ++v) verttab[v] = static_cast<SCOTCH_Num>(shared->xadj[v]);
        for (int e = 0; e < ne; ++e)   edgetab[e] = static_cast<SCOTCH_Num>(shared->adjncy[e]);
        edgenbr = static_cast<SCOTCH_Num>(ne);
    } else {
        build_scotch_graph(*csr_i, *csr_j, nnz, dim, verttab, edgetab, edgenbr);
    }

    SCOTCH_Graph graf;
    SCOTCH_graphInit(&graf);
    if (SCOTCH_graphBuild(&graf, 0, (SCOTCH_Num)dim, verttab.data(), verttab.data() + 1,
                          nullptr, nullptr, edgenbr, edgetab.data(), nullptr) != 0) {
        std::fprintf(stderr, "runSCOTCH: SCOTCH_graphBuild failed\n");
        SCOTCH_graphExit(&graf);
        return 1;
    }

    SCOTCH_Strat strat;
    SCOTCH_stratInit(&strat);
    int strat_rc = 1;
    if (dim > 1000) {
        // Three distinct SCOTCH strategies, selected by seed value. Each
        // explores a different quality/balance trade-off so the 3-variant
        // benchmark covers more of the solution space than just different
        // random seeds with the same parameters.
        //
        // seed 0  (SCOTCH):  balanced — current default, good all-round
        // seed 42 (SCOTCH2): quality  — finer ND, tighter balance, more
        //                    refinement passes. Best for regular FEM meshes.
        // seed 7  (FSCOTCH): fast     — coarser ND, relaxed balance, fewer
        //                    passes. Better for irregular/graph matrices.
        const char* quality_strat;
        if (seed == 42) {
            // SCOTCH2: quality — fine ND, tight balance, heavy refinement
            quality_strat =
                "n{sep=m{vert=50,low=h{pass=20}f{bal=0.02},"
                "asc=b{bnd=f{bal=0.02},org=f{bal=0.02}}},"
                "ole=d{cmin=20,cmax=100000,frat=0},"
                "ose=g{pass=20}}";
        } else if (seed == 7) {
            // FSCOTCH: fast — coarse ND, relaxed balance
            quality_strat =
                "n{sep=m{vert=200,low=h{pass=5}f{bal=0.10},"
                "asc=b{bnd=f{bal=0.10},org=f{bal=0.10}}},"
                "ole=d{cmin=100,cmax=100000,frat=0},"
                "ose=g{pass=5}}";
        } else {
            // SCOTCH (default): balanced
            quality_strat =
                "n{sep=m{vert=100,low=h{pass=10}f{bal=0.05},"
                "asc=b{bnd=f{bal=0.05},org=f{bal=0.05}}},"
                "ole=d{cmin=50,cmax=100000,frat=0},"
                "ose=g{pass=10}}";
        }
        strat_rc = SCOTCH_stratGraphOrder(&strat, quality_strat);
        if (strat_rc != 0)
            std::fprintf(stderr, "runSCOTCH: custom strategy failed, using built-in\n");
    }
    if (strat_rc != 0) {
        const SCOTCH_Num flags = SCOTCH_STRATQUALITY | SCOTCH_STRATRECURSIVE | SCOTCH_STRATBALANCE;
        strat_rc = SCOTCH_stratGraphOrderBuild(&strat, flags, 0, 0.05);
    }
    if (strat_rc != 0) {
        std::fprintf(stderr, "runSCOTCH: strategy build failed\n");
        SCOTCH_stratExit(&strat);
        SCOTCH_graphExit(&graf);
        return 1;
    }

    if (!use_ctx) {
        // global random state (old path); context path seeds its own below
        SCOTCH_randomSeed(static_cast<SCOTCH_Num>(seed));
        SCOTCH_randomReset();
    }
    std::vector<SCOTCH_Num> sp(dim), si(dim);

    // Allocate separator-tree buffers only when caller wants them.
    SCOTCH_Num cblknbr_out = 0;
    std::vector<SCOTCH_Num> rangtab_buf;
    std::vector<SCOTCH_Num> treetab_buf;
    SCOTCH_Num* cblknbr_ptr = nullptr;
    SCOTCH_Num* rangtab_ptr = nullptr;
    SCOTCH_Num* treetab_ptr = nullptr;
    if (tree_out) {
        rangtab_buf.assign(static_cast<std::size_t>(dim) + 1, 0);
        treetab_buf.assign(static_cast<std::size_t>(dim), 0);
        cblknbr_ptr = &cblknbr_out;
        rangtab_ptr = rangtab_buf.data();
        treetab_ptr = treetab_buf.data();
    }

    int order_rc;
    if (use_ctx) {
        // Per-call context: isolated random + deterministic, no shared global state
        // -> safe to run concurrently with other SCOTCH calls (no mutex needed).
        SCOTCH_Context ctx;
        SCOTCH_contextInit(&ctx);
        SCOTCH_contextOptionSetNum(&ctx, SCOTCH_OPTIONNUMDETERMINISTIC, 1);
        SCOTCH_contextRandomSeed(&ctx, static_cast<SCOTCH_Num>(seed));
        SCOTCH_contextRandomReset(&ctx);
        SCOTCH_Graph bound;
        SCOTCH_graphInit(&bound);
        SCOTCH_contextBindGraph(&ctx, &graf, &bound);
        order_rc = SCOTCH_graphOrder(&bound, &strat, sp.data(), si.data(),
                                     cblknbr_ptr, rangtab_ptr, treetab_ptr);
        SCOTCH_graphExit(&bound);
        SCOTCH_contextExit(&ctx);
    } else {
        order_rc = SCOTCH_graphOrder(&graf, &strat, sp.data(), si.data(),
                                     cblknbr_ptr, rangtab_ptr, treetab_ptr);
    }
    if (order_rc != 0) {
        std::fprintf(stderr, "runSCOTCH: SCOTCH_graphOrder failed\n");
        SCOTCH_stratExit(&strat);
        SCOTCH_graphExit(&graf);
        return 1;
    }
    SCOTCH_stratExit(&strat);
    SCOTCH_graphExit(&graf);

    for (int i = 0; i < dim; ++i) (*iperm)[i] = (int)sp[i];
    for (int i = dim; i < N;   ++i) (*iperm)[i] = i;
    if (!build_perm_from_iperm(*perm, *iperm, N)) {
        std::fprintf(stderr, "runSCOTCH: SCOTCH returned an invalid permutation (dim=%d, N=%d)\n", dim, N);
        return 1;   // reject: caller falls back to another ordering (or identity)
    }

    if (tree_out) {
        const int cblknbr = static_cast<int>(cblknbr_out);
        tree_out->cblknbr = cblknbr;
        tree_out->rangtab.resize(static_cast<std::size_t>(cblknbr) + 1);
        tree_out->treetab.resize(static_cast<std::size_t>(cblknbr));
        for (int k = 0; k <= cblknbr; ++k)
            tree_out->rangtab[k] = static_cast<int>(rangtab_buf[k]);
        for (int k = 0; k < cblknbr; ++k)
            tree_out->treetab[k] = static_cast<int>(treetab_buf[k]);
    }

    return 0;
}

int runSCOTCH(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, int num_cores, int seed, const SharedAdjCSR* shared) {
    return runSCOTCH_impl(csr_i, csr_j, N, nnz, m, perm, iperm, num_cores, seed, nullptr, shared);
}

int runSCOTCH_with_tree(int** csr_i, int** csr_j, int N, int nnz, int m,
                        int** perm, int** iperm, int num_cores, int seed,
                        ScotchTree* tree_out, const SharedAdjCSR* shared) {
    return runSCOTCH_impl(csr_i, csr_j, N, nnz, m, perm, iperm, num_cores, seed, tree_out, shared);
}


int runASCOTCH(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, int num_cores, int seed) {
    // Same determinism + serialization as runSCOTCH_impl (STILES_SCOTCH_CTX: per-call
    // context -> no mutex, runs concurrently with other SCOTCH calls).
    const char* _e_ctx = std::getenv("STILES_SCOTCH_CTX");
    // DEFAULT ON everywhere (see runSCOTCH_impl: macOS/arm64 needs the context path;
    // the non-context path corrupts memory there). STILES_SCOTCH_CTX=0 forces off.
    const bool use_ctx = !(_e_ctx && std::atoi(_e_ctx) == 0);
    std::unique_lock<std::mutex> scotch_lock(g_scotch_global_mutex, std::defer_lock);
    if (!use_ctx) {
        setenv("SCOTCH_DETERMINISTIC", "1", 1);
        scotch_lock.lock();
    }

    int* save_rows  = nullptr;
    int* save_cols  = nullptr;
    int* pperm      = nullptr;
    bool double_perm = false;

    if (m == 0) {
        save_rows = static_cast<int*>(std::malloc(static_cast<size_t>(nnz) * sizeof(int)));
        save_cols = static_cast<int*>(std::malloc(static_cast<size_t>(nnz) * sizeof(int)));
        pperm     = static_cast<int*>(std::malloc(static_cast<size_t>(N)   * sizeof(int)));

        if (!save_rows || !save_cols || !pperm) {
            std::fprintf(stderr, "runASCOTCH: memory allocation failed for save buffers or pperm\n");
            std::free(save_rows); std::free(save_cols); std::free(pperm);
            return 1;
        }

        std::memcpy(save_rows, *csr_i, nnz * sizeof(int));
        std::memcpy(save_cols, *csr_j, nnz * sizeof(int));

        double_perm = true;
        m = stiles_createSmartPermutation(csr_i, csr_j, nnz, N, &pperm);
    }

    const int dim = N - m;
    if (dim <= 0) {
        std::fprintf(stderr, "runASCOTCH: dim = N - m <= 0 (N = %d, m = %d)\n", N, m);
        std::free(save_rows); std::free(save_cols); std::free(pperm);
        return 1;
    }

    std::vector<SCOTCH_Num> verttab, edgetab;
    SCOTCH_Num edgenbr = 0;
    build_scotch_graph(*csr_i, *csr_j, nnz, dim, verttab, edgetab, edgenbr);

    SCOTCH_Graph graf;
    SCOTCH_graphInit(&graf);
    if (SCOTCH_graphBuild(&graf, 0, (SCOTCH_Num)dim, verttab.data(), verttab.data() + 1,
                          nullptr, nullptr, edgenbr, edgetab.data(), nullptr) != 0) {
        std::fprintf(stderr, "runASCOTCH: SCOTCH_graphBuild failed\n");
        SCOTCH_graphExit(&graf);
        std::free(save_rows); std::free(save_cols); std::free(pperm);
        return 1;
    }

    SCOTCH_Strat strat;
    SCOTCH_stratInit(&strat);
    // ASCOTCH uses SCOTCH's built-in QUALITY|RECURSIVE preset with relaxed
    // imbalance (0.1). This works better than custom strategy strings when
    // combined with the SmartPermutation pre-ordering, because the built-in
    // preset's internal heuristics complement the pre-permutation's locality.
    const SCOTCH_Num strat_flags_a = (dim > 400000)
        ? (SCOTCH_STRATQUALITY | SCOTCH_STRATRECURSIVE | SCOTCH_STRATBALANCE)
        : (SCOTCH_STRATQUALITY | SCOTCH_STRATRECURSIVE);
    const double strat_imbal_a = (dim > 400000) ? 0.05 : 0.1;
    if (SCOTCH_stratGraphOrderBuild(&strat, strat_flags_a, 0, strat_imbal_a) != 0) {
        std::fprintf(stderr, "runASCOTCH: SCOTCH_stratGraphOrderBuild failed\n");
        SCOTCH_stratExit(&strat);
        SCOTCH_graphExit(&graf);
        std::free(save_rows); std::free(save_cols); std::free(pperm);
        return 1;
    }

    if (!use_ctx) {
        SCOTCH_randomSeed(static_cast<SCOTCH_Num>(seed));
        SCOTCH_randomReset();
    }
    std::vector<SCOTCH_Num> scotch_perm(dim), scotch_iperm(dim);
    int order_rc;
    if (use_ctx) {
        SCOTCH_Context ctx;
        SCOTCH_contextInit(&ctx);
        SCOTCH_contextOptionSetNum(&ctx, SCOTCH_OPTIONNUMDETERMINISTIC, 1);
        SCOTCH_contextRandomSeed(&ctx, static_cast<SCOTCH_Num>(seed));
        SCOTCH_contextRandomReset(&ctx);
        SCOTCH_Graph bound;
        SCOTCH_graphInit(&bound);
        SCOTCH_contextBindGraph(&ctx, &graf, &bound);
        order_rc = SCOTCH_graphOrder(&bound, &strat, scotch_perm.data(), scotch_iperm.data(), nullptr, nullptr, nullptr);
        SCOTCH_graphExit(&bound);
        SCOTCH_contextExit(&ctx);
    } else {
        order_rc = SCOTCH_graphOrder(&graf, &strat, scotch_perm.data(), scotch_iperm.data(), nullptr, nullptr, nullptr);
    }
    if (order_rc != 0) {
        std::fprintf(stderr, "runASCOTCH: SCOTCH_graphOrder failed\n");
        SCOTCH_stratExit(&strat);
        SCOTCH_graphExit(&graf);
        std::free(save_rows); std::free(save_cols); std::free(pperm);
        return 1;
    }
    SCOTCH_stratExit(&strat);
    SCOTCH_graphExit(&graf);

    for (int i = 0; i < dim; ++i) (*iperm)[i] = (int)scotch_perm[i];
    for (int i = dim; i < N;   ++i) (*iperm)[i] = i;
    if (!build_perm_from_iperm(*perm, *iperm, N)) {
        std::fprintf(stderr, "runASCOTCH: SCOTCH returned an invalid permutation (dim=%d, N=%d)\n", dim, N);
        std::free(save_rows); std::free(save_cols); std::free(pperm);
        return 1;
    }

    if (double_perm) {
        // Restore original sparsity pattern for caller
        std::memcpy(*csr_i, save_rows, nnz * sizeof(int));
        std::memcpy(*csr_j, save_cols, nnz * sizeof(int));

        int* newperm = static_cast<int*>(std::malloc(static_cast<size_t>(N) * sizeof(int)));
        if (!newperm) {
            std::fprintf(stderr, "runASCOTCH: memory allocation failed for newperm\n");
            std::free(save_rows); std::free(save_cols); std::free(pperm);
            return 1;
        }

        // Compose: newperm[i] = iperm[pperm[i]]  (AVX2 gather)
        gather_int(newperm, *iperm, pperm, N);

        std::memcpy(*iperm, newperm, N * sizeof(int));
        const bool perm_ok = build_perm_from_iperm(*perm, *iperm, N);

        std::free(newperm);
        std::free(save_rows);
        std::free(save_cols);
        std::free(pperm);
        if (!perm_ok) {
            std::fprintf(stderr, "runASCOTCH: composed permutation invalid (N=%d)\n", N);
            return 1;
        }
    }

    return 0;
}

} // namespace sTiles
