/**
 * parallel_chol.cpp — demonstrate sTiles nested parallelism:
 *   N Cholesky factorizations running CONCURRENTLY, each on its own cores.
 *
 * This is the INLA-style pattern: one group with N "calls", each call an
 * independent factorization of the matrix, all running at once. We time the N
 * factorizations run serially (one after another) vs concurrently, so you can
 * see the parallel speedup, and check every call gets the same log-determinant.
 *
 * Usage:  ./parallel_chol <matrix.mtx> [n_calls] [cores_per_call] [tile_size]
 *   default: n_calls=4, cores_per_call=2  (so 8 cores busy at peak)
 */
#include <stiles.h>
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

struct Coo { int n = 0, nnz = 0; std::vector<int> row, col; std::vector<double> val; };

static bool load_mtx(const char* path, Coo& m) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return false; }
    std::string line; bool sym = false;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] == '%') { if (line.find("symmetric") != std::string::npos) sym = true; continue; }
        break;
    }
    int nr, nc, nnzf; std::istringstream hs(line); hs >> nr >> nc >> nnzf;
    if (nr != nc) { std::fprintf(stderr, "matrix not square\n"); return false; }
    m.n = nr;
    int r, c; double v;
    while (f >> r >> c >> v) { --r; --c; if (sym && r < c) std::swap(r, c); m.row.push_back(r); m.col.push_back(c); m.val.push_back(v); }
    m.nnz = (int)m.row.size();
    return m.nnz > 0;
}

static const char* basename_of(const char* p) { const char* s = std::strrchr(p, '/'); return s ? s + 1 : p; }

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <matrix.mtx> [n_calls] [cores_per_call] [tile_size]\n", argv[0]);
        return 2;
    }
    const char* path        = argv[1];
    const int n_calls       = (argc > 2) ? std::atoi(argv[2]) : 4;
    const int cores_per_call= (argc > 3) ? std::atoi(argv[3]) : 2;
    const int tile          = (argc > 4) ? std::atoi(argv[4]) : 40;

    Coo m;
    if (!load_mtx(path, m)) return 2;

    setenv("STILES_NO_BANNER", "1", 0);
    setenv("STILES_LOG_LEVEL", "0", 0);

    // Nested parallelism: outer loop over calls, inner tiled factorization per call.
    omp_set_nested(1);
    omp_set_max_active_levels(2);

    sTiles_expert_user();
    sTiles_set_log_level(-1);
    sTiles_set_tile_size(tile);
    sTiles_set_tile_type_mode(3);              // auto

    // One group, n_calls concurrent factorizations, each using cores_per_call cores.
    const int  num_groups        = 1;
    int        calls_per_group[] = { n_calls };
    int        cores_per_group[] = { cores_per_call };
    int        chol_type[]       = { 0 };
    bool       get_inverse[]     = { false };  // chol + logdet only for this demo

    void* st = nullptr;
    sTiles_create(&st, num_groups, calls_per_group, cores_per_group, chol_type, get_inverse);
    for (int c = 0; c < n_calls; ++c)
        sTiles_assign_graph_one_call(0, c, &st, m.n, m.nnz, m.row.data(), m.col.data());
    sTiles_init_group(0, &st);                 // symbolic (shared)

    auto assign_all = [&]() {                  // (re)load values into every call
        for (int c = 0; c < n_calls; ++c)
            sTiles_assign_values(0, c, &st, m.val.data());
    };
    auto chol_call = [&](int c) {
        sTiles_bind(0, c, &st); sTiles_chol(0, c, &st); sTiles_unbind(0, c, &st);
    };

    // ---- Serial: the N factorizations one after another (each on cores_per_call) ----
    assign_all();
    auto s0 = clk::now();
    for (int c = 0; c < n_calls; ++c) chol_call(c);
    const double t_serial = secs(s0, clk::now());

    // ---- Concurrent: all N factorizations at once (each on cores_per_call) ----
    assign_all();
    auto p0 = clk::now();
    #pragma omp parallel for num_threads(n_calls) schedule(static, 1)
    for (int c = 0; c < n_calls; ++c) chol_call(c);
    const double t_parallel = secs(p0, clk::now());

    // ---- Correctness: every call must yield the same log-determinant ----
    double ld0 = sTiles_get_logdet(0, 0, &st);
    double max_ld_diff = 0.0;
    for (int c = 0; c < n_calls; ++c)
        max_ld_diff = std::fmax(max_ld_diff, std::fabs(sTiles_get_logdet(0, c, &st) - ld0));
    const bool ok = std::isfinite(ld0) && max_ld_diff < 1e-9;

    sTiles_quit();

    std::printf("\n");
    std::printf("matrix          : %s  (n=%d, nnz=%d)\n", basename_of(path), m.n, m.nnz);
    std::printf("concurrent chols: %d  x  %d cores each  = %d cores at peak\n",
                n_calls, cores_per_call, n_calls * cores_per_call);
    std::printf("logdet          : %.4f  (max diff across calls: %.1e)\n", ld0, max_ld_diff);
    std::printf("serial   (1 at a time): %8.4f s\n", t_serial);
    std::printf("concurrent (all %2d)   : %8.4f s   -> %.2fx faster\n",
                n_calls, t_parallel, t_parallel > 0 ? t_serial / t_parallel : 0.0);
    std::printf("status          : %s\n\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
