/**
 * speedtest.cpp — minimal sTiles speed + correctness smoke test.
 *
 * Loads one sparse SPD matrix (Matrix Market .mtx), runs the full INLA path in
 * AUTO mode (symbolic -> Cholesky -> log-determinant -> selected inverse ->
 * solve), times each phase, and verifies the result by the solve residual
 * ||A x - 1||_inf (x = A^-1 * ones). Prints one row and exits 0 on PASS,
 * 1 on FAIL. Doubles as a minimal usage example of the public sTiles API.
 *
 * Usage:  ./speedtest <matrix.mtx> [cores] [tile_size]
 */
#include <stiles.h>

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

// Matrix Market loader. sTiles wants a single triangle with row >= col, so we
// normalize symmetric entries into it (never mirror). Same convention as the
// benchmark driver in run/bench.cpp.
static bool load_mtx(const char* path, Coo& m) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return false; }
    std::string line; bool sym = false;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] == '%') {
            if (line.find("symmetric") != std::string::npos) sym = true;
            continue;
        }
        break;
    }
    int nr, nc, nnzf; std::istringstream hs(line); hs >> nr >> nc >> nnzf;
    if (nr != nc) { std::fprintf(stderr, "matrix not square\n"); return false; }
    m.n = nr;
    int r, c; double v;
    while (f >> r >> c >> v) {
        --r; --c;
        if (sym && r < c) std::swap(r, c);
        m.row.push_back(r); m.col.push_back(c); m.val.push_back(v);
    }
    m.nnz = (int)m.row.size();
    return m.nnz > 0;
}

static const char* basename_of(const char* p) {
    const char* s = std::strrchr(p, '/');
    return s ? s + 1 : p;
}

// Manufactured solution X_TRUE (n x nrhs, column-major; distinct per column) and
// B = A*X_TRUE (A stored as one triangle -> mirror). Same convention as the
// reference harness run/accuracy/check_chol_selinv_paths.cpp, so the residuals
// here line up with its res(r=1)/res(r=T+1) columns.
static std::vector<double> make_x_true(int n, int nrhs) {
    std::vector<double> x((size_t)n * nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            x[(size_t)j * n + i] = std::sin(0.1 * i + 0.37 * j) + 1.0;
    return x;
}
static std::vector<double> sym_matvec(const Coo& m, const std::vector<double>& x, int nrhs) {
    std::vector<double> b((size_t)m.n * nrhs, 0.0);
    for (int j = 0; j < nrhs; ++j) {
        double* bc = b.data() + (size_t)j * m.n;
        const double* xc = x.data() + (size_t)j * m.n;
        for (int k = 0; k < m.nnz; ++k) {
            const int r = m.row[k], c = m.col[k];
            bc[r] += m.val[k] * xc[c];
            if (r != c) bc[c] += m.val[k] * xc[r];
        }
    }
    return b;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <matrix.mtx> [cores] [tile_size]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    const int cores = (argc > 2) ? std::atoi(argv[2]) : 4;
    const int tile  = (argc > 3) ? std::atoi(argv[3]) : 40;

    Coo m;
    if (!load_mtx(path, m)) return 2;

    // Keep stdout to just our result row (banner + [TIME] lines off).
    // Portable setenv: MinGW has no POSIX setenv (uses _putenv_s).
#if defined(_WIN32) || defined(_WIN64)
    _putenv_s("STILES_NO_BANNER", "1");
    _putenv_s("STILES_LOG_LEVEL", "0");
#else
    setenv("STILES_NO_BANNER", "1", /*overwrite=*/0);
    setenv("STILES_LOG_LEVEL", "0", /*overwrite=*/0);
#endif

    // ---- Configure (auto mode: the selector picks dense/semi/sparse) --------
    sTiles_expert_user();
    sTiles_set_log_level(-1);          // silence library [TIME]/[INFO] output
    sTiles_set_tile_size(tile);
    sTiles_set_tile_type_mode(3);      // 3 = auto

    const int  calls[]   = {1};
    const int  cpg[]     = {cores};
    const int  choltype[]= {0};        // 0 = sparse variant (auto resolves it)
    const bool getinv[]  = {true};
    void* st = nullptr;
    sTiles_create(&st, 1, calls, cpg, choltype, getinv);
    sTiles_assign_graph_one_call(0, 0, &st, m.n, m.nnz, m.row.data(), m.col.data());

    // ---- Symbolic ----
    auto t0 = clk::now();
    sTiles_init_group(0, &st);
    auto t1 = clk::now();

    sTiles_assign_values(0, 0, &st, m.val.data());

    // ---- Numeric Cholesky ----
    auto t2 = clk::now();
    sTiles_bind(0, 0, &st); sTiles_chol(0, 0, &st); sTiles_unbind(0, 0, &st);
    auto t3 = clk::now();

    const double  logdet = sTiles_get_logdet(0, 0, &st);
    const long long nnzL = sTiles_get_nnz_factor(0, 0, &st);

    // ---- Selected inverse ----
    auto t4 = clk::now();
    sTiles_bind(0, 0, &st); sTiles_selinv(0, 0, &st); sTiles_unbind(0, 0, &st);
    auto t5 = clk::now();

    // ---- Selected-inverse CORRECTNESS (Z is fresh here) ----------------------
    // selinv is only TIMED above; nothing gates its VALUES. Cross-check a spread
    // of diag(Q^-1) entries against ground truth from the (residual-gated) solve
    // path: (Q^-1)[i][i] == (solve Q x = e_i)[i]. The diagonal is always in the
    // factor pattern, so selinv returns it exactly. Read Z FIRST (before the
    // reference solve can touch internal state); one multi-RHS solve does all K.
    double zerr = 0.0;
    {
        const int    K      = (m.n < 16 ? m.n : 16);
        const size_t stride = (size_t)m.n;                 // column stride
        std::vector<int>    probe(K);
        std::vector<double> zii(K);
        for (int k = 0; k < K; ++k) {
            probe[k] = (int)((long long)k * m.n / K);      // spread across [0,n)
            zii[k]   = sTiles_get_selinv_elm(0, 0, probe[k], probe[k], &st);
        }
        std::vector<double> Bd(stride * (size_t)K, 0.0);
        for (int k = 0; k < K; ++k) Bd[(size_t)k * stride + probe[k]] = 1.0;   // e_{probe[k]}
        sTiles_bind(0, 0, &st);
        sTiles_solve_LLT(0, 0, &st, Bd.data(), K);         // in place -> columns of Q^-1
        sTiles_unbind(0, 0, &st);
        for (int k = 0; k < K; ++k) {
            const double truth = Bd[(size_t)k * stride + probe[k]];           // (Q^-1)[i][i]
            zerr = std::fmax(zerr, std::fabs(zii[k] - truth) / std::fmax(std::fabs(truth), 1e-300));
        }
    }

    // ---- Solve A X = B at two RHS widths: 1 and tile+1 (e.g. 41 for tile 40).
    // Manufactured X_TRUE, B = A*X_TRUE, solve in place, worst-column relative
    // residual ||x - x_true|| / ||x_true||. Best-of-3 (the first solve pays a
    // one-time lazy setup on the sparse path, so a single cold solve misleads).
    const size_t nn = (size_t)m.n;
    auto solve_resid = [&](int nrhs, double& t_out) -> double {
        std::vector<double> xt = make_x_true(m.n, nrhs);
        std::vector<double> b  = sym_matvec(m, xt, nrhs);
        std::vector<double> work;
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            work = b;                               // fresh RHS each rep
            auto s0 = clk::now();
            sTiles_solve_LLT(0, 0, &st, work.data(), nrhs);   // in-place -> x
            best = std::fmin(best, secs(s0, clk::now()));
        }
        t_out = best;
        double worst = 0.0;
        for (int j = 0; j < nrhs; ++j) {
            double se = 0.0, sr = 0.0;
            const double* xc = xt.data()   + (size_t)j * nn;
            const double* wc = work.data() + (size_t)j * nn;
            for (size_t i = 0; i < nn; ++i) { double e = wc[i]-xc[i]; se += e*e; sr += xc[i]*xc[i]; }
            worst = std::fmax(worst, std::sqrt(se / std::fmax(sr, 1e-300)));
        }
        return worst;
    };

    double t_slv1 = 0.0, t_slvT = 0.0;
    sTiles_bind(0, 0, &st);
    const double res1 = solve_resid(1,        t_slv1);   // rhs = 1
    const double resT = solve_resid(tile + 1, t_slvT);   // rhs = tile+1 (41 by default)
    sTiles_unbind(0, 0, &st);

    sTiles_quit();

    // ---- Verdict ----
    const bool ok = std::isfinite(logdet) && res1 < 1e-6 && resT < 1e-6 && zerr < 1e-6;
    const double fill = m.nnz ? (double)nnzL / m.nnz : 0.0;

    std::printf("%-14s %8d %10d %6.2f %8.4f %8.4f %8.4f %8.4f %15.4f %10.2e %10.2e %10.2e  %s\n",
                basename_of(path), m.n, m.nnz, fill,
                secs(t2, t3),   // chol
                secs(t4, t5),   // selinv
                t_slv1,         // solve rhs=1
                t_slvT,         // solve rhs=tile+1
                logdet, res1, resT,
                zerr,           // selinv diag error vs solve ground truth
                ok ? "PASS" : "FAIL");
    (void)t0; (void)t1;
    return ok ? 0 : 1;
}
