// scotch_iso.cpp — isolated SCOTCH ordering harness (NO MKL / no libstiles).
// Mirrors runSCOTCH_impl's non-context path with tree capture (what the bake-off
// uses for SCOTCH variants), so valgrind/ASan can check SCOTCH alone, fast.
//   ./scotch_iso <matrix.mtx> [seed]     seed 0=SCOTCH 7=FSCOTCH 42=SCOTCH2
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include "scotch.h"

// ---- load .mtx (one triangle, row>=col) ----
struct Coo { int n = 0, nnz = 0; std::vector<int> row, col; };
static bool load_mtx(const char* path, Coo& m) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return false; }
    std::string line; bool sym = false;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] == '%') { if (line.find("symmetric") != std::string::npos) sym = true; continue; }
        break;
    }
    int nr, nc, nnzf; std::istringstream hs(line); hs >> nr >> nc >> nnzf;
    m.n = nr; int r, c; double v;
    while (f >> r >> c >> v) { --r; --c; if (sym && r < c) std::swap(r, c); m.row.push_back(r); m.col.push_back(c); }
    m.nnz = (int)m.row.size(); return m.nnz > 0;
}

// ---- exact copy of sTiles' build_scotch_graph ----
static void build_scotch_graph(const int* rows, const int* cols, int nnz, int dim,
                               std::vector<SCOTCH_Num>& verttab,
                               std::vector<SCOTCH_Num>& edgetab, SCOTCH_Num& edgenbr) {
    std::vector<SCOTCH_Num> deg(dim, 0);
    for (int k = 0; k < nnz; ++k) { int r = rows[k], c = cols[k];
        if (r >= 0 && c >= 0 && r < dim && c < dim && r != c) { deg[r]++; deg[c]++; } }
    std::vector<SCOTCH_Num> raw_vtab(dim + 1, 0);
    for (int v = 0; v < dim; ++v) raw_vtab[v+1] = raw_vtab[v] + deg[v];
    std::vector<SCOTCH_Num> raw_etab(raw_vtab[dim]);
    std::vector<SCOTCH_Num> pos(dim, 0);
    for (int k = 0; k < nnz; ++k) { int r = rows[k], c = cols[k];
        if (r >= 0 && c >= 0 && r < dim && c < dim && r != c) {
            raw_etab[raw_vtab[r] + pos[r]++] = c; raw_etab[raw_vtab[c] + pos[c]++] = r; } }
    std::vector<SCOTCH_Num> new_deg(dim);
    for (int v = 0; v < dim; ++v) { auto* beg = raw_etab.data() + raw_vtab[v]; auto* end = beg + deg[v];
        std::sort(beg, end); new_deg[v] = (SCOTCH_Num)(std::unique(beg, end) - beg); }
    verttab.resize(dim + 1); verttab[0] = 0;
    for (int v = 0; v < dim; ++v) verttab[v+1] = verttab[v] + new_deg[v];
    edgenbr = verttab[dim];
    edgetab.resize(edgenbr);
    for (int v = 0; v < dim; ++v)
        std::memcpy(edgetab.data() + verttab[v], raw_etab.data() + raw_vtab[v], new_deg[v] * sizeof(SCOTCH_Num));
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <mtx> [seed]\n", argv[0]); return 2; }
    Coo m; if (!load_mtx(argv[1], m)) return 2;
    const int seed = (argc > 2) ? std::atoi(argv[2]) : 0;
    const int dim = m.n;                       // fixed_col = 0

    std::vector<SCOTCH_Num> verttab, edgetab; SCOTCH_Num edgenbr = 0;
    build_scotch_graph(m.row.data(), m.col.data(), m.nnz, dim, verttab, edgetab, edgenbr);
    std::fprintf(stderr, "iso: dim=%d edgenbr=%ld\n", dim, (long)edgenbr);

    SCOTCH_Graph graf; SCOTCH_graphInit(&graf);
    if (SCOTCH_graphBuild(&graf, 0, (SCOTCH_Num)dim, verttab.data(), verttab.data()+1,
                          nullptr, nullptr, edgenbr, edgetab.data(), nullptr) != 0) {
        std::fprintf(stderr, "graphBuild failed\n"); return 1; }

    SCOTCH_Strat strat; SCOTCH_stratInit(&strat);
    const char* qs;
    if (seed == 42) qs = "n{sep=m{vert=50,low=h{pass=20}f{bal=0.02},asc=b{bnd=f{bal=0.02},org=f{bal=0.02}}},ole=d{cmin=20,cmax=100000,frat=0},ose=g{pass=20}}";
    else if (seed == 7) qs = "n{sep=m{vert=200,low=h{pass=5}f{bal=0.10},asc=b{bnd=f{bal=0.10},org=f{bal=0.10}}},ole=d{cmin=100,cmax=100000,frat=0},ose=g{pass=5}}";
    else qs = "n{sep=m{vert=100,low=h{pass=10}f{bal=0.05},asc=b{bnd=f{bal=0.05},org=f{bal=0.05}}},ole=d{cmin=50,cmax=100000,frat=0},ose=g{pass=10}}";
    if (SCOTCH_stratGraphOrder(&strat, qs) != 0) { std::fprintf(stderr, "strat failed\n"); return 1; }

    SCOTCH_randomSeed((SCOTCH_Num)seed); SCOTCH_randomReset();

    std::vector<SCOTCH_Num> sp(dim), si(dim);
    SCOTCH_Num cblknbr = 0;
    std::vector<SCOTCH_Num> rangtab(dim + 1, 0), treetab(dim, 0);
    int rc = SCOTCH_graphOrder(&graf, &strat, sp.data(), si.data(), &cblknbr, rangtab.data(), treetab.data());
    std::fprintf(stderr, "iso: graphOrder rc=%d cblknbr=%ld\n", rc, (long)cblknbr);
    SCOTCH_stratExit(&strat); SCOTCH_graphExit(&graf);
    if (rc != 0) return 1;

    // Validate sp is a permutation of [0,dim) — this is what would scatter-write wild.
    std::vector<char> seen(dim, 0);
    int bad = 0;
    for (int i = 0; i < dim; ++i) { long p = sp[i];
        if (p < 0 || p >= dim) { if (bad < 8) std::fprintf(stderr, "  sp[%d]=%ld OUT OF RANGE\n", i, p); bad++; }
        else { if (seen[p]) { if (bad < 8) std::fprintf(stderr, "  sp[%d]=%ld DUP\n", i, p); bad++; } seen[p] = 1; } }
    if (cblknbr < 0 || cblknbr > dim) std::fprintf(stderr, "  cblknbr=%ld OUT OF RANGE [0,%d]\n", (long)cblknbr, dim);
    std::fprintf(stderr, "iso: seed=%d  bad_perm_entries=%d  %s\n", seed, bad, bad ? "*** INVALID ***" : "ok");
    return bad ? 3 : 0;
}
