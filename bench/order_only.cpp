/**
 * order_only.cpp — run ONLY the symbolic/ordering phase (the SCOTCH bake-off)
 * and quit, so a memory checker (valgrind/ASan) focuses on the ordering and
 * doesn't grind through the numeric factorization. Usage: ./order_only m.mtx [cores] [tile]
 */
#include <stiles.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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
    if (nr != nc) { std::fprintf(stderr, "not square\n"); return false; }
    m.n = nr;
    int r, c; double v;
    while (f >> r >> c >> v) { --r; --c; if (sym && r < c) std::swap(r, c); m.row.push_back(r); m.col.push_back(c); }
    m.nnz = (int)m.row.size();
    return m.nnz > 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <matrix.mtx> [cores] [tile]\n", argv[0]); return 2; }
    const char* path = argv[1];
    const int cores = (argc > 2) ? std::atoi(argv[2]) : 4;
    const int tile  = (argc > 3) ? std::atoi(argv[3]) : 40;

    Coo m;
    if (!load_mtx(path, m)) return 2;

    setenv("STILES_NO_BANNER", "1", 0);
    setenv("STILES_LOG_LEVEL", "0", 0);

    sTiles_expert_user();
    sTiles_set_log_level(-1);
    sTiles_set_tile_size(tile);
    sTiles_set_tile_type_mode(3);      // auto

    const int  calls[]   = {1};
    const int  cpg[]     = {cores};
    const int  choltype[]= {0};
    const bool getinv[]  = {true};
    void* st = nullptr;
    sTiles_create(&st, 1, calls, cpg, choltype, getinv);
    sTiles_assign_graph_one_call(0, 0, &st, m.n, m.nnz, m.row.data(), m.col.data());
    sTiles_init_group(0, &st);         // <-- symbolic: the ordering bake-off (SCOTCH) runs here
    std::fprintf(stderr, "order_only: init_group done (n=%d nnz=%d)\n", m.n, m.nnz);
    sTiles_quit();
    return 0;
}
