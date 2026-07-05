#pragma once
//
// Shared symmetric-adjacency graph for the external orderings.
//
// The bake-off in symbolic_phase runs several orderings (SCOTCH/ASCOTCH/FSCOTCH,
// METIS, AMD, CAMD) over the SAME matrix. Each wrapper today rebuilds the same
// symmetric, deduplicated adjacency graph from the COO independently — the SCOTCH
// family alone rebuilds it three times. That build is O(nnz log deg) and identical
// every time, so it is pure redundant work.
//
// build_shared_adj_csr() produces that graph ONCE in a canonical form:
//   - 0-based
//   - symmetric (both (i,j) and (j,i) present)
//   - diagonal excluded (no self-loops)
//   - each adjacency list sorted ascending and deduplicated
//   - integer type `int`
//
// This is, vertex-for-vertex, the graph SCOTCH (build_scotch_graph) and METIS
// (stiles_runMETIS_direct) already construct. AMD/CAMD currently see the same
// graph plus self-loops, but AMD/CAMD ignore diagonal entries, so feeding the
// diagonal-free graph leaves their ordering unchanged. COLAMD/RCM/ND keep their
// own builders (COLAMD's AᵀA semantics differ; RCM/ND are 1-based / specialised).
//
// Consumers whose library integer type is wider than int (64-bit SCOTCH_Num /
// idx_t builds) cast-copy into a typed buffer — an O(edges) linear pass, far
// cheaper than the sort+dedup it replaces. When the widths match (the common
// 32-bit build) the arrays are used directly.
//
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstddef>

namespace sTiles {

struct SharedAdjCSR {
    int dim = 0;
    std::vector<int> xadj;    // size dim+1, 0-based offsets
    std::vector<int> adjncy;  // size xadj[dim], 0-based neighbours, sorted+unique

    // A shared graph is usable by a wrapper only if it was built for the same
    // active dimension (dim = N - fixed_col) the wrapper is about to order.
    bool valid_for(int d) const {
        return dim == d && static_cast<int>(xadj.size()) == d + 1;
    }
};

// Build the canonical graph from a COO over the active block [0,dim).
// Entries touching the trailing fixed/dense columns (endpoint >= dim) or out of
// range are dropped — exactly as each wrapper's own builder does.
// num_cores parallelises the two dominant, embarrassingly-parallel passes (the
// per-vertex sort+unique and the compact copy). The result is independent of
// thread scheduling — each vertex slice is sorted in isolation — so the graph is
// byte-identical to the serial build. The degree count and edge fill stay serial
// (both are cheap linear passes; the fill carries a per-vertex cursor dependency
// that is not safe to parallelise over edges). The build runs before the ordering
// fan-out, so all num_cores are free at that point.
inline SharedAdjCSR build_shared_adj_csr(const int* rows, const int* cols,
                                         int nnz, int dim, int num_cores = 1) {
    SharedAdjCSR g;
    g.dim = dim;
    if (dim <= 0) { g.xadj.assign(1, 0); return g; }

    // Pass 1: raw symmetric degree (off-diagonal, in-range only).
    std::vector<int> deg(dim, 0);
    for (int k = 0; k < nnz; ++k) {
        const int r = rows[k], c = cols[k];
        if (r >= 0 && c >= 0 && r < dim && c < dim && r != c) { ++deg[r]; ++deg[c]; }
    }

    // Raw offsets (pre-dedup, symmetric -> 2 entries per off-diagonal nz).
    std::vector<int> raw(dim + 1, 0);
    for (int v = 0; v < dim; ++v) raw[v + 1] = raw[v] + deg[v];

    // Pass 2: fill raw edges via per-vertex cursors.
    std::vector<int> ebuf(static_cast<std::size_t>(raw[dim]));
    std::vector<int> pos(dim, 0);
    for (int k = 0; k < nnz; ++k) {
        const int r = rows[k], c = cols[k];
        if (r >= 0 && c >= 0 && r < dim && c < dim && r != c) {
            ebuf[raw[r] + pos[r]++] = c;
            ebuf[raw[c] + pos[c]++] = r;
        }
    }

    // Sort + unique each vertex's slice. Disjoint per-vertex ranges -> parallel-safe.
    std::vector<int> ndeg(dim);
    #pragma omp parallel for num_threads(num_cores) schedule(dynamic, 256) if(num_cores > 1)
    for (int v = 0; v < dim; ++v) {
        int* b = ebuf.data() + raw[v];
        int* e = b + deg[v];
        std::sort(b, e);
        ndeg[v] = static_cast<int>(std::unique(b, e) - b);
    }

    // Compact into the final deduplicated CSR. The prefix sum is sequential; the
    // per-vertex copies write disjoint destination ranges -> parallel-safe.
    g.xadj.assign(dim + 1, 0);
    for (int v = 0; v < dim; ++v) g.xadj[v + 1] = g.xadj[v] + ndeg[v];
    g.adjncy.resize(static_cast<std::size_t>(g.xadj[dim]));
    #pragma omp parallel for num_threads(num_cores) schedule(dynamic, 256) if(num_cores > 1)
    for (int v = 0; v < dim; ++v)
        std::memcpy(g.adjncy.data() + g.xadj[v], ebuf.data() + raw[v],
                    static_cast<std::size_t>(ndeg[v]) * sizeof(int));
    return g;
}

} // namespace sTiles
