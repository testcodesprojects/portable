/**
 * @file api.hpp
 * @brief Internal helpers that drive the non-uniform tile Cholesky path
 *        from inside sTiles' existing C-ABI entry points.
 *
 * Conceptually this is the same idea as the semisparse path (mode 1) — both
 * partition L into a 2D grid of dense rectangular blocks indexed by
 * (row-group, column-group) — but with a different cell-sizing rule:
 *
 *   semisparse  (mode 1): uniform tile_size × tile_size, per-column bitmap
 *                         marks which entries are actually non-zero
 *   non-uniform (mode 2): variable cell sizes; cell width = column-supernode
 *                         width, cell rows = row-supernode run length;
 *                         each cell is fully dense (sparsity captured by
 *                         the cell layout, not within cells)
 *
 * These are NOT a public API. There is no `sTiles_sparse_*` C ABI — users
 * call the standard sTiles_* entry points (`sTiles_assign_graph_one_call`,
 * `sTiles_init_group`, `sTiles_assign_values`, `sTiles_chol`, …) and those
 * route here when `tile_type_mode == 2` and `scheme->sparse_handle` is set.
 *
 * Storage model: the non-uniform tile module exposes its objects as C++
 * classes in `namespace sTiles::sparse` (Etree / Symbolic / CellStore /
 * SpsState). This API hides them behind opaque `void*` handles attached
 * to each TiledMatrix scheme.
 *
 * Constraints inherited from the sparse module (see tools/sparse/...):
 *   - double real SPD only
 *   - LL^T Cholesky only
 *   - User permutation always — sTiles' ordering pipeline
 *     (tools/ordering/stiles_ordering.hpp) supplies it via
 *     `scheme->element_perm`; the module never picks its own.
 */

#ifndef _STILES_SPARSE_API_HPP_
#define _STILES_SPARSE_API_HPP_

namespace sTiles { namespace sparse { namespace api {

// ── Lifecycle ────────────────────────────────────────────────────────────────
int  init();
void quit();
int  create(void** obj, int num_cores);
void freeGroup(void** obj);

// ── Configuration ────────────────────────────────────────────────────────────
int  set_user_permutation(void** obj, const int* perm, int n);
void set_max_supernode(void** obj, int max_snode);
void set_group_id(void** obj, int group_id);

// ── Symbolic + numeric phases ────────────────────────────────────────────────
int  assign_graph(void** obj, int n, int nnz, const int* row, const int* col);
int  assign_values(void** obj, const double* values);
int  chol_pthreads(void** obj);
int  chol_omp(void** obj);
int  selinv_pthreads(void** obj);
int  selinv_omp(void** obj);
int  solve_LLT(void** obj, double* b, int nrhs, int ldb, int tile_size = 0);
int  solve_L  (void** obj, double* b, int nrhs, int ldb, int tile_size = 0);
int  solve_LT (void** obj, double* b, int nrhs, int ldb, int tile_size = 0);

// Reset Z_cs to zero (no-op for diagonal — selinv repopulates the diagonal
// from L_cs.diag at the start of every selinv pass). Caller is expected to
// invoke selinv again before reading get_selinv_elm.
int  clear_selinv(void** obj);

// ── Element access ───────────────────────────────────────────────────────────
double    get_chol_elm(void** obj, int i, int j);
double    get_selinv_elm(void** obj, int i, int j);
double    get_logdet(void** obj);
long long get_nnz_factor(void** obj);

}}} // namespace sTiles::sparse::api

#endif // _STILES_SPARSE_API_HPP_
