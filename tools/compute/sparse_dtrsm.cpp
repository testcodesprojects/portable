/**
 * @file sparse_dtrsm.cpp
 * @brief Sparse-path LL^T / L / L^T solve shims. solve is currently serial;
 *        omp and pthreads variants do the same thing for symmetry with the
 *        dense path's two-axis dispatch.
 */

#include "../sparse/api.hpp"
#include "../common/stiles_logger.hpp"
#include "../common/stiles_structs.hpp"
#include "../common/stiles_types.hpp"

namespace sTiles {

static StatusCode sparse_dtrsm_impl(TiledMatrix* scheme,
                                    double* B, int nrhs, int ldb) {
    if (!scheme || !scheme->sparse_handle) return StatusCode::IllegalValue;
    const int rc = sTiles::sparse::api::solve_LLT(&scheme->sparse_handle, B, nrhs, ldb, scheme->tile_size);
    return (rc == 0) ? StatusCode::Success : StatusCode::Failure;
}

static StatusCode sparse_dtrsm_forward_impl(TiledMatrix* scheme,
                                            double* B, int nrhs, int ldb) {
    if (!scheme || !scheme->sparse_handle) return StatusCode::IllegalValue;
    const int rc = sTiles::sparse::api::solve_L(&scheme->sparse_handle, B, nrhs, ldb, scheme->tile_size);
    return (rc == 0) ? StatusCode::Success : StatusCode::Failure;
}

static StatusCode sparse_dtrsm_backward_impl(TiledMatrix* scheme,
                                             double* B, int nrhs, int ldb) {
    if (!scheme || !scheme->sparse_handle) return StatusCode::IllegalValue;
    const int rc = sTiles::sparse::api::solve_LT(&scheme->sparse_handle, B, nrhs, ldb, scheme->tile_size);
    return (rc == 0) ? StatusCode::Success : StatusCode::Failure;
}

StatusCode pthreads_sparse_dtrsm(int /*global_index*/, TiledMatrix* scheme,
                                 double* B, int nrhs, int ldb) {
    return sparse_dtrsm_impl(scheme, B, nrhs, ldb);
}

StatusCode omp_sparse_dtrsm(int /*global_index*/, TiledMatrix* scheme,
                            double* B, int nrhs, int ldb) {
    return sparse_dtrsm_impl(scheme, B, nrhs, ldb);
}

StatusCode pthreads_sparse_dtrsm_forward(int /*global_index*/, TiledMatrix* scheme,
                                         double* B, int nrhs, int ldb) {
    return sparse_dtrsm_forward_impl(scheme, B, nrhs, ldb);
}

StatusCode omp_sparse_dtrsm_forward(int /*global_index*/, TiledMatrix* scheme,
                                    double* B, int nrhs, int ldb) {
    return sparse_dtrsm_forward_impl(scheme, B, nrhs, ldb);
}

StatusCode pthreads_sparse_dtrsm_backward(int /*global_index*/, TiledMatrix* scheme,
                                          double* B, int nrhs, int ldb) {
    return sparse_dtrsm_backward_impl(scheme, B, nrhs, ldb);
}

StatusCode omp_sparse_dtrsm_backward(int /*global_index*/, TiledMatrix* scheme,
                                     double* B, int nrhs, int ldb) {
    return sparse_dtrsm_backward_impl(scheme, B, nrhs, ldb);
}

} // namespace sTiles
