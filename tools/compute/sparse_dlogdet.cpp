/**
 * @file sparse_dlogdet.cpp
 * @brief Sparse-path log(det(A)) shim. logdet just sums diagonal cells of
 *        L_cs, so omp/pthreads variants do the same thing — symmetry with
 *        the dense path's two-axis dispatch.
 */

#include "../sparse/api.hpp"
#include "../common/stiles_structs.hpp"

namespace sTiles {

double pthreads_sparse_dlogdet(TiledMatrix* scheme) {
    if (!scheme || !scheme->sparse_handle) return 0.0;
    return sTiles::sparse::api::get_logdet(&scheme->sparse_handle);
}

double omp_sparse_dlogdet(TiledMatrix* scheme) {
    if (!scheme || !scheme->sparse_handle) return 0.0;
    return sTiles::sparse::api::get_logdet(&scheme->sparse_handle);
}

} // namespace sTiles
