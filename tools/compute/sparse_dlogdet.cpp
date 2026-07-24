/**
 * @file sparse_dlogdet.cpp
 * @brief Sparse-path log(det(A)) shim. logdet just sums diagonal cells of
 *        L_cs, so omp/pthreads variants do the same thing — symmetry with
 *        the dense path's two-axis dispatch.
 */

#include "../sparse/api.hpp"
#ifdef STILES_MFRONT
#include "../mfront/api.hpp"
#endif
#include "../common/stiles_structs.hpp"

namespace sTiles {

double pthreads_sparse_dlogdet(TiledMatrix* scheme) {
    if (!scheme || !scheme->sparse_handle) return 0.0;
#ifdef STILES_MFRONT
    return (scheme->sparse_backend == 4)
        ? sTiles::mfront::api::get_logdet(&scheme->sparse_handle)
        : sTiles::sparse::api::get_logdet(&scheme->sparse_handle);
#else
    return sTiles::sparse::api::get_logdet(&scheme->sparse_handle);
#endif
}

double omp_sparse_dlogdet(TiledMatrix* scheme) {
    if (!scheme || !scheme->sparse_handle) return 0.0;
#ifdef STILES_MFRONT
    return (scheme->sparse_backend == 4)
        ? sTiles::mfront::api::get_logdet(&scheme->sparse_handle)
        : sTiles::sparse::api::get_logdet(&scheme->sparse_handle);
#else
    return sTiles::sparse::api::get_logdet(&scheme->sparse_handle);
#endif
}

} // namespace sTiles
