/**
 * @file sparse_dpotri.cpp
 * @brief Sparse-path selected-inverse shim. selinv is currently a serial
 *        Takahashi sweep, so omp/pthreads variants do the same thing — they
 *        exist for symmetry with the dense path's two-axis dispatch.
 */

#include "../sparse/api.hpp"
#ifdef STILES_MFRONT
#include "../mfront/api.hpp"
#endif
#include "../common/stiles_logger.hpp"
#include "../common/stiles_structs.hpp"
#include "../common/stiles_types.hpp"

namespace sTiles {

StatusCode pthreads_sparse_dpotri(int /*global_index*/, TiledMatrix* scheme) {
    if (!scheme || !scheme->sparse_handle) return StatusCode::IllegalValue;
#ifdef STILES_MFRONT
    const int rc = (scheme->sparse_backend == 4)
        ? sTiles::mfront::api::selinv_pthreads(&scheme->sparse_handle)
        : sTiles::sparse::api::selinv_pthreads(&scheme->sparse_handle);
#else
    const int rc = sTiles::sparse::api::selinv_pthreads(&scheme->sparse_handle);
#endif
    return (rc == 0) ? StatusCode::Success : StatusCode::Failure;
}

StatusCode omp_sparse_dpotri(int /*global_index*/, TiledMatrix* scheme) {
    if (!scheme || !scheme->sparse_handle) return StatusCode::IllegalValue;
#ifdef STILES_MFRONT
    const int rc = (scheme->sparse_backend == 4)
        ? sTiles::mfront::api::selinv_omp(&scheme->sparse_handle)
        : sTiles::sparse::api::selinv_omp(&scheme->sparse_handle);
#else
    const int rc = sTiles::sparse::api::selinv_omp(&scheme->sparse_handle);
#endif
    return (rc == 0) ? StatusCode::Success : StatusCode::Failure;
}

} // namespace sTiles
