/**
 * @file sparse_dpotrf.cpp
 * @brief Sparse-path Cholesky shim. Mirrors `pthreads_dpotrf` / `omp_dpotrf`
 *        in dpotrf.cpp: two parallel entry points selected by param[8]
 *        (UseOMP) at the sTiles_chol top-level dispatch.
 */

#include "../sparse/api.hpp"
#ifdef STILES_MFRONT
#include "../mfront/api.hpp"
#endif
#include "../common/stiles_logger.hpp"
#include "../common/stiles_structs.hpp"
#include "../common/stiles_types.hpp"

namespace sTiles {

StatusCode pthreads_sparse_dpotrf(int /*global_index*/, TiledMatrix* scheme) {
    if (!scheme || !scheme->sparse_handle) return StatusCode::IllegalValue;
#ifdef STILES_MFRONT
    const int rc = (scheme->sparse_backend == 4)
        ? sTiles::mfront::api::chol_pthreads(&scheme->sparse_handle)
        : sTiles::sparse::api::chol_pthreads(&scheme->sparse_handle);
#else
    const int rc = sTiles::sparse::api::chol_pthreads(&scheme->sparse_handle);
#endif
    return (rc == 0) ? StatusCode::Success : StatusCode::Failure;
}

StatusCode omp_sparse_dpotrf(int /*global_index*/, TiledMatrix* scheme) {
    if (!scheme || !scheme->sparse_handle) return StatusCode::IllegalValue;
#ifdef STILES_MFRONT
    const int rc = (scheme->sparse_backend == 4)
        ? sTiles::mfront::api::chol_omp(&scheme->sparse_handle)
        : sTiles::sparse::api::chol_omp(&scheme->sparse_handle);
#else
    const int rc = sTiles::sparse::api::chol_omp(&scheme->sparse_handle);
#endif
    return (rc == 0) ? StatusCode::Success : StatusCode::Failure;
}

} // namespace sTiles
