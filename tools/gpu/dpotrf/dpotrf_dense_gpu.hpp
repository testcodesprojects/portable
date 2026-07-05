/**
 * @file    dpotrf_dense_gpu.hpp
 * @brief   GPU versions of the four dense Cholesky expansion kernels.
 *
 *   pthreads_dpotrf_expansion_dense_serial             -> gpu_dpotrf_expansion_dense_serial
 *   omp_dpotrf_expansion_dense_serial                  -> gpu_dpotrf_expansion_dense_serial_omp
 *   pthreads_dpotrf_expansion_from_chol_tasks_dense_parallel
 *                                                       -> gpu_dpotrf_expansion_dense_parallel
 *   omp_dpotrf_expansion_from_chol_tasks_dense_parallel
 *                                                       -> gpu_dpotrf_expansion_dense_parallel_omp
 *
 * The serial variants enqueue every task on a single stream. The parallel
 * variants enqueue on stream[rank], pair every POTRF/TRSM completion with
 * a cudaEventRecord, and gate every cross-rank consumer through a
 * cudaStreamWaitEvent on top of the existing ss_/dep_ host flag. This
 * mirrors the CPU dependency graph exactly while keeping the GPU work
 * asynchronous.
 *
 * Sandbox: not yet wired into the build. The four functions assume that
 *   - scheme->dense_tiles_gpu[i].x has been bound to a GpuTileSlab via
 *     bind_slab_to_scheme(), and
 *   - the caller passes an initialized GpuPersistentContext.
 */
#ifndef STILES_GPU_DPOTRF_DENSE_GPU_HPP
#define STILES_GPU_DPOTRF_DENSE_GPU_HPP

#ifdef STILES_GPU

// Pull in full definitions for the types this header references as pointers.
// We can't forward-declare them as `struct TAG;` because they are defined as
// `typedef struct { ... } NAME;` (anonymous-struct typedefs), so a tag-named
// forward declaration creates a *different* type. Either we include the
// real definitions or we use the typedef name directly without a struct tag.
#include "../../control/stiles_control.hpp"   // omp_dep_tracker_t (typedef)
#include "../../common/stiles_structs.hpp"     // TiledMatrix      (typedef)
#include "../../control/common.h"              // stiles_context_t (typedef)

namespace sTiles { namespace gpu { namespace dpotrf {

struct GpuPersistentContext;

void gpu_dpotrf_expansion_dense_serial(TiledMatrix*           scheme,
                                       stiles_context_t*      stile,
                                       GpuPersistentContext&  ctx);

void gpu_dpotrf_expansion_dense_serial_omp(TiledMatrix*           scheme,
                                           omp_dep_tracker_t*     dep_tracker,
                                           GpuPersistentContext&  ctx);

void gpu_dpotrf_expansion_dense_parallel(TiledMatrix*           scheme,
                                         stiles_context_t*      stile,
                                         GpuPersistentContext&  ctx);

void gpu_dpotrf_expansion_dense_parallel_omp(TiledMatrix*           scheme,
                                             omp_dep_tracker_t*     dep_tracker,
                                             GpuPersistentContext&  ctx,
                                             int                    worldsize);

}}}  // namespace sTiles::gpu::dpotrf

#endif // STILES_GPU
#endif // STILES_GPU_DPOTRF_DENSE_GPU_HPP
