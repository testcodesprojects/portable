/**
 * @file stiles_threadsettings.c
 *
 * Source file for LAPACK threading management in sTiles.
 *
 * sTiles is an advanced extension of the PLASMA software package, originally developed by:
 * - University of Tennessee
 * - University of California, Berkeley
 * - University of Colorado, Denver
 *
 * The sTiles framework has been redesigned and improved by Esmail Abdul Fattah
 * at King Abdullah University of Science and Technology (KAUST) and the sTiles team.
 *
 * This file provides utilities to manage threading behavior for LAPACK operations,
 * enabling seamless integration with various threading libraries such as ACML,
 * MKL, and OpenBLAS. Functions include setting thread counts and managing affinities
 * for multi-threaded and sequential modes.
 *
 * @version 1.0.0
 * @redesigned_by Esmail Abdul Fattah
 * @original_authors Azzam Haidar
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @date 2025-01-30
 */


#include "common.h"
/***************************************************************************//**
 * switch lapack thread_num initialization
 **/
#if defined(STILES_WITH_ACML) || defined(STILES_WITH_MKL)
#include <omp.h>
#endif

#if defined(STILES_WITH_MKL)
#include <mkl_service.h>
#endif

#if defined(STILES_WITH_OPENBLAS)
    void openblas_set_num_threads(int threads);
#endif
/////////////////////////////////////////////////////////////
static inline void stiles_setlapack_numthreads(int num_threads)
{
#if defined(STILES_WITH_ACML)
    omp_set_num_threads( num_threads );
#elif defined(STILES_WITH_MKL)
    mkl_set_num_threads( num_threads );
#elif defined(STILES_WITH_OPENBLAS)
    openblas_set_num_threads( num_threads );
#else
    if (num_threads > 1) {
        sTiles::Logger::warning(
            "Attempting to use multiple threads without a linked multithreaded BLAS/LAPACK library. "
            "Performance will be sequential. Please link with MKL, OpenBLAS, etc., and enable OpenMP."
        );
    }
#endif
}
/***************************************************************************//**
 * switch lapack to multi threading and unset stile affinity
 **/
void stiles_setlapack_multithreads(int num_threads)
{
    sTiles::Control::UnsetAffinity();
    stiles_setlapack_numthreads(num_threads);
}

/***************************************************************************//**
 * switch lapack to sequential and setting stile affinity
 **/
void stiles_setlapack_sequential(stiles_context_t *stile)
{
    /*
     * All threads need to call the parallel section that
     * set omp/mkl numthreads to 1, then only the master (0) will
     * set stile affinity
     */
    sTiles::static_call(stile, stiles_psetlapack_numthreads);
    sTiles::Control::SetAffinity(stile->thread_bind[0]);
}

void stiles_psetlapack_numthreads(stiles_context_t *stile)
{
    stiles_setlapack_numthreads(1);
}
