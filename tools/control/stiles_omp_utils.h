// File: sTiles_utils.h

#ifndef STILES_UTILS_H
#define STILES_UTILS_H

#include <stdio.h>
#include <omp.h>

#ifdef __linux__
#include <sched.h>
#include <errno.h>
#include <string.h>
#endif

// A simple macro for finding the maximum of two numbers
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/**
 * @brief Determines and sets the optimal number of OpenMP threads.
 *
 * This function sets the number of threads to the maximum of either the number
 * of available processors or the value configured by the user (e.g., via the
 * OMP_NUM_THREADS environment variable). This ensures full utilization while
 * respecting user overrides.
 *
 * @return The number of threads that were configured.
 */
static inline int sTiles_configure_thread_count() {
    int num_procs = omp_get_num_procs();
    int max_threads = omp_get_max_threads();
    int num_to_set = MAX(num_procs, max_threads);
    
    omp_set_num_threads(num_to_set);
    return num_to_set;
}


/**
 * @brief Initializes the scheduling policy for every thread in the OpenMP pool.
 *
 * This should be called after sTiles_configure_thread_count(). It creates a
 * parallel region where each thread sets its scheduling policy to SCHED_OTHER,
 * ensuring a consistent and predictable parallel environment.
 */
static inline void sTiles_initialize_thread_scheduler() {
    #pragma omp parallel
    {   
        struct sched_param param;
        param.sched_priority = 0;
        
        #ifdef __linux__
        if (sched_setscheduler(0, SCHED_OTHER, &param) == -1) {
            #pragma omp critical
            {
                // Use a critical section for thread-safe printing
                fprintf(stderr, "Warning: Thread %d failed to set scheduler: %s\n",
                        omp_get_thread_num(), strerror(errno));
            }
        }
        #endif

        // The master thread prints a single confirmation message
        #pragma omp master
        {
            printf("[sTiles Init] Scheduler initialized for all %d threads.\n", omp_get_num_threads());
        }
    }
}


#endif // STILES_UTILS_H