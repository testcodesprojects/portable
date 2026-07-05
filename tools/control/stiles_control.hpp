#ifndef OMP_DEP_TRACKER_H
#define OMP_DEP_TRACKER_H

#include <atomic> // The core of this solution
#include <sched.h> // For sched_yield()

// --- 1. Hardware-Specific Pause Instruction ---
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #include <immintrin.h>
    #define hpc_cpu_pause() _mm_pause()
#elif defined(__aarch64__)
    #define hpc_cpu_pause() asm volatile("yield")
#else
    #define hpc_cpu_pause() ((void)0)
#endif

// Hybrid pause: spin with CPU pause first, then yield to OS after threshold.
// Threshold = 1024 pauses (~1 µs on modern x86) matches the pthreads
// STILES_WAIT_ADAPTIVE_BUDGET in common.h. Most inter-task waits in the chol
// task graph are sub-microsecond; the old value of 64 (~64 ns) sent OMP into
// sched_yield() far too aggressively and made short waits ~10× more expensive
// than the equivalent pthreads path.
#ifndef HPC_SPIN_THRESHOLD
#  define HPC_SPIN_THRESHOLD 1024
#endif
static inline void hpc_pause_hybrid(int& spin_count) {
    if (spin_count < HPC_SPIN_THRESHOLD) {
        hpc_cpu_pause();
        ++spin_count;
    } else {
        sched_yield();
        spin_count = 0;
    }
}


// --- 2. The Struct Definition using C++ Atomics ---
// This is the key change. We now use the standard atomic types.
typedef struct OmpDependencyTracker {
    int ld;
    std::atomic<int> abort_flag;
    std::atomic<int>* progress_table;
    int workspace_offset;
    // Byte-progress variant (sparse): one atomic byte per active tile,
    // indexed via task indexN (not (i,j)). Used by chol kernels whose
    // dependency values are restricted to {0, 1}. See dep_init_byte /
    // dep_set_done_b / dep_wait_for_b below.
    std::atomic<unsigned char>* slots;
    int nslots;
} omp_dep_tracker_t;


// --- 3. The High-Performance Macros using C++ Atomics ---

/**
 * @brief Initializes the dependency tracker within an OpenMP parallel region.
 * Must be called by all threads, but only one thread will perform the allocation.
 * @param m The first dimension of the progress table.
 * @param n The second dimension of the progress table.
 * @param init_val The initial value to set for all progress entries.
 */
#define dep_init(m, n, init_val) \
    _Pragma("omp single") \
    { \
        dep_tracker->progress_table = new std::atomic<int>[(m)*(n)]; \
        for (int i = 0; i < (m)*(n); ++i) { \
            dep_tracker->progress_table[i].store((init_val), std::memory_order_relaxed); \
        } \
        dep_tracker->ld = (m); \
        dep_tracker->abort_flag.store(0, std::memory_order_relaxed); \
    } \
    _Pragma("omp barrier")

/**
 * @brief Finalizes and cleans up the dependency tracker resources.
 * Must be called by all threads within an OpenMP parallel region.
 */
#define dep_finalize() \
    _Pragma("omp barrier") \
    _Pragma("omp single") \
    { \
        delete[] dep_tracker->progress_table; \
        dep_tracker->progress_table = nullptr; \
    }

/**
 * @brief Sparse byte-progress variant of dep_init.
 * Allocates an atomic<unsigned char>[nslots] table (typically nslots ==
 * numActiveTiles) instead of an int[m*n] dense grid. Indexed via the
 * task's indexN, NOT by (i, j). Only valid when stored values fit in a
 * single byte (the chol semi paths only ever store {0, 1}).
 */
/* NB: macro arg is `nslots_arg`, not `nslots`, to avoid colliding with
 * the OmpDependencyTracker struct's `nslots` field when expanded inside
 * `dep_tracker->nslots = ...`.
 *
 * The slots array is the SCHEME-RESIDENT persistent atomic buffer
 * (allocated once in sTiles_init_group, freed at scheme destruction).
 * `dep_init_byte` rebinds dep_tracker->slots to it and stores the init
 * value across the first nslots entries; `dep_finalize_byte` just nulls
 * the pointer. No per-chol new/delete. */
#define dep_init_byte(scheme, nslots_arg, init_val) \
    _Pragma("omp single") \
    { \
        dep_tracker->slots = (scheme)->byte_progress_buf_omp; \
        for (int i = 0; i < (nslots_arg); ++i) { \
            dep_tracker->slots[i].store((unsigned char)(init_val), std::memory_order_relaxed); \
        } \
        dep_tracker->nslots = (nslots_arg); \
        dep_tracker->abort_flag.store(0, std::memory_order_relaxed); \
    } \
    _Pragma("omp barrier")

#define dep_finalize_byte() \
    _Pragma("omp barrier") \
    _Pragma("omp single") \
    { \
        dep_tracker->slots = nullptr; \
        dep_tracker->nslots = 0; \
    }

#define dep_set_done_b(slot, val) \
    dep_tracker->slots[(slot)].store((unsigned char)(val), std::memory_order_release)

#define dep_wait_for_b(slot, val) \
    { \
        int _spin_count = 0; \
        while (dep_tracker->slots[(slot)].load(std::memory_order_relaxed) != (unsigned char)(val)) { \
            if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break; \
            hpc_pause_hybrid(_spin_count); \
        } \
        std::atomic_thread_fence(std::memory_order_acquire); \
        if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) \
            break; \
    }

/**
 * @brief Producer: Signals that a task at coordinate (m, n) is complete.
 * Uses a 'release' memory order to ensure that all memory writes made by this
 * thread before this call are visible to any consumer thread that sees the new value.
 * @param m The first coordinate of the task.
 * @param n The second coordinate of the task.
 * @param val The value to set, signaling completion.
 */
#define dep_set_done(m, n, val) \
    dep_tracker->progress_table[(m) + dep_tracker->ld * (n)].store((val), std::memory_order_release)

/**
 * @brief Consumer: Waits for a task at coordinate (m, n) to reach the specified value.
 * Uses hybrid spinning: fast CPU pause for short waits, OS yield for longer waits.
 * @param m The first coordinate of the task to wait for.
 * @param n The second coordinate of the task to wait for.
 * @param val The value to wait for.
 */
#define dep_wait_for(m, n, val) \
    { \
        int _spin_count = 0; \
        /* Spin with relaxed load (no per-iteration fence) and place a single \
         * acquire fence on success. Functionally equivalent to a per-iteration \
         * acquire load on x86 TSO, but avoids the explicit fence cost on each \
         * spin iteration. Matches the pthreads volatile-int spin behavior. */ \
        while (dep_tracker->progress_table[(m) + dep_tracker->ld * (n)].load(std::memory_order_relaxed) != (val)) { \
            if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break; \
            hpc_pause_hybrid(_spin_count); \
        } \
        std::atomic_thread_fence(std::memory_order_acquire); \
        if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) \
            break; \
    }

/**
 * @brief Checks if the abort flag has been set.
 * @return Non-zero if an abort has been signaled, zero otherwise.
 */
#define dep_is_aborted() (dep_tracker->abort_flag.load(std::memory_order_relaxed))

/**
 * @brief Signals all waiting threads to stop their work and break out of their loops.
 * Uses a 'release' memory order to ensure visibility to all consumer threads.
 */
#define dep_abort_all() dep_tracker->abort_flag.store(1, std::memory_order_release)

/**
 * @brief Consumer: Waits for a tree dependency array entry to reach the specified value.
 * Similar to dep_wait_for but operates on a separate dependency array (not the progress_table).
 * @param m The index in the dependency array.
 * @param val The value to wait for.
 * @param dep_array Pointer to the dependency array (volatile int*).
 */
#define dep_wait_for_tree(m, val, dep_array) \
    { \
        int _spin_count = 0; \
        while (dep_array[m] != (val)) { \
            if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) break; \
            hpc_pause_hybrid(_spin_count); \
        } \
        std::atomic_thread_fence(std::memory_order_acquire); \
        if (dep_tracker->abort_flag.load(std::memory_order_relaxed)) \
            break; \
    }


#endif // OMP_DEP_TRACKER_H