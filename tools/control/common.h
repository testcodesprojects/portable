/**
 * @file common.h
 *
 * Header file for shared utilities and global definitions in the sTiles framework.
 * These routines and definitions provide foundational support for tiled matrix computations
 * and include facilities for memory management, context handling, and thread settings.
 *
 * Redesigned by:
 * - Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST).
 *
 * Originally developed as part of the PLASMA project by:
 * - University of Tennessee
 * - University of California, Berkeley
 * - University of Colorado, Denver
 *
 * This file provides the following key functionalities:
 * - Global utilities for mathematical operations and memory management.
 * - Static schedulers and dependency management macros.
 * - Integration of various modules like context, control, and auxiliary routines.
 * - Definitions for tiled matrix properties and thread settings.
 *
 * @version 1.0.0
 * @author Esmail Abdul Fattah
 * @original_author Jakub Kurzak, Mathieu Faverge
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @date 2025-01-30
 *
 * @license Proprietary
 *
 * Copyright (c) 2025, Esmail Abdul Fattah, KAUST.
 * All rights reserved.
 *
 * This software is proprietary and confidential. Unauthorized copying, distribution, or modification 
 * of this software, via any medium, is strictly prohibited. Permission is granted to use the software 
 * in binary form for non-commercial purposes only, provided that this copyright notice and permission 
 * notice are included in all copies or substantial portions of the software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. 
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, 
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE 
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef _STILES_COMMON_H_
#define _STILES_COMMON_H_

#include "stiles_dispatch.h"
#include "context.h"
#include "control.h"
#include "stiles_core_blas.h"
#include "allocate.h"
#include "auxiliary.h"
#include "tile.h"
#include "stiles_threadsetting.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined( _WIN32 ) || defined( _WIN64 )
# include <io.h>
#else  /* NOT WIN32 */
# include <unistd.h>
#endif  /* IF WIN32 */

#define STILES_NB          stile->nb
//#define STILES_SCHEDULING  stile->scheduling
#define STILES_RANK        stiles_rank(stile)
#define STILES_SIZE        stile->world_size
#define STILES_GRPSIZE     stile->group_size
#define STILES_NBNBSIZE    stile->nbnbsize
#define STILES_RHBLK       stile->rhblock
#define STILES_TNT_MODE    stile->tournament
#define STILES_TNT_SIZE    stile->tntsize

/***************************************************************************//*
 *  Global utilities
/***************************************************************************/

#ifndef f_max
#define f_max(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef f_min
#define f_min(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef f_roundup
#define f_roundup(a, b) (b <= 0) ? (a) : (((a) + (b)-1) & ~((b)-1))
#endif

/***************************************************************************//*
 *  Static scheduler
/***************************************************************************/


#define ss_init(m, n, init_val) \
{ \
    if (STILES_RANK == 0) { \
        stile->ss_progress = (volatile int *)stiles_shared_alloc(stile, (m)*(n), sTilesInteger); \
        sTiles::aux::mem_set_int(stile->ss_progress, m * n, init_val); \
    } \
    stile->ss_ld = (m); \
    stile->ss_abort = 0; \
    sTiles::Control::Barrier(stile); \
}

/* Sparse byte-progress initializer. For chol paths whose dependency values
 * are restricted to {0, 1}: points stile->ss_slots at the SCHEME-RESIDENT
 * persistent byte buffer (allocated once in sTiles_init_group, freed at
 * scheme destruction) and zeros the first `nslots` bytes for this call.
 * No per-chol malloc/free; pair with ss_finalize_byte(). */
#define ss_init_byte(scheme, nslots, init_val) \
{ \
    if (STILES_RANK == 0) { \
        /* Lazily allocate the per-scheme byte-progress buffer if it was not \
         * provisioned in sTiles_init_group. This covers schemes reached only \
         * via call mapping (global_index_mapped), whose buffers the init-time \
         * loop does not visit — otherwise ss_slots would be NULL here and the \
         * memset below would write to a wild address. Freed in free.cpp. */ \
        if (!(scheme)->byte_progress_buf && (nslots) > 0) { \
            (scheme)->byte_progress_buf = (unsigned char*)std::malloc((size_t)(nslots)); \
        } \
        stile->ss_slots  = (scheme)->byte_progress_buf; \
        if (stile->ss_slots) memset((void*)stile->ss_slots, (init_val), (nslots)); \
        stile->ss_nslots = (nslots); \
    } \
    stile->ss_abort = 0; \
    sTiles::Control::Barrier(stile); \
}

#define ss_finalize_byte() \
{ \
    sTiles::Control::Barrier(stile); \
    if (STILES_RANK == 0) { \
        stile->ss_slots  = NULL; \
        stile->ss_nslots = 0; \
    } \
}

/* =====================================================================
 * Compile-time dispatch: STILES_BYTE_PROGRESS
 *
 *   Defined  -> chol kernels use the sparse byte-progress path
 *               (ss_slots / dep_tracker->slots, O(numActiveTiles))
 *   Undef'd  -> kernels use the legacy dense int progress
 *               (ss_progress / dep_tracker->progress_table, O(num_tiles_per_dim^2))
 *
 * Only safe for kernels whose stored values are <= 255 (the four chol
 * semi kernels stay 0/1). The dispatch macros below take BOTH the legacy
 * (m_coord, n_coord) and the new (slot_idx) so call sites are uniform;
 * the active path picks whichever it needs.
 *
 * Default: ON. Override at build time with -UNSTILES_BYTE_PROGRESS to
 * disable, or comment out the line below to fall back to legacy.
 * =====================================================================*/
#define STILES_BYTE_PROGRESS

#ifdef STILES_BYTE_PROGRESS
    /* --- pthreads-side dispatch (uses stile + scheme's persistent buffer) --- */
    #define ss_init_dp(scheme, num_tiles_per_dim, num_active, init_val) \
        do { (void)(num_tiles_per_dim); \
             ss_init_byte((scheme), (num_active), (init_val)); } while (0)
    #define ss_finalize_dp() \
        ss_finalize_byte()
    #define ss_set_done_dp(m_coord, n_coord, slot, val) \
        do { (void)(m_coord); (void)(n_coord); \
             stile->ss_slots[(slot)] = (unsigned char)(val); } while (0)
    /* SS_WAIT_DP: fast-spin variant (mirrors SS_WAIT_IMP2 semantics).
     * break-friendly (used inside switch{}); aborts on stile->ss_abort. */
    #define SS_WAIT_DP(m_coord, n_coord, slot, val) \
        { \
            (void)(m_coord); (void)(n_coord); \
            int _spin_ct = 0; \
            while (!stile->ss_abort && \
                    stile->ss_slots[(slot)] != (unsigned char)(val)) { \
                hpc_pause_hybrid(_spin_ct); \
            } \
            if (stile->ss_abort) break; \
        }
    /* SS_COND_WAIT_DP: polite/yielding wait (mirrors ss_cond_wait semantics).
     * Used by reduction kernels where waits can be longer; uses STILES_WAIT_LOOP. */
    #define SS_COND_WAIT_DP(m_coord, n_coord, slot, val) \
        { \
            (void)(m_coord); (void)(n_coord); \
            STILES_WAIT_LOOP(stile->ss_slots[(slot)] != (unsigned char)(val)) \
            if (stile->ss_abort) break; \
        }

    /* --- OMP-side dispatch (uses dep_tracker + scheme's persistent buffer) --- */
    #define dep_init_dp(scheme, num_tiles_per_dim, num_active, init_val) \
        do { (void)(num_tiles_per_dim); \
             dep_init_byte((scheme), (num_active), (init_val)); } while (0)
    #define dep_finalize_dp() \
        dep_finalize_byte()
    #define dep_set_done_dp(m_coord, n_coord, slot, val) \
        do { (void)(m_coord); (void)(n_coord); \
             dep_set_done_b((slot), (val)); } while (0)
    #define dep_wait_for_dp(m_coord, n_coord, slot, val) \
        { (void)(m_coord); (void)(n_coord); dep_wait_for_b((slot), (val)); }
#else
    /* --- pthreads-side legacy (uses stile->ss_progress) --- */
    #define ss_init_dp(scheme, num_tiles_per_dim, num_active, init_val) \
        do { (void)(scheme); (void)(num_active); \
             ss_init((num_tiles_per_dim), (num_tiles_per_dim), (init_val)); } while (0)
    #define ss_finalize_dp() \
        ss_finalize()
    #define ss_set_done_dp(m_coord, n_coord, slot, val) \
        do { (void)(slot); ss_cond_set((m_coord), (n_coord), (val)); } while (0)
    #define SS_WAIT_DP(m_coord, n_coord, slot, val) \
        { \
            (void)(slot); \
            int _spin_ct = 0; \
            while (!stile->ss_abort && \
                    stile->ss_progress[(m_coord) + stile->ss_ld * (n_coord)] != (val)) { \
                hpc_pause_hybrid(_spin_ct); \
            } \
            if (stile->ss_abort) break; \
        }
    /* Plain-block (not do-while) so the inner `break` in ss_cond_wait
     * still escapes the enclosing for/switch — same as the original
     * direct ss_cond_wait callsite. */
    #define SS_COND_WAIT_DP(m_coord, n_coord, slot, val) \
        { (void)(slot); ss_cond_wait((m_coord), (n_coord), (val)); }

    /* --- OMP-side legacy (uses dep_tracker->progress_table) --- */
    #define dep_init_dp(scheme, num_tiles_per_dim, num_active, init_val) \
        do { (void)(scheme); (void)(num_active); \
             dep_init((num_tiles_per_dim), (num_tiles_per_dim), (init_val)); } while (0)
    #define dep_finalize_dp() \
        dep_finalize()
    #define dep_set_done_dp(m_coord, n_coord, slot, val) \
        do { (void)(slot); dep_set_done((m_coord), (n_coord), (val)); } while (0)
    #define dep_wait_for_dp(m_coord, n_coord, slot, val) \
        { (void)(slot); dep_wait_for((m_coord), (n_coord), (val)); }
#endif

// #define ss_init(m, n, init_val) \
// { \
//     if (STILES_RANK == 0) { \
//         stile->ss_progress = (volatile int *)stiles_shared_alloc(stile, (m)*(n), sTilesInteger); \
//         sTiles::aux::mem_set_int(stile->ss_progress, m * n, init_val); \
//     } \
//     stile->ss_ld = (m); \
//     stile->ss_abort = 0; \
//     sTiles::Control::Barrier(stile); \
// }

#define ss_print_progress(m, n)                          \
{                                                               \
    if (STILES_RANK == 0) {                                     \
        printf("==== ss_progress (%dx%d) ====\n", (m), (n));    \
        for (int j = 0; j < (n); ++j) {                         \
            for (int i = 0; i < (m); ++i) {                     \
                printf("%3d ", stile->ss_progress[i + j * stile->ss_ld]); \
            }                                                   \
            printf("\n");                                       \
        }                                                       \
        printf("==============================\n");               \
    }                                                           \
    sTiles::Control::Barrier(stile);                                      \
}


#define nd_init(m, n, init_val, k) \
{ \
    if (STILES_RANK == 0) { \
        stile->ss_progress = (volatile int *)stiles_shared_alloc(stile, (m)*(n), sTilesInteger); \
        sTiles::aux::mem_set_int((int*)stile->ss_progress, (m)*(n), (init_val)); \
        for (int i = 0; i <= k; i++) { \
            for (int j = i; j < n; j++) { \
                (stile)->ss_progress[i * n + j] = 1; \
            } \
        } \
    } \
    stile->ss_ld = (m); \
    stile->ss_abort = 0; \
    sTiles::Control::Barrier(stile); \
}

#define ndb_init(m, n, init_val, ONOFF, sizes) \
{ \
    if (STILES_RANK == 0) { \
        stile->ss_progress = (volatile int *)stiles_shared_alloc(stile, (m)*(n), sTilesInteger); \
        sTiles::aux::mem_set_int((int*)stile->ss_progress, (m)*(n), (init_val)); \
        for (int j = 0; j < m; j++) { \
            for (int i = 0; i <= j; i++) { \
                if (j > (sizes[0]-1) && j <= (sizes[0]+sizes[1]-2) && i <= (sizes[0]-1)) { \
                    (stile)->ss_progress[i * n + j] = 1; \
                } \
            } \
        } \
    } \
    stile->ss_ld = (m); \
    stile->ss_abort = 0; \
    sTiles::Control::Barrier(stile); \
}

#define ss_abort()   stile->ss_abort = 1;
#define ss_aborted() stile->ss_abort
#define nd_aborted() stile->ss_abort
#define nd_abort()   stile->ss_abort = 1;

#define ss_cond_set(m, n, val) \
{ \
    stile->ss_progress[(m)+stile->ss_ld*(n)] = (val); \
}

/* ============================================================================
 * Wait-strategy selection (compile-time) — applies to ALL cond_wait macros
 * below: ss_cond_wait, in_cond_wait, ss_cond_wait_tree, and
 * ss_cond_wait_tree_e_s_t_y_l_e. Selectable via STILES_WAIT_MODE in make.inc.
 *
 *   STILES_WAIT_MODE = 0  LEGACY    sched_yield() per spin iteration
 *                                    (preserved verbatim for easy rollback)
 *   STILES_WAIT_MODE = 1  PAUSE     CPU pause hint only, no syscalls
 *                                    (best when waits are short)
 *   STILES_WAIT_MODE = 2  ADAPTIVE  pause for N iters, then sched_yield()
 *                                    (good on >=8 cores)
 *
 * Default is 0 (LEGACY) to match the original committed behavior; flip to 2
 * via make.inc / -DSTILES_WAIT_MODE=2 on machines where ADAPTIVE wins.
 * ========================================================================= */
#ifndef STILES_WAIT_MODE
#  define STILES_WAIT_MODE 0
#endif

/* CPU pause hint — single instruction on x86 (PAUSE), ARM (YIELD); no-op
 * elsewhere. Tells the CPU we're spin-waiting so it can back off the pipeline
 * and free up SMT siblings. ~1 ns vs ~500 ns for sched_yield(). */
#if defined(__x86_64__) || defined(__i386__)
#  define STILES_WAIT_CPU_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#  define STILES_WAIT_CPU_PAUSE() __asm__ __volatile__("yield" ::: "memory")
#else
#  define STILES_WAIT_CPU_PAUSE() ((void)0)
#endif

/* Iterations of CPU pause before falling back to sched_yield() in ADAPTIVE
 * mode. ~1024 pauses ≈ 1 µs on modern hardware — covers typical immediate
 * waits without ever entering the kernel. */
#ifndef STILES_WAIT_ADAPTIVE_BUDGET
#  define STILES_WAIT_ADAPTIVE_BUDGET 1024
#endif

/* STILES_WAIT_LOOP(predicate) — runs the spin loop, polling `predicate` until
 * it becomes false (or ss_abort fires). Behaviour selected at compile time
 * by STILES_WAIT_MODE. Used by all four cond_wait macros below. */
#if STILES_WAIT_MODE == 1
#  define STILES_WAIT_LOOP(predicate) \
        while (!stile->ss_abort && (predicate)) STILES_WAIT_CPU_PAUSE();
#elif STILES_WAIT_MODE == 2
#  define STILES_WAIT_LOOP(predicate) \
    { \
        int __wait_spin = 0; \
        while (!stile->ss_abort && (predicate)) { \
            if (__wait_spin < STILES_WAIT_ADAPTIVE_BUDGET) { \
                STILES_WAIT_CPU_PAUSE(); ++__wait_spin; \
            } else { \
                sTiles::Control::Yield(); __wait_spin = 0; \
            } \
        } \
    }
#else
/* LEGACY (preserved verbatim from the original implementation) */
#  define STILES_WAIT_LOOP(predicate) \
        while (!stile->ss_abort && (predicate)) sTiles::Control::Yield();
#endif

#define ss_cond_wait(m, n, val) \
{ \
    STILES_WAIT_LOOP(stile->ss_progress[(m)+stile->ss_ld*(n)] != (val)) \
    if (stile->ss_abort) break; \
}

// #define ss_cond_set(m, n, val) \
//     reinterpret_cast<std::atomic<int>*>( \
//         const_cast<int*>(&stile->ss_progress[(m)+stile->ss_ld*(n)]) \
//     )->store((val), std::memory_order_release);

// #define ss_cond_wait(m, n, val) \
//     while (!stile->ss_abort && \
//         reinterpret_cast<std::atomic<int>*>( \
//             const_cast<int*>(&stile->ss_progress[(m)+stile->ss_ld*(n)]) \
//         )->load(std::memory_order_acquire) != (val)) \
//         sTiles::Control::Yield(); \
//     if (stile->ss_abort) \
//         break;


#define init_tree() \
{ \
    stile->ss_abort = 0;\
}

#define ss_cond_set_tree(m, val, dep) \
{ \
    dep[m] = (val); \
}

#define ss_cond_wait_tree(m, val, dep) \
{ \
    STILES_WAIT_LOOP(dep[m] < (val)) \
    if (stile->ss_abort) break; \
}


#define ss_finalize() \
{ \
    sTiles::Control::Barrier(stile); \
    if (STILES_RANK == 0) \
        stiles_shared_free(stile, (void*)stile->ss_progress); \
}

#define nd_finalize() \
{ \
    sTiles::Control::Barrier(stile); \
    if (STILES_RANK == 0) \
        stiles_shared_free(stile, (void*)stile->ss_progress); \
}

#define ss_cond_wait_tree_e_s_t_y_l_e(m, val, dep) \
{ \
    STILES_WAIT_LOOP(dep[m] != (val)) \
    if (stile->ss_abort) break; \
}

/*************************************************************************************************************************/

#define ss_init_customized(m, n, init_val, main_rank_partition1, main_rank_partition2) \
{ \
    if (STILES_RANK == main_rank_partition1) { \
        stile->ss_progress1 = (volatile int *)stiles_shared_alloc(stile, (m)*(n), sTilesInteger); \
        sTiles::aux::mem_set_int((int*)stile->ss_progress1, (m)*(n), (init_val)); \
        stile->ss_progress3 = (volatile int *)stiles_shared_alloc(stile, (m)*(n), sTilesInteger); \
        sTiles::aux::mem_set_int((int*)stile->ss_progress3, (m)*(n), (init_val)); \
    } else if (STILES_RANK == main_rank_partition2) { \
        stile->ss_progress2 = (volatile int *)stiles_shared_alloc(stile, (m)*(n), sTilesInteger); \
        sTiles::aux::mem_set_int((int*)stile->ss_progress2, (m)*(n), (init_val)); \
    } \
    stile->ss_ld1 = (m); \
    stile->ss_abort1 = 0; \
    stile->ss_ld3 = (m); \
    stile->ss_abort3 = 0; \
    stile->ss_ld2 = (m); \
    stile->ss_abort2 = 0; \
    sTiles::Control::Barrier(stile); \
}

#define in_init(m, n, init_val) \
{ \
    if (STILES_RANK == 0) { \
        stile->ss_progress = (volatile int *)stiles_shared_alloc(stile, (m)*(n), sTilesInteger); \
        sTiles::aux::mem_set_int((int*)stile->ss_progress, (m)*(n), (init_val)); \
    } \
    stile->ss_ld = (m); \
    stile->ss_abort = 0; \
    sTiles::Control::Barrier(stile); \
}

#define in_abort()   stile->ss_abort = 1;
#define in_aborted() stile->ss_abort

#define in_cond_set(m, n, val) \
{ \
    stile->ss_progress[(m)+stile->ss_ld*(n)] = (val); \
}

/* in_cond_wait — same wait-strategy dispatch as ss_cond_wait above
 * (controlled by STILES_WAIT_MODE). Used by selinv (pdtrtri.cpp). */
#define in_cond_wait(m, n, val) \
{ \
    STILES_WAIT_LOOP(stile->ss_progress[(m)+stile->ss_ld*(n)] != (val)) \
    if (stile->ss_abort) break; \
}

#define in_finalize() \
{ \
    sTiles::Control::Barrier(stile); \
    if (STILES_RANK == 0) \
        stiles_shared_free(stile, (void*)stile->ss_progress); \
}

#define print_dependency() { \
    for (int i = 0; i < A.nt; ++i) { \
        for (int j = 0; j < A.nt; ++j) { \
            printf("%d ", stile->ss_progress[i * A.nt + j]); \
        } \
        printf("\n"); \
    } \
    printf("\n"); \
}

/*************************************************************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

// int stiles_desc_check(TilesDescriptor *desc, int call_index);
// int stiles_desc_mat_alloc(TilesDescriptor *desc);
// int stiles_desc_mat_free(TilesDescriptor *desc);

#include "stiles_pcompute.h"

#ifdef __cplusplus
}
#endif

#endif


