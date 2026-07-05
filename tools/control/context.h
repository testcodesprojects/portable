
/**
 * @file context.h
 * @brief Header for the sTiles context management system.
 *
 * @version 5.0.0 (Dynamically Sized Contexts)
 */
#ifndef STILES_CONTEXT_H
#define STILES_CONTEXT_H

#include <pthread.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
struct stiles_context_struct;

/***************************************************************************//**
 *  sTiles context structure (Dynamically Sized)
 ***************************************************************************/
typedef struct stiles_context_struct {
    /* Thread control */
    int world_size, group_size;
    int index;
    int is_active;
    int threads_created;
    int max_cores_sys;
    int workspace_offset;

    // === KEY CHANGE: Static arrays are now pointers ===
    int *thread_bind;
    int *thread_rank;
    pthread_t *thread_id;
    // =================================================

    pthread_attr_t thread_attr;

    /* Master-worker communication */
    pthread_mutex_t action_mutex;
    pthread_cond_t action_condt;
    sTiles::Action action; 
    void (*parallel_func_ptr)(struct stiles_context_struct*);
    unsigned char args_buff[1024];

    // /* Boolean flags */
    // int errors_enabled;
    // int warnings_enabled;
    // int autotuning_enabled;
    // int dynamic_section;

    /* Enum flags */
    // int scheduling;     // static or dynamic scheduling

    // /* Matrix tile attributes */
    // int nb;
    // int nbnbsize;   // tile size in elements (possibly padded)
    // int rhblock;    // block size for tree-based (reduction) Householder
    // int tntsize;    // Tournament pivoting grouping size

    /* Barrier implementation */
    /* Busy waiting version */
    volatile int *barrier_in;
    volatile int *barrier_out;

    /* Conditional version */
    int volatile    barrier_id;             /*+ ID of the barrier                     +*/
    int volatile    barrier_nblocked_thrds; /*+ Number of threads locked in the barrier +*/
    pthread_mutex_t barrier_synclock;       /*+ Mutex for the barrier                 +*/
    pthread_cond_t  barrier_synccond;       /*+ Condition for the barrier             +*/

    /* Static scheduler implementation */
    int ss_ld;                  // static scheduler progress table leading dimension
    volatile int ss_abort;      // static scheduler abort flag
    volatile int *ss_progress;  // static scheduler progress table

    /* Sparse byte-progress (semi-mode chol only; values are 0/1).
     * Allocated for numActiveTiles slots; indexed via task indexN,
     * not (i,j). Used by the four semi chol kernels when they switch
     * to the byte-progress macros (ss_init_byte / dep_*_b). */
    volatile unsigned char *ss_slots;
    int ss_nslots;

    // --- Per-thread SmartTile workspaces (opaque pointers) ---
    // Allocated on demand by compute paths that need per-core scratch.
    // Points to sTiles::ThreadWorkspace instances cast to void* to avoid
    // C++ type dependencies in this C-compatible header.
    void **thread_workspaces;
    int     thread_workspaces_count;

} stiles_context_t;

/***************************************************************************//**
 *  Global variables (declarations for other files)
 ***************************************************************************/
extern stiles_context_t **g_context_array_ptr;
extern pthread_mutex_t *g_context_locks;
extern int g_num_contexts;

typedef struct stiles_context_map_struct {
    pthread_t thread_id;        // thread id
    stiles_context_t *context;  // pointer to associated context
} stiles_context_map_t;


/***************************************************************************//**
 *  Function Prototypes (the public API from context.c)
 ***************************************************************************/
int stiles_context_create_once_all(int num_indices);
stiles_context_t* stiles_context_get_by_index(int call_index);
void stiles_context_destroy_all(void);
stiles_context_t *stiles_context_self(int call_index);
int stiles_rank(stiles_context_t *stile);


#ifdef __cplusplus
}
#endif

#endif // STILES_CONTEXT_H
