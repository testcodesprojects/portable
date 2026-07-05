

#include <stdlib.h>
#if defined( _WIN32 ) || defined( _WIN64 )
#include "stileswinthread.h"
#else
#include <pthread.h>
#endif

#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "context.h"
#include "new_global.hpp"   // CONTEXT_THREADS_MAX
#include "../memory/stile_threadWorkspace.hpp"
#include "auxiliary.h"

// =========================================================================
// ==              GLOBAL VARIABLE DEFINITIONS                            ==
// =========================================================================
stiles_context_t **g_context_array_ptr = NULL;
pthread_mutex_t *g_context_locks = NULL;
int g_num_contexts = 0;

// =========================================================================
// ==                      INTERNAL HELPER FUNCTION                       ==
// =========================================================================
static void _stiles_context_init_internal(stiles_context_t *stile, int index, long max_cores) {
    memset(stile, 0, sizeof(stiles_context_t));
    
    stile->index = index;
    stile->is_active = 0;
    
    // Initialize all pthread objects
    pthread_mutex_init(&stile->action_mutex, NULL);
    pthread_cond_init(&stile->action_condt, NULL);
    pthread_mutex_init(&stile->barrier_synclock, NULL);
    pthread_cond_init(&stile->barrier_synccond, NULL);
    pthread_attr_init(&stile->thread_attr);
    pthread_attr_setscope(&stile->thread_attr, PTHREAD_SCOPE_SYSTEM);

    // Allocate internal arrays.
    // GetAffinityThreads() unconditionally fills CONTEXT_THREADS_MAX (256) entries,
    // so these buffers must hold at least that many regardless of the machine's core
    // count. Without this, machines with <256 online CPUs (e.g. 128-core Rome) overrun
    // thread_bind during affinity setup -> heap corruption ("free(): invalid next size").
    long arr_len = (max_cores > CONTEXT_THREADS_MAX) ? max_cores : CONTEXT_THREADS_MAX;
    stile->thread_bind  = (int*)malloc(arr_len * sizeof(int));
    stile->thread_rank  = (int*)malloc(arr_len * sizeof(int));
    stile->thread_id    = (pthread_t*)malloc(arr_len * sizeof(pthread_t));
    stile->barrier_in   = (volatile int*)calloc(arr_len, sizeof(int));
    stile->barrier_out  = (volatile int*)calloc(arr_len, sizeof(int));
    
    // Check for allocation failure
    if (!stile->thread_bind || !stile->thread_rank || !stile->thread_id ||
        !stile->barrier_in || !stile->barrier_out) {
        sTiles::Logger::fatal("Memory allocation failed for internal arrays in context index ", index);
        exit(EXIT_FAILURE);
    }
    
    // Set default values
    stile->action = sTiles::Action::StandBy;
    // stile->scheduling = 1;
    // stile->errors_enabled = 1;
    // stile->warnings_enabled = 1;
    // stile->autotuning_enabled = 1;
    // stile->nb = 160;
    // stile->nbnbsize = 25600;
    // stile->rhblock = 4;
    // stile->tntsize = 4;
}

// =========================================================================
// ==                     PUBLIC API IMPLEMENTATIONS                      ==
// =========================================================================
int stiles_context_create_once_all(int num_indices) {
    if (g_context_array_ptr != NULL) return 0; // Already initialized
    if (num_indices <= 0) return -1;

    long max_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (max_cores <= 0) max_cores = 256; // A safe fallback

    g_num_contexts = num_indices;

    g_context_array_ptr = (stiles_context_t **)calloc(g_num_contexts, sizeof(stiles_context_t*));
    if (!g_context_array_ptr) return -1;

    g_context_locks = (pthread_mutex_t *)malloc(g_num_contexts * sizeof(pthread_mutex_t));
    if (!g_context_locks) { free(g_context_array_ptr); return -1; }

    for (int i = 0; i < g_num_contexts; i++) {
        g_context_array_ptr[i] = (stiles_context_t *)malloc(sizeof(stiles_context_t));
        if (!g_context_array_ptr[i]) {
            stiles_context_destroy_all(); // *** FIX: Proper cleanup on failure
            return -1;
        }
        _stiles_context_init_internal(g_context_array_ptr[i], i, max_cores);
        pthread_mutex_init(&g_context_locks[i], NULL);
    }
    return 0;
}

void stiles_context_destroy_all(void) {
    if (g_context_array_ptr == NULL) return;

    for (int i = 0; i < g_num_contexts; i++) {
        stiles_context_t* stile = g_context_array_ptr[i];
        if (stile != NULL) {
            // Free any per-thread workspaces allocated
            if (stile->thread_workspaces) {
                for (int r = 0; r < stile->thread_workspaces_count; ++r) {
                    if (stile->thread_workspaces[r]) {
                        auto *tw = reinterpret_cast<sTiles::ThreadWorkspace*>(stile->thread_workspaces[r]);
                        delete tw;
                        stile->thread_workspaces[r] = nullptr;
                    }
                }
                free(stile->thread_workspaces);
                stile->thread_workspaces = nullptr;
                stile->thread_workspaces_count = 0;
            }
            free(stile->thread_bind);
            free(stile->thread_rank);
            free(stile->thread_id);
            free((void*)stile->barrier_in);
            free((void*)stile->barrier_out);
            pthread_mutex_destroy(&stile->action_mutex);
            pthread_cond_destroy(&stile->action_condt);
            pthread_mutex_destroy(&stile->barrier_synclock);
            pthread_cond_destroy(&stile->barrier_synccond);
            pthread_attr_destroy(&stile->thread_attr);
            free(stile);
        }
        if (g_context_locks != NULL) {
            pthread_mutex_destroy(&g_context_locks[i]);
        }
    }
    free(g_context_locks);
    free(g_context_array_ptr);
    g_context_array_ptr = NULL;
    g_context_locks = NULL;
    g_num_contexts = 0;
}

stiles_context_t* stiles_context_get_by_index(int call_index) {
    if (call_index < 0 || call_index >= g_num_contexts) return NULL;
    return g_context_array_ptr[call_index];
}

// This function remains an alias, useful for consistency.
stiles_context_t* stiles_context_self(int call_index) {
    return stiles_context_get_by_index(call_index);
}

int stiles_rank(stiles_context_t *stile) {
    pthread_t thread_id = pthread_self();

    // The master thread (rank 0) is always the one calling activate/deactivate
    if (pthread_equal(stile->thread_id[0], thread_id)) {
        return 0;
    }

    // For worker threads, search the list
    for (int rank = 1; rank < stile->world_size; rank++) {
        if (pthread_equal(stile->thread_id[rank], thread_id)) {
            return stile->thread_rank[rank]; // Return stored rank
        }
    }
    return -105;
}
