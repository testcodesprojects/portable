 #include <stdio.h>
 #include <stdlib.h>
 #ifdef __linux__
 #include <sched.h>  // Linux-specific
 #endif
 #if defined(STILES_WITH_MKL)
 #include <omp.h>
 #endif
 #if defined( _WIN32 ) || defined( _WIN64 )
 #include "stileswinthread.h"
 #else
 #include <pthread.h>
 #endif
 #include <errno.h>
 #include <string.h>
 #include <unistd.h> // Needed for getpid()
 
 #include "common.h"
 #include "auxiliary.h"
 #include "allocate.h"
 
 #include <stdio.h>
 #include <stdlib.h>
 #include <unistd.h>
 #include <sys/types.h>
 #include <math.h>

 #ifdef __linux__
 #include <fcntl.h>
 #include <dirent.h>
 #elif defined(__APPLE__)
 #include <sys/types.h>
 #include <sys/sysctl.h>
 #elif defined(_WIN32)
 #include <windows.h>
 #endif
 
 #ifdef _WIN32
 #include <windows.h>
 #elif defined(__APPLE__)
 #include <sys/types.h>
 #include <sys/sysctl.h>
 #else
 #include <dirent.h>
 #endif
 

static double get_elapsed_ms(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_nsec - start->tv_nsec) / 1000000.0;
}

// Thread-local storage for the currently bound call index (for rescale support)
static thread_local int tls_bound_call_index = -1;

 long long get_l3_cache_size() {
     long long total_cache = -1;
    #ifdef __linux__
        // Avoid overcounting by checking unique shared_cpu_list entries
        char seen[128][128];  // store shared_cpu_list strings
        int count = 0;
        total_cache = 0;
    
        for (int cpu = 0; cpu < 256; cpu++) {
            char size_path[256];
            snprintf(size_path, sizeof(size_path),
                    "/sys/devices/system/cpu/cpu%d/cache/index3/size", cpu);
            FILE *size_file = fopen(size_path, "r");
            if (!size_file) break;
    
            long long size = 0;
            char unit;
            if (fscanf(size_file, "%lld%c", &size, &unit) == 2) {
                if (unit == 'K') size *= 1024;
                else if (unit == 'M') size *= 1024 * 1024;
                else if (unit == 'G') size *= 1024 * 1024 * 1024;
            }
            fclose(size_file);
    
            char share_path[256];
            snprintf(share_path, sizeof(share_path),
                    "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", cpu);
            FILE *share_file = fopen(share_path, "r");
            if (!share_file) continue;
    
            char shared_list[128];
            if (fgets(shared_list, sizeof(shared_list), share_file)) {
                int seen_before = 0;
                for (int i = 0; i < count; i++) {
                    if (strcmp(shared_list, seen[i]) == 0) {
                        seen_before = 1;
                        break;
                    }
                }
                if (!seen_before) {
                    strcpy(seen[count++], shared_list);
                    total_cache += size;
                }
            }
            fclose(share_file);
        }
    
    #elif defined(__APPLE__)
        // macOS: Use sysctl to get L3 cache size
        size_t len = sizeof(total_cache);
        if (sysctlbyname("hw.l3cachesize", &total_cache, &len, NULL, 0) != 0) {
            perror("sysctl failed");
            total_cache = -1;
        }
    
    #elif defined(_WIN32)
        // Windows: Use GetLogicalProcessorInformationEx
        DWORD length = 0;
        GetLogicalProcessorInformationEx(RelationCache, NULL, &length);
    
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *buffer =
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)malloc(length);
        if (!buffer || !GetLogicalProcessorInformationEx(RelationCache, buffer, &length)) {
            total_cache = -1;
        } else {
            total_cache = 0;
            char *ptr = (char *)buffer;
            char *end = ptr + length;
    
            while (ptr < end) {
                SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info =
                    (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)ptr;
    
                if (info->Relationship == RelationCache &&
                    info->Cache.Level == 3) {
                    total_cache += info->Cache.CacheSize;
                }
    
                ptr += info->Size;
            }
        }
        free(buffer);
    
    #else
        // Fallback: assume 40MB
        total_cache = 40 * 1024 * 1024;
    #endif
 
     return total_cache;
 }
 
#ifdef STILES_HWLOC
size_t get_L3_cache(void)
{
    hwloc_topology_t topology;
    size_t l3 = 0;

    // Initialize and load the topology
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    int depth = hwloc_topology_get_depth(topology);
    for (int i = 0; i < depth; i++) {
        int num_objs = hwloc_get_nbobjs_by_depth(topology, i);
        for (int j = 0; j < num_objs; j++) {
            hwloc_obj_t obj = hwloc_get_obj_by_depth(topology, i, j);
            if (obj->type == HWLOC_OBJ_L3CACHE &&
                (obj->attr->cache.type == HWLOC_OBJ_CACHE_UNIFIED || obj->attr->cache.type == HWLOC_OBJ_CACHE_DATA)) {
                l3 = (size_t)(obj->attr->cache.size / (1024 * 1024));
                return l3;
            }
        }
    }

    hwloc_topology_destroy(topology);
    return l3;
}
#else
size_t get_L3_cache(void)
{
    // Fallback: use get_l3_cache_size() which uses sysctl/sysfs
    long long cache = get_l3_cache_size();
    return (cache > 0) ? (size_t)(cache / (1024 * 1024)) : 40;
}
#endif

#ifdef STILES_HWLOC
size_t numa_get_L3_cache(int nnode)
{
    hwloc_topology_t topology;
    hwloc_obj_t numa_node;
    hwloc_obj_t obj = NULL;
    size_t l3 = 0;

    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    if ((numa_node = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, nnode))) {
        while (l3 == 0 && (obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_L3CACHE, obj)) != NULL) {
            if (obj->type == HWLOC_OBJ_L3CACHE &&
                (obj->attr->cache.type == HWLOC_OBJ_CACHE_UNIFIED || obj->attr->cache.type == HWLOC_OBJ_CACHE_DATA)) {
                // Check if the cache is in the subtree of NUMA node 0
                if (hwloc_obj_is_in_subtree(topology, obj, numa_node)) {
                    l3 = (size_t) (obj->attr->cache.size / (1024 * 1024));
                }
            }
        }
    }
    hwloc_topology_destroy(topology);
    return l3;
}
#else
size_t numa_get_L3_cache(int nnode)
{
    (void)nnode;
    long long cache = get_l3_cache_size();
    return (cache > 0) ? (size_t)(cache / (1024 * 1024)) : 40;
}
#endif

namespace sTiles {

    namespace get {

        int auto_tile_size()
        {

            sTiles::Logger::info("│ ↳ Determining auto tile size based on L3 cache topology...");

            int auto_tile_size = 40;
#ifdef STILES_HWLOC
            hwloc_topology_t topology;
            hwloc_obj_t numa_node;
            hwloc_obj_t obj = NULL;
            size_t l3_total = 0;

            hwloc_topology_init(&topology);
            hwloc_topology_load(topology);

            int num_numa_nodes = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);

            // INFO block
            const int box_width = 60;
            char line[128];
            sTiles::Logger::debug("  - NUMA nodes detected: ", num_numa_nodes);


            for(int j=0; j<num_numa_nodes; j++){

                if ((numa_node = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, j))) {
                    while ((obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_L3CACHE, obj)) != NULL) {
                        if ((obj->attr->cache.type == HWLOC_OBJ_CACHE_UNIFIED || obj->attr->cache.type == HWLOC_OBJ_CACHE_DATA) &&
                            hwloc_obj_is_in_subtree(topology, obj, numa_node)) {
                            l3_total += obj->attr->cache.size;
                            size_t mb = obj->attr->cache.size / (1024 * 1024);
                            sTiles::Logger::trace("    - Found L3 cache of size: ", mb, " MB");
                        }
                    }
                }
            }

            hwloc_topology_destroy(topology);
            size_t cache_size = l3_total / (1024 * 1024);  // return MB
#else
            // Fallback without hwloc: use sysctl/sysfs-based detection
            long long l3_bytes = get_l3_cache_size();
            size_t cache_size = (l3_bytes > 0) ? (size_t)(l3_bytes / (1024 * 1024)) : 40;
#endif
            sTiles::Logger::debug("  - Total L3 cache size across all nodes: ", cache_size, " MB");

            if (cache_size < 50)
                auto_tile_size = 40;
            else if (cache_size < 60)
                auto_tile_size = 80;
            else if (cache_size >= 60)
                auto_tile_size = 120;
            else
                auto_tile_size = 40;

            sTiles::Logger::info("│ ✓ Auto tile size selected: ", auto_tile_size);
            return auto_tile_size;
        }

    }
}
int set_scheduler() {
    struct sched_param param;
    param.sched_priority = 0;  // Only real-time schedulers require priority

    #ifdef __linux__
        if (sched_setscheduler(0, SCHED_OTHER, &param) == -1) {
            sTiles::Logger::warning("sched_setscheduler failed: ", strerror(errno));
            return -1;
        }
    #endif
    return 0;
}

sTiles::Control::CoreInfo sTiles::Control::GetCoreInfo() {

    CoreInfo info = {0, 0, 0, 0};

#ifdef STILES_HWLOC
    hwloc_topology_t topology;

    // Initialize topology object
    hwloc_topology_init(&topology);
    // Perform the topology detection
    hwloc_topology_load(topology);

    // Get the number of sockets
    info.numSockets = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_SOCKET);
    if (info.numSockets == 0) {
        // Assume there is at least one socket
        info.numSockets = 1;
    }

    // Get total number of cores
    info.totalCores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
    if (info.totalCores == 0) {
        // Fall back to counting PUs (processing units) if no cores are detected
        info.totalCores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
    }

    // Calculate cores per socket if possible
    if (info.numSockets > 0) {
        info.coresPerSocket = info.totalCores / info.numSockets;
    }

    int num_numa_nodes = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
    // Calculate NUMA nodes per socket
    if (info.numSockets > 0) {
        info.numaNodesPerSocket = num_numa_nodes / info.numSockets;
        info.numSockets = num_numa_nodes;
    }

    // Destroy topology object
    hwloc_topology_destroy(topology);

#else
    // Fallback without hwloc: detect cores via sysconf/sysctl
    info.numSockets = 1;
#ifdef _SC_NPROCESSORS_ONLN
    info.totalCores = sysconf(_SC_NPROCESSORS_ONLN);
#else
    info.totalCores = 1;
#endif
    info.coresPerSocket = info.totalCores;
    info.numaNodesPerSocket = 1;
#endif

    return info;
}

/***************************************************************************//**
 *  Busy-waiting barrier initialization
 **/
void sTiles::Control::BarrierInit(stiles_context_t *stile)
{
    stile->barrier_id = 0;
    stile->barrier_nblocked_thrds = 0;
    pthread_mutex_init(&(stile->barrier_synclock), NULL);
    pthread_cond_init( &(stile->barrier_synccond), NULL);
}

/***************************************************************************//**
 *  Busy-waiting barrier finalize
 **/
void sTiles::Control::BarrierFinalize(stiles_context_t *stile)
{
    pthread_mutex_destroy(&(stile->barrier_synclock));
    pthread_cond_destroy( &(stile->barrier_synccond));
}


/***************************************************************************//**
 *  Busy-waiting barrier initialization
 **/
void sTiles::Control::BarrierBusyWaitInit(stiles_context_t *stile)
{
    int core;
    int max_cores_sys = stile->max_cores_sys;
    for (core = 0; core < max_cores_sys; core++) {
        stile->barrier_in[core] = 0;
        stile->barrier_out[core] = 0;
    }
}

/***************************************************************************//**
 *  Busy-waiting barrier finalize
 **/
void sTiles::Control::BarrierBusyWaitFinalize(stiles_context_t *stile)
{
}

void sTiles::Control::Barrier(stiles_context_t *stile) {
    pthread_mutex_lock(&(stile->barrier_synclock));
    int id = stile->barrier_id;
    stile->barrier_nblocked_thrds = stile->barrier_nblocked_thrds + 1;
    if (stile->barrier_nblocked_thrds == stile->world_size) {
        stile->barrier_nblocked_thrds = 0;
        stile->barrier_id = stile->barrier_id + 1;
        pthread_cond_broadcast(&(stile->barrier_synccond));
    }
    while (id == stile->barrier_id)
        pthread_cond_wait(&(stile->barrier_synccond), &(stile->barrier_synclock));
    pthread_mutex_unlock(&(stile->barrier_synclock));
}

// =========================================================================
// ==           BUSY-WAIT BARRIER (WITH CRITICAL FIX)                     ==
// =========================================================================

void sTiles::Control::BarrierBusyWait(stiles_context_t *stile) {
    int rank = stiles_rank(stile);
    int size = stile->world_size;

    if (rank == 0) { // Master thread
        for (int core = 1; core < size; core++)
            while (stile->barrier_in[core] == 0);

        for (int core = 1; core < size; core++)
            stile->barrier_in[core] = 0;

        for (int core = 1; core < size; core++)
            stile->barrier_out[core] = 1;
    }
    else { // Worker threads
        stile->barrier_in[rank] = 1;
        while (stile->barrier_out[rank] == 0);
        stile->barrier_out[rank] = 0;
    }
}


/***************************************************************************//**
 *  Main thread control
 **/
void *sTiles::Control::ParallelSection(void *stiles_ptr)
{
    stiles_context_t *stile = (stiles_context_t*)(stiles_ptr);
    sTiles::Action action;

    /* Set thread affinity for the worker */
    SetAffinity(stile->thread_bind[stiles_rank(stile)]);

    Barrier(stile);
    while(1) {
        pthread_mutex_lock(&stile->action_mutex);
        while ((action = stile->action) == sTiles::Action::StandBy)
            pthread_cond_wait(&stile->action_condt, &stile->action_mutex);
        pthread_mutex_unlock(&stile->action_mutex);
        Barrier(stile);

        switch (action) {
            case sTiles::Action::Parallel:
                stile->parallel_func_ptr(stile);
                break;
            case sTiles::Action::Dynamic:
                break;
            case sTiles::Action::Finalize:
                return NULL;
            default:
                sTiles::Logger::fatal("Undefined action encountered in parallel section.");
                return NULL;
        }
        Barrier(stile);
    }

    UnsetAffinity();
    return NULL;
}


//-------------------------------------------------------------------------------------------------------------//


/***************************************************************************//**
 *                            NEW PUBLIC API
 ***************************************************************************/

int sTiles::Control::InitializeGlobal(int num_contexts) {

    if (stiles_context_create_once_all(num_contexts) != 0) {
        sTiles::Logger::fatal("Failed to create global context array.");
        return -106;
    }
    sTiles::Control::TopologyInit();
    set_scheduler();
    return 0;
}


int sTiles::Control::ActivateTeam(int cores, int *coresbind, int call_index) {

    //struct timespec total_start, total_end;
    //clock_gettime(CLOCK_MONOTONIC, &total_start);

    stiles_context_t *stile = stiles_context_get_by_index(call_index);
    if (stile == NULL) return -104;

    pthread_mutex_lock(&g_context_locks[call_index]);
    if (stile->is_active) {
        pthread_mutex_unlock(&g_context_locks[call_index]);
        return -110;
    }

    // --- Configuration ---
    stile->world_size = cores;
    stile->workspace_offset = 0;
    if (coresbind != NULL) {
        memcpy(stile->thread_bind, coresbind, cores * sizeof(int));
    } else {
        sTiles::Control::GetAffinityThreads(stile->thread_bind, call_index);
    }
    
    // --- Initialization for this activation ---
    sTiles::Control::BarrierInit(stile);
    sTiles::Control::BarrierBusyWaitInit(stile);
    pthread_attr_init(&stile->thread_attr);
    pthread_attr_setscope(&stile->thread_attr, PTHREAD_SCOPE_SYSTEM);
    
    stile->thread_id[0] = pthread_self();
    stile->thread_rank[0] = 0; // Explicitly set master's rank
    for (int core = 1; core < stile->world_size; core++) {
        stile->thread_rank[core] = core;
    }
    
    // --- Launch worker threads ---
    for (int core = 1; core < stile->world_size; core++) {
        pthread_create(&stile->thread_id[core], &stile->thread_attr,
                       sTiles::Control::ParallelSection, (void*)stile);
    }
    
    // --- Synchronize with workers ---
    sTiles::Control::Barrier(stile);
    
    stile->is_active = 1;
    pthread_mutex_unlock(&g_context_locks[call_index]);
    
    sTiles::Control::SetAffinity(stile->thread_bind[0]);
    // ================================================================
    //clock_gettime(CLOCK_MONOTONIC, &total_end);
    //printf("[MEASURE] TOTAL ActivateTeam affinity setup took: %.3f ms\n\n", get_elapsed_ms(&total_start, &total_end));
    return 0;
}

int sTiles::Control::DeactivateTeam(int call_index) {

    stiles_context_t *stile = stiles_context_get_by_index(call_index);
    if (stile == NULL) return -104;

    pthread_mutex_lock(&g_context_locks[call_index]);
    if (!stile->is_active) {
        pthread_mutex_unlock(&g_context_locks[call_index]);
        return -110;
    }

    pthread_mutex_lock(&stile->action_mutex);
    stile->action = sTiles::Action::Finalize;
    pthread_mutex_unlock(&stile->action_mutex);
    pthread_cond_broadcast(&stile->action_condt);

    sTiles::Control::Barrier(stile);

    for (int core = 1; core < stile->world_size; core++) {
        pthread_join(stile->thread_id[core], NULL);
    }

    // Clean up resources used for this activation.
    sTiles::Control::BarrierFinalize(stile);
    pthread_attr_destroy(&stile->thread_attr);
    
    // Reset the context state for the next activation.
    stile->is_active = 0;
    stile->action = sTiles::Action::StandBy;
    
    pthread_mutex_unlock(&g_context_locks[call_index]);
    return 0;
}

// Forward declaration
int sTiles_Destroy_Persistent_Team(int call_index);

void sTiles::Control::FinalizeGlobal() {
    // First, properly destroy all persistent teams (join threads)
    if (g_context_array_ptr != NULL) {
        for (int i = 0; i < g_num_contexts; i++) {
            sTiles_Destroy_Persistent_Team(i);
        }
    }
    // Then destroy the contexts
    stiles_context_destroy_all();
    sTiles::Control::TopologyFinalize();
}

// ---------------------------------------------------------------------------------
// Persistent Team (namespaced C++ API mirroring legacy C entry points)
// ---------------------------------------------------------------------------------

int sTiles::Control::CreatePersistentTeam(int cores, const int *coresbind, int call_index) {
    
    stiles_context_t *stile = stiles_context_get_by_index(call_index);
    if (stile == NULL) return -104;

    pthread_mutex_lock(&g_context_locks[call_index]);

    if (stile->threads_created) {
        pthread_mutex_unlock(&g_context_locks[call_index]);
        return 0;
    }

    stile->world_size = cores;
    stile->workspace_offset = 0;
    if (coresbind != NULL) {
        memcpy(stile->thread_bind, coresbind, cores * sizeof(int));
    } else {
        sTiles::Control::GetAffinityThreads(stile->thread_bind, call_index);
    }

    sTiles::Control::BarrierInit(stile);
    sTiles::Control::BarrierBusyWaitInit(stile);
    pthread_attr_init(&stile->thread_attr);
    pthread_attr_setscope(&stile->thread_attr, PTHREAD_SCOPE_SYSTEM);

    stile->thread_id[0] = pthread_self();
    stile->thread_rank[0] = 0;
    for (int core = 1; core < stile->world_size; core++) {
        stile->thread_rank[core] = core;
    }

    for (int core = 1; core < stile->world_size; core++) {
        int err = pthread_create(&stile->thread_id[core], &stile->thread_attr,
                                 sTiles::Control::ParallelSection, (void*)stile);
        if (err != 0) {
            sTiles::Logger::fatal("pthread_create failed with error code: ", err);
            pthread_mutex_unlock(&g_context_locks[call_index]);
            return -1;
        }
    }

    sTiles::Control::Barrier(stile);
    stile->threads_created = 1;
    pthread_mutex_unlock(&g_context_locks[call_index]);

    return 0;
}

int sTiles::Control::ActivatePersistentTeam(int call_index) {
    stiles_context_t *stile = stiles_context_get_by_index(call_index);
    if (stile == NULL) return -104;

    pthread_mutex_lock(&g_context_locks[call_index]);
    if (stile->is_active) {
        pthread_mutex_unlock(&g_context_locks[call_index]);
        return -110;
    }
    stile->is_active = 1;
    pthread_mutex_unlock(&g_context_locks[call_index]);
    pthread_mutex_lock(&stile->action_mutex);
    stile->thread_id[0] = pthread_self();
    pthread_mutex_unlock(&stile->action_mutex);
    sTiles::Control::SetAffinity(stile->thread_bind[0]);

    // Store the bound call index in TLS for rescale support
    tls_bound_call_index = call_index;

    return 0;
}

int sTiles::Control::DeactivatePersistentTeam(int call_index) {
    stiles_context_t *stile = stiles_context_get_by_index(call_index);
    if (stile == NULL) return -104;

    pthread_mutex_lock(&g_context_locks[call_index]);
    if (!stile->is_active) {
        pthread_mutex_unlock(&g_context_locks[call_index]);
        return -110;
    }

    pthread_mutex_lock(&stile->action_mutex);
    stile->action = sTiles::Action::StandBy;
    stile->parallel_func_ptr = NULL;
    pthread_mutex_unlock(&stile->action_mutex);
    pthread_cond_broadcast(&stile->action_condt);

    stile->is_active = 0;
    pthread_mutex_unlock(&g_context_locks[call_index]);

    // Clear the TLS
    tls_bound_call_index = -1;

    return 0;
}



extern "C" {
void stiles_barrier_init(stiles_context_t *stile) { sTiles::Control::BarrierInit(stile); }
void stiles_barrier_finalize(stiles_context_t *stile) { sTiles::Control::BarrierFinalize(stile); }
void stiles_barrier(stiles_context_t *stile) { sTiles::Control::Barrier(stile); }
void stiles_barrier_bw_init(stiles_context_t *stile) { sTiles::Control::BarrierBusyWaitInit(stile); }
void stiles_barrier_bw_finalize(stiles_context_t *stile) { sTiles::Control::BarrierBusyWaitFinalize(stile); }
void stiles_barrier_bw(stiles_context_t *stile) { sTiles::Control::BarrierBusyWait(stile); }
void *stiles_parallel_section(void *stiles_ptr) { return sTiles::Control::ParallelSection(stiles_ptr); }
void stiles_topology_init() { sTiles::Control::TopologyInit(); }
void stiles_topology_finalize() { sTiles::Control::TopologyFinalize(); }
int stiles_setaffinity(int rank) { return sTiles::Control::SetAffinity(rank); }
int stiles_unsetaffinity() { return sTiles::Control::UnsetAffinity(); }
int stiles_yield() { return sTiles::Control::Yield(); }
int stiles_get_numthreads() { return sTiles::Control::GetNumThreads(); }
int stiles_get_numthreads_numa() { return sTiles::Control::GetNumThreadsNuma(); }
int stiles_get_affthreads(int *coresbind, int call_index) { return sTiles::Control::GetAffinityThreads(coresbind, call_index); }
int stiles_getnuma_size() { return sTiles::Control::GetNumaSize(); }
}
int sTiles_Create_Persistent_Team(int cores, int *coresbind, int call_index) {
    // Legacy C API forwards to namespaced implementation
    return sTiles::Control::CreatePersistentTeam(cores, coresbind, call_index);
}

/**
 * @brief Internal wrapper dispatched to all threads; unpacks the user callback
 *        and user_data from the args buffer, then invokes the user function.
 */
static void stiles_user_dispatch_wrapper(stiles_context_t* ctx) {
    sTiles_user_func_t func;
    void* user_data;
    sTiles::unpack_args(ctx, func, user_data);
    func(stiles_rank(ctx), ctx->world_size, user_data);
}

int sTiles::Control::DispatchUserFunction(int call_index,
                                          sTiles_user_func_t user_func,
                                          void* user_data) {
    stiles_context_t* ctx = stiles_context_self(call_index);
    if (!ctx) {
        sTiles::Logger::error("DispatchUserFunction: context not initialized for index ", call_index);
        return -1;
    }
    if (!ctx->is_active) {
        sTiles::Logger::error("DispatchUserFunction: context is not active. Call sTiles_bind first.");
        return -1;
    }

    sTiles::parallel_call(ctx, stiles_user_dispatch_wrapper, user_func, user_data);
    return 0;
}

int sTiles_Activate_Team(int call_index) {
    return sTiles::Control::ActivatePersistentTeam(call_index);
}

int sTiles_Deactivate_Team(int call_index) {
    return sTiles::Control::DeactivatePersistentTeam(call_index);
}

int sTiles_Destroy_Persistent_Team(int call_index) {

    stiles_context_t *stile = stiles_context_get_by_index(call_index);
    if (stile == NULL) return -104;
    pthread_mutex_lock(&g_context_locks[call_index]);
    if (!stile->threads_created) {
        pthread_mutex_unlock(&g_context_locks[call_index]);
        return 0;
    }
    pthread_mutex_lock(&stile->action_mutex);
    stile->action = sTiles::Action::Finalize;
    pthread_mutex_unlock(&stile->action_mutex);
    pthread_cond_broadcast(&stile->action_condt);
    sTiles::Control::Barrier(stile);
    for (int core = 1; core < stile->world_size; core++) {
        pthread_join(stile->thread_id[core], NULL);
    }
    sTiles::Control::BarrierFinalize(stile);
    stile->threads_created = 0;
    stile->action = sTiles::Action::StandBy;
    
    pthread_mutex_unlock(&g_context_locks[call_index]);
    
    return 0;
}
