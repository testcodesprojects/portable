/**
 * @file stiles_process.h
 *
 * @version 1.0.0
 * @Developed Esmail Abdul Fattah
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @date 2025-01-30
 */

#ifndef STILES_PROCESS_H
#define STILES_PROCESS_H

#include "../common/stiles_structs.hpp"
#include "stiles.h"
#include "../common/stiles_types.hpp"
#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NUMA_ENABLED 
    #include <numa.h>
#endif

long long get_l3_cache_size();
size_t get_L3_cache(void);
size_t numa_get_L3_cache(int nnode);

int sTiles_Activate_Team(int call_index);
int sTiles_Deactivate_Team(int call_index);
int sTiles_Create_Persistent_Team(int cores, int *coresbind, int call_index);
#ifdef __cplusplus
}
#endif

namespace sTiles {
namespace get {

    /**
     * @brief Determines an optimal tile size based on the machine's total L3 cache size.
     * @return An integer representing the automatically determined tile size.
     */
    int auto_tile_size();

} // namespace get

namespace Control {
    int InitializeGlobal(int num_contexts);
    void FinalizeGlobal();
    int ActivateTeam(int cores, int *coresbind, int call_index);
    int DeactivateTeam(int call_index);
    // Persistent team API (C++ namespace)
    int CreatePersistentTeam(int cores, const int *coresbind, int call_index);
    int ActivatePersistentTeam(int call_index);
    int DeactivatePersistentTeam(int call_index);
    // User parallel dispatch
    int DispatchUserFunction(int call_index, sTiles_user_func_t user_func, void* user_data);
}

namespace Binding {
    // C++ namespaced convenience wrapper for team setup
    int Setup_all_teams(void** obj);
}
} // namespace sTiles


#endif // STILES_PROCESS_H
