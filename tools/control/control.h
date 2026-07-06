/**
 * @file control.h
 *
 * Header file for core control and management routines in the sTiles framework.
 * These routines provide functionality for thread affinity, topology initialization,
 * and barriers required for high-performance tiled matrix computations.
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
 * - Initialization and finalization of the sTiles framework.
 * - Barrier synchronization for multi-threaded environments.
 * - Thread affinity management to optimize core utilization.
 * - Topology discovery and thread scheduling.
 *
 * @version 1.0.0
 * @author Esmail Abdul Fattah
 * @original_author Jakub Kurzak
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

#ifndef STILES_CONTROL_HPP
#define STILES_CONTROL_HPP

#ifdef STILES_HWLOC
extern "C" {
#include "hwloc.h"
}
#endif

#ifndef __cplusplus
extern int pthread_getconcurrency(void);
extern int pthread_setconcurrency(int);
#endif

#include "../common/stiles_types.hpp"

namespace sTiles::Control {

struct CoreInfo {
    int numSockets;
    int totalCores;
    int coresPerSocket;
    int numaNodesPerSocket;
};

void BarrierInit(stiles_context_t *stiles);
void BarrierFinalize(stiles_context_t *stiles);
void Barrier(stiles_context_t *stiles);
void BarrierBusyWaitInit(stiles_context_t *stiles);
void BarrierBusyWaitFinalize(stiles_context_t *stiles);
void BarrierBusyWait(stiles_context_t *stiles);
void *ParallelSection(void *stiles);

void TopologyInit();
void TopologyFinalize();
int SetAffinity(int rank);
int UnsetAffinity();
int YieldCPU();   /* was Yield(): <windows.h> #defines Yield as a macro */
int GetNumThreads();
int GetNumThreadsNuma();
int GetAffinityThreads(int *coresbind, int callIndex);

CoreInfo GetCoreInfo();

int InitializeGlobal(int numContexts);
void FinalizeGlobal();
int ActivateTeam(int cores, int *coresbind, int callIndex);
int DeactivateTeam(int callIndex);
int GetNumaSize();

// Persistent team management (namespaced C++ API)
int CreatePersistentTeam(int cores, const int *coresbind, int callIndex);
int ActivatePersistentTeam(int callIndex);
int DeactivatePersistentTeam(int callIndex);

} // namespace sTiles::Control

#endif
