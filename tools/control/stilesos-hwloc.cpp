/**
 * @file stilesos_hwloc.c
 *
 * Source file for sTiles hardware locality (hwloc) integration.
 *
 * sTiles is an advanced extension of the PLASMA software package, originally developed by:
 * - University of Tennessee
 * - University of California, Berkeley
 * - University of Colorado, Denver
 *
 * The sTiles framework has been redesigned and improved by Esmail Abdul Fattah
 * at King Abdullah University of Science and Technology (KAUST) and the sTiles team.
 *
 * This file provides an interface for hardware locality (hwloc) to enable optimized thread
 * and NUMA node management. These utilities support topology detection, thread affinity,
 * and NUMA size computation to enhance performance on modern multicore architectures.
 *
 * @version 1.0.0
 * @redesigned_by Esmail Abdul Fattah
 * @original_authors Piotr Luszczek, Mathieu Faverge
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @date 2025-01-30
 */


#include <stdlib.h>
#include <atomic>

#ifdef STILES_HWLOC
#include <hwloc.h>
#endif
static hwloc_topology_t g_hwlocTopology = NULL; /* Topology object */
static int g_hwlocGroupSize = -1; /* Size of NUMA nodes */
static volatile int     g_hwlocInstanceCount = 0;

void sTiles::Control::TopologyInit(){

    pthread_mutex_lock(&mutextopo);
    if (!topo_initialized) {

        /* Allocate and initialize topology object.  */
        hwloc_topology_init(&g_hwlocTopology);

        /* Perform the topology detection.  */
        hwloc_topology_load(g_hwlocTopology);

        /* Get the number of cores (We don't want to use HyperThreading */
        sys_corenbr = hwloc_get_nbobjs_by_type(g_hwlocTopology, HWLOC_OBJ_CORE);

        topo_initialized = 1;
    }
    g_hwlocInstanceCount = g_hwlocInstanceCount + 1;
    pthread_mutex_unlock(&mutextopo);
}

void sTiles::Control::TopologyFinalize(){

    sTiles::Control::UnsetAffinity();

    pthread_mutex_lock(&mutextopo);
    g_hwlocInstanceCount = g_hwlocInstanceCount - 1;
    if ((topo_initialized ==1) && (g_hwlocInstanceCount == 0)) {
        /* Destroy tpology */
        hwloc_topology_destroy(g_hwlocTopology);

        topo_initialized = 0;
    }
    pthread_mutex_unlock(&mutextopo);
}

/**
 * Bind the calling thread to the logical core identified by `rank`.
 * Ranks start at 0. When multiple sTiles instances exist the mapping may overlap.
 */

#ifdef __APPLE__
int sTiles::Control::SetAffinity(int rank) {
    // macOS does not support strict CPU affinity binding.
    // Simply return success without trying to bind.
    return 0;
}

#else

int sTiles::Control::SetAffinity(int rank) {
    hwloc_obj_t      obj;      /* Hwloc object    */
    hwloc_cpuset_t   cpuset;   /* HwLoc cpuset    */

    if (!topo_initialized) {
        sTiles::Logger::error("Topology not initialized when trying to set affinity for rank ", rank);
        return -110;
    }

    /* Get last one.  */
    obj = hwloc_get_obj_by_type(g_hwlocTopology, HWLOC_OBJ_CORE, rank);
    if (!obj)
        return -110;

    /* Get a copy of its cpuset that we may modify.  */
    /* Get only one logical processor (in case the core is SMT/hyperthreaded).  */
#if !defined(HWLOC_BITMAP_H)
    cpuset = hwloc_cpuset_dup(obj->cpuset);
    hwloc_cpuset_singlify(cpuset);
#else
    cpuset = hwloc_bitmap_dup(obj->cpuset);
    hwloc_bitmap_singlify(cpuset);
#endif

    /* And try to bind ourself there.  */
    if (hwloc_set_cpubind(g_hwlocTopology, cpuset, HWLOC_CPUBIND_THREAD)) {
        char *str = NULL;
#if !defined(HWLOC_BITMAP_H)
        hwloc_cpuset_asprintf(&str, obj->cpuset);
#else
        hwloc_bitmap_asprintf(&str, obj->cpuset);
#endif

        sTiles::Logger::warning("Couldn't bind thread to cpuset ", (str ? str : "N/A"));

        free(str);
        return -110;
    }

    /* Get the number at Proc level ( We don't want to use HyperThreading ) */
    rank = obj->children[0]->os_index;

    /* Free our cpuset copy */
#if !defined(HWLOC_BITMAP_H)
    hwloc_cpuset_free(cpuset);
#else
    hwloc_bitmap_free(cpuset);
#endif
    return 0;
}
#endif

/**
 * Remove affinity previously set by `SetAffinity`.
 */
#ifdef __APPLE__
int sTiles::Control::UnsetAffinity() {
    // macOS does not support strict CPU affinity binding.
    // Simply return success without trying to bind.
    return 0;
}
 
#else

int sTiles::Control::UnsetAffinity() {
    hwloc_obj_t      obj;      /* Hwloc object    */
    hwloc_cpuset_t   cpuset;   /* HwLoc cpuset    */

    if (!topo_initialized) {
        sTiles::Logger::error("Topology not initialized when unsetting affinity");
        return -110;
    }

    /* Get last one.  */
    obj = hwloc_get_obj_by_type(g_hwlocTopology, HWLOC_OBJ_MACHINE, 0);
    if (!obj) {
        sTiles::Logger::warning("Could not get hwloc machine object to unset affinity");
        return -110;
    }

    /* Get a copy of its cpuset that we may modify.  */
    /* Get only one logical processor (in case the core is SMT/hyperthreaded).  */
#if !defined(HWLOC_BITMAP_H)
    cpuset = hwloc_cpuset_dup(obj->cpuset);
#else
    cpuset = hwloc_bitmap_dup(obj->cpuset);
#endif

    /* And try to bind ourself there.  */
    if (hwloc_set_cpubind(g_hwlocTopology, cpuset, HWLOC_CPUBIND_THREAD)) {
        char *str = NULL;
#if !defined(HWLOC_BITMAP_H)
        hwloc_cpuset_asprintf(&str, obj->cpuset);
#else
        hwloc_bitmap_asprintf(&str, obj->cpuset);
#endif
        sTiles::Logger::warning("Could not unbind thread to the whole machine with cpuset ", (str ? str : "N/A"));
        free(str);
        return -110;
    }

    /* Free our cpuset copy */
#if !defined(HWLOC_BITMAP_H)
    hwloc_cpuset_free(cpuset);
#else
    hwloc_bitmap_free(cpuset);
#endif
    return 0;
}
#endif

int sTiles::Control::GetNumaSize()
{
    if ( g_hwlocGroupSize == -1 ) {
        hwloc_topology_t full_topology;
        hwloc_cpuset_t   cpuset = NULL;
        hwloc_obj_t      obj;
        int i;
        int nodesnbr = 1;

        /* Allocate and initialize topology object.  */
        hwloc_topology_init(&full_topology);

        /* Set flag for the whole system */
        hwloc_topology_set_flags(full_topology, HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM);

        /* Perform the topology detection.  */
        hwloc_topology_load(full_topology);

        /* Compute number of NUMA nodes */
        obj = hwloc_get_obj_by_type(full_topology, HWLOC_OBJ_MACHINE, 0);
        if (obj != NULL) {
#if !defined(HWLOC_BITMAP_H)
            cpuset = hwloc_cpuset_dup(obj->cpuset);
#else
            cpuset = hwloc_bitmap_dup(obj->cpuset);
#endif
            nodesnbr = hwloc_get_nbobjs_inside_cpuset_by_type(g_hwlocTopology, cpuset, HWLOC_OBJ_NODE);

#if !defined(HWLOC_BITMAP_H)
            hwloc_cpuset_free(cpuset);
#else
            hwloc_bitmap_free(cpuset);
#endif
        }
        nodesnbr = (nodesnbr > 0) ? nodesnbr : 1;

        /* Search size of NUMA nodes */
        for (i = 0; i < nodesnbr; i++) {
            obj = hwloc_get_obj_by_type(full_topology, HWLOC_OBJ_NODE, i);

            if (obj != NULL) {
#if !defined(HWLOC_BITMAP_H)
                cpuset = hwloc_cpuset_dup(obj->cpuset);
#else
                cpuset = hwloc_bitmap_dup(obj->cpuset);
#endif
                g_hwlocGroupSize = hwloc_get_nbobjs_inside_cpuset_by_type(g_hwlocTopology, cpuset, HWLOC_OBJ_CORE);

#if !defined(HWLOC_BITMAP_H)
                hwloc_cpuset_free(cpuset);
#else
                hwloc_bitmap_free(cpuset);
#endif

                if (g_hwlocGroupSize > 0)
                    break;
            }
        }

        hwloc_topology_destroy(full_topology);

        g_hwlocGroupSize = (g_hwlocGroupSize > 0) ? g_hwlocGroupSize : 1;
    }

    return g_hwlocGroupSize;
}
