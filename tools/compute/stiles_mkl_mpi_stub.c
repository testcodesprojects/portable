/*
 * Stub for MKLMPI_Get_wrappers, referenced by libmkl_core.a.
 *
 * libmkl_core.a contains the entry points for MKL's distributed (MPI) layer
 * (ScaLAPACK / Cluster Sparse Solver / etc.). Even sequential, single-node
 * MKL pulls those object files in when libstiles.so is built with
 * `-Wl,--whole-archive` over the static MKL archives, leaving an undefined
 * reference to MKLMPI_Get_wrappers if no MPI library is linked.
 *
 * sTiles never enters MKL's distributed code paths, so this stub satisfies
 * the linker without dragging in libmpi/libscalapack. The assert ensures
 * that *if* something ever does enter that code path at runtime, the failure
 * is loud and traceable rather than silent corruption.
 */
#include <assert.h>

void MKLMPI_Get_wrappers(void)
{
    assert(0 == 1 &&
           "MKLMPI_Get_wrappers called: sTiles links MKL sequentially and "
           "should never reach MKL's distributed code path. If this fires, "
           "either MKL's MPI layer was invoked unexpectedly or the MKL "
           "static archives were rebuilt with cluster components linked "
           "against this stub by mistake.");
}
