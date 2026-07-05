/* Configured metis.h for METIS_INLINE=1 builds.
 *
 * The cmake sub-build generates its metis.h by prepending the integer/real
 * widths to the source header (build/xinclude/metis.h, local/include/metis.h).
 * The inline build does not run cmake, so this shim reproduces that exactly and
 * forwards to the in-tree source header — no dependency on metis-5.1.0/local/.
 *
 * sTiles' own C++ sources include this via -DSTILES_METIS_HEADER and -I; the
 * METIS .c sources get the same widths from -DIDXTYPEWIDTH in MT_CFLAGS.
 */
#define IDXTYPEWIDTH 32
#define REALTYPEWIDTH 32
#include "../metis-5.1.0/include/metis.h"
