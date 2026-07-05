/* Forwarding header for SUITESPARSE_INLINE=1 builds.
 *
 * sTiles wrappers include <suitesparse/amd.h>. The cmake sub-build installs the
 * headers under suitesparse_local/include/suitesparse/, but the inline build
 * does not run cmake, so this shim provides the <suitesparse/...> prefix by
 * forwarding to the in-tree source header. The transitive "SuiteSparse_config.h"
 * is resolved via -I.../SuiteSparse_config (set in make.inc inline branch).
 */
#include "../../SuiteSparse/AMD/Include/amd.h"
