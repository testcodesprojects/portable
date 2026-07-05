#!/usr/bin/env bash
###############################################################################
# METIS + GKlib automatic build helper
# - Builds and installs GKlib into tools/ordering/GKlib/local
# - Builds and installs METIS into tools/ordering/metis-5.1.0/local
###############################################################################
set -euo pipefail

# Clear inherited MAKEFLAGS so child `make` invocations don't misinterpret
# parent flags (e.g. -w) as build targets.
unset MAKEFLAGS MFLAGS MAKELEVEL

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GKLIB_DIR="${DIR}/GKlib"
METIS_DIR="${DIR}/metis-5.1.0"
GKLIB_PREFIX="${GKLIB_DIR}/local"
METIS_PREFIX="${METIS_DIR}/local"

if [ ! -d "${GKLIB_DIR}" ]; then
    echo "ERROR: GKlib source tree not found at ${GKLIB_DIR}" >&2
    exit 1
fi
if [ ! -d "${METIS_DIR}" ]; then
    echo "ERROR: metis-5.1.0 source tree not found at ${METIS_DIR}" >&2
    exit 1
fi

printf "\n================================================================================\n"
printf "Building GKlib\n"
printf "Install prefix : %s\n" "${GKLIB_PREFIX}"
printf "================================================================================\n\n"

cd "${GKLIB_DIR}"
make config prefix="${GKLIB_PREFIX}"
make install

# Some distros (RHEL/Fedora multilib) install to lib64/ instead of lib/.
# The Makefile and METIS build expect lib/, so create a symlink if needed.
normalize_lib_dir() {
    local prefix="$1"
    if [ ! -d "${prefix}/lib" ] && [ -d "${prefix}/lib64" ]; then
        ln -s lib64 "${prefix}/lib"
    fi
}
normalize_lib_dir "${GKLIB_PREFIX}"

if ! [ -e "${GKLIB_PREFIX}/lib/libGKlib.a" ]; then
    echo "ERROR: GKlib install did not produce ${GKLIB_PREFIX}/lib/libGKlib.a" >&2
    echo "Contents of GKlib install dir:" >&2
    find "${GKLIB_PREFIX}" -name "libGKlib*" >&2 || true
    exit 1
fi

printf "\n================================================================================\n"
printf "Building METIS 5.1.0\n"
printf "Install prefix : %s\n" "${METIS_PREFIX}"
printf "================================================================================\n\n"

cd "${METIS_DIR}"
make config prefix="${METIS_PREFIX}" gklib_path="${GKLIB_PREFIX}"
make install
normalize_lib_dir "${METIS_PREFIX}"

printf "\n================================================================================\n"
printf "METIS + GKlib build complete.\n"
printf "  GKlib:  %s\n" "${GKLIB_PREFIX}"
printf "  METIS:  %s\n" "${METIS_PREFIX}"
printf "================================================================================\n\n"
