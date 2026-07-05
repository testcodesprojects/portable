#!/usr/bin/env bash
###############################################################################
# libxsmm automatic build helper
# - Clones the upstream libxsmm project (depth=1)
# - Builds static + shared libraries with PIC
# - Installs into tools/libxsmm/libxsmm_local
###############################################################################
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBXSMM_SRC="${DIR}/libxsmm"
LIBXSMM_INSTALL="${DIR}/libxsmm_local"
LIBXSMM_REF="${LIBXSMM_REF:-main}"

# Detect number of cores
if command -v nproc >/dev/null 2>&1; then
    NCPU=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
else
    NCPU=4
fi

printf "\n================================================================================\n"
printf "Building libxsmm (ref: %s)\n" "${LIBXSMM_REF}"
printf "Install prefix : %s\n" "${LIBXSMM_INSTALL}"
printf "================================================================================\n\n"

if [ -d "${LIBXSMM_SRC}" ]; then
    echo "Removing existing libxsmm source tree..."
    rm -rf "${LIBXSMM_SRC}"
fi
echo "Cloning libxsmm from GitHub..."
git clone --depth 1 --branch "${LIBXSMM_REF}" https://github.com/libxsmm/libxsmm.git "${LIBXSMM_SRC}"

# libxsmm uses plain Make (not CMake). Build static + shared with PIC.
cd "${LIBXSMM_SRC}"

# STATIC=1 builds libxsmm.a (we'll embed into libstiles.so with --whole-archive
# elsewhere if needed). PREFIX sets install location. PIC=1 ensures the static
# archive can be linked into a shared library.
make -j"${NCPU}" \
    STATIC=1 \
    BLAS=0 \
    FORTRAN=0 \
    INTEL=0 \
    CC=gcc \
    CXX=g++ \
    PREFIX="${LIBXSMM_INSTALL}"

make install-minimal PREFIX="${LIBXSMM_INSTALL}"

printf "\n================================================================================\n"
printf "libxsmm installed to: %s\n" "${LIBXSMM_INSTALL}"
printf "  Headers : %s/include/libxsmm.h\n" "${LIBXSMM_INSTALL}"
printf "  Static  : %s/lib/libxsmm.a\n" "${LIBXSMM_INSTALL}"
printf "================================================================================\n\n"
