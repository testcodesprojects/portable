#!/usr/bin/env bash
###############################################################################
# SuiteSparse automatic build helper
# - Clones the upstream SuiteSparse project (depth=1)
# - Configures only the components we need (suitesparse_config, AMD family, BTF)
# - Installs into tools/suitesparse/suitesparse_local
###############################################################################
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITESPARSE_SRC="${DIR}/SuiteSparse"
SUITESPARSE_BUILD="${SUITESPARSE_SRC}/build"
SUITESPARSE_INSTALL="${DIR}/suitesparse_local"
SUITESPARSE_PROJECTS="suitesparse_config;amd;camd;colamd;ccolamd;btf"

# Detect number of cores for parallel builds
if command -v nproc >/dev/null 2>&1; then
    NCPU=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
else
    NCPU=4
fi

printf "\n================================================================================\n"
printf "Building SuiteSparse (projects: %s)\n" "${SUITESPARSE_PROJECTS}"
printf "Install prefix : %s\n" "${SUITESPARSE_INSTALL}"
printf "================================================================================\n\n"

if [ -d "${SUITESPARSE_SRC}" ]; then
    echo "Removing existing SuiteSparse source tree..."
    rm -rf "${SUITESPARSE_SRC}"
fi
echo "Cloning SuiteSparse from GitHub..."
git clone --depth 1 https://github.com/DrTimothyAldenDavis/SuiteSparse "${SUITESPARSE_SRC}"

rm -rf "${SUITESPARSE_BUILD}"
# Build static libraries with PIC for embedding into libstiles.so
cmake -S "${SUITESPARSE_SRC}" -B "${SUITESPARSE_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${SUITESPARSE_INSTALL}" \
    -DSUITESPARSE_ENABLE_PROJECTS="${SUITESPARSE_PROJECTS}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_C_FLAGS="-O3 -fPIC" \
    -DCMAKE_CXX_FLAGS="-O3 -fPIC"

cmake --build "${SUITESPARSE_BUILD}" -j"${NCPU}"
cmake --install "${SUITESPARSE_BUILD}"

# Handle lib vs lib64 directory differences across systems
# Some systems (RHEL/CentOS) install to lib64/, others (Debian/Ubuntu) to lib/
# Create symlinks so both paths work regardless of which cmake chose
if [ -d "${SUITESPARSE_INSTALL}/lib64" ] && [ ! -d "${SUITESPARSE_INSTALL}/lib" ]; then
    echo "Creating lib -> lib64 symlink for compatibility..."
    ln -sf lib64 "${SUITESPARSE_INSTALL}/lib"
elif [ -d "${SUITESPARSE_INSTALL}/lib" ] && [ ! -d "${SUITESPARSE_INSTALL}/lib64" ]; then
    echo "Creating lib64 -> lib symlink for compatibility..."
    ln -sf lib "${SUITESPARSE_INSTALL}/lib64"
fi

# Determine actual library directory
if [ -d "${SUITESPARSE_INSTALL}/lib64" ] && [ ! -L "${SUITESPARSE_INSTALL}/lib64" ]; then
    LIB_DIR="${SUITESPARSE_INSTALL}/lib64"
else
    LIB_DIR="${SUITESPARSE_INSTALL}/lib"
fi

printf "\nSuiteSparse components installed under %s\n" "${SUITESPARSE_INSTALL}"
ls -1 "${LIB_DIR}" 2>/dev/null || true

# Remove .git directory to avoid conflicts with parent repo
if [ -d "${SUITESPARSE_SRC}/.git" ]; then
    echo "Removing SuiteSparse .git directory to avoid git conflicts..."
    rm -rf "${SUITESPARSE_SRC}/.git"
fi
