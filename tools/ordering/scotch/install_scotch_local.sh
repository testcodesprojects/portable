#!/usr/bin/env bash
###############################################################################
# SCOTCH Automatic Build Script
# Builds SCOTCH from bundled source code for cross-platform compatibility
###############################################################################
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use bundled SCOTCH source (not external clone)
SCOTCH_SRC="${DIR}/scotch"
SCOTCH_BUILD="${SCOTCH_SRC}/build"
SCOTCH_INSTALL="${DIR}/scotch_local"

echo "================================================================================"
echo "Building SCOTCH graph ordering library"
echo "================================================================================"
echo "Source directory  : ${SCOTCH_SRC}"
echo "Build directory   : ${SCOTCH_BUILD}"
echo "Install directory : ${SCOTCH_INSTALL}"
echo ""

# Verify bundled source exists
if [ ! -d "${SCOTCH_SRC}/src" ]; then
    echo "ERROR: SCOTCH source not found in ${SCOTCH_SRC}"
    echo "Please ensure the scotch/ directory contains the complete source tree."
    exit 1
fi

# Detect number of CPU cores (cross-platform)
if command -v nproc > /dev/null 2>&1; then
    # Linux
    NCPU=$(nproc)
elif command -v sysctl > /dev/null 2>&1; then
    # macOS, BSD
    NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
else
    # Fallback
    NCPU=4
fi

echo "Detected ${NCPU} CPU cores for parallel build"
echo ""

# macOS: Check for modern bison (system bison 2.3 is too old for SCOTCH)
BISON_OPTS=""
if [[ "$(uname -s)" == "Darwin" ]]; then
    BISON_PATH=""

    # Function to check if bison version is 3.0 or newer
    check_bison_version() {
        local bison_bin="$1"
        if [ -x "$bison_bin" ]; then
            local ver=$("$bison_bin" --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            local major=$(echo "$ver" | cut -d. -f1)
            if [ -n "$major" ] && [ "$major" -ge 3 ] 2>/dev/null; then
                return 0  # Version 3.x or newer
            fi
        fi
        return 1  # Not found or too old
    }

    # 1. First check if bison in PATH is modern enough
    if command -v bison > /dev/null 2>&1; then
        CANDIDATE=$(command -v bison)
        if check_bison_version "$CANDIDATE"; then
            BISON_PATH="$CANDIDATE"
        fi
    fi

    # 2. If not found, check common locations
    if [ -z "$BISON_PATH" ]; then
        COMMON_LOCATIONS=(
            "$HOME/.local/bin/bison"
            "/opt/homebrew/opt/bison/bin/bison"
            "/usr/local/opt/bison/bin/bison"
            "/opt/local/bin/bison"
        )
        for loc in "${COMMON_LOCATIONS[@]}"; do
            if check_bison_version "$loc"; then
                BISON_PATH="$loc"
                break
            fi
        done
    fi

    # 3. If still not found, show error with instructions
    if [ -z "$BISON_PATH" ]; then
        echo "================================================================================"
        echo "ERROR: Modern bison (3.0+) not found!"
        echo ""
        echo "macOS ships with bison 2.3 (from 2006) which is too old for SCOTCH."
        echo ""
        echo "Option 1 - If you have Homebrew access:"
        echo "    brew install bison"
        echo ""
        echo "Option 2 - Build bison locally (no sudo required):"
        echo "    cd ~"
        echo "    curl -L https://ftp.gnu.org/gnu/bison/bison-3.8.2.tar.gz | tar xz"
        echo "    cd bison-3.8.2"
        echo "    ./configure --prefix=\$HOME/.local"
        echo "    make -j4 && make install"
        echo "    # Add to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo ""
        echo "Then run 'make cleanscotch && make all' again."
        echo "================================================================================"
        exit 1
    fi

    BISON_VERSION=$("${BISON_PATH}" --version | head -1)
    echo "Using bison: ${BISON_PATH}"
    echo "Version: ${BISON_VERSION}"
    echo ""
    BISON_OPTS="-DBISON_EXECUTABLE=${BISON_PATH}"
fi

# Clean and create build directory
rm -rf "${SCOTCH_BUILD}"
mkdir -p "${SCOTCH_BUILD}"
cd "${SCOTCH_BUILD}"

# Configure with CMake - build STATIC libraries for embedding in libstiles.so
# NOTE: We need -fPIC for static libs to be embedded in a shared library
echo "Configuring SCOTCH with CMake (static libraries with PIC)..."
cmake .. \
    -DBUILD_PTSCOTCH=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_C_FLAGS="-O3 -fPIC" \
    -DCMAKE_CXX_FLAGS="-O3 -fPIC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSCOTCH_DETERMINISTIC=FULL \
    -DCMAKE_INSTALL_PREFIX="${SCOTCH_INSTALL}" \
    ${BISON_OPTS}

echo ""
echo "Building SCOTCH (this may take 30-60 seconds)..."
make -j"${NCPU}"

echo ""
echo "Installing SCOTCH to ${SCOTCH_INSTALL}..."
make install

# Handle lib vs lib64 directory differences across systems
# Some systems (RHEL/CentOS) install to lib64/, others (Debian/Ubuntu) to lib/
# Create symlinks so both paths work regardless of which cmake chose
if [ -d "${SCOTCH_INSTALL}/lib64" ] && [ ! -d "${SCOTCH_INSTALL}/lib" ]; then
    echo "Creating lib -> lib64 symlink for compatibility..."
    ln -sf lib64 "${SCOTCH_INSTALL}/lib"
elif [ -d "${SCOTCH_INSTALL}/lib" ] && [ ! -d "${SCOTCH_INSTALL}/lib64" ]; then
    echo "Creating lib64 -> lib symlink for compatibility..."
    ln -sf lib "${SCOTCH_INSTALL}/lib64"
fi

# Determine actual library directory
if [ -d "${SCOTCH_INSTALL}/lib64" ] && [ ! -L "${SCOTCH_INSTALL}/lib64" ]; then
    LIB_DIR="${SCOTCH_INSTALL}/lib64"
else
    LIB_DIR="${SCOTCH_INSTALL}/lib"
fi

echo ""
echo "================================================================================"
echo "SCOTCH build completed successfully!"
echo "================================================================================"
echo "Headers installed in: ${SCOTCH_INSTALL}/include"
echo "Libraries installed in: ${LIB_DIR}"
echo ""
ls -lh "${LIB_DIR}"/*.a
echo "================================================================================"

# Remove .git directory to avoid conflicts with parent repo
if [ -d "${SCOTCH_SRC}/.git" ]; then
    echo "Removing SCOTCH .git directory to avoid git conflicts..."
    rm -rf "${SCOTCH_SRC}/.git"
fi
