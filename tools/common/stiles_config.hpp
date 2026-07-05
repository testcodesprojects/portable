/**
 * @file stiles_config.hpp
 * @brief Library configuration, version information, and license management.
 *
 * Defines version macros, license expiration settings, ANSI color codes for
 * terminal output, and compile-time configuration options for the sTiles library.
 *
 * @author Esmail Abdul Fattah
 * @affiliation King Abdullah University of Science and Technology (KAUST)
 * @version 3.0.0
 *
 * @copyright Copyright (c) 2026, KAUST. All rights reserved.
 *
 * PROPRIETARY AND CONFIDENTIAL
 * This software is the exclusive property of KAUST. Unauthorized copying, modification,
 * distribution, or use of this software, in whole or in part, is strictly prohibited.
 * This code may not be reproduced, reverse-engineered, or used to create derivative works
 * without explicit written permission from the copyright holder.
 */

#ifndef STILES_CONFIG_HPP
#define STILES_CONFIG_HPP

#include <cstdio>
#include <cstdlib>
#include <ctime>

// =============================================================================
//  Preprocessor Macros (for build systems & string literal concatenation)
// =============================================================================

/**
 * @name Version Macros
 * @brief Preprocessor definitions for versioning, primarily for build system
 *        integration and cases where stringification is needed.
 * @{
 */
#define STILES_VERSION_MAJOR 3
#define STILES_VERSION_MINOR 0
#define STILES_VERSION_MICRO 0
#define STILES_VERSION_STRING "3.0.0"
/** @} */

/**
 * @name ANSI Color Code Macros
 * @brief Macros for ANSI colors. Kept as macros to allow easy concatenation
 *        within the STILES_LOGO_STRING literal.
 * @{
 */
#define STILES_COLOR_BOLD_CYAN   "\033[1;96m"
#define STILES_COLOR_GRAY        "\033[0;90m"
#define STILES_COLOR_WHITE       "\033[0;97m"
#define STILES_COLOR_RESET       "\033[0m"
/** @} */

/**
 * @name ASCII Art Macro
 * @brief The library's ASCII art logo. Defined as a macro to handle its
 *        multi-line nature and embedded color codes cleanly.
 * @{
 */
#define STILES_LOGO_STRING \
    STILES_COLOR_GRAY "          ╭────────────────────────────────────────────────────────────╮\n" \
    "          │" STILES_COLOR_BOLD_CYAN "                                                            " STILES_COLOR_GRAY "│\n" \
    "          │" STILES_COLOR_BOLD_CYAN "                *******      *                              " STILES_COLOR_GRAY "│\n" \
    "          │" STILES_COLOR_BOLD_CYAN "                   *         *                              " STILES_COLOR_GRAY "│\n" \
    "          │" STILES_COLOR_BOLD_CYAN "          ****     *    *    *    *****      ****           " STILES_COLOR_GRAY "│\n" \
    "          │" STILES_COLOR_BOLD_CYAN "         *         *         *   *     *    *               " STILES_COLOR_GRAY "│\n" \
    "          │" STILES_COLOR_BOLD_CYAN "          ****     *    *    *    *****      ****           " STILES_COLOR_GRAY "│\n" \
    "          │" STILES_COLOR_BOLD_CYAN "              *    *    *    *    *              *          " STILES_COLOR_GRAY "│\n" \
    "          │" STILES_COLOR_BOLD_CYAN "          ****     *    *    *     ****      ****           " STILES_COLOR_GRAY "│\n" \
    "          │" STILES_COLOR_BOLD_CYAN "                                                            " STILES_COLOR_GRAY "│\n" \
    "          │" STILES_COLOR_WHITE "    High Performance Framework for Tiling Sparse Matrices   " STILES_COLOR_GRAY "│\n" \
    "          │                                                            │\n" \
    "          ╰────────────────────────────────────────────────────────────╯" STILES_COLOR_RESET
/** @} */

namespace sTiles {

// =============================================================================
//  Compile-Time Configuration Constants (for C++ code)
// =============================================================================
namespace config {

    /**
     * @name Version Constants
     * @brief Type-safe, constexpr variables for use within C++ code.
     * @{
     */
    constexpr int versionMajor = STILES_VERSION_MAJOR;
    constexpr int versionMinor = STILES_VERSION_MINOR;
    constexpr int versionMicro = STILES_VERSION_MICRO;
    static constexpr const char* versionString = STILES_VERSION_STRING;
    /** @} */

} // namespace config


// =============================================================================
//  Library Utility Functions
// =============================================================================

/**
 * @brief Displays the sTiles ASCII art logo and version information.
 *
 * @note This is defined as an inline function in the header for convenience.
 */
inline void show_logo() {
    // Print the banner at most once per process. Bench / harnesses that
    // create multiple sTiles_object instances would otherwise repeat it.
    static bool shown = false;
    if (shown) return;
    shown = true;
    // STILES_NO_BANNER=1 suppresses it entirely (test harnesses printing
    // their own tables, CSV pipelines, etc.).
    if (const char* nb = getenv("STILES_NO_BANNER")) {
        if (nb[0] == '1') return;
    }
    printf("\n%s\n", STILES_LOGO_STRING);
    printf("          Version: " STILES_COLOR_WHITE "%s" STILES_COLOR_RESET " | Copyright (c) 2026 KAUST\n\n",
           config::versionString);
}

} // namespace sTiles

#endif // STILES_CONFIG_HPP
