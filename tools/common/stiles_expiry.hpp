/**
 * @file stiles_expiry.hpp
 * @brief Hard expiration protection for sTiles library.
 *
 * This file implements time-limited functionality that expires on a fixed date.
 * Multiple layers of protection make binary bypass difficult.
 */

#ifndef STILES_EXPIRY_HPP
#define STILES_EXPIRY_HPP

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <sys/time.h>
#endif

namespace sTiles {
namespace internal {

// ============================================================================
//  CONFIGURATION
// ============================================================================

// How many years the library is valid after build (default: 2 years)
#ifndef STILES_VALID_YEARS
#define STILES_VALID_YEARS 2
#endif

// Build date macros (set by Makefile via -D flags, or use defaults)
#ifndef STILES_BUILD_YEAR
#define STILES_BUILD_YEAR 2026
#endif
#ifndef STILES_BUILD_MONTH
#define STILES_BUILD_MONTH 1
#endif
#ifndef STILES_BUILD_DAY
#define STILES_BUILD_DAY 1
#endif

// Computed expiration: build date + STILES_VALID_YEARS
constexpr int BUILD_Y = STILES_BUILD_YEAR;
constexpr int BUILD_M = STILES_BUILD_MONTH;
constexpr int BUILD_D = STILES_BUILD_DAY;
constexpr int EXP_Y = BUILD_Y + STILES_VALID_YEARS;
constexpr int EXP_M = BUILD_M;
constexpr int EXP_D = BUILD_D;

// Obfuscation constants (change these to any random values for your build)
constexpr unsigned int OBF_KEY1 = 0xA7B3C2D1;
constexpr unsigned int OBF_KEY2 = 0x3F2E1D0C;

// ============================================================================
//  Time retrieval (bypass LD_PRELOAD by using direct syscall)
// ============================================================================

/**
 * @brief Get current time using direct syscall (harder to hook)
 */
inline time_t get_secure_time() {
#ifdef __linux__
    struct timespec ts;
    // Direct syscall bypasses LD_PRELOAD hooks on time()/gettimeofday()
    if (syscall(SYS_clock_gettime, CLOCK_REALTIME, &ts) == 0) {
        return ts.tv_sec;
    }
#endif

#ifdef __APPLE__
    struct timeval tv;
    // macOS: gettimeofday is still hookable, but we add redundancy
    if (gettimeofday(&tv, nullptr) == 0) {
        return tv.tv_sec;
    }
#endif

    // Fallback (hookable, but we have other checks)
    return time(nullptr);
}

/**
 * @brief Convert date to days since epoch (simplified)
 */
constexpr inline long date_to_days(int y, int m, int d) {
    return static_cast<long>(y) * 365L + y/4 - y/100 + y/400 + (m-1)*30 + d;
}

/**
 * @brief Get current date as days (using secure time)
 */
inline long get_current_days() {
    time_t now = get_secure_time();
    struct tm* t = localtime(&now);
    return date_to_days(t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
}

// Expiration date as days (computed at compile time)
constexpr long EXP_DAYS = date_to_days(EXP_Y, EXP_M, EXP_D);

// ============================================================================
//  Obfuscated expiration check
// ============================================================================

/**
 * @brief Check if expired using obfuscated comparison
 * @return Obfuscated result (0 if valid, non-zero pattern if expired)
 */
inline unsigned int check_time_validity() {
    long current = get_current_days();
    long limit = EXP_DAYS;

    // Obfuscated comparison - don't use obvious (current > limit)
    unsigned int result = 0;
    if ((current ^ OBF_KEY1) > (limit ^ OBF_KEY1)) {
        result = OBF_KEY2;
    }
    return result;
}

/**
 * @brief Returns a factor that becomes 0.0 after expiration
 * Use this in computations to silently break results
 */
inline double validity_factor() {
    unsigned int v = check_time_validity();
    // If valid: v=0, so (v == 0) = 1, factor = 1.0
    // If expired: v=OBF_KEY2, so (v == 0) = 0, factor = 0.0
    return static_cast<double>(v == 0);
}

/**
 * @brief Returns a scale that corrupts results after expiration
 * More subtle than validity_factor - introduces small errors
 */
inline double validity_scale() {
    unsigned int v = check_time_validity();
    if (v == 0) return 1.0;
    // Return a value close to 1 but wrong - causes subtle numerical errors
    return 0.9999999;  // Will accumulate errors in iterative algorithms
}

// ============================================================================
//  Exit handler (called from main check points)
// ============================================================================

/**
 * @brief Print expiration message and exit
 * Message is built character-by-character to avoid obvious strings
 */
inline void handle_expiration() {
    // Build message without obvious string literals
    char m1[64], m2[64], m3[64];

    // "sTiles evaluation period has ended."
    snprintf(m1, sizeof(m1), "%cTiles %s period %s.",
             's', "evaluation", "has ended");

    // "Please visit https://..."
    snprintf(m2, sizeof(m2), "Please %s for a %s.",
             "contact the authors", "licensed version");

    // "Exiting."
    snprintf(m3, sizeof(m3), "%s.", "Exiting");

    fprintf(stderr, "\n");
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "  %s\n", m1);
    fprintf(stderr, "  %s\n", m2);
    fprintf(stderr, "  %s\n", m3);
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "\n");

    // Use _Exit to avoid calling atexit handlers (harder to hook)
    _Exit(1);
}

// ============================================================================
//  Primary check (call at initialization)
// ============================================================================

/**
 * @brief Main expiration check - call at library init
 */
inline void verify_expiry() {
    if (check_time_validity() != 0) {
        handle_expiration();
    }
}

// ============================================================================
//  Cached validity (zero runtime overhead after init)
// ============================================================================

namespace detail {
    // Cached validity state - set once at init, never rechecked
    inline double& cached_validity_factor() {
        static double val = 1.0;  // Default valid
        return val;
    }

    inline bool& is_initialized() {
        static bool val = false;
        return val;
    }
}

/**
 * @brief Initialize cached validity (call once at startup)
 * After this, validity_factor_cached() returns 1.0 or 0.0 with zero overhead
 */
inline void cache_validity_state() {
    if (check_time_validity() != 0) {
        detail::cached_validity_factor() = 0.0;
    } else {
        detail::cached_validity_factor() = 1.0;
    }
    detail::is_initialized() = true;
}

/**
 * @brief Get cached validity factor (zero overhead - just returns stored value)
 */
inline double validity_factor_cached() {
    return detail::cached_validity_factor();
}

// ============================================================================
//  Optional secondary checks (use sparingly - only where acceptable)
// ============================================================================

/**
 * @brief Quick inline check - use only in non-performance-critical paths
 * Call as: STILES_CHECK_VALID()
 */
#define STILES_CHECK_VALID() \
    do { if (sTiles::internal::check_time_validity() != 0) \
         sTiles::internal::handle_expiration(); } while(0)

/**
 * @brief Zero-overhead validation using cached result
 * Use in computations: val = STILES_VALIDATED_CACHED(val)
 * After expiry, multiplies by 0.0 (corrupts results)
 */
#define STILES_VALIDATED_CACHED(x) ((x) * sTiles::internal::validity_factor_cached())

// Legacy macros (for backwards compatibility - use cached versions for performance)
#define STILES_VALIDATED(x) STILES_VALIDATED_CACHED(x)
#define STILES_SCALED(x) ((x) * sTiles::internal::validity_factor_cached())

// ============================================================================
//  Counter-based check (detects clock manipulation during execution)
// ============================================================================

namespace detail {
    inline long& initial_days() {
        static long val = -1;
        return val;
    }
}

/**
 * @brief Initialize time tracking (call once at start)
 */
inline void init_time_tracking() {
    detail::initial_days() = get_current_days();
}

/**
 * @brief Check if clock has gone backwards (sign of manipulation)
 */
inline bool clock_manipulated() {
    long now = get_current_days();
    long init = detail::initial_days();
    if (init < 0) return false;  // Not initialized yet
    // If current time is before initial time, clock was set back
    return (now < init - 1);  // Allow 1 day tolerance for timezone issues
}

/**
 * @brief Combined check: expiry + clock manipulation
 */
inline void full_validity_check() {
    if (check_time_validity() != 0 || clock_manipulated()) {
        handle_expiration();
    }
}

#define STILES_FULL_CHECK() \
    do { sTiles::internal::full_validity_check(); } while(0)

} // namespace internal
} // namespace sTiles

#endif // STILES_EXPIRY_HPP
