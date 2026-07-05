/**
 * @file stiles_logger.hpp
 * @brief Configurable logging system for sTiles library.
 *
 * Implements a flexible logging framework with compile-time log level filtering,
 * timestamping, source location support, and variadic message formatting for
 * debugging and runtime diagnostics.
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

#ifndef STILES_LOGGER_HPP
#define STILES_LOGGER_HPP

#include <iostream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <sstream>         // For variadic template message building
#include <utility>         // For std::forward
#if __has_include(<source_location>)
  #include <source_location>
  #if defined(__cpp_lib_source_location)
    #define STILES_HAVE_SOURCE_LOCATION 1
  #else
    #define STILES_HAVE_SOURCE_LOCATION 0
  #endif
#else
  #define STILES_HAVE_SOURCE_LOCATION 0
#endif
#include <cstdlib>         // For std::exit
#include <chrono>          // For timestamps
#include <mutex>           // For thread-safe stream writes
#include <vector>          // For holding extracted tags
#include <ctime>           // For time formatting helpers
#include <array>           // For known log level tags

// Bring in sTiles core types (StatusCode, etc.)
#include "stiles_types.hpp"  // <<<--- ADD THIS LINE

// --- Compile-Time Log Level Definitions ---
#define STILES_LOG_LEVEL_TIMEONLY  -2
#define STILES_LOG_LEVEL_NONE      -1
#define STILES_LOG_LEVEL_TIME       0
#define STILES_LOG_LEVEL_INFO       1
#define STILES_LOG_LEVEL_DEBUG      2
#define STILES_LOG_LEVEL_TRACE      3

#ifndef STILES_LOG_LEVEL
#define STILES_LOG_LEVEL STILES_LOG_LEVEL_TIMEONLY   // or STILES_LOG_LEVEL_NONE
#endif

namespace sTiles {

#if STILES_HAVE_SOURCE_LOCATION
using SourceLocation = std::source_location;
#else
struct SourceLocation {
    const char* file;
    const char* function;
    unsigned line_value;

    static constexpr SourceLocation current(const char* file_,
                                            const char* function_,
                                            unsigned line_) noexcept {
        return SourceLocation{file_, function_, line_};
    }

    const char* file_name() const noexcept { return file; }
    const char* function_name() const noexcept { return function; }
    unsigned line_number() const noexcept { return line_value; }
    unsigned line() const noexcept { return line_value; }
};
#endif

#if STILES_HAVE_SOURCE_LOCATION
  #define STILES_CURRENT_SOURCE_LOCATION ::sTiles::SourceLocation::current()
#else
  #define STILES_CURRENT_SOURCE_LOCATION ::sTiles::SourceLocation::current(__FILE__, __func__, __LINE__)
#endif

// --- Log Level Enum ---
enum class Level {
    TimingOnly = STILES_LOG_LEVEL_TIMEONLY,                 // Emit timing markers only
    None       = STILES_LOG_LEVEL_NONE,
    Time       = STILES_LOG_LEVEL_TIME,
    Info       = STILES_LOG_LEVEL_INFO,
    Debug      = STILES_LOG_LEVEL_DEBUG,
    Trace      = STILES_LOG_LEVEL_TRACE
};

namespace detail {
    inline Level current_level_ = Level::Time;
    // Serializes stream writes so concurrent log calls (e.g. the parallel
    // candidate-eval threads) emit whole lines instead of interleaving.
    inline std::mutex log_mutex_;
} // namespace detail

// --- Runtime Log Level Control ---
inline void setLevel(Level level, bool enable = true) {
    detail::current_level_ = enable ? level : Level::None;
}
inline Level getLevel() {
    return detail::current_level_;
}

// --- Logger (variadic, zero-cost when compiled out) ---
class Logger {
public:
    template<typename... Args> static void info(Args&&... args) {
#if STILES_LOG_LEVEL >= STILES_LOG_LEVEL_INFO
        if (static_cast<int>(sTiles::getLevel()) >= static_cast<int>(Level::Info)) {
            log_impl("[INFO]", std::forward<Args>(args)...);
        }
#endif
    }

    template<typename... Args> static void debug(Args&&... args) {
#if STILES_LOG_LEVEL >= STILES_LOG_LEVEL_DEBUG
        if (static_cast<int>(sTiles::getLevel()) >= static_cast<int>(Level::Debug)) {
            log_impl("[DEBUG]", std::forward<Args>(args)...);
        }
#endif
    }

    template<typename... Args> static void trace(Args&&... args) {
#if STILES_LOG_LEVEL >= STILES_LOG_LEVEL_TRACE
        if (static_cast<int>(sTiles::getLevel()) >= static_cast<int>(Level::Trace)) {
            log_impl("[TRACE]", std::forward<Args>(args)...);
        }
#endif
    }

    template<typename... Args> static void timing(Args&&... args) {
#if STILES_LOG_LEVEL >= STILES_LOG_LEVEL_TIME
        const auto level = sTiles::getLevel();
        if (level == Level::TimingOnly || static_cast<int>(level) >= static_cast<int>(Level::Time)) {
            log_impl("[TIME]", std::forward<Args>(args)...);
        }
#endif
    }

    // Always-on variant of timing(): emits the same "[TIME] <timestamp>" line
    // format but ignores the runtime log level, so the message is visible even
    // at level None / in --csv/quiet sweeps. Use only for decisions that
    // downstream tooling greps for (e.g. the chosen ordering, the corner
    // probe). Keep any literal grep token (e.g. "[ORDERING]") in the message
    // body so existing parsers keep matching.
    template<typename... Args> static void timing_always(Args&&... args) {
        // "Always" means independent of the timing gate, not of an EXPLICIT
        // request for silence: Level::None (sTiles_set_log_level(-1)) wins.
        if (sTiles::getLevel() == Level::None) return;
        log_impl("[TIME]", std::forward<Args>(args)...);
    }

    // Raw output without any prefix or timestamp - for formatted boxes
    template<typename... Args> static void raw(Args&&... args) {
#if STILES_LOG_LEVEL >= STILES_LOG_LEVEL_TIME
        const auto level = sTiles::getLevel();
        if (level == Level::TimingOnly || static_cast<int>(level) >= static_cast<int>(Level::Time)) {
            std::cout << build_message(std::forward<Args>(args)...) << '\n';
        }
#endif
    }

    template<typename... Args> static void warning(Args&&... args) {
#if STILES_LOG_LEVEL >= STILES_LOG_LEVEL_INFO
        if (static_cast<int>(sTiles::getLevel()) >= static_cast<int>(Level::Info)) {
            log_with_context_impl("[WARNING]", STILES_CURRENT_SOURCE_LOCATION, std::forward<Args>(args)...);
        }
#endif
    }

    template<typename... Args> static void error(Args&&... args) {
        log_with_context_impl("[ERROR]", STILES_CURRENT_SOURCE_LOCATION, std::forward<Args>(args)...);
    }

    template<typename... Args> static void fatal(Args&&... args) {
        log_with_context_impl("[FATAL]", STILES_CURRENT_SOURCE_LOCATION, std::forward<Args>(args)...);
        std::exit(EXIT_FAILURE);
    }

private:
    template<typename... Args>
    static std::string build_message(Args&&... args) {
        std::stringstream ss;
        ((ss << std::forward<Args>(args)), ...);
        return ss.str();
    }

    template<typename... Args>
    static void log_impl(const char* label, Args&&... args) {
        log_to_stream(std::cout, label, build_message(std::forward<Args>(args)...));
    }

    template<typename... Args>
    static void log_with_context_impl(const char* label,
                                      const SourceLocation& loc,
                                      Args&&... args) {
        auto base_message = build_message(std::forward<Args>(args)...);
        std::ostringstream ctx;
        ctx << base_message
            << " (in " << loc.function_name()
            << " at " << loc.file_name() << ":"
#if STILES_HAVE_SOURCE_LOCATION
            << loc.line()
#else
            << loc.line_number()
#endif
            << ")";
        log_to_stream(std::cerr, label, ctx.str());
    }

    static void log_to_stream(std::ostream& os,
                              const char* label,
                              const std::string& message) {
        std::lock_guard<std::mutex> _lk(detail::log_mutex_);
        const std::string timestamp = current_timestamp();
        const std::string level_label(label);
        std::istringstream reader(message);
        std::string line;
        bool processed_line = false;

        // Process each line individually so multi-line messages stay well formatted.
        while (std::getline(reader, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            auto parsed = extract_leading_tags(line, level_label);
            const std::string& line_label = parsed.label_override.empty() ? level_label : parsed.label_override;
            emit_line(os, line_label, timestamp, parsed.tags, line);
            processed_line = true;
        }

        if (!processed_line) {
            emit_line(os, level_label, timestamp, {}, std::string{});
        }
    }

    static std::string current_timestamp() {
        using clock_t = std::chrono::system_clock;
        const auto now = clock_t::now();
        const auto time_t_now = clock_t::to_time_t(now);

        std::tm tm_buf{};
#if defined(_WIN32)
        localtime_s(&tm_buf, &time_t_now);
#else
        localtime_r(&time_t_now, &tm_buf);
#endif

        const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % std::chrono::seconds(1);

        std::ostringstream ts;
        ts << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << millis.count();
        return ts.str();
    }

    struct TagParseResult {
        std::vector<std::string> tags;
        std::string label_override;
    };

    static bool is_level_tag(const std::string& tag) {
        static constexpr std::array<const char*, 7> known = {
            "[NONE]", "[TIME]", "[INFO]", "[DEBUG]", "[TRACE]", "[WARNING]", "[ERROR]"
        };

        return std::find_if(known.begin(), known.end(), [&](const char* known_tag) {
            return tag == known_tag;
        }) != known.end() || tag == "[FATAL]";
    }

    static TagParseResult extract_leading_tags(std::string& line,
                                               const std::string& primary_label) {
        TagParseResult result;
        std::size_t pos = 0;
        bool override_locked = false;

        while (pos < line.size() && line[pos] == '[') {
            const auto close = line.find(']', pos);
            if (close == std::string::npos) {
                break;
            }

            std::string tag = line.substr(pos, close - pos + 1);
            pos = close + 1;

            while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) {
                ++pos;
            }

            if (tag == primary_label) {
                continue;
            }

            if (!override_locked && is_level_tag(tag)) {
                result.label_override = std::move(tag);
                override_locked = true;
                continue;
            }

            const auto already_added = std::find(result.tags.begin(), result.tags.end(), tag) != result.tags.end();
            if (!already_added) {
                result.tags.push_back(std::move(tag));
            }
        }

        if (pos > 0) {
            line.erase(0, pos);
        }

        return result;
    }

    static void emit_line(std::ostream& os,
                          const std::string& level_label,
                          const std::string& timestamp,
                          const std::vector<std::string>& tags,
                          const std::string& content) {
        std::ostringstream formatted;

        static constexpr std::size_t kLabelWidth = 9; // Longest known label (e.g., [WARNING])
        std::string label_column = level_label;
        if (label_column.size() < kLabelWidth) {
            label_column.append(kLabelWidth - label_column.size(), ' ');
        }

        formatted << label_column << ' ' << timestamp;

        if (!tags.empty() || !content.empty()) {
            formatted << "  ";
        }

        if (!tags.empty()) {
            for (std::size_t i = 0; i < tags.size(); ++i) {
                formatted << tags[i];
                if (i + 1 < tags.size()) {
                    formatted << ' ';
                }
            }
            if (!content.empty()) {
                formatted << "  ";
            }
        }

        formatted << content;

        os << formatted.str() << '\n';
    }
}; // class Logger

// ---- StatusCode helpers (now safe because stiles_types.hpp is included) ----
inline const char* toString(StatusCode status) {
    switch (status) {
        case StatusCode::Success:           return "Operation completed successfully";
        case StatusCode::Failure:           return "General failure";
        case StatusCode::NotInitialized:    return "Library has not been initialized";
        case StatusCode::Reinitialized:     return "Library was reinitialized incorrectly";
        case StatusCode::NotSupported:      return "The requested operation is not supported";
        case StatusCode::IllegalValue:      return "An illegal value was passed to a function";
        case StatusCode::NotFound:          return "The requested item could not be found";
        case StatusCode::OutOfResources:    return "Insufficient resources (e.g., memory) to proceed";
        case StatusCode::InternalLimit:     return "An internal library limit was reached";
        case StatusCode::Unallocated:       return "A resource was not allocated";
        case StatusCode::FilesystemError:   return "A filesystem-related error occurred";
        case StatusCode::Unexpected:        return "An unexpected internal error occurred";
        case StatusCode::SequenceFlushed:   return "A sequence of operations was flushed prematurely";
        case StatusCode::ExecutionFailed:   return "A computation failed due to a numerical issue";
        default:                            return "Unknown status code";
    }
}

inline std::ostream& operator<<(std::ostream& os, StatusCode status) {
    os << toString(status);
    return os;
}

namespace detail {
    inline void log_check_success(const char* fn_call_str) {
    #if STILES_LOG_LEVEL >= STILES_LOG_LEVEL_INFO
        if (static_cast<int>(sTiles::getLevel()) >= static_cast<int>(sTiles::Level::Info)) {
            sTiles::Logger::info("Success: ", fn_call_str);
        }
    #endif
    }

    static const char* levelToString(sTiles::Level level) {
        switch (level) {
            case sTiles::Level::TimingOnly: return "Timing Only (-2)";
            case sTiles::Level::None:  return "None (-1)";
            case sTiles::Level::Time:  return "Time (0)";
            case sTiles::Level::Info:  return "Info (1)";
            case sTiles::Level::Debug: return "Debug (2)";
            case sTiles::Level::Trace: return "Trace (3)";
            default:                   return "Unknown";
        }
    }
} // namespace detail

} // namespace sTiles

// ----------------------------------------------------------------------------
//  Updated STILES_CHECK macro
// ----------------------------------------------------------------------------
#define STILES_CHECK(fn_call)                                                \
    do {                                                                     \
        int _status = (fn_call);                                             \
        if (_status != 0) {                                                  \
            sTiles::Logger::error("Function call failed: " #fn_call,         \
                                  " with status code ", _status);            \
            return _status;                                                  \
        }                                                                    \
        sTiles::detail::log_check_success(#fn_call);                         \
    } while (0)

#endif // STILES_LOGGER_HPP
