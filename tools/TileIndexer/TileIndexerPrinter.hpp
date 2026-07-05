/**
 * @file    TileIndexerPrinter.hpp
 * @brief   Formatting helpers for TileIndexer test/benchmark tables and
 *          human-readable byte sizes.
 *
 * @project sTiles (Sparse Tiles Library)
 * @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @version 3.0.0
 * @date 1 1 2026
 * @license Proprietary
 *
 * @note This file is part of the sTiles library, a proprietary software package.
 *       Redistribution or modification without prior permission is prohibited.
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
 *
 * @license
 * This software is proprietary and confidential. Unauthorized copying, distribution, or modification
 * of this software, via any medium, is strictly prohibited. Permission is granted to use the software
 * in binary form for non-commercial purposes only, provided that this copyright notice and permission
 * notice are included in all copies or substantial portions of the software.
 *
 * DISCLAIMER:
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cmath>

#include "TileIndexer.hpp"

namespace tilecounter_printer {

// Column widths for the test3 table
struct Widths {
    int method  = 28;
    int kind    = 10;
    int threads = 14;
    int active  = 14;
    int time    = 14;
    int speedup = 10;
    int status  = 10;
};

inline int total_width(const Widths& w) {
    return w.method + w.kind + w.threads + w.active + w.time + w.speedup + w.status;
}

/*
Function: format_bytes
Purpose:  Render a byte count as a human-readable string with units.
*/
inline std::string format_bytes(std::size_t bytes) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (unit < 4 && value >= 1024.0) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream oss;
    const int precision = (value < 10.0) ? 2 : 1;
    oss << std::fixed << std::setprecision(precision) << value << ' ' << units[unit];
    return oss.str();
}

/*
Function: print_test3_header
Purpose:  Print a header banner for the thread sweep benchmark.
*/
inline void print_test3_header(const std::string& matrix,
                               int n, int nnz, int tile_size, int num_tiles,
                               unsigned hw_threads)
{
    std::cout << "\n[TEST3] Thread sweep comparison\n";
    std::cout << "Matrix='\"" << matrix << "\"', n=" << n
              << ", nnz=" << nnz
              << ", tile_size=" << tile_size
              << ", num_tiles=" << num_tiles
              << ", hw_threads=" << hw_threads
              << "\n";
}

/*
Function: format_threads
Purpose:  Format requested/used thread counts in a compact form.
*/
inline std::string format_threads(int requested, int used) {
    if (requested <= 0 && used <= 0) return "-";
    if (requested == used) return std::to_string(std::max(requested, 0));
    return std::to_string(std::max(requested, 0)) + "->" + std::to_string(std::max(used, 0));
}

/*
Function: print_table_header
Purpose:  Print a generic table header with configured column widths.
*/
inline void print_table_header(const Widths& w) {
    std::cout << std::left  << std::setw(w.method)  << "Method"
              << std::left  << std::setw(w.kind)    << "Kind"
              << std::right << std::setw(w.threads) << "Threads"
              << std::right << std::setw(w.active)  << "Active"
              << std::right << std::setw(w.time)    << "Time (ms)"
              << std::right << std::setw(w.speedup) << "Speedup"
              << std::right << std::setw(w.status)  << "Status"
              << "\n";
    std::cout << std::string(total_width(w), '-') << "\n";
}

/*
Function: print_table_row
Purpose:  Print a single row for the comparison table.
*/
inline void print_table_row(const Widths& w,
                            const std::string& method_name,
                            const std::string& kind,
                            const std::string& threads,
                            long long active,
                            double time_ms,
                            double speedup,
                            const char* status)
{
    auto trim = [](const std::string& s) -> std::string {
        if (s.empty()) return s;
        std::size_t start = s.find_first_not_of(' ');
        if (start == std::string::npos) return "";
        std::size_t end = s.find_last_not_of(' ');
        return s.substr(start, end - start + 1);
    };
    const std::string method_cell = trim(method_name);
    const std::string kind_cell   = trim(kind);
    const std::string thread_cell = trim(threads);

    std::cout << std::left  << std::setw(w.method)  << method_cell
              << std::left  << std::setw(w.kind)    << kind_cell
              << std::right << std::setw(w.threads) << thread_cell
              << std::right << std::setw(w.active)  << active
              << std::right << std::setw(w.time)    << std::fixed << std::setprecision(3) << time_ms
              << std::right << std::setw(w.speedup) << std::fixed << std::setprecision(2) << speedup
              << std::right << std::setw(w.status)  << status
              << "\n";
}

inline const char* method_kind(tilecounter::Method m) {
    switch (m) {
        case tilecounter::Method::Auto:
            return "Auto";
        case tilecounter::Method::CharMask:
        case tilecounter::Method::BoolMask:
        case tilecounter::Method::BitsetMask:
        case tilecounter::Method::LazyLookUp:
            return "Dense";
        case tilecounter::Method::TiledBoolMask:
        case tilecounter::Method::TiledBitsetMask:
        case tilecounter::Method::PagedMask:
        case tilecounter::Method::HashSet:
        case tilecounter::Method::SortUnique:
            return "Sparse";
    }
    return "?";
}

struct FillSummaryRow {
    int requested_threads = 1;
    int used_threads = 1;
    int filled = 0;
    double ms = 0.0;
    double speedup = 1.0;
    bool ok = true;
};

struct FillSummary {
    std::string label;
    tilecounter::Method method = tilecounter::Method::CharMask;
    std::string kind;
    std::vector<FillSummaryRow> rows;
};

struct GraphSummaryRow {
    int requested_threads = 1;
    int used_threads = 1;
    int edges = 0;
    double ms = 0.0;
    double speedup = 1.0;
    bool ok = true;
};

struct GraphSummary {
    std::string label;
    tilecounter::Method method = tilecounter::Method::CharMask;
    std::string kind;
    std::vector<GraphSummaryRow> rows;
};

/*
Function: print_fill_table
Purpose:  Render a timing table for fill passes across configurations.
*/
inline void print_fill_table(const Widths& w, const std::vector<FillSummary>& summaries) {
    if (summaries.empty()) return;
    std::cout << "\nFill Timing\n";
    std::cout << std::left  << std::setw(w.method)  << "Method"
              << std::left  << std::setw(w.kind)    << "Kind"
              << std::right << std::setw(w.threads) << "Threads"
              << std::right << std::setw(w.active)  << "Filled"
              << std::right << std::setw(w.time)    << "Time (ms)"
              << std::right << std::setw(w.speedup) << "Speedup"
              << std::right << std::setw(w.status)  << "Status"
              << "\n";
    std::cout << std::string(total_width(w), '-') << "\n";
    for (const auto& summary : summaries) {
        bool first = true;
        for (const auto& row : summary.rows) {
            print_table_row(w,
                first ? summary.label : "",
                first ? summary.kind : "",
                format_threads(row.requested_threads, row.used_threads),
                row.filled,
                row.ms,
                row.speedup,
                row.ok ? "OK" : "DIFF");
            first = false;
        }
    }
}

/*
Function: print_graph_table
Purpose:  Render a timing table for graph builds across configurations.
*/
inline void print_graph_table(const Widths& w, const std::vector<GraphSummary>& summaries) {
    if (summaries.empty()) return;
    std::cout << "\nGraph Build Timing\n";
    std::cout << std::left  << std::setw(w.method)  << "Method"
              << std::left  << std::setw(w.kind)    << "Kind"
              << std::right << std::setw(w.threads) << "Threads"
              << std::right << std::setw(w.active)  << "Edges"
              << std::right << std::setw(w.time)    << "Time (ms)"
              << std::right << std::setw(w.speedup) << "Speedup"
              << std::right << std::setw(w.status)  << "Status"
              << "\n";
    std::cout << std::string(total_width(w), '-') << "\n";
    for (const auto& summary : summaries) {
        bool first = true;
        for (const auto& row : summary.rows) {
            print_table_row(w,
                first ? summary.label : "",
                first ? summary.kind : "",
                format_threads(row.requested_threads, row.used_threads),
                row.edges,
                row.ms,
                row.speedup,
                row.ok ? "OK" : "DIFF");
            first = false;
        }
    }
}

} // namespace tilecounter_printer
