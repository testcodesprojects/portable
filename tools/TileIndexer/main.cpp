/**
 * @file    main.cpp
 * @brief   Test/benchmark harness for TileIndexer: loads matrices, exercises
 *          counting/fill strategies, and prints timing/summary tables.
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

#include "TileIndexer.hpp"
#include "TileIndexerCounter.hpp"
#include "MatrixIO.hpp"
#include "TileIndexerFill.hpp"
#include "TileIndexerGraphBuilder.hpp"
#include "TileIndexerMapper.hpp"
#include "TileIndexerIsActive.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <array>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "TileIndexerPrinter.hpp"
#include "TileIndexerAutoStrict.hpp"
#include "TileIndexerMemoryUtils.hpp"
#include "../algorithms/chol_algorithms.hpp"

namespace {

using tilecounter_utils::bind_is_active;

} // namespace


#ifndef TILEINDEXER_LIBRARY_ONLY

// New minimal main: read inla_Q_group1_call1.bin, tile_size=2, fill, check tile[3,4], print all active tiles with their mapped index
// (no longer need off-diagonal helper; we scan (i,j) directly)

/**
 * @brief  Load a small matrix, count active tiles, run a single fill pass,
 *         and print selected results including the mapping of active tiles.
 * @return 0 on success, non-zero on failure.
 */
int test1() {
    std::filesystem::path path = "matrices/inla_Q_group1_call1_set10.bin";
    const int tile_size = 3;

    int n = 0, nnz = 0; int* rows = nullptr; int* cols = nullptr;
    // Resolve via matrices/ if needed
    auto resolve_path = [](std::filesystem::path p) {
        if (std::filesystem::exists(p)) return p;
        std::filesystem::path alt = std::filesystem::path("matrices") / p;
        if (std::filesystem::exists(alt)) return alt;
        return p;
    };
    path = resolve_path(path);
    if (!TileIndexer::loadMatrixIndices(path, n, nnz, rows, cols)) {
        std::cerr << "Error: failed to load '" << path << "'\n";
        return 1;
    }
    const int num_tiles = (n + tile_size - 1) / tile_size;

    TileIndexer::State state;
    //const auto method = TileIndexer::Method::BitsetMask;
    const auto method = TileIndexer::Method::TiledBoolMask;

    
    const int active = TileIndexer::countActiveTiles(rows, cols, nnz, n, tile_size, method, &state, 1);
    const int fill_threads = 1; // serial fill for this test harness
    const int filled = TileIndexer::FillTiles(state, method, num_tiles, active, fill_threads);

    tilecounter_utils::bind_is_active(state, method);

    std::cout << "Matrix: '" << path << "'\n"
              << "  n=" << n << ", nnz=" << nnz << ", tile_size=" << tile_size
              << ", num_tiles=" << num_tiles << "\n"
              << "  active(before fill)=" << active << ", after fill=" << filled << "\n\n";

    // Check tile[3,4]
    int ti = 3, tj = 4;
    bool act = state.isActive(ti, tj, num_tiles);
    std::cout << "tile[3,4] active? " << (act ? "yes" : "no") << "\n\n";

    // Print full element-wise matrix (n x n) as 0/1 from the input structure
    // {
    //     std::vector<unsigned char> full(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0);
    //     for (int k = 0; k < nnz; ++k) {
    //         int r = rows[k], c = cols[k];
    //         if (r >= 0 && r < n && c >= 0 && c < n) {
    //             full[static_cast<std::size_t>(r) * static_cast<std::size_t>(n) + static_cast<std::size_t>(c)] = 1;
    //             // If you want to visualize symmetric structure, also set the transpose:
    //             // full[static_cast<std::size_t>(c) * static_cast<std::size_t>(n) + static_cast<std::size_t>(r)] = 1;
    //         }
    //     }
    //     std::cout << "Full matrix (" << n << "x" << n << ") as 0/1:\n";
    //     for (int i = 0; i < n; ++i) {
    //         for (int j = 0; j < n; ++j) {
    //             std::cout << int(full[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)]);
    //             if (j + 1 < n) std::cout << ' ';
    //         }
    //         std::cout << '\n';
    //     }
    //     std::cout << '\n';
    // }

    // Print all active tiles in the same order the rank mapper builds k: ascending packed index u
    std::cout << "(i,j) -> k (CSC order)\n";
    int k_counter = 0;
    for (int j = 0; j < num_tiles; ++j) {
        for (int i = 0; i <= j; ++i) {
            if (!state.isActive(i, j, num_tiles)) continue;
            std::cout << "(" << i << "," << j << ") -> " << k_counter << "\n";
            ++k_counter;
        }
    }

    state.reset();
    sTiles::TileIndexerMemoryManager::freeAll();

    TileIndexer::freeMatrixIndices(rows, cols);
    return 0;
}


/**
 * @brief  Sweep matrices and thread counts across methods, reporting timings
 *         and speedups in a compact table.
 * @return 0 on success, non-zero on failure.
 */
int test2() {
    // Fixed settings (no CLI parsing for now)
    int tile_size = 40;
    // Discover all .bin matrices under ./matrices for test3
    std::vector<std::string> files;
    {
        const std::filesystem::path mdir = "matrices";
        if (std::filesystem::exists(mdir) && std::filesystem::is_directory(mdir)) {
            for (const auto& ent : std::filesystem::directory_iterator(mdir)) {
                if (!ent.is_regular_file()) continue;
                const auto& p = ent.path();
                if (p.extension() == ".bin") files.push_back(p.string());
            }
            std::sort(files.begin(), files.end());
        }
        // Fallback to a couple of known cases if directory is missing/empty
        if (files.empty()) {
            files = {"matrices/TCcase2_num_1_dim_81438.bin",
                     "matrices/TCcase2_num_2_dim_1628760.bin"};
        }
    }
    std::vector<int> thread_sweep; // will be populated from hardware_concurrency

    unsigned hc = std::thread::hardware_concurrency();
    if (thread_sweep.empty()) {
        unsigned max_t = (hc == 0 ? 16u : std::max<unsigned>(hc, 16u));
        std::vector<unsigned> seeds;
        seeds.push_back(1u);
        unsigned t = 2;
        while (t <= max_t) {
            seeds.push_back(t);
            t <<= 1;
        }
        for (auto v : seeds) thread_sweep.push_back(static_cast<int>(v));
        if (hc > 0) thread_sweep.push_back(static_cast<int>(hc));
    }
    std::sort(thread_sweep.begin(), thread_sweep.end());
    thread_sweep.erase(std::remove_if(thread_sweep.begin(), thread_sweep.end(), [](int x){ return x < 1; }), thread_sweep.end());
    thread_sweep.erase(std::unique(thread_sweep.begin(), thread_sweep.end()), thread_sweep.end());

    const std::vector<TileIndexer::Method> methods = {
        TileIndexer::Method::CharMask,
        TileIndexer::Method::BoolMask,
        TileIndexer::Method::BitsetMask,
        TileIndexer::Method::TiledBoolMask,
        TileIndexer::Method::TiledBitsetMask,
        TileIndexer::Method::PagedMask,
        //TileIndexer::Method::HashSet,
        //TileIndexer::Method::SortUnique
    };

    for (const auto& fname : files) {
        int n = 0, nnz = 0; int* rows = nullptr; int* cols = nullptr;
        // try as given; if not found, try matrices/ prefix
        auto resolve_file = [&](const std::string& s) {
            if (std::filesystem::exists(s)) return std::filesystem::path(s);
            std::filesystem::path alt = std::filesystem::path("matrices") / s;
            if (std::filesystem::exists(alt)) return alt;
            return std::filesystem::path(s);
        };
        const auto fpath = resolve_file(fname);
        if (!TileIndexer::loadMatrixIndices(fpath.string(), n, nnz, rows, cols)) {
            std::cerr << "Error: failed to load '" << fpath << "'\n";
            continue;
        }
        const int num_tiles = (n + tile_size - 1) / tile_size;

        tilecounter_printer::Widths W;
        std::vector<tilecounter_printer::FillSummary> fill_summaries;
        fill_summaries.reserve(methods.size() + 2);
        std::vector<tilecounter_printer::GraphSummary> graph_summaries;
        graph_summaries.reserve(methods.size() + 2);
        tilecounter_printer::print_test3_header(fname, n, nnz, tile_size, num_tiles, (hc ? hc : 0));
        tilecounter_printer::print_table_header(W);

        // Plan-based options (single-phase and two-phase) using auto-selected settings,
        // evaluated across the common thread sweep (requested vs. auto-used).
        {
            // Strict single-phase plan
            auto P1 = tilecounter::make_strict_single_phase(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(nnz), tile_size);
            tilecounter_printer::FillSummary s1;
            s1.label = std::string("StrictSinglePhase (") + TileIndexer::to_string(P1.fill_method) + ")";
            s1.method = P1.fill_method;
            s1.kind = tilecounter_printer::method_kind(P1.fill_method);
            tilecounter_printer::GraphSummary g1;
            g1.label = s1.label;
            g1.method = P1.fill_method;
            g1.kind = s1.kind;

            bool first_row = true;
            double base_ms = 0.0; bool base_set = false;
            double base_fill_ms = 0.0; bool base_fill_set = false;
            TileIndexer::State graph_state_s1;
            bool graph_state_set = false;

            const int count_cap_s1 = std::max(1, P1.count_threads);
            const int fill_cap_s1  = std::max(1, P1.fill_threads);

            for (int tc : thread_sweep) {
                const int requested = tc;
                const int used_t = std::min(requested, count_cap_s1);

                TileIndexer::State st1;
                double t0 = omp_get_wtime();
                int active1 = TileIndexer::countActiveTiles(rows, cols, nnz, n, tile_size, P1.count_method, &st1, used_t);
                double t1 = omp_get_wtime();
                double ms = (t1 - t0) * 1000.0;

                double sp = 1.0;
                if (!base_set) { base_ms = ms; base_set = true; }
                else if (base_ms > 0.0 && ms > 0.0) sp = base_ms / ms;

                tilecounter_printer::print_table_row(W,
                    first_row ? "StrictSinglePhase" : "",
                    first_row ? tilecounter_printer::method_kind(P1.count_method) : "",
                    tilecounter_printer::format_threads(requested, used_t),
                    active1,
                    ms,
                    sp,
                    "OK");
                first_row = false;

                const int used_tf = std::min(requested, fill_cap_s1);
                TileIndexer::State fill_state = st1;
                double ft0 = omp_get_wtime();
                int filled = TileIndexer::FillTiles(fill_state, P1.fill_method, num_tiles, active1, used_tf);
                double ft1 = omp_get_wtime();
                double fms = (ft1 - ft0) * 1000.0;

                double fsp = 1.0;
                if (!base_fill_set) { base_fill_ms = fms; base_fill_set = true; }
                else if (base_fill_ms > 0.0 && fms > 0.0) fsp = base_fill_ms / fms;

                s1.rows.push_back(tilecounter_printer::FillSummaryRow{requested, used_tf, filled, fms, fsp, true});

                if (!graph_state_set) {
                    graph_state_s1 = fill_state;
                    graph_state_set = true;
                }
            }

            if (graph_state_set) {
                std::vector<int> g_offsets, g_edges;
                double gt0 = omp_get_wtime();
            TileIndexer::build_graph(graph_state_s1, P1.fill_method, num_tiles, g_offsets, g_edges);
            std::vector<int> g_offsets_lower, g_edges_lower;
            TileIndexer::build_graph_lower(graph_state_s1, P1.fill_method, num_tiles, g_offsets_lower, g_edges_lower);
            double gt1 = omp_get_wtime();
            double gms = (gt1 - gt0) * 1000.0;
            g1.rows.push_back(tilecounter_printer::GraphSummaryRow{fill_cap_s1, fill_cap_s1, static_cast<int>(g_edges.size()), gms, 1.0, true});
            if (!g_offsets.empty() && !g_offsets_lower.empty()) {
                const auto upper_deg = g_offsets.back();
                const auto lower_deg = g_offsets_lower.back();
                if (upper_deg != lower_deg) {
                    std::cout << "  [Debug] S1 upper/lower mismatch: " << upper_deg << " vs " << lower_deg << "\n";
                }
            }
            graph_summaries.push_back(std::move(g1));
        }
            fill_summaries.push_back(std::move(s1));

            // Strict two-phase plan
            auto P2 = tilecounter::make_strict_two_phase(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(nnz), tile_size);
            tilecounter_printer::FillSummary s2;
            s2.label = std::string("StrictTwoPhase (") + TileIndexer::to_string(P2.fill_method) + ")";
            s2.method = P2.fill_method;
            s2.kind = tilecounter_printer::method_kind(P2.fill_method);
            tilecounter_printer::GraphSummary g2;
            g2.label = s2.label;
            g2.method = P2.fill_method;
            g2.kind = s2.kind;

            bool first_row2 = true;
            double base_ms2 = 0.0; bool base_set2 = false;
            double base_fill_ms2 = 0.0; bool base_fill_set2 = false;
            TileIndexer::State base_state_fill;
            const int fill_count_threads = std::max(1, tilecounter::strict_count_threads(P2.fill_method, num_tiles));
            int active_fill = TileIndexer::countActiveTiles(rows, cols, nnz, n, tile_size, P2.fill_method, &base_state_fill, fill_count_threads);
            TileIndexer::State graph_state_s2;
            bool graph_state_set2 = false;

            const int count_cap_s2 = std::max(1, P2.count_threads);
            const int fill_cap_s2  = std::max(1, P2.fill_threads);

            for (int tc : thread_sweep) {
                const int requested = tc;
                const int used_t = std::min(requested, count_cap_s2);

                TileIndexer::State st2;
                double ct0 = omp_get_wtime();
                int active2 = TileIndexer::countActiveTiles(rows, cols, nnz, n, tile_size, P2.count_method, &st2, used_t);
                double ct1 = omp_get_wtime();
                double cms = (ct1 - ct0) * 1000.0;

                double sp = 1.0;
                if (!base_set2) { base_ms2 = cms; base_set2 = true; }
                else if (base_ms2 > 0.0 && cms > 0.0) sp = base_ms2 / cms;

                tilecounter_printer::print_table_row(W,
                    first_row2 ? "StrictTwoPhase" : "",
                    first_row2 ? tilecounter_printer::method_kind(P2.count_method) : "",
                    tilecounter_printer::format_threads(requested, used_t),
                    active2,
                    cms,
                    sp,
                    "OK");
                first_row2 = false;

                const int used_tf = std::min(requested, fill_cap_s2);
                TileIndexer::State fill_state2 = base_state_fill;
                double ft0_plan2 = omp_get_wtime();
                int filled2 = TileIndexer::FillTiles(fill_state2, P2.fill_method, num_tiles, active_fill, used_tf);
                double ft1_plan2 = omp_get_wtime();
                double fms2 = (ft1_plan2 - ft0_plan2) * 1000.0;

                double fsp = 1.0;
                if (!base_fill_set2) { base_fill_ms2 = fms2; base_fill_set2 = true; }
                else if (base_fill_ms2 > 0.0 && fms2 > 0.0) fsp = base_fill_ms2 / fms2;

                s2.rows.push_back(tilecounter_printer::FillSummaryRow{requested, used_tf, filled2, fms2, fsp, true});

                if (!graph_state_set2) {
                    graph_state_s2 = fill_state2;
                    graph_state_set2 = true;
                }
            }

            if (graph_state_set2) {
                std::vector<int> g2_offsets, g2_edges;
                double gt0_plan2 = omp_get_wtime();
                TileIndexer::build_graph(graph_state_s2, P2.fill_method, num_tiles, g2_offsets, g2_edges);
                std::vector<int> g2_offsets_lower, g2_edges_lower;
                TileIndexer::build_graph_lower(graph_state_s2, P2.fill_method, num_tiles, g2_offsets_lower, g2_edges_lower);
                double gt1_plan2 = omp_get_wtime();
                double gms2 = (gt1_plan2 - gt0_plan2) * 1000.0;
                g2.rows.push_back(tilecounter_printer::GraphSummaryRow{fill_cap_s2, fill_cap_s2, static_cast<int>(g2_edges.size()), gms2, 1.0, true});
                if (!g2_offsets.empty() && !g2_offsets_lower.empty()) {
                    const auto upper_deg = g2_offsets.back();
                    const auto lower_deg = g2_offsets_lower.back();
                    if (upper_deg != lower_deg) {
                        std::cout << "  [Debug] S2 upper/lower mismatch: " << upper_deg << " vs " << lower_deg << "\n";
                    }
                }
                graph_summaries.push_back(std::move(g2));
            }
            fill_summaries.push_back(std::move(s2));

            auto P3 = tilecounter::make_universal_plan(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(nnz), tile_size);
            tilecounter_printer::FillSummary s3;
            s3.label = std::string("UniversalAuto (") + TileIndexer::to_string(P3.fill_method) + ")";
            s3.method = P3.fill_method;
            s3.kind = tilecounter_printer::method_kind(P3.fill_method);
            tilecounter_printer::GraphSummary g3;
            g3.label = s3.label;
            g3.method = P3.fill_method;
            g3.kind = s3.kind;

            if (!P3.reason.empty()) {
                std::cout << "  " << s3.label << " rationale: " << P3.reason << "\n";
            }

            bool first_row3 = true;
            double base_ms3 = 0.0; bool base_set3 = false;
            double base_fill_ms3 = 0.0; bool base_fill_set3 = false;
            TileIndexer::State graph_state_s3;
            bool graph_state_set3 = false;

            const int count_cap_s3 = std::max(1, P3.count_threads);
            const int fill_cap_s3  = std::max(1, P3.fill_threads);

            for (int tc : thread_sweep) {
                const int requested = tc;
                const int used_t = std::min(requested, count_cap_s3);

                TileIndexer::State st3;
                double ct0 = omp_get_wtime();
                int active3 = TileIndexer::countActiveTiles(rows, cols, nnz, n, tile_size, P3.count_method, &st3, used_t);
                double ct1 = omp_get_wtime();
                double cms3 = (ct1 - ct0) * 1000.0;

                double sp = 1.0;
                if (!base_set3) { base_ms3 = cms3; base_set3 = true; }
                else if (base_ms3 > 0.0 && cms3 > 0.0) sp = base_ms3 / cms3;

                tilecounter_printer::print_table_row(W,
                    first_row3 ? "UniversalAuto" : "",
                    first_row3 ? tilecounter_printer::method_kind(P3.count_method) : "",
                    tilecounter_printer::format_threads(requested, used_t),
                    active3,
                    cms3,
                    sp,
                    "OK");
                first_row3 = false;

                const int used_tf = std::min(requested, fill_cap_s3);
                TileIndexer::State fill_state3 = st3;
                double ft0_plan3 = omp_get_wtime();
                int filled3 = TileIndexer::FillTiles(fill_state3, P3.fill_method, num_tiles, active3, used_tf);
                double ft1_plan3 = omp_get_wtime();
                double fms3 = (ft1_plan3 - ft0_plan3) * 1000.0;

                double fsp3 = 1.0;
                if (!base_fill_set3) { base_fill_ms3 = fms3; base_fill_set3 = true; }
                else if (base_fill_ms3 > 0.0 && fms3 > 0.0) fsp3 = base_fill_ms3 / fms3;

                s3.rows.push_back(tilecounter_printer::FillSummaryRow{requested, used_tf, filled3, fms3, fsp3, true});

                if (!graph_state_set3) {
                    graph_state_s3 = fill_state3;
                    graph_state_set3 = true;
                }
            }

            if (graph_state_set3) {
                std::vector<int> g3_offsets, g3_edges;
                double gt0_plan3 = omp_get_wtime();
                TileIndexer::build_graph(graph_state_s3, P3.fill_method, num_tiles, g3_offsets, g3_edges);
                std::vector<int> g3_offsets_lower, g3_edges_lower;
                TileIndexer::build_graph_lower(graph_state_s3, P3.fill_method, num_tiles, g3_offsets_lower, g3_edges_lower);
                double gt1_plan3 = omp_get_wtime();
                double gms3 = (gt1_plan3 - gt0_plan3) * 1000.0;
                g3.rows.push_back(tilecounter_printer::GraphSummaryRow{fill_cap_s3, fill_cap_s3, static_cast<int>(g3_edges.size()), gms3, 1.0, true});
                if (!g3_offsets.empty() && !g3_offsets_lower.empty()) {
                    const auto upper_deg = g3_offsets.back();
                    const auto lower_deg = g3_offsets_lower.back();
                    if (upper_deg != lower_deg) {
                        std::cout << "  [Debug] S3 upper/lower mismatch: " << upper_deg << " vs " << lower_deg << "\n";
                    }
                }
                graph_summaries.push_back(std::move(g3));
            }
            fill_summaries.push_back(std::move(s3));
        }

        // Explicit methods

        for (auto m : methods) {
            TileIndexer::State base_state;
            double t0 = omp_get_wtime();
            int base_active = TileIndexer::countActiveTiles(rows, cols, nnz, n, tile_size, m, &base_state, 1);
            double t1 = omp_get_wtime();
            double base_ms = (t1 - t0) * 1000.0;
            tilecounter_printer::print_table_row(W, TileIndexer::to_string(m), tilecounter_printer::method_kind(m), tilecounter_printer::format_threads(1, 1), base_active, base_ms, 1.0, "OK");

            for (int tc : thread_sweep) {
                if (tc == 1) continue;
                TileIndexer::State stp;

                t0 = omp_get_wtime();
                const int active = TileIndexer::countActiveTiles(rows, cols, nnz, n, tile_size, m, &stp, tc);
                t1 = omp_get_wtime();
                const double ms = (t1 - t0) * 1000.0;
                const bool same = (active == base_active);
                const double sp = base_ms > 0 ? (base_ms / ms) : 0.0;
                tilecounter_printer::print_table_row(W, "", "", tilecounter_printer::format_threads(tc, tc), active, ms, sp, same ? "OK" : "DIFF");
            }

            std::vector<int> fill_thread_list = thread_sweep;
            if (fill_thread_list.empty()) fill_thread_list.push_back(1);
            if (fill_thread_list.front() != 1)
                fill_thread_list.insert(fill_thread_list.begin(), 1);
            fill_thread_list.erase(std::unique(fill_thread_list.begin(), fill_thread_list.end()), fill_thread_list.end());

            tilecounter_printer::FillSummary summary;
            summary.label = TileIndexer::to_string(m);
            summary.method = m;
            summary.kind = tilecounter_printer::method_kind(m);

            tilecounter_printer::GraphSummary gsummary;
            gsummary.label = summary.label;
            gsummary.method = m;
            gsummary.kind = summary.kind;
            bool graph_recorded = false;
            double baseline_ms = 0.0;
            int baseline_filled = 0;
            bool baseline_set = false;

            for (int tc : fill_thread_list) {
                TileIndexer::State fill_state = base_state;
                double ft0 = omp_get_wtime();
                int filled = TileIndexer::FillTiles(fill_state, m, num_tiles, base_active, tc);
                double ft1 = omp_get_wtime();
                double ms = (ft1 - ft0) * 1000.0;

                if (!baseline_set) {
                    baseline_set = true;
                    baseline_ms = ms;
                    baseline_filled = filled;
                }

                if (!graph_recorded) {
                    std::vector<int> g_offsets, g_edges;
                    double gt0 = omp_get_wtime();
                    TileIndexer::build_graph(fill_state, m, num_tiles, g_offsets, g_edges);
                    double gt1 = omp_get_wtime();
                    double graph_ms = (gt1 - gt0) * 1000.0;
                    gsummary.rows.push_back(tilecounter_printer::GraphSummaryRow{tc, 1, static_cast<int>(g_edges.size()), graph_ms, 1.0, true});
                    graph_recorded = true;
                }

                double sp = (tc == fill_thread_list.front() || ms <= 0.0 || baseline_ms <= 0.0)
                                ? 1.0
                                : (baseline_ms / ms);
                summary.rows.push_back(tilecounter_printer::FillSummaryRow{tc, tc, filled, ms, sp, filled == baseline_filled});
            }

            fill_summaries.push_back(std::move(summary));
            if (graph_recorded) {
                graph_summaries.push_back(std::move(gsummary));
            }
        }

    tilecounter_printer::print_fill_table(W, fill_summaries);
    tilecounter_printer::print_graph_table(W, graph_summaries);

    sTiles::TileIndexerMemoryManager::freeAll();
        TileIndexer::freeMatrixIndices(rows, cols);
    }
    return 0;
}

/**
 * @brief  Build adjacency graphs, then run a Green-tree phase 0 routine across
 *         simulated ranks to validate neighborhood queries.
 * @return 0 on success, non-zero on failure.
 */
int test3() {
    
    std::filesystem::path path = "matrices/TCcase2_num_1_dim_81438.bin";
    const int tile_size = 37;
    int n = 0, nnz = 0;
    int* rows = nullptr;
    int* cols = nullptr;

    auto resolve_path = [](std::filesystem::path p) {
        if (std::filesystem::exists(p)) return p;
        std::filesystem::path alt = std::filesystem::path("matrices") / p;
        if (std::filesystem::exists(alt)) return alt;
        return p;
    };
    path = resolve_path(path);
    if (!TileIndexer::loadMatrixIndices(path, n, nnz, rows, cols)) {
        std::cerr << "Error: failed to load '" << path << "'\n";
        return 1;
    }

    const int num_tiles = (n + tile_size - 1) / tile_size;

    struct TestOption {
        std::string label;
        bool use_plan;
        TileIndexer::Method method;
        tilecounter::Plan plan;
    };

    const auto plan_s1 = tilecounter::make_strict_single_phase(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(nnz), tile_size);
    const auto plan_s2 = tilecounter::make_strict_two_phase(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(nnz), tile_size);
    const auto plan_s3 = tilecounter::make_universal_plan(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(nnz), tile_size);
    std::vector<TestOption> options;
    // options.push_back({
    //     "StrictSinglePhase",
    //     true,
    //     plan_s1.fill_method,
    //     plan_s1
    // });
    // options.push_back({
    //     "StrictTwoPhase",
    //     true,
    //     plan_s2.fill_method,
    //     plan_s2
    // });
    // options.push_back({
    //     "UniversalAuto",
    //     true,
    //     plan_s3.fill_method,
    //     plan_s3
    // });
    options.push_back({
        "BitsetMask",
        false,
        TileIndexer::Method::BitsetMask,
        {}
    });

    std::cout << "Test4: chol_green_tree_phase0 on '" << path << "' (n=" << n
              << ", nnz=" << nnz << ", tile_size=" << tile_size
              << ", num_tiles=" << num_tiles << ")\n";

    const int worldsize = 10;
    for (const auto& opt : options) {
        TileIndexer::State state;
        int active = 0;
        int filled = 0;
        TileIndexer::Method run_method = opt.method;

        if (opt.use_plan) {
            active = tilecounter::countActiveTiles(rows, cols, nnz, n, tile_size,
                                                   opt.plan.count_method, &state, opt.plan.count_threads);
            filled = TileIndexer::FillTiles(state, opt.plan.fill_method, num_tiles, active, opt.plan.fill_threads);
            run_method = opt.plan.fill_method;
            tilecounter_utils::bind_is_active(state, run_method);
            std::cout << "\nOption: " << opt.label
                      << " (count=" << TileIndexer::to_string(opt.plan.count_method)
                      << " @" << opt.plan.count_threads << "t, fill="
                      << TileIndexer::to_string(opt.plan.fill_method)
                      << " @" << opt.plan.fill_threads << "t)";
            if (!opt.plan.reason.empty()) {
                std::cout << "\n    reason: " << opt.plan.reason;
            }
        } else {
            active = tilecounter::countActiveTiles(rows, cols, nnz, n, tile_size,
                                                   opt.method, &state, 1);
            filled = TileIndexer::FillTiles(state, opt.method, num_tiles, active, 1);
            tilecounter_utils::bind_is_active(state, opt.method);
            std::cout << "\nOption: " << opt.label
                      << " (method=" << TileIndexer::to_string(opt.method) << ")";
        }

        std::cout << " active=" << active
                  << " filled=" << filled << "\n";

        const auto usage = sTiles::TileIndexerMemoryManager::snapshotUsage();
        std::cout << "    TileIndexer tracked memory: " << tilecounter_printer::format_bytes(usage.total) << "\n";
        for (std::size_t g = 0; g < tilecounter_utils::kGroupInfo.size(); ++g) {
            const auto bytes = usage.groups[g];
            if (bytes == 0) continue;
            std::cout << "      - " << std::setw(16) << tilecounter_utils::kGroupInfo[g].second
                      << tilecounter_printer::format_bytes(bytes) << "\n";
        }

        // Prebuild CSR graphs once to avoid rebuilding per rank
        tilecounter::build_graphs_up_lo(state, num_tiles, /*include_self=*/true);
        // std::vector<std::array<int,4>> all_tasks; // {rank,n,k,m}
        // all_tasks.reserve(1024);
        for (int rank = 0; rank < worldsize; ++rank) {
            int counter = sTiles::algorithms::chol_green_tree_phase0(state, rank, worldsize, num_tiles, nullptr);
            std::cout << "  Rank " << rank << " -> counter=" << counter << "\n";
        }
        // Skipping aggregated all_tasks sorting/printing for now.

        state.reset();
        sTiles::TileIndexerMemoryManager::freeAll();
    }

    TileIndexer::freeMatrixIndices(rows, cols);
    return 0;
}

/**
 * @brief Entry point for the test harness. Individual tests can be toggled
 *        directly. For library builds, this main is excluded by
 *        TILEINDEXER_LIBRARY_ONLY.
 */
int main(int argc, char** argv){
    (void)argc; (void)argv;

    //test1(argc, argv);
    //test2();
    //test3();

    return 1;
}

#endif // TILEINDEXER_LIBRARY_ONLY
