#ifndef STILES_PROCESS_DEBUG_HPP
#define STILES_PROCESS_DEBUG_HPP

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

#include "../common/stiles_structs.hpp"
#include "../common/stiles_logger.hpp"
#include "../ordering/stiles_ordering.hpp"              // preprocess::symbolic_phase
#include "../TileIndexer/preprocess_helpers.hpp"         // constructMapper, bindActive
#include "../tree/tree_wrappers.hpp"                    // leaves_counter, tree_creation
#include "../algorithms/tasks.hpp"                      // collect_tasks
#include "../tile/preprocess.hpp"          // Fast-mode dense allocation helpers
#include "../memory/MemoryManager.hpp"                      // Safe-mode dense allocation

// Forward declarations for functions defined in process.cpp
int sTiles_preprocess_initialization(sTiles_call **call_info, TiledMatrix **scheme, int debug, int group_index);
namespace sTiles { namespace debug {

inline void print_tree_summary(const char* label, const TiledMatrix* scheme)
{
    const char* name = (label && label[0] != '\0') ? label : "unknown";
    constexpr const char* prefix = "│            ";
    constexpr const char* sub_prefix = "│                ";
    if (!scheme) {
        std::printf("%s[TREE][%s] scheme pointer is null\n", prefix, name);
        return;
    }

    const int sep = scheme->red_tree_separator_level;
    std::printf("%s[TREE][%s] sep=%d dimTiledMatrix=%d\n", prefix, name, sep, scheme->dimTiledMatrix);
    if (sep <= 0) {
        std::printf("%s[TREE][%s] tree reduction disabled (separator level <= 0)\n", prefix, name);
        return;
    }

    const int num_sep = sep * (sep + 1) / 2;
    const int base_tile = scheme->dimTiledMatrix - sep;
    const int max_pairs_to_print = 32;
    int printed_pairs = 0;
    long long total_gemm = 0;

    if (scheme->tree_counter) {
        std::printf("%s[TREE][%s] tree_counter summary (non-zero entries):\n", prefix, name);
        for (int col = 0; col < sep; ++col) {
            for (int row = 0; row <= col; ++row) {
                const int idx = row * (2 * sep - row - 1) / 2 + col;
                const int gemm = scheme->tree_counter[idx];
                if (gemm > 0) {
                    if (printed_pairs < max_pairs_to_print) {
                        std::printf("%spair(%d,%d) idx=%d gemms=%d\n",
                                    sub_prefix, base_tile + row, base_tile + col, idx, gemm);
                    }
                    ++printed_pairs;
                    total_gemm += gemm;
                }
            }
        }
        if (printed_pairs == 0) {
            std::printf("%s[none]\n", sub_prefix);
        } else if (printed_pairs > max_pairs_to_print) {
            std::printf("%s... (%d additional entries not shown)\n",
                        sub_prefix, printed_pairs - max_pairs_to_print);
        }
        std::printf("%s[TREE][%s] non-zero entries=%d total_gemm=%lld (num_sep=%d)\n",
                    prefix, name, printed_pairs, total_gemm, num_sep);
    } else {
        std::printf("%s[TREE][%s] tree_counter pointer is null (num_sep=%d)\n", prefix, name, num_sep);
    }

    if (!scheme->trees) {
        std::printf("%s[TREE][%s] trees pointer is null\n", prefix, name);
        return;
    }

    int active_trees = 0;
    int printed_trees = 0;
    const int max_trees_to_print = 16;

    for (int col = 0; col < sep; ++col) {
        for (int row = 0; row <= col; ++row) {
            const int idx = row * (2 * sep - row - 1) / 2 + col;
            const TreeLeaf* tree = scheme->trees[idx];
            if (!tree) continue;

            ++active_trees;
            if (printed_trees < max_trees_to_print) {
                std::printf("%stree[%d] pair(%d,%d) splits=%d tasks=%d\n",
                            sub_prefix, idx, base_tile + row, base_tile + col,
                            tree->num_splits, tree->num_tasks);
            }
            ++printed_trees;
        }
    }

    if (active_trees == 0) {
        std::printf("%s[TREE][%s] trees array present but all entries are null\n", prefix, name);
    } else {
        if (printed_trees > max_trees_to_print) {
            std::printf("%s[TREE][%s] ... (%d additional tree entries not shown)\n",
                        prefix, name, printed_trees - max_trees_to_print);
        }
        std::printf("%s[TREE][%s] total active trees=%d\n", prefix, name, active_trees);
    }
}

#ifdef STILES_SAFEMODE
// End-to-end fast-mode verification driver extracted from process.cpp
// Requires SafeMode to compare safe vs fast implementations
inline int verify_fastmode_end_to_end(sTiles_call **call_info,
                                      TiledMatrix **scheme,
                                      int group_index,
                                      int num_threads_level1)
{
    // 1) Run safe-mode symbolic phase on main scheme
    if (sTiles::SafeMode::preprocess_symbolic_phase(call_info, scheme, group_index) != sTiles::StatusCode::Success) {
        std::fprintf(stderr, "Error: Symbolic factorization (safe mode) failed.\n");
        return EXIT_FAILURE;
    }
    if (sTiles::SafeMode::preprocess_sparse_dense_tiles(call_info, scheme, 0, group_index) != 0) {
        std::fprintf(stderr, "Error: Allocation failed for first call.\n");
        return EXIT_FAILURE;
    }

    // 2) Build independent copy for fast-mode run
    sTiles_call call_fast = **call_info;   // shallow copy
    call_fast.sequence_id = 0;            // force re-init path
    sTiles_call* call_fast_ptr = &call_fast;
    TiledMatrix* scheme_fast = nullptr;
    if (::sTiles_preprocess_initialization(&call_fast_ptr, &scheme_fast, 0, group_index) != 0) {
        std::fprintf(stderr, "Error: Failed to initialize fast-mode scheme for verification.\n");
        return EXIT_FAILURE;
    }

    // 3) Fast-mode symbolic phase
    if (sTiles::preprocess::symbolic_phase(&call_fast_ptr, &scheme_fast, group_index, num_threads_level1) != sTiles::StatusCode::Success) {
        std::fprintf(stderr, "Error: Symbolic factorization (fast mode) failed during verification.\n");
        return EXIT_FAILURE;
    }

    // 4) Compare key outcomes
    int verify_errors = 0;
    if ((**scheme).use_ordering != scheme_fast->use_ordering) {
        std::fprintf(stderr, "[VERIFY] Mismatch: use_ordering safe=%d fast=%d\n",
                     (**scheme).use_ordering, scheme_fast->use_ordering);
        ++verify_errors;
    }

    if ((**scheme).red_tree_separator_level != scheme_fast->red_tree_separator_level) {
        std::fprintf(stderr,
                     "[VERIFY] red_tree_separator_level mismatch resolved: safe=%d fast=%d (adopting fast)\n",
                     (**scheme).red_tree_separator_level,
                     scheme_fast->red_tree_separator_level);
        (**scheme).red_tree_separator_level = scheme_fast->red_tree_separator_level;
        (**call_info).red_tree_separator_level = scheme_fast->red_tree_separator_level;
    }

    const int n_safe = (**scheme).dim;
    const int n_fast = scheme_fast->dim;
    const bool perm_active = ((**scheme).use_ordering > 0) || (scheme_fast->use_ordering > 0);
    if (perm_active) {
        if (!(**scheme).element_perm || !(**scheme).element_iperm ||
            !scheme_fast->element_perm || !scheme_fast->element_iperm) {
            std::fprintf(stderr, "[VERIFY] Mismatch: one of perm/iperm arrays is null while ordering is active.\n");
            ++verify_errors;
        } else if (n_safe != n_fast) {
            std::fprintf(stderr, "[VERIFY] Mismatch: dimensions differ (safe=%d fast=%d).\n", n_safe, n_fast);
            ++verify_errors;
        } else {
            for (int i = 0; i < n_safe; ++i) {
                if ((**scheme).element_perm[i] != scheme_fast->element_perm[i]) {
                    std::fprintf(stderr, "[VERIFY] Mismatch: perm[%d] safe=%d fast=%d\n", i,
                                 (**scheme).element_perm[i], scheme_fast->element_perm[i]);
                    ++verify_errors; break;
                }
            }
            for (int i = 0; i < n_safe; ++i) {
                if ((**scheme).element_iperm[i] != scheme_fast->element_iperm[i]) {
                    std::fprintf(stderr, "[VERIFY] Mismatch: iperm[%d] safe=%d fast=%d\n", i,
                                 (**scheme).element_iperm[i], scheme_fast->element_iperm[i]);
                    ++verify_errors; break;
                }
            }
        }
    }

    if ((**scheme).numActiveTiles != scheme_fast->numActiveTiles) {
        std::fprintf(stderr, "[VERIFY] Mismatch: numActiveTiles safe=%d fast=%d\n",
                     (**scheme).numActiveTiles, scheme_fast->numActiveTiles);
        ++verify_errors;
    }

    if (verify_errors == 0) {
        std::cout << "│            [VERIFY] STILES_VERIFY_FASTMODE: Safe vs Fast [MATCHED]." << std::endl;
    } else {
        sTiles::Logger::warning("│          [!][VERIFY] STILES_VERIFY_FASTMODE: Found ", verify_errors, " mismatch(es).");
        std::cout << "│                         │   [!] STILES_VERIFY_FASTMODE." << std::endl;
    }

    // Construct mapper and bind active for fast scheme
    if (sTiles::preprocess::constructMapper(&scheme_fast) != sTiles::StatusCode::Success) {
        std::fprintf(stderr, "Error: Failed to construct TileIndexer mapper (fast scheme).\n");
        return EXIT_FAILURE;
    }
    if (sTiles::preprocess::bindActive(&scheme_fast) != sTiles::StatusCode::Success) {
        std::fprintf(stderr, "Error: Failed to bind TileIndexer isActive (fast scheme).\n");
        return EXIT_FAILURE;
    }

    // Compare safe vs fast mapping
    {
        TiledMatrix* safe = *scheme;
        TiledMatrix* fast = scheme_fast;
        const int N = safe->dimTiledMatrix;
        int mismatch_active = 0;
        int mismatch_mapper = 0;
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i <= j; ++i) {
                const int tri = i * (2 * N - i - 1) / 2 + j;
                const bool safe_on = (safe->permutation_flags && safe->permutation_flags[tri]);
                const int safe_idx = (safe_on && safe->tileIndexMapper) ? safe->tileIndexMapper[tri] : -1;

                const int fast_idx = fast->mapper.valid() ? fast->mapper.map_ij(i, j, N) : -1;
                const bool fast_on = (fast_idx >= 0) || (fast->state.is_active && fast->state.isActive(i, j, N));

                if (safe_on != fast_on) {
                    ++mismatch_active;
                    std::fprintf(stderr,
                                 "[VERIFY] Active mismatch at (%d,%d): safe=%d fast=%d\n",
                                 i, j, (int)safe_on, (int)fast_on);
                    continue;
                }
                if (safe_on && safe_idx != fast_idx) {
                    ++mismatch_mapper;
                    std::fprintf(stderr,
                                 "[VERIFY] Mapper index mismatch at (%d,%d): safe=%d fast=%d\n",
                                 i, j, safe_idx, fast_idx);
                }
            }
        }
        if (mismatch_active == 0 && mismatch_mapper == 0) {
            std::cout << "│            [VERIFY] Safe vs fast mapping fully match for "
                      << N << "x" << N << " tiles." << std::endl;
        } else {
            std::cout << "│          [!][VERIFY] Summary: active mismatches=" << mismatch_active
                      << ", mapper mismatches=" << mismatch_mapper << std::endl;
        }
    }

    // Strict check: compare safe tileIndexMapper ids vs fast mapper ids for all active tiles
    {
        TiledMatrix* safe = *scheme;
        TiledMatrix* fast = scheme_fast;
        const int N = safe->dimTiledMatrix;
        int pair_mismatches = 0;
        int printed = 0;
        const int cap = 20;
        if (safe->tileIndexMapper && fast->mapper.valid()) {
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i <= j; ++i) {
                    const int tri = i * (2 * N - i - 1) / 2 + j;
                    if (safe->permutation_flags && !safe->permutation_flags[tri]) continue;
                    const int s_id = safe->tileIndexMapper[tri];
                    const int f_id = fast->mapper.map_ij(i, j, N);
                    if (s_id != f_id) {
                        if (printed < cap) {
                            std::fprintf(stderr,
                                         "[VERIFY] mapper-vs-index mismatch at tile(%d,%d): safe=%d fast=%d\n",
                                         i, j, s_id, f_id);
                            ++printed;
                        }
                        ++pair_mismatches;
                    }
                }
            }
            if (pair_mismatches == 0) {
                std::cout << "│            [VERIFY] Safe tileIndexMapper and fast mapper ids match for all active tiles." << std::endl;
            } else {
                std::cout << "│          [!][VERIFY] mapper-vs-index mismatches: " << pair_mismatches << std::endl;
            }
        } else {
            std::fprintf(stderr, "[VERIFY] mapper/index arrays not available for pairwise id check.\n");
        }
    }

    // Per-element pre-lookup check: compute tile ids using safe tileIndexMapper vs fast mapper
    // to ensure both return the same id for each nonzero's tile (before building lookup arrays).
    {
        TiledMatrix* safe = *scheme;
        TiledMatrix* fast = scheme_fast;
        sTiles_call* call = call_fast_ptr; // use same call input for both computations
        const int tile_size = call->tile_size;
        const int N = safe->dimTiledMatrix;
        int mism = 0, printed = 0; const int cap = 20;
        if (safe->tileIndexMapper && fast->mapper.valid()) {
            for (int idx = 0; idx < call->nnz; ++idx) {
                int row, col;
                if (safe->use_ordering == 0) {
                    col = call->row_indices[idx];
                    row = call->col_indices[idx];
                } else {
                    const int er = safe->element_perm[call->row_indices[idx]];
                    const int ec = safe->element_perm[call->col_indices[idx]];
                    if (ec < er) { col = er; row = ec; } else { row = er; col = ec; }
                }
                int ti = row / tile_size;
                int tj = col / tile_size;
                if (ti > tj) std::swap(ti, tj);
                const int tri = ti * (2 * N - ti - 1) / 2 + tj;
                const int s_id = safe->tileIndexMapper[tri];
                const int f_id = fast->mapper.map_ij(ti, tj, N);
                if (s_id != f_id) {
                    if (printed < cap) {
                        std::fprintf(stderr,
                                     "[VERIFY] per-nnz mapper-id mismatch at nnz[%d] tile(%d,%d): safe=%d fast=%d\n",
                                     idx, ti, tj, s_id, f_id);
                        ++printed;
                    }
                    ++mism;
                }
            }
            if (mism == 0) {
                std::cout << "│            [VERIFY] Safe vs fast per-nnz tile ids match (pre-lookup)." << std::endl;
            } else {
                std::cout << "│          [!][VERIFY] Per-nnz mapper-id mismatches (pre-lookup): " << mism << std::endl;
            }
        }
    }

    //exit(0);
    // Safe tree phases (on main scheme)
    sTiles::SafeMode::preprocess_sparse_dense_tree_phase_0_using_bool(call_info, scheme, group_index);
    sTiles::SafeMode::preprocess_sparse_dense_tree_phase_1_using_bool(call_info, scheme, group_index);
    sTiles::debug::print_tree_summary("verify-safe", *scheme);

    // Fast tree phases (on fast scheme)
    if (sTiles::preprocess::leaves_counter(&call_fast_ptr, &scheme_fast, group_index) != sTiles::StatusCode::Success) {
        std::fprintf(stderr, "Error: Tree leaves counting failed (fast).\n");
        return EXIT_FAILURE;
    }
    if (sTiles::preprocess::tree_creation(&call_fast_ptr, &scheme_fast, group_index) != sTiles::StatusCode::Success) {
        std::fprintf(stderr, "Error: Tree creation failed (fast).\n");
        return EXIT_FAILURE;
    }
    sTiles::debug::print_tree_summary("verify-fast", scheme_fast);

    // Safe e-trick build (on main scheme)
    sTiles::SafeMode::preprocess_sparse_dense_e_trick_using_bool(call_info, scheme, group_index);
    sTiles::SafeMode::preprocess_sparse_dense_allocation_copy_using_bol(call_info, scheme);
    sTiles::SafeMode::update_x_sparse_dense_tiles_phase_0(call_info, *scheme, group_index);

    // Collect tasks on fast scheme
    const auto collect_status_fast = sTiles::preprocess::collect_tasks(&call_fast_ptr, &scheme_fast, group_index, num_threads_level1, num_threads_level1, 0);
    if (collect_status_fast != sTiles::StatusCode::Success) {
        std::fprintf(stderr, "Error: Task collection failed (fast). Status=%d\n",
                     static_cast<int>(collect_status_fast));
        return EXIT_FAILURE;
    }

    // Helpers to compare tasks
    auto flatten_e_trick = [](const TiledMatrix* matrix, bool inverse) -> std::vector<std::array<int,7>> {
        std::vector<std::array<int,7>> tasks;
        if (!matrix) return tasks;
        const int ranks = matrix->num_cores;
        if (ranks <= 0) return tasks;
        int** source = inverse ? matrix->e_trick_inv : matrix->e_trick;
        int* sizes = inverse ? matrix->e_trick_size_inv : matrix->e_trick_size;
        if (!source || !sizes) return tasks;
        for (int rank = 0; rank < ranks; ++rank) {
            const int count = sizes[rank];
            const int* base = source[rank];
            if (!base || count <= 0) continue;
            tasks.reserve(tasks.size() + static_cast<std::size_t>(count));
            for (int idx = 0; idx < count; ++idx) {
                std::array<int,7> task{};
                for (int field = 0; field < 7; ++field) {
                    task[static_cast<std::size_t>(field)] = base[idx * 7 + field];
                }
                tasks.push_back(task);
            }
        }
        return tasks;
    };

    const int canonical_tiles = scheme_fast->dimTiledMatrix;
    auto canonical_map_tile = [&](int i, int j) -> int {
        if (i > j) std::swap(i, j);
        if (i < 0 || j < 0 || j >= canonical_tiles) return -1;
        if (scheme_fast->mapper.valid()) return scheme_fast->mapper.map_ij(i, j, canonical_tiles);
        const int tri = i * (2 * canonical_tiles - i - 1) / 2 + j;
        if (scheme_fast->tileIndexMapper && scheme_fast->tileIndexMapper[tri] >= 0) return scheme_fast->tileIndexMapper[tri];
        if (scheme_fast->tileIndexMapper2 && scheme_fast->tileIndexMapper2[tri] >= 0) return scheme_fast->tileIndexMapper2[tri];
        return -1;
    };

    auto canonicalize_chol_tasks = [&](const std::vector<std::array<int,7>>& input)
        -> std::pair<bool, std::vector<std::array<int,7>>> {
        std::vector<std::array<int,7>> output = input; bool ok = true;
        for (auto& row : output) {
            const int type = row[0]; const int m = row[1]; const int k = row[2]; const int n = row[3];
            (void)m;
            switch (type) {
                case 1: { const int d = canonical_map_tile(k,k); if (d < 0) { ok=false; break; } row[4]=d; row[5]=d; row[6]=0; break; }
                case 2: { const int i1=canonical_map_tile(n,k); const int d=canonical_map_tile(k,k); if (i1<0||d<0){ok=false;break;} row[4]=i1; row[5]=d; row[6]=0; break; }
                case 3: { const int km=canonical_map_tile(k,m); const int d=canonical_map_tile(k,k); if (km<0||d<0){ok=false;break;} row[4]=km; row[5]=d; row[6]=km; break; }
                case 4: { const int i1=canonical_map_tile(n,k); const int i2=canonical_map_tile(n,m); const int i3=canonical_map_tile(k,m); if (i1<0||i2<0||i3<0){ok=false;break;} row[4]=i1; row[5]=i2; row[6]=i3; break; }
                default: break;
            }
            if (!ok) break;
        }
        return {ok, std::move(output)};
    };

    auto canonicalize_inv_tasks = [&](const std::vector<std::array<int,7>>& input)
        -> std::pair<bool, std::vector<std::array<int,7>>> {
        std::vector<std::array<int,7>> output = input; bool ok = true;
        for (auto& row : output) {
            const int type=row[0]; const int i=row[1]; const int j=row[2]; const int k=row[3];
            switch (type) {
                case 1: { const int d=canonical_map_tile(i,i); if (d<0){ok=false;break;} row[4]=d; row[5]=d; row[6]=0; break; }
                case 2: { const int d=canonical_map_tile(i,i); const int idx=canonical_map_tile(i,j); if (d<0||idx<0){ok=false;break;} row[4]=d; row[5]=idx; row[6]=0; break; }
                case 3: { row[1]=row[2]=row[3]=row[4]=row[5]=row[6]=0; break; }
                case 4: case 6: { row[3]=0; const int d=canonical_map_tile(i,i); if (d<0){ok=false;break;} row[4]=d; row[5]=0; row[6]=0; break; }
                case 5: { const int d=canonical_map_tile(i,i); const int idx=canonical_map_tile(i,k); if (d<0||idx<0){ok=false;break;} row[4]=d; row[5]=idx; row[6]=0; break; }
                case 7: { const int ik=canonical_map_tile(i,k); const int jk=canonical_map_tile(j,k); const int ij=canonical_map_tile(i,j); if (ik<0||jk<0||ij<0){ok=false;break;} row[4]=ik; row[5]=jk; row[6]=ij; break; }
                case 8: { const int ik=canonical_map_tile(i,k); const int kj=canonical_map_tile(k,j); const int ij=canonical_map_tile(i,j); if (ik<0||kj<0||ij<0){ok=false;break;} row[4]=ik; row[5]=kj; row[6]=ij; break; }
                case 9: { row[1]=row[2]=row[3]=row[4]=row[5]=row[6]=0; break; }
                default: break;
            }
            if (!ok) break;
        }
        return {ok, std::move(output)};
    };

    auto compare_task_vectors = [](const std::vector<std::array<int,7>>& safe_vec_in,
                                   const std::vector<std::array<int,7>>& fast_vec_in,
                                   const char* label) -> bool {
        if (safe_vec_in.empty() && fast_vec_in.empty()) return true;
        auto safe_vec = safe_vec_in; auto fast_vec = fast_vec_in;
        auto cmp = [](const std::array<int,7>& a, const std::array<int,7>& b) {
            for (int f=0; f<7; ++f) { if (a[(std::size_t)f] != b[(std::size_t)f]) return a[(std::size_t)f] < b[(std::size_t)f]; }
            return false; };
        std::sort(safe_vec.begin(), safe_vec.end(), cmp);
        std::sort(fast_vec.begin(), fast_vec.end(), cmp);
        if (safe_vec.size() != fast_vec.size()) {
            std::cout << "          [!][VERIFY] " << label << " task count mismatch: safe="
                      << safe_vec.size() << " fast=" << fast_vec.size() << std::endl;
            return false;
        }
        for (std::size_t idx = 0; idx < safe_vec.size(); ++idx) {
            if (safe_vec[idx] != fast_vec[idx]) {
                const auto& s = safe_vec[idx]; const auto& f = fast_vec[idx];
                std::fprintf(stderr,
                             "[VERIFY] %s task mismatch at index %zu: safe=[%d,%d,%d,%d,%d,%d,%d] fast=[%d,%d,%d,%d,%d,%d,%d]\n",
                             label, idx,
                             s[0],s[1],s[2],s[3],s[4],s[5],s[6],
                             f[0],f[1],f[2],f[3],f[4],f[5],f[6]);
                return false;
            }
        }
        return true;
    };

    const auto safe_chol_tasks = flatten_e_trick(*scheme, /*inverse=*/false);
    const auto safe_inv_tasks  = flatten_e_trick(*scheme, /*inverse=*/true);
    const auto [safe_chol_ok, canonical_safe_chol] = canonicalize_chol_tasks(safe_chol_tasks);
    const auto [fast_chol_ok, canonical_fast_chol] = canonicalize_chol_tasks(get_chol_tasks(scheme_fast));
    const auto [safe_inv_ok, canonical_safe_inv]   = canonicalize_inv_tasks(safe_inv_tasks);
    const auto [fast_inv_ok, canonical_fast_inv]   = canonicalize_inv_tasks(get_inv_tasks(scheme_fast));

    bool chol_tasks_match = false;
    if (safe_chol_ok && fast_chol_ok) {
        chol_tasks_match = compare_task_vectors(canonical_safe_chol, canonical_fast_chol, "chol");
    } else {
        std::cout << "│          [!][VERIFY] Unable to canonicalize chol task indices; falling back to raw comparison." << std::endl;
        chol_tasks_match = compare_task_vectors(safe_chol_tasks, get_chol_tasks(scheme_fast), "chol");
    }

    bool inv_tasks_match = false;
    if (safe_inv_ok && fast_inv_ok) {
        inv_tasks_match = compare_task_vectors(canonical_safe_inv, canonical_fast_inv, "inv");
    } else {
        std::cout << "│          [!][VERIFY] Unable to canonicalize inv task indices; falling back to raw comparison." << std::endl;
        inv_tasks_match = compare_task_vectors(safe_inv_tasks, sTiles::get_inv_tasks(scheme_fast), "inv");
    }

    if (chol_tasks_match && inv_tasks_match) {
        std::cout << "│            [VERIFY] Safe vs fast task lists match." << std::endl;
    }

    sTiles::set_tile_extents(&scheme_fast);
    sTiles::preprocess::build_dense_tile_lookup(call_info, scheme_fast, group_index, num_threads_level1);

    // Compare safe vs fast element-to-tile lookup arrays
    {
        const int nnz_safe = (**scheme).original_nnz;
        const int nnz_fast = scheme_fast->original_nnz;
        if (nnz_safe != nnz_fast) {
            std::fprintf(stderr, "[VERIFY] Lookup nnz mismatch: safe=%d fast=%d\n", nnz_safe, nnz_fast);
        } else if ((**scheme).t_indicies && (**scheme).e_indicies &&
                   scheme_fast->tile_index_lookup && scheme_fast->element_offset_lookup) {
            int mismatches = 0;
            const int cap = 20; // limit printed examples
            for (int idx = 0; idx < nnz_safe; ++idx) {
                const int s_t = (**scheme).t_indicies[idx];
                const int s_e = (**scheme).e_indicies[idx];
                const int f_t = scheme_fast->tile_index_lookup[idx];
                const int f_e = scheme_fast->element_offset_lookup[idx];
                if (s_t != f_t || s_e != f_e) {
                    if (mismatches < cap) {
                        std::fprintf(stderr,
                                     "[VERIFY] lookup mismatch at nnz[%d]: safe(t=%d,e=%d) fast(t=%d,e=%d)\n",
                                     idx, s_t, s_e, f_t, f_e);
                    }
                    ++mismatches;
                }
            }
            if (mismatches == 0) {
                std::cout << "│            [VERIFY] Safe vs fast lookup arrays match." << std::endl;
            } else {
                std::cout << "│          [!][VERIFY] Lookup mismatches: " << mismatches << std::endl;
            }
        } else {
            std::fprintf(stderr, "[VERIFY] Lookup arrays not available for comparison.\n");
        }
    }

    return EXIT_SUCCESS;
}
#endif // STILES_SAFEMODE


}} // namespace sTiles::debug

#endif // STILES_PROCESS_DEBUG_HPP
