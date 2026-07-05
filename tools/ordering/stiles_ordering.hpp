
#ifndef STILES_WRAPPER_ORDERING_H
#define STILES_WRAPPER_ORDERING_H
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "async_bigstack.hpp"   // big-stack async for the ordering bake-off (macOS)
#include "fill-in/symbolic_chol_fillin.hpp"
#include "fill-in/symbolic_chol_fillin_left.hpp"

#include "../memory/OrderingMemoryManager.hpp"
#include "../common/stiles_exporter.hpp"
#include "../common/stiles_logger.hpp"
#include "../TileIndexer/TileIndexer.hpp"
#include "../TileIndexer/TileIndexerFill.hpp"  // inline FillTiles definitions
#include "../TileIndexer/TileIndexerGraphBuilder.hpp"  // build_graphs_up_lo
#include "ordering_utils.hpp"
#include <omp.h>
#include <unordered_set>
#include <atomic>
#include <future>
#include <chrono>

// Direct METIS ordering — declared outside namespace (extern "C" linkage)
extern "C" int stiles_runMETIS_direct(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, const sTiles::SharedAdjCSR* shared = nullptr);

namespace sTiles {

// Forward declarations for user permutation accessors (defined in process.cpp)
const int* get_user_permutation(int group_index);
int get_user_permutation_size(int group_index);
bool get_user_permutation_force(int group_index);
bool get_forced_partition_sizes(int group_index, int& p1, int& p2, int& sep);

// ---- Mode-aware padding gate (fix for the mode-2 corruption) -------------
// Partitioning/padding MUST only touch semisparse(1)/dense(0) matrices; padding
// a sparse(2) matrix corrupts the supernodal factor ("not positive-definite").
// The mode isn't resolved until preprocess, but fill (nnz_factor) and L
// (L_colptr/L_rowind) are set by the winner-install before the padding
// decision, so we predict the mode here. These mirror mean_occupancy_from_L +
// the selector in process.cpp (~L202 / L1438) — KEEP IN SYNC. Named distinctly
// and file-local to avoid an ODR clash with process.cpp's copy in the same TU.
static inline double pad_gate_block_density(const TiledMatrix* scheme, int ts) {
    const int* colptr = scheme->L_colptr;
    const int* rowind = scheme->L_rowind;
    const int n = scheme->dim;
    if (!colptr || !rowind || n <= 0 || ts <= 0) return 0.0;
    const int ntr = (n - 1) / ts + 1;
    std::vector<long long> nnzR(static_cast<std::size_t>(ntr), 0);
    std::vector<int> touched;
    double dens_sum = 0.0; long long nblk = 0;
    for (int C = 0; C < ntr; ++C) {
        const int c0 = C * ts;
        const int w  = std::min(n, c0 + ts) - c0;
        touched.clear();
        for (int c = c0; c < c0 + w; ++c)
            for (long long p = colptr[c]; p < colptr[c + 1]; ++p) {
                const int R = static_cast<int>(rowind[p] / ts);
                if (R <= C) continue;
                if (nnzR[R] == 0) touched.push_back(R);
                nnzR[R] += 1;
            }
        nblk += static_cast<long long>(touched.size());
        for (int R : touched) {
            const int r0 = R * ts, h = std::min(n, r0 + ts) - r0;
            dens_sum += static_cast<double>(nnzR[R]) / (static_cast<double>(h) * w);
            nnzR[R] = 0;
        }
    }
    return nblk ? dens_sum / static_cast<double>(nblk) : 0.0;
}
// Predicted tile mode (0=dense,1=semisparse,2=sparse). Honors a forced mode
// (0/1/2); only for auto (3) does it predict from fill + block density.
static inline int pad_gate_would_be_mode(const TiledMatrix* scheme, int ttm, int ts) {
    if (ttm == 0 || ttm == 1 || ttm == 2) return ttm;            // forced mode
    const long long innz = scheme->nnz > 0 ? scheme->nnz : 1;
    const double fill = static_cast<double>(scheme->nnz_factor) / static_cast<double>(innz);
    if (fill >= 3.5) return 2;                                   // high fill -> sparse
    if (std::getenv("STILES_LEGACY_FILL")) return 1;
    const double occ = pad_gate_block_density(scheme, ts);
    return (occ >= 0.15 && fill <= 2.0) ? 1 : 2;                 // dense+low-fill -> semi
}

inline std::string ordering_name_from_id(int id) {
    switch (id) {
        case 1:  return "RCM";
        case 2:  return "ND";
        case 21: return "METIS";
        case 3:  return "ND+RCM";
        case 4:  return "ND";
        case 41: return "AND";
        case 42: return "FND";
        case 5:  return "AMD";
        case 6:  return "CAMD";
        case 7:  return "COLAMD";
        case 8:  return "CCOLAMD";
        case 9:  return "SYMAMD";
        case 10: return "User-provided";
        case 11: return "Hub-CAMD";
        case 12: return "AMD";
        case 13: return "AMDBAR";
        case 14: return "AMDBARNEW";
        case 15: return "AMDNEW";
        case 16: return "GenMMD";
        case 17: return "GPS";
        case 18: return "SMTP-Band";
        default: break;
    }
    return "Custom_" + std::to_string(id);
}




struct StrategyResult {
    int id = 0;
    std::string name;
    int *perm = nullptr;
    int *iperm = nullptr;
    int *partition_sizes = nullptr;  // ND output; owned by this result
    TileIndexer::State state;
    int active = -1;
    int filled = -1;
    int tree_sep = 0;
    StatusCode status = StatusCode::IllegalValue;
    double elapsed = 0.0;

    // Helper to clean up if this strategy is NOT chosen
    void discard() {
        if (perm) {
            OrderingMemoryManager::deallocate(perm);
            perm = nullptr;
        }
        if (iperm) {
            OrderingMemoryManager::deallocate(iperm);
            iperm = nullptr;
        }
        if (partition_sizes) {
            delete[] partition_sizes;
            partition_sizes = nullptr;
        }
        TileIndexer::release_state_resources(state);
    }
};

inline std::vector<int> unique_digits_1_to_9(long long n) {
    n = std::llabs(n);
    if (n == 0) {
        std::cerr << "Error: digits must be 1..9 only\n";
        std::exit(EXIT_FAILURE);
    }

    bool seen[10] = {false};
    while (n > 0) {
        const int d = static_cast<int>(n % 10);
        if (d < 1 || d > 9) {
            std::cerr << "Error: digit " << d << " is not allowed (only 1..9)\n";
            std::exit(EXIT_FAILURE);
        }
        seen[d] = true;
        n /= 10;
    }

    std::vector<int> out;
    for (int d = 1; d <= 9; ++d) {
        if (seen[d]) out.push_back(d);
    }
    return out;
}


// Forward declarations for ordering algorithms
extern "C" int stiles_createSmartPermutation(int** row_indices, int** col_indices, int nnz, int node_num, int** perm);
int runRCM(int** csr_i, int** csr_j, int N, int nnz, int m, int** iperm, int** perm, bool verbose);
void runND(int** csr_i, int** csr_j, int N, int nnz, int m, int** iperm, int** perm, int num_sep, int** sizes);
void runNDRCM(int** csr_i, int** csr_j, int N, int nnz, int m, int** iperm, int** perm, int num_sep, int** sizes);
// runSCOTCH is declared in ordering_utils.hpp (included above) with the optional
// `const SharedAdjCSR* shared = nullptr` param — do NOT redeclare it here, or the
// 9-arg calls become ambiguous against that overload.
int runSuiteSparse(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, int num_cores, int strategy_num, const SharedAdjCSR* shared = nullptr);

// Forward declarations for extern and tile-level orderings (defined in their own .cpp files,
// already in namespace sTiles — no wrapper needed here since we are already inside it)
void stiles_runAMD(int* csr_i, int* csr_j, int N, int nnz, int** perm, int** iperm);
void stiles_runAMDBAR(int* csr_i, int* csr_j, int N, int nnz, int** perm, int** iperm);
void stiles_runAMDBARNEW(int* csr_i, int* csr_j, int N, int nnz, int** perm, int** iperm);
void stiles_runAMDNEW(int* csr_i, int* csr_j, int N, int nnz, int** perm, int** iperm);
void stiles_runGenMMD(int* csr_i, int* csr_j, int N, int nnz, int** perm, int** iperm, int delta = 0);
void stiles_runGPS(int* csr_i, int* csr_j, int N, int nnz, int** perm, int** iperm, int mode = 0);
void stiles_runTileOrdering(int* elem_i, int* elem_j, int N, int nnz,
                             int sub_ordering,
                             int** perm, int** iperm,
                             int tile_size = 2);

inline StatusCode run_permutation(int** my_perm, int** my_iperm, int* N, int* nnz, int fixed_col, int** indices_i, int** indices_j, int ordering, int* tree_sep, int group_index, int** sizes, int num_cores = 0, ScotchTree* scotch_tree_out = nullptr, const SharedAdjCSR* shared = nullptr) {

    if (!N || !nnz || !indices_i || !indices_j || !my_perm || !my_iperm || !tree_sep) {
        sTiles::Logger::error("run_permutation: null argument.");
        return StatusCode::IllegalValue;
    }

    *tree_sep = 0;
    sTiles::Logger::info("│     • Ordering stage initialised (strategy " + std::to_string(ordering) + ")");

    *my_perm  = OrderingMemoryManager::allocate<int>(*N, group_index);
    *my_iperm = OrderingMemoryManager::allocate<int>(*N, group_index);
    if (!*my_perm || !*my_iperm) {
        sTiles::Logger::error("run_permutation: allocation failed for perm/iperm.");
        return StatusCode::OutOfResources;
    }

    // External orderings (12+) allocate *perm/*iperm with malloc internally,
    // replacing the pre-allocations above.  This helper frees the pre-allocated
    // buffers first, runs the external function, then copies the malloc result
    // into fresh buffers so the caller can always use
    // OrderingMemoryManager::deallocate safely.
    auto wrap_extern_ordering = [&](auto fn) {
        OrderingMemoryManager::deallocate(*my_perm);
        OrderingMemoryManager::deallocate(*my_iperm);
        fn(my_perm, my_iperm);
        int* rp = *my_perm;   // extern perm:  new→old
        int* ri = *my_iperm;  // extern iperm: old→new
        *my_perm  = OrderingMemoryManager::allocate<int>(*N, group_index);
        *my_iperm = OrderingMemoryManager::allocate<int>(*N, group_index);
        if (*my_perm && *my_iperm && rp && ri) {
            // count_active_tiles_with_perm expects old→new in perm (METIS convention)
            std::copy(ri, ri + *N, *my_perm);   // old->new -> *my_perm
            std::copy(rp, rp + *N, *my_iperm);  // new->old -> *my_iperm
        }
        free(rp);
        free(ri);
    };

    if (ordering == 0) {
        for (int i = 0; i < *N; ++i) {
            (*my_perm)[i]  = i;
            (*my_iperm)[i] = i;
        }
        sTiles::Logger::debug("│     • Identity permutation selected (ordering=0).");
        return StatusCode::Success;
    }

    StatusCode code = StatusCode::Success;

    if (ordering == 1) {
        const double t_ref_start = omp_get_wtime();
        *tree_sep = sTiles::runRCM(indices_i, indices_j, *N, *nnz, fixed_col, my_iperm, my_perm, true);
        const double t_ref = omp_get_wtime() - t_ref_start;
        sTiles::Logger::debug("│   ↪ sTiles_ordering_1 completed in " + sTiles::format_seconds(t_ref) + " s");
        
    } else if (ordering == 2) {
        const double t_ref_start = omp_get_wtime();
        const int num_sep = 2;
        sTiles::runND(indices_i, indices_j, *N, *nnz, fixed_col, my_iperm, my_perm, num_sep, sizes);
        const double t_ref = omp_get_wtime() - t_ref_start;
        sTiles::Logger::debug("│   ↪ sTiles_ordering_2 completed in " + sTiles::format_seconds(t_ref) + " s");
        //fprintf(stderr, "[debug] g_nd_hub_skipped=%d\n", (int)::g_nd_hub_skipped);
        if (::g_nd_hub_skipped) return StatusCode::NotSupported;
    } else if (ordering == 3) {
        sTiles::runRCM_scipy(indices_i, indices_j, *N, *nnz, fixed_col, my_iperm, my_perm, true);

    } else if (ordering == 21) {
        const double t_ref_start = omp_get_wtime();
        int rc = ::stiles_runMETIS_direct(indices_i, indices_j, *N, *nnz, fixed_col, my_perm, my_iperm, shared);
        const double t_ref = omp_get_wtime() - t_ref_start;
        if (rc != 0) {
            sTiles::Logger::warning("run_permutation: METIS direct ordering failed, falling back to identity.");
            for (int i = 0; i < *N; ++i) { (*my_perm)[i] = i; (*my_iperm)[i] = i; }
            code = StatusCode::ExecutionFailed;
        } else {
            sTiles::Logger::debug("│   ↪ sTiles_ordering_21 (METIS direct) completed in " + sTiles::format_seconds(t_ref) + " s");
        }

    } else if (ordering == 41) {
        // ASCOTCH: pre-permute with stiles_createSmartPermutation (RCM-like
        // bandwidth reduction) then run SCOTCH on the reordered graph. The
        // pre-permutation gives SCOTCH better locality for its multi-level
        // coarsening, which typically produces lower nnz(L). Seed=42 selects
        // the quality strategy (tight balance, fine ND) which works best
        // WITH the pre-permutation — the pre-perm already provides locality,
        // so tight balance + deep ND gives the best separators.
        const int seed = 42;
        const double t_ref_start = omp_get_wtime();
        int rc = sTiles::runASCOTCH(indices_i, indices_j, *N, *nnz, fixed_col, my_iperm, my_perm, num_cores, seed);
        const double t_ref = omp_get_wtime() - t_ref_start;

        if (rc != 0) {
            sTiles::Logger::warning("run_permutation: ASCOTCH ordering failed, falling back to identity.");
            for (int i = 0; i < *N; ++i) {
                (*my_perm)[i]  = i;
                (*my_iperm)[i] = i;
            }
            code = StatusCode::ExecutionFailed;
        } else {
            sTiles::Logger::debug("│   ↪ sTiles_ordering_41 (ASCOTCH) completed in " + sTiles::format_seconds(t_ref) + " s");
        }

    } else if (ordering == 4 || ordering == 42) {
        // SCOTCH (id=4): balanced strategy, seed=0
        // FSCOTCH (id=42): relaxed strategy, seed=7
        const int seed = (ordering == 4) ? 0 : 7;
        const double t_ref_start = omp_get_wtime();
        int rc = scotch_tree_out
                 ? sTiles::runSCOTCH_with_tree(indices_i, indices_j, *N, *nnz, fixed_col, my_iperm, my_perm, num_cores, seed, scotch_tree_out, shared)
                 : sTiles::runSCOTCH(indices_i, indices_j, *N, *nnz, fixed_col, my_iperm, my_perm, num_cores, seed, shared);
        const double t_ref = omp_get_wtime() - t_ref_start;

        if (rc != 0) {
            sTiles::Logger::warning("run_permutation: SCOTCH ordering failed, falling back to identity.");
            for (int i = 0; i < *N; ++i) {
                (*my_perm)[i]  = i;
                (*my_iperm)[i] = i;
            }
            code = StatusCode::ExecutionFailed;
        } else {
            sTiles::Logger::debug("│   ↪ sTiles_ordering_" + std::to_string(ordering) + " completed in " + sTiles::format_seconds(t_ref) + " s");
        }

    } else if (ordering == 5) {
        const double t_ref_start = omp_get_wtime();
        int rc = sTiles::runSuiteSparse(indices_i, indices_j, *N, *nnz, fixed_col, my_perm, my_iperm, num_cores, 0, shared);
        const double t_ref = omp_get_wtime() - t_ref_start;

        if (rc != 0) {
            sTiles::Logger::warning("run_permutation: SuiteSparse ordering failed, falling back to identity.");
            for (int i = 0; i < *N; ++i) {
                (*my_perm)[i]  = i;
                (*my_iperm)[i] = i;
            }
            code = StatusCode::ExecutionFailed;
        } else {
            sTiles::Logger::debug("│   ↪ sTiles_ordering_5 completed in " + sTiles::format_seconds(t_ref) + " s");
        }
    } else if (ordering == 6) {
        const double t_ref_start = omp_get_wtime();
        int rc = sTiles::runSuiteSparse(indices_i, indices_j, *N, *nnz, fixed_col, my_perm, my_iperm, num_cores, 1, shared);
        const double t_ref = omp_get_wtime() - t_ref_start;

        if (rc != 0) {
            sTiles::Logger::warning("run_permutation: SuiteSparse ordering failed, falling back to identity.");
            for (int i = 0; i < *N; ++i) {
                (*my_perm)[i]  = i;
                (*my_iperm)[i] = i;
            }
            code = StatusCode::ExecutionFailed;
        } else {
            sTiles::Logger::debug("│   ↪ sTiles_ordering_6 completed in " + sTiles::format_seconds(t_ref) + " s");
        }
    } else if (ordering == 7) {
        const double t_ref_start = omp_get_wtime();
        int rc = sTiles::runSuiteSparse(indices_i, indices_j, *N, *nnz, fixed_col, my_perm, my_iperm, num_cores, 2);
        const double t_ref = omp_get_wtime() - t_ref_start;

        if (rc != 0) {
            sTiles::Logger::warning("run_permutation: SuiteSparse ordering failed, falling back to identity.");
            for (int i = 0; i < *N; ++i) {
                (*my_perm)[i]  = i;
                (*my_iperm)[i] = i;
            }
            code = StatusCode::ExecutionFailed;
        } else {
            sTiles::Logger::debug("│   ↪ sTiles_ordering_7 completed in " + sTiles::format_seconds(t_ref) + " s");
        }
    } else if (ordering == 8) {
        const double t_ref_start = omp_get_wtime();
        int rc = sTiles::runSuiteSparse(indices_i, indices_j, *N, *nnz, fixed_col, my_perm, my_iperm, num_cores, 3);
        const double t_ref = omp_get_wtime() - t_ref_start;

        if (rc != 0) {
            sTiles::Logger::warning("run_permutation: SuiteSparse ordering failed, falling back to identity.");
            for (int i = 0; i < *N; ++i) {
                (*my_perm)[i]  = i;
                (*my_iperm)[i] = i;
            }
            code = StatusCode::ExecutionFailed;
        } else {
            sTiles::Logger::debug("│   ↪ sTiles_ordering_8 completed in " + sTiles::format_seconds(t_ref) + " s");
        }
    } else if (ordering == 9) {
        const double t_ref_start = omp_get_wtime();
        int rc = sTiles::runSuiteSparse(indices_i, indices_j, *N, *nnz, fixed_col, my_perm, my_iperm, num_cores, 4);
        const double t_ref = omp_get_wtime() - t_ref_start;

        if (rc != 0) {
            sTiles::Logger::warning("run_permutation: SuiteSparse ordering failed, falling back to identity.");
            for (int i = 0; i < *N; ++i) {
                (*my_perm)[i]  = i;
                (*my_iperm)[i] = i;
            }
            code = StatusCode::ExecutionFailed;
        } else {
            sTiles::Logger::debug("│   ↪ sTiles_ordering_9 completed in " + sTiles::format_seconds(t_ref) + " s");
        }
    } else if (ordering == 10) {
        // User-provided permutation
        const int* user_perm = get_user_permutation(group_index);
        const int user_perm_size = get_user_permutation_size(group_index);

        if (!user_perm || user_perm_size != *N) {
            sTiles::Logger::error("run_permutation: user permutation unavailable or size mismatch for group ",
                                 group_index, " (expected size ", *N, ", got ", user_perm_size, ").");
            for (int i = 0; i < *N; ++i) {
                (*my_perm)[i]  = i;
                (*my_iperm)[i] = i;
            }
            return StatusCode::IllegalValue;
        }

        // Copy user permutation
        std::copy(user_perm, user_perm + *N, *my_perm);
        compute_inverse_permutation(*my_perm, *my_iperm, *N);

        sTiles::Logger::debug("│     • User-provided permutation applied for group ", group_index, " (ordering=10).");
        code = StatusCode::Success;
    } else if (ordering == 12) {
        const double t_ref_start = omp_get_wtime();
        wrap_extern_ordering([&](int** p, int** ip) { sTiles::stiles_runAMD(*indices_i, *indices_j, *N, *nnz, p, ip); });
        sTiles::Logger::debug("│   ↪ sTiles_ordering_12 (AMD) completed in " + sTiles::format_seconds(omp_get_wtime() - t_ref_start) + " s");
    } else if (ordering == 13) {
        const double t_ref_start = omp_get_wtime();
        wrap_extern_ordering([&](int** p, int** ip) { sTiles::stiles_runAMDBAR(*indices_i, *indices_j, *N, *nnz, p, ip); });
        sTiles::Logger::debug("│   ↪ sTiles_ordering_13 (AMDBAR) completed in " + sTiles::format_seconds(omp_get_wtime() - t_ref_start) + " s");
    } else if (ordering == 14) {
        const double t_ref_start = omp_get_wtime();
        wrap_extern_ordering([&](int** p, int** ip) { sTiles::stiles_runAMDBARNEW(*indices_i, *indices_j, *N, *nnz, p, ip); });
        sTiles::Logger::debug("│   ↪ sTiles_ordering_14 (AMDBARNEW) completed in " + sTiles::format_seconds(omp_get_wtime() - t_ref_start) + " s");
    } else if (ordering == 15) {
        const double t_ref_start = omp_get_wtime();
        wrap_extern_ordering([&](int** p, int** ip) { sTiles::stiles_runAMDNEW(*indices_i, *indices_j, *N, *nnz, p, ip); });
        sTiles::Logger::debug("│   ↪ sTiles_ordering_15 (AMDNEW) completed in " + sTiles::format_seconds(omp_get_wtime() - t_ref_start) + " s");
    } else if (ordering == 16) {
        const double t_ref_start = omp_get_wtime();
        wrap_extern_ordering([&](int** p, int** ip) { sTiles::stiles_runGenMMD(*indices_i, *indices_j, *N, *nnz, p, ip); });
        sTiles::Logger::debug("│   ↪ sTiles_ordering_16 (GenMMD) completed in " + sTiles::format_seconds(omp_get_wtime() - t_ref_start) + " s");
    } else if (ordering == 17) {
        const double t_ref_start = omp_get_wtime();
        wrap_extern_ordering([&](int** p, int** ip) { sTiles::stiles_runGPS(*indices_i, *indices_j, *N, *nnz, p, ip); });
        sTiles::Logger::debug("│   ↪ sTiles_ordering_17 (GPS) completed in " + sTiles::format_seconds(omp_get_wtime() - t_ref_start) + " s");
    } else {
        sTiles::Logger::warning("run_permutation: unknown ordering = " + std::to_string(ordering) + ", using identity.");
        for (int i = 0; i < *N; ++i) {
            (*my_perm)[i]  = i;
            (*my_iperm)[i] = i;
        }
        return StatusCode::Success;
    }

    if (code == StatusCode::Success) {
        if (!sTiles::check_inverse_permutation(*my_perm, *my_iperm, *N)) {
            sTiles::Logger::error("run_permutation: invalid permutation produced for ordering "
                                  + std::to_string(ordering)
                                  + ", falling back to identity permutation.");
            for (int i = 0; i < *N; ++i) {
                (*my_perm)[i]  = i;
                (*my_iperm)[i] = i;
            }
            return StatusCode::IllegalValue;
        }
        sTiles::Logger::debug("│     • Inverse permutation check result: 1");
    }

    return code;
}

namespace preprocess{

// =============================================================================
// Depth-D ND partition extraction from a SCOTCH separator tree.
//
// Given the SCOTCH tree (rangtab/treetab) and a cut depth D, produce
// 2^(D+1) - 1 partition descriptors in POSTORDER:
//
//   [L_0] [L_1] [S_...] [L_2] [L_3] [S_...] [S_...] ... [S_root]
//
// Where 2^D "cut leaves" are roots of D-deep subtrees treated as atomic
// regions, and 2^D - 1 interior separators are at levels 0..D-1.
// Core assignment halves per level down: a cores total, leaves get ~a/2^D
// each, separators inherit their subtree's core range.
//
// Returns true on success. False for degenerate trees (non-binary root,
// missing children). Caller (apply_scotch_block_padding) should fall back
// to the legacy 3-way path on false.
//
// Column ranges are in the OLD (un-padded) permuted space. Caller pads.
// =============================================================================
struct NwayRegion {
    int start_col;
    int end_col;
    int first_core;
    int num_cores;
    bool is_leaf;
    int depth_level;
    std::string label;
};

static bool build_nway_regions(
    const TiledMatrix& scheme,
    int target_depth,
    int num_cores,
    std::vector<NwayRegion>& out)
{
    out.clear();
    if (target_depth < 1) return false;
    const int cb = scheme.scotch_cblknbr;
    const int* rangtab = scheme.scotch_rangtab;
    const int* treetab = scheme.scotch_treetab;
    if (cb <= 0 || !rangtab || !treetab) return false;

    int root = -1;
    for (int b = 0; b < cb; ++b) {
        if (treetab[b] == -1) {
            if (root != -1) return false;
            root = b;
        }
    }
    if (root < 0) return false;

    std::vector<std::vector<int>> children(static_cast<std::size_t>(cb));
    for (int b = 0; b < cb; ++b) {
        const int p = treetab[b];
        if (p >= 0 && p < cb) children[static_cast<std::size_t>(p)].push_back(b);
    }

    // Bottom-up first_desc: SCOTCH postorder → children have smaller indices
    // than parents, so a single forward pass correctly folds child-min into
    // each parent.
    std::vector<int> first_desc(static_cast<std::size_t>(cb), -1);
    for (int b = 0; b < cb; ++b) {
        if (children[static_cast<std::size_t>(b)].empty()) {
            first_desc[static_cast<std::size_t>(b)] = b;
        } else {
            int mn = b;
            for (int c : children[static_cast<std::size_t>(b)]) {
                if (first_desc[static_cast<std::size_t>(c)] >= 0 &&
                    first_desc[static_cast<std::size_t>(c)] < mn) {
                    mn = first_desc[static_cast<std::size_t>(c)];
                }
            }
            first_desc[static_cast<std::size_t>(b)] = mn;
        }
    }

    std::function<bool(int, int, int, int)> emit =
        [&](int node, int cur_depth, int first_core, int num_cores_here) -> bool {
        auto emit_leaf = [&](int n) {
            const int first = first_desc[static_cast<std::size_t>(n)];
            NwayRegion r;
            r.start_col = rangtab[first];
            r.end_col = rangtab[n + 1];
            r.first_core = first_core;
            r.num_cores = std::max(1, num_cores_here);
            r.is_leaf = true;
            r.depth_level = cur_depth;
            int leaf_idx = 0;
            for (const auto& p : out) if (p.is_leaf) ++leaf_idx;
            r.label = "L" + std::to_string(leaf_idx);
            out.push_back(std::move(r));
        };
        if (cur_depth >= target_depth) {
            emit_leaf(node);
            return true;
        }
        const auto& ch = children[static_cast<std::size_t>(node)];
        if (ch.empty()) {
            emit_leaf(node);
            return true;
        }
        if (ch.size() > 2) return false;

        int c1 = ch[0], c2 = (ch.size() == 2 ? ch[1] : -1);
        if (c2 >= 0 && c1 > c2) std::swap(c1, c2);

        // Halve cores across children. When num_cores_here drops to 1 we
        // still recurse with (1,1) so both subtrees keep a valid core, but
        // that legitimately oversubscribes — the earlier std::max(1,...)
        // on the INITIAL call from emit(...) already guarantees ≥1 here.
        const int half = std::max(1, num_cores_here / 2);
        const int rem  = std::max(1, num_cores_here - half);
        if (!emit(c1, cur_depth + 1, first_core, half)) return false;
        if (c2 >= 0) {
            if (!emit(c2, cur_depth + 1, first_core + half, rem)) return false;
        }
        NwayRegion r;
        r.start_col = rangtab[node];
        r.end_col = rangtab[node + 1];
        r.first_core = first_core;
        r.num_cores = std::max(1, num_cores_here);
        r.is_leaf = false;
        r.depth_level = cur_depth;
        r.label = (cur_depth == 0) ? std::string("S_root")
                                    : ("S_" + std::to_string(node));
        out.push_back(std::move(r));
        return true;
    };

    return emit(root, 0, 0, std::max(1, num_cores));
}

// Pad the three top-level SCOTCH regions (P1 | P2 | Sep) to tile_size — NOT
// every block. Walks the SCOTCH tree once, finds the root (treetab[b]==-1) and
// its two children, classifies each block's subtree membership, then adds at
// most `3 * (tile_size - 1)` padding rows total. Updates scheme.element_perm,
// element_iperm, L_colptr, L_rowind, tile state, COO arrays in `call`,
// scheme.scotch_rangtab (entries shifted by their region's cumulative pad so
// they still point to valid positions in the padded permuted space), and
// scheme.partition_sizes[3] — populating {P1_new, P2_new, Sep_new} so
// collect_tasks Path 2 engages.
//
// Dense-node tail (did_dense_perm upstream): columns [rangtab[cblknbr], dim)
// are not part of any SCOTCH block; they are folded into Sep (the last region)
// since they're already positioned after the root separator in permuted space.
//
// Tree layout assumption: SCOTCH emits blocks in postorder so subtrees are
// contiguous in rangtab index order, matching column-order in permuted space
// (left subtree columns first, then right subtree, then root separator).
// We verify by classifying blocks via treetab-walk (not index arithmetic) and
// checking that region sizes sum to scheme.dim.
static StatusCode apply_scotch_block_padding(
    sTiles_call& call, TiledMatrix& scheme,
    int group_index, int num_cores)
{
    const int ts = scheme.tile_size;
    if (ts <= 1 || scheme.scotch_cblknbr <= 0 ||
        !scheme.scotch_rangtab || !scheme.scotch_treetab || !scheme.element_perm) {
        return StatusCode::Success;
    }

    const int cb = scheme.scotch_cblknbr;
    const int old_dim = scheme.dim;
    const int* rangtab = scheme.scotch_rangtab;
    const int* treetab = scheme.scotch_treetab;

    if (rangtab[0] != 0 || rangtab[cb] > old_dim) {
        sTiles::Logger::warning("│   ↪ SCOTCH block padding: rangtab bounds out of range, skipping");
        return StatusCode::Success;
    }

    // --- Walk the SCOTCH tree to find root and its children. ---
    int root = -1;
    for (int b = 0; b < cb; ++b) {
        if (treetab[b] == -1) {
            if (root != -1) {
                sTiles::Logger::warning("│   ↪ SCOTCH block padding: multiple roots, skipping");
                return StatusCode::Success;
            }
            root = b;
        }
    }
    if (root < 0) {
        sTiles::Logger::warning("│   ↪ SCOTCH block padding: no root found, skipping");
        return StatusCode::Success;
    }

    // --- Determine the three region boundaries in OLD permuted space. ---
    // p1_end_old = end of left-subtree columns
    // p2_end_old = end of right-subtree columns
    // Everything from p2_end_old to old_dim (root block + optional dense tail)
    // goes to Sep.
    int p1_end_old = 0;
    int p2_end_old = 0;

    if (cb == 1) {
        // Single-block tree: whole matrix is one separator. P1 = P2 = 0.
        p1_end_old = 0;
        p2_end_old = 0;
    } else {
        // Find root's children.
        int c1 = -1, c2 = -1;
        bool too_many_children = false;
        for (int b = 0; b < cb; ++b) {
            if (b == root || treetab[b] != root) continue;
            if      (c1 == -1) c1 = b;
            else if (c2 == -1) c2 = b;
            else { too_many_children = true; break; }
        }
        if (too_many_children) {
            sTiles::Logger::warning("│   ↪ SCOTCH block padding: root has >2 children, skipping");
            return StatusCode::Success;
        }
        if (c1 == -1) {
            sTiles::Logger::warning("│   ↪ SCOTCH block padding: root has no children, skipping");
            return StatusCode::Success;
        }
        if (c2 == -1) {
            // Only one child — degenerate tree. Everything below root is P1.
            p1_end_old = rangtab[root];
            p2_end_old = rangtab[root];
        } else {
            if (c1 > c2) std::swap(c1, c2);
            p1_end_old = rangtab[c1 + 1];
            p2_end_old = rangtab[c2 + 1];
            // Postorder sanity check: root's column range should start exactly
            // where c2's subtree ends.
            if (rangtab[root] != p2_end_old) {
                sTiles::Logger::warning(
                    "│   ↪ SCOTCH block padding: postorder layout violated "
                    "(rangtab[root]=" + std::to_string(rangtab[root]) +
                    " != p2_end_old=" + std::to_string(p2_end_old) +
                    "), skipping");
                return StatusCode::Success;
            }
        }
    }

    const int sep_end_old = old_dim;  // Sep includes root block + any dense tail.

    // Raw region sizes (before padding).
    const int P1_raw  = p1_end_old;
    const int P2_raw  = p2_end_old - p1_end_old;
    const int Sep_raw = sep_end_old - p2_end_old;
    if (P1_raw + P2_raw + Sep_raw != old_dim) {
        sTiles::Logger::warning(
            "│   ↪ SCOTCH block padding: regions don't sum to dim, skipping");
        return StatusCode::Success;
    }

    // Per-region pad — at most ts-1 each, so total_pad ≤ 3*(ts-1).
    const int pad_P1  = (P1_raw  == 0 || P1_raw  % ts == 0) ? 0 : (ts - P1_raw  % ts);
    const int pad_P2  = (P2_raw  == 0 || P2_raw  % ts == 0) ? 0 : (ts - P2_raw  % ts);
    const int pad_Sep = (Sep_raw == 0 || Sep_raw % ts == 0) ? 0 : (ts - Sep_raw % ts);
    const int total_pad = pad_P1 + pad_P2 + pad_Sep;

    const int P1_new  = P1_raw  + pad_P1;
    const int P2_new  = P2_raw  + pad_P2;
    const int Sep_new = Sep_raw + pad_Sep;

    // Helper to populate partition_sizes (shared between zero-pad early return
    // and the full path below).
    auto install_partitions = [&]() {
        if (scheme.partition_sizes) OrderingMemoryManager::deallocate(scheme.partition_sizes);
        scheme.partition_sizes = OrderingMemoryManager::allocate<int>(3, group_index);
        if (scheme.partition_sizes) {
            scheme.partition_sizes[0] = P1_new;
            scheme.partition_sizes[1] = P2_new;
            scheme.partition_sizes[2] = Sep_new;
            sTiles::Logger::timing("│   ↪ SCOTCH top-level partitions: P1="
                + std::to_string(P1_new) + ", P2=" + std::to_string(P2_new)
                + ", Sep=" + std::to_string(Sep_new)
                + " (total=" + std::to_string(P1_new + P2_new + Sep_new) + ")");
            // Option A composed-path enablement (3-way fallback). P1/P2/Sep are
            // each padded to a tile multiple, so the root separator (Sep) is the
            // last grid-tile run. ProcessCore synthesizes [P1,P2,Sep] when
            // num_partitions==0 and treats Sep as the root. The structural sep
            // (= Sep_new/ts) is handed to corner_probe; tree fires only if it
            // passes the <=max_sep + heavy gates, else Sep stays a normal ND part.
            const bool three_way_ok = (P1_new % ts == 0) && (P2_new % ts == 0)
                                    && (Sep_new % ts == 0) && (Sep_new > 0);
            scheme.scotch_root_sep_tiles       = three_way_ok ? (Sep_new / ts) : 0;
            // Gate 1 (see N-way path): >= C_min cores AND each leaf (P1,P2) has
            // >= K tiles. Env-tunable for the sweep. Below threshold → composed
            // path off → flat / pure-ND fallback.
            int g_cmin = 4, g_k = 8;
            if (const char* e = std::getenv("STILES_PART_MIN_CORES"))          { const int v = std::atoi(e); if (v > 0) g_cmin = v; }
            if (const char* e = std::getenv("STILES_PART_MIN_TILES_PER_LEAF")) { const int v = std::atoi(e); if (v > 0) g_k    = v; }
            const int min_leaf_t = std::min(P1_new / ts, P2_new / ts);
            const bool gate1 = (num_cores >= g_cmin) && (min_leaf_t >= g_k);
            scheme.scotch_partition_collection = three_way_ok && gate1;

            // ND-independence scan (3-way P1|P2|Sep) over the padded factor L.
            // The two leaf partitions P1 and P2 must have ZERO coupling
            // (the separator Sep absorbs all cross-leaf fill); P1/P2 may only
            // couple UP to Sep. P2<-P1 == 0  ⇒  the two partitions are
            // independent. (new_dim == P1_new+P2_new+Sep_new here.)
            if (std::getenv("STILES_PAD_DUMP") && scheme.L_colptr && scheme.L_rowind) {
                const int nd  = P1_new + P2_new + Sep_new;
                const int p1e = P1_new;             // P1 = [0, p1e)
                const int p2e = P1_new + P2_new;    // P2 = [p1e, p2e), Sep = [p2e, nd)
                long long p2p1 = 0, sepp1 = 0, sepp2 = 0;
                for (int c = 0; c < p1e; ++c)
                    for (int p = scheme.L_colptr[c]; p < scheme.L_colptr[c + 1]; ++p) {
                        const int r = scheme.L_rowind[p];
                        if      (r >= p1e && r < p2e) ++p2p1;
                        else if (r >= p2e)            ++sepp1;
                    }
                for (int c = p1e; c < p2e; ++c)
                    for (int p = scheme.L_colptr[c]; p < scheme.L_colptr[c + 1]; ++p)
                        if (scheme.L_rowind[p] >= p2e) ++sepp2;
                std::fprintf(stderr,
                    "[PAD_INDEP/3way] ts=%d nd=%d P1=[0,%d) P2=[%d,%d) Sep=[%d,%d): "
                    "P2<-P1=%lld %s | Sep<-P1=%lld Sep<-P2=%lld => %s\n",
                    ts, nd, p1e, p1e, p2e, p2e, nd, p2p1,
                    p2p1 == 0 ? "" : "**LEAF-LEAF: NOT INDEPENDENT**", sepp1, sepp2,
                    p2p1 == 0 ? "INDEPENDENT" : "NOT INDEPENDENT");
            }
        }
    };

    if (total_pad == 0) {
        sTiles::Logger::timing("│   ↪ SCOTCH block padding: all 3 top-level partitions already tile-aligned");
        scheme.nd_padding = 0;
        install_partitions();
        return StatusCode::Success;
    }

    const int new_dim = old_dim + total_pad;
    const int new_nnz = scheme.nnz + total_pad;

    // Position shift: each old permuted position shifts by its region's cumulative pad.
    // - p < p1_end_old:           shift = 0          (stays in [0, P1_raw))
    // - p1_end_old ≤ p < p2_end_old: shift = pad_P1  (P2 region, right after P1 pad)
    // - p2_end_old ≤ p < old_dim:   shift = pad_P1 + pad_P2 (Sep region)
    std::vector<int> pos_shift(old_dim);
    for (int p = 0; p < old_dim; ++p) {
        int shift;
        if      (p < p1_end_old) shift = 0;
        else if (p < p2_end_old) shift = pad_P1;
        else                     shift = pad_P1 + pad_P2;
        pos_shift[p] = p + shift;
    }

    // Allocate extended perm/iperm.
    int* new_perm  = OrderingMemoryManager::allocate<int>(new_dim, group_index);
    int* new_iperm = OrderingMemoryManager::allocate<int>(new_dim, group_index);
    if (!new_perm || !new_iperm) {
        OrderingMemoryManager::deallocate(new_perm);
        OrderingMemoryManager::deallocate(new_iperm);
        return StatusCode::OutOfResources;
    }

    // Shift original indices.
    for (int i = 0; i < old_dim; ++i)
        new_perm[i] = pos_shift[scheme.element_perm[i]];

    // Assign pad slots: pad rows go at the tail of each region in permuted space.
    //   P1 pad rows: [P1_raw, P1_new)
    //   P2 pad rows: [P1_new + P2_raw, P1_new + P2_new)
    //   Sep pad rows: [P1_new + P2_new + Sep_raw, new_dim)
    {
        int pad_idx = old_dim;
        for (int k = 0; k < pad_P1; ++k)  new_perm[pad_idx++] = P1_raw + k;
        for (int k = 0; k < pad_P2; ++k)  new_perm[pad_idx++] = P1_new + P2_raw + k;
        for (int k = 0; k < pad_Sep; ++k) new_perm[pad_idx++] = P1_new + P2_new + Sep_raw + k;
    }

    compute_inverse_permutation(new_perm, new_iperm, new_dim);
    if (!check_inverse_permutation(new_perm, new_iperm, new_dim)) {
        OrderingMemoryManager::deallocate(new_perm);
        OrderingMemoryManager::deallocate(new_iperm);
        sTiles::Logger::error("│   ↪ SCOTCH block padding: extended permutation invalid");
        return StatusCode::IllegalValue;
    }

    // Build padded COO: original entries + diagonal entries for pad slots.
    int* new_rows = static_cast<int*>(std::malloc(new_nnz * sizeof(int)));
    int* new_cols = static_cast<int*>(std::malloc(new_nnz * sizeof(int)));
    if (!new_rows || !new_cols) {
        std::free(new_rows); std::free(new_cols);
        OrderingMemoryManager::deallocate(new_perm);
        OrderingMemoryManager::deallocate(new_iperm);
        return StatusCode::OutOfResources;
    }
    std::copy(call.row_indices, call.row_indices + scheme.nnz, new_rows);
    std::copy(call.col_indices, call.col_indices + scheme.nnz, new_cols);
    for (int k = 0; k < total_pad; ++k) {
        new_rows[scheme.nnz + k] = old_dim + k;
        new_cols[scheme.nnz + k] = old_dim + k;
    }

    // Symbolic Cholesky fill-in on the padded, permuted lower-triangle CSC.
    std::vector<int> colptr(new_dim + 1, 0);
    for (int k = 0; k < new_nnz; ++k) {
        int nr = new_perm[new_rows[k]];
        int nc = new_perm[new_cols[k]];
        if (nr < nc) std::swap(nr, nc);
        ++colptr[nc + 1];
    }
    for (int j = 0; j < new_dim; ++j) colptr[j + 1] += colptr[j];
    std::vector<int> rowind(colptr[new_dim]);
    {
        std::vector<int> pos(colptr.begin(), colptr.begin() + new_dim);
        for (int k = 0; k < new_nnz; ++k) {
            int nr = new_perm[new_rows[k]];
            int nc = new_perm[new_cols[k]];
            if (nr < nc) std::swap(nr, nc);
            rowind[pos[nc]++] = nr;
        }
    }
    for (int j = 0; j < new_dim; ++j)
        std::sort(rowind.begin() + colptr[j], rowind.begin() + colptr[j + 1]);
    std::vector<int> cp, ri;
    const int new_nnzL = sTiles::symbolic_chol_fillin(
        new_dim, colptr, rowind, cp, ri, 2'000'000'000UL);
    if (new_nnzL < 0) {
        std::free(new_rows); std::free(new_cols);
        OrderingMemoryManager::deallocate(new_perm);
        OrderingMemoryManager::deallocate(new_iperm);
        sTiles::Logger::error("│   ↪ SCOTCH block padding: symbolic_chol_fillin exceeded limit");
        return StatusCode::ExecutionFailed;
    }

    // Count active tiles on L.
    std::vector<int> L_row(new_nnzL), L_col(new_nnzL);
    for (int j = 0; j < new_dim; ++j)
        for (int p = cp[j]; p < cp[j + 1]; ++p) {
            L_row[p] = ri[p];
            L_col[p] = j;
        }
    const int new_grid = (new_dim + ts - 1) / ts;
    TileIndexer::State new_state;
    int new_tiles = TileIndexer::countActiveTiles(
        L_row.data(), L_col.data(), new_nnzL,
        new_dim, ts, scheme.neighbor_lookup_method, &new_state, num_cores);
    if (new_tiles < 0) {
        std::free(new_rows); std::free(new_cols);
        OrderingMemoryManager::deallocate(new_perm);
        OrderingMemoryManager::deallocate(new_iperm);
        sTiles::Logger::error("│   ↪ SCOTCH block padding: countActiveTiles failed on padded L");
        return StatusCode::ExecutionFailed;
    }
    TileIndexer::ensure_diagonal_tiles_active(new_state,
        scheme.neighbor_lookup_method, new_grid, new_tiles);

    // Replace scheme's existing structures.
    OrderingMemoryManager::deallocate(scheme.element_perm);
    OrderingMemoryManager::deallocate(scheme.element_iperm);
    OrderingMemoryManager::deallocate(scheme.L_colptr);
    OrderingMemoryManager::deallocate(scheme.L_rowind);
    TileIndexer::release_state_resources(scheme.state);

    scheme.element_perm  = new_perm;
    scheme.element_iperm = new_iperm;
    scheme.L_colptr = OrderingMemoryManager::allocate<int>(new_dim + 1, group_index);
    scheme.L_rowind = OrderingMemoryManager::allocate<int>(new_nnzL, group_index);
    if (scheme.L_colptr && scheme.L_rowind) {
        std::copy(cp.begin(), cp.end(), scheme.L_colptr);
        std::copy(ri.begin(), ri.end(), scheme.L_rowind);
    }
    scheme.nnz_factor     = static_cast<long long>(new_nnzL);
    scheme.state          = std::move(new_state);
    scheme.numActiveTiles = new_tiles;
    scheme.dim            = new_dim;
    scheme.nnz            = new_nnz;
    scheme.dimTiledMatrix = new_grid;
    scheme.original_order = new_dim;
    scheme.original_nnz   = new_nnz;
    scheme.nd_padding     = total_pad;

    // Update scotch_rangtab: shift each entry by its region's cumulative pad so
    // it still points to a valid position in the padded permuted space. We use
    // the same thresholds as pos_shift (strict-less-than), so boundary entries
    // at p1_end_old / p2_end_old map to the start of the NEXT region in the
    // padded space, matching the invariant rangtab[b]_new = rangtab[b]_old + pad.
    for (int b = 0; b <= cb; ++b) {
        const int r = rangtab[b];
        int shift;
        if      (r < p1_end_old) shift = 0;
        else if (r < p2_end_old) shift = pad_P1;
        else                     shift = pad_P1 + pad_P2;
        scheme.scotch_rangtab[b] = r + shift;
    }

    // Swap call's COO pointers. Old pointers stay owned by the caller.
    call.row_indices = new_rows;
    call.col_indices = new_cols;
    call.order       = new_dim;
    call.nnz         = new_nnz;

    sTiles::Logger::timing("│   ↪ SCOTCH top-level padding applied: pad_P1="
        + std::to_string(pad_P1) + ", pad_P2=" + std::to_string(pad_P2)
        + ", pad_Sep=" + std::to_string(pad_Sep)
        + " (total=" + std::to_string(total_pad) + ")"
        + ", new_dim=" + std::to_string(new_dim)
        + ", nnz(L)=" + std::to_string(new_nnzL)
        + ", tiles(L)=" + std::to_string(new_tiles));

    install_partitions();

    return StatusCode::Success;
}

// =============================================================================
// N-way depth-D padding. Mirrors apply_scotch_block_padding but iterates over
// 2^(D+1)-1 regions from build_nway_regions instead of 3 hardcoded regions.
// Pads each region boundary to tile_size, populates scheme.partitions with
// tile-aligned PartitionDesc entries, and also writes partition_sizes[3] as
// a top-level summary (left-half, right-half, root-sep) for backward compat.
//
// Control: target_depth >= 2; num_cores_for_regions = cores to distribute
// across leaves. The cores field in scheme.partitions is derived by the
// walker (halves per level down).
// =============================================================================
static StatusCode apply_scotch_block_padding_nway(
    sTiles_call& call, TiledMatrix& scheme,
    int group_index, int num_cores,
    int target_depth)
{
    const int ts = scheme.tile_size;
    if (ts <= 1 || scheme.scotch_cblknbr <= 0 ||
        !scheme.scotch_rangtab || !scheme.scotch_treetab || !scheme.element_perm) {
        return StatusCode::Success;
    }
    const int cb = scheme.scotch_cblknbr;
    const int old_dim = scheme.dim;

    // Cap target_depth at floor(log2(num_cores)) — the walker halves cores
    // at each level and clamps to 1 at the bottom, so deeper than this
    // causes adjacent leaves to share core indices (work overlap).
    int effective_depth = target_depth;
    if (num_cores > 0) {
        int log2_cores = 0;
        for (int v = num_cores; v > 1; v >>= 1) ++log2_cores;
        if (effective_depth > log2_cores) {
            sTiles::Logger::timing("│   ↪ SCOTCH N-way padding: target_depth=" +
                std::to_string(target_depth) + " exceeds log2(cores=" +
                std::to_string(num_cores) + ")=" + std::to_string(log2_cores) +
                "; clamping to " + std::to_string(log2_cores));
            effective_depth = log2_cores;
        }
    }
    if (effective_depth < 2) {
        return apply_scotch_block_padding(call, scheme, group_index, num_cores);
    }

    // 1. Get the region layout from the tree walker.
    std::vector<NwayRegion> regions;
    if (!build_nway_regions(scheme, effective_depth, num_cores, regions) || regions.empty()) {
        sTiles::Logger::warning("│   ↪ SCOTCH N-way padding: build_nway_regions failed; "
            "falling back to legacy 3-way path");
        return apply_scotch_block_padding(call, scheme, group_index, num_cores);
    }

    // 2. Regions from the walker cover SCOTCH-tracked columns. Any dense-node
    // tail (columns after rangtab[cb]) gets folded into the LAST region (root
    // separator) so P1+...+Sep = old_dim.
    if (regions.back().end_col < old_dim) {
        regions.back().end_col = old_dim;
    }

    // 3. Verify regions cover [0, old_dim) contiguously without gaps.
    int expected_start = 0;
    for (const auto& r : regions) {
        if (r.start_col != expected_start || r.end_col < r.start_col) {
            sTiles::Logger::warning("│   ↪ SCOTCH N-way padding: region layout has gaps, "
                "falling back to legacy 3-way");
            return apply_scotch_block_padding(call, scheme, group_index, num_cores);
        }
        expected_start = r.end_col;
    }
    if (expected_start != old_dim) {
        sTiles::Logger::warning("│   ↪ SCOTCH N-way padding: regions don't sum to dim, "
            "falling back to legacy 3-way");
        return apply_scotch_block_padding(call, scheme, group_index, num_cores);
    }

    const int N = static_cast<int>(regions.size());

    // 4. Compute per-region pad (at most ts-1 each) and cumulative pad offsets.
    std::vector<int> region_pad(N, 0);
    std::vector<int> cum_pad_before(N, 0);
    int total_pad = 0;
    for (int r = 0; r < N; ++r) {
        const int sz = regions[r].end_col - regions[r].start_col;
        const int rem = (sz <= 0) ? 0 : sz % ts;
        region_pad[r] = (rem == 0) ? 0 : (ts - rem);
        cum_pad_before[r] = total_pad;
        total_pad += region_pad[r];
    }
    const int new_dim = old_dim + total_pad;
    const int new_nnz = scheme.nnz + total_pad;

    // 5. Populate scheme.partitions (tile-aligned new coords) and summary.
    auto install_n_partitions = [&]() {
        scheme.partitions.clear();
        scheme.partitions.reserve(static_cast<std::size_t>(N));
        for (int r = 0; r < N; ++r) {
            const int start_new = regions[r].start_col + cum_pad_before[r];
            const int raw_sz    = regions[r].end_col - regions[r].start_col;
            const int padded_sz = raw_sz + region_pad[r];
            TiledMatrix::PartitionDesc pd;
            pd.start_tile = start_new / ts;
            pd.end_tile   = (start_new + padded_sz) / ts;
            pd.first_core = regions[r].first_core;
            pd.num_cores  = regions[r].num_cores;
            // label strings are stored owned in NwayRegion; scheme.partitions
            // stores a pointer, so we need stable storage. Stash the strings
            // on the scheme itself via a small side-car… for simplicity, use
            // the raw C string from a static buffer via the label itself —
            // it's safe because regions' strings are kept alive for the
            // duration of this call and we only log them here.
            // For robustness, store persistent labels by hashing depth+index
            // into a small pool. Keep it simple: use strdup-like alloc into
            // OrderingMemoryManager? Overkill. Instead, set pd.label to a
            // static literal based on depth_level/is_leaf; this is
            // diagnostic-only anyway.
            pd.label = regions[r].is_leaf ? "L" : "S";
            scheme.partitions.push_back(pd);
        }
        scheme.num_partitions = N;

        // ---- Precision verification + Option A composed-path enablement ----
        // The composed ND+tree scheduler requires every partition to be
        // tile-exact, contiguous over [0, grid), and core-bounded within the
        // walker's [0, num_cores) range. Only then is
        //   cutoff = grid - root_sep_tiles
        // well-defined and Region A (the non-root partitions) tiles [0, cutoff)
        // with no gap/overlap, and no region's tasks get dropped because a core
        // index exceeds the executor's rank count.
        const int grid_tiles = (new_dim + ts - 1) / ts;   // padded grid (new_dim % ts == 0)
        bool layout_ok = (N >= 1);
        int expect_tile = 0;
        for (int r = 0; r < N && layout_ok; ++r) {
            const auto& pd = scheme.partitions[static_cast<std::size_t>(r)];
            if (pd.start_tile != expect_tile || pd.end_tile < pd.start_tile) layout_ok = false;
            if (pd.first_core < 0 || pd.first_core + pd.num_cores > num_cores) layout_ok = false;
            expect_tile = pd.end_tile;
        }
        if (layout_ok && expect_tile != grid_tiles) layout_ok = false;

        // Root separator = last (S_root) partition. Its tile span is the
        // structural `sep` candidate handed to corner_probe (Option A).
        const int root_sep_tiles = layout_ok
            ? (scheme.partitions.back().end_tile - scheme.partitions.back().start_tile)
            : 0;
        scheme.scotch_root_sep_tiles       = root_sep_tiles;

        // Gate 1: only engage the composed ND+tree collection when there is
        // enough parallelism to amortize the per-partition collection — at
        // least C_min cores AND each leaf carries at least K tiles (so a leaf's
        // core has real work). Env-tunable for the threshold sweep:
        //   STILES_PART_MIN_CORES           (default 4)
        //   STILES_PART_MIN_TILES_PER_LEAF  (default 8)
        // Below threshold the flag stays off → composed path declines and the
        // scheme falls back to the existing flat / pure-ND scheduling.
        int g_cmin = 4, g_k = 8;
        if (const char* e = std::getenv("STILES_PART_MIN_CORES"))          { const int v = std::atoi(e); if (v > 0) g_cmin = v; }
        if (const char* e = std::getenv("STILES_PART_MIN_TILES_PER_LEAF")) { const int v = std::atoi(e); if (v > 0) g_k    = v; }
        int n_leaves = 0, min_leaf_t = grid_tiles + 1;
        if (layout_ok) {
            for (int r = 0; r < N; ++r) if (regions[r].is_leaf) {
                ++n_leaves;
                const int lt = scheme.partitions[static_cast<std::size_t>(r)].end_tile
                             - scheme.partitions[static_cast<std::size_t>(r)].start_tile;
                if (lt < min_leaf_t) min_leaf_t = lt;
            }
        }
        const bool gate1 = (num_cores >= g_cmin) && (n_leaves > 0) && (min_leaf_t >= g_k);
        scheme.scotch_partition_collection = layout_ok && (root_sep_tiles > 0) && gate1;
        if (!layout_ok) {
            sTiles::Logger::warning("│   ↪ SCOTCH N-way: partition layout not tile-exact / "
                "core-bounded; composed ND+tree collection disabled");
        } else if (!gate1) {
            sTiles::Logger::timing("│   ↪ SCOTCH N-way: Gate 1 not met (cores=" + std::to_string(num_cores)
                + " < " + std::to_string(g_cmin) + " or min_leaf_tiles=" + std::to_string(min_leaf_t)
                + " < " + std::to_string(g_k) + ") → composed path off (flat/pure-ND fallback)");
        }

        // Env-gated split dump: prints every partition's padded tile boundary
        // with the raw element start/size and their `% ts` residuals, so the
        // tile-alignment of the split can be verified directly across tile
        // sizes. A correct split has start_new%%ts == 0 and padded_sz%%ts == 0
        // for EVERY region (⇒ contiguous, gap-free tile partition of [0,grid)).
        if (std::getenv("STILES_PAD_DUMP")) {
            std::fprintf(stderr,
                "[PAD_DUMP] ts=%d old_dim=%d new_dim=%d total_pad=%d grid=%d "
                "root_sep_tiles=%d layout_ok=%d collect=%d N=%d\n",
                ts, old_dim, new_dim, total_pad, grid_tiles, root_sep_tiles,
                (int)layout_ok, (int)scheme.scotch_partition_collection, N);
            for (int r = 0; r < N; ++r) {
                const int start_new = regions[r].start_col + cum_pad_before[r];
                const int raw_sz    = regions[r].end_col - regions[r].start_col;
                const int padded_sz = raw_sz + region_pad[r];
                std::fprintf(stderr,
                    "[PAD_DUMP]   %-2d %s tiles[%d,%d) span=%d  raw=%d pad=%d "
                    "cores[%d,%d)  start%%ts=%d size%%ts=%d %s\n",
                    r, regions[r].is_leaf ? "L" : "S",
                    scheme.partitions[r].start_tile, scheme.partitions[r].end_tile,
                    scheme.partitions[r].end_tile - scheme.partitions[r].start_tile,
                    raw_sz, region_pad[r],
                    scheme.partitions[r].first_core,
                    scheme.partitions[r].first_core + scheme.partitions[r].num_cores,
                    start_new % ts, padded_sz % ts,
                    (start_new % ts == 0 && padded_sz % ts == 0) ? "OK" : "**MISALIGNED**");
            }

            // ND-independence scan over the padded factor L (element-level,
            // permuted space — scheme.L_colptr/L_rowind are the exact symbolic
            // Cholesky fill computed just above, so this is the definitive test).
            // For partition i (columns) vs later partition j (rows), count L
            // nonzeros with col ∈ part-i element range and row ∈ part-j element
            // range. Sibling/cousin LEAF partitions MUST have ZERO such coupling
            // (the ND separators absorb all cross-leaf fill); a leaf may only
            // couple UP to its ancestor separators. Any leaf↔leaf coupling means
            // the partitions are NOT independent.
            if (scheme.L_colptr && scheme.L_rowind) {
                auto erange = [&](int p, int& e0, int& e1) {
                    e0 = scheme.partitions[p].start_tile * ts;
                    e1 = std::min(scheme.partitions[p].end_tile * ts, new_dim);
                };
                int leaf_leaf_violations = 0;
                std::fprintf(stderr, "[PAD_INDEP] cross-partition L coupling (later<-earlier, nnz):\n");
                for (int j = 0; j < N; ++j) {
                    int rj0, rj1; erange(j, rj0, rj1);
                    for (int i = 0; i < j; ++i) {
                        int ci0, ci1; erange(i, ci0, ci1);
                        long long cnt = 0;
                        for (int c = ci0; c < ci1; ++c)
                            for (int p = scheme.L_colptr[c]; p < scheme.L_colptr[c + 1]; ++p) {
                                const int r = scheme.L_rowind[p];
                                if (r >= rj0 && r < rj1) ++cnt;
                            }
                        if (cnt > 0) {
                            const bool leaf_leaf = regions[i].is_leaf && regions[j].is_leaf;
                            if (leaf_leaf) ++leaf_leaf_violations;
                            std::fprintf(stderr,
                                "[PAD_INDEP]   p%d(%s) <- p%d(%s): %lld %s\n",
                                j, regions[j].is_leaf ? "L" : "S",
                                i, regions[i].is_leaf ? "L" : "S", cnt,
                                leaf_leaf ? "**LEAF-LEAF: NOT INDEPENDENT**" : "(leaf->sep, ok)");
                        }
                    }
                }
                std::fprintf(stderr, "[PAD_INDEP] leaf-leaf coupling violations: %d  => partitions %s\n",
                    leaf_leaf_violations,
                    leaf_leaf_violations == 0 ? "INDEPENDENT" : "NOT INDEPENDENT");
            } else {
                std::fprintf(stderr, "[PAD_INDEP] L structure unavailable; independence scan skipped\n");
            }
        }

        // Legacy 3-way summary for backward-compat readers:
        // P1 = sum of region sizes in the FIRST half (leaves+seps before root)
        // P2 = sum of region sizes in the SECOND half
        // Sep = root separator alone (last entry)
        int P1_sum = 0, P2_sum = 0, Sep_sum = 0;
        const int root_idx = N - 1;
        const int mid = root_idx / 2;  // rough split
        for (int r = 0; r < root_idx; ++r) {
            const int padded_sz = (regions[r].end_col - regions[r].start_col) + region_pad[r];
            if (r <= mid) P1_sum += padded_sz;
            else          P2_sum += padded_sz;
        }
        Sep_sum = (regions[root_idx].end_col - regions[root_idx].start_col) + region_pad[root_idx];
        if (scheme.partition_sizes) OrderingMemoryManager::deallocate(scheme.partition_sizes);
        scheme.partition_sizes = OrderingMemoryManager::allocate<int>(3, group_index);
        if (scheme.partition_sizes) {
            scheme.partition_sizes[0] = P1_sum;
            scheme.partition_sizes[1] = P2_sum;
            scheme.partition_sizes[2] = Sep_sum;
        }

        // Per-region log.
        sTiles::Logger::timing("│   ↪ SCOTCH N-way partitions (depth=" +
            std::to_string(effective_depth) + ", regions=" + std::to_string(N) + "):");
        for (int r = 0; r < N; ++r) {
            const int raw_sz    = regions[r].end_col - regions[r].start_col;
            const int padded_sz = raw_sz + region_pad[r];
            sTiles::Logger::timing("│       " + regions[r].label +
                " size=" + std::to_string(padded_sz) +
                " (pad=" + std::to_string(region_pad[r]) + ")" +
                " cores=[" + std::to_string(regions[r].first_core) + ".." +
                std::to_string(regions[r].first_core + regions[r].num_cores) + ")");
        }
    };

    if (total_pad == 0) {
        scheme.nd_padding = 0;
        install_n_partitions();
        sTiles::Logger::timing("│   ↪ SCOTCH N-way padding: all regions already tile-aligned");
        return StatusCode::Success;
    }

    // 6. Position shift: old permuted position p sits in some region r;
    // its new position = p + cum_pad_before[r].
    std::vector<int> pos_shift(old_dim);
    {
        int r = 0;
        for (int p = 0; p < old_dim; ++p) {
            while (r < N && p >= regions[r].end_col) ++r;
            if (r >= N) {
                sTiles::Logger::error("│   ↪ SCOTCH N-way padding: position out of regions");
                return StatusCode::IllegalValue;
            }
            pos_shift[p] = p + cum_pad_before[r];
        }
    }

    // 7. Allocate extended perm/iperm.
    int* new_perm  = OrderingMemoryManager::allocate<int>(new_dim, group_index);
    int* new_iperm = OrderingMemoryManager::allocate<int>(new_dim, group_index);
    if (!new_perm || !new_iperm) {
        OrderingMemoryManager::deallocate(new_perm);
        OrderingMemoryManager::deallocate(new_iperm);
        return StatusCode::OutOfResources;
    }

    // 8. Shift original indices.
    for (int i = 0; i < old_dim; ++i)
        new_perm[i] = pos_shift[scheme.element_perm[i]];

    // 9. Assign pad slots per region: for each region with pad > 0, fill the
    // trailing [region_start_new + raw_sz, region_start_new + padded_sz) with
    // pad indices drawn from [old_dim, new_dim).
    {
        int pad_idx = old_dim;
        for (int r = 0; r < N; ++r) {
            if (region_pad[r] == 0) continue;
            const int start_new = regions[r].start_col + cum_pad_before[r];
            const int raw_sz    = regions[r].end_col - regions[r].start_col;
            for (int k = 0; k < region_pad[r]; ++k) {
                new_perm[pad_idx++] = start_new + raw_sz + k;
            }
        }
    }

    compute_inverse_permutation(new_perm, new_iperm, new_dim);
    if (!check_inverse_permutation(new_perm, new_iperm, new_dim)) {
        OrderingMemoryManager::deallocate(new_perm);
        OrderingMemoryManager::deallocate(new_iperm);
        sTiles::Logger::error("│   ↪ SCOTCH N-way padding: extended permutation invalid");
        return StatusCode::IllegalValue;
    }

    // 10. Build padded COO.
    int* new_rows = static_cast<int*>(std::malloc(new_nnz * sizeof(int)));
    int* new_cols = static_cast<int*>(std::malloc(new_nnz * sizeof(int)));
    if (!new_rows || !new_cols) {
        std::free(new_rows); std::free(new_cols);
        OrderingMemoryManager::deallocate(new_perm);
        OrderingMemoryManager::deallocate(new_iperm);
        return StatusCode::OutOfResources;
    }
    std::copy(call.row_indices, call.row_indices + scheme.nnz, new_rows);
    std::copy(call.col_indices, call.col_indices + scheme.nnz, new_cols);
    for (int k = 0; k < total_pad; ++k) {
        new_rows[scheme.nnz + k] = old_dim + k;
        new_cols[scheme.nnz + k] = old_dim + k;
    }

    // 11. Symbolic Cholesky fill-in on padded matrix.
    std::vector<int> colptr(new_dim + 1, 0);
    for (int k = 0; k < new_nnz; ++k) {
        int nr = new_perm[new_rows[k]];
        int nc = new_perm[new_cols[k]];
        if (nr < nc) std::swap(nr, nc);
        ++colptr[nc + 1];
    }
    for (int j = 0; j < new_dim; ++j) colptr[j + 1] += colptr[j];
    std::vector<int> rowind(colptr[new_dim]);
    {
        std::vector<int> pos(colptr.begin(), colptr.begin() + new_dim);
        for (int k = 0; k < new_nnz; ++k) {
            int nr = new_perm[new_rows[k]];
            int nc = new_perm[new_cols[k]];
            if (nr < nc) std::swap(nr, nc);
            rowind[pos[nc]++] = nr;
        }
    }
    for (int j = 0; j < new_dim; ++j)
        std::sort(rowind.begin() + colptr[j], rowind.begin() + colptr[j + 1]);
    std::vector<int> cp, ri;
    const int new_nnzL = sTiles::symbolic_chol_fillin(
        new_dim, colptr, rowind, cp, ri, 2'000'000'000UL);
    if (new_nnzL < 0) {
        std::free(new_rows); std::free(new_cols);
        OrderingMemoryManager::deallocate(new_perm);
        OrderingMemoryManager::deallocate(new_iperm);
        sTiles::Logger::error("│   ↪ SCOTCH N-way padding: symbolic_chol_fillin exceeded limit");
        return StatusCode::ExecutionFailed;
    }

    // 12. Count active tiles.
    std::vector<int> L_row(new_nnzL), L_col(new_nnzL);
    for (int j = 0; j < new_dim; ++j)
        for (int p = cp[j]; p < cp[j + 1]; ++p) {
            L_row[p] = ri[p];
            L_col[p] = j;
        }
    const int new_grid = (new_dim + ts - 1) / ts;
    TileIndexer::State new_state;
    int new_tiles = TileIndexer::countActiveTiles(
        L_row.data(), L_col.data(), new_nnzL,
        new_dim, ts, scheme.neighbor_lookup_method, &new_state, num_cores);
    if (new_tiles < 0) {
        std::free(new_rows); std::free(new_cols);
        OrderingMemoryManager::deallocate(new_perm);
        OrderingMemoryManager::deallocate(new_iperm);
        sTiles::Logger::error("│   ↪ SCOTCH N-way padding: countActiveTiles failed");
        return StatusCode::ExecutionFailed;
    }
    TileIndexer::ensure_diagonal_tiles_active(new_state,
        scheme.neighbor_lookup_method, new_grid, new_tiles);

    // 13. Install new structures on scheme.
    OrderingMemoryManager::deallocate(scheme.element_perm);
    OrderingMemoryManager::deallocate(scheme.element_iperm);
    OrderingMemoryManager::deallocate(scheme.L_colptr);
    OrderingMemoryManager::deallocate(scheme.L_rowind);
    TileIndexer::release_state_resources(scheme.state);

    scheme.element_perm  = new_perm;
    scheme.element_iperm = new_iperm;
    scheme.L_colptr = OrderingMemoryManager::allocate<int>(new_dim + 1, group_index);
    scheme.L_rowind = OrderingMemoryManager::allocate<int>(new_nnzL, group_index);
    if (scheme.L_colptr && scheme.L_rowind) {
        std::copy(cp.begin(), cp.end(), scheme.L_colptr);
        std::copy(ri.begin(), ri.end(), scheme.L_rowind);
    }
    scheme.nnz_factor     = static_cast<long long>(new_nnzL);
    scheme.state          = std::move(new_state);
    scheme.numActiveTiles = new_tiles;
    scheme.dim            = new_dim;
    scheme.nnz            = new_nnz;
    scheme.dimTiledMatrix = new_grid;
    scheme.original_order = new_dim;
    scheme.original_nnz   = new_nnz;
    scheme.nd_padding     = total_pad;

    // 14. Update scotch_rangtab with per-region cumulative shift.
    for (int b = 0; b <= cb; ++b) {
        const int val = scheme.scotch_rangtab[b];
        int r = 0;
        while (r < N && val >= regions[r].end_col) ++r;
        const int shift = (r < N) ? cum_pad_before[r] : total_pad;
        scheme.scotch_rangtab[b] = val + shift;
    }

    // 15. Swap call's COO pointers.
    call.row_indices = new_rows;
    call.col_indices = new_cols;
    call.order       = new_dim;
    call.nnz         = new_nnz;

    sTiles::Logger::timing("│   ↪ SCOTCH N-way padding applied (depth=" +
        std::to_string(effective_depth) + ", regions=" + std::to_string(N) +
        "): total_pad=" + std::to_string(total_pad) +
        ", new_dim=" + std::to_string(new_dim) +
        ", nnz(L)=" + std::to_string(new_nnzL) +
        ", tiles(L)=" + std::to_string(new_tiles));

    install_n_partitions();
    return StatusCode::Success;
}

StatusCode symbolic_phase(sTiles_call **call_info, TiledMatrix **Tmatrix, int group_index, int num_cores) {

    // Runtime-controllable toggles live in the global control-param array
    // (stiles_control_params[20], [21], [22]):
    //   param[20] = force SCOTCH winner (0=off default, 1=on)
    //               → sTiles_set_force_scotch_ordering()
    //   param[21] = enable top-level SCOTCH padding + partition_sizes
    //               → sTiles_set_scotch_padding()
    //   param[22] = Path 2 tree depth (0/1 = legacy 3-way; >=2 = N-way)
    //               → sTiles_set_path2_depth()
    // These replace the old STILES_FORCE_ORDERING / STILES_ENABLE_SCOTCH_PADDING
    // env vars — no set/unset dance needed across runs, just call the setter.
    const bool FORCE_SCOTCH          = (stiles_control_params[20] != 0);
    const bool ENABLE_SCOTCH_PADDING = (stiles_control_params[21] != 0);
    const int  PATH2_DEPTH           = stiles_control_params[22];

    const double t_symbolic_start = omp_get_wtime();
    if (!call_info || !*call_info || !Tmatrix || !*Tmatrix) {
        sTiles::Logger::error("preprocess_symbolic_phase: null argument(s)");
        return StatusCode::IllegalValue;
    }

    sTiles_call& call = **call_info;
    TiledMatrix& scheme = **Tmatrix;
    scheme.nd_padding = 0;
    scheme.L_colptr = nullptr;
    scheme.L_rowind = nullptr;

    // Check for user-provided permutation for this group (must be done before any ordering path)
    const TileIndexer::Method method = scheme.neighbor_lookup_method;
    const int* user_perm = get_user_permutation(group_index);
    const int user_perm_size = get_user_permutation_size(group_index);
    const bool user_perm_force = get_user_permutation_force(group_index);
    const bool have_user_perm = (user_perm && user_perm_size == scheme.dim);

    // ordered_state declared here so it is in scope for both early-return paths below.
    TileIndexer::State ordered_state;

    if (call.factorization_variant == 2) {
        if (call.ordering_strategy > 0) {
            sTiles::Logger::info("│   ↪ Dense factorization selected (variant=2); skipping ordering requests.");
        }
        call.ordering_strategy = 0;
        scheme.use_ordering = 0;
        scheme.nd_padding = 0;
        scheme.red_tree_separator_level = 0;
        return StatusCode::Success;
    }

    if (have_user_perm) {
        const double t1 = omp_get_wtime();
        sTiles::Logger::info(user_perm_force
            ? "│   ↪ Using forced user-provided permutation"
            : "│   ↪ Using user-provided permutation");

        // Allocate and copy user permutation
        scheme.element_perm = OrderingMemoryManager::allocate<int>(scheme.dim, group_index);
        scheme.element_iperm = OrderingMemoryManager::allocate<int>(scheme.dim, group_index);

        if (!scheme.element_perm || !scheme.element_iperm) {
            sTiles::Logger::error("Failed to allocate memory for user permutation");
            return StatusCode::OutOfResources;
        }

        std::copy(user_perm, user_perm + scheme.dim, scheme.element_perm);
        compute_inverse_permutation(scheme.element_perm, scheme.element_iperm, scheme.dim);

        // Validate permutation
        if (!check_inverse_permutation(scheme.element_perm, scheme.element_iperm, scheme.dim)) {
            sTiles::Logger::error("Invalid user permutation: not a valid permutation");
            OrderingMemoryManager::deallocate(scheme.element_perm);
            OrderingMemoryManager::deallocate(scheme.element_iperm);
            return StatusCode::IllegalValue;
        }

        // Exact etree fill-in under user permutation
        int user_active = -1;
        {
            const int n_u = scheme.dim;
            std::vector<int> colptr_u(n_u + 1, 0);
            for (int k = 0; k < scheme.nnz; ++k) {
                int nr = scheme.element_perm[call.row_indices[k]];
                int nc = scheme.element_perm[call.col_indices[k]];
                if (nr < nc) std::swap(nr, nc);
                ++colptr_u[nc + 1];
            }
            for (int j = 0; j < n_u; ++j) colptr_u[j + 1] += colptr_u[j];
            std::vector<int> rowind_u(colptr_u[n_u]);
            {
                std::vector<int> pos(colptr_u.begin(), colptr_u.begin() + n_u);
                for (int k = 0; k < scheme.nnz; ++k) {
                    int nr = scheme.element_perm[call.row_indices[k]];
                    int nc = scheme.element_perm[call.col_indices[k]];
                    if (nr < nc) std::swap(nr, nc);
                    rowind_u[pos[nc]++] = nr;
                }
            }
            for (int j = 0; j < n_u; ++j)
                std::sort(rowind_u.begin() + colptr_u[j], rowind_u.begin() + colptr_u[j + 1]);
            std::vector<int> cp_u, ri_u;
            const int nnzL_u = sTiles::symbolic_chol_fillin(n_u, colptr_u, rowind_u, cp_u, ri_u, 2'000'000'000UL);
            if (nnzL_u >= 0) {
                std::vector<int> L_row(nnzL_u), L_col(nnzL_u);
                for (int j = 0; j < n_u; ++j)
                    for (int p = cp_u[j]; p < cp_u[j + 1]; ++p) {
                        L_row[p] = ri_u[p];
                        L_col[p] = j;
                    }
                int total = TileIndexer::countActiveTiles(
                    L_row.data(), L_col.data(), nnzL_u,
                    n_u, scheme.tile_size, method, &ordered_state, num_cores);
                if (total >= 0) {
                    TileIndexer::ensure_diagonal_tiles_active(ordered_state, method,
                        scheme.dimTiledMatrix, total);
                    user_active = total;
                    scheme.nnz_factor = static_cast<long long>(nnzL_u);
                    scheme.L_colptr = OrderingMemoryManager::allocate<int>(n_u + 1, group_index);
                    scheme.L_rowind = OrderingMemoryManager::allocate<int>(nnzL_u, group_index);
                    std::copy(cp_u.begin(), cp_u.end(), scheme.L_colptr);
                    std::copy(ri_u.begin(), ri_u.end(), scheme.L_rowind);
                }
            } else {
                sTiles::Logger::error(
                    "ERROR: symbolic_chol_fillin exceeded 2B nnz limit for user permutation"
                    " (dim=", n_u, "). Matrix fill-in is too large.");
                OrderingMemoryManager::deallocate(scheme.element_perm);
                OrderingMemoryManager::deallocate(scheme.element_iperm);
                return StatusCode::ExecutionFailed;
            }
        }

        if (user_active < 0) {
            sTiles::Logger::error("Failed to count tiles with user permutation");
            OrderingMemoryManager::deallocate(scheme.element_perm);
            OrderingMemoryManager::deallocate(scheme.element_iperm);
            return StatusCode::ExecutionFailed;
        }

        sTiles::Logger::timing("│   ↪ Symbolic factorization (user permutation): tiles(L)="
                             + std::to_string(user_active)
                             + ", time=" + sTiles::format_seconds(omp_get_wtime() - t1) + " s");

        scheme.state            = std::move(ordered_state);
        scheme.numActiveTiles   = user_active;
        scheme.use_ordering     = 1;
        scheme.selected_ordering = "user_permutation";
        scheme.red_tree_separator_level = 0;
        scheme.nd_padding       = 0;
        return StatusCode::Success;
    }

    // --- Dense node preprocessing ---
    // Detect nodes with degree > sqrt(nnz) and move them to the end of the
    // ordering. This prevents catastrophic fill-in from high-degree nodes
    // (e.g. global hyperparameters in hierarchical models) regardless of which
    // ordering is used later.
    // Non-destructive: a filtered subgraph COO is built and passed to orderings;
    // the caller's COO data is never modified.
    // sub2old_vec[sub_idx] = original_node  (for the non-dense subgraph)
    // dense_orig_vec[j]    = original dense node j  (appended at end of final perm)
    std::vector<int> sub2old_vec;
    std::vector<int> dense_orig_vec;
    std::vector<int> sub_rows_storage, sub_cols_storage;  // subgraph COO (lives until composition)
    bool did_dense_perm = false;
    int* orig_row_ptr = nullptr;
    int* orig_col_ptr = nullptr;
    int  orig_nnz     = 0;
    int  mean_deg     = 0;  // hoisted for identity evaluation gate

    if (scheme.fixed_column_size == 0) {
        const int dim = scheme.dim;
        const int nnz_val = scheme.nnz;

        // Compute per-node degree (symmetric: count both endpoints of each edge)
        std::vector<int> degree(dim, 0);
        #pragma omp parallel num_threads(num_cores)
        {
            std::vector<int> local_deg(dim, 0);
            #pragma omp for nowait
            for (int k = 0; k < nnz_val; ++k) {
                local_deg[call.row_indices[k]]++;
                local_deg[call.col_indices[k]]++;
            }
            #pragma omp critical
            for (int i = 0; i < dim; ++i) degree[i] += local_deg[i];
        }

        // Threshold = 5 × mean degree
        long long deg_sum = 0;
        for (int i = 0; i < dim; ++i) deg_sum += degree[i];
        mean_deg  = (int)(deg_sum / dim);
        const int threshold = 5 * mean_deg;

        int n_dense = 0;
        #pragma omp parallel for reduction(+:n_dense) num_threads(num_cores)
        for (int i = 0; i < dim; ++i)
            if (degree[i] > threshold) n_dense++;

        sTiles::Logger::timing("│   ↪ Dense node preprocessing: threshold=" + std::to_string(threshold)
            + " (5×mean=" + std::to_string(mean_deg) + ") → " + std::to_string(n_dense) + " nodes");

        int max_deg = *std::max_element(degree.begin(), degree.end());
        bool star_like = (n_dense > 0 && max_deg > 20 * mean_deg);

        if (n_dense >= std::max(5, dim / 500) || star_like) {
            const int sub_dim = dim - n_dense;

            // Build old2sub mapping (original_node → sub_index, -1 for dense)
            std::vector<int> old2sub(dim, -1);
            sub2old_vec.resize(sub_dim);
            dense_orig_vec.reserve(n_dense);
            int si = 0;
            for (int i = 0; i < dim; ++i) {
                if (degree[i] <= threshold) { old2sub[i] = si; sub2old_vec[si++] = i; }
                else                        { dense_orig_vec.push_back(i); }
            }

            // Count subgraph edges (both endpoints non-dense)
            int sub_nnz = 0;
            for (int k = 0; k < nnz_val; ++k) {
                int r = call.row_indices[k], c = call.col_indices[k];
                if (old2sub[r] >= 0 && old2sub[c] >= 0) sub_nnz++;
            }

            // Build filtered + remapped subgraph COO
            sub_rows_storage.resize(sub_nnz);
            sub_cols_storage.resize(sub_nnz);
            {
                int pos = 0;
                for (int k = 0; k < nnz_val; ++k) {
                    int r = old2sub[call.row_indices[k]], c = old2sub[call.col_indices[k]];
                    if (r >= 0 && c >= 0) { sub_rows_storage[pos] = r; sub_cols_storage[pos++] = c; }
                }
            }

            // Redirect call to subgraph (save originals for restoration after orderings)
            orig_row_ptr = call.row_indices;
            orig_col_ptr = call.col_indices;
            orig_nnz     = scheme.nnz;
            call.row_indices = sub_rows_storage.data();
            call.col_indices = sub_cols_storage.data();
            scheme.nnz = sub_nnz;
            scheme.fixed_column_size = n_dense;
            did_dense_perm = true;

            sTiles::Logger::timing("│   ↪ Moved " + std::to_string(n_dense)
                + " nodes (span > " + std::to_string(threshold) + ") to end of ordering.");
        }
    }

    // Baseline symbolic factorization (no ordering)
    TileIndexer::State base_state;

    double t_count = omp_get_wtime();
    int active1 = TileIndexer::countActiveTiles(call.row_indices, call.col_indices, scheme.nnz, scheme.dim, scheme.tile_size, method, &base_state, num_cores);
    double t_count_end = omp_get_wtime();

    if (active1 < 0) {
        sTiles::Logger::error("Tile counting failed in baseline path");
        return StatusCode::ExecutionFailed;
    }
    sTiles::Logger::timing("│   ↪ countActiveTiles (baseline): active="
                           + std::to_string(active1)
                           + ", time=" + sTiles::format_seconds(t_count_end - t_count) + " s");

    int filled1 = 0;  // baseline FillTiles skipped

    if (!have_user_perm && call.factorization_variant != 2)
    // --- Test all orderings 1-9 and compare with baseline ---
    {
        // Force-ordering override — driven by the FORCE_SCOTCH toggle at the
        // top of this function. When FORCE_SCOTCH=true, skip the adaptive
        // benchmark and run only SCOTCH (id=4). Bypasses the sparse-class
        // SCOTCH prune too.
        int force_id_override = 0;
        if (FORCE_SCOTCH) {
            force_id_override = 4;
            sTiles::Logger::timing("│   ↪ FORCE_SCOTCH=true -> evaluating only SCOTCH (id=4)"
                " (flip the toggle at the top of symbolic_phase to disable)");
        }
        // M3 controlled comparison: STILES_FORCE_ORDERING=<id> pins the bake-off to one
        // ordering (21=METIS) so every solver factors the same fill. Inert unless set.
        if (const char* e = std::getenv("STILES_FORCE_ORDERING")) {
            force_id_override = std::atoi(e);
            sTiles::Logger::timing("│   ↪ STILES_FORCE_ORDERING -> evaluating only id="
                + std::to_string(force_id_override));
        }

        // Select orderings based on matrix size.
        // IDs: 1=RCM, 4=SCOTCH, 41=ASCOTCH, 42=FSCOTCH, 21=METIS, 5=AMD, 6=CAMD, 7=COLAMD
        std::vector<int> orderings_vec;
        if (force_id_override > 0) {
            orderings_vec = {force_id_override};
        } else {
            const int n = scheme.dim;
            if (n < 5000) {
                // Small: run all
                orderings_vec = {1, 4, 41, 42, 21, 5, 6, 7}; // RCM, SCOTCH×3, METIS, AMD, CAMD, COLAMD
            } else if (n < 15000) {
                // Medium: drop RCM
                orderings_vec = {4, 41, 42, 21, 5, 6, 7};    // SCOTCH×3, METIS, AMD, CAMD, COLAMD
            } else if (n < 100000) {
                // Large: drop COLAMD and RCM
                orderings_vec = {4, 41, 42, 5, 21, 1, 6}; 

            } else if (n < 200000) {
                // Large: drop COLAMD and RCM
                orderings_vec = {4, 41, 42, 5, 6};       // SCOTCH×3, AMD, CAMD
            } else {
                // Very large: consistent winners only (METIS runs only as the rescue)
                orderings_vec = {4, 41, 42, 6};          // SCOTCH×3, CAMD
            }
        }

        // Adaptive pruning based on matrix class.
        // Sparse class (mean_deg < 50, GMRF/graph-like): SCOTCH-family orderings tend
        // to lose to CAMD/AMD on nnz(L), and they're 2-3× slower to run. Drop them to
        // cut the ordering-benchmark wall-clock.
        // Dense class (mean_deg >= 50, FEM/structural): keep current set; ND variants
        // dominate and CAMD serves as a safety candidate.
        // (Prior version kept SCOTCH in the pool for very large sparse matrices
        //  (n >= 500K) on the theory that its separator-tree parallelism could
        //  beat CAMD/AMD on wall-clock. Measurement shows otherwise: on
        //  net1628760 and yU0G1u (both ~1.5M dim graph-sparse), CAMD wins on
        //  nnz(L) AND on wall-clock, while SCOTCH variants each take 20-40s
        //  computing a permutation that gets thrown away. The dim cap is
        //  removed below so SCOTCH is pruned regardless of size when the
        //  graph-like predicate fires.)
        // Skip pruning when a forced ordering was requested — respect user's choice.
        if (force_id_override == 0 && mean_deg > 0 && mean_deg < 50) {
            std::vector<int> pruned;
            pruned.reserve(orderings_vec.size());
            for (int id : orderings_vec) {
                // Drop SCOTCH family (4, 41, 42) — they lose on sparse matrices
                if (id != 4 && id != 41 && id != 42) pruned.push_back(id);
            }
            if (!pruned.empty()) {
                orderings_vec = std::move(pruned);
            }
        }

        const int* all_orderings = orderings_vec.data();
        const int n_ord = static_cast<int>(orderings_vec.size());

        // Candidate pool actually evaluated (after size-bucket selection and the
        // sparse-class pruning above). Printed as a clean [TIME] line so the
        // bake-off contestants are visible alongside the chosen winner below.
        {
            auto ordering_name = [](int id) -> const char* {
                switch (id) {
                    case 1:  return "RCM";    case 4:  return "SCOTCH";
                    case 41: return "ASCOTCH"; case 42: return "FSCOTCH";
                    case 21: return "METIS";  case 5:  return "AMD";
                    case 6:  return "CAMD";   case 7:  return "COLAMD";
                    default: return "?";
                }
            };
            std::string pool;
            for (int i = 0; i < n_ord; ++i) {
                if (i) pool += ", ";
                pool += ordering_name(all_orderings[i]);
            }
            sTiles::Logger::timing("│   ↪ Ordering candidates (n=" + std::to_string(scheme.dim)
                + "): " + pool);
        }

        double strategy_timeout_s = 120.0;  // 2-minute per-strategy timeout
        //if (scheme.tile_size < 40) strategy_timeout_s += 120.0;

        struct OrdResult {
            int id = 0;
            std::string name;
            int active = -1, filled = -1;
            double elapsed = 0.0;
            double ord_elapsed = 0.0;   // time for run_permutation only (ordering itself)
            StatusCode status = StatusCode::IllegalValue;
            std::vector<int> saved_perm;  // saved for composed ordering tests
            ScotchTree saved_tree;        // populated only for SCOTCH variants (id 4/41/42)
        };
        std::vector<OrdResult> ord_results(n_ord);

        // Pre-fill id/name so the parallel loop has no shared writes to name strings
        for (int i = 0; i < n_ord; ++i) {
            ord_results[i].id   = all_orderings[i];
            ord_results[i].name = ordering_name_from_id(all_orderings[i]);
        }

        // --- Shared symmetric adjacency graph (env-gated, DEFAULT OFF) ---
        // SCOTCH/FSCOTCH (id 4/42), METIS (21), AMD (5) and CAMD (6) all consume
        // the identical symmetric/deduplicated/diagonal-free graph. Building it
        // once here and handing each wrapper a pointer removes the redundant
        // per-ordering rebuilds (the SCOTCH family alone rebuilds it up to 3x).
        // The graph is proven graph-identical to each wrapper's own build (see
        // ordering_shared_csr.hpp + test). ASCOTCH (41) is excluded — it
        // pre-permutes the matrix, so its graph differs; COLAMD (7) is excluded
        // — its AᵀA semantics differ.
        //
        // DEFAULT: auto-ON for sparse-class (mean_deg<50). Measured ~16% off the
        // bake-off there (thermomech_dM 0.20s->0.17s, bit-identical) because
        // AMD/CAMD/METIS ARE the critical path and their redundant graph builds
        // (esp. SuiteSparse's vector<vector<int>>) are a big fraction of it. On
        // dense-class it is a measured wash — SCOTCH's own ordering compute
        // dominates and the build is <10% of it — so it stays OFF by default
        // there to avoid the prebuild overhead. Override (either direction):
        //   STILES_SHARED_CSR=1  -> force ON  (all classes)
        //   STILES_SHARED_CSR=0  -> force OFF
        // DECLARED BEFORE the graveyard so it outlives any timed-out future that
        // still references it (objects destruct in reverse declaration order).
        SharedAdjCSR shared_csr;
        const SharedAdjCSR* shared_csr_ptr = nullptr;
        {
            bool use_shared = (mean_deg > 0 && mean_deg < 50);  // default: sparse-class only
            if (const char* _e_sc = std::getenv("STILES_SHARED_CSR"))
                use_shared = (std::atoi(_e_sc) != 0);            // explicit override wins
            const int shared_dim = scheme.dim - scheme.fixed_column_size;
            if (use_shared && shared_dim > 0) {
                shared_csr = build_shared_adj_csr(call.row_indices, call.col_indices,
                                                  scheme.nnz, shared_dim, num_cores);
                shared_csr_ptr = &shared_csr;
                sTiles::Logger::timing("│   ↪ Shared adjacency graph built once for the bake-off (dim="
                    + std::to_string(shared_dim) + ", edges=" + std::to_string(shared_csr.xadj[shared_dim]) + ")");
            }
        }

        // Graveyard: timed-out futures stay alive until end of this block so their
        // destructor doesn't block immediately (they are awaited when the graveyard
        // goes out of scope, after results have already been collected).
        std::vector<std::future<void>> graveyard;
        std::mutex graveyard_mutex;

        // Deadline shared by all strategies: every strategy gets strategy_timeout_s
        // from the start of the batch.
        const auto batch_start = std::chrono::steady_clock::now();
        const auto deadline    = batch_start + std::chrono::duration<double>(strategy_timeout_s);


        // Launch each ordering candidate as its own concurrent async task. Each
        // candidate's tile count + fill pass runs single-threaded: the default
        // CharMask closure has no parallel variant, and splitting cores across
        // candidates (via a parallel-capable mask) was measured slower and dropped.
        std::vector<std::future<void>> elem_futures(n_ord);
        for (int i = 0; i < n_ord; ++i) {
            elem_futures[i] = sTiles::async_bigstack([&, i]() {
                auto& res = ord_results[i];
                // External/Fortran orderings (id>=12, id<40, id!=21) allocate O(N) heap — skip on very large N.
                // SCOTCH variants (id=41,42) and METIS direct (id=21) are fine for large matrices.
                if ((res.id >= 12 && res.id < 40 && res.id != 21) && scheme.dim > 600000) {
                    res.name += " (skipped-large)";
                    res.status = StatusCode::IllegalValue;
                    return;
                }
                const double ts = omp_get_wtime();

                int* perm  = nullptr;
                int* iperm = nullptr;
                int* csr_i = new int[scheme.nnz];
                int* csr_j = new int[scheme.nnz];
                std::copy(call.row_indices, call.row_indices + scheme.nnz, csr_i);
                std::copy(call.col_indices, call.col_indices + scheme.nnz, csr_j);

                int local_dim = scheme.dim, local_nnz = scheme.nnz, tree_sep = 0;
                int* sizes = nullptr;
                const int fixed_col_ord = scheme.fixed_column_size;

                // Capture SCOTCH separator tree for the winner-installation step
                // (only meaningful for SCOTCH variants 4/41/42; ignored for others).
                ScotchTree* tree_arg = (res.id == 4 || res.id == 41 || res.id == 42)
                                     ? &res.saved_tree : nullptr;

                res.status = sTiles::run_permutation(
                    &perm, &iperm, &local_dim, &local_nnz,
                    fixed_col_ord, &csr_i, &csr_j,
                    res.id, &tree_sep, group_index, &sizes, 1, tree_arg, shared_csr_ptr
                );
                res.ord_elapsed = omp_get_wtime() - ts;  // ordering only
                delete[] csr_i;
                delete[] csr_j;
                if (sizes) delete[] sizes;

            if (res.status == StatusCode::Success && perm) {
                TileIndexer::State ord_state;
                res.active = count_active_tiles_with_perm(
                    call.row_indices, call.col_indices, scheme.nnz,
                    scheme.dim, scheme.tile_size, perm,
                    method, &ord_state, 1, group_index
                );
                if (res.active >= 0) {
                    res.filled = TileIndexer::FillTiles(ord_state, method, scheme.dimTiledMatrix, res.active, 1);
                    int tot = res.active + res.filled;
                    TileIndexer::ensure_diagonal_tiles_active(ord_state, method, scheme.dimTiledMatrix, tot);
                    res.filled = tot - res.active;
                }
                TileIndexer::release_state_resources(ord_state);

            }

                // Save perm for composed ordering tests
                if (perm && res.active >= 0 && res.filled >= 0)
                    res.saved_perm.assign(perm, perm + scheme.dim);

                OrderingMemoryManager::deallocate(perm);
                OrderingMemoryManager::deallocate(iperm);
                res.elapsed = omp_get_wtime() - ts;
            }); // end async lambda
        }

        // Collect results: wait for each future up to the shared deadline.
        // Timed-out futures are moved to the graveyard — they keep running in the
        // background and are joined when the graveyard goes out of scope (after the
        // results table has already been logged and the best strategy selected).
        for (int i = 0; i < n_ord; ++i) {
            if (elem_futures[i].wait_until(deadline) == std::future_status::timeout) {
                ord_results[i].name += " (timeout)";
                std::lock_guard<std::mutex> lk(graveyard_mutex);
                graveyard.push_back(std::move(elem_futures[i]));
            }
        }
        // graveyard destructs at end of this block, joining any still-running threads.

        // Bake-off report: one clean [TIME] row per candidate ordering, showing
        // how long it took and the fill proxy it produced. "fill-proxy" =
        // active+filled tiles (lower is better; the min-proxy candidate is the
        // winner that gets installed). "ord" is the ordering call alone; "total"
        // adds the tile-count pass. The natural-ordering baseline is in the header.
        sTiles::Logger::timing("│   ↪ Ordering comparison (baseline fill-proxy="
            + std::to_string(active1 + filled1) + "):");
        {
            long long best_total = LLONG_MAX;
            for (const auto& res : ord_results)
                if (res.status == StatusCode::Success && res.active >= 0 && res.filled >= 0)
                    best_total = std::min(best_total,
                                          static_cast<long long>(res.active) + res.filled);
            for (const auto& res : ord_results) {
                char row[192];
                if (res.status == StatusCode::Success && res.active >= 0 && res.filled >= 0) {
                    const long long total = static_cast<long long>(res.active) + res.filled;
                    std::snprintf(row, sizeof(row),
                        "│       %-8s ord=%7.3fs  total=%7.3fs  fill-proxy=%-9lld%s",
                        res.name.c_str(), res.ord_elapsed, res.elapsed, total,
                        (total == best_total ? "  <- winner" : ""));
                } else {
                    std::snprintf(row, sizeof(row),
                        "│       %-8s ord=    --     total=%7.3fs  failed/timeout (status=%d)",
                        res.name.c_str(), res.elapsed, static_cast<int>(res.status));
                }
                sTiles::Logger::timing(row);
            }
        }

        // --- Tile-level orderings (disabled — applied after nnz(L) winner is selected) ---
        struct TileOrdResult {
            std::string label;
            int id = 0;
            int active = -1, filled = -1;
            double elapsed = 0.0;
            std::vector<int> saved_perm;
        };
        std::vector<TileOrdResult> tile_ord_results;
        if (false) { // tile-level orderings disabled — proxy metric (active+filled) doesn't predict actual nnz(L) for composed orderings; they look great on proxy but lose on exact fill

            // --- Tile-level orderings on best element winner ---
            // Find best element ordering winner, build tile graph from its permuted matrix,
            // run standard orderings on the tile graph, compose back to element level.
            {
                // Find best element ordering winner
                int best_elem_score = INT_MAX;
                int best_elem_idx   = -1;
                for (int i = 0; i < n_ord; ++i) {
                    const auto& r = ord_results[i];
                    if (r.active >= 0 && r.filled >= 0 && !r.saved_perm.empty()) {
                        const int score = r.active + r.filled;
                        if (score < best_elem_score) { best_elem_score = score; best_elem_idx = i; }
                    }
                }

                if (best_elem_idx >= 0) {
                    const auto& best_elem       = ord_results[best_elem_idx];
                    const std::vector<int>& base_perm = best_elem.saved_perm;
                    const int ts         = scheme.tile_size;
                    const int tiles_dim  = (scheme.dim + ts - 1) / ts;

                    // Build tile graph from winner's permuted matrix
                    std::unordered_set<std::uint64_t> tile_map;
                    tile_map.reserve(static_cast<std::size_t>(scheme.nnz));
                    for (int k = 0; k < scheme.nnz; ++k) {
                        const int ri = base_perm[call.row_indices[k]];
                        const int ci = base_perm[call.col_indices[k]];
                        const int tR = ri / ts, tC = ci / ts;
                        if (tR == tC) continue;
                        const int lo = std::min(tR, tC), hi = std::max(tR, tC);
                        tile_map.insert((static_cast<std::uint64_t>(lo) << 32)
                                        | static_cast<std::uint32_t>(hi));
                    }
                    const int tile_nnz = tiles_dim + static_cast<int>(tile_map.size());
                    std::vector<int> tile_i(tile_nnz), tile_j(tile_nnz);
                    {
                        int pos = 0;
                        for (int t = 0; t < tiles_dim; ++t) { tile_i[pos] = t; tile_j[pos] = t; ++pos; }
                        std::vector<std::uint64_t> sorted_keys(tile_map.begin(), tile_map.end());
                        std::sort(sorted_keys.begin(), sorted_keys.end());
                        for (const auto key : sorted_keys) {
                            tile_i[pos] = static_cast<int>(key >> 32);
                            tile_j[pos] = static_cast<int>(key & 0xFFFFFFFFu);
                            ++pos;
                        }
                    }

                    // Choose orderings based on tiles_dim (same size thresholds, tile graph)
                    std::vector<int> tile_orderings;
                    if      (tiles_dim < 5000)   tile_orderings = {1, 4, 5, 6, 7};
                    else if (tiles_dim < 15000)  tile_orderings = {4, 5, 6, 7};
                    else if (tiles_dim < 200000) tile_orderings = {4, 5, 6, 1, 21, 7};
                    else                         tile_orderings = {4, 6};

                    const int n_to = static_cast<int>(tile_orderings.size());
                    std::vector<TileOrdResult> tl_results(n_to);
                    for (int i = 0; i < n_to; ++i) {
                        tl_results[i].id    = tile_orderings[i];
                        tl_results[i].label = best_elem.name + "+"
                                            + ordering_name_from_id(tile_orderings[i]) + "(tile)";
                    }

                    sTiles::Logger::timing("│   ↪ Tile-level ordering (on element winner: "
                                        + best_elem.name + ", tiles_dim=" + std::to_string(tiles_dim) + "):");

                    const auto tl_deadline = std::chrono::steady_clock::now()
                                        + std::chrono::duration<double>(strategy_timeout_s);
                    std::vector<std::future<void>> tl_futures(n_to);
                    std::vector<std::future<void>> tl_graveyard;
                    std::mutex tl_graveyard_mutex;

                    for (int i = 0; i < n_to; ++i) {
                        tl_futures[i] = sTiles::async_bigstack([&, i]() {
                            auto& tr = tl_results[i];
                            const double t0 = omp_get_wtime();

                            int* ti_copy = new int[tile_nnz];
                            int* tj_copy = new int[tile_nnz];
                            std::copy(tile_i.data(), tile_i.data() + tile_nnz, ti_copy);
                            std::copy(tile_j.data(), tile_j.data() + tile_nnz, tj_copy);

                            int* tperm = nullptr, *tiperm = nullptr;
                            int local_tiles_dim = tiles_dim, local_tile_nnz = tile_nnz;
                            int tree_sep = 0, *sizes = nullptr;

                            StatusCode sc = sTiles::run_permutation(
                                &tperm, &tiperm, &local_tiles_dim, &local_tile_nnz,
                                0, &ti_copy, &tj_copy, tr.id, &tree_sep, group_index, &sizes, 1
                            );
                            delete[] ti_copy; delete[] tj_copy;
                            if (sizes) delete[] sizes;
                            OrderingMemoryManager::deallocate(tiperm);

                            if (sc == StatusCode::Success && tperm) {
                                // Block-align: force partial last tile to stay last
                                const int last_tile   = tiles_dim - 1;
                                const int partial_size = scheme.dim - last_tile * ts;
                                std::vector<int> tp(tperm, tperm + tiles_dim);
                                if (partial_size != ts && tp[last_tile] != last_tile) {
                                    std::vector<int> inv(tiles_dim);
                                    for (int t = 0; t < tiles_dim; ++t) inv[tp[t]] = t;
                                    const int q = inv[last_tile];
                                    const int p = tp[last_tile];
                                    tp[last_tile] = last_tile;
                                    tp[q]         = p;
                                }

                                // Compose: composed[e] = tp[base_perm[e]/ts]*ts + (base_perm[e]%ts)
                                std::vector<int> composed(scheme.dim);
                                for (int e = 0; e < scheme.dim; ++e) {
                                    const int pe = base_perm[e];
                                    composed[e]  = tp[pe / ts] * ts + (pe % ts);
                                }

                                TileIndexer::State tstate;
                                tr.active = count_active_tiles_with_perm(
                                    call.row_indices, call.col_indices, scheme.nnz,
                                    scheme.dim, scheme.tile_size, composed.data(),
                                    method, &tstate, 1, group_index
                                );
                                if (tr.active >= 0) {
                                    tr.filled = TileIndexer::FillTiles(tstate, method,
                                        scheme.dimTiledMatrix, tr.active, 1);
                                    int tot = tr.active + tr.filled;
                                    TileIndexer::ensure_diagonal_tiles_active(tstate, method,
                                        scheme.dimTiledMatrix, tot);
                                    tr.filled = tot - tr.active;
                                }
                                TileIndexer::release_state_resources(tstate);

                                if (tr.active >= 0 && tr.filled >= 0)
                                    tr.saved_perm = std::move(composed);
                            }

                            OrderingMemoryManager::deallocate(tperm);
                            tr.elapsed = omp_get_wtime() - t0;
                        });
                    }

                    // Collect with deadline → graveyard
                    for (int i = 0; i < n_to; ++i) {
                        if (tl_futures[i].wait_until(tl_deadline) == std::future_status::timeout) {
                            tl_results[i].label += " (timeout)";
                            std::lock_guard<std::mutex> lk(tl_graveyard_mutex);
                            tl_graveyard.push_back(std::move(tl_futures[i]));
                        }
                    }

                    // Log and collect
                    for (auto& tr : tl_results) {
                        std::string msg = "│       " + tr.label + ": ";
                        if (tr.active >= 0 && tr.filled >= 0) {
                            const int total = tr.active + tr.filled;
                            const int delta = total - (active1 + filled1);
                            msg += "active=" + std::to_string(tr.active)
                                + ", filled=" + std::to_string(tr.filled)
                                + ", total=" + std::to_string(total)
                                + " (delta=" + (delta <= 0 ? "" : "+") + std::to_string(delta) + ")"
                                + ", time=" + sTiles::format_seconds(tr.elapsed) + " s";
                        } else {
                            msg += "failed, time=" + sTiles::format_seconds(tr.elapsed) + " s";
                        }
                        sTiles::Logger::timing(msg);
                        tile_ord_results.push_back(std::move(tr));
                    }
                }
            } // end tile-level ordering section inner block
            
            
        } // end if(false) — tile-level orderings disabled during benchmarking

        sTiles::Logger::timing("│   ↪ Symbolic_phase ordering: "
            + sTiles::format_seconds(omp_get_wtime() - t_symbolic_start) + " s");

        // --- Restore original COO if dense node preprocessing redirected it ---
        if (did_dense_perm) {
            call.row_indices         = orig_row_ptr;
            call.col_indices         = orig_col_ptr;
            scheme.nnz               = orig_nnz;
            scheme.fixed_column_size = 0;
        }

        // --- Collect top candidates by tile count, then pick by exact nnz(L) ---
        struct Candidate {
            const std::vector<int>* perm_ptr = nullptr;
            std::string label;
            int tile_score = INT_MAX;
            const ScotchTree* tree_ptr = nullptr;  // non-null only for SCOTCH variants
            int id = -1;  // element-level ordering id (-1 for composed/tile-level)
        };
        std::vector<Candidate> candidates;
        for (const auto& r : ord_results) {
            if (r.active >= 0 && r.filled >= 0 && !r.saved_perm.empty()) {
                const ScotchTree* tp = (r.id == 4 || r.id == 41 || r.id == 42)
                                     ? &r.saved_tree : nullptr;
                candidates.push_back({&r.saved_perm, r.name, r.active + r.filled, tp, r.id});
            }
        }
        for (const auto& r : tile_ord_results) {
            if (r.active >= 0 && r.filled >= 0 && !r.saved_perm.empty())
                candidates.push_back({&r.saved_perm, r.label, r.active + r.filled, nullptr, -1});
        }
        // Sort by tile score and keep top 5 for exact nnz(L) evaluation
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) { return a.tile_score < b.tile_score; });
        // Select best 3 element-level orderings for nnz(L) evaluation.
        // At most 1 SCOTCH variant (they're similar), plus diverse orderings.
        int max_eval = 0;
        bool has_scotch = false;
        for (int i = 0; i < static_cast<int>(candidates.size()) && max_eval < 3; ++i) {
            // Skip tile-level orderings
            if (candidates[i].label.find('+') != std::string::npos) continue;
            // At most 1 SCOTCH variant
            const bool is_scotch = (candidates[i].label.find("ND") != std::string::npos);
            if (is_scotch && has_scotch) continue;
            if (is_scotch) has_scotch = true;
            std::swap(candidates[max_eval], candidates[i]);
            ++max_eval;
        }

        // Evaluation result — keeps L structure and tile state for winner installation
        struct EvalResult {
            int nnz_L = -1;
            int tiles_L = -1;
            std::vector<int> cp;
            std::vector<int> ri;
            TileIndexer::State state;
            std::vector<int> full_perm;
        };

        auto evaluate_candidate = [&](const std::vector<int>& perm_vec, int eval_threads) -> EvalResult {
            EvalResult result;
            const int n = scheme.dim;
            result.full_perm.resize(n);
            if (did_dense_perm) {
                const int sub_dim = static_cast<int>(sub2old_vec.size());
                for (int si = 0; si < sub_dim; ++si)
                    result.full_perm[sub2old_vec[si]] = perm_vec[si];
                for (int j = 0; j < static_cast<int>(dense_orig_vec.size()); ++j)
                    result.full_perm[dense_orig_vec[j]] = sub_dim + j;
            } else {
                std::copy(perm_vec.begin(), perm_vec.end(), result.full_perm.begin());
            }
            std::vector<int> colptr(n + 1, 0);
            for (int k = 0; k < scheme.nnz; ++k) {
                int nr = result.full_perm[call.row_indices[k]];
                int nc = result.full_perm[call.col_indices[k]];
                if (nr < nc) std::swap(nr, nc);
                ++colptr[nc + 1];
            }
            for (int j = 0; j < n; ++j) colptr[j + 1] += colptr[j];
            std::vector<int> rowind(colptr[n]);
            {
                std::vector<int> pos(colptr.begin(), colptr.begin() + n);
                for (int k = 0; k < scheme.nnz; ++k) {
                    int nr = result.full_perm[call.row_indices[k]];
                    int nc = result.full_perm[call.col_indices[k]];
                    if (nr < nc) std::swap(nr, nc);
                    rowind[pos[nc]++] = nr;
                }
            }
            // Per-column sorts are independent (disjoint ranges) -> safe to parallelize.
            #pragma omp parallel for num_threads(eval_threads) schedule(dynamic, 256) if(eval_threads > 1)
            for (int j = 0; j < n; ++j)
                std::sort(rowind.begin() + colptr[j], rowind.begin() + colptr[j + 1]);
            const bool _sp = (std::getenv("STILES_PROFILE_SYM") != nullptr);
            double _t_fill = omp_get_wtime();
            result.nnz_L = sTiles::symbolic_chol_fillin(n, colptr, rowind, result.cp, result.ri, 2'000'000'000UL);
            if (_sp) sTiles::Logger::timing_always("│   ↪ symbolic_chol_fillin(serial): ", omp_get_wtime()-_t_fill, " s  (nnz_L=", result.nnz_L, ")");
            if (result.nnz_L < 0) return result;

            // Build COO from L and count tiles. Outer-j is independent (each p written
            // once, disjoint per column) -> safe to parallelize.
            std::vector<int> L_row(result.nnz_L), L_col(result.nnz_L);
            #pragma omp parallel for num_threads(eval_threads) schedule(dynamic, 256) if(eval_threads > 1)
            for (int j = 0; j < n; ++j)
                for (int p = result.cp[j]; p < result.cp[j + 1]; ++p) {
                    L_row[p] = result.ri[p];
                    L_col[p] = j;
                }
            double _t_cnt = omp_get_wtime();
            result.tiles_L = TileIndexer::countActiveTiles(
                L_row.data(), L_col.data(), result.nnz_L,
                n, scheme.tile_size, method, &result.state, eval_threads);
            if (_sp) sTiles::Logger::timing_always("│   ↪ countActiveTiles(", eval_threads, " core): ", omp_get_wtime()-_t_cnt, " s  (tiles_L=", result.tiles_L, ")");
            if (result.tiles_L >= 0) {
                TileIndexer::ensure_diagonal_tiles_active(result.state, method,
                    scheme.dimTiledMatrix, result.tiles_L);
            }
            return result;
        };

        // Evaluate top candidates — pick winner by metric adaptive to matrix class.
        //
        // Sparse class (mean_deg < 50, GMRF/graph-like):
        //   Use min nnz(L) with tiles(L) as tiebreaker. Cholesky FLOPs scale with
        //   nnz(L)^1.5 and dominate tile-management overhead on these matrices.
        //
        // Dense class (mean_deg >= 50, FEM/structural):
        //   Use min tiles(L) (original heuristic). ND orderings typically minimize
        //   both nnz(L) and tiles(L) together here, and per-tile overhead is the
        //   dominant tile-management cost.
        const bool sparse_class = (mean_deg > 0 && mean_deg < 50);
        const char* metric_label = sparse_class ? "nnz(L)" : "tiles(L)";
        // sTiles::Logger::timing(std::string("│   ↪ Evaluating top ") + std::to_string(max_eval)
        //     + " candidates by " + metric_label + " (mean_deg=" + std::to_string(mean_deg) + "):");

        // Evaluate all candidates up front; keep their states until a winner is chosen.
        // Storing all evals (max_eval ≤ 3) lets us apply a post-hoc ND override without
        // re-running the exact fill-in pass.
        std::vector<EvalResult> evals(max_eval);
        {
            // The max_eval (≤3) candidate evaluations are INDEPENDENT: each builds its
            // own permuted colptr/rowind, runs symbolic_chol_fillin into its own
            // result.cp/ri, and counts tiles into its own result.state. No shared
            // mutable state, no ordering dependency (the winner-selection below runs
            // AFTER the barrier). So they can run concurrently. Both inner calls
            // (symbolic_chol_fillin, countActiveTiles) are serial, so each eval uses 1
            // core; we have num_cores (>=max_eval here) free at this point because the
            // ordering bake-off futures have already joined. Gated for A/B:
            // STILES_PARALLEL_EVAL=0 forces the original serial loop. DEFAULT ON.
            const char* _pe = std::getenv("STILES_PARALLEL_EVAL");
            const bool parallel_eval = !(_pe && std::atoi(_pe) == 0) && max_eval > 1;
            // Inner (nested) parallelism budget PER candidate for the embarrassingly
            // parallel sub-steps (per-column sort, COO build, countActiveTiles). When
            // candidates run concurrently we split num_cores across them so the nested
            // OMP regions never oversubscribe (max_eval * inner_threads <= num_cores);
            // when serial, each candidate may use all cores. STILES_EVAL_INNER=0 forces
            // the original fully-serial inner path (=1). DEFAULT ON.
            const char* _ei = std::getenv("STILES_EVAL_INNER");
            const bool inner_par = !(_ei && std::atoi(_ei) == 0);
            const int inner_threads = !inner_par ? 1
                : (parallel_eval ? std::max(1, num_cores / max_eval) : std::max(1, num_cores));
            const bool _spe = (std::getenv("STILES_PROFILE_SYM") != nullptr);
            const double _t_eval = omp_get_wtime();
            if (parallel_eval) {
                std::vector<std::future<EvalResult>> _ef(max_eval);
                for (int i = 0; i < max_eval; ++i)
                    _ef[i] = sTiles::async_bigstack(
                                        [&, i]() { return evaluate_candidate(*candidates[i].perm_ptr, inner_threads); });
                for (int i = 0; i < max_eval; ++i)
                    evals[i] = _ef[i].get();
            } else {
                for (int i = 0; i < max_eval; ++i)
                    evals[i] = evaluate_candidate(*candidates[i].perm_ptr, inner_threads);
            }
            if (_spe) {
                sTiles::Logger::timing_always("│   ↪ eval loop [", parallel_eval ? "parallel" : "serial",
                                              "] max_eval=", max_eval, " inner_threads=", inner_threads,
                                              ": ", omp_get_wtime() - _t_eval, " s");
                for (int i = 0; i < max_eval; ++i)
                    sTiles::Logger::timing_always("│       cand ", i, ": nnz_L=", evals[i].nnz_L,
                                                  " tiles_L=", evals[i].tiles_L);
            }
        }

        // Primary-metric selection (class-adaptive).
        int best_idx = -1;
        int best_tiles_L = INT_MAX;
        int best_nnzL_seen = INT_MAX;
        for (int i = 0; i < max_eval; ++i) {
            const int nnzL = evals[i].nnz_L;
            const int tilesL = evals[i].tiles_L;
            if (tilesL < 0 || nnzL < 0) continue;
            bool is_better = false;
            if (best_idx == -1) {
                is_better = true;
            } else if (sparse_class) {
                if (nnzL < best_nnzL_seen) is_better = true;
                else if (nnzL == best_nnzL_seen && tilesL < best_tiles_L) is_better = true;
            } else {
                // Dense class: min tiles(L) with nnz(L)-override.
                // Compute scales as nnz(L)^1.5, so >=15% less fill dominates
                // extra tile-management overhead. Prefer such candidates even
                // when they have more tiles.
                if (tilesL < best_tiles_L) is_better = true;
                else if (tilesL == best_tiles_L && nnzL < best_nnzL_seen) is_better = true;
                else if (best_nnzL_seen > 0 &&
                         (double)nnzL <= 0.85 * (double)best_nnzL_seen) is_better = true;
            }
            if (is_better) {
                best_tiles_L = tilesL;
                best_nnzL_seen = nnzL;
                best_idx = i;
            } else if (nnzL < best_nnzL_seen) {
                best_nnzL_seen = nnzL;
            }
        }

        // --- ND parallelism override for medium/large matrices ---
        // ND orderings (SCOTCH/ASCOTCH/FSCOTCH/METIS) produce a shallow separator
        // tree that enables parallel factorization via red_tree_separator and the
        // scotch rangtab/treetab scheduling. At scale, this pays back a modest
        // tiles(L) / nnz(L) penalty many times over. If the primary-metric winner
        // is non-ND, swap to the best ND candidate when it falls within tolerance.
        //
        // Tolerance scales with available parallelism: Cholesky FLOPs grow as
        // nnz(L)^1.5 and parallel speedup is Amdahl-bounded roughly at p^(2/3)
        // effective cores, so ND can tolerate up to ~p^(2/3) more fill and
        // still win wall-clock. For banded critical-path GMRFs (very sparse,
        // huge N), this is the only way to expose any parallelism at all.
        auto is_nd_id = [](int id) {
            return id == 4 || id == 41 || id == 42 || id == 21;
        };
        // Parallelism budget: cap at 32 effective cores (memory bandwidth and
        // Amdahl limit scaling past this on typical sparse Cholesky).
        const int eff_cores = std::max(1, std::min(num_cores, 32));
        const double parallel_tol = std::pow(static_cast<double>(eff_cores), 2.0 / 3.0);
        if (best_idx >= 0 && !is_nd_id(candidates[best_idx].id)) {
            const int n = scheme.dim;
            double tiles_tol = 0.0, nnz_tol = 0.0;
            bool apply = false;
            if (!sparse_class) {
                // Dense class: ND dominates at scale — looser tolerance.
                if      (n >= 500000) { tiles_tol = 1.20; nnz_tol = 1.15; apply = true; }
                else if (n >= 100000) { tiles_tol = 1.10; nnz_tol = 1.08; apply = true; }
            } else if (mean_deg >= 15) {
                // Sparse class, not trivially-banded.
                // Tolerance on nnz(L) only (tile count less load-bearing here).
                if      (n >= 500000) { nnz_tol = 1.10; apply = true; }
                else if (n >= 100000) { nnz_tol = 1.05; apply = true; }
            }
            // Huge-matrix parallelism gate (CONDITIONAL): for sparse-class
            // matrices with n >= 500K AND multi-core, we originally wanted to
            // open nnz_tol to parallel_tol so ND can win even at higher fill.
            // This backfires on block-banded spatiotemporal matrices (hierarchical
            // SPDE × AR(1) is the canonical example) where SCOTCH's top-level
            // ND split gives only ~2x parallelism (time chain still serial
            // inside each half) and accepting 30-40% more fill costs way more
            // FLOPs than the parallel split recovers. Gate the wide tolerance
            // on ND *actually* reducing tile count substantially — if ND and
            // AMD/CAMD are near-tie on tiles(L) then ND's tree is flat and
            // parallel benefit is an illusion.
            // The ND-tile-advantage check happens below (after we've found
            // nd_tiles), so here we just record whether the parallel gate
            // WOULD fire if the tile advantage is real.
            const bool parallel_gate_eligible =
                (sparse_class && n >= 500000 && eff_cores >= 4);
            if (apply) {
                // Pick best ND candidate among the evaluated set by the primary metric.
                int nd_idx = -1;
                int nd_tiles = INT_MAX, nd_nnz = INT_MAX;
                for (int i = 0; i < max_eval; ++i) {
                    if (!is_nd_id(candidates[i].id)) continue;
                    const int tL = evals[i].tiles_L, nL = evals[i].nnz_L;
                    if (tL < 0 || nL < 0) continue;
                    bool nd_better = (nd_idx == -1);
                    if (!nd_better) {
                        if (sparse_class)
                            nd_better = (nL < nd_nnz) || (nL == nd_nnz && tL < nd_tiles);
                        else
                            nd_better = (tL < nd_tiles) || (tL == nd_tiles && nL < nd_nnz);
                    }
                    if (nd_better) { nd_idx = i; nd_tiles = tL; nd_nnz = nL; }
                }
                if (nd_idx >= 0) {
                    // Conditional parallel_tol: only open the wide tolerance if
                    // ND gives >=15% tile-count reduction. Without meaningful tile
                    // reduction the ND tree is flat (typical of block-banded /
                    // time-chain matrices) and its parallel potential is bounded
                    // to ~2x at best — insufficient to justify extra fill.
                    if (parallel_gate_eligible && best_tiles_L > 0 &&
                        (double)nd_tiles < 0.85 * (double)best_tiles_L) {
                        nnz_tol = std::max(nnz_tol, parallel_tol);
                    }
                    const bool passes_tiles = (tiles_tol <= 0.0) || (best_tiles_L <= 0) ||
                        ((double)nd_tiles <= tiles_tol * (double)best_tiles_L);
                    const bool passes_nnz = (nnz_tol <= 0.0) || (best_nnzL_seen <= 0) ||
                        ((double)nd_nnz <= nnz_tol * (double)best_nnzL_seen);
                    const bool passes = sparse_class ? passes_nnz
                                                     : (passes_tiles && passes_nnz);
                    if (passes) {
                        char ratio_buf[96];
                        std::snprintf(ratio_buf, sizeof(ratio_buf),
                            " (tiles ratio=%.2f, nnz ratio=%.2f)",
                            best_tiles_L > 0 ? (double)nd_tiles / (double)best_tiles_L : 0.0,
                            best_nnzL_seen > 0 ? (double)nd_nnz / (double)best_nnzL_seen : 0.0);
                        sTiles::Logger::timing("│   ↪ ND parallelism override (n="
                            + std::to_string(n) + "): "
                            + candidates[best_idx].label + " → "
                            + candidates[nd_idx].label + ratio_buf);
                        best_idx = nd_idx;
                        best_tiles_L = nd_tiles;
                        best_nnzL_seen = nd_nnz;
                    }
                }
            }
        }

        // Move winner's EvalResult out; release resources for the rest.
        EvalResult best_eval;
        for (int i = 0; i < max_eval; ++i) {
            if (i == best_idx) {
                best_eval = std::move(evals[i]);
            } else if (evals[i].tiles_L >= 0) {
                TileIndexer::release_state_resources(evals[i].state);
            }
        }

        // Evaluate identity (no ordering) as baseline competitor.
        // Only worth trying when mean degree is small — high mean degree
        // guarantees catastrophic fill-in without ordering.
        // Cap nnz(L) at the best ordering's value for early abort.
        if (mean_deg > 0 && mean_deg <= 100) {
            const int n = scheme.dim;
            const long long id_nnz_limit = (best_nnzL_seen < INT_MAX)
                ? static_cast<long long>(best_nnzL_seen) : 2'000'000'000LL;
            std::vector<int> colptr_id(n + 1, 0);
            for (int k = 0; k < scheme.nnz; ++k) {
                int nr = call.row_indices[k];
                int nc = call.col_indices[k];
                if (nr < nc) std::swap(nr, nc);
                ++colptr_id[nc + 1];
            }
            for (int j = 0; j < n; ++j) colptr_id[j + 1] += colptr_id[j];
            std::vector<int> rowind_id(colptr_id[n]);
            {
                std::vector<int> pos(colptr_id.begin(), colptr_id.begin() + n);
                for (int k = 0; k < scheme.nnz; ++k) {
                    int nr = call.row_indices[k];
                    int nc = call.col_indices[k];
                    if (nr < nc) std::swap(nr, nc);
                    rowind_id[pos[nc]++] = nr;
                }
            }
            for (int j = 0; j < n; ++j)
                std::sort(rowind_id.begin() + colptr_id[j], rowind_id.begin() + colptr_id[j + 1]);
            std::vector<int> cp_id, ri_id;
            const int nnzL_id = sTiles::symbolic_chol_fillin(n, colptr_id, rowind_id, cp_id, ri_id, id_nnz_limit);
            int tilesL_id = -1;
            if (nnzL_id >= 0) {
                std::vector<int> L_row(nnzL_id), L_col(nnzL_id);
                for (int j = 0; j < n; ++j)
                    for (int p = cp_id[j]; p < cp_id[j + 1]; ++p) {
                        L_row[p] = ri_id[p];
                        L_col[p] = j;
                    }
                TileIndexer::State id_eval_state;
                tilesL_id = TileIndexer::countActiveTiles(
                    L_row.data(), L_col.data(), nnzL_id,
                    n, scheme.tile_size, method, &id_eval_state, num_cores);
                if (tilesL_id >= 0) {
                    TileIndexer::ensure_diagonal_tiles_active(id_eval_state, method,
                        scheme.dimTiledMatrix, tilesL_id);
                }
                TileIndexer::release_state_resources(id_eval_state);
            }
            // sTiles::Logger::timing("│       identity: nnz(L)=" + std::to_string(nnzL_id)
            //     + ", tiles(L)=" + std::to_string(tilesL_id));
            if (tilesL_id >= 0 && tilesL_id <= best_tiles_L) {
                if (best_eval.tiles_L >= 0)
                    TileIndexer::release_state_resources(best_eval.state);
                best_eval = EvalResult();
                best_tiles_L = tilesL_id;
                best_idx = -1;
                sTiles::Logger::timing("│   ↪ Identity (no ordering) wins or ties — skipping ordering");
            }
        } else if (mean_deg > 100) {
            sTiles::Logger::timing("│       identity: skipped (mean_deg=" + std::to_string(mean_deg) + " > 100)");
        }

        bool installed = false;
        if (best_idx >= 0) {
            const auto& winner = candidates[best_idx];
            const int n = scheme.dim;
            int* final_perm  = OrderingMemoryManager::allocate<int>(n, group_index);
            int* final_iperm = OrderingMemoryManager::allocate<int>(n, group_index);
            if (final_perm && final_iperm) {
                std::copy(best_eval.full_perm.begin(), best_eval.full_perm.end(), final_perm);
                compute_inverse_permutation(final_perm, final_iperm, n);

                scheme.element_perm      = final_perm;
                scheme.element_iperm     = final_iperm;
                scheme.state             = std::move(best_eval.state);
                scheme.numActiveTiles    = best_eval.tiles_L;
                scheme.nnz_factor        = static_cast<long long>(best_eval.nnz_L);
                scheme.L_colptr = OrderingMemoryManager::allocate<int>(n + 1, group_index);
                scheme.L_rowind = OrderingMemoryManager::allocate<int>(best_eval.nnz_L, group_index);
                std::copy(best_eval.cp.begin(), best_eval.cp.end(), scheme.L_colptr);
                std::copy(best_eval.ri.begin(), best_eval.ri.end(), scheme.L_rowind);
                scheme.use_ordering      = 1;
                scheme.selected_ordering = winner.label;
                scheme.red_tree_separator_level = 0;
                scheme.nd_padding        = 0;

                // If the winner is a SCOTCH variant, persist its ND separator tree
                bool have_scotch_tree = false;
                if (winner.tree_ptr && winner.tree_ptr->cblknbr > 0) {
                    const ScotchTree& tr = *winner.tree_ptr;
                    const int cb = tr.cblknbr;
                    scheme.scotch_cblknbr = cb;
                    scheme.scotch_rangtab = OrderingMemoryManager::allocate<int>(cb + 1, group_index);
                    scheme.scotch_treetab = OrderingMemoryManager::allocate<int>(cb,     group_index);
                    if (scheme.scotch_rangtab && scheme.scotch_treetab) {
                        std::copy(tr.rangtab.begin(), tr.rangtab.begin() + (cb + 1), scheme.scotch_rangtab);
                        std::copy(tr.treetab.begin(), tr.treetab.begin() + cb,       scheme.scotch_treetab);
                        sTiles::Logger::timing("│   ↪ SCOTCH separator tree captured: cblknbr="
                            + std::to_string(cb));
                        have_scotch_tree = true;
                    } else {
                        OrderingMemoryManager::deallocate(scheme.scotch_rangtab);
                        OrderingMemoryManager::deallocate(scheme.scotch_treetab);
                        scheme.scotch_rangtab = nullptr;
                        scheme.scotch_treetab = nullptr;
                        scheme.scotch_cblknbr = 0;
                        sTiles::Logger::warning("│   ↪ SCOTCH separator tree allocation failed; tree discarded");
                    }
                } else {
                    scheme.scotch_cblknbr = 0;
                    scheme.scotch_rangtab = nullptr;
                    scheme.scotch_treetab = nullptr;
                }
                TileIndexer::release_state_resources(base_state);
                installed = true;
                // Single always-on winner line (immune to log level, so it stays
                // visible in --csv/quiet sweeps). Reports the chosen ordering and
                // its key metrics; no other line repeats the winner.
                sTiles::Logger::timing_always("│   ↪ Ordering selected: " + winner.label
                    + "  (tiles=" + std::to_string(best_eval.tiles_L)
                    + ", nnz(L)=" + std::to_string((long long)best_eval.nnz_L)
                    + ", sep=" + std::to_string(winner.tree_ptr ? winner.tree_ptr->cblknbr : 0) + ")");

                // Top-level SCOTCH block padding — pads P1|P2|Sep boundaries to
                // tile_size multiples so diagonal tiles never straddle partition
                // boundaries, then populates scheme.partition_sizes so Path 2
                // partition-aware task scheduling engages.
                // Controlled by the ENABLE_SCOTCH_PADDING toggle at the top of
                // symbolic_phase (replaces the old STILES_ENABLE_SCOTCH_PADDING
                // env var). Default off — feature is still under tuning.
                //
                // SKIP for the supernodal sparse module (tile_type_mode == 2):
                // it runs symbolic_phase only to obtain element_perm, then does
                // its own supernodal symbolic/scheduling on the matrix it
                // ingests (scheme->dim/nnz + the COO). ND block padding is a
                // TILE-path concept (modes 0/1); applying it here grows
                // scheme->nnz and swaps in the padded COO, but the user later
                // hands sTiles_assign_values the ORIGINAL value array → the
                // sparse module reads past it (heap-buffer-overflow at
                // sparse/api.cpp assign_values). Mode 2 must stay unpadded.
                const int _ttm_pad = stiles_control_params[3];
                // Mode-aware partitioning gate: engage padding ONLY when the
                // matrix will resolve to a PARTITIONABLE mode (semisparse/dense),
                // NEVER sparse (mode 2) -- padding a sparse matrix corrupts the
                // supernodal factor ("not positive-definite at step 1"). Auto-
                // engage for such matrices with enough cores; the manual slot 21
                // (ENABLE_SCOTCH_PADDING) is ALSO mode-checked now, so a manual
                // --pad on a sparse matrix is safely declined. Env knob:
                // STILES_AUTO_PART_MIN_CORES (default 8).
                const int  _wb_mode = pad_gate_would_be_mode(&scheme, _ttm_pad, scheme.tile_size);
                const bool _mode_partitionable = (_wb_mode == 0 || _wb_mode == 1);
                const int  _eff_cores_auto = (scheme.num_cores > 0) ? scheme.num_cores : std::max(1, num_cores);
                int _auto_min_cores = 8;
                if (const char* e = std::getenv("STILES_AUTO_PART_MIN_CORES")) { const int v = std::atoi(e); if (v > 0) _auto_min_cores = v; }
                const bool _auto_partition = have_scotch_tree && _mode_partitionable
                                          && _eff_cores_auto >= _auto_min_cores;
                const bool _do_pad    = ENABLE_SCOTCH_PADDING || _auto_partition;
                const int  _eff_depth = (PATH2_DEPTH >= 2) ? PATH2_DEPTH : (_auto_partition ? 2 : PATH2_DEPTH);
                if (have_scotch_tree && _do_pad && _mode_partitionable) {
                    StatusCode pad_sc;
                    if (_eff_depth >= 2) {
                        // Walker must distribute cores across leaves using
                        // the actual per-call thread count (set by sTiles at
                        // scheme.num_cores = call_info->num_cores). Do NOT
                        // use num_cores (= eff_cores, a matrix-size
                        // heuristic) or omp_get_max_threads() (hardware
                        // default) — either may exceed the real thread
                        // count and leave high-index regions unclaimed.
                        const int walker_cores = (scheme.num_cores > 0)
                            ? scheme.num_cores
                            : std::max(1, num_cores);
                        sTiles::Logger::timing(std::string("│   ↪ SCOTCH padding: N-way mode")
                            + ((PATH2_DEPTH < 2 && _auto_partition) ? " [auto: mode=" + std::to_string(_wb_mode) + "]" : "")
                            + ", depth=" + std::to_string(_eff_depth)
                            + ", walker_cores=" + std::to_string(walker_cores));
                        pad_sc = apply_scotch_block_padding_nway(
                            call, scheme, group_index, walker_cores, _eff_depth);
                    } else {
                        pad_sc = apply_scotch_block_padding(
                            call, scheme, group_index, num_cores);
                    }
                    if (pad_sc != StatusCode::Success) {
                        sTiles::Logger::warning("│   ↪ SCOTCH block padding failed; continuing without padding");
                    }
                }
            } else {
                OrderingMemoryManager::deallocate(final_perm);
                OrderingMemoryManager::deallocate(final_iperm);
                TileIndexer::release_state_resources(best_eval.state);
            }
        }

        // [METIS RESCUE] If no cheap ordering installed (all exceeded the 2B fill guard)
        // and METIS was not in the pool (it is kept out of large-N pools for speed), run
        // METIS once now. It usually finds a representable nested-dissection ordering where
        // SCOTCH/CAMD blow up (e.g. Fault_639, Emilia_923). Guarded to !did_dense_perm so
        // the perm maps straight through evaluate_candidate's identity branch.
        if (!installed && !did_dense_perm &&
            std::find(orderings_vec.begin(), orderings_vec.end(), 21) == orderings_vec.end()) {
            int* perm = nullptr; int* iperm = nullptr;
            int* csr_i = new int[scheme.nnz]; int* csr_j = new int[scheme.nnz];
            std::copy(call.row_indices, call.row_indices + scheme.nnz, csr_i);
            std::copy(call.col_indices, call.col_indices + scheme.nnz, csr_j);
            int ld = scheme.dim, ln = scheme.nnz, tsep = 0; int* sz = nullptr;
            const StatusCode rst = sTiles::run_permutation(
                &perm, &iperm, &ld, &ln, scheme.fixed_column_size,
                &csr_i, &csr_j, 21, &tsep, group_index, &sz, 1, nullptr);
            delete[] csr_i; delete[] csr_j; if (sz) delete[] sz;
            if (rst == StatusCode::Success && perm) {
                std::vector<int> mperm(perm, perm + scheme.dim);
                OrderingMemoryManager::deallocate(perm);
                OrderingMemoryManager::deallocate(iperm);
                EvalResult me = evaluate_candidate(mperm, num_cores);  // single sequential eval -> full cores
                if (me.nnz_L >= 0 && me.tiles_L >= 0) {
                    const int n = scheme.dim;
                    int* fp  = OrderingMemoryManager::allocate<int>(n, group_index);
                    int* fip = OrderingMemoryManager::allocate<int>(n, group_index);
                    if (fp && fip) {
                        std::copy(me.full_perm.begin(), me.full_perm.end(), fp);
                        compute_inverse_permutation(fp, fip, n);
                        scheme.element_perm  = fp;
                        scheme.element_iperm = fip;
                        scheme.state         = std::move(me.state);
                        scheme.numActiveTiles = me.tiles_L;
                        scheme.nnz_factor    = static_cast<long long>(me.nnz_L);
                        scheme.L_colptr = OrderingMemoryManager::allocate<int>(n + 1, group_index);
                        scheme.L_rowind = OrderingMemoryManager::allocate<int>(me.nnz_L, group_index);
                        std::copy(me.cp.begin(), me.cp.end(), scheme.L_colptr);
                        std::copy(me.ri.begin(), me.ri.end(), scheme.L_rowind);
                        scheme.use_ordering = 1;
                        scheme.selected_ordering = "METIS-rescue";
                        scheme.red_tree_separator_level = 0;
                        scheme.nd_padding = 0;
                        installed = true;
                        sTiles::Logger::timing_always("│   ↪ Ordering selected: METIS-rescue  (nnz(L)="
                            + std::to_string(me.nnz_L) + ", tiles=" + std::to_string(me.tiles_L)
                            + "; cheap orderings exceeded 2B guard)");
                    }
                } else if (me.tiles_L >= 0) {
                    TileIndexer::release_state_resources(me.state);
                }
            } else {
                OrderingMemoryManager::deallocate(perm);
                OrderingMemoryManager::deallocate(iperm);
            }
        }

        if (!installed) {
            // No ordering improved — fall back to identity permutation on original matrix.
            // base_state was built on the (possibly subgraph-redirected) COO; discard and rebuild.
            TileIndexer::release_state_resources(base_state);
            TileIndexer::State id_state;
            int id_active = -1;

            // Exact etree fill-in under identity permutation
            {
                const int n_id = scheme.dim;
                std::vector<int> colptr_id(n_id + 1, 0);
                for (int k = 0; k < scheme.nnz; ++k) {
                    int nr = call.row_indices[k];
                    int nc = call.col_indices[k];
                    if (nr < nc) std::swap(nr, nc);
                    ++colptr_id[nc + 1];
                }
                for (int j = 0; j < n_id; ++j) colptr_id[j + 1] += colptr_id[j];
                std::vector<int> rowind_id(colptr_id[n_id]);
                {
                    std::vector<int> pos(colptr_id.begin(), colptr_id.begin() + n_id);
                    for (int k = 0; k < scheme.nnz; ++k) {
                        int nr = call.row_indices[k];
                        int nc = call.col_indices[k];
                        if (nr < nc) std::swap(nr, nc);
                        rowind_id[pos[nc]++] = nr;
                    }
                }
                for (int j = 0; j < n_id; ++j)
                    std::sort(rowind_id.begin() + colptr_id[j], rowind_id.begin() + colptr_id[j + 1]);
                std::vector<int> cp_id, ri_id;
                const int nnzL_id = sTiles::symbolic_chol_fillin(n_id, colptr_id, rowind_id, cp_id, ri_id, 2'000'000'000UL);
                if (nnzL_id >= 0) {
                    std::vector<int> L_row(nnzL_id), L_col(nnzL_id);
                    for (int j = 0; j < n_id; ++j)
                        for (int p = cp_id[j]; p < cp_id[j + 1]; ++p) {
                            L_row[p] = ri_id[p];
                            L_col[p] = j;
                        }
                    int total = TileIndexer::countActiveTiles(
                        L_row.data(), L_col.data(), nnzL_id,
                        n_id, scheme.tile_size, method, &id_state, num_cores);
                    if (total >= 0) {
                        TileIndexer::ensure_diagonal_tiles_active(id_state, method,
                            scheme.dimTiledMatrix, total);
                        id_active = total;
                        scheme.nnz_factor = static_cast<long long>(nnzL_id);
                        scheme.L_colptr = OrderingMemoryManager::allocate<int>(n_id + 1, group_index);
                        scheme.L_rowind = OrderingMemoryManager::allocate<int>(nnzL_id, group_index);
                        std::copy(cp_id.begin(), cp_id.end(), scheme.L_colptr);
                        std::copy(ri_id.begin(), ri_id.end(), scheme.L_rowind);
                    } else {
                        TileIndexer::release_state_resources(id_state);
                        return StatusCode::ExecutionFailed;
                    }
                } else {
                    sTiles::Logger::error(
                        "ERROR: symbolic_chol_fillin exceeded 2B nnz limit for identity permutation"
                        " (dim=", n_id, "). Matrix fill-in is too large.");
                    return StatusCode::ExecutionFailed;
                }
            }

            int* id_perm  = OrderingMemoryManager::allocate<int>(scheme.dim, group_index);
            int* id_iperm = OrderingMemoryManager::allocate<int>(scheme.dim, group_index);
            if (!id_perm || !id_iperm) {
                OrderingMemoryManager::deallocate(id_perm);
                OrderingMemoryManager::deallocate(id_iperm);
                TileIndexer::release_state_resources(id_state);
                return StatusCode::OutOfResources;
            }
            for (int i = 0; i < scheme.dim; ++i) { id_perm[i] = i; id_iperm[i] = i; }

            scheme.element_perm      = id_perm;
            scheme.element_iperm     = id_iperm;
            scheme.state             = std::move(id_state);
            scheme.numActiveTiles    = id_active;
            scheme.use_ordering      = 1;
            scheme.selected_ordering = "identity";
            scheme.nd_padding        = 0;

            // Separator heuristic (matches symbolic_phase_original fallback)
            int sep = 0;
            if (scheme.num_cores >= 2) {
                const int thick     = scheme.fixed_column_size;
                const int tile      = scheme.tile_size;
                const int num_tiles = scheme.dimTiledMatrix;
                sep = (tile > 0) ? ((thick + tile - 1) / tile) : 0;
                if ((num_tiles - 1) < 2 * scheme.num_cores) sep = 0;
            }
            scheme.red_tree_separator_level = sep;
            sTiles::Logger::debug("│   ↪ No ordering improved baseline — using identity permutation");
        }
    }

    // --- Tile sparsity summary ---
    //
    // DISABLED for speed via `if (false)` — pure diagnostic. All inputs
    // (tile_size, dimTiledMatrix, numActiveTiles, nnz_factor) come from
    // already-computed scheme fields; nothing downstream reads the local
    // tile_capacity / avg_density / tile_fill values. Flip to `if (true)`
    // when investigating tile occupancy.
    if (false) {
        const int ts        = scheme.tile_size;
        const int n         = scheme.dim;
        const int grid      = scheme.dimTiledMatrix;
        const int n_tiles   = scheme.numActiveTiles;
        const int total_lower_tiles = grid * (grid + 1) / 2;
        const long long nnzL = scheme.nnz_factor;
        const long long tile_capacity = static_cast<long long>(n_tiles) * ts * ts;
        const double avg_density = (tile_capacity > 0)
            ? 100.0 * static_cast<double>(nnzL) / static_cast<double>(tile_capacity)
            : 0.0;
        const double tile_fill = (total_lower_tiles > 0)
            ? 100.0 * static_cast<double>(n_tiles) / static_cast<double>(total_lower_tiles)
            : 0.0;

        char pct_buf[16];
        std::snprintf(pct_buf, sizeof(pct_buf), "%.1f", tile_fill);
        sTiles::Logger::timing("│   ↪ Tile summary: grid=" + std::to_string(grid) + "x" + std::to_string(grid)
            + ", tiles(L)=" + std::to_string(n_tiles) + "/" + std::to_string(total_lower_tiles)
            + " (" + pct_buf + "%)"
            + ", nnz(L)=" + std::to_string(nnzL)
            + ", avg tile density=" + std::to_string(static_cast<int>(avg_density + 0.5)) + "%");
    }

    // --- Export permutation (debug helper, gated on STILES_EXPORT_PERM env) ---
    if (scheme.element_perm && scheme.dim > 0) {
        const char* env = std::getenv("STILES_EXPORT_PERM");
        if (env && env[0]) {
            std::string path = env;
            FILE* fp = std::fopen(path.c_str(), "wb");
            if (fp) {
                std::fwrite(&scheme.dim, sizeof(int), 1, fp);
                std::fwrite(scheme.element_perm, sizeof(int), scheme.dim, fp);
                std::fclose(fp);
                sTiles::Logger::timing("│   ↪ Exported permutation to " + path
                    + " (n=" + std::to_string(scheme.dim) + ")");
            }
        }
    }

    return StatusCode::Success;
}

// ============================================================================
// Per-Partition Tile-Level Ordering Helper Functions
// ============================================================================

/**
 * @brief Data structure for partition submatrix
 */
struct PartitionData {
    int start;          // Partition start in permuted space
    int size;           // Partition size (elements)
    int* local_row;     // Partition-local row indices
    int* local_col;     // Partition-local col indices
    int local_nnz;      // Number of entries within partition

    PartitionData() : start(0), size(0), local_row(nullptr), local_col(nullptr), local_nnz(0) {}

    void cleanup(int group_index) {
        if (local_row) OrderingMemoryManager::deallocate(local_row);
        if (local_col) OrderingMemoryManager::deallocate(local_col);
        local_row = nullptr;
        local_col = nullptr;
    }
};

/**
 * @brief Extract partition submatrix from global matrix
 */
inline PartitionData extract_partition_submatrix(
    const int* row_indices,
    const int* col_indices,
    int nnz,
    const int* element_perm,
    int partition_start,
    int partition_size,
    int group_index)
{
    PartitionData pd;
    pd.start = partition_start;
    pd.size = partition_size;

    const int partition_end = partition_start + partition_size;

    // First pass: count entries within partition
    int count = 0;
    for (int k = 0; k < nnz; ++k) {
        const int ri = row_indices[k];
        const int cj = col_indices[k];

        const int perm_row = element_perm[ri];
        const int perm_col = element_perm[cj];

        const bool row_in_partition = (perm_row >= partition_start && perm_row < partition_end);
        const bool col_in_partition = (perm_col >= partition_start && perm_col < partition_end);

        if (row_in_partition && col_in_partition) {
            count++;
        }
    }

    pd.local_nnz = count;

    if (count == 0) {
        return pd;  // Empty partition
    }

    pd.local_row = OrderingMemoryManager::allocate<int>(count, group_index);
    pd.local_col = OrderingMemoryManager::allocate<int>(count, group_index);

    if (!pd.local_row || !pd.local_col) {
        pd.cleanup(group_index);
        return pd;
    }

    // Second pass: extract and convert to local coordinates
    int idx = 0;
    for (int k = 0; k < nnz; ++k) {
        const int ri = row_indices[k];
        const int cj = col_indices[k];

        const int perm_row = element_perm[ri];
        const int perm_col = element_perm[cj];

        const bool row_in_partition = (perm_row >= partition_start && perm_row < partition_end);
        const bool col_in_partition = (perm_col >= partition_start && perm_col < partition_end);

        if (row_in_partition && col_in_partition) {
            pd.local_row[idx] = perm_row - partition_start;
            pd.local_col[idx] = perm_col - partition_start;
            idx++;
        }
    }

    return pd;
}

/**
 * @brief Partition tile graph data structure
 */
struct PartitionTileGraph {
    int tiles_dim;
    int tile_nnz;
    int* tile_indices_i;
    int* tile_indices_j;

    PartitionTileGraph() : tiles_dim(0), tile_nnz(0), tile_indices_i(nullptr), tile_indices_j(nullptr) {}

    void cleanup(int group_index) {
        if (tile_indices_i) OrderingMemoryManager::deallocate(tile_indices_i);
        if (tile_indices_j) OrderingMemoryManager::deallocate(tile_indices_j);
        tile_indices_i = nullptr;
        tile_indices_j = nullptr;
    }
};

/**
 * @brief Build partition-local tile graph
 */
inline PartitionTileGraph build_partition_tile_graph(
    const PartitionData& pd,
    int tile_size,
    int group_index)
{
    PartitionTileGraph graph;
    graph.tiles_dim = (pd.size + tile_size - 1) / tile_size;

    // Build unique tile edges using unordered_set
    std::unordered_set<std::uint64_t> tile_map;
    tile_map.reserve(static_cast<std::size_t>(pd.local_nnz));

    for (int k = 0; k < pd.local_nnz; ++k) {
        const int row = pd.local_row[k];
        const int col = pd.local_col[k];

        if (row < 0 || row >= pd.size || col < 0 || col >= pd.size) continue;

        const int tileRow = row / tile_size;
        const int tileCol = col / tile_size;

        if (tileRow < 0 || tileRow >= graph.tiles_dim ||
            tileCol < 0 || tileCol >= graph.tiles_dim) continue;

        if (tileRow == tileCol) continue; // Skip diagonal

        int lo = tileRow, hi = tileCol;
        if (lo > hi) std::swap(lo, hi);

        const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(lo)) << 32)
                                | static_cast<std::uint32_t>(hi);
        tile_map.insert(key);
    }

    // Add diagonal entries for all tiles
    const int unique_edges = static_cast<int>(tile_map.size());
    graph.tile_nnz = unique_edges + graph.tiles_dim;

    graph.tile_indices_i = OrderingMemoryManager::allocate<int>(graph.tile_nnz, group_index);
    graph.tile_indices_j = OrderingMemoryManager::allocate<int>(graph.tile_nnz, group_index);

    if (!graph.tile_indices_i || !graph.tile_indices_j) {
        graph.cleanup(group_index);
        return graph;
    }

    int pos = 0;

    // Add diagonal entries
    for (int t = 0; t < graph.tiles_dim; ++t) {
        graph.tile_indices_i[pos] = t;
        graph.tile_indices_j[pos] = t;
        pos++;
    }

    // Add off-diagonal edges (sorted for deterministic ordering across runs)
    std::vector<std::uint64_t> sorted_keys(tile_map.begin(), tile_map.end());
    std::sort(sorted_keys.begin(), sorted_keys.end());
    for (const std::uint64_t key : sorted_keys) {
        const int lo = static_cast<int>(key >> 32);
        const int hi = static_cast<int>(key & 0xFFFFFFFFu);
        graph.tile_indices_i[pos] = lo;
        graph.tile_indices_j[pos] = hi;
        pos++;
    }

    return graph;
}

/**
 * @brief Result structure for partition tile ordering
 */
struct PartitionTileResult {
    int partition_id;           // 0=P1, 1=P2, 2=Sep
    int* best_tile_perm;
    int* best_tile_iperm;
    int tiles_dim;
    int best_active;
    int best_filled;
    std::string best_strategy;

    PartitionTileResult() : partition_id(0), best_tile_perm(nullptr), best_tile_iperm(nullptr),
                           tiles_dim(0), best_active(-1), best_filled(-1), best_strategy("NONE") {}

    void cleanup(int group_index) {
        if (best_tile_perm) OrderingMemoryManager::deallocate(best_tile_perm);
        if (best_tile_iperm) OrderingMemoryManager::deallocate(best_tile_iperm);
        best_tile_perm = nullptr;
        best_tile_iperm = nullptr;
    }
};

/**
 * @brief Evaluate tile ordering strategies for a partition
 */
inline PartitionTileResult evaluate_partition_tile_strategies(
    const PartitionData& pd,
    const PartitionTileGraph& graph,
    int partition_id,
    int tile_size,
    TileIndexer::Method method,
    int group_index,
    const std::vector<int>& strategies,
    int num_cores)
{
    PartitionTileResult result;
    result.partition_id = partition_id;
    result.tiles_dim = graph.tiles_dim;

    // Baseline: no tile-level reordering
    TileIndexer::State baseline_state;
    int baseline_active = TileIndexer::countActiveTiles(
        graph.tile_indices_i, graph.tile_indices_j, graph.tile_nnz,
        graph.tiles_dim, 1, method, &baseline_state, 1
    );

    int baseline_filled = -1;
    if (baseline_active >= 0) {
        baseline_filled = TileIndexer::FillTiles(
            baseline_state, method, graph.tiles_dim, baseline_active, 1
        );
        int total = baseline_active + baseline_filled;
        TileIndexer::ensure_diagonal_tiles_active(baseline_state, method,
                                                  graph.tiles_dim, total);
        baseline_filled = total - baseline_active;
    }
    TileIndexer::release_state_resources(baseline_state);

    int min_total = baseline_active + baseline_filled;
    result.best_active = baseline_active;
    result.best_filled = baseline_filled;
    result.best_strategy = "NONE";

    // Try each strategy
    const int num_strategies = static_cast<int>(strategies.size());
    std::vector<StrategyResult> strategy_results(num_strategies);

    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_cores)
    for (int i = 0; i < num_strategies; ++i) {
        auto& res = strategy_results[i];
        res.id = strategies[i];
        res.name = ordering_name_from_id(res.id);

        // Make thread-local copies
        int* tile_i_copy = new int[graph.tile_nnz];
        int* tile_j_copy = new int[graph.tile_nnz];
        std::copy(graph.tile_indices_i, graph.tile_indices_i + graph.tile_nnz, tile_i_copy);
        std::copy(graph.tile_indices_j, graph.tile_indices_j + graph.tile_nnz, tile_j_copy);

        int* tile_perm = nullptr;
        int* tile_iperm = nullptr;
        int* tile_sizes = nullptr;
        int tree_sep = 0;
        int local_tiles_dim = graph.tiles_dim;
        int local_tile_nnz = graph.tile_nnz;

        // Run ordering strategy
        res.status = sTiles::run_permutation(
            &tile_perm, &tile_iperm, &local_tiles_dim, &local_tile_nnz,
            0, &tile_i_copy, &tile_j_copy, res.id, &tree_sep,
            group_index, &tile_sizes, 1
        );

        delete[] tile_i_copy;
        delete[] tile_j_copy;
        if (tile_sizes) delete[] tile_sizes;

        if (res.status == StatusCode::Success && tile_perm && tile_iperm) {
            // Evaluate on partition tile graph
            int* tile_i_perm = new int[graph.tile_nnz];
            int* tile_j_perm = new int[graph.tile_nnz];
            for (int k = 0; k < graph.tile_nnz; ++k) {
                tile_i_perm[k] = tile_perm[graph.tile_indices_i[k]];
                tile_j_perm[k] = tile_perm[graph.tile_indices_j[k]];
            }

            TileIndexer::State ordered_state;
            res.active = TileIndexer::countActiveTiles(
                tile_i_perm, tile_j_perm, graph.tile_nnz,
                graph.tiles_dim, 1, method, &ordered_state, 1
            );

            res.filled = -1;
            if (res.active >= 0) {
                res.filled = TileIndexer::FillTiles(
                    ordered_state, method, graph.tiles_dim, res.active, 1
                );
                int total = res.active + res.filled;
                TileIndexer::ensure_diagonal_tiles_active(ordered_state, method,
                                                         graph.tiles_dim, total);
                res.filled = total - res.active;
            }

            TileIndexer::release_state_resources(ordered_state);
            delete[] tile_i_perm;
            delete[] tile_j_perm;

            // Store permutation for later composition
            res.perm = tile_perm;
            res.iperm = tile_iperm;
        } else {
            OrderingMemoryManager::deallocate(tile_perm);
            OrderingMemoryManager::deallocate(tile_iperm);
        }
    }

    // Select best strategy
    for (int i = 0; i < num_strategies; ++i) {
        const auto& res = strategy_results[i];
        if (res.status == StatusCode::Success && res.active >= 0 && res.filled >= 0) {
            int total = res.active + res.filled;
            if (total < min_total) {
                // Clean up previous best
                if (result.best_tile_perm) {
                    OrderingMemoryManager::deallocate(result.best_tile_perm);
                    OrderingMemoryManager::deallocate(result.best_tile_iperm);
                }

                min_total = total;
                result.best_active = res.active;
                result.best_filled = res.filled;
                result.best_strategy = res.name;
                result.best_tile_perm = strategy_results[i].perm;
                result.best_tile_iperm = strategy_results[i].iperm;
                strategy_results[i].perm = nullptr;  // Prevent cleanup
                strategy_results[i].iperm = nullptr;
            }
        }
    }

    // Clean up non-selected strategies
    for (auto& res : strategy_results) {
        if (res.perm) OrderingMemoryManager::deallocate(res.perm);
        if (res.iperm) OrderingMemoryManager::deallocate(res.iperm);
    }

    return result;
}

/**
 * @brief Compose partition tile permutations into global element permutation
 */
inline void compose_partition_tile_permutations(
    const std::vector<PartitionTileResult>& partition_results,
    const int* element_perm,
    int dim,
    int tile_size,
    const int* partition_sizes,  // Array of 3 partition sizes
    int* composed_perm,
    int* composed_iperm)
{
    // Partition info structure
    struct PartitionInfo {
        int start;
        int size;
        int tiles_dim;
        std::vector<int> tile_sizes;
        std::vector<int> tile_base_positions;
    };

    std::vector<PartitionInfo> partitions(3);

    // Initialize partition boundaries
    partitions[0].start = 0;
    partitions[0].size = partition_sizes[0];
    partitions[1].start = partition_sizes[0];
    partitions[1].size = partition_sizes[1];
    partitions[2].start = partition_sizes[0] + partition_sizes[1];
    partitions[2].size = partition_sizes[2];

    // Compute tile base positions for each partition
    for (int p = 0; p < 3; ++p) {
        auto& part = partitions[p];
        const auto& result = partition_results[p];

        part.tiles_dim = (part.size + tile_size - 1) / tile_size;
        part.tile_sizes.resize(part.tiles_dim);
        part.tile_base_positions.resize(part.tiles_dim);

        // Compute actual size of each tile (last tile may be partial)
        for (int t = 0; t < part.tiles_dim; ++t) {
            int tile_start = t * tile_size;
            int tile_end = std::min((t + 1) * tile_size, part.size);
            part.tile_sizes[t] = tile_end - tile_start;
        }

        // Compute base positions after tile reordering
        if (result.best_tile_perm) {
            int running_pos = 0;
            for (int new_tile_idx = 0; new_tile_idx < part.tiles_dim; ++new_tile_idx) {
                // Find which old tile maps to new_tile_idx
                int old_tile = -1;
                for (int t = 0; t < part.tiles_dim; ++t) {
                    if (result.best_tile_perm[t] == new_tile_idx) {
                        old_tile = t;
                        break;
                    }
                }

                part.tile_base_positions[new_tile_idx] = running_pos;
                running_pos += part.tile_sizes[old_tile];
            }
        } else {
            // No tile reordering for this partition - identity
            for (int t = 0; t < part.tiles_dim; ++t) {
                part.tile_base_positions[t] = t * tile_size;
            }
        }
    }

    // Compose permutations element by element
    for (int e = 0; e < dim; ++e) {
        int current_pos = element_perm[e];  // Position after ND ordering

        // Find which partition this position belongs to
        int partition_id = -1;
        if (current_pos < partitions[0].start + partitions[0].size) {
            partition_id = 0;
        } else if (current_pos < partitions[1].start + partitions[1].size) {
            partition_id = 1;
        } else {
            partition_id = 2;
        }

        const auto& part = partitions[partition_id];
        const auto& result = partition_results[partition_id];

        if (result.best_tile_perm) {
            // Apply partition tile permutation
            int local_pos = current_pos - part.start;
            int local_tile = local_pos / tile_size;
            int local_offset = local_pos % tile_size;
            int new_local_tile = result.best_tile_perm[local_tile];
            composed_perm[e] = part.start + part.tile_base_positions[new_local_tile] + local_offset;
        } else {
            // No tile reordering for this partition
            composed_perm[e] = current_pos;
        }
    }

    // Compute inverse
    compute_inverse_permutation(composed_perm, composed_iperm, dim);
}


StatusCode symbolic_ND_phase(sTiles_call **call_info, TiledMatrix **Tmatrix, int group_index, int num_cores) {

    if (!call_info || !*call_info || !Tmatrix || !*Tmatrix) {
        sTiles::Logger::error("symbolic_ND_phase: null argument(s)");
        return StatusCode::IllegalValue;
    }

    sTiles_call& call = **call_info;
    TiledMatrix& scheme = **Tmatrix;

    const TileIndexer::Method method = scheme.neighbor_lookup_method;
    scheme.nd_padding = 0;
    scheme.L_colptr = nullptr;
    scheme.L_rowind = nullptr;

    // Check for user-provided permutation
    const int* user_perm = get_user_permutation(group_index);
    const int user_perm_size = get_user_permutation_size(group_index);
    const bool user_perm_force = get_user_permutation_force(group_index);

    // ========== BASELINE SYMBOLIC FACTORIZATION ==========
    double t0 = omp_get_wtime();
    TileIndexer::State base_state;
    int active1 = TileIndexer::countActiveTiles(call.row_indices, call.col_indices, scheme.nnz, scheme.dim, scheme.tile_size, method, &base_state, num_cores);

    if (active1 < 0) {
        sTiles::Logger::error("Tile counting failed in baseline path");
        return StatusCode::ExecutionFailed;
    }

    int filled1 = TileIndexer::FillTiles(base_state, method, scheme.dimTiledMatrix, active1, num_cores);
    // Ensure all diagonal tiles are marked as active (required for Cholesky)
    int total_with_diag_baseline = active1 + filled1;
    TileIndexer::ensure_diagonal_tiles_active(base_state, method, scheme.dimTiledMatrix, total_with_diag_baseline);
    filled1 = total_with_diag_baseline - active1;
    sTiles::Logger::timing("│   ↪ Symbolic factorization (baseline pattern): active="
                           + std::to_string(active1)
                           + ", filled=" + std::to_string(filled1)
                           + ", time=" + sTiles::format_seconds(omp_get_wtime() - t0) + " s");
    if (std::getenv("STILES_PROFILE_SYM"))
        sTiles::Logger::timing_always("│   ↪ fill-pattern(countActiveTiles): ", omp_get_wtime() - t0, " s");

    // ========== ND ORDERING WITH PADDING ==========
    {
        const double t_nd_start = omp_get_wtime();

        // Allocate permutation vectors
        int* perm_nd = nullptr;
        int* iperm_nd = nullptr;
        int tree_sep = 0;

        // Make local copies for ND ordering (will be modified)
        int* nd_row_indices = static_cast<int*>(malloc(scheme.nnz * sizeof(int)));
        int* nd_col_indices = static_cast<int*>(malloc(scheme.nnz * sizeof(int)));
        if (!nd_row_indices || !nd_col_indices) {
            free(nd_row_indices);
            free(nd_col_indices);
            return StatusCode::OutOfResources;
        }
        std::copy(call.row_indices, call.row_indices + scheme.nnz, nd_row_indices);
        std::copy(call.col_indices, call.col_indices + scheme.nnz, nd_col_indices);

        int nd_dim = scheme.dim;
        int nd_nnz = scheme.nnz;

        // Check if partition sizes are forced
        int forced_p1 = -1, forced_p2 = -1, forced_sep = -1;
        bool have_forced_partitions = get_forced_partition_sizes(group_index, forced_p1, forced_p2, forced_sep);

        StatusCode sc = StatusCode::Success;
        bool skip_nd_computation = false;

        // If we have both user permutation AND forced partitions, skip ND computation entirely
        bool use_user_perm = (user_perm && user_perm_size == scheme.dim && user_perm_force);
        if (use_user_perm && have_forced_partitions) {
            sTiles::Logger::info("│   ↪ Using forced user permutation and partition sizes (skipping ND computation)");

            // Allocate partition_sizes and copy forced values
            scheme.partition_sizes = OrderingMemoryManager::allocate<int>(3, group_index);
            if (!scheme.partition_sizes) {
                sTiles::Logger::error("Failed to allocate partition_sizes");
                free(nd_row_indices);
                free(nd_col_indices);
                TileIndexer::release_state_resources(base_state);
                return StatusCode::OutOfResources;
            }
            scheme.partition_sizes[0] = forced_p1;
            scheme.partition_sizes[1] = forced_p2;
            scheme.partition_sizes[2] = forced_sep;

            // Allocate and copy user permutation
            perm_nd = OrderingMemoryManager::allocate<int>(scheme.dim, group_index);
            iperm_nd = OrderingMemoryManager::allocate<int>(scheme.dim, group_index);

            if (!perm_nd || !iperm_nd) {
                sTiles::Logger::error("Failed to allocate memory for user permutation");
                free(nd_row_indices);
                free(nd_col_indices);
                OrderingMemoryManager::deallocate(perm_nd);
                OrderingMemoryManager::deallocate(iperm_nd);
                OrderingMemoryManager::deallocate(scheme.partition_sizes);
                TileIndexer::release_state_resources(base_state);
                return StatusCode::OutOfResources;
            }

            std::copy(user_perm, user_perm + scheme.dim, perm_nd);
            compute_inverse_permutation(perm_nd, iperm_nd, scheme.dim);

            sTiles::Logger::info("│   ↪ User permutation set (size=" + std::to_string(scheme.dim) + ")");
            sTiles::Logger::info("│   ↪ Forced partition sizes: P1=" + std::to_string(forced_p1) +
                                ", P2=" + std::to_string(forced_p2) + ", Sep=" + std::to_string(forced_sep));
            skip_nd_computation = true;
        } else {
            // Run ND ordering to get partition sizes (even if we'll use user permutation)
            // We need partition sizes to compute padding requirements
            sc = sTiles::run_permutation(
                &perm_nd, &iperm_nd, &nd_dim, &nd_nnz,
                scheme.fixed_column_size, &nd_row_indices, &nd_col_indices,
                2,  // ordering=2 for METIS ND
                &tree_sep, group_index, &scheme.partition_sizes, num_cores
            );
        }

        if (!skip_nd_computation && (sc != StatusCode::Success || !scheme.partition_sizes)) {
            sTiles::Logger::error("ND ordering failed");
            free(nd_row_indices);
            free(nd_col_indices);
            OrderingMemoryManager::deallocate(perm_nd);
            OrderingMemoryManager::deallocate(iperm_nd);
            // Fall back to baseline (no ordering)
            scheme.use_ordering = 0;
            scheme.state = std::move(base_state);
            scheme.numActiveTiles = filled1;
            scheme.red_tree_separator_level = 0;
            scheme.nd_padding = 0;
            sTiles::Logger::info("│   ↪ Falling back to baseline (no ordering)");
            // Control flows to finalization section
        } else {
            // If ND was computed and user permutation is forced, replace the permutation
            if (!skip_nd_computation && use_user_perm) {
                sTiles::Logger::info("│   ↪ ND partition computed, but using forced user permutation");
                // Note: ND ordering timing already logged by run_permutation()

                // Discard computed ND permutation, use user permutation instead
                OrderingMemoryManager::deallocate(perm_nd);
                OrderingMemoryManager::deallocate(iperm_nd);

                // Allocate and copy user permutation
                perm_nd = OrderingMemoryManager::allocate<int>(scheme.dim, group_index);
                iperm_nd = OrderingMemoryManager::allocate<int>(scheme.dim, group_index);

                if (!perm_nd || !iperm_nd) {
                    sTiles::Logger::error("Failed to allocate memory for user permutation");
                    free(nd_row_indices);
                    free(nd_col_indices);
                    OrderingMemoryManager::deallocate(perm_nd);
                    OrderingMemoryManager::deallocate(iperm_nd);
                    TileIndexer::release_state_resources(base_state);
                    return StatusCode::OutOfResources;
                }

                std::copy(user_perm, user_perm + scheme.dim, perm_nd);
                compute_inverse_permutation(perm_nd, iperm_nd, scheme.dim);

                sTiles::Logger::info("│   ↪ User permutation set (size=" + std::to_string(scheme.dim) + ")");
            }
            // Note: ND ordering timing already logged by run_permutation()

            // Extract partition sizes (allocated by run_permutation or set above if forced)
            int p1_size = scheme.partition_sizes[0];
            int p2_size = scheme.partition_sizes[1];
            int sep_size = scheme.partition_sizes[2];

            sTiles::Logger::info("│   ↪ ND partition sizes: P1=" +
                std::to_string(p1_size) + ", P2=" +
                std::to_string(p2_size) + ", Sep=" +
                std::to_string(sep_size));

            // ========== STEP 3: CALCULATE PADDING REQUIREMENTS ==========
            // Calculate padding to align partitions with tile boundaries
            auto calculate_padding = [](int size, int tile_size) -> int {
                int remainder = size % tile_size;
                return (remainder == 0) ? 0 : (tile_size - remainder);
            };

            int pad_p1 = calculate_padding(p1_size, scheme.tile_size);
            int pad_p2 = calculate_padding(p2_size, scheme.tile_size);
            int pad_sep = calculate_padding(sep_size, scheme.tile_size);
            int total_padding = pad_p1 + pad_p2 + pad_sep;
            scheme.nd_padding = total_padding;

            sTiles::Logger::info("│   ↪ Padding required: P1+" +
                std::to_string(pad_p1) + ", P2+" +
                std::to_string(pad_p2) + ", Sep+" +
                std::to_string(pad_sep) + " (total=" +
                std::to_string(total_padding) + ")");

            if (total_padding == 0) {
                sTiles::Logger::info("│   ↪ Partitions already tile-aligned, no padding needed");
            }

            // ========== STEP 4: EXTEND MATRIX WITH DIAGONAL PADDING ==========
            // Extend dimension and nnz
            int old_dim = nd_dim;
            int old_nnz = nd_nnz;
            int new_dim = old_dim + total_padding;
            int new_nnz = old_nnz + total_padding;  // Add diagonal elements

            // Reallocate arrays with padding using malloc (to match original allocation)
            int* padded_row_indices = static_cast<int*>(malloc(new_nnz * sizeof(int)));
            int* padded_col_indices = static_cast<int*>(malloc(new_nnz * sizeof(int)));

            if (!padded_row_indices || !padded_col_indices) {
                sTiles::Logger::error("Failed to allocate padded index arrays");
                free(padded_row_indices);
                free(padded_col_indices);
                free(nd_row_indices);
                free(nd_col_indices);
                OrderingMemoryManager::deallocate(perm_nd);
                OrderingMemoryManager::deallocate(iperm_nd);
                TileIndexer::release_state_resources(base_state);
                return StatusCode::OutOfResources;
            }

            // Copy original indices
            std::copy(nd_row_indices, nd_row_indices + old_nnz, padded_row_indices);
            std::copy(nd_col_indices, nd_col_indices + old_nnz, padded_col_indices);

            // Add diagonal padding elements at the end
            for (int i = 0; i < total_padding; ++i) {
                int padded_index = old_dim + i;
                padded_row_indices[old_nnz + i] = padded_index;
                padded_col_indices[old_nnz + i] = padded_index;
            }

            // Clean up temporary arrays
            free(nd_row_indices);
            free(nd_col_indices);
            nd_row_indices = padded_row_indices;
            nd_col_indices = padded_col_indices;
            nd_dim = new_dim;
            nd_nnz = new_nnz;

            // ========== STEP 5: EXTEND PERMUTATION VECTORS ==========
            // Allocate extended permutation
            int* extended_perm = OrderingMemoryManager::allocate<int>(new_dim, group_index);
            int* extended_iperm = OrderingMemoryManager::allocate<int>(new_dim, group_index);

            if (!extended_perm || !extended_iperm) {
                sTiles::Logger::error("Failed to allocate extended permutation");
                free(nd_row_indices);
                free(nd_col_indices);
                OrderingMemoryManager::deallocate(perm_nd);
                OrderingMemoryManager::deallocate(iperm_nd);
                OrderingMemoryManager::deallocate(extended_perm);
                OrderingMemoryManager::deallocate(extended_iperm);
                TileIndexer::release_state_resources(base_state);
                return StatusCode::OutOfResources;
            }

            // Copy original permutation (shifts will be applied below)
            std::copy(perm_nd, perm_nd + old_dim, extended_perm);

            // Shift mappings to account for padding insertion:
            // - P1 elements (0 to p1_size-1): no shift needed
            // - P2 elements (p1_size to p1_size+p2_size-1): shift by pad_p1
            // - Sep elements (p1_size+p2_size to old_dim-1): shift by pad_p1+pad_p2

            for (int i = 0; i < old_dim; ++i) {
                int mapped_pos = extended_perm[i];

                if (mapped_pos >= p1_size + p2_size) {
                    // Separator element: shift by pad_p1 + pad_p2
                    extended_perm[i] = mapped_pos + pad_p1 + pad_p2;
                } else if (mapped_pos >= p1_size) {
                    // P2 element: shift by pad_p1
                    extended_perm[i] = mapped_pos + pad_p1;
                }
                // P1 elements: no shift needed
            }

            // Add padding indices to permutation
            // Strategy: Map padding indices to partition boundaries in permuted space

            int pad_idx = old_dim;

            // P1 padding: map to positions [p1_size, p1_size+pad_p1)
            for (int i = 0; i < pad_p1; ++i) {
                extended_perm[pad_idx++] = p1_size + i;
            }

            // P2 padding: map to positions [p1_size+pad_p1+p2_size, p1_size+pad_p1+p2_size+pad_p2)
            for (int i = 0; i < pad_p2; ++i) {
                extended_perm[pad_idx++] = p1_size + pad_p1 + p2_size + i;
            }

            // Separator padding: map to positions [p1_size+pad_p1+p2_size+pad_p2+sep_size, new_dim)
            for (int i = 0; i < pad_sep; ++i) {
                extended_perm[pad_idx++] = p1_size + pad_p1 + p2_size + pad_p2 + sep_size + i;
            }

            // Compute inverse permutation
            compute_inverse_permutation(extended_perm, extended_iperm, new_dim);

            // Validate permutation
            if (!check_inverse_permutation(extended_perm, extended_iperm, new_dim)) {
                sTiles::Logger::error("Extended permutation validation failed!");
                free(nd_row_indices);
                free(nd_col_indices);
                OrderingMemoryManager::deallocate(extended_perm);
                OrderingMemoryManager::deallocate(extended_iperm);
                OrderingMemoryManager::deallocate(perm_nd);
                OrderingMemoryManager::deallocate(iperm_nd);
                TileIndexer::release_state_resources(base_state);
                return StatusCode::IllegalValue;
            }

            // Clean up original permutation
            OrderingMemoryManager::deallocate(perm_nd);
            OrderingMemoryManager::deallocate(iperm_nd);

            // ========== STEP 6: COUNT TILES WITH PADDED PERMUTATION ==========
            // Count active tiles with padded permutation
            TileIndexer::State nd_state;
            int active_nd = count_active_tiles_with_perm(
                nd_row_indices, nd_col_indices,
                nd_nnz, nd_dim, scheme.tile_size,
                extended_perm, method, &nd_state,
                num_cores, group_index
            );

            if (active_nd < 0) {
                sTiles::Logger::error("Tile counting failed with padded ND ordering");
                free(nd_row_indices);
                free(nd_col_indices);
                OrderingMemoryManager::deallocate(extended_perm);
                OrderingMemoryManager::deallocate(extended_iperm);
                TileIndexer::release_state_resources(nd_state);
                TileIndexer::release_state_resources(base_state);
                return StatusCode::ExecutionFailed;
            }

            // Compute fill-in (only if promising)
            int filled_nd = -1;
            if (active_nd <= active1 + filled1) {
                int new_dimTiledMatrix = (new_dim + scheme.tile_size - 1) / scheme.tile_size;
                filled_nd = TileIndexer::FillTiles(nd_state, method, new_dimTiledMatrix, active_nd, num_cores);
            }

            // ALWAYS ensure all diagonal tiles are marked as active (required for Cholesky)
            // This must happen even if FillTiles failed, because we need diagonals for the mapper
            if (active_nd >= 0) {
                int new_dimTiledMatrix = (new_dim + scheme.tile_size - 1) / scheme.tile_size;
                int total_count = active_nd + (filled_nd >= 0 ? filled_nd : 0);
                TileIndexer::ensure_diagonal_tiles_active(nd_state, method, new_dimTiledMatrix, total_count);
                filled_nd = total_count - active_nd;
            }

            const double nd_elapsed = omp_get_wtime() - t_nd_start;
            if (std::getenv("STILES_PROFILE_SYM"))
                sTiles::Logger::timing_always("│   ↪ ND ordering(SCOTCH+padding): ", nd_elapsed, " s");

            sTiles::Logger::timing("│   ↪ Symbolic factorization (ND with padding): active=" +
                std::to_string(active_nd) + ", filled=" +
                std::to_string(filled_nd) + ", time=" +
                sTiles::format_seconds(nd_elapsed) + " s");

            // ========== STEP 7: APPLY ND UNCONDITIONALLY ==========
            // Always use ND ordering (no comparison with baseline)
            sTiles::Logger::info("│   ↪ Applying ND ordering with padding");

            // Update scheme with padded matrix and permutation
            scheme.use_ordering = 1;
            scheme.element_perm = extended_perm;
            scheme.element_iperm = extended_iperm;

            // Update dimensions and nnz so downstream tiling allocates correct sizes
            scheme.dim = new_dim;
            scheme.nnz = new_nnz;
            scheme.dimTiledMatrix = (new_dim + scheme.tile_size - 1) / scheme.tile_size;
            scheme.original_order = new_dim;
            scheme.original_nnz   = new_nnz;
            call.order = new_dim;
            call.nnz   = new_nnz;

            // Update partition sizes (now tile-aligned)
            scheme.partition_sizes[0] = p1_size + pad_p1;
            scheme.partition_sizes[1] = p2_size + pad_p2;
            scheme.partition_sizes[2] = sep_size + pad_sep;

            // Replace original indices with padded version
            // Note: We don't free the old pointers here because the caller may still
            // have references to them and will handle cleanup
            call.row_indices = nd_row_indices;
            call.col_indices = nd_col_indices;

            // Update state
            TileIndexer::release_state_resources(base_state);
            scheme.state = std::move(nd_state);
            scheme.red_tree_separator_level = tree_sep;
            scheme.numActiveTiles = filled_nd;

            sTiles::Logger::info("│   ↪ Padded partition sizes (tile-aligned): P1=" +
                std::to_string(scheme.partition_sizes[0]) + ", P2=" +
                std::to_string(scheme.partition_sizes[1]) + ", Sep=" +
                std::to_string(scheme.partition_sizes[2]));
        }
    }

    // ========== PER-PARTITION TILE-LEVEL ORDERING ==========
    // Apply tile-level ordering independently to each partition (P1, P2, Separator)
    if (scheme.use_ordering && scheme.partition_sizes &&
        stiles_control_params[4] > 0 && !user_perm_force) {

        const double t_partition_tile = omp_get_wtime();
        sTiles::Logger::debug("│   ↪ Starting per-partition tile-level ordering");

        // Check if user forced permutation (shouldn't reach here, but double-check)
        const bool user_perm = (user_perm_force);
        if (user_perm) {
            sTiles::Logger::info("│   ↪ User permutation forced, skipping per-partition tile ordering");
        } else {
            // Extract partition boundaries
            const int p1_size = scheme.partition_sizes[0];
            const int p2_size = scheme.partition_sizes[1];
            const int sep_size = scheme.partition_sizes[2];

            // Parse tile-level ordering strategies
            auto strategies = unique_digits_1_to_9(stiles_control_params[4]);

            if (strategies.empty()) {
                sTiles::Logger::warning("│   ↪ No valid tile ordering strategies found");
            } else {
                // Store partition data and results
                std::vector<PartitionData> partition_data(3);
                std::vector<PartitionTileGraph> partition_graphs(3);
                std::vector<PartitionTileResult> partition_results(3);

                const std::string part_names[3] = {"P1", "P2", "Sep"};
                const int part_starts[3] = {0, p1_size, p1_size + p2_size};
                const int part_sizes[3] = {p1_size, p2_size, sep_size};

                // Process each partition
                for (int p = 0; p < 3; ++p) {
                    // Skip empty or tiny partitions
                    if (part_sizes[p] < scheme.tile_size) {
                        sTiles::Logger::info("│   ↪ Partition " + part_names[p] +
                                           " too small (size=" + std::to_string(part_sizes[p]) +
                                           "), skipping tile ordering");
                        partition_results[p].partition_id = p;
                        partition_results[p].tiles_dim = 0;
                        continue;
                    }

                    // Extract partition submatrix
                    partition_data[p] = extract_partition_submatrix(
                        call.row_indices, call.col_indices, scheme.nnz,
                        scheme.element_perm, part_starts[p], part_sizes[p],
                        group_index
                    );

                    if (partition_data[p].local_nnz == 0) {
                        sTiles::Logger::info("│   ↪ Partition " + part_names[p] +
                                           " has no entries, skipping tile ordering");
                        partition_results[p].partition_id = p;
                        partition_results[p].tiles_dim = 0;
                        continue;
                    }

                    // Build partition tile graph
                    partition_graphs[p] = build_partition_tile_graph(
                        partition_data[p], scheme.tile_size, group_index
                    );

                    if (partition_graphs[p].tiles_dim <= 1) {
                        sTiles::Logger::info("│   ↪ Partition " + part_names[p] +
                                           " has only " + std::to_string(partition_graphs[p].tiles_dim) +
                                           " tile(s), skipping tile ordering");
                        partition_results[p].partition_id = p;
                        partition_results[p].tiles_dim = partition_graphs[p].tiles_dim;
                        continue;
                    }

                    // Evaluate strategies for this partition
                    partition_results[p] = evaluate_partition_tile_strategies(
                        partition_data[p], partition_graphs[p], p,
                        scheme.tile_size, method, group_index,
                        strategies, num_cores
                    );

                    // Log partition result
                    sTiles::Logger::debug("│   ↪ Partition " + part_names[p] +
                                         ": tiles_dim=" + std::to_string(partition_results[p].tiles_dim) +
                                         ", best_strategy=" + partition_results[p].best_strategy +
                                         ", active=" + std::to_string(partition_results[p].best_active) +
                                         ", filled=" + std::to_string(partition_results[p].best_filled));
                }

                // Compose partition tile permutations into global permutation
                int* composed_perm = OrderingMemoryManager::allocate<int>(scheme.dim, group_index);
                int* composed_iperm = OrderingMemoryManager::allocate<int>(scheme.dim, group_index);

                if (!composed_perm || !composed_iperm) {
                    sTiles::Logger::error("Failed to allocate composed permutation arrays");
                    if (composed_perm) OrderingMemoryManager::deallocate(composed_perm);
                    if (composed_iperm) OrderingMemoryManager::deallocate(composed_iperm);
                } else {
                    // Compose permutations
                    compose_partition_tile_permutations(
                        partition_results, scheme.element_perm, scheme.dim,
                        scheme.tile_size, scheme.partition_sizes,
                        composed_perm, composed_iperm
                    );

                    // Validate composed permutation
                    if (!check_inverse_permutation(composed_perm, composed_iperm, scheme.dim)) {
                        sTiles::Logger::error("Composed permutation validation failed!");
                        OrderingMemoryManager::deallocate(composed_perm);
                        OrderingMemoryManager::deallocate(composed_iperm);
                    } else {
                        // Evaluate composed permutation on full matrix
                        TileIndexer::State final_state;
                        int active_final = count_active_tiles_with_perm(
                            call.row_indices, call.col_indices, scheme.nnz,
                            scheme.dim, scheme.tile_size, composed_perm,
                            method, &final_state, num_cores, group_index
                        );

                        int filled_final = -1;
                        // Compute ND baseline from current numActiveTiles (which stores filled_nd)
                        // We need to recount active tiles from current state for accurate baseline
                        int nd_baseline_total = scheme.numActiveTiles;

                        // Get active count from current ND state for comparison
                        TileIndexer::State temp_baseline_state;
                        int nd_active = count_active_tiles_with_perm(
                            call.row_indices, call.col_indices, scheme.nnz,
                            scheme.dim, scheme.tile_size, scheme.element_perm,
                            method, &temp_baseline_state, num_cores, group_index
                        );
                        if (nd_active >= 0) {
                            int nd_filled = TileIndexer::FillTiles(
                                temp_baseline_state, method, scheme.dimTiledMatrix, nd_active, num_cores
                            );
                            nd_baseline_total = nd_active + nd_filled;
                        }
                        TileIndexer::release_state_resources(temp_baseline_state);

                        if (active_final >= 0 && active_final <= nd_baseline_total) {
                            filled_final = TileIndexer::FillTiles(
                                final_state, method, scheme.dimTiledMatrix, active_final, num_cores
                            );
                            int total_with_diag = active_final + filled_final;
                            TileIndexer::ensure_diagonal_tiles_active(final_state, method,
                                                                     scheme.dimTiledMatrix, total_with_diag);
                            filled_final = total_with_diag - active_final;
                        }

                        // Accept only if improves over ND baseline
                        if (active_final >= 0 && filled_final >= 0 &&
                            active_final + filled_final < nd_baseline_total) {

                            sTiles::Logger::debug("│   ↪ Per-partition tile ordering accepted: active=" +
                                                 std::to_string(active_final) + ", filled=" +
                                                 std::to_string(filled_final) +
                                                 " (improved from " + std::to_string(nd_baseline_total) + ")");
                            // Record winning tile strategies per partition
                            {
                                std::string parts;
                                for (int p = 0; p < 3; ++p) {
                                    if (partition_results[p].best_strategy != "NONE") {
                                        if (!parts.empty()) parts += "/";
                                        parts += partition_results[p].best_strategy;
                                    }
                                }
                                scheme.selected_tile_ordering = parts.empty() ? "partition" : parts;
                            }

                            // Replace permutation
                            OrderingMemoryManager::deallocate(scheme.element_perm);
                            OrderingMemoryManager::deallocate(scheme.element_iperm);
                            TileIndexer::release_state_resources(scheme.state);

                            scheme.element_perm = composed_perm;
                            scheme.element_iperm = composed_iperm;
                            scheme.state = std::move(final_state);
                            scheme.numActiveTiles = filled_final;

                            composed_perm = nullptr;  // Prevent cleanup
                            composed_iperm = nullptr;
                        } else {
                            sTiles::Logger::debug(std::string("│   ↪ Per-partition tile ordering rejected: no improvement ") +
                                                 "(current=" + std::to_string(active_final + filled_final) +
                                                 ", baseline=" + std::to_string(nd_baseline_total) + ")");
                            scheme.selected_tile_ordering = "rejected";
                            TileIndexer::release_state_resources(final_state);
                        }

                        // Clean up composed permutation if not accepted
                        if (composed_perm) OrderingMemoryManager::deallocate(composed_perm);
                        if (composed_iperm) OrderingMemoryManager::deallocate(composed_iperm);
                    }
                }

                // Clean up partition data
                for (int p = 0; p < 3; ++p) {
                    partition_data[p].cleanup(group_index);
                    partition_graphs[p].cleanup(group_index);
                    partition_results[p].cleanup(group_index);
                }

                const double partition_tile_elapsed = omp_get_wtime() - t_partition_tile;
                sTiles::Logger::debug("│   ↪ Per-partition tile ordering completed in " +
                                     sTiles::format_seconds(partition_tile_elapsed) + " s");
            }
        }
    }

    // ========== FINALIZATION ==========
    // Normalize separator level (only if ND was applied)
    if (scheme.use_ordering) {
        if (scheme.num_cores < 2) {
            scheme.red_tree_separator_level = 0;
        } else {
            scheme.red_tree_separator_level = (scheme.red_tree_separator_level + scheme.tile_size - 1) / scheme.tile_size;
            if ((scheme.dim / std::max(1, scheme.tile_size - 1)) < (2 * scheme.num_cores)) {
                scheme.red_tree_separator_level = 0;
            }
        }
    }

    sTiles::Logger::debug("│     • symbolic_ND_phase complete: active_tiles=" +
                          std::to_string(scheme.numActiveTiles) +
                          ", use_ordering=" + std::to_string(scheme.use_ordering));

    return StatusCode::Success;
}

} // namespace preprocess
} // namespace sTiles


#endif // STILES_LOGGER_H

