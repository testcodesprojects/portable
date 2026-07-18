#include "ordering_utils.hpp"

#ifdef STILES_WITH_SUITESPARSE
#include <suitesparse/amd.h>
#include <suitesparse/camd.h>
#include <suitesparse/colamd.h>
#include <suitesparse/ccolamd.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "../common/stiles_logger.hpp"

namespace sTiles {

namespace {

void* suitesparse_allocate(size_t n, size_t size) {
    return std::calloc(n, size);
}

void suitesparse_release(void* p) {
    std::free(p);
}

enum class SuiteSparseStrategy {
    AMD = 0,
    CAMD = 1,
    COLAMD = 2,
    CCOLAMD = 3,
    SYMAMD = 4
};

}

static int fill_permutation_from_vector(int N, int dim, const std::vector<int>& src, int** perm, int** iperm) {
    for (int i = 0; i < dim; ++i) {
        (*iperm)[i] = src[i];
    }
    for (int i = dim; i < N; ++i) {
        (*iperm)[i] = i;
    }

    for (int i = 0; i < N; ++i) {
        const int p = (*iperm)[i];
        if (p < 0 || p >= N) {
            return 1;
        }
        (*perm)[p] = i;
    }
    return 0;
}

int runSuiteSparse(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, int num_cores, int strategy_num, const SharedAdjCSR* shared) {
    (void) num_cores;
    const int dim = N - m;
    if (dim <= 0) {
        sTiles::Logger::errorf("runSuiteSparse: dim = N - m <= 0 (N = %d, m = %d)", N, m);
        return 1;
    }

    const SuiteSparseStrategy strategy = static_cast<SuiteSparseStrategy>(strategy_num);

    // The prebuilt canonical graph is diagonal-free; AMD/CAMD ignore diagonal
    // entries, so it yields an identical ordering while skipping the rebuild.
    // COLAMD/CCOLAMD/SYMAMD have AᵀA semantics and/or mutate the arrays in place,
    // so they keep their own build.
    const bool use_shared = (shared && shared->valid_for(dim)
                             && (strategy == SuiteSparseStrategy::AMD
                                 || strategy == SuiteSparseStrategy::CAMD));

    std::vector<int> col_ptr;   // built only when not reusing the shared graph
    std::vector<int> row_idx;
    if (!use_shared) {
        std::vector<std::vector<int>> columns(dim);
        columns.shrink_to_fit();

        for (int k = 0; k < nnz; ++k) {
            const int r = (*csr_i)[k];
            const int c = (*csr_j)[k];
            if (r < 0 || c < 0 || r >= dim || c >= dim) {
                continue;
            }
            columns[c].push_back(r);
            if (r != c) {
                columns[r].push_back(c);
            }
        }

        col_ptr.resize(dim + 1);
        size_t total = 0;
        for (int col = 0; col < dim; ++col) {
            auto &entries = columns[col];
            std::sort(entries.begin(), entries.end());
            entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
            col_ptr[col] = static_cast<int>(total);
            total += entries.size();
        }
        col_ptr[dim] = static_cast<int>(total);

        row_idx.resize(total);
        size_t offset = 0;
        for (int col = 0; col < dim; ++col) {
            for (int row : columns[col]) {
                row_idx[offset++] = row;
            }
        }
    }

    // Canonical pointers consumed by AMD/CAMD below — either the shared graph or
    // the just-built one. (COLAMD/CCOLAMD/SYMAMD always take the built arrays.)
    const int* CP = use_shared ? shared->xadj.data()   : col_ptr.data();
    const int* RI = use_shared ? shared->adjncy.data() : row_idx.data();

    switch (strategy) {
        case SuiteSparseStrategy::CAMD:
        {
            std::vector<double> control(CAMD_CONTROL);
            std::vector<double> info(CAMD_INFO);
            camd_defaults(control.data());
            std::vector<int> camd_perm(dim);
            const int status = camd_order(dim, CP, RI, camd_perm.data(), control.data(), info.data(), nullptr);
            if (status != CAMD_OK && status != CAMD_OK_BUT_JUMBLED) {
                sTiles::Logger::errorf("runSuiteSparse: camd_order failed with status %d", status);
                return 1;
            }
            if (fill_permutation_from_vector(N, dim, camd_perm, perm, iperm) != 0) {
                sTiles::Logger::errorf("runSuiteSparse: invalid permutation from CAMD");
                return 1;
            }
            return 0;
        }
        case SuiteSparseStrategy::COLAMD:
        case SuiteSparseStrategy::CCOLAMD:
        {
            const size_t recommended = (strategy == SuiteSparseStrategy::COLAMD)
                ? colamd_recommended(static_cast<int32_t>(row_idx.size()), dim, dim)
                : ccolamd_recommended(static_cast<int32_t>(row_idx.size()), dim, dim);
            if (recommended == 0) {
                sTiles::Logger::errorf("runSuiteSparse: %s recommended workspace failed",
                              strategy == SuiteSparseStrategy::COLAMD ? "colamd" : "ccolamd");
                return 1;
            }
            std::vector<int> A_data(recommended);
            std::vector<int> p_data(dim + 1);
            std::copy(row_idx.begin(), row_idx.end(), A_data.begin());
            std::copy(col_ptr.begin(), col_ptr.end(), p_data.begin());
            int success = 0;
            if (strategy == SuiteSparseStrategy::COLAMD) {
                double knobs[COLAMD_KNOBS];
                int stats[COLAMD_STATS];
                colamd_set_defaults(knobs);
                success = colamd(dim, dim, static_cast<int>(recommended), A_data.data(), p_data.data(), knobs, stats);
                if (!success) {
                    sTiles::Logger::errorf("runSuiteSparse: colamd failed with status %d", stats[COLAMD_STATUS]);
                    return 1;
                }
            } else {
                double knobs[CCOLAMD_KNOBS];
                int stats[CCOLAMD_STATS];
                ccolamd_set_defaults(knobs);
                success = ccolamd(dim, dim, static_cast<int>(recommended), A_data.data(), p_data.data(), knobs, stats, nullptr);
                if (!success) {
                    sTiles::Logger::errorf("runSuiteSparse: ccolamd failed with status %d", stats[CCOLAMD_STATUS]);
                    return 1;
                }
            }

            std::vector<int> perm_vec(dim);
            for (int i = 0; i < dim; ++i) {
                perm_vec[i] = p_data[i];
            }
            if (fill_permutation_from_vector(N, dim, perm_vec, perm, iperm) != 0) {
                sTiles::Logger::errorf("runSuiteSparse: invalid permutation from %s",
                              strategy == SuiteSparseStrategy::COLAMD ? "colamd" : "ccolamd");
                return 1;
            }
            return 0;
        }
        case SuiteSparseStrategy::SYMAMD:
        {
            std::vector<int> A_copy(row_idx.begin(), row_idx.end());
            std::vector<int> p_copy(col_ptr.begin(), col_ptr.end());
            std::vector<int> perm_vec(dim + 1);
            double knobs[COLAMD_KNOBS];
            int stats[COLAMD_STATS];
            colamd_set_defaults(knobs);
            const int status = symamd(dim, A_copy.data(), p_copy.data(), perm_vec.data(), knobs, stats,
                                      suitesparse_allocate, suitesparse_release);
            if (!status) {
                sTiles::Logger::errorf("runSuiteSparse: symamd failed with status %d", stats[COLAMD_STATUS]);
                return 1;
            }
            perm_vec.resize(dim);
            if (fill_permutation_from_vector(N, dim, perm_vec, perm, iperm) != 0) {
                sTiles::Logger::errorf("runSuiteSparse: invalid permutation from symamd");
                return 1;
            }
            return 0;
        }
        case SuiteSparseStrategy::AMD:
        default:
        {
            std::vector<double> control(AMD_CONTROL);
            std::vector<double> info(AMD_INFO);
            amd_defaults(control.data());
            std::vector<int> amd_perm(dim);
            const int status = amd_order(dim, CP, RI, amd_perm.data(), control.data(), info.data());
            if (status != AMD_OK && status != AMD_OK_BUT_JUMBLED) {
                sTiles::Logger::errorf("runSuiteSparse: amd_order failed with status %d", status);
                return 1;
            }
            if (fill_permutation_from_vector(N, dim, amd_perm, perm, iperm) != 0) {
                sTiles::Logger::errorf("runSuiteSparse: invalid permutation from AMD");
                return 1;
            }
            return 0;
        }
    }
}

} // namespace sTiles

#else

namespace sTiles {
int runSuiteSparse(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, int num_cores, int strategy_num, const SharedAdjCSR* shared) {
    (void) csr_i; (void) csr_j; (void) N; (void) nnz; (void) m; (void) perm; (void) iperm; (void) num_cores; (void) strategy_num; (void) shared;
    sTiles::Logger::errorf("runSuiteSparse: SuiteSparse support not enabled in this build.");
    return 1;
}
}

#endif
