/**
 * @file free.cpp
 * @brief Implementation of TiledMatrix / scheme cleanup routines.
 *
 * See free.hpp for the public surface. These functions are called from
 * sTiles_freeGroup and sTiles_quit in tools/process/process.cpp.
 */

#include "free.hpp"

#include <cstdlib>
#include <set>
#include <vector>

#include "../tile/meta.hpp"
#include "../TileIndexer/TileIndexer.hpp"
#include "../TileIndexer/TileIndexerMapper.hpp"
#include "../sparse/api.hpp"

#ifdef STILES_GPU
#include "../gpu/compute_gpu.hpp"  // For sTiles::gpu::GpuPersistentContext
#endif

namespace sTiles {

void destroy_tiled_matrix(TiledMatrix* tm) {
    if (!tm) return;

    // Release the per-call supernodal sparse handle (tile_type_mode==2 path).
    // Each TiledMatrix owns its own; freeGroup deletes the underlying Handle
    // (etree, symbolic, CellStore for L/Z) and nulls the pointer.
    if (tm->sparse_handle) {
        sTiles::sparse::api::freeGroup(&tm->sparse_handle);
    }

    // Free persistent GPU context (handles, streams, events) - shared across chol/inv/solve
    #ifdef STILES_GPU
    if (tm->gpu_persistent_ctx) {
        auto* persistent = static_cast<sTiles::gpu::GpuPersistentContext*>(tm->gpu_persistent_ctx);
        delete persistent;
        tm->gpu_persistent_ctx = nullptr;
    }
    #endif

    // Free workspaces array and individual workspace objects
    if (tm->workspaces) {
        for (int r = 0; r < tm->num_workspaces; ++r) {
            if (tm->workspaces[r]) {
                delete tm->workspaces[r];
                tm->workspaces[r] = nullptr;
            }
        }
        std::free(tm->workspaces);
        tm->workspaces = nullptr;
        tm->num_workspaces = 0;
    }

    // Free CSC factor values array packed by pack_L_values()
    if (tm->L_values) {
        delete[] tm->L_values;
        tm->L_values = nullptr;
    }
    if (tm->L_src) {
        delete[] tm->L_src;
        tm->L_src = nullptr;
    }

    // Free persistent byte-progress buffers (allocated once at init_group
    // for the chol executor; sized numActiveTiles bytes / slots). Safe
    // under -UNSTILES_BYTE_PROGRESS as well — the legacy int path simply
    // never uses these pointers.
    if (tm->byte_progress_buf) {
        std::free((void*)tm->byte_progress_buf);
        tm->byte_progress_buf = nullptr;
    }
    if (tm->byte_progress_buf_omp) {
        delete[] tm->byte_progress_buf_omp;
        tm->byte_progress_buf_omp = nullptr;
    }

    // Release legacy TileIndexer state (older path)
    TileIndexer::release_state_resources(tm->tile_indexer_state);

    // Release new TileIndexer unified State (containers with custom allocators)
    TileIndexer::release_state_resources(tm->state);

    // Reset mapper variant to drop any internal unordered_map/vector storage
    tm->mapper = TileIndexer::Mapper{};

    tm->tile_indexer_graph.offsets.clear();
    tm->tile_indexer_graph.offsets.shrink_to_fit();
    tm->tile_indexer_graph.edges.clear();
    tm->tile_indexer_graph.edges.shrink_to_fit();
    // Release underlying buffers no matter whether shrink_to_fit is honored
    std::vector<int>().swap(tm->tile_indexer_graph.offsets);
    std::vector<int>().swap(tm->tile_indexer_graph.edges);

    // Call destructors for objects with std::vector members (constructed with placement-new)
    // These must be destroyed before TileMemoryManager::freeAll() to release internal allocations
    const int num_active = (tm->numActiveTiles > 0) ? tm->numActiveTiles : 0;

    if (tm->semisparseTileMetaCore && num_active > 0) {
        for (int t = 0; t < num_active; ++t) {
            tm->semisparseTileMetaCore[t].~SemisparseTileMetaCore();
        }
        tm->semisparseTileMetaCore = nullptr;
    }

    // Also destroy SparseTileMetaCore and SparseTileMetaData (contain vectors)
    if (tm->sparseTileMetaCore && num_active > 0) {
        for (int t = 0; t < num_active; ++t) {
            tm->sparseTileMetaCore[t].~SparseTileMetaCore();
        }
        tm->sparseTileMetaCore = nullptr;
    }
    if (tm->invSparseTileMetaCore && num_active > 0) {
        for (int t = 0; t < num_active; ++t) {
            tm->invSparseTileMetaCore[t].~SparseTileMetaCore();
        }
        tm->invSparseTileMetaCore = nullptr;
    }
    if (tm->sparseTileMetaData && num_active > 0) {
        for (int t = 0; t < num_active; ++t) {
            tm->sparseTileMetaData[t].~SparseTileMetaData();
        }
        tm->sparseTileMetaData = nullptr;
    }

    tm->chol_tasks.reset();
    tm->chol_task_offsets.reset();
    tm->inv_tasks.reset();
    tm->inv_task_offsets.reset();
    tm->solve_fwd_tasks.reset();
    tm->solve_fwd_offsets.reset();
    tm->solve_bwd_tasks.reset();
    tm->solve_bwd_offsets.reset();
    tm->gpu_solve_fwd_tasks.reset();
    tm->gpu_solve_bwd_tasks.reset();
    tm->gpu_solve_fwd_offsets.reset();
    tm->gpu_solve_bwd_offsets.reset();
}

void destroy_all_schemes_for_group(sTiles_object* s, int group_index) {
    if (!s || group_index < 0 || group_index >= s->num_call_groups) return;
    sTiles_group& grp = s->stiles_groups[group_index];
    for (int i = 0; i < grp.num_calls; ++i) {
        const sTiles_call& call = grp.stiles_calls[i];
        const int global = call.global_index;
        if (global >= 0 && global < s->total_calls) {
            destroy_tiled_matrix(s->schemes[global]);
            // Drop pointer reference to avoid accidental reuse after group free
            s->schemes[global] = nullptr;
        }
    }
}

void destroy_all_schemes(sTiles_object* s) {
    if (!s) return;

    // Track which tile metadata arrays have been freed to avoid double-free
    std::set<void*> freed_semisparse;
    std::set<void*> freed_sparse;
    std::set<void*> freed_inv_sparse;
    std::set<void*> freed_sparse_data;

    for (int i = 0; i < s->total_calls; ++i) {
        TiledMatrix* tm = s->schemes ? s->schemes[i] : nullptr;
        if (!tm) continue;

        // Check if this pointer was already destroyed (appears multiple times in array)
        bool already_destroyed = false;
        for (int j = 0; j < i; ++j) {
            if (s->schemes[j] == tm) {
                already_destroyed = true;
                break;
            }
        }

        if (already_destroyed) {
            s->schemes[i] = nullptr;
            continue;
        }

        // Mark shared tile metadata as already freed before calling destroy
        if (tm->semisparseTileMetaCore && freed_semisparse.count(tm->semisparseTileMetaCore)) {
            tm->semisparseTileMetaCore = nullptr;
        } else if (tm->semisparseTileMetaCore) {
            freed_semisparse.insert(tm->semisparseTileMetaCore);
        }

        if (tm->sparseTileMetaCore && freed_sparse.count(tm->sparseTileMetaCore)) {
            tm->sparseTileMetaCore = nullptr;
        } else if (tm->sparseTileMetaCore) {
            freed_sparse.insert(tm->sparseTileMetaCore);
        }

        if (tm->invSparseTileMetaCore && freed_inv_sparse.count(tm->invSparseTileMetaCore)) {
            tm->invSparseTileMetaCore = nullptr;
        } else if (tm->invSparseTileMetaCore) {
            freed_inv_sparse.insert(tm->invSparseTileMetaCore);
        }

        if (tm->sparseTileMetaData && freed_sparse_data.count(tm->sparseTileMetaData)) {
            tm->sparseTileMetaData = nullptr;
        } else if (tm->sparseTileMetaData) {
            freed_sparse_data.insert(tm->sparseTileMetaData);
        }

        destroy_tiled_matrix(tm);
        s->schemes[i] = nullptr;
    }
}

} // namespace sTiles
