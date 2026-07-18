/**
 * @file stiles_structs.hpp
 * @brief Core data structures for tiled matrix representation.
 *
 * Defines the fundamental data structures including TiledMatrix, DenseTile,
 * sTiles_object, and related types that form the backbone of the sTiles
 * sparse matrix factorization library.
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

#ifndef _STILES_STRUCTS_H_
#define _STILES_STRUCTS_H_

#include <stdio.h>
#include <stdbool.h>
#include <vector>
#include <array>
#include <memory>
#include <string>

#include "stiles_params.hpp"   // named control-parameter indices (sTiles::param::*)
#include <atomic>

namespace sTiles {
#ifdef SMART_TILES
    class SmartTile;
#endif
}

#ifdef STILES_GPU
    #include <cuda_runtime.h>
    #include <cusolverDn.h>
    #include <cublas_v2.h>
#endif

// Use explicit relative path to the shared headers in tools/../include
#ifdef SMART_TILES
#endif
#include "stiles_types.hpp"
#include "../TileIndexer/TileIndexer.hpp"
#include "../TileIndexer/TileIndexerMapper.hpp"
#include "../tree/tree_structs.hpp"
#include "../memory/workspace.hpp"
#ifdef STILES_GPU
#include "../gpu/gpu_dispatch_plan.hpp"   // sTiles::gpu::DispatchPlan
#endif

namespace sTiles {
    using DenseTile = double*; // dense tile points to column-major storage
    struct TileMetaCore;           // metadata describing (row,col,width,height)
    struct SemisparseTileMetaCore;           // metadata describing (row,col,width,height)
    struct SparseTileMetaCore;           // metadata describing (row,col,width,height)
    struct SparseTileMetaData;           // metadata describing (row,col,width,height)
#ifdef SPARSE_STILES
    struct SparseTileCSC;                      // lightweight CSC sparse tile (colptr, rowind, nnz)
#endif
    struct SymbolicTileBitmaskCore;           // metadata describing (row,col,width,height)

    /// Pre-computed task schedule for a specific core count (rescale support)
    struct TaskSchedule {
        int num_cores{0};
        int bind_index{-1};
        std::shared_ptr<std::vector<std::array<int,7>>> chol_tasks;
        // 64-bit: on huge matrices the chol task count exceeds INT32_MAX, so the
        // per-core prefix offsets into chol_tasks must be 64-bit or they wrap and
        // the executor indexes chol_tasks out of bounds (SEGV on bern_spd).
        std::shared_ptr<std::vector<long long>>          chol_task_offsets;
        std::shared_ptr<std::vector<std::array<int,7>>> inv_tasks;
        std::shared_ptr<std::vector<long long>>          inv_task_offsets; // 64-bit: inv task count can exceed INT32_MAX
        std::shared_ptr<std::vector<std::array<int,6>>> solve_fwd_tasks;
        std::shared_ptr<std::vector<int>>                solve_fwd_offsets;
        std::shared_ptr<std::vector<std::array<int,6>>> solve_bwd_tasks;
        std::shared_ptr<std::vector<int>>                solve_bwd_offsets;
    };
}

/** ****************************************************************************
 * Debugging and configuration macros
 ******************************************************************************/
#define DEBUG 
#define E_STILES_SYMBOLIC_FACTORIZATION /**< Boolean-based symbolic factorization */
//#define E_STILES_BIT_SYMBOLIC_FACTORIZATION // E_STILES_SYMBOLIC_FACTORIZATION should not be defined and bit is used if this is defined
//#define TECH_DEBUG

/** ****************************************************************************
 * Structure representing a matrix tile
 ******************************************************************************/
/** ****************************************************************************
 * Structure representing a matrix tile (safemode storage)
 ******************************************************************************/
typedef struct DenseTileSafeMode {
    int row, col;       /**< Row and column indices of the tile */
    int width, height;  /**< Dimensions of the tile */
    double *elements;   /**< Pointer to the tile's data */
} DenseTileSafeMode;

/** ****************************************************************************
 * Structure representing a matrix tile for GPU
 ******************************************************************************/
typedef struct DenseGpuTile {
    int width, height;  /**< Dimensions of the tile */
    double *x;  // Pointer to a 2D array
} DenseGpuTile;

/** ****************************************************************************
 * Structure representing a configuration for specific solution types
 ******************************************************************************/
typedef struct SolveTrickConfig {
    int **trick_data;       /**< Pointer to solution tricks */
    int *trick_sizes;   /**< Sizes of solution tricks */
    int chunk_num;
    int ng;
} SolveTrickConfig;

/**
 * @brief Structure representing a matrix descriptor in the sTiles framework.
 *
 * This structure provides metadata and additional parameters for managing and
 * operating on tile-based matrices within the sTiles library. It includes 
 * information about matrix dimensions, tiling, submatrices, and optimization-specific
 * configurations.
 */
typedef struct stiles_desc_t {

    /** Matrix data pointers and offsets **/
    void *mat;          /**< Pointer to the beginning of the matrix data. */
    size_t A21;         /**< Offset to the A21 block in the matrix. */
    size_t A12;         /**< Offset to the A12 bloc#ifdef __cplusplus*/
    size_t A22;         /**< Offset to the A22 block in the matrix. */

    /** Matrix metadata **/
    int dtyp;   /**< Precision or data type of the matrix (e.g., float, double). */
    int mb;             /**< Number of rows in a single tile. */
    int nb;             /**< Number of columns in a single tile. */
    int bsiz;           /**< Total size of a tile, including padding (in elements). */
    int lm;             /**< Total number of rows in the full matrix. */
    int ln;             /**< Total number of columns in the full matrix. */

    /** Tiling and submatrix details **/
    int lm1;            /**< Number of tile rows in the A11 block (derived parameter). */
    int ln1;            /**< Number of tile columns in the A11 block (derived parameter). */
    int lmt;            /**< Total number of tile rows in the entire matrix. */
    int lnt;            /**< Total number of tile columns in the entire matrix. */
    int i;              /**< Row index of the starting point for a submatrix. */
    int j;              /**< Column index of the starting point for a submatrix. */
    int m;              /**< Number of rows in the submatrix. */
    int n;              /**< Number of columns in the submatrix. */
    int mt;             /**< Number of tile rows in the submatrix (derived parameter). */
    int nt;             /**< Number of tile columns in the submatrix (derived parameter). */

    /** Tiling and performance optimization details **/
    bool **on_off_tiles;       /**< Boolean matrix indicating active/inactive tiles. */
    double stiles_call;        /**< Timing or metadata related to tile calls. */
    
    DenseTileSafeMode **tiles;              
    DenseTileSafeMode *dense_tiles;              
    DenseTileSafeMode *rhs_tiles;
    DenseTileSafeMode *inverse_tiles;                

    int call_index;            /**< Index for keeping track of matrix call operations. */
    int *separators;           /**< Array of separators for specific tree structures. */
    double *stiles;            /**< Auxiliary array for specific tile computations. */
    int *magic_perm1;          /**< Array for storing optimized permutations. */
    bool *of_perm;             /**< Flags indicating specific permutations. */
    bool activated_nd;         /**< Flag indicating if nested dissection ordering is active. */

    /** Trick-based optimizations **/
    int **e_trick;                  /**< Matrix of tricks for optimization. */
    int *e_trick_size;              /**< Sizes corresponding to trick entries. */
    int **e_trick_partition1;       /**< Tricks for the first partition. */
    int *e_trick_size_partition1;   /**< Sizes for the first partition tricks. */
    int **e_trick_partition2;       /**< Tricks for the second partition. */
    int *e_trick_size_partition2;   /**< Sizes for the second partition tricks. */
    int **e_trick_inv;              /**< Inverse tricks for optimization. */
    int *e_trick_size_inv;          /**< Sizes corresponding to inverse tricks. */

    /** Tree structures **/
    TreeLeaf **trees;         /**< Array of trees used for factorization and optimization. */
    int tree_sep;             /**< Tree separator level. */
    int tree_stgy;            /**< Strategy indicator for tree-based optimizations. */
    bool boosted_e_trick;     /**< Flag indicating if boosted tricks are enabled. */

    /** Additional metadata **/
    int original_N;           /**< Original matrix size before transformations. */
    int *flops_mat;           /**< Array tracking floating-point operations (FLOPs) for each tile. */
    double *B;                /**< Pointer to auxiliary matrix/vector used in computations. */
    int sindex;               /**< Index indicating a specific solution phase or step. */
    int sversion;             /**< Version identifier for internal configurations. */

    /** Solution-specific structures **/
    SolveTrickConfig *solve_trick_type0; /**< Trick configuration for type 0 solutions. */
    SolveTrickConfig *solve_trick_type1; /**< Trick configuration for type 1 solutions. */
    int total_tiles;                     /**< Total number of tiles in the matrix. */

    DenseGpuTile *dense_tiles_gpu;                       /**< Matrix tiles */
    DenseGpuTile *inverse_tiles_gpu;                   /**< Inverse matrix tiles */
    DenseGpuTile *gpu_trees;                       /**< Matrix tiles */
    int GPU_ID;

#ifdef STILES_GPU
    cudaStream_t *streams;  /**< Pointer to dynamically allocated CUDA streams */
    cudaEvent_t *events;
#endif

#ifdef SMART_TILES
    sTiles::SmartTile **facTiles;              /**< Array of pointers to individual tiles. */
    sTiles::SmartTile **invTiles;              /**< Array of pointers to individual tiles. */
    sTiles::SmartTile **tmpTiles;              /**< Array of pointers to individual tiles. */
#endif

} TilesDescriptor;


/** ****************************************************************************
 * Structure defining the main sTiles computational scheme
 ******************************************************************************/
typedef struct TiledMatrix {
    int dim;                       /**< Number of rows/columns */
    int nnz;                     /**< Number of non-zeros */
    int original_order;              /**< Original matrix row/column count */
    int original_nnz;            /**< Original matrix non-zero count */
    int tile_size;               /**< Size of a tile */
    int remainderTileSize;          /**< Size of the last tile */
    int numActiveTiles;              /**< Size of all tiles (safemode semantics) */
    /**< Persistent byte-progress buffers for the chol executor.
     *   Allocated once (size numActiveTiles bytes / std::atomic slots) when
     *   numActiveTiles is finalized, freed at scheme destruction.
     *   ss_init_byte / dep_init_byte point stile->ss_slots / dep_tracker->slots
     *   at these and reset to the init value; no per-chol malloc/free.
     *   Unused (but kept allocated) under -UNSTILES_BYTE_PROGRESS.        */
    volatile unsigned char*       byte_progress_buf     = nullptr;
    std::atomic<unsigned char>*   byte_progress_buf_omp = nullptr;
    long long nnz_factor{0};        /**< Number of non-zeros in the Cholesky factor L */
    bool preprocess_failed{false};  /**< Set true if symbolic/ordering preprocessing failed
                                     *   for this scheme (e.g. auto-mode could not resolve a
                                     *   tile mode, or fill-in exceeded the 2B nnz guard).
                                     *   chol/solve/selinv/logdet refuse to run when set, so a
                                     *   failed preprocess aborts cleanly instead of cascading. */
    bool prefer_row_layout{false};  /**< Solve-time layout choice (set by wrapper_solve):
                                     *   true  -> multi-RHS kernels run row-major B
                                     *           (API transposes col-major B in/out)
                                     *   false -> multi-RHS kernels run col-major B
                                     *           (no API-boundary transpose)
                                     *   Decided by fill+nnz/n heuristic — very thin
                                     *   matrices (sem_* family: fill<2 AND nnz/row<6)
                                     *   regress under row-major; everything else wins. */
    int64_t* L_colptr{nullptr};     /**< Column pointers of L factor, size dim+1 (CSC).
                                     *   64-bit: the cumulative offset reaches nnz_factor,
                                     *   which exceeds INT_MAX for very large factors
                                     *   (e.g. bern: ~6.0e9 nonzeros). */
    int* L_rowind{nullptr};         /**< Row indices of L factor, size nnz_factor (CSC).
                                     *   Elements are row indices (< dim), so stay 32-bit. */
    double* L_values{nullptr};      /**< Packed CSC values of L factor, size nnz_factor.
                                     *   Allocated by sTiles_packing(group, obj) after chol.
                                     *   Buffer is reused across re-packs; freshness is
                                     *   tracked by `packed` (sTiles_chol invalidates it).
                                     *   Freed by sTiles_unpacking. */
    bool packed{false};             /**< True iff L_values mirrors the current factor.
                                     *   Set by sTiles_packing, cleared by sTiles_chol
                                     *   (stale buffer) and sTiles_unpacking. The solve
                                     *   csc fast-path gate reads this — never L_values
                                     *   alone — to avoid using stale data after a re-chol. */
    const double** L_src{nullptr};  /**< Precomputed source pointers, size nnz_factor.
                                     *   L_src[ptr] points to the tile slot for CSC entry
                                     *   ptr (or &pack_zero for structural zeros). Built
                                     *   once on first pack — same lifecycle as L_values.
                                     *   Hot pack loop is `L_values[ptr] = *L_src[ptr]`,
                                     *   collapsing all per-iteration index math + the
                                     *   tile-kind dispatch into a single indirection. */
    double pack_zero{0.0};           /**< Sentinel target for CSC entries with no live
                                     *   tile slot — L_src points here so the hot loop
                                     *   stays branch-free. */
    bool pack_prep_done{false};     /**< Gate for the once-per-scheme prepare step.
                                     *   True once _pack_prep_once has built L_src,
                                     *   allocated L_values, and cached has_dense /
                                     *   has_semi / Adesc_lm1. Cleared by anything that
                                     *   invalidates the structural prep: sTiles_unpacking
                                     *   (frees L_values/L_src), tile re-allocation,
                                     *   re-symbolic factor. Numeric re-chol does NOT
                                     *   clear it — values change, structure does not. */
    bool has_dense_cached{false};   /**< Cached `scheme->denseTiles != nullptr` snapshot
                                     *   taken at prep time. Read by pack_L_values to
                                     *   skip the dense-vs-semi dispatch re-derivation. */
    bool has_semi_cached{false};    /**< Cached semisparse-tiles-present flag (all four
                                     *   conditions: chunkedDenseTiles, semisparseTileMetaCore,
                                     *   tileMetaCore, diagonal_bmapper). Same lifecycle as
                                     *   pack_prep_done. */
    int  Adesc_lm1_cached{0};       /**< Cached `dim / tile_size` (= number of full tile rows)
                                     *   used by the pack hot loop for last-row ld math.
                                     *   Same lifecycle as pack_prep_done. */
    int dimTiledMatrix;    /**< Total number of used tiles */
    int fixed_column_size;               /**< Fixed column size */
    int triangular_size;                 /**< Original factor size */
    int factorization_variant;           /**< Factorization variant (0=sparse, 1=single, 2=scaled) */
    int tile_type_mode;                  /**< RESOLVED tile mode for THIS scheme (0=dense,
                                          *   1=semisparse, 2=non-uniform). Auto mode (3) is
                                          *   resolved per matrix during preprocessing;
                                          *   sTiles_preprocess_group snapshots the result here.
                                          *   -1 = not yet resolved (set at scheme allocation).
                                          *   Compute-phase code reads this via
                                          *   stiles_scheme_tile_mode(), NOT the global control
                                          *   slot [3], so groups with different resolutions can
                                          *   run concurrently. */
    int num_cores;                     /**< Number of cores used */
    int num_gpu_streams;               /**< Number of GPU streams (= num_cores when GPU active) */
    int internal_version;        /**< Version of internal processing */
    int use_ordering;                /**< Matrix ordering type */
    int nd_nnz;                  /**< Number of non-zeros for ND ordering */
    int nd_order;                    /**< Number of nodes for ND ordering */
    int nd_padding;                 /**< Extra diagonal entries injected by ND padding */
    int red_tree_separator_level;            /**< Separation level for red tree */
    int red_tree_cores{0};                   /**< Tree-reduction worker count (worldsize_b/tree_cores) used at
                                                  task generation; 0 => use STILES_SIZE. The reduce-wait loops in
                                                  pthreads_dpotrf_reduction_{semi,dense} must bound by this, not
                                                  STILES_SIZE, or ranks >= red_tree_cores (which never run Region B,
                                                  hence never signal their tree dependency) are waited on forever. */

    int *new_partition_sizes;               /**< New sizes array */
    int *partition_sizes;                  /**< Sizes array */
    int *tileIndexMapper;            /**< Mapping for compressed sparse columns */

    int *element_perm;               /**< Element permutation */
    int *element_iperm;              /**< Inverse element permutation */
    int *tree_counter;

    /* SCOTCH nested-dissection separator tree of the winning ordering, if a SCOTCH
     * variant was selected. cblknbr=0 indicates no tree available. The block indices
     * reference positions [0, dim - dense_count) in the final permuted space; the
     * trailing dense nodes (if any) are not part of any block. */
    int  scotch_cblknbr{0};          /**< Number of column blocks in SCOTCH ND tree */
    int* scotch_rangtab{nullptr};    /**< Block boundaries, size cblknbr+1            */
    int* scotch_treetab{nullptr};    /**< Parent block index, size cblknbr (-1=root)  */

    /** Generalized N-way ND partition layout consumed by ProcessCore's Path 2
     *  Cholesky scheduler. Each entry describes one region (leaf or separator)
     *  of the SCOTCH tree that the scheduler treats as an atomic partition:
     *    - [start_tile, end_tile) = tile-index range in the permuted matrix
     *    - [first_core, first_core + num_cores) = consecutive cores assigned
     *      to emit / execute tasks for this region
     *    - label = short human-readable tag ("L0", "S01", "S_root", etc.)
     *
     *  Layout convention: postorder. For a depth-D binary tree the entries are
     *    [L_0, L_1, S_01, L_2, L_3, S_23, S_0123, ..., S_root]
     *  with 2^D leaves and 2^D - 1 internal separators (total 2^(D+1) - 1
     *  entries). For the legacy 3-way case (depth 1) there are 3 entries:
     *    [P1, P2, Sep].
     *
     *  Empty vector (num_partitions == 0) means Path 2 is not configured and
     *  the scheduler falls back to standard Path 3. When backward compatibility
     *  is needed (old paths that set partition_sizes[3] only), the scheduler
     *  synthesizes a 3-entry partition list on the fly. */
    struct PartitionDesc {
        int start_tile{0};
        int end_tile{0};
        int first_core{0};
        int num_cores{0};
        const char* label{""};
    };
    int num_partitions{0};
    std::vector<PartitionDesc> partitions;

    /** Composed ND-partition + tree scheduling (semisparse, Option A).
     *  Set during SCOTCH block padding when the winning ordering is a SCOTCH
     *  variant (ND/AND/FND) and a valid partition layout was produced.
     *    - scotch_partition_collection: gate for the per-partition Region-A
     *      collection path in ProcessCore. True only for SCOTCH-family winners.
     *    - scotch_root_sep_tiles: tile count of the root separator (the last,
     *      "S_root" partition) in padded space. This is the structural `sep`
     *      candidate handed to corner_probe: the tree reduces exactly the root
     *      separator corner [num_tiles - scotch_root_sep_tiles, num_tiles) iff
     *      it passes the heavy/aggregate/<=max_sep gates; otherwise the root
     *      separator stays a normal ND partition (pure ND). */
    bool scotch_partition_collection{false};
    int  scotch_root_sep_tiles{0};

    int **call_lookup_table;           /**< Matrix for function calls */
    unsigned char *workspace_bit_array;    /**< Bit array for operations */

    bool use_boosted_e_trick;          /**< Flag for boosting tricks */
    bool compute_inverse;            /**< Flag for computing the inverse */
    bool use_gpu;
    bool use_banded_gpu;              /**< Use banded format for diagonal tiles on GPU */
    bool compute_inverse_on_gpu;            
    bool copy_on_preprocess;
    bool *permutation_flags;               /**< Permutation flags */
    bool **on_off_tiles;               /**< On/off status of tiles */

    DenseTileSafeMode *dense_tiles;                       
    DenseTileSafeMode *rhs_tiles;
    DenseTileSafeMode *saved_tiles;                 
    DenseTileSafeMode *inverse_tiles;  
         




    TreeLeaf **trees;                  /**< Array of tree structures */

    int **e_trick;                     /**< Trick data */
    int *e_trick_size;                 /**< Sizes of tricks */
    int **e_trick_inv;                 /**< Inverse tricks */
    bool *e_trick_copy_ind;                 /**< Inverse tricks */
    int *e_trick_size_inv;             /**< Sizes of inverse tricks */
    int *t_indicies;                   /**< Tile indices */
    int *e_indicies;                   /**< Element indices */

    SolveTrickConfig *solve_trick_type0;     /**< Trick type 0 */
    SolveTrickConfig *solve_trick_type1;     /**< Trick type 1 */

    DenseGpuTile *dense_tiles_gpu;                       /**< Matrix tiles */
    DenseGpuTile *inverse_tiles_gpu;                   /**< Inverse matrix tiles */
    DenseGpuTile *gpu_trees;                       /**< Matrix tiles */
    int GPU_ID;

    double* timings;

    /** Cholesky factorization status.
     *  0  = success (positive definite).
     *  >0 = the leading minor of order chol_info is not positive definite.
     *  Written by Process::pthreads_pdpotrf; read by callers after factorization. */
    std::atomic<int> chol_info{0};

#ifdef STILES_GPU
    cudaStream_t *streams;  /**< Pointer to dynamically allocated CUDA streams */
    cudaEvent_t *events;
    void* gpu_persistent_ctx;  /**< Persistent GPU context (GpuPersistentContext*) - shared across chol/inv/solve */
    double* d_tmp_tile;  /**< GPU workspace for DSYRK/DGEMM scatter operations (semisparse) */
    /**< Per-call multi-GPU dispatch plan stored on the PRIMARY scheme (call 0)
     *   of the group; the allocator and per-call init paths read this to bind
     *   tiles to the device the planner picked. Empty/default on non-primary
     *   schemes and when GPU mode is off. */
    sTiles::gpu::DispatchPlan gpu_dispatch_plan;
#endif

    /**< Opaque handle for the non-uniform tile path (sTiles::sparse). Cells
     *   have variable rows×cols driven by the elimination tree (column width
     *   = supernode width, row count = row-supernode run length); each cell
     *   is dense, no within-cell sparsity bitmap.
     *   Set when factorization_variant==0 && tile_type_mode==2; nullptr otherwise.
     *   Owned by the scheme; freed via sTiles::sparse::api::freeGroup in destroy_tiled_matrix. */
    void* sparse_handle = nullptr;

    //dense tiles
    sTiles::DenseTile *denseTiles;                       
    sTiles::DenseTile *rhTiles;
    sTiles::DenseTile *savedTiles;                 
    sTiles::DenseTile *inverseTiles;  
    sTiles::TileMetaCore *tileMetaCore; 
    int *element_offset_lookup;

    //sparse tiles
#ifdef SMART_TILES
    sTiles::SmartTile *facTiles;
    sTiles::SmartTile *invTiles;
    sTiles::SmartTile *tmpTiles;
    sTiles::SmartTile *rhsTiles;
#endif
    sTiles::SparseTileMetaCore *sparseTileMetaCore;
    sTiles::SparseTileMetaCore *invSparseTileMetaCore;
    sTiles::SparseTileMetaData *sparseTileMetaData;
#ifdef SPARSE_STILES
    sTiles::SparseTileCSC *sparseTileCSC;
#endif
    int* nnz_tile_counter;

    //chunked tiles
    sTiles::DenseTile *chunkedDenseTiles;
    sTiles::DenseTile *chunkedRhsTiles;
    sTiles::DenseTile *chunkedSavedTiles;
    sTiles::DenseTile *chunkedInverseTiles;
    sTiles::SemisparseTileMetaCore *semisparseTileMetaCore;                       


    sTiles::SymbolicTileBitmaskCore *symbolicTileBitmaskCore;

    sTiles::TileStorage storage;
    int *tileIndexMapper2;            /**< Mapping for compressed sparse columns */


    TileIndexer::Method neighbor_lookup_method;
    TileIndexer::State tile_indexer_state;
    TileIndexer::GraphBuffers tile_indexer_graph;
    TileIndexer::State state;
    TileIndexer::Mapper mapper;
    int  *diagonal_mapper;
    bool *diagonal_bmapper;
    std::shared_ptr<std::vector<std::array<int,7>>> chol_tasks;
    std::shared_ptr<std::vector<long long>> chol_task_offsets; // 64-bit: task count can exceed INT32_MAX
    std::shared_ptr<std::vector<std::array<int,7>>> inv_tasks;
    std::shared_ptr<std::vector<long long>> inv_task_offsets; // 64-bit: inv task count can exceed INT32_MAX

    // Precomputed gather offsets for semisparse inv tasks (cases 7/8).
    // inv_gather_packed: flat buffer of int32_t data (offsets + valid indices)
    // inv_gather_index:  3 ints per inv task [data_offset, n_valid, flags]
    //   flags: 0=skip, 1=all_valid, 2=partial, 0xFF=not case7/8
    //   For flags=1: packed data = [col_offset_0, ..., col_offset_{n_valid-1}]
    //   For flags=2: packed data = [valid_idx_0, col_offset_0, valid_idx_1, col_offset_1, ...]
    std::shared_ptr<std::vector<int32_t>> inv_gather_packed;
    std::shared_ptr<std::vector<int32_t>> inv_gather_index;

    // Precomputed scatter info for semisparse chol tasks (case 4 DGEMM).
    // chol_scatter_index: 2 int64_ts per chol task [data_offset, path_flag]
    //   data_offset is int64_t so packed.size() can exceed INT32_MAX on huge matrices.
    //   path_flag: 0=direct_gemm, 1=fused_contig, 2=fused_scatter, 3=blas_contig, 4=blas_scatter, 0xFF=not case4
    //   For path 3/4: packed data = [slot_0, slot_1, ..., slot_{cols_b-1}] (precomputed acol_map[aind_b[j]])
    std::shared_ptr<std::vector<int32_t>> chol_scatter_packed;
    std::shared_ptr<std::vector<int64_t>> chol_scatter_index;

    // Solve tasks: pre-collected (k,m) tile pairs with metadata
    // Format: [type, k, m, tile_idx, lda, tempkm, tempmm]
    //   type=1: diagonal TRSM (m==k)
    //   type=2: off-diagonal GEMM (m!=k)
    std::shared_ptr<std::vector<std::array<int,6>>> solve_fwd_tasks;
    std::shared_ptr<std::vector<int>> solve_fwd_offsets;
    std::shared_ptr<std::vector<std::array<int,6>>> solve_bwd_tasks;
    std::shared_ptr<std::vector<int>> solve_bwd_offsets;
    // Update-counter sync (pdtrsm tasked variants).
    // solve_fwd_expected[m] = number of off-diagonal active tasks (k, m) for k<m,
    // i.e. how many ss[m]++ increments the diagonal at (m, m) must wait for.
    // Built once during symbolic; read at trsm-call time. Enables sparse iteration
    // (skip empty (k, m) pairs entirely) instead of the dense O(num_tiles^2) walk.
    std::shared_ptr<std::vector<int>> solve_fwd_expected;
    std::shared_ptr<std::vector<int>> solve_bwd_expected;

    // GPU solve tasks: per-stream bins with offsets for multi-stream execution
    // Same format as CPU solve tasks, partitioned across GPU streams
    std::shared_ptr<std::vector<std::array<int,6>>> gpu_solve_fwd_tasks;
    std::shared_ptr<std::vector<std::array<int,6>>> gpu_solve_bwd_tasks;
    // GPU multi-stream solve offsets: gpu_solve_*_offsets[stream_id] to [stream_id+1]
    std::shared_ptr<std::vector<int>> gpu_solve_fwd_offsets;
    std::shared_ptr<std::vector<int>> gpu_solve_bwd_offsets;

    int *tile_index_lookup;
    int *withinTileRow;
    int *withinTileCol;

    sTiles::Workspace** workspaces;
    int num_workspaces;  // Number of allocated workspace slots

    // Element-level symbolic fill-in (COO, unpermuted space, strictly lower-triangular when no ordering)
    std::shared_ptr<std::vector<int>> filled_rows;
    std::shared_ptr<std::vector<int>> filled_cols;
    int filled_nnz{0};

    // Pre-computed task schedule for a different (rescaled) core count
    sTiles::TaskSchedule rescale_schedule;
    std::atomic<int> use_rescale{0};

    std::string selected_ordering;
    std::string selected_tile_ordering;

} TiledMatrix;

/**
 * Resolved tile mode for a scheme (0=dense, 1=semisparse, 2=non-uniform).
 *
 * Preferred accessor for COMPUTE-phase code (chol/solve/selinv/logdet): reads
 * the per-scheme snapshot taken at the end of preprocessing, falling back to
 * the global control slot [3] only when the snapshot is absent (mid-preprocess
 * callers, or schemes produced by paths that predate the snapshot). The
 * fallback keeps behavior identical to the historical global read.
 */
extern int stiles_control_params[];
inline int stiles_scheme_tile_mode(const TiledMatrix* scheme) {
    return (scheme && scheme->tile_type_mode >= 0) ? scheme->tile_type_mode
                                                   : stiles_control_params[sTiles::param::TileTypeMode];
}

namespace sTiles {

inline std::vector<std::array<int,7>>& ensure_chol_tasks(TiledMatrix* tm) {
    if (!tm->chol_tasks) tm->chol_tasks = std::make_shared<std::vector<std::array<int,7>>>();
    return *tm->chol_tasks;
}

inline std::vector<long long>& ensure_chol_task_offsets(TiledMatrix* tm) {
    if (!tm->chol_task_offsets) tm->chol_task_offsets = std::make_shared<std::vector<long long>>();
    return *tm->chol_task_offsets;
}

inline std::vector<std::array<int,7>>& ensure_inv_tasks(TiledMatrix* tm) {
    if (!tm->inv_tasks) tm->inv_tasks = std::make_shared<std::vector<std::array<int,7>>>();
    return *tm->inv_tasks;
}

inline std::vector<long long>& ensure_inv_task_offsets(TiledMatrix* tm) {
    if (!tm->inv_task_offsets) tm->inv_task_offsets = std::make_shared<std::vector<long long>>();
    return *tm->inv_task_offsets;
}

inline const std::vector<std::array<int,7>>& get_chol_tasks(const TiledMatrix* tm) {
    static const std::vector<std::array<int,7>> empty;
    if (!tm) return empty;
    if (tm->use_rescale.load(std::memory_order_acquire) > 0 && tm->rescale_schedule.chol_tasks)
        return *tm->rescale_schedule.chol_tasks;
    return tm->chol_tasks ? *tm->chol_tasks : empty;
}

inline const std::vector<long long>& get_chol_task_offsets(const TiledMatrix* tm) {
    static const std::vector<long long> empty;
    if (!tm) return empty;
    if (tm->use_rescale.load(std::memory_order_acquire) > 0 && tm->rescale_schedule.chol_task_offsets)
        return *tm->rescale_schedule.chol_task_offsets;
    return tm->chol_task_offsets ? *tm->chol_task_offsets : empty;
}

inline const std::vector<std::array<int,7>>& get_inv_tasks(const TiledMatrix* tm) {
    static const std::vector<std::array<int,7>> empty;
    if (!tm) return empty;
    if (tm->use_rescale.load(std::memory_order_acquire) > 0 && tm->rescale_schedule.inv_tasks)
        return *tm->rescale_schedule.inv_tasks;
    return tm->inv_tasks ? *tm->inv_tasks : empty;
}

inline const std::vector<long long>& get_inv_task_offsets(const TiledMatrix* tm) {
    static const std::vector<long long> empty;
    if (!tm) return empty;
    if (tm->use_rescale.load(std::memory_order_acquire) > 0 && tm->rescale_schedule.inv_task_offsets)
        return *tm->rescale_schedule.inv_task_offsets;
    return tm->inv_task_offsets ? *tm->inv_task_offsets : empty;
}

// Solve task helpers
inline std::vector<std::array<int,6>>& ensure_solve_fwd_tasks(TiledMatrix* tm) {
    if (!tm->solve_fwd_tasks) tm->solve_fwd_tasks = std::make_shared<std::vector<std::array<int,6>>>();
    return *tm->solve_fwd_tasks;
}

inline std::vector<int>& ensure_solve_fwd_offsets(TiledMatrix* tm) {
    if (!tm->solve_fwd_offsets) tm->solve_fwd_offsets = std::make_shared<std::vector<int>>();
    return *tm->solve_fwd_offsets;
}

inline std::vector<std::array<int,6>>& ensure_solve_bwd_tasks(TiledMatrix* tm) {
    if (!tm->solve_bwd_tasks) tm->solve_bwd_tasks = std::make_shared<std::vector<std::array<int,6>>>();
    return *tm->solve_bwd_tasks;
}

inline std::vector<int>& ensure_solve_bwd_offsets(TiledMatrix* tm) {
    if (!tm->solve_bwd_offsets) tm->solve_bwd_offsets = std::make_shared<std::vector<int>>();
    return *tm->solve_bwd_offsets;
}

inline std::vector<int>& ensure_solve_fwd_expected(TiledMatrix* tm) {
    if (!tm->solve_fwd_expected) tm->solve_fwd_expected = std::make_shared<std::vector<int>>();
    return *tm->solve_fwd_expected;
}

inline std::vector<int>& ensure_solve_bwd_expected(TiledMatrix* tm) {
    if (!tm->solve_bwd_expected) tm->solve_bwd_expected = std::make_shared<std::vector<int>>();
    return *tm->solve_bwd_expected;
}

inline const std::vector<int>& get_solve_fwd_expected(const TiledMatrix* tm) {
    static const std::vector<int> empty;
    if (!tm) return empty;
    return tm->solve_fwd_expected ? *tm->solve_fwd_expected : empty;
}

inline const std::vector<int>& get_solve_bwd_expected(const TiledMatrix* tm) {
    static const std::vector<int> empty;
    if (!tm) return empty;
    return tm->solve_bwd_expected ? *tm->solve_bwd_expected : empty;
}

inline const std::vector<std::array<int,6>>& get_solve_fwd_tasks(const TiledMatrix* tm) {
    static const std::vector<std::array<int,6>> empty;
    if (!tm) return empty;
    if (tm->use_rescale.load(std::memory_order_acquire) > 0 && tm->rescale_schedule.solve_fwd_tasks)
        return *tm->rescale_schedule.solve_fwd_tasks;
    return tm->solve_fwd_tasks ? *tm->solve_fwd_tasks : empty;
}

inline const std::vector<int>& get_solve_fwd_offsets(const TiledMatrix* tm) {
    static const std::vector<int> empty;
    if (!tm) return empty;
    if (tm->use_rescale.load(std::memory_order_acquire) > 0 && tm->rescale_schedule.solve_fwd_offsets)
        return *tm->rescale_schedule.solve_fwd_offsets;
    return tm->solve_fwd_offsets ? *tm->solve_fwd_offsets : empty;
}

inline const std::vector<std::array<int,6>>& get_solve_bwd_tasks(const TiledMatrix* tm) {
    static const std::vector<std::array<int,6>> empty;
    if (!tm) return empty;
    if (tm->use_rescale.load(std::memory_order_acquire) > 0 && tm->rescale_schedule.solve_bwd_tasks)
        return *tm->rescale_schedule.solve_bwd_tasks;
    return tm->solve_bwd_tasks ? *tm->solve_bwd_tasks : empty;
}

inline const std::vector<int>& get_solve_bwd_offsets(const TiledMatrix* tm) {
    static const std::vector<int> empty;
    if (!tm) return empty;
    if (tm->use_rescale.load(std::memory_order_acquire) > 0 && tm->rescale_schedule.solve_bwd_offsets)
        return *tm->rescale_schedule.solve_bwd_offsets;
    return tm->solve_bwd_offsets ? *tm->solve_bwd_offsets : empty;
}

// GPU solve task helpers (sequential order, actual coordinates)
inline std::vector<std::array<int,6>>& ensure_gpu_solve_fwd_tasks(TiledMatrix* tm) {
    if (!tm->gpu_solve_fwd_tasks) tm->gpu_solve_fwd_tasks = std::make_shared<std::vector<std::array<int,6>>>();
    return *tm->gpu_solve_fwd_tasks;
}

inline std::vector<std::array<int,6>>& ensure_gpu_solve_bwd_tasks(TiledMatrix* tm) {
    if (!tm->gpu_solve_bwd_tasks) tm->gpu_solve_bwd_tasks = std::make_shared<std::vector<std::array<int,6>>>();
    return *tm->gpu_solve_bwd_tasks;
}

inline const std::vector<std::array<int,6>>& get_gpu_solve_fwd_tasks(const TiledMatrix* tm) {
    static const std::vector<std::array<int,6>> empty;
    return (tm && tm->gpu_solve_fwd_tasks) ? *tm->gpu_solve_fwd_tasks : empty;
}

inline const std::vector<std::array<int,6>>& get_gpu_solve_bwd_tasks(const TiledMatrix* tm) {
    static const std::vector<std::array<int,6>> empty;
    return (tm && tm->gpu_solve_bwd_tasks) ? *tm->gpu_solve_bwd_tasks : empty;
}

// GPU multi-stream solve offset helpers
inline std::vector<int>& ensure_gpu_solve_fwd_offsets(TiledMatrix* tm) {
    if (!tm->gpu_solve_fwd_offsets) tm->gpu_solve_fwd_offsets = std::make_shared<std::vector<int>>();
    return *tm->gpu_solve_fwd_offsets;
}

inline std::vector<int>& ensure_gpu_solve_bwd_offsets(TiledMatrix* tm) {
    if (!tm->gpu_solve_bwd_offsets) tm->gpu_solve_bwd_offsets = std::make_shared<std::vector<int>>();
    return *tm->gpu_solve_bwd_offsets;
}

inline const std::vector<int>& get_gpu_solve_fwd_offsets(const TiledMatrix* tm) {
    static const std::vector<int> empty;
    return (tm && tm->gpu_solve_fwd_offsets) ? *tm->gpu_solve_fwd_offsets : empty;
}

inline const std::vector<int>& get_gpu_solve_bwd_offsets(const TiledMatrix* tm) {
    static const std::vector<int> empty;
    return (tm && tm->gpu_solve_bwd_offsets) ? *tm->gpu_solve_bwd_offsets : empty;
}

} // namespace sTiles

typedef struct sTiles_call {

    int global_index;
    int call_index;   // Index of the call within the group
    int mapped_call_index;
    int mapped_group_index;
    int num_cores;        // Number of STILES_CORES allocated for this call
    int *core_bind_ids;  // Array to store the IDs of the num_cores bound to this call
    int local_offset;
    bool save_factor; 
    bool compute_inverse; 
    bool compute_log_determinant; 
    int ordering_strategy; 
    int sequence_id; 
    int order;
    int nnz;
    int* row_indices;  // <-- updated from row_indicies
    int* col_indices;  // <-- updated from col_indicies
    double* matrix_values;
    int red_tree_separator_level; // Array for fixed columns per group 
    int tile_size; // Array for fixed columns per group 
    int factorization_variant; // Array for fixed columns per group 
    int arrowhead_thickness; // Array for fixed columns per group
    int preprocess_level; // Array for fixed columns per group
    int* parameters; // Array for fixed columns per group
    bool use_nested_dissection;
    int bandwidth;
    int num_right_hand_sides;

    //sTiles_context* context

} sTiles_call;

typedef struct sTiles_group {

    int group_index;      // Index of the group
    int group_offset;      // Index of the group
    int num_calls;        // Number of stiles_calls in this group
    int arrowhead_size_per_group; // Array for fixed columns per group
    sTiles_call *stiles_calls;     // Array of sTiles_call structures for stiles_calls in this group
    bool same_group; 

} sTiles_group;

struct sTiles_Global_Pool {
    pthread_t* worker_threads = nullptr;
};


typedef struct sTiles_object {

    //global parameters:
    unsigned int magic;
    int num_call_groups;          // Number of stiles_groups
    int *num_calls_per_group;     // Array for number of stiles_calls per group
    int *num_cores_per_group;     // Array for number of num_cores per group
    int *factorization_type_per_group;      // dense or sparse chol
    int max_cores_sys;       // Maximum number of num_cores
    int num_total_indices;         // Total number of Cholesky indices (sum of all stiles_calls)
    int* global_indicies;
    int** call_matrix;
    int total_calls;
    bool numa_enabled;

    sTiles_group *stiles_groups;           // Array of sTiles_group to store information for each group and its stiles_calls
    TiledMatrix** schemes;
    sTiles_Global_Pool global_pool;
    int total_worker_threads;

} sTiles_object;




#endif /* _STILES_STRUCTS_H_ */
