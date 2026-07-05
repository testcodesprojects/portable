    static inline void build_dense_tile_lookup_init_parallel(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {
        const int original_nnz = scheme->original_nnz;
        const int num_tiles = scheme->dimTiledMatrix;
        const int num_tiles_active = scheme->numActiveTiles;

        const int threads = (num_cores > 0) ? num_cores : omp_get_max_threads();
        const bool nested_enabled = (omp_get_max_active_levels() > 1);
        const bool can_spawn = (!omp_in_parallel()) || nested_enabled;

        if (!can_spawn || threads < 2) {
            // Fallback must use the DENSE initializer, not semisparse
            build_dense_tile_lookup_init_serial(call_info, scheme, group_index);
            return;
        }

        scheme->tile_index_lookup      = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->element_offset_lookup  = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->diagonal_bmapper       = TileMemoryManager::allocateZero<bool>(num_tiles_active, group_index);

        std::atomic<int> error_flag{0};

        if (scheme->use_ordering == 0) {
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (int index = 0; index < original_nnz; ++index) {
                const int col = (*call_info)->row_indices[index];
                const int row = (*call_info)->col_indices[index];

                const int tileRow = row / (*call_info)->tile_size;
                const int tileCol = col / (*call_info)->tile_size;

                if (tileRow > tileCol) {
                    error_flag.store(1, std::memory_order_relaxed);
                    continue;
                }

                const int indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
                if (tileRow == tileCol) {
                    #pragma omp atomic write
                    scheme->diagonal_bmapper[indexed_tile] = true;
                }

                const int withinTileRow = row % (*call_info)->tile_size;
                const int withinTileCol = col % (*call_info)->tile_size;
                const int tile_ld = (tileRow == num_tiles - 1)
                                  ? scheme->remainderTileSize
                                  : scheme->tile_size;
                scheme->tile_index_lookup[index]     = indexed_tile;
                scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);
            }
        } else if (scheme->use_ordering == 1) {
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (int index = 0; index < original_nnz; ++index) {
                int row, col;
                if (scheme->element_perm[(*call_info)->col_indices[index]] < scheme->element_perm[(*call_info)->row_indices[index]]) {
                    col = scheme->element_perm[(*call_info)->row_indices[index]];
                    row = scheme->element_perm[(*call_info)->col_indices[index]];
                } else {
                    row = scheme->element_perm[(*call_info)->row_indices[index]];
                    col = scheme->element_perm[(*call_info)->col_indices[index]];
                }

                const int tileRow = row / scheme->tile_size;
                const int tileCol = col / scheme->tile_size;

                if (tileRow <= tileCol) {
                    const int indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
                    const int withinTileRow = row % (*call_info)->tile_size;
                    const int withinTileCol = col % (*call_info)->tile_size;

                    if (tileRow == tileCol) {
                        #pragma omp atomic write
                        scheme->diagonal_bmapper[indexed_tile] = true;

                        const int tile_ld = (tileRow == num_tiles - 1)
                                          ? scheme->remainderTileSize
                                          : scheme->tile_size;
                        scheme->tile_index_lookup[index] = indexed_tile;
                        if (withinTileRow <= withinTileCol) {
                            scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);
                        } else {
                            scheme->element_offset_lookup[index] = withinTileCol + (withinTileRow * tile_ld);
                        }
                    } else {
                        const int tile_ld = (tileRow == num_tiles - 1)
                                          ? scheme->remainderTileSize
                                          : scheme->tile_size;
                        scheme->tile_index_lookup[index]     = indexed_tile;
                        scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);
                    }
                } else {
                    const int indexed_tile   = scheme->mapper.map_ij(tileCol, tileRow, num_tiles);
                    const int withinTileRow  = col % (*call_info)->tile_size;
                    const int withinTileCol  = row % (*call_info)->tile_size;
                    const int dest_row_tile  = tileCol;  // swapped: destination row tile is tileCol
                    const int tile_ld        = (dest_row_tile == num_tiles - 1)
                                             ? scheme->remainderTileSize
                                             : scheme->tile_size;

                    scheme->tile_index_lookup[index]     = indexed_tile;
                    scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);
                }
            }
        }

        if (error_flag.load(std::memory_order_relaxed)) {
            sTiles::Logger::error("build_dense_tile_lookup_init_parallel: unexpected lower-triangular entry encountered (parallel path).");
            std::exit(EXIT_FAILURE);
        }
    }

    static inline void build_dense_tile_lookup_phase1(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {
        const int n = scheme->original_order;
        scheme->remainderTileSize = (n % scheme->tile_size == 0) ? scheme->tile_size : (n % scheme->tile_size);

        // Check if lookup tables were already allocated by build_all_tile_lookup (mode==2)
        // If so, skip re-allocation to avoid redundancy and conflicts
        if (scheme->tile_index_lookup && scheme->element_offset_lookup && scheme->diagonal_bmapper) {
            return;
        }

        if (num_cores < 2 || scheme->original_nnz < 5000) { build_dense_tile_lookup_init_serial(call_info, scheme, group_index); return; }
    #ifndef _OPENMP
        build_dense_tile_lookup_init_serial(call_info, scheme, group_index); return;
    #else
        const bool nested_enabled = (omp_get_max_active_levels() > 1);
        const bool can_spawn = (!omp_in_parallel()) || nested_enabled;
        if (!can_spawn) { build_dense_tile_lookup_init_serial(call_info, scheme, group_index); return; }
        build_dense_tile_lookup_init_parallel(call_info, scheme, group_index, num_cores);
    #endif
    }

    inline StatusCode build_dense_tile_lookup(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {
        // Check tile type mode
        int* params = sTiles_get_params();
        const int tile_type_mode = params[3];

        // Only proceed if tile_type_mode is 0 (dense only) or 3 (dense + semisparse)
        if (tile_type_mode != 0 && tile_type_mode != 3) {
            // Don't proceed with dense tile lookup
            return StatusCode::Success;
        }

        if (!call_info || !*call_info || !scheme) return StatusCode::InvalidArgument;
        const int fact_variant = (*call_info)->factorization_variant;
        // For variants 1 (single tile) and 2 (scaled dense), use different lookup functions
        if (fact_variant == 1 || fact_variant == 2) {
            // Skip this function - caller should use variant-specific functions
            return StatusCode::Success;
        }
        if (fact_variant > 3) {
            std::fprintf(stderr, "ERROR: Unsupported factorization_variant %d\n", fact_variant);
            exit(0);
            return StatusCode::InvalidArgument;
        }

        scheme->tileMetaCore = nullptr;

        const int num_active = scheme->numActiveTiles;
        if (num_active < 0) return StatusCode::InvalidArgument;

        scheme->tileMetaCore = TileMemoryManager::allocate<TileMetaCore>(num_active, group_index);
        if (!scheme->tileMetaCore) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for denseTiles or tileMetaCore.\n");
            return StatusCode::OutOfResources;
        }

        set_tile_extents(&scheme);
        build_dense_tile_lookup_phase1(call_info, scheme, group_index, num_cores);

        return StatusCode::Success;
    }

    inline StatusCode build_dense_tile_lookup_variant1(sTiles_call **call_info, TiledMatrix *scheme, int group_index) {
        if (!call_info || !*call_info || !scheme) return StatusCode::InvalidArgument;

        const int original_nnz = scheme->original_nnz;
        const int matrix_size = scheme->dim;
        const int num_active = scheme->numActiveTiles;

        if (num_active != 1) {
            std::fprintf(stderr, "ERROR: build_dense_tile_lookup_variant1 expects numActiveTiles=1, got %d\n", num_active);
            return StatusCode::InvalidArgument;
        }

        // Allocate lookup tables
        scheme->tile_index_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->element_offset_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->diagonal_bmapper = TileMemoryManager::allocateZero<bool>(num_active, group_index);

        if (!scheme->tile_index_lookup || !scheme->element_offset_lookup || !scheme->diagonal_bmapper) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for lookup tables in variant1\n");
            return StatusCode::OutOfResources;
        }

        // The single tile is diagonal
        scheme->diagonal_bmapper[0] = true;

        // Allocate and set tileMetaCore
        scheme->tileMetaCore = TileMemoryManager::allocate<TileMetaCore>(num_active, group_index);
        if (!scheme->tileMetaCore) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for tileMetaCore in variant1\n");
            return StatusCode::OutOfResources;
        }

        // Set tile metadata (full matrix as single tile at position 0,0)
        scheme->tileMetaCore[0].index = 0;
        scheme->tileMetaCore[0].row = 0;
        scheme->tileMetaCore[0].col = 0;
        scheme->tileMetaCore[0].height = matrix_size;
        scheme->tileMetaCore[0].width = matrix_size;

        // Build lookup: all elements map to tile 0
        // Input format: row_indices stores column, col_indices stores row (CSC format)
        for (int index = 0; index < original_nnz; ++index) {
            int col = (*call_info)->row_indices[index];
            int row = (*call_info)->col_indices[index];

            // Ensure upper triangular (row <= col)
            if (row > col) {
                std::fprintf(stderr, "ERROR: variant1 encountered lower-triangular entry (row=%d, col=%d)\n", row, col);
                return StatusCode::InvalidArgument;
            }

            // All elements go to tile 0
            scheme->tile_index_lookup[index] = 0;

            // Column-major offset in the full matrix
            scheme->element_offset_lookup[index] = row + (col * matrix_size);
        }

        return StatusCode::Success;
    }

    inline StatusCode build_dense_tile_lookup_variant2(sTiles_call **call_info, TiledMatrix *scheme, int group_index) {
        if (!call_info || !*call_info || !scheme) return StatusCode::InvalidArgument;

        const int original_nnz = scheme->original_nnz;
        const int num_tiles = scheme->dimTiledMatrix;
        const int num_active = scheme->numActiveTiles;
        const int tile_size = scheme->tile_size;
        const int matrix_size = scheme->dim;

        const int expected_active = (num_tiles * (num_tiles + 1)) / 2;
        if (num_active != expected_active) {
            std::fprintf(stderr, "ERROR: build_dense_tile_lookup_variant2 expects numActiveTiles=%d, got %d\n",
                         expected_active, num_active);
            return StatusCode::InvalidArgument;
        }

        // Allocate lookup tables
        scheme->tile_index_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->element_offset_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->diagonal_bmapper = TileMemoryManager::allocateZero<bool>(num_active, group_index);

        if (!scheme->tile_index_lookup || !scheme->element_offset_lookup || !scheme->diagonal_bmapper) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for lookup tables in variant2\n");
            return StatusCode::OutOfResources;
        }

        // Allocate and set tileMetaCore
        scheme->tileMetaCore = TileMemoryManager::allocate<TileMetaCore>(num_active, group_index);
        if (!scheme->tileMetaCore) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for tileMetaCore in variant2\n");
            return StatusCode::OutOfResources;
        }

        // Set tile dimensions for all triangular tiles
        const int remainder = (matrix_size % tile_size == 0) ? tile_size : (matrix_size % tile_size);
        for (int j = 0; j < num_tiles; ++j) {
            const int width = (j == num_tiles - 1) ? remainder : tile_size;
            for (int i = 0; i <= j; ++i) {
                const int height = (i == num_tiles - 1) ? remainder : tile_size;
                // Upper triangular index for tile (i, j)
                const int dense_idx = i * num_tiles - (i * (i - 1)) / 2 + (j - i);

                if (dense_idx >= 0 && dense_idx < num_active) {
                    scheme->tileMetaCore[dense_idx].index = dense_idx;
                    scheme->tileMetaCore[dense_idx].row = i;
                    scheme->tileMetaCore[dense_idx].col = j;
                    scheme->tileMetaCore[dense_idx].height = height;
                    scheme->tileMetaCore[dense_idx].width = width;

                    // Mark diagonal tiles
                    if (i == j) {
                        scheme->diagonal_bmapper[dense_idx] = true;
                    }
                }
            }
        }

        // Build lookup: map each element to its tile
        // Input format: row_indices stores column, col_indices stores row (CSC format)
        for (int index = 0; index < original_nnz; ++index) {
            int col = (*call_info)->row_indices[index];
            int row = (*call_info)->col_indices[index];

            // Compute tile indices
            int tileRow = row / tile_size;
            int tileCol = col / tile_size;

            // Ensure upper triangular (tileRow <= tileCol)
            if (tileRow > tileCol) {
                std::fprintf(stderr, "ERROR: variant2 encountered lower-triangular entry (row=%d, col=%d, tileRow=%d, tileCol=%d)\n",
                             row, col, tileRow, tileCol);
                return StatusCode::InvalidArgument;
            }

            // Upper triangular index for tile (tileRow, tileCol)
            const int dense_idx = tileRow * num_tiles - (tileRow * (tileRow - 1)) / 2 + (tileCol - tileRow);

            if (dense_idx < 0 || dense_idx >= num_active) {
                std::fprintf(stderr, "ERROR: variant2 computed invalid tile index %d (numActive=%d)\n",
                             dense_idx, num_active);
                return StatusCode::InvalidArgument;
            }

            scheme->tile_index_lookup[index] = dense_idx;

            // Compute within-tile position
            int withinTileRow = row % tile_size;
            int withinTileCol = col % tile_size;

            // Leading dimension based on the tile's row index
            const int tile_ld = (tileRow == num_tiles - 1) ? remainder : tile_size;

            // Column-major offset within the tile
            scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);
        }

        return StatusCode::Success;
    }

    static StatusCode allocate_buffers_for_tile(TiledMatrix* scheme, int idx, int group_index) {
        // For variants 1 and 2, tileMetaCore may not be allocated
        // Use matrix dim directly in that case
        int h, w;
        if (scheme->tileMetaCore) {
            const TileMetaCore& m = scheme->tileMetaCore[idx];
            h = (m.height > 0) ? m.height : scheme->tile_size;
            w = (m.width  > 0) ? m.width  : scheme->tile_size;
        } else {
            // Variant 1: single tile covering entire matrix
            h = scheme->dim;
            w = scheme->dim;
        }

        // Get pruning mode from control parameters
        int* params = sTiles_get_params();
        const int pruning_mode = params[0];
        const int tile_type_mode = params[3];

        // Check if we should skip allocation based on semisparse metadata
        // BUT: never skip diagonal tiles (they always have all columns active)
        // Also skip this check if pruning_mode is 0 (no pruning)
        // ALSO: never skip for tile_type_mode == 2 (compare mode) - need all dense tiles
        if (pruning_mode > 0 && tile_type_mode != 3) {

            const bool is_diagonal = scheme->diagonal_bmapper && scheme->diagonal_bmapper[idx];
            if (!is_diagonal && scheme->semisparseTileMetaCore) {
                const SemisparseTileMetaCore& semi = scheme->semisparseTileMetaCore[idx];
                if (semi.sa == 0) {
                    // No active columns, skip allocation
                    scheme->denseTiles[idx] = nullptr;
                    if (scheme->compute_inverse) {
                        scheme->inverseTiles[idx] = nullptr;
                        scheme->savedTiles[idx] = nullptr;
                    }
                    return StatusCode::Success;
                }
            }
        }

        const std::size_t elems = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);

        // Dense (factor) tile
        scheme->denseTiles[idx] = TileMemoryManager::allocate<double>(elems, group_index);
        if (!scheme->denseTiles[idx]) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for denseTiles[%d].\n", idx);
            return StatusCode::OutOfResources;
        }

        if (scheme->compute_inverse) {
            // Inverse tile
            scheme->inverseTiles[idx] = TileMemoryManager::allocate<double>(elems, group_index);
            if (!scheme->inverseTiles[idx]) {
                std::fprintf(stderr, "ERROR: Memory allocation failed for inverseTiles[%d].\n", idx);
                // Clean up already allocated denseTiles[idx]
                TileMemoryManager::deallocate(scheme->denseTiles[idx]);
                scheme->denseTiles[idx] = nullptr;
                return StatusCode::OutOfResources;
            }
            // Zero initialize (column-major leading dimension = h)
            LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, scheme->inverseTiles[idx], h);

            // Saved tile
            scheme->savedTiles[idx] = TileMemoryManager::allocate<double>(elems, group_index);
            if (!scheme->savedTiles[idx]) {
                std::fprintf(stderr, "ERROR: Memory allocation failed for savedTiles[%d].\n", idx);
                // Clean up already allocated buffers
                TileMemoryManager::deallocate(scheme->inverseTiles[idx]);
                scheme->inverseTiles[idx] = nullptr;
                TileMemoryManager::deallocate(scheme->denseTiles[idx]);
                scheme->denseTiles[idx] = nullptr;
                return StatusCode::OutOfResources;
            }
        }

        return StatusCode::Success;
    }

    static inline StatusCode init_inverse_identity_on_diagonals(TiledMatrix* scheme) {
        if (!scheme->compute_inverse || !scheme->inverseTiles) return StatusCode::Success;

        if (!scheme->tileMetaCore) return StatusCode::Success;

        for (int tile_idx = 0; tile_idx < scheme->dimTiledMatrix; ++tile_idx) {
            const int dense_idx = scheme->diagonal_mapper[tile_idx];
            if (dense_idx < 0) continue; // safety

            const TileMetaCore& m = scheme->tileMetaCore[dense_idx];
            const int h = (m.height > 0) ? m.height : scheme->tile_size;
            const int w = (m.width  > 0) ? m.width  : scheme->tile_size;

            double* inv = scheme->inverseTiles[dense_idx];
            if (!inv) continue; // should already be allocated

            const int ld  = h;                  // column-major leading dimension
            const int dsz = std::min(h, w);     // <-- handle rectangular tiles safely
            for (int ii = 0; ii < dsz; ++ii) {
                inv[ii * ld + ii] = 1.0;
            }
        }
        return StatusCode::Success;
    }

    static inline StatusCode init_inverse_identity_on_diagonals_variant1(TiledMatrix* scheme) {
        // Variant 1: Full dense - only one tile (index 0) covering entire matrix
        if (!scheme->compute_inverse || !scheme->inverseTiles) return StatusCode::Success;

        const int dense_idx = 0;  // Only one tile in variant 1
        double* inv = scheme->inverseTiles[dense_idx];
        if (!inv) return StatusCode::IllegalValue;

        // Tile dimensions are the full matrix size
        const int matrix_size = scheme->dim;
        const int ld  = matrix_size;
        const int dsz = matrix_size;

        for (int ii = 0; ii < dsz; ++ii) {
            inv[ii * ld + ii] = 1.0;
        }

        return StatusCode::Success;
    }

    static inline StatusCode init_inverse_identity_on_diagonals_variant2(TiledMatrix* scheme) {
        // Variant 2: Scaled dense - diagonal tiles in upper triangular layout
        if (!scheme->compute_inverse || !scheme->inverseTiles) return StatusCode::Success;

        const int N = scheme->dimTiledMatrix;
        const int tile_size = scheme->tile_size;
        const int matrix_size = scheme->dim;

        // In upper triangular storage, diagonal tile (i,i) is at position:
        // sum from k=0 to i-1 of (N-k) = i*N - i*(i-1)/2
        for (int i = 0; i < N; ++i) {
            const int dense_idx = i * N - (i * (i - 1)) / 2;

            if (dense_idx >= scheme->numActiveTiles) continue; // safety

            double* inv = scheme->inverseTiles[dense_idx];
            if (!inv) continue;

            // Diagonal tiles are always square
            // Last diagonal tile may be smaller if matrix_size is not a multiple of tile_size
            const int tile_dim = (i == N - 1) ? (matrix_size - i * tile_size) : tile_size;

            for (int ii = 0; ii < tile_dim; ++ii) {
                inv[ii * tile_dim + ii] = 1.0;
            }
        }

        return StatusCode::Success;
    }
    
    static void allocate_buffers_for_all_tiles(TiledMatrix* scheme, int group_index, int num_cores = 1) {
        #ifdef _OPENMP
        if (num_cores > 1) {
            std::atomic<int> error_code{0};

            #pragma omp parallel for schedule(static) num_threads(num_cores)
            for (int idx = 0; idx < scheme->numActiveTiles; ++idx) {
                if (error_code.load(std::memory_order_relaxed) != 0) {
                    continue;
                }

                const StatusCode sc = allocate_buffers_for_tile(scheme, idx, group_index);
                if (sc != StatusCode::Success) {
                    int expected = 0;
                    const int cast_code = static_cast<int>(sc);
                    error_code.compare_exchange_strong(expected, cast_code,
                                                        std::memory_order_relaxed,
                                                        std::memory_order_relaxed);
                }
            }

            const int final_code = error_code.load(std::memory_order_relaxed);
            if (final_code != 0) {
                std::fprintf(stderr, "ERROR: Parallel tile allocation failed with code %d. Exiting.\n", final_code);
                std::exit(EXIT_FAILURE);
            }

            const StatusCode init_status = init_inverse_identity_on_diagonals(scheme);
            if (init_status != StatusCode::Success) {
                std::fprintf(stderr, "ERROR: Failed to initialize inverse diagonals (parallel path). Exiting.\n");
                std::exit(EXIT_FAILURE);
            }
            return;
        }
        #else
        (void)num_cores; // suppress unused warning when OpenMP is disabled
        #endif

        for (int idx = 0; idx < scheme->numActiveTiles; ++idx) {
            const StatusCode sc = allocate_buffers_for_tile(scheme, idx, group_index);
            if (sc != StatusCode::Success) {
                std::fprintf(stderr, "ERROR: Tile allocation failed at index %d. Exiting.\n", idx);
                std::exit(EXIT_FAILURE);
            }
        }

        const StatusCode init_status = init_inverse_identity_on_diagonals(scheme);
        if (init_status != StatusCode::Success) {
            std::fprintf(stderr, "ERROR: Failed to initialize inverse diagonals (serial path). Exiting.\n");
            std::exit(EXIT_FAILURE);
        }
    }
    
    inline StatusCode allocate_dense_tiles(TiledMatrix *scheme, int group_index, int num_cores = 1) {
        // Check tile type mode
        int* params = sTiles_get_params();
        const int tile_type_mode = params[3];
        const int tile_corr_mode = params[0];
        const int num_active = scheme->numActiveTiles;

        if (tile_type_mode == 0 && tile_corr_mode > 0) {
            auto release_chunked_tiles = [&](DenseTile*& tile_array) {
                if (!tile_array) {
                    return;
                }
                for (int idx = 0; idx < num_active; ++idx) {
                    if (tile_array[idx]) {
                        TileMemoryManager::deallocate(tile_array[idx]);
                    }
                }
                TileMemoryManager::deallocate(tile_array);
                tile_array = nullptr;
            };

            release_chunked_tiles(scheme->chunkedDenseTiles);
            release_chunked_tiles(scheme->chunkedRhsTiles);
            release_chunked_tiles(scheme->chunkedSavedTiles);
            release_chunked_tiles(scheme->chunkedInverseTiles);
        }

        // Only proceed if tile_type_mode is 0 (dense only) or 3 (dense + semisparse)
        if (tile_type_mode != 0 && tile_type_mode != 3) {
            // Don't proceed with dense tile allocation
            return StatusCode::Success;
        }

        scheme->denseTiles = nullptr;
        scheme->inverseTiles = nullptr;
        scheme->savedTiles = nullptr;
#ifdef SMART_TILES
        scheme->rhsTiles = nullptr;
#endif

        scheme->denseTiles = TileMemoryManager::allocate<DenseTile>(num_active, group_index);
        if (!scheme->denseTiles || !scheme->tileMetaCore) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for denseTiles or tileMetaCore.\n");
            // Clean up denseTiles if it was allocated
            if (scheme->denseTiles) {
                TileMemoryManager::deallocate(scheme->denseTiles);
                scheme->denseTiles = nullptr;
            }
            return StatusCode::OutOfResources;
        }

        if (scheme->compute_inverse) {
            scheme->inverseTiles = TileMemoryManager::allocate<DenseTile>(num_active, group_index);
            scheme->savedTiles = TileMemoryManager::allocate<DenseTile>(num_active, group_index);
            if (!scheme->inverseTiles || !scheme->savedTiles) {
                std::fprintf(stderr, "ERROR: Memory allocation failed for inverseTiles or savedTiles.\n");
                // Clean up allocated arrays
                if (scheme->savedTiles) {
                    TileMemoryManager::deallocate(scheme->savedTiles);
                    scheme->savedTiles = nullptr;
                }
                if (scheme->inverseTiles) {
                    TileMemoryManager::deallocate(scheme->inverseTiles);
                    scheme->inverseTiles = nullptr;
                }
                TileMemoryManager::deallocate(scheme->denseTiles);
                scheme->denseTiles = nullptr;
                return StatusCode::OutOfResources;
            }
        } else {
            scheme->inverseTiles = nullptr;
            scheme->savedTiles = nullptr;
        }

        allocate_buffers_for_all_tiles(scheme, group_index, num_cores);

        return StatusCode::Success;
    }

    inline StatusCode allocate_dense_tiles_variant1(TiledMatrix *scheme, int group_index, int num_cores = 1) {
        // Variant 1: Full dense factorization - single tile for entire matrix
        // numActiveTiles should already be set to 1

        scheme->denseTiles = nullptr;
        scheme->inverseTiles = nullptr;
        scheme->savedTiles = nullptr;
#ifdef SMART_TILES
        scheme->rhsTiles = nullptr;
#endif

        // Allocate single dense tile for the entire matrix
        scheme->denseTiles = TileMemoryManager::allocate<DenseTile>(1, group_index);
        if (!scheme->denseTiles) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for variant 1 dense tile.\n");
            return StatusCode::OutOfResources;
        }

        if (scheme->compute_inverse) {
            scheme->inverseTiles = TileMemoryManager::allocate<DenseTile>(1, group_index);
            scheme->savedTiles = TileMemoryManager::allocate<DenseTile>(1, group_index);
            if (!scheme->inverseTiles || !scheme->savedTiles) {
                std::fprintf(stderr, "ERROR: Memory allocation failed for variant 1 inverse tiles.\n");
                if (scheme->savedTiles) {
                    TileMemoryManager::deallocate(scheme->savedTiles);
                    scheme->savedTiles = nullptr;
                }
                if (scheme->inverseTiles) {
                    TileMemoryManager::deallocate(scheme->inverseTiles);
                    scheme->inverseTiles = nullptr;
                }
                TileMemoryManager::deallocate(scheme->denseTiles);
                scheme->denseTiles = nullptr;
                return StatusCode::OutOfResources;
            }
        } else {
            scheme->inverseTiles = nullptr;
            scheme->savedTiles = nullptr;
        }

        // Allocate buffers for the single tile
        const StatusCode sc = allocate_buffers_for_tile(scheme, 0, group_index);
        if (sc != StatusCode::Success) {
            std::fprintf(stderr, "ERROR: Buffer allocation failed for variant 1 tile.\n");
            return sc;
        }

        // Initialize inverse tile to identity
        const StatusCode init_status = init_inverse_identity_on_diagonals_variant1(scheme);
        if (init_status != StatusCode::Success) {
            std::fprintf(stderr, "ERROR: Failed to initialize inverse diagonal (variant 1).\n");
            return init_status;
        }

        return StatusCode::Success;
    }

    inline StatusCode allocate_dense_tiles_variant2(TiledMatrix *scheme, int group_index, int num_cores = 1) {
        // Variant 2: Scaled dense factorization - upper triangular tiles
        // numActiveTiles should already be set to triangular_size
        const int num_active = scheme->numActiveTiles;

        scheme->denseTiles = nullptr;
        scheme->inverseTiles = nullptr;
        scheme->savedTiles = nullptr;
#ifdef SMART_TILES
        scheme->rhsTiles = nullptr;
#endif

        // Allocate upper triangular tiles
        scheme->denseTiles = TileMemoryManager::allocate<DenseTile>(num_active, group_index);
        if (!scheme->denseTiles) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for variant 2 dense tiles.\n");
            return StatusCode::OutOfResources;
        }

        if (scheme->compute_inverse) {
            scheme->inverseTiles = TileMemoryManager::allocate<DenseTile>(num_active, group_index);
            scheme->savedTiles = TileMemoryManager::allocate<DenseTile>(num_active, group_index);
            if (!scheme->inverseTiles || !scheme->savedTiles) {
                std::fprintf(stderr, "ERROR: Memory allocation failed for variant 2 inverse tiles.\n");
                if (scheme->savedTiles) {
                    TileMemoryManager::deallocate(scheme->savedTiles);
                    scheme->savedTiles = nullptr;
                }
                if (scheme->inverseTiles) {
                    TileMemoryManager::deallocate(scheme->inverseTiles);
                    scheme->inverseTiles = nullptr;
                }
                TileMemoryManager::deallocate(scheme->denseTiles);
                scheme->denseTiles = nullptr;
                return StatusCode::OutOfResources;
            }
        } else {
            scheme->inverseTiles = nullptr;
            scheme->savedTiles = nullptr;
        }

        // Allocate buffers for all tiles
        for (int idx = 0; idx < num_active; ++idx) {
            const StatusCode sc = allocate_buffers_for_tile(scheme, idx, group_index);
            if (sc != StatusCode::Success) {
                std::fprintf(stderr, "ERROR: Buffer allocation failed for variant 2 tile %d.\n", idx);
                return sc;
            }
        }

        // Initialize inverse diagonal tiles to identity
        const StatusCode init_status = init_inverse_identity_on_diagonals_variant2(scheme);
        if (init_status != StatusCode::Success) {
            std::fprintf(stderr, "ERROR: Failed to initialize inverse diagonals (variant 2).\n");
            return init_status;
        }

        return StatusCode::Success;
    }

    inline void update_x_dense_tiles(int global_index, TiledMatrix **scheme, double *x, bool nested) {

        if (!scheme || !scheme[global_index] || !x) {
            return;
        }

        TiledMatrix* S = scheme[global_index];
        // Skip the pad tail of the COO (appended diagonals from ND / SCOTCH
        // block padding). scatter_nd_padding_dense_tiles() sets those cells
        // below; reading x[] past the user's allocation would be UB.
        const int nnz = S->original_nnz - S->nd_padding;
        const int cores = S->num_cores;
        // If denseTiles or tileMetaCore are not available (e.g., SmartTiles-only builds),
        // skip this routine. The debug path can reconstruct a dense view directly from x.
        if (!S->denseTiles || !S->tileMetaCore) {
            return;
        }

        // Without the lookup structures, we cannot scatter nnz values into tiles safely.
        if (!S->tile_index_lookup || !S->element_offset_lookup) {
            sTiles::Logger::error("[update_x_sparse_dense_tiles] Missing lookup tables (tile_index_lookup=",
                                  static_cast<const void*>(S->tile_index_lookup),
                                  ", element_offset_lookup=",
                                  static_cast<const void*>(S->element_offset_lookup),
                                  ") — ensure build_dense_tile_lookup ran successfully.");
            std::exit(EXIT_FAILURE);
        }
        // Allow caller to hint nested capability; also consult runtime
        const bool nested_enabled =
        #ifdef _OPENMP
            (nested || (omp_get_max_active_levels() > 1))
        #else
            nested
        #endif
        ;
        const bool can_spawn =
        #ifdef _OPENMP
            (!omp_in_parallel()) || nested_enabled
        #else
            false
        #endif
        ;
        const bool do_parallel = (cores > 1 && nnz > 5000 && can_spawn);

        //const bool use_smart_tiles = (S->facTiles != nullptr);

        //if (!use_smart_tiles) {
            // Dense tiles path (fast mode): clear tiles then scatter nnz values
        #ifdef _OPENMP
            if (do_parallel) {
                #pragma omp parallel num_threads(cores)
                {
                    #pragma omp for schedule(static)
                    for (int t = 0; t < S->numActiveTiles; ++t) {
                        if (!S->denseTiles[t]) continue;
                        const TileMetaCore& m = S->tileMetaCore[t];
                        const int h = (m.height > 0) ? m.height : S->tile_size;
                        const int w = (m.width  > 0) ? m.width  : S->tile_size;
                        LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, S->denseTiles[t], h);
                    }

                    #pragma omp for schedule(static)
                    for (int idx = 0; idx < nnz; ++idx) {
                        const int tile = S->tile_index_lookup[idx];
                        if (!S->denseTiles[tile]) continue;
                        const int off  = S->element_offset_lookup[idx];
                        S->denseTiles[tile][off] = x[idx];
                    }
                }
            } else
        #endif
            {
                for (int t = 0; t < S->numActiveTiles; ++t) {
                    if (!S->denseTiles[t]) continue;
                    const TileMetaCore& m = S->tileMetaCore[t];
                    const int h = (m.height > 0) ? m.height : S->tile_size;
                    const int w = (m.width  > 0) ? m.width  : S->tile_size;
                    LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, S->denseTiles[t], h);
                }
                for (int idx = 0; idx < nnz; ++idx) {
                    const int tile = S->tile_index_lookup[idx];
                    if (!S->denseTiles[tile]) continue;
                    const int off  = S->element_offset_lookup[idx];
                    S->denseTiles[tile][off] = x[idx];
                }
            }

            // Optional GPU path left commented as in original
            // #ifdef STILES_GPU
            //     INTERNAL_UPDATE_X_APPROACH_2_PHASE_1_CPU_TO_GPU_serial(global_index, scheme);
            // #endif
        scatter_nd_padding_dense_tiles(S);
    }


    inline void update_x_dense_tiles_ones(int global_index, TiledMatrix **scheme, bool nested) {

        if (!scheme || !scheme[global_index]) {
            return;
        }

        TiledMatrix* S = scheme[global_index];
        const int nnz = S->original_nnz;
        const int cores = S->num_cores;
        // If denseTiles or tileMetaCore are not available (e.g., SmartTiles-only builds),
        // skip this routine. The debug path can reconstruct a dense view directly from x.
        if (!S->denseTiles || !S->tileMetaCore) {
            return;
        }

        // Without the lookup structures, we cannot scatter nnz values into tiles safely.
        if (!S->tile_index_lookup || !S->element_offset_lookup) {
            sTiles::Logger::error("[update_x_sparse_dense_tiles] Missing lookup tables (tile_index_lookup=",
                                  static_cast<const void*>(S->tile_index_lookup),
                                  ", element_offset_lookup=",
                                  static_cast<const void*>(S->element_offset_lookup),
                                  ") — ensure build_dense_tile_lookup ran successfully.");
            std::exit(EXIT_FAILURE);
        }
        // Allow caller to hint nested capability; also consult runtime
        const bool nested_enabled =
        #ifdef _OPENMP
            (nested || (omp_get_max_active_levels() > 1))
        #else
            nested
        #endif
        ;
        const bool can_spawn =
        #ifdef _OPENMP
            (!omp_in_parallel()) || nested_enabled
        #else
            false
        #endif
        ;
        const bool do_parallel = (cores > 1 && nnz > 5000 && can_spawn);

        //const bool use_smart_tiles = (S->facTiles != nullptr);

        //if (!use_smart_tiles) {
            // Dense tiles path (fast mode): clear tiles then scatter nnz values
        #ifdef _OPENMP
            if (do_parallel) {
                #pragma omp parallel num_threads(cores)
                {
                    #pragma omp for schedule(static)
                    for (int t = 0; t < S->numActiveTiles; ++t) {
                        if (!S->denseTiles[t]) continue;
                        const TileMetaCore& m = S->tileMetaCore[t];
                        const int h = (m.height > 0) ? m.height : S->tile_size;
                        const int w = (m.width  > 0) ? m.width  : S->tile_size;
                        LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, S->denseTiles[t], h);
                    }

                    #pragma omp for schedule(static)
                    for (int idx = 0; idx < nnz; ++idx) {
                        const int tile = S->tile_index_lookup[idx];
                        if (!S->denseTiles[tile]) continue;
                        const int off  = S->element_offset_lookup[idx];
                        S->denseTiles[tile][off] = 1.0;
                    }
                }
            } else
        #endif
            {
                for (int t = 0; t < S->numActiveTiles; ++t) {
                    if (!S->denseTiles[t]) continue;
                    const TileMetaCore& m = S->tileMetaCore[t];
                    const int h = (m.height > 0) ? m.height : S->tile_size;
                    const int w = (m.width  > 0) ? m.width  : S->tile_size;
                    LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, S->denseTiles[t], h);
                }
                for (int idx = 0; idx < nnz; ++idx) {
                    const int tile = S->tile_index_lookup[idx];
                    if (!S->denseTiles[tile]) continue;
                    const int off  = S->element_offset_lookup[idx];
                    S->denseTiles[tile][off] = 1.0;
                }
            }

        scatter_nd_padding_dense_tiles(S);
    }


    // Optimized for variant 1: single tile covering entire matrix
    inline void update_x_dense_tiles_variant1(int global_index, TiledMatrix **scheme, double *x, bool nested) {
        (void)nested; // unused in optimized version

        if (!scheme || !scheme[global_index] || !x) {
            return;
        }

        TiledMatrix* S = scheme[global_index];
        // Variant 1 almost never gets padding (single tile ⇒ already aligned),
        // but keep the subtraction for symmetry with the other scatter paths.
        const int nnz = S->original_nnz - S->nd_padding;
        const int cores = S->num_cores;

        // Variant 1: single tile at index 0
        if (!S->denseTiles || !S->denseTiles[0]) {
            return;
        }

        // element_offset_lookup gives column-major offset directly into the single tile
        if (!S->element_offset_lookup) {
            sTiles::Logger::error("[update_x_dense_tiles_variant1] Missing element_offset_lookup");
            std::exit(EXIT_FAILURE);
        }

        double* tile = S->denseTiles[0];
        const int N = S->dim;  // tile size = matrix size for variant 1

        // Clear the single tile
        LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', N, N, 0.0, 0.0, tile, N);

        // Scatter nnz values into the single tile
    #ifdef _OPENMP
        const bool do_parallel = (cores > 1 && nnz > 5000);
        if (do_parallel) {
            #pragma omp parallel for schedule(static) num_threads(cores)
            for (int idx = 0; idx < nnz; ++idx) {
                const int off = S->element_offset_lookup[idx];
                tile[off] = x[idx];
            }
        } else
    #endif
        {
            for (int idx = 0; idx < nnz; ++idx) {
                const int off = S->element_offset_lookup[idx];
                tile[off] = x[idx];
            }
        }
        // No ND padding for variant 1
    }

    inline void update_x_dense_tiles_variant2(int global_index, TiledMatrix **scheme, double *x, bool nested) {

        if (!scheme || !scheme[global_index] || !x) {
            return;
        }

        TiledMatrix* S = scheme[global_index];
        // Skip the pad-diagonal tail; scatter_nd_padding_dense_tiles() below
        // sets those cells, and reading x[] past the user's allocation is UB.
        const int nnz = S->original_nnz - S->nd_padding;
        const int cores = S->num_cores;
        // If denseTiles or tileMetaCore are not available (e.g., SmartTiles-only builds),
        // skip this routine. The debug path can reconstruct a dense view directly from x.
        if (!S->denseTiles || !S->tileMetaCore) {
            return;
        }

        // Without the lookup structures, we cannot scatter nnz values into tiles safely.
        if (!S->tile_index_lookup || !S->element_offset_lookup) {
            sTiles::Logger::error("[update_x_sparse_dense_tiles] Missing lookup tables (tile_index_lookup=",
                                  static_cast<const void*>(S->tile_index_lookup),
                                  ", element_offset_lookup=",
                                  static_cast<const void*>(S->element_offset_lookup),
                                  ") — ensure build_dense_tile_lookup ran successfully.");
            std::exit(EXIT_FAILURE);
        }
        // Allow caller to hint nested capability; also consult runtime
        const bool nested_enabled =
        #ifdef _OPENMP
            (nested || (omp_get_max_active_levels() > 1))
        #else
            nested
        #endif
        ;
        const bool can_spawn =
        #ifdef _OPENMP
            (!omp_in_parallel()) || nested_enabled
        #else
            false
        #endif
        ;
        const bool do_parallel = (cores > 1 && nnz > 5000 && can_spawn);

        //const bool use_smart_tiles = (S->facTiles != nullptr);

        //if (!use_smart_tiles) {
            // Dense tiles path (fast mode): clear tiles then scatter nnz values
        #ifdef _OPENMP
            if (do_parallel) {
                #pragma omp parallel num_threads(cores)
                {
                    #pragma omp for schedule(static)
                    for (int t = 0; t < S->numActiveTiles; ++t) {
                        if (!S->denseTiles[t]) continue;
                        const TileMetaCore& m = S->tileMetaCore[t];
                        const int h = (m.height > 0) ? m.height : S->tile_size;
                        const int w = (m.width  > 0) ? m.width  : S->tile_size;
                        LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, S->denseTiles[t], h);
                    }

                    #pragma omp for schedule(static)
                    for (int idx = 0; idx < nnz; ++idx) {
                        const int tile = S->tile_index_lookup[idx];
                        if (!S->denseTiles[tile]) continue;
                        const int off  = S->element_offset_lookup[idx];
                        S->denseTiles[tile][off] = x[idx];
                    }
                }
            } else
        #endif
            {
                for (int t = 0; t < S->numActiveTiles; ++t) {
                    if (!S->denseTiles[t]) continue;
                    const TileMetaCore& m = S->tileMetaCore[t];
                    const int h = (m.height > 0) ? m.height : S->tile_size;
                    const int w = (m.width  > 0) ? m.width  : S->tile_size;
                    LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, S->denseTiles[t], h);
                }
                for (int idx = 0; idx < nnz; ++idx) {
                    const int tile = S->tile_index_lookup[idx];
                    if (!S->denseTiles[tile]) continue;
                    const int off  = S->element_offset_lookup[idx];
                    S->denseTiles[tile][off] = x[idx];
                }
            }

            // Optional GPU path left commented as in original
            // #ifdef STILES_GPU
            //     INTERNAL_UPDATE_X_APPROACH_2_PHASE_1_CPU_TO_GPU_serial(global_index, scheme);
            // #endif
        scatter_nd_padding_dense_tiles(S);
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /*
        NEW: Unified allocation function that allocates ALL lookup structures once (for all modes):
        1. tile_index_lookup      - needed by all modes
        2. withinTileRow          - needed by sparse and semisparse modes
        3. withinTileCol          - needed by sparse and semisparse modes
        4. diagonal_bmapper       - needed by all modes
        5. nnz_tile_counter       - needed by sparse mode
        6. element_offset_lookup  - needed by dense mode

        NOTE: This function allocates MORE memory than individual mode functions,
        but ensures all structures are available regardless of mode.
    */

