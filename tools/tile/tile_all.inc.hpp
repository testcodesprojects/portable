    static inline void build_all_tile_lookup_init_serial(sTiles_call **call_info, TiledMatrix *scheme, int group_index) {
        const int num_tiles = scheme->dimTiledMatrix;
        const int num_tiles_active = scheme->numActiveTiles;
        const int original_nnz = scheme->original_nnz;

        // Allocate structures needed by ALL modes
        scheme->tile_index_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->diagonal_bmapper = TileMemoryManager::allocateZero<bool>(num_tiles_active, group_index);

        // Allocate structures needed by SPARSE and SEMISPARSE modes
        scheme->withinTileRow = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->withinTileCol = TileMemoryManager::allocateZero<int>(original_nnz, group_index);

        // Allocate structure needed by SPARSE mode only
        scheme->nnz_tile_counter = TileMemoryManager::allocateZero<int>(num_tiles_active, group_index);

        // Allocate structure needed by DENSE mode only
        scheme->element_offset_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
    
        if (scheme->use_ordering == 0) {
            for (int index = 0; index < original_nnz; ++index) {
                const int col = (*call_info)->row_indices[index];
                const int row = (*call_info)->col_indices[index];

                const int tileRow = row / (*call_info)->tile_size;
                const int tileCol = col / (*call_info)->tile_size;

                if (tileRow > tileCol) {
                    sTiles::Logger::error("build_all_tile_lookup_init_serial: unexpected lower-triangular entry (row=", 
                                          row, ", col=", col, ") while ordering=0.");
                    std::exit(EXIT_FAILURE);
                }

                const int indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
                
                // 1. Sparse Counter
                scheme->nnz_tile_counter[indexed_tile]++;

                // 2. Diagonal Mapper
                if (tileRow == tileCol) scheme->diagonal_bmapper[indexed_tile] = true;

                // 3. Tile Index
                scheme->tile_index_lookup[index] = indexed_tile;

                // 4. Within Tile Coords
                const int withinTileRow = row % (*call_info)->tile_size;
                const int withinTileCol = col % (*call_info)->tile_size;
                scheme->withinTileRow[index] = withinTileRow;
                scheme->withinTileCol[index] = withinTileCol;

                // 5. Dense Offset
                // Leading dimension depends on the row tile (if it's the last one, it's remainder)
                const int tile_ld = (tileRow == num_tiles - 1)
                                    ? scheme->remainderTileSize
                                    : scheme->tile_size;
                scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);
            }
        } else if (scheme->use_ordering == 1) {
            for (int index = 0; index < original_nnz; ++index) {
                int row, col;
                // Apply permutation and ensure upper triangular relative to permutation
                if (scheme->element_perm[(*call_info)->col_indices[index]] < scheme->element_perm[(*call_info)->row_indices[index]]) {
                    col = scheme->element_perm[(*call_info)->row_indices[index]];
                    row = scheme->element_perm[(*call_info)->col_indices[index]];
                } else {
                    row = scheme->element_perm[(*call_info)->row_indices[index]];
                    col = scheme->element_perm[(*call_info)->col_indices[index]];
                }

                const int tileRow = row / scheme->tile_size;
                const int tileCol = col / scheme->tile_size;

                int indexed_tile, withinTileRow, withinTileCol, tile_ld;

                if (tileRow <= tileCol) {
                    indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
                    withinTileRow = row % (*call_info)->tile_size;
                    withinTileCol = col % (*call_info)->tile_size;
                    tile_ld = (tileRow == num_tiles - 1) ? scheme->remainderTileSize : scheme->tile_size;

                    if (tileRow == tileCol) {
                        scheme->diagonal_bmapper[indexed_tile] = true;
                    }

                    scheme->withinTileRow[index] = withinTileRow;
                    scheme->withinTileCol[index] = withinTileCol;

                    int row_for_offset = withinTileRow;
                    int col_for_offset = withinTileCol;
                    if (tileRow == tileCol && row_for_offset > col_for_offset) {
                        std::swap(row_for_offset, col_for_offset);
                    }
                    scheme->element_offset_lookup[index] = row_for_offset + (col_for_offset * tile_ld);
                } else {
                    // This block handles cases where logic might flip tileRow/tileCol relative to perm, 
                    // though the initial swap should handle most. 
                    indexed_tile = scheme->mapper.map_ij(tileCol, tileRow, num_tiles);
                    withinTileRow = col % (*call_info)->tile_size;
                    withinTileCol = row % (*call_info)->tile_size;
                    
                    // Destination tile row is tileCol here
                    tile_ld = (tileCol == num_tiles - 1) ? scheme->remainderTileSize : scheme->tile_size;
                    
                    scheme->withinTileRow[index] = withinTileRow;
                    scheme->withinTileCol[index] = withinTileCol;
                    scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);
                }

                scheme->tile_index_lookup[index] = indexed_tile;
                scheme->nnz_tile_counter[indexed_tile]++;
            }
        }
    }

    static inline void build_all_tile_lookup_init_parallel(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {
        const int original_nnz = scheme->original_nnz;
        const int num_tiles = scheme->dimTiledMatrix;
        const int num_tiles_active = scheme->numActiveTiles;

        const int threads = (num_cores > 0) ? num_cores : omp_get_max_threads();
        const bool nested_enabled = (omp_get_max_active_levels() > 1);
        const bool can_spawn = (!omp_in_parallel()) || nested_enabled;

        if (!can_spawn || threads < 2) {
            build_all_tile_lookup_init_serial(call_info, scheme, group_index);
            return;
        }

        scheme->tile_index_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->diagonal_bmapper = TileMemoryManager::allocateZero<bool>(num_tiles_active, group_index);
        scheme->withinTileRow = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->withinTileCol = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->nnz_tile_counter = TileMemoryManager::allocateZero<int>(num_tiles_active, group_index);
        scheme->element_offset_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);

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
                
                // Atomic increment for sparse counter
                #pragma omp atomic
                scheme->nnz_tile_counter[indexed_tile]++;

                if (tileRow == tileCol) {
                    #pragma omp atomic write
                    scheme->diagonal_bmapper[indexed_tile] = true;
                }

                const int withinTileRow = row % (*call_info)->tile_size;
                const int withinTileCol = col % (*call_info)->tile_size;
                
                scheme->tile_index_lookup[index] = indexed_tile;
                scheme->withinTileRow[index] = withinTileRow;
                scheme->withinTileCol[index] = withinTileCol;

                const int tile_ld = (tileRow == num_tiles - 1) ? scheme->remainderTileSize : scheme->tile_size;
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
                int indexed_tile, tile_ld;

                if (tileRow <= tileCol) {
                    indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
                    const int withinTileRow = row % (*call_info)->tile_size;
                    const int withinTileCol = col % (*call_info)->tile_size;
                    tile_ld = (tileRow == num_tiles - 1) ? scheme->remainderTileSize : scheme->tile_size;

                    #pragma omp atomic
                    scheme->nnz_tile_counter[indexed_tile]++;

                    if (tileRow == tileCol) {
                        #pragma omp atomic write
                        scheme->diagonal_bmapper[indexed_tile] = true;
                    }

                    scheme->withinTileRow[index] = withinTileRow;
                    scheme->withinTileCol[index] = withinTileCol;

                    int row_for_offset = withinTileRow;
                    int col_for_offset = withinTileCol;
                    if (tileRow == tileCol && row_for_offset > col_for_offset) {
                        std::swap(row_for_offset, col_for_offset);
                    }
                    scheme->element_offset_lookup[index] = row_for_offset + (col_for_offset * tile_ld);
                } else {
                    indexed_tile = scheme->mapper.map_ij(tileCol, tileRow, num_tiles);
                    const int withinTileRow = col % (*call_info)->tile_size;
                    const int withinTileCol = row % (*call_info)->tile_size;
                    tile_ld = (tileCol == num_tiles - 1) ? scheme->remainderTileSize : scheme->tile_size;

                    #pragma omp atomic
                    scheme->nnz_tile_counter[indexed_tile]++;

                    scheme->withinTileRow[index] = withinTileRow;
                    scheme->withinTileCol[index] = withinTileCol;
                    scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);
                }
                scheme->tile_index_lookup[index] = indexed_tile;
            }
        }

        if (error_flag.load(std::memory_order_relaxed)) {
            sTiles::Logger::error("build_all_tile_lookup_init_parallel: unexpected lower-triangular entry encountered.");
            std::exit(EXIT_FAILURE);
        }
    }

    static inline void build_all_tile_lookup_phase1(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) { 
        const int n = scheme->original_order;
        scheme->remainderTileSize = (n % scheme->tile_size == 0) ? scheme->tile_size : (n % scheme->tile_size);

        if (num_cores < 2 || scheme->original_nnz < 5000) { 
            build_all_tile_lookup_init_serial(call_info, scheme, group_index); 
            return; 
        }
    #ifndef _OPENMP
        build_all_tile_lookup_init_serial(call_info, scheme, group_index); 
        return;
    #else
        const bool nested_enabled = (omp_get_max_active_levels() > 1);
        const bool can_spawn = (!omp_in_parallel()) || nested_enabled;
        if (!can_spawn) { 
            build_all_tile_lookup_init_serial(call_info, scheme, group_index); 
            return; 
        }
        build_all_tile_lookup_init_parallel(call_info, scheme, group_index, num_cores);
    #endif
    }

    inline StatusCode build_all_tile_lookup(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {
        // Check tile type mode
        int* params = sTiles_get_params();
        const int tile_type_mode = params[sTiles::param::TileTypeMode];

        // Only proceed if tile_type_mode is 3 (dense + semisparse together)
        if (tile_type_mode != 3) {
            // Don't proceed with all tile lookup
            return StatusCode::Success;
        }

        if (!call_info || !*call_info || !scheme) return StatusCode::InvalidArgument;
        
        const int num_active = scheme->numActiveTiles;
        if (num_active < 0) return StatusCode::InvalidArgument;

        // Reset all core pointers
        scheme->tileMetaCore = nullptr;
        scheme->semisparseTileMetaCore = nullptr;
        scheme->sparseTileMetaCore = nullptr;
        scheme->sparseTileMetaData = nullptr;
        scheme->invSparseTileMetaCore = nullptr;

        // 1. Allocate Dense Core
        scheme->tileMetaCore = TileMemoryManager::allocate<TileMetaCore>(num_active, group_index);
        
        // 2. Allocate Semisparse Core
        scheme->semisparseTileMetaCore = TileMemoryManager::allocate<SemisparseTileMetaCore>(num_active, group_index);
        if (!scheme->semisparseTileMetaCore) {
            sTiles::Logger::errorf("Memory allocation failed for semisparseTileMetaCore (build_all).");
            return StatusCode::OutOfResources;
        }
        // Construct SemisparseTileMetaCore objects (contains std::vector members)
        for (int t = 0; t < num_active; ++t) {
            new (&scheme->semisparseTileMetaCore[t]) SemisparseTileMetaCore();
        }

        // 3. Allocate Sparse Core & Data
        scheme->sparseTileMetaCore = TileMemoryManager::allocate<SparseTileMetaCore>(num_active, group_index);
        scheme->sparseTileMetaData = TileMemoryManager::allocate<SparseTileMetaData>(num_active, group_index);
        // Construct SparseTileMetaCore and SparseTileMetaData objects (contain std::vector members)
        if (scheme->sparseTileMetaCore && scheme->sparseTileMetaData) {
            for (int t = 0; t < num_active; ++t) {
                new (&scheme->sparseTileMetaCore[t]) SparseTileMetaCore();
                new (&scheme->sparseTileMetaData[t]) SparseTileMetaData();
            }
        }
        if (scheme->compute_inverse) {
            scheme->invSparseTileMetaCore = TileMemoryManager::allocate<SparseTileMetaCore>(num_active, group_index);
            // Construct invSparseTileMetaCore objects (contain std::vector members)
            if (scheme->invSparseTileMetaCore) {
                for (int t = 0; t < num_active; ++t) {
                    new (&scheme->invSparseTileMetaCore[t]) SparseTileMetaCore();
                }
            }
        }

        if (!scheme->tileMetaCore || !scheme->semisparseTileMetaCore || !scheme->sparseTileMetaCore || !scheme->sparseTileMetaData) {
             sTiles::Logger::errorf("Memory allocation failed for one or more core metadata structures in build_all.");
             return StatusCode::OutOfResources;
        }

        // 4. Initialize Extents (Dense Metadata)
        set_tile_extents(&scheme);

        // 5. Phase 1: Build combined lookup tables (fills offsets, indices, counters)
        build_all_tile_lookup_phase1(call_info, scheme, group_index, num_cores);

        // 6. Phase 2: Populate specific sparse/semisparse internal structures based on the lookups
        // Note: These functions assume lookups are ready. 
        build_semisparse_tile_lookup_phase2(call_info, scheme, group_index, num_cores);
        build_sparse_tile_lookup_phase2(call_info, scheme, group_index, num_cores);

        return StatusCode::Ok;
    }

    inline StatusCode allocate_all_tiles(TiledMatrix *scheme, int group_index, int num_cores) {
        StatusCode sc;

        // Allocate Dense Tiles (denseTiles, inverseTiles, savedTiles)
        sc = allocate_dense_tiles(scheme, group_index, num_cores);
        if (sc != StatusCode::Success) return sc;

        // Allocate Semisparse Tiles (chunkedDenseTiles, etc.)
        sc = allocate_semisparse_tiles(scheme, group_index, num_cores);
        if (sc != StatusCode::Success) return sc;

        // Allocate Sparse Tiles (facTiles, etc.)
        sc = allocate_sparse_tiles(scheme, group_index, num_cores);
        if (sc != StatusCode::Success) return sc;

        return StatusCode::Success;
    }


    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    inline void export_x_dense_tiles_serial(TiledMatrix *S, double *out) {
        if (!S || !out) {
            return;
        }

        const int nnz = S->original_nnz;
        if (nnz <= 0) {
            return;
        }

        if (!S->denseTiles || !S->tileMetaCore) {
            return;
        }

        if (!S->tile_index_lookup || !S->element_offset_lookup) {
            sTiles::Logger::error("[export_x_dense_tiles_serial] Missing lookup tables (tile_index_lookup=",
                                  static_cast<const void*>(S->tile_index_lookup),
                                  ", element_offset_lookup=",
                                  static_cast<const void*>(S->element_offset_lookup),
                                  ") - ensure build_dense_tile_lookup ran successfully.");
            std::exit(EXIT_FAILURE);
        }

        for (int idx = 0; idx < nnz; ++idx) {
            const int tile = S->tile_index_lookup[idx];
            if (tile < 0 || tile >= S->numActiveTiles) {
                out[idx] = 0.0;
                continue;
            }

            double* A = S->denseTiles[tile];
            if (!A) {
                out[idx] = 0.0;
                continue;
            }

            const int off = S->element_offset_lookup[idx];
            out[idx] = A[off];
        }
    }

    inline void export_x_dense_tiles_serial(int global_index, TiledMatrix **scheme, double *out) {
        if (!scheme || global_index < 0 || !out) {
            return;
        }
        TiledMatrix* S = scheme[global_index];
        export_x_dense_tiles_serial(S, out);
    }

    inline void export_x_semisparse_tiles_serial(TiledMatrix *S, double *out) {
        if (!S || !out) {
            return;
        }
        const int nnz = S->original_nnz;
        const bool *diag_map = S->diagonal_bmapper;

        if (!S->chunkedDenseTiles || !S->tileMetaCore || !S->semisparseTileMetaCore) return;

        if (!S->tile_index_lookup || !S->withinTileRow || !S->withinTileCol) {
            sTiles::Logger::error("[export_x_semisparse_tiles_serial] Missing lookup tables (tile_index_lookup=",
                                static_cast<const void*>(S->tile_index_lookup),
                                ", withinTileRow=",
                                static_cast<const void*>(S->withinTileRow),
                                ", withinTileCol=",
                                static_cast<const void*>(S->withinTileCol),
                                ") - ensure build_semisparse_tile_lookup ran successfully.");
            std::exit(EXIT_FAILURE);
        }

        auto is_diagonal_tile = [&](int tile_idx) -> bool {
            const TileMetaCore& meta = S->tileMetaCore[tile_idx];
            return meta.row == meta.col;
        };

        auto uses_lapack_diagonal = [&](int tile_idx) -> bool {
            return diag_map && diag_map[tile_idx];
        };

        for (int idx = 0; idx < nnz; ++idx) {
            const int tile = S->tile_index_lookup[idx];
            if (tile < 0 || tile >= S->numActiveTiles) { out[idx] = 0.0; continue; }

            double* chunk = S->chunkedDenseTiles[tile];
            if (!chunk) { out[idx] = 0.0; continue; }

            const SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[tile];
            const bool lapack_diag = uses_lapack_diagonal(tile);
            const bool is_diag = is_diagonal_tile(tile);

            const int local_row = S->withinTileRow[idx];
            const int local_col = S->withinTileCol[idx];
            if (local_row < 0 || local_col < 0) { out[idx] = 0.0; continue; }

            const TileMetaCore& meta = S->tileMetaCore[tile];
            const int h = (meta.height > 0) ? meta.height : S->tile_size;
            const int w = (meta.width  > 0) ? meta.width  : S->tile_size;
            if (h <= 0 || w <= 0) { out[idx] = 0.0; continue; }

            int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
            if ((lapack_diag || is_diag) && diag_cols <= 0) diag_cols = w;

            if (lapack_diag) {
                if (local_col >= w || local_row >= h) { out[idx] = 0.0; continue; }
                if (diag_cols <= 0) { out[idx] = 0.0; continue; }
                const int band = local_col - local_row;
                if (band < 0 || band >= diag_cols) { out[idx] = 0.0; continue; }
                const int lapack_row = diag_cols - 1 - band;
                const std::size_t off = static_cast<std::size_t>(lapack_row) + static_cast<std::size_t>(diag_cols) * static_cast<std::size_t>(local_col);
                out[idx] = chunk[off];
                continue;
            }

            if (local_row >= h) { out[idx] = 0.0; continue; }

            if (is_diag && diag_cols > 0) {
                const int band = local_col - local_row;
                if (band < 0 || band >= diag_cols) { out[idx] = 0.0; continue; }
                const std::size_t off = static_cast<std::size_t>(local_row) + static_cast<std::size_t>(band) * static_cast<std::size_t>(h);
                out[idx] = chunk[off];
                continue;
            }

            if (semi.acol.empty() || semi.sa <= 0) { out[idx] = 0.0; continue; }
            if (local_col >= static_cast<int>(semi.acol.size())) { out[idx] = 0.0; continue; }

            const int active_col = semi.acol[static_cast<std::size_t>(local_col)];
            if (active_col < 0 || active_col >= semi.sa) { out[idx] = 0.0; continue; }

            const std::size_t off = static_cast<std::size_t>(local_row) + static_cast<std::size_t>(active_col) * static_cast<std::size_t>(h);
            out[idx] = chunk[off];
        }
    }

    inline void export_x_semisparse_tiles_serial(int global_index, TiledMatrix **scheme, double *out) {
        if (!scheme || global_index < 0 || !out) {
            return;
        }
        TiledMatrix* S = scheme[global_index];
        export_x_semisparse_tiles_serial(S, out);
    }


    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------


}}

 // void build_tile_lookup(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {
    //     // freshly generated dense lookup so both representations stay in sync.
    //     if (call_info && *call_info) {

    //         // For comparison builds only: also build SmartTiles lookup so both
    //         // representations are available. Guarded by STILES_SMART_COMPARE.
    //         #ifdef STILES_SMART_COMPARE
    //             if (call_info && *call_info) {
    //                 const int saved_variant = (*call_info)->factorization_variant;
    //                 (*call_info)->factorization_variant = 3;
    //                 build_semisparse_tile_lookup_phase1(call_info, scheme, group_index, num_cores);
    //                 build_dense_tile_lookup(call_info, scheme, group_index, num_cores);
    //                 (*call_info)->factorization_variant = saved_variant;
    //             }
    //         #else

    //             #ifdef STILES_FASTMODE

    //                 if((*call_info)->factorization_variant == 0){
    //                     build_semisparse_tile_lookup_phase1(call_info, scheme, group_index, num_cores);
    //                 }else if((*call_info)->factorization_variant == 1){
    //                     sTiles::Logger::error("factorization_variant 1 not implemented in build_tile_lookup.");
    //                     std::exit(EXIT_FAILURE);
    //                 }else if((*call_info)->factorization_variant == 2){
    //                     sTiles::Logger::error("factorization_variant 2 not implemented in build_tile_lookup.");
    //                     std::exit(EXIT_FAILURE);
    //                 }else if((*call_info)->factorization_variant == 3){
    //                     build_semisparse_tile_lookup_phase1(call_info, scheme, group_index, num_cores);
    //                     build_dense_tile_lookup(call_info, scheme, group_index, num_cores);
    //                 }

    //             #elif STILES_SEMISPARSEMODE

    //                 if((*call_info)->factorization_variant == 0){
    //                     build_semisparse_tile_lookup_phase1(call_info, scheme, group_index, num_cores);
    //                     build_semisparse_tile_lookup(call_info, scheme, group_index, num_cores); //it use metaTiles instead of factiles.
    //                 }else{

    //                     std::cout << "FIX ME " << std::endl;
    //                     exit(0);
    //                 }

    //             #endif 

    //         #endif
    //     }

        

    // }

    // void build_sparse_tile_lookup_init_serial(sTiles_call **call_info, TiledMatrix *scheme, int group_index) {
    //     int k, i, j, indexed_tile, temp_count;

    //     const int num_tiles = scheme->dimTiledMatrix;
    //     const int num_tiles_active = scheme->numActiveTiles;
    //     const int original_nnz = scheme->original_nnz;

    //     scheme->tile_index_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
    //     scheme->withinTileRow = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
    //     scheme->withinTileCol = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
    //     scheme->diagonal_bmapper = TileMemoryManager::allocateZero<bool>(num_tiles_active, group_index);
    //     scheme->nnz_tile_counter = TileMemoryManager::allocateZero<int>(num_tiles_active, group_index);

    //     if(scheme->use_ordering==0){

    //         //-----------------------------------------------------------------------------------------------------------------------------------------------------
    //         //-----------------------------------------------------------------------------------------------------------------------------------------------------
            
    //         for (int index = 0; index < scheme->original_nnz; index++) {
            
    //             int col = (*call_info)->row_indices[index];
    //             int row = (*call_info)->col_indices[index];

    //             int tileRow = row / (*call_info)->tile_size;
    //             int tileCol = col / (*call_info)->tile_size;

    //             if(tileRow <= tileCol){

    //                 indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
    //                 if (tileRow == tileCol) scheme->diagonal_bmapper[indexed_tile] = true;
    //                 scheme->nnz_tile_counter[indexed_tile]++;

    //             }else{

    //                 sTiles::Logger::error("build_tile_lookup_serial: unexpected lower-triangular entry (row=",
    //                                       row, ", col=", col, ") while ordering=0.");
    //                 std::exit(EXIT_FAILURE);
    //             }

    //             int withinTileRow = row % (*call_info)->tile_size;
    //             int withinTileCol = col % (*call_info)->tile_size;

    //             // Leading dimension depends only on the destination tile's row index
    //             const int dest_row_tile = tileRow; // tileRow <= tileCol in this branch
    //             const int tile_ld = (dest_row_tile == num_tiles - 1)
    //                                 ? scheme->remainderTileSize
    //                                 : scheme->tile_size;

    //             scheme->tile_index_lookup[index] = indexed_tile;
    //             scheme->withinTileRow[index] = withinTileRow;
    //             scheme->withinTileCol[index] = withinTileCol;

    //         }

    //         //-----------------------------------------------------------------------------------------------------------------------------------------------------
    //         //-----------------------------------------------------------------------------------------------------------------------------------------------------
       
    //     }else if(scheme->use_ordering==1){

    //         //-----------------------------------------------------------------------------------------------------------------------------------------------------
    //         //-----------------------------------------------------------------------------------------------------------------------------------------------------
            
    //         int row, col, tileRow, tileCol;

    //         for (int index = 0; index < scheme->original_nnz; index++) { 
            
    //             if(scheme->element_perm[(*call_info)->col_indices[index]] < scheme->element_perm[(*call_info)->row_indices[index]]){
    //                 col = scheme->element_perm[(*call_info)->row_indices[index]];
    //                 row = scheme->element_perm[(*call_info)->col_indices[index]]; 

    //             }else{
    //                 row = scheme->element_perm[(*call_info)->row_indices[index]];
    //                 col = scheme->element_perm[(*call_info)->col_indices[index]]; 

    //             }

    //             tileRow = row / scheme->tile_size;
    //             tileCol = col / scheme->tile_size;

    //             if(tileRow <= tileCol){
                    
    //                 indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
    //                 withinTileRow = row % (*call_info)->tile_size;
    //                 withinTileCol = col % (*call_info)->tile_size;
    //                 scheme->nnz_tile_counter[indexed_tile]++;

    //                 if (tileRow == tileCol) {
    //                     scheme->diagonal_bmapper[indexed_tile] = true;
    //                 }

    //                 if(tileRow == tileCol){

    //                     if(withinTileRow <= withinTileCol){
                            
    //                         scheme->tile_index_lookup[index] = indexed_tile;
    //                         const int dest_row_tile = tileRow; // diagonal/upper, use tileRow
    //                         const int tile_ld = (dest_row_tile == num_tiles - 1)
    //                                           ? scheme->remainderTileSize
    //                                           : scheme->tile_size;
    //                         scheme->withinTileRow[index] = withinTileCol;
    //                         scheme->withinTileCol[index] = withinTileRow;                                

    //                     }else {
                            
    //                         scheme->tile_index_lookup[index] = indexed_tile;
    //                         const int dest_row_tile = tileRow; // diagonal/upper, use tileRow
    //                         const int tile_ld = (dest_row_tile == num_tiles - 1)
    //                                           ? scheme->remainderTileSize
    //                                           : scheme->tile_size;

    //                         scheme->withinTileRow[index] = withinTileRow;
    //                         scheme->withinTileCol[index] = withinTileCol;
    //                     }
                        
    //                 }else{

    //                     scheme->tile_index_lookup[index] = indexed_tile;
    //                     const int dest_row_tile = tileRow; // upper
    //                     const int tile_ld = (dest_row_tile == num_tiles - 1)
    //                                       ? scheme->remainderTileSize
    //                                       : scheme->tile_size;

    //                     scheme->withinTileRow[index] = withinTileRow;
    //                     scheme->withinTileCol[index] = withinTileCol;                            

    //                 }

    //             }else{

    //                 indexed_tile = scheme->mapper.map_ij(tileCol, tileRow, num_tiles);
    //                 withinTileRow = col % (*call_info)->tile_size;
    //                 withinTileCol = row % (*call_info)->tile_size;
    //                 scheme->nnz_tile_counter[indexed_tile]++;

    //                 scheme->tile_index_lookup[index] = indexed_tile;
    //                 const int dest_row_tile = tileCol;  // swapped: destination row tile is tileCol
    //                 const int tile_ld = (dest_row_tile == num_tiles - 1)
    //                                   ? scheme->remainderTileSize
    //                                   : scheme->tile_size;

    //                 scheme->withinTileRow[index] = withinTileRow;
    //                 scheme->withinTileCol[index] = withinTileCol;

    //             }

    //         }

    //         //-----------------------------------------------------------------------------------------------------------------------------------------------------
    //         //-----------------------------------------------------------------------------------------------------------------------------------------------------
            
    //     }
    // }

namespace sTiles{ namespace process{


    // Force-populate dense tiles from x regardless of SmartTile presence
    inline void update_x_sparse_dense_tiles_force_dense(int global_index, TiledMatrix **scheme, double *x, bool nested) {

        if (!scheme || !scheme[global_index] || !x) {
            return;
        }

        TiledMatrix* S = scheme[global_index];
        if (!S->denseTiles || !S->tileMetaCore) {
            return;
        }
        const int nnz = S->original_nnz;
        const int cores = S->num_cores;
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

        #ifdef _OPENMP
        if (do_parallel) {
            #pragma omp parallel num_threads(cores)
            {
                #pragma omp for schedule(static)
                for (int t = 0; t < S->numActiveTiles; ++t) {
                    double* buf = S->denseTiles[t];
                    if (!buf) continue;
                    const TileMetaCore& m = S->tileMetaCore[t];
                    const int h = (m.height > 0) ? m.height : S->tile_size;
                    const int w = (m.width  > 0) ? m.width  : S->tile_size;
                    LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, buf, h);
                }

                #pragma omp for schedule(static)
                for (int idx = 0; idx < nnz; ++idx) {
                    const int tile = S->tile_index_lookup[idx];
                    if (tile < 0 || tile >= S->numActiveTiles) continue;
                    double* buf = S->denseTiles[tile];
                    if (!buf) continue;
                    const int off  = S->element_offset_lookup[idx];
                    buf[off] = x[idx];
                }
            }
            return;
        }
        #endif

        for (int t = 0; t < S->numActiveTiles; ++t) {
            double* buf = S->denseTiles[t];
            if (!buf) continue;
            const TileMetaCore& m = S->tileMetaCore[t];
            const int h = (m.height > 0) ? m.height : S->tile_size;
            const int w = (m.width  > 0) ? m.width  : S->tile_size;
            LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', h, w, 0.0, 0.0, buf, h);
        }

        for (int idx = 0; idx < nnz; ++idx) {
            const int tile = S->tile_index_lookup[idx];
            if (tile < 0 || tile >= S->numActiveTiles) continue;
            double* buf = S->denseTiles[tile];
            if (!buf) continue;
            const int off  = S->element_offset_lookup[idx];
            buf[off] = x[idx];
        }
    }

#ifdef SMART_TILES
    inline void update_x_sparse_sparse_tiles(int global_index, TiledMatrix **scheme, double *x, bool nested) {

        if (!scheme || !scheme[global_index] || !x) return;

        TiledMatrix* S = scheme[global_index];
        const int nnz = S->original_nnz;
        const int cores = S->num_cores;

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

        // SmartTiles should be present for sparse-sparse path
        if (!S->facTiles) return;

        #ifdef _OPENMP
        if (do_parallel) {
            #pragma omp parallel num_threads(cores)
            {
                #pragma omp for schedule(static)
                for (int t = 0; t < S->numActiveTiles; ++t) {
                    if (S->facTiles[t]) {
                        S->facTiles[t]->populateValues(x);
                    }
                }
            }
            return;
        }
        #endif

        for (int t = 0; t < S->numActiveTiles; ++t) {
            if (S->facTiles[t]) {
                S->facTiles[t]->populateValues(x);
            }
        }
    }
#endif // SMART_TILES

}}


// Fast-mode element accessor for selected inverse (sparse-dense variant).
// Uses TileIndexer mapper and tile metadata instead of safemode arrays.
inline double sTiles_get_selinv_elm_fast_dense_wrapper(int global_index,
                                                      int irow,
                                                      int icol,
                                                      TiledMatrix **scheme)
{
    TiledMatrix* S = scheme[global_index];
    if (!S) return 0.0;

    // Require inverse tiles and a valid mapper in fast mode
    if (!S->inverseTiles || !S->tileMetaCore || !S->mapper.valid()) {
        // Fallback: not available in fast mode; behave safely
        return 0.0;
    }

    // Normalize to upper triangle element (row <= col) in element space
    int row = irow;
    int col = icol;
    if (row > col) std::swap(row, col);

    // Apply element permutation if ordering is active.
    // In the swap branch, capture both permuted values into a tmp before
    // assigning — otherwise `col` is overwritten before `row` is read,
    // producing `element_perm[element_perm[old_row]]` (garbage) and making
    // every off-diagonal accessor read the wrong tile/cell.
    if (S->use_ordering == 1 && S->element_perm) {
        if (S->element_perm[col] < S->element_perm[row]) {
            int tmp = S->element_perm[row];
            row = S->element_perm[col];
            col = tmp;
        } else {
            row = S->element_perm[row];
            col = S->element_perm[col];
        }
    }

    const int ts = S->tile_size;
    const int tileRow = row / ts;
    const int tileCol = col / ts;

    int dense_idx = -1;
    int withinRow = 0;
    int withinCol = 0;

    if (tileRow <= tileCol) {
        dense_idx = S->mapper.map_ij(tileRow, tileCol, S->dimTiledMatrix);
        withinRow = row % ts;
        withinCol = col % ts;
    } else {
        // lower triangle: map to symmetric upper tile
        dense_idx = S->mapper.map_ij(tileCol, tileRow, S->dimTiledMatrix);
        withinRow = col % ts;
        withinCol = row % ts;
    }

    if (dense_idx < 0 || dense_idx >= S->numActiveTiles) {
        // Not part of the selected inverse (inactive tile)
        return 0.0;
    }

    const sTiles::TileMetaCore& m = S->tileMetaCore[dense_idx];
    const int h = (m.height > 0) ? m.height : S->tile_size;
    // Guard inverseTiles pointer per tile
    double* inv = S->inverseTiles[dense_idx];
    if (!inv) return 0.0;

    const int off = withinRow + (withinCol * h);
    return inv[off];
}



// void verify_smart_vs_dense_tiles(const TiledMatrix *scheme, double tolerance = 1e-9) {

//     if (!scheme) {
//         std::cerr << "Error: TiledMatrix scheme is null." << std::endl;
//         return;
//     }
//     if (!scheme->facTiles || !scheme->dense_tiles) {
//         std::cerr << "Error: One or both tile arrays (facTiles, dense_tiles) are not initialized." << std::endl;
//         return;
//     }

//     std::cout << "\n===== VERIFICATION: SMART vs. DENSE TILES =====" << std::endl;
//     int mismatch_count = 0;
//     const int max_mismatches_to_print = 5; // To avoid flooding the console

//     for (int t = 0; t < scheme->numActiveTiles; ++t) {
//         const sTiles::SmartTile* smart_tile = scheme->facTiles[t];
//         const auto& dense_tile = scheme->dense_tiles[t];

//         if (!smart_tile && !dense_tile.elements) {
//             continue; 
//         }
//         if (!smart_tile) {
//             if (mismatch_count < max_mismatches_to_print) {
//                 std::cerr << "MISMATCH in Tile " << t << ": SmartTile is null, but DenseTileSafeMode exists." << std::endl;
//             }
//             mismatch_count++;
//             continue;
//         }
//         if (!dense_tile.elements) {
//             if (mismatch_count < max_mismatches_to_print) {
//                 std::cerr << "MISMATCH in Tile " << t << ": DenseTileSafeMode is null, but SmartTile exists." << std::endl;
//             }
//             mismatch_count++;
//             continue;
//         }

//         bool are_equal = false;
//         try {
//             are_equal = smart_tile->isEqualToDense(dense_tile.elements, dense_tile.height, tolerance);
//         } catch (const std::runtime_error& e) {
//             if (mismatch_count < max_mismatches_to_print) {
//                 std::cerr << "ERROR in Tile " << t << ": Comparison failed! Details: " << e.what() << std::endl;
//             }
//             are_equal = false;
//         }

//         if (!are_equal) {
//             mismatch_count++;
//             // The isEqualToDense function already prints details. We add a summary here.
//             if (mismatch_count <= max_mismatches_to_print) {
//                  std::cout << "--> Mismatch confirmed for Tile " << t << "." << std::endl;
//                  if (mismatch_count == max_mismatches_to_print) {
//                      std::cout << "--> (Further mismatch details will be suppressed)" << std::endl;
//                  }
//             }
//         }
//     }

//     std::cout << "\n----------------- SUMMARY -----------------" << std::endl;
//     if (mismatch_count == 0) {
//         std::cout << "SUCCESS: All " << scheme->numActiveTiles << " active tiles match." << std::endl;
//     } else {
//         std::cout << "FAILURE: Found " << mismatch_count << " mismatch(es) out of "
//                   << scheme->numActiveTiles << " tiles." << std::endl;
//     }
//     std::cout << "===========================================\n" << std::endl;
// }

// void sTiles_update_x_sparse_dense_tiles_phase_1(int global_index, TiledMatrix **scheme, double *x) {

//     int nested = (omp_get_max_active_levels() > 1);
//     if(nested && scheme[global_index]->num_cores > 1){

//         #pragma omp parallel num_threads(scheme[global_index]->num_cores)
//         {
//             #pragma omp for
//             for (int ind = 0; ind < scheme[global_index]->numActiveTiles; ind++) {
//                 LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', scheme[global_index]->dense_tiles[ind].height, scheme[global_index]->dense_tiles[ind].width, 0.0, 0.0, scheme[global_index]->dense_tiles[ind].elements, scheme[global_index]->dense_tiles[ind].height);
//             }
    
//             #pragma omp for
//             for (int index = 0; index < scheme[global_index]->original_nnz; index++) {
//                 scheme[global_index]->dense_tiles[scheme[global_index]->t_indicies[index]].elements[scheme[global_index]->e_indicies[index]] = x[index]; 
//             }

//             #pragma omp for
//             for (int t = 0; t < scheme[global_index]->numActiveTiles; ++t) {
//                 if (scheme[global_index]->facTiles[t]) {
//                     scheme[global_index]->facTiles[t]->populateValues(x);
//                 }
//             }

//         }

//     }else{

//         for (int ind = 0; ind < scheme[global_index]->numActiveTiles; ind++) {
//             LAPACKE_dlaset_work(LAPACK_COL_MAJOR, 'A', scheme[global_index]->dense_tiles[ind].height, scheme[global_index]->dense_tiles[ind].width, 0.0, 0.0, scheme[global_index]->dense_tiles[ind].elements, scheme[global_index]->dense_tiles[ind].height);
//         }

//         for (int index = 0; index < scheme[global_index]->original_nnz; index++) {
//             scheme[global_index]->dense_tiles[scheme[global_index]->t_indicies[index]].elements[scheme[global_index]->e_indicies[index]] = x[index]; 
//         }

//         for (int t = 0; t < scheme[global_index]->numActiveTiles; ++t) {
//             if (scheme[global_index]->facTiles[t]) {
//                 scheme[global_index]->facTiles[t]->populateValues(x);
//                 //scheme[global_index]->facTiles[t]->printAsDense();

//             }
//         }

//     }

//     #ifdef STILES_GPU
//         INTERNAL_UPDATE_X_APPROACH_2_PHASE_1_CPU_TO_GPU_serial(global_index, scheme);
//         //if(omp_get_nested()) INTERNAL_UPDATE_X_APPROACH_2_PHASE_1_GPU_TO_CPU_parallel(global_index, scheme);
//     #endif

//     verify_smart_vs_dense_tiles(scheme[global_index]);

// }

// void sTiles_update_x_sparse_sparse_tiles_phase_0(int global_index, TiledMatrix **scheme, double *x){

// }

// void sTiles_update_x_sparse_sparse_tiles_phase_1(int global_index, TiledMatrix **scheme, double *x){

// }

// double sTiles_get_selinv_elm_sparse_dense_wrapper(int global_index, int irow, int icol, TiledMatrix **scheme){

//     int col, row, tileRow, tileCol, withinTileRow, withinTileCol, indexed_tile;

//     if(scheme[global_index]->use_ordering==0){
        
//         int tmp =0;
//         if(irow > icol){
//             tmp = irow;
//             irow = icol;
//             icol = tmp;
//         }
//         tileRow = irow / scheme[global_index]->tile_size;
//         tileCol = icol / scheme[global_index]->tile_size;

//         if(tileRow <= tileCol){

//             if(!scheme[global_index]->permutation_flags[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol]){

//                 printf("Error: NOT PART OF THE SELECTED INVERSE.\n"); // Optional error message
//                 exit(EXIT_FAILURE);

//             }
            
//             indexed_tile = scheme[global_index]->tileIndexMapper[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol];
//             withinTileRow = irow % scheme[global_index]->tile_size;
//             withinTileCol = icol % scheme[global_index]->tile_size;
            
//             return scheme[global_index]->inverse_tiles[indexed_tile].elements[withinTileRow + (withinTileCol*scheme[global_index]->dense_tiles[indexed_tile].height)];                        

//         }
        

//     }else if(scheme[global_index]->use_ordering==1){

//         if(scheme[global_index]->element_perm[icol] < scheme[global_index]->element_perm[irow]){
//             col = scheme[global_index]->element_perm[irow];
//             row = scheme[global_index]->element_perm[icol]; 

//         }else{
//             row = scheme[global_index]->element_perm[irow];
//             col = scheme[global_index]->element_perm[icol]; 

//         }

//         tileRow = row / scheme[global_index]->tile_size;
//         tileCol = col / scheme[global_index]->tile_size;


//         if(tileRow <= tileCol){
            
            
//             if(!scheme[global_index]->permutation_flags[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol]){

//                 printf("Error: NOT PART OF THE SELECTED INVERSE.\n"); // Optional error message
//                 exit(EXIT_FAILURE);

//             }
            
//             indexed_tile = scheme[global_index]->tileIndexMapper[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol];
//             withinTileRow = row % scheme[global_index]->tile_size;
//             withinTileCol = col % scheme[global_index]->tile_size;

//             return scheme[global_index]->inverse_tiles[indexed_tile].elements[withinTileRow + (withinTileCol*scheme[global_index]->dense_tiles[indexed_tile].height)];                        

//         }else{

//             if(!scheme[global_index]->permutation_flags[tileCol*(2*scheme[global_index]->dimTiledMatrix-tileCol-1)/2 + tileRow]){

//                 printf("Error: NOT PART OF THE SELECTED INVERSE.\n"); // Optional error message
//                 exit(EXIT_FAILURE);

//             }

//             indexed_tile = scheme[global_index]->tileIndexMapper[tileCol*(2*scheme[global_index]->dimTiledMatrix-tileCol-1)/2 + tileRow];
//             withinTileRow = col % scheme[global_index]->tile_size;
//             withinTileCol = row % scheme[global_index]->tile_size;
//             return scheme[global_index]->inverse_tiles[indexed_tile].elements[withinTileRow + (withinTileCol*scheme[global_index]->dense_tiles[indexed_tile].height)];                        
            
//         }

//     }
    
//     return 0;

// }

// double sTiles_get_chol_elm(int group_index, int call_index, int irow, int icol, TiledMatrix **scheme){

//     int global_index =  scheme[0]->call_lookup_table[group_index][call_index];
//     int col, row, tileRow, tileCol, withinTileRow, withinTileCol, indexed_tile;


//     if(scheme[global_index]->use_ordering==0){
        
//         int tmp =0;
//         if(irow > icol){
//             tmp = irow;
//             irow = icol;
//             icol = tmp;
//         }
//         tileRow = irow / scheme[global_index]->tile_size;
//         tileCol = icol / scheme[global_index]->tile_size;

//         if(tileRow <= tileCol){

//             if(!scheme[global_index]->permutation_flags[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol]){

//                 printf("Error: NOT PART OF THE SELECTED INVERSE.\n"); // Optional error message
//                 exit(EXIT_FAILURE);

//             }
            
//             indexed_tile = scheme[global_index]->tileIndexMapper[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol];
//             withinTileRow = irow % scheme[global_index]->tile_size;
//             withinTileCol = icol % scheme[global_index]->tile_size;
            
//             return scheme[global_index]->dense_tiles[indexed_tile].elements[withinTileRow + (withinTileCol*scheme[global_index]->dense_tiles[indexed_tile].height)];                        

//         }
        

//     }else if(scheme[global_index]->use_ordering==1){

//         if(scheme[global_index]->element_perm[icol] < scheme[global_index]->element_perm[irow]){
//             col = scheme[global_index]->element_perm[irow];
//             row = scheme[global_index]->element_perm[icol]; 

//         }else{
//             row = scheme[global_index]->element_perm[irow];
//             col = scheme[global_index]->element_perm[icol]; 

//         }

//         tileRow = row / scheme[global_index]->tile_size;
//         tileCol = col / scheme[global_index]->tile_size;


//         if(tileRow <= tileCol){
            
            
//             if(!scheme[global_index]->permutation_flags[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol]){

//                 printf("Error: NOT PART OF THE SELECTED INVERSE.\n"); // Optional error message
//                 exit(EXIT_FAILURE);

//             }
            
//             indexed_tile = scheme[global_index]->tileIndexMapper[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol];
//             withinTileRow = row % scheme[global_index]->tile_size;
//             withinTileCol = col % scheme[global_index]->tile_size;

//             return scheme[global_index]->dense_tiles[indexed_tile].elements[withinTileRow + (withinTileCol*scheme[global_index]->dense_tiles[indexed_tile].height)];                        

//         }else{

//             if(!scheme[global_index]->permutation_flags[tileCol*(2*scheme[global_index]->dimTiledMatrix-tileCol-1)/2 + tileRow]){

//                 printf("Error: NOT PART OF THE SELECTED INVERSE.\n"); // Optional error message
//                 exit(EXIT_FAILURE);

//             }

//             indexed_tile = scheme[global_index]->tileIndexMapper[tileCol*(2*scheme[global_index]->dimTiledMatrix-tileCol-1)/2 + tileRow];
//             withinTileRow = col % scheme[global_index]->tile_size;
//             withinTileCol = row % scheme[global_index]->tile_size;
//             return scheme[global_index]->dense_tiles[indexed_tile].elements[withinTileRow + (withinTileCol*scheme[global_index]->dense_tiles[indexed_tile].height)];                        
            
//         }

//     }
    
//     return 0;

// }


// void verify_smart_vs_dense_tiles(const TiledMatrix *scheme, double tolerance = 1e-9) {

//     if (!scheme) {
//         std::cerr << "Error: TiledMatrix scheme is null." << std::endl;
//         return;
//     }
//     if (!scheme->facTiles || !scheme->dense_tiles) {
//         std::cerr << "Error: One or both tile arrays (facTiles, dense_tiles) are not initialized." << std::endl;
//         return;
//     }

//     std::cout << "\n===== VERIFICATION: SMART vs. DENSE TILES =====" << std::endl;
//     int mismatch_count = 0;
//     const int max_mismatches_to_print = 5; // To avoid flooding the console

//     for (int t = 0; t < scheme->numActiveTiles; ++t) {
//         const sTiles::SmartTile* smart_tile = scheme->facTiles[t];
//         const auto& dense_tile = scheme->dense_tiles[t];

//         if (!smart_tile && !dense_tile.elements) {
//             continue; 
//         }
//         if (!smart_tile) {
//             if (mismatch_count < max_mismatches_to_print) {
//                 std::cerr << "MISMATCH in Tile " << t << ": SmartTile is null, but DenseTileSafeMode exists." << std::endl;
//             }
//             mismatch_count++;
//             continue;
//         }
//         if (!dense_tile.elements) {
//             if (mismatch_count < max_mismatches_to_print) {
//                 std::cerr << "MISMATCH in Tile " << t << ": DenseTileSafeMode is null, but SmartTile exists." << std::endl;
//             }
//             mismatch_count++;
//             continue;
//         }

//         bool are_equal = false;
//         try {
//             are_equal = smart_tile->isEqualToDense(dense_tile.elements, dense_tile.height, tolerance);
//         } catch (const std::runtime_error& e) {
//             if (mismatch_count < max_mismatches_to_print) {
//                 std::cerr << "ERROR in Tile " << t << ": Comparison failed! Details: " << e.what() << std::endl;
//             }
//             are_equal = false;
//         }

//         if (!are_equal) {
//             mismatch_count++;
//             // The isEqualToDense function already prints details. We add a summary here.
//             if (mismatch_count <= max_mismatches_to_print) {
//                  std::cout << "--> Mismatch confirmed for Tile " << t << "." << std::endl;
//                  if (mismatch_count == max_mismatches_to_print) {
//                      std::cout << "--> (Further mismatch details will be suppressed)" << std::endl;
//                  }
//             }
//         }
//     }

//     std::cout << "\n----------------- SUMMARY -----------------" << std::endl;
//     if (mismatch_count == 0) {
//         std::cout << "SUCCESS: All " << scheme->numActiveTiles << " active tiles match." << std::endl;
//     } else {
//         std::cout << "FAILURE: Found " << mismatch_count << " mismatch(es) out of "
//                   << scheme->numActiveTiles << " tiles." << std::endl;
//     }
//     std::cout << "===========================================\n" << std::endl;
// }

// double sTiles_get_selinv_elm_sparse_dense_wrapper(int global_index, int irow, int icol, TiledMatrix **scheme){

//     int col, row, tileRow, tileCol, withinTileRow, withinTileCol, indexed_tile;

//     if(scheme[global_index]->use_ordering==0){
        
//         int tmp =0;
//         if(irow > icol){
//             tmp = irow;
//             irow = icol;
//             icol = tmp;
//         }
//         tileRow = irow / scheme[global_index]->tile_size;
//         tileCol = icol / scheme[global_index]->tile_size;

//         if(tileRow <= tileCol){

//             if(!scheme[global_index]->permutation_flags[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol]){

//                 printf("Error: NOT PART OF THE SELECTED INVERSE.\n"); // Optional error message
//                 exit(EXIT_FAILURE);

//             }
            
//             indexed_tile = scheme[global_index]->tileIndexMapper[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol];
//             withinTileRow = irow % scheme[global_index]->tile_size;
//             withinTileCol = icol % scheme[global_index]->tile_size;
            
//             return scheme[global_index]->inverse_tiles[indexed_tile].elements[withinTileRow + (withinTileCol*scheme[global_index]->dense_tiles[indexed_tile].height)];                        

//         }
        

//     }else if(scheme[global_index]->use_ordering==1){

//         if(scheme[global_index]->element_perm[icol] < scheme[global_index]->element_perm[irow]){
//             col = scheme[global_index]->element_perm[irow];
//             row = scheme[global_index]->element_perm[icol]; 

//         }else{
//             row = scheme[global_index]->element_perm[irow];
//             col = scheme[global_index]->element_perm[icol]; 

//         }

//         tileRow = row / scheme[global_index]->tile_size;
//         tileCol = col / scheme[global_index]->tile_size;


//         if(tileRow <= tileCol){
            
            
//             if(!scheme[global_index]->permutation_flags[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol]){

//                 printf("Error: NOT PART OF THE SELECTED INVERSE.\n"); // Optional error message
//                 exit(EXIT_FAILURE);

//             }
            
//             indexed_tile = scheme[global_index]->tileIndexMapper[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol];
//             withinTileRow = row % scheme[global_index]->tile_size;
//             withinTileCol = col % scheme[global_index]->tile_size;

//             return scheme[global_index]->inverse_tiles[indexed_tile].elements[withinTileRow + (withinTileCol*scheme[global_index]->dense_tiles[indexed_tile].height)];                        

//         }else{

//             if(!scheme[global_index]->permutation_flags[tileCol*(2*scheme[global_index]->dimTiledMatrix-tileCol-1)/2 + tileRow]){

//                 printf("Error: NOT PART OF THE SELECTED INVERSE.\n"); // Optional error message
//                 exit(EXIT_FAILURE);

//             }

//             indexed_tile = scheme[global_index]->tileIndexMapper[tileCol*(2*scheme[global_index]->dimTiledMatrix-tileCol-1)/2 + tileRow];
//             withinTileRow = col % scheme[global_index]->tile_size;
//             withinTileCol = row % scheme[global_index]->tile_size;
//             return scheme[global_index]->inverse_tiles[indexed_tile].elements[withinTileRow + (withinTileCol*scheme[global_index]->dense_tiles[indexed_tile].height)];                        
            
//         }

//     }
    
//     return 0;

// }

// double sTiles_get_chol_elm(int group_index, int call_index, int irow, int icol, TiledMatrix **scheme){

//     int global_index =  scheme[0]->call_lookup_table[group_index][call_index];
//     int col, row, tileRow, tileCol, withinTileRow, withinTileCol, indexed_tile;


//     if(scheme[global_index]->use_ordering==0){
        
//         int tmp =0;
//         if(irow > icol){
//             tmp = irow;
//             irow = icol;
//             icol = tmp;
//         }
//         tileRow = irow / scheme[global_index]->tile_size;
//         tileCol = icol / scheme[global_index]->tile_size;

//         if(tileRow <= tileCol){

//             if(!scheme[global_index]->permutation_flags[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol]){

//                 printf("Error: NOT PART OF THE SELECTED INVERSE.\n"); // Optional error message
//                 exit(EXIT_FAILURE);

//             }
            
//             indexed_tile = scheme[global_index]->tileIndexMapper[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol];
//             withinTileRow = irow % scheme[global_index]->tile_size;
//             withinTileCol = icol % scheme[global_index]->tile_size;
            
//             return scheme[global_index]->dense_tiles[indexed_tile].elements[withinTileRow + (withinTileCol*scheme[global_index]->dense_tiles[indexed_tile].height)];                        

//         }
        

//     }else if(scheme[global_index]->use_ordering==1){

//         if(scheme[global_index]->element_perm[icol] < scheme[global_index]->element_perm[irow]){
//             col = scheme[global_index]->element_perm[irow];
//             row = scheme[global_index]->element_perm[icol]; 

//         }else{
//             row = scheme[global_index]->element_perm[irow];
//             col = scheme[global_index]->element_perm[icol]; 

//         }

//         tileRow = row / scheme[global_index]->tile_size;
//         tileCol = col / scheme[global_index]->tile_size;


//         if(tileRow <= tileCol){
            
            
//             if(!scheme[global_index]->permutation_flags[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol]){

//                 printf("Error: NOT PART OF THE SELECTED INVERSE.\n"); // Optional error message
//                 exit(EXIT_FAILURE);

//             }
            
//             indexed_tile = scheme[global_index]->tileIndexMapper[tileRow*(2*scheme[global_index]->dimTiledMatrix-tileRow-1)/2 + tileCol];
//             withinTileRow = row % scheme[global_index]->tile_size;
//             withinTileCol = col % scheme[global_index]->tile_size;

//             return scheme[global_index]->dense_tiles[indexed_tile].elements[withinTileRow + (withinTileCol*scheme[global_index]->dense_tiles[indexed_tile].height)];                        

//         }else{

//             if(!scheme[global_index]->permutation_flags[tileCol*(2*scheme[global_index]->dimTiledMatrix-tileCol-1)/2 + tileRow]){

//                 printf("Error: NOT PART OF THE SELECTED INVERSE.\n"); // Optional error message
//                 exit(EXIT_FAILURE);

//             }

//             indexed_tile = scheme[global_index]->tileIndexMapper[tileCol*(2*scheme[global_index]->dimTiledMatrix-tileCol-1)/2 + tileRow];
//             withinTileRow = col % scheme[global_index]->tile_size;
//             withinTileCol = row % scheme[global_index]->tile_size;
//             return scheme[global_index]->dense_tiles[indexed_tile].elements[withinTileRow + (withinTileCol*scheme[global_index]->dense_tiles[indexed_tile].height)];                        
            
//         }

//     }
    
//     return 0;

// }

// Implementation of sTiles::preprocess functions
