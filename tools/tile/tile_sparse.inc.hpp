
    static inline void build_sparse_tile_lookup_init_serial(sTiles_call **call_info, TiledMatrix *scheme, int group_index) {
        int indexed_tile;

        const int num_tiles = scheme->dimTiledMatrix;
        const int num_tiles_active = scheme->numActiveTiles;
        const int original_nnz = scheme->original_nnz;

        scheme->tile_index_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->withinTileRow = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->withinTileCol = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->diagonal_bmapper = TileMemoryManager::allocateZero<bool>(num_tiles_active, group_index);
        scheme->nnz_tile_counter = TileMemoryManager::allocateZero<int>(num_tiles_active, group_index);

        if (scheme->use_ordering == 0) {
            for (int index = 0; index < original_nnz; ++index) {
                const int col = (*call_info)->row_indices[index];
                const int row = (*call_info)->col_indices[index];

                const int tileRow = row / (*call_info)->tile_size;
                const int tileCol = col / (*call_info)->tile_size;

                if (tileRow <= tileCol) {
                    indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
                    if (tileRow == tileCol) scheme->diagonal_bmapper[indexed_tile] = true;
                    scheme->nnz_tile_counter[indexed_tile]++;
                } else {
                    sTiles::Logger::error("build_sparse_tile_lookup_init_serial: unexpected lower-triangular entry (row=", row, ", col=", col, ") while ordering=0.");
                    std::exit(EXIT_FAILURE);
                }

                const int withinTileRow = row % (*call_info)->tile_size;
                const int withinTileCol = col % (*call_info)->tile_size;

                scheme->tile_index_lookup[index] = indexed_tile;
                scheme->withinTileRow[index] = withinTileRow;
                scheme->withinTileCol[index] = withinTileCol;
            }
        } else if (scheme->use_ordering == 1) {
            int row, col, tileRow, tileCol, withinTileRow, withinTileCol;

            for (int index = 0; index < original_nnz; ++index) {
                if (scheme->element_perm[(*call_info)->col_indices[index]] < scheme->element_perm[(*call_info)->row_indices[index]]) {
                    col = scheme->element_perm[(*call_info)->row_indices[index]];
                    row = scheme->element_perm[(*call_info)->col_indices[index]];
                } else {
                    row = scheme->element_perm[(*call_info)->row_indices[index]];
                    col = scheme->element_perm[(*call_info)->col_indices[index]];
                }

                tileRow = row / scheme->tile_size;
                tileCol = col / scheme->tile_size;

                if (tileRow <= tileCol) {
                    indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
                    withinTileRow = row % (*call_info)->tile_size;
                    withinTileCol = col % (*call_info)->tile_size;
                    scheme->nnz_tile_counter[indexed_tile]++;

                    if (tileRow == tileCol) {
                        scheme->diagonal_bmapper[indexed_tile] = true;
                    }

                    scheme->tile_index_lookup[index] = indexed_tile;
                    scheme->withinTileRow[index] = withinTileRow;
                    scheme->withinTileCol[index] = withinTileCol;
                } else {
                    indexed_tile = scheme->mapper.map_ij(tileCol, tileRow, num_tiles);
                    withinTileRow = col % (*call_info)->tile_size;
                    withinTileCol = row % (*call_info)->tile_size;
                    scheme->nnz_tile_counter[indexed_tile]++;

                    scheme->tile_index_lookup[index] = indexed_tile;
                    scheme->withinTileRow[index] = withinTileRow;
                    scheme->withinTileCol[index] = withinTileCol;
                }
            }
        }
    }

    static inline void build_sparse_tile_lookup_init_parallel(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {
        const int original_nnz = scheme->original_nnz;
        const int num_tiles = scheme->dimTiledMatrix;
        const int num_tiles_active = scheme->numActiveTiles;

        const int threads = (num_cores > 0) ? num_cores : omp_get_max_threads();
        const bool nested_enabled = (omp_get_max_active_levels() > 1);
        const bool can_spawn = (!omp_in_parallel()) || nested_enabled;

        if (!can_spawn || threads < 2) {
            build_sparse_tile_lookup_init_serial(call_info, scheme, group_index);
            return;
        }

        scheme->tile_index_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->withinTileRow = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->withinTileCol = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->diagonal_bmapper = TileMemoryManager::allocateZero<bool>(num_tiles_active, group_index);
        scheme->nnz_tile_counter = TileMemoryManager::allocateZero<int>(num_tiles_active, group_index);

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
                #pragma omp atomic
                scheme->nnz_tile_counter[indexed_tile]++;

                scheme->tile_index_lookup[index] = indexed_tile;
                scheme->withinTileRow[index] = row % (*call_info)->tile_size;
                scheme->withinTileCol[index] = col % (*call_info)->tile_size;
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
                    }
                    #pragma omp atomic
                    scheme->nnz_tile_counter[indexed_tile]++;

                    scheme->tile_index_lookup[index] = indexed_tile;
                    scheme->withinTileRow[index] = withinTileRow;
                    scheme->withinTileCol[index] = withinTileCol;
                } else {
                    const int indexed_tile = scheme->mapper.map_ij(tileCol, tileRow, num_tiles);
                    const int withinTileRow = col % (*call_info)->tile_size;
                    const int withinTileCol = row % (*call_info)->tile_size;

                    #pragma omp atomic
                    scheme->nnz_tile_counter[indexed_tile]++;

                    scheme->tile_index_lookup[index] = indexed_tile;
                    scheme->withinTileRow[index] = withinTileRow;
                    scheme->withinTileCol[index] = withinTileCol;
                }
            }
        }

        if (error_flag.load(std::memory_order_relaxed)) {
            sTiles::Logger::error("build_sparse_tile_lookup_init_parallel: unexpected lower-triangular entry encountered (parallel path).");
            std::exit(EXIT_FAILURE);
        }
    }

    static inline void build_sparse_tile_lookup_phase1(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {
        const int n = scheme->original_order;
        scheme->remainderTileSize = (n % scheme->tile_size == 0) ? scheme->tile_size : (n % scheme->tile_size);
        if (num_cores < 2 || scheme->original_nnz < 5000) { build_sparse_tile_lookup_init_serial(call_info, scheme, group_index); return; }
    #ifndef _OPENMP
        build_sparse_tile_lookup_init_serial(call_info, scheme, group_index); return;
    #else
        const bool nested_enabled = (omp_get_max_active_levels() > 1);
        const bool can_spawn = (!omp_in_parallel()) || nested_enabled;
        if (!can_spawn) { build_sparse_tile_lookup_init_serial(call_info, scheme, group_index); return; }
        build_sparse_tile_lookup_init_parallel(call_info, scheme, group_index, num_cores);
    #endif
    }

    static inline void build_sparse_tile_lookup_serial(sTiles_call **call_info, TiledMatrix *scheme, int group_index) {
        if (!call_info || !*call_info || !scheme) {
            sTiles::Logger::error("[SmartTileLookup] Invalid input to build_sparse_tile_lookup_serial.");
            return;
        }

        const int num_tiles = scheme->dimTiledMatrix;
        const int num_tiles_active = scheme->numActiveTiles;
        const int base_tile_size = scheme->tile_size;

        if (num_tiles <= 0 || num_tiles_active <= 0) {
            sTiles::Logger::error("[SmartTileLookup] Invalid tiling configuration. tiles=", num_tiles, " active=", num_tiles_active);
            return;
        }

        if (base_tile_size <= 0) {
            sTiles::Logger::error("[SmartTileLookup] Invalid tile size ", base_tile_size);
            return;
        }

        const int original_nnz = scheme->original_nnz;

        if (!scheme->nnz_tile_counter || !scheme->withinTileRow || !scheme->withinTileCol || !scheme->tile_index_lookup) {
            sTiles::Logger::error("[SmartTileLookup] Missing within-tile metadata; ensure build_tile_lookup ran for variant 3.");
            return;
        }

        if (!scheme->sparseTileMetaData) {
            scheme->sparseTileMetaData = TileMemoryManager::allocate<SparseTileMetaData>(num_tiles_active, group_index);
            for (int t = 0; t < num_tiles_active; ++t) {
                new (&scheme->sparseTileMetaData[t]) SparseTileMetaData();
            }
        }

        SparseTileMetaCore *core = scheme->sparseTileMetaCore;
        SparseTileMetaData *data = scheme->sparseTileMetaData;
        TileMetaCore *dense_meta = scheme->tileMetaCore;

        if (!core || !data || !dense_meta) {
            sTiles::Logger::error("[SmartTileLookup] Missing sparse tile core or data arrays.");
            return;
        }

        scheme->remainderTileSize = (scheme->original_order % scheme->tile_size == 0) ? scheme->tile_size : scheme->original_order % scheme->tile_size;

        for (int t = 0; t < num_tiles_active; ++t) {
            SparseTileMetaData &dt = data[t];
            SparseTileMetaCore &c = core[t];
            TileMetaCore &dense = dense_meta[t];

            const int expected_nnz = scheme->nnz_tile_counter[t];

            if (dense.width <= 0) dense.width = base_tile_size;
            if (dense.height <= 0) dense.height = dense.width;

            c.upper_bw = 0;
            c.nnz = 0;
            c.colptr.clear();
            c.rowind.clear();
            c.tmp_columns.clear();

            dt.coo_elements_.clear();
            dt.indices_sorted.clear();
            dt.current_index_.store(0, std::memory_order_relaxed);

            if (expected_nnz > 0) {
                dt.coo_elements_.resize(static_cast<std::size_t>(expected_nnz));
            }
        }

        int *dmap = scheme->diagonal_mapper;
        if (dmap) {
            for (int t = 0; t < num_tiles; ++t) {
                int diag_index = dmap[t];
                if (diag_index < 0 || diag_index >= num_tiles_active) {
                    continue;
                }
                SparseTileMetaCore &c = core[diag_index];
                c.upper_bw = 1;  // mark as diagonal tile, kd computed later
            }
        }

        for (int idx = 0; idx < original_nnz; ++idx) {
            const int tile = scheme->tile_index_lookup[idx];
            if (tile < 0 || tile >= num_tiles_active) {
                continue;
            }

            SparseTileMetaCore &c = core[tile];
            SparseTileMetaData &dt = data[tile];

            int local_row = scheme->withinTileRow[idx];
            int local_col = scheme->withinTileCol[idx];

            if (c.upper_bw == 1 && local_row < local_col) {
                int tmp = local_row;
                local_row = local_col;
                local_col = tmp;
            }

            dt.appendElement(local_row, local_col, idx);
        }

        for (int t = 0; t < num_tiles_active; ++t) {
            SparseTileMetaCore& c = core[t];
            SparseTileMetaData& dt = data[t];
            const TileMetaCore& dense = dense_meta[t];
            dt.finalizeConstructionSparse(c, dense.width);
        }

    }

    static inline void build_sparse_tile_lookup_phase2(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {
        if (!call_info || !*call_info || !scheme) {
            return;
        }

    #ifndef _OPENMP
        (void)num_cores;
        build_sparse_tile_lookup_serial(call_info, scheme, group_index);
        return;
    #else
        const int nnz = scheme->original_nnz;
        const bool nested_enabled = (omp_get_max_active_levels() > 1);
        const bool can_spawn = (!omp_in_parallel()) || nested_enabled;

        if (!can_spawn || num_cores <= 1 || nnz <= 5000) {
            build_sparse_tile_lookup_serial(call_info, scheme, group_index);
            return;
        }

        const int num_tiles = scheme->dimTiledMatrix;
        const int num_tiles_active = scheme->numActiveTiles;
        const int base_tile_size = scheme->tile_size;

        if (num_tiles <= 0 || num_tiles_active <= 0 || base_tile_size <= 0 || nnz <= 0) {
            build_sparse_tile_lookup_serial(call_info, scheme, group_index);
            return;
        }

        if (!scheme->nnz_tile_counter || !scheme->withinTileRow || !scheme->withinTileCol || !scheme->tile_index_lookup) {
            sTiles::Logger::error("[SmartTileLookup] Missing sparse lookup metadata; ensure phase1 succeeded.");
            build_sparse_tile_lookup_serial(call_info, scheme, group_index);
            return;
        }

        if (!scheme->sparseTileMetaCore) {
            sTiles::Logger::error("[SmartTileLookup] Missing sparseTileMetaCore array.");
            build_sparse_tile_lookup_serial(call_info, scheme, group_index);
            return;
        }

        if (!scheme->tileMetaCore) {
            sTiles::Logger::error("[SmartTileLookup] Missing tileMetaCore array needed for widths.");
            build_sparse_tile_lookup_serial(call_info, scheme, group_index);
            return;
        }

        if (!scheme->sparseTileMetaData) {
            scheme->sparseTileMetaData = TileMemoryManager::allocate<SparseTileMetaData>(num_tiles_active, group_index);
            for (int t = 0; t < num_tiles_active; ++t) {
                new (&scheme->sparseTileMetaData[t]) SparseTileMetaData();
            }
        }

        SparseTileMetaCore *core = scheme->sparseTileMetaCore;
        SparseTileMetaData *data = scheme->sparseTileMetaData;

        if (!core || !data) {
            build_sparse_tile_lookup_serial(call_info, scheme, group_index);
            return;
        }

        const int threads = (num_cores > 0) ? num_cores : omp_get_max_threads();
        const int original_nnz = nnz;
        TileMetaCore *dense_meta = scheme->tileMetaCore;

        scheme->remainderTileSize = (scheme->original_order % scheme->tile_size == 0) ? scheme->tile_size : scheme->original_order % scheme->tile_size;

        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int t = 0; t < num_tiles_active; ++t) {
            SparseTileMetaCore &c = core[t];
            SparseTileMetaData &dt = data[t];
            TileMetaCore &dense = dense_meta[t];

            const int expected_nnz = scheme->nnz_tile_counter[t];

            if (dense.width <= 0) dense.width = base_tile_size;
            if (dense.height <= 0) dense.height = dense.width;

            c.upper_bw = 0;
            c.nnz = 0;
            c.colptr.clear();
            c.rowind.clear();
            c.tmp_columns.clear();

            dt.coo_elements_.clear();
            dt.indices_sorted.clear();
            dt.current_index_.store(0, std::memory_order_relaxed);

            if (expected_nnz > 0) {
                dt.coo_elements_.resize(static_cast<std::size_t>(expected_nnz));
            }
        }

        int *dmap = scheme->diagonal_mapper;
        if (dmap) {
            for (int t = 0; t < num_tiles; ++t) {
                const int diag_index = dmap[t];
                if (diag_index < 0 || diag_index >= num_tiles_active) {
                    continue;
                }
                core[diag_index].upper_bw = 1;
            }
        }

        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int idx = 0; idx < original_nnz; ++idx) {
            const int tile = scheme->tile_index_lookup[idx];
            if (tile < 0 || tile >= num_tiles_active) {
                continue;
            }

            SparseTileMetaCore &c = core[tile];
            SparseTileMetaData &dt = data[tile];

            int local_row = scheme->withinTileRow[idx];
            int local_col = scheme->withinTileCol[idx];

            if (c.upper_bw == 1 && local_row < local_col) {
                std::swap(local_row, local_col);
            }

            dt.appendElement(local_row, local_col, idx);
        }

        #pragma omp parallel for schedule(dynamic) num_threads(threads)
        for (int t = 0; t < num_tiles_active; ++t) {
            SparseTileMetaCore& c = core[t];
            SparseTileMetaData& dt = data[t];
            const TileMetaCore& dense = dense_meta[t];
            dt.finalizeConstructionSparse(c, dense.width);
        }

    #endif // _OPENMP
    }

    inline StatusCode build_sparse_tile_lookup(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {
        if (!call_info || !*call_info || !scheme) return StatusCode::InvalidArgument;

        const int fact_variant = (*call_info)->factorization_variant;
        // For single-tile (variant 1) or scaled-dense (variant 2), skip sparse tile lookup
        if (fact_variant == 1 || fact_variant == 2) {
            return StatusCode::Success;
        }
        if (fact_variant != 0 && fact_variant != 3) {
            std::fprintf(stderr, "ERROR: Unsupported factorization_variant %d\n", fact_variant);
            return StatusCode::InvalidArgument;
        }

        const int num_active = scheme->numActiveTiles;
        if (num_active < 0) return StatusCode::InvalidArgument;

        scheme->tileMetaCore = nullptr;
        scheme->sparseTileMetaCore = nullptr;
        scheme->sparseTileMetaData = nullptr;
        scheme->invSparseTileMetaCore = nullptr;

        scheme->sparseTileMetaCore = TileMemoryManager::allocate<SparseTileMetaCore>(num_active, group_index);
        scheme->sparseTileMetaData = TileMemoryManager::allocate<SparseTileMetaData>(num_active, group_index);
        if (!scheme->sparseTileMetaCore || !scheme->sparseTileMetaData) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for sparse metadata.\n");
            return StatusCode::OutOfResources;
        }
        // Construct SparseTileMetaCore and SparseTileMetaData objects (contain std::vector members)
        for (int t = 0; t < num_active; ++t) {
            new (&scheme->sparseTileMetaCore[t]) SparseTileMetaCore();
            new (&scheme->sparseTileMetaData[t]) SparseTileMetaData();
        }

        scheme->tileMetaCore = TileMemoryManager::allocate<TileMetaCore>(num_active, group_index);
        if (!scheme->tileMetaCore) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for denseTiles or tileMetaCore.\n");
            return StatusCode::OutOfResources;
        }

        if (scheme->compute_inverse) {
            scheme->invSparseTileMetaCore = TileMemoryManager::allocate<SparseTileMetaCore>(num_active, group_index);
            if (!scheme->invSparseTileMetaCore) {
                std::fprintf(stderr, "ERROR: Memory allocation failed for invSparseTileMetaCore.\n");
                return StatusCode::OutOfResources;
            }
            // Construct invSparseTileMetaCore objects (contain std::vector members)
            for (int t = 0; t < num_active; ++t) {
                new (&scheme->invSparseTileMetaCore[t]) SparseTileMetaCore();
            }
        } 


        set_tile_extents(&scheme);
        build_sparse_tile_lookup_phase1(call_info, scheme, group_index, num_cores);
        build_sparse_tile_lookup_phase2(call_info, scheme, group_index, num_cores);
        
        return StatusCode::Ok;
    }

#ifdef SMART_TILES
    inline StatusCode allocate_sparse_tiles(TiledMatrix *scheme, int group_index, int num_cores) {
        const int num_active = scheme->numActiveTiles;

        scheme->facTiles = nullptr;
        scheme->invTiles = nullptr;
        scheme->tmpTiles = nullptr;
        scheme->rhsTiles = nullptr;

        scheme->facTiles = TileMemoryManager::allocate<SmartTile>(num_active, group_index);
        if (!scheme->facTiles) {
            std::fprintf(stderr, "ERROR: Memory allocation failed for facTiles.\n");
            return StatusCode::OutOfResources;
        }

        //default_construct_smart_tiles(scheme->facTiles, num_active, num_cores);

        if (scheme->compute_inverse) {
            scheme->invTiles = TileMemoryManager::allocate<SmartTile>(num_active, group_index);
            scheme->tmpTiles = TileMemoryManager::allocate<SmartTile>(num_active, group_index);
            if (!scheme->invTiles || !scheme->tmpTiles) {
                std::fprintf(stderr, "ERROR: Memory allocation failed for invTiles/tmpTiles.\n");
                return StatusCode::OutOfResources;
            }

            //default_construct_smart_tiles(scheme->invTiles, num_active, num_cores);
            //default_construct_smart_tiles(scheme->tmpTiles, num_active, num_cores);
        } else {
            scheme->invTiles = nullptr;
            scheme->tmpTiles = nullptr;
        }

        // scheme->rhsTiles = TileMemoryManager::allocate<SmartTile>(num_active, group_index);
        return StatusCode::Ok;
    }
#endif // SMART_TILES


    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /*
        1. tile_index_lookup
        2. diagonal_bmapper
        3. remainderTileSize
        4. element_offset_lookup
    */

   static inline void build_dense_tile_lookup_init_serial(sTiles_call **call_info, TiledMatrix *scheme, int group_index) {

        int k, i, j, indexed_tile, temp_count;

        const int num_tiles = scheme->dimTiledMatrix;
        const int num_tiles_active = scheme->numActiveTiles;
        const int original_nnz = scheme->original_nnz;

        scheme->tile_index_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->element_offset_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->diagonal_bmapper = TileMemoryManager::allocateZero<bool>(num_tiles_active, group_index);

        if(scheme->use_ordering==0){

            //-----------------------------------------------------------------------------------------------------------------------------------------------------
            //-----------------------------------------------------------------------------------------------------------------------------------------------------
            
            for (int index = 0; index < scheme->original_nnz; index++) {
            
                int col = (*call_info)->row_indices[index];
                int row = (*call_info)->col_indices[index];

                int tileRow = row / (*call_info)->tile_size;
                int tileCol = col / (*call_info)->tile_size;

                if(tileRow <= tileCol){

                    indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
                    if (tileRow == tileCol) scheme->diagonal_bmapper[indexed_tile] = true;

                }else{

                    sTiles::Logger::error("build_tile_lookup_serial: unexpected lower-triangular entry (row=",
                                          row, ", col=", col, ") while ordering=0.");
                    std::exit(EXIT_FAILURE);
                }

                int withinTileRow = row % (*call_info)->tile_size;
                int withinTileCol = col % (*call_info)->tile_size;

                // Leading dimension depends only on the destination tile's row index
                const int dest_row_tile = tileRow; // tileRow <= tileCol in this branch
                const int tile_ld = (dest_row_tile == num_tiles - 1)
                                    ? scheme->remainderTileSize
                                    : scheme->tile_size;

                scheme->tile_index_lookup[index] = indexed_tile;
                scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);

            }

            //-----------------------------------------------------------------------------------------------------------------------------------------------------
            //-----------------------------------------------------------------------------------------------------------------------------------------------------
       
        }else if(scheme->use_ordering==1){

            //-----------------------------------------------------------------------------------------------------------------------------------------------------
            //-----------------------------------------------------------------------------------------------------------------------------------------------------
            
            int row, col, tileRow, tileCol;

            for (int index = 0; index < scheme->original_nnz; index++) { 
            
                if(scheme->element_perm[(*call_info)->col_indices[index]] < scheme->element_perm[(*call_info)->row_indices[index]]){
                    col = scheme->element_perm[(*call_info)->row_indices[index]];
                    row = scheme->element_perm[(*call_info)->col_indices[index]]; 

                }else{
                    row = scheme->element_perm[(*call_info)->row_indices[index]];
                    col = scheme->element_perm[(*call_info)->col_indices[index]]; 

                }

                tileRow = row / scheme->tile_size;
                tileCol = col / scheme->tile_size;

                if(tileRow <= tileCol){
                    
                    int indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
                    int withinTileRow = row % (*call_info)->tile_size;
                    int withinTileCol = col % (*call_info)->tile_size;

                    if (tileRow == tileCol) {
                        scheme->diagonal_bmapper[indexed_tile] = true;
                    }

                    if(tileRow == tileCol){

                        if(withinTileRow <= withinTileCol){
                            
                            scheme->tile_index_lookup[index] = indexed_tile;
                            const int dest_row_tile = tileRow; // diagonal/upper, use tileRow
                            const int tile_ld = (dest_row_tile == num_tiles - 1)
                                              ? scheme->remainderTileSize
                                              : scheme->tile_size;
                            scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);
                               

                        }else {
                            
                            scheme->tile_index_lookup[index] = indexed_tile;
                            const int dest_row_tile = tileRow; // diagonal/upper, use tileRow
                            const int tile_ld = (dest_row_tile == num_tiles - 1)
                                              ? scheme->remainderTileSize
                                              : scheme->tile_size;

                            scheme->element_offset_lookup[index] = withinTileCol + (withinTileRow * tile_ld);

                        }
                        
                    }else{

                        scheme->tile_index_lookup[index] = indexed_tile;
                        const int dest_row_tile = tileRow; // upper
                        const int tile_ld = (dest_row_tile == num_tiles - 1)
                                          ? scheme->remainderTileSize
                                          : scheme->tile_size;

                        scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);
                          

                    }

                }else{

                    int indexed_tile = scheme->mapper.map_ij(tileCol, tileRow, num_tiles);
                    int withinTileRow = col % (*call_info)->tile_size;
                    int withinTileCol = row % (*call_info)->tile_size;

                    scheme->tile_index_lookup[index] = indexed_tile;
                    const int dest_row_tile = tileCol;  // swapped: destination row tile is tileCol
                    const int tile_ld = (dest_row_tile == num_tiles - 1) ? scheme->remainderTileSize : scheme->tile_size;
                    scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);

                }

            }

            //-----------------------------------------------------------------------------------------------------------------------------------------------------
            //-----------------------------------------------------------------------------------------------------------------------------------------------------
            
        }
    }

