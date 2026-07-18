
    static inline void set_tile_extents(TiledMatrix **scheme) {
        if (!scheme) return;
        TiledMatrix* matrix = *scheme;
        if (!matrix || !matrix->tileMetaCore) return;

        const int num_tiles  = matrix->dimTiledMatrix;
        const int max_active = matrix->numActiveTiles;
        if (num_tiles <= 0 || max_active <= 0) return;
        if (!matrix->mapper.valid()) return;

        const int tile_size = matrix->tile_size;
        if (tile_size <= 0) return;

        const int n = matrix->original_order;
        const int remainder = (n % tile_size == 0) ? tile_size : (n % tile_size);

        const int nthreads_ext = std::max(1, omp_get_num_procs());
        #pragma omp parallel for schedule(static) num_threads(nthreads_ext) if(nthreads_ext > 1)
        for (int j = 0; j < num_tiles; ++j) {
            const int width = (j == num_tiles - 1) ? remainder : tile_size;
            for (int i = 0; i <= j; ++i) {
                const int dense_idx = matrix->mapper.map_ij(i, j, num_tiles);
                if (dense_idx < 0 || dense_idx >= max_active) continue;

                const int height = (i == num_tiles - 1) ? remainder : tile_size;

                TileMetaCore& meta = matrix->tileMetaCore[dense_idx];
                meta.index  = dense_idx;
                meta.row    = i;
                meta.col    = j;
                meta.width  = width;
                meta.height = height;
            }
        }
    }

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /*
        1. tile_index_lookup
        2. withinTileRow
        3. withinTileCol
        4. diagonal_bmapper
        5. remainderTileSize
    */

    static inline void build_semisparse_tile_lookup_init_serial(sTiles_call **call_info, TiledMatrix *scheme, int group_index) {

        const int num_tiles = scheme->dimTiledMatrix;
        const int num_tiles_active = scheme->numActiveTiles;
        const int original_nnz = scheme->original_nnz;

        scheme->tile_index_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->withinTileRow = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->withinTileCol = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
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

                    const int indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
                    scheme->tile_index_lookup[index] = indexed_tile;
                    if (tileRow == tileCol) scheme->diagonal_bmapper[indexed_tile] = true;

                }else{

                    sTiles::Logger::error("build_tile_lookup_serial: unexpected lower-triangular entry (row=",
                                          row, ", col=", col, ") while ordering=0.");
                    std::exit(EXIT_FAILURE);
                }

                int withinTileRow = row % (*call_info)->tile_size;
                int withinTileCol = col % (*call_info)->tile_size;

                scheme->withinTileRow[index] = withinTileRow;
                scheme->withinTileCol[index] = withinTileCol;
            }

            //-----------------------------------------------------------------------------------------------------------------------------------------------------
            //-----------------------------------------------------------------------------------------------------------------------------------------------------
       
        }else if(scheme->use_ordering==1){

            //-----------------------------------------------------------------------------------------------------------------------------------------------------
            //-----------------------------------------------------------------------------------------------------------------------------------------------------
            
            int row, col;

            for (int index = 0; index < scheme->original_nnz; index++) { 
            
                if(scheme->element_perm[(*call_info)->col_indices[index]] < scheme->element_perm[(*call_info)->row_indices[index]]){
                    col = scheme->element_perm[(*call_info)->row_indices[index]];
                    row = scheme->element_perm[(*call_info)->col_indices[index]]; 

                }else{
                    row = scheme->element_perm[(*call_info)->row_indices[index]];
                    col = scheme->element_perm[(*call_info)->col_indices[index]]; 

                }

                int tileRow = row / scheme->tile_size;
                int tileCol = col / scheme->tile_size;

                if(tileRow <= tileCol){
                    
                    const int indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
                    scheme->tile_index_lookup[index] = indexed_tile;

                    int withinTileRow = row % (*call_info)->tile_size;
                    int withinTileCol = col % (*call_info)->tile_size;

                    if (tileRow == tileCol) {
                        scheme->diagonal_bmapper[indexed_tile] = true;
                    }

                    scheme->withinTileRow[index] = withinTileRow;
                    scheme->withinTileCol[index] = withinTileCol;

                }else{

                    const int indexed_tile = scheme->mapper.map_ij(tileCol, tileRow, num_tiles);
                    scheme->tile_index_lookup[index] = indexed_tile;

                    int withinTileRow = col % (*call_info)->tile_size;
                    int withinTileCol = row % (*call_info)->tile_size;

                    scheme->withinTileRow[index] = withinTileRow;
                    scheme->withinTileCol[index] = withinTileCol;

                }

            }

            //-----------------------------------------------------------------------------------------------------------------------------------------------------
            //-----------------------------------------------------------------------------------------------------------------------------------------------------
            
        }
    }

    static inline void build_semisparse_tile_lookup_init_parallel(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {

        const int original_nnz = scheme->original_nnz;
        const int num_tiles = scheme->dimTiledMatrix;
        const int num_tiles_active = scheme->numActiveTiles;

        const int threads = (num_cores > 0) ? num_cores : omp_get_max_threads();
        const bool nested_enabled = (omp_get_max_active_levels() > 1);
        const bool can_spawn = (!omp_in_parallel()) || nested_enabled;

        if (!can_spawn || threads < 2) {
            build_semisparse_tile_lookup_init_serial(call_info, scheme, group_index);
            return;
        }

        scheme->tile_index_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->withinTileRow = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->withinTileCol = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->element_offset_lookup = TileMemoryManager::allocateZero<int>(original_nnz, group_index);
        scheme->diagonal_bmapper = TileMemoryManager::allocateZero<bool>(num_tiles_active, group_index);

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

                scheme->tile_index_lookup[index] = indexed_tile;
                const int tile_ld = (tileRow == num_tiles - 1)
                                  ? scheme->remainderTileSize
                                  : scheme->tile_size;
                const int withinTileRow = row % (*call_info)->tile_size;
                const int withinTileCol = col % (*call_info)->tile_size;
                scheme->withinTileRow[index] = withinTileRow;
                scheme->withinTileCol[index] = withinTileCol;
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

                const int tile_ld_diag = (tileRow == num_tiles - 1)
                                         ? scheme->remainderTileSize
                                         : scheme->tile_size;

                if (tileRow <= tileCol) {
                    const int indexed_tile = scheme->mapper.map_ij(tileRow, tileCol, num_tiles);
                    const int withinTileRow = row % (*call_info)->tile_size;
                    const int withinTileCol = col % (*call_info)->tile_size;

                    scheme->tile_index_lookup[index] = indexed_tile;
                    scheme->withinTileRow[index] = withinTileRow;
                    scheme->withinTileCol[index] = withinTileCol;

                    int row_for_offset = withinTileRow;
                    int col_for_offset = withinTileCol;

                    if (tileRow == tileCol) {
                        #pragma omp atomic write
                        scheme->diagonal_bmapper[indexed_tile] = true;

                        if (row_for_offset > col_for_offset) {
                            std::swap(row_for_offset, col_for_offset);
                        }
                    }

                    scheme->element_offset_lookup[index] = row_for_offset + (col_for_offset * tile_ld_diag);
                } else {
                    const int indexed_tile = scheme->mapper.map_ij(tileCol, tileRow, num_tiles);
                    const int withinTileRow = col % (*call_info)->tile_size;
                    const int withinTileCol = row % (*call_info)->tile_size;
                    const int tile_ld = (tileCol == num_tiles - 1)
                                      ? scheme->remainderTileSize
                                      : scheme->tile_size;

                    scheme->tile_index_lookup[index] = indexed_tile;
                    scheme->withinTileRow[index] = withinTileRow;
                    scheme->withinTileCol[index] = withinTileCol;
                    scheme->element_offset_lookup[index] = withinTileRow + (withinTileCol * tile_ld);
                }
            }
        }

        if (error_flag.load(std::memory_order_relaxed)) {
            sTiles::Logger::error("build_semisparse_tile_lookup_init_parallel: unexpected lower-triangular entry encountered (parallel path).");
            std::exit(EXIT_FAILURE);
        }
    }

    static inline void build_semisparse_tile_lookup_phase1(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {
        
        const int n = scheme->original_order;
        scheme->remainderTileSize = (n % scheme->tile_size == 0) ? scheme->tile_size : (n % scheme->tile_size);

        if (num_cores < 2 || scheme->original_nnz < 5000) { build_semisparse_tile_lookup_init_serial(call_info, scheme, group_index); return; }
    #ifndef _OPENMP
        build_semisparse_tile_lookup_init_serial(call_info, scheme, group_index); return;
    #else
        const bool nested_enabled = (omp_get_max_active_levels() > 1);
        const bool can_spawn = (!omp_in_parallel()) || nested_enabled;
        if (!can_spawn) { build_semisparse_tile_lookup_init_serial(call_info, scheme, group_index); return; }
        build_semisparse_tile_lookup_init_parallel(call_info, scheme, group_index, num_cores);
    #endif
    }

    static inline void build_semisparse_tile_lookup_serial(sTiles_call **call_info, TiledMatrix *scheme, int group_index) 
    {
 
        if (!call_info || !*call_info || !scheme) {
            sTiles::Logger::error("[SmartTileLookup] Invalid input to build_semisparse_tile_lookup_serial.");
            return;
        }

        const int num_tiles        = scheme->dimTiledMatrix;   // logical tiles in full grid
        const int num_tiles_active = scheme->numActiveTiles;   // actually used tiles
        const int base_tile_size   = scheme->tile_size;
        const int remainder        = (scheme->remainderTileSize > 0)
                                   ? scheme->remainderTileSize
                                   : base_tile_size;
        const int original_nnz     = scheme->original_nnz;

        if (num_tiles <= 0 || num_tiles_active <= 0) {
            sTiles::Logger::error("[SmartTileLookup] Invalid tiling configuration. tiles=",
                                num_tiles, " active=", num_tiles_active);
            return;
        }

        if (base_tile_size <= 0) {
            sTiles::Logger::error("[SmartTileLookup] Invalid tile size ", base_tile_size);
            return;
        }

        if (!scheme->withinTileRow || !scheme->withinTileCol || !scheme->tile_index_lookup) {
            sTiles::Logger::error("[SmartTileLookup] Missing within-tile metadata; "
                                "ensure build_tile_lookup ran for variant 3.");
            return;
        }

        SemisparseTileMetaCore *core = scheme->semisparseTileMetaCore;
        const bool *bmap             = scheme->diagonal_bmapper;

        if (!core || !bmap) {
            sTiles::Logger::error("[SmartTileLookup] Missing semisparse metadata arrays.");
            return;
        }

        // 1) reset per tile metadata and clear vectors
        for (int t = 0; t < num_tiles_active; ++t) {
            SemisparseTileMetaCore &c = core[t];
            c.fa       = -1;
            c.la       = -1;
            c.sa       = 0;
            c.upper_bw = 0;
            c.aind.clear();
            //c.rind.clear();
            c.acol.clear();   // we will size this once we know width
        }

        // 2) determine width of each active tile and size acol accordingly
        //    tile columns are 0..num_tiles-1
        for (int j = 0; j < num_tiles; ++j) {

            const int width = (j == num_tiles - 1) ? remainder : base_tile_size;

            for (int i = 0; i <= j; ++i) {
                const int tile = scheme->mapper.map_ij(i, j, num_tiles);
                if (tile < 0 || tile >= num_tiles_active) {
                    continue;
                }

                SemisparseTileMetaCore &c = core[tile];

                if (c.acol.empty()) {
                    // first time we encounter this active tile, set its width
                    c.acol.assign(static_cast<std::size_t>(width), -1);
                } else if (static_cast<int>(c.acol.size()) != width) {
                    // safety check for inconsistent mappings
                    sTiles::Logger::warning("[SmartTileLookup] Inconsistent tile width for tile ",
                                            tile, " existing=", static_cast<int>(c.acol.size()),
                                            " new=", width);
                }
            }
        }

        // 3) Mark active columns and bandwidth from exact L fill-in pattern
        //    L is lower-triangular (row >= col), tiles are stored upper-triangular.
        //    For off-diagonal entries: swap tile indices and local coordinates.
        if (scheme->L_colptr && scheme->L_rowind) {
            const int64_t* Lcp = scheme->L_colptr;
            const int* Lri = scheme->L_rowind;
            const int n    = scheme->dim;

            for (int j = 0; j < n; ++j) {
                const int gCol   = j;                        // global column in L (smaller)
                const int tCol   = gCol / base_tile_size;
                const int lCol   = gCol % base_tile_size;

                for (int64_t p = Lcp[j]; p < Lcp[j + 1]; ++p) {
                    const int gRow = Lri[p];                 // global row in L (>= gCol)
                    const int tRow = gRow / base_tile_size;
                    const int lRow = gRow % base_tile_size;

                    if (tRow == tCol) {
                        // Diagonal tile
                        const int tile = scheme->mapper.map_ij(tRow, tCol, num_tiles);
                        if (tile < 0 || tile >= num_tiles_active) continue;
                        SemisparseTileMetaCore &c = core[tile];
                        // bandwidth in upper-tri convention: col >= row
                        const int bw = lRow - lCol;  // always >= 0 in L
                        if (bw > c.upper_bw) c.upper_bw = bw;
                    } else {
                        // Off-diagonal: L has tRow > tCol, map to upper-tri tile
                        const int tile = scheme->mapper.map_ij(tCol, tRow, num_tiles);
                        if (tile < 0 || tile >= num_tiles_active) continue;
                        SemisparseTileMetaCore &c = core[tile];
                        // In upper-tri tile (tCol, tRow):
                        //   withinTileCol = lRow (from the larger tile index)
                        //   withinTileRow = lCol (from the smaller tile index)
                        const int wCol = lRow;  // column in upper-tri tile
                        const int width = static_cast<int>(c.acol.size());
                        if (width <= 0 || wCol < 0 || wCol >= width) continue;
                        if (c.acol[static_cast<std::size_t>(wCol)] == -1) {
                            c.acol[static_cast<std::size_t>(wCol)] = 1;
                            if (c.fa < 0 || wCol < c.fa) c.fa = wCol;
                            if (c.la < 0 || wCol > c.la) c.la = wCol;
                            ++c.sa;
                        }
                    }
                }
            }
        }

        if (false) {
        // OLD: single pass over original nnz entries to fill semisparse info
        for (int idx = 0; idx < original_nnz; ++idx) {

            const int tile = scheme->tile_index_lookup[idx];
            if (tile < 0 || tile >= num_tiles_active) {
                continue;
            }

            SemisparseTileMetaCore &c = core[tile];
            const int width        = static_cast<int>(c.acol.size());
            if (width <= 0) {
                continue;
            }

            const int local_col = scheme->withinTileCol[idx];
            if (local_col < 0 || local_col >= width) {
                continue;
            }

            if (bmap[tile]) {
                // diagonal tile, update bandwidth
                const int local_row = scheme->withinTileRow[idx];
                const int bw        = local_col - local_row;
                if (bw > c.upper_bw) {
                    c.upper_bw = bw;
                }
            } else {
                // off diagonal tile, mark active columns
                if (c.acol[static_cast<std::size_t>(local_col)] == -1) {
                    c.acol[static_cast<std::size_t>(local_col)] = 1;

                    if (c.fa < 0 || local_col < c.fa) {
                        c.fa = local_col;
                    }
                    if (c.la < 0 || local_col > c.la) {
                        c.la = local_col;
                    }
                    ++c.sa;

                    // if you later want aind, you can push_back here
                    // for now you said you do not want to touch aind
                }
            }
        }
        } // end if(false) old path
    }

    static inline void build_semisparse_tile_lookup_phase2(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores)                                                                                     
    {
        if (!call_info || !*call_info || !scheme) {
            return;
        }

    #ifndef _OPENMP
        (void)num_cores;
        (void)group_index;

        build_semisparse_tile_lookup_serial(call_info, scheme, group_index);
        return;
    #else
        const int nnz               = scheme->original_nnz;
        const bool nested_enabled   = (omp_get_max_active_levels() > 1);
        const bool can_spawn        = (!omp_in_parallel()) || nested_enabled;

        //esm
        if (!can_spawn || (num_cores <= 1 && nnz <= 5000)) {
            build_semisparse_tile_lookup_serial(call_info, scheme, group_index);
            return;
        }

        const int num_tiles         = scheme->dimTiledMatrix;   // logical tiles
        const int num_tiles_active  = scheme->numActiveTiles;   // actually used tiles
        const int base_tile_size    = scheme->tile_size;

        if (num_tiles <= 0 || num_tiles_active <= 0 || nnz <= 0 || base_tile_size <= 0) {
            sTiles::Logger::error("[SmartTileLookup] num_tiles... errror.");
            //build_semisparse_tile_lookup_serial(call_info, scheme, group_index);
            return;
        }

        if (!scheme->withinTileRow ||
            !scheme->withinTileCol ||
            !scheme->tile_index_lookup ||
            !scheme->semisparseTileMetaCore ||
            !scheme->diagonal_bmapper) {
            sTiles::Logger::error("[SmartTileLookup] Missing semisparse metadata or within-tile mappings.");
            //build_semisparse_tile_lookup_serial(call_info, scheme, group_index);
            return;
        }

        // keep your ordering and remainder logic (as in the old parallel version)
        if (scheme->use_ordering == 4) {
            (*call_info)->use_nested_dissection = true;
        }

        const int remainder = (scheme->remainderTileSize > 0) ? scheme->remainderTileSize : base_tile_size;

        SemisparseTileMetaCore *core = scheme->semisparseTileMetaCore;
        const bool *bmap          = scheme->diagonal_bmapper;
        const int threads         = (num_cores > 0) ? num_cores : omp_get_max_threads();

        // 1) reset per tile metadata; do not size acol yet
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int t = 0; t < num_tiles_active; ++t) {
            SemisparseTileMetaCore &c = core[t];
            c.fa       = -1;
            c.la       = -1;
            c.sa       = 0;
            c.upper_bw = 0;
            c.aind.clear();
            //c.rind.clear();
            c.acol.clear();
        }

        // 2) determine width for each active tile using mapper
        // Iterate active tiles directly via tileMetaCore (populated by set_tile_extents).
        // Previous version iterated all O(num_tiles^2 / 2) upper-tri positions — far more
        // than num_tiles_active, wasting work on sparse matrices.
        std::vector<int> tile_width(static_cast<std::size_t>(num_tiles_active), 0);
        if (scheme->tileMetaCore) {
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (int t = 0; t < num_tiles_active; ++t) {
                const int j = scheme->tileMetaCore[t].col;
                const int width = (j == num_tiles - 1) ? remainder : base_tile_size;
                tile_width[static_cast<std::size_t>(t)] = width;
            }
        } else {
            // Fallback: original O(num_tiles^2) scan if tileMetaCore is unavailable
            for (int j = 0; j < num_tiles; ++j) {
                const int width = (j == num_tiles - 1) ? remainder : base_tile_size;
                for (int i = 0; i <= j; ++i) {
                    const int tile = scheme->mapper.map_ij(i, j, num_tiles);
                    if (tile < 0 || tile >= num_tiles_active) continue;
                    tile_width[static_cast<std::size_t>(tile)] = width;
                }
            }
        }

        // 3) size acol for each tile based on tile_width
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int t = 0; t < num_tiles_active; ++t) {
            SemisparseTileMetaCore &c = core[t];
            int width = tile_width[static_cast<std::size_t>(t)];

            if (width <= 0) {
                // fallback, should not normally happen if mapper is consistent
                width = base_tile_size;
            }

            c.acol.assign(static_cast<std::size_t>(width), -1);
        }

        // 4) process all nnz in parallel using per tile locks
        std::vector<omp_lock_t> tile_locks(static_cast<std::size_t>(num_tiles_active));
        for (int t = 0; t < num_tiles_active; ++t) {
            omp_init_lock(&tile_locks[static_cast<std::size_t>(t)]);
        }

        // 4) Mark active columns and bandwidth from exact L fill-in pattern
        //    L is lower-triangular (row >= col), tiles are stored upper-triangular.
        //    For off-diagonal entries: swap tile indices and local coordinates.
        //    Use all available system threads for this scan — it's a one-time setup cost.
        if (scheme->L_colptr && scheme->L_rowind) {
            const int64_t* Lcp = scheme->L_colptr;
            const int* Lri = scheme->L_rowind;
            const int n    = scheme->dim;
            const int scan_threads = std::max(threads, omp_get_max_threads());

            #pragma omp parallel for schedule(static) num_threads(scan_threads)
            for (int j = 0; j < n; ++j) {
                const int gCol = j;                        // global column in L (smaller)
                const int tCol = gCol / base_tile_size;
                const int lCol = gCol % base_tile_size;

                for (int64_t p = Lcp[j]; p < Lcp[j + 1]; ++p) {
                    const int gRow = Lri[p];               // global row in L (>= gCol)
                    const int tRow = gRow / base_tile_size;
                    const int lRow = gRow % base_tile_size;

                    if (tRow == tCol) {
                        // Diagonal tile
                        const int tile = scheme->mapper.map_ij(tRow, tCol, num_tiles);
                        if (tile < 0 || tile >= num_tiles_active) continue;
                        SemisparseTileMetaCore &c = core[tile];
                        const int bw = lRow - lCol;  // always >= 0 in L
                        omp_set_lock(&tile_locks[static_cast<std::size_t>(tile)]);
                        if (bw > c.upper_bw) c.upper_bw = bw;
                        omp_unset_lock(&tile_locks[static_cast<std::size_t>(tile)]);
                    } else {
                        // Off-diagonal: L has tRow > tCol, map to upper-tri tile
                        const int tile = scheme->mapper.map_ij(tCol, tRow, num_tiles);
                        if (tile < 0 || tile >= num_tiles_active) continue;
                        SemisparseTileMetaCore &c = core[tile];
                        // In upper-tri tile (tCol, tRow):
                        //   withinTileCol = lRow (from the larger tile index)
                        const int wCol = lRow;
                        const int width = static_cast<int>(c.acol.size());
                        if (width <= 0 || wCol < 0 || wCol >= width) continue;
                        omp_set_lock(&tile_locks[static_cast<std::size_t>(tile)]);
                        if (c.acol[static_cast<std::size_t>(wCol)] == -1) {
                            c.acol[static_cast<std::size_t>(wCol)] = 1;
                            if (c.fa < 0 || wCol < c.fa) c.fa = wCol;
                            if (c.la < 0 || wCol > c.la) c.la = wCol;
                            ++c.sa;
                        }
                        omp_unset_lock(&tile_locks[static_cast<std::size_t>(tile)]);
                    }
                }
            }
        }

        if (false) {
        // OLD: process all nnz in parallel using per tile locks
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int idx = 0; idx < nnz; ++idx) {

            const int tile = scheme->tile_index_lookup[idx];
            if (tile < 0 || tile >= num_tiles_active) {
                continue;
            }

            SemisparseTileMetaCore &c = core[tile];
            const int width = static_cast<int>(c.acol.size());
            if (width <= 0) {
                continue;
            }

            const int local_col = scheme->withinTileCol[idx];
            if (local_col < 0 || local_col >= width) {
                continue;
            }

            if (bmap[tile]) {
                // diagonal tile: update bandwidth
                const int local_row = scheme->withinTileRow[idx];
                const int bw        = local_col - local_row;

                omp_set_lock(&tile_locks[static_cast<std::size_t>(tile)]);
                if (bw > c.upper_bw) {
                    c.upper_bw = bw;
                }
                omp_unset_lock(&tile_locks[static_cast<std::size_t>(tile)]);

            } else {
                // off diagonal tile: mark active columns
                omp_set_lock(&tile_locks[static_cast<std::size_t>(tile)]);

                if (c.acol[static_cast<std::size_t>(local_col)] == -1) {
                    c.acol[static_cast<std::size_t>(local_col)] = 1;

                    if (c.fa < 0 || local_col < c.fa) {
                        c.fa = local_col;
                    }
                    if (c.la < 0 || local_col > c.la) {
                        c.la = local_col;
                    }
                    ++c.sa;
                    // aind stays untouched as you wanted
                }

                omp_unset_lock(&tile_locks[static_cast<std::size_t>(tile)]);
            }
        }
        } // end if(false) old path

        for (int t = 0; t < num_tiles_active; ++t) {
            omp_destroy_lock(&tile_locks[static_cast<std::size_t>(t)]);
        }

        (void)group_index; // not used here but kept for signature symmetry

        return;
    #endif // _OPENMP
    }

    inline StatusCode build_semisparse_tile_lookup(sTiles_call **call_info, TiledMatrix *scheme, int group_index, int num_cores) {
        // Check tile type mode and correction mode
        int* params = sTiles_get_params();
        const int tile_type_mode = params[sTiles::param::TileTypeMode];
        const int correction_mode = params[sTiles::param::SemisparsePruningMode];

        // Only proceed if:
        // - tile_type_mode == 1 (semisparse tiles only), OR
        // - tile_type_mode == 2 (dense and semisparse together), OR
        // - tile_type_mode == 0 (dense tiles) AND correction_mode > 0 (correction enabled)
        if (!(tile_type_mode == 1 || tile_type_mode == 2 || (tile_type_mode == 0 && correction_mode > 0))) {
            // Don't proceed with semisparse tile lookup
            return StatusCode::Success;
        }

        if (!call_info || !*call_info || !scheme) return StatusCode::InvalidArgument;
        const int fact_variant = (*call_info)->factorization_variant;
        // Semisparse only makes sense for variant 0 (sparse tiled) and variant 3 (dense with correction)
        // For variant 1 (single dense tile) or variant 2 (scaled dense), skip semisparse processing
        if (fact_variant == 1 || fact_variant == 2) {
            // Single tile or scaled dense - use dense path instead of semisparse
            return StatusCode::Success;
        }
        if (fact_variant > 3) {
            sTiles::Logger::errorf("Unsupported factorization_variant %d", fact_variant);
            exit(0);
            return StatusCode::InvalidArgument;
        }

        scheme->tileMetaCore = nullptr;
        scheme->semisparseTileMetaCore = nullptr;

        const int num_active = scheme->numActiveTiles;
        if (num_active < 0) return StatusCode::InvalidArgument;

        scheme->semisparseTileMetaCore = TileMemoryManager::allocate<SemisparseTileMetaCore>(num_active, group_index);
        if (!scheme->semisparseTileMetaCore) {
            sTiles::Logger::errorf("Memory allocation failed for semisparseTileMetaCore.");
            return StatusCode::OutOfResources;
        }
        // Construct SemisparseTileMetaCore objects (contains std::vector members)
        for (int t = 0; t < num_active; ++t) {
            new (&scheme->semisparseTileMetaCore[t]) SemisparseTileMetaCore();
        }

        scheme->tileMetaCore = TileMemoryManager::allocate<TileMetaCore>(num_active, group_index);
        if (!scheme->tileMetaCore) {
            sTiles::Logger::errorf("Memory allocation failed for denseTiles or tileMetaCore.");
            return StatusCode::OutOfResources;
        }

        // Redundant: Phase 1 below calls init_parallel/serial which does the same thing. so!
        // if (!scheme->tile_index_lookup || !scheme->withinTileRow ||
        //     !scheme->withinTileCol || !scheme->diagonal_bmapper) {
        //     build_semisparse_tile_lookup_init_parallel(call_info, scheme, group_index, num_cores);
        // }

        set_tile_extents(&scheme);
        const double _t_p1 = omp_get_wtime();
        build_semisparse_tile_lookup_phase1(call_info, scheme, group_index, num_cores);
        const double _t_p2 = omp_get_wtime();
        build_semisparse_tile_lookup_phase2(call_info, scheme, group_index, num_cores);
        const double _t_p3 = omp_get_wtime();
        sTiles::Logger::timing("│   ↪   [diag] phase1: ", (_t_p2 - _t_p1), " s, phase2: ", (_t_p3 - _t_p2), " s");

        // Phase 3 removed: active columns and bandwidth are now computed
        // directly from L_colptr/L_rowind in Phase 2 (serial and parallel paths).

        // DEBUG: Print first 40 tiles
        // {
        //     const int num_to_print = std::min(40, scheme->numActiveTiles);
        //     sTiles::Logger::errorf("\n========== DEBUG: First %d Semisparse Tiles ==========", num_to_print);

        //     for (int t = 0; t < num_to_print; ++t) {
        //         const SemisparseTileMetaCore& semi = scheme->semisparseTileMetaCore[t];
        //         const bool is_diag = scheme->diagonal_bmapper ? scheme->diagonal_bmapper[t] : false;

        //         sTiles::Logger::errorf("\n--- Tile %d [%s] ---", t, is_diag ? "DIAGONAL" : "OFF-DIAG");
        //         sTiles::Logger::errorf("  fa=%d, la=%d, sa=%d, upper_bw=%d",
        //                      semi.fa, semi.la, semi.sa, semi.upper_bw);

        //         // Print acol
        //         sTiles::Logger::errorf("  acol[%zu]: ", semi.acol.size());
        //         for (std::size_t c = 0; c < semi.acol.size() && c < 20; ++c) {
        //             sTiles::Logger::errorf("%d ", semi.acol[c]);
        //         }
        //         if (semi.acol.size() > 20) sTiles::Logger::errorf("...");
        //         sTiles::Logger::errorf("");

        //         // Print aind
        //         sTiles::Logger::errorf("  aind[%zu]: ", semi.aind.size());
        //         for (std::size_t i = 0; i < semi.aind.size() && i < 20; ++i) {
        //             sTiles::Logger::errorf("%d ", semi.aind[i]);
        //         }
        //         if (semi.aind.size() > 20) sTiles::Logger::errorf("...");
        //         sTiles::Logger::errorf("");
        //     }
        //     sTiles::Logger::errorf("\n========== END DEBUG ==========");
        //     //std::exit(0);
        // }

        // ── Build per-tile CSC from exact L fill-in (L_colptr / L_rowind) ──
        // Single-pass, parallel. Each tile is independent (writes own semi fields).
        const double _t_csc_start = omp_get_wtime();
        if (scheme->L_colptr && scheme->L_rowind && num_active > 0) {
            const int ts = scheme->tile_size;
            const int64_t* L_cp = scheme->L_colptr;
            const int* L_ri = scheme->L_rowind;
            const int n_global = scheme->dim;
            // Use max of passed num_cores and system procs — this setup phase is
            // embarrassingly parallel and should always use all available cores,
            // even when compute is single-threaded.
            const int nthreads = std::max({1, num_cores, omp_get_num_procs()});

            #pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads) if(nthreads > 1)
            for (int t = 0; t < num_active; ++t) {
                const TileMetaCore& meta = scheme->tileMetaCore[t];
                SemisparseTileMetaCore& semi = scheme->semisparseTileMetaCore[t];
                const int tile_row = meta.row;
                const int tile_col = meta.col;
                const int h = meta.height;
                if (h <= 0) continue;

                const int r0 = tile_row * ts;
                const int r1 = r0 + h;
                const int c0 = tile_col * ts;
                const int tile_w = (tile_col == (n_global - 1) / ts)
                    ? (n_global - c0) : ts; // handle last tile column
                const int c1 = c0 + tile_w;

                semi.csc_colptr.clear();
                semi.csc_rowind.clear();

                if (tile_row < tile_col) {
                    // Upper triangular off-diagonal tile: chunk_a stores L[n-block,k-block]^T
                    // "column" i of chunk_a = L row c0+aind[i] restricted to cols [r0,r1)
                    // "row" r of chunk_a    = L column r0+r
                    // Build CSC indexed by packed active column (0..sa-1) using acol lookup.
                    // acol must hold packed indices, not Phase 2 flags — compress now.
                    compress_semisparse_columns(semi);
                    const int sa = semi.sa;
                    if (sa <= 0 || semi.acol.empty()) {
                        semi.csc_nnz = 0;
                        continue;
                    }
                    // Pass 1: count nnz per packed column
                    std::vector<int> col_cnt(static_cast<std::size_t>(sa), 0);
                    for (int gc = r0; gc < r1 && gc < n_global; ++gc) {
                        const int64_t p_end = L_cp[gc + 1];
                        for (int64_t p = L_cp[gc]; p < p_end; ++p) {
                            const int gr = L_ri[p];
                            if (gr >= c0 && gr < c1) {
                                const int local_col = gr - c0;
                                if (local_col < static_cast<int>(semi.acol.size())) {
                                    const int ki = semi.acol[local_col];
                                    if (ki >= 0) ++col_cnt[static_cast<std::size_t>(ki)];
                                }
                            }
                        }
                    }
                    // Build colptr (size sa+1)
                    semi.csc_colptr.resize(static_cast<std::size_t>(sa) + 1, 0);
                    for (int i = 0; i < sa; ++i)
                        semi.csc_colptr[static_cast<std::size_t>(i) + 1] =
                            semi.csc_colptr[static_cast<std::size_t>(i)] + col_cnt[static_cast<std::size_t>(i)];
                    const int nnz = semi.csc_colptr[static_cast<std::size_t>(sa)];
                    semi.csc_rowind.resize(static_cast<std::size_t>(nnz));
                    semi.csc_nnz = nnz;
                    if (nnz == 0) continue;
                    // Pass 2: fill rowind
                    std::fill(col_cnt.begin(), col_cnt.end(), 0);
                    for (int gc = r0; gc < r1 && gc < n_global; ++gc) {
                        const int row_r = gc - r0;
                        const int64_t p_end = L_cp[gc + 1];
                        for (int64_t p = L_cp[gc]; p < p_end; ++p) {
                            const int gr = L_ri[p];
                            if (gr >= c0 && gr < c1) {
                                const int local_col = gr - c0;
                                if (local_col < static_cast<int>(semi.acol.size())) {
                                    const int ki = semi.acol[local_col];
                                    if (ki >= 0) {
                                        const std::size_t base = static_cast<std::size_t>(semi.csc_colptr[static_cast<std::size_t>(ki)]);
                                        semi.csc_rowind[base + static_cast<std::size_t>(col_cnt[static_cast<std::size_t>(ki)]++)] = row_r;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Lower triangular or diagonal tile: iterate L columns in [c0,c1),
                    // collect rows in [r0,r1). CSC indexed by L-active columns in order.
                    semi.csc_colptr.reserve(static_cast<std::size_t>(tile_w) + 1);
                    semi.csc_colptr.push_back(0);
                    for (int gc = c0; gc < c1 && gc < n_global; ++gc) {
                        const int start_sz = static_cast<int>(semi.csc_rowind.size());
                        const int64_t p_end = L_cp[gc + 1];
                        for (int64_t p = L_cp[gc]; p < p_end; ++p) {
                            const int gr = L_ri[p];
                            if (gr >= r0 && gr < r1) {
                                semi.csc_rowind.push_back(gr - r0);
                            }
                        }
                        if (static_cast<int>(semi.csc_rowind.size()) > start_sz) {
                            semi.csc_colptr.push_back(static_cast<int>(semi.csc_rowind.size()));
                        }
                    }
                    semi.csc_nnz = static_cast<int>(semi.csc_rowind.size());
                }
            }

            // Log density statistics
            long long total_nnz = 0, total_capacity = 0;
            double occ_sum = 0.0; long long occ_tiles = 0;   // mean active-column occupancy (sa/tile_w)
            for (int t = 0; t < num_active; ++t) {
                const SemisparseTileMetaCore& semi = scheme->semisparseTileMetaCore[t];
                const TileMetaCore& meta = scheme->tileMetaCore[t];
                const int n_cols = static_cast<int>(semi.csc_colptr.size()) - 1;
                total_nnz += semi.csc_nnz;
                total_capacity += static_cast<long long>(meta.height) * n_cols;
                const int c0 = meta.col * ts;
                const int tile_w = (meta.col == (scheme->dim - 1) / ts) ? (scheme->dim - c0) : ts;
                if (tile_w > 0) { occ_sum += static_cast<double>(semi.sa) / tile_w; ++occ_tiles; }
            }
            const double avg_density = total_capacity > 0
                ? 100.0 * static_cast<double>(total_nnz) / total_capacity : 0.0;
            // Mean tile occupancy: candidate feature for semi-vs-sparse mode selection.
            // High (clustered active cols, e.g. sem_n*) -> semisparse; low (scattered) -> sparse.
            const double mean_occ = occ_tiles > 0 ? occ_sum / occ_tiles : 0.0;   // in (0,1]
            if (std::getenv("STILES_PRINT_OCC"))
                sTiles::Logger::errorf("[occ] mean_tile_occupancy=%.4f active_tiles=%lld avg_csc_density=%.1f%%",
                        mean_occ, occ_tiles, avg_density);
            sTiles::Logger::timing("│   ↪ Per-tile CSC built: tiles="
                + std::to_string(num_active)
                + ", total_nnz=" + std::to_string(total_nnz)
                + ", avg_density=" + std::to_string(static_cast<int>(avg_density)) + "%");
        }
        sTiles::Logger::timing("│   ↪   [diag] per-tile CSC: ", (omp_get_wtime() - _t_csc_start), " s");
        return StatusCode::Ok;

    }

    static inline void compress_semisparse_columns(SemisparseTileMetaCore& semi) {
        const int active_cols = semi.sa;
        const int width = static_cast<int>(semi.acol.size());
        if (active_cols <= 0 || semi.acol.empty()) {
            semi.aind.clear();
            semi.is_contiguous = false;
            semi.is_full_width = false;
            return;
        }

        // Nearly-dense promotion: if <= 5 columns are inactive, promote to fully dense.
        // The extra zeros cost less than scatter overhead in DGEMM.
        constexpr int DENSE_THRESHOLD = 5;
        if (width > 0 && (width - active_cols) <= DENSE_THRESHOLD) {
            // Mark all columns as active
            semi.sa = width;
            semi.fa = 0;
            semi.la = width - 1;
            semi.aind.resize(static_cast<std::size_t>(width));
            for (int col = 0; col < width; ++col) {
                semi.acol[col] = col;
                semi.aind[col] = col;
            }
            semi.is_contiguous = true;
            semi.is_full_width = true;
            return;
        }

        semi.aind.assign(static_cast<std::size_t>(active_cols), std::int32_t(-1));
        int next = 0;
        for (std::size_t col = 0; col < semi.acol.size(); ++col) {
            if (semi.acol[col] >= 0) {
                if (next >= active_cols) {
                    semi.aind.push_back(static_cast<std::int32_t>(col));
                } else {
                    semi.aind[static_cast<std::size_t>(next)] = static_cast<std::int32_t>(col);
                }
                semi.acol[col] = next;
                ++next;
            } else {
                semi.acol[col] = -1;
            }
        }

        if (next != active_cols) {
            semi.aind.resize(static_cast<std::size_t>(next));
            semi.sa = next;
        }

        // Precompute flags for fast-path detection in numeric kernels
        semi.is_contiguous = (semi.sa > 0 && (semi.la - semi.fa + 1 == semi.sa));
        semi.is_full_width = (semi.sa == static_cast<int>(semi.acol.size()));
    }

    static inline void clone_semisparse_metadata(SemisparseTileMetaCore* dest, const SemisparseTileMetaCore* src, int count) {
        if (!dest || !src || count <= 0) {
            return;
        }

        for (int idx = 0; idx < count; ++idx) {
            const SemisparseTileMetaCore& s = src[idx];
            SemisparseTileMetaCore& d = dest[idx];
            d.fa        = s.fa;
            d.la        = s.la;
            d.upper_bw  = s.upper_bw;
            d.sa        = s.sa;
            d.aind      = s.aind;   // std::vector copies produce independent storage
            d.acol      = s.acol;
        }
    }

    static inline std::size_t chunked_tile_element_count(TileMetaCore* cores, SemisparseTileMetaCore* semicores, int idx, bool diagonal_tile = false, bool force_full_diagonal = false, bool force_all_dense = false) {
        if (!cores || !semicores) {
            return 0;
        }

        SemisparseTileMetaCore& semi = semicores[idx];
        compress_semisparse_columns(semi);

        const TileMetaCore& meta = cores[idx];
        const int height = (meta.height > 0) ? meta.height : 0;
        const int width = (meta.width > 0) ? meta.width : meta.height;

        // For inverse tiles with force_all_dense=true, ALL tiles use full dense format
        // because the inverse has fill-in beyond the original sparsity pattern
        if (force_all_dense) {
            return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
        }

        // For inverse tiles (force_full_diagonal=true), diagonal tiles use full n×n
        // because the inverse of a banded tile is generally dense
        if (diagonal_tile && force_full_diagonal) {
            return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
        }

        int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
        if (diagonal_tile && diag_cols <= 0) {
            diag_cols = width;
        }
        const int raw_cols = diagonal_tile ? diag_cols : semi.sa;
        const int active_cols = (raw_cols > 0) ? raw_cols : 0;
        if (height <= 0 || active_cols <= 0) {
            return 0;
        }

        return static_cast<std::size_t>(height) * static_cast<std::size_t>(active_cols);
    }

    /**
     * @brief Compute element count for inverse tiles.
     *
     * @param cores           Tile metadata
     * @param semicores       Semisparse tile metadata
     * @param idx             Tile index
     * @param diagonal_tile   True if this is a diagonal tile
     * @param inverse_mode    0=all dense, 1=diag dense/off-diag sparse
     * @return Element count for allocation
     */
    static inline std::size_t chunked_inverse_tile_element_count(
        TileMetaCore* cores, SemisparseTileMetaCore* semicores, int idx,
        bool diagonal_tile, int inverse_mode) {

        if (!cores || !semicores) {
            return 0;
        }

        SemisparseTileMetaCore& semi = semicores[idx];
        compress_semisparse_columns(semi);

        const TileMetaCore& meta = cores[idx];
        const int height = (meta.height > 0) ? meta.height : 0;
        const int width = (meta.width > 0) ? meta.width : meta.height;

        if (height <= 0) return 0;

        // Mode 0: All dense
        if (inverse_mode == 0) {
            return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
        }

        // Mode 1: Diagonal dense, off-diagonal active-cols
        if (inverse_mode == 1) {
            if (diagonal_tile) {
                return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
            } else {
                const int sa = (semi.sa > 0) ? semi.sa : 0;
                return (sa > 0) ? static_cast<std::size_t>(height) * static_cast<std::size_t>(sa) : 0;
            }
        }

        // Default: dense
        return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    }

    static inline void initialize_dense_tile_arrays(DenseTile** tile_sets, int num_sets, TileMetaCore* cores, SemisparseTileMetaCore* semicores, const bool* diagonal_map, int count, int num_cores, int group_index, bool force_full_diagonal = false, bool force_all_dense = false) {
        if (!tile_sets || count <= 0 || num_sets <= 0) {
            return;
        }

        // Arena allocator: replace N per-tile zero'd mallocs with one bulk
        // zero'd allocation per tile_set, and hand out slices to tiles[idx].
        //
        // Same observable behaviour (tiles[idx] is still a double* of the
        // correct length, zero-initialised, callable by all existing
        // kernels), but adjacent tiles now live in adjacent memory. The
        // L_src pack hot loop sees clustered pointer targets → fewer TLB
        // misses, better hardware prefetch, better L2/L3 hit rate.
        //
        // Memory ownership is unchanged: the arena is registered with
        // TileMemoryManager just like the original per-tile allocations,
        // so freeAllGroup() at shutdown reclaims it the same way.

        // First pass (serial): compute per-tile sizes and the prefix sum
        // that gives each tile its offset within the arena.
        std::vector<std::size_t> elem_count(static_cast<std::size_t>(count));
        std::vector<std::size_t> offset(static_cast<std::size_t>(count) + 1, 0);
        for (int idx = 0; idx < count; ++idx) {
            const bool meta_diag = (cores && cores[idx].row == cores[idx].col);
            const bool diag_hint = (diagonal_map && diagonal_map[idx]);
            const bool is_diag_tile = diag_hint || meta_diag;
            elem_count[static_cast<std::size_t>(idx)] =
                chunked_tile_element_count(cores, semicores, idx, is_diag_tile,
                                           force_full_diagonal, force_all_dense);
            offset[static_cast<std::size_t>(idx) + 1] =
                offset[static_cast<std::size_t>(idx)]
                + elem_count[static_cast<std::size_t>(idx)];
        }
        const std::size_t total_elems = offset[static_cast<std::size_t>(count)];

        // Allocate one zero-initialised arena per non-null tile_set.
        std::vector<double*> arena(static_cast<std::size_t>(num_sets), nullptr);
        for (int set_idx = 0; set_idx < num_sets; ++set_idx) {
            if (!tile_sets[set_idx] || total_elems == 0) continue;
            arena[static_cast<std::size_t>(set_idx)] =
                TileMemoryManager::allocateZero<double>(total_elems, group_index);
            if (!arena[static_cast<std::size_t>(set_idx)]) {
                sTiles::Logger::errorf("arena allocation failed for tile_set %d (%zu doubles).",
                    set_idx, total_elems);
                // Leave tiles[idx] = nullptr for this set; kernels guard on it.
            }
        }

        // Second pass (parallel-safe — pure pointer arithmetic, no allocator):
        // hand out slices to each tile.
        auto assign_for_index = [&](int idx) {
            const std::size_t e = elem_count[static_cast<std::size_t>(idx)];
            const std::size_t o = offset[static_cast<std::size_t>(idx)];
            for (int set_idx = 0; set_idx < num_sets; ++set_idx) {
                DenseTile* tiles = tile_sets[set_idx];
                if (!tiles) continue;
                double* base = arena[static_cast<std::size_t>(set_idx)];
                tiles[idx] = (e > 0 && base) ? (base + o) : nullptr;
            }
        };

        #ifdef _OPENMP
        if (num_cores > 1) {
            #pragma omp parallel for schedule(static) num_threads(num_cores)
            for (int idx = 0; idx < count; ++idx) {
                assign_for_index(idx);
            }
            return;
        }
        #endif

        for (int idx = 0; idx < count; ++idx) {
            assign_for_index(idx);
        }
    }

    // ─── Reference: pre-arena per-tile allocator (kept commented for diff/A-B) ─
    //
    // Original implementation of initialize_dense_tile_arrays. Each tile
    // gets its own zero'd malloc. The replacement above gives the same
    // observable behaviour but with one arena per tile_set, which keeps
    // adjacent tiles in adjacent memory and gives ~8-13% semisparse-solve
    // speedup on n>=100K matrices (measured on ferris/spacetime).
    //
    //   static inline void initialize_dense_tile_arrays_per_tile(DenseTile** tile_sets, int num_sets,
    //       TileMetaCore* cores, SemisparseTileMetaCore* semicores, const bool* diagonal_map,
    //       int count, int num_cores, int group_index,
    //       bool force_full_diagonal = false, bool force_all_dense = false) {
    //       if (!tile_sets || count <= 0 || num_sets <= 0) return;
    //       auto allocate_for_index = [&](int idx) {
    //           const bool meta_diag = (cores && cores[idx].row == cores[idx].col);
    //           const bool diag_hint = (diagonal_map && diagonal_map[idx]);
    //           const bool is_diag_tile = diag_hint || meta_diag;
    //           const std::size_t elems = chunked_tile_element_count(cores, semicores, idx,
    //                                       is_diag_tile, force_full_diagonal, force_all_dense);
    //           for (int set_idx = 0; set_idx < num_sets; ++set_idx) {
    //               DenseTile* tiles = tile_sets[set_idx];
    //               if (!tiles) continue;
    //               // Use allocateZero — semisparse DSYRK accumulates with +=.
    //               tiles[idx] = (elems > 0)
    //                   ? TileMemoryManager::allocateZero<double>(elems, group_index)
    //                   : nullptr;
    //           }
    //       };
    //       #ifdef _OPENMP
    //       if (num_cores > 1) {
    //           #pragma omp parallel for schedule(static) num_threads(num_cores)
    //           for (int idx = 0; idx < count; ++idx) allocate_for_index(idx);
    //           return;
    //       }
    //       #endif
    //       for (int idx = 0; idx < count; ++idx) allocate_for_index(idx);
    //   }

    /**
     * @brief Initialize inverse tile arrays.
     *
     * @param tile_sets     Array of tile pointer arrays to allocate
     * @param num_sets      Number of tile sets (typically 2: inverseTiles and savedTiles)
     * @param cores         Tile metadata
     * @param semicores     Semisparse tile metadata
     * @param diagonal_map  Map of diagonal tiles
     * @param count         Number of tiles
     * @param num_cores     Number of cores for parallel allocation
     * @param group_index   Memory group index
     * @param inverse_mode  0=all dense, 1=diag dense/off-diag sparse
     */
    static inline void initialize_inverse_tile_arrays(DenseTile** tile_sets, int num_sets,
        TileMetaCore* cores, SemisparseTileMetaCore* semicores, const bool* diagonal_map,
        int count, int num_cores, int group_index, int inverse_mode) {

        if (!tile_sets || count <= 0 || num_sets <= 0) {
            return;
        }

        // Arena allocator — same pattern as initialize_dense_tile_arrays, but
        // uses chunked_inverse_tile_element_count for sizing and skips the
        // zero-init (inverse tiles are populated by
        // init_chunked_inverse_identity_on_diagonals + selinv kernels, not
        // assumed zero on entry).

        // First pass: per-tile sizes + prefix sum.
        std::vector<std::size_t> elem_count(static_cast<std::size_t>(count));
        std::vector<std::size_t> offset(static_cast<std::size_t>(count) + 1, 0);
        for (int idx = 0; idx < count; ++idx) {
            const bool meta_diag = (cores && cores[idx].row == cores[idx].col);
            const bool diag_hint = (diagonal_map && diagonal_map[idx]);
            const bool is_diag_tile = diag_hint || meta_diag;
            elem_count[static_cast<std::size_t>(idx)] =
                chunked_inverse_tile_element_count(cores, semicores, idx,
                                                   is_diag_tile, inverse_mode);
            offset[static_cast<std::size_t>(idx) + 1] =
                offset[static_cast<std::size_t>(idx)]
                + elem_count[static_cast<std::size_t>(idx)];
        }
        const std::size_t total_elems = offset[static_cast<std::size_t>(count)];

        // One arena per non-null tile_set.
        std::vector<double*> arena(static_cast<std::size_t>(num_sets), nullptr);
        for (int set_idx = 0; set_idx < num_sets; ++set_idx) {
            if (!tile_sets[set_idx] || total_elems == 0) continue;
            arena[static_cast<std::size_t>(set_idx)] =
                TileMemoryManager::allocate<double>(total_elems, group_index);
            if (!arena[static_cast<std::size_t>(set_idx)]) {
                sTiles::Logger::errorf("arena allocation failed for inverse tile_set %d (%zu doubles).",
                    set_idx, total_elems);
            }
        }

        // Second pass: hand out per-tile slices (parallel-safe).
        auto assign_for_index = [&](int idx) {
            const std::size_t e = elem_count[static_cast<std::size_t>(idx)];
            const std::size_t o = offset[static_cast<std::size_t>(idx)];
            for (int set_idx = 0; set_idx < num_sets; ++set_idx) {
                DenseTile* tiles = tile_sets[set_idx];
                if (!tiles) continue;
                double* base = arena[static_cast<std::size_t>(set_idx)];
                tiles[idx] = (e > 0 && base) ? (base + o) : nullptr;
            }
        };

        #ifdef _OPENMP
        if (num_cores > 1) {
            #pragma omp parallel for schedule(static) num_threads(num_cores)
            for (int idx = 0; idx < count; ++idx) {
                assign_for_index(idx);
            }
            return;
        }
        #endif

        for (int idx = 0; idx < count; ++idx) {
            assign_for_index(idx);
        }
    }

    // ─── Reference: pre-arena per-tile allocator for inverse/saved sets ───────
    //
    // Original implementation of initialize_inverse_tile_arrays. Same shape
    // as the dense version but with chunked_inverse_tile_element_count for
    // sizing and plain allocate (no zero-init — selinv populates entries).
    //
    //   static inline void initialize_inverse_tile_arrays_per_tile(DenseTile** tile_sets, int num_sets,
    //       TileMetaCore* cores, SemisparseTileMetaCore* semicores, const bool* diagonal_map,
    //       int count, int num_cores, int group_index, int inverse_mode) {
    //       if (!tile_sets || count <= 0 || num_sets <= 0) return;
    //       auto allocate_for_index = [&](int idx) {
    //           const bool meta_diag = (cores && cores[idx].row == cores[idx].col);
    //           const bool diag_hint = (diagonal_map && diagonal_map[idx]);
    //           const bool is_diag_tile = diag_hint || meta_diag;
    //           const std::size_t elems = chunked_inverse_tile_element_count(cores, semicores, idx,
    //                                                                        is_diag_tile, inverse_mode);
    //           for (int set_idx = 0; set_idx < num_sets; ++set_idx) {
    //               DenseTile* tiles = tile_sets[set_idx];
    //               if (!tiles) continue;
    //               tiles[idx] = (elems > 0)
    //                   ? TileMemoryManager::allocate<double>(elems, group_index)
    //                   : nullptr;
    //           }
    //       };
    //       #ifdef _OPENMP
    //       if (num_cores > 1) {
    //           #pragma omp parallel for schedule(static) num_threads(num_cores)
    //           for (int idx = 0; idx < count; ++idx) allocate_for_index(idx);
    //           return;
    //       }
    //       #endif
    //       for (int idx = 0; idx < count; ++idx) allocate_for_index(idx);
    //   }

    static inline StatusCode init_chunked_inverse_identity_on_diagonals(TiledMatrix* scheme) {
        if (!scheme || !scheme->compute_inverse || !scheme->chunkedInverseTiles) return StatusCode::Success;
        if (!scheme->tileMetaCore || !scheme->semisparseTileMetaCore) return StatusCode::Success;

        const int num_active = scheme->numActiveTiles;
        const bool* diag_map = scheme->diagonal_bmapper;

        // Check inverse storage mode: params[7]
        //   0 = dense: all inverse tiles are full h×w
        //   1 = semisparse: diagonal tiles are DENSE, off-diagonal tiles use active-cols format
        int* params = sTiles_get_params();
        const int inverse_storage_mode = params ? params[sTiles::param::InverseStorageMode] : 0;
        const bool semisparse_offdiag = (inverse_storage_mode == 1);

        int diag_count = 0;
        for (int t = 0; t < num_active; ++t) {
            double* inv = scheme->chunkedInverseTiles[t];
            if (!inv) {
                sTiles::Logger::warning("[init_inv_identity] Tile ", t, " has null chunkedInverseTiles pointer");
                continue;
            }

            const TileMetaCore& meta = scheme->tileMetaCore[t];
            const SemisparseTileMetaCore& semi = scheme->semisparseTileMetaCore[t];
            const int h = (meta.height > 0) ? meta.height : scheme->tile_size;
            const int w = (meta.width > 0) ? meta.width : scheme->tile_size;

            const bool meta_diag = (meta.row == meta.col);
            const bool lapack_diag = (diag_map && diag_map[t]);
            const bool is_diag = lapack_diag || meta_diag;

            if (h <= 0 || w <= 0) continue;

            // Initialize based on storage format
            if (is_diag) {
                // Diagonal tiles use dense format h × w
                const std::size_t elems = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
                std::fill(inv, inv + elems, 0.0);
                const int dsz = std::min(h, w);
                for (int i = 0; i < dsz; ++i) {
                    inv[i + i * h] = 1.0;
                }
                diag_count++;
            } else if (semisparse_offdiag && semi.sa > 0) {
                // Off-diagonal tiles in semisparse mode use active-cols (h × sa)
                const std::size_t elems = static_cast<std::size_t>(h) * static_cast<std::size_t>(semi.sa);
                std::fill(inv, inv + elems, 0.0);
            } else {
                // Dense off-diagonal (h × w)
                const std::size_t elems = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
                std::fill(inv, inv + elems, 0.0);
            }
        }
        sTiles::Logger::debug("[init_inv_identity] Initialized identity on ", diag_count, " diagonal tiles out of ", num_active, " active tiles",
                              " (inverse_storage_mode=", inverse_storage_mode, ")");
        return StatusCode::Success;
    }

    inline StatusCode allocate_semisparse_tiles(TiledMatrix *scheme, int group_index, int num_cores) {
        // Check tile type mode and correction mode
        int* params = sTiles_get_params();
        const int tile_type_mode = params[sTiles::param::TileTypeMode];
        const int correction_mode = params[sTiles::param::SemisparsePruningMode];
        bool no_inverse = false;

        // Only proceed if:
        // - tile_type_mode == 1 (semisparse tiles only), OR
        // - tile_type_mode == 2 (both dense and semisparse for debugging), OR
        // - tile_type_mode == 0 (dense tiles) AND correction_mode > 0 (correction enabled)
        if (!(tile_type_mode == 1 || tile_type_mode == 2 || (tile_type_mode == 0 && correction_mode > 0))) {
            // Don't proceed with semisparse allocation
            return StatusCode::Success;
        }

        // Skip semisparse allocation for variant 1 (single tile) and variant 2 (scaled dense)
        // These variants use dense processing which doesn't benefit from semisparse
        // Note: process.cpp now forces tile_type_mode=0 for single-tile cases,
        // so this function won't be called for single-tile matrices
        if (!scheme->semisparseTileMetaCore) {
            // semisparseTileMetaCore not populated - likely variant 1 or 2, skip
            return StatusCode::Success;
        }
        if(tile_type_mode == 0 && correction_mode > 0){
            no_inverse = true;
        }

        const int num_active = scheme->numActiveTiles;

        scheme->chunkedDenseTiles = nullptr;
        scheme->chunkedRhsTiles = nullptr;
        scheme->chunkedSavedTiles = nullptr;
        scheme->chunkedInverseTiles = nullptr;

        scheme->chunkedDenseTiles = TileMemoryManager::allocate<DenseTile>(num_active, group_index);
        if (!scheme->chunkedDenseTiles) {
            sTiles::Logger::errorf("Memory allocation failed for chunkedDenseTiles.");
            return StatusCode::OutOfResources;
        }

#ifdef SPARSE_STILES
        scheme->sparseTileCSC = new (std::nothrow) sTiles::SparseTileCSC[static_cast<std::size_t>(num_active)]();
        if (!scheme->sparseTileCSC) {
            sTiles::Logger::errorf("Memory allocation failed for sparseTileCSC.");
            return StatusCode::OutOfResources;
        }
#endif

        DenseTile* dense_sets[] = { scheme->chunkedDenseTiles };
        initialize_dense_tile_arrays(dense_sets,
                                     1,
                                     scheme->tileMetaCore,
                                     scheme->semisparseTileMetaCore,
                                     scheme->diagonal_bmapper,
                                     num_active,
                                     num_cores,
                                     group_index);

        if (scheme->compute_inverse && !no_inverse) {
            scheme->chunkedInverseTiles = TileMemoryManager::allocate<DenseTile>(num_active, group_index);
            scheme->chunkedSavedTiles = TileMemoryManager::allocate<DenseTile>(num_active, group_index);
            if (!scheme->chunkedInverseTiles || !scheme->chunkedSavedTiles) {
                sTiles::Logger::errorf("Memory allocation failed for chunkedInverseTiles/chunkedSavedTiles.");
                if (scheme->chunkedSavedTiles) {
                    TileMemoryManager::deallocate(scheme->chunkedSavedTiles);
                    scheme->chunkedSavedTiles = nullptr;
                }
                if (scheme->chunkedInverseTiles) {
                    TileMemoryManager::deallocate(scheme->chunkedInverseTiles);
                    scheme->chunkedInverseTiles = nullptr;
                }
                for (int idx = 0; idx < num_active; ++idx) {
                    if (scheme->chunkedDenseTiles[idx]) {
                        TileMemoryManager::deallocate(scheme->chunkedDenseTiles[idx]);
                    }
                }
                TileMemoryManager::deallocate(scheme->chunkedDenseTiles);
                scheme->chunkedDenseTiles = nullptr;
                return StatusCode::OutOfResources;
            }

            // Check inverse storage mode: params[7]
            //   0 = dense (default): all inverse tiles use full h×w format
            //   1 = semisparse: off-diagonal inverse tiles use active-cols, diagonal stays dense
            const int inverse_storage_mode = params[sTiles::param::InverseStorageMode];

            // Inverse tiles allocation
            DenseTile* inv_and_saved_sets[] = { scheme->chunkedInverseTiles, scheme->chunkedSavedTiles };
            initialize_inverse_tile_arrays(inv_and_saved_sets,
                                          2,
                                          scheme->tileMetaCore,
                                          scheme->semisparseTileMetaCore,
                                          scheme->diagonal_bmapper,
                                          num_active,
                                          num_cores,
                                          group_index,
                                          inverse_storage_mode);

            const StatusCode sc = init_chunked_inverse_identity_on_diagonals(scheme);
            if (sc != StatusCode::Success) return sc;

        } else {
            scheme->chunkedInverseTiles = nullptr;
            scheme->chunkedSavedTiles = nullptr;
        }

        // DEBUG: Print first 40 tiles after aind is built
        // {
        //     const int num_to_print = std::min(40, num_active);
        //     sTiles::Logger::errorf("\n========== DEBUG: After allocate_semisparse_tiles (aind built) ==========");
        //     sTiles::Logger::errorf("Total active tiles: %d", num_active);

        //     for (int t = 0; t < num_to_print; ++t) {
        //         const SemisparseTileMetaCore& semi = scheme->semisparseTileMetaCore[t];
        //         const TileMetaCore& meta = scheme->tileMetaCore[t];
        //         const bool is_diag = scheme->diagonal_bmapper ? scheme->diagonal_bmapper[t] : false;

        //         sTiles::Logger::errorf("\n--- Tile %d [%s] (row=%d, col=%d, h=%d, w=%d) ---",
        //                      t, is_diag ? "DIAGONAL" : "OFF-DIAG",
        //                      meta.row, meta.col, meta.height, meta.width);
        //         sTiles::Logger::errorf("  fa=%d, la=%d, sa=%d, upper_bw=%d",
        //                      semi.fa, semi.la, semi.sa, semi.upper_bw);

        //         // Print acol (now contains compressed indices, not just markers)
        //         sTiles::Logger::errorf("  acol[%zu]: ", semi.acol.size());
        //         for (std::size_t c = 0; c < semi.acol.size() && c < 20; ++c) {
        //             sTiles::Logger::errorf("%d ", semi.acol[c]);
        //         }
        //         if (semi.acol.size() > 20) sTiles::Logger::errorf("...");
        //         sTiles::Logger::errorf("");

        //         // Print aind (maps compressed index -> original column)
        //         sTiles::Logger::errorf("  aind[%zu]: ", semi.aind.size());
        //         for (std::size_t i = 0; i < semi.aind.size() && i < 20; ++i) {
        //             sTiles::Logger::errorf("%d ", semi.aind[i]);
        //         }
        //         if (semi.aind.size() > 20) sTiles::Logger::errorf("...");
        //         sTiles::Logger::errorf("");
        //     }
        //     sTiles::Logger::errorf("\n========== END DEBUG (allocate_semisparse_tiles) ==========");
        // }

        return StatusCode::Ok;
    }

    // Allocate semisparse (chunked) tile buffers for clone calls, using primary's metadata
    inline StatusCode allocate_semisparse_buffers_from_primary(const TiledMatrix* primary, TiledMatrix* clone, int group_index, int num_threads) {
        if (!primary || !clone) {
            return StatusCode::Failure;
        }

        // Check tile type mode - only allocate if semisparse mode is active
        int* params = sTiles_get_params();
        const int tile_type_mode = params[sTiles::param::TileTypeMode];
        const int correction_mode = params[sTiles::param::SemisparsePruningMode];

        // Only proceed if semisparse tiles are needed
        if (!(tile_type_mode == 1 || tile_type_mode == 2 || (tile_type_mode == 0 && correction_mode > 0))) {
            // Not using semisparse tiles
            clone->chunkedDenseTiles = nullptr;
            clone->chunkedRhsTiles = nullptr;
            clone->chunkedSavedTiles = nullptr;
            clone->chunkedInverseTiles = nullptr;
            return StatusCode::Success;
        }

        const int num_active = primary->numActiveTiles;
        if (num_active <= 0) {
            clone->chunkedDenseTiles = nullptr;
            clone->chunkedRhsTiles = nullptr;
            clone->chunkedSavedTiles = nullptr;
            clone->chunkedInverseTiles = nullptr;
            return StatusCode::Success;
        }

        // Allocate chunkedDenseTiles
        clone->chunkedDenseTiles = TileMemoryManager::allocate<DenseTile>(num_active, group_index);
        if (!clone->chunkedDenseTiles) {
            sTiles::Logger::errorf("Memory allocation failed for chunkedDenseTiles (clone).");
            return StatusCode::OutOfResources;
        }

#ifdef SPARSE_STILES
        clone->sparseTileCSC = new (std::nothrow) sTiles::SparseTileCSC[static_cast<std::size_t>(num_active)]();
        if (!clone->sparseTileCSC) {
            sTiles::Logger::errorf("Memory allocation failed for sparseTileCSC (clone).");
            return StatusCode::OutOfResources;
        }
#endif

        // Use primary's metadata for allocation (clone shares semisparseTileMetaCore with primary)
        DenseTile* dense_sets[] = { clone->chunkedDenseTiles };
        initialize_dense_tile_arrays(dense_sets,
                                     1,
                                     clone->tileMetaCore,
                                     clone->semisparseTileMetaCore,
                                     clone->diagonal_bmapper,
                                     num_active,
                                     num_threads,
                                     group_index);

        // Allocate inverse and saved tiles if needed
        bool no_inverse = (tile_type_mode == 0 && correction_mode > 0);
        if (clone->compute_inverse && !no_inverse) {
            clone->chunkedInverseTiles = TileMemoryManager::allocate<DenseTile>(num_active, group_index);
            clone->chunkedSavedTiles = TileMemoryManager::allocate<DenseTile>(num_active, group_index);

            if (!clone->chunkedInverseTiles || !clone->chunkedSavedTiles) {
                sTiles::Logger::errorf("Memory allocation failed for chunkedInverseTiles/chunkedSavedTiles (clone).");
                // Cleanup
                if (clone->chunkedSavedTiles) {
                    TileMemoryManager::deallocate(clone->chunkedSavedTiles);
                    clone->chunkedSavedTiles = nullptr;
                }
                if (clone->chunkedInverseTiles) {
                    TileMemoryManager::deallocate(clone->chunkedInverseTiles);
                    clone->chunkedInverseTiles = nullptr;
                }
                for (int idx = 0; idx < num_active; ++idx) {
                    if (clone->chunkedDenseTiles[idx]) {
                        TileMemoryManager::deallocate(clone->chunkedDenseTiles[idx]);
                    }
                }
                TileMemoryManager::deallocate(clone->chunkedDenseTiles);
                clone->chunkedDenseTiles = nullptr;
                return StatusCode::OutOfResources;
            }

            // Check inverse storage mode: params[7]
            //   0 = dense (default): all inverse tiles use full h×w format
            //   1 = semisparse: off-diagonal inverse tiles use active-cols, diagonal stays dense
            const int inverse_storage_mode = params[sTiles::param::InverseStorageMode];

            DenseTile* inv_and_saved_sets[] = { clone->chunkedInverseTiles, clone->chunkedSavedTiles };
            initialize_inverse_tile_arrays(inv_and_saved_sets,
                                          2,
                                          clone->tileMetaCore,
                                          clone->semisparseTileMetaCore,
                                          clone->diagonal_bmapper,
                                          num_active,
                                          num_threads,
                                          group_index,
                                          inverse_storage_mode);

            // Initialize inverse identity on diagonals
            const StatusCode sc = init_chunked_inverse_identity_on_diagonals(clone);
            if (sc != StatusCode::Success) return sc;
        } else {
            clone->chunkedInverseTiles = nullptr;
            clone->chunkedSavedTiles = nullptr;
        }

        clone->chunkedRhsTiles = nullptr;

        return StatusCode::Success;
    }

    inline int map_padding_index(const TiledMatrix *S, int global_index) {
        if (!S) return -1;
        if (global_index < 0 || global_index >= S->dim) return -1;
        if (S->use_ordering && S->element_perm) {
            return S->element_perm[global_index];
        }
        return global_index;
    }

    inline void scatter_nd_padding_dense_tiles(TiledMatrix *S) {
        if (!S || S->nd_padding <= 0) {
            return;
        }
        if (!S->denseTiles || !S->tileMetaCore) {
            return;
        }

        const int base_index = (S->nd_padding > 0)
                              ? std::max(0, S->dim - S->nd_padding)
                              : S->dim;
        const int padding_end = std::min(S->dim, base_index + S->nd_padding);
        const int tile_size = S->tile_size;
        const int num_tiles = S->dimTiledMatrix;

        for (int global = base_index; global < padding_end; ++global) {
            const int ordered = map_padding_index(S, global);
            if (ordered < 0) {
                continue;
            }

            const int tile_row = ordered / tile_size;
            if (tile_row < 0 || tile_row >= num_tiles) {
                continue;
            }

            const int tile_index = S->mapper.map_ij(tile_row, tile_row, num_tiles);
            if (tile_index < 0 || tile_index >= S->numActiveTiles) {
                continue;
            }

            double *tile = S->denseTiles[tile_index];
            if (!tile) {
                continue;
            }

            const TileMetaCore &meta = S->tileMetaCore[tile_index];
            const int h = (meta.height > 0) ? meta.height : S->tile_size;
            const int w = (meta.width  > 0) ? meta.width  : S->tile_size;
            const int local_row = ordered - tile_row * tile_size;
            if (local_row < 0 || local_row >= h || local_row >= w) {
                continue;
            }

            const int ld = (tile_row == S->dimTiledMatrix - 1) ? S->remainderTileSize : S->tile_size;
            if (ld <= 0 || local_row >= ld) {
                continue;
            }

            const std::size_t offset = static_cast<std::size_t>(local_row)
                                     + static_cast<std::size_t>(local_row) * static_cast<std::size_t>(ld);
            tile[offset] = 1.0;
        }
    }

    inline void scatter_nd_padding_semisparse_tiles(TiledMatrix *S) {
        if (!S || S->nd_padding <= 0) {
            return;
        }
        if (!S->chunkedDenseTiles || !S->tileMetaCore || !S->semisparseTileMetaCore) {
            return;
        }

        const int base_index = (S->nd_padding > 0)
                              ? std::max(0, S->dim - S->nd_padding)
                              : S->dim;
        const int padding_end = std::min(S->dim, base_index + S->nd_padding);
        const int tile_size = S->tile_size;
        const int num_tiles = S->dimTiledMatrix;

        for (int global = base_index; global < padding_end; ++global) {
            const int ordered = map_padding_index(S, global);
            if (ordered < 0) {
                continue;
            }

            const int tile_row = ordered / tile_size;
            if (tile_row < 0 || tile_row >= num_tiles) {
                continue;
            }
            const int tile_index = S->mapper.map_ij(tile_row, tile_row, num_tiles);
            if (tile_index < 0 || tile_index >= S->numActiveTiles) {
                continue;
            }

            double *chunk = S->chunkedDenseTiles[tile_index];
            if (!chunk) {
                continue;
            }

            const TileMetaCore &meta = S->tileMetaCore[tile_index];
            const SemisparseTileMetaCore &semi = S->semisparseTileMetaCore[tile_index];
            const int h = (meta.height > 0) ? meta.height : S->tile_size;
            const int w = (meta.width  > 0) ? meta.width  : S->tile_size;
            const int local_row = ordered - tile_row * tile_size;
            const int local_col = local_row;
            if (local_row < 0 || local_col < 0 || local_row >= h || local_col >= w) {
                continue;
            }

            const bool lapack_diag = (S->diagonal_bmapper && S->diagonal_bmapper[tile_index]);
            const bool diag_tile = (meta.row == meta.col);
            int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
            if ((lapack_diag || diag_tile) && diag_cols <= 0) {
                diag_cols = w;
            }

            const int band = local_col - local_row;
            if (lapack_diag) {
                if (diag_cols <= 0 || band < 0 || band >= diag_cols) {
                    continue;
                }
                const int lapack_row = diag_cols - 1 - band;
                const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                         + static_cast<std::size_t>(diag_cols) * static_cast<std::size_t>(local_col);
                chunk[offset] = 1.0;
                continue;
            }

            if (diag_tile && diag_cols > 0) {
                if (band < 0 || band >= diag_cols) {
                    continue;
                }
                const std::size_t offset = static_cast<std::size_t>(local_row)
                                         + static_cast<std::size_t>(band) * static_cast<std::size_t>(h);
                chunk[offset] = 1.0;
                continue;
            }

            if (semi.acol.empty() || local_col >= static_cast<int>(semi.acol.size())) {
                continue;
            }
            const int active_col = semi.acol[static_cast<std::size_t>(local_col)];
            if (active_col < 0 || active_col >= semi.sa || local_row >= h) {
                continue;
            }

            const std::size_t offset = static_cast<std::size_t>(local_row)
                                     + static_cast<std::size_t>(active_col) * static_cast<std::size_t>(h);
            chunk[offset] = 1.0;
        }
    }

    inline void update_x_semisparse_tiles(int global_index, TiledMatrix **scheme, double *x, bool nested) {

        if (!scheme || !scheme[global_index] || !x) {
            return;
        }

        TiledMatrix* S = scheme[global_index];
        // Skip the pad tail of the COO (appended diagonals from ND / SCOTCH
        // block padding). scatter_nd_padding_semisparse_tiles() sets those
        // cells below; reading x[] past the user's allocation would be UB.
        const int nnz = S->original_nnz - S->nd_padding;
        const int cores = S->num_cores;
        const bool *diag_map = S->diagonal_bmapper;
        auto is_diagonal_tile = [&](int tile_idx) -> bool {
            if (!S->tileMetaCore) return false;
            const TileMetaCore& meta = S->tileMetaCore[tile_idx];
            return meta.row == meta.col;
        };
        auto uses_lapack_diagonal = [&](int tile_idx) -> bool {
            return diag_map && diag_map[tile_idx];
        };

        if (!S->chunkedDenseTiles || !S->tileMetaCore || !S->semisparseTileMetaCore) {
            return;
        }

        if (!S->tile_index_lookup || !S->withinTileRow || !S->withinTileCol) {
            sTiles::Logger::error("[update_x_semisparse_tiles] Missing lookup tables (tile_index_lookup=",
                                  static_cast<const void*>(S->tile_index_lookup),
                                  ", withinTileRow=",
                                  static_cast<const void*>(S->withinTileRow),
                                  ", withinTileCol=",
                                  static_cast<const void*>(S->withinTileCol),
                                  ") — ensure build_semisparse_tile_lookup ran successfully.");
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

        auto zero_tile = [&](int t) {
            if (t < 0 || t >= S->numActiveTiles) {
                return;
            }
            double* chunk = S->chunkedDenseTiles[t];
            if (!chunk) {
                return;
            }
            const TileMetaCore& meta = S->tileMetaCore[t];
            const int h = (meta.height > 0) ? meta.height : S->tile_size;
            const int w = (meta.width  > 0) ? meta.width  : S->tile_size;
            const SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[t];
            const bool lapack_diag = uses_lapack_diagonal(t);
            const bool is_diag = is_diagonal_tile(t);
            int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
            if ((lapack_diag || is_diag) && diag_cols <= 0) {
                diag_cols = w;
            }

            if (lapack_diag) {
                if (w <= 0 || diag_cols <= 0) {
                    return;
                }
                const std::size_t elems = static_cast<std::size_t>(diag_cols)
                                        * static_cast<std::size_t>(w);
                std::fill(chunk, chunk + elems, 0.0);
                return;
            }

            const int active_cols = (is_diag && diag_cols > 0) ? diag_cols : semi.sa;
            if (h <= 0 || active_cols <= 0) {
                return;
            }
            const std::size_t elems = static_cast<std::size_t>(h)
                                    * static_cast<std::size_t>(active_cols);
            std::fill(chunk, chunk + elems, 0.0);
        };

        auto scatter_entry = [&](int idx) {
            const int tile = S->tile_index_lookup[idx];
            if (tile < 0 || tile >= S->numActiveTiles) {
                return;
            }
            double* chunk = S->chunkedDenseTiles[tile];
            if (!chunk) {
                return;
            }
            const SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[tile];
            const bool lapack_diag = uses_lapack_diagonal(tile);
            const bool is_diag = is_diagonal_tile(tile);
            int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
            const int local_col = S->withinTileCol[idx];
            if (local_col < 0) {
                return;
            }
            const int local_row = S->withinTileRow[idx];
            if (local_row < 0) {
                return;
            }
            const TileMetaCore& meta = S->tileMetaCore[tile];
            const int h = (meta.height > 0) ? meta.height : S->tile_size;
            const int w = (meta.width  > 0) ? meta.width  : S->tile_size;

            if ((lapack_diag || is_diag) && diag_cols <= 0) {
                diag_cols = w;
            }

            // Handle LAPACK banded format for diagonal tiles first
            // (don't need acol for this case)
            if (lapack_diag) {
                if (w <= 0 || diag_cols <= 0) {
                    return;
                }
                if (local_col >= w || local_row >= h) {
                    return;
                }
                const int band = local_col - local_row;
                if (band < 0 || band >= diag_cols) {
                    return;
                }
                const int lapack_row = diag_cols - 1 - band;
                const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                         + static_cast<std::size_t>(diag_cols)
                                         * static_cast<std::size_t>(local_col);
                chunk[offset] = x[idx];
                return;
            }

            if (local_row >= h) {
                return;
            }
            if (is_diag && diag_cols > 0) {
                const int band = local_col - local_row;
                if (band < 0 || band >= diag_cols) {
                    return;
                }
                const std::size_t offset = static_cast<std::size_t>(local_row)
                                         + static_cast<std::size_t>(band) * static_cast<std::size_t>(h);
                chunk[offset] = x[idx];
                return;
            }

            // For non-diagonal or non-LAPACK tiles, check acol bounds now
            if (semi.acol.empty() || semi.sa <= 0 || local_col >= static_cast<int>(semi.acol.size())) {
                return;
            }
            const int active_col = semi.acol[static_cast<std::size_t>(local_col)];
            if (active_col < 0 || active_col >= semi.sa) {
                return;
            }
            const std::size_t offset = static_cast<std::size_t>(local_row)
                                     + static_cast<std::size_t>(active_col) * static_cast<std::size_t>(h);
            chunk[offset] = x[idx];
        };

        #ifdef _OPENMP
            if (do_parallel) {
                #pragma omp parallel num_threads(cores)
                {
                    #pragma omp for schedule(static)
                    for (int t = 0; t < S->numActiveTiles; ++t) {
                        zero_tile(t);
                    }

                    #pragma omp for schedule(static)
                    for (int idx = 0; idx < nnz; ++idx) {
                        scatter_entry(idx);
                    }
                }
            } else
        #endif
            {
                for (int t = 0; t < S->numActiveTiles; ++t) {
                    zero_tile(t);
                }
                for (int idx = 0; idx < nnz; ++idx) {
                    scatter_entry(idx);
                }
            }

            scatter_nd_padding_semisparse_tiles(S);

#ifdef SPARSE_STILES
            if (S->sparseTileCSC) {
                // Zero sparse tile values
                for (int t = 0; t < S->numActiveTiles; ++t) {
                    SparseTileCSC &csc = S->sparseTileCSC[t];
                    if (!csc.values.empty())
                        std::fill(csc.values.begin(), csc.values.end(), 0.0);
                }
                // Scatter x[] directly into SparseTileCSC values
                for (int idx = 0; idx < nnz; ++idx) {
                    const int tile = S->tile_index_lookup[idx];
                    if (tile < 0 || tile >= S->numActiveTiles) continue;

                    SparseTileCSC &csc = S->sparseTileCSC[tile];
                    if (csc.values.empty()) continue;

                    const int local_row = S->withinTileRow[idx];
                    const int local_col = S->withinTileCol[idx];
                    if (local_row < 0 || local_col < 0) continue;

                    const SemisparseTileMetaCore &semi = S->semisparseTileMetaCore[tile];
                    const TileMetaCore &meta = S->tileMetaCore[tile];
                    const bool is_diag = is_diagonal_tile(tile) || uses_lapack_diagonal(tile);

                    if (is_diag) {
                        // Banded format: same indexing as semisparse diagonal scatter
                        const int kd = (semi.upper_bw >= 0) ? semi.upper_bw : 0;
                        const int band = local_col - local_row;
                        if (band < 0 || band > kd) continue;
                        const int ldab = kd + 1;
                        const int lapack_row = kd - band;
                        const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                                 + static_cast<std::size_t>(ldab)
                                                 * static_cast<std::size_t>(local_col);
                        csc.values[offset] = x[idx];
                    } else {
                        // CSC format: find position in colptr/rowind
                        const int width = (meta.width > 0) ? meta.width : S->tile_size;
                        if (local_col >= width) continue;
                        const int col_start = csc.colptr[static_cast<std::size_t>(local_col)];
                        const int col_end   = csc.colptr[static_cast<std::size_t>(local_col) + 1];
                        for (int p = col_start; p < col_end; ++p) {
                            if (csc.rowind[static_cast<std::size_t>(p)] == local_row) {
                                csc.values[static_cast<std::size_t>(p)] = x[idx];
                                break;
                            }
                        }
                    }
                }
            }
#endif

            // #ifdef STILES_GPU
            //     INTERNAL_UPDATE_X_APPROACH_2_PHASE_1_CPU_TO_GPU_serial(global_index, scheme);
            // #endif
    }

    inline void update_x_semisparse_tiles_ones(int global_index, TiledMatrix **scheme, bool nested) {

        if (!scheme || !scheme[global_index]) {
            return;
        }

        TiledMatrix* S = scheme[global_index];
        const int nnz = S->original_nnz;
        const int cores = S->num_cores;
        const bool *diag_map = S->diagonal_bmapper;
        auto is_diagonal_tile = [&](int tile_idx) -> bool {
            if (!S->tileMetaCore) return false;
            const TileMetaCore& meta = S->tileMetaCore[tile_idx];
            return meta.row == meta.col;
        };
        auto uses_lapack_diagonal = [&](int tile_idx) -> bool {
            return diag_map && diag_map[tile_idx];
        };

        if (!S->chunkedDenseTiles || !S->tileMetaCore || !S->semisparseTileMetaCore) {
            return;
        }

        if (!S->tile_index_lookup || !S->withinTileRow || !S->withinTileCol) {
            sTiles::Logger::error("[update_x_semisparse_tiles] Missing lookup tables (tile_index_lookup=",
                                  static_cast<const void*>(S->tile_index_lookup),
                                  ", withinTileRow=",
                                  static_cast<const void*>(S->withinTileRow),
                                  ", withinTileCol=",
                                  static_cast<const void*>(S->withinTileCol),
                                  ") — ensure build_semisparse_tile_lookup ran successfully.");
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

        auto zero_tile = [&](int t) {
            if (t < 0 || t >= S->numActiveTiles) {
                return;
            }
            double* chunk = S->chunkedDenseTiles[t];
            if (!chunk) {
                return;
            }
            const TileMetaCore& meta = S->tileMetaCore[t];
            const int h = (meta.height > 0) ? meta.height : S->tile_size;
            const int w = (meta.width  > 0) ? meta.width  : S->tile_size;
            const SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[t];
            const bool lapack_diag = uses_lapack_diagonal(t);
            const bool is_diag = is_diagonal_tile(t);
            int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
            if ((lapack_diag || is_diag) && diag_cols <= 0) {
                diag_cols = w;
            }

            if (lapack_diag) {
                if (w <= 0 || diag_cols <= 0) {
                    return;
                }
                const std::size_t elems = static_cast<std::size_t>(diag_cols)
                                        * static_cast<std::size_t>(w);
                std::fill(chunk, chunk + elems, 0.0);
                return;
            }

            const int active_cols = (is_diag && diag_cols > 0) ? diag_cols : semi.sa;
            if (h <= 0 || active_cols <= 0) {
                return;
            }
            const std::size_t elems = static_cast<std::size_t>(h)
                                    * static_cast<std::size_t>(active_cols);
            std::fill(chunk, chunk + elems, 0.0);
        };

        auto scatter_entry = [&](int idx) {
            const int tile = S->tile_index_lookup[idx];
            if (tile < 0 || tile >= S->numActiveTiles) {
                return;
            }
            double* chunk = S->chunkedDenseTiles[tile];
            if (!chunk) {
                return;
            }
            const SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[tile];
            const bool lapack_diag = uses_lapack_diagonal(tile);
            const bool is_diag = is_diagonal_tile(tile);
            int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
            const int local_col = S->withinTileCol[idx];
            if (local_col < 0) {
                return;
            }
            const int local_row = S->withinTileRow[idx];
            if (local_row < 0) {
                return;
            }
            const TileMetaCore& meta = S->tileMetaCore[tile];
            const int h = (meta.height > 0) ? meta.height : S->tile_size;
            const int w = (meta.width  > 0) ? meta.width  : S->tile_size;

            if ((lapack_diag || is_diag) && diag_cols <= 0) {
                diag_cols = w;
            }

            // Generate small unique pseudo-random value from index to prevent cancellation
            // Uses a simple hash: value in range [0.1, 1.0)
            auto pseudo_random = [](int i) -> double {
                unsigned int x = static_cast<unsigned int>(i) + 12345;  // different seed
                x = ((x >> 16) ^ x) * 0x85ebca6b;  // different constants
                x = ((x >> 16) ^ x) * 0xc2b2ae35;
                x = (x >> 16) ^ x;
                return 0.1 + 0.9 * (static_cast<double>(x) / 4294967296.0);
            };

            // Handle LAPACK banded format for diagonal tiles first
            // (don't need acol for this case)
            if (lapack_diag) {
                if (w <= 0 || diag_cols <= 0) {
                    return;
                }
                if (local_col >= w || local_row >= h) {
                    return;
                }
                const int band = local_col - local_row;
                if (band < 0 || band >= diag_cols) {
                    return;
                }
                const int lapack_row = diag_cols - 1 - band;
                const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                         + static_cast<std::size_t>(diag_cols)
                                         * static_cast<std::size_t>(local_col);
                chunk[offset] = pseudo_random(idx);
                return;
            }

            if (local_row >= h) {
                return;
            }
            if (is_diag && diag_cols > 0) {
                const int band = local_col - local_row;
                if (band < 0 || band >= diag_cols) {
                    return;
                }
                const std::size_t offset = static_cast<std::size_t>(local_row)
                                         + static_cast<std::size_t>(band) * static_cast<std::size_t>(h);
                chunk[offset] = pseudo_random(idx);
                return;
            }

            // For non-diagonal or non-LAPACK tiles, check acol bounds now
            if (semi.acol.empty() || semi.sa <= 0 || local_col >= static_cast<int>(semi.acol.size())) {
                return;
            }
            const int active_col = semi.acol[static_cast<std::size_t>(local_col)];
            if (active_col < 0 || active_col >= semi.sa) {
                return;
            }
            const std::size_t offset = static_cast<std::size_t>(local_row)
                                     + static_cast<std::size_t>(active_col) * static_cast<std::size_t>(h);
            chunk[offset] = pseudo_random(idx);
        };

        #ifdef _OPENMP
            if (do_parallel) {
                #pragma omp parallel num_threads(cores)
                {
                    #pragma omp for schedule(static)
                    for (int t = 0; t < S->numActiveTiles; ++t) {
                        zero_tile(t);
                    }

                    #pragma omp for schedule(static)
                    for (int idx = 0; idx < nnz; ++idx) {
                        scatter_entry(idx);
                    }
                }
            } else
        #endif
            {
                for (int t = 0; t < S->numActiveTiles; ++t) {
                    zero_tile(t);
                }
                for (int idx = 0; idx < nnz; ++idx) {
                    scatter_entry(idx);
                }
            }

            scatter_nd_padding_semisparse_tiles(S);

            // #ifdef STILES_GPU
            //     INTERNAL_UPDATE_X_APPROACH_2_PHASE_1_CPU_TO_GPU_serial(global_index, scheme);
            // #endif
    }

    inline void update_symbolic_semisparse_tiles(TiledMatrix *S) {

        if (!S) {
            return;
        }

        bool nested = false;
        const int nnz = S->original_nnz;
        const int cores = S->num_cores;
        const bool *diag_map = S->diagonal_bmapper;
        auto is_diagonal_tile = [&](int tile_idx) -> bool {
            if (!S->tileMetaCore) return false;
            const TileMetaCore& meta = S->tileMetaCore[tile_idx];
            return meta.row == meta.col;
        };
        auto uses_lapack_diagonal = [&](int tile_idx) -> bool {
            return diag_map && diag_map[tile_idx];
        };

        if (!S->chunkedDenseTiles || !S->tileMetaCore || !S->semisparseTileMetaCore) {
            return;
        }

        if (!S->tile_index_lookup || !S->withinTileRow || !S->withinTileCol) {
            sTiles::Logger::error("[update_x_semisparse_tiles] Missing lookup tables (tile_index_lookup=",
                                  static_cast<const void*>(S->tile_index_lookup),
                                  ", withinTileRow=",
                                  static_cast<const void*>(S->withinTileRow),
                                  ", withinTileCol=",
                                  static_cast<const void*>(S->withinTileCol),
                                  ") — ensure build_semisparse_tile_lookup ran successfully.");
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

        auto zero_tile = [&](int t) {
            if (t < 0 || t >= S->numActiveTiles) {
                return;
            }
            double* chunk = S->chunkedDenseTiles[t];
            if (!chunk) {
                return;
            }
            const TileMetaCore& meta = S->tileMetaCore[t];
            const int h = (meta.height > 0) ? meta.height : S->tile_size;
            const int w = (meta.width  > 0) ? meta.width  : S->tile_size;
            const SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[t];
            const bool lapack_diag = uses_lapack_diagonal(t);
            const bool is_diag = is_diagonal_tile(t);
            int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
            if ((lapack_diag || is_diag) && diag_cols <= 0) {
                diag_cols = w;
            }

            if (lapack_diag) {
                if (w <= 0 || diag_cols <= 0) {
                    return;
                }
                const std::size_t elems = static_cast<std::size_t>(diag_cols)
                                        * static_cast<std::size_t>(w);
                std::fill(chunk, chunk + elems, 0.0);
                return;
            }

            const int active_cols = (is_diag && diag_cols > 0) ? diag_cols : semi.sa;
            if (h <= 0 || active_cols <= 0) {
                return;
            }
            const std::size_t elems = static_cast<std::size_t>(h)
                                    * static_cast<std::size_t>(active_cols);
            std::fill(chunk, chunk + elems, 0.0);
        };

        auto scatter_entry = [&](int idx) {
            const int tile = S->tile_index_lookup[idx];
            if (tile < 0 || tile >= S->numActiveTiles) {
                return;
            }
            double* chunk = S->chunkedDenseTiles[tile];
            if (!chunk) {
                return;
            }
            const SemisparseTileMetaCore& semi = S->semisparseTileMetaCore[tile];
            const bool lapack_diag = uses_lapack_diagonal(tile);
            const bool is_diag = is_diagonal_tile(tile);
            int diag_cols = (semi.upper_bw >= 0) ? (semi.upper_bw + 1) : 0;
            const int local_col = S->withinTileCol[idx];
            if (local_col < 0 || local_col >= static_cast<int>(semi.acol.size())) {
                return;
            }
            const int local_row = S->withinTileRow[idx];
            if (local_row < 0) {
                return;
            }
            const TileMetaCore& meta = S->tileMetaCore[tile];
            const int h = (meta.height > 0) ? meta.height : S->tile_size;
            const int w = (meta.width  > 0) ? meta.width  : S->tile_size;

            if ((lapack_diag || is_diag) && diag_cols <= 0) {
                diag_cols = w;
            }

            if (lapack_diag) {
                if (w <= 0 || diag_cols <= 0) {
                    return;
                }
                if (local_col >= w || local_row >= h) {
                    return;
                }
                const int band = local_col - local_row;
                if (band < 0 || band >= diag_cols) {
                    return;
                }
                const int lapack_row = diag_cols - 1 - band;
                const std::size_t offset = static_cast<std::size_t>(lapack_row)
                                         + static_cast<std::size_t>(diag_cols)
                                         * static_cast<std::size_t>(local_col);
                chunk[offset] = 1.0;
                return;
            }

            if (local_row >= h) {
                return;
            }
            if (is_diag && diag_cols > 0) {
                const int band = local_col - local_row;
                if (band < 0 || band >= diag_cols) {
                    return;
                }
                const std::size_t offset = static_cast<std::size_t>(local_row)
                                         + static_cast<std::size_t>(band) * static_cast<std::size_t>(h);
                chunk[offset] = 1.0;
                return;
            }

            if (semi.acol.empty() || semi.sa <= 0) {
                return;
            }
            const int active_col = semi.acol[static_cast<std::size_t>(local_col)];
            if (active_col < 0 || active_col >= semi.sa) {
                return;
            }
            const std::size_t offset = static_cast<std::size_t>(local_row)
                                     + static_cast<std::size_t>(active_col) * static_cast<std::size_t>(h);
            chunk[offset] = 1.0;
        };

        #ifdef _OPENMP
            if (do_parallel) {
                #pragma omp parallel num_threads(cores)
                {
                    #pragma omp for schedule(static)
                    for (int t = 0; t < S->numActiveTiles; ++t) {
                        zero_tile(t);
                    }

                    #pragma omp for schedule(static)
                    for (int idx = 0; idx < nnz; ++idx) {
                        scatter_entry(idx);
                    }
                }
            } else
        #endif
            {
                for (int t = 0; t < S->numActiveTiles; ++t) {
                    zero_tile(t);
                }
                for (int idx = 0; idx < nnz; ++idx) {
                    scatter_entry(idx);
                }
            }

            // #ifdef STILES_GPU
            //     INTERNAL_UPDATE_X_APPROACH_2_PHASE_1_CPU_TO_GPU_serial(global_index, scheme);
            // #endif
    }


    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /*
        1. tile_index_lookup
        2. withinTileRow
        3. withinTileCol
        4. diagonal_bmapper
        5. remainderTileSize
        6. nnz_tile_counter
    */
