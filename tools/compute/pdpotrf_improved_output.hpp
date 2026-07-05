// Improved debugging output for dpotrf_expansion_from_chol_tasks_semisparse_symbolic
// This file contains the refactored case statements with better formatting and comparison

// Copy these helper structures and lambdas into your function after line 511:

/*
// Helper to capture semisparse structure for comparison
struct SemiMetaSnapshot {
    int fa, la, sa, upper_bw;
    std::vector<int> acol, aind;
    bool operator==(const SemiMetaSnapshot &other) const {
        return fa == other.fa && la == other.la && sa == other.sa &&
               upper_bw == other.upper_bw && acol == other.acol && aind == other.aind;
    }
    bool operator!=(const SemiMetaSnapshot &other) const { return !(*this == other); }
};

auto capture_semi_meta = [&](int idx) -> SemiMetaSnapshot {
    SemiMetaSnapshot snap;
    if (!semiMeta) return snap;
    const SemisparseTileMetaCore &c = semiMeta[idx];
    snap.fa = c.fa;
    snap.la = c.la;
    snap.sa = c.sa;
    snap.upper_bw = c.upper_bw;
    snap.acol = c.acol;
    snap.aind = c.aind;
    return snap;
};

// Helper to get dense pattern as vector for comparison
auto get_dense_pattern = [&](const double *data, int idx) -> std::vector<std::vector<int>> {
    std::vector<std::vector<int>> pattern;
    if (!data || !tileMeta) return pattern;
    const TileMetaCore &meta = tileMeta[idx];
    const int rows = (meta.height > 0) ? meta.height : tile_size;
    const int cols = (meta.width > 0) ? meta.width : tile_size;
    const int ld = rows;
    const double eps = 0.0;
    pattern.resize(cols);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            double v = data[j * ld + i];
            if (std::abs(v) > eps) {
                pattern[j].push_back(i);
            }
        }
    }
    return pattern;
};

// Updated print_semi_meta with comparison support
auto print_semi_meta = [&](const char *tag, int idx, const SemiMetaSnapshot *comparison = nullptr) {
    if (!semiMeta || !tileMeta) return;

    const SemisparseTileMetaCore &c = semiMeta[idx];
    const TileMetaCore &meta = tileMeta[idx];

    const int rows = (meta.height > 0) ? meta.height : tile_size;
    const int cols = (meta.width > 0) ? meta.width : tile_size;

    bool structure_changed = false;
    if (comparison) {
        SemiMetaSnapshot current = capture_semi_meta(idx);
        structure_changed = (current != *comparison);
    }

    std::cout << "  [Semisparse] " << tag << " | tile=" << idx << " (" << meta.row << "," << meta.col << ") | dim=" << rows << "x" << cols << std::endl;
    std::cout << "    fa=" << c.fa << ", la=" << c.la << ", sa=" << c.sa << ", upper_bw=" << c.upper_bw;
    if (structure_changed) std::cout << " ← STRUCTURE MODIFIED";
    std::cout << std::endl;

    std::vector<int> col_active(cols, 0);
    for (std::size_t k = 0; k < c.acol.size(); ++k) {
        int j = c.acol[k];
        if (0 <= j && j < cols) col_active[j] = 1;
    }

    int active_cols = 0;
    for (int j = 0; j < cols; ++j) {
        if (col_active[j]) active_cols++;
    }

    if (active_cols == 0) {
        std::cout << "    All columns inactive (diagonal/dense mode)" << std::endl;
    } else {
        for (int j = 0; j < cols; ++j) {
            if (!col_active[j]) continue;

            std::cout << "    col " << std::setw(2) << j << " | [";
            bool first = true;
            for (std::size_t k = 0; k < c.acol.size(); ++k) {
                if (c.acol[k] == j) {
                    int row = (k < c.aind.size()) ? c.aind[k] : -1;
                    if (!first) std::cout << " ";
                    std::cout << row;
                    first = false;
                }
            }
            std::cout << "]" << std::endl;
        }
    }
};

// Updated print_dense_pattern with comparison support
auto print_dense_pattern = [&](const char *tag, int idx, const double *data, const std::vector<std::vector<int>> *comparison = nullptr) {
    if (!data || !tileMeta) return;
    const TileMetaCore &meta = tileMeta[idx];
    const int rows = (meta.height > 0) ? meta.height : tile_size;
    const int cols = (meta.width > 0) ? meta.width : tile_size;
    const int ld = rows;

    std::cout << "  [Dense] " << tag << " | tile=" << idx << " (" << meta.row << "," << meta.col << ") | dim=" << rows << "x" << cols << std::endl;

    const double eps = 0.0;
    for (int j = 0; j < cols; ++j) {
        std::vector<int> current_col;
        for (int i = 0; i < rows; ++i) {
            double v = data[j * ld + i];
            if (std::abs(v) > eps) {
                current_col.push_back(i);
            }
        }

        bool differs = false;
        if (comparison && j < static_cast<int>(comparison->size())) {
            differs = (current_col != (*comparison)[j]);
        }

        std::cout << "    col " << std::setw(2) << j << " | [";
        bool first = true;
        for (int row : current_col) {
            if (!first) std::cout << " ";
            std::cout << row;
            first = false;
        }
        std::cout << "]";
        if (differs) std::cout << " ← CHANGED";
        std::cout << std::endl;
    }
};
*/

// ═══════════════════════════════════════════════════════════════════════════
// CASE 1: DPOTRF - Improved Output
// ═══════════════════════════════════════════════════════════════════════════
/*
case 1: { // DPOTRF
    std::cout << "\n╔═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "║ DPOTRF: Factorize Diagonal Tile" << std::endl;
    std::cout << "║ Tile " << index1 << " (" << tileMeta[index1].row << "," << tileMeta[index1].col << ") | dim=" << tempkn << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════" << std::endl;

    // Symbolic semisparse factorization
    SemiMetaSnapshot before_semi = capture_semi_meta(index1);
    {
        SemisparseTileMetaCore &Acore = semiMeta[index1];
        print_semi_meta("BEFORE symbolic", index1);
        sTiles::core_sspotrf(Acore);
        print_semi_meta("AFTER symbolic", index1, &before_semi);
    }

    std::cout << "\n  ──────────────────────────────────────────────────────────────" << std::endl;

    // Dense numeric factorization
    double *tile_out = tiledMatrix->denseTiles[index1];
    if (!tile_out) {
        sTiles::Logger::warning("│   [ESMAIL_CHECK] null tile_out in potrf (index1=", index1, ")");
        break;
    }

    auto before_dense = get_dense_pattern(tile_out, index1);
#if PRINT_DENSE_VALUES
    sTiles::Utils::dump_dense_tile_values("BEFORE numeric", index1, tileMeta[index1], tile_out, tempkn, tempkn, ldak);
#else
    print_dense_pattern("BEFORE numeric", index1, tile_out);
#endif

    sTiles::StatusCode status = sTiles::core_dpotrf(sTiles::Uplo::Upper, tempkn, tile_out, ldak);
    (void)status;

#if PRINT_DENSE_VALUES
    sTiles::Utils::dump_dense_tile_values("AFTER numeric", index1, tileMeta[index1], tile_out, tempkn, tempkn, ldak);
#else
    print_dense_pattern("AFTER numeric", index1, tile_out, &before_dense);
#endif

    // Verification summary
    auto after_dense = get_dense_pattern(tile_out, index1);
    SemiMetaSnapshot after_semi = capture_semi_meta(index1);
    bool dense_unchanged = (before_dense == after_dense);
    bool semi_unchanged = (before_semi == after_semi);

    std::cout << "\n  [Verification]" << std::endl;
    std::cout << "    Symbolic structure: " << (semi_unchanged ? "UNCHANGED ✓" : "MODIFIED") << std::endl;
    std::cout << "    Dense pattern:      " << (dense_unchanged ? "UNCHANGED ✓" : "MODIFIED (expected for potrf)") << std::endl;

    ss_cond_set(k, k, 1);
    break;
}
*/

// ═══════════════════════════════════════════════════════════════════════════
// CASE 2: DSYRK - Improved Output
// ═══════════════════════════════════════════════════════════════════════════
/*
case 2: { // DSYRK
    ss_cond_wait(k, n, 1);

    std::cout << "\n╔═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "║ DSYRK: Symmetric Rank-K Update" << std::endl;
    std::cout << "║ C[" << index2 << "] -= B[" << index1 << "]^T * B[" << index1 << "]" << std::endl;
    std::cout << "║ Tiles: C(" << tileMeta[index2].row << "," << tileMeta[index2].col << "), B(" << tileMeta[index1].row << "," << tileMeta[index1].col << ") | dim=" << tempkn << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════" << std::endl;

    double *tile_in = tiledMatrix->denseTiles[index1];
    double *tile_out = tiledMatrix->denseTiles[index2];

    // Symbolic semisparse update
    SemiMetaSnapshot before_semi = capture_semi_meta(index2);
    {
        SemisparseTileMetaCore &Acore = semiMeta[index2];
        const SemisparseTileMetaCore &Bcore = semiMeta[index1];
#if PRINT_DENSE_VALUES
        dump_dense_tile("Input B (numeric)", index1, tile_in);
#else
        print_dense_pattern("Input B (numeric)", index1, tile_in);
#endif
        print_semi_meta("BEFORE symbolic (C)", index2);
        print_semi_meta("Input B (symbolic)", index1);
        sTiles::core_ssdsyrk(Acore, Bcore);
        print_semi_meta("AFTER symbolic (C)", index2, &before_semi);
    }

    if (!tile_out) {
        sTiles::Logger::warning("│   [ESMAIL_CHECK] null tile_out in dsyrk (index2=", index2, ")");
        break;
    }

    std::cout << "\n  ──────────────────────────────────────────────────────────────" << std::endl;

    // Dense numeric update
    auto before_dense = get_dense_pattern(tile_out, index2);
#if PRINT_DENSE_VALUES
    dump_dense_tile("BEFORE numeric (C)", index2, tile_out);
#else
    print_dense_pattern("BEFORE numeric (C)", index2, tile_out);
#endif

    sTiles::core_dsyrk(sTiles::Uplo::Upper, sTiles::Op::Trans, tempkn, tile_size, mzone, tile_in, ldan, zone, tile_out, ldak);

#if PRINT_DENSE_VALUES
    dump_dense_tile("AFTER numeric (C)", index2, tile_out);
#else
    print_dense_pattern("AFTER numeric (C)", index2, tile_out, &before_dense);
#endif

    // Verification summary
    auto after_dense = get_dense_pattern(tile_out, index2);
    SemiMetaSnapshot after_semi = capture_semi_meta(index2);
    bool dense_unchanged = (before_dense == after_dense);
    bool semi_unchanged = (before_semi == after_semi);

    std::cout << "\n  [Verification]" << std::endl;
    std::cout << "    Symbolic structure: " << (semi_unchanged ? "UNCHANGED ✓" : "MODIFIED ⚠") << std::endl;
    std::cout << "    Dense pattern:      " << (dense_unchanged ? "UNCHANGED ✓" : "MODIFIED ⚠") << std::endl;

    break;
}
*/

// ═══════════════════════════════════════════════════════════════════════════
// CASE 3: DTRSM - Improved Output
// ═══════════════════════════════════════════════════════════════════════════
/*
case 3: { // DTRSM
    ss_cond_wait(k, k, 1);

    std::cout << "\n╔═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "║ DTRSM: Triangular Solve" << std::endl;
    std::cout << "║ A[" << index1 << "] = solve(B[" << index2 << "], A[" << index1 << "])" << std::endl;
    std::cout << "║ Tiles: A(" << tileMeta[index1].row << "," << tileMeta[index1].col << "), B(" << tileMeta[index2].row << "," << tileMeta[index2].col << ")" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════" << std::endl;

    double *tile_rhs = tiledMatrix->denseTiles[index2];
    double *tile_out = tiledMatrix->denseTiles[index1];

    // Symbolic semisparse solve
    SemiMetaSnapshot before_semi = capture_semi_meta(index1);
    {
        SemisparseTileMetaCore &Acore = semiMeta[index1];
        const SemisparseTileMetaCore &Bcore = semiMeta[index2];

        print_semi_meta("BEFORE symbolic (A)", index1);
        print_semi_meta("Input B (symbolic)", index2);
        sTiles::core_ssdtrsm(Acore, Bcore);
        print_semi_meta("AFTER symbolic (A)", index1, &before_semi);
    }

    if (!tile_out) {
        sTiles::Logger::warning("│   [ESMAIL_CHECK] null tile_out in dtrsm (index1=", index1, ")");
        break;
    }

    std::cout << "\n  ──────────────────────────────────────────────────────────────" << std::endl;

    // Dense numeric solve
    auto before_dense = get_dense_pattern(tile_out, index1);
#if PRINT_DENSE_VALUES
    dump_dense_tile("BEFORE numeric (A)", index1, tile_out);
#else
    print_dense_pattern("BEFORE numeric (A)", index1, tile_out);
#endif

    sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::Trans, sTiles::Diag::NonUnit, tile_size, tempmn, zone, tile_rhs, ldak, tile_out, ldak);

#if PRINT_DENSE_VALUES
    dump_dense_tile("AFTER numeric (A)", index1, tile_out);
#else
    print_dense_pattern("AFTER numeric (A)", index1, tile_out, &before_dense);
#endif

    // Verification summary
    auto after_dense = get_dense_pattern(tile_out, index1);
    SemiMetaSnapshot after_semi = capture_semi_meta(index1);
    bool dense_unchanged = (before_dense == after_dense);
    bool semi_unchanged = (before_semi == after_semi);

    std::cout << "\n  [Verification]" << std::endl;
    std::cout << "    Symbolic structure: " << (semi_unchanged ? "UNCHANGED ✓" : "MODIFIED ⚠") << std::endl;
    std::cout << "    Dense pattern:      " << (dense_unchanged ? "UNCHANGED ✓" : "MODIFIED ⚠") << std::endl;

    ss_cond_set(m, k, 1);
    break;
}
*/

// ═══════════════════════════════════════════════════════════════════════════
// CASE 4: DGEMM - Improved Output
// ═══════════════════════════════════════════════════════════════════════════
/*
case 4: { // DGEMM
    ss_cond_wait(k, n, 1);
    ss_cond_wait(m, n, 1);

    std::cout << "\n╔═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "║ DGEMM: General Matrix Multiply" << std::endl;
    std::cout << "║ C[" << index3 << "] -= A[" << index1 << "]^T * B[" << index2 << "]" << std::endl;
    std::cout << "║ Tiles: C(" << tileMeta[index3].row << "," << tileMeta[index3].col << "), A(" << tileMeta[index1].row << "," << tileMeta[index1].col << "), B(" << tileMeta[index2].row << "," << tileMeta[index2].col << ")" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════" << std::endl;

    double *tile_a = tiledMatrix->denseTiles[index1];
    double *tile_b = tiledMatrix->denseTiles[index2];
    double *tile_out = tiledMatrix->denseTiles[index3];

    // Symbolic semisparse multiply
    SemiMetaSnapshot before_semi = capture_semi_meta(index3);
    {
        const SemisparseTileMetaCore &Acore = semiMeta[index1];
        const SemisparseTileMetaCore &Bcore = semiMeta[index2];
        SemisparseTileMetaCore &Ccore = semiMeta[index3];

        print_semi_meta("BEFORE symbolic (C)", index3);
        print_semi_meta("Input A (symbolic)", index1);
        print_semi_meta("Input B (symbolic)", index2);
        sTiles::core_ssdgemm(Acore, Bcore, Ccore);
        print_semi_meta("AFTER symbolic (C)", index3, &before_semi);
    }

    if (!tile_out) {
        sTiles::Logger::warning("│   [ESMAIL_CHECK] null tile_out in dgemm (index3=", index3, ")");
        break;
    }

    std::cout << "\n  ──────────────────────────────────────────────────────────────" << std::endl;

    // Dense numeric multiply
    auto before_dense = get_dense_pattern(tile_out, index3);
#if PRINT_DENSE_VALUES
    dump_dense_tile("BEFORE numeric (C)", index3, tile_out);
#else
    print_dense_pattern("BEFORE numeric (C)", index3, tile_out);
#endif

    sTiles::core_dgemm(sTiles::Op::Trans, sTiles::Op::NoTrans, tile_size, tempmn, tile_size, mzone, tile_a, ldan, tile_b, ldan, zone, tile_out, ldak);

#if PRINT_DENSE_VALUES
    dump_dense_tile("AFTER numeric (C)", index3, tile_out);
#else
    print_dense_pattern("AFTER numeric (C)", index3, tile_out, &before_dense);
#endif

    // Verification summary
    auto after_dense = get_dense_pattern(tile_out, index3);
    SemiMetaSnapshot after_semi = capture_semi_meta(index3);
    bool dense_unchanged = (before_dense == after_dense);
    bool semi_unchanged = (before_semi == after_semi);

    std::cout << "\n  [Verification]" << std::endl;
    std::cout << "    Symbolic structure: " << (semi_unchanged ? "UNCHANGED ✓" : "MODIFIED ⚠") << std::endl;
    std::cout << "    Dense pattern:      " << (dense_unchanged ? "UNCHANGED ✓" : "MODIFIED ⚠") << std::endl;

    break;
}
*/
