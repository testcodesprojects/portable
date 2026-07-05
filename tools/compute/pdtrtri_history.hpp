/**
 * @file pdtrtri_history.hpp
 * @brief Dead code removed from pdtrtri.cpp on 2026-07-04 (never include/compile).
 *
 * All functions below had ZERO call sites (verified repo-wide) at removal time:
 *  - pdtrtri: legacy e_trick executor (e_trick_inv is never populated in fastmode)
 *  - pdtrtri_semi_sparse_inv / _imp1 / _imp3 / _imp2_analysis / _imp1_serial:
 *    superseded selective-inverse iterations (imp4 -> pthreads_dpotri_semi_sparse_parallel,
 *    imp2_serial -> pthreads_dpotri_semi_sparse_serial are the live ones)
 *  - gemm_shape_bin: helper used only by imp2_analysis
 *  - debug_print_tile / debug_print_semisparse_tile: disabled stubs
 *  - stiles_pdtrtri_cpu / stiles_pdtrtri_cpu_no: legacy TilesDescriptor/e_trick API
 */
#if 0  // preserved for reference only - never compiled

void pdtrtri(TiledMatrix *tiledMatrix, stiles_context_t *stile)
{

    const int rank = STILES_RANK;
    const int N = tiledMatrix->dim;
    const int tile_size = tiledMatrix->tile_size;
    const double zone  = (double) 1.0;
    const double mzone = (double)-1.0;
    const int num_tasks = tiledMatrix->e_trick_size_inv[STILES_RANK];
    const int num_tiles_per_dim = (N == 0) ? 0 : (N-1)/tile_size + 1;

    int info;
    int myroutine, i, j, k;
    int index1, index2, index3; 

    in_init(num_tiles_per_dim, num_tiles_per_dim, 0);
    int global_in = 0;

    auto tile_dims = [&](int dense_idx, int& h, int& w) {
        if (tiledMatrix->tileMetaCore && dense_idx >= 0) {
            const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[dense_idx];
            h = (meta.height > 0) ? meta.height : tile_size;
            w = (meta.width  > 0) ? meta.width  : tile_size;
        } else {
            h = tile_size;
            w = tile_size;
        }
    };

    for (int in = 0; in < num_tasks; ++in) {

        myroutine = tiledMatrix->e_trick_inv[STILES_RANK][0+7*in];
        i = tiledMatrix->e_trick_inv[STILES_RANK][1+in*7];
        j = tiledMatrix->e_trick_inv[STILES_RANK][2+in*7];
        k = tiledMatrix->e_trick_inv[STILES_RANK][3+in*7];
        index1 = tiledMatrix->e_trick_inv[STILES_RANK][4+in*7];
        index2 = tiledMatrix->e_trick_inv[STILES_RANK][5+in*7];
        index3 = tiledMatrix->e_trick_inv[STILES_RANK][6+in*7];

        switch (myroutine) {

            case 1:
            {
                int h = tile_size, w = tile_size;
                tile_dims(index1, h, w);
                double* fact = tiledMatrix->denseTiles[index1];
                double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                if (fact && inv) {
                    sTiles::core_dtrsm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans,
                                       sTiles::Diag::NonUnit,
                                       h, w, 1.0,
                                       fact, h,
                                       inv,  h);
                }
                break;
            }


            case 2:
            {
                int inv_h = tile_size, inv_w = tile_size;
                tile_dims(index1, inv_h, inv_w);
                int mh = tile_size, mw = tile_size;
                tile_dims(index2, mh, mw);
                double* inv  = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                double* fact = tiledMatrix->denseTiles[index2];
                if (inv && fact) {
                    sTiles::core_dtrmm(sTiles::Side::Left,
                                       sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans,
                                       sTiles::Diag::NonUnit,
                                       mh, mw,
                                       zone,
                                       inv,  inv_h,
                                       fact, mh);
                }

                break;
            }

            case 3:
                sTiles::Control::Barrier(stile);
                global_in = in + 1;
                goto exit_first_loop;

        }

    }

    exit_first_loop:  

    for (int in = global_in; in < num_tasks; ++in) {

        myroutine = tiledMatrix->e_trick_inv[STILES_RANK][0+7*in];
        i = tiledMatrix->e_trick_inv[STILES_RANK][1+in*7];
        j = tiledMatrix->e_trick_inv[STILES_RANK][2+in*7];
        k = tiledMatrix->e_trick_inv[STILES_RANK][3+in*7];
        index1 = tiledMatrix->e_trick_inv[STILES_RANK][4+in*7];
        index2 = tiledMatrix->e_trick_inv[STILES_RANK][5+in*7];
        index3 = tiledMatrix->e_trick_inv[STILES_RANK][6+in*7];

        switch (myroutine) {

            case 4:
            {
                int h = tile_size, w = tile_size;
                tile_dims(index1, h, w);
                double* inv = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                if (inv) {
                    sTiles::core_dlauum(sTiles::Uplo::Upper, w, inv, h);
                    mirroring(h, inv, inv, h);
                }

                break;
            }

            case 5:
            {
                in_cond_wait(i, k, 2);
                int mh = tile_size, mw = tile_size;
                tile_dims(index2, mh, mw);
                int inv1_h = tile_size, inv1_w = tile_size;
                tile_dims(index1, inv1_h, inv1_w);
                int inv2_h = tile_size, inv2_w = tile_size;
                tile_dims(index2, inv2_h, inv2_w);
                double* fact = tiledMatrix->denseTiles[index2];
                double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                double* inv1 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index1] : nullptr;
                if (fact && inv2 && inv1) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::Trans,
                                       mh, mh, mw,
                                       mzone,
                                       fact, mh,
                                       inv2, inv2_h,
                                       zone,
                                       inv1, inv1_h);
                }

                break;
            }

            case 6:

                in_cond_set(i, i, 2);

                break;

            case 7:
            {
                in_cond_wait(j, k, 2);
                int mh = tile_size, mw = tile_size;
                tile_dims(index1, mh, mw);
                int inv2_h = tile_size, inv2_w = tile_size;
                tile_dims(index2, inv2_h, inv2_w);
                int inv3_h = tile_size, inv3_w = tile_size;
                tile_dims(index3, inv3_h, inv3_w);
                double* fact = tiledMatrix->denseTiles[index1];
                double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                if (fact && inv2 && inv3) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::Trans,
                                       mh, inv2_h, mw,
                                       -1.0,
                                       fact, mh,
                                       inv2, inv2_h,
                                       zone,
                                       inv3, inv3_h);
                }


                break;
            }

            case 8:
            {
                in_cond_wait(k, j, 2);
                int mh = tile_size, mw = tile_size;
                tile_dims(index1, mh, mw);
                int inv2_h = tile_size, inv2_w = tile_size;
                tile_dims(index2, inv2_h, inv2_w);
                int inv3_h = tile_size, inv3_w = tile_size;
                tile_dims(index3, inv3_h, inv3_w);
                double* fact = tiledMatrix->denseTiles[index1];
                double* inv2 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index2] : nullptr;
                double* inv3 = tiledMatrix->inverseTiles ? tiledMatrix->inverseTiles[index3] : nullptr;
                if (fact && inv2 && inv3) {
                    sTiles::core_dgemm(sTiles::Op::NoTrans,
                                       sTiles::Op::NoTrans,
                                       mh, inv2_w, mw,
                                       mzone,
                                       fact, mh,
                                       inv2, inv2_h,
                                       zone,
                                       inv3, inv3_h);
                }


                break;
            }

            case 9:

                in_cond_set(i, j, 2);
                break;

        }


    }

    in_finalize();

}

// ============================================================================

// Debug helper to print a tile (disabled for production)
static void debug_print_tile(const char* /*label*/, int /*idx*/, double* /*tile*/, int /*h*/, int /*w*/, int /*rank*/) {
    return;  // Disabled for production
    // if (rank != 0 || !tile) return;
    // std::cout << "\n[DEBUG TILE rank=" << rank << "] " << label << " idx=" << idx << " (" << h << "x" << w << "):\n";
    // for (int i = 0; i < h && i < 8; ++i) {
    //     std::cout << "  row " << i << ": ";
    //     for (int j = 0; j < w && j < 8; ++j) {
    //         std::cout << std::scientific << std::setprecision(4) << tile[i + j * h] << " ";
    //     }
    //     std::cout << "\n";
    // }
}

// ============================================================================

// Debug helper to print semisparse tile with metadata (disabled for production)
// Note: upper_bw > 0 means banded (diagonal tiles only), upper_bw == 0 with sa > 0 means active columns
static void debug_print_semisparse_tile(const char* /*label*/, int /*idx*/, double* /*tile*/, int /*h*/, int /*w*/,
                                         int /*upper_bw*/, int /*sa*/, const std::vector<int>& /*aind*/, int /*rank*/) {
    // Disabled for production - entire function body commented out
}

// ============================================================================

// ========== Semisparse selective inverse (inverse_storage_mode=1) ==========
// This version stores only selective inverse elements in semisparse format:
//   - Diagonal inverse tiles: banded format (kd+1) x h (only elements within original band)
//   - Off-diagonal inverse tiles: active-columns format h x sa (only at original sparsity positions)
//
// Algorithm:
//   1. Phase 1: Compute L^{-1} for diagonal tiles (temporary dense), then L^{-1} * L_offdiag
//   2. Phase 2: Compute A^{-1} = L^{-T} * L^{-1}, extract only elements at original positions
//
void pdtrtri_semi_sparse_inv(TiledMatrix *tiledMatrix, stiles_context_t *stile)
{
    const int N = tiledMatrix->dim;
    const int tile_size = tiledMatrix->tile_size;
    const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

    // Use pre-allocated workspace for B matrix in cases 7/8
    const int rank = STILES_RANK;
    if (rank >= tiledMatrix->num_workspaces) {
        sTiles::Logger::error("[trtri_selinv] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
        return;
    }
    double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();

    const auto &tasks   = sTiles::get_inv_tasks(tiledMatrix);
    const auto &offsets = sTiles::get_inv_task_offsets(tiledMatrix);
    size_t start = 0, end = tasks.size();
    if (!offsets.empty()) {
        const size_t r = static_cast<size_t>(STILES_RANK);
        if (r < offsets.size()) start = static_cast<size_t>(offsets[r]);
        if (r + 1 < offsets.size()) end = static_cast<size_t>(offsets[r + 1]);
    }

    in_init(num_tiles_per_dim, num_tiles_per_dim, 0);
    int global_in = -1;

    // ========== Phase 1: Compute L^{-1} for diagonal tiles ==========
    for (size_t in = start; in < end; ++in) {
        const auto &t = tasks[in];
        const int myroutine = t[0];
        const int index1 = t[4];
        const int index2 = t[5];

        switch (myroutine) {
            case 1: // TRSM: Compute L^{-1} for diagonal tile
            {
                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                if (!fact || !inv) break;

                const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                const int n = meta.height;
                const int kd = semi.upper_bw;

                if (kd == 0) {
                    // Pure diagonal L: L^{-1} = diag(1/L[i])
                    std::memset(inv, 0, static_cast<std::size_t>(n) * n * sizeof(double));
                    for (int ii = 0; ii < n; ++ii)
                        inv[ii + ii * n] = 1.0 / fact[ii];
                } else {
                    // Banded triangular solve: L * X = I => X = L^{-1}
                    const int ldab = kd + 1;
                    LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N',
                                   n, kd, n, fact, ldab, inv, n);
                }
                break;
            }
            case 2: // TRMM: Compute L^{-1} * L_offdiag (stored back in factor tile)
            {
                double* inv = tiledMatrix->chunkedInverseTiles[index1];
                double* fact = tiledMatrix->chunkedDenseTiles[index2];
                if (!inv || !fact) break;

                const int h_inv = tiledMatrix->tileMetaCore[index1].height;
                const sTiles::SemisparseTileMetaCore& semi_inv = tiledMatrix->semisparseTileMetaCore[index1];
                const int m = tiledMatrix->tileMetaCore[index2].height;
                const int sa = tiledMatrix->semisparseTileMetaCore[index2].sa;
                if (sa <= 0) break;

                if (semi_inv.upper_bw == 0) {
                    // L^{-1} is diagonal: TRMM = row scaling
                    // fact[r, c] *= inv[r, r] = 1/L[r,r]
                    for (int cc = 0; cc < sa; ++cc) {
                        double* col = fact + cc * m;
                        for (int r = 0; r < m; ++r) {
                            col[r] *= inv[r + r * m];
                        }
                    }
                } else {
                    sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                       m, sa, 1.0, inv, h_inv, fact, m);
                }
                break;
            }
            case 3: // Barrier
            {
                sTiles::Control::Barrier(stile);
                global_in = static_cast<int>(in) + 1;
                goto exit_phase1_semisparse_selinv;
            }
            default:
                break;
        }
    }

exit_phase1_semisparse_selinv:
    if (global_in < 0) global_in = static_cast<int>(start);

    // ========== Phase 2: Compute selective inverse entries ==========
    for (size_t in = static_cast<size_t>(global_in); in < end; ++in) {
        const auto &t = tasks[in];
        const int myroutine = t[0];
        const int i = t[1];
        const int j = t[2];
        const int k = t[3];
        const int index1 = t[4];
        const int index2 = t[5];
        const int index3 = t[6];

        switch (myroutine) {
            case 4: // LAUUM + mirror: Compute L^{-T} * L^{-1} for diagonal tile
            {
                double* inv = tiledMatrix->chunkedInverseTiles[index1];
                if (!inv) break;

                const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                const int n = meta.height;

                if (semi.upper_bw == 0) {
                    // L^{-1} is diagonal: L^{-T}*L^{-1} = diag(d[i]^2)
                    // Case 1 already set inv to diagonal (off-diag = 0), just square in-place
                    for (int ii = 0; ii < n; ++ii)
                        inv[ii + ii * n] *= inv[ii + ii * n];
                } else {
                    sTiles::core_dlauum(sTiles::Uplo::Upper, n, inv, n);
                    mirroring(n, inv, inv, n);
                }
                break;
            }
            case 5: // Diagonal tile update: inv1 -= fact * inv2^T
            {
                in_cond_wait(i, k, 2);

                double* fact = tiledMatrix->chunkedDenseTiles[index2];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv1 = tiledMatrix->chunkedInverseTiles[index1];
                if (!fact || !inv2 || !inv1) break;

                const int m = tiledMatrix->tileMetaCore[index2].height;
                const int sa2 = tiledMatrix->semisparseTileMetaCore[index2].sa;
                const int n1 = tiledMatrix->tileMetaCore[index1].height;

                if (sa2 <= 0) break;

                const int kd1 = tiledMatrix->semisparseTileMetaCore[index1].upper_bw;
                if (kd1 == 0) {
                    // Diagonal tile: only diagonal elements needed
                    // inv1[r,r] -= dot(fact[r,:], inv2[r,:])
                    // O(n1 * sa2) instead of O(n1^2 * sa2)
                    for (int r = 0; r < n1; ++r) {
                        double dot = 0.0;
                        for (int kk = 0; kk < sa2; ++kk)
                            dot += fact[r + kk * m] * inv2[r + kk * m];
                        inv1[r + r * n1] -= dot;
                    }
                } else {
                    // Nearly full: use BLAS dgemm (vectorized, cache-optimized)
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                n1, n1, sa2, -1.0,
                                fact, m,
                                inv2, m,
                                1.0, inv1, n1);
                }

                break;
            }
            case 6:
                in_cond_set(i, i, 2);
                break;

            case 7: // GEMM: Off-diagonal update: inv3 -= fact * inv2^T
            {
                // fact(m×sa1) active-cols, inv2 dense(h×h) or active-cols(h×sa2), inv3 active-cols(m×sa3)
                // Build B(sa1×sa3), then inv3 -= fact * B via BLAS dgemm
                in_cond_wait(j, k, 2);

                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                if (!fact || !inv2 || !inv3) break;

                const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                const int sa1 = semi1.sa;
                const int sa3 = semi3.sa;

                if (sa1 <= 0 || sa3 <= 0) break;

                // B(sa1 × sa3): B[jj,cc] = inv2[semi3.aind[cc], col_for(semi1.aind[jj])]
                double* B = tmp_tile;
                const bool inv2_is_diag = (tiledMatrix->tileMetaCore[index2].row == tiledMatrix->tileMetaCore[index2].col);
                const int* aind1 = semi1.aind.data();
                const int* aind3 = semi3.aind.data();

                if (inv2_is_diag) {
                    // inv2 dense h×h — cc outer for contiguous B column writes
                    for (int cc = 0; cc < sa3; ++cc) {
                        const int c_row = aind3[cc];
                        double* B_dst = B + cc * sa1;  // contiguous column of B
                        for (int jj = 0; jj < sa1; ++jj) {
                            B_dst[jj] = inv2[c_row + aind1[jj] * m_inv2];
                        }
                    }
                } else {
                    // inv2 active-cols — precompute fact-column → inv2-stored-column mapping
                    const int sa2 = semi2.sa;
                    const int* acol2 = semi2.acol.data();
                    const int acol2_sz = static_cast<int>(semi2.acol.size());

                    // Map sa1 entries: jj → stored column index in inv2 (or -1)
                    // Use end of workspace as scratch for the mapping (sa1 ints fit easily)
                    int* col_map = reinterpret_cast<int*>(B + static_cast<std::size_t>(sa1) * sa3);
                    int valid_count = 0;
                    for (int jj = 0; jj < sa1; ++jj) {
                        const int k = aind1[jj];
                        const int idx = (k >= 0 && k < acol2_sz) ? acol2[k] : -1;
                        col_map[jj] = (idx >= 0 && idx < sa2) ? idx : -1;
                        if (col_map[jj] >= 0) ++valid_count;
                    }
                    if (valid_count == 0) { std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double)); }
                    else {
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                const int stored = col_map[jj];
                                B_dst[jj] = (stored >= 0) ? inv2[c_row + stored * m_inv2] : 0.0;
                            }
                        }
                    }
                }

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);
                break;
            }
            case 8: // GEMM: Off-diagonal update: inv3 -= fact * inv2
            {
                // fact(m×sa1) active-cols, inv2 dense(h×h) or active-cols(h×sa2), inv3 active-cols(m×sa3)
                // Build B(sa1×sa3), then inv3 -= fact * B via BLAS dgemm
                in_cond_wait(k, j, 2);

                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                if (!fact || !inv2 || !inv3) break;

                const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                const int sa1 = semi1.sa;
                const int sa3 = semi3.sa;

                if (sa1 <= 0 || sa3 <= 0) break;

                // B(sa1 × sa3): B[jj,cc] = inv2[semi1.aind[jj], col_for(semi3.aind[cc])]
                double* B = tmp_tile;
                const bool inv2_is_diag = (tiledMatrix->tileMetaCore[index2].row == tiledMatrix->tileMetaCore[index2].col);
                const int* aind1 = semi1.aind.data();
                const int* aind3 = semi3.aind.data();

                if (inv2_is_diag) {
                    // inv2 dense h×h — iterate column-first for contiguous inv2 access
                    for (int cc = 0; cc < sa3; ++cc) {
                        const int c_col = aind3[cc];
                        const double* inv2_col = inv2 + c_col * m_inv2;
                        double* B_dst = B + cc * sa1;
                        for (int jj = 0; jj < sa1; ++jj) {
                            B_dst[jj] = inv2_col[aind1[jj]];
                        }
                    }
                } else {
                    // inv2 active-cols — precompute inv3-column → inv2-stored-column mapping
                    const int sa2 = semi2.sa;
                    const int* acol2 = semi2.acol.data();
                    const int acol2_sz = static_cast<int>(semi2.acol.size());

                    // Map sa3 entries: cc → stored column index in inv2 (or -1)
                    int* col_map = reinterpret_cast<int*>(B + static_cast<std::size_t>(sa1) * sa3);
                    int valid_count = 0;
                    for (int cc = 0; cc < sa3; ++cc) {
                        const int c = aind3[cc];
                        const int idx = (c >= 0 && c < acol2_sz) ? acol2[c] : -1;
                        col_map[cc] = (idx >= 0 && idx < sa2) ? idx : -1;
                        if (col_map[cc] >= 0) ++valid_count;
                    }
                    if (valid_count == 0) { std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double)); }
                    else {
                        for (int cc = 0; cc < sa3; ++cc) {
                            double* B_dst = B + cc * sa1;
                            const int stored = col_map[cc];
                            if (stored >= 0) {
                                const double* inv2_col = inv2 + stored * m_inv2;
                                for (int jj = 0; jj < sa1; ++jj) {
                                    B_dst[jj] = inv2_col[aind1[jj]];
                                }
                            } else {
                                std::memset(B_dst, 0, sa1 * sizeof(double));
                            }
                        }
                    }
                }

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);
                break;
            }
            case 9:
                in_cond_set(i, j, 2);
                break;

            default:
                break;
        }
    }

    in_finalize();

    // Stats: count diagonal tiles and how many are pure diagonal (kd==0)
    if (STILES_RANK == 0) {
        int n_diag = 0, n_kd0 = 0, n_kd_le2 = 0, n_full = 0, kd_max = 0;
        const int num_active = tiledMatrix->numActiveTiles;
        for (int t = 0; t < num_active; ++t) {
            if (tiledMatrix->tileMetaCore[t].row == tiledMatrix->tileMetaCore[t].col) {
                ++n_diag;
                const int kd = tiledMatrix->semisparseTileMetaCore[t].upper_bw;
                const int h = tiledMatrix->tileMetaCore[t].height;
                if (kd > kd_max) kd_max = kd;
                if (kd == 0) ++n_kd0;
                if (kd <= 2) ++n_kd_le2;
                if (kd >= h - 3) ++n_full;
            }
        }
        // std::cout << "  [pdtrtri_semi_sparse_inv] diagonal tiles: " << n_diag
        //           << "  kd=0: " << n_kd0
        //           << "  kd<=2: " << n_kd_le2
        //           << "  nearly_full(kd>=h-3): " << n_full
        //           << "  kd_max=" << kd_max << "\n";
    }
}

// ============================================================================

void pdtrtri_semi_sparse_inv_imp1(TiledMatrix *tiledMatrix, stiles_context_t *stile)
{
    const int N = tiledMatrix->dim;
    const int tile_size = tiledMatrix->tile_size;
    const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

    // Use pre-allocated workspace for B matrix in cases 7/8, and diagonal extraction in case 2
    const int rank = STILES_RANK;
    if (rank >= tiledMatrix->num_workspaces) {
        sTiles::Logger::error("[trtri_selinv_imp1] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
        return;
    }
    double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();

    // Precomputed gather info (may be null if not built)
    const int32_t* gather_index = (tiledMatrix->inv_gather_index) ? tiledMatrix->inv_gather_index->data() : nullptr;
    const int32_t* gather_packed = (tiledMatrix->inv_gather_packed) ? tiledMatrix->inv_gather_packed->data() : nullptr;
    const bool has_gather_info = (gather_index != nullptr && gather_packed != nullptr
        && tiledMatrix->inv_gather_index && static_cast<int>(tiledMatrix->inv_gather_index->size()) >= static_cast<int>(tiledMatrix->inv_tasks ? tiledMatrix->inv_tasks->size() : 0) * 3);

    const auto &tasks   = sTiles::get_inv_tasks(tiledMatrix);
    const auto &offsets = sTiles::get_inv_task_offsets(tiledMatrix);
    size_t start = 0, end = tasks.size();
    if (!offsets.empty()) {
        const size_t r = static_cast<size_t>(STILES_RANK);
        if (r < offsets.size()) start = static_cast<size_t>(offsets[r]);
        if (r + 1 < offsets.size()) end = static_cast<size_t>(offsets[r + 1]);
    }

    in_init(num_tiles_per_dim, num_tiles_per_dim, 0);
    int global_in = -1;

    // ========== Phase 1: Compute L^{-1} for diagonal tiles ==========
    for (size_t in = start; in < end; ++in) {
        const auto &t = tasks[in];
        const int myroutine = t[0];
        const int index1 = t[4];
        const int index2 = t[5];

        switch (myroutine) {
            case 1: // TRSM: Compute L^{-1} for diagonal tile
            {
                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                if (!fact || !inv) { break; }

                const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                const int n = meta.height;
                const int kd = semi.upper_bw;

                if (kd == 0) {
                    // Pure diagonal L: L^{-1} = diag(1/L[i])
                    std::memset(inv, 0, static_cast<std::size_t>(n) * n * sizeof(double));
                    for (int ii = 0; ii < n; ++ii)
                        inv[ii + ii * n] = 1.0 / fact[ii];
                } else {
                    // Banded triangular solve: L * X = I => X = L^{-1}
                    const int ldab = kd + 1;
                    LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N',
                                   n, kd, n, fact, ldab, inv, n);
                }
                break;
            }
            case 2: // TRMM: Compute L^{-1} * L_offdiag (stored back in factor tile)
            {
                double* inv = tiledMatrix->chunkedInverseTiles[index1];
                double* fact = tiledMatrix->chunkedDenseTiles[index2];
                if (!inv || !fact) { break; }

                const int h_inv = tiledMatrix->tileMetaCore[index1].height;
                const sTiles::SemisparseTileMetaCore& semi_inv = tiledMatrix->semisparseTileMetaCore[index1];
                const int m = tiledMatrix->tileMetaCore[index2].height;
                const int sa = tiledMatrix->semisparseTileMetaCore[index2].sa;
                if (sa <= 0) { break; }

                if (semi_inv.upper_bw == 0) {
                    // L^{-1} is diagonal: TRMM = row scaling
                    // Extract diagonal into contiguous array to avoid stride-(m+1) access
                    double* diag = tmp_tile;
                    for (int r = 0; r < m; ++r)
                        diag[r] = inv[r + r * m];
                    for (int cc = 0; cc < sa; ++cc) {
                        double* col = fact + cc * m;
                        #pragma omp simd
                        for (int r = 0; r < m; ++r) {
                            col[r] *= diag[r];
                        }
                    }
                } else {
                    sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                       m, sa, 1.0, inv, h_inv, fact, m);
                }
                break;
            }
            case 3: // Barrier
            {
                sTiles::Control::Barrier(stile);
                global_in = static_cast<int>(in) + 1;
                goto exit_phase1_semisparse_selinv;
            }
            default:
                break;
        }
    }

exit_phase1_semisparse_selinv:
    if (global_in < 0) global_in = static_cast<int>(start);

    // Hoisted base pointers + bounds for the per-iteration prefetch block.
    double** const cdt = tiledMatrix->chunkedDenseTiles;
    double** const cit = tiledMatrix->chunkedInverseTiles;
    const auto* const ssm = tiledMatrix->semisparseTileMetaCore;
    const int ntiles_bounds = tiledMatrix->numActiveTiles;

    // ========== Phase 2: Compute selective inverse entries ==========
    for (size_t in = static_cast<size_t>(global_in); in < end; ++in) {
        const auto &t = tasks[in];
        const int myroutine = t[0];
        const int i = t[1];
        const int j = t[2];
        const int k = t[3];
        const int index1 = t[4];
        const int index2 = t[5];
        const int index3 = t[6];

        // Prefetch next task's tile/meta backing storage. The gather_packed
        // indirection is the unpredictable load that benefits most. Prefetch is
        // a hint only — safe under the cond_wait sync protocol below.
        if (in + 1 < end) {
            const auto &nt = tasks[in + 1];
            const int n1 = nt[4], n2 = nt[5], n3 = nt[6];
            if (n1 >= 0 && n1 < ntiles_bounds) {
                if (cdt[n1]) __builtin_prefetch(cdt[n1], 0, 2);
                if (cit[n1]) __builtin_prefetch(cit[n1], 1, 2);
                __builtin_prefetch(&ssm[n1], 0, 3);
            }
            if (n2 >= 0 && n2 < ntiles_bounds) {
                if (cdt[n2]) __builtin_prefetch(cdt[n2], 0, 2);
                if (cit[n2]) __builtin_prefetch(cit[n2], 0, 2);
                __builtin_prefetch(&ssm[n2], 0, 3);
            }
            if (n3 >= 0 && n3 < ntiles_bounds && cit[n3]) {
                __builtin_prefetch(cit[n3], 1, 2);
            }
            if (has_gather_info) {
                const int gi = static_cast<int>(in + 1) * 3;
                __builtin_prefetch(&gather_index[gi], 0, 3);
                const int next_off = gather_index[gi];
                if (next_off >= 0) __builtin_prefetch(gather_packed + next_off, 0, 3);
            }
        }

        switch (myroutine) {
            case 7: // GEMM: Off-diagonal update: inv3 -= fact * inv2^T
            {
                in_cond_wait(j, k, 2);

                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                if (!fact || !inv2 || !inv3) { break; }

                const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                const int sa1 = semi1.sa;
                const int sa3 = semi3.sa;

                double* B = tmp_tile;
                const int* aind3 = semi3.aind.data();

                // Use precomputed gather info if available, otherwise fall back
                if (has_gather_info) {
                    const int gi_base = static_cast<int>(in) * 3;
                    const int data_off = gather_index[gi_base];
                    const int n_valid  = gather_index[gi_base + 1];
                    const int flags    = gather_index[gi_base + 2];

                    if (flags == 0) {
                        // skip: no overlap
                        break;
                    }

                    if (flags == 1) {
                        // all_valid: offsets stored sequentially, branch-free
                        const int32_t* col_offsets = gather_packed + data_off;
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < n_valid; ++jj) {
                                B_dst[jj] = inv2[c_row + col_offsets[jj]];
                            }
                        }
                    } else {
                        // partial: interleaved [valid_jj, col_offset, ...]
                        const int32_t* pairs = gather_packed + data_off;
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int v = 0; v < n_valid; ++v) {
                                const int jj     = pairs[v * 2];
                                const int offset = pairs[v * 2 + 1];
                                B_dst[jj] = inv2[c_row + offset];
                            }
                        }
                    }
                } else {
                    // Fallback: compute col_map at runtime (original path)
                    const int* aind1 = semi1.aind.data();
                    const int sa2 = semi2.sa;
                    const int* acol2 = semi2.acol.data();
                    const int acol2_sz = static_cast<int>(semi2.acol.size());

                    int* col_map = reinterpret_cast<int*>(B + static_cast<std::size_t>(sa1) * sa3);
                    int valid_count = 0;
                    for (int jj = 0; jj < sa1; ++jj) {
                        const int k = aind1[jj];
                        const int idx = (k >= 0 && k < acol2_sz) ? acol2[k] : -1;
                        col_map[jj] = (idx >= 0 && idx < sa2) ? idx : -1;
                        if (col_map[jj] >= 0) ++valid_count;
                    }
                    if (valid_count == 0) {
                        break;
                    }
                    if (valid_count == sa1) {
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2[c_row + col_map[jj] * m_inv2];
                            }
                        }
                    } else {
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                const int stored = col_map[jj];
                                if (stored >= 0) B_dst[jj] = inv2[c_row + stored * m_inv2];
                            }
                        }
                    }
                }

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                break;
            }
            case 8: // GEMM: Off-diagonal update: inv3 -= fact * inv2
            {
                in_cond_wait(k, j, 2);

                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                if (!fact || !inv2 || !inv3) { break; }

                const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                const int sa1 = semi1.sa;
                const int sa3 = semi3.sa;

                if (sa1 <= 0 || sa3 <= 0) { break; }

                double* B = tmp_tile;
                const int* aind1 = semi1.aind.data();
                const int* aind3 = semi3.aind.data();

                // Use precomputed gather info if available
                if (has_gather_info) {
                    const int gi_base = static_cast<int>(in) * 3;
                    const int data_off = gather_index[gi_base];
                    const int n_valid  = gather_index[gi_base + 1];
                    const int flags    = gather_index[gi_base + 2];

                    if (flags == 0) {
                        // skip: no overlap
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                    } else if (flags == 3) {
                        // diagonal inv2: direct access, no col_map needed
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_col = aind3[cc];
                            const double* inv2_col = inv2 + c_col * m_inv2;
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2_col[aind1[jj]];
                            }
                        }
                    } else if (flags == 1) {
                        // all_valid: precomputed offsets, branch-free
                        const int32_t* col_offsets = gather_packed + data_off;
                        for (int cc = 0; cc < n_valid; ++cc) {
                            const double* inv2_col = inv2 + col_offsets[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2_col[aind1[jj]];
                            }
                        }
                    } else {
                        // partial: interleaved [valid_cc, col_offset, ...]
                        const int32_t* pairs = gather_packed + data_off;
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        for (int v = 0; v < n_valid; ++v) {
                            const int cc     = pairs[v * 2];
                            const int offset = pairs[v * 2 + 1];
                            const double* inv2_col = inv2 + offset;
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2_col[aind1[jj]];
                            }
                        }
                    }
                } else {
                    // Fallback: compute at runtime (original path)
                    const bool inv2_is_diag = (tiledMatrix->tileMetaCore[index2].row == tiledMatrix->tileMetaCore[index2].col);

                    if (inv2_is_diag) {
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_col = aind3[cc];
                            const double* inv2_col = inv2 + c_col * m_inv2;
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2_col[aind1[jj]];
                            }
                        }
                    } else {
                        const int sa2 = semi2.sa;
                        const int* acol2 = semi2.acol.data();
                        const int acol2_sz = static_cast<int>(semi2.acol.size());

                        int* col_map = reinterpret_cast<int*>(B + static_cast<std::size_t>(sa1) * sa3);
                        int valid_count = 0;
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c = aind3[cc];
                            const int idx = (c >= 0 && c < acol2_sz) ? acol2[c] : -1;
                            col_map[cc] = (idx >= 0 && idx < sa2) ? idx : -1;
                            if (col_map[cc] >= 0) ++valid_count;
                        }
                        if (valid_count == 0) { std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double)); }
                        else {
                            for (int cc = 0; cc < sa3; ++cc) {
                                const int stored = col_map[cc];
                                if (stored >= 0) {
                                    const double* inv2_col = inv2 + stored * m_inv2;
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj) {
                                        B_dst[jj] = inv2_col[aind1[jj]];
                                    }
                                } else {
                                    std::memset(B + cc * sa1, 0, sa1 * sizeof(double));
                                }
                            }
                        }
                    }
                }

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                break;
            }
            case 4: // LAUUM + mirror: Compute L^{-T} * L^{-1} for diagonal tile
            {
                double* inv = tiledMatrix->chunkedInverseTiles[index1];
                if (!inv) { break; }

                const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                const int n = meta.height;

                if (semi.upper_bw == 0) {
                    for (int ii = 0; ii < n; ++ii)
                        inv[ii + ii * n] *= inv[ii + ii * n];
                } else {
                    sTiles::core_dlauum(sTiles::Uplo::Upper, n, inv, n);
                    mirroring(n, inv, inv, n);
                }
                break;
            }
            case 5: // Diagonal tile update: inv1 -= fact * inv2^T
            {
                in_cond_wait(i, k, 2);

                double* fact = tiledMatrix->chunkedDenseTiles[index2];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv1 = tiledMatrix->chunkedInverseTiles[index1];
                if (!fact || !inv2 || !inv1) { break; }

                const int m = tiledMatrix->tileMetaCore[index2].height;
                const int sa2 = tiledMatrix->semisparseTileMetaCore[index2].sa;
                const int n1 = tiledMatrix->tileMetaCore[index1].height;

                if (sa2 <= 0) { break; }

                const int kd1 = tiledMatrix->semisparseTileMetaCore[index1].upper_bw;
                if (kd1 == 0) {
                    for (int kk = 0; kk < sa2; ++kk) {
                        const double* f_col = fact + kk * m;
                        const double* i_col = inv2 + kk * m;
                        for (int r = 0; r < n1; ++r) {
                            inv1[r + r * n1] -= f_col[r] * i_col[r];
                        }
                    }
                } else {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                n1, n1, sa2, -1.0,
                                fact, m,
                                inv2, m,
                                1.0, inv1, n1);
                }

                break;
            }
            case 6:
                in_cond_set(i, i, 2);
                break;
            case 9:
                in_cond_set(i, j, 2);
                break;

            default:
                break;
        }
    }

    in_finalize();

    // Stats: count diagonal tiles and how many are pure diagonal (kd==0)
    if (STILES_RANK == 0) {
        int n_diag = 0, n_kd0 = 0, n_kd_le2 = 0, n_full = 0, kd_max = 0;
        const int num_active = tiledMatrix->numActiveTiles;
        for (int t = 0; t < num_active; ++t) {
            if (tiledMatrix->tileMetaCore[t].row == tiledMatrix->tileMetaCore[t].col) {
                ++n_diag;
                const int kd = tiledMatrix->semisparseTileMetaCore[t].upper_bw;
                const int h = tiledMatrix->tileMetaCore[t].height;
                if (kd > kd_max) kd_max = kd;
                if (kd == 0) ++n_kd0;
                if (kd <= 2) ++n_kd_le2;
                if (kd >= h - 3) ++n_full;
            }
        }
        // std::cout << "  [pdtrtri_semi_sparse_inv] diagonal tiles: " << n_diag
        //           << "  kd=0: " << n_kd0
        //           << "  kd<=2: " << n_kd_le2
        //           << "  nearly_full(kd>=h-3): " << n_full
        //           << "  kd_max=" << kd_max << "\n";
    }
}

// ============================================================================

void pdtrtri_semi_sparse_inv_imp3(TiledMatrix *tiledMatrix, stiles_context_t *stile)
{
    const int N = tiledMatrix->dim;
    const int tile_size = tiledMatrix->tile_size;
    const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

    // Use pre-allocated workspace for B matrix in cases 7/8, and diagonal extraction in case 2
    const int rank = STILES_RANK;
    if (rank >= tiledMatrix->num_workspaces) {
        sTiles::Logger::error("[trtri_selinv_imp1] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
        return;
    }
    double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();

    // Precomputed gather info (may be null if not built)
    const int32_t* gather_index = (tiledMatrix->inv_gather_index) ? tiledMatrix->inv_gather_index->data() : nullptr;
    const int32_t* gather_packed = (tiledMatrix->inv_gather_packed) ? tiledMatrix->inv_gather_packed->data() : nullptr;
    const bool has_gather_info = (gather_index != nullptr && gather_packed != nullptr
        && tiledMatrix->inv_gather_index && static_cast<int>(tiledMatrix->inv_gather_index->size()) >= static_cast<int>(tiledMatrix->inv_tasks ? tiledMatrix->inv_tasks->size() : 0) * 3);

    const auto &tasks   = sTiles::get_inv_tasks(tiledMatrix);
    const auto &offsets = sTiles::get_inv_task_offsets(tiledMatrix);
    size_t start = 0, end = tasks.size();
    if (!offsets.empty()) {
        const size_t r = static_cast<size_t>(STILES_RANK);
        if (r < offsets.size()) start = static_cast<size_t>(offsets[r]);
        if (r + 1 < offsets.size()) end = static_cast<size_t>(offsets[r + 1]);
    }

    in_init(num_tiles_per_dim, num_tiles_per_dim, 0);
    int global_in = -1;

    constexpr int FUSE_THRESH = 8;

    // ========== Phase 1: Compute L^{-1} for diagonal tiles ==========
    for (size_t in = start; in < end; ++in) {
        const auto &t = tasks[in];
        const int myroutine = t[0];
        const int index1 = t[4];
        const int index2 = t[5];

        switch (myroutine) {
            case 1: // TRSM: Compute L^{-1} for diagonal tile
            {
                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                if (!fact || !inv) { break; }

                const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                const int n = meta.height;
                const int kd = semi.upper_bw;

                if (kd == 0) {
                    // Pure diagonal L: L^{-1} = diag(1/L[i])
                    std::memset(inv, 0, static_cast<std::size_t>(n) * n * sizeof(double));
                    for (int ii = 0; ii < n; ++ii)
                        inv[ii + ii * n] = 1.0 / fact[ii];
                } else {
                    // Banded triangular solve: L * X = I => X = L^{-1}
                    const int ldab = kd + 1;
                    LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N',
                                   n, kd, n, fact, ldab, inv, n);
                }
                break;
            }
            case 2: // TRMM: Compute L^{-1} * L_offdiag (stored back in factor tile)
            {
                double* inv = tiledMatrix->chunkedInverseTiles[index1];
                double* fact = tiledMatrix->chunkedDenseTiles[index2];
                if (!inv || !fact) { break; }

                const int h_inv = tiledMatrix->tileMetaCore[index1].height;
                const sTiles::SemisparseTileMetaCore& semi_inv = tiledMatrix->semisparseTileMetaCore[index1];
                const int m = tiledMatrix->tileMetaCore[index2].height;
                const int sa = tiledMatrix->semisparseTileMetaCore[index2].sa;
                if (sa <= 0) { break; }

                if (semi_inv.upper_bw == 0) {
                    // L^{-1} is diagonal: TRMM = row scaling
                    // Extract diagonal into contiguous array to avoid stride-(m+1) access
                    double* diag = tmp_tile;
                    for (int r = 0; r < m; ++r)
                        diag[r] = inv[r + r * m];
                    for (int cc = 0; cc < sa; ++cc) {
                        double* col = fact + cc * m;
                        #pragma omp simd
                        for (int r = 0; r < m; ++r) {
                            col[r] *= diag[r];
                        }
                    }
                } else {
                    sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                       m, sa, 1.0, inv, h_inv, fact, m);
                }
                break;
            }
            case 3: // Barrier
            {
                sTiles::Control::Barrier(stile);
                global_in = static_cast<int>(in) + 1;
                goto exit_phase1_semisparse_selinv;
            }
            default:
                break;
        }
    }

exit_phase1_semisparse_selinv:
    if (global_in < 0) global_in = static_cast<int>(start);

    // Hoisted base pointers + bounds for the per-iteration prefetch block.
    double** const cdt = tiledMatrix->chunkedDenseTiles;
    double** const cit = tiledMatrix->chunkedInverseTiles;
    const auto* const ssm = tiledMatrix->semisparseTileMetaCore;
    const int ntiles_bounds = tiledMatrix->numActiveTiles;

    // ========== Phase 2: Compute selective inverse entries ==========
    for (size_t in = static_cast<size_t>(global_in); in < end; ++in) {
        const auto &t = tasks[in];
        const int myroutine = t[0];
        const int i = t[1];
        const int j = t[2];
        const int k = t[3];
        const int index1 = t[4];
        const int index2 = t[5];
        const int index3 = t[6];

        // Prefetch next task's tile/meta backing storage. The gather_packed
        // indirection is the unpredictable load that benefits most. Prefetch is
        // a hint only — safe under the cond_wait sync protocol below.
        if (in + 1 < end) {
            const auto &nt = tasks[in + 1];
            const int n1 = nt[4], n2 = nt[5], n3 = nt[6];
            if (n1 >= 0 && n1 < ntiles_bounds) {
                if (cdt[n1]) __builtin_prefetch(cdt[n1], 0, 2);
                if (cit[n1]) __builtin_prefetch(cit[n1], 1, 2);
                __builtin_prefetch(&ssm[n1], 0, 3);
            }
            if (n2 >= 0 && n2 < ntiles_bounds) {
                if (cdt[n2]) __builtin_prefetch(cdt[n2], 0, 2);
                if (cit[n2]) __builtin_prefetch(cit[n2], 0, 2);
                __builtin_prefetch(&ssm[n2], 0, 3);
            }
            if (n3 >= 0 && n3 < ntiles_bounds && cit[n3]) {
                __builtin_prefetch(cit[n3], 1, 2);
            }
            if (has_gather_info) {
                const int gi = static_cast<int>(in + 1) * 3;
                __builtin_prefetch(&gather_index[gi], 0, 3);
                const int next_off = gather_index[gi];
                if (next_off >= 0) __builtin_prefetch(gather_packed + next_off, 0, 3);
            }
        }

        switch (myroutine) {
            case 7: // GEMM: Off-diagonal update: inv3 -= fact * inv2^T
            {
                in_cond_wait(j, k, 2);

                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                if (!fact || !inv2 || !inv3) { break; }

                const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                const int sa1 = semi1.sa;
                const int sa3 = semi3.sa;

                const int* aind3 = semi3.aind.data();

                // ── Fused gather+compute for small tiles ──
                if (sa1 <= FUSE_THRESH && sa3 <= FUSE_THRESH && has_gather_info) {
                    const int gi_base = static_cast<int>(in) * 3;
                    const int data_off = gather_index[gi_base];
                    const int n_valid  = gather_index[gi_base + 1];
                    const int flags    = gather_index[gi_base + 2];

                    if (flags == 0) { break; }

                    if (flags == 1) {
                        const int32_t* col_offsets = gather_packed + data_off;
                        for (int jj = 0; jj < n_valid; ++jj) {
                            const int col_off = col_offsets[jj];
                            const double* fact_col = fact + jj * m_fact;
                            for (int cc = 0; cc < sa3; ++cc) {
                                const double b_val = inv2[aind3[cc] + col_off];
                                double* inv3_col = inv3 + cc * m_inv3;
                                #pragma omp simd
                                for (int r = 0; r < m_inv3; ++r)
                                    inv3_col[r] -= fact_col[r] * b_val;
                            }
                        }
                    } else {
                        const int32_t* pr = gather_packed + data_off;
                        for (int v = 0; v < n_valid; ++v) {
                            const int jj     = pr[v * 2];
                            const int offset = pr[v * 2 + 1];
                            const double* fact_col = fact + jj * m_fact;
                            for (int cc = 0; cc < sa3; ++cc) {
                                const double b_val = inv2[aind3[cc] + offset];
                                double* inv3_col = inv3 + cc * m_inv3;
                                #pragma omp simd
                                for (int r = 0; r < m_inv3; ++r)
                                    inv3_col[r] -= fact_col[r] * b_val;
                            }
                        }
                    }
                    break;
                }

                // ── Standard gather + BLAS path ──
                double* B = tmp_tile;

                if (has_gather_info) {
                    const int gi_base = static_cast<int>(in) * 3;
                    const int data_off = gather_index[gi_base];
                    const int n_valid  = gather_index[gi_base + 1];
                    const int flags    = gather_index[gi_base + 2];

                    if (flags == 0) {
                        break;
                    }

                    if (flags == 1) {
                        const int32_t* col_offsets = gather_packed + data_off;
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < n_valid; ++jj) {
                                B_dst[jj] = inv2[c_row + col_offsets[jj]];
                            }
                        }
                    } else {
                        const int32_t* pairs = gather_packed + data_off;
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int v = 0; v < n_valid; ++v) {
                                const int jj     = pairs[v * 2];
                                const int offset = pairs[v * 2 + 1];
                                B_dst[jj] = inv2[c_row + offset];
                            }
                        }
                    }
                } else {
                    const int* aind1 = semi1.aind.data();
                    const int sa2 = semi2.sa;
                    const int* acol2 = semi2.acol.data();
                    const int acol2_sz = static_cast<int>(semi2.acol.size());

                    int* col_map = reinterpret_cast<int*>(B + static_cast<std::size_t>(sa1) * sa3);
                    int valid_count = 0;
                    for (int jj = 0; jj < sa1; ++jj) {
                        const int k = aind1[jj];
                        const int idx = (k >= 0 && k < acol2_sz) ? acol2[k] : -1;
                        col_map[jj] = (idx >= 0 && idx < sa2) ? idx : -1;
                        if (col_map[jj] >= 0) ++valid_count;
                    }
                    if (valid_count == 0) {
                        break;
                    }
                    if (valid_count == sa1) {
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2[c_row + col_map[jj] * m_inv2];
                            }
                        }
                    } else {
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                const int stored = col_map[jj];
                                if (stored >= 0) B_dst[jj] = inv2[c_row + stored * m_inv2];
                            }
                        }
                    }
                }

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                break;
            }
            case 8: // GEMM: Off-diagonal update: inv3 -= fact * inv2
            {
                in_cond_wait(k, j, 2);

                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                if (!fact || !inv2 || !inv3) { break; }

                const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                const int sa1 = semi1.sa;
                const int sa3 = semi3.sa;

                if (sa1 <= 0 || sa3 <= 0) { break; }

                const int* aind1 = semi1.aind.data();
                const int* aind3 = semi3.aind.data();

                // ── Fused gather+compute for small tiles ──
                if (sa1 <= FUSE_THRESH && sa3 <= FUSE_THRESH && has_gather_info) {
                    const int gi_base = static_cast<int>(in) * 3;
                    const int data_off = gather_index[gi_base];
                    const int n_valid  = gather_index[gi_base + 1];
                    const int flags    = gather_index[gi_base + 2];

                    if (flags == 0) { break; } // no overlap, inv3 unchanged

                    if (flags == 3) {
                        // diagonal inv2
                        for (int jj = 0; jj < sa1; ++jj) {
                            const double* fact_col = fact + jj * m_fact;
                            const int a1 = aind1[jj];
                            for (int cc = 0; cc < sa3; ++cc) {
                                const double b_val = inv2[aind3[cc] * m_inv2 + a1];
                                double* inv3_col = inv3 + cc * m_inv3;
                                #pragma omp simd
                                for (int r = 0; r < m_inv3; ++r)
                                    inv3_col[r] -= fact_col[r] * b_val;
                            }
                        }
                    } else if (flags == 1) {
                        // all_valid
                        const int32_t* col_offsets = gather_packed + data_off;
                        for (int jj = 0; jj < sa1; ++jj) {
                            const double* fact_col = fact + jj * m_fact;
                            const int a1 = aind1[jj];
                            for (int cc = 0; cc < n_valid; ++cc) {
                                const double b_val = inv2[col_offsets[cc] + a1];
                                double* inv3_col = inv3 + cc * m_inv3;
                                #pragma omp simd
                                for (int r = 0; r < m_inv3; ++r)
                                    inv3_col[r] -= fact_col[r] * b_val;
                            }
                        }
                    } else {
                        // partial
                        const int32_t* pr = gather_packed + data_off;
                        for (int jj = 0; jj < sa1; ++jj) {
                            const double* fact_col = fact + jj * m_fact;
                            const int a1 = aind1[jj];
                            for (int v = 0; v < n_valid; ++v) {
                                const int cc     = pr[v * 2];
                                const double b_val = inv2[pr[v * 2 + 1] + a1];
                                double* inv3_col = inv3 + cc * m_inv3;
                                #pragma omp simd
                                for (int r = 0; r < m_inv3; ++r)
                                    inv3_col[r] -= fact_col[r] * b_val;
                            }
                        }
                    }
                    break;
                }

                // ── Standard gather + BLAS path ──
                double* B = tmp_tile;

                if (has_gather_info) {
                    const int gi_base = static_cast<int>(in) * 3;
                    const int data_off = gather_index[gi_base];
                    const int n_valid  = gather_index[gi_base + 1];
                    const int flags    = gather_index[gi_base + 2];

                    if (flags == 0) {
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                    } else if (flags == 3) {
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_col = aind3[cc];
                            const double* inv2_col = inv2 + c_col * m_inv2;
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2_col[aind1[jj]];
                            }
                        }
                    } else if (flags == 1) {
                        const int32_t* col_offsets = gather_packed + data_off;
                        for (int cc = 0; cc < n_valid; ++cc) {
                            const double* inv2_col = inv2 + col_offsets[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2_col[aind1[jj]];
                            }
                        }
                    } else {
                        const int32_t* pairs = gather_packed + data_off;
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        for (int v = 0; v < n_valid; ++v) {
                            const int cc     = pairs[v * 2];
                            const int offset = pairs[v * 2 + 1];
                            const double* inv2_col = inv2 + offset;
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2_col[aind1[jj]];
                            }
                        }
                    }
                } else {
                    const bool inv2_is_diag = (tiledMatrix->tileMetaCore[index2].row == tiledMatrix->tileMetaCore[index2].col);

                    if (inv2_is_diag) {
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_col = aind3[cc];
                            const double* inv2_col = inv2 + c_col * m_inv2;
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2_col[aind1[jj]];
                            }
                        }
                    } else {
                        const int sa2 = semi2.sa;
                        const int* acol2 = semi2.acol.data();
                        const int acol2_sz = static_cast<int>(semi2.acol.size());

                        int* col_map = reinterpret_cast<int*>(B + static_cast<std::size_t>(sa1) * sa3);
                        int valid_count = 0;
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c = aind3[cc];
                            const int idx = (c >= 0 && c < acol2_sz) ? acol2[c] : -1;
                            col_map[cc] = (idx >= 0 && idx < sa2) ? idx : -1;
                            if (col_map[cc] >= 0) ++valid_count;
                        }
                        if (valid_count == 0) { std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double)); }
                        else {
                            for (int cc = 0; cc < sa3; ++cc) {
                                const int stored = col_map[cc];
                                if (stored >= 0) {
                                    const double* inv2_col = inv2 + stored * m_inv2;
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj) {
                                        B_dst[jj] = inv2_col[aind1[jj]];
                                    }
                                } else {
                                    std::memset(B + cc * sa1, 0, sa1 * sizeof(double));
                                }
                            }
                        }
                    }
                }

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                break;
            }
            case 4: // LAUUM + mirror: Compute L^{-T} * L^{-1} for diagonal tile
            {
                double* inv = tiledMatrix->chunkedInverseTiles[index1];
                if (!inv) { break; }

                const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                const int n = meta.height;

                if (semi.upper_bw == 0) {
                    for (int ii = 0; ii < n; ++ii)
                        inv[ii + ii * n] *= inv[ii + ii * n];
                } else {
                    sTiles::core_dlauum(sTiles::Uplo::Upper, n, inv, n);
                    mirroring(n, inv, inv, n);
                }
                break;
            }
            case 5: // Diagonal tile update: inv1 -= fact * inv2^T
            {
                in_cond_wait(i, k, 2);

                double* fact = tiledMatrix->chunkedDenseTiles[index2];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv1 = tiledMatrix->chunkedInverseTiles[index1];
                if (!fact || !inv2 || !inv1) { break; }

                const int m = tiledMatrix->tileMetaCore[index2].height;
                const int sa2 = tiledMatrix->semisparseTileMetaCore[index2].sa;
                const int n1 = tiledMatrix->tileMetaCore[index1].height;

                if (sa2 <= 0) { break; }

                const int kd1 = tiledMatrix->semisparseTileMetaCore[index1].upper_bw;
                if (kd1 == 0) {
                    for (int kk = 0; kk < sa2; ++kk) {
                        const double* f_col = fact + kk * m;
                        const double* i_col = inv2 + kk * m;
                        for (int r = 0; r < n1; ++r) {
                            inv1[r + r * n1] -= f_col[r] * i_col[r];
                        }
                    }
                } else {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                n1, n1, sa2, -1.0,
                                fact, m,
                                inv2, m,
                                1.0, inv1, n1);
                }
                break;
            }
            case 6:
                in_cond_set(i, i, 2);
                break;
            case 9:
                in_cond_set(i, j, 2);
                break;

            default:
                break;
        }
    }

    in_finalize();
}

// ============================================================================

inline int gemm_shape_bin(double flops) {
    if (flops < 1e3) return 0;
    if (flops < 1e4) return 1;
    if (flops < 1e5) return 2;
    if (flops < 1e6) return 3;
    if (flops < 1e7) return 4;
    if (flops < 1e8) return 5;
    if (flops < 1e9) return 6;
    return 7;
}

// ============================================================================

void pdtrtri_semi_sparse_inv_imp2_analysis(TiledMatrix *tiledMatrix, stiles_context_t *stile)
{
    const int N = tiledMatrix->dim;
    const int tile_size = tiledMatrix->tile_size;
    const int num_tiles_per_dim = (N == 0) ? 0 : (N - 1) / tile_size + 1;

    // Use pre-allocated workspace for B matrix in cases 7/8, and diagonal extraction in case 2
    const int rank = STILES_RANK;
    if (rank >= tiledMatrix->num_workspaces) {
        sTiles::Logger::error("[trtri_selinv_imp1] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
        return;
    }
    double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();

    // ── analysis: rank 0 (re)allocates the per-rank stats array; barrier so all
    //    ranks see it before they begin populating their own slot.
    if (rank == 0) {
        g_inv_analysis_stats.assign(stile->world_size, InvAnalysisStats{});
    }
    sTiles::Control::Barrier(stile);
    InvAnalysisStats& S = g_inv_analysis_stats[rank];
    const double t_func_start = omp_get_wtime();
    // ANALYSIS_INSTRUMENTATION_MARKER

    // Compressed-row pilot: when STILES_INV_COMPRESSED_ROWS=1, case 7/8 BLAS path
    // scans `fact` rows, gathers nonzero rows, runs the smaller GEMM, and scatters
    // back to `inv3`. Validates the upper bound on FLOPs savings from row-pattern
    // compression without changing tile storage globally.
    static const bool g_compressed_rows_enabled = []() {
        const char* env = ::getenv("STILES_INV_COMPRESSED_ROWS");
        return env && env[0] == '1';
    }();
    // libxsmm pilot: when STILES_INV_LIBXSMM=1, replace cblas_dgemm in case 7/8
    // BLAS path with a JIT-dispatched libxsmm kernel. Falls back to cblas_dgemm
    // when libxsmm is not built into libstiles or the env var is unset.
    static const bool g_libxsmm_enabled = []() {
#ifdef STILES_WITH_LIBXSMM
        const char* env = ::getenv("STILES_INV_LIBXSMM");
        return env && env[0] == '1';
#else
        return false;
#endif
    }();
    // Per-thread scratch reused across iterations to avoid allocator churn.
    thread_local std::vector<int>    cr_active_rows;
    thread_local std::vector<double> cr_compF;
    thread_local std::vector<double> cr_compR;

    // Precomputed gather info (may be null if not built)
    const int32_t* gather_index = (tiledMatrix->inv_gather_index) ? tiledMatrix->inv_gather_index->data() : nullptr;
    const int32_t* gather_packed = (tiledMatrix->inv_gather_packed) ? tiledMatrix->inv_gather_packed->data() : nullptr;
    const bool has_gather_info = (gather_index != nullptr && gather_packed != nullptr
        && tiledMatrix->inv_gather_index && static_cast<int>(tiledMatrix->inv_gather_index->size()) >= static_cast<int>(tiledMatrix->inv_tasks ? tiledMatrix->inv_tasks->size() : 0) * 3);

    const auto &tasks   = sTiles::get_inv_tasks(tiledMatrix);
    const auto &offsets = sTiles::get_inv_task_offsets(tiledMatrix);
    size_t start = 0, end = tasks.size();
    if (!offsets.empty()) {
        const size_t r = static_cast<size_t>(STILES_RANK);
        if (r < offsets.size()) start = static_cast<size_t>(offsets[r]);
        if (r + 1 < offsets.size()) end = static_cast<size_t>(offsets[r + 1]);
    }

    in_init(num_tiles_per_dim, num_tiles_per_dim, 0);
    int global_in = -1;

    constexpr int FUSE_THRESH = 8;

    const double t_phase1_start = omp_get_wtime();

    // ========== Phase 1: Compute L^{-1} for diagonal tiles ==========
    for (size_t in = start; in < end; ++in) {
        const auto &t = tasks[in];
        const int myroutine = t[0];
        const int index1 = t[4];
        const int index2 = t[5];

        const double t_iter = omp_get_wtime();
        const int ridx = (myroutine >= 0 && myroutine < 10) ? myroutine : 0;

        switch (myroutine) {
            case 1: // TRSM: Compute L^{-1} for diagonal tile
            {
                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                if (!fact || !inv) { break; }

                const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                const int n = meta.height;
                const int kd = semi.upper_bw;

                if (kd == 0) {
                    // Pure diagonal L: L^{-1} = diag(1/L[i])
                    std::memset(inv, 0, static_cast<std::size_t>(n) * n * sizeof(double));
                    for (int ii = 0; ii < n; ++ii)
                        inv[ii + ii * n] = 1.0 / fact[ii];
                } else {
                    // Banded triangular solve: L * X = I => X = L^{-1}
                    const int ldab = kd + 1;
                    LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N',
                                   n, kd, n, fact, ldab, inv, n);
                }
                break;
            }
            case 2: // TRMM: Compute L^{-1} * L_offdiag (stored back in factor tile)
            {
                double* inv = tiledMatrix->chunkedInverseTiles[index1];
                double* fact = tiledMatrix->chunkedDenseTiles[index2];
                if (!inv || !fact) { break; }

                const int h_inv = tiledMatrix->tileMetaCore[index1].height;
                const sTiles::SemisparseTileMetaCore& semi_inv = tiledMatrix->semisparseTileMetaCore[index1];
                const int m = tiledMatrix->tileMetaCore[index2].height;
                const int sa = tiledMatrix->semisparseTileMetaCore[index2].sa;
                if (sa <= 0) { break; }

                if (semi_inv.upper_bw == 0) {
                    // L^{-1} is diagonal: TRMM = row scaling
                    // Extract diagonal into contiguous array to avoid stride-(m+1) access
                    double* diag = tmp_tile;
                    for (int r = 0; r < m; ++r)
                        diag[r] = inv[r + r * m];
                    for (int cc = 0; cc < sa; ++cc) {
                        double* col = fact + cc * m;
                        #pragma omp simd
                        for (int r = 0; r < m; ++r) {
                            col[r] *= diag[r];
                        }
                    }
                } else {
                    sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                       m, sa, 1.0, inv, h_inv, fact, m);
                }
                break;
            }
            case 3: // Barrier
            {
                sTiles::Control::Barrier(stile);
                S.time_case[ridx] += omp_get_wtime() - t_iter;
                S.count[ridx]++;
                global_in = static_cast<int>(in) + 1;
                S.phase1_time = omp_get_wtime() - t_phase1_start;
                goto exit_phase1_semisparse_selinv;
            }
            default:
                break;
        }
        S.time_case[ridx] += omp_get_wtime() - t_iter;
        S.count[ridx]++;
    }

exit_phase1_semisparse_selinv:
    if (global_in < 0) {
        global_in = static_cast<int>(start);
        S.phase1_time = omp_get_wtime() - t_phase1_start;
    }

    // Hoisted base pointers + bounds for the per-iteration prefetch block.
    double** const cdt = tiledMatrix->chunkedDenseTiles;
    double** const cit = tiledMatrix->chunkedInverseTiles;
    const auto* const ssm = tiledMatrix->semisparseTileMetaCore;
    const int ntiles_bounds = tiledMatrix->numActiveTiles;

    const double t_phase2_start = omp_get_wtime();

    // ── Diagnostic helpers (libxsmm/batching/wait pilot) ───────────────────
    auto bucket_wait = [&](double dt) -> bool {
        if (dt < 1e-6) { S.wait_immediate++; return true; }
        if (dt < 1e-5) { S.wait_short++; }
        else if (dt < 1e-4) { S.wait_medium++; }
        else if (dt < 1e-3) { S.wait_long++; }
        else                { S.wait_xlong++; }
        S.wait_blocked_time += dt;
        return false;
    };
    long cur_run_immediate = 0;
    auto flush_batch_run = [&]() {
        if (cur_run_immediate > 0) {
            S.batch_runs_count++;
            S.batch_runs_sum += cur_run_immediate;
            if (cur_run_immediate > S.batch_runs_max) S.batch_runs_max = cur_run_immediate;
            cur_run_immediate = 0;
        }
    };

    // ========== Phase 2: Compute selective inverse entries ==========
    for (size_t in = static_cast<size_t>(global_in); in < end; ++in) {
        const auto &t = tasks[in];
        const int myroutine = t[0];
        const int i = t[1];
        const int j = t[2];
        const int k = t[3];
        const int index1 = t[4];
        const int index2 = t[5];
        const int index3 = t[6];

        // Prefetch next task's tile/meta backing storage. The gather_packed
        // indirection is the unpredictable load that benefits most. Prefetch is
        // a hint only — safe under the cond_wait sync protocol below.
        if (in + 1 < end) {
            const auto &nt = tasks[in + 1];
            const int n1 = nt[4], n2 = nt[5], n3 = nt[6];
            if (n1 >= 0 && n1 < ntiles_bounds) {
                if (cdt[n1]) __builtin_prefetch(cdt[n1], 0, 2);
                if (cit[n1]) __builtin_prefetch(cit[n1], 1, 2);
                __builtin_prefetch(&ssm[n1], 0, 3);
            }
            if (n2 >= 0 && n2 < ntiles_bounds) {
                if (cdt[n2]) __builtin_prefetch(cdt[n2], 0, 2);
                if (cit[n2]) __builtin_prefetch(cit[n2], 0, 2);
                __builtin_prefetch(&ssm[n2], 0, 3);
            }
            if (n3 >= 0 && n3 < ntiles_bounds && cit[n3]) {
                __builtin_prefetch(cit[n3], 1, 2);
            }
            if (has_gather_info) {
                const int gi = static_cast<int>(in + 1) * 3;
                __builtin_prefetch(&gather_index[gi], 0, 3);
                const int next_off = gather_index[gi];
                if (next_off >= 0) __builtin_prefetch(gather_packed + next_off, 0, 3);
            }
        }

        const double t_iter = omp_get_wtime();
        const int ridx = (myroutine >= 0 && myroutine < 10) ? myroutine : 0;
        // GEMM tracking — set inside cases 7/8 to record completed-vs-skipped + dims.
        int gemm_path = 0;          // 0=skipped/null, 1=fused, 2=BLAS
        int gemm_sa1 = 0, gemm_sa3 = 0, gemm_m = 0;
        // Wait diagnostics — captured inside the wait blocks of cases 5/7/8.
        double t_post_wait = t_iter;
        bool wait_was_immediate = true;

        switch (myroutine) {
            case 7: // GEMM: Off-diagonal update: inv3 -= fact * inv2^T
            {
                {
                    const double t_wait = omp_get_wtime();
                    in_cond_wait(j, k, 2);
                    t_post_wait = omp_get_wtime();
                    const double wait_dt = t_post_wait - t_wait;
                    S.wait_time += wait_dt;
                    S.wait_count++;
                    wait_was_immediate = bucket_wait(wait_dt);
                }

                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                if (!fact || !inv2 || !inv3) { break; }

                const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                const int sa1 = semi1.sa;
                const int sa3 = semi3.sa;

                gemm_sa1 = sa1; gemm_sa3 = sa3; gemm_m = m_inv3;

                const int* aind3 = semi3.aind.data();

                // ── Fused gather+compute for small tiles ──
                if (sa1 <= FUSE_THRESH && sa3 <= FUSE_THRESH && has_gather_info) {
                    const int gi_base = static_cast<int>(in) * 3;
                    const int data_off = gather_index[gi_base];
                    const int n_valid  = gather_index[gi_base + 1];
                    const int flags    = gather_index[gi_base + 2];

                    if (flags == 0) { break; }

                    if (flags == 1) {
                        const int32_t* col_offsets = gather_packed + data_off;
                        for (int jj = 0; jj < n_valid; ++jj) {
                            const int col_off = col_offsets[jj];
                            const double* fact_col = fact + jj * m_fact;
                            for (int cc = 0; cc < sa3; ++cc) {
                                const double b_val = inv2[aind3[cc] + col_off];
                                double* inv3_col = inv3 + cc * m_inv3;
                                #pragma omp simd
                                for (int r = 0; r < m_inv3; ++r)
                                    inv3_col[r] -= fact_col[r] * b_val;
                            }
                        }
                    } else {
                        const int32_t* pr = gather_packed + data_off;
                        for (int v = 0; v < n_valid; ++v) {
                            const int jj     = pr[v * 2];
                            const int offset = pr[v * 2 + 1];
                            const double* fact_col = fact + jj * m_fact;
                            for (int cc = 0; cc < sa3; ++cc) {
                                const double b_val = inv2[aind3[cc] + offset];
                                double* inv3_col = inv3 + cc * m_inv3;
                                #pragma omp simd
                                for (int r = 0; r < m_inv3; ++r)
                                    inv3_col[r] -= fact_col[r] * b_val;
                            }
                        }
                    }
                    gemm_path = 1;
                    break;
                }

                // ── Standard gather + BLAS path ──
                double* B = tmp_tile;

                if (has_gather_info) {
                    const int gi_base = static_cast<int>(in) * 3;
                    const int data_off = gather_index[gi_base];
                    const int n_valid  = gather_index[gi_base + 1];
                    const int flags    = gather_index[gi_base + 2];

                    if (flags == 0) {
                        break;
                    }

                    if (flags == 1) {
                        const int32_t* col_offsets = gather_packed + data_off;
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < n_valid; ++jj) {
                                B_dst[jj] = inv2[c_row + col_offsets[jj]];
                            }
                        }
                    } else {
                        const int32_t* pairs = gather_packed + data_off;
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int v = 0; v < n_valid; ++v) {
                                const int jj     = pairs[v * 2];
                                const int offset = pairs[v * 2 + 1];
                                B_dst[jj] = inv2[c_row + offset];
                            }
                        }
                    }
                } else {
                    const int* aind1 = semi1.aind.data();
                    const int sa2 = semi2.sa;
                    const int* acol2 = semi2.acol.data();
                    const int acol2_sz = static_cast<int>(semi2.acol.size());

                    int* col_map = reinterpret_cast<int*>(B + static_cast<std::size_t>(sa1) * sa3);
                    int valid_count = 0;
                    for (int jj = 0; jj < sa1; ++jj) {
                        const int k = aind1[jj];
                        const int idx = (k >= 0 && k < acol2_sz) ? acol2[k] : -1;
                        col_map[jj] = (idx >= 0 && idx < sa2) ? idx : -1;
                        if (col_map[jj] >= 0) ++valid_count;
                    }
                    if (valid_count == 0) {
                        break;
                    }
                    if (valid_count == sa1) {
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2[c_row + col_map[jj] * m_inv2];
                            }
                        }
                    } else {
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                const int stored = col_map[jj];
                                if (stored >= 0) B_dst[jj] = inv2[c_row + stored * m_inv2];
                            }
                        }
                    }
                }

                if (g_compressed_rows_enabled) {
                    // Scan fact (m_fact x sa1, col-major) for nonzero rows.
                    cr_active_rows.clear();
                    cr_active_rows.reserve(static_cast<size_t>(m_fact));
                    for (int r = 0; r < m_fact; ++r) {
                        bool any_nz = false;
                        for (int c = 0; c < sa1; ++c) {
                            if (fact[r + c * m_fact] != 0.0) { any_nz = true; break; }
                        }
                        if (any_nz) cr_active_rows.push_back(r);
                    }
                    const int n_act = static_cast<int>(cr_active_rows.size());
                    if (n_act > 0) {
                        cr_compF.resize(static_cast<size_t>(n_act) * sa1);
                        for (int c = 0; c < sa1; ++c) {
                            double* dst = cr_compF.data() + c * n_act;
                            const double* src = fact + c * m_fact;
                            for (int r = 0; r < n_act; ++r) dst[r] = src[cr_active_rows[r]];
                        }
                        cr_compR.assign(static_cast<size_t>(n_act) * sa3, 0.0);
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    n_act, sa3, sa1, -1.0,
                                    cr_compF.data(), n_act, B, sa1,
                                    0.0, cr_compR.data(), n_act);
                        for (int c = 0; c < sa3; ++c) {
                            const double* src = cr_compR.data() + c * n_act;
                            double* dst_col = inv3 + c * m_inv3;
                            for (int r = 0; r < n_act; ++r) dst_col[cr_active_rows[r]] += src[r];
                        }
                    }
                    S.compressed_count++;
                    S.compressed_active_rows_sum += n_act;
                    S.compressed_full_rows_sum   += m_fact;
                    S.compressed_gemm_flops      += 2.0 * n_act * sa3 * sa1;
                } else {
                    const double t_pre_dgemm = omp_get_wtime();
#ifdef STILES_WITH_LIBXSMM
                    if (g_libxsmm_enabled) {
                        auto kernel = sTiles::XSMM::global_cache().get_or_dispatch(
                            m_inv3, sa3, sa1, m_fact, sa1, m_inv3);
                        if (kernel) {
                            // libxsmm only supports alpha=1: negate B in place
                            // (B is per-thread scratch, freshly filled this iter)
                            // so that inv3 += fact * (-B) == inv3 - fact * B.
                            const std::size_t bn = static_cast<std::size_t>(sa1) * sa3;
                            #pragma omp simd
                            for (std::size_t i = 0; i < bn; ++i) B[i] = -B[i];
                            sTiles::XSMM::invoke(kernel, fact, B, inv3);
                        } else {
                            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                        m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);
                        }
                    } else
#endif
                    {
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);
                    }
                    const double t_post_dgemm = omp_get_wtime();
                    S.gemm_kernel_time += t_post_dgemm - t_pre_dgemm;
                    S.gemm_gather_time += t_pre_dgemm - t_post_wait;
                    S.shape_set.insert(((int64_t)m_inv3 << 40) | ((int64_t)sa3 << 20) | sa1);
                }

                gemm_path = 2;
                break;
            }
            case 8: // GEMM: Off-diagonal update: inv3 -= fact * inv2
            {
                {
                    const double t_wait = omp_get_wtime();
                    in_cond_wait(k, j, 2);
                    t_post_wait = omp_get_wtime();
                    const double wait_dt = t_post_wait - t_wait;
                    S.wait_time += wait_dt;
                    S.wait_count++;
                    wait_was_immediate = bucket_wait(wait_dt);
                }

                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                if (!fact || !inv2 || !inv3) { break; }

                const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                const int sa1 = semi1.sa;
                const int sa3 = semi3.sa;

                if (sa1 <= 0 || sa3 <= 0) { break; }

                gemm_sa1 = sa1; gemm_sa3 = sa3; gemm_m = m_inv3;

                const int* aind1 = semi1.aind.data();
                const int* aind3 = semi3.aind.data();

                // ── Fused gather+compute for small tiles ──
                if (sa1 <= FUSE_THRESH && sa3 <= FUSE_THRESH && has_gather_info) {
                    const int gi_base = static_cast<int>(in) * 3;
                    const int data_off = gather_index[gi_base];
                    const int n_valid  = gather_index[gi_base + 1];
                    const int flags    = gather_index[gi_base + 2];

                    if (flags == 0) { break; } // no overlap, inv3 unchanged

                    if (flags == 3) {
                        // diagonal inv2
                        for (int jj = 0; jj < sa1; ++jj) {
                            const double* fact_col = fact + jj * m_fact;
                            const int a1 = aind1[jj];
                            for (int cc = 0; cc < sa3; ++cc) {
                                const double b_val = inv2[aind3[cc] * m_inv2 + a1];
                                double* inv3_col = inv3 + cc * m_inv3;
                                #pragma omp simd
                                for (int r = 0; r < m_inv3; ++r)
                                    inv3_col[r] -= fact_col[r] * b_val;
                            }
                        }
                    } else if (flags == 1) {
                        // all_valid
                        const int32_t* col_offsets = gather_packed + data_off;
                        for (int jj = 0; jj < sa1; ++jj) {
                            const double* fact_col = fact + jj * m_fact;
                            const int a1 = aind1[jj];
                            for (int cc = 0; cc < n_valid; ++cc) {
                                const double b_val = inv2[col_offsets[cc] + a1];
                                double* inv3_col = inv3 + cc * m_inv3;
                                #pragma omp simd
                                for (int r = 0; r < m_inv3; ++r)
                                    inv3_col[r] -= fact_col[r] * b_val;
                            }
                        }
                    } else {
                        // partial
                        const int32_t* pr = gather_packed + data_off;
                        for (int jj = 0; jj < sa1; ++jj) {
                            const double* fact_col = fact + jj * m_fact;
                            const int a1 = aind1[jj];
                            for (int v = 0; v < n_valid; ++v) {
                                const int cc     = pr[v * 2];
                                const double b_val = inv2[pr[v * 2 + 1] + a1];
                                double* inv3_col = inv3 + cc * m_inv3;
                                #pragma omp simd
                                for (int r = 0; r < m_inv3; ++r)
                                    inv3_col[r] -= fact_col[r] * b_val;
                            }
                        }
                    }
                    gemm_path = 1;
                    break;
                }

                // ── Standard gather + BLAS path ──
                double* B = tmp_tile;

                if (has_gather_info) {
                    const int gi_base = static_cast<int>(in) * 3;
                    const int data_off = gather_index[gi_base];
                    const int n_valid  = gather_index[gi_base + 1];
                    const int flags    = gather_index[gi_base + 2];

                    if (flags == 0) {
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                    } else if (flags == 3) {
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_col = aind3[cc];
                            const double* inv2_col = inv2 + c_col * m_inv2;
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2_col[aind1[jj]];
                            }
                        }
                    } else if (flags == 1) {
                        const int32_t* col_offsets = gather_packed + data_off;
                        for (int cc = 0; cc < n_valid; ++cc) {
                            const double* inv2_col = inv2 + col_offsets[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2_col[aind1[jj]];
                            }
                        }
                    } else {
                        const int32_t* pairs = gather_packed + data_off;
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        for (int v = 0; v < n_valid; ++v) {
                            const int cc     = pairs[v * 2];
                            const int offset = pairs[v * 2 + 1];
                            const double* inv2_col = inv2 + offset;
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2_col[aind1[jj]];
                            }
                        }
                    }
                } else {
                    const bool inv2_is_diag = (tiledMatrix->tileMetaCore[index2].row == tiledMatrix->tileMetaCore[index2].col);

                    if (inv2_is_diag) {
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_col = aind3[cc];
                            const double* inv2_col = inv2 + c_col * m_inv2;
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                B_dst[jj] = inv2_col[aind1[jj]];
                            }
                        }
                    } else {
                        const int sa2 = semi2.sa;
                        const int* acol2 = semi2.acol.data();
                        const int acol2_sz = static_cast<int>(semi2.acol.size());

                        int* col_map = reinterpret_cast<int*>(B + static_cast<std::size_t>(sa1) * sa3);
                        int valid_count = 0;
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c = aind3[cc];
                            const int idx = (c >= 0 && c < acol2_sz) ? acol2[c] : -1;
                            col_map[cc] = (idx >= 0 && idx < sa2) ? idx : -1;
                            if (col_map[cc] >= 0) ++valid_count;
                        }
                        if (valid_count == 0) { std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double)); }
                        else {
                            for (int cc = 0; cc < sa3; ++cc) {
                                const int stored = col_map[cc];
                                if (stored >= 0) {
                                    const double* inv2_col = inv2 + stored * m_inv2;
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj) {
                                        B_dst[jj] = inv2_col[aind1[jj]];
                                    }
                                } else {
                                    std::memset(B + cc * sa1, 0, sa1 * sizeof(double));
                                }
                            }
                        }
                    }
                }

                if (g_compressed_rows_enabled) {
                    cr_active_rows.clear();
                    cr_active_rows.reserve(static_cast<size_t>(m_fact));
                    for (int r = 0; r < m_fact; ++r) {
                        bool any_nz = false;
                        for (int c = 0; c < sa1; ++c) {
                            if (fact[r + c * m_fact] != 0.0) { any_nz = true; break; }
                        }
                        if (any_nz) cr_active_rows.push_back(r);
                    }
                    const int n_act = static_cast<int>(cr_active_rows.size());
                    if (n_act > 0) {
                        cr_compF.resize(static_cast<size_t>(n_act) * sa1);
                        for (int c = 0; c < sa1; ++c) {
                            double* dst = cr_compF.data() + c * n_act;
                            const double* src = fact + c * m_fact;
                            for (int r = 0; r < n_act; ++r) dst[r] = src[cr_active_rows[r]];
                        }
                        cr_compR.assign(static_cast<size_t>(n_act) * sa3, 0.0);
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    n_act, sa3, sa1, -1.0,
                                    cr_compF.data(), n_act, B, sa1,
                                    0.0, cr_compR.data(), n_act);
                        for (int c = 0; c < sa3; ++c) {
                            const double* src = cr_compR.data() + c * n_act;
                            double* dst_col = inv3 + c * m_inv3;
                            for (int r = 0; r < n_act; ++r) dst_col[cr_active_rows[r]] += src[r];
                        }
                    }
                    S.compressed_count++;
                    S.compressed_active_rows_sum += n_act;
                    S.compressed_full_rows_sum   += m_fact;
                    S.compressed_gemm_flops      += 2.0 * n_act * sa3 * sa1;
                } else {
                    const double t_pre_dgemm = omp_get_wtime();
#ifdef STILES_WITH_LIBXSMM
                    if (g_libxsmm_enabled) {
                        auto kernel = sTiles::XSMM::global_cache().get_or_dispatch(
                            m_inv3, sa3, sa1, m_fact, sa1, m_inv3);
                        if (kernel) {
                            const std::size_t bn = static_cast<std::size_t>(sa1) * sa3;
                            #pragma omp simd
                            for (std::size_t i = 0; i < bn; ++i) B[i] = -B[i];
                            sTiles::XSMM::invoke(kernel, fact, B, inv3);
                        } else {
                            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                        m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);
                        }
                    } else
#endif
                    {
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);
                    }
                    const double t_post_dgemm = omp_get_wtime();
                    S.gemm_kernel_time += t_post_dgemm - t_pre_dgemm;
                    S.gemm_gather_time += t_pre_dgemm - t_post_wait;
                    S.shape_set.insert(((int64_t)m_inv3 << 40) | ((int64_t)sa3 << 20) | sa1);
                }

                gemm_path = 2;
                break;
            }
            case 4: // LAUUM + mirror: Compute L^{-T} * L^{-1} for diagonal tile
            {
                double* inv = tiledMatrix->chunkedInverseTiles[index1];
                if (!inv) { break; }

                const sTiles::TileMetaCore& meta = tiledMatrix->tileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                const int n = meta.height;

                if (semi.upper_bw == 0) {
                    for (int ii = 0; ii < n; ++ii)
                        inv[ii + ii * n] *= inv[ii + ii * n];
                } else {
                    sTiles::core_dlauum(sTiles::Uplo::Upper, n, inv, n);
                    mirroring(n, inv, inv, n);
                }
                break;
            }
            case 5: // Diagonal tile update: inv1 -= fact * inv2^T
            {
                {
                    const double t_wait = omp_get_wtime();
                    in_cond_wait(i, k, 2);
                    t_post_wait = omp_get_wtime();
                    const double wait_dt = t_post_wait - t_wait;
                    S.wait_time += wait_dt;
                    S.wait_count++;
                    wait_was_immediate = bucket_wait(wait_dt);
                }

                double* fact = tiledMatrix->chunkedDenseTiles[index2];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv1 = tiledMatrix->chunkedInverseTiles[index1];
                if (!fact || !inv2 || !inv1) { break; }

                const int m = tiledMatrix->tileMetaCore[index2].height;
                const int sa2 = tiledMatrix->semisparseTileMetaCore[index2].sa;
                const int n1 = tiledMatrix->tileMetaCore[index1].height;

                if (sa2 <= 0) { break; }

                const int kd1 = tiledMatrix->semisparseTileMetaCore[index1].upper_bw;
                if (kd1 == 0) {
                    for (int kk = 0; kk < sa2; ++kk) {
                        const double* f_col = fact + kk * m;
                        const double* i_col = inv2 + kk * m;
                        for (int r = 0; r < n1; ++r) {
                            inv1[r + r * n1] -= f_col[r] * i_col[r];
                        }
                    }
                } else {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                n1, n1, sa2, -1.0,
                                fact, m,
                                inv2, m,
                                1.0, inv1, n1);
                }
                break;
            }
            case 6:
                in_cond_set(i, i, 2);
                break;
            case 9:
                in_cond_set(i, j, 2);
                break;

            default:
                break;
        }

        S.time_case[ridx] += omp_get_wtime() - t_iter;
        S.count[ridx]++;

        if (myroutine == 7 || myroutine == 8) {
            if (gemm_path != 0) {
                S.gemm_total++;
                if (gemm_path == 1) S.gemm_fused++;
                else                S.gemm_blas++;
                const double flops = 2.0 * static_cast<double>(gemm_m) * gemm_sa3 * gemm_sa1;
                S.gemm_flops += flops;
                S.shape_bins[gemm_shape_bin(flops)]++;
                S.sum_sa1    += gemm_sa1;
                S.sum_sa3    += gemm_sa3;
                S.sum_m_inv3 += gemm_m;
                if (gemm_sa1 > S.max_sa1)    S.max_sa1    = gemm_sa1;
                if (gemm_sa3 > S.max_sa3)    S.max_sa3    = gemm_sa3;
                if (gemm_m   > S.max_m_inv3) S.max_m_inv3 = gemm_m;
            } else {
                S.gemm_total++;
                S.gemm_skipped++;
            }
        }

        // Batching potential: a run is consecutive case 7/8 BLAS-completing
        // tasks whose waits were all "immediate" (deps already satisfied →
        // they could be batched into one cblas_dgemm_batch call).
        if ((myroutine == 7 || myroutine == 8) && gemm_path == 2 && wait_was_immediate) {
            cur_run_immediate++;
        } else {
            flush_batch_run();
        }
    }
    flush_batch_run();

    S.phase2_time = omp_get_wtime() - t_phase2_start;
    in_finalize();
    S.total_time = omp_get_wtime() - t_func_start;

    sTiles::Control::Barrier(stile);
    if (rank == 0) {
        const int W = stile->world_size;
        InvAnalysisStats T{};
        for (int r = 0; r < W; ++r) {
            const auto& s = g_inv_analysis_stats[r];
            for (int c = 0; c < 10; ++c) {
                T.count[c]     += s.count[c];
                T.time_case[c] += s.time_case[c];
            }
            T.gemm_total   += s.gemm_total;
            T.gemm_fused   += s.gemm_fused;
            T.gemm_blas    += s.gemm_blas;
            T.gemm_skipped += s.gemm_skipped;
            T.gemm_flops   += s.gemm_flops;
            T.sum_sa1      += s.sum_sa1;
            T.sum_sa3      += s.sum_sa3;
            T.sum_m_inv3   += s.sum_m_inv3;
            if (s.max_sa1    > T.max_sa1)    T.max_sa1    = s.max_sa1;
            if (s.max_sa3    > T.max_sa3)    T.max_sa3    = s.max_sa3;
            if (s.max_m_inv3 > T.max_m_inv3) T.max_m_inv3 = s.max_m_inv3;
            for (int b = 0; b < 8; ++b) T.shape_bins[b] += s.shape_bins[b];
            T.wait_time  += s.wait_time;
            T.wait_count += s.wait_count;
            T.compressed_count            += s.compressed_count;
            T.compressed_active_rows_sum  += s.compressed_active_rows_sum;
            T.compressed_full_rows_sum    += s.compressed_full_rows_sum;
            T.compressed_gemm_flops       += s.compressed_gemm_flops;
            T.wait_immediate              += s.wait_immediate;
            T.wait_short                  += s.wait_short;
            T.wait_medium                 += s.wait_medium;
            T.wait_long                   += s.wait_long;
            T.wait_xlong                  += s.wait_xlong;
            T.wait_blocked_time           += s.wait_blocked_time;
            T.gemm_gather_time            += s.gemm_gather_time;
            T.gemm_kernel_time            += s.gemm_kernel_time;
            for (int64_t key : s.shape_set) T.shape_set.insert(key);
            T.batch_runs_count            += s.batch_runs_count;
            T.batch_runs_sum              += s.batch_runs_sum;
            if (s.batch_runs_max > T.batch_runs_max) T.batch_runs_max = s.batch_runs_max;
        }

        double max_total = 0.0, min_total = 1e300, sum_total = 0.0;
        for (int r = 0; r < W; ++r) {
            const double tt = g_inv_analysis_stats[r].total_time;
            sum_total += tt;
            if (tt > max_total) max_total = tt;
            if (tt < min_total) min_total = tt;
        }
        const double avg_total = (W > 0) ? sum_total / W : 0.0;

        std::cout << "\n========== pdtrtri_semi_sparse_inv_imp3 analysis ==========\n";
        std::cout << "  N=" << N << "  tile_size=" << tile_size
                  << "  ranks=" << W
                  << "  total_tasks=" << tasks.size() << "\n";

        std::cout << "\n  Per-rank totals:\n";
        std::cout << "    rank | tasks   | phase1(s) | phase2(s) | total(s) | wait(s)  | wait/total\n";
        for (int r = 0; r < W; ++r) {
            const auto& s = g_inv_analysis_stats[r];
            long ntasks = 0;
            for (int c = 0; c < 10; ++c) ntasks += s.count[c];
            const double waitfrac = (s.total_time > 0.0) ? (100.0 * s.wait_time / s.total_time) : 0.0;
            std::cout << "    " << std::setw(4) << r
                      << " | " << std::setw(7) << ntasks
                      << " | " << std::setw(9) << std::fixed << std::setprecision(4) << s.phase1_time
                      << " | " << std::setw(9) << s.phase2_time
                      << " | " << std::setw(8) << s.total_time
                      << " | " << std::setw(8) << s.wait_time
                      << " | " << std::setw(6) << std::setprecision(2) << waitfrac << "%\n";
        }
        if (W > 1 && min_total > 0.0) {
            std::cout << "    imbalance: max/avg=" << std::setprecision(3) << (max_total / avg_total)
                      << "  max/min=" << (max_total / min_total) << "\n";
        }

        const char* case_names[10] = {"-", "TRSM", "TRMM", "Barrier", "LAUUM",
                                      "GEMM-d", "Set6", "GEMM-7", "GEMM-8", "Set9"};
        std::cout << "\n  Per-case (summed across ranks):\n";
        std::cout << "    case | name    | count    | total(s) | avg(us)\n";
        for (int c = 1; c < 10; ++c) {
            if (T.count[c] == 0 && T.time_case[c] == 0.0) continue;
            const double avg_us = (T.count[c] > 0) ? (T.time_case[c] * 1e6 / T.count[c]) : 0.0;
            std::cout << "    " << std::setw(4) << c
                      << " | " << std::setw(7) << case_names[c]
                      << " | " << std::setw(8) << T.count[c]
                      << " | " << std::setw(8) << std::fixed << std::setprecision(4) << T.time_case[c]
                      << " | " << std::setw(7) << std::setprecision(2) << avg_us << "\n";
        }

        std::cout << "\n  GEMM stats (cases 7+8):\n";
        std::cout << "    total=" << T.gemm_total
                  << "  fused=" << T.gemm_fused
                  << "  blas=" << T.gemm_blas
                  << "  skipped=" << T.gemm_skipped << "\n";
        const long gemm_done = T.gemm_fused + T.gemm_blas;
        if (gemm_done > 0) {
            std::cout << "    avg sa1="    << std::fixed << std::setprecision(1) << ((double)T.sum_sa1    / gemm_done)
                      << "  avg sa3="     << ((double)T.sum_sa3    / gemm_done)
                      << "  avg m_inv3="  << ((double)T.sum_m_inv3 / gemm_done) << "\n";
            std::cout << "    max sa1="    << T.max_sa1
                      << "  max sa3="     << T.max_sa3
                      << "  max m_inv3="  << T.max_m_inv3 << "\n";
            std::cout << "    total flops=" << std::scientific << std::setprecision(3) << T.gemm_flops << "\n";
        }
        std::cout << "    FLOPs histogram (per GEMM):\n";
        const char* bin_labels[8] = {"<1e3 ", "<1e4 ", "<1e5 ", "<1e6 ",
                                     "<1e7 ", "<1e8 ", "<1e9 ", ">=1e9"};
        for (int b = 0; b < 8; ++b) {
            std::cout << "      " << bin_labels[b] << ": " << T.shape_bins[b] << "\n";
        }

        // ── Wait diagnostics: sync overhead vs real dependency latency ─────
        const long total_waits = T.wait_immediate + T.wait_short + T.wait_medium
                               + T.wait_long + T.wait_xlong;
        if (total_waits > 0) {
            std::cout << "\n  Wait-time diagnostics (cases 5/7/8):\n";
            const double pct_imm = 100.0 * T.wait_immediate / (double)total_waits;
            const double pct_blk = 100.0 - pct_imm;
            std::cout << "    total waits=" << total_waits
                      << "  immediate=" << T.wait_immediate << " (" << std::fixed << std::setprecision(1) << pct_imm << "%)"
                      << "  blocked=" << (total_waits - T.wait_immediate) << " (" << pct_blk << "%)\n";
            std::cout << "    bucket counts: <1us=" << T.wait_immediate
                      << "  <10us=" << T.wait_short
                      << "  <100us=" << T.wait_medium
                      << "  <1ms=" << T.wait_long
                      << "  >=1ms=" << T.wait_xlong << "\n";
            std::cout << "    blocked-wait time = " << std::fixed << std::setprecision(4) << T.wait_blocked_time << " s"
                      << "  (= sum of waits >= 1us)\n";
        }

        // ── GEMM kernel vs gather split (libxsmm benefit ceiling) ──────────
        if (T.gemm_kernel_time > 0.0) {
            const double k = T.gemm_kernel_time;
            const double g = T.gemm_gather_time;
            std::cout << "\n  GEMM time split (cases 7/8 BLAS path, summed across ranks):\n";
            std::cout << "    cblas_dgemm = " << std::fixed << std::setprecision(4) << k << " s"
                      << "  gather+misc = " << g << " s\n";
            std::cout << "    upper-bound libxsmm savings: replacing dgemm dispatch (~70%% of "
                      << k << " s) ≈ " << std::setprecision(2) << (k * 0.7) << " s.\n";
            std::cout << "    distinct (m,n,k) shapes seen = " << T.shape_set.size()
                      << "  (libxsmm JIT cache size if adopted)\n";
        }

        // ── Batching potential ──────────────────────────────────────────────
        if (T.batch_runs_count > 0) {
            const double avg_run = (double)T.batch_runs_sum / T.batch_runs_count;
            std::cout << "\n  Batching potential (consecutive case 7/8 with immediate-deps):\n";
            std::cout << "    runs=" << T.batch_runs_count
                      << "  total_tasks_in_runs=" << T.batch_runs_sum
                      << "  avg_run_len=" << std::fixed << std::setprecision(1) << avg_run
                      << "  max_run_len=" << T.batch_runs_max << "\n";
            const long blas_total = T.gemm_blas;
            if (blas_total > 0) {
                const double pct_batchable = 100.0 * T.batch_runs_sum / (double)blas_total;
                std::cout << "    batchable fraction of BLAS GEMMs = "
                          << std::setprecision(1) << pct_batchable << "%\n";
            }
        }

        if (T.compressed_count > 0) {
            const double full_flops = T.gemm_flops;
            const double comp_flops = T.compressed_gemm_flops;
            const double row_ratio  = (T.compressed_full_rows_sum > 0)
                ? (double)T.compressed_active_rows_sum / T.compressed_full_rows_sum : 0.0;
            const double flop_ratio = (full_flops > 0.0) ? comp_flops / full_flops : 0.0;
            std::cout << "\n  Compressed-row pilot (STILES_INV_COMPRESSED_ROWS=1):\n";
            std::cout << "    GEMMs taken: " << T.compressed_count << "\n";
            std::cout << "    avg active_rows / full_rows = " << std::fixed << std::setprecision(3) << row_ratio
                      << "  (compressed_rows=" << T.compressed_active_rows_sum
                      << ", full_rows=" << T.compressed_full_rows_sum << ")\n";
            std::cout << "    compressed flops = " << std::scientific << std::setprecision(3) << comp_flops
                      << "  vs full flops = " << full_flops
                      << "  (ratio=" << std::fixed << std::setprecision(3) << flop_ratio << ")\n";
        }
        std::cout << "============================================================\n\n";
    }
    sTiles::Control::Barrier(stile);
}

// ============================================================================

void pdtrtri_semi_sparse_inv_imp1_serial(TiledMatrix *tiledMatrix, stiles_context_t *stile)
{
    (void)stile; // unused in serial version
    const int rank = STILES_RANK;
    if (rank >= tiledMatrix->num_workspaces) {
        sTiles::Logger::error("[trtri_selinv_imp1_serial] rank ", rank, " >= num_workspaces ", tiledMatrix->num_workspaces);
        return;
    }
    double* tmp_tile = tiledMatrix->workspaces[rank]->aligned_tile();

    // Precomputed gather info (may be null if not built)
    const int32_t* gather_index = (tiledMatrix->inv_gather_index) ? tiledMatrix->inv_gather_index->data() : nullptr;
    const int32_t* gather_packed = (tiledMatrix->inv_gather_packed) ? tiledMatrix->inv_gather_packed->data() : nullptr;
    const bool has_gather_info = (gather_index != nullptr && gather_packed != nullptr
        && tiledMatrix->inv_gather_index && static_cast<int>(tiledMatrix->inv_gather_index->size()) >= static_cast<int>(tiledMatrix->inv_tasks ? tiledMatrix->inv_tasks->size() : 0) * 3);

    const auto &tasks = sTiles::get_inv_tasks(tiledMatrix);
    const size_t start = 0, end = tasks.size();

    // Hoisted base pointers + bounds for the per-iteration prefetch block.
    double** const cdt = tiledMatrix->chunkedDenseTiles;
    double** const cit = tiledMatrix->chunkedInverseTiles;
    const auto* const ssm = tiledMatrix->semisparseTileMetaCore;
    const int ntiles_bounds = tiledMatrix->numActiveTiles;

    // Serial: single pass through all tasks — no barrier, no sync
    for (size_t in = start; in < end; ++in) {
        const auto &t = tasks[in];
        const int myroutine = t[0];
        const int i = t[1];
        const int j = t[2];
        const int k = t[3];
        const int index1 = t[4];
        const int index2 = t[5];
        const int index3 = t[6];

        // Prefetch next task's tile/meta backing storage. The gather_packed
        // indirection is the unpredictable load that benefits most.
        if (in + 1 < end) {
            const auto &nt = tasks[in + 1];
            const int n1 = nt[4], n2 = nt[5], n3 = nt[6];
            if (n1 >= 0 && n1 < ntiles_bounds) {
                if (cdt[n1]) __builtin_prefetch(cdt[n1], 0, 2);
                if (cit[n1]) __builtin_prefetch(cit[n1], 1, 2);
                __builtin_prefetch(&ssm[n1], 0, 3);
            }
            if (n2 >= 0 && n2 < ntiles_bounds) {
                if (cdt[n2]) __builtin_prefetch(cdt[n2], 0, 2);
                if (cit[n2]) __builtin_prefetch(cit[n2], 0, 2);
                __builtin_prefetch(&ssm[n2], 0, 3);
            }
            if (n3 >= 0 && n3 < ntiles_bounds && cit[n3]) {
                __builtin_prefetch(cit[n3], 1, 2);
            }
            if (has_gather_info) {
                const int gi = static_cast<int>(in + 1) * 3;
                __builtin_prefetch(&gather_index[gi], 0, 3);
                const int next_off = gather_index[gi];
                if (next_off >= 0) __builtin_prefetch(gather_packed + next_off, 0, 3);
            }
        }

        switch (myroutine) {
            case 1: // TRSM
            {
                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv  = tiledMatrix->chunkedInverseTiles[index1];
                if (!fact || !inv) { break; }

                const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];
                const int n = tiledMatrix->tileMetaCore[index1].height;
                const int kd = semi.upper_bw;

                if (kd == 0) {
                    std::memset(inv, 0, static_cast<std::size_t>(n) * n * sizeof(double));
                    for (int ii = 0; ii < n; ++ii)
                        inv[ii + ii * n] = 1.0 / fact[ii];
                } else {
                    const int ldab = kd + 1;
                    LAPACKE_dtbtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N',
                                   n, kd, n, fact, ldab, inv, n);
                }
                break;
            }
            case 2: // TRMM
            {
                double* inv = tiledMatrix->chunkedInverseTiles[index1];
                double* fact = tiledMatrix->chunkedDenseTiles[index2];
                if (!inv || !fact) { break; }

                const int h_inv = tiledMatrix->tileMetaCore[index1].height;
                const sTiles::SemisparseTileMetaCore& semi_inv = tiledMatrix->semisparseTileMetaCore[index1];
                const int m = tiledMatrix->tileMetaCore[index2].height;
                const int sa = tiledMatrix->semisparseTileMetaCore[index2].sa;
                if (sa <= 0) { break; }

                if (semi_inv.upper_bw == 0) {
                    double* diag = tmp_tile;
                    for (int r = 0; r < m; ++r)
                        diag[r] = inv[r + r * m];
                    for (int cc = 0; cc < sa; ++cc) {
                        double* col = fact + cc * m;
                        #pragma omp simd
                        for (int r = 0; r < m; ++r) {
                            col[r] *= diag[r];
                        }
                    }
                } else {
                    sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper,
                                       sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                       m, sa, 1.0, inv, h_inv, fact, m);
                }
                break;
            }
            case 3: // Barrier — no-op in serial
            case 6: // Signal — no-op in serial
            case 9: // Signal — no-op in serial
                break;
            case 4: // LAUUM
            {
                double* inv = tiledMatrix->chunkedInverseTiles[index1];
                if (!inv) { break; }

                const int n = tiledMatrix->tileMetaCore[index1].height;
                const sTiles::SemisparseTileMetaCore& semi = tiledMatrix->semisparseTileMetaCore[index1];

                if (semi.upper_bw == 0) {
                    for (int ii = 0; ii < n; ++ii)
                        inv[ii + ii * n] *= inv[ii + ii * n];
                } else {
                    sTiles::core_dlauum(sTiles::Uplo::Upper, n, inv, n);
                    mirroring(n, inv, inv, n);
                }
                break;
            }
            case 5: // Diagonal tile update
            {
                double* fact = tiledMatrix->chunkedDenseTiles[index2];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv1 = tiledMatrix->chunkedInverseTiles[index1];
                if (!fact || !inv2 || !inv1) { break; }

                const int m = tiledMatrix->tileMetaCore[index2].height;
                const int sa2 = tiledMatrix->semisparseTileMetaCore[index2].sa;
                const int n1 = tiledMatrix->tileMetaCore[index1].height;

                if (sa2 <= 0) { break; }

                const int kd1 = tiledMatrix->semisparseTileMetaCore[index1].upper_bw;
                if (kd1 == 0) {
                    for (int kk = 0; kk < sa2; ++kk) {
                        const double* f_col = fact + kk * m;
                        const double* i_col = inv2 + kk * m;
                        for (int r = 0; r < n1; ++r) {
                            inv1[r + r * n1] -= f_col[r] * i_col[r];
                        }
                    }
                } else {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                n1, n1, sa2, -1.0,
                                fact, m, inv2, m,
                                1.0, inv1, n1);
                }
                break;
            }
            case 7: // GEMM: inv3 -= fact * inv2^T
            {
                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                if (!fact || !inv2 || !inv3) { break; }

                const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                const int sa1 = semi1.sa;
                const int sa3 = semi3.sa;

                double* B = tmp_tile;
                const int* aind3 = semi3.aind.data();

                if (has_gather_info) {
                    const int gi_base = static_cast<int>(in) * 3;
                    const int data_off = gather_index[gi_base];
                    const int n_valid  = gather_index[gi_base + 1];
                    const int flags    = gather_index[gi_base + 2];

                    if (flags == 0) {
                        break;
                    }
                    if (flags == 1) {
                        const int32_t* col_offsets = gather_packed + data_off;
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < n_valid; ++jj) {
                                B_dst[jj] = inv2[c_row + col_offsets[jj]];
                            }
                        }
                    } else {
                        const int32_t* pairs = gather_packed + data_off;
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int v = 0; v < n_valid; ++v) {
                                B_dst[pairs[v * 2]] = inv2[c_row + pairs[v * 2 + 1]];
                            }
                        }
                    }
                } else {
                    const int* aind1 = semi1.aind.data();
                    const int sa2 = semi2.sa;
                    const int* acol2 = semi2.acol.data();
                    const int acol2_sz = static_cast<int>(semi2.acol.size());

                    int* col_map = reinterpret_cast<int*>(B + static_cast<std::size_t>(sa1) * sa3);
                    int valid_count = 0;
                    for (int jj = 0; jj < sa1; ++jj) {
                        const int kk = aind1[jj];
                        const int idx = (kk >= 0 && kk < acol2_sz) ? acol2[kk] : -1;
                        col_map[jj] = (idx >= 0 && idx < sa2) ? idx : -1;
                        if (col_map[jj] >= 0) ++valid_count;
                    }
                    if (valid_count == 0) {
                        break;
                    }
                    if (valid_count == sa1) {
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj)
                                B_dst[jj] = inv2[c_row + col_map[jj] * m_inv2];
                        }
                    } else {
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c_row = aind3[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj) {
                                const int stored = col_map[jj];
                                if (stored >= 0) B_dst[jj] = inv2[c_row + stored * m_inv2];
                            }
                        }
                    }
                }

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                break;
            }
            case 8: // GEMM: inv3 -= fact * inv2
            {
                double* fact = tiledMatrix->chunkedDenseTiles[index1];
                double* inv2 = tiledMatrix->chunkedInverseTiles[index2];
                double* inv3 = tiledMatrix->chunkedInverseTiles[index3];
                if (!fact || !inv2 || !inv3) { break; }

                const sTiles::SemisparseTileMetaCore& semi1 = tiledMatrix->semisparseTileMetaCore[index1];
                const sTiles::SemisparseTileMetaCore& semi2 = tiledMatrix->semisparseTileMetaCore[index2];
                const sTiles::SemisparseTileMetaCore& semi3 = tiledMatrix->semisparseTileMetaCore[index3];
                const int m_fact = tiledMatrix->tileMetaCore[index1].height;
                const int m_inv2 = tiledMatrix->tileMetaCore[index2].height;
                const int m_inv3 = tiledMatrix->tileMetaCore[index3].height;
                const int sa1 = semi1.sa;
                const int sa3 = semi3.sa;

                if (sa1 <= 0 || sa3 <= 0) { break; }

                double* B = tmp_tile;
                const int* aind1 = semi1.aind.data();
                const int* aind3 = semi3.aind.data();

                if (has_gather_info) {
                    const int gi_base = static_cast<int>(in) * 3;
                    const int data_off = gather_index[gi_base];
                    const int n_valid  = gather_index[gi_base + 1];
                    const int flags    = gather_index[gi_base + 2];

                    if (flags == 0) {
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                    } else if (flags == 3) {
                        for (int cc = 0; cc < sa3; ++cc) {
                            const double* inv2_col = inv2 + aind3[cc] * m_inv2;
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj)
                                B_dst[jj] = inv2_col[aind1[jj]];
                        }
                    } else if (flags == 1) {
                        const int32_t* col_offsets = gather_packed + data_off;
                        for (int cc = 0; cc < n_valid; ++cc) {
                            const double* inv2_col = inv2 + col_offsets[cc];
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj)
                                B_dst[jj] = inv2_col[aind1[jj]];
                        }
                    } else {
                        const int32_t* pairs = gather_packed + data_off;
                        std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        for (int v = 0; v < n_valid; ++v) {
                            const double* inv2_col = inv2 + pairs[v * 2 + 1];
                            double* B_dst = B + pairs[v * 2] * sa1;
                            for (int jj = 0; jj < sa1; ++jj)
                                B_dst[jj] = inv2_col[aind1[jj]];
                        }
                    }
                } else {
                    const bool inv2_is_diag = (tiledMatrix->tileMetaCore[index2].row == tiledMatrix->tileMetaCore[index2].col);
                    if (inv2_is_diag) {
                        for (int cc = 0; cc < sa3; ++cc) {
                            const double* inv2_col = inv2 + aind3[cc] * m_inv2;
                            double* B_dst = B + cc * sa1;
                            for (int jj = 0; jj < sa1; ++jj)
                                B_dst[jj] = inv2_col[aind1[jj]];
                        }
                    } else {
                        const int sa2 = semi2.sa;
                        const int* acol2 = semi2.acol.data();
                        const int acol2_sz = static_cast<int>(semi2.acol.size());
                        int* col_map = reinterpret_cast<int*>(B + static_cast<std::size_t>(sa1) * sa3);
                        int valid_count = 0;
                        for (int cc = 0; cc < sa3; ++cc) {
                            const int c = aind3[cc];
                            const int idx = (c >= 0 && c < acol2_sz) ? acol2[c] : -1;
                            col_map[cc] = (idx >= 0 && idx < sa2) ? idx : -1;
                            if (col_map[cc] >= 0) ++valid_count;
                        }
                        if (valid_count == 0) {
                            std::memset(B, 0, static_cast<std::size_t>(sa1) * sa3 * sizeof(double));
                        } else {
                            for (int cc = 0; cc < sa3; ++cc) {
                                const int stored = col_map[cc];
                                if (stored >= 0) {
                                    const double* inv2_col = inv2 + stored * m_inv2;
                                    double* B_dst = B + cc * sa1;
                                    for (int jj = 0; jj < sa1; ++jj)
                                        B_dst[jj] = inv2_col[aind1[jj]];
                                } else {
                                    std::memset(B + cc * sa1, 0, sa1 * sizeof(double));
                                }
                            }
                        }
                    }
                }

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m_inv3, sa3, sa1, -1.0, fact, m_fact, B, sa1, 1.0, inv3, m_inv3);

                break;
            }
            default:
                break;
        }
    }
}

// ============================================================================

void stiles_pdtrtri_cpu(stiles_context_t *stile)
{
    int uplo;
    TilesDescriptor A;
    STILES_sequence *sequence;
    STILES_request *request;

    int info;
    int myroutine, i, j, k, task_reached;
    int index1, index2, index3; 

    double zone  = (double) 1.0;
    double mzone = (double)-1.0;

    sTiles::unpack_args(stile, uplo, A, sequence, request);
    if (sequence->status != 0) printf("Error! \n");
        //return 1;

    in_init(A.nt, A.nt, 0);

    int global_in = 0;
    for(int in=0; in<A.e_trick_size_inv[STILES_RANK];in++){

        myroutine = A.e_trick_inv[STILES_RANK][0+7*in];
        i = A.e_trick_inv[STILES_RANK][1+in*7];
        j = A.e_trick_inv[STILES_RANK][2+in*7];
        k = A.e_trick_inv[STILES_RANK][3+in*7];
        index1 = A.e_trick_inv[STILES_RANK][4+in*7];
        index2 = A.e_trick_inv[STILES_RANK][5+in*7];
        index3 = A.e_trick_inv[STILES_RANK][6+in*7];

        switch (myroutine) {

            case 1:
                
                sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit, A.dense_tiles[index1].width, A.dense_tiles[index1].width, 1.0,
                        A.dense_tiles[index1].elements, A.dense_tiles[index1].width,  // A: triangular matrix
                        A.inverse_tiles[index1].elements, A.inverse_tiles[index1].width); // B: identity matrix (right-hand side)


                break;

            case 2:

                sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                    A.dense_tiles[index2].height, A.dense_tiles[index2].width,  // M and N (rows and columns of B)
                    zone, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].height,  // A and its leading dimension (LDA)
                    A.dense_tiles[index2].elements, A.dense_tiles[index2].height); 

                break;

            case 3:
                sTiles::Control::Barrier(stile);
                global_in = in + 1;  // Save the next index for the second loop
                goto exit_first_loop;  // Break out of the first loop

        }

    }

    exit_first_loop:  

    for(int in=global_in; in<A.e_trick_size_inv[STILES_RANK];in++){

        myroutine = A.e_trick_inv[STILES_RANK][0+7*in];
        i = A.e_trick_inv[STILES_RANK][1+in*7];
        j = A.e_trick_inv[STILES_RANK][2+in*7];
        k = A.e_trick_inv[STILES_RANK][3+in*7];
        index1 = A.e_trick_inv[STILES_RANK][4+in*7];
        index2 = A.e_trick_inv[STILES_RANK][5+in*7];
        index3 = A.e_trick_inv[STILES_RANK][6+in*7];

        switch (myroutine) {

            case 4:

                sTiles::core_dlauum(sTiles::Uplo::Upper, A.inverse_tiles[index1].width, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].height); //L * L ^T
                mirroring( A.inverse_tiles[index1].height, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].elements,  A.inverse_tiles[index1].height); //copy the upper of dense_tiles tile to the upper & lower inverse_tiles

                break;

            case 5:
                
                in_cond_wait(i, k, 2);
                sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                    A.dense_tiles[index2].height, A.dense_tiles[index2].height, A.dense_tiles[index2].width,
                    mzone, A.dense_tiles[index2].elements, A.dense_tiles[index2].height,
                            A.inverse_tiles[index2].elements, A.inverse_tiles[index2].height,
                        zone, A.inverse_tiles[index1].elements, A.inverse_tiles[index1].height);

                break;

            case 6:

                in_cond_set(i, i, 2);

                break;

            case 7:

                in_cond_wait(j, k, 2);
                sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                    A.dense_tiles[index1].height, A.inverse_tiles[index2].height, A.dense_tiles[index1].width,
                    -1, A.dense_tiles[index1].elements, A.dense_tiles[index1].height, // i is less than j
                            A.inverse_tiles[index2].elements, A.inverse_tiles[index2].height,
                        zone, A.inverse_tiles[index3].elements, A.inverse_tiles[index3].height);


                break;

            case 8:

                in_cond_wait(k, j, 2);
                sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                    A.dense_tiles[index1].height, A.inverse_tiles[index2].width, A.dense_tiles[index1].width,
                    mzone, A.dense_tiles[index1].elements, A.dense_tiles[index1].height, // i is less than j
                            A.inverse_tiles[index2].elements, A.inverse_tiles[index2].height,
                        zone, A.inverse_tiles[index3].elements, A.inverse_tiles[index3].height);


                break;

            case 9:

                in_cond_set(i, j, 2);
                break;

        }


    }

    in_finalize();

}

// ============================================================================

void stiles_pdtrtri_cpu_no(stiles_context_t *stile)
{
    // --- Argument Unpacking & Initial Setup ---
    int uplo_enum;
    TilesDescriptor A;
    STILES_sequence *sequence;
    STILES_request *request;
    sTiles::unpack_args(stile, uplo_enum, A, sequence, request);

    if (sequence->status != 0) {
        printf("Error! Sequence status not success in stiles_pdtrtri_cpu.\n");
        return;
    }

    // --- Enum Conversions (for new API) ---
    sTiles::Uplo stiles_uplo = sTiles::Uplo::Upper;

    // --- Variable Declarations ---
    int myroutine, i, j, k;
    int index1, index2, index3;
    int global_in = 0;
    double zone = 1.0;
    double mzone = -1.0;
    
    in_init(A.nt, A.nt, 0);

    // --- Phase 1: Triangular Solve & Updates ---
    for (int in = 0; in < A.e_trick_size_inv[STILES_RANK]; in++) {
        myroutine = A.e_trick_inv[STILES_RANK][0 + 7 * in];
        i = A.e_trick_inv[STILES_RANK][1 + 7 * in];
        j = A.e_trick_inv[STILES_RANK][2 + 7 * in];
        k = A.e_trick_inv[STILES_RANK][3 + 7 * in];
        index1 = A.e_trick_inv[STILES_RANK][4 + 7 * in];
        index2 = A.e_trick_inv[STILES_RANK][5 + 7 * in];
        index3 = A.e_trick_inv[STILES_RANK][6 + 7 * in];

        switch (myroutine) {
            case 1: // DTRSM
            {
                auto& tile_A = A.dense_tiles[index1];
                auto& tile_invA = A.inverse_tiles[index1];
                
                auto new_op = [&]() {
                    sTiles::core_dtrsm(sTiles::Side::Left, stiles_uplo, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                       tile_A.width, tile_A.width, zone,
                                       tile_A.elements, tile_A.width,
                                       tile_invA.elements, tile_invA.width);
                };
                auto old_op = [&](double* tile, int lda) { // Corrected signature
                    sTiles::core_dtrsm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                      tile_A.width, tile_A.width, zone,
                                      tile_A.elements, tile_A.width,
                                      tile, lda); // Use provided lda
                };
                
                #if defined(STILE_VERIFY_AGAINST_OLD)
                    STILE_VERIFY_EXEC("TRTRI_DTRSM", tile_A.width, tile_A.width, tile_invA.elements, tile_invA.width, new_op, old_op);
                #elif defined(STILE_USE_NEW_API)
                    new_op();
                #else
                    old_op(tile_invA.elements, tile_invA.width);
                #endif
                break;
            }
            case 2: // DTRMM
            {
                auto& tile_invA_diag = A.inverse_tiles[index1];
                auto& tile_B = A.dense_tiles[index2];
                
                auto new_op = [&]() {
                    sTiles::core_dtrmm(sTiles::Side::Left, stiles_uplo, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                       tile_B.height, tile_B.width, zone,
                                       tile_invA_diag.elements, tile_invA_diag.height,
                                       tile_B.elements, tile_B.height);
                };
                auto old_op = [&](double* tile, int lda) { // Corrected signature
                    sTiles::core_dtrmm(sTiles::Side::Left, sTiles::Uplo::Upper, sTiles::Op::NoTrans, sTiles::Diag::NonUnit,
                                      tile_B.height, tile_B.width, zone,
                                      tile_invA_diag.elements, tile_invA_diag.height,
                                      tile, lda); // Use provided lda
                };

                #if defined(STILE_VERIFY_AGAINST_OLD)
                    STILE_VERIFY_EXEC("TRTRI_DTRMM", tile_B.height, tile_B.width, tile_B.elements, tile_B.height, new_op, old_op);
                #elif defined(STILE_USE_NEW_API)
                    new_op();
                #else
                    old_op(tile_B.elements, tile_B.height);
                #endif
                break;
            }
            case 3: // Synchronization point
                sTiles::Control::Barrier(stile);
                global_in = in + 1;
                goto exit_first_loop;
        }
    }

exit_first_loop:

    // --- Phase 2: Form the full inverse matrix ---
    for (int in = global_in; in < A.e_trick_size_inv[STILES_RANK]; in++) {
        myroutine = A.e_trick_inv[STILES_RANK][0 + 7 * in];
        i = A.e_trick_inv[STILES_RANK][1 + 7 * in];
        j = A.e_trick_inv[STILES_RANK][2 + 7 * in];
        k = A.e_trick_inv[STILES_RANK][3 + 7 * in];
        index1 = A.e_trick_inv[STILES_RANK][4 + 7 * in];
        index2 = A.e_trick_inv[STILES_RANK][5 + 7 * in];
        index3 = A.e_trick_inv[STILES_RANK][6 + 7 * in];

        switch (myroutine) {
            case 4: // DLAUUM
            {
                auto& tile_invA = A.inverse_tiles[index1];
                auto new_op = [&]() {
                    sTiles::core_dlauum(stiles_uplo, tile_invA.width, tile_invA.elements, tile_invA.height);
                    sTiles::corr_dmirr(tile_invA.height, tile_invA.elements, tile_invA.elements, tile_invA.height);
                };
                auto old_op = [&](double* tile, int lda) { // Corrected signature
                    sTiles::core_dlauum(sTiles::Uplo::Upper, tile_invA.width, tile, lda);
                    mirroring(tile_invA.height, tile, tile, lda);
                };
                
                #if defined(STILE_VERIFY_AGAINST_OLD)
                    STILE_VERIFY_EXEC("TRTRI_DLAUUM", tile_invA.width, tile_invA.width, tile_invA.elements, tile_invA.height, new_op, old_op);
                #elif defined(STILE_USE_NEW_API)
                    new_op();
                #else
                    old_op(tile_invA.elements, tile_invA.height);
                #endif
                break;
            }
            case 5: // DGEMM for diagonal update
            {
                auto& tile_A_offdiag = A.dense_tiles[index2];
                auto& tile_invA_offdiag = A.inverse_tiles[index2];
                auto& tile_invA_diag = A.inverse_tiles[index1];
                auto new_op = [&]() { /* ... */ };
                auto old_op = [&](double* tile, int lda) { // Corrected signature
                    sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                      tile_A_offdiag.height, tile_A_offdiag.height, tile_A_offdiag.width,
                                      mzone, tile_A_offdiag.elements, tile_A_offdiag.height,
                                      tile_invA_offdiag.elements, tile_invA_offdiag.height,
                                      zone, tile, lda); // Use provided lda
                };
                
                in_cond_wait(i, k, 2);
                #if defined(STILE_VERIFY_AGAINST_OLD)
                    STILE_VERIFY_EXEC("TRTRI_DGEMM_DIAG", tile_A_offdiag.height, tile_A_offdiag.height, tile_invA_diag.elements, tile_invA_diag.height, new_op, old_op);
                #elif defined(STILE_USE_NEW_API)
                    new_op();
                #else
                    old_op(tile_invA_diag.elements, tile_invA_diag.height);
                #endif
                break;
            }
            case 6: // Set dependency flag
                in_cond_set(i, i, 2);
                break;
            case 7: // DGEMM for off-diagonal update
            {
                auto& tile_A = A.dense_tiles[index1];
                auto& tile_invA1 = A.inverse_tiles[index2];
                auto& tile_invA2 = A.inverse_tiles[index3];
                auto new_op = [&]() { /* ... */ };
                auto old_op = [&](double* tile, int lda) { // Corrected signature
                    sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::Trans,
                                      tile_A.height, tile_invA1.height, tile_A.width,
                                      -1.0, tile_A.elements, tile_A.height,
                                      tile_invA1.elements, tile_invA1.height,
                                      zone, tile, lda); // Use provided lda
                };
                
                in_cond_wait(j, k, 2);
                #if defined(STILE_VERIFY_AGAINST_OLD)
                    STILE_VERIFY_EXEC("TRTRI_DGEMM_OFF1", tile_A.height, tile_invA1.height, tile_invA2.elements, tile_invA2.height, new_op, old_op);
                #elif defined(STILE_USE_NEW_API)
                    new_op();
                #else
                    old_op(tile_invA2.elements, tile_invA2.height);
                #endif
                break;
            }
            case 8: // DGEMM for off-diagonal update
            {
                auto& tile_A = A.dense_tiles[index1];
                auto& tile_invA1 = A.inverse_tiles[index2];
                auto& tile_invA2 = A.inverse_tiles[index3];
                auto new_op = [&]() { /* ... */ };
                auto old_op = [&](double* tile, int lda) { // Corrected signature
                    sTiles::core_dgemm(sTiles::Op::NoTrans, sTiles::Op::NoTrans,
                                      tile_A.height, tile_invA1.width, tile_A.width,
                                      mzone, tile_A.elements, tile_A.height,
                                      tile_invA1.elements, tile_invA1.height,
                                      zone, tile, lda); // Use provided lda
                };

                in_cond_wait(k, j, 2);
                #if defined(STILE_VERIFY_AGAINST_OLD)
                    STILE_VERIFY_EXEC("TRTRI_DGEMM_OFF2", tile_A.height, tile_invA1.width, tile_invA2.elements, tile_invA2.height, new_op, old_op);
                #elif defined(STILE_USE_NEW_API)
                    new_op();
                #else
                    old_op(tile_invA2.elements, tile_invA2.height);
                #endif
                break;
            }
            case 9: // Set dependency flag
                in_cond_set(i, j, 2);
                break;
        }
    }

    in_finalize();
}

#endif  // preserved history
