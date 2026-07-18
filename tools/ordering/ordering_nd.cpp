#include <iostream>
#include <vector>
#include <stdio.h>
#include <cstring>
#include <algorithm>  // for sTiles::sort
#include <numeric>    // for std::iota
#include <cstdio>     // for FILE and associated functions
#include <iostream>   // for std::cout, std::endl
#include <iomanip>  // Needed for std::setprecision and std::fixed
#include <stdlib.h>
#include <ostream>
#include <vector>
#include <algorithm>
#include <cstring> // for std::memcpy
#include <fstream> // Include this header for file stream operations
#include <cmath> // Include this header for mathematical operations
#include <omp.h> // for omp_get_wtime() timing
#include "../common/stiles_logger.hpp"

// METIS header — Makefile passes the resolved absolute path via
// STILES_METIS_HEADER (either system /usr/include/metis.h or the bundled
// install). Absolute path bypasses include-path search so SCOTCH's metis.h
// stub (lacking METIS_NodeNDP) cannot shadow the real header.
#ifdef STILES_METIS_HEADER
#include STILES_METIS_HEADER
#else
#include "metis-5.1.0/local/include/metis.h"
#endif

thread_local bool g_nd_hub_skipped = false;

#include "ordering_utils.hpp"
#include "../common/stiles_exporter.hpp"
#include "../sort/stiles_sort_dispatch.hpp"

extern "C" int stiles_createSmartPermutation(int** row_indices, int** col_indices, int nnz, int node_num, int** perm);

// When 1: use METIS_NodeNDP (returns partition sizes, needed for sTiles_force_ND parallel tree)
// When 0: use METIS_NodeND  (simpler, better quality for standard ordering mode 2)
static int g_nd_use_ndp = 0;
extern "C" void stiles_set_nd_use_ndp(int v) { g_nd_use_ndp = (v != 0) ? 1 : 0; }

int get_new_size(int tile_size, int blockSize) {

    int remainder = blockSize % tile_size;
    if (remainder == 0) {
        return 0; // Already divisible by tile_size
    } else if(blockSize < tile_size){
        return tile_size - blockSize;
    }else{
        return tile_size - remainder; // Amount to add to make it divisible by tile_size
    }
}

bool** smf_createBoolMatrix(int num_tiles) {
    bool** matrix = (bool**)malloc(num_tiles * sizeof(bool*));
    if (!matrix) { // Check for successful allocation
        return NULL;
    }

    for (int i = 0; i < num_tiles; i++) {
        matrix[i] = (bool*)malloc(num_tiles * sizeof(bool)); // Corrected 'cols' to 'num_tiles'
        if (!matrix[i]) { // Check for successful allocation
            for (int j = 0; j < i; j++) {
                free(matrix[j]); // Free previously allocated memory
            }
            free(matrix);
            return NULL;
        }

        for (int j = 0; j < num_tiles; j++) {
            matrix[i][j] = false; // Initialize all tiles as 'off'
        }
    }
    return matrix;
}

const char* find_source_vector(int index, int sizes[], int additionalSizes[], int separator_size, int ncov) {

    if (index < sizes[0]) {
        return "partition1";
    } else if (index < sizes[0] + additionalSizes[0]) {
        return "additions1";
    } else if (index < sizes[0] + additionalSizes[0] + sizes[1]) {
        return "partition2";
    } else if (index < sizes[0] + additionalSizes[0] + sizes[1] + additionalSizes[1]) {
        return "additions2";
    } else if (index < sizes[0] + additionalSizes[0] + sizes[1] + additionalSizes[1] + separator_size) {
        return "temp1";
    } else {
        return "temp2";
    }
}


int NESTED_DISSECTION_LEVEL_1_FINAL(bool*** on_off_tiles, int full_dim, int *new_dim, int full_nnz, int *new_nnz, 
                                    int bandwidth, int ncov, int* sizes, int* newsizes, int* additionalSize, int **perm, 
                                    int tile_size, int **indices_i, int **indices_j, bool **of_perm, int* total_num_used_tiles){


    bool checks = false;
    int dim = full_dim - ncov;  // Example dimension
    //int check_num1 = 2*(5*bandwidth) + bandwidth;
    //if(check_num1 > dim) return -1;

    if(checks) sTiles::Logger::debugf("        -  bandwidth %d", bandwidth);

    //bandwidth -= 100;
    double star_points[] = {
        0.5 * (dim - 4)    // 0.5 * dim
    };
    int n_points = sizeof(star_points) / sizeof(star_points[0]);

    int lower_bounds[n_points];
    int upper_bounds[n_points];
    int separator_sizes[n_points];
    int partition_sizes[n_points+1];  // +1 to include last partition size

    //int test_num = 1, test_num2 = (0.7*dim)/bandwidth;
    int test_num = 1, test_num2 = (0.49*dim)/bandwidth;

    for (int i = 0; i < 1; i++) {
        lower_bounds[i] = test_num2*bandwidth;//*start_index;
        upper_bounds[i] = lower_bounds[i] + test_num*bandwidth -1;//*end_index;
        separator_sizes[i] = upper_bounds[i] - lower_bounds[i] + 1;
        // Calculate size of left partition
        int start_of_partition = (i == 0) ? 0 : upper_bounds[i-1] + 1;
        partition_sizes[i] = lower_bounds[i] - start_of_partition;
    }


    // Calculate size of the last partition
    int last_partition_size = dim - upper_bounds[n_points - 1] - 1;
    partition_sizes[n_points] = last_partition_size;

    sizes[0] = partition_sizes[n_points - 1];
    sizes[1] = last_partition_size;
    sizes[2] = separator_sizes[0] + ncov;

    int size_sizes = 3;

    int sum_additionalSize = 0;
    for(int i = 0; i < size_sizes-1; i++) {
        additionalSize[i] = get_new_size(tile_size, sizes[i]) + tile_size;
        newsizes[i] = sizes[i] + additionalSize[i];
        sum_additionalSize += additionalSize[i];
        
    }


    additionalSize[size_sizes-1] = 0;
    newsizes[size_sizes-1] = sizes[size_sizes-1];

    if(checks){

        sTiles::Logger::debugf("        - EXPANSION SIZE           : %d.", sum_additionalSize);
        for (int i = 0; i < size_sizes; ++i) {
            sTiles::Logger::debugf("                    - partition[%d] was: %8d, then partition[%d] becomes %8d with aditional size %d.", i, sizes[i], i, newsizes[i], additionalSize[i]);
            //printf("                    - partition[%d] was: %8d, then partition[%d] becomes %8d with aditional size %d.\n", i, sizes[i], i, newsizes[i]/tile_size, additionalSize[i]);

        }

    }


    *new_dim = full_dim + additionalSize[0] + additionalSize[1];
    int i = 0;
    *perm = (int*)calloc(*new_dim, sizeof(int)); 
    

    // filling perm:

    int perm_index = 0;

    //i am at the place of partition 1, and it is identity
    int counter_p = 0;
    for (i = 0; i < sizes[0]; i++) {
        (*perm)[counter_p] = i;
        counter_p++;
    }
    //printf("P1        Start Index = %d and End Index %d\n", 0 , sizes[0] - 1);


    //i am at the place of separator 1, and i am moving it
    int counter_elements = sizes[0] + sizes[1] + additionalSize[0] + additionalSize[1];
    for (i = 0; i < separator_sizes[0]; i++) {
        (*perm)[counter_p] = counter_elements;
        counter_p++;
        counter_elements++;
    }
    //printf("SEP0        Start Index = %d and End Index %d\n", sizes[0] + sizes[1] + additionalSize[0], sizes[0] + sizes[1] + additionalSize[0] + separator_sizes[0] - 1);

    counter_elements = sizes[0] + additionalSize[0];
    for (i = 0; i < sizes[1]; i++) {
        (*perm)[counter_p] = counter_elements;
        counter_p++;
        counter_elements++;
    }
    //printf("P2        Start Index = %d and End Index %d\n", sizes[0] + additionalSize[0], sizes[0] + additionalSize[0] + sizes[1] - 1);

    //i am at the place of covariates, i need to move them somewhere
    counter_elements = *new_dim - ncov;
    for (int i = 0; i < ncov; i++) {
        (*perm)[counter_p] = counter_elements;  // Identity permutation for the main matrix
        counter_p++;
        counter_elements++;
    }
    //printf("covariates        Start Index = %d and End Index %d\n", *new_dim - ncov, *new_dim - ncov + ncov - 1);

    //i am at the place of the additions, i need to move them somewhere
    counter_elements = sizes[0];
    for (int i = 0; i < additionalSize[0]; i++) {
        (*perm)[counter_p] = counter_elements;  // Identity permutation for the main matrix
        counter_p++;
        counter_elements++;
    }
    //printf("additions0        Start Index = %d and End Index %d\n", sizes[0], sizes[0] + additionalSize[0] - 1);

    counter_elements = sizes[0] + additionalSize[0] + sizes[1];
    for (int i = 0; i < additionalSize[1]; i++) {
        (*perm)[counter_p] = counter_elements;  // Identity permutation for the main matrix
        counter_p++;
        counter_elements++;
    }
    //printf("additions1        Start Index = %d and End Index %d\n", sizes[0] + additionalSize[0] + sizes[1], sizes[0] + additionalSize[0] + sizes[1] + additionalSize[1]- 1);

    /*int save1 = (*perm)[0];
    int save2 = (*perm)[1];

    (*perm)[0] = (*perm)[2];
    (*perm)[1] = save1;
    (*perm)[2] = save2;*/

    // Check if perm contains unique numbers from 0 to *new_dim - 1
    bool is_unique = true;
    for (i = 0; i < perm_index; i++) {
        for (int j = i + 1; j < perm_index; j++) {
            if ((*perm)[i] == (*perm)[j]) {
                is_unique = false;
                printf("########################### CHECK:::Repeated number %d found at indices %d (%s) and %d (%s).\n", (*perm)[i], i, find_source_vector(i, sizes, additionalSize, separator_sizes[0], ncov), j, find_source_vector(j, sizes, additionalSize, separator_sizes[0], ncov));
                // Optionally, break here if only the first repetition is needed
            }
        }
        // Optionally, break here if only the first repetition is needed
    }

    if (is_unique) {
        //printf("        - CHECK PERMUTATION VECTOR : All numbers in perm are unique.\n");
    } else {
        printf("########################### CHECK:::There are repeated numbers in perm ########################### \n");
    }

    *new_nnz = full_nnz + additionalSize[0] + additionalSize[1];
    //printf("        -  *total_num_used_tiles %d\n", *total_num_used_tiles);
    *total_num_used_tiles = (*new_dim % tile_size == 0) ? (*new_dim / tile_size) : (*new_dim / tile_size + 1);
    //printf("        -  *total_num_used_tiles %d\n", *total_num_used_tiles);

    //printf("        -  full_nnz %d\n", full_nnz);
    //printf("        -  *new_nnz %d\n", *new_nnz);

    //printf(" NEEEEW DIMESNION       -  *new_dim %d\n", *new_dim);

    //----------------------------------------------------------------------------------> last part
    // Allocate new arrays
    int* newFull_i = new (std::nothrow) int[(*new_nnz)];
    int* newFull_j = new (std::nothrow) int[(*new_nnz)];
    //double* newFull_x = new (std::nothrow) double[(*new_nnz)];

    // Check for allocation failure
    if (!newFull_i || !newFull_j) {
        delete[] newFull_i;
        delete[] newFull_j;
        //delete[] newFull_x;
        sTiles::Logger::error("Failed to allocate memory.");
        return -1;  // or handle the error as appropriate
    }


    // Copy old data to new arrays
    for (int i = 0; i < full_nnz; ++i) {
        newFull_i[i] = (*indices_i)[i];
        newFull_j[i] = (*indices_j)[i];
        //newFull_x[i] = (*full_x)[i];
    }


    // Initialize remaining elements in the new arrays
    int counter = full_dim;
    for (i = full_nnz; i < (*new_nnz); ++i) {
        newFull_i[i] = counter; // or some other default value
        newFull_j[i] = counter; // or some other default value
        //newFull_x[i] = 1.0; // or some other default value
        counter++;
    }

    // Delete old arrays
    //free(*indices_i);
    //free(*indices_j);
    //delete[] *full_x;

    // Assign new arrays to the pointers
    *indices_i = newFull_i;
    *indices_j = newFull_j;
    //*full_x = newFull_x;

    int of_size = (((*total_num_used_tiles) * (*total_num_used_tiles)) - (*total_num_used_tiles))/2 + (*total_num_used_tiles);
    *of_perm = (bool*)calloc(of_size, sizeof(bool));

    //printf("------1------ \n");
    *on_off_tiles = smf_createBoolMatrix((*total_num_used_tiles));
    //printf("-----2------- \n");



    int j, index;
    int tileRow = 0, tileCol = 0, counted_tiles = 0;
    for (index = 0; index < (*new_nnz); index++) {

        int tileRow = (*perm)[(*indices_j)[index]] / tile_size;
        int tileCol = (*perm)[(*indices_i)[index]] / tile_size;

        if (tileRow <= tileCol) {

                (*on_off_tiles)[tileRow][tileCol] = true;
                if(!(*of_perm)[tileRow*(2*(*total_num_used_tiles)-tileRow-1)/2 + tileCol]){
                    (*of_perm)[tileRow*(2*(*total_num_used_tiles)-tileRow-1)/2 + tileCol] = true;
                    counted_tiles++;
                }

        } else{ 

                (*on_off_tiles)[tileCol][tileRow] = true;

                if(!(*of_perm)[tileCol*(2*(*total_num_used_tiles)-tileCol-1)/2 + tileRow]){
                    (*of_perm)[tileCol*(2*(*total_num_used_tiles)-tileCol-1)/2 + tileRow] = true;
                    counted_tiles++;
                }

        }
    }

    
    bool sumT = true, res = true;
    for (i = 0; i < (*total_num_used_tiles); i++) {
        for (j = 0; j < i; j++) {

            sumT = false;
            for (int k = 0; k < j; k++){
                res = (*on_off_tiles)[k][i] * (*on_off_tiles)[k][j];
                if(res) sumT = true;
            }

            if(sumT && !((*on_off_tiles)[j][i])){

                (*on_off_tiles)[j][i] = true;
            }


            sumT = false;
            for (int k = 0; k < j; k++){
                res = (*of_perm)[k*(2*(*total_num_used_tiles)-k-1)/2 + i] * (*of_perm)[k*(2*(*total_num_used_tiles)-k-1)/2 + j];
                if(res) sumT = true;
            }

            if(sumT && !((*of_perm)[j*(2*(*total_num_used_tiles)-j-1)/2 + i])){

                (*of_perm)[j*(2*(*total_num_used_tiles)-j-1)/2 + i] = true;
                counted_tiles++;
            }
        }
    }

    //printTileStatus((*on_off_tiles), (*total_num_used_tiles));

    /*
        printf("\n");
        for (int j = 0; j < (*total_num_used_tiles); j++) {
            for (int i = 0; i <= j; i++) {
                
                //printf("%s ", matrix[i*(2*num_tiles-i-1)/2 + j] ? " on  " : " off ");

                if(j > (43-1) && j <= (43+42-2) && i <= (43-1)) printf("%s ", (*of_perm)[i*(2*(*total_num_used_tiles)-i-1)/2 + j] ? "?" : "o");
                else printf("%s ", (*of_perm)[i*(2*(*total_num_used_tiles)-i-1)/2 + j] ? "1" : " ");

            }
            printf("\n");
        }

                printf("\n");
        for (int j = 0; j < (*total_num_used_tiles); j++) {
            for (int i = 0; i <= j; i++) {
                
                //printf("%s ", matrix[i*(2*num_tiles-i-1)/2 + j] ? " on  " : " off ");
                printf("%s ", (*of_perm)[i*(2*(*total_num_used_tiles)-i-1)/2 + j] ? "1" : " ");

            }
            printf("\n");
        }*/
    
   // exit(0);
        //---------------------------------------



    return counted_tiles; // or your function's successful completion code
}

// ---------------------------------------------------------------------------
// Node-level graph coarsening for DOF-structured FEM matrices.
// Groups consecutive k DOFs into one node, builds a coarsened graph using the
// UNION of all DOF-level neighbors per node, runs METIS on the small graph,
// and expands the permutation back to DOF level.
// Returns true on success (perm/iperm/sizes filled), false to fall back.
// ---------------------------------------------------------------------------
static bool try_node_level_metis(idx_t nvtxs, idx_t* xadj, idx_t* adjncy,
                                  idx_t npes, idx_t* options,
                                  idx_t* perm, idx_t* iperm, idx_t* sizes,
                                  int size_sizes, int use_ndp) {
    const int candidates[] = {6, 4, 3, 2};
    const int ncand = 4;

    std::vector<idx_t> node_nbrs;

    for (int ci = 0; ci < ncand; ci++) {
        const idx_t k = candidates[ci];
        if (nvtxs % k != 0) continue;

        const idx_t coarse_n = nvtxs / k;
        if (coarse_n < 100) continue;

        // --- Build coarsened graph (union of DOF-level neighbors per node) ---
        std::vector<idx_t> c_xadj(coarse_n + 1);
        std::vector<idx_t> c_adjncy;
        c_xadj[0] = 0;

        for (idx_t node = 0; node < coarse_n; node++) {
            node_nbrs.clear();
            // Union over all k DOFs
            for (idx_t d = 0; d < k; d++) {
                const idx_t dof = node * k + d;
                for (idx_t j = xadj[dof]; j < xadj[dof + 1]; j++) {
                    idx_t nn = adjncy[j] / k;
                    if (nn != node) node_nbrs.push_back(nn);
                }
            }
            std::sort(node_nbrs.begin(), node_nbrs.end());
            node_nbrs.erase(std::unique(node_nbrs.begin(), node_nbrs.end()), node_nbrs.end());

            for (auto nn : node_nbrs) c_adjncy.push_back(nn);
            c_xadj[node + 1] = static_cast<idx_t>(c_adjncy.size());
        }

        // Sanity check: coarsening should significantly reduce the edge count.
        // For a true k-DOF matrix, coarsened edges ≈ original / k².
        // Reject if coarsened edges are more than original / k (no real benefit).
        const idx_t orig_edges = xadj[nvtxs];
        const idx_t coarse_edges = static_cast<idx_t>(c_adjncy.size());
        if (coarse_edges * k > orig_edges) {
            sTiles::Logger::errorf("[ND-COARSEN] block_size=%d: coarsened %d edges vs original %d — "
                    "not enough reduction, skipping",
                    (int)k, (int)coarse_edges, (int)orig_edges);
            continue;
        }

        sTiles::Logger::errorf("[ND-COARSEN] block_size=%d: %d nodes (from %d DOFs), "
                "edges %d -> %d (%.1fx reduction)",
                (int)k, (int)coarse_n, (int)nvtxs,
                (int)orig_edges, (int)coarse_edges,
                (double)orig_edges / (double)coarse_edges);

        // --- Run METIS on the small graph ---
        idx_t* c_perm  = new idx_t[coarse_n];
        idx_t* c_iperm = new idx_t[coarse_n];
        idx_t* c_sizes = new idx_t[size_sizes];

        double t_m = omp_get_wtime();
        int ret;
        if (use_ndp) {
            ret = METIS_NodeNDP(coarse_n, c_xadj.data(), c_adjncy.data(),
                                NULL, npes, options, c_perm, c_iperm, c_sizes);
            sTiles::Logger::errorf("[ND-COARSEN] METIS_NodeNDP on coarsened: %.3f s (ret=%d)",
                    omp_get_wtime() - t_m, ret);
        } else {
            idx_t coarse_n_mut = coarse_n;
            ret = METIS_NodeND(&coarse_n_mut, c_xadj.data(), c_adjncy.data(),
                               NULL, options, c_perm, c_iperm);
            sTiles::Logger::errorf("[ND-COARSEN] METIS_NodeND on coarsened: %.3f s (ret=%d)",
                    omp_get_wtime() - t_m, ret);
        }

        if (ret != METIS_OK) {
            delete[] c_perm; delete[] c_iperm; delete[] c_sizes;
            sTiles::Logger::errorf("[ND-COARSEN] METIS failed on coarsened graph, falling back");
            continue;
        }

        // --- Expand permutation to DOF level ---
        for (idx_t i = 0; i < coarse_n; i++) {
            for (idx_t d = 0; d < k; d++) {
                perm [i * k + d] = c_perm [i] * k + d;
                iperm[i * k + d] = c_iperm[i] * k + d;
            }
        }

        // Scale separator sizes (only valid for NDP)
        if (use_ndp) {
            for (int i = 0; i < size_sizes; i++) {
                sizes[i] = c_sizes[i] * k;
            }
        }

        delete[] c_perm; delete[] c_iperm; delete[] c_sizes;
        return true;
    }

    return false;  // no block structure found — caller runs full METIS
}

int metis_ordering_nested_dissection(int* xadj_, int* adjncy_, int nvtxs_, int upper_nnz, int** perm_, int** iperm_, int num_sep, int** sizes_)  {

    // Dense graphs (avg_degree > 50) rarely have structurally-equivalent vertices,
    // so COMPRESS adds overhead without benefit. Sparse graphs (lattices, meshes)
    // often have many equivalent interior nodes — COMPRESS helps there.
    const double avg_degree = nvtxs_ > 0 ? (2.0 * upper_nnz) / nvtxs_ : 0.0;
    const int use_compress = (avg_degree <= 50) ? 1 : 0;

    // Hub-detection guard: skip METIS on graphs with extreme degree variance.
    // Hub nodes cause coarsening to stall, making METIS orders of magnitude slower
    // with no quality benefit (SCOTCH handles these graphs better).
    if (avg_degree > 10.0) {
        int max_deg = 0;
        for (int i = 0; i < nvtxs_; ++i) {
            int deg = xadj_[i+1] - xadj_[i];
            if (deg > max_deg) max_deg = deg;
        }
        if (max_deg > 5 * avg_degree) {
            //sTiles::Logger::errorf("[metis_nd] skipping: hub detected (max_deg=%d > 5*avg_deg=%.1f) — identity permutation",
            //        max_deg, avg_degree);
            g_nd_hub_skipped = true;
            if (perm_ && *perm_ && iperm_ && *iperm_)
                for (int i = 0; i < nvtxs_; ++i) { (*perm_)[i] = i; (*iperm_)[i] = i; }
            if (sizes_) {
                int sz = (int)std::pow(2, std::log2(num_sep) + 1) - 1;
                *sizes_ = new int[sz]();
            }
            return 0;
        }
    }

    idx_t options[METIS_NOPTIONS];
    if (g_nd_use_ndp) {
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_COMPRESS] = use_compress;
        options[METIS_OPTION_PFACTOR]  = 200;
        options[METIS_OPTION_NSEPS]    = 4;
        options[METIS_OPTION_SEED]     = 2704;
    } else {
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_NUMBERING] = 0;
        options[METIS_OPTION_SEED]      = 2704;
        options[METIS_OPTION_COMPRESS]  = use_compress;
        options[METIS_OPTION_PFACTOR]   = 200;  // prune dense vertices
        options[METIS_OPTION_NITER]     = 3;    // fewer refinement passes (default=10)
    }
    // Runtime overrides (defaults above unchanged unless the env var is set).
    // STILES_METIS_NITER=10 -> full-quality refinement (MUMPS-style, less fill).
    if (const char* e = std::getenv("STILES_METIS_NITER"))   options[METIS_OPTION_NITER]   = std::atoi(e);
    if (const char* e = std::getenv("STILES_METIS_PFACTOR")) options[METIS_OPTION_PFACTOR] = std::atoi(e);
    if (const char* e = std::getenv("STILES_METIS_SEED"))    options[METIS_OPTION_SEED]    = std::atoi(e);

    idx_t nvtxs = static_cast<idx_t>(nvtxs_);
    idx_t npes = static_cast<idx_t>(num_sep);

    // Allocate and copy xadj
    idx_t* xadj = new idx_t[nvtxs+1];
    for (idx_t i = 0; i < nvtxs+1; i++) {
        xadj[i] = static_cast<idx_t>(xadj_[i]);
    }

    // Allocate and copy adjncy
    idx_t adjncy_size = static_cast<idx_t>(2*upper_nnz);
    idx_t* adjncy = new idx_t[adjncy_size];
    for (idx_t i = 0; i < adjncy_size; i++) {
        adjncy[i] = static_cast<idx_t>(adjncy_[i]);
    }

    // Allocate memory for perm and iperm
    idx_t* perm = new idx_t[nvtxs];
    idx_t* iperm = new idx_t[nvtxs];

    int size_sizes = (int)pow(2, log2(num_sep) + 1) - 1;
    idx_t *sizes = new idx_t[size_sizes];

    //double _t0 = omp_get_wtime();
    bool coarsened = try_node_level_metis(nvtxs, xadj, adjncy, npes, options,
                                          perm, iperm, sizes, size_sizes, g_nd_use_ndp);
    //sTiles::Logger::errorf("[metis_nd] coarsen_attempt=%.3fs coarsened=%d n=%d edges=%d",
    //        omp_get_wtime() - _t0, (int)coarsened, (int)nvtxs, (int)(2*upper_nnz));

    if (!coarsened) {
        //double _t1 = omp_get_wtime();
        if (g_nd_use_ndp) {
            int ret = METIS_NodeNDP(nvtxs, xadj, adjncy, NULL, npes, options, perm, iperm, sizes);
            //sTiles::Logger::errorf("[metis_nd] METIS_NodeNDP=%.3fs ret=%d", omp_get_wtime()-_t1, ret);
        } else {
            int ret = METIS_NodeND(&nvtxs, xadj, adjncy, NULL, options, perm, iperm);
            //sTiles::Logger::errorf("[metis_nd] METIS_NodeND=%.3fs ret=%d n=%d avg_deg=%.1f compress=%d niter=%d",
            //        omp_get_wtime()-_t1, ret, (int)nvtxs, avg_degree,
            //        (int)options[METIS_OPTION_COMPRESS],
            //        (int)options[METIS_OPTION_NITER]);
            for (int i = 0; i < size_sizes; i++) sizes[i] = 0;
        }
    }

    if (!perm_ || !iperm_ || !*perm_ || !*iperm_) {
        sTiles::Logger::errorf("metis_ordering_nested_dissection: null permutation buffers provided.");
        delete[] xadj; delete[] adjncy; delete[] perm; delete[] iperm; delete[] sizes;
        return -1;
    }

    for (idx_t i = 0; i < nvtxs; i++) {
        (*perm_)[i] = static_cast<int>(perm[i]);
        (*iperm_)[i] = static_cast<int>(iperm[i]);
    }
    *sizes_ = new int[size_sizes];
    for (idx_t i = 0; i < size_sizes; i++) (*sizes_)[i] = static_cast<int>(sizes[i]);

    delete[] xadj;
    delete[] adjncy;
    delete[] perm;
    delete[] iperm;
    delete[] sizes;

    return 0; // Success
}

int metis_ordering_nested_dissection_m(int* xadj_, int* adjncy_, int nvtxs_, int upper_nnz, int** perm_, int** iperm_, int num_sep, int** sizes_, int added)  {

    const double avg_degree = nvtxs_ > 0 ? (2.0 * upper_nnz) / nvtxs_ : 0.0;
    const int use_compress = (avg_degree <= 50) ? 1 : 0;

    // Hub-detection guard: skip METIS on graphs with extreme degree variance.
    // Hub nodes cause coarsening to stall, making METIS orders of magnitude slower
    // with no quality benefit (SCOTCH handles these graphs better).
    if (avg_degree > 10.0) {
        int max_deg = 0;
        for (int i = 0; i < nvtxs_; ++i) {
            int deg = xadj_[i+1] - xadj_[i];
            if (deg > max_deg) max_deg = deg;
        }
        if (max_deg > 5 * avg_degree) {
            //sTiles::Logger::errorf("[metis_nd] skipping: hub detected (max_deg=%d > 5*avg_deg=%.1f) — identity permutation",
            //        max_deg, avg_degree);
            g_nd_hub_skipped = true;
            if (perm_ && *perm_ && iperm_ && *iperm_)
                for (int i = 0; i < nvtxs_; ++i) { (*perm_)[i] = i; (*iperm_)[i] = i; }
            if (sizes_) {
                int sz = (int)std::pow(2, std::log2(num_sep) + 1) - 1;
                *sizes_ = new int[sz]();
            }
            return 0;
        }
    }

    idx_t options[METIS_NOPTIONS];
    if (g_nd_use_ndp) {
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_COMPRESS] = use_compress;
        options[METIS_OPTION_PFACTOR]  = 200;
        options[METIS_OPTION_NSEPS]    = 4;
        options[METIS_OPTION_SEED]     = 2704;
    } else {
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_NUMBERING] = 0;
        options[METIS_OPTION_SEED]      = 2704;
        options[METIS_OPTION_COMPRESS]  = use_compress;
        options[METIS_OPTION_PFACTOR]   = 200;  // prune dense vertices
        options[METIS_OPTION_NITER]     = 3;    // fewer refinement passes (default=10)
    }
    // Runtime overrides (defaults above unchanged unless the env var is set).
    // STILES_METIS_NITER=10 -> full-quality refinement (MUMPS-style, less fill).
    if (const char* e = std::getenv("STILES_METIS_NITER"))   options[METIS_OPTION_NITER]   = std::atoi(e);
    if (const char* e = std::getenv("STILES_METIS_PFACTOR")) options[METIS_OPTION_PFACTOR] = std::atoi(e);
    if (const char* e = std::getenv("STILES_METIS_SEED"))    options[METIS_OPTION_SEED]    = std::atoi(e);

    idx_t nvtxs = static_cast<idx_t>(nvtxs_);
    idx_t npes = static_cast<idx_t>(num_sep);

    // Allocate and copy xadj
    idx_t* xadj = new idx_t[nvtxs+1];
    for (idx_t i = 0; i < nvtxs+1; i++) {
        xadj[i] = static_cast<idx_t>(xadj_[i]);
    }

    // Allocate and copy adjncy
    idx_t adjncy_size = static_cast<idx_t>(2*upper_nnz);
    idx_t* adjncy = new idx_t[adjncy_size];
    for (idx_t i = 0; i < adjncy_size; i++) {
        adjncy[i] = static_cast<idx_t>(adjncy_[i]);
    }

    // Allocate memory for perm and iperm
    idx_t* perm = new idx_t[nvtxs];
    idx_t* iperm = new idx_t[nvtxs];

    int size_sizes = (int)pow(2, log2(num_sep) + 1) - 1;
    idx_t *sizes = new idx_t[size_sizes];

    bool coarsened = try_node_level_metis(nvtxs, xadj, adjncy, npes, options,
                                          perm, iperm, sizes, size_sizes, g_nd_use_ndp);
    if (!coarsened) {
        if (g_nd_use_ndp) {
            METIS_NodeNDP(nvtxs, xadj, adjncy, NULL, npes, options, perm, iperm, sizes);
        } else {
            METIS_NodeND(&nvtxs, xadj, adjncy, NULL, options, perm, iperm);
            for (int i = 0; i < size_sizes; i++) sizes[i] = 0;
        }
    }

    if (!perm_ || !iperm_ || !*perm_ || !*iperm_) {
        sTiles::Logger::errorf("metis_ordering_nested_dissection_m: null permutation buffers provided.");
        delete[] xadj; delete[] adjncy; delete[] perm; delete[] iperm; delete[] sizes;
        return -1;
    }

    const int total = nvtxs + added;
    for (idx_t i = 0; i < nvtxs; i++) {
        (*perm_)[i] = static_cast<int>(perm[i]);
        (*iperm_)[i] = static_cast<int>(iperm[i]);
    }
    *sizes_ = new int[size_sizes];
    for (idx_t i = 0; i < size_sizes; i++) (*sizes_)[i] = static_cast<int>(sizes[i]);

    // BUGFIX: Add the 'added' fixed columns to the separator size (last partition)
    // The fixed columns are appended at the end and should be part of the separator
    if (added > 0 && size_sizes >= 3) {
        (*sizes_)[2] += added;  // Add to separator (third partition)
    }

    /*printf("hereeeeeeeeee \n");

    int min_perm = *std::min_element(*perm_, *perm_ + nvtxs);
    int max_perm = *std::max_element(*perm_, *perm_ + nvtxs);

    // Finding min and max values for iperm_
    int min_iperm = *std::min_element(*iperm_, *iperm_ + nvtxs);
    int max_iperm = *std::max_element(*iperm_, *iperm_ + nvtxs);

    // Print the results
    std::cout << "perm_: min = " << min_perm << ", max = " << max_perm << std::endl;
    std::cout << "iperm_: min = " << min_iperm << ", max = " << max_iperm << std::endl;*/

    if (added > 0) {
        for (int i = nvtxs; i < total; ++i) {
            (*perm_)[i] = i;
            (*iperm_)[i] = i;
        }
    }

    delete[] xadj;
    delete[] adjncy;
    delete[] perm;
    delete[] iperm;
    delete[] sizes;

    return 0; // Success
}

void convert_for_nd_submatrix_of_full_matrix(int** row_indices, int** col_indices, int nnz, int node_num, int m, int** perm, int** iperm, int num_sep, int** sizes) {

    int dim = node_num - m;
    std::vector<int> new_rows_vector;
    std::vector<int> new_cols_vector;

    size_t counter_nnz = 0; int row, col;
    for (int i = 0; i < nnz; i++) {
        row = (*row_indices)[i];
        col = (*col_indices)[i];
        if (row < dim && col < dim) {

            new_rows_vector.push_back(row+1);
            new_cols_vector.push_back(col+1);
            counter_nnz++;
            
        }
    }
    size_t upper_nnz = counter_nnz - dim;
    std::vector<int> xadj;
    std::vector<int> adjncy;
    std::vector<int> full_i;    
    std::vector<int> full_j;
    std::vector<int> full_sorted_order_;

    // Resize and initialize to zero
    full_i.resize(2 * upper_nnz + dim, 0);
    full_j.resize(2 * upper_nnz + dim, 0);

    std::copy(new_rows_vector.begin(), new_rows_vector.end(), full_i.begin());
    std::copy(new_cols_vector.begin(), new_cols_vector.end(), full_j.begin());

    std::vector<int> indices;
    indices.reserve(upper_nnz);

    int count1 = 0, count2 = 0;
    size_t offset = counter_nnz;  // Starting index for new elements
    for (size_t idx = 0; idx < counter_nnz; ++idx) {
        if (new_rows_vector[idx] != new_cols_vector[idx]) {
            if (offset < full_i.size()) {  // Check to prevent out-of-bounds access
                full_i[offset] = new_cols_vector[idx];
                full_j[offset] = new_rows_vector[idx];
                offset++;
            }
            indices.push_back(idx);
            count1++;
        } else {
            count2++;
        }
    }

    full_sorted_order_.resize(counter_nnz);
    std::iota(full_sorted_order_.begin(), full_sorted_order_.end(), 0); // Fill it with 0 to dim-1.
    full_sorted_order_.insert(full_sorted_order_.end(), indices.begin(), indices.end());

    std::vector<size_t> sorted_indices(full_i.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);


    sTiles::sort(sorted_indices.begin(), sorted_indices.end(),
    [&full_i, &full_j](size_t a, size_t b) {
        return full_j[a] < full_j[b] || (full_j[a] == full_j[b] && full_i[a] < full_i[b]);
    });

    std::vector<int> new_order(full_sorted_order_.size());
    for (int i = 0; i < sorted_indices.size(); ++i) {
        new_order[i] = full_sorted_order_[sorted_indices[i]];
    }

    full_sorted_order_ = new_order;

    std::vector<int> new_full_i(full_i.size());
    std::vector<int> new_full_j(full_j.size());

    for (int i = 0; i < sorted_indices.size(); ++i) {
        new_full_i[i] = full_i[sorted_indices[i]];
        new_full_j[i] = full_j[sorted_indices[i]];
    }

    // Now replace the old vectors with the new ordered vectors

    full_i = new_full_i;
    full_j = new_full_j;

    xadj.resize(dim + 1);
    std::fill(xadj.begin(), xadj.end(), 0);

    adjncy.reserve(2*upper_nnz);

    if(false){
        xadj[0] = 0; xadj[1] = 0;
        int idx = 0, row_index = 1;
        int ele = full_j[0];
        do{
            size_t start_index = 0;
            while(ele==full_j[idx]){
                start_index++;
                if(full_i[idx]!=row_index) adjncy.push_back(full_i[idx]-1);
                idx++;
            }

            if(row_index<(dim + 1)) xadj[row_index] = xadj[row_index-1] + start_index - 1;
            row_index++;
            ele = full_j[idx];

        }while(idx<full_j.size());
    }

    xadj[0] = 0;
    if (dim >= 1) xadj[1] = 0;

    std::size_t idx = 0;
    int row_index = 1;

    while (idx < full_j.size()) {
        const int ele = full_j[idx];
        std::size_t start_index = 0;

        while (idx < full_j.size() && full_j[idx] == ele) {
            start_index++;
            if (full_i[idx] != row_index) {
                adjncy.push_back(full_i[idx] - 1);
            }
            idx++;
        }

        if (row_index < dim + 1) {
            xadj[row_index] = xadj[row_index - 1] + static_cast<int>(start_index) - 1;
        }
        row_index++;
    }

    metis_ordering_nested_dissection_m(xadj.data(), adjncy.data(), dim, upper_nnz, perm, iperm, num_sep, sizes, m);

}

void wrapper_nd(int** csr_i, int** csr_j, int dim, int nnz, int m, int** perm, int** iperm, int num_sep, int** sizes){

        // Input is 1-based (caller adds 1 before calling this function).
        // Build symmetric CSR (0-based) using O(nnz + dim) counting sort.
        // Only the first (dim - m) nodes are ordered; the last m fixed columns
        // are excluded from the graph and left as identity at the end.
        const int active_dim = dim - m;

        // Step 1: count off-diagonal degrees (active nodes only)
        std::vector<int> xadj(active_dim + 1, 0);
        for (int k = 0; k < nnz; ++k) {
            int i = (*csr_i)[k] - 1;  // convert to 0-based
            int j = (*csr_j)[k] - 1;
            if (i != j && i < active_dim && j < active_dim) {
                xadj[i + 1]++;
                xadj[j + 1]++;
            }
        }

        // Step 2: prefix sum → row pointers
        for (int i = 1; i <= active_dim; ++i) xadj[i] += xadj[i - 1];

        // Step 3: fill adjacency
        const int sym_nnz = xadj[active_dim];
        std::vector<int> adjncy(sym_nnz);
        std::vector<int> pos(xadj.begin(), xadj.begin() + active_dim);
        for (int k = 0; k < nnz; ++k) {
            int i = (*csr_i)[k] - 1;
            int j = (*csr_j)[k] - 1;
            if (i != j && i < active_dim && j < active_dim) {
                adjncy[pos[i]++] = j;
                adjncy[pos[j]++] = i;
            }
        }

        // Sort each node's adjacency list — needed for good METIS coarsening quality
        for (int i = 0; i < active_dim; ++i)
            std::sort(adjncy.begin() + xadj[i], adjncy.begin() + xadj[i + 1]);

        const int upper_nnz = sym_nnz / 2;  // off-diagonal pairs
        metis_ordering_nested_dissection(xadj.data(), adjncy.data(), active_dim, upper_nnz, perm, iperm, num_sep, sizes);

        // Fixed columns: identity mapping
        if (perm && *perm && iperm && *iperm)
            for (int i = active_dim; i < dim; ++i) { (*perm)[i] = i; (*iperm)[i] = i; }

}

// Direct METIS_NodeND — no coarsening, no hub detection, no NDP.
// Best quality for FEM/structural matrices where METIS excels.
extern "C"
int stiles_runMETIS_direct(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, const sTiles::SharedAdjCSR* shared) {
    const int active_dim = N - m;
    if (active_dim <= 0) return 1;

    // METIS input arrays (idx_t). Filled either from the prebuilt canonical graph
    // (identical to the build below — both are 0-based, symmetric, dedup, no
    // diagonal) or from the local build when no shared graph is supplied.
    std::vector<idx_t> xadj_m;
    std::vector<idx_t> adjncy_m;
    if (shared && shared->valid_for(active_dim)) {
        const int ne = shared->xadj[active_dim];
        xadj_m.resize(static_cast<std::size_t>(active_dim) + 1);
        adjncy_m.resize(static_cast<std::size_t>(ne));
        for (int i = 0; i <= active_dim; ++i) xadj_m[i] = static_cast<idx_t>(shared->xadj[i]);
        for (int e = 0; e < ne; ++e)          adjncy_m[e] = static_cast<idx_t>(shared->adjncy[e]);
    } else {
    // Build symmetric CSR (0-based) from COO input
    std::vector<int> xadj(active_dim + 1, 0);
    for (int k = 0; k < nnz; ++k) {
        int i = (*csr_i)[k];
        int j = (*csr_j)[k];
        if (i != j && i < active_dim && j < active_dim) {
            xadj[i + 1]++;
            xadj[j + 1]++;
        }
    }
    for (int i = 1; i <= active_dim; ++i) xadj[i] += xadj[i - 1];

    const int sym_nnz = xadj[active_dim];
    std::vector<int> adjncy(sym_nnz);
    std::vector<int> pos(xadj.begin(), xadj.begin() + active_dim);
    for (int k = 0; k < nnz; ++k) {
        int i = (*csr_i)[k];
        int j = (*csr_j)[k];
        if (i != j && i < active_dim && j < active_dim) {
            adjncy[pos[i]++] = j;
            adjncy[pos[j]++] = i;
        }
    }
    // Sort and deduplicate each adjacency list
    int dedup_nnz = 0;
    for (int i = 0; i < active_dim; ++i) {
        sTiles::sort(adjncy.begin() + xadj[i], adjncy.begin() + xadj[i + 1]);
        auto it = std::unique(adjncy.begin() + xadj[i], adjncy.begin() + xadj[i + 1]);
        int new_len = static_cast<int>(it - (adjncy.begin() + xadj[i]));
        // Compact in-place
        if (dedup_nnz != xadj[i]) {
            std::copy(adjncy.begin() + xadj[i], it, adjncy.begin() + dedup_nnz);
        }
        xadj[i] = dedup_nnz;
        dedup_nnz += new_len;
    }
    xadj[active_dim] = dedup_nnz;

    // Copy to idx_t arrays for METIS
    xadj_m.resize(active_dim + 1);
    adjncy_m.resize(dedup_nnz);
    for (int i = 0; i <= active_dim; ++i) xadj_m[i] = static_cast<idx_t>(xadj[i]);
    for (int i = 0; i < dedup_nnz; ++i) adjncy_m[i] = static_cast<idx_t>(adjncy[i]);
    }

    idx_t nvtxs = static_cast<idx_t>(active_dim);
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_SEED]      = 42;
    options[METIS_OPTION_COMPRESS]  = 1;      // compress structurally equivalent vertices
    options[METIS_OPTION_PFACTOR]   = 200;    // prune dense vertices
    options[METIS_OPTION_NSEPS]     = 4;      // try 4 separators at each bisection
    options[METIS_OPTION_NITER]     = 10;     // refinement iterations

    // Runtime overrides (defaults above unchanged unless env var set). Lets us A/B
    // sТiles' tuned METIS vs MUMPS-style defaults (PFACTOR=0, NSEPS=1) without rebuild.
    if (const char* e = std::getenv("STILES_METIS_NITER"))   options[METIS_OPTION_NITER]   = std::atoi(e);
    if (const char* e = std::getenv("STILES_METIS_PFACTOR")) options[METIS_OPTION_PFACTOR] = std::atoi(e);
    if (const char* e = std::getenv("STILES_METIS_NSEPS"))   options[METIS_OPTION_NSEPS]   = std::atoi(e);
    if (const char* e = std::getenv("STILES_METIS_SEED"))    options[METIS_OPTION_SEED]    = std::atoi(e);

    std::vector<idx_t> mp(active_dim), mip(active_dim);
    int ret = METIS_NodeND(&nvtxs, xadj_m.data(), adjncy_m.data(), NULL, options, mp.data(), mip.data());

    if (ret != METIS_OK) return 1;

    // METIS: mp[old]=new, mip[new]=old
    // Match SCOTCH convention: *perm = iperm (new→old), *iperm = perm (old→new)
    for (int i = 0; i < active_dim; ++i) {
        (*iperm)[i] = static_cast<int>(mp[i]);    // old→new (METIS perm)
        (*perm)[i]  = static_cast<int>(mip[i]);   // new→old (METIS iperm)
    }
    // Fixed columns: identity
    for (int i = active_dim; i < N; ++i) { (*perm)[i] = i; (*iperm)[i] = i; }

    return 0;
}

namespace sTiles {

void stiles_runND_original(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, int num_sep, int** sizes) {

    int* save_rows = (int*)malloc(nnz * sizeof(int));
    int* save_cols = (int*)malloc(nnz * sizeof(int));
    int* pperm = (int*)malloc(N * sizeof(int));
    bool double_perm = false;

    if(m==0){

        for(int i=0; i<nnz;i++){

            save_rows[i] = (*csr_i)[i];
            save_cols[i] = (*csr_j)[i];
        }

        double_perm = true;
        m =  stiles_createSmartPermutation(csr_i, csr_j, nnz, N, &pperm);

    }

    if(m > 0 ){

        convert_for_nd_submatrix_of_full_matrix(csr_i, csr_j, nnz, N, m, perm, iperm, num_sep, sizes);

    }else{

        for(int i=0; i<nnz;i++) {(*csr_i)[i] +=1;  (*csr_j)[i] +=1;}
        wrapper_nd(csr_i, csr_j, N, nnz, m, perm, iperm, num_sep, sizes);
        for(int i=0; i<nnz;i++) {(*csr_i)[i] -=1;  (*csr_j)[i] -=1;}

    }
    
    if(double_perm){

        for(int i=0; i<nnz;i++){
          (*csr_i)[i] = save_rows[i];
          (*csr_j)[i] = save_cols[i];
        }

        int* newperm = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) {
          newperm[i] = (*iperm)[pperm[i]];
        }

        if (!newperm) {
            sTiles::Logger::errorf("Memory allocation failed for newperm.");
            free(save_rows);
            free(save_cols);
            free(pperm);
            return;
        }

        for (int i = 0; i < N; i++) {
          (*iperm)[i] = newperm[i];
        }

        for (int i = 0; i < N; i++) {
          (*perm)[(*iperm)[i]] =  i;
        }

        free(newperm);

    }

}

void stiles_runND(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, int num_sep, int** sizes) {

    g_nd_hub_skipped = false;
    for(int i=0; i<nnz;i++) {(*csr_i)[i] +=1;  (*csr_j)[i] +=1;}
    wrapper_nd(csr_i, csr_j, N, nnz, m, perm, iperm, num_sep, sizes);
    for(int i=0; i<nnz;i++) {(*csr_i)[i] -=1;  (*csr_j)[i] -=1;}

}

}

void stiles_ND_RCM(int** csr_i, int** csr_j, int N, int nnz, int m, int** perm, int** iperm, int num_sep, int** sizes) {

    int* save_rows = (int*)malloc(nnz * sizeof(int));
    int* save_cols = (int*)malloc(nnz * sizeof(int));
    int* pperm = (int*)malloc(N * sizeof(int));
    bool double_perm = false;

    if(m==0){

        for(int i=0; i<nnz;i++){

            save_rows[i] = (*csr_i)[i];
            save_cols[i] = (*csr_j)[i];

        }

        double_perm = true;
        m =  stiles_createSmartPermutation(csr_i, csr_j, nnz, N, &pperm);
    }

    if(m > 0 ){

        convert_for_nd_submatrix_of_full_matrix(csr_i, csr_j, nnz, N, m, perm, iperm, num_sep, sizes);

    }else{
    
        for(int i=0; i<nnz;i++) {(*csr_i)[i] +=1;  (*csr_j)[i] +=1;}
        wrapper_nd(csr_i, csr_j, N, nnz, m, perm, iperm, num_sep, sizes);
        for(int i=0; i<nnz;i++) {(*csr_i)[i] -=1;  (*csr_j)[i] -=1;}

    }
    
    if(double_perm){

        for(int i=0; i<nnz;i++){
          (*csr_i)[i] = save_rows[i];
          (*csr_j)[i] = save_cols[i];
        }

        int* newperm = (int*)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) {
          newperm[i] = (*iperm)[pperm[i]];
        }

        if (!newperm) {
            sTiles::Logger::errorf("Memory allocation failed for newperm.");
            free(save_rows);
            free(save_cols);
            free(pperm);
            return;
        }

        for (int i = 0; i < N; i++) {
          (*iperm)[i] = newperm[i];
        }

        for (int i = 0; i < N; i++) {
          (*perm)[(*iperm)[i]] =  i;
        }

        free(newperm);

    }

    //std::cout << (*sizes)[0] << std::endl;
    //std::cout << (*sizes)[1] << std::endl;
    //std::cout << (*sizes)[2] << std::endl;

}
