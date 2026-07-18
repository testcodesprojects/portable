#ifndef STILES_VERIFY_HPP
#define STILES_VERIFY_HPP

#ifdef STILE_VERIFY_AGAINST_OLD

#include <cmath>
#include <cstdio>
#include <vector>
#include <complex>
#include <functional>
#include <cstring> // For memcpy

namespace sTiles {
namespace verify {

// --- NormTraits helper (unchanged) ---
template <typename T>
struct NormTraits {
    static double magnitude_squared(T val) { return static_cast<double>(val) * static_cast<double>(val); }
};
template <typename T>
struct NormTraits<std::complex<T>> {
    static double magnitude_squared(std::complex<T> val) { return std::norm(val); }
};

// --- Templated comparison function (unchanged) ---
template <typename T>
void compare_tiles(const char* kernel_name,
                   int M, int N,
                   const T* result_new, int lda_new,
                   const T* result_old, int lda_old,
                   double tolerance = 1e-12)
{
    double error_norm_sq = 0.0;
    double old_norm_sq = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            T diff = result_new[j * lda_new + i] - result_old[j * lda_old + i];
            error_norm_sq += NormTraits<T>::magnitude_squared(diff);
            old_norm_sq += NormTraits<T>::magnitude_squared(result_old[j * lda_old + i]);
        }
    }
    double error_norm = std::sqrt(error_norm_sq);
    double old_norm = std::sqrt(old_norm_sq);
    double relative_error = (old_norm > 0) ? (error_norm / old_norm) : error_norm;
    if (relative_error > tolerance) {
        sTiles::Logger::errorf("[VERIFY FAILED] %s: Relative error = %e > tolerance (%e)", kernel_name, relative_error, tolerance);
    }
}


// --- *** CORRECTED *** Verification Wrapper ---
template <typename T, typename NewOp, typename OldOp>
void verify_kernel_call(
    const char* kernel_name,
    int M_out, int N_out,
    T* output_tile, int lda_output, // The tile to be modified and its actual LDA
    NewOp new_op,
    OldOp old_op
) {
    // 1. Allocate a CONTIGUOUS buffer for the old API result.
    // Its size is M_out * N_out elements. Its own leading dimension is M_out.
    size_t tile_elements = (size_t)M_out * N_out;
    T* tile_for_old_api = new T[tile_elements];

    // **SAFER MEMCPY**: Copy the original data from the (potentially non-contiguous)
    // output tile into our new contiguous buffer, one column at a time.
    for (int j = 0; j < N_out; ++j) {
        memcpy(
            tile_for_old_api + (size_t)j * M_out, // Destination: contiguous
            output_tile + (size_t)j * lda_output,      // Source: potentially strided
            (size_t)M_out * sizeof(T)            // Size: one column
        );
    }

    // 2. Execute the new operation. It modifies `output_tile` in place.
    new_op();

    // 3. Execute the old operation on the copied, contiguous data.
    // The old_op lambda must be passed the correct LDA for its buffer, which is M_out.
    old_op(tile_for_old_api, M_out);

    // 4. Compare the results.
    compare_tiles<T>(kernel_name, M_out, N_out,
                     output_tile,      lda_output, // New result and its LDA
                     tile_for_old_api, M_out);     // Old result and its contiguous LDA

    // 5. Cleanup
    delete[] tile_for_old_api;
}

} // namespace verify
} // namespace sTiles

// --- *** CORRECTED *** Macro for the call site ---
// It now takes the LDA of the output tile as a parameter.
// The `old_op_lambda` is expected to accept two arguments: the tile pointer and its LDA.
#define STILE_VERIFY_EXEC(kernel_name, M_out, N_out, output_tile, lda_output, new_op_lambda, old_op_lambda) \
    sTiles::verify::verify_kernel_call(kernel_name, M_out, N_out, output_tile, lda_output, new_op_lambda, old_op_lambda)

#else // STILE_VERIFY_AGAINST_OLD is NOT defined

// If verification is disabled, this macro simply expands to executing the new code.
#define STILE_VERIFY_EXEC(kernel_name, M_out, N_out, output_tile, lda_output, new_op_lambda, old_op_lambda) \
    new_op_lambda()

#endif // STILE_VERIFY_AGAINST_OLD
#endif // STILES_VERIFY_HPP
