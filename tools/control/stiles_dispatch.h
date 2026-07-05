
#ifndef _STILES_DISPATCH_H_
#define _STILES_DISPATCH_H_

// Core sTiles framework headers
#include "../include/stiles_internal.h"
#include "../common/stiles_types.hpp"
#include "../common/stiles_logger.hpp"
#include "context.h"
#include "control.h"
#include "auxiliary.h"

// System and C++ standard library headers
#include <cstddef> // For std::size_t
#include <cstring> // For std::memcpy

#if defined(_WIN32) || defined(_WIN64)
#include "stileswinthread.h"
#else
#include <pthread.h>
#endif

namespace sTiles {

/// The size of the buffer used for serializing arguments for parallel functions.
constexpr std::size_t ARGS_BUFF_SIZE = 1024;

/**
 * @brief Packs a variable number of arguments into the context's argument buffer.
 *
 * This function serializes the provided arguments into a pre-allocated buffer
 * within the sTiles context, allowing them to be safely passed to worker threads.
 *
 * @tparam Args The types of the arguments to pack.
 * @param[in,out] stile The sTiles context containing the argument buffer.
 * @param[in] args The arguments to pack.
 */
template<typename... Args>
inline void pack_args(stiles_context_t* stile, const Args&... args) {
    unsigned char* ptr = stile->args_buff;
    std::size_t total_size = (sizeof(Args) + ...);

    if (total_size > ARGS_BUFF_SIZE) {

        Logger::fatal("Argument buffer overflow. Required: ", total_size, " bytes, but only ", ARGS_BUFF_SIZE, " are available.");

    }

    auto copy_arg = [&](const auto& arg) {
        std::memcpy(ptr, &arg, sizeof(arg));
        ptr += sizeof(arg);
    };
    (copy_arg(args), ...);
}

/**
 * @brief Unpacks arguments from the context's argument buffer into variables.
 *
 * This function deserializes data from the context's argument buffer back into
 * the provided variables, typically within a parallel function on a worker thread.
 *
 * @tparam Args The types of the arguments to unpack.
 * @param[in] stile The sTiles context containing the argument buffer.
 * @param[out] args The variables to unpack the data into.
 */
template<typename... Args>
inline void unpack_args(const stiles_context_t* stile, Args&... args) {
    const unsigned char* ptr = stile->args_buff;
    auto copy_arg = [&](auto& arg) {
        std::memcpy(&arg, ptr, sizeof(arg));
        ptr += sizeof(arg);
    };
    (copy_arg(args), ...);
}

/**
 * @brief Dispatches a function for parallel execution and waits for completion.
 *
 * This function orchestrates the execution of a given parallel function on all
 * worker threads. It signals the workers, waits for them at a barrier, lets the
 * main thread also execute the function, and finally synchronizes all threads
 * at a second barrier.
 *
 * @param[in,out] stile The sTiles context.
 * @param[in] parallel_function A pointer to the function to be executed by all threads.
 */
inline void static_call(stiles_context_t* stile, void(*parallel_function)(stiles_context_t*)) {

    pthread_mutex_lock(&stile->action_mutex);
    stile->action = sTiles::Action::Parallel;
    stile->parallel_func_ptr = parallel_function;
    pthread_mutex_unlock(&stile->action_mutex);

    // Wake up worker threads
    pthread_cond_broadcast(&stile->action_condt);

    // Sync before execution
    sTiles::Control::Barrier(stile);

    // Main thread also participates
    stile->action = sTiles::Action::StandBy;
    parallel_function(stile);

    // Sync after execution
    sTiles::Control::Barrier(stile);
}

/**
 * @brief Packs arguments and dispatches a function for parallel execution.
 *
 * A convenience wrapper that combines `pack_args` and `static_call` into a
 * single function call for a streamlined parallel dispatch.
 *
 * @tparam Args The types of the arguments to pack.
 * @param[in,out] stile The sTiles context.
 * @param[in] parallel_function A pointer to the function to be executed.
 * @param[in] args The arguments to pack and pass to the function.
 */
template<typename... Args>
inline void parallel_call(stiles_context_t* stile,
                          void(*parallel_function)(stiles_context_t*),
                          const Args&... args) {
    pack_args(stile, args...);
    static_call(stile, parallel_function);
}

} // namespace sTiles

#endif // _STILES_DISPATCH_H_
