#include "new_context.hpp"
#include <stdexcept>
#include <string>
#include <iostream>

namespace sTiles {

std::vector<Context::ContextMapEntry> Context::context_map;
std::vector<pthread_mutex_t> Context::context_map_locks;
std::vector<int> Context::next_available_slot;
thread_local std::vector<Context*> Context::thread_context_pointers;

void Context::initialize(int num_indices) {

    if (!context_map.empty()) {
        throw std::runtime_error("Context::initialize: Global context already initialized.");
    }
    if (num_indices <= 0) {
        throw std::runtime_error("Context::initialize: Number of indices must be positive.");
    }

    try {

        context_map.resize(static_cast<size_t>(num_indices) * CONTEXTS_PER_GROUP_MAX);
        context_map_locks.resize(num_indices);
        next_available_slot.assign(num_indices, 0); // Initialize all hints to 0
    } catch (const std::bad_alloc& e) {
        finalize(); // Clean up partial allocations
        throw std::runtime_error("Failed to allocate memory for context map: " + std::string(e.what()));
    }
    
    for (int i = 0; i < num_indices; i++) {
        if (pthread_mutex_init(&context_map_locks[i], nullptr) != 0) {

            for (int j = 0; j < i; j++) {
                pthread_mutex_destroy(&context_map_locks[j]);
            }
            finalize(); // Clear vectors
            throw std::runtime_error("Failed to initialize a context mutex.");
        }
    }
}

void Context::finalize() {

    for (auto& m : context_map_locks) {
        pthread_mutex_destroy(&m);
    }
    
    context_map.clear();
    context_map.shrink_to_fit();
    context_map_locks.clear();
    context_map_locks.shrink_to_fit();
    next_available_slot.clear();
    next_available_slot.shrink_to_fit();
}

Context* Context::getCurrent(int call_index) {

    if (static_cast<size_t>(call_index) >= thread_context_pointers.size()) {
        return nullptr; 
    }
    return thread_context_pointers[call_index];
}

Context::Context(int call_index) : m_call_index(call_index), m_map_slot_index(-1) { // Initialize index to -1
    
    if (context_map.empty()) {
        throw std::runtime_error("Context::Context(): Global context not initialized. Call Context::initialize() first.");
    }
    if (call_index < 0 || static_cast<size_t>(call_index) >= context_map_locks.size()) {
        throw std::runtime_error("Context::Context(): Invalid call_index.");
    }

    data = std::make_unique<ContextData>();
    pthread_mutex_init(&data->action_mutex, nullptr);
    pthread_cond_init(&data->action_condt, nullptr);
    pthread_t self_id = pthread_self();
    size_t start_offset = static_cast<size_t>(call_index) * CONTEXTS_PER_GROUP_MAX;
    
    pthread_mutex_lock(&context_map_locks[call_index]);

    int found_slot = -1;
    int start_search_idx = next_available_slot[call_index];

    for (int i = 0; i < CONTEXTS_PER_GROUP_MAX; ++i) {
        int current_idx = (start_search_idx + i) % CONTEXTS_PER_GROUP_MAX;
        if (context_map[start_offset + current_idx].context == nullptr) {

            auto& entry = context_map[start_offset + current_idx];
            entry.context = this;
            entry.thread_id = self_id;
            const_cast<int&>(m_map_slot_index) = current_idx;
            found_slot = current_idx;
            
            // Update the hint for the next thread
            next_available_slot[call_index] = (current_idx + 1) % CONTEXTS_PER_GROUP_MAX;
            break;
        }
    }

    pthread_mutex_unlock(&context_map_locks[call_index]);

    if (found_slot == -1) {
        pthread_mutex_destroy(&data->action_mutex);
        pthread_cond_destroy(&data->action_condt);
        throw std::runtime_error("Context::Context(): No available slot in context map.");
    }

    if (static_cast<size_t>(call_index) >= thread_context_pointers.size()) {
        thread_context_pointers.resize(call_index + 1, nullptr);
    }
    thread_context_pointers[call_index] = this;
}

Context::~Context() {

    if (data) {
        pthread_mutex_destroy(&data->action_mutex);
        pthread_cond_destroy(&data->action_condt);
    }
    
    if (context_map.empty()) {
        return; 
    }

    if (static_cast<size_t>(m_call_index) < thread_context_pointers.size()) {
        thread_context_pointers[m_call_index] = nullptr;
    }

    size_t start_offset = static_cast<size_t>(m_call_index) * CONTEXTS_PER_GROUP_MAX;
    size_t absolute_index = start_offset + m_map_slot_index;

    pthread_mutex_lock(&context_map_locks[m_call_index]);
    
    if (context_map[absolute_index].context == this) {
        context_map[absolute_index].context = nullptr;
        context_map[absolute_index].thread_id = {}; // Clear thread id
    } else {

        stiles_fatal_error("Context::~Context()", "Context corruption detected. Mismatch in map.");
    }
    
    pthread_mutex_unlock(&context_map_locks[m_call_index]);
}

Int Context::getRank() const {
    pthread_t self_id = pthread_self();

    for (Int rank = 0; rank < data->world_size; ++rank) {
        if (pthread_equal(data->thread_id[rank], self_id)) {
            return rank;
        }
    }
    // Return a value from the StatusCode enum, cast to Int.
    // This is a common pattern for functions returning an int that can also be an error code.
    return static_cast<Int>(StatusCode::NotFound);
}

StatusCode Context::tune(Function func, Size M, Size N, Size NRHS, Int tile_size) {
    if (!data->autotuning_enabled) {
        return StatusCode::Success;
    }

    switch (func) {
        case Function::DPOSV:
            data->nb = 120; // Default value
            if (tile_size > 0) {
                data->nb = tile_size;
            }
            break;

        default:
            error("tune", "Unsupported function type for tuning.");
            return StatusCode::NotSupported; // Use the specific error code
    }

    // Use the Size type for potentially large values
    data->nbnbsize = static_cast<Size>(data->nb) * data->nb;
    return StatusCode::Success;
}

void Context::warning(const char* function_name, const char* message) const {
    if (data->warnings_enabled) {
        sTiles::Logger::warning(function_name, "(): ", message);
    }
}

void Context::error(const char* function_name, const char* message) const {
    if (data->errors_enabled) {
        sTiles::Logger::error(function_name, "(): ", message);
    }
}

} // namespace sTiles