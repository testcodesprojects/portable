#ifndef STILES_CONTEXT_HPP
#define STILES_CONTEXT_HPP

#if defined(_WIN32) || defined(_WIN64)
#include "stileswinthread.h"
#else
#include <pthread.h>
#endif

#include "common.h" 
#include <memory>
#include <vector> 
#include "../include/stiles_types.hpp"

namespace sTiles {


class Context {

    public:

        static void initialize(int num_indices);
        static void finalize();
        static Context* getCurrent(int call_index);
        explicit Context(int call_index);
        ~Context();
        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;
        Context(Context&&) = delete;
        Context& operator=(Context&&) = delete;

        Int getRank() const;
        StatusCode tune(Function func, Size M, Size N, Size NRHS, Int tile_size);
        void warning(const char* function_name, const char* message) const;
        void error(const char* function_name, const char* message) const;

    private:
      
        struct ContextData;
        struct ContextMapEntry;
        
        struct ContextData {

            Bool initialized;

            int world_size, group_size;
            int thread_bind[CONTEXT_THREADS_MAX];
            int thread_rank[CONTEXT_THREADS_MAX];
            pthread_attr_t thread_attr;
            pthread_t thread_id[CONTEXT_THREADS_MAX];

            pthread_mutex_t action_mutex;
            pthread_cond_t action_condt;
            volatile int action;
            void (*parallel_func_ptr)(ContextData*); // Pointer now takes ContextData
            unsigned char args_buff[ARGS_BUFF_SIZE];

            Bool errors_enabled;
            Bool warnings_enabled;
            Bool autotuning_enabled;
            Bool dynamic_section;

            int scheduling;
            int nb;
            int nbnbsize;
            int rhblock;
            int tntsize;

            volatile int barrier_in[CONTEXT_THREADS_MAX];
            volatile int barrier_out[CONTEXT_THREADS_MAX];
            int volatile    barrier_id;
            int volatile    barrier_nblocked_thrds;
            pthread_mutex_t barrier_synclock;
            pthread_cond_t  barrier_synccond;

            int ss_ld;
            volatile int ss_abort;
            volatile int *ss_progress;
            // Sparse byte-progress (semi-mode chol paths only; values are 0/1).
            // Allocated for numActiveTiles slots; indexed by task indexN, not (i,j).
            volatile unsigned char *ss_slots;
            int ss_nslots;
        };

        struct ContextMapEntry {
            pthread_t thread_id{}; // Use {} for default initialization
            Context*  context = nullptr;
        };


        // --- Member variables ---
        std::unique_ptr<ContextData> data;
        const int m_call_index; // Make const as it shouldn't change
        const int m_map_slot_index; // Store our slot in the map to avoid searching on destruction

        // --- Static variables for global map ---
        static std::vector<ContextMapEntry> context_map;
        static std::vector<pthread_mutex_t> context_map_locks;
        static std::vector<int> next_available_slot; // Hint for faster insertion

        // --- The key optimization for getCurrent ---
        // A vector of context pointers, one for each call_index, unique to each thread.
        static thread_local std::vector<Context*> thread_context_pointers;
    };

} 

#endif // STILES_CONTEXT_HPP



