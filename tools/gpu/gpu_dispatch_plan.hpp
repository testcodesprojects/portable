/**
 * @file    gpu_dispatch_plan.hpp
 * @brief   Multi-GPU + multi-call dispatch planner for sTiles_preprocess_group.
 *
 * Given N calls each requiring ~B bytes of GPU memory and a node with G GPUs
 * each with M_g free bytes, decide which device runs which call.
 *
 *   case A:  N <= K_per_gpu(best)             → 1 GPU, all N calls
 *   case B:  K_per_gpu < N <= K_per_gpu * G   → spread across multiple GPUs
 *   case C:  N > K_per_gpu * G                → not enough — caller falls back to CPU
 *
 * Used from sTiles_preprocess_group after the symbolic phase fixes the per-call
 * GPU footprint (B). Returns a per-call dispatch vector that downstream
 * GpuTileManager::allocate_for_group(s) consume to bind tiles to devices.
 *
 * Behavior guards:
 *   - Compiled out when STILES_GPU is not defined (header is a no-op).
 *   - STILES_GPU_GRACE_HOPPER selects unified-memory paths inside the planner;
 *     the API stays the same.
 */
#ifndef STILES_GPU_DISPATCH_PLAN_HPP
#define STILES_GPU_DISPATCH_PLAN_HPP

#ifdef STILES_GPU

#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <algorithm>

namespace sTiles { namespace gpu {

// ---------------------------------------------------------------------------
// Grace Hopper runtime toggle.
//
// Compile-time:   STILES_GPU_GRACE_HOPPER  → enables the GH200 code paths.
// Runtime:        STILES_GH200_UNIFIED env var or sTiles::gpu::gh200_unified::set()
//                 → decides at run time whether to actually take the unified
//                   memory path. Lets you build once and A/B compare on the
//                   same binary by flipping the toggle.
//
// Default:        ON when compiled with STILES_GPU_GRACE_HOPPER, otherwise OFF.
// Override:       STILES_GH200_UNIFIED=0 → disable even when compiled in.
//                 STILES_GH200_UNIFIED=1 → force-on (redundant for GH200 build).
// ---------------------------------------------------------------------------
namespace gh200_unified {

inline std::atomic<int>& runtime_state() {
    // -1 = unset (read env on first query)
    //  0 = disabled
    //  1 = enabled
    static std::atomic<int> v{-1};
    return v;
}

inline void set(bool on)  { runtime_state().store(on ? 1 : 0); }
inline void reset_default() { runtime_state().store(-1); }

inline bool enabled() {
#ifndef STILES_GPU_GRACE_HOPPER
    return false;
#else
    int s = runtime_state().load();
    if (s == 0) return false;
    if (s == 1) return true;
    // Unset: read env once and cache.
    const char* env = std::getenv("STILES_GH200_UNIFIED");
    int decided = 1;  // default ON for a GH200 build
    if (env && *env) decided = (std::atoi(env) != 0) ? 1 : 0;
    runtime_state().store(decided);
    return decided != 0;
#endif
}

}  // namespace gh200_unified

struct DeviceInfo {
    int    device_id;
    size_t total_bytes;
    size_t free_bytes;
};

struct DispatchPlan {
    // gpu_id[c] = device id assigned to call c.  -1 means "use CPU for this call".
    std::vector<int>    gpu_id;
    // slot[c] = which concurrent slot (0..K-1) on that GPU. -1 if CPU.
    std::vector<int>    slot;
    // Distinct GPUs actually used (sorted ascending). Empty if CPU fallback.
    std::vector<int>    devices_used;
    // Per-call estimated bytes (carried through for downstream allocator).
    size_t              bytes_per_call = 0;
    // Whether GH200-style unified memory is requested for backing tiles.
    bool                use_unified_memory = false;
    // Diagnostic: which case fired ('A','B','C','X' for CPU fallback).
    char                strategy = 'X';
};

// Enumerate visible CUDA devices. Returns empty vector on failure.
inline std::vector<DeviceInfo> enumerate_devices()
{
    std::vector<DeviceInfo> out;
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) return out;
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
        if (cudaSetDevice(i) != cudaSuccess) continue;
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) continue;
        out.push_back({i, total_b, free_b});
    }
    // Best-GPU-first: most free memory at front so case-A planner picks it.
    std::sort(out.begin(), out.end(),
              [](const DeviceInfo& a, const DeviceInfo& b){
                  return a.free_bytes > b.free_bytes;
              });
    return out;
}

// Compute max concurrent calls that fit on one GPU with `free_bytes` free.
inline int k_per_gpu(size_t free_bytes, size_t bytes_per_call, double safety_margin)
{
    if (bytes_per_call == 0) return 0;
    const double usable = (1.0 - safety_margin) * static_cast<double>(free_bytes);
    const long long k = static_cast<long long>(usable / static_cast<double>(bytes_per_call));
    return (k > 0) ? static_cast<int>(k) : 0;
}

// Plan dispatch for `num_calls` calls, each needing `bytes_per_call`.
//
//   safety_margin in [0,1) — fraction of each GPU's free memory to leave alone.
//   Typical: 0.15.
//
// On success (case A or B): fills `out` with per-call (gpu_id, slot) and
// returns true. On case C / no GPUs: returns false and `out.strategy='X'`.
inline bool plan_dispatch(int num_calls,
                          size_t bytes_per_call,
                          double safety_margin,
                          DispatchPlan& out)
{
    out.gpu_id.assign(num_calls, -1);
    out.slot.assign  (num_calls, -1);
    out.devices_used.clear();
    out.bytes_per_call = bytes_per_call;
    out.strategy = 'X';
    // Runtime-gated: only ON when compiled with STILES_GPU_GRACE_HOPPER AND
    // the runtime toggle says yes. Lets the same binary A/B both paths.
    out.use_unified_memory = gh200_unified::enabled();

    if (num_calls <= 0 || bytes_per_call == 0) return false;

    std::vector<DeviceInfo> devs = enumerate_devices();
    if (devs.empty()) return false;

    // Try case A: all calls on the GPU with most free memory.
    const int K_best = k_per_gpu(devs[0].free_bytes, bytes_per_call, safety_margin);
    if (num_calls <= K_best) {
        out.devices_used.push_back(devs[0].device_id);
        for (int c = 0; c < num_calls; ++c) {
            out.gpu_id[c] = devs[0].device_id;
            out.slot[c]   = c;
        }
        out.strategy = 'A';
        return true;
    }

    // Try case B: spread across multiple GPUs (best-first), using each GPU's
    // own K_per_gpu. Pack greedily until all calls are placed.
    int placed = 0;
    std::vector<int> slot_cursor(devs.size(), 0);
    for (size_t d = 0; d < devs.size() && placed < num_calls; ++d) {
        const int K_d = k_per_gpu(devs[d].free_bytes, bytes_per_call, safety_margin);
        if (K_d <= 0) continue;
        const int can_take = std::min(K_d, num_calls - placed);
        for (int i = 0; i < can_take; ++i) {
            out.gpu_id[placed + i] = devs[d].device_id;
            out.slot[placed + i]   = slot_cursor[d]++;
        }
        if (can_take > 0) {
            out.devices_used.push_back(devs[d].device_id);
            placed += can_take;
        }
    }
    if (placed == num_calls) {
        out.strategy = (out.devices_used.size() == 1) ? 'A' : 'B';
        return true;
    }

    // Case C: even all GPUs combined can't fit. Wipe and signal CPU fallback.
    out.gpu_id.assign(num_calls, -1);
    out.slot.assign  (num_calls, -1);
    out.devices_used.clear();
    out.strategy = 'C';
    return false;
}

// Pretty-print the plan to stderr (or a logger) for diagnostics.
inline void print_plan(const DispatchPlan& plan, int num_calls)
{
    std::fprintf(stderr, "[gpu_dispatch] strategy=%c  bytes/call=%.2f GiB  unified_mem=%d\n",
                 plan.strategy,
                 static_cast<double>(plan.bytes_per_call) / (1024.0 * 1024.0 * 1024.0),
                 plan.use_unified_memory ? 1 : 0);
    if (plan.strategy == 'X' || plan.strategy == 'C') {
        std::fprintf(stderr, "[gpu_dispatch]   no fit -> CPU fallback\n");
        return;
    }
    std::fprintf(stderr, "[gpu_dispatch]   devices used: ");
    for (int d : plan.devices_used) std::fprintf(stderr, "%d ", d);
    std::fprintf(stderr, "\n[gpu_dispatch]   per-call assignment (showing first 8 of %d):\n", num_calls);
    const int show = std::min(num_calls, 8);
    for (int c = 0; c < show; ++c) {
        std::fprintf(stderr, "[gpu_dispatch]     call %d -> gpu %d slot %d\n",
                     c, plan.gpu_id[c], plan.slot[c]);
    }
    if (num_calls > show) std::fprintf(stderr, "[gpu_dispatch]     ... (%d more)\n", num_calls - show);
}

}}  // namespace sTiles::gpu

#endif  // STILES_GPU
#endif  // STILES_GPU_DISPATCH_PLAN_HPP
