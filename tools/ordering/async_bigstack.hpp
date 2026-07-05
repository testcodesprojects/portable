#pragma once
//============================================================================
// async_bigstack: like std::async(std::launch::async, f), but on macOS the
// worker thread gets a LARGE stack.
//
// Why: macOS gives std::thread / std::async worker threads only a 512 KB
// stack, whereas on Linux they inherit the main thread's ~8 MB. METIS's
// nested-dissection ordering recurses deeply on large graphs and overflows the
// 512 KB stack, corrupting GKlib's thread-local memory core (observed as
// "Unknown mop type" / SIGABRT on the spacetime matrix). Running the ordering
// candidates on a pthread with a 256 MB stack removes the overflow while
// keeping the concurrent bake-off. On Linux the default std::async is used.
//============================================================================
#include <future>
#include <memory>
#include <utility>

#if defined(__APPLE__)
#include <pthread.h>

namespace sTiles {

template <class F>
auto async_bigstack(F&& f) -> std::future<decltype(f())> {
    using R = decltype(f());
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    std::future<R> fut = task->get_future();

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, static_cast<size_t>(256) << 20);   // 256 MB
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);        // future syncs completion

    // Heap-owned shared_ptr copy handed to the thread; `task` is retained here
    // so the synchronous fallback below can still run if thread creation fails.
    auto* heap = new std::shared_ptr<std::packaged_task<R()>>(task);
    pthread_t th;
    const int rc = pthread_create(&th, &attr,
        [](void* p) -> void* {
            std::unique_ptr<std::shared_ptr<std::packaged_task<R()>>> owner(
                static_cast<std::shared_ptr<std::packaged_task<R()>>*>(p));
            (**owner)();
            return nullptr;
        }, heap);
    pthread_attr_destroy(&attr);

    if (rc != 0) {              // thread creation failed: run inline so fut still resolves
        delete heap;
        (*task)();
    }
    return fut;
}

}  // namespace sTiles

#else  // ---- non-macOS: plain std::async (inherits the big main-thread stack) ----

namespace sTiles {

template <class F>
auto async_bigstack(F&& f) -> std::future<decltype(f())> {
    return std::async(std::launch::async, std::forward<F>(f));
}

}  // namespace sTiles

#endif
