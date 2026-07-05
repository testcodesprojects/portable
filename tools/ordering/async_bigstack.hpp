#pragma once
//============================================================================
// async_bigstack: like std::async(std::launch::async, f), used for the ordering
// bake-off (RCM / METIS-ND / SCOTCH candidates run "at once").
//
//   Linux : real std::async — the candidates run concurrently.
//   macOS : the task runs on a 256 MB pthread stack AND is JOINED immediately,
//           i.e. the bake-off is SERIALIZED (one candidate at a time).
//
// Why macOS is special: METIS uses GKlib, whose memory core is thread-local via
// __thread — but on macOS that does NOT survive the concurrent bake-off. Two
// candidates touching GKlib at once corrupt it ("Unknown mop type" /
// "gkmcoreDel should never have been here" -> SIGABRT/SIGSEGV on the spacetime
// matrix). Joining each task before starting the next removes the race. The
// 256 MB stack additionally covers METIS's deep nested-dissection recursion
// (macOS threads otherwise get only 512 KB). This serializes ONLY the ordering
// bake-off (a small symbolic step); the numeric factorization / selinv / solve
// parallelism is untouched.
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
    // JOINABLE (the default): we join below, which serializes the bake-off.

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

    if (rc == 0)
        pthread_join(th, nullptr);   // wait now -> no two candidates in GKlib at once
    else {                           // thread creation failed: run inline so fut still resolves
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
