#pragma once
// SPDX-License-Identifier: MIT
// Part of etree — https://github.com/NickAlger/ellipsoid_tree

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace etree {
namespace detail {

inline int hardware_thread_count()
{
    unsigned int n = std::thread::hardware_concurrency();
    return ( n == 0u ) ? 1 : static_cast<int>(n);
}

// Run f(chunk_begin, chunk_end) over subranges covering [begin, end).
//
// Chunks are claimed by worker threads through a shared atomic counter
// (dynamic scheduling), so uneven per-index work balances itself without any
// upfront shuffling, and each index is processed exactly once. The calling
// thread participates as a worker.
//
//   num_threads <= 0 : use hardware_thread_count()
//   num_threads == 1 : run f(begin, end) directly on the calling thread
//   grain_size  <= 0 : pick a chunk size giving each thread ~8 chunks
//
// Defining ETREE_NO_THREADS forces the serial path. If f throws, the first
// exception is rethrown on the calling thread after all workers finish;
// remaining chunks are abandoned.
template <class F>
void parallel_for( std::ptrdiff_t begin,
                   std::ptrdiff_t end,
                   F&&            f,
                   int            num_threads = 0,
                   std::ptrdiff_t grain_size  = 0 )
{
    const std::ptrdiff_t n = end - begin;
    if ( n <= 0 )
    {
        return;
    }

    int nt = ( num_threads > 0 ) ? num_threads : hardware_thread_count();
#ifdef ETREE_NO_THREADS
    nt = 1;
#endif
    if ( static_cast<std::ptrdiff_t>(nt) > n )
    {
        nt = static_cast<int>(n);
    }

    if ( nt <= 1 )
    {
        f(begin, end);
        return;
    }

    const std::ptrdiff_t chunk =
        ( grain_size > 0 ) ? grain_size
                           : std::max<std::ptrdiff_t>(1, n / (8 * nt));

    std::atomic<std::ptrdiff_t> next(begin);
    std::exception_ptr          first_exception;
    std::mutex                  exception_mutex;

    auto work = [&]()
    {
        try
        {
            while ( true )
            {
                const std::ptrdiff_t chunk_begin = next.fetch_add(chunk);
                if ( chunk_begin >= end )
                {
                    break;
                }
                f(chunk_begin, std::min(chunk_begin + chunk, end));
            }
        }
        catch (...)
        {
            {
                std::lock_guard<std::mutex> lock(exception_mutex);
                if ( !first_exception )
                {
                    first_exception = std::current_exception();
                }
            }
            next.store(end); // abandon remaining chunks
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(nt - 1);
    for ( int tt = 0; tt < nt - 1; ++tt )
    {
        workers.emplace_back(work);
    }
    work();
    for ( std::thread& worker : workers )
    {
        worker.join();
    }

    if ( first_exception )
    {
        std::rethrow_exception(first_exception);
    }
}

template <class F>
void parallel_for( std::ptrdiff_t n,
                   F&&            f,
                   int            num_threads = 0,
                   std::ptrdiff_t grain_size  = 0 )
{
    parallel_for(0, n, f, num_threads, grain_size);
}

} // end namespace detail
} // end namespace etree
