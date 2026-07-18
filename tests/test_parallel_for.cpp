// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "etree/detail/parallel_for.hpp"

#include <atomic>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

using etree::detail::parallel_for;

TEST_CASE("parallel_for visits every index exactly once")
{
    const std::ptrdiff_t n = 10007;
    for ( int num_threads : {0, 1, 2, 7, 64} )
    {
        for ( std::ptrdiff_t grain : {std::ptrdiff_t(0), std::ptrdiff_t(1), std::ptrdiff_t(13)} )
        {
            // Each index lies in exactly one chunk, so unsynchronized writes
            // to distinct entries are race-free.
            std::vector<int> visit_counts(n, 0);
            parallel_for(n, [&](std::ptrdiff_t aa, std::ptrdiff_t bb)
            {
                for ( std::ptrdiff_t ii = aa; ii < bb; ++ii )
                {
                    visit_counts[ii] += 1;
                }
            }, num_threads, grain);

            bool all_visited_once = true;
            for ( int count : visit_counts )
            {
                all_visited_once = all_visited_once && (count == 1);
            }
            CHECK(all_visited_once);
        }
    }
}

TEST_CASE("parallel_for computes correct sum via per-chunk accumulation")
{
    const std::ptrdiff_t n = 123457;
    std::atomic<long long> total(0);
    parallel_for(n, [&](std::ptrdiff_t aa, std::ptrdiff_t bb)
    {
        long long chunk_sum = 0;
        for ( std::ptrdiff_t ii = aa; ii < bb; ++ii )
        {
            chunk_sum += ii;
        }
        total.fetch_add(chunk_sum);
    }, 4);
    CHECK(total.load() == static_cast<long long>(n) * (n - 1) / 2);
}

TEST_CASE("parallel_for handles empty and tiny ranges")
{
    std::atomic<int> calls(0);
    parallel_for(0, [&](std::ptrdiff_t, std::ptrdiff_t) { calls.fetch_add(1); }, 4);
    CHECK(calls.load() == 0);

    std::vector<int> hits(1, 0);
    parallel_for(1, [&](std::ptrdiff_t aa, std::ptrdiff_t bb)
    {
        for ( std::ptrdiff_t ii = aa; ii < bb; ++ii ) { hits[ii] += 1; }
    }, 4);
    CHECK(hits[0] == 1);

    // More threads than items
    std::vector<int> hits3(3, 0);
    parallel_for(3, [&](std::ptrdiff_t aa, std::ptrdiff_t bb)
    {
        for ( std::ptrdiff_t ii = aa; ii < bb; ++ii ) { hits3[ii] += 1; }
    }, 16);
    CHECK(hits3 == std::vector<int>({1, 1, 1}));
}

TEST_CASE("parallel_for respects offset ranges")
{
    std::vector<int> visited(20, 0);
    parallel_for(5, 12, [&](std::ptrdiff_t aa, std::ptrdiff_t bb)
    {
        for ( std::ptrdiff_t ii = aa; ii < bb; ++ii ) { visited[ii] += 1; }
    }, 3, 2);

    for ( int ii = 0; ii < 20; ++ii )
    {
        CHECK(visited[ii] == ((5 <= ii && ii < 12) ? 1 : 0));
    }
}

TEST_CASE("parallel_for propagates exceptions from workers")
{
    auto throwing_loop = [](std::ptrdiff_t aa, std::ptrdiff_t bb)
    {
        for ( std::ptrdiff_t ii = aa; ii < bb; ++ii )
        {
            if ( ii == 501 )
            {
                throw std::runtime_error("boom");
            }
        }
    };
    CHECK_THROWS_AS(parallel_for(0, 1000, throwing_loop, 4), std::runtime_error);
    CHECK_THROWS_AS(parallel_for(0, 1000, throwing_loop, 1), std::runtime_error);
}
