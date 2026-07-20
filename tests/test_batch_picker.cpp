// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "ellipsoid_tree/batch_picker.hpp"
#include "test_helpers.hpp"

#include <random>
#include <set>

using namespace ellipsoid_tree;
namespace th = test_helpers;

namespace {

std::vector<Ellipsoid> overlapping_family( int d, int n, std::mt19937& gen )
{
    // Dense enough that many ellipsoids overlap, so batching is nontrivial
    std::vector<Ellipsoid> out(n);
    for ( int ii = 0; ii < n; ++ii )
    {
        out[ii] = Ellipsoid{th::randn_vector(d, gen, 1.0), 0.02 * th::random_spd(d, gen, 0.3, 3.0)};
    }
    return out;
}

} // end anonymous namespace

TEST_CASE("batches partition the ellipsoids and are internally non-overlapping")
{
    std::mt19937 gen(5001);
    for ( int rep = 0; rep < 3; ++rep )
    {
        const int d = 2 + (rep % 2);
        const int n = 200;
        const double tau = 1.0;
        EllipsoidTree T(overlapping_family(d, n, gen), tau);

        std::vector<std::vector<int>> batches = pick_ellipsoid_batches(T);

        // Partition: every index exactly once
        std::set<int> seen;
        int total = 0;
        for ( const std::vector<int>& batch : batches )
        {
            CHECK(!batch.empty());
            for ( int idx : batch )
            {
                seen.insert(idx);
                total += 1;
            }
        }
        CHECK(total == n);
        CHECK(static_cast<int>(seen.size()) == n);

        // Independence: no two ellipsoids in a batch intersect
        for ( const std::vector<int>& batch : batches )
        {
            for ( size_t aa = 0; aa < batch.size(); ++aa )
            {
                for ( size_t bb = aa + 1; bb < batch.size(); ++bb )
                {
                    CHECK(!intersects(T.object(batch[aa]), T.object(batch[bb]), tau));
                }
            }
        }

        // Nontrivial problem: more than one batch, fewer batches than ellipsoids
        CHECK(batches.size() > 1);
        CHECK(batches.size() < static_cast<size_t>(n));
    }
}

TEST_CASE("batch picking is deterministic and respects max_batches")
{
    std::mt19937 gen(5002);
    const int n = 150;
    EllipsoidTree T(overlapping_family(2, n, gen), 1.0);

    std::vector<std::vector<int>> b1 = pick_ellipsoid_batches(T);
    std::vector<std::vector<int>> b2 = pick_ellipsoid_batches(T);
    CHECK(b1 == b2);

    // Parallel distance updates do not change the result
    std::vector<std::vector<int>> b4 = pick_ellipsoid_batches(T, -1, 4);
    CHECK(b4 == b1);

    // Explicit center anchors match the convenience overload
    Eigen::MatrixXd anchors(2, n);
    for ( int ii = 0; ii < n; ++ii )
    {
        anchors.col(ii) = T.object(ii).mu;
    }
    CHECK(pick_ellipsoid_batches(T, anchors) == b1);

    // max_batches truncates
    REQUIRE(b1.size() >= 3);
    std::vector<std::vector<int>> limited = pick_ellipsoid_batches(T, 2);
    CHECK(limited.size() == 2);
    CHECK(limited[0] == b1[0]);
    CHECK(limited[1] == b1[1]);

    // Anchor count mismatch throws
    Eigen::MatrixXd bad_anchors(2, n - 1);
    CHECK_THROWS_AS(pick_ellipsoid_batches(T, bad_anchors), std::invalid_argument);
}

TEST_CASE("disjoint ellipsoids give a single batch")
{
    // Far-apart tiny ellipsoids: everything fits in one batch
    std::vector<Ellipsoid> objs;
    for ( int ii = 0; ii < 20; ++ii )
    {
        Eigen::Vector2d mu(10.0 * ii, 0.0);
        objs.push_back(Ellipsoid{mu, 0.01 * Eigen::Matrix2d::Identity()});
    }
    EllipsoidTree T(objs, 1.0);
    std::vector<std::vector<int>> batches = pick_ellipsoid_batches(T);
    CHECK(batches.size() == 1);
    CHECK(batches[0].size() == 20);
}
