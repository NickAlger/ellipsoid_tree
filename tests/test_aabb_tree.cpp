// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "etree/aabb_tree.hpp"
#include "etree/geometric_sort.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <random>
#include <set>

using namespace etree;
namespace th = test_helpers;

namespace {

// Random leaf boxes: centers ~ N(0, 2), half-widths in (0.05, ~0.6)
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> random_boxes( int d, int n, std::mt19937& gen )
{
    Eigen::MatrixXd lo(d, n);
    Eigen::MatrixXd hi(d, n);
    for ( int ii = 0; ii < n; ++ii )
    {
        Eigen::VectorXd c = th::randn_vector(d, gen, 2.0);
        Eigen::VectorXd w = th::randn_vector(d, gen, 0.25).cwiseAbs()
                            + Eigen::VectorXd::Constant(d, 0.05);
        lo.col(ii) = c - w;
        hi.col(ii) = c + w;
    }
    return {lo, hi};
}

std::vector<int> sorted( std::vector<int> v )
{
    std::sort(v.begin(), v.end());
    return v;
}

} // end anonymous namespace

TEST_CASE("AABBTree structural invariants")
{
    std::mt19937 gen(2001);
    for ( int n : {1, 2, 3, 7, 100, 1000} )
    {
        const int d = 1 + (n % 4);
        auto [lo, hi] = random_boxes(d, n, gen);
        AABBTree T(lo, hi);

        CHECK(T.num_leaves() == n);
        CHECK(T.num_nodes() == 2 * n - 1);

        std::set<int> seen_externals;
        for ( int b = 0; b < T.num_nodes(); ++b )
        {
            if ( T.is_leaf(b) )
            {
                const int e = T.external_index(b);
                CHECK(e >= 0);
                CHECK(e < n);
                seen_externals.insert(e);
                CHECK((T.node_lo(b) - lo.col(e)).cwiseAbs().maxCoeff() == 0.0);
                CHECK((T.node_hi(b) - hi.col(e)).cwiseAbs().maxCoeff() == 0.0);
            }
            else
            {
                CHECK(T.external_index(b) == -1);
                CHECK(T.right_child(b) < T.num_nodes()); // complete tree: both children exist
                for ( int child : {T.left_child(b), T.right_child(b)} )
                {
                    CHECK((T.node_lo(b).array() <= T.node_lo(child).array()).all());
                    CHECK((T.node_hi(child).array() <= T.node_hi(b).array()).all());
                }
            }
        }
        CHECK(static_cast<int>(seen_externals.size()) == n);
    }

    AABBTree empty;
    CHECK(empty.empty());
    CHECK(empty.point_collisions(Eigen::VectorXd::Zero(2)).empty());
}

TEST_CASE("AABBTree point/box/ball collisions against brute force")
{
    std::mt19937 gen(2002);
    for ( int rep = 0; rep < 6; ++rep )
    {
        const int d = 1 + (rep % 4);
        const int n = 300;
        auto [lo, hi] = random_boxes(d, n, gen);
        AABBTree T(lo, hi);

        for ( int qq = 0; qq < 30; ++qq )
        {
            Eigen::VectorXd p = th::randn_vector(d, gen, 2.0);
            std::vector<int> brute_p;
            for ( int ii = 0; ii < n; ++ii )
            {
                if ( (lo.col(ii).array() <= p.array()).all()
                     && (p.array() <= hi.col(ii).array()).all() )
                {
                    brute_p.push_back(ii);
                }
            }
            CHECK(sorted(T.point_collisions(p)) == brute_p);

            Eigen::VectorXd qc = th::randn_vector(d, gen, 2.0);
            Eigen::VectorXd qw = th::randn_vector(d, gen, 0.4).cwiseAbs()
                                 + Eigen::VectorXd::Constant(d, 0.05);
            Eigen::VectorXd qlo = qc - qw;
            Eigen::VectorXd qhi = qc + qw;
            std::vector<int> brute_b;
            for ( int ii = 0; ii < n; ++ii )
            {
                if ( (qlo.array() <= hi.col(ii).array()).all()
                     && (lo.col(ii).array() <= qhi.array()).all() )
                {
                    brute_b.push_back(ii);
                }
            }
            CHECK(sorted(T.box_collisions(qlo, qhi)) == brute_b);

            const double r = 0.2 + 0.5 * qq / 30.0;
            std::vector<int> brute_ball;
            for ( int ii = 0; ii < n; ++ii )
            {
                Eigen::VectorXd closest = qc.cwiseMax(lo.col(ii)).cwiseMin(hi.col(ii));
                if ( (closest - qc).squaredNorm() <= r * r )
                {
                    brute_ball.push_back(ii);
                }
            }
            CHECK(sorted(T.ball_collisions(qc, r)) == brute_ball);
        }
    }
}

TEST_CASE("AABBTree visit early exit stops the traversal")
{
    std::mt19937 gen(2003);
    auto [lo, hi] = random_boxes(2, 200, gen);
    AABBTree T(lo, hi);

    // A query point covered by many boxes: stop after the first hit
    Eigen::VectorXd p = 0.5 * (lo.col(0) + hi.col(0));
    REQUIRE(T.point_collisions(p).size() >= 1);

    int calls = 0;
    const bool completed = T.visit(
        [&]( const auto& blo, const auto& bhi )
        { return (blo.array() <= p.array()).all() && (p.array() <= bhi.array()).all(); },
        [&]( int ) { calls += 1; return false; });
    CHECK(!completed);
    CHECK(calls == 1);
}

TEST_CASE("visit_pairs and visit_self_pairs against brute force")
{
    std::mt19937 gen(2004);
    for ( int rep = 0; rep < 6; ++rep )
    {
        const int d = 2 + (rep % 2);
        const int n = 120;
        const int m = 90;
        auto [loA, hiA] = random_boxes(d, n, gen);
        auto [loB, hiB] = random_boxes(d, m, gen);
        AABBTree A(loA, hiA);
        AABBTree B(loB, hiB);

        auto boxes_overlap = [&]( const Eigen::VectorXd& l1, const Eigen::VectorXd& h1,
                                  const Eigen::VectorXd& l2, const Eigen::VectorXd& h2 )
        { return (l1.array() <= h2.array()).all() && (l2.array() <= h1.array()).all(); };

        // Cross pairs
        std::set<std::pair<int, int>> brute;
        for ( int ii = 0; ii < n; ++ii )
        {
            for ( int jj = 0; jj < m; ++jj )
            {
                if ( boxes_overlap(loA.col(ii), hiA.col(ii), loB.col(jj), hiB.col(jj)) )
                {
                    brute.insert({ii, jj});
                }
            }
        }
        std::set<std::pair<int, int>> dual;
        visit_pairs(A, B,
                    [&]( const auto& l1, const auto& h1, const auto& l2, const auto& h2 )
                    { return (l1.array() <= h2.array()).all() && (l2.array() <= h1.array()).all(); },
                    [&]( int ia, int ib )
                    {
                        const bool inserted = dual.insert({ia, ib}).second;
                        CHECK(inserted); // no duplicate leaf pairs
                        return true;
                    });
        CHECK(dual == brute);

        // Self pairs
        std::set<std::pair<int, int>> brute_self;
        for ( int ii = 0; ii < n; ++ii )
        {
            for ( int jj = ii + 1; jj < n; ++jj )
            {
                if ( boxes_overlap(loA.col(ii), hiA.col(ii), loA.col(jj), hiA.col(jj)) )
                {
                    brute_self.insert({ii, jj});
                }
            }
        }
        std::set<std::pair<int, int>> dual_self;
        visit_self_pairs(A,
                         [&]( const auto& l1, const auto& h1, const auto& l2, const auto& h2 )
                         { return (l1.array() <= h2.array()).all() && (l2.array() <= h1.array()).all(); },
                         [&]( int ia, int ib )
                         {
                             const bool inserted =
                                 dual_self.insert({std::min(ia, ib), std::max(ia, ib)}).second;
                             CHECK(inserted); // each unordered pair at most once
                             CHECK(ia != ib);
                             return true;
                         });
        CHECK(dual_self == brute_self);
    }

    // Empty trees are fine
    AABBTree empty;
    std::mt19937 gen2(1);
    auto [lo1, hi1] = random_boxes(2, 5, gen2);
    AABBTree small(lo1, hi1);
    int calls = 0;
    visit_pairs(empty, small, [](const auto&, const auto&, const auto&, const auto&) { return true; },
                [&](int, int) { calls += 1; return true; });
    visit_self_pairs(empty, [](const auto&, const auto&, const auto&, const auto&) { return true; },
                     [&](int, int) { calls += 1; return true; });
    CHECK(calls == 0);
}

TEST_CASE("geometric_sort: permutation, determinism, block structure")
{
    std::mt19937 gen(2005);

    // Permutation and determinism
    Eigen::MatrixXd P = th::randn_matrix(3, 257, gen);
    std::vector<int> s1 = geometric_sort(P);
    std::vector<int> s2 = geometric_sort(P);
    CHECK(s1 == s2);
    std::vector<int> check = s1;
    std::sort(check.begin(), check.end());
    for ( int ii = 0; ii < 257; ++ii )
    {
        CHECK(check[ii] == ii);
    }

    // 1D: descending order (axis 0, descending comparator at every level)
    Eigen::MatrixXd P1 = th::randn_matrix(1, 100, gen);
    std::vector<int> s = geometric_sort(P1);
    for ( int ii = 0; ii + 1 < 100; ++ii )
    {
        CHECK(P1(0, s[ii]) >= P1(0, s[ii + 1]));
    }

    // 2D: after the top-level split, every first-half x is >= every second-half x
    Eigen::MatrixXd P2 = th::randn_matrix(2, 64, gen);
    std::vector<int> s3 = geometric_sort(P2);
    double min_first = std::numeric_limits<double>::infinity();
    double max_second = -std::numeric_limits<double>::infinity();
    for ( int ii = 0; ii < 32; ++ii )
    {
        min_first = std::min(min_first, P2(0, s3[ii]));
        max_second = std::max(max_second, P2(0, s3[32 + ii]));
    }
    CHECK(min_first >= max_second);
}
