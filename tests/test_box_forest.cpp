// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "ellipsoid_tree/box_forest.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <random>
#include <vector>

using namespace ellipsoid_tree;
namespace th = test_helpers;

namespace {

// Random leaf boxes around given cluster centers.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> clustered_boxes(
    int d, int n_per_cluster, const std::vector<Eigen::VectorXd>& centers,
    std::mt19937& gen )
{
    const int n = n_per_cluster * static_cast<int>(centers.size());
    Eigen::MatrixXd lo(d, n), hi(d, n);
    int col = 0;
    for ( const auto& cc : centers )
    {
        for ( int ii = 0; ii < n_per_cluster; ++ii, ++col )
        {
            Eigen::VectorXd c = cc + th::randn_vector(d, gen, 0.5);
            Eigen::VectorXd w = th::randn_vector(d, gen, 0.1).cwiseAbs()
                                + Eigen::VectorXd::Constant(d, 0.02);
            lo.col(col) = c - w;
            hi.col(col) = c + w;
        }
    }
    return {lo, hi};
}

bool box_contains( const Eigen::Ref<const Eigen::VectorXd>& blo,
                   const Eigen::Ref<const Eigen::VectorXd>& bhi,
                   const Eigen::Ref<const Eigen::VectorXd>& lo,
                   const Eigen::Ref<const Eigen::VectorXd>& hi )
{
    return (blo.array() <= lo.array()).all()
        && (hi.array() <= bhi.array()).all();
}

bool boxes_overlap( const Eigen::Ref<const Eigen::VectorXd>& alo,
                    const Eigen::Ref<const Eigen::VectorXd>& ahi,
                    const Eigen::Ref<const Eigen::VectorXd>& blo,
                    const Eigen::Ref<const Eigen::VectorXd>& bhi )
{
    return (alo.array() <= bhi.array()).all()
        && (blo.array() <= ahi.array()).all();
}

} // namespace

TEST_CASE("tree_cut: budget, coverage, edge cases")
{
    std::mt19937 gen(1234);
    for ( const int d : {2, 3} )
    {
        const auto [lo, hi] = clustered_boxes(
            d, 40, {Eigen::VectorXd::Zero(d)}, gen);
        const AABBTree tree(lo, hi);

        for ( const int k : {1, 2, 7, 16, 200} )
        {
            const BoxForest f = tree_cut(tree, k);
            CHECK(f.count() >= 1);
            CHECK(f.count() <= std::min(k, tree.num_leaves()));
            // every leaf is contained in at least one cut box
            for ( int leaf = 0; leaf < tree.num_leaves(); ++leaf )
            {
                bool covered = false;
                for ( int bb = 0; bb < f.count() && !covered; ++bb )
                {
                    covered = box_contains(f.lo.col(bb), f.hi.col(bb),
                                           lo.col(leaf), hi.col(leaf));
                }
                CHECK(covered);
            }
        }
    }

    // empty tree
    const AABBTree empty_tree;
    CHECK(tree_cut(empty_tree, 8).empty());
}

TEST_CASE("tree_cut: far-apart clusters get separate tight boxes")
{
    std::mt19937 gen(99);
    const int d = 2;
    std::vector<Eigen::VectorXd> centers;
    for ( const double x : {0.0, 100.0, 200.0} )
    {
        Eigen::VectorXd c(d);
        c << x, 0.0;
        centers.push_back(c);
    }
    const auto [lo, hi] = clustered_boxes(d, 30, centers, gen);
    const AABBTree tree(lo, hi);

    // k = 1: one huge box spanning ~200 in x
    const BoxForest f1 = tree_cut(tree, 1);
    REQUIRE(f1.count() == 1);
    CHECK((f1.hi(0, 0) - f1.lo(0, 0)) > 150.0);

    // The tree splits by COUNT MEDIAN (not spatial gap), so a box
    // straddling a gap sheds its minority leaves geometrically: full
    // separation of C clusters of m leaves needs k ~ C * log2(m), not
    // ~C.  Here 3 x 30 leaves separate by k = 16 (k = 8 does not).
    const BoxForest f16 = tree_cut(tree, 16);
    CHECK(f16.count() <= 16);
    for ( int bb = 0; bb < f16.count(); ++bb )
    {
        CHECK((f16.hi(0, bb) - f16.lo(0, bb)) < 50.0);
    }
}

TEST_CASE("forest_query: matches brute force, sorted unique")
{
    std::mt19937 gen(777);
    for ( const int d : {2, 3} )
    {
        Eigen::VectorXd far_center = Eigen::VectorXd::Zero(d);
        far_center(0) = 30.0;
        const auto [lo, hi] = clustered_boxes(
            d, 60, {Eigen::VectorXd::Zero(d), far_center}, gen);
        const AABBTree tree(lo, hi);

        // forest: a few random query boxes, one per region + one empty zone
        Eigen::MatrixXd flo(d, 3), fhi(d, 3);
        flo.col(0) = Eigen::VectorXd::Constant(d, -1.0);
        fhi.col(0) = Eigen::VectorXd::Constant(d, 1.0);
        flo.col(1) = far_center - Eigen::VectorXd::Constant(d, 1.5);
        fhi.col(1) = far_center + Eigen::VectorXd::Constant(d, 1.5);
        flo.col(2) = Eigen::VectorXd::Constant(d, 500.0);
        fhi.col(2) = Eigen::VectorXd::Constant(d, 501.0);
        const BoxForest forest{flo, fhi};

        const std::vector<int> got = forest_query(tree, forest);

        std::vector<int> want;
        for ( int leaf = 0; leaf < tree.num_leaves(); ++leaf )
        {
            for ( int bb = 0; bb < forest.count(); ++bb )
            {
                if ( boxes_overlap(lo.col(leaf), hi.col(leaf),
                                   forest.lo.col(bb), forest.hi.col(bb)) )
                {
                    want.push_back(leaf);
                    break;
                }
            }
        }
        CHECK(got == want);                       // brute force, ascending
        CHECK(std::is_sorted(got.begin(), got.end()));
        CHECK(std::adjacent_find(got.begin(), got.end()) == got.end());
        CHECK(!got.empty());
        CHECK(got.size() < static_cast<std::size_t>(tree.num_leaves()));
    }

    // empty cases
    const AABBTree empty_tree;
    BoxForest      empty_forest;
    empty_forest.lo.resize(2, 0);
    empty_forest.hi.resize(2, 0);
    CHECK(forest_query(empty_tree, empty_forest).empty());
}
