// SPDX-License-Identifier: MIT
#ifndef ELLIPSOID_TREE_BOX_FOREST_HPP
#define ELLIPSOID_TREE_BOX_FOREST_HPP

/// \file box_forest.hpp
/// Compact box-forest summaries of AABB trees, and point/box queries
/// against them.
///
/// A *box forest* is a small set of axis-aligned boxes summarizing a
/// collection of geometric objects (e.g., the ellipsoid footprints owned
/// by one distributed-memory rank).  `tree_cut` extracts a forest as a
/// *cut* of an AABB tree: repeatedly splitting the largest box until the
/// budget is reached, so far-apart clusters (multi-component subdomains)
/// end up in separate tight boxes instead of one huge one.  `forest_query`
/// finds which leaves of a tree touch any box of a forest — the
/// conservative candidate query of the distributed halo protocol.
///
/// Both functions are pure geometry (no MPI): the caller exchanges
/// forests however it likes.  Summary quality affects candidate-set size
/// only, never correctness — downstream consumers resolve candidates
/// exactly.

#include "ellipsoid_tree/aabb_tree.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <queue>
#include <utility>
#include <vector>

namespace ellipsoid_tree {

/// A set of axis-aligned boxes, stored as matrix columns (dim x count).
/// The wire format of the distributed halo protocol's summaries.
struct BoxForest
{
    Eigen::MatrixXd lo; ///< Lower corners, one box per column.
    Eigen::MatrixXd hi; ///< Upper corners, one box per column.

    int  dim() const   { return static_cast<int>(lo.rows()); }
    int  count() const { return static_cast<int>(lo.cols()); }
    bool empty() const { return lo.cols() == 0; }
};

/// Cut of an AABB tree with at most `max_boxes` boxes.
///
/// Greedy refinement: starting from the root box, repeatedly replace the
/// box with the largest measure (sum of edge lengths — robust for thin
/// boxes, consistent with the descend heuristic of `visit_pairs`) by its
/// two children, until the budget is exhausted or every cut member is a
/// tree leaf.  Deterministic: ties break on the node id.  The returned
/// boxes cover every leaf of the tree.
inline BoxForest tree_cut( const AABBTree& tree, int max_boxes )
{
    BoxForest forest;
    if ( tree.empty() || max_boxes < 1 )
    {
        forest.lo.resize(tree.empty() ? 0 : tree.dim(), 0);
        forest.hi.resize(tree.empty() ? 0 : tree.dim(), 0);
        return forest;
    }

    const auto measure = [&tree]( int node )
    {
        return (tree.node_hi(node) - tree.node_lo(node)).sum();
    };

    // Max-heap of (measure, node); node id breaks ties for determinism.
    using Entry = std::pair<double, int>;
    std::priority_queue<Entry> heap;
    heap.emplace(measure(0), 0);
    std::vector<int> cut;

    while ( !heap.empty()
            && static_cast<int>(cut.size()) + static_cast<int>(heap.size())
                   < max_boxes )
    {
        const int node = heap.top().second;
        heap.pop();
        if ( tree.is_leaf(node) )
        {
            cut.push_back(node);
        }
        else
        {
            heap.emplace(measure(tree.left_child(node)), tree.left_child(node));
            heap.emplace(measure(tree.right_child(node)), tree.right_child(node));
        }
    }
    while ( !heap.empty() )
    {
        cut.push_back(heap.top().second);
        heap.pop();
    }
    std::sort(cut.begin(), cut.end());

    forest.lo.resize(tree.dim(), static_cast<int>(cut.size()));
    forest.hi.resize(tree.dim(), static_cast<int>(cut.size()));
    for ( std::size_t ii = 0; ii < cut.size(); ++ii )
    {
        forest.lo.col(static_cast<int>(ii)) = tree.node_lo(cut[ii]);
        forest.hi.col(static_cast<int>(ii)) = tree.node_hi(cut[ii]);
    }
    return forest;
}

/// External indices of the tree leaves whose boxes touch ANY box of the
/// forest, sorted ascending (deterministic order — required by the
/// distributed determinism discipline) and unique.
///
/// Typical use: `tree` bounds a rank's column points (or their leaf
/// boxes); `forest` is a remote rank's footprint summary; the result is
/// that rank's conservative candidate set.
inline std::vector<int> forest_query( const AABBTree& tree,
                                      const BoxForest& forest )
{
    std::vector<int> hits;
    if ( tree.empty() || forest.empty() )
    {
        return hits;
    }
    const AABBTree forest_tree(forest.lo, forest.hi);
    visit_pairs(
        tree, forest_tree,
        []( const Eigen::Ref<const Eigen::VectorXd>& alo,
            const Eigen::Ref<const Eigen::VectorXd>& ahi,
            const Eigen::Ref<const Eigen::VectorXd>& blo,
            const Eigen::Ref<const Eigen::VectorXd>& bhi )
        {
            return (alo.array() <= bhi.array()).all()
                && (blo.array() <= ahi.array()).all();
        },
        [&hits]( int leaf_index, int /*forest_box*/ )
        {
            hits.push_back(leaf_index);
            return true;
        } );
    std::sort(hits.begin(), hits.end());
    hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
    return hits;
}

} // end namespace ellipsoid_tree

#endif // ELLIPSOID_TREE_BOX_FOREST_HPP
