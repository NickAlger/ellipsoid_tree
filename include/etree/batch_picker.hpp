#pragma once
// SPDX-License-Identifier: MIT
// Part of etree — https://github.com/NickAlger/ellipsoid_tree

/// @file
/// @brief Greedy partition of a family of ellipsoids into batches of mutually non-overlapping members.
///
/// Motivating application: point-spread-function probing, where impulse
/// responses with non-overlapping supports can be probed simultaneously, so
/// fewer batches means fewer operator applications.
///
/// The pairwise overlap graph is computed once by dual-tree self-collision;
/// batches are then greedy independent sets. Candidates are considered in
/// farthest-point order of their anchor points — ellipsoids whose anchors are
/// far from everything already picked come first — which spreads each batch
/// evenly through the domain.

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "etree/object_tree.hpp"
#include "etree/detail/parallel_for.hpp"

namespace etree {

/// Partition the ellipsoids of `tree` (at the tree's tau) into batches of
/// mutually non-intersecting ellipsoids. anchor_points (one column per
/// ellipsoid) drive the farthest-point ordering. Every ellipsoid lands in
/// exactly one batch; if max_batches > 0, stop after that many batches (later
/// ellipsoids then remain unassigned).
inline std::vector<std::vector<int>>
pick_ellipsoid_batches( const EllipsoidTree&                     tree,
                        const Eigen::Ref<const Eigen::MatrixXd>& anchor_points,
                        int                                      max_batches = -1,
                        int                                      num_threads = 0 )
{
    const int n = tree.size();
    if ( static_cast<int>(anchor_points.cols()) != n )
    {
        throw std::invalid_argument(
            "etree::pick_ellipsoid_batches: need one anchor point per ellipsoid");
    }

    // Pairwise overlap graph, each edge computed once.
    std::vector<std::vector<int>> adjacent(n);
    for ( const std::pair<int, int>& pr : tree.self_collision_pairs() )
    {
        adjacent[pr.first].push_back(pr.second);
        adjacent[pr.second].push_back(pr.first);
    }

    std::vector<double> squared_distances(n, -1.0); // -1: no anchor picked yet
    std::vector<char>   in_batch(n, 0);
    int num_batched = 0;

    std::vector<std::vector<int>> batches;
    while ( num_batched < n
            && (max_batches < 0 || static_cast<int>(batches.size()) < max_batches) )
    {
        // Farthest-point candidate order; stable so the first round (all
        // distances equal) proceeds in index order.
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&]( int ii, int jj )
                         { return squared_distances[ii] > squared_distances[jj]; });

        std::vector<char> pickable(n);
        for ( int ii = 0; ii < n; ++ii )
        {
            pickable[ii] = !in_batch[ii];
        }

        std::vector<int> batch;
        for ( int idx : order )
        {
            if ( pickable[idx] )
            {
                batch.push_back(idx);
                pickable[idx] = 0;
                in_batch[idx] = 1;
                num_batched  += 1;
                for ( int jj : adjacent[idx] )
                {
                    pickable[jj] = 0;
                }
            }
        }

        // Fold this batch's anchors into each point's distance-to-picked.
        detail::parallel_for(0, n, [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
        {
            for ( std::ptrdiff_t ii = aa; ii < bb; ++ii )
            {
                double dsq = squared_distances[ii];
                for ( int ind : batch )
                {
                    const double cand =
                        (anchor_points.col(ind) - anchor_points.col(ii)).squaredNorm();
                    if ( dsq < 0.0 || cand < dsq )
                    {
                        dsq = cand;
                    }
                }
                squared_distances[ii] = dsq;
            }
        }, num_threads);

        batches.push_back(std::move(batch));
    }
    return batches;
}

/// Convenience: use the ellipsoid centers as anchors.
inline std::vector<std::vector<int>>
pick_ellipsoid_batches( const EllipsoidTree& tree, int max_batches = -1, int num_threads = 0 )
{
    const int n = tree.size();
    const int d = tree.dim();
    Eigen::MatrixXd anchors(d, n);
    for ( int ii = 0; ii < n; ++ii )
    {
        anchors.col(ii) = tree.object(ii).mu;
    }
    return pick_ellipsoid_batches(tree, anchors, max_batches, num_threads);
}

} // end namespace etree
