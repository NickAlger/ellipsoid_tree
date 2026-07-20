#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_tree — https://github.com/NickAlger/ellipsoid_tree

/// @file
/// @brief Axis-alternating geometric ordering of a point cloud (kd-tree ordering).
///
/// Recursively sort by one coordinate axis, split at the median, and recurse
/// on each half with the next axis. Nearby points end up nearby in the
/// ordering, which improves cache behavior of downstream sweeps.

#include <algorithm>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

namespace ellipsoid_tree {

namespace detail {

inline void geometric_sort_helper( int                                      start,
                                   int                                      stop,
                                   int                                      depth,
                                   const Eigen::Ref<const Eigen::MatrixXd>& points,
                                   std::vector<int>&                        sort_inds )
{
    const int num_pts_local = stop - start;
    if ( num_pts_local >= 2 )
    {
        const int axis = depth % static_cast<int>(points.rows());
        std::sort(sort_inds.begin() + start, sort_inds.begin() + stop,
                  [&](int ii, int jj) { return points(axis, ii) > points(axis, jj); });

        const int mid = start + (num_pts_local / 2);
        geometric_sort_helper(start, mid,  depth + 1, points, sort_inds);
        geometric_sort_helper(mid,   stop, depth + 1, points, sort_inds);
    }
}

} // end namespace detail

/// Returns a permutation of {0, ..., N-1} placing the points (columns of the
/// input) in axis-alternating geometric order.
inline std::vector<int> geometric_sort( const Eigen::Ref<const Eigen::MatrixXd>& points )
{
    std::vector<int> sort_inds(points.cols());
    std::iota(sort_inds.begin(), sort_inds.end(), 0);
    detail::geometric_sort_helper(0, static_cast<int>(points.cols()), 0, points, sort_inds);
    return sort_inds;
}

} // end namespace ellipsoid_tree
