#pragma once
// SPDX-License-Identifier: MIT
// Part of etree — https://github.com/NickAlger/ellipsoid_tree

// k-nearest-neighbor search: median-split kd-tree with contiguous leaf
// blocks. Points are stored permuted into tree order so leaf blocks scan
// linearly; the median point of each internal range doubles as its splitting
// plane. Build is O(n log n) via nth_element.

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "etree/detail/parallel_for.hpp"

namespace etree {

class KDTree
{
public:
    int block_size = 32; // leaf ranges of at most this many points are scanned linearly

    KDTree() = default;

    explicit KDTree( const Eigen::Ref<const Eigen::MatrixXd>& points )
    {
        build(points);
    }

    void build( const Eigen::Ref<const Eigen::MatrixXd>& input_points )
    {
        dim_     = static_cast<int>(input_points.rows());
        num_pts_ = static_cast<int>(input_points.cols());

        std::vector<int> perm(num_pts_);
        std::iota(perm.begin(), perm.end(), 0);
        build_subtree(0, num_pts_, 0, input_points, perm);

        points_.resize(dim_, num_pts_);
        perm_i2e_.resize(num_pts_);
        for ( int ii = 0; ii < num_pts_; ++ii )
        {
            perm_i2e_(ii) = perm[ii];
            points_.col(ii) = input_points.col(perm[ii]);
        }
    }

    int size() const { return num_pts_; }
    int dim() const  { return dim_; }

    // k nearest neighbors of each query point (columns). Returns (indices,
    // squared distances), each of shape (k_eff, num_queries) with
    // k_eff = min(k, size()), sorted by increasing distance per query.
    std::pair<Eigen::MatrixXi, Eigen::MatrixXd>
    query( const Eigen::Ref<const Eigen::MatrixXd>& query_points,
           int num_neighbors,
           int num_threads = 0 ) const
    {
        if ( num_neighbors < 1 )
        {
            throw std::invalid_argument("etree::KDTree::query: num_neighbors must be >= 1");
        }
        const int k_eff = std::min(num_neighbors, num_pts_);
        const int num_queries = static_cast<int>(query_points.cols());

        Eigen::MatrixXi inds(k_eff, num_queries);
        Eigen::MatrixXd dsqs(k_eff, num_queries);

        detail::parallel_for(0, num_queries, [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
        {
            for ( std::ptrdiff_t qq = aa; qq < bb; ++qq )
            {
                query_one(query_points.col(qq), k_eff, static_cast<int>(qq), inds, dsqs);
            }
        }, num_threads);

        return std::make_pair(std::move(inds), std::move(dsqs));
    }

private:
    // Bounded max-heap of (squared distance, internal index) candidates.
    struct KnnQueue
    {
        std::priority_queue<std::pair<double, int>> heap;
        int capacity;

        void offer( double dsq, int ind )
        {
            if ( static_cast<int>(heap.size()) < capacity )
            {
                heap.push({dsq, ind});
            }
            else if ( dsq < heap.top().first )
            {
                heap.pop();
                heap.push({dsq, ind});
            }
        }

        double worst() const
        {
            return ( static_cast<int>(heap.size()) < capacity )
                   ? std::numeric_limits<double>::infinity()
                   : heap.top().first;
        }
    };

    void build_subtree( int start, int stop, int depth,
                        const Eigen::Ref<const Eigen::MatrixXd>& pts,
                        std::vector<int>& perm )
    {
        if ( stop - start <= block_size )
        {
            return; // leaf block: order irrelevant
        }
        const int axis = depth % dim_;
        const int mid  = start + (stop - start) / 2;
        std::nth_element(perm.begin() + start, perm.begin() + mid, perm.begin() + stop,
                         [&]( int ii, int jj ) { return pts(axis, ii) < pts(axis, jj); });
        build_subtree(start,   mid,  depth + 1, pts, perm);
        build_subtree(mid + 1, stop, depth + 1, pts, perm);
    }

    void query_subtree( const Eigen::Ref<const Eigen::VectorXd>& q,
                        KnnQueue& best, int start, int stop, int depth ) const
    {
        if ( stop - start <= block_size )
        {
            for ( int ii = start; ii < stop; ++ii )
            {
                best.offer((points_.col(ii) - q).squaredNorm(), ii);
            }
            return;
        }

        const int    axis  = depth % dim_;
        const int    mid   = start + (stop - start) / 2;
        const double delta = q(axis) - points_(axis, mid);

        int near_start, near_stop, far_start, far_stop;
        if ( delta <= 0.0 )
        {
            near_start = start;   near_stop = mid;
            far_start  = mid + 1; far_stop  = stop;
        }
        else
        {
            near_start = mid + 1; near_stop = stop;
            far_start  = start;   far_stop  = mid;
        }

        query_subtree(q, best, near_start, near_stop, depth + 1);
        best.offer((points_.col(mid) - q).squaredNorm(), mid);
        if ( delta * delta <= best.worst() )
        {
            query_subtree(q, best, far_start, far_stop, depth + 1);
        }
    }

    void query_one( const Eigen::Ref<const Eigen::VectorXd>& q, int k, int col,
                    Eigen::MatrixXi& inds, Eigen::MatrixXd& dsqs ) const
    {
        KnnQueue best;
        best.capacity = k;
        query_subtree(q, best, 0, num_pts_, 0);

        for ( int pos = k - 1; pos >= 0; --pos ) // drain max-heap into ascending order
        {
            const std::pair<double, int> top = best.heap.top();
            best.heap.pop();
            inds(pos, col) = perm_i2e_(top.second);
            dsqs(pos, col) = top.first;
        }
    }

    int             dim_     = 0;
    int             num_pts_ = 0;
    Eigen::MatrixXd points_;   // (dim, n), permuted into tree order
    Eigen::VectorXi perm_i2e_; // internal -> external index
};

} // end namespace etree
