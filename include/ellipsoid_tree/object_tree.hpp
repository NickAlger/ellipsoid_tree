#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_tree — https://github.com/NickAlger/ellipsoid_tree

/// @file
/// @brief Concrete spatial indexes over the geometric types: BoxTree, BallTree, EllipsoidTree, SimplexTree.
///
/// Each is a thin façade over one shared pattern: contiguous element storage
/// + an AABBTree over the elements' bounding boxes + broad-phase traversal +
/// narrow-phase tests from the intersection table.
///
/// collisions(query [, tau]) returns the indices of elements intersecting the
/// query (order unspecified). collisions_batch(...) runs many queries with
/// parallel_for. collision_pairs(treeA, treeB) and self_collision_pairs() use
/// dual-tree traversal.
///
/// Per-element data that the free predicates would recompute per call is
/// hoisted at build time: EllipsoidTree stores each Sigma^{-1}, SimplexTree
/// stores each affine-coordinate transform. Ellipsoid queries use tiered
/// pruning: a cheap bounding-box test at every node, then (by default) the
/// exact ellipsoid-box QP on survivors — toggle with exact_ellipsoid_pruning.
///
/// tau semantics: EllipsoidTree fixes tau for its elements at construction
/// (leaf boxes depend on it). Queries default to that tau; passing a smaller
/// tau is allowed (pruning stays conservative), a larger one throws — call
/// rebuild(new_tau) instead. Trees of non-ellipsoid elements take tau
/// explicitly on ellipsoid queries.

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "ellipsoid_tree/geometry.hpp"
#include "ellipsoid_tree/intersections.hpp"
#include "ellipsoid_tree/aabb_tree.hpp"
#include "ellipsoid_tree/detail/parallel_for.hpp"

namespace ellipsoid_tree {

namespace detail {

template <class NodePredicate, class Narrow>
inline std::vector<int> tree_collisions( const AABBTree& tree, NodePredicate&& np, Narrow&& narrow )
{
    std::vector<int> out;
    tree.visit(np, [&]( int e )
    {
        if ( narrow(e) )
        {
            out.push_back(e);
        }
        return true;
    });
    return out;
}

// Node predicates capture their query by value so the returned closures are
// self-contained.
inline auto point_node_pred( Eigen::VectorXd p )
{
    return [p = std::move(p)]( const auto& lo, const auto& hi )
    { return (lo.array() <= p.array()).all() && (p.array() <= hi.array()).all(); };
}

inline auto box_node_pred( Box q )
{
    return [q = std::move(q)]( const auto& lo, const auto& hi )
    { return (q.lo.array() <= hi.array()).all() && (lo.array() <= q.hi.array()).all(); };
}

inline auto ball_node_pred( Ball q )
{
    return [q = std::move(q)]( const auto& lo, const auto& hi )
    {
        Eigen::VectorXd closest = q.center.cwiseMax(lo.matrix()).cwiseMin(hi.matrix());
        return (closest - q.center).squaredNorm() <= q.radius * q.radius;
    };
}

inline auto segment_node_pred( Segment q )
{
    return [q = std::move(q)]( const auto& lo, const auto& hi )
    { return intersects(q, Box{lo, hi}); };
}

inline auto halfspace_node_pred( Halfspace q )
{
    return [q = std::move(q)]( const auto& lo, const auto& hi )
    { return intersects(Box{lo, hi}, q); };
}

// Tiered ellipsoid-query predicate: bounding-box reject first, then the
// exact ellipsoid-box QP (with early exit at tau^2) on survivors.
inline auto ellipsoid_node_pred( Ellipsoid q, double tau, bool exact )
{
    Box qbb = bounding_box(q, tau);
    Eigen::MatrixXd Minv = inverse_spd(q.Sigma); // hoisted once per query
    const double t2 = tau * tau;
    return [q = std::move(q), qbb = std::move(qbb), Minv = std::move(Minv), t2, exact]
           ( const auto& lo, const auto& hi )
    {
        if ( !((qbb.lo.array() <= hi.array()).all() && (lo.array() <= qbb.hi.array()).all()) )
        {
            return false;
        }
        if ( !exact )
        {
            return true;
        }
        BoxClosestPointOptions opts;
        opts.stop_below = t2;
        return closest_point_in_box(q.mu, Box{lo, hi}, Minv, opts).distance_squared <= t2;
    };
}

// Batched queries, shared by all façades via CRTP.
template <class Derived>
struct TreeBatchQueries
{
    template <class Query>
    std::vector<std::vector<int>> collisions_batch( const std::vector<Query>& queries,
                                                    int num_threads = 0 ) const
    {
        const Derived& self = static_cast<const Derived&>(*this);
        std::vector<std::vector<int>> out(queries.size());
        parallel_for(0, static_cast<std::ptrdiff_t>(queries.size()),
                     [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
                     {
                         for ( std::ptrdiff_t ii = aa; ii < bb; ++ii )
                         {
                             out[ii] = self.collisions(queries[ii]);
                         }
                     }, num_threads);
        return out;
    }

    template <class Query>
    std::vector<std::vector<int>> collisions_batch( const std::vector<Query>& queries,
                                                    double tau, int num_threads = 0 ) const
    {
        const Derived& self = static_cast<const Derived&>(*this);
        std::vector<std::vector<int>> out(queries.size());
        parallel_for(0, static_cast<std::ptrdiff_t>(queries.size()),
                     [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
                     {
                         for ( std::ptrdiff_t ii = aa; ii < bb; ++ii )
                         {
                             out[ii] = self.collisions(queries[ii], tau);
                         }
                     }, num_threads);
        return out;
    }

    // Points as columns of a matrix.
    std::vector<std::vector<int>> point_collisions_batch( const Eigen::Ref<const Eigen::MatrixXd>& points,
                                                          int num_threads = 0 ) const
    {
        const Derived& self = static_cast<const Derived&>(*this);
        std::vector<std::vector<int>> out(points.cols());
        parallel_for(0, static_cast<std::ptrdiff_t>(points.cols()),
                     [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
                     {
                         for ( std::ptrdiff_t ii = aa; ii < bb; ++ii )
                         {
                             out[ii] = self.collisions(points.col(ii));
                         }
                     }, num_threads);
        return out;
    }
};

template <class Narrow>
inline std::vector<std::pair<int, int>> collect_pairs( const AABBTree& A, const AABBTree& B,
                                                       Narrow&& narrow )
{
    std::vector<std::pair<int, int>> out;
    visit_pairs(A, B,
                []( const auto& la, const auto& ha, const auto& lb, const auto& hb )
                { return (la.array() <= hb.array()).all() && (lb.array() <= ha.array()).all(); },
                [&]( int ia, int ib )
                {
                    if ( narrow(ia, ib) )
                    {
                        out.emplace_back(ia, ib);
                    }
                    return true;
                });
    return out;
}

template <class Narrow>
inline std::vector<std::pair<int, int>> collect_self_pairs( const AABBTree& T, Narrow&& narrow )
{
    std::vector<std::pair<int, int>> out;
    visit_self_pairs(T,
                     []( const auto& la, const auto& ha, const auto& lb, const auto& hb )
                     { return (la.array() <= hb.array()).all() && (lb.array() <= ha.array()).all(); },
                     [&]( int ia, int ib )
                     {
                         if ( narrow(ia, ib) )
                         {
                             out.emplace_back(std::min(ia, ib), std::max(ia, ib));
                         }
                         return true;
                     });
    return out;
}

} // end namespace detail


// ------------------------------------------------------------------
//  BoxTree
// ------------------------------------------------------------------

/// Spatial index over a collection of axis-aligned boxes.
class BoxTree : public detail::TreeBatchQueries<BoxTree>
{
public:
    bool exact_ellipsoid_pruning = true; ///< Enable the exact ellipsoid-box test during node pruning (vs. bounding-box only).

    /// Empty tree (indexes no boxes).
    BoxTree() = default;

    /// Index a collection of axis-aligned boxes, building an AABBTree over them.
    explicit BoxTree( std::vector<Box> boxes )
        : objects_(std::move(boxes))
    {
        const int n = static_cast<int>(objects_.size());
        const int d = (n > 0) ? static_cast<int>(objects_[0].lo.size()) : 0;
        Eigen::MatrixXd lo(d, n);
        Eigen::MatrixXd hi(d, n);
        for ( int ii = 0; ii < n; ++ii )
        {
            if ( objects_[ii].lo.size() != d || objects_[ii].hi.size() != d )
            {
                throw std::invalid_argument("ellipsoid_tree::BoxTree: inconsistent box dimensions");
            }
            lo.col(ii) = objects_[ii].lo;
            hi.col(ii) = objects_[ii].hi;
        }
        aabb_.build(lo, hi);
    }

    /// Index boxes given as columns of lower- and upper-corner matrices (dim x n).
    BoxTree( const Eigen::Ref<const Eigen::MatrixXd>& lo,
             const Eigen::Ref<const Eigen::MatrixXd>& hi )
        : BoxTree([&]
          {
              std::vector<Box> boxes(lo.cols());
              for ( int ii = 0; ii < lo.cols(); ++ii )
              {
                  boxes[ii] = Box{lo.col(ii), hi.col(ii)};
              }
              return boxes;
          }())
    {
    }

    /// Number of indexed boxes.
    int size() const { return static_cast<int>(objects_.size()); }
    /// Spatial dimension of the indexed boxes.
    int dim() const  { return aabb_.dim(); }
    /// The ii-th indexed box.
    const Box& object( int ii ) const           { return objects_[ii]; }
    /// All indexed boxes, in index order.
    const std::vector<Box>& objects() const     { return objects_; }
    /// The underlying AABB tree over the elements' bounding boxes.
    const AABBTree& tree() const                { return aabb_; }

    /// Indices of the boxes containing the query point.
    std::vector<int> collisions( const Eigen::Ref<const Eigen::VectorXd>& p ) const
    {
        return detail::tree_collisions(aabb_, detail::point_node_pred(p),
            [&]( int e ) { return intersects(p, objects_[e]); });
    }
    /// Indices of the boxes intersecting the query box.
    std::vector<int> collisions( const Box& q ) const
    {
        return detail::tree_collisions(aabb_, detail::box_node_pred(q),
            [&]( int e ) { return intersects(objects_[e], q); });
    }
    /// Indices of the boxes intersecting the query ball.
    std::vector<int> collisions( const Ball& q ) const
    {
        return detail::tree_collisions(aabb_, detail::ball_node_pred(q),
            [&]( int e ) { return intersects(objects_[e], q); });
    }
    /// Indices of the boxes intersecting the query ellipsoid at scale tau.
    std::vector<int> collisions( const Ellipsoid& q, double tau ) const
    {
        return detail::tree_collisions(aabb_,
            detail::ellipsoid_node_pred(q, tau, exact_ellipsoid_pruning),
            [&]( int e ) { return intersects(objects_[e], q, tau); });
    }
    /// Indices of the boxes intersecting the query simplex.
    std::vector<int> collisions( const Simplex& q ) const
    {
        return detail::tree_collisions(aabb_, detail::box_node_pred(bounding_box(q)),
            [&]( int e ) { return intersects(objects_[e], q); });
    }
    /// Indices of the boxes intersecting the query segment.
    std::vector<int> collisions( const Segment& q ) const
    {
        return detail::tree_collisions(aabb_, detail::segment_node_pred(q),
            [&]( int e ) { return intersects(q, objects_[e]); });
    }
    /// Indices of the boxes intersecting the query halfspace.
    std::vector<int> collisions( const Halfspace& q ) const
    {
        return detail::tree_collisions(aabb_, detail::halfspace_node_pred(q),
            [&]( int e ) { return intersects(objects_[e], q); });
    }

    /// Every intersecting pair within the family of boxes (the overlap graph).
    std::vector<std::pair<int, int>> self_collision_pairs() const
    {
        return detail::collect_self_pairs(aabb_,
            [&]( int ii, int jj ) { return intersects(objects_[ii], objects_[jj]); });
    }

private:
    std::vector<Box> objects_;
    AABBTree         aabb_;
};


// ------------------------------------------------------------------
//  BallTree
// ------------------------------------------------------------------

/// Spatial index over a collection of balls.
class BallTree : public detail::TreeBatchQueries<BallTree>
{
public:
    bool exact_ellipsoid_pruning = true; ///< Enable the exact ellipsoid-box test during node pruning (vs. bounding-box only).

    /// Empty tree (indexes no balls).
    BallTree() = default;

    /// Index a collection of balls, building an AABBTree over their bounding boxes.
    explicit BallTree( std::vector<Ball> balls )
        : objects_(std::move(balls))
    {
        const int n = static_cast<int>(objects_.size());
        const int d = (n > 0) ? static_cast<int>(objects_[0].center.size()) : 0;
        Eigen::MatrixXd lo(d, n);
        Eigen::MatrixXd hi(d, n);
        for ( int ii = 0; ii < n; ++ii )
        {
            if ( objects_[ii].center.size() != d || objects_[ii].radius < 0.0 )
            {
                throw std::invalid_argument("ellipsoid_tree::BallTree: inconsistent dimensions or negative radius");
            }
            Box bb = bounding_box(objects_[ii]);
            lo.col(ii) = bb.lo;
            hi.col(ii) = bb.hi;
        }
        aabb_.build(lo, hi);
    }

    /// Points with radii; zero radii give a point tree usable for region queries.
    BallTree( const Eigen::Ref<const Eigen::MatrixXd>& centers,
              const Eigen::Ref<const Eigen::VectorXd>& radii )
        : BallTree([&]
          {
              if ( centers.cols() != radii.size() )
              {
                  throw std::invalid_argument("ellipsoid_tree::BallTree: centers/radii count mismatch");
              }
              std::vector<Ball> balls(centers.cols());
              for ( int ii = 0; ii < centers.cols(); ++ii )
              {
                  balls[ii] = Ball{centers.col(ii), radii(ii)};
              }
              return balls;
          }())
    {
    }

    /// Number of indexed balls.
    int size() const { return static_cast<int>(objects_.size()); }
    /// Spatial dimension of the indexed balls.
    int dim() const  { return aabb_.dim(); }
    /// The ii-th indexed ball.
    const Ball& object( int ii ) const          { return objects_[ii]; }
    /// All indexed balls, in index order.
    const std::vector<Ball>& objects() const    { return objects_; }
    /// The underlying AABB tree over the elements' bounding boxes.
    const AABBTree& tree() const                { return aabb_; }

    /// Indices of the balls containing the query point.
    std::vector<int> collisions( const Eigen::Ref<const Eigen::VectorXd>& p ) const
    {
        return detail::tree_collisions(aabb_, detail::point_node_pred(p),
            [&]( int e ) { return intersects(p, objects_[e]); });
    }
    /// Indices of the balls intersecting the query box.
    std::vector<int> collisions( const Box& q ) const
    {
        return detail::tree_collisions(aabb_, detail::box_node_pred(q),
            [&]( int e ) { return intersects(q, objects_[e]); });
    }
    /// Indices of the balls intersecting the query ball.
    std::vector<int> collisions( const Ball& q ) const
    {
        return detail::tree_collisions(aabb_, detail::ball_node_pred(q),
            [&]( int e ) { return intersects(objects_[e], q); });
    }
    /// Indices of the balls intersecting the query ellipsoid at scale tau.
    std::vector<int> collisions( const Ellipsoid& q, double tau ) const
    {
        return detail::tree_collisions(aabb_,
            detail::ellipsoid_node_pred(q, tau, exact_ellipsoid_pruning),
            [&]( int e ) { return intersects(objects_[e], q, tau); });
    }
    /// Indices of the balls intersecting the query simplex.
    std::vector<int> collisions( const Simplex& q ) const
    {
        return detail::tree_collisions(aabb_, detail::box_node_pred(bounding_box(q)),
            [&]( int e ) { return intersects(objects_[e], q); });
    }
    /// Indices of the balls intersecting the query segment.
    std::vector<int> collisions( const Segment& q ) const
    {
        return detail::tree_collisions(aabb_, detail::segment_node_pred(q),
            [&]( int e ) { return intersects(q, objects_[e]); });
    }
    /// Indices of the balls intersecting the query halfspace.
    std::vector<int> collisions( const Halfspace& q ) const
    {
        return detail::tree_collisions(aabb_, detail::halfspace_node_pred(q),
            [&]( int e ) { return intersects(objects_[e], q); });
    }

    /// Every intersecting pair within the family of balls (the overlap graph).
    std::vector<std::pair<int, int>> self_collision_pairs() const
    {
        return detail::collect_self_pairs(aabb_,
            [&]( int ii, int jj ) { return intersects(objects_[ii], objects_[jj]); });
    }

private:
    std::vector<Ball> objects_;
    AABBTree          aabb_;
};


// ------------------------------------------------------------------
//  EllipsoidTree
// ------------------------------------------------------------------

/// Spatial index over a collection of ellipsoids sharing a fixed scale tau.
class EllipsoidTree : public detail::TreeBatchQueries<EllipsoidTree>
{
public:
    bool exact_ellipsoid_pruning = true; ///< Enable the exact ellipsoid-box test during node pruning (vs. bounding-box only).

    /// Empty tree (indexes no ellipsoids).
    EllipsoidTree() = default;

    /// Index a family of ellipsoids at the shared scale tau, precomputing each Sigma^{-1} and building the tree over their tau-scaled bounding boxes.
    EllipsoidTree( std::vector<Ellipsoid> ellipsoids, double tau, int num_threads = 0 )
        : objects_(std::move(ellipsoids)), tau_(tau)
    {
        if ( !(tau > 0.0) )
        {
            throw std::invalid_argument("ellipsoid_tree::EllipsoidTree: tau must be positive");
        }
        const int n = static_cast<int>(objects_.size());
        const int d = (n > 0) ? static_cast<int>(objects_[0].mu.size()) : 0;
        for ( int ii = 0; ii < n; ++ii )
        {
            if ( objects_[ii].mu.size() != d
                 || objects_[ii].Sigma.rows() != d || objects_[ii].Sigma.cols() != d )
            {
                throw std::invalid_argument("ellipsoid_tree::EllipsoidTree: inconsistent dimensions");
            }
        }
        sigma_inv_.resize(n);
        detail::parallel_for(0, n, [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
        {
            for ( std::ptrdiff_t ii = aa; ii < bb; ++ii )
            {
                sigma_inv_[ii] = detail::inverse_spd(objects_[ii].Sigma);
            }
        }, num_threads);
        rebuild_boxes();
    }

    /// Number of indexed ellipsoids.
    int size() const   { return static_cast<int>(objects_.size()); }
    /// Spatial dimension of the indexed ellipsoids.
    int dim() const    { return aabb_.dim(); }
    /// The shared ellipsoid scale tau fixed at construction (used for the leaf boxes).
    double tau() const { return tau_; }
    /// The ii-th indexed ellipsoid.
    const Ellipsoid& object( int ii ) const          { return objects_[ii]; }
    /// All indexed ellipsoids, in index order.
    const std::vector<Ellipsoid>& objects() const    { return objects_; }
    /// The underlying AABB tree over the elements' bounding boxes.
    const AABBTree& tree() const                     { return aabb_; }
    /// The precomputed inverse Sigma^{-1} of the ii-th ellipsoid's covariance.
    const Eigen::MatrixXd& sigma_inverse( int ii ) const { return sigma_inv_[ii]; }

    /// Rebuild the leaf boxes (and tree) at a new scale; Sigma^{-1} is unchanged.
    void rebuild( double new_tau )
    {
        if ( !(new_tau > 0.0) )
        {
            throw std::invalid_argument("ellipsoid_tree::EllipsoidTree::rebuild: tau must be positive");
        }
        tau_ = new_tau;
        rebuild_boxes();
    }

    /// Indices of the ellipsoids (at the build tau) that contain the query point.
    std::vector<int> collisions( const Eigen::Ref<const Eigen::VectorXd>& p ) const
    { return collisions(p, tau_); }
    /// Indices of the ellipsoids at scale tau (at most the build tau) that contain the query point.
    std::vector<int> collisions( const Eigen::Ref<const Eigen::VectorXd>& p, double tau ) const
    {
        check_query_tau(tau);
        const double t2 = tau * tau;
        return detail::tree_collisions(aabb_, detail::point_node_pred(p),
            [&]( int e )
            {
                Eigen::VectorXd z = p - objects_[e].mu;
                return z.dot(sigma_inv_[e] * z) <= t2;
            });
    }

    /// Indices of the ellipsoids (at the build tau) intersecting the query box.
    std::vector<int> collisions( const Box& q ) const { return collisions(q, tau_); }
    /// Indices of the ellipsoids at scale tau (at most the build tau) intersecting the query box.
    std::vector<int> collisions( const Box& q, double tau ) const
    {
        check_query_tau(tau);
        const double t2 = tau * tau;
        return detail::tree_collisions(aabb_, detail::box_node_pred(q),
            [&]( int e )
            {
                BoxClosestPointOptions opts;
                opts.stop_below = t2;
                return closest_point_in_box(objects_[e].mu, q, sigma_inv_[e], opts).distance_squared <= t2;
            });
    }

    /// Indices of the ellipsoids (at the build tau) intersecting the query ball.
    std::vector<int> collisions( const Ball& q ) const { return collisions(q, tau_); }
    /// Indices of the ellipsoids at scale tau (at most the build tau) intersecting the query ball.
    std::vector<int> collisions( const Ball& q, double tau ) const
    {
        check_query_tau(tau);
        return detail::tree_collisions(aabb_, detail::ball_node_pred(q),
            [&]( int e ) { return intersects(q, objects_[e], tau); });
    }

    /// Indices of the ellipsoids (at the build tau) intersecting the query ellipsoid at that same scale.
    std::vector<int> collisions( const Ellipsoid& q ) const { return collisions(q, tau_); }
    /// Indices of the ellipsoids intersecting the query ellipsoid, both taken at scale tau (at most the build tau).
    std::vector<int> collisions( const Ellipsoid& q, double tau ) const
    {
        check_query_tau(tau);
        return detail::tree_collisions(aabb_,
            detail::ellipsoid_node_pred(q, tau, exact_ellipsoid_pruning),
            [&]( int e ) { return intersects(objects_[e], q, tau); });
    }

    /// Indices of the ellipsoids (at the build tau) intersecting the query simplex.
    std::vector<int> collisions( const Simplex& q ) const { return collisions(q, tau_); }
    /// Indices of the ellipsoids at scale tau (at most the build tau) intersecting the query simplex.
    std::vector<int> collisions( const Simplex& q, double tau ) const
    {
        check_query_tau(tau);
        const double t2 = tau * tau;
        return detail::tree_collisions(aabb_, detail::box_node_pred(bounding_box(q)),
            [&]( int e )
            {
                return closest_point_in_simplex(objects_[e].mu, q.V, sigma_inv_[e]).distance_squared <= t2;
            });
    }

    /// Indices of the ellipsoids (at the build tau) intersecting the query segment.
    std::vector<int> collisions( const Segment& q ) const { return collisions(q, tau_); }
    /// Indices of the ellipsoids at scale tau (at most the build tau) intersecting the query segment.
    std::vector<int> collisions( const Segment& q, double tau ) const
    {
        check_query_tau(tau);
        const double t2 = tau * tau;
        return detail::tree_collisions(aabb_, detail::segment_node_pred(q),
            [&]( int e )
            {
                return detail::segment_mahalanobis_min(q.a, q.b, objects_[e].mu, sigma_inv_[e]) <= t2;
            });
    }

    /// Indices of the ellipsoids (at the build tau) intersecting the query halfspace.
    std::vector<int> collisions( const Halfspace& q ) const { return collisions(q, tau_); }
    /// Indices of the ellipsoids at scale tau (at most the build tau) intersecting the query halfspace.
    std::vector<int> collisions( const Halfspace& q, double tau ) const
    {
        check_query_tau(tau);
        return detail::tree_collisions(aabb_, detail::halfspace_node_pred(q),
            [&]( int e ) { return intersects(objects_[e], q, tau); });
    }

    /// Every intersecting pair within the family of ellipsoids at the build tau (the overlap graph).
    std::vector<std::pair<int, int>> self_collision_pairs() const
    {
        return detail::collect_self_pairs(aabb_,
            [&]( int ii, int jj ) { return intersects(objects_[ii], objects_[jj], tau_); });
    }

private:
    void check_query_tau( double tau ) const
    {
        if ( !(tau > 0.0) )
        {
            throw std::invalid_argument("ellipsoid_tree::EllipsoidTree: query tau must be positive");
        }
        if ( tau > tau_ * (1.0 + 1e-12) )
        {
            throw std::invalid_argument(
                "ellipsoid_tree::EllipsoidTree: query tau exceeds the build tau; call rebuild(new_tau) first");
        }
    }

    void rebuild_boxes()
    {
        const int n = static_cast<int>(objects_.size());
        const int d = (n > 0) ? static_cast<int>(objects_[0].mu.size()) : 0;
        Eigen::MatrixXd lo(d, n);
        Eigen::MatrixXd hi(d, n);
        for ( int ii = 0; ii < n; ++ii )
        {
            Box bb = bounding_box(objects_[ii], tau_);
            lo.col(ii) = bb.lo;
            hi.col(ii) = bb.hi;
        }
        aabb_.build(lo, hi);
    }

    std::vector<Ellipsoid>       objects_;
    std::vector<Eigen::MatrixXd> sigma_inv_;
    double                       tau_ = 1.0;
    AABBTree                     aabb_;
};


// ------------------------------------------------------------------
//  SimplexTree
// ------------------------------------------------------------------

/// Spatial index over a collection of simplices (e.g. a mesh).
class SimplexTree : public detail::TreeBatchQueries<SimplexTree>
{
public:
    bool exact_ellipsoid_pruning = true; ///< Enable the exact ellipsoid-box test during node pruning (vs. bounding-box only).

    /// Empty tree (indexes no simplices).
    SimplexTree() = default;

    /// Index a collection of simplices, precomputing each affine-coordinate transform and building the tree over their bounding boxes.
    explicit SimplexTree( std::vector<Simplex> simplices, int num_threads = 0 )
        : objects_(std::move(simplices))
    {
        const int n = static_cast<int>(objects_.size());
        const int d = (n > 0) ? static_cast<int>(objects_[0].V.rows()) : 0;
        Eigen::MatrixXd lo(d, n);
        Eigen::MatrixXd hi(d, n);
        for ( int ii = 0; ii < n; ++ii )
        {
            if ( objects_[ii].V.rows() != d || objects_[ii].V.cols() < 1 )
            {
                throw std::invalid_argument("ellipsoid_tree::SimplexTree: inconsistent simplex dimensions");
            }
            Box bb = bounding_box(objects_[ii]);
            lo.col(ii) = bb.lo;
            hi.col(ii) = bb.hi;
        }
        transform_A_.resize(n);
        transform_b_.resize(n);
        detail::parallel_for(0, n, [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
        {
            for ( std::ptrdiff_t ii = aa; ii < bb; ++ii )
            {
                std::pair<Eigen::MatrixXd, Eigen::VectorXd> Ab = simplex_transform(objects_[ii].V);
                transform_A_[ii] = std::move(Ab.first);
                transform_b_[ii] = std::move(Ab.second);
            }
        }, num_threads);
        aabb_.build(lo, hi);
    }

    /// Mesh convenience: vertices (dim x num_vertices), cells (K x num_cells)
    /// of vertex indices; one simplex per cell column.
    SimplexTree( const Eigen::Ref<const Eigen::MatrixXd>& vertices,
                 const Eigen::Ref<const Eigen::MatrixXi>& cells,
                 int num_threads = 0 )
        : SimplexTree([&]
          {
              if ( cells.size() > 0
                   && (cells.minCoeff() < 0 || cells.maxCoeff() >= vertices.cols()) )
              {
                  throw std::invalid_argument("ellipsoid_tree::SimplexTree: cell vertex index out of range");
              }
              std::vector<Simplex> simplices(cells.cols());
              for ( int cc = 0; cc < cells.cols(); ++cc )
              {
                  Eigen::MatrixXd V(vertices.rows(), cells.rows());
                  for ( int kk = 0; kk < cells.rows(); ++kk )
                  {
                      V.col(kk) = vertices.col(cells(kk, cc));
                  }
                  simplices[cc] = Simplex{std::move(V)};
              }
              return simplices;
          }(), num_threads)
    {
    }

    /// Number of indexed simplices.
    int size() const { return static_cast<int>(objects_.size()); }
    /// Spatial dimension of the indexed simplices.
    int dim() const  { return aabb_.dim(); }
    /// The ii-th indexed simplex.
    const Simplex& object( int ii ) const          { return objects_[ii]; }
    /// All indexed simplices, in index order.
    const std::vector<Simplex>& objects() const    { return objects_; }
    /// The underlying AABB tree over the elements' bounding boxes.
    const AABBTree& tree() const                   { return aabb_; }

    /// Affine (barycentric) coordinates of p with respect to element ii.
    Eigen::VectorXd affine_coordinates( int ii, const Eigen::Ref<const Eigen::VectorXd>& p ) const
    {
        return transform_A_[ii] * p + transform_b_[ii];
    }

    /// Indices of the simplices containing the query point.
    std::vector<int> collisions( const Eigen::Ref<const Eigen::VectorXd>& p ) const
    {
        return detail::tree_collisions(aabb_, detail::point_node_pred(p),
            [&]( int e ) { return point_in_element(e, p); });
    }

    /// Index of one element containing p, or -1 (early-exit traversal).
    int first_collision( const Eigen::Ref<const Eigen::VectorXd>& p ) const
    {
        int found = -1;
        aabb_.visit(detail::point_node_pred(p),
                    [&]( int e )
                    {
                        if ( point_in_element(e, p) )
                        {
                            found = e;
                            return false;
                        }
                        return true;
                    });
        return found;
    }

    /// Indices of the simplices intersecting the query box.
    std::vector<int> collisions( const Box& q ) const
    {
        return detail::tree_collisions(aabb_, detail::box_node_pred(q),
            [&]( int e ) { return intersects(q, objects_[e]); });
    }
    /// Indices of the simplices intersecting the query simplex.
    std::vector<int> collisions( const Simplex& q ) const
    {
        return detail::tree_collisions(aabb_, detail::box_node_pred(bounding_box(q)),
            [&]( int e ) { return intersects(objects_[e], q); });
    }
    /// Indices of the simplices intersecting the query ball.
    std::vector<int> collisions( const Ball& q ) const
    {
        return detail::tree_collisions(aabb_, detail::ball_node_pred(q),
            [&]( int e ) { return intersects(q, objects_[e]); });
    }
    /// Indices of the simplices intersecting the query ellipsoid at scale tau.
    std::vector<int> collisions( const Ellipsoid& q, double tau ) const
    {
        const double t2 = tau * tau;
        Eigen::MatrixXd Minv = detail::inverse_spd(q.Sigma); // hoisted once per query
        return detail::tree_collisions(aabb_,
            detail::ellipsoid_node_pred(q, tau, exact_ellipsoid_pruning),
            [&]( int e )
            {
                return closest_point_in_simplex(q.mu, objects_[e].V, Minv).distance_squared <= t2;
            });
    }
    /// Indices of the simplices intersecting the query segment.
    std::vector<int> collisions( const Segment& q ) const
    {
        return detail::tree_collisions(aabb_, detail::segment_node_pred(q),
            [&]( int e )
            {
                if ( objects_[e].V.cols() == dim() + 1 )
                {
                    return detail::segment_affine_feasible(
                        transform_A_[e] * q.a + transform_b_[e],
                        transform_A_[e] * (q.b - q.a));
                }
                return intersects(q, objects_[e]);
            });
    }
    /// Indices of the simplices intersecting the query halfspace.
    std::vector<int> collisions( const Halfspace& q ) const
    {
        return detail::tree_collisions(aabb_, detail::halfspace_node_pred(q),
            [&]( int e ) { return intersects(objects_[e], q); });
    }

    /// Every intersecting pair within the family of simplices (the overlap graph).
    std::vector<std::pair<int, int>> self_collision_pairs() const
    {
        return detail::collect_self_pairs(aabb_,
            [&]( int ii, int jj ) { return intersects(objects_[ii], objects_[jj]); });
    }

private:
    bool point_in_element( int e, const Eigen::Ref<const Eigen::VectorXd>& p ) const
    {
        if ( objects_[e].V.cols() == dim() + 1 )
        {
            return ((transform_A_[e] * p + transform_b_[e]).array() >= 0.0).all();
        }
        return intersects(p, objects_[e]); // lower-dimensional: tolerance semantics
    }

    std::vector<Simplex>         objects_;
    std::vector<Eigen::MatrixXd> transform_A_;
    std::vector<Eigen::VectorXd> transform_b_;
    AABBTree                     aabb_;
};


// ------------------------------------------------------------------
//  Tree-vs-tree collision pairs (dual traversal + narrow phase)
// ------------------------------------------------------------------

/// Each tree's tau is folded into the covariances so differing scales are handled exactly.
inline std::vector<std::pair<int, int>> collision_pairs( const EllipsoidTree& A, const EllipsoidTree& B )
{
    const double sa = A.tau() * A.tau();
    const double sb = B.tau() * B.tau();
    return detail::collect_pairs(A.tree(), B.tree(), [&]( int ii, int jj )
    {
        Ellipsoid Ea{A.object(ii).mu, sa * A.object(ii).Sigma};
        Ellipsoid Eb{B.object(jj).mu, sb * B.object(jj).Sigma};
        return intersects(Ea, Eb, 1.0);
    });
}

/// Every intersecting simplex-ellipsoid pair between the two families, found in one simultaneous descent of both trees.
inline std::vector<std::pair<int, int>> collision_pairs( const SimplexTree& A, const EllipsoidTree& B )
{
    const double t2 = B.tau() * B.tau();
    return detail::collect_pairs(A.tree(), B.tree(), [&]( int ii, int jj )
    {
        return closest_point_in_simplex(B.object(jj).mu, A.object(ii).V,
                                        B.sigma_inverse(jj)).distance_squared <= t2;
    });
}

/// Every intersecting ball-ellipsoid pair between the two families, found in one simultaneous descent of both trees.
inline std::vector<std::pair<int, int>> collision_pairs( const BallTree& A, const EllipsoidTree& B )
{
    return detail::collect_pairs(A.tree(), B.tree(), [&]( int ii, int jj )
    { return intersects(A.object(ii), B.object(jj), B.tau()); });
}

/// Every intersecting box-ellipsoid pair between the two families, found in one simultaneous descent of both trees.
inline std::vector<std::pair<int, int>> collision_pairs( const BoxTree& A, const EllipsoidTree& B )
{
    const double t2 = B.tau() * B.tau();
    return detail::collect_pairs(A.tree(), B.tree(), [&]( int ii, int jj )
    {
        BoxClosestPointOptions opts;
        opts.stop_below = t2;
        return closest_point_in_box(B.object(jj).mu, A.object(ii),
                                    B.sigma_inverse(jj), opts).distance_squared <= t2;
    });
}

/// Every intersecting box-box pair between the two families, found in one simultaneous descent of both trees.
inline std::vector<std::pair<int, int>> collision_pairs( const BoxTree& A, const BoxTree& B )
{
    return detail::collect_pairs(A.tree(), B.tree(), [&]( int ii, int jj )
    { return intersects(A.object(ii), B.object(jj)); });
}

/// Every intersecting ball-ball pair between the two families, found in one simultaneous descent of both trees.
inline std::vector<std::pair<int, int>> collision_pairs( const BallTree& A, const BallTree& B )
{
    return detail::collect_pairs(A.tree(), B.tree(), [&]( int ii, int jj )
    { return intersects(A.object(ii), B.object(jj)); });
}

/// Every intersecting box-ball pair between the two families, found in one simultaneous descent of both trees.
inline std::vector<std::pair<int, int>> collision_pairs( const BoxTree& A, const BallTree& B )
{
    return detail::collect_pairs(A.tree(), B.tree(), [&]( int ii, int jj )
    { return intersects(A.object(ii), B.object(jj)); });
}

/// Every intersecting ball-simplex pair between the two families, found in one simultaneous descent of both trees.
inline std::vector<std::pair<int, int>> collision_pairs( const BallTree& A, const SimplexTree& B )
{
    return detail::collect_pairs(A.tree(), B.tree(), [&]( int ii, int jj )
    { return intersects(A.object(ii), B.object(jj)); });
}

/// Every intersecting simplex-simplex pair between the two families, found in one simultaneous descent of both trees.
inline std::vector<std::pair<int, int>> collision_pairs( const SimplexTree& A, const SimplexTree& B )
{
    return detail::collect_pairs(A.tree(), B.tree(), [&]( int ii, int jj )
    { return intersects(A.object(ii), B.object(jj)); });
}

/// Every intersecting box-simplex pair between the two families, found in one simultaneous descent of both trees.
inline std::vector<std::pair<int, int>> collision_pairs( const BoxTree& A, const SimplexTree& B )
{
    return detail::collect_pairs(A.tree(), B.tree(), [&]( int ii, int jj )
    { return intersects(A.object(ii), B.object(jj)); });
}

namespace detail {

inline std::vector<std::pair<int, int>> flip_pairs( std::vector<std::pair<int, int>> pairs )
{
    for ( std::pair<int, int>& pr : pairs )
    {
        std::swap(pr.first, pr.second);
    }
    return pairs;
}

} // end namespace detail

/// Every intersecting pair between the ellipsoids and the simplices (arguments reversed).
inline std::vector<std::pair<int, int>> collision_pairs( const EllipsoidTree& A, const SimplexTree& B )
{ return detail::flip_pairs(collision_pairs(B, A)); }
/// Every intersecting pair between the ellipsoids and the balls (arguments reversed).
inline std::vector<std::pair<int, int>> collision_pairs( const EllipsoidTree& A, const BallTree& B )
{ return detail::flip_pairs(collision_pairs(B, A)); }
/// Every intersecting pair between the ellipsoids and the boxes (arguments reversed).
inline std::vector<std::pair<int, int>> collision_pairs( const EllipsoidTree& A, const BoxTree& B )
{ return detail::flip_pairs(collision_pairs(B, A)); }
/// Every intersecting pair between the balls and the boxes (arguments reversed).
inline std::vector<std::pair<int, int>> collision_pairs( const BallTree& A, const BoxTree& B )
{ return detail::flip_pairs(collision_pairs(B, A)); }
/// Every intersecting pair between the simplices and the balls (arguments reversed).
inline std::vector<std::pair<int, int>> collision_pairs( const SimplexTree& A, const BallTree& B )
{ return detail::flip_pairs(collision_pairs(B, A)); }
/// Every intersecting pair between the simplices and the boxes (arguments reversed).
inline std::vector<std::pair<int, int>> collision_pairs( const SimplexTree& A, const BoxTree& B )
{ return detail::flip_pairs(collision_pairs(B, A)); }

} // end namespace ellipsoid_tree
