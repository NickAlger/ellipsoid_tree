#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_tree — https://github.com/NickAlger/ellipsoid_tree

/// @file
/// @brief The basic geometric object types.
///
/// All objects are solid (filled) and closed: boundary points count as
/// belonging to the object, and tangency counts as intersecting.

#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

namespace ellipsoid_tree {

/// Axis-aligned box {x : lo <= x <= hi componentwise}.
struct Box
{
    Eigen::VectorXd lo; ///< Lower corner (componentwise minimum).
    Eigen::VectorXd hi; ///< Upper corner (componentwise maximum).
};

/// Ball {x : ||x - center|| <= radius}.
struct Ball
{
    Eigen::VectorXd center; ///< Center point.
    double          radius; ///< Radius (>= 0).
};

/// Ellipsoid E(tau) = {x : (x - mu)^T Sigma^{-1} (x - mu) <= tau^2}.
///
/// Sigma must be symmetric positive definite. The scale parameter tau is
/// supplied at call time by every function that consumes an Ellipsoid; it is
/// deliberately not stored here, so that a family of ellipsoids sharing a
/// covariance structure can be queried at any confidence scale. Baking a
/// scale in is always possible via Sigma <- tau^2 * Sigma.
struct Ellipsoid
{
    Eigen::VectorXd mu;    ///< Center (mean).
    Eigen::MatrixXd Sigma; ///< Symmetric positive-definite shape (covariance) matrix.
};

/// Simplex conv(V.col(0), ..., V.col(K-1)): K vertices in R^dim, K <= dim+1.
///
/// K == dim+1 gives a full-dimensional simplex; smaller K gives a
/// lower-dimensional simplex (point, segment, triangle, ...) embedded in R^dim.
struct Simplex
{
    Eigen::MatrixXd V; ///< Vertices, shape (dim, K); columns are the vertices.
};

/// Line segment from a to b (query object only; not a tree element type).
struct Segment
{
    Eigen::VectorXd a; ///< Start point.
    Eigen::VectorXd b; ///< End point.
};

/// Halfspace {x : normal . x <= offset} (query object only; not a tree element type).
struct Halfspace
{
    Eigen::VectorXd normal; ///< Boundary normal (points out of the halfspace).
    double          offset; ///< Signed offset of the boundary along the normal.
};


// ------------------------------------------------------------------
//  Bounding boxes
// ------------------------------------------------------------------

/// Degenerate bounding box of a single point (lo == hi == point).
inline Box bounding_box( const Eigen::Ref<const Eigen::VectorXd>& point )
{
    return Box{point, point};
}

/// Bounding box of a box (the box itself).
inline Box bounding_box( const Box& B )
{
    return B;
}

/// Axis-aligned bounding box of a ball.
inline Box bounding_box( const Ball& B )
{
    Eigen::VectorXd w = Eigen::VectorXd::Constant(B.center.size(), B.radius);
    return Box{B.center - w, B.center + w};
}

/// Tight axis-aligned bounding box of an ellipsoid at scale tau:
/// half-width tau * sqrt(Sigma_kk) along axis k.
inline Box bounding_box( const Ellipsoid& E, double tau )
{
    Eigen::VectorXd w = (E.Sigma.diagonal().array().sqrt() * tau).matrix();
    return Box{E.mu - w, E.mu + w};
}

/// Axis-aligned bounding box of a simplex (componentwise min/max of its vertices).
inline Box bounding_box( const Simplex& S )
{
    return Box{S.V.rowwise().minCoeff(), S.V.rowwise().maxCoeff()};
}

/// Axis-aligned bounding box of a segment.
inline Box bounding_box( const Segment& S )
{
    return Box{S.a.cwiseMin(S.b), S.a.cwiseMax(S.b)};
}


// ------------------------------------------------------------------
//  Simplex affine coordinates
// ------------------------------------------------------------------

/// Affine (barycentric) coordinate operator of a simplex.
///
/// Returns (A, b) such that alpha = A x + b are the affine coordinates of x,
/// i.e. x = V alpha with sum(alpha) = 1 whenever x lies in the affine hull of
/// the vertices. For a lower-dimensional simplex the coordinates are those of
/// the orthogonal projection of x onto the affine hull.
inline std::pair<Eigen::MatrixXd, Eigen::VectorXd>
simplex_transform( const Eigen::Ref<const Eigen::MatrixXd>& V )
{
    const int dim = V.rows();
    const int K   = V.cols();
    if ( K < 1 )
    {
        throw std::invalid_argument("ellipsoid_tree::simplex_transform: simplex has no vertices");
    }

    Eigen::MatrixXd A(K, dim);
    Eigen::VectorXd b(K);

    if ( K == 1 )
    {
        A.setZero();
        b.setOnes();
    }
    else
    {
        Eigen::MatrixXd dV(dim, K - 1);
        for ( int jj = 1; jj < K; ++jj )
        {
            dV.col(jj - 1) = V.col(jj) - V.col(0);
        }
        // Pseudo-inverse of dV: reduced coordinates of x are A_reduced (x - v_0).
        Eigen::MatrixXd A_reduced =
            dV.colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(dim, dim));

        A.bottomRows(K - 1) = A_reduced;
        A.row(0) = -Eigen::RowVectorXd::Ones(K - 1) * A_reduced;
        b.setZero();
        b(0) = 1.0;
        b -= A * V.col(0);
    }
    return std::make_pair(A, b);
}

} // end namespace ellipsoid_tree
