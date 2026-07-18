#pragma once
// SPDX-License-Identifier: MIT
// Part of etree — https://github.com/NickAlger/ellipsoid_tree

/// @file
/// @brief The pairwise intersection table: intersects(A, B [, tau]) overloads for all
/// combinations of {point, Box, Ball, Ellipsoid, Simplex} plus the query-only
/// objects {Segment, Halfspace}, and the closest-point machinery behind the
/// solver-backed cells.
///
/// Both argument orders are provided for every pair.
///
/// Semantics (see geometry.hpp): all objects are solid and closed, so touching
/// counts as intersecting. Points are plain Eigen vectors. Every function
/// consuming an Ellipsoid takes the scale tau at call time; a pair of
/// ellipsoids shares a single common tau (for heterogeneous scales, fold the
/// scale into Sigma). Cells backed by iterative solvers (ellipsoid-ellipsoid,
/// ellipsoid-box) are exact up to solver tolerance: verdicts may go either way
/// for configurations within ~1e-8 of exact tangency.
///
/// The ellipsoid-ellipsoid test follows:
///   I. Gilitschenski and U. D. Hanebeck, "A Direct Method for Checking
///   Overlap of Two Hyperellipsoids", Proc. 2014 Sensor Data Fusion: Trends,
///   Solutions, Applications (SDF), 2014.
/// The simplex-simplex and box-simplex cells are decided by a small phase-I
/// LP (detail/linear_feasibility.hpp): do the two convex hulls share a point.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "etree/geometry.hpp"
#include "etree/detail/minimize_scalar.hpp"
#include "etree/detail/linear_feasibility.hpp"

namespace etree {

// ------------------------------------------------------------------
//  Closest-point machinery (the solver-backed narrow phase)
// ------------------------------------------------------------------

/// Result of a closest-point query: the closest point and its squared distance.
struct ClosestPointResult
{
    Eigen::VectorXd point; ///< The closest point found.
    double          distance_squared; ///< Squared distance from the query point to that closest point.
};

/// Closest point to p in the solid simplex conv(V), in the metric
/// ||y||_M^2 = y^T M y with M symmetric positive definite.
///
/// Exact up to roundoff: enumerates all faces (vertex subsets), M-projects p
/// onto each face's affine hull, and returns the best projection that lands
/// inside its face. The minimizer lies in the relative interior of some face,
/// where the projection onto that face's hull is both feasible and optimal, so
/// the scan always finds it; affinely-degenerate vertex configurations are
/// covered by their nondegenerate subsets. Cost is O(2^K) small solves for K
/// vertices — intended for K <= ~10.
inline ClosestPointResult
closest_point_in_simplex( const Eigen::Ref<const Eigen::VectorXd>& p,
                          const Eigen::Ref<const Eigen::MatrixXd>& V,
                          const Eigen::Ref<const Eigen::MatrixXd>& M )
{
    const int dim = V.rows();
    const int K   = V.cols();
    if ( K < 1 )
    {
        throw std::invalid_argument("etree::closest_point_in_simplex: simplex has no vertices");
    }
    if ( K > 30 )
    {
        throw std::invalid_argument("etree::closest_point_in_simplex: too many vertices (cost is 2^K)");
    }

    // Center the vertices at p: for affine coordinates alpha (sum = 1),
    // x - p = (V - p 1^T) alpha, so the squared M-distance is alpha^T Gc alpha
    // with the centered Gram matrix Gc below. This avoids cancellation.
    Eigen::MatrixXd Vc = V.colwise() - p;
    Eigen::MatrixXd Gc = Vc.transpose() * (M * Vc); // (K, K), positive semidefinite

    double           best_dsq = std::numeric_limits<double>::infinity();
    Eigen::VectorXd  best_alpha;
    std::vector<int> best_idx;

    std::vector<int> idx;
    idx.reserve(K);
    for ( unsigned int mask = 1; mask < (1u << K); ++mask )
    {
        idx.clear();
        for ( int jj = 0; jj < K; ++jj )
        {
            if ( mask & (1u << jj) )
            {
                idx.push_back(jj);
            }
        }
        const int j0 = idx[0];
        const int m  = static_cast<int>(idx.size()) - 1;

        double          dsq;
        Eigen::VectorXd alpha(m + 1);
        if ( m == 0 )
        {
            alpha(0) = 1.0;
            dsq = Gc(j0, j0);
        }
        else
        {
            // Unconstrained M-projection onto the affine hull of the subset:
            // solve H a = rhs, where H is the M-Gram matrix of the edge
            // vectors v_ji - v_j0 and rhs_i = -<v_ji - v_j0, v_j0 - p>_M.
            Eigen::MatrixXd H(m, m);
            Eigen::VectorXd rhs(m);
            for ( int ii = 0; ii < m; ++ii )
            {
                const int ji = idx[ii + 1];
                rhs(ii) = Gc(j0, j0) - Gc(ji, j0);
                for ( int kk = 0; kk < m; ++kk )
                {
                    const int jk = idx[kk + 1];
                    H(ii, kk) = Gc(ji, jk) - Gc(ji, j0) - Gc(j0, jk) + Gc(j0, j0);
                }
            }
            Eigen::VectorXd a = H.ldlt().solve(rhs);

            const double a_sum = a.sum();
            if ( !(a.array() >= 0.0).all() || !(a_sum <= 1.0) )
            {
                continue; // lands outside this face; a subface covers the optimum
            }
            alpha(0) = 1.0 - a_sum;
            alpha.tail(m) = a;
            dsq = Gc(j0, j0) - rhs.dot(a);
        }

        if ( dsq < best_dsq )
        {
            best_dsq   = dsq;
            best_alpha = alpha;
            best_idx   = idx;
        }
    }

    Eigen::VectorXd point = Eigen::VectorXd::Zero(dim);
    for ( int ii = 0; ii < static_cast<int>(best_idx.size()); ++ii )
    {
        point += best_alpha(ii) * V.col(best_idx[ii]);
    }
    // Recompute the distance from the reconstructed point: the Gram-form value
    // best_dsq suffers big-minus-big cancellation (~1e-16 absolute noise),
    // which would swamp tiny distances; the direct form is accurate near zero.
    Eigen::VectorXd y = point - p;
    return ClosestPointResult{point, y.dot(M * y)};
}

/// Euclidean special case (M = I).
inline ClosestPointResult
closest_point_in_simplex( const Eigen::Ref<const Eigen::VectorXd>& p,
                          const Eigen::Ref<const Eigen::MatrixXd>& V )
{
    return closest_point_in_simplex(p, V, Eigen::MatrixXd::Identity(V.rows(), V.rows()));
}

/// Options controlling the iterative (M-metric) closest_point_in_box solve.
struct BoxClosestPointOptions
{
    double tol        = 1.0e-12; ///< convergence: max coordinate step, relative to box size
    int    max_sweeps = 256; ///< Maximum number of coordinate-descent sweeps.
    double stop_below = -std::numeric_limits<double>::infinity();
    ///< Early exit: return once distance_squared <= stop_below. The reported
    ///< distance is always an achieved (upper-bound) value, so early "close
    ///< enough" verdicts are certificates.
};

/// Closest point to p in the box, Euclidean metric: clamping, exact.
inline ClosestPointResult
closest_point_in_box( const Eigen::Ref<const Eigen::VectorXd>& p, const Box& B )
{
    Eigen::VectorXd x = p.cwiseMax(B.lo).cwiseMin(B.hi);
    return ClosestPointResult{x, (x - p).squaredNorm()};
}

/// Closest point to p in the box, in the metric ||y||_M^2 = y^T M y with M
/// symmetric positive definite. Projected cyclic coordinate descent, which
/// converges to the unique minimizer for box constraints; the returned
/// distance_squared is an achieved value (an upper bound on the true minimum
/// that converges to it).
inline ClosestPointResult
closest_point_in_box( const Eigen::Ref<const Eigen::VectorXd>& p,
                      const Box&                               B,
                      const Eigen::Ref<const Eigen::MatrixXd>& M,
                      BoxClosestPointOptions                   opts = BoxClosestPointOptions() )
{
    const int dim = p.size();

    Eigen::VectorXd x = p.cwiseMax(B.lo).cwiseMin(B.hi); // warm start
    Eigen::VectorXd r = M * (x - p);                     // half-gradient
    double          f = (x - p).dot(r);

    const double step_scale = 1.0 + (B.hi - B.lo).cwiseAbs().maxCoeff();

    for ( int sweep = 0; sweep < opts.max_sweeps; ++sweep )
    {
        if ( f <= opts.stop_below )
        {
            break;
        }

        double max_step = 0.0;
        for ( int kk = 0; kk < dim; ++kk )
        {
            const double mkk = M(kk, kk);
            if ( mkk <= 0.0 )
            {
                continue; // cannot happen for SPD M
            }
            double x_new = x(kk) - r(kk) / mkk;
            x_new = std::min(std::max(x_new, B.lo(kk)), B.hi(kk));
            const double step = x_new - x(kk);
            if ( step != 0.0 )
            {
                r += step * M.col(kk);
                x(kk) = x_new;
                max_step = std::max(max_step, std::abs(step));
            }
        }

        r = M * (x - p); // refresh to prevent incremental drift
        f = (x - p).dot(r);

        if ( max_step <= opts.tol * step_scale )
        {
            break;
        }
    }
    return ClosestPointResult{x, std::max(f, 0.0)};
}


namespace detail {

inline Eigen::MatrixXd inverse_spd( const Eigen::Ref<const Eigen::MatrixXd>& Sigma )
{
    const int dim = Sigma.rows();
    Eigen::MatrixXd Minv = Sigma.ldlt().solve(Eigen::MatrixXd::Identity(dim, dim));
    return 0.5 * (Minv + Minv.transpose()); // enforce exact symmetry
}

// Containment tolerance for measure-zero membership questions (a point lying
// on a lower-dimensional simplex), relative to the data scale.
inline double membership_tolerance( const Eigen::Ref<const Eigen::MatrixXd>& V,
                                    const Eigen::Ref<const Eigen::VectorXd>& p )
{
    return 1.0e-9 * (1.0 + V.cwiseAbs().maxCoeff() + p.cwiseAbs().maxCoeff());
}

// min over t in [0, 1] of (a + t (b - a) - mu)^T Minv (a + t (b - a) - mu),
// for callers that hold a precomputed Minv = Sigma^{-1}.
inline double segment_mahalanobis_min( const Eigen::Ref<const Eigen::VectorXd>& a,
                                       const Eigen::Ref<const Eigen::VectorXd>& b,
                                       const Eigen::Ref<const Eigen::VectorXd>& mu,
                                       const Eigen::Ref<const Eigen::MatrixXd>& Minv )
{
    const Eigen::VectorXd w  = a - mu;
    const Eigen::VectorXd u  = b - a;
    const Eigen::VectorXd zw = Minv * w;
    const double c0 = w.dot(zw);
    const double c1 = u.dot(zw);
    const double c2 = u.dot(Minv * u);
    double t = 0.0;
    if ( c2 > 0.0 )
    {
        t = std::min(1.0, std::max(0.0, -c1 / c2));
    }
    return c0 + 2.0 * c1 * t + c2 * t * t;
}

// Is alpha_a + t * alpha_u componentwise nonnegative for some t in [0, 1]?
// (Affine coordinates along a segment are linear in t; membership in a
// full-dimensional simplex is an interval intersection.)
inline bool segment_affine_feasible( const Eigen::Ref<const Eigen::VectorXd>& alpha_a,
                                     const Eigen::Ref<const Eigen::VectorXd>& alpha_u )
{
    double t_lo = 0.0;
    double t_hi = 1.0;
    for ( int ii = 0; ii < alpha_a.size(); ++ii )
    {
        const double c = alpha_a(ii);
        const double u = alpha_u(ii);
        if ( u == 0.0 )
        {
            if ( c < 0.0 )
            {
                return false;
            }
        }
        else if ( u > 0.0 )
        {
            t_lo = std::max(t_lo, -c / u);
        }
        else
        {
            t_hi = std::min(t_hi, -c / u);
        }
        if ( t_lo > t_hi )
        {
            return false;
        }
    }
    return true;
}

} // end namespace detail


// ------------------------------------------------------------------
//  Closed-form cells
// ------------------------------------------------------------------

inline bool intersects( const Eigen::Ref<const Eigen::VectorXd>& p, const Box& B )
{
    return (B.lo.array() <= p.array()).all() && (p.array() <= B.hi.array()).all();
}

inline bool intersects( const Eigen::Ref<const Eigen::VectorXd>& p, const Ball& B )
{
    return (p - B.center).squaredNorm() <= B.radius * B.radius;
}

inline bool intersects( const Eigen::Ref<const Eigen::VectorXd>& p, const Ellipsoid& E, double tau )
{
    Eigen::VectorXd z = p - E.mu;
    return E.Sigma.ldlt().solve(z).dot(z) <= tau * tau;
}

inline bool intersects( const Eigen::Ref<const Eigen::VectorXd>& p, const Halfspace& H )
{
    return H.normal.dot(p) <= H.offset;
}

inline bool intersects( const Box& A, const Box& B )
{
    return (A.lo.array() <= B.hi.array()).all() && (B.lo.array() <= A.hi.array()).all();
}

inline bool intersects( const Box& A, const Ball& B )
{
    Eigen::VectorXd closest = B.center.cwiseMax(A.lo).cwiseMin(A.hi);
    return (closest - B.center).squaredNorm() <= B.radius * B.radius;
}

inline bool intersects( const Box& A, const Halfspace& H )
{
    double support = 0.0; // min over the box of normal . x
    for ( int kk = 0; kk < A.lo.size(); ++kk )
    {
        support += std::min(H.normal(kk) * A.lo(kk), H.normal(kk) * A.hi(kk));
    }
    return support <= H.offset;
}

inline bool intersects( const Ball& A, const Ball& B )
{
    const double r = A.radius + B.radius;
    return (A.center - B.center).squaredNorm() <= r * r;
}

inline bool intersects( const Ball& A, const Halfspace& H )
{
    return H.normal.dot(A.center) - A.radius * H.normal.norm() <= H.offset;
}

inline bool intersects( const Ellipsoid& E, const Halfspace& H, double tau )
{
    // min over E(tau) of normal . x = normal . mu - tau * sqrt(normal^T Sigma normal)
    const double spread = std::sqrt(H.normal.dot(E.Sigma * H.normal));
    return H.normal.dot(E.mu) - tau * spread <= H.offset;
}

inline bool intersects( const Simplex& S, const Halfspace& H )
{
    return (H.normal.transpose() * S.V).minCoeff() <= H.offset;
}

inline bool intersects( const Segment& S, const Box& B )
{
    // Slab method on x(t) = a + t (b - a), t in [0, 1].
    double t_lo = 0.0;
    double t_hi = 1.0;
    for ( int kk = 0; kk < S.a.size(); ++kk )
    {
        const double u = S.b(kk) - S.a(kk);
        if ( u == 0.0 )
        {
            if ( S.a(kk) < B.lo(kk) || S.a(kk) > B.hi(kk) )
            {
                return false;
            }
        }
        else
        {
            double t0 = (B.lo(kk) - S.a(kk)) / u;
            double t1 = (B.hi(kk) - S.a(kk)) / u;
            if ( t0 > t1 )
            {
                std::swap(t0, t1);
            }
            t_lo = std::max(t_lo, t0);
            t_hi = std::min(t_hi, t1);
            if ( t_lo > t_hi )
            {
                return false;
            }
        }
    }
    return true;
}

inline bool intersects( const Segment& S, const Ball& B )
{
    const Eigen::VectorXd u  = S.b - S.a;
    const double          uu = u.squaredNorm();
    double t = 0.0;
    if ( uu > 0.0 )
    {
        t = std::min(1.0, std::max(0.0, u.dot(B.center - S.a) / uu));
    }
    return (S.a + t * u - B.center).squaredNorm() <= B.radius * B.radius;
}

inline bool intersects( const Segment& S, const Ellipsoid& E, double tau )
{
    // q(t) = (a + t u - mu)^T Sigma^{-1} (a + t u - mu) is a convex quadratic;
    // minimize over t in [0, 1] in closed form.
    const Eigen::VectorXd w = S.a - E.mu;
    const Eigen::VectorXd u = S.b - S.a;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(E.Sigma);
    const Eigen::VectorXd zw = ldlt.solve(w);
    const Eigen::VectorXd zu = ldlt.solve(u);
    const double c0 = w.dot(zw);
    const double c1 = u.dot(zw);
    const double c2 = u.dot(zu);
    double t = 0.0;
    if ( c2 > 0.0 )
    {
        t = std::min(1.0, std::max(0.0, -c1 / c2));
    }
    return c0 + 2.0 * c1 * t + c2 * t * t <= tau * tau;
}


// ------------------------------------------------------------------
//  Simplex membership cells
// ------------------------------------------------------------------

/// Point in simplex. For a full-dimensional simplex (dim+1 vertices) this is
/// the exact barycentric test; a degenerate (zero-volume) full-dimensional
/// simplex yields false. For a lower-dimensional simplex, membership is a
/// measure-zero question and is decided within a small relative tolerance.
inline bool intersects( const Eigen::Ref<const Eigen::VectorXd>& p, const Simplex& S )
{
    const int dim = S.V.rows();
    const int K   = S.V.cols();

    if ( K == dim + 1 )
    {
        Eigen::MatrixXd Mb(dim + 1, dim + 1); // [V; 1^T] alpha = [p; 1]
        Mb.topRows(dim) = S.V;
        Mb.row(dim).setOnes();
        Eigen::VectorXd rhs(dim + 1);
        rhs.head(dim) = p;
        rhs(dim) = 1.0;
        Eigen::VectorXd alpha = Mb.partialPivLu().solve(rhs);
        return (alpha.array() >= 0.0).all();
    }

    const double tol = detail::membership_tolerance(S.V, p);
    return closest_point_in_simplex(p, S.V).distance_squared <= tol * tol;
}

inline bool intersects( const Ball& B, const Simplex& S )
{
    return closest_point_in_simplex(B.center, S.V).distance_squared <= B.radius * B.radius;
}

inline bool intersects( const Segment& seg, const Simplex& S )
{
    const int dim = S.V.rows();
    const int K   = S.V.cols();

    if ( K == dim + 1 )
    {
        // Affine coords along the segment are linear in t: intersect the
        // intervals where each coordinate is nonnegative.
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> Ab = simplex_transform(S.V);
        const Eigen::VectorXd alpha_a = Ab.first * seg.a + Ab.second;
        const Eigen::VectorXd alpha_u = Ab.first * (seg.b - seg.a);
        return detail::segment_affine_feasible(alpha_a, alpha_u);
    }

    // Lower-dimensional simplex: distance from the segment to the simplex is
    // convex in t (distance to a convex set composed with an affine map), so
    // a 1D minimization is globally correct; membership within tolerance.
    const double tol  = detail::membership_tolerance(S.V, seg.a);
    const double tol2 = tol * tol;
    auto dist2 = [&](double t)
    {
        Eigen::VectorXd x = seg.a + t * (seg.b - seg.a);
        return closest_point_in_simplex(x, S.V).distance_squared;
    };
    detail::MinimizeScalarOptions opts;
    opts.stop_below = tol2;
    return detail::minimize_scalar(dist2, 0.0, 1.0, opts).f <= tol2;
}


// ------------------------------------------------------------------
//  LP-backed cells: convex hull vs convex hull
// ------------------------------------------------------------------

/// Simplices A and B intersect iff some point is a convex combination of both
/// vertex sets: exists alpha, beta >= 0 with sum(alpha) = sum(beta) = 1 and
/// V_A alpha = V_B beta — a linear feasibility problem. Works for any vertex
/// counts (arbitrary convex hulls of point sets), including lower-dimensional
/// simplices, with the LP's epsilon-tolerance semantics.
inline bool intersects( const Simplex& A, const Simplex& B )
{
    if ( !intersects(bounding_box(A), bounding_box(B)) )
    {
        return false; // cheap certificate of separation
    }
    const int d  = static_cast<int>(A.V.rows());
    const int KA = static_cast<int>(A.V.cols());
    const int KB = static_cast<int>(B.V.cols());

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(d + 2, KA + KB);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(d + 2);
    M.topLeftCorner(d, KA)  = A.V;
    M.topRightCorner(d, KB) = -B.V;
    M.row(d).head(KA).setOnes();
    M.row(d + 1).tail(KB).setOnes();
    rhs(d)     = 1.0;
    rhs(d + 1) = 1.0;
    return detail::linear_feasibility(M, rhs);
}

/// Box and simplex intersect iff exists alpha >= 0, sum(alpha) = 1, with
/// lo <= V alpha <= hi; the two-sided bound becomes equalities with slack
/// variables s, t >= 0: V alpha - s = lo, s + t = hi - lo.
inline bool intersects( const Box& A, const Simplex& S )
{
    if ( !intersects(A, bounding_box(S)) )
    {
        return false; // cheap certificate of separation
    }
    const int d = static_cast<int>(A.lo.size());
    const int K = static_cast<int>(S.V.cols());

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(2 * d + 1, K + 2 * d);
    Eigen::VectorXd rhs(2 * d + 1);
    M.topLeftCorner(d, K)          = S.V;
    M.block(0, K, d, d)            = -Eigen::MatrixXd::Identity(d, d);
    M.block(d, K, d, d)            = Eigen::MatrixXd::Identity(d, d);
    M.block(d, K + d, d, d)        = Eigen::MatrixXd::Identity(d, d);
    M.row(2 * d).head(K).setOnes();
    rhs.head(d)     = A.lo;
    rhs.segment(d, d) = A.hi - A.lo;
    rhs(2 * d)      = 1.0;
    return detail::linear_feasibility(M, rhs);
}


// ------------------------------------------------------------------
//  Ellipsoid solver-backed cells
// ------------------------------------------------------------------

/// Exact ellipsoid-ellipsoid overlap (Gilitschenski & Hanebeck 2014): with the
/// generalized eigendecomposition Sigma_A Phi = Sigma_B Phi Lambda,
/// Phi^T Sigma_B Phi = I, and v = Phi^T (mu_A - mu_B), the ellipsoids overlap
/// iff min over s in [0,1] of
///   K(s) = 1 - s(1-s)/tau^2 * sum_i v_i^2 / (1 + s (lambda_i - 1))
/// is nonnegative. Any s with K(s) < 0 certifies separation, so the
/// minimization exits early on that evidence.
inline bool intersects( const Ellipsoid& A, const Ellipsoid& B, double tau )
{
    if ( !intersects(bounding_box(A, tau), bounding_box(B, tau)) )
    {
        return false; // cheap certificate of separation
    }

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(A.Sigma, B.Sigma);
    if ( es.info() != Eigen::Success )
    {
        throw std::runtime_error(
            "etree::intersects(Ellipsoid, Ellipsoid): generalized eigensolve failed (Sigma not SPD?)");
    }
    const Eigen::VectorXd lambdas = es.eigenvalues();
    const Eigen::VectorXd v       = es.eigenvectors().transpose() * (A.mu - B.mu);

    auto K = [&](double s)
    {
        double acc = 0.0;
        for ( int ii = 0; ii < lambdas.size(); ++ii )
        {
            acc += (v(ii) * v(ii)) / (1.0 + s * (lambdas(ii) - 1.0));
        }
        return 1.0 - (s * (1.0 - s) / (tau * tau)) * acc;
    };

    detail::MinimizeScalarOptions opts;
    opts.stop_below = 0.0;
    return detail::minimize_scalar(K, 0.0, 1.0, opts).f >= 0.0;
}

inline bool intersects( const Ball& B, const Ellipsoid& E, double tau )
{
    if ( B.radius <= 0.0 )
    {
        return intersects(B.center, E, tau);
    }
    // The ball is the ellipsoid with Sigma = (radius/tau)^2 I at scale tau.
    const int dim = B.center.size();
    const double s = (B.radius / tau) * (B.radius / tau);
    Ellipsoid ball_as_ellipsoid{B.center, s * Eigen::MatrixXd::Identity(dim, dim)};
    return intersects(E, ball_as_ellipsoid, tau);
}

inline bool intersects( const Box& A, const Ellipsoid& E, double tau )
{
    if ( !intersects(A, bounding_box(E, tau)) )
    {
        return false; // cheap certificate of separation
    }
    BoxClosestPointOptions opts;
    opts.stop_below = tau * tau;
    return closest_point_in_box(E.mu, A, detail::inverse_spd(E.Sigma), opts).distance_squared
           <= tau * tau;
}

inline bool intersects( const Ellipsoid& E, const Simplex& S, double tau )
{
    if ( !intersects(bounding_box(S), bounding_box(E, tau)) )
    {
        return false; // cheap certificate of separation
    }
    return closest_point_in_simplex(E.mu, S.V, detail::inverse_spd(E.Sigma)).distance_squared
           <= tau * tau;
}


// ------------------------------------------------------------------
//  Reversed argument orders
// ------------------------------------------------------------------

inline bool intersects( const Box&  B, const Eigen::Ref<const Eigen::VectorXd>& p ) { return intersects(p, B); }
inline bool intersects( const Ball& B, const Eigen::Ref<const Eigen::VectorXd>& p ) { return intersects(p, B); }
inline bool intersects( const Ellipsoid& E, const Eigen::Ref<const Eigen::VectorXd>& p, double tau ) { return intersects(p, E, tau); }
inline bool intersects( const Simplex& S, const Eigen::Ref<const Eigen::VectorXd>& p ) { return intersects(p, S); }
inline bool intersects( const Halfspace& H, const Eigen::Ref<const Eigen::VectorXd>& p ) { return intersects(p, H); }

inline bool intersects( const Ball& B, const Box& A )                    { return intersects(A, B); }
inline bool intersects( const Ellipsoid& E, const Box& A, double tau )   { return intersects(A, E, tau); }
inline bool intersects( const Halfspace& H, const Box& A )               { return intersects(A, H); }
inline bool intersects( const Box& A, const Segment& S )                 { return intersects(S, A); }

inline bool intersects( const Ellipsoid& E, const Ball& B, double tau )  { return intersects(B, E, tau); }
inline bool intersects( const Simplex& S, const Ball& B )                { return intersects(B, S); }
inline bool intersects( const Halfspace& H, const Ball& B )              { return intersects(B, H); }
inline bool intersects( const Ball& B, const Segment& S )                { return intersects(S, B); }

inline bool intersects( const Simplex& S, const Ellipsoid& E, double tau )  { return intersects(E, S, tau); }
inline bool intersects( const Halfspace& H, const Ellipsoid& E, double tau ) { return intersects(E, H, tau); }
inline bool intersects( const Ellipsoid& E, const Segment& S, double tau )  { return intersects(S, E, tau); }

inline bool intersects( const Halfspace& H, const Simplex& S )           { return intersects(S, H); }
inline bool intersects( const Simplex& S, const Segment& seg )           { return intersects(seg, S); }
inline bool intersects( const Simplex& S, const Box& A )                 { return intersects(A, S); }

} // end namespace etree
