// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "etree/intersections.hpp"
#include "test_helpers.hpp"

#include <random>

using namespace etree;
namespace th = test_helpers;

namespace {

// Support interval of an object along direction n: [min n.x, max n.x].
struct SupportInterval { double lo; double hi; };

SupportInterval support_ellipsoid( const Ellipsoid& E, double tau, const Eigen::VectorXd& n )
{
    const double spread = tau * std::sqrt(n.dot(E.Sigma * n));
    const double mid = n.dot(E.mu);
    return {mid - spread, mid + spread};
}

SupportInterval support_ball( const Ball& B, const Eigen::VectorXd& n )
{
    const double spread = B.radius * n.norm();
    const double mid = n.dot(B.center);
    return {mid - spread, mid + spread};
}

// Certified classification of an ellipsoid-vs-X overlap question:
//   +1 = definitely intersect (witness point found in both objects)
//   -1 = definitely disjoint (bounding boxes disjoint, or a strictly
//        separating direction found)
//    0 = unknown (razor-edge case; caller skips it)
// q_other(x) <= 1 must characterize membership in the second object, and
// support_other must give its support interval along a direction.
template <class QOther, class SupportOther>
int classify_vs_ellipsoid( const Ellipsoid& A, double tau, const Box& other_bbox,
                           QOther&& q_other, SupportOther&& support_other,
                           std::mt19937& gen )
{
    const int d = A.mu.size();
    Box bA = bounding_box(A, tau);
    Eigen::VectorXd lo = bA.lo.cwiseMax(other_bbox.lo);
    Eigen::VectorXd hi = bA.hi.cwiseMin(other_bbox.hi);
    if ( !((lo.array() <= hi.array()).all()) )
    {
        return -1; // bounding boxes disjoint
    }

    // Witness search over the bbox intersection region
    auto in_both = [&](const Eigen::VectorXd& x)
    {
        return th::mahalanobis_sq(x, A.mu, A.Sigma) <= tau * tau * (1.0 - 1e-9)
               && q_other(x) <= 1.0 - 1e-9;
    };
    if ( d == 2 )
    {
        const int N = 120;
        for ( int ii = 0; ii < N; ++ii )
        {
            for ( int jj = 0; jj < N; ++jj )
            {
                Eigen::Vector2d x(lo(0) + (hi(0) - lo(0)) * ii / (N - 1.0),
                                  lo(1) + (hi(1) - lo(1)) * jj / (N - 1.0));
                if ( in_both(x) )
                {
                    return +1;
                }
            }
        }
    }
    else
    {
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        for ( int ss = 0; ss < 40000; ++ss )
        {
            Eigen::VectorXd x(d);
            for ( int kk = 0; kk < d; ++kk )
            {
                x(kk) = lo(kk) + (hi(kk) - lo(kk)) * unif(gen);
            }
            if ( in_both(x) )
            {
                return +1;
            }
        }
    }

    // Separating direction search (strict separation certifies disjointness)
    auto separated_along = [&](const Eigen::VectorXd& n)
    {
        SupportInterval sA = support_ellipsoid(A, tau, n);
        SupportInterval sO = support_other(n);
        return sA.lo > sO.hi || sO.lo > sA.hi;
    };
    if ( d == 2 )
    {
        for ( int aa = 0; aa < 3600; ++aa )
        {
            const double theta = aa * 3.141592653589793 / 3600.0;
            if ( separated_along(Eigen::Vector2d(std::cos(theta), std::sin(theta))) )
            {
                return -1;
            }
        }
    }
    else
    {
        for ( int ss = 0; ss < 20000; ++ss )
        {
            Eigen::VectorXd n = th::randn_vector(d, gen);
            if ( separated_along(n) )
            {
                return -1;
            }
        }
    }
    return 0;
}

} // end anonymous namespace

TEST_CASE("ellipsoid-ellipsoid: exact answers for proportional covariances")
{
    // With Sigma_A = r_A^2 C and Sigma_B = r_B^2 C the ellipsoids are a common
    // affine image of two balls: they intersect iff
    // || C^{-1/2} (mu_B - mu_A) || <= tau (r_A + r_B). Exact in any dimension.
    std::mt19937 gen(77);
    for ( int d : {2, 3, 6} )
    {
        for ( int rep = 0; rep < 15; ++rep )
        {
            Eigen::MatrixXd C = th::random_spd(d, gen, 0.3, 3.0);
            Eigen::MatrixXd C_half = th::matrix_sqrt_spd(C);
            const double rA = 0.5 + 0.15 * (rep % 5);
            const double rB = 0.4 + 0.2 * (rep % 4);
            const double tau = 0.7 + 0.2 * (rep % 3);

            Eigen::VectorXd muA = th::randn_vector(d, gen, 2.0);
            Eigen::VectorXd g = th::randn_vector(d, gen);
            g /= g.norm();

            Ellipsoid A{muA, rA * rA * C};
            for ( double theta : {0.5, 0.9, 0.99} )
            {
                Ellipsoid B{muA + theta * tau * (rA + rB) * (C_half * g), rB * rB * C};
                CHECK(intersects(A, B, tau));
            }
            for ( double theta : {1.01, 1.1, 2.0} )
            {
                Ellipsoid B{muA + theta * tau * (rA + rB) * (C_half * g), rB * rB * C};
                CHECK(!intersects(A, B, tau));
            }
        }
    }
}

TEST_CASE("ellipsoid-ellipsoid: certified random cases")
{
    std::mt19937 gen(88);
    int classified = 0;
    const int num_cases = 150;
    for ( int rep = 0; rep < num_cases; ++rep )
    {
        const int d = (rep % 3 == 0) ? 3 : 2;
        Ellipsoid A{th::randn_vector(d, gen, 1.2), th::random_spd(d, gen, 0.2, 3.0)};
        Ellipsoid B{th::randn_vector(d, gen, 1.2), th::random_spd(d, gen, 0.2, 3.0)};
        const double tau = 0.6 + 0.1 * (rep % 6);

        auto q_B = [&](const Eigen::VectorXd& x)
        {
            return th::mahalanobis_sq(x, B.mu, B.Sigma) / (tau * tau);
        };
        auto support_B = [&](const Eigen::VectorXd& n) { return support_ellipsoid(B, tau, n); };

        const int verdict = classify_vs_ellipsoid(A, tau, bounding_box(B, tau), q_B, support_B, gen);
        if ( verdict != 0 )
        {
            CHECK(intersects(A, B, tau) == (verdict > 0));
            classified += 1;
        }
    }
    CHECK(classified >= (3 * num_cases) / 4);
}

TEST_CASE("ellipsoid-ellipsoid: symmetry, invariance, scaling, containment")
{
    std::mt19937 gen(99);
    for ( int rep = 0; rep < 100; ++rep )
    {
        const int d = 2 + (rep % 2);
        Ellipsoid A{th::randn_vector(d, gen), th::random_spd(d, gen, 0.2, 4.0)};
        Ellipsoid B{th::randn_vector(d, gen), th::random_spd(d, gen, 0.2, 4.0)};
        const double tau = 0.5 + 0.15 * (rep % 5);

        const bool ab = intersects(A, B, tau);

        // Symmetry
        CHECK(intersects(B, A, tau) == ab);

        // Rigid motion invariance
        Eigen::MatrixXd R = th::random_rotation(d, gen);
        Eigen::VectorXd t = th::randn_vector(d, gen, 2.0);
        Ellipsoid A2{R * A.mu + t, R * A.Sigma * R.transpose()};
        Ellipsoid B2{R * B.mu + t, R * B.Sigma * R.transpose()};
        CHECK(intersects(A2, B2, tau) == ab);

        // Folding tau into the covariances
        Ellipsoid A3{A.mu, tau * tau * A.Sigma};
        Ellipsoid B3{B.mu, tau * tau * B.Sigma};
        CHECK(intersects(A3, B3, 1.0) == ab);
    }

    // Containment without boundary crossing
    std::mt19937 gen2(100);
    for ( int rep = 0; rep < 20; ++rep )
    {
        const int d = 2 + (rep % 3);
        Ellipsoid big{th::randn_vector(d, gen2), th::random_spd(d, gen2, 1.0, 4.0)};
        Ellipsoid tiny{big.mu + 0.01 * th::randn_vector(d, gen2),
                       1e-4 * th::random_spd(d, gen2, 0.5, 2.0)};
        CHECK(intersects(big, tiny, 1.0));
        CHECK(intersects(tiny, big, 1.0));
    }
}

TEST_CASE("ball-ellipsoid: certified random cases and degenerate radius")
{
    std::mt19937 gen(111);
    int classified = 0;
    const int num_cases = 100;
    for ( int rep = 0; rep < num_cases; ++rep )
    {
        const int d = (rep % 3 == 0) ? 3 : 2;
        Ellipsoid E{th::randn_vector(d, gen, 1.2), th::random_spd(d, gen, 0.2, 3.0)};
        Ball B{th::randn_vector(d, gen, 1.5), 0.3 + 0.15 * (rep % 5)};
        const double tau = 0.6 + 0.12 * (rep % 6);

        auto q_ball = [&](const Eigen::VectorXd& x)
        {
            return (x - B.center).squaredNorm() / (B.radius * B.radius);
        };
        auto support_b = [&](const Eigen::VectorXd& n) { return support_ball(B, n); };

        const int verdict = classify_vs_ellipsoid(E, tau, bounding_box(B), q_ball, support_b, gen);
        if ( verdict != 0 )
        {
            CHECK(intersects(B, E, tau) == (verdict > 0));
            CHECK(intersects(E, B, tau) == (verdict > 0));
            classified += 1;
        }
    }
    CHECK(classified >= (3 * num_cases) / 4);

    // Zero-radius ball behaves like a point
    Ellipsoid E{Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity()};
    CHECK(intersects(Ball{Eigen::Vector2d(0.5, 0.0), 0.0}, E, 1.0));
    CHECK(!intersects(Ball{Eigen::Vector2d(1.5, 0.0), 0.0}, E, 1.0));
}

TEST_CASE("segment-ellipsoid against dense parameter sampling")
{
    std::mt19937 gen(122);
    int classified = 0;
    const int num_cases = 100;
    for ( int rep = 0; rep < num_cases; ++rep )
    {
        const int d = 2 + (rep % 3);
        Ellipsoid E{th::randn_vector(d, gen), th::random_spd(d, gen, 0.2, 3.0)};
        Segment S{th::randn_vector(d, gen, 2.0), th::randn_vector(d, gen, 2.0)};
        const double tau = 0.7 + 0.1 * (rep % 5);

        const int n_grid = 20000;
        double q_min = std::numeric_limits<double>::infinity();
        for ( int ii = 0; ii <= n_grid; ++ii )
        {
            const double t = ii / static_cast<double>(n_grid);
            Eigen::VectorXd x = S.a + t * (S.b - S.a);
            q_min = std::min(q_min, th::mahalanobis_sq(x, E.mu, E.Sigma));
        }

        const double t2 = tau * tau;
        if ( q_min <= 0.98 * t2 )
        {
            CHECK(intersects(S, E, tau));
            classified += 1;
        }
        else if ( q_min >= 1.02 * t2 )
        {
            CHECK(!intersects(S, E, tau));
            classified += 1;
        }
    }
    CHECK(classified >= (9 * num_cases) / 10);
}
