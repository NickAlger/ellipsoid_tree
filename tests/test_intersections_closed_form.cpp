// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "etree/intersections.hpp"
#include "test_helpers.hpp"

#include <random>

using namespace etree;
namespace th = test_helpers;

TEST_CASE("point-box and point-ball")
{
    std::mt19937 gen(11);
    for ( int rep = 0; rep < 50; ++rep )
    {
        const int d = 1 + (rep % 4);
        Eigen::VectorXd c  = th::randn_vector(d, gen);
        Eigen::VectorXd w  = th::randn_vector(d, gen).cwiseAbs() + Eigen::VectorXd::Constant(d, 0.1);
        Box B{c - w, c + w};

        // Interior point by construction
        Eigen::VectorXd t = th::randn_vector(d, gen).unaryExpr(
            [](double z) { return std::tanh(z); }); // in (-1, 1)
        Eigen::VectorXd p_in = c + w.cwiseProduct(t);
        CHECK(intersects(p_in, B));

        // Pushed out along one coordinate
        Eigen::VectorXd p_out = p_in;
        const int kk = rep % d;
        p_out(kk) = c(kk) + 1.5 * w(kk);
        CHECK(!intersects(p_out, B));

        Ball ball{c, w.minCoeff()};
        CHECK(intersects(c, ball));
        Eigen::VectorXd dir = th::randn_vector(d, gen);
        dir /= dir.norm();
        CHECK(intersects(Eigen::VectorXd(c + 0.99 * ball.radius * dir), ball));
        CHECK(!intersects(Eigen::VectorXd(c + 1.01 * ball.radius * dir), ball));
    }
}

TEST_CASE("box-box overlap and separation")
{
    std::mt19937 gen(22);
    for ( int rep = 0; rep < 50; ++rep )
    {
        const int d = 1 + (rep % 4);
        Eigen::VectorXd cA = th::randn_vector(d, gen);
        Eigen::VectorXd wA = th::randn_vector(d, gen).cwiseAbs() + Eigen::VectorXd::Constant(d, 0.1);
        Box A{cA - wA, cA + wA};

        // B sharing the point cA (in both) -> overlap
        Eigen::VectorXd wB = th::randn_vector(d, gen).cwiseAbs() + Eigen::VectorXd::Constant(d, 0.1);
        Eigen::VectorXd cB = cA + 0.9 * wB.cwiseProduct(th::randn_vector(d, gen).unaryExpr(
            [](double z) { return std::tanh(z); }));
        Box B{cB - wB, cB + wB};
        CHECK(intersects(A, B));
        CHECK(intersects(B, A));

        // Shift B beyond the combined half-widths along one axis -> disjoint
        const int kk = rep % d;
        Box B2 = B;
        const double shift = (A.hi(kk) - B2.lo(kk)) + 0.01;
        B2.lo(kk) += shift;
        B2.hi(kk) += shift;
        CHECK(!intersects(A, B2));

        // Exactly touching faces count as intersecting
        Box B3 = B2;
        B3.lo(kk) = A.hi(kk);
        B3.hi(kk) = A.hi(kk) + (B2.hi(kk) - B2.lo(kk));
        CHECK(intersects(A, B3));
    }
}

TEST_CASE("box-ball at corners and faces")
{
    Box B{Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(1.0, 1.0)};
    const double dist_to_corner = std::sqrt(2.0) * 0.5;

    Ball touching{Eigen::Vector2d(1.5, 1.5), dist_to_corner * 1.001};
    CHECK(intersects(B, touching));
    CHECK(intersects(touching, B));

    Ball missing{Eigen::Vector2d(1.5, 1.5), dist_to_corner * 0.999};
    CHECK(!intersects(B, missing));

    Ball inside{Eigen::Vector2d(0.5, 0.5), 0.01};
    CHECK(intersects(B, inside)); // containment, no boundary crossing

    Ball engulfing{Eigen::Vector2d(0.5, 0.5), 10.0};
    CHECK(intersects(B, engulfing));
}

TEST_CASE("ball-ball threshold")
{
    std::mt19937 gen(33);
    for ( int rep = 0; rep < 50; ++rep )
    {
        const int d = 1 + (rep % 5);
        Ball A{th::randn_vector(d, gen), 0.3 + 0.1 * (rep % 7)};
        double rB = 0.2 + 0.05 * (rep % 9);
        Eigen::VectorXd dir = th::randn_vector(d, gen);
        dir /= dir.norm();
        const double R = A.radius + rB;
        CHECK(intersects(A, Ball{A.center + 0.99 * R * dir, rB}));
        CHECK(!intersects(A, Ball{A.center + 1.01 * R * dir, rB}));
    }
}

TEST_CASE("halfspace support against brute force")
{
    std::mt19937 gen(44);
    for ( int rep = 0; rep < 20; ++rep )
    {
        const int d = 2 + (rep % 2);
        Eigen::VectorXd n = th::randn_vector(d, gen);

        // Box: corners enumerated exactly by a 2-point-per-axis grid
        Eigen::VectorXd c = th::randn_vector(d, gen);
        Eigen::VectorXd w = th::randn_vector(d, gen).cwiseAbs() + Eigen::VectorXd::Constant(d, 0.1);
        Box B{c - w, c + w};
        double sB = th::grid_min_over_box([&](const Eigen::VectorXd& x) { return n.dot(x); },
                                          B.lo, B.hi, 2);
        CHECK(intersects(B, Halfspace{n, sB + 1e-9}));
        CHECK(!intersects(B, Halfspace{n, sB - 1e-9}));

        // Simplex: support attained at a vertex
        Simplex S{th::randn_matrix(d, d + 1, gen)};
        double sS = (n.transpose() * S.V).minCoeff();
        CHECK(intersects(S, Halfspace{n, sS + 1e-9}));
        CHECK(!intersects(S, Halfspace{n, sS - 1e-9}));

        // Ball: sampled boundary support with generous margin
        Ball ball{c, 0.5 + 0.1 * rep};
        double sBall = std::numeric_limits<double>::infinity();
        for ( int ss = 0; ss < 20000; ++ss )
        {
            Eigen::VectorXd u = th::randn_vector(d, gen);
            u /= u.norm();
            sBall = std::min(sBall, n.dot(Eigen::VectorXd(ball.center + ball.radius * u)));
        }
        const double margin_ball = 0.05 * ball.radius * n.norm();
        CHECK(intersects(ball, Halfspace{n, sBall + margin_ball}));
        CHECK(!intersects(ball, Halfspace{n, sBall - margin_ball}));

        // Ellipsoid: sampled boundary support with generous margin
        Ellipsoid E{th::randn_vector(d, gen), th::random_spd(d, gen, 0.3, 3.0)};
        const double tau = 0.8 + 0.05 * rep;
        Eigen::MatrixXd L = th::matrix_sqrt_spd(E.Sigma);
        double sE = std::numeric_limits<double>::infinity();
        double spread = 0.0;
        for ( int ss = 0; ss < 20000; ++ss )
        {
            Eigen::VectorXd u = th::randn_vector(d, gen);
            u /= u.norm();
            double val = n.dot(Eigen::VectorXd(E.mu + tau * (L * u)));
            sE = std::min(sE, val);
            spread = std::max(spread, std::abs(val - n.dot(E.mu)));
        }
        const double margin_E = 0.05 * (spread + 1e-6);
        CHECK(intersects(E, Halfspace{n, sE + margin_E}, tau));
        CHECK(!intersects(E, Halfspace{n, sE - margin_E}, tau));

        // Point
        Eigen::VectorXd p = th::randn_vector(d, gen);
        CHECK(intersects(p, Halfspace{n, n.dot(p) + 1e-12}));
        CHECK(!intersects(p, Halfspace{n, n.dot(p) - 1e-9}));
    }
}

TEST_CASE("segment-box against dense parameter sampling")
{
    std::mt19937 gen(55);
    int classified = 0;
    for ( int rep = 0; rep < 100; ++rep )
    {
        const int d = 2 + (rep % 2);
        Eigen::VectorXd c = th::randn_vector(d, gen);
        Eigen::VectorXd w = th::randn_vector(d, gen).cwiseAbs() + Eigen::VectorXd::Constant(d, 0.2);
        Box B{c - w, c + w};
        Segment S{th::randn_vector(d, gen, 2.0), th::randn_vector(d, gen, 2.0)};

        // Ground truth: distance from segment points to the box along a fine t-grid
        const int n_grid = 20000;
        double min_dist = std::numeric_limits<double>::infinity();
        for ( int ii = 0; ii <= n_grid; ++ii )
        {
            const double t = ii / static_cast<double>(n_grid);
            Eigen::VectorXd x = S.a + t * (S.b - S.a);
            Eigen::VectorXd cl = x.cwiseMax(B.lo).cwiseMin(B.hi);
            min_dist = std::min(min_dist, (x - cl).norm());
        }
        const double grid_err = (S.b - S.a).norm() / n_grid; // Lipschitz-1 in x(t)

        if ( min_dist == 0.0 )
        {
            CHECK(intersects(S, B));
            classified += 1;
        }
        else if ( min_dist > 10.0 * grid_err )
        {
            CHECK(!intersects(S, B));
            classified += 1;
        }
    }
    CHECK(classified >= 90);
}

TEST_CASE("segment-ball against dense parameter sampling")
{
    std::mt19937 gen(66);
    for ( int rep = 0; rep < 100; ++rep )
    {
        const int d = 2 + (rep % 3);
        Ball B{th::randn_vector(d, gen), 0.5 + 0.2 * (rep % 5)};
        Segment S{th::randn_vector(d, gen, 2.0), th::randn_vector(d, gen, 2.0)};

        // Ground truth: dense sampling of the segment parameter (independent
        // of the library's projection formula)
        const int n_grid = 20000;
        double min_dist = std::numeric_limits<double>::infinity();
        for ( int ii = 0; ii <= n_grid; ++ii )
        {
            const double t = ii / static_cast<double>(n_grid);
            min_dist = std::min(min_dist, (S.a + t * (S.b - S.a) - B.center).norm());
        }
        const double grid_err = (S.b - S.a).norm() / n_grid;
        if ( min_dist <= B.radius - 10.0 * grid_err )
        {
            CHECK(intersects(S, B));
        }
        else if ( min_dist >= B.radius + 10.0 * grid_err )
        {
            CHECK(!intersects(S, B));
        }

        // Degenerate segment (a == b) behaves like a point
        Segment P{S.a, S.a};
        CHECK(intersects(P, B) == intersects(S.a, B));
    }
}
