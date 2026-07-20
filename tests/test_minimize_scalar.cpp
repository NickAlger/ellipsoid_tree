// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "ellipsoid_tree/detail/minimize_scalar.hpp"

#include <cmath>
#include <vector>

using ellipsoid_tree::detail::minimize_scalar;
using ellipsoid_tree::detail::MinimizeScalarOptions;
using ellipsoid_tree::detail::MinimizeScalarResult;

TEST_CASE("quadratic minimum is found precisely")
{
    auto f = [](double x) { return (x - 3.0) * (x - 3.0) + 5.0; };
    MinimizeScalarResult res = minimize_scalar(f, 0.0, 10.0);
    CHECK(res.converged);
    CHECK(std::abs(res.x - 3.0) < 1e-7);
    CHECK(std::abs(res.f - 5.0) < 1e-12);
    CHECK(res.evaluations == res.iterations + 1);
}

TEST_CASE("shifted quadratics across the interval")
{
    for ( double r = 0.05; r < 1.0; r += 0.1 )
    {
        auto f = [r](double x) { return 2.5 * (x - r) * (x - r) - 1.0; };
        MinimizeScalarResult res = minimize_scalar(f, 0.0, 1.0);
        CHECK(res.converged);
        CHECK(std::abs(res.x - r) < 1e-7);
    }
}

TEST_CASE("quartic (flat) minimum converges in x")
{
    auto f = [](double x) { double y = x - 1.0; return y * y * y * y; };
    MinimizeScalarResult res = minimize_scalar(f, -4.0, 3.0);
    CHECK(res.converged);
    CHECK(std::abs(res.x - 1.0) < 1e-5);
}

TEST_CASE("minimum at interval boundary")
{
    MinimizeScalarResult res_left = minimize_scalar([](double x) { return x; }, 0.0, 1.0);
    CHECK(res_left.converged);
    CHECK(res_left.x < 1e-6);

    MinimizeScalarResult res_right = minimize_scalar([](double x) { return -x; }, 0.0, 1.0);
    CHECK(res_right.converged);
    CHECK(res_right.x > 1.0 - 1e-6);
}

TEST_CASE("nonsmooth absolute value")
{
    auto f = [](double x) { return std::abs(x - 0.7); };
    MinimizeScalarResult res = minimize_scalar(f, 0.0, 2.0);
    CHECK(res.converged);
    CHECK(std::abs(res.x - 0.7) < 1e-6);
}

TEST_CASE("cosine on [0, 2 pi]")
{
    MinimizeScalarResult res = minimize_scalar([](double x) { return std::cos(x); },
                                               0.0, 6.283185307179586);
    CHECK(res.converged);
    CHECK(std::abs(res.x - 3.141592653589793) < 1e-6);
}

TEST_CASE("reversed bounds are accepted")
{
    auto f = [](double x) { return (x - 0.25) * (x - 0.25); };
    MinimizeScalarResult res = minimize_scalar(f, 1.0, 0.0);
    CHECK(res.converged);
    CHECK(std::abs(res.x - 0.25) < 1e-7);
}

TEST_CASE("max_iterations is respected")
{
    auto f = [](double x) { double y = x - 1.0; return y * y * y * y; };
    MinimizeScalarOptions opts;
    opts.max_iterations = 5;
    MinimizeScalarResult res = minimize_scalar(f, -4.0, 3.0, opts);
    CHECK(!res.converged);
    CHECK(res.iterations == 5);
    CHECK(res.evaluations == 6);
}

TEST_CASE("stop_below triggers early exit")
{
    auto f = [](double x) { return (x - 0.5) * (x - 0.5) - 1.0; };
    MinimizeScalarOptions opts;
    opts.stop_below = -0.5;
    MinimizeScalarResult res = minimize_scalar(f, 0.0, 1.0, opts);
    CHECK(res.stopped_early);
    CHECK(res.f < -0.5);
    CHECK(res.evaluations < 20);

    // Without the threshold the same problem runs to convergence.
    MinimizeScalarResult full = minimize_scalar(f, 0.0, 1.0);
    CHECK(full.converged);
    CHECK(!full.stopped_early);
    CHECK(std::abs(full.f - (-1.0)) < 1e-12);
}

// The shape of function minimized inside the ellipsoid-ellipsoid overlap test
// (Gilitschenski & Hanebeck): K(s) = 1 - s(1-s)/tau^2 * sum_i v_i^2 / (1 + s (lambda_i - 1)).
TEST_CASE("overlap-test K functions match brute-force grid minimum")
{
    struct KCase { std::vector<double> lambdas; std::vector<double> v; double tau; };
    std::vector<KCase> cases = {
        { {0.5, 2.0, 4.0},        {0.3, -0.2, 0.7},       1.0 },
        { {1.0, 1.0},             {1.5, -0.4},            1.0 },
        { {0.01, 100.0},          {0.9, 0.02},            0.5 },
        { {3.0, 0.2, 0.7, 1.4},   {-1.0, 0.3, 0.0, 2.0},  2.0 },
    };

    for ( const KCase& kc : cases )
    {
        auto K = [&kc](double s)
        {
            double acc = 0.0;
            for ( size_t ii = 0; ii < kc.lambdas.size(); ++ii )
            {
                acc += (kc.v[ii] * kc.v[ii]) / (1.0 + s * (kc.lambdas[ii] - 1.0));
            }
            return 1.0 - (s * (1.0 - s) / (kc.tau * kc.tau)) * acc;
        };

        double grid_min = K(0.0);
        const int num_grid = 100000;
        for ( int ii = 1; ii <= num_grid; ++ii )
        {
            double s = static_cast<double>(ii) / num_grid;
            grid_min = std::min(grid_min, K(s));
        }

        MinimizeScalarResult res = minimize_scalar(K, 0.0, 1.0);
        CHECK(res.converged);

        // The minimizer must do at least as well as the grid; the grid value
        // itself is only O(K'' h^2)-accurate, so allow slack in that direction.
        CHECK(res.f <= grid_min + 1e-9);
        CHECK(res.f >= grid_min - 1e-3);

        // Local-minimum certificate: nearby points are no better.
        for ( double delta : {1e-6, 1e-4} )
        {
            for ( double sign : {-1.0, 1.0} )
            {
                double s_nearby = std::min(1.0, std::max(0.0, res.x + sign * delta));
                CHECK(K(s_nearby) >= res.f - 1e-9);
            }
        }
    }
}
