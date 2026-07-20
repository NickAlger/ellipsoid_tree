// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "ellipsoid_tree/intersections.hpp"
#include "test_helpers.hpp"

#include <random>

using namespace ellipsoid_tree;
namespace th = test_helpers;

TEST_CASE("point-simplex via barycentric construction")
{
    std::mt19937 gen(1001);
    for ( int rep = 0; rep < 60; ++rep )
    {
        const int d = 1 + (rep % 3);
        Simplex S{th::randn_matrix(d, d + 1, gen)};

        // Interior point: nonnegative coordinates
        Eigen::VectorXd alpha = th::random_simplex_alpha(d + 1, gen);
        CHECK(intersects(Eigen::VectorXd(S.V * alpha), S));

        // Outside point: one clearly negative coordinate, coordinates sum to 1
        Eigen::VectorXd beta = th::random_simplex_alpha(d + 1, gen);
        beta.tail(d) *= (1.2 / beta.tail(d).sum());
        beta(0) = -0.2;
        CHECK(!intersects(Eigen::VectorXd(S.V * beta), S));

        // Near a face, on either side (avoids the razor-edge exact-zero case)
        Eigen::VectorXd gamma = th::random_simplex_alpha(d + 1, gen);
        gamma.tail(d) *= ((1.0 - 1e-6) / gamma.tail(d).sum());
        gamma(0) = 1e-6;
        CHECK(intersects(Eigen::VectorXd(S.V * gamma), S));
        gamma.tail(d) *= ((1.0 + 1e-6) / gamma.tail(d).sum());
        gamma(0) = -1e-6;
        CHECK(!intersects(Eigen::VectorXd(S.V * gamma), S));
    }
}

TEST_CASE("point in lower-dimensional simplex")
{
    std::mt19937 gen(1002);
    for ( int rep = 0; rep < 20; ++rep )
    {
        // Segment (2 vertices) embedded in R^3
        Eigen::MatrixXd V = th::randn_matrix(3, 2, gen);
        Simplex S{V};
        Eigen::VectorXd mid = 0.5 * (V.col(0) + V.col(1));
        CHECK(intersects(mid, S));
        CHECK(intersects(Eigen::VectorXd(V.col(0)), S)); // endpoint

        // Slightly off the segment: outside
        Eigen::VectorXd u = V.col(1) - V.col(0);
        Eigen::VectorXd n = th::randn_vector(3, gen);
        n -= (n.dot(u) / u.squaredNorm()) * u; // orthogonal to the segment
        n /= n.norm();
        CHECK(!intersects(Eigen::VectorXd(mid + 1e-3 * n), S));
        CHECK(intersects(Eigen::VectorXd(mid + 1e-12 * n), S)); // within tolerance

        // Beyond an endpoint along the segment direction: outside
        CHECK(!intersects(Eigen::VectorXd(V.col(1) + 0.01 * u), S));
    }
}

TEST_CASE("closest_point_in_simplex: analytic triangle cases in 3D")
{
    Eigen::MatrixXd V(3, 3);
    V << 0.0, 1.0, 0.0,
         0.0, 0.0, 1.0,
         0.0, 0.0, 0.0; // triangle in the z = 0 plane

    struct Case { Eigen::Vector3d p; Eigen::Vector3d expected; double dsq; };
    std::vector<Case> cases = {
        {{0.2, 0.3, 1.0}, {0.2, 0.3, 0.0}, 1.0},  // foot in the interior
        {{0.5, -1.0, 2.0}, {0.5, 0.0, 0.0}, 5.0}, // foot on an edge
        {{-1.0, -1.0, 0.0}, {0.0, 0.0, 0.0}, 2.0}, // foot at a vertex
        {{2.0, 2.0, 0.0}, {0.5, 0.5, 0.0}, 4.5},  // foot on the hypotenuse
    };
    for ( const Case& c : cases )
    {
        ClosestPointResult res = closest_point_in_simplex(c.p, V);
        CHECK((res.point - c.expected).norm() < 1e-12);
        CHECK(res.distance_squared == doctest::Approx(c.dsq).epsilon(1e-12));
    }
}

TEST_CASE("closest_point_in_simplex against barycentric grid (Euclidean and metric)")
{
    std::mt19937 gen(1003);
    for ( int rep = 0; rep < 60; ++rep )
    {
        const int d = 2 + (rep % 2);
        const int K = 2 + (rep % d); // segments, triangles, tetrahedra; some lower-dim
        Eigen::MatrixXd V = th::randn_matrix(d, K, gen);
        Eigen::VectorXd p = th::randn_vector(d, gen, 1.5);
        Eigen::MatrixXd M = (rep % 2 == 0)
            ? Eigen::MatrixXd(Eigen::MatrixXd::Identity(d, d))
            : th::random_spd(d, gen, 0.3, 3.0);

        ClosestPointResult res = closest_point_in_simplex(p, V, M);

        const int N = (d == 2) ? 200 : 60;
        auto f = [&](const Eigen::VectorXd& x)
        {
            Eigen::VectorXd y = x - p;
            return y.dot(M * y);
        };
        const double gm = th::grid_min_over_simplex(f, V, N);

        // The solver can only do better than the grid (up to roundoff) ...
        CHECK(res.distance_squared <= gm + 1e-9);
        // ... and not much worse than it, since the grid is fairly fine.
        CHECK(res.distance_squared >= gm - 0.05 * (1.0 + gm));

        // Returned point lies in the simplex and matches the reported distance
        CHECK(closest_point_in_simplex(res.point, V).distance_squared < 1e-10);
        Eigen::VectorXd y = res.point - p;
        CHECK(y.dot(M * y) == doctest::Approx(res.distance_squared).epsilon(1e-6));
    }
}

TEST_CASE("closest_point_in_box: metric version against grid, Euclidean is exact clamp")
{
    std::mt19937 gen(1004);
    for ( int rep = 0; rep < 60; ++rep )
    {
        const int d = 2 + (rep % 2);
        Eigen::VectorXd c = th::randn_vector(d, gen);
        Eigen::VectorXd w = th::randn_vector(d, gen).cwiseAbs() + Eigen::VectorXd::Constant(d, 0.2);
        Box B{c - w, c + w};
        Eigen::VectorXd p = th::randn_vector(d, gen, 1.5);
        Eigen::MatrixXd M = th::random_spd(d, gen, 0.3, 3.0);

        // Euclidean special case: coordinate descent equals plain clamping
        ClosestPointResult eu = closest_point_in_box(p, B);
        ClosestPointResult eu2 = closest_point_in_box(p, B, Eigen::MatrixXd::Identity(d, d));
        CHECK((eu.point - eu2.point).norm() < 1e-12);
        CHECK(eu.distance_squared == doctest::Approx(eu2.distance_squared).epsilon(1e-12));

        ClosestPointResult res = closest_point_in_box(p, B, M);
        CHECK((res.point.array() >= B.lo.array() - 1e-12).all());
        CHECK((res.point.array() <= B.hi.array() + 1e-12).all());

        const int N = (d == 2) ? 200 : 45;
        auto f = [&](const Eigen::VectorXd& x)
        {
            Eigen::VectorXd y = x - p;
            return y.dot(M * y);
        };
        const double gm = th::grid_min_over_box(f, B.lo, B.hi, N);
        CHECK(res.distance_squared <= gm + 1e-9);
        CHECK(res.distance_squared >= gm - 0.05 * (1.0 + gm));

        // stop_below early exit still returns a certified value
        BoxClosestPointOptions opts;
        opts.stop_below = res.distance_squared * 4.0 + 1.0;
        ClosestPointResult early = closest_point_in_box(p, B, M, opts);
        CHECK(early.distance_squared <= opts.stop_below);
    }
}

TEST_CASE("ellipsoid-simplex against barycentric grid")
{
    std::mt19937 gen(1005);
    int classified = 0;
    const int num_cases = 90;
    for ( int rep = 0; rep < num_cases; ++rep )
    {
        const int d = (rep % 3 == 0) ? 3 : 2;
        Simplex S{th::randn_matrix(d, d + 1, gen)};
        Ellipsoid E{th::randn_vector(d, gen, 1.2), th::random_spd(d, gen, 0.2, 3.0)};
        const double tau = 0.6 + 0.12 * (rep % 6);
        const double t2 = tau * tau;

        const int N = (d == 2) ? 150 : 40;
        auto q = [&](const Eigen::VectorXd& x) { return th::mahalanobis_sq(x, E.mu, E.Sigma); };
        const double gm = th::grid_min_over_simplex(q, S.V, N);

        const double delta = (d == 2) ? 0.05 : 0.12;
        if ( gm <= (1.0 - delta) * t2 )
        {
            CHECK(intersects(E, S, tau));
            CHECK(intersects(S, E, tau));
            classified += 1;
        }
        else if ( gm >= (1.0 + delta) * t2 )
        {
            CHECK(!intersects(E, S, tau));
            classified += 1;
        }

        // Independent certificate: a vertex inside the ellipsoid forces overlap
        for ( int jj = 0; jj < S.V.cols(); ++jj )
        {
            if ( q(S.V.col(jj)) <= t2 * (1.0 - 1e-9) )
            {
                CHECK(intersects(E, S, tau));
                break;
            }
        }
    }
    CHECK(classified >= (2 * num_cases) / 3);
}

TEST_CASE("ellipsoid-simplex and box-ellipsoid containment cases")
{
    std::mt19937 gen(1006);
    for ( int rep = 0; rep < 20; ++rep )
    {
        const int d = 2 + (rep % 2);

        // Tiny ellipsoid strictly inside a large simplex: no boundary contact
        Eigen::MatrixXd V = 10.0 * th::randn_matrix(d, d + 1, gen);
        Eigen::VectorXd alpha = th::random_simplex_alpha(d + 1, gen);
        alpha = (alpha + Eigen::VectorXd::Constant(d + 1, 0.5)) / (1.0 + 0.5 * (d + 1));
        Eigen::VectorXd inside = V * alpha;
        Ellipsoid tiny{inside, 1e-6 * th::random_spd(d, gen, 0.5, 2.0)};
        CHECK(intersects(tiny, Simplex{V}, 1.0));

        // Small simplex strictly inside a large ellipsoid
        Ellipsoid big{th::randn_vector(d, gen), 100.0 * th::random_spd(d, gen, 0.5, 2.0)};
        Simplex small{(0.01 * th::randn_matrix(d, d + 1, gen)).colwise() + big.mu};
        CHECK(intersects(big, small, 1.0));

        // Tiny ellipsoid strictly inside a box
        Box B{inside - Eigen::VectorXd::Constant(d, 1.0),
              inside + Eigen::VectorXd::Constant(d, 1.0)};
        CHECK(intersects(B, tiny, 1.0));

        // Box strictly inside a large ellipsoid
        Box B2{big.mu - Eigen::VectorXd::Constant(d, 0.01),
               big.mu + Eigen::VectorXd::Constant(d, 0.01)};
        CHECK(intersects(B2, big, 1.0));
    }
}

TEST_CASE("box-ellipsoid against grid, and the thin-diagonal-ellipsoid corner case")
{
    std::mt19937 gen(1007);
    int classified = 0;
    const int num_cases = 90;
    for ( int rep = 0; rep < num_cases; ++rep )
    {
        const int d = (rep % 3 == 0) ? 3 : 2;
        Eigen::VectorXd c = th::randn_vector(d, gen);
        Eigen::VectorXd w = th::randn_vector(d, gen).cwiseAbs() + Eigen::VectorXd::Constant(d, 0.2);
        Box B{c - w, c + w};
        Ellipsoid E{th::randn_vector(d, gen, 1.2), th::random_spd(d, gen, 0.2, 3.0)};
        const double tau = 0.6 + 0.12 * (rep % 6);
        const double t2 = tau * tau;

        const int N = (d == 2) ? 150 : 40;
        auto q = [&](const Eigen::VectorXd& x) { return th::mahalanobis_sq(x, E.mu, E.Sigma); };
        const double gm = th::grid_min_over_box(q, B.lo, B.hi, N);

        const double delta = (d == 2) ? 0.05 : 0.12;
        if ( gm <= (1.0 - delta) * t2 )
        {
            CHECK(intersects(B, E, tau));
            CHECK(intersects(E, B, tau));
            classified += 1;
        }
        else if ( gm >= (1.0 + delta) * t2 )
        {
            CHECK(!intersects(B, E, tau));
            classified += 1;
        }
    }
    CHECK(classified >= (2 * num_cases) / 3);

    // Thin ellipsoid along the diagonal vs a box near the (+,-) corner: the
    // bounding boxes overlap but the exact test must say disjoint. This is
    // precisely the configuration where tier-2 pruning earns its keep.
    const double s2 = std::sqrt(0.5);
    Eigen::Matrix2d R;
    R << s2, -s2,
         s2,  s2;
    Eigen::Vector2d eigs(4.0, 0.01);
    Ellipsoid E{Eigen::Vector2d(0.0, 0.0),
                R * eigs.asDiagonal() * R.transpose()};
    Box corner_box{Eigen::Vector2d(1.0, -1.4), Eigen::Vector2d(1.4, -1.0)};
    CHECK(intersects(bounding_box(E, 1.0), corner_box)); // broad phase passes
    CHECK(!intersects(corner_box, E, 1.0));              // exact test rejects
}

TEST_CASE("ball-simplex against barycentric grid")
{
    std::mt19937 gen(1008);
    int classified = 0;
    const int num_cases = 60;
    for ( int rep = 0; rep < num_cases; ++rep )
    {
        const int d = (rep % 3 == 0) ? 3 : 2;
        Simplex S{th::randn_matrix(d, d + 1, gen)};
        Ball B{th::randn_vector(d, gen, 1.2), 0.3 + 0.1 * (rep % 6)};

        const int N = (d == 2) ? 150 : 40;
        auto q = [&](const Eigen::VectorXd& x) { return (x - B.center).squaredNorm(); };
        const double gm = th::grid_min_over_simplex(q, S.V, N);
        const double r2 = B.radius * B.radius;

        const double delta = (d == 2) ? 0.03 : 0.08;
        if ( gm <= (1.0 - delta) * r2 )
        {
            CHECK(intersects(B, S));
            classified += 1;
        }
        else if ( gm >= (1.0 + delta) * r2 )
        {
            CHECK(!intersects(B, S));
            classified += 1;
        }
    }
    CHECK(classified >= (2 * num_cases) / 3);
}

TEST_CASE("segment-simplex: interval method against sampled distances")
{
    std::mt19937 gen(1009);
    int classified = 0;
    const int num_cases = 60;
    for ( int rep = 0; rep < num_cases; ++rep )
    {
        const int d = 2 + (rep % 2);
        Simplex S{th::randn_matrix(d, d + 1, gen)};
        Segment seg{th::randn_vector(d, gen, 1.5), th::randn_vector(d, gen, 1.5)};

        const int n_grid = 2000;
        double min_dsq = std::numeric_limits<double>::infinity();
        for ( int ii = 0; ii <= n_grid; ++ii )
        {
            const double t = ii / static_cast<double>(n_grid);
            Eigen::VectorXd x = seg.a + t * (seg.b - seg.a);
            min_dsq = std::min(min_dsq, closest_point_in_simplex(x, S.V).distance_squared);
        }
        const double scale = 1.0 + S.V.cwiseAbs().maxCoeff() + seg.a.cwiseAbs().maxCoeff();

        if ( min_dsq <= 1e-12 * scale * scale )
        {
            CHECK(intersects(seg, S));
            classified += 1;
        }
        else if ( min_dsq >= 1e-6 * scale * scale )
        {
            CHECK(!intersects(seg, S));
            classified += 1;
        }
    }
    CHECK(classified >= (2 * num_cases) / 3);

    // Lower-dimensional target: two segments in 3D
    Eigen::MatrixXd V(3, 2);
    V << -1.0, 1.0,
          0.0, 0.0,
          0.0, 0.0; // segment along the x-axis
    Simplex S{V};
    Segment crossing{Eigen::Vector3d(0.3, -1.0, 0.0), Eigen::Vector3d(0.3, 2.0, 0.0)};
    CHECK(intersects(crossing, S));
    Segment skew{Eigen::Vector3d(0.3, -1.0, 0.5), Eigen::Vector3d(0.3, 2.0, 0.5)};
    CHECK(!intersects(skew, S));
}
