// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "ellipsoid_tree/geometry.hpp"
#include "test_helpers.hpp"

#include <random>

using namespace ellipsoid_tree;
namespace th = test_helpers;

TEST_CASE("bounding_box of point, ball, segment, simplex")
{
    std::mt19937 gen(101);
    Eigen::VectorXd p = th::randn_vector(3, gen);

    Box bp = bounding_box(Eigen::Ref<const Eigen::VectorXd>(p));
    CHECK(bp.lo == p);
    CHECK(bp.hi == p);

    Ball ball{p, 2.5};
    Box bb = bounding_box(ball);
    CHECK((bb.lo - (p.array() - 2.5).matrix()).norm() == doctest::Approx(0.0));
    CHECK((bb.hi - (p.array() + 2.5).matrix()).norm() == doctest::Approx(0.0));

    Segment seg{th::randn_vector(3, gen), th::randn_vector(3, gen)};
    Box bs = bounding_box(seg);
    for ( int kk = 0; kk < 3; ++kk )
    {
        CHECK(bs.lo(kk) == std::min(seg.a(kk), seg.b(kk)));
        CHECK(bs.hi(kk) == std::max(seg.a(kk), seg.b(kk)));
    }

    Simplex S{th::randn_matrix(3, 4, gen)};
    Box bx = bounding_box(S);
    for ( int kk = 0; kk < 3; ++kk )
    {
        CHECK(bx.lo(kk) == S.V.row(kk).minCoeff());
        CHECK(bx.hi(kk) == S.V.row(kk).maxCoeff());
    }
}

TEST_CASE("ellipsoid bounding box contains boundary samples and is tight")
{
    std::mt19937 gen(202);
    for ( int rep = 0; rep < 20; ++rep )
    {
        const int d = 2 + (rep % 3);
        Ellipsoid E{th::randn_vector(d, gen), th::random_spd(d, gen, 0.2, 5.0)};
        const double tau = 0.5 + 0.1 * rep;
        Box B = bounding_box(E, tau);

        Eigen::MatrixXd L = th::matrix_sqrt_spd(E.Sigma);

        // Containment of boundary samples x = mu + tau * L u, ||u|| = 1
        for ( int ss = 0; ss < 200; ++ss )
        {
            Eigen::VectorXd u = th::randn_vector(d, gen);
            u /= u.norm();
            Eigen::VectorXd x = E.mu + tau * (L * u);
            CHECK((B.lo.array() - 1e-12 <= x.array()).all());
            CHECK((x.array() <= B.hi.array() + 1e-12).all());
        }

        // Tightness: the boundary point maximizing coordinate k touches the face
        for ( int kk = 0; kk < d; ++kk )
        {
            Eigen::VectorXd u = L.row(kk).transpose(); // L^T e_k (L symmetric)
            u /= u.norm();
            Eigen::VectorXd x = E.mu + tau * (L * u);
            CHECK(x(kk) == doctest::Approx(B.hi(kk)).epsilon(1e-10));
        }
    }
}

TEST_CASE("simplex_transform reproduces barycentric coordinates")
{
    std::mt19937 gen(303);
    for ( int rep = 0; rep < 30; ++rep )
    {
        const int d = 1 + (rep % 4);
        const int K = 1 + (rep % (d + 1)); // 1 .. d+1 vertices
        Eigen::MatrixXd V = th::randn_matrix(d, K, gen);

        std::pair<Eigen::MatrixXd, Eigen::VectorXd> Ab = simplex_transform(V);
        const Eigen::MatrixXd& A = Ab.first;
        const Eigen::VectorXd& b = Ab.second;

        // Vertices map to unit coordinate vectors
        for ( int jj = 0; jj < K; ++jj )
        {
            Eigen::VectorXd alpha = A * V.col(jj) + b;
            for ( int ii = 0; ii < K; ++ii )
            {
                CHECK(alpha(ii) == doctest::Approx(ii == jj ? 1.0 : 0.0).epsilon(1e-9));
            }
        }

        // Coordinates sum to one for arbitrary points
        Eigen::VectorXd x = th::randn_vector(d, gen, 3.0);
        CHECK((A * x + b).sum() == doctest::Approx(1.0).epsilon(1e-9));

        // Interior points are recovered exactly (full-dimensional case)
        if ( K == d + 1 )
        {
            Eigen::VectorXd alpha_in = th::random_simplex_alpha(K, gen);
            Eigen::VectorXd alpha_out = A * (V * alpha_in) + b;
            CHECK((alpha_out - alpha_in).cwiseAbs().maxCoeff() < 1e-9);
        }
    }
}
