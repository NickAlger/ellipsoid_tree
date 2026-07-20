// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "ellipsoid_tree/simplex_mesh.hpp"
#include "test_helpers.hpp"

#include <random>

using namespace ellipsoid_tree;
namespace th = test_helpers;

namespace {

// Unit square [0,1]^2, m x m quads, two triangles each.
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> structured_triangle_mesh( int m )
{
    Eigen::MatrixXd vertices(2, (m + 1) * (m + 1));
    for ( int jj = 0; jj <= m; ++jj )
    {
        for ( int ii = 0; ii <= m; ++ii )
        {
            vertices.col(jj * (m + 1) + ii) = Eigen::Vector2d(ii / double(m), jj / double(m));
        }
    }
    Eigen::MatrixXi cells(3, 2 * m * m);
    int cc = 0;
    for ( int jj = 0; jj < m; ++jj )
    {
        for ( int ii = 0; ii < m; ++ii )
        {
            const int v00 = jj * (m + 1) + ii;
            const int v10 = v00 + 1;
            const int v01 = v00 + (m + 1);
            const int v11 = v01 + 1;
            cells.col(cc++) = Eigen::Vector3i(v00, v10, v11);
            cells.col(cc++) = Eigen::Vector3i(v00, v11, v01);
        }
    }
    return {vertices, cells};
}

Eigen::Vector2d clamp_to_unit_square( const Eigen::Vector2d& p )
{
    return Eigen::Vector2d(std::min(1.0, std::max(0.0, p(0))),
                           std::min(1.0, std::max(0.0, p(1))));
}

} // end anonymous namespace

TEST_CASE("SimplexMesh on the unit square: location, membership, closest point")
{
    auto [vertices, cells] = structured_triangle_mesh(6);
    SimplexMesh mesh(vertices, cells);
    CHECK(mesh.num_cells() == 72);
    CHECK(mesh.num_boundary_faces() == 4 * 6); // 6 edges per side

    std::mt19937 gen(6001);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    Eigen::MatrixXd interior(2, 100);
    Eigen::MatrixXd exterior(2, 100);
    for ( int ii = 0; ii < 100; ++ii )
    {
        interior.col(ii) = Eigen::Vector2d(unif(gen), unif(gen));
        const double angle = 6.283185307179586 * unif(gen);
        exterior.col(ii) = Eigen::Vector2d(0.5 + (0.8 + unif(gen)) * std::cos(angle),
                                           0.5 + (0.8 + unif(gen)) * std::sin(angle));
        if ( (exterior.col(ii).array() >= 0.0).all() && (exterior.col(ii).array() <= 1.0).all() )
        {
            exterior.col(ii) = Eigen::Vector2d(2.0 + unif(gen), -1.0 - unif(gen));
        }
    }

    // Interior points land in a containing cell with clean barycentric coords
    auto locate_result = mesh.locate_points(interior);
    for ( int ii = 0; ii < 100; ++ii )
    {
        const int e = locate_result.first(ii);
        CHECK(e >= 0);
        Eigen::VectorXd alpha = locate_result.second.col(ii);
        CHECK((alpha.array() >= -1e-12).all());
        CHECK(alpha.sum() == doctest::Approx(1.0).epsilon(1e-9));
        Eigen::MatrixXd V(2, 3);
        for ( int kk = 0; kk < 3; ++kk )
        {
            V.col(kk) = vertices.col(cells(kk, e));
        }
        CHECK((V * alpha - interior.col(ii)).norm() < 1e-12);
    }
    CHECK(mesh.point_is_in_mesh(interior).all());
    CHECK(!mesh.point_is_in_mesh(exterior).any());

    // Closest point: analytic (clamp to the unit square)
    Eigen::MatrixXd cps = mesh.closest_points(exterior);
    for ( int ii = 0; ii < 100; ++ii )
    {
        CHECK((cps.col(ii) - clamp_to_unit_square(exterior.col(ii))).norm() < 1e-11);
    }
    Eigen::MatrixXd cps_in = mesh.closest_points(interior);
    for ( int ii = 0; ii < 100; ++ii )
    {
        CHECK((cps_in.col(ii) - interior.col(ii)).norm() == 0.0);
    }
}

TEST_CASE("SimplexMesh CG1 evaluation reproduces linear functions")
{
    auto [vertices, cells] = structured_triangle_mesh(5);
    SimplexMesh mesh(vertices, cells);

    // Two linear functions f(x) = a . x + b, exactly representable in CG1
    Eigen::MatrixXd F(2, vertices.cols());
    Eigen::Vector2d a1(1.5, -0.7), a2(-0.3, 2.1);
    const double b1 = 0.25, b2 = -1.0;
    for ( int vv = 0; vv < vertices.cols(); ++vv )
    {
        F(0, vv) = a1.dot(vertices.col(vv)) + b1;
        F(1, vv) = a2.dot(vertices.col(vv)) + b2;
    }

    std::mt19937 gen(6002);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    Eigen::MatrixXd pts(2, 60);
    for ( int ii = 0; ii < 60; ++ii )
    {
        pts.col(ii) = Eigen::Vector2d(unif(gen), unif(gen));
    }

    Eigen::MatrixXd vals = mesh.eval_cg1(F, pts);
    for ( int ii = 0; ii < 60; ++ii )
    {
        CHECK(vals(0, ii) == doctest::Approx(a1.dot(pts.col(ii)) + b1).epsilon(1e-11));
        CHECK(vals(1, ii) == doctest::Approx(a2.dot(pts.col(ii)) + b2).epsilon(1e-11));
    }

    // Without reflection, exterior points evaluate to zero
    Eigen::MatrixXd outside(2, 2);
    outside.col(0) = Eigen::Vector2d(1.2, 0.5);
    outside.col(1) = Eigen::Vector2d(-0.3, 0.4);
    Eigen::MatrixXd vals_out = mesh.eval_cg1(F, outside, false);
    CHECK(vals_out.cwiseAbs().maxCoeff() == 0.0);

    // With reflection, an exterior point near the boundary evaluates the
    // function at its reflection 2*closest - p (computable analytically here)
    Eigen::MatrixXd vals_refl = mesh.eval_cg1(F, outside, true);
    for ( int ii = 0; ii < 2; ++ii )
    {
        Eigen::Vector2d p = outside.col(ii);
        Eigen::Vector2d reflected = 2.0 * clamp_to_unit_square(p) - p;
        CHECK(vals_refl(0, ii) == doctest::Approx(a1.dot(reflected) + b1).epsilon(1e-10));
        CHECK(vals_refl(1, ii) == doctest::Approx(a2.dot(reflected) + b2).epsilon(1e-10));
    }

    // A far-outside point whose reflection is still outside gives zero
    Eigen::MatrixXd far_pt(2, 1);
    far_pt.col(0) = Eigen::Vector2d(3.0, 0.5); // reflects to (-1, 0.5), still outside
    CHECK(mesh.eval_cg1(F, far_pt, true).cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("SimplexMesh in 3D: two tetrahedra sharing a face")
{
    Eigen::MatrixXd vertices(3, 5);
    vertices.col(0) = Eigen::Vector3d(0, 0, 0);
    vertices.col(1) = Eigen::Vector3d(1, 0, 0);
    vertices.col(2) = Eigen::Vector3d(0, 1, 0);
    vertices.col(3) = Eigen::Vector3d(0, 0, 1);
    vertices.col(4) = Eigen::Vector3d(1, 1, 1);
    Eigen::MatrixXi cells(4, 2);
    cells.col(0) = Eigen::Vector4i(0, 1, 2, 3);
    cells.col(1) = Eigen::Vector4i(1, 2, 3, 4);
    SimplexMesh mesh(vertices, cells);

    CHECK(mesh.num_boundary_faces() == 6); // shared face {1,2,3} is interior

    std::mt19937 gen(6003);
    for ( int rep = 0; rep < 50; ++rep )
    {
        // Random point in one of the tets via barycentric sampling
        const int cell = rep % 2;
        Eigen::VectorXd alpha = th::random_simplex_alpha(4, gen);
        Eigen::MatrixXd V(3, 4);
        for ( int kk = 0; kk < 4; ++kk )
        {
            V.col(kk) = vertices.col(cells(kk, cell));
        }
        Eigen::VectorXd p = V * alpha;

        auto loc = mesh.locate_points(p);
        const int e = loc.first(0);
        CHECK(e >= 0);
        Eigen::MatrixXd Ve(3, 4);
        for ( int kk = 0; kk < 4; ++kk )
        {
            Ve.col(kk) = vertices.col(cells(kk, e));
        }
        CHECK((Ve * loc.second.col(0) - p).norm() < 1e-12);

        // Closest point of an exterior point: brute force over boundary faces
        Eigen::VectorXd q = th::randn_vector(3, gen, 1.5);
        if ( !mesh.point_is_in_mesh(q)(0) )
        {
            Eigen::VectorXd cp = mesh.closest_point(q);
            double best = std::numeric_limits<double>::infinity();
            for ( int ff = 0; ff < mesh.num_boundary_faces(); ++ff )
            {
                Eigen::MatrixXd Vf(3, 3);
                for ( int kk = 0; kk < 3; ++kk )
                {
                    Vf.col(kk) = vertices.col(mesh.boundary_faces()(kk, ff));
                }
                best = std::min(best, closest_point_in_simplex(q, Vf).distance_squared);
            }
            CHECK((cp - q).squaredNorm() == doctest::Approx(best).epsilon(1e-9));
        }
    }
}

TEST_CASE("SimplexMesh in 1D: interval mesh")
{
    const int m = 10;
    Eigen::MatrixXd vertices(1, m + 1);
    Eigen::MatrixXi cells(2, m);
    for ( int ii = 0; ii <= m; ++ii )
    {
        vertices(0, ii) = ii / double(m);
    }
    for ( int ii = 0; ii < m; ++ii )
    {
        cells.col(ii) = Eigen::Vector2i(ii, ii + 1);
    }
    SimplexMesh mesh(vertices, cells);
    CHECK(mesh.num_boundary_faces() == 2); // the two endpoints

    Eigen::MatrixXd pts(1, 3);
    pts << 0.35, -0.2, 1.7;
    CHECK(mesh.point_is_in_mesh(pts)(0));
    CHECK(!mesh.point_is_in_mesh(pts)(1));
    CHECK(!mesh.point_is_in_mesh(pts)(2));

    Eigen::MatrixXd cps = mesh.closest_points(pts);
    CHECK(cps(0, 0) == doctest::Approx(0.35));
    CHECK(cps(0, 1) == doctest::Approx(0.0));
    CHECK(cps(0, 2) == doctest::Approx(1.0));

    // Linear reproduction in 1D
    Eigen::MatrixXd F(1, m + 1);
    for ( int ii = 0; ii <= m; ++ii )
    {
        F(0, ii) = 3.0 * vertices(0, ii) - 0.5;
    }
    Eigen::MatrixXd vals = mesh.eval_cg1(F, pts.leftCols(1));
    CHECK(vals(0, 0) == doctest::Approx(3.0 * 0.35 - 0.5).epsilon(1e-12));
}

TEST_CASE("SimplexMesh ellipsoid queries match brute force")
{
    auto [vertices, cells] = structured_triangle_mesh(6);
    SimplexMesh mesh(vertices, cells);

    std::mt19937 gen(6004);
    for ( int rep = 0; rep < 20; ++rep )
    {
        Ellipsoid E{Eigen::Vector2d(th::randn_vector(2, gen, 0.6)) + Eigen::Vector2d(0.5, 0.5),
                    0.01 * th::random_spd(2, gen, 0.3, 3.0)};
        const double tau = 0.8 + 0.05 * rep;

        std::vector<int> brute;
        for ( int cc = 0; cc < mesh.num_cells(); ++cc )
        {
            Eigen::MatrixXd V(2, 3);
            for ( int kk = 0; kk < 3; ++kk )
            {
                V.col(kk) = vertices.col(cells(kk, cc));
            }
            if ( intersects(E, Simplex{V}, tau) )
            {
                brute.push_back(cc);
            }
        }
        std::vector<int> got = mesh.cells_intersecting(E, tau);
        std::sort(got.begin(), got.end());
        CHECK(got == brute);
    }

    // Mesh x EllipsoidTree via dual traversal
    std::vector<Ellipsoid> family;
    for ( int ii = 0; ii < 40; ++ii )
    {
        family.push_back(Ellipsoid{Eigen::Vector2d(th::randn_vector(2, gen, 0.8)),
                                   0.02 * th::random_spd(2, gen, 0.3, 3.0)});
    }
    EllipsoidTree T(family, 1.0);
    std::set<std::pair<int, int>> brute_pairs;
    for ( int cc = 0; cc < mesh.num_cells(); ++cc )
    {
        Eigen::MatrixXd V(2, 3);
        for ( int kk = 0; kk < 3; ++kk )
        {
            V.col(kk) = vertices.col(cells(kk, cc));
        }
        for ( int jj = 0; jj < T.size(); ++jj )
        {
            if ( intersects(T.object(jj), Simplex{V}, 1.0) )
            {
                brute_pairs.insert({cc, jj});
            }
        }
    }
    std::vector<std::pair<int, int>> got_pairs = mesh.cell_ellipsoid_pairs(T);
    CHECK(std::set<std::pair<int, int>>(got_pairs.begin(), got_pairs.end()) == brute_pairs);
}
