// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "ellipsoid_tree/intersections.hpp"
#include "ellipsoid_tree/object_tree.hpp"
#include "ellipsoid_tree/simplex_mesh.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <random>
#include <set>

using namespace ellipsoid_tree;
namespace th = test_helpers;

namespace {

// ----- Separating-axis ground truth (exact for convex polytopes with the
// complete axis set: face normals of both plus all edge-direction cross
// products in 3D). Returns the signed margin: the minimum over axes of the
// projection-interval overlap. margin < 0 certifies disjoint; margin > 0
// certifies intersection.

std::pair<double, double> project( const Eigen::MatrixXd& V, const Eigen::VectorXd& n )
{
    Eigen::VectorXd vals = V.transpose() * n;
    return {vals.minCoeff(), vals.maxCoeff()};
}

double sat_margin( const Eigen::MatrixXd& VA, const Eigen::MatrixXd& VB,
                   const std::vector<Eigen::VectorXd>& axes )
{
    double margin = std::numeric_limits<double>::infinity();
    for ( const Eigen::VectorXd& axis : axes )
    {
        const double norm = axis.norm();
        if ( norm < 1e-12 )
        {
            continue; // degenerate axis (parallel edges)
        }
        Eigen::VectorXd n = axis / norm;
        std::pair<double, double> ia = project(VA, n);
        std::pair<double, double> ib = project(VB, n);
        margin = std::min(margin, std::min(ia.second, ib.second)
                                  - std::max(ia.first, ib.first));
    }
    return margin;
}

std::vector<Eigen::VectorXd> polygon_edge_normals_2d( const Eigen::MatrixXd& V )
{
    std::vector<Eigen::VectorXd> axes;
    const int K = static_cast<int>(V.cols());
    for ( int ii = 0; ii < K; ++ii )
    {
        for ( int jj = ii + 1; jj < K; ++jj )
        {
            Eigen::Vector2d e = V.col(jj) - V.col(ii);
            axes.push_back(Eigen::Vector2d(-e(1), e(0)));
        }
    }
    return axes;
}

std::vector<Eigen::VectorXd> tet_face_normals( const Eigen::MatrixXd& V )
{
    std::vector<Eigen::VectorXd> axes;
    for ( int skip = 0; skip < 4; ++skip )
    {
        std::vector<int> f;
        for ( int kk = 0; kk < 4; ++kk )
        {
            if ( kk != skip ) { f.push_back(kk); }
        }
        Eigen::Vector3d a = V.col(f[1]) - V.col(f[0]);
        Eigen::Vector3d b = V.col(f[2]) - V.col(f[0]);
        axes.push_back(a.cross(b));
    }
    return axes;
}

std::vector<Eigen::Vector3d> tet_edges( const Eigen::MatrixXd& V )
{
    std::vector<Eigen::Vector3d> edges;
    for ( int ii = 0; ii < 4; ++ii )
    {
        for ( int jj = ii + 1; jj < 4; ++jj )
        {
            edges.push_back(V.col(jj) - V.col(ii));
        }
    }
    return edges;
}

double sat_margin_tri_tri_2d( const Eigen::MatrixXd& VA, const Eigen::MatrixXd& VB )
{
    std::vector<Eigen::VectorXd> axes = polygon_edge_normals_2d(VA);
    for ( const Eigen::VectorXd& n : polygon_edge_normals_2d(VB) )
    {
        axes.push_back(n);
    }
    return sat_margin(VA, VB, axes);
}

double sat_margin_tet_tet_3d( const Eigen::MatrixXd& VA, const Eigen::MatrixXd& VB )
{
    std::vector<Eigen::VectorXd> axes = tet_face_normals(VA);
    for ( const Eigen::VectorXd& n : tet_face_normals(VB) )
    {
        axes.push_back(n);
    }
    for ( const Eigen::Vector3d& ea : tet_edges(VA) )
    {
        for ( const Eigen::Vector3d& eb : tet_edges(VB) )
        {
            axes.push_back(ea.cross(eb));
        }
    }
    return sat_margin(VA, VB, axes);
}

Eigen::MatrixXd box_corners( const Box& B )
{
    const int d = static_cast<int>(B.lo.size());
    Eigen::MatrixXd C(d, 1 << d);
    for ( int mask = 0; mask < (1 << d); ++mask )
    {
        for ( int kk = 0; kk < d; ++kk )
        {
            C(kk, mask) = (mask >> kk) & 1 ? B.hi(kk) : B.lo(kk);
        }
    }
    return C;
}

double sat_margin_box_simplex( const Box& B, const Eigen::MatrixXd& VS )
{
    const int d = static_cast<int>(B.lo.size());
    std::vector<Eigen::VectorXd> axes;
    for ( int kk = 0; kk < d; ++kk )
    {
        axes.push_back(Eigen::VectorXd::Unit(d, kk)); // box face normals
    }
    if ( d == 2 )
    {
        for ( const Eigen::VectorXd& n : polygon_edge_normals_2d(VS) )
        {
            axes.push_back(n);
        }
    }
    else
    {
        for ( const Eigen::VectorXd& n : tet_face_normals(VS) )
        {
            axes.push_back(n);
        }
        for ( int kk = 0; kk < 3; ++kk )
        {
            for ( const Eigen::Vector3d& es : tet_edges(VS) )
            {
                axes.push_back(Eigen::Vector3d(Eigen::Vector3d::Unit(kk)).cross(es));
            }
        }
    }
    return sat_margin(box_corners(B), VS, axes);
}

std::vector<int> sorted( std::vector<int> v )
{
    std::sort(v.begin(), v.end());
    return v;
}

} // end anonymous namespace

TEST_CASE("simplex-simplex against separating-axis ground truth")
{
    std::mt19937 gen(7001);
    int classified_2d = 0;
    for ( int rep = 0; rep < 200; ++rep )
    {
        Eigen::MatrixXd VA = th::randn_matrix(2, 3, gen);
        Eigen::MatrixXd VB = th::randn_matrix(2, 3, gen);
        const double margin = sat_margin_tri_tri_2d(VA, VB);
        if ( std::abs(margin) > 1e-8 )
        {
            CHECK(intersects(Simplex{VA}, Simplex{VB}) == (margin > 0.0));
            classified_2d += 1;
        }
    }
    CHECK(classified_2d >= 190);

    int classified_3d = 0;
    for ( int rep = 0; rep < 100; ++rep )
    {
        Eigen::MatrixXd VA = th::randn_matrix(3, 4, gen);
        Eigen::MatrixXd VB = th::randn_matrix(3, 4, gen);
        const double margin = sat_margin_tet_tet_3d(VA, VB);
        if ( std::abs(margin) > 1e-8 )
        {
            CHECK(intersects(Simplex{VA}, Simplex{VB}) == (margin > 0.0));
            classified_3d += 1;
        }
    }
    CHECK(classified_3d >= 90);
}

TEST_CASE("box-simplex against separating-axis ground truth, and corner-hull consistency")
{
    std::mt19937 gen(7002);
    for ( int d : {2, 3} )
    {
        int classified = 0;
        for ( int rep = 0; rep < 100; ++rep )
        {
            Eigen::VectorXd c = th::randn_vector(d, gen);
            Eigen::VectorXd w = th::randn_vector(d, gen, 0.4).cwiseAbs()
                                + Eigen::VectorXd::Constant(d, 0.1);
            Box B{c - w, c + w};
            Eigen::MatrixXd VS = th::randn_matrix(d, d + 1, gen);

            const double margin = sat_margin_box_simplex(B, VS);
            if ( std::abs(margin) > 1e-8 )
            {
                CHECK(intersects(B, Simplex{VS}) == (margin > 0.0));
                classified += 1;
            }

            // The box is the convex hull of its corners: the simplex-simplex
            // LP on the corner hull must agree with the box-simplex LP.
            CHECK(intersects(B, Simplex{VS})
                  == intersects(Simplex{box_corners(B)}, Simplex{VS}));
        }
        CHECK(classified >= 85);
    }
}

TEST_CASE("LP cells: containment, tangency, lower-dimensional and degenerate hulls")
{
    // Containment without boundary crossing
    Eigen::MatrixXd big(2, 3);
    big << -10.0, 10.0, 0.0,
           -10.0, -10.0, 10.0;
    Eigen::MatrixXd small_tri(2, 3);
    small_tri << -0.1, 0.1, 0.0,
                 -0.1, -0.1, 0.1;
    CHECK(intersects(Simplex{big}, Simplex{small_tri}));
    CHECK(intersects(Box{Eigen::Vector2d(-1, -1), Eigen::Vector2d(1, 1)}, Simplex{small_tri}));
    CHECK(intersects(Box{Eigen::Vector2d(-0.01, -0.01), Eigen::Vector2d(0.01, 0.01)}, Simplex{big}));

    // Two triangles sharing exactly one vertex: touching counts
    Eigen::MatrixXd t1(2, 3), t2(2, 3);
    t1 << 0.0, 1.0, 0.0,
          0.0, 0.0, 1.0;
    t2 << 0.0, -1.0, 0.0,
          0.0, 0.0, -1.0;
    CHECK(intersects(Simplex{t1}, Simplex{t2}));

    // Clearly separated
    Eigen::MatrixXd t3 = t2;
    t3.row(0).array() -= 0.5;
    t3.row(1).array() -= 0.5;
    CHECK(!intersects(Simplex{t1}, Simplex{t3}));

    // Crossing segments in 2D (lower-dimensional hulls)
    Eigen::MatrixXd s1(2, 2), s2(2, 2);
    s1 << -1.0, 1.0,
           0.0, 0.0;
    s2 << 0.3, 0.3,
          -1.0, 1.0;
    CHECK(intersects(Simplex{s1}, Simplex{s2}));

    // Segments in 3D: crossing through a shared midpoint vs a near miss
    Eigen::MatrixXd u1(3, 2), u2(3, 2);
    u1 << -1.0, 1.0,   0.0, 0.0,   0.0, 0.0;
    u2 <<  0.0, 0.0,  -1.0, 1.0,   0.0, 0.0;
    CHECK(intersects(Simplex{u1}, Simplex{u2}));
    Eigen::MatrixXd u3 = u2;
    u3.row(2).array() += 1e-3;
    CHECK(!intersects(Simplex{u1}, Simplex{u3}));

    // Point-as-simplex agrees with the point-in-simplex predicate
    std::mt19937 gen(7003);
    for ( int rep = 0; rep < 30; ++rep )
    {
        const int d = 2 + (rep % 2);
        Eigen::MatrixXd V = th::randn_matrix(d, d + 1, gen);
        Eigen::VectorXd p = th::randn_vector(d, gen);
        CHECK(intersects(Simplex{p}, Simplex{V}) == intersects(p, Simplex{V}));
    }

    // Higher dimension sanity (d = 5): shared vertex vs far translation
    Eigen::MatrixXd VA = th::randn_matrix(5, 6, gen);
    Eigen::MatrixXd VB = th::randn_matrix(5, 6, gen);
    VB.col(0) = VA.col(0); // shared vertex
    CHECK(intersects(Simplex{VA}, Simplex{VB}));
    Eigen::MatrixXd VC = VB;
    VC.row(0).array() += 100.0;
    CHECK(!intersects(Simplex{VA}, Simplex{VC}));
}

TEST_CASE("tree wiring for the LP cells matches brute force")
{
    std::mt19937 gen(7004);
    const int d = 2;

    std::vector<Simplex> simplices(80);
    for ( int ii = 0; ii < 80; ++ii )
    {
        Eigen::VectorXd c = th::randn_vector(d, gen, 1.5);
        simplices[ii] = Simplex{(0.35 * th::randn_matrix(d, d + 1, gen)).colwise() + c};
    }
    std::vector<Box> boxes(60);
    for ( int ii = 0; ii < 60; ++ii )
    {
        Eigen::VectorXd c = th::randn_vector(d, gen, 1.5);
        Eigen::VectorXd w = th::randn_vector(d, gen, 0.2).cwiseAbs()
                            + Eigen::VectorXd::Constant(d, 0.05);
        boxes[ii] = Box{c - w, c + w};
    }
    SimplexTree TS(simplices);
    BoxTree TX(boxes);

    for ( int qq = 0; qq < 12; ++qq )
    {
        Box qb = boxes[qq];
        std::vector<int> brute_box;
        for ( int ii = 0; ii < TS.size(); ++ii )
        {
            if ( intersects(qb, TS.object(ii)) ) { brute_box.push_back(ii); }
        }
        CHECK(sorted(TS.collisions(qb)) == brute_box);

        Simplex qs = simplices[qq];
        std::vector<int> brute_sx;
        for ( int ii = 0; ii < TS.size(); ++ii )
        {
            if ( intersects(TS.object(ii), qs) ) { brute_sx.push_back(ii); }
        }
        CHECK(sorted(TS.collisions(qs)) == brute_sx);

        std::vector<int> brute_bx_sx;
        for ( int ii = 0; ii < TX.size(); ++ii )
        {
            if ( intersects(TX.object(ii), qs) ) { brute_bx_sx.push_back(ii); }
        }
        CHECK(sorted(TX.collisions(qs)) == brute_bx_sx);
    }

    // Self-collision and cross-tree pairs
    auto as_set = []( const std::vector<std::pair<int, int>>& v )
    { return std::set<std::pair<int, int>>(v.begin(), v.end()); };
    {
        std::set<std::pair<int, int>> brute;
        for ( int ii = 0; ii < TS.size(); ++ii )
        {
            for ( int jj = ii + 1; jj < TS.size(); ++jj )
            {
                if ( intersects(TS.object(ii), TS.object(jj)) ) { brute.insert({ii, jj}); }
            }
        }
        CHECK(as_set(TS.self_collision_pairs()) == brute);
    }
    {
        std::set<std::pair<int, int>> brute;
        for ( int ii = 0; ii < TX.size(); ++ii )
        {
            for ( int jj = 0; jj < TS.size(); ++jj )
            {
                if ( intersects(TX.object(ii), TS.object(jj)) ) { brute.insert({ii, jj}); }
            }
        }
        CHECK(as_set(collision_pairs(TX, TS)) == brute);
    }
}

TEST_CASE("mesh-mesh cell pairs: dual traversal vs brute force, conforming-mesh structure")
{
    // Two structured triangulations of the unit square, the second shifted
    auto build_mesh = []( int m, double shift_x, double shift_y )
    {
        Eigen::MatrixXd vertices(2, (m + 1) * (m + 1));
        for ( int jj = 0; jj <= m; ++jj )
        {
            for ( int ii = 0; ii <= m; ++ii )
            {
                vertices.col(jj * (m + 1) + ii) =
                    Eigen::Vector2d(ii / double(m) + shift_x, jj / double(m) + shift_y);
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
        return SimplexMesh(vertices, cells);
    };

    SimplexMesh A = build_mesh(5, 0.0, 0.0);
    SimplexMesh B = build_mesh(4, 0.37, 0.23);

    std::set<std::pair<int, int>> brute;
    for ( int ii = 0; ii < A.num_cells(); ++ii )
    {
        for ( int jj = 0; jj < B.num_cells(); ++jj )
        {
            if ( intersects(A.cell_tree().object(ii), B.cell_tree().object(jj)) )
            {
                brute.insert({ii, jj});
            }
        }
    }
    std::vector<std::pair<int, int>> got = A.cell_pairs(B);
    CHECK(std::set<std::pair<int, int>>(got.begin(), got.end()) == brute);
    CHECK(!brute.empty());

    // Conforming mesh: two closed cells intersect iff they share a vertex
    std::set<std::pair<int, int>> vertex_sharing;
    for ( int ii = 0; ii < A.num_cells(); ++ii )
    {
        for ( int jj = ii + 1; jj < A.num_cells(); ++jj )
        {
            bool share = false;
            for ( int aa = 0; aa < 3; ++aa )
            {
                for ( int bb = 0; bb < 3; ++bb )
                {
                    share = share || (A.cells()(aa, ii) == A.cells()(bb, jj));
                }
            }
            if ( share )
            {
                vertex_sharing.insert({ii, jj});
            }
        }
    }
    std::vector<std::pair<int, int>> self_pairs = A.cell_tree().self_collision_pairs();
    CHECK(std::set<std::pair<int, int>>(self_pairs.begin(), self_pairs.end()) == vertex_sharing);
}
