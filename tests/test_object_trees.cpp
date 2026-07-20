// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "ellipsoid_tree/object_tree.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <random>
#include <set>

using namespace ellipsoid_tree;
namespace th = test_helpers;

namespace {

std::vector<int> sorted( std::vector<int> v )
{
    std::sort(v.begin(), v.end());
    return v;
}

std::vector<Box> random_box_objects( int d, int n, std::mt19937& gen )
{
    std::vector<Box> out(n);
    for ( int ii = 0; ii < n; ++ii )
    {
        Eigen::VectorXd c = th::randn_vector(d, gen, 1.5);
        Eigen::VectorXd w = th::randn_vector(d, gen, 0.2).cwiseAbs()
                            + Eigen::VectorXd::Constant(d, 0.05);
        out[ii] = Box{c - w, c + w};
    }
    return out;
}

std::vector<Ball> random_ball_objects( int d, int n, std::mt19937& gen )
{
    std::uniform_real_distribution<double> ur(0.05, 0.5);
    std::vector<Ball> out(n);
    for ( int ii = 0; ii < n; ++ii )
    {
        out[ii] = Ball{th::randn_vector(d, gen, 1.5), ur(gen)};
    }
    return out;
}

std::vector<Ellipsoid> random_ellipsoid_objects( int d, int n, std::mt19937& gen )
{
    std::vector<Ellipsoid> out(n);
    for ( int ii = 0; ii < n; ++ii )
    {
        out[ii] = Ellipsoid{th::randn_vector(d, gen, 1.5), 0.05 * th::random_spd(d, gen, 0.2, 3.0)};
    }
    return out;
}

std::vector<Simplex> random_simplex_objects( int d, int n, std::mt19937& gen )
{
    std::vector<Simplex> out(n);
    for ( int ii = 0; ii < n; ++ii )
    {
        Eigen::VectorXd c = th::randn_vector(d, gen, 1.5);
        out[ii] = Simplex{(0.4 * th::randn_matrix(d, d + 1, gen)).colwise() + c};
    }
    return out;
}

// Compare tree.collisions(query...) against a brute-force scan with the
// (already validated) free intersection predicates.
template <class Tree, class BruteNarrow, class... QueryAndTau>
void check_query( const Tree& tree, BruteNarrow&& brute_narrow, const QueryAndTau&... query_args )
{
    std::vector<int> brute;
    for ( int ii = 0; ii < tree.size(); ++ii )
    {
        if ( brute_narrow(ii) )
        {
            brute.push_back(ii);
        }
    }
    CHECK(sorted(tree.collisions(query_args...)) == brute);
}

} // end anonymous namespace

TEST_CASE("BoxTree queries match brute force")
{
    std::mt19937 gen(3001);
    for ( int rep = 0; rep < 4; ++rep )
    {
        const int d = 2 + (rep % 2);
        BoxTree T(random_box_objects(d, 120, gen));

        for ( int qq = 0; qq < 15; ++qq )
        {
            Eigen::VectorXd p = th::randn_vector(d, gen, 1.5);
            check_query(T, [&]( int ii ) { return intersects(p, T.object(ii)); }, p);

            Box qb = random_box_objects(d, 1, gen)[0];
            check_query(T, [&]( int ii ) { return intersects(T.object(ii), qb); }, qb);

            Ball qball = random_ball_objects(d, 1, gen)[0];
            check_query(T, [&]( int ii ) { return intersects(T.object(ii), qball); }, qball);

            Ellipsoid qe{th::randn_vector(d, gen, 1.5), 0.2 * th::random_spd(d, gen, 0.2, 3.0)};
            const double tau = 0.8 + 0.1 * qq;
            check_query(T, [&]( int ii ) { return intersects(T.object(ii), qe, tau); }, qe, tau);

            Segment qs{th::randn_vector(d, gen, 2.0), th::randn_vector(d, gen, 2.0)};
            check_query(T, [&]( int ii ) { return intersects(qs, T.object(ii)); }, qs);

            Halfspace qh{th::randn_vector(d, gen), 0.3 * qq - 2.0};
            check_query(T, [&]( int ii ) { return intersects(T.object(ii), qh); }, qh);
        }
    }
}

TEST_CASE("BallTree queries match brute force")
{
    std::mt19937 gen(3002);
    for ( int rep = 0; rep < 4; ++rep )
    {
        const int d = 2 + (rep % 2);
        BallTree T(random_ball_objects(d, 120, gen));

        for ( int qq = 0; qq < 15; ++qq )
        {
            Eigen::VectorXd p = th::randn_vector(d, gen, 1.5);
            check_query(T, [&]( int ii ) { return intersects(p, T.object(ii)); }, p);

            Box qb = random_box_objects(d, 1, gen)[0];
            check_query(T, [&]( int ii ) { return intersects(qb, T.object(ii)); }, qb);

            Ball qball = random_ball_objects(d, 1, gen)[0];
            check_query(T, [&]( int ii ) { return intersects(T.object(ii), qball); }, qball);

            Ellipsoid qe{th::randn_vector(d, gen, 1.5), 0.2 * th::random_spd(d, gen, 0.2, 3.0)};
            const double tau = 0.8 + 0.1 * qq;
            check_query(T, [&]( int ii ) { return intersects(T.object(ii), qe, tau); }, qe, tau);

            Simplex qsx = random_simplex_objects(d, 1, gen)[0];
            check_query(T, [&]( int ii ) { return intersects(T.object(ii), qsx); }, qsx);

            Segment qs{th::randn_vector(d, gen, 2.0), th::randn_vector(d, gen, 2.0)};
            check_query(T, [&]( int ii ) { return intersects(qs, T.object(ii)); }, qs);

            Halfspace qh{th::randn_vector(d, gen), 0.3 * qq - 2.0};
            check_query(T, [&]( int ii ) { return intersects(T.object(ii), qh); }, qh);
        }
    }
}

TEST_CASE("EllipsoidTree queries match brute force")
{
    std::mt19937 gen(3003);
    for ( int rep = 0; rep < 4; ++rep )
    {
        const int d = 2 + (rep % 2);
        const double tau_build = 1.2;
        EllipsoidTree T(random_ellipsoid_objects(d, 120, gen), tau_build);

        for ( int qq = 0; qq < 12; ++qq )
        {
            const double tau = (qq % 3 == 0) ? tau_build : 0.7; // build tau and a smaller one

            Eigen::VectorXd p = th::randn_vector(d, gen, 1.5);
            check_query(T, [&]( int ii ) { return intersects(p, T.object(ii), tau); }, p, tau);

            Box qb = random_box_objects(d, 1, gen)[0];
            check_query(T, [&]( int ii ) { return intersects(qb, T.object(ii), tau); }, qb, tau);

            Ball qball = random_ball_objects(d, 1, gen)[0];
            check_query(T, [&]( int ii ) { return intersects(qball, T.object(ii), tau); }, qball, tau);

            Ellipsoid qe{th::randn_vector(d, gen, 1.5), 0.2 * th::random_spd(d, gen, 0.2, 3.0)};
            check_query(T, [&]( int ii ) { return intersects(T.object(ii), qe, tau); }, qe, tau);

            Simplex qsx = random_simplex_objects(d, 1, gen)[0];
            check_query(T, [&]( int ii ) { return intersects(T.object(ii), qsx, tau); }, qsx, tau);

            Segment qs{th::randn_vector(d, gen, 2.0), th::randn_vector(d, gen, 2.0)};
            check_query(T, [&]( int ii ) { return intersects(qs, T.object(ii), tau); }, qs, tau);

            Halfspace qh{th::randn_vector(d, gen), 0.3 * qq - 2.0};
            check_query(T, [&]( int ii ) { return intersects(T.object(ii), qh, tau); }, qh, tau);
        }

        // Default-tau overloads use the build tau
        Eigen::VectorXd p0 = th::randn_vector(d, gen);
        CHECK(T.collisions(p0) == T.collisions(p0, tau_build));
    }
}

TEST_CASE("EllipsoidTree tau discipline: larger tau throws until rebuild")
{
    std::mt19937 gen(3004);
    const int d = 2;
    std::vector<Ellipsoid> objs = random_ellipsoid_objects(d, 60, gen);
    EllipsoidTree T(objs, 1.0);

    Eigen::VectorXd p = th::randn_vector(d, gen);
    CHECK_NOTHROW(T.collisions(p, 0.5));
    CHECK_THROWS_AS(T.collisions(p, 1.5), std::invalid_argument);
    CHECK_THROWS_AS(T.collisions(p, -1.0), std::invalid_argument);

    T.rebuild(1.5);
    CHECK(T.tau() == 1.5);
    std::vector<int> brute;
    for ( int ii = 0; ii < T.size(); ++ii )
    {
        if ( intersects(p, T.object(ii), 1.5) )
        {
            brute.push_back(ii);
        }
    }
    CHECK(sorted(T.collisions(p, 1.5)) == brute);
}

TEST_CASE("exact ellipsoid pruning does not change results")
{
    std::mt19937 gen(3005);
    for ( int rep = 0; rep < 3; ++rep )
    {
        const int d = 2 + (rep % 2);
        SimplexTree TS(random_simplex_objects(d, 100, gen));
        BallTree TB(random_ball_objects(d, 100, gen));

        for ( int qq = 0; qq < 20; ++qq )
        {
            // Elongated tilted ellipsoids: the case where tier 2 prunes hardest
            Eigen::MatrixXd Q = th::random_rotation(d, gen);
            Eigen::VectorXd eigs = Eigen::VectorXd::Constant(d, 0.001);
            eigs(0) = 2.0;
            Ellipsoid qe{th::randn_vector(d, gen, 1.5),
                         Eigen::MatrixXd(Q * eigs.asDiagonal() * Q.transpose())};

            SimplexTree& ts = TS;
            ts.exact_ellipsoid_pruning = true;
            std::vector<int> with_exact = sorted(ts.collisions(qe, 1.0));
            ts.exact_ellipsoid_pruning = false;
            CHECK(sorted(ts.collisions(qe, 1.0)) == with_exact);
            ts.exact_ellipsoid_pruning = true;

            BallTree& tb = TB;
            tb.exact_ellipsoid_pruning = true;
            std::vector<int> with_exact_b = sorted(tb.collisions(qe, 1.0));
            tb.exact_ellipsoid_pruning = false;
            CHECK(sorted(tb.collisions(qe, 1.0)) == with_exact_b);
            tb.exact_ellipsoid_pruning = true;
        }
    }
}

TEST_CASE("SimplexTree queries match brute force")
{
    std::mt19937 gen(3006);
    for ( int rep = 0; rep < 4; ++rep )
    {
        const int d = 2 + (rep % 2);
        SimplexTree T(random_simplex_objects(d, 120, gen));

        for ( int qq = 0; qq < 15; ++qq )
        {
            Eigen::VectorXd p = th::randn_vector(d, gen, 1.5);
            check_query(T, [&]( int ii ) { return intersects(p, T.object(ii)); }, p);

            Ball qball = random_ball_objects(d, 1, gen)[0];
            check_query(T, [&]( int ii ) { return intersects(qball, T.object(ii)); }, qball);

            Ellipsoid qe{th::randn_vector(d, gen, 1.5), 0.2 * th::random_spd(d, gen, 0.2, 3.0)};
            const double tau = 0.8 + 0.1 * qq;
            check_query(T, [&]( int ii ) { return intersects(qe, T.object(ii), tau); }, qe, tau);

            Segment qs{th::randn_vector(d, gen, 2.0), th::randn_vector(d, gen, 2.0)};
            check_query(T, [&]( int ii ) { return intersects(qs, T.object(ii)); }, qs);

            Halfspace qh{th::randn_vector(d, gen), 0.3 * qq - 2.0};
            check_query(T, [&]( int ii ) { return intersects(T.object(ii), qh); }, qh);
        }
    }
}

TEST_CASE("SimplexTree from a structured triangulation: point location")
{
    // Unit square, m x m quads, two triangles each
    const int m = 8;
    const int nv = (m + 1) * (m + 1);
    Eigen::MatrixXd vertices(2, nv);
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
    SimplexTree T(vertices, cells);
    CHECK(T.size() == 2 * m * m);

    std::mt19937 gen(3007);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for ( int qq = 0; qq < 200; ++qq )
    {
        Eigen::Vector2d p(unif(gen), unif(gen));
        const int e = T.first_collision(p);
        CHECK(e >= 0);
        CHECK(intersects(Eigen::VectorXd(p), T.object(e)));

        // Barycentric coordinates: nonnegative, sum to one, reproduce p
        Eigen::VectorXd alpha = T.affine_coordinates(e, p);
        CHECK((alpha.array() >= -1e-12).all());
        CHECK(alpha.sum() == doctest::Approx(1.0).epsilon(1e-9));
        CHECK((T.object(e).V * alpha - p).norm() < 1e-12);

        // Points outside the mesh find nothing
        Eigen::Vector2d outside(2.0 + unif(gen), unif(gen));
        CHECK(T.first_collision(outside) == -1);
    }
}

TEST_CASE("collision_pairs and self_collision_pairs match brute force")
{
    std::mt19937 gen(3008);
    const int d = 2;
    std::vector<Ellipsoid> eobjs = random_ellipsoid_objects(d, 80, gen);
    std::vector<Simplex> sobjs = random_simplex_objects(d, 70, gen);
    std::vector<Ball> bobjs = random_ball_objects(d, 60, gen);
    std::vector<Box> xobjs = random_box_objects(d, 60, gen);

    EllipsoidTree TE(eobjs, 1.0);
    EllipsoidTree TE2(random_ellipsoid_objects(d, 50, gen), 0.8);
    SimplexTree   TS(sobjs);
    BallTree      TB(bobjs);
    BoxTree       TX(xobjs);

    auto as_set = []( const std::vector<std::pair<int, int>>& v )
    { return std::set<std::pair<int, int>>(v.begin(), v.end()); };

    // EllipsoidTree x EllipsoidTree with different build taus
    {
        std::set<std::pair<int, int>> brute;
        for ( int ii = 0; ii < TE.size(); ++ii )
        {
            for ( int jj = 0; jj < TE2.size(); ++jj )
            {
                Ellipsoid Ea{TE.object(ii).mu, TE.tau() * TE.tau() * TE.object(ii).Sigma};
                Ellipsoid Eb{TE2.object(jj).mu, TE2.tau() * TE2.tau() * TE2.object(jj).Sigma};
                if ( intersects(Ea, Eb, 1.0) )
                {
                    brute.insert({ii, jj});
                }
            }
        }
        CHECK(as_set(collision_pairs(TE, TE2)) == brute);
    }

    // SimplexTree x EllipsoidTree (and the flipped order)
    {
        std::set<std::pair<int, int>> brute;
        for ( int ii = 0; ii < TS.size(); ++ii )
        {
            for ( int jj = 0; jj < TE.size(); ++jj )
            {
                if ( intersects(TE.object(jj), TS.object(ii), TE.tau()) )
                {
                    brute.insert({ii, jj});
                }
            }
        }
        CHECK(as_set(collision_pairs(TS, TE)) == brute);

        std::set<std::pair<int, int>> brute_flipped;
        for ( const std::pair<int, int>& pr : brute )
        {
            brute_flipped.insert({pr.second, pr.first});
        }
        CHECK(as_set(collision_pairs(TE, TS)) == brute_flipped);
    }

    // BallTree x EllipsoidTree, BoxTree x EllipsoidTree
    {
        std::set<std::pair<int, int>> brute;
        for ( int ii = 0; ii < TB.size(); ++ii )
        {
            for ( int jj = 0; jj < TE.size(); ++jj )
            {
                if ( intersects(TB.object(ii), TE.object(jj), TE.tau()) )
                {
                    brute.insert({ii, jj});
                }
            }
        }
        CHECK(as_set(collision_pairs(TB, TE)) == brute);
    }
    {
        std::set<std::pair<int, int>> brute;
        for ( int ii = 0; ii < TX.size(); ++ii )
        {
            for ( int jj = 0; jj < TE.size(); ++jj )
            {
                if ( intersects(TX.object(ii), TE.object(jj), TE.tau()) )
                {
                    brute.insert({ii, jj});
                }
            }
        }
        CHECK(as_set(collision_pairs(TX, TE)) == brute);
    }

    // Homogeneous pairs and self-collisions
    {
        std::set<std::pair<int, int>> brute;
        for ( int ii = 0; ii < TX.size(); ++ii )
        {
            for ( int jj = 0; jj < TB.size(); ++jj )
            {
                if ( intersects(TX.object(ii), TB.object(jj)) )
                {
                    brute.insert({ii, jj});
                }
            }
        }
        CHECK(as_set(collision_pairs(TX, TB)) == brute);
    }
    {
        std::set<std::pair<int, int>> brute;
        for ( int ii = 0; ii < TE.size(); ++ii )
        {
            for ( int jj = ii + 1; jj < TE.size(); ++jj )
            {
                if ( intersects(TE.object(ii), TE.object(jj), TE.tau()) )
                {
                    brute.insert({ii, jj});
                }
            }
        }
        CHECK(as_set(TE.self_collision_pairs()) == brute);
    }
    {
        std::set<std::pair<int, int>> brute;
        for ( int ii = 0; ii < TB.size(); ++ii )
        {
            for ( int jj = ii + 1; jj < TB.size(); ++jj )
            {
                if ( intersects(TB.object(ii), TB.object(jj)) )
                {
                    brute.insert({ii, jj});
                }
            }
        }
        CHECK(as_set(TB.self_collision_pairs()) == brute);
    }
}

TEST_CASE("batched queries agree with per-query loops across thread counts")
{
    std::mt19937 gen(3009);
    const int d = 2;
    EllipsoidTree T(random_ellipsoid_objects(d, 100, gen), 1.0);

    std::vector<Ball> queries = random_ball_objects(d, 40, gen);
    std::vector<std::vector<int>> serial(queries.size());
    for ( size_t ii = 0; ii < queries.size(); ++ii )
    {
        serial[ii] = T.collisions(queries[ii]);
    }
    CHECK(T.collisions_batch(queries, 1) == serial);
    CHECK(T.collisions_batch(queries, 4) == serial);
    CHECK(T.collisions_batch(queries, 0.7, 4)
          == [&]{ std::vector<std::vector<int>> out(queries.size());
                  for ( size_t ii = 0; ii < queries.size(); ++ii )
                  { out[ii] = T.collisions(queries[ii], 0.7); }
                  return out; }());

    Eigen::MatrixXd points = th::randn_matrix(d, 50, gen);
    std::vector<std::vector<int>> serial_pts(points.cols());
    for ( int ii = 0; ii < points.cols(); ++ii )
    {
        serial_pts[ii] = T.collisions(points.col(ii));
    }
    CHECK(T.point_collisions_batch(points, 3) == serial_pts);

    // Empty trees answer queries with empty results
    EllipsoidTree empty;
    CHECK(empty.size() == 0);
    BoxTree empty_boxes;
    CHECK(empty_boxes.collisions(Eigen::VectorXd::Zero(0)).empty());
}
