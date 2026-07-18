// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "etree/kd_tree.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <random>

using namespace etree;
namespace th = test_helpers;

TEST_CASE("KDTree k-nearest-neighbors against brute force")
{
    std::mt19937 gen(4001);
    for ( int d : {1, 2, 3, 5} )
    {
        for ( int n : {1, 10, 500, 2000} )
        {
            Eigen::MatrixXd pts = th::randn_matrix(d, n, gen);
            KDTree T(pts);
            CHECK(T.size() == n);

            for ( int k : {1, 3, 32, n, n + 7} )
            {
                const int k_eff = std::min(k, n);
                Eigen::MatrixXd queries = th::randn_matrix(d, 25, gen, 1.5);
                std::pair<Eigen::MatrixXi, Eigen::MatrixXd> res = T.query(queries, k);
                CHECK(res.first.rows() == k_eff);

                for ( int qq = 0; qq < queries.cols(); ++qq )
                {
                    // Brute force: sort all squared distances
                    std::vector<std::pair<double, int>> all(n);
                    for ( int ii = 0; ii < n; ++ii )
                    {
                        all[ii] = {(pts.col(ii) - queries.col(qq)).squaredNorm(), ii};
                    }
                    std::sort(all.begin(), all.end());

                    for ( int jj = 0; jj < k_eff; ++jj )
                    {
                        CHECK(res.first(jj, qq) == all[jj].second);
                        CHECK(res.second(jj, qq) == doctest::Approx(all[jj].first).epsilon(1e-12));
                        // Reported distance matches the reported index
                        CHECK((pts.col(res.first(jj, qq)) - queries.col(qq)).squaredNorm()
                              == doctest::Approx(res.second(jj, qq)).epsilon(1e-12));
                    }
                }
            }
        }
    }
}

TEST_CASE("KDTree with heavily duplicated points")
{
    std::mt19937 gen(4002);
    const int d = 2;
    Eigen::MatrixXd distinct = th::randn_matrix(d, 5, gen);
    Eigen::MatrixXd pts(d, 100);
    for ( int ii = 0; ii < 100; ++ii )
    {
        pts.col(ii) = distinct.col(ii % 5);
    }
    KDTree T(pts);

    Eigen::MatrixXd q = th::randn_matrix(d, 10, gen);
    std::pair<Eigen::MatrixXi, Eigen::MatrixXd> res = T.query(q, 20);
    for ( int qq = 0; qq < 10; ++qq )
    {
        // Distances must match the brute-force multiset (indices are ambiguous)
        std::vector<double> brute(100);
        for ( int ii = 0; ii < 100; ++ii )
        {
            brute[ii] = (pts.col(ii) - q.col(qq)).squaredNorm();
        }
        std::sort(brute.begin(), brute.end());
        for ( int jj = 0; jj < 20; ++jj )
        {
            CHECK(res.second(jj, qq) == doctest::Approx(brute[jj]).epsilon(1e-12));
        }
    }
}

TEST_CASE("KDTree query is deterministic across thread counts")
{
    std::mt19937 gen(4003);
    Eigen::MatrixXd pts = th::randn_matrix(3, 1500, gen);
    KDTree T(pts);
    Eigen::MatrixXd q = th::randn_matrix(3, 64, gen);

    std::pair<Eigen::MatrixXi, Eigen::MatrixXd> serial = T.query(q, 8, 1);
    std::pair<Eigen::MatrixXi, Eigen::MatrixXd> parallel = T.query(q, 8, 4);
    CHECK(serial.first == parallel.first);
    CHECK(serial.second == parallel.second);

    CHECK_THROWS_AS(T.query(q, 0), std::invalid_argument);
}
