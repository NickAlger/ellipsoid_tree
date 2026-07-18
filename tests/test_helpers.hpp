// SPDX-License-Identifier: MIT
#pragma once

#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <Eigen/Dense>

namespace test_helpers {

inline Eigen::VectorXd randn_vector( int d, std::mt19937& gen, double scale = 1.0 )
{
    std::normal_distribution<double> dist(0.0, scale);
    Eigen::VectorXd v(d);
    for ( int ii = 0; ii < d; ++ii )
    {
        v(ii) = dist(gen);
    }
    return v;
}

inline Eigen::MatrixXd randn_matrix( int rows, int cols, std::mt19937& gen, double scale = 1.0 )
{
    std::normal_distribution<double> dist(0.0, scale);
    Eigen::MatrixXd A(rows, cols);
    for ( int jj = 0; jj < cols; ++jj )
    {
        for ( int ii = 0; ii < rows; ++ii )
        {
            A(ii, jj) = dist(gen);
        }
    }
    return A;
}

inline Eigen::MatrixXd random_rotation( int d, std::mt19937& gen )
{
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(randn_matrix(d, d, gen));
    Eigen::MatrixXd Q = qr.householderQ();
    if ( Q.determinant() < 0.0 )
    {
        Q.col(0) *= -1.0;
    }
    return Q;
}

// SPD matrix with eigenvalues log-uniform in [eig_lo, eig_hi].
inline Eigen::MatrixXd random_spd( int d, std::mt19937& gen, double eig_lo, double eig_hi )
{
    std::uniform_real_distribution<double> unif(std::log(eig_lo), std::log(eig_hi));
    Eigen::VectorXd eigs(d);
    for ( int ii = 0; ii < d; ++ii )
    {
        eigs(ii) = std::exp(unif(gen));
    }
    Eigen::MatrixXd Q = random_rotation(d, gen);
    return Q * eigs.asDiagonal() * Q.transpose();
}

inline Eigen::MatrixXd matrix_sqrt_spd( const Eigen::MatrixXd& A )
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    return es.eigenvectors()
           * es.eigenvalues().cwiseSqrt().asDiagonal()
           * es.eigenvectors().transpose();
}

// Uniform (Dirichlet(1,...,1)) barycentric coordinates: alpha >= 0, sum = 1.
inline Eigen::VectorXd random_simplex_alpha( int K, std::mt19937& gen )
{
    std::exponential_distribution<double> dist(1.0);
    Eigen::VectorXd alpha(K);
    for ( int ii = 0; ii < K; ++ii )
    {
        alpha(ii) = dist(gen);
    }
    return alpha / alpha.sum();
}

inline double mahalanobis_sq( const Eigen::VectorXd& p,
                              const Eigen::VectorXd& mu,
                              const Eigen::MatrixXd& Sigma )
{
    Eigen::VectorXd z = p - mu;
    return Sigma.ldlt().solve(z).dot(z);
}

// Minimum of f over a regular grid on the box [lo, hi], n points per axis,
// boundary included.
template <class F>
double grid_min_over_box( F&& f, const Eigen::VectorXd& lo, const Eigen::VectorXd& hi, int n )
{
    const int d = lo.size();
    std::vector<int> idx(d, 0);
    Eigen::VectorXd x(d);
    double best = std::numeric_limits<double>::infinity();
    while ( true )
    {
        for ( int kk = 0; kk < d; ++kk )
        {
            x(kk) = lo(kk) + (hi(kk) - lo(kk)) * idx[kk] / static_cast<double>(n - 1);
        }
        best = std::min(best, f(x));

        int kk = 0;
        while ( kk < d )
        {
            idx[kk] += 1;
            if ( idx[kk] < n )
            {
                break;
            }
            idx[kk] = 0;
            kk += 1;
        }
        if ( kk == d )
        {
            break;
        }
    }
    return best;
}

// Minimum of f over the barycentric grid {alpha = c / N : c integer, sum c = N}
// on the simplex conv(V), boundary faces included.
template <class F>
double grid_min_over_simplex( F&& f, const Eigen::MatrixXd& V, int N )
{
    const int K = V.cols();
    std::vector<int> counts(K, 0);
    double best = std::numeric_limits<double>::infinity();

    std::function<void(int, int)> recurse = [&](int pos, int remaining)
    {
        if ( pos == K - 1 )
        {
            counts[pos] = remaining;
            Eigen::VectorXd alpha(K);
            for ( int ii = 0; ii < K; ++ii )
            {
                alpha(ii) = counts[ii] / static_cast<double>(N);
            }
            best = std::min(best, f(Eigen::VectorXd(V * alpha)));
            return;
        }
        for ( int vv = 0; vv <= remaining; ++vv )
        {
            counts[pos] = vv;
            recurse(pos + 1, remaining - vv);
        }
    };
    recurse(0, N);
    return best;
}

} // end namespace test_helpers
