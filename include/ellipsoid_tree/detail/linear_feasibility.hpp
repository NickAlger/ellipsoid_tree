#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_tree — https://github.com/NickAlger/ellipsoid_tree

// Feasibility of the standard-form system {x >= 0 : A x = b}, decided by a
// phase-I dense tableau simplex method with Bland's rule (anti-cycling).
// Intended for the tiny systems arising from polytope-polytope intersection
// (a handful of rows and a few dozen columns); everything is dense.
//
// Verdicts carry the usual epsilon-semantics: rows are scaled to unit
// infinity norm and the system counts as feasible when the phase-I objective
// (total artificial-variable mass, i.e. the residual of the best point found)
// is at most feasibility_tol. Configurations within that tolerance of exact
// tangency may resolve either way.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

namespace ellipsoid_tree {
namespace detail {

inline bool linear_feasibility( Eigen::MatrixXd A,
                                Eigen::VectorXd b,
                                double feasibility_tol = 1e-9 )
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const double pivot_tol = 1e-11;

    // Scale each row of [A | b] to unit infinity norm; flip signs so b >= 0.
    for ( int ii = 0; ii < m; ++ii )
    {
        const double scale = std::max(A.row(ii).cwiseAbs().maxCoeff(), std::abs(b(ii)));
        if ( scale > 0.0 )
        {
            A.row(ii) /= scale;
            b(ii)     /= scale;
        }
        if ( b(ii) < 0.0 )
        {
            A.row(ii) = -A.row(ii);
            b(ii)     = -b(ii);
        }
    }

    // Tableau columns: [x (n) | artificials (m) | rhs]; artificials start basic.
    Eigen::MatrixXd T(m, n + m + 1);
    T.leftCols(n)      = A;
    T.middleCols(n, m) = Eigen::MatrixXd::Identity(m, m);
    T.col(n + m)       = b;

    std::vector<int> basis(m);
    std::iota(basis.begin(), basis.end(), n);

    // Phase-I reduced costs (costs: 0 on x, 1 on artificials):
    // d_j = c_j - sum_i T(i, j), which is 0 on the artificial columns.
    Eigen::RowVectorXd dcost(n + m);
    for ( int jj = 0; jj < n + m; ++jj )
    {
        dcost(jj) = ((jj >= n) ? 1.0 : 0.0) - T.col(jj).sum();
    }

    const int max_iterations = 50 * (n + m + 5); // backstop; Bland's rule terminates
    for ( int iter = 0; iter < max_iterations; ++iter )
    {
        // Bland entering rule: smallest column index with negative reduced cost
        int enter = -1;
        for ( int jj = 0; jj < n + m; ++jj )
        {
            if ( dcost(jj) < -pivot_tol )
            {
                enter = jj;
                break;
            }
        }
        if ( enter < 0 )
        {
            break; // optimal
        }

        // Ratio test; Bland tie-break on the smallest leaving variable index
        int    leave      = -1;
        double best_ratio = std::numeric_limits<double>::infinity();
        for ( int ii = 0; ii < m; ++ii )
        {
            if ( T(ii, enter) > pivot_tol )
            {
                const double ratio = T(ii, n + m) / T(ii, enter);
                if ( ratio < best_ratio - 1e-15
                     || (ratio < best_ratio + 1e-15
                         && (leave < 0 || basis[ii] < basis[leave])) )
                {
                    best_ratio = ratio;
                    leave      = ii;
                }
            }
        }
        if ( leave < 0 )
        {
            break; // unbounded: cannot happen for a phase-I objective; bail safely
        }

        // Pivot
        T.row(leave) /= T(leave, enter);
        for ( int ii = 0; ii < m; ++ii )
        {
            if ( ii != leave && T(ii, enter) != 0.0 )
            {
                T.row(ii) -= T(ii, enter) * T.row(leave);
            }
        }
        dcost -= dcost(enter) * T.row(leave).head(n + m);
        basis[leave] = enter;
    }

    // Objective value = total mass left on basic artificial variables
    double artificial_mass = 0.0;
    for ( int ii = 0; ii < m; ++ii )
    {
        if ( basis[ii] >= n )
        {
            artificial_mass += std::max(T(ii, n + m), 0.0);
        }
    }
    return artificial_mass <= feasibility_tol;
}

} // end namespace detail
} // end namespace ellipsoid_tree
