#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_tree — https://github.com/NickAlger/ellipsoid_tree

// Brent's method for minimizing a scalar function on an interval, implemented
// from the algorithm's published description:
//
//   R. P. Brent, "Algorithms for Minimization without Derivatives",
//   Prentice-Hall, 1973, chapter 5 (procedure `localmin`).
//
// The method maintains a bracket [a, b] and the three best points seen so
// far. Each iteration tries a parabolic-interpolation step through those
// points, falling back to a golden-section step whenever the parabola is
// unreliable (step too large, or landing outside the bracket). This gives
// superlinear convergence on smooth functions while retaining the
// golden-section worst case. Function values are never computed at the
// bracket endpoints.

#include <cmath>
#include <limits>
#include <utility>

namespace ellipsoid_tree {
namespace detail {

struct MinimizeScalarOptions
{
    double xtol_rel       = 1.4901161193847656e-08; // sqrt(double machine epsilon)
    double xtol_abs       = 1.0e-11;
    int    max_iterations = 200;

    // Early exit: return as soon as any evaluated value is < stop_below.
    // Useful when only "is the minimum below this threshold?" matters.
    double stop_below     = -std::numeric_limits<double>::infinity();
};

struct MinimizeScalarResult
{
    double x;             // best point found
    double f;             // f(x)
    int    iterations;
    int    evaluations;
    bool   converged;     // met the interval tolerance
    bool   stopped_early; // hit the stop_below threshold
};

template <class F>
MinimizeScalarResult minimize_scalar( F&&                   f,
                                      double                a,
                                      double                b,
                                      MinimizeScalarOptions opts = MinimizeScalarOptions() )
{
    if ( b < a )
    {
        std::swap(a, b);
    }

    const double golden = 0.3819660112501051; // (3 - sqrt(5)) / 2

    MinimizeScalarResult result;
    result.converged     = false;
    result.stopped_early = false;

    double x  = a + golden * (b - a); // best point
    double w  = x;                    // second-best point
    double v  = x;                    // previous value of w
    double fx = f(x);
    double fw = fx;
    double fv = fx;

    int evaluations = 1;
    int iter        = 0;

    double d = 0.0; // most recent step
    double e = 0.0; // step taken two iterations ago (parabolic safeguard)

    if ( fx < opts.stop_below )
    {
        result.stopped_early = true;
    }

    while ( !result.stopped_early && iter < opts.max_iterations )
    {
        const double m = 0.5 * (a + b);
        const double t = opts.xtol_rel * std::abs(x) + opts.xtol_abs;

        if ( std::abs(x - m) <= 2.0 * t - 0.5 * (b - a) )
        {
            result.converged = true;
            break;
        }

        bool take_golden_step = true;

        if ( std::abs(e) > t )
        {
            // Fit a parabola through (v,fv), (w,fw), (x,fx); its minimum is
            // at x + p/q after the sign normalization below.
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if ( q > 0.0 )
            {
                p = -p;
            }
            else
            {
                q = -q;
            }

            const double e_old = e;
            e = d;

            // Accept the parabolic step only if it lands inside (a, b) and
            // moves less than half the step taken two iterations ago.
            if ( std::abs(p) < std::abs(0.5 * q * e_old) &&
                 p > q * (a - x) &&
                 p < q * (b - x) )
            {
                d = p / q;
                const double u_trial = x + d;
                if ( (u_trial - a) < 2.0 * t || (b - u_trial) < 2.0 * t )
                {
                    d = ( x < m ) ? t : -t; // stay clear of the bracket edges
                }
                take_golden_step = false;
            }
        }

        if ( take_golden_step )
        {
            e = ( x < m ) ? (b - x) : (a - x); // step into the larger segment
            d = golden * e;
        }

        // Evaluate at least a tolerance away from the current best point.
        double u;
        if ( std::abs(d) >= t )
        {
            u = x + d;
        }
        else
        {
            u = ( d > 0.0 ) ? x + t : x - t;
        }

        const double fu = f(u);
        evaluations += 1;
        iter        += 1;

        if ( fu <= fx )
        {
            if ( u < x ) { b = x; }
            else         { a = x; }
            v = w;  fv = fw;
            w = x;  fw = fx;
            x = u;  fx = fu;
        }
        else
        {
            if ( u < x ) { a = u; }
            else         { b = u; }
            if ( fu <= fw || w == x )
            {
                v = w;  fv = fw;
                w = u;  fw = fu;
            }
            else if ( fu <= fv || v == x || v == w )
            {
                v = u;  fv = fu;
            }
        }

        if ( fx < opts.stop_below )
        {
            result.stopped_early = true;
        }
    }

    result.x           = x;
    result.f           = fx;
    result.iterations  = iter;
    result.evaluations = evaluations;
    return result;
}

} // end namespace detail
} // end namespace ellipsoid_tree
