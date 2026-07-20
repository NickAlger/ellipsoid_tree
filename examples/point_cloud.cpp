// Which points of a cloud does an ellipsoid cover?
//
// A point cloud is a BallTree with zero radii: build the tree once, then ask
// which points lie inside a query ellipsoid. The broad phase prunes with the
// tree's boxes and the tiered exact ellipsoid-box test; the narrow phase is
// the Mahalanobis membership test. Covered points are drawn in vermillion.

#include <algorithm>
#include <cstdio>
#include <random>

#include "ellipsoid_tree/ellipsoid_tree.hpp"
#include "ellipsoid_tree/plot2d.hpp"

using namespace ellipsoid_tree;

int main()
{
    // A deterministic pseudo-random cloud of 60 points in [-1.5, 1.5]^2
    std::mt19937 gen(7);
    auto uniform = [&]() { return 3.0 * (gen() / 4294967296.0) - 1.5; };
    Eigen::MatrixXd points(2, 60);
    for ( int ii = 0; ii < 60; ++ii )
    {
        points.col(ii) = Eigen::Vector2d(uniform(), uniform());
    }
    BallTree cloud(points, Eigen::VectorXd::Zero(60));

    // A tilted ellipsoid: mu, Sigma = R diag(1.1^2, 0.35^2) R^T at -25 degrees
    const double th = -0.4363;
    Eigen::Matrix2d R;
    R << std::cos(th), -std::sin(th),
         std::sin(th),  std::cos(th);
    Ellipsoid E{Eigen::Vector2d(0.1, -0.1),
                R * Eigen::Vector2d(1.21, 0.1225).asDiagonal() * R.transpose()};

    std::vector<int> covered = cloud.collisions(E, 1.0);
    std::sort(covered.begin(), covered.end());

    std::printf("%d of %d points are covered by the ellipsoid:\n",
                static_cast<int>(covered.size()), cloud.size());
    for ( int idx : covered )
    {
        std::printf("  point %2d at (%+.3f, %+.3f)\n", idx, points(0, idx), points(1, idx));
    }

    Plot2D fig;
    fig.add(E, 1.0, Style{colors::blue(), 1.8, with_alpha(colors::blue(), 0.12)});
    for ( int ii = 0; ii < cloud.size(); ++ii )
    {
        fig.add_marker(points.col(ii), 3.0, Style{colors::transparent(), 0.0, colors::gray()});
    }
    for ( int idx : covered )
    {
        fig.add_marker(points.col(idx), 4.0, Style{colors::transparent(), 0.0, colors::vermillion()});
    }
    fig.save_svg("covered_points.svg", 700);
    return 0;
}
