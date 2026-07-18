// Batch picking: partition ellipsoids into non-overlapping groups
//
// pick_ellipsoid_batches partitions a family of ellipsoids into batches of
// mutually non-intersecting members, considering candidates in farthest-point
// order so each batch spreads across the domain. This is the workhorse of
// point-spread-function probing: impulse responses with non-overlapping
// supports can be probed in a single operator application, so fewer batches
// means fewer applications. Each batch gets one color; within a color, no
// two ellipsoids overlap.

#include <cstdio>
#include <random>

#include "etree/etree.hpp"
#include "etree/plot2d.hpp"

using namespace etree;

int main()
{
    // A jittered 8 x 8 grid of anisotropic ellipsoids, dense enough to overlap
    std::mt19937 gen(11);
    auto uniform = [&]() { return gen() / 4294967296.0; };
    std::vector<Ellipsoid> family;
    for ( int jj = 0; jj < 8; ++jj )
    {
        for ( int ii = 0; ii < 8; ++ii )
        {
            Eigen::Vector2d mu(ii + 0.7 * uniform(), jj + 0.7 * uniform());
            const double th = 3.1416 * uniform();
            Eigen::Matrix2d R;
            R << std::cos(th), -std::sin(th),
                 std::sin(th),  std::cos(th);
            const double a = 0.45 + 0.35 * uniform();
            const double b = 0.12 + 0.15 * uniform();
            family.push_back(Ellipsoid{mu, R * Eigen::Vector2d(a * a, b * b).asDiagonal() * R.transpose()});
        }
    }
    EllipsoidTree tree(family, /*tau=*/1.0);

    std::vector<std::vector<int>> batches = pick_ellipsoid_batches(tree);
    std::printf("%d ellipsoids were partitioned into %d batches with sizes:",
                tree.size(), static_cast<int>(batches.size()));
    for ( const std::vector<int>& batch : batches )
    {
        std::printf(" %d", static_cast<int>(batch.size()));
    }
    std::printf("\n");

    Plot2D fig;
    draw_batches(fig, tree, batches);
    fig.save_svg("batches.svg", 780);
    return 0;
}
