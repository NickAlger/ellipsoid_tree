// SimplexMesh: point location, closest points, and mesh-vs-ellipsoid
//
// A SimplexMesh wraps vertices and cells with spatial indexes: locate_points
// returns the containing cell and barycentric coordinates of each query
// point, closest_points projects exterior points onto the mesh, and
// cells_intersecting finds every cell touching an ellipsoid. In the figure,
// cells hit by the ellipsoid are vermillion, the exterior query points are
// green with segments to their closest mesh points, and the located interior
// points are blue.

#include <cstdio>
#include <random>

#include "etree/etree.hpp"
#include "etree/plot2d.hpp"

using namespace etree;

namespace {

// Structured triangulation of the unit square: m x m quads, two cells each
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> unit_square_mesh( int m )
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

} // end anonymous namespace

int main()
{
    auto [vertices, cells] = unit_square_mesh(10);
    SimplexMesh mesh(vertices, cells);
    std::printf("mesh: %d vertices, %d cells, %d boundary faces\n",
                mesh.num_vertices(), mesh.num_cells(), mesh.num_boundary_faces());

    // Point location: containing cell + barycentric coordinates
    Eigen::MatrixXd interior(2, 3);
    interior << 0.23, 0.51, 0.87,
                0.34, 0.72, 0.12;
    auto [cell_inds, coords] = mesh.locate_points(interior);
    for ( int ii = 0; ii < 3; ++ii )
    {
        std::printf("point (%.2f, %.2f) is in cell %d with barycentric (%.3f, %.3f, %.3f)\n",
                    interior(0, ii), interior(1, ii), cell_inds(ii),
                    coords(0, ii), coords(1, ii), coords(2, ii));
    }

    // Closest points on the mesh for exterior queries
    Eigen::MatrixXd exterior(2, 2);
    exterior << 1.35, -0.20,
                0.30, 1.25;
    Eigen::MatrixXd projected = mesh.closest_points(exterior);
    for ( int ii = 0; ii < 2; ++ii )
    {
        std::printf("closest mesh point to (%+.2f, %+.2f) is (%.3f, %.3f)\n",
                    exterior(0, ii), exterior(1, ii), projected(0, ii), projected(1, ii));
    }

    // Mesh cells intersecting an ellipsoid
    const double th = 0.5;
    Eigen::Matrix2d R;
    R << std::cos(th), -std::sin(th),
         std::sin(th),  std::cos(th);
    Ellipsoid E{Eigen::Vector2d(0.55, 0.45),
                R * Eigen::Vector2d(0.09, 0.004).asDiagonal() * R.transpose()};
    std::vector<int> hit_cells = mesh.cells_intersecting(E, 1.0);
    std::printf("the ellipsoid intersects %d of %d cells\n",
                static_cast<int>(hit_cells.size()), mesh.num_cells());

    Plot2D fig;
    DrawTreeOptions base;
    base.node_boxes   = false;
    base.object_style = Style{colors::gray(), 0.7, colors::transparent()};
    draw_tree(fig, mesh.cell_tree(), base);
    draw_elements(fig, mesh.cell_tree(), hit_cells,
                  Style{colors::vermillion(), 1.2, with_alpha(colors::vermillion(), 0.35)});
    fig.add(E, 1.0, Style{colors::blue(), 2.0, colors::transparent()});
    for ( int ii = 0; ii < 3; ++ii )
    {
        fig.add_marker(interior.col(ii), 4.0, Style{colors::transparent(), 0.0, colors::blue()});
    }
    for ( int ii = 0; ii < 2; ++ii )
    {
        fig.add(Segment{exterior.col(ii), projected.col(ii)},
                Style{colors::green(), 1.6, colors::transparent()});
        fig.add_marker(exterior.col(ii), 4.0, Style{colors::transparent(), 0.0, colors::green()});
    }
    fig.save_svg("mesh_queries.svg", 780);
    return 0;
}
