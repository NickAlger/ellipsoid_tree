// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "ellipsoid_tree/plot2d.hpp"

#include <cstdio>
#include <fstream>

using namespace ellipsoid_tree;

namespace {

const unsigned char* px( const RenderedImage& im, int x, int y )
{
    REQUIRE(x >= 0);
    REQUIRE(x < im.width);
    REQUIRE(y >= 0);
    REQUIRE(y < im.height);
    return &im.rgb[3 * (static_cast<size_t>(y) * im.width + x)];
}

bool near_color( const unsigned char* p, int r, int g, int b, int tol )
{
    return std::abs(p[0] - r) <= tol && std::abs(p[1] - g) <= tol && std::abs(p[2] - b) <= tol;
}

} // end anonymous namespace

TEST_CASE("rasterizer: filled ball with antialiased edge on white background")
{
    Plot2D fig;
    fig.axes(false);
    fig.set_bounds(Box{Eigen::Vector2d(-1, -1), Eigen::Vector2d(1, 1)});
    // Radius chosen so the edge lands between pixel centers (0.5 would put it
    // exactly on a pixel boundary, leaving no fractionally covered pixels)
    fig.add(Ball{Eigen::Vector2d(0, 0), 0.47},
            Style{colors::transparent(), 0.0, Color{1, 0, 0, 1}});

    RenderedImage im = fig.render_rgb(400);
    // margins without axes: left 10, right 14 -> s = 376/2 = 188
    CHECK(im.width == 400);
    CHECK(im.height == 400); // 376 + 14 + 10

    CHECK(near_color(px(im, 198, 202), 255, 0, 0, 0));   // deep inside: exact fill
    CHECK(near_color(px(im, 20, 20), 255, 255, 255, 0)); // inside plot, far away: white
    CHECK(near_color(px(im, 2, 2), 255, 255, 255, 0));   // outside clip: white

    // Antialiasing: the row through the center crosses the edge with
    // intermediate red values
    int intermediate = 0;
    for ( int x = 10; x < 386; ++x )
    {
        const unsigned char g = px(im, x, 202)[1];
        if ( g > 10 && g < 245 )
        {
            intermediate += 1;
        }
    }
    CHECK(intermediate >= 1);
}

TEST_CASE("rasterizer: rotated thin ellipse fills along its major axis only")
{
    const double th = 0.78539816339744831; // 45 degrees
    Eigen::Matrix2d R;
    R << std::cos(th), -std::sin(th),
         std::sin(th),  std::cos(th);
    Eigen::Matrix2d Sigma = R * Eigen::Vector2d(4.0, 0.01).asDiagonal() * R.transpose();

    Plot2D fig;
    fig.axes(false);
    fig.set_bounds(Box{Eigen::Vector2d(-2, -2), Eigen::Vector2d(2, 2)});
    fig.add(Ellipsoid{Eigen::Vector2d(0, 0), Sigma}, 1.0,
            Style{colors::transparent(), 0.0, Color{1, 0, 0, 1}});

    RenderedImage im = fig.render_rgb(400);
    const double s = 376.0 / 4.0;
    auto X = [&]( double x ) { return static_cast<int>(10 + s * (x + 2.0)); };
    auto Y = [&]( double y ) { return static_cast<int>(14 + s * (2.0 - y)); };

    // 1.8 units along the major axis (inside; major semi-axis is 2)
    const double c = std::cos(th) * 1.8 / std::sqrt(2.0) * std::sqrt(2.0); // = 1.8*cos(45)
    CHECK(px(im, X(1.8 * std::cos(th)), Y(1.8 * std::sin(th)))[0] > 200);
    CHECK(px(im, X(1.8 * std::cos(th)), Y(1.8 * std::sin(th)))[1] < 80);
    (void)c;

    // 0.5 units along the minor axis (outside; minor semi-axis is 0.1)
    CHECK(near_color(px(im, X(-0.5 * std::sin(th)), Y(0.5 * std::cos(th))), 255, 255, 255, 5));
}

TEST_CASE("rasterizer: axis tick labels appear as dark pixels in the margin")
{
    Plot2D fig;
    fig.set_bounds(Box{Eigen::Vector2d(0, 0), Eigen::Vector2d(10, 4)});
    fig.add(Ball{Eigen::Vector2d(5, 2), 1.0}, Style{});
    RenderedImage im = fig.render_rgb(700);

    // Bottom margin holds the x tick labels
    int dark = 0;
    for ( int y = im.height - 40; y < im.height; ++y )
    {
        for ( int x = 0; x < im.width; ++x )
        {
            if ( px(im, x, y)[0] < 150 )
            {
                dark += 1;
            }
        }
    }
    CHECK(dark >= 10);
}

TEST_CASE("CG1 field rendering matches viridis at probe points")
{
    // Unit square, two triangles, field f(x, y) = x
    Eigen::MatrixXd vertices(2, 4);
    vertices << 0.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 1.0, 1.0;
    Eigen::MatrixXi cells(3, 2);
    cells.col(0) = Eigen::Vector3i(0, 1, 2);
    cells.col(1) = Eigen::Vector3i(0, 2, 3);
    SimplexMesh mesh(vertices, cells);
    Eigen::VectorXd values = vertices.row(0).transpose(); // f = x

    Plot2D fig;
    fig.axes(false);
    draw_cg1_field(fig, mesh, values);

    // SVG: two flat-fill polygons
    CHECK(fig.to_svg().find("<polygon") != std::string::npos);

    RenderedImage im = fig.render_rgb(300);
    // auto bounds: [-0.05, 1.05]^2, margins 10/14 -> s = 276/1.1
    const double s = 276.0 / 1.1;
    auto X = [&]( double x ) { return static_cast<int>(10 + s * (x + 0.05)); };
    auto Y = [&]( double y ) { return static_cast<int>(14 + s * (1.05 - y)); };

    double r, g, b;
    detail::viridis_rgb(0.5, r, g, b);
    CHECK(near_color(px(im, X(0.5), Y(0.5)),
                     static_cast<int>(255 * r), static_cast<int>(255 * g),
                     static_cast<int>(255 * b), 30));
    detail::viridis_rgb(0.1, r, g, b);
    CHECK(near_color(px(im, X(0.1), Y(0.5)),
                     static_cast<int>(255 * r), static_cast<int>(255 * g),
                     static_cast<int>(255 * b), 30));
    detail::viridis_rgb(0.9, r, g, b);
    CHECK(near_color(px(im, X(0.9), Y(0.5)),
                     static_cast<int>(255 * r), static_cast<int>(255 * g),
                     static_cast<int>(255 * b), 30));

    // Value-count mismatch throws
    Plot2D fig2;
    Eigen::VectorXd bad(3);
    CHECK_THROWS_AS(draw_cg1_field(fig2, mesh, bad), std::invalid_argument);
}

TEST_CASE("halfspace fill shades the feasible side in both backends")
{
    Plot2D fig;
    fig.axes(false);
    fig.set_bounds(Box{Eigen::Vector2d(-1, -1), Eigen::Vector2d(1, 1)});
    // {x : x <= 0.1}, feasible side shaded pure red
    fig.add(Halfspace{Eigen::Vector2d(1.0, 0.0), 0.1},
            Style{colors::black(), 2.0, Color{1, 0, 0, 1}});

    // SVG gains a fill polygon in addition to the boundary line
    const std::string svg = fig.to_svg(400);
    CHECK(svg.find("<polygon") != std::string::npos);
    CHECK(svg.find("<line") != std::string::npos);

    RenderedImage im = fig.render_rgb(400);
    const double s = 376.0 / 2.0;
    const int x_feasible   = static_cast<int>(10 + s * (-0.5 + 1.0)); // world x = -0.5
    const int x_infeasible = static_cast<int>(10 + s * (0.7 + 1.0));  // world x = +0.7
    const int y_mid        = static_cast<int>(14 + s * (1.0 - 0.0));  // world y = 0
    CHECK(near_color(px(im, x_feasible, y_mid), 255, 0, 0, 0));
    CHECK(near_color(px(im, x_infeasible, y_mid), 255, 255, 255, 0));
}

TEST_CASE("save_png writes a valid PNG file")
{
    Plot2D fig;
    fig.add(Ball{Eigen::Vector2d(0, 0), 1.0},
            Style{colors::blue(), 2.0, with_alpha(colors::blue(), 0.3)});
    const std::string path = "test_plot2d_output.png";
    fig.save_png(path, 300);

    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());
    unsigned char sig[8] = {0};
    in.read(reinterpret_cast<char*>(sig), 8);
    const unsigned char expected[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    for ( int ii = 0; ii < 8; ++ii )
    {
        CHECK(sig[ii] == expected[ii]);
    }
    in.close();
    std::remove(path.c_str());
}
