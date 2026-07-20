// SPDX-License-Identifier: MIT
#include "doctest/doctest.h"
#include "ellipsoid_tree/plot2d.hpp"
#include "ellipsoid_tree/batch_picker.hpp"
#include "test_helpers.hpp"

#include <random>

using namespace ellipsoid_tree;
namespace th = test_helpers;

namespace {

int count_occurrences( const std::string& hay, const std::string& needle )
{
    int count = 0;
    size_t pos = 0;
    while ( (pos = hay.find(needle, pos)) != std::string::npos )
    {
        count += 1;
        pos += needle.size();
    }
    return count;
}

// Extract the numeric value of the first `attr="..."` after `from`
double attr_value( const std::string& svg, const std::string& attr, size_t from = 0 )
{
    const size_t pos = svg.find(attr + "=\"", from);
    REQUIRE(pos != std::string::npos);
    return std::atof(svg.c_str() + pos + attr.size() + 2);
}

} // end anonymous namespace

TEST_CASE("Plot2D emits well-formed SVG with the expected elements")
{
    Plot2D fig;
    fig.add(Ellipsoid{Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d(Eigen::Vector2d(4.0, 1.0).asDiagonal())},
            1.0, Style{});
    fig.add(Ball{Eigen::Vector2d(3.0, 1.0), 0.5}, Style{});
    fig.add(Box{Eigen::Vector2d(-1.0, -1.0), Eigen::Vector2d(1.0, 1.0)}, Style{});
    Eigen::MatrixXd tri(2, 3);
    tri << 0.0, 1.0, 0.0,
           0.0, 0.0, 1.0;
    fig.add(Simplex{tri}, Style{});
    fig.add(Segment{Eigen::Vector2d(0, 0), Eigen::Vector2d(2, 2)}, Style{});
    fig.add(Halfspace{Eigen::Vector2d(1.0, 0.0), 0.5}, Style{});
    fig.add_marker(Eigen::Vector2d(0.5, 0.5), 3.0, Style{});

    const std::string svg = fig.to_svg(800);

    CHECK(count_occurrences(svg, "<svg") == 1);
    CHECK(count_occurrences(svg, "</svg>") == 1);
    CHECK(count_occurrences(svg, "<ellipse") == 2);  // ellipsoid + ball
    CHECK(count_occurrences(svg, "<polygon") == 1);  // triangle
    CHECK(count_occurrences(svg, "<polyline") >= 5); // segment + axis tick lines
    CHECK(count_occurrences(svg, "<circle") == 1);   // marker
    CHECK(count_occurrences(svg, "clipPath") >= 1);
    CHECK(count_occurrences(svg, "<text") >= 4);     // tick labels
    CHECK(count_occurrences(svg, "<line") == 1);     // halfspace boundary
}

TEST_CASE("Plot2D geometry survives the world-to-canvas transform")
{
    // Fixed bounds and width make the scale exactly computable
    Plot2D fig;
    fig.axes(false);
    fig.set_bounds(Box{Eigen::Vector2d(-5.0, -5.0), Eigen::Vector2d(5.0, 5.0)});

    // Axis-aligned ellipse with semi-axes 2 (x) and 1 (y)
    fig.add(Ellipsoid{Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d(Eigen::Vector2d(4.0, 1.0).asDiagonal())},
            1.0, Style{});

    const int width = 900;
    const std::string svg = fig.to_svg(width);
    // margins without axes: left 10, right 14 -> plot_w = width - 24
    const double s = (width - 24.0) / 10.0;

    const size_t epos = svg.find("<ellipse");
    REQUIRE(epos != std::string::npos);
    const double rx = attr_value(svg, "rx", epos);
    const double ry = attr_value(svg, "ry", epos);
    CHECK(rx == doctest::Approx(2.0 * s).epsilon(1e-9));
    CHECK(ry == doctest::Approx(1.0 * s).epsilon(1e-9));

    // A rotated ellipse reports its rotation angle (y-flip negates it)
    Plot2D fig2;
    fig2.axes(false);
    const double theta = 0.4;
    Eigen::Matrix2d R;
    R << std::cos(theta), -std::sin(theta),
         std::sin(theta),  std::cos(theta);
    Eigen::Matrix2d Sigma = R * Eigen::Vector2d(9.0, 1.0).asDiagonal() * R.transpose();
    fig2.add(Ellipsoid{Eigen::Vector2d(1.0, 2.0), Sigma}, 1.0, Style{});
    const std::string svg2 = fig2.to_svg(600);
    const size_t rpos = svg2.find("rotate(");
    REQUIRE(rpos != std::string::npos);
    const double deg = std::atof(svg2.c_str() + rpos + 7);
    CHECK(std::abs(deg - (-theta * 180.0 / 3.141592653589793)) < 1e-6);
}

TEST_CASE("draw_tree renders objects and the AABB hierarchy")
{
    std::mt19937 gen(8001);
    std::vector<Ellipsoid> objs(20);
    for ( int ii = 0; ii < 20; ++ii )
    {
        objs[ii] = Ellipsoid{th::randn_vector(2, gen), 0.05 * th::random_spd(2, gen, 0.3, 3.0)};
    }
    EllipsoidTree T(objs, 1.0);

    Plot2D fig;
    DrawTreeOptions opts;
    opts.leaf_boxes = true;
    draw_tree(fig, T, opts);
    const std::string svg = fig.to_svg();

    // 20 ellipses; node boxes: 19 internal + 20 leaf boxes = 39 rects (+ axes frame)
    CHECK(count_occurrences(svg, "<ellipse") == 20);
    CHECK(count_occurrences(svg, "<rect") == 39 + 3); // + background + clip + axes frame

    // max_depth = 0 draws only the root among internal boxes
    Plot2D fig2;
    DrawTreeOptions opts2;
    opts2.objects   = false;
    opts2.max_depth = 0;
    draw_tree(fig2, T, opts2);
    CHECK(count_occurrences(fig2.to_svg(), "<rect") == 1 + 3);

    // 3D trees are rejected
    std::vector<Ellipsoid> objs3(5);
    for ( int ii = 0; ii < 5; ++ii )
    {
        objs3[ii] = Ellipsoid{th::randn_vector(3, gen), th::random_spd(3, gen, 0.3, 3.0)};
    }
    EllipsoidTree T3(objs3, 1.0);
    Plot2D fig3;
    CHECK_THROWS_AS(draw_tree(fig3, T3), std::invalid_argument);
}

TEST_CASE("draw_kdtree renders markers and the expected splitting lines")
{
    // n = 7, block_size = 1: internal splits at the root and both halves -> 3 lines
    Eigen::MatrixXd pts(2, 7);
    pts << 0.1, 0.9, 0.35, 0.6, 0.2, 0.75, 0.5,
           0.8, 0.2, 0.55, 0.3, 0.1, 0.95, 0.45;
    KDTree T;
    T.block_size = 1;
    T.build(pts);

    Plot2D fig;
    fig.axes(false);
    draw_kdtree(fig, T);
    const std::string svg = fig.to_svg();
    CHECK(count_occurrences(svg, "<circle") == 7);
    CHECK(count_occurrences(svg, "<polyline") == 3);

    // Default block_size 32: no splits for small point sets, only markers
    KDTree T2(pts);
    Plot2D fig2;
    fig2.axes(false);
    draw_kdtree(fig2, T2);
    CHECK(count_occurrences(fig2.to_svg(), "<polyline") == 0);
}

TEST_CASE("draw_batches colors ellipsoids by batch")
{
    std::mt19937 gen(8002);
    std::vector<Ellipsoid> objs(60);
    for ( int ii = 0; ii < 60; ++ii )
    {
        objs[ii] = Ellipsoid{th::randn_vector(2, gen, 0.8), 0.03 * th::random_spd(2, gen, 0.3, 3.0)};
    }
    EllipsoidTree T(objs, 1.0);
    std::vector<std::vector<int>> batches = pick_ellipsoid_batches(T);
    REQUIRE(batches.size() >= 2);

    Plot2D fig;
    draw_batches(fig, T, batches);
    const std::string svg = fig.to_svg();
    CHECK(count_occurrences(svg, "<ellipse") == 60);

    // At least two distinct batch colors appear
    const Color c0 = palette_color(0);
    const Color c1 = palette_color(1);
    char buf0[40], buf1[40];
    std::snprintf(buf0, sizeof(buf0), "rgb(%d,%d,%d)",
                  (int)std::lround(255 * c0.r), (int)std::lround(255 * c0.g), (int)std::lround(255 * c0.b));
    std::snprintf(buf1, sizeof(buf1), "rgb(%d,%d,%d)",
                  (int)std::lround(255 * c1.r), (int)std::lround(255 * c1.g), (int)std::lround(255 * c1.b));
    CHECK(count_occurrences(svg, buf0) >= 1);
    CHECK(count_occurrences(svg, buf1) >= 1);
}

TEST_CASE("draw_elements highlights query results")
{
    std::mt19937 gen(8003);
    std::vector<Simplex> objs(30);
    for ( int ii = 0; ii < 30; ++ii )
    {
        Eigen::VectorXd c = th::randn_vector(2, gen);
        objs[ii] = Simplex{(0.3 * th::randn_matrix(2, 3, gen)).colwise() + c};
    }
    SimplexTree T(objs);
    Ellipsoid q{Eigen::Vector2d(0.0, 0.0), 0.2 * Eigen::Matrix2d::Identity()};
    std::vector<int> hits = T.collisions(q, 1.0);

    Plot2D fig;
    DrawTreeOptions base;
    base.node_boxes   = false;
    base.object_style = Style{colors::light_gray(), 1.0, colors::transparent()};
    draw_tree(fig, T, base);
    draw_elements(fig, T, hits, Style{colors::vermillion(), 1.6, with_alpha(colors::vermillion(), 0.3)});
    fig.add(q, 1.0, Style{colors::black(), 1.6, colors::transparent()});

    const std::string svg = fig.to_svg();
    CHECK(count_occurrences(svg, "<polygon") == 30 + static_cast<int>(hits.size()));
    CHECK(count_occurrences(svg, "<ellipse") == 1);
}

TEST_CASE("axis tick labels reflect the physical extents")
{
    Plot2D fig;
    fig.set_bounds(Box{Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(10.0, 4.0)});
    fig.add(Ball{Eigen::Vector2d(5.0, 2.0), 1.0}, Style{});
    const std::string svg = fig.to_svg();

    // Nice ticks for [0, 10] must include several integer labels
    CHECK(count_occurrences(svg, ">2</text>") >= 1);
    CHECK(count_occurrences(svg, ">4</text>") >= 1);
    CHECK(count_occurrences(svg, ">8</text>") >= 1);

    // save_svg round-trips through a file
    const std::string path = "test_plot2d_output.svg";
    fig.save_svg(path);
    std::ifstream in(path);
    REQUIRE(in.good());
    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(contents == svg);
    std::remove(path.c_str());
}
