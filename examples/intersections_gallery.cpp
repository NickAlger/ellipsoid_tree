// Pairwise intersection tests, visually
//
// Every cell of ellipsoid_tree's pairwise `intersects` table, shown rather than told:
// for each pair of object types there is one clearly intersecting and one
// clearly disjoint configuration. The first object of each pair is drawn in
// blue, the second in vermillion; halfspaces shade their feasible side. The
// printed booleans below are the actual return values of `intersects`, and
// the panel images are regenerated from this program, so the pictures and
// the results can never drift apart.
//
// One panel to linger on: `ellipsoid_box_no` shows a thin diagonal ellipsoid
// whose axis-aligned bounding box (light gray) overlaps the box even though
// the ellipsoid itself misses it — the configuration where exact tests earn
// their keep over bounding-box filtering.

#include <cstdio>
#include <string>

#include "ellipsoid_tree/ellipsoid_tree.hpp"
#include "ellipsoid_tree/plot2d.hpp"

using namespace ellipsoid_tree;

namespace {

const Style A_STYLE{colors::blue(), 1.8, with_alpha(colors::blue(), 0.25)};
const Style B_STYLE{colors::vermillion(), 1.8, with_alpha(colors::vermillion(), 0.25)};
const Style B_HALF{colors::vermillion(), 1.8, with_alpha(colors::vermillion(), 0.12)};
const Style B_LINE{colors::vermillion(), 2.2, colors::transparent()};
const Style B_POINT{colors::transparent(), 0.0, colors::vermillion()};

Eigen::Matrix2d rotated_cov( double angle, double a, double b )
{
    Eigen::Matrix2d R;
    R << std::cos(angle), -std::sin(angle),
         std::sin(angle),  std::cos(angle);
    return R * Eigen::Vector2d(a * a, b * b).asDiagonal() * R.transpose();
}

template <class DrawBoth>
void panel( const std::string& name, bool result, DrawBoth&& draw )
{
    Plot2D fig;
    draw(fig);
    fig.save_svg(name + ".svg", 320);
    std::printf("%-26s -> %s\n", name.c_str(), result ? "true" : "false");
}

} // end anonymous namespace

int main()
{
    // Recurring first objects
    const Ellipsoid EA{Eigen::Vector2d(0.0, 0.0), rotated_cov(0.5236, 1.0, 0.4)}; // 30 degrees
    const Box       BA{Eigen::Vector2d(-0.8, -0.5), Eigen::Vector2d(0.3, 0.4)};
    const Ball      CA{Eigen::Vector2d(0.0, 0.0), 0.7};
    Eigen::MatrixXd SA_V(2, 3);
    SA_V << -0.8, 0.9, -0.1,
            -0.6, -0.3, 0.8;
    const Simplex SA{SA_V};
    const double tau = 1.0;

    // ---- point vs everything ----
    {
        Eigen::Vector2d p_in(-0.2, 0.1), p_out(0.8, 0.8);
        panel("point_box_yes", intersects(p_in, BA),
              [&]( Plot2D& f ) { f.add(BA, A_STYLE); f.add_marker(p_in, 4.0, B_POINT); });
        panel("point_box_no", intersects(p_out, BA),
              [&]( Plot2D& f ) { f.add(BA, A_STYLE); f.add_marker(p_out, 4.0, B_POINT); });

        Eigen::Vector2d q_in(0.3, -0.4), q_out(0.7, 0.6);
        panel("point_ball_yes", intersects(q_in, CA),
              [&]( Plot2D& f ) { f.add(CA, A_STYLE); f.add_marker(q_in, 4.0, B_POINT); });
        panel("point_ball_no", intersects(q_out, CA),
              [&]( Plot2D& f ) { f.add(CA, A_STYLE); f.add_marker(q_out, 4.0, B_POINT); });

        Eigen::Vector2d e_in(0.5, 0.3), e_out(-0.5, 0.7);
        panel("point_ellipsoid_yes", intersects(e_in, EA, tau),
              [&]( Plot2D& f ) { f.add(EA, tau, A_STYLE); f.add_marker(e_in, 4.0, B_POINT); });
        panel("point_ellipsoid_no", intersects(e_out, EA, tau),
              [&]( Plot2D& f ) { f.add(EA, tau, A_STYLE); f.add_marker(e_out, 4.0, B_POINT); });

        Eigen::Vector2d s_in(0.0, -0.05), s_out(0.6, 0.5);
        panel("point_simplex_yes", intersects(s_in, SA),
              [&]( Plot2D& f ) { f.add(SA, A_STYLE); f.add_marker(s_in, 4.0, B_POINT); });
        panel("point_simplex_no", intersects(s_out, SA),
              [&]( Plot2D& f ) { f.add(SA, A_STYLE); f.add_marker(s_out, 4.0, B_POINT); });
    }

    // ---- box vs ... ----
    {
        Box B_yes{Eigen::Vector2d(0.0, 0.1), Eigen::Vector2d(1.0, 1.0)};
        Box B_no{Eigen::Vector2d(0.6, -0.2), Eigen::Vector2d(1.5, 0.6)};
        panel("box_box_yes", intersects(BA, B_yes),
              [&]( Plot2D& f ) { f.add(BA, A_STYLE); f.add(B_yes, B_STYLE); });
        panel("box_box_no", intersects(BA, B_no),
              [&]( Plot2D& f ) { f.add(BA, A_STYLE); f.add(B_no, B_STYLE); });

        Ball b_yes{Eigen::Vector2d(0.6, 0.6), 0.5};
        Ball b_no{Eigen::Vector2d(0.6, 0.6), 0.25}; // same center, smaller radius
        panel("box_ball_yes", intersects(BA, b_yes),
              [&]( Plot2D& f ) { f.add(BA, A_STYLE); f.add(b_yes, B_STYLE); });
        panel("box_ball_no", intersects(BA, b_no),
              [&]( Plot2D& f ) { f.add(BA, A_STYLE); f.add(b_no, B_STYLE); });

        // The showcase: bounding boxes overlap, the exact test says no
        Ellipsoid E_thin{Eigen::Vector2d(0.0, 0.0), rotated_cov(0.7854, 1.6, 0.12)}; // 45 degrees
        Box corner_box{Eigen::Vector2d(0.75, -1.2), Eigen::Vector2d(1.35, -0.6)};
        Box overlap_box{Eigen::Vector2d(0.1, -0.4), Eigen::Vector2d(1.1, 0.6)};
        panel("ellipsoid_box_yes", intersects(overlap_box, E_thin, tau),
              [&]( Plot2D& f ) { f.add(E_thin, tau, A_STYLE); f.add(overlap_box, B_STYLE); });
        panel("ellipsoid_box_no", intersects(corner_box, E_thin, tau),
              [&]( Plot2D& f )
              {
                  f.add(bounding_box(E_thin, tau), Style{colors::light_gray(), 1.2, colors::transparent()});
                  f.add(E_thin, tau, A_STYLE);
                  f.add(corner_box, B_STYLE);
              });

        Segment s_yes{Eigen::Vector2d(-1.2, -1.0), Eigen::Vector2d(0.9, 0.9)};
        Segment s_no{Eigen::Vector2d(0.6, -0.8), Eigen::Vector2d(1.4, 0.2)};
        panel("box_segment_yes", intersects(s_yes, BA),
              [&]( Plot2D& f ) { f.add(BA, A_STYLE); f.add(s_yes, B_LINE); });
        panel("box_segment_no", intersects(s_no, BA),
              [&]( Plot2D& f ) { f.add(BA, A_STYLE); f.add(s_no, B_LINE); });

        Halfspace h_yes{Eigen::Vector2d(0.0, -1.0), -0.2}; // { y >= 0.2 }
        Halfspace h_no{Eigen::Vector2d(0.0, -1.0), -0.8};  // { y >= 0.8 }
        panel("box_halfspace_yes", intersects(BA, h_yes),
              [&]( Plot2D& f ) { f.add(BA, A_STYLE); f.add(h_yes, B_HALF); });
        panel("box_halfspace_no", intersects(BA, h_no),
              [&]( Plot2D& f ) { f.add(BA, A_STYLE); f.add(h_no, B_HALF); });

        Eigen::MatrixXd T_yes(2, 3), T_no(2, 3);
        T_yes << 0.0, 1.0, 0.4,
                 0.0, 0.3, 1.0;
        T_no << 0.7, 1.6, 1.0,
                0.5, 0.8, 1.4;
        panel("box_simplex_yes", intersects(BA, Simplex{T_yes}),
              [&]( Plot2D& f ) { f.add(BA, A_STYLE); f.add(Simplex{T_yes}, B_STYLE); });
        panel("box_simplex_no", intersects(BA, Simplex{T_no}),
              [&]( Plot2D& f ) { f.add(BA, A_STYLE); f.add(Simplex{T_no}, B_STYLE); });
    }

    // ---- ball vs ... ----
    {
        Ball b_yes{Eigen::Vector2d(0.9, 0.5), 0.45};
        Ball b_no{Eigen::Vector2d(1.3, 0.8), 0.4};
        panel("ball_ball_yes", intersects(CA, b_yes),
              [&]( Plot2D& f ) { f.add(CA, A_STYLE); f.add(b_yes, B_STYLE); });
        panel("ball_ball_no", intersects(CA, b_no),
              [&]( Plot2D& f ) { f.add(CA, A_STYLE); f.add(b_no, B_STYLE); });

        Ball eb_yes{Eigen::Vector2d(0.9, 0.55), 0.35};
        Ball eb_no{Eigen::Vector2d(-1.0, 1.0), 0.3};
        panel("ellipsoid_ball_yes", intersects(eb_yes, EA, tau),
              [&]( Plot2D& f ) { f.add(EA, tau, A_STYLE); f.add(eb_yes, B_STYLE); });
        panel("ellipsoid_ball_no", intersects(eb_no, EA, tau),
              [&]( Plot2D& f ) { f.add(EA, tau, A_STYLE); f.add(eb_no, B_STYLE); });

        Eigen::MatrixXd T_yes(2, 3), T_no(2, 3);
        T_yes << 0.4, 1.3, 0.7,
                 0.2, 0.5, 1.2;
        T_no << 0.9, 1.8, 1.2,
                0.6, 0.9, 1.6;
        panel("ball_simplex_yes", intersects(CA, Simplex{T_yes}),
              [&]( Plot2D& f ) { f.add(CA, A_STYLE); f.add(Simplex{T_yes}, B_STYLE); });
        panel("ball_simplex_no", intersects(CA, Simplex{T_no}),
              [&]( Plot2D& f ) { f.add(CA, A_STYLE); f.add(Simplex{T_no}, B_STYLE); });

        Segment s_yes{Eigen::Vector2d(-0.9, 0.8), Eigen::Vector2d(0.8, -0.5)};
        Segment s_no{Eigen::Vector2d(0.8, 0.6), Eigen::Vector2d(1.8, 1.2)};
        panel("ball_segment_yes", intersects(s_yes, CA),
              [&]( Plot2D& f ) { f.add(CA, A_STYLE); f.add(s_yes, B_LINE); });
        panel("ball_segment_no", intersects(s_no, CA),
              [&]( Plot2D& f ) { f.add(CA, A_STYLE); f.add(s_no, B_LINE); });

        Halfspace h_yes{Eigen::Vector2d(1.0, 0.0), -0.3}; // { x <= -0.3 }
        Halfspace h_no{Eigen::Vector2d(1.0, 0.0), -1.0};  // { x <= -1.0 }
        panel("ball_halfspace_yes", intersects(CA, h_yes),
              [&]( Plot2D& f ) { f.add(CA, A_STYLE); f.add(h_yes, B_HALF); });
        panel("ball_halfspace_no", intersects(CA, h_no),
              [&]( Plot2D& f ) { f.add(CA, A_STYLE); f.add(h_no, B_HALF); });
    }

    // ---- ellipsoid vs ellipsoid / simplex / segment / halfspace ----
    {
        Ellipsoid E_yes{Eigen::Vector2d(0.6, 0.35), rotated_cov(-0.6981, 1.0, 0.4)}; // -40 degrees
        Ellipsoid E_no{Eigen::Vector2d(2.0, 1.5), rotated_cov(-0.6981, 1.0, 0.4)};
        panel("ellipsoid_ellipsoid_yes", intersects(EA, E_yes, tau),
              [&]( Plot2D& f ) { f.add(EA, tau, A_STYLE); f.add(E_yes, tau, B_STYLE); });
        panel("ellipsoid_ellipsoid_no", intersects(EA, E_no, tau),
              [&]( Plot2D& f ) { f.add(EA, tau, A_STYLE); f.add(E_no, tau, B_STYLE); });

        Eigen::MatrixXd T_yes(2, 3), T_no(2, 3);
        T_yes << 0.2, 1.2, 0.5,
                 0.0, 0.4, 1.0;
        T_no << 1.4, 2.2, 1.7,
                0.9, 1.2, 1.8;
        panel("ellipsoid_simplex_yes", intersects(EA, Simplex{T_yes}, tau),
              [&]( Plot2D& f ) { f.add(EA, tau, A_STYLE); f.add(Simplex{T_yes}, B_STYLE); });
        panel("ellipsoid_simplex_no", intersects(EA, Simplex{T_no}, tau),
              [&]( Plot2D& f ) { f.add(EA, tau, A_STYLE); f.add(Simplex{T_no}, B_STYLE); });

        Segment s_yes{Eigen::Vector2d(-1.2, 0.8), Eigen::Vector2d(1.0, -0.9)};
        Segment s_no{Eigen::Vector2d(1.2, 0.6), Eigen::Vector2d(2.0, 1.6)};
        panel("ellipsoid_segment_yes", intersects(s_yes, EA, tau),
              [&]( Plot2D& f ) { f.add(EA, tau, A_STYLE); f.add(s_yes, B_LINE); });
        panel("ellipsoid_segment_no", intersects(s_no, EA, tau),
              [&]( Plot2D& f ) { f.add(EA, tau, A_STYLE); f.add(s_no, B_LINE); });

        Halfspace h_yes{Eigen::Vector2d(1.0, 0.0), -0.2}; // { x <= -0.2 }
        Halfspace h_no{Eigen::Vector2d(1.0, 0.0), -1.3};  // { x <= -1.3 }
        panel("ellipsoid_halfspace_yes", intersects(EA, h_yes, tau),
              [&]( Plot2D& f ) { f.add(EA, tau, A_STYLE); f.add(h_yes, B_HALF); });
        panel("ellipsoid_halfspace_no", intersects(EA, h_no, tau),
              [&]( Plot2D& f ) { f.add(EA, tau, A_STYLE); f.add(h_no, B_HALF); });
    }

    // ---- simplex vs simplex / segment / halfspace ----
    {
        Eigen::MatrixXd T_yes(2, 3), T_no(2, 3);
        T_yes << 0.0, 1.2, 0.6,
                 -0.1, 0.2, -1.0;
        T_no << 1.2, 2.4, 1.8,
                0.5, 0.8, -0.4;
        panel("simplex_simplex_yes", intersects(SA, Simplex{T_yes}),
              [&]( Plot2D& f ) { f.add(SA, A_STYLE); f.add(Simplex{T_yes}, B_STYLE); });
        panel("simplex_simplex_no", intersects(SA, Simplex{T_no}),
              [&]( Plot2D& f ) { f.add(SA, A_STYLE); f.add(Simplex{T_no}, B_STYLE); });

        Segment s_yes{Eigen::Vector2d(-1.2, -1.2), Eigen::Vector2d(0.5, 0.9)};
        Segment s_no{Eigen::Vector2d(1.0, 0.2), Eigen::Vector2d(1.8, 1.0)};
        panel("simplex_segment_yes", intersects(s_yes, SA),
              [&]( Plot2D& f ) { f.add(SA, A_STYLE); f.add(s_yes, B_LINE); });
        panel("simplex_segment_no", intersects(s_no, SA),
              [&]( Plot2D& f ) { f.add(SA, A_STYLE); f.add(s_no, B_LINE); });

        Halfspace h_yes{Eigen::Vector2d(0.0, 1.0), -0.4}; // { y <= -0.4 }
        Halfspace h_no{Eigen::Vector2d(0.0, 1.0), -0.9};  // { y <= -0.9 }
        panel("simplex_halfspace_yes", intersects(SA, h_yes),
              [&]( Plot2D& f ) { f.add(SA, A_STYLE); f.add(h_yes, B_HALF); });
        panel("simplex_halfspace_no", intersects(SA, h_no),
              [&]( Plot2D& f ) { f.add(SA, A_STYLE); f.add(h_no, B_HALF); });
    }

    return 0;
}
