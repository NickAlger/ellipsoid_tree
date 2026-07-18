#pragma once
// SPDX-License-Identifier: MIT
// Part of etree — https://github.com/NickAlger/ellipsoid_tree

// Optional 2D visualization: build a figure from the library's geometric
// objects and trees, then write it as an SVG (vector; ellipses keep their
// true center/axes/rotation as readable XML). This header is deliberately
// NOT included by the etree.hpp umbrella — include it only if you want it.
// It has no dependencies beyond the library itself.
//
//   Plot2D fig;
//   fig.add(ellipsoid, tau, Style{...});
//   draw_tree(fig, ellipsoid_tree, {});         // objects + AABB hierarchy
//   draw_kdtree(fig, kd);                       // splitting-line partition
//   draw_batches(fig, ellipsoid_tree, batches); // color by batch
//   fig.save_svg("figure.svg");                 // axes with physical extents
//
// All draw functions require dim() == 2 and throw std::invalid_argument
// otherwise. A PNG raster backend is planned as a follow-up.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "etree/geometry.hpp"
#include "etree/object_tree.hpp"
#include "etree/kd_tree.hpp"

namespace etree {

// ------------------------------------------------------------------
//  Colors and styles
// ------------------------------------------------------------------

struct Color
{
    double r = 0.0, g = 0.0, b = 0.0, a = 1.0; // components in [0, 1]
};

namespace colors {
inline Color transparent() { return Color{0, 0, 0, 0}; }
inline Color black()       { return Color{0.10, 0.10, 0.10, 1}; }
inline Color gray()        { return Color{0.62, 0.62, 0.62, 1}; }
inline Color light_gray()  { return Color{0.85, 0.85, 0.85, 1}; }
// Okabe-Ito colorblind-safe palette
inline Color orange()      { return Color{0.902, 0.624, 0.000, 1}; }
inline Color sky_blue()    { return Color{0.337, 0.706, 0.914, 1}; }
inline Color green()       { return Color{0.000, 0.620, 0.451, 1}; }
inline Color yellow()      { return Color{0.941, 0.894, 0.259, 1}; }
inline Color blue()        { return Color{0.000, 0.447, 0.698, 1}; }
inline Color vermillion()  { return Color{0.835, 0.369, 0.000, 1}; }
inline Color purple()      { return Color{0.800, 0.475, 0.655, 1}; }
} // end namespace colors

inline Color palette_color( int index )
{
    static const Color cycle[8] = {colors::blue(),   colors::orange(),     colors::green(),
                                   colors::vermillion(), colors::sky_blue(), colors::purple(),
                                   colors::yellow(), colors::black()};
    return cycle[((index % 8) + 8) % 8];
}

inline Color with_alpha( Color c, double a )
{
    c.a = a;
    return c;
}

struct Style
{
    Color  stroke       = colors::black();
    double stroke_width = 1.3;                  // in canvas pixels
    Color  fill         = colors::transparent();
};


// ------------------------------------------------------------------
//  Plot2D: a world-coordinate scene with an SVG writer
// ------------------------------------------------------------------

class Plot2D
{
public:
    // Add geometric objects (world coordinates, y up)
    void add( const Ellipsoid& E, double tau, const Style& style )
    {
        require_dim2(static_cast<int>(E.mu.size()), "Ellipsoid");
        Primitive p;
        p.kind  = Primitive::Kind::ellipse;
        p.style = style;
        p.center = E.mu;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(E.Sigma);
        p.radius_a = tau * std::sqrt(std::max(es.eigenvalues()(1), 0.0)); // major
        p.radius_b = tau * std::sqrt(std::max(es.eigenvalues()(0), 0.0)); // minor
        Eigen::Vector2d v = es.eigenvectors().col(1);
        p.angle = std::atan2(v(1), v(0));
        include_bounds(bounding_box(E, tau));
        prims_.push_back(std::move(p));
    }

    void add( const Ball& B, const Style& style )
    {
        require_dim2(static_cast<int>(B.center.size()), "Ball");
        Primitive p;
        p.kind     = Primitive::Kind::ellipse;
        p.style    = style;
        p.center   = B.center;
        p.radius_a = B.radius;
        p.radius_b = B.radius;
        p.angle    = 0.0;
        include_bounds(bounding_box(B));
        prims_.push_back(std::move(p));
    }

    void add( const Box& B, const Style& style )
    {
        require_dim2(static_cast<int>(B.lo.size()), "Box");
        Primitive p;
        p.kind  = Primitive::Kind::rect;
        p.style = style;
        p.lo    = B.lo;
        p.hi    = B.hi;
        include_bounds(B);
        prims_.push_back(std::move(p));
    }

    void add( const Segment& S, const Style& style )
    {
        require_dim2(static_cast<int>(S.a.size()), "Segment");
        Eigen::MatrixXd pts(2, 2);
        pts.col(0) = S.a;
        pts.col(1) = S.b;
        add_polyline(pts, style, /*closed=*/false);
    }

    void add( const Simplex& S, const Style& style )
    {
        require_dim2(static_cast<int>(S.V.rows()), "Simplex");
        const int K = static_cast<int>(S.V.cols());
        if ( K == 1 )
        {
            add_marker(S.V.col(0), 3.0, style);
        }
        else if ( K == 2 )
        {
            add_polyline(S.V, style, /*closed=*/false);
        }
        else
        {
            // Order vertices by angle around the centroid (convex hulls only)
            Eigen::Vector2d c = S.V.rowwise().mean();
            std::vector<int> order(K);
            for ( int ii = 0; ii < K; ++ii ) { order[ii] = ii; }
            std::sort(order.begin(), order.end(), [&]( int ii, int jj )
            {
                return std::atan2(S.V(1, ii) - c(1), S.V(0, ii) - c(0))
                     < std::atan2(S.V(1, jj) - c(1), S.V(0, jj) - c(0));
            });
            Eigen::MatrixXd pts(2, K);
            for ( int ii = 0; ii < K; ++ii ) { pts.col(ii) = S.V.col(order[ii]); }
            add_polyline(pts, style, /*closed=*/true);
        }
    }

    void add( const Halfspace& H, const Style& style )
    {
        require_dim2(static_cast<int>(H.normal.size()), "Halfspace");
        Primitive p;
        p.kind   = Primitive::Kind::halfspace; // boundary line, clipped at write time
        p.style  = style;
        p.center = H.normal;
        p.offset = H.offset;
        prims_.push_back(std::move(p));
    }

    // Marker radius is in canvas pixels (independent of zoom)
    void add_marker( const Eigen::Ref<const Eigen::Vector2d>& pt, double radius_px, const Style& style )
    {
        Primitive p;
        p.kind      = Primitive::Kind::marker;
        p.style     = style;
        p.center    = pt;
        p.radius_px = radius_px;
        include_point(pt);
        prims_.push_back(std::move(p));
    }

    void add_polyline( const Eigen::Ref<const Eigen::MatrixXd>& pts, const Style& style, bool closed = false )
    {
        require_dim2(static_cast<int>(pts.rows()), "polyline");
        Primitive p;
        p.kind   = closed ? Primitive::Kind::polygon : Primitive::Kind::polyline;
        p.style  = style;
        p.points = pts;
        for ( int ii = 0; ii < pts.cols(); ++ii )
        {
            include_point(pts.col(ii));
        }
        prims_.push_back(std::move(p));
    }

    void add_text( const Eigen::Ref<const Eigen::Vector2d>& pt, std::string text,
                   double size_px = 12.0, Color color = colors::black() )
    {
        Primitive p;
        p.kind      = Primitive::Kind::text;
        p.style     = Style{color, 0.0, color};
        p.center    = pt;
        p.text      = std::move(text);
        p.radius_px = size_px;
        include_point(pt);
        prims_.push_back(std::move(p));
    }

    // Axes with the physical extents of the drawn data (on by default)
    void axes( bool on ) { axes_on_ = on; }

    // Override the automatic (data + 5% pad) plot bounds
    void set_bounds( const Box& bounds )
    {
        user_bounds_    = bounds;
        has_user_bounds_ = true;
    }

    void save_svg( const std::string& path, int width_px = 900 ) const
    {
        std::ofstream out(path);
        if ( !out )
        {
            throw std::runtime_error("etree::Plot2D::save_svg: cannot open " + path);
        }
        out << to_svg(width_px);
    }

    std::string to_svg( int width_px = 900 ) const
    {
        const Frame fr = make_frame(width_px);
        std::string svg;
        svg.reserve(4096 + 256 * prims_.size());

        svg += "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" + fmt(fr.width)
             + "\" height=\"" + fmt(fr.height)
             + "\" viewBox=\"0 0 " + fmt(fr.width) + " " + fmt(fr.height) + "\">\n";
        svg += "<rect x=\"0\" y=\"0\" width=\"" + fmt(fr.width) + "\" height=\"" + fmt(fr.height)
             + "\" fill=\"white\"/>\n";
        svg += "<clipPath id=\"plotclip\"><rect x=\"" + fmt(fr.ml) + "\" y=\"" + fmt(fr.mt)
             + "\" width=\"" + fmt(fr.plot_w) + "\" height=\"" + fmt(fr.plot_h)
             + "\"/></clipPath>\n";
        svg += "<g clip-path=\"url(#plotclip)\">\n";
        for ( const Primitive& p : prims_ )
        {
            emit_svg(svg, p, fr);
        }
        svg += "</g>\n";
        if ( axes_on_ )
        {
            emit_axes(svg, fr);
        }
        svg += "</svg>\n";
        return svg;
    }

private:
    struct Primitive
    {
        enum class Kind { ellipse, rect, polyline, polygon, marker, text, halfspace };
        Kind            kind;
        Style           style;
        Eigen::Vector2d center;       // ellipse/marker/text center; halfspace normal
        double          radius_a = 0; // ellipse semi-axes (world units)
        double          radius_b = 0;
        double          angle    = 0; // ellipse rotation (world radians)
        double          radius_px = 0; // marker radius / text size (canvas px)
        double          offset    = 0; // halfspace offset
        Eigen::Vector2d lo, hi;       // rect
        Eigen::MatrixXd points;       // polyline/polygon, shape (2, n)
        std::string     text;
    };

    struct Frame
    {
        double ml, mt, plot_w, plot_h, width, height; // margins and sizes (px)
        double xlo, xhi, ylo, yhi;                    // world bounds
        double s;                                     // world -> px scale (uniform)

        double X( double x ) const { return ml + s * (x - xlo); }
        double Y( double y ) const { return mt + s * (yhi - y); }
    };

    static void require_dim2( int d, const char* what )
    {
        if ( d != 2 )
        {
            throw std::invalid_argument(std::string("etree::Plot2D: ") + what
                                        + " must be 2D to be drawn");
        }
    }

    void include_point( const Eigen::Vector2d& pt )
    {
        if ( !has_data_bounds_ )
        {
            data_lo_ = pt;
            data_hi_ = pt;
            has_data_bounds_ = true;
        }
        else
        {
            data_lo_ = data_lo_.cwiseMin(pt);
            data_hi_ = data_hi_.cwiseMax(pt);
        }
    }

    void include_bounds( const Box& B )
    {
        include_point(B.lo);
        include_point(B.hi);
    }

    Frame make_frame( int width_px ) const
    {
        Eigen::Vector2d lo(0.0, 0.0), hi(1.0, 1.0);
        if ( has_user_bounds_ )
        {
            lo = user_bounds_.lo;
            hi = user_bounds_.hi;
        }
        else if ( has_data_bounds_ )
        {
            Eigen::Vector2d pad = 0.05 * (data_hi_ - data_lo_)
                                  + Eigen::Vector2d(1e-12, 1e-12);
            lo = data_lo_ - pad;
            hi = data_hi_ + pad;
        }

        Frame fr;
        fr.ml     = axes_on_ ? 58.0 : 10.0;
        fr.mt     = 14.0;
        const double mr = 14.0;
        const double mb = axes_on_ ? 42.0 : 10.0;
        fr.xlo = lo(0); fr.xhi = hi(0);
        fr.ylo = lo(1); fr.yhi = hi(1);
        fr.plot_w = std::max(10.0, width_px - fr.ml - mr);
        fr.s      = fr.plot_w / (fr.xhi - fr.xlo);
        fr.plot_h = fr.s * (fr.yhi - fr.ylo);
        fr.width  = width_px;
        fr.height = fr.plot_h + fr.mt + mb;
        return fr;
    }

    static std::string fmt( double v )
    {
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%.10g", v);
        return std::string(buf);
    }

    static std::string rgb( const Color& c )
    {
        char buf[40];
        std::snprintf(buf, sizeof(buf), "rgb(%d,%d,%d)",
                      static_cast<int>(std::lround(255 * c.r)),
                      static_cast<int>(std::lround(255 * c.g)),
                      static_cast<int>(std::lround(255 * c.b)));
        return std::string(buf);
    }

    static std::string style_attrs( const Style& st )
    {
        std::string s;
        if ( st.stroke.a > 0.0 && st.stroke_width > 0.0 )
        {
            s += " stroke=\"" + rgb(st.stroke) + "\" stroke-width=\"" + fmt(st.stroke_width) + "\"";
            if ( st.stroke.a < 1.0 )
            {
                s += " stroke-opacity=\"" + fmt(st.stroke.a) + "\"";
            }
        }
        else
        {
            s += " stroke=\"none\"";
        }
        if ( st.fill.a > 0.0 )
        {
            s += " fill=\"" + rgb(st.fill) + "\"";
            if ( st.fill.a < 1.0 )
            {
                s += " fill-opacity=\"" + fmt(st.fill.a) + "\"";
            }
        }
        else
        {
            s += " fill=\"none\"";
        }
        return s;
    }

    void emit_svg( std::string& svg, const Primitive& p, const Frame& fr ) const
    {
        switch ( p.kind )
        {
        case Primitive::Kind::ellipse:
        {
            const double deg = -p.angle * 180.0 / 3.14159265358979323846; // y-flip negates angles
            svg += "<ellipse cx=\"0\" cy=\"0\" rx=\"" + fmt(fr.s * p.radius_a)
                 + "\" ry=\"" + fmt(fr.s * p.radius_b)
                 + "\" transform=\"translate(" + fmt(fr.X(p.center(0))) + " " + fmt(fr.Y(p.center(1)))
                 + ") rotate(" + fmt(deg) + ")\"" + style_attrs(p.style) + "/>\n";
            break;
        }
        case Primitive::Kind::rect:
        {
            svg += "<rect x=\"" + fmt(fr.X(p.lo(0))) + "\" y=\"" + fmt(fr.Y(p.hi(1)))
                 + "\" width=\"" + fmt(fr.s * (p.hi(0) - p.lo(0)))
                 + "\" height=\"" + fmt(fr.s * (p.hi(1) - p.lo(1))) + "\""
                 + style_attrs(p.style) + "/>\n";
            break;
        }
        case Primitive::Kind::polyline:
        case Primitive::Kind::polygon:
        {
            svg += (p.kind == Primitive::Kind::polygon) ? "<polygon points=\"" : "<polyline points=\"";
            for ( int ii = 0; ii < p.points.cols(); ++ii )
            {
                if ( ii > 0 ) { svg += " "; }
                svg += fmt(fr.X(p.points(0, ii))) + "," + fmt(fr.Y(p.points(1, ii)));
            }
            svg += "\"" + style_attrs(p.style) + "/>\n";
            break;
        }
        case Primitive::Kind::marker:
        {
            svg += "<circle cx=\"" + fmt(fr.X(p.center(0))) + "\" cy=\"" + fmt(fr.Y(p.center(1)))
                 + "\" r=\"" + fmt(p.radius_px) + "\"" + style_attrs(p.style) + "/>\n";
            break;
        }
        case Primitive::Kind::text:
        {
            svg += "<text x=\"" + fmt(fr.X(p.center(0))) + "\" y=\"" + fmt(fr.Y(p.center(1)))
                 + "\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"" + fmt(p.radius_px)
                 + "\" fill=\"" + rgb(p.style.fill) + "\">" + p.text + "</text>\n";
            break;
        }
        case Primitive::Kind::halfspace:
        {
            // Clip the boundary line normal.x = offset to the plot bounds
            const Eigen::Vector2d n = p.center;
            std::vector<Eigen::Vector2d> hits;
            auto try_edge = [&]( const Eigen::Vector2d& a, const Eigen::Vector2d& b )
            {
                const double fa = n.dot(a) - p.offset;
                const double fb = n.dot(b) - p.offset;
                if ( (fa <= 0.0 && fb >= 0.0) || (fa >= 0.0 && fb <= 0.0) )
                {
                    const double denom = fa - fb;
                    if ( std::abs(denom) > 1e-300 )
                    {
                        const double t = fa / denom;
                        hits.push_back(a + t * (b - a));
                    }
                }
            };
            const Eigen::Vector2d c00(fr.xlo, fr.ylo), c10(fr.xhi, fr.ylo);
            const Eigen::Vector2d c11(fr.xhi, fr.yhi), c01(fr.xlo, fr.yhi);
            try_edge(c00, c10); try_edge(c10, c11); try_edge(c11, c01); try_edge(c01, c00);
            if ( hits.size() >= 2 )
            {
                // Farthest pair among collected intersection points
                int ia = 0, ib = 1;
                double best = -1.0;
                for ( size_t aa = 0; aa < hits.size(); ++aa )
                {
                    for ( size_t bb = aa + 1; bb < hits.size(); ++bb )
                    {
                        const double d2 = (hits[aa] - hits[bb]).squaredNorm();
                        if ( d2 > best )
                        {
                            best = d2;
                            ia = static_cast<int>(aa);
                            ib = static_cast<int>(bb);
                        }
                    }
                }
                svg += "<line x1=\"" + fmt(fr.X(hits[ia](0))) + "\" y1=\"" + fmt(fr.Y(hits[ia](1)))
                     + "\" x2=\"" + fmt(fr.X(hits[ib](0))) + "\" y2=\"" + fmt(fr.Y(hits[ib](1)))
                     + "\"" + style_attrs(p.style) + "/>\n";
            }
            break;
        }
        }
    }

    // "Nice numbers" ticks: step is 1, 2, or 5 times a power of ten
    static std::vector<double> nice_ticks( double lo, double hi )
    {
        const double raw = (hi - lo) / 6.0;
        const double mag = std::pow(10.0, std::floor(std::log10(raw)));
        const double norm = raw / mag;
        const double step = mag * (norm < 1.5 ? 1.0 : norm < 3.5 ? 2.0 : norm < 7.5 ? 5.0 : 10.0);
        std::vector<double> ticks;
        for ( double t = std::ceil(lo / step) * step; t <= hi + 1e-9 * step; t += step )
        {
            ticks.push_back(std::abs(t) < 1e-12 * step ? 0.0 : t);
        }
        return ticks;
    }

    static std::string tick_label( double v )
    {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%g", v);
        return std::string(buf);
    }

    void emit_axes( std::string& svg, const Frame& fr ) const
    {
        const std::string axis_color = "rgb(60,60,60)";
        svg += "<rect x=\"" + fmt(fr.ml) + "\" y=\"" + fmt(fr.mt) + "\" width=\"" + fmt(fr.plot_w)
             + "\" height=\"" + fmt(fr.plot_h) + "\" fill=\"none\" stroke=\"" + axis_color
             + "\" stroke-width=\"1\"/>\n";

        const double y0 = fr.mt + fr.plot_h;
        for ( double t : nice_ticks(fr.xlo, fr.xhi) )
        {
            const double x = fr.X(t);
            svg += "<line x1=\"" + fmt(x) + "\" y1=\"" + fmt(y0) + "\" x2=\"" + fmt(x)
                 + "\" y2=\"" + fmt(y0 + 4.0) + "\" stroke=\"" + axis_color + "\" stroke-width=\"1\"/>\n";
            svg += "<text x=\"" + fmt(x) + "\" y=\"" + fmt(y0 + 17.0)
                 + "\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"11\" fill=\"" + axis_color
                 + "\" text-anchor=\"middle\">" + tick_label(t) + "</text>\n";
        }
        for ( double t : nice_ticks(fr.ylo, fr.yhi) )
        {
            const double y = fr.Y(t);
            svg += "<line x1=\"" + fmt(fr.ml - 4.0) + "\" y1=\"" + fmt(y) + "\" x2=\"" + fmt(fr.ml)
                 + "\" y2=\"" + fmt(y) + "\" stroke=\"" + axis_color + "\" stroke-width=\"1\"/>\n";
            svg += "<text x=\"" + fmt(fr.ml - 8.0) + "\" y=\"" + fmt(y + 4.0)
                 + "\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"11\" fill=\"" + axis_color
                 + "\" text-anchor=\"end\">" + tick_label(t) + "</text>\n";
        }
    }

    std::vector<Primitive> prims_;
    Eigen::Vector2d        data_lo_, data_hi_;
    bool                   has_data_bounds_ = false;
    Box                    user_bounds_;
    bool                   has_user_bounds_ = false;
    bool                   axes_on_         = true;
};


// ------------------------------------------------------------------
//  Tree drawing
// ------------------------------------------------------------------

struct DrawTreeOptions
{
    bool objects    = true;
    bool leaf_boxes = false; // bounding box of each element
    bool node_boxes = true;  // composite boxes of the AABB hierarchy
    int  max_depth  = -1;    // draw node boxes down to this depth only (-1: all)
    bool color_node_boxes_by_depth = true;

    Style object_style   = Style{colors::blue(), 1.3, with_alpha(colors::blue(), 0.15)};
    Style leaf_box_style = Style{colors::gray(), 0.8, colors::transparent()};
    Style node_box_style = Style{colors::black(), 1.0, colors::transparent()};
};

namespace detail {

inline int heap_depth( int node )
{
    int depth = 0;
    for ( int v = node + 1; v > 1; v /= 2 )
    {
        depth += 1;
    }
    return depth;
}

template <class AddObject>
inline void draw_tree_impl( Plot2D& fig, const AABBTree& tree, int num_objects,
                            const DrawTreeOptions& opts, AddObject&& add_object )
{
    if ( !tree.empty() && tree.dim() != 2 )
    {
        throw std::invalid_argument("etree::draw_tree: tree must be 2D");
    }
    for ( int b = 0; b < tree.num_nodes(); ++b )
    {
        const bool leaf = tree.is_leaf(b);
        if ( leaf ? opts.leaf_boxes : opts.node_boxes )
        {
            const int depth = heap_depth(b);
            if ( !leaf && opts.max_depth >= 0 && depth > opts.max_depth )
            {
                continue;
            }
            Style st = leaf ? opts.leaf_box_style : opts.node_box_style;
            if ( !leaf && opts.color_node_boxes_by_depth )
            {
                st.stroke = with_alpha(palette_color(depth), st.stroke.a);
            }
            fig.add(Box{tree.node_lo(b), tree.node_hi(b)}, st);
        }
    }
    if ( opts.objects )
    {
        for ( int ii = 0; ii < num_objects; ++ii )
        {
            add_object(ii);
        }
    }
}

} // end namespace detail

inline void draw_tree( Plot2D& fig, const EllipsoidTree& T, DrawTreeOptions opts = DrawTreeOptions() )
{
    detail::draw_tree_impl(fig, T.tree(), T.size(), opts,
        [&]( int ii ) { fig.add(T.object(ii), T.tau(), opts.object_style); });
}

inline void draw_tree( Plot2D& fig, const BoxTree& T, DrawTreeOptions opts = DrawTreeOptions() )
{
    detail::draw_tree_impl(fig, T.tree(), T.size(), opts,
        [&]( int ii ) { fig.add(T.object(ii), opts.object_style); });
}

inline void draw_tree( Plot2D& fig, const BallTree& T, DrawTreeOptions opts = DrawTreeOptions() )
{
    detail::draw_tree_impl(fig, T.tree(), T.size(), opts,
        [&]( int ii ) { fig.add(T.object(ii), opts.object_style); });
}

inline void draw_tree( Plot2D& fig, const SimplexTree& T, DrawTreeOptions opts = DrawTreeOptions() )
{
    detail::draw_tree_impl(fig, T.tree(), T.size(), opts,
        [&]( int ii ) { fig.add(T.object(ii), opts.object_style); });
}

// Draw only the listed elements (e.g. the results of a collision query).
inline void draw_elements( Plot2D& fig, const EllipsoidTree& T, const std::vector<int>& inds, const Style& style )
{
    for ( int ii : inds ) { fig.add(T.object(ii), T.tau(), style); }
}
inline void draw_elements( Plot2D& fig, const BoxTree& T, const std::vector<int>& inds, const Style& style )
{
    for ( int ii : inds ) { fig.add(T.object(ii), style); }
}
inline void draw_elements( Plot2D& fig, const BallTree& T, const std::vector<int>& inds, const Style& style )
{
    for ( int ii : inds ) { fig.add(T.object(ii), style); }
}
inline void draw_elements( Plot2D& fig, const SimplexTree& T, const std::vector<int>& inds, const Style& style )
{
    for ( int ii : inds ) { fig.add(T.object(ii), style); }
}

// Ellipsoids colored by batch (from pick_ellipsoid_batches).
inline void draw_batches( Plot2D& fig, const EllipsoidTree& T,
                          const std::vector<std::vector<int>>& batches,
                          double fill_alpha = 0.3, double stroke_width = 1.3 )
{
    for ( size_t bb = 0; bb < batches.size(); ++bb )
    {
        const Color c = palette_color(static_cast<int>(bb));
        const Style st{c, stroke_width, with_alpha(c, fill_alpha)};
        for ( int ii : batches[bb] )
        {
            fig.add(T.object(ii), T.tau(), st);
        }
    }
}


// ------------------------------------------------------------------
//  kd-tree drawing: recursive splitting lines through the median points
// ------------------------------------------------------------------

struct DrawKDTreeOptions
{
    bool   points    = true;
    bool   splits    = true;
    int    max_depth = -1;   // draw splitting lines down to this depth (-1: all)
    double marker_radius_px = 2.5;
    bool   color_splits_by_depth = true;

    Style point_style = Style{colors::transparent(), 0.0, colors::black()};
    Style split_style = Style{colors::black(), 1.1, colors::transparent()};
};

inline void draw_kdtree( Plot2D& fig, const KDTree& T, DrawKDTreeOptions opts = DrawKDTreeOptions() )
{
    if ( T.size() == 0 )
    {
        return;
    }
    if ( T.dim() != 2 )
    {
        throw std::invalid_argument("etree::draw_kdtree: tree must be 2D");
    }
    const Eigen::MatrixXd& pts = T.ordered_points();
    Eigen::Vector2d lo = pts.rowwise().minCoeff();
    Eigen::Vector2d hi = pts.rowwise().maxCoeff();
    Eigen::Vector2d pad = 0.05 * (hi - lo) + Eigen::Vector2d(1e-12, 1e-12);
    lo -= pad;
    hi += pad;

    std::function<void(int, int, int, Eigen::Vector2d, Eigen::Vector2d)> recurse =
        [&]( int start, int stop, int depth, Eigen::Vector2d rlo, Eigen::Vector2d rhi )
    {
        if ( stop - start <= T.built_block_size() )
        {
            return;
        }
        const int axis = depth % 2;
        const int mid  = start + (stop - start) / 2;
        const double c = pts(axis, mid);

        if ( opts.splits && (opts.max_depth < 0 || depth <= opts.max_depth) )
        {
            Style st = opts.split_style;
            if ( opts.color_splits_by_depth )
            {
                st.stroke = with_alpha(palette_color(depth), st.stroke.a);
            }
            Eigen::MatrixXd seg(2, 2);
            if ( axis == 0 )
            {
                seg.col(0) = Eigen::Vector2d(c, rlo(1));
                seg.col(1) = Eigen::Vector2d(c, rhi(1));
            }
            else
            {
                seg.col(0) = Eigen::Vector2d(rlo(0), c);
                seg.col(1) = Eigen::Vector2d(rhi(0), c);
            }
            fig.add_polyline(seg, st);
        }

        Eigen::Vector2d near_hi = rhi;
        near_hi(axis) = c;
        Eigen::Vector2d far_lo = rlo;
        far_lo(axis) = c;
        recurse(start,   mid,  depth + 1, rlo,    near_hi);
        recurse(mid + 1, stop, depth + 1, far_lo, rhi);
    };
    recurse(0, T.size(), 0, lo, hi);

    if ( opts.points )
    {
        for ( int ii = 0; ii < T.size(); ++ii )
        {
            fig.add_marker(pts.col(ii), opts.marker_radius_px, opts.point_style);
        }
    }
}

} // end namespace etree
