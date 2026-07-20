#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_tree — https://github.com/NickAlger/ellipsoid_tree

/// @file
/// @brief Build a 2D figure from the library's geometric objects and trees and write it as SVG or PNG.
///
/// Optional 2D visualization: build a figure from the library's geometric
/// objects and trees, then write it as an SVG (vector; ellipses keep their
/// true center/axes/rotation as readable XML). This header is deliberately
/// NOT included by the ellipsoid_tree.hpp umbrella — include it only if you want it.
/// It has no dependencies beyond the library itself.
///
///   Plot2D fig;
///   fig.add(ellipsoid, tau, Style{...});
///   draw_tree(fig, ellipsoid_tree, {});         // objects + AABB hierarchy
///   draw_kdtree(fig, kd);                       // splitting-line partition
///   draw_batches(fig, ellipsoid_tree, batches); // color by batch
///   fig.save_svg("figure.svg");                 // axes with physical extents
///
/// All draw functions require dim() == 2 and throw std::invalid_argument
/// otherwise. A PNG raster backend is planned as a follow-up.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "ellipsoid_tree/geometry.hpp"
#include "ellipsoid_tree/object_tree.hpp"
#include "ellipsoid_tree/kd_tree.hpp"
#include "ellipsoid_tree/simplex_mesh.hpp"
#include "ellipsoid_tree/detail/raster2d.hpp"

namespace ellipsoid_tree {

// ------------------------------------------------------------------
//  Colors and styles
// ------------------------------------------------------------------

/// An RGBA color with components in [0, 1].
struct Color
{
    double r = 0.0, g = 0.0, b = 0.0, a = 1.0; ///< Components in [0, 1] (red, green, blue, alpha).
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

/// Stroke and fill styling for a drawn primitive.
struct Style
{
    Color  stroke       = colors::black(); ///< Outline (stroke) color.
    double stroke_width = 1.3;                  ///< Stroke width in canvas pixels.
    Color  fill         = colors::transparent(); ///< Interior fill color.
};

/// Horizontal text alignment relative to the anchor point.
enum class TextAnchor { start, middle, end };

/// The viridis colormap, t in [0, 1].
inline Color colormap_viridis( double t )
{
    Color c;
    detail::viridis_rgb(t, c.r, c.g, c.b);
    c.a = 1.0;
    return c;
}

/// An in-memory RGB raster produced by Plot2D::render_rgb.
struct RenderedImage
{
    int width  = 0; ///< Image width in pixels.
    int height = 0; ///< Image height in pixels.
    std::vector<unsigned char> rgb; ///< Pixels, row-major, 3 bytes (RGB) per pixel.
};


// ------------------------------------------------------------------
//  Plot2D: a world-coordinate scene with an SVG writer
// ------------------------------------------------------------------

/// A world-coordinate 2D scene of drawable primitives with SVG and PNG writers.
class Plot2D
{
public:
    /// Add geometric objects (world coordinates, y up)
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

    /// Add a ball to the scene.
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

    /// Add an axis-aligned box to the scene.
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

    /// Add a line segment to the scene.
    void add( const Segment& S, const Style& style )
    {
        require_dim2(static_cast<int>(S.a.size()), "Segment");
        Eigen::MatrixXd pts(2, 2);
        pts.col(0) = S.a;
        pts.col(1) = S.b;
        add_polyline(pts, style, /*closed=*/false);
    }

    /// Add a simplex (drawn as a point, segment, or filled polygon) to the scene.
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

    /// Add a halfspace, drawn as a boundary line clipped to the plot bounds.
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

    /// Marker radius is in canvas pixels (independent of zoom)
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

    /// Add a polyline through the given points, or a closed polygon when closed is true.
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

    /// Add a text label at pt (text size in canvas pixels).
    void add_text( const Eigen::Ref<const Eigen::Vector2d>& pt, std::string text,
                   double size_px = 12.0, Color color = colors::black(),
                   TextAnchor anchor = TextAnchor::start )
    {
        Primitive p;
        p.kind      = Primitive::Kind::text;
        p.style     = Style{color, 0.0, color};
        p.center    = pt;
        p.text      = std::move(text);
        p.radius_px = size_px;
        p.anchor    = anchor;
        include_point(pt);
        prims_.push_back(std::move(p));
    }

    /// Triangle carrying normalized scalar values t in [0, 1] at its vertices:
    /// the PNG backend interpolates the VALUE barycentrically per pixel and
    /// applies the viridis colormap there, so the colormap curve is followed
    /// exactly even on coarse meshes (unlike Gouraud color interpolation).
    /// SVG falls back to a flat fill at the mean value.
    void add_cg1_triangle( const Eigen::Ref<const Eigen::MatrixXd>& pts,
                           double t0, double t1, double t2 )
    {
        require_dim2(static_cast<int>(pts.rows()), "triangle");
        Primitive p;
        p.kind   = Primitive::Kind::tri_field;
        p.points = pts.leftCols(3);
        p.value_colormap = true;
        p.vvalue[0] = t0;
        p.vvalue[1] = t1;
        p.vvalue[2] = t2;
        const Color avg = colormap_viridis((t0 + t1 + t2) / 3.0);
        p.style = Style{avg, 0.8, avg}; // stroke in the same color covers seams
        for ( int ii = 0; ii < 3; ++ii )
        {
            include_point(pts.col(ii));
        }
        prims_.push_back(std::move(p));
    }

    /// Triangle with per-vertex colors: barycentric color interpolation in the
    /// PNG backend (Gouraud); flat average-color fill in SVG. pts has shape (2, 3).
    void add_filled_triangle( const Eigen::Ref<const Eigen::MatrixXd>& pts,
                              const Color& c0, const Color& c1, const Color& c2 )
    {
        require_dim2(static_cast<int>(pts.rows()), "triangle");
        Primitive p;
        p.kind   = Primitive::Kind::tri_field;
        p.points = pts.leftCols(3);
        p.vcolor[0] = c0;
        p.vcolor[1] = c1;
        p.vcolor[2] = c2;
        Color avg{(c0.r + c1.r + c2.r) / 3.0, (c0.g + c1.g + c2.g) / 3.0,
                  (c0.b + c1.b + c2.b) / 3.0, 1.0};
        p.style = Style{avg, 0.8, avg}; // stroke in the same color covers seams
        for ( int ii = 0; ii < 3; ++ii )
        {
            include_point(pts.col(ii));
        }
        prims_.push_back(std::move(p));
    }

    /// Axes with the physical extents of the drawn data (on by default)
    void axes( bool on ) { axes_on_ = on; }

    /// Override the automatic (data + 5% pad) plot bounds
    void set_bounds( const Box& bounds )
    {
        user_bounds_    = bounds;
        has_user_bounds_ = true;
    }

    /// Write the figure to an SVG file at the given pixel width.
    void save_svg( const std::string& path, int width_px = 900 ) const
    {
        std::ofstream out(path);
        if ( !out )
        {
            throw std::runtime_error("ellipsoid_tree::Plot2D::save_svg: cannot open " + path);
        }
        out << to_svg(width_px);
    }

    /// Return the figure as an SVG document string at the given pixel width.
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
            for ( const Primitive& p : axes_primitives(fr) )
            {
                emit_svg(svg, p, fr);
            }
        }
        svg += "</svg>\n";
        return svg;
    }

    /// Rasterize the figure (white background, 1px antialiasing).
    RenderedImage render_rgb( int width_px = 900 ) const
    {
        const Frame fr = make_frame(width_px);
        detail::Raster img;
        img.init(static_cast<int>(std::lround(fr.width)), static_cast<int>(std::lround(fr.height)));
        for ( const Primitive& p : prims_ )
        {
            rasterize(img, p, fr, /*clip=*/true);
        }
        if ( axes_on_ )
        {
            for ( const Primitive& p : axes_primitives(fr) )
            {
                rasterize(img, p, fr, /*clip=*/false);
            }
        }
        RenderedImage out;
        out.width  = img.w;
        out.height = img.h;
        out.rgb    = std::move(img.rgb);
        return out;
    }

    /// Write the figure to a PNG file at the given pixel width.
    void save_png( const std::string& path, int width_px = 900 ) const
    {
        const RenderedImage im = render_rgb(width_px);
        if ( !stbi_write_png(path.c_str(), im.width, im.height, 3, im.rgb.data(), 3 * im.width) )
        {
            throw std::runtime_error("ellipsoid_tree::Plot2D::save_png: cannot write " + path);
        }
    }

private:
    struct Primitive
    {
        enum class Kind { ellipse, rect, polyline, polygon, marker, text, halfspace, tri_field };
        Kind            kind;
        Style           style;
        Eigen::Vector2d center;       // ellipse/marker/text center; halfspace normal
        double          radius_a = 0; // ellipse semi-axes (world units)
        double          radius_b = 0;
        double          angle    = 0; // ellipse rotation (world radians)
        double          radius_px = 0; // marker radius / text size (canvas px)
        double          offset    = 0; // halfspace offset
        Eigen::Vector2d lo, hi;       // rect
        Eigen::MatrixXd points;       // polyline/polygon/tri_field, shape (2, n)
        std::string     text;
        TextAnchor      anchor = TextAnchor::start;
        bool            canvas_coords = false; // axes internals live in pixel space
        Color           vcolor[3];    // tri_field per-vertex colors (Gouraud)
        double          vvalue[3] = {0, 0, 0}; // tri_field per-vertex scalars in [0, 1]
        bool            value_colormap = false; // interpolate the value, then colormap
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
            throw std::invalid_argument(std::string("ellipsoid_tree::Plot2D: ") + what
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

    // Boundary line of a halfspace primitive clipped to the plot bounds;
    // false if it misses the plot area entirely.
    static bool halfspace_boundary( const Primitive& p, const Frame& fr,
                                    Eigen::Vector2d& out_a, Eigen::Vector2d& out_b )
    {
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
                    hits.push_back(a + (fa / denom) * (b - a));
                }
            }
        };
        const Eigen::Vector2d c00(fr.xlo, fr.ylo), c10(fr.xhi, fr.ylo);
        const Eigen::Vector2d c11(fr.xhi, fr.yhi), c01(fr.xlo, fr.yhi);
        try_edge(c00, c10); try_edge(c10, c11); try_edge(c11, c01); try_edge(c01, c00);
        if ( hits.size() < 2 )
        {
            return false;
        }
        size_t ia = 0, ib = 1;
        double best = -1.0;
        for ( size_t aa = 0; aa < hits.size(); ++aa )
        {
            for ( size_t bb = aa + 1; bb < hits.size(); ++bb )
            {
                const double d2 = (hits[aa] - hits[bb]).squaredNorm();
                if ( d2 > best )
                {
                    best = d2;
                    ia = aa;
                    ib = bb;
                }
            }
        }
        out_a = hits[ia];
        out_b = hits[ib];
        return true;
    }

    // Feasible-side polygon of a halfspace clipped to the plot bounds (world
    // coordinates, ordered around the centroid); empty if nothing is visible.
    static std::vector<Eigen::Vector2d> halfspace_region( const Primitive& p, const Frame& fr )
    {
        std::vector<Eigen::Vector2d> pts;
        const Eigen::Vector2d n = p.center;
        const Eigen::Vector2d corners[4] = {{fr.xlo, fr.ylo}, {fr.xhi, fr.ylo},
                                            {fr.xhi, fr.yhi}, {fr.xlo, fr.yhi}};
        for ( const Eigen::Vector2d& c : corners )
        {
            if ( n.dot(c) <= p.offset )
            {
                pts.push_back(c);
            }
        }
        Eigen::Vector2d a, b;
        if ( halfspace_boundary(p, fr, a, b) )
        {
            pts.push_back(a);
            pts.push_back(b);
        }
        if ( pts.size() < 3 )
        {
            return {};
        }
        Eigen::Vector2d centroid(0, 0);
        for ( const Eigen::Vector2d& q : pts )
        {
            centroid += q;
        }
        centroid /= static_cast<double>(pts.size());
        std::sort(pts.begin(), pts.end(), [&]( const Eigen::Vector2d& u, const Eigen::Vector2d& v )
        {
            return std::atan2(u(1) - centroid(1), u(0) - centroid(0))
                 < std::atan2(v(1) - centroid(1), v(0) - centroid(0));
        });
        return pts;
    }

    void emit_svg( std::string& svg, const Primitive& p, const Frame& fr ) const
    {
        // Canvas-space primitives (axes internals) pass through untransformed
        auto CX = [&]( double x ) { return p.canvas_coords ? x : fr.X(x); };
        auto CY = [&]( double y ) { return p.canvas_coords ? y : fr.Y(y); };
        const double sc = p.canvas_coords ? 1.0 : fr.s;
        switch ( p.kind )
        {
        case Primitive::Kind::ellipse:
        {
            const double deg = -p.angle * 180.0 / 3.14159265358979323846; // y-flip negates angles
            svg += "<ellipse cx=\"0\" cy=\"0\" rx=\"" + fmt(sc * p.radius_a)
                 + "\" ry=\"" + fmt(sc * p.radius_b)
                 + "\" transform=\"translate(" + fmt(CX(p.center(0))) + " " + fmt(CY(p.center(1)))
                 + ") rotate(" + fmt(deg) + ")\"" + style_attrs(p.style) + "/>\n";
            break;
        }
        case Primitive::Kind::rect:
        {
            double x, y, w, h;
            if ( p.canvas_coords )
            {
                x = p.lo(0);
                y = p.lo(1);
                w = p.hi(0) - p.lo(0);
                h = p.hi(1) - p.lo(1);
            }
            else
            {
                x = fr.X(p.lo(0));
                y = fr.Y(p.hi(1));
                w = fr.s * (p.hi(0) - p.lo(0));
                h = fr.s * (p.hi(1) - p.lo(1));
            }
            svg += "<rect x=\"" + fmt(x) + "\" y=\"" + fmt(y) + "\" width=\"" + fmt(w)
                 + "\" height=\"" + fmt(h) + "\"" + style_attrs(p.style) + "/>\n";
            break;
        }
        case Primitive::Kind::polyline:
        case Primitive::Kind::polygon:
        {
            svg += (p.kind == Primitive::Kind::polygon) ? "<polygon points=\"" : "<polyline points=\"";
            for ( int ii = 0; ii < p.points.cols(); ++ii )
            {
                if ( ii > 0 ) { svg += " "; }
                svg += fmt(CX(p.points(0, ii))) + "," + fmt(CY(p.points(1, ii)));
            }
            svg += "\"" + style_attrs(p.style) + "/>\n";
            break;
        }
        case Primitive::Kind::tri_field:
        {
            svg += "<polygon points=\"";
            for ( int ii = 0; ii < 3; ++ii )
            {
                if ( ii > 0 ) { svg += " "; }
                svg += fmt(CX(p.points(0, ii))) + "," + fmt(CY(p.points(1, ii)));
            }
            svg += "\"" + style_attrs(p.style) + "/>\n"; // flat average-color fill
            break;
        }
        case Primitive::Kind::marker:
        {
            svg += "<circle cx=\"" + fmt(CX(p.center(0))) + "\" cy=\"" + fmt(CY(p.center(1)))
                 + "\" r=\"" + fmt(p.radius_px) + "\"" + style_attrs(p.style) + "/>\n";
            break;
        }
        case Primitive::Kind::text:
        {
            std::string anchor_attr;
            if ( p.anchor == TextAnchor::middle ) { anchor_attr = " text-anchor=\"middle\""; }
            if ( p.anchor == TextAnchor::end )    { anchor_attr = " text-anchor=\"end\""; }
            svg += "<text x=\"" + fmt(CX(p.center(0))) + "\" y=\"" + fmt(CY(p.center(1)))
                 + "\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"" + fmt(p.radius_px)
                 + "\" fill=\"" + rgb(p.style.fill) + "\"" + anchor_attr + ">" + p.text + "</text>\n";
            break;
        }
        case Primitive::Kind::halfspace:
        {
            if ( p.style.fill.a > 0.0 ) // shade the feasible side
            {
                std::vector<Eigen::Vector2d> region = halfspace_region(p, fr);
                if ( region.size() >= 3 )
                {
                    svg += "<polygon points=\"";
                    for ( size_t ii = 0; ii < region.size(); ++ii )
                    {
                        if ( ii > 0 ) { svg += " "; }
                        svg += fmt(fr.X(region[ii](0))) + "," + fmt(fr.Y(region[ii](1)));
                    }
                    svg += "\"" + style_attrs(Style{colors::transparent(), 0.0, p.style.fill}) + "/>\n";
                }
            }
            Eigen::Vector2d a, b;
            if ( halfspace_boundary(p, fr, a, b) )
            {
                svg += "<line x1=\"" + fmt(fr.X(a(0))) + "\" y1=\"" + fmt(fr.Y(a(1)))
                     + "\" x2=\"" + fmt(fr.X(b(0))) + "\" y2=\"" + fmt(fr.Y(b(1)))
                     + "\"" + style_attrs(Style{p.style.stroke, p.style.stroke_width,
                                                colors::transparent()}) + "/>\n";
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

    // Axes as canvas-space primitives, shared by the SVG and PNG backends.
    std::vector<Primitive> axes_primitives( const Frame& fr ) const
    {
        const Color axis_col{60.0 / 255.0, 60.0 / 255.0, 60.0 / 255.0, 1.0};
        const Style line_style{axis_col, 1.0, colors::transparent()};

        std::vector<Primitive> out;

        Primitive frame;
        frame.kind          = Primitive::Kind::rect;
        frame.canvas_coords = true;
        frame.style         = line_style;
        frame.lo            = Eigen::Vector2d(fr.ml, fr.mt);
        frame.hi            = Eigen::Vector2d(fr.ml + fr.plot_w, fr.mt + fr.plot_h);
        out.push_back(std::move(frame));

        auto add_line = [&]( double x1, double y1, double x2, double y2 )
        {
            Primitive tick;
            tick.kind          = Primitive::Kind::polyline;
            tick.canvas_coords = true;
            tick.style         = line_style;
            tick.points.resize(2, 2);
            tick.points << x1, x2, y1, y2;
            out.push_back(std::move(tick));
        };
        auto add_label = [&]( double x, double y, const std::string& label, TextAnchor anchor )
        {
            Primitive txt;
            txt.kind          = Primitive::Kind::text;
            txt.canvas_coords = true;
            txt.style         = Style{axis_col, 0.0, axis_col};
            txt.center        = Eigen::Vector2d(x, y);
            txt.text          = label;
            txt.radius_px     = 11.0;
            txt.anchor        = anchor;
            out.push_back(std::move(txt));
        };

        const double y0 = fr.mt + fr.plot_h;
        for ( double t : nice_ticks(fr.xlo, fr.xhi) )
        {
            add_line(fr.X(t), y0, fr.X(t), y0 + 4.0);
            add_label(fr.X(t), y0 + 17.0, tick_label(t), TextAnchor::middle);
        }
        for ( double t : nice_ticks(fr.ylo, fr.yhi) )
        {
            add_line(fr.ml - 4.0, fr.Y(t), fr.ml, fr.Y(t));
            add_label(fr.ml - 8.0, fr.Y(t) + 4.0, tick_label(t), TextAnchor::end);
        }
        return out;
    }

    // Software-rasterize one primitive (SDF coverage, 1px antialiasing).
    void rasterize( detail::Raster& img, const Primitive& p, const Frame& fr, bool clip ) const
    {
        const double clx0 = clip ? fr.ml : 0.0;
        const double cly0 = clip ? fr.mt : 0.0;
        const double clx1 = clip ? fr.ml + fr.plot_w : static_cast<double>(img.w);
        const double cly1 = clip ? fr.mt + fr.plot_h : static_cast<double>(img.h);

        auto CX = [&]( double x ) { return p.canvas_coords ? x : fr.X(x); };
        auto CY = [&]( double y ) { return p.canvas_coords ? y : fr.Y(y); };
        const double sc = p.canvas_coords ? 1.0 : fr.s;

        const Style& st = p.style;
        const bool has_fill   = st.fill.a > 0.0;
        const bool has_stroke = st.stroke.a > 0.0 && st.stroke_width > 0.0;

        // Paint every pixel of a bounding region with fill/stroke coverage
        // computed from a signed distance callback.
        auto paint_sdf = [&]( double x0, double x1, double y0, double y1, auto&& sdf )
        {
            const int ix0 = std::max(0, static_cast<int>(std::floor(std::max(x0, clx0))));
            const int ix1 = std::min(img.w - 1, static_cast<int>(std::ceil(std::min(x1, clx1))));
            const int iy0 = std::max(0, static_cast<int>(std::floor(std::max(y0, cly0))));
            const int iy1 = std::min(img.h - 1, static_cast<int>(std::ceil(std::min(y1, cly1))));
            for ( int iy = iy0; iy <= iy1; ++iy )
            {
                for ( int ix = ix0; ix <= ix1; ++ix )
                {
                    const double px = ix + 0.5;
                    const double py = iy + 0.5;
                    if ( px < clx0 || px > clx1 || py < cly0 || py > cly1 )
                    {
                        continue;
                    }
                    const double d = sdf(px, py);
                    if ( has_fill )
                    {
                        img.blend(ix, iy, st.fill.r, st.fill.g, st.fill.b,
                                  st.fill.a * detail::fill_coverage(d));
                    }
                    if ( has_stroke )
                    {
                        img.blend(ix, iy, st.stroke.r, st.stroke.g, st.stroke.b,
                                  st.stroke.a * detail::stroke_coverage(d, st.stroke_width));
                    }
                }
            }
        };

        const double pad = 0.5 * st.stroke_width + 1.5;

        switch ( p.kind )
        {
        case Primitive::Kind::ellipse:
        {
            const double cx = CX(p.center(0));
            const double cy = CY(p.center(1));
            const double rx = sc * p.radius_a;
            const double ry = sc * p.radius_b;
            const double ang = -p.angle; // canvas rotation (y-flip)
            const double ca = std::cos(ang);
            const double sa = std::sin(ang);
            const double R  = std::max(rx, ry) + pad;
            paint_sdf(cx - R, cx + R, cy - R, cy + R, [&]( double px, double py )
            {
                const double dx = px - cx;
                const double dy = py - cy;
                return detail::sd_ellipse(ca * dx + sa * dy, -sa * dx + ca * dy, rx, ry);
            });
            break;
        }
        case Primitive::Kind::rect:
        {
            double x0, y0, x1, y1;
            if ( p.canvas_coords )
            {
                x0 = p.lo(0); y0 = p.lo(1); x1 = p.hi(0); y1 = p.hi(1);
            }
            else
            {
                x0 = fr.X(p.lo(0)); y0 = fr.Y(p.hi(1));
                x1 = fr.X(p.hi(0)); y1 = fr.Y(p.lo(1));
            }
            const double cx = 0.5 * (x0 + x1);
            const double cy = 0.5 * (y0 + y1);
            const double hx = 0.5 * (x1 - x0);
            const double hy = 0.5 * (y1 - y0);
            paint_sdf(x0 - pad, x1 + pad, y0 - pad, y1 + pad, [&]( double px, double py )
            {
                return detail::sd_box(px, py, cx, cy, hx, hy);
            });
            break;
        }
        case Primitive::Kind::polyline:
        case Primitive::Kind::polygon:
        {
            const int n = static_cast<int>(p.points.cols());
            std::vector<double> pts(2 * n);
            double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
            for ( int ii = 0; ii < n; ++ii )
            {
                pts[2 * ii]     = CX(p.points(0, ii));
                pts[2 * ii + 1] = CY(p.points(1, ii));
                x0 = std::min(x0, pts[2 * ii]);     x1 = std::max(x1, pts[2 * ii]);
                y0 = std::min(y0, pts[2 * ii + 1]); y1 = std::max(y1, pts[2 * ii + 1]);
            }
            if ( p.kind == Primitive::Kind::polygon )
            {
                paint_sdf(x0 - pad, x1 + pad, y0 - pad, y1 + pad, [&]( double px, double py )
                {
                    return detail::sd_polygon(px, py, pts);
                });
            }
            else
            {
                paint_sdf(x0 - pad, x1 + pad, y0 - pad, y1 + pad, [&]( double px, double py )
                {
                    double d = 1e300;
                    for ( int ii = 0; ii + 1 < n; ++ii )
                    {
                        d = std::min(d, detail::sd_segment(px, py, pts[2 * ii], pts[2 * ii + 1],
                                                           pts[2 * ii + 2], pts[2 * ii + 3]));
                    }
                    return d;
                });
            }
            break;
        }
        case Primitive::Kind::tri_field:
        {
            std::vector<double> pts(6);
            double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
            for ( int ii = 0; ii < 3; ++ii )
            {
                pts[2 * ii]     = CX(p.points(0, ii));
                pts[2 * ii + 1] = CY(p.points(1, ii));
                x0 = std::min(x0, pts[2 * ii]);     x1 = std::max(x1, pts[2 * ii]);
                y0 = std::min(y0, pts[2 * ii + 1]); y1 = std::max(y1, pts[2 * ii + 1]);
            }
            // Barycentric coordinates for color interpolation
            const double ux = pts[2] - pts[0], uy = pts[3] - pts[1];
            const double vx = pts[4] - pts[0], vy = pts[5] - pts[1];
            const double det = ux * vy - uy * vx;
            if ( std::abs(det) < 1e-14 )
            {
                break; // degenerate triangle
            }
            const int ix0 = std::max(0, static_cast<int>(std::floor(std::max(x0 - 1.0, clx0))));
            const int ix1 = std::min(img.w - 1, static_cast<int>(std::ceil(std::min(x1 + 1.0, clx1))));
            const int iy0 = std::max(0, static_cast<int>(std::floor(std::max(y0 - 1.0, cly0))));
            const int iy1 = std::min(img.h - 1, static_cast<int>(std::ceil(std::min(y1 + 1.0, cly1))));
            for ( int iy = iy0; iy <= iy1; ++iy )
            {
                for ( int ix = ix0; ix <= ix1; ++ix )
                {
                    const double px = ix + 0.5;
                    const double py = iy + 0.5;
                    if ( px < clx0 || px > clx1 || py < cly0 || py > cly1 )
                    {
                        continue;
                    }
                    // Half-pixel dilation hides seams between adjacent triangles
                    const double cov = detail::fill_coverage(detail::sd_polygon(px, py, pts) - 0.5);
                    if ( cov <= 0.0 )
                    {
                        continue;
                    }
                    double l1 = ((px - pts[0]) * vy - (py - pts[1]) * vx) / det;
                    double l2 = (-(px - pts[0]) * uy + (py - pts[1]) * ux) / det;
                    l1 = std::min(1.0, std::max(0.0, l1));
                    l2 = std::min(1.0, std::max(0.0, l2));
                    const double l0 = std::max(0.0, 1.0 - l1 - l2);
                    const double norm = l0 + l1 + l2;
                    double r, g, b;
                    if ( p.value_colormap )
                    {
                        const double t = (l0 * p.vvalue[0] + l1 * p.vvalue[1] + l2 * p.vvalue[2]) / norm;
                        detail::viridis_rgb(t, r, g, b);
                    }
                    else
                    {
                        r = (l0 * p.vcolor[0].r + l1 * p.vcolor[1].r + l2 * p.vcolor[2].r) / norm;
                        g = (l0 * p.vcolor[0].g + l1 * p.vcolor[1].g + l2 * p.vcolor[2].g) / norm;
                        b = (l0 * p.vcolor[0].b + l1 * p.vcolor[1].b + l2 * p.vcolor[2].b) / norm;
                    }
                    img.blend(ix, iy, r, g, b, cov);
                }
            }
            break;
        }
        case Primitive::Kind::marker:
        {
            const double cx = CX(p.center(0));
            const double cy = CY(p.center(1));
            const double R  = p.radius_px + pad;
            paint_sdf(cx - R, cx + R, cy - R, cy + R, [&]( double px, double py )
            {
                const double dx = px - cx;
                const double dy = py - cy;
                return std::sqrt(dx * dx + dy * dy) - p.radius_px;
            });
            break;
        }
        case Primitive::Kind::text:
        {
            const double scale = p.radius_px / 7.0;
            const double adv   = 6.0 * scale;
            const int    len   = static_cast<int>(p.text.size());
            const double wtot  = (len > 0) ? len * adv - scale : 0.0;
            double x_left = CX(p.center(0));
            if ( p.anchor == TextAnchor::middle ) { x_left -= 0.5 * wtot; }
            if ( p.anchor == TextAnchor::end )    { x_left -= wtot; }
            const double y_top = CY(p.center(1)) - 7.0 * scale;

            const int ix0 = std::max(0, static_cast<int>(std::floor(x_left)));
            const int ix1 = std::min(img.w - 1, static_cast<int>(std::ceil(x_left + wtot)));
            const int iy0 = std::max(0, static_cast<int>(std::floor(y_top)));
            const int iy1 = std::min(img.h - 1, static_cast<int>(std::ceil(y_top + 7.0 * scale)));
            for ( int iy = iy0; iy <= iy1; ++iy )
            {
                for ( int ix = ix0; ix <= ix1; ++ix )
                {
                    const double u = (ix + 0.5 - x_left) / adv;
                    const int    gi = static_cast<int>(std::floor(u));
                    if ( gi < 0 || gi >= len )
                    {
                        continue;
                    }
                    const int col = static_cast<int>(std::floor((u - gi) * 6.0));
                    const int row = static_cast<int>(std::floor((iy + 0.5 - y_top) / scale));
                    if ( col < 0 || col > 4 || row < 0 || row > 6 )
                    {
                        continue;
                    }
                    const unsigned char* glyph = detail::glyph5x7(p.text[gi]);
                    if ( glyph[row] & (0x10 >> col) )
                    {
                        img.blend(ix, iy, st.fill.r, st.fill.g, st.fill.b, st.fill.a);
                    }
                }
            }
            break;
        }
        case Primitive::Kind::halfspace:
        {
            if ( has_fill ) // shade the feasible side
            {
                std::vector<Eigen::Vector2d> region = halfspace_region(p, fr);
                if ( region.size() >= 3 )
                {
                    std::vector<double> pts(2 * region.size());
                    double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
                    for ( size_t ii = 0; ii < region.size(); ++ii )
                    {
                        pts[2 * ii]     = fr.X(region[ii](0));
                        pts[2 * ii + 1] = fr.Y(region[ii](1));
                        x0 = std::min(x0, pts[2 * ii]);     x1 = std::max(x1, pts[2 * ii]);
                        y0 = std::min(y0, pts[2 * ii + 1]); y1 = std::max(y1, pts[2 * ii + 1]);
                    }
                    const int jx0 = std::max(0, static_cast<int>(std::floor(std::max(x0 - 1.0, clx0))));
                    const int jx1 = std::min(img.w - 1, static_cast<int>(std::ceil(std::min(x1 + 1.0, clx1))));
                    const int jy0 = std::max(0, static_cast<int>(std::floor(std::max(y0 - 1.0, cly0))));
                    const int jy1 = std::min(img.h - 1, static_cast<int>(std::ceil(std::min(y1 + 1.0, cly1))));
                    for ( int iy = jy0; iy <= jy1; ++iy )
                    {
                        for ( int ix = jx0; ix <= jx1; ++ix )
                        {
                            const double px = ix + 0.5;
                            const double py = iy + 0.5;
                            if ( px < clx0 || px > clx1 || py < cly0 || py > cly1 )
                            {
                                continue;
                            }
                            img.blend(ix, iy, st.fill.r, st.fill.g, st.fill.b,
                                      st.fill.a * detail::fill_coverage(detail::sd_polygon(px, py, pts)));
                        }
                    }
                }
            }
            Eigen::Vector2d a, b;
            if ( has_stroke && halfspace_boundary(p, fr, a, b) )
            {
                const double ax = fr.X(a(0)), ay = fr.Y(a(1));
                const double bx = fr.X(b(0)), by = fr.Y(b(1));
                const int jx0 = std::max(0, static_cast<int>(std::floor(std::max(std::min(ax, bx) - pad, clx0))));
                const int jx1 = std::min(img.w - 1, static_cast<int>(std::ceil(std::min(std::max(ax, bx) + pad, clx1))));
                const int jy0 = std::max(0, static_cast<int>(std::floor(std::max(std::min(ay, by) - pad, cly0))));
                const int jy1 = std::min(img.h - 1, static_cast<int>(std::ceil(std::min(std::max(ay, by) + pad, cly1))));
                for ( int iy = jy0; iy <= jy1; ++iy )
                {
                    for ( int ix = jx0; ix <= jx1; ++ix )
                    {
                        const double px = ix + 0.5;
                        const double py = iy + 0.5;
                        if ( px < clx0 || px > clx1 || py < cly0 || py > cly1 )
                        {
                            continue;
                        }
                        img.blend(ix, iy, st.stroke.r, st.stroke.g, st.stroke.b,
                                  st.stroke.a * detail::stroke_coverage(
                                      detail::sd_segment(px, py, ax, ay, bx, by), st.stroke_width));
                    }
                }
            }
            break;
        }
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

/// Options controlling what draw_tree renders and how.
struct DrawTreeOptions
{
    bool objects    = true; ///< Draw the stored objects.
    bool leaf_boxes = false; ///< bounding box of each element
    bool node_boxes = true;  ///< composite boxes of the AABB hierarchy
    int  max_depth  = -1;    ///< draw node boxes down to this depth only (-1: all)
    bool color_node_boxes_by_depth = true; ///< Color node boxes by their depth in the hierarchy.

    Style object_style   = Style{colors::blue(), 1.3, with_alpha(colors::blue(), 0.15)}; ///< Style for the stored objects.
    Style leaf_box_style = Style{colors::gray(), 0.8, colors::transparent()}; ///< Style for the per-element leaf boxes.
    Style node_box_style = Style{colors::black(), 1.0, colors::transparent()}; ///< Style for the hierarchy node boxes.
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
        throw std::invalid_argument("ellipsoid_tree::draw_tree: tree must be 2D");
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

/// Draw an EllipsoidTree's objects and its AABB node hierarchy.
inline void draw_tree( Plot2D& fig, const EllipsoidTree& T, DrawTreeOptions opts = DrawTreeOptions() )
{
    detail::draw_tree_impl(fig, T.tree(), T.size(), opts,
        [&]( int ii ) { fig.add(T.object(ii), T.tau(), opts.object_style); });
}

/// Draw a BoxTree's objects and its AABB node hierarchy.
inline void draw_tree( Plot2D& fig, const BoxTree& T, DrawTreeOptions opts = DrawTreeOptions() )
{
    detail::draw_tree_impl(fig, T.tree(), T.size(), opts,
        [&]( int ii ) { fig.add(T.object(ii), opts.object_style); });
}

/// Draw a BallTree's objects and its AABB node hierarchy.
inline void draw_tree( Plot2D& fig, const BallTree& T, DrawTreeOptions opts = DrawTreeOptions() )
{
    detail::draw_tree_impl(fig, T.tree(), T.size(), opts,
        [&]( int ii ) { fig.add(T.object(ii), opts.object_style); });
}

/// Draw a SimplexTree's objects and its AABB node hierarchy.
inline void draw_tree( Plot2D& fig, const SimplexTree& T, DrawTreeOptions opts = DrawTreeOptions() )
{
    detail::draw_tree_impl(fig, T.tree(), T.size(), opts,
        [&]( int ii ) { fig.add(T.object(ii), opts.object_style); });
}

/// Draw only the listed elements (e.g. the results of a collision query).
inline void draw_elements( Plot2D& fig, const EllipsoidTree& T, const std::vector<int>& inds, const Style& style )
{
    for ( int ii : inds ) { fig.add(T.object(ii), T.tau(), style); }
}
/// Draw the listed box elements in the given style.
inline void draw_elements( Plot2D& fig, const BoxTree& T, const std::vector<int>& inds, const Style& style )
{
    for ( int ii : inds ) { fig.add(T.object(ii), style); }
}
/// Draw the listed ball elements in the given style.
inline void draw_elements( Plot2D& fig, const BallTree& T, const std::vector<int>& inds, const Style& style )
{
    for ( int ii : inds ) { fig.add(T.object(ii), style); }
}
/// Draw the listed simplex elements in the given style.
inline void draw_elements( Plot2D& fig, const SimplexTree& T, const std::vector<int>& inds, const Style& style )
{
    for ( int ii : inds ) { fig.add(T.object(ii), style); }
}

/// Ellipsoids colored by batch (from pick_ellipsoid_batches).
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

/// Options controlling what draw_kdtree renders and how.
struct DrawKDTreeOptions
{
    bool   points    = true; ///< Draw the stored points as markers.
    bool   splits    = true; ///< Draw the recursive splitting lines.
    int    max_depth = -1;   ///< draw splitting lines down to this depth (-1: all)
    double marker_radius_px = 2.5; ///< Point marker radius in canvas pixels.
    bool   color_splits_by_depth = true; ///< Color splitting lines by their recursion depth.

    Style point_style = Style{colors::transparent(), 0.0, colors::black()}; ///< Style for the point markers.
    Style split_style = Style{colors::black(), 1.1, colors::transparent()}; ///< Style for the splitting lines.
};

/// Draw a kd-tree: its recursive splitting lines and the stored points.
inline void draw_kdtree( Plot2D& fig, const KDTree& T, DrawKDTreeOptions opts = DrawKDTreeOptions() )
{
    if ( T.size() == 0 )
    {
        return;
    }
    if ( T.dim() != 2 )
    {
        throw std::invalid_argument("ellipsoid_tree::draw_kdtree: tree must be 2D");
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


// ------------------------------------------------------------------
//  CG1 (piecewise linear) scalar field rendering
// ------------------------------------------------------------------

/// Options controlling how draw_cg1_field maps field values to colors.
struct FieldOptions
{
    double vmin = std::numeric_limits<double>::quiet_NaN(); ///< NaN: use the data minimum
    double vmax = std::numeric_limits<double>::quiet_NaN(); ///< NaN: use the data maximum
    bool   wireframe  = false; ///< Overlay the triangle edges.
    Style  wire_style = Style{with_alpha(colors::black(), 0.35), 0.5, colors::transparent()}; ///< Style for the wireframe overlay.
};

/// Render a CG1 nodal field on a 2D SimplexMesh through the viridis colormap.
/// The PNG backend interpolates colors barycentrically per pixel (a
/// dependency-free tripcolor); the SVG backend uses flat per-triangle fill.
inline void draw_cg1_field( Plot2D& fig, const SimplexMesh& mesh,
                            const Eigen::Ref<const Eigen::VectorXd>& vertex_values,
                            FieldOptions opts = FieldOptions() )
{
    if ( mesh.dim() != 2 )
    {
        throw std::invalid_argument("ellipsoid_tree::draw_cg1_field: mesh must be 2D");
    }
    if ( vertex_values.size() != mesh.num_vertices() )
    {
        throw std::invalid_argument("ellipsoid_tree::draw_cg1_field: one value per vertex required");
    }
    double vmin = std::isnan(opts.vmin) ? vertex_values.minCoeff() : opts.vmin;
    double vmax = std::isnan(opts.vmax) ? vertex_values.maxCoeff() : opts.vmax;
    if ( !(vmax > vmin) )
    {
        vmax = vmin + 1.0;
    }

    for ( int cc = 0; cc < mesh.num_cells(); ++cc )
    {
        Eigen::MatrixXd V(2, 3);
        double          t[3];
        for ( int kk = 0; kk < 3; ++kk )
        {
            const int vidx = mesh.cells()(kk, cc);
            V.col(kk) = mesh.vertices().col(vidx);
            t[kk]     = (vertex_values(vidx) - vmin) / (vmax - vmin);
        }
        fig.add_cg1_triangle(V, t[0], t[1], t[2]);
        if ( opts.wireframe )
        {
            fig.add_polyline(V, opts.wire_style, /*closed=*/true);
        }
    }
}

} // end namespace ellipsoid_tree
