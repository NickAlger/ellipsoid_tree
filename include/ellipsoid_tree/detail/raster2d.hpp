#pragma once
// SPDX-License-Identifier: MIT
// Part of ellipsoid_tree — https://github.com/NickAlger/ellipsoid_tree

// Software rasterization support for Plot2D's PNG backend: an RGB canvas
// with alpha blending, signed-distance-function coverage for the primitive
// shapes (1-pixel smoothstep antialiasing), a tiny 5x7 bitmap font for axis
// labels, and the viridis colormap. PNG encoding is delegated to the
// vendored stb_image_write (public domain), included here with
// STB_IMAGE_WRITE_STATIC so this stays safe in a header-only library.

#include <algorithm>
#include <cmath>
#include <vector>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "ellipsoid_tree/thirdparty/stb/stb_image_write.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace ellipsoid_tree {

struct Color; // defined in plot2d.hpp

namespace detail {

struct Raster
{
    int w = 0;
    int h = 0;
    std::vector<unsigned char> rgb; // row-major, 3 bytes per pixel

    void init( int width, int height )
    {
        w = width;
        h = height;
        rgb.assign(static_cast<size_t>(3) * w * h, 255); // white background
    }

    void blend( int x, int y, double r, double g, double b, double alpha )
    {
        if ( x < 0 || x >= w || y < 0 || y >= h || alpha <= 0.0 )
        {
            return;
        }
        const double a = std::min(alpha, 1.0);
        unsigned char* px = &rgb[3 * (static_cast<size_t>(y) * w + x)];
        px[0] = static_cast<unsigned char>(std::lround(px[0] * (1.0 - a) + 255.0 * r * a));
        px[1] = static_cast<unsigned char>(std::lround(px[1] * (1.0 - a) + 255.0 * g * a));
        px[2] = static_cast<unsigned char>(std::lround(px[2] * (1.0 - a) + 255.0 * b * a));
    }
};

// ---- signed distances (canvas pixel units; negative inside) ----

inline double sd_box( double px, double py, double cx, double cy, double hx, double hy )
{
    const double qx = std::abs(px - cx) - hx;
    const double qy = std::abs(py - cy) - hy;
    const double ox = std::max(qx, 0.0);
    const double oy = std::max(qy, 0.0);
    return std::sqrt(ox * ox + oy * oy) + std::min(std::max(qx, qy), 0.0);
}

inline double sd_segment( double px, double py, double ax, double ay, double bx, double by )
{
    const double ux = bx - ax;
    const double uy = by - ay;
    const double uu = ux * ux + uy * uy;
    double t = 0.0;
    if ( uu > 0.0 )
    {
        t = std::min(1.0, std::max(0.0, ((px - ax) * ux + (py - ay) * uy) / uu));
    }
    const double dx = px - (ax + t * ux);
    const double dy = py - (ay + t * uy);
    return std::sqrt(dx * dx + dy * dy);
}

// First-order approximate signed distance to an axis-aligned ellipse centered
// at the origin (accurate near the boundary, which is where coverage matters).
inline double sd_ellipse( double x, double y, double rx, double ry )
{
    const double eps = 1e-9;
    rx = std::max(rx, eps);
    ry = std::max(ry, eps);
    const double f = std::sqrt((x * x) / (rx * rx) + (y * y) / (ry * ry));
    if ( f < 1e-12 )
    {
        return -std::min(rx, ry);
    }
    const double gx = x / (rx * rx);
    const double gy = y / (ry * ry);
    return (f - 1.0) * f / std::sqrt(gx * gx + gy * gy);
}

// Signed distance to a closed polygon (pts: pairs (x0,y0,x1,y1,...)),
// negative inside (even-odd crossing rule).
inline double sd_polygon( double px, double py, const std::vector<double>& pts )
{
    const int n = static_cast<int>(pts.size() / 2);
    double dmin = 1e300;
    bool inside = false;
    for ( int ii = 0, jj = n - 1; ii < n; jj = ii++ )
    {
        const double xi = pts[2 * ii], yi = pts[2 * ii + 1];
        const double xj = pts[2 * jj], yj = pts[2 * jj + 1];
        dmin = std::min(dmin, sd_segment(px, py, xi, yi, xj, yj));
        if ( (yi > py) != (yj > py) )
        {
            const double x_int = xi + (py - yi) * (xj - xi) / (yj - yi);
            if ( px < x_int )
            {
                inside = !inside;
            }
        }
    }
    return inside ? -dmin : dmin;
}

// ---- coverage from signed distance (1px smoothing) ----

inline double fill_coverage( double d )
{
    return std::min(1.0, std::max(0.0, 0.5 - d));
}

inline double stroke_coverage( double d, double width )
{
    return std::min(1.0, std::max(0.0, 0.5 * width + 0.5 - std::abs(d)));
}

// ---- 5x7 bitmap font for axis labels (digits, '.', '-', '+', 'e') ----

inline const unsigned char* glyph5x7( char ch )
{
    static const unsigned char digits[10][7] = {
        {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}, // 0
        {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, // 1
        {0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F}, // 2
        {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E}, // 3
        {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, // 4
        {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}, // 5
        {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}, // 6
        {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}, // 7
        {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}, // 8
        {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}, // 9
    };
    static const unsigned char dot[7]   = {0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06};
    static const unsigned char minus[7] = {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00};
    static const unsigned char plus[7]  = {0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00};
    static const unsigned char e_lc[7]  = {0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E};
    static const unsigned char blank[7] = {0, 0, 0, 0, 0, 0, 0};

    if ( ch >= '0' && ch <= '9' ) { return digits[ch - '0']; }
    if ( ch == '.' ) { return dot; }
    if ( ch == '-' ) { return minus; }
    if ( ch == '+' ) { return plus; }
    if ( ch == 'e' || ch == 'E' ) { return e_lc; }
    if ( ch == ' ' ) { return blank; }
    return blank; // unknown characters render as space
}

// ---- viridis colormap (11 published anchors, linear interpolation) ----

inline void viridis_rgb( double t, double& r, double& g, double& b )
{
    static const double anchors[11][3] = {
        {0.267004, 0.004874, 0.329415}, {0.282623, 0.140926, 0.457517},
        {0.253935, 0.265254, 0.529983}, {0.206756, 0.371758, 0.553117},
        {0.163625, 0.471133, 0.558148}, {0.127568, 0.566949, 0.550556},
        {0.134692, 0.658636, 0.517649}, {0.266941, 0.748751, 0.440573},
        {0.477504, 0.821444, 0.318195}, {0.741388, 0.873449, 0.149561},
        {0.993248, 0.906157, 0.143936}};
    t = std::min(1.0, std::max(0.0, t));
    const double u = t * 10.0;
    const int    i0 = std::min(9, static_cast<int>(u));
    const double w = u - i0;
    r = anchors[i0][0] * (1.0 - w) + anchors[i0 + 1][0] * w;
    g = anchors[i0][1] * (1.0 - w) + anchors[i0 + 1][1] * w;
    b = anchors[i0][2] * (1.0 - w) + anchors[i0 + 1][2] * w;
}

} // end namespace detail
} // end namespace ellipsoid_tree
