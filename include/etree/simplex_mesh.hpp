#pragma once
// SPDX-License-Identifier: MIT
// Part of etree — https://github.com/NickAlger/ellipsoid_tree

/// @file
/// @brief Simplicial mesh in R^d (cells are d-simplices): point location with
/// barycentric coordinates, closest point on/in the mesh, CG1 (piecewise
/// linear) finite element evaluation, and ellipsoid queries against the cells.
///
/// The boundary is extracted as the faces belonging to exactly one cell.
/// Closest-point queries use a nearest-boundary-vertex kd-tree bound to
/// shortlist candidate boundary faces, then project onto each candidate with
/// closest_point_in_simplex — whose face enumeration replaces the explicit
/// subface bookkeeping (edges, corners, ...) of a hand-rolled projector.

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "etree/geometry.hpp"
#include "etree/intersections.hpp"
#include "etree/object_tree.hpp"
#include "etree/kd_tree.hpp"
#include "etree/detail/parallel_for.hpp"

namespace etree {

/// Simplicial mesh in R^d (cells are d-simplices) with point location, closest-point, CG1 evaluation, and ellipsoid queries.
class SimplexMesh
{
public:
    SimplexMesh( const Eigen::Ref<const Eigen::MatrixXd>& vertices, // (dim, num_vertices)
                 const Eigen::Ref<const Eigen::MatrixXi>& cells,    // (dim+1, num_cells)
                 int num_threads = 0 )
        : vertices_(vertices), cells_(cells)
    {
        dim_ = static_cast<int>(vertices.rows());
        if ( vertices.cols() < 1 )
        {
            throw std::invalid_argument("etree::SimplexMesh: no vertices");
        }
        if ( cells.cols() < 1 )
        {
            throw std::invalid_argument("etree::SimplexMesh: no cells");
        }
        if ( cells.rows() != dim_ + 1 )
        {
            throw std::invalid_argument("etree::SimplexMesh: cells must have dim+1 vertices");
        }

        cell_tree_ = SimplexTree(vertices_, cells_, num_threads);

        // Boundary faces: sorted (dim)-tuples of vertex indices belonging to
        // exactly one cell.
        std::map<std::vector<int>, int> face_counts;
        for ( int cc = 0; cc < cells_.cols(); ++cc )
        {
            for ( int opposite = 0; opposite < dim_ + 1; ++opposite )
            {
                std::vector<int> face;
                face.reserve(dim_);
                for ( int kk = 0; kk < dim_ + 1; ++kk )
                {
                    if ( kk != opposite )
                    {
                        face.push_back(cells_(kk, cc));
                    }
                }
                std::sort(face.begin(), face.end());
                face_counts[face] += 1;
            }
        }

        std::vector<std::vector<int>> boundary;
        for ( const auto& fc : face_counts )
        {
            if ( fc.second == 1 )
            {
                boundary.push_back(fc.first);
            }
        }
        faces_.resize(dim_, static_cast<int>(boundary.size()));
        for ( int ff = 0; ff < static_cast<int>(boundary.size()); ++ff )
        {
            for ( int kk = 0; kk < dim_; ++kk )
            {
                faces_(kk, ff) = boundary[ff][kk];
            }
        }
        face_tree_ = SimplexTree(vertices_, faces_, num_threads);

        // kd-tree over the boundary vertices (for closest-point distance bounds)
        std::set<int> boundary_vertex_set;
        for ( int ff = 0; ff < faces_.cols(); ++ff )
        {
            for ( int kk = 0; kk < dim_; ++kk )
            {
                boundary_vertex_set.insert(faces_(kk, ff));
            }
        }
        Eigen::MatrixXd boundary_vertices(dim_, boundary_vertex_set.size());
        int vv = 0;
        for ( int ind : boundary_vertex_set )
        {
            boundary_vertices.col(vv) = vertices_.col(ind);
            vv += 1;
        }
        boundary_kdtree_.build(boundary_vertices);
    }

    int dim() const                { return dim_; }
    int num_vertices() const       { return static_cast<int>(vertices_.cols()); }
    int num_cells() const          { return static_cast<int>(cells_.cols()); }
    int num_boundary_faces() const { return static_cast<int>(faces_.cols()); }

    const Eigen::MatrixXd& vertices() const       { return vertices_; }
    const Eigen::MatrixXi& cells() const          { return cells_; }
    const Eigen::MatrixXi& boundary_faces() const { return faces_; }
    const SimplexTree& cell_tree() const          { return cell_tree_; }
    const SimplexTree& boundary_face_tree() const { return face_tree_; }

    // ------------------------------------------------------------------
    //  Point location
    // ------------------------------------------------------------------

    /// For each query point (column): the index of a containing cell (-1 if
    /// outside the mesh) and its barycentric coordinates in that cell
    /// (columns of the second output; zero-filled for outside points).
    std::pair<Eigen::VectorXi, Eigen::MatrixXd>
    locate_points( const Eigen::Ref<const Eigen::MatrixXd>& points, int num_threads = 0 ) const
    {
        const int np = static_cast<int>(points.cols());
        Eigen::VectorXi cell_inds(np);
        Eigen::MatrixXd coords = Eigen::MatrixXd::Zero(dim_ + 1, np);

        detail::parallel_for(0, np, [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
        {
            for ( std::ptrdiff_t ii = aa; ii < bb; ++ii )
            {
                const int e = cell_tree_.first_collision(points.col(ii));
                cell_inds(ii) = e;
                if ( e >= 0 )
                {
                    coords.col(ii) = cell_tree_.affine_coordinates(e, points.col(ii));
                }
            }
        }, num_threads);

        return std::make_pair(std::move(cell_inds), std::move(coords));
    }

    Eigen::Array<bool, Eigen::Dynamic, 1>
    point_is_in_mesh( const Eigen::Ref<const Eigen::MatrixXd>& points, int num_threads = 0 ) const
    {
        return (locate_points(points, num_threads).first.array() >= 0);
    }

    // ------------------------------------------------------------------
    //  Closest point on/in the mesh
    // ------------------------------------------------------------------

    Eigen::VectorXd closest_point( const Eigen::Ref<const Eigen::VectorXd>& p ) const
    {
        if ( cell_tree_.first_collision(p) >= 0 )
        {
            return p;
        }

        // Distance to the nearest boundary vertex bounds the distance to the
        // boundary; every face that could contain the closest point meets
        // this ball.
        std::pair<Eigen::MatrixXi, Eigen::MatrixXd> nearest = boundary_kdtree_.query(p, 1, 1);
        const double radius = (1.0 + 1e-14) * std::sqrt(nearest.second(0, 0));

        std::vector<int> candidates = face_tree_.collisions(Ball{p, radius});
        if ( candidates.empty() ) // cannot happen for a valid volumetric mesh
        {
            candidates.resize(faces_.cols());
            std::iota(candidates.begin(), candidates.end(), 0);
        }

        Eigen::VectorXd best_point;
        double best_dsq = std::numeric_limits<double>::infinity();
        for ( int ff : candidates )
        {
            ClosestPointResult res = closest_point_in_simplex(p, face_tree_.object(ff).V);
            if ( res.distance_squared < best_dsq )
            {
                best_dsq   = res.distance_squared;
                best_point = std::move(res.point);
            }
        }
        return best_point;
    }

    Eigen::MatrixXd closest_points( const Eigen::Ref<const Eigen::MatrixXd>& points,
                                    int num_threads = 0 ) const
    {
        const int np = static_cast<int>(points.cols());
        Eigen::MatrixXd out(dim_, np);
        detail::parallel_for(0, np, [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
        {
            for ( std::ptrdiff_t ii = aa; ii < bb; ++ii )
            {
                out.col(ii) = closest_point(points.col(ii));
            }
        }, num_threads);
        return out;
    }

    // ------------------------------------------------------------------
    //  CG1 (piecewise linear) finite element evaluation
    // ------------------------------------------------------------------

    /// functions_at_vertices: one function per row, shape (num_functions,
    /// num_vertices); points: shape (dim, num_points). Returns function values
    /// at the points, shape (num_functions, num_points). Points outside the
    /// mesh evaluate to zero, unless reflect_exterior is set, in which case
    /// they are first reflected across the boundary through their closest
    /// boundary point (points whose reflection is still outside give zero).
    Eigen::MatrixXd eval_cg1( const Eigen::Ref<const Eigen::MatrixXd>& functions_at_vertices,
                              const Eigen::Ref<const Eigen::MatrixXd>& points,
                              bool reflect_exterior = false,
                              int num_threads = 0 ) const
    {
        if ( functions_at_vertices.cols() != vertices_.cols() )
        {
            throw std::invalid_argument(
                "etree::SimplexMesh::eval_cg1: functions_at_vertices needs one column per vertex");
        }
        const int num_functions = static_cast<int>(functions_at_vertices.rows());
        const int np = static_cast<int>(points.cols());
        Eigen::MatrixXd out = Eigen::MatrixXd::Zero(num_functions, np);

        detail::parallel_for(0, np, [&]( std::ptrdiff_t aa, std::ptrdiff_t bb )
        {
            for ( std::ptrdiff_t ii = aa; ii < bb; ++ii )
            {
                Eigen::VectorXd p = points.col(ii);
                int e = cell_tree_.first_collision(p);
                if ( e < 0 && reflect_exterior )
                {
                    p = 2.0 * closest_point(p) - p;
                    e = cell_tree_.first_collision(p);
                }
                if ( e >= 0 )
                {
                    Eigen::VectorXd alpha = cell_tree_.affine_coordinates(e, p);
                    for ( int kk = 0; kk < dim_ + 1; ++kk )
                    {
                        out.col(ii) += alpha(kk) * functions_at_vertices.col(cells_(kk, e));
                    }
                }
            }
        }, num_threads);

        return out;
    }

    // ------------------------------------------------------------------
    //  Ellipsoid queries against the cells
    // ------------------------------------------------------------------

    std::vector<int> cells_intersecting( const Ellipsoid& E, double tau ) const
    {
        return cell_tree_.collisions(E, tau);
    }

    /// All (cell, ellipsoid) collision pairs against an EllipsoidTree, via
    /// dual-tree traversal.
    std::vector<std::pair<int, int>> cell_ellipsoid_pairs( const EllipsoidTree& ellipsoids ) const
    {
        return collision_pairs(cell_tree_, ellipsoids);
    }

    /// All (cell of this mesh, cell of the other mesh) pairs whose closed
    /// cells intersect — the geometric kernel of supermeshing / conservative
    /// field transfer between non-matching meshes.
    std::vector<std::pair<int, int>> cell_pairs( const SimplexMesh& other ) const
    {
        return collision_pairs(cell_tree_, other.cell_tree_);
    }

private:
    int             dim_ = 0;
    Eigen::MatrixXd vertices_; // (dim, num_vertices)
    Eigen::MatrixXi cells_;    // (dim+1, num_cells)
    Eigen::MatrixXi faces_;    // (dim, num_boundary_faces)
    SimplexTree     cell_tree_;
    SimplexTree     face_tree_;
    KDTree          boundary_kdtree_;
};

} // end namespace etree
