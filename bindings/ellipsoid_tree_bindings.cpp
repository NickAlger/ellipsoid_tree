// SPDX-License-Identifier: MIT
// Part of ellipsoid_tree — https://github.com/NickAlger/ellipsoid_tree
//
// Python bindings. Array-layout convention at the Python boundary: POINTS ARE
// ROWS, matching numpy/scipy practice — point sets are (n, d), simplex
// vertices are (K, d), mesh cells are (num_cells, d+1), k-NN results are
// (num_queries, k). Internally ellipsoid_tree stores points as columns; the transpose
// happens here, once, at the boundary.

#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "ellipsoid_tree/ellipsoid_tree.hpp"
#include "ellipsoid_tree/plot2d.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace ellipsoid_tree;

namespace {

using RowsXd = Eigen::Ref<const Eigen::MatrixXd>;
using RowsXi = Eigen::Ref<const Eigen::MatrixXi>;

Eigen::MatrixXd cols_from_rows( const RowsXd& rows ) { return rows.transpose(); }
Eigen::MatrixXi icols_from_rows( const RowsXi& rows ) { return rows.transpose(); }

std::vector<Ellipsoid> ellipsoids_from_arrays( const RowsXd& mus, const py::array_t<double>& Sigmas )
{
    const int n = static_cast<int>(mus.rows());
    const int d = static_cast<int>(mus.cols());
    if ( Sigmas.ndim() != 3 || Sigmas.shape(0) != n || Sigmas.shape(1) != d || Sigmas.shape(2) != d )
    {
        throw std::invalid_argument("Sigmas must have shape (n, d, d) matching mus (n, d)");
    }
    auto S = Sigmas.unchecked<3>();
    std::vector<Ellipsoid> out(n);
    for ( int ii = 0; ii < n; ++ii )
    {
        Eigen::MatrixXd Sig(d, d);
        for ( int rr = 0; rr < d; ++rr )
        {
            for ( int cc = 0; cc < d; ++cc )
            {
                Sig(rr, cc) = S(ii, rr, cc);
            }
        }
        out[ii] = Ellipsoid{mus.row(ii).transpose(), std::move(Sig)};
    }
    return out;
}

py::array_t<unsigned char> image_to_numpy( const RenderedImage& im )
{
    py::array_t<unsigned char> arr({im.height, im.width, 3});
    std::memcpy(arr.mutable_data(), im.rgb.data(), im.rgb.size());
    return arr;
}

template <class Tree>
void def_common_tree( py::class_<Tree>& cls )
{
    cls.def("__len__", &Tree::size)
       .def_property_readonly("size", &Tree::size)
       .def_property_readonly("dim", &Tree::dim)
       .def_readwrite("exact_ellipsoid_pruning", &Tree::exact_ellipsoid_pruning)
       .def("point_collisions_batch",
            []( const Tree& t, const RowsXd& points, int num_threads )
            { return t.point_collisions_batch(cols_from_rows(points), num_threads); },
            "points"_a, "num_threads"_a = 0, py::call_guard<py::gil_scoped_release>(),
            "Collision lists for many points at once; points has shape (m, d).");
}

} // end anonymous namespace

PYBIND11_MODULE(ellipsoid_tree, m)
{
    m.doc() = "ellipsoid_tree: exact ellipsoid intersection tests and spatial trees.\n\n"
              "Array convention: points are rows — point sets are (n, d), simplex\n"
              "vertices (K, d), mesh cells (num_cells, d+1). Ellipsoids are\n"
              "E(tau) = {x : (x - mu)^T Sigma^{-1} (x - mu) <= tau^2}; tau defaults to 1.";
    m.attr("__version__") = py::str(std::to_string(ELLIPSOID_TREE_VERSION_MAJOR) + "."
                                    + std::to_string(ELLIPSOID_TREE_VERSION_MINOR) + "."
                                    + std::to_string(ELLIPSOID_TREE_VERSION_PATCH));

    // ------------------------------------------------------------------
    //  Geometry types
    // ------------------------------------------------------------------
    py::class_<Box>(m, "Box")
        .def(py::init<Eigen::VectorXd, Eigen::VectorXd>(), "lo"_a, "hi"_a)
        .def_readonly("lo", &Box::lo)
        .def_readonly("hi", &Box::hi)
        .def("__repr__", []( const Box& b )
             { return "Box(dim=" + std::to_string(b.lo.size()) + ")"; });

    py::class_<Ball>(m, "Ball")
        .def(py::init<Eigen::VectorXd, double>(), "center"_a, "radius"_a)
        .def_readonly("center", &Ball::center)
        .def_readonly("radius", &Ball::radius)
        .def("__repr__", []( const Ball& b )
             { return "Ball(dim=" + std::to_string(b.center.size())
                      + ", radius=" + std::to_string(b.radius) + ")"; });

    py::class_<Ellipsoid>(m, "Ellipsoid")
        .def(py::init<Eigen::VectorXd, Eigen::MatrixXd>(), "mu"_a, "Sigma"_a)
        .def_readonly("mu", &Ellipsoid::mu)
        .def_readonly("Sigma", &Ellipsoid::Sigma)
        .def("__repr__", []( const Ellipsoid& e )
             { return "Ellipsoid(dim=" + std::to_string(e.mu.size()) + ")"; });

    py::class_<Simplex>(m, "Simplex")
        .def(py::init([]( const RowsXd& vertices ) { return Simplex{cols_from_rows(vertices)}; }),
             "vertices"_a, "Vertices as rows: shape (K, d), K <= d + 1.")
        .def_property_readonly("vertices",
             []( const Simplex& s ) { return Eigen::MatrixXd(s.V.transpose()); })
        .def("__repr__", []( const Simplex& s )
             { return "Simplex(dim=" + std::to_string(s.V.rows())
                      + ", num_vertices=" + std::to_string(s.V.cols()) + ")"; });

    py::class_<Segment>(m, "Segment")
        .def(py::init<Eigen::VectorXd, Eigen::VectorXd>(), "a"_a, "b"_a)
        .def_readonly("a", &Segment::a)
        .def_readonly("b", &Segment::b);

    py::class_<Halfspace>(m, "Halfspace")
        .def(py::init<Eigen::VectorXd, double>(), "normal"_a, "offset"_a,
             "The set {x : normal . x <= offset}.")
        .def_readonly("normal", &Halfspace::normal)
        .def_readonly("offset", &Halfspace::offset);

    m.def("bounding_box", []( const Ball& b ) { return bounding_box(b); }, "ball"_a);
    m.def("bounding_box", []( const Ellipsoid& e, double tau ) { return bounding_box(e, tau); },
          "ellipsoid"_a, "tau"_a = 1.0);
    m.def("bounding_box", []( const Simplex& s ) { return bounding_box(s); }, "simplex"_a);

    // ------------------------------------------------------------------
    //  Pairwise intersection tests (both argument orders)
    // ------------------------------------------------------------------
    auto def_pair = [&]( auto&& fn ) { fn(); };
    (void)def_pair;

    // point vs ...
    m.def("intersects", []( const Eigen::VectorXd& p, const Box& b ) { return intersects(p, b); });
    m.def("intersects", []( const Box& b, const Eigen::VectorXd& p ) { return intersects(p, b); });
    m.def("intersects", []( const Eigen::VectorXd& p, const Ball& b ) { return intersects(p, b); });
    m.def("intersects", []( const Ball& b, const Eigen::VectorXd& p ) { return intersects(p, b); });
    m.def("intersects", []( const Eigen::VectorXd& p, const Ellipsoid& e, double tau )
          { return intersects(p, e, tau); }, "point"_a, "ellipsoid"_a, "tau"_a = 1.0);
    m.def("intersects", []( const Ellipsoid& e, const Eigen::VectorXd& p, double tau )
          { return intersects(p, e, tau); }, "ellipsoid"_a, "point"_a, "tau"_a = 1.0);
    m.def("intersects", []( const Eigen::VectorXd& p, const Simplex& s ) { return intersects(p, s); });
    m.def("intersects", []( const Simplex& s, const Eigen::VectorXd& p ) { return intersects(p, s); });
    m.def("intersects", []( const Eigen::VectorXd& p, const Halfspace& h ) { return intersects(p, h); });
    m.def("intersects", []( const Halfspace& h, const Eigen::VectorXd& p ) { return intersects(p, h); });

    // box vs ...
    m.def("intersects", []( const Box& a, const Box& b ) { return intersects(a, b); });
    m.def("intersects", []( const Box& a, const Ball& b ) { return intersects(a, b); });
    m.def("intersects", []( const Ball& b, const Box& a ) { return intersects(a, b); });
    m.def("intersects", []( const Box& a, const Ellipsoid& e, double tau )
          { return intersects(a, e, tau); }, "box"_a, "ellipsoid"_a, "tau"_a = 1.0);
    m.def("intersects", []( const Ellipsoid& e, const Box& a, double tau )
          { return intersects(a, e, tau); }, "ellipsoid"_a, "box"_a, "tau"_a = 1.0);
    m.def("intersects", []( const Box& a, const Simplex& s ) { return intersects(a, s); });
    m.def("intersects", []( const Simplex& s, const Box& a ) { return intersects(a, s); });
    m.def("intersects", []( const Segment& g, const Box& a ) { return intersects(g, a); });
    m.def("intersects", []( const Box& a, const Segment& g ) { return intersects(g, a); });
    m.def("intersects", []( const Box& a, const Halfspace& h ) { return intersects(a, h); });
    m.def("intersects", []( const Halfspace& h, const Box& a ) { return intersects(a, h); });

    // ball vs ...
    m.def("intersects", []( const Ball& a, const Ball& b ) { return intersects(a, b); });
    m.def("intersects", []( const Ball& b, const Ellipsoid& e, double tau )
          { return intersects(b, e, tau); }, "ball"_a, "ellipsoid"_a, "tau"_a = 1.0);
    m.def("intersects", []( const Ellipsoid& e, const Ball& b, double tau )
          { return intersects(b, e, tau); }, "ellipsoid"_a, "ball"_a, "tau"_a = 1.0);
    m.def("intersects", []( const Ball& b, const Simplex& s ) { return intersects(b, s); });
    m.def("intersects", []( const Simplex& s, const Ball& b ) { return intersects(b, s); });
    m.def("intersects", []( const Segment& g, const Ball& b ) { return intersects(g, b); });
    m.def("intersects", []( const Ball& b, const Segment& g ) { return intersects(g, b); });
    m.def("intersects", []( const Ball& b, const Halfspace& h ) { return intersects(b, h); });
    m.def("intersects", []( const Halfspace& h, const Ball& b ) { return intersects(b, h); });

    // ellipsoid vs ...
    m.def("intersects", []( const Ellipsoid& a, const Ellipsoid& b, double tau )
          { return intersects(a, b, tau); }, "A"_a, "B"_a, "tau"_a = 1.0);
    m.def("intersects", []( const Ellipsoid& e, const Simplex& s, double tau )
          { return intersects(e, s, tau); }, "ellipsoid"_a, "simplex"_a, "tau"_a = 1.0);
    m.def("intersects", []( const Simplex& s, const Ellipsoid& e, double tau )
          { return intersects(e, s, tau); }, "simplex"_a, "ellipsoid"_a, "tau"_a = 1.0);
    m.def("intersects", []( const Segment& g, const Ellipsoid& e, double tau )
          { return intersects(g, e, tau); }, "segment"_a, "ellipsoid"_a, "tau"_a = 1.0);
    m.def("intersects", []( const Ellipsoid& e, const Segment& g, double tau )
          { return intersects(g, e, tau); }, "ellipsoid"_a, "segment"_a, "tau"_a = 1.0);
    m.def("intersects", []( const Ellipsoid& e, const Halfspace& h, double tau )
          { return intersects(e, h, tau); }, "ellipsoid"_a, "halfspace"_a, "tau"_a = 1.0);
    m.def("intersects", []( const Halfspace& h, const Ellipsoid& e, double tau )
          { return intersects(e, h, tau); }, "halfspace"_a, "ellipsoid"_a, "tau"_a = 1.0);

    // simplex vs ...
    m.def("intersects", []( const Simplex& a, const Simplex& b ) { return intersects(a, b); });
    m.def("intersects", []( const Segment& g, const Simplex& s ) { return intersects(g, s); });
    m.def("intersects", []( const Simplex& s, const Segment& g ) { return intersects(g, s); });
    m.def("intersects", []( const Simplex& s, const Halfspace& h ) { return intersects(s, h); });
    m.def("intersects", []( const Halfspace& h, const Simplex& s ) { return intersects(s, h); });

    m.def("closest_point_in_simplex",
          []( const Eigen::VectorXd& p, const RowsXd& vertices )
          {
              ClosestPointResult r = closest_point_in_simplex(p, cols_from_rows(vertices));
              return py::make_tuple(r.point, r.distance_squared);
          },
          "point"_a, "vertices"_a,
          "Closest point in conv(vertices) (rows) to `point`; returns (point, dist_sq).");
    m.def("closest_point_in_simplex",
          []( const Eigen::VectorXd& p, const RowsXd& vertices, const Eigen::MatrixXd& M )
          {
              ClosestPointResult r = closest_point_in_simplex(p, cols_from_rows(vertices), M);
              return py::make_tuple(r.point, r.distance_squared);
          },
          "point"_a, "vertices"_a, "M"_a,
          "Closest point in the metric ||y||^2 = y^T M y; returns (point, dist_sq).");

    // ------------------------------------------------------------------
    //  Object trees
    // ------------------------------------------------------------------
    py::class_<BoxTree> box_tree(m, "BoxTree");
    box_tree.def(py::init([]( const RowsXd& lo, const RowsXd& hi )
                 { return BoxTree(cols_from_rows(lo), cols_from_rows(hi)); }),
                 "lo"_a, "hi"_a, "Box corners as rows: lo and hi have shape (n, d).")
        .def("object", &BoxTree::object, py::return_value_policy::reference_internal)
        .def("collisions", []( const BoxTree& t, const Eigen::VectorXd& p ) { return t.collisions(p); })
        .def("collisions", []( const BoxTree& t, const Box& q ) { return t.collisions(q); })
        .def("collisions", []( const BoxTree& t, const Ball& q ) { return t.collisions(q); })
        .def("collisions", []( const BoxTree& t, const Ellipsoid& q, double tau )
             { return t.collisions(q, tau); }, "query"_a, "tau"_a = 1.0)
        .def("collisions", []( const BoxTree& t, const Simplex& q ) { return t.collisions(q); })
        .def("collisions", []( const BoxTree& t, const Segment& q ) { return t.collisions(q); })
        .def("collisions", []( const BoxTree& t, const Halfspace& q ) { return t.collisions(q); })
        .def("self_collision_pairs", &BoxTree::self_collision_pairs,
             py::call_guard<py::gil_scoped_release>());
    def_common_tree(box_tree);

    py::class_<BallTree> ball_tree(m, "BallTree");
    ball_tree.def(py::init([]( const RowsXd& centers, const Eigen::VectorXd& radii )
                  { return BallTree(cols_from_rows(centers), radii); }),
                  "centers"_a, "radii"_a,
                  "Centers as rows (n, d); zero radii give a point tree for region queries.")
        .def("object", &BallTree::object, py::return_value_policy::reference_internal)
        .def("collisions", []( const BallTree& t, const Eigen::VectorXd& p ) { return t.collisions(p); })
        .def("collisions", []( const BallTree& t, const Box& q ) { return t.collisions(q); })
        .def("collisions", []( const BallTree& t, const Ball& q ) { return t.collisions(q); })
        .def("collisions", []( const BallTree& t, const Ellipsoid& q, double tau )
             { return t.collisions(q, tau); }, "query"_a, "tau"_a = 1.0)
        .def("collisions", []( const BallTree& t, const Simplex& q ) { return t.collisions(q); })
        .def("collisions", []( const BallTree& t, const Segment& q ) { return t.collisions(q); })
        .def("collisions", []( const BallTree& t, const Halfspace& q ) { return t.collisions(q); })
        .def("self_collision_pairs", &BallTree::self_collision_pairs,
             py::call_guard<py::gil_scoped_release>());
    def_common_tree(ball_tree);

    py::class_<EllipsoidTree> ell_tree(m, "EllipsoidTree");
    ell_tree.def(py::init([]( const RowsXd& mus, const py::array_t<double>& Sigmas,
                              double tau, int num_threads )
                 { return EllipsoidTree(ellipsoids_from_arrays(mus, Sigmas), tau, num_threads); }),
                 "mus"_a, "Sigmas"_a, "tau"_a = 1.0, "num_threads"_a = 0,
                 "Centers as rows (n, d); Sigmas as a stack (n, d, d).")
        .def(py::init<std::vector<Ellipsoid>, double, int>(),
             "ellipsoids"_a, "tau"_a = 1.0, "num_threads"_a = 0)
        .def_property_readonly("tau", &EllipsoidTree::tau)
        .def("rebuild", &EllipsoidTree::rebuild, "new_tau"_a)
        .def("object", &EllipsoidTree::object, py::return_value_policy::reference_internal)
        .def("collisions", []( const EllipsoidTree& t, const Eigen::VectorXd& p ) { return t.collisions(p); })
        .def("collisions", []( const EllipsoidTree& t, const Eigen::VectorXd& p, double tau )
             { return t.collisions(p, tau); })
        .def("collisions", []( const EllipsoidTree& t, const Box& q ) { return t.collisions(q); })
        .def("collisions", []( const EllipsoidTree& t, const Ball& q ) { return t.collisions(q); })
        .def("collisions", []( const EllipsoidTree& t, const Ellipsoid& q ) { return t.collisions(q); })
        .def("collisions", []( const EllipsoidTree& t, const Simplex& q ) { return t.collisions(q); })
        .def("collisions", []( const EllipsoidTree& t, const Segment& q ) { return t.collisions(q); })
        .def("collisions", []( const EllipsoidTree& t, const Halfspace& q ) { return t.collisions(q); })
        .def("self_collision_pairs", &EllipsoidTree::self_collision_pairs,
             py::call_guard<py::gil_scoped_release>());
    def_common_tree(ell_tree);

    py::class_<SimplexTree> sx_tree(m, "SimplexTree");
    sx_tree.def(py::init([]( const RowsXd& vertices, const RowsXi& cells, int num_threads )
                { return SimplexTree(cols_from_rows(vertices), icols_from_rows(cells), num_threads); }),
                "vertices"_a, "cells"_a, "num_threads"_a = 0,
                "Vertices as rows (nv, d); cells as rows of vertex indices (nc, K).")
        .def("object", &SimplexTree::object, py::return_value_policy::reference_internal)
        .def("first_collision", []( const SimplexTree& t, const Eigen::VectorXd& p )
             { return t.first_collision(p); })
        .def("collisions", []( const SimplexTree& t, const Eigen::VectorXd& p ) { return t.collisions(p); })
        .def("collisions", []( const SimplexTree& t, const Box& q ) { return t.collisions(q); })
        .def("collisions", []( const SimplexTree& t, const Ball& q ) { return t.collisions(q); })
        .def("collisions", []( const SimplexTree& t, const Ellipsoid& q, double tau )
             { return t.collisions(q, tau); }, "query"_a, "tau"_a = 1.0)
        .def("collisions", []( const SimplexTree& t, const Simplex& q ) { return t.collisions(q); })
        .def("collisions", []( const SimplexTree& t, const Segment& q ) { return t.collisions(q); })
        .def("collisions", []( const SimplexTree& t, const Halfspace& q ) { return t.collisions(q); })
        .def("self_collision_pairs", &SimplexTree::self_collision_pairs,
             py::call_guard<py::gil_scoped_release>());
    def_common_tree(sx_tree);

    // Tree-vs-tree collision pairs
    m.def("collision_pairs", []( const EllipsoidTree& a, const EllipsoidTree& b )
          { return collision_pairs(a, b); }, py::call_guard<py::gil_scoped_release>());
    m.def("collision_pairs", []( const SimplexTree& a, const EllipsoidTree& b )
          { return collision_pairs(a, b); }, py::call_guard<py::gil_scoped_release>());
    m.def("collision_pairs", []( const EllipsoidTree& a, const SimplexTree& b )
          { return collision_pairs(a, b); }, py::call_guard<py::gil_scoped_release>());
    m.def("collision_pairs", []( const SimplexTree& a, const SimplexTree& b )
          { return collision_pairs(a, b); }, py::call_guard<py::gil_scoped_release>());
    m.def("collision_pairs", []( const BallTree& a, const EllipsoidTree& b )
          { return collision_pairs(a, b); }, py::call_guard<py::gil_scoped_release>());
    m.def("collision_pairs", []( const BoxTree& a, const EllipsoidTree& b )
          { return collision_pairs(a, b); }, py::call_guard<py::gil_scoped_release>());
    m.def("collision_pairs", []( const BoxTree& a, const BoxTree& b )
          { return collision_pairs(a, b); }, py::call_guard<py::gil_scoped_release>());
    m.def("collision_pairs", []( const BallTree& a, const BallTree& b )
          { return collision_pairs(a, b); }, py::call_guard<py::gil_scoped_release>());
    m.def("collision_pairs", []( const BoxTree& a, const BallTree& b )
          { return collision_pairs(a, b); }, py::call_guard<py::gil_scoped_release>());
    m.def("collision_pairs", []( const BallTree& a, const SimplexTree& b )
          { return collision_pairs(a, b); }, py::call_guard<py::gil_scoped_release>());

    m.def("pick_ellipsoid_batches",
          []( const EllipsoidTree& t, int max_batches, int num_threads )
          { return pick_ellipsoid_batches(t, max_batches, num_threads); },
          "tree"_a, "max_batches"_a = -1, "num_threads"_a = 0,
          py::call_guard<py::gil_scoped_release>(),
          "Greedy batches of mutually non-overlapping ellipsoids (anchors = centers).");
    m.def("pick_ellipsoid_batches",
          []( const EllipsoidTree& t, const RowsXd& anchors, int max_batches, int num_threads )
          { return pick_ellipsoid_batches(t, cols_from_rows(anchors), max_batches, num_threads); },
          "tree"_a, "anchor_points"_a, "max_batches"_a = -1, "num_threads"_a = 0,
          py::call_guard<py::gil_scoped_release>());

    // ------------------------------------------------------------------
    //  SimplexMesh
    // ------------------------------------------------------------------
    py::class_<SimplexMesh>(m, "SimplexMesh")
        .def(py::init([]( const RowsXd& vertices, const RowsXi& cells, int num_threads )
             { return SimplexMesh(cols_from_rows(vertices), icols_from_rows(cells), num_threads); }),
             "vertices"_a, "cells"_a, "num_threads"_a = 0,
             "Vertices as rows (nv, d); cells as rows of vertex indices (nc, d+1).")
        .def_property_readonly("num_vertices", &SimplexMesh::num_vertices)
        .def_property_readonly("num_cells", &SimplexMesh::num_cells)
        .def_property_readonly("num_boundary_faces", &SimplexMesh::num_boundary_faces)
        .def_property_readonly("dim", &SimplexMesh::dim)
        .def_property_readonly("vertices",
             []( const SimplexMesh& mesh ) { return Eigen::MatrixXd(mesh.vertices().transpose()); })
        .def_property_readonly("cells",
             []( const SimplexMesh& mesh ) { return Eigen::MatrixXi(mesh.cells().transpose()); })
        .def_property_readonly("boundary_faces",
             []( const SimplexMesh& mesh ) { return Eigen::MatrixXi(mesh.boundary_faces().transpose()); })
        .def_property_readonly("cell_tree", &SimplexMesh::cell_tree,
             py::return_value_policy::reference_internal)
        .def("locate_points",
             []( const SimplexMesh& mesh, const RowsXd& points, int num_threads )
             {
                 std::pair<Eigen::VectorXi, Eigen::MatrixXd> res;
                 {
                     py::gil_scoped_release release; // GIL back before making Python objects
                     res = mesh.locate_points(cols_from_rows(points), num_threads);
                 }
                 return py::make_tuple(res.first, Eigen::MatrixXd(res.second.transpose()));
             },
             "points"_a, "num_threads"_a = 0,
             "Returns (cell_indices (m,), barycentric_coords (m, d+1)); index -1 = outside.")
        .def("point_is_in_mesh",
             []( const SimplexMesh& mesh, const RowsXd& points, int num_threads )
             {
                 auto flags = mesh.point_is_in_mesh(cols_from_rows(points), num_threads);
                 std::vector<bool> out(flags.size());
                 for ( int ii = 0; ii < flags.size(); ++ii ) { out[ii] = flags(ii); }
                 return out;
             },
             "points"_a, "num_threads"_a = 0, py::call_guard<py::gil_scoped_release>())
        .def("closest_points",
             []( const SimplexMesh& mesh, const RowsXd& points, int num_threads )
             { return Eigen::MatrixXd(mesh.closest_points(cols_from_rows(points), num_threads).transpose()); },
             "points"_a, "num_threads"_a = 0, py::call_guard<py::gil_scoped_release>())
        .def("eval_cg1",
             []( const SimplexMesh& mesh, const Eigen::VectorXd& f, const RowsXd& points,
                 bool reflect_exterior, int num_threads )
             {
                 Eigen::MatrixXd F = f.transpose(); // one function, one row
                 Eigen::MatrixXd vals = mesh.eval_cg1(F, cols_from_rows(points),
                                                      reflect_exterior, num_threads);
                 return Eigen::VectorXd(vals.row(0).transpose());
             },
             "vertex_values"_a, "points"_a, "reflect_exterior"_a = false, "num_threads"_a = 0,
             py::call_guard<py::gil_scoped_release>(),
             "Evaluate one CG1 function (values per vertex, shape (nv,)) at points (m, d).")
        .def("eval_cg1",
             []( const SimplexMesh& mesh, const RowsXd& F, const RowsXd& points,
                 bool reflect_exterior, int num_threads )
             {
                 Eigen::MatrixXd vals = mesh.eval_cg1(F, cols_from_rows(points),
                                                      reflect_exterior, num_threads);
                 return Eigen::MatrixXd(vals.transpose()); // (m, num_functions)
             },
             "vertex_values"_a, "points"_a, "reflect_exterior"_a = false, "num_threads"_a = 0,
             py::call_guard<py::gil_scoped_release>(),
             "Evaluate several functions ((nf, nv), one per row) at points; returns (m, nf).")
        .def("cells_intersecting",
             []( const SimplexMesh& mesh, const Ellipsoid& e, double tau )
             { return mesh.cells_intersecting(e, tau); },
             "ellipsoid"_a, "tau"_a = 1.0)
        .def("cell_ellipsoid_pairs", &SimplexMesh::cell_ellipsoid_pairs, "ellipsoids"_a,
             py::call_guard<py::gil_scoped_release>())
        .def("cell_pairs", &SimplexMesh::cell_pairs, "other"_a,
             py::call_guard<py::gil_scoped_release>())
        .def("closest_point",
             []( const SimplexMesh& mesh, const Eigen::VectorXd& p )
             { return mesh.closest_point(p); }, "point"_a);

    // ------------------------------------------------------------------
    //  KDTree and geometric sort
    // ------------------------------------------------------------------
    py::class_<KDTree>(m, "KDTree")
        .def(py::init([]( const RowsXd& points, int block_size )
             {
                 KDTree t;
                 t.block_size = block_size;
                 t.build(cols_from_rows(points));
                 return t;
             }),
             "points"_a, "block_size"_a = 32, "Points as rows: shape (n, d).")
        .def("__len__", &KDTree::size)
        .def_property_readonly("size", &KDTree::size)
        .def_property_readonly("dim", &KDTree::dim)
        .def("query",
             []( const KDTree& t, const Eigen::VectorXd& point, int k )
             {
                 auto res = t.query(point, k, 1);
                 return py::make_tuple(Eigen::VectorXi(res.first.col(0)),
                                       Eigen::VectorXd(res.second.col(0)));
             },
             "point"_a, "k"_a,
             "k nearest neighbors of one point; returns (indices (k,), sq_distances (k,)).")
        .def("query",
             []( const KDTree& t, const RowsXd& points, int k, int num_threads )
             {
                 std::pair<Eigen::MatrixXi, Eigen::MatrixXd> res;
                 {
                     py::gil_scoped_release release; // GIL back before making Python objects
                     res = t.query(cols_from_rows(points), k, num_threads);
                 }
                 return py::make_tuple(Eigen::MatrixXi(res.first.transpose()),
                                       Eigen::MatrixXd(res.second.transpose()));
             },
             "points"_a, "k"_a, "num_threads"_a = 0,
             "Batched k-NN; points (m, d) -> (indices (m, k), sq_distances (m, k)).");

    m.def("geometric_sort",
          []( const RowsXd& points ) { return geometric_sort(cols_from_rows(points).eval()); },
          "points"_a, "Axis-alternating geometric ordering of points (rows).");

    // ------------------------------------------------------------------
    //  2D visualization
    // ------------------------------------------------------------------
    py::class_<Color>(m, "Color")
        .def(py::init<double, double, double, double>(), "r"_a, "g"_a, "b"_a, "a"_a = 1.0)
        .def_readwrite("r", &Color::r)
        .def_readwrite("g", &Color::g)
        .def_readwrite("b", &Color::b)
        .def_readwrite("a", &Color::a);

    py::class_<Style>(m, "Style")
        .def(py::init([]( const Color& stroke, double stroke_width, const Color& fill )
             { return Style{stroke, stroke_width, fill}; }),
             "stroke"_a = colors::black(), "stroke_width"_a = 1.3,
             "fill"_a = colors::transparent())
        .def_readwrite("stroke", &Style::stroke)
        .def_readwrite("stroke_width", &Style::stroke_width)
        .def_readwrite("fill", &Style::fill);

    m.def("palette_color", &palette_color, "index"_a);
    m.def("with_alpha", &with_alpha, "color"_a, "alpha"_a);
    m.def("colormap_viridis", &colormap_viridis, "t"_a);

    py::class_<DrawTreeOptions>(m, "DrawTreeOptions")
        .def(py::init<>())
        .def_readwrite("objects", &DrawTreeOptions::objects)
        .def_readwrite("leaf_boxes", &DrawTreeOptions::leaf_boxes)
        .def_readwrite("node_boxes", &DrawTreeOptions::node_boxes)
        .def_readwrite("max_depth", &DrawTreeOptions::max_depth)
        .def_readwrite("color_node_boxes_by_depth", &DrawTreeOptions::color_node_boxes_by_depth)
        .def_readwrite("object_style", &DrawTreeOptions::object_style)
        .def_readwrite("leaf_box_style", &DrawTreeOptions::leaf_box_style)
        .def_readwrite("node_box_style", &DrawTreeOptions::node_box_style);

    py::class_<DrawKDTreeOptions>(m, "DrawKDTreeOptions")
        .def(py::init<>())
        .def_readwrite("points", &DrawKDTreeOptions::points)
        .def_readwrite("splits", &DrawKDTreeOptions::splits)
        .def_readwrite("max_depth", &DrawKDTreeOptions::max_depth)
        .def_readwrite("marker_radius_px", &DrawKDTreeOptions::marker_radius_px)
        .def_readwrite("color_splits_by_depth", &DrawKDTreeOptions::color_splits_by_depth)
        .def_readwrite("point_style", &DrawKDTreeOptions::point_style)
        .def_readwrite("split_style", &DrawKDTreeOptions::split_style);

    py::class_<FieldOptions>(m, "FieldOptions")
        .def(py::init<>())
        .def_readwrite("vmin", &FieldOptions::vmin)
        .def_readwrite("vmax", &FieldOptions::vmax)
        .def_readwrite("wireframe", &FieldOptions::wireframe)
        .def_readwrite("wire_style", &FieldOptions::wire_style);

    py::class_<Plot2D>(m, "Figure")
        .def(py::init<>())
        .def("add", []( Plot2D& f, const Ellipsoid& e, double tau, const Style& s )
             { f.add(e, tau, s); }, "ellipsoid"_a, "tau"_a = 1.0, "style"_a = Style{})
        .def("add", []( Plot2D& f, const Box& b, const Style& s ) { f.add(b, s); },
             "box"_a, "style"_a = Style{})
        .def("add", []( Plot2D& f, const Ball& b, const Style& s ) { f.add(b, s); },
             "ball"_a, "style"_a = Style{})
        .def("add", []( Plot2D& f, const Simplex& sx, const Style& s ) { f.add(sx, s); },
             "simplex"_a, "style"_a = Style{})
        .def("add", []( Plot2D& f, const Segment& g, const Style& s ) { f.add(g, s); },
             "segment"_a, "style"_a = Style{})
        .def("add", []( Plot2D& f, const Halfspace& h, const Style& s ) { f.add(h, s); },
             "halfspace"_a, "style"_a = Style{})
        .def("add_marker", []( Plot2D& f, const Eigen::Vector2d& p, double r, const Style& s )
             { f.add_marker(p, r, s); }, "point"_a, "radius_px"_a = 3.0, "style"_a = Style{})
        .def("add_polyline", []( Plot2D& f, const RowsXd& pts, const Style& s, bool closed )
             { f.add_polyline(cols_from_rows(pts), s, closed); },
             "points"_a, "style"_a = Style{}, "closed"_a = false)
        .def("add_text", []( Plot2D& f, const Eigen::Vector2d& p, const std::string& text,
                             double size_px, const Color& c )
             { f.add_text(p, text, size_px, c); },
             "point"_a, "text"_a, "size_px"_a = 12.0, "color"_a = colors::black())
        .def("axes", &Plot2D::axes, "on"_a)
        .def("set_bounds", &Plot2D::set_bounds, "bounds"_a)
        .def("to_svg", &Plot2D::to_svg, "width_px"_a = 900)
        .def("save_svg", &Plot2D::save_svg, "path"_a, "width_px"_a = 900)
        .def("save_png", &Plot2D::save_png, "path"_a, "width_px"_a = 900)
        .def("render_rgb", []( const Plot2D& f, int w ) { return image_to_numpy(f.render_rgb(w)); },
             "width_px"_a = 900, "Rasterize to a (height, width, 3) uint8 array.")
        .def("_repr_svg_", []( const Plot2D& f ) { return f.to_svg(700); });

    m.def("draw_tree", []( Plot2D& f, const EllipsoidTree& t, const DrawTreeOptions& o )
          { draw_tree(f, t, o); }, "figure"_a, "tree"_a, "options"_a = DrawTreeOptions{});
    m.def("draw_tree", []( Plot2D& f, const BoxTree& t, const DrawTreeOptions& o )
          { draw_tree(f, t, o); }, "figure"_a, "tree"_a, "options"_a = DrawTreeOptions{});
    m.def("draw_tree", []( Plot2D& f, const BallTree& t, const DrawTreeOptions& o )
          { draw_tree(f, t, o); }, "figure"_a, "tree"_a, "options"_a = DrawTreeOptions{});
    m.def("draw_tree", []( Plot2D& f, const SimplexTree& t, const DrawTreeOptions& o )
          { draw_tree(f, t, o); }, "figure"_a, "tree"_a, "options"_a = DrawTreeOptions{});

    m.def("draw_elements", []( Plot2D& f, const EllipsoidTree& t, const std::vector<int>& ii, const Style& s )
          { draw_elements(f, t, ii, s); });
    m.def("draw_elements", []( Plot2D& f, const BoxTree& t, const std::vector<int>& ii, const Style& s )
          { draw_elements(f, t, ii, s); });
    m.def("draw_elements", []( Plot2D& f, const BallTree& t, const std::vector<int>& ii, const Style& s )
          { draw_elements(f, t, ii, s); });
    m.def("draw_elements", []( Plot2D& f, const SimplexTree& t, const std::vector<int>& ii, const Style& s )
          { draw_elements(f, t, ii, s); });

    m.def("draw_batches", &draw_batches,
          "figure"_a, "tree"_a, "batches"_a, "fill_alpha"_a = 0.3, "stroke_width"_a = 1.3);
    m.def("draw_kdtree", &draw_kdtree, "figure"_a, "tree"_a, "options"_a = DrawKDTreeOptions{});
    m.def("draw_cg1_field", []( Plot2D& f, const SimplexMesh& mesh, const Eigen::VectorXd& v,
                                const FieldOptions& o )
          { draw_cg1_field(f, mesh, v, o); },
          "figure"_a, "mesh"_a, "vertex_values"_a, "options"_a = FieldOptions{});
}
