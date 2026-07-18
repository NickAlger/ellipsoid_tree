# etree

**Exact ellipsoid intersection tests and spatial trees.** Header-only C++17,
no CGAL, no Boost — Eigen is the only dependency.

<p align="center">
<img src="docs/img/batch_picking__batches.svg" width="640">
</p>

*Above: a family of anisotropic ellipsoids partitioned into batches of
mutually non-overlapping members ([example](docs/examples/batch_picking.md)) —
the geometric kernel of point-spread-function probing, where impulse
responses with disjoint supports can be probed in a single operator
application.*

## What it does

- **Pairwise intersection tests** between points, boxes, balls, ellipsoids,
  simplices, segments, and halfspaces — see the
  [visual gallery of every pair](docs/examples/intersections_gallery.md).
  Ellipsoid–ellipsoid overlap is the exact Gilitschenski–Hanebeck test;
  ellipsoid-vs-simplex/box are small exact convex solvers; simplex–simplex
  is an LP feasibility test; the rest are closed form.
- **Spatial trees** over each object type (`BoxTree`, `BallTree`,
  `EllipsoidTree`, `SimplexTree`): "which elements intersect this query?"
  in logarithmic time, for any query type in the table, with tiered
  exact pruning for ellipsoid queries.
- **Tree-vs-tree collision** by dual-tree simultaneous descent:
  all intersecting pairs between two families in one traversal, including
  self-collision (the input to batch picking).
- **Simplicial meshes** (`SimplexMesh`): point location with barycentric
  coordinates, closest boundary point, CG1 finite element evaluation,
  mesh × ellipsoid and [mesh × mesh](docs/examples/mesh_mesh.md) queries.
- **Supporting cast**: k-nearest-neighbor `KDTree`, axis-alternating
  `geometric_sort`, greedy non-overlapping
  [ellipsoid batch picking](docs/examples/batch_picking.md).
- **Optional zero-dependency 2D visualization** (`etree/plot2d.hpp`):
  SVG and PNG figures of objects, trees, queries, and CG1 fields — every
  figure in the documentation is drawn with it.

Ellipsoids follow the convention E(τ) = {x : (x−μ)ᵀ Σ⁻¹ (x−μ) ≤ τ²} with
Σ symmetric positive definite; the scale τ is passed at call time. All
objects are solid and closed (touching counts as intersecting); the
solver-backed tests are exact up to documented solver tolerances.
Everything is dimension-generic except the visualization (2D).

## Quick start

```cpp
#include "etree/etree.hpp"
using namespace etree;

Ellipsoid A{mu_a, Sigma_a}, B{mu_b, Sigma_b};
bool overlap = intersects(A, B, /*tau=*/1.0);

EllipsoidTree tree(family_of_ellipsoids, /*tau=*/1.0);
std::vector<int> hits = tree.collisions(some_box);
auto batches = pick_ellipsoid_batches(tree);
```

CMake: `add_subdirectory(ellipsoid_tree)` (or FetchContent) and link
`etree::etree`; Eigen is found via `find_package(Eigen3)` with an automatic
pinned download as fallback.

## Examples ("show, don't tell")

Every page in [`docs/examples/`](docs/README.md) is a complete program, its
actual output, and the figures it draws — regenerated from the code by
`docs/generate_examples.py` and checked in CI, so the documentation cannot
drift from the library:

- [Pairwise intersection tests, visually](docs/examples/intersections_gallery.md)
- [Which points of a cloud does an ellipsoid cover?](docs/examples/point_cloud.md)
- [EllipsoidTree spatial queries](docs/examples/ellipsoid_tree_queries.md)
- [Batch picking](docs/examples/batch_picking.md)
- [SimplexMesh: location, closest points, mesh × ellipsoid](docs/examples/mesh_queries.md)
- [Mesh vs mesh cell pairs](docs/examples/mesh_mesh.md)
- [KDTree nearest neighbors and its partition](docs/examples/kdtree_knn.md)
- [Rendering a CG1 finite element field](docs/examples/cg1_field.md)

## Building and testing

Header-only: add `include/` to your include path. To run the tests and
examples:

```sh
cmake -S . -B build && cmake --build build -j && ctest --test-dir build
python3 docs/generate_examples.py   # regenerate the example documentation
```

## References and acknowledgements

- I. Gilitschenski and U. D. Hanebeck, *A Direct Method for Checking Overlap
  of Two Hyperellipsoids*, Sensor Data Fusion: Trends, Solutions,
  Applications (SDF), 2014 — the ellipsoid–ellipsoid overlap test.
- R. P. Brent, *Algorithms for Minimization without Derivatives*,
  Prentice-Hall, 1973 — the scalar minimizer used inside it.
- [Eigen](https://eigen.tuxfamily.org) for linear algebra;
  [stb_image_write](https://github.com/nothings/stb) (public domain) for PNG
  encoding; [doctest](https://github.com/doctest/doctest) for testing.

MIT license.
