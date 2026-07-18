# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-07-18

Initial public release.

### Added
- Exact pairwise intersection tests (`intersects`) across points, boxes, balls,
  ellipsoids, and simplices, with segment and halfspace query objects — a full
  table of closed-form and solver-backed cells, exact to documented tolerance.
- Spatial trees `BoxTree`, `BallTree`, `EllipsoidTree`, `SimplexTree` with
  tree-vs-object `collisions`, tree-vs-tree `collision_pairs`, and
  `self_collision_pairs` (the overlap graph of a family).
- `SimplexMesh`: point location with barycentric coordinates, closest boundary
  point, CG1 finite-element evaluation, and mesh × ellipsoid / mesh × mesh queries.
- `KDTree` k-nearest-neighbor queries, axis-alternating `geometric_sort`, and
  greedy non-overlapping ellipsoid batch picking.
- Optional zero-dependency 2D visualization (`etree/plot2d.hpp`): SVG and PNG.
- Python bindings (pybind11) with a NumPy interface (points are `(n, d)` rows);
  figures render inline in Jupyter.
- Doxygen API reference published to GitHub Pages.
- CMake install/export (`find_package(etree)`) and a pip package
  (`ellipsoid-tree`; import name `etree`).

[0.1.0]: https://github.com/NickAlger/ellipsoid_tree/releases/tag/v0.1.0
