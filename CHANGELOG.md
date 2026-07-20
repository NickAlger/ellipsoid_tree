# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-07-20

### Changed
- **Breaking: the import name, C++ namespace, include directory and CMake
  package are renamed `etree` → `ellipsoid_tree`.** The distribution name
  (`ellipsoid-tree`) and the repository are unchanged, so `pip install
  ellipsoid-tree` is unaffected; only `import etree` → `import ellipsoid_tree`
  and `#include "etree/..."` → `#include "ellipsoid_tree/..."` change, along
  with `find_package(etree)` → `find_package(ellipsoid_tree)`, the target
  `etree::etree` → `ellipsoid_tree::ellipsoid_tree`, and the `ETREE_` macro and
  CMake-option prefix → `ELLIPSOID_TREE_`.

  The old name collided with [The Etree Library](https://www.cs.cmu.edu/~droh/papers/etree-tr.pdf)
  (Tu, O'Hallaron and López, CMU Quake project) — an established C library for
  manipulating large octrees on disk, used to build finite element meshes for
  earthquake simulation. That is the same problem domain as this library, so a
  shared name risked genuine confusion as well as concrete `find_package` and
  include-path conflicts for any project using both. (`etree` is also widely
  read as XML's ElementTree, via `xml.etree` and `lxml.etree`.) No short alias
  is shipped; documentation and tests use a reader-chosen
  `import ellipsoid_tree as et`.

[0.2.0]: https://github.com/NickAlger/ellipsoid_tree/releases/tag/v0.2.0

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
