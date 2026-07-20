# ellipsoid_tree API reference {#mainpage}

**Exact intersection tests for ellipsoids and friends** — points, boxes, balls,
ellipsoids, and simplices in R^d; single pairs, tree-accelerated queries, and
tree-vs-tree sweeps. Header-only C++17, with Eigen as the only dependency.

This site is the **API reference** (generated from the headers). For the project
overview, installation, the intersection table, and worked examples with figures,
see the main documentation on GitHub:

- [Project README](https://github.com/NickAlger/ellipsoid_tree#readme) — overview, installation, the intersection table
- [Worked examples](https://github.com/NickAlger/ellipsoid_tree/tree/main/docs) — each page is a complete program, its actual output, and the figures it draws
- [Source repository](https://github.com/NickAlger/ellipsoid_tree)

## Where to start

Include the umbrella header `ellipsoid_tree/ellipsoid_tree.hpp` and reach for:

- **Objects** — `ellipsoid_tree::Box`, `ellipsoid_tree::Ball`, `ellipsoid_tree::Ellipsoid`, `ellipsoid_tree::Simplex`
- **Pairwise tests** — the `ellipsoid_tree::intersects` overloads
- **Spatial trees** — `ellipsoid_tree::BoxTree`, `ellipsoid_tree::BallTree`, `ellipsoid_tree::EllipsoidTree`, `ellipsoid_tree::SimplexTree`
- **Meshes & neighbors** — `ellipsoid_tree::SimplexMesh`, `ellipsoid_tree::KDTree`

Use **Classes** and **Files** in the sidebar to browse the full surface.
