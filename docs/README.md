# ellipsoid_tree examples

Each page shows a complete program, its actual output, and the figures it draws — all regenerated from the code by `docs/generate_examples.py`.

- [Batch picking: partition ellipsoids into non-overlapping groups](examples/batch_picking.md)
- [Rendering a piecewise linear finite element field](examples/cg1_field.md)
- [EllipsoidTree: spatial queries against a family of ellipsoids](examples/ellipsoid_tree_queries.md)
- [Pairwise intersection tests, visually](examples/intersections_gallery.md)
- [KDTree: k nearest neighbors, and the partition behind them](examples/kdtree_knn.md)
- [Mesh vs mesh: all intersecting cell pairs between two meshes](examples/mesh_mesh.md)
- [SimplexMesh: point location, closest points, and mesh-vs-ellipsoid](examples/mesh_queries.md)
- [Which points of a cloud does an ellipsoid cover?](examples/point_cloud.md)

## Python

- [Non-overlapping ellipsoid batches (Jupyter notebook)](../examples/python_batch_picking.ipynb) — the batch-picking example via the Python bindings; re-executed in CI (outputs not diffed) and rendered inline by GitHub.
