# SPDX-License-Identifier: MIT
"""Tests for the ellipsoid_tree Python bindings.

Array convention at the Python boundary: points are rows — point sets are
(n, d), simplex vertices (K, d), mesh cells (num_cells, d+1).
"""

import numpy as np
import pytest

import ellipsoid_tree as et


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def unit_square_mesh(m):
    xs = np.linspace(0.0, 1.0, m + 1)
    vertices = np.array([[x, y] for y in xs for x in xs])
    cells = []
    for jj in range(m):
        for ii in range(m):
            v00 = jj * (m + 1) + ii
            v10, v01 = v00 + 1, v00 + (m + 1)
            v11 = v01 + 1
            cells.append([v00, v10, v11])
            cells.append([v00, v11, v01])
    return vertices, np.array(cells, dtype=np.int32)


def test_pairwise_intersects_known_cases():
    a = et.Ball(np.zeros(2), 1.0)
    assert et.intersects(a, et.Ball(np.array([1.9, 0.0]), 1.0))
    assert not et.intersects(a, et.Ball(np.array([2.1, 0.0]), 1.0))

    # Thin diagonal ellipsoid vs corner box: bounding boxes overlap, exact test says no
    S = rot(np.pi / 4) @ np.diag([1.6**2, 0.12**2]) @ rot(np.pi / 4).T
    E = et.Ellipsoid(np.zeros(2), S)
    box = et.Box(np.array([0.75, -1.2]), np.array([1.35, -0.6]))
    assert et.intersects(et.bounding_box(E, tau=1.0), box)
    assert not et.intersects(box, E, tau=1.0)

    tri = et.Simplex(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    assert et.intersects(np.array([0.2, 0.2]), tri)
    assert not et.intersects(np.array([0.8, 0.8]), tri)
    assert et.intersects(tri, et.Halfspace(np.array([1.0, 0.0]), 0.5))


def test_point_in_ellipsoid_matches_numpy():
    rng = np.random.default_rng(0)
    S = rot(0.7) @ np.diag([1.3, 0.2]) @ rot(0.7).T
    mu = np.array([0.3, -0.2])
    E = et.Ellipsoid(mu, S)
    Sinv = np.linalg.inv(S)
    for _ in range(200):
        p = rng.normal(size=2)
        expected = (p - mu) @ Sinv @ (p - mu) <= 1.0
        assert et.intersects(p, E, tau=1.0) == expected


def test_ellipsoid_tree_matches_brute_force():
    rng = np.random.default_rng(1)
    n, d = 40, 2
    mus = rng.normal(size=(n, d))
    Sigmas = np.empty((n, d, d))
    for ii in range(n):
        R = rot(rng.uniform(0, np.pi))
        Sigmas[ii] = R @ np.diag([0.05, 0.01]) @ R.T
    tree = et.EllipsoidTree(mus, Sigmas, tau=1.0)
    assert len(tree) == n

    query = et.Ellipsoid(np.array([0.2, 0.1]), np.diag([0.3, 0.05]))
    hits = sorted(tree.collisions(query))
    brute = [ii for ii in range(n)
             if et.intersects(tree.object(ii), query, tau=1.0)]
    assert hits == brute

    p = np.array([0.0, 0.0])
    assert sorted(tree.collisions(p)) == [
        ii for ii in range(n) if et.intersects(p, tree.object(ii), tau=1.0)]

    with pytest.raises(ValueError):
        tree.collisions(p, 2.0)  # larger than the build tau
    tree.rebuild(2.0)
    assert tree.tau == 2.0
    tree.collisions(p, 2.0)  # fine now


def test_point_cloud_coverage_matches_numpy():
    rng = np.random.default_rng(2)
    pts = rng.uniform(-1.5, 1.5, size=(80, 2))
    cloud = et.BallTree(pts, np.zeros(80))
    S = rot(-0.4) @ np.diag([1.1**2, 0.35**2]) @ rot(-0.4).T
    mu = np.array([0.1, -0.1])
    covered = sorted(cloud.collisions(et.Ellipsoid(mu, S), tau=1.0))
    mask = np.einsum("nd,dk,nk->n", pts - mu, np.linalg.inv(S), pts - mu) <= 1.0
    assert covered == list(np.nonzero(mask)[0])


def test_kdtree_matches_numpy():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(500, 3))
    tree = et.KDTree(pts)
    queries = rng.normal(size=(20, 3))

    inds, dsqs = tree.query(queries, 4)
    assert inds.shape == (20, 4) and dsqs.shape == (20, 4)
    for qq in range(20):
        brute = np.argsort(np.sum((pts - queries[qq]) ** 2, axis=1))[:4]
        assert list(inds[qq]) == list(brute)

    one_inds, one_dsqs = tree.query(queries[0], 4)
    assert one_inds.shape == (4,)
    assert list(one_inds) == list(inds[0])
    assert np.allclose(one_dsqs, dsqs[0])


def test_mesh_location_eval_and_closest():
    vertices, cells = unit_square_mesh(6)
    mesh = et.SimplexMesh(vertices, cells)
    assert mesh.num_cells == 72
    assert mesh.vertices.shape == (49, 2)
    assert mesh.cells.shape == (72, 3)

    rng = np.random.default_rng(4)
    pts = rng.uniform(0.05, 0.95, size=(30, 2))
    cell_inds, bary = mesh.locate_points(pts)
    assert bary.shape == (30, 3)
    for ii in range(30):
        assert cell_inds[ii] >= 0
        cell_vertices = vertices[cells[cell_inds[ii]]]
        assert np.allclose(bary[ii] @ cell_vertices, pts[ii], atol=1e-12)

    # CG1 reproduces linear functions exactly
    f = 1.5 * vertices[:, 0] - 0.7 * vertices[:, 1] + 0.25
    vals = mesh.eval_cg1(f, pts)
    assert np.allclose(vals, 1.5 * pts[:, 0] - 0.7 * pts[:, 1] + 0.25, atol=1e-10)

    outside = np.array([[1.4, 0.3], [-0.2, 1.3]])
    assert mesh.point_is_in_mesh(outside) == [False, False]
    cp = mesh.closest_points(outside)
    assert np.allclose(cp, np.clip(outside, 0.0, 1.0), atol=1e-10)

    E = et.Ellipsoid(np.array([0.5, 0.5]), np.diag([0.02, 0.005]))
    hit = sorted(mesh.cells_intersecting(E, tau=1.0))
    tri_of = lambda cc: et.Simplex(vertices[cells[cc]])
    brute = [cc for cc in range(mesh.num_cells)
             if et.intersects(E, tri_of(cc), tau=1.0)]
    assert hit == brute


def test_batches_partition_and_are_independent():
    rng = np.random.default_rng(5)
    n = 60
    mus = rng.normal(size=(n, 2))
    Sigmas = np.empty((n, 2, 2))
    for ii in range(n):
        R = rot(rng.uniform(0, np.pi))
        Sigmas[ii] = R @ np.diag([0.03, 0.008]) @ R.T
    tree = et.EllipsoidTree(mus, Sigmas, tau=1.0)

    batches = et.pick_ellipsoid_batches(tree)
    flat = sorted(ii for batch in batches for ii in batch)
    assert flat == list(range(n))
    for batch in batches:
        for aa in range(len(batch)):
            for bb in range(aa + 1, len(batch)):
                assert not et.intersects(
                    tree.object(batch[aa]), tree.object(batch[bb]), tau=1.0)


def test_collision_pairs_between_trees():
    vertices, cells = unit_square_mesh(4)
    sx_tree = et.SimplexTree(vertices, cells)
    mus = np.array([[0.2, 0.2], [0.8, 0.7], [2.5, 2.5]])
    Sigmas = np.tile(np.diag([0.01, 0.01]), (3, 1, 1))
    e_tree = et.EllipsoidTree(mus, Sigmas, tau=1.0)

    pairs = set(et.collision_pairs(sx_tree, e_tree))
    tri_of = lambda cc: et.Simplex(vertices[cells[cc]])
    brute = {(cc, jj) for cc in range(len(sx_tree)) for jj in range(3)
             if et.intersects(e_tree.object(jj), tri_of(cc), tau=1.0)}
    assert pairs == brute
    assert all(jj != 2 for (_, jj) in pairs)  # the far ellipsoid touches nothing


def test_figure_and_drawing(tmp_path):
    fig = et.Figure()
    fig.add(et.Ellipsoid(np.zeros(2), np.diag([4.0, 1.0])),
            tau=1.0, style=et.Style(fill=et.with_alpha(et.palette_color(0), 0.3)))
    fig.add(et.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0])))
    svg = fig.to_svg()
    assert "<svg" in svg and "<ellipse" in svg
    assert fig._repr_svg_().startswith("<svg")

    img = fig.render_rgb(300)
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3

    fig.save_svg(str(tmp_path / "fig.svg"))
    fig.save_png(str(tmp_path / "fig.png"), 300)
    assert (tmp_path / "fig.svg").exists()
    assert (tmp_path / "fig.png").read_bytes()[:4] == b"\x89PNG"

    # Tree drawing and the CG1 field renderer run end to end
    vertices, cells = unit_square_mesh(4)
    mesh = et.SimplexMesh(vertices, cells)
    fig2 = et.Figure()
    opts = et.FieldOptions()
    opts.wireframe = True
    et.draw_cg1_field(fig2, mesh, vertices[:, 0].copy(), opts)
    et.draw_tree(fig2, mesh.cell_tree)
    assert "<polygon" in fig2.to_svg()

    kd = et.KDTree(np.random.default_rng(6).uniform(size=(30, 2)), block_size=1)
    fig3 = et.Figure()
    et.draw_kdtree(fig3, kd)
    assert "<circle" in fig3.to_svg()


def test_simplex_and_sort_roundtrips():
    V = np.array([[0.0, 0.0], [2.0, 0.5], [1.0, 3.0]])
    assert np.allclose(et.Simplex(V).vertices, V)

    pts = np.random.default_rng(7).normal(size=(50, 2))
    order = et.geometric_sort(pts)
    assert sorted(order) == list(range(50))
