"""Mesh quality: hb.Model.mesh_quality per element (size, aspect, elements per shortest
wavelength, flags) and per body (volumes, edge counts), and the mesh warnings of the
rules (the warning codes: coarse mesh 2, mesh orientation 3, mesh open 4, mesh volume 5) with the
body and element."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import flat, surface
from meshes import box_panels, cylinder_panels
from problems import add_mesh, analytic, model


def body_of(v, **kw):
    return {"mesh": surface(v, symmetry="xz_and_yz"), **kw}


def quality_model(body, omegas=(1.0,), **kw):
    return model(body, list(omegas), (), options={"diffraction": False, **kw.pop("options", {})}, **kw)


def mesh_warnings(body, omegas=(1.0,), **kw):
    """The mesh warnings of the rules (as the engine's warnings are issued: hb.HydroBEMWarning)."""
    found = quality_model(body, omegas, **kw).check()
    return [hb.HydroBEMWarning(d.message, d.code, d.location) for d in found if d.code >= 2]


BOX = box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True)


def test_a_closed_box():
    q = quality_model(body_of(BOX)).mesh_quality()
    assert all(f == () for f in q.flags)
    assert {k: int(v[0]) for k, v in vars(q.edges).items()} == {"inconsistent": 0, "free": 0, "non_manifold": 0,
                                                              "open_waterline": 0}  # fmt: skip
    assert np.allclose(q.volumes, 8.0)
    # squares 0.25 x 0.25 (the quarter box 2 x 1 x 1 in 8 x 4 x 4): d = 0.25 sqrt 2, aspect d^2 / 2A = 1
    lam = 2 * math.pi / hb.wavenumber(1.0)
    assert np.allclose(q.size, 0.25 * math.sqrt(2)) and np.allclose(q.aspect, 1.0)
    assert np.allclose(q.per_wavelength, lam / (0.25 * math.sqrt(2)))
    assert mesh_warnings(body_of(BOX)) == []


def test_a_flipped_panel_and_a_gap():
    v = BOX.copy()
    v[5] = v[5][::-1]
    (w,) = [m for m in mesh_warnings(body_of(v)) if m.code == 3]
    assert w.location.body == 0  # the element: the first of the panel and its neighbours
    q = quality_model(body_of(v)).mesh_quality()
    flagged = [i + 1 for i, f in enumerate(q.flags) if "orientation" in f]
    assert 6 in flagged and w.location.element in flagged
    assert q.edges.inconsistent[0] == 16  # its four edges in the four reflected copies
    gap = np.delete(BOX, 9, axis=0)
    codes = {m.code for m in mesh_warnings(body_of(gap))}
    assert 4 in codes
    q = quality_model(body_of(gap)).mesh_quality()
    assert q.edges.free[0] > 0 and any("free_edge" in f for f in q.flags)


def test_normals_into_the_body():
    (w,) = mesh_warnings(body_of(BOX[:, ::-1]))
    assert w.code == 5 and "not positive" in str(w) and w.location == hb.Location(body=0)


def test_coarse_panels_at_short_waves():
    body = body_of(BOX)
    k = 2 * math.pi / (5.0 * 0.25 * math.sqrt(2))  # five panel diameters per wavelength
    w = math.sqrt(9.80665 * k)
    (m,) = mesh_warnings(body, omegas=(0.5, w))
    assert m.code == 2 and m.location.body == 0
    q = quality_model(body, (0.5, w)).mesh_quality()
    assert q.min_per_wavelength[0] == pytest.approx(2 * math.pi / k / q.size.max(), rel=1e-9)
    assert q.per_wavelength[m.location.element - 1] == pytest.approx(q.min_per_wavelength[0])


def test_a_column_on_the_sea_bed_is_open_there():
    col = cylinder_panels(1.0, 3.0, n_phi=8, n_z=6, n_r=2)
    side = col[np.abs(col[:, :, 2].max(axis=1) + 3.0) > 1e-9]  # without the bottom (its panels lie in z = -3)
    m = {"mesh": surface(side, symmetry="xz_and_yz")}
    assert mesh_warnings(m, environment={"depth": 3.0}) == []


def test_higher_order_volumes_and_spans():
    r, d = 1.0, 0.5
    mesh = analytic("circular_cylinder", (4, 2, 3, 3, 3, 4), radius=r, draft=d)
    q = quality_model({"mesh": mesh}, options={"higher_order": True}).mesh_quality()
    np.testing.assert_allclose(q.volumes, math.pi * r * r * d, rtol=1e-14)  # the exact map: rounding (3e-16)
    edges = (q.edges.inconsistent[0], q.edges.free[0], q.edges.non_manifold[0], q.edges.open_waterline[0])
    assert edges == (0, 0, 0, 0) and np.all(q.aspect >= 1.0) and np.all(np.isfinite(q.per_wavelength))


def _patch_edges(v):
    mesh = add_mesh(hb.Model(), "m", flat(v, (1, 1, 3, 3, 3, 4), symmetry="xz_and_yz"))
    q = mesh.quality()
    return (q.edges.inconsistent[0], q.edges.free[0], q.edges.non_manifold[0], q.edges.open_waterline[0]), q.flags


def test_the_sides_of_patch_meshes():
    box = box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True)  # bottom, x face, y face
    assert _patch_edges(box)[0] == (0, 0, 0, 0)
    # a side bordering smaller patches (the bottom split in x, the y face not): still closed
    for k in (2, 8):  # with 8, every point of the long side lies at a corner of two neighbours
        split = box_panels(1.0, 0.5, 0.5, n=(k, 1, 1), quarter=True)
        assert _patch_edges(np.concatenate([split[:k], box[1:3]]))[0] == (0, 0, 0, 0)
    # the y face missing: the sides of the bottom and the x face next to it are free, and the
    # waterline of the x face ends in the open (each in the four copies of the symmetry)
    edges, flags = _patch_edges(box[:2])
    assert edges[1] == 8 and edges[3] == 4 and "free_edge" in flags[0]
    # a patch turned over: its sides run with those of its neighbours
    turned = box.copy()
    turned[0] = turned[0][::-1]
    edges, flags = _patch_edges(turned)
    assert edges[0] > 0 and "orientation" in flags[0]
    # in the rules: the warning of an open patch surface
    found = mesh_warnings({"mesh": flat(box[:2], (1, 1, 3, 3, 3, 4), symmetry="xz_and_yz")},
                          options={"higher_order": True})
    assert any("8 patch sides below the waterline meet no other patch" in str(w) for w in found)


def test_panels_need_not_meet_at_their_vertices():
    # a T-junction: the bottom split in x, the y face not (its bottom edge along the edges of the
    # smaller panels); the low order needs no common vertices, so the surface is closed
    def edges(v):
        q = add_mesh(hb.Model(), "m", surface(v, symmetry="xz_and_yz")).quality()
        return (q.edges.inconsistent[0], q.edges.free[0], q.edges.non_manifold[0], q.edges.open_waterline[0]), q.flags

    box = box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True)
    for k in (2, 8):
        split = box_panels(1.0, 0.5, 0.5, n=(k, 1, 1), quarter=True)
        assert edges(np.concatenate([split[:k], box[1:3]]))[0] == (0, 0, 0, 0)
    # a small panel turned over along the long edge: the senses disagree
    split = box_panels(1.0, 0.5, 0.5, n=(2, 1, 1), quarter=True)
    turned = np.concatenate([split[:2], box[1:3]])
    turned[1] = turned[1][::-1]
    found, flags = edges(turned)
    assert found[0] > 0 and "orientation" in flags[1]
    # a gap of a hundredth of the edge between the small panels and the long edge: free
    gap = np.concatenate([split[:2], box[1:3]])
    gap[:2, :, 1] *= 0.99
    assert edges(gap)[0][1] > 0

def test_a_gap_in_a_mesh_of_its_own_names_no_record_of_the_hull():
    # the hull's sides with the bottom as a pressure surface given as a mesh of its own (two patches):
    # the appended elements close the hull; a gap between them is reported without naming an element
    # of the hull (the flagged elements belong to the other mesh)
    box = box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True)
    bottom = box_panels(1.0, 0.5, 0.5, n=(2, 1, 1), quarter=True)[:2]

    def warnings_with(cushion):
        rows = (1, 1, 3, 3, 3, 4)
        body = {"mesh": flat(box[1:3], rows, symmetry="xz_and_yz"),
                "pressure_surfaces": [{"mesh": flat(cushion, rows, symmetry="xz_and_yz"),
                                       "pressure_modes": [{"name": "p0", "shape": "uniform"}]}]}  # fmt: skip
        return mesh_warnings(body, options={"higher_order": True})

    assert warnings_with(bottom) == []
    gap = bottom.copy()
    gap[1][np.isclose(gap[1][:, 0], 0.5), 0] += 0.01  # the second patch moved off the first
    found = [w for w in warnings_with(gap) if "not closed" in str(w)]
    assert len(found) == 1 and found[0].location.body == 0 and found[0].location.element is None
