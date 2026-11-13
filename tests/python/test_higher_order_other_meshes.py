"""The higher order with surfaces of meshes of their own (reference/inputs-bodies §2,
reference/inputs-model-objects §1).

A damping lid in global coordinates, the dipoles, a pressure surface and an interior surface of a
body given as meshes of their own solve as the same elements of the body's mesh do, byte for byte;
a lid follows its body's position and yaw; flat patches with B-spline ones become bilinear B-spline
patches of the same surface; the combinations that cannot be joined are refused."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import flat, heave_pieces, moonpool_barge, panel_size_nuv, part, plate_column, spline
from meshes import ACV_FACES, MOONPOOL_FACES, PLATE_FACES, PLATE_NUV, box_panels, face_panels
from problems import G, model, view

HO = {"higher_order": True, "diffraction_formulation": "total"}
QS = ("added_mass", "radiation_damping", "excitation_diffraction")


def _faces(faces):
    return np.array([face_panels(o, du, dv, 1, 1, n)[0] for o, du, dv, n in faces])


def _split(vertices, rows, k, symmetry="xz_and_yz"):
    """The flat mesh of the elements 1..k and that of the others, and the whole mesh."""
    rows = np.asarray(rows)
    return (flat(vertices[:k], rows[:k], symmetry=symmetry), flat(vertices[k:], rows[k:], symmetry=symmetry),
            flat(vertices, rows, symmetry=symmetry))  # fmt: skip


def _same(a, b):
    for q in QS:
        np.testing.assert_array_equal(view(a, q), view(b, q), err_msg=q)


def _barge_rows():
    q = _faces(MOONPOOL_FACES)
    nuv = [(max(1, round(np.linalg.norm(x[1] - x[0]) * 2)), max(1, round(np.linalg.norm(x[3] - x[0]) * 2))) for x in q]
    return q, [(a, b, 3, 3, 3, 4) for a, b in nuv]


def test_a_lid_mesh_of_its_own():
    # the moonpool lid as a mesh in global coordinates (the barge at the origin) and as element 7
    # of the barge's mesh placed with it: the same results, the lid elevation too
    w = np.sqrt(np.array([0.8, 1.6]) * G)
    q, rows = _barge_rows()
    hull, lid, _ = _split(q, rows, 6)
    barge = moonpool_barge()
    own = model({"name": "b", "mesh": hull}, w, [0.0, 45.0], options=HO,
                lids=[{"name": "l", "mesh": lid, "damping_factor": 0.5}]).solve()  # fmt: skip
    placed = {"name": "l", "mesh": part(barge, 7, 7), "placed_with": "b", "damping_factor": 0.5}
    rec = model({"name": "b", "mesh": part(barge, 1, 6)}, w, [0.0, 45.0], options=HO, lids=[placed]).solve()
    _same(own, rec)
    np.testing.assert_array_equal(own.lids["l"].elevation_diffraction, rec.lids["l"].elevation_diffraction)


def test_dipoles_pressure_and_interior_meshes_of_their_own():
    w = 2 * np.pi / np.array([5.0, 8.0])
    # the column with its heave plate: the plate's dipoles as a mesh of their own
    q = _faces(PLATE_FACES)
    hull, plate, whole = _split(q, [(a, b, 3, 3, 3, 4) for a, b in PLATE_NUV], 3)
    rec = plate_column()
    a = model({"name": "body", "mesh": hull, "dipoles": [{"mesh": plate}]}, w, [0.0], options=HO).solve()
    b = model(rec, w, [0.0], options=HO).solve()
    _same(a, b)
    # a generalized mode with a piece on the hull's elements: the appended elements keep out of it
    modes = heave_pieces(part(hull, 1, 3))
    body = {"name": "body", "mesh": hull, "dipoles": [{"mesh": plate}], "generalized_modes": modes}
    a = model(body, w, [0.0], options=HO).solve()
    b = model({**rec, "generalized_modes": heave_pieces(rec["mesh"])}, w, [0.0], options=HO).solve()
    _same(a, b)
    # the air-cushion vessel: its cushion as a mesh of its own
    v = _faces(ACV_FACES)
    hull, cushion, whole = _split(v, [(*panel_size_nuv(x, 1.0), 3, 3, 3, 4) for x in v], 6)
    modes = [{"name": "p0", "shape": "sign_x"}, {"name": "p1", "shape": "uniform"}]
    env = {"depth": 20.0}
    a = model({"name": "acv", "mesh": hull, "pressure_surfaces": [{"mesh": cushion, "pressure_modes": modes}]},
              w, [0.0],
              environment=env, options=HO).solve()  # fmt: skip
    surf = {"mesh": part(whole, 7, 7), "pressure_modes": modes}
    b = model({"name": "acv", "mesh": part(whole, 1, 6), "pressure_surfaces": [surf]}, w, [0.0], environment=env,
              options=HO).solve()  # fmt: skip
    _same(a, b)
    # the box with its interior surface as a mesh of its own (at an irregular frequency region too)
    box = np.array(box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True, lid=True))
    hull, lid, whole = _split(box, [(4, 2, 3, 3, 3, 4), (2, 2, 3, 3, 3, 4), (4, 2, 3, 3, 3, 4), (4, 2, 3, 3, 3, 4)], 3)
    wi = np.sqrt(np.array([1.0, 2.0, 3.5]) * G)
    a = model({"mesh": hull, "interior_surface": {"method": "mesh", "mesh": lid}}, wi, [30.0], options=HO).solve()
    rec = {"mesh": part(whole, 1, 3), "interior_surface": {"method": "mesh", "mesh": part(whole, 4, 4)}}
    b = model(rec, wi, [30.0], options=HO).solve()
    _same(a, b)


def _full(q, rows):
    """The four images of the quadrant patches (the vertex order reversed in the odd ones)."""
    out, nuv = [], []
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        p = q * [sx, sy, 1]
        out.append(p if sx * sy > 0 else p[:, ::-1])
        nuv += [r if sx * sy > 0 else (r[1], r[0], *r[2:]) for r in rows]  # the reversed order exchanges u and v
    return np.concatenate(out), nuv


def test_a_lid_follows_its_body():
    # the barge without symmetry flags at (3, -2) with yaw 30 degrees and its lid in global
    # coordinates there: the added mass and damping of the barge at the origin with its lid
    q, rows = _barge_rows()
    v, nuv = _full(q, rows)
    hull_idx = [i for i in range(len(v)) if i % 7 != 6]
    lid_idx = [i for i in range(len(v)) if i % 7 == 6]
    hull = flat(v[hull_idx], np.asarray(nuv)[hull_idx])
    yaw = math.radians(30.0)
    rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    lid_at = flat(v[lid_idx] @ rot.T + [3.0, -2.0, 0.0], np.asarray(nuv)[lid_idx])
    lid0 = flat(v[lid_idx], np.asarray(nuv)[lid_idx])
    w = np.sqrt(np.array([0.8, 1.6]) * G)
    opts = {**HO, "diffraction": False}
    moved = model({"name": "b", "mesh": hull, "position": [3.0, -2.0, 0.0], "yaw": 30.0}, w, (), options=opts,
                  lids=[{"name": "l", "mesh": lid_at, "damping_factor": 0.5}]).solve()  # fmt: skip
    origin = model({"name": "b", "mesh": hull}, w, (), options=opts,
                   lids=[{"name": "l", "mesh": lid0, "damping_factor": 0.5}]).solve()  # fmt: skip
    none = model({"name": "b", "mesh": hull}, w, (), options=opts).solve()
    for q_ in ("added_mass", "radiation_damping"):
        a, b, c = view(moved, q_), view(origin, q_), view(none, q_)
        assert np.abs(a - b).max() <= 1e-10 * np.abs(b).max(), q_
        assert np.abs(b - c).max() >= 0.1 * np.abs(c).max(), q_  # the lid changes them


def test_flat_patches_with_spline_ones():
    # the box's interior surface given as a bilinear B-spline patch: the box's flat patches become
    # B-spline patches of the same surface, the results those of the flat box (round-off)
    box = np.array(box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True, lid=True))
    rows = [(4, 2, 3, 3, 3, 4), (2, 2, 3, 3, 3, 4), (4, 2, 3, 3, 3, 4), (4, 2, 3, 3, 3, 4)]
    hull, lid, whole = _split(box, rows, 3)
    p = box[3]  # the interior element: u from vertex 0 to 3, v from 0 to 1 (the conversion of a flat patch)
    lid_spline = spline([(2, 2, (0, 0, 1, 1), (0, 0, 1, 1), np.array([[p[0], p[3]], [p[1], p[2]]]))],
                        [(2, 4, 3, 3, 3, 4)], symmetry="xz_and_yz")  # fmt: skip
    w = np.sqrt(np.array([1.0, 2.0]) * G)
    a = model({"mesh": hull, "interior_surface": {"method": "mesh", "mesh": lid_spline}}, w, [30.0], options=HO).solve()
    rec = {"mesh": part(whole, 1, 3), "interior_surface": {"method": "mesh", "mesh": part(whole, 4, 4)}}
    b = model(rec, w, [30.0], options=HO).solve()
    for q in QS:
        x, y = view(a, q), view(b, q)
        assert np.abs(x - y).max() <= 1e-10 * np.abs(y).max(), q


def test_meshes_that_cannot_be_joined():
    w = [1.0]
    q, rows = _barge_rows()
    hull, lid, _ = _split(q, rows, 6)
    for kw, match in (
        ({"position": [1.0, 0.0, 0.0]},
         r"lids/l: its mesh \(joined to the first body 'b'\) with symmetry needs the body on its"),
        ({"yaw": 90.0}, "with symmetry needs the body on its symmetry planes without yaw"),
    ):
        with pytest.raises(ValueError, match=match):
            model({"name": "b", "mesh": hull, **kw}, w, [0.0], options=HO,
                  lids=[{"name": "l", "mesh": lid, "damping_factor": 0.5}]).route()  # fmt: skip
    half = flat(q[6:], np.asarray(rows)[6:], symmetry="yz")
    with pytest.raises(ValueError, match="must have the symmetry of the body's mesh"):
        model({"name": "b", "mesh": hull}, w, [0.0], options=HO,
              lids=[{"name": "l", "mesh": half, "damping_factor": 0.5}]).route()  # fmt: skip
    m = hb.Model()
    m.meshes.analytic("cyl", "circular_cylinder", (4, 4, 3, 3, 3, 4), radius=1.0, draft=0.5)
    disc = flat(np.array([face_panels((0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), 1, 1, (0, 0, 1))[0]]),
                [(2, 2, 3, 3, 3, 4)], symmetry="xz_and_yz")  # fmt: skip
    with pytest.raises(ValueError, match="analytic patches join only analytic ones"):
        model({"mesh": lambda mo, name: mo.meshes.analytic(name, "circular_cylinder", (4, 4, 3, 3, 3, 4), radius=1.0,
                                                           draft=0.5),
               "interior_surface": {"method": "mesh", "mesh": disc}}, w, [0.0], options=HO).route()  # fmt: skip


def test_inspections_list_the_meshes_of_their_own():
    # as in the low order: the body's elements, then the other meshes by their own elements; a lid
    # in global coordinates with the water (body -1)
    q, rows = _barge_rows()
    hull, lid, _ = _split(q, rows, 6)
    barge = moonpool_barge()
    ins = {"diffraction": False, **HO}
    own = model({"name": "b", "mesh": hull}, [1.0], (), options=ins,
                lids=[{"name": "l", "mesh": lid, "damping_factor": 0.5}])  # fmt: skip
    placed = {"name": "l", "mesh": part(barge, 7, 7), "placed_with": "b", "damping_factor": 0.5}
    rec = model({"name": "b", "mesh": part(barge, 1, 6)}, [1.0], (), options=ins, lids=[placed])
    a, b = own.panels(include_reflections=False), rec.panels(include_reflections=False)
    assert list(a.body) == [0] * 6 + [-1] and list(a.element) == [1, 2, 3, 4, 5, 6, 1]
    assert list(a.role) == ["wetted"] * 6 + ["lid"] and a.owner[6] == 0
    np.testing.assert_allclose(a.centroids, b.centroids, atol=1e-12)
    t = own.tessellation(divisions=(2, 2), include_reflections=False)
    assert set(t.element[t.body == 0]) == set(range(1, 7)) and set(t.element[t.body == -1]) == {1}
    assert sorted(set(own.mesh_quality().element)) == list(range(1, 7))
    # the column's dipoles as a mesh of their own
    f = _faces(PLATE_FACES)
    hull, plate, _ = _split(f, [(a, b, 3, 3, 3, 4) for a, b in PLATE_NUV], 3)
    col = model({"name": "body", "mesh": hull, "dipoles": [{"mesh": plate}]}, [1.0], (), options=ins)
    p = col.panels(include_reflections=False)
    assert list(p.role) == ["wetted"] * 3 + ["dipole"] * 2 and list(p.element) == [1, 2, 3, 1, 2]
    one = model(plate_column(), [1.0], (), options=ins).panels(include_reflections=False)
    np.testing.assert_allclose(p.centroids, one.centroids, atol=1e-12)
    assert sorted(set(col.mesh_quality().element)) == [1, 2, 3]
    t = col.tessellation(divisions=(2, 2), include_reflections=False)
    assert set(t.element) == {1, 2, 3}


def test_a_lid_of_the_second_body():
    # two barges without symmetry flags, the second at (12, 3) with yaw 30 degrees: its moonpool lid
    # in global coordinates (joined to the first body's mesh) gives A and B of the lid placed with
    # the second body on elements of its mesh, to rounding (the lid's rows belong to another body,
    # summed in another order)
    q, rows = _barge_rows()
    v, nuv = _full(q, rows)
    nuv = np.asarray(nuv)
    hull_idx = [i for i in range(len(v)) if i % 7 != 6]
    lid_idx = [i for i in range(len(v)) if i % 7 == 6]
    hull = flat(v[hull_idx], nuv[hull_idx])
    both = flat(v[hull_idx + lid_idx], nuv[hull_idx + lid_idx])
    n = len(hull_idx)
    yaw = math.radians(30.0)
    rot = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    lid_at = flat(v[lid_idx] @ rot.T + [12.0, 3.0, 0.0], nuv[lid_idx])
    w = np.sqrt(np.array([0.8, 1.6]) * G)
    opts = {**HO, "diffraction": False}
    second = {"name": "b", "position": [12.0, 3.0, 0.0], "yaw": 30.0}
    joined = model([{"name": "a", "mesh": hull}, {**second, "mesh": hull}], w, (), options=opts,
                   lids=[{"name": "l", "mesh": lid_at, "damping_factor": 0.5}]).solve()  # fmt: skip
    placed = {"name": "l", "mesh": part(both, n + 1, n + len(lid_idx)), "placed_with": "b", "damping_factor": 0.5}
    own = model([{"name": "a", "mesh": hull}, {**second, "mesh": part(both, 1, n)}], w, (), options=opts,
                lids=[placed]).solve()  # fmt: skip
    none = model([{"name": "a", "mesh": hull}, {**second, "mesh": hull}], w, (), options=opts).solve()
    for q_ in ("added_mass", "radiation_damping"):
        a, b, c = view(joined, q_), view(own, q_), view(none, q_)
        assert np.abs(a - b).max() <= 1e-8 * np.abs(b).max(), q_  # 3e-10 found
        assert np.abs(b - c)[:, 6:, 6:].max() >= 0.1 * np.abs(c[:, 6:, 6:]).max(), q_  # the lid acts on b
