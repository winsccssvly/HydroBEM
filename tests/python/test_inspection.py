"""Model inspection for viewers: the elements as the engine uses them
(Model.panels), a quadrilateral mesh of the surfaces (Model.tessellation) and points of
patches (MeshObject.patch_points). Meshes are procedural (tests/python/meshes.py)."""

import math
from collections import Counter

import numpy as np
import pytest

import hydrobem as hb

from bodies import acv_body, part, plate_column, surface
from meshes import box_panels, face_panels, flat_plate
from problems import analytic, model


def box(**kw):
    mesh = surface(box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True), symmetry="xz_and_yz")
    return {"mesh": mesh, "mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.5],
            "radii_of_gyration": [[0.5, 0, 0], [0, 1, 0], [0, 0, 1]], **kw}  # fmt: skip


def test_low_order_panels_area_normals_and_reflections():
    m = model(box(), [1.0], (), options={"diffraction": False})
    full, stored = m.panels(), m.panels(include_reflections=False)
    assert len(full.areas) == 4 * len(stored.areas)
    # the wetted box 4 x 2 x 1: bottom 8, sides 2 (4 + 2)
    assert full.areas.sum() == pytest.approx(20.0, rel=1e-12)
    assert set(full.role) == {"wetted"} and np.all(full.owner == -1) and np.all(full.tank == -1)
    assert np.allclose(np.linalg.norm(full.normals, axis=1), 1.0)
    # into the fluid: out of the box (centre (0, 0, -0.5))
    assert np.all(np.einsum("ij,ij->i", full.normals, full.centroids - [0.0, 0.0, -0.5]) > 0.0)
    assert {tuple(r) for r in full.reflection} == {(1, 1), (-1, 1), (1, -1), (-1, -1)}
    # the vertex order of a copy keeps the normal: (v2 - v0) x (v3 - v1) along it
    v = full.vertices
    n = np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 1])
    assert np.all(np.einsum("ij,ij->i", n, full.normals) > 0.0)
    # the placement of the body
    moved = model(box(position=[10.0, -2.0, 0.0], yaw=90.0), [1.0], (), options={"diffraction": False}).panels()
    c, cm = full.centroids, moved.centroids
    assert np.allclose(cm, np.c_[10.0 - c[:, 1], -2.0 + c[:, 0], c[:, 2]])


def test_roles_tanks_pressure_surfaces_sheets_and_dampers():
    hull = box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True)
    walls = box_panels(0.8, 0.4, 0.5, n=(4, 2, 2), quarter=True)[:, ::-1] + [0.0, 0.0, -0.1]
    mesh = surface(np.concatenate([hull, walls]), symmetry="xz_and_yz")
    tank = {"mesh": part(mesh, len(hull) + 1, len(hull) + len(walls))}
    q = model({"mesh": part(mesh, 1, len(hull)), "tanks": [tank]}, [1.0], (),
              options={"diffraction": False}).panels(include_reflections=False)  # fmt: skip
    t = q.element > len(hull)
    assert set(q.role[t]) == {"tank"} and np.all(q.tank[t] == 0) and np.all(q.tank[~t] == -1)
    # tank walls: normals into the tank fluid
    assert np.all(np.einsum("ij,ij->i", q.normals[t], q.centroids[t] - [0.0, 0.0, -0.35]) < 0.0)
    acv = acv_body()
    q = model(acv, [1.0], (), options={"diffraction": False}).panels()
    assert set(q.role) == {"wetted", "pressure_surface"} and set(q.owner[q.role == "pressure_surface"]) == {0}
    # higher order: the two sheets of the column, then one as a damper of the body and of the water
    col = plate_column()
    ho = {"diffraction": False, "higher_order": True}
    q = model(col, [1.0], (), options=ho).panels(include_reflections=False)
    assert list(q.role) == ["wetted"] * 3 + ["dipole"] * 2
    src = col["mesh"][0]
    rigid = {"mesh": part(src, 1, 3), "dipoles": [{"mesh": part(src, 4, 4)}]}
    damper = {"mesh": part(src, 5, 5), "damping_factor": 1.0}
    q = model({**rigid, "dipole_dampers": [damper]}, [1.0], (), options=ho).panels(include_reflections=False)
    assert q.role[3] == "dipole" and q.role[4] == "dipole_damper" and q.owner[4] == 0
    q = model(rigid, [1.0], (), options=ho, dampers=[damper]).panels(include_reflections=False)
    assert q.role[4] == "fixed_damper" and q.owner[4] == 0 and q.body[4] == -1  # no body's element


def test_generated_surfaces_and_water_dampers_in_the_panels():
    hull = box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True)
    box = surface(hull, symmetry="xz_and_yz")
    # a generated interior surface as the engine uses it, and as a mesh of the model
    m = model({"mesh": box, "interior_surface": {"method": "graded"}}, [1.0, 2.0], (), options={"diffraction": False})
    q = m.bodies["body"].panels(role="interior_surface", include_reflections=False)
    assert set(q.role) == {"interior_surface"} and len(q.element) > 0 and np.all(q.body == 0)
    assert np.allclose(q.centroids[:, 2], 0.0) and np.all(q.normals[:, 2] > 0.999)
    assert len(m.bodies["body"].panels(role="interior_surface").element) == 4 * len(q.element)
    assert np.sum(q.areas) == pytest.approx(2.0 * 1.0, rel=1e-12)  # the quarter waterplane
    lid = m.bodies["body"].trimmed_mesh("lid", "interior_surface")
    assert len(lid.vertices) == len(q.element) and lid.symmetry == "xz_and_yz" and not np.any(lid.source_elements)
    r0 = m.solve()
    m.bodies["body"].update(interior_surface="mesh", interior_surface_mesh="lid")  # the same surface given
    r1 = m.solve()
    for k in ("added_mass", "radiation_damping"):  # the same panels: the same bits
        assert r1.quantity(k).tobytes() == r0.quantity(k).tobytes(), k
    # a given interior surface of another mesh, and a generated control surface on request
    assert len(m.bodies["body"].panels(role="interior_surface", include_reflections=False).element) == len(q.element)
    cs = {"method": "cylinder", "radius": 4.0, "depth": 3.0, "inner_radius": 3.0, "panel_size": 1.0}
    m = model({"mesh": box, "modes": [], "control_surface": cs}, [1.0], [0.0],
              options={"drift_control_surface": True})  # fmt: skip
    c = m.bodies["body"].panels(role="control_surface", include_reflections=False)
    assert len(c.element) > 0 and set(c.role) == {"control_surface"} and np.all(c.body == 0)
    assert "control_surface" not in set(m.panels(include_reflections=False).role)
    # a fixed damper on elements of the body's mesh is no body's element
    plate = face_panels((0.0, 0.0, -1.5), (1.4, 0, 0), (0, 1.4, 0), 2, 2, (0, 0, 1))
    mesh = surface(np.concatenate([hull, plate]), symmetry="xz_and_yz")
    n = len(hull)
    m = model({"mesh": part(mesh, 1, n)}, [1.0], [0.0], options={"diffraction_formulation": "scattered"},
              dampers=[{"mesh": part(mesh, n + 1, n + 4), "damping_factor": 1.0}])  # fmt: skip
    every, own = m.panels(include_reflections=False), m.bodies["body"].panels(include_reflections=False)
    w = every.role == "fixed_damper"
    assert w.sum() == 4 and np.all(every.body[w] == -1) and set(every.element[w]) == {n + 1, n + 2, n + 3, n + 4}
    assert set(own.role) == {"wetted"} and len(own.element) == n
    fixed = m.panels(role="fixed_damper", include_reflections=False)
    assert np.all(fixed.body == -1) and set(fixed.role) == {"fixed_damper"}
    assert np.array_equal(fixed.element, every.element[w])


def test_panels_by_role():
    # every role of one body, and a lid of no body
    hull = box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True, lid=True)  # the waterplane: 32 elements at the end
    walls = box_panels(0.8, 0.4, 0.5, n=(4, 2, 2), quarter=True)[:, ::-1] + [0.0, 0.0, -0.1]
    mesh = surface(np.concatenate([hull, walls]), symmetry="xz_and_yz")
    nh, nt = len(hull) - 32, len(walls)
    lid = surface(flat_plate(center=(6, 0)))
    b = {"mesh": part(mesh, 1, nh), "tanks": [{"mesh": part(mesh, nh + 33, nh + 32 + nt)}],
         "interior_surface": part(mesh, nh + 1, nh + 32)}
    m = model(b, [1.0], (), options={"diffraction": False}, lids=[{"name": "far", "mesh": lid}])
    every = m.panels(include_reflections=False)
    assert set(every.role) == {"wetted", "tank", "interior_surface", "lid"}
    assert len(m.bodies["body"].panels(role="wetted", include_reflections=False).element) == nh
    tanks = m.bodies["body"].panels(role="tank", include_reflections=False)
    assert set(tanks.role) == {"tank"} and sorted(tanks.element) == list(range(nh + 33, nh + 33 + nt))
    interior = m.bodies["body"].panels(role="interior_surface", include_reflections=False)
    assert set(interior.role) == {"interior_surface"} and len(interior.element) == 32
    lids = m.panels(role="lid", include_reflections=False)
    assert set(lids.role) == {"lid"} and np.all(lids.body == -1) and np.allclose(lids.centroids[:, 2], 0.0)
    # the panels of the model are those of its bodies, then the lids and fixed dampers of no body
    own = m.bodies["body"].panels(include_reflections=False)
    assert np.array_equal(every.centroids, np.concatenate([own.centroids, lids.centroids]))
    assert len(m.bodies["body"].panels(role="dipole").element) == 0
    with pytest.raises(ValueError, match="role"):
        m.panels(role="hull")


def test_higher_order_patches_tessellation_and_points():
    r, d = 1.0, 0.5
    m = model({"mesh": analytic("circular_cylinder", (4, 2, 3, 3, 3, 4), radius=r, draft=d)}, [1.0], (),
              options={"diffraction": False, "higher_order": True})  # fmt: skip
    q = m.panels()
    # the wetted cylinder: side 2 pi r d, bottom pi r^2
    assert q.areas.sum() == pytest.approx(2 * math.pi * r * d + math.pi * r**2, rel=1e-14)  # the exact map (7e-16)
    t = m.tessellation(divisions=(6, 4))
    x, n = t.vertices.reshape(-1, 3), t.normals.reshape(-1, 3)
    assert len(t.element) == len(q.element) * 24
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)  # also at the axis, where the map is degenerate
    side = np.abs(n[:, 2]) < 0.5
    assert np.allclose(np.hypot(x[side, 0], x[side, 1]), r) and np.all(np.hypot(x[:, 0], x[:, 1]) <= r + 1e-12)
    # into the fluid: radially out on the side, down on the bottom
    assert np.allclose(n[side, :2], x[side, :2] / r) and np.allclose(n[~side], [0.0, 0.0, -1.0])
    assert np.allclose(x[~side, 2], -d)
    uv = np.array([[-1.0, -1.0], [0.0, 0.0], [0.5, -0.25], [1.0, 1.0]])
    mesh = m.meshes[m.bodies["body"].mesh]
    pp = mesh.patch_points(1, uv)
    assert pp.points.shape == (4, 3) and np.allclose(np.linalg.norm(pp.normals, axis=1), 1.0)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        mesh.patch_points(1, [[1.5, 0.0]])
    with pytest.raises(ValueError, match="does not exist"):
        mesh.patch_points(9, uv)


# ---- the keys of the panels and the tessellation (one walk over the elements of a scope)


def keys(q):
    """The seven keys of every row of a Panels or Tessellation answer."""
    return [(int(b), str(r), int(t), int(o), int(e), int(p), int(s[0]), int(s[1]))
            for b, r, t, o, e, p, s in zip(q.body, q.role, q.tank, q.owner, q.element, q.part, q.reflection,
                                           strict=True)]  # fmt: skip


def joined(scope, role=None, divisions=2, include_reflections=True):
    """The panels and tessellation of a scope (a model or a body) with the same arguments, checked:
    the keys unique in the panels, every quad with the keys of one panels row and every row with
    its quads (a low-order panel one, a patch divisions^2), no element 0. Returns both."""
    p = scope.panels(role=role, include_reflections=include_reflections)
    t = scope.tessellation(role=role, divisions=divisions, include_reflections=include_reflections)
    kp = keys(p)
    assert len(set(kp)) == len(kp), "the keys of the panels are not unique"
    assert np.all(p.element >= 1) and np.all(t.element >= 1) and np.all(p.part >= 0)
    counts = Counter(keys(t))
    assert set(counts) == set(kp), "the keys of the tessellation are not those of the panels"
    assert set(counts.values()) <= {1, divisions * divisions}
    return p, t


def test_the_keys_join_the_tessellation_to_the_panels_low_order():
    hull = box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True, lid=True)
    walls = box_panels(0.8, 0.4, 0.5, n=(4, 2, 2), quarter=True)[:, ::-1] + [0.0, 0.0, -0.1]
    sheet = face_panels((0.0, 0.0, -0.3), (0.3, 0, 0), (0, 0.15, 0), 2, 2, (0, 0, 1))
    plate = face_panels((0.0, 0.0, -1.5), (1.4, 0, 0), (0, 1.4, 0), 2, 2, (0, 0, 1))
    mesh = surface(np.concatenate([hull, walls, sheet, plate]), symmetry="xz_and_yz")
    nh, nt, ns = len(hull) - 32, len(walls), len(sheet)
    w0 = nh + 32 + 1
    tank = {"mesh": part(mesh, w0, w0 + nt - 1), "dipoles": [{"mesh": part(mesh, w0 + nt, w0 + nt + ns - 1)}]}
    other = surface(face_panels((0.0, 0.0, -0.8), (0.5, 0, 0), (0, 0.5, 0), 2, 2, (0, 0, 1)), symmetry="xz_and_yz")
    b = {"mesh": part(mesh, 1, nh), "tanks": [tank], "interior_surface": part(mesh, nh + 1, nh + 32),
         "dipoles": [{"mesh": other, "ranges": [(1, 1), (4, 4)]}, {"mesh": other, "ranges": [(2, 3)]}]}  # fmt: skip
    p0 = w0 + nt + ns
    m = model(b, [1.0], (), options={"diffraction": False},
              lids=[{"name": "far", "mesh": surface(flat_plate(center=(6, 0)))}],
              dampers=[{"mesh": part(mesh, p0, p0 + len(plate) - 1), "damping_factor": 1.0},
                       {"mesh": surface(flat_plate(center=(-6, 0), z=-1.0)), "damping_factor": 0.5}])  # fmt: skip
    for include in (True, False):
        p, t = joined(m, include_reflections=include)
        assert len(t.element) == len(p.element)  # low order: one quad a panel
    p = m.panels()
    # the owners: the sheet of a tank its place in the tank's dipoles, tank walls their tank, the
    # dipoles of another mesh their places in the body's dipoles, the fixed dampers theirs
    assert set(p.owner[p.role == "tank"]) == {0} and set(p.tank[p.role == "tank"]) == {0}
    inside = (p.role == "dipole") & (p.tank == 0)
    assert inside.sum() == 4 * ns and set(p.owner[inside]) == {0}
    outside = (p.role == "dipole") & (p.tank == -1)
    assert set(p.owner[outside]) == {0, 1}
    assert set(p.owner[p.role == "fixed_damper"]) == {0, 1} and np.all(p.body[p.role == "fixed_damper"] == -1)
    # the elements of the dipoles of the other mesh: the second dipole takes elements 2 and 3
    assert set(p.element[outside & (p.owner == 1)]) == {2, 3} and set(p.element[outside & (p.owner == 0)]) == {1, 4}
    # every index is zero-based but the element; the roles of a body only in a body's query
    q, _ = joined(m.bodies["body"])
    assert np.all(q.body == 0) and "fixed_damper" not in set(q.role) and "lid" not in set(q.role)
    # several roles at once: the rows of those roles, in the order of the default query
    some = m.panels(role=["lid", "wetted"])
    sel = np.isin(p.role, ["wetted", "lid"])
    assert keys(some) == [k for k, s in zip(keys(p), sel, strict=True) if s]
    joined(m, role=["tank", "dipole", "fixed_damper"])


def test_the_keys_of_a_trim_that_cuts_elements_and_breaks_the_symmetry():
    # a pitch keeps the plane y = 0 only: the copies x -> -x become elements of their own, each
    # with the reflection of the copy it was cut from; a cut element has two parts
    mesh = surface(box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True), symmetry="xz_and_yz")
    m = model({"mesh": mesh, "trim": [0.3, 5.0, 0.0]}, [1.0], (), options={"diffraction": False})
    for include in (True, False):
        p, _ = joined(m, include_reflections=include)
    p = m.panels(include_reflections=False)
    assert {tuple(r) for r in p.reflection} == {(1, 1), (-1, 1)}
    assert p.part.max() >= 1  # a pentagon below the waterline: a quadrilateral and a triangle
    full = m.panels()
    assert {tuple(r) for r in full.reflection} == {(1, 1), (-1, 1), (1, -1), (-1, -1)}
    assert len(full.element) == 2 * len(p.element)
    assert m.mesh_quality().part.max() == p.part.max()
    # higher order: a patch cut by the waterline into parts, the interior surface the trim generates
    ho = {"diffraction": False, "higher_order": True}
    cyl = analytic("circular_cylinder", (4, 2, 3, 3, 3, 4), radius=1.0, draft=0.5)
    m = model({"mesh": cyl, "trim": [0.1, 4.0, 0.0], "interior_surface": "waterline"}, [1.0], (), options=ho)
    p, t = joined(m, divisions=3)
    gen = p.role == "interior_surface"
    assert gen.sum() > 0 and np.all(p.element[gen] >= 1)
    assert len(t.element) == 9 * len(p.element)


def test_every_role_is_tessellated_higher_order():
    ho = {"diffraction": False, "higher_order": True}
    col = plate_column()
    src = col["mesh"][0]
    rigid = {"mesh": part(src, 1, 3), "dipoles": [{"mesh": part(src, 4, 4)}]}
    damper = {"mesh": part(src, 5, 5), "damping_factor": 1.0}
    for m in (model(col, [1.0], (), options=ho), model({**rigid, "dipole_dampers": [damper]}, [1.0], (), options=ho),
              model(rigid, [1.0], (), options=ho, dampers=[damper])):  # fmt: skip
        for include in (True, False):
            p, t = joined(m, divisions=2, include_reflections=include)
            assert len(t.element) == 4 * len(p.element)
    # a pressure surface and a lid on other meshes the higher order joins to the body's mesh
    acv = acv_body()
    m = model(acv, [1.0], (), options={"diffraction": False})
    joined(m)


def test_the_generated_control_surface_is_tessellated_on_its_exact_surface():
    # low order: the generated cylinder control surface is a patch mesh; its quads lie on the
    # exact cylinder and their areas tend to the exact areas of its patches
    r, d = 4.0, 3.0
    box = surface(box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True), symmetry="xz_and_yz")
    cs = {"method": "cylinder", "radius": r, "depth": d, "inner_radius": 3.0, "panel_size": 1.0}
    m = model({"mesh": box, "modes": [], "control_surface": cs}, [1.0], [0.0], options={"drift_control_surface": True})
    p, t = joined(m, role=["wetted", "control_surface"], divisions=4)
    c = p.role == "control_surface"
    side = c & (np.abs(np.hypot(p.centroids[:, 0], p.centroids[:, 1]) - r) < 1e-9)
    assert p.areas[side].sum() == pytest.approx(2 * math.pi * r * d, rel=1e-9)
    tc = t.role == "control_surface"
    x = t.vertices[tc].reshape(-1, 3)
    on_side = np.abs(t.normals[tc].reshape(-1, 3)[:, 2]) < 0.5
    assert np.allclose(np.hypot(x[on_side, 0], x[on_side, 1]), r)

    def quad_area(v):
        return 0.5 * np.linalg.norm(np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 1]), axis=1)

    exact = p.areas[c].sum()
    errors = []
    for n in (2, 8, 32):
        tn = m.tessellation(role="control_surface", divisions=n)
        errors.append(abs(quad_area(tn.vertices).sum() - exact) / exact)
    assert errors[0] > errors[1] > errors[2] and errors[2] < 1e-3


def test_the_divisions_of_a_tessellation():
    ho = {"diffraction": False, "higher_order": True}
    rows = (4, 2, 3, 3, 3, 4)
    m = model({"mesh": analytic("circular_cylinder", rows, radius=1.0, draft=0.5)}, [1.0], (), options=ho)
    n = len(m.panels(include_reflections=False).element)
    assert len(m.tessellation(divisions=3, include_reflections=False).element) == 9 * n
    assert len(m.tessellation(divisions=(3, 2), include_reflections=False).element) == 6 * n
    # one quad per span of the potential discretization of each patch
    mesh = m.meshes[m.bodies["body"].mesh]
    spans = int(sum(int(a) * int(b) for a, b in np.asarray(mesh.discretization)[:, :2]))
    assert len(m.tessellation(divisions="spans", include_reflections=False).element) == spans
    assert len(mesh.tessellation(divisions="spans", include_reflections=False).element) == spans
    for bad, err in ((0, ValueError), ("all", ValueError), ((1, 2, 3), TypeError), (2.0, TypeError),
                     (True, TypeError)):  # fmt: skip
        with pytest.raises(err):
            m.tessellation(divisions=bad)
    with pytest.raises(ValueError, match="divisions"):
        m.tessellation(divisions=300)


def test_a_mesh_alone_and_the_refused_names():
    m = model(box(), [1.0], (), options={"diffraction": False})
    mesh = m.meshes[m.bodies["body"].mesh]
    t = mesh.tessellation()
    assert set(t.role) == {""} and np.all(t.body == -1) and np.all(t.owner == -1) and np.all(t.tank == -1)
    assert np.all(t.part == 0) and set(t.element) == set(mesh.elements().element)
    el = mesh.elements()
    assert isinstance(el, hb.MeshElements) and set(el.distinct_vertices) <= {3, 4}
    # only the role words; the list names and the old row words are refused, naming the roles
    for old in ("body", "wet", "dipoles", "sheet", "damper", "pressure", "plate", "interior", "lids"):
        with pytest.raises(ValueError, match="role must be one of wetted, interior_surface, dipole"):
            m.panels(role=old)
        with pytest.raises(ValueError, match="role must be one of"):
            m.tessellation(role=[old])
    with pytest.raises(ValueError, match="not of a body"):
        m.bodies["body"].panels(role="free_surface")
    # an empty list (a filter that matched nothing) is no request for the default roles
    for q in (m.panels, m.tessellation, m.bodies["body"].panels, m.bodies["body"].tessellation):
        with pytest.raises(ValueError, match="role is an empty list"):
            q(role=[])
    with pytest.raises(TypeError):
        m.panels(expand=False)
    with pytest.raises(TypeError):
        m.tessellation(nu=2, nv=2)
    with pytest.raises(AttributeError):
        mesh.records()
    with pytest.raises(ValueError, match="role must be one of wetted"):
        m.bodies["body"].trimmed_mesh("x", "body")
