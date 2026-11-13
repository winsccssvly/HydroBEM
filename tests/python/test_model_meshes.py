"""The meshes of a model as its objects: read from files, made from analytic shapes and
control surfaces, written, the trimmed surfaces of a body, and their queries. Written meshes read
back to the same items; the analytic shapes have their patches; the trimmed surface of a body names
the elements of its source. (The volumes of the shapes and meshes: test_queries_by_owner.py.)"""


import numpy as np
import pytest

import hydrobem as hb

from bodies import coarse_cylinder, lo_free_cylinder, spline_cylinder, surface
from meshes import box_panels, cylinder_panels
from problems import add_mesh, flat_surface, model


def _value(v):
    return v.tobytes() if isinstance(v, np.ndarray) else v


def _items(obj, skip=()) -> dict:
    """Every item and child of a model object, arrays as bytes."""
    out = {k: _value(v) for k, v in obj.items().items() if k not in skip}
    for name in sorted(_lists(obj)):
        out[name] = [_items(c) for c in getattr(obj, name)]
    return out


def _lists(obj):
    return [k for k, d in hb.engine.schema()[obj.object_type]["lists"].items() if not d["single"]]


def test_load_a_gdf_file(tmp_path):
    m = hb.Model()
    cyl = add_mesh(m, "cyl", coarse_cylinder())
    cyl.write(tmp_path / "c.gdf")
    new = m.meshes.load("m", tmp_path / "c.gdf")
    # the file's gravity is kept (not used), an empty title is written as the default one; the rest is the same
    skip = ("file_gravity", "label", "source_file", "source_spl")  # what only the file gives
    assert _items(new, skip) == _items(cyl, skip) and new.file_gravity == 9.80665
    sp = add_mesh(m, "sp", spline_cylinder())
    sp.write(tmp_path / "s.gdf", spl=tmp_path / "s.spl")
    new = m.meshes.load("s", tmp_path / "s.gdf", higher_order=True, spl=tmp_path / "s.spl")
    assert new.source_spl == str((tmp_path / "s.spl").resolve())
    assert _items(new, skip) == _items(sp, skip)
    # the discretization from panel_size instead of the SPL file
    new = m.meshes.load("p", tmp_path / "s.gdf", higher_order=True, panel_size=0.4)
    assert new.discretization.shape == (len(sp.patches), 6) and np.all(new.discretization[:, :2] >= 1)
    assert _items(new, skip + ("discretization",)) == _items(sp, skip + ("discretization",))
    with pytest.raises(OSError, match=r"missing\.gdf"):
        m.meshes.load("x", tmp_path / "missing.gdf")
    assert "x" not in m.meshes  # nothing is left behind


@pytest.mark.parametrize("shape, dims, npatch", [
    ("circular_cylinder", {"radius": 1.0, "draft": 0.5}, 2),
    ("sphere", {"radius": 1.0}, 1),  # the hemisphere
    ("barge", {"half_length": 2.0, "half_beam": 1.0, "draft": 0.5}, 3),
    ("moonpool_cylinder", {"radius": 2.0, "draft": 1.0, "moonpool_radius": 0.5}, 3),
])  # fmt: skip
def test_analytic_shapes(shape, dims, npatch):
    m = hb.Model()
    mesh = m.meshes.analytic("m", shape, (4, 4, 3, 3, 3, 4), **dims)
    assert mesh.kind == "analytic_patches" and len(mesh.patches) == npatch and mesh.symmetry == "xz_and_yz"


def test_control_surfaces(tmp_path):
    # the automatic surface of a CSF file is the one generated for the body with the same radius,
    # depth and panel size: the same control-surface drift, bit for bit
    opts = {"motion": True, "drift_control_surface": True}
    cs = {"method": "automatic", "radius": 2.0, "depth": 1.5, "panel_size": 0.25}
    generated = model(lo_free_cylinder(name="hull", control_surface=cs), [1.0, 2.0], [0.0, 30.0], options=opts)
    given = model(lo_free_cylinder(name="hull"), [1.0, 2.0], [0.0, 30.0], options=opts)
    csf = tmp_path / "a.csf"
    csf.write_text("\n".join(["control surface", "1", "1 1", "0 0 0.25", "2.0 1.5", "0"]) + "\n", encoding="ascii")
    d = given.meshes.load_csf("d", csf, body="hull")
    assert d.kind == "analytic_patches" and len(d.patches) > 0
    given.bodies["hull"].update(**flat_surface("control_surface", {"method": "mesh", "mesh": "d"}))
    assert given.bodies["hull"].control_surface_mesh == "d"
    a, b = generated.solve(), given.solve()
    for q in ("drift_control_surface", "drift_control_surface_body_axes"):
        assert np.asarray(a.quantity(q)).tobytes() == np.asarray(b.quantity(q)).tobytes(), q


def test_trim_low_order():
    v = box_panels(1.0, 0.5, 1.0, n=(4, 2, 4), quarter=False) + [0.0, 0.0, 0.3]
    m = hb.Model()
    m.meshes.add("box", kind="panels", vertices=v)
    m.environment.update(omegas=[1.0], headings=[0.0])
    body = m.bodies.add("b", mesh="box", trim=[-0.1, 3.0, 2.0])
    new = body.trimmed_mesh("t")
    assert new.kind == "panels" and np.all(new.vertices[:, :, 2] <= 1e-12)  # the submerged part
    assert len(new.source_elements) == len(new.vertices) and set(new.source_elements) <= set(range(1, len(v) + 1))
    # the box rotated rigidly: the hydrostatics of the trimmed body are the geometry of its submerged
    # part (the same elements, the same quadrature) closed by its waterplane
    h, g = body.hydrostatics(), new.geometry()
    assert h.volumes[2] > 0.0 and h.volumes.tobytes() == g.volumes.tobytes() and h.volume == g.volume
    assert h.waterplane_area == g.waterplane_area
    # a tank cut at its own free surface: its elements end at z = -0.2 instead of z = 0
    body.update(mesh="box", ranges=[(9, len(v))])
    body.trim = [0.0, 2.0, 0.0]
    body.tanks.add(mesh="box", ranges=[(1, 8)], surface_level=-0.2)
    cut = body.trimmed_mesh("tc", "tank")
    assert set(cut.source_elements) <= set(range(1, 9)) and np.all(cut.vertices[:, :, 2] <= -0.2 + 1e-12)


def test_trim_patches_with_the_interior_surface():
    m = hb.Model()
    cyl = m.meshes.analytic("cyl", "circular_cylinder", (4, 4, 3, 3, 3, 4), radius=1.0, draft=0.5)
    before = _items(cyl)
    m.options.higher_order = True
    m.environment.update(omegas=[1.0], headings=[0.0])
    # raised by 0.3 (the open top above the water: a closed waterline)
    body = m.bodies.add("b", mesh="cyl", trim=[0.3, 3.0, 0.0], interior_surface="waterline")
    new = body.trimmed_mesh("t")
    assert new.kind == "trimmed_patches" and new.source == "t_source"
    # the source of the trim: the patches of the mesh, then those of the interior free surface
    src = _items(m.meshes["t_source"])
    k = len(before["patches"])
    assert src.pop("patches")[:k] == before.pop("patches")
    assert len(src.pop("discretization")) > len(before.pop("discretization"))
    assert src == before
    n = len(new.trim_elements)
    assert new.parameter_ranges.shape == (n, 4) and new.discretization.shape == (n, 6) and len(new.source_elements) == n
    assert set(new.source_elements) <= {1, 2}
    # the generated interior surface: patches of the source beyond those of the mesh
    interior = body.trimmed_mesh("ti", "interior_surface")
    assert len(interior.source_elements) and np.all(interior.source_elements > k)
    # a trimmed mesh that cannot be added leaves no copy of its source behind
    names = m.meshes.names()
    with pytest.raises(ValueError, match="duplicate mesh name 't'"):
        body.trimmed_mesh("t")
    assert m.meshes.names() == names


def _waterplane(mesh):
    """The ranges of the elements of a mesh in z = 0 with upward normals (None: no element)."""
    rec = mesh.elements()
    on = (rec.normals[:, 2] > 0.5) & (np.abs(rec.vertices[..., 2]).max(axis=1) <= 1e-12)
    return hb.ranges(on) if on.any() else None


def test_write_and_queries(tmp_path):
    m = hb.Model()
    mesh = add_mesh(m, "cyl", coarse_cylinder())
    mesh.write(tmp_path / "a.gdf")
    back = m.meshes.load("back", tmp_path / "a.gdf")
    back.write(tmp_path / "b.gdf")
    assert (tmp_path / "a.gdf").read_bytes() == (tmp_path / "b.gdf").read_bytes()
    assert _waterplane(mesh) is None  # an open cylinder
    lid = add_mesh(m, "lid", surface(box_panels(1.0, 0.5, 0.5, n=(2, 1, 1), lid=True)))
    assert _waterplane(lid) == [(len(lid.vertices) - 1, len(lid.vertices))]
    pm = add_mesh(m, "sp", spline_cylinder())
    uv = np.array([[0.0, 0.0], [0.5, -0.5]])
    pp = pm.patch_points(1, uv)
    p, n = pp.points, pp.normals
    # the radius 1 to rounding: u = 0 and 0.5 are Greville abscissae, where the spline interpolates the
    # circle (between them it is off by up to 7.4e-5)
    np.testing.assert_allclose(np.hypot(p[:, 0], p[:, 1]), 1.0, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, rtol=1e-14)
    pm.write(tmp_path / "s.gdf", spl=tmp_path / "s.spl")
    again = m.meshes.load("sp2", tmp_path / "s.gdf", higher_order=True, spl=tmp_path / "s.spl")
    assert again.patch_points(1, uv).points.tobytes() == p.tobytes()
    an = m.meshes.analytic("an", "circular_cylinder", (4, 4, 3, 3, 3, 4), radius=1.0, draft=0.5)
    with pytest.raises(ValueError, match="kept in the model file"):
        an.write(tmp_path / "x.gdf")


def test_a_trimmed_case_saves_its_source_elements(tmp_path):
    m = hb.Model()
    m.meshes.add("box", kind="panels", vertices=box_panels(1.0, 0.5, 1.0, n=(4, 2, 4)) + [0.0, 0.0, 0.3])
    t = m.bodies.add("b", mesh="box", trim=[0.0, 3.0, 0.0]).trimmed_mesh("t")
    m.bodies.add("c", mesh="t", position=[5.0, 0.0, 0.0])
    m.save(tmp_path / "case.hbm")
    back = hb.Model.load(tmp_path / "case.hbm")
    assert back.meshes["t"].source_elements.tobytes() == t.source_elements.tobytes()
    assert back.meshes["t"].vertices.tobytes() == t.vertices.tobytes()


def test_the_knots_of_each_spline_patch_have_their_length():
    # a wrong pair of knot lengths with the right sum is refused per patch, not sliced across
    mesh = spline_cylinder()
    p = mesh["patches"]
    p[0]["u_knots"], p[0]["v_knots"] = np.append(p[0]["u_knots"], 1.0), p[0]["v_knots"][:-1]
    m = hb.Model()
    add_mesh(m, "s", mesh)
    m.bodies.add("b", mesh="s")
    m.environment.omegas, m.environment.headings = [1.0], [0.0]
    m.options.higher_order = True
    with pytest.raises(ValueError, match="u_knots and v_knots need u_order"):
        m.check()


def test_a_mesh_read_from_a_file_keeps_its_absolute_path(tmp_path, monkeypatch):
    # the GDF (and SPL) a mesh was read from, as an absolute path, kept in the model file
    m = hb.Model()
    add_mesh(m, "cyl", surface(cylinder_panels(1.0, 0.5, quarter=True),
                                  symmetry="xz_and_yz")).write(tmp_path / "c.gdf")
    monkeypatch.chdir(tmp_path)
    loaded = m.meshes.load("read", "c.gdf")  # a relative path
    assert loaded.source_file == str((tmp_path / "c.gdf").resolve())
    assert m.meshes["cyl"].source_file == ""  # given as arrays
    m.save(tmp_path / "case.hbm")
    assert hb.Model.load(tmp_path / "case.hbm").meshes["read"].source_file == loaded.source_file
