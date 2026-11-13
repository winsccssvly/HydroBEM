"""The queries asked of the owner of what they need: a mesh answers its own geometry (volume,
centre of volume, area, bounds, waterplane, elements, tessellation, quality) in its own
coordinates; a body its hydrostatics (with the metacentric heights), panels, mesh quality and
tessellation; the water its panels. The lists of dipoles, dipole dampers, tanks and pressure surfaces take optional
names. The analytic shapes give their exact geometry (the one place of the closed-form volumes), the
polyhedral meshes the exact values of their polygons and of the one-point rule."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import part, surface
from meshes import box_panels, cylinder_panels, face_panels, flat_plate, moonpool_lid_panels, moonpool_panels
from problems import RHO, G, add_mesh, analytic, model
from regression_problems import PROBLEMS

DISC = (4, 4, 3, 3, 3, 4)


def _analytic(shape, **dims):
    return hb.Model().meshes.analytic("m", shape, DISC, **dims)


# ------------------------------------------------------------- the geometry of a mesh


@pytest.mark.parametrize("shape, dims, volume, zc, area, wp, ixx, iyy, lo, hi", [
    # a cylinder of radius 1 and draft 0.5: the side 2 pi r T and the bottom pi r^2
    ("circular_cylinder", {"radius": 1.0, "draft": 0.5}, math.pi / 2, -0.25, 2 * math.pi, math.pi, math.pi / 4,
     math.pi / 4, (-1, -1, -0.5), (1, 1, 0)),
    # a hemisphere of radius 1: z_c = -3/8
    ("sphere", {"radius": 1.0}, 2 * math.pi / 3, -3.0 / 8, 2 * math.pi, math.pi, math.pi / 4, math.pi / 4,
     (-1, -1, -1), (1, 1, 0)),
    # a barge 4 x 2 x 0.5: the bottom 8 and the sides 4 + 2; int x^2 = 32/3, int y^2 = 8/3
    ("barge", {"half_length": 2.0, "half_beam": 1.0, "draft": 0.5}, 4.0, -0.25, 14.0, 8.0, 32.0 / 3, 8.0 / 3,
     (-2, -1, -0.5), (2, 1, 0)),
])  # fmt: skip
def test_geometry_of_the_analytic_shapes_is_exact(shape, dims, volume, zc, area, wp, ixx, iyy, lo, hi):
    g = _analytic(shape, **dims).geometry()
    assert isinstance(g, hb.MeshGeometry)
    np.testing.assert_allclose(g.volumes, volume, rtol=1e-12)
    assert g.volume == pytest.approx(volume, rel=1e-12)
    np.testing.assert_allclose(g.center_of_volume, [0.0, 0.0, zc], atol=1e-12)
    assert g.area == pytest.approx(area, rel=1e-12)
    assert g.waterplane_area == pytest.approx(wp, rel=1e-12)
    np.testing.assert_allclose(g.waterplane_centroid, 0.0, atol=1e-12)
    np.testing.assert_allclose(g.waterplane_moments, [ixx, iyy, 0.0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g.bounds, [lo, hi], atol=1e-12)
    with pytest.raises(ValueError, match="read-only"):
        g.volumes[0] = 1.0


@pytest.mark.parametrize("shape, dims, volume", [
    ("circular_cylinder", {"radius": 1.0, "draft": 0.5, "mapping": "nonuniform"}, math.pi / 2),
    ("elliptic_cylinder", {"a": 2.0, "b": 1.0, "draft": 0.5}, math.pi),  # pi a b T
    ("sphere", {"radius": 1.0, "mapping": "nonuniform"}, 2 * math.pi / 3),
    ("ellipsoid", {"a": 2.0, "b": 0.5, "c": 0.25}, math.pi / 6),  # the half ellipsoid 2 pi a b c / 3
    ("moonpool_cylinder", {"radius": 2.0, "draft": 1.0, "moonpool_radius": 0.5}, math.pi * (2.0**2 - 0.5**2)),
    ("moonpool_cylinder", {"radius": 6.0, "draft": 8.0, "moonpool_radius": 2.5, "mapping": "nonuniform"},
     math.pi * (36 - 6.25) * 8),
])  # fmt: skip
def test_the_volumes_of_the_other_analytic_shapes_are_exact(shape, dims, volume):
    # the closed form to rounding (5e-13 for the nonuniform sphere); the hydrostatics of a body on the
    # mesh have the same volumes, bit for bit
    m = hb.Model()
    g = m.meshes.analytic("m", shape, DISC, **dims).geometry()
    np.testing.assert_allclose(g.volumes, volume, rtol=1e-12)
    m.bodies.add("b", mesh="m")
    m.options.higher_order, m.first_order.radiation, m.first_order.diffraction = True, False, False
    assert m.bodies["b"].hydrostatics().volumes.tobytes() == g.volumes.tobytes()


def test_geometry_of_polyhedral_meshes():
    # a box of panels is exact; a 32-gon cylinder to its discretization (the polygon's own values)
    box = add_mesh(hb.Model(), "box", surface(box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True),
                                                 symmetry="xz_and_yz"))
    g = box.geometry()
    np.testing.assert_allclose(g.volumes, 1.0, rtol=1e-14)
    np.testing.assert_allclose(g.center_of_volume, [0.0, 0.0, -0.25], atol=1e-15)
    assert (g.area, g.waterplane_area) == pytest.approx((5.0, 2.0), rel=1e-14)
    # the moments of the one-point rule of the low order: the bottom panels at their centroids
    rec = box.elements()
    bottom = rec.normals[:, 2] < -0.5
    c, a = rec.centroids[bottom], rec.areas[bottom]
    np.testing.assert_allclose(g.waterplane_moments, [4 * (a * c[:, 0] ** 2).sum(), 4 * (a * c[:, 1] ** 2).sum(), 0.0],
                               rtol=1e-14, atol=1e-15)  # fmt: skip
    np.testing.assert_allclose(g.bounds, [[-1.0, -0.5, -0.5], [1.0, 0.5, 0.0]])
    cyl = add_mesh(hb.Model(), "c", surface(cylinder_panels(1.0, 0.5, n_phi=8, n_z=2, n_r=2), symmetry="xz_and_yz"))
    g = cyl.geometry()
    polygon = 0.5 * 32 * math.sin(2 * math.pi / 32)  # the area of the 32-gon
    np.testing.assert_allclose(g.volumes, 0.5 * polygon, rtol=1e-14)
    assert g.waterplane_area == pytest.approx(polygon, rel=1e-14)
    assert g.center_of_volume[2] == pytest.approx(-0.25, rel=1e-14)
    # the one-point rule on the bottom panels (two rings of the 32-gon; pi R^4 / 4 to 6 %), as for the box
    rec = cyl.elements()
    bottom = rec.normals[:, 2] < -0.5
    c, a = rec.centroids[bottom], rec.areas[bottom]
    np.testing.assert_allclose(g.waterplane_moments, [4 * (a * c[:, 0] ** 2).sum(), 4 * (a * c[:, 1] ** 2).sum(), 0.0],
                               rtol=1e-14, atol=1e-15)  # fmt: skip


def test_the_geometry_volumes_are_the_quality_volumes():
    # one computation (the divergence theorem over the elements, x, y and z) serves both queries: equal bytes
    for mesh in (_analytic("circular_cylinder", radius=1.0, draft=0.5),
                 add_mesh(hb.Model(), "box", surface(box_panels(quarter=True), symmetry="xz_and_yz"))):  # fmt: skip
        g, q = mesh.geometry(), mesh.quality()
        assert q.volumes.shape == (1, 3) and q.volumes[0].tobytes() == g.volumes.tobytes()


def test_elements_tessellation_and_quality_of_a_mesh():
    v = box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True, lid=True)
    mesh = add_mesh(hb.Model(), "box", surface(v, symmetry="xz_and_yz"))
    rec = mesh.elements()
    n = len(v)
    assert isinstance(rec, hb.MeshElements) and rec.element.tolist() == list(range(1, n + 1))  # not expanded
    assert rec.vertices.shape == (n, 4, 3) and np.array_equal(rec.vertices, v)  # mesh coordinates
    np.testing.assert_allclose(np.linalg.norm(rec.normals, axis=1), 1.0)
    np.testing.assert_allclose(rec.centroids, v.mean(axis=1), atol=1e-15)
    assert 4 * rec.areas.sum() == pytest.approx(mesh.geometry().area, rel=1e-14)  # the stored quarter
    t = mesh.tessellation()
    assert len(t.element) == 4 * n and np.all(t.body == -1)
    assert len(mesh.tessellation(include_reflections=False).element) == n
    q = mesh.quality()
    assert np.all(q.body == -1) and np.all(np.isinf(q.per_wavelength)) and len(q.element) == n
    # the hull and its waterplane lid: a closed surface, every edge used twice
    assert (q.edges.free[0], q.edges.non_manifold[0], q.edges.inconsistent[0]) == (0, 0, 0)
    # a panel turned over: its edges disagree with those of its neighbours
    w = v.copy()
    w[0] = w[0][::-1]
    q = add_mesh(hb.Model(), "w", surface(w, symmetry="xz_and_yz")).quality()
    assert q.edges.inconsistent[0] > 0 and "orientation" in q.flags[0]
    # a patch mesh: its elements at the centre of the patch, its patch sides checked
    pq = _analytic("circular_cylinder", radius=1.0, draft=0.5)
    assert len(pq.elements().element) == 2 and pq.quality().edges.free[0] == 0
    assert len(pq.tessellation(divisions=(3, 2), include_reflections=False).element) == 2 * 6


def test_a_mesh_has_no_hydrostatics_and_no_trim():
    mesh = add_mesh(hb.Model(), "box", surface(box_panels(quarter=True), symmetry="xz_and_yz"))
    for gone in ("hydrostatics", "waterplane_panels", "trimmed", "select"):
        assert not hasattr(mesh, gone)


# ------------------------------------------------------------- the queries of a body


def test_body_hydrostatics_are_those_of_its_results_bit_for_bit():
    # the hydrostatics of each body are the hydrostatics quantities of a solve of the model
    m = PROBLEMS["lo_two_bodies"]()
    r = m.solve()
    for k, b in enumerate(m.bodies):
        h = b.hydrostatics()
        assert h.volumes.tobytes() == r.hydrostatics.volumes[k].tobytes() and h.volume == r.hydrostatics.volume[k]
        assert h.displaced_mass == r.hydrostatics.displaced_mass[k]
        assert h.waterplane_area == r.hydrostatics.waterplane_area[k]
        assert h.waterplane_centroid.tobytes() == r.hydrostatics.waterplane_centroid[k].tobytes()
        assert h.stiffness.tobytes() == r.hydrostatics.stiffness[k].tobytes()
        assert h.stiffness_rho_g.tobytes() == r.hydrostatics.stiffness_rho_g[k].tobytes()
        assert h.center_of_buoyancy.tobytes() == r.hydrostatics.center_of_buoyancy[k].tobytes()


def test_metacentric_heights_with_the_weight_terms():
    # the analytic barge 4 x 2 x 0.5 with G at z = -0.1: GM = I / V + z_B - z_G
    barge = analytic("barge", DISC, half_length=2.0, half_beam=1.0, draft=0.5)
    m = model({"mesh": barge, "mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.1]}, [1.0], [0.0],
              options={"higher_order": True})  # fmt: skip
    h = m.bodies[0].hydrostatics()
    assert h.has_weight and h.displaced_mass == pytest.approx(RHO * 4.0, rel=1e-12) and h.mass == h.displaced_mass
    np.testing.assert_allclose(h.metacentric_height, [8.0 / 3 / 4 - 0.25 + 0.1, 32.0 / 3 / 4 - 0.25 + 0.1], rtol=1e-12)
    np.testing.assert_allclose(h.metacentric_height, np.diag(h.stiffness)[3:5] / (RHO * G * h.volumes[0]), rtol=1e-15)
    m.bodies[0].center_of_gravity = None
    h = m.bodies[0].hydrostatics()
    assert not h.has_weight and h.metacentric_height is None


def test_panels_quality_and_tessellation_of_a_body_and_of_the_lids():
    hull = box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True)
    lid = surface(flat_plate(center=(6, 0)))
    m = model([{"name": "a", "mesh": surface(hull, symmetry="xz_and_yz")},
               {"name": "b", "mesh": surface(hull, symmetry="xz_and_yz"), "position": [3.0, 0.0, 0.0]}],
              [1.0], (), options={"diffraction": False}, lids=[{"name": "far", "mesh": lid}])  # fmt: skip
    a, b = m.bodies["a"].panels(), m.bodies["b"].panels()
    w = m.panels(role="lid")
    assert np.all(a.body == 0) and np.all(b.body == 1) and np.all(w.body == -1) and set(w.role) == {"lid"}
    every = m.panels()
    assert np.array_equal(every.centroids, np.concatenate([a.centroids, b.centroids, w.centroids]))
    np.testing.assert_allclose(b.centroids, a.centroids + [3.0, 0.0, 0.0])  # placed
    q = m.bodies["b"].mesh_quality()
    assert np.all(q.body == 1) and np.array_equal(m.mesh_quality().body, np.r_[np.zeros(len(hull)), q.body])
    t, tb = m.tessellation(divisions=(2, 2)), m.bodies["b"].tessellation(divisions=(2, 2))
    assert len(t.element) == 2 * len(tb.element) + 4 and np.array_equal(t.body, np.r_[0 * tb.body, tb.body, [-1] * 4])
    with pytest.raises(ValueError, match="role must be one of"):
        m.panels("a")  # the model's panels cover everything (its first argument the role)
    # the roles are texts, also of no elements
    assert a.role.dtype.kind == "U" and m.panels(role="control_surface").role.dtype.kind == "U"


def test_a_lid_placed_with_a_body_is_an_element_of_the_body():
    """A lid placed with a body is an element of the body (role "lid", by its stored element), not an
    element of no body; a role a scope never has is refused."""
    hull, lid = moonpool_panels(n=8), moonpool_lid_panels(n=8)
    both = surface(np.concatenate([hull, lid]), symmetry="xz_and_yz")
    nh, nl = len(hull), len(lid)
    m = model({"name": "b", "mesh": part(both, 1, nh)}, [1.0], [0.0],
              lids=[{"name": "moon", "mesh": part(both, nh + 1, nh + nl), "placed_with": "b"}])  # fmt: skip
    b = m.bodies["b"]
    p = b.panels(role="lid", include_reflections=False)
    assert set(p.role) == {"lid"} and p.element.tolist() == list(range(nh + 1, nh + nl + 1))
    assert np.all(p.owner == 0) and np.all(p.body == 0)
    assert np.all(m.panels(role="lid").body == 0) and len(b.panels(include_reflections=False).element) == nh + nl
    with pytest.raises(ValueError, match="role must be one of .*, got 'fixed_damper' .*not of a body"):
        b.panels(role="fixed_damper")
    assert len(m.panels(role="fixed_damper").element) == 0  # the model has them (none here)


def test_the_queries_of_a_body_need_only_their_rules():
    """The queries of a body check the environment, the geometry and the surfaces, not the rules of
    the outputs and the field points; the queries of the model check every rule."""
    m = model({"name": "b", "mesh": surface(box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True),
                                            symmetry="xz_and_yz")},
              [1.0], [0.0])  # fmt: skip
    h = m.bodies["b"].hydrostatics()
    m.first_order.field_points = [[5.0, 0.0, -1.0]]  # field points without a field output
    with pytest.raises(ValueError, match="no field output"):
        m.counts()
    b = m.bodies["b"]
    assert b.hydrostatics().volumes.tobytes() == h.volumes.tobytes()
    assert len(b.panels().element) and len(b.mesh_quality().element) and len(b.tessellation().element)
    with pytest.raises(ValueError, match="no field output"):
        m.panels()
    # the rules of the geometry still apply
    b.dipoles.add(mesh=b.mesh, ranges=[(1, 1)])
    with pytest.raises(ValueError, match="taken by both"):
        b.hydrostatics()


# ------------------------------------------------------------- optional names


def _spar():
    hull = cylinder_panels(1.0, 1.0, n_phi=4, n_z=2, n_r=2, quarter=True)
    plate = face_panels((0.0, 0.0, -1.5), (1.2, 0, 0), (0, 1.2, 0), 2, 2, (0, 0, 1))
    mesh = surface(np.concatenate([hull, plate]), symmetry="xz_and_yz")
    return mesh, len(hull), len(plate)


def test_optional_names_add_find_and_unique():
    mesh, nh, npl = _spar()
    m = model({"name": "spar", "mesh": part(mesh, 1, nh)}, [1.0], [0.0])
    spar = m.bodies["spar"]
    d = spar.dipoles.add("plate", mesh=spar.mesh, ranges=[(nh + 1, nh + 2)])
    e = spar.dipoles.add(mesh=spar.mesh, ranges=[(nh + 3, nh + npl)])  # without a name, as before
    assert spar.dipoles["plate"] == d == spar.dipoles[0] and spar.dipoles[1] == e
    assert spar.dipoles.names() == ["plate", ""] and "plate" in spar.dipoles and "" not in spar.dipoles
    assert (d.name, e.name) == ("plate", "")
    # the path names an object by its name, one without by its position
    assert d.path == "bodies/spar/dipoles/plate" and e.path == "bodies/spar/dipoles/2"
    with pytest.raises(KeyError):
        spar.dipoles["fin"]
    with pytest.raises(ValueError, match="bodies/spar: duplicate dipole name 'plate'"):
        spar.dipoles.add("plate", mesh=spar.mesh, ranges=[(nh + 3, nh + 3)])
    with pytest.raises(ValueError, match="'2' is a position"):
        spar.dipoles.add("2", mesh=spar.mesh, ranges=[(nh + 3, nh + 3)])
    assert len(spar.dipoles) == 2
    # renamed, or the name removed
    spar.dipoles.rename("plate", "heave_plate")
    assert spar.dipoles["heave_plate"] == d
    spar.dipoles.rename(1, "fin")
    assert spar.dipoles.names() == ["heave_plate", "fin"]
    with pytest.raises(ValueError, match="duplicate dipole name 'fin'"):
        spar.dipoles.rename(0, "fin")
    spar.dipoles.rename("fin", "")
    assert spar.dipoles.names() == ["heave_plate", ""]
    # added again, a name or none
    spar.dipoles.clear()
    spar.dipoles.add("a", mesh=spar.mesh, ranges=[(nh + 1, nh + 2)])
    spar.dipoles.add(mesh=spar.mesh, ranges=[(nh + 3, nh + 4)])
    assert spar.dipoles.names() == ["a", ""]
    m.check()
    # every list with optional names; a list without names refuses one
    spar.tanks.add("aft", mesh=spar.mesh, ranges=[(1, 1)])
    spar.pressure_surfaces.add("cushion", mesh=spar.mesh, ranges=[(2, 2)])
    spar.dipole_dampers.add("ring", mesh=spar.mesh, ranges=[(3, 3)])
    m.fixed_dampers.add("far", mesh=spar.mesh, ranges=[(4, 4)])
    assert spar.tanks["aft"].path == "bodies/spar/tanks/aft" and m.fixed_dampers["far"].path == \
        "fixed_dampers/far"  # fmt: skip
    assert spar.pressure_surfaces["cushion"].pressure_modes.names() == []
    g = spar.generalized_modes.add("bend")
    with pytest.raises(ValueError, match="have no names"):
        g.pieces.insert(0, "p")


def test_optional_names_in_the_error_locations():
    mesh, nh, npl = _spar()
    m = model({"name": "spar", "mesh": part(mesh, 1, nh)}, [1.0], [0.0])
    spar = m.bodies["spar"]
    name = spar.mesh
    spar.dipoles.add("plate", mesh=name, ranges=[(nh, nh + npl)])  # its first element is the hull's
    with pytest.raises(ValueError, match=f"element {nh} of the mesh '{name}' is taken by both the wetted surface "
                                         r"\(mesh\) and dipoles 'plate'"):  # fmt: skip
        m.check()
    spar.dipoles.clear()
    spar.dipoles.add(mesh=name, ranges=[(nh, nh + npl)])  # without a name: its position, as before
    with pytest.raises(ValueError, match="and dipoles 1 "):
        m.check()
    spar.dipoles.clear()
    spar.tanks.add("aft", mesh=name, ranges=[(nh + 1, nh + npl + 1)])
    with pytest.raises(ValueError, match=r"bodies/spar: tanks 'aft': the element range .* is beyond the mesh"):
        m.check()
    # a value of an item: the path of the object, by its name
    with pytest.raises(ValueError, match="bodies/spar/tanks/aft: density_ratio"):
        spar.tanks["aft"].density_ratio = -1.0


def test_optional_names_survive_save_and_load(tmp_path):
    mesh, nh, npl = _spar()
    m = model({"name": "spar", "mesh": part(mesh, 1, nh)}, [1.0], [0.0])
    spar = m.bodies["spar"]
    spar.dipoles.add("plate", mesh=spar.mesh, ranges=[(nh + 1, nh + 2)])
    spar.dipoles.add(mesh=spar.mesh, ranges=[(nh + 3, nh + npl)])
    m.fixed_dampers.add("far", mesh=spar.mesh, ranges=[(nh + 3, nh + 3)])
    m.save(tmp_path / "case.hbm")
    back = hb.Model.load(tmp_path / "case.hbm")
    assert back.bodies["spar"].dipoles.names() == ["plate", ""] and back.fixed_dampers.names() == ["far"]
    assert str(back) == str(m)


def _rows(p, keep):
    """The fields of hb.Panels at the rows keep, as arrays."""
    return {k: np.asarray(getattr(p, k))[keep] for k in ("vertices", "body", "role", "owner", "element", "centroids")}


def test_the_queries_of_a_body_check_that_body_alone():
    # the queries of a body apply the rules of that body (the environment, its geometry, its
    # surfaces), not those of the frequencies, the orders, the options or the other bodies; the
    # model's check applies every rule
    cyl = surface(cylinder_panels(1.0, 0.5, n_phi=4, n_z=2, n_r=2, quarter=True), symmetry="xz_and_yz")
    m = model([{"name": "a", "mesh": cyl, "position": [-3.0, 0.0, 0.0]},
               {"name": "b", "mesh": cyl, "position": [3.0, 0.0, 0.0]}], [1.0], [0.0])  # fmt: skip
    a, b = m.bodies["a"], m.bodies["b"]
    m.check()
    h, pa, pb, q, t = a.hydrostatics(), a.panels(), b.panels(), a.mesh_quality(), a.tessellation()
    # the rows of a body's query are those of the body in the model's query
    whole = m.panels()
    for p, k in ((pa, 0), (pb, 1)):
        mine, there = _rows(p, slice(None)), _rows(whole, np.asarray(whole.body) == k)
        for key in mine:
            np.testing.assert_array_equal(mine[key], there[key], err_msg=key)

    def same():
        assert a.hydrostatics().volumes.tobytes() == h.volumes.tobytes()
        assert np.asarray(a.panels().centroids).tobytes() == np.asarray(pa.centroids).tobytes()
        assert np.asarray(b.panels().body).tolist() == np.asarray(pb.body).tolist()
        assert np.asarray(a.mesh_quality().size).tobytes() == np.asarray(q.size).tobytes()
        assert np.asarray(a.tessellation().vertices).tobytes() == np.asarray(t.vertices).tobytes()

    # no frequencies with radiation and diffraction on
    m.environment.omegas = []
    with pytest.raises(ValueError, match="omegas is empty"):
        m.check()
    same()
    # the motions with a body without a centre of gravity, either body
    m.environment.omegas = [1.0]
    a.update(mass="displaced", center_of_gravity=[0.0, 0.0, -0.1], radii_of_gyration=[0.5, 0.5, 0.5])
    m.first_order.motion = True
    with pytest.raises(ValueError, match="bodies/b: .*requires center_of_gravity"):
        m.check()
    same()
    b.hydrostatics()
    # another body without a mesh: refused by the model and by its own queries, not by those of a
    m.first_order.motion = False
    m.bodies.add("c")
    for query in (m.check, m.bodies["c"].hydrostatics, m.bodies["c"].panels):
        with pytest.raises(ValueError, match="bodies/c: the body has no wetted surface"):
            query()
    same()
    del m.bodies["c"]
    # the body's own invalid input is refused, located at the body
    a.ranges = [(1, 10_000)]
    for query in (a.hydrostatics, a.panels, a.mesh_quality, a.tessellation):
        with pytest.raises(ValueError, match="beyond the mesh") as e:
            query()
        assert e.value.location.body == 0
    b.hydrostatics()


def test_the_rows_of_a_body_query_are_numbered_as_in_the_model():
    # two moonpool hulls, each with a lid of elements of its mesh placed with it: the lid of the
    # second body is lid 1 (owner) in its own query as in the model's, its body 1
    hull, lid = moonpool_panels(n=4), moonpool_lid_panels(n=4)
    both = surface(np.concatenate([hull, lid]), symmetry="xz_and_yz")
    bodies = [{"name": n, "mesh": part(both, 1, len(hull)), "position": [x, 0.0, 0.0]}
              for n, x in (("a", -4.0), ("b", 4.0))]  # fmt: skip
    lids = [{"name": f"moon_{n}", "mesh": part(both, len(hull) + 1, len(hull) + len(lid)), "placed_with": n,
             "damping_factor": 0.1} for n in ("a", "b")]  # fmt: skip
    m = model(bodies, [1.0], [0.0], lids=lids, options={"diffraction_formulation": "total"})
    m.check()
    whole = m.panels()
    for k, name in enumerate(("a", "b")):
        mine = m.bodies[name].panels()
        there = _rows(whole, np.asarray(whole.body) == k)
        for key, v in _rows(mine, slice(None)).items():
            np.testing.assert_array_equal(v, there[key], err_msg=key)
        lid_rows = np.asarray(mine.role) == "lid"
        assert lid_rows.any() and set(np.asarray(mine.owner)[lid_rows]) == {k}
        t = m.bodies[name].tessellation()
        assert set(np.asarray(t.body)) == {k} and set(np.asarray(t.owner)[np.asarray(t.role) == "lid"]) == {k}
    # a disabled body is numbered as if it alone were enabled among the disabled
    m.bodies["a"].enabled = False
    assert set(np.asarray(m.bodies["b"].panels().body)) == {0}
    assert set(np.asarray(m.bodies["a"].panels().body)) == {0}
