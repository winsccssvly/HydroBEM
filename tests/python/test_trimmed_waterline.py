"""Trimmed waterline (the item trim of a body, and its trimmed surfaces body.trimmed_mesh; reference/inputs-bodies §1.2,
reference/inputs-meshes §4): low order and higher order.

A box built above the free surface and lowered by a vertical shift that lands
on a row of panel edges is the box of the smaller draft, element for element; a
pitched or rolled box keeps the volume 2a 2b T of the upright box (the
waterplane turns about its centre line), with the three volume components
equal; the symmetry flags that the angles break are removed and the
trimmed elements name their source elements. Higher order: the remapped cylinder has the exact volume
and waterplane area of the pitched cylinder, the map of the patches (patch_points) gives the
engine's volume, and the generated interior surface removes the irregular frequency; it covers the
waterplane exactly once, also around a moonpool and for two hulls in one mesh. A tank cut at
its own free surface adds the closed form of its fluid. A trimmed body equals its trimmed surfaces
given as one mesh with their elements, and gives the trimmed surface of each role. The remapped flat
barge against the trimmed low-order barge of many panels is validation, run when the method
changes, not a test. Meshes are procedural (tests/python/meshes.py)."""


import numpy as np
import pytest

from bodies import flat, part, surface
from meshes import MOONPOOL_FACES, box_panels, cylinder_panels, face_panels
from problems import analytic, flat_surface, hydrostatics, model, peak, solve, view

A, B, T, H = 2.0, 1.0, 1.0, 1.0  # half length, half beam, draft, height above the waterline


def tall_box(quarter=True):
    """The box -T <= z <= H, divided at z = 0 and z = -0.5."""
    q = box_panels(A, B, T + H, n=(4, 2, 6), quarter=quarter)
    q = q + [0.0, 0.0, H]
    return surface(q, symmetry="xz_and_yz" if quarter else "none")


def trim(mesh, elevation=0.0, pitch=0.0, roll=0.0, *, interior_surface=False, role="wetted"):
    """The trimmed surface of a role of a body of the mesh (a dict of mesh items, or a callable of
    problems.model: an analytic shape) with that trim, in a model of its own (higher order for a
    patch mesh; the generated interior surface of the higher order with interior_surface)."""
    ho = callable(mesh) or mesh.get("kind", "panels") != "panels"
    body = {"mesh": mesh, "trim": [elevation, pitch, roll]}
    if interior_surface:
        body["interior_surface"] = "waterline"
    return model(body, [1.0], [0.0], options={"higher_order": ho}).bodies["body"].trimmed_mesh("t", role)


def same(r1, r2, names=("added_mass", "radiation_damping", "excitation_diffraction")):
    for q in names:
        assert view(r1, q).tobytes() == view(r2, q).tobytes(), q


def test_vertical_shift_is_the_smaller_draft():
    # an identity of the input: the cut on a row of edges keeps the panels below it, element for
    # element, at vertices moved by the shift (rounding: 3e-16), so the results of the box of the
    # smaller draft to 1e-6: the results are stored in float32, and vertices differing in the last
    # bit may round to the neighbouring float32 (2e-16 on one processor, 1e-7 on another)
    tm = trim(tall_box(), elevation=-2 * H / 3)  # z = 0 on the row of edges at zeta = 2/3
    ref = surface(box_panels(A, B, T + 2 * H / 3, n=(4, 2, 5), quarter=True), symmetry="xz_and_yz")
    assert tm.symmetry == "xz_and_yz" and len(tm.vertices) == len(ref["vertices"])
    np.testing.assert_allclose(tm.vertices, ref["vertices"], rtol=0, atol=1e-12)
    w = np.array([0.8, 1.6])
    r, r0 = solve({"mesh": tall_box(), "trim": [-2 * H / 3, 0.0, 0.0]}, w), solve({"mesh": ref}, w)
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        np.testing.assert_allclose(view(r, q), view(r0, q), rtol=0, atol=1e-6 * peak(view(r0, q)))


@pytest.mark.parametrize("pitch, roll", [(5.0, 0.0), (0.0, 8.0), (4.0, 6.0)])
def test_pitched_and_rolled_box_keeps_its_volume(pitch, roll):
    # the closed form 2a 2b T of the submerged box (the waterplane through the origin), exact for
    # the plane panels but for rounding: 1e-9
    tm = trim(tall_box(), pitch=pitch, roll=roll)
    assert tm.symmetry == {(1, 1): "xz_and_yz", (1, 0): "yz", (0, 1): "xz", (0, 0): "none"}[(pitch == 0.0, roll == 0.0)]
    assert np.all(tm.vertices[..., 2] <= 1e-12)
    m = model({"mesh": tall_box(), "trim": [0.0, pitch, roll]}, [1.0], options={"diffraction": False})
    vol = np.asarray(hydrostatics(m)[0].volumes)
    # the submerged part of the box xi-axis section: zeta <= xi tan(pitch) / cos(roll) ...
    # the waterplane passes through the origin, so the volume is that of the upright box
    np.testing.assert_allclose(vol, 4 * A * B * T, rtol=1e-9)


def test_trimmed_records_name_their_source():
    hull = box_panels(A, B, T + H, n=(4, 2, 6), quarter=True) + [0.0, 0.0, H]
    fin = np.array(face_panels((A, 0.2, -0.9), (0.6, 0, 0), (0, 0, 0.6), 2, 2, (0, -1, 0)))  # a sheet
    mesh = surface(np.concatenate([hull, fin]), symmetry="xz_and_yz")
    n = len(hull)
    body = {"mesh": part(mesh, 1, n), "trim": [0.0, 0.0, 5.0], "dipoles": [{"mesh": part(mesh, n + 1, n + len(fin))}]}
    sheet = model(body, [1.0], [0.0]).bodies["body"].trimmed_mesh("t", "dipole")
    assert len(sheet.vertices) == 2 * len(fin)  # the fin and its mirror image in y (the roll breaks it)
    assert np.all(sheet.source_elements > n)


def test_trim_needs_a_submerged_part():
    with pytest.raises(ValueError, match="no submerged"):
        trim(tall_box(), elevation=10.0)
    m = model({"mesh": tall_box(), "trim": [10.0, 0.0, 0.0]}, [1.0], options={"diffraction": False})
    with pytest.raises(ValueError, match="bodies/body: trim: .*no submerged"):
        m.route()


def ho_cylinder(nu=8, nv=8):
    return analytic("circular_cylinder", [(nu, nv, 3, 3, 3, 4), (nu, nv // 2, 3, 3, 3, 4)], radius=1.0, draft=2.0)


def test_higher_order_pitched_cylinder_is_exact():
    """R = 1 built to a draft of 2, raised by 1 and pitched by 15 deg: the volume pi (2 - 1/cos 15)
    and the waterplane, an ellipse of area pi / cos 15, to the precision of the quadrature."""
    tm = trim(ho_cylinder(), 1.0, 15.0, 0.0)
    assert tm.symmetry == "xz" and [r[3] for r in tm.trim_elements] == [2, 0, 2, 0]
    m = model({"mesh": ho_cylinder(), "trim": [1.0, 15.0, 0.0]}, [1.0], [0.0], options={"higher_order": True})
    h = hydrostatics(m)[0]
    c = np.cos(np.radians(15.0))
    np.testing.assert_allclose(h.volumes, np.pi * (2 - 1 / c), rtol=1e-12)
    np.testing.assert_allclose(h.waterplane_area, np.pi / c, rtol=1e-12)
    # the map of the patches (patch_points) gives the engine's volume (the same quadrature of
    # the volume, z n_z over the elements)
    g, w = np.polynomial.legendre.leggauss(16)
    U, V = np.meshgrid(g, g, indexing="ij")
    e = 1e-6

    def point(rec, u, v):
        return tm.patch_points(rec, np.c_[u.reshape(-1), v.reshape(-1)]).points.reshape(*u.shape, 3)

    vol = 0.0
    for rec in range(1, len(tm.trim_elements) + 1):
        x = point(rec, U, V)
        xu = (point(rec, U + e, V) - point(rec, U - e, V)) / (2 * e)
        xv = (point(rec, U, V + e) - point(rec, U, V - e)) / (2 * e)
        n = -np.cross(xu, xv)  # out of the body (x_u x x_v points into it)
        vol += np.sum(np.outer(w, w) * x[..., 2] * n[..., 2]) * (2.0 if tm.symmetry in ("xz", "xz_and_yz") else 1.0)
    np.testing.assert_allclose(vol, np.pi * (2 - 1 / c), rtol=1e-6)


def test_generated_interior_surface_removes_the_irregular_frequency():
    """Across the first surge irregular frequency of the trimmed cylinder (KR near 3.75 with
    the draft about 1) the added mass without the interior surface has a jump; with the
    generated interior surface it is smooth (the continuous solution has none): over steps of
    0.05 in KR its second differences are those of the smooth curve, below 0.1 of the jump (0.03
    against 3.2 measured; the jump above 1, the plain solve's spike)."""
    def a11(interior, w):
        isurf = {"interior_surface": {"method": "waterline"}} if interior else {}
        body = {"mesh": ho_cylinder(4, 4), "trim": [1.0, 15.0, 0.0], "modes": ["surge"], **isurf}
        m = model(body, w, (), options={"higher_order": True, "diffraction": False})
        return np.asarray(m.solve().added_mass)[:, 0, 0]

    w = np.sqrt(9.80665 * np.linspace(3.65, 3.85, 5))
    jump = np.abs(np.diff(a11(False, w), 2)).max()
    smooth = np.abs(np.diff(a11(True, w), 2)).max()
    assert jump > 1.0 and smooth < 0.1 * jump
    inside = trim(ho_cylinder(), 1.0, 15.0, 0.0, interior_surface=True, role="interior_surface")
    assert len(inside.trim_elements) and all(r[3] == 0 for r in inside.trim_elements)


def test_higher_order_trim_rejections():
    with pytest.raises(ValueError, match="bodies/body: trim: .*no submerged"):
        trim(ho_cylinder(), 10.0)
    # a trimmed mesh has elements (without them it would be its source, untrimmed)
    m = model((), options={"higher_order": True})
    m.meshes.analytic("box", "circular_cylinder", [(4, 4, 3, 3, 3, 4), (4, 2, 3, 3, 3, 4)], radius=1.0, draft=2.0)
    with pytest.raises(ValueError, match="meshes/t: trim_elements: a trimmed mesh has at least one element"):
        m.meshes.add("t", kind="trimmed_patches", source="box", trim=[1.0, 0.0, 0.0],
                     trim_elements=np.zeros((0, 4), np.int64), parameter_ranges=np.zeros((0, 4)),
                     discretization=np.zeros((0, 6), np.int64)).geometry()  # fmt: skip


def test_trimmed_tank_with_its_free_surface_is_closed():
    """A tank of a trimmed low-order body cut at its own level free surface (its surface_level):
    the heaving tank's potential phi = z - Z_T + 1/K holds for any tank, so the tank adds
    rho (V_T - g A_T / omega^2) to A33 and nothing to B33; V_T and A_T of the tank pitched by
    3 deg below z = -0.3. The hull has a freeboard of 0.5 m, so the trim cuts its sides (a hull
    without one put a part of its open deck under water). The closed form is exact for the
    continuous tank; 3e-3 for A and 2e-3 for B (of the added mass): the constant-panel solution of
    the tank's potential on 12 x 6 x 6 panels a quadrant (1.8e-3 and 9.8e-4 measured, 6.7e-4 and
    3.7e-4 on twice the panels; the same without the trim)."""
    import math as _m

    from meshes import box_panels

    hull = box_panels(2.0, 1.0, 1.5, n=(8, 4, 6), quarter=True) + [0.0, 0.0, 0.5]
    walls = box_panels(0.8, 0.4, 0.6, n=(12, 6, 6), quarter=True)[:, ::-1] + [0.0, 0.0, 0.0]
    raw = surface(np.concatenate([hull, walls]), symmetry="xz_and_yz")
    zt, pitch = -0.3, 3.0
    w = np.array([1.5])
    out = {}
    for rho_t in (0.0, 1.0):
        # the tank of the trimmed body is cut at its own free surface
        tank = {"mesh": part(raw, len(hull) + 1, len(hull) + len(walls)), "density_ratio": rho_t, "surface_level": zt}
        body = {"mesh": part(raw, 1, len(hull)), "trim": [0.0, pitch, 0.0], "tanks": [tank], "modes": ["heave"]}
        out[rho_t] = model(body, w, (), options={"diffraction": False}).solve()
    # the tank pitched about the origin: its walls 1.6 x 0.8 m from zeta = -0.6; the level cut z = -0.3
    # is zeta = (-0.3 - xi sin)/cos in the tank's axes, so the fluid below it has the mean height
    # 0.6 - 0.3/cos over the 1.6 x 0.8 m bottom, and the free surface the area 1.6 x 0.8 / cos
    c = _m.cos(_m.radians(pitch))
    area = 1.6 * 0.8 / c
    vol = 1.6 * 0.8 * (0.6 - 0.3 / c)
    expect = 1000.0 * (vol - 9.80665 * area / w[0] ** 2)
    da = out[1.0].added_mass[0, 2, 0] - out[0.0].added_mass[0, 2, 0]  # rows: the six coordinates
    assert da == pytest.approx(expect, rel=3e-3)
    assert abs(out[1.0].radiation_damping[0, 2, 0] - out[0.0].radiation_damping[0, 2, 0]) <= 2e-3 * abs(expect)
    # the tank of a trimmed patch mesh is not cut at a free surface of its own
    two = flat(np.concatenate([hull[:1], walls[:1]]), (2, 2, 3, 3, 3, 4))
    body = {"mesh": part(two, 1, 1), "trim": [0.0, 1.0, 0.0], "tanks": [{"mesh": part(two, 2, 2), "surface_level": zt}]}
    with pytest.raises(ValueError, match="low-order"):
        model(body, w, (), options={"higher_order": True}).route()


def tank_hull():
    """A cylinder built above the water with a tank (elements 1..nh the hull, nh+1..nh+nt the tank
    walls) and the trim of the tests."""
    hull = cylinder_panels(1.0, 1.2, n_phi=8, n_z=6, n_r=3, quarter=True) + [0.0, 0.0, 0.4]
    walls = box_panels(0.4, 0.3, 0.4, n=(4, 3, 3), quarter=True)[:, ::-1] + [0.0, 0.0, -0.2]
    return surface(np.concatenate([hull, walls]), symmetry="xz_and_yz"), len(hull), len(walls)


TRIM = (0.0, 2.0, 0.0)
TANK = {"density_ratio": 1.0, "surface_level": -0.3}


def trimmed_tank_body(mesh, nh, nt, **kw):
    return {"name": "spar", "mesh": part(mesh, 1, nh), "trim": list(TRIM),
            "tanks": [{"mesh": part(mesh, nh + 1, nh + nt), **TANK}], **kw}  # fmt: skip


def _one_mesh(*meshes):
    """Low-order trimmed surfaces as one surface mesh, in order (a dict of its items)."""
    assert len({mm.symmetry for mm in meshes}) == 1
    return surface(np.concatenate([mm.vertices for mm in meshes]), symmetry=meshes[0].symmetry)


def test_trimmed_body_equals_its_trimmed_surfaces_as_one_mesh():
    """The trim of a body (with a tank cut at its free surface and a generated interior surface) is
    its trimmed surfaces given as one mesh with their elements, bit for bit."""
    mesh, nh, nt = tank_hull()
    w = [1.0, 1.5]
    new = model(trimmed_tank_body(mesh, nh, nt, **flat_surface("interior_surface", {"method": "graded"})), w, [0.0])
    b = new.bodies["spar"]
    wet, tank = b.trimmed_mesh("spar_wet", "wetted"), b.trimmed_mesh("spar_tank", "tank")
    both, nw = _one_mesh(wet, tank), len(wet.vertices)
    old = model({"name": "spar", "mesh": part(both, 1, nw), "interior_surface": {"method": "graded"},
                 "tanks": [{"mesh": part(both, nw + 1, len(both["vertices"])), **TANK}]}, w, [0.0])  # fmt: skip
    same(new.solve(), old.solve())
    # the ranges stay those of the stored elements of the source mesh
    assert b.mesh == "spar" and b.ranges.tolist() == [[1, nh]] and b.tanks[0].ranges.tolist() == [[nh + 1, nh + nt]]
    # the hydrostatics of the trimmed body: its submerged part, and the tank fluid below its free surface
    h, h0 = b.hydrostatics(), old.bodies["spar"].hydrostatics()
    for k in ("volumes", "center_of_buoyancy", "stiffness_rho_g"):
        assert getattr(h, k).tobytes() == getattr(h0, k).tobytes(), k
    np.testing.assert_allclose(h.volumes, wet.geometry().volumes, rtol=1e-14)


def test_higher_order_trimmed_body_equals_the_trimmed_mesh_with_its_interior_surface():
    # the same equations: bit for bit
    def ho(new):
        m = model((), [2.0], [0.0], options={"higher_order": True})
        m.meshes.analytic("box", "circular_cylinder", [(4, 4, 3, 3, 3, 4), (4, 2, 3, 3, 3, 4)], radius=1.0, draft=2.0)
        modes = ["surge", "heave", "pitch"]
        if new:
            m.bodies.add("box", mesh="box", trim=[1.0, 15.0, 0.0], modes=modes,
                         **flat_surface("interior_surface", {"method": "waterline"}))
            return m
        # the trimmed surfaces of the body (wetted, interior) as one trimmed patch mesh of their source
        tb = m.bodies.add("box", mesh="box", trim=[1.0, 15.0, 0.0], interior_surface="waterline")
        wet, inside = tb.trimmed_mesh("tw", "wetted"), tb.trimmed_mesh("ti", "interior_surface")
        del m.bodies["box"]
        nw, ni = len(wet.trim_elements), len(inside.trim_elements)
        m.meshes.add("t", kind="trimmed_patches", source=wet.source, trim=wet.trim, symmetry=wet.symmetry,
                     trim_elements=np.concatenate([wet.trim_elements, inside.trim_elements]),
                     parameter_ranges=np.concatenate([wet.parameter_ranges, inside.parameter_ranges]),
                     discretization=np.concatenate([wet.discretization, inside.discretization]))  # fmt: skip
        m.bodies.add("box", mesh="t", ranges=[(1, nw)], modes=modes,
                     interior_surface="mesh", interior_surface_mesh="t",
                     interior_surface_ranges=[(nw + 1, nw + ni)])  # fmt: skip
        return m

    same(ho(True).solve(), ho(False).solve())


def test_trimmed_mesh_of_each_role_and_panels_by_role():
    """body.trimmed_mesh gives the surface of one role as the engine uses it (after the trim), the
    elements m.panels gives for that role (a generated interior surface too, made from the new
    waterline), the same vertices to rounding. That an untrimmed body of those meshes solves as the
    trimmed body is test_trimmed_body_equals_its_trimmed_surfaces_as_one_mesh."""
    mesh, nh, nt = tank_hull()
    w = [1.0, 1.5]
    m = model(trimmed_tank_body(mesh, nh, nt, **flat_surface("interior_surface", {"method": "graded"})), w, [0.0])
    b = m.bodies["spar"]
    made = {}
    for role in ("wetted", "tank"):
        made[role] = b.trimmed_mesh(f"spar_{role}", role)
        q = b.panels(role=role, include_reflections=False)
        assert len(q.element) == len(made[role].vertices) > 0, role
        assert set(q.role.tolist()) == {role}, role
        np.testing.assert_allclose(q.vertices, made[role].vertices, atol=1e-12)
    # the elements of the hull come from the hull elements of the source, the tank's from its walls
    assert np.all(made["wetted"].source_elements <= nh) and np.all(made["tank"].source_elements > nh)
    wet = b.panels(role="wetted")
    assert np.all(wet.vertices[..., 2] <= 1e-12) and len(b.panels().element) == len(m.panels().element)
    assert np.all(b.panels(role="interior_surface").vertices[..., 2] == 0.0)
    lid = b.trimmed_mesh("spar_interior", "interior_surface")
    q = b.panels(role="interior_surface", include_reflections=False)
    assert len(q.element) == len(lid.vertices) > 0 and not np.any(lid.source_elements)
    np.testing.assert_allclose(q.vertices, lid.vertices, atol=1e-12)
    assert made["tank"].symmetry == made["wetted"].symmetry


def test_the_records_of_a_trimmed_body_are_its_stored_records():
    """The queries of a trimmed body number its records by the stored records of its mesh, the
    numbering of the mesh references (a record the trim cut: the record it was cut from; the trim
    drops and splits records, so that they do not map one to one); mode_displacements takes them,
    and refuses a record the body does not use."""
    mesh, nh, nt = tank_hull()
    m = model(trimmed_tank_body(mesh, nh, nt), [1.0], [0.0])
    b = m.bodies["spar"]
    for role, name in (("wetted", "w"), ("tank", "t")):
        source = b.trimmed_mesh(name, role).source_elements
        assert b.panels(role=role, include_reflections=False).element.tolist() == source.tolist(), role
    p = b.panels(include_reflections=False)
    source = p.element
    assert len(source) != nh + nt and set(source.tolist()) <= set(range(1, nh + nt + 1))
    assert not np.array_equal(p.element, np.arange(1, len(p.element) + 1))
    assert np.array_equal(b.tessellation(divisions=(1, 1), include_reflections=False).element, p.element)
    assert np.array_equal(b.mesh_quality().element, p.element)
    c = m.counts().per_body[0]
    assert c.stored == nh + nt and c.wetted == np.count_nonzero(source <= nh)
    # the rigid modes at the centroids; a dry element of the hull is no element of the body
    u = m.mode_displacements(p.centroids, p.body, p.element)
    np.testing.assert_allclose(u[:, 2], [[0.0, 0.0, 1.0]] * len(p.element))  # heave
    dry = sorted(set(range(1, nh + 1)) - set(p.element.tolist()))[0]
    with pytest.raises(ValueError, match=f"element {dry} of the body's mesh is not used by the body"):
        m.mode_displacements(p.centroids[:1], [0], [dry])


def test_trimmed_tank_hull_refusals_and_a_zero_trim():
    mesh, nh, nt = tank_hull()
    # a given interior surface lies on the waterline of the untrimmed hull: a trimmed body generates it
    lid = surface(np.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0]]]),
                  symmetry="xz_and_yz")
    m = model(trimmed_tank_body(mesh, nh, nt, **flat_surface("interior_surface", (lid, (1, 1)))), [1.0], [0.0])
    with pytest.raises(ValueError, match="a trimmed body has a generated interior surface"):
        m.route()
    # a lid from the elements of a trimmed body: given as a mesh of its own
    both = surface(np.concatenate([mesh["vertices"][:nh], lid["vertices"]]), symmetry="xz_and_yz")
    m = model({"name": "spar", "mesh": part(both, 1, nh), "trim": list(TRIM)}, [1.0], [0.0],
              lids=[{"name": "l", "mesh": part(both, nh + 1, nh + 1), "placed_with": "spar"}])  # fmt: skip
    with pytest.raises(ValueError, match="a lid from the elements of the trimmed body 'spar' is not available"):
        m.route()
    # unset, the mesh is the wetted surface; [0, 0, 0] cuts the hull at z = 0
    m = model({"name": "spar", "mesh": part(mesh, 1, nh)}, [1.0], [0.0])
    m.bodies["spar"].trim = [0.0, 0.0, 0.0]
    assert np.all(m.bodies["spar"].panels(role="wetted").vertices[..., 2] <= 1e-12)
    m.bodies["spar"].trim = None
    assert m.bodies["spar"].trim is None and m.bodies["spar"].panels(role="wetted").vertices[..., 2].max() > 0.3


def _signed_areas(v):
    """The signed areas of the quadrilaterals v [n, 4, 3] (a triangle with a corner twice) seen from
    above (the shoelace formula of the corners)."""
    x, y = v[..., 0], v[..., 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)


def moonpool_hull():
    """The square barge 4 x 4 x 1 m with a 2 x 2 m moonpool (its hull: the quadrant of flat patches
    of MOONPOOL_FACES without the moonpool's free surface)."""
    q = np.array([face_panels(o, du, dv, 1, 1, n)[0] for o, du, dv, n in MOONPOOL_FACES[:6]])
    return flat(q, (2, 2, 3, 3, 3, 4), symmetry="xz_and_yz")


def two_hulls():
    """Two boxes 2 x 1 x 0.5 m side by side (y = -1.5, 1.5) as one mesh of flat patches."""
    q = np.concatenate([box_panels(1.0, 0.5, 0.5, n=(1, 1, 1)) + [0.0, s, 0.0] for s in (-1.5, 1.5)])
    return flat(q, (2, 2, 3, 3, 3, 4))


@pytest.mark.parametrize("mesh, trim", [
    (moonpool_hull, [0.2, 0.0, 0.0]), (moonpool_hull, [0.2, 3.0, 0.0]), (moonpool_hull, [0.2, 3.0, 2.0]),
    (two_hulls, [0.1, 0.0, 0.0]), (two_hulls, [0.1, 2.0, 1.0]),
])  # fmt: skip
def test_generated_interior_surface_covers_the_waterplane_once(mesh, trim):
    """The generated interior surface of a trimmed waterline that is not one star-shaped loop (a
    moonpool: a hole, or with the symmetry planes kept a loop around it; two hulls: two loops) is a
    triangulation of the waterplane: its triangles, all with normals up, sum to the waterplane area
    of the hydrostatics (the integral over the wetted surface), to 1e-6: the quadrature of the cut
    elements and the vertices of straight runs merged within the length tolerance (7.2e-8 measured,
    2e-15 without a pitch); a fan from one centre would cover the moonpool or the water between
    the hulls too."""
    m = model({"mesh": mesh(), "trim": trim, "interior_surface": {"method": "waterline"}}, [1.0], (),
              options={"higher_order": True, "diffraction": False})  # fmt: skip
    a = _signed_areas(m.bodies["body"].panels(role="interior_surface").vertices)
    sign = np.sign(a[np.argmax(np.abs(a))])
    assert np.all(sign * a > 0.0)
    area = hydrostatics(m)[0].waterplane_area
    np.testing.assert_allclose(np.abs(a).sum(), area, rtol=1e-6)
    m.route()
