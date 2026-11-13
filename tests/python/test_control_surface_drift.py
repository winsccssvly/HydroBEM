"""Mean drift by the momentum flux through a control surface (theory §11.3).

The mean momentum balance (theory §11.3) is an identity of the continuous problem: the horizontal
drift of a free cylinder through a control surface equals the momentum (far-field) drift, and no
component depends on the control surface; the bounds are what the discretization of the body and
of the surfaces leaves (the low-order fields of constant sources at the panel centroids of the
surface, O(h); the higher order much smaller). Exact: the free-surface part enters the vertical
force and the horizontal moments only, the two forms of the squared-elevation line term have the
same vertical terms, a generated surface (a method of the body's control surface) is the surface
of the same CSF file. Further identities: the symmetry sectors, the body origin below the
waterline, the waterline terms of a flared body; the inputs are checked. The patch surfaces
against the panel surface (two discretizations of one surface) are validation, not a test.
Meshes are procedural (tests/python/meshes.py)."""


import numpy as np
import pytest

import hydrobem as hb

from bodies import lo_free_cylinder, lo_free_flared_cylinder, part, surface
from meshes import cylinder_panels
from problems import RHO, G, add_mesh, analytic, flat_surface, model


def annulus(r0, r1, n_phi, n_r):
    """The free surface r0 <= r <= r1 (quadrant), upward normals."""
    ph, rs = np.linspace(0.0, np.pi / 2, n_phi + 1), np.linspace(r0, r1, n_r + 1)
    a = np.array([[[r * np.cos(t), r * np.sin(t), 0.0] for r, t in
                   ((rs[i], ph[j]), (rs[i + 1], ph[j]), (rs[i + 1], ph[j + 1]), (rs[i], ph[j + 1]))]
                  for i in range(n_r) for j in range(n_phi)])  # fmt: skip
    n = np.cross(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])
    return a[:, ::-1] if n[0, 2] < 0 else a


def control_surface(radius, depth=1.5):
    side_bottom = cylinder_panels(radius, depth, n_phi=16, n_z=12, n_r=12, quarter=True)
    return surface(np.concatenate([side_bottom, annulus(1.0, radius, 16, 8)]), symmetry="xz_and_yz")


def cylinder_cs(radius, depth, inner_radius, panel_size):
    """The generated cylindrical control surface (the method "cylinder" of a body's control
    surface): one quadrant of the cylinder with the free surface between the circular waterline
    ``inner_radius`` and it."""
    return {"method": "cylinder", "radius": radius, "depth": depth, "inner_radius": inner_radius,
            "panel_size": panel_size}  # fmt: skip


def auto_cs(radius, depth, panel_size, **kw):
    """The automatic control surface around the body (the method "automatic"), generated from its
    waterline at the build of the problem; ``radius`` None for a ``boundary`` (outer "polygon")."""
    # the switch outer always named: a switch set before keeps its value (nothing resets it)
    outer = {"outer": "polygon"} if radius is None else {"outer": "circle", "radius": radius}
    return {"method": "automatic", **outer, "depth": depth, "panel_size": panel_size, **kw}


def csf(path):
    """A control surface read from a CSF file around the body "body" (MeshList.load_csf)."""
    return lambda m, name: m.meshes.load_csf(name, path, body="body")


W = np.array([2.5, 3.5])


def drifts(radius, fixed=False):
    b = {**lo_free_cylinder(), "control_surface": {"method": "mesh", "mesh": control_surface(radius)},
         **({"modes": []} if fixed else {})}  # fmt: skip
    opts = {"motion": not fixed, "drift_momentum_conservation": True, "drift_control_surface": True}
    return model(b, W, [0.0, 30.0], options=opts).solve()


@pytest.mark.slow
def test_horizontal_drift_equals_the_momentum_drift_and_the_surface_does_not_matter():
    # 3 % of the peak: the low-order fields at the centroids of the control-surface panels (the same
    # 16 x 12 x 12 and 16 x 8 panels at both radii) against the far field (1.5 % and 2.5 % observed);
    # the radii against each other 2 % (heave, the moments; 1 % surge and sway)
    d2, d3 = drifts(2.0), drifts(3.0)
    mom = d2.drift.momentum_conservation.real  # [F, P, 3] surge, sway, yaw (global = body axes here)
    peak = np.abs(mom[..., :2]).max()
    for d in (d2, d3):
        cs = d.drift.control_surface.real
        assert np.abs(cs[..., 0] - mom[..., 0]).max() <= 0.03 * peak
        assert np.abs(cs[..., 1] - mom[..., 1]).max() <= 0.03 * peak
    c2, c3 = d2.drift.control_surface.real, d3.drift.control_surface.real
    for k in range(6):
        assert np.abs(c2[..., k] - c3[..., k]).max() <= 0.03 * np.abs(c2[..., k]).max() + 1e-6 * peak, k


def test_fixed_body_vertical_force_does_not_depend_on_the_surface():
    # the balance holds on any surface; 3 %: the low-order fields at the panel centroids of the two
    # surfaces, whose free-surface parts differ (1.7 % observed)
    f2, f3 = (drifts(r, fixed=True).drift.control_surface.real[..., 2] for r in (2.0, 3.0))
    assert np.abs(f2 - f3).max() <= 0.03 * np.abs(f2).max()


@pytest.mark.slow
def test_vertical_drift_of_a_body_that_is_not_wall_sided():
    """The waterline terms with the motions only, (1/2) rho g int n_z / sqrt(1 - n_z^2) <zeta_b^2> dl
    (physics 16), vanish on a wall-sided body; on a flared one they are larger than the drift
    itself, and the control-surface force agrees with the pressure integration only with them."""
    flare = 0.3
    b = {**lo_free_flared_cylinder(flare), "control_surface": auto_cs(2.5, 1.5, 0.1)}
    opts = {"motion": True, "drift_control_surface": True, "drift_pressure_integration": True}
    r = model(b, W, [0.0], options=opts).solve()
    cs, pressure = r.drift.control_surface.real[:, 0, 2], r.drift.pressure_integration.real[:, 0, 2]
    # the term in head seas on the circular waterline of radius R: zeta_b = xi_3 - alpha_2 R cos(theta),
    # the slope of the side n_z / sqrt(1 - n_z^2) = flare / draft
    xi3, alpha2 = r.motion[:, 0, 2], r.motion[:, 0, 4]
    radius, slope = 1.0 + flare, flare / 0.5
    term = 0.25 * RHO * G * slope * radius * (2 * np.pi * np.abs(xi3) ** 2 + np.pi * radius**2 * np.abs(alpha2) ** 2)
    assert np.all(term > np.abs(cs))
    assert np.all(np.abs(cs - pressure) <= 0.1 * term)


def test_control_surface_inputs_are_checked():
    b = lo_free_cylinder()
    with pytest.raises(ValueError, match="needs a control_surface"):
        model(b, W, [0.0], options={"motion": True, "drift_control_surface": True}).counts()
    bc = {**b, "control_surface": {"method": "mesh", "mesh": control_surface(2.0)}}
    m = model(bc, W, [0.0], options={"motion": True})
    with pytest.raises(ValueError, match="a control_surface needs second_order.drift_control_surface"):
        m.counts()
    with pytest.raises(ValueError, match="bodies/body: control_surface_mesh: unknown mesh 'cs'"):
        m.bodies["body"].update(**flat_surface("control_surface", {"method": "mesh", "mesh": "cs"}))
    # the low order solves the drift by sources, which cannot represent a thin sheet
    fin = [[[0.5, 0.25, -0.4], [0.8, 0.25, -0.4], [0.8, 0.25, -0.1], [0.5, 0.25, -0.1]]]
    pan = np.concatenate([cylinder_panels(1.0, 0.5, quarter=True), fin])
    sm = surface(pan, symmetry="xz_and_yz")
    sheet = {"mesh": part(sm, 1, len(pan) - 1), "dipoles": [{"mesh": part(sm, len(pan), len(pan))}], "modes": [],
             "control_surface": {"method": "mesh", "mesh": control_surface(2.0)}}
    with pytest.raises(hb.UnsupportedProblemError, match="control-surface drift, .* with thin sheets"):
        model(sheet, W, [0.0], options={"drift_control_surface": True}).route()
    # an air cushion: the control-surface formula would take the cushion's free surface as free, a
    # drift 7 % from the momentum drift and in another direction
    from bodies import acv_body

    acv = {**acv_body(modes=("uniform",)), "modes": [], "control_surface": {"method": "mesh",
                                                                            "mesh": control_surface(15.0, 3.0)}}
    with pytest.raises(ValueError, match="not available with pressure surfaces"):
        model(acv, W, [0.0], options={"drift_control_surface": True}).counts()
    # generalized modes, dipole dampers and tanks are not in the formula; the diffraction is
    fixed = {**b, "modes": [], "control_surface": {"method": "mesh", "mesh": control_surface(2.0)}}
    hull = b["mesh"]
    for extra in ({"generalized_modes": [{"name": "g", "pieces": [{"w": [[1.0, 0, 0, 0]]}]}]},
                  {"mesh": part(hull, 3, len(hull["vertices"])), "tanks": [{"mesh": part(hull, 1, 2)}]}):
        with pytest.raises(ValueError, match="generalized modes, dipole dampers or tanks"):
            model({**fixed, **extra}, W, [0.0], options={"drift_control_surface": True, "motion": True}).counts()
    with pytest.raises(ValueError, match="requires the diffraction solution"):
        model(fixed, W, [], options={"drift_control_surface": True, "diffraction": False}).counts()


def test_symmetric_body_gives_the_drift_of_the_full_mesh():
    """The control-surface fields of a symmetric problem come from one point of every orbit: 1e-5,
    the float32 panel integrals summed in another order, the drift quadratic in them (6e-7
    observed)."""

    def run(quarter):
        b = {**lo_free_cylinder(quarter=quarter), "control_surface": {"method": "mesh", "mesh": control_surface(2.0)}}
        opts = {"motion": True, "drift_control_surface": True}
        res = model(b, W, [0.0, 30.0], options=opts).solve()
        return res.drift.control_surface.real

    a, b = run(True), run(False)
    assert np.abs(a - b).max() <= 1e-5 * np.abs(b).max()


def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(["control surface", *lines]) + "\n", encoding="ascii")
    return p


def test_patch_control_surfaces_agree_with_the_panel_surface(tmp_path):
    """The same cylinder R = 2, depth 1.5 as patches: the cylinder and the automatic surface traced
    from the body waterline, as CSF files. The two differ in their free-surface part only (its inner
    boundary the circle or the polygon of the body waterline), which enters the vertical force and
    the horizontal moments only: surge, sway and yaw identical. The generated surfaces (the methods
    of the body's control surface) are those of the files."""
    ccyl = csf(_write(tmp_path, "c.csf", ["1", "1 1", "3 -1001 0.25", "2", "2.0 1.5 1.0", "0"]))
    auto = csf(_write(tmp_path, "a.csf", ["1", "1 1", "0 0 0.25", "2.0 1.5", "0"]))
    m = model(lo_free_cylinder())
    # side, bottom, one per waterline segment
    assert len(ccyl(m, "ccyl").patches) == 3 and len(auto(m, "auto").patches) == 2 + 8
    a, c = drifts_with(ccyl), drifts_with(auto)
    np.testing.assert_array_equal(a[..., [0, 1, 5]], c[..., [0, 1, 5]])
    assert np.abs(a[..., 2] - c[..., 2]).max() > 0.0
    np.testing.assert_array_equal(drifts_with(cylinder_cs(2.0, 1.5, 1.0, 0.25)), a)
    np.testing.assert_array_equal(drifts_with(auto_cs(2.0, 1.5, 0.25)), c)


def drifts_with(cs, body=None):
    """The control-surface drift of the body (default the free cylinder) with the control surface
    ``cs``: the items of a generated surface (a dict with its method), a dict of mesh items, or a
    callable ``f(model, name)`` made once the body is in the model."""
    opts = {"motion": True, "drift_control_surface": True}
    m = model(body or lo_free_cylinder(), W, [0.0, 30.0], options=opts)
    if isinstance(cs, dict) and "method" in cs:
        m.bodies["body"].update(**flat_surface("control_surface", cs))
        return m.solve().drift.control_surface.real
    if callable(cs):
        cs(m, "cs")
    else:
        add_mesh(m, "cs", cs)
    m.bodies["body"].update(**flat_surface("control_surface", {"method": "mesh", "mesh": "cs"}))
    return m.solve().drift.control_surface.real


def test_free_surface_part_changes_only_the_vertical_force_and_the_horizontal_moments():
    # the free-surface term is e_z times an integral, its moment x times it: surge, sway and yaw
    # are the same sums without it (exact)
    # (the automatic surface always has its free-surface part: here a given surface with it, and
    # the same surface without it)
    full = drifts_with(control_surface(2.0))
    outer = drifts_with(surface(cylinder_panels(2.0, 1.5, n_phi=16, n_z=12, n_r=12, quarter=True),
                                symmetry="xz_and_yz"))  # fmt: skip
    peak = np.abs(full).max()
    np.testing.assert_array_equal(full[..., [0, 1, 5]], outer[..., [0, 1, 5]])
    assert np.abs(full[..., 2] - outer[..., 2]).max() > 1e-3 * peak


def test_control_surface_with_the_body_origin_below_the_waterline():
    """The body raised in its axes and placed back (z position -0.3): the same horizontal drift;
    the heave drift changes by the term -rho g A_wp (alpha_1^2 + alpha_2^2) Z_o / 2 of the
    hydrostatic part, the time mean of the products. The mass is the
    displaced one: an unbalanced body's restoring depends on the reference point. 1e-6 and 1e-5:
    the float32 panel integrals of the shifted vertices (2e-8 and 1e-7 observed)."""
    b = {**lo_free_cylinder(), "mass": "displaced"}
    v = np.asarray(b["mesh"]["vertices"]).copy()
    v[..., 2] += 0.3
    b2 = {**b, "mesh": surface(v, symmetry="xz_and_yz"), "position": [0.0, 0.0, -0.3], "mass": "displaced",
          "center_of_gravity": [0.0, 0.0, 0.1]}
    opts = {"motion": True, "drift_control_surface": True}
    out = []
    for body in (b, b2):  # the automatic surface takes the waterline from the body position
        out.append(model({**body, "control_surface": auto_cs(2.0, 1.5, 0.25)}, W, [0.0, 30.0], options=opts).solve())
    a, c = (r.drift.control_surface.real for r in out)
    assert np.abs(a[..., :2] - c[..., :2]).max() <= 1e-6 * np.abs(a).max()
    hs = out[0].model.bodies[0].hydrostatics()
    al = np.asarray(out[0].motion)[..., 3:5]
    term = -1000.0 * 9.80665 * hs.waterplane_area * 0.25 * (np.abs(al) ** 2).sum(axis=-1) * (-0.3)
    assert np.abs(c[..., 2] - a[..., 2] - term).max() <= 1e-5 * np.abs(term).max()


def test_new_patch_kinds_and_the_csf_forms(tmp_path):
    m = model(lo_free_cylinder())
    e = {"type": "AnalyticPatch", "map": "ellipses", "profile": "linear", "a": 2.2, "b": 0.3, "theta0": 0.0,
         "theta1": np.pi / 2, "p": [2.0, 0.25, 0.0, 0.0]}  # fmt: skip
    s = {"type": "AnalyticPatch", "map": "segment_arc", "profile": "linear", "a": 2.0, "b": 2.0, "theta0": 0.0,
         "theta1": np.pi / 2, "p": [0.0, 0.0, 0.0, 0.0], "vertices": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}  # fmt: skip
    kinds = add_mesh(m, "kinds", {"kind": "analytic_patches", "symmetry": "xz_and_yz", "patches": [e, s]})
    kinds.discretization = np.tile([2, 2, 3, 3, 3, 4], (2, 1))
    x = kinds.patch_points(1, [[-1.0, -1.0], [1.0, 1.0]]).points
    assert np.allclose(x[0], [2.0, 0.0, 0.0]) and np.allclose(x[1], [0.0, 0.3, 0.0])
    x = kinds.patch_points(2, [[0.0, -1.0], [0.0, 1.0]]).points
    assert np.allclose(x[0], [0.5, 0.5, 0.0])
    assert np.allclose(x[1], [np.sqrt(2), np.sqrt(2), 0.0])
    el = m.meshes.load_csf("e", _write(tmp_path, "e.csf", ["1", "1 1", "1 -1003 0.5", "2", "2.2 0.3 0.3", "2.0 0.25"]))
    assert len(el.patches) == 1 and el.patches[0].map == "arc"  # NPATCSF = 1: the ellipsoid without its free surface
    f = m.meshes.load_csf("f", _write(tmp_path, "f.csf", ["1", "1 0", "1 0 10.0",
                                                          "100 -50 -15 100 30 -15 100 30 0 100 -50 0"]))  # fmt: skip
    assert f.kind == "flat_patches" and f.discretization[0][0] == 9 and f.discretization[0][4] == 3
    lo = m.meshes.load_csf("l", _write(tmp_path, "l.csf", ["0", "1 1", "1", "2 0 0 2 0 -1 2 1 -1 2 1 0"]))
    assert lo.kind == "panels"
    for lines, msg in (
        (["2", "a.csf b.csf"], "ILOWHICSF = 2"),
        (["1", "1 1", "0 0 -1", "2 1", "0"], "PSZCSF"),
        (["1", "1 1", "0 0 1", "2 1", "1"], r"NV\(1\)"),
        (["1", "1 1", "1 2 1"], "ICDEF = 2"),
    ):
        with pytest.raises(ValueError, match=msg):
            m.meshes.load_csf("x", _write(tmp_path, "x.csf", lines), body="body")
    with pytest.raises(ValueError, match="needs the body mesh"):
        m.meshes.load_csf("y", _write(tmp_path, "y.csf", ["1", "1 1", "0 0 1", "2 1", "0"]))


def ho_cylinder():
    return analytic("circular_cylinder", (4, 4, 3, 3, 3, 4), radius=1.0, draft=0.5)


@pytest.mark.slow
def test_higher_order_drift_equals_the_momentum_drift_in_both_forms():
    """The higher-order solution (the fields by differences, one point per orbit of the quadrant
    symmetry; its waterline exact) at an oblique heading. The momentum balance: the horizontal
    drift and the yaw moment equal the momentum drift (both from the same potential), and the
    vertical force does not depend on the surface. The two forms of the squared-elevation line
    term (``control_surface.waterline_term``): on CL, or on the body waterline with the integral of
    zeta grad zeta over the free-surface part, equal by the divergence theorem on S_f; the vertical
    force and the horizontal moments are the same terms in both (exact). The bounds: the Gauss
    rules of the surface patches (0.5) and the float32 field rows, 1e-5 (5e-7 observed); the second
    form 1e-3, the free-surface integral of zeta grad zeta by the patch rule (1.8e-4); the radii
    1e-3, the free-surface part differs between them (8e-5). (The low-order waterline terms take
    the potential of the waterline panels at their centroids, so there waterline_term body is the
    less accurate one.)"""
    opts = {"higher_order": True, "motion": True, "drift_control_surface": True, "drift_momentum_conservation": True}
    out = {}
    for alt in ("control_surface", "body"):
        b = {**lo_free_cylinder(), "mesh": ho_cylinder(),
             "control_surface": {**cylinder_cs(2.0, 1.5, 1.0, 0.5), "waterline_term": alt}}  # fmt: skip
        out[alt] = model(b, W, [0.0, 30.0], options=opts).solve()
    c1, c2 = out["control_surface"].drift.control_surface.real, out["body"].drift.control_surface.real
    mom = out["control_surface"].drift.momentum_conservation.real
    peak = np.abs(mom[..., :2]).max()
    assert np.abs(c1[..., :2] - mom[..., :2]).max() <= 1e-5 * peak
    assert np.abs(c1[..., 5] - mom[..., 2]).max() <= 1e-5 * peak
    np.testing.assert_array_equal(c2[..., 2:5], c1[..., 2:5])
    assert np.abs(c2[..., :2] - mom[..., :2]).max() <= 1e-3 * peak
    ho3 = {**lo_free_cylinder(), "mesh": ho_cylinder(), "control_surface": cylinder_cs(3.0, 2.0, 1.0, 0.5)}
    cs3 = model(ho3, W, [0.0, 30.0], options=opts).solve().drift.control_surface.real
    assert np.abs(cs3[..., 2] - c1[..., 2]).max() <= 1e-3 * np.abs(c1[..., 2]).max()


def test_control_surface_waterline_term_is_checked():
    m = model({**lo_free_cylinder(), "control_surface": cylinder_cs(2.0, 1.5, 1.0, 0.5)}, W, [0.0])
    m.bodies["body"].control_surface_waterline_term = "body"
    with pytest.raises(ValueError, match=r"waterline_term = 'body' needs second_order.drift_control_surface"):
        m.counts()
    m.second_order.drift_control_surface = True
    with pytest.raises(ValueError, match="waterline_term must be one of control_surface, body"):
        m.bodies["body"].control_surface_waterline_term = "ship"


def _free_surface_area(mesh):
    s = np.linspace(-1.0, 1.0, 201)
    m = 0.5 * (s[1:] + s[:-1])
    U, V = np.meshgrid(m, m, indexing="ij")

    def point(rec, u, v):
        return mesh.patch_points(rec, np.c_[u.reshape(-1), v.reshape(-1)]).points.reshape(*u.shape, 3)

    total = 0.0
    for rec in range(1, len(mesh.patches) + 1):
        x = point(rec, U, V)
        if np.abs(x[..., 2]).max() > 1e-9:
            continue
        xu = (point(rec, U + 1e-6, V) - point(rec, U - 1e-6, V)) / 2e-6
        xv = (point(rec, U, V + 1e-6) - point(rec, U, V - 1e-6)) / 2e-6
        total += np.linalg.norm(np.cross(xu, xv), axis=-1).sum() * 1e-4
    return total


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_quadrilateral_and_partitioned_automatic_surfaces(tmp_path):
    """The quadrilateral outer boundary (no radius, the first partition) and partitions:
    the free surface of the analytic cylinder inside the
    square 2 x 2 of the quadrant is exact (circular waterline traced exactly); the CSF form reads the
    same surface; a partition along the symmetry planes, extended to the circle, leaves the circular
    surface's area; the horizontal drift through the square equals the momentum drift (the
    momentum balance; 1e-4, the Gauss rules of the square's patches and the float32 field rows: 9e-6
    observed)."""
    opts = {"higher_order": True, "motion": True, "drift_control_surface": True, "drift_momentum_conservation": True}
    m = model({**lo_free_cylinder(), "mesh": ho_cylinder()}, W[:1], [0.0], options=opts)
    q = _write(tmp_path, "q.csf", ["1", "1 1", "0 0 0.25", "0.0 1.5", "1", "3", "2.0 0.0", "2.0 2.0", "0.0 2.0"])
    quad = m.meshes.load_csf("quad", q, body="body")
    assert _free_surface_area(quad) == pytest.approx(4.0 - np.pi / 4, rel=1e-6)
    p = _write(tmp_path, "p.csf", ["1", "1 1", "0 0 0.25", "2.0 1.5", "1", "3", "0.0 1.5", "0.0 0.0", "1.5 0.0"])
    assert _free_surface_area(m.meshes.load_csf("part", p, body="body")) == pytest.approx(0.75 * np.pi, rel=1e-6)
    m.bodies["body"].update(**flat_surface("control_surface", {"method": "mesh", "mesh": "quad"}))
    r = m.solve()
    cs, mom = r.drift.control_surface.real, r.drift.momentum_conservation.real
    assert np.abs(cs[..., 0] - mom[..., 0]).max() <= 1e-4 * np.abs(mom[..., 0]).max()
    # the generated surfaces are those of the files (the items of the method: the boundary, the
    # partitions as polylines each ended by a row of NaN)
    m.bodies["body"].control_surface = "none"  # off between them (its items are kept, unused)
    square = auto_cs(None, 1.5, 0.25, boundary=[(2, 0), (2, 2), (0, 2)])
    m.bodies["body"].update(**flat_surface("control_surface", square))
    np.testing.assert_array_equal(m.solve().drift.control_surface.real, cs)
    m.bodies["body"].control_surface = "none"
    m.bodies["body"].update(**flat_surface("control_surface", {"method": "mesh", "mesh": "part"}))
    parted = m.solve().drift.control_surface.real
    m.bodies["body"].control_surface = "none"
    m.bodies["body"].update(**flat_surface("control_surface", auto_cs(2.0, 1.5, 0.25,
                                               partitions=[(0, 1.5), (0, 0), (1.5, 0), (np.nan, np.nan)])))  # fmt: skip
    np.testing.assert_array_equal(m.solve().drift.control_surface.real, parted)
    # the outer surface of outer: the polygon needs its boundary (unset here: switching outer
    # resets nothing, the boundary of before is removed by hand)
    m.bodies["body"].update(control_surface_outer="polygon", control_surface_boundary=None)
    with pytest.raises(ValueError, match="control_surface = 'automatic' with control_surface_outer = 'polygon' "
                                         "needs control_surface_boundary"):  # fmt: skip
        m.counts()
    with pytest.raises(ValueError, match="needs the outer boundary"):
        m.meshes.load_csf("x", _write(tmp_path, "x.csf", ["1", "1 1", "0 0 0.25", "0.0 1.5", "0"]), body="body")


def test_control_surface_forms():
    """A body's control surface: the item control_surface chooses its method ("none" the default),
    the items of the method prefixed control_surface_; the check names a needed item missing; the
    items of a method not chosen are kept unused (nothing refused or reset)."""
    m = model(lo_free_cylinder(), W, [0.0], options={"motion": True, "drift_control_surface": True})
    b = m.bodies["body"]
    assert b.control_surface == "none"
    b.control_surface = "cylinder"
    with pytest.raises(ValueError, match="control_surface = 'cylinder' needs control_surface_radius"):
        m.counts()
    b.update(control_surface_radius=2.0, control_surface_depth=1.5, control_surface_inner_radius=1.0,
             control_surface_panel_size=0.5, control_surface_waterline_term="body")
    m.counts()
    b.control_surface = "ellipsoid"  # the items of the cylinder kept, unused
    assert b.control_surface_waterline_term == "body" and b.control_surface_radius == 2.0
    with pytest.raises(ValueError, match="control_surface = 'ellipsoid' needs control_surface_semi_axes"):
        m.counts()
    b.update(control_surface="automatic", control_surface_panel_size=0.25)
    assert m.counts().meshes == 2  # the generated surface is a mesh of the problem
    b.control_surface_outer = "polygon"
    with pytest.raises(ValueError, match="control_surface = 'automatic' with control_surface_outer = 'polygon' "
                                         "needs control_surface_boundary"):  # fmt: skip
        m.counts()
    add_mesh(m, "cs", control_surface(2.0))
    b.update(control_surface="mesh", control_surface_mesh="cs", control_surface_ranges=[(1, 10)])
    assert b.control_surface_mesh == "cs" and b.control_surface_ranges.tolist() == [[1, 10]]
    b.control_surface = "none"
    with pytest.raises(ValueError, match="needs a control_surface"):
        m.counts()


def test_two_bodies_sum_to_the_momentum_drift_and_mirror_each_other():
    # two floating cylinders at y = +-3, each in its own control surface (radius 2): the far-field
    # momentum balance of the whole fluid gives the total horizontal drift, so the sum of the bodies'
    # control-surface drifts equals the momentum drift (an identity of the problem; 3 %: the
    # low-order fields at the centroids of the surfaces' panels, as for one body); the wave along x,
    # body b's drift is body a's mirrored about y = 0 (sway, roll and yaw change sign; float32: 1e-5).
    # Measured: the sum 0.44 % of the peak (the drift at 2.5 rad/s is itself small: 92 against 31 of
    # a peak of 13857), the mirror 2.1e-7
    bodies = [{**lo_free_cylinder(), "name": "a", "position": [0.0, 3.0, 0.0],
               "control_surface": {"method": "mesh", "mesh": control_surface(2.0)}},
              {**lo_free_cylinder(), "name": "b", "position": [0.0, -3.0, 0.0],
               "control_surface": {"method": "mesh", "mesh": control_surface(2.0)}}]  # fmt: skip
    opts = {"motion": True, "drift_momentum_conservation": True, "drift_control_surface": True}
    r = model(bodies, W, [0.0], options=opts).solve()
    cs = r.drift.control_surface[:, 0, :].real  # [F, 12]
    mc = r.drift.momentum_conservation[:, 0, :].real  # [F, 3] surge, sway, yaw of all bodies
    total = cs[:, [0, 1]] + cs[:, [6, 7]]
    scale = np.abs(mc[:, :2]).max()
    assert np.abs(total - mc[:, :2]).max() <= 3e-2 * scale, np.abs(total - mc[:, :2]).max() / scale
    sign = np.array([1, -1, 1, -1, 1, -1])
    a, b = cs[:, :6], cs[:, 6:]
    assert np.abs(b - sign * a).max() <= 1e-5 * np.abs(a).max(), np.abs(b - sign * a).max() / np.abs(a).max()


def test_the_footprint_of_a_patch_control_surface_is_its_curve():
    # the rule that a control surface holds its body alone takes the footprint of a generated
    # surface (patches) from the boundaries of its patches: a small body inside the circle but
    # outside the polygon of three points a quadrant (0.924 of the radius at 22.5 degrees) is inside
    # the surface, refused
    t = np.radians(22.5)
    small = surface(cylinder_panels(0.05, 0.1, n_phi=2, n_z=1, n_r=1, quarter=True), symmetry="xz_and_yz")
    a = {**lo_free_cylinder(), "name": "a", "control_surface": cylinder_cs(3.0, 1.5, 1.0, 0.5)}
    b = {"name": "b", "mesh": small, "modes": [], "position": [2.88 * np.cos(t), 2.88 * np.sin(t), 0.0],
         "control_surface": cylinder_cs(0.2, 0.2, 0.05, 0.05)}
    with pytest.raises(ValueError, match="its control surface reaches bodies/b"):
        model([a, b], W[:1], [0.0], options={"motion": True, "drift_control_surface": True}).route()


def test_a_control_surface_holds_its_body_alone():
    # the momentum balance gives the force on everything inside the surface: a surface that reaches
    # another body, or holds a damping lid in its free surface, is refused (an input rule)
    two = [{**lo_free_cylinder(), "name": "a", "control_surface": {"method": "mesh", "mesh": control_surface(2.0)}},
           {**lo_free_cylinder(), "name": "b", "position": [2.5, 0.0, 0.0],
            "control_surface": {"method": "mesh", "mesh": control_surface(2.0)}}]  # fmt: skip
    opts = {"motion": True, "drift_control_surface": True}
    with pytest.raises(ValueError, match="its control surface reaches"):
        model(two, W[:1], [0.0], options=opts).route()
    one = {**lo_free_cylinder(), "control_surface": {"method": "mesh", "mesh": control_surface(2.0)}}
    lid = {"name": "lid", "mesh": surface(annulus(1.2, 1.6, 4, 2), symmetry="xz_and_yz")}
    with pytest.raises(ValueError, match="damping lid lid lies inside"):
        model(one, W[:1], [0.0], options=opts, lids=[lid]).route()


def _box_quads(n):
    """The box |x|, |y| <= 2 down to z = -1.5 as n x n panels per face, one quadrant (the sides
    x = 2 and y = 2 and the bottom), outward normals."""
    t = np.linspace(0.0, 1.0, n + 1)
    quads = []
    for a, b in (([2.0, 0.0], [2.0, 2.0]), ([2.0, 2.0], [0.0, 2.0])):
        a, b = np.asarray(a), np.asarray(b)
        for i in range(n):
            p, q = a + (b - a) * t[i], a + (b - a) * t[i + 1]
            for j in range(n):
                z0, z1 = -1.5 * t[j], -1.5 * t[j + 1]
                quads.append([[*p, z0], [*p, z1], [*q, z1], [*q, z0]])
    for i in range(n):
        for j in range(n):
            x0, x1, y0, y1 = 2 * t[i], 2 * t[i + 1], 2 * t[j], 2 * t[j + 1]
            quads.append([[x1, y0, -1.5], [x0, y0, -1.5], [x0, y1, -1.5], [x1, y1, -1.5]])
    return quads


def test_a_flat_patch_control_surface_is_checked_and_solved(tmp_path):
    # a control surface of flat patches (a CSF file of the higher-order form with ICDEF = 0: the
    # box |x|, |y| <= 2 down to z = -1.5, one quadrant, two sides and the bottom) around a
    # low-order body. The input rule that the surface holds its body alone takes the points of its
    # patches (an untrimmed flat patch by the bilinear map of its vertices: it has no spline data,
    # whose evaluation once aborted the check). Surge and sway agree with those of the same box as
    # panels (the CSF's low-order form; the low-order fields at the panel centroids against those
    # at the patch points: 3 %, as the two radii of the panel surfaces above; 0.88 % observed), and
    # another body inside the patch box is refused
    side_x, side_y = "2 0 0 2 0 -1.5 2 2 -1.5 2 2 0", "2 2 0 2 2 -1.5 0 2 -1.5 0 2 0"
    bottom = "2 0 -1.5 0 0 -1.5 0 2 -1.5 2 2 -1.5"
    patches = _write(tmp_path, "p.csf", ["1", "1 1", "3 0 0.25", side_x, side_y, bottom])
    quads = _box_quads(8)
    panels = _write(tmp_path, "l.csf", ["0", "1 1", str(len(quads)),
                                        *(" ".join(f"{c:g}" for v in q for c in v) for q in quads)])  # fmt: skip
    flat, pan = drifts_with(csf(patches)), drifts_with(csf(panels))
    peak = np.abs(pan[..., [0, 1]]).max()
    assert np.abs(flat[..., [0, 1]] - pan[..., [0, 1]]).max() <= 3e-2 * peak
    # (body a first: the check of its patch box refuses; b's panel surface would reach a too)
    two = [{**lo_free_cylinder(), "name": "a"},
           {**lo_free_cylinder(), "name": "b", "position": [2.5, 0.0, 0.0],
            "control_surface": {"method": "mesh", "mesh": control_surface(2.0)}}]  # fmt: skip
    m = model(two, W[:1], [0.0], options={"motion": True, "drift_control_surface": True})
    m.meshes.load_csf("cs", patches)
    m.bodies["a"].update(**flat_surface("control_surface", {"method": "mesh", "mesh": "cs"}))
    with pytest.raises(ValueError, match="its control surface reaches"):
        m.route()
