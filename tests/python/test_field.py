"""Field pressure below the free surface and field velocity (theory §4.1).

The field output p/(rho g) at z < 0 continues the free-surface elevation;
the velocity of the source formulation equals (i g / w) times the gradient of
the pressure of Green's identity (central differences; an identity of the continuous problem, the
two formulations discretize different unknowns, so the bound is their discretization error, O(h)
of the constant panels); the incident velocity is the analytic one; symmetry sectors equal the
full solve; rules. The convergence of the two formulations to each other with the mesh is
validation, not a test. Meshes are procedural (tests/python/meshes.py)."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import part, spline_cylinder, surface
from meshes import box_panels, cylinder_panels, moonpool_lid_panels, moonpool_panels
from problems import model

G = 9.80665
W = np.sqrt(np.array([0.6, 1.5]) * G)


def cylinder(quarter=True, n=16):
    """The cylinder of radius 1 and draft 0.5, n panels a quadrant around it, n / 2 down and across."""
    pan = cylinder_panels(1.0, 0.5, n_phi=n if quarter else 4 * n, n_z=n // 2, n_r=n // 2, quarter=quarter)
    return {"mesh": surface(pan, symmetry="xz_and_yz" if quarter else "none")}


def solve(body, points, depth=math.inf, velocity=True):
    return model(body, W, [0.0, 40.0], field_points=np.array(points), environment={"depth": depth},
                 options={"field_pressure": True, "field_velocity": velocity}).solve()  # fmt: skip


@pytest.mark.parametrize("depth", [math.inf, 4.0])
def test_velocity_is_the_gradient_of_the_pressure(depth):
    # two formulations (sources for the velocity, Green's identity for the pressure) of one
    # continuous problem: 5e-2, the discretization error of the constant panels on 16 a quadrant
    # (3.3e-2 observed, the radiation; 1.4e-2 the diffraction), slow because the source strength is
    # singular at the bottom edge of the cylinder; the central difference adds O(h^2) = 1e-6 and the
    # float32 pressure about 1e-4
    h = 1e-3
    r = solve(cylinder(), stencil([1.8, 0.7, -0.6], h), depth)
    for kind in ("diffraction", "radiation"):
        e = gradient_error(group(r, "field"), r.omegas, h, key=kind).max()
        assert e < 5e-2, (kind, e)


def test_field_rows_made_at_evaluation_equal_the_stored_ones(monkeypatch):
    # beyond a memory limit the static rows of the field points are made for each point when it is
    # evaluated instead of stored (HB_FIELD_STATICS_LIMIT lowers the limit): the same values, in
    # infinite and finite depth, pressure and velocity
    pts = np.array([[1.8, 0.7, -0.6], [2.5, -1.0, 0.0], [-3.0, 0.2, -1.2]])
    for depth in (math.inf, 4.0):
        stored = solve(cylinder(), pts, depth)
        monkeypatch.setenv("HB_FIELD_STATICS_LIMIT", "0")
        made = solve(cylinder(), pts, depth)
        monkeypatch.delenv("HB_FIELD_STATICS_LIMIT")
        for q in ("field_points_pressure_diffraction", "field_points_pressure_radiation",
                  "field_points_velocity_diffraction", "field_points_velocity_radiation"):  # fmt: skip
            np.testing.assert_array_equal(made.quantity(q), stored.quantity(q), err_msg=q)


def test_incident_velocity_and_sectors():
    # the incident velocity against the analytic one, 1e-6: its complex64 output (2.4e-8 observed);
    # the symmetry sectors against the full solve, 1e-6: the float32 panel integrals summed in
    # another order (1.3e-7 observed)
    pts = [[2.5, -1.0, -0.4], [0.3, 0.2, -1.2]]
    a = solve(cylinder(n=8), pts)
    k = a.omegas**2 / G
    for h, beta in enumerate(np.radians(a.headings)):
        x = np.array(pts)
        phase = np.exp(-1j * k[:, None] * (x[:, 0] * math.cos(beta) + x[:, 1] * math.sin(beta)) + k[:, None] * x[:, 2])
        phi = (1j * G / a.omegas)[:, None] * phase
        ref = np.stack([-1j * k[:, None] * math.cos(beta) * phi, -1j * k[:, None] * math.sin(beta) * phi,
                        k[:, None] * phi], axis=-1)  # fmt: skip
        assert np.abs(a.field_points.velocity_incident[:, :, h] - ref).max() <= 1e-6 * np.abs(ref).max()
    b = solve(cylinder(quarter=False, n=8), pts)
    for key in ("velocity_diffraction", "velocity_radiation", "pressure_diffraction"):
        x, y = a.quantity("field_points_" + key), b.quantity("field_points_" + key)
        assert np.abs(x - y).max() <= 1e-6 * np.abs(y).max(), key


def test_submerged_pressure_continues_the_elevation():
    # the field output p/(rho g) is continuous at z = 0
    r = solve(cylinder(), [[2.0, 0.5, 0.0], [2.0, 0.5, -1e-4]], velocity=False)
    for key in ("pressure_diffraction", "pressure_radiation"):
        f = r.quantity("field_points_" + key)
        assert np.abs(f[:, 0] - f[:, 1]).max() <= 1e-3 * np.abs(f[:, 0]).max(), key


def test_field_rules():
    body = cylinder()
    with pytest.raises(ValueError, match="sea bed"):
        solve(body, [[3.0, 0.0, -5.0]], depth=4.0, velocity=False)
    with pytest.raises(ValueError, match="field points"):
        model(body, W, [0.0], options={"field_velocity": True}).route()
    fin = [[[0.5, 0.25, -0.4], [0.8, 0.25, -0.4], [0.8, 0.25, -0.1], [0.5, 0.25, -0.1]]]
    pan = np.concatenate([box_panels(1.0, 0.5, 0.5, n=(2, 1, 1), quarter=True), fin])
    m = surface(pan, symmetry="xz_and_yz")
    body = {"mesh": part(m, 1, len(pan) - 1), "dipoles": [{"mesh": part(m, len(pan), len(pan))}]}
    with pytest.raises(hb.UnsupportedProblemError, match="sheets"):
        model(body, W, [0.0], field_points=np.array([[5.0, 0.0, -0.5]]), options={"field_velocity": True}).route()


# the same mesh solved two ways: its coarseness at the highest frequency is not what is tested
@pytest.mark.filterwarnings("ignore:.*larger than a sixth of the shortest wavelength")
def test_velocities_beside_a_wall_equal_the_mirror_body():
    # beside the wall x = 0 the field and body velocities (source formulation
    # with the image panels) equal those of the body and its mirror image in open sea
    # under the incident wave and its reflection (headings 30 and 150 deg)
    pan = box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=False)
    fp = np.array([[4.0, -1.0, -0.3], [1.0, 2.5, -0.1]])
    opt = dict(field_pressure=True, field_velocity=True, body_velocity=True)
    m = surface(pan)
    a = model({"mesh": m, "position": [2, 0.5, 0], "yaw": 20.0, "modes": []}, W, [30.0], field_points=fp,
              options={"domain": "x_wall", **opt}).solve()  # fmt: skip
    m2 = surface((pan * [-1, 1, 1])[:, ::-1])
    b = model([{"name": "a", "mesh": m, "position": [2, 0.5, 0], "yaw": 20.0, "modes": []},
               {"name": "b", "mesh": m2, "position": [-2, 0.5, 0], "yaw": -20.0, "modes": []}],
              W, [30.0, 150.0], field_points=fp, options=opt).solve()  # fmt: skip
    for key in ("pressure_diffraction", "velocity_diffraction"):
        fa, fb = a.quantity("field_points_" + key), b.quantity("field_points_" + key)
        x, y = fa[:, :, 0], fb[:, :, 0] + fb[:, :, 1]
        assert np.abs(x - y).max() <= 1e-6 * np.abs(y).max(), key
    n = a.body_points.velocity_diffraction.shape[1]
    v = b.body_points.velocity_diffraction
    x, y = a.body_points.velocity_diffraction[:, :, 0], v[:, :n, 0] + v[:, :n, 1]
    assert np.abs(x - y).max() <= 1e-6 * np.abs(y).max()


def test_body_pressure_and_velocity():
    # the integral of the body pressure is the exciting force (the same centroid rule: 1e-6, the
    # complex64 output; 3e-8 observed), the normal radiation velocity the prescribed body velocity
    # i w n_j (theory §4.1), the boundary condition the source formulation solves at the centroids
    # (1e-5, the float32 panel integrals; 1.1e-6 observed)
    r = model(cylinder(), W, [30.0], options={"body_pressure": True, "body_velocity": True}).solve()
    b = group(r, "body")
    n, area = np.asarray(b["normals"]), np.asarray(b["areas"])
    pts = np.asarray(b["points"])
    nj = np.concatenate([n, np.cross(pts, n)], axis=1)  # about the origin (the rotation centre)
    rho = r.model.environment.density
    # X = -rho g int (p / rho g) n dS (n into the fluid: the force on the body is -int p n)
    X = -rho * G * np.einsum("fq,qi,q->fi", b["pressure_diffraction"][:, :, 0], nj, area)
    assert np.abs(X - r.excitation.diffraction[:, 0, :]).max() <= 1e-6 * np.abs(r.excitation.diffraction).max()
    vn = np.einsum("fqkd,qd->fqk", b["velocity_radiation"], n)
    ref = 1j * r.omegas[:, None, None] * nj[None, :, :]
    assert np.abs(vn - ref).max() <= 1e-5 * np.abs(ref).max()
    assert set(np.asarray(b["ids"])[:, 0]) == {0} and len(b["areas"]) == 4 * 16 * 16


def test_higher_order_field_velocity():
    # differences of the potential: the gradient of the pressure output on a wider
    # stencil, a point in z = 0 (one-sided in z) continuous with one just below it

    x0 = np.array([1.7, 0.6, -0.5])
    hs = 2e-2
    pts = [x0] + [x0 + s * hs * e for e in np.eye(3) for s in (1, -1)] + [[1.8, 0.3, 0.0], [1.8, 0.3, -1e-3]]
    r = model({"mesh": spline_cylinder(n=4)}, W, [0.0, 40.0], field_points=np.array(pts),
              options={"higher_order": True, "field_pressure": True, "field_velocity": True}).solve()  # fmt: skip
    for key, vkey in (("pressure_diffraction", "velocity_diffraction"), ("pressure_radiation", "velocity_radiation")):
        f = r.quantity("field_points_" + key).astype(complex)
        grad = np.stack([(f[:, 1 + 2 * d] - f[:, 2 + 2 * d]) / (2 * hs) for d in range(3)], axis=-1)
        ref = (1j * G / r.omegas)[:, None, None] * grad
        v = r.quantity("field_points_" + vkey)[:, 0].astype(complex)
        assert np.abs(v - ref).max() <= 2e-3 * np.abs(ref).max(), key  # O(hs^2) of the wide stencil
        v = r.quantity("field_points_" + vkey).astype(complex)
        top, below = v[:, 7], v[:, 8]
        assert np.abs(top - below).max() <= 1e-2 * np.abs(top).max(), key


def group(r, prefix):
    """The views of the group of points of results, by their names ("pressure_diffraction", "points"):
    "field" the field points, "body" the body points; the views computed only."""
    g = {"field": r.field_points, "body": r.body_points}[prefix]
    names = [k for k, v in vars(type(g)).items() if isinstance(v, property)]
    return {k: getattr(g, k) for k in names if getattr(g, k) is not None}


def gradient_error(field, omegas, h, first=0, key="diffraction"):
    """|v - (i g / w) grad p| / |grad p| per frequency at point `first`, the
    gradient by central differences over the next six points."""
    f = np.asarray(field[f"pressure_{key}"])[:, first:first + 7].astype(complex)
    grad = np.stack([(f[:, 1 + 2 * d] - f[:, 2 + 2 * d]) / (2 * h) for d in range(3)], axis=-1)
    ref = (1j * G / omegas)[:, None, None] * grad
    v = np.asarray(field[f"velocity_{key}"])[:, first].astype(complex)
    return np.abs(v - ref).max(axis=(1, 2)) / np.abs(ref).max(axis=(1, 2))


def stencil(x0, h):
    x0 = np.asarray(x0, dtype=float)
    return np.array([x0] + [x0 + s * h * e for e in np.eye(3) for s in (1, -1)])


def test_source_velocity_with_an_interior_surface():
    # the sources on the interior free surface (d phi/dz = 0 inside) remove the
    # irregular frequency of the source formulation as of Green's identity: at the
    # first irregular frequency of the cylinder the velocity without the interior
    # surface is wrong (0.78 observed), with it equal to the pressure gradient to the discretization
    # of the constant panels, 16 a quadrant (0.033 observed at the irregular frequency, 0.015 at the
    # regular one)
    mesh = surface(cylinder_panels(1.0, 0.5, n_phi=16, n_z=8, n_r=8, quarter=True), symmetry="xz_and_yz")
    w = 2 * math.pi / np.array([2.006403, 1.1818])  # regular, the first irregular frequency (m = 0)
    h = 1e-3
    pts = stencil([1.3, 0.4, -0.15], h)
    err = {}
    for irr in (False, True):
        body = {"mesh": mesh, "modes": [], **({"interior_surface": {"method": "graded"}} if irr else {})}
        r = model(body, w, [0.0], field_points=pts, options={"field_pressure": True, "field_velocity": True}).solve()
        err[irr] = gradient_error(group(r, "field"), r.omegas, h)
    assert err[False][1] > 0.3 and err[True][1] < 0.06, err
    assert err[True][0] < 0.02 and err[False][0] < 0.02, err


def test_source_velocity_with_a_damping_lid():
    # lid rows 4 pi sigma + (K - kappa) phi_S = (kappa - K) phi_I; the velocity
    # below the lid of a moonpool and outside equals the gradient of Green's identity to
    # the discretization, as without the lid (below the pumping resonance, K a ~ 0.6, where
    # the flow inside the moonpool is not small; the scattered potential with the lid
    # condition of the incident wave omitted gave 0.12 and 0.28 inside). The bounds are the
    # discretization of the constant panels, 12 a quadrant: 0.08 below the lid, near the moonpool
    # walls and their bottom edge (0.061 observed), 0.02 outside (0.013)

    moon = surface(moonpool_panels(n=12), symmetry="xz_and_yz")
    lid = surface(moonpool_lid_panels(n=12), symmetry="xz_and_yz")
    w = np.sqrt(np.array([0.2, 0.4]) * G)
    h = 1e-3
    pts = np.concatenate([stencil([0.3, 0.2, -0.3], h), stencil([3.0, 1.0, -0.4], h)])
    opts = {"diffraction_formulation": "total", "field_pressure": True, "field_velocity": True}
    res = {}
    for eps in (None, 0.5):
        lids = [] if eps is None else [{"name": "m", "mesh": lid, "damping_factor": eps}]
        res[eps] = model({"mesh": moon, "modes": []}, w, [0.0], field_points=pts, lids=lids, options=opts).solve()
    for eps, r in res.items():
        for first, bound in ((0, 0.08), (7, 0.02)):  # below the lid, outside
            e = gradient_error(group(r, "field"), r.omegas, h, first)
            assert e.max() < bound, (eps, first, e)
    v0, v1 = res[None].field_points.velocity_diffraction[:, 0], res[0.5].field_points.velocity_diffraction[:, 0]
    assert np.abs(v1 - v0).max() > 0.05 * np.abs(v0).max()  # the lid changes the moonpool flow


def test_source_velocity_with_pressure_surfaces():
    # L phi = -u on pressure surfaces: in z = 0 the sources are -u / 4 pi, so that a zero-pressure
    # surface there is the free surface: the same diffraction velocity as without it (an identity
    # of the problem; 1e-6, the float32 panel integrals). The velocity of the air-cushion vessel
    # against the pressure gradient converges slowly (the uniform pressure mode is discontinuous at
    # the skirt) and is validation

    hull, lid = moonpool_panels(n=12), moonpool_lid_panels(n=12)[:, [0, 3, 2, 1]]
    m = surface(np.concatenate([hull, lid]), symmetry="xz_and_yz")
    surf = {"mesh": part(m, len(hull) + 1, len(hull) + len(lid)), "pressure_modes": [{"name": "p", "shape": "uniform"}]}
    h = 1e-3
    pts = np.concatenate([stencil([0.3, 0.2, -0.3], h), stencil([3.0, 1.0, -0.4], h)])
    opts = {"diffraction_formulation": "total", "field_pressure": True, "field_velocity": True}
    w = np.sqrt(np.array([0.2, 0.4]) * G)
    a = model({"mesh": surface(hull, symmetry="xz_and_yz")}, w, [0.0], field_points=pts, options=opts).solve()
    b = model({"mesh": part(m, 1, len(hull)), "pressure_surfaces": [surf]}, w, [0.0], field_points=pts,
              options=opts).solve()  # fmt: skip
    va, vb = a.field_points.velocity_diffraction, b.field_points.velocity_diffraction
    assert np.abs(va - vb).max() <= 1e-6 * np.abs(va).max()


def test_source_velocity_in_a_channel():
    # the channel gradients (near images at the reflected points, series images,
    # propagating tail): the velocity equals the pressure gradient to the discretization of the
    # constant panels of the box, 8 x 4 x 4 a quadrant (0.04; 0.027 observed)
    mesh = surface(box_panels(1.0, 0.5, 0.5, n=(8, 4, 4), quarter=True), symmetry="xz_and_yz")
    h = 1e-3
    pts = np.concatenate([stencil([2.0, 1.0, -0.3], h), stencil([0.5, 2.5, -0.2], h)])
    r = model({"mesh": mesh}, np.sqrt(np.array([0.5, 1.0, 2.0]) * G), [0.0], field_points=pts,
              environment={"depth": 3.0},
              options={"domain": "channel", "channel_width": 6.0, "field_pressure": True,
                       "field_velocity": True}).solve()  # fmt: skip
    for q in (0, 7):
        for key in ("diffraction", "radiation"):
            assert gradient_error(group(r, "field"), r.omegas, h, q, key).max() < 0.04, (q, key)


def test_higher_order_body_pressure_and_velocity():
    # at the (k + 1)^2 parametric points of every panel with their weights J du dv
    # the pressure integrates to the exciting force and the added mass (midpoint rule),
    # and the normal radiation velocity is i w n_j; a specified point at a default point
    # takes its values (the closest point is the point itself)

    opts = {"higher_order": True, "body_pressure": True, "body_velocity": True}
    w = W[:1]
    r = model({"mesh": spline_cylinder(n=8)}, w, [30.0], options=opts).solve()
    b = group(r, "body")
    n, area, pts = np.asarray(b["normals"]), np.asarray(b["areas"]), np.asarray(b["points"])
    nj = np.concatenate([n, np.cross(pts, n)], axis=1)
    rho = r.model.environment.density
    X = -rho * G * np.einsum("fq,qi,q->fi", b["pressure_diffraction"][:, :, 0], nj, area)
    assert np.abs(X - r.excitation.diffraction[:, 0]).max() <= 2e-3 * np.abs(r.excitation.diffraction).max()
    # A - i B / w = -rho int phi_j n_i dS (normals into the fluid), p / (rho g) = (w^2 / g) phi_j
    F = -rho * G / r.omegas[:, None, None] ** 2 * np.einsum("fqj,qi,q->fij", b["pressure_radiation"], nj, area)
    Z = r.added_mass - 1j * r.radiation_damping / r.omegas[:, None, None]
    assert np.abs(F - Z).max() <= 2e-3 * np.abs(Z).max()
    vn = np.einsum("fqjd,qd->fqj", b["velocity_radiation"], n)
    ref = 1j * r.omegas[:, None, None] * nj[None, :, :]
    assert np.abs(vn - ref).max() <= 1e-4 * np.abs(ref).max()
    sel = [3, 200, 517]
    q = model({"mesh": spline_cylinder(n=8), "body_points": pts[sel]}, w, [30.0], options=opts).solve()
    for key in ("pressure_diffraction", "pressure_radiation", "velocity_diffraction", "velocity_radiation"):
        x, y = q.quantity("body_points_" + key), b[key][:, sel]
        assert np.abs(x - y).max() <= 1e-6 * np.abs(y).max(), key


def test_low_order_body_points():
    # the inverse-distance mean of the nearest centroids; a point at a centroid
    # takes that panel's values; the rules (every body or none, a body output requested)
    body = cylinder()
    opts = {"body_pressure": True, "body_velocity": True}
    a = model(body, W, [30.0], options=opts).solve()
    pts = np.asarray(a.body_points.points)[[5, 40]]
    b = model({**body, "body_points": pts}, W, [30.0], options=opts).solve()
    for key in ("pressure_diffraction", "velocity_radiation"):
        x, y = b.quantity("body_points_" + key), a.quantity("body_points_" + key)[:, [5, 40]]
        assert np.abs(x - y).max() <= 1e-6 * np.abs(y).max(), key
    mid = 0.5 * (pts[0] + pts[1])
    c = model({**body, "body_points": [mid], "body_point_neighbors": 1}, W, [30.0], options=opts).solve()
    near = np.argmin(np.linalg.norm(np.asarray(a.body_points.points) - mid, axis=1))
    assert np.allclose(c.body_points.pressure_diffraction[:, 0], a.body_points.pressure_diffraction[:, near])
    with pytest.raises(ValueError, match="no body output"):
        model({**body, "body_points": pts}, W, [0.0]).route()
    two = [{**body, "name": "a", "body_points": pts}, {**body, "name": "b", "position": [5, 0, 0]}]
    with pytest.raises(ValueError, match="every body or for none"):
        model(two, W, [0.0], options=opts).route()
    m = model(body, W, [0.0])
    with pytest.raises(ValueError, match="body_point_neighbors"):
        m.bodies["body"].body_point_neighbors = 0
