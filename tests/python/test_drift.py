"""Mean drift force and moment by momentum conservation (theory §11.1, test-plan B).

The energy flux of the radiated far field equals the damping; for a fixed
body without dissipation the momentum drift equals its energy-conservation
form (optical theorem); a symmetric body in head seas has no sway or yaw
drift; a translated body has the same force and the transported yaw moment;
a free body held by a stiff spring has the fixed body's drift; the numeric file; the
pressure drift's symmetries, placements, walls, sectors and several bodies; the rule of the
waterline of warped panels; the chosen Kochin angle count; routes that are not available are
rejected. Generalized modes (theory §12.15): the rigid modes given as generalized modes have the
rigid body's pressure drift (fixed, and floating with the translations), and with the rotations
too it differs by the terms of the second-order rotation alone; the pressure drift of a cylinder
with flexible modes meets its momentum drift with the panels (the generalized motions in both: a
test of the generalized terms). The pressure drift of a rigid body against the momentum drift
with the mesh, the higher against the low order and the convergence in the Kochin angle count
are validation, not tests. The bounds of 1e-5 of the identities are the float32 panel integrals
and outputs summed in another order (theory §6.1), the drift quadratic in them. Meshes are
procedural (tests/python/meshes.py)."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import free_cylinder, lo_flexible_cylinder, lo_free_cylinder, part, spline_cylinder, surface
from meshes import box_panels, cylinder_panels
from problems import drift_constants as _constants
from problems import model, rigid_modes_as_generalized, written

G = 9.80665
W = np.sqrt(np.array([0.5, 1.0, 2.0]) * G)
HO = {"higher_order": True, "drift_momentum_conservation": True}


ENV = {}


def solve(body, w=W, heads=(0.0, 30.0), opts=HO, env=ENV):
    return model(body, w, list(heads), environment=env, options=opts).solve()


def b2(r, f, values):
    """|b|^2 of Kochin values (theory §11.1)."""
    k, C, _ = _constants(float(r.omegas[f]), G, r.model.environment.depth)
    return (C / (4 * math.pi)) ** 2 * 2 / (math.pi * k) * np.abs(values) ** 2


@pytest.mark.parametrize("depth", [math.inf, 3.0])
def test_energy_flux_equals_damping(depth):
    # the energy balance (Newman 1977): the flux of the radiated far field is the damping. A moving
    # body: its drift needs the motions (outputs.motion); here the radiated far field is tested. The
    # Kochin function and the damping are two integrals of one higher-order solution with different
    # rules (the far-field kernel at the inner Gauss points against the pressure on the hull), the
    # angle sum spectrally exact: 1e-3 surge and heave (1.5e-4, 2.7e-4 observed), 3e-3 pitch (1.3e-3:
    # its moment arm weights the bottom edge, where the potential is singular)
    body = {"mesh": spline_cylinder(n=8), "mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.2],
            "radii_of_gyration": [0.6, 0.6, 0.7]}
    r = solve(body, heads=(0.0,), env={"depth": depth}, opts={**HO, "motion": True})
    hr = r.kochin.radiation
    n = hr.shape[1]
    for f, w in enumerate(r.omegas):
        k, _, Ke = _constants(float(w), G, depth)
        for j, tol in ((0, 1e-3), (2, 1e-3), (4, 3e-3)):
            flux = 1000.0 * w * Ke / (2 * k) * np.sum(b2(r, f, hr[f, :, j])) * 2 * math.pi / n
            B = r.radiation_damping[f, j, j]
            assert abs(flux - B) <= tol * abs(B), (f, j)


def test_fixed_body_drift_is_the_optical_theorem_form():
    # the identity of theory §11.1 for a fixed body without dissipation, evaluated here on the Kochin
    # samples: 1e-5, the Kochin samples returned in float32 (complex64) against the engine's own, the
    # angle sum cancelling more as the frequency rises (7e-8 to 5e-6 observed)
    r = solve({"mesh": spline_cylinder(n=8), "modes": []})
    th = r.kochin.angles
    hd = r.kochin.diffraction
    for f, w in enumerate(r.omegas):
        _, _, Ke = _constants(float(w), G, math.inf)
        for h, beta in enumerate(np.radians(r.headings)):
            e = b2(r, f, hd[f, :, h]) * 2 * math.pi / th.size
            F = 1000.0 * Ke / 4 * np.array([np.sum(e * (math.cos(beta) - np.cos(th))),
                                            np.sum(e * (math.sin(beta) - np.sin(th)))])  # fmt: skip
            assert np.abs(r.drift.momentum_conservation.real[f, h, :2] - F).max() <= 1e-5 * np.abs(F).max()
    # a (nearly) axisymmetric body: the drift along the waves, no yaw moment
    d = r.drift.momentum_conservation.real
    assert np.abs(d[:, 0, 1:]).max() <= 1e-6 * np.abs(d[:, 0, 0]).max()
    assert np.allclose(d[:, 1, 1] / d[:, 1, 0], math.tan(math.radians(30.0)), rtol=1e-4)  # B-spline circle


def test_head_seas_and_translation():
    # the symmetry of the problem: no sway or yaw drift in head seas (1e-6; 6e-16 observed, the
    # mirrored Kochin samples equal); the translated body (1e-5, float32: 7e-8 observed)
    mesh = surface(box_panels(1.0, 0.5, 0.5, n=(8, 4, 4), quarter=True), symmetry="xz_and_yz")
    opts = {"drift_momentum_conservation": True}
    head = solve({"mesh": mesh, "modes": []}, heads=(0.0,), opts=opts)
    dh = head.drift.momentum_conservation.real
    assert np.abs(dh[:, 0, 1:]).max() <= 1e-6 * np.abs(dh[:, 0, 0]).max()
    a = solve({"mesh": mesh, "yaw": 20.0, "modes": []}, heads=(50.0,), opts=opts)
    x0, y0 = 3.0, -2.0
    b = solve({"mesh": mesh, "yaw": 20.0, "position": [x0, y0, 0.0], "modes": []}, heads=(50.0,), opts=opts)
    da, db = a.drift.momentum_conservation.real[:, 0], b.drift.momentum_conservation.real[:, 0]
    scale = np.abs(da[:, :2]).max()
    assert np.abs(da[:, :2] - db[:, :2]).max() <= 1e-5 * scale
    moved = da[:, 2] + x0 * da[:, 1] - y0 * da[:, 0]
    assert np.abs(db[:, 2] - moved).max() <= 1e-5 * scale * (abs(x0) + abs(y0))


def test_drift_follows_the_motions(tmp_path):
    # the free body held by a stiff spring (motions about 1e-8 of the free body's) has the fixed
    # body's drift: the radiated part of the far field vanishes with the motions (4e-9 observed);
    # the free body's drift is another (93 % of the peak here)
    mesh = surface(cylinder_panels(1.0, 0.5, n_phi=8, n_z=4, n_r=4, quarter=True), symmetry="xz_and_yz")
    m = 1000.0 * math.pi * 0.5  # the displaced mass
    free = {"mesh": mesh, "center_of_gravity": [0.0, 0.0, -0.2], "mass": m, "inertia_input": "components",
            "Ixx": m * 0.36, "Ixy": 0.0, "Ixz": 0.0,
            "Iyy": m * 0.36, "Iyz": 0.0, "Izz": m * 0.49, "inertia_origin": "center_of_gravity"}  # fmt: skip
    opts = {"drift_momentum_conservation": True, "motion": True}
    r = solve(free, opts=opts)
    held = solve({**free, "additional_stiffness": (1e12 * np.eye(6)).tolist()}, opts=opts)
    fixed = solve({**free, "modes": []}, opts={"drift_momentum_conservation": True})
    d, d0 = r.drift.momentum_conservation.real, fixed.drift.momentum_conservation.real
    assert np.abs(held.motion).max() < 1e-6
    assert np.abs(held.drift.momentum_conservation.real - d0).max() <= 1e-6 * np.abs(d0).max()
    assert np.abs(d - d0).max() > 0.1 * np.abs(d0).max()
    # the numeric file
    r.write_wamit(tmp_path / "drift")
    paths = written(tmp_path / "drift")
    rows = [ln.split() for ln in (tmp_path / "drift.8").read_text().splitlines()[1:]]
    assert len(rows) == 3 * 2 * 3 and {int(x[3]) for x in rows} == {1, 2, 6}
    assert any(p.suffix == ".8" for p in paths)


def test_bidirectional_pairs():
    # the pairs beta_a <= beta_b in heading order; a pair of equal headings is the unidirectional
    # drift (real), a pair of two headings is complex
    mesh = surface(box_panels(1.0, 0.5, 0.5, n=(8, 4, 4), quarter=True), symmetry="xz_and_yz")
    body = {"mesh": mesh, "yaw": 20.0, "position": [0.5, 0.3, 0.0], "modes": []}
    heads = (0.0, 50.0, 120.0)
    r = solve(body, heads=heads, opts={"drift_momentum_conservation": True, "drift_bidirectional": True})
    uni = solve(body, heads=heads, opts={"drift_momentum_conservation": True})
    pairs = [(a, b) for a in range(3) for b in range(a, 3)]
    np.testing.assert_array_equal(r.drift.heading_pairs, [[heads[a], heads[b]] for a, b in pairs])
    np.testing.assert_array_equal(uni.drift.heading_pairs, [[h, h] for h in heads])
    d = r.drift.momentum_conservation  # [F, P, 3]
    assert d.shape == (len(W), len(pairs), 3) and d.dtype == np.complex128
    diag = [pairs.index((h, h)) for h in range(3)]
    np.testing.assert_allclose(d[:, diag].real, uni.drift.momentum_conservation.real, rtol=1e-12)
    assert np.abs(d[:, diag].imag).max() <= 1e-9 * np.abs(d).max()
    assert np.abs(d[:, pairs.index((0, 1))].imag).max() >= 1e-3 * np.abs(d).max()


def test_unavailable_drift_routes_are_rejected():
    mesh = surface(cylinder_panels(1.0, 0.5, n_phi=4, n_z=2, n_r=2, quarter=True), symmetry="xz_and_yz")
    with pytest.raises(hb.UnsupportedProblemError, match="walls"):
        model({"mesh": mesh, "position": [3.0, 0.0, 0.0], "modes": []}, [1.0], [0.0],
              options={"drift_momentum_conservation": True, "domain": "x_wall"}).route()  # fmt: skip
    with pytest.raises(ValueError, match="diffraction"):
        model({"mesh": mesh, "modes": []}, [1.0], (),
              options={"drift_momentum_conservation": True, "diffraction": False}).route()  # fmt: skip


# ---------------------------------------------------------------- pressure integration (theory §11.2)


PD = {"higher_order": True, "drift_pressure_integration": True, "drift_momentum_conservation": True, "motion": True}


def test_pressure_drift_placement_and_symmetry():
    # identities of the problem: the mirror symmetry (1e-6; 3e-15 observed), the placement in body
    # axes (1e-5, float32: 6e-8 observed)
    a = solve(free_cylinder(), heads=(0.0, 30.0), opts=PD)
    d = a.drift.pressure_integration.real
    # head seas on a body symmetric about y = 0: no sway force, roll or yaw moment
    assert np.abs(d[:, 0, [1, 3, 5]]).max() <= 1e-6 * np.abs(d[:, 0]).max()
    # body axes: a translated body, or a yawed one in waves turned with it, has the same drift
    b = solve(free_cylinder(position=[3.0, -2.0, 0.0]), heads=(0.0, 30.0), opts=PD)
    c = solve(free_cylinder(yaw=40.0), heads=(40.0, 70.0), opts=PD)
    for other in (b, c):
        assert np.abs(other.drift.pressure_integration.real - d).max() <= 1e-5 * np.abs(d).max()
    # the moment about the moving origin differs by <xi x F1> only
    assert a.drift.pressure_integration_body_axes.shape == d[:, :, 3:].shape


@pytest.mark.parametrize("higher_order", [False, True])
def test_pressure_drift_beside_a_wall_turned_by_90_degrees(higher_order):
    # a free body beside the wall x = 0 and the same problem turned by 90 deg
    # (the wall y = 0, the body and the heading turned) have the same drift in body axes; 1e-5: the
    # float32 panel integrals of the turned panels and images
    if higher_order:
        def body(**kw):
            return free_cylinder(**kw)
    else:
        mesh = surface(box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True), symmetry="xz_and_yz")

        def body(**kw):
            return {"mesh": mesh, "mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.1],
                    "radii_of_gyration": [0.4, 0.6, 0.7],
                    "inertia_origin": "center_of_gravity", **kw}  # fmt: skip
    opts = {**PD, "higher_order": higher_order, "drift_pressure_integration": True,
            "drift_momentum_conservation": False}  # fmt: skip
    env = {"gravity": G, "depth": 3.0}
    a = solve(body(position=[2.5, 0.3, 0.0], yaw=20.0), heads=(30.0,), opts={**opts, "domain": "x_wall"}, env=env)
    b = solve(body(position=[-0.3, 2.5, 0.0], yaw=110.0), heads=(120.0,), opts={**opts, "domain": "y_wall"}, env=env)
    for key in ("drift_pressure_integration", "drift_pressure_integration_body_axes"):
        x, y = a.quantity(key), b.quantity(key)
        assert np.abs(x - y).max() <= 1e-5 * np.abs(y).max(), key


def test_pressure_drift_file(tmp_path):
    r = solve(free_cylinder(), heads=(0.0,), opts=PD)
    assert r.drift.pressure_integration is not None
    r.write_wamit(tmp_path / "pd")
    rows = [ln.split() for ln in (tmp_path / "pd.9").read_text().splitlines()[1:]]
    # the six modes about O, then -4, -5, -6: the moment about the moving origin (reference/file-formats §2)
    assert {int(x[3]) for x in rows} == {1, 2, 3, 4, 5, 6, -4, -5, -6}


def test_unavailable_pressure_drift_routes_are_rejected():
    body = free_cylinder(position=[3.0, 3.0, 0.0])
    opts = {**PD, "domain": "orthogonal_walls"}
    with pytest.raises(hb.UnsupportedProblemError, match="walls"):
        model(body, [1.0], [0.0], options=opts).route()


LO_PD = {"drift_pressure_integration": True, "drift_momentum_conservation": True, "motion": True}


def test_low_order_pressure_drift_sectors_and_placement():
    # the source formulation per character: symmetry sectors equal the full solve (1e-5, the float32
    # panel integrals summed in another order: 1.3e-6 observed); the mirror symmetry (1e-6); the
    # placement (1e-5: 1.6e-7 observed)
    a = solve(lo_free_cylinder(), heads=(0.0, 30.0), opts=LO_PD)
    b = solve(lo_free_cylinder(quarter=False), heads=(0.0, 30.0), opts=LO_PD)
    d = a.drift.pressure_integration.real
    assert np.abs(b.drift.pressure_integration.real - d).max() <= 1e-5 * np.abs(d).max()
    assert np.abs(d[:, 0, [1, 3, 5]]).max() <= 1e-6 * np.abs(d[:, 0]).max()
    c = solve(lo_free_cylinder(position=[2.0, 1.0, 0.0], yaw=25.0), heads=(25.0, 55.0), opts=LO_PD)
    assert np.abs(c.drift.pressure_integration.real - d).max() <= 1e-5 * np.abs(d).max()


def test_low_order_waterline_of_warped_panels():
    # the input rule (reference/inputs-meshes §1): the waterline edges are found on the vertices as given, so that
    # the projection of a warped panel with a flare, which moves its top vertices off z = 0, does not
    # drop the waterline term. The waterline row flares out by 15 %; twisting its top vertices about
    # the axis (shared vertices alike, the symmetry planes fixed) warps the panels. A guard, not an
    # identity: the bound 0.1 lies between the change of the hull (4 %) and the missing term (160 %)
    def hull(twist):
        v = cylinder_panels(1.0, 0.5, n_phi=8, n_z=4, n_r=2, quarter=True)
        for q in v:
            for x in q:
                if x[2] == 0.0:
                    phi = math.atan2(x[1], x[0])
                    phi += twist * math.sin(2.0 * phi)
                    x[:2] = 1.15 * np.array([math.cos(phi), math.sin(phi)])
        return v

    d = [solve({"mesh": surface(hull(t), symmetry="xz_and_yz"), "modes": []}, heads=(0.0,),
               opts=LO_PD).drift.pressure_integration.real for t in (0.0, 0.05)]  # fmt: skip
    assert np.abs(d[1][:, 0, 0] - d[0][:, 0, 0]).max() <= 0.1 * np.abs(d[0][:, 0, 0]).max()


def test_low_order_pressure_drift_routes():
    mesh = surface(cylinder_panels(1.0, 0.5, n_phi=4, n_z=2, n_r=2, quarter=True), symmetry="xz_and_yz")
    body = {"mesh": mesh, "modes": [], "interior_surface": {"method": "waterline"}}
    # interior surfaces are source panels in z = 0, a channel is available
    model(body, [1.0], [0.0], options={"drift_pressure_integration": True}).route()
    model({"mesh": mesh, "modes": []}, [1.0], [0.0], environment={"depth": 3.0},
          options={"drift_pressure_integration": True, "domain": "channel",
                   "channel_width": 10.0}).route()  # fmt: skip
    fin = [[[1.0, 0.2, -0.3], [1.5, 0.2, -0.3], [1.5, 0.2, -0.1], [1.0, 0.2, -0.1]]]
    sheet = np.concatenate([cylinder_panels(1.0, 0.5, n_phi=4, n_z=2, n_r=2, quarter=True), fin])
    m2 = surface(sheet, symmetry="xz_and_yz")
    with pytest.raises(hb.UnsupportedProblemError, match="sheets"):
        fin_sheet = [{"mesh": part(m2, len(sheet), len(sheet))}]
        model({"mesh": part(m2, 1, len(sheet) - 1), "modes": [], "dipoles": fin_sheet}, [1.0], [0.0],
              options={"drift_pressure_integration": True}).route()  # fmt: skip


@pytest.mark.parametrize("higher_order", [False, True])
def test_pressure_drift_of_several_bodies(higher_order):
    # per body in its axes; swapping the bodies swaps the blocks (1e-5: the float32 panel integrals
    # in another order, 1.1e-6 observed in the low order; 2e-15 in the higher)
    make, opts = (free_cylinder, PD) if higher_order else (lo_free_cylinder, LO_PD)
    a = make(name="a", position=[3.0, 0.5, 0.0])
    b = make(name="b", position=[-2.0, 0.0, 0.0], yaw=30.0)
    r1 = solve([a, b], heads=(20.0,), opts=opts)
    r2 = solve([b, a], heads=(20.0,), opts=opts)
    d1, d2 = r1.drift.pressure_integration.real, r2.drift.pressure_integration.real
    assert d1.shape[2] == 12
    scale = np.abs(d1).max()
    assert np.abs(d1[..., :6] - d2[..., 6:]).max() <= 1e-5 * scale
    assert np.abs(d1[..., 6:] - d2[..., :6]).max() <= 1e-5 * scale
    assert r1.drift.pressure_integration_body_axes.shape[2] == 6


WK = np.array([0.8, 1.4, 2.2])


def test_the_engine_chooses_the_kochin_angles():
    body = lo_free_cylinder(position=[3.0, 4.0, 0.0])
    r = model(body, WK, [0.0], options={"motion": True, "drift_momentum_conservation": True}).solve()
    # the auto count of theory §11.1 and theory §11.4, 8 ceil((2 ceil(k R) + 96) / 8) with R the radius
    # of the body about the origin (5 + 1)
    k = hb.wavenumber(WK.max())
    assert r.kochin.angles.size == 8 * math.ceil((2 * math.ceil(k * 6.0) + 96) / 8)
    d = r.drift.momentum_conservation  # [F, P, 3] of the pair (0, 0): real
    assert d.shape == (3, 1, 3) and d.dtype == np.complex128
    assert np.array_equal(r.drift.heading_pairs, [[0.0, 0.0]]) and np.abs(d.imag).max() <= 1e-12 * np.abs(d).max()


# ---------------------------------------------------------------- generalized modes (theory §12.15)


GEN = {"drift_pressure_integration": True, "motion": True}


@pytest.mark.parametrize("case", ["fixed", "translations"])
def test_rigid_modes_as_generalized_modes_have_the_rigid_pressure_drift(case):
    # the same equations (the generalized coordinates with the rigid body's mass and restoring), the
    # same displacement and normal: the six rigid rows of both bodies, and the generalized rows
    # against the rigid rows of their modes (R = O: a rotation's row the moment about O). Fixed: the
    # rigid body without modes against the generalized one held by 1e12 (its motions 1e-9: 9e-9
    # observed); floating with the translations as generalized modes and the rotations rigid: the
    # same numbers (2e-16 observed: the first-order solves agree to the bit, the drift sums the
    # displacement in another order)
    which = range(6) if case == "fixed" else (0, 1, 2)
    rigid = lo_free_cylinder(modes=[]) if case == "fixed" else lo_free_cylinder()
    a = solve(rigid, heads=(0.0, 30.0), opts=GEN)
    b = solve(rigid_modes_as_generalized(lo_free_cylinder(), which, fixed=case == "fixed"), heads=(0.0, 30.0), opts=GEN)
    tol = 1e-7 if case == "fixed" else 1e-12
    d, scale = a.drift.pressure_integration, np.abs(a.drift.pressure_integration).max()
    g = b.drift.pressure_integration  # on the coordinates: the six rigid rows, then the generalized ones
    assert g.shape == (len(W), 2, 6 + len(which)) and np.all(np.isfinite(g))
    assert np.abs(g[..., :6] - d).max() <= tol * scale
    for k, c in enumerate(which):
        assert np.abs(g[..., 6 + k] - d[..., c]).max() <= tol * scale


def _hydrostatic_rotation_terms(r, body_panels, rho, g):
    """The drift of the rigid rotations given as generalized modes less the rigid body's
    (theory §12.15): the linear map x + α × x has the area element cof(A) n = (α αᵀ) n and no
    second-order displacement, the rotation H n and H x (theory §11.2), so with P0 = −ρ g z and
    P = <α αᵀ>: ΔF = ρg ∫ [z (P − H) n − (H x)_z n] dS, ΔM = ρg ∫ [z (x × P n + P (x × n) − H (x × n))
    − (H x)_z (x × n)] dS, the same first-order terms and waterline on both sides."""
    p = body_panels
    x, n, a = p.centroids, p.normals, p.areas
    out = np.zeros((len(r.omegas), len(r.headings), 6))
    for f in range(len(r.omegas)):
        for h in range(len(r.headings)):
            al = r.motion[f, h, 3:6].astype(complex)
            mean = 0.5 * np.real(np.outer(al, np.conj(al)))  # <α_i α_k>
            hm = np.array([[-0.5 * (mean[1, 1] + mean[2, 2]), 0, 0],
                           [mean[0, 1], -0.5 * (mean[0, 0] + mean[2, 2]), 0],
                           [mean[0, 2], mean[1, 2], -0.5 * (mean[0, 0] + mean[1, 1])]])  # fmt: skip
            hx = x @ hm.T
            xn = np.cross(x, n)
            z = x[:, 2:3]
            df = z * (n @ (mean - hm).T) - hx[:, 2:3] * n
            dm = z * (np.cross(x, n @ mean.T) + xn @ mean.T - xn @ hm.T) - hx[:, 2:3] * xn
            out[f, h] = rho * g * np.concatenate([(df * a[:, None]).sum(0), (dm * a[:, None]).sum(0)])
    return out


def test_rigid_rotations_as_generalized_modes_differ_by_the_second_order_rotation():
    # every rigid mode a generalized mode, floating: the same first-order motions (exact), the
    # linear map of the rotations stretches the body where the rigid rotation turns it, so the
    # rigid rows differ by the hydrostatic terms of H and of the stretched area element alone, here
    # computed from the panels (1e-12 of the drift, 2e-15 observed: the double sums; the heave
    # drift apart by 1.6 times the largest drift, the horizontal force the same, 1e-15); the
    # rotations' generalized rows are the work against the mode shape at the material point, not
    # the moment of the displaced point (definitions apart)
    env = {"density": 1000.0, "gravity": G}
    a = solve(lo_free_cylinder(), heads=(0.0, 30.0), opts=GEN, env=env)
    gb = rigid_modes_as_generalized(lo_free_cylinder(), range(6), env)
    b = solve(gb, heads=(0.0, 30.0), opts=GEN, env=env)
    assert np.abs(b.motion[..., 6:] - a.motion).max() <= 1e-6 * np.abs(a.motion).max()
    d = a.drift.pressure_integration.real
    scale = np.abs(d).max()
    m = model(lo_free_cylinder(), W, [0.0, 30.0], environment=env)
    expect = _hydrostatic_rotation_terms(a, m.bodies[0].panels(role="wetted"), 1000.0, G)
    diff = b.drift.pressure_integration.real[..., :6] - d  # the rigid rows (the generalized ones after them)
    assert np.abs(diff - expect).max() <= 1e-12 * scale, np.abs(diff - expect).max() / scale
    assert np.abs(diff[..., :2]).max() <= 1e-12 * scale and np.abs(diff[..., 2]).max() > 0.1 * scale


def test_flexible_modes_pressure_drift_against_the_momentum_drift():
    # a floating cylinder with a bulging bottom and bending sides (their motions 1.3 to 1.4 m/m):
    # the momentum drift holds the generalized motions in its far field (theory §11.1), the pressure
    # drift in its terms; with the panels the two meet as for a rigid body (surge and sway, of the
    # largest: 24 %, 11 %, 5 % on 8, 16, 32 panels a quadrant; the rigid cylinder 14 % and 8 % on 16
    # and 32), while the flexible modes change the surge drift by up to 57 % (1.8 rad/s: 886 N/m^2
    # rigid, 385 flexible by the momentum drift); the generalized rows converge too (1 % from 16 to
    # 32)
    w = np.array([1.8, 2.4, 3.0, 3.6, 4.2])
    opts = {"drift_pressure_integration": True, "drift_momentum_conservation": True, "motion": True}
    env = {"density": 1000.0, "gravity": G}
    err = []
    for n in (8, 16):
        r = solve(lo_flexible_cylinder(n), w=w, heads=(0.0, 30.0), opts=opts, env=env)
        pi, mc = r.drift.pressure_integration.real, r.drift.momentum_conservation.real
        err.append(np.abs(pi[..., :2] - mc[..., :2]).max() / np.abs(mc[..., :2]).max())
    assert err[0] < 0.3 and err[1] < 0.6 * err[0], err
    rigid = solve(lo_flexible_cylinder(16, flexible=False), w=w, heads=(0.0, 30.0), opts=opts, env=env)
    mr = rigid.drift.momentum_conservation.real
    assert (np.abs(mr[..., 0] - mc[..., 0]) / np.abs(mr[..., 0])).max() > 0.4
    g = r.drift.pressure_integration  # the rigid rows, then the generalized ones
    assert g.shape[-1] == 8 and np.all(np.isfinite(g))
    assert np.abs(g[..., 6:].imag).max() <= 1e-12 * np.abs(g[..., 6:]).max()  # equal headings: real
