"""The low-order method in infinite depth and open sea; motion matrices, hydrostatics, threads.

A4 (Hulme 1982), A7, A8 and the identities of test-plan B that apply to this route, each on one
mesh: the identities hold exactly in the continuous problem, and each bound states what the
discretization (O(h^2) of constant panels) or the float32 storage of the solution leaves. The
convergence of the identities with the mesh and the comparison of the centroid wave integration
with the logarithmic one are validation, run when the method changes, not tests. The residual of
the equation of motion is in test_motions. Meshes are procedural (tests/python/meshes.py)."""

import math

import numpy as np
import pytest

from bodies import hemi, surface
from meshes import box_panels, cylinder_panels, hemisphere_panels
from problems import hulme, hydrostatics, model, peak, view, written

RHO, G = 1000.0, 9.80665


def matrices(m):
    """(inertia, stiffness, damping) of the motion equation of a model."""
    e = m.motion_equation()
    return e.inertia, e.stiffness, e.damping


def solve(bodies, omegas, headings=None, **kw):
    return model(bodies, omegas, headings, environment={"density": RHO, "gravity": G}, **kw).solve()


@pytest.fixture(scope="module")
def hemi_results():
    # 24 panels a quadrant edge, Ka = 0.4, 1.0, 1.8, headings every 15 degrees
    w = np.sqrt(np.array([0.4, 1.0, 1.8]) * G)
    return solve({"mesh": hemi(24)}, w, headings=np.arange(0.0, 360.0, 15.0))


# ------------------------------------------------------------- route


def test_route_by_depth_and_domain():
    # input rule: the depth and the domain select the route
    m = hemi(4)
    assert model({"mesh": m}, [1.0], [0.0]).route() == "low_order_infinite_open_sea"
    p = model({"mesh": m}, [1.0], [0.0], environment={"depth": 10.0})
    assert p.route() == "low_order_finite_open_sea"
    wall = model({"mesh": m, "position": [5, 0, 0]}, [1.0], [0.0], options={"domain": "x_wall"})
    assert wall.route() == "low_order_infinite_x_wall"


# ------------------------------------------------------------- test-plan B identities


def test_symmetry_and_positive_damping(hemi_results):
    # Green's reciprocity: A and B symmetric; the radiated energy: B_jj >= 0 (exact identities). The
    # discrete collocation is not symmetric: 1.7e-6 at 10 panels a quadrant edge, 6e-8 here (the
    # float32 level); B_jj >= -1e-6 of the peak: the float32 round-off of the zero yaw damping
    r = hemi_results
    for f in range(len(r.omegas)):
        A, B = r.added_mass[f], r.radiation_damping[f]
        assert peak(A - A.T) <= 1e-5 * peak(A)
        assert peak(B - B.T) <= 1e-5 * peak(B)
        assert np.all(np.diag(B) >= -1e-6 * peak(B))
        assert np.all(np.diag(A)[[0, 1, 2]] > 0)


def haskind_error(r):
    """The largest difference of the Haskind and the pressure exciting force relative to its peak."""
    return peak(r.excitation.haskind - r.excitation.diffraction) / peak(r.excitation.diffraction)


def test_haskind_equals_pressure_excitation(hemi_results):
    # the Haskind relation (Newman 1977, exact): the force from the radiation potentials equals that
    # from the diffraction potential; the two are computed from different unknowns, so they differ
    # by the O(h^2) discretization: 3.6e-4 here (1.8e-3 at 10 panels a quadrant edge)
    assert haskind_error(hemi_results) <= 1e-3


def energy_error(r):
    # Newman (1977, eq. 6.173): B_jj = k / (8 pi rho g Cg) int_0^2pi |X_j(theta)|^2 dtheta, Cg = g / (2 w)
    worst = 0.0
    for f, w in enumerate(r.omegas):
        k = w * w / G
        x = np.abs(r.excitation.haskind[f]) ** 2  # [H, K]
        integral = x.mean(axis=0) * 2 * math.pi  # periodic trapezoid
        b = k / (8 * math.pi * RHO * G * (G / (2 * w))) * integral
        diag = np.diag(r.radiation_damping[f])[:5]  # yaw radiates no waves for a body of revolution
        worst = max(worst, np.abs(b[:5] - diag).max() / peak(r.radiation_damping[f]))
    return worst


def test_energy_relation(hemi_results):
    # the energy relation (Newman 1977, exact) from the Haskind force over the headings (the
    # periodic trapezoid of 24 headings integrates the cos^2, sin^2 and constants of a body of
    # revolution exactly): the O(h^2) discretization leaves 1.2e-3 here (4.7e-3 at 12 panels a
    # quadrant edge)
    assert energy_error(hemi_results) <= 2e-3


def flux_error(n):
    """The damping from the radiated power through circles of KR = 40 and 80 (deep water),
    B_jj = rho g Cg R / w^2 int |eta_j|^2 dtheta per unit displacement up to O(1/KR), extrapolated
    in 1/KR (Richardson), against the computed B of surge, sway and heave of the hemisphere of n
    panels a quadrant edge at Ka = 1; one solve, the points of both circles."""
    w = math.sqrt(1.0 * G)
    K = w * w / G
    th = np.radians(np.arange(0.0, 360.0, 10.0))
    radii = (40.0 / K, 80.0 / K)
    pts = np.concatenate([np.c_[R * np.cos(th), R * np.sin(th), 0 * th] for R in radii])
    r = solve({"mesh": hemi(n)}, [w], field_points=pts, options={"diffraction": False})
    eta = r.field_points.pressure_radiation[0].reshape(2, len(th), -1)
    c = RHO * G * (G / (2 * w)) / w**2 * 2 * math.pi
    est = [c * R * (np.abs(e) ** 2).mean(axis=0) for R, e in zip(radii, eta, strict=True)]
    B = np.diag(r.radiation_damping[0])
    return np.abs(2 * est[1][:3] - est[0][:3] - B[:3]).max() / B.max()


def test_field_point_energy_flux_equals_damping():
    # the energy relation through field points (exact up to O(1/KR), which the extrapolation in
    # 1/KR removes; the trapezoid over 36 angles exact for the cos^2, sin^2 and constants):
    # the O(h^2) discretization leaves 6.5e-4 at 24 panels a quadrant edge (2.2e-3 at 12)
    assert flux_error(24) <= 1e-3


def test_total_and_scattered_formulations_agree():
    # the total and the scattered potential solve one continuous problem (exact): they discretize
    # different unknowns and differ by O(h^2), 1.7e-3 at 16 panels a quadrant edge (6.8e-3 at 8); a
    # wrong term in either gives O(1). The radiation is the same solve: bit-identical
    w = np.sqrt(np.array([0.5, 1.5]) * G)
    fp = [(1.5, 0.0, 0.0), (0.0, 3.0, 0.0), (-6.0, 2.0, 0.0)]
    opts = {"diffraction_formulation": "total"}
    tot = solve({"mesh": hemi(16)}, w, headings=[0.0, 45.0], options=opts, field_points=fp)
    sca = solve({"mesh": hemi(16)}, w, headings=[0.0, 45.0], field_points=fp)
    assert peak(tot.excitation.diffraction - sca.excitation.diffraction) <= 2e-3 * peak(sca.excitation.diffraction)
    np.testing.assert_array_equal(tot.added_mass, sca.added_mass)
    # field elevations: the scattered field uses the total-potential identity
    ref = peak(sca.field_points.pressure_diffraction)
    assert peak(tot.field_points.pressure_diffraction - sca.field_points.pressure_diffraction) <= 2e-3 * ref
    assert peak(tot.field_points.pressure_scattering - sca.field_points.pressure_scattering) <= 2e-3 * ref


def test_symmetry_sectors_equal_full_solve():
    # the quarter mesh with the xy symmetry and its four mirror images as one mesh: the same
    # discrete problem (exact); the float32 storage and the sums in another order leave 2e-7
    w = np.sqrt(np.array([0.5, 1.5]) * G)
    quarter = hemisphere_panels(1.0, 6, 6)
    full = np.concatenate([quarter * [sx, sy, 1] for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1))])
    full[len(quarter) :] = np.array([q[[0, 3, 2, 1]] for q in full[len(quarter) :]])  # keep orientation
    full[len(quarter) * 3 :] = np.array([q[[0, 3, 2, 1]] for q in full[len(quarter) * 3 :]])
    rs = solve({"mesh": surface(quarter, symmetry="xz_and_yz")}, w, headings=[0.0, 30.0, 135.0])
    rf = solve({"mesh": surface(full)}, w, headings=[0.0, 30.0, 135.0])
    for name in ("added_mass", "radiation_damping", "excitation_diffraction"):
        a, b = view(rs, name), view(rf, name)
        assert peak(a - b) <= 1e-6 * peak(b), name


@pytest.mark.parametrize("threads", [(2, 1), (3, 1), (1, 3), (2, 2)])
def test_threads_byte_identical(threads):
    # the results do not depend on the thread counts (each frequency and each block is summed in a
    # fixed order): bit-identical
    w = np.sqrt(np.linspace(0.3, 2.0, 5) * G)
    body = {"mesh": hemi(6), "mass": "displaced", "center_of_gravity": [0, 0, -0.3],
            "inertia_input": "components", "Ixx": 300.0, "Ixy": 0.0,
            "Ixz": 0.0, "Iyy": 300.0,
            "Iyz": 0.0, "Izz": 400.0}  # fmt: skip
    kw = dict(headings=[0.0, 90.0], field_points=[[3.0, 0.0, 0.0]], options={"motion": True})
    r1 = solve(body, w, **kw)
    kw["options"] = {"motion": True, "frequency_threads": threads[0], "inner_threads": threads[1]}
    rn = solve(body, w, **kw)
    for name in ("added_mass", "radiation_damping", "excitation_diffraction", "motion"):
        assert view(r1, name).tobytes() == view(rn, name).tobytes(), name
    assert r1.field_points.pressure_diffraction.tobytes() == rn.field_points.pressure_diffraction.tobytes()


def test_partial_radiation_columns_are_subsets():
    # input rule: the radiation modes asked for are columns of the full solve (the same factorization,
    # each column its own right-hand side; 0 observed, 1e-6 for the float32 sums of a blocked solve)
    w = np.sqrt(np.array([0.7, 1.4]) * G)
    full = solve({"mesh": hemi(8)}, w)
    part = solve({"mesh": hemi(8), "modes": ["sway", "pitch"]}, w)
    assert part.radiation_mode_ids.tolist() == [2, 5]
    np.testing.assert_allclose(
        part.added_mass, full.added_mass[:, :, [1, 4]], rtol=0, atol=1e-6 * peak(full.added_mass)
    )


def test_placement_and_translation_invariance():
    m = surface(box_panels(n=(4, 2, 2)))
    w = np.sqrt(np.array([0.6, 1.2]) * G)
    heads = [0.0, 60.0]
    base = solve({"mesh": m}, w, headings=heads)
    p, yaw = (3.0, -2.0, 0.0), 40.0
    moved = solve({"mesh": m, "position": list(p), "yaw": yaw}, w, headings=heads)
    turned = solve({"mesh": m}, w, headings=[h - yaw for h in heads])
    # identity of the problem: radiation in body axes is unchanged, the excitation is the result at
    # the local heading times the phase of the incident wave at the new origin; the float32 storage
    # and the phase in another order leave 1.3e-7
    np.testing.assert_allclose(moved.added_mass, base.added_mass, rtol=0, atol=1e-5 * peak(base.added_mass))
    for f, wf in enumerate(w):
        k = wf * wf / G
        for h, beta in enumerate(np.radians(heads)):
            phase = np.exp(-1j * k * (p[0] * np.cos(beta) + p[1] * np.sin(beta)))
            np.testing.assert_allclose(
                moved.excitation.diffraction[f, h],
                turned.excitation.diffraction[f, h] * phase,
                rtol=0,
                atol=1e-5 * peak(base.excitation.diffraction),
            )


def test_two_bodies_permutation_and_translation():
    # identity of the problem: the bodies in the other order permute the coordinates, a common
    # translation multiplies the excitation by the phase of the incident wave; float32 leaves 2e-7
    m = surface(box_panels(n=(4, 2, 2)))
    w = np.sqrt(np.array([0.8]) * G)
    a = {"name": "a", "mesh": m, "position": [0.0, 0.0, 0.0]}
    b = {"name": "b", "mesh": m, "position": [0.5, 3.0, 0.0], "yaw": 20.0}
    r_ab = solve([a, b], w, headings=[0.0, 70.0])
    r_ba = solve([b, a], w, headings=[0.0, 70.0])
    perm = list(range(6, 12)) + list(range(6))
    np.testing.assert_allclose(
        r_ba.added_mass[0], r_ab.added_mass[0][np.ix_(perm, perm)], rtol=0, atol=1e-5 * peak(r_ab.added_mass)
    )
    np.testing.assert_allclose(
        r_ba.excitation.diffraction[0],
        r_ab.excitation.diffraction[0][:, perm],
        rtol=0,
        atol=1e-5 * peak(r_ab.excitation.diffraction),
    )
    d = np.array([2.0, -1.0, 0.0])
    a2 = {"name": "a", "mesh": m, "position": list(d)}
    b2 = {"name": "b", "mesh": m, "position": list(np.array(b["position"]) + d), "yaw": 20.0}
    r_t = solve([a2, b2], w, headings=[0.0, 70.0])
    np.testing.assert_allclose(r_t.added_mass, r_ab.added_mass, rtol=0, atol=1e-5 * peak(r_ab.added_mass))
    k = w[0] ** 2 / G
    for h, beta in enumerate(np.radians([0.0, 70.0])):
        ph = np.exp(-1j * k * (d[0] * np.cos(beta) + d[1] * np.sin(beta)))
        np.testing.assert_allclose(
            r_t.excitation.diffraction[0, h],
            r_ab.excitation.diffraction[0, h] * ph,
            rtol=0,
            atol=1e-5 * peak(r_ab.excitation.diffraction),
        )


def test_centroid_wave_option_is_used():
    # input rule: logarithmic_integration = False selects the centroid rule of the wave part (how
    # close it is to the logarithmic integration is validation)
    w = np.sqrt(np.array([1.0]) * G)
    lg = solve({"mesh": hemi(4)}, w)
    ct = solve({"mesh": hemi(4)}, w, options={"logarithmic_integration": False})
    assert peak(lg.added_mass - ct.added_mass) > 0


# ------------------------------------------------------------- A8: the matrices of the motions (theory §3.1)


def _inertia_body(m, **kw):
    return {"mesh": m, "mass": "displaced", "center_of_gravity": [0.1, -0.05, -0.2],
            "inertia_input": "components", "Ixx": 420.0, "Ixy": 12.0,
            "Ixz": -8.0, "Iyy": 390.0,
            "Iyz": 5.0, "Izz": 610.0, **kw}  # fmt: skip


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _transport(P, R):
    T = np.eye(6)
    T[:3, 3:] = -_skew(np.asarray(P) - np.asarray(R))
    return T


def test_A8_inertia_and_matrix_transport():
    # the parallel-axis theorem and K_R = T^T K_P T against the direct algebra (double precision:
    # 5e-15 observed, 1e-12 the round-off)
    m = hemi(4)
    R = np.array([0.2, 0.1, -0.1])
    Gc = np.array([0.1, -0.05, -0.2])
    O = np.array([0.4, 0.3, 0.0])
    J_O = np.array([[1420.0, 12.0, -8.0], [12.0, 1390.0, 5.0], [-8.0, 5.0, 1610.0]])
    mass = 2000.0
    P = np.array([1.0, -0.5, 0.2])
    K_P = np.arange(36.0).reshape(6, 6) + 7 * np.eye(6)
    D_P = np.arange(36.0).reshape(6, 6)[::-1] + np.eye(6)
    body = {
        "mesh": m,
        "rotation_center": list(R),
        "center_of_gravity": list(Gc),
        "mass": mass,
        "inertia_input": "components", "Ixx": J_O[0, 0],
        "Ixy": J_O[0, 1],
        "Ixz": J_O[0, 2],
        "Iyy": J_O[1, 1],
        "Iyz": J_O[1, 2],
        "Izz": J_O[2, 2],
        "inertia_origin": list(O),
        "additional_stiffness": K_P,
        "stiffness_origin": list(P),
        "additional_damping": D_P,
        "damping_origin": "center_of_gravity",
    }
    p = model(body, [1.0], [0.0], environment={"density": RHO, "gravity": G})
    M, C, B = matrices(p)
    c, a = Gc - R, Gc - O
    J_R = J_O - mass * (a @ a * np.eye(3) - np.outer(a, a)) + mass * (c @ c * np.eye(3) - np.outer(c, c))
    M_ref = np.block([[mass * np.eye(3), -mass * _skew(c)], [mass * _skew(c), J_R]])
    np.testing.assert_allclose(M, M_ref, rtol=0, atol=1e-12 * peak(M_ref))
    hst = hydrostatics(p)[0].stiffness
    T = _transport(P, R)
    np.testing.assert_allclose(C - hst, T.T @ K_P @ T, rtol=0, atol=1e-12 * peak(K_P))
    Tg = _transport(Gc, R)
    np.testing.assert_allclose(B, Tg.T @ D_P @ Tg, rtol=0, atol=1e-12 * peak(D_P))


def test_A8_coupling_matrix_columns_use_each_body_origin():
    # the transport of a coupling stiffness: rows about this body's origin, columns about the other
    # body's (direct algebra, double precision); the input rule of the other body's origin
    m = hemi(4)
    P1, P2 = np.array([0.3, 0.0, 0.0]), np.array([-0.2, 0.4, 0.1])
    K = np.arange(72.0).reshape(6, 12) / 10.0
    b1 = {"name": "a", "mesh": m, "additional_stiffness": K[:, :6], "stiffness_origin": list(P1),
          "couplings": [{"body": "b", "stiffness": K[:, 6:]}]}  # fmt: skip
    b2 = {"name": "b", "mesh": m, "position": [5.0, 0.0, 0.0], "stiffness_origin": list(P2),
          "rotation_center": [0.1, 0.0, 0.0]}  # fmt: skip
    p = model([b1, b2], [1.0], [0.0], environment={"density": RHO, "gravity": G})
    mats = p.motion_equation()
    assert not mats.ready and mats.inertia is None  # no mass properties: not ready
    # the stiffness is still assembled by the engine query used for .hst; check through the engine model directly
    _, stiffness, _, ready = p._doc.motion_equation()
    assert not ready
    hst = [h.stiffness for h in hydrostatics(p)]
    Cm = np.array(stiffness)
    Cm[:6, :6] -= hst[0]
    Cm[6:, 6:] -= hst[1]
    T1 = _transport(P1, np.zeros(3))
    T2 = _transport(P2, np.array([0.1, 0.0, 0.0]))
    np.testing.assert_allclose(Cm[:6, :6], T1.T @ K[:, :6] @ T1, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(Cm[:6, 6:], T1.T @ K[:, 6:] @ T2, rtol=1e-9, atol=1e-12)
    assert peak(Cm[6:, :]) == 0.0
    # the origin of the other body's columns is checked too: its centre of gravity, which it lacks
    p.bodies["b"].stiffness_origin = "center_of_gravity"
    with pytest.raises(ValueError, match=r"bodies/a/couplings/1 \(with 'b'\): the columns of its stiffness are "
                                         r"the motions of bodies/b about its stiffness_origin='center_of_gravity', "
                                         "which it lacks"):  # fmt: skip
        p.check()


# ------------------------------------------------------------- A7 hydrostatics


@pytest.mark.parametrize("n", [(2, 1, 1), (4, 2, 2), (6, 3, 1)])
def test_A7_box(n):
    # closed forms of the discrete hydrostatics (the centroid rule integrates y^2 to -A h^2 / 12):
    # exact, 1e-12 the double round-off
    a, b, d = 1.5, 0.75, 0.6
    m = surface(box_panels(a, b, d, n=n))
    R = (0.2, -0.1, -0.15)
    Gc = (0.05, 0.02, -0.25)
    p = model(
        {"mesh": m, "rotation_center": list(R), "mass": "displaced", "center_of_gravity": list(Gc),
         "position": [3, 4, 0]},
        [1.0],
        [0.0],
        environment={"density": RHO, "gravity": G},
    )
    h = hydrostatics(p)[0]
    Lx, Ly, V = 2 * a, 2 * b, 4 * a * b * d
    Aw = Lx * Ly
    hx, hy = Lx / n[0], Ly / n[1]
    x0, y0 = -R[0], -R[1]  # waterplane centre relative to R
    Ixx_disc = Aw * (Ly**2 / 12 - hy**2 / 12 + y0**2)  # int y^2 about R (centroid rule)
    Iyy_disc = Aw * (Lx**2 / 12 - hx**2 / 12 + x0**2)
    Ixy = Aw * x0 * y0
    zb_rel = -d / 2 - R[2]
    rg, mass = RHO * G, RHO * V
    zg = Gc[2] - R[2]
    # the centres and the waterplane centroid in body coordinates (the box about the body origin),
    # the stiffness about R
    for got, want in (
        (h.volumes, (V, V, V)),
        (h.volume, V),
        (h.waterplane_area, Aw),
        (h.waterplane_centroid, (0.0, 0.0)),
        (h.center_of_buoyancy, (0.0, 0.0, -d / 2)),
        (h.center_of_gravity, Gc),
        (h.metacentric_height, h.stiffness[[3, 4], [3, 4]] / (rg * V)),
    ):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    c = h.stiffness
    np.testing.assert_allclose(c[2, 2], rg * Aw, rtol=1e-12)
    np.testing.assert_allclose(c[2, 3], rg * Aw * y0, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(c[2, 4], -rg * Aw * x0, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(c[3, 3], rg * (Ixx_disc + V * zb_rel) - mass * G * zg, rtol=1e-12)
    np.testing.assert_allclose(c[4, 4], rg * (Iyy_disc + V * zb_rel) - mass * G * zg, rtol=1e-12)
    np.testing.assert_allclose(c[3, 4], -rg * Ixy, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(c[3, 5], -rg * V * x0 + mass * G * (Gc[0] - R[0]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(c[4, 5], -rg * V * y0 + mass * G * (Gc[1] - R[1]), rtol=1e-12, atol=1e-9)


def test_A7_polygonal_cylinder():
    # the volume and waterplane area of the polygonal cylinder: exact for flat panels (1e-12 round-off)
    nphi = 12
    m = surface(cylinder_panels(1.0, 0.5, n_phi=nphi, n_z=2, n_r=3, quarter=False))
    h = hydrostatics(model({"mesh": m}, [1.0], [0.0], environment={"density": RHO, "gravity": G}))[0]
    poly_area = 0.5 * nphi * math.sin(2 * math.pi / nphi)  # regular nphi-gon of unit radius
    np.testing.assert_allclose(h.volumes, (0.5 * poly_area,) * 3, rtol=1e-12)
    np.testing.assert_allclose(h.waterplane_area, poly_area, rtol=1e-12)


# ------------------------------------------------------------- A4: Hulme (1982)


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_A4_hemisphere_against_hulme():
    """Floating hemisphere a = 1 m, Ka = 0.5, 1, 2; heave A(0), B(0) and surge
    A(1), B(1) of Hulme (1982) tables 1 and 2 (tests/data/hulme1982_full.txt from
    scripts/extract_hulme.py); three refined meshes; each quantity within 1e-3
    of its largest published value at the finest mesh and converging, as test-plan A4 asks (the
    four-digit tables and the O(h^2) discretization)."""
    ref = hulme()
    ka = ref[:, 0]
    w = np.sqrt(ka * G)
    m0 = RHO * 2 / 3 * math.pi
    errs = []
    for n in (16, 32, 48):
        r = solve(
            {"mesh": hemi(n), "modes": ["surge", "heave"]},
            w,
            options={"diffraction": False},
        )
        A, B = r.added_mass, r.radiation_damping
        vals = np.c_[A[:, 2, 1] / m0, B[:, 2, 1] / (w * m0), A[:, 0, 0] / m0, B[:, 0, 0] / (w * m0)]
        errs.append(np.abs(vals - ref[:, 1:]).max(axis=0) / np.abs(ref[:, 1:]).max(axis=0))
    errs = np.array(errs)
    assert np.all(errs[1] < errs[0]) and np.all(errs[2] < errs[1])
    assert np.all(errs[2] <= 1e-3), errs[2]


def test_write_files_from_a_solve(tmp_path):
    # input rule: a solve with motions and field points gives every file of the writers (their layout
    # is in test_writers, on results made up for them)
    body = _inertia_body(hemi(6))
    w = np.sqrt(np.array([0.5, 1.2]) * G)
    r = solve(
        body, w, headings=[0.0, 90.0], field_points=[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], options={"motion": True}
    )
    r.write_wamit(tmp_path / "hemi", timestamp="2026-10-24 00:00:00")
    paths = written(tmp_path / "hemi")
    assert sorted(p.suffix for p in paths) == sorted(
        [".1", ".2", ".2fk", ".2sc", ".3", ".3fk", ".3sc", ".4", ".6p", ".fpt", ".hst", ".out"]
    )
    assert r.route == "low_order_infinite_open_sea"


def test_radii_of_gyration_give_the_inertia_of_the_displaced_mass():
    """``radii_of_gyration`` with the mass unset gives I_ij = m r_ij |r_ij|
    with the displaced mass m, the same motion matrices as the
    explicit inertia; a negative r_ij gives a negative product of inertia."""
    r = ((1.0, 0.2, 0.0), (0.2, 0.8, -0.1), (0.0, -0.1, 0.9))
    mesh = hemi(6, quarter=False)
    kw = {"mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.2], "inertia_origin": "center_of_gravity"}
    p = model({"mesh": mesh, "radii_of_gyration": [list(x) for x in r], **kw}, [1.0], [0.0])
    h = hydrostatics(p)[0]
    m = h.mass
    assert m == RHO * h.volumes[0] == h.displaced_mass  # the displaced mass rho V_x of the mesh
    inertia = {k: m * r[i][j] * abs(r[i][j]) for k, (i, j) in zip(
        ("Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"), ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)), strict=True)}
    expl = model({"mesh": mesh, "mass": m, "inertia_input": "components", **inertia, **kw}, [1.0], [0.0])
    for a, b in zip(matrices(p), matrices(expl), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)
    assert inertia["Iyz"] < 0
    # three values are the diagonal
    diag = model({"mesh": mesh, "radii_of_gyration": [1.0, 2.0, 3.0], **kw}, [1.0], [0.0])
    full = model({"mesh": mesh, "radii_of_gyration": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], **kw},
                 [1.0], [0.0])  # fmt: skip
    for a, b in zip(matrices(diag), matrices(full), strict=True):
        np.testing.assert_array_equal(a, b)


def test_radii_of_gyration_rules():
    # input rules
    mesh = hemi(4)

    def check(**kw):
        model({"mesh": mesh, **kw}, [1.0], [0.0]).check()

    # both given: inertia_input chooses, the other not read
    check(radii_of_gyration=[1, 1, 1], mass=1.0, center_of_gravity=[0, 0, 0], Ixx=1, Ixy=0, Ixz=0, Iyy=1, Iyz=0, Izz=1)
    with pytest.raises(ValueError, match="symmetric"):
        check(radii_of_gyration=[[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match=r"shape \(3,\) or \(3, 3\)"):
        check(radii_of_gyration=[1, 1])
    # the inertia enters the motions only: without them a radius needs neither a mass nor a centre
    check(radii_of_gyration=[1, 1, 1], mass="displaced")


def test_hydrostatics_only_problem():
    """Radiation and diffraction off: the route 'hydrostatics', no frequencies
    needed, the hydrostatics quantities only (the same computation as in a full solve: 1e-14);
    omegas empty otherwise rejected."""
    body = {"mesh": hemi(6), "mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.1]}
    opts = {"radiation": False, "diffraction": False}
    p = model(body, [], (), options=opts)
    assert p.route() == "hydrostatics"
    r = p.solve()
    assert r.added_mass is None and r.excitation.diffraction is None and len(r.omegas) == 0
    full = solve(body, [1.0])
    for key in ("volumes", "volume", "center_of_buoyancy", "waterplane_area", "waterplane_centroid", "mass"):
        np.testing.assert_allclose(r.quantity("hydrostatics_" + key)[0], full.quantity("hydrostatics_" + key)[0],
                                   rtol=1e-14)  # fmt: skip
    np.testing.assert_allclose(r.hydrostatics.stiffness[0], full.hydrostatics.stiffness[0], rtol=1e-14)
    with pytest.raises(ValueError, match="omegas is empty"):
        model(body, [], ()).check()


def test_log_tolerance_per_pair():
    """The per-pair analytic log integral. A tolerance small enough to
    select every pair equals 0 (every pair) bitwise; the default changes A, B, X by less
    than 1e-4 of their peaks (the tolerance bounds the correction of each pair left out,
    relative to its panel area, theory §6); a large tolerance changes them more; bad values are rejected."""
    body = {"mesh": surface(cylinder_panels(1.0, 0.5, n_phi=8, n_z=4, n_r=4, quarter=True), symmetry="xz_and_yz"),
            "position": [8.0, 0.0, 0.0]}  # fmt: skip
    far = {"name": "far", "mesh": body["mesh"], "position": [-8.0, 0.0, 0.0]}
    w = np.sqrt(np.array([0.6, 1.4]) * G)

    def run(tol):
        return solve([body, far], w, options={"log_tolerance": tol})

    every, tiny, default, loose = run(0.0), run(1e-12), run(1e-4), run(1e-2)
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        assert np.array_equal(view(tiny, q), view(every, q)), q
        ref = view(every, q)
        assert peak(view(default, q) - ref) <= 1e-4 * peak(ref), q
    d_default = peak(default.added_mass - every.added_mass)
    assert peak(loose.added_mass - every.added_mass) > d_default
    m = model([body, far], w)
    for bad in (-1e-3, math.inf, math.nan):
        with pytest.raises(ValueError, match="log_tolerance"):
            m.options.log_tolerance = bad
