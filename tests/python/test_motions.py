"""The equation of motion the engine solves (theory §3.1), checked by its residual: a low-order body
with a full inertia tensor and additional stiffness, several bodies, the higher order, the Haskind
exciting force, internal tanks, generalized modes with their mass, stiffness and damping, and
pressure modes. The residual of

    [C - w^2 (M + A(w)) + i w (B(w) + B_ext)] xi = X

is formed from the results and the model's matrices alone, an exact identity of the solve: the
engine's complex LU of the same matrices and the float32 storage of the motions leave about 1e-7
(1e-6 the bound)."""

import numpy as np

from bodies import acv_body, lo_free_cylinder, spline_cylinder, surface, tank_vessel
from meshes import cylinder_panels, hemisphere_panels, spheroid_panels
from problems import FEATURES, G, model, peak

W = np.sqrt(np.array([0.5, 1.2]) * G)


def full_matrices(r, body_items=()):
    """M, C, B_ext of every coordinate: the rigid 6N of ``motion_equation``, the inputs
    ``mode_mass``, ``mode_stiffness``, ``mode_damping`` of the bodies (given in ``body_items``, one
    body here), and the computed ``mode_restoring`` and ``pressure_mode_damping`` of the results."""
    e = r.model.motion_equation()
    M6, C6, B6 = e.inertia, e.stiffness, e.damping
    D = r.added_mass.shape[1]
    M, C, B = (np.zeros((D, D)) for _ in range(3))
    n = M6.shape[0]
    M[:n, :n], C[:n, :n], B[:n, :n] = M6, C6, B6
    for key, target in (("mode_mass", M), ("mode_stiffness", C), ("mode_damping", B)):
        given = dict(body_items).get(key)
        if given is not None:
            target[: len(given), : len(given)] += given
    if r.mode_restoring is not None:
        C += r.mode_restoring
    if r.pressure_mode_damping is not None:
        B += r.pressure_mode_damping
    return M, C, B


def residual(r, force="excitation_diffraction", body_items=()):
    """The largest residual of the equation of motion relative to its force, over F and H."""
    M, C, B = full_matrices(r, body_items)
    worst = 0.0
    for f, w in enumerate(r.omegas):
        Z = C - w * w * (M + r.added_mass[f]) + 1j * w * (r.radiation_damping[f] + B)
        for h in range(len(r.headings)):
            X = r.quantity(force)[f, h]
            worst = max(worst, peak(Z @ r.motion[f, h] - X) / peak(X))
    return worst


def test_full_inertia_and_additional_stiffness():
    # the hemisphere of 8 panels a quadrant edge, a given mass, a full inertia tensor about an
    # off-centre centre of gravity and an additional stiffness (5.6e-8 observed)
    hemi = surface(hemisphere_panels(1.0, 8, 8, quarter=True), symmetry="xz_and_yz")
    body = {"mesh": hemi, "mass": 1900.0, "center_of_gravity": [0.1, -0.05, -0.2],
            "inertia_input": "components", "Ixx": 420.0, "Ixy": 12.0,
            "Ixz": -8.0, "Iyy": 390.0, "Iyz": 5.0, "Izz": 610.0,
            "additional_stiffness": np.diag([1e3, 2e3, 0, 0, 0, 5e3])}  # fmt: skip
    w = np.sqrt(np.array([0.4, 1.0, 2.0]) * G)
    r = model(body, w, [0.0, 30.0], options={"motion": True}).solve()
    assert residual(r) <= 1e-6


def test_two_bodies():
    cyl = {"name": "cyl", "mesh": surface(cylinder_panels(1.0, 0.5, n_phi=16, n_z=3, n_r=3, quarter=False)),
           "mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.1], "radii_of_gyration": [0.6, 0.6, 0.7]}
    sph = {"name": "sph", "mesh": surface(spheroid_panels(2.0, 0.25, n_x=8, n_t=8), symmetry="xz_and_yz"),
           "position": [0.0, 3.0, 0.0], "mass": "displaced", "center_of_gravity": [0.0, 0.0, 0.0],
           "radii_of_gyration": [0.3, 1.0, 1.0]}  # fmt: skip
    r = model([cyl, sph], W, [0.0, 30.0], options={"motion": True}).solve()
    assert r.motion.shape == (2, 2, 12) and residual(r) <= 1e-6


def test_higher_order():
    body = {**lo_free_cylinder(), "mesh": spline_cylinder(n=4)}
    r = model(body, W, [0.0], options={"motion": True, "higher_order": True}).solve()
    assert residual(r) <= 1e-6


def test_haskind_exciting_force():
    # excitation_method "haskind": the motions of the Haskind force, which differs from the pressure
    # force by the discretization, within 1 % on this mesh (the motions within 1.5 %); the force
    # left unused would give 0
    rh = model({**lo_free_cylinder(), "excitation_method": "haskind"}, W, [0.0, 30.0], options={"motion": True}).solve()
    rp = model(lo_free_cylinder(), W, [0.0, 30.0], options={"motion": True}).solve()
    assert residual(rh, "excitation_haskind") <= 1e-6
    assert 0.0 < peak(rh.motion - rp.motion) <= 1.5e-2 * peak(rp.motion)


def test_internal_tank():
    body = {**tank_vessel(1.0), "center_of_gravity": [0.0, 0.0, -0.2], "radii_of_gyration": [0.4, 0.8, 0.8]}
    r = model(body, W, [0.0], options={"motion": True}).solve()
    assert residual(r) <= 1e-6


def test_generalized_modes_with_their_mass_stiffness_and_damping():
    m = FEATURES["generalized"]()
    n = 6 + len(m.bodies["box"].generalized_modes)
    items = {k: np.zeros((n, n)) for k in ("mode_mass", "mode_stiffness", "mode_damping")}
    items["mode_mass"][6:, 6:] = np.diag([50.0, 20.0])
    items["mode_stiffness"][6:, 6:] = np.diag([300.0, 900.0])
    items["mode_damping"][6:, 6:] = np.diag([40.0, 15.0])
    m.bodies["box"].update(**items)
    r = m.solve()
    assert residual(r, body_items=items) <= 1e-6
    # the mode damping is in the equation: without it the motions of the modes change
    m.bodies["box"].mode_damping = None
    assert peak(m.solve().motion[..., 6:] - r.motion[..., 6:]) >= 1e-2 * peak(r.motion[..., 6:])


def test_pressure_modes():
    body = acv_body(modes=("sign_x", "uniform"), mass="displaced", center_of_gravity=[0.0, 0.0, 0.0],
                    radii_of_gyration=[2.5, 5.0, 5.0])  # fmt: skip
    r = model(body, 2 * np.pi / np.array([4.0, 6.0]), [150.0], environment={"depth": 20.0},
              options={"motion": True}).solve()  # fmt: skip
    assert r.mode_restoring is not None and residual(r) <= 1e-6
