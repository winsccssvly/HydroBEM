"""Internal tanks, low order (theory §10.2).

A tank adds to the body its own radiation problem and restoring. At omega = 0
(a rigid lid) the tank fluid surges with the walls, at omega = inf (phi = 0
on the free surface) it heaves with them, so the tank's added mass is the
fluid mass rho_T V_T there; at omega = 0 a mode with a net flux through the
walls (heave) has no solution and is NaN; the tank radiates no waves,
its computed damping is small away from the sloshing modes; its
free surface reduces C33 by rho_T g A_T; density 0 is the body without the
tank; the first sloshing mode of a rectangular tank sits at
omega^2 = g k tanh(k h), k = pi / L; a circular tank forced in surge and pitch has the added mass of
the closed-form series (Faltinsen & Timokha 2009). Each on one mesh with the cause of its bound; the
convergence with the mesh and the higher order against a fine low-order tank are validation, run
when the method changes, not tests. Meshes are procedural (tests/python/meshes.py)."""

import contextlib
import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import TANK_B as B
from bodies import TANK_H as H
from bodies import TANK_L as L
from bodies import TANK_ZT as ZT
from bodies import flat, part, surface
from bodies import tank_vessel as vessel
from meshes import box_panels, cylinder_panels, face_panels
from problems import RHO, G, circular_tank, endpoints, model, peak, solve, view


def limits(body):
    """The added mass at omega = 0 and inf; heave at omega = 0 is not defined (NaN), as its
    warning says."""
    with _heave_not_defined(any(t.get("density_ratio", 1.0) for t in body.get("tanks", []))):
        return _limits(body)


def _limits(body):
    return endpoints(model(body, [0.0, math.inf], (), environment={"density": RHO, "gravity": G},
                 options={"diffraction": False}).solve())  # fmt: skip


def _heave_not_defined(tank):
    """The warning of a tank at omega = 0, when there is a tank fluid."""
    return pytest.warns(hb.HydroBEMWarning, match="not defined") if tank else contextlib.nullcontext()


def tank_limit_errors(n):
    """The tank's surge and sway added mass at omega = 0 and its heave at omega = inf (the vessel with
    the tank less the vessel with an empty one) against the fluid mass rho_T V_T, on the tank walls
    of n panels: the relative errors."""
    mass = RHO * L * B * H
    with_tank, empty = limits(vessel(1.0, n=n)), limits(vessel(0.0, n=n))
    e = [with_tank["zero"][0, 0] - empty["zero"][0, 0], with_tank["zero"][1, 1] - empty["zero"][1, 1],
         with_tank["infinite"][2, 2] - empty["infinite"][2, 2]]  # fmt: skip
    return np.abs(np.array(e) / mass - 1.0)


def test_tank_fluid_moves_with_the_walls_at_the_limits():
    # exact in the continuous problem (the fluid moves rigidly with the walls); constant panels on
    # the box of the tank with its edges: 0.8 to 2.5 % on these 8 x 4 x 3 panels a quadrant (a third
    # of that on twice as many: the convergence is validation)
    err = tank_limit_errors((8, 4, 3))
    assert np.all(err < 0.03), err


def test_tank_does_not_radiate_and_density_zero_removes_it():
    w = np.array([1.0, 2.0])
    r1, r0 = solve(vessel(1.0), w, headings=[0.0]), solve(vessel(0.0), w, headings=[0.0])
    hull = solve({"mesh": surface(box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True), symmetry="xz_and_yz")}, w,
                 headings=[0.0])  # fmt: skip
    # a closed domain radiates nothing (exact): the wave parts of the Green function cancel over the
    # closed tank in the continuous problem, the panels leave a damping of the discretization, away
    # from the sloshing modes 1.5e-3 of the peak here (5.7e-4 on twice as many panels), kept
    assert peak(r1.radiation_damping - r0.radiation_damping) <= 3e-3 * peak(r0.radiation_damping)
    # density 0 keeps the tank elements out of the exterior problem: the hull alone, to the bit
    assert peak(r0.excitation.diffraction - r1.excitation.diffraction) == 0.0
    assert peak(r0.added_mass - hull.added_mass) == 0.0
    assert peak(r0.excitation.diffraction - hull.excitation.diffraction) == 0.0


def test_zero_frequency_modes_with_a_net_flux_are_not_defined(tmp_path):
    """Input rule. At omega = 0 the tank free surface is rigid, so a mode with a net volume flux through
    the tank walls has no solution: its row and column of the zero-frequency added mass are NaN,
    with a warning; the other entries stay (surge and sway of the centred tank)."""
    with pytest.warns(hb.HydroBEMWarning, match="tank 1: .* the heave mode has a net volume flux"):
        a0 = _limits(vessel(1.0))["zero"]
    assert np.all(np.isnan(a0[2, :])) and np.all(np.isnan(a0[:, 2]))
    keep = [0, 1, 3, 4, 5]
    assert np.all(np.isfinite(a0[np.ix_(keep, keep)]))
    # a tank off the y axis (x > 0): pitch carries a net flux too, roll does not
    hull = box_panels(2.0, 1.0, 1.0, n=(8, 4, 4))
    walls = box_panels(0.4, 0.3, 0.3, n=(2, 2, 2))[:, ::-1] + [0.6, 0.0, ZT]
    mesh = surface(np.concatenate([hull, walls]))
    body = {"mesh": part(mesh, 1, len(hull)), "tanks": [{"mesh": part(mesh, len(hull) + 1, len(hull) + len(walls))}]}
    with pytest.warns(hb.HydroBEMWarning, match="the heave, pitch mode has a net volume flux"):
        a0 = _limits(body)["zero"]
    assert [bool(np.isnan(a0[i, i])) for i in range(6)] == [False, False, True, False, True, False]
    # the files list the defined entries only
    with pytest.warns(hb.HydroBEMWarning, match="net volume flux"):
        r = model(vessel(1.0), [0.0, 1.0], (), options={"diffraction": False}).solve()
    r.write_wamit(tmp_path / "v")
    one = (tmp_path / "v.1").read_text().splitlines()
    assert one and not any("nan" in x.lower() for x in one)
    assert "undefined" in (tmp_path / "v.out").read_text()


def test_a_tank_in_a_channel_is_the_tank_in_the_open_sea():
    """The tank problem is of the tank's closed fluid alone (theory §10.2), the same in a channel as
    in the open sea of the same depth: the tank's part of A and B (the vessel less the vessel with
    the tank removed) agrees to 1e-6 of A (float32, the sea parts subtracted). Before, the tank
    problem kept the channel's width without its domain and the solve was refused."""
    w = np.array([1.0, 2.0])

    def part_of_tank(opts):
        r1, r0 = (model(vessel(d), w, (), environment={"depth": 6.0}, options={"diffraction": False, **opts}).solve()
                  for d in (1.0, 0.0))  # fmt: skip
        return np.asarray(r1.added_mass) - np.asarray(r0.added_mass), np.asarray(r1.added_mass)

    (ch, scale), (sea, _) = part_of_tank({"domain": "channel", "channel_width": 8.0}), part_of_tank({})
    assert peak(ch - sea) <= 1e-6 * peak(scale)


def test_tank_restoring_and_mass():
    h1 = model(vessel(0.5), [1.0], (), options={"diffraction": False}).solve()
    h0 = model(vessel(0.0), [1.0], (), options={"diffraction": False}).solve()
    c1, c0 = (np.asarray(h.hydrostatics.stiffness_rho_g[0]) for h in (h1, h0))
    # the closed forms of the tank's free surface and fluid weight (theory §10.2): C33 exact for flat
    # panels (1e-12 round-off)
    np.testing.assert_allclose(c0[2, 2] - c1[2, 2], 0.5 * RHO * G * L * B, rtol=1e-12)
    # roll: the free-surface moment of inertia and the weight of the tank fluid (centroid ZT - H/2);
    # the one-point rule of the constant panels (theory §3.2) misses the own second moment of each
    # bottom panel, A h_y^2 / 12 for the panel width h_y = (B/2) / 4 of the tank walls (the weight
    # term and C33 are exact for flat panels), so the discrete value holds to round-off
    hy = (B / 2) / 4
    d44 = 0.5 * RHO * G * (L * B**3 / 12 - L * B * hy**2 / 12 + L * B * H * (ZT - H / 2))
    np.testing.assert_allclose(c0[3, 3] - c1[3, 3], d44, rtol=1e-12)
    assert h1.hydrostatics.mass[0] == pytest.approx(h0.hydrostatics.mass[0] - 0.5 * RHO * L * B * H, rel=1e-12)


def test_sloshing_resonance():
    # the first sloshing mode of the rectangular tank, omega^2 = g k tanh(k h), k = pi / L (analytic):
    # the tank's surge added mass changes sign across it, 10 % below and above
    k = math.pi / L
    wn = math.sqrt(G * k * math.tanh(k * H))
    w = wn * np.array([0.9, 1.1])
    r1, r0 = solve(vessel(1.0, n=(12, 6, 4)), w, options={"diffraction": False}), \
        solve(vessel(0.0, n=(12, 6, 4)), w, options={"diffraction": False})  # fmt: skip
    a = r1.added_mass[:, 0, 0] - r0.added_mass[:, 0, 0]
    assert a[0] > RHO * L * B * H and a[1] < 0.0  # below the resonance larger than the fluid mass, above negative


def test_circular_tank_forced_in_surge_and_pitch():
    """The fluid in a vertical circular tank of radius 1 and depth 1 (the walls of zero thickness:
    the hull's panels reversed), forced in surge and in pitch about the centre of its free surface
    at K = 0.5, 1.0, 2.5, away from the first sloshing mode K1 = 1.751 (k1 = 1.841, J1'(k1) = 0):
    the tank's A11, A15 and A55 (the body with the tank less the body with an empty one) against
    the closed-form series (problems.circular_tank, Faltinsen & Timokha 2009, §5.4.4). The bound is
    the discretization of constant panels on the tank with its edge, 16 x 16 x 8 a quadrant (around,
    down, across the bottom; cosine-spaced): 1e-2 of the largest value of each (6.0e-3 found for
    A11 at K = 1, 3.3e-3 for A15, 2.2e-3 for A55; 8 x 8 x 4: 2.3e-2)."""
    K = np.array([0.5, 1.0, 2.5])
    hull = cylinder_panels(1.0, 1.0, 16, 16, 8, cosine=True)
    mesh = surface(np.concatenate([hull, hull[:, ::-1]]), symmetry="xz_and_yz")
    a = []
    for density in (1.0, 0.0):
        body = {"mesh": part(mesh, 1, len(hull)), "modes": ["surge", "pitch"], "interior_surface": "none",
                "tanks": [{"mesh": part(mesh, len(hull) + 1, 2 * len(hull)), "density_ratio": density}]}  # fmt: skip
        r = solve(body, np.sqrt(G * K), options={"diffraction": False})
        a.append(np.array([r.coefficient("added_mass", ("body", i), ("body", j))
                           for i, j in (("surge", "surge"), ("surge", "pitch"), ("pitch", "pitch"))]))  # fmt: skip
    ref = circular_tank(1.0, K)
    err = np.abs((a[0] - a[1]) / RHO - ref).max(axis=1) / np.abs(ref).max(axis=1)
    assert np.all(err <= 1e-2), err


def test_tank_inputs_are_checked():
    # input rules
    m = model(vessel(), [1.0], ())
    tank = m.bodies["body"].tanks[0]
    with pytest.raises(ValueError, match="density_ratio"):
        tank.density_ratio = -1.0
    tank.surface_level = ZT - 0.2
    with pytest.raises(ValueError, match="above its surface level"):
        m.counts()


def baffled_vessel(eps=None, n=(8, 4, 3)):
    """The vessel with a transverse baffle in its tank at x = L/4 (0 <= y <= B/2, from the tank bottom
    to 0.6 of the fluid depth), elements of the body's mesh: a dipole of the tank, or with ``eps`` a
    dipole damper of the tank."""
    hull = box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True)
    walls = box_panels(L / 2, B / 2, H, n=n, quarter=True)[:, ::-1] + [0.0, 0.0, ZT]
    baffle = np.array(face_panels((L / 4, 0.0, ZT - H), (0, B / 2, 0), (0, 0, 0.6 * H), 4, 3, (1, 0, 0)))
    mesh = surface(np.concatenate([hull, walls, baffle]), symmetry="xz_and_yz")
    n0, n1 = len(hull), len(hull) + len(walls)
    sheet = {"mesh": part(mesh, n1 + 1, n1 + len(baffle))}
    tank = {"mesh": part(mesh, n0 + 1, n1)}
    if eps is None:
        tank["dipoles"] = [sheet]
    else:
        tank["dipole_dampers"] = [{**sheet, "damping_factor": eps}]
    return {"mesh": part(mesh, 1, n0), "mass": "displaced", "tanks": [tank]}


SLOSH = np.array([2.0, 3.0, 4.5])  # about the first sloshing mode of the tank (3.8 rad/s)


def test_tank_baffle_is_in_the_tank_problem():
    # input rule: the baffle is a sheet of the tank fluid, so the exterior problem (the exciting
    # force) is that of the vessel, the same equations: bit-identical; the tank's surge added mass
    # changes by O(1) near the sloshing mode (16 % here; a baffle left out of the tank problem: 0)
    r0, r1 = (solve(b, SLOSH, headings=[30.0]) for b in (vessel(), baffled_vessel()))
    assert peak(r1.excitation.diffraction - r0.excitation.diffraction) == 0.0
    assert peak(r1.added_mass[:, 0, 0] - r0.added_mass[:, 0, 0]) >= 1e-2 * peak(r0.added_mass[:, 0, 0])
    # the panels as the engine uses them: the baffle a sheet of tank 0, the walls of tank 0
    m = model(baffled_vessel(), SLOSH)
    b = m.bodies[0]
    p, w = b.panels(role="dipole", include_reflections=False), b.panels(role="tank", include_reflections=False)
    assert len(p.role) == 12 and set(p.role) == {"dipole"} and set(p.tank) == {0}
    assert set(w.role) == {"tank"} and set(w.tank) == {0} and not set(p.element) & set(w.element)


def test_tank_damper_limits():
    # a damper of the tank, limits of its law: no jump at eps = 0, the vessel (exact; float32 leaves
    # 4e-8, 1e-5 the bound of a damper at eps = 0); the rigid baffle as eps grows, the flux through
    # the sheet O(1/eps) (the damping 3.9e-4, 3.9e-5, 3.9e-6 of the peak at 1e3, 1e4, 1e5)
    ref0, ref1 = (solve(b, SLOSH, options={"diffraction": False}) for b in (vessel(), baffled_vessel()))
    for eps, ref, tol in ((0.0, ref0, 1e-5), (1e5, ref1, 1e-4)):
        r = solve(baffled_vessel(eps), SLOSH, options={"diffraction": False})
        for q in ("added_mass", "radiation_damping"):
            assert peak(view(r, q) - view(ref, q)) <= tol * peak(view(ref, q)), (eps, q)
    m = model(baffled_vessel(2.0), SLOSH)
    p = m.bodies[0].panels(role="dipole_damper", include_reflections=False)
    assert len(p.role) == 12 and set(p.role) == {"dipole_damper"} and set(p.tank) == {0} and set(p.owner) == {0}


def test_tank_sheet_inputs_are_checked():
    # input rules
    b = baffled_vessel()
    other = surface(box_panels(0.2, 0.2, 0.2, n=(1, 1, 1), quarter=True), symmetry="xz_and_yz")
    bad = {**b, "tanks": [{**b["tanks"][0], "dipoles": [{"mesh": other}]}]}
    with pytest.raises(ValueError, match="bodies/body/tanks/1/dipoles/1: a sheet of a tank takes elements of the body"):
        model(bad, [1.0], ()).route()
    # an element of the walls is not also a sheet of the tank
    mesh, (first, last) = b["tanks"][0]["mesh"]
    bad = {**b, "tanks": [{**b["tanks"][0], "dipoles": [{"mesh": (mesh, (last, last))}]}]}
    with pytest.raises(ValueError, match="is taken by both tanks 1 and tanks 1 dipoles 1"):
        model(bad, [1.0], ()).route()


def vessel_ho(density=1.0, spans=4):
    """The vessel of flat higher-order patches: hull and tank walls one patch per face."""
    hull = box_panels(2.0, 1.0, 1.0, n=(1, 1, 1), quarter=True)
    walls = box_panels(L / 2, B / 2, H, n=(1, 1, 1), quarter=True)[:, ::-1] + [0.0, 0.0, ZT]
    mesh = flat(np.concatenate([hull, walls]), (spans, spans, 3, 3, 3, 4), symmetry="xz_and_yz")
    return {"mesh": part(mesh, 1, 3), "tanks": [{"mesh": part(mesh, 4, 6), "density_ratio": density}]}


def tank_part(make, higher_order):
    """The tank's own added mass (the vessel with the tank less with density 0) on both sides of the
    first sloshing mode, 0.8 and 1.25 times its frequency; ``make(density)`` gives the vessel."""
    k = math.pi / L
    w = math.sqrt(G * k * math.tanh(k * H)) * np.array([0.8, 1.25])
    env, opts = {"density": RHO, "gravity": G}, {"diffraction": False, "higher_order": higher_order}
    r1, r0 = (model([make(d)], w, (), environment=env, options=opts).solve() for d in (1.0, 0.0))
    return np.asarray(r1.added_mass) - np.asarray(r0.added_mass)


def test_higher_order_tank_fluid_moves_with_the_walls():
    """The limits give the tank fluid mass: exact in the continuous problem, and the flat patches
    represent the box and the rigid flow in it exactly (0 observed; 1e-6 the float32 storage)."""
    mass = RHO * L * B * H

    def limits_ho(density):
        with _heave_not_defined(density != 0.0):
            m = model([vessel_ho(density)], [0.0, math.inf], (), environment={"density": RHO, "gravity": G},
                      options={"diffraction": False, "higher_order": True})  # fmt: skip
            return endpoints(m.solve())

    e1, e0 = limits_ho(1.0), limits_ho(0.0)
    for d, key in ((0, "zero"), (1, "zero"), (2, "infinite")):
        assert abs((e1[key][d, d] - e0[key][d, d]) / mass - 1.0) < 1e-6, (d, key)


# ---- field points inside tanks -------------------------------------------------------------

TANK_POINTS = np.array([[0.0, 0.0, ZT], [0.0, 0.0, ZT - 0.25], [0.2, 0.1, ZT - 0.2]])


def with_points(body, points):
    """The body with the field points of its tank (body axes; the body at the origin: global)."""
    return {**body, "tanks": [{**body["tanks"][0], "field_points": np.asarray(points, dtype=float)}]}


def tank_fields(body, w, higher_order=False, extra=None):
    """The fields at TANK_POINTS in the tank (``r.bodies[0].tanks[0]``) and at ``extra`` in the sea
    (``r.field_points``)."""
    m = model(with_points(body, TANK_POINTS), w, [0.0], field_points=extra,
              environment={"density": RHO, "gravity": G},
              options={"field_pressure": True, "field_velocity": True, "higher_order": higher_order})  # fmt: skip
    return m.solve()


def heave_exact(w):
    """A heaving tank: phi = (z - ZT) + 1/K per unit velocity, so p / (rho g) = 1 + K (z - ZT) per
    unit amplitude (the elevation 1 at the tank free surface) and the velocity is (0, 0, i omega)."""
    K = w**2 / G
    return 1.0 + K[:, None] * (TANK_POINTS[:, 2] - ZT)[None, :]


def test_tank_field_points_heave_closed_form_higher_order():
    # the closed form of a heaving tank (heave_exact); the flat patches represent it exactly, the
    # float32 storage of the fields leaves 1.2e-7
    w = np.array([1.5, 3.0])
    r = tank_fields(vessel_ho(1.0), w, higher_order=True)
    t = r.bodies[0].tanks[0]
    np.testing.assert_array_equal(t.points, TANK_POINTS)  # body axes, as given
    np.testing.assert_allclose(t.pressure_radiation[:, :, 2], heave_exact(w), atol=1e-6)
    v = t.velocity_radiation[:, :, 2]
    np.testing.assert_allclose(
        v, np.broadcast_to(1j * w[:, None, None] * np.array([0.0, 0.0, 1.0]), v.shape), atol=1e-6
    )
    # no incident or diffracted wave inside a tank: a tank has radiation outputs only, and without
    # points in the sea there are no field outputs of the sea
    assert not hasattr(t, "pressure_diffraction") and r.field_points.pressure_diffraction is None


def tank_field_errors(n):
    """The heave pressure (by Green's identity) and vertical velocity (by the sources) at the tank
    points of the low-order vessel with tank walls of n panels against the closed form: the largest
    relative errors."""
    w = np.array([1.5])
    t = tank_fields(vessel(1.0, n=n), w).bodies[0].tanks[0]
    e = np.abs(t.pressure_radiation[:, :, 2] / heave_exact(w) - 1.0).max()
    ev = np.abs(t.velocity_radiation[0, :, 2, 2] / (1j * w[0]) - 1.0).max()
    return e, ev


def test_tank_field_points_heave_closed_form_low_order():
    # the closed form of a heaving tank (heave_exact): the pressure by Green's identity with constant
    # panels in the box, 0.6 % on 8 x 4 x 3 panels a quadrant (0.2 % on twice as many); the velocity
    # by the sources converges at first order only (55 % here, 29 % on twice as many: validation)
    e, _ = tank_field_errors((8, 4, 3))
    assert e < 0.01, e


def test_tank_field_points_leave_the_exterior_unchanged():
    # input rule: the points inside a tank do not enter the exterior problem: bit-identical
    w = np.array([2.0])
    ext = np.array([[3.0, 0.5, 0.0], [2.5, -1.0, -0.4]])
    r = tank_fields(vessel(1.0), w, extra=ext)
    r0 = model(vessel(1.0), w, [0.0], field_points=ext, environment={"density": RHO, "gravity": G},
               options={"field_pressure": True, "field_velocity": True}).solve()  # fmt: skip
    for key in ("pressure_radiation", "pressure_diffraction", "velocity_radiation", "velocity_diffraction"):
        np.testing.assert_array_equal(r.quantity("field_points_" + key), r0.quantity("field_points_" + key))
    np.testing.assert_array_equal(r.added_mass, r0.added_mass)


def test_tank_field_point_inputs_are_checked():
    # input rules
    def counts(tank_points=None, sea_points=None, density=1.0):
        b = vessel(density)
        if tank_points is not None:
            b = with_points(b, tank_points)
        return model(b, [1.0], (), field_points=sea_points,
                     options={"diffraction": False, "field_pressure": True}).counts()  # fmt: skip

    with pytest.raises(ValueError, match="above the free surface of its tank"):
        counts([[0.0, 0.0, ZT + 0.05]])
    # a point of a tank is named by its number in the tank's field_points, with the tank
    with pytest.raises(ValueError, match="^field point 2 of bodies/body/tanks/1 lies above") as e:
        counts([[0.0, 0.0, ZT - 0.1], [0.0, 0.0, ZT + 0.05]], sea_points=[[5.0, 0.0, 0.0]])
    assert (e.value.location.field_point, e.value.location.body) == (1, 0)
    with pytest.raises(ValueError, match="empty tank"):
        counts([[0.0, 0.0, ZT - 0.1]], density=0.0)
    # a tank point inside the hull's water plane is not an exterior point
    c = counts([[0.0, 0.0, ZT]])
    assert (c.field_points, c.tank_points) == (0, 1)
    with pytest.raises(ValueError, match="water plane"):
        counts(sea_points=[[0.0, 0.0, 0.0]])
    # a tank's field points are of shape (N, 3)
    with pytest.raises(ValueError, match="field_points must have shape"):
        counts([0.0, 0.0, ZT - 0.1])


def test_pressure_drift_with_a_tank_is_low_order_only():
    # input rule: the higher order rejects the pressure drift with a tank (the low-order drift with a
    # tank is checked by the quadratic QTF, whose difference diagonal is that drift)
    ho = {**vessel_ho(1.0), "mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.6]}
    opts = {"higher_order": True, "motion": True, "drift_pressure_integration": True}
    with pytest.raises(ValueError, match="low order only"):
        model(ho, [1.0], [0.0], options=opts).counts()
