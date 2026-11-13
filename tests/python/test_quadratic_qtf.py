"""The quadratic QTFs by pressure integration (theory §12.1, test-plan B).

Every term of the pressure drift is the mean of a product of two first-order quantities; the QTF
takes the product for two waves at their difference and their sum frequency. The difference
coefficient of a wave with itself is the mean drift (both methods, moving bodies, several
bodies, pressure surfaces with the closed forms, tanks); the coefficients are Hermitian
(difference) and symmetric (sum) in the pair; the heading pairs of "bidirectional" contain the
equal headings and, at equal frequencies, follow the momentum drift of the heading pairs; the
bidirectional drift of a pair of headings is the diagonal of the difference QTF of that pair; the
engine computes the drifts the QTFs rest on and returns them only when asked for; the frequencies
asked for; the lists of methods; Newman's approximation (theory §12.13) of each of the three
drifts, its diagonal the drift. Generalized modes (theory §12.15): the rigid modes given as
generalized modes have the rigid body's QTFs (fixed, and floating with the translations); the
generalized forces of flexible modes have the identities of the rigid rows (the diagonal the drift,
Hermitian and symmetric pairs). Meshes are procedural (tests/python/meshes.py)."""

import numpy as np
import pytest

from bodies import acv_body, free_cylinder, lo_flexible_cylinder, lo_free_cylinder, surface, tank_vessel
from meshes import cylinder_panels
from problems import model, rigid_modes_as_generalized

G = 9.80665
W = np.array([0.9, 1.6, 2.4, 3.1])
ENV = {"density": 1000.0, "gravity": G}


PRESSURE = {"drift_pressure_integration": True, "qtf_quadratic_pressure_integration": True}


def solve(bodies, w=W, heads=(0.0, 30.0), **outputs):
    out = {**PRESSURE, **outputs}
    return model(bodies, w, list(heads), environment=ENV, outputs=out).solve()


def pair_list(heads, bidirectional):
    """The heading pairs of the drifts and QTFs: (h, h), or every h_a <= h_b in heading order."""
    n = len(heads)
    return [(a, b) for a in range(n) for b in range(a, n) if bidirectional or a == b]


def check_diagonal_and_symmetry(r):
    # unidirectional: the pairs (h, h), the drift real
    dif = r.qtf.quadratic.pressure_integration.difference
    sm, pd = r.qtf.quadratic.pressure_integration.sum, r.drift.pressure_integration
    F, P = dif.shape[0], dif.shape[2]
    assert dif.shape == (F, F, P, pd.shape[2]) and sm.shape == dif.shape
    # on the coordinates: the rows the method computes, NaN on the others (the pressure coordinates)
    rows = [c.kind != "pressure" for c in r.coordinates]
    none = np.logical_not(rows)
    assert np.all(np.isnan(dif[..., none])) and np.all(np.isnan(sm[..., none])) and np.all(np.isnan(pd[..., none]))
    dif, sm, pd = dif[..., rows], sm[..., rows], pd[..., rows]
    assert np.array_equal(r.qtf.heading_pairs, np.repeat(np.asarray(r.headings)[:, None], 2, axis=1))
    diag = np.stack([dif[f, f] for f in range(F)])
    scale = np.abs(pd).max()
    assert np.abs(pd.imag).max() == 0.0
    assert np.abs(diag.real - pd.real).max() <= 1e-12 * scale
    assert np.abs(diag.imag).max() <= 1e-12 * scale
    np.testing.assert_allclose(dif, np.conj(np.swapaxes(dif, 0, 1)), rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(sm, np.swapaxes(sm, 0, 1), rtol=0, atol=1e-12 * scale)
    # the pairs of different frequencies are not the diagonal's
    assert np.abs(dif[0, 1] - dif[0, 0]).max() > 1e-3 * scale
    assert np.all(np.isfinite(dif)) and np.all(np.isfinite(sm))


@pytest.mark.parametrize("higher_order", [False, True])
def test_the_difference_qtf_of_a_wave_with_itself_is_the_mean_drift(higher_order):
    # two moving bodies (the motion terms), headings at 0 and 30 degrees
    make = free_cylinder if higher_order else lo_free_cylinder
    bodies = [make(name="a", position=[3.0, 0.5, 0.0]), make(name="b", position=[-2.0, 0.0, 0.0], yaw=30.0)]
    r = model(bodies, W, [0.0, 30.0], environment=ENV,
              options={"higher_order": higher_order, "motion": True, **PRESSURE}).solve()  # fmt: skip
    check_diagonal_and_symmetry(r)
    assert r.qtf.quadratic.pressure_integration.difference.shape[3] == 12


def test_pressure_surfaces_and_tanks():
    # the closed forms of the hydrostatic part (pressure surfaces) and the tank fluid on its walls
    acv = acv_body(modes=("uniform", "sign_x"), mass="displaced", center_of_gravity=[0.0, 0.0, 0.0],
                   radii_of_gyration=[2.5, 5.0, 5.0])  # fmt: skip
    with pytest.warns(UserWarning, match="wetted surface only"):
        r = model(acv, W[:3], [150.0], environment=ENV, options={"motion": True, **PRESSURE}).solve()
    check_diagonal_and_symmetry(r)
    tank = {**tank_vessel(n=(8, 4, 3)), "center_of_gravity": [0.0, 0.0, -0.6],
            "radii_of_gyration": [[0.6, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1]]}  # fmt: skip
    r = model(tank, np.array([2.5, 3.0, 3.5]), [0.0, 30.0], environment=ENV,
              options={"motion": True, **PRESSURE}).solve()  # fmt: skip
    check_diagonal_and_symmetry(r)


def test_heading_pairs():
    # bidirectional: the pairs h_a <= h_b in heading order; the equal headings are the
    # unidirectional coefficients, Hermitian (difference) or symmetric (sum) in the frequencies; at
    # equal frequencies the horizontal force follows the momentum drift of the heading pairs (a
    # fixed body; the near and the far field of one solution, equal in the continuous problem):
    # 3 % at equal headings, 8 % between them, the slow convergence of the pressure drift (the
    # velocities of the source formulation at the centroids) on this mesh (2.3 % and 6.7 % observed);
    # conjugated pairs differ by their size (170 %).
    # The bidirectional pressure drift is the diagonal of the difference QTF of the pairs
    pan = cylinder_panels(1.0, 0.5, n_phi=16, n_z=8, n_r=8, quarter=True, cosine=True)
    body = {"mesh": surface(pan, symmetry="xz_and_yz"), "modes": [], "position": [0.4, -0.2, 0.0]}
    heads = (0.0, 45.0, 120.0)
    r = solve(body, w=W[:3], heads=heads, qtf_bidirectional=True, drift_bidirectional=True,
              drift_pressure_integration=True, drift_momentum_conservation=True)  # fmt: skip
    u = solve(body, w=W[:3], heads=heads)
    pairs = pair_list(heads, True)
    expect = [[heads[a], heads[b]] for a, b in pairs]
    np.testing.assert_array_equal(r.qtf.heading_pairs, expect)
    np.testing.assert_array_equal(r.drift.heading_pairs, expect)
    np.testing.assert_array_equal(u.qtf.heading_pairs, [[h, h] for h in heads])
    np.testing.assert_array_equal(u.drift.heading_pairs, [[h, h] for h in heads])
    dp, sp = r.qtf.quadratic.pressure_integration.difference, r.qtf.quadratic.pressure_integration.sum
    assert dp.shape == (3, 3, len(pairs), 6) and sp.shape == dp.shape
    eq = np.array([a == b for a, b in pairs])
    scale = np.abs(dp).max()
    np.testing.assert_allclose(dp[:, :, eq], u.qtf.quadratic.pressure_integration.difference, rtol=0,
                               atol=1e-12 * scale)  # fmt: skip
    np.testing.assert_allclose(sp[:, :, eq], u.qtf.quadratic.pressure_integration.sum, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(dp[:, :, eq], np.conj(np.swapaxes(dp[:, :, eq], 0, 1)), rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(sp[:, :, eq], np.swapaxes(sp[:, :, eq], 0, 1), rtol=0, atol=1e-12 * scale)
    # the drift of the pairs: the diagonal of the difference QTF; that of (h, h) the unidirectional one
    pd = r.drift.pressure_integration  # [F, P, 6]
    near = np.stack([dp[f, f] for f in range(3)])
    np.testing.assert_allclose(pd, near, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(pd[:, eq], u.drift.pressure_integration, rtol=0, atol=1e-12 * scale)
    assert np.abs(pd[:, ~eq].imag).max() > 1e-3 * np.abs(pd).max()  # a pair of two headings is complex
    moving = r.drift.pressure_integration_body_axes
    assert np.all(np.isnan(moving[:, ~eq])) and np.all(np.isfinite(moving[:, eq]))
    mp = r.drift.momentum_conservation  # [F, P, 3]: surge, sway, yaw (global axes; the body is not turned)
    e = np.abs(near[..., :2] - mp[..., :2]) / np.abs(mp[..., :2]).max()
    assert e[:, eq].max() <= 0.03 and e[:, ~eq].max() <= 0.08, (e[:, eq].max(), e[:, ~eq].max())


def test_bidirectional_drift_of_a_moving_body_without_the_qtf_output():
    # drift_bidirectional alone: the pressure drift of the pairs of a moving body (the
    # motion terms) without the QTFs asked for; its pairs (h, h) are the unidirectional drift
    heads = (0.0, 30.0)
    opts = {"motion": True, "drift_pressure_integration": True}
    u = model(lo_free_cylinder(), W[:2], list(heads), environment=ENV, options=opts).solve()
    b = model(lo_free_cylinder(), W[:2], list(heads), environment=ENV,
              options={**opts, "drift_bidirectional": True}).solve()  # fmt: skip
    pairs = pair_list(heads, True)
    np.testing.assert_array_equal(b.drift.heading_pairs, [[heads[i], heads[j]] for i, j in pairs])
    assert b.qtf.heading_pairs is None and b.qtf.quadratic.pressure_integration.difference is None
    eq = [pairs.index((h, h)) for h in range(len(heads))]
    d, d0 = b.drift.pressure_integration, u.drift.pressure_integration
    np.testing.assert_allclose(d[:, eq], d0, rtol=0, atol=1e-12 * np.abs(d0).max())
    assert np.all(np.isfinite(d))


def test_the_switches_of_the_second_order_are_checked():
    m = model(lo_free_cylinder(), W[:1], [0.0], options={"motion": True})
    with pytest.raises(TypeError, match="second_order: drift_pressure_integration must be a bool"):
        m.second_order.drift_pressure_integration = "pressure"
    # a QTF at no frequencies, Newman's approximation without a drift
    m.second_order.update(qtf_quadratic_pressure_integration=True, qtf_difference=False, qtf_sum=False)
    with pytest.raises(ValueError, match="qtf_difference and qtf_sum are both false"):
        m.check()
    m.second_order.update(qtf_quadratic_pressure_integration=False, qtf_difference=True, qtf_newman=True)
    with pytest.raises(ValueError,
                       match="qtf_newman makes Newman's approximation from the drifts switched on: switch a "
                                         "drift on"):  # fmt: skip
        m.check()
    m.second_order.drift_momentum_conservation = True
    m.check()


def test_the_qtf_computes_the_drift_it_needs_and_the_frequencies_asked_for():
    # without drift_pressure_integration the engine computes the drift for the QTFs and
    # does not return it; the coefficients are the same; "difference" and "sum" return those only
    w = W[:2]
    opts = {"motion": True, "qtf_quadratic_pressure_integration": True}
    a = model(lo_free_cylinder(), w, [0.0], options={**opts, "drift_pressure_integration": True}).solve()
    b = model(lo_free_cylinder(), w, [0.0], options=opts).solve()
    assert b.quantity("drift_pressure_integration") is None and a.quantity("drift_pressure_integration") is not None
    name = "qtf_quadratic_pressure_integration_"
    for kind in ("difference", "sum"):
        np.testing.assert_array_equal(a.quantity(name + kind), b.quantity(name + kind))
    for kind, gone in (("difference", name + "sum"), ("sum", name + "difference")):
        c = model(lo_free_cylinder(), w, [0.0],
                  options={**opts, "qtf_difference": kind == "difference", "qtf_sum": kind == "sum"}).solve()
        assert c.quantity(gone) is None
        kept = name + kind
        np.testing.assert_array_equal(c.quantity(kept), a.quantity(kept))


# ---------------------------------------------------------------- the control surface (theory §11.3)


def cs_qtfs(radius, higher_order):
    make = free_cylinder if higher_order else lo_free_cylinder
    cs = {"method": "cylinder", "radius": radius, "depth": 1.5, "inner_radius": 1.0, "panel_size": 0.15}
    opts = {"higher_order": higher_order, "motion": True, "drift_pressure_integration": True,
            "drift_control_surface": True,
            "qtf_quadratic_pressure_integration": True, "qtf_quadratic_control_surface": True}  # fmt: skip
    return model({**make(), "control_surface": cs}, W[:3] + 0.6, [0.0, 30.0], environment=ENV, options=opts).solve()


@pytest.mark.slow  # seconds to half a minute: deselected by scripts/test.* fast
@pytest.mark.parametrize("higher_order", [False, True])
def test_the_control_surface_qtf_does_not_depend_on_the_surface(higher_order):
    # a free cylinder: the difference diagonal is the control-surface drift; Hermitian and
    # symmetric pairs; a control surface of radius 2 and one of radius 3 give the same
    # coefficients at all pairs, also of different frequencies, where the time derivatives over
    # the free surface inside the surface (which grows with it) and over the hull take part. The
    # balance is an identity of the continuous problem; the bound is the discretization of the two
    # surfaces (panels of 0.15, the fields at their points): 1e-4 in the higher order (2.3e-6
    # observed), 2e-3 in the low order, whose fields of constant sources converge as O(h) (5.1e-4)
    a, b = cs_qtfs(2.0, higher_order), cs_qtfs(3.0, higher_order)
    for r in (a, b):
        dif, sm = r.qtf.quadratic.control_surface.difference, r.qtf.quadratic.control_surface.sum
        cd = r.drift.control_surface
        scale = np.abs(cd).max()
        diag = np.stack([dif[f, f] for f in range(dif.shape[0])])
        assert np.abs(diag - cd).max() <= 1e-12 * scale
        np.testing.assert_allclose(dif, np.conj(np.swapaxes(dif, 0, 1)), rtol=0, atol=1e-12 * scale)
        np.testing.assert_allclose(sm, np.swapaxes(sm, 0, 1), rtol=0, atol=1e-12 * scale)
    tol = 1e-4 if higher_order else 2e-3
    for q in ("qtf_quadratic_control_surface_difference", "qtf_quadratic_control_surface_sum"):
        x, y = a.quantity(q), b.quantity(q)
        assert np.abs(x - y).max() <= tol * np.abs(y).max(), q


def test_the_control_surface_qtf_alone_returns_its_own_coefficients():
    # qtf_quadratic_control_surface alone: the pressure QTFs and both drifts are computed
    # for it (the hull values) and not returned; the coefficients are those of both methods
    cs = {"method": "cylinder", "radius": 2.0, "depth": 1.5, "inner_radius": 1.0, "panel_size": 0.3}
    body = {**lo_free_cylinder(), "control_surface": cs}
    opts = {"motion": True}
    a = model(body, W[:2], [0.0], options={**opts, "qtf_quadratic_control_surface": True}).solve()
    both = {**opts, "qtf_quadratic_pressure_integration": True, "qtf_quadratic_control_surface": True}
    b = model(body, W[:2], [0.0], options=both).solve()
    for q in ("qtf_quadratic_pressure_integration_difference", "drift_pressure_integration",
              "drift_control_surface"):  # fmt: skip
        assert a.quantity(q) is None, q
    assert b.quantity("qtf_quadratic_pressure_integration_difference") is not None
    np.testing.assert_array_equal(a.qtf.quadratic.control_surface.difference,
                                  b.qtf.quadratic.control_surface.difference)


def test_newman_approximation():
    # Newman's approximation from the drifts (theory §12.13): identities of its definition — the
    # diagonal is the drift (the same numbers), the matrix symmetric, geometric T_ij^2 = T_ii T_jj
    # where the two drifts have one sign and 0 where not, arithmetic their mean; of every drift
    # switched on, the others none
    w = np.array([1.5, 2.0, 2.5, 3.0])
    out = {"drift_pressure_integration": True, "drift_momentum_conservation": True, "qtf_newman": True}
    r = model(lo_free_cylinder(), w, [0.0, 45.0], options={"motion": True}, outputs=out).solve()
    assert r.qtf.newman.control_surface.difference is None
    for t, d in ((r.qtf.newman.pressure_integration.difference, r.drift.pressure_integration),
                 (r.qtf.newman.momentum_conservation.difference, r.drift.momentum_conservation)):  # fmt: skip
        assert t.shape[:3] == (4, 4, 2) and np.isrealobj(t)
        np.testing.assert_array_equal(t, np.swapaxes(t, 0, 1))
        dg = np.diagonal(t, axis1=0, axis2=1)  # [H, n, F]
        if d is not None:
            np.testing.assert_array_equal(np.moveaxis(dg, -1, 0), d.real)
        a, b = dg[..., :, None], dg[..., None, :]  # T_ii, T_jj as [H, n, F, F]
        tt = np.moveaxis(t, (0, 1), (-2, -1))
        same = a * b > 0
        np.testing.assert_allclose(tt[same] ** 2, (a * b)[same], rtol=1e-12)
        assert np.all(np.sign(tt[same]) == np.sign(np.broadcast_to(a, tt.shape)[same]))
        assert np.all(tt[~same] == 0.0)
    out["qtf_newman_average"] = "arithmetic"
    t = model(lo_free_cylinder(), w, [0.0, 45.0], options={"motion": True}, outputs=out).solve()
    t = t.qtf.newman.pressure_integration.difference
    dg = np.diagonal(t, axis1=0, axis2=1)
    np.testing.assert_allclose(np.moveaxis(t, (0, 1), (-2, -1)), 0.5 * (dg[..., :, None] + dg[..., None, :]),
                               rtol=1e-12, atol=0)  # fmt: skip


def test_newman_approximation_of_every_drift():
    # the three drifts it takes (theory §12.13): the diagonal of each is its drift, the same
    # numbers (the control-surface and the momentum drift asked for); without a drift switched on
    # it is refused (no drift is computed for it alone); with bidirectional headings it
    # takes the pairs of equal headings, the unidirectional drifts (to their rounding, 1e-12)
    w = np.array([1.5, 2.0, 2.5])
    cs = {"method": "cylinder", "radius": 2.0, "depth": 1.5, "inner_radius": 1.0, "panel_size": 0.3}
    body = {**lo_free_cylinder(), "control_surface": cs}
    methods = ["pressure_integration", "control_surface", "momentum_conservation"]

    drifts = {f"drift_{x}": True for x in methods}

    def solved(**out):
        return model(body, w, [0.0, 45.0], options={"motion": True}, outputs={"qtf_newman": True, **out})

    r = solved(**drifts).solve()
    with pytest.raises(ValueError, match="switch a drift on"):
        solved().check()
    bi = solved(**drifts, drift_bidirectional=True).solve()
    assert bi.drift.heading_pairs.shape == (3, 2)
    for name in methods:
        t = getattr(r.qtf.newman, name).difference
        d = getattr(r.drift, name)
        assert t.shape == (3, 3, 2, d.shape[2]) and np.isrealobj(t)
        np.testing.assert_array_equal(np.moveaxis(np.diagonal(t, axis1=0, axis2=1), -1, 0), d.real)
        np.testing.assert_allclose(getattr(bi.qtf.newman, name).difference, t, rtol=0, atol=1e-12 * np.abs(t).max())


@pytest.mark.parametrize("case", ["fixed", "translations"])
def test_rigid_modes_as_generalized_modes_have_the_rigid_qtfs(case):
    # as the drift (test_drift): the six rigid rows of both bodies and the generalized rows against
    # the rigid rows of their modes, every pair of frequencies and headings; fixed (held by 1e12:
    # 9e-9 observed) and floating with the translations as generalized modes (4e-16 observed)
    which = range(6) if case == "fixed" else (0, 1, 2)
    rigid = lo_free_cylinder(modes=[]) if case == "fixed" else lo_free_cylinder()
    held = rigid_modes_as_generalized(lo_free_cylinder(), which, ENV, fixed=case == "fixed")
    opts = {"motion": True, **PRESSURE, "qtf_bidirectional": True, "drift_bidirectional": True}
    a, b = (model(x, W[:3], [0.0, 30.0], environment=ENV, options=opts).solve() for x in (rigid, held))
    tol = 1e-7 if case == "fixed" else 1e-12
    # the bidirectional drift: the pair of two headings from the diagonal of the difference QTF
    # (on the coordinates: the six rigid rows, then the generalized ones)
    d, g = a.drift.pressure_integration, b.drift.pressure_integration
    for k, c in enumerate(which):
        assert np.abs(g[..., 6 + k] - d[..., c]).max() <= tol * np.abs(d).max()
    for kind in ("difference", "sum"):
        x = a.quantity(f"qtf_quadratic_pressure_integration_{kind}")
        g = b.quantity(f"qtf_quadratic_pressure_integration_{kind}")
        scale = np.abs(x).max()
        assert g.shape == x.shape[:3] + (6 + len(which),) and np.all(np.isfinite(g))
        assert np.abs(g[..., :6] - x).max() <= tol * scale
        for k, c in enumerate(which):
            assert np.abs(g[..., 6 + k] - x[..., c]).max() <= tol * scale


def test_the_generalized_forces_of_flexible_modes():
    # the bulging bottom and the bending sides of a floating cylinder: the generalized rows have the
    # identities of the rigid rows, exact by their definitions (the difference coefficient of a wave
    # with itself is their drift, Hermitian differences, symmetric sums, 1e-12)
    r = model(lo_flexible_cylinder(), W, [0.0, 30.0], environment=ENV, options={"motion": True, **PRESSURE}).solve()
    check_diagonal_and_symmetry(r)
    dif = r.qtf.quadratic.pressure_integration.difference[..., 6:]  # the generalized rows
    sm = r.qtf.quadratic.pressure_integration.sum[..., 6:]
    pd = r.drift.pressure_integration[..., 6:]
    scale = np.abs(pd).max()
    assert scale > 0 and np.all(np.isfinite(dif)) and np.all(np.isfinite(sm))
    diag = np.stack([dif[f, f] for f in range(len(W))])
    assert np.abs(diag - pd).max() <= 1e-12 * scale
    np.testing.assert_allclose(dif, np.conj(np.swapaxes(dif, 0, 1)), rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(sm, np.swapaxes(sm, 0, 1), rtol=0, atol=1e-12 * scale)
    assert np.abs(dif[0, 1] - dif[0, 0]).max() > 1e-3 * scale
    # a body's view: its six rigid rows and its own modes
    whole = r.qtf.quadratic.pressure_integration.sum
    assert np.shares_memory(r.bodies[0].qtf.quadratic.pressure_integration.sum, whole)
    assert r.bodies[0].drift.pressure_integration.shape[-1] == 8
