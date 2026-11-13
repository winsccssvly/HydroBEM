"""Prescribed-pressure surfaces (air cushions, chambers), low order; the
higher-order pressure patches are in test_higher_order.py.

theory §9.3 / test-plan B: a pressure surface in z = 0 without applied pressure is the free
surface itself (rigid-mode results unchanged, its elevation is the free-surface
elevation); Haskind and pressure exciting forces agree for the pressure modes;
the capture width of an axisymmetric chamber is 1/k (Evans 1982); the total
and scattered formulations agree; symmetry sectors equal the full solve; the
restoring conventions; routes that are not available are rejected.
Meshes are procedural (tests/python/meshes.py)."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import acv_body, part, surface
from meshes import acv_panels, box_panels, cylinder_panels, moonpool_lid_panels, moonpool_panels
from problems import flat_surface, model, peak, view

RHO, G = 1000.0, 9.80665
ENV = {"density": RHO, "gravity": G}


W = 2 * np.pi / np.array([4.0, 6.0, 9.0])


def test_zero_pressure_surface_in_z0_is_the_free_surface():
    # the free surface of the annular moonpool (moonpool_panels) as a pressure surface in z = 0: an
    # identity of the problem (theory §9.3) and of the discrete equations, 1e-5: float32 round-off
    # (7.8e-8 measured); the elevation of the surface against the field points, two evaluations of
    # the same potential, 1e-4 (5.5e-6 measured)

    hull = moonpool_panels(n=12)
    lid = moonpool_lid_panels(n=12)[:, [0, 3, 2, 1]]  # downward normals
    m = surface(np.concatenate([hull, lid]), symmetry="xz_and_yz")
    surf = {"mesh": part(m, len(hull) + 1, len(hull) + len(lid)), "pressure_modes": [{"name": "p", "shape": "uniform"}]}
    w = np.sqrt(np.array([0.8, 1.5, 2.4]) * G)
    opts = {"diffraction_formulation": "total"}
    a = model({"mesh": surface(hull, symmetry="xz_and_yz")}, w, [0.0], environment=ENV, options=opts).solve()
    b = model({"mesh": part(m, 1, len(hull)), "pressure_surfaces": [surf]}, w, [0.0], environment=ENV,
              options=opts).solve()  # fmt: skip
    for q in ("added_mass", "radiation_damping"):
        x, y = view(a, q), view(b, q)[:, :6, :6]
        assert peak(x - y) <= 1e-5 * peak(x), q
    assert peak(a.excitation.diffraction - b.excitation.diffraction[:, :, :6]) <= 1e-5 * peak(a.excitation.diffraction)
    # the diffraction elevation of the surface is the free-surface elevation there
    pts = b.quantity("pressure_surfaces")
    f = model({"mesh": surface(hull, symmetry="xz_and_yz")}, w, [0.0], field_points=pts, environment=ENV,
              options={**opts, "field_pressure": True}).solve()  # fmt: skip
    eta = f.field_points.pressure_diffraction[:, :, 0]
    assert peak(eta - b.quantity("pressure_surfaces_elevation_diffraction")[:, :, 0]) <= 1e-4 * peak(eta)


def test_haskind_equals_pressure_force_for_pressure_modes():
    # the Haskind relation for the pressure modes (theory §9.3); 2e-2: the pressure integration over
    # the pressure panels, whose velocity is singular at their edges on the hull, on the coarse
    # cushion (4.6e-3 sign_x, 1.1e-3 uniform measured)
    r = model(acv_body(modes=("sign_x", "uniform")), W, [150.0], environment=ENV,
              options={"diffraction_formulation": "total"}).solve()  # fmt: skip
    hk = r.excitation.haskind  # [F, H, K]
    for i in (6, 7):
        x = r.excitation.diffraction[:, 0, i]
        assert peak(hk[:, 0, i] - x) <= 2e-2 * peak(x), i
    # damping of the pressure modes is non-negative (no energy created)
    assert np.all(np.diagonal(r.radiation_damping, axis1=1, axis2=2)[:, 6:] >= 0)


def test_capture_width_of_an_axisymmetric_chamber():
    """The restrained annular tube (moonpool_panels: radii 1 and 2, draft 1) with its inner free
    surface as a uniform pressure surface, a chamber, in deep water at ka = 0.4, 0.7, 1.0 (a the
    outer radius): the largest mean power of the chamber with the optimum turbine, |X|^2 / (8 B) per
    unit amplitude squared, is the power of the incident wave over 1/k of crest, k P_max / P_w = 1,
    P_w = rho g c_g / 2 (Evans 1982, J. Fluid Mech. 114, 481-499: the capture width of an
    axisymmetric pressure distribution; the damping-exciting force relation of its one radiating
    order). The bound is the discretization, B from the radiation and X from the diffraction
    solution on 16 panels a quadrant around: 1e-2 (6.7e-3 found at ka = 1, 3.0e-3 at 0.7; 1.9e-2 on
    8 panels); with the Haskind X, from the far field of the same radiation solution, 5e-3 (2.0e-3
    found)."""
    n = 16
    hull = moonpool_panels(n=n)
    lid = moonpool_lid_panels(n=n)[:, [0, 3, 2, 1]]  # downward normals
    m = surface(np.concatenate([hull, lid]), symmetry="xz_and_yz")
    surf = {"mesh": part(m, len(hull) + 1, len(hull) + len(lid)), "pressure_modes": [{"name": "p", "shape": "uniform"}]}
    k = np.array([0.4, 0.7, 1.0]) / 2.0
    w = np.sqrt(G * k)
    body = {"mesh": part(m, 1, len(hull)), "modes": [], "pressure_surfaces": [surf], "interior_surface": "none"}
    r = model(body, w, [0.0], environment={**ENV, "depth": math.inf}).solve()
    b = r.coefficient("radiation_damping", ("body", "p"), ("body", "p"))
    power = 0.5 * RHO * G * G / (2 * w)  # c_g = g / (2 w)
    for name, bound in (("excitation_diffraction", 1e-2), ("excitation_haskind", 5e-3)):
        x = r.coefficient(name, ("body", "p"))[:, 0]
        ratio = k * np.abs(x) ** 2 / (8 * b) / power
        assert np.all(np.abs(ratio - 1) <= bound), (name, ratio)


def test_total_and_scattered_formulations_agree():
    # one continuous problem in two unknowns; 2e-2: their discretization differs most at the
    # singular edges of the pressure surface on the coarse cushion (7.8e-3 measured); the radiation
    # problem is the same equations, bit-identical
    out = [model(acv_body(), W, [180.0], environment=ENV, options={"diffraction_formulation": f}).solve()
           for f in ("total", "scattered")]  # fmt: skip
    x0, x1 = out[0].excitation.diffraction, out[1].excitation.diffraction
    assert peak(x0 - x1) <= 2e-2 * peak(x0)
    np.testing.assert_array_equal(out[0].added_mass, out[1].added_mass)


def test_symmetry_sectors_equal_full_solve():
    # exact for the discrete equations (theory §6.3); 1e-5: float32 round-off (2.9e-7 measured);
    # the restoring is the same sums of the panels in another order
    res = [model(acv_body(modes=("sign_x", "uniform", "sign_y"), quarter=q), W[:2], [30.0], environment=ENV).solve()
           for q in (True, False)]  # fmt: skip
    for name in ("added_mass", "radiation_damping", "excitation_diffraction"):
        a, b = view(res[0], name), view(res[1], name)
        assert peak(a - b) <= 1e-5 * peak(b), name
    for key in ("mode_restoring",):
        np.testing.assert_allclose(view(res[0], key), view(res[1], key), rtol=1e-10, atol=1e-6)


def test_restoring_conventions():
    # the closed forms of the restoring of the modes over the cushion of 4 x 9 x 4 m (theory §9.3),
    # exact but for rounding (1e-12); the mode stiffness and damping as given (input rules)
    rg = RHO * G
    area = 4 * 9.0 * 4.0  # the cushion surface
    for conv in ("variational", "numeric_compatibility"):
        b = acv_body(convention=conv)
        r = model(b, W[:1], [180.0], environment=ENV).solve()
        C = r.mode_restoring
        sgn = -1.0 if conv == "variational" else 1.0
        assert C[6, 6] == pytest.approx(sgn * rg * area, rel=1e-12)
        assert C[2, 6] == pytest.approx(rg * area, rel=1e-12)  # roof force in the heave row
        assert C[6, 2] == pytest.approx(rg * area if conv == "variational" else 0.0, abs=1e-6 * rg * area)
        assert abs(C[3, 6]) <= 1e-9 * rg * area and abs(C[4, 6]) <= 1e-9 * rg * area  # symmetric chamber
    r = model(acv_body(chamber=False), W[:1], [180.0], environment=ENV).solve()
    assert r.mode_restoring[2, 6] == 0.0
    # mode stiffness and damping on the diagonal
    hull, ps = acv_panels()
    m = surface(np.concatenate([hull, ps]), symmetry="xz_and_yz")
    surf = {"mesh": part(m, len(hull) + 1, len(hull) + len(ps)),
            "pressure_modes": [{"name": "p", "shape": "uniform", "additional_stiffness": 5e5,
                                "additional_damping": 2e3}]}  # fmt: skip
    r = model({"mesh": part(m, 1, len(hull)), "pressure_surfaces": [surf]}, W[:1], [180.0], environment=ENV).solve()
    assert r.mode_restoring[6, 6] == pytest.approx(-rg * area + 5e5, rel=1e-12)
    assert r.pressure_mode_damping[6, 6] == 2e3


def test_unavailable_pressure_routes_are_rejected():
    # input rules
    def check(m, what):
        with pytest.raises(hb.UnsupportedProblemError, match=what):
            m.route()

    check(model(acv_body(), [1.0, math.inf], [0.0], environment=ENV), "omega = inf")
    m = model(acv_body(), [1.0], [0.0], environment=ENV)
    with pytest.raises(TypeError, match="shape must be a string"):  # named shapes only
        m.bodies["acv"].pressure_surfaces[0].pressure_modes[0].shape = lambda x, y: np.ones_like(x)
    # an interior surface on another body is a separate surface
    cyl = surface(cylinder_panels(1.0, 0.5, 4, 2, 2), symmetry="xz_and_yz")
    two = [acv_body(), {"name": "c", "mesh": cyl, "position": [20, 0, 0], "interior_surface": {"method": "waterline"}}]
    model(two, [1.0], [0.0], environment=ENV).route()


def test_pressure_surface_with_an_interior_surface():
    """An interior free surface on a body with pressure surfaces. Away from the irregular
    frequencies it leaves the solution (an identity of the continuous problem): 1e-3 of the peaks
    at a long period, the discretization of the interior surface (3.8e-5 measured)."""
    w = [0.8]
    plain = model(acv_body(), w, [0.0], environment=ENV).solve()
    body = acv_body(**flat_surface("interior_surface", {"method": "projected"}))
    irr = model(body, w, [0.0], environment=ENV).solve()
    assert peak(irr.added_mass - plain.added_mass) <= 1e-3 * peak(plain.added_mass)
    assert peak(irr.excitation.diffraction - plain.excitation.diffraction) <= 1e-3 * peak(plain.excitation.diffraction)


def test_pressure_and_interior_surfaces_given_either_way_up():
    # the elements of a pressure surface or of an interior surface may point either way, also mixed
    # within one surface (as lids): the engine turns them to its convention, the same equations, so
    # the results bit for bit
    flip = lambda v, sel: np.where(sel[:, None, None], v[:, [0, 3, 2, 1]], v)  # noqa: E731
    hull, ps = acv_panels(n=2)

    def acv(sel):
        m = surface(np.concatenate([hull, flip(ps, sel)]), symmetry="xz_and_yz")
        surf = {"mesh": part(m, len(hull) + 1, len(hull) + len(ps)), "pressure_modes": [{"name": "p0",
                                                                                         "shape": "uniform"}]}
        body = {"name": "acv", "mesh": part(m, 1, len(hull)), "pressure_surfaces": [surf]}
        return model(body, [0.8], [0.0], environment=ENV).solve()

    box = box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True, lid=False)
    lid = box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True, lid=True)[len(box) :]

    def boxed(sel):
        m = surface(np.concatenate([box, flip(lid, sel)]), symmetry="xz_and_yz")
        body = {"mesh": part(m, 1, len(box)), "interior_surface": part(m, len(box) + 1, len(box) + len(lid))}
        return model(body, [4.0], [0.0], environment=ENV).solve()

    for make, n in ((acv, len(ps)), (boxed, len(lid))):
        ref = make(np.zeros(n, bool))
        for sel in (np.arange(n) % 2 == 0, np.ones(n, bool)):  # mixed, all the other way
            r = make(sel)
            np.testing.assert_array_equal(r.added_mass, ref.added_mass)
            np.testing.assert_array_equal(r.excitation.diffraction, ref.excitation.diffraction)
