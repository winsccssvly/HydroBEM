"""Wavemakers in the walls (theory §10.3).

A body lying wholly in a wall plane radiates from the known sources of its normal
velocity: the work of a hinged paddle equals the energy flux of its waves through a far
quarter circle (the energy balance, both methods), and the problem options are checked. The low
and the higher order against each other is validation, not a test. Meshes are procedural
(tests/python/meshes.py)."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import flat, surface
from meshes import face_panels
from problems import model

DEPTH, ZH = 4.0, -2.0


def paddle_mode():
    """A hinged paddle: u = (z - Z_h, 0, 0), the rotation about its lower edge."""
    return {"name": "paddle", "pieces": [{"u": [[1.0, 0, 0, 1], [-ZH, 0, 0, 0]]}]}


def paddle(higher_order):
    """The paddle x = 0, 2 <= y <= 3, Z_h <= z <= 0 (normal +x, into the fluid)."""
    v = np.array([[[0.0, 2.0, ZH], [0.0, 3.0, ZH], [0.0, 3.0, 0.0], [0.0, 2.0, 0.0]]])
    if higher_order:
        mesh = flat(v, [(2, 2, 3, 3, 3, 4)])
    else:
        mesh = surface(face_panels((0.0, 2.0, ZH), (0.0, 1.0, 0.0), (0.0, 0.0, -ZH), 8, 16, (1.0, 0.0, 0.0)))
    return {"name": "paddle", "mesh": mesh, "modes": [], "generalized_modes": [paddle_mode()]}


def solve(higher_order, omegas, points, **kw):
    opts = {"radiation": True, "diffraction": False, "field_pressure": True, "domain": "orthogonal_walls",
            "higher_order": higher_order, **kw}  # fmt: skip
    return model(paddle(higher_order), omegas, [], field_points=points, environment={"depth": DEPTH},
                 options=opts).solve()  # fmt: skip


def wavenumber(w):
    k = w * w / 9.80665
    for _ in range(60):
        k = w * w / (9.80665 * math.tanh(k * DEPTH))
    return k


@pytest.mark.parametrize("higher_order", [False, True])
def test_paddle_work_equals_the_radiated_energy_flux(higher_order):
    # the energy balance (Newman 1977): the work of the mode, (1/2) w^2 B, is the flux of its waves,
    # (1/2) rho g |eta|^2 c_g over the quarter circle. 2e-3: the plane-wave form of the flux at a
    # finite R (kR = 24 and 81, the O(1/(kR)^2) of the Hankel function's asymptote; the evanescent
    # modes negligible there) and the midpoint sum of 400 points (3e-4 observed); the discretization
    # of the paddle enters both sides alike
    R, n = 80.0, 400
    th = (np.arange(n) + 0.5) * (np.pi / 2) / n
    pts = np.stack([R * np.cos(th), R * np.sin(th), np.zeros(n)], axis=1)
    for T in (2.0, 4.0):
        w = 2 * math.pi / T
        k = wavenumber(w)
        cg = 0.5 * w / k * (1 + 2 * k * DEPTH / math.sinh(2 * k * DEPTH))
        r = solve(higher_order, [w], pts)
        eta = np.asarray(r.field_points.pressure_radiation)[0, :, 0]  # elevation per unit rotation
        flux = np.sum(0.5 * 1000.0 * 9.80665 * np.abs(eta) ** 2 * cg * R * (np.pi / 2) / n)
        work = 0.5 * w * w * np.asarray(r.radiation_damping)[0].flat[-1]
        assert abs(flux / work - 1.0) <= 2e-3, (T, flux, work)


def test_wavemaker_options_are_checked():
    w = [2.0]
    with pytest.raises(hb.UnsupportedProblemError, match="diffraction off"):
        model(paddle(False), w, [0.0], environment={"depth": DEPTH}, options={"domain": "orthogonal_walls"}).solve()
    other = {"name": "plate", "mesh": surface(face_panels((2.0, 2.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2, 2,
                                                          (0.0, 0.0, -1.0))), "modes": ["heave"]}  # fmt: skip
    with pytest.raises(hb.UnsupportedProblemError, match="other bodies"):
        model([paddle(False), other], w, [], environment={"depth": DEPTH},
              options={"diffraction": False, "domain": "orthogonal_walls"}).solve()  # fmt: skip
    half = np.array([[[0.0, 2.0, ZH], [0.5, 3.0, ZH], [0.5, 3.0, 0.0], [0.0, 2.0, 0.0]]])
    with pytest.raises(ValueError, match="fluid side of the walls"):
        model({"mesh": surface(half), "modes": ["surge"]}, w, [], environment={"depth": DEPTH},
              options={"diffraction": False, "domain": "orthogonal_walls"}).solve()  # fmt: skip
