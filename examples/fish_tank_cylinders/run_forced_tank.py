"""The forced surge and pitch of a circular fish tank with no fluid outside: its added masses against
the closed-form interior series. Requires SciPy.

run_forced_tank.py: the tank alone (Fig. 1 of the paper), low-order panels:
    out/run_forced_tank/fish_tank_forced_d1.hbr (the model with the tank, with its results)

The case: J. N. Newman (2021), Linear motions of fish tanks, 36th International Workshop on Water
Waves and Floating Bodies (IWWWFB), 25-28 April 2021, Fig. 1: the added masses A11, A15 and A55 of the
fluid inside a circular cylindrical tank forced in surge and pitch, and the closed-form interior
solution of O. M. Faltinsen, A. N. Timokha (2009), Sloshing, Cambridge University Press, Section
5.4.4, as in the paper's appendix (eqs. 2-7).

The parameters taken: radius a = 1, draft d = 1 (the paper's drafts are 0.5, 0.715278, 1 and 2; set
DRAFT for another), the density and the gravity 1 (the paper's nondimensional results, K = omega^2 / g),
the origin and the centre of rotation at the centre of the tank's free surface, the wavenumbers
K = 0.5, 1, 2.5 and 3.5 (away from the first sloshing wavenumber K1 = k1 tanh(k1 d) = 1.7508, where
the panels' shift of K1 dominates the difference).

The model: the hull (side and bottom, normals out into the sea) and the tank (the same panels with
their vertices reversed, normals into the tank) are two sets of elements of one mesh. The results hold
the added mass of a body with its tanks, not that of a tank alone: the tank's is the difference of
two solves of the same hull, with the tank and without it (the two fluids do not influence each
other, so the parts of the sea cancel). The mesh: a quadrant, 16 panels around, 24 down the side and
14 across the bottom, cosine-spaced; the interior surface "graded" for the exterior problem.

The series, derived here for the potentials per unit velocity, the normal into the tank fluid and
A_ij = -rho times the integral of n_i phi_j over the wall (physics section 3). With k_n the zeros of
J1', a_n = 2 / (k_n^2 - 1) and C, S, T the cosh, sinh and tanh of k_n d:
- surge: c_n = K a_n / (k_n S - K C), A11 = pi (d + sum c_n S / k_n);
- pitch about the origin: beta_n = -2 a_n / k_n, alpha_n = [a_n + beta_n (k_n C - K S)] / (K C - k_n S),
  A55 = pi [d^3/3 - d/4 + sum ((alpha_n (2 - C) - beta_n S) / k_n^2 + beta_n d / k_n)],
  A15 = pi [-d^2/2 + sum (alpha_n S + beta_n (C - 1)) / k_n];
200 terms, written with exp(-k_n d) so that nothing overflows.

What the script prints: A11, A15 and A55 per unit density of the panels and of the series at the four
wavenumbers, and the check: their largest difference relative to the largest |value| of each
quantity. The full comparison is in the validation report,
docs/validation/cases/fish_tank_cylinders.md.
"""

from pathlib import Path

import numpy as np
from scipy.special import jnp_zeros

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_forced_tank"  # the outputs of this script
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
DENSITY, GRAVITY = 1.0, 1.0  # nondimensional: lengths in units of the radius
RADIUS, DRAFT = 1.0, 1.0
WAVENUMBERS = np.array([0.5, 1.0, 2.5, 3.5])  # K = omega^2 / g
N_THETA, N_WALL, N_BOTTOM = 16, 24, 14  # panels around a quadrant, down the side, across the bottom
N_TERMS = 200  # the terms of the series
SURGE, PITCH = 0, 4  # the coordinates of surge and pitch among the six of the body
TOLERANCE = 2e-2


def spacing(n, fine):
    """n + 1 fractions from 0 to 1, cosine-spaced: small at both ends ("both") or at the start ("start")."""
    s = np.linspace(0.0, 1.0, n + 1)
    return 0.5 * (1.0 - np.cos(np.pi * s)) if fine == "both" else 1.0 - np.cos(0.5 * np.pi * s)


def tank_panels(draft):
    """The panels of a quadrant of the cylinder and the elements of its two faces: "hull" the side
    downward and the bottom inward (normals out into the sea), "tank" the same panels with their
    vertices reversed (normals into the tank); the panels at r = 0 are triangles."""
    r = np.r_[np.full(N_WALL + 1, RADIUS), RADIUS * (1.0 - spacing(N_BOTTOM, "start")[1:])]
    z = np.r_[-draft * spacing(N_WALL, "both"), np.full(N_BOTTOM, -draft)]
    phi = np.linspace(0.0, 0.5 * np.pi, N_THETA + 1)

    def p(i, j):
        return [r[i] * np.cos(phi[j]), r[i] * np.sin(phi[j]), z[i]]

    hull = [[p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)] for i in range(len(r) - 1) for j in range(N_THETA)]
    n = len(hull)
    return np.array(hull + [q[::-1] for q in hull]), {"hull": (1, n), "tank": (n + 1, 2 * n)}


def build(tank=True) -> hb.Model:
    """The tank forced in surge and pitch (radiation only): the hull with the tank inside or alone."""
    vertices, records = tank_panels(DRAFT)
    m = hb.Model()
    m.meshes.add(
        name="tank",
        kind="panels",
        vertices=vertices,
        reference_length=RADIUS,
        symmetry="xz_and_yz",
        label=f"fish tank a = 1, d = {DRAFT:g}: the hull and the tank faces of the wall",
    )
    body = m.bodies.add(
        name="tank",
        mesh="tank",
        ranges=[records["hull"]],
        modes=["surge", "pitch"],
        interior_surface="graded",
    )
    if tank:
        body.tanks.add(
            name="interior",
            mesh="tank",
            ranges=[records["tank"]],
            density_ratio=1.0,
            surface_level=0.0,
        )
    m.environment.update(
        depth=np.inf,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=np.sqrt(GRAVITY * WAVENUMBERS),
        headings=[],
    )
    m.options.update(
        frequency_threads=FREQUENCY_THREADS,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=True,
        diffraction=False,
        motion=False,
    )
    return m


def series(draft, k):
    """The closed-form interior solution (the docstring) at the wavenumbers k: A11, A15 and A55 per unit
    density, each [F]."""
    kn = jnp_zeros(1, N_TERMS)[:, None]
    big_k = np.atleast_1d(k)[None, :]
    e = np.exp(-kn * draft)
    t = np.tanh(kn * draft)
    sech = 2.0 * e / (1.0 + e * e)
    a = 2.0 / (kn**2 - 1.0)
    q = big_k * a / (kn * t - big_k)  # c_n C, finite
    a11 = np.pi * (draft + np.sum(q * t / kn, axis=0))
    beta = -2.0 * a / kn
    delta_c = (a + beta * (kn + big_k) * e) / (big_k - kn * t)  # (alpha_n + beta_n) C, finite
    terms = (-2.0 * beta + beta * e + 2.0 * delta_c * sech - delta_c) / kn**2 + beta * draft / kn
    a55 = np.pi * (draft**3 / 3 - draft / 4 + np.sum(terms, axis=0))
    a15 = np.pi * (-(draft**2) / 2 + np.sum((beta * e - beta + t * delta_c) / kn, axis=0))
    return np.column_stack([a11, a15, a55])


def added_mass(r):
    """The added mass on (surge, pitch) [F, 2, 2]."""
    columns = [list(r.radiation_mode_ids).index(i + 1) for i in (SURGE, PITCH)]
    return r.added_mass[:, [SURGE, PITCH]][:, :, columns]


def main():
    m = build()
    print(m, "| route:", m.route())
    full = m.solve()
    path = OUT / f"fish_tank_forced_d{DRAFT:g}.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(full.summary())
    hull = build(tank=False).solve()
    a = (added_mass(full) - added_mass(hull)) / DENSITY  # the tank's added mass alone
    panels = np.column_stack([a[:, 0, 0], a[:, 0, 1], a[:, 1, 1]])
    ref = series(DRAFT, WAVENUMBERS)
    print(f"the tank alone, d = {DRAFT:g}: HydroBEM / the series of Faltinsen & Timokha (2009)")
    print(f"{'K':>5} " + " ".join(f"{name:>17}" for name in ("A11", "A15", "A55")))
    for k, p, s in zip(WAVENUMBERS, panels, ref, strict=True):
        print(f"{k:5.2f} " + " ".join(f"{x:8.4f} {y:8.4f}" for x, y in zip(p, s, strict=True)))
    diff = (np.abs(panels - ref).max(axis=0) / np.abs(ref).max(axis=0)).max()
    print(f"check: largest difference / largest |value| {diff:.1e} <= {TOLERANCE:g}:",
          "ok" if diff <= TOLERANCE else "NOT MET")  # fmt: skip
    print("saved:", path)


if __name__ == "__main__":
    main()
