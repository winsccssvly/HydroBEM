"""A freely floating circular fish tank in waves: surge, pitch and the sloshing of the fluid inside.

run.py: the tank free in surge and pitch in waves (Fig. 2 of the paper), low-order panels:
    out/run/fish_tank_d2.hbr (the model with its results)

The case: J. N. Newman (2021), Linear motions of fish tanks, 36th International Workshop on Water
Waves and Floating Bodies (IWWWFB), 25-28 April 2021: circular cylindrical tanks with vertical axes
floating in infinite depth, the tank's wall of zero thickness separating the fluid inside (the same
density, its free surface at the same height) from the sea outside, so that there is neither a
rigid-body mass nor a hydrostatic restoring moment; surge, pitch and the first sloshing mode.

The parameters taken:
- radius a = 1, draft d = 2 (the paper's drafts are 0.5, 0.715278, 1 and 2; set DRAFT for another),
  infinite depth, the density and the gravity 1, so that the results are the paper's nondimensional
  ones (lengths in units of a, K = omega^2 / g the wavenumber outside, the pitch in radians);
- the origin and the centre of rotation at the centre of the interior free surface;
- incident waves along +x of unit amplitude; the wavenumbers K = 0.05 and every 0.002 from 1.801 to
  1.899 around the first sloshing wavenumber K1 = k1 tanh(k1 d) = 1.8389 (none at K1 itself, where
  the coefficients of the tank are infinite);
- the sloshing amplitude: the elevation of the interior free surface at x = 0.9, y = 0.

The model: the hull (side and bottom, normals out into the sea) and the tank (the same panels with
their vertices reversed, normals into the tank) are two sets of elements of one mesh on the same
cylinder: the wall of zero thickness is two coincident surfaces with opposite normals. The tank has
the density of the sea (density_ratio 1) and its free surface at z = 0; its added mass and
hydrostatics are added to the body's, and the restoring of surge and pitch cancels to zero, as in the
paper. The body itself is massless (mass 0, a zero inertia), and the engine solves its motions with
the inertia of the water alone, (C - omega^2 A + i omega B) xi = X (physics section 8); the sloshing
amplitude is the sum of the motions times the tank's elevation per unit motion.

The mesh: a quadrant (the planes x = 0 and y = 0 are symmetry planes), 16 panels around, 32 down the
side and 14 across the bottom, cosine-spaced; the interior surface "graded" removes the irregular
frequencies of the exterior problem. The polygon of 64 sides puts the panel model's K1 0.003 above
the exact one (run_higher_order.py: on it).

What the script prints: the surge RAO at K = 0.05 (it tends to 1 as K -> 0) and the check: the zeros
of xi1 and xi5 with both modes free just above K1 against the paper's values for d = 2 (Section 3:
K = 1.841 and 1.845). The full comparison is in the validation report,
docs/validation/cases/fish_tank_cylinders.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
DENSITY, GRAVITY = 1.0, 1.0  # nondimensional: lengths in units of the radius
RADIUS, DRAFT = 1.0, 2.0
K1_ROOT = 1.8411837813406593  # the first zero of J1' (Abramowitz & Stegun 1964, Table 9.5)
WAVENUMBERS = np.r_[0.05, 1.801 + 0.002 * np.arange(50)]  # K = omega^2 / g
SLOSHING_POINT = [0.9, 0.0, 0.0]  # the point of the sloshing amplitude in the tank's free surface
N_THETA, N_WALL, N_BOTTOM = 16, 32, 14  # panels around a quadrant, down the side, across the bottom
SURGE, PITCH = 0, 4  # the coordinates of surge and pitch among the six of the body
PAPER_ZEROS = (1.841, 1.845)  # Newman (2021), Section 3, d = 2: the zeros of xi1 and xi5 with both free
TOLERANCE = 5e-3


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


def build() -> hb.Model:
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
        center_of_gravity=[0.0, 0.0, 0.0],
        mass=0.0,  # massless, its inertia zero (the radii of gyration zero by default)
        interior_surface="graded",
    )
    body.tanks.add(
        name="interior",
        mesh="tank",
        ranges=[records["tank"]],
        density_ratio=1.0,
        surface_level=0.0,
        field_points=np.array([SLOSHING_POINT]),
    )
    m.environment.update(
        depth=np.inf,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=np.sqrt(GRAVITY * WAVENUMBERS),
        headings=[0.0],
    )
    m.options.update(
        frequency_threads=FREQUENCY_THREADS,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=True,
        diffraction=True,
        motion=True,
        field_pressure=True,
    )
    return m


def first_zero(k, xi):
    """The first zero of the complex RAO xi on k[1:]: the sign change of its component along its phase at
    k[1] (the phase flips by 180 degrees at the zero), interpolated linearly."""
    y = (xi[1:] * np.exp(-1j * np.angle(xi[1]))).real
    i = np.nonzero(np.sign(y[:-1]) != np.sign(y[1:]))[0][0]
    return k[1 + i] - y[i] * (k[2 + i] - k[1 + i]) / (y[i + 1] - y[i])


def main():
    m = build()
    print(m, "| route:", m.route())
    r = m.solve()
    path = OUT / f"fish_tank_d{DRAFT:g}.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(r.summary())
    c = r.hydrostatics.stiffness[0]
    print(f"restoring of the hull with the tank: c11 = {c[0, 0]:.1e}, c15 = {c[0, 4]:.1e}, c55 = {c[4, 4]:.1e}")

    xi = r.bodies["tank"].motion[:, 0, [SURGE, PITCH]]  # the RAOs (xi1, xi5) per unit wave amplitude
    columns = [list(r.radiation_mode_ids).index(i + 1) for i in (SURGE, PITCH)]
    # the elevation of the tank's free surface per unit surge and pitch (p / (rho g) with rho = g = 1)
    eta = r.bodies["tank"].tanks["interior"].pressure_radiation[:, 0, columns]
    zeta = np.sum(xi * eta, axis=1)
    print(f"K = {WAVENUMBERS[0]:g}: |xi1| = {abs(xi[0, 0]):.3f}, |xi5| = {abs(xi[0, 1]):.3f},",
          f"|zeta| = {abs(zeta[0]):.3f}")  # fmt: skip
    print(f"K1 = {K1_ROOT * np.tanh(K1_ROOT * DRAFT):.4f} (exact)")
    zeros = [first_zero(WAVENUMBERS, xi[:, j]) for j in (0, 1)]
    diff = max(abs(z / p - 1.0) for z, p in zip(zeros, PAPER_ZEROS, strict=True))
    for name, z, p in zip(("xi1", "xi5"), zeros, PAPER_ZEROS, strict=True):
        print(f"both free, the zero of {name}: K = {z:.4f} (Newman 2021: {p})")
    print(f"check: largest relative difference of the zeros {diff:.1e} <= {TOLERANCE:g}:",
          "ok" if diff <= TOLERANCE else "NOT MET")  # fmt: skip
    print("saved:", path)


if __name__ == "__main__":
    main()
