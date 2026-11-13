"""The forced surge and pitch of a circular fish tank whose free surface is open, half covered by a
deck or fully covered: plates of the tank's free surface.

run_covered_tank.py: the tank of run_forced_tank.py (a = 1, d = 1), low-order panels:
    out/run_covered_tank/fish_tank_covered.hbr (the model with the half-covered tank, with its results)

The case: the tank of J. N. Newman (2021), Linear motions of fish tanks, 36th International Workshop
on Water Waves and Floating Bodies (IWWWFB), 25-28 April 2021, Fig. 1, forced in surge and pitch,
with a rigid plate lying on the tank's free surface: none (the open tank of run_forced_tank.py), an
annular deck from the wall to the radius a / sqrt(2) (half the free surface covered, the water open
in the middle) and a full cover. The plates are free-surface plates of the tank's problem
(tank.free_surface_plates): massless, rigid, moving with the body, with the tank fluid below them
only (theory §9.7 and §10.2 of the documentation). A full cover makes the tank a closed container
filled with water: in surge the fluid then moves as a rigid body (the potential phi = x meets the
condition on every wall and on the cover), and the tank's added masses are those of its fluid mass
M = rho pi a^2 d at its centre z = -d/2, A11 = M and A15 = -M d/2, independent of the frequency.
The half cover leaves the fluid a smaller free surface to slosh on: the added masses change sign
through the first sloshing resonance between K = 2.5 and 3.5, above that of the open tank
(K1 = k1 tanh(k1 d) = 1.7508, k1 the first zero of J1').

The parameters taken: radius a = 1, draft d = 1, the density and the gravity 1 (lengths in units
of the radius, K = omega^2 / g), the wavenumbers K = 0.5, 1, 2.5 and 3.5; the hull and the tank two
sets of elements of one mesh (a quadrant, 16 panels around, 24 down the side and 14 across the
bottom, cosine-spaced, as in run_forced_tank.py); the plates meshes of their own in z = 0, 16
panels around (their edges on the wall's panels) and 6 rings. The tank's added mass is the
difference of two solves of the same hull, with the tank and without it.

What the script prints: A11, A15 and A55 per unit density of the tank alone, open, half covered and
covered, at the four wavenumbers, and the check: A11 and A15 of the covered tank against M and
-M d/2, the largest difference relative to M.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_covered_tank"  # the outputs of this script
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
DENSITY, GRAVITY = 1.0, 1.0  # nondimensional: lengths in units of the radius
RADIUS, DRAFT = 1.0, 1.0
WAVENUMBERS = np.array([0.5, 1.0, 2.5, 3.5])  # K = omega^2 / g
N_THETA, N_WALL, N_BOTTOM = 16, 24, 14  # panels around a quadrant, down the side, across the bottom
N_PLATE = 6  # rings of a plate
COVERS = {"open": None, "half covered": RADIUS / np.sqrt(2.0), "covered": 0.0}  # the inner radius of the plate
SURGE, PITCH = 0, 4  # the coordinates of surge and pitch among the six of the body
TOLERANCE = 1e-2  # of the fluid mass


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


def plate_panels(inner):
    """A quadrant of the annulus inner < r < a in z = 0 (a disc for inner = 0), N_THETA panels around
    and N_PLATE rings (either orientation: a plate is turned to face the fluid below it)."""
    phi = np.linspace(0.0, 0.5 * np.pi, N_THETA + 1)
    r = np.linspace(inner, RADIUS, N_PLATE + 1)

    def p(i, j):
        return [r[i] * np.cos(phi[j]), r[i] * np.sin(phi[j]), 0.0]

    return np.array([[p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)]
                     for i in range(N_PLATE) for j in range(N_THETA)])  # fmt: skip


def build(tank=True, inner=None) -> hb.Model:
    """The tank forced in surge and pitch (radiation only): the hull with the tank inside, its free
    surface covered from the wall to the radius inner (None: open), or the hull alone."""
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
        interior = body.tanks.add(
            name="interior",
            mesh="tank",
            ranges=[records["tank"]],
            density_ratio=1.0,
            surface_level=0.0,
        )
        if inner is not None:
            m.meshes.add(
                name="deck",
                kind="panels",
                vertices=plate_panels(inner),
                reference_length=RADIUS,
                symmetry="xz_and_yz",
                label=f"the plate {inner:.4f} < r < 1 on the tank's free surface",
            )
            interior.free_surface_plates.add(
                mesh="deck",
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


def added_mass(r):
    """The added mass on (surge, pitch) [F, 2, 2]."""
    columns = [list(r.radiation_mode_ids).index(i + 1) for i in (SURGE, PITCH)]
    return r.added_mass[:, [SURGE, PITCH]][:, :, columns]


def main():
    hull = added_mass(build(tank=False).solve())
    tanks = {}
    for name, inner in COVERS.items():
        m = build(inner=inner)
        if name == "half covered":
            print(m, "| route:", m.route())
        results = m.solve()
        if name == "half covered":
            path = OUT / "fish_tank_covered.hbr"
            m.save_results(
                path,
                overwrite=True,
            )
            print(results.summary())
        a = (added_mass(results) - hull) / DENSITY  # the tank's added mass alone
        tanks[name] = np.column_stack([a[:, 0, 0], a[:, 0, 1], a[:, 1, 1]])
    print(f"the tank alone, d = {DRAFT:g}, per unit density: open / half covered / covered")
    print(f"{'K':>5} " + " ".join(f"{name:>26}" for name in ("A11", "A15", "A55")))
    for k, rows in zip(WAVENUMBERS, zip(*tanks.values(), strict=True), strict=True):
        print(f"{k:5.2f} " + " ".join(" ".join(f"{row[q]:8.4f}" for row in rows) for q in range(3)))
    mass = np.pi * RADIUS**2 * DRAFT  # the fluid mass per unit density
    covered = tanks["covered"]
    diff = max(np.abs(covered[:, 0] - mass).max(), np.abs(covered[:, 1] + mass * DRAFT / 2).max()) / mass
    print(f"covered: M = pi a^2 d = {mass:.4f}, -M d/2 = {-mass * DRAFT / 2:.4f}")
    print(f"check: A11 and A15 of the covered tank against M and -M d/2, difference / M {diff:.1e} <= "
          f"{TOLERANCE:g}:", "ok" if diff <= TOLERANCE else "NOT MET")  # fmt: skip
    print("saved:", path)


if __name__ == "__main__":
    main()
