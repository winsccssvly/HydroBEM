"""The freely floating fish tank of run.py by the higher-order method, on exact geometry.

run_higher_order.py: the tank free in surge and pitch in waves (Fig. 2 of the paper), analytic patches:
    out/run_higher_order/fish_tank_d2.hbr (the model with its results)

The case: J. N. Newman (2021), Linear motions of fish tanks, 36th International Workshop on Water
Waves and Floating Bodies (IWWWFB), 25-28 April 2021: a circular cylindrical tank with a wall of zero
thickness floating in infinite depth, the fluid inside of the sea's density with its free surface at
the same height; surge, pitch and the first sloshing mode.

The parameters taken (as run.py): radius a = 1, draft d = 2, infinite depth, the density and the
gravity 1 (the paper's nondimensional results, K = omega^2 / g), the origin at the centre of the
interior free surface, waves along +x of unit amplitude at K = 0.05 and every 0.002 from 1.801 to
1.899 around the first sloshing wavenumber K1 = k1 tanh(k1 d) = 1.8389, the sloshing amplitude at
x = 0.9, y = 0.

The mesh: analytic patches, each a quadrant of a revolved straight segment of the meridian (the map
"line", exact circles), five elements of one mesh: the hull's side and bottom (the wetted surface,
normals out into the sea), the waterplane disc r < 1 in z = 0 (the interior surface that removes the
irregular frequencies of the exterior problem) and the tank's side and bottom, the hull's two surfaces
traversed the other way (normals into the tank); the tank takes elements of the body's mesh. The
potential: B-splines of order 3 (Gauss orders 3 and 4), spans at most 0.25 long; along the meridian
the profile clusters the spans at the bottom corner, where the exterior flow is singular. The body
is massless (mass 0, a zero inertia): the engine solves its motions with the inertia of the water
alone, (C - omega^2 A + i omega B) xi = X (physics section 8). The exact circle puts the panel
model's K1 on the exact one (run.py's polygon: 0.003 above it), so the zeros below lie where the
paper has them.

What the script prints: the surge RAO at K = 0.05 and the check: the zeros of xi1 and xi5 with both
modes free just above K1 against the paper's values for d = 2 (Section 3: K = 1.841 and 1.845). The
full comparison is in the validation report, docs/validation/cases/fish_tank_cylinders.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_higher_order"  # the outputs of this script
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
DENSITY, GRAVITY = 1.0, 1.0  # nondimensional: lengths in units of the radius
RADIUS, DRAFT = 1.0, 2.0
K1_ROOT = 1.8411837813406593  # the first zero of J1' (Abramowitz & Stegun 1964, Table 9.5)
WAVENUMBERS = np.r_[0.05, 1.801 + 0.002 * np.arange(50)]  # K = omega^2 / g
SLOSHING_POINT = [0.9, 0.0, 0.0]  # the point of the sloshing amplitude in the tank's free surface
PANEL_SIZE = 0.25  # the largest span of the potential, in units of the radius
ORDERS = [3, 3, 3, 4]  # the B-spline orders (u, v) and the Gauss orders (outer, inner)
REVERSED = {"linear": "linear", "quadratic_start": "quadratic_end", "quadratic_end": "quadratic_start"}
SURGE, PITCH = 0, 4  # the coordinates of surge and pitch among the six of the body
PAPER_ZEROS = (1.841, 1.845)  # Newman (2021), Section 3, d = 2: the zeros of xi1 and xi5 with both free
TOLERANCE = 2e-3


def segments(draft):
    """The segments of the meridian, ((r0, z0), (r1, z1), profile, role), traversed so that the
    parametric normal points into the body (the wetted side): the hull's side down and its bottom
    inward, the waterplane disc outward, then the tank's walls, the hull's segments reversed (the
    normals into the tank); role "hull", "interior" or "tank"."""
    a = RADIUS
    hull = [((a, 0.0), (a, -draft), "quadratic_end", "hull"), ((a, -draft), (0.0, -draft), "quadratic_start", "hull")]
    tank = [(end, start, REVERSED[profile], "tank") for start, end, profile, _ in hull]
    return hull + [((0.0, 0.0), (a, 0.0), "linear", "interior")] + tank


def spans(length):
    """The spans of a patch edge of the given length, each at most PANEL_SIZE * RADIUS long."""
    return max(1, int(np.ceil(length / (PANEL_SIZE * RADIUS) - 1.0e-9)))


def build() -> hb.Model:
    parts = segments(DRAFT)
    rows = [[spans(0.5 * np.pi * max(r0, r1)), spans(np.hypot(r1 - r0, z1 - z0))] + ORDERS
            for (r0, z0), (r1, z1), _, _ in parts]  # fmt: skip
    m = hb.Model()
    mesh = m.meshes.add(
        name="tank",
        kind="analytic_patches",
        discretization=np.array(rows),
        reference_length=RADIUS,
        symmetry="xz_and_yz",
        label=f"fish tank a = 1, d = {DRAFT:g}: the hull, the waterplane and the tank faces of the wall",
    )
    records = {}
    for i, ((r0, z0), (r1, z1), profile, role) in enumerate(parts):
        mesh.patches.add(
            type="AnalyticPatch",
            map="line",
            profile=profile,
            a=1.0,
            b=1.0,
            theta0=0.0,
            theta1=0.5 * np.pi,
            p=[r0, z0, r1, z1],
        )
        first, _ = records.get(role, (i + 1, i + 1))
        records[role] = (first, i + 1)
    body = m.bodies.add(
        name="tank",
        mesh="tank",
        ranges=[records["hull"]],
        modes=["surge", "pitch"],
        center_of_gravity=[0.0, 0.0, 0.0],
        mass=0.0,  # massless, its inertia zero (the radii of gyration zero by default)
        interior_surface="mesh",
        interior_surface_mesh="tank",
        interior_surface_ranges=[records["interior"]],
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
        higher_order=True,
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
