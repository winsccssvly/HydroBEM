"""A truncated vertical circular cylinder with a thin heave plate, the plate a dipole sheet: added mass in
surge and heave.

run_thin_plate.py: the thin plate, low-order panels, out/run_thin_plate/thin_plate.hbr

The case: S.-Y. Han, B. Bouscasse, V. Leroy, D. Le Touze (2024), Linear diffraction and radiation
theory of water waves by a truncated vertical circular cylinder with heave plate in deep and shallow
drafts, International Journal of Naval Architecture and Ocean Engineering 16, 100580,
doi:10.1016/j.ijnaoe.2023.100580, Fig. 6: the cylinder with the thin plate, solved there by matched
eigenfunction expansions. The reference values below are read by eye from the page image of Fig. 6
(the paper has no tables of results).

The parameters taken (Table 1, after the HiPRWind platform): D_c = 7 m, D_d = 20 m, the draft
d = 15.5 m to the plate's bottom, t_d = 0.1 m, the depth h = 100 m, heading 0; three frequencies
omega^2 D_c / g = 0.05, 0.5 and 1 of the figure's range 0.05 ... 3. The normalization of the figure:
a_11 / (rho D_c^3), a_33 / (rho D_c^3).

The choices of the example: the plate is modelled without thickness, a dipole sheet (a body's
`dipoles`) in its mid-plane z = -15.45 m: the annulus D_c / 2 < r < D_d / 2, both faces wetted; the
cylinder's side and bottom are the hull. Low-order panels on a quadrant, the hull's meridian divided
into cosine-spaced intervals of at most D_c / 18, the sheet's of at most D_c / 36, 24 panels around
the quadrant; the interior surface "graded". The sheet's free edge carries the singularity of the
flow around it, where constant panels converge slowly, with an error proportional to the interval
across the sheet (a_33 lies about 1.5 per cent above the converged value here; two sheets and the
extrapolation 2 fine - coarse, or the higher order of run_higher_order.py, remove it). One frequency
is solved at a time (FREQUENCY_THREADS = 1).

Prints: the model, the summary of the results, and a_11, a_33 at the three frequencies beside the
values read from Fig. 6, with the largest relative difference. The full comparison is in the
validation report, docs/validation/cases/cylinder_with_heave_plate.md.
"""

import time
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_thin_plate"  # the outputs of this script
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
DENSITY, GRAVITY = 1000.0, 9.80665  # kg/m^3, m/s^2 (the normalized quantities do not depend on them)
CYLINDER_DIAMETER, PLATE_DIAMETER, DRAFT, PLATE_THICKNESS = 7.0, 20.0, 15.5, 0.1  # m (Table 1)
DEPTH = 100.0  # m (Table 1)
NU = np.array([0.05, 0.5, 1.0])  # omega^2 D_c / g
HULL_SIZE = CYLINDER_DIAMETER / 18.0  # m, the largest interval of the hull's meridian
SHEET_SIZE = CYLINDER_DIAMETER / 36.0  # m, the largest interval across the sheet
N_THETA = 24  # panels around the quadrant
TOLERANCE = 3e-2  # the reading of the figure and the sheet's discretization, with margin

# Fig. 6, the lines of the analytical method read by eye from the page image: {quantity: {nu: value}}
FIG_6 = {
    "a11": {0.05: 1.65},
    "a33": {0.05: 6.88, 0.5: 6.93, 1.0: 6.78},
}


def cosine(n):
    """n + 1 fractions from 0 to 1, cosine-spaced: small at both ends."""
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))


def meridian(points, size):
    """The vertices (r, z) of the polyline through `points`, each side divided into cosine-spaced
    intervals of at most about `size` in the middle."""
    r, z = [points[0][0]], [points[0][1]]
    for (r0, z0), (r1, z1) in zip(points[:-1], points[1:], strict=True):
        n = max(2, int(np.ceil(0.5 * np.pi * np.hypot(r1 - r0, z1 - z0) / size)))
        s = cosine(n)[1:]
        r += list(r0 + (r1 - r0) * s)
        z += list(z0 + (z1 - z0) * s)
    return np.array(r), np.array(z)


def sweep(r, z, n_theta):
    """The quadrant 0 <= phi <= pi/2 of the surface of revolution of the meridian (r[i], z[i]): the
    normals point outward on a wall traversed downward and down on a ring traversed inward, up on a
    ring traversed outward (the panels at r = 0 are triangles, a repeated vertex)."""
    phi = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)

    def p(i, j):
        return [r[i] * np.cos(phi[j]), r[i] * np.sin(phi[j]), z[i]]

    return [[p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)] for i in range(len(r) - 1) for j in range(n_theta)]


def body_panels():
    """The panels of the quadrant and the elements of its parts: "hull" (the cylinder's side and
    bottom at the sheet's plane) and "plate" (the sheet outside the cylinder)."""
    rc, rd = 0.5 * CYLINDER_DIAMETER, 0.5 * PLATE_DIAMETER
    zs = 0.5 * PLATE_THICKNESS - DRAFT  # the plate's mid-plane
    hull = sweep(*meridian([(rc, 0.0), (rc, zs), (0.0, zs)], HULL_SIZE), N_THETA)
    plate = sweep(*meridian([(rc, zs), (rd, zs)], SHEET_SIZE), N_THETA)
    records = {"hull": (1, len(hull)), "plate": (len(hull) + 1, len(hull) + len(plate))}
    return np.array(hull + plate), records


def build() -> hb.Model:
    vertices, records = body_panels()
    m = hb.Model()
    m.meshes.add(
        name="thin_plate",
        kind="panels",
        vertices=vertices,
        reference_length=CYLINDER_DIAMETER,
        symmetry="xz_and_yz",
        label="cylinder with a thin heave plate (a dipole sheet), a quadrant",
    )
    body = m.bodies.add(
        name="thin_plate",
        mesh="thin_plate",
        ranges=[records["hull"]],
        rotation_center=[0.0, 0.0, 0.0],
        modes=["surge", "heave", "pitch"],
        interior_surface="graded",
    )
    body.dipoles.add(
        name="plate",
        mesh="thin_plate",
        ranges=[records["plate"]],
    )
    m.environment.update(
        depth=DEPTH,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=np.sqrt(NU * GRAVITY / CYLINDER_DIAMETER),
        headings=[0.0],
    )
    m.options.update(
        frequency_threads=FREQUENCY_THREADS,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=True,
        diffraction=True,
        motion=False,
    )
    return m


def main():
    start = time.perf_counter()
    m = build()
    print(m, "| route:", m.route())
    r = m.solve()
    path = OUT / "thin_plate.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(r.summary())

    def added_mass(mode):
        a = r.coefficient("added_mass", ("thin_plate", mode), ("thin_plate", mode))
        return a / (DENSITY * CYLINDER_DIAMETER**3)

    got = {"a11": added_mass("surge"), "a33": added_mass("heave")}
    print("\nFig. 6 of Han et al. (read from the figure) / HydroBEM at omega^2 D_c / g:")
    worst = 0.0
    for key, points in FIG_6.items():
        values = [(nu, ref, got[key][list(NU).index(nu)]) for nu, ref in points.items()]
        worst = max(worst, *(abs(v / ref - 1.0) for _, ref, v in values))
        print(f"  {key:>4}  " + ",  ".join(f"{nu:g}: {ref:.3g} / {v:.4g}" for nu, ref, v in values))
    verdict = "ok" if worst <= TOLERANCE else "NOT MET"
    print(f"check: largest relative difference {worst:.1e} <= {TOLERANCE:.0e}: {verdict}")
    print(f"saved: {path}; {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
