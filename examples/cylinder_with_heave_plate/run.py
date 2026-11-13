"""A truncated vertical circular cylinder with a thick heave plate at its bottom: exciting forces, added
mass and damping in surge, heave and pitch.

run.py: the thick plate, low-order panels, out/run/thick_plate.hbr
run_thin_plate.py: the thin plate as a dipole sheet
run_higher_order.py: the thick and the thin plate on exact-geometry patches
run_model_scale.py: the experiments' model with the thin plate: radiated waves and mean drift

The case: S.-Y. Han, B. Bouscasse, V. Leroy, D. Le Touze (2024), Linear diffraction and radiation
theory of water waves by a truncated vertical circular cylinder with heave plate in deep and shallow
drafts, International Journal of Naval Architecture and Ocean Engineering 16, 100580,
doi:10.1016/j.ijnaoe.2023.100580: a surface-piercing cylinder (diameter D_c) standing on a coaxial
circular plate (diameter D_d, thickness t_d, its bottom at the draft d), at rest, solved there by
matched eigenfunction expansions (its lines "MEEM" in the figures). The paper has no tables of
results: the reference values below are read by eye from the page image of its Fig. 4.

The parameters taken (Table 1, the cylinder with the thick plate, after the DeepCwind platform):
D_c = 12 m, D_d = 24 m, the draft d = 20 m to the plate's bottom, t_d = 6 m, the depth h = 100 m,
heading 0 (waves along +x); three frequencies omega^2 D_c / g = 0.5, 1 and 2 of the figure's range
0.05 ... 3. The normalization of the figure: F_x / (rho g D_c^2 A) (A the wave amplitude),
a_33 / (rho D_c^3), b_ii / (omega rho D_c^n); pitch about the origin in the free surface on the axis.

The choices of the example: low-order panels on a quadrant (symmetry about x = 0 and y = 0), the
meridian divided into cosine-spaced intervals (small at the waterline and at every corner) of at
most D_c / 18, 24 panels around the quadrant; the plate a disc of its thickness, part of the hull.
The interior surface "graded" removes the irregular frequencies. The body is at rest: the radiation
and the diffraction problems, no motions. One frequency is solved at a time (FREQUENCY_THREADS = 1;
more threads give the same results).

Prints: the model, the summary of the results, and F_x and a_33 at the three frequencies beside
the values read from Fig. 4, with the largest relative difference (the reading is good to about
1 per cent). The full comparison is in the validation report,
docs/validation/cases/cylinder_with_heave_plate.md.
"""

import time
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
DENSITY, GRAVITY = 1000.0, 9.80665  # kg/m^3, m/s^2 (not stated; the normalized quantities do not depend on them)
CYLINDER_DIAMETER, PLATE_DIAMETER, DRAFT, PLATE_THICKNESS = 12.0, 24.0, 20.0, 6.0  # m (Table 1)
DEPTH = 100.0  # m (Table 1)
NU = np.array([0.5, 1.0, 2.0])  # omega^2 D_c / g
PANEL_SIZE = CYLINDER_DIAMETER / 18.0  # m, the largest interval of the meridian
N_THETA = 24  # panels around the quadrant
TOLERANCE = 3e-2  # the reading of the figure (about 1 per cent), with margin

# Fig. 4, the lines of the analytical method read by eye from the page image: {quantity: {nu: value}}
FIG_4 = {
    "Fx": {0.5: 1.34, 1.0: 1.68, 2.0: 1.12},
    "a33": {0.5: 2.68, 1.0: 2.58, 2.0: 2.47},
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
    """The quadrant 0 <= phi <= pi/2 of the surface of revolution of the meridian (r[i], z[i]), traversed
    down the walls and inward along the bottom so that the normals point into the water (the panels at
    r = 0 are triangles, a repeated vertex)."""
    phi = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)

    def p(i, j):
        return [r[i] * np.cos(phi[j]), r[i] * np.sin(phi[j]), z[i]]

    return np.array([[p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)]
                     for i in range(len(r) - 1) for j in range(n_theta)])  # fmt: skip


def hull_panels():
    """The wetted surface: the cylinder's side, the plate's top annulus, its side and its bottom."""
    rc, rd = 0.5 * CYLINDER_DIAMETER, 0.5 * PLATE_DIAMETER
    top = PLATE_THICKNESS - DRAFT
    r, z = meridian(
        points=[(rc, 0.0), (rc, top), (rd, top), (rd, -DRAFT), (0.0, -DRAFT)],
        size=PANEL_SIZE,
    )
    return sweep(r, z, N_THETA)


def build() -> hb.Model:
    m = hb.Model()
    m.meshes.add(
        name="thick_plate",
        kind="panels",
        vertices=hull_panels(),
        reference_length=CYLINDER_DIAMETER,
        symmetry="xz_and_yz",
        label="cylinder with a thick heave plate, a quadrant",
    )
    m.bodies.add(
        name="thick_plate",
        mesh="thick_plate",
        rotation_center=[0.0, 0.0, 0.0],
        modes=["surge", "heave", "pitch"],
        interior_surface="graded",
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
    path = OUT / "thick_plate.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(r.summary())

    dc = CYLINDER_DIAMETER
    surge = r.coordinate(body="thick_plate", mode="surge")[0]
    got = {
        "Fx": np.abs(r.excitation.diffraction[:, 0, surge]) / (DENSITY * GRAVITY * dc**2),
        "a33": r.coefficient("added_mass", ("thick_plate", "heave"), ("thick_plate", "heave")) / (DENSITY * dc**3),
    }
    print("\nFig. 4 of Han et al. (read from the figure) / HydroBEM at omega^2 D_c / g:")
    worst = 0.0
    for key, points in FIG_4.items():
        values = [(nu, ref, got[key][list(NU).index(nu)]) for nu, ref in points.items()]
        worst = max(worst, *(abs(v / ref - 1.0) for _, ref, v in values))
        print(f"  {key:>4}  " + ",  ".join(f"{nu:g}: {ref:.3g} / {v:.4g}" for nu, ref, v in values))
    verdict = "ok" if worst <= TOLERANCE else "NOT MET"
    print(f"check: largest relative difference {worst:.1e} <= {TOLERANCE:.0e}: {verdict}")
    print(f"saved: {path}; {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
