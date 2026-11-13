"""The experiments' model of a cylinder with a thin heave plate at the shallow draft: the surge exciting
force, the wave radiated in surge at a wave gauge (a field point) and the mean drift force.

run_model_scale.py: out/run_model_scale/plate_shallow_radiation.hbr and plate_shallow_diffraction.hbr

The case: S.-Y. Han, B. Bouscasse, V. Leroy, D. Le Touze (2024), Linear diffraction and radiation
theory of water waves by a truncated vertical circular cylinder with heave plate in deep and shallow
drafts, International Journal of Naval Architecture and Ocean Engineering 16, 100580,
doi:10.1016/j.ijnaoe.2023.100580, Figs. 9 and 11: the lines of its analytical method (matched
eigenfunction expansions) for the models of the experiments. The reference values below are read by
eye from the page images of the figures (the paper has no tables of results).

The parameters taken (Table 2, Section 4): the cylinder D_c = 0.35 m with the plate D_d = 1.00 m,
t_d = 0.005 m at the shallow draft d = 0.25 m (to the plate's bottom), the depth of the tank
h = 5.0 m, heading 0; three frequencies omega^2 D_c / g = 1, 1.15 and 1.5. The quantities: the
exciting force F_x / (rho g D_c^2 A) of the fixed model (Fig. 9 (a)), the mean drift force
F_x / (rho g D_c A^2) of the fixed model by momentum conservation (Fig. 9 (b)), and the amplitude
|eta| / xi_1 of the wave radiated in surge at the wave gauge r / D_c = 4, theta = pi (WG3, on the
side the waves come from; Fig. 11 (a)).

The choices of the example: the plate a disc of its thickness, part of the hull (its rim carries a
part of F_x at this draft); low-order panels on a quadrant, the meridian divided into cosine-spaced
intervals of at most D_c / 18, the plate's D_c / 36, 24 panels around the quadrant; the interior
surface "graded". Two solves: the radiation problems with the field point in z = 0 (its pressure
p / (rho g) per unit motion is the elevation), and the diffraction of the fixed body (modes none)
with its drift (the drift of a moving body needs its motions). One frequency is solved at a time
(FREQUENCY_THREADS = 1). The measurements of the paper include the viscous effects of the flow
separating at the plate's edge and are not compared.

Prints: the two models and their summaries, then F_x, the drift and the radiated wave beside the
values read from the figures, with the largest relative difference (the reading is good to a few
per cent). The full comparison is in the validation report,
docs/validation/cases/cylinder_with_heave_plate.md.
"""

import time
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_model_scale"  # the outputs of this script
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
DENSITY, GRAVITY = 1000.0, 9.80665  # kg/m^3, m/s^2 (the normalized quantities do not depend on them)
CYLINDER_DIAMETER, PLATE_DIAMETER, PLATE_THICKNESS = 0.35, 1.00, 0.005  # m (Table 2)
DRAFT = 0.25  # m, the shallow draft (Table 2)
DEPTH = 5.0  # m, the tank (Section 4)
NU = np.array([1.0, 1.15, 1.5])  # omega^2 D_c / g
GAUGE = [[-4.0 * CYLINDER_DIAMETER, 0.0, 0.0]]  # WG3: r / D_c = 4, theta = pi (m, global axes, in z = 0)
HULL_SIZE = CYLINDER_DIAMETER / 18.0  # m, the largest interval of the cylinder's meridian
PLATE_SIZE = CYLINDER_DIAMETER / 36.0  # m, the largest interval of the plate's meridian
N_THETA = 24  # panels around the quadrant
TOLERANCE = 5e-2  # the reading of the figures, with margin

# the lines of the analytical method read by eye from the page images (the plate at the shallow
# draft): {quantity: {nu: value}}
PAPER = {
    "Fx": {1.0: 1.27, 1.15: 1.29, 1.5: 1.16},  # Fig. 9 (a)
    "drift": {1.0: 0.152, 1.5: 0.483},  # Fig. 9 (b)
    "eta1": {1.0: 0.243, 1.5: 0.403},  # Fig. 11 (a)
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
    so that the normals point into the water (the panels at r = 0 are triangles, a repeated vertex)."""
    phi = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)

    def p(i, j):
        return [r[i] * np.cos(phi[j]), r[i] * np.sin(phi[j]), z[i]]

    return np.array([[p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)]
                     for i in range(len(r) - 1) for j in range(n_theta)])  # fmt: skip


def hull_panels():
    """The wetted surface: the cylinder's side, then the plate's top annulus, its rim and its bottom."""
    rc, rd = 0.5 * CYLINDER_DIAMETER, 0.5 * PLATE_DIAMETER
    top = PLATE_THICKNESS - DRAFT
    r, z = meridian([(rc, 0.0), (rc, top)], HULL_SIZE)
    r_plate, z_plate = meridian([(rc, top), (rd, top), (rd, -DRAFT), (0.0, -DRAFT)], PLATE_SIZE)
    return sweep(np.r_[r, r_plate[1:]], np.r_[z, z_plate[1:]], N_THETA)


def build(radiation) -> hb.Model:
    """The radiation problems with the wave gauge (radiation), or the diffraction of the fixed body
    with its mean drift."""
    m = hb.Model()
    m.meshes.add(
        name="plate_shallow",
        kind="panels",
        vertices=hull_panels(),
        reference_length=CYLINDER_DIAMETER,
        symmetry="xz_and_yz",
        label="model with a thin heave plate, shallow draft, a quadrant",
    )
    m.bodies.add(
        name="plate_shallow",
        mesh="plate_shallow",
        rotation_center=[0.0, 0.0, 0.0],
        modes=["surge", "heave", "pitch"] if radiation else [],
        interior_surface="graded",
    )
    m.environment.update(
        depth=DEPTH,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=np.sqrt(NU * GRAVITY / CYLINDER_DIAMETER),
        headings=[] if radiation else [0.0],
    )
    m.options.update(
        frequency_threads=FREQUENCY_THREADS,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=radiation,
        diffraction=not radiation,
        motion=False,
    )
    if radiation:
        m.first_order.update(
            field_points=GAUGE,
            field_pressure=True,
        )
    else:
        m.second_order.update(
            drift_momentum_conservation=True,
        )
    return m


def main():
    start = time.perf_counter()
    got = {}
    for kind in ("radiation", "diffraction"):
        m = build(radiation=kind == "radiation")
        print(m, "| route:", m.route())
        r = m.solve()
        path = OUT / f"plate_shallow_{kind}.hbr"
        m.save_results(
            path,
            overwrite=True,
        )
        print(r.summary())
        if kind == "radiation":
            k = r.coordinate(body="plate_shallow", mode="surge")[1]
            got["eta1"] = np.abs(r.field_points.pressure_radiation[:, 0, k])  # the elevation per unit surge
        else:
            surge = r.coordinate(body="plate_shallow", mode="surge")[0]
            got["Fx"] = np.abs(r.excitation.diffraction[:, 0, surge]) / (DENSITY * GRAVITY * CYLINDER_DIAMETER**2)
            got["drift"] = r.drift.momentum_conservation[:, 0, 0].real / (DENSITY * GRAVITY * CYLINDER_DIAMETER)

    print("\nFigs. 9 and 11 of Han et al. (read from the figures) / HydroBEM at omega^2 D_c / g:")
    worst = 0.0
    for key, points in PAPER.items():
        values = [(nu, ref, got[key][list(NU).index(nu)]) for nu, ref in points.items()]
        worst = max(worst, *(abs(v / ref - 1.0) for _, ref, v in values))
        print(f"  {key:>5}  " + ",  ".join(f"{nu:g}: {ref:.3g} / {v:.4g}" for nu, ref, v in values))
    verdict = "ok" if worst <= TOLERANCE else "NOT MET"
    print(f"check: largest relative difference {worst:.1e} <= {TOLERANCE:.0e}: {verdict}")
    print(f"saved: {OUT}; {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
