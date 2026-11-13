"""The cylinder with the thick heave plate and with the thin plate (a dipole patch) by the higher-order
method, on exact geometry: added mass in heave against Han et al. (2024).

run_higher_order.py: both bodies, out/run_higher_order/thick_plate.hbr and thin_plate.hbr

The case: S.-Y. Han, B. Bouscasse, V. Leroy, D. Le Touze (2024), Linear diffraction and radiation
theory of water waves by a truncated vertical circular cylinder with heave plate in deep and shallow
drafts, International Journal of Naval Architecture and Ocean Engineering 16, 100580,
doi:10.1016/j.ijnaoe.2023.100580, Figs. 4 and 6 (matched eigenfunction expansions). The reference
values below are read by eye from the page images of the figures (the paper has no tables).

The parameters taken (Table 1): the thick plate (after DeepCwind) D_c = 12 m, D_d = 24 m, the draft
d = 20 m to the plate's bottom, t_d = 6 m; the thin plate (after HiPRWind) D_c = 7 m, D_d = 20 m,
d = 15.5 m, modelled as a sheet in its mid-plane z = -15.45 m; the depth h = 100 m, heading 0; three
frequencies omega^2 D_c / g = 0.05 (0.5 for the thick plate), 1 and 2. The normalization of the
figures: a_33 / (rho D_c^3).

The choices of the example: analytic patches, each a quadrant of a revolved straight segment of the
meridian (the map "line", exact circles), one patch per segment: the wetted hull, the waterplane
disc r < D_c / 2 in z = 0 as the interior surface (interior_surface "mesh"), and for the thin plate
the annulus D_c / 2 < r < D_d / 2, a dipole patch of the same mesh. The potential on each patch:
B-splines of order 3 (Gauss orders 3 and 4), the spans at most D_c / 8 along the meridian and around
the quadrant; the profile "smooth" clusters the spans at the ends of each segment, at the waterline,
the corners and the sheet's free edge, where the flow is singular. The sheet's edge converges here in
one solve (the low order of run_thin_plate.py errs in proportion to its interval across the sheet).
One frequency is solved at a time (FREQUENCY_THREADS = 1).

Prints: for each body the model and the summary of the results, then a_33 of both bodies beside the
values read from the figures, with the largest relative difference (the reading is good to about
1 per cent). The full comparison is in the validation report,
docs/validation/cases/cylinder_with_heave_plate.md.
"""

import time
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_higher_order"  # the outputs of this script
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
DENSITY, GRAVITY = 1000.0, 9.80665  # kg/m^3, m/s^2 (the normalized quantities do not depend on them)
DEPTH = 100.0  # m (Table 1)
PANEL_SIZE = 1.0 / 8.0  # the largest span of the potential, relative to D_c
ORDERS = [3, 3, 3, 4]  # the B-spline orders (u, v) and the Gauss orders (outer, inner)
TOLERANCE = 3e-2  # the reading of the figures (about 1 per cent), with margin

# Table 1 (m) and the lines of the analytical method of Figs. 4 and 6 read by eye from the page
# images, a_33 / (rho D_c^3) at omega^2 D_c / g
BODIES = {
    "thick_plate": dict(cylinder_diameter=12.0, plate_diameter=24.0, draft=20.0, plate_thickness=6.0, thick=True,
                        a33={0.5: 2.68, 1.0: 2.58, 2.0: 2.47}),  # Fig. 4 (b)
    "thin_plate": dict(cylinder_diameter=7.0, plate_diameter=20.0, draft=15.5, plate_thickness=0.1, thick=False,
                       a33={0.05: 6.88, 1.0: 6.78, 2.0: 6.77}),  # Fig. 6 (b)
}  # fmt: skip


def segments(body):
    """The segments of the meridian, ((r0, z0), (r1, z1), profile, role), traversed so that the
    parametric normal points into the body: down the walls, outward under a ring facing up, inward
    under a ring facing down; role "hull", "interior" (the waterplane disc) or "plate" (the sheet)."""
    rc, rd = 0.5 * body["cylinder_diameter"], 0.5 * body["plate_diameter"]
    d, t = body["draft"], body["plate_thickness"]
    if body["thick"]:
        hull = [((rc, 0.0), (rc, t - d)), ((rc, t - d), (rd, t - d)), ((rd, t - d), (rd, -d)), ((rd, -d), (0.0, -d))]
        plate = []
    else:
        zs = 0.5 * t - d
        hull = [((rc, 0.0), (rc, zs)), ((rc, zs), (0.0, zs))]
        plate = [((rc, zs), (rd, zs), "smooth", "plate")]
    return [(a, b, "smooth", "hull") for a, b in hull] + [((0.0, 0.0), (rc, 0.0), "linear", "interior")] + plate


def spans(length, size):
    """The spans of a patch edge of the given length, each at most `size` long (at least one)."""
    return max(1, int(np.ceil(length / size - 1.0e-9)))


def build(name, body) -> hb.Model:
    size = PANEL_SIZE * body["cylinder_diameter"]
    parts = segments(body)
    # per patch: the spans around the quadrant and along the segment, then the orders
    rows = [[spans(0.5 * np.pi * max(r0, r1), size), spans(np.hypot(r1 - r0, z1 - z0), size)] + ORDERS
            for (r0, z0), (r1, z1), _, _ in parts]  # fmt: skip
    m = hb.Model()
    mesh = m.meshes.add(
        name=name,
        kind="analytic_patches",
        discretization=np.array(rows),
        reference_length=body["cylinder_diameter"],
        symmetry="xz_and_yz",
        label=f"{name}: cylinder with heave plate, a quadrant of revolved segments",
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
    b = m.bodies.add(
        name=name,
        mesh=name,
        ranges=[records["hull"]],
        rotation_center=[0.0, 0.0, 0.0],
        modes=["surge", "heave", "pitch"],
        interior_surface="mesh",
        interior_surface_mesh=name,
        interior_surface_ranges=[records["interior"]],
    )
    if "plate" in records:
        b.dipoles.add(
            name="plate",
            mesh=name,
            ranges=[records["plate"]],
        )
    nu = np.array(list(body["a33"]))
    m.environment.update(
        depth=DEPTH,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=np.sqrt(nu * GRAVITY / body["cylinder_diameter"]),
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
        motion=False,
    )
    return m


def main():
    start = time.perf_counter()
    lines, worst = [], 0.0
    for name, body in BODIES.items():
        m = build(name, body)
        print(m, "| route:", m.route())
        r = m.solve()
        path = OUT / f"{name}.hbr"
        m.save_results(
            path,
            overwrite=True,
        )
        print(r.summary())
        a33 = r.coefficient("added_mass", (name, "heave"), (name, "heave")) / (DENSITY * body["cylinder_diameter"] ** 3)
        pairs = list(zip(body["a33"].items(), a33, strict=True))
        worst = max(worst, *(abs(v / ref - 1.0) for (_, ref), v in pairs))
        lines.append(f"  {name:>11} a33  " + ",  ".join(f"{nu:g}: {ref:.3g} / {v:.4g}" for (nu, ref), v in pairs))
    print("\nFigs. 4 and 6 of Han et al. (read from the figures) / HydroBEM at omega^2 D_c / g:")
    print("\n".join(lines))
    verdict = "ok" if worst <= TOLERANCE else "NOT MET"
    print(f"check: largest relative difference {worst:.1e} <= {TOLERANCE:.0e}: {verdict}")
    print(f"saved: {OUT}; {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
