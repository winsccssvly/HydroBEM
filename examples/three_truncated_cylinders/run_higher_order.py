"""Three floating truncated vertical cylinders on an equilateral triangle, by the higher-order method.

The case: Raghavan, Loukogeorgaki, Mantadakis, Metrikine & Lavidas (2024), HAMS-MREL, a new open
source multiple body solver for marine renewable energies, Renewable Energy 237, 121577,
doi:10.1016/j.renene.2024.121577, §3.3 "Case 3 - Three semi-submerged cylinders"; the configuration
of Mavrakos (1991), Hydrodynamic coefficients for groups of interacting vertical axisymmetric bodies,
Ocean Engineering 18(5), 485-515, doi:10.1016/0029-8018(91)90027-N (Figs. 5b-c).

The parameters taken: three cylinders of radius b = 3 m and draft 10.95 m at the corners of an
equilateral triangle of side 15 m (cylinder 1 at (15/sqrt(3), 0)), depth 120 m, heading 0 deg, the
six rigid modes of each about its origin on the axis in the free surface.

The choices: options.higher_order = True; the mesh is the named analytic shape "circular_cylinder"
(the exact side, bottom and waterplane), its quadrant stored (symmetry "xz_and_yz") and expanded by
the engine for each body at its position; the waterplane patch is the interior free surface
(interior_surface "detect"); mapping "nonuniform"; per patch 4 spans around the quadrant, 6 down the
side, 3 across the bottom and the waterplane, B-splines of order 3, Gauss orders 3 and 4.

Prints, at k b = 0.6, Mavrakos' coefficients A / (rho pi b^3) and B / (rho omega pi b^3) (b^4 for the
pitch-surge) beside the values read by eye from his Figs. 5b and 5c (reading accuracy about 0.05), and
the differences. The full comparison is in the validation report,
docs/validation/cases/three_truncated_cylinders.md.
"""

import math
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_higher_order"  # the outputs of this script
RADIUS, DRAFT = 3.0, 10.95  # m: b, 3.65b
SIDE = 15.0  # m, the side of the triangle, centre to centre (5b)
DEPTH = 120.0  # m (40b)
HEADINGS = [0.0]  # deg
KB = [0.6]  # the wavenumber times the radius of the check
N_AROUND, N_DOWN, N_ACROSS = 4, 6, 3  # spans around the quadrant, down the side, across bottom and waterplane
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)

# Mavrakos (1991), Figs. 5b and 5c at k b = 0.6, read by eye from the page images
FIG_5 = {"a11_11": 2.71, "a11_22": 3.02, "b11_11": 1.59, "b11_12": -0.84, "b51_11": -1.87, "b51_12": 0.96}
# the bounds, in the normalized units: the reading (0.05) and the discretization (0.01 on these
# panels); the cylinders' own added mass a11_11, a11_22 lies above the curves read, by 0.11 at
# k b = 0.6 in the converged solution (0.12 on the low-order panels; an offset of the published curves, the
# validation report), so its bound is that offset, the discretization and a margin
TOLERANCE = 0.06
TOLERANCE_OWN_ADDED_MASS = 0.15
OWN_ADDED_MASS = ("a11_11", "a11_22")


def build() -> hb.Model:
    m = hb.Model()
    m.meshes.analytic(
        name="cylinder",
        shape="circular_cylinder",
        discretization=[
            (N_AROUND, N_DOWN, 3, 3, 3, 4),  # side
            (N_AROUND, N_ACROSS, 3, 3, 3, 4),  # bottom
            (N_AROUND, N_ACROSS, 3, 3, 3, 4),  # waterplane
        ],
        parts=["side", "bottom", "waterplane"],
        mapping="nonuniform",
        reference_length=1.0,
        symmetry="xz_and_yz",
        label="cylinder b = 3 m, H = 10.95 m, analytic patches",
        radius=RADIUS,
        draft=DRAFT,
    )
    r = SIDE / math.sqrt(3.0)  # the distance of the axes from the centre of the triangle
    positions = {
        "cylinder_1": [r, 0.0, 0.0],
        "cylinder_2": [-r / 2, SIDE / 2, 0.0],
        "cylinder_3": [-r / 2, -SIDE / 2, 0.0],
    }
    for name, position in positions.items():
        m.bodies.add(
            name=name,
            mesh="cylinder",
            position=position,
            rotation_center=[0.0, 0.0, 0.0],
            modes=["surge", "sway", "heave", "roll", "pitch", "yaw"],
            center_of_gravity=[0.0, 0.0, 0.0],
            mass="displaced",
            interior_surface="detect",
        )
    k = np.asarray(KB) / RADIUS
    m.environment.update(
        depth=DEPTH,
        frequency_input="omegas",
        omegas=np.sqrt(m.environment.gravity * k * np.tanh(k * DEPTH)),  # the dispersion relation
        headings=HEADINGS,
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


def figure_quantities(results):
    """Mavrakos' coefficients divided by pi, {"a11_12": an array over the frequencies, ...}: the force
    (i = 1, surge) or moment (i = 5, pitch) on cylinder q from the surge of cylinder p."""
    rho, omega = results.model.environment.density, results.omegas
    rows = {"1": "surge", "5": "pitch"}
    q = {}
    for i in ("1", "5"):
        scale = rho * np.pi * RADIUS ** (3 if i == "1" else 4)
        for qp in ("11", "22", "12"):
            row, col = (f"cylinder_{qp[0]}", rows[i]), (f"cylinder_{qp[1]}", "surge")
            q[f"a{i}1_{qp}"] = results.coefficient("added_mass", row, col) / scale
            q[f"b{i}1_{qp}"] = results.coefficient("radiation_damping", row, col) / (omega * scale)
    return q


def main():
    m = build()
    print(m, "| route:", m.route())
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / "three_truncated_cylinders.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())

    q = figure_quantities(results)
    print(f"\nk b = {KB[0]}: HydroBEM / Mavrakos (1991), Figs. 5b-c (read from the figures)")
    ok = True
    for key, ref in FIG_5.items():
        got = q[key][0]
        bound = TOLERANCE_OWN_ADDED_MASS if key in OWN_ADDED_MASS else TOLERANCE
        ok = ok and abs(got - ref) <= bound
        print(f"  {key:>6} {got:7.3f} {ref:7.2f}   difference {abs(got - ref):.3f} <= {bound}")
    print(f"check: every difference within its bound: {'ok' if ok else 'NOT MET'}")
    print("saved:", path)


if __name__ == "__main__":
    main()
