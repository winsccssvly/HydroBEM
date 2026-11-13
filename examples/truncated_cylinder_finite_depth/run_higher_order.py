"""A floating truncated vertical cylinder in finite depth in the higher order: heave and surge added
mass and damping against an eigenfunction-matching solution.

The case: a vertical circular cylinder of radius a = 1 m floating with draft d = 1 m in the depth
h = 2 m, rho = 1000 kg/m^3, g = 9.80665 m/s^2, the radiation problems of surge and heave about the
centre of the waterplane at Ka = 0.4, 0.8 and 1.5 (K = omega^2/g), normalized as A/(rho a^3) and
B/(rho omega a^3). The reference is HydroBEM's own computation: the eigenfunction matching
(Garrett 1971; Yeung 1981) of scripts/gen_cylinder_reference.py, after R. W. Yeung (1981), Added
mass and damping of a vertical cylinder in finite-depth waters, Applied Ocean Research 3(3),
119-133, doi:10.1016/0141-1187(81)90101-2, and C. J. R. Garrett (1971), Wave forces on a circular
dock, J. Fluid Mech. 46(1), 129-139, doi:10.1017/S0022112071000430; converged to better than 5e-6.

The higher-order mesh: the named analytic shape "circular_cylinder" of the engine, the quadrant
x, y >= 0 as two exact analytic patches (the symmetry planes x = 0 and y = 0 give the rest), the
potential a B-spline of order 3 with Gauss orders 3 (outer) and 4 (inner):
- the side, 8 spans around the quarter circle and 8 down to the bottom edge;
- the bottom, 8 spans around and 4 along the radius;
- the nonuniform mapping clusters the spans towards the waterline and the bottom edge, where the
  potential varies most.
No interior surface: the first irregular frequency lies above the frequencies of the case. The
options: options.higher_order = True; one frequency is solved at a time (FREQUENCY_THREADS = 1).

Prints the model, the summary of the results and the four coefficients beside the reference, with
the largest difference relative to the largest reference value of each quantity; saves
out/run_higher_order/truncated_cylinder_h2.hbr. The full comparison is in the validation report,
docs/validation/cases/truncated_cylinder_finite_depth.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_higher_order"  # the outputs of this script
RADIUS, DRAFT = 1.0, 1.0  # m
DEPTH = 2.0  # m
DENSITY, GRAVITY = 1000.0, 9.80665
SIDE = (8, 8, 3, 3, 3, 4)  # spans around the quarter circle and down the side, spline orders, Gauss orders
BOTTOM = (8, 4, 3, 3, 3, 4)  # spans around the quarter circle and along the radius, orders as the side
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
# HydroBEM's eigenfunction matching (Garrett 1971; Yeung 1981), scripts/gen_cylinder_reference.py,
# h/a = 2, d/a = 1: Ka, A33/(rho a^3), B33/(rho omega a^3), A11/(rho a^3), B11/(rho omega a^3)
REFERENCE = np.array([
    [0.4, 1.940154076, 0.7107076064, 2.370760208, 0.5503827251],
    [0.8, 1.775587193, 0.3687504810, 2.144233779, 1.403494321],
    [1.5, 1.864562768, 0.07744079602, 0.8730149751, 1.541754199],
])  # fmt: skip
TOLERANCE = 1e-4  # of the largest reference value of each quantity


def build(ka) -> hb.Model:
    m = hb.Model()
    m.meshes.analytic(
        name="cylinder",
        shape="circular_cylinder",
        discretization=[SIDE, BOTTOM],
        parts=["side", "bottom"],
        mapping="nonuniform",
        radius=RADIUS,
        draft=DRAFT,
        reference_length=RADIUS,
        symmetry="xz_and_yz",
        label="cylinder a = 1 m, d = 1 m, analytic patches, 8 spans a quarter circle",
    )
    m.bodies.add(
        name="cylinder",
        mesh="cylinder",
        rotation_center=[0.0, 0.0, 0.0],
        modes=["surge", "heave"],
        interior_surface="none",
    )
    m.environment.update(
        depth=DEPTH,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=np.sqrt(np.asarray(ka) * GRAVITY / RADIUS),
        headings=[],
    )
    m.options.update(
        higher_order=True,
        frequency_threads=FREQUENCY_THREADS,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=True,
        diffraction=False,
        motion=False,
    )
    return m


def main():
    m = build(ka=REFERENCE[:, 0])
    print(m, "| route:", m.route())
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / f"truncated_cylinder_h{DEPTH:g}.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())
    print("saved:", path)

    def coefficient(name, mode):
        return results.coefficient(name, ("cylinder", mode), ("cylinder", mode))

    w, scale = results.omegas, DENSITY * RADIUS**3
    got = np.c_[
        coefficient("added_mass", "heave") / scale,
        coefficient("radiation_damping", "heave") / (w * scale),
        coefficient("added_mass", "surge") / scale,
        coefficient("radiation_damping", "surge") / (w * scale),
    ]
    ref = REFERENCE[:, 1:]
    print("\nHydroBEM / eigenfunction matching:  A33/(rho a^3), B33/(rho w a^3), A11/(rho a^3), B11/(rho w a^3)")
    for ka, g, r in zip(REFERENCE[:, 0], got, ref, strict=True):
        print(f"  Ka = {ka:3.1f}:  " + ",  ".join(f"{x:.4f} / {y:.4f}" for x, y in zip(g, r, strict=True)))
    diff = (np.abs(got - ref).max(axis=0) / np.abs(ref).max(axis=0)).max()
    print(f"check: largest difference / largest reference value {diff:.1e} <= {TOLERANCE:.0e}:",
          "ok" if diff <= TOLERANCE else "NOT MET")  # fmt: skip


if __name__ == "__main__":
    main()
