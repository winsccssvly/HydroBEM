"""A floating hemisphere in deep water in the higher order: heave and surge added mass and damping
against Hulme (1982).

The case: A. Hulme (1982), The wave forces acting on a floating hemisphere undergoing forced
periodic oscillations, J. Fluid Mech. 121, 443-463, doi:10.1017/S0022112082001980, tables 1 and 2:
the hemisphere of radius a = 1 m floating with its equator in the free surface, deep water,
rho = 1000 kg/m^3, g = 9.80665 m/s^2, the radiation problems of surge and heave about the centre at
Ka = 0.5, 1.0 and 2.0 (K = omega^2/g), normalized as A/m0 and B/(omega m0), m0 = 2 pi rho a^3/3.

The higher-order mesh: the named analytic shape "sphere" of the engine, the quadrant x, y >= 0 as
two exact analytic patches (the symmetry planes x = 0 and y = 0 give the rest):
- the wetted hull, one patch from the waterline to the bottom pole, the potential a B-spline of
  order 3 on 8 x 8 spans, Gauss orders 3 (outer) and 4 (inner); the nonuniform mapping clusters the
  spans towards the waterline;
- the waterplane, the quarter disc in z = 0 inside the waterline, 8 x 2 spans, as the interior
  surface of the body (interior_surface "mesh", the second patch of the same mesh): it removes the
  irregular frequencies. The higher order takes as interior surface only patches: the generated
  surfaces "projected" and "graded" of the low order are refused.
The options: options.higher_order = True. The differences from the four-digit tables are their
rounding (about 1e-4 of the largest values).

Prints the model, the summary of the results and the four coefficients beside Hulme's values, with
the largest difference relative to the largest tabulated value of each quantity; saves
out/run_higher_order/floating_hemisphere.hbr. The full comparison is in the validation report,
docs/validation/cases/floating_hemisphere.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_higher_order"  # the outputs of this script
RADIUS = 1.0  # m
DENSITY, GRAVITY = 1000.0, 9.80665
HULL = (8, 8, 3, 3, 3, 4)  # spans around the quadrant and down the meridian, spline orders, Gauss orders
WATERPLANE = (8, 2, 3, 3, 3, 4)  # spans around the quadrant and along the radius, orders as the hull
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
# Hulme (1982), tables 1 and 2: Ka, A33/m0, B33/(omega m0), A11/m0, B11/(omega m0)
HULME = np.array([
    [0.5, 0.5861, 0.3391, 0.6439, 0.0987],
    [1.0, 0.4284, 0.2484, 0.5740, 0.3535],
    [2.0, 0.3884, 0.1031, 0.2493, 0.3424],
])  # fmt: skip
TOLERANCE = 5e-4  # of the largest tabulated value of each quantity (the tables' rounding: 1.5e-4)


def build(ka) -> hb.Model:
    m = hb.Model()
    m.meshes.analytic(
        name="hemisphere",
        shape="sphere",
        discretization=[HULL, WATERPLANE],
        parts=["hull", "waterplane"],
        mapping="nonuniform",
        radius=RADIUS,
        reference_length=RADIUS,
        symmetry="xz_and_yz",
        label="hemisphere a = 1 m, analytic patches, 8 x 8 spans",
    )
    m.bodies.add(
        name="hemisphere",
        mesh="hemisphere",
        ranges=[(1, 1)],  # the hull patch
        rotation_center=[0.0, 0.0, 0.0],
        modes=["surge", "heave"],
        interior_surface="mesh",
        interior_surface_mesh="hemisphere",
        interior_surface_ranges=[(2, 2)],  # the waterplane patch
    )
    m.environment.update(
        depth=np.inf,
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
    m = build(ka=HULME[:, 0])
    print(m, "| route:", m.route())
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / "floating_hemisphere.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())
    print("saved:", path)

    m0 = 2.0 / 3.0 * np.pi * DENSITY * RADIUS**3
    w = results.omegas

    def coefficient(name, mode):
        return results.coefficient(
            name=name,
            row=("hemisphere", mode),
            column=("hemisphere", mode),
        )

    got = np.c_[
        coefficient("added_mass", "heave") / m0,
        coefficient("radiation_damping", "heave") / (w * m0),
        coefficient("added_mass", "surge") / m0,
        coefficient("radiation_damping", "surge") / (w * m0),
    ]
    ref = HULME[:, 1:]
    print("\nHydroBEM / Hulme (1982):  A33/m0, B33/(w m0), A11/m0, B11/(w m0)")
    for ka, g, r in zip(HULME[:, 0], got, ref, strict=True):
        print(f"  Ka = {ka:3.1f}:  " + ",  ".join(f"{x:.4f} / {y:.4f}" for x, y in zip(g, r, strict=True)))
    diff = (np.abs(got - ref).max(axis=0) / np.abs(ref).max(axis=0)).max()
    print(f"check: largest difference / largest tabulated value {diff:.1e} <= {TOLERANCE:.0e}:",
          "ok" if diff <= TOLERANCE else "NOT MET")  # fmt: skip


if __name__ == "__main__":
    main()
