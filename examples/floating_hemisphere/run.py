"""A floating hemisphere in deep water: heave and surge added mass and damping against Hulme (1982).

The case: A. Hulme (1982), The wave forces acting on a floating hemisphere undergoing forced
periodic oscillations, J. Fluid Mech. 121, 443-463, doi:10.1017/S0022112082001980, tables 1 and 2:
the added mass and damping of the heave (his mode 0) and the surge (mode 1) of a hemisphere of
radius a floating with its equator in the free surface, in deep water.

The parameters taken:
- the hemisphere of radius a = 1 m, its centre in the free surface, deep water, rho = 1000 kg/m^3,
  g = 9.80665 m/s^2;
- three of the tabulated frequencies, Ka = 0.5, 1.0 and 2.0 (K = omega^2/g);
- the radiation problems of surge and heave about the centre (no diffraction, no motions);
- the normalization of the tables: A/m0 and B/(omega m0) with m0 = 2 pi rho a^3/3, the displaced
  mass.

The choices of the example:
- low-order panels on one quadrant (the symmetry planes x = 0 and y = 0 give the rest), 32
  divisions from the waterline to the bottom pole and 32 around the quadrant (1024 panels; set
  DIVISIONS for a finer or coarser mesh), the polar divisions graded towards the waterline;
- the interior surface "projected" (the downward-facing panels projected onto z = 0) removes the
  irregular frequencies (the first near Ka = 3.9);
- one frequency is solved at a time (FREQUENCY_THREADS = 1; more threads give the same results).

Prints the model, the summary of the results and the four coefficients beside Hulme's values at
the three frequencies, with the largest difference relative to the largest tabulated value of each
quantity; saves out/run/floating_hemisphere.hbr. The full comparison is in the validation report,
docs/validation/cases/floating_hemisphere.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
RADIUS = 1.0  # m
DENSITY, GRAVITY = 1000.0, 9.80665
DIVISIONS = 32  # per quadrant edge: waterline to pole, and around the quadrant
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
# Hulme (1982), tables 1 and 2: Ka, A33/m0, B33/(omega m0), A11/m0, B11/(omega m0)
HULME = np.array([
    [0.5, 0.5861, 0.3391, 0.6439, 0.0987],
    [1.0, 0.4284, 0.2484, 0.5740, 0.3535],
    [2.0, 0.3884, 0.1031, 0.2493, 0.3424],
])  # fmt: skip
TOLERANCE = 5e-3  # of the largest tabulated value of each quantity


def hemisphere_quadrant(radius, n_polar, n_azimuth):
    """The quadrant x >= 0, y >= 0 of the wetted hemisphere, vertices on the sphere, the polar
    divisions graded towards the waterline; vertices ordered so that the normals point out of the
    body into the water (the panels at the bottom pole are triangles: a repeated vertex)."""
    s = 1.0 - np.cos(0.5 * np.pi * np.linspace(0.0, 1.0, n_polar + 1))  # fine at s = 0, the waterline
    theta = 0.5 * np.pi * (1.0 + s)  # the polar angle from +z: pi/2 at the waterline, pi at the pole
    phi = np.linspace(0.0, 0.5 * np.pi, n_azimuth + 1)

    def p(t, f):
        return [radius * np.sin(t) * np.cos(f), radius * np.sin(t) * np.sin(f), radius * np.cos(t)]

    return np.array([[p(theta[i], phi[j]), p(theta[i + 1], phi[j]), p(theta[i + 1], phi[j + 1]),
                      p(theta[i], phi[j + 1])] for i in range(n_polar) for j in range(n_azimuth)])  # fmt: skip


def build(ka) -> hb.Model:
    m = hb.Model()
    m.meshes.add(
        name="hemisphere",
        kind="panels",
        vertices=hemisphere_quadrant(RADIUS, DIVISIONS, DIVISIONS),
        reference_length=RADIUS,
        symmetry="xz_and_yz",
        label=f"hemisphere a = 1 m, {DIVISIONS} graded divisions",
    )
    m.bodies.add(
        name="hemisphere",
        mesh="hemisphere",
        rotation_center=[0.0, 0.0, 0.0],
        modes=["surge", "heave"],
        interior_surface="projected",
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
        return results.coefficient(name, ("hemisphere", mode), ("hemisphere", mode))

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
