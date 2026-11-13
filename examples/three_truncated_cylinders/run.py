"""Three floating truncated vertical cylinders at the corners of an equilateral triangle, finite depth.

The case: Raghavan, Loukogeorgaki, Mantadakis, Metrikine & Lavidas (2024), HAMS-MREL, a new open
source multiple body solver for marine renewable energies, Renewable Energy 237, 121577,
doi:10.1016/j.renene.2024.121577, §3.3 "Case 3 - Three semi-submerged cylinders". It is the
three-cylinder configuration of Mavrakos (1991), Hydrodynamic coefficients for groups of interacting
vertical axisymmetric bodies, Ocean Engineering 18(5), 485-515, doi:10.1016/0029-8018(91)90027-N
(Fig. 3b, Figs. 5b-c: depth 40b, draught 3.65b, spacing 5b).

The parameters taken:
- three identical vertical circular cylinders, radius b = 3 m, draft 10.95 m, their axes at the
  corners of an equilateral triangle of side 15 m about the origin, cylinder 1 at (15/sqrt(3), 0);
  Table 3 of Raghavan et al. lists "Radius, b: 6 m", read as the diameter (Mavrakos' ratios give
  b = 3 m);
- water depth 120 m (DEPTH = math.inf gives the infinite-depth case of Raghavan et al.'s timing
  table); heading 0 deg;
- 900 low-order panels on each cylinder (Raghavan et al., Fig. 11): 30 around, 20 down the side,
  10 across the bottom, the whole surface;
- the six rigid modes of each cylinder about its origin on the axis in the free surface.

The choices: the interior free surface of each cylinder is panelled ("graded") to remove the
irregular frequencies; the motions are not solved.

Prints, at k b = 0.6, Mavrakos' coefficients a_i1^qp / pi = A / (rho pi b^3) and b_i1^qp / pi =
B / (rho omega pi b^3) (b^4 for the pitch-surge i = 5; qp = 11, 22 a cylinder's own, 12 the surge
force or pitch moment on cylinder 1 from the surge of cylinder 2) beside the values read by eye from
his Figs. 5b and 5c (reading accuracy about 0.05), and the differences. The full comparison
is in the validation report, docs/validation/cases/three_truncated_cylinders.md.
"""

import math
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
RADIUS, DRAFT = 3.0, 10.95  # m: b, 3.65b
SIDE = 15.0  # m, the side of the triangle, centre to centre (5b)
PAPER_DEPTH = 120.0  # m (40b), the depth of the check
DEPTH = PAPER_DEPTH  # m; math.inf for the infinite depth
HEADINGS = [0.0]  # deg
KB = [0.6]  # the wavenumber times the radius of the check
N_THETA, N_Z, N_R = 30, 20, 10  # panels around, down the side, across the bottom: 900
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


def cylinder_panels(radius, draft, n_theta, n_z, n_r):
    """The wetted surface of a truncated vertical cylinder, axis z, top in z = 0: side and bottom,
    vertices ordered so that the normals point out of the body into the water."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    z = np.linspace(0.0, -draft, n_z + 1)
    r = np.linspace(radius, 0.0, n_r + 1)

    def p(rad, angle, depth):
        return [rad * np.cos(angle), rad * np.sin(angle), depth]

    side = [[p(radius, phi[j], z[i]), p(radius, phi[j], z[i + 1]), p(radius, phi[j + 1], z[i + 1]),
             p(radius, phi[j + 1], z[i])] for i in range(n_z) for j in range(n_theta)]  # fmt: skip
    bottom = [[p(r[i], phi[j], -draft), p(r[i + 1], phi[j], -draft), p(r[i + 1], phi[j + 1], -draft),
               p(r[i], phi[j + 1], -draft)] for i in range(n_r) for j in range(n_theta)]  # fmt: skip
    return np.array(side + bottom)


def omegas_of_wavenumbers(k, depth, gravity):
    """The frequencies of the wavenumbers k by the dispersion relation omega^2 = g k tanh(k h)."""
    k = np.asarray(k, dtype=float)
    return np.sqrt(gravity * k * (np.tanh(k * depth) if math.isfinite(depth) else 1.0))


def build() -> hb.Model:
    m = hb.Model()
    m.meshes.add(
        name="cylinder",
        kind="panels",
        vertices=cylinder_panels(RADIUS, DRAFT, N_THETA, N_Z, N_R),
        reference_length=1.0,
        symmetry="none",
        label="cylinder b = 3 m, H = 10.95 m, 900 panels",
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
            interior_surface="graded",
        )
    m.environment.update(
        depth=DEPTH,
        frequency_input="omegas",
        omegas=omegas_of_wavenumbers(np.asarray(KB) / RADIUS, DEPTH, m.environment.gravity),
        headings=HEADINGS,
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
    if DEPTH != PAPER_DEPTH:  # no reference values
        print("saved:", path)
        return

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
