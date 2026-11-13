"""Two floating truncated vertical cylinders side by side in finite depth, in head seas.

The case: Raghavan, Loukogeorgaki, Mantadakis, Metrikine & Lavidas (2024), HAMS-MREL, a new open
source multiple body solver for marine renewable energies, Renewable Energy 237, 121577,
doi:10.1016/j.renene.2024.121577, §3.1 "Case 1 - Semi-immersed cylinders". It is geometry 2 of
Kagemoto & Yue (1986), Interactions among multiple three-dimensional bodies in water waves: an exact
algebraic method, J. Fluid Mech. 166, 189-209, doi:10.1017/S0022112086000101 (§4, h/D = 2, H/D = 1,
L/D = 1.3).

The parameters taken:
- two identical vertical circular cylinders, diameter D = 4 m, draft H = 4 m, centre-to-centre
  spacing L = 5.2 m along the x axis; water depth h = 8 m (DEPTH = math.inf gives the infinite-depth
  case of Raghavan et al.'s timing table); heading 0 deg (along the line of the cylinders);
- 1200 low-order panels on each cylinder (Raghavan et al., Fig. 3): 40 around, 20 down the side,
  10 across the bottom, the whole surface;
- both cylinders with their six rigid modes about their origin on the axis in the free surface.

The choices: the interior free surface of each cylinder is panelled ("graded") to remove the
irregular frequencies; the motions are not solved (the masses are not part of the case).

Prints the totals over the two cylinders that Kagemoto & Yue plot (the surge exciting force F1, the
surge damping N11 and added mass M11, the heave exciting force F3 and added mass M33, in their
normalization (4.1)) at k D = 3.1 beside their table 2 (the full diffraction computation), and the
largest relative difference. The full comparison is in the validation report,
docs/validation/cases/two_truncated_cylinders.md.
"""

import math
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
RADIUS, DRAFT = 2.0, 4.0  # m: D = 4 m, H = D
SPACING = 5.2  # m, centre to centre (L/D = 1.3)
PAPER_DEPTH = 8.0  # m (h = 2D), the depth of the check
DEPTH = PAPER_DEPTH  # m; math.inf for the infinite depth
HEADINGS = [0.0]  # deg
KD = [3.1]  # the wavenumber times the diameter of the check
N_THETA, N_Z, N_R = 40, 20, 10  # panels around, down the side, across the bottom: 1200
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)

# Kagemoto & Yue (1986), table 2, the full diffraction computation at k D = 3.1
TABLE_2 = {"F1": 0.18828, "N11": 0.18146, "M11": 0.39797, "F3": 0.00779, "M33": 0.34852}
TOLERANCE = 0.05  # relative (the heave force, 0.008, is the least accurate)


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
        label="cylinder D = 4 m, H = 4 m, 1200 panels",
    )
    for name, x in (("cylinder_1", -SPACING / 2), ("cylinder_2", SPACING / 2)):
        m.bodies.add(
            name=name,
            mesh="cylinder",
            position=[x, 0.0, 0.0],
            rotation_center=[0.0, 0.0, 0.0],
            modes=["surge", "sway", "heave", "roll", "pitch", "yaw"],
            center_of_gravity=[0.0, 0.0, 0.0],
            mass="displaced",
            interior_surface="graded",
        )
    m.environment.update(
        depth=DEPTH,
        frequency_input="omegas",
        omegas=omegas_of_wavenumbers(np.asarray(KD) / (2 * RADIUS), DEPTH, m.environment.gravity),
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
    """The totals over the two cylinders in the normalization (4.1) of Kagemoto & Yue, N = 2 legs,
    unit wave amplitude: {name: an array over the frequencies}."""
    env = results.model.environment
    rho, g, omega = env.density, env.gravity, results.omegas
    scale = np.pi * RADIUS * DRAFT * 2  # pi a H N
    bodies = ("cylinder_1", "cylinder_2")

    def total(name, mode):
        return sum(results.coefficient(name, (b, mode), (c, mode)) for b in bodies for c in bodies)

    def force(mode):
        return np.abs(sum(results.coefficient("excitation_diffraction", (b, mode)) for b in bodies)[:, 0])

    return {
        "F1": force("surge") / (rho * g * scale),
        "N11": total("radiation_damping", "surge") / (omega * rho * RADIUS * scale),
        "M11": total("added_mass", "surge") / (rho * RADIUS * scale),
        "F3": force("heave") / (rho * g * scale),
        "M33": total("added_mass", "heave") / (rho * RADIUS * scale),
    }


def main():
    m = build()
    print(m, "| route:", m.route())
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / "two_truncated_cylinders.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())
    if DEPTH != PAPER_DEPTH:  # no reference values
        print("saved:", path)
        return

    q = figure_quantities(results)
    print(f"\nk D = {KD[0]}: HydroBEM / Kagemoto & Yue (1986), table 2")
    diff = 0.0
    for key, ref in TABLE_2.items():
        got = q[key][0]
        diff = max(diff, abs(got / ref - 1.0))
        print(f"  {key:>4} {got:9.5f} {ref:9.5f}")
    print(
        f"check: largest relative difference {diff:.1e} <= {TOLERANCE:.0e}: {'ok' if diff <= TOLERANCE else 'NOT MET'}"
    )
    print("saved:", path)


if __name__ == "__main__":
    main()
