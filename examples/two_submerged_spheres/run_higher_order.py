"""Two submerged spheres side by side in infinite depth, waves at 45 degrees, by the higher-order method.

The case: Raghavan, Loukogeorgaki, Mantadakis, Metrikine & Lavidas (2024), HAMS-MREL, a new open
source multiple body solver for marine renewable energies, Renewable Energy 237, 121577,
doi:10.1016/j.renene.2024.121577, §3.2 "Case 2 - Two submerged spheres"; case 2 of Wu (1995), The
interaction of water waves with a group of submerged spheres, Applied Ocean Research 17(3), 165-184,
doi:10.1016/0141-1187(95)00010-0 (Figs. 2-10).

The parameters taken: two spheres of radius a = 3 m, 12 m apart along x, their centres 4.5 m below the
free surface, infinite depth, heading 45 deg, the six rigid modes of each about its centre.

The choices: options.higher_order = True; the exact sphere as one analytic patch per quadrant, the
revolved arc (map "arc") of the meridian from the top pole to the bottom pole, centred at the mesh
origin, stored as the quadrant x, y >= 0 (symmetry "xz_and_yz") and expanded by the engine for each
body; each body is placed with its centre at its position (x = -6 or +6 m, z = -4.5 m), so the
rotation centre and the centre of gravity are the body origin. Per patch 4 spans around the quadrant
and 8 down the meridian, B-splines of order 3, Gauss orders 3 and 4. No interior surface: the spheres
are submerged.

Prints, at nu a = 1, Wu's quantities of sphere 2 with both spheres moving in the same mode (a_jj,
b_jj, f_j in his normalization) beside the values read by eye from his Figs. 2-10, and the largest
relative difference. The full comparison is in the validation report,
docs/validation/cases/two_submerged_spheres.md.
"""

import math
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_higher_order"  # the outputs of this script
RADIUS = 3.0  # m: D = 6 m
SPACING = 12.0  # m, centre to centre (4a)
SUBMERGENCE = 4.5  # m, the depth of the centres (1.5a)
HEADINGS = [45.0]  # deg
NU_A = [1.0]  # the deep-water wavenumber times the radius of the check
DISCRETIZATION = (4, 8, 3, 3, 3, 4)  # per quadrant patch: spans around, down the meridian; orders; Gauss orders
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)

# Wu (1995), the curves of case 2 at nu a = 1 read by eye from Figs. 2, 3, 4 (a), 6, 7 (b) and 8 (f)
FIGS_2_10 = {"a11": 0.519, "a22": 0.455, "a33": 0.491, "b22": 0.109, "b33": 0.156, "f1": 0.562}
TOLERANCE = 0.05  # relative


def build() -> hb.Model:
    m = hb.Model()
    mesh = m.meshes.add(
        name="sphere",
        kind="analytic_patches",
        discretization=[DISCRETIZATION],
        reference_length=1.0,
        symmetry="xz_and_yz",
        label="sphere D = 6 m about its centre, one arc patch per quadrant",
    )
    # x = (a cos(phi) cos(theta), a cos(phi) sin(theta), -a sin(phi)): theta from 0 to pi/2 along u,
    # phi from -pi/2 (the top pole) to pi/2 (the bottom pole) along v; u x v points into the sphere
    mesh.patches.add(
        None,
        type="AnalyticPatch",
        map="arc",
        profile="linear",
        a=RADIUS,
        b=RADIUS,
        theta0=0.0,
        theta1=0.5 * math.pi,
        p=[RADIUS, -0.5 * math.pi, 0.5 * math.pi, 0.0],
    )
    for name, x in (("sphere_1", -SPACING / 2), ("sphere_2", SPACING / 2)):
        m.bodies.add(
            name=name,
            mesh="sphere",
            position=[x, 0.0, -SUBMERGENCE],
            rotation_center=[0.0, 0.0, 0.0],
            modes=["surge", "sway", "heave", "roll", "pitch", "yaw"],
            center_of_gravity=[0.0, 0.0, 0.0],
            mass="displaced",
            interior_surface="none",
        )
    m.environment.update(
        depth=np.inf,
        frequency_input="omegas",
        omegas=np.sqrt(m.environment.gravity * np.asarray(NU_A) / RADIUS),
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
    """Wu's quantities of sphere 2, unit wave amplitude: {"a11": an array over the frequencies, ...}."""
    env = results.model.environment
    omega = results.omegas
    nu = omega**2 / env.gravity
    mass = env.density * 4.0 / 3.0 * np.pi * RADIUS**3
    q = {}
    for j, mode in enumerate(("surge", "sway", "heave"), start=1):
        row = ("sphere_2", mode)
        a = sum(results.coefficient("added_mass", row, (s, mode)) for s in ("sphere_1", "sphere_2"))
        b = sum(results.coefficient("radiation_damping", row, (s, mode)) for s in ("sphere_1", "sphere_2"))
        e = np.abs(results.coefficient("excitation_diffraction", row)[:, 0])
        q[f"a{j}{j}"] = a / mass
        q[f"b{j}{j}"] = b / (omega * mass)
        q[f"f{j}"] = e / (omega**2 * mass * np.exp(-nu * RADIUS))
    return q


def main():
    m = build()
    print(m, "| route:", m.route())
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / "two_submerged_spheres.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())

    q = figure_quantities(results)
    print(f"\nnu a = {NU_A[0]}: HydroBEM / Wu (1995), Figs. 2-10 (read from the figures)")
    diff = 0.0
    for key, ref in FIGS_2_10.items():
        got = q[key][0]
        diff = max(diff, abs(got / ref - 1.0))
        print(f"  {key:>4} {got:7.4f} {ref:7.4f}")
    print(
        f"check: largest relative difference {diff:.1e} <= {TOLERANCE:.0e}: {'ok' if diff <= TOLERANCE else 'NOT MET'}"
    )
    print("saved:", path)


if __name__ == "__main__":
    main()
