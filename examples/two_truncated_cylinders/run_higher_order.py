"""Two floating truncated vertical cylinders side by side in finite depth, by the higher-order method.

The case: Raghavan, Loukogeorgaki, Mantadakis, Metrikine & Lavidas (2024), HAMS-MREL, a new open
source multiple body solver for marine renewable energies, Renewable Energy 237, 121577,
doi:10.1016/j.renene.2024.121577, §3.1 "Case 1 - Semi-immersed cylinders"; geometry 2 of Kagemoto &
Yue (1986), Interactions among multiple three-dimensional bodies in water waves: an exact algebraic
method, J. Fluid Mech. 166, 189-209, doi:10.1017/S0022112086000101.

The parameters taken: two cylinders D = 4 m, H = 4 m, 5.2 m apart along x, depth 8 m, heading 0 deg,
both with their six rigid modes about their origin on the axis in the free surface.

The choices: options.higher_order = True; the mesh is the named analytic shape "circular_cylinder"
(the exact side, bottom and waterplane), its quadrant stored (symmetry "xz_and_yz") and expanded by
the engine for each body at its position; the waterplane patch in z = 0 is the interior free surface
(interior_surface "detect"), which removes the irregular frequencies; mapping "nonuniform" clusters
the spans at the waterline and the bottom edge. Per patch 4 spans around the quadrant, 6 down the
side, 3 across the bottom and the waterplane, B-splines of order 3, Gauss orders 3 and 4.

Prints the totals over the two cylinders of Kagemoto & Yue (F1, N11, M11, F3, M33 in their
normalization (4.1)) at k D = 3.1 beside their table 2, and the largest relative difference. The full
comparison is in the validation report, docs/validation/cases/two_truncated_cylinders.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_higher_order"  # the outputs of this script
RADIUS, DRAFT = 2.0, 4.0  # m: D = 4 m, H = D
SPACING = 5.2  # m, centre to centre (L/D = 1.3)
DEPTH = 8.0  # m (h = 2D)
HEADINGS = [0.0]  # deg
KD = [3.1]  # the wavenumber times the diameter of the check
N_AROUND, N_DOWN, N_ACROSS = 4, 6, 3  # spans around the quadrant, down the side, across bottom and waterplane
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)

# Kagemoto & Yue (1986), table 2, the full diffraction computation at k D = 3.1
TABLE_2 = {"F1": 0.18828, "N11": 0.18146, "M11": 0.39797, "F3": 0.00779, "M33": 0.34852}
TOLERANCE = 0.05  # relative (the heave force, 0.008, is the least accurate)


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
        label="cylinder D = 4 m, H = 4 m, analytic patches",
        radius=RADIUS,
        draft=DRAFT,
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
            interior_surface="detect",
        )
    k = np.asarray(KD) / (2 * RADIUS)
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
