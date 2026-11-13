"""A floating oscillating water column (OWC) as two bodies, the hull and a piston filling its column,
by the higher-order method.

The case: Raghavan, Loukogeorgaki, Mantadakis, Metrikine & Lavidas (2024), HAMS-MREL, a new open
source multiple body solver for marine renewable energies, Renewable Energy 237, 121577,
doi:10.1016/j.renene.2024.121577, §3.4 "Case 4 - Floating OWC": the two-body method, the hull of the
OWC and an imaginary rigid piston that fills its water column, the piston's side on the hull's inner
wall.

The parameters taken: the hull a tube of inner radius r1 = 2 m, outer radius r2 = 3 m and draft
H = 10 m; the piston of radius r1 and length H; infinite depth, heading 0 deg, both bodies with their
six rigid modes about their origin on the axis in the free surface, no interior surface.

The choices: options.higher_order = True; quadrants of two analytic shapes (symmetry "xz_and_yz"),
the hull "moonpool_cylinder" (outer wall "side", bottom annulus, inner wall "moonpool_wall") and the
piston "circular_cylinder" (side, bottom disc); mapping "nonuniform" clusters the spans at the bottom
edges; B-splines of order 3, Gauss orders 3 and 4. The geometry is exact (circles, where the paper's
meshes are 16-sided polygons of 2.5 per cent less area: its A33 and B33 lie a few per cent lower).

The piston's heave added mass is also computed at omega = infinity, where the free surface is
phi = 0. The column mass plus this A33 is close to the A33 at omega = infinity of the massless plate
piston of the example floating_owc_cover, which models the same column by a free-surface plate on
its water (theory §9.7 of the documentation), and a little larger: the rigid piston holds the
column's water to one motion that the plate leaves free (Kelvin's minimum energy theorem; 0.5 % in
the higher order of both examples).

Prints the piston's heave added mass and damping at three frequencies and its added mass at
omega = infinity, and checks the energy relation of the two bodies' heave in deep water,
B_ij = omega^3 Re(X_i conj(X_j)) / (2 rho g^3), X the heave exciting forces of the hull and the
piston (the energy and Haskind relations, J. N. Newman, Marine Hydrodynamics, 1977). The full
comparison is in the validation report, docs/validation/cases/floating_owc_piston.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_higher_order"  # the outputs of this script
R_INNER, R_OUTER, DRAFT = 2.0, 3.0, 10.0  # m
HEADINGS = [0.0]  # deg
OMEGAS = [0.3, 0.765, 1.5]  # rad/s
SPLINE_ORDER, OUTER_GAUSS, INNER_GAUSS = 3, 3, 4
HULL_SPANS = [(3, 6), (3, 2), (2, 6)]  # the hull: outer wall, bottom annulus, inner wall (around, down or across)
PISTON_SPANS = [(2, 6), (2, 3)]  # the piston: side wall, bottom disc
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
TOLERANCE = 0.02  # of the largest damping of the heave block


def rows(spans):
    """The discretization rows of the patches: their spans, the spline orders and the Gauss orders."""
    return [[nu, nv, SPLINE_ORDER, SPLINE_ORDER, OUTER_GAUSS, INNER_GAUSS] for nu, nv in spans]


def build() -> hb.Model:
    m = hb.Model()
    m.meshes.analytic(
        name="hull",
        shape="moonpool_cylinder",
        discretization=rows(HULL_SPANS),
        parts=["side", "bottom", "moonpool_wall"],
        mapping="nonuniform",
        radius=R_OUTER,
        draft=DRAFT,
        moonpool_radius=R_INNER,
        reference_length=1.0,
        symmetry="xz_and_yz",
        label="OWC hull r1 = 2 m, r2 = 3 m, H = 10 m, analytic patches",
    )
    m.meshes.analytic(
        name="piston",
        shape="circular_cylinder",
        discretization=rows(PISTON_SPANS),
        parts=["side", "bottom"],
        mapping="nonuniform",
        radius=R_INNER,
        draft=DRAFT,
        reference_length=1.0,
        symmetry="xz_and_yz",
        label="OWC piston r1 = 2 m, length 10 m, analytic patches",
    )
    for name in ("hull", "piston"):
        m.bodies.add(
            name=name,
            mesh=name,
            position=[0.0, 0.0, 0.0],
            rotation_center=[0.0, 0.0, 0.0],
            modes=["surge", "sway", "heave", "roll", "pitch", "yaw"],
            center_of_gravity=[0.0, 0.0, -DRAFT / 2],
            mass="displaced",
            interior_surface="none",
        )
    m.environment.update(
        depth=np.inf,
        frequency_input="omegas",
        omegas=[*OMEGAS, np.inf],  # with the limit omega = infinity
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


def energy_check(results):
    """The largest difference of the heave damping block of hull and piston from omega^3 Re(X_i
    conj(X_j)) / (2 rho g^3), relative to its largest value."""
    env = results.model.environment
    w = results.omegas
    heave = [("hull", "heave"), ("piston", "heave")]
    b = np.stack([np.stack([results.coefficient("radiation_damping", i, j) for j in heave], -1) for i in heave], -2)
    x = np.stack([results.coefficient("excitation_diffraction", i)[:, 0] for i in heave], -1)
    energy = (
        w[:, None, None] ** 3 * np.real(x[:, :, None] * np.conj(x[:, None, :])) / (2 * env.density * env.gravity**3)
    )
    return np.abs(b - energy).max() / np.abs(b).max()


def main():
    m = build()
    print(m, "| route:", m.route())
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / "floating_owc_piston.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())
    column_mass = m.environment.density * np.pi * R_INNER**2 * DRAFT  # the water column the piston replaces
    a33 = results.coefficient("added_mass", ("piston", "heave"), ("piston", "heave"))
    b33 = results.coefficient("radiation_damping", ("piston", "heave"), ("piston", "heave"))
    print(f"\npiston heave (column mass rho pi r1^2 H = {column_mass:.0f} kg):")
    print("  omega [rad/s]  A33 [kg]  column + A33 [kg]  B33 [kg/s]")
    for w, a, b in zip(results.omegas, a33, b33, strict=True):
        print(f"  {w:13.3f} {a:9.1f} {column_mass + a:18.1f} {b:11.2f}")
    a_inf = results.coefficient("added_mass_infinite_frequency", ("piston", "heave"), ("piston", "heave"))
    print(f"  {'infinity':>13s} {a_inf:9.1f} {column_mass + a_inf:18.1f}")
    diff = energy_check(results)
    print(f"check: energy relation of the heave damping, difference {diff:.1e} <= {TOLERANCE:.0e}: "
          f"{'ok' if diff <= TOLERANCE else 'NOT MET'}")  # fmt: skip
    print("saved:", path)


if __name__ == "__main__":
    main()
