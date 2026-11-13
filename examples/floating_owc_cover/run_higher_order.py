"""The floating OWC with a light piston on its inner free surface, in the higher order.

The case: the floating OWC of Raghavan, Loukogeorgaki, Mantadakis, Metrikine & Lavidas (2024),
HAMS-MREL, a new open source multiple body solver for marine renewable energies, Renewable Energy
237, 121577, doi:10.1016/j.renene.2024.121577, §3.4 (Table 4: a vertical tube of inner radius
r1 = 2 m, outer radius r2 = 3 m and draft H = 10 m), its water column covered by a rigid piston of
zero thickness (the light piston of the OWC theory of D. V. Evans (1982), Wave-power absorption by
systems of oscillating surface pressure distributions, J. Fluid Mech. 114, 481-499,
doi:10.1017/S0022112082000196): a free-surface plate, the body "piston" of a plate only, its mass 1 %
of the water column's.

How the problem is posed (options.higher_order = True): the quadrant of the analytic shape
"moonpool_cylinder" (radius r2, draft H, moonpool_radius r1) with its parts, the outer wall
("side"), the bottom annulus, the inner wall ("moonpool_wall") and the inner free surface
("moonpool_surface"); the hull is the body of the first three patches, the piston the body of the
fourth patch as its free-surface plate; the hull's interior surface, the annulus r1 < r < r2 in
z = 0, is a mesh of its own of one revolved-segment analytic patch. The potential: B-splines of
order 3, Gauss orders 3 (outer) and 4 (inner); mapping "nonuniform" clusters the spans at the
edges. Infinite depth, heading 0 deg, both bodies in heave, the diffraction problem in the total
potential, the default with a free-surface plate (diffraction_formulation "auto", not set here;
inside the column the incident and the scattered wave nearly cancel at the higher frequencies).

Also computed, as in run.py: A(infinity), the limit omega = infinity added to the frequencies, and
the mean drift of both bodies by momentum conservation in the far field (H. Maruo (1960), The drift
of a body floating on waves, J. Ship Res. 4(3), 1-10; J. N. Newman (1967), The drift force and
moment on ships in waves, J. Ship Res. 11(1), 51-60, doi:10.5957/jsr.1967.11.1.51). The drift by
pressure integration and its quadratic QTFs need the uniform mapping here (the nonuniform profiles
meet the waterline, and the engine refuses them), and the potential QTF is of the low order: both
are in run.py.

Prints the heave RAOs of the piston (the motion of the inner free surface) and of the hull,
A(infinity) of the heave pair and the surge drift, and checks the energy relation of the two bodies'
heave in deep water, B_ij = omega^3 Re(X_i conj(X_j)) / (2 rho g^3) (J. N. Newman, Marine
Hydrodynamics, 1977, the energy and Haskind relations). No page of the validation report compares
this model of the column. The same tube, its water column modelled by a piston body that fills it,
is compared in docs/validation/cases/floating_owc_piston.md. The page
docs/validation/cases/floating_owc_cover.md is about another body, a smaller tube whose moonpool is
closed by a cover of the hull's own body, and docs/validation/cases/dock.md checks the plate against
independent solutions.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_higher_order"  # the outputs of this script
R_INNER, R_OUTER, DRAFT = 2.0, 3.0, 10.0  # m
DENSITY, GRAVITY = 1000.0, 9.80665
OMEGAS = [0.4, 0.6, 0.8, 1.1, 1.4]  # rad/s
PISTON_MASS = 0.01 * DENSITY * np.pi * R_INNER**2 * DRAFT  # kg, a light piston
SPLINE_ORDER, OUTER_GAUSS, INNER_GAUSS = 3, 3, 4
SPANS = {  # the spans of the potential per patch of the quadrant: around, down or across
    "side": (6, 10),
    "bottom": (6, 2),
    "moonpool_wall": (4, 10),
    "moonpool_surface": (4, 4),
    "waterplane": (6, 2),
}
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
TOLERANCE = 0.01  # of the largest damping of the heave block


def row(part):
    """The discretization row of a patch: its spans, the spline orders and the Gauss orders."""
    return [*SPANS[part], SPLINE_ORDER, SPLINE_ORDER, OUTER_GAUSS, INNER_GAUSS]


def build() -> hb.Model:
    m = hb.Model()
    parts = ["side", "bottom", "moonpool_wall", "moonpool_surface"]
    m.meshes.analytic(
        name="owc",
        shape="moonpool_cylinder",
        discretization=[row(p) for p in parts],
        parts=parts,
        mapping="nonuniform",
        radius=R_OUTER,
        draft=DRAFT,
        moonpool_radius=R_INNER,
        reference_length=R_OUTER,
        symmetry="xz_and_yz",
        label="OWC tube r1 = 2 m, r2 = 3 m, H = 10 m",
    )
    waterplane = m.meshes.add(
        name="waterplane",
        kind="analytic_patches",
        discretization=[row("waterplane")],
        reference_length=R_OUTER,
        symmetry="xz_and_yz",
        label="the annulus between the walls in z = 0",
    )
    waterplane.patches.add(
        type="AnalyticPatch",
        map="line",
        profile="linear",
        a=1.0,
        b=1.0,
        theta0=0.0,
        theta1=0.5 * np.pi,
        p=[R_INNER, 0.0, R_OUTER, 0.0],  # the radius and z from the inner to the outer wall
    )
    m.bodies.add(
        name="hull",
        mesh="owc",
        ranges=[(1, 3)],  # the walls and the bottom
        rotation_center=[0.0, 0.0, 0.0],
        modes=["heave"],
        center_of_gravity=[0.0, 0.0, -DRAFT / 2],
        mass="displaced",
        interior_surface="mesh",
        interior_surface_mesh="waterplane",
    )
    piston = m.bodies.add(
        name="piston",  # a body of a plate only: no mesh of its own
        rotation_center=[0.0, 0.0, 0.0],
        modes=["heave"],
        center_of_gravity=[0.0, 0.0, 0.0],
        mass=PISTON_MASS,
    )
    piston.free_surface_plates.add(
        mesh="owc",
        ranges=[(4, 4)],  # the inner free surface
    )
    m.environment.update(
        depth=np.inf,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=[*OMEGAS, np.inf],  # with the limit omega = infinity
        headings=[0.0],
    )
    m.options.update(
        higher_order=True,
        frequency_threads=FREQUENCY_THREADS,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=True,
        diffraction=True,  # in the total potential, the default with a free-surface plate
        motion=True,
    )
    m.second_order.update(
        drift_momentum_conservation=True,
    )
    return m


def energy_check(results):
    """The largest difference of the heave damping block of hull and piston from omega^3 Re(X_i
    conj(X_j)) / (2 rho g^3), relative to its largest value."""
    w = results.omegas
    heave = [("hull", "heave"), ("piston", "heave")]
    b = np.stack([np.stack([results.coefficient("radiation_damping", i, j) for j in heave], -1) for i in heave], -2)
    x = np.stack([results.coefficient("excitation_diffraction", i)[:, 0] for i in heave], -1)
    energy = w[:, None, None] ** 3 * np.real(x[:, :, None] * np.conj(x[:, None, :])) / (2 * DENSITY * GRAVITY**3)
    return np.abs(b - energy).max() / np.abs(b).max()


def main():
    m = build()
    print(m, "| route:", m.route())
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / "floating_owc_cover.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())
    piston = np.abs(results.coefficient("motion", ("piston", "heave"))[:, 0])
    hull = np.abs(results.coefficient("motion", ("hull", "heave"))[:, 0])
    print("\nheave RAO [m/m]:")
    print("  omega [rad/s]  piston   hull")
    for w, a, b in zip(results.omegas, piston, hull, strict=True):
        print(f"  {w:13.2f} {a:7.3f} {b:6.3f}")
    heave = [("hull", "heave"), ("piston", "heave")]
    a_inf = [[results.coefficient("added_mass_infinite_frequency", i, j) for j in heave] for i in heave]
    print("\nA(infinity) of the heave [kg]:     hull       piston")
    for (name, _), row in zip(heave, a_inf, strict=True):
        print(f"  {name:6s}" + "".join(f"{x:13.1f}" for x in row))
    momentum = results.drift.momentum_conservation.real[:, 0, 0]  # of both bodies
    print("\nmean surge drift of hull and piston by momentum conservation [N/m^2]:")
    print("  omega [rad/s]   momentum")
    for omega, a in zip(results.omegas, momentum, strict=True):
        print(f"  {omega:13.2f} {a:10.1f}")
    diff = energy_check(results)
    print(f"check: energy relation of the heave damping, difference {diff:.1e} <= {TOLERANCE:.0e}: "
          f"{'ok' if diff <= TOLERANCE else 'NOT MET'}")  # fmt: skip
    print("saved:", path)


if __name__ == "__main__":
    main()
