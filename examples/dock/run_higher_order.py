"""The floating circular dock of zero draft in finite depth, in the higher order.

The case: C. J. R. Garrett (1971), Wave forces on a circular dock, J. Fluid Mech. 46, 129-139,
doi:10.1017/S0022112071000430: a rigid circular disc of radius a floating in the free surface in
water of depth h = 2a, heaving. The reference: its added mass and damping by eigenfunction matching
(Garrett 1971; R. W. Yeung (1981), Added mass and damping of a vertical cylinder in finite-depth
waters, Applied Ocean Research 3, 119-133, doi:10.1016/0141-1187(81)90101-2, with the draft
d = 0), computed with HydroBEM's script scripts/gen_cylinder_reference.py (d = 0, the truncations
200, 400 and 800 extrapolated; converged to 1e-8).

The parameters taken: the disc of radius a = 10 m in z = 0, the depth h = 20 m, rho = 1000 kg/m^3,
g = 9.80665 m/s^2; Ka = 0.4, 0.8 and 1.5 and the limit omega = infinity; the radiation problem of
heave, and for the drift the diffraction problem of the heading 0 deg and the motions (the dock
free in heave, massless).

How the problem is posed (options.higher_order = True): the disc is a free-surface plate of the
body "dock", a body of a plate only. Its mesh: the quadrant of the disc as one analytic patch, a
radial line from the edge to the centre revolved through 90 degrees (map "line", the radius and z
of its ends in p; either direction: a plate is turned to face the water), the potential a
B-spline of order 3 on 6 x 8 spans (around, along the radius), Gauss orders 3 (outer) and 4
(inner).

Also computed, as in run.py: A33 at omega = infinity (in deep water half of the disc's (8/3) rho a^3
in unbounded fluid, J. N. Newman (1977), Marine Hydrodynamics, MIT Press, Fig. 4.8) and the mean
surge drift by momentum conservation (H. Maruo (1960), The drift of a body floating on waves,
J. Ship Res. 4(3), 1-10), the one drift available for a plate whose edge borders the free surface
(the velocity is singular at that edge; theory §9.7 of the documentation).

Prints A33/(rho a^3) and B33/(rho omega a^3) beside the eigenfunction matching, A33(infinity) and
the drift, and checks the largest relative difference of A33 and B33. The full comparison is in the
validation report, docs/validation/cases/dock.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_higher_order"  # the outputs of this script
RADIUS, DEPTH = 10.0, 20.0  # m
DENSITY, GRAVITY = 1000.0, 9.80665
DISC = (6, 8, 3, 3, 3, 4)  # spans around the quadrant and along the radius, spline orders, Gauss orders
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
# the eigenfunction matching of the dock (h/a = 2, d = 0), scripts/gen_cylinder_reference.py:
# Ka: (A33/(rho a^3), B33/(rho omega a^3))
REFERENCE = {0.4: (2.160952, 1.090685), 0.8: (1.872721, 1.032607), 1.5: (1.591845, 0.921432)}
TOLERANCE = 5e-4


def build() -> hb.Model:
    m = hb.Model()
    disc = m.meshes.add(
        name="disc",
        kind="analytic_patches",
        discretization=[DISC],
        reference_length=RADIUS,
        symmetry="xz_and_yz",
        label="the dock a = 10 m in z = 0, one analytic patch",
    )
    disc.patches.add(
        type="AnalyticPatch",
        map="line",
        profile="linear",
        a=1.0,
        b=1.0,
        theta0=0.0,
        theta1=0.5 * np.pi,
        p=[RADIUS, 0.0, 0.0, 0.0],  # the radius and z of the edge, then of the centre
    )
    dock = m.bodies.add(
        name="dock",  # a body of a plate only: no mesh of its own
        rotation_center=[0.0, 0.0, 0.0],
        modes=["heave"],
        center_of_gravity=[0.0, 0.0, 0.0],
        mass=0.0,  # a massless mat: its inertia is its added mass
    )
    dock.free_surface_plates.add(
        mesh="disc",
    )
    m.environment.update(
        depth=DEPTH,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=[*np.sqrt(np.array(list(REFERENCE)) * GRAVITY / RADIUS), np.inf],  # with omega = infinity
        headings=[0.0],
    )
    m.options.update(
        higher_order=True,
        frequency_threads=FREQUENCY_THREADS,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=True,
        diffraction=True,  # for the drift
        motion=True,  # the dock free in heave
    )
    # the mean drift by momentum conservation in the far field, the one drift of a plate whose edge
    # borders the free surface: there the vertical velocity jumps from the plate's to the free
    # surface's and the velocity across the edge is singular, so the drifts by pressure integration
    # and through a control surface (their line terms at the edge) and the potential QTF (its
    # free-surface forcing at the edge) have no finite value and are refused
    m.second_order.update(
        drift_momentum_conservation=True,
    )
    return m


def main():
    m = build()
    print(m, "| route:", m.route())
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / "dock.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())
    w = results.omegas
    a33 = results.coefficient("added_mass", ("dock", "heave"), ("dock", "heave")) / (DENSITY * RADIUS**3)
    b33 = results.coefficient("radiation_damping", ("dock", "heave"), ("dock", "heave")) / (DENSITY * w * RADIUS**3)
    print("\nheave of the dock, HydroBEM / eigenfunction matching:")
    print("   Ka   A33/(rho a^3)        B33/(rho omega a^3)")
    worst = 0.0
    for ka, a, b in zip(REFERENCE, a33, b33, strict=True):
        ra, rb = REFERENCE[ka]
        worst = max(worst, abs(a / ra - 1.0), abs(b / rb - 1.0))
        print(f"  {ka:3.1f}  {a:.5f} / {ra:.5f}   {b:.5f} / {rb:.5f}")
    a_inf = results.coefficient("added_mass_infinite_frequency", ("dock", "heave"), ("dock", "heave"))
    print(f"A33(infinity)/(rho a^3) {a_inf / (DENSITY * RADIUS**3):.5f}",
          "(in deep water 4/3, half of the disc in unbounded fluid)")  # fmt: skip
    drift = results.drift.momentum_conservation.real[:, 0, 0] / (DENSITY * GRAVITY * RADIUS)
    print("mean surge drift of the dock free in heave, F1/(rho g a A^2):", np.array2string(drift, precision=5))
    verdict = "ok" if worst <= TOLERANCE else "NOT MET"
    print(f"check: largest relative difference {worst:.1e} <= {TOLERANCE:.0e}: {verdict}")
    print("saved:", path)


if __name__ == "__main__":
    main()
