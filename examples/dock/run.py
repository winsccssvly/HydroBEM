"""A floating circular dock of zero draft in finite depth: heave added mass and damping.

The case: C. J. R. Garrett (1971), Wave forces on a circular dock, J. Fluid Mech. 46, 129-139,
doi:10.1017/S0022112071000430: a rigid circular disc of radius a floating in the free surface (a
dock of zero draft) in water of depth h, here heaving with h/a = 2. Its added mass and damping by
eigenfunction matching (the inner region the whole depth under the disc; Garrett 1971, R. W.
Yeung (1981), Added mass and damping of a vertical cylinder in finite-depth waters, Applied Ocean
Research 3, 119-133, doi:10.1016/0141-1187(81)90101-2, with the draft d = 0), computed with
HydroBEM's script scripts/gen_cylinder_reference.py (its solve with d = 0, the truncations 200, 400
and 800 extrapolated; converged to 1e-8): the three values below.

The parameters taken:
- the disc of radius a = 10 m in z = 0, the water depth h = 20 m, rho = 1000 kg/m^3,
  g = 9.80665 m/s^2;
- the frequencies Ka = 0.4, 0.8 and 1.5, K = omega^2/g, and the limit omega = infinity; the
  radiation problem of heave, and for the drift the diffraction problem of the heading 0 deg and
  the motions, the dock free in heave and massless (a floating mat);
- the normalization A33/(rho a^3) and B33/(rho omega a^3).

How the problem is posed: the disc is a free-surface plate, a massless rigid plate in z = 0 with
the water below it only; the body "dock" is a body of a plate only (no mesh of its own). Its panels:
one quadrant (the symmetry planes x = 0 and y = 0 give the rest), 16 uniform rings and 32
divisions around the quadrant, normals down into the water. (At omega = 0 in finite depth the
heave of the plate has a net volume flux and is refused; in deep water its omega = 0 added mass is
Rayleigh's piston in a rigid plane, (8/3) rho a^3: Lord Rayleigh, The Theory of Sound, 1896, §307.)

Also computed: A33 at omega = infinity, where the free surface is phi = 0 and the plate, by the odd
image, a rigid disc in unbounded fluid with water on both faces (theory §9.7 of the documentation):
in deep water half of the disc's (8/3) rho a^3 (J. N. Newman (1977), Marine Hydrodynamics, MIT
Press, Fig. 4.8, the disc as a spheroid of zero thickness), raised a little by the depth h = 2a;
the low order approaches it as the panel size (the square-root edge of the sheet). And the mean
surge drift by momentum conservation in the far field (H. Maruo (1960), The drift of a body floating
on waves, J. Ship Res. 4(3), 1-10), normalized by rho g a A^2. That is the one mean drift of a plate
whose edge borders the free surface: the velocity is singular at such an edge, so the drifts by
pressure integration and through a control surface (their line terms at the edge) and the
potential QTF (its free-surface forcing there) have no finite value and are refused.

Prints A33/(rho a^3) and B33/(rho omega a^3) beside the eigenfunction matching, A33(infinity) and
the drift, and checks the largest relative difference of A33 and B33; the damping converges more
slowly than the added mass (the velocity is singular at the edge of the plate bordering the free
surface). The full comparison is in the validation report, docs/validation/cases/dock.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
RADIUS, DEPTH = 10.0, 20.0  # m
DENSITY, GRAVITY = 1000.0, 9.80665
N_RINGS, N_AROUND = 16, 32  # the quadrant of the disc
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
# the eigenfunction matching of the dock (h/a = 2, d = 0), scripts/gen_cylinder_reference.py:
# Ka: (A33/(rho a^3), B33/(rho omega a^3))
REFERENCE = {0.4: (2.160952, 1.090685), 0.8: (1.872721, 1.032607), 1.5: (1.591845, 0.921432)}
TOLERANCE = 5e-3


def disc_quadrant(radius, n_r, n_phi):
    """The quadrant x >= 0, y >= 0 of a horizontal disc in z = 0, uniform rings, normals down
    (into the water below; the panels at the centre are triangles: a repeated vertex)."""
    phi = np.linspace(0.0, 0.5 * np.pi, n_phi + 1)
    r = np.linspace(0.0, radius, n_r + 1)

    def p(rad, angle):
        return [rad * np.cos(angle), rad * np.sin(angle), 0.0]

    return np.array([[p(r[j], phi[i]), p(r[j], phi[i + 1]), p(r[j + 1], phi[i + 1]), p(r[j + 1], phi[i])]
                     for i in range(n_phi) for j in range(n_r)])  # fmt: skip


def build() -> hb.Model:
    m = hb.Model()
    m.meshes.add(
        name="disc",
        kind="panels",
        vertices=disc_quadrant(RADIUS, N_RINGS, N_AROUND),
        reference_length=RADIUS,
        symmetry="xz_and_yz",
        label="the dock a = 10 m in z = 0",
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
