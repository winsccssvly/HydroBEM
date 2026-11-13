"""A floating oscillating water column (OWC) whose inner free surface is covered by a light piston.

The case: the floating OWC of Raghavan, Loukogeorgaki, Mantadakis, Metrikine & Lavidas (2024),
HAMS-MREL, a new open source multiple body solver for marine renewable energies, Renewable Energy
237, 121577, doi:10.1016/j.renene.2024.121577, §3.4 (Table 4: a vertical tube of inner radius
r1 = 2 m, outer radius r2 = 3 m and draft H = 10 m), with the water column modelled by a rigid
piston of zero thickness on its free surface (the light piston of the OWC theory of D. V. Evans
(1982), Wave-power absorption by systems of oscillating surface pressure distributions, J. Fluid
Mech. 114, 481-499, doi:10.1017/S0022112082000196, where the inner free surface moves as one): a
free-surface plate, the body "piston" of a plate only, its mass 1 % of the water column's.

The parameters taken:
- the hull: outer wall, bottom annulus and inner wall, 16 low-order panels around, the whole
  surface; its mass the displaced mass, its interior surface "projected" (the bottom annulus
  projected onto z = 0: the annulus r1 < r < r2 of the waterplane, which removes the irregular
  frequencies of the hull);
- the piston: the disc r < r1 in z = 0, 16 panels around (its edge on the hull's inner wall) and 6
  rings, as a body of its own with the mode heave and the mass 0.01 rho pi r1^2 H; its restoring
  is rho g pi r1^2, its waterplane;
- infinite depth, heading 0 deg, five frequencies across the resonance of the water column
  (omega about sqrt(g / H) = 1 rad/s); both bodies heave only (axisymmetric: the heave of the
  pair does not couple to the other modes in head seas);
- the diffraction problem in the total potential: the default (diffraction_formulation "auto")
  with a free-surface plate, not set here. Inside the column the incident and the scattered wave
  nearly cancel at the higher frequencies, and the scattered formulation would lose their small
  difference (its exciting force on the piston is several per cent off at omega = 1.1 rad/s on
  these panels).

The second-order loads with the plate (theory §9.7, §11, §12 of the documentation):
- A(infinity), the limit omega = infinity added to the frequencies: the free surface is phi = 0
  there and the piston a rigid thin sheet of the odd-image problem;
- the mean drift by momentum conservation in the far field (H. Maruo (1960), The drift of a body
  floating on waves, J. Ship Res. 4(3), 1-10; J. N. Newman (1967), The drift force and moment on
  ships in waves, J. Ship Res. 11(1), 51-60, doi:10.5957/jsr.1967.11.1.51), of both bodies
  together, and by pressure integration over each body (J. A. Pinkster (1980), Low frequency second
  order wave exciting forces on floating structures, PhD thesis, Delft University of Technology),
  available because the piston's edge lies on the hull's inner wall, along which it slides. The
  pressure drift converges slowly at the sharp bottom edges of the tube (as h^(1/3) in the panel
  size h, theory §11.2: the velocity is singular there, most of all when the bodies move against
  the water): here it misses the momentum drift by several times its size at the column's
  resonance (omega = 0.8 rad/s), and both agree to 1 % to 9 % above it. The momentum drift needs no
  velocity on the body and is the one to trust. The drift through a control surface (theory §11.3)
  is not available here: a control surface holds its body alone, and the piston, a body of its
  own, lies inside the hull's (the example trimmed_moonpool_cover has it for a cover fixed to its
  hull);
- the quadratic QTF at the difference frequencies by pressure integration (Pinkster 1980; its
  diagonal is the pressure drift) and the potential QTF, the force of the second-order potential,
  by the indirect method (B. Molin (1979), Second-order diffraction loads upon three-dimensional
  bodies, Applied Ocean Research 1(4), 197-202, doi:10.1016/0141-1187(79)90027-0), its free
  surface ending at the hull's outer waterline (the generated panels leave the piston out).

Prints the heave RAOs of the piston (the motion of the inner free surface) and of the hull,
A(infinity) of the heave pair, the surge drift by the two methods and the surge difference QTF of
both parts at three pairs of frequencies, and checks the energy relation of the two bodies' heave:
in deep water the heave of axisymmetric bodies radiates the axisymmetric wave only, and
B_ij = omega^3 Re(X_i conj(X_j)) / (2 rho g^3), X the heave exciting forces of the hull and the
piston (J. N. Newman, Marine Hydrodynamics, 1977, the energy and Haskind relations). No page of
the validation report compares this model of the column. The same tube, its water column modelled
by a piston body that fills it, is compared in docs/validation/cases/floating_owc_piston.md. The
page docs/validation/cases/floating_owc_cover.md is about another body, a smaller tube whose
moonpool is closed by a cover of the hull's own body, and docs/validation/cases/dock.md checks the
plate against independent solutions.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
R_INNER, R_OUTER, DRAFT = 2.0, 3.0, 10.0  # m
DENSITY, GRAVITY = 1000.0, 9.80665
OMEGAS = [0.4, 0.6, 0.8, 1.1, 1.4]  # rad/s
N_THETA = 16  # panels around
N_OUTER, N_ANNULUS, N_INNER = 20, 2, 18  # the hull: outer wall down, bottom across, inner wall down
N_PISTON = 6  # rings of the piston
PISTON_MASS = 0.01 * DENSITY * np.pi * R_INNER**2 * DRAFT  # kg, a light piston
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
TOLERANCE = 0.01  # of the largest damping of the heave block


def p(rad, angle, depth):
    return [rad * np.cos(angle), rad * np.sin(angle), depth]


def wall(radius, z_top, z_bottom, n_theta, n_z, outward):
    """A vertical cylindrical wall, its normals pointing away from the axis (outward) or to it."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    z = np.linspace(z_top, z_bottom, n_z + 1)
    panels = [[p(radius, phi[j], z[i]), p(radius, phi[j], z[i + 1]), p(radius, phi[j + 1], z[i + 1]),
               p(radius, phi[j + 1], z[i])] for i in range(n_z) for j in range(n_theta)]  # fmt: skip
    return panels if outward else [q[::-1] for q in panels]


def annulus(r_out, r_in, depth, n_theta, n_r):
    """A horizontal annulus (a disc with r_in = 0) at z = depth, its normals pointing down."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    r = np.linspace(r_out, r_in, n_r + 1)
    return [[p(r[i], phi[j], depth), p(r[i + 1], phi[j], depth), p(r[i + 1], phi[j + 1], depth),
             p(r[i], phi[j + 1], depth)] for i in range(n_r) for j in range(n_theta)]  # fmt: skip


def hull_panels():
    """The OWC hull, normals into the water (outward on the outer wall, down on the bottom, towards
    the axis on the inner wall)."""
    return np.array(wall(R_OUTER, 0.0, -DRAFT, N_THETA, N_OUTER, outward=True)
                    + annulus(R_OUTER, R_INNER, -DRAFT, N_THETA, N_ANNULUS)
                    + wall(R_INNER, 0.0, -DRAFT, N_THETA, N_INNER, outward=False))  # fmt: skip


def build() -> hb.Model:
    m = hb.Model()
    m.meshes.add(
        name="hull",
        kind="panels",
        vertices=hull_panels(),
        reference_length=1.0,
        symmetry="none",
        label="OWC hull r1 = 2 m, r2 = 3 m, H = 10 m",
    )
    m.meshes.add(
        name="cover",
        kind="panels",
        vertices=np.array(annulus(R_INNER, 0.0, 0.0, N_THETA, N_PISTON)),  # the disc r < r1 in z = 0
        reference_length=1.0,
        symmetry="none",
        label="the inner free surface r < 2 m",
    )
    m.bodies.add(
        name="hull",
        mesh="hull",
        rotation_center=[0.0, 0.0, 0.0],
        modes=["heave"],
        center_of_gravity=[0.0, 0.0, -DRAFT / 2],
        mass="displaced",
        interior_surface="projected",
    )
    piston = m.bodies.add(
        name="piston",  # a body of a plate only: no mesh of its own
        rotation_center=[0.0, 0.0, 0.0],
        modes=["heave"],
        center_of_gravity=[0.0, 0.0, 0.0],
        mass=PISTON_MASS,
    )
    piston.free_surface_plates.add(
        mesh="cover",
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
        drift_pressure_integration=True,
        qtf_quadratic_pressure_integration=True,
        qtf_potential_indirect=True,
        qtf_difference=True,
        qtf_sum=False,
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


def second_order(results):
    """Prints A(infinity) of the heave pair, the surge drift by both methods and the surge difference
    QTFs."""
    heave = [("hull", "heave"), ("piston", "heave")]
    a_inf = [[results.coefficient("added_mass_infinite_frequency", i, j) for j in heave] for i in heave]
    print("\nA(infinity) of the heave [kg]:     hull       piston")
    for (name, _), row in zip(heave, a_inf, strict=True):
        print(f"  {name:6s}" + "".join(f"{x:13.1f}" for x in row))
    w = results.omegas
    momentum = results.drift.momentum_conservation.real[:, 0, 0]  # of both bodies
    pressure = sum(results.bodies[b].drift.pressure_integration.real[:, 0, 0] for b in ("hull", "piston"))
    print("\nmean surge drift of hull and piston [N/m^2] (per unit wave amplitude squared):")
    print("  omega [rad/s]   momentum   pressure")
    for omega, a, b in zip(w, momentum, pressure, strict=True):
        print(f"  {omega:13.2f} {a:10.1f} {b:10.1f}")
    quadratic = sum(results.bodies[b].qtf.quadratic.pressure_integration.difference[:, :, 0, 0]
                    for b in ("hull", "piston"))  # fmt: skip
    potential = sum(results.bodies[b].qtf.potential.indirect.difference[:, :, 0, 0] for b in ("hull", "piston"))
    print("\nsurge difference QTF of hull and piston [N/m^2], |Q-(omega_a, omega_b)|:")
    print("  omega_a omega_b   quadratic   potential")
    for a, b in ((0, 1), (1, 2), (2, 3)):
        print(f"  {w[a]:7.2f} {w[b]:7.2f} {abs(quadratic[a, b]):11.1f} {abs(potential[a, b]):11.1f}")


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
    second_order(results)
    diff = energy_check(results)
    print(f"check: energy relation of the heave damping, difference {diff:.1e} <= {TOLERANCE:.0e}: "
          f"{'ok' if diff <= TOLERANCE else 'NOT MET'}")  # fmt: skip
    print("saved:", path)


if __name__ == "__main__":
    main()
