"""A floating oscillating water column (OWC) as two bodies: the hull and a piston filling its column.

The case: Raghavan, Loukogeorgaki, Mantadakis, Metrikine & Lavidas (2024), HAMS-MREL, a new open
source multiple body solver for marine renewable energies, Renewable Energy 237, 121577,
doi:10.1016/j.renene.2024.121577, §3.4 "Case 4 - Floating OWC" (Table 4, Figs. 14, 16-18): a
simplified spar-buoy OWC modelled by the two-body method, the hull and an imaginary rigid piston that
fills the inner water column over its whole length, the two meshes overlapping on the inner wall.

The parameters taken:
- the hull: a vertical circular tube, inner radius r1 = 2 m, outer radius r2 = 3 m, draft H = 10 m
  (outer wall, bottom annulus, inner wall), 640 low-order panels;
- the piston: a vertical circular cylinder of radius r1 and length H, top in z = 0 (its side on the
  hull's inner wall, its bottom disc in the plane of the hull's bottom), 384 low-order panels;
- 16 panels around (the paper gives the counts, not their division), the whole surface;
- infinite depth; heading 0 deg; both bodies with the six rigid modes about their origin on the axis
  in the free surface.

The choices: as in the paper no interior surface (the first irregular frequency, of the piston's
interior, is near omega = 3.4 rad/s, above the paper's range); the motions are not solved.

The piston's heave added mass is also computed at omega = infinity, where the free surface is
phi = 0. The column mass plus this A33 is close to the A33 at omega = infinity of the massless plate
piston of the example floating_owc_cover, which models the same column by a free-surface plate on
its water (theory §9.7 of the documentation), and a little larger: the rigid piston holds the
column's water to one motion that the plate leaves free (Kelvin's minimum energy theorem; 0.5 % in
the higher order of both examples).

Prints the piston's heave added mass and damping (the paper's Figs. 17 and 18; its damping peaks at
about 760 kg/s near omega = 0.765 rad/s) at three frequencies and its added mass at omega = infinity,
and checks the energy relation of the two bodies' heave: for axisymmetric bodies in deep water the
heave radiates the axisymmetric wave only, and B_ij = omega^3 Re(X_i conj(X_j)) / (2 rho g^3), X the
heave exciting forces of the hull and the piston (the energy and Haskind relations, J. N. Newman,
Marine Hydrodynamics, 1977). The full comparison is in the validation report,
docs/validation/cases/floating_owc_piston.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
R_INNER, R_OUTER, DRAFT = 2.0, 3.0, 10.0  # m
HEADINGS = [0.0]  # deg
OMEGAS = [0.3, 0.765, 1.5]  # rad/s
N_THETA = 16  # panels around
N_OUTER, N_ANNULUS, N_INNER = 20, 2, 18  # the hull: outer wall down, bottom across, inner wall down (640)
N_PISTON_BOTTOM = 6  # the piston: its side the N_INNER rows of the inner wall, its bottom across (384)
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
TOLERANCE = 0.02  # of the largest damping of the heave block


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
    """The OWC hull: outer wall, bottom annulus and inner wall, normals into the water (outward on
    the outer wall, down on the bottom, towards the axis on the inner wall)."""
    return np.array(wall(R_OUTER, 0.0, -DRAFT, N_THETA, N_OUTER, outward=True)
                    + annulus(R_OUTER, R_INNER, -DRAFT, N_THETA, N_ANNULUS)
                    + wall(R_INNER, 0.0, -DRAFT, N_THETA, N_INNER, outward=False))  # fmt: skip


def piston_panels():
    """The piston: side wall on the hull's inner wall (normals outward, opposite to the hull's
    there) and bottom disc (normals down)."""
    return np.array(wall(R_INNER, 0.0, -DRAFT, N_THETA, N_INNER, outward=True)
                    + annulus(R_INNER, 0.0, -DRAFT, N_THETA, N_PISTON_BOTTOM))  # fmt: skip


def build() -> hb.Model:
    m = hb.Model()
    m.meshes.add(
        name="hull",
        kind="panels",
        vertices=hull_panels(),
        reference_length=1.0,
        symmetry="none",
        label="OWC hull r1 = 2 m, r2 = 3 m, H = 10 m, 640 panels",
    )
    m.meshes.add(
        name="piston",
        kind="panels",
        vertices=piston_panels(),
        reference_length=1.0,
        symmetry="none",
        label="OWC piston r1 = 2 m, length 10 m, 384 panels",
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
