"""The force of the second-order potential on a fixed bottom-mounted cylinder against Kim & Yue (1990).

The case: M.-H. Kim & D. K. P. Yue (1990), The complete second-order diffraction solution for an
axisymmetric body. Part 2. Bichromatic incident waves and body motions, J. Fluid Mech. 211,
557-593, doi:10.1017/S0022112090001690, table 3: a vertical circular cylinder of radius a standing
on the sea bed in water of depth h = a, fixed, in two regular waves of amplitudes A_j, A_l along
+x. The table gives the modulus of the horizontal force of the second-order potential at the
difference and the sum frequency, |f_2| / (rho g a A_j A_l), by their semi-analytic method, three
decimals each (the four values of KIM_YUE below), with nu = omega^2/g the deep-water wavenumber of
each first-order frequency. The force of the
second-order potential is the part of the second-order force that the quadratic products of the
first-order solution leave out (the potential QTF); the quadratic part is not compared here.

The parameters taken:
- the cylinder of radius a = 1 m in the depth h = 1 m (h/a = 1), rho = 1000 kg/m^3,
  g = 9.80665 m/s^2, heading 0 deg;
- the four first-order frequencies nu a = 1.0, 1.4, 1.6 and 2.0 (omega = sqrt(nu g)); the pairs
  of the table are among the pairs of the solve, which computes every pair;
- the normalization of the table: |f_2| / (rho g a A_j A_l), the surge force of the QTF per unit
  amplitudes.

The choices of the example:
- low-order panels on one quadrant of the wall (the symmetry planes x = 0 and y = 0 give the
  rest; no bottom: the cylinder stands on the sea bed): 12 around the quarter circle and 12 down
  the side, cosine-spaced in depth; no interior surface;
- the free surface of the potential QTF as the engine generates it by default: panels from the
  waterline to the inner radius R = 4a, their size the automatic one (a third of the shortest
  combined wavelength, at most a/4), the quadrature zone and the far field automatic;
- the potential QTF by both methods of the engine in one solve: indirect (Green's identity with an
  assisting radiation potential, qtf_potential_indirect) and direct (the second-order potential
  solved, qtf_potential_direct); they discretize the same problem in two ways and agree to the
  discretization error;
- the free surface keeps its automatic panel size on every body mesh, so the differences from the
  three-decimal values do not fall further with a finer body mesh (they stay within about 1 %);
- one frequency is solved at a time (FREQUENCY_THREADS = 1; more threads give the same results).

Prints the model, the summary of the results and the four values of both methods beside Kim &
Yue's, with the largest relative difference; saves out/run/second_order_bottom_cylinder.hbr. The
full comparison is in the validation report, docs/validation/cases/second_order_bottom_cylinder.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
# Kim & Yue (1990), table 3, semi-analytic, h/a = 1: (kind, nu_j a, nu_l a): |f_2| / (rho g a A_j A_l)
KIM_YUE = {
    ("difference", 1.0, 2.0): 1.936,
    ("difference", 1.4, 1.6): 0.435,
    ("sum", 1.0, 2.0): 2.656,
    ("sum", 1.4, 1.6): 2.875,
}
TOLERANCE = 2e-2  # relative
RADIUS, DEPTH = 1.0, 1.0  # m
DENSITY, GRAVITY = 1000.0, 9.80665
NU_A = [1.0, 1.4, 1.6, 2.0]  # the deep-water wavenumber times the radius of the first-order frequencies
N_THETA, N_Z = 12, 12  # panels around the quarter circle, down the side
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)


def cosine(n):
    """n intervals on [0, 1], refined at both ends."""
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))


def wall_quadrant(radius, depth, n_theta, n_z):
    """The quadrant x >= 0, y >= 0 of the wall of a vertical cylinder from z = 0 to the sea bed,
    cosine-spaced in depth; vertices ordered so that the normals point out of the body into the
    water."""
    phi = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)
    z = -depth * cosine(n_z)

    def p(angle, height):
        return [radius * np.cos(angle), radius * np.sin(angle), height]

    return np.array([[p(phi[j], z[i]), p(phi[j], z[i + 1]), p(phi[j + 1], z[i + 1]), p(phi[j + 1], z[i])]
                     for i in range(n_z) for j in range(n_theta)])  # fmt: skip


def build() -> hb.Model:
    m = hb.Model()
    m.meshes.add(
        name="cylinder",
        kind="panels",
        vertices=wall_quadrant(RADIUS, DEPTH, N_THETA, N_Z),
        reference_length=RADIUS,
        symmetry="xz_and_yz",
        label=f"bottom-mounted cylinder a = 1 m, h = 1 m, {N_THETA} x {N_Z} panels",
    )
    m.bodies.add(
        name="cylinder",
        mesh="cylinder",
        rotation_center=[0.0, 0.0, 0.0],
        modes=[],
        interior_surface="none",
    )
    m.environment.update(
        depth=DEPTH,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=np.sqrt(np.asarray(NU_A) / RADIUS * GRAVITY),
        headings=[0.0],
    )
    m.options.update(
        frequency_threads=FREQUENCY_THREADS,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=False,
        diffraction=True,
        motion=False,
    )
    m.second_order.update(
        qtf_potential_indirect=True,
        qtf_potential_direct=True,
        qtf_difference=True,
        qtf_sum=True,
        free_surface_method="panels",
        free_surface_panel_size="auto",
        free_surface_inner_radius="auto",
    )
    return m


def main():
    m = build()
    print(m, "| route:", m.route())
    points = {zone: len(m.free_surface_points(zone).weights) for zone in ("panelled", "quadrature", "outer_circle")}
    print("free-surface points of the potential QTF:", points)
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / "second_order_bottom_cylinder.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())
    print("saved:", path)

    potential = results.qtf.potential  # each [F, F, heading pair, D]
    print("\n|f_2|/(rho g a A_j A_l), (nu_j a, nu_l a): Kim & Yue / indirect, direct")
    worst = 0.0
    for (kind, a, b), ref in KIM_YUE.items():
        i, j = NU_A.index(a), NU_A.index(b)
        f2 = [abs(getattr(method, kind)[i, j, 0, 0]) / (DENSITY * GRAVITY * RADIUS)
              for method in (potential.indirect, potential.direct)]  # fmt: skip
        worst = max(worst, *(abs(f / ref - 1.0) for f in f2))
        print(f"  {kind:10s} ({a}, {b}):  {ref:.3f} / {f2[0]:.3f}, {f2[1]:.3f}")
    print(f"check: largest relative difference {worst:.1e} <= {TOLERANCE:.0e}:",
          "ok" if worst <= TOLERANCE else "NOT MET")  # fmt: skip


if __name__ == "__main__":
    main()
