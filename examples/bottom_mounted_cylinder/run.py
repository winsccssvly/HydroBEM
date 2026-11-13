"""A fixed bottom-mounted vertical cylinder in head seas: the surge exciting force against the
closed form of MacCamy & Fuchs (1954) and the mean surge drift force against the analytic solution.

The case: a vertical circular cylinder of radius a standing on the sea bed and piercing the free
surface, in water of depth h, fixed, in regular waves of amplitude A along +x. The first-order
exciting force is the closed form of R. C. MacCamy & R. A. Fuchs (1954), Wave forces on piles: a
diffraction theory, Technical Memorandum No. 69, Beach Erosion Board, U.S. Army Corps of
Engineers; the mean drift force is the pressure integration of the same analytic scattering
solution, which in deep water is eq. (36) of T. H. Havelock (1940), The pressure of water waves
upon a fixed obstacle, Proc. R. Soc. Lond. A 175(963), 409-421, doi:10.1098/rspa.1940.0066.

The analytic solution (time factor e^{i omega t}, k tanh kh = nu = omega^2/g, Z(z) = cosh k(z + h)
/ cosh kh, H_m = H_m^(2) the Hankel function, eps_0 = 1, eps_m = 2): the total potential is

    phi = (g A / omega) Z(z) sum_m eps_m (-i)^m [J_m(kr) - J_m'(ka) H_m(kr) / H_m'(ka)] cos m theta,

and on the wall r = a, by the Wronskian J_m Y_m' - J_m' Y_m = 2/(pi ka),

    phi = (g A / omega) Z(z) S(theta),  S = sum_m eps_m (-i)^m (-2i) / (pi ka H_m'(ka)) cos m theta.

- The surge exciting force -int p n_x dS with p = -i omega rho phi (MacCamy & Fuchs 1954):
  |X_1| = 4 rho g A tanh(kh) / (k^2 |H_1'(ka)|).
- The mean force of the fixed body, n_x out of the body (the pressure integration of the
  second-order mean pressure; Havelock 1940 in deep water):
  F_x = (rho/4) int int |grad phi|^2 n_x dS - (rho g/4) oint |zeta|^2 n_x dl, zeta = -(i omega/g) phi(z = 0),
  with |grad phi|^2 = (g A/omega)^2 [Z^2 |S'|^2/a^2 + Z_z^2 |S|^2] on the wall and the depth
  integrals int Z^2 dz = (h/2 + sinh 2kh/(4k)) / cosh^2 kh and
  int Z_z^2 dz = k^2 (sinh 2kh/(4k) - h/2) / cosh^2 kh, from -h to 0.

The parameters taken:
- the cylinder of radius a = 1 m in the depth h = 2 m (h/a = 2), rho = 1000 kg/m^3,
  g = 9.80665 m/s^2, heading 0 deg;
- three frequencies, ka = 0.5, 1.0 and 2.0;
- the normalization: |X_1| / (rho g A a^2) and F_x / (rho g A^2 a).

The choices of the example:
- low-order panels on one quadrant of the wall (the symmetry planes x = 0 and y = 0 give the rest;
  no bottom: the cylinder stands on the sea bed), 32 around the quarter circle and 32 down the side,
  cosine-spaced in depth;
- the interior surface "graded" (a fan graded towards the waterline in z = 0 inside the column)
  removes the irregular frequencies, the first at ka = 2.405 (where J_0(ka) = 0);
- the mean drift by both methods of the engine: pressure integration over the body
  (drift_pressure_integration) and momentum conservation in the far field
  (drift_momentum_conservation). They converge to the same force, the momentum drift as fast as
  the exciting force, the pressure integration of constant panels (the velocities along the
  wall) more slowly: the check takes the exciting force and the momentum drift;
- one frequency is solved at a time (FREQUENCY_THREADS = 1; more threads give the same results).

Requires SciPy (the Bessel functions of the closed form).

Prints the model, the summary of the results and, at each ka, the exciting force and the two drift
forces beside the analytic values, with the largest difference relative to the largest analytic
value; saves out/run/bottom_mounted_cylinder.hbr. The full comparison is in the validation report,
docs/validation/cases/bottom_mounted_cylinder.md.
"""

import math
from pathlib import Path

import numpy as np
from scipy import special

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
RADIUS, DEPTH = 1.0, 2.0  # m
DENSITY, GRAVITY = 1000.0, 9.80665
KA = np.array([0.5, 1.0, 2.0])  # the wavenumber times the radius
N_THETA, N_Z = 32, 32  # panels around the quarter circle, down the side
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
TOLERANCE = 6e-3  # the exciting force and the momentum drift, of the largest analytic value


def hankel2_derivative(m, x):
    """H_m^(2)'(x) = J_m'(x) - i Y_m'(x)."""
    return special.jvp(m, x) - 1j * special.yvp(m, x)


def analytic(ka, h, a=1.0):
    """|X_1| / (rho g A a^2) and F_x / (rho g A^2 a) of the bottom-mounted cylinder (the formulas of
    the docstring), the series to the order ka + 40 and the integrals over theta by the
    trapezoidal rule of the periodic integrand."""
    k = ka / a
    nu = k * math.tanh(k * h)
    theta = np.linspace(0.0, 2.0 * np.pi, 4097)[:-1]
    s, ds = np.zeros_like(theta, dtype=complex), np.zeros_like(theta, dtype=complex)
    for m in range(int(ka + 40) + 1):
        c = (1.0 if m == 0 else 2.0) * (-1j) ** m * (-2j) / (math.pi * ka * hankel2_derivative(m, ka))
        s += c * np.cos(m * theta)
        ds -= m * c * np.sin(m * theta)
    force = 4.0 * math.tanh(k * h) / (k * k * abs(hankel2_derivative(1, ka))) / a**2
    c2 = math.cosh(k * h) ** 2
    z2 = (h / 2 + math.sinh(2 * k * h) / (4 * k)) / c2  # int Z^2 dz
    zz2 = k * k * (math.sinh(2 * k * h) / (4 * k) - h / 2) / c2  # int Z_z^2 dz

    def integral(f):  # over theta, 0 to 2 pi
        return f.mean() * 2.0 * np.pi

    # (g A/omega)^2 = g A^2/nu; dS = a dtheta dz, dl = a dtheta, n_x = cos theta
    wall = (a / 4) / nu * integral((z2 * abs(ds) ** 2 / a**2 + zz2 * abs(s) ** 2) * np.cos(theta))
    line = -(a / 4) * integral(abs(s) ** 2 * np.cos(theta))
    return force, (wall + line) / a


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
        label=f"bottom-mounted cylinder a = 1 m, h = 2 m, {N_THETA} x {N_Z} panels",
    )
    m.bodies.add(
        name="cylinder",
        mesh="cylinder",
        rotation_center=[0.0, 0.0, 0.0],
        modes=[],
        interior_surface="graded",
    )
    k = KA / RADIUS
    m.environment.update(
        depth=DEPTH,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=np.sqrt(GRAVITY * k * np.tanh(k * DEPTH)),
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
        drift_pressure_integration=True,
        drift_momentum_conservation=True,
    )
    return m


def main():
    m = build()
    print(m, "| route:", m.route())
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / "bottom_mounted_cylinder.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())
    print("saved:", path)

    ref = np.array([analytic(x, DEPTH, RADIUS) for x in KA])
    force = np.abs(results.coefficient("excitation_diffraction", ("cylinder", "surge"))[:, 0])
    force /= DENSITY * GRAVITY * RADIUS**2
    scale = DENSITY * GRAVITY * RADIUS
    pressure = results.bodies["cylinder"].drift.pressure_integration[:, 0, 0].real / scale  # [F, heading pair, D]
    momentum = results.drift.momentum_conservation[:, 0, 0].real / scale  # [F, heading pair, surge sway yaw]
    print("\n|X1|/(rho g A a^2) HydroBEM / MacCamy & Fuchs;  Fx/(rho g A^2 a) pressure, momentum / analytic")
    for i, x in enumerate(KA):
        print(f"  ka = {x:3.1f}:  {force[i]:.4f} / {ref[i, 0]:.4f};  "
              f"{pressure[i]:.4f}, {momentum[i]:.4f} / {ref[i, 1]:.4f}")  # fmt: skip
    diff = max(np.abs(force - ref[:, 0]).max() / np.abs(ref[:, 0]).max(),
               np.abs(momentum - ref[:, 1]).max() / np.abs(ref[:, 1]).max())  # fmt: skip
    print(f"check: largest difference / largest analytic value {diff:.1e} <= {TOLERANCE:.0e}:",
          "ok" if diff <= TOLERANCE else "NOT MET")  # fmt: skip


if __name__ == "__main__":
    main()
