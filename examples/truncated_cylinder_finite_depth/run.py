"""A floating truncated vertical cylinder in finite depth: heave and surge added mass and damping
against an eigenfunction-matching solution.

The case: a vertical circular cylinder of radius a floating with draft d = a in water of depth
h = 2a, radiating in heave and in surge. The reference is HydroBEM's own computation, not a
published table: the eigenfunction matching (Garrett 1971; Yeung 1981) of
scripts/gen_cylinder_reference.py, after R. W. Yeung (1981), Added mass and damping of a vertical
cylinder in finite-depth waters, Applied Ocean Research 3(3), 119-133,
doi:10.1016/0141-1187(81)90101-2, and C. J. R. Garrett (1971), Wave forces on a circular dock,
J. Fluid Mech. 46(1), 129-139, doi:10.1017/S0022112071000430 (heave as D. D. Bhatta & M. Rahman
(1995), Int. J. Math. Math. Sci. 18(1), 151-170, doi:10.1155/S0161171295000202; surge as
D. D. Bhatta (2003), Int. J. Math. Math. Sci. 2003(57), 3643-3656, doi:10.1155/S0161171203209285):
the potential expanded in the eigenfunctions under the cylinder and outside it, matched on r = a,
the truncation converged to better than 5e-6. An independent solution of the same linear problem.

The parameters taken:
- the cylinder of radius a = 1 m and draft d = 1 m in the depth h = 2 m (set DEPTH for another;
  the reference values below are those of h = 2 m), rho = 1000 kg/m^3, g = 9.80665 m/s^2;
- the frequencies Ka = 0.4, 0.8 and 1.5 (K = omega^2/g);
- the radiation problems of surge and heave about the centre of the waterplane (no diffraction,
  no motions); the normalization A/(rho a^3) and B/(rho omega a^3).

The choices of the example:
- low-order panels on one quadrant (the symmetry planes x = 0 and y = 0 give the rest): 48 around
  the quarter circle, 48 down the side and 24 across the bottom, cosine-spaced in depth and radius
  (small at the waterline and at the bottom edge, where the potential varies most);
- no interior surface: the first irregular frequency lies above the frequencies of the case;
- one frequency is solved at a time (FREQUENCY_THREADS = 1; more threads give the same results).

Prints the model, the summary of the results and the four coefficients beside the reference, with
the largest difference relative to the largest reference value of each quantity; saves
out/run/truncated_cylinder_h2.hbr. The full comparison is in the validation report,
docs/validation/cases/truncated_cylinder_finite_depth.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
RADIUS, DRAFT = 1.0, 1.0  # m
DEPTH = 2.0  # m
DENSITY, GRAVITY = 1000.0, 9.80665
N_THETA, N_Z, N_R = 48, 48, 24  # panels around the quarter circle, down the side, across the bottom
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)
# HydroBEM's eigenfunction matching (Garrett 1971; Yeung 1981), scripts/gen_cylinder_reference.py,
# h/a = 2, d/a = 1: Ka, A33/(rho a^3), B33/(rho omega a^3), A11/(rho a^3), B11/(rho omega a^3)
REFERENCE = np.array([
    [0.4, 1.940154076, 0.7107076064, 2.370760208, 0.5503827251],
    [0.8, 1.775587193, 0.3687504810, 2.144233779, 1.403494321],
    [1.5, 1.864562768, 0.07744079602, 0.8730149751, 1.541754199],
])  # fmt: skip
TOLERANCE = 3e-3  # of the largest reference value of each quantity


def cosine(n):
    """n intervals on [0, 1], refined at both ends."""
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))


def cylinder_quadrant(radius, draft, n_theta, n_z, n_r):
    """The quadrant x >= 0, y >= 0 of the wetted surface of a truncated vertical cylinder, axis z,
    top in z = 0: side and bottom, cosine-spaced in depth and radius; vertices ordered so that the
    normals point out of the body into the water (the panels at the axis are triangles)."""
    phi = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)
    z = -draft * cosine(n_z)
    r = radius * (1.0 - cosine(n_r))

    def p(rad, angle, depth):
        return [rad * np.cos(angle), rad * np.sin(angle), depth]

    side = [[p(radius, phi[j], z[i]), p(radius, phi[j], z[i + 1]), p(radius, phi[j + 1], z[i + 1]),
             p(radius, phi[j + 1], z[i])] for i in range(n_z) for j in range(n_theta)]  # fmt: skip
    bottom = [[p(r[i], phi[j], -draft), p(r[i + 1], phi[j], -draft), p(r[i + 1], phi[j + 1], -draft),
               p(r[i], phi[j + 1], -draft)] for i in range(n_r) for j in range(n_theta)]  # fmt: skip
    return np.array(side + bottom)


def build(ka) -> hb.Model:
    m = hb.Model()
    m.meshes.add(
        name="cylinder",
        kind="panels",
        vertices=cylinder_quadrant(RADIUS, DRAFT, N_THETA, N_Z, N_R),
        reference_length=RADIUS,
        symmetry="xz_and_yz",
        label=f"cylinder a = 1 m, d = 1 m, {N_THETA} panels a quarter circle",
    )
    m.bodies.add(
        name="cylinder",
        mesh="cylinder",
        rotation_center=[0.0, 0.0, 0.0],
        modes=["surge", "heave"],
        interior_surface="none",
    )
    m.environment.update(
        depth=DEPTH,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=np.sqrt(np.asarray(ka) * GRAVITY / RADIUS),
        headings=[],
    )
    m.options.update(
        frequency_threads=FREQUENCY_THREADS,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=True,
        diffraction=False,
        motion=False,
    )
    return m


def main():
    m = build(ka=REFERENCE[:, 0])
    print(m, "| route:", m.route())
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / f"truncated_cylinder_h{DEPTH:g}.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())
    print("saved:", path)

    def coefficient(name, mode):
        return results.coefficient(name, ("cylinder", mode), ("cylinder", mode))

    w, scale = results.omegas, DENSITY * RADIUS**3
    got = np.c_[
        coefficient("added_mass", "heave") / scale,
        coefficient("radiation_damping", "heave") / (w * scale),
        coefficient("added_mass", "surge") / scale,
        coefficient("radiation_damping", "surge") / (w * scale),
    ]
    ref = REFERENCE[:, 1:]
    print("\nHydroBEM / eigenfunction matching:  A33/(rho a^3), B33/(rho w a^3), A11/(rho a^3), B11/(rho w a^3)")
    for ka, g, r in zip(REFERENCE[:, 0], got, ref, strict=True):
        print(f"  Ka = {ka:3.1f}:  " + ",  ".join(f"{x:.4f} / {y:.4f}" for x, y in zip(g, r, strict=True)))
    diff = (np.abs(got - ref).max(axis=0) / np.abs(ref).max(axis=0)).max()
    print(f"check: largest difference / largest reference value {diff:.1e} <= {TOLERANCE:.0e}:",
          "ok" if diff <= TOLERANCE else "NOT MET")  # fmt: skip


if __name__ == "__main__":
    main()
