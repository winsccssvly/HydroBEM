"""Two submerged spheres side by side in infinite depth, waves at 45 degrees.

The case: Raghavan, Loukogeorgaki, Mantadakis, Metrikine & Lavidas (2024), HAMS-MREL, a new open
source multiple body solver for marine renewable energies, Renewable Energy 237, 121577,
doi:10.1016/j.renene.2024.121577, §3.2 "Case 2 - Two submerged spheres". It is case 2 of Wu (1995),
The interaction of water waves with a group of submerged spheres, Applied Ocean Research 17(3),
165-184, doi:10.1016/0141-1187(95)00010-0 (Fig. 1, §7, Figs. 2-10).

The parameters taken:
- two identical spheres of radius a = 3 m, centre-to-centre spacing 4a = 12 m along the x axis, their
  centres 1.5a = 4.5 m below the free surface (Wu §7; Table 2 of Raghavan et al. lists 9 m, but their
  curves, which agree with Wu's, are those of 4.5 m);
- infinite depth; heading 45 deg;
- 625 low-order panels on each sphere (Raghavan et al., Fig. 6): 25 meridians by 25 bands of
  latitude, the whole surface;
- the six rigid modes of each sphere about its centre.

The choices: submerged bodies have no interior surface and no irregular frequencies; the motions are
not solved (a neutrally buoyant submerged sphere has no hydrostatic restoring).

Prints, at nu a = 1 (nu = omega^2/g), Wu's quantities of sphere 2 with both spheres moving in the same
mode: a_jj = (A21 + A22) / (rho (4/3) pi a^3), b_jj = (B21 + B22) / (rho omega (4/3) pi a^3) and
f_j = |E2_j| / (rho omega^2 (4/3) pi a^3 A exp(-nu a)), beside the values read by eye from his
Figs. 2-10 (reading accuracy about 1 per cent), and the largest relative difference. The full
comparison is in the validation report, docs/validation/cases/two_submerged_spheres.md.
"""

from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
RADIUS = 3.0  # m: D = 6 m
SPACING = 12.0  # m, centre to centre (4a)
SUBMERGENCE = 4.5  # m, the depth of the centres (1.5a)
HEADINGS = [45.0]  # deg
NU_A = [1.0]  # the deep-water wavenumber times the radius of the check
N_MERIDIANS, N_BANDS = 25, 25  # 625 panels
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)

# Wu (1995), the curves of case 2 at nu a = 1 read by eye from Figs. 2, 3, 4 (a), 6, 7 (b) and 8 (f)
FIGS_2_10 = {"a11": 0.519, "a22": 0.455, "a33": 0.491, "b22": 0.109, "b33": 0.156, "f1": 0.562}
TOLERANCE = 0.05  # relative


def sphere_panels(radius, centre_depth, n_meridians, n_bands):
    """A sphere about (0, 0, -centre_depth), vertices ordered so that the normals point out of the
    sphere into the water (the panels at the poles have two coincident vertices)."""
    theta = np.linspace(0.0, np.pi, n_bands + 1)  # polar angle from the top
    phi = np.linspace(0.0, 2.0 * np.pi, n_meridians + 1)

    def p(t, f):
        return [radius * np.sin(t) * np.cos(f), radius * np.sin(t) * np.sin(f), -centre_depth + radius * np.cos(t)]

    return np.array([[p(theta[i], phi[j]), p(theta[i + 1], phi[j]), p(theta[i + 1], phi[j + 1]),
                      p(theta[i], phi[j + 1])] for i in range(n_bands) for j in range(n_meridians)])  # fmt: skip


def build() -> hb.Model:
    m = hb.Model()
    m.meshes.add(
        name="sphere",
        kind="panels",
        vertices=sphere_panels(RADIUS, SUBMERGENCE, N_MERIDIANS, N_BANDS),
        reference_length=1.0,
        symmetry="none",
        label="sphere D = 6 m, 625 panels",
    )
    for name, x in (("sphere_1", -SPACING / 2), ("sphere_2", SPACING / 2)):
        m.bodies.add(
            name=name,
            mesh="sphere",
            position=[x, 0.0, 0.0],
            rotation_center=[0.0, 0.0, -SUBMERGENCE],
            modes=["surge", "sway", "heave", "roll", "pitch", "yaw"],
            center_of_gravity=[0.0, 0.0, -SUBMERGENCE],
            mass="displaced",
            interior_surface="none",
        )
    m.environment.update(
        depth=np.inf,
        frequency_input="omegas",
        omegas=np.sqrt(m.environment.gravity * np.asarray(NU_A) / RADIUS),
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


def figure_quantities(results):
    """Wu's quantities of sphere 2, unit wave amplitude: {"a11": an array over the frequencies, ...}."""
    env = results.model.environment
    omega = results.omegas
    nu = omega**2 / env.gravity
    mass = env.density * 4.0 / 3.0 * np.pi * RADIUS**3
    q = {}
    for j, mode in enumerate(("surge", "sway", "heave"), start=1):
        row = ("sphere_2", mode)
        a = sum(results.coefficient("added_mass", row, (s, mode)) for s in ("sphere_1", "sphere_2"))
        b = sum(results.coefficient("radiation_damping", row, (s, mode)) for s in ("sphere_1", "sphere_2"))
        e = np.abs(results.coefficient("excitation_diffraction", row)[:, 0])
        q[f"a{j}{j}"] = a / mass
        q[f"b{j}{j}"] = b / (omega * mass)
        q[f"f{j}"] = e / (omega**2 * mass * np.exp(-nu * RADIUS))
    return q


def main():
    m = build()
    print(m, "| route:", m.route())
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    path = OUT / "two_submerged_spheres.hbr"
    m.save_results(
        path,
        overwrite=True,
    )
    print(results.summary())

    q = figure_quantities(results)
    print(f"\nnu a = {NU_A[0]}: HydroBEM / Wu (1995), Figs. 2-10 (read from the figures)")
    diff = 0.0
    for key, ref in FIGS_2_10.items():
        got = q[key][0]
        diff = max(diff, abs(got / ref - 1.0))
        print(f"  {key:>4} {got:7.4f} {ref:7.4f}")
    print(
        f"check: largest relative difference {diff:.1e} <= {TOLERANCE:.0e}: {'ok' if diff <= TOLERANCE else 'NOT MET'}"
    )
    print("saved:", path)


if __name__ == "__main__":
    main()
