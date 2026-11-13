"""The heave of an open OWC chamber (a floating tube, no turbine) free in heave.

The case: S. A. Mavrakos & D. N. Konispoliatis (2012), Hydrodynamics of a free floating vertical
axisymmetric oscillating water column device, Journal of Applied Mathematics 2012, 142850,
doi:10.1155/2012/142850, Table 3 (page 24): the heave |x_30| / (H/2) of a tube of inner radius b,
outer radius a = 1.2 b and draught b/3 in the depth 2b, free in heave, beside the open moon pool of
its reference [29] (the table's first columns).

The parameters taken: b = 1 m, a = 1.2 m, draught 1/3 m, depth 2 m, heading 0; free in heave with the
mass of the displaced water (rho = 1000 kg/m^3, g = 9.80665 m/s^2, the defaults: the RAO does not
depend on them); the six frequencies omega^2 b / g of the table.

How the problem is posed: the chamber is open, its free surface the open sea's (not panelled); the
annulus between the walls in z = 0 is the interior surface (panels of the mesh), which removes the
irregular frequencies. The heave of an axisymmetric body does not couple with its surge and pitch in
head waves, so the centre of gravity and the pitch inertia, which the paper does not give, do not
enter. A quadrant: 24 panels around, 8 down each wall, 4 across the bottom and the annulus,
cosine-spaced.

Prints the heave RAO beside the table's open moon pool values and the largest difference. The
table's other ratios a/b are not compared; the validation report compares the chamber with a turbine,
docs/validation/cases/floating_owc_chamber.md.
"""

import time
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_open_heave"  # the outputs of this script
B_IN, A_OUT, DRAFT, DEPTH = 1.0, 1.2, 1.0 / 3.0, 2.0  # m
N_THETA, N_WALL, N_BOTTOM = 24, 8, 4  # panels of the quadrant
# Mavrakos & Konispoliatis (2012), Table 3, a/b = 1.2: omega^2 b / g and |x_30| / (H/2) of the open
# moon pool of their reference [29]
W2B_G = [0.037, 0.883, 1.223, 1.348, 1.761, 1.926]
PAPER = [0.9733, 0.9582, 0.5061, 0.2075, 0.3720, 0.6149]
# the bound, absolute, of the RAO: the largest difference, 0.024 at omega^2 b / g = 0.037, is the
# table's: there the heave RAO of a floating body is 1 + O(K b) (it follows the long wave) and
# HydroBEM gives 0.997 on these panels and on two and three times as many alike, the table 0.973;
# at the other frequencies the differences are 0.008 or less here and up to 0.017 (at 1.926, on the
# steep side of the resonance) on three times the panels, which approach a limit 0.02 below the table
TOLERANCE = 0.03


def spacing(n):
    """n + 1 fractions from 0 to 1, cosine-spaced: small at both ends."""
    return 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n + 1)))


def sweep(r, z, n_theta):
    """The quadrant 0 <= phi <= pi/2 of the surface of revolution of the meridian (r[i], z[i]): the normals
    point outward on a wall traversed downward, down on a ring traversed inward and up on a ring traversed
    outward."""
    phi = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)

    def p(i, j):
        return [r[i] * np.cos(phi[j]), r[i] * np.sin(phi[j]), z[i]]

    return [[p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)] for i in range(len(r) - 1) for j in range(n_theta)]


def tube():
    """The panels of the tube: the hull (outer wall down, bottom annulus inward, inner wall up, normals into
    the water), then the annulus between the walls in z = 0 (normals up); and the number of hull panels."""
    sw, sb = spacing(N_WALL), spacing(N_BOTTOM)
    r = np.r_[np.full(N_WALL + 1, A_OUT), A_OUT - (A_OUT - B_IN) * sb[1:], np.full(N_WALL, B_IN)]
    z = np.r_[-DRAFT * sw, np.full(N_BOTTOM, -DRAFT), -DRAFT * (1.0 - sw[1:])]
    hull = sweep(r, z, N_THETA)
    waterplane = sweep(B_IN + (A_OUT - B_IN) * sb, np.zeros(N_BOTTOM + 1), N_THETA)
    return np.array(hull + waterplane), len(hull)


def build() -> hb.Model:
    vertices, n_hull = tube()
    m = hb.Model()
    m.meshes.add(
        name="owc",
        kind="panels",
        vertices=vertices,
        reference_length=A_OUT,
        symmetry="xz_and_yz",
        label=f"open OWC tube a = {A_OUT:g} m, b = {B_IN:g} m, draught {DRAFT:.3g} m",
    )
    m.bodies.add(
        name="owc",
        mesh="owc",
        ranges=[(1, n_hull)],
        modes=["heave"],
        mass="displaced",
        center_of_gravity=[0.0, 0.0, -0.5 * DRAFT],
        interior_surface="mesh",
        interior_surface_mesh="owc",
        interior_surface_ranges=[(n_hull + 1, len(vertices))],
    )
    m.environment.update(
        depth=DEPTH,
        frequency_input="omegas",
        omegas=np.sqrt(np.array(W2B_G) * m.environment.gravity / B_IN),
        headings=[0.0],
    )
    m.options.update(
        frequency_threads=1,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=True,
        diffraction=True,
        motion=True,
    )
    return m


def main():
    start = time.perf_counter()
    m = build()
    print(m, "| route:", m.route())
    r = m.solve()
    m.save_results(OUT / "open_owc.hbr", overwrite=True)
    print(r.summary())
    rao = np.abs(r.motion[:, 0, r.coordinate("owc", "heave")[0]])
    print("heave |x_30| / (H/2): HydroBEM / Mavrakos & Konispoliatis Table 3 (open moon pool [29]), a/b = 1.2")
    print("  " + ",  ".join(f"{v:g}: {x:.4f} / {p:.4f}" for v, x, p in zip(W2B_G, rao, PAPER, strict=True)))
    worst = np.abs(rao - PAPER).max()
    print(f"check: largest difference {worst:.1e} <= {TOLERANCE:g}: {'ok' if worst <= TOLERANCE else 'NOT MET'}")
    print(f"saved: {OUT / 'open_owc.hbr'}; {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
