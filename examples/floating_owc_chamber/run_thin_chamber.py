"""A restrained thin-walled OWC chamber: its radiation admittance and the largest absorbed power.

The case: S. A. Mavrakos & D. N. Konispoliatis (2012), Hydrodynamics of a free floating vertical
axisymmetric oscillating water column device, Journal of Applied Mathematics 2012, 142850,
doi:10.1155/2012/142850, Fig. 10 (their comparison with Evans & Porter, their reference [7]): a
restrained thin-walled chamber, wall thickness 0.005 d, radius a = d/4, draught d/2 in the depth d.
The capture width of the best complex turbine admittance (eq. 11.4) is 1/k for an axisymmetric
chamber (Evans 1982, the bound of an axisymmetric pressure distribution): k times it is 1, the check
printed here.

The parameters taken: d = 10 m, a = 2.5 m, the wall 0.05 m thick, draught 5 m, heading 0; the
frequencies omega^2 d / g = 0.5, 1, 1.4 and 1.8 (about the peak of the absorption: away from it the
chamber radiates little, f_PP is small and the bound is met less closely on these panels); rho = 1000
kg/m^3, g = 9.80665 m/s^2 (the defaults).

How the problem is posed: the hull is a body without modes (restrained); its inner free surface is a
prescribed-pressure surface with one uniform pressure mode "chamber" (coordinate xi, the pressure
p_in = -rho g xi), whose radiation gives the admittance of the chamber and whose exciting force the
exciting volume flow. From the impedance Z = C - omega^2 A + i omega B of the mode (C its restoring):
the volume flow up q = -(f_PP + i g_PP) p_in, f_PP + i g_PP = -i omega Z / (rho g)^2 (eq. 5.9, time
factor e^{i omega t}), the exciting flow q_D = omega X / (rho g), and the largest mean power
|q_D|^2 / (8 f_PP) (eq. 11.4). The thin wall is panelled as a tube with both faces and one panel
across its bottom, no interior surface (its irregular frequencies lie far above these). A quadrant:
16 panels around, 36 down each wall, 16 rings across the inner free surface, cosine-spaced.

Prints f_PP, g_PP and k P_max / P_w at each frequency and the largest |k P_max / P_w - 1|. The full
comparison (the figure's other radii, the capture width of the best real admittance, Fig. 6) is in the
validation report, docs/validation/cases/floating_owc_chamber.md.
"""

import time
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_thin_chamber"  # the outputs of this script
DEPTH = 10.0  # m
A_OUT = 0.25 * DEPTH  # m, the outer radius
B_IN = A_OUT - 0.005 * DEPTH  # m, the inner radius
DRAFT = 0.5 * DEPTH  # m
NU_D = np.array([0.5, 1.0, 1.4, 1.8])  # omega^2 d / g
N_THETA, N_WALL, N_BOTTOM, N_DISC = 16, 36, 1, 16  # panels of the quadrant
TOLERANCE = 0.03


def spacing(n, fine):
    """n + 1 fractions from 0 to 1, cosine-spaced: small at both ends ("both") or at the start ("start")."""
    s = np.linspace(0.0, 1.0, n + 1)
    return 0.5 * (1.0 - np.cos(np.pi * s)) if fine == "both" else 1.0 - np.cos(0.5 * np.pi * s)


def sweep(r, z, n_theta):
    """The quadrant 0 <= phi <= pi/2 of the surface of revolution of the meridian (r[i], z[i]): the normals
    point outward on a wall traversed downward and down on a ring traversed inward (the panels at r = 0
    are triangles, a repeated vertex)."""
    phi = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)

    def p(i, j):
        return [r[i] * np.cos(phi[j]), r[i] * np.sin(phi[j]), z[i]]

    return [[p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)] for i in range(len(r) - 1) for j in range(n_theta)]


def tube():
    """The panels of the tube (outer wall down, bottom inward, inner wall up), then the inner free surface
    (normals down); and the number of tube panels."""
    sw, sb = spacing(N_WALL, "both"), spacing(N_BOTTOM, "both")
    r = np.r_[np.full(N_WALL + 1, A_OUT), A_OUT - (A_OUT - B_IN) * sb[1:], np.full(N_WALL, B_IN)]
    z = np.r_[-DRAFT * sw, np.full(N_BOTTOM, -DRAFT), -DRAFT * (1.0 - sw[1:])]
    hull = sweep(r, z, N_THETA)
    chamber = sweep(B_IN * (1.0 - spacing(N_DISC, "start")), np.zeros(N_DISC + 1), N_THETA)
    return np.array(hull + chamber), len(hull)


def build() -> hb.Model:
    vertices, n_hull = tube()
    m = hb.Model()
    m.meshes.add(
        name="owc",
        kind="panels",
        vertices=vertices,
        reference_length=A_OUT,
        symmetry="xz_and_yz",
        label=f"thin OWC tube a = {A_OUT:g} m, wall {A_OUT - B_IN:g} m, draught {DRAFT:g} m",
    )
    body = m.bodies.add(
        name="owc",
        mesh="owc",
        ranges=[(1, n_hull)],
        modes=[],  # restrained
        interior_surface="none",
    )
    surface = body.pressure_surfaces.add(
        name="inner_free_surface",
        mesh="owc",
        ranges=[(n_hull + 1, len(vertices))],
    )
    surface.pressure_modes.add(
        name="chamber",
        shape="uniform",
    )
    m.environment.update(
        depth=DEPTH,
        frequency_input="omegas",
        omegas=np.sqrt(NU_D * m.environment.gravity / DEPTH),
        headings=[0.0],
    )
    m.options.update(
        frequency_threads=1,
        inner_threads=1,
    )
    m.first_order.update(
        radiation=True,
        diffraction=True,
        motion=False,
    )
    return m


def wavenumber(omega, depth, g):
    """The wave number k of omega in the depth (omega^2 = g k tanh(k depth)), by Newton's method."""
    nu = omega**2 / g
    k = np.maximum(nu, np.sqrt(nu / depth))
    for _ in range(50):
        t = np.tanh(k * depth)
        k = k - (k * t - nu) / (t + k * depth * (1.0 - t * t))
    return k


def main():
    start = time.perf_counter()
    m = build()
    print(m, "| route:", m.route())
    r = m.solve()
    m.save_results(OUT / "thin_chamber.hbr", overwrite=True)
    print(r.summary())

    rho, g = m.environment.density, m.environment.gravity
    w = r.omegas
    i = r.coordinate("owc", "chamber")[0]
    a = r.coefficient("added_mass", ("owc", "chamber"), ("owc", "chamber"))
    b = r.coefficient("radiation_damping", ("owc", "chamber"), ("owc", "chamber"))
    z = r.mode_restoring[i, i] - w**2 * a + 1j * w * b  # the impedance of the chamber mode
    y = -1j * w * z / (rho * g) ** 2  # f_PP + i g_PP
    q_d = w * r.excitation.diffraction[:, 0, i] / (rho * g)  # the exciting volume flow
    p_max = np.abs(q_d) ** 2 / (8.0 * y.real)  # eq. 11.4, per unit wave amplitude squared
    k = wavenumber(w, DEPTH, g)
    cg = w / (2.0 * k) * (1.0 + 2.0 * k * DEPTH / np.sinh(2.0 * k * DEPTH))
    capture = k * p_max / (0.5 * rho * g * cg)  # k times the capture width, 1 in theory
    print("omega^2 d/g, f_PP and g_PP (m^5/(N s)), k P_max / P_w (1 in theory):")
    for v, f, c in zip(NU_D, y, capture, strict=True):
        print(f"  {v:4.1f}: f_PP {f.real:.4e}, g_PP {f.imag:+.4e}, k P_max / P_w {c:.4f}")
    worst = np.abs(capture - 1.0).max()
    verdict = "ok" if worst <= TOLERANCE else "NOT MET"
    print(f"check: largest |k P_max / P_w - 1| {worst:.1e} <= {TOLERANCE:g}: {verdict}")
    print(f"saved: {OUT / 'thin_chamber.hbr'}; {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
