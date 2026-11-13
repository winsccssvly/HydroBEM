"""An oscillating water column (OWC) chamber with a linear air turbine, restrained and free in heave.

run.py: the device of Section 12 on low-order panels, out/run/floating_owc_chamber_ka_<ka>.hbr
run_higher_order.py: the same on exact-geometry patches
run_open_heave.py: an open chamber (no turbine) free in heave, the paper's Table 3
run_thin_chamber.py: restrained thin-walled chambers, the capture width of the paper's Fig. 10
(The exciting forces on the open device, the paper's Fig. 2, are the open tube of run_open_heave.py
with other radii and without the motions; its values are in the validation report.)

The case: S. A. Mavrakos & D. N. Konispoliatis (2012), Hydrodynamics of a free floating vertical
axisymmetric oscillating water column device, Journal of Applied Mathematics 2012, 142850,
doi:10.1155/2012/142850: a vertical thick-walled tube open at its bottom, the air in its chamber at
a uniform pressure p_in0 that drives a Wells turbine (a volume flow proportional to the pressure,
eq. 5.10, its admittance g_T), solved there by matched eigenfunction expansions.

The parameters taken (Section 12):
- inner radius b = 2 m, outer radius a = 4 m, draught 5 m, water depth 15 m, heading 0;
- the turbine g_T = 1 m^5/(kN s) (the paper works in t, kN, kPa); the air incompressible;
- rho = 1025 kg/m^3 and g = 9.81 m/s^2 (not stated in the paper);
- free in heave with the mass of the displaced water, or restrained (the heave and the chamber of
  an axisymmetric device do not couple with its surge and pitch in head waves).

How the problem is posed: the hull (outer wall, bottom annulus, inner wall) is a body; its inner free
surface is a prescribed-pressure surface with one uniform pressure mode "chamber", whose coordinate
xi is the pressure head, p_in0 = -rho g xi; chamber_coupling adds the air pressure on the roof to the
heave equation and the volume flow relative to the hull to the chamber equation. The turbine is the
damping of that mode: its power g_T |p_in0|^2 / 2 is B omega^2 |xi|^2 / 2, so B = (rho g)^2 g_T /
omega^2, which depends on the frequency: each frequency is a solve of its own with that
`additional_damping` and the motions solved. The annulus between the walls in z = 0 is the interior
surface (panels of the mesh), which removes the irregular frequencies.

The mesh: a quadrant (symmetry about x = 0 and y = 0), 16 panels around, 36 down each wall and 8
across the bottom, cosine-spaced (small at the waterline and at the bottom edge), 8 rings across the
chamber and 8 across the interior annulus.

Prints the chamber pressure |p_in0| / (H/2) (kPa/m) at ka = 0.4, 1 and 2, restrained (Table 1) and
free in heave (Table 2), beside the paper's values, and the largest relative difference. The full
comparison is in the validation report, docs/validation/cases/floating_owc_chamber.md.
"""

import time
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
DENSITY, GRAVITY = 1025.0, 9.81  # kg/m^3, m/s^2 (not stated in the paper)
B_IN, A_OUT, DRAFT, DEPTH = 2.0, 4.0, 5.0, 15.0  # m: inner and outer radius, draught, water depth
G_T = 1.0e-3  # the turbine admittance, m^5/(N s): g_T = 1 m^5/(kN s) of the paper
N_THETA, N_WALL, N_BOTTOM, N_DISC = 16, 36, 8, 8  # panels of the quadrant
# Mavrakos & Konispoliatis (2012), Tables 1 and 2 (page 21), the column g_T = 1: |p_in0| / (H/2) in
# kPa/m at ka: (restrained, free in heave)
PAPER = {0.4: (4.71, 0.568), 1.0: (1.26, 0.398), 2.0: (0.162, 0.0621)}
TOLERANCE = 0.05


def wavenumber_omega(k, depth):
    """The frequency of the wave number k in the depth, omega^2 = g k tanh(k depth)."""
    return np.sqrt(GRAVITY * k * np.tanh(k * depth))


def spacing(n, fine):
    """n + 1 fractions from 0 to 1, cosine-spaced: small at both ends ("both") or at the start ("start")."""
    s = np.linspace(0.0, 1.0, n + 1)
    return 0.5 * (1.0 - np.cos(np.pi * s)) if fine == "both" else 1.0 - np.cos(0.5 * np.pi * s)


def sweep(r, z, n_theta):
    """The quadrant 0 <= phi <= pi/2 of the surface of revolution of the meridian (r[i], z[i]): the normals
    point outward on a wall traversed downward, down on a ring traversed inward and up on a ring traversed
    outward (the panels at r = 0 are triangles, a repeated vertex)."""
    phi = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)

    def p(i, j):
        return [r[i] * np.cos(phi[j]), r[i] * np.sin(phi[j]), z[i]]

    return [[p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)] for i in range(len(r) - 1) for j in range(n_theta)]


def tube():
    """The panels of the tube and the elements of its parts: "hull" (outer wall down, bottom annulus inward,
    inner wall up, normals into the water), "chamber" (the inner free surface, normals down) and
    "waterplane" (the annulus between the walls in z = 0, normals up)."""
    sw, sb = spacing(N_WALL, "both"), spacing(N_BOTTOM, "both")
    r = np.r_[np.full(N_WALL + 1, A_OUT), A_OUT - (A_OUT - B_IN) * sb[1:], np.full(N_WALL, B_IN)]
    z = np.r_[-DRAFT * sw, np.full(N_BOTTOM, -DRAFT), -DRAFT * (1.0 - sw[1:])]
    parts = {
        "hull": sweep(r, z, N_THETA),
        "chamber": sweep(B_IN * (1.0 - spacing(N_DISC, "start")), np.zeros(N_DISC + 1), N_THETA),
        "waterplane": sweep(B_IN + (A_OUT - B_IN) * sb, np.zeros(N_BOTTOM + 1), N_THETA),
    }
    records, first = {}, 1
    for name, panels in parts.items():
        records[name] = (first, first + len(panels) - 1)
        first += len(panels)
    return np.array([q for panels in parts.values() for q in panels]), records


def build(omega, floating) -> hb.Model:
    """The device at one frequency, free in heave (floating) or restrained, the turbine the damping of
    the chamber mode."""
    vertices, records = tube()
    m = hb.Model()
    m.meshes.add(
        name="owc",
        kind="panels",
        vertices=vertices,
        reference_length=A_OUT,
        symmetry="xz_and_yz",
        label=f"OWC tube a = {A_OUT:g} m, b = {B_IN:g} m, draught {DRAFT:g} m",
    )
    body = m.bodies.add(
        name="owc",
        mesh="owc",
        ranges=[records["hull"]],
        modes=["heave"] if floating else [],
        mass="displaced",
        center_of_gravity=[0.0, 0.0, -0.5 * DRAFT],
        interior_surface="mesh",
        interior_surface_mesh="owc",
        interior_surface_ranges=[records["waterplane"]],
    )
    surface = body.pressure_surfaces.add(
        name="inner_free_surface",
        mesh="owc",
        ranges=[records["chamber"]],
        chamber_coupling=True,
    )
    surface.pressure_modes.add(
        name="chamber",
        shape="uniform",
        additional_damping=(DENSITY * GRAVITY) ** 2 * G_T / omega**2,  # the turbine
    )
    m.environment.update(
        depth=DEPTH,
        density=DENSITY,
        gravity=GRAVITY,
        frequency_input="omegas",
        omegas=[omega],
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
    print("chamber pressure |p_in0| / (H/2) (kPa/m), g_T = 1: HydroBEM / Mavrakos & Konispoliatis Tables 1, 2")
    worst = 0.0
    for ka, paper in PAPER.items():
        omega = wavenumber_omega(ka / A_OUT, DEPTH)
        got = []
        for floating in (False, True):
            m = build(omega, floating)
            r = m.solve()
            m.save_results(OUT / f"floating_owc_chamber_ka_{ka:g}{'_free' if floating else ''}.hbr", overwrite=True)
            xi = r.motion[0, 0, r.coordinate("owc", "chamber")[0]]  # the pressure head per unit amplitude
            got.append(abs(xi) * DENSITY * GRAVITY / 1.0e3)
        if ka == 0.4:
            print(m, "| route:", m.route())
            print(r.summary())
        worst = max(worst, *(abs(g / p - 1.0) for g, p in zip(got, paper, strict=True)))
        print(f"  ka = {ka:g}: restrained {got[0]:.4g} / {paper[0]:.4g}, free in heave {got[1]:.4g} / {paper[1]:.4g}")
    verdict = "ok" if worst <= TOLERANCE else "NOT MET"
    print(f"check: largest relative difference {worst:.1e} <= {TOLERANCE:g}: {verdict}")
    print(f"saved: {OUT}; {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
