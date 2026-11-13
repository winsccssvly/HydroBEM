"""The OWC chamber with a linear air turbine in the higher order, restrained and free in heave.

The case: S. A. Mavrakos & D. N. Konispoliatis (2012), Hydrodynamics of a free floating vertical
axisymmetric oscillating water column device, Journal of Applied Mathematics 2012, 142850,
doi:10.1155/2012/142850, the device of Section 12: a vertical thick-walled tube open at its bottom,
the air in its chamber at a uniform pressure p_in0 that drives a Wells turbine of admittance g_T
(eq. 5.10).

The parameters taken: inner radius b = 2 m, outer radius a = 4 m, draught 5 m, water depth 15 m,
heading 0; g_T = 1 m^5/(kN s); the air incompressible; rho = 1025 kg/m^3, g = 9.81 m/s^2 (not stated
in the paper); free in heave with the mass of the displaced water, or restrained.

How the problem is posed: as in run.py, the inner free surface is a prescribed-pressure surface with
one uniform pressure mode "chamber" (coordinate xi, p_in0 = -rho g xi) and chamber_coupling; the
turbine is the mode's `additional_damping` B = (rho g)^2 g_T / omega^2, one solve per frequency. The
mesh (options.higher_order = True): the quadrant of the analytic shape "moonpool_cylinder" (radius a,
draft 5 m, moonpool_radius b) with its parts, the outer wall ("side"), the bottom annulus, the inner
wall ("moonpool_wall") and the inner free surface ("moonpool_surface", the pressure surface); the
interior surface, the annulus b < r < a in z = 0, is a mesh of its own of one revolved-segment
analytic patch. mapping="nonuniform" clusters the spans at the bottom edges, where the flow is
singular. The potential: B-splines of order 3, Gauss orders 3 (outer) and 4 (inner).

Prints the chamber pressure |p_in0| / (H/2) (kPa/m) at ka = 0.4, 1 and 2, restrained (Table 1) and
free in heave (Table 2), beside the paper's values, and the largest relative difference. The full
comparison is in the validation report, docs/validation/cases/floating_owc_chamber.md.
"""

import time
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run_higher_order"  # the outputs of this script
DENSITY, GRAVITY = 1025.0, 9.81  # kg/m^3, m/s^2 (not stated in the paper)
B_IN, A_OUT, DRAFT, DEPTH = 2.0, 4.0, 5.0, 15.0  # m: inner and outer radius, draught, water depth
G_T = 1.0e-3  # the turbine admittance, m^5/(N s): g_T = 1 m^5/(kN s) of the paper
SPLINE_ORDER, OUTER_GAUSS, INNER_GAUSS = 3, 3, 4
SPANS = {  # the spans of the potential per patch of the quadrant: around, down or across
    "side": (8, 12),
    "bottom": (8, 6),
    "moonpool_wall": (4, 12),
    "moonpool_surface": (4, 6),
    "waterplane": (8, 4),
}
# Mavrakos & Konispoliatis (2012), Tables 1 and 2 (page 21), the column g_T = 1: |p_in0| / (H/2) in
# kPa/m at ka: (restrained, free in heave)
PAPER = {0.4: (4.71, 0.568), 1.0: (1.26, 0.398), 2.0: (0.162, 0.0621)}
TOLERANCE = 0.02


def row(part):
    """The discretization row of a patch: its spans, the spline orders and the Gauss orders."""
    return [*SPANS[part], SPLINE_ORDER, SPLINE_ORDER, OUTER_GAUSS, INNER_GAUSS]


def build(omega, floating) -> hb.Model:
    """The device at one frequency, free in heave (floating) or restrained, the turbine the damping of
    the chamber mode."""
    m = hb.Model()
    parts = ["side", "bottom", "moonpool_wall", "moonpool_surface"]
    m.meshes.analytic(
        name="owc",
        shape="moonpool_cylinder",
        discretization=[row(p) for p in parts],
        parts=parts,
        mapping="nonuniform",
        radius=A_OUT,
        draft=DRAFT,
        moonpool_radius=B_IN,
        reference_length=A_OUT,
        symmetry="xz_and_yz",
        label=f"OWC tube a = {A_OUT:g} m, b = {B_IN:g} m, draught {DRAFT:g} m",
    )
    waterplane = m.meshes.add(
        name="waterplane",
        kind="analytic_patches",
        discretization=[row("waterplane")],
        reference_length=A_OUT,
        symmetry="xz_and_yz",
        label="the annulus between the walls in z = 0",
    )
    waterplane.patches.add(
        type="AnalyticPatch",
        map="line",
        profile="linear",
        a=1.0,
        b=1.0,
        theta0=0.0,
        theta1=0.5 * np.pi,
        p=[B_IN, 0.0, A_OUT, 0.0],  # the radius and z from the inner to the outer wall
    )
    body = m.bodies.add(
        name="owc",
        mesh="owc",
        ranges=[(1, 3)],  # the walls and the bottom
        modes=["heave"] if floating else [],
        mass="displaced",
        center_of_gravity=[0.0, 0.0, -0.5 * DRAFT],
        interior_surface="mesh",
        interior_surface_mesh="waterplane",
    )
    surface = body.pressure_surfaces.add(
        name="inner_free_surface",
        mesh="owc",
        ranges=[(4, 4)],  # the moonpool surface
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
        higher_order=True,
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
        k = ka / A_OUT
        omega = np.sqrt(GRAVITY * k * np.tanh(k * DEPTH))
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
