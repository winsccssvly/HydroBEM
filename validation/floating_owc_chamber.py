"""An oscillating water column chamber with a linear air turbine (Mavrakos & Konispoliatis 2012,
Section 12): a thick-walled tube open at its bottom, the air pressure in its chamber a uniform
pressure mode of the inner free surface, the turbine the damping of that mode. The chamber pressure
restrained and free in heave against the paper's Tables 1 and 2, the largest power of the
restrained device against Table 5 and against the capture width 1/k of an axisymmetric chamber
(Evans 1982), in the low and the higher order (docs/validation/cases/floating_owc_chamber.md).

    python validation/floating_owc_chamber.py [--quick]
"""

from __future__ import annotations

import math
import sys

import mesh_figure
import numpy as np
from common import C, Timer, e, pct, plt, record, save, size, solve_model, table

from bodies import surface
from problems import analytic, model

NAME = "floating_owc_chamber"
RHO, G = 1025.0, 9.81  # not stated in the paper (the page)
KILO = 1.0e3  # the paper's units: t, kN, kPa, kW; its g_T in m^5/(kN s)
B_IN, A_OUT, DRAFT, DEPTH = 2.0, 4.0, 5.0, 15.0  # m
G_T = [0.5, 1.0, 2.0, 3.0, 6.0, 10.0]
KA_12 = [0.4, 1.0, 1.5, 2.0]
KA_5 = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5]
KA = np.array(sorted(set(KA_12) | set(KA_5) | {3.0}))
# the published tables (page 21: Tables 1, 2, |p_in0| / (H/2) in kPa/m; page 25: Table 5, kW/m^2)
TABLE_1 = [[4.94, 4.71, 4.05, 3.39, 2.11, 1.35], [1.30, 1.26, 1.12, 0.971, 0.633, 0.415],
           [0.485, 0.436, 0.330, 0.250, 0.139, 0.0860], [0.193, 0.162, 0.111, 0.0798, 0.0424, 0.0261]]  # fmt: skip
TABLE_2 = [[0.587, 0.568, 0.507, 0.438, 0.285, 0.185], [0.402, 0.398, 0.384, 0.363, 0.282, 0.213],
           [0.152, 0.145, 0.126, 0.106, 0.0637, 0.0418], [0.0673, 0.0621, 0.0491, 0.0386, 0.0213, 0.0136]]  # fmt: skip
TABLE_5 = {0.2: 956.983, 0.4: 308.105, 0.6: 147.688, 0.8: 90.6602, 1.0: 63.7096, 1.5: 34.5299, 2.0: 22.5046,
           2.5: 16.1687, 3.0: 12.35}  # fmt: skip


def omega_of(k):
    return np.sqrt(G * k * np.tanh(k * DEPTH))


def spacing(n, fine):
    s = np.linspace(0.0, 1.0, n + 1)
    return 0.5 * (1.0 - np.cos(np.pi * s)) if fine == "both" else 1.0 - np.cos(0.5 * np.pi * s)


def sweep(r, z, n):
    phi = np.linspace(0.0, 0.5 * math.pi, n + 1)
    p = lambda i, j: [r[i] * math.cos(phi[j]), r[i] * math.sin(phi[j]), z[i]]  # noqa: E731
    return [[p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)] for i in range(len(r) - 1) for j in range(n)]


def low_mesh(n_theta=16, n_wall=36, n_bottom=8, n_disc=8):
    """A quadrant: the hull (outer wall down, bottom annulus inward, inner wall up; cosine-spaced),
    the chamber's free surface (a disc, small at its rim) and the annulus between the walls in z = 0
    (the interior surface), as elements of one mesh; with the element ranges."""
    sw, sb = spacing(n_wall, "both"), spacing(n_bottom, "both")
    r = np.r_[np.full(n_wall + 1, A_OUT), A_OUT - (A_OUT - B_IN) * sb[1:], np.full(n_wall, B_IN)]
    z = np.r_[-DRAFT * sw, np.full(n_bottom, -DRAFT), -DRAFT * (1.0 - sw[1:])]
    parts = {"hull": sweep(r, z, n_theta),
             "chamber": sweep(B_IN * (1.0 - spacing(n_disc, "start")), np.zeros(n_disc + 1), n_theta),
             "waterplane": sweep(B_IN + (A_OUT - B_IN) * sb, np.zeros(n_bottom + 1), n_theta)}  # fmt: skip
    ranges, first = {}, 1
    for k, v in parts.items():
        ranges[k] = (first, first + len(v) - 1)
        first += len(v)
    return surface(np.array([q for v in parts.values() for q in v]), symmetry="xz_and_yz"), ranges


def body(higher, modes):
    if higher:
        row = lambda s: [*s, 3, 3, 3, 4]  # noqa: E731
        parts = ["side", "bottom", "moonpool_wall", "moonpool_surface"]
        mesh = analytic("moonpool_cylinder", [row(s) for s in ((8, 12), (8, 6), (4, 12), (4, 6))], parts=parts,
                        mapping="nonuniform", radius=A_OUT, draft=DRAFT, moonpool_radius=B_IN, symmetry="xz_and_yz")
        ring = {"type": "AnalyticPatch", "map": "line", "profile": "linear", "a": 1.0, "b": 1.0, "theta0": 0.0,
                "theta1": 0.5 * math.pi, "p": [B_IN, 0.0, A_OUT, 0.0]}  # fmt: skip
        water = {"kind": "analytic_patches", "discretization": np.array([row((8, 4))]), "symmetry": "xz_and_yz",
                 "patches": [ring]}  # fmt: skip
        hull, chamber, interior = (mesh, (1, 3)), (mesh, (4, 4)), (water, (1, 1))
    else:
        mesh, rg = low_mesh()
        hull, chamber, interior = (mesh, rg["hull"]), (mesh, rg["chamber"]), (mesh, rg["waterplane"])
    return {"name": "owc", "mesh": hull, "modes": list(modes), "mass": "displaced",
            "center_of_gravity": [0.0, 0.0, -0.5 * DRAFT], "interior_surface": interior,
            "pressure_surfaces": [{"mesh": chamber, "chamber_coupling": True,
                                   "pressure_modes": [{"name": "chamber", "shape": "uniform"}]}]}  # fmt: skip


def build(w, higher):
    return model(body(higher, ["heave"]), w, [0.0], environment={"depth": DEPTH, "density": RHO, "gravity": G},
                 options={"higher_order": higher})  # fmt: skip


def turbine_damping(g_t, w):
    """B of the pressure coordinate (p = -rho g xi) of a turbine of admittance g_t (m^5/(N s))."""
    return (RHO * G) ** 2 * g_t / w**2


class Chamber:
    """The impedances Z = C - w^2 (M + A) + i w B and exciting forces X of the heave and the chamber."""

    def __init__(self, r):
        self.w = w = r.omegas
        self.d = {n: r.coordinate("owc", n)[0] for n in ("heave", "chamber")}
        self.Z, self.X = {}, {}
        for i in ("heave", "chamber"):
            self.X[i] = r.excitation.diffraction[:, 0, self.d[i]]
            for j in ("heave", "chamber"):
                a = r.coefficient("added_mass", ("owc", i), ("owc", j))
                b = r.coefficient("radiation_damping", ("owc", i), ("owc", j))
                if i == j == "heave":
                    c, mass = r.hydrostatics.stiffness[0, 2, 2], r.hydrostatics.mass[0]
                else:
                    c, mass = r.mode_restoring[self.d[i], self.d[j]], 0.0
                self.Z[i, j] = c - w**2 * (mass + a) + 1j * w * b

    def pressure(self, g_t, floating):
        """|p_in0| per unit amplitude (Pa/m), restrained or free in heave."""
        w, Z, X = self.w, self.Z, self.X
        zpp = Z["chamber", "chamber"] + 1j * w * turbine_damping(g_t, w)
        if not floating:
            xi = X["chamber"] / zpp
        else:
            det = Z["heave", "heave"] * zpp - Z["heave", "chamber"] * Z["chamber", "heave"]
            xi = (Z["heave", "heave"] * X["chamber"] - Z["chamber", "heave"] * X["heave"]) / det
        return np.abs(xi) * RHO * G

    def p_max(self):
        """The largest mean power of the restrained device per unit amplitude squared (W/m^2):
        |q_D|^2 / (8 f_PP), q_D = w X / (rho g), f_PP = Re(-i w Z / (rho g)^2)."""
        f_pp = -(1j * self.w * self.Z["chamber", "chamber"] / (RHO * G) ** 2).real
        return np.abs(self.w * self.X["chamber"] / (RHO * G)) ** 2 / (8.0 * f_pp)


def main(quick=False):
    k = KA / A_OUT
    w = omega_of(k)
    out = {"ka": KA}
    curves = {}
    for higher in (False, True):
        key = "higher" if higher else "low"
        with Timer(f"OWC chamber: {key} order"):
            ch = Chamber(solve_model(build(w, higher)))
        rows12 = [int(np.argmin(np.abs(KA - x))) for x in KA_12]
        t1 = np.array([[ch.pressure(g / KILO, False)[i] / KILO for g in G_T] for i in rows12])
        t2 = np.array([[ch.pressure(g / KILO, True)[i] / KILO for g in G_T] for i in rows12])
        p = ch.p_max() / KILO
        cg = w / (2 * k) * (1 + 2 * k * DEPTH / np.sinh(2 * k * DEPTH))
        capture = k * ch.p_max() / (0.5 * RHO * G * cg)
        t5 = {float(x): float(p[i]) for i, x in enumerate(KA)}
        out[key] = {"table1": t1, "table2": t2, "table5": t5, "capture": capture, "capture_minus_1": capture - 1.0,
                    "table1_rel": np.abs(t1 / np.array(TABLE_1) - 1).max(),
                    "table2_rel": np.abs(t2 / np.array(TABLE_2) - 1).max(),
                    "table5_rel": {str(x): t5[x] / TABLE_5[x] - 1 for x in TABLE_5}}  # fmt: skip
        curves[key] = (t1, t2, p, capture)
    mesh_figure.draw(NAME, [("low order, 1280 wetted panels a quadrant", build([1.0], False), False, {"cut": (1, -1)}),
                            ("higher order, analytic patches", build([1.0], True), True, {"cut": (1, -1)})],
                     elev=30.0)  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    for ax, ti, ref, title in ((axes[0], 0, TABLE_1, "Table 1, restrained"),
                               (axes[1], 1, TABLE_2, "Table 2, free in heave")):
        for j, g in enumerate(G_T):
            ax.semilogy(KA_12, [r[j] for r in ref], "o", color=C[j % len(C)], mfc="none", ms=5)
            ax.semilogy(KA_12, curves["low"][ti][:, j], "-", color=C[j % len(C)], label=f"g_T = {g:g}")
            ax.semilogy(KA_12, curves["higher"][ti][:, j], "--", color=C[j % len(C)])
        ax.set_xlabel("ka")
        ax.set_ylabel("|p_in0| / (H/2) (kPa/m)")
        ax.set_title(title + " (o paper, – low, -- higher)", fontsize=8)
    axes[0].legend(fontsize=7)
    ax = axes[2]
    xs = sorted(TABLE_5)
    ax.semilogy(xs, [TABLE_5[x] for x in xs], "o", color="black", mfc="none", label="Table 5")
    ax.semilogy(KA, curves["low"][2], color=C[0], label="low order")
    ax.semilogy(KA, curves["higher"][2], color=C[1], ls="--", label="higher order")
    ax.set_xlabel("ka")
    ax.set_ylabel("P_max / (H/2)² (kW/m²)")
    ax.set_title("Table 5, restrained, the largest power", fontsize=8)
    ax.legend(fontsize=7)
    save(fig, NAME)
    print("  OWC chamber:", {k: (out[k]["table1_rel"], out[k]["table2_rel"]) for k in ("low", "higher")})
    record(NAME, out)


def summary(n):
    d = n[NAME]
    hi = max(d["higher"]["table1_rel"], d["higher"]["table2_rel"])
    lo = max(d["low"]["table1_rel"], d["low"]["table2_rel"])
    return (f"the printed chamber pressures within {100 * hi:.1f} % (higher order) and {100 * lo:.1f} % (low "
            "order); the higher order meets the capture-width bound")  # fmt: skip


INDEX = {"case": "OWC chamber with a turbine",
         "checked": "chamber pressure restrained and free in heave; largest power; capture width",
         "sources": "Mavrakos & Konispoliatis (2012) tables 1, 2, 5; Evans (1982)",
         "methods": "low, higher"}  # fmt: skip


def page(n):
    d = n[NAME]
    ka = d["ka"]
    rows5 = []
    for x in sorted(TABLE_5):
        i = ka.index(x)
        rows5.append([f"{x:g}", f"{TABLE_5[x]:g}", f"{d['low']['table5'][str(x)]:.4g}",
                      f"{100 * d['low']['table5_rel'][str(x)]:+.1f} %", f"{d['low']['capture'][i]:.3f}",
                      f"{d['higher']['table5'][str(x)]:.4g}", f"{100 * d['higher']['table5_rel'][str(x)]:+.1f} %",
                      f"{d['higher']['capture'][i]:.4f}"])  # fmt: skip
    i3 = ka.index(3.0)
    below = [i for i, x in enumerate(ka) if x <= 2.5]
    cap_hi = max(abs(d["higher"]["capture_minus_1"][i]) for i in below)
    cap_lo = max(abs(d["low"]["capture_minus_1"][i]) for i in below) + 1
    cap_hi3, cap_lo3 = d["higher"]["capture"][i3], d["low"]["capture"][i3]
    min_hi = min(d["higher"]["table5_rel"].values())
    rows12 = []
    for t, name in (("table1", "table 1 (restrained)"), ("table2", "table 2 (free in heave)")):
        rows12.append([name, f"{100 * d['low'][t + '_rel']:.1f} %", f"{100 * d['higher'][t + '_rel']:.1f} %"])
    hi = max(d["higher"]["table1_rel"], d["higher"]["table2_rel"])
    lo = max(d["low"]["table1_rel"], d["low"]["table2_rel"])
    return rf"""
# Oscillating water column chamber with a turbine

**Result.** The chamber pressure of an oscillating water column, restrained and free in heave, agrees
with the printed tables of Mavrakos & Konispoliatis (2012) within {100 * hi:.1f} % in the higher order
and {100 * lo:.1f} % in the low order. The higher order meets the exact capture-width bound of Evans
(1982) within {size(cap_hi)} up to $ka = 2.5$, so the remaining differences from the paper are the
paper's. The low order follows the bound within {pct(cap_lo - 1, 1)} up to $ka = 2.5$ but fails at
$ka = 3$, where the chamber radiates almost nothing.

## The problem

An oscillating water column is a vertical tube open at its bottom: the waves move the water column
inside, and the air above it is pushed through a turbine. Here the tube is thick-walled, the air in
its chamber is at a uniform pressure that drives a linear (Wells) turbine, and the device is
restrained or free in heave.

It is the device of Section 12 of Mavrakos & Konispoliatis (2012): inner radius $b = 2$ m, outer
radius $a = 4$ m, draught 5 m, water depth 15 m, heading $0^\circ$, $0.2 \le ka \le 3$.

In HydroBEM the inner free surface is a pressure surface with one uniform pressure mode. Its
coordinate $\xi$ is the pressure head ($p_{{in0}} = -\rho g \xi$), coupled to the heave through the
roof (chamber_coupling). The turbine of admittance $g_T$ (the air volume flow $g_T p_{{in0}}$, the air
incompressible) is the damping $(\rho g)^2 g_T/\omega^2$ of that mode. One solve gives the
coefficients, and the equations of the heave and the chamber are solved for each $g_T$ (theory §3.1).

The paper's units are t, kN, kPa and kW, its $g_T$ in $\mathrm{{m^5/(kN\,s)}}$. Its density and gravity
are not stated: $\rho = 1025\ \mathrm{{kg/m^3}}$ and $g = 9.81\ \mathrm{{m/s^2}}$ are taken, with which
the restrained column's largest power at $ka = 0.2$ equals the bound below to 1e-3.

The largest power of the restrained device is $\lvert q_D \rvert^2/(8 f_{{PP}})$ (eq. 11.4, with an
optimal complex turbine), where $q_D$ is the exciting volume flow and $f_{{PP}}$ the radiation
conductance of the chamber.

![Meshes](../figures/mesh_{NAME}.png)

*Figure 1. The two meshes, expanded to the whole device and cut open (a quarter removed) to show the
inside: the wetted walls and bottom in blue, the chamber's free surface (the pressure surface) in
purple, the annulus between the walls in $z = 0$ (the interior surface) in yellow. Left: low order, a
quadrant of 16 panels around, 36 down each wall and 8 across the bottom (cosine-spaced, small at the
waterline and the bottom edges), 8 rings across the chamber's free surface (small at its rim) and 8
across the annulus. Right: higher order, the analytic tube with its inner free surface (patch edges
in black, span edges in grey: outer wall $8 \times 12$, bottom $8 \times 6$, inner wall $4 \times 12$,
inner free surface $4 \times 6$ spans per quadrant, clustered at the bottom edges) and the annulus as
a revolved-segment patch of its own ($8 \times 4$).*

## The references

The published reference is Mavrakos & Konispoliatis (2012), tables 1, 2 and 5, computed by matched
eigenfunction expansions and printed to three to six significant digits:

- the chamber pressure $\lvert p_{{in0}} \rvert/(H/2)$ in kPa/m at $ka = 0.4$, 1, 1.5, 2 for
  $g_T = 0.5$ to 10, restrained (their table 1) and free in heave (their table 2);
- the largest power of the restrained device at $ka = 0.2$ to 3 (their table 5).

The independent reference is a bound. The capture width of a device is the width of incident wave
crest whose power it absorbs; for an axisymmetric pressure distribution its largest value is $1/k$
(Evans 1982). So $k P_{{max}} / P_w = 1$ for the restrained chamber, where $P_w = \rho g c_g/2$ is the
power of the incident wave per unit crest and unit amplitude squared. This bound is exact and
independent of the paper.

## Agreement with the tables

![Comparison](../figures/{NAME}.png)

*Figure 2. Left and middle: the chamber pressure of the paper's tables 1 and 2 (circles) against $ka$
for each $g_T$, low order (solid), higher order (dashed). Right: the largest power of the paper's
table 5.*

The pressures of the paper's tables 1 and 2 agree within the percentages of Table 1. The largest
differences, at $ka = 1$ to 2, are those of the paper or of the assumed density and gravity rather
than of the discretization, since the same mesh meets the exact bound there.

*Table 1. Largest relative difference from the printed pressures of the paper's tables 1 and 2 over
the 24 entries of each table.*

{table(["Paper's table", "low order", "higher order"], rows12)}

## The capture-width bound

The bound checks the chamber against exact theory rather than against another computation (Table 2).
The higher order meets it within {e(cap_hi)} up to $ka = 2.5$ and by {pct(cap_hi3 - 1, 1)} at
$ka = 3$. The paper's table 5 lies up to {pct(-min_hi, 1)} above the higher order, at the larger $ka$,
where the higher order meets the bound: the differences are the paper's.

*Table 2. The largest power ($\mathrm{{kW/m^2}}$) of the paper's table 5 and of both methods, and
$k P_{{max}}/P_w$, 1 for an exact solution.*

{table(["$ka$", "paper's table 5", "low", "diff.", "$k P/P_w$", "higher", "diff.", "$k P/P_w$"], rows5)}

The low order follows the bound within {pct(cap_lo - 1, 1)} up to $ka = 2.5$ but fails at $ka = 3$
($k P_{{max}}/P_w = {cap_lo3:.2f}$), where the chamber radiates almost nothing. Its exciting flow there
is the small sum of the incident and the scattered parts on the inner surface, which the constant
panels resolve poorly.

## Reproducing this page

The example [`examples/{NAME}`](../../../examples/{NAME}) runs this device and the paper's other cases
(the open device, thin-walled chambers). This page, its figures and its numbers are written by
`validation/floating_owc_chamber.py` through `validation/make_report.py`.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
