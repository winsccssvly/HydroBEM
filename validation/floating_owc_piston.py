"""A floating oscillating water column as two bodies (the two-body method of Raghavan et al. 2024,
case 4): the hull, a tube, and a rigid piston filling its water column, the piston's side wall on
the hull's inner wall. The piston's heave added mass and damping on the coarse polygonal panels of
the paper, on fine panels and in the higher order on the exact geometry
(docs/validation/cases/floating_owc_piston.md). No independent reference: the comparison is of the
methods with each other.

    python validation/floating_owc_piston.py [--quick]
"""

from __future__ import annotations

import math
import sys

import mesh_figure
import numpy as np
from common import RHO, C, G, Timer, e, pc, plt, record, save, solve_model, table

from bodies import surface
from problems import analytic, model

NAME = "floating_owc_piston"
R_IN, R_OUT, DRAFT = 2.0, 3.0, 10.0  # m


def ring(r, phi, z):
    return [r * math.cos(phi), r * math.sin(phi), z]


def wall(radius, n_theta, zs, outward, quarter):
    phi = np.linspace(0.0, 0.5 * math.pi if quarter else 2 * math.pi, n_theta + 1)
    q = [[ring(radius, phi[j], zs[i]), ring(radius, phi[j], zs[i + 1]), ring(radius, phi[j + 1], zs[i + 1]),
          ring(radius, phi[j + 1], zs[i])] for i in range(len(zs) - 1) for j in range(n_theta)]  # fmt: skip
    return q if outward else [p[::-1] for p in q]


def annulus(rs, z, n_theta, quarter):
    """A horizontal ring at z through the radii rs (decreasing), normals down."""
    phi = np.linspace(0.0, 0.5 * math.pi if quarter else 2 * math.pi, n_theta + 1)
    return [[ring(rs[i], phi[j], z), ring(rs[i + 1], phi[j], z), ring(rs[i + 1], phi[j + 1], z),
             ring(rs[i], phi[j + 1], z)] for i in range(len(rs) - 1) for j in range(n_theta)]  # fmt: skip


def panels(fine):
    """The hull and the piston: the paper's division (16 around the whole circle, the outer wall
    20 down, the bottom annulus 2 across, the inner wall 18 down, the piston's bottom 6 across; 640
    and 384 panels) or a fine one a quadrant (16 around the quadrant, 40, 4, 36 and 12, cosine-spaced
    in depth)."""
    if fine:
        n, (no, na, ni, nb), q = 16, (40, 4, 36, 12), True
        zo = -DRAFT * 0.5 * (1 - np.cos(np.pi * np.arange(no + 1) / no))
        zi = -DRAFT * 0.5 * (1 - np.cos(np.pi * np.arange(ni + 1) / ni))
    else:
        n, (no, na, ni, nb), q = 16, (20, 2, 18, 6), False
        zo, zi = np.linspace(0.0, -DRAFT, no + 1), np.linspace(0.0, -DRAFT, ni + 1)
    hull = (wall(R_OUT, n, zo, True, q) + annulus(np.linspace(R_OUT, R_IN, na + 1), -DRAFT, n, q)
            + wall(R_IN, n, zi, False, q))  # fmt: skip
    piston = wall(R_IN, n, zi, True, q) + annulus(np.linspace(R_IN, 0.0, nb + 1), -DRAFT, n, q)
    sym = "xz_and_yz" if q else "none"
    return surface(np.array(hull), symmetry=sym), surface(np.array(piston), symmetry=sym)


def higher_meshes():
    """The analytic tube and cylinder (exact circles), the spans of the example: the hull's outer
    wall 3 x 6, bottom 3 x 2, inner wall 2 x 6, the piston's side 2 x 6 and bottom 2 x 3 a quadrant."""
    rows = lambda spans: [[a, b, 3, 3, 3, 4] for a, b in spans]  # noqa: E731
    hull = analytic("moonpool_cylinder", rows([(3, 6), (3, 2), (2, 6)]), parts=["side", "bottom", "moonpool_wall"],
                    mapping="nonuniform", radius=R_OUT, draft=DRAFT, moonpool_radius=R_IN, symmetry="xz_and_yz")
    piston = analytic("circular_cylinder", rows([(2, 6), (2, 3)]), parts=["side", "bottom"], mapping="nonuniform",
                      radius=R_IN, draft=DRAFT, symmetry="xz_and_yz")  # fmt: skip
    return hull, piston


def build(w, kind):
    hull, piston = higher_meshes() if kind == "higher" else panels(kind == "fine")
    bodies = [{"name": name, "mesh": mesh, "modes": ["surge", "heave", "pitch"], "interior_surface": "none",
               "rotation_center": [0.0, 0.0, 0.0]} for name, mesh in (("hull", hull), ("piston", piston))]  # fmt: skip
    return model(bodies, w, [], environment={"depth": math.inf, "density": RHO, "gravity": G},
                 options={"higher_order": kind == "higher", "diffraction": False})  # fmt: skip


def main(quick=False):
    w = np.round(np.arange(0.075, 3.01, 0.15 if quick else 0.075), 6)
    res = {}
    for kind in ("paper", "fine", "higher"):
        with Timer(f"OWC piston: {kind}"):
            r = solve_model(build(w, kind))
        res[kind] = np.array([r.coefficient(q, ("piston", "heave"), ("piston", "heave"))
                              for q in ("added_mass", "radiation_damping")])  # fmt: skip
    mesh_figure.draw(NAME, [("low order, the paper's 1024 panels", build([1.0], "paper"), False, {"cut": (1, -1)}),
                            ("low order, fine, 2048 panels a quadrant", build([1.0], "fine"), False,
                             {"cut": (1, -1)}),
                            ("higher order, analytic patches", build([1.0], "higher"), True, {"cut": (1, -1)})],
                     elev=30.0)  # fmt: skip
    column = RHO * math.pi * R_IN**2 * DRAFT
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, i, lab in ((axes[0], 0, "total mass ρπr₁²H + A₃₃ (t)"), (axes[1], 1, "damping B₃₃ (t/s)")):
        for j, (kind, name) in enumerate((("paper", "low order, paper's panels"), ("fine", "low order, fine"),
                                          ("higher", "higher order"))):  # fmt: skip
            ax.plot(w, (res[kind][i] + (column if i == 0 else 0.0)) / 1e3, color=C[j], ls=("-", "-.", "--")[j],
                    label=name)  # fmt: skip
        ax.set_xlabel("ω (rad/s)")
        ax.set_ylabel(lab)
    axes[0].legend()
    save(fig, NAME)
    ho = res["higher"]
    scale = np.abs(ho).max(axis=1)
    out = {"omega": w, "column_mass": column}
    for kind in ("paper", "fine"):
        out[kind] = (np.abs(res[kind] - ho).max(axis=1) / scale).tolist()
    ib = int(np.argmax(ho[1]))
    out["peak"] = {"omega": float(w[ib]), **{k: float(res[k][1][ib]) for k in res}}
    out["total_mass_paper_vs_higher"] = float(np.abs((res["paper"][0] - ho[0]) / (column + ho[0])).max())
    print("  OWC piston:", out)
    record(NAME, out)


def summary(n):
    d = n[NAME]
    return (f"no independent reference; the fine low order converges to the higher order, within "
            f"{pc(max(d['fine']))}")


INDEX = {"case": "Floating OWC, two bodies", "checked": "heave $A$, $B$ of a piston filling the column of a tube",
         "sources": "Raghavan et al. (2024) case 4 (geometry only); methods against each other",
         "methods": "low, higher"}  # fmt: skip


def page(n):
    d = n[NAME]
    p = d["peak"]
    rows = [["low order, the paper's panels (16 around)", e(d["paper"][0]), e(d["paper"][1])],
            ["low order, fine (64 around)", e(d["fine"][0]), e(d["fine"][1])]]  # fmt: skip
    return rf"""
# Floating oscillating water column: hull and piston

**Result.** No independent solution of this two-body model is known, so the case checks consistency,
not accuracy. Fine low-order panels converge to the higher order on the exact geometry, within
{pc(max(d['fine']))} of its largest value. The paper's coarse polygons stay up to
{pc(max(d['paper']))} away, mostly by their smaller area; the total mass, dominated by the column,
differs by at most {100 * d["total_mass_paper_vs_higher"]:.1f} %.

## The problem

A floating oscillating water column is modelled here as two bodies (the two-body method): the hull of
the device, a vertical tube, and an imaginary rigid piston that fills its water column, the piston's
side wall lying on the hull's inner wall. The case is that of Raghavan et al. (2024, §3.4, case 4).

The hull has inner radius $r_1 = 2$ m, outer radius $r_2 = 3$ m and draft $H = 10$ m (outer wall,
bottom annulus, inner wall). The piston has radius $r_1$, length $H$ and its top in $z = 0$ (side wall
and bottom disc). The water is infinitely deep, $\rho = 1000\ \mathrm{{kg/m^3}}$, and
$0.075 \le \omega \le 3$ rad/s. We compare the piston's heave added mass $A_{{33}}$ and damping
$B_{{33}}$; the paper's figures 17 and 18 show the total mass $\rho \pi r_1^2 H + A_{{33}}$ and
$B_{{33}}$.

There is no interior surface, as in the paper. The first irregular frequency of the piston's interior,
where the integral equation fails because the water that would fill the piston resonates, is at
$k r_1 = 2.405$, near 3.4 rad/s, above the range.

![Meshes](../figures/mesh_{NAME}.png)

*Figure 1. The three meshes, expanded to the whole device and cut open (a quarter removed) to show the
piston inside the tube: all surfaces are wetted (blue). Left: the paper's division, 16 panels around
the circle, the outer wall 20 down, the bottom annulus 2 across, the inner wall 18 down (640 panels),
the piston's side on the same 18 rows and its bottom 6 across (384). Middle: a quadrant of 16 around,
40, 4 and 36 down (cosine-spaced in depth) and 12 across the piston's bottom. Right: the analytic tube
and cylinder (patch edges in black, span edges in grey), the spans per quadrant $3 \times 6$ (outer
wall), $3 \times 2$ (bottom), $2 \times 6$ (inner wall), $2 \times 6$ and $2 \times 3$ (the piston),
order 3, clustered at the bottom edges.*

## The reference

No independent solution of the two-body model is known to us, and the paper's curves are not read:
they are those of its coarse meshes, 16-sided polygons whose area is 2.5 % smaller than the circle's.
The comparison is therefore of three discretizations of the same problem: the paper's coarse
polygons, fine panels and the higher order on the exact circles, which serves as the reference. The
physics of the piston model (a rigid lid on the column instead of a free surface) is the paper's
choice, not checked here.

## Agreement between the methods

![Comparison](../figures/{NAME}.png)

*Figure 2. The piston's total mass and damping in heave against $\omega$.*

The fine panels converge to the higher order (Table 1). The paper's polygons stay a few per cent
away, mostly by the smaller area of the polygons. At the peak of $B_{{33}}$, at
$\omega = {p["omega"]:.3g}$ rad/s, the damping is {p["paper"] / 1e3:.3g} t/s on the paper's panels,
{p["fine"] / 1e3:.3g} t/s on the fine ones and {p["higher"] / 1e3:.3g} t/s in the higher order. The
total mass, dominated by the column, differs by at most {100 * d["total_mass_paper_vs_higher"]:.1f} %.

*Table 1. Largest difference from the higher order over $\omega$, relative to its largest value.*

{table(["Mesh", "$A_{33}$", "$B_{33}$"], rows)}

The coincident walls of the two bodies need nothing special in either method. This case shows
consistency, not accuracy against an independent solution ([not verified](../not_verified.md)).

## Reproducing this page

The example [`examples/{NAME}`](../../../examples/{NAME}) runs the case of Raghavan et al. (2024,
§3.4) at three of its frequencies and at the infinite frequency. This page, its figures and its
numbers are written by `validation/floating_owc_piston.py` through `validation/make_report.py`.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
