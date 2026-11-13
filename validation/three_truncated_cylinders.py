"""Three floating truncated vertical cylinders at the corners of an equilateral triangle in finite
depth (the group of Mavrakos 1991: depth 40b, draught 3.65b, side 5b), head seas: the surge and
pitch-surge added mass and damping of each cylinder and their interaction, and the surge exciting
forces, against the semi-analytic interaction solution of scripts/gen_cylinder_array_reference.py
(evaluated here over the whole range) and the values read from Mavrakos' figures 5b and 5c
(docs/validation/cases/three_truncated_cylinders.md).

    python validation/three_truncated_cylinders.py [--quick]
"""

from __future__ import annotations

import math
import sys

import mesh_figure
import numpy as np
from common import (
    READ,
    REF,
    RHO,
    C,
    G,
    Timer,
    at_read,
    e,
    omega_of,
    pc,
    pct,
    plt,
    read_differences,
    record,
    rel_cols,  # fmt: skip
    save,
    size,
    solve_model,
    table,
)

from bodies import surface
from meshes import cylinder_panels
from problems import analytic, load_script, model

NAME = "three_truncated_cylinders"
DRAFT, DEPTH = 3.65, 40.0  # radius b = 1
S3 = math.sqrt(3.0)
CENTERS = ((5.0 / S3, 0.0), (-2.5 / S3, 2.5), (-2.5 / S3, -2.5))
# Mavrakos (1991), figs. 5b and 5c read by eye from the page images: {quantity: {kb: value}}, a or b
# of the modes ij and the cylinders qp divided by pi (a11_12: the surge force on cylinder 1 from the
# surge of cylinder 2); the scale of fig. 5c taken linear from its ticks 2 and -2
FIG_5 = {
    "a11_11": {0.25: 3.44, 0.4: 3.76, 0.5: 3.46, 0.6: 2.71, 0.75: 2.46, 1.0: 2.58, 1.25: 1.80, 1.45: 1.55},
    "a11_22": {0.4: 3.60, 0.6: 3.02, 0.75: 2.58, 1.0: 2.37, 1.45: 1.63},
    "a11_12": {0.25: -0.31, 0.4: -0.65, 0.6: -0.10, 0.75: 0.32, 1.0: 0.75, 1.25: -0.02, 1.45: -0.35},
    "b11_11": {0.25: 0.26, 0.5: 1.52, 0.6: 1.59, 0.75: 1.15, 0.9: 1.06, 1.1: 1.50, 1.25: 1.38, 1.45: 0.88},
    "b11_22": {0.5: 1.26, 0.9: 1.24, 1.1: 1.41},
    "b11_12": {0.5: -0.66, 0.6: -0.84, 0.9: -0.31, 1.1: 0.48, 1.25: 0.57},
    "a51_11": {0.25: -5.40, 0.4: -5.69, 0.6: -4.25, 0.75: -3.96, 1.0: -4.09, 1.25: -3.52, 1.45: -3.48},
    "a51_12": {0.25: 0.43, 0.4: 0.87, 0.6: 0.15, 0.75: -0.33, 1.0: -0.64, 1.25: 0.03, 1.45: 0.28},
    "b51_11": {0.4: -1.08, 0.5: -1.89, 0.6: -1.87, 0.75: -1.21, 0.9: -1.00, 1.1: -1.25, 1.45: -0.58},
    "b51_12": {0.4: 0.22, 0.5: 0.78, 0.6: 0.96, 0.9: 0.29, 1.0: -0.10, 1.1: -0.40, 1.25: -0.42, 1.45: -0.15},
}
READ_ACCURACY = 0.05  # in the units of the figures
# the quantities of the semi-analytic solution (cylinder 1 moving in surge) in the figures' units
ANALYTIC = ["a11_11", "b11_11", "a51_11", "b51_11", "a11_12", "b11_12", "X1", "X2"]
PLOTTED = ["a11_11", "b11_11", "a51_11", "b51_11", "a11_12", "b11_12", "a51_12", "b51_12", "a11_22", "b11_22",
           "X1", "X2"]  # fmt: skip


def bodies(higher):
    """The three cylinders: low order 16 panels a quarter circle, 24 down the side and 8 across the
    bottom (cosine-spaced), the interior surface graded; higher order the analytic cylinder, 4 x 8,
    4 x 3 and 4 x 3 spans a quadrant on the side, bottom and waterplane (the interior surface)."""
    if higher:
        mesh = analytic("circular_cylinder", [(4, 8, 3, 3, 3, 4), (4, 3, 3, 3, 3, 4), (4, 3, 3, 3, 3, 4)],
                        parts=["side", "bottom", "waterplane"], mapping="nonuniform", radius=1.0, draft=DRAFT,
                        symmetry="xz_and_yz")  # fmt: skip
        interior = "detect"
    else:
        mesh = surface(cylinder_panels(1.0, DRAFT, n_phi=16, n_z=24, n_r=8, quarter=True, cosine=True),
                       symmetry="xz_and_yz")  # fmt: skip
        interior = "graded"
    return [{"name": f"c{i + 1}", "mesh": mesh, "position": [x, y, 0.0], "rotation_center": [0.0, 0.0, 0.0],
             "modes": ["surge", "pitch"], "interior_surface": interior}
            for i, (x, y) in enumerate(CENTERS)]  # fmt: skip


def build(ka, higher):
    return model(bodies(higher), omega_of(ka, DEPTH), [0.0], environment={"depth": DEPTH, "density": RHO, "gravity": G},
                 options={"higher_order": higher})  # fmt: skip


def figure_quantities(r):
    """Mavrakos' a/(rho pi b^3), b/(rho omega pi b^3) (b^4 for pitch-surge) and |X|/(rho g A b^2)."""
    w = r.omegas
    rows = {"1": "surge", "5": "pitch"}
    q = {}
    for i in ("1", "5"):
        scale = RHO * math.pi
        for qp in ("11", "22", "12"):
            row, col = (f"c{qp[0]}", rows[i]), (f"c{qp[1]}", "surge")
            q[f"a{i}1_{qp}"] = r.coefficient("added_mass", row, col) / scale
            q[f"b{i}1_{qp}"] = r.coefficient("radiation_damping", row, col) / (w * scale)
    for p in ("1", "2"):
        q["X" + p] = np.abs(r.coefficient("excitation_diffraction", (f"c{p}", "surge"))[:, 0]) / (RHO * G)
    return q


def reference(ka):
    """The semi-analytic solution (cylinder 1 moving in surge; a11_12 is the surge force on 2 from
    1, equal to that on 1 from 2 by reciprocity): N = 200 and 400 extrapolated (order 4/3)."""
    gen = load_script("gen_cylinder_array_reference")
    f = 2 ** (4 / 3) - 1
    rows = []
    for k in ka:
        v200, v400 = (gen.quantities("three", float(k), n) for n in (200, 400))
        rows.append(v400 + (v400 - v200) / f)
    q = np.array(rows)
    q[:, :6] /= math.pi
    return dict(zip(ANALYTIC, q.T, strict=True))


def main(quick=False):
    ka = np.round(np.arange(0.1, 2.01, 0.2 if quick else 0.1), 6)
    ka = np.union1d(ka, sorted({x for pts in FIG_5.values() for x in pts}))
    with Timer("three cylinders: reference"):
        ref = reference(ka)
    res = {}
    for higher in (False, True):
        with Timer(f"three cylinders: {'higher' if higher else 'low'} order"):
            res[higher] = figure_quantities(solve_model(build(ka, higher)))
    mesh_figure.draw(NAME, [("low order, 512 panels a quadrant", build([1.0], False), False),
                            ("higher order, analytic patches", build([1.0], True), True)], elev=35.0)  # fmt: skip
    fig, axes = plt.subplots(3, 4, figsize=(13.0, 8.0), sharex=True)
    for ax, q in zip(axes.ravel(), PLOTTED, strict=True):
        if q in ref:
            ax.plot(ka, ref[q], **{**REF, "label": "semi-analytic"})
        if q in FIG_5:
            x, y, _ = at_read(ka, res[True][q], FIG_5[q])
            ax.plot(x, y, **{**READ, "label": "Mavrakos (1991), read"})
        ax.plot(ka, res[False][q], color=C[0], label="low order")
        ax.plot(ka, res[True][q], color=C[1], ls="--", label="higher order")
        ax.set_title(q if not q.startswith("X") else f"|X{q[1]}|/(ρgAb²)", fontsize=9)
    for ax in axes[2]:
        ax.set_xlabel("kb")
    axes[0, 0].legend(fontsize=7)
    save(fig, NAME)
    out = {"ka": ka}
    for higher, key in ((False, "low"), (True, "higher")):
        v = res[higher]
        out[key] = {"analytic": dict(zip(ANALYTIC, rel_cols(np.array([v[q] for q in ANALYTIC]).T,
                                                            np.array([ref[q] for q in ANALYTIC]).T), strict=True)),
                    "figure": read_differences(ka, v, FIG_5)}  # fmt: skip
    out["reference_figure"] = read_differences(ka, ref, {k: v for k, v in FIG_5.items() if k in ref})
    # the own added mass against the figure, signed, at the points read (the largest relative)
    for q in ("a11_11", "a51_11"):
        _, r, v = at_read(ka, ref[q], FIG_5[q])
        out[f"reference_{q}_signed"] = [float((v / r - 1).min()), float((v / r - 1).max())]
    print("  three cylinders:", {k: out[k] for k in ("low", "higher")})
    record(NAME, out)


def summary(n):
    d = n[NAME]
    return (f"semi-analytic solution within {pc(max(d['low']['analytic'].values()))} (low order) and "
            f"{size(max(d['higher']['analytic'].values()))} (higher order); the figures read agree to about "
            "their reading")  # fmt: skip


INDEX = {"case": "Three truncated cylinders",
         "checked": "own and interaction $A$, $B$ (surge, pitch-surge), surge $X$ of three cylinders, finite depth",
         "sources": "Mavrakos (1991) figs. 5b, 5c (read); semi-analytic interaction", "methods": "low, higher"}


def head(q):
    """The header of a quantity in the page's tables: a11_12 as a_11^12, X1 as X_1."""
    if q.startswith("X"):
        return f"$X_{q[1:]}$"
    return f"${q[0]}_{{{q[1:3]}}}^{{{q[4:6]}}}$"


def page(n):
    d = n[NAME]
    rows = [[label] + [e(d[key]["analytic"][q]) for q in ANALYTIC]
            for key, label in (("low", "low order"), ("higher", "higher order"))]  # fmt: skip
    keys = list(FIG_5)
    rows_fig = [[label] + [pct(d[key]["figure"][q], 0) for q in keys]
                for key, label in (("low", "low order"), ("higher", "higher order"))]  # fmt: skip
    rows_fig.append(["semi-analytic"] + [pct(d["reference_figure"][q], 0) if q in d["reference_figure"] else "—"
                                         for q in keys])  # fmt: skip
    a11, a51 = d["reference_a11_11_signed"], d["reference_a51_11_signed"]
    lo, hi = max(d["low"]["analytic"].values()), max(d["higher"]["analytic"].values())
    return rf"""
# Three truncated cylinders in finite depth

**Result.** For three interacting cylinders in finite depth, both methods agree with a semi-analytic
interaction solution over the whole range: the low order within {pc(lo)} and the higher order within
{size(hi)} of the largest value of each quantity. The damping and the interaction coefficients agree
with the published figures of Mavrakos (1991) to the accuracy of reading them. The cylinders' own added
mass lies {pct(min(a11[0], a51[0]), 0)} to {pct(max(a11[1], a51[1]), 0)} above the published curves,
in the converged solution as much as in HydroBEM.

## The problem

Three floating truncated vertical circular cylinders stand at the corners of an equilateral triangle
in head seas. Their radius is $b = 1$ m, their draught $3.65b$ and the water depth $40b$. The axes are
at the corners of a triangle of side $5b$: cylinder 1 at $(5b/\sqrt{{3}}, 0)$, cylinders 2 and 3 at
$(-5b/(2\sqrt{{3}}), \pm 5b/2)$. The heading is $0^\circ$ and the range $0.1 \le kb \le 2.0$.

We compare the added mass and damping of each cylinder and of their interaction, and the exciting
forces. The coefficients are those of Mavrakos (1991), about each cylinder's own point on its axis in
the free surface: $a_{{11}}^{{qp}}$ is the surge force on cylinder $q$ from the surge of cylinder $p$,
and $a_{{51}}^{{qp}}$ the pitch moment on $q$ from the surge of $p$. They are divided by $\rho \pi b^3$
($\rho \pi b^4$ for pitch-surge), the damping also by $\omega$. $X$ is the surge exciting force on a
cylinder over $\rho g A b^2$.

![Meshes](../figures/mesh_{NAME}.png)

*Figure 1. The two meshes, expanded to the three cylinders and seen from above the free surface: the
wetted surface in blue, the interior surface in the waterplane in yellow. Left: low order, one
quadrant of 16 panels per quarter circle, 24 down the side and 8 across the bottom (cosine-spaced),
expanded at each cylinder, with the graded interior surface. Right: higher order, the analytic
cylinder (patch edges in black, span edges in grey), $4 \times 8$ spans on the side and $4 \times 3$
on the bottom and on the waterplane (the interior surface) per quadrant, order 3, the nonuniform
mapping.*

The interior surface, a lid over the waterplane inside each column, removes the irregular
frequencies, at which the integral equation of the exterior problem fails because the water that
would fill the column resonates. Here the first lies near $kb = 2.4$, just above the range.

## The references

The main reference is the semi-analytic interaction solution of
`scripts/gen_cylinder_array_reference.py` (the case "three"), derived and implemented in this
repository. It matches eigenfunctions at each cylinder and scatters the waves exactly between them
(Kagemoto & Yue 1986). With cylinder 1 moving in surge it gives $a_{{11}}^{{11}}$, $b_{{11}}^{{11}}$,
$a_{{51}}^{{11}}$, $b_{{51}}^{{11}}$, the interaction $a_{{11}}^{{12}} = a_{{11}}^{{21}}$ (reciprocity),
$b_{{11}}^{{12}}$ and the exciting forces on cylinders 1 and 2. It is evaluated here at every $kb$
with 200 and 400 vertical modes, extrapolated with the order 4/3; the two extrapolations of
`tests/data/cylinder_array.txt` agree to $4 \times 10^{{-5}}$.

The published reference is Mavrakos (1991), figures 5b and 5c: the solid curves (the own coefficients
of cylinders I and II) and the dash-dot curves (the interaction). They are **read by eye** from the
page images at $kb = 0.25$ to 1.45, to about $\pm {READ_ACCURACY}$ in the units of the figures. That is
1 to 3 % of the own coefficients, more where a curve is steep; the scale of fig. 5c is taken linear
from its ticks 2 and $-2$. These figures are the only reference of $a_{{11}}^{{22}}$, $b_{{11}}^{{22}}$,
$a_{{51}}^{{12}}$ and $b_{{51}}^{{12}}$.

## Agreement with the references

![Comparison](../figures/{NAME}.png)

*Figure 2. The coefficients against $kb$: the semi-analytic solution (circles), Mavrakos' figures
(crosses, read), HydroBEM low order (solid) and higher order (dashed).*

Both methods agree with the semi-analytic interaction solution to their discretization over the range
(Table 1).

*Table 1. Largest difference from the semi-analytic solution over $0.1 \le kb \le 2$, relative to its
largest value of each quantity.*

{table(["Method"] + [head(q) for q in ANALYTIC], rows)}

The damping and the interaction coefficients agree with the curves of Mavrakos (1991) to the reading
accuracy (Table 2).

*Table 2. Largest difference from Mavrakos' figures 5b and 5c at the points read, relative to the
largest value read of each quantity.*

{table(["Against figs. 5b, 5c (read)"] + [head(q) for q in keys], rows_fig)}

The cylinders' own added mass of the semi-analytic solution lies above the curves read:
$a_{{11}}^{{11}}$ by {pct(a11[0], 0)} to {pct(a11[1], 0)} and $a_{{51}}^{{11}}$ by {pct(a51[0], 0)} to
{pct(a51[1], 0)} in magnitude at the points read. Both HydroBEM methods lie above them by the same
amount. The offset is therefore between the published curves and the converged solution of the stated
geometry, not a discretization error; its cause in the publication is not known.

## Reproducing this page

The example [`examples/{NAME}`](../../../examples/{NAME}) runs the case at full scale ($b = 3$ m).
This page, its figures and its numbers are written by `validation/three_truncated_cylinders.py`
through `validation/make_report.py`.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
