"""Two floating truncated vertical cylinders side by side in finite depth, head seas (geometry 2 of
Kagemoto & Yue 1986: h/D = 2, H/D = 1, L/D = 1.3), both moving together in surge and in heave:
the total added mass, damping and exciting forces against the semi-analytic interaction solution of
scripts/gen_cylinder_array_reference.py (eigenfunction matching with the exact multiple scattering,
evaluated here over the whole range), against Kagemoto & Yue's table 2 and against the values read
from their figure 4 (docs/validation/cases/two_truncated_cylinders.md).

    python validation/two_truncated_cylinders.py [--quick]
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

NAME = "two_truncated_cylinders"
DRAFT, DEPTH, CENTERS = 2.0, 4.0, ((-1.3, 0.0), (1.3, 0.0))  # radius a = 1
SCALE = math.pi * DRAFT * 2  # Kagemoto & Yue (4.1): pi a H N, N = 2 legs, a = 1
QUANTITIES = ["M11", "N11", "F1", "M33", "N33", "F3"]
# the quantities in the headers of the page's tables
HEAD = {"M11": "$M_{11}$", "N11": "$N_{11}$", "F1": r"$\lvert F_1 \rvert$", "M33": "$M_{33}$", "N33": "$N_{33}$",
        "F3": r"$\lvert F_3 \rvert$"}  # fmt: skip
LABELS = {"M11": "surge added mass M₁₁", "N11": "surge damping N₁₁", "F1": "surge force |F₁|",
          "M33": "heave added mass M₃₃", "N33": "heave damping N₃₃", "F3": "heave force |F₃|"}  # fmt: skip
# Kagemoto & Yue (1986), figure 4, the curve L read by eye from the page images (kD: value)
FIG_4 = {
    "F1": {0.5: 0.319, 1.0: 0.368, 1.5: 0.264, 2.0: 0.150, 2.5: 0.112, 3.0: 0.180, 3.5: 0.207, 4.0: 0.207},
    "N11": {0.5: 0.065, 1.0: 0.224, 1.5: 0.249, 2.0: 0.199, 2.5: 0.178, 3.0: 0.181, 3.5: 0.178, 4.0: 0.164},
    "M11": {0.5: 0.625, 1.0: 0.595, 1.5: 0.465, 2.0: 0.420, 2.5: 0.415, 3.0: 0.407, 3.5: 0.389, 4.0: 0.368},
    "F3": {0.5: 0.308, 1.0: 0.130, 1.5: 0.044, 2.0: 0.008},
    "M33": {0.5: 0.350, 1.0: 0.309, 1.5: 0.317, 2.0: 0.332, 2.5: 0.344, 3.0: 0.348, 3.5: 0.354, 4.0: 0.354},
}
READ_ACCURACY = 0.005  # the reading accuracy of figure 4, in its units
# Kagemoto & Yue (1986), table 2, the full diffraction computation at kD = 3.1, as printed
TABLE_2 = {"F1": 0.18828, "N11": 0.18146, "M11": 0.39797, "F3": 0.00779, "M33": 0.34852}


def bodies(higher):
    """The two cylinders: low order 16 panels a quarter circle, 20 down the side and 8 across the
    bottom (cosine-spaced), the interior surface graded; higher order the analytic cylinder (side,
    bottom and its waterplane as the interior surface), 4 x 6, 4 x 3 and 4 x 3 spans a quadrant,
    the nonuniform mapping."""
    if higher:
        mesh = analytic("circular_cylinder", [(4, 6, 3, 3, 3, 4), (4, 3, 3, 3, 3, 4), (4, 3, 3, 3, 3, 4)],
                        parts=["side", "bottom", "waterplane"], mapping="nonuniform", radius=1.0, draft=DRAFT,
                        symmetry="xz_and_yz")  # fmt: skip
        interior = "detect"
    else:
        mesh = surface(cylinder_panels(1.0, DRAFT, n_phi=16, n_z=20, n_r=8, quarter=True, cosine=True),
                       symmetry="xz_and_yz")  # fmt: skip
        interior = "graded"
    return [{"name": f"c{i + 1}", "mesh": mesh, "position": [x, y, 0.0], "rotation_center": [0.0, 0.0, 0.0],
             "modes": ["surge", "heave"], "interior_surface": interior}
            for i, (x, y) in enumerate(CENTERS)]  # fmt: skip


def build(ka, higher):
    w = omega_of(ka, DEPTH)
    return model(bodies(higher), w, [0.0], environment={"depth": DEPTH, "density": RHO, "gravity": G},
                 options={"higher_order": higher})  # fmt: skip


def figure_quantities(r):
    """Kagemoto & Yue's normalization (4.1) of the totals of both cylinders."""
    names = ("c1", "c2")
    w = r.omegas
    out = {}
    for mode, i in (("surge", "1"), ("heave", "3")):
        a = sum(r.coefficient("added_mass", (p, mode), (q, mode)) for p in names for q in names)
        b = sum(r.coefficient("radiation_damping", (p, mode), (q, mode)) for p in names for q in names)
        x = sum(r.coefficient("excitation_diffraction", (p, mode))[:, 0] for p in names)
        out["M" + i + i] = a / (RHO * SCALE)
        out["N" + i + i] = b / (w * RHO * SCALE)
        out["F" + i] = np.abs(x) / (RHO * G * SCALE)
    return out


def reference(ka):
    """The semi-analytic solution in the same normalization: N = 200 and 400 vertical modes
    extrapolated with the order 4/3 of the corner at the bottom edge."""
    gen = load_script("gen_cylinder_array_reference")
    f = 2 ** (4 / 3) - 1
    rows = []
    for k in ka:
        v200, v400 = (gen.quantities("two", float(k), n) for n in (200, 400))
        rows.append(v400 + (v400 - v200) / f)
    q = np.array(rows) / SCALE  # surge A, B, heave A, B, surge |X|, heave |X|
    return dict(zip(["M11", "N11", "M33", "N33", "F1", "F3"], q.T, strict=True))


def main(quick=False):
    ka = np.round(np.arange(0.1, 2.01, 0.2 if quick else 0.1), 6)
    ka = np.union1d(ka, [0.25, 0.75, 1.25, 1.55, 1.75])
    kd = 2 * ka
    with Timer("two cylinders: reference"):
        ref = reference(ka)
    res = {}
    for higher in (False, True):
        with Timer(f"two cylinders: {'higher' if higher else 'low'} order"):
            res[higher] = figure_quantities(solve_model(build(ka, higher)))
    mesh_figure.draw(NAME, [("low order, 448 panels a quadrant", build([1.0], False), False),
                            ("higher order, analytic patches", build([1.0], True), True)])  # fmt: skip
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), sharex=True)
    for ax, q in zip(axes.ravel(), QUANTITIES, strict=True):
        ax.plot(kd, ref[q], **{**REF, "label": "semi-analytic"})
        if q in FIG_4:
            x, y, _ = at_read(kd, ref[q], FIG_4[q])
            ax.plot(x, y, **{**READ, "label": "Kagemoto & Yue fig. 4 (read)"})
        if q in TABLE_2:
            ax.plot([3.1], [TABLE_2[q]], "s", color="black", ms=5, label="Kagemoto & Yue table 2")
        ax.plot(kd, res[False][q], color=C[0], label="low order")
        ax.plot(kd, res[True][q], color=C[1], ls="--", label="higher order")
        ax.set_title(LABELS[q])
    for ax in axes[1]:
        ax.set_xlabel("kD")
    axes[0, 0].legend(fontsize=7)
    save(fig, NAME)
    out = {"ka": ka}
    i31 = int(np.argmin(np.abs(kd - 3.1)))
    for higher, key in ((False, "low"), (True, "higher")):
        v = res[higher]
        out[key] = {"analytic": dict(zip(QUANTITIES, rel_cols(np.c_[[v[q] for q in QUANTITIES]].T,
                                                              np.c_[[ref[q] for q in QUANTITIES]].T), strict=True)),
                    "figure": read_differences(kd, v, FIG_4),
                    "table2": {q: float(v[q][i31] / TABLE_2[q] - 1.0) for q in TABLE_2}}  # fmt: skip
    out["reference_figure"] = read_differences(kd, ref, FIG_4)
    out["reference_table2"] = {q: float(ref[q][i31] / TABLE_2[q] - 1.0) for q in TABLE_2}
    print("  two cylinders:", {k: out[k] for k in ("low", "higher")})
    record(NAME, out)


def summary(n):
    d = n[NAME]
    return (f"semi-analytic solution within {pc(max(d['low']['analytic'].values()))} (low order) and "
            f"{size(max(d['higher']['analytic'].values()))} (higher order); the figure read agrees to its "
            "reading")  # fmt: skip


INDEX = {"case": "Two truncated cylinders", "checked": "total $A$, $B$, $X$ of two interacting cylinders, finite depth",
         "sources": "Kagemoto & Yue (1986) fig. 4 (read), table 2; semi-analytic interaction",
         "methods": "low, higher"}  # fmt: skip


def page(n):
    d = n[NAME]
    rows = []
    for key, label in (("low", "low order, 448 panels a quadrant"), ("higher", r"higher order, $4 \times 6$ spans")):
        rows.append([label] + [e(d[key]["analytic"][q]) for q in QUANTITIES])
    rows_fig = []
    for key, label in (("low", "low order"), ("higher", "higher order"), (None, "semi-analytic")):
        src = d[key]["figure"] if key else d["reference_figure"]
        rows_fig.append([label] + [pct(src[q]) for q in FIG_4])
    rows_t2 = []
    for key, label in (("low", "low order"), ("higher", "higher order"), (None, "semi-analytic")):
        src = d[key]["table2"] if key else d["reference_table2"]
        rows_t2.append([label] + [f"{100 * src[q]:+.2f} %" for q in TABLE_2])
    lo, hi = max(d["low"]["analytic"].values()), max(d["higher"]["analytic"].values())
    fig_ref = max(d["reference_figure"].values())
    fig_bem = max(max(d["low"]["figure"].values()), max(d["higher"]["figure"].values()))
    t2_ref = max(abs(v) for q, v in d["reference_table2"].items() if q != "F3")
    t2_f3 = abs(d["reference_table2"]["F3"])
    return rf"""
# Two truncated cylinders in finite depth

**Result.** For two interacting cylinders in finite depth, both methods agree with a semi-analytic
interaction solution over the whole range: the low order within {pc(lo)} and the higher order within
{size(hi)} of the largest value of each quantity. The published figure of Kagemoto & Yue (1986) is
matched to within {pct(fig_bem, 0)} of its largest value read, about as closely as the semi-analytic
solution matches it, so what remains is the reading. Their printed table differs from both by up to
{pct(t2_ref, 1)}.

## The problem

Two floating truncated vertical circular cylinders stand side by side in head seas, both moving
together in surge and in heave. This is geometry 2 of Kagemoto & Yue (1986): radius $a = 1$ m
(diameter $D = 2a$), draft $H = 2a = D$, water depth $h = 4a = 2D$, centres at $x = \pm 1.3a$ on the
$x$ axis (spacing $L = 1.3D$), heading $0^\circ$, along the line of the cylinders. The range is
$0.1 \le ka \le 2.0$ ($kD = 0.2$ to 4), below the first irregular frequency of the columns.

The quantities are the totals of the two cylinders moving together: $A_{{11}} + A_{{17}} + A_{{71}} +
A_{{77}}$ for surge (each cylinder's six modes in turn) and the summed exciting force. They are given
in the normalization (4.1) of Kagemoto & Yue (1986): force $/ (\rho g \pi a H A N)$, added mass
$/ (\rho \pi a^2 H N)$ and damping $/ (\omega \rho \pi a^2 H N)$, with $N = 2$ legs and $A$ the wave
amplitude.

![Meshes](../figures/mesh_{NAME}.png)

*Figure 1. The two meshes, expanded to both cylinders and seen from above the free surface: the
wetted surface in blue, the interior surface in the waterplane in yellow. Left: low order, one
quadrant of 16 panels per quarter circle, 20 down the side and 8 across the bottom (cosine-spaced),
expanded over $x = 0$ and $y = 0$ at each cylinder, with the graded interior surface. Right: higher
order, the analytic cylinder (exact side, bottom and waterplane; patch edges in black, span edges in
grey), $4 \times 6$ spans on the side and $4 \times 3$ on the bottom and the waterplane per quadrant,
splines of order 3, the spans clustered at the waterline and at the bottom edge (mapping
"nonuniform"); its waterplane is the interior surface.*

## The references

The main reference is the semi-analytic interaction solution of
`scripts/gen_cylinder_array_reference.py`, derived and implemented in this repository. It matches
eigenfunctions at each cylinder (Yeung 1981, Garrett 1971) and scatters the propagating and
evanescent modes exactly between them (Kagemoto & Yue 1986). Here it is evaluated at every $ka$ of the
figure with 200 and 400 vertical modes, extrapolated with the order 4/3 of the corner at the bottom
edge. Its two extrapolations agree to $2 \times 10^{{-6}}$ of the largest value at the three
tabulated $ka$ of `tests/data/cylinder_array.txt`.

Two published sources are compared as well:

- **Kagemoto & Yue (1986), table 2**: their full diffraction computation at $kD = 3.1$, as printed.
- **Kagemoto & Yue (1986), figure 4**: the curve L (their interaction theory), **read by eye** from
  the page images at $kD = 0.5$ to 4, to about $\pm {READ_ACCURACY}$ in the units of the figure (1 to
  3 % of the values, more where a curve is steep).

## Agreement with the references

![Comparison](../figures/{NAME}.png)

*Figure 2. The totals against $kD$: the semi-analytic solution (circles), Kagemoto & Yue figure 4
(crosses, read) and table 2 (square), HydroBEM low order (solid) and higher order (dashed).*

Both methods agree with the semi-analytic interaction solution to their discretization over the whole
range (Table 1): the low order (3584 panels in all) within {e(lo)}, the higher order (864 unknowns)
within {e(hi)} of the largest value of each quantity.

*Table 1. Largest difference from the semi-analytic solution over $0.1 \le ka \le 2$, relative to its
largest value of each quantity.*

{table(["Method"] + [HEAD[q] for q in QUANTITIES], rows)}

The semi-analytic solution matches the figure read as closely as HydroBEM does, within
{pct(fig_ref, 0)} of the largest value read (Table 2). The remaining differences are therefore those of
the reading and of the published curve.

*Table 2. Largest difference from Kagemoto & Yue figure 4 at the points read, relative to the largest
value read of each quantity.*

{table(["Against figure 4 (read)"] + [HEAD[q] for q in FIG_4], rows_fig)}

Kagemoto & Yue's table 2 differs from the semi-analytic solution by up to {pct(t2_ref, 1)} in the
added mass and the damping, and by {pct(t2_f3, 0)} in the small heave force, printed as 0.00779
(Table 3). It differs from HydroBEM alike: the published full diffraction computation is itself a
discretization, whose accuracy the paper does not state.

*Table 3. Signed relative difference from the values printed in Kagemoto & Yue table 2 at
$kD = 3.1$ ($F_3$ is small there, 0.008).*

{table(["Against table 2 ($kD = 3.1$)"] + [HEAD[q] for q in TABLE_2], rows_t2)}

## Reproducing this page

The example [`examples/{NAME}`](../../../examples/{NAME}) runs the case of the paper at full scale.
This page, its figures and its numbers are written by `validation/two_truncated_cylinders.py` through
`validation/make_report.py`.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
