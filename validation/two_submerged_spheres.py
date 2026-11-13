"""Two submerged spheres side by side in deep water, waves at 45 degrees (case 2 of Wu 1995): on
sphere 2, with both spheres moving together in surge, sway and heave, the added mass, damping and
exciting force against the multipole solution of scripts/gen_sphere_array_reference.py (evaluated
here over the whole range) and against the values read from Wu's figures 2-10
(docs/validation/cases/two_submerged_spheres.md).

    python validation/two_submerged_spheres.py [--quick]
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
    pc,
    pct,
    plt,
    read_differences,
    record,
    rel_cols,
    save,
    size,
    solve_model,  # fmt: skip
    table,
)

from bodies import ho_sphere, surface
from problems import load_script, model

NAME = "two_submerged_spheres"
CENTERS = ((-2.0, 0.0, -1.5), (2.0, 0.0, -1.5))  # radius a = 1
HEADING = 45.0
NAMES = ["a11", "b11", "f1", "a22", "b22", "f2", "a33", "b33", "f3"]
# Wu (1995), figs. 2-10, the curves of case 2 read by eye from the page images: {quantity: {nu a: value}}
FIGS = {
    "a11": {0.25: 0.522, 0.5: 0.479, 1.0: 0.519, 1.5: 0.443, 2.0: 0.384, 2.5: 0.411},
    "a22": {0.25: 0.593, 1.0: 0.455, 1.5: 0.429, 2.0: 0.419, 2.5: 0.427},
    "a33": {0.5: 0.558, 1.0: 0.491, 1.5: 0.385, 2.5: 0.371},
    "b11": {0.5: 0.058, 0.75: 0.0175, 1.5: 0.117},
    "b22": {0.5: 0.117, 0.75: 0.133, 1.0: 0.109},
    "b33": {0.5: 0.1835, 0.75: 0.152, 1.0: 0.156, 1.5: 0.172},
    "f1": {0.5: 0.846, 0.75: 0.698, 1.0: 0.562, 1.5: 0.394, 2.0: 0.308, 2.5: 0.251},
    "f2": {1.0: 0.648, 1.5: 0.466, 2.0: 0.341, 2.5: 0.253},
    "f3": {0.5: 1.296, 1.5: 0.534, 2.0: 0.400, 2.5: 0.313},
}


def sphere_panels(n_bands, n_meridians):
    """The quadrant x, y >= 0 of the unit sphere about the origin, equal angles, normals out of the
    sphere (the panels at the poles have a repeated vertex)."""
    t = np.linspace(0.0, math.pi, n_bands + 1)
    f = np.linspace(0.0, 0.5 * math.pi, n_meridians + 1)

    def p(a, b):
        return [math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)]

    return np.array([[p(t[i], f[j]), p(t[i + 1], f[j]), p(t[i + 1], f[j + 1]), p(t[i], f[j + 1])]
                     for i in range(n_bands) for j in range(n_meridians)])  # fmt: skip


def build(nu, higher):
    mesh = ho_sphere(1.0, (4, 8)) if higher else surface(sphere_panels(24, 12), symmetry="xz_and_yz")
    bodies = [{"name": f"s{i + 1}", "mesh": mesh, "position": list(c), "rotation_center": [0.0, 0.0, 0.0],
               "modes": ["surge", "sway", "heave"], "interior_surface": "none"}
              for i, c in enumerate(CENTERS)]  # fmt: skip
    return model(bodies, np.sqrt(G * np.asarray(nu)), [HEADING],
                 environment={"depth": math.inf, "density": RHO, "gravity": G},
                 options={"higher_order": higher})  # fmt: skip


def figure_quantities(r, nu):
    """Sphere 2, both moving: A/(rho V), B/(rho w V) of A21 + A22, and |E2| / (rho w^2 V A exp(-nu a)),
    the normalization of Wu's figures."""
    w = r.omegas
    v = RHO * 4 * math.pi / 3
    q = {}
    for i, mode in (("1", "surge"), ("2", "sway"), ("3", "heave")):
        q["a" + i + i] = sum(r.coefficient("added_mass", ("s2", mode), (b, mode)) for b in ("s1", "s2")) / v
        b = sum(r.coefficient("radiation_damping", ("s2", mode), (s, mode)) for s in ("s1", "s2"))
        q["b" + i + i] = b / (v * w)
        q["f" + i] = np.abs(r.coefficient("excitation_diffraction", ("s2", mode))[:, 0]) / (v * w * w * np.exp(-nu))
    return q


def reference(nu):
    """The multipole solution (degree 10, 160 wave-integral points), its force put in the figures'
    normalization (exp(-nu a) instead of the script's exp(-nu S), S = 1.5a)."""
    gen = load_script("gen_sphere_array_reference")
    spheres = gen.SphereArray(CENTERS, 10, 160)
    rows = np.array([spheres.evaluate(float(x), math.radians(HEADING)) for x in nu])
    rows[:, 2::3] *= np.exp(-0.5 * np.asarray(nu))[:, None]
    return dict(zip(["a11", "b11", "f1", "a22", "b22", "f2", "a33", "b33", "f3"], rows.T, strict=True))


def main(quick=False):
    nu = np.round(np.arange(0.1, 3.01, 0.2 if quick else 0.1), 6)
    nu = np.union1d(nu, sorted({x for pts in FIGS.values() for x in pts}))
    with Timer("two spheres: reference"):
        ref = reference(nu)
    res = {}
    for higher in (False, True):
        with Timer(f"two spheres: {'higher' if higher else 'low'} order"):
            res[higher] = figure_quantities(solve_model(build(nu, higher)), nu)
    mesh_figure.draw(NAME, [("low order, 288 panels a quadrant", build([1.0], False), False),
                            ("higher order, 4 × 8 spans a quadrant", build([1.0], True), True)], elev=15.0)  # fmt: skip
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.0), sharex=True)
    for ax, q in zip(axes.T.ravel(), NAMES, strict=True):
        ax.plot(nu, ref[q], **{**REF, "ms": 3.5, "label": "multipole"})
        if q in FIGS:
            x, y, _ = at_read(nu, ref[q], FIGS[q])
            ax.plot(x, y, **{**READ, "label": "Wu (1995), read"})
        ax.plot(nu, res[False][q], color=C[0], label="low order")
        ax.plot(nu, res[True][q], color=C[1], ls="--", label="higher order")
        ax.set_title(q, fontsize=9)
    for ax in axes[2]:
        ax.set_xlabel("νa")
    axes[0, 0].legend(fontsize=7)
    save(fig, NAME)
    out = {"nu": nu}
    for higher, key in ((False, "low"), (True, "higher")):
        v = res[higher]
        out[key] = {"analytic": dict(zip(NAMES, rel_cols(np.array([v[q] for q in NAMES]).T,
                                                         np.array([ref[q] for q in NAMES]).T), strict=True)),
                    "figure": read_differences(nu, v, FIGS)}  # fmt: skip
    out["reference_figure"] = read_differences(nu, ref, FIGS)
    print("  two spheres:", {k: out[k] for k in ("low", "higher")})
    record(NAME, out)


def summary(n):
    d = n[NAME]
    return (f"multipole solution within {pc(max(d['low']['analytic'].values()))} (low order) and "
            f"{size(max(d['higher']['analytic'].values()))} (higher order); the figures read agree to their "
            "reading")  # fmt: skip


INDEX = {"case": "Two submerged spheres", "checked": "$A$, $B$, $X$ of two interacting submerged spheres, deep water",
         "sources": "Wu (1995) figs. 2–10 (read); multipole solution", "methods": "low, higher"}  # fmt: skip


def head(q):
    """The header of a quantity in the page's tables: a11 as a_11."""
    return f"${q[0]}_{{{q[1:]}}}$"


def page(n):
    d = n[NAME]
    rows = [[label] + [e(d[key]["analytic"][q]) for q in NAMES]
            for key, label in (("low", "low order"), ("higher", "higher order"))]  # fmt: skip
    keys = list(FIGS)
    rows_fig = [[label] + [pct(src[q], 1) for q in keys]
                for src, label in ((d["low"]["figure"], "low order"), (d["higher"]["figure"], "higher order"),
                                   (d["reference_figure"], "multipole"))]  # fmt: skip
    lo, hi = max(d["low"]["analytic"].values()), max(d["higher"]["analytic"].values())
    fig = max(d["reference_figure"].values())
    return rf"""
# Two submerged spheres

**Result.** For two interacting submerged spheres in deep water, the higher order agrees with a
multipole solution within {size(hi)} of the largest value of each quantity, and the low order within
{pc(lo)}, its discretization. The published figures of Wu (1995) are matched within {pct(fig, 1)} of
their largest value read, by the multipole solution as by HydroBEM: the accuracy of reading them.

## The problem

Two spheres of radius $a = 1$ m are submerged side by side in deep water, their centres at
$(\pm 2a, 0, -1.5a)$: spacing $4a$, $1.5a$ below the free surface. Waves come at $45^\circ$, over
$0.1 \le \nu a \le 3$ with $\nu a = \omega^2 a/g$. Both spheres move together in each mode.

The quantities are those of sphere 2 with both spheres moving in the same mode (Wu 1995, §7): the
added mass and damping $A^{{21}} + A^{{22}}$ over $\rho V$ and $\rho \omega V$, with $V = 4\pi a^3/3$,
and the exciting force $\lvert E^2 \rvert$ over $\rho \omega^2 V A e^{{-\nu a}}$. That is the
normalization of the figures: the text's eq. (46) has $e^{{-\nu h}}$ with $h$ the depth of the
centres, but the figures agree with $e^{{-\nu a}}$.

![Meshes](../figures/mesh_{NAME}.png)

*Figure 1. The two meshes, expanded to both spheres: the wetted surface in blue. Left: low order,
one quadrant of 24 bands of latitude by 12 meridians (equal angles), expanded about each centre.
Right: higher order, the exact sphere as one analytic patch per quadrant (the revolved meridian;
patch edges in black, span edges in grey), $4 \times 8$ spans of order 3. There is no interior
surface: the spheres are submerged, so there are no irregular frequencies.*

## The references

The main reference is the multipole solution of `scripts/gen_sphere_array_reference.py`, the method
of Wu (1995), derived and implemented in this repository: spherical multipoles about each centre with
their free-surface images and the wave integral. It uses degree 10 and 160 points of the wave
integral; degree 8 reproduces it to $7 \times 10^{{-7}}$ of the largest value
(`tests/data/two_spheres.txt`). It is evaluated here at every $\nu a$.

The published reference is Wu (1995), figures 2–10: the curves of case 2, **read by eye** from the
page images at $\nu a = 0.25$ to 2.5. They are read to about $\pm 0.005$ in the added mass and the
damping and $\pm 0.01$ in the exciting force, about 1 % of the values, more where a curve is steep.

## Agreement with the references

![Comparison](../figures/{NAME}.png)

*Figure 2. Sphere 2 against $\nu a$: the multipole solution (circles), Wu's figures (crosses, read),
HydroBEM low order (solid) and higher order (dashed); the columns are surge, sway and heave.*

The higher order agrees with the multipole solution within {e(hi)} of the largest value of each
quantity with 480 unknowns in all (Table 1). The low order of 2304 flat panels agrees within
{e(lo)}, its discretization, which is largest in the damping, whose peak is small.

*Table 1. Largest difference from the multipole solution over $0.1 \le \nu a \le 3$, relative to its
largest value of each quantity.*

{table(["Method"] + [head(q) for q in NAMES], rows)}

The figures read are matched by the multipole solution within {pct(fig, 1)} of the largest value read,
as by HydroBEM (Table 2): that is the reading accuracy.

*Table 2. Largest difference from Wu's figures 2–10 at the points read, relative to the largest
value read of each quantity.*

{table(["Against figs. 2–10 (read)"] + [head(q) for q in keys], rows_fig)}

## Reproducing this page

The example [`examples/{NAME}`](../../../examples/{NAME}) runs the case at full scale ($a = 3$ m).
This page, its figures and its numbers are written by `validation/two_submerged_spheres.py` through
`validation/make_report.py`.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
