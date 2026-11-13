"""A free-surface plate: a rigid massless disc in z = 0 heaving (theory §9.7). In water of depth 2a
its added mass and damping against the eigenfunction matching of scripts/gen_cylinder_reference.py
with zero draft (the dock: the whole depth under the plate is the inner region); at zero frequency
in deep water against Rayleigh's piston in a rigid plane, A33(0) = (8/3) rho a^3; the low order on
three meshes and the higher order (docs/validation/cases/dock.md).

    python validation/dock.py [--quick]
"""

from __future__ import annotations

import math
import sys

import mesh_figure
import numpy as np
from common import REF, RHO, C, G, Timer, e, pc, plt, record, save, size, solve_model, table

from bodies import spline, surface
from meshes import QUARTER_CIRCLE, QUARTER_WEIGHTS, disc_panels
from problems import load_script, model

NAME = "dock"
DEPTH = 2.0  # radius a = 1
LOW = [8, 16, 32]  # rings a quadrant (2n divisions around the quadrant)
HIGH = [4, 8]  # spans a quadrant in each direction


def plate(kind, n):
    if kind == "higher":
        disk = np.array([[(r * a, r * b, 0.0) for a, b in QUARTER_CIRCLE] for r in (1.0, 0.0)])  # normals down
        patch = (3, 2, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), disk, np.array([QUARTER_WEIGHTS] * 2))
        return spline([patch], (n, n, 3, 3, 3, 4), symmetry="xz_and_yz")
    return surface(disc_panels(1.0, n_r=n, n_phi=2 * n, quarter=True), symmetry="xz_and_yz")


def build(w, kind, n, depth=DEPTH):
    body = {"name": "plate", "modes": ["heave"], "free_surface_plates": [{"mesh": plate(kind, n)}]}
    return model(body, w, [], environment={"depth": depth, "density": RHO, "gravity": G},
                 options={"higher_order": kind == "higher", "diffraction": False})  # fmt: skip


def reference(ka):
    """A/(rho a^3), B/(rho w a^3) of the eigenfunction matching with zero draft, N = 200 and 400
    extrapolated with the observed order N^-2."""
    gen = load_script("gen_cylinder_reference")
    out = []
    for k in ka:
        a, b = (np.array(gen.solve(DEPTH, 0.0, float(k), n, "heave")) for n in (200, 400))
        out.append(b + (b - a) / 3.0)
    return np.array(out)


def heave(r):
    return np.c_[r.quantity("added_mass")[:, 2, 0] / RHO, r.quantity("radiation_damping")[:, 2, 0] / (RHO * r.omegas)]


def main(quick=False):
    ka = np.round(np.linspace(0.1, 3.0, 8 if quick else 30), 6)
    w = np.sqrt(ka * G)
    with Timer("dock: reference"):
        ref = reference(ka)
    res, a0 = {}, {}
    with Timer("dock: meshes"):
        for kind, ns in (("low", LOW), ("higher", HIGH)):
            for n in ns:
                res[kind, n] = heave(solve_model(build(w, kind, n)))
                r0 = solve_model(build([0.0], kind, n, depth=math.inf))
                a0[kind, n] = float(r0.quantity("added_mass_zero_frequency")[2, 0] / (RHO * 8.0 / 3.0)) - 1.0
    mesh_figure.draw(NAME, [(f"low order, {LOW[1]} rings a quadrant", build([1.0], "low", LOW[1]), False),
                            (f"higher order, {HIGH[-1]} × {HIGH[-1]} spans a quadrant",
                             build([1.0], "higher", HIGH[-1]),
                             True)], elev=35.0)  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, col, lab in ((axes[0], 0, "A₃₃/(ρa³)"), (axes[1], 1, "B₃₃/(ρωa³)")):
        ax.plot(ka, ref[:, col], **{**REF, "label": "eigenfunction matching"})
        ax.plot(ka, res["low", LOW[-1]][:, col], color=C[0], label=f"low order, {LOW[-1]} rings")
        ax.plot(ka, res["higher", HIGH[-1]][:, col], color=C[1], ls="--", label=f"higher order, {HIGH[-1]}² spans")
        ax.set_xlabel("Ka = ω²a/g")
        ax.set_ylabel(lab)
    axes[0].axhline(8.0 / 3.0, color="#888888", lw=0.8, ls=":")
    axes[0].text(0.1, 8.0 / 3.0 * 1.01, "8/3 (ω = 0, deep water)", fontsize=7, color="#555555")
    axes[0].legend()
    save(fig, NAME)
    out = {"ka": ka}
    for (kind, n), v in res.items():
        out[f"{kind}{n}"] = {"A": float(np.abs(v[:, 0] - ref[:, 0]).max() / np.abs(ref[:, 0]).max()),
                             "B": float(np.abs(v[:, 1] - ref[:, 1]).max() / np.abs(ref[:, 1]).max()),
                             "A0": abs(a0[kind, n])}  # fmt: skip
    print("  dock:", out)
    record(NAME, out)


def summary(n):
    d = n[NAME]
    lo, hi = d[f"low{LOW[-1]}"], d[f"higher{HIGH[-1]}"]
    return (f"eigenfunction matching within {pc(max(lo['A'], lo['B']))} (low order) and "
            f"{size(max(hi['A'], hi['B']))} (higher order); Rayleigh's piston within {size(lo['A0'])} and "
            f"{size(hi['A0'])}")  # fmt: skip


INDEX = {"case": "Dock (free-surface plate)",
         "checked": r"heave $A$, $B$ of a rigid disc in the free surface; $\omega = 0$",
         "sources": "eigenfunction matching (Garrett 1971; Yeung 1981); Rayleigh (1896)",
         "methods": "low, higher"}  # fmt: skip


def page(n):
    d = n[NAME]
    rows = [[f"low order, {k} rings a quadrant", e(d[f"low{k}"]["A"]), e(d[f"low{k}"]["B"]), e(d[f"low{k}"]["A0"])]
            for k in LOW]  # fmt: skip
    rows += [[rf"higher order, ${k} \times {k}$ spans", e(d[f"higher{k}"]["A"]), e(d[f"higher{k}"]["B"]),
              e(d[f"higher{k}"]["A0"])] for k in HIGH]  # fmt: skip
    b = [d[f"low{k}"]["B"] for k in LOW]
    rb = [b[0] / b[1], b[1] / b[2]]
    ra = (d[f"low{LOW[0]}"]["A0"] / d[f"low{LOW[-1]}"]["A0"]) ** (1 / (len(LOW) - 1))
    rh = d[f"higher{HIGH[0]}"]["B"] / d[f"higher{HIGH[-1]}"]["B"]
    lo, hi = d[f"low{LOW[-1]}"], d[f"higher{HIGH[-1]}"]
    return rf"""
# Dock: a rigid plate in the free surface

**Result.** A rigid disc heaving in the free surface agrees with an eigenfunction-matching solution
within {pc(max(lo['A'], lo['B']))} of the largest reference value in the low order
({LOW[-1]} rings per quadrant) and {size(max(hi['A'], hi['B']))} in the higher order
(${HIGH[-1]} \times {HIGH[-1]}$ spans). At zero frequency both methods reproduce Rayleigh's piston,
within {size(lo['A0'])} and {size(hi['A0'])}. The damping, set by the edge of the plate, is the larger
difference throughout and converges as $h^2$.

## The problem

A free-surface plate (theory §9.7) is a rigid, massless surface lying in the free surface $z = 0$.
The water touches only its underside, and the free surface around it carries the usual condition. It
models a dock, a cover over a moonpool ([floating OWC cover](floating_owc_cover.md)) or a very thin
floating body.

Here the plate is a disc of radius $a = 1$ m in $z = 0$, heaving. We compare its added mass
$A_{{33}}/(\rho a^3)$ and damping $B_{{33}}/(\rho \omega a^3)$ in water of depth $h = 2a$ at {len(d["ka"])} values
$0.1 \le Ka \le 3$, where $Ka = \omega^2 a/g$, and its added mass at $\omega = 0$ in deep water.

![Meshes](../figures/mesh_{NAME}.png)

*Figure 1. The two meshes of the disc, expanded to the whole plate and seen from above, the plate in
red. Left: low order, a quadrant of $n$ uniform rings and $2n$ divisions around (shown
$n = {LOW[1]}$; the comparison uses $n = {", ".join(map(str, LOW))}$). Right: higher order, one
rational quarter-disk patch per quadrant (the exact circle; patch edges in black, span edges in
light grey) with $n \times n$ spans of order 3 (shown $n = {HIGH[-1]}$; the comparison uses
$n = {", ".join(map(str, HIGH))}$).*

## The references

In depth $2a$ the reference is the eigenfunction-matching solution of
`scripts/gen_cylinder_reference.py`, the formulation of Garrett (1971) and Yeung (1981), derived and
implemented in this repository, with zero draft: the whole depth under the plate is the inner
region. Its truncations $N = 200$ and 400 are extrapolated with the order $N^{{-2}}$ observed; the step
from 200 to 400 is $5 \times 10^{{-6}}$ of the value.

At $\omega = 0$ the free surface is the rigid plane $z = 0$, and the heaving disc is a piston in it
(Rayleigh 1896, §307). Its added mass is exact:

$$
A_{{33}}(0) = \frac{{\rho}}{{2\pi}} \iint \frac{{\mathrm{{d}}S \, \mathrm{{d}}S'}}{{\lvert x - x' \rvert}}
= \frac{{8}}{{3}} \rho a^3.
$$

## Agreement with the references

![Comparison](../figures/{NAME}.png)

*Figure 2. $A_{{33}}$ and $B_{{33}}$ against $Ka$ in depth $2a$: eigenfunction matching (circles), the
finest low-order mesh (solid) and the finer higher-order mesh (dashed).*

The curves lie on the reference over the whole range. Table 1 gives the largest difference of each
quantity on every mesh.

*Table 1. Largest difference from the references over $0.1 \le Ka \le 3$, relative to the largest
reference value ($A_{{33}}(0)$: relative to $8/3$).*

{table(["Mesh", "$A_{33}$, depth $2a$", "$B_{33}$, depth $2a$", "$A_{33}(0)$, deep ($8/3$)"], rows)}

## Does it converge?

The added mass converges fast in both methods (Table 1). The damping is governed by the edge of the
plate, where it borders the free surface and the velocity is singular. It is the larger difference
throughout, and it falls by a factor of {rb[0]:.1f} and {rb[1]:.1f} per halving of the low-order
panels, about $h^2$.

At $\omega = 0$ the low order approaches Rayleigh's piston by a factor of about {ra:.0f} per halving
of the uniform rings. The higher order on uniform spans of the rational patch reaches the accuracy of
the finest low-order mesh with $4 \times 4$ spans per quadrant, and its damping difference falls by a
factor of {rh:.1f} from 4 to 8 spans.

## Reproducing this page

This page, its figures and its numbers are written by `validation/dock.py` through
`validation/make_report.py`.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
