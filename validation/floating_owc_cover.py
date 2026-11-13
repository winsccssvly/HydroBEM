"""A floating tube whose moonpool is closed by a rigid cover in the free surface (a free-surface
plate of the hull's body, theory §9.7), deep water: the heave and pitch added mass, damping and
exciting forces on low-order meshes refined towards the higher order, with the identities of the
problem (the energy of the radiated waves of an axisymmetric body and the Haskind relation)
(docs/validation/cases/floating_owc_cover.md). No independent solution of this body is known to us.

    python validation/floating_owc_cover.py [--quick]
"""

from __future__ import annotations

import math
import sys

import mesh_figure
import numpy as np
from common import RHO, C, G, Timer, e, pc, plt, record, save, size, solve_model, table

from bodies import surface
from meshes import moonpool_lid_panels, moonpool_panels
from problems import analytic, model

NAME = "floating_owc_cover"
LOW = [8, 16, 32]  # panels around the quadrant (n/2 down the walls and across the bottom and the cover)
NAMES = ["A33", "B33", "X3", "A55", "B55", "X5"]


def body(kind, n):
    """The tube (outer radius 2, moonpool 1, draft 1) with the cover r < 1 as a plate of its body."""
    if kind == "higher":
        rows = [[2 * n, 2 * n, 3, 3, 3, 4], [2 * n, n, 3, 3, 3, 4], [n, 2 * n, 3, 3, 3, 4], [n, n, 3, 3, 3, 4]]
        mesh = analytic("moonpool_cylinder", rows, parts=["side", "bottom", "moonpool_wall", "moonpool_surface"],
                        mapping="nonuniform", radius=2.0, draft=1.0, moonpool_radius=1.0, symmetry="xz_and_yz")
        hull, cover = (mesh, (1, 3)), (mesh, (4, 4))
    else:
        h, c = moonpool_panels(n=n), moonpool_lid_panels(n=n)
        mesh = surface(np.concatenate([h, c]), symmetry="xz_and_yz")
        hull, cover = (mesh, (1, len(h))), (mesh, (len(h) + 1, len(h) + len(c)))
    return {"mesh": hull, "modes": ["heave", "pitch"], "interior_surface": "none", "rotation_center": [0, 0, 0],
            "free_surface_plates": [{"mesh": cover}]}  # fmt: skip


def build(w, kind, n):
    return model(body(kind, n), w, [0.0], environment={"depth": math.inf, "density": RHO, "gravity": G},
                 options={"higher_order": kind == "higher"})  # fmt: skip


def quantities(r):
    w = r.omegas
    x, xh = r.excitation.diffraction[:, 0, :], r.excitation.haskind[:, 0, :]
    out = {}
    for i, mode, n in (("3", "heave", 3), ("5", "pitch", 4)):
        d, k = r.coordinate("body", mode)
        out["A" + i + i] = r.coefficient("added_mass", ("body", mode), ("body", mode)) / (RHO * 1.0**n)
        out["B" + i + i] = r.coefficient("radiation_damping", ("body", mode), ("body", mode)) / (RHO * w)
        out["X" + i] = np.abs(x[:, d]) / (RHO * G)
        out["H" + i] = np.abs(x[:, d] - xh[:, k]) / np.abs(x[:, d]).max()
    # the energy of the radiated waves of an axisymmetric body in deep water: B33 = w^3 |X3|^2 / (2 rho g^3)
    b33 = r.coefficient("radiation_damping", ("body", "heave"), ("body", "heave"))
    out["energy"] = np.abs(b33 - w**3 * (out["X3"] * RHO * G) ** 2 / (2 * RHO * G**3)) / np.abs(b33).max()
    return out


def main(quick=False):
    ka = np.round(np.arange(0.1, 2.51, 0.3 if quick else 0.1), 6)  # K times the moonpool radius 1
    w = np.sqrt(ka * G)
    res = {}
    for kind, ns in (("low", LOW), ("higher", [2, 4])):
        for n in ns:
            with Timer(f"OWC cover: {kind} {n}"):
                res[kind, n] = quantities(solve_model(build(w, kind, n)))
    ho = res["higher", 4]
    out = {"ka": ka}
    for (kind, n), v in res.items():
        out[f"{kind}{n}"] = {"vs_higher": {q: float(np.abs(v[q] - ho[q]).max() / np.abs(ho[q]).max()) for q in NAMES},
                             "energy": float(v["energy"].max()),
                             "haskind": float(max(v["H3"].max(), v["H5"].max()))}  # fmt: skip
    mesh_figure.draw(NAME, [(f"low order, {LOW[1]} panels around a quadrant", build([1.0], "low", LOW[1]), False,
                             {"cut": (1, -1)}),
                            ("higher order, analytic patches", build([1.0], "higher", 4), True, {"cut": (1, -1)})],
                     elev=30.0)  # fmt: skip
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), sharex=True)
    for ax, q in zip(axes.ravel(), NAMES, strict=True):
        for j, n in enumerate(LOW):
            ax.plot(ka, res["low", n][q], color=C[j], lw=1.2, label=f"low order, {n} around")
        ax.plot(ka, ho[q], color="black", ls="--", label="higher order")
        ax.set_title(q, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Ka (a = 1 m, the moonpool radius)")
    axes[0, 0].legend(fontsize=7)
    save(fig, NAME)
    print("  OWC cover:", out)
    record(NAME, out)


def summary(n):
    d = n[NAME]
    return (f"no independent reference; the low order converges to the higher order "
            f"({pc(max(d[f'low{LOW[-1]}']['vs_higher'].values()))} at {LOW[-1]} panels around); energy identity "
            f"within {size(d['higher4']['energy'])}")  # fmt: skip


INDEX = {"case": "Floating OWC cover (free-surface plate)",
         "checked": "heave, pitch $A$, $B$, $X$ of a tube with a rigid cover over its moonpool",
         "sources": "identities (energy, Haskind); low against higher order", "methods": "low, higher"}  # fmt: skip


def head(q):
    """The header of a quantity in the page's tables: A33 as A_33, X3 as |X_3|."""
    if q.startswith("X"):
        return rf"$\lvert X_{q[1:]} \rvert$"
    return f"${q[0]}_{{{q[1:]}}}$"


def page(n):
    d = n[NAME]
    rows = [[f"low order, {k} around"] + [e(d[f"low{k}"]["vs_higher"][q]) for q in NAMES] for k in LOW]
    rows.append(["higher order, half the spans"] + [e(d["higher2"]["vs_higher"][q]) for q in NAMES])
    ident = [[f"low order, {k} around", e(d[f"low{k}"]["energy"]), e(d[f"low{k}"]["haskind"])] for k in LOW]
    ident += [[f"higher order, {lab}", e(d[f"higher{k}"]["energy"]), e(d[f"higher{k}"]["haskind"])]
              for k, lab in ((2, "half the spans"), (4, "the spans above"))]  # fmt: skip
    lo = [max(d[f"low{k}"]["vs_higher"].values()) for k in LOW]
    h = d["higher4"]
    return rf"""
# Floating OWC cover: a plate over a moonpool

**Result.** No independent solution of this body is known, so the case checks consistency, not
accuracy. The low order converges to the higher order as the mesh is refined, its largest difference
falling from {pc(lo[0])} to {pc(lo[-1])} over {", ".join(map(str, LOW))} panels around a quadrant. The
energy identity and the Haskind relation hold in the higher order within {size(h['energy'])} and
{size(h['haskind'])}, and in both methods to their discretization.

## The problem

A floating tube has its moonpool closed by a rigid cover lying in the free surface. The cover is a
free-surface plate of the hull's body (theory §9.7) and moves with it, as the roof of a closed
oscillating water column whose air chamber has collapsed, or a hatch. The [dock](dock.md) checks the
plate against independent solutions; here the plate is part of a floating body.

The tube has outer radius 2 m, moonpool radius $a = 1$ m and draft 1 m, in deep water, heading
$0^\circ$; the cover is the disc $r < 1$ m in $z = 0$, its edge on the moonpool wall. We compare heave
and pitch about the origin in the free surface: $A_{{33}}/(\rho a^3)$, $B_{{33}}/(\rho \omega a^3)$,
$\lvert X_3 \rvert/(\rho g A a^2)$, $A_{{55}}/(\rho a^5)$ (as $A_{{55}}/\rho$ with $a = 1$), $B_{{55}}$
and $\lvert X_5 \rvert$, at $0.1 \le Ka \le 2.5$ with $Ka = \omega^2 a/g$.

The range lies below the first irregular frequency of the tube's wall, near $Ka = 3$, where the
integral equation would fail because the water that would fill the wall resonates; so no interior
surface is used.

![Meshes](../figures/mesh_{NAME}.png)

*Figure 1. The two meshes, expanded to the whole body and cut open (a quarter removed): the wetted
walls and bottom in blue, the cover in red. Left: low order, a quadrant of $n$ panels around, $n/2$
down each wall, across the bottom and across the cover (shown $n = {LOW[1]}$; the comparison uses
$n = {", ".join(map(str, LOW))}$). Right: higher order, the analytic tube with its inner free surface as
the plate (patch edges in black, span edges in grey), $8 \times 8$ spans on the walls, $8 \times 4$ on
the bottom, $4 \times 8$ on the inner wall and $4 \times 4$ on the cover per quadrant (the comparison
also uses half these), order 3, clustered at the bottom edges.*

## The references

No published or analytic solution of this body is known to us. Three checks take its place:

- **The energy of the radiated waves.** The waves a body radiates carry away exactly the energy its
  damping absorbs. For an axisymmetric body in deep water this ties the heave damping to the exciting
  force, $B_{{33}} = \omega^3 \lvert X_3 \rvert^2/(2\rho g^3)$, exactly (Newman 1977, the Haskind
  relation for the damping), with the cover part of the radiating and diffracting surface.
- **The Haskind relation.** The exciting force can be computed from the radiation potentials alone,
  without solving the diffraction problem; it is compared with the force from the pressure of the
  diffraction problem.
- **The methods against each other.** The low order on three meshes is compared with the higher order
  on the exact geometry.

## Agreement between the methods

![Comparison](../figures/{NAME}.png)

*Figure 2. Heave and pitch against $Ka$: the low order on three meshes and the higher order (dashed).*

The low order converges to the higher order as the mesh is refined (Table 1), at the rate of its other
problems: the cover's edge meets the moonpool wall at a right angle of the fluid.

*Table 1. Largest difference from the higher order over $Ka$, relative to its largest value of each
quantity.*

{table(["Mesh"] + [head(q) for q in NAMES], rows)}

## The identities

The energy identity and the Haskind relation hold in both methods to their discretization (Table 2).

*Table 2. The identities: largest $\lvert B_{{33}} - \omega^3 \lvert X_3 \rvert^2/(2\rho g^3) \rvert$
relative to the largest $B_{{33}}$, and largest $\lvert X - X_{{\mathrm{{Haskind}}}} \rvert$ relative to the
largest $\lvert X \rvert$ (heave and pitch).*

{table(["Mesh", "energy identity", "Haskind relation"], ident)}

This case shows consistency, not accuracy against an independent solution
([not verified](../not_verified.md)).

## Reproducing this page

This page, its figures and its numbers are written by `validation/floating_owc_cover.py` through
`validation/make_report.py`.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
