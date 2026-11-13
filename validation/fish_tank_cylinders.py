"""The circular fish tanks of Newman (2021): the fluid in a vertical circular tank (walls of zero
thickness) forced in surge and in pitch, its added masses A11, A15 and A55 against the closed-form
interior series of Faltinsen & Timokha (2009), section 5.4.4 (tests/python/problems.circular_tank),
for the four drafts of the paper, in the low and the higher order
(docs/validation/cases/fish_tank_cylinders.md).

    python validation/fish_tank_cylinders.py [--quick]
"""

from __future__ import annotations

import math
import sys

import mesh_figure
import numpy as np
from common import READ, REF, C, Timer, e, pc, plt, record, save, solve_model, table, tex
from scipy.special import jnp_zeros

from bodies import surface
from meshes import cylinder_panels
from problems import circular_tank, model

NAME = "fish_tank_cylinders"
RHO, G = 1.0, 1.0  # the paper's nondimensional units (radius 1)
K1 = float(jnp_zeros(1, 1)[0])  # 1.841184, the first zero of J1'
DRAFTS = [0.5, float(np.arccosh(2.0)) / K1, 1.0, 2.0]  # the second: cosh(k1 d) = 2, the non-resonant draft
AWAY = 0.25  # the comparison leaves out |K - K1 tanh(k1 d)| <= AWAY (the resonance; the panels shift it)
NAMES = ["A11", "A15", "A55"]
# Newman (2021), Fig. 2, the RAOs of the tank free in waves read from the page image (its left axis,
# 0 to 1; the axes calibrated on their tick marks, each value checked against its curve), at
# wavenumbers away from the zeros and the resonances: {draft: {quantity: {K: |xi|}}}; xi1, xi5 both
# modes free (red, orange), xi1_only, xi5_only one mode alone (blue, green); the sloshing (dashed,
# right axis) is not read
FIG_2 = {
    0.5: {"xi1": {0.5: 0.72, 1.0: 0.40, 2.5: 0.41, 3.0: 0.55, 3.5: 0.53},
          "xi5": {0.5: 0.41, 1.0: 0.60, 2.5: 0.44, 3.0: 0.31, 3.5: 0.17},
          "xi1_only": {0.5: 0.65, 1.0: 0.27, 2.5: 0.54, 3.0: 0.57, 3.5: 0.50},
          "xi5_only": {0.5: 0.34, 1.0: 0.07, 2.5: 0.07, 3.0: 0.06, 3.5: 0.05}},
    0.715278: {"xi1": {0.5: 0.65, 1.0: 0.33, 3.0: 0.53, 3.5: 0.41},
               "xi5": {0.5: 0.37, 1.0: 0.48, 2.5: 0.27, 3.0: 0.17, 3.5: 0.06},
               "xi1_only": {0.5: 0.68, 1.0: 0.36, 2.5: 0.38, 3.0: 0.51, 3.5: 0.40},
               "xi5_only": {0.5: 0.83, 1.0: 0.72, 2.5: 0.20, 3.0: 0.13, 3.5: 0.09}},
    1.0: {"xi1": {0.5: 0.60, 1.0: 0.30, 3.0: 0.51, 3.5: 0.25},
          "xi5": {0.5: 0.24, 1.0: 0.26, 2.5: 0.24, 3.0: 0.24},
          "xi1_only": {0.5: 0.67, 1.0: 0.37, 3.0: 0.42, 3.5: 0.20},
          "xi5_only": {2.5: 0.26, 3.0: 0.14, 3.5: 0.09}},
    2.0: {"xi1": {0.5: 0.81, 1.0: 0.53, 3.0: 0.60, 3.5: 0.36},
          "xi5": {0.5: 0.27, 1.0: 0.27, 3.0: 0.45, 3.5: 0.28},
          "xi1_only": {0.5: 0.58, 1.0: 0.31, 3.0: 0.07, 3.5: 0.04},
          "xi5_only": {0.5: 0.53, 1.0: 0.25, 2.5: 0.03, 3.5: 0.01}},
}  # fmt: skip
FIG_2_READ = 0.01  # the reading accuracy of Fig. 2, in its units (|xi|), more where a curve is steep
RAOS = ["xi1", "xi5", "xi1_only", "xi5_only"]
RAO_LABELS = ["|ξ₁|, both free", "|ξ₅|, both free", "|ξ₁|, surge alone", "|ξ₅|, pitch alone"]


def low_mesh(d):
    """The hull (side and bottom, normals into the sea) and the tank (the same panels reversed),
    16 around the quadrant, 16 + 8d down the side, 14 across the bottom (cosine-spaced)."""
    hull = cylinder_panels(1.0, d, n_phi=16, n_z=int(16 + 8 * d), n_r=14, quarter=True, cosine=True)
    mesh = surface(np.concatenate([hull, hull[:, ::-1]]), symmetry="xz_and_yz")
    n = len(hull)
    return (mesh, (1, n)), (mesh, (n + 1, 2 * n)), "graded"


def higher_mesh(d, size=0.25):
    """Analytic revolved segments a quadrant: the hull's side and bottom, the waterplane disc (the
    interior surface) and the tank's side and bottom (the hull's reversed); spans of at most 0.25,
    clustered at the bottom corner."""
    rev = {"linear": "linear", "quadratic_start": "quadratic_end", "quadratic_end": "quadratic_start"}
    hull = [((1.0, 0.0), (1.0, -d), "quadratic_end"), ((1.0, -d), (0.0, -d), "quadratic_start")]
    segs = [(*s, "hull") for s in hull] + [((0.0, 0.0), (1.0, 0.0), "linear", "interior")]
    segs += [(b, a, rev[p], "tank") for a, b, p in hull]
    spans = lambda length: max(1, int(np.ceil(length / size - 1e-9)))  # noqa: E731
    rows, patches, rg = [], [], {}
    for i, ((r0, z0), (r1, z1), profile, role) in enumerate(segs):
        rows.append([spans(0.5 * np.pi * max(r0, r1)), spans(np.hypot(r1 - r0, z1 - z0)), 3, 3, 3, 4])
        patches.append({"type": "AnalyticPatch", "map": "line", "profile": profile, "a": 1.0, "b": 1.0,
                        "theta0": 0.0, "theta1": 0.5 * np.pi, "p": [r0, z0, r1, z1]})  # fmt: skip
        rg[role] = (rg.get(role, (i + 1, i + 1))[0], i + 1)
    mesh = {"kind": "analytic_patches", "discretization": np.array(rows), "symmetry": "xz_and_yz", "patches": patches}
    return (mesh, rg["hull"]), (mesh, rg["tank"]), (mesh, rg["interior"])


def build(d, K, higher, density=1.0, waves=False):
    """The tank forced in surge and pitch; with waves the massless tank free in both in head waves
    (the paper's section 3)."""
    hull, tank, interior = higher_mesh(d) if higher else low_mesh(d)
    body = {"mesh": hull, "modes": ["surge", "pitch"], "interior_surface": interior,
            "tanks": [{"mesh": tank, "density_ratio": density}]}  # fmt: skip
    if waves:
        body.update(mass=0.0, center_of_gravity=[0.0, 0.0, 0.0])
    return model(body, np.sqrt(G * np.asarray(K)), [0.0] if waves else [],
                 environment={"depth": math.inf, "density": RHO, "gravity": G},
                 options={"higher_order": higher, "diffraction": waves, "motion": waves})  # fmt: skip


def free_raos(d, K, higher):
    """|ξ1|, |ξ5| with both modes free and with each alone [4, K]: the massless tank (with its fluid)
    in head waves of unit amplitude. Both free is the engine's motion; one mode alone solves the
    equation of motion (theory §3.1) of that mode, C - ω² A + iωB, with the same coefficients; the
    two modes solved from them reproduce the engine's motion (returned as its largest relative
    difference)."""
    r = solve_model(build(d, K, higher, waves=True))
    w = np.sqrt(G * np.asarray(K))
    modes = ("surge", "pitch")
    c = np.asarray(r.hydrostatics.stiffness[0])[np.ix_([0, 4], [0, 4])]
    a = np.array([[r.coefficient("added_mass", ("body", i), ("body", j)) for j in modes] for i in modes])
    b = np.array([[r.coefficient("radiation_damping", ("body", i), ("body", j)) for j in modes] for i in modes])
    x = np.array([r.coefficient("excitation_diffraction", ("body", i))[:, 0] for i in modes])
    z = c[:, :, None] - w**2 * a + 1j * w * b  # [2, 2, K]
    both = r.bodies["body"].motion[:, 0, [0, 4]].T
    check = np.linalg.solve(np.moveaxis(z, 2, 0), x.T[:, :, None])[:, :, 0].T
    alone = x / np.array([z[0, 0], z[1, 1]])
    return np.abs(np.r_[both, alone]), float(np.abs(check - both).max() / np.abs(both).max())


def tank_added_mass(d, K, higher):
    """A11, A15, A55 of the tank alone [3, K]: the body with the tank less the body with an empty one
    (the two fluids do not influence each other)."""
    a = []
    for density in (1.0, 0.0):
        r = solve_model(build(d, K, higher, density))
        a.append(np.array([r.coefficient("added_mass", ("body", i), ("body", j))
                           for i, j in (("surge", "surge"), ("surge", "pitch"), ("pitch", "pitch"))]))  # fmt: skip
    return (a[0] - a[1]) / RHO


def main(quick=False):
    K = np.round(np.arange(0.05, 4.01, 0.2 if quick else 0.05), 6)
    out = {"K": K, "drafts": DRAFTS}
    res = {}
    fig, axes = plt.subplots(len(DRAFTS), 3, figsize=(11.0, 10.0), sharex=True)
    for row, d in enumerate(DRAFTS):
        kres = K1 * math.tanh(K1 * d)
        ref = circular_tank(d, K)
        far = np.abs(K - kres) > AWAY
        res[d] = {}
        for higher in (False, True):
            key = "higher" if higher else "low"
            with Timer(f"fish tank d = {d:.4g}: {key} order"):
                res[d][key] = tank_added_mass(d, K, higher)
        out[f"d{d:.4g}"] = {"K1": kres, **{key: (np.abs(v - ref)[:, far].max(axis=1) / np.abs(ref[:, far]).max(axis=1))
                                            for key, v in res[d].items()}}  # fmt: skip
        for col in range(3):
            ax = axes[row, col]
            ax.plot(K, ref[col], **{**REF, "ms": 3, "label": "Faltinsen & Timokha series"})
            ax.plot(K, res[d]["low"][col], color=C[0], label="low order")
            ax.plot(K, res[d]["higher"][col], color=C[1], ls="--", label="higher order")
            lim = np.abs(ref[col][far]).max() * 1.6
            ax.set_ylim(-lim, lim)
            ax.axvline(kres, color="#bbbbbb", lw=0.8)
            ax.set_title(f"{NAMES[col]}, d = {d:.4g}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("K = ω²a/g")
    axes[0, 0].legend(fontsize=7)
    save(fig, NAME)
    mesh_figure.draw(NAME, [("low order, d = 1", build(1.0, [1.0], False), False, {"cut": (1, -1)}),
                            ("higher order, d = 1", build(1.0, [1.0], True), True, {"cut": (1, -1)})],
                     elev=30.0)  # fmt: skip
    reads = [k for v in FIG_2.values() for q in v.values() for k in q]
    out["free"] = free_in_waves(np.union1d(K, np.round(reads, 6)))
    print("  fish tanks:", out)
    record(NAME, out)


def free_in_waves(K):
    """The tank free in waves against the values read from Newman's Fig. 2: per draft and method
    the largest |difference| of each RAO at the points read, and the check of the equation of motion."""
    out = {}
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True, sharey=True)
    for ax, d, read in zip(axes.flat, DRAFTS, FIG_2.values(), strict=True):
        res = {}
        for higher in (False, True):
            key = "higher" if higher else "low"
            with Timer(f"fish tank d = {d:.4g} in waves: {key} order"):
                res[key], check = free_raos(d, K, higher)
            diffs = []
            for i, q in enumerate(RAOS):
                at = np.array([np.argmin(np.abs(K - k)) for k in read[q]])
                assert np.allclose(K[at], list(read[q])), q
                diffs.append(float(np.abs(res[key][i, at] - np.array(list(read[q].values()))).max()))
            out[f"d{d:.4g}_{key}"] = diffs
            out[f"d{d:.4g}_{key}_check"] = check
        for i, (q, label) in enumerate(zip(RAOS, RAO_LABELS, strict=True)):
            ax.plot(K, res["higher"][i], color=C[i], label=label)
            ax.plot(K, res["low"][i], color=C[i], ls="--", lw=1.0)
            read_label = "Newman (2021), read" if i == 0 else None
            ax.plot(list(read[q]), list(read[q].values()), **{**READ, "label": read_label})
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"d = {d:.6g}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("K = ω²a/g")
    for ax in axes[:, 0]:
        ax.set_ylabel("|ξ| (m/m, rad/m)")
    axes[0, 0].legend(fontsize=7)
    save(fig, f"{NAME}_free")
    return out


def summary(n):
    d = n[NAME]
    # the small A15 of the non-resonant draft is left out of the headline (its own line on the page)
    lo = max(v for x in DRAFTS for i, v in enumerate(d[f"d{x:.4g}"]["low"]) if not (x == DRAFTS[1] and i == 1))
    hi = max(max(d[f"d{x:.4g}"]["higher"]) for x in DRAFTS)
    return (f"interior series within {pc(lo)} (low order) and {pc(hi)} (higher order) away from the "
            f"sloshing resonance (low order: {pc(d[f'd{DRAFTS[1]:.4g}']['low'][1])} for the small $A_{{15}}$ of "
            f"the non-resonant draft); the RAOs free in waves within {free_worst(d)[1]:.3f} of the values read "
            "from Newman's Fig. 2 (higher order)")  # fmt: skip


def free_worst(d):
    """The largest |difference| from Fig. 2 read over the drafts and RAOs: (low, higher)."""
    return tuple(max(max(d["free"][f"d{x:.4g}_{k}"]) for x in DRAFTS) for k in ("low", "higher"))


INDEX = {"case": "Fish tanks", "checked": "added mass of the fluid in a circular tank forced in surge and pitch; "
         "surge and pitch RAOs of the tank free in waves", "sources": "Faltinsen & Timokha (2009) series, as Newman "
         "(2021); Newman (2021) fig. 2 (read)", "methods": "low, higher"}  # fmt: skip


def page(n):
    d = n[NAME]
    rows = []
    for x in DRAFTS:
        v = d[f"d{x:.4g}"]
        rows.append([f"{x:.6g}", f"{v['K1']:.4f}"] + [e(a) for a in v["low"]] + [e(a) for a in v["higher"]])
    nr, nr_hi = d[f"d{DRAFTS[1]:.4g}"]["low"][1], d[f"d{DRAFTS[1]:.4g}"]["higher"][1]
    lo = max(v for x in DRAFTS for i, v in enumerate(d[f"d{x:.4g}"]["low"]) if not (x == DRAFTS[1] and i == 1))
    hi = max(max(d[f"d{x:.4g}"]["higher"]) for x in DRAFTS)
    f = d["free"]
    f_lo, f_hi = free_worst(d)
    check = max(v for k, v in f.items() if k.endswith("_check"))
    free_rows = [[f"{x:.6g}", method] + [f"{v:.3f}" for v in f[f"d{x:.4g}_{key}"]]
                 for x in DRAFTS for method, key in (("low order", "low"), ("higher order", "higher"))]  # fmt: skip
    free_header = ["$d$", "Method", r"$\lvert \xi_1 \rvert$ both", r"$\lvert \xi_5 \rvert$ both",
                   r"$\lvert \xi_1 \rvert$ alone", r"$\lvert \xi_5 \rvert$ alone"]  # fmt: skip
    return rf"""
# Fish tanks: the fluid in a circular tank

**Result.** The added mass of the water inside a circular tank, forced in surge and pitch, agrees with
the closed-form interior series away from the sloshing resonance: within {pc(lo)} of its largest value
in the low order, the discretization of its polygon, and within {pc(hi)} in the higher order. Both
methods follow the series through the sign changes near the resonance. The one larger difference,
{pc(nr)} in the low order, is that of a coefficient that is itself small, $A_{{15}}$ at the
non-resonant draft. Free in waves, the tank's surge and pitch RAOs agree with the curves of Newman's
Fig. 2, read from the page, within {f_hi:.3f} (higher order) and {f_lo:.3f} (low order), against a
reading accuracy of about {FIG_2_READ}.

## The problem

Circular cylindrical fish tanks float in deep water (Newman 2021). The tank's wall has zero thickness
and separates the fluid inside, of the same density, from the sea. Here the tank alone is forced in
surge and in pitch, the check Newman makes against the closed-form interior solution.

The radius is $a = 1$ and the drafts $d = 0.5$, 0.715278, 1 and 2, in infinite depth with
$\rho = g = 1$; the second is the non-resonant draft $\operatorname{{arccosh}}(2)/k_1$ of Newman's eq.
(1). The origin is at the centre of the tank's free surface. The tank is forced in surge and in pitch
about the origin at $0.05 \le K \le 4$, $K = \omega^2 a/g$. $A_{{11}}$, $A_{{15}}$ and $A_{{55}}$ are
those of the tank fluid alone: the body with the tank less the same body with an empty tank. The two
fluids do not influence each other, so the sea's parts cancel.

The first sloshing resonance lies at $K_1 = k_1 \tanh(k_1 d)$, where $k_1 = 1.841184$ is the first
zero of $J_1'$. There the tank's coefficients are infinite.

![Meshes](../figures/mesh_{NAME}.png)

*Figure 1. The two meshes for $d = 1$, expanded to the whole tank and cut open (a quarter removed) to
show the inside: the hull's wetted surface in blue, the tank's walls in teal, the interior surface in
the waterplane in yellow. The tank's walls are the hull's panels reversed (normals into the tank), so
the two colours overlap on the same surfaces. Left: low order, a quadrant of 16 panels around,
$16 + 8d$ down the side and 14 across the bottom (cosine-spaced), with the graded interior surface,
which removes the exterior's irregular frequencies. Right: higher order, revolved segments per
quadrant (exact circles; patch edges in black, span edges in grey), spans of at most 0.25, order 3,
clustered at the bottom corner; the waterplane disc is the interior surface.*

## The reference

The reference is the interior series of Faltinsen & Timokha (2009), §5.4.4, as in Newman's appendix
(his eqs. 2–7). It is derived in this repository for the potentials per unit velocity
(`tests/python/problems.py`, `circular_tank`) and summed to 200 terms, which leave less than
$10^{{-6}}$.

The comparison leaves out $\lvert K - K_1 \rvert \le {AWAY}$. There the coefficients pass through
infinity, and a shift of $K_1$ by the discretization (0.003 for the low-order polygon) dominates any
difference.

## Agreement with the series

![Comparison](../figures/{NAME}.png)

*Figure 2. $A_{{11}}$, $A_{{15}}$, $A_{{55}}$ of the tank fluid against $K$ for the four drafts (rows): the
series (circles), low order (solid), higher order (dashed); the grey line marks $K_1$.*

Both methods reproduce the interior series across the resonance's sign changes: $A_{{15}}$ and
$A_{{55}}$ change sign below $K_1$ for the larger drafts, as Newman describes. The low order agrees to
about 1e-2, the discretization of its polygon, and the higher order two orders of magnitude closer
(Table 1).

*Table 1. Largest difference from the series away from $K_1$ ($\lvert K - K_1 \rvert > {AWAY}$),
relative to its largest value there.*

{table(["$d$", "$K_1$", "$A_{11}$ low", "$A_{15}$ low", "$A_{55}$ low", "$A_{11}$ higher", "$A_{15}$ higher",
        "$A_{55}$ higher"], rows)}

## The non-resonant draft

At the draft with $\cosh k_1 d = 2$ the pitch coefficients $A_{{15}}$ and $A_{{55}}$ stay finite at $K_1$,
as Newman's eq. (1) predicts; $A_{{55}}$ is about 0.17 throughout. This checks that the cancellation of
the resonance is reproduced.

There $A_{{15}}$ is small, 0.02 to 0.04. The low order's polygon, whose $k_1$ differs slightly from the
circle's, does not cancel the resonance exactly: it leaves a residual of 0.03 next to $K_1$ and an
offset of 2e-3, {e(nr)} relative to this small value. The higher order on the exact circle leaves
{e(nr_hi)}.

## Free in waves

Newman's section 3 lets the massless tank, with the fluid inside, float freely in head waves of unit
amplitude: surge and pitch both free, or each mode alone. The tank has neither a mass of its own nor a
hydrostatic restoring moment, so its motions follow from the added mass, the damping and the exciting
force alone (theory §3.1 with $M = 0$). HydroBEM solves both modes free as an ordinary motion. One
mode alone is solved here from the same coefficients,
$\xi_i = X_i/(C_{{ii}} - \omega^2 A_{{ii}} + i\omega B_{{ii}})$, the equation of motion of that mode;
the two modes solved from these coefficients reproduce the engine's motion within {tex(check)}.

The reference is Newman's Fig. 2. Its RAOs (the solid lines, on the left axis from 0 to 1) are
**read** from the page image, with the axes calibrated on their tick marks, at $K = 0.5$, 1, 2.5, 3
and 3.5 where a curve is at neither a zero nor a resonance, to about $\pm {FIG_2_READ}$ (more where a
curve is steep). The sloshing amplitudes (the dashed lines, on the right axis) are not read.

![Free in waves](../figures/{NAME}_free.png)

*Figure 3. The RAOs of the tank free in waves against $K$ for the four drafts: both modes free and
each mode alone, higher order (solid), low order (dashed) and the values read from Newman's Fig. 2
(crosses). Near $K_1$ the curves pass through their zeros and the resonances of the tank.*

*Table 2. Largest difference of each RAO from the values read from Newman's Fig. 2, in the units of
the figure (m/m in surge, rad/m in pitch, the radius 1).*

{table(free_header, free_rows)}

Both methods agree with the figure to about the accuracy of reading it at every draft: the largest
difference is {f_hi:.3f} in the higher order and {f_lo:.3f} in the low order. The discretized tank
keeps a small damping that the closed tank does not have (theory §10.2), so the RAOs that vanish at
$K_1$ in the paper are small there but not zero; the points read leave the zeros out.

## Reproducing this page

The example [`examples/{NAME}`](../../../examples/{NAME}) solves the tank of $d = 2$ free in surge and
pitch in waves and checks the zeros of its RAOs against Newman's section 3. This page, its figures and
its numbers are written by `validation/fish_tank_cylinders.py` through `validation/make_report.py`.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
