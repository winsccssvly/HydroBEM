"""The second-order loads (docs/validation/cases/second_order_bottom_cylinder.md and cylinder.md):
the force of the second-order potential
on a bottom-mounted cylinder (h/a = 1) against Kim & Yue (1990), table 3
(tests/data/kim_yue1990.txt, extracted by scripts/extract_kim_yue.py), on three meshes and by both
methods; Newman's approximation against the full difference-frequency QTF (quadratic and potential
parts) of a fixed cylinder; the convergence of the potential QTF in the inner radius R, the
panel size, the panels against the cells, the quadrature zone, the expansion order and the
outer-circle segments; the momentum drift in the Kochin angle count.

    python validation/qtf.py [--quick]
"""

from __future__ import annotations

import math
import sys

import matplotlib.ticker
import mesh_figure
import numpy as np
from common import DATA, REF, RHO, C, G, Timer, e, plt, record, save, solve_model, table

from bodies import _free, surface
from meshes import cylinder_panels
from problems import model

COMPS = (0, 2, 4)  # surge, heave, pitch


def fixed_cylinder(n=12):
    """The truncated cylinder of radius 1 and draft 0.5 (n panels a quarter circle, cosine-spaced)."""
    return {"mesh": surface(cylinder_panels(1.0, 0.5, n_phi=n, n_z=n // 2, n_r=n // 2, quarter=True, cosine=True),
                            symmetry="xz_and_yz"), "modes": []}  # fmt: skip


def floating_cylinder(n=12):
    return _free(fixed_cylinder(n)["mesh"])


def wavenumber(w, h):
    nu = w * w / G
    if not math.isfinite(h):
        return nu
    k = max(nu, math.sqrt(nu / h))
    for _ in range(100):
        t = math.tanh(k * h)
        k -= (k * t - nu) / (t + k * h * (1 - t * t))
    return k


def potential_case(omegas=(2.0, 3.0), depth=math.inf, floating=False, n=12, method="indirect", free_surface="panels",
                   **fs):  # fmt: skip
    """The potential QTF of the cylinder by `method` (direct or indirect): {"sum": [[re, im], ...],
    "difference": ...} over the pairs a <= b and the components surge, heave, pitch (divided by
    rho g), and the free-surface counts; `free_surface` the method of the panelled zone (panels or
    cells), `fs` the other items of the free surface by their short names (problems.flat_surface:
    inner_radius as second_order.free_surface_inner_radius, quadrature_annuli as it is)."""
    body = floating_cylinder(n) if floating else fixed_cylinder(n)
    out = {"diffraction": True, "radiation": floating, "motion": floating, f"qtf_potential_{method}": True}
    m = model(body, list(omegas), [0.0], environment={"density": RHO, "gravity": G, "depth": depth}, outputs=out,
              qtf_free_surface={"method": free_surface, **fs})  # fmt: skip
    counts = {z: int(len(m.free_surface_points(z).weights)) for z in ("panelled", "outer_circle")}
    r = solve_model(m)
    res = {"counts": counts}
    nw = len(omegas)
    for kind in ("sum", "difference"):
        q = r.quantity(f"qtf_potential_{method}_{kind}")[:, :, 0, :] / (RHO * G)
        vals = [q[a, b, c] for a in range(nw) for b in range(a, nw) for c in COMPS
                if not (kind == "difference" and a == b)]  # fmt: skip
        res[kind] = [[float(v.real), float(v.imag)] for v in vals]
    return res


def as_complex(res):
    return {k: np.array([complex(*v) for v in res[k]]) for k in ("sum", "difference")}


def change(a, b):
    """The largest difference of two cases relative to the largest coefficient, per kind."""
    a, b = as_complex(a), as_complex(b)
    return {k: float(np.abs(a[k] - b[k]).max() / np.abs(b[k]).max()) for k in a}


# ------------------------------------------------------------------ Kim & Yue (1990)


def kim_yue_reference():
    """The rows (kind, nu_j a, nu_l a, value) of tests/data/kim_yue1990.txt and their labels."""
    ref = [line.split() for line in (DATA / "kim_yue1990.txt").read_text(encoding="utf-8").splitlines()
           if line and not line.startswith("#")]  # fmt: skip
    return ref, [f"{'diff.' if k == 'difference' else 'sum'} ({a}, {b})" for k, a, b, _ in ref]


def kim_yue(quick):
    ref, labels = kim_yue_reference()
    nus = [1.0, 1.4, 1.6, 2.0]
    ns = [8, 12] if quick else [8, 12, 16]
    got = {}
    with Timer("Kim & Yue cylinder"):
        for n in ns:
            pan = cylinder_panels(1.0, 1.0, n_phi=n, n_z=n, n_r=0, quarter=True, cosine=True)
            out = {"diffraction": True, "qtf_potential_indirect": True, "qtf_potential_direct": True}
            m = model({"mesh": surface(pan, symmetry="xz_and_yz"), "modes": []}, np.sqrt(np.array(nus) * G), [0.0],
                      environment={"density": RHO, "gravity": G, "depth": 1.0}, outputs=out)  # fmt: skip
            if n == 12:
                mesh_figure.draw("second_order_bottom_cylinder", [
                    ("low order, 12 panels a quarter circle, and the free surface to R = 4a", m, False,
                     {"free_surface": True})], elev=35.0, size=(6.0, 4.2))  # fmt: skip
            r = solve_model(m)
            for method in ("indirect", "direct"):
                vals = []
                for kind, a, b, _ in ref:
                    i, j = nus.index(float(a)), nus.index(float(b))
                    q = r.quantity(f"qtf_potential_{method}_{kind}")
                    vals.append(abs(q[i, j, 0, 0]) / (RHO * G))
                got[(n, method)] = np.array(vals)
    refv = np.array([float(x[3]) for x in ref])
    out = {"labels": labels, "reference": refv}
    for (n, method), v in got.items():
        out[f"{method}_n{n}"] = (v - refv) / refv
    plot_kim_yue(out)
    print("  Kim & Yue:", {k: np.round(v, 4) for k, v in out.items() if k.startswith(("indirect", "direct"))})
    return out


def plot_kim_yue(d):
    """The figure qtf_kim_yue from the numbers of kim_yue (also as stored: the signed relative
    differences of each mesh and method)."""
    labels, refv = kim_yue_reference()[1], np.asarray(d["reference"])
    ns = sorted(int(k.split("_n")[1]) for k in d if k.startswith("indirect_n"))
    got = {(n, m): refv * (1.0 + np.asarray(d[f"{m}_n{n}"])) for n in ns for m in ("indirect", "direct")}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    x = np.arange(len(refv))
    ax.plot(x, refv, **{**REF, "ms": 7, "label": "Kim & Yue (1990), semi-analytic"})
    for j, method in enumerate(("indirect", "direct")):
        ax.plot(x + 0.08 * (j + 1), got[(ns[-1], method)], "x" if j == 0 else "+", color=C[j], ms=8,
                ls="none", label=f"HydroBEM {method}, {ns[-1]} panels a quadrant")  # fmt: skip
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"$|f_2|/(\rho g a A_j A_l)$")
    ax.set_xlabel(r"frequency kind and $(\nu_j a, \nu_l a)$")
    ax.legend(fontsize=7)
    ax = axes[1]
    for j, method in enumerate(("indirect", "direct")):
        e = np.array([np.abs(got[(n, method)] - refv) / refv for n in ns])
        for i in range(len(refv)):
            ax.plot(ns, e[:, i], "o-" if j == 0 else "s--", color=C[i], ms=4, mfc=None if j == 0 else "none",
                    label=f"{labels[i]}" if j == 0 else None)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xlabel("panels a quarter circle (and down the side)")
    ax.set_ylabel("|HydroBEM − Kim & Yue| / Kim & Yue")
    ax.set_title("solid: indirect, dashed: direct")
    ax.legend(fontsize=7)
    save(fig, "qtf_kim_yue")


# ------------------------------------------------------------------ Newman's approximation


def newman(quick):
    w = np.round(np.arange(2.0, 3.51, 0.3 if quick else 0.1), 6)
    # Newman's approximation from the pressure-integration drift (switched on for it)
    out = {"diffraction": True, "qtf_difference": True, "qtf_sum": False,
           "qtf_quadratic_pressure_integration": True, "qtf_potential_indirect": True,
           "drift_pressure_integration": True, "qtf_newman": True}  # fmt: skip
    m = model(fixed_cylinder(12), w, [0.0], environment={"density": RHO, "gravity": G}, outputs=out)
    mesh_figure.draw("cylinder", [("low order, 12 panels a quarter circle, and the free surface to R = 4 m", m, False,
                                   {"free_surface": True})], elev=35.0, size=(6.0, 4.2))  # fmt: skip
    with Timer("Newman's approximation"):
        r = solve_model(m)
    full = (r.quantity("qtf_quadratic_pressure_integration_difference")
            + r.quantity("qtf_potential_indirect_difference"))[:, :, 0, :] / (RHO * G)  # fmt: skip
    newm = r.quantity("qtf_newman_pressure_integration_difference")[:, :, 0, :] / (RHO * G)
    quad = r.quantity("qtf_quadratic_pressure_integration_difference")[:, :, 0, :] / (RHO * G)
    steps = [1, 2, 4] if not quick else [1, 2]
    dw = w[1] - w[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    res = {"omegas": w}
    for ax, c, name in ((axes[0], 0, "surge"), (axes[1], 2, "heave")):
        scale = np.abs(full[..., c]).max()
        for j, s in enumerate(steps):
            i = np.arange(len(w) - s)
            mid = 0.5 * (w[i] + w[i + s])
            ax.plot(mid, np.abs(full[i, i + s, c]), color=C[j], label=f"full, Δω = {s * dw:.1f} rad/s")
            ax.plot(mid, np.abs(newm[i, i + s, c]), color=C[j], ls="--", label=f"Newman, Δω = {s * dw:.1f} rad/s")
            res[f"{name}_dw{s * dw:.1f}"] = float(np.abs(full[i, i + s, c] - newm[i, i + s, c]).max() / scale)
            res[f"{name}_dw{s * dw:.1f}_quadratic_only"] = float(np.abs(quad[i, i + s, c] - newm[i, i + s, c]).max()
                                                                  / scale)  # fmt: skip
        ax.plot(w, np.abs(full[np.arange(len(w)), np.arange(len(w)), c]), **{**REF, "label": "diagonal (mean drift)"})
        ax.set_xlabel(r"mean frequency $(\omega_i + \omega_j)/2$ (rad/s)")
        ax.set_ylabel(r"$|T_{ij}|/(\rho g)$ (m)")
        ax.set_title(f"{name}, fixed cylinder, deep water")
    axes[0].legend(fontsize=7)  # one legend for both panels: the heave curves fill the other
    save(fig, "qtf_newman")
    print("  Newman:", {k: round(v, 4) for k, v in res.items() if k != "omegas"})
    return res


# ------------------------------------------------------------------ convergence of the potential QTF


def potential_convergence(quick):
    res = {}
    with Timer("potential QTF: R"):
        radii = [3.0, 4.0, 6.0, 8.0] if quick else [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
        for depth, lab in ((math.inf, "deep"), (3.0, "h = 3 m")):
            cases = [potential_case(depth=depth, inner_radius=R, panel_size=0.1) for R in radii]
            # the (3, 3) sum per component
            s33 = [[abs(x) for x in as_complex(c)["sum"][-3:]] for c in cases]
            res[f"R_{lab}"] = {"radii": radii, "sum": [change(c, cases[-1])["sum"] for c in cases],
                              "difference": [change(c, cases[-1])["difference"] for c in cases],
                              "sum_33_surge_heave_pitch": s33}  # fmt: skip
    with Timer("potential QTF: panel size"):
        sizes = [0.4, 0.3, 0.2, 0.1] if quick else [0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07]
        cases = [potential_case(inner_radius=4.0, panel_size=s) for s in sizes]
        cells = [potential_case(inner_radius=4.0, panel_size=s, free_surface="cells") for s in sizes]
        res["panel_size"] = {
            "sizes": sizes,
            "panels, sums": [change(c, cases[-1])["sum"] for c in cases],
            "panels, differences": [change(c, cases[-1])["difference"] for c in cases],
            "cells against panels, sums": [change(a, b)["sum"] for a, b in zip(cells, cases, strict=True)],
            "points": [c["counts"]["panelled"] for c in cases],
        }
    with Timer("potential QTF: quadrature zone"):
        base = potential_case(inner_radius=6.0, panel_size=0.07)
        nodes = [4, 8, 12] if quick else [2, 4, 6, 8, 12, 16]
        res["quadrature_zone"] = {"nodes": nodes}
        for rin, ann in ((2.0, 2), (3.0, 3)):
            res["quadrature_zone"][f"R = {rin:g} m + {ann} annuli to 6 m"] = [
                change(potential_case(inner_radius=rin, panel_size=0.07, quadrature_annuli=ann,
                                      quadrature_radius_step=(6.0 - rin) / ann, quadrature_radial_nodes=q), base)["sum"]
                for q in nodes]  # fmt: skip
    kmax = wavenumber(6.0, math.inf)
    with Timer("potential QTF: expansion order"):
        margins = [-12, -8, -4, 0, 5, 10, 15, 25, 40]
        base = potential_case(inner_radius=4.0, panel_size=0.15, expansion_order=int(math.ceil(kmax * 4.0)) + 60)
        ys = {"sums": [], "differences": []}
        for mg in margins:  # M below kR: the series truncated inside its band
            c = change(potential_case(inner_radius=4.0, panel_size=0.15,
                                      expansion_order=int(math.ceil(kmax * 4.0)) + mg), base)  # fmt: skip
            ys["sums"].append(c["sum"])
            ys["differences"].append(c["difference"])
        res["expansion_order"] = {"kR": kmax * 4.0, "margins": margins, **ys}
    with Timer("potential QTF: outer-circle segments"):
        auto = potential_case(inner_radius=4.0, panel_size=0.15)
        nauto = auto["counts"]["outer_circle"]
        segs = sorted({4, 8, 12, 16, 24, 32, nauto // 2, nauto, 2 * nauto, 4 * nauto})
        cases = [potential_case(inner_radius=4.0, panel_size=0.15, outer_circle_segments=s) for s in segs]
        res["outer_segments"] = {
            "segments": segs,
            "auto": nauto,
            "sums": [change(c, cases[-1])["sum"] for c in cases],
            "differences": [change(c, cases[-1])["difference"] for c in cases],
        }
    with Timer("momentum drift: Kochin angles"):
        res["kochin"] = kochin(quick)
    plot_convergence(res)
    print("  potential QTF convergence:", res)
    return res


FLOOR = 1e-12  # the changes below it are drawn at it; the reference itself (0) is not drawn


def plot_convergence(res):
    """The figure qtf_convergence from the numbers of potential_convergence (also as stored)."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))
    axes = axes.ravel()

    def plot(ax, x, ys, xlabel, title, auto=None):
        for j, (lab, y) in enumerate(ys.items()):
            y = np.asarray(y, dtype=float)
            keep = y > 0  # the reference itself (0) is not drawn
            ax.plot(np.asarray(x)[keep], np.maximum(y[keep], FLOOR), "o-", color=C[j], ms=4, label=lab)
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        if hi / lo < 100:  # less than two decades: labels at 1, 2 and 5 of each
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=False))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if auto is not None:
            ax.axvline(auto, color="#888888", ls=":", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("largest change / largest coefficient")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)

    ys = {}
    radii = res["R_deep"]["radii"]
    for lab in ("deep", "h = 3 m"):
        ys[f"{lab}, sums"] = res[f"R_{lab}"]["sum"]
        ys[f"{lab}, differences"] = res[f"R_{lab}"]["difference"]
    plot(axes[0], radii, ys, "inner radius R (m)", "R (against R = %g m), panels 0.1 m" % radii[-1], auto=4.0)
    p = res["panel_size"]
    ys = {k: p[k] for k in ("panels, sums", "panels, differences", "cells against panels, sums")}
    plot(axes[1], p["sizes"], ys, "panel size s (m)", f"panel size (against s = {p['sizes'][-1]} m), R = 4 m",
         auto=0.25)  # fmt: skip
    axes[1].invert_xaxis()
    q = res["quadrature_zone"]
    ys = {k: v for k, v in q.items() if k != "nodes"}
    axes[2].set_xticks(q["nodes"])
    plot(axes[2], q["nodes"], ys, "Gauss nodes across an annulus", "quadrature zone (against panels to 6 m)",
         auto=12)  # fmt: skip
    e = res["expansion_order"]
    ys = {"sums": e["sums"], "differences": e["differences"]}
    plot(axes[3], e["margins"], ys, "M − ⌈kR⌉", "expansion order (against + 60), R = 4 m", auto=15)
    o = res["outer_segments"]
    segs = o["segments"]
    plot(axes[4], segs, {"sums": o["sums"], "differences": o["differences"]}, "outer-circle segments",
         f"outer circle (against {segs[-1]})", auto=o["auto"])  # fmt: skip
    axes[4].set_xscale("log")
    axes[4].set_xticks(segs[::2], [str(v) for v in segs[::2]])
    axes[4].minorticks_off()
    k = res["kochin"]
    ax = axes[5]
    ax.semilogy(k["counts"], np.maximum(k["errors"], FLOOR), "o-", color=C[0], ms=4, label="surge, sway, yaw")
    ax.semilogy([k["auto"]], [max(k["auto_error"], FLOOR)], "*", color=C[1], ms=10, label=f"auto (N = {k['auto']})")
    ax.set_xscale("log")
    ax.set_xlabel("Kochin angles N")
    ax.set_ylabel("largest change / largest drift")
    ax.set_title("momentum drift (against N = 4096)", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "qtf_convergence")


def kochin(quick):
    """The drift by momentum conservation of the floating cylinder against the Kochin angle count."""
    w = np.linspace(0.6, 4.0, 6 if quick else 12)
    counts = [8, 16, 24, 32, 48, 64, 128, 512]

    def drift(n):
        out = {"diffraction": True, "radiation": True, "motion": True, "drift_momentum_conservation": True,
               "drift_momentum_conservation_angles": n}  # fmt: skip
        m = model(floating_cylinder(12), w, [0.0, 30.0], environment={"density": RHO, "gravity": G, "depth": math.inf},
                  outputs=out)  # fmt: skip
        r = solve_model(m)
        return r.quantity("drift_momentum_conservation")

    ref = drift(4096)
    auto = drift("auto")
    errs = [float(np.abs(drift(n) - ref).max() / np.abs(ref).max()) for n in counts]
    e_auto = float(np.abs(auto - ref).max() / np.abs(ref).max())
    kR = (4.0**2 / G) * 1.0  # the largest k times R = 1 m
    n_auto = 8 * math.ceil((2 * math.ceil(kR) + 96) / 8)
    return {"counts": counts, "errors": errs, "auto": n_auto, "auto_error": e_auto}


def replot(numbers):
    """Redraws the figures whose numbers numbers.json holds (qtf_kim_yue, qtf_convergence)."""
    plot_kim_yue(numbers["qtf_kim_yue"])
    plot_convergence(numbers["qtf_convergence"])


def main(quick=False):
    record("qtf_kim_yue", kim_yue(quick))
    record("qtf_newman", newman(quick))
    record("qtf_convergence", potential_convergence(quick))


def kim_yue_worst(d, n_min=0):
    """The largest |relative difference| of both methods over the meshes from n_min panels on."""
    return max(abs(x) for k in d if k.startswith(("indirect_n", "direct_n")) and int(k.split("_n")[1]) >= n_min
               for x in d[k])  # fmt: skip


def summary_kim_yue(n):
    d = n["qtf_kim_yue"]
    ns = sorted(int(k.split("_n")[1]) for k in d if k.startswith("indirect_n"))
    return (f"the indirect and the direct method agree with the published values within "
            f"{100 * kim_yue_worst(d, 12):.1f} % from {ns[1]} panels on (low order)")  # fmt: skip


def page_kim_yue(n):
    d = n["qtf_kim_yue"]
    labels = d["labels"]
    ns = sorted(int(k.split("_n")[1]) for k in d if k.startswith("indirect_n"))
    rows = [[str(k), m] + [f"{100 * x:+.2f} %" for x in d[f"{m}_n{k}"]] for k in ns for m in ("indirect", "direct")]
    worst = kim_yue_worst(d)
    fine = kim_yue_worst(d, ns[1])
    return rf"""
# Second-order force on a bottom-mounted cylinder

**Result.** The force of the second-order potential on a fixed bottom-mounted cylinder agrees with the
semi-analytic values of Kim & Yue (1990) within {100 * fine:.1f} % from {ns[1]} panels per quarter
circle on, by both the indirect and the direct method, and within {100 * worst:.1f} % on every mesh
compared. This is the size of the free surface's own discretization at its automatic panel size.

## The problem

In bichromatic waves, waves of two frequencies together, a body feels second-order forces at the sum
and the difference of the two frequencies; the quadratic transfer function (QTF) gives them per pair
of frequencies. Part of that force comes from quadratic products of the first-order solution, and part
from the second-order potential, which those products leave out. This page checks the second part.

A vertical circular cylinder of radius $a = 1$ m stands fixed on the sea bed in water of depth $h = a$,
in head seas. The first-order frequencies are $\nu a = 1.0$, 1.4, 1.6, 2.0, with $\nu = \omega^2/g$.
We compare the modulus of the surge force of the second-order potential,
$\lvert f_2 \rvert/(\rho g a A_j A_l)$, at the pairs $(\nu_j a, \nu_l a) = (1.0, 2.0)$ and
$(1.4, 1.6)$, at the difference and the sum frequencies.

HydroBEM computes it in two ways, both in one solve: the indirect method (Green's identity with an
assisting radiation potential) and the direct method (the second-order potential solved). The
potential QTF is available in the low order only (theory §12.10), so there is no higher-order
result.

![Mesh](../figures/mesh_second_order_bottom_cylinder.png)

*Figure 1. The low-order mesh, expanded to the whole body and seen from above: the cylinder's wetted
wall in blue and the free surface of the potential QTF in grey. The cylinder has $n$ panels per
quarter circle and $n$ down the side (cosine-spaced; shown $n = 12$, the comparison uses
$n = {", ".join(map(str, ns))}$) and no bottom, since it stands on the sea bed. The engine generates the
free surface from the waterline to $R = 4a$ with the automatic panel size; the quadrature zone and the
far field beyond are automatic.*

## The reference

The reference is Kim & Yue (1990), table 3: their semi-analytic values, given to three decimals and
extracted into `tests/data/kim_yue1990.txt` by `scripts/extract_kim_yue.py`.

## Agreement with the table

![Kim and Yue](../figures/qtf_kim_yue.png)

*Figure 2. Left: $\lvert f_2 \rvert$ of the finest mesh and both methods against the published values.
Right: the relative difference against $n$ (solid: indirect, dashed: direct).*

Every value is within {100 * worst:.1f} % (Table 1).

*Table 1. Signed relative difference from Kim & Yue (1990).*

{table(["$n$", "method"] + labels, rows)}

## Does it converge?

The differences do not fall further with the body mesh. The published values have three decimals, and
the free surface keeps its automatic panel size, whose discretization, a few tenths of a per cent
([cylinder](cylinder.md)), is of the same size as the differences.

## Reproducing this page

The example [`examples/second_order_bottom_cylinder`](../../../examples/second_order_bottom_cylinder)
runs the case. This page, its figures and its numbers are written by `validation/qtf.py` through
`validation/make_report.py`.
"""


def summary_cylinder(n):
    d = n["qtf_newman"]
    return (f"Newman's approximation within {100 * d['surge_dw0.1']:.0f} % of the full QTF at "
            "$\\Delta\\omega = 0.1$ rad/s; the free-surface parameters converged (no independent reference)")


TABLE6_HEADER = ['Parameter', 'Range', 'Largest change']
TABLE6 = [
    [
        'inner radius $R$ (panels 0.1 m), deep water',
        '3 to 10 m against 12 m',
        'sums 7.7e-3, differences 3.3e-3; heave of the (3, 3) sum 1.0614 to 1.0588 (0.25 %)',
    ],
    [
        'inner radius $R$, depth 3 m',
        '3 to 10 m against 12 m',
        'sums 9.4e-3, differences 6.4e-4; heave of the (3, 3) sum 1.0951 at $R = 3$ m, 1.0847 to 1.0868 '
        'from 4 to 12 m (0.2 %)',
    ],
    [
        'panel size $s$ ($R = 4$ m)',
        '0.4 to 0.1 m against 0.07 m',
        'sums 4.7e-2 (0.3 m), 2.9e-2 (0.25 m, the automatic size), 3.1e-3 (0.1 m)',
    ],
    [
        'cells against panels',
        'the same sizes',
        '4.6e-2 at 0.3 m, 3.5e-3 at 0.1 m, 1.5e-3 at 0.07 m',
    ],
    [
        'quadrature zone, Gauss nodes across an annulus (panels 0.07 m)',
        '2 to 16; panels to 2 or 3 m, annuli to 6 m; against panels to 6 m',
        'converged from 6 to 8 nodes; then 6.1e-3 (from 2 m) and 3.2e-3 (from 3 m)',
    ],
    [
        r'expansion order $M - \lceil kR \rceil$ ($kR = 14.7$)',
        '$-12$ to $+40$ against $+60$',
        r'6.3e-3 at $-12$; $\le$ 5e-8 from $-8$ on',
    ],
    [
        'outer-circle segments',
        '4 to 256 against 512 (automatic 128)',
        r'$\le$ 1.4e-7',
    ],
]


def page_cylinder(n):
    d = n["qtf_newman"]
    rows = [[f"{dw}", pct_(d[f"surge_dw{dw}"]), pct_(d[f"surge_dw{dw}_quadratic_only"]), pct_(d[f"heave_dw{dw}"]),
             pct_(d[f"heave_dw{dw}_quadratic_only"])]
            for dw in ("0.1", "0.2", "0.4") if f"surge_dw{dw}" in d]  # fmt: skip
    c = n["qtf_convergence"]
    k = c["kochin"]
    kochin = ["Kochin angles $N$ (momentum drift)", f"{k['counts'][0]} to {k['counts'][-1]} against 4096",
              f"{e(k['errors'][0])} at {k['counts'][0]}; $\\le$ {e(max(k['errors'][1:]))} from {k['counts'][1]} on "
              f"(automatic {k['auto']}: {e(k['auto_error'])})"]  # fmt: skip
    TABLE6_TABLE = table(TABLE6_HEADER, TABLE6 + [kochin])  # noqa: N806
    header = [r"$\Delta\omega$ (rad/s)", "surge, full", "surge, quadratic part", "heave, full", "heave, quadratic part"]
    dws = [dw for dw in ("0.1", "0.2", "0.4") if f"surge_dw{dw}" in d]
    return rf"""
# Truncated cylinder in deep water: second-order loads

**Result.** No independent reference exists for this body, so the page checks the second-order free
surface by convergence. Newman's approximation differs from the full difference-frequency QTF by
{pct_(d[f"surge_dw{dws[0]}"])} in surge and {pct_(d[f"heave_dw{dws[0]}"])} in heave at
$\Delta\omega = {dws[0]}$ rad/s, growing linearly with $\Delta\omega$; in heave most of it is the
potential part, which the approximation leaves out. The force of the second-order potential depends
on the free-surface discretization by a few tenths of a per cent of the largest coefficient at the
automatic settings, and every series and quadrature beyond the inner radius is converged far below
that.

## The problem

The body is the truncated cylinder of radius 1 m and draft 0.5 m in deep water, the body of the
example [`examples/cylinder`](../../../examples/cylinder), in head seas, fixed (floating where stated).
Without an independent reference, the checks of this page are the grounds of the second-order free
surface ([not verified](../not_verified.md)). Three ranges are used:

- $\omega = 2.0$ to 3.5 rad/s in steps of 0.1 rad/s for Newman's approximation;
- $\omega = 2$ and 3 rad/s for the convergence of the potential QTF (surge, heave and pitch, sums and
  differences);
- 0.6 to 4 rad/s for the momentum drift.

The quadratic transfer function (QTF) gives the second-order force per pair of wave frequencies. Its
quadratic part comes from products of the first-order solution; its potential part, the force of the
second-order potential, needs the free surface around the body to be discretized as well.

![Mesh](../figures/mesh_cylinder.png)

*Figure 1. The low-order mesh, expanded to the whole body and seen from below: the cylinder's wetted
surface in blue and the free surface of the potential QTF in grey. The cylinder has 12 panels per
quarter circle, 6 down the side and 6 across the bottom (cosine-spaced); the free surface is
generated to the automatic $R = 4$ m with the automatic panel size. The potential QTF is available in
the low order only.*

## Newman's approximation

Newman's approximation estimates the difference-frequency QTF near its diagonal from the mean drift
forces alone, without the second-order potential. This section measures how far it is from the full
QTF.

![Newman's approximation](../figures/qtf_newman.png)

*Figure 2. The modulus of the full difference-frequency QTF (the quadratic part by pressure
integration plus the force of the second-order potential, indirect method) along the off-diagonals
$\Delta\omega = \omega_j - \omega_i$ (solid), Newman's approximation (the geometric mean of the
pressure drifts, dashed) and the diagonal, the mean drift (markers).*

The difference grows linearly with $\Delta\omega$, as an approximation about the diagonal should
(Table 1). In heave the potential part, which the approximation does not contain, is most of it.

*Table 1. Largest difference between Newman's approximation and the QTF along each off-diagonal,
relative to the largest modulus of the full QTF.*

{table(header, rows)}

## Does the second-order free surface converge?

The force of the second-order potential depends on how the free surface is discretized: an inner
radius $R$ with panels or cells, a quadrature zone of annuli beyond it, and a far field summed as a
series on an outer circle. Each parameter is varied about a reference value while the others are held
(Figure 3, Table 2). The momentum drift depends on one parameter of its own, the number of angles at
which the Kochin functions, the far-field amplitudes of the waves around the body, are evaluated.

![Potential QTF convergence](../figures/qtf_convergence.png)

*Figure 3. $\omega = 2$ and 3 rad/s, indirect method: the largest change of the sum and difference
coefficients (surge, heave, pitch) relative to the largest coefficient, when one parameter of the free
surface (theory §12.8–§12.9, reference/inputs-orders §3) is varied about a reference value; the dotted line marks the
automatic value. Last panel: the momentum drift of the floating cylinder ($0.6 \le \omega \le 4$
rad/s, headings $0^\circ$ and $30^\circ$) against the Kochin angle count. Changes below 1e-12 are drawn
at 1e-12; the reference case itself (no change) is not drawn.*

*Table 2. The convergence of the potential QTF and of the momentum drift.*

{TABLE6_TABLE}

The inner radius $R$: past the body's near field the coefficients change by less than 1 % of the
largest and the differences by less than 0.35 %. The changes are not monotone in $R$, because the
generated panels differ with $R$, and their discretization (0.3 % between $s = 0.1$ and 0.07 m) is of
the same size.

The panel size: the sums oscillate on the free surface with the combined wavenumber $2k + k_\Omega$ of
the pair, here 5.5 /m, a combined wavelength of 1.14 m. The automatic size, $a/4 = 0.25$ m (a third of
the combined wavelength is 0.38 m), leaves 2.9 % in the sums and 0.6 % in the differences against
0.07 m; 0.1 m, about 11 panels per combined wavelength, leaves 0.3 %. The cells and the generated
panels agree to the same discretization error.

The quadrature zone converges in its radial nodes from 6 to 8. It differs from panels to the same
radius by 0.3 % (from 3 m) and 0.6 % (from 2 m, closer to the body), the size of the panels' own
discretization (0.2 % between 0.1 and 0.07 m at $R = 6$ m); the azimuthal nodes change nothing.

The expansion order and the outer-circle segments: the far-field series and the trapezoidal rule on
the circle are converged far below every other error. The automatic order (+15) has a margin of more
than 20 orders over the first one that changes the results, and the automatic segment count a factor
of ten.

The Kochin angle count: $N = 16$ is converged for this body ($kR \le 1.6$), as the rule
$N \ge 2kR + \text{{about }} 10(kR)^{{1/3}}$ of theory §11.4 predicts; the automatic count adds the
margin 96.

## Reproducing this page

The example [`examples/cylinder`](../../../examples/cylinder) solves the same body. This page, its
figures and its numbers are written by `validation/qtf.py` through `validation/make_report.py`.
"""


def pct_(x):
    return f"{100 * float(x):.2g} %"


PAGES = {
    "second_order_bottom_cylinder": ({"case": "Second-order force, bottom-mounted cylinder",
                                      "checked": "force of the second-order potential, sum and difference",
                                      "sources": "Kim & Yue (1990) table 3", "methods": "low"},
                                     page_kim_yue, summary_kim_yue, ("qtf_kim_yue",)),
    "cylinder": ({"case": "Truncated cylinder, second order", "checked": "Newman's approximation; convergence of "
                  "the potential QTF and the momentum drift in their parameters", "sources": "convergence only",
                  "methods": "low"}, page_cylinder, summary_cylinder, ("qtf_newman", "qtf_convergence")),
}  # fmt: skip


if __name__ == "__main__":
    main("--quick" in sys.argv)
