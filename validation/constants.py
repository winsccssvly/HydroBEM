"""The sensitivity of the fixed constants of the method (docs/validation/cases/constants.md): each
constant varied about its value, the largest change of the results relative to their largest value.
The constants of inputs (the higher-order discretization from a panel size, body_point_neighbors) are
varied through them;
the constants of the second-order free surface through the engine's validation hooks, the
environment variables HB_VALIDATE_<name> (crates/hydrobem/src/env_hooks.rs, read once per
process: one process per value):

    FIT_RADIUS_RATIO 1.1, FIT_LOCAL_ORDERS 30, FIT_DEPENDENCE 0.05 (theory §12.9, the fit);
    STENCIL_WAVENUMBER 0.02, STENCIL_RADIUS 1e-3, WATERLINE_OFFSET 1e-6 (theory §12.8);
    OUTER_WAVELENGTHS 4, OUTER_RADII 10, OUTER_TAPER 0.3, OUTER_PANEL 0.25 (theory §12.9, the outer integral);
    FS_GAP_FACTOR 0.45 (theory §12.8, several bodies); WATERLINE_GAUSS 6 (theory §11.4, the higher-order
    pressure drift).

    python validation/constants.py [--quick]
"""

from __future__ import annotations

import sys

import matplotlib.ticker
import numpy as np
import qtf  # noqa: F401  (the cases run in the processes of run_isolated)
from common import RHO, C, G, Timer, e, plt, record, run_isolated, save, solve_model, table

from bodies import _free, hemi, spline
from meshes import nurbs_hemisphere_patches, spline_cylinder_patches
from problems import model

# the legend names of the hooked constants (plain words; the values are those of the method)
NAMES = {
    "FIT_RADIUS_RATIO": "second circle of the fit (1.1 R)",
    "FIT_DEPENDENCE": "dependence cutoff (0.05)",
    "STENCIL": "stencil step δ",
    "WATERLINE_OFFSET": "waterline offset, direct (1e-6 R)",
    "OUTER_WAVELENGTHS": "length in wavelengths (4)",
    "OUTER_RADII": "length in radii (10 R)",
    "OUTER_TAPER": "taper (30 %)",
    "OUTER_PANEL": "Gauss panel width (¼ of 2π/(3kₛ))",
    "FS_GAP_FACTOR": "box widening (0.45 of the gap)",
}


def sweep(name, values, base, function="potential_case", module="qtf", **kwargs):
    """The relative change (sums, differences) of the potential QTF for every value of the hook
    against the method's value."""
    ref = run_isolated(module, function, {f"HB_VALIDATE_{name}": base}, **kwargs)
    out = []
    for v in values:
        r = run_isolated(module, function, {f"HB_VALIDATE_{name}": v}, **kwargs)
        out.append(qtf.change(r, ref))
    return {"values": values, "base": base, "sum": [c["sum"] for c in out],
            "difference": [c["difference"] for c in out]}  # fmt: skip


def two_bodies_case(**fs):
    """Two fixed cylinders (radius 1, draft 0.5) at y = +-1.6 m (a gap of 1.2 m between the boxes of
    their waterlines), the cells about each body's box (several bodies, theory §12.8)."""
    bodies = [{**qtf.fixed_cylinder(12), "name": "a", "position": [0.0, 1.6, 0.0]},
              {**qtf.fixed_cylinder(12), "name": "b", "position": [0.0, -1.6, 0.0]}]  # fmt: skip
    out = {"diffraction": True, "qtf_potential_indirect": True}
    m = model(bodies, [1.5, 2.0], [0.0], environment={"density": RHO, "gravity": G}, outputs=out,
              qtf_free_surface={"method": "cells", "inner_radius": 5.0, "panel_size": 0.25, **fs})  # fmt: skip
    r = solve_model(m)
    res = {}
    for kind in ("sum", "difference"):
        q = r.quantity(f"qtf_potential_indirect_{kind}")[:, :, 0, :] / (RHO * G)
        vals = [
            q[a, b, c]
            for a in range(2)
            for b in range(a, 2)
            for c in range(12)
            if not (kind == "difference" and a == b)
        ]
        res[kind] = [[float(v.real), float(v.imag)] for v in vals]
    return res


def ho_drift_case():
    """The pressure drift (surge, heave, pitch) of the floating spline cylinder in the higher order,
    0.6 to 3 rad/s, head seas, 8 spans a quarter circle."""
    mesh = spline(spline_cylinder_patches(1.0, 0.5, spans=4, order=4), (8, 4, 3, 3, 3, 4), symmetry="xz_and_yz")
    out = {"diffraction": True, "radiation": True, "motion": True, "drift_pressure_integration": True}
    w = np.linspace(0.6, 3.0, 9)
    m = model(_free(mesh), w, [0.0], environment={"density": RHO, "gravity": G}, options={"higher_order": True},
              outputs=out)  # fmt: skip
    r = solve_model(m)
    d = r.quantity("drift_pressure_integration")[:, 0, :][:, [0, 2, 4]].real / (RHO * G)
    return {"sum": [[float(x), 0.0] for x in d.ravel()], "difference": [[1.0, 0.0]]}


def potential_hooks(quick):
    studies = {}
    deep = {}
    with Timer("constants: the fit"):
        studies["FIT_RADIUS_RATIO"] = sweep("FIT_RADIUS_RATIO", [1.03, 1.05, 1.2, 1.4], 1.1, **deep)
        studies["FIT_LOCAL_ORDERS"] = sweep("FIT_LOCAL_ORDERS", [-1, 0, 2, 5, 10, 20], 30, **deep)
        studies["FIT_LOCAL_ORDERS_depth3"] = sweep("FIT_LOCAL_ORDERS", [-1, 0, 2, 5, 10, 20], 30, depth=3.0)
        studies["FIT_DEPENDENCE"] = sweep("FIT_DEPENDENCE", [0.005, 0.01, 0.02, 0.1, 0.2], 0.05, **deep)
    with Timer("constants: the stencil and the waterline points"):
        # the step min(0.02/k_s, 1e-3 R): both factors scaled together
        f = [0.03, 0.1, 0.3, 3.0, 10.0]
        ref = run_isolated("qtf", "potential_case", {}, floating=True)
        ch = [qtf.change(run_isolated("qtf", "potential_case", {"HB_VALIDATE_STENCIL_WAVENUMBER": 0.02 * x,
                                                              "HB_VALIDATE_STENCIL_RADIUS": 1e-3 * x},
                                       floating=True), ref) for x in f]  # fmt: skip
        studies["STENCIL"] = {"values": f, "base": 1.0, "sum": [c["sum"] for c in ch],
                              "difference": [c["difference"] for c in ch]}  # fmt: skip
        studies["WATERLINE_OFFSET"] = sweep("WATERLINE_OFFSET", [1e-9, 1e-8, 1e-7, 1e-5, 1e-4, 1e-3], 1e-6,
                                            method="direct", floating=True)  # fmt: skip
    with Timer("constants: the outer integral"):
        studies["OUTER_WAVELENGTHS"] = sweep("OUTER_WAVELENGTHS", [1.0, 2.0, 3.0, 6.0, 8.0], 4.0, **deep)
        studies["OUTER_RADII"] = sweep("OUTER_RADII", [2.0, 5.0, 20.0], 10.0, omegas=[3.0, 3.5])
        studies["OUTER_TAPER"] = sweep("OUTER_TAPER", [0.05, 0.1, 0.2, 0.5, 0.7], 0.3, **deep)
        studies["OUTER_PANEL"] = sweep("OUTER_PANEL", [0.0625, 0.125, 0.5, 1.0], 0.25, **deep)
    with Timer("constants: several bodies"):
        studies["FS_GAP_FACTOR"] = sweep("FS_GAP_FACTOR", [0.2, 0.3, 0.4, 0.49], 0.45, function="two_bodies_case",
                                         module="constants")  # fmt: skip
    return studies


def sweep_drift(values, base):
    ref = run_isolated("constants", "ho_drift_case", {"HB_VALIDATE_WATERLINE_GAUSS": base})
    out = [qtf.change(run_isolated("constants", "ho_drift_case", {"HB_VALIDATE_WATERLINE_GAUSS": v}), ref)["sum"]
           for v in values]  # fmt: skip
    return {"values": values, "base": base, "sum": out}


def gauss_on_waterline():
    """The 6-point rule on the integrand of the waterline term of a curved spline side: the square
    of a cubic B-spline trace times the arc-length element of the quarter circle of
    spline_cylinder_patches, against 64 points, per span count."""
    from scipy.interpolate import BSpline

    patch = spline_cylinder_patches(1.0, 0.5, spans=4, order=4)[0]
    t = np.asarray(patch[2])
    cx, cy = patch[4][0, :, 0], patch[4][0, :, 1]
    geo = (BSpline(t, cx, 3), BSpline(t, cy, 3))
    rng = np.random.default_rng(3)
    out = {}
    for spans in (2, 4, 8):
        tk = np.r_[[0.0] * 3, np.linspace(0, 1, spans + 1)[1:-1], [1.0] * 3]  # quadratic potential spline
        coef = rng.normal(size=len(tk) - 3) + 1j * rng.normal(size=len(tk) - 3)
        phi = (BSpline(tk, coef.real, 2), BSpline(tk, coef.imag, 2))

        def f(u, phi=phi):
            dl = np.hypot(geo[0].derivative()(u), geo[1].derivative()(u))
            return (phi[0](u) ** 2 + phi[1](u) ** 2) * dl

        def rule(n, spans=spans, f=f):
            x, w = np.polynomial.legendre.leggauss(n)
            total = 0.0
            for k in range(spans):
                a, b = k / spans, (k + 1) / spans
                total += 0.5 * (b - a) * np.sum(w * f(0.5 * (a + b) + 0.5 * (b - a) * x))
            return total

        exact = rule(64)
        out[spans] = {n: abs(rule(n) - exact) / abs(exact) for n in (2, 3, 4, 6, 8)}
    return out


def body_point_neighbors(quick):
    """The diffraction pressure at points of the hemisphere (not centroids) from the low order with
    k nearest centroids against the higher order's pressure at the same points."""
    rng = np.random.default_rng(7)
    # points on the wetted quadrant x, y > 0, z < 0 (polar angle from +z beyond pi/2)
    th, ph = rng.uniform(0.05, 0.5 * np.pi - 0.05, 40), rng.uniform(0.5 * np.pi + 0.05, np.pi - 0.05, 40)
    pts = np.c_[np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)]
    w = np.sqrt(np.array([0.5, 1.0, 2.0]) * G)
    out = {"diffraction": True, "body_pressure": True, "radiation": False}
    ho = solve_model(model({"mesh": spline(nurbs_hemisphere_patches(1.0), (8, 8, 3, 3, 3, 4), symmetry="xz_and_yz"),
                            "modes": [], "body_points": pts}, w, [0.0], options={"higher_order": True},
                           outputs=out, environment={"density": RHO, "gravity": G}))  # fmt: skip
    ref = ho.quantity("body_points_pressure_diffraction")
    ks = [1, 2, 3, 4, 6, 8, 12]
    res = {"k": ks}
    for n in [12, 24] if quick else [12, 24, 48]:
        e = []
        for k in ks:
            m = model({"mesh": hemi(n), "modes": [], "body_points": pts, "body_point_neighbors": k}, w, [0.0],
                      outputs=out, environment={"density": RHO, "gravity": G})  # fmt: skip
            r = solve_model(m)
            e.append(float(np.abs(r.quantity("body_points_pressure_diffraction") - ref).max() / np.abs(ref).max()))
        res[f"n{n}"] = e
    return res


def discretization_defaults(quick):
    """The hemisphere with the spans of the panel_size rule, spline orders 2..5 and Gauss orders: the largest
    difference of A, B (heave, surge) from Hulme (1982) at Ka = 0.5, 1, 2 and the solve time."""
    import time

    import hemisphere as hm

    ref = hm.hulme()
    sel = np.isin(ref[:, 0], [0.5, 1.0, 2.0])
    ka, r = ref[sel, 0], ref[sel, 1:]
    res = []
    sizes = [1.6, 0.8, 0.4] if quick else [1.6, 0.8, 0.4, 0.2]
    for k, qo, qi in ((2, 2, 3), (2, 3, 4), (3, 2, 3), (3, 3, 4), (3, 4, 5), (4, 3, 4), (4, 4, 5), (5, 5, 6)):
        for s in sizes:
            # the spans of the panel_size rule: both patch edges are quarter circles of length pi/2
            n = int(np.floor(0.5 * np.pi / s + 1e-5)) + 1
            m = spline(nurbs_hemisphere_patches(1.0), (n, n, k, k, qo, qi), symmetry="xz_and_yz")
            t = time.time()
            v = hm.coefficients({"mesh": m}, ka, higher=True)
            res.append({"order": k, "outer": qo, "inner": qi, "size": s, "error": float(hm.errors(v, r).max()),
                        "unknowns": (n + k - 1) ** 2, "seconds": time.time() - t})  # fmt: skip
    return res


def main(quick=False):
    studies = potential_hooks(quick)
    with Timer("constants: waterline Gauss points"):
        drift = sweep_drift([1, 2, 3, 4, 8, 12], 6)
        gauss = gauss_on_waterline()
    with Timer("constants: body_point_neighbors"):
        bpn = body_point_neighbors(quick)
    with Timer("constants: panel_size defaults"):
        disc = discretization_defaults(quick)
    out = {"hooks": studies, "waterline_gauss_drift": drift, "waterline_gauss_integrand": gauss,
           "body_point_neighbors": bpn, "panel_size_defaults": disc}  # fmt: skip
    plot(out)
    print("  constants:", out)
    record("constants", out)


FLOOR = 1e-9  # the changes below it are drawn at it (the drift panel at 1e-12)


def plot(numbers):
    """The figures constants_second_order and constants_inputs from the numbers of main (also as
    stored)."""
    studies, drift = numbers["hooks"], numbers["waterline_gauss_drift"]
    bpn, disc = numbers["body_point_neighbors"], numbers["panel_size_defaults"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    axes = axes.ravel()

    def rel_plot(ax, keys, title, xlabel, log_x=True):
        for j, key in enumerate(keys):
            s = studies[key] if key in studies else None
            x = np.array(s["values"], dtype=float) / s["base"]
            y = np.maximum(np.maximum(s["sum"], s["difference"]), FLOOR)
            ax.plot(x, y, "o-", color=C[j], ms=4, label=NAMES[key])
        ax.axvline(1.0, color="#888888", ls=":", lw=1)
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("largest change / largest coefficient")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)

    rel_plot(axes[0], ["FIT_RADIUS_RATIO", "FIT_DEPENDENCE"], "far-field fit", "value / method's value")
    ax = axes[1]
    for j, key in enumerate(["FIT_LOCAL_ORDERS", "FIT_LOCAL_ORDERS_depth3"]):
        s = studies[key]
        ax.semilogy(s["values"], np.maximum(np.maximum(s["sum"], s["difference"]), FLOOR), "o-", color=C[j], ms=4,
                    label="deep" if j == 0 else "h = 3 m")  # fmt: skip
    ax.set_xlabel("highest order with a local term (−1: none)")
    ax.set_ylabel("largest change / largest coefficient")
    ax.set_title("local orders (against 30)", fontsize=9)
    ax.legend(fontsize=7)
    rel_plot(axes[2], ["STENCIL", "WATERLINE_OFFSET"], "stencil step, waterline offset (floating)",
             "value / method's value")  # fmt: skip
    rel_plot(axes[3], ["OUTER_WAVELENGTHS", "OUTER_RADII", "OUTER_TAPER", "OUTER_PANEL"], "outer integral",
             "value / method's value")  # fmt: skip
    rel_plot(axes[4], ["FS_GAP_FACTOR"], "box widening of several bodies", "value / method's value", log_x=False)
    ax = axes[5]
    ax.semilogy(drift["values"], np.maximum(drift["sum"], 1e-12), "o-", color=C[0], ms=4,
                label="pressure drift (higher order)")  # fmt: skip
    ax.axvline(6, color="#888888", ls=":", lw=1)
    ax.set_xlabel("Gauss points per waterline panel side")
    ax.set_ylabel("largest change / largest drift")
    ax.set_title("waterline rule (against 6)", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "constants_second_order")
    # the inputs: body_point_neighbors and the panel_size defaults
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    ax = axes[0]
    for j, key in enumerate(k for k in bpn if k.startswith("n")):
        ax.semilogy(bpn["k"], bpn[key], "o-", color=C[j], ms=4, label=f"{key[1:]} panels a quadrant edge")
    ax.axvline(4, color="#888888", ls=":", lw=1)
    ax.set_xlabel("body_point_neighbors k")
    ax.set_ylabel("max |p − p(higher order)| / max |p|")
    ax.set_title("low-order pressure at body points", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    combos = sorted({(d["order"], d["outer"], d["inner"]) for d in disc})
    # the colour by the spline order, the line and marker by the outer Gauss order against it
    style = {-1: ("--", "s"), 0: ("-", "o"), 1: (":", "^")}
    for cmb in combos:
        rows = [d for d in disc if (d["order"], d["outer"], d["inner"]) == cmb]
        label = f"order {cmb[0]}, Gauss {cmb[1]}/{cmb[2]}" + (" (default)" if cmb == (3, 3, 4) else "")
        ls, mk = style[cmb[1] - cmb[0]]
        ax.loglog([d["unknowns"] for d in rows], [d["error"] for d in rows], marker=mk, color=C[cmb[0] - 2],
                  ms=4, ls=ls, lw=2.2 if cmb == (3, 3, 4) else 1.4, label=label)  # fmt: skip
    ax.set_xlabel("unknowns a quadrant (panel sizes 1.6 to 0.2 m)")
    ax.axhline(5e-5 / 0.86, color="#888888", lw=0.8, ls="--")
    ax.set_ylabel("max |HydroBEM − Hulme| / max |Hulme|")
    ax.set_title("higher order from panel_size: orders and Gauss rules", fontsize=9)
    ax.legend(fontsize=6)
    save(fig, "constants_inputs")


def replot(numbers):
    """Redraws the figures whose numbers numbers.json holds (both of this group)."""
    plot(numbers["constants"])


NAME = "constants"
GROUPS = ("constants",)
INDEX = {"case": "Fixed constants of the method", "checked": "sensitivity of every constant without an input",
         "sources": "variation about the value; arguments", "methods": "low, higher"}  # fmt: skip


def summary(n):
    return (f"each of the {len(TABLE7)} constants lies in a flat range of its sensitivity; the default "
            f"higher-order discretization reaches its best agreement with Hulme's tables at "
            f"{default_floor(n['constants'])[1]} unknowns per quadrant")


TABLE7_HEADER = ['Constant (where)', 'Value', 'Varied over', 'Largest change', 'Ground']
TABLE7 = [
    [
        'second circle of the fit (theory §12.9)',
        '$1.1 R$',
        '1.03 to $1.4 R$',
        '1.9e-4 to 9.1e-4 for 1.05 to $1.4 R$; 5.7e-3 at $1.03 R$',
        r'the $2 \times 2$ fit of each order loses its conditioning as the circles close and the local term '
        'decays between them more slowly as they part; 1.1 lies in the flat range',
    ],
    [
        'orders with a local term (theory §12.9)',
        '30',
        'none, 0 to 20',
        '3.0e-2 without, 2.5e-3 with order 0 only, 1.1e-5 to 2, about 1e-9 and below from 5 (deep water '
        'and depth 3 m alike)',
        r'the local part of order $m$ decays as $r^{-3-\lvert m \rvert}$: above a few orders it is below the '
        'float32 fields; 30 is ample',
    ],
    [
        'dependence cutoff of the fit (theory §12.9)',
        '0.05',
        '0.005 to 0.2',
        'none up to 0.1; 6.7e-3 at 0.2',
        'it only drops the local term of orders whose two radial functions are nearly dependent between '
        'the circles; inactive here from 0.005 to 0.1',
    ],
    [
        r'stencil step $\delta$ (theory §12.8)',
        r'$\min(0.02/k_s, 10^{-3} R)$',
        r'both factors $\times 0.03$ to $\times 10$',
        r'$\le$ 4.5e-4 for $\times 0.03$ to $\times 3$; 3.1e-3 at $\times 10$',
        r'central differences: the truncation $O(\delta^2)$ shows beyond $\times 3$; below, the rounding of '
        'the single-precision fields (the 4e-4 floor)',
    ],
    [
        'waterline offset of the direct method (theory §12.8)',
        '$10^{-6} R$',
        '$10^{-9}$ to $10^{-3} R$',
        r'$\le$ 8.2e-6 up to $10^{-6} R$; then linear, 7.4e-5 at $10^{-5} R$',
        r'the offset moves the waterline values by $O(\text{offset})$: at $10^{-6} R$ it is below the other '
        'errors and still clear of the body',
    ],
    [
        'length of the outer integral (theory §12.9)',
        r'4 longest wavelengths, $\ge 10 R$',
        '1 to 8 wavelengths; 2 to $20 R$ (where it governs)',
        '9.5e-4 for 3 to 8 wavelengths, 2.9e-3 for 1–2; 2.8e-3 (sums 7.6e-4) at $20 R$, 6.9e-3 at 2–$5 R$',
        'the integrand decays slowly and oscillates; with the taper the sum converges, to about 1e-3 at '
        "the method's length",
    ],
    [
        'taper of the outer integral (theory §12.9)',
        r'$\cos^2$ over 30 %',
        '5 % to 70 %',
        r'$\le$ 1.8e-3 for 20–70 %; 3.2e-2 at 5 %',
        'a taper of a few per cent is a truncation; from 20 % on the length decides',
    ],
    [
        'Gauss panel width of the outer integral (theory §12.9)',
        r'$\tfrac{1}{4} \cdot 2\pi/(3k_s)$',
        r'$\times 0.25$ to $\times 4$',
        r'$\le$ 1.5e-7',
        '6 Gauss points per quarter of the shortest oscillation integrate it far beyond need',
    ],
    [
        'box widening of several bodies (theory §12.8)',
        '0.45 of the gap',
        '0.2 to 0.49',
        r'$\le$ 3.2e-4 for 0.3–0.49; 1.9e-3 at 0.2',
        'the boxes must not meet (below 0.5); a wide box keeps the cells close to each waterline',
    ],
    [
        'Gauss points per waterline panel side (theory §11.4)',
        '6',
        '1 to 12',
        r'$\le$ 7.5e-14 from 4; 6.7e-10 at 3; 1.7e-6 at 2',
        'the integrand on a side is a product of two potential traces of degree $k - 1$ times the arc-'
        'length element: 6 points are exact to degree 11; on the curved cubic side of the test cylinder, '
        'against 64 points: 3.7e-12 (4 spans), 6.4e-15 (8 spans)',
    ],
    [
        '`body_point_neighbors` (theory §4.2)',
        '4',
        '1 to 12',
        '48 panels: 3.0e-2 ($k = 1$), 7.6e-3 (3), 8.9e-3 (4), 9.8e-3 to 1.2e-2 (6–12); 24 panels: 7.3e-2 '
        '(1), 1.4e-2 (3), 2.5e-2 (4), 1.1e-2 (12); 12 panels: 1.3e-1 (1), 2.9e-2 (3), 4.8e-2 (4)',
        'the error falls with the mesh more than with $k$, and no $k$ is best on every mesh: $k = 3$ on 12 '
        'and 48 panels, $k = 12$ on 24 (where $k = 4$ errs 2.3 times more than $k = 12$); $k = 1$ four to '
        'five times worse than $k = 3$. The default 4, the centroids that surround a point on a '
        'quadrilateral mesh, stays within 1.2 (48 panels) to 2.3 times (24) of the best $k$ and is kept',
    ],
    [
        '`panel_size` defaults (theory §7.3)',
        'order 3, Gauss 3/4',
        'orders 2–5, Gauss 2/3 to 5/6',
        'the default reaches the comparison floor (1.4e-4) at 36 unknowns per quadrant; order 3 with '
        'Gauss 2/3 needs 100; order 4 with Gauss 4/5 25',
        'the outer rule needs at least as many points as the spline order: Gauss 2/3 with order 3 errs by'
        ' 1.5e-2 at $s = 0.8$ m (the default 8.9e-4); higher orders save unknowns at a larger cost per '
        'unknown',
    ],
]


def default_floor(c):
    """The comparison floor of the panel_size study (the smallest difference from Hulme's tables on
    the finest spans) and the unknowns per quadrant at which the default (order 3, Gauss 3/4) first
    comes within 5 % of it."""
    rows = c["panel_size_defaults"]
    finest = min(r["size"] for r in rows)
    floor = min(r["error"] for r in rows if r["size"] == finest)
    default = sorted((r for r in rows if r["order"] == 3 and r["outer"] == 3), key=lambda r: r["unknowns"])
    return floor, next(r["unknowns"] for r in default if r["error"] <= 1.05 * floor)


def page(n):
    c = n["constants"]
    floor, unknowns = default_floor(c)
    head = PAGE_RESULT.format(count=len(TABLE7), floor=e(floor), unknowns=unknowns)
    return head + PAGE.replace("{TABLE7_TABLE}", table(TABLE7_HEADER, TABLE7))


PAGE_RESULT = r"""
# Sensitivity of the fixed constants

**Result.** Each of the {count} fixed constants of Table 1 lies in a flat range of its sensitivity:
varied about its value, it changes the results little, and the change grows only away from the
chosen value. The default
higher-order discretization reaches its best agreement with Hulme's tables ({floor}) at
{unknowns} unknowns per quadrant. The constants that cannot be varied rest on an argument with a measurement.
"""

PAGE = r"""
## What is checked

The constants of [theory](../../theory/README.md) and [second-order](../../theory/12-second-order-loads.md)
that are fixed by the method, with no input to set them, are varied about their values. The measure is
the largest change of the results relative to their largest value. A constant is well chosen when it
sits in a range where this change is flat and small; the page shows that range for each.

## How the constants are varied

The constants of inputs are varied through the inputs. Those of the second-order free surface and of
the higher-order waterline rule have no input; they are varied through the engine's validation hooks.
The environment variable `HB_VALIDATE_<name>` sets a constant for the process when the engine first
reads it (`crates/hydrobem/src/env_hooks.rs`; `validation/constants.py` lists the names). Without
the variables every constant is the method's and the results are unchanged.

An unknown `HB_VALIDATE_` name or a value that does not parse refuses the solve; a set hook is a
warning and is listed in the meta data of the results. The hooks are for validation only and never
change a default: `VALIDATION_CONSTANTS` in that file lists every hooked constant with its value, and
a unit test holds each value to the theory documents.

Where a constant cannot be varied, its ground is an argument with a measurement (the last section).

## The second-order free surface and the waterline rule

![Constants of the second-order free surface](../figures/constants_second_order.png)

*Figure 1. The truncated cylinder of radius 1 m and draft 0.5 m ([cylinder](cylinder.md); deep water,
the automatic $R = 4$ m and panel size; the stencil and the waterline offset on the floating cylinder,
the offset with the direct method; the length of the outer integral in radii at $\omega = 3$ and 3.5
rad/s, where it governs; the box widening on two fixed cylinders 3.2 m apart, cells of 0.25 m to
$R = 5$ m, $\omega = 1.5$ and 2 rad/s): the largest change of the sum and difference coefficients
relative to the largest coefficient against each constant over its value. Last panel: the
higher-order pressure drift (surge, heave, pitch) of the floating spline cylinder (8 spans per quarter
circle, $0.6 \le \omega \le 3$ rad/s) against the Gauss points per waterline panel side. Changes below
1e-9 (in the last panel 1e-12), exact zeros among them, are drawn at that floor.*

## The constants of inputs

![Constants of inputs](../figures/constants_inputs.png)

*Figure 2. Left: the low-order diffraction pressure at 40 points on the wetted hemisphere (not
centroids; $Ka = 0.5$, 1, 2) as the inverse-distance mean of the $k$ nearest centroids, against the
higher-order pressure at the same points ($8 \times 8$ spans), on 12, 24, 48 panels per quadrant edge.
Right: the hemisphere ([floating hemisphere](floating_hemisphere.md)) in the higher order with the
spans of the `panel_size` rule ($n = \lfloor L/s + 10^{-5} \rfloor + 1$, $s = 1.6$ to 0.2 m), each
spline order and pair of Gauss orders, against Hulme (1982) at $Ka = 0.5$, 1, 2, over the unknowns per
quadrant (colour: the spline order; solid: Gauss orders $k/k + 1$, dotted: one higher, dashed: one
lower; the dashed line the rounding of the four-digit tables).*

## The sensitivity of each constant

Table 1 gives, for each constant, the range over which it was varied, the largest change it caused
and the reason for its value.

*Table 1. The constants without an input and their sensitivity.*

{TABLE7_TABLE}

## The constants that cannot be varied

Some constants cannot be varied without changing the method itself. Each rests on an argument and a
measurement:

- **The +15 of the expansion order $M = \lceil kR \rceil + 15$ and the cap 160** (theory §12.9). The
  input `expansion_order` varies $M$ ([cylinder](cylinder.md)): the results are unchanged from
  $\lceil kR \rceil - 8$ on, so the margin 15 is a reserve of more than 20 orders here. The fit's own
  order $M_f = \max(M, \lceil kR_{\mathrm{OC}} \rceil + 15)$ keeps the margin: the orders above $kR$ of a
  disturbance from bodies within a radius $a$ appear at $R$ with about the factor
  $(a/R)^{\lvert m \rvert - kR}$, $4^{-15} \approx 10^{-9}$ at the automatic $R = 4a$. The cap 160 is the
  highest order to which the Bessel recurrences are tested (test-plan A9): against mpmath to order
  150, by the Wronskian to 160; $M \le 160$ holds for $kR$ up to 145.
- **The 1e-5 of the `panel_size` rule** $n = \lfloor L/s + 10^{-5} \rfloor + 1$: a patch edge whose
  length is an exact multiple of $s$ ($L/s$ an integer up to the rounding of the measured arc length)
  gets the intended $n$; 1e-5 is far above that rounding and far below a meaningful fraction of a
  panel.
- **Galerkin testing in the higher order** (theory §7): Maniar (1995, §3.4) weighed collocation
  at Gauss points, an overdetermined system solved by least squares whose accuracy depends on the
  points chosen and whose solution and storage cost outweigh its simpler assembly, against the
  Galerkin procedure, which gives a square system of the size of the unknowns; Galerkin
  approximants also give integrated quantities $\int \phi g \,\mathrm{d}S$ more accurately than the
  solution itself (Maniar 1995, §3.3). Measured: the higher order reaches the comparison floor with
  Hulme's tables with 36 unknowns per quadrant in the default order 3 (Gauss 3/4), 25 in order 4
  (Gauss 4/5) ([floating hemisphere](floating_hemisphere.md) and Figure 2).
- **No condition estimate** (theory §13.1): the second-kind equations are well conditioned except
  near the irregular frequencies, where the condition number grows without bound; the remedy is the
  interior surface ([floating hemisphere](floating_hemisphere.md)), which removes the
  near-singularity rather than reporting it. A zero
  pivot and non-finite values are refused. The measured condition numbers are in theory §7 and §13.1.
- **The layout of the generated free-surface panels** (theory §12.8: the square of half-width
  $\sqrt{2}\, r_c$, $r_0 = 2 r_c$, the spokes rounded): the layout is one discretization of the same
  integral; the cells, an independent layout, agree with it to the discretization error at every size
  ([cylinder](cylinder.md)), so the layout adds no error beyond that of the panel size.
- **The pressure-mode problem posed in the applied pressure** (theory §9.3): the pressure head of a
  mode is the coordinate the user prescribes (the uniform pressure of an air cushion), so the
  radiation problem is posed in it directly; the elevation-mode form of Lee & Newman (2000) gives
  the same coefficients (their eqs. 20–21) and needs modes that span the elevation, which is not
  uniform over the surface.

## Reproducing this page

This page, its figures and its numbers are written by `validation/constants.py` through
`validation/make_report.py`; the second-order cases run in processes of their own, one per value of a
hook.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
