"""The floating truncated vertical cylinder (radius a = 1 m, draft d = a) in water of depth h = 2a and
4a against the eigenfunction-matching reference of scripts/gen_cylinder_reference.py (the
formulation of Yeung 1981 and Garrett 1971; heave as Bhatta & Rahman 1995, surge as Bhatta 2003):
the heave and surge added mass and damping over Ka in the low and the higher order, the reference's
truncation convergence, and the mesh convergence of the low order with Richardson's extrapolation
(docs/validation/cases/truncated_cylinder_finite_depth.md).

    python validation/cylinder.py [--quick]
"""

from __future__ import annotations

import sys

import matplotlib.ticker
import mesh_figure
import numpy as np
from common import DATA, REF, RHO, C, G, Timer, e, observed_order, pc, plt, record, save, size, solve, table, tex

from bodies import graded_cylinder
from problems import analytic, load_script, model

NAMES = ["A₃₃/(ρa³)", "B₃₃/(ρωa³)", "A₁₁/(ρa³)", "B₁₁/(ρωa³)"]
ref_script = load_script("gen_cylinder_reference")


def higher_mesh():
    """The analytic cylinder (exact side and bottom), 8 x 8 spans on the side and 8 x 4 on the bottom
    a quadrant, order 3, the nonuniform mapping (the spans clustered at the waterline and the
    bottom edge)."""
    return analytic("circular_cylinder", [(8, 8, 3, 3, 3, 4), (8, 4, 3, 3, 3, 4)], parts=["side", "bottom"],
                    mapping="nonuniform", radius=1.0, draft=1.0, symmetry="xz_and_yz")  # fmt: skip


def bem(n, h, ka):
    """[Ka, 4] on the graded mesh of n panels a quarter circle (n = 0: the higher order)."""
    w = np.sqrt(np.asarray(ka) * G)
    body = {"mesh": higher_mesh(), "interior_surface": "none"} if n == 0 else {"mesh": graded_cylinder(n)}
    r = solve({**body, "modes": ["surge", "heave"]}, w, depth=h,
              options={"diffraction": False, "higher_order": n == 0})  # fmt: skip
    A, B = r.added_mass / RHO, r.radiation_damping / RHO
    return np.c_[A[:, 2, 1], B[:, 2, 1] / w, A[:, 0, 0], B[:, 0, 0] / w]


def reference(h, ka):
    out = []
    for k in ka:
        (a3, b3), _ = ref_script.converged(h, 1.0, float(k), "heave")
        (a1, b1), _ = ref_script.converged(h, 1.0, float(k), "surge")
        out.append([a3, b3, a1, b1])
    return np.array(out)


def curves(quick):
    ka = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    if quick:
        ka = ka[::2]
    res = {}
    with Timer("cylinder: reference and mesh of 48 over Ka"):
        for h in (2.0, 4.0):
            res[h] = (reference(h, ka), bem(48 if not quick else 24, h, ka), bem(0, h, ka))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    for ax, (ia, ib, mode) in zip(axes, ((0, 1, "heave"), (2, 3, "surge")), strict=True):
        for j, h in enumerate((2.0, 4.0)):
            ref, got, ho = res[h]
            for col, ls in ((ia, "-"), (ib, "--")):
                ax.plot(ka, ref[:, col], **{**REF, "marker": "o" if col == ia else "s",
                                             "label": f"reference h/a = {h:g}" if col == ia else None})  # fmt: skip
                ax.plot(ka, got[:, col], color=C[j], ls=ls,
                        label=f"low order h/a = {h:g}, {'A' if col == ia else 'B/ω'}")  # fmt: skip
                ax.plot(ka, ho[:, col], color=C[j + 2], ls=":", lw=2.2,
                        label="higher order" if (col == ia and j == 0) else None)  # fmt: skip
        ax.set_xlabel("Ka = ω²a/g")
        ax.set_ylabel("A/(ρa³) (solid), B/(ρωa³) (dashed)")
        ax.set_title(f"{mode}, d/a = 1")
        ax.legend(fontsize=7)
    save(fig, "cylinder_finite_depth")
    err = {f"h{h:g}": np.abs(res[h][1] - res[h][0]).max(axis=0) / np.abs(res[h][0]).max(axis=0) for h in res}
    err_ho = {f"h{h:g}": np.abs(res[h][2] - res[h][0]).max(axis=0) / np.abs(res[h][0]).max(axis=0) for h in res}
    print("  cylinder over Ka:", {k: np.round(v, 5) for k, v in err.items()},
          {k: np.round(v, 6) for k, v in err_ho.items()})
    n = 48 if not quick else 24
    env = {"density": RHO, "gravity": G, "depth": 2.0}
    mesh_figure.draw("truncated_cylinder_finite_depth", [
        (f"low order, {n} panels a quarter circle", model({"mesh": graded_cylinder(n)}, [1.0], [0.0], environment=env),
         False),
        ("higher order, 8 × 8 and 8 × 4 spans", model({"mesh": higher_mesh(), "interior_surface": "none"}, [1.0], [0.0],
                                                     environment=env, options={"higher_order": True}), True)],
        elev=-20.0)  # fmt: skip
    return {"mesh": n, "ka": ka, "errors": err, "errors_higher": err_ho}


def convergence(quick):
    """The six tabulated cases (tests/data/cylinder_finite_depth.txt) on 12, 16, 24, 32, 48 panels a
    quarter circle; the reference's truncation N = 25 .. 800 at one case."""
    tab = np.loadtxt(DATA / "cylinder_finite_depth.txt")
    ns = [12, 16, 24, 32] if quick else [12, 16, 24, 32, 48]
    vals = []
    with Timer("cylinder: meshes"):
        for n in ns:
            vals.append(np.vstack([bem(n, h, tab[tab[:, 0] == h, 2]) for h in (2.0, 4.0)]))
    exact = tab[:, 3:]
    scale = np.abs(exact).max(axis=0)
    errs = np.array([np.abs(v - exact).max(axis=0) / scale for v in vals])
    p, extrap = observed_order(vals[-3:], [1.0 / n for n in ns[-3:]])
    e_ex = np.abs(extrap - exact).max(axis=0) / scale
    signed = np.array([(v - exact) / scale for v in vals])  # monotone: the signs and magnitudes
    monotone = bool(np.all(np.abs(signed[1:]) <= np.abs(signed[:-1]) + 1e-12))
    # the truncation of the reference
    Ns = [25, 50, 100, 200, 400, 800]
    with Timer("cylinder: reference truncation"):
        tr = np.array([ref_script.solve(2.0, 1.0, 0.8, N, "heave") + ref_script.solve(2.0, 1.0, 0.8, N, "surge")
                       for N in Ns])  # fmt: skip
    best = exact[1]  # h/a = 2, Ka = 0.8
    etr = np.abs(tr - best).max(axis=1) / np.abs(best).max()
    out = {"meshes": ns, "errors": errs, "order_median": np.median(p, axis=0), "extrapolated": e_ex,
           "monotone": monotone, "truncation_N": Ns, "truncation_errors": etr}  # fmt: skip
    plot_convergence(out)
    print("  cylinder meshes", np.round(errs.max(axis=1), 5), "p", np.round(np.median(p, axis=0), 2),
          "extrapolated", np.round(e_ex, 6), "monotone", monotone)  # fmt: skip
    return out


def plot_convergence(d):
    """The figure cylinder_convergence from the numbers of convergence (also as stored)."""
    ns, errs, e_ex = d["meshes"], np.asarray(d["errors"]), d["extrapolated"]
    Ns, etr = d["truncation_N"], np.asarray(d["truncation_errors"])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    ax = axes[0]
    for i, name in enumerate(NAMES):
        ax.loglog(ns, errs[:, i], "o-", color=C[i], ms=4, label=name)
        ax.loglog([ns[-1]], [e_ex[i]], "*", color=C[i], ms=9)
    nn = np.array([ns[0], ns[-1]], dtype=float)
    ax.loglog(nn, errs[0].max() * (nn / ns[0]) ** -1.8, color="#888888", ls=":", label="∝ n⁻¹·⁸")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("panels per quarter circle n")
    ax.set_ylabel("max |HydroBEM − reference| / max |reference|")
    ax.set_title("low-order mesh (★ Richardson extrapolation)")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.loglog(Ns, etr, "o-", color=C[0], ms=4, label="h/a = 2, Ka = 0.8, heave and surge")
    ax.loglog(Ns, etr[1] * (np.array(Ns) / Ns[1]) ** (-4 / 3), color="#888888", ls=":", label="∝ N^(−4/3)")
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("terms of the eigenfunction expansions N")
    ax.set_ylabel("difference from the extrapolated reference")
    ax.set_title("reference: truncation")
    ax.legend(fontsize=7)
    save(fig, "cylinder_convergence")


def replot(numbers):
    """Redraws the figures whose numbers numbers.json holds (cylinder_convergence)."""
    plot_convergence(numbers["cylinder_convergence"])


def main(quick=False):
    record("cylinder_finite_depth", curves(quick))
    record("cylinder_convergence", convergence(quick))


NAME = "truncated_cylinder_finite_depth"
GROUPS = ("cylinder_finite_depth", "cylinder_convergence")
INDEX = {"case": "Truncated cylinder, finite depth",
         "checked": "heave, surge $A$, $B$ at $h/a = 2$, 4; mesh convergence",
         "sources": "eigenfunction matching (Yeung 1981; Garrett 1971; Bhatta & Rahman 1995; Bhatta 2003)",
         "methods": "low, higher"}  # fmt: skip


def summary(n):
    f = n["cylinder_finite_depth"]
    lo = max(max(v) for v in f["errors"].values())
    hi = max(max(v) for v in f["errors_higher"].values())
    return (f"both methods agree with the reference, within {pc(lo)} (low order, {f['mesh']} panels) and "
            f"{size(hi)} (higher order)")  # fmt: skip


def page(n):
    f, c = (n[g] for g in GROUPS)
    rows = [[f"{h[1:]}", f"low, {f['mesh']}"] + [e(x) for x in f["errors"][h]] for h in f["errors"]]
    rows += [[f"{h[1:]}", r"higher, $8 \times 8$ spans"] + [e(x) for x in f["errors_higher"][h]]
             for h in f["errors_higher"]]
    errs = np.asarray(c["errors"]).max(axis=1)
    p = c["order_median"]
    lo = max(max(v) for v in f["errors"].values())
    hi = max(max(v) for v in f["errors_higher"].values())
    ex = c["extrapolated"]
    fall = "fall monotonically" if c["monotone"] else "fall, though not monotonically,"
    return rf"""
# Truncated cylinder in finite depth

**Result.** In water two and four radii deep, the low order on {f['mesh']} panels per quarter circle
agrees with an eigenfunction-matching solution within {pc(lo)} of the largest reference value, and
the higher order within {size(hi)}. The low order converges at the order observed,
{min(p):.1f} to {max(p):.1f}, and its extrapolation to zero panel size reaches the reference
within {tex(min(ex))} to {tex(max(ex))}.

## The problem

A truncated vertical cylinder of radius $a = 1$ m and draft $d = a$ floats in water of depth $h = 2a$
and $4a$. We compare its heave and surge added mass $A/(\rho a^3)$ and damping $B/(\rho\omega a^3)$
at 11 values $0.1 \le Ka \le 3$, where $Ka = \omega^2 a/g$.

![Meshes](../figures/mesh_{NAME}.png)

*Figure 1. The two meshes, expanded to the whole body and seen from below: the wetted surface (side
and bottom) in blue. Left: low order, {f['mesh']} panels per quarter circle, cosine-spaced in depth
and radius. Right: higher order, the analytic cylinder (patch edges in black, span edges in grey),
$8 \times 8$ spans on the side and $8 \times 4$ on the bottom per quadrant, order 3, the spans
clustered at the waterline and the bottom edge. There is no interior surface: the first irregular
frequency lies above the range.*

## The reference

The reference is the eigenfunction-matching solution of `scripts/gen_cylinder_reference.py`: the
formulation of Yeung (1981) and Garrett (1971), heave as Bhatta & Rahman (1995) and surge as Bhatta
(2003), derived and implemented in this repository. Its truncation is extrapolated (Figure 3,
right), and the result is converged to better than $5 \times 10^{{-6}}$.

Its sources give their results as figures only, so no published number anchors it. It rests on the
published formulation, its truncation convergence and its agreement with the panel method
(test-plan A5).

## Agreement with the reference

![Truncated cylinder in finite depth](../figures/cylinder_finite_depth.png)

*Figure 2. Heave and surge added mass $A/(\rho a^3)$ (solid) and damping $B/(\rho\omega a^3)$
(dashed) against $Ka$ for $h/a = 2$ and 4: the reference (markers), the low order and the higher
order (dotted).*

The curves of both methods lie on the reference at both depths. Table 1 gives the largest
difference of each quantity.

*Table 1. Largest difference from the reference over $0.1 \le Ka \le 3$, relative to the largest
reference value of each quantity.*

{table(["$h/a$", "method", "$A_{33}$", "$B_{33}$", "$A_{11}$", "$B_{11}$"], rows)}

## Does it converge?

A reference built in this repository needs a check of its own, and the low order needs its rate.
Figure 3 refines the low-order mesh on the six tabulated cases and the reference's truncation. The
order $p$ is observed on three meshes, $v_i = v^* + c\,h_i^p$, and Richardson's extrapolation $v^*$
is the limit of zero panel size that this rate implies.

![Cylinder convergence](../figures/cylinder_convergence.png)

*Figure 3. Left: the six tabulated cases (`tests/data/cylinder_finite_depth.txt`) on
{", ".join(map(str, c['meshes']))} panels per quarter circle; the stars are Richardson's
extrapolation from the three finest meshes with the order observed for each value. Right: the
reference's truncation, $N$ terms in each eigenfunction expansion, against its extrapolation
($h/a = 2$, $Ka = 0.8$, heave and surge).*

The low order's differences {fall} from {e(errs[0])} to {e(errs[-1])} ($n = {c['meshes'][0]}$ to
{c['meshes'][-1]}). The observed orders are {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f} and {p[3]:.2f} for
$A_{{33}}$, $B_{{33}}$, $A_{{11}}$ and $B_{{11}}$; the bottom edge of the cylinder keeps them below
2. The extrapolation leaves {", ".join(e(x) for x in ex)}.

The reference converges as $N^{{-4/3}}$, the rate set by the $270^\circ$ corner of the water at the
bottom edge. Its extrapolations from $N = 200$, 400 and from 400, 800 agree to better than
$5 \times 10^{{-6}}$; the script refuses otherwise.

The higher order on 160 unknowns per quadrant agrees with the reference to about $10^{{-5}}$
directly, without extrapolation.

## Reproducing this page

The example [`examples/{NAME}`](../../../examples/{NAME}) runs the tabulated cases. This page, its
figures and its numbers are written by `validation/cylinder.py` through `validation/make_report.py`.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
