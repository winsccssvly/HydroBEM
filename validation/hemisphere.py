"""The floating hemisphere of radius a = 1 m in deep water against Hulme (1982), tables 1 and 2
(tests/data/hulme1982_full.txt, extracted by scripts/extract_hulme.py): the added mass and damping
of heave and surge over 0.1 <= Ka <= 10 in the low order (graded mesh, interior surface) and the
higher order (rational B-spline patches); the added masses at omega = 0 and inf against the exact
1/2 and Hulme's eqs. (3.18) and (5.15) (tests/data/hulme1982_limits.txt, the same script); the
mesh convergence of both; the irregular frequencies
with and without the interior surface (docs/validation/cases/floating_hemisphere.md; its helpers
serve the constants, docs/validation/cases/constants.md).

    python validation/hemisphere.py [--quick]
"""

from __future__ import annotations

import math
import sys

import mesh_figure
import numpy as np
from common import DATA, REF, RHO, C, G, Timer, e, observed_order, pc, plt, record, save, size, solve, table, tex

from bodies import hemi, part, spline, surface
from meshes import QUARTER_CIRCLE, QUARTER_WEIGHTS, hemisphere_panels, nurbs_hemisphere_patches
from problems import model

M0 = RHO * 2.0 / 3.0 * math.pi
NAMES = ["A₃₃/m₀", "B₃₃/(ω m₀)", "A₁₁/m₀", "B₁₁/(ω m₀)"]


def hulme():
    return np.loadtxt(DATA / "hulme1982_full.txt")


def coefficients(body, ka, higher=False):
    """[Ka, 4]: heave A, B and surge A, B normalized as Hulme's tables."""
    w = np.sqrt(np.asarray(ka) * G)
    r = solve({"modes": ["surge", "heave"], **body}, w,
              options={"diffraction": False, "higher_order": higher})  # fmt: skip
    A, B = r.added_mass, r.radiation_damping
    return np.c_[A[:, 2, 1] / M0, B[:, 2, 1] / (w * M0), A[:, 0, 0] / M0, B[:, 0, 0] / (w * M0)]


def lo_graded(n):
    return {"mesh": surface(hemisphere_panels(1.0, n, n, quarter=True, graded=True), symmetry="xz_and_yz"),
            "interior_surface": "projected"}  # fmt: skip


def ho_nurbs(spans, order=3, interior=True):
    """The rational hemisphere patch; with the interior surface a rational quarter disk in z = 0
    (v from the axis to the waterline: the normal up), a patch of the same mesh."""
    rows = (spans, spans, order, order, order, order + 1)
    if not interior:
        return {"mesh": spline(nurbs_hemisphere_patches(1.0), rows, symmetry="xz_and_yz")}
    tu, tv = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    disk = np.array([[(r * a, r * b, 0.0) for a, b in QUARTER_CIRCLE] for r in (0.0, 1.0)])
    lid = (3, 2, tu, tv, disk, np.array([QUARTER_WEIGHTS] * 2))
    m = spline(nurbs_hemisphere_patches(1.0) + [lid], rows, symmetry="xz_and_yz")
    return {"mesh": part(m, 1, 1), "interior_surface": part(m, 2, 2)}


def errors(vals, ref):
    """Per quantity the largest difference over Ka relative to the largest reference value."""
    return np.abs(vals - ref).max(axis=0) / np.abs(ref).max(axis=0)


def against_hulme(quick):
    ref = hulme()
    ka = ref[:, 0]
    n_lo = 24 if quick else 48
    with Timer("hemisphere: low order over the range"):
        lo = coefficients(lo_graded(n_lo), ka)
    with Timer("hemisphere: higher order over the range"):
        ho = coefficients(ho_nurbs(4 if quick else 8), ka, higher=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    for ax, (ia, ib, mode) in zip(axes, ((0, 1, "heave"), (2, 3, "surge")), strict=True):
        for col, lab, mk in ((ia, "A/m₀", "o"), (ib, "B/(ω m₀)", "s")):
            ax.plot(ka, ref[:, 1 + col], **{**REF, "marker": mk, "label": f"Hulme (1982) {lab}"})
            ax.plot(ka, lo[:, col], color=C[0], label=f"low order, {n_lo} graded divisions" if col == ia else None)
            ax.plot(ka, ho[:, col], color=C[1], ls="--",
                    label=f"higher order, {(4 if quick else 8)}² spans" if col == ia else None)  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlabel("Ka = ω²a/g")
        ax.set_ylabel("A/m₀, B/(ω m₀)")
        ax.set_title(f"{mode}")
    axes[0].legend(loc="upper right")  # one legend for both panels: the surge curves fill the other
    save(fig, "hemisphere_hulme")
    env = {"density": RHO, "gravity": G}
    mesh_figure.draw("floating_hemisphere", [
        (f"low order, {n_lo} graded divisions", model(lo_graded(n_lo), [1.0], [0.0], environment=env), False),
        (f"higher order, {4 if quick else 8}² spans, rational patches",
         model(ho_nurbs(4 if quick else 8), [1.0], [0.0], environment=env, options={"higher_order": True}), True)],
        elev=20.0)  # fmt: skip
    out = {"low_order_mesh": n_lo, "low_order": errors(lo, ref[:, 1:]), "higher_order": errors(ho, ref[:, 1:]),
           "higher_order_spans": 4 if quick else 8}  # fmt: skip
    print("  Hulme: low", np.round(out["low_order"], 6), "higher", np.round(out["higher_order"], 6))
    return out


def ticks(ax, values):
    """Plain tick labels at the values of a logarithmic axis."""
    ax.set_xticks(values, [str(v) for v in values])
    ax.minorticks_off()


def mesh_convergence(quick):
    """The low order on n x n panels a quadrant (uniform in the angles) and the higher order on
    spans x spans patches, at Ka = 0.5, 1, 2 (the rows of test-plan A4)."""
    ref = hulme()
    sel = np.isin(ref[:, 0], [0.5, 1.0, 2.0])
    ka, r = ref[sel, 0], ref[sel, 1:]
    ns = [8, 12, 16, 24, 32] if quick else [8, 12, 16, 24, 32, 48]
    with Timer("hemisphere: low-order meshes"):
        lo = [coefficients({"mesh": hemi(n)}, ka) for n in ns]
    spans = [1, 2, 4] if quick else [1, 2, 4, 8]
    with Timer("hemisphere: higher-order meshes"):
        ho3 = [coefficients(ho_nurbs(s, 3, False), ka, True) for s in spans]
        ho4 = [coefficients(ho_nurbs(s, 4, False), ka, True) for s in spans]
    e_lo = np.array([errors(v, r) for v in lo])
    e3 = np.array([errors(v, r) for v in ho3])
    e4 = np.array([errors(v, r) for v in ho4])
    # the observed order and Richardson's extrapolation of the low order from its three finest meshes
    h = [1.0 / n for n in ns[-3:]]
    p, extrap = observed_order(lo[-3:], h)
    e_ex = errors(extrap, r)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax = axes[0]
    for i, name in enumerate(NAMES):
        ax.loglog(ns, e_lo[:, i], "o-", color=C[i], ms=4, label=name)
    nn = np.array([ns[0], ns[-1]], dtype=float)
    ax.loglog(nn, e_lo[0].max() * (nn / ns[0]) ** -2, color="#888888", ls=":", label="∝ n⁻²")
    ax.axhline(5e-5 / 0.86, color="#888888", lw=0.8, ls="--")
    ax.text(ns[0], 5e-5 / 0.86 * 1.15, "rounding of the 4-digit tables", fontsize=7, color="#555555")
    ticks(ax, ns)
    ax.set_xlabel("panels per quadrant edge n")
    ax.set_ylabel("max |HydroBEM − Hulme| / max |Hulme|")
    ax.set_title("low order")
    ax.legend()
    ax = axes[1]
    for i, name in enumerate(NAMES):
        ax.loglog(spans, e3[:, i], "o-", color=C[i], ms=4, label=f"{name}, order 3")
        ax.loglog(spans, e4[:, i], "s--", color=C[i], ms=4, mfc="none", label="order 4" if i == 0 else None)
    ax.axhline(5e-5 / 0.86, color="#888888", lw=0.8, ls="--")
    ticks(ax, spans)
    ax.set_xlabel("spans per patch direction")
    ax.set_ylabel("max |HydroBEM − Hulme| / max |Hulme|")
    ax.set_title("higher order (one rational patch a quadrant)")
    ax.legend(fontsize=7)
    save(fig, "hemisphere_convergence")
    out = {"low_meshes": ns, "low_errors": e_lo, "low_order_p_median": np.median(p, axis=0),
           "low_extrapolated_errors": e_ex, "higher_spans": spans, "higher_order3_errors": e3,
           "higher_order4_errors": e4}  # fmt: skip
    print("  low errors", np.round(e_lo.max(axis=1), 6), "p", np.round(np.median(p, axis=0), 2),
          "extrapolated", np.round(e_ex, 6))  # fmt: skip
    print("  higher errors 3", np.round(e3.max(axis=1), 6), "4", np.round(e4.max(axis=1), 6))
    return out


def irregular(quick):
    """The plain low order against the interior surfaces (projected, waterline, graded), 16 panels
    a quadrant edge, Ka from 0.5 to 10."""
    ref = hulme()
    tab = ref[(ref[:, 0] >= 0.5) & (ref[:, 0] <= 10.0)]
    ka = np.union1d(np.round(np.linspace(0.5, 10.0, 40 if quick else 96), 6), tab[:, 0])
    mesh = surface(hemisphere_panels(1.0, 16, 16, quarter=True, graded=True), symmetry="xz_and_yz")
    runs = {}
    with Timer("hemisphere: irregular frequencies"):
        for kind in (None, "projected", "waterline", "graded"):
            body = {"mesh": mesh} if kind is None else {"mesh": mesh, "interior_surface": kind}
            runs[kind or "none"] = coefficients(body, ka)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharex=True)
    for ax, col, lab in ((axes[0], 1, "B₃₃/(ω m₀)"), (axes[1], 2, "A₁₁/m₀")):
        ax.plot(ref[:, 0], ref[:, 1 + col], **{**REF, "label": "Hulme (1982)"})
        ax.plot(ka, runs["none"][:, col], color=C[1], label="without interior surface")
        ax.plot(ka, runs["projected"][:, col], color=C[0], label="interior surface (projected)")
        ax.set_xlim(0.5, 10)
        ax.set_xlabel("Ka = ω²a/g")
        ax.set_ylabel(lab)
        ax.legend()
    save(fig, "hemisphere_irregular")
    # the largest difference from Hulme at the tabulated Ka per variant
    at = np.searchsorted(ka, tab[:, 0])
    out = {k: errors(v[at], tab[:, 1:]) for k, v in runs.items()}
    # the generators against each other
    for k in ("waterline", "graded"):
        out[f"{k}_vs_projected"] = errors(runs[k], runs["projected"])
    print("  irregular:", {k: f"{np.max(v):.2e}" for k, v in out.items()})
    return out


LIMIT_NAMES = ["heave, ω = 0", "heave, ω = ∞", "surge, ω = 0", "surge, ω = ∞"]


def hulme_limits():
    """[heave ω = 0, heave ω = ∞, surge ω = 0, surge ω = ∞] of Hulme (1982) (A/m₀;
    tests/data/hulme1982_limits.txt, extracted by scripts/extract_hulme.py)."""
    ref = np.loadtxt(DATA / "hulme1982_limits.txt")
    return np.array([ref[0, 1], ref[1, 1], ref[0, 2], ref[1, 2]])


def limit_values(body, higher=False):
    """The added masses at ω = 0 and ω = ∞ in the order of hulme_limits()."""
    r = solve({"modes": ["surge", "heave"], **body}, [0.0, math.inf],
              options={"diffraction": False, "higher_order": higher})  # fmt: skip
    a0, ai = r.added_mass_zero_frequency / M0, r.added_mass_infinite_frequency / M0
    return np.array([a0[2, 1], ai[2, 1], a0[0, 0], ai[0, 0]])


def limits(quick):
    """The added masses at ω = 0 and ω = ∞ on the meshes of the comparison with the tables, and
    the low order's convergence on n x n panels a quadrant (uniform in the angles)."""
    ref = hulme_limits()
    n_lo, spans = (24, 4) if quick else (48, 8)
    with Timer("hemisphere: the limits"):
        lo = limit_values(lo_graded(n_lo))
        ho = limit_values(ho_nurbs(spans), higher=True)
        ns = [8, 12, 16] if quick else [12, 24, 48]
        conv = [limit_values({"mesh": hemi(n)}) for n in ns]
    p, extrap = observed_order(conv, [1.0 / n for n in ns])
    out = {"reference": ref, "low": lo, "higher": ho, "low_rel": np.abs(lo / ref - 1.0),
           "higher_rel": np.abs(ho / ref - 1.0), "meshes": ns,
           "convergence_rel": [np.abs(v / ref - 1.0) for v in conv], "p": p,
           "extrapolated_rel": np.abs(extrap / ref - 1.0)}  # fmt: skip
    print("  limits: low", np.round(out["low_rel"], 6), "higher", np.round(out["higher_rel"], 6), "p", np.round(p, 2),
          "extrapolated", np.round(out["extrapolated_rel"], 6))  # fmt: skip
    return out


def main(quick=False):
    record("hemisphere_hulme", against_hulme(quick))
    record("hemisphere_convergence", mesh_convergence(quick))
    record("hemisphere_irregular", irregular(quick))
    record("hemisphere_limits", limits(quick))


NAME = "floating_hemisphere"
GROUPS = ("hemisphere_hulme", "hemisphere_convergence", "hemisphere_irregular", "hemisphere_limits")
INDEX = {"case": "Floating hemisphere", "checked": r"heave, surge $A$, $B$ over $0.1 \le Ka \le 10$; $A$ at "
         r"$\omega = 0$, $\infty$; mesh convergence; irregular frequencies",
         "sources": "Hulme (1982) tables 1, 2, eqs. (3.18), (5.15)", "methods": "low, higher"}  # fmt: skip


def summary(n):
    h, lim = n["hemisphere_hulme"], n["hemisphere_limits"]
    lo, hi = max(h["low_order"]), max(h["higher_order"])
    return (f"both methods agree with the tables, within {pc(lo)} (low order) and {pc(hi)} (higher order) of "
            f"the largest tabulated value; the limits $\\omega = 0$, $\\infty$ within {pc(max(lim['low_rel']))} "
            f"and {pc(max(lim['higher_rel']))}")  # fmt: skip


def converged_at(values, counts, factor=1.1):
    """The first count from which the values stay within the factor of the last one."""
    for i in range(len(values)):
        if all(x <= factor * values[-1] for x in values[i:]):
            return counts[i]
    return counts[-1]


def page(n):
    h, c, irr, lim = (n[g] for g in GROUPS)
    # the reference values from their file (numbers.json keeps four digits), the differences as run
    sources = ["Hulme eq. (3.18), $L$", "exact, $1/2$", "exact, $1/2$", "Hulme eq. (5.15), $C_1$"]
    small = lambda x: e(x) if x >= 1e-7 else r"$< 10^{-7}$"  # noqa: E731 (below the single precision)
    lim_rows = [[name.replace("ω = 0", r"$\omega = 0$").replace("ω = ∞", r"$\omega = \infty$"), src, f"{r:g}",
                 f"{a:.4f}", small(da), f"{b:.4f}", small(db)]
                for name, src, r, a, da, b, db in zip(
                    LIMIT_NAMES, sources, hulme_limits(), lim["low"], lim["low_rel"], lim["higher"],
                    lim["higher_rel"], strict=True)]  # fmt: skip
    lim_conv = ", ".join(e(max(x)) for x in lim["convergence_rel"])
    rows = [[f"low order, {h['low_order_mesh']} graded divisions, interior surface"] + [e(x) for x in h["low_order"]],
            [rf"higher order, ${h['higher_order_spans']} \times {h['higher_order_spans']}$ spans, interior surface"]
            + [e(x) for x in h["higher_order"]]]  # fmt: skip
    lo = np.asarray(c["low_errors"]).max(axis=1)
    p = c["low_order_p_median"]
    e3, e4 = np.asarray(c["higher_order3_errors"]).max(axis=1), np.asarray(c["higher_order4_errors"]).max(axis=1)
    sp = c["higher_spans"]
    ns = c["low_meshes"]
    ex = c["low_extrapolated_errors"]
    gen = max(max(irr["waterline_vs_projected"]), max(irr["graded_vs_projected"]))
    worst = max(max(h["low_order"]), max(h["higher_order"]))
    s3, s4 = converged_at(list(e3), sp), converged_at(list(e4), sp)
    rounding = 6e-5
    level = ("the rounding of the tables themselves" if max(ex) < 3 * rounding
             else f"{tex(max(ex))}, against a rounding of the tables of {tex(rounding, 0)}")  # fmt: skip
    rate = "$h^2$" if all(1.8 <= x <= 2.3 for x in p) else "$h^p$ with $p$ = " + ", ".join(f"{x:.1f}" for x in p)
    lid = ("removes the irregular frequencies without changing the results elsewhere"
           if max(irr["projected"]) < max(irr["none"]) else "does not lower the largest difference on the coarse mesh")
    header = ["Method, mesh", "$A_{33}/m_0$", r"$B_{33}/(\omega m_0)$", "$A_{11}/m_0$", r"$B_{11}/(\omega m_0)$"]
    return rf"""
# Floating hemisphere

**Result.** Both methods reproduce the published added mass and damping of a floating hemisphere
(Hulme 1982) to within {pc(worst)} of the largest tabulated value. The low order converges as {rate},
and extrapolated to zero panel size it lands at {level}; the higher order is converged with {s3} spans
per quadrant. The added masses at $\omega = 0$ and $\omega = \infty$ agree with their exact values and
Hulme's series within {pc(max(max(lim['low_rel']), max(lim['higher_rel'])))}. An interior surface {lid}.

## The problem

A hemisphere of radius $a = 1$ m floats with its equator in the free surface of deep water. We
compare its heave and surge added mass $A/m_0$ and damping $B/(\omega m_0)$, made dimensionless by
the displaced mass $m_0 = 2\pi\rho a^3/3$, with Hulme's tables at the 24 tabulated values
$0.1 \le Ka \le 10$, where $Ka = \omega^2 a/g$.

![Meshes](../figures/mesh_{NAME}.png)

*Figure 1. The two meshes, expanded to the whole body and seen from above the free surface: the
wetted surface in blue, the interior surface in the waterplane in yellow. Left: low order,
{h['low_order_mesh']} divisions per quadrant edge graded towards the waterline, with the projected
interior surface. Right: higher order, one rational B-spline patch per quadrant (the exact sphere;
patch edges in black, span edges in grey) with ${h['higher_order_spans']} \times
{h['higher_order_spans']}$ spans of order 3, and a rational quarter-disk patch per quadrant as
interior surface.*

## The reference

The reference is Hulme (1982), tables 1 and 2: $A^{{(0)}}$, $B^{{(0)}}$ (heave) and $A^{{(1)}}$,
$B^{{(1)}}$ (surge), extracted into `tests/data/hulme1982_full.txt` by `scripts/extract_hulme.py`.
They are given to four digits. Rounding to four digits alone can leave a difference of up to
{tex(rounding, 0)} of the largest value, so agreement at that level is the best a comparison can show.

## Agreement with the tables

![Hemisphere against Hulme](../figures/hemisphere_hulme.png)

*Figure 2. Heave and surge added mass $A/m_0$ and damping $B/(\omega m_0)$ against $Ka$: the two
methods (lines) and Hulme's values (points).*

The curves lie on the published points over the whole range. Table 1 gives the largest difference
of each quantity.

*Table 1. Largest difference from Hulme (1982) over the 24 tabulated values $0.1 \le Ka \le 10$,
relative to the largest tabulated value of each quantity.*

{table(header, rows)}

## Does it converge?

Agreement on one mesh could be luck, so Figure 3 refines both meshes at three frequencies. For the
low order the rate of convergence is measured as well: the order $p$ observed on three meshes,
$v_i = v^* + c\,h_i^p$, and Richardson's extrapolation $v^*$, the limit of zero panel size that this
rate implies.

![Hemisphere convergence](../figures/hemisphere_convergence.png)

*Figure 3. Largest difference from Hulme (1982) of each quantity at $Ka = 0.5$, 1 and 2 as the meshes
are refined. Left: low order, $n \times n$ panels per quadrant, uniform in the two angles,
$n = {ns[0]}$ to {ns[-1]}. Right: higher order, one rational patch per quadrant with $s \times s$
spans, potential splines of order 3 (Gauss 3/4) and 4 (Gauss 4/5). The dashed line is the rounding
of the four-digit tables.*

The low order's difference falls from {size(lo[0])} ($n = {ns[0]}$) to {size(lo[-1])}
($n = {ns[-1]}$). The orders observed on the three finest meshes are {p[0]:.1f}, {p[1]:.1f},
{p[2]:.1f} and {p[3]:.1f} for $A_{{33}}$, $B_{{33}}$, $A_{{11}}$ and $B_{{11}}$, the $O(h^2)$ expected
of constant panels. Extrapolating with them lands within {tex(min(ex))} to {tex(max(ex))} of the
tables, the level of their rounding.

The higher order with splines of order 3 differs by {", ".join(e(x) for x in e3)} on
{", ".join(map(str, sp))} spans, and with order 4 by {", ".join(e(x) for x in e4)}. The difference
reaches the agreement with the tables at {s3} spans per quadrant (order 3) or {s4} (order 4), and
further refinement changes nothing visible, because the tables are the limit.

## The limits of zero and infinite frequency

At $\omega = 0$ the free surface acts as a rigid wall ($\partial\phi/\partial z = 0$), and as
$\omega \to \infty$ the potential vanishes on it ($\phi = 0$). In both limits the hemisphere and its
mirror image above $z = 0$ form a whole sphere in unbounded fluid, the image moving with the body or
against it. Heave at $\omega = \infty$ and surge at $\omega = 0$ are then the translating sphere, and
the added mass is half the sphere's, $A/m_0 = 1/2$ exactly. Heave at $\omega = 0$ (the two halves
moving apart) and surge at $\omega = \infty$ (the halves moving in opposite directions) are given by
Hulme (1982) as series: $L = 0.830951$ (his eq. 3.18) and $C_1 = 0.273239$ (his eq. 5.15), also the
first and last rows of his tables 1 and 2 to four digits. Both constants are extracted into
`tests/data/hulme1982_limits.txt` by `scripts/extract_hulme.py`. HydroBEM solves the two limits
directly, as frequencies of their own, on the meshes of Table 1.

*Table 2. The added mass $A/m_0$ at $\omega = 0$ and $\omega = \infty$: the reference, the low order
({h['low_order_mesh']} graded divisions, interior surface) and the higher order
(${h['higher_order_spans']} \times {h['higher_order_spans']}$ spans, interior surface), with the
difference relative to the reference value.*

{table(["Mode, limit", "Reference", "$A/m_0$", "Low order", "Difference", "Higher order", "Difference"], lim_rows)}

Both methods reproduce the four limits: the low order within {tex(max(lim['low_rel']))} and the
higher order within {tex(max(lim['higher_rel']))} of each value. On $n \times n$ panels per quadrant,
uniform in the angles, the low order's largest difference is {lim_conv} for
$n = {", ".join(map(str, lim['meshes']))}$. The orders observed are
{", ".join(f"{x:.1f}" for x in lim['p'])} (in the order of Table 2), and the extrapolation lands within
{tex(max(lim['extrapolated_rel']))} of the references. Surge at $\omega = \infty$ converges the
slowest in both methods: there the velocity of the mirrored sphere changes sign at the waterline, and
the potential is singular along it.

## Irregular frequencies

The integral equation of the exterior problem fails at irregular frequencies, the eigenfrequencies
of the water that would fill the inside of the hull; for this hemisphere the first three lie near
$Ka \approx 3.9$, 5.6 and 7.1. An interior surface, a lid over the waterplane inside the body,
removes them. This section checks that it does, on a mesh coarse enough for the effect to show.

![Irregular frequencies](../figures/hemisphere_irregular.png)

*Figure 4. Heave damping and surge added mass on a coarse mesh (16 divisions per quadrant edge,
graded towards the waterline), $0.5 \le Ka \le 10$, without and with an interior surface, with
Hulme's values.*

Without the interior surface the results depart from the tables near those frequencies; the largest
difference over the tabulated values is {size(max(irr['none']))}. With it the difference is
{size(max(irr['projected']))}, which is just the coarseness of this mesh. The three ways HydroBEM can
generate the interior surface (`projected`, `waterline`, `graded`) agree with each other within
{tex(gen)} of the largest value.

## Reproducing this page

The example [`examples/{NAME}`](../../../examples/{NAME}) solves the same body. This page, its
figures and its numbers are written by `validation/hemisphere.py` through
`validation/make_report.py`.
"""


if __name__ == "__main__":
    main("--quick" in sys.argv)
