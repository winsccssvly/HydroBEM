"""The sketches of the theory chapters (docs/theory/figures/<name>.png): one function per figure,
named as its file, prefixed by its chapter (ch01_, ch06_, ...).

    python scripts/theory_figures.py            # every figure
    python scripts/theory_figures.py ch08_      # the figures whose names start so

The sketches show the geometry of the problem (the fluid domain, its boundaries and normals, the
reference points of a body, the interior surface of a hull), constructions of the methods
(quadrature regions, maps, image points, the steps of a triangulation) and schematic drawings of
the special surfaces, the generalized modes and tanks and the threads of a solve; they are drawn
from the formulas and parameters stated in the chapters, not from runs of the engine, and carry
no results.
matplotlib is not a dependency of the package.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "theory" / "figures"

# the colours of the sketches, in a fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a8a8a"]
LIGHT = ["#cfe0f5", "#fadbc9", "#c8eedf", "#fbe7b5", "#f8dbe7", "#e4e4e4"]
INK = "#333333"

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 110,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "legend.fontsize": 8,
    }
)


def _save(fig, name):
    """Writes the figure as docs/theory/figures/<name>.png (no time stamp in the file)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", metadata={"Software": None})
    plt.close(fig)


def _bare(ax):
    """An axes without frame and ticks, equal scales."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _gauss(n):
    """Gauss-Legendre nodes on [0, 1]."""
    x, _ = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0)


# ---------------------------------------------------------------------------------------------
# chapter 6: the low-order method


def ch06_rankine_tiers():
    """The evaluation of the Rankine integrals of a panel by the distance of the field point from
    its centroid, in panel radii a (theory 짠6.1)."""
    v = np.array([[-1.0, -0.8], [1.1, -0.7], [0.9, 0.9], [-0.9, 0.7]])
    c = v.mean(axis=0)
    a = np.max(np.linalg.norm(v - c, axis=1))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.add_patch(Rectangle((-11 * a, -9.5 * a), 22 * a, 19 * a, color=LIGHT[5], zorder=0))
    for r, col in ((8, LIGHT[0]), (4, LIGHT[2]), (3, LIGHT[3])):
        ax.add_patch(Circle(c, r * a, facecolor=col, edgecolor=INK, lw=0.6, zorder=1))
    ax.add_patch(Circle(c, 2 * a, facecolor=LIGHT[1], edgecolor=INK, lw=0.6, zorder=1))
    ax.add_patch(Polygon(v, closed=True, facecolor="white", edgecolor=INK, lw=1.0, zorder=3))
    ax.plot(*c, "k.", zorder=4)
    ax.annotate("", xy=v[2], xytext=c, arrowprops={"arrowstyle": "->", "lw": 0.8}, zorder=4)
    ax.text(c[0] + 0.15, c[1] + 0.45, "$a$", zorder=4)
    labels = [
        (9.2, 0.0, r"$3 \times 3$"),
        (6.0, 0.0, r"$4 \times 4$"),
        (3.5, 0.6, r"$5 \times 5$"),
        (2.5, 1.2, r"$6 \times 6$"),
    ]
    for r, t, s in labels:
        ax.text(
            c[0] + r * a * math.cos(t), c[1] + r * a * math.sin(t), s, ha="center", va="center", fontsize=8, zorder=4
        )
    ax.text(c[0] - 0.1 * a, c[1] + 1.45 * a, "closed forms", ha="center", va="center", fontsize=8, zorder=4)
    for r in (2, 3, 4, 8):
        ax.text(c[0], c[1] - r * a - 0.1 * a, f"${r}a$", fontsize=8, ha="center", va="top", zorder=4)
    ax.set_xlim(-10.5 * a, 10.5 * a)
    ax.set_ylim(-9 * a, 9 * a)
    _bare(ax)
    _save(fig, "ch06_rankine_tiers")


def _signed_triangles(ax, x, poly):
    """The triangles from x to every edge of a polygon, shaded by the sign of their orientation."""
    n = len(poly)
    for k in range(n):
        p, q = poly[k], poly[(k + 1) % n]
        s = (p[0] - x[0]) * (q[1] - x[1]) - (p[1] - x[1]) * (q[0] - x[0])
        col, hatch = (LIGHT[0], None) if s > 0 else (LIGHT[1], "///")
        ax.add_patch(
            Polygon(
                [x, p, q], closed=True, facecolor=col, edgecolor=C[0] if s > 0 else C[1], lw=0.6, hatch=hatch, alpha=0.8
            )
        )


def ch06_log_correction():
    """The distance d that selects the pairs of the logarithmic correction, and the signed
    triangles of the in-plane integral (theory 짠6.2)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    # (a) side view: field point x, the source panel j with its centroid c_j and its image
    a1.axhline(0.0, color=C[0], lw=1.2)
    a1.text(-0.55, 0.06, "$z = 0$", color=C[0], ha="left", va="bottom")
    cj, x = np.array([2.2, -1.0]), np.array([0.0, -0.55])
    a1.plot([cj[0] - 0.45, cj[0] + 0.45], [cj[1] - 0.08, cj[1] + 0.08], color=INK, lw=2.5)
    a1.plot(*cj, "o", color=INK, ms=3)
    a1.text(cj[0] + 0.1, cj[1] - 0.22, "panel $j$, centroid $c_j$", ha="center", va="top", fontsize=8)
    ci = np.array([cj[0], -cj[1]])
    a1.plot(*ci, "o", mfc="white", mec=INK, ms=4)
    a1.text(ci[0] + 0.1, ci[1] + 0.08, r"image of $c_j$ in $z = 0$", fontsize=8, va="bottom", ha="center")
    a1.plot(*x, "o", color=C[1], ms=4)
    a1.text(x[0] - 0.1, x[1], "$x$", color=C[1], ha="right", va="center")
    a1.plot([x[0], ci[0]], [x[1], ci[1]], "--", color=INK, lw=0.8)
    mid = 0.5 * (x + ci)
    a1.text(mid[0] - 0.05, mid[1] + 0.1, "$r'$", ha="right", va="bottom")
    a1.annotate("", xy=(ci[0] + 0.6, ci[1]), xytext=(ci[0] + 0.6, x[1]), arrowprops={"arrowstyle": "<->", "lw": 0.8})
    a1.plot([x[0], ci[0] + 0.7], [x[1], x[1]], ":", color=INK, lw=0.6)
    a1.text(ci[0] + 0.68, 0.5 * (ci[1] + x[1]), r"$|z + \zeta |$", va="center")
    a1.text(
        -0.2, -1.65, r"$d = r' + |z + \zeta |$; the correction where $0.6\,(a_j/d)^2 > \epsilon$", fontsize=8, va="top"
    )
    a1.set_xlim(-0.6, 3.6)
    a1.set_ylim(-1.95, 1.4)
    _bare(a1)
    a1.set_title("(a) the distance of a pair", fontsize=9)
    # (b) a point and a panel in z = 0: signed triangles to every edge
    poly = np.array([[0.4, -0.6], [1.6, -0.4], [1.3, 0.7], [0.3, 0.5]])
    x = np.array([-0.6, -0.1])
    _signed_triangles(a2, x, poly)
    a2.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=INK, lw=1.4))
    a2.plot(*x, "o", color=C[1], ms=4)
    a2.text(x[0] - 0.06, x[1], "$x$", color=C[1], ha="right", va="center")
    a2.text(1.0, 0.0, "panel", ha="center", va="center", fontsize=8)
    a2.text(
        0.25,
        -0.95,
        "positive (blue) and negative (hatched) triangles;\ntheir sum is the panel",
        fontsize=8,
        ha="center",
        va="top",
    )
    a2.set_xlim(-0.9, 1.8)
    a2.set_ylim(-1.35, 0.95)
    _bare(a2)
    a2.set_title(r"(b) a point and a panel in $z = 0$", fontsize=9)
    _save(fig, "ch06_log_correction")


# ---------------------------------------------------------------------------------------------
# chapter 7: the higher-order method


def ch07_patch_map():
    """A patch: the parameter square with its spans and the Gauss points of one of its panels,
    and the map to the surface (theory 짠7)."""
    nu, nv, qo, qi = 4, 3, 3, 4
    fig = plt.figure(figsize=(7.6, 3.4))
    a1 = fig.add_subplot(1, 2, 1)
    for k in range(nu + 1):
        a1.plot([-1 + 2 * k / nu] * 2, [-1, 1], color=INK, lw=0.6)
    for k in range(nv + 1):
        a1.plot([-1, 1], [-1 + 2 * k / nv] * 2, color=INK, lw=0.6)
    a1.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor=INK, lw=1.4))
    iu, iv = 2, 1
    u0, v0, du, dv = -1 + 2 * iu / nu, -1 + 2 * iv / nv, 2 / nu, 2 / nv
    a1.add_patch(Rectangle((u0, v0), du, dv, color=LIGHT[0]))
    go, gi = _gauss(qo), _gauss(qi)
    uo, vo = np.meshgrid(u0 + du * go, v0 + dv * go)
    ui, vi = np.meshgrid(u0 + du * gi, v0 + dv * gi)
    a1.plot(ui.ravel(), vi.ravel(), ".", color=C[0], ms=4, label=f"inner (source) points, $q_i = {qi}$")
    a1.plot(uo.ravel(), vo.ravel(), "o", mfc="none", mec=C[1], ms=6, label=f"outer (test) points, $q_o = {qo}$")
    a1.set_xlabel("$u$")
    a1.set_ylabel("$v$")
    a1.set_xticks([-1, 0, 1])
    a1.set_yticks([-1, 0, 1])
    a1.set_aspect("equal")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    a1.set_title(rf"(a) parameter square, $n_u \times n_v = {nu} \times {nv}$ panels", fontsize=9)

    def x_of(u, v):  # a quarter of a vertical cylinder of radius 1 and draft 1
        t = 0.25 * math.pi * (u + 1)
        return np.cos(t), np.sin(t), 0.5 * (v - 1)

    a2 = fig.add_subplot(1, 2, 2, projection="3d")
    s = np.linspace(-1, 1, 41)
    for k in range(nu + 1):
        a2.plot(*x_of(np.full_like(s, -1 + 2 * k / nu), s), color=INK, lw=0.6)
    for k in range(nv + 1):
        a2.plot(*x_of(s, np.full_like(s, -1 + 2 * k / nv)), color=INK, lw=0.6)
    a2.plot(*x_of(ui.ravel(), vi.ravel()), ".", color=C[0], ms=4)
    a2.plot(*x_of(uo.ravel(), vo.ravel()), "o", mfc="none", mec=C[1], ms=6)
    a2.set_box_aspect((1, 1, 0.6), zoom=1.3)
    a2.view_init(elev=24, azim=-135)
    a2.set_xticks([])
    a2.set_yticks([])
    a2.set_zticks([])
    a2.set_title(r"(b) the patch $x(u, v)$", fontsize=9)
    _save(fig, "ch07_patch_map")


def _quadrisect(cells, u0, u1, v0, v1, xt, depth, leaf=1.5, maxdepth=7):
    """The leaves of the adaptive quadrisection of a flat square panel for a target xt: a cell
    is a leaf once its centre is more than `leaf` cell sizes (centre-corner distances) away."""
    c = np.array([0.5 * (u0 + u1), 0.5 * (v0 + v1), 0.0])
    size = 0.5 * math.hypot(u1 - u0, v1 - v0)
    if depth >= maxdepth or size * leaf < np.linalg.norm(c - xt):
        cells.append((u0, u1, v0, v1))
        return
    um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    for a, b, c_, d in ((u0, um, v0, vm), (um, u1, v0, vm), (um, u1, vm, v1), (u0, um, vm, v1)):
        _quadrisect(cells, a, b, c_, d, xt, depth + 1, leaf, maxdepth)


def ch07_static_tiers():
    """The rules of the static integrals of a panel by the distance of the target, and the
    adaptive quadrisection below 1.5 panel sizes (theory 짠7.1)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.3), gridspec_kw={"width_ratios": [1.5, 1]})
    bands = [
        (0.3, 1.5, LIGHT[1], "adaptive\nquadrisection,\n" + r"$6 \times 6$ leaves"),
        (1.5, 6, LIGHT[0], r"stored $6 \times 6$"),
        (6, 12, LIGHT[2], "inner rule\n" + r"($6 \times 6$ if $q_i < 4$)"),
        (12, 40, LIGHT[3], r"$3 \times 3$"),
    ]
    for lo, hi, col, t in bands:
        a1.axvspan(lo, hi, color=col)
        a1.text(math.sqrt(lo * hi), 0.5, t, ha="center", va="center", fontsize=8)
    a1.set_xscale("log")
    a1.set_xlim(0.3, 40)
    a1.set_ylim(0, 1)
    a1.set_yticks([])
    a1.set_xticks([0.5, 1.5, 6, 12, 30])
    a1.set_xticklabels(["0.5", "1.5", "6", "12", "30"])
    a1.set_xlabel("distance from the panel centre, in panel sizes")
    a1.set_title("(a) the tiers", fontsize=9)
    xt = np.array([0.55, 0.35, 0.04])
    cells = []
    _quadrisect(cells, -1, 1, -1, 1, xt, 0)
    for u0, u1, v0, v1 in cells:
        a2.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, facecolor=LIGHT[0], edgecolor=INK, lw=0.4))
    a2.plot(xt[0], xt[1], "o", color=C[1], ms=4)
    a2.text(xt[0] - 0.05, xt[1] + 0.12, "target", color=C[1], fontsize=8, ha="right")
    a2.set_xlim(-1.05, 1.05)
    a2.set_ylim(-1.05, 1.05)
    _bare(a2)
    a2.set_title("(b) quadrisection of a near panel", fontsize=9)
    _save(fig, "ch07_static_tiers")


def ch07_duffy():
    """The target's own panel: four rectangles with a corner at the target, two Duffy triangles
    each; the grading of an elongated triangle in q (theory 짠7.1)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    uc, vc = 0.35, 0.6
    a1.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, lw=1.4))
    a1.plot([uc, uc], [0, 1], color=INK, lw=0.8)
    a1.plot([0, 1], [vc, vc], color=INK, lw=0.8)
    for cu in (0, 1):
        for cv in (0, 1):
            a1.plot([uc, cu], [vc, cv], "--", color=INK, lw=0.6)
    # the Gauss points of one triangle: u = uc + a p, v = vc + b p q
    a, b = 1 - uc, 1 - vc
    g = _gauss(8)
    p, q = np.meshgrid(g, g)
    a1.plot(uc + a * p, vc + b * p * q, ".", color=C[0], ms=2.5)
    a1.add_patch(Polygon([[uc, vc], [1, vc], [1, 1]], closed=True, color=LIGHT[0], zorder=0))
    a1.plot(uc, vc, "o", color=C[1], ms=5)
    a1.text(uc - 0.03, vc + 0.03, "target", color=C[1], ha="right", va="bottom", fontsize=8)
    a1.text(0.5 * (uc + 1), 0.08, "four rectangles,\ntwo triangles each", ha="center", fontsize=8)
    a1.text(1.02, vc + 0.5 * b, r"$u = u_c + ap$" + "\n" + r"$v = v_c + bpq$", fontsize=8, va="center")
    a1.set_xlim(-0.05, 1.45)
    a1.set_ylim(-0.05, 1.05)
    _bare(a1)
    a1.set_title("(a) the panel of the target", fontsize=9)
    # an elongated triangle (rho = 4): the q range split at 1/rho, 2/rho, 1
    rho = 4.0
    A, B = 1.0, rho
    a2.add_patch(Polygon([[0, 0], [A, 0], [A, B]], closed=True, fill=False, edgecolor=INK, lw=1.2))
    cuts = [0, 1 / rho, 2 / rho, 1.0]
    for k, qk in enumerate(cuts[1:-1]):
        a2.plot([0, A], [0, B * qk], color=C[1], lw=0.8)
        a2.text(A + 0.05, B * qk, rf"$q = {['1', '2'][k]}/\rho$", color=C[1], fontsize=8, va="center")
    g4 = _gauss(4)
    for q0, q1 in zip(cuts[:-1], cuts[1:], strict=True):
        p, qq = np.meshgrid(g4, q0 + (q1 - q0) * g4)
        a2.plot(A * p, B * p * qq, ".", color=C[0], ms=2.5)
    a2.plot(0, 0, "o", color=C[1], ms=5)
    a2.text(-0.08, 0, "target", color=C[1], ha="right", va="center", fontsize=8)
    a2.text(0.5 * A, -0.35, r"leg along $p$", ha="center", va="top", fontsize=8)
    a2.text(A + 0.05, 0.85 * B, r"leg along $q$, $\rho$ times longer", fontsize=8, va="center")
    a2.set_xlim(-1.2, 3.4)
    a2.set_ylim(-0.7, B + 0.3)
    _bare(a2)
    a2.set_title(r"(b) grading in $q$, $\rho = 4$ ($4 \times 4$ points drawn)", fontsize=9)
    _save(fig, "ch07_duffy")


# ---------------------------------------------------------------------------------------------
# chapter 8: the free-surface Green function


def ch08_inf_regions():
    """The evaluation of the infinite-depth wave part over (X, Y) (theory 짠8.1)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.7))
    top = 40
    for ax, lim in ((a1, top), (a2, 3.2)):
        ax.add_patch(Rectangle((0, 0), 30, 30, color=LIGHT[0], zorder=0))
        ax.add_patch(Rectangle((30, 0), top, 30, color=LIGHT[2], zorder=0))
        ax.add_patch(Rectangle((0, 30), top + 10, top, color=LIGHT[3], zorder=0))
        ax.add_patch(Rectangle((0, 0), 2, 2, color=LIGHT[1], zorder=1))
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$X = KR$")
        ax.set_ylabel(r"$Y = K |z + \zeta |$")
    for k in range(0, 31, 5):
        a1.plot([k, k], [0, 30], color="white", lw=0.5, zorder=0.5)
        a1.plot([0, 30], [k, k], color="white", lw=0.5, zorder=0.5)
    a1.text(16, 15, "direct 2D tables\non unit tiles,\nChebyshev degree 6", ha="center", va="center", fontsize=8)
    a1.text(35, 15, "Hankel and\nStruve\nasymptotics", ha="center", va="center", fontsize=8)
    a1.text(20, 35, r"the same series without the $e^{-Y}$ terms", ha="center", va="center", fontsize=8)
    a1.annotate(r"$[0, 2)^2$", xy=(1, 1), xytext=(5, 6), fontsize=8, arrowprops={"arrowstyle": "->", "lw": 0.6})
    a1.set_xticks([0, 10, 20, 30, 40])
    a1.set_yticks([0, 10, 20, 30, 40])
    a1.set_title("(a) the whole quadrant", fontsize=9)
    for k in range(0, 4):
        a2.plot([k, k], [0, 3.2], color="white", lw=0.8, zorder=0.5)
        a2.plot([0, 3.2], [k, k], color="white", lw=0.8, zorder=0.5)
    a2.add_patch(Rectangle((0, 0), 2, 2, color=LIGHT[1], zorder=1))
    a2.add_patch(
        Polygon(
            [[0, 0], [0.1, 0.1], [0.1, 3.2], [0, 3.2]],
            closed=True,
            facecolor="none",
            edgecolor=C[4],
            hatch="////",
            lw=0.6,
            zorder=2,
        )
    )
    a2.text(
        1.0,
        1.0,
        "identity form:\n1D tables of\n" + r"$J_0, J_1, \tilde{Y}, \tilde{Y}'$" + "\n" + r"and 2D of $V$",
        ha="center",
        va="center",
        fontsize=8,
        zorder=3,
    )
    a2.text(2.6, 2.6, "direct\ntables", ha="center", va="center", fontsize=8)
    a2.annotate(
        r"near the axis: $X < 0.1$, $Y > X$",
        xy=(0.05, 2.6),
        xytext=(0.35, 2.85),
        fontsize=8,
        color=C[4],
        arrowprops={"arrowstyle": "->", "lw": 0.6, "color": C[4]},
        zorder=3,
    )
    a2.plot(0, 0, "o", color=C[1], ms=4, clip_on=False, zorder=4)
    a2.text(0.08, 0.08, "log", color=C[1], fontsize=8, zorder=4)
    a2.set_xticks([0, 1, 2, 3])
    a2.set_yticks([0, 1, 2, 3])
    a2.set_title("(b) near the origin", fontsize=9)
    _save(fig, "ch08_inf_regions")


def ch08_fin_images():
    """The six Rankine terms of finite depth as points on the vertical through the source, and the
    two forms of the kernel over (R/h, z) (theory 짠8.2)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    h, zeta, z, R = 1.0, -0.3, -0.65, 0.9
    pts = [
        (1, zeta, r"$\zeta$ (the source)"),
        (2, -zeta - 2 * h, r"$-\zeta - 2h$"),
        (3, -zeta, r"$-\zeta$"),
        (4, zeta - 2 * h, r"$\zeta - 2h$"),
        (5, zeta + 2 * h, r"$\zeta + 2h$"),
        (6, -zeta - 4 * h, r"$-\zeta - 4h$"),
    ]
    a1.axhspan(-h, 0, color=LIGHT[0], zorder=0)
    a1.axhline(0, color=C[0], lw=1.0)
    a1.axhline(-h, color=INK, lw=1.4)
    a1.text(1.55, 0.03, "$z = 0$", color=C[0], ha="right", va="bottom", fontsize=8)
    a1.text(1.55, -h - 0.03, "$z = -h$", ha="right", va="top", fontsize=8)
    a1.axvline(0, color=INK, lw=0.5, ls=":")
    for p, zp, t in pts:
        a1.plot(0, zp, "o", color=C[1] if p == 1 else INK, mfc=C[1] if p == 1 else "white", ms=5)
        a1.text(-0.08, zp, t, ha="right", va="center", fontsize=8)
        a1.plot([0, R], [zp, z], color=INK, lw=0.4, ls="--")
        a1.text(0.12, zp + 0.06, f"$p = {p}$", fontsize=7, color=INK, va="bottom")
    a1.plot(R, z, "o", color=C[0], ms=5)
    a1.text(R + 0.08, z - 0.12, "$x$", color=C[0], va="top")
    a1.set_xlim(-1.15, 1.6)
    a1.set_ylim(-4.5, 2.1)
    _bare(a1)
    a1.set_title(r"(a) the points of the terms $1/r_p$", fontsize=9)
    a2.add_patch(Rectangle((0, -1), 0.5, 1, color=LIGHT[1]))
    a2.add_patch(Rectangle((0.5, -1), 1.5, 1, color=LIGHT[2]))
    a2.axvline(0.5, color=INK, lw=1.0)
    a2.text(
        0.25,
        -0.5,
        "six Rankine\nterms, four Ft,\n" + r"$Q$ series" + "\n" + r"in $s = (R/h)^2$," + "\n" + r"$m \leq 9$",
        ha="center",
        va="center",
        fontsize=8,
    )
    a2.text(
        1.25,
        -0.5,
        r"eigenfunction series:" + "\n" + r"$Y_0$ term and $K_0$ terms," + "\n" + r"to $k_n R \geq 18$",
        ha="center",
        va="center",
        fontsize=8,
    )
    a2.set_xlim(0, 2)
    a2.set_ylim(-1, 0)
    a2.set_xticks([0, 0.5, 1, 1.5, 2])
    a2.set_xticklabels(["0", "1/2", "1", "3/2", "2"])
    a2.set_yticks([-1, -0.5, 0])
    a2.set_xlabel("$R/h$")
    a2.set_ylabel("$z/h$, $\\zeta/h$")
    a2.set_title(r"(b) the two forms of $\mathrm{Re} G$", fontsize=9)
    _save(fig, "ch08_fin_images")


def ch08_channel():
    """The wall images of a source in a channel and how each is summed (theory 짠8.5)."""
    W, eta, Me, M = 1.0, 0.22, 4, 7
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    for k in range(-M, M + 1):
        ax.axvline(
            (k + 0.5) * W, color=INK if abs(k + 0.5) == 0.5 else "#bbbbbb", lw=1.6 if abs(k + 0.5) == 0.5 else 0.6
        )
    ax.axvspan(-0.5 * W, 0.5 * W, color=LIGHT[0], zorder=0)
    for m in range(-M, M + 1):
        y = (eta if m % 2 == 0 else -eta) + m * W
        if m == 0:
            col, mk = C[1], "o"
        elif abs(m) == 1:
            col, mk = C[0], "o"
        elif abs(m) <= Me:
            col, mk = C[2], "s"
        else:
            col, mk = C[5], "^"
        ax.plot(y, 0, mk, color=col, ms=6)
        ax.text(y, -0.12, f"${m}$", ha="center", va="top", fontsize=7)
    ax.text(0, 0.42, "channel", ha="center", fontsize=8)
    ax.text(-0.5 * W - 0.05, 0.62, r"$y = -W/2$", ha="right", fontsize=8)
    ax.text(0.5 * W + 0.05, 0.62, r"$y = W/2$", ha="left", fontsize=8)
    ax.text(-M - 0.5, -0.32, "$m$:", ha="left", va="top", fontsize=7)
    handles = [
        plt.Line2D([], [], ls="none", marker="o", color=C[1], label="the source ($m = 0$): full panel integrals"),
        plt.Line2D([], [], ls="none", marker="o", color=C[0], label=r"$m = \pm 1$: full panel integrals"),
        plt.Line2D(
            [], [], ls="none", marker="s", color=C[2], label=r"$2 \leq |m |\leq M_e$: evanescent series at the centroid"
        ),
        plt.Line2D([], [], ls="none", marker="^", color=C[5], label=r"$|m |> M_e$: propagating mode only"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.75), ncol=2)
    ax.text(
        0,
        -0.42,
        r"the propagating mode of every $|m |\geq 2$: the lattice-sum tail",
        ha="center",
        fontsize=8,
        color=INK,
        bbox={"color": "white", "pad": 1},
    )
    ax.set_xlim(-M - 0.6, M + 0.6)
    ax.set_ylim(-0.6, 0.8)
    ax.set_yticks([])
    ax.set_xlabel("$y / W$ (image $m$ at $\\eta + mW$ for even $m$, $-\\eta + mW$ for odd $m$)")
    ax.spines["left"].set_visible(False)
    _save(fig, "ch08_channel")


# ---------------------------------------------------------------------------------------------
# chapter 14: geometry preprocessing


def ch14_midpoint_plane():
    """A skew quadrilateral and its projection onto the plane through its side midpoints
    (theory 짠14)."""
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.12], [1.0, 1.0, -0.02], [0.0, 1.0, 0.14]])
    n = np.cross(v[2] - v[0], v[3] - v[1])
    n /= np.linalg.norm(n)
    c = v.mean(axis=0)
    w = v - np.outer((v - c) @ n, n)
    mids = 0.5 * (v + np.roll(v, -1, axis=0))
    fig = plt.figure(figsize=(5.2, 3.8))
    ax = fig.add_subplot(projection="3d")
    loop = np.vstack([v, v[:1]])
    ax.plot(*loop.T, color=INK, lw=1.2, label="raw vertices")
    loop = np.vstack([w, w[:1]])
    ax.plot(*loop.T, color=C[0], lw=1.6, label="projected panel")
    ml = np.vstack([mids, mids[:1]])
    ax.plot(*ml.T, ":", color=C[1], lw=1.0, label="side midpoints")
    ax.plot(*mids.T, "o", color=C[1], ms=4)
    for a, b in zip(v, w, strict=True):
        ax.plot(*np.array([a, b]).T, "--", color=INK, lw=0.6)
    ax.quiver(*c, *(0.25 * n), color=C[2], lw=1.2, label=r"normal $(v_2 - v_0) \times (v_3 - v_1)$")
    for k in range(4):
        ax.text(*(v[k] + np.array([-0.06, -0.06, 0.03])), f"$v_{k}$", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_zlim(-0.2, 0.4)
    ax.set_box_aspect((1.1, 1.1, 0.6), zoom=1.1)
    ax.view_init(elev=20, azim=-58)
    fig.legend(loc="lower center", ncol=2)
    _save(fig, "ch14_midpoint_plane")


def ch14_triangulation():
    """The triangulation of a region with holes: the bridge, an ear, an edge flip (theory 짠14)."""
    outer = np.array([[0, 0], [4, 0], [4.6, 1.5], [4, 3], [0.5, 3.2], [-0.4, 1.6]])
    hole = np.array([[1.2, 1.0], [1.4, 2.1], [2.4, 2.2], [2.6, 1.1]])[::-1]
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.9))
    a1, a2, a3 = axs
    for ax in (a1, a2):
        ax.add_patch(Polygon(outer, closed=True, facecolor=LIGHT[0], edgecolor=INK, lw=1.0))
        ax.add_patch(Polygon(hole, closed=True, facecolor="white", edgecolor=INK, lw=1.0))
    hr = hole[np.argmax(hole[:, 0])]
    tgt = outer[2]
    a1.plot([hr[0], tgt[0]], [hr[1], tgt[1]], color=C[1], lw=1.6)
    a1.plot(*hr, "o", color=C[1], ms=4)
    a1.text(hr[0] - 0.1, hr[1] - 0.3, "rightmost\nvertex", fontsize=7, ha="center", va="top", color=C[1])
    a1.text(3.3, 1.55, "bridge", fontsize=8, color=C[1], ha="center", va="bottom")
    a1.text(0.4, 1.6, "region", fontsize=8)
    a1.text(1.9, 1.6, "hole", fontsize=8, ha="center")
    a1.set_title("(a) the hole bridged", fontsize=9)
    # an ear of the bridged polygon: the corner at outer[3]
    ear = np.array([outer[2], outer[3], outer[4]])
    a2.add_patch(Polygon(ear, closed=True, facecolor=LIGHT[3], edgecolor=C[3], lw=1.0))
    a2.plot([hr[0], tgt[0]], [hr[1], tgt[1]], color=C[1], lw=1.0)
    a2.text(*(ear.mean(axis=0) + np.array([0, 0.25])), "an ear", fontsize=8, ha="center")
    a2.set_title("(b) ears cut off", fontsize=9)
    # an edge flip: the diagonal of a convex quadrilateral is flipped when the fourth point lies
    # inside the circumcircle of a triangle
    q = np.array([[0.0, 0.0], [2.0, -0.4], [2.6, 1.2], [0.4, 1.0]])
    for k, (a, b) in enumerate(((0, 2), (1, 3))):
        off = np.array([0.0, 0.0]) if k == 0 else np.array([3.4, 0.0])
        a3.add_patch(Polygon(q + off, closed=True, facecolor=LIGHT[0], edgecolor=INK, lw=1.0))
        a3.plot(*(q[[a, b]] + off).T, color=C[1] if k == 0 else C[2], lw=1.4)
    # the circumcircle of the triangle (q0, q1, q2)
    (ax_, ay), (bx, by), (cx, cy) = q[0], q[1], q[2]
    d = 2 * (ax_ * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax_**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax_**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax_ - cx) + (cx**2 + cy**2) * (bx - ax_)) / d
    a3.add_patch(Circle((ux, uy), math.hypot(ax_ - ux, ay - uy), fill=False, ls=":", edgecolor=C[1], lw=0.8))
    a3.annotate("", xy=(3.2, 0.4), xytext=(2.75, 0.4), arrowprops={"arrowstyle": "->", "lw": 0.8})
    a3.text(1.3, -1.25, "illegal", color=C[1], fontsize=8, ha="center")
    a3.text(4.7, -1.25, "flipped", color=C[2], fontsize=8, ha="center")
    a3.set_title("(c) an edge flip", fontsize=9)
    for ax in axs:
        ax.autoscale_view()
        _bare(ax)
    a1.set_xlim(-0.6, 4.8)
    a2.set_xlim(-0.6, 4.8)
    a3.set_xlim(-0.5, 6.3)
    a3.set_ylim(-1.5, 2.3)
    for ax in (a1, a2):
        ax.set_ylim(-0.3, 3.5)
    _save(fig, "ch14_triangulation")


def _refine(boundary, lmax, min_angle=25.0):
    """Delaunay refinement of a convex polygon (counter-clockwise): the vertex at the circumcentre
    of the worst triangle whose longest edge exceeds lmax or whose smallest angle is below
    min_angle, or the midpoint of the boundary edge that circumcentre encroaches upon (in its
    diametral circle) or lies beyond. The points and the triangles."""
    from scipy.spatial import Delaunay

    bnd = [np.asarray(p, dtype=float) for p in boundary]
    inner = []
    while True:
        pts = np.array(bnd + inner)
        tris = Delaunay(pts).simplices
        worst, bad = None, 0.0
        for t in tris:
            a, b, c = pts[t]
            e = [np.linalg.norm(b - c), np.linalg.norm(c - a), np.linalg.norm(a - b)]
            ang = min(math.degrees(math.acos(np.clip((e[(k + 1) % 3] ** 2 + e[(k + 2) % 3] ** 2 - e[k] ** 2)
                                                     / (2 * e[(k + 1) % 3] * e[(k + 2) % 3]), -1, 1)))
                      for k in range(3))  # fmt: skip
            score = max(max(e) / lmax, math.sin(math.radians(min_angle)) / math.sin(math.radians(ang)))
            if score > 1.0 + 1e-9 and score > bad:
                worst, bad = t, score
        if worst is None:
            return pts, tris
        a, b, c = pts[worst]
        d = 2 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        b2, c2 = np.sum((b - a) ** 2), np.sum((c - a) ** 2)
        cc = a + np.array([(c[1] - a[1]) * b2 - (b[1] - a[1]) * c2, (b[0] - a[0]) * c2 - (c[0] - a[0]) * b2]) / d
        n = len(bnd)
        hit = None
        for k in range(n):
            p, q = bnd[k], bnd[(k + 1) % n]
            beyond = (q[0] - p[0]) * (cc[1] - p[1]) - (q[1] - p[1]) * (cc[0] - p[0]) < 0
            if beyond or np.dot(p - cc, q - cc) < 0:
                hit = k
                break
        if hit is None:
            inner.append(cc)
        else:
            bnd.insert(hit + 1, 0.5 * (bnd[hit] + bnd[(hit + 1) % n]))


def ch14_interior_generators():
    """The three generated interior surfaces of an elliptic waterplane (theory 짠14)."""
    ns = 24
    th = 2 * math.pi * np.arange(ns) / ns
    ea, eb = 1.4, 1.0  # the semi-axes of the waterline
    wl = np.c_[ea * np.cos(th), eb * np.sin(th)]
    seg = np.mean(np.linalg.norm(np.roll(wl, -1, axis=0) - wl, axis=1))
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.4))
    # projected: the panels of a half-ellipsoid (6 rings, graded towards the waterline) seen from above
    a1 = axs[0]
    rings = np.sin(0.5 * math.pi * np.sin(0.5 * math.pi * np.arange(7) / 6))
    for i in range(6):
        for j in range(ns):
            r0, r1 = rings[i], rings[i + 1]
            t0, t1 = th[j], th[(j + 1) % ns] + (2 * math.pi if j == ns - 1 else 0)
            quad = [
                [ea * r0 * math.cos(t0), eb * r0 * math.sin(t0)],
                [ea * r1 * math.cos(t0), eb * r1 * math.sin(t0)],
                [ea * r1 * math.cos(t1), eb * r1 * math.sin(t1)],
                [ea * r0 * math.cos(t1), eb * r0 * math.sin(t1)],
            ]
            a1.add_patch(Polygon(quad, closed=True, facecolor=LIGHT[3], edgecolor=INK, lw=0.4))
    a1.set_title("(a) projected", fontsize=9)
    # waterline: the triangulation of the waterline polygon with Delaunay refinement
    a2 = axs[1]
    pts, tris = _refine(wl, 2 * seg)
    for t in tris:
        a2.add_patch(Polygon(pts[t], closed=True, facecolor=LIGHT[3], edgecolor=INK, lw=0.4))
    a2.set_title("(b) waterline", fontsize=9)
    # graded: a fan from the centroid, the rays divided at sin(pi k / 2N)
    a3 = axs[2]
    N = math.ceil(np.mean(np.linalg.norm(wl, axis=1)) / seg)
    t = np.sin(math.pi * np.arange(N + 1) / (2 * N))
    for j in range(ns):
        p, q = wl[j], wl[(j + 1) % ns]
        for k in range(N):
            cell = [t[k] * p, t[k + 1] * p, t[k + 1] * q, t[k] * q]
            a3.add_patch(Polygon(cell, closed=True, facecolor=LIGHT[3], edgecolor=INK, lw=0.4))
    a3.set_title(rf"(c) graded, $N = {N}$", fontsize=9)
    for ax in axs:
        ax.add_patch(Polygon(wl, closed=True, fill=False, edgecolor=C[0], lw=1.2))
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.05, 1.05)
        _bare(ax)
    _save(fig, "ch14_interior_generators")


def ch14_trim():
    """A patch cut by the waterline, in its parameter square: the lines of constant u, the split
    into wholly submerged, cut and dry parts, and the remap of the cut part (theory 짠14)."""

    # the height of the patch z = (v + 1)/2 - c0 + c1 u: increasing in v, the waterline crosses the square
    c0, c1 = 0.7, 0.5
    u_sub = (c0 - 1.0) / c1  # z(u, +1) = 0: the lines to the left are wholly submerged
    u_dry = min(c0 / c1, 1.0)  # z(u, -1) = 0: beyond, wholly dry (here outside the square)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    a1.add_patch(Rectangle((-1, -1), u_sub + 1, 2, color=LIGHT[0]))
    a1.add_patch(Rectangle((u_sub, -1), u_dry - u_sub, 2, color=LIGHT[2]))
    uu = np.linspace(u_sub, u_dry, 50)
    vw = 2 * (c0 - c1 * uu) - 1
    a1.fill_between(uu, vw, 1, color=LIGHT[5])
    a1.plot(uu, vw, color=C[0], lw=1.6)
    for u in np.linspace(-0.95, 0.95, 14):
        vt = min(1.0, 2 * (c0 - c1 * u) - 1)
        a1.plot([u, u], [-1, vt], color=INK, lw=0.6)
    a1.axvline(u_sub, color=C[1], lw=1.0, ls="--")
    a1.text(-0.8, 0.0, "wholly\nsubmerged", ha="center", fontsize=8, rotation=90, va="center")
    a1.text(0.3, -0.7, "cut", ha="center", fontsize=8)
    a1.text(0.6, 0.6, "dry", ha="center", fontsize=8)
    a1.text(0.05, 0.42, "waterline", color=C[0], fontsize=8, rotation=-45)
    a1.text(u_sub, 1.04, "split", color=C[1], fontsize=8, ha="center", va="bottom")
    a1.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor=INK, lw=1.2))
    a1.set_xlabel("$u$")
    a1.set_ylabel("$v$")
    a1.set_aspect("equal")
    a1.set_xlim(-1.05, 1.05)
    a1.set_ylim(-1.05, 1.2)
    a1.set_xticks([-1, 0, 1])
    a1.set_yticks([-1, 0, 1])
    a1.set_title("(a) the source patch", fontsize=9)
    # the cut element: its own square, each line of constant u from v = -1 to the waterline
    a2.add_patch(Rectangle((-1, -1), 2, 2, facecolor=LIGHT[2], edgecolor=INK, lw=1.2))
    for u in np.linspace(-0.95, 0.95, 10):
        a2.plot([u, u], [-1, 1], color=INK, lw=0.6)
    a2.plot([-1, 1], [1, 1], color=C[0], lw=2.0)
    a2.text(0, 1.04, "the waterline", color=C[0], ha="center", va="bottom", fontsize=8)
    a2.text(0, -1.08, "the submerged ends", ha="center", va="top", fontsize=8)
    a2.set_aspect("equal")
    a2.set_xlim(-1.05, 1.05)
    a2.set_ylim(-1.25, 1.2)
    a2.set_xticks([-1, 0, 1])
    a2.set_yticks([-1, 0, 1])
    a2.set_xlabel("$u$ of the element")
    a2.set_ylabel("$v$ of the element")
    a2.set_title("(b) the cut part, remapped", fontsize=9)
    _save(fig, "ch14_trim")


def ch14_mesh_quality():
    """The edge check of the mesh quality: a T-junction met at the points 1/8, 3/8, 5/8, 7/8 of an
    edge, and a gap above the tolerance (theory 짠14)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw={"width_ratios": [1.3, 1]})
    big = np.array([[0, 0], [2, 0], [2, 1], [0, 1]])
    s1 = np.array([[0, -1], [1, -1], [1, 0], [0, 0]])
    s2 = np.array([[1, -1], [2, -1], [2, 0], [1, 0]])
    for p, col in ((big, LIGHT[0]), (s1, LIGHT[2]), (s2, LIGHT[3])):
        a1.add_patch(Polygon(p, closed=True, facecolor=col, edgecolor=INK, lw=0.8))
    for f in (1 / 8, 3 / 8, 5 / 8, 7 / 8):
        a1.plot(2 * f, 0, "o", color=C[1], ms=4, zorder=3)
    a1.annotate("", xy=(0.4, 0.12), xytext=(1.6, 0.12), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": C[0]})
    a1.annotate("", xy=(0.85, -0.12), xytext=(0.15, -0.12), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": C[2]})
    a1.annotate("", xy=(1.85, -0.12), xytext=(1.15, -0.12), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": C[3]})
    a1.text(1.0, 0.55, "one panel", ha="center", fontsize=8)
    a1.text(0.5, -0.6, "two smaller panels", ha="left", fontsize=8)
    a1.text(1.0, 1.08, r"points at $1/8, 3/8, 5/8, 7/8$ of the edge", ha="center", va="bottom", fontsize=8, color=C[1])
    a1.set_xlim(-0.1, 2.1)
    a1.set_ylim(-1.1, 1.3)
    _bare(a1)
    a1.set_title("(a) a T-junction: closed", fontsize=9)
    gap = 0.12
    a2.add_patch(Polygon([[0, 0], [2, 0], [2, 1], [0, 1]], closed=True, facecolor=LIGHT[0], edgecolor=INK, lw=0.8))
    a2.add_patch(
        Polygon([[0, -1], [2, -1], [2, -gap], [0, -gap]], closed=True, facecolor=LIGHT[2], edgecolor=INK, lw=0.8)
    )
    a2.axhspan(-0.03, 0.0, xmin=0.05, xmax=0.95, color=C[1], alpha=0.35)
    for f in (1 / 8, 3 / 8, 5 / 8, 7 / 8):
        a2.plot(2 * f, 0, "o", color=C[1], ms=4, zorder=3)
    a2.annotate(
        "gap above the tolerance:\nan open edge, reported",
        xy=(1.0, -0.06),
        xytext=(1.0, 0.45),
        ha="center",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.6},
    )
    a2.set_xlim(-0.1, 2.1)
    a2.set_ylim(-1.1, 1.3)
    _bare(a2)
    a2.set_title(r"(b) a gap (tolerance $10^{-3}$ of the edge, exaggerated)", fontsize=9)
    _save(fig, "ch14_mesh_quality")


# ---------------------------------------------------------------------------------------------
# chapters 9, 10 and 13: the special surfaces, generalized modes and tanks, the threads

# the colours of the sketches of a body in the water
WATER = "#dceefb"
BODY = "#b9b9b9"
EDGE = "#333333"
SURF = "#2a78d6"  # the special surface of a sketch
ACCENT = "#eb6834"
AIR = "#fff6dc"


def _save_sketch(fig, name):
    """Writes the figure as docs/theory/figures/<name>.png (no time stamp: the bytes are stable)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", metadata={"Software": None}, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def _arrow(ax, a, b, color=EDGE, lw=1.2, style="-|>", scale=10):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=scale, color=color, lw=lw))


def _sea(ax, x0, x1, z_bottom, wavy=True):
    """Water from z_bottom to the free surface z = 0 (a gentle wave outside the bodies)."""
    ax.add_patch(Rectangle((x0, z_bottom), x1 - x0, -z_bottom, color=WATER, zorder=0))
    x = np.linspace(x0, x1, 400)
    return x, (0.06 * np.sin(2.2 * x) if wavy else 0 * x)


def ch09_damping_lid():
    """A damping lid on the free surface of a moonpool (theory §9.1)."""
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    x, eta = _sea(ax, -5, 5, -3)
    hull = [(-3.2, 0.5), (-0.8, 0.5), (-0.8, -1.6), (-3.2, -1.6)]
    for sgn in (1, -1):
        ax.add_patch(Polygon([(sgn * px, pz) for px, pz in hull], fc=BODY, ec=EDGE, zorder=2))
    out = np.abs(x) > 3.2
    ax.plot(np.where(out, x, np.nan), np.where(out, eta, np.nan), color=EDGE)
    ax.plot([-0.8, 0.8], [0, 0], color=SURF, lw=4, solid_capstyle="butt", zorder=3)
    ax.text(0, 0.2, "lid $L$", ha="center", color=SURF)
    ax.text(
        0,
        -0.55,
        r"$\phi_z = \kappa\phi$" "\n" r"$\kappa = \omega^2/(g + i\omega\epsilon)$",
        ha="center",
        va="top",
        color=SURF,
    )
    ax.text(4.1, -0.45, "free surface\n" r"$\phi_z = K\phi$", ha="center", va="top")
    ax.text(-2.0, -0.6, "hull", ha="center")
    ax.text(-4.1, -2.4, "water", ha="center", color="#5a7f9c")
    ax.text(0, -2.4, "moonpool", ha="center", color="#5a7f9c")
    _blank(ax, (-5, 5), (-3, 1.0))
    _save_sketch(fig, "ch09_damping_lid")


def ch09_pressure_surface():
    """A prescribed-pressure surface under the air of a chamber (theory §9.3)."""
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x, eta = _sea(ax, -5, 5, -3)
    # a chamber: two walls and a roof, the water surface inside it under the air pressure
    ax.add_patch(
        Polygon(
            [(-2.6, 1.3), (2.6, 1.3), (2.6, -1.4), (2.2, -1.4), (2.2, 0.9), (-2.2, 0.9), (-2.2, -1.4), (-2.6, -1.4)],
            fc=BODY,
            ec=EDGE,
            zorder=2,
        )
    )
    ax.add_patch(Rectangle((-2.2, 0), 4.4, 0.9, fc=AIR, ec="none", zorder=1))
    out = np.abs(x) > 2.6
    ax.plot(np.where(out, x, np.nan), np.where(out, eta, np.nan), color=EDGE)
    ax.plot([-2.2, 2.2], [0, 0], color=SURF, lw=4, solid_capstyle="butt", zorder=3)
    for px in np.linspace(-1.6, 1.6, 5):
        _arrow(ax, (px, 0.75), (px, 0.12), color=ACCENT)
    ax.text(0, 1.05, "roof", ha="center")
    ax.text(0, -0.25, "pressure surface $S_p$", ha="center", va="top", color=SURF)
    ax.text(0, -0.75, r"$-\omega^2\phi + g\phi_z = -i\omega p/\rho$", ha="center", va="top", color=SURF)
    ax.text(3.0, 0.75, r"$p = -\rho g \sum_m \xi_m n_m(x, y)$", ha="left", color=ACCENT)
    ax.text(-3.8, 0.25, "free surface", ha="center")
    _blank(ax, (-5, 5.6), (-3, 1.6))
    _save_sketch(fig, "ch09_pressure_surface")


def ch09_thin_sheets():
    """A sheet with its two faces, and a body with a rigid sheet, a porous damper and a fixed damper
    (theory §9.5)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1, 1.3]})
    # (a) the faces of a sheet
    s = np.linspace(-1.5, 1.5, 100)
    a.plot(s, 0.12 * s**2, color=SURF, lw=3)
    _arrow(a, (0.0, 0.0), (0.0, 0.9))
    a.text(0.08, 0.75, "$n$")
    a.text(-1.0, 0.75, r"$+$ side, $\phi^+$", ha="center")
    a.text(-1.0, -0.55, r"$-$ side, $\phi^-$", ha="center")
    a.text(1.55, 0.27, "$S_d$", color=SURF)
    a.text(0, -1.25, r"$\mu = \phi^+ - \phi^-$,  $\partial\phi^\pm/\partial n = V_n$", ha="center")
    a.set_title("(a) a sheet", fontsize=9)
    _blank(a, (-2, 2), (-1.5, 1.2))
    # (b) a spar with a heave plate (rigid sheet), a porous skirt and a fixed damper at rest
    x, eta = _sea(b, -4, 4, -4.2)
    out = np.abs(x) > 0.5
    b.plot(np.where(out, x, np.nan), np.where(out, eta, np.nan), color=EDGE)
    b.add_patch(Rectangle((-0.5, -2.8), 1.0, 3.4, fc=BODY, ec=EDGE, zorder=2))
    b.plot([-1.6, 1.6], [-2.8, -2.8], color=SURF, lw=3, zorder=3)
    b.plot([-1.6, -1.6], [-2.8, -2.0], color=ACCENT, lw=2.5, ls=(0, (1.5, 1.2)), zorder=3)
    b.plot([1.6, 1.6], [-2.8, -2.0], color=ACCENT, lw=2.5, ls=(0, (1.5, 1.2)), zorder=3)
    b.plot([2.6, 3.6], [-3.6, -3.6], color=ACCENT, lw=2.5, ls=(0, (1.5, 1.2)))
    b.text(0, -3.15, "rigid sheet (heave plate)", ha="center", va="top", color=SURF)
    b.text(-1.75, -1.75, "damper of\nthe body", ha="center", va="bottom", color=ACCENT)
    b.text(3.1, -3.45, "fixed\ndamper", ha="center", va="bottom", color=ACCENT)
    b.text(0, 0.9, "spar", ha="center")
    b.set_title("(b) solid and porous sheets", fontsize=9)
    _blank(b, (-4, 4), (-4.2, 1.3))
    fig.tight_layout()
    _save_sketch(fig, "ch09_thin_sheets")


def ch09_plate_owc():
    """A free-surface plate: the light piston on the water column of an oscillating water column
    (theory §9.7)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x, eta = _sea(ax, -5, 5, -3.6)
    for x0 in (1.0, -1.5):
        ax.add_patch(Rectangle((x0, -2.8), 0.5, 3.6, fc=BODY, ec=EDGE, zorder=2))
    out = np.abs(x) > 1.5
    ax.plot(np.where(out, x, np.nan), np.where(out, eta, np.nan), color=EDGE)
    ax.plot([-1.0, 1.0], [0, 0], color=SURF, lw=4, solid_capstyle="butt", zorder=3)
    _arrow(ax, (0, 0.05), (0, 0.75), color=ACCENT)
    _arrow(ax, (0.55, -0.05), (0.55, -0.6))
    ax.text(0.65, -0.55, r"$n = -z$")
    ax.text(0.1, 0.55, r"$\xi_3$", color=ACCENT)
    ax.text(-0.35, -0.45, "plate $P$", ha="center", color=SURF)
    ax.text(0, -1.0, r"$\partial\phi_j/\partial n = n_j$", ha="center", va="top", color=SURF)
    ax.text(1.65, -1.5, "column wall", ha="left")
    ax.text(0, -3.25, "open bottom", ha="center", color="#5a7f9c")
    ax.text(3.3, -0.45, r"free surface $\phi_z = K\phi$", ha="center", va="top")
    _blank(ax, (-5, 5), (-3.6, 1.1))
    _save_sketch(fig, "ch09_plate_owc")


def ch10_generalized_mode():
    """A generalized mode: a bending shape of a barge given as a displacement field (theory §10.1)."""
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    L, D = 5.0, 0.6
    xs = np.linspace(-L, L, 200)
    w = 0.35 * (1.5 * (xs / L) ** 2 - 0.5)  # a Legendre polynomial P2 in x / L
    ax.plot([-6, 6], [0, 0], color="#5a7f9c", lw=0.8)
    ax.add_patch(Rectangle((-L, -D), 2 * L, 2 * D, fc="none", ec="#999999", ls="--"))
    ax.add_patch(Polygon(np.r_[np.c_[xs, w + D], np.c_[xs[::-1], w[::-1] - D]], fc=BODY, ec=EDGE, alpha=0.9))
    for px in (-L, -L / 2, 0, L / 2, L):
        pw = 0.35 * (1.5 * (px / L) ** 2 - 0.5)
        if abs(pw) > 0.05:
            _arrow(ax, (px, -D), (px, -D + pw), color=ACCENT, scale=8)
    ax.text(0, -1.25, r"$w_j(x) = P_2(x/L)$: the vertical displacement of mode $j$", ha="center", color=ACCENT)
    ax.text(5.3, 0.1, "$z = 0$", color="#5a7f9c")
    ax.text(-L, 1.05, "rest (dashed) and the displaced hull", ha="left")
    _blank(ax, (-6, 6.2), (-1.5, 1.3))
    _save_sketch(fig, "ch10_generalized_mode")


def ch10_tank():
    """An internal tank in a hull: its own fluid domain with its free surface z = Z_T (theory §10.2)."""
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x, eta = _sea(ax, -5, 5, -2.6)
    out = np.abs(x) > 3.0
    ax.plot(np.where(out, x, np.nan), np.where(out, eta, np.nan), color=EDGE)
    ax.add_patch(Polygon([(-3, 1.2), (3, 1.2), (3, -1.5), (-3, -1.5)], fc=BODY, ec=EDGE, zorder=2))
    ax.add_patch(Rectangle((-1.8, -1.1), 3.6, 1.6, fc="white", ec=EDGE, zorder=3))
    ax.add_patch(Rectangle((-1.8, -1.1), 3.6, 1.1, fc=WATER, ec="none", zorder=3))
    ax.plot([-1.8, 1.8], [0, 0], color=SURF, lw=2.5, zorder=4)
    ax.plot([-5, -3.4], [0, 0], color="none")
    ax.text(0, 0.1, "$z = Z_T$", ha="center", va="bottom", color=SURF, zorder=5)
    ax.text(0, -0.55, r"tank fluid $\rho_T$, $K\phi = \phi_z$", ha="center", va="center", zorder=5)
    for p, d in (((-1.8, -0.75), (0.45, 0)), ((1.8, -0.75), (-0.45, 0)), ((0.9, -1.1), (0, 0.4))):
        _arrow(ax, p, (p[0] + d[0], p[1] + d[1]), scale=8)
    ax.text(-2.4, 0.75, "hull", ha="center")
    ax.text(4.1, 0.25, "sea, $z = 0$", ha="center")
    ax.text(0, -2.2, "walls: normals into the tank fluid", ha="center", color="#5a7f9c")
    _blank(ax, (-5, 5), (-2.6, 1.5))
    _save_sketch(fig, "ch10_tank")


def ch13_threads():
    """The frequency workers and their inner pools, the statics shared by all (theory §13.3)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    def box(x0, y0, w, h, text, fc, fs=8.5):
        ax.add_patch(
            FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=EDGE, lw=1.0)
        )
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs)

    box(
        0.0,
        3.3,
        9.6,
        0.6,
        "statics: influence data of the body, field and control-surface rows, panel rules"
        r"  (shared; $f \times i$ threads)",
        "#eeeeee",
    )
    for k, x0 in enumerate((0.0, 3.3, 6.6)):
        label = "frequency worker " + ("$f$" if k == 2 else str(k + 1))
        box(x0, 0.4, 3.0, 2.5, "", "#f4f9fd")
        ax.text(x0 + 1.5, 2.65, label, ha="center", fontsize=8.5)
        ax.text(x0 + 1.5, 2.3, "matrices of one frequency", ha="center", fontsize=7.5, color="#5a7f9c")
        for t in range(3):
            box(x0 + 0.2 + 0.9 * t, 0.65, 0.8, 1.2, "inner\nthread" if t < 2 else "...", "#dceefb", fs=7)
        ax.text(x0 + 1.5, 2.0, r"inner pool, $i$ threads", ha="center", fontsize=7.5)
    ax.text(
        4.8,
        -0.05,
        r"assembly tiles, symmetry characters, LU blocks: on the inner pool of their worker",
        ha="center",
        fontsize=8,
    )
    ax.text(
        4.8,
        -0.45,
        r"after the last frequency: the second-order phases on $f \times i$ threads",
        ha="center",
        fontsize=8,
        color=ACCENT,
    )
    ax.set_xlim(-0.2, 9.8)
    ax.set_ylim(-0.7, 4.1)
    ax.axis("off")
    _save_sketch(fig, "ch13_threads")


# the colours of the sketches of chapters 1 to 5: water, body, free surface, normals, the incident wave, walls
WATER15 = "#dbe9f6"
BODY15 = "#9db4d3"
EDGE15 = "#2b3a4f"
SURFACE = "#2a78d6"
NORMAL = "#d1495b"
WAVE = "#eb6834"
WALL = "#7a7a7a"
INTERIOR = "#f2c14e"


def _arrow_path(ax, p, q, color, lw=1.3, style="-|>", ms=9):
    """An arrow from p to q."""
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw))


def _hull_section(n=60, half_beam=1.0, draft=0.8):
    """The points of a rounded hull section below z = 0, from (-half_beam, 0) to (half_beam, 0)."""
    t = np.linspace(np.pi, 2 * np.pi, n)
    x = half_beam * np.cos(t)
    z = draft * np.sin(t) * (1 - 0.15 * np.cos(t) ** 2)
    return x, z


def ch01_domain():
    """The fluid domain in a vertical section (left): the free surface z = 0, the bottom z = -h,
    the wetted surface S_B with its normal n into the fluid, the rotation centre R; and in plan
    (right): the heading beta of the incident wave, from +x counter-clockwise."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"width_ratios": [1.45, 1]})
    # section
    a.add_patch(Rectangle((-3, -2), 6, 2, color=WATER15, lw=0))
    xs = np.linspace(-3, 3, 200)
    a.plot(xs, 0.06 * np.sin(2.2 * xs), color=SURFACE, lw=1.0, alpha=0.5)
    a.plot([-3, 3], [0, 0], color=SURFACE, lw=1.6)
    a.add_patch(Rectangle((-3, -2.18), 6, 0.18, color="#c9b48a", lw=0, hatch="////", fill=False))
    a.plot([-3, 3], [-2, -2], color="#8a6d3b", lw=1.6)
    x, z = _hull_section()
    a.add_patch(Polygon(np.c_[x, z], closed=True, fc=BODY15, ec=EDGE15, lw=1.4))
    a.add_patch(Polygon([[-1.0, 0], [1.0, 0], [0.8, 0.35], [-0.8, 0.35]], closed=True, fc="#c7d3e3", ec=EDGE15, lw=1.0))
    for t in (np.pi * 1.2, np.pi * 1.5, np.pi * 1.8):
        p = np.array([np.cos(t), 0.8 * np.sin(t) * (1 - 0.15 * np.cos(t) ** 2)])
        d = np.array([np.cos(t) / 1.0, np.sin(t) / 0.8])
        d /= np.linalg.norm(d)
        _arrow_path(a, p, p + 0.45 * d, NORMAL)
    a.text(0.62, -0.95, "$n$", color=NORMAL, fontsize=11)
    a.text(-0.35, -0.42, "$S_B$", fontsize=11)
    a.plot([0.0], [-0.2], "o", color=EDGE15, ms=3.5)
    a.text(0.07, -0.18, "$R$", fontsize=10)
    a.text(1.55, 0.12, "$z = 0$: $-\\omega^2\\phi + g\\phi_z = 0$", color=SURFACE, fontsize=9)
    a.text(1.7, -1.88, "$z = -h$: $\\phi_z = 0$", color="#8a6d3b", fontsize=9)
    a.text(-2.85, -1.2, "$\\nabla^2\\phi = 0$", fontsize=10)
    a.text(-2.85, -1.55, "fluid", fontsize=9, color="#555555")
    _arrow_path(a, (-2.9, 0.45), (-1.9, 0.45), WAVE, lw=1.6)
    a.text(-2.9, 0.55, "incident wave", color=WAVE, fontsize=9)
    _arrow_path(a, (2.6, -1.45), (2.6, -0.85), EDGE15, lw=1.0)
    _arrow_path(a, (2.6, -1.45), (3.2, -1.45), EDGE15, lw=1.0)
    a.text(2.5, -0.8, "$z$", fontsize=10)
    a.text(3.22, -1.5, "$x$", fontsize=10)
    a.set_xlim(-3.05, 3.4)
    a.set_ylim(-2.25, 0.85)
    a.set_aspect("equal")
    a.axis("off")
    a.set_title("vertical section", fontsize=9)
    # plan
    b.add_patch(Rectangle((-2, -2), 4, 4, color=WATER15, lw=0))
    th = np.linspace(0, 2 * np.pi, 100)
    b.add_patch(Polygon(np.c_[1.0 * np.cos(th), 0.45 * np.sin(th)], closed=True, fc=BODY15, ec=EDGE15, lw=1.2))
    _arrow_path(b, (0, 0), (1.85, 0), EDGE15, lw=1.0)
    _arrow_path(b, (0, 0), (0, 1.85), EDGE15, lw=1.0)
    b.text(1.88, -0.12, "$x$", fontsize=10)
    b.text(-0.15, 1.85, "$y$", fontsize=10)
    beta = np.deg2rad(35)
    d = np.array([np.cos(beta), np.sin(beta)])
    crest = np.array([-np.sin(beta), np.cos(beta)]) * 0.45
    for at in (-1.65, -1.3, -0.95):
        p = at * d
        b.plot([p[0] - crest[0], p[0] + crest[0]], [p[1] - crest[1], p[1] + crest[1]], color=WAVE, lw=1.0)
    _arrow_path(b, (-1.4 * np.cos(beta), -1.4 * np.sin(beta)), (1.5 * np.cos(beta), 1.5 * np.sin(beta)), WAVE, lw=1.6)
    b.add_patch(Arc((0, 0), 1.6, 1.6, theta1=0, theta2=35, color=WAVE, lw=1.0))
    b.text(0.85, 0.2, "$\\beta$", color=WAVE, fontsize=11)
    b.text(-1.9, -1.85, "crests $\\perp$ direction of propagation", color=WAVE, fontsize=8)
    b.set_xlim(-2, 2.1)
    b.set_ylim(-2, 2.05)
    b.set_aspect("equal")
    b.axis("off")
    b.set_title("plan, $z = 0$", fontsize=9)
    _save(fig, "ch01_domain")


def ch01_walls():
    """The domains with walls in plan: one wall x = 0, two orthogonal walls, the channel
    |y| < W/2; the fluid shaded, the incident wave and its mirror images."""
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.0))
    beta = np.deg2rad(30)

    def wave(ax, p, ang, color=WAVE):
        d = np.array([np.cos(ang), np.sin(ang)])
        _arrow_path(ax, p - 0.6 * d, p + 0.6 * d, color, lw=1.4)

    def body(ax, c):
        th = np.linspace(0, 2 * np.pi, 60)
        ax.add_patch(
            Polygon(np.c_[c[0] + 0.35 * np.cos(th), c[1] + 0.2 * np.sin(th)], closed=True, fc=BODY15, ec=EDGE15, lw=1.0)
        )

    # x wall
    a = axs[0]
    a.add_patch(Rectangle((0, -2), 2.5, 4, color=WATER15, lw=0))
    a.plot([0, 0], [-2, 2], color=WALL, lw=3)
    body(a, (1.2, 0.0))
    wave(a, np.array([1.2, -1.2]), beta)
    wave(a, np.array([1.2, 1.2]), np.pi - beta, color="#f2a77c")
    a.text(1.25, -0.75, "wave", color=WAVE, fontsize=8)
    a.text(1.0, 1.55, "mirror image", color="#e0895a", fontsize=8)
    a.text(0.08, 1.75, "$x = 0$", fontsize=9)
    a.text(1.35, -1.85, "$x > 0$", fontsize=9)
    a.set_title("one wall ($x = 0$)", fontsize=9)
    # orthogonal walls
    a = axs[1]
    a.add_patch(Rectangle((0, 0), 2.5, 2.5, color=WATER15, lw=0))
    a.plot([0, 0], [0, 2.5], color=WALL, lw=3)
    a.plot([0, 2.5], [0, 0], color=WALL, lw=3)
    body(a, (1.3, 1.2))
    wave(a, np.array([0.8, 0.5]), beta)
    a.text(1.4, 2.25, "$x > 0$, $y > 0$", fontsize=9)
    a.set_title("orthogonal walls", fontsize=9)
    a.set_xlim(-0.3, 2.6)
    a.set_ylim(-0.3, 2.6)
    # channel
    a = axs[2]
    a.add_patch(Rectangle((-2, -1), 4, 2, color=WATER15, lw=0))
    a.plot([-2, 2], [1, 1], color=WALL, lw=3)
    a.plot([-2, 2], [-1, -1], color=WALL, lw=3)
    body(a, (0.3, 0.0))
    wave(a, np.array([-1.2, 0.0]), 0.0)
    a.text(-1.9, 1.12, "$y = W/2$", fontsize=9)
    a.text(-1.9, -1.3, "$y = -W/2$", fontsize=9)
    a.text(-1.5, 0.2, "$\\beta = 0^\\circ$ or $180^\\circ$", fontsize=8, color=WAVE)
    a.set_title("channel", fontsize=9)
    a.set_ylim(-1.6, 1.6)
    for a in axs:
        a.set_aspect("equal")
        a.axis("off")
    axs[0].set_xlim(-0.3, 2.6)
    axs[0].set_ylim(-2.1, 2.1)
    _save(fig, "ch01_walls")


def ch03_reference_points():
    """The points of a body's motion equation in a section: the body origin O, the centre of
    gravity G, the rotation centre R, the point P of additional matrices, and the vectors
    a = G - O and c = G - R of the inertia transfer."""
    fig, a = plt.subplots(figsize=(5.6, 3.0))
    a.add_patch(Rectangle((-3, -1.6), 6, 1.6, color=WATER15, lw=0))
    a.plot([-3, 3], [0, 0], color=SURFACE, lw=1.4)
    hull = [[-2.4, 0.9], [2.4, 0.9], [2.4, -0.9], [2.1, -1.2], [-2.1, -1.2], [-2.4, -0.9]]
    a.add_patch(Polygon(hull, closed=True, fc=BODY15, ec=EDGE15, lw=1.3, alpha=0.9))
    pts = {"O": (-1.6, -1.2), "G": (0.3, 0.35), "R": (0.0, 0.0), "P": (1.7, -0.6)}
    for x, z in pts.values():
        a.plot([x], [z], "o", color=EDGE15, ms=4)
    a.text(-1.75, -1.47, "$O$ (inertia origin)", fontsize=9)
    a.text(0.4, 0.45, "$G$", fontsize=10)
    a.text(-0.42, 0.08, "$R$", fontsize=10)
    a.text(1.8, -0.55, "$P$", fontsize=10)
    _arrow_path(a, pts["O"], pts["G"], NORMAL, lw=1.2)
    a.text(-1.55, -0.35, "$a = G - O$", color=NORMAL, fontsize=9)
    _arrow_path(a, pts["R"], pts["G"], WAVE, lw=1.2)
    a.text(0.55, 0.05, "$c = G - R$", color=WAVE, fontsize=9)
    _arrow_path(a, pts["R"], pts["P"], "#555555", lw=1.0)
    a.text(0.75, -0.62, "$P - R$", color="#555555", fontsize=9)
    a.text(2.5, 0.08, "$z = 0$", color=SURFACE, fontsize=9)
    a.set_xlim(-3, 3.2)
    a.set_ylim(-1.6, 1.0)
    a.set_aspect("equal")
    a.axis("off")
    _save(fig, "ch03_reference_points")


def ch05_interior_surface():
    """The interior surface S_i of a hull in a vertical section: the wetted surface S_b with its
    normal into the fluid, S_i in the waterplane z = 0 inside the body with its normal up, and the
    interior water whose eigenfrequencies are the irregular frequencies."""
    fig, a = plt.subplots(figsize=(6.4, 3.0))
    a.add_patch(Rectangle((-2.6, -1.5), 5.2, 1.5, color=WATER15, lw=0))
    a.plot([-2.6, -1.0], [0, 0], color=SURFACE, lw=1.6)
    a.plot([1.0, 2.6], [0, 0], color=SURFACE, lw=1.6)
    x, z = _hull_section(80, 1.0, 1.0)
    a.add_patch(Polygon(np.c_[x, z], closed=True, fc="#fbf1d6", ec="none", lw=0))
    a.plot(x, z, color=EDGE15, lw=1.6)
    a.plot([-1.0, 1.0], [0, 0], color=INTERIOR, lw=3.0)
    a.plot([-1.0, -1.0, 1.0, 1.0], [0.0, 0.55, 0.55, 0.0], color=EDGE15, lw=1.0, ls="--")
    for t in (np.pi * 1.25, np.pi * 1.5, np.pi * 1.75):
        p = np.array([np.cos(t), np.sin(t) * (1 - 0.15 * np.cos(t) ** 2)])
        d = np.array([np.cos(t), np.sin(t)])
        _arrow_path(a, p, p + 0.4 * d, NORMAL)
    for xx in (-0.5, 0.5):
        _arrow_path(a, (xx, 0), (xx, 0.38), INTERIOR, lw=1.4)
    a.text(0.85, -1.05, "$n$", color=NORMAL, fontsize=11)
    a.text(-1.45, -0.85, "$S_b$", fontsize=11)
    a.text(-0.12, 0.08, "$S_i$", fontsize=11, color="#9a7310")
    a.text(-0.42, -0.42, "interior water", fontsize=8.5, color="#444444")
    a.text(-0.6, -0.62, "(not part of the fluid)", fontsize=8, color="#444444")
    a.text(1.3, 0.1, "$z = 0$", color=SURFACE, fontsize=9)
    a.text(-2.5, -1.35, "exterior fluid", fontsize=9, color="#555555")
    a.set_xlim(-2.6, 2.6)
    a.set_ylim(-1.5, 0.7)
    a.set_aspect("equal")
    a.axis("off")
    _save(fig, "ch05_interior_surface")


# ---------------------------------------------------------------------------------------------
# chapters 11 and 12: the mean drift and the second-order loads

C11_WATER = "#dceefb"
C11_BODY = "#b9b9b9"
C11_EDGE = INK
C11_SURF = C[0]  # the surface a sketch is about
C11_ACCENT = C[1]


def _c11_save(fig, name):
    """Writes the figure as docs/theory/figures/<name>.png, cropped to its drawing."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", metadata={"Software": None}, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _c11_blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def _c11_arrow(ax, a, b, color=C11_EDGE, lw=1.2, style="-|>", scale=10):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=scale, color=color, lw=lw))


def _c11_sea(ax, x0, x1, z_bottom):
    """Water from z_bottom to the free surface z = 0."""
    ax.add_patch(Rectangle((x0, z_bottom), x1 - x0, -z_bottom, color=C11_WATER, zorder=0))


def ch11_control_surface():
    """The three surfaces on which the mean momentum balance is taken: the body surface, a control
    surface about the body with the free surface inside it, and a far cylinder (theory §11)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _c11_sea(ax, -7, 7, -3.6)
    # the body and its mean wetted surface S_B
    hull = [(-1.5, 0.6), (1.5, 0.6), (1.5, -0.9), (1.1, -1.3), (-1.1, -1.3), (-1.5, -0.9)]
    ax.add_patch(Polygon(hull, fc=C11_BODY, ec=C11_EDGE, zorder=2))
    ax.plot([-1.5, -1.5, -1.1, 1.1, 1.5, 1.5], [0, -0.9, -1.3, -1.3, -0.9, 0], color=C11_EDGE, lw=3, zorder=3)
    ax.text(0, 0.15, "body", ha="center", zorder=4)
    ax.text(0, -1.45, "$S_B$", ha="center", va="top")
    # the free surface outside S_c, S_f inside it, the control surface S_c
    for sgn in (1, -1):
        ax.plot([sgn * 3.6, sgn * 7], [0, 0], color=C11_EDGE)
        ax.plot([sgn * 1.5, sgn * 3.6], [0, 0], color=C11_ACCENT, lw=3, solid_capstyle="butt", zorder=3)
        ax.plot([sgn * 1.5], [0], "o", color=C11_EDGE, ms=4, zorder=4)
        ax.plot([sgn * 3.6], [0], "o", color=C11_SURF, ms=4, zorder=4)
    ax.plot([-3.6, -3.6, 3.6, 3.6], [0, -2.3, -2.3, 0], color=C11_SURF, lw=2.2, zorder=3)
    ax.text(2.55, 0.12, "$S_f$", ha="center", va="bottom", color=C11_ACCENT)
    ax.text(1.5, 0.12, "WL", ha="center", va="bottom")
    ax.text(3.75, 0.12, "CL", ha="left", va="bottom", color=C11_SURF)
    ax.text(3.75, -1.05, "$S_c$", ha="left", color=C11_SURF)
    _c11_arrow(ax, (3.6, -1.6), (4.4, -1.6), color=C11_SURF)
    ax.text(4.45, -1.6, "$n$", va="center", color=C11_SURF)
    # the far cylinder of the momentum conservation
    for sgn in (1, -1):
        ax.plot([sgn * 6.4, sgn * 6.4], [0, -3.6], color=C11_EDGE, ls="--", lw=1.0)
    ax.text(6.3, -2.0, r"$S_\infty$", ha="right")
    # the incident wave
    _c11_arrow(ax, (-6.0, 0.55), (-4.6, 0.55))
    ax.text(-5.3, 0.7, r"incident wave, heading $\beta$", ha="center", va="bottom")
    ax.text(
        0,
        -2.6,
        "pressure integration on $S_B$ (§11.2); control surface $S_c$ with $S_f$ (§11.3);\n"
        r"momentum conservation on $S_\infty$ (§11.1)",
        ha="center",
        va="top",
        fontsize=8,
    )
    _c11_blank(ax, (-7, 7), (-3.6, 1.2))
    _c11_save(fig, "ch11_control_surface")


def ch12_free_surface_zones():
    """The zones of the free surface of the potential QTF, seen from above (theory §12.8, §12.9):
    the generated panels to the inner radius R_I, the quadrature zone to the outer circle R_OC, and
    the far field beyond it."""
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    th = np.linspace(0, 2 * np.pi, 361)
    rwl, ri, roc, rfar = 1.0, 3.0, 5.2, 7.6
    grid = "#7d93a6"
    ax.add_patch(Circle((0, 0), rfar, color="#f1f7fc", zorder=0))
    ax.add_patch(Circle((0, 0), roc, color=C11_WATER, zorder=0))
    # the panelled zone: a square grid about the waterline, layers blended to the circle r0, rings
    b = np.sqrt(2.0) * rwl * 0.95
    r0 = 2.0 * rwl * 0.95
    for v in np.linspace(-b, b, 7):
        ax.plot([v, v], [-b, b], color=grid, lw=0.5, zorder=1)
        ax.plot([-b, b], [v, v], color=grid, lw=0.5, zorder=1)
    for t in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        c, s = np.cos(t), np.sin(t)
        sq = b / max(abs(c), abs(s))
        ax.plot([sq * c, ri * c], [sq * s, ri * s], color=grid, lw=0.5, zorder=1)
    sq = b / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
    rr = 0.5 * sq + 0.5 * r0
    ax.plot(rr * np.cos(th), rr * np.sin(th), color=grid, lw=0.5, zorder=1)
    for r in np.linspace(r0, ri, 4):
        ax.plot(r * np.cos(th), r * np.sin(th), color=grid, lw=0.5, zorder=1)
    # the body inside its waterline (the triangles between the grid and the waterline not drawn)
    ax.add_patch(Circle((0, 0), rwl, fc=C11_BODY, ec=C11_EDGE, lw=1.6, zorder=3))
    ax.text(0, 0, "body", ha="center", va="center", zorder=4)
    # the inner radius, the quadrature zone with its nodes in one sector, the outer circle
    ax.plot(ri * np.cos(th), ri * np.sin(th), color=C11_EDGE, lw=1.4, zorder=2)
    gl = (np.polynomial.legendre.leggauss(4)[0] + 1) / 2
    for a0, a1 in ((ri, (ri + roc) / 2), ((ri + roc) / 2, roc)):
        ax.plot(a1 * np.cos(th), a1 * np.sin(th), color=grid, lw=0.6, ls=":", zorder=2)
        tt = np.linspace(np.pi * 0.56, np.pi * 0.94, 9)
        for u in gl:
            r = a0 + (a1 - a0) * u
            ax.plot(r * np.cos(tt), r * np.sin(tt), ".", color=C11_EDGE, ms=2.5, zorder=3)
    ax.plot(roc * np.cos(th), roc * np.sin(th), color=C11_SURF, lw=2.4, zorder=3)
    ax.plot(1.1 * roc * np.cos(th), 1.1 * roc * np.sin(th), color=C11_SURF, lw=0.9, ls="--", zorder=3)
    # the far field: radial Gauss panels to the outer radius
    for t in (-0.62, -0.82):
        ax.plot([roc * np.cos(t), rfar * np.cos(t)], [roc * np.sin(t), rfar * np.sin(t)], color=C11_ACCENT, lw=0.8)
    ax.text(
        rfar,
        -6.15,
        "far field: outgoing series\n(Hankel and local term),\n"
        r"exact in $\theta$, in $r$ to $R_{OC} + L$ with a taper",
        ha="right",
        va="top",
        fontsize=8,
        color=C11_ACCENT,
    )
    # labels
    ax.annotate(
        "waterline",
        (rwl * np.cos(1.0), rwl * np.sin(1.0)),
        (1.4, 3.6),
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": C11_EDGE, "lw": 0.7},
    )
    ax.text(0.0, -2.4, "panelled zone", ha="center", va="center", fontsize=8, bbox={"fc": "white", "ec": "none"})
    ax.text(ri * np.cos(-2.2) - 0.05, ri * np.sin(-2.2) - 0.05, "$R_I$", ha="right", va="top")
    ax.text(-3.95, -1.25, "quadrature\nzone", ha="center", va="center", fontsize=8)
    ax.text(
        roc * np.cos(-2.35) - 0.1,
        roc * np.sin(-2.35) - 0.1,
        r"outer circle $R_{OC}$",
        ha="right",
        va="top",
        color=C11_SURF,
    )
    ax.text(
        1.1 * roc * np.cos(0.55) + 0.1,
        1.1 * roc * np.sin(0.55),
        r"$1.1 R_{OC}$ (fit)",
        ha="left",
        fontsize=8,
        color=C11_SURF,
    )
    _c11_blank(ax, (-rfar, rfar), (-rfar, rfar))
    _c11_save(fig, "ch12_free_surface_zones")


def ch12_wall_image():
    """One wall: the body B in the fluid x > 0 and its mirror image SB, the incident wave and its
    image, and the free surface of the whole plane (theory §12.14)."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    th = np.linspace(0, 2 * np.pi, 361)
    c = (0.0, 0.6)
    ax.add_patch(Rectangle((0, -3.4), 5.6, 7.2, color=C11_WATER, zorder=0))
    ax.add_patch(Rectangle((-5.6, -3.4), 5.6, 7.2, color="#f3f3f3", zorder=0))
    ax.plot(c[0] + 3.1 * np.cos(th), c[1] + 3.1 * np.sin(th), color=C11_EDGE, lw=1.2, zorder=2)
    ax.plot(c[0] + 3.4 * np.cos(th), c[1] + 3.4 * np.sin(th), color=C11_SURF, lw=2.0, zorder=2)
    for sgn, ls, fc, lab in ((1, "-", C11_BODY, "$B$"), (-1, "--", "white", "$SB$")):
        ax.add_patch(Ellipse((sgn * 1.6, 0.6), 1.5, 0.9, fc=fc, ec=C11_EDGE, ls=ls, lw=1.4, zorder=3))
        ax.text(sgn * 1.6, 0.6, lab, ha="center", va="center", zorder=4)
    ax.plot([0, 0], [-3.4, 3.8], color=C11_EDGE, lw=3, zorder=4)
    ax.text(0.15, 3.7, "wall $x = 0$", ha="left", va="top", fontsize=8)
    # the incident wave of heading beta and its image of heading pi - beta
    b = np.deg2rad(150.0)
    d = 1.3 * np.array([np.cos(b), np.sin(b)])
    p = np.array([4.6, -2.9])
    _c11_arrow(ax, tuple(p), tuple(p + d))
    ax.text(p[0] + 0.1, p[1] - 0.05, r"$\beta$", va="top")
    q = np.array([-4.6, -2.9])
    _c11_arrow(ax, tuple(q), tuple(q + d * np.array([-1, 1])), color="#888888")
    ax.text(q[0] - 0.1, q[1] - 0.05, r"$\beta' = \pi - \beta$", ha="right", va="top", color="#555555")
    ax.text(4.2, 3.0, "fluid $x > 0$", ha="center", fontsize=8)
    ax.text(-4.2, 3.0, "mirror image", ha="center", fontsize=8, color="#555555")
    ax.plot([c[0]], [c[1]], "+", color=C11_EDGE, ms=8, zorder=5)
    ax.text(c[0] + 0.15, c[1] - 0.2, "centre", ha="left", va="top", fontsize=7)
    ax.text(3.6, 1.85, "outer circle", ha="left", fontsize=8, color=C11_SURF)
    ax.text(
        -0.2,
        -1.5,
        "panels, quadrature zone,\nouter circle: the whole\nplane at half weight",
        ha="right",
        va="center",
        fontsize=7.5,
    )
    ax.text(
        0.2, -1.5, "waterline and body\nintegrals over $B$\nonly, at full weight", ha="left", va="center", fontsize=7.5
    )
    _c11_blank(ax, (-5.6, 5.6), (-3.4, 3.8))
    _c11_save(fig, "ch12_wall_image")


FIGURES_OF = {name: f for name, f in sorted(globals().items()) if name[:2] == "ch" and name[2:4].isdigit()}


def main(argv):
    prefixes = argv or [""]
    for name, f in FIGURES_OF.items():
        if any(name.startswith(p) for p in prefixes):
            f()
            print(f"docs/theory/figures/{name}.png")


if __name__ == "__main__":
    main(sys.argv[1:])
