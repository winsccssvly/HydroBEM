"""Error map of the Gauss rules of the panel integrals.

The engine integrates over a panel by n x n Gauss rules of the bilinear map of its four corners
(a triangle repeats its third corner) when the field point is far enough, the order chosen by
the distance r = |x - c| / a (c the centroid, a the largest centroid-vertex distance):

- Rankine (src/panel_integrals.rs, rankine_tier and FAR_RATIO): S = int 1/R dS and
  D = int n.(x - xi)/R^3 dS by 3 x 3 beyond 12 a and from 8 a, 4 x 4 from 4 a, 5 x 5 from 3 a,
  6 x 6 from 2 a; closer, the exact forms;
- the logarithmic singularity (log_singularity): L = int psi dS and Ld = int dpsi/dn dS,
  psi = log(R - d_z), d = xi - p, p the image point (above the panel), by 3 x 3 beyond 12 a and
  4/5/6/8/10 from 6/3/2/1.5/1 a, the last two only when p is at least a/2 above the panel's
  highest corner; else an adaptive rule.

For each tier the script samples panels of four shapes in random orientations and field points
at the tier's lower distance and inside the tier, and measures the rule against a composite
reference (8 x 8 cells of 16 x 16 Gauss points; its own error estimated against 12 x 12 cells of
20 x 20). Errors are measured on the scale of each integral: S relative to |S|, D relative to
A / r_c^2 (r_c = |x - c|), L relative to A max(1, |psi(c)|), Ld relative to A / d_c with
d_c = R(c) - d_z(c).

Writes docs/development/error-maps/panel_rule_error_map.md.

    python scripts/panel_rule_error_map.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "development" / "error-maps" / "panel_rule_error_map.md"
SAMPLES = 400  # per shape and tier: half at the tier's lower distance, half inside the tier

RANKINE_TIERS = [(12.0, None, 3), (8.0, 12.0, 3), (4.0, 8.0, 4), (3.0, 4.0, 5), (2.0, 3.0, 6)]
# (lower r, upper r, order, least clearance / a)
LOG_TIERS = [(12.0, None, 3, 0.0), (6.0, 12.0, 4, 0.0), (3.0, 6.0, 5, 0.0), (2.0, 3.0, 6, 0.0),
             (1.5, 2.0, 8, 0.5), (1.0, 1.5, 10, 0.5)]  # fmt: skip

SHAPES = {
    "square": [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
    "rectangle 4:1": [[-2, -0.5, 0], [2, -0.5, 0], [2, 0.5, 0], [-2, 0.5, 0]],
    "triangle": [[-1, -0.8, 0], [1.2, -0.6, 0], [0.1, 1.0, 0], [0.1, 1.0, 0]],
    "skew quadrilateral": [[-1, -0.9, 0], [1.4, -0.5, 0], [0.9, 1.1, 0], [-1.2, 0.6, 0]],
}


def bilinear(corners, u, v):
    a, b, c, d = corners
    x = (np.multiply.outer(0.25 * (1 - u) * (1 - v), a) + np.multiply.outer(0.25 * (1 + u) * (1 - v), b)
         + np.multiply.outer(0.25 * (1 + u) * (1 + v), c) + np.multiply.outer(0.25 * (1 - u) * (1 + v), d))  # fmt: skip
    xu = np.multiply.outer(0.25 * (1 - v), b - a) + np.multiply.outer(0.25 * (1 + v), c - d)
    xv = np.multiply.outer(0.25 * (1 - u), d - a) + np.multiply.outer(0.25 * (1 + u), c - b)
    return x, np.linalg.norm(np.cross(xu, xv), axis=-1)


def rule(n, cells=1):
    """Points and weights on [-1, 1]^2: cells x cells cells of n x n Gauss points."""
    g, w = np.polynomial.legendre.leggauss(n)
    edges = np.linspace(-1.0, 1.0, cells + 1)
    t = np.concatenate([0.5 * (edges[k] + edges[k + 1]) + 0.5 * (edges[k + 1] - edges[k]) * g for k in range(cells)])
    wt = np.concatenate([0.5 * (edges[k + 1] - edges[k]) * w for k in range(cells)])
    u, v = np.meshgrid(t, t, indexing="ij")
    return u.ravel(), v.ravel(), np.outer(wt, wt).ravel()


def integrate(corners, normal, p, kind, nodes):
    u, v, w = nodes
    x, jac = bilinear(corners, u, v)
    ww = w * jac
    d = x - p
    R = np.linalg.norm(d, axis=-1)
    if kind == "rankine":  # 1/R and n.(p - xi)/R^3
        return ww @ (1.0 / R), ww @ (-(d @ normal) / R**3)
    den = R - d[:, 2]
    return ww @ np.log(den), ww @ (((d @ normal) / R - normal[2]) / den)


def rotation(rng):
    q = rng.normal(size=4)
    a, b, c, d = q / np.linalg.norm(q)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])  # fmt: skip


def panel(shape, rng):
    corners = np.array(SHAPES[shape], float) @ rotation(rng).T
    distinct = corners[:3] if shape == "triangle" else corners
    n = np.cross(corners[1] - corners[0], corners[2] - corners[0])  # the shapes lie in z = 0
    n /= np.linalg.norm(n)
    u, v, w = rule(16, 4)
    x, jac = bilinear(corners, u, v)
    area = (w * jac).sum()
    c = ((w * jac)[:, None] * x).sum(axis=0) / area
    a = np.max(np.linalg.norm(distinct - c, axis=1))
    return corners, n, c, a, area


def field_point(c, a, r, rng, top=None, clearance=0.0):
    """A point at r radii from c; for the log terms above the panel's top by the clearance, or
    None when 200 directions give none (the orientation may admit none)."""
    for _ in range(200):
        e = rng.normal(size=3)
        e /= np.linalg.norm(e)
        p = c + r * a * e
        if top is None or p[2] - top >= clearance * a:
            return p
    return None


def tier_map(kind, tiers, rng, ref, fine):
    rows, worst_ref = [], 0.0
    for tier in tiers:
        lo, hi, order = tier[:3]
        clear = tier[3] if len(tier) > 3 else 0.0
        nodes = rule(order)
        err = np.zeros(2)
        for shape in SHAPES:
            for s in range(SAMPLES):
                r = lo if s % 2 == 0 else (lo * (1 + rng.random()) if hi is None else lo + (hi - lo) * rng.random())
                p = None
                while p is None:
                    corners, n, c, a, area = panel(shape, rng)
                    top = corners[:, 2].max() if kind == "log" else None
                    p = field_point(c, a, r, rng, top, clear)
                exact = np.array(integrate(corners, n, p, kind, ref))
                if s % 50 == 0:
                    worst_ref = max(worst_ref, np.max(np.abs(exact - integrate(corners, n, p, kind, fine))
                                                      / scale(kind, c, p, area, exact)))  # fmt: skip
                got = np.array(integrate(corners, n, p, kind, nodes))
                err = np.maximum(err, np.abs(got - exact) / scale(kind, c, p, area, exact))
        rows.append((lo, hi, order, clear, err))
    return rows, worst_ref


def scale(kind, c, p, area, exact):
    d = c - p
    rc = np.linalg.norm(d)
    if kind == "rankine":
        return np.array([abs(exact[0]), area / rc**2])
    dc = rc - d[2]
    return np.array([area * max(1.0, abs(np.log(dc))), area / dc])


def main():
    t0 = time.time()
    rng = np.random.default_rng(2026)
    ref, fine = rule(16, 8), rule(20, 12)
    rank, ref_r = tier_map("rankine", RANKINE_TIERS, rng, ref, fine)
    logs, ref_l = tier_map("log", LOG_TIERS, rng, ref, fine)

    # the page in GitHub math with its table captions (docs/development/rules.md)
    def band(lo, hi):
        return f"$r > {lo:g}$" if hi is None else rf"${lo:g} \le r < {hi:g}$"

    def gauss(o):
        return rf"${o} \times {o}$"

    lines = [
        "# Error map of the Gauss rules of the panel integrals",
        "",
        "Generated by `scripts/panel_rule_error_map.py` (do not edit).",
        "",
        f"Shapes: {', '.join(SHAPES)}; {SAMPLES} random orientations and field points per shape and",
        r"tier, half at the tier's lower distance. $r = \lvert x - c \rvert / a$, $a$ the largest",
        "centroid-vertex distance.",
        "",
        "## Rankine source $S$ and dipole $D$",
        "",
        r"Error of $S$ relative to $\lvert S \rvert$, of $D$ relative to $A / r_c^2$ (Table 1).",
        "",
        "*Table 1. The largest error of the Gauss rules of the Rankine integrals per distance tier.*",
        "",
        "| tier | rule | $S$ | $D$ |",
        "|---|---|---|---|",
    ]
    lines += [f"| {band(lo, hi)} | {gauss(o)} | {e[0]:.2e} | {e[1]:.2e} |" for lo, hi, o, _, e in rank]
    lines += [
        "",
        r"## Logarithmic singularity $L = \int \psi \,\mathrm{d}S$ and "
        r"$L_d = \int \partial\psi/\partial n \,\mathrm{d}S$",
        "",
        r"Error of $L$ relative to $A \max(1, \lvert \psi(c) \rvert)$, of $L_d$ relative to $A / d_c$,",
        "$d_c = R(c) - d_z(c)$ (Table 2).",
        "",
        "*Table 2. The largest error of the Gauss rules of the logarithmic integrals per distance tier.*",
        "",
        "| tier | rule | least clearance / $a$ | $L$ | $L_d$ |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {band(lo, hi)} | {gauss(o)} | {cl:g} | {e[0]:.2e} | {e[1]:.2e} |" for lo, hi, o, cl, e in logs]
    lines += [
        "",
        rf"- reference: $8 \times 8$ cells of $16 \times 16$ Gauss points; its largest difference from"
        rf" $12 \times 12$ cells of $20 \times 20$ on the same scales: {max(ref_r, ref_l):.1e}",
        f"- run time: {time.time() - t0:.0f} s",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
