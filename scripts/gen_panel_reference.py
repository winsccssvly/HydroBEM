"""Reference values for test A1 (test-plan): Rankine source/dipole integrals and
the logarithmic-singularity integrals over plane panels, by mpmath adaptive
quadrature in polar coordinates about the foot point of the field point
(independent of the closed forms used by the engine).

Writes tests/data/panel_reference.txt; each line:
  kind v0x v0y v0z ... v3x v3y v3z  px py pz  value1 value2
kind "R": S = int 1/r dS, D = int n.(x-xi)/r^3 dS  at the field point p
kind "L": L = int log(|xi-p| + p_z - xi_z) dS, Ld = int d/dn_xi of it  for the image point p
Vertices are counter-clockwise about the normal (into the fluid).

Usage: python scripts/gen_panel_reference.py [--workers N]
"""

from __future__ import annotations

import argparse
import multiprocessing as mpc
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "panel_reference.txt"
DPS = 25


def _panels():
    sq = [(0, 0, -1), (1, 0, -1), (1, 1, -1), (0, 1, -1)]  # horizontal, normal +z
    skew = [(0.0, 0.0, -0.5), (1.3, 0.1, -0.6), (1.5, 0.9, -0.4), (0.2, 1.1, -0.3)]
    tri = [(0, 0, -2), (2, 0, -2), (0.5, 1.5, -2), (0.5, 1.5, -2)]
    side = [(2, -0.5, -1.0), (2, 0.5, -1.0), (2, 0.5, 0.0), (2, -0.5, 0.0)]  # vertical, normal +x
    thin = [(0, 0, -1), (3, 0, -1), (3, 0.05, -1), (0, 0.05, -1)]
    out = []
    for name, v in (("sq", sq), ("skew", skew), ("tri", tri), ("side", side), ("thin", thin)):
        v = np.array(v, dtype=float)
        # make the skewed panel planar: project on the plane of the diagonals through the mean
        n = np.cross(v[2] - v[0], v[3] - v[1])
        n /= np.linalg.norm(n)
        c = v.mean(axis=0)
        v = v - np.outer((v - c) @ n, n)
        out.append((name, v))
    return out


def _normal(v):
    n = np.cross(v[2] - v[0], v[3] - v[1])
    return n / np.linalg.norm(n)


def _distinct(v):
    out = [v[0]]
    for q in v[1:]:
        if np.linalg.norm(q - out[-1]) > 1e-12 and np.linalg.norm(q - out[0]) > 1e-12:
            out.append(q)
    return out


def _polar_integral(verts, n, p, f):
    """Integral of f(xi) over the plane polygon in polar coordinates about the
    foot point O of p (signed fan of triangles O, va, vb)."""
    n = [mp.mpf(float(c)) for c in n]
    P = [mp.mpf(float(c)) for c in p]
    V = [[mp.mpf(float(c)) for c in q] for q in verts]
    w = sum((P[i] - V[0][i]) * n[i] for i in range(3))
    O = [P[i] - w * n[i] for i in range(3)]
    # in-plane orthonormal axes
    e1 = [V[1][i] - V[0][i] for i in range(3)]
    l1 = mp.sqrt(sum(c * c for c in e1))
    e1 = [c / l1 for c in e1]
    e2 = [n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0]]

    def uv(q):
        d = [q[i] - O[i] for i in range(3)]
        return sum(d[i] * e1[i] for i in range(3)), sum(d[i] * e2[i] for i in range(3))

    total = mp.mpf(0)
    m = len(V)
    for k in range(m):
        ax, ay = uv(V[k])
        bx, by = uv(V[(k + 1) % m])
        cr = ax * by - ay * bx  # signed twice-area of (O, a, b)
        if abs(cr) < mp.mpf(10) ** (-20):
            continue
        ta, tb = mp.atan2(ay, ax), mp.atan2(by, bx)
        dth = tb - ta
        while dth > mp.pi:
            dth -= 2 * mp.pi
        while dth < -mp.pi:
            dth += 2 * mp.pi
        ex, ey = bx - ax, by - ay
        # edge line: points a + s (b - a); distance d from O and normal angle
        el = mp.sqrt(ex * ex + ey * ey)
        nx, ny = ey / el, -ex / el
        d = ax * nx + ay * ny
        if d < 0:
            nx, ny, d = -nx, -ny, -d
        phi = mp.atan2(ny, nx)

        def inner(th, d=d, phi=phi):
            re = d / mp.cos(th - phi)
            c, s = mp.cos(th), mp.sin(th)

            def g(r):
                xi = [O[i] + r * (c * e1[i] + s * e2[i]) for i in range(3)]
                return f(xi) * r

            brk = [mp.mpf(0)]
            if 0 < abs(w) < re:
                brk.append(abs(w))
            brk.append(re)
            return mp.quad(g, brk)

        lo, hi = min(ta, ta + dth), max(ta, ta + dth)
        th_pts = [lo, hi]
        # angle of the foot of the perpendicular to the edge, if inside the range
        for shift in (-2 * mp.pi, 0, 2 * mp.pi):
            if lo < phi + shift < hi:
                th_pts = [lo, phi + shift, hi]
        val = mp.quad(inner, th_pts)
        total += val if dth > 0 else -val  # signed fan: orientation of (O, a, b)
    return total


def _case(args):
    kind, verts, p = args
    mp.mp.dps = DPS
    v = np.array(verts)
    n = _normal(v)
    dv = _distinct(v)
    P = [mp.mpf(float(c)) for c in p]
    N = [mp.mpf(float(c)) for c in n]
    if kind == "R":

        def fS(xi):
            return 1 / mp.sqrt(sum((P[i] - xi[i]) ** 2 for i in range(3)))

        def fD(xi):
            r = mp.sqrt(sum((P[i] - xi[i]) ** 2 for i in range(3)))
            return sum(N[i] * (P[i] - xi[i]) for i in range(3)) / r**3

        a = _polar_integral(dv, n, p, fS)
        w = sum((P[i] - mp.mpf(float(dv[0][i]))) * N[i] for i in range(3))
        if abs(w) < mp.mpf(10) ** (-15):
            b = mp.mpf(0)  # principal value on the panel plane
        else:
            b = _polar_integral(dv, n, p, fD)
    else:

        def fL(xi):
            d = [xi[i] - P[i] for i in range(3)]
            R = mp.sqrt(sum(c * c for c in d))
            return mp.log(R - d[2])

        def fLd(xi):
            d = [xi[i] - P[i] for i in range(3)]
            R = mp.sqrt(sum(c * c for c in d))
            return (sum(d[i] * N[i] for i in range(3)) / R - N[2]) / (R - d[2])

        a = _polar_integral(dv, n, p, fL)
        b = _polar_integral(dv, n, p, fLd)
    return kind, verts, p, mp.nstr(a, 20), mp.nstr(b, 20)


def _cases():
    rng = np.random.default_rng(20261024)
    cases = []
    for _name, v in _panels():
        n = _normal(v)
        c = np.mean(_distinct(v), axis=0)
        diam = max(np.linalg.norm(a - b) for a in v for b in v)
        e = v[1] - v[0]
        pts = [c, c + 1e-3 * diam * n, c - 0.3 * diam * n, c + 0.05 * diam * n]  # self, near above/below
        pts += [v[0] + 1e-4 * diam * n, 0.5 * (v[0] + v[1]) + 1e-4 * diam * n]  # near vertex, near edge
        pts += [0.5 * (v[0] + v[1]) - 0.2 * e, v[0] - 0.3 * e]  # in plane, outside
        for f in (0.7, 1.5, 3.0, 8.0, 14.0, 25.0, 60.0):  # distances in diameters
            d = rng.normal(size=3)
            pts.append(c + f * diam * d / np.linalg.norm(d))
        for q in pts:
            cases.append(("R", v.tolist(), list(map(float, q))))
        # log singularity: image points above the free surface (p_z >= 0)
        top = max(v[:, 2])
        for q in (c, v[0], 0.5 * (v[1] + v[2])):
            for h in (1e-3, 0.1, 0.5, 2.0):
                pp = np.array([q[0], q[1], -top + h * diam if top < 0 else h * diam])
                cases.append(("L", v.tolist(), list(map(float, pp))))
        for f in (5.0, 15.0, 40.0):
            d = rng.normal(size=2)
            pp = np.array([c[0], c[1], 0.0]) + f * diam * np.array([*d / np.linalg.norm(d), 0.0])
            pp[2] = 0.2 * diam
            cases.append(("L", v.tolist(), list(map(float, pp))))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mpc.cpu_count() - 2))
    args = ap.parse_args()
    with mpc.Pool(args.workers) as pool:
        res = pool.map(_case, _cases())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Generated by scripts/gen_panel_reference.py -- do not edit."]
    for kind, verts, p, a, b in res:
        flat = [f"{x:.17g}" for q in verts for x in q]
        lines.append(" ".join([kind, *flat, *(f"{x:.17g}" for x in p), a, b]))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(res)} cases written to {OUT}")


if __name__ == "__main__":
    main()
