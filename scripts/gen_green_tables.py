"""Generates crates/hydrobem/src/tables/green_inf_tables.rs.

Tables (Chebyshev fits at Chebyshev nodes of the first kind, stored in the
power basis of the tile variable in [-1, 1], to
arbitrary-precision reference values from green_reference.py):

  1D, X in [0, XT], tile width 1: J0, J1, Yt, Yt'
  2D, (X, Y) in [0, XT] x [0, YT], tile 1 x 1: V, dV/dX (used near the
      origin), and directly Ft, dFt/dX, exp(-Y) J0, exp(-Y) J1, which are
      smooth away from the origin and used outside [0, 2) x [0, 2)

Asymptotic coefficients (exact rational/Gamma expressions, A&S 9.2.9-10 and
12.1.29): Hankel P, Q for orders 0, 1 and H_nu - Y_nu for orders 0, 1.

Every fit is checked on an error map (random points per tile, against the
reference) and the run fails if a tolerance is exceeded. A FNV-1a 64 checksum
of all coefficients is written into the table and verified by the engine
before first use.

Usage: python scripts/gen_green_tables.py [--workers N]
"""

from __future__ import annotations

import argparse
import multiprocessing as mpc
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import green_reference as gr  # noqa: E402
import green_tables_rust as rt  # noqa: E402

XT = 30  # extent of the tabulated domain in X and Y
YT = 30
N1 = 16  # 1D degree
N2 = 7  # 2D degree per direction of the V tables (error map: <= 2e-10)
N3 = 6  # degree of the direct tables: the cheapest meeting 2e-6 with margin (5 gave 5e-6)
CHECK_PTS = 12  # error-map points per tile
TOL_1D = 1e-13  # absolute error target, scaled by max(1, |f|)
TOL_2D = 1e-9
TOL_DIRECT = 3e-7  # >= 6x margin to the 2e-6 target of theory §8
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "crates" / "hydrobem" / "src" / "tables" / "green_inf_tables.rs"
REPORT = ROOT / "docs" / "development" / "error-maps" / "green_inf_error_map.md"


def cheb_nodes(n):
    k = np.arange(n + 1)
    return np.cos(np.pi * (k + 0.5) / (n + 1))  # descending in [-1, 1]


def cheb_coef_1d(values):
    n = len(values) - 1
    k = np.arange(n + 1)
    c = np.array([2.0 / (n + 1) * np.sum(values * np.cos(np.pi * j * (k + 0.5) / (n + 1))) for j in range(n + 1)])
    c[0] *= 0.5
    return c


def cheb_eval_1d(c, t):
    b1 = b2 = 0.0
    for a in c[:0:-1]:
        b1, b2 = 2 * t * b1 - b2 + a, b1
    return t * b1 - b2 + c[0]


def cheb_coef_2d(F):  # F[i, j] at (x_i, y_j); returns c[jy, jx]
    tmp = np.array([cheb_coef_1d(F[:, j]) for j in range(F.shape[1])])  # [j_y_node, cx]
    return np.array([cheb_coef_1d(tmp[:, a]) for a in range(tmp.shape[1])]).T  # [cy, cx]


def to_monomial_1d(c):
    """Chebyshev coefficients on [-1, 1] -> power-basis coefficients (same length)."""
    p = np.polynomial.chebyshev.cheb2poly(c)
    return np.concatenate([p, np.zeros(len(c) - len(p))])


def to_monomial_2d(c):  # c[cy, cx] -> p[jx, ay] (x power major, y index contiguous)
    tmp = np.array([to_monomial_1d(row) for row in c])  # [cy, jx]
    return np.array([to_monomial_1d(tmp[:, j]) for j in range(tmp.shape[1])])  # [jx, ay]


def mono_eval_2d(p, tx, ty):
    r = np.zeros(p.shape[1])
    for j in range(p.shape[0] - 1, -1, -1):  # Horner in x for every y power at once
        r = r * tx + p[j]
    v = 0.0
    for a in range(len(r) - 1, -1, -1):
        v = v * ty + r[a]
    return v


def mono_eval_1d(p, t):
    v = 0.0
    for a in p[::-1]:
        v = v * t + a
    return v


# ------------------------------------------------------------- workers


def _tile_1d(i):
    mp.mp.dps = gr.DPS
    t = cheb_nodes(N1)
    x = i + 0.5 + 0.5 * t
    vals = np.zeros((4, N1 + 1))
    for k, xv in enumerate(x):
        X = mp.mpf(xv)
        y, dy = gr.Yt(X)
        vals[:, k] = [float(mp.besselj(0, X)), float(mp.besselj(1, X)), float(y), float(dy)]
    coef = [cheb_coef_1d(v) for v in vals]
    rng = np.random.default_rng(1000 + i)
    err = np.zeros(4)
    for tt in np.concatenate([[-1.0, 1.0], rng.uniform(-1, 1, CHECK_PTS)]):
        X = mp.mpf(i + 0.5 + 0.5 * tt)
        y, dy = gr.Yt(X)
        ref = [float(mp.besselj(0, X)), float(mp.besselj(1, X)), float(y), float(dy)]
        for f in range(4):
            e = abs(mono_eval_1d(to_monomial_1d(coef[f]), tt) - ref[f]) / max(1.0, abs(ref[f]))
            err[f] = max(err[f], e)
    return i, coef, err


def _direct(X, Y, v, vx):
    """Ft, dFt/dX, exp(-Y) J0, exp(-Y) J1 from V, dV/dX (the identity of green_reference.py, from
    Newman 1985 eqs. 3b and 5)."""
    X, Y = mp.mpf(X), mp.mpf(Y)
    R0 = mp.sqrt(X * X + Y * Y)
    eY = mp.exp(-Y)
    J0, J1 = mp.besselj(0, X), mp.besselj(1, X)
    yt, dyt = gr.Yt(X)
    lg = mp.log(Y + R0) + mp.euler - mp.log(2)
    F = -2 * eY * J0 * lg - mp.pi * eY * yt - 2 * R0 * v
    FX = 2 * eY * J1 * lg - 2 * eY * J0 * X / (R0 * (Y + R0)) - mp.pi * eY * dyt - 2 * X / R0 * v - 2 * R0 * vx
    return [float(F), float(FX), float(eY * J0), float(eY * J1)]


def _tile_2d(args):
    ix, iy = args
    mp.mp.dps = gr.DPS
    t = cheb_nodes(N2)
    V = np.zeros((N2 + 1, N2 + 1))
    VX = np.zeros_like(V)
    for a, tx in enumerate(t):
        for b, ty in enumerate(t):
            v, vx = gr.V_VX(ix + 0.5 + 0.5 * tx, iy + 0.5 + 0.5 * ty)
            V[a, b], VX[a, b] = float(v), float(vx)
    t3 = cheb_nodes(N3)
    DF = np.zeros((4, N3 + 1, N3 + 1))
    for a, tx in enumerate(t3):
        for b, ty in enumerate(t3):
            X, Y = ix + 0.5 + 0.5 * tx, iy + 0.5 + 0.5 * ty
            v, vx = gr.V_VX(X, Y)
            DF[:, a, b] = _direct(X, Y, v, vx)
    cV, cVX = to_monomial_2d(cheb_coef_2d(V)), to_monomial_2d(cheb_coef_2d(VX))
    cD = [to_monomial_2d(cheb_coef_2d(DF[f])) for f in range(4)]
    rng = np.random.default_rng(7 * ix + 131 * iy)
    pts = rng.uniform(-1, 1, (CHECK_PTS, 2))
    pts = np.concatenate([pts, [[-1, -1], [1, 1], [-1, 1], [1, -1]]])
    err = np.zeros(6)
    for tx, ty in pts:
        X, Y = ix + 0.5 + 0.5 * tx, iy + 0.5 + 0.5 * ty
        if X == 0 and Y == 0:  # the reference needs R0 > 0: check at a nearby point
            X = 1e-6
            tx = 2.0 * (X - ix) - 1.0
        v, vx = gr.V_VX(X, Y)
        for f, (c, r) in enumerate(((cV, v), (cVX, vx))):
            e = abs(mono_eval_2d(c, tx, ty) - float(r)) / max(1.0, abs(float(r)))
            err[f] = max(err[f], e)
        if not (ix < 2 and iy < 2):  # the direct tables are used outside [0, 2) x [0, 2)
            for f, r in enumerate(_direct(X, Y, v, vx)):
                e = abs(mono_eval_2d(cD[f], tx, ty) - r) / max(1.0, abs(r))
                err[2 + f] = max(err[2 + f], e)
    return ix, iy, cV, cVX, err, cD


# ------------------------------------------------------------- asymptotics


def hankel_coefficients(nu, terms):
    """a_k(nu) = prod_{j=1..k} (4 nu^2 - (2j-1)^2) / (k! 8^k)  (A&S 9.2.9-10)."""
    mu = 4 * nu * nu
    out = []
    for k in range(terms):
        p = mp.mpf(1)
        for j in range(1, k + 1):
            p *= mu - (2 * j - 1) ** 2
        out.append(p / (mp.factorial(k) * mp.mpf(8) ** k))
    return out


def struve_minus_y_coefficients(nu, terms):
    """H_nu(x) - Y_nu(x) ~ sum_k c_k x^(nu-2k-1),
    c_k = Gamma(k+1/2) 2^(2k+1-nu) / (pi Gamma(nu+1/2-k))  (A&S 12.1.29)."""
    return [
        mp.gamma(k + mp.mpf(0.5)) * mp.mpf(2) ** (2 * k + 1 - nu) / (mp.pi * mp.gamma(nu + mp.mpf(0.5) - k))
        for k in range(terms)
    ]


def check_asymptotics(hank, smy, x0):
    """Maximum relative error of the asymptotic forms on [x0, 4 x0]."""
    worst = 0.0
    for x in np.linspace(x0, 4 * x0, 41):
        X = mp.mpf(x)
        for nu in (0, 1):
            a = hank[nu]
            chi = X - (mp.mpf(nu) / 2 + mp.mpf(1) / 4) * mp.pi
            P = sum((-1) ** k * a[2 * k] / X ** (2 * k) for k in range(len(a) // 2))
            Q = sum((-1) ** k * a[2 * k + 1] / X ** (2 * k + 1) for k in range(len(a) // 2))
            s = mp.sqrt(2 / (mp.pi * X))
            J = s * (P * mp.cos(chi) - Q * mp.sin(chi))
            Yv = s * (P * mp.sin(chi) + Q * mp.cos(chi))
            HmY = sum(c * X ** (nu - 2 * k - 1) for k, c in enumerate(smy[nu]))
            for got, ref in (
                (J, mp.besselj(nu, X)),
                (Yv, mp.bessely(nu, X)),
                (HmY, mp.struveh(nu, X) - mp.bessely(nu, X)),
            ):
                worst = max(worst, float(abs(got - ref)) / max(float(s), float(abs(ref))))
    return worst


# ------------------------------------------------------------- output


fnv1a64 = rt.fnv1a64  # used by gen_green_finite_tables.py


def write_tables(t1, t2, t3, asym, checksum, out=OUT):
    """The Rust module of the tables."""
    rt.write_rust_tables(
        out,
        "scripts/gen_green_tables.py",
        "Infinite-depth Green function tables.",
        [("XT", XT), ("YT", YT), ("N1", N1), ("N2", N2), ("N3", N3), ("HANKEL_TERMS", 14), ("STRUVE_TERMS", 14)],
        [],
        checksum,
        [
            ("TABLE_1D", t1, 3, "1D tiles: [tile][power][function], functions J0, J1, Yt, Yt'"),
            ("TABLE_2D", t2, 3, "2D tiles: [iy][ix][function][x power][y power], functions V, V_X"),
            (
                "TABLE_2D_DIRECT",
                t3,
                3,
                "direct 2D tiles: [iy][ix][function][x power][y power], Ft, Ft_X, exp(-Y) J0, exp(-Y) J1",
            ),
            ("ASYMPTOTIC", asym, 3, "Hankel P, Q of orders 0, 1, then H_nu - Y_nu of orders 0, 1"),
        ],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mpc.cpu_count() - 2))
    args = ap.parse_args()
    mp.mp.dps = gr.DPS
    t0 = time.time()
    with mpc.Pool(args.workers) as pool:
        r1 = sorted(pool.map(_tile_1d, range(XT)), key=lambda r: r[0])
        tiles = [(ix, iy) for iy in range(YT) for ix in range(XT)]
        r2 = pool.map(_tile_2d, tiles, chunksize=4)
    r2 = {(r[0], r[1]): r for r in r2}
    err1 = np.max([r[2] for r in r1], axis=0)
    err2 = np.max([r[4] for r in r2.values()], axis=0)

    hank = {nu: hankel_coefficients(nu, 14) for nu in (0, 1)}
    smy = {nu: struve_minus_y_coefficients(nu, 14) for nu in (0, 1)}
    err_asym = check_asymptotics(hank, smy, XT)

    # layout: 1D [tile][power][function]; 2D [iy][ix][function][x power][y power]
    # 1D: [tile][power][function] (four functions interleaved for one Horner loop)
    t1 = np.array([[[to_monomial_1d(r[1][f])[j] for f in range(4)] for j in range(N1 + 1)] for r in r1])
    t2 = np.zeros((YT, XT, 2, N2 + 1, N2 + 1))
    t3 = np.zeros((YT, XT, 4, N3 + 1, N3 + 1))
    for (ix, iy), r in r2.items():
        t2[iy, ix, 0] = r[2]
        t2[iy, ix, 1] = r[3]
        for f in range(4):
            t3[iy, ix, f] = r[5][f]
    asym = []
    for nu in (0, 1):
        asym += [float(v) for v in hank[nu]]
    for nu in (0, 1):
        asym += [float(v) for v in smy[nu]]
    allv = np.concatenate([t1.ravel(), t2.ravel(), t3.ravel(), np.array(asym)])
    checksum = fnv1a64(allv)

    report = "\n".join(
        [
            "# Error map of the infinite-depth Green function tables",
            "",
            "Generated by `scripts/gen_green_tables.py` (do not edit).",
            "",
            # in GitHub math (docs/development/rules.md)
            rf"- 1D tiles on $[0, {XT}]$, degree {N1}: max error / $\max(1, \lvert f \rvert)$ for $J_0$, $J_1$, "
            r"$\tilde Y$, $\tilde Y'$ = " + ", ".join(f"{e:.2e}" for e in err1),
            rf"- 2D tiles on $[0, {XT}] \times [0, {YT}]$, degree {N2}: max error / $\max(1, \lvert f \rvert)$ for "
            r"$V$, $\partial V/\partial X$ = " + ", ".join(f"{e:.2e}" for e in err2[:2]),
            rf"- direct 2D tiles (outside $[0, 2) \times [0, 2)$), degree {N3}: max error / "
            r"$\max(1, \lvert f \rvert)$ for $\mathrm{Ft}$, $\partial\mathrm{Ft}/\partial X$, $e^{-Y} J_0$, "
            r"$e^{-Y} J_1$ = " + ", ".join(f"{e:.2e}" for e in err2[2:]),
            rf"- Hankel and Struve asymptotics (14 terms) on $[{XT}, {4 * XT}]$: max error relative to "
            rf"$\max(\sqrt{{2/(\pi x)}}, \lvert f \rvert)$ = {err_asym:.2e}",
            f"- reference precision: mpmath {gr.DPS} digits; {CHECK_PTS} random points plus the tile "
            "corners per tile",
            f"- checksum (FNV-1a 64 of all coefficients): 0x{checksum:016X}",
            f"- run time: {time.time() - t0:.0f} s",
            "",
        ]
    )
    print(report)
    # the tolerances first: a failing run writes nothing, so the tables in the tree stay as they were
    if err1.max() > TOL_1D or err2[:2].max() > TOL_2D or err2[2:].max() > TOL_DIRECT or err_asym > 1e-13:
        sys.exit("error map exceeds the tolerance; nothing written")
    write_tables(t1.ravel(), t2.ravel(), t3.ravel(), asym, checksum)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
