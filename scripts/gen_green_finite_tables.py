"""Generates crates/hydrobem/src/tables/green_fin_tables.rs.

Finite-depth Green function, Newman's (1985) decomposition for R/h < 1/2
(green_finite_quad.py, Wuillaume 2022 eqs. 12-20):

    H I(Xb, Vb, H) = sum_{m=0..M} s^m Q_m(w, H),   s = Xb^2, w = Vb^2 in [0, 4].

Tables of Q_m, all m together:
  - regions of tau = log H on [log H_MIN, log H_C], 2D Chebyshev interpolation
    at Lobatto nodes, degree NW in w and NH in tau; the regions follow from the
    error map: a region is halved until the fit meets the tolerances;
  - H >= H_C: the same in t = 1/H on [0, 1/H_C] (t = 0 gives the omega = inf limit chi_m);
  - H < H_MIN and the omega = 0 limit: l_m(w) = lim (Q_m + delta_m0 log(H)/2), a 1D fit
    in w (the remainder is O(H log H), below 2e-10 at H_MIN = 1e-12).
Stored as Chebyshev coefficients c[region][m][j_tau][j_w] (the engine sums
over tau once per frequency and converts to the power basis in y = w/2 - 1).

Scaled modified Bessel functions for the series form (k_1 R >= 1/2):
  e^x sqrt(x) K0(x), e^x sqrt(x) K1(x) on [1/2, 48) (the engine evaluates them below 18, the
  series cut-off of the engine; the tiles above are kept for a larger cut-off),
  tiles [1/2, 1), [1, 2), ..., [47, 48), the lowest degree meeting 1e-14.

Every fit is checked against independent evaluations (points between the
nodes) and the run fails if a tolerance is exceeded. A FNV-1a 64 checksum of
all coefficients is written into the table and verified by the engine.

Reproducibility: a rerun reproduces the tables bit for bit. No LAPACK or BLAS routine enters
(the interpolation coefficients are cosine sums by math.fsum, the Gauss-Legendre rules Newton's
method in mpmath), and the complex arithmetic of the quadrature is written in real float64
operations (NumPy's complex multiply rounds differently with and without FMA). What remains is
elementwise float64 arithmetic, NumPy's sums and exp, cos, sin of the C runtime. The tables were
generated with Python 3.11.18, NumPy 2.4.6 and mpmath 1.4.1 on Windows x86-64; another NumPy or
C runtime may change the last bits (not the accuracy), which --check shows.

Usage: python scripts/gen_green_finite_tables.py [--check]   (about 7 minutes)
  --check: compute the tables and compare with the files instead of writing them.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path

import mpmath as mp
import numpy as np
from numpy.polynomial import chebyshev as C

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_green_tables as gt  # noqa: E402
import green_finite_quad as fq  # noqa: E402
import green_tables_rust as rt  # noqa: E402

M = 9  # powers of s: the omitted term is below 2e-12 (s <= 1/4)
NW = 12  # degree in w (10 left 1.7e-8 in dQ/dVb near H = 8, see the error map)
NH = 14  # degree in tau within a region
NT = 8  # degree in t = 1/H for H >= H_C
H_MIN = 1e-12
H_C = 50.0
TOL_Q = 1e-9  # sum_m 4^-m |error of Q_m|, absolute (|HI| is O(1) to O(log 1/H))
TOL_DQ = 1e-8  # the same for dQ/dVb
TOL_K = 1e-14  # relative error of the scaled Bessel functions
K_LO, K_HI = 0.5, 48
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "crates" / "hydrobem" / "src" / "tables" / "green_fin_tables.rs"
REPORT = ROOT / "docs" / "development" / "error-maps" / "green_fin_error_map.md"
WEIGHT = 0.25 ** np.arange(M + 1)


def lobatto(n):
    return np.cos(np.pi * np.arange(n + 1) / n)[::-1]  # ascending in [-1, 1]


def lobatto_coef(values):
    """Chebyshev coefficients of the interpolant at the ascending Lobatto nodes lobatto(n):
    c_j = (2/n) sum''_k f(cos(pi k/n)) cos(pi j k/n), the end terms of the sum and c_0, c_n halved
    (Trefethen 2019, ch. 3). The sums are math.fsum of separately rounded products, so that no
    LAPACK routine enters (a least-squares fit rounds with the BLAS library)."""
    f = [float(v) for v in values][::-1]  # at cos(pi k/n), k = 0..n
    n = len(f) - 1
    f[0] *= 0.5
    f[n] *= 0.5
    c = [
        2.0 / n * math.fsum(f[k] * math.cos(math.pi * ((j * k) % (2 * n)) / n) for k in range(n + 1))
        for j in range(n + 1)
    ]
    c[0] *= 0.5
    c[n] *= 0.5
    return np.array(c)


YW = lobatto(NW)
WN = 2.0 * (YW + 1.0)  # w nodes
W_TEST = np.linspace(0.0, 4.0, 81)


def H_of(kind, x):
    return math.exp(x) if kind == 0 else (1.0 / x if x > 0 else math.inf)


def q_values(kind, x, w):
    """Q_m, dQ_m/dVb [M+1, len(w)] and quadrature error estimate."""
    H = H_of(kind, x)
    if math.isinf(H):
        q, dq = fq.limit_inf(np.sqrt(w), M)
        return q, dq, 0.0
    return fq.Q_checked(np.sqrt(w), H, M)


def fit_region(kind, a, b, n):
    """Chebyshev coefficients c[m][j_x][j_w] of the region and its error map."""
    yx = lobatto(n)
    xs = a + (b - a) * (yx + 1) / 2
    vals = np.empty((n + 1, M + 1, NW + 1))
    qerr = 0.0
    for i, x in enumerate(xs):
        q, _, e = q_values(kind, x, WN)
        vals[i] = q
        qerr = max(qerr, e)
    # 2D interpolation: first along w, then along x
    c = np.empty((M + 1, n + 1, NW + 1))
    for m in range(M + 1):
        cw = np.array([lobatto_coef(vals[i, m]) for i in range(n + 1)])  # [x node, j_w]
        c[m] = np.array([lobatto_coef(cw[:, j]) for j in range(NW + 1)]).T  # [j_x, j_w]
    # error map: points between the nodes in x, dense in w
    xt = a + (b - a) * (np.concatenate([(lobatto(2 * n)[1::2] + 1) / 2, np.linspace(0, 1, 7)]))
    ev = edv = 0.0
    for x in xt:
        q, dq, e = q_values(kind, x, W_TEST)
        qerr = max(qerr, e)
        f, df = eval_region(c, 2 * (x - a) / (b - a) - 1, W_TEST)
        ev = max(ev, float((WEIGHT[:, None] * np.abs(f - q)).sum(axis=0).max()))
        edv = max(edv, float((WEIGHT[:, None] * np.abs(df - dq)).sum(axis=0).max()))
    return c, ev, edv, qerr


def eval_region(c, yx, w):
    """Q_m and dQ_m/dVb from region coefficients at the region variable yx."""
    n = c.shape[1] - 1
    Tx = C.chebvander(np.array([yx]), n)[0]  # [j_x]
    cw = (c * Tx[None, :, None]).sum(axis=1)  # Chebyshev coefficients in y = w/2 - 1
    y = w / 2 - 1
    f = np.array([C.chebval(y, cw[m]) for m in range(M + 1)])
    # dQ/dVb = dQ/dw * 2 Vb = dQ/dy * Vb
    df = np.array([C.chebval(y, C.chebder(cw[m])) for m in range(M + 1)]) * np.sqrt(w)[None, :]
    return f, df


def build_regions(log):
    todo = [(math.log(H_MIN), math.log(H_C))]
    done = []
    while todo:
        a, b = todo.pop(0)
        c, ev, edv, qe = fit_region(0, a, b, NH)
        if (ev > TOL_Q or edv > TOL_DQ) and b - a > 0.05:
            m = 0.5 * (a + b)
            todo[:0] = [(a, m), (m, b)]
            continue
        log(
            f"  tau [{a:8.4f}, {b:8.4f}] (H {math.exp(a):.3g} .. {math.exp(b):.3g}): "
            f"Q {ev:.1e}, dQ/dVb {edv:.1e}, quadrature {qe:.0e}"
        )
        done.append((0, a, b, c, ev, edv, qe))
    c, ev, edv, qe = fit_region(1, 0.0, 1.0 / H_C, NT)
    log(f"  t = 1/H in [0, {1 / H_C:g}]: Q {ev:.1e}, dQ/dVb {edv:.1e}, quadrature {qe:.0e}")
    done.append((1, 0.0, 1.0 / H_C, c, ev, edv, qe))
    return done


def fit_limit_zero():
    q, _ = fq.limit_zero(np.sqrt(WN), M)
    c = np.array([lobatto_coef(q[m]) for m in range(M + 1)])
    qt, dqt = fq.limit_zero(np.sqrt(W_TEST), M)
    y = W_TEST / 2 - 1
    f = np.array([C.chebval(y, c[m]) for m in range(M + 1)])
    df = np.array([C.chebval(y, C.chebder(c[m])) for m in range(M + 1)]) * np.sqrt(W_TEST)[None, :]
    ev = float((WEIGHT[:, None] * np.abs(f - qt)).sum(axis=0).max())
    edv = float((WEIGHT[:, None] * np.abs(df - dqt)).sum(axis=0).max())
    # the limit against Q at small H (remainder O(H log H))
    rem = []
    for H in (1e-8, 1e-10, H_MIN):
        qs, _, _ = fq.Q_checked(np.sqrt(W_TEST), H, M)
        qs[0] += math.log(H) / 2
        rem.append((H, float((WEIGHT[:, None] * np.abs(qs - qt)).sum(axis=0).max())))
    return c, ev, edv, rem


def bessel_tiles(log):
    """Tiles of e^x sqrt(x) K_nu(x): power-basis coefficients [tile][power][nu]."""
    mp.mp.dps = 30
    edges = [K_LO] + list(range(1, K_HI + 1))

    def g(x, nu):
        x = mp.mpf(x)
        return float(mp.exp(x) * mp.sqrt(x) * mp.besselk(nu, x))

    history = []  # (degree, max relative error) per degree tried, for the page
    for n in range(6, 20):
        worst = 0.0
        tiles = []
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            t = gt.cheb_nodes(n)
            xs = a + (b - a) * (t + 1) / 2
            cs = [gt.cheb_coef_1d(np.array([g(x, nu) for x in xs])) for nu in (0, 1)]
            for x in np.linspace(a, b, 9):
                tt = 2 * (x - a) / (b - a) - 1
                for nu in (0, 1):
                    ref = g(x, nu)
                    worst = max(worst, abs(gt.cheb_eval_1d(cs[nu], tt) - ref) / abs(ref))
            tiles.append([gt.to_monomial_1d(c) for c in cs])
        log(f"  scaled K0, K1 tiles, degree {n}: max relative error {worst:.2e}")
        history.append((n, worst))
        if worst <= TOL_K:
            arr = np.array(tiles)  # [tile][nu][power]
            return n, np.transpose(arr, (0, 2, 1)).ravel(), history
    raise SystemExit("Bessel tiles did not reach the tolerance")


def write_tables(nreg, nh_max, nk, meta, coefs, l0, ktab, checksum, out=OUT):
    """The Rust module of the tables."""
    rt.write_rust_tables(
        out,
        "scripts/gen_green_finite_tables.py",
        "Finite-depth Green function tables.",
        [("M", M), ("NW", NW), ("REGIONS", nreg), ("NK", nk)],
        [("H_MIN", H_MIN), ("HC", H_C), ("K_LO", K_LO)],
        checksum,
        [
            (
                "REGION_META",
                meta,
                5,
                "per region: kind (0: tau = log H, 1: t = 1/H), lower, upper, degree, offset into Q",
            ),
            ("Q", coefs, 3, "Chebyshev coefficients [m][j_x][j_w] of each region"),
            ("L0", l0, 3, "omega = 0 limit: Chebyshev coefficients [m][j_w]"),
            (
                "BESSEL_K",
                ktab,
                2,
                "e^x sqrt(x) K_nu(x): tiles [1/2, 1), [1, 2), ...; [tile][power][nu] in t in [-1, 1]",
            ),
        ],
    )


def tex_power(x):
    """A power of ten as TeX: 1e-08 -> 10^{-8}."""
    return f"10^{{{round(math.log10(x))}}}"


def report_text(data, checksum, seconds):
    """The error map page in GitHub math with its table captions (docs/development/rules.md),
    in the notation of theory §8.2; seconds None leaves out the run time (--check). data:
    "regions" [(kind, a, b, ev, edv, qe)], "zero" (ev, edv, [(H, remainder)]), "inf" the error at
    t = 0, "bessel" [(degree, error)]."""
    rows = []
    for kind, a, b, ev, edv, qe in data["regions"]:
        if kind == 0:
            region = rf"$\tau \in [{a:.4f}, {b:.4f}]$"
            hs = f"{math.exp(a):.3g} to {math.exp(b):.3g}"
        else:
            region = rf"$t \in [{a:g}, {b:g}]$"
            hs = rf"$H \ge {1 / b:g}$"
        rows.append(f"| {region} | {hs} | {ev:.1e} | {edv:.1e} | {qe:.0e} |")
    ev0, edv0, rem = data["zero"]
    rem_text = ", ".join(f"{e:.1e} at $H = {tex_power(H)}$" for H, e in rem)
    bessel = data["bessel"]
    h_min = tex_power(H_MIN)
    return "\n".join(
        [
            "# Error map of the finite-depth Green function tables",
            "",
            "Generated by `scripts/gen_green_finite_tables.py` (do not edit).",
            "",
            "The tables give the coefficients $Q_m(w, H)$ of the decomposition of the finite-depth Green",
            rf"function for $R/h < 1/2$, $Q = \sum_{{m=0}}^{{{M}}} s^m Q_m(w, H)$ (theory §8.2), with",
            r"$s = (R/h)^2$, $w = (v/h)^2 \in [0, 4]$ and $H = Kh$. Each $Q_m$ is a Chebyshev interpolant",
            rf"of degree {NW} in $w$ and, per region, of degree {NH} in $\tau = \log H$ for",
            rf"${h_min} \le H \le {H_C:g}$ and of degree {NT} in $t = 1/H$ for $H \ge {H_C:g}$.",
            "",
            r"The error of a fit is the largest over the test points of $\sum_m 4^{-m} \lvert \Delta Q_m \rvert$,",
            r"for the values and for the derivatives $\partial Q_m / \partial \sqrt{w}$, with the tolerances",
            f"{TOL_Q:g} and {TOL_DQ:g}. The reference is a verified double-precision contour quadrature",
            "(20 against 30 Gauss nodes per panel, cross-checked against mpmath); its own error estimate is",
            "the last column of Table 1.",
            "",
            "*Table 1. The largest error of the tables of $Q_m$ per region.*",
            "",
            r"| region | $H$ | $Q_m$ | $\partial Q_m / \partial \sqrt{w}$ | quadrature |",
            "|---|---|---|---|---|",
            *rows,
            "",
            "## Frequency limits",
            "",
            rf"Below $H = {h_min}$ and at $\omega = 0$ the tables use the limit",
            r"$\ell_m(w) = \lim_{H \to 0} (Q_m + \delta_{m0} \log(H)/2)$, interpolated in $w$ of degree",
            f"{NW}: its error is {ev0:.1e} (values) and {edv0:.1e} (derivatives). The remainder of the",
            rf"limit, which is $O(H \log H)$, is {rem_text}.",
            "",
            r"At $\omega = \infty$ the region in $t$ at $t = 0$ is the limit $H \to \infty$ of $Q_m$: it",
            f"differs from the directly computed limit by {data['inf']:.1e}.",
            "",
            "## Modified Bessel functions",
            "",
            r"The eigenfunction series uses $e^x \sqrt{x} K_0(x)$ and $e^x \sqrt{x} K_1(x)$ on unit tiles of",
            rf"$[1/2, {K_HI:g})$. The degree is raised until the largest relative error is below {TOL_K:g}",
            f"(Table 2); degree {bessel[-1][0]} is kept.",
            "",
            "*Table 2. The largest relative error of the tiles of the scaled Bessel functions per degree.*",
            "",
            "| degree | relative error |",
            "|---|---|",
            *[f"| {n} | {e:.2e} |" for n, e in bessel],
            "",
            f"- checksum (FNV-1a 64 of all coefficients): 0x{checksum:016X}",
            *([] if seconds is None else [f"- run time: {seconds:.0f} s"]),
            "",
        ]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check that the tables and the page are up to date")
    check = ap.parse_args().check
    t0 = time.time()

    def log(s):  # the progress on the console; the page is written from data
        print(s, flush=True)

    log("Q_m regions:")
    regions = build_regions(log)
    data = {"regions": [(kind, a, b, ev, edv, qe) for kind, a, b, _, ev, edv, qe in regions]}
    for *_, ev, edv, qe in regions:
        if ev > TOL_Q or edv > TOL_DQ or qe > 1e-12:
            sys.exit("a region fails the tolerances")
    l0, ev0, edv0, rem = fit_limit_zero()
    log(f"omega = 0 limit l_m(w), degree {NW}: Q {ev0:.1e}, dQ/dVb {edv0:.1e}")
    for H, e in rem:
        log(f"  remainder of the H -> 0 limit at H = {H:g}: {e:.1e}")
    data["zero"] = (ev0, edv0, rem)
    if ev0 > TOL_Q or edv0 > TOL_DQ or rem[-1][1] > TOL_Q:
        sys.exit("the omega = 0 limit fails the tolerances")
    # t = 0 of the last region is the omega = inf limit
    li, _ = fq.limit_inf(np.sqrt(W_TEST), M)
    fi, _ = eval_region(regions[-1][3], -1.0, W_TEST)
    einf = float((WEIGHT[:, None] * np.abs(fi - li)).sum(axis=0).max())
    log(f"omega = inf limit (t = 0 of the last region) vs chi_m: {einf:.1e}")
    data["inf"] = einf
    if einf > TOL_Q:
        sys.exit("the omega = inf limit fails the tolerance")
    nk, ktab, data["bessel"] = bessel_tiles(log)

    nreg = len(regions)
    nh_max = max(r[3].shape[1] for r in regions) - 1
    meta, coefs = [], []
    for kind, a, b, c, *_ in regions:
        n = c.shape[1] - 1
        meta += [float(kind), a, b, float(n), float(len(coefs))]
        coefs += list(np.transpose(c, (0, 1, 2)).ravel())  # [m][j_x][j_w]
    allv = list(meta) + list(coefs) + list(l0.ravel()) + list(ktab)
    checksum = gt.fnv1a64(allv)
    if check:
        with tempfile.TemporaryDirectory() as tmp:
            new = Path(tmp) / OUT.name
            write_tables(nreg, nh_max, nk, meta, coefs, l0.ravel(), ktab, checksum, out=new)
            same = new.read_bytes() == OUT.read_bytes()
        page = [s for s in REPORT.read_text(encoding="utf-8").split("\n") if not s.startswith("- run time:")]
        same_page = "\n".join(page) == report_text(data, checksum, None)
        print(f"checksum 0x{checksum:016X} ({len(allv)} coefficients), {time.time() - t0:.0f} s")
        if not (same and same_page):
            sys.exit(f"{OUT if not same else REPORT} is not up to date: run scripts/gen_green_finite_tables.py")
        print("up to date")
        return
    write_tables(nreg, nh_max, nk, meta, coefs, l0.ravel(), ktab, checksum)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_bytes(report_text(data, checksum, time.time() - t0).encode("utf-8"))
    print(f"wrote {OUT} ({len(allv)} coefficients)")


if __name__ == "__main__":
    main()
