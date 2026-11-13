"""Reference data for test A2 (finite depth, test-plan): the regular part of the
finite-depth Green function and its derivatives at points covering R/h from 0
to 5 (the switch at 1/2 on both sides), field and source depths from the free
surface to the bottom, H = K h from 1e-14 to 1e5 (the table region edges) and
the omega = 0 and omega = inf limits (green_finite_reference.py, 30 digits).

Values: series form for R/h >= 0.2, integral form below; both forms are
cross-checked at a subset of points with 0.2 <= R/h < 0.6 (the run fails above 1e-18).

Columns: mode (0 ordinary, 1 omega = 0, 2 omega = inf) h K R z zeta
  Wr dWr/dR dWr/dzeta dWr/dz Wi dWi/dR dWi/dzeta dWi/dz scale scaleD
with Wr = Re G - sum_p sign_p / r_p, Wi = Im G; scale = sum_p 1/r_p and
scaleD = sum_p 1/r_p^2 (the local magnitudes of the singular parts), plus for
the imaginary part A0 E and A0 E k0 folded into the same columns' maximum.

Writes tests/data/green_fin_reference.txt.

Usage: python scripts/gen_green_finite_reference_data.py [--workers N]
"""

from __future__ import annotations

import argparse
import multiprocessing as mpc
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import green_finite_reference as fr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "green_fin_reference.txt"
SIGNS = {0: [1] * 6, 1: [1] * 6, 2: [1, 1, -1, -1, -1, -1]}


def _points():
    rng = np.random.default_rng(7)
    Hs = [
        1e-14,
        1e-12,
        1e-9,
        1e-6,
        3.65e-4,
        0.01,
        0.135,
        0.5,
        1.0,
        2.6,
        4.25,
        7.0,
        10.0,
        18.7,
        30.0,
        49.99,
        50.01,
        100.0,
        1e3,
        1e5,
    ]
    Rs = [0.0, 1e-4, 0.01, 0.1, 0.25, 0.4999, 0.5, 0.5001, 0.7, 1.0, 2.0, 5.0]
    zz = [(-0.5, -0.5), (-0.01, -0.02), (-0.99, -0.98), (-1.0, -1.0), (-0.3, -0.8), (0.0, -0.4), (-0.6, -0.05)]
    pts = []
    for H in Hs:
        for R in Rs:
            for z, ze in zz:
                pts.append((0, 1.0, H, R, z, ze))
        for _ in range(6):
            z, ze = -rng.uniform(0, 1), -rng.uniform(0, 1)
            pts.append((0, 1.0, H, 10 ** rng.uniform(-3, 0.5), z, ze))
    for mode in (1, 2):
        for R in Rs:
            for z, ze in zz:
                pts.append((mode, 1.0, 0.0, R, z, ze))
    # dimensional scaling: depth 450 m (a TLP case), K = w^2/g of a few frequencies; the third value
    # of a point is H = K h (_eval divides it by h), so the TLP points give K h
    for K in (0.004, 0.05, 0.16):
        for R in (3.0, 80.0, 200.0, 300.0):
            for z, ze in ((-10.0, -30.0), (-28.0, -28.0), (-2.0, -0.5)):
                pts.append((0, 450.0, K * 450.0, R, z, ze))
    # drop coincident points and their bottom images (r_p = 0)
    return [p for p in pts if not (p[3] == 0 and (p[4] == p[5] or p[4] + p[5] == -2 * p[1]))]


def _eval(p):
    mp.mp.dps = fr.DPS
    mode, h, K, R, z, ze = p
    h, K, R, z, ze = (mp.mpf(x) for x in (h, K, R, z, ze))
    if mode == 0:
        K = K / h  # H = K h
    sig = SIGNS[mode]
    S, SR, SZ = fr.rankine_sum(R, z, ze, h, sig)
    _, _, Sz = fr.rankine_sum(R, ze, z, h, sig)
    check = 0.0
    if R / h >= 0.2:
        if mode == 0:
            re, im, reR, reZ, imR, imZ = fr.G_series(R, z, ze, h, K)
            rez, imz = fr.G_series(R, ze, z, h, K)[3], fr.G_series(R, ze, z, h, K)[5]
        else:
            f = fr.G0_series if mode == 1 else fr.Ginf_series
            re, reR, reZ = f(R, z, ze, h)
            rez = f(R, ze, z, h)[2]
            im = imR = imZ = imz = mp.mpf(0)
        if R / h < 0.6 and mode == 0 and p[4] > -1 and p[5] > -1:
            ri = fr.G_integral(R, z, ze, h, K)
            check = float(max(abs(ri[0] - re), abs(ri[1] - reR), abs(ri[2] - reZ), abs(ri[3] - rez)) * h)
    else:
        if mode == 0:
            re, reR, reZ, rez = fr.G_integral(R, z, ze, h, K)
        else:
            # limits: the integral forms of Wuillaume eqs. 67, 74 (verified against the series)
            g = fr.G0_integral if mode == 1 else fr.Ginf_integral
            e = mp.mpf(10) ** -12
            re = g(R, z, ze, h)
            reR = (g(R + e, z, ze, h) - g(abs(R - e), z, ze, h)) / (2 * e) if R > e else mp.mpf(0)
            reZ = (g(R, z, ze + e, h) - g(R, z, ze - e, h)) / (2 * e)
            rez = (g(R, z + e, ze, h) - g(R, z - e, ze, h)) / (2 * e)
        if mode == 0:
            A0, E, Ez, k0 = fr.propagating(R, z, ze, h, K)
            Ef = fr.propagating(R, ze, z, h, K)[2]  # d E / d z (E is symmetric in z, zeta)
            J0, J1 = mp.besselj(0, k0 * R), mp.besselj(1, k0 * R)
            im, imR, imZ, imz = -A0 * E * J0, A0 * E * k0 * J1, -A0 * Ez * J0, -A0 * Ef * J0
        else:
            im = imR = imZ = imz = mp.mpf(0)
    v = fr.vdist(z, ze, h)
    scale = sum(1 / mp.sqrt(R * R + vp * vp) for vp in v)
    scaleD = sum(1 / (R * R + vp * vp) for vp in v)
    if mode == 0:
        A0, E, _, k0 = fr.propagating(R, z, ze, h, K)
        scale = max(scale, A0 * E)
        scaleD = max(scaleD, A0 * E * k0)
    vals = [re - S, reR - SR, reZ - SZ, rez - Sz, im, imR, imZ, imz, scale, scaleD]
    head = f"{mode} {float(h)!r} {float(K)!r} {float(R)!r} {float(z)!r} {float(ze)!r}"
    return head + " " + " ".join(mp.nstr(x, 20) for x in vals), check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mpc.cpu_count() - 2))
    args = ap.parse_args()
    with mpc.Pool(args.workers) as pool:
        res = pool.map(_eval, _points(), chunksize=2)
    worst = max(r[1] for r in res)
    lines = [
        "# Generated by scripts/gen_green_finite_reference_data.py -- do not edit.",
        "# mode h K R z zeta Wr dWr/dR dWr/dzeta dWr/dz Wi dWi/dR dWi/dzeta dWi/dz scale scaleD",
    ]
    lines += [r[0] for r in res]
    OUT.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    print(f"{len(res)} points written; largest series/integral difference {worst:.2e}")
    if worst > 1e-18:
        sys.exit("cross-check failed")


if __name__ == "__main__":
    main()
