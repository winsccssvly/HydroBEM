"""Verified double-precision quadrature of the functions tabulated for the
finite-depth Green function.

With u = kH, Newman's (1985) expansion of the function I (Wuillaume 2022
eqs. 17-20, 25-27) in the horizontal variable s = (R/h)^2 reads

    H I(Xb, Vb, H) = sum_m s^m Q_m(Vb, H),
    Q_m = (-1)^m / (m!^2 4^m) PV int_0^inf g(u) u^(2m) [exp(-u(2 - Vb)) + exp(-u(2 + Vb))] du,
    g = f - q = q^2 exp(-2u) / (1 - q exp(-2u)),  q = (u + H)/(u - H),
    f = (u + H) / (u (1 - exp(-2u)) - H (1 + exp(-2u))),

the last form of g free of cancellation. g has the real poles u = H and
u tanh u = H, and imaginary poles i t with t tan t = -H, t > pi/2. The
principal value is the real part of the integral along
0 -> i b -> U + i b -> U -> U + 70 (b = 1/2, U beyond both real poles), the
integrand being real on the real axis; for H >= 80 the poles lie where the
integrand is below 1e-30 and the real axis [0, 70] is used. Gauss-Legendre
panels: geometric towards 0 on the imaginary segment (the poles approach the
origin like sqrt(H) for small H), length 1/4 on the horizontal one. The
error is estimated by repeating the quadrature with 30 instead of 20 nodes
per panel.

Limits: Q_0 + log(H)/2 and Q_m (m >= 1) tend to l_m(Vb) as H -> 0
(Wuillaume eq. 45), Q_m tends to chi_m(Vb) as H -> inf (eq. 46); both are
evaluated here by their own integrals and checked against Q_m at extreme H.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mpmath as mp
import numpy as np

B = 0.5


def _legendre(n, x):
    """P_n(x) and P_n'(x) by the three-term recurrence (|x| < 1)."""
    p0, p1 = mp.mpf(1), x
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    return p1, n * (x * p1 - p0) / (x * x - 1)


@lru_cache(maxsize=None)
def _gl(n):
    """Gauss-Legendre nodes (ascending) and weights on [-1, 1] as correctly rounded doubles:
    Newton's method on P_n in 40 digits (no LAPACK eigenvalue routine, whose rounding depends on
    the BLAS library)."""
    xs, ws = [], []
    with mp.workdps(40):
        for i in range(1, n + 1):
            x = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
            for _ in range(100):
                p, dp = _legendre(n, x)
                x -= p / dp
                if abs(p / dp) < mp.mpf(10) ** -36:
                    break
            p, dp = _legendre(n, x)
            xs.append(float(x))
            ws.append(float(2 / ((1 - x * x) * dp * dp)))
    return np.array(xs[::-1]), np.array(ws[::-1])


def _cmul(ar, ai, br, bi):
    """(ar + i ai)(br + i bi) in separately rounded float64 operations."""
    return ar * br - ai * bi, ar * bi + ai * br


def _cdiv(ar, ai, br, bi):
    """(ar + i ai)/(br + i bi) in separately rounded float64 operations."""
    d = br * br + bi * bi
    return (ar * br + ai * bi) / d, (ai * br - ar * bi) / d


def _cexp(xr, xi):
    """exp(xr + i xi) as (real part, imaginary part)."""
    e = np.exp(xr)
    return e * np.cos(xi), e * np.sin(xi)


def _panels(H, n):
    """Nodes and weights of the contour for H as real and imaginary parts (ur, ui, wr, wi)."""
    x, w = _gl(n)
    segs = []  # (a, b) complex endpoints

    def add(a, b, pieces):
        for i in range(pieces):
            segs.append((a + (b - a) * i / pieces, a + (b - a) * (i + 1) / pieces))

    if H >= 80:
        add(0.0, 70.0, 70)
    else:
        ubar = _ubar(H)
        U = max(H, ubar) + 2.0
        scale = min(H, math.sqrt(H), 1.0)
        t = B
        pts = [B]
        while t > 0.02 * scale:
            t /= 2
            pts.append(t)
        pts.append(0.0)
        pts = pts[::-1]
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            segs.append((1j * a, 1j * b))
        add(1j * B, U + 1j * B, max(1, int(math.ceil(U / 0.25))))
        add(U + 1j * B, U + 0j, 1)
        add(U + 0j, U + 70.0 + 0j, 70)
    a = np.array([s[0] for s in segs], dtype=complex)
    b = np.array([s[1] for s in segs], dtype=complex)
    mr, mi = (a.real + b.real) / 2, (a.imag + b.imag) / 2
    hr, hi = (b.real - a.real) / 2, (b.imag - a.imag) / 2
    ur = (mr[:, None] + hr[:, None] * x[None, :]).ravel()
    ui = (mi[:, None] + hi[:, None] * x[None, :]).ravel()
    return ur, ui, (hr[:, None] * w[None, :]).ravel(), (hi[:, None] * w[None, :]).ravel()


def _ubar(H):
    """Real root of u tanh u = H."""
    u = max(H, math.sqrt(H))
    for _ in range(100):
        t = math.tanh(u)
        du = (u * t - H) / (t + u * (1 - t * t))
        u -= du
        if abs(du) <= 1e-16 * u:
            break
    return u


def coef(m):
    return (-1) ** m / (math.factorial(m) ** 2 * 4.0**m)


def Q(Vb, H, M, n=20):
    """Q_m(Vb, H) and dQ_m/dVb for m = 0..M; Vb array -> arrays [M+1, len(Vb)].

    The complex arithmetic is written in real float64 operations: NumPy's complex multiply
    rounds differently with and without FMA, which depends on its version and SIMD dispatch."""
    Vb = np.atleast_1d(np.asarray(Vb, dtype=float))
    ur, ui, wr, wi = _panels(H, n)
    e2r, e2i = _cexp(-2 * ur, -2 * ui)
    qr, qi = _cdiv(ur + H, ui, ur - H, ui)
    tr, ti = _cmul(qr, qi, e2r, e2i)  # q exp(-2u)
    nr, ni = _cmul(qr, qi, tr, ti)  # q^2 exp(-2u)
    gr, gi = _cdiv(nr, ni, 1 - tr, -ti)
    ar, ai = _cexp(-ur[:, None] * (2 - Vb[None, :]), -ui[:, None] * (2 - Vb[None, :]))
    br, bi = _cexp(-ur[:, None] * (2 + Vb[None, :]), -ui[:, None] * (2 + Vb[None, :]))
    out = np.empty((M + 1, Vb.size))
    dout = np.empty((M + 1, Vb.size))
    u2r, u2i = _cmul(ur, ui, ur, ui)
    pr, pi = np.ones_like(ur), np.zeros_like(ur)  # u^(2m)
    wgr, wgi = _cmul(wr, wi, gr, gi)
    for m in range(M + 1):
        cr, ci = _cmul(wgr, wgi, pr, pi)
        cr, ci = cr[:, None], ci[:, None]
        out[m] = coef(m) * (cr * (ar + br) - ci * (ai + bi)).sum(axis=0)
        dr, di = _cmul(cr, ci, ur[:, None], ui[:, None])
        dout[m] = coef(m) * (dr * (ar - br) - di * (ai - bi)).sum(axis=0)
        pr, pi = _cmul(pr, pi, u2r, u2i)
    return out, dout


def Q_checked(Vb, H, M):
    """Q, dQ and the estimated absolute quadrature error (20 vs 30 nodes)."""
    q1, d1 = Q(Vb, H, M, 20)
    q2, d2 = Q(Vb, H, M, 30)
    return q2, d2, max(np.abs(q1 - q2).max(), np.abs(d1 - d2).max())


def _real_axis(n=30):
    x, w = _gl(n)
    edges = np.concatenate([[0.0], np.geomspace(1e-6, 1.0, 13), np.arange(2.0, 81.0)])
    a, b = edges[:-1], edges[1:]
    mid, half = (a + b) / 2, (b - a) / 2
    return (mid[:, None] + half[:, None] * x[None, :]).ravel(), (half[:, None] * w[None, :]).ravel()


def limit_zero(Vb, M):
    """l_m(Vb) = lim_{H->0} [Q_m + delta_m0 log(H)/2] and d/dVb.
    m >= 1: c_m int u^(2m) [e^{-u(4-Vb)} + e^{-u(4+Vb)}] / (1 - e^{-2u}) du;
    m = 0: psi_0 - log 2 - 1 with psi_0 = int [e^{-u(4-Vb)} + e^{-u(4+Vb)} - 2 e^{-4u}] / (1 - e^{-2u}) du
    (Wuillaume 2022 eqs. 34, 45 with alpha_0 = gamma - log 2; verified numerically by the generator)."""
    Vb = np.atleast_1d(np.asarray(Vb, dtype=float))
    u, wt = _real_axis()
    den = -np.expm1(-2 * u)
    a = np.exp(-u[:, None] * (4 - Vb[None, :]))
    b = np.exp(-u[:, None] * (4 + Vb[None, :]))
    out = np.empty((M + 1, Vb.size))
    dout = np.empty((M + 1, Vb.size))
    for m in range(M + 1):
        if m == 0:
            kern = (a + b - 2 * np.exp(-4 * u)[:, None]) / den[:, None]
            out[0] = (wt[:, None] * kern).sum(axis=0) - math.log(2) - 1
        else:
            out[m] = coef(m) * (wt[:, None] * (u ** (2 * m) / den)[:, None] * (a + b)).sum(axis=0)
        dout[m] = coef(m) * (wt[:, None] * (u ** (2 * m + 1) / den)[:, None] * (a - b)).sum(axis=0)
    return out, dout


def limit_inf(Vb, M):
    """chi_m(Vb) = lim_{H->inf} Q_m = c_m int u^(2m) [e^{-u(4-Vb)} + e^{-u(4+Vb)}] / (1 + e^{-2u}) du."""
    Vb = np.atleast_1d(np.asarray(Vb, dtype=float))
    u, wt = _real_axis()
    den = 1 + np.exp(-2 * u)
    a = np.exp(-u[:, None] * (4 - Vb[None, :]))
    b = np.exp(-u[:, None] * (4 + Vb[None, :]))
    out = np.empty((M + 1, Vb.size))
    dout = np.empty((M + 1, Vb.size))
    for m in range(M + 1):
        out[m] = coef(m) * (wt[:, None] * (u ** (2 * m) / den)[:, None] * (a + b)).sum(axis=0)
        dout[m] = coef(m) * (wt[:, None] * (u ** (2 * m + 1) / den)[:, None] * (a - b)).sum(axis=0)
    return out, dout
