"""Arbitrary-precision reference for the finite-depth free-surface Green
function (mpmath). Used by the table generator and by the A2 reference data.

Definitions (John 1950; Newman 1985 §3; Wuillaume 2022 eqs. 1-17, 92, whose
time factor exp(-i w t) is conjugated here to exp(+i w t)). Depth h, K = w^2/g,
k0 tanh(k0 h) = K, kn tan(kn h) = -K (n >= 1), field point (x, z), source
(xi, zeta), R the horizontal distance, and the six vertical distances

    v1 = |z - zeta|, v2 = z + zeta + 2h, v3 = |z + zeta|,
    v4 = z - zeta + 2h, v5 = -z + zeta + 2h, v6 = z + zeta + 4h,  r_p = sqrt(R^2 + v_p^2).

Integral form (any R):
    G = 1/r1 + 1/r2 + int_C f(k) dk,
    f = 2 (k + K) cosh k(z+h) cosh k(zeta+h) exp(-kh) J0(kR) / (k sinh kh - K cosh kh),
the contour passing above the pole k0 (outgoing waves), so that
    Re G = 1/r1 + 1/r2 + PV int f dk,
    Im G = -A0 E J0(k0 R),  A0 = 2 pi k0 / (1 - exp(-4 k0 h) + 4 k0 h exp(-2 k0 h)),
    E = sum_{p=3..6} exp(-k0 v_p) = 4 exp(-2 k0 h) cosh k0(z+h) cosh k0(zeta+h).
Series form (R > 0):
    G = -A0 E (Y0(k0 R) + i J0(k0 R)) + 4 sum_n (kn^2 + K^2)/(kn^2 h + K^2 h - K)
        cos kn(z+h) cos kn(zeta+h) K0(kn R).

The principal value is evaluated as the real part of the integral along a
contour in the upper half plane below the first imaginary pole i pi/(2h): the
integrand is real on the real axis, so Re int_C = PV int. The infinite-depth
integrand (k + K)/(k - K) exp(k(z + zeta)) J0(kR), whose principal value is
1/r3 + K Ft(KR, K v3) (green_reference.py), is subtracted so that the
remainder decays like exp(-kh).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_dispersion_reference as dr  # noqa: E402
import green_reference as gr  # noqa: E402

DPS = 30


def vdist(z, zeta, h):
    return [abs(z - zeta), z + zeta + 2 * h, abs(z + zeta), z - zeta + 2 * h, -z + zeta + 2 * h, z + zeta + 4 * h]


def dv_dzeta(z, zeta):
    s = mp.sign(z - zeta)
    return [-s, mp.mpf(1), mp.sign(z + zeta) if z + zeta != 0 else mp.mpf(-1), mp.mpf(-1), mp.mpf(1), mp.mpf(1)]


def rankine_sum(R, z, zeta, h, signs):
    """sum_p s_p / r_p, d/dR and d/dzeta."""
    v, dv = vdist(z, zeta, h), dv_dzeta(z, zeta)
    S = SR = SZ = mp.mpf(0)
    for p in range(6):
        if signs[p] == 0:
            continue
        r = mp.sqrt(R * R + v[p] ** 2)
        S += signs[p] / r
        SR -= signs[p] * R / r**3
        SZ -= signs[p] * v[p] * dv[p] / r**3
    return S, SR, SZ


def propagating(R, z, zeta, h, K):
    """A0, E, dE/dzeta, k0."""
    k0 = dr.x0(K * h) / h
    A0 = 2 * mp.pi * k0 / (1 - mp.exp(-4 * k0 * h) + 4 * k0 * h * mp.exp(-2 * k0 * h))
    v, dv = vdist(z, zeta, h), dv_dzeta(z, zeta)
    E = sum(mp.exp(-k0 * v[p]) for p in range(2, 6))
    Ez = sum(-k0 * dv[p] * mp.exp(-k0 * v[p]) for p in range(2, 6))
    return A0, E, Ez, k0


def _series_sum(R, z, zeta, h, coef, kn_of, tol):
    """sum_n coef(n, kn) cos kn(z+h) cos kn(zeta+h) K0(kn R) and derivatives."""
    S = SR = SZ = mp.mpf(0)
    n, small = 1, 0
    while True:
        kn = kn_of(n)
        c = coef(n, kn)
        k0v, k1v = mp.besselk(0, kn * R), mp.besselk(1, kn * R)
        cz, cs = mp.cos(kn * (z + h)), mp.cos(kn * (zeta + h))
        t = c * cz * cs * k0v
        S += t
        SR -= c * cz * cs * kn * k1v
        SZ -= c * cz * kn * mp.sin(kn * (zeta + h)) * k0v
        small = small + 1 if abs(c) * k0v < tol else 0
        if small >= 3:
            return S, SR, SZ
        n += 1


def G_series(R, z, zeta, h, K, tol=None):
    """(Re G, Im G, dRe/dR, dRe/dzeta, dIm/dR, dIm/dzeta) from the series, R > 0."""
    tol = tol or mp.mpf(10) ** (-DPS + 5) / h
    A0, E, Ez, k0 = propagating(R, z, zeta, h, K)
    x = k0 * R
    Y0, Y1, J0, J1 = mp.bessely(0, x), mp.bessely(1, x), mp.besselj(0, x), mp.besselj(1, x)
    re, im = -A0 * E * Y0, -A0 * E * J0
    reR, imR = A0 * E * k0 * Y1, A0 * E * k0 * J1
    reZ, imZ = -A0 * Ez * Y0, -A0 * Ez * J0
    H = K * h
    S, SR, SZ = _series_sum(
        R,
        z,
        zeta,
        h,
        lambda n, kn: 4 * (kn * kn + K * K) / (kn * kn * h + K * K * h - K),
        lambda n: dr.xn(H, n) / h,
        tol,
    )
    return re + S, im, reR + SR, reZ + SZ, imR, imZ


def G0_series(R, z, zeta, h, tol=None):
    """omega = 0 limit, the gauge of Wuillaume (2022) eq. 67: lim (G - C(K, h)),
    C = -(log(K h) + i pi)/h in this time convention; (Re, d/dR, d/dzeta)."""
    tol = tol or mp.mpf(10) ** (-DPS + 5) / h
    S, SR, SZ = _series_sum(R, z, zeta, h, lambda n, kn: 4 / h, lambda n: n * mp.pi / h, tol)
    return -2 / h * (mp.log(R / (2 * h)) + mp.euler) + S, -2 / (h * R) + SR, SZ


def Ginf_series(R, z, zeta, h, tol=None):
    """omega = inf limit (Re, d/dR, d/dzeta)."""
    tol = tol or mp.mpf(10) ** (-DPS + 5) / h
    return _series_sum(R, z, zeta, h, lambda n, kn: 4 / h, lambda n: (n - mp.mpf(1) / 2) * mp.pi / h, tol)


def _contour(h, poles, tail=None):
    """Points of a polyline from 0 above the real poles, below i pi/(2h)."""
    beta = mp.pi / (4 * h)
    pmin = min(poles)
    pts = [mp.mpf(0)]
    t = beta
    vert = []
    while t > pmin / 8 and t > beta * mp.mpf(10) ** -12:
        vert.append(t)
        t /= 4
    pts += [mp.mpc(0, s) for s in reversed(vert)]
    U = max(poles) + 4 / h
    n = max(1, int(mp.ceil((U * h) / 2)))
    pts += [mp.mpc(U * i / n, beta) for i in range(1, n + 1)]
    pts.append(U)
    pts.append(tail if tail is not None else U + 80 / h)
    return pts


def G_integral(R, z, zeta, h, K):
    """(Re G, dRe/dR, dRe/dzeta, dRe/dz) from the integral form (any R >= 0, R < h)."""
    k0 = dr.x0(K * h) / h
    cache = {}

    def f(k):
        key = (mp.re(k), mp.im(k))
        if key not in cache:
            d = k * mp.sinh(k * h) - K * mp.cosh(k * h)
            a = 2 * (k + K) * mp.exp(-k * h) / d
            cz, sz = mp.cosh(k * (z + h)), mp.sinh(k * (z + h))
            cs, ss = mp.cosh(k * (zeta + h)), mp.sinh(k * (zeta + h))
            inf = (k + K) / (k - K) * mp.exp(k * (z + zeta))
            J0, J1 = mp.besselj(0, k * R), mp.besselj(1, k * R)
            cache[key] = (
                (a * cz * cs - inf) * J0,
                -(a * cz * cs - inf) * k * J1,
                (a * cz * ss - inf) * k * J0,
                (a * sz * cs - inf) * k * J0,
            )
        return cache[key]

    H = K * h
    if H > 100:
        # the remainder decays like exp(-kh): the poles near K contribute below exp(-50)
        top = min(K / 2, 80 / h)
        pts = [mp.mpf(0), top / 8, top / 2, top]
    else:
        pts = _contour(h, [k0, K])
    vals = [mp.re(mp.quad(lambda k, i=i: f(k)[i], pts)) for i in range(4)]
    S, SR, SZ = rankine_sum(R, z, zeta, h, [1, 1, 0, 0, 0, 0])
    _, _, Sz = rankine_sum(R, zeta, z, h, [1, 1, 0, 0, 0, 0])  # symmetric in (z, zeta)
    # infinite-depth part: PV int f_inf dk = 1/r3 + K Ft(K R, K v3), v3 = -(z + zeta)
    v3 = abs(z + zeta)
    r3 = mp.sqrt(R * R + v3 * v3)
    F, FX = gr.Ft_3b(K * R, K * v3)
    FY = -F - 2 / (K * r3)
    re = S + vals[0] + 1 / r3 + K * F
    reR = SR + vals[1] - R / r3**3 + K * K * FX
    reZ = SZ + vals[2] + v3 / r3**3 - K * K * FY
    rez = Sz + vals[3] + v3 / r3**3 - K * K * FY
    return re, reR, reZ, rez


def L0(Xb, Vb):
    """Wuillaume (2022) eq. 55."""
    Xb, Vb = mp.mpf(Xb), mp.mpf(Vb)
    def g(u):
        a = (mp.exp(-u * (4 - Vb)) + mp.exp(-u * (4 + Vb))) * mp.besselj(0, u * Xb) - 2 * mp.exp(-2 * u)
        return a / -mp.expm1(-2 * u)

    return mp.quad(g, [0, 1, 10, mp.inf]) - mp.log(2)


def Linf(Xb, Vb):
    """Wuillaume (2022) eq. 61."""
    Xb, Vb = mp.mpf(Xb), mp.mpf(Vb)
    g = lambda u: (mp.exp(-u * (4 - Vb)) + mp.exp(-u * (4 + Vb))) * mp.besselj(0, u * Xb) / (1 + mp.exp(-2 * u))  # noqa: E731
    return mp.quad(g, [0, 1, 10, mp.inf])


def G0_integral(R, z, zeta, h):
    v = vdist(z, zeta, h)
    S, _, _ = rankine_sum(R, z, zeta, h, [1] * 6)
    return S + (L0(R / h, v[0] / h) + L0(R / h, v[1] / h)) / h


def Ginf_integral(R, z, zeta, h):
    v = vdist(z, zeta, h)
    S, _, _ = rankine_sum(R, z, zeta, h, [1, 1, -1, -1, -1, -1])
    return S + (Linf(R / h, v[0] / h) + Linf(R / h, v[1] / h)) / h


def I_direct(Xb, Vb, H):
    """Wuillaume (2022) eq. 17 (nondimensional k = k_dim / K), real PV."""
    Xb, Vb, H = mp.mpf(Xb), mp.mpf(Vb), mp.mpf(H)
    kb2 = dr.x0(H) / H  # pole of the first term: k tanh(kH) = 1

    def f(k):
        return (
            (1 / (k * mp.sinh(k * H) - mp.cosh(k * H)) - 2 * mp.exp(-k * H) / (k - 1))
            * (k + 1)
            * mp.cosh(k * Vb * H)
            * mp.exp(-k * H)
            * mp.besselj(0, k * Xb * H)
        )

    if H > 100:
        return mp.quad(f, [0, mp.mpf(1) / 4, mp.mpf(1) / 2])
    return mp.re(mp.quad(f, _contour(H, [kb2, mp.mpf(1)], tail=max(kb2, 1) + 4 / H + 60 / H)))


def decomposition(R, z, zeta, h, K):
    """Re G by Newman's decomposition (Wuillaume 2022 eqs. 13, 14) with I from eq. 17."""
    H = K * h
    v = vdist(z, zeta, h)
    S, _, _ = rankine_sum(R, z, zeta, h, [1] * 6)
    X = K * R
    tot = S
    for p in range(2):
        V = K * v[p]
        tot += K * (gr.Ft_3b(X, 2 * H - V)[0] + gr.Ft_3b(X, 2 * H + V)[0] + I_direct(R / h, v[p] / h, H))
    return tot
