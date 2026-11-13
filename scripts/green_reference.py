"""Arbitrary-precision reference for the infinite-depth free-surface Green
function (mpmath). Used by the table generator and by the A2 reference data.

Definitions (Newman 1985, J. Eng. Math. 19, eqs. 1-3; time factor exp(i w t)):

    G = 1/r + 1/r' + K Ft(X, Y) - 2 pi i K exp(-Y) J0(X),
    X = K R, Y = K |z + zeta|,  R0 = sqrt(X^2 + Y^2),
    Ft = F - 1/R0 = 2 PV int_0^inf exp(-kY) J0(kX) / (k - 1) dk            (3a)
       = -pi exp(-Y) (H0(X) + Y0(X)) - 2 int_0^Y exp(t-Y) (X^2+t^2)^-1/2 dt (3b)

The smooth function tabulated by the engine:

    Ft = -2 exp(-Y) J0(X) (log(Y + R0) + gamma - log 2) - pi exp(-Y) Yt(X) - 2 R0 V(X, Y),
    Yt(X) = Y0(X) - (2/pi) (log(X/2) + gamma) J0(X)   (entire),
    V = exp(-Y) [ pi H0(X)/2 + int_0^Y (exp(t) - J0(X)) (X^2+t^2)^-1/2 dt ] / R0.

The identity follows from (3b) with Y0 = Yt + (2/pi)(log(X/2)+gamma) J0 and
asinh(Y/X) = log(Y + R0) - log X.
"""

from __future__ import annotations

import mpmath as mp

DPS = 30


def _pts(X, Y):
    pts = [mp.mpf(0)]
    if 0 < X < Y:
        pts.append(X)
    pts.append(Y)
    return pts


def V_VX(X, Y):
    """V and dV/dX at (X, Y) >= 0, R0 > 0."""
    X, Y = mp.mpf(X), mp.mpf(Y)
    R0 = mp.sqrt(X * X + Y * Y)
    J0, J1 = mp.besselj(0, X), mp.besselj(1, X)
    H0, H1 = mp.struveh(0, X), mp.struveh(1, X)
    eY = mp.exp(-Y)
    if Y > 0:
        pts = _pts(X, Y)
        i1 = mp.quad(lambda t: (mp.exp(t - Y) - eY * J0) / mp.sqrt(X * X + t * t), pts)
        i2 = mp.quad(
            lambda t: eY * J1 / mp.sqrt(X * X + t * t) - (mp.exp(t - Y) - eY * J0) * X / (X * X + t * t) ** 1.5,
            pts,
        )
    else:
        i1 = i2 = mp.mpf(0)
    ne = eY * mp.pi * H0 / 2 + i1  # exp(-Y) N
    if X == 0:
        # V = exp(-Y) T(X^2, Y) is even in X; the integrand of i2 tends to a
        # delta-like limit at X = 0, so the derivative is set from symmetry.
        return ne / R0, mp.mpf(0)
    nxe = eY * (1 - mp.pi * H1 / 2) + i2  # exp(-Y) dN/dX, using H0' = 2/pi - H1
    return ne / R0, nxe / R0 - ne * X / R0**3


def Yt(X):
    """Yt(X) = Y0 - (2/pi)(log(X/2) + gamma) J0 and its derivative."""
    X = mp.mpf(X)
    if X == 0:
        return mp.mpf(0), mp.mpf(0)
    J0, J1 = mp.besselj(0, X), mp.besselj(1, X)
    L = mp.log(X / 2) + mp.euler
    y = mp.bessely(0, X) - 2 / mp.pi * L * J0
    dy = -mp.bessely(1, X) - 2 / mp.pi * (J0 / X - L * J1)
    return y, dy


def Ft_3b(X, Y):
    """Ft and dFt/dX from Havelock's form (3b)."""
    X, Y = mp.mpf(X), mp.mpf(Y)
    eY = mp.exp(-Y)
    if X == 0:  # vertical axis: Ft = 2 PV int exp(-kY)/(k-1) dk = -2 exp(-Y) Ei(Y); Ft is even in X
        return -2 * eY * mp.ei(Y), mp.mpf(0)
    H0, H1 = mp.struveh(0, X), mp.struveh(1, X)
    Y0, Y1 = mp.bessely(0, X), mp.bessely(1, X)
    if Y > 0:
        pts = _pts(X, Y)
        P = mp.quad(lambda t: mp.exp(t - Y) / mp.sqrt(X * X + t * t), pts)
        PX = -X * mp.quad(lambda t: mp.exp(t - Y) / (X * X + t * t) ** 1.5, pts)
    else:
        P = PX = mp.mpf(0)
    F = -mp.pi * eY * (H0 + Y0) - 2 * P
    FX = -mp.pi * eY * (2 / mp.pi - H1 - Y1) - 2 * PX
    return F, FX


def Ft_3a(X, Y):
    """Ft from the principal-value integral (3a), an independent check (Y > 0):
    PV int_0^inf f(k)/(k-1) dk = int_0^2 (f(k) - f(1))/(k-1) dk + int_2^inf f(k)/(k-1) dk."""
    X, Y = mp.mpf(X), mp.mpf(Y)

    def f(k):
        return mp.exp(-k * Y) * mp.besselj(0, k * X)

    f1 = f(1)
    a = mp.quad(lambda k: (f(k) - f1) / (k - 1), [0, 1, 2])
    b = mp.quad(lambda k: f(k) / (k - 1), [2, mp.inf])
    return 2 * (a + b)


def Ft_series5(X, Y, terms=80):
    """Ft from Newman's uniformly convergent expansion (5), for small R0 checks."""
    X, Y = mp.mpf(X), mp.mpf(Y)
    R0 = mp.sqrt(X * X + Y * Y)
    C = {}
    for n in range(1, 2 * terms + 2):
        C[(0, n)] = 1 / ((n + 1) * mp.factorial(n + 1))
    s = mp.mpf(0)
    for m in range(0, terms):
        for n in range(1, 2 * terms - 2 * m):
            if m > 0:
                C[(m, n)] = -mp.mpf(n + 2) / (n + 1) * C[(m - 1, n + 2)]
            s += C[(m, n)] * X ** (2 * m) * Y**n
    if X == 0:
        head = mp.log(Y) + mp.euler  # limit of J0 asinh(Y/X) + (pi/2) Y0(X) as X -> 0
        tail = R0 * (mp.mpf(1) + s)  # pi H0(X)/(2X) -> 1
    else:
        head = mp.besselj(0, X) * mp.asinh(Y / X) + mp.pi / 2 * mp.bessely(0, X)
        tail = R0 * (mp.pi / (2 * X) * mp.struveh(0, X) + s)
    return -2 * mp.exp(-Y) * (head + tail)
