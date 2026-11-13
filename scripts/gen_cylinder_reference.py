"""Semi-analytic reference for test A5 (test-plan): heave and surge added mass
and damping of a floating truncated vertical circular cylinder (radius a,
draft d) in water of finite depth h, by eigenfunction matching (the method of
Yeung 1981, Appl. Ocean Res. 3; Garrett 1971), derived and implemented here.

Time factor exp(i w t), K = w^2/g, k0 tanh k0 h = K, kn tan kn h = -K;
u = z + h. Exterior r > a, 0 < u < h:
    phi = cos(m theta) sum_n A_n Z_n(u) R_n(r),  Z_0 = cosh k0 u, Z_n = cos kn u,
    R_0 = H_m^(2)(k0 r) / H_m^(2)(k0 a) (outgoing waves), R_n = K_m(kn r) / K_m(kn a);
interior r < a, 0 < u < L = h - d, lambda_j = j pi / L, c_j = cos lambda_j u:
    heave (m = 0): phi = (u^2 - r^2/2) / (2L) + sum_j B_j c_j I_0(lambda_j r) / I_0(lambda_j a)
                   (the particular part has d phi/du = 1 on the body bottom u = L);
    surge (m = 1): phi = cos theta [B_0 r/a + sum_{j>=1} B_j c_j I_1(lambda_j r) / I_1(lambda_j a)].
Matching at r = a: the potential on 0 < u < L projected on c_j (Galerkin),
the radial velocity on 0 < u < h projected on Z_n, with d phi/dr = n_r of the
mode on the wetted side L < u < h. All projection integrals are in closed form.

Forces (normal out of the body into the fluid, theory §2.1):
    A - i B / w = -rho int n_i phi_i dS:
    heave:  rho int_0^a phi(r, L) 2 pi r dr,   surge: -rho pi a sum_n A_n int_L^h Z_n du.

The truncation N = J terms is 200, 400, 800, extrapolated (see converged()).

Writes tests/data/cylinder_finite_depth.txt:
    h/a d/a Ka  A33/(rho a^3)  B33/(rho w a^3)  A11/(rho a^3)  B11/(rho w a^3)

Usage: python scripts/gen_cylinder_reference.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import special as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "cylinder_finite_depth.txt"
CASES = [(2.0, 1.0, ka) for ka in (0.4, 0.8, 1.5)] + [(4.0, 1.0, ka) for ka in (0.4, 0.8, 1.5)]


def roots(H, n):
    """x0 tanh x0 = H and xn tan xn = -H, n = 1..n (Newton with bisection)."""
    x = max(H, math.sqrt(H))
    for _ in range(200):
        t = math.tanh(x)
        dx = (x * t - H) / (t + x * (1 - t * t))
        x -= dx
        if abs(dx) < 1e-15 * x:
            break
    out = [x]
    for k in range(1, n + 1):
        lo, hi = (k - 0.5) * math.pi, k * math.pi
        y = k * math.pi - math.atan(H / (k * math.pi))
        for _ in range(200):
            f = y * math.tan(y) + H
            if f > 0:
                hi = y
            else:
                lo = y
            yn = y - f / (math.tan(y) + y / math.cos(y) ** 2)
            if not (lo < yn < hi):
                yn = 0.5 * (lo + hi)
            if abs(yn - y) < 1e-15 * y:
                y = yn
                break
            y = yn
        out.append(y)
    return np.array(out)


def ccos(p, q, L):
    """int_0^L cos(p u) cos(q u) du."""
    if abs(p - q) < 1e-14 * max(1.0, p):
        return L if p == 0 else L / 2 + math.sin(2 * p * L) / (4 * p)
    return 0.5 * (math.sin((p - q) * L) / (p - q) + (math.sin((p + q) * L) / (p + q) if p + q else L))


def ccosh(k, q, L):
    """int_0^L cosh(k u) cos(q u) du."""
    return (k * math.sinh(k * L) * math.cos(q * L) + q * math.cosh(k * L) * math.sin(q * L)) / (k * k + q * q)


def solve(h, d, ka, N, mode):
    a = 1.0
    K = ka / a
    L = h - d
    x = roots(K * h, N - 1)
    k = x / h
    lam = np.arange(N) * math.pi / L
    # projections
    C = np.empty((N, N))  # C[n, j] = int_0^L Z_n c_j
    for n in range(N):
        for j in range(N):
            C[n, j] = ccosh(k[0], lam[j], L) if n == 0 else ccos(k[n], lam[j], L)
    Nn = np.array(
        [h / 2 + math.sinh(2 * k[0] * h) / (4 * k[0])] + [h / 2 + math.sin(2 * kk * h) / (4 * kk) for kk in k[1:]]
    )
    eps = np.array([L] + [L / 2] * (N - 1))
    wet = np.array(  # int_L^h Z_n du
        [(math.sinh(k[0] * h) - math.sinh(k[0] * L)) / k[0]]
        + [(math.sin(kk * h) - math.sin(kk * L)) / kk for kk in k[1:]]
    )
    m = 0 if mode == "heave" else 1
    # exterior radial derivative R_n'(a)
    H2 = sp.hankel2(m, k[0] * a)
    dH2 = sp.h2vp(m, k[0] * a)
    Rp = np.empty(N, dtype=complex)
    Rp[0] = k[0] * dH2 / H2
    kn = k[1:] * a
    Rp[1:] = (
        -k[1:] * (sp.kve(abs(m - 1), kn) + sp.kve(m + 1, kn)) / (2 * sp.kve(m, kn))
    )  # K_m' = -(K_{m-1} + K_{m+1})/2
    # interior radial derivative D_j
    D = np.empty(N)
    if m == 0:
        D[0] = 0.0
        D[1:] = lam[1:] * sp.ive(1, lam[1:] * a) / sp.ive(0, lam[1:] * a)
    else:
        D[0] = 1.0 / a
        x1 = lam[1:] * a
        D[1:] = lam[1:] * (sp.ive(0, x1) + sp.ive(2, x1)) / (2 * sp.ive(1, x1))  # I_1' = (I_0 + I_2)/2
    # unknowns [A_0..A_{N-1}, B_0..B_{N-1}]
    M = np.zeros((2 * N, 2 * N), dtype=complex)
    rhs = np.zeros(2 * N, dtype=complex)
    for j in range(N):  # potential continuity on 0 < u < L
        M[j, :N] = -C[:, j]
        M[j, N + j] = eps[j]
        if m == 0:
            # int_0^L (u^2 - a^2/2) / (2L) c_j du
            q = lam[j]
            u2 = L**3 / 3 if j == 0 else ((q * q * L * L - 2) * math.sin(q * L) + 2 * q * L * math.cos(q * L)) / q**3
            c0 = L if j == 0 else math.sin(q * L) / q
            rhs[j] = -(u2 - a * a / 2 * c0) / (2 * L)
    for n in range(N):  # radial velocity on 0 < u < h
        M[N + n, n] = Rp[n] * Nn[n]
        M[N + n, N:] = -D * C[n, :]
        if m == 0:
            rhs[N + n] = -a / (2 * L) * C[n, 0]  # d phi_p / dr = -r / (2L), int_0^L Z_n
        else:
            rhs[N + n] = wet[n]  # d phi / dr = cos theta on the wetted side
    sol = np.linalg.solve(M, rhs)
    A, B = sol[:N], sol[N:]
    if m == 0:
        # rho int_0^a phi(r, L) 2 pi r dr, c_j(L) = (-1)^j, int_0^a I0(l r) r dr = a I1(l a) / l
        part = L * a * a / 4 - a**4 / (16 * L)
        s = B[0] * a * a / 2
        for j in range(1, N):
            s += B[j] * (-1) ** j * a * sp.ive(1, lam[j] * a) / (lam[j] * sp.ive(0, lam[j] * a))
        F = 2 * math.pi * (part + s)
    else:
        F = -math.pi * a * np.sum(A * wet)
    w = math.sqrt(K)  # g = 1
    return F.real, -F.imag * w / w  # A, B / w  (A - i B / w = F)


def converged(h, d, ka, mode):
    """Richardson extrapolation in N with the known order 4/3 (the fluid sees a
    270-degree corner at the bottom edge, phi ~ rho^(2/3), so the truncation
    error of the matched expansions decays like N^(-4/3)); the extrapolations
    from (200, 400) and (400, 800) must agree to 5e-6 of the largest value."""
    v = {N: np.array(solve(h, d, ka, N, mode)) for N in (200, 400, 800)}
    f = 2 ** (4 / 3) - 1
    e1 = v[400] + (v[400] - v[200]) / f
    e2 = v[800] + (v[800] - v[400]) / f
    err = np.abs(e2 - e1).max() / np.abs(e2).max()
    if err > 5e-6:
        sys.exit(f"no convergence for {mode} h={h} d={d} Ka={ka}: {e1} vs {e2}")
    return e2, err


def main():
    lines = [
        "# Generated by scripts/gen_cylinder_reference.py -- do not edit.",
        "# Eigenfunction matching for a truncated vertical cylinder in finite depth (the formulation of",
        "# Yeung 1981, Appl. Ocean Res. 3, 119-133; Garrett 1971, J. Fluid Mech. 46; heave as Bhatta and",
        "# Rahman 1995, surge as Bhatta 2003), derived and implemented in this script; truncation",
        "# converged and extrapolated (the last digits of the extrapolation printed by the script).",
        "# h/a d/a Ka A33/(rho a^3) B33/(rho w a^3) A11/(rho a^3) B11/(rho w a^3)",
    ]
    for h, d, ka in CASES:
        (a3, b3), n3 = converged(h, d, ka, "heave")
        (a1, b1), n1 = converged(h, d, ka, "surge")
        print(
            f"h/a={h} d/a={d} Ka={ka}: heave {a3:.7f} {b3:.7f} (extrapolation {n3:.1e}), "
            f"surge {a1:.7f} {b1:.7f} (extrapolation {n1:.1e})"
        )
        lines.append(f"{h} {d} {ka} {a3:.9e} {b3:.9e} {a1:.9e} {b1:.9e}")
    OUT.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


if __name__ == "__main__":
    main()
