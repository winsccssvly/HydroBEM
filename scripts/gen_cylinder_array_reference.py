"""Semi-analytic reference for groups of floating truncated vertical circular cylinders in finite
depth: eigenfunction matching at each cylinder (the formulation of Yeung 1981, Appl. Ocean Res. 3,
119-133, and Garrett 1971, J. Fluid Mech. 46, 129-139, as in scripts/gen_cylinder_reference.py)
coupled by the exact multiple scattering of Kagemoto & Yue (1986), J. Fluid Mech. 166, 189-209,
doi:10.1017/S0022112086000101, derived and implemented here.

Radius a = 1, density 1, gravity 1; time factor exp(i w t), K = w^2 / g, k0 tanh k0 h = K,
kn tan kn h = -K; u = z + h, L = h - d (d the draft), lambda_j = j pi / L. About each cylinder q
(local polar coordinates r, theta) the exterior potential is
    phi = sum_m e^(i m theta) sum_n Z_n(u) [A_qmn R_mn(r) + I_qmn S_mn(r)],
    Z_0 = cosh k0 u / cosh k0 h, Z_n = cos kn u,
    R_m0 = H_m^(2)(k0 r) / H_m^(2)(k0) (outgoing), R_mn = K_m(kn r) / K_m(kn),
    S_m0 = J_m(k0 r), S_mn = I_m(kn r) / I_m(kn) (the regular field coming in),
and under it phi = sum_m e^(i m theta) sum_j B_qmj cos(lambda_j u) I_m(lambda_j r) / I_m(lambda_j)
(the j = 0 term r^|m|) plus, in heave, the particular part (u^2 - r^2/2) / (2L). The matching at
r = 1 (the potential on 0 < u < L projected on cos(lambda_j u), the radial velocity on 0 < u < h on
Z_n, with the body's normal velocity on L < u < h; all projections in closed form) eliminates B and
gives, per order m, the outgoing coefficients A_q of the incoming I_q and of the body's motion: the
diffraction and radiation characteristics of one cylinder. The incoming field at q is the incident
wave plus the outgoing fields of the other cylinders, translated to q's centre by Graf's addition
theorems (H_m^(2) to J_l and K_m to I_l; Abramowitz & Stegun 9.1.79, 9.6 with 9.1.79 for the
modified functions); the coupled system for all A_q is solved by GMRES (Kagemoto & Yue's exact
algebraic method). Forces from the potential on the wall and on the bottom (normal out of the body):
A - i B / w = -int n_i phi_j dS (unit velocity), X = -int n_i phi dS of the total potential of the
incident wave of unit amplitude (rho g A = 1); pitch about the cylinder's own axis in z = 0.

The truncation: the orders |m| <= 10 (converged to 1e-9), N = 200, 400, 800 vertical and cosine modes,
extrapolated with the order 4/3 of the corner at the bottom edge (phi ~ rho^(2/3) at the 270-degree
corner: the matched expansions converge like N^(-4/3)); the extrapolations from (200, 400) and
(400, 800) must agree to 5e-5 of the largest value of each quantity (found: 1.7e-6 for "two", 3.9e-5
for "three", whose tall inner region, L = 36.35, resolves the corner with fewer cosine modes per unit
length). One cylinder alone reproduces tests/data/cylinder_finite_depth.txt (scripts/
gen_cylinder_reference.py, the same expansions in another form) to 1e-6 (2e-10 found).

The cases (heading 0, along +x):
- "two": two cylinders of draft 2 in the depth 4 at (-1.3, 0) and (1.3, 0) (geometry 2 of Kagemoto &
  Yue 1986: h/D = 2, H/D = 1, L/D = 1.3), both moving together in one mode (sums over both bodies):
  surge A/(rho a^3), B/(rho w a^3), heave A, B, and the exciting surge and heave |X|/(rho g A a^2)
  summed over both;
- "three": three cylinders of draft 3.65 in the depth 40 at the corners of the equilateral triangle
  of side 5, (5/sqrt 3, 0) and (-5/(2 sqrt 3), +-5/2) (the group of Mavrakos 1991, Ocean Eng. 18,
  485-515, doi:10.1016/0029-8018(91)90027-N), cylinder 1 alone in surge: the surge A, B on cylinder 1,
  the pitch-surge A51, B51 on it (/(rho a^4), /(rho w a^4)), the surge A, B on cylinder 2 (the
  interaction), and the exciting surge |X| on cylinders 1 and 2.

Writes tests/data/cylinder_array.txt:
    two ka surge_A surge_B heave_A heave_B surge_X heave_X
    three ka A11 B11 A51 B51 A21 B21 X1 X2

Usage: python scripts/gen_cylinder_array_reference.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import special as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, gmres

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "cylinder_array.txt"
SINGLE = ROOT / "tests" / "data" / "cylinder_finite_depth.txt"
ORDERS = 10  # the azimuthal orders |m| <= ORDERS
LEVELS = (200, 400, 800)  # the vertical (and cosine) modes of the extrapolation
KA = (0.5, 1.0, 1.6)
S3 = math.sqrt(3.0)
CASES = {
    "two": (4.0, 2.0, ((-1.3, 0.0), (1.3, 0.0))),
    "three": (40.0, 3.65, ((5.0 / S3, 0.0), (-2.5 / S3, 2.5), (-2.5 / S3, -2.5))),
}


class CylinderArray:
    """Identical cylinders of radius 1 and draft ``draft`` at ``centers`` in the depth ``depth``."""

    def __init__(self, depth, draft, centers):
        self.depth, self.draft, self.centers = depth, draft, tuple(centers)

    def evaluate(self, ka, vertical, azimuth, excitations):
        """The integrated forces of unit boundary velocities or of the incident wave: ``excitations``
        lists (body, mode) of radiation or (None, heading in radians) of diffraction; returns
        {"surge", "heave", "pitch"} each [bodies, columns], A - i B / w for radiation, X / (rho g A)
        for diffraction."""
        h, d = self.depth, self.draft
        L = h - d
        k = np.empty(vertical)
        k[0] = ka
        H = ka * math.tanh(ka * h) * h
        for n in range(1, vertical):
            k[n] = brentq(lambda x: x * math.sin(x) + H * math.cos(x), (n - 0.5) * math.pi, n * math.pi) / h
        j = np.arange(vertical)
        lam = j * np.pi / L
        eps = np.where(j == 0, L, L / 2)  # int_0^L cos^2(lambda_j u) du
        # c[n, j] = int_0^L Z_n cos(lambda_j u) du
        c = np.empty((vertical, vertical))
        top = math.cosh(ka * h)
        c[0] = ka * math.sinh(ka * L) * (-1.0) ** j / (ka * ka + lam * lam) / top
        c[1:] = 0.5 * L * (np.sinc((k[1:, None] - lam) * L / np.pi) + np.sinc((k[1:, None] + lam) * L / np.pi))
        norm = h / 2 + np.sin(2 * k * h) / (4 * k)  # int_0^h Z_n^2 du
        norm[0] = (h / 2 + math.sinh(2 * ka * h) / (4 * ka)) / top**2
        wet = (np.sin(k * h) - np.sin(k * L)) / k  # int_L^h Z_n du
        wet[0] = (math.sinh(ka * h) - math.sinh(ka * L)) / (ka * top)
        pitch_wet = d * np.sin(k * L) / k + (np.cos(k * h) - np.cos(k * L)) / k**2  # int_L^h Z_n z du
        pitch_wet[0] = (d * math.sinh(ka * L) / ka - (top - math.cosh(ka * L)) / ka**2) / top
        m_all = np.arange(-azimuth, azimuth + 1)
        count = len(self.centers)
        # the radial functions at r = 1 and their derivatives (outgoing R, regular S)
        hankel = sp.hankel2(m_all, ka)
        kv = sp.kve(m_all[:, None], k[1:])
        regular = np.empty((len(m_all), vertical))
        regular[:, 0] = sp.jv(m_all, ka)
        regular[:, 1:] = 1.0
        d_regular = regular.copy()
        d_regular[:, 0] = ka * sp.jvp(m_all, ka)
        d_regular[:, 1:] = k[1:] * (sp.ive(m_all[:, None] - 1, k[1:]) + sp.ive(m_all[:, None] + 1, k[1:])) / (
            2 * sp.ive(m_all[:, None], k[1:]))  # fmt: skip
        d_outgoing = np.empty_like(regular, dtype=complex)
        d_outgoing[:, 0] = ka * sp.h2vp(m_all, ka) / hankel
        d_outgoing[:, 1:] = -k[1:] * (sp.kve(m_all[:, None] - 1, k[1:]) + sp.kve(m_all[:, None] + 1, k[1:])) / (2 * kv)
        # the interior coefficients eliminated: the radial velocity of the interior field of a given
        # potential on 0 < u < L, projected on Z_n (the Dirichlet-to-Neumann map of the inner region)
        by_m = {}
        for m in range(azimuth + 1):
            dI = np.empty(vertical)
            dI[0] = m
            dI[1:] = lam[1:] * (sp.ive(m - 1, lam[1:]) + sp.ive(m + 1, lam[1:])) / (2 * sp.ive(m, lam[1:]))
            op = ((c * (dI / eps)) @ c.T) / norm[:, None]
            by_m[m] = op, lu_factor(np.diag(d_outgoing[azimuth + m]) - op)
        operators = np.asarray([by_m[abs(m)][0] for m in m_all])
        factors = [by_m[abs(m)][1] for m in m_all]
        # Graf's addition theorems: the outgoing field of ``source`` as regular fields about ``target``
        trans = np.zeros((count, count, vertical, len(m_all), len(m_all)), dtype=complex)
        dm = m_all[None, :] - m_all[:, None]
        sign = np.where(m_all % 2 == 0, 1, -1)
        for t in range(count):
            for s in range(count):
                if t == s:
                    continue
                dx, dy = np.subtract(self.centers[t], self.centers[s])
                dist = math.hypot(dx, dy)
                if dist <= 2:
                    raise ValueError("the cylinders overlap")
                phase = np.exp(1j * dm * math.atan2(dy, dx))
                trans[t, s, 0] = phase * sp.hankel2(dm, ka * dist) / hankel[None, :]
                x = k[1:, None, None]
                trans[t, s, 1:] = (phase[None] * sign[None, :, None] * sp.kve(dm, x * dist)
                                   * sp.ive(m_all, k[1:, None])[:, :, None] / kv.T[:, None, :]
                                   * np.exp(-x * (dist - 2)))  # fmt: skip
        shape = (count, len(m_all), vertical)

        def translated(out):
            return np.einsum("tsnml,sln->tmn", trans, out, optimize=True)

        def local(incoming, forcing=None):
            """The outgoing coefficients of each cylinder alone for its incoming field and forcing."""
            v = np.einsum("mij,bmj->bmi", operators, regular * incoming) - d_regular * incoming
            if forcing is not None:
                v += forcing
            return np.stack([lu_solve(f, v[:, i, :].T).T for i, f in enumerate(factors)], axis=1)

        def coupled(x):
            out = x.reshape(shape)
            return (out - local(translated(out))).ravel()

        system = LinearOperator((int(np.prod(shape)),) * 2, matvec=coupled, dtype=np.complex128)
        # the heave particular part (u^2 - r^2/2) / (2L) at r = 1 projected on cos(lambda_j u), / eps
        particular = -L * (-1.0) ** j / (np.pi * np.maximum(j, 1)) ** 2
        particular[0] = -L / 6 + 1 / (4 * L)
        particular[1:] *= 2
        radial_heave = np.empty(vertical)  # int_0^1 I_0(lambda r) / I_0(lambda) r dr, cos(lambda L) = (-1)^j
        radial_heave[0] = 0.5
        radial_heave[1:] = (-1.0) ** j[1:] * sp.ive(1, lam[1:]) / (lam[1:] * sp.ive(0, lam[1:]))
        radial_pitch = np.empty(vertical)  # int_0^1 I_1(lambda r) / I_1(lambda) r^2 dr
        radial_pitch[0] = 0.25
        radial_pitch[1:] = (-1.0) ** j[1:] * sp.ive(2, lam[1:]) / (lam[1:] * sp.ive(1, lam[1:]))
        out = {name: [] for name in ("surge", "heave", "pitch")}
        m0, mm, mp = azimuth, azimuth - 1, azimuth + 1
        for body, mode in excitations:
            incident = np.zeros(shape, dtype=complex)
            forcing = np.zeros(shape, dtype=complex)
            if body is None:  # exp(-i k x') of unit amplitude: Jacobi-Anger about each centre
                along = np.asarray(self.centers) @ np.array([math.cos(mode), math.sin(mode)])
                incident[:, :, 0] = np.exp(-1j * ka * along[:, None]) * (-1j) ** m_all * np.exp(-1j * m_all * mode)
            elif mode == "surge":  # n_r = cos(theta) on the wall: m = +-1, 1/2 each
                forcing[body, [mm, mp], :] = 0.5 * wet / norm
            elif mode == "heave":
                dI = np.r_[0, lam[1:] * sp.ive(1, lam[1:]) / sp.ive(0, lam[1:])]
                forcing[body, m0] = (c @ (dI * particular) - c[:, 0] / (2 * L)) / norm
            else:
                raise ValueError(f"mode {mode}")
            rhs = local(incident, forcing)
            x, info = gmres(system, rhs.ravel(), rtol=1e-12, atol=1e-14, restart=40, maxiter=20)
            if info:
                raise RuntimeError(f"GMRES did not converge at ka = {ka:g} ({info})")
            x = x.reshape(shape)
            wall = x + regular * (incident + translated(x))  # the potential's Z_n coefficients at r = 1
            bottom = wall[:, m0] @ c / eps
            if body is not None and mode == "heave":
                bottom[body] += particular
            horizontal = wall[:, mm] + wall[:, mp]
            out["surge"].append(-np.pi * (horizontal @ wet))
            heave = 2 * np.pi * (bottom @ radial_heave)
            if body is not None and mode == "heave":
                heave[body] += 2 * np.pi * (L / 4 - 1 / (16 * L))
            out["heave"].append(heave)
            out["pitch"].append(-np.pi * (horizontal @ pitch_wet + (horizontal @ c / eps) @ radial_pitch))
        return {key: np.stack(cols, axis=1) for key, cols in out.items()}


def quantities(case, ka, vertical):
    """The written quantities of a case at one ka and truncation (see the docstring)."""
    depth, draft, centers = CASES[case]
    a = CylinderArray(depth, draft, centers)
    if case == "two":
        v = a.evaluate(ka, vertical, ORDERS, [(0, "surge"), (1, "surge"), (0, "heave"), (1, "heave"), (None, 0.0)])
        surge, heave = v["surge"][:, :2].sum(), v["heave"][:, 2:4].sum()
        return np.array([surge.real, -surge.imag, heave.real, -heave.imag,
                         abs(v["surge"][:, 4].sum()), abs(v["heave"][:, 4].sum())])  # fmt: skip
    v = a.evaluate(ka, vertical, ORDERS, [(0, "surge"), (None, 0.0)])
    s11, s51, s21 = v["surge"][0, 0], v["pitch"][0, 0], v["surge"][1, 0]
    return np.array([s11.real, -s11.imag, s51.real, -s51.imag, s21.real, -s21.imag,
                     abs(v["surge"][0, 1]), abs(v["surge"][1, 1])])  # fmt: skip


def converged(case):
    """The rows of a case extrapolated in N (order 4/3), and the agreement of the two extrapolations
    relative to the largest value of each quantity."""
    v = {n: np.array([quantities(case, ka, n) for ka in KA]) for n in LEVELS}
    f = 2 ** (4 / 3) - 1
    e1 = v[LEVELS[1]] + (v[LEVELS[1]] - v[LEVELS[0]]) / f
    e2 = v[LEVELS[2]] + (v[LEVELS[2]] - v[LEVELS[1]]) / f
    err = np.abs(e2 - e1).max(axis=0) / np.abs(e2).max(axis=0)
    if err.max() > 5e-5:
        sys.exit(f"no convergence for {case}: {err}")
    return e2, err


def check_single():
    """One cylinder against scripts/gen_cylinder_reference.py (tests/data/cylinder_finite_depth.txt,
    h/a = 2, d/a = 1): heave and surge A, B / w, extrapolated as above, to 1e-6 of the largest."""
    ref = np.loadtxt(SINGLE)
    ref = ref[ref[:, 0] == 2.0]
    one = CylinderArray(2.0, 1.0, ((0.0, 0.0),))
    got = []
    for K in ref[:, 2]:
        ka = brentq(lambda x, K=K: x * math.tanh(2.0 * x) - K, 1e-9, 50.0)
        v = {}
        for n in LEVELS[1:]:
            r = one.evaluate(ka, n, 2, [(0, "heave"), (0, "surge")])
            heave, surge = r["heave"][0, 0], r["surge"][0, 1]
            v[n] = np.array([heave.real, -heave.imag, surge.real, -surge.imag])
        got.append(v[LEVELS[2]] + (v[LEVELS[2]] - v[LEVELS[1]]) / (2 ** (4 / 3) - 1))
    err = np.abs(np.array(got) - ref[:, 3:]).max() / np.abs(ref[:, 3:]).max()
    if err > 1e-6:
        sys.exit(f"one cylinder disagrees with {SINGLE.name}: {err:.1e}")
    return err


def main():
    print(f"one cylinder against {SINGLE.name}: {check_single():.1e}")
    lines = [
        "# Generated by scripts/gen_cylinder_array_reference.py -- do not edit.",
        "# Truncated vertical cylinders a = 1 in finite depth, heading 0: eigenfunction matching (Yeung 1981;",
        "# Garrett 1971) and the multiple scattering of Kagemoto & Yue (1986), J. Fluid Mech. 166, derived and",
        "# implemented in the script; orders |m| <= 10, N = 200, 400, 800 vertical modes extrapolated (order 4/3),",
        "# the two extrapolations agree to 1.7e-6 (two) and 3.9e-5 (three) of the largest value of each quantity.",
        "# two: draft 2, depth 4, centres (+-1.3, 0) (Kagemoto & Yue geometry 2), both in one mode, sums over both:",
        "#   ka surge A/(rho a^3) B/(rho w a^3) heave A B, surge |X|/(rho g A a^2) heave |X|",
        "# three: draft 3.65, depth 40, triangle of side 5 (Mavrakos 1991), cylinder 1 at (5/sqrt3, 0) in surge:",
        "#   ka A11 B11 A51 B51 (on 1, pitch / a^4) A21 B21 (surge on 2), |X1| |X2| (surge)",
    ]
    for case in CASES:
        rows, err = converged(case)
        print(f"{case}: the extrapolations agree to {np.array2string(err, precision=1)}")
        lines += [f"{case} {ka} " + " ".join(f"{x:.9e}" for x in row) for ka, row in zip(KA, rows, strict=True)]
    OUT.write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
