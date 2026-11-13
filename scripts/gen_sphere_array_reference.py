"""Semi-analytic reference for two submerged spheres in deep water: the multipole method of
Wu (1995), The interaction of water waves with a group of submerged spheres, Applied Ocean
Research 17(3), 165-184, doi:10.1016/0141-1187(95)00010-0 (§2-§4; the case of its §7), derived
and implemented here.

Radius a = 1, density rho = 1, gravity g = 1; time factor exp(i w t), nu = w^2 / g. The two spheres
are centred at (-2, 0, -1.5) and (2, 0, -1.5) (spacing 4a, centres 1.5a below the free surface, as
Wu §7), the waves come at 45 degrees.

About each centre c_q the potential is expanded in spherical harmonics Y_lm (l <= L) of the local
polar angles, phi = sum_lm [s_lm r^-(l+1) + t_lm r^l] Y_lm. The singular coefficients s_lm are the
unknowns. Each singular term radiates with the deep-water Green function, the multipole of the
Rankine source 1/r, its rigid-lid image 1/r' (the multipole reflected in z = 0: the factor
(-1)^(l+m) at the reflected centre) and the wave part
    2 nu int_0^inf exp(k (z + zeta)) J_0(k R) / (k - nu) dk,  the pole passed below
(outgoing waves: the principal value and -i pi times the residue). The regular coefficients t_lm at a
centre are the projection on Y_lm of the field of every singular term but its own (the other sphere's
multipoles, every image) and of the incident wave. The projection is numerical: Gauss-Legendre in
the polar angle (3L + 8 points) and the trapezoidal rule in azimuth (6L + 16 points) on the unit
sphere about the centre, exact for the harmonics of degree <= 2L + 7 and resolving the Rankine
fields whose nearest singularity is at 3a or more. The wave part is projected in closed form: a
plane harmonic exp(k (z + i (x cos a + y sin a))) has a direction vector of zero square, so only
its Taylor term of degree l contributes to the degree l projection (its moment); the azimuthal
average over a gives J_(m - m')(k R) between the centres (Graf's addition theorem). The k integral
is Gauss-Legendre on [0, 48] (Q points) of the pole-subtracted integrand, the subtracted pole and the
residue in closed form (the integrand is below exp(-3 k) k^(2L): < 1e-28 at 48).

The Neumann condition on r = 1, degree by degree: -(l + 1) s_lm + l t_lm = v_lm, the projection of
the normal velocity (radiation: both spheres moving together in surge, sway or heave, Wu §7) or
minus that of the incident wave (diffraction).

Forces on sphere 2 (normal out of the body into the fluid): F = -int n phi dS (rho = 1), so
A - i B / w = F of the radiation potential (unit velocity) and |E| = |F| of the total diffraction
potential (unit amplitude, rho g A = 1). Normalized as Wu (45)-(46): A / (rho V), B / (rho w V),
|E| / (rho w^2 V A exp(-nu S)), V = 4 pi a^3 / 3, S = 1.5a the depth of the centres; A, B are the sums
A^(21) + A^(22) (the force on sphere 2 when both spheres move).

The truncation (L, Q) = (10, 160) is written; (8, 128) must agree with it to 2e-6 of the largest
value of each quantity (7e-7 found; the change from (10, 160) to (12, 192) is below 1e-8).

Writes tests/data/two_spheres.txt:
    nu_a  surge A B E  sway A B E  heave A B E  (normalized as above)

Usage: python scripts/gen_sphere_array_reference.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import special as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "two_spheres.txt"
CENTERS = ((-2.0, 0.0, -1.5), (2.0, 0.0, -1.5))  # in units of the radius
HEADING = 45.0  # deg
NU_A = (0.5, 1.4, 2.5)
TRUNCATIONS = ((8, 128), (10, 160))  # (L, Q): the largest degree, the k-integral points
CUTOFF = 48.0  # the k integral on [0, CUTOFF]


class SphereArray:
    """The multipole system of the unit spheres at ``centers`` truncated at degree ``order``, with
    ``points`` Gauss points of the wave integral."""

    def __init__(self, centers, order, points):
        self.centers = np.asarray(centers, dtype=np.float64)
        self.count = len(self.centers)
        self.l = np.repeat(np.arange(order + 1), 2 * np.arange(order + 1) + 1)
        self.m = np.concatenate([np.arange(-n, n + 1) for n in range(order + 1)])
        self.size = len(self.l)
        self.degree = np.tile(self.l, self.count)
        self.mirror = (-1.0) ** (self.l + self.m)  # Y_lm reflected in z = 0
        self.factor = (2 * self.l + 1) / (4 * np.pi)  # the addition theorem of the Legendre polynomials
        mu, wt = sp.roots_legendre(3 * order + 8)
        n_az = 6 * order + 16
        azimuth = np.arange(n_az) * 2 * np.pi / n_az
        theta, az = np.meshgrid(np.arccos(mu), azimuth, indexing="ij")
        theta, az = theta.ravel(), az.ravel()
        weights = np.repeat(wt, n_az) * 2 * np.pi / n_az
        self.points = np.column_stack((np.sin(theta) * np.cos(az), np.sin(theta) * np.sin(az), np.cos(theta)))
        harmonics = sp.sph_harm_y(self.l[None, :], self.m[None, :], theta[:, None], az[:, None])
        self.project = harmonics.conj().T * weights  # a field on the unit sphere -> its Y_lm coefficients
        self.normals = self.project @ self.points  # n_x, n_y, n_z (out of the sphere)
        # the degree-l Taylor term of exp(z + i x) (direction (i, 0, 1), zero square) projected on Y_lm
        plane = self.points[:, 2] + 1j * self.points[:, 0]
        self.moment = np.sum(self.project * plane[None, :] ** self.l[:, None], axis=1) / sp.factorial(self.l)
        x, w = sp.roots_legendre(points)
        self.grid = (x + 1) * CUTOFF / 2
        self.weights = w * CUTOFF / 2
        self.blocks = {}
        for target in range(self.count):
            for source in range(self.count):
                image = self.centers[source] * [1, 1, -1]
                block = self.rankine(self.centers[target] - image) * self.mirror
                if target != source:
                    block += self.rankine(self.centers[target] - self.centers[source])
                self.blocks[target, source] = block, self.wave(self.grid, target, source)

    def rankine(self, shift):
        """The projection on the target sphere of r^-(l+1) Y_lm about a centre at -shift from it."""
        p = self.points + shift
        r = np.linalg.norm(p, axis=1)
        theta, az = np.arccos(p[:, 2] / r), np.arctan2(p[:, 1], p[:, 0])
        values = sp.sph_harm_y(self.l[None, :], self.m[None, :], theta[:, None], az[:, None])
        return self.project @ (values / r[:, None] ** (self.l + 1))

    def wave(self, k, target, source):
        """The wave integrand at the wavenumbers k: the regular projection at ``target`` of the
        exp(k (z + zeta)) J_0(k R) part of the multipoles of ``source`` ([k, row, column])."""
        x, y, z = self.centers[target]
        u, v, zs = self.centers[source]
        distance = np.hypot(x - u, y - v)
        alpha = np.arctan2(y - v, x - u)
        delta = self.m[None, :] - self.m[:, None]
        moment = self.moment * k[:, None] ** self.l
        # the azimuthal average of exp(i k D cos(a - alpha)) exp(i (m_source - m_target) a): Bessel's integral
        translation = sp.jv(delta, k[:, None, None] * distance) * (1j**delta) * np.exp(1j * delta * alpha)
        return (np.exp(k * (z + zs))[:, None, None] * translation * moment[:, :, None]
                * moment.conj()[:, None, :] * self.factor)  # fmt: skip

    def evaluate(self, nu, heading):
        """Sphere 2 (the last): surge, sway, heave A / V, B / (w V), |E| / (w^2 V exp(-nu S))."""
        n = self.count * self.size
        regular = np.empty((n, n), dtype=complex)
        for (target, source), (static, grid) in self.blocks.items():
            pole = self.wave(np.array([nu]), target, source)[0]
            # 1 / (k - nu + i0): the pole subtracted, its principal value and residue in closed form
            gap = self.grid - nu
            at = abs(gap) < 1e-8
            divided = (grid - pole) / np.where(at, 1, gap)[:, None, None]
            if np.any(at):
                step = 1e-5
                pair = self.wave(np.array([nu - step, nu + step]), target, source)
                divided[at] = (pair[1] - pair[0]) / (2 * step)
            integral = np.einsum("k,kij->ij", self.weights, divided)
            integral += (np.log((CUTOFF - nu) / nu) - 1j * np.pi) * pole
            rows = slice(target * self.size, (target + 1) * self.size)
            cols = slice(source * self.size, (source + 1) * self.size)
            regular[rows, cols] = static + 2 * nu * integral
        matrix = self.degree[:, None] * regular - np.diag(self.degree + 1)
        velocity = np.tile(self.normals, (self.count, 1))  # both spheres moving in one mode
        along = self.centers[:, 0] * np.cos(heading) + self.centers[:, 1] * np.sin(heading)
        incident = (np.exp(nu * (self.centers[:, 2] - 1j * along))[:, None] * self.moment
                    * nu**self.l * np.exp(-1j * self.m * (heading + np.pi))).ravel()  # fmt: skip
        rhs = np.column_stack((velocity, -self.degree * incident))
        singular = np.linalg.solve(matrix, rhs)
        potential = singular + regular @ singular  # on r = 1: s_lm + t_lm
        potential[:, -1] += incident
        force = -self.normals.conj().T @ potential[-self.size:]
        radiation = np.diag(force[:, :3])
        exciting = abs(force[:, 3]) / (nu * np.exp(nu * self.centers[-1, 2]))
        volume = 4 * np.pi / 3
        return np.column_stack((radiation.real, -radiation.imag, exciting)).ravel() / volume


def table():
    """The rows (nu a, the nine quantities) at the written truncation, and the largest change from
    the coarser one relative to the largest value of each quantity."""
    heading = np.deg2rad(HEADING)
    values = []
    for order, points in TRUNCATIONS:
        spheres = SphereArray(CENTERS, order, points)
        values.append(np.array([spheres.evaluate(nu, heading) for nu in NU_A]))
    change = np.abs(values[1] - values[0]).max(axis=0) / np.abs(values[1]).max(axis=0)
    if not np.all(np.isfinite(values[1])) or change.max() > 2e-6:
        sys.exit(f"the truncation has not converged: {change}")
    return np.column_stack((NU_A, values[1])), change


def text(rows) -> str:
    lines = [
        "# Generated by scripts/gen_sphere_array_reference.py -- do not edit.",
        "# Two submerged spheres in deep water, radius a = 1, centres (-2, 0, -1.5) and (2, 0, -1.5), waves",
        "# at 45 deg: the multipole method of Wu (1995), Appl. Ocean Res. 17, 165-184 (section 7), derived and",
        "# implemented in the script; truncation degree 10, 160 wave-integral points (degree 8 agrees to 7e-7",
        "# of the largest).",
        "# Sphere 2, both spheres moving: A/(rho V), B/(rho w V) of A21 + A22, |E|/(rho w^2 V A exp(-nu S)).",
        "# nu_a surge_A surge_B surge_E sway_A sway_B sway_E heave_A heave_B heave_E",
    ]
    lines += [" ".join(f"{x:.9e}" if i else f"{x}" for i, x in enumerate(row)) for row in rows]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check that the table is up to date")
    a = ap.parse_args()
    rows, change = table()
    t = text(rows)
    if a.check:
        if OUT.read_text(encoding="ascii") != t:
            sys.exit(f"{OUT} is not up to date: run scripts/gen_sphere_array_reference.py")
        print("up to date")
        return
    print("truncation change per quantity:", np.array2string(change, precision=2))
    OUT.write_bytes(t.encode("ascii"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
