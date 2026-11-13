"""Bound constant of the per-pair log-integral rule (LOG_BOUND in
crates/hydrobem/src/lo_static.rs). The correction the per-pair rule skips: delta = int_j psi dS - A psi(c),
delta_n = int_j dpsi/dn dS - A dpsi/dn(c), psi = log(R + |d_z|), d = xi - p,
p the image point (z >= 0) of the field point, for square and triangular
panels of radius a (largest centroid-vertex distance) in every orientation, at
distances d = R(c) + |d_z(c)| (the measure of the rule). Bound constants
C = max |delta| / A / (a/d)^2 and C_n = max |delta_n| d^3 / (A a^2), sampled over 400 random
orientations, depths and image points per shape (2.5 <= d/a <= 60, a 48 x 48 Gauss rule).
The engine takes kLogBound = 0.6, a margin over the largest C (0.56). C_n is printed for the
record: the rule bounds delta only, the normal-derivative correction being of one order of a/d
higher in this measure.

    python scripts/log_correction_bound.py
"""
import numpy as np

rng = np.random.default_rng(1)
g, w = np.polynomial.legendre.leggauss(48)


def psi(xi, p, n):
    d = xi - p
    R = np.linalg.norm(d, axis=-1)
    den = R - d[..., 2]  # R + |d_z|, d_z <= 0
    grad = (d / R[..., None] - np.array([0, 0, 1.0])) / den[..., None]  # d psi / d xi
    return np.log(den), grad @ n


def panel_integral(corners, p):
    """Gauss rule on the bilinear map of the corners (a triangle repeats a corner)."""
    a, b, c, dd = corners
    u, v = np.meshgrid(g, g, indexing="ij")
    W = np.outer(w, w)
    n0, n1, n2, n3 = (0.25 * (1 - u) * (1 - v), 0.25 * (1 + u) * (1 - v), 0.25 * (1 + u) * (1 + v),
                      0.25 * (1 - u) * (1 + v))  # fmt: skip
    x = n0[..., None] * a + n1[..., None] * b + n2[..., None] * c + n3[..., None] * dd
    xu = (0.25 * (1 - v))[..., None] * (b - a) + (0.25 * (1 + v))[..., None] * (c - dd)
    xv = (0.25 * (1 - u))[..., None] * (dd - a) + (0.25 * (1 + u))[..., None] * (c - b)
    cr = np.cross(xu, xv)
    jac = np.linalg.norm(cr, axis=-1)
    nrm = cr.sum(axis=(0, 1))
    nrm /= np.linalg.norm(nrm)
    f, fn = psi(x, p, nrm)
    A = np.sum(W * jac)
    cen = np.sum((W * jac)[..., None] * x, axis=(0, 1)) / A
    fc, fnc = psi(cen[None, None], p, nrm)
    return A, cen, np.sum(W * jac * f) - A * fc[0, 0], np.sum(W * jac * fn) - A * fnc[0, 0]


def rotation(r):
    q = r.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])  # fmt: skip


shapes = {
    "square": np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float),
    "rect 4:1": np.array([[-2, -0.5, 0], [2, -0.5, 0], [2, 0.5, 0], [-2, 0.5, 0]], float),
    "triangle": np.array([[-1, -1, 0], [1, -1, 0], [0, 1.2, 0], [0, 1.2, 0]], float),
}
for name, base in shapes.items():
    best, best_n = 0.0, 0.0
    for _ in range(400):
        Rm = rotation(rng)
        scale = 0.05
        corners = (base @ Rm.T) * scale
        cen0 = corners.mean(axis=0)
        a = np.max(np.linalg.norm(corners - cen0, axis=1))
        # panel centroid below z = 0 at a random depth, image point at a random place
        dist = a * 10 ** rng.uniform(np.log10(2.5), np.log10(60))
        depth = rng.uniform(0, 1) * dist
        c_z = -max(depth, 1.5 * a)
        corners = corners - cen0 + np.array([0, 0, c_z])
        if np.max(corners[:, 2]) > -1e-9:
            continue
        # p: image of a field point (p_z >= 0) so that R(c) + |d_z(c)| = dist
        t = rng.uniform(0, 1)
        pz = max(0.0, t * dist - abs(c_z))
        horiz = rng.normal(size=2)
        horiz /= np.linalg.norm(horiz)
        dz = pz - c_z
        if dist - dz <= dz:
            continue
        ph = np.sqrt(max((dist - dz) ** 2 - dz ** 2, 0.0)) * horiz
        p = np.array([ph[0], ph[1], pz])
        A, cen, dl, dd = panel_integral(corners, p)
        dmeas = np.linalg.norm(cen - p) + abs(cen[2] - p[2])
        best = max(best, abs(dl) / A / (a / dmeas) ** 2)
        best_n = max(best_n, abs(dd) * dmeas ** 3 / (A * a * a))
    print(f"{name:9s}  C = {best:.3f}   C_n = {best_n:.3f}")
