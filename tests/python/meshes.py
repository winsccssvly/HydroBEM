"""Procedural test meshes (normals pointing out of the body into the fluid)."""

from __future__ import annotations

import math

import numpy as np


def orient(quad, normal):
    q = np.asarray(quad, dtype=float)
    n = np.cross(q[2] - q[0], q[3] - q[1])
    return q if np.dot(n, normal) > 0 else q[[0, 3, 2, 1]]


def face_panels(origin, du, dv, nu, nv, normal):
    origin, du, dv = (np.asarray(v, dtype=float) for v in (origin, du, dv))
    out = []
    a, b = du / nu, dv / nv
    for i in range(nu):
        for j in range(nv):
            p = origin + a * i + b * j
            out.append(orient([p, p + a, p + a + b, p + b], normal))
    return out


def box_panels(a=1.0, b=0.5, d=0.5, n=(4, 2, 2), quarter=False, lid=False):
    """Surface-piercing box |x| <= a, |y| <= b, -d <= z <= 0 as (P, 4, 3).

    quarter: only the part x >= 0, y >= 0 (for the symmetry planes x = 0 and y = 0); lid: append
    the interior water plane z = 0 with upward normals after the hull panels.
    Panel counts per face: bottom nx*ny, x faces ny*nz, y faces nx*nz.
    """
    nx, ny, nz = n
    x0, y0 = (0.0, 0.0) if quarter else (-a, -b)
    lx, ly = a - x0, b - y0
    panels = []
    panels += face_panels((x0, y0, -d), (lx, 0, 0), (0, ly, 0), nx, ny, (0, 0, -1))
    panels += face_panels((a, y0, -d), (0, ly, 0), (0, 0, d), ny, nz, (1, 0, 0))
    panels += face_panels((x0, b, -d), (lx, 0, 0), (0, 0, d), nx, nz, (0, 1, 0))
    if not quarter:
        panels += face_panels((-a, y0, -d), (0, ly, 0), (0, 0, d), ny, nz, (-1, 0, 0))
        panels += face_panels((x0, -b, -d), (lx, 0, 0), (0, 0, d), nx, nz, (0, -1, 0))
    if lid:
        panels += face_panels((x0, y0, 0.0), (lx, 0, 0), (0, ly, 0), nx, ny, (0, 0, 1))
    return np.array(panels)


def hemisphere_panels(radius=1.0, n_theta=8, n_phi=8, quarter=True, graded=False):
    """Floating hemisphere z <= 0 (vertices on the sphere). quarter: the
    part x >= 0, y >= 0 (symmetric about x = 0 and y = 0). n_theta divisions in polar angle
    from the waterline to the bottom pole, n_phi in azimuth over the part;
    graded: polar divisions refined towards the waterline (sine spacing)."""
    phis = np.linspace(0.0, np.pi / 2 if quarter else 2 * np.pi, n_phi + 1)
    s = np.linspace(0.0, 1.0, n_theta + 1)
    if graded:
        s = 1.0 - np.cos(0.5 * np.pi * s)
    ths = np.pi / 2 + 0.5 * np.pi * s  # polar angle from +z

    def pt(t, p):
        return radius * np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])

    out = []
    for i in range(n_theta):
        for j in range(n_phi):
            a, b = pt(ths[i], phis[j]), pt(ths[i], phis[j + 1])
            c, d = pt(ths[i + 1], phis[j + 1]), pt(ths[i + 1], phis[j])
            quad = [a, b, c, d] if i + 1 < n_theta else [a, b, c, c]
            mid = np.mean(quad, axis=0)
            out.append(orient(quad, mid))  # outward = radial
    return np.array(out)


def _cosine(n):
    """n intervals on [0, 1], refined at both ends (none for n = 0)."""
    if n == 0:
        return np.zeros(1)
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))


def cylinder_panels(radius=1.0, draft=0.5, n_phi=8, n_z=2, n_r=2, quarter=True, cosine=False):
    """Floating vertical circular cylinder (polygonal), side and bottom;
    cosine=True refines the vertical and radial spacing towards the edges."""
    phis = np.linspace(0.0, np.pi / 2 if quarter else 2 * np.pi, n_phi + 1)
    zs = -draft * (_cosine(n_z) if cosine else np.linspace(0.0, 1.0, n_z + 1))
    rs = radius * (1.0 - (_cosine(n_r) if cosine else np.linspace(0.0, 1.0, n_r + 1)))
    out = []
    for i in range(n_z):
        for j in range(n_phi):
            p0, p1 = phis[j], phis[j + 1]
            q = [
                [radius * np.cos(p0), radius * np.sin(p0), zs[i]],
                [radius * np.cos(p1), radius * np.sin(p1), zs[i]],
                [radius * np.cos(p1), radius * np.sin(p1), zs[i + 1]],
                [radius * np.cos(p0), radius * np.sin(p0), zs[i + 1]],
            ]
            mid = np.mean(q, axis=0)
            out.append(orient(q, [mid[0], mid[1], 0.0]))
    for i in range(n_r):
        for j in range(n_phi):
            p0, p1 = phis[j], phis[j + 1]
            r0, r1 = rs[i], rs[i + 1]
            q = [
                [r0 * np.cos(p0), r0 * np.sin(p0), -draft],
                [r0 * np.cos(p1), r0 * np.sin(p1), -draft],
                [r1 * np.cos(p1), r1 * np.sin(p1), -draft],
                [r1 * np.cos(p0), r1 * np.sin(p0), -draft],
            ]
            if i + 1 == n_r:
                q = q[:3] + [q[2]]
            out.append(orient(q, [0.0, 0.0, -1.0]))
    return np.array(out)


def spheroid_panels(half_length=2.0, radius=0.25, n_x=8, n_t=8):
    """Quadrant (x >= 0, y >= 0) of a half-submerged prolate spheroid with its
    axis on the free surface; cosine spacing along the axis."""
    xs = half_length * np.cos(0.5 * np.pi * np.arange(n_x + 1) / n_x)[::-1]  # 0 .. L/2, fine at the end
    ts = np.linspace(0.0, 0.5 * np.pi, n_t + 1)  # 0: waterline (y axis), pi/2: bottom

    def pt(x, t):
        r = radius * np.sqrt(max(0.0, 1.0 - (x / half_length) ** 2))
        return np.array([x, r * np.cos(t), -r * np.sin(t)])

    out = []
    for i in range(n_x):
        for j in range(n_t):
            q = [pt(xs[i], ts[j]), pt(xs[i + 1], ts[j]), pt(xs[i + 1], ts[j + 1]), pt(xs[i], ts[j + 1])]
            if i + 1 == n_x:  # the end of the spheroid: triangle
                q = [q[0], q[1], q[3], q[3]]
            mid = np.mean(q, axis=0)
            out.append(orient(q, [mid[0] / half_length**2, mid[1] / radius**2, mid[2] / radius**2]))
    return np.array(out)


def flat_plate(a=1.0, b=1.0, z=0.0, n=(2, 2), up=True, center=(0.0, 0.0)):
    nx, ny = n
    return np.array(
        face_panels((center[0] - a / 2, center[1] - b / 2, z), (a, 0, 0), (0, b, 0), nx, ny, (0, 0, 1 if up else -1))
    )


def tlp_panels(radius=8.435, spacing=86.25, width=7.5, height=10.5, draft=35.0, n=(2, 10, 3, 2, 1, 4)):
    """Quadrant (x >= 0, y >= 0, symmetric about x = 0 and y = 0) of a four-column TLP in
    the layout of the ISSC TLP (Eatock Taylor & Jefferys 1986): a column of the given radius centred at (s, s),
    s = spacing/2, and the two half pontoons of width x height on the bottom
    that run from the column to the symmetry planes x = 0 and y = 0.

    n = (n_w, n_arc, n_zc, n_zp, n_r, n_len): azimuthal panels in each pontoon
    window of the column wall, azimuthal panels on the remaining wall, vertical
    panels of the column above and beside the pontoons, rings of the column
    bottom, and panels along each pontoon. Pontoon panels meet the column wall
    at its azimuthal nodes."""
    n_w, n_arc, n_zc, n_zp, n_r, n_len = n
    s, h = spacing / 2, width / 2
    alpha = np.arcsin(h / radius)
    z_top = -(draft - height)
    # azimuthal nodes: window around pi (towards x = 0), the short arc, the
    # window around 3 pi / 2 (towards y = 0), the long arc back to pi - alpha
    short = 0.5 * np.pi - 2 * alpha
    long_ = 1.5 * np.pi - 2 * alpha
    n_short = max(1, round(n_arc * short / (short + long_)))
    n_long = n_arc - n_short
    segs = [
        (np.pi - alpha, np.pi + alpha, n_w, True),
        (np.pi + alpha, 1.5 * np.pi - alpha, n_short, False),
        (1.5 * np.pi - alpha, 1.5 * np.pi + alpha, n_w, True),
        (1.5 * np.pi + alpha, 3 * np.pi - alpha, n_long, False),
    ]
    th, window = [], []
    for a, b, k, win in segs:
        for i in range(k):
            th.append((a + (b - a) * i / k, a + (b - a) * (i + 1) / k))
            window.append(win)

    def col(t, z):
        return [s + radius * np.cos(t), s + radius * np.sin(t), z]

    out = []
    zs_c = np.linspace(0.0, z_top, n_zc + 1)
    zs_p = np.linspace(z_top, -draft, n_zp + 1)
    for (t0, t1), win in zip(th, window, strict=True):
        tm = 0.5 * (t0 + t1)
        radial = [np.cos(tm), np.sin(tm), 0.0]
        for zs, skip in ((zs_c, False), (zs_p, win)):
            if skip:
                continue
            for i in range(len(zs) - 1):
                out.append(orient([col(t0, zs[i]), col(t1, zs[i]), col(t1, zs[i + 1]), col(t0, zs[i + 1])], radial))
        rs = radius * np.linspace(1.0, 0.0, n_r + 1)
        for i in range(n_r):
            r0, r1 = rs[i], rs[i + 1]
            q = [[s + r * np.cos(t), s + r * np.sin(t), -draft] for r, t in ((r0, t0), (r0, t1), (r1, t1), (r1, t0))]
            if i + 1 == n_r:
                q = q[:3] + [q[2]]
            out.append(orient(q, [0.0, 0.0, -1.0]))
    # pontoon towards y = 0 (window around 3 pi / 2); the other one by x <-> y
    nodes = [t0 for (t0, _), w in zip(th, window, strict=True) if w and np.cos(t0 + alpha) > -0.5]
    nodes = [t for t in nodes if abs(t - (1.5 * np.pi - alpha)) < 2 * alpha + 1e-12] + [1.5 * np.pi + alpha]
    xs = [s + radius * np.cos(t) for t in nodes]
    ye = [s + radius * np.sin(t) for t in nodes]
    fr = np.linspace(0.0, 1.0, n_len + 1)
    pont = []
    for i in range(len(xs) - 1):
        for k in range(n_len):
            for z, nz in ((z_top, 1.0), (-draft, -1.0)):
                q = [[xs[i], fr[k] * ye[i], z], [xs[i + 1], fr[k] * ye[i + 1], z],
                     [xs[i + 1], fr[k + 1] * ye[i + 1], z], [xs[i], fr[k + 1] * ye[i], z]]  # fmt: skip
                pont.append(orient(q, [0.0, 0.0, nz]))
    for x, y_end, nx in ((s - h, ye[0], -1.0), (s + h, ye[-1], 1.0)):
        for k in range(n_len):
            for i in range(n_zp):
                y0, y1 = fr[k] * y_end, fr[k + 1] * y_end
                q = [[x, y0, zs_p[i]], [x, y1, zs_p[i]], [x, y1, zs_p[i + 1]], [x, y0, zs_p[i + 1]]]
                pont.append(orient(q, [nx, 0.0, 0.0]))
    out += pont
    out += [np.asarray(q)[[0, 3, 2, 1]][:, [1, 0, 2]] for q in pont]  # mirror x <-> y keeps the orientation
    return np.array(out, dtype=float)


def spline_cylinder_patches(radius=1.0, draft=0.5, spans=4, order=4):
    """Quadrant (x >= 0, y >= 0) of a floating vertical cylinder as two
    B-spline patches (IGDEF = 1): the side (u: azimuth 0..pi/2, v: depth) and
    the bottom (u: azimuth, v: from the rim to the axis). The quarter
    circle is the B-spline of the given order on `spans` uniform spans that
    interpolates the circle at the Greville abscissae; linear in the other
    direction. Normals into the fluid follow the right-hand rule u -> v into
    the body. Returns a list of (u_order, v_order, u_knots, v_knots, coefficients
    (NV, NU, 3))."""
    from scipy.interpolate import BSpline

    k = order - 1
    t = np.r_[[0.0] * order, np.linspace(0.0, 1.0, spans + 1)[1:-1], [1.0] * order]
    n = len(t) - order
    greville = np.array([t[i + 1 : i + order].mean() for i in range(n)])
    Bm = BSpline.design_matrix(greville, t, k).toarray()
    theta = 0.5 * np.pi * greville
    cx = np.linalg.solve(Bm, np.cos(theta))
    cy = np.linalg.solve(Bm, np.sin(theta))
    tv = (0.0, 0.0, 1.0, 1.0)
    # side: v from the waterline down (u x v must point into the body)
    side = np.zeros((2, n, 3))
    for j, z in enumerate((0.0, -draft)):
        side[j, :, 0], side[j, :, 1], side[j, :, 2] = radius * cx, radius * cy, z
    # bottom: v from the rim to the axis
    bottom = np.zeros((2, n, 3))
    for j, r in enumerate((radius, 0.0)):
        bottom[j, :, 0], bottom[j, :, 1], bottom[j, :, 2] = r * cx, r * cy, -draft
    out = []
    for c in (side, bottom):
        out.append((order, 2, tuple(t), tv, c))
    return out


# the rational quarter circle of order 3 (Piegl & Tiller 1997, §7.3): control points on the unit
# square's corners, the middle weight cos(45 deg)
QUARTER_CIRCLE = ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
QUARTER_WEIGHTS = (1.0, math.sqrt(0.5), 1.0)


def nurbs_cylinder_patches(radius=1.0, draft=0.5):
    """Quadrant of a floating vertical cylinder as two rational patches of order 3 in u (the exact
    quarter circle) and 2 in v: the side (v from the waterline down) and the bottom (v from the rim
    to the axis). Returns a list of (u_order, v_order, u_knots, v_knots, coefficients (NV, NU, 3),
    weights (NV, NU))."""
    tu, tv = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    w = np.array([QUARTER_WEIGHTS] * 2)
    side = np.array([[(radius * a, radius * b, z) for a, b in QUARTER_CIRCLE] for z in (0.0, -draft)])
    bottom = np.array([[(r * a, r * b, -draft) for a, b in QUARTER_CIRCLE] for r in (radius, 0.0)])
    return [(3, 2, tu, tv, side, w), (3, 2, tu, tv, bottom, w)]


def nurbs_hemisphere_patches(radius=1.0):
    """Quadrant of a floating hemisphere as one rational patch of order 3 in u and v, the surface
    of revolution of the quarter-circle meridian (v from the waterline to the pole, which is a
    collapsed edge): the control points the products of the two quarter circles, the weights the
    products of their weights."""
    t = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    meridian = ((1.0, 0.0), (1.0, -1.0), (0.0, -1.0))  # (rho, z) / radius
    c = np.array([[(radius * r * a, radius * r * b, radius * z) for a, b in QUARTER_CIRCLE] for r, z in meridian])
    w = np.outer(QUARTER_WEIGHTS, QUARTER_WEIGHTS)
    return [(3, 3, t, t, c, w)]


def moonpool_panels(outer=2.0, inner=1.0, draft=1.0, n=8):
    """Quadrant of a floating annular cylinder (moonpool of radius `inner`),
    vertices counter-clockwise seen from the fluid."""
    out = []
    ph = np.linspace(0.0, np.pi / 2, n + 1)
    zs = np.linspace(0.0, -draft, n // 2 + 1)
    rs = np.linspace(inner, outer, n // 2 + 1)

    def p(r, k, z):
        return [r * np.cos(ph[k]), r * np.sin(ph[k]), z]

    for i in range(n):
        for j in range(n // 2):
            z0, z1, r0, r1 = zs[j], zs[j + 1], rs[j], rs[j + 1]
            out.append([p(outer, i, z0), p(outer, i, z1), p(outer, i + 1, z1), p(outer, i + 1, z0)])  # outward
            out.append([p(inner, i, z0), p(inner, i + 1, z0), p(inner, i + 1, z1), p(inner, i, z1)])  # to the axis
            out.append([p(r0, i, -draft), p(r0, i + 1, -draft), p(r1, i + 1, -draft), p(r1, i, -draft)])  # down
    return np.array(out)


def mirror(v, axis):
    w = np.array(v, dtype=float)
    w[..., axis] *= -1.0
    return w[:, ::-1]  # orientation kept (normals into the fluid)


def moonpool_lid_panels(r=1.0, n=8):
    """Quadrant of the moonpool free surface r < `r` at z = 0, upward normals."""
    ph = np.linspace(0.0, np.pi / 2, n + 1)
    rs = np.linspace(0.0, r, n // 2 + 1)

    def p(rr, k):
        return [rr * np.cos(ph[k]), rr * np.sin(ph[k]), 0.0]

    return np.array(
        [[p(rs[j], i), p(rs[j + 1], i), p(rs[j + 1], i + 1), p(rs[j], i + 1)] for i in range(n) for j in range(n // 2)]
    )


def disc_panels(radius=1.0, n_r=8, n_phi=8, quarter=True, graded=False, inner=0.0, z=0.0):
    """A horizontal disc (an annulus with inner > 0) of `radius` at z, downward normals (into the
    water below), n_r rings and n_phi divisions in azimuth over the quadrant (quarter) or the
    circle; graded: the rings refined towards the outer edge (r = R sin of uniform angles), the
    edge of a plate bordering the free surface."""
    ph = np.linspace(0.0, np.pi / 2 if quarter else 2 * np.pi, n_phi + 1)
    s = np.linspace(0.0, 1.0, n_r + 1)
    rs = inner + (radius - inner) * (np.sin(0.5 * np.pi * s) if graded else s)

    def p(rr, k):
        return [rr * np.cos(ph[k]), rr * np.sin(ph[k]), z]

    return np.array(
        [[p(rs[j], i), p(rs[j], i + 1), p(rs[j + 1], i + 1), p(rs[j + 1], i)] for i in range(n_phi) for j in range(n_r)]
    )


def acv_panels(n=2, quarter=True, zc=-1.0):
    """Air-cushion vessel 20 x 10 x 2 m, skirts 1 m, cushion surface at zc
    (also as a full mesh)."""
    L, B, T, w = 10.0, 5.0, 2.0, 1.0
    pan = []
    pan += face_panels((L, 0, -T), (0, B, 0), (0, 0, T), 5 * n, 2 * n, (1, 0, 0))
    pan += face_panels((0, B, -T), (L, 0, 0), (0, 0, T), 10 * n, 2 * n, (0, 1, 0))
    pan += face_panels((L - w, 0, -T), (w, 0, 0), (0, B, 0), n, 5 * n, (0, 0, -1))
    pan += face_panels((0, B - w, -T), (L - w, 0, 0), (0, w, 0), 9 * n, n, (0, 0, -1))
    pan += face_panels((L - w, 0, -T), (0, B - w, 0), (0, 0, T + zc), 4 * n, n, (-1, 0, 0))
    pan += face_panels((0, B - w, -T), (L - w, 0, 0), (0, 0, T + zc), 9 * n, n, (0, -1, 0))
    ps = face_panels((0, 0, zc), (L - w, 0, 0), (0, B - w, 0), 9 * n, 4 * n, (0, 0, -1))
    hull, ps = np.array(pan), np.array(ps)
    if not quarter:
        def full(q):
            out = [q]
            for sx, sy in ((-1, 1), (1, -1), (-1, -1)):
                r = q * [sx, sy, 1]
                out.append(r[:, [0, 3, 2, 1]] if sx * sy < 0 else r)
            return np.concatenate(out)
        hull, ps = full(hull), full(ps)
    return hull, ps


MOONPOOL_FACES = [  # quadrant of a square barge 4 x 4 x 1 m with a 2 x 2 m moonpool: origin, du, dv, normal
    ((2, 0, -1), (0, 2, 0), (0, 0, 1), (1, 0, 0)), ((0, 2, -1), (2, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((1, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, -1)), ((0, 1, -1), (2, 0, 0), (0, 1, 0), (0, 0, -1)),
    ((1, 0, -1), (0, 1, 0), (0, 0, 1), (-1, 0, 0)), ((0, 1, -1), (1, 0, 0), (0, 0, 1), (0, -1, 0)),
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),  # the moonpool free surface (the lid)
]


PLATE_FACES = [  # quadrant of a square column 2 x 2 x 1 m with a heave plate 1 <= |x|, |y| <= 1.6 m at z = -1
    ((1, 0, -1), (0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, 1, -1), (1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 0, -1), (0, 1, 0), (1, 0, 0), (0, 0, -1)),
    ((1, 0, -1), (0.6, 0, 0), (0, 1.6, 0), (0, 0, 1)), ((0, 1, -1), (1, 0, 0), (0, 0.6, 0), (0, 0, 1)),
]


PLATE_NUV = [(4, 4), (4, 4), (4, 4), (2, 6), (4, 2)]


# the air-cushion vessel of acv_panels (quadrant, 20 x 10 x 2 m, skirts 1 m,
# cushion at z = -1) as seven flat patches, NU, NV by the rule of panel_size
ACV_FACES = [((10, 0, -2), (0, 5, 0), (0, 0, 2), (1, 0, 0)), ((0, 5, -2), (10, 0, 0), (0, 0, 2), (0, 1, 0)),
             ((9, 0, -2), (1, 0, 0), (0, 5, 0), (0, 0, -1)), ((0, 4, -2), (9, 0, 0), (0, 1, 0), (0, 0, -1)),
             ((9, 0, -2), (0, 4, 0), (0, 0, 1), (-1, 0, 0)), ((0, 4, -2), (9, 0, 0), (0, 0, 1), (0, -1, 0)),
             ((0, 0, -1), (9, 0, 0), (0, 4, 0), (0, 0, -1))]  # fmt: skip
