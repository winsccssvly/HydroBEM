"""Test meshes and bodies shared by the test files and the regression problems, as dicts of the
items of the engine's objects (``problems.model`` puts them into an ``hb.Model``; a mesh dict,
alone or in the test shorthand ``(mesh, ranges)`` for the items mesh and ranges, is added to the
model once); vertex generators are in meshes.py."""

import functools
import math

import numpy as np

from meshes import (
    ACV_FACES,
    MOONPOOL_FACES,
    PLATE_FACES,
    PLATE_NUV,
    acv_panels,
    box_panels,
    cylinder_panels,
    face_panels,
    hemisphere_panels,
    moonpool_lid_panels,
    moonpool_panels,
    spline_cylinder_patches,
)


def surface(vertices, symmetry="none", **items):
    """A low-order mesh: four vertices per panel (normals into the fluid)."""
    return {"kind": "panels", "vertices": np.asarray(vertices, dtype=np.float64), "symmetry": symmetry, **items}


def discretization(rows, count):
    """Per patch (nu, nv, ku, kv, iquo, iqui): one row for all ``count`` patches, or one per patch."""
    a = np.asarray(rows, dtype=np.int64)
    return np.tile(a, (count, 1)) if a.ndim == 1 else a.reshape(-1, 6)


def flat(vertices, rows, symmetry="none", **items):
    """A higher-order mesh of flat patches (IGDEF = 0)."""
    v = np.asarray(vertices, dtype=np.float64)
    return {"kind": "flat_patches", "vertices": v, "discretization": discretization(rows, len(v)), "symmetry": symmetry,
            **items}  # fmt: skip


def spline(patches, rows, symmetry="none", **items):
    """A higher-order mesh of B-spline patches (IGDEF = 1): ``patches`` as (u_order, v_order,
    u_knots, v_knots, coefficients), or with the weights of a rational patch as a sixth entry."""
    p = [{"type": "SplinePatch", "u_order": q[0], "v_order": q[1], "u_knots": np.asarray(q[2], dtype=np.float64),
          "v_knots": np.asarray(q[3], dtype=np.float64), "coefficients": np.asarray(q[4], dtype=np.float64),
          **({"weights": np.asarray(q[5], dtype=np.float64)} if len(q) > 5 else {})}
         for q in patches]  # fmt: skip
    return {"kind": "spline_patches", "discretization": discretization(rows, len(p)), "symmetry": symmetry,
            "patches": p, **items}  # fmt: skip


def part(mesh, first, last):
    """The stored elements first..last of a mesh (one-based, inclusive): the test shorthand
    ``(mesh, ranges)`` of the items mesh and ranges (problems._Meshes)."""
    return (mesh, (first, last))


def hemi(n=8, quarter=True):
    """The floating hemisphere of radius 1: n x n panels a quadrant (a quarter with its symmetry),
    or the whole of n x 4 n panels."""
    v = hemisphere_panels(1.0, n, n if quarter else 4 * n, quarter=quarter)
    return surface(v, symmetry="xz_and_yz" if quarter else "none")


def coarse_cylinder():
    """The quarter of the cylinder of radius 1 and draft 0.5, 8 x 2 x 2 panels, with its symmetry:
    a mesh of the API tests (a solve of milliseconds)."""
    return surface(cylinder_panels(1.0, 0.5, n_phi=8, n_z=2, n_r=2, quarter=True), symmetry="xz_and_yz")


def slow_cylinder():
    """The cylinder of radius 1 and draft 1 in 2048 panels without symmetry: a solve of seconds
    (the tests of progress, cancelling and the threads of a solve that runs meanwhile)."""
    return surface(cylinder_panels(1.0, 1.0, n_phi=64, n_z=16, n_r=16, quarter=False))


def graded_cylinder(n=8, quarter=True, draft=1.0):
    v = cylinder_panels(1.0, draft, n_phi=n if quarter else 4 * n, n_z=n, n_r=n // 2, quarter=quarter, cosine=True)
    return surface(v, symmetry="xz_and_yz" if quarter else "none")


def acv_body(modes=("uniform",), quarter=True, chamber=True, convention="variational", n=2, **kw):
    hull, ps = acv_panels(n=n, quarter=quarter)
    m = surface(np.concatenate([hull, ps]), symmetry="xz_and_yz" if quarter else "none")
    surf = {"mesh": part(m, len(hull) + 1, len(hull) + len(ps)),
            "pressure_modes": [{"name": f"p{k}", "shape": s} for k, s in enumerate(modes)],
            "chamber_coupling": chamber, "response_convention": convention}  # fmt: skip
    return {"name": "acv", "mesh": part(m, 1, len(hull)), "pressure_surfaces": [surf], **kw}


def spline_cylinder(n=4, k=3):
    return spline(spline_cylinder_patches(1.0, 0.5, spans=4, order=4), (n, max(2, n // 2), k, k, k, k + 1),
                  symmetry="xz_and_yz")  # fmt: skip


def full_cylinder():
    """The spline cylinder as four quadrant copies (no symmetry flags)."""
    patches = []
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        for ku, kv, tu, tv, c in spline_cylinder_patches(1.0, 0.5, spans=4, order=4):
            c = np.array(c) * [sx, sy, 1]
            if sx * sy < 0:
                c = c[:, ::-1]  # reverse u: the normal keeps pointing into the fluid
                tu = tuple(-t for t in tu[::-1])
            patches.append((ku, kv, tuple(tu), tuple(tv), c))
    return spline(patches, (4, 2, 3, 3, 3, 4))


def submerged_plate(z, a, b, n=(12, 7), t=None):
    """The quadrant of a horizontal plate 0 <= x <= a, 0 <= y <= b at depth z: a sheet (normal up)
    or, with a thickness t, the closed slab about it (low order, n panels along x and y)."""
    if t is None:
        return surface(face_panels((0, 0, z), (a, 0, 0), (0, b, 0), *n, (0, 0, 1)), symmetry="xz_and_yz")
    p = face_panels((0, 0, z + t / 2), (a, 0, 0), (0, b, 0), *n, (0, 0, 1))
    p += face_panels((0, 0, z - t / 2), (a, 0, 0), (0, b, 0), *n, (0, 0, -1))
    p += face_panels((a, 0, z - t / 2), (0, b, 0), (0, 0, t), n[1], 1, (1, 0, 0))
    p += face_panels((0, b, z - t / 2), (a, 0, 0), (0, 0, t), n[0], 1, (0, 1, 0))
    return surface(p, symmetry="xz_and_yz")


def ho_submerged_plate(z, a, b, n=(6, 4), t=None):
    """submerged_plate as flat patches (n panels per patch along x and y)."""
    if t is None:
        q = np.array([face_panels((0, 0, z), (a, 0, 0), (0, b, 0), 1, 1, (0, 0, 1))[0]])
        return flat(q, [(*n, 3, 3, 3, 4)], symmetry="xz_and_yz")
    faces = [((0, 0, z + t / 2), (a, 0, 0), (0, b, 0), (0, 0, 1)),
             ((0, 0, z - t / 2), (a, 0, 0), (0, b, 0), (0, 0, -1)),
             ((a, 0, z - t / 2), (0, b, 0), (0, 0, t), (1, 0, 0)),
             ((0, b, z - t / 2), (a, 0, 0), (0, 0, t), (0, 1, 0))]
    q = np.array([face_panels(o, du, dv, 1, 1, nn)[0] for o, du, dv, nn in faces])
    return flat(q, [(*n, 3, 3, 3, 4)] * 2 + [(n[1], 1, 3, 3, 3, 4), (n[0], 1, 3, 3, 3, 4)], symmetry="xz_and_yz")


def acv_with(zc=-1.0, n=2, modes=("uniform", "sign_x")):
    """The air-cushion vessel of acv_body with its cushion at zc (z = 0 or submerged)."""
    hull, ps = acv_panels(n=n, zc=zc)
    m = surface(np.concatenate([hull, ps]), symmetry="xz_and_yz")
    surf = {"mesh": part(m, len(hull) + 1, len(hull) + len(ps)),
            "pressure_modes": [{"name": f"p{k}", "shape": s} for k, s in enumerate(modes)]}  # fmt: skip
    return {"name": "acv", "mesh": part(m, 1, len(hull)), "pressure_surfaces": [surf]}


def moonpool_barge():
    """The square barge 4 x 4 x 1 m with a 2 x 2 m moonpool as flat patches (quadrant): elements 1-6
    its hull, element 7 the moonpool's free surface."""
    q = np.array([face_panels(o, du, dv, 1, 1, n)[0] for o, du, dv, n in MOONPOOL_FACES])
    nuv = [(max(1, round(np.linalg.norm(x[1] - x[0]) * 2)), max(1, round(np.linalg.norm(x[3] - x[0]) * 2))) for x in q]
    return flat(q, [(a, b, 3, 3, 3, 4) for a, b in nuv], symmetry="xz_and_yz")


def panel_size_nuv(q, size):
    """NU, NV of a flat patch (u from vertex 1 to 2, v from 2 to 3) by the rule of panel_size."""
    lu = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    lv = max(np.linalg.norm(q[2] - q[1]), np.linalg.norm(q[3] - q[0]))
    return int(lu // size) + 1, int(lv // size) + 1


def ho_acv(size=1.0, modes=("sign_x", "uniform"), convention="variational", chamber=True):
    v = np.array([face_panels(o, du, dv, 1, 1, n)[0] for o, du, dv, n in ACV_FACES])
    m = flat(v, [(*panel_size_nuv(q, size), 3, 3, 3, 4) for q in v], symmetry="xz_and_yz")
    surf = {"mesh": part(m, 7, 7), "pressure_modes": [{"name": f"p{k}", "shape": x} for k, x in enumerate(modes)],
            "chamber_coupling": chamber, "response_convention": convention}  # fmt: skip
    return {"name": "acv", "mesh": part(m, 1, 6), "pressure_surfaces": [surf]}


def plate_column(quarter=True, flip=False, f=1):
    q = np.array([face_panels(o, du, dv, 1, 1, n)[0] for o, du, dv, n in PLATE_FACES])
    if flip:  # the sheets with the opposite vertex order (normal)
        q[3:] = q[3:, ::-1]
    nuv = [(a * f, b * f) for a, b in PLATE_NUV]  # the reversed order keeps u along the same edge
    if not quarter:
        def images(p):
            return [p, (p * [-1, 1, 1])[:, ::-1], (p * [1, -1, 1])[:, ::-1], p * [-1, -1, 1]]
        hull, sheet = np.concatenate(images(q[:3])), np.concatenate(images(q[3:]))
        q = np.concatenate([hull, sheet])
        nuv = (nuv[:3] * 4) + (nuv[3:] * 4)
    m = flat(q, [(a, b, 3, 3, 3, 4) for a, b in nuv], symmetry="xz_and_yz" if quarter else "none")
    ns = 2 if quarter else 8
    sheets = [{"mesh": part(m, len(q) - ns + 1, len(q))}]
    return {"name": "body", "mesh": part(m, 1, len(q) - ns), "dipoles": sheets}


W_ACV = 2 * np.pi / np.array([4.0, 6.0, 9.0])


def _free(mesh, **kw):
    m = 1000.0 * math.pi * 0.5
    return {"name": "body", "mesh": mesh, "center_of_gravity": [0.0, 0.0, -0.2], "mass": m,
            "inertia_input": "components",
            "Ixx": m * 0.36,
            "Ixy": 0.0, "Ixz": 0.0, "Iyy": m * 0.36, "Iyz": 0.0, "Izz": m * 0.49,
            "inertia_origin": "center_of_gravity", **kw}  # fmt: skip


def free_cylinder(n=4, **kw):
    return _free(spline_cylinder(n=n), **kw)


def lo_cosine_cylinder(fixed=False, **kw):
    """The cylinder of radius 1 and draft 0.5 of the potential QTF tests (8 x 4 x 4 cosine-spaced
    panels a quadrant, with its symmetry): floating (``_free``: its mass and inertia), or fixed (no
    modes)."""
    pan = cylinder_panels(1.0, 0.5, n_phi=8, n_z=4, n_r=4, quarter=True, cosine=True)
    mesh = surface(pan, symmetry="xz_and_yz")
    return {"mesh": mesh, "modes": [], **kw} if fixed else _free(mesh, **kw)


def lo_free_cylinder(quarter=True, **kw):
    pan = cylinder_panels(1.0, 0.5, n_phi=8 if quarter else 32, n_z=4, n_r=4, quarter=quarter)  # n_phi per part
    return _free(surface(pan, symmetry="xz_and_yz" if quarter else "none"), **kw)


# two flexible modes of the cylinder of radius 1 and draft 0.5 (theory §10.1): the bottom bulging,
# w = 1 - x^2 - y^2 (zero on the side, r = 1), and the side bending, u = 4 z^2 (zero at the
# waterline); continuous, with a displacement gradient that tilts and stretches the panels
CYLINDER_BULGE = {"name": "bulge", "pieces": [{"w": [[1.0, 0, 0, 0], [-1.0, 2, 0, 0], [-1.0, 0, 2, 0]]}],
                  "parity": "heave"}  # fmt: skip
CYLINDER_BEND = {"name": "bend", "pieces": [{"u": [[4.0, 0, 0, 2]]}], "parity": "surge"}


def heave_pieces(ref):
    """One generalized mode "g" of heave parity with two pieces: w = x^2 on the elements of ``ref``
    (``(mesh, ranges)``: its ranges), w = 0.3 on the rest (the tests of the elements a piece
    selects)."""
    pieces = [{"w": [[1.0, 2, 0, 0]], "mesh": ref}, {"w": [[0.3, 0, 0, 0]]}]
    return [{"name": "g", "parity": "heave", "pieces": pieces}]


def lo_flexible_cylinder(n=8, flexible=True, **kw):
    """The free cylinder of radius 1 and draft 0.5 (n panels a quadrant around, n/2 down and across
    the bottom, cosine spacing) with the flexible modes CYLINDER_BULGE and CYLINDER_BEND (not with
    ``flexible`` false): their mass 200 kg, stiffness 3000 N/m and damping 500 kg/s each; the rigid
    heave, roll and pitch damped (3000, 500, 500) so that no resonance dominates the drift between
    1.8 and 4.2 rad/s."""
    pan = cylinder_panels(1.0, 0.5, n_phi=n, n_z=n // 2, n_r=n // 2, quarter=True, cosine=True)
    damped = {"additional_damping": np.diag([0.0, 0.0, 3000.0, 500.0, 500.0, 0.0]).tolist()}
    if not flexible:
        return _free(surface(pan, symmetry="xz_and_yz"), **damped, **kw)
    d = np.diag([0.0] * 6 + [1.0, 1.0])
    return _free(surface(pan, symmetry="xz_and_yz"), generalized_modes=[CYLINDER_BULGE, CYLINDER_BEND],
                 mode_mass=200.0 * d,
                 mode_stiffness=3000.0 * d, mode_damping=500.0 * d, **damped, **kw)  # fmt: skip


def lo_free_flared_cylinder(flare, n_phi=16, n_z=8, n_r=8):
    """The free cylinder of radius 1 and draft 0.5 with its side flared: the radius grows linearly
    from 1 at the bottom to 1 + flare at the waterline (not wall-sided: the normal at the waterline
    has a vertical component); its mass the displaced one."""
    p = cylinder_panels(1.0, 0.5, n_phi=n_phi, n_z=n_z, n_r=n_r, quarter=True, cosine=True).copy()
    s = 1.0 + flare * (p[..., 2] + 0.5) / 0.5
    p[..., 0] *= s
    p[..., 1] *= s
    return _free(surface(p, symmetry="xz_and_yz"), mass="displaced")


@functools.lru_cache(maxsize=None)
def moonpool_mesh():
    """The quarter annular cylinder of the lid tests (moonpool radius 1, 12 panels per quadrant)."""
    return surface(moonpool_panels(n=12), symmetry="xz_and_yz")


@functools.lru_cache(maxsize=None)
def moonpool_lid_mesh():
    """The moonpool free surface of moonpool_mesh() (upward normals)."""
    return surface(moonpool_lid_panels(n=12), symmetry="xz_and_yz")


TANK_L, TANK_B, TANK_H, TANK_ZT = 1.6, 0.8, 0.5, -0.1  # tank length, breadth, fluid depth, free surface


def tank_vessel(density=1.0, tank=True, n=(8, 4, 3)):
    """A box barge 2 x 1 x 1 with a central internal tank (quadrant), its mass displaced."""
    hull = box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True)
    # tank walls: the quarter box x <= L/2, y <= B/2, ZT - H <= z <= ZT, normals into the tank fluid
    walls = box_panels(TANK_L / 2, TANK_B / 2, TANK_H, n=n, quarter=True)[:, ::-1] + [0.0, 0.0, TANK_ZT]
    mesh = surface(np.concatenate([hull, walls]), symmetry="xz_and_yz")
    if not tank:
        return {"mesh": mesh, "mass": "displaced"}
    tanks = [{"mesh": part(mesh, len(hull) + 1, len(hull) + len(walls)), "density_ratio": density}]
    return {"mesh": part(mesh, 1, len(hull)), "mass": "displaced", "tanks": tanks}


def named_tank_vessel(name, x, **kw):
    """tank_vessel() named ``name`` at (x, 0, 0), its tank named "t", with a centre of gravity and
    radii of gyration; ``kw`` adds or replaces items."""
    b = tank_vessel()
    return {**b, "name": name, "position": [x, 0.0, 0.0], "center_of_gravity": [0.0, 0.0, -0.2],
            "radii_of_gyration": [0.4, 0.5, 0.5], "tanks": [{**b["tanks"][0], "name": "t"}], **kw}  # fmt: skip


# points in the tank of a named_tank_vessel at x = 0, in that of one at x = 6, and in the sea
TANK_POINTS = np.array([[0.0, 0.0, TANK_ZT - 0.2], [6.0, 0.0, TANK_ZT - 0.2], [3.0, 2.0, -0.3]])



def ho_sphere(radius=1.0, n=(4, 8)):
    """The exact sphere about the body origin as one analytic patch a quadrant (map "arc": the
    meridian revolved from the top pole to the bottom one, the polar angle linear in v), n spans
    around the quadrant and down the meridian, B-splines of order 3, Gauss orders 3 and 4; for a
    submerged sphere placed by the body's position."""
    patch = {"type": "AnalyticPatch", "map": "arc", "profile": "linear", "a": radius, "b": radius,
             "theta0": 0.0, "theta1": 0.5 * math.pi, "p": [radius, -0.5 * math.pi, 0.5 * math.pi, 0.0]}  # fmt: skip
    return {"kind": "analytic_patches", "discretization": np.array([[*n, 3, 3, 3, 4]]), "symmetry": "xz_and_yz",
            "patches": [patch]}  # fmt: skip
