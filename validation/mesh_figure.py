"""Figures of the meshes of the validation cases (docs/validation/figures/mesh_<case>.png): the
surfaces of a model as the engine uses them, through the public queries Model.panels (the elements
with their roles, with their reflections about the symmetry planes) and Model.tessellation (the
same elements, the patches of the higher order as quadrilaterals for display, each quad with the
keys of its element), coloured by role.

    draw("floating_hemisphere", [("low order", model_lo, False), ("higher order", model_ho, True)])

The low-order panels are drawn with their edges. A higher-order patch is drawn as its tessellation
(8 x 8 quads a patch, shaded) with its four sides (dark) and the edges of the spans of its potential
(grey: the tessellation with one quad per span, divisions="spans"); the lines on the far side of a
wetted surface are left out. The roles (reference/queries): the wetted surface, the interior free
surface, damping lids, free-surface plates, pressure surfaces, thin sheets (dipoles and dipole
dampers), tank walls, fixed dampers and the free surface of the potential QTF. For the validation
report only: matplotlib is not a dependency of the package.
"""

from __future__ import annotations

import math

import numpy as np
from common import plt, save
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# the role of Panels.role: (legend name, face colour)
ROLES = {
    "wetted": ("wetted surface", "#9db4d3"),
    "interior_surface": ("interior free surface", "#f2c14e"),
    "lid": ("damping lid", "#7bc47f"),
    "free_surface_plate": ("free-surface plate", "#e07a5f"),
    "pressure_surface": ("pressure surface", "#b48ead"),
    "dipole": ("thin sheet (dipoles)", "#d1495b"),
    "dipole_damper": ("dipole damper", "#8f2d56"),
    "tank": ("tank wall", "#5fb3b3"),
    "fixed_damper": ("fixed damper", "#8a8a8a"),
    "free_surface": ("free surface of the potential QTF", "#d9d9d9"),
}
OTHER = ("other", "#cccccc")


def _role(r):
    return ROLES.get(str(r), OTHER)


def _normal(v):
    """The unit normal of a polygon (the orientation of its vertices)."""
    n = np.cross(v[2] - v[0], v[-1] - v[1]) if len(v) > 3 else np.cross(v[1] - v[0], v[2] - v[0])
    s = np.linalg.norm(n)
    return n / s if s > 0 else np.zeros(3)


def collect(model, higher_order=False, free_surface=False, nu=8):
    """The faces [(role, vertices)] of a model and its lines [(role, polyline, normal, width)]: the
    patch sides and the span edges of the higher order."""
    faces, lines = [], []
    if not higher_order:
        p = model.panels(include_reflections=True)
        for v, n, role in zip(p.vertices, p.distinct_vertices, p.role, strict=True):
            faces.append((str(role), v[: max(3, int(n))]))
    else:
        # every element of the panels as nu x nu quads, each with the role of its element
        t = model.tessellation(divisions=nu, include_reflections=True)
        roles = [str(r) for r in t.role]
        faces = list(zip(roles, t.vertices, strict=True))
        # the spans of the potential: one quad per span
        s = model.tessellation(divisions="spans", include_reflections=True)
        spans = zip(s.vertices, s.normals, s.role, strict=True)
        lines = _patch_sides(t, roles, nu) + [(str(r), np.r_[v, v[:1]], n.mean(axis=0), 0.35) for v, n, r in spans]
    if free_surface:
        f = model.panels(role="free_surface", include_reflections=True)
        for v, n in zip(f.vertices, f.distinct_vertices, strict=True):
            faces.append(("free_surface", v[: max(3, int(n))]))
    return faces, lines


def _patch_sides(t, roles, nu):
    """The sides of each tessellated patch (its nu x nu consecutive quads): the quad edges that
    belong to one quad of the patch only (a collapsed side, at a pole, has no length)."""
    lines = []
    v, nrm = t.vertices, t.normals.mean(axis=1)
    for start in range(0, len(v) - nu * nu + 1, nu * nu):
        count, seg = {}, {}
        for k in range(start, start + nu * nu):
            for a in range(4):
                p, q = v[k][a], v[k][(a + 1) % 4]
                if np.linalg.norm(p - q) < 1e-12:
                    continue
                key = tuple(sorted((tuple(np.round(p, 9)), tuple(np.round(q, 9)))))
                count[key] = count.get(key, 0) + 1
                seg[key] = (np.array([p, q]), nrm[k])
        lines += [(roles[start], *seg[key], 0.9) for key, c in count.items() if c == 1]
    return lines


def _equal(ax, pts):
    """Equal scales in x, y and z, the box fitted to the points."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-3 * (hi - lo).max())
    ax.set_xlim(lo[0], lo[0] + span[0])
    ax.set_ylim(lo[1], lo[1] + span[1])
    ax.set_zlim(lo[2], lo[2] + span[2])
    ax.set_box_aspect(tuple(span), zoom=1.1)


def _shaded(polys, colours):
    """The face colour of each polygon lit from above and in front (the sign of the normal ignored)."""
    light = np.array([-0.4, -0.6, 0.7]) / np.linalg.norm([-0.4, -0.6, 0.7])
    return np.array([np.r_[np.array(to_rgb(c)) * (0.6 + 0.4 * abs(_normal(v) @ light)), 1.0]
                     for v, c in zip(polys, colours, strict=True)])  # fmt: skip


def _kept(cut, pts):
    sx, sy = cut
    return not (sx * pts[:, 0].mean() > 1e-9 and sy * pts[:, 1].mean() > 1e-9)


def panel(ax, faces, lines, title, elev, azim, cut=None):
    """Draws the faces (one collection, sorted in depth) and the lines on a 3D axis; ``cut`` drops
    the faces whose centroid lies in the quadrant (sign x, sign y) given, so that the inside shows.
    The lines of a wetted surface whose normal (out of the body) faces away from the eye are left
    out."""
    if cut is not None:
        faces = [(r, v) for r, v in faces if _kept(cut, v)]
        lines = [ln for ln in lines if _kept(cut, ln[1])]
    el, az = math.radians(elev), math.radians(azim)
    eye = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    lines = [ln for ln in lines if ln[0] != "wetted" or ln[2] @ eye > -0.05 * np.linalg.norm(ln[2])]
    polys = [v for _, v in faces]
    lw = 0.5 if len(polys) < 600 else 0.3 if len(polys) < 3000 else 0.15
    ax.computed_zorder = False
    coll = Poly3DCollection(polys, facecolors=_shaded(polys, [_role(r)[1] for r, _ in faces]),
                            edgecolors="none" if lines else "#333333", linewidths=lw, zorder=1)  # fmt: skip
    ax.add_collection3d(coll)
    for width in sorted({ln[3] for ln in lines}):
        segs = [ln[1] for ln in lines if ln[3] == width]
        ax.add_collection3d(Line3DCollection(segs, colors="#666666" if width < 0.5 else "#111111", linewidths=width,
                                             zorder=2 if width < 0.5 else 3))  # fmt: skip
    _equal(ax, np.concatenate(polys))
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    return [r for r, _ in faces]


def draw(name, views, elev=25.0, azim=-60.0, cut=None, size=(4.2, 3.8)):
    """Writes docs/validation/figures/mesh_<name>.png: one panel per (title, model, higher_order[,
    options]) of ``views`` (options: free_surface, cut, elev, azim), one legend of the roles."""
    fig = plt.figure(figsize=(size[0] * len(views), size[1]))
    roles = []
    for i, view in enumerate(views):
        title, model, higher = view[:3]
        opts = view[3] if len(view) > 3 else {}
        faces, lines = collect(model, higher, opts.get("free_surface", False))
        ax = fig.add_subplot(1, len(views), i + 1, projection="3d")
        e, a = opts.get("elev", elev), opts.get("azim", azim)
        ax.view_init(elev=e, azim=a)
        roles += panel(ax, faces, lines, title, e, a, opts.get("cut", cut))
    handles = [Patch(facecolor=_role(r)[1], edgecolor="#333333", label=_role(r)[0]) for r in dict.fromkeys(roles)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.1, wspace=0.0)
    save(fig, f"mesh_{name}", tight=False)
