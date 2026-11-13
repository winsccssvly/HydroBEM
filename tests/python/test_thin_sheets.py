"""Thin (dipole) sheets, low order, infinite and finite depth (theory §9.5, theory §9.6, test-plan B).

Each check on the basis it rests on. A reference: the added mass of a submerged thin disk against
the unbounded-fluid value (8/3) rho a^3 (Newman 1977, ch. 4), at the finest of three meshes and
extrapolated. Identities of the problem: reversing the sheet winding leaves the forces unchanged;
symmetry sectors equal the full solve (also in finite depth, which tends to infinite depth for a
deep bottom, O(exp(-2 K h))); a sheet beside a wall equals the sheet and its mirror image; an
interior surface leaves the solution away from the irregular frequencies; a lid without damping
is no lid; the dipoles as a mesh of their own. Input rules: routes that are not available are
rejected. The closed slabs that tend to the sheet (alone, with a lid, with a pressure surface,
higher order) are validation, run when the method changes, not tests. Meshes are procedural
(tests/python/meshes.py)."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import flat, heave_pieces, moonpool_lid_mesh, moonpool_mesh, part, submerged_plate, surface
from meshes import PLATE_FACES, box_panels, cylinder_panels, face_panels
from problems import endpoints, model, peak, view

RHO, G = 1000.0, 9.80665


def moonpool_lid_panels(a, z, nr, nphi, quarter=True):
    """Horizontal disk of radius a at depth -z, radial spacing refined at the edge."""
    ph = np.linspace(0.0, np.pi / 2 if quarter else 2 * np.pi, nphi + 1)
    rs = a * np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, nr + 1))
    out = []
    for i in range(nr):
        for j in range(nphi):
            q = [[r * np.cos(t), r * np.sin(t), z] for r, t in ((rs[i], ph[j]), (rs[i + 1], ph[j]),
                                                                  (rs[i + 1], ph[j + 1]), (rs[i], ph[j + 1]))]
            if i == 0:
                q = [q[0], q[1], q[2], q[0]]
            out.append(q)
    return np.array(out, dtype=float)


def sheet_body(vertices, symmetry="xz_and_yz", **kw):
    """A body of one sheet: no wetted surface, the sheet its dipoles."""
    return {"dipoles": [{"mesh": surface(vertices, symmetry=symmetry)}], **kw}


def with_sheets(m, n_hull, n_sheet, **kw):
    """A body of the hull elements 1..n_hull of a mesh with the next n_sheet elements as its sheets."""
    return {"mesh": part(m, 1, n_hull), "dipoles": [{"mesh": part(m, n_hull + 1, n_hull + n_sheet)}], **kw}


def limits(body):
    return endpoints(model(body, [0.0, math.inf], (), environment={"density": RHO, "gravity": G},
                 options={"diffraction": False}).solve())  # fmt: skip


def test_submerged_disk_added_mass_converges():
    # the closed form (8/3) rho a^3 of the disk in unbounded fluid. Depth 10 a: the free-surface
    # images change A33 by O((a/d)^3) with opposite signs at omega = 0 and inf; their mean is the
    # unbounded value. The jump has a square-root singularity at the edge: first-order convergence,
    # 3.5e-2 at 24 panels (3.1e-2 found) and 1e-2 after the extrapolation in h (2.1e-3 found)
    a, ref = 1.0, 8.0 / 3.0 * RHO
    err = []
    for n in (8, 16, 24):
        e = limits(sheet_body(moonpool_lid_panels(a, -10.0, n, n), modes=["heave"]))
        err.append(0.5 * (e["zero"][2, 0] + e["infinite"][2, 0]) / ref - 1.0)
    err = np.array(err)
    assert abs(err[-1]) <= 0.035, err
    h = 1.0 / np.array([8, 16, 24])
    extrap = err[1] - (err[2] - err[1]) * h[1] / (h[2] - h[1])
    assert abs(extrap) <= 0.01, (err, extrap)


def test_winding_reversal_leaves_forces_unchanged():
    w = np.sqrt(np.array([0.4, 1.2]) * G)
    hull = cylinder_panels(1.0, 1.0, n_phi=6, n_z=4, n_r=3, quarter=True)
    # a vertical fin in the plane y = 0.5, off the symmetry planes
    fins = np.array(face_panels((1.0, 0.5, -0.9), (0.6, 0, 0), (0, 0, 0.6), 3, 3, (0, -1, 0)))
    res = []
    for f in (fins, fins[:, ::-1]):
        b = with_sheets(surface(np.concatenate([hull, f]), symmetry="xz_and_yz"), len(hull), len(f))
        res.append(model(b, w, [0.0, 60.0], environment={"density": RHO, "gravity": G}).solve())
    # reversing the winding reverses mu and its normal (theory §9.5): the same equations up to signs;
    # 1e-6: float32 sums in another order (1e-16 found)
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        a, b = view(res[0], q), view(res[1], q)
        assert peak(a - b) <= 1e-6 * peak(a), q
    # the Haskind force with the jump term of the sheet: winding-invariant (as above), and the
    # Haskind relation within 1e-2, the discretization of the coarse hull (2.7e-3 found, 2.6e-3 for
    # the hull without the fin)
    hk0, hk1 = res[0].excitation.haskind, res[1].excitation.haskind
    np.testing.assert_allclose(hk0, hk1, rtol=0, atol=1e-6 * peak(hk0))
    hk = res[0].excitation.haskind
    assert peak(hk - res[0].excitation.diffraction) <= 1e-2 * peak(res[0].excitation.diffraction)


def fins_cylinder(quarter):
    """Cylinder with four radial vertical fins at 45 deg (two planes of symmetry)."""
    hull = cylinder_panels(1.0, 1.0, n_phi=8 if quarter else 32, n_z=4, n_r=3, quarter=quarter)
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    fin = []
    rs, zs = np.linspace(1.0, 1.6, 4), np.linspace(-0.9, -0.2, 4)
    for i in range(3):
        for k in range(3):
            fin.append([[rs[i] * c, rs[i] * s, zs[k]], [rs[i + 1] * c, rs[i + 1] * s, zs[k]],
                        [rs[i + 1] * c, rs[i + 1] * s, zs[k + 1]], [rs[i] * c, rs[i] * s, zs[k + 1]]])  # fmt: skip
    fin = np.array(fin)
    if not quarter:
        fin = np.concatenate([fin * [sx, sy, 1] for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
    return hull, fin


def test_symmetry_sectors_equal_full_solve():
    w = np.sqrt(np.array([0.5, 1.5]) * G)
    out = []
    for quarter in (True, False):
        hull, fin = fins_cylinder(quarter)
        m = surface(np.concatenate([hull, fin]), symmetry="xz_and_yz" if quarter else "none")
        b = with_sheets(m, len(hull), len(fin))
        out.append(model(b, w, [0.0, 30.0], environment={"density": RHO, "gravity": G}).solve())
    # the same equations; 2e-5: the float32 sums of the dipole rows in another order (9.1e-8 found)
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        a, b = view(out[0], q), view(out[1], q)
        assert peak(a - b) <= 2e-5 * peak(b), q


def test_wall_equals_explicit_mirror():
    w = np.sqrt(np.array([0.6, 1.4]) * G)
    plate = np.array(face_panels((-0.5, -0.4, -0.6), (1.0, 0, 0), (0, 0.8, 0), 4, 4, (0, 0, 1)))
    hull = cylinder_panels(0.4, 0.4, n_phi=6, n_z=2, n_r=2, quarter=False)
    pan = np.concatenate([hull, plate])
    mesh = surface(pan)
    env = {"density": RHO, "gravity": G}
    body = with_sheets(mesh, len(hull), len(plate), position=[1.5, 0, 0], modes=["heave"])
    a = model(body, w, [30.0], environment=env, options={"domain": "x_wall"}).solve()
    # the explicit mirror body: the mesh reflected in x = 0 (orientation restored)
    refl = pan * [-1, 1, 1]
    refl = refl[:, [0, 3, 2, 1]]
    m2 = surface(refl)
    b = model([with_sheets(mesh, len(hull), len(plate), name="a", position=[1.5, 0, 0], modes=["heave"]),
               with_sheets(m2, len(hull), len(plate), name="b", position=[-1.5, 0, 0], modes=["heave"])],
              w, [30.0, 150.0], environment=env).solve()  # fmt: skip
    # heave of the body plus the heave of its image (the image moves as the mirror body); 2e-5: the
    # float32 sums of the dipole rows in another order (1.6e-7 found)
    A = b.added_mass[:, 2, 0] + b.added_mass[:, 2, 1]
    assert peak(a.added_mass[:, 2, 0] - A) <= 2e-5 * peak(A)
    # incident wave plus its reflection: headings 30 and 150 deg
    X = b.excitation.diffraction[:, 0, 2] + b.excitation.diffraction[:, 1, 2]
    assert peak(a.excitation.diffraction[:, 0, 2] - X) <= 2e-5 * peak(X)


def fin_body(quarter):
    """Truncated cylinder with a vertical fin off the symmetry planes (and its images)."""
    hull = cylinder_panels(1.0, 1.0, n_phi=6, n_z=4, n_r=3, quarter=True)
    f = np.array(face_panels((1.0, 0.5, -0.9), (0.6, 0, 0), (0, 0, 0.6), 3, 3, (0, -1, 0)))
    if quarter:
        return with_sheets(surface(np.concatenate([hull, f]), symmetry="xz_and_yz"), len(hull), len(f))
    # the images in x = 0 and y = 0 (orientation restored after one reflection)
    def images(p):
        return [p, (p * [-1, 1, 1])[:, ::-1], (p * [1, -1, 1])[:, ::-1], p * [-1, -1, 1]]
    hull, f = np.concatenate(images(hull)), np.concatenate(images(f))
    return with_sheets(surface(np.concatenate([hull, f]), symmetry="none"), len(hull), len(f))


def solve_fd(body, depth, w, heads):
    return model(body, w, heads, environment={"density": RHO, "gravity": G, "depth": depth}).solve()


def test_finite_depth_sheet_deep_water_limit():
    # depth 40 (K h >= 16): the bottom changes the forces by O(exp(-2 K h))
    # and the static bottom images by O((a/2h)^3), below 1e-4
    w = np.sqrt(np.array([0.4, 1.2]) * G)
    a, b = solve_fd(fin_body(True), 40.0, w, [0.0, 60.0]), solve_fd(fin_body(True), math.inf, w, [0.0, 60.0])
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        x, y = view(a, q), view(b, q)
        assert peak(x - y) <= 1e-4 * peak(y), q


def test_finite_depth_sheet_sectors_equal_full_solve():
    w = np.sqrt(np.array([0.5, 1.5]) * G / 3.0)
    heads = [0.0, 60.0]
    a, b = solve_fd(fin_body(True), 3.0, w, heads), solve_fd(fin_body(False), 3.0, w, heads)
    # the same equations; 2e-5: the float32 sums of the dipole rows in another order (2.8e-7 found)
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        x, y = view(a, q), view(b, q)
        assert peak(x - y) <= 2e-5 * peak(y), q


def test_unavailable_sheet_routes_are_rejected():
    # a disk under a small cylinder: the total potential on a sheet is rejected (radiation alone is
    # not affected), Haskind with sheets is valid; a sheet in a symmetry plane of its mesh is not
    hull = cylinder_panels(0.5, 0.5, n_phi=4, n_z=2, n_r=2, quarter=True)
    disk = moonpool_lid_panels(1.0, -3.0, 4, 4)
    body = with_sheets(surface(np.concatenate([hull, disk]), symmetry="xz_and_yz"), len(hull), len(disk))
    with pytest.raises(ValueError, match="scattered"):
        model(body, [1.0], [0.0], options={"diffraction_formulation": "total"}).route()
    model(body, [1.0], (), options={"diffraction": False, "diffraction_formulation": "total"}).route()
    model({**body, "excitation_method": "haskind", "mass": "displaced", "center_of_gravity": [0, 0, 0]},
          [1.0], [0.0]).route()  # fmt: skip
    keel = np.array(face_panels((0.0, 0.0, -1.0), (1.0, 0, 0), (0, 0, 0.5), 2, 2, (0, 1, 0)))
    box = cylinder_panels(0.3, 0.3, n_phi=4, n_z=1, n_r=1, quarter=False)
    keeled = with_sheets(surface(np.concatenate([box[box[:, :, 1].mean(axis=1) >= 0], keel]), symmetry="xz"),
                         int((box[:, :, 1].mean(axis=1) >= 0).sum()), len(keel))  # fmt: skip
    with pytest.raises(hb.UnsupportedProblemError, match="symmetry plane"):
        model(keeled, [1.0], [0.0]).route()
    # the same in the higher order: a keel patch in y = 0 of a mesh symmetric about it would coincide
    # with its own image (the symmetric solve differed from the full one by 28 % in added mass)
    full = box_panels(1.0, 0.5, 0.5, n=(2, 2, 2))
    half = full[full[:, :, 1].mean(axis=1) > 1e-9]
    fin = np.array(face_panels((0.0, 0.0, -1.0), (1.0, 0, 0), (0, 0, 0.5), 2, 1, (0, 1, 0)))
    pm = flat(np.concatenate([half, fin]), (2, 2, 3, 3, 3, 4), symmetry="xz")
    ho_keeled = {"mesh": part(pm, 1, len(half)), "dipoles": [{"mesh": part(pm, len(half) + 1, len(half) + len(fin))}]}
    with pytest.raises(hb.UnsupportedProblemError, match="symmetry plane"):
        model(ho_keeled, [1.0], [0.0], options={"higher_order": True, "diffraction_formulation": "total"}).route()


def test_sheets_with_an_interior_surface():
    """Thin sheets (fins) on a body with an interior free surface; away from the irregular
    frequencies the interior surface leaves the solution (1e-3 of the peaks at a long period)."""
    hull, fin = fins_cylinder(quarter=True)
    m = surface(np.concatenate([hull, fin]), symmetry="xz_and_yz")
    body = with_sheets(m, len(hull), len(fin))
    w = [1.0]
    plain = model(body, w, [30.0]).solve()
    irr = model({**body, "interior_surface": {"method": "waterline"}}, w, [30.0]).solve()
    assert peak(irr.added_mass - plain.added_mass) <= 1e-3 * peak(plain.added_mass)
    assert peak(irr.excitation.diffraction - plain.excitation.diffraction) <= 1e-3 * peak(plain.excitation.diffraction)


def test_dipole_mesh_equals_the_dipole_records():
    # the sheets as a mesh of their own solve as the same elements given by ranges of the body mesh
    hull, sheet = [], []
    for k, (o, du, dv, nn) in enumerate(PLATE_FACES):
        nu, nv = (max(1, round(np.linalg.norm(e) * 6)) for e in (du, dv))
        (hull if k < 3 else sheet).extend(face_panels(o, du, dv, nu, nv, nn))
    hull, sheet = np.array(hull), np.array(sheet)
    kw = dict(environment={"depth": 4.0}, options={"diffraction_formulation": "scattered"})
    w = [1.0, 2.5]
    # a generalized mode with a piece on hull elements only: its range never selects the sheets of
    # the other mesh, whose own element numbers it covers
    assert len(hull) > len(sheet)
    whole, hm = surface(np.concatenate([hull, sheet]), symmetry="xz_and_yz"), surface(hull, symmetry="xz_and_yz")
    a = model(with_sheets(whole, len(hull), len(sheet), generalized_modes=heave_pieces(part(whole, 1, len(hull)))), w,
              [0.0, 30.0], **kw).solve()  # fmt: skip
    b = model({"mesh": hm, "dipoles": [{"mesh": surface(sheet, symmetry="xz_and_yz")}],
               "generalized_modes": heave_pieces(part(hm, 1, len(hull)))}, w, [0.0, 30.0], **kw).solve()  # fmt: skip
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        x, y = a.quantity(q), b.quantity(q)
        assert peak(x - y) <= 1e-10 * peak(x), q


# ------------------------------------------------------------- sheets with lids and pressure surfaces

# a fixed horizontal plate under the body
G = 9.80665
QS = ("added_mass", "radiation_damping", "excitation_diffraction")


def _first(r, q, d):
    """The rows (and the excitation) of the first body's d coordinates."""
    a = view(r, q)
    return a[..., :d] if q == "excitation_diffraction" else a[:, :d, :]


def test_sheet_with_a_damping_lid():
    # the moonpool cylinder with its lid (scattered formulation) over a sheet at z = -1.6, identities
    # of the problem: the lid at zero damping is no lid (1e-6: float32, the same equations; 2e-7
    # found), the lid changes the coefficients by 20 to 63 % (so that this check sees it); the
    # Haskind relation with the jump and the lid terms to the discretization of the moonpool, which
    # the lid does not change (1.5 times its level without the lid: 4.7e-3 against 5.8e-3)
    w = np.sqrt(np.array([0.6, 1.5]) * G)

    def run(eps=0.5, lid=True):
        p = {"dipoles": [{"mesh": submerged_plate(-1.6, 3.0, 3.0)}]}
        lids = [{"name": "moon", "mesh": moonpool_lid_mesh(), "damping_factor": eps}] if lid else []
        return model([{"name": "cyl", "mesh": moonpool_mesh()}, {"name": "plate", "modes": [], **p}], w, [0.0],
                     lids=lids).solve()  # fmt: skip

    sheet, zero, none = run(), run(0.0), run(lid=False)
    for q in QS:
        assert peak(_first(zero, q, 6) - _first(none, q, 6)) <= 1e-6 * peak(_first(none, q, 6)), q
        assert peak(_first(sheet, q, 6) - _first(none, q, 6)) >= 0.1 * peak(_first(none, q, 6)), q
    err = [peak(r.excitation.haskind[..., :6] - X) / peak(X)
           for r in (sheet, none) for X in [r.excitation.diffraction[..., :6]]]  # fmt: skip
    assert err[0] <= 1.5 * err[1], err


def test_higher_order_sheets_with_an_interior_patch():
    # a fin on the box with its interior patch: away from the irregular frequencies the interior
    # patch leaves the solution of the continuous problem (its column in the sheet rows, K times the
    # source derivative); 1e-3: the discretization of the interior patch
    q = box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True, lid=True)
    fin = np.array([face_panels((1.0, 0.25, -0.4), (0.5, 0, 0), (0, 0, 0.3), 1, 1, (0, 1, 0))[0]])
    disc = [(*nv, 3, 3, 3, 4) for nv in ((4, 2), (2, 2), (4, 2), (4, 2), (2, 2))]
    m = flat(np.concatenate([q, fin]), disc, symmetry="xz_and_yz")
    body = {"mesh": part(m, 1, 3), "dipoles": [{"mesh": part(m, 5, 5)}]}
    opts = {"higher_order": True, "diffraction_formulation": "total"}
    w = np.sqrt(np.array([1.0, 2.0]) * G)
    plain = model(body, w, [30.0], options=opts).solve()
    irr = model({**body, "interior_surface": part(m, 4, 4)}, w, [30.0], options=opts).solve()
    for name in QS:
        assert peak(view(irr, name) - view(plain, name)) <= 1e-3 * peak(view(plain, name)), name
