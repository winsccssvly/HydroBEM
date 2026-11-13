"""Physical damping lids, single and coupled, several lids (theory §9.1, test-plan B).

Each check on the basis it rests on. Identities of the problem: a damping lid with eps = 0 equals
the case without the lid (and its elevation the open-water elevation there); the Haskind relation
including the lid contribution; the symmetry of A and B (kappa phi_i phi_j is symmetric) and B >= 0;
sectors, threads, a lid of body elements and the same lid as a mesh of its own, its orientation;
lids coupled with bodies, interior surfaces and walls; at omega = 0 and inf the lid is the free
surface; the scattered formulation with a lid; lids with pressure surfaces. Input rules: the items
and the placement. How the resonance falls with the damping and the comparisons of meshes are
validation, run when the method changes, not tests. Meshes are procedural (tests/python/meshes.py)."""

import math

import numpy as np
import pytest

from bodies import acv_with, flat, graded_cylinder, moonpool_lid_mesh, moonpool_mesh, panel_size_nuv, part, surface
from meshes import ACV_FACES, face_panels, moonpool_lid_panels, moonpool_panels
from problems import model, peak, solve, view

G = 9.80665
# the analytic log integral for every pair: the identities below hold to round-off
TOTAL = {"diffraction_formulation": "total"}


MOON, LID = moonpool_mesh(), moonpool_lid_mesh()
# six frequencies across the moonpool resonance: the identities hold at each
KA = np.linspace(0.5, 3.0, 6)
W = np.sqrt(KA * G)


@pytest.fixture(scope="module")
def sweep():
    out = {None: solve({"mesh": MOON}, W, [0.0], options=TOTAL)}
    for eps in (0.0, 0.02, 0.1, 0.5):
        out[eps] = solve({"mesh": MOON}, W, [0.0], options=TOTAL,
                         lids=[{"name": "moon", "mesh": LID, "damping_factor": eps}])  # fmt: skip
    return out


def test_lid_with_zero_damping_equals_no_lid(sweep):
    # eps = 0 is the free-surface condition (theory §9.1): the same problem; 1e-5: float32 matrices
    # and LU with the lid rows in another place (7.2e-7 found)
    a, b = sweep[None], sweep[0.0]
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        assert peak(view(a, q) - view(b, q)) <= 1e-5 * peak(view(a, q)), q
    # the lid trace is the open-water elevation at the lid centroids: the same potential from the
    # lid unknowns and from the field-point integrals, both float32; 1e-4 (1.4e-5 found for the
    # diffraction, 1.1e-6 for the radiation)
    pts = b.quantity("lids")
    assert pts.shape == (288, 3) and np.all(b.quantity("lids_ids") == 1)
    f = solve({"mesh": MOON}, W, [0.0], options=TOTAL, field_points=pts)
    eta = f.field_points.pressure_diffraction[:, :, 0]
    assert peak(eta - b.quantity("lids_elevation_diffraction")[:, :, 0]) <= 1e-4 * peak(eta)
    # radiation per unit displacement: eta = K phi_j at eps = 0
    fr = f.field_points.pressure_radiation
    assert peak(fr - b.quantity("lids_elevation_radiation")) <= 1e-4 * peak(fr)


def test_haskind_includes_the_lid_contribution(sweep):
    # the Haskind relation with the lid term is an identity of the problem; it holds to the
    # discretization of the sharp-edged moonpool, which the lid does not change: twice the level of
    # the moonpool with an undamped lid, or 2e-2 if larger (1.3e-2 at every eps)
    x0 = sweep[0.0].excitation.diffraction
    ref = peak(x0 - sweep[0.0].excitation.haskind) / peak(x0)
    for eps in (0.02, 0.1, 0.5):
        r = sweep[eps]
        d = peak(r.excitation.diffraction - r.excitation.haskind) / peak(r.excitation.diffraction)
        assert d <= max(2 * ref, 2e-2), (eps, d)


def test_symmetry_and_damping_with_lids(sweep):
    """Reciprocity holds with lids (kappa phi_i phi_j is symmetric) and the lid dissipates (B >= 0;
    -1e-6: round-off). The discrete A, B of this sharp-edged moonpool are symmetric only up to its
    discretization, which the lid does not change: 1.5 times the asymmetry without the lid plus
    1e-4 (A 9.2e-4 against 7.0e-4, B 2.6e-3 against 5.1e-3). How it falls with the mesh is
    validation."""
    w = np.sqrt(np.array([1.0, 1.5, 2.0]) * G)
    mesh = surface(moonpool_panels(n=12), symmetry="xz_and_yz")
    lid = surface(moonpool_lid_panels(n=12), symmetry="xz_and_yz")
    asym = []
    for lids in ([{"name": "m", "mesh": lid, "damping_factor": 0.5}], []):
        r = solve({"mesh": mesh}, w, [0.0], options=TOTAL, lids=lids)
        A, B = r.added_mass, r.radiation_damping
        asym.append([peak(A - A.transpose(0, 2, 1)) / peak(A), peak(B - B.transpose(0, 2, 1)) / peak(B)])
        assert np.all(np.diagonal(B, axis1=1, axis2=2) >= -1e-6 * peak(B))
    asym = np.array(asym)
    assert np.all(asym[0] <= 1.5 * asym[1] + 1e-4), asym
    for eps in (0.02, 0.1, 0.5):
        B = sweep[eps].radiation_damping
        assert np.all(np.diagonal(B, axis1=1, axis2=2) >= -1e-6 * peak(B))


def quadrants(q, shift=(0.0, 0.0, 0.0)):
    """All four quadrant copies of a z = 0 quadrant mesh, upward normals kept."""
    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            c = q * np.array([sx, sy, 1.0])
            out.append(c if sx * sy > 0 else c[:, ::-1])
    return np.concatenate(out) + np.array(shift)


def test_body_record_lid_equals_external_lid_and_sector_solve():
    # the lid as elements of the body mesh (placed with the body) and as a whole mesh: the same
    # equations, solved with the body's sectors; 1e-6 allows float32 sums in another order (0 found,
    # also for the trace)
    hull, lid = moonpool_panels(n=8), moonpool_lid_panels(n=8)
    both = surface(np.concatenate([hull, lid]), symmetry="xz_and_yz")
    pos = [1.5, -0.5, 0.0]
    w = np.sqrt(np.array([1.2, 2.0]) * G)
    rec = solve(
        {"name": "b", "mesh": part(both, 1, len(hull)), "position": pos},
        w,
        [30.0],
        options=TOTAL,
        lids=[{"name": "moon", "mesh": part(both, len(hull) + 1, len(hull) + len(lid)), "placed_with": "b",
               "damping_factor": 0.1}],  # fmt: skip
    )
    ext_lid = surface(quadrants(lid, pos))
    ext = solve(
        {"mesh": surface(hull, symmetry="xz_and_yz"), "position": pos},
        w,
        [30.0],
        options=TOTAL,
        lids=[{"name": "moon", "mesh": ext_lid, "damping_factor": 0.1}],
    )
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        assert peak(view(rec, q) - view(ext, q)) <= 1e-6 * peak(view(rec, q)), q
    # the external lid is solved in the body frame with the body's sectors;
    # its trace keeps the order of the external mesh elements
    quads = quadrants(lid, pos)
    centre = quads.mean(axis=1)
    size = np.abs(quads[:, 0] - quads[:, 2]).max()
    assert np.abs(ext.quantity("lids")[:, :2] - centre[:, :2]).max() <= 0.3 * size
    assert peak(
        np.sort(np.abs(ext.quantity("lids_elevation_diffraction")[0, :, 0]))
        - np.sort(np.abs(rec.quantity("lids_elevation_diffraction")[0, :, 0]))
    ) <= (1e-6 * peak(rec.quantity("lids_elevation_diffraction")))


def test_external_lid_of_a_yawed_body_keeps_its_results():
    """A full external lid around a placed and yawed body: the frame solve
    with sectors equals the same problem with the lid perturbed out of
    symmetry by 1e-5 (beyond the matching tolerance 1e-6 of the panel size:
    solved without sectors in the global frame). 1e-5: float32 matrices and LU of other sizes
    (1.8e-7 found)."""
    hull, lid = moonpool_panels(n=8), moonpool_lid_panels(n=8)
    pos, yaw = (2.0, 1.0, 0.0), 30.0
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    q = quadrants(lid)
    q = np.einsum("ij,pkj->pki", np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), q) + np.array(pos)
    body = {"mesh": surface(hull, symmetry="xz_and_yz"), "position": list(pos), "yaw": yaw}
    w = np.sqrt(np.array([1.2, 2.2]) * G)
    a = solve(body, w, [20.0], options=TOTAL, lids=[{"name": "l", "mesh": surface(q), "damping_factor": 0.2}])
    q2 = q.copy()
    q2[0, 1, 0] += 1e-5  # breaks the symmetry of one panel only
    b = solve(body, w, [20.0], options=TOTAL, lids=[{"name": "l", "mesh": surface(q2), "damping_factor": 0.2}])
    for name in ("added_mass", "radiation_damping", "excitation_diffraction"):
        assert peak(view(a, name) - view(b, name)) <= 1e-5 * peak(view(a, name)), name
    # the lid trace is float32 rounding apart (8e-6 here, 1.2e-5 with another processor's
    # linear-algebra kernels; the 1e-5 perturbation itself adds 2.4e-6)
    la, lb = a.quantity("lids_elevation_diffraction"), b.quantity("lids_elevation_diffraction")
    assert peak(la - lb) <= 5e-5 * peak(la)
    np.testing.assert_allclose(a.quantity("lids"), b.quantity("lids"), atol=1e-5)  # same order


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_several_lids_coupled_with_interior_surfaces_and_walls():
    """Two bodies with interior surfaces beside a wall, two lids with different
    damping (moonpool of one body, an external patch between the bodies)."""
    w = np.sqrt(np.array([0.9, 1.6]) * G)
    surf = "waterline"
    patch = surface(np.array([[[3.0, -0.5, 0.0], [4.0, -0.5, 0.0], [4.0, 0.5, 0.0], [3.0, 0.5, 0.0]]]), symmetry="none")
    bodies = [
        {"name": "m", "mesh": MOON, "position": [5.5, 0.0, 0.0], "interior_surface": surf},
        {"name": "c", "mesh": graded_cylinder(6), "position": [1.5, 0.0, 0.0], "interior_surface": surf},
    ]
    # the moonpool lid of body m: an external global mesh around (5.5, 0)
    full = quadrants(moonpool_lid_panels(n=12), (5.5, 0.0, 0.0))
    lids = [
        {"name": "moon", "mesh": surface(full), "damping_factor": 0.05},
        {"name": "gap", "mesh": patch, "damping_factor": 0.3},
    ]
    opts = {"domain": "x_wall", "diffraction_formulation": "total"}
    r = solve(bodies, w, [15.0], options=opts, lids=lids)
    opts2 = {"domain": "x_wall", "diffraction_formulation": "total", "frequency_threads": 2, "inner_threads": 2}
    t = solve(bodies, w, [15.0], options=opts2, lids=lids)
    assert np.array_equal(r.added_mass, t.added_mass)
    assert np.array_equal(r.quantity("lids_elevation_diffraction"), t.quantity("lids_elevation_diffraction"))
    assert set(np.unique(r.quantity("lids_ids"))) == {1, 2}
    # both lids at eps = 0 reproduce the problem without lids (the same equations; 1e-6: float32)
    lids0 = [{**lid, "damping_factor": 0.0} for lid in lids]
    a = solve(bodies, w, [15.0], options=opts, lids=lids0)
    b = solve(bodies, w, [15.0], options=opts)
    # discrete symmetry of A (reciprocity) to the discretization of the coarse problem without lids
    A, B, A0 = r.added_mass, r.radiation_damping, b.added_mass
    assert peak(A - A.transpose(0, 2, 1)) <= 4 * peak(A0 - A0.transpose(0, 2, 1)) / peak(A0) * peak(A)
    assert np.all(np.diagonal(B, axis1=1, axis2=2) >= -1e-6 * peak(B))
    assert peak(a.added_mass - b.added_mass) <= 1e-6 * peak(b.added_mass)
    assert peak(a.excitation.diffraction - b.excitation.diffraction) <= 1e-6 * peak(b.excitation.diffraction)


def test_lid_limits_behave_as_free_surface():
    # at omega = 0 kappa = K = 0 (phi_z = 0), at omega = inf both are infinite (phi = 0): the lid is
    # the free surface (theory §9.1), the same equations; 1e-6: float32 (6.7e-8 found)
    w = [0.0, math.inf, 1.5]
    lid = [{"name": "moon", "mesh": LID, "damping_factor": 0.2}]
    opts = {"diffraction": False, "diffraction_formulation": "total"}
    a = solve({"mesh": MOON, "modes": ["surge", "pitch"]}, w, options=opts, lids=lid)
    b = solve({"mesh": MOON, "modes": ["surge", "pitch"]}, w, options=opts)
    for key in ("zero", "infinite"):
        x, y = a.quantity("added_mass_" + key + "_frequency"), b.quantity("added_mass_" + key + "_frequency")
        assert peak(x - y) <= 1e-6 * peak(y)


def test_lid_orientation_does_not_matter():
    # elements oriented downward (into the fluid) are reoriented: bit-identical results for the
    # external lid; the body lid and the external one solve the same equations, 1e-6 allows float32
    # sums in another order (0 found)
    w = W[::2]
    up = solve({"mesh": MOON}, w, [0.0], options=TOTAL, lids=[{"name": "m", "mesh": LID, "damping_factor": 0.1}])
    down_mesh = surface(moonpool_lid_panels(n=12)[:, [0, 3, 2, 1]], symmetry="xz_and_yz")
    down = solve({"mesh": MOON}, w, [0.0], options=TOTAL,
                 lids=[{"name": "m", "mesh": down_mesh, "damping_factor": 0.1}])  # fmt: skip
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        assert view(up, q).tobytes() == view(down, q).tobytes(), q
    hull = moonpool_panels(n=12)
    for lid in (moonpool_lid_panels(n=12), moonpool_lid_panels(n=12)[:, [0, 3, 2, 1]]):
        mesh = surface(np.concatenate([hull, lid]), symmetry="xz_and_yz")
        r = solve({"name": "b", "mesh": part(mesh, 1, len(hull))}, w, [0.0], options=TOTAL,
                  lids=[{"name": "m", "mesh": part(mesh, len(hull) + 1, len(hull) + len(lid)), "placed_with": "b",
                         "damping_factor": 0.1}])  # fmt: skip
        assert peak(r.added_mass - up.added_mass) <= 1e-6 * peak(up.added_mass)


def test_formulation_auto_is_total_with_a_lid():
    # the default formulation "auto" solves a model with a lid in the total formulation, one
    # without in the scattered formulation
    w = W[::2]
    lid = [{"name": "m", "mesh": LID, "damping_factor": 0.1}]
    auto = solve({"mesh": MOON}, w, [0.0], lids=lid)
    total = solve({"mesh": MOON}, w, [0.0], options=TOTAL, lids=lid)
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        assert view(auto, q).tobytes() == view(total, q).tobytes(), q
    plain = solve({"mesh": MOON}, w, [0.0])
    scattered = solve({"mesh": MOON}, w, [0.0], options={"diffraction_formulation": "scattered"})
    assert plain.excitation.diffraction.tobytes() == scattered.excitation.diffraction.tobytes()


def test_lid_items_and_placement_are_checked():
    hull, lid = moonpool_panels(n=8), moonpool_lid_panels(n=8)
    both = surface(np.concatenate([hull, lid]), symmetry="xz_and_yz")
    m = model({"name": "b", "mesh": part(both, 1, len(hull))}, [1.0], [0.0],
              lids=[{"name": "moon", "mesh": part(both, len(hull) + 1, len(hull) + len(lid)), "placed_with": "b",
                     "damping_factor": 0.1}])  # fmt: skip
    moon = m.lids["moon"]
    assert moon.placed_with == "b" and moon.mesh == "b"
    assert moon.ranges.tolist() == [[len(hull) + 1, len(hull) + len(lid)]]
    with pytest.raises(ValueError, match="damping_factor"):
        moon.damping_factor = -1.0
    # a lid of elements of a trimmed body is refused (a lid is given as a mesh of its own then)
    m.bodies["b"].trim = [0.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="trim"):
        m.route()


# ------------------------------------------------------------- the scattered formulation, pressure surfaces

QS = ("added_mass", "radiation_damping", "excitation_diffraction")


def test_scattered_formulation_with_a_lid():
    # the scattered potential with its boundary value (K - kappa) phi_I on the lid: the total and
    # the scattered formulation differ by what they differ without the lid (eps = 0, the excitation
    # 3.8e-3 of the peak, the field 6.2e-3, the lid elevation 1.4e-2), which halves and more with
    # twice the panels (1.3e-3, 2.1e-3, 6.1e-3)
    w = np.sqrt(np.array([0.5, 1.5]) * G)
    pts = np.array([[3.0, 0.5, 0.0], [0.2, 0.3, -0.5], [2.5, 2.0, -0.3]])

    def diff(eps):
        out = [solve({"mesh": MOON}, w, [0.0], options={"diffraction_formulation": f}, field_points=pts,
                     outputs={"field_pressure": True}, lids=[{"name": "moon", "mesh": LID, "damping_factor": eps}])
               for f in ("total", "scattered")]  # fmt: skip
        t, s = out
        lt, ls = (r.lids["moon"].elevation_diffraction for r in out)
        ft, fs = (r.field_points.pressure_diffraction for r in out)
        for q in ("added_mass", "radiation_damping"):
            np.testing.assert_array_equal(view(t, q), view(s, q))
        xt, xs = t.excitation.diffraction, s.excitation.diffraction
        return [peak(xt - xs) / peak(xt), peak(ft - fs) / peak(ft), peak(lt - ls) / peak(lt)]

    d0, d = diff(0.0), diff(0.5)
    assert d[0] <= 1.1 * d0[0] + 1e-4 and d[1] <= 1.5 * d0[1] and d[2] <= 1.5 * d0[2], (d, d0)
    assert max(d) <= 2e-2, d


def _acv_lid(eps, n=2):
    # the air-cushion vessel and a lid beside it, x in [12, 20], y in [0, 6] and its reflections
    lid = surface(face_panels((12, 0, 0), (8, 0, 0), (0, 6, 0), 4 * n, 3 * n, (0, 0, 1)), symmetry="xz_and_yz")
    return [{"name": "l", "mesh": lid, "damping_factor": eps}]


@pytest.mark.parametrize("zc", [-1.0, 0.0])
def test_pressure_surface_with_a_lid(zc):
    # the lid at zero damping is no lid; with damping 0.5 it takes 5 % of the damping and 4 % of the
    # exciting force; the two formulations differ as without the lid, the Haskind force too
    w = 2 * np.pi / np.array([5.0, 8.0])

    def run(lids, f):
        return model(acv_with(zc), w, [0.0], environment={"depth": 20.0}, lids=lids,
                     options={"diffraction_formulation": f}).solve()  # fmt: skip

    none, none_s = run([], "total"), run([], "scattered")
    zero, lid, lid_s = run(_acv_lid(0.0), "total"), run(_acv_lid(0.5), "total"), run(_acv_lid(0.5), "scattered")
    for q in QS:
        assert peak(view(zero, q) - view(none, q)) <= 1e-6 * peak(view(none, q)), q
    assert peak(lid.radiation_damping - none.radiation_damping) >= 1e-2 * peak(none.radiation_damping)
    dx = [peak(a.excitation.diffraction - b.excitation.diffraction) / peak(a.excitation.diffraction)
          for a, b in ((lid, lid_s), (none, none_s))]  # fmt: skip
    assert dx[0] <= 1.1 * dx[1] + 1e-4, dx
    hk = [peak(r.excitation.haskind - r.excitation.diffraction) / peak(r.excitation.diffraction) for r in (lid, none)]
    assert hk[0] <= 1.1 * hk[1] + 1e-4, hk


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_higher_order_pressure_surface_with_a_lid():
    # the same in the higher order (the lid a patch of the vessel's mesh): no lid at zero damping
    # (1e-10: the same equations in double precision); A and B symmetric within 1e-4, the patch
    # discretization as without the lid. The lid's effect against the low order is validation
    w = 2 * np.pi / np.array([5.0, 8.0])
    faces = ACV_FACES + [((12, 0, 0), (8, 0, 0), (0, 6, 0), (0, 0, 1))]
    v = np.array([face_panels(o, du, dv, 1, 1, n)[0] for o, du, dv, n in faces])
    m = flat(v, [(*panel_size_nuv(q, 1.0), 3, 3, 3, 4) for q in v], symmetry="xz_and_yz")
    modes = [{"name": "p0", "shape": "uniform"}, {"name": "p1", "shape": "sign_x"}]
    body = {"name": "acv", "mesh": part(m, 1, 6),
            "pressure_surfaces": [{"mesh": part(m, 7, 7), "pressure_modes": modes}]}

    def run(eps):
        lid = {"name": "l", "mesh": part(m, 8, 8), "placed_with": "acv", "damping_factor": eps}
        lids = [] if eps is None else [lid]
        return model(body, w, [0.0], environment={"depth": 20.0}, lids=lids,
                     options={"higher_order": True, "diffraction_formulation": "total"}).solve()  # fmt: skip

    none, zero, lid = run(None), run(0.0), run(0.5)
    for q in QS:
        assert peak(view(zero, q) - view(none, q)) <= 1e-10 * peak(view(none, q)), q
    A, B = lid.added_mass, lid.radiation_damping
    assert peak(A - A.transpose(0, 2, 1)) <= 1e-4 * peak(A) and peak(B - B.transpose(0, 2, 1)) <= 1e-4 * peak(B)


def test_the_low_order_velocity_on_a_pressure_surface_in_z0():
    # the panel coincides with its free-surface image: the vertical velocity from the identity of the
    # source rows (the free-surface condition holds exactly), the horizontal one the limit of a
    # cushion lowered towards z = 0 (linear in its depth)
    w = 1.5
    out = {}
    for zc in (0.0, -0.01, -0.02):
        outputs = {"body_velocity": True, "body_pressure": True}
        m = model(acv_with(zc=zc, modes=("uniform",)), [w], [0.0], options=outputs)
        r = m.solve()
        pressure = np.array([x == "pressure_surface" for x in m.panels().role])
        bp = r.body_points
        out[zc] = (bp.velocity_diffraction[0, pressure, 0], bp.pressure_diffraction[0, pressure, 0])
    v, p = out[0.0]
    assert np.all(np.isfinite(v))
    assert np.abs(v[:, 2] - 1j * w * p).max() <= 1e-5 * np.abs(v[:, 2]).max()  # phi_z = K phi
    d1, d2 = (np.abs(out[zc][0][:, :2] - v[:, :2]).max() / np.abs(v).max() for zc in (-0.01, -0.02))
    assert d1 <= 0.1 and 1.6 <= d2 / d1 <= 2.4
