"""Walls (x_wall, y_wall, orthogonal_walls), interior
surfaces (elements of the body's mesh, a mesh of its own, projected, waterline, graded) and coupled
bodies with interior surfaces.

test-plan B, each test on the basis it rests on: identities of the problem (a wall equals the body
plus its explicit mirror body in open water on the same panels, orthogonal walls four bodies;
interior-surface results equal the plain results away from the irregular frequencies and remove the
spike at them; symmetry sectors, formulations, body permutation, threads); the exact waterplane of
the generated interior meshes; A4 over the whole published range of Hulme (1982) with
irregular-frequency removal; the input rules. The generated interior meshes against an explicit one,
the methods against each other and the convergence of A4 with the mesh are validation, run when the
method changes, not tests. Meshes are procedural."""

import math
from pathlib import Path

import numpy as np
import pytest

import hydrobem as hb

from bodies import graded_cylinder, surface
from meshes import box_panels, cylinder_panels, flat_plate, hemisphere_panels, mirror, moonpool_panels
from problems import add_mesh, flat_surface, hydrostatics, interior_vertices, model, peak, solve, view

RHO, G = 1000.0, 9.80665
DATA = Path(__file__).resolve().parents[1] / "data"


def interior(surface):
    """The body item of an interior surface: a method ("waterline", "graded", "projected"), the
    surface given ((mesh, ranges), or a dict of mesh items: the method "mesh")."""
    if isinstance(surface, str):
        return {"interior_surface": {"method": surface}}
    return {"interior_surface": {"method": "mesh", "mesh": surface}}


# ------------------------------------------------------------- walls

BOX = box_panels(1.0, 0.5, 0.5, n=(4, 2, 2))
# signs of the mirrored rigid modes: reflection x -> -x maps (surge, sway,
# heave, roll, pitch, yaw) velocities to (-, +, +, +, -, -); y -> -y to (+, -, +, -, +, -)
SX = np.array([-1, 1, 1, 1, -1, -1.0])
SY = np.array([1, -1, 1, -1, 1, -1.0])


def test_x_wall_equals_body_plus_mirror_body():
    # the method of images, exact for the discrete equations (theory §6.3); 1e-6: the float32
    # matrices summed in another order (2.2e-7 measured). The Haskind force with the reflected
    # incident wave (theory §2.1) against the pressure integration: 2e-3, the discretization of the
    # coarse box (9.7e-4 measured, 9.4e-4 for the same box in open water)
    w = np.array([1.2, 2.0])
    pos = (2.5, 0.3, 0.0)
    wall = solve({"mesh": surface(BOX), "position": list(pos)}, w, [30.0], options={"domain": "x_wall"})
    two = solve(
        [
            {"name": "a", "mesh": surface(BOX), "position": list(pos)},
            {"name": "b", "mesh": surface(mirror(BOX, 0)), "position": [-pos[0], pos[1], 0.0]},
        ],
        w,
        [30.0, 150.0],  # the incident wave and its reflection in x = 0
    )
    A = two.added_mass[:, :6, :6] + two.added_mass[:, :6, 6:12] * SX
    B = two.radiation_damping[:, :6, :6] + two.radiation_damping[:, :6, 6:12] * SX
    X = two.excitation.diffraction[:, 0, :6] + two.excitation.diffraction[:, 1, :6]
    assert peak(wall.added_mass - A) <= 1e-6 * peak(A)
    assert peak(wall.radiation_damping - B) <= 1e-6 * peak(B)
    assert peak(wall.excitation.diffraction[:, 0] - X) <= 1e-6 * peak(X)
    assert wall.route == "low_order_infinite_x_wall"
    Hk = wall.excitation.haskind
    assert peak(wall.excitation.diffraction - Hk) <= 2e-3 * peak(wall.excitation.diffraction)


def test_y_wall_finite_depth_equals_body_plus_mirror_body():
    # as the x wall, in finite depth: 1e-6, float32 round-off (2.5e-7 measured)
    w = np.array([1.0, 1.7])
    pos = (0.4, 1.8, 0.0)
    wall = solve(
        {"mesh": surface(BOX), "position": list(pos), "yaw": 20.0},
        w,
        [60.0],
        depth=4.0,
        options={"domain": "y_wall"},
    )
    # the image of a body yawed by 20 degrees is the mirrored mesh yawed by -20 degrees
    two = solve(
        [
            {"name": "a", "mesh": surface(BOX), "position": list(pos), "yaw": 20.0},
            {"name": "b", "mesh": surface(mirror(BOX, 1)), "position": [pos[0], -pos[1], 0.0], "yaw": -20.0},
        ],
        w,
        [60.0, -60.0],
        depth=4.0,
    )
    # body b's local frame is the mirror image of body a's: the image motion of
    # a's mode j is mode j of b with the signs SY
    A = two.added_mass[:, :6, :6] + two.added_mass[:, :6, 6:12] * SY
    B = two.radiation_damping[:, :6, :6] + two.radiation_damping[:, :6, 6:12] * SY
    X = two.excitation.diffraction[:, 0, :6] + two.excitation.diffraction[:, 1, :6]
    assert peak(wall.added_mass - A) <= 1e-6 * peak(A)
    assert peak(wall.radiation_damping - B) <= 1e-6 * peak(B)
    assert peak(wall.excitation.diffraction[:, 0] - X) <= 1e-6 * peak(X)


def test_orthogonal_walls_equal_four_bodies():
    # the two reflections of the walls: 1e-6, float32 round-off (2.1e-7 measured)
    w = np.array([1.4])
    pos = (2.0, 1.5, 0.0)
    wall = solve(
        {"mesh": surface(BOX), "position": list(pos)}, w, [20.0], options={"domain": "orthogonal_walls"}
    )
    four = solve(
        [
            {"name": "a", "mesh": surface(BOX), "position": list(pos)},
            {"name": "b", "mesh": surface(mirror(BOX, 0)), "position": [-pos[0], pos[1], 0.0]},
            {"name": "c", "mesh": surface(mirror(BOX, 1)), "position": [pos[0], -pos[1], 0.0]},
            {"name": "d", "mesh": surface(mirror(mirror(BOX, 0), 1)), "position": [-pos[0], -pos[1], 0.0]},
        ],
        w,
        [20.0, 160.0, -20.0, -160.0],
    )
    s = [np.ones(6), SX, SY, SX * SY]
    A = sum(four.added_mass[:, :6, 6 * i : 6 * i + 6] * s[i] for i in range(4))
    B = sum(four.radiation_damping[:, :6, 6 * i : 6 * i + 6] * s[i] for i in range(4))
    X = four.excitation.diffraction[:, :, :6].sum(axis=1)
    assert peak(wall.added_mass - A) <= 1e-6 * peak(A)
    assert peak(wall.radiation_damping - B) <= 1e-6 * peak(B)
    assert peak(wall.excitation.diffraction[:, 0] - X) <= 1e-6 * peak(X)


# ------------------------------------------------------------- interior surfaces

KA_IRR = 2.4446  # first irregular frequency of the cylinder a = d = 1: K = j01 coth(j01 d / a)


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_interior_surface_removes_the_irregular_frequency():
    ka = np.array([1.0, 1.5, 2.40, 2.45, 2.50])  # the discrete irregular frequency of this mesh is near 2.45
    w = np.sqrt(ka * G)
    plain = solve({"mesh": graded_cylinder(20)}, w)
    for method in ("projected", "waterline", "graded"):
        irr = solve({"mesh": graded_cylinder(20), **interior(method)}, w)
        # away from the irregular frequency the results agree (test-plan B: 1e-3; the
        # difference is a discretization error, O(h^2): 3.6e-3, 1.7e-3, 9.6e-4 at 8, 12, 16)
        for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
            a, b = view(plain, q)[:2], view(irr, q)[:2]
            assert peak(a - b) <= 1e-3 * peak(a), (method, q)
        # near it the plain heave added mass is polluted, the extended one smooth
        A33, P33 = irr.added_mass[:, 2, 2], plain.added_mass[:, 2, 2]
        assert abs(P33[3] - A33[3]) > 5e-3 * peak(A33)
        assert abs(A33[3] - 0.5 * (A33[2] + A33[4])) <= 1e-3 * peak(A33)


def test_interior_surface_keeps_the_velocity_at_the_waterline():
    """the source formulation with an interior surface prescribes there the vertical velocity of the
    exterior flow at the waterline (K phi): the body velocity at the waterline equals that without
    the interior surface far below the irregular frequencies (an identity of the continuous
    problem). 2.5e-3 of the peak on 12 panels a quadrant: the discretization of the interior surface
    next to the waterline (1.7e-3 measured, falling with the mesh). With a rigid lid (zero vertical
    velocity) the panels next to the waterline differed by 3.6 % (and the mean drift by pressure
    integration followed the momentum drift no longer)."""
    pan = cylinder_panels(1.0, 0.5, n_phi=12, n_z=6, n_r=6, quarter=True, cosine=True)
    v = []
    for irr in ({}, interior("waterline")):
        m = model({"mesh": surface(pan, symmetry="xz_and_yz"), "modes": [], **irr}, [2.0], [0.0],
                  environment={"depth": math.inf}, outputs={"body_velocity": True})  # fmt: skip
        v.append(m.solve().body_points.velocity_diffraction[0, :, 0, :])
    d = np.linalg.norm(v[1] - v[0], axis=1) / np.linalg.norm(v[0], axis=1).max()
    assert d.max() <= 2.5e-3, d.max()


def test_interior_surface_as_records_or_a_mesh_of_its_own():
    """the lid elements of the body's mesh and the same panels as a mesh of its own are the same
    equations: bit for bit. (The generated surfaces against these are validation: different
    panels, a discretization difference.)"""
    w = np.sqrt(np.array([0.8, 1.6]) * G)
    hull = box_panels(1.0, 0.5, 0.5, n=(8, 4, 4), quarter=True, lid=False)
    lid = box_panels(1.0, 0.5, 0.5, n=(8, 4, 4), quarter=True, lid=True)[len(hull) :]
    with_lid = surface(np.concatenate([hull, lid]), symmetry="xz_and_yz")
    n0 = len(hull)
    a = solve({"mesh": (with_lid, (1, n0)), **interior((with_lid, (n0 + 1, n0 + len(lid))))}, w, [0.0, 30.0])
    b = solve({"mesh": surface(hull, symmetry="xz_and_yz"), **interior(surface(lid, symmetry="xz_and_yz"))}, w,
              [0.0, 30.0])  # fmt: skip
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        np.testing.assert_array_equal(view(a, q), view(b, q), err_msg=q)


def test_interior_surface_is_set_from_a_dict_or_none():
    # the one interior surface of a body, made with it (method detect): a dict makes it anew, None
    # removes it; a value the engine refuses keeps the old one
    hull = box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True, lid=False)
    lid = box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True, lid=True)[len(hull) :]
    n0, n = len(hull), len(hull) + len(lid)
    m = hb.Model()
    add_mesh(m, "box", surface(np.concatenate([hull, lid]), symmetry="xz_and_yz"))
    m.environment.omegas, m.environment.headings = [1.0], [0.0]
    b = m.bodies.add("box", mesh="box", ranges=[(1, n0)])
    assert b.interior_surface == "detect"
    assert m.counts().per_body[0].interior_surface == 0  # no wetted element in z = 0: none
    b.update(**flat_surface("interior_surface", {"method": "graded"}))
    assert (b.interior_surface, b.interior_surface_mesh) == ("graded", None)
    b.update(**flat_surface("interior_surface", {"method": "mesh", "mesh": "box", "ranges": [(n0 + 1, n)]}))
    assert (b.interior_surface, b.interior_surface_mesh) == ("mesh", "box")
    m.check()
    b.interior_surface = "waterline"  # the mesh and ranges kept, unused
    assert (b.interior_surface, b.interior_surface_mesh) == ("waterline", "box")
    b.update(interior_surface="mesh", interior_surface_ranges=[(n0 + 1, n0 + 2), (n0 + 3, n)])
    assert b.interior_surface_ranges.tolist() == [[n0 + 1, n0 + 2], [n0 + 3, n]]
    m.check()
    with pytest.raises(ValueError, match="interior_surface must be one of none, detect, waterline, graded, projected"):
        b.interior_surface = "bogus"
    assert b.interior_surface == "mesh"
    with pytest.raises(TypeError, match="interior_surface must be a string"):
        b.interior_surface = {"method": "graded"}
    b.interior_surface_mesh = None
    with pytest.raises(ValueError, match="interior_surface = 'mesh' needs interior_surface_mesh"):
        m.check()
    b.update(**flat_surface("interior_surface", {"method": "mesh", "mesh": "box", "ranges": [(n0, n)]}))
    with pytest.raises(ValueError, match=f"element {n0} of the mesh 'box' is taken by both"):
        m.check()
    b.interior_surface = "none"
    assert b.interior_surface == "none"
    m.check()
    # the whole mesh with the default interior surface: the lid elements detected
    m.bodies.add("whole", mesh="box", position=[5.0, 0.0, 0.0])
    assert m.counts().per_body[1].interior_surface == len(lid) and m.counts().per_body[1].wetted == n0

@pytest.fixture(scope="module")
def irr_results():
    w = np.sqrt(np.array([1.0, KA_IRR]) * G)
    return solve({"mesh": graded_cylinder(8), **interior("waterline")}, w, [0.0, 45.0])


def test_symmetry_and_haskind_with_interior(irr_results):
    A, B = irr_results.added_mass, irr_results.radiation_damping
    X, Hk = irr_results.excitation.diffraction, irr_results.excitation.haskind
    # A and B are symmetric (the reciprocity of the radiation potentials, also with the interior
    # surface) and the Haskind force is the pressure force (Newman 1977), to the discretization of 8
    # panels per quadrant edge at and below the irregular frequency: 3e-3 for the symmetry (1.1e-3 and
    # 1.9e-3 measured), 1e-2 for Haskind (2.1e-3 measured); B >= 0 on the diagonal to rounding
    assert peak(A - A.transpose(0, 2, 1)) <= 3e-3 * peak(A)
    assert peak(B - B.transpose(0, 2, 1)) <= 3e-3 * peak(B)
    assert np.all(np.diagonal(B, axis1=1, axis2=2) >= -1e-6 * peak(B))
    assert peak(X - Hk) <= 1e-2 * peak(X)
    assert np.all(np.isfinite(A))


def test_total_and_scattered_with_interior():
    # one continuous problem in two unknowns: the formulations differ by the discretization error,
    # as without the interior surface: 3e-3 at 16 panels per quadrant edge (2.1e-3 measured); the
    # radiation problem is the same equations, bit-identical
    w = np.sqrt(np.array([1.0, KA_IRR]) * G)
    surf = interior("waterline")
    sca = solve({"mesh": graded_cylinder(16), **surf}, w, [0.0, 45.0])
    tot = solve(
        {"mesh": graded_cylinder(16), **surf}, w, [0.0, 45.0],
        options={"diffraction_formulation": "total"},
    )
    assert peak(tot.excitation.diffraction - sca.excitation.diffraction) <= 3e-3 * peak(sca.excitation.diffraction)
    np.testing.assert_array_equal(tot.added_mass, sca.added_mass)


def test_sector_solve_equals_full_solve_with_interior():
    # exact for the discrete equations (theory §6.3); 1e-5: float32 round-off (1.9e-7 measured)
    w = np.sqrt(np.array([1.3, KA_IRR]) * G)
    q = solve({"mesh": graded_cylinder(8), **interior("projected")}, w, [25.0])
    f = solve({"mesh": graded_cylinder(8, quarter=False), **interior("projected")}, w, [25.0])
    for name in ("added_mass", "radiation_damping", "excitation_diffraction"):
        a, b = view(q, name), view(f, name)
        assert peak(a - b) <= 1e-5 * peak(a), name


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_limits_and_finite_depth_with_interior():
    w = [0.0, math.inf, math.sqrt(1.5 * G)]
    plain = solve({"mesh": graded_cylinder(20), "modes": ["surge", "pitch"]}, w, depth=3.0,
                  options={"diffraction": False})  # fmt: skip
    irr = solve(
        {"mesh": graded_cylinder(20), "modes": ["surge", "pitch"], **interior("waterline")},
        w,
        depth=3.0,
        options={"diffraction": False},
    )
    # the limits have no irregular frequencies and are solved without the interior unknowns: the
    # same equations but for the sums, 1e-6 (6e-17 measured); at the finite frequency the identity
    # away from the irregular frequencies, 2e-3: the discretization of the interior surface, O(h^2)
    # (1.3e-3 measured)
    for key in ("zero", "infinite"):
        a, b = plain.quantity(f"added_mass_{key}_frequency"), irr.quantity(f"added_mass_{key}_frequency")
        assert peak(a - b) <= 1e-6 * peak(a), key
    assert peak(plain.added_mass - irr.added_mass) <= 2e-3 * peak(plain.added_mass)  # discretization, O(h^2)


def test_coupled_bodies_with_interior_surfaces_and_walls():
    w = np.sqrt(np.array([0.9, 1.8]) * G)
    surf = interior("projected")
    bodies = [
        {"name": "a", "mesh": graded_cylinder(6), "position": [2.0, 0.0, 0.0], **surf},
        {"name": "b", "mesh": graded_cylinder(6, draft=0.7), "position": [5.0, 1.0, 0.0], "yaw": 30.0,
         **surf},
    ]
    r = solve(bodies, w, [10.0], options={"domain": "x_wall"})
    p = solve(bodies[::-1], w, [10.0], options={"domain": "x_wall"})
    # body permutation: the coordinate blocks are exchanged, exact for the discrete equations;
    # 1e-5: float32 round-off of the sums in another order (2.0e-7 measured); threads bit-identical
    perm = np.r_[6:12, 0:6]
    assert peak(r.added_mass - p.added_mass[:, perm][:, :, perm]) <= 1e-5 * peak(r.added_mass)
    X = r.excitation.diffraction
    assert peak(X - p.excitation.diffraction[:, :, perm]) <= 1e-5 * peak(X)
    t = solve(bodies, w, [10.0], options={"domain": "x_wall", "frequency_threads": 2, "inner_threads": 2})
    assert np.array_equal(r.added_mass, t.added_mass)
    assert np.array_equal(r.excitation.diffraction, t.excitation.diffraction)


# ------------------------------------------------------------- generated meshes


def _area(q):
    return 0.5 * np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])[:, 2]


def test_generated_interior_meshes_cover_the_waterplane():
    mp = surface(moonpool_panels(), symmetry="xz_and_yz")
    for method in ("waterline", "projected"):
        m = model({"mesh": mp, **interior(method)}, [1.0], [0.0])
        q = interior_vertices(m)
        wp = hydrostatics(m)[0].waterplane_area
        a = _area(q)
        assert np.all(a > 0) and np.isclose(a.sum(), wp, rtol=1e-12), method
        r = np.hypot(q[..., 0], q[..., 1])
        assert r.min() >= 1.0 - 1e-9 and r.max() <= 2.0 + 1e-9  # the moonpool stays open
    # four columns without symmetry and with quarter symmetry
    col = cylinder_panels(0.5, 1.0, n_phi=16, n_z=2, n_r=2, quarter=False)
    tlp = np.concatenate([col + np.array([sx * 2.0, sy * 2.0, 0.0]) for sx in (-1, 1) for sy in (-1, 1)])
    m = model({"mesh": surface(tlp), **interior("waterline")}, [1.0], [0.0])
    q = interior_vertices(m)
    assert np.isclose(_area(q).sum(), hydrostatics(m)[0].waterplane_area, rtol=1e-12)


def test_generation_rejects_what_it_cannot_handle():
    # an upward-facing wetted face (the top of a submerged part): multi-valued projection
    col = cylinder_panels(0.5, 1.0, n_phi=8, n_z=2, n_r=2, quarter=False)
    top = flat_plate(0.5, 0.5, z=-0.5, up=True, center=(3.0, 0.0))
    body = {"mesh": surface(np.concatenate([col, top])), **interior("projected")}
    with pytest.raises(ValueError, match="multi-valued"):
        interior_vertices(model(body, [1.0], [0.0]))
    # a half cylinder without symmetry flags: the waterline does not close
    half = cylinder_panels(1.0, 1.0, n_phi=4, n_z=2, n_r=2, quarter=True)
    body = {"mesh": surface(half), **interior("waterline")}
    with pytest.raises(ValueError, match="does not close"):
        interior_vertices(model(body, [1.0], [0.0]))


# ------------------------------------------------------------- A4, whole published range


def test_A4_hemisphere_whole_range_with_interior_surface():
    """Hulme (1982) tables 1 and 2 at every Ka > 0 common to both tables (0.1 to 10), on a mesh
    graded towards the waterline of 16 divisions per quadrant edge (test-plan A4, the wider range):
    the projected interior surface removes the irregular frequencies in it. Without it the
    difference from the tables at Ka = 4 and 7 is about 1e-2 (1.1e-2 found), five to ten times
    that of the neighbouring Ka; with it at most 2e-3 there (1.3e-3, 1.0e-3), and at most 7e-3
    over the whole range (5.7e-3 at Ka = 1.2: the discretization of 16 divisions, O(h^2)).
    The whole range on 48 divisions (within 1e-3) and the convergence over 16, 32 and 48 are
    validation; the finest mesh against the tables at Ka = 0.5, 1, 2 is test_low_order's A4."""
    ref = np.loadtxt(DATA / "hulme1982_full.txt")
    ka = ref[:, 0]
    w = np.sqrt(ka * G)
    m0 = RHO * 2 / 3 * math.pi
    m = surface(hemisphere_panels(1.0, 16, 16, quarter=True, graded=True), symmetry="xz_and_yz")

    def err(inner):  # per Ka: the largest difference of the four quantities, relative to their peak
        with pytest.warns(hb.HydroBEMWarning, match="larger than a sixth of the shortest wavelength"):  # Ka 10
            r = solve({"mesh": m, "modes": ["surge", "heave"], **interior(inner)}, w, options={"diffraction": False})
        A, B = r.added_mass, r.radiation_damping
        vals = np.c_[A[:, 2, 1] / m0, B[:, 2, 1] / (w * m0), A[:, 0, 0] / m0, B[:, 0, 0] / (w * m0)]
        return (np.abs(vals - ref[:, 1:]) / np.abs(ref[:, 1:]).max(axis=0)).max(axis=1)

    irregular = np.isin(ka, (4.0, 7.0))
    without, with_ = err("none"), err("projected")
    assert np.all(without[irregular] >= 5e-3), without  # the mesh resolves the irregular frequencies
    assert np.all(with_[irregular] <= 2e-3), with_
    assert with_.max() <= 7e-3, with_


# ------------------------------------------------------------- generated meshes


def _poly_moments(loops):
    """Area and first moments of a region bounded by counter-clockwise outer and
    clockwise inner polygons (shoelace formulas)."""
    a = sx = sy = 0.0
    for p in loops:
        x, y = p[:, 0], p[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        c = x * y1 - x1 * y
        a += 0.5 * c.sum()
        sx += ((x + x1) * c).sum() / 6.0  # int x dA
        sy += ((y + y1) * c).sum() / 6.0  # int y dA
    return np.array([a, sx, sy])


def _mesh_moments(q):
    """Area and first moments of the panels (polygons with a possibly repeated vertex)."""
    out = np.zeros(3)
    for panel in q:
        pts = [panel[0]]
        for v in panel[1:]:
            if np.linalg.norm(v - pts[-1]) > 1e-12 and np.linalg.norm(v - pts[0]) > 1e-12:
                pts.append(v)
        out += _poly_moments([np.array(pts)])
    return out


def _place(pts, pos, yaw):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return np.c_[c * pts[:, 0] - s * pts[:, 1] + pos[0], s * pts[:, 0] + c * pts[:, 1] + pos[1]]


def test_generated_meshes_area_and_first_moments():
    """test-plan B: area and first moments of the generated interior meshes
    against the exact polygonal waterline of the input mesh, placed and yawed."""
    pos, yaw = (1.3, -0.7, 0.0), 25.0
    n = 12

    def ring(r, m):
        t = np.linspace(0, 2 * np.pi, 4 * m + 1)[:-1]
        return np.c_[r * np.cos(t), r * np.sin(t)]

    cases = [
        (
            surface(box_panels(1.0, 0.5, 0.5, n=(4, 2, 2))),
            [np.array([[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]])],
        ),
        (graded_cylinder(n), [ring(1.0, n)]),
        (surface(moonpool_panels(n=n), symmetry="xz_and_yz"), [ring(2.0, n), ring(1.0, n)[::-1]]),
    ]
    for mesh, loops in cases:
        exact = _poly_moments([_place(lp, pos, yaw) for lp in loops])
        for method in ("projected", "waterline") + (("graded",) if len(loops) == 1 else ()):
            body = {"mesh": mesh, "position": list(pos), "yaw": yaw, **interior(method)}
            q = interior_vertices(model(body, [1.0], [0.0]))
            np.testing.assert_allclose(q[..., 2], 0.0)
            got = _mesh_moments(q)
            assert np.all(np.abs(got - exact) <= 1e-12 * np.abs(exact).max()), (method, got, exact)


def _prism(poly, draft, bottom):
    """A vertical prism with the counter-clockwise waterline polygon `poly` and
    a flat bottom of the given rectangles (x0, y0, x1, y1), normals into the fluid."""
    out = []
    for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1], strict=True):
        out.append([[ax, ay, 0.0], [ax, ay, -draft], [bx, by, -draft], [bx, by, 0.0]])
    for x0, y0, x1, y1 in bottom:
        out.append([[x0, y0, -draft], [x0, y1, -draft], [x1, y1, -draft], [x1, y0, -draft]])
    return np.array(out, dtype=float)


def test_graded_interior_layout():
    """The cylinder quadrant (16 waterline segments) is fanned from the
    axis with N = ceil(mean ray / mean segment) radial cells at the sine
    divisions; triangles at the centroid, quadrilaterals elsewhere, upward
    normals; several regions (four columns without symmetry) each have their
    own fan."""
    n = 16
    graded = interior("graded")
    q = interior_vertices(model({"mesh": graded_cylinder(n), **graded}, [1.0], [0.0]))
    seg = 2.0 * math.sin(math.pi / (4 * n))  # chord of the n divisions of the quadrant
    N = math.ceil(1.0 / seg)
    assert q.shape[0] == 4 * n * N
    assert np.all(_area(q) > 0)
    chord = np.unique(np.round(np.hypot(q[..., 0], q[..., 1]), 9))
    expect = np.sin(0.5 * math.pi * np.arange(N + 1) / N)
    # vertices lie on the rays to the waterline nodes: radius t_k
    assert np.allclose(chord, np.unique(np.round(expect, 9))), chord
    tri = np.any(np.all(np.isclose(q, np.roll(q, -1, axis=1)), axis=2), axis=1)  # a repeated corner
    assert tri.sum() == 4 * n and np.allclose(np.hypot(q[tri, ..., 0], q[tri, ..., 1]).min(axis=1), 0.0)
    col = cylinder_panels(0.5, 1.0, n_phi=16, n_z=2, n_r=2, quarter=False)
    tlp = np.concatenate([col + np.array([sx * 2.0, sy * 2.0, 0.0]) for sx in (-1, 1) for sy in (-1, 1)])
    m = model({"mesh": surface(tlp), **interior("graded")}, [1.0], [0.0])
    q = interior_vertices(m)
    assert np.isclose(_area(q).sum(), hydrostatics(m)[0].waterplane_area, rtol=1e-12)
    # every column is fanned from its own centre: the cells that touch (+-2, +-2)
    touch = np.isclose(np.abs(q[..., 0]), 2.0) & np.isclose(np.abs(q[..., 1]), 2.0)
    assert touch.any(axis=1).sum() == 4 * 16


def test_graded_interior_rejects_what_it_cannot_fan():
    # the moonpool: a region with an opening (full mesh); in the stored quadrant not star-shaped
    v = moonpool_panels()
    full = np.concatenate([v, mirror(v, 0), mirror(v, 1), mirror(mirror(v, 0), 1)])
    cases = ((surface(full), "opening"), (surface(v, symmetry="xz_and_yz"), "more than once"))
    for mp, why in cases:
        with pytest.raises(ValueError, match=why):
            interior_vertices(model({"mesh": mp, **interior("graded")}, [1.0], [0.0]))
    # a U-shaped waterplane: a ray from its centroid (1.5, 0.9) meets the waterline twice
    u = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (2.0, 2.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
    body = {"mesh": surface(_prism(u, 0.5, [(0, 0, 3, 1), (0, 1, 1, 2), (2, 1, 3, 2)])),
            **interior("graded")}  # fmt: skip
    with pytest.raises(ValueError, match="more than once"):
        interior_vertices(model(body, [1.0], [0.0]))
    body = {**body, **interior("waterline")}
    assert np.isclose(_area(interior_vertices(model(body, [1.0], [0.0]))).sum(), 5.0, rtol=1e-12)


def _triangle_quality(q):
    """The smallest angle (degrees) and the longest edge of each generated triangle (a panel with
    a repeated corner, which reflected panels have at another place)."""
    same = np.all(np.isclose(q, np.roll(q, -1, axis=1)), axis=2)
    assert np.all(same.sum(axis=1) == 1)  # triangles
    keep = ~same
    t = q[keep].reshape(len(q), 3, 3)[..., :2]
    e = np.roll(t, -1, axis=1) - t  # edges a->b, b->c, c->a
    length = np.linalg.norm(e, axis=2)
    cos = -np.sum(e * np.roll(e, 1, axis=1), axis=2) / (length * np.roll(length, 1, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).min(axis=1), length.max(axis=1)


def test_waterline_interior_quality():
    """The refined waterline triangulation (theory §14): on a cylinder, a moonpool annulus, a
    U-shaped prism (a non-convex waterline) and four columns every triangle has angles of at
    least 25 degrees (none of these waterlines has a sharper corner) and edges of at most twice
    the mean waterline segment (its value per case). The triangulation without refinement had
    slivers of 5.6 degrees on the cylinder, edges ten times the waterline segment at the
    waterline and four times the panels. The cylinder's added mass, damping and exciting force
    on it stay within 1e-4 of the peak of those with the graded surface, at and away from the
    irregular frequency Ka = 2.44 (4.1e-5 measured; both within 1.5e-3 of a fine interior
    surface)."""
    col = cylinder_panels(0.5, 1.0, n_phi=16, n_z=2, n_r=2, quarter=False)
    u = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (2.0, 2.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
    cases = [
        (graded_cylinder(16), 2.0 * math.sin(math.pi / 64)),
        (surface(moonpool_panels(n=16), symmetry="xz_and_yz"), 3.0 * math.sin(math.pi / 64)),
        (surface(_prism(u, 0.5, [(0, 0, 3, 1), (0, 1, 1, 2), (2, 1, 3, 2)])), 1.5),
        (surface(np.concatenate([col + np.array([sx * 2.0, sy * 2.0, 0.0]) for sx in (-1, 1) for sy in (-1, 1)])),
         math.sin(math.pi / 16)),
    ]  # fmt: skip
    for mesh, seg in cases:
        q = interior_vertices(model({"mesh": mesh, **interior("waterline")}, [1.0], [0.0]))
        angle, edge = _triangle_quality(q)
        assert angle.min() >= 25.0, angle.min()
        assert edge.max() <= 2.0 * seg * (1 + 1e-9), (edge.max(), seg)
    ka = np.array([1.0, 2.0, 2.44, 3.0])
    w = np.sqrt(ka * G)
    r = [solve({"mesh": graded_cylinder(16), **interior(m)}, w) for m in ("waterline", "graded")]
    for qn in ("added_mass", "radiation_damping", "excitation_diffraction"):
        a, b = view(r[0], qn), view(r[1], qn)
        assert peak(a - b) <= 1e-4 * peak(b), (qn, peak(a - b) / peak(b))


@pytest.mark.parametrize("method", ["waterline", "graded"])
def test_generated_interior_surface_of_a_half_mesh(method):
    """A mesh with one symmetry plane (x = 0 or y = 0): its waterline ends on both sides of the
    other axis, and the generated surface covers the whole waterplane (the 16-gon)."""
    full = cylinder_panels(1.0, 0.5, n_phi=16, n_z=2, n_r=2, quarter=False)
    polygon = 8.0 * math.sin(2.0 * math.pi / 16)
    for keep, sym in ((lambda c: c[0] >= -1e-9, "yz"), (lambda c: c[1] >= -1e-9, "xz")):
        v = full[[keep(q.mean(axis=0)) for q in full]]
        b = {"mesh": surface(v, symmetry=sym), **interior(method)}
        im = interior_vertices(model(b, [1.0], [0.0]))
        area = 0.5 * np.linalg.norm(np.cross(im[:, 2] - im[:, 0], im[:, 3] - im[:, 1]), axis=1).sum()
        assert area == pytest.approx(polygon, rel=1e-12)


@pytest.mark.parametrize("method", ["projected", "waterline", "graded"])
def test_generated_interior_surface_of_a_body_placed_below_its_mesh_origin(method):
    # the free surface at body-local z = -position z: a mesh lifted by 1 and placed at z = -1 is the
    # same body (its translational coefficients the same; the rotational ones are about the origin)
    pan = cylinder_panels(1.0, 0.5, n_phi=8, n_z=4, n_r=4, quarter=True)
    lifted = pan.copy()
    lifted[..., 2] += 1.0
    out = []
    for v, z in ((pan, 0.0), (lifted, -1.0)):
        body = {"mesh": surface(v, symmetry="xz_and_yz"), "position": [0.0, 0.0, z], "interior_surface": method}
        r = model(body, [2.0], [0.0]).solve()
        out.append((r.added_mass[:, :3, :3], r.radiation_damping[:, :3, :3], r.excitation.diffraction[:, :, :3]))
    # the lifted vertices round differently: equal to float32 round-off (to the bit on some
    # processors, not on others)
    for a, b in zip(*out, strict=True):
        assert peak(a - b) <= 1e-6 * peak(a)
