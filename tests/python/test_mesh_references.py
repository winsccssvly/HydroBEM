"""Element ranges (hb.ranges), mesh references, mesh rules and readers (reference/inputs-meshes)."""

import numpy as np
import pytest

import hydrobem as hb

from bodies import flat, spline, surface
from meshes import box_panels, flat_plate
from problems import add_mesh, flat_surface

# ------------------------------------------------------------- element ranges


def test_ranges_of_masks_and_numbers():
    # a mask over the stored elements, or one-based numbers in any order (a number twice counts once)
    assert hb.ranges([True, True, False, True, False]) == [(1, 2), (4, 4)]
    assert hb.ranges(np.array([False, True, True, True])) == [(2, 4)]
    assert hb.ranges([45, 3, 1, 2, 30, 46, 2]) == [(1, 3), (30, 30), (45, 46)]
    assert hb.ranges(np.array([7], dtype=np.int32)) == [(7, 7)]
    assert all(type(x) is int for r in hb.ranges(np.arange(1, 5)) for x in r)


@pytest.mark.parametrize("bad, err, match", [([False, False], ValueError, "no element is selected"),
                                             ([], ValueError, "no element is selected"),
                                             ([0, 1], ValueError, "one-based"),
                                             ([[1, 2]], ValueError, "one dimension"),
                                             (3, ValueError, "one dimension"),
                                             ([1.0, 2.0], TypeError, "boolean mask or integer"),
                                             (["1-3"], TypeError, "boolean mask or integer")])  # fmt: skip
def test_ranges_refuse(bad, err, match):
    with pytest.raises(err, match=match):
        hb.ranges(bad)


def test_ranges_round_trip_through_the_ranges_of_a_surface():
    # a mask by geometry over the elements of a mesh: the ranges of a surface that takes exactly them
    m = hb.Model()
    mesh = add_mesh(m, "box", surface(box_panels(lid=True)))
    rec = mesh.elements()
    top = rec.normals[:, 2] > 0.5
    body = m.bodies.add("b", mesh="box", ranges=hb.ranges(~top))
    body.update(**flat_surface("interior_surface", {"method": "mesh", "mesh": "box", "ranges": hb.ranges(top)}))
    assert body.interior_surface_ranges.tolist() == [list(r) for r in hb.ranges(top)]
    taken = np.zeros(len(rec.element), dtype=bool)
    for a, b in body.interior_surface_ranges:
        taken[a - 1 : b] = True
    assert np.array_equal(taken, top) and hb.ranges(rec.element[taken]) == hb.ranges(top)
    m.environment.update(omegas=[1.0], headings=[0.0])
    assert m.counts().per_body[0].interior_surface == int(top.sum())


def test_ranges_refuse_numbers_beyond_int32():
    # the engine's elements are int32: 2^32 + 1 is refused, not taken as element 1
    m = hb.Model()
    add_mesh(m, "m", surface(box_panels(quarter=True), symmetry="xz_and_yz"))
    body = m.bodies.add("b", mesh="m")
    for last in (2**32 + 1, 2**31):
        with pytest.raises(ValueError, match=rf"invalid element range \(1, {last}\)"):
            body.ranges = [(1, last)]
    with pytest.raises((TypeError, OverflowError)):
        body.ranges = [(2**65, 2**65)]
    body.ranges = [(1, 2**31 - 1)]
    assert body.ranges.tolist() == [[1, 2147483647]]


def test_ranges_are_checked():
    # one-based ranges first <= last that do not overlap; a range beyond the stored elements of the
    # mesh is refused by the rules
    m = hb.Model()
    n = len(add_mesh(m, "m", surface(box_panels(quarter=True), symmetry="xz_and_yz")).vertices)
    m.environment.omegas, m.environment.headings = [1.0], [0.0]
    body = m.bodies.add("b", mesh="m")
    for r in ((0, 3), (3, 2), (-1, 2)):
        with pytest.raises(ValueError, match=r"invalid element range .* \(one-based, first <= last\)"):
            body.ranges = [r]
    with pytest.raises(ValueError, match="the ranges select no elements"):
        body.ranges = np.zeros((0, 2), dtype=int)
    with pytest.raises(ValueError, match=r"overlapping element ranges \(1, 2\) and \(2, 3\)"):
        body.ranges = [(1, 2), (2, 3)]
    for bad in ([(1.5, 2)], [(1, 2, 3)], "1-2", 5):
        with pytest.raises((TypeError, ValueError), match="ranges must"):
            body.ranges = bad
    for bad in (("m", (1, 2)), 5):
        with pytest.raises(TypeError, match="mesh must be a string"):
            body.mesh = bad
    with pytest.raises(ValueError, match="unknown mesh 'x'"):
        body.mesh = "x"
    assert body.mesh == "m" and body.ranges is None  # a refused value keeps the old one
    body.ranges = [(1, n + 1)]
    with pytest.raises(ValueError, match=str(n + 1)):
        m.check()
    body.ranges = [(n, n), (1, 2)]
    assert body.ranges.tolist() == [[1, 2], [n, n]]


# ------------------------------------------------------------- low-order mesh


def _checked(mesh: dict):
    """A model of one body on the mesh, checked (the mesh rules apply when a model is checked)."""
    m = hb.Model()
    add_mesh(m, "m", mesh)
    m.bodies.add("b", mesh="m")
    m.environment.omegas = [1.0]
    m.environment.headings = [0.0]
    m.options.higher_order = mesh["kind"] != "panels"
    m.check()
    return m.meshes["m"]


def test_mesh_valid_box():
    m = _checked(surface(box_panels(), reference_length=2.0, symmetry="xz"))
    assert len(m.vertices) == 32 and m.reference_length == 2.0 and m.symmetry == "xz"


def _with_record(bad, index=2):
    v = box_panels().copy()
    v[index] = bad
    return v


@pytest.mark.parametrize(
    "bad, reason",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]], "fewer than three distinct vertices"),
        ([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], "zero area"),
        ([[0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]], "convex|self-intersecting"),
        ([[0, 0, 0], [2, 0, 0], [0.5, 0.5, 0], [0, 2, 0]], "convex"),
        ([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]], "self-intersecting"),
    ],
)
def test_mesh_record_errors_name_the_record(bad, reason):
    with pytest.raises(ValueError, match=rf"mesh element 3: .*({reason})"):
        _checked(surface(_with_record(bad)))


def test_non_finite_vertices_are_rejected_when_set():
    with pytest.raises(ValueError, match="vertices must be finite"):
        _checked(surface(_with_record([[0, 0, 0], [1, 0, 0], [1, 1, np.nan], [0, 1, 0]])))


def test_triangle_and_warped_quad_accepted():
    # below the free surface (a wetted element wholly in z = 0 is refused)
    tri = [[0, 0, -0.5], [1, 0, -0.5], [1, 1, -0.5], [1, 1, -0.5]]
    warped = [[0, 0, -0.49], [1, 0, -0.51], [1, 1, -0.49], [0, 1, -0.51]]
    _checked(surface([tri, warped]))


def test_mesh_header_rules():
    mesh = hb.Model().meshes.add("m", kind="panels", vertices=box_panels())
    with pytest.raises(ValueError, match="reference_length must be a positive"):
        mesh.reference_length = 0.0
    with pytest.raises(ValueError, match="symmetry must be one of none, yz, xz, xz_and_yz"):
        mesh.symmetry = "xy"
    with pytest.raises(TypeError, match="symmetry must be a string"):
        mesh.symmetry = (1, 1)
    with pytest.raises(ValueError, match=r"vertices must have shape \(P, 4, 3\)"):
        mesh.vertices = np.zeros((3, 3, 3))


# ------------------------------------------------------------- GDF reader


def _write_gdf(path, panels, header="test mesh", isx=0, isy=0, trailer=None, comments=True, per_line=3):
    lines = [
        header,
        f"1.0 9.80665{'   ULEN GRAV' if comments else ''}",
        f"{isx} {isy}{'   ISX ISY' if comments else ''}",
        f"{len(panels)}{'   NPAN' if comments else ''}",
    ]
    flat = np.asarray(panels).reshape(-1)
    for i in range(0, flat.size, per_line):
        lines.append(" ".join(f"{x:.8g}" for x in flat[i : i + per_line]))
    if trailer:
        lines.append(trailer)
    path.write_text("\n".join(lines) + "\n")


def _load(path, **kw):
    return hb.Model().meshes.load("m", path, **kw)


def test_load_gdf(tmp_path):
    v = box_panels(quarter=True)
    f = tmp_path / "box.gdf"
    _write_gdf(f, v, isx=1, isy=1, per_line=12)
    m = _load(f)
    assert m.label == "test mesh" and m.symmetry == "xz_and_yz"
    np.testing.assert_allclose(m.vertices, v)


def test_load_gdf_fortran_exponent_and_free_format(tmp_path):
    f = tmp_path / "plate.gdf"
    f.write_text("plate\n1.0 9.81\n0 0\n1\n0d0 0 0 1.0D0 0 0\n1 1 0\n0 1 0\n")
    m = _load(f)
    assert len(m.vertices) == 1 and m.file_gravity == 9.81


def test_gdf_zero_trailer(tmp_path):
    f = tmp_path / "t.gdf"
    _write_gdf(f, box_panels(), trailer="0 0 0 0 0 0 0 0 0 0 0 0")
    with pytest.raises(ValueError, match="zero_trailer"):
        _load(f)
    assert len(_load(f, zero_trailer=True).vertices) == 32


def test_gdf_errors(tmp_path):
    f = tmp_path / "t.gdf"
    _write_gdf(f, box_panels(), trailer="1 2 3")
    with pytest.raises(ValueError, match="unexpected data"):
        _load(f, zero_trailer=True)
    _write_gdf(f, box_panels()[:3])
    text = f.read_text().replace("\n3   NPAN", "\n4   NPAN")
    f.write_text(text)
    with pytest.raises(ValueError, match="ends after 3"):
        _load(f)
    f.write_text("plate\n1.0 0.0\n0 0\n1\n0 0 0 1 0 0\n1 1 0\n0 1 0\n")
    with pytest.raises(ValueError, match="GRAV"):
        _load(f)
    for flags in ("2 0", "0 -1"):  # a plane of symmetry or none
        f.write_text(f"plate\n1.0 9.81\n{flags}\n1\n0 0 0 1 0 0\n1 1 0\n0 1 0\n")
        with pytest.raises(ValueError, match="must be 0 or 1"):
            _load(f)


def _waterplane(mesh, z=0.0):
    """The elements of a mesh on the plane z with upward normals: a mask over its elements."""
    rec = mesh.elements()
    return (rec.normals[:, 2] > 0.5) & (np.abs(rec.vertices[..., 2] - z).max(axis=1) <= 1e-9)


def test_waterplane_records():
    v = box_panels(lid=True)
    m = hb.Model()
    assert hb.ranges(_waterplane(add_mesh(m, "a", surface(v)))) == [(33, 40)]
    shifted = v.copy()
    shifted[:, :, 2] -= 0.2
    assert hb.ranges(_waterplane(add_mesh(m, "b", surface(shifted)), z=-0.2)) == [(33, 40)]
    assert not _waterplane(m.meshes["a"], z=-0.1).any()


# ------------------------------------------------------------- higher order

D = (1, 1, 3, 3, 4, 4)


def test_flat_patch_mesh():
    m = _checked(flat(box_panels(n=(2, 1, 1)), D))
    assert len(m.vertices) == 8 and m.discretization.shape == (8, 6)
    with pytest.raises(ValueError, match="patch 1: .*coplanar"):
        _checked(flat([[[0, 0, 0.1], [1, 0, -0.1], [1, 1, 0.1], [0, 1, -0.1]]], D))
    with pytest.raises(ValueError, match="four distinct"):
        _checked(flat([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]]], D))


@pytest.mark.parametrize("kw, match", [(dict(nu=0), r"spans nu, nv \(the columns 1, 2 of discretization\)"),
                                       (dict(ku=7), r"orders ku, kv \(the columns 3, 4 of discretization\)"),
                                       (dict(kv=1), r"orders ku, kv \(the columns 3, 4 of discretization\)"),
                                       (dict(iqui=0), r"Gauss orders iquo, iqui \(the columns 5, 6 of")])  # fmt: skip
def test_discretization_rules(kw, match):
    args = dict(nu=1, nv=1, ku=3, kv=3, iquo=4, iqui=4) | kw
    with pytest.raises(ValueError, match=match):
        _checked(flat(flat_plate(), tuple(args.values())))


def _spline_plate(u_knots, order=3, n=None, z=0.0):
    n = n or len(u_knots) - order
    xs = np.linspace(0, 1, n)
    coef = np.zeros((n, n, 3))
    coef[..., 0] = xs[None, :]
    coef[..., 1] = xs[:, None]
    coef[..., 2] = z
    return (order, order, u_knots, u_knots, coef)


def test_spline_patch_valid_and_c0_rejected():
    # below the free surface (a wetted element wholly in z = 0 is refused)
    good = _spline_plate([0, 0, 0, 0.5, 1, 1, 1], z=-0.5)
    _checked(spline([good], D))
    c0 = _spline_plate([0, 0, 0, 0.5, 0.5, 1, 1, 1])
    with pytest.raises(ValueError, match="C0"):
        _checked(spline([c0], D))
    incomplete = _spline_plate([0, 0, 0.2, 0.5, 1, 1, 1])
    with pytest.raises(ValueError, match="not complete"):
        _checked(spline([incomplete], D))
    # the evaluators hold at most order 8 (an order-9 patch overflowed a fixed array)
    with pytest.raises(ValueError, match="2..8"):
        _checked(spline([_spline_plate([0] * 9 + [1] * 9, order=9)], D))


def test_patch_normals_and_waterplane():
    # spline plate in z = 0: x_u x x_v = +z points into the body, so the
    # normal into the fluid is -z (not an upward waterplane element)
    m = hb.Model()
    assert not _waterplane(add_mesh(m, "s", spline([_spline_plate([0, 0, 0, 1, 1, 1])], D))).any()
    assert hb.ranges(_waterplane(add_mesh(m, "f", flat(flat_plate(), D)))) == [(1, 4)]


def test_mesh_items_are_read_only_copies():
    m = hb.Model()
    mesh = add_mesh(m, "m", surface(box_panels()))
    v = mesh.vertices
    with pytest.raises(ValueError, match="read-only"):
        v[0, 0, 0] = 1.0
    # an item changes only when it is set
    mesh.vertices = 2.0 * v
    assert np.array_equal(mesh.vertices, 2.0 * v) and not np.array_equal(v, 2.0 * v)
