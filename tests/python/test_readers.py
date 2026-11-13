"""Higher-order GDF (IGDEF = 0 and 1) and SPL readers (reference/inputs-meshes §2), the engine's readers of
``MeshList.load``. Test files are generated here in the layouts of
reference/inputs-meshes §2."""

import numpy as np
import pytest

import hydrobem as hb

from bodies import surface
from meshes import box_panels, flat_plate
from problems import add_mesh

D3 = (2, 2, 3, 3, 3, 4)  # (nu, nv, ku, kv, iquo, iqui)


def _load(path, spl=None, **kw):
    """The higher-order mesh of a GDF (and SPL) file read into a new model."""
    return hb.Model().meshes.load("m", path, higher_order=True, spl=spl, **kw)


def _load_lo(path, **kw):
    """The low-order mesh of a GDF file read into a new model."""
    return hb.Model().meshes.load("m", path, **kw)


def _fmt(values):
    return " ".join(repr(float(v)) for v in values)


def _header(npatch, igdef, isx=0, isy=0, comments=True):
    c = (lambda s: "   " + s) if comments else (lambda s: "")
    return [
        "generated patch mesh",
        "1.0 9.80665" + c("ULEN GRAV"),
        f"{isx} {isy}" + c("ISX ISY"),
        f"{npatch} {igdef}" + c("NPATCH IGDEF"),
    ]


def _wrap(values, per_line):
    return [_fmt(values[i : i + per_line]) for i in range(0, len(values), per_line)]


def spline_plate(nug=2, nvg=1, kug=3, kvg=2, x=(0.0, 2.0), y=(0.0, 1.0), z=-0.5, swap=False):
    """A plane patch whose control points are a regular grid: X_ij = (x_i, y_j, z)
    (u along x, v along y), or u along y and v along x with swap=True."""
    mu, mv = nug + kug - 1, nvg + kvg - 1
    ku = np.concatenate([[0.0] * kug, np.linspace(0, 1, nug + 1)[1:-1], [1.0] * kug])
    kv = np.concatenate([[0.0] * kvg, np.linspace(0, 1, nvg + 1)[1:-1], [1.0] * kvg])
    # file knot vectors have NUG + 2 KUG - 1 = M_u + KUG values
    assert ku.size == nug + 2 * kug - 1 and kv.size == nvg + 2 * kvg - 1
    a, b = np.linspace(*x, mu), np.linspace(*y, mv)
    coef = np.zeros((mv, mu, 3))
    for j in range(mv):
        for i in range(mu):
            coef[j, i] = (b[j], a[i], z) if swap else (a[i], b[j], z)
    return dict(nug=nug, nvg=nvg, kug=kug, kvg=kvg, ku=ku, kv=kv, coef=coef)


def write_spline_gdf(path, patches, knots_per_line=3, trailer=None, **kw):
    lines = _header(len(patches), 1, **kw)
    for p in patches:
        lines.append(f"  {p['nug']} {p['nvg']}")
        lines.append(f"  {p['kug']} {p['kvg']}")
        lines += _wrap(list(p["ku"]), knots_per_line)
        lines += _wrap(list(p["kv"]), knots_per_line)
        for j in range(p["coef"].shape[0]):  # u index fastest
            for i in range(p["coef"].shape[1]):
                lines.append(_fmt(p["coef"][j, i]))
    if trailer:
        lines.append(trailer)
    path.write_text("\n".join(lines) + "\n")


def write_spl(path, rows):
    path.write_text("\n".join(["generated SPL"] + rows) + "\n")


# ------------------------------------------------------------- IGDEF = 0


def test_flat_patch_gdf(tmp_path):
    v = box_panels(n=(2, 1, 1))
    f = tmp_path / "flat.gdf"
    f.write_text("\n".join(_header(len(v), 0) + _wrap(list(v.reshape(-1)), 12)) + "\n")
    m = _load(f, discretization=[D3] * len(v))
    assert m.kind == "flat_patches" and len(m.discretization) == 8
    np.testing.assert_array_equal(m.vertices, v)
    # three numbers per line reads the same
    f.write_text("\n".join(_header(len(v), 0, comments=False) + _wrap(list(v.reshape(-1)), 3)) + "\n")
    np.testing.assert_array_equal(_load(f, discretization=[D3] * len(v)).vertices, v)


def test_line4_needs_npatch_and_igdef(tmp_path):
    v = flat_plate(up=False, z=-1.0)
    f = tmp_path / "x.gdf"
    lines = _header(len(v), 0)
    lines[3] = f"{len(v)}   NPAN"
    f.write_text("\n".join(lines + _wrap(list(v.reshape(-1)), 12)) + "\n")
    with pytest.raises(ValueError, match="NPATCH IGDEF"):
        _load(f, discretization=[D3] * len(v))
    for igdef, message in ((2, "IGDEF = 2 is not supported"), (-6, "IGDEF = -6 is not an available analytic shape")):
        lines[3] = f"{len(v)} {igdef}"
        f.write_text("\n".join(lines + ["1  NLINES", "1.0"]) + "\n")
        with pytest.raises(ValueError, match=message):
            _load(f, discretization=[D3] * len(v))


# ------------------------------------------------------------- IGDEF = 1


def test_spline_gdf_layout_and_coefficient_order(tmp_path):
    p1 = spline_plate()
    p2 = spline_plate(nug=1, nvg=2, kug=2, kvg=3, x=(0.0, 2.0), y=(1.0, 2.0), z=-0.5, swap=False)
    f = tmp_path / "s.gdf"
    write_spline_gdf(f, [p1, p2], knots_per_line=3, isx=1)
    m = _load(f, discretization=[D3, D3])
    assert m.kind == "spline_patches" and len(m.patches) == 2 and m.symmetry == "yz"
    for got, want in zip(m.patches, (p1, p2), strict=True):
        assert (got.u_order, got.v_order) == (want["kug"], want["kvg"])
        np.testing.assert_array_equal(got.u_knots, want["ku"])
        np.testing.assert_array_equal(got.v_knots, want["kv"])
        np.testing.assert_array_equal(got.coefficients, want["coef"])  # (M_v, M_u, 3)
    # knot vectors on one line each read the same
    write_spline_gdf(f, [p1, p2], knots_per_line=100)
    np.testing.assert_array_equal(_load(f, discretization=[D3, D3]).patches[0].coefficients, p1["coef"])


def test_spline_gdf_orientation(tmp_path):
    # x_u x x_v points into the body (reference/inputs-meshes §2): u along x, v along y gives
    # x_u x x_v = +z, so a z = 0 patch faces down; swapping u and v faces up
    f = tmp_path / "s.gdf"
    write_spline_gdf(f, [spline_plate(z=0.0)])
    assert _load(f, discretization=[D3]).elements().normals[0, 2] == pytest.approx(-1.0)
    write_spline_gdf(f, [spline_plate(z=0.0, swap=True)])
    assert _load(f, discretization=[D3]).elements().normals[0, 2] == pytest.approx(1.0)


def test_spline_gdf_errors(tmp_path):
    f = tmp_path / "s.gdf"
    p = spline_plate()
    write_spline_gdf(f, [p], trailer="1.0 2.0 3.0")
    with pytest.raises(ValueError, match="unexpected data after NPATCH"):
        _load(f, discretization=[D3])
    write_spline_gdf(f, [p], trailer="0 0 0")
    with pytest.raises(ValueError, match="zero_trailer"):
        _load(f, discretization=[D3])
    assert len(_load(f, discretization=[D3], zero_trailer=True).patches) == 1
    # a knot vector of zero length: an error before the patch is sized by panel_size, which
    # evaluates its edges (and must not read out of bounds)
    bad = spline_plate()
    bad["ku"] = np.zeros_like(np.asarray(bad["ku"], dtype=float))
    write_spline_gdf(f, [bad])
    m = hb.Model()
    with pytest.raises(ValueError, match="patch 1: u knot vector has zero length"):
        m.meshes.load("s", f, higher_order=True, panel_size=0.25)
    write_spline_gdf(f, [p, p])
    f.write_text(f.read_text().replace("\n2 1   NPATCH", "\n3 1   NPATCH"))
    with pytest.raises(ValueError, match="ends while reading NUG NVG of patch 3"):
        _load(f, discretization=[D3] * 3)
    # a knot line with more numbers than the vector needs (line 7: u knots)
    write_spline_gdf(f, [p], knots_per_line=100)
    lines = f.read_text().splitlines()
    lines[6] += " 0.25"
    f.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="line 7: more numbers than the u knot vector"):
        _load(f, discretization=[D3])
    bad = spline_plate()
    bad["ku"] = np.array([0, 0, 0.5, 0.5, 0.5, 1, 1])  # not complete
    write_spline_gdf(f, [bad])
    with pytest.raises(ValueError, match="patch 1: u knot vector is not complete"):
        _load(f, discretization=[D3])


def test_comment_after_record(tmp_path):
    f = tmp_path / "s.gdf"
    write_spline_gdf(f, [spline_plate()])
    lines = f.read_text().splitlines()
    lines[4] += "   NUG NVG (patch 1)"
    lines[5] += "   KUG KVG"
    f.write_text("\n".join(lines) + "\n")
    assert _load(f, discretization=[D3]).patches[0].u_order == 3


# ------------------------------------------------------------- SPL


@pytest.fixture
def two_patch_gdf(tmp_path):
    f = tmp_path / "s.gdf"
    write_spline_gdf(f, [spline_plate(), spline_plate(y=(1.0, 2.0))])
    return f


def test_spl_full(tmp_path, two_patch_gdf):
    s = tmp_path / "s.spl"
    write_spl(s, ["4 2  NU NV (patch 1)", "3 4", "3 3", "4 4", "", "2 1", "2 2", "2 2", "5 5"])
    m = _load(two_patch_gdf, s)
    assert m.discretization.tolist() == [[4, 2, 3, 4, 3, 4], [2, 1, 2, 2, 2, 5]]


def test_spl_with_keywords(tmp_path, two_patch_gdf):
    s = tmp_path / "s.spl"
    write_spl(s, ["4 2", "2 2"])
    m = _load(two_patch_gdf, s, spline_order=3, outer_quadrature=3, inner_quadrature=4)
    assert m.discretization.tolist() == [[4, 2, 3, 3, 3, 4], [2, 2, 3, 3, 3, 4]]
    write_spl(s, ["4 2", "3 3", "2 2", "4 4"])
    m = _load(two_patch_gdf, s, spline_order=3, inner_quadrature=5)
    assert m.discretization[1].tolist() == [2, 2, 3, 3, 4, 5]


def test_spl_value_never_both_ways(tmp_path, two_patch_gdf):
    s = tmp_path / "s.spl"
    write_spl(s, ["4 2", "3 3", "3 3", "4 4"] * 2)
    with pytest.raises(ValueError, match="never both"):
        _load(two_patch_gdf, s, spline_order=3)
    write_spl(s, ["4 2", "3 3"])
    with pytest.raises(ValueError, match="never both"):
        _load(two_patch_gdf, s)


def test_spl_errors(tmp_path, two_patch_gdf):
    s = tmp_path / "s.spl"
    write_spl(s, ["4 2", "3 3", "3 2", "4 4", "2 2", "3 3", "3 3", "4 4"])
    with pytest.raises(ValueError, match="IQUO IQVO must be equal"):
        _load(two_patch_gdf, s)
    write_spl(s, ["4", "3 3", "3 3", "4 4", "2 2", "3 3", "3 3", "4 4"])
    with pytest.raises(ValueError, match="expected the pair NU NV of patch 1"):
        _load(two_patch_gdf, s)
    write_spl(s, ["4 2 1", "3 3", "3 3", "4 4", "2 2", "3 3", "3 3", "4 4"])
    with pytest.raises(ValueError, match="expected the pair NU NV"):
        _load(two_patch_gdf, s)
    write_spl(s, ["0 2", "3 3", "3 3", "4 4", "2 2", "3 3", "3 3", "4 4"])
    with pytest.raises(ValueError, match="spans nu, nv"):
        _load(two_patch_gdf, s)
    write_spl(s, ["4 2", "2 2"])
    with pytest.raises(TypeError, match="spline_order"):
        _load(two_patch_gdf, s, spline_order=3.0, outer_quadrature=3, inner_quadrature=4)


def test_discretization_ownership(tmp_path, two_patch_gdf):
    s = tmp_path / "s.spl"
    write_spl(s, ["4 2", "3 3", "3 3", "4 4"] * 2)
    with pytest.raises(ValueError, match="not both"):
        _load(two_patch_gdf, s, discretization=[D3, D3])
    with pytest.raises(ValueError, match="only complete an SPL file"):
        _load(two_patch_gdf, discretization=[D3, D3], spline_order=3)
    with pytest.raises(ValueError, match="give `discretization`"):
        _load(two_patch_gdf)
    # one row is the discretization of every patch, as for the analytic shapes
    one, two = _load(two_patch_gdf, discretization=D3), _load(two_patch_gdf, discretization=[D3, D3])
    assert one.discretization.tobytes() == two.discretization.tobytes()
    with pytest.raises(ValueError, match="3 rows for NPATCH = 2"):
        _load(two_patch_gdf, discretization=[D3, D3, D3])


# ------------------------------------------------------------- defaults


def test_pressure_surface_defaults():
    m = hb.Model()
    add_mesh(m, "m", surface(flat_plate(up=False, z=-1.0)))
    b = m.bodies.add("body", mesh="m")
    ps = b.pressure_surfaces.add(mesh="m")
    ps.pressure_modes.add("p")
    assert ps.chamber_coupling is False and ps.response_convention == "variational"


# ------------------------------------------------------------- the engine's readers


def test_zero_trailer_is_accepted_when_asked(tmp_path):
    q = box_panels(1.0, 0.5, 0.5, n=(2, 1, 1), quarter=True)
    lines = ["box", "1.0 9.80665", "1 1", f"{len(q)}"] + [_fmt(v) for v in q.reshape(-1, 3)]
    (tmp_path / "plain.gdf").write_text("\n".join(lines) + "\n")
    (tmp_path / "zeros.gdf").write_text("\n".join(lines + ["0 0 0 0 0 0 0 0 0 0 0 0"]) + "\n")
    with pytest.raises(ValueError, match=f"trailing record of zeros after NPAN = {len(q)}"):
        _load_lo(tmp_path / "zeros.gdf")
    a, b = _load_lo(tmp_path / "plain.gdf"), _load_lo(tmp_path / "zeros.gdf", zero_trailer=True)
    assert a.vertices.tobytes() == b.vertices.tobytes()
    with pytest.raises(OSError, match=r"missing\.gdf"):
        _load_lo(tmp_path / "missing.gdf")


def test_written_spline_mesh_reads_back(tmp_path):
    write_spline_gdf(tmp_path / "s.gdf", [spline_plate(), spline_plate(nug=1, nvg=2, kug=2, kvg=3, z=-1.0)])
    m = _load(tmp_path / "s.gdf", discretization=[D3, D3])
    m.write(tmp_path / "t.gdf", tmp_path / "t.spl")
    n = _load(tmp_path / "t.gdf", tmp_path / "t.spl")
    assert n.discretization.tobytes() == m.discretization.tobytes()
    for p, q in zip(m.patches, n.patches, strict=True):
        assert p.u_knots.tobytes() == q.u_knots.tobytes() and p.v_knots.tobytes() == q.v_knots.tobytes()
        assert p.coefficients.tobytes() == q.coefficients.tobytes()


def test_d_exponents_and_commas(tmp_path):
    q = flat_plate(1.0, 1.0, z=-0.5, n=(1, 1))
    rows = [" , ".join(repr(float(x)).replace("e", "D") if "e" in repr(float(x)) else repr(float(x)) for x in v)
            for v in q.reshape(-1, 3)]  # fmt: skip
    (tmp_path / "c.gdf").write_text("\n".join(["plate", "1.0D0 9.80665", "0, 0", "1  NPAN"] + rows) + "\n")
    assert np.array_equal(_load_lo(tmp_path / "c.gdf").vertices, np.asarray(q, dtype=float).reshape(1, 4, 3))


def test_analytic_gdf_with_spl_keeps_the_spl_path(tmp_path):
    # an analytic-geometry GDF (IGDEF < 0) takes its discretization from the SPL file like a patch GDF,
    # and the mesh keeps where both were read from
    f = tmp_path / "c.gdf"
    lines = _header(2, -1)
    lines[3] = "2 -1"
    f.write_text("\n".join(lines + ["2  NLINES", "1.0 0.5", "1"]) + "\n")
    s = tmp_path / "c.spl"
    write_spl(s, ["2 4", "3 3", "3 3", "4 4", "2 2", "3 3", "3 3", "4 4"])
    m = _load(f, s)
    assert m.kind == "analytic_patches" and m.source_spl == str(s.resolve())
    assert m.discretization.tolist() == [[2, 4, 3, 3, 3, 4], [2, 2, 3, 3, 3, 4]]
