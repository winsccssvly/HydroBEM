"""The geometry file .hbg (reference/inputs-meshes §5): one mesh as JSON, read and written by the engine.

Every mesh kind (surface, flat, B-spline also rational, analytic) is written and read back to the
same items, also through the object of the file (``to_hbg``, ``from_hbg``); a mesh read from GDF
and SPL files and the same mesh through a geometry file give byte-identical results; the
discretization comes from exactly one place (per patch, the rows of the file, a panel size, or the
reader), the panel size by the rule of the GDF reader; the format, the version, the members, their
kinds and the JSON itself are checked with the place of the error."""

import json
import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import flat, spline, spline_cylinder, surface
from meshes import box_panels, cylinder_panels, face_panels, nurbs_cylinder_patches
from problems import G, add_mesh, model, view

DISC = (4, 2, 3, 3, 3, 4)


def _items(mesh):
    """The items of a mesh and of its patches that a geometry file carries, comparable."""
    out = {"kind": mesh.kind, "symmetry": mesh.symmetry, "reference_length": mesh.reference_length,
           "label": mesh.label}  # fmt: skip
    if mesh.kind in ("panels", "flat_patches"):
        out["vertices"] = np.asarray(mesh.vertices).tolist()
    if mesh.kind != "panels":
        out["discretization"] = np.asarray(mesh.discretization).tolist()
    out["patches"] = [p.name for p in mesh.patches] if mesh.kind in ("spline_patches", "analytic_patches") else []
    return out


def _meshes(m):
    """A mesh of every kind (the B-spline one rational, with patch names)."""
    out = {
        "panels": add_mesh(m, "s", surface(cylinder_panels(1.0, 0.5, n_phi=4, n_z=2, n_r=2, quarter=True),
                                           symmetry="xz_and_yz", label="cylinder")),  # fmt: skip
        "flat_patches": add_mesh(m, "f", flat(
                np.array(box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True)),
                DISC,
                symmetry="xz_and_yz",
                reference_length=2.0,
            )),  # fmt: skip
        "analytic_patches": m.meshes.analytic(
            "a",
            "circular_cylinder",
            DISC,
            radius=1.0,
            draft=0.5,
            parts=["side", "bottom", "waterplane"],
            mapping="nonuniform",
        ),  # fmt: skip
    }
    obj = add_mesh(m, "tmp", spline(nurbs_cylinder_patches(1.0, 0.5), DISC, symmetry="xz_and_yz")).to_hbg()
    obj["patches"][0]["name"], obj["patches"][1]["name"] = "side", "bottom"
    out["spline_patches"] = m.meshes.from_hbg("n", obj)
    return out


@pytest.mark.parametrize("kind", ["panels", "flat_patches", "spline_patches", "analytic_patches"])
def test_every_kind_round_trips(tmp_path, kind):
    m = hb.Model()
    mesh = _meshes(m)[kind]
    mesh.write(tmp_path / "x.hbg")
    back = m.meshes.load("back", tmp_path / "x.hbg")
    assert _items(back) == _items(mesh)
    assert back.to_hbg() == mesh.to_hbg()  # the patch items too, every number exact
    assert back.source_file.endswith("x.hbg")
    np.testing.assert_array_equal(back.geometry().volumes, mesh.geometry().volumes)
    # the object of the file: plain JSON values, read back as the same mesh
    obj = mesh.to_hbg()
    assert json.loads(json.dumps(obj)) == obj and obj["format"] == "hbg" and obj["version"] == 3
    assert m.meshes.from_hbg("obj", obj).to_hbg() == obj
    if kind == "spline_patches":
        assert [p.name for p in back.patches] == ["side", "bottom"]
        np.testing.assert_array_equal(back.patches[0].weights, mesh.patches[0].weights)


def test_gdf_and_spl_against_the_geometry_file(tmp_path):
    # the same B-spline mesh from GDF + SPL and through a geometry file: byte-identical results
    m = hb.Model()
    src = add_mesh(m, "src", spline_cylinder())
    src.write(tmp_path / "c.gdf", tmp_path / "c.spl")
    gdf = m.meshes.load("gdf", tmp_path / "c.gdf", higher_order=True, spl=tmp_path / "c.spl")
    gdf.write(tmp_path / "c.hbg")
    hbg = m.meshes.load("hbg", tmp_path / "c.hbg")
    w = np.sqrt(np.array([0.5, 1.5]) * G)
    files = (lambda mo, name: mo.meshes.load(name, tmp_path / "c.gdf", higher_order=True, spl=tmp_path / "c.spl"),
             lambda mo, name: mo.meshes.load(name, tmp_path / "c.hbg"))  # fmt: skip
    out = [model({"mesh": f}, w, [0.0], options={"higher_order": True}).solve() for f in files]
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        np.testing.assert_array_equal(view(out[0], q), view(out[1], q))
    assert _items(hbg)["discretization"] == _items(gdf)["discretization"]


def test_the_discretization_comes_from_one_place(tmp_path):
    m = hb.Model()
    base = add_mesh(m, "b", spline(nurbs_cylinder_patches(1.0, 0.5), DISC, symmetry="xz_and_yz")).to_hbg()
    rows = [p.pop("discretization") for p in base["patches"]]
    assert rows == [list(DISC)] * 2

    def load(name, **members):
        return m.meshes.from_hbg(name, {**base, **members}).discretization.tolist()

    assert load("one", discretization=list(DISC)) == [list(DISC)] * 2  # one row for every patch
    assert load("two", discretization=[list(DISC), [2, 2, 3, 3, 3, 4]]) == [list(DISC), [2, 2, 3, 3, 3, 4]]
    # a panel size: the arc lengths of the edges, the rule of the GDF reader (pi/2 / 0.25 -> 7 spans)
    assert m.meshes.from_hbg("p", {**base, "panel_size": 0.25}).discretization.tolist() == \
        [[7, 3, 3, 3, 3, 4], [7, 5, 3, 3, 3, 4]]  # fmt: skip
    q = m.meshes.from_hbg("q", {**base, "panel_size": 0.5, "spline_order": 4, "inner_quadrature": 5})
    assert q.discretization.tolist() == [[4, 2, 4, 4, 3, 5], [4, 3, 4, 4, 3, 5]]
    # with the reader when the file has none
    path = tmp_path / "none.hbg"
    path.write_text(json.dumps(base), encoding="utf-8")
    assert m.meshes.load("r", path, panel_size=0.25).discretization.tolist() == [[7, 3, 3, 3, 3, 4], [7, 5, 3, 3, 3, 4]]
    for kw, match in (({}, r"the discretization once.*\(0 given\)"),
                      ({"panel_size": 0.25, "discretization": list(DISC)}, r"\(2 given\)")):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            m.meshes.from_hbg("e", {**base, **kw})
    with pytest.raises(ValueError, match=r"\(2 given\)"):
        m.meshes.load("e", _write(tmp_path, {**base, "panel_size": 0.3}), panel_size=0.25)
    with pytest.raises(ValueError, match="for every patch or for none"):
        m.meshes.from_hbg("e", {**base, "patches": [{**base["patches"][0], "discretization": list(DISC)},
                                                    base["patches"][1]]})  # fmt: skip
    with pytest.raises(ValueError, match="a panels mesh has no discretization"):
        square = np.array(face_panels((0, 0, -1), (1, 0, 0), (0, 1, 0), 1, 1, (0, 0, -1)))
        m.meshes.from_hbg("e", {"format": "hbg", "version": 3, "kind": "panels", "panel_size": 1.0,
                                "vertices": square})  # fmt: skip


def _write(tmp_path, obj, name="f.hbg"):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_files_are_checked(tmp_path):
    m = hb.Model()
    ok = _meshes(m)["spline_patches"].to_hbg()
    for obj, match in (
        ({**ok, "format": "gdf"}, r"the member \"format\" must be \"hbg\""),
        ({**ok, "version": 1}, "version 1; this HydroBEM reads version 3"),
        ({**ok, "kind": "trimmed_patches"}, "a trimmed mesh is made by the engine"),
        ({**ok, "kind": "mesh"}, "kind must be one of"),
        ({**ok, "colour": "red"}, r"unknown member \"colour\""),
        ({**ok, "source_file": "x"}, r"\"source_file\" is set by the engine"),
        ({**ok, "vertices": [[[0, 0, 0]] * 4]}, r"\"vertices\" is not an item of a spline_patches mesh"),
        ({**ok, "symmetry": 1}, "symmetry: must be a string, got a number"),
        ({**ok, "reference_length": -1.0}, "reference_length must be a positive"),
        ({k: v for k, v in ok.items() if k != "patches"}, r"a spline_patches mesh needs \"patches\""),
        ({**ok, "patches": [{**ok["patches"][0], "u_knots": [0, 0, 1]}]},
         r"f.hbg: meshes/bad/patches/side: u_knots and v_knots need"),
        ({**ok, "patches": [{**ok["patches"][0], "colour": 1}]},
         r"patch 1: unknown member \"colour\" of a SplinePatch"),
        ({**ok, "patches": [{**ok["patches"][0], "type": "AnalyticPatch"}]}, "has patches of type SplinePatch"),
        ({**ok, "patches": [{**ok["patches"][0], "coefficients": [[[1, 0, 0], [1, 1]]]}]}, "equal lengths"),
        ({**ok, "patches": [{**ok["patches"][0], "coefficients": [[1, [0, 0]]]}]}, "mixes values and arrays"),
        ({**ok, "patches": [{**ok["patches"][0], "u_order": 2.5}]}, "u_order: must be an integer, got 2.5"),
        ({**ok, "patches": [ok["patches"][0], {**ok["patches"][1], "name": "side"}]}, "duplicate patches name"),
    ):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            m.meshes.load("bad", _write(tmp_path, obj))
    assert "bad" not in [x.name for x in m.meshes]  # nothing kept of a failed read
    # the JSON itself, with the place of the error
    p = tmp_path / "broken.hbg"
    p.write_text('{"format": "hbg",\n "version": 1,\n "kind": "spline" "x": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken.hbg: JSON at line 3, column 19: ',' or '}' expected"):
        m.meshes.load("bad", p)
    with pytest.raises(ValueError, match="a geometry file is a JSON object"):
        m.meshes.from_hbg("bad", "[1, 2]")
    with pytest.raises(ValueError, match="without an SPL file"):
        m.meshes.load("bad", _write(tmp_path, ok), spl=tmp_path / "x.spl")
    # the rational patch is still refused as GDF, written as a geometry file
    mesh = m.meshes["n"]
    with pytest.raises(ValueError, match="a GDF file has no weights"):
        mesh.write(tmp_path / "n.gdf")
    mesh.write(tmp_path / "n.hbg")
    assert math.isclose(m.meshes.load("n2", tmp_path / "n.hbg").geometry().volumes[0], math.pi / 2, rel_tol=1e-12)


def test_a_flat_patch_file_with_a_panel_size():
    # the patch edges of a flat-patch mesh measured on its bilinear maps (they were read as splines)
    box = box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True)
    obj = {"format": "hbg", "version": 3, "kind": "flat_patches", "symmetry": "xz_and_yz", "vertices": box.tolist(),
           "panel_size": 0.25}  # fmt: skip
    mesh = hb.Model().meshes.from_hbg("p", obj)
    # the spans floor(L / 0.25) + 1 of the edges: the bottom and the y face 0.5 x 1, the x face 0.5 x 0.5
    assert [list(r[:2]) for r in mesh.discretization] == [[3, 5], [3, 3], [3, 5]]


def test_text_after_the_numbers_of_a_panel_is_a_comment(tmp_path):
    # a low-order GDF and CSF whose vertex lines carry text after their numbers ("end of panel 1")
    # read as the same files without it; a line that begins with text is refused
    pan = cylinder_panels(1.0, 0.5, n_phi=4, n_z=2, n_r=2, quarter=True)
    lines = []
    for k, p in enumerate(pan):
        for j, v in enumerate(p):
            lines.append(" ".join(f"{x:.17g}" for x in v) + (f"   end of panel {k + 1}" if j == 3 else ""))
    head = ["cylinder", "1.0 9.80665    ULEN, GRAV", "1 1    ISX, ISY", f"{len(pan)}    NPAN"]
    (tmp_path / "c.gdf").write_text("\n".join(head + lines) + "\n")
    (tmp_path / "c.csf").write_text(
        "\n".join(["cs", "0    ILOWHICSF", "1 1    ISX, ISY", f"{len(pan)}    NPAN"] + lines) + "\n"
    )
    m = hb.Model()
    a = m.meshes.load("a", tmp_path / "c.gdf")
    b = m.meshes.load_csf("b", tmp_path / "c.csf")
    want = pan.reshape(-1, 4, 3)
    np.testing.assert_array_equal(np.asarray(a.vertices).reshape(-1, 4, 3), want)
    np.testing.assert_array_equal(np.asarray(b.vertices).reshape(-1, 4, 3), want)
    (tmp_path / "d.gdf").write_text("\n".join(head + lines[:4] + ["end"] + lines[4:]) + "\n")
    with pytest.raises(ValueError, match="invalid number 'end'"):
        m.meshes.load("d", tmp_path / "d.gdf")
