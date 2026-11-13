"""The model as objects with named data items in the engine: the schema, the items and
their checks, the object tree, and the model file read and written by the engine."""

import math

import numpy as np
import pytest

import hydrobem as hb
from hydrobem import _core
from hydrobem.objects import schema
from hydrobem.queries import quantity_names

from meshes import cylinder_panels
from problems import FEATURES
from regression_problems import PROBLEMS


def _tree(o):
    """Every item set and every child of an object, with arrays as their dtype, shape and bytes."""
    out = {"type": o.type, "name": o.name, "items": {}}
    for name, d in schema()[o.type]["items"].items():
        if d["kinds"] and o.type == "Mesh" and o.get("kind") not in d["kinds"]:
            continue
        if o.describe(name)["is_set"]:
            v = o.get(name)
            out["items"][name] = (v.dtype.str, v.shape, v.tobytes()) if isinstance(v, np.ndarray) else v
    out["lists"] = {n: [_tree(c) for c in o.children(n)] for n in schema()[o.type]["lists"]}
    return out


def _saved_twice(m, tmp_path):
    """The trees of a model, of the engine model loaded from its model file, and of that saved and
    loaded again; and the loaded document."""
    m.save(tmp_path / "a.hbm")
    doc = _core.Model.load(tmp_path / "a.hbm")
    doc.save(tmp_path / "b.hbm")
    again = _core.Model.load(tmp_path / "b.hbm")
    return _tree(m._doc.object()), _tree(doc.object()), _tree(again.object()), doc


def _problems():
    cases = dict(PROBLEMS)
    for name, build in FEATURES.items():
        cases[f"feature_{name}"] = build

    def tool_text():
        m = PROBLEMS["lo_cylinder_drift"]()
        m.tool.add("gui", text='{"window": [800, 600], "layers": ["hull", "lid"]}')
        return m

    cases["tool_text"] = tool_text
    return cases


CASES = _problems()


@pytest.mark.parametrize("name", list(CASES))
def test_the_engine_reads_the_model_it_writes(name, tmp_path):
    # a model saved as a model file and loaded has every object and item of the original, bit for
    # bit, and saved again it loads to the same
    a, b, c, _ = _saved_twice(CASES[name](), tmp_path)
    assert a == b and b == c


def test_items_of_a_loaded_case(tmp_path):
    m = PROBLEMS["lo_two_bodies"]()
    _, _, _, doc = _saved_twice(m, tmp_path)
    root = doc.object()
    bodies = root.children("bodies")
    assert [b.name for b in bodies] == m.bodies.names()
    for b, mb in zip(bodies, m.bodies, strict=True):
        assert b.get("position").tolist() == mb.position.tolist()
        assert b.get("modes") == mb.modes
        assert b.get("mesh") == mb.mesh and doc.object(f"meshes/{mb.mesh}") is not None
        assert b.path == f"bodies/{mb.name}"
    om = doc.object("environment").get("omegas")
    assert om.tobytes() == m.environment.omegas.tobytes()
    mesh = doc.object(f"meshes/{bodies[0].get('mesh')}")
    first = m.meshes[m.bodies[0].mesh]
    assert mesh.get("kind") == "panels"
    assert mesh.get("vertices").tobytes() == first.vertices.tobytes()
    assert mesh.get("symmetry") == first.symmetry


def test_schema():
    s = schema()
    assert list(s)[0] == "Model"
    assert {"Environment", "Options", "FirstOrder", "SecondOrder", "Body", "Mesh", "DampingLid", "DipoleDamper",
            "FixedDamper", "Tank", "PressureSurface", "Tool"} <= set(s)
    assert not {"QtfFreeSurface", "InteriorSurface", "ControlSurface"} & set(s)  # the surfaces are items
    body = s["Body"]["items"]
    assert body["mass"]["unit"] == "kg" and body["mass"]["default"] is None
    assert body["position"]["shape"] == "3" and body["position"]["unit"] == "m"
    assert body["mesh"]["kind"] == "reference" and body["mesh"]["target"] == "meshes"
    assert body["ranges"]["kind"] == "array" and body["ranges"]["dtype"] == "int" and body["ranges"]["shape"] == "N,2"
    assert body["control_surface_inner_radius"]["switch"] == ("control_surface",)
    assert body["control_surface_inner_radius"]["switch_values"] == ("cylinder",)
    assert s["Environment"]["items"]["periods"]["switch"] == ("frequency_input",)
    assert s["DampingLid"]["items"]["placed_with"]["kind"] == "reference"
    assert s["DampingLid"]["items"]["placed_with"]["target"] == "bodies"
    assert body["inertia_origin"]["choices"] == ("center_of_gravity", "rotation_center")
    assert s["Body"]["lists"]["generalized_modes"]["named"]
    assert s["Environment"]["items"]["domain"]["choices"][0] == "open_sea"
    assert s["FirstOrder"]["items"]["diffraction_formulation"]["choices"] == ("auto", "scattered", "total")
    assert body["interior_surface"]["default"] == '"detect"' and s["SecondOrder"]["lists"] == {}
    assert s["SecondOrder"]["items"]["free_surface_method"]["default"] == '"panels"'
    assert s["Mesh"]["items"]["vertices"]["kinds"] == ("panels", "flat_patches")
    assert s["Mesh"]["lists"]["patches"]["types"] == ("SplinePatch", "AnalyticPatch")


def test_defaults_of_a_new_model():
    doc = _core.Model()
    env, opt, out = doc.object("environment"), doc.object("options"), doc.object("first_order")
    assert env.get("depth") == math.inf and env.get("gravity") == 9.80665 and env.get("density") == 1000.0
    assert out.get("motion") is False and env.get("domain") == "open_sea" and env.get("channel_width") is None
    assert opt.get("log_tolerance") == 1e-4 and opt.get("frequency_threads") == 1
    assert out.get("diffraction_formulation") == "auto"
    assert not out.describe("motion")["is_set"]
    assert doc.object("second_order").get("free_surface_method") == "panels" and doc.object().children("lids") == []
    assert doc.object().children("bodies") == []
    assert doc.object("environment").get("omegas").shape == (0,)


def _model():
    doc = _core.Model()
    root = doc.object()
    mesh = root.add("meshes", "hull")
    mesh.set("vertices", cylinder_panels(1.0, 0.5, n_phi=8, n_z=2, n_r=2, quarter=True))
    mesh.set("symmetry", "xz_and_yz")
    body = root.add("bodies", "hull")
    body.set("mesh", "hull")
    return doc, root, mesh, body


def test_set_get_reset():
    doc, root, mesh, body = _model()
    body.set("mass", 2.5e3)
    assert body.get("mass") == 2.5e3 and body.describe("mass")["is_set"]
    body.set("mass", None)
    assert body.get("mass") is None
    body.set("position", (1.0, 2, -3.0))
    assert body.get("position").tolist() == [1.0, 2.0, -3.0]
    body.set("modes", ("heave", "pitch"))
    assert body.get("modes") == ("heave", "pitch")
    body.set("inertia_origin", "rotation_center")
    assert body.get("inertia_origin") == "rotation_center"
    body.set("inertia_origin", [0.0, 0.0, -0.2])
    assert body.get("inertia_origin").tolist() == [0.0, 0.0, -0.2]
    sheet = body.add("dipoles")
    sheet.set("mesh", "hull")
    sheet.set("ranges", [(45, 45), (1, 30), (31, 31)])  # ranges in any order
    assert sheet.get("ranges").tolist() == [[1, 30], [31, 31], [45, 45]]
    body.set("ranges", [[5, 9]])
    assert body.get("mesh") == "hull" and body.get("ranges").tolist() == [[5, 9]]
    body.set("additional_stiffness", np.eye(6))
    assert body.get("additional_stiffness").shape == (6, 6)
    doc.object("environment").set("domain", "channel")
    doc.object("environment").set("depth", 30)  # an integer is a real number
    assert doc.object("environment").get("depth") == 30.0
    f = doc.object("environment")
    f.set("omegas", [1.0])
    f.set("periods", [4.0, 8.0])  # both kept: frequency_input chooses
    assert f.get("omegas").tolist() == [1.0] and f.get("periods").tolist() == [4.0, 8.0]
    assert f.get("frequency_input") == "omegas"


@pytest.mark.parametrize("path, item, value, error, match", [
    ("bodies/hull", "mass", -1.0, ValueError, r"^bodies/hull: mass must be a finite number >= 0, got -1\.0$"),
    ("bodies/hull", "mass", "heavy", ValueError, "mass must be one of displaced, got 'heavy'"),
    ("bodies/hull", "mass", [1.0], TypeError, "mass must be a real number or one of displaced"),
    ("bodies/hull", "position", [1.0, 2.0], ValueError, r"position must have shape \(3,\), got \(2,\)"),
    ("bodies/hull", "position", [1.0, math.nan, 0.0], ValueError, "position must be finite"),
    ("bodies/hull", "modes", ("pitch", "heave"), ValueError, "must be unique and in the order"),
    ("bodies/hull", "modes", ("bend",), ValueError, "unknown mode 'bend'"),
    ("bodies/hull", "ranges", [(1, 3), (2, 2)], ValueError, r"overlapping element ranges \(1, 3\) and \(2, 2\)"),
    ("bodies/hull", "ranges", [(3, 1)], ValueError, r"invalid element range \(3, 1\)"),
    ("bodies/hull", "ranges", np.zeros((0, 2), dtype=int), ValueError, "the ranges select no elements"),
    ("bodies/hull", "ranges", [1, 3], ValueError, r"ranges must have shape \(N, 2\)"),
    ("bodies/hull", "mesh", "deck", ValueError, "unknown mesh 'deck'"),
    ("bodies/hull", "mesh", ("deck", (1, 2)), TypeError, "mesh must be a string"),
    ("bodies/hull", "inertia_origin", "keel", ValueError, "must be one of center_of_gravity, rotation_center"),
    ("bodies/hull", "additional_stiffness", np.eye(5), ValueError, r"shape \(6, 6\), got \(5, 5\)"),
    ("bodies/hull", "radii_of_gyration", [[1, 2, 0], [0, 1, 0], [0, 0, 1]], ValueError, "symmetric"),
    ("environment", "domain", "lake", ValueError, "^environment: domain must be one of open_sea, x_wall"),
    ("options", "frequency_threads", 0, ValueError, r"frequency_threads must be in 1\.\.1024"),
    ("environment", "channel_reflect", 1.5, ValueError, r"channel_reflect must be in 0\.\.1"),
    ("first_order", "motion", 1, TypeError, "^first_order: motion must be a bool"),
    ("first_order", "diffraction_formulation", "both", ValueError, "must be one of auto, scattered, total"),
    ("environment", "depth", 0.0, ValueError, "depth must be > 0 or inf"),
    ("environment", "omegas", [1.0, -1.0], ValueError, "must be >= 0"),
    ("meshes/hull", "symmetry", "xy", ValueError, "symmetry must be one of none, yz, xz, xz_and_yz"),
])  # fmt: skip
def test_checks_of_a_value_alone(path, item, value, error, match):
    doc, *_ = _model()
    o = doc.object(path)
    before = o.get(item)
    with pytest.raises(error, match=match):
        o.set(item, value)
    after = o.get(item)  # a failed set leaves the item unchanged
    assert (before is None and after is None) or np.array_equal(np.asarray(before), np.asarray(after))


def test_mesh_geometry_is_checked_on_set():
    doc, root, mesh, _ = _model()
    v = np.array(mesh.get("vertices"))
    v[3] = 0.0
    with pytest.raises(ValueError, match="meshes/hull: .*element 4"):
        mesh.set("vertices", v)
    with pytest.raises(ValueError, match="has no item .discretization."):
        mesh.set("discretization", [[2, 2, 3, 3, 3, 4]])


def test_the_object_tree():
    doc, root, mesh, body = _model()
    other = root.add("bodies", "barge", index=0)
    assert [b.name for b in root.children("bodies")] == ["barge", "hull"]
    other.move(1)
    assert [b.name for b in root.children("bodies")] == ["hull", "barge"]
    with pytest.raises(ValueError, match="duplicate body name 'hull'"):
        root.add("bodies", "hull")
    with pytest.raises(ValueError, match="names must not be blank"):
        root.add("bodies", " ")
    tank = body.add("tanks")
    tank.set("mesh", "hull")
    tank.set("ranges", [(1, 4)])
    assert tank.path == "bodies/hull/tanks/1" and doc.object("bodies/hull/tanks/1") == tank
    gm = body.add("generalized_modes", "bend")
    gm.add("pieces").set("w", [[0.4, 2, 0, 0]])
    assert doc.object("bodies/hull/generalized_modes/bend/pieces/1").get("w").tolist() == [[0.4, 2.0, 0.0, 0.0]]
    assert body.get("interior_surface") == "detect"  # an item of the body, its method
    body.set("interior_surface", "waterline")
    with pytest.raises(ValueError, match="no list 'interior_surface'"):
        body.add("interior_surface")
    # renaming a mesh renames the references to it, the ranges kept
    mesh.rename("hull2")
    assert body.get("mesh") == "hull2" and doc.object("meshes/hull2") == mesh
    assert tank.get("mesh") == "hull2" and tank.get("ranges").tolist() == [[1, 4]]
    body.set("name", "ship")
    assert doc.object("bodies/ship") == body
    other.remove()
    assert [b.name for b in root.children("bodies")] == ["ship"]
    with pytest.raises(Exception, match="cannot be removed"):
        doc.object("options").remove()
    copy = doc.copy()
    copy.object("bodies/ship").set("mass", 7.0)
    assert body.get("mass") is None and copy.object("bodies/ship").get("mass") == 7.0


def test_a_model_built_in_the_engine_is_saved_and_loaded(tmp_path):
    doc, root, mesh, body = _model()
    body.set("center_of_gravity", [0.0, 0.0, -0.1])
    body.set("radii_of_gyration", [0.4, 0.4, 0.4])
    doc.object("environment").set("frequency_input", "periods")
    doc.object("environment").set("periods", [4.0, 8.0])
    doc.object("environment").set("headings", [0.0])
    doc.object("first_order").set("motion", True)
    doc.save(tmp_path / "case.hbm")
    assert [f.name for f in tmp_path.iterdir()] == ["case.hbm"]  # the meshes inside
    with pytest.raises(OSError, match="exists"):
        doc.save(tmp_path / "case.hbm")
    doc.save(tmp_path / "case.hbm", overwrite=True)
    m = hb.Model.load(tmp_path / "case.hbm")
    assert m.environment.periods.tolist() == [4.0, 8.0] and m.first_order.motion  # the periods as given
    assert not m.environment.info("omegas").is_set
    assert m.meshes["hull"].vertices.tobytes() == mesh.get("vertices").tobytes()  # every bit
    again = _core.Model.load(tmp_path / "case.hbm")
    assert again.object("bodies/hull").get("radii_of_gyration").tolist() == [0.4, 0.4, 0.4]


def test_load_errors(tmp_path):
    with pytest.raises(OSError, match=r"missing\.hbm"):
        _core.Model.load(tmp_path / "missing.hbm")
    (tmp_path / "case.toml").write_text('format = "hydrobem-case"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a HydroBEM model or results file"):  # no case files
        _core.Model.load(tmp_path / "case.toml")


# ------------------------------------------------------------------ the problem of the model


def _loaded(m, tmp_path):
    m.save(tmp_path / "case.hbm")
    return _core.Model.load(tmp_path / "case.hbm")


def _engine_quantities(doc):
    """The route and every quantity of a solve of an engine model."""
    r = doc.solve()
    return r.route, {quantity_names()[k]: r.quantity(k) for k in r.available()}


@pytest.mark.parametrize("name", ["lo_cylinder_drift", "ho_spline_cylinder_drift"])
def test_the_model_and_its_model_file_solve_the_same(name, tmp_path):
    # a model with its items set in Python and the model read from its model file solve to the
    # same bits (an IO rule). The loaded tree equals the original bit for bit for every case
    # (test_the_engine_reads_the_model_it_writes), so the solve follows from it; one low-order and
    # one higher-order case check that the solve reads nothing outside the tree
    m = CASES[name]()
    route, q = _engine_quantities(m._doc)
    route2, q2 = _engine_quantities(_loaded(m, tmp_path))  # the items read from the model file
    assert route2 == route
    assert sorted(q2) == sorted(q)
    for k in q:
        assert q2[k].dtype == q[k].dtype and q2[k].tobytes() == q[k].tobytes(), k


def test_queries_on_the_model(tmp_path):
    m = PROBLEMS["lo_two_bodies"]()
    doc = _loaded(m, tmp_path)
    body = doc.object().children("bodies")[1]
    h = body.body_hydrostatics()
    assert h["volumes"][2] == m.bodies[1].hydrostatics().volumes[2] and h["volume"] == m.bodies[1].hydrostatics().volume
    # a change of an item is seen by the next query (the built problem is cached until a change)
    doc.object("environment").set("density", 1025.0)
    h = body.body_hydrostatics()
    m.environment.density = 1025.0
    assert h["stiffness"][2][2] == m.bodies[1].hydrostatics().stiffness[2][2]


def _checked(doc):
    doc.check()


@pytest.mark.parametrize("change, match", [
    (lambda d: d.object("bodies/hull").set("interior_surface", "mesh"),
     "bodies/hull: interior_surface = 'mesh' needs interior_surface_mesh"),
    (lambda d: d.object("bodies/hull").set("additional_stiffness", np.eye(6)),
     "stiffness_origin='center_of_gravity' requires|requires center_of_gravity"),
    (lambda d: d.object("bodies/hull").set("mode_mass", np.zeros((7, 7))),
     r"bodies/hull: mode_mass must be 6 x 6 \(6 \+ G, G = 0 generalized modes\), not 7 x 7"),
    (lambda d: d.object("bodies/hull").add("tanks"), "bodies/hull/tanks/1: a Tank needs its mesh"),
    (lambda d: d.object("bodies/hull").add("pressure_surfaces"), "a PressureSurface needs its mesh"),
    (lambda d: d.object().add("lids", "lid"), "lids/lid: a DampingLid needs its mesh"),
    (lambda d: d.object().add("fixed_dampers"), "fixed_dampers/1: a FixedDamper needs its mesh"),
    (lambda d: d.object("bodies/hull").set("control_surface", "cylinder"),
     r"bodies/hull: control_surface = 'cylinder' needs control_surface_radius"),
    (lambda d: [d.object("bodies/hull").set(k, v) for k, v in (
        ("control_surface", "automatic"), ("control_surface_outer", "polygon"), ("control_surface_depth", 1.0),
        ("control_surface_panel_size", 0.5))],
     r"bodies/hull: control_surface = 'automatic' with control_surface_outer = 'polygon' needs "
     r"control_surface_boundary"),
    (lambda d: d.object("bodies/hull").set("control_surface", "automatic"),
     "control_surface_waterline_term = 'body' needs second_order.drift_control_surface"),
    (lambda d: d.object("environment").set("frequency_input", "periods"),
     r"environment: periods is empty \(allowed only with first_order.radiation = false and diffraction = false\)"),
    (lambda d: d.object("second_order").set("qtf_newman", True), "qtf_newman .* switch a drift on"),
    (lambda d: d.object("bodies/hull").remove(), "the model has no bodies"),
    (lambda d: d.object("environment").set("omegas", []), "omegas is empty"),
])  # fmt: skip
def test_check_applies_the_rules_between_objects(change, match):
    doc, *_ = _model()
    doc.object("environment").set("omegas", [1.0])
    doc.object("environment").set("headings", [0.0])
    _checked(doc)
    change(doc)
    if "stiffness_origin" in match:
        doc.object("bodies/hull").set("stiffness_origin", "center_of_gravity")
    if "drift_control_surface" in match:
        for k, v in (("radius", 2.0), ("depth", 1.0), ("panel_size", 0.5), ("waterline_term", "body")):
            doc.object("bodies/hull").set(f"control_surface_{k}", v)
    with pytest.raises(ValueError, match=match):
        _checked(doc)


def test_check_applies_the_model_rules():
    doc, *_ = _model()
    doc.object("environment").set("omegas", [1.0])
    doc.object("environment").set("headings", [0.0])
    doc.object("first_order").set("motion", True)  # the motions need the mass properties (reference/inputs-rules §1)
    with pytest.raises(ValueError, match="center of gravity|center_of_gravity"):
        _checked(doc)
