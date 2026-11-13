"""The classes of the model's objects generated from the schema: the generated file is up
to date, every item and list of the schema has a property with its description, and the objects
of a model are of their classes."""

import inspect
import math
import os
import sys
from pathlib import Path

import pytest

import hydrobem as hb
from hydrobem import _classes

from bodies import surface
from meshes import cylinder_panels
from problems import add_mesh, load_script

ROOT = Path(__file__).resolve().parents[2]


# the generated file is the checkout's: on an installed wheel (the checkout's package moved out of
# the path) there is none to compare
CHECKOUT = pytest.mark.skipif(bool(os.environ.get("HYDROBEM_TEST_INSTALLED")), reason="the checkout's file")


@CHECKOUT
def test_the_generated_file_is_up_to_date():
    text = (ROOT / "python" / "hydrobem" / "_classes.py").read_text(encoding="utf-8")
    assert text == load_script("gen_model_classes").generate(), "run scripts/gen_model_classes.py"


@CHECKOUT
def test_the_check_compares_the_bytes(tmp_path, monkeypatch):
    # a copy with CRLF line endings is not the output (the file is written with LF); a missing file
    # fails the check, which writes nothing
    gen = load_script("gen_model_classes")
    monkeypatch.setattr(sys, "argv", ["gen_model_classes.py", "--check"])
    out = tmp_path / "_classes.py"
    monkeypatch.setattr(gen, "OUT", out)
    assert gen.main() == 1 and not out.exists()
    out.write_bytes((ROOT / "python" / "hydrobem" / "_classes.py").read_bytes())
    assert gen.main() == 0
    out.write_bytes(out.read_bytes().replace(b"\n", b"\r\n"))
    assert gen.main() == 1


def test_every_item_and_list_has_a_documented_property():
    schema = hb.engine.schema()
    for t, v in schema.items():
        cls = _classes.ModelLists if t == "Model" else _classes.CLASSES[t]
        for name, d in {**v["items"], **v["lists"]}.items():
            prop = inspect.getattr_static(cls, name)
            assert isinstance(prop, property), (t, name)
            doc = inspect.getdoc(prop)
            assert d["description"].split(" ")[0] in doc, (t, name)
        assert schema[t]["lists"] == {} or all(d["description"] for d in v["lists"].values()), t
        assert all(d["description"] and len(d["description"]) > 12 for d in v["items"].values()), t


def _model():
    m = hb.Model()
    add_mesh(m, "cyl", surface(cylinder_panels(1.0, 0.5, quarter=True), symmetry="xz_and_yz"))
    m.bodies.add("cyl", mesh="cyl", interior_surface="waterline").tanks.add(mesh="cyl", ranges=[(1, 2)])
    return m


def test_the_objects_are_of_their_classes():
    m = _model()
    assert isinstance(m.options, _classes.Options) and isinstance(m.environment, _classes.Environment)
    assert isinstance(m.first_order, _classes.FirstOrder) and isinstance(m.second_order, _classes.SecondOrder)
    body = m.bodies["cyl"]
    assert type(body) is _classes.Body and type(m.meshes["cyl"]) is _classes.Mesh
    assert isinstance(m.meshes["cyl"], hb.engine.MeshObject)  # with the mesh functions
    assert isinstance(body, hb.objects.BodyObject)  # with the body functions (trimmed_mesh)
    assert type(body.tanks[0]) is _classes.Tank
    assert body.interior_surface == "waterline"
    assert [type(b) for b in m.bodies] == [_classes.Body]
    body.tanks.clear()  # the whole mesh is its wetted surface
    t = body.trimmed_mesh("t")
    assert type(t) is _classes.Mesh


def test_the_properties_are_the_engine_items():
    m = _model()
    body = m.bodies["cyl"]
    body.mass = 250.0  # through the property: the engine's item
    assert body.mass == 250.0 and m.bodies["cyl"].info("mass").is_set
    with pytest.raises(ValueError, match="bodies/cyl: mass must be a finite number >= 0"):
        body.mass = -1.0
    body.tanks.clear()
    body.tanks.add(mesh="cyl", ranges=[(1, 3)], density_ratio=0.5)
    assert len(body.tanks) == 1 and body.tanks[0].density_ratio == 0.5
    assert m.environment.depth == math.inf and m.environment.domain == "open_sea"
    with pytest.raises(AttributeError, match="no item 'weight'"):
        body.weight = 1.0
    # the docstring of an item, as an environment shows it
    assert "displaced volume" in inspect.getdoc(type(body).mass) and "kg" in inspect.getdoc(type(body).mass)


def test_add_and_update_take_every_item_by_keyword():
    """``add`` of each list and ``update`` of each object have a keyword per own item and surface
    (not the other lists, whose objects are added to them), each described in the Args of the
    docstring (what an editor shows on a keyword of a call)."""
    schema = hb.engine.schema()
    for t, v in schema.items():
        if t == "Model":
            continue
        keys = [n for n in v["items"] if n != "name"]
        methods = [_classes.CLASSES[t].update] if keys else []
        if t in _classes.LISTS:
            methods.append(_classes.LISTS[t].add)
        for f in methods:
            sig = inspect.signature(f)
            params = {n: p for n, p in sig.parameters.items() if p.kind is p.KEYWORD_ONLY}
            assert list(params) == keys, (t, f.__name__)
            assert all(p.default is ... for p in params.values()), (t, f.__name__)
            doc = inspect.getdoc(f)
            assert all(f"\n    {k}: " in doc for k in keys), (t, f.__name__)


def test_the_lists_are_of_their_classes():
    m = _model()
    assert type(m.bodies) is _classes.BodyList and type(m.meshes) is _classes.MeshList
    assert type(m.lids) is _classes.DampingLidList and type(m.bodies["cyl"].tanks) is _classes.TankList
    assert type(m.fixed_dampers) is _classes.FixedDamperList
    assert isinstance(m.meshes, hb.objects.MeshListBase)  # with the functions that make meshes
    assert type(m.meshes["cyl"].patches) is hb.objects.ObjectList  # two types: the generic list


def test_the_keywords_given_are_set_the_others_kept():
    m = _model()
    body = m.bodies.add("b", mesh="cyl", mass=300.0)
    body.tanks.add(mesh="cyl", ranges=[(1, 1)])
    assert body.mass == 300.0 and len(body.tanks) == 1 and not body.info("yaw").is_set
    body.update(yaw=10.0)
    assert body.yaw == 10.0 and body.mass == 300.0 and len(body.tanks) == 1  # the others kept
    body.update(mass=None)  # None: back to the default
    assert not body.info("mass").is_set
    m.first_order.update(motion=True)
    m.first_order.update(radiation=False)
    assert m.first_order.motion and not m.first_order.radiation
    with pytest.raises(TypeError, match="unexpected keyword argument 'weight'"):
        m.bodies.add("c", mesh="cyl", weight=1.0)
    with pytest.raises(TypeError, match="unexpected keyword argument 'motion_'"):
        m.first_order.update(motion_=True)
    with pytest.raises(TypeError, match="unexpected keyword argument 'motion'"):
        m.options.update(motion=True)  # an output, not an option
