"""The views of the results generated from the engine's result tree: the generated module and the
table of the reference are up to date, every view of the tree is a documented property reached by
its path, and a view is the engine's selection (a slice where the rows are contiguous, else a
read-only copy of the rows; None for a body of a view of the results alone)."""

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import hydrobem as hb
from hydrobem import _core, _result_views

from bodies import lo_free_cylinder
from problems import load_script, model

ROOT = Path(__file__).resolve().parents[2]


# the generated files are those of the checkout: on an installed wheel (the checkout's package moved
# out of the path) there is none to compare
CHECKOUT = pytest.mark.skipif(bool(os.environ.get("HYDROBEM_TEST_INSTALLED")), reason="the checkout's files")


@CHECKOUT
def test_the_generated_files_are_up_to_date():
    gen = load_script("gen_result_views")
    assert (ROOT / "python" / "hydrobem" / "_result_views.py").read_text(encoding="utf-8") == gen.generate(), \
        "run scripts/gen_result_views.py"
    assert gen.DOC.read_text(encoding="utf-8") == gen.doc_text(), "run scripts/gen_result_views.py"


@CHECKOUT
def test_the_check_writes_nothing_and_compares_the_bytes(tmp_path, monkeypatch):
    # --check of a missing module writes no stub of it; a copy with CRLF line endings is not the
    # output (the files are written with LF)
    gen = load_script("gen_result_views")
    monkeypatch.setattr(sys, "argv", ["gen_result_views.py", "--check"])
    out, doc = tmp_path / "_result_views.py", tmp_path / "results.md"
    monkeypatch.setattr(gen, "OUT", out)
    assert gen.main() == 1 and not out.exists()
    out.write_bytes((ROOT / "python" / "hydrobem" / "_result_views.py").read_bytes())
    doc.write_bytes((ROOT / "docs" / "reference" / "results.md").read_bytes())
    monkeypatch.setattr(gen, "DOC", doc)
    assert gen.main() == 0
    out.write_bytes(out.read_bytes().replace(b"\n", b"\r\n"))
    assert gen.main() == 1


def _owner_class(path: str, start=hb.Results):
    """The class of the view that holds a path (from the results, or from ``start``), and the
    attribute of the path on it."""
    *groups, leaf = path.split(".")
    lists = {"pressure_surfaces": hb.PressureSurfaceResults, "tanks": hb.TankResults}
    if groups and groups[0] in lists:
        start, groups = lists[groups[0]], groups[1:]
    cls = start
    for g in groups:
        if g == "lids":
            cls = hb.LidResults
            continue
        prop = inspect.getattr_static(cls, g)
        cls = inspect.signature(prop.fget).return_annotation
        cls = getattr(_result_views, cls) if isinstance(cls, str) else cls
    return cls, leaf


def test_every_view_is_a_documented_property():
    # on the results, and on a body for the rules of a body's part
    for v in _core.result_views():
        alone = v["owner"] in ("global", "lid", "pressure_surface", "tank")
        starts = [hb.Results] if alone else [hb.Results, hb.BodyResults]
        for start in starts:
            cls, leaf = _owner_class(v["path"], start)
            prop = inspect.getattr_static(cls, leaf)
            assert isinstance(prop, property), (v["path"], start)
            assert v["quantity"] in inspect.getdoc(prop), v["path"]


def test_every_results_class_is_generated_and_the_tree_is_the_only_attribute_path():
    # the public classes of the results are those of the generated module, on their hand-written
    # bases; a quantity is an attribute only at the path of its view (``quantity`` reads any by name)
    for name in ("Results", "BodyResults", "BodyPointResults", "ExcitationResults", "FieldPointResults",
                 "LidResults", "PressureSurfaceResults", "TankResults"):  # fmt: skip
        assert getattr(hb, name).__module__ == "hydrobem._result_views", name
    paths = {v["quantity"]: v["path"] for v in _core.result_views()}
    groups = {g["path"] for g in _core.result_groups()}
    for q in hb.quantities():
        if paths.get(q) != q:
            # coordinates: the Coordinate tuple; the points of a group of points: the group's name
            assert not hasattr(hb.Results, q) or q == "coordinates" or q in groups, q


def test_a_view_is_the_engines_selection():
    r = model(lo_free_cylinder(), np.array([0.8, 1.4]), (0.0, 30.0), field_points=[[3.0, 0.0, 0.0]],
              options={"motion": True, "drift_pressure_integration": True, "drift_momentum_conservation": True,
                       "body_pressure": True}).solve()  # fmt: skip
    body = r.bodies[0]
    for v in _core.result_views():
        if v["owner"] in ("lid", "pressure_surface", "tank"):
            continue
        s = r._r.selection(v["path"], 0 if v["owner"] != "global" else None)
        a = r.quantity(v["quantity"])
        if a is None:
            assert s is None
            continue
        code, axes = s
        assert code == v["code"] and all(isinstance(rows, slice) for _, rows in axes), v["path"]  # contiguous
    assert body.drift.momentum_conservation is None
    assert r.drift.momentum_conservation is r.quantity("drift_momentum_conservation")
    assert body.drift.heading_pairs is r.drift.heading_pairs and np.shares_memory(body.motion, r.motion)
    assert r.omegas is r.quantity("omegas") and r.kochin.angles is r.quantity("kochin_angles")
    with pytest.raises(ValueError, match="no result view"):
        r._select("drift.pressure")
    with pytest.raises(ValueError, match="body 1 does not exist"):
        r._r.selection("added_mass", 1)
