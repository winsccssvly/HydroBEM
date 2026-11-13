"""The extension module hydrobem._core (PyO3): the engine's errors as the package's exceptions with
their messages and locations, the checks of a value where it is set, the cancel request before a
solve, and the solve with the GIL released: the model is read from another thread while it
calculates and a cancel from another thread stops it (the refusals of changes and queries while
it calculates are in test_progress.py). A model is given as objects."""

import threading

import pytest

import hydrobem as hb
from hydrobem import _core

from bodies import slow_cylinder, surface
from meshes import box_panels
from problems import model


def _valid():
    """A model of a box in the open sea, one frequency and one heading, given as objects."""
    doc = _core.Model()
    root = doc.object()
    root.add("meshes", "box").set("vertices", box_panels())
    root.add("bodies", "b").set("mesh", "box")
    doc.object("environment").set("omegas", [1.0])
    doc.object("environment").set("headings", [0.0])
    return doc


def test_a_mesh_error_names_the_mesh_and_record():
    doc = _valid()
    bad = box_panels().copy()
    bad[4] = 0.0
    with pytest.raises(ValueError, match=r"^meshes/box: mesh element 5: fewer than three distinct vertices$") as e:
        doc.object("meshes/box").set("vertices", bad)
    assert e.value.location == hb.Location(element=5)


def test_values_are_checked_where_they_are_set():
    doc = _valid()
    with pytest.raises(ValueError, match=r"^environment: depth must be > 0 or inf, got -2\.0$"):
        doc.object("environment").set("depth", -2.0)
    with pytest.raises(ValueError, match="^environment: domain must be one of"):
        doc.object("environment").set("domain", "lake")
    with pytest.raises(ValueError, match="frequency_threads"):
        doc.object("options").set("frequency_threads", 1025)
    # text kept in a model is UTF-8: a string with a lone surrogate is refused
    label = doc.object("meshes/box")
    with pytest.raises(ValueError):
        label.set("label", "caf\udce9")
    label.set("label", "café ✓")
    assert label.get("label") == "café ✓"
    assert doc.route() == "low_order_infinite_open_sea"  # the model is unchanged by the failures
    assert hb.Model(doc).counts().radiation_mode_ids == (1, 2, 3, 4, 5, 6)


def test_the_errors_are_the_exceptions_of_their_kind(tmp_path):
    doc = _valid()
    with pytest.raises(ValueError, match="no object 'bodies/x'"):  # an argument
        doc.object("bodies/x")
    with pytest.raises(TypeError, match="^first_order: motion must be a bool"):
        doc.object("first_order").set("motion", 1)
    with pytest.raises(OSError, match=r"missing\.hbm"):
        _core.Model.load(tmp_path / "missing.hbm")
    doc.save(tmp_path / "a.hbm")
    with pytest.raises(FileExistsError, match=r"exists \(pass overwrite=True\)"):
        doc.save(tmp_path / "a.hbm")
    with pytest.raises(RuntimeError, match="no results"):  # a state
        doc.results()


def test_cancel_before_the_solve_applies_to_it():
    # a cancel sent while no solve runs cancels the next one; the request is cleared when the solve
    # returns, so the solve after it runs
    m = hb.Model(_valid())
    m.cancel()
    with pytest.raises(hb.SolveCancelledError):
        m.solve()
    assert m.state == "reset"
    with pytest.raises(RuntimeError, match="no results"):
        _ = m.results
    r = m.solve()
    assert m.state == "calculated" and r.route == "low_order_infinite_open_sea"
    # a solve that fails before it starts clears the cancel too
    m.cancel()
    m.bodies["b"].mesh = None
    with pytest.raises(ValueError):
        m.solve()
    m.bodies["b"].mesh = "box"
    assert m.solve().added_mass is not None


def test_the_motion_equation_needs_mass_properties():
    e = model({"mesh": surface(box_panels())}, [1.0], [0.0]).motion_equation()
    assert not e.ready and e.inertia is None and e.stiffness is None and e.damping is None
    inertia, stiffness, damping, ready = _valid().motion_equation()
    assert not ready and inertia.shape == stiffness.shape == damping.shape == (6, 6)
    assert not stiffness.flags.writeable


def test_an_empty_model_names_what_it_lacks():
    with pytest.raises(ValueError, match="the model has no bodies"):
        _core.Model().check()


def test_quantities_are_read_only_views_of_the_results():
    r = hb.Model(_valid()).solve()
    a = r.added_mass
    assert not a.flags.writeable and not a.flags.owndata
    with pytest.raises(ValueError):
        a[0, 0, 0] = 1.0
    assert r.added_mass is a  # read once


def _slow_model():
    """2048 panels without symmetry on one inner thread: only so that the solve lasts seconds and this
    thread runs while the engine computes (a callback that blocks would release the GIL itself and
    prove nothing); the cancel ends it after the first report, so the test stays cheap."""
    return model({"mesh": slow_cylinder()}, [0.5, 1.0, 1.5, 2.0], [0.0], options={"inner_threads": 1})


def test_the_solve_releases_the_gil():
    # the solve runs on another thread while this one reads the model and cancels it
    m = _slow_model()
    started = threading.Event()
    outcome = {}

    def report(p):
        started.set()

    def run():
        try:
            m.solve(report)
            outcome["done"] = True
        except BaseException as e:  # noqa: B036 (checked below)
            outcome["error"] = e

    t = threading.Thread(target=run)
    t.start()
    assert started.wait(60)
    assert m.state == "calculating"
    assert m.bodies["body"].mesh is not None  # the items are read meanwhile
    m.cancel()
    t.join(120)
    assert not t.is_alive()
    assert isinstance(outcome.get("error"), hb.SolveCancelledError), outcome
    assert m.state == "reset"
    m.environment.depth = 30.0  # changes again once the solve has ended
    m.close()
