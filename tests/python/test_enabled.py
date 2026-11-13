"""Suppressed objects: an object with ``enabled = False`` stays in the model with all its items,
and the solve takes the model as if it were absent (its children with it). The numeric files of a
model with a disabled object are those of the model without it, byte for byte."""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

import hydrobem as hb

from bodies import TANK_POINTS as POINTS
from bodies import TANK_ZT, tank_vessel
from problems import model, two_tank_vessels
from regression_problems import PROBLEMS


def files(results) -> dict:
    """The numeric files of results, without the line of the date."""
    with tempfile.TemporaryDirectory() as d:
        results.write_wamit(Path(d) / "x", overwrite=True)
        out = {}
        for p in sorted(Path(d).iterdir()):
            lines = p.read_bytes().split(b"\n")
            k = next((i for i, s in enumerate(lines) if b"Produced by HydroBEM on" in s), 0)
            out[p.name] = lines[:k] + lines[k + 1:]
        return out


def two_vessels():
    """Two barges with a tank each, a coupled to b, field points in both tanks and outside."""
    return two_tank_vessels({"couplings": [{"body": "b", "stiffness": np.eye(6) * 1e3, "damping": np.eye(6)}]})


def test_a_disabled_body_is_solved_as_removed_and_back():
    m = two_vessels()
    original = files(m.solve())
    m.bodies["b"].enabled = False
    assert m.bodies["b"].enabled is False and m.bodies.names() == ["a", "b"]
    r = m.solve()
    # its coupling and the field point in its tank go with it; the numbering is of the bodies solved
    assert r.bodies.names() == ["a"] and len(r.coordinates) == 6
    np.testing.assert_array_equal(r.field_points.points, POINTS[[2]])  # the sea's
    np.testing.assert_array_equal(r.bodies["a"].tanks["t"].points, POINTS[[0]])  # a's tank (at the origin)
    assert len(r.quantity("tanks")) == 1
    assert r.model.bodies.names() == ["a", "b"] and r.model.bodies["b"].enabled is False
    with pytest.raises(KeyError, match="'b' was disabled when the model was solved"):
        r.bodies["b"]
    with pytest.raises(KeyError, match="'b' was disabled"):
        r.coordinate("b", "heave")
    n = two_vessels()
    del n.bodies["a"].couplings[0]
    del n.bodies["b"]
    assert files(r) == files(n.solve())
    m.bodies["b"].enabled = True
    assert files(m.solve()) == original


def test_the_counts_and_queries_of_the_model_are_of_the_enabled_objects():
    m = two_vessels()
    m.bodies["b"].enabled = False
    c = m.counts()
    assert c.bodies == 1 and len(c.per_body) == 1 and (c.field_points, c.tank_points) == (1, 1)
    assert set(m.panels().body) <= {0, -1}
    assert m.motion_equation().stiffness.shape == (6, 6)
    assert list(m.classify_points(POINTS[[0]], m.bodies["a"].tanks["t"]).status) == ["tank"]
    assert list(m.classify_points(POINTS[[1]] - [6.0, 0.0, 0.0], m.bodies["b"].tanks["t"]).status) == ["no_tank"]


def test_the_queries_of_a_disabled_body_take_it_as_enabled():
    m = two_vessels()
    on = {"hydrostatics": m.bodies["b"].hydrostatics(), "panels": m.bodies["b"].panels(),
          "quality": m.bodies["b"].mesh_quality(), "tessellation": m.bodies["b"].tessellation()}  # fmt: skip
    m.bodies["b"].enabled = False
    np.testing.assert_array_equal(m.bodies["b"].hydrostatics().stiffness, on["hydrostatics"].stiffness)
    np.testing.assert_array_equal(m.bodies["b"].panels().centroids, on["panels"].centroids)
    np.testing.assert_array_equal(m.bodies["b"].mesh_quality().size, on["quality"].size)
    np.testing.assert_array_equal(m.bodies["b"].tessellation().vertices, on["tessellation"].vertices)
    assert m.bodies["b"].trimmed_mesh("b_wet").object_type == "Mesh"


def test_the_queries_of_a_body_leave_its_disabled_children_out():
    # a body's queries are its rows of the model's queries: its own disabled children (a tank) are
    # left out as the solve leaves them, the body enabled or itself disabled (taken as enabled)
    m = two_vessels()
    m.bodies["a"].tanks["t"].enabled = False
    rows = m.panels()
    keep = np.asarray(rows.body) == 0
    a = m.bodies["a"].panels()
    assert "tank" not in set(a.role) and "tank" not in set(np.asarray(rows.role)[keep])
    np.testing.assert_array_equal(a.centroids, np.asarray(rows.centroids)[keep])
    n = two_vessels()
    del n.bodies["a"].tanks["t"]
    for x, y in ((m.bodies["a"], n.bodies["a"]), (m.bodies["b"], n.bodies["b"])):
        np.testing.assert_array_equal(x.hydrostatics().stiffness, y.hydrostatics().stiffness)
        np.testing.assert_array_equal(x.mesh_quality().size, y.mesh_quality().size)
        np.testing.assert_array_equal(x.tessellation().vertices, y.tessellation().vertices)
    m.bodies["a"].enabled = False
    assert "tank" not in set(m.bodies["a"].panels().role)
    np.testing.assert_array_equal(m.bodies["a"].hydrostatics().stiffness, n.bodies["a"].hydrostatics().stiffness)
    np.testing.assert_array_equal(m.bodies["a"].panels().centroids, n.bodies["a"].panels().centroids)


def _problem(name):
    """A regression problem at its first frequency alone: the identity disabled = removed holds at
    every frequency, so one keeps the solves cheap."""
    m = PROBLEMS[name]()
    m.environment.omegas = m.environment.omegas[:1]
    return m


def _same_as_removed(name, disable, remove):
    """The files (or the error) of a problem with an object disabled and with it removed."""
    def outcome(change):
        m = _problem(name)
        change(m)
        with warnings.catch_warnings(record=True) as w:  # the same warnings, as the same files
            warnings.simplefilter("always")
            try:
                out = files(m.solve())
            except (ValueError, hb.UnsupportedProblemError) as e:
                out = str(e)
        return out, [str(x.message) for x in w]

    disabled, removed = outcome(disable), outcome(remove)
    assert disabled == removed
    return disabled[0]


@pytest.mark.parametrize("name, path", [
    ("lo_tank_haskind_pressure_drift", "bodies/body/tanks/1"),
    ("lo_plate_sheet", "bodies/body/dipoles/1"),
    ("lo_moonpool_lid", "lids/moon"),
    ("lo_acv_two_modes", "bodies/acv/pressure_surfaces/1"),
])  # fmt: skip
def test_a_disabled_object_is_solved_as_removed(name, path):
    def obj(m):
        o = m._doc.object(path)
        return o

    def disable(m):
        obj(m).set("enabled", False)

    def remove(m):
        obj(m).remove()

    out = _same_as_removed(name, disable, remove)
    assert isinstance(out, dict) and out != files(_problem(name).solve())


def test_a_disabled_damper_and_coupling_are_solved_as_removed():
    def with_damper(m):
        b = m.bodies[0]
        sheet = b.dipoles[0].mesh
        del b.dipoles[0]
        b.dipole_dampers.add(mesh=sheet, damping_factor=0.5)
        return b

    def disable(m):
        with_damper(m).dipole_dampers[0].enabled = False

    def remove(m):
        del with_damper(m).dipole_dampers[0]

    _same_as_removed("lo_plate_sheet", disable, remove)

    def coupled(m):
        m.bodies["cyl"].couplings.add(body="sph", stiffness=np.eye(6) * 50.0)
        return m.bodies["cyl"].couplings[0]

    def uncoupled(m):
        coupled(m)
        del m.bodies["cyl"].couplings[0]

    out = _same_as_removed("lo_two_bodies", lambda m: setattr(coupled(m), "enabled", False), uncoupled)
    m = _problem("lo_two_bodies")
    coupled(m)
    assert files(m.solve()) != out


def test_disabled_tanks_leave_their_field_points_out():
    m = two_vessels()
    m.bodies["a"].tanks["t"].enabled = False
    r = m.solve()
    np.testing.assert_array_equal(r.field_points.points, POINTS[[2]])  # the sea's
    assert len(r.bodies["a"].tanks) == 0 and r.bodies["b"].tanks["t"].points.shape == (1, 3)  # b's alone
    # no field point left: no field outputs
    m.bodies["b"].tanks["t"].enabled = False
    m.first_order.field_points = None
    r = m.solve()
    assert r.field_points.pressure_radiation is None and r.field_points.points is None
    assert r.quantity("tanks") is None and r.quantity("tanks_pressure_radiation") is None


def test_every_body_disabled_is_refused():
    m = two_vessels()
    for b in m.bodies:
        b.enabled = False
    with pytest.raises(ValueError, match="every body is disabled"):
        m.check()
    # the queries of a body still answer
    assert m.bodies["a"].hydrostatics().volume > 0


def test_check_warns_of_a_disabled_object_that_would_fail():
    m = two_vessels()
    m.bodies.add("broken", mesh=m.bodies["b"].mesh, position=[0.0, 8.0, 0.0], radii_of_gyration=[1.0, 1.0, 1.0],
                 enabled=False)  # fmt: skip
    m.bodies["a"].tanks["t"].enabled = False  # fits the model: no warning
    w = m.check()
    assert [x.message for x in w if "is disabled" in x.message] == [
        "bodies/broken is disabled; enabled, it would fail: bodies/broken: inertia_origin='center_of_gravity' "
        "requires center_of_gravity"]
    m.solve()  # the model solved has no broken body


def test_a_message_names_an_unnamed_object_by_its_place_in_the_model():
    b = tank_vessel()
    tank = b["tanks"][0]
    # the first tank disabled, the second (the same walls) with its surface level below their top
    b["tanks"] = [{**tank, "enabled": False}, {**tank, "surface_level": TANK_ZT - 0.3}]
    m = model([{**b, "name": "a"}], [1.2], [0.0])
    with pytest.raises(ValueError, match=r"bodies/a: tank 2: a tank element lies above its surface level"):
        m.check()


def test_a_lid_placed_with_a_disabled_body_is_refused():
    m = PROBLEMS["ho_moonpool_lid"]()
    m.bodies.add("other", mesh=m.bodies["b"].mesh, position=[10.0, 0.0, 0.0])
    m.bodies["b"].enabled = False
    with pytest.raises(ValueError, match="lids/l: placed_with: the body 'b' is disabled"):
        m.check()
    m.lids["l"].enabled = False
    m.check()


def test_the_model_file_keeps_enabled(tmp_path):
    m = two_vessels()
    m.bodies["b"].enabled = False
    m.bodies["a"].couplings[0].enabled = False
    m.save(tmp_path / "m.hbm")
    n = hb.Model.load(tmp_path / "m.hbm")
    assert n.bodies["b"].enabled is False and n.bodies["a"].couplings[0].enabled is False
    assert n.bodies["a"].enabled is True and str(n) == str(m)
    # the results file records the enabled state of its solve
    m.solve()
    m.save_results(tmp_path / "m.hbr")
    r = hb.load_results(tmp_path / "m.hbr")
    assert r.bodies.names() == ["a"] and r.model.bodies["b"].enabled is False
