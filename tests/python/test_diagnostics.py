"""Diagnostics: engine errors name where they occurred (body, mesh, element,
field point, lid, frequency) in ``exc.location``; warnings of the rules reach Python as
``HydroBEMWarning`` with a code and location and are listed in
``Results.warnings``."""

import warnings

import numpy as np
import pytest

import hydrobem as hb
from hydrobem import _core

from bodies import acv_body, part, surface
from meshes import box_panels
from problems import add_mesh, flat_surface, model

MESH = surface(box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True), symmetry="xz_and_yz")
N = len(MESH["vertices"])


def _overlapping_roles():
    # element 2 of the mesh both a dipole and the interior surface of body b
    a = {"name": "a", "mesh": MESH}
    b = {"name": "b", "mesh": part(MESH, 4, N), "position": [10.0, 0.0, 0.0], "dipoles": [{"mesh": part(MESH, 1, 2)}],
         "interior_surface": part(MESH, 2, 3)}  # fmt: skip
    return model([a, b], [1.0], [0.0])


def test_errors_carry_the_body_and_record():
    with pytest.raises(ValueError, match="bodies/b: element 2 of the mesh 'a' is taken by both dipoles 1 and interior"):
        _overlapping_roles().route()
    # a rule of one body without an element
    a = {"name": "a", "mesh": MESH}
    c = {"name": "c", "mesh": MESH, "position": [10.0, 0.0, 0.0], "excitation_method": "haskind"}
    with pytest.raises(ValueError, match="haskind") as e:
        model([a, c], [1.0], [0.0], options={"radiation": False}).route()
    assert e.value.location == hb.Location(body=1)


def test_one_role_per_record_of_another_mesh():
    """An element of a mesh other than the body's has one role in the body as well: the interior
    surface, the dipoles and the pressure surfaces that take elements of one other mesh."""
    m = model({"name": "a", "mesh": MESH}, [1.0], [0.0])
    add_mesh(m, "plate", surface(box_panels(0.5, 0.5, 0.2, n=(2, 2, 1)) + [0.0, 0.0, -2.0]))
    b = m.bodies["a"]
    b.dipoles.add(mesh="plate", ranges=[(1, 2)])
    b.update(**flat_surface("interior_surface", {"method": "mesh", "mesh": "plate", "ranges": [(2, 3)]}))
    with pytest.raises(ValueError, match="bodies/a: element 2 of the mesh 'plate' is taken by both dipoles 1 and "
                                         "interior_surface") as e:  # fmt: skip
        m.route()
    assert e.value.location == hb.Location(body=0)
    b.interior_surface = "none"
    b.pressure_surfaces.add(mesh="plate").pressure_modes.add("p", shape="uniform")
    with pytest.raises(ValueError, match="element 1 of the mesh 'plate' is taken by both dipoles 1 and pressure"):
        m.route()


def test_overlapping_roles_carry_the_body_and_record():
    with pytest.raises(ValueError, match="taken by both") as e:
        _overlapping_roles().route()
    assert e.value.location == hb.Location(body=1, element=2)


def test_mesh_reference_rules_carry_their_location():
    a = {"name": "a", "mesh": MESH}
    # a range beyond the mesh: the body
    b = {"name": "b", "mesh": part(MESH, 1, N + 1), "position": [10.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="beyond the mesh") as e:
        model([a, b], [1.0], [0.0]).route()
    assert e.value.location == hb.Location(body=1)
    # an element of a body taken by a lid placed with it: the body, the element and the lid
    lid = {"name": "lid", "mesh": part(MESH, 1, 1), "placed_with": "a"}
    with pytest.raises(ValueError, match="element 1 of the mesh 'a' is taken by both") as e:
        model(a, [1.0], [0.0], lids=[lid]).route()
    assert e.value.location == hb.Location(body=0, element=1, lid=0)
    # a lid from the elements of a trimmed body: the lid and the body
    with pytest.raises(ValueError, match="a lid from the elements of the trimmed body") as e:
        model({**a, "trim": [0.0, 0.0, 0.0]}, [1.0], [0.0], lids=[lid]).route()
    assert e.value.location == hb.Location(body=0, lid=0)
    # the trim of a body: the body
    c = {"name": "c", "mesh": MESH, "position": [10.0, 0.0, 0.0], "trim": [5.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="bodies/c: trim: ") as e:
        model([a, c], [1.0], [0.0]).route()
    assert e.value.location == hb.Location(body=1)
    # the rules of a lid: the lid
    plate = surface(np.array([[[3.0, 0.0, -0.5], [4.0, 0.0, -0.5], [4.0, 1.0, -0.5], [3.0, 1.0, -0.5]]]))
    lids = [{"name": "l0", "mesh": surface(np.array([[[3.0, 2.0, 0.0], [4.0, 2.0, 0.0], [4.0, 3.0, 0.0],
                                                      [3.0, 3.0, 0.0]]]))}, {"name": "l1", "mesh": plate}]  # fmt: skip
    with pytest.raises(ValueError, match="lids/l1") as e:
        model(a, [1.0], [0.0], lids=lids).route()
    assert e.value.location == hb.Location(lid=1)


def test_errors_carry_the_field_point():
    pts = [[5.0, 0.0, 0.0], [0.5, 0.2, 0.0]]
    m = model({"name": "a", "mesh": MESH}, [1.0], [0.0], field_points=pts, options={"field_pressure": True})
    with pytest.raises(ValueError, match="field point 2 lies inside the water plane") as e:
        m.route()
    assert e.value.location == hb.Location(body=0, field_point=1)


def test_warnings_with_their_body():
    acv = acv_body(mass="displaced", center_of_gravity=[0.0, 0.0, -0.5],
                   radii_of_gyration=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    m = model([{"name": "box", "mesh": MESH, "position": [12.0, 0.0, 0.0], "modes": []}, acv], [1.0], [0.0],
              options={"motion": True, "drift_pressure_integration": True})  # fmt: skip
    rec = m.check()
    assert len(rec) == 1 and rec[0].location == hb.Location(body=1) and rec[0].code == 1
    assert "wetted surface only" in rec[0].message
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = m.solve()
    assert len([w for w in rec if issubclass(w.category, hb.HydroBEMWarning)]) == 1  # once per solve
    (d,) = res.warnings
    assert d.code == 1 and d.location == hb.Location(body=1) and d.severity == "warning"
    # no warning without the combination
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert model(acv, [1.0], [0.0]).solve().warnings == ()


def test_mesh_records_of_a_mesh_object():
    v = box_panels(1.0, 1.0, 1.0, n=(2, 2, 2))
    v[3] = v[3][[0, 0, 0, 0]]  # a degenerate element (Python's own checks are not passed)
    doc = _core.Model()
    mesh = doc.object().add("meshes", "box")
    with pytest.raises(ValueError, match="meshes/box: mesh element 4") as e:
        mesh.set("vertices", v)
    assert e.value.location == hb.Location(element=4)
    doc.close()
