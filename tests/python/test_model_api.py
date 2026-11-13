"""The Python API on the engine model: hb.Model with its objects and lists,
objects set from dicts of their items, the queries of the model, and hb.Results on the engine's
results (the .hbr file, the model and the meta data of a solve)."""

import math

import numpy as np
import pytest

import hydrobem as hb
from hydrobem.objects import ModelObject

from bodies import coarse_cylinder
from problems import add_mesh, flat_surface
from regression_problems import PROBLEMS, same_outputs


def _model():
    m = hb.Model()
    add_mesh(m, "cyl", coarse_cylinder())
    m.bodies.add("cyl", mesh="cyl", mass="displaced", center_of_gravity=[0.0, 0.0, -0.1],
                 radii_of_gyration=[0.4, 0.4, 0.4])
    m.environment.omegas = [1.0, 2.0]
    m.environment.headings = [0.0]
    m.first_order.motion = True
    return m


def test_attributes_are_the_engine_items():
    m = _model()
    body = m.bodies["cyl"]
    assert body.mass == "displaced" and body.mesh == "cyl"  # one of its names
    body.mass = 250.0
    assert m.bodies["cyl"].mass == 250.0 and body.info("mass").is_set
    body.reset("mass")
    assert body.mass is None  # unset: no mass (it has no default)
    body.mass = "displaced"
    assert body.position.tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="bodies/cyl: mass must be a finite number >= 0"):
        body.mass = -1.0
    with pytest.raises(AttributeError, match="a Body has no item"):
        body.weight = 3.0
    with pytest.raises(AttributeError, match="no item or list 'weight'"):
        _ = body.weight
    assert "mass" in dir(body) and "tanks" in dir(body) and "update" in dir(body)
    body.update(mass=300.0, yaw=10.0)
    assert (body.mass, body.yaw) == (300.0, 10.0)
    assert m.environment.depth == math.inf
    m.environment.depth = 40.0
    assert m.environment.depth == 40.0
    assert m.second_order.drift_momentum_conservation_angles == "auto"  # an item of the model only
    assert repr(body) == "<Body bodies/cyl>"


def test_lists_and_single_children(tmp_path):
    m = _model()
    m.bodies.add("other", mesh="cyl", position=[5.0, 0.0, 0.0])
    m.bodies.insert(0, "first", mesh="cyl", position=[-5.0, 0.0, 0.0])
    assert m.bodies.names() == ["first", "cyl", "other"] and len(m.bodies) == 3 and "other" in m.bodies
    m.bodies.rename("other", "third")
    del m.bodies["first"]
    assert [b.name for b in m.bodies] == ["cyl", "third"]
    with pytest.raises(ValueError, match="duplicate body name 'cyl'"):
        m.bodies.add("cyl", mesh="cyl")
    assert m.bodies.names() == ["cyl", "third"]  # a failed add leaves nothing behind
    body = m.bodies["cyl"]
    assert body.interior_surface == "detect"  # an item of the body, its method
    body.interior_surface = "waterline"
    assert body.interior_surface == "waterline"
    body.interior_surface = "none"
    assert body.interior_surface == "none"
    body.interior_surface = None  # its default
    assert body.interior_surface == "detect"
    tank = body.tanks.add(mesh="cyl", ranges=[(1, 4)], density_ratio=0.5)
    assert body.tanks[0] == tank and tank.path == "bodies/cyl/tanks/1"
    gm = body.generalized_modes.add("bend", parity="heave")
    gm.pieces.add(w=[[0.4, 2, 0, 0]])
    assert gm.pieces[0].w.tolist() == [[0.4, 2.0, 0.0, 0.0]]
    # a list is not assigned: its objects are added, one by one, after their parent
    with pytest.raises(TypeError, match="tanks is not assigned"):
        body.tanks = [{"mesh": "cyl", "ranges": [(1, 2)]}]
    twist = body.generalized_modes.add("twist")
    twist.pieces.add(u=[[1.0, 0, 1, 0]])
    assert body.generalized_modes.names() == ["bend", "twist"] and twist.pieces[0].u.shape == (1, 4)
    body.tanks.clear()
    assert len(body.tanks) == 0
    free = hb.Model()  # meshes no body uses
    for n in ("b", "c"):
        add_mesh(free, n, m.meshes["cyl"].items())  # items() copies the mesh
    assert free.meshes.names() == ["b", "c"]
    with pytest.raises(ValueError):
        free.meshes.add("bad", kind="panels", vertices=[[1.0]])
    assert free.meshes.names() == ["b", "c"]  # a failed add leaves nothing behind
    other = m.bodies.add("other", mesh="cyl", **flat_surface("interior_surface", {"method": "waterline"}))
    other.tanks.add(mesh="cyl", ranges=[(1, 3)])
    assert m.bodies["other"].interior_surface == "waterline" and len(m.bodies["other"].tanks) == 1
    # a mesh read from a file by the engine
    m.meshes["cyl"].write(tmp_path / "c.gdf")
    mesh = m.meshes.load("file", tmp_path / "c.gdf")
    assert mesh.kind == "panels" and mesh.vertices.tobytes() == m.meshes["cyl"].vertices.tobytes()


def test_names_and_quoted_keys(tmp_path):
    # a name is a part of an object path: '/' and control characters are refused
    m = _model()
    for bad in ("a/b", "a\nb", "tab\there"):
        with pytest.raises(ValueError, match="has a '/' or a control character"):
            m.bodies.add(bad, mesh="cyl")
        with pytest.raises(ValueError, match="has a '/' or a control character"):
            m.meshes.rename("cyl", bad)
    # the text of a tool, control characters and all, is kept as given
    m.tool.add("t", text="a\tb = 1\nc\x01 = 2\n")
    m.save(tmp_path / "case.hbm")
    assert hb.Model.load(tmp_path / "case.hbm").tool["t"].text == "a\tb = 1\nc\x01 = 2\n"


def test_save_load_copy(tmp_path):
    m = _model()
    m.save(tmp_path / "case.hbm")  # one file, the meshes inside
    assert sorted(f.name for f in tmp_path.iterdir()) == ["case.hbm"]
    with pytest.raises(OSError, match="exists"):
        m.save(tmp_path / "case.hbm")
    n = hb.Model.load(tmp_path / "case.hbm")
    assert n.bodies["cyl"].radii_of_gyration.tolist() == [0.4, 0.4, 0.4]
    c = n.copy()
    c.bodies["cyl"].mass = 9.0
    assert n.bodies["cyl"].mass == "displaced"


def test_check_route_and_queries():
    m = _model()
    assert m.check() == []
    assert m.route() == "low_order_infinite_open_sea"
    assert (m.bodies["cyl"].hydrostatics().volumes == m.bodies[0].hydrostatics().volumes).all()
    assert (m.bodies[0].hydrostatics().volumes == m.meshes["cyl"].geometry().volumes).all()
    c = m.counts()
    assert (c.bodies, c.ordinary_frequencies, c.headings, c.coordinates) == (1, 2, 1, 6)
    assert m.panels().areas.size == c.per_body[0].wetted_with_reflections and m.memory_estimate().peak > 0
    for gone in ("hydrostatics", "interior_mesh"):  # the queries of a body are the body's
        assert not hasattr(m, gone)
    m.bodies["cyl"].update(inertia_input="components", Ixx=1.0, Ixy=2.0)  # the others 0.0, their default
    with pytest.raises(ValueError, match="the inertia of the free coordinates is not positive semi-definite"):
        m.check()


def test_model_solve_and_the_solve_of_its_copy():
    m = _model()
    same_outputs(m.solve(), m.copy().solve())


def test_results_file_and_model_of_the_results(tmp_path):
    # the API of a results file: hb.load_results gives hb.Results with its hb.Model and hb.Meta (every
    # quantity written and read back bit for bit is the IO rule of test_results_file.py)
    p = _model()
    r = p.solve()
    r.save(tmp_path / "run.hbr")
    with pytest.raises(FileExistsError, match=r"run\.hbr exists"):
        r.save(tmp_path / "run.hbr")
    q = hb.load_results(tmp_path / "run.hbr")
    assert isinstance(q, hb.Results) and q.route == r.route
    assert r.meta.route == r.route and r.meta.api_version == hb.build_info().api_version and isinstance(r.meta, hb.Meta)
    assert r.meta.frequency_threads >= 1 and r.meta.solved_at is not None
    assert r.meta.environment_hooks == ()  # no hook of the environment set in the tests
    assert isinstance(q.model, hb.Model) and q.model.bodies.names() == p.bodies.names()
    assert str(q.model) == str(p)  # the model solved, every item given
    # change and solve again from the model of a results file
    m = q.model
    m.environment.density = 1025.0
    assert m.solve().hydrostatics.mass[0] != r.hydrostatics.mass[0]  # the displaced mass follows the density


def test_drift_momentum_conservation_angles_item():
    m = PROBLEMS["lo_cylinder_drift"]()
    n = m.solve().kochin.angles.size
    m.second_order.drift_momentum_conservation_angles = 2 * n
    assert m.solve().kochin.angles.size == 2 * n


def test_reset_clear_and_close():
    # reset keeps the inputs, clear empties the model, close frees it
    with _model() as m:
        m.solve()
        assert m.state == "calculated"
        m.reset()
        assert m.state == "reset" and m.bodies.names() == ["cyl"]
        with pytest.raises(RuntimeError, match="no results"):
            _ = m.results
        m.solve()
        m.clear()
        assert m.state == "reset" and m.bodies.names() == [] and m.environment.depth == math.inf


@pytest.mark.filterwarnings("ignore:.*integrated over the wetted surface only")  # the note of the problem
def test_public_shapes_of_haskind_and_the_drifts():
    # the engine gives Haskind as [F, H, K] and the drifts as force-and-moment and moving-origin moment
    r = PROBLEMS["ho_acv_pressure_drift"]().solve()
    F, H, K = len(r.omegas), len(r.headings), len(r.radiation_mode_ids)
    assert r.excitation.haskind.shape == (F, H, K)
    nb = len(r.model.bodies)
    P = len(r.drift.heading_pairs)  # the pairs (h, h) of the unidirectional drifts
    assert P == H and r.drift.heading_pairs.shape == (P, 2)
    # on the coordinates: the rigid rows of each body, NaN on its pressure coordinates
    D = len(r.coordinates)
    rigid = [c.kind == "rigid" for c in r.coordinates]
    assert D > 6 * nb and r.drift.pressure_integration.shape == (F, P, D)
    assert np.all(np.isfinite(r.drift.pressure_integration[..., rigid]))
    assert np.all(np.isnan(r.drift.pressure_integration[..., np.logical_not(rigid)]))
    assert r.drift.pressure_integration_body_axes.shape == (F, P, 3 * nb)


def test_a_referenced_object_is_not_removed():
    # a mesh a body uses, a body a lid uses, stay until the reference changes
    m = _model()
    with pytest.raises(ValueError, match="meshes/cyl: is used by bodies/cyl"):
        del m.meshes["cyl"]
    m.lids.add("lid", mesh="cyl", ranges=[(1, 2)], placed_with="cyl")
    with pytest.raises(ValueError, match="bodies/cyl: is used by lids/lid"):
        del m.bodies["cyl"]
    m.meshes.rename("cyl", "c")  # a rename changes the references, the ranges kept
    assert m.bodies["cyl"].mesh == "c" and m.lids["lid"].mesh == "c" and m.lids["lid"].ranges.tolist() == [[1, 2]]
    del m.lids["lid"]
    del m.bodies["cyl"]
    del m.meshes["c"]
    assert m.meshes.names() == []


def test_errors_files_and_folders(tmp_path):
    # an argument of the engine is a ValueError, an existing file a FileExistsError, and the
    # writers make the folder of their files
    m = _model()
    with pytest.raises(ValueError, match="no object 'bodies/nobody'"):
        m.meshes.load_csf("cs", tmp_path / "cs.csf", body="nobody")
    with pytest.raises(ValueError, match="role must be one of wetted"):
        m.bodies["cyl"].panels(role="hull")
    m.save(tmp_path / "new" / "case.hbm")
    assert (tmp_path / "new" / "case.hbm").is_file()
    with pytest.raises(FileExistsError, match=r"exists \(pass overwrite=True\)"):
        m.save(tmp_path / "new" / "case.hbm")
    r = m.solve()
    m.save_results(tmp_path / "a" / "run.hbr")
    r.save(tmp_path / "b" / "run.hbr")
    with pytest.raises(FileExistsError, match=r"run\.hbr exists"):
        r.save(tmp_path / "b" / "run.hbr")
    r.write_wamit(tmp_path / "c" / "cyl")
    assert (tmp_path / "c" / "cyl.out").is_file()
    m.close()
    with pytest.raises(RuntimeError, match="the model is closed"):
        m.bodies.names()


def test_item_info_from_the_schema():
    m = _model()
    info = m.bodies["cyl"].info("mass")
    assert info.unit == "kg" and info.kind == "real" and info.is_set and info.description
    assert info.choices == ("displaced",) and isinstance(info, hb.ItemInfo)
    assert m.environment.info("domain").choices[0] == "open_sea"
    with pytest.raises(AttributeError):  # read-only
        info.unit = "g"
    with pytest.raises(AttributeError, match="no item 'weight'"):
        m.bodies["cyl"].info("weight")
    # one way to open a results file: hb.load_results (hb.Model.load_results for the model with it)
    assert not hasattr(hb.Results, "open") and not hasattr(hb.Results, "from_engine")


def test_objects_taken_out_of_their_model_are_refused():
    # a removed object, the objects of a cleared model and of a closed one raise rather than
    # read freed memory
    m = _model()
    m.bodies.add("other", mesh="cyl", position=[5.0, 0.0, 0.0])
    b, outs = m.bodies["other"], m.first_order
    del m.bodies["other"]
    with pytest.raises(ValueError, match="the object was removed"):
        b.mass = 5.0
    with pytest.raises(ValueError, match="the object was removed"):
        _ = b.mass
    body = m.bodies["cyl"]
    m.clear()
    for get in (lambda: outs.motion, lambda: body.mesh):
        with pytest.raises(ValueError, match="the object was removed"):
            get()
    m2 = _model()
    o = m2.first_order
    m2.close()
    with pytest.raises(RuntimeError, match="closed"):
        o.motion = True
    # a body of another model is not a body of this one
    m3, m4 = _model(), _model()
    with pytest.raises(ValueError, match="a Body of this model"):
        m3.meshes.load_csf("cs", "cs.csf", body=m4.bodies["cyl"])


def test_the_surfaces_are_items_chosen_by_their_method():
    # the items of a method not chosen are kept unused and not checked: nothing is refused or reset
    # when one is set; a refused value keeps the old one
    m = _model()
    body = m.bodies["cyl"]
    body.interior_surface = "waterline"
    with pytest.raises(ValueError, match="bodies/cyl: interior_surface must be one of none, detect, waterline"):
        body.interior_surface = "bogus"
    assert body.interior_surface == "waterline"
    with pytest.raises(ValueError, match="interior_surface_mesh: unknown mesh 'nomesh'"):
        body.update(interior_surface="mesh", interior_surface_mesh="nomesh")
    assert body.interior_surface == "waterline" and body.interior_surface_mesh is None
    body.interior_surface_mesh = "cyl"  # kept, unused by "waterline"
    m.check()
    body.update(control_surface="cylinder", control_surface_radius=2.0, control_surface_depth=1.0,
                control_surface_inner_radius=1.2, control_surface_panel_size=0.5)  # fmt: skip
    body.control_surface_mesh = "cyl"  # an item of another method: kept, unused
    body.control_surface = "mesh"
    assert (body.control_surface_radius, body.control_surface_mesh) == (2.0, "cyl")
    body.control_surface = "none"
    assert body.control_surface_radius == 2.0


def test_a_failed_update_changes_nothing():
    # update sets its items all or nothing
    m = _model()
    with pytest.raises(ValueError, match="inner_threads"):
        m.options.update(higher_order=True, inner_threads=-1)
    assert m.options.higher_order is False and m.options.items() == hb.Model().options.items()
    m.environment.update(frequency_input="periods", periods=[2.0, 4.0], omegas=None)
    with pytest.raises(TypeError):
        m.environment.update(omegas=[1.0], frequency_input="omegas", headings="x")
    assert list(m.environment.periods) == [2.0, 4.0] and len(m.environment.omegas) == 0
    assert m.environment.frequency_input == "periods"
    body = m.bodies["cyl"]
    with pytest.raises(ValueError, match="unknown mesh 'nomesh'"):
        body.update(yaw=10.0, mesh="nomesh")
    assert body.yaw == 0.0 and body.mesh == "cyl"
    with pytest.raises(ValueError, match="unknown mesh 'nomesh'"):
        body.update(yaw=10.0, control_surface="mesh", control_surface_mesh="nomesh")
    assert body.yaw == 0.0 and body.control_surface == "none"
    with pytest.raises(AttributeError, match="no item 'bogus'"):
        ModelObject.update(body, yaw=5.0, bogus=1)
    assert body.yaw == 0.0


def test_the_model_and_its_objects_as_python_objects(tmp_path):
    # a model has only its objects and lists as attributes; the objects are views of it
    import copy
    import pickle

    m = _model()
    with pytest.raises(AttributeError, match="no 'optionz'"):
        m.optionz = 3
    with pytest.raises(AttributeError, match="always there"):
        m.options = {}
    m.bodies.clear()
    m.bodies.add("a", mesh="cyl")
    m.bodies.add("b", mesh="cyl", position=[5.0, 0.0, 0.0])
    assert m.bodies.names() == ["a", "b"] and [b.name for b in m.bodies[0:2]] == ["a", "b"]
    with pytest.raises(TypeError, match="indexed by name or position"):
        _ = m.bodies[1.5]
    c = copy.copy(m)
    assert isinstance(c, hb.Model) and c.bodies.names() == ["a", "b"] and c._doc is not m._doc
    for obj in (m.bodies["a"], m.bodies, m):
        with pytest.raises(TypeError, match="pickled"):
            pickle.dumps(obj)
    with pytest.raises(TypeError, match="not copied"):
        copy.deepcopy(m.bodies["a"])
    # the mesh writer: the folder made, an existing file kept unless overwrite
    path = tmp_path / "new" / "cyl.gdf"
    m.meshes["cyl"].write(path)
    assert path.exists()
    with pytest.raises(FileExistsError, match="overwrite=True"):
        m.meshes["cyl"].write(path)
    m.meshes["cyl"].write(path, overwrite=True)


def test_constants_are_the_engines():
    # one default gravity; the text selections are not part of the API (a mesh reference takes ranges)
    from hydrobem.queries import STANDARD_GRAVITY

    for gone in ("normalize_panels", "format_panels", "PanelRange"):
        assert not hasattr(hb, gone)
    assert STANDARD_GRAVITY == float(hb.engine.schema()["Environment"]["items"]["gravity"]["default"])


def test_print_shows_what_is_given():
    # print(obj) and print(m) show the items given (not their defaults), the children and
    # long arrays by their shape
    m = _model()
    m.bodies["cyl"].tanks.add(mesh="cyl", ranges=[(1, 2)])
    text = str(m.bodies["cyl"])
    assert text.startswith("Body 'cyl'") and "mass = " in text and "tanks: 1" in text
    assert "ranges = [[1, 2]]" in text
    assert "yaw" not in text  # a default is not shown
    whole = str(m)
    assert whole.startswith("hb.Model") and "bodies: 1" in whole and "array (" in whole
    assert repr(m.bodies["cyl"]) == "<Body bodies/cyl>"


def test_lists_are_not_assigned_and_clear_is_all_or_nothing():
    # a list's objects are made by add() and insert(): a list is not assigned (nor given to add or
    # update); clear() empties it, refused (nothing removed) while another object refers to one
    m = _model()
    for value in ([{"name": "cyl2", "kind": "panels"}], []):
        with pytest.raises(TypeError, match=r"meshes is not assigned: make its objects with meshes.add\(...\)"):
            m.meshes = value
    with pytest.raises(TypeError, match="dipoles is not assigned"):
        m.bodies["cyl"].dipoles = []
    with pytest.raises(TypeError, match="unexpected keyword argument 'tanks'"):
        m.bodies.add("b", mesh="cyl", tanks=[])  # the keywords of add are the own items
    assert m.bodies.names() == ["cyl"]
    with pytest.raises(TypeError, match="tanks is not assigned"):
        ModelObject.update(m.bodies["cyl"], yaw=5.0, tanks=[])
    assert m.bodies["cyl"].yaw == 0.0
    with pytest.raises(ValueError, match="bodies/cyl refers to meshes/cyl"):
        m.meshes.clear()
    assert m.meshes.names() == ["cyl"]
    m.bodies.clear()
    m.meshes.clear()
    assert len(m.bodies) == 0 and len(m.meshes) == 0


def test_items_copy_an_object():
    # items() gives the object's own items, not its name nor its lists: add(**items()) copies them
    m = _model()
    b = m.bodies["cyl"]
    b.tanks.add(mesh="cyl", ranges=[(1, 2)])
    items = b.items()
    assert "name" not in items and "tanks" not in items and items["interior_surface"] == "detect"
    c = m.bodies.add("copy", **{**items, "position": [5.0, 0.0, 0.0]})
    assert c.mass == b.mass and len(c.tanks) == 0


def test_the_objects_by_ownership():
    # what moves with a body is under it, what is at rest in the water (lids, fixed dampers) a list
    # of the model; the method in options, what is computed in first_order and second_order, the
    # water itself in environment
    m = _model()
    assert m.first_order.diffraction_formulation == "auto" and m.first_order.motion
    assert not m.second_order.drift_pressure_integration
    assert m.environment.domain == "open_sea" and m.environment.channel_width is None
    m.second_order.update(drift_pressure_integration=True, drift_momentum_conservation=True)
    assert m.second_order.drift_bidirectional is False
    assert len(m.lids) == 0 and len(m.fixed_dampers) == 0
    assert m.first_order.field_points is None
    m.first_order.field_points = [[3.0, 0.0, -0.5]]
    assert m.first_order.field_points.tolist() == [[3.0, 0.0, -0.5]]
    with pytest.raises(AttributeError, match="a SecondOrder has no item 'qtf_free_surface'"):
        m.second_order.qtf_free_surface = None  # its items are second_order's: free_surface_method, ...
    damper = m.fixed_dampers.add(mesh="cyl", ranges=[(1, 2)], damping_factor=0.5)
    assert damper.path == "fixed_dampers/1" and damper.mesh == "cyl" and damper.ranges.tolist() == [[1, 2]]
    body = m.bodies["cyl"]
    body.dipole_dampers.add(mesh="cyl", ranges=[(3, 4)], damping_factor=1.0)
    assert body.dipole_dampers[0].path == "bodies/cyl/dipole_dampers/1"
    # the former places are gone
    for obj, name in ((m.options, "motion"), (m.options, "domain"), (body, "dipole_panels"), (damper, "condition")):
        with pytest.raises(AttributeError, match=f"no item or list '{name}'"):
            getattr(obj, name)
    for name in ("water", "outputs", "field_points"):
        with pytest.raises(AttributeError, match=f"no {name!r}"):
            getattr(m, name)
    assert not any(hasattr(m.meshes, f) for f in ("ccyl_cs", "ellipsoid_cs", "automatic_control_surface"))


def test_the_records_of_a_surface_are_its_mesh_and_its_ranges():
    # the mesh a name, the ranges one-based (first, last) pairs, sorted; None every element
    m = _model()
    body = m.bodies["cyl"]
    body.ranges = [(17, 32), (1, 8)]
    assert body.mesh == "cyl" and body.ranges.tolist() == [[1, 8], [17, 32]]
    for bad in (5, ("cyl", (1, 2))):
        with pytest.raises(TypeError, match="mesh must be a string"):
            body.mesh = bad
    for bad in ("1-8", [1, 2, 3]):
        with pytest.raises((TypeError, ValueError), match="ranges must"):
            body.ranges = bad
    assert body.ranges.tolist() == [[1, 8], [17, 32]]  # a refused value keeps the old one
    with pytest.raises(ValueError, match="unknown mesh 'nomesh'"):
        body.mesh = "nomesh"
    body.ranges = [(1, 40)]
    with pytest.raises(ValueError, match="cyl"):
        m.check()  # a range beyond the elements of the mesh, refused by the rules
    body.ranges = None
    assert body.ranges is None
    m.check()


def test_the_summary_shows_the_items_the_switches_choose():
    # print(obj): the items given that the switches choose, and a line naming the unused ones by the
    # switch value that leaves them out (a prefix whose items are all unused by its name)
    m = _model()
    body = m.bodies["cyl"]
    body.update(control_surface_radius=2.0, Ixx=1.0)
    text = str(body)
    assert "control_surface_radius" not in text.replace("control_surface_*", "") and "Ixx =" not in text
    assert text.splitlines()[-1].strip() == (
        'unused: Ixx ... Izz (inertia_input = "radii_of_gyration"), interior_surface_* (interior_surface = "detect"), '
        'control_surface_* (control_surface = "none")')  # fmt: skip
    body.update(control_surface="automatic", control_surface_outer="polygon")
    assert 'control_surface_radius (control_surface = "automatic" with control_surface_outer = "polygon")' in str(body)
    assert "control_surface_radius = 2.0" not in str(body)
    body.control_surface = "cylinder"
    assert "control_surface_radius = 2.0" in str(body)
    m.environment.frequency_input = "periods"
    assert 'unused: omegas (frequency_input = "periods")' in str(m.environment)
    # info() names the switch and its values
    i = body.info("control_surface_radius")
    assert i.switch == ("control_surface", "control_surface_outer")
    assert i.switch_values == ("cylinder", "automatic/circle")
    assert body.info("yaw").switch == ()
    assert set(body.items()) >= {"Ixx", "control_surface_radius", "interior_surface_mesh"}  # items() complete


def test_list_insert_repr_reset_and_the_single_child_of_the_water():
    m = _model()
    m.bodies.add("b", mesh="cyl", position=[5.0, 0.0, 0.0])
    m.bodies.insert(-1, "c", mesh="cyl", position=[10.0, 0.0, 0.0])  # as list.insert: before the last
    m.bodies.insert(-10, "d", mesh="cyl", position=[15.0, 0.0, 0.0])  # beyond the start: first
    m.bodies.insert(10, "e", mesh="cyl", position=[20.0, 0.0, 0.0])  # beyond the end: last
    assert m.bodies.names() == ["d", "cyl", "c", "b", "e"]
    with pytest.raises(AttributeError, match="no item 'mas'"):
        m.bodies["b"].reset("mas")  # as info()
    with pytest.raises(AttributeError, match="no item 'mas'"):
        m.bodies["b"].info("mas")
    # the free surface of the potential QTF: items of second_order, set all or nothing by update
    so = m.second_order
    so.free_surface_panel_size = 0.5
    with pytest.raises(AttributeError, match="no item 'bogus'"):
        ModelObject.update(so, free_surface_inner_radius=4.0, bogus=1)
    assert so.free_surface_panel_size == 0.5 and so.free_surface_inner_radius == "auto"
    with pytest.raises(TypeError):
        so.update(free_surface_inner_radius=4.0, free_surface_panel_size=[1.0])
    assert so.free_surface_inner_radius == "auto" and so.free_surface_panel_size == 0.5
    with pytest.raises(TypeError, match="a tank of this model"):
        m.classify_points([[3.0, 0.0, -0.5], [4.0, 0.0, -0.5]], m.bodies["b"])
    with pytest.raises(IndexError, match="no body 7"):
        m.meshes.load_csf("cs", "missing.csf", body=7)
    body, bodies, options = m.bodies["b"], m.bodies, m.options
    m.close()
    assert (repr(m), repr(bodies), repr(options)) == ("<hb.Model (closed)>", "<bodies (closed)>", "<Options (closed)>")
    assert repr(body).endswith("(closed)>")


def test_the_order_of_a_list_is_changed_by_move():
    # the order of a list is the order of its objects in the results and the files: move changes it
    m = _model()
    for name in ("b", "c"):
        m.bodies.add(name, mesh="cyl", position=[10.0 * (len(m.bodies)), 0.0, 0.0], modes=[])
    m.bodies.move("c", 0)
    assert m.bodies.names() == ["c", "cyl", "b"]
    m.bodies.move(0, -1)  # negative from the end, as a Python index
    assert m.bodies.names() == ["cyl", "b", "c"]
    m.bodies.move("cyl", 1)
    assert m.bodies.names() == ["b", "cyl", "c"]
    with pytest.raises(IndexError, match="bodies.move: index 3 out of range"):
        m.bodies.move("b", 3)
    with pytest.raises(TypeError, match="the index of bodies.move must be an integer"):
        m.bodies.move("b", 1.0)
    with pytest.raises(KeyError):
        m.bodies.move("none", 0)
    p = m.panels(include_reflections=False)  # the bodies numbered in their new order: b (x = 10) the first
    assert abs(p.centroids[p.body == 0][:, 0].mean() - 10.0) < 1.0


def test_the_warnings_of_the_model_and_the_diagnostics_of_the_results():
    m = _model()
    assert m.warnings == ()  # before a check
    w = m.check()
    assert m.warnings == tuple(w)
    r = m.solve()
    assert isinstance(r.diagnostics, tuple) and all(isinstance(d, hb.Diagnostic) for d in r.diagnostics)
    assert r.warnings == tuple(d for d in r.diagnostics if d.severity == "warning")
    # the quantities the results have, in the order of the quantity table
    names = list(hb.quantities())
    assert list(r.available) == sorted(r.available, key=names.index)
    assert "added_mass" in r.available and "drift_pressure_integration" not in r.available
    assert all(r.quantity(n) is not None for n in r.available)


def test_the_element_type_of_an_item_and_the_roles_of_the_panels():
    m = _model()
    assert m.environment.info("omegas").dtype == "real" and m.bodies["cyl"].info("modes").dtype == "text"
    assert hb.engine.schema()["Mesh"]["items"]["discretization"]["dtype"] == "int"
    assert m.environment.info("omegas").value_dtype == "real" and m.bodies["cyl"].info("mass").dtype == "real"
    # a role by the name the rows carry, or several; the list names and the old row words refused
    assert m.panels(["wetted", "fixed_damper"]).areas.tolist() == m.panels("wetted").areas.tolist()
    assert len(m.bodies["cyl"].panels("wetted").areas) == m.counts().per_body[0].wetted * 4  # the reflections
    for old in ("hull", "body", "wet", "interior", "fixed_dampers"):
        with pytest.raises(ValueError, match=r"role must be one of wetted, interior_surface, dipole, .*, got "):
            m.panels(old)
    with pytest.raises(TypeError, match="role must be None"):
        m.panels(["wetted", 3])


def test_results_of_checks_the_quantities_given_together():
    # Results.of refuses at once a drift without its companions, naming the one missing
    m = _model()
    F, H, D = 2, 1, 6
    z, zm = np.zeros((F, H, D), complex), np.zeros((F, H, 3), complex)  # zm: the moment about the moving origin
    with pytest.raises(ValueError, match="drift_pressure_integration_body_axes is missing"):
        hb.Results.of(m, drift_pressure_integration=z)
    with pytest.raises(ValueError, match="drift_control_surface is missing"):
        hb.Results.of(m, drift_control_surface_body_axes=zm)
    with pytest.raises(ValueError, match="drift_pressure_integration needs drift_heading_pairs"):
        hb.Results.of(m, drift_pressure_integration=z, drift_pressure_integration_body_axes=zm)
    ids = np.array([[0, 1, 1, 1]], np.int32)  # a row of a panel: its points, normals and areas needed
    with pytest.raises(ValueError, match="body_points, body_points_normals and body_points_areas: body_points is "
                                         "missing"):
        hb.Results.of(m, body_points_ids=ids)


def test_the_points_of_the_free_surface_are_read_only_and_the_summary_counts():
    m = _model()
    m.second_order.qtf_potential_indirect = True
    fs = m.free_surface_points("panelled")
    x, w = fs.points, fs.weights
    assert not x.flags.writeable and not w.flags.writeable and x.shape == (len(w), 3)
    first = m.solve().summary().splitlines()[0]
    assert ": 1 body, 2 ordinary frequencies, 1 heading, 6 coordinates, 6 radiation columns" in first
