"""Model rules of reference/inputs-rules §1: every rule raises its documented error before any
computation; a valid problem either has a solver route or is rejected
explicitly with UnsupportedProblemError."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import TANK_H, TANK_ZT, flat, lo_free_cylinder, part, spline, spline_cylinder, surface, tank_vessel
from meshes import box_panels, cylinder_panels, flat_plate, spline_cylinder_patches
from problems import add_mesh, flat_surface, hydrostatics, interior_vertices, model
from regression_problems import PROBLEMS

HULL, LID = (1, 32), (33, 40)  # the hull and the waterplane elements of box_panels(lid=True)


@pytest.fixture(scope="module")
def mesh():
    return surface(box_panels(lid=True))


@pytest.fixture(scope="module")
def hull():
    return surface(box_panels())


def body(source, **kw):
    """The box of box_panels(lid=True) (the mesh dict source): by default its hull the wetted
    surface and its waterplane the interior surface."""
    kw.setdefault("interior_surface", (source, LID))
    return {"mesh": (source, HULL), **kw}


def problem(bodies, omegas=(1.0,), headings=None, **kw):
    return model(bodies, omegas, headings, **kw)


def validate(p):
    return p.route()


def assert_valid(p):
    """The model rules accept p and it has a solver route (the check applies both)."""
    p.check()


def _refusal(f):
    with pytest.raises(hb.UnsupportedProblemError, match="no solver route") as e:
        f()
    return str(e.value)


def test_momentum_drift_in_a_channel_has_no_route(hull):
    # the momentum drift in a channel is valid but has no route: the check, the route and the
    # solve refuse it alike, with one message
    p = problem(
        {"mesh": hull, "modes": []},
        options={"drift_momentum_conservation": True},
        environment={"depth": 10.0, "domain": "channel", "channel_width": 10.0},
    )
    why = _refusal(p.route)
    assert "channel" in why
    assert _refusal(p.check) == why
    assert _refusal(p.solve) == why
    assert issubclass(hb.UnsupportedProblemError, NotImplementedError)
    assert not issubclass(hb.UnsupportedProblemError, ValueError)
    # without the drift the model has its route and passes the check
    p.second_order.drift_momentum_conservation = False
    assert all(isinstance(w, hb.Diagnostic) for w in p.check())
    assert p.route() == "low_order_finite_channel"


def _small_cylinder(draft=1.0):
    return surface(cylinder_panels(1.0, draft, n_phi=4, n_z=2, n_r=2, quarter=True), symmetry="xz_and_yz")


def _unsupported_models():
    """Valid models without a solver route, each with a word of its refusal."""
    small = _small_cylinder()
    fin = [[[1.0, 0.2, -0.3], [1.5, 0.2, -0.3], [1.5, 0.2, -0.1], [1.0, 0.2, -0.1]]]
    sheet = np.concatenate([cylinder_panels(1.0, 0.5, n_phi=4, n_z=2, n_r=2, quarter=True), fin])
    m2 = surface(sheet, symmetry="xz_and_yz")
    finite = {"depth": 10.0}
    return [
        # a channel narrower than depth / pi
        (model({"mesh": small}, [1.0], [0.0], environment={"depth": 20.0},
               options={"domain": "channel", "channel_width": 5.0}), "narrower"),
        # the momentum drift by a wall
        (model({"mesh": _small_cylinder(0.5), "position": [3.0, 0.0, 0.0], "modes": []}, [1.0], [0.0],
               options={"drift_momentum_conservation": True, "domain": "x_wall"}), "walls"),
        # the low-order pressure drift with a thin sheet
        (model({"mesh": part(m2, 1, len(sheet) - 1), "modes": [],
                "dipoles": [{"mesh": part(m2, len(sheet), len(sheet))}]}, [1.0], [0.0],
               options={"drift_pressure_integration": True}), "sheets"),
        # the momentum drift with a damping lid
        (model({"mesh": _small_cylinder(), "modes": []}, [1.0], [0.0], environment=finite,
               options={"drift_momentum_conservation": True},
               lids=[{"name": "l", "mesh": surface(flat_plate(center=(5, 0)))}]), "damping lids"),
    ]  # fmt: skip


def test_the_check_refuses_every_model_the_route_refuses():
    # a valid model without a route: the check raises UnsupportedProblemError (not ValueError) with
    # the message of the route, so that a model that passes the check has a route
    for p, word in _unsupported_models():
        why = _refusal(p.route)
        assert word in why, why
        assert _refusal(p.check) == why


# ------------------------------------------------------------- rule 1: types


def _body(m, **kw):
    m.bodies["body"].update(**kw)


def _pressure_surface(m, modes=(), **items):
    """A pressure surface of the body with its pressure modes (dicts of their items), added to it
    once it is made; the elements of the lid taken from the interior surface."""
    m.bodies["body"].interior_surface = "none"
    ps = m.bodies["body"].pressure_surfaces.add(**items)
    for d in modes:
        ps.pressure_modes.add(**d)


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda m: m.first_order.update(motion=1), r"first_order: motion must be a bool"),
        (lambda m: m.environment.update(domain=3), r"environment: domain must be a string"),
        (lambda m: m.bodies.add("other", mesh=m.meshes["body"].vertices), r"bodies/other: mesh must be a string"),
        (lambda m: _body(m, mesh=("body", (1, 3))), r"bodies/body: mesh must be a string"),
        (lambda m: _body(m, ranges="1-3"), r"bodies/body: ranges must be an array"),
        (lambda m: _body(m, yaw="0"), r"yaw must be a real number"),
        (lambda m: _body(m, modes="heave"), r"modes must be a sequence of strings"),
        (lambda m: _body(m, modes=(3.0,)), r"modes must be a sequence of strings"),
        (lambda m: _body(m, interior_surface=("body", (1, 3))), r"bodies/body: interior_surface must be a string"),
        (lambda m: m.environment.update(omegas=[1j]), r"omegas must be real"),
        (lambda m: m.environment.update(depth="deep"), r"depth must be a real number"),
        (lambda m: m.lids.add("lid", mesh=m.meshes["body"].vertices), r"lids/lid: mesh must be a string"),
        (lambda m: setattr(m.bodies["body"].pressure_surfaces.add(mesh="body", ranges=[LID]), "pressure_modes",
                           ["uniform"]),
         r"pressure_modes is not assigned"),
        (lambda m: m.options.update(frequency_threads=1.0), r"frequency_threads must be an integer"),
        (lambda m: m.bodies.add(1, mesh="body"), r"a name must be a string, got int"),
    ],
)
def test_type_errors(mesh, make, match):
    m = problem(body(mesh))
    with pytest.raises(TypeError, match=match):
        make(m)
        m.check()


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda m: m.environment.update(domain="wall"),
         r"domain must be one of open_sea, x_wall, y_wall, orthogonal_walls, channel, got 'wall'"),
        (lambda m: m.first_order.update(diffraction_formulation="direct"),
         r"diffraction_formulation must be one of auto, scattered, total, got 'direct'"),
        (lambda m: _body(m, modes=["heave", "surge"]),
         r"modes must be unique and in the order surge, sway, heave, roll, pitch, yaw"),
        (lambda m: _body(m, modes=["heave", "heave"]),
         r"modes must be unique and in the order surge, sway, heave, roll, pitch, yaw"),
        (lambda m: _body(m, modes=["bob"]),
         r"modes: unknown mode 'bob' \(use surge, sway, heave, roll, pitch, yaw\)"),
        (lambda m: _body(m, excitation_method="kochin"),
         r"excitation_method must be one of diffraction, haskind, got 'kochin'"),
        (lambda m: m.bodies.add("  ", mesh="body"), r"names must not be blank"),
        (lambda m: _body(m, mass=-1.0), r"mass must be a finite number >= 0, got -1\.0"),
        (lambda m: _body(m, additional_stiffness=np.eye(6), stiffness_origin="center_of_gravity"),
         r"stiffness_origin='center_of_gravity' requires center_of_gravity"),
        (lambda m: _body(m, additional_stiffness=np.eye(5)),
         r"additional_stiffness must have shape \(6, 6\), got \(5, 5\)"),
        (lambda m: _body(m, additional_damping=np.full((6, 6), np.inf)),
         r"additional_damping must contain finite real numbers that fit float32"),
        (lambda m: _body(m, position=(0, 0, math.nan)), r"position must be finite"),
        (lambda m: _body(m, ranges=[(0, 3)]), r"bodies/body: ranges: invalid element range \(0, 3\)"),
        (lambda m: _body(m, ranges=[(5, 3)]), r"invalid element range \(5, 3\) \(one-based, first <= last\)"),
        (lambda m: _body(m, ranges=[(1, 10), (10, 32)]),
         r"overlapping element ranges \(1, 10\) and \(10, 32\)"),
        (lambda m: _body(m, ranges=np.zeros((0, 2), dtype=int)), r"the ranges select no elements"),
        (lambda m: _body(m, **flat_surface("interior_surface", {"method": "mesh", "mesh": None})),
         r"bodies/body: interior_surface = 'mesh' needs interior_surface_mesh"),
        (lambda m: _body(m, interior_surface="flat"), r"interior_surface must be one of none, detect, waterline"),
        (lambda m: _pressure_surface(m, [{"name": "p"}]), r"a PressureSurface needs its mesh"),
        (lambda m: _pressure_surface(m, mesh="body", ranges=[LID]),
         r"a PressureSurface needs at least one PressureMode"),
        (lambda m: _pressure_surface(m, [{"name": "p", "shape": "sign_z"}], mesh="body", ranges=[LID]),
         r"shape must be one of uniform, sign_x, sign_y, got 'sign_z'"),
        (lambda m: m.lids.add("lid"), r"lids/lid: a DampingLid needs its mesh"),
        (lambda m: m.lids.add("lid", mesh="body", damping_factor=-0.1),
         r"damping_factor must be a finite number >= 0"),
        (lambda m: m.environment.update(depth=-1.0), r"depth must be > 0 or inf, got -1\.0"),
        (lambda m: m.environment.update(depth=math.nan), r"depth must be > 0 or inf, got nan"),
        (lambda m: m.environment.update(gravity=0.0), r"gravity must be a positive finite number, got 0\.0"),
        (lambda m: m.environment.update(density=math.inf),
         r"density must be a positive finite number, got inf"),
        (lambda m: m.environment.update(channel_width=-1.0),
         r"channel_width must be a positive finite number, got -1\.0"),
        (lambda m: m.options.update(inner_threads=0), r"inner_threads must be in 1\.\.1024"),
        (lambda m: _body(m, position=(0, 0)), r"position must have shape \(3,\), got \(2,\)"),
    ],
)
def test_field_value_errors(mesh, make, match):
    m = problem(body(mesh))
    with pytest.raises(ValueError, match=match):
        make(m)
        m.check()


def test_the_items_of_a_method_not_chosen_are_kept_unused(mesh):
    # as inertia_input: nothing is refused or reset when an item of another method is set, and the
    # check reads the items of the method chosen only
    m = problem(body(mesh))
    b = m.bodies["body"]
    with pytest.raises(AttributeError, match=r"a Body has no item 'interior_surface_radius'"):
        b.interior_surface_radius = 1.0
    b.update(control_surface_boundary=[[2.0, 0.0], [0.0, 2.0]], control_surface_radius=2.0)  # under "none"
    b.update(control_surface="cylinder", control_surface_depth=1.0, control_surface_inner_radius=1.5,
             control_surface_panel_size=0.5)  # fmt: skip
    assert b.control_surface_boundary.tolist() == [[2.0, 0.0], [0.0, 2.0]]  # kept, unused
    b.control_surface = "automatic"
    assert (b.control_surface_inner_radius, b.control_surface_radius) == (1.5, 2.0)


def test_the_interior_surface_detected_in_the_wetted_records(mesh):
    # the whole box (its waterplane in z = 0 among the elements): the default method detect takes
    # the elements in z = 0 as the interior surface, the rest is wetted
    m = problem({"mesh": mesh})
    c = m.counts().per_body[0]
    assert (c.wetted, c.interior_surface) == (HULL[1], LID[1] - LID[0] + 1)
    assert m.bodies["body"].interior_surface == "detect"
    # without an interior surface the elements in z = 0 are refused
    m.bodies["body"].interior_surface = "none"
    with pytest.raises(ValueError, match="lies in the free surface z = 0"):
        m.check()


def test_a_rule_names_the_stored_element(mesh):
    # the elements of a body's mesh no part takes are left out of the engine's mesh, so the engine
    # numbers the rest from 1; a rule names the element as stored (the numbering of ranges and of
    # Panels.element), in its message and its location
    m = problem({"mesh": part(mesh, 5, LID[1]), "interior_surface": "none"})
    with pytest.raises(ValueError, match=r"bodies/body: element 33 lies in the free surface z = 0") as e:
        m.check()
    assert (e.value.location.body, e.value.location.element) == (0, 33)


# ------------------------------------------------------------- rule 2


def test_duplicate_body_names(mesh):
    with pytest.raises(ValueError, match="duplicate body name"):
        validate(problem([body(mesh), body(mesh, position=(5, 0, 0))]))


def test_higher_order_requires_patch_mesh(mesh):
    with pytest.raises(ValueError, match="requires a patch mesh"):
        validate(problem(body(mesh), options={"higher_order": True}))
    pm = flat(box_panels(), (1, 1, 3, 3, 4, 4))
    with pytest.raises(ValueError, match="higher_order = true"):
        validate(problem({"mesh": pm}))
    # integers beyond int32 are refused, not wrapped
    big = flat(box_panels(), (2**32 + 1, 1, 3, 3, 4, 4))
    with pytest.raises(ValueError, match="exceeds int32"):
        validate(problem({"mesh": big}, options={"higher_order": True}))


def test_selection_outside_mesh(mesh):
    b = body(mesh, interior_surface=part(mesh, 33, 41))
    with pytest.raises(ValueError, match=r"interior_surface: the element range \(33, 41\) is beyond the mesh 'body' "
                                         r"\(40 elements\)"):  # fmt: skip
        validate(problem(b))
    with pytest.raises(ValueError, match=r"the wetted surface \(mesh\): the element range \(1, 41\) is beyond"):
        validate(problem(body(mesh, mesh=part(mesh, 1, 41), interior_surface="none")))


def test_a_record_has_one_role_in_a_body(mesh):
    b = body(mesh, dipoles=[{"mesh": part(mesh, 40, 40)}])
    with pytest.raises(ValueError, match="element 40 of the mesh 'body' is taken by both dipoles 1 and interior"):
        validate(problem(b))
    b = body(mesh, interior_surface="none", tanks=[{"mesh": part(mesh, 30, 33)}])
    with pytest.raises(ValueError, match=r"element 30 of the mesh 'body' is taken by both the wetted surface \(mesh\) "
                                         "and tanks 1"):  # fmt: skip
        validate(problem(b))
    # a lid placed with the body takes its elements too
    lid = {"name": "lid", "mesh": part(mesh, 33, 33), "placed_with": "body"}
    with pytest.raises(ValueError, match="element 33 of the mesh 'body' is taken by both interior_surface and lids/"):
        validate(problem(body(mesh), lids=[lid]))
    # the same mesh may be referenced by several bodies, each with its own roles
    two = [body(mesh), body(mesh, name="other", position=(5, 0, 0), interior_surface="waterline")]
    assert_valid(problem(two))


def test_a_body_has_a_wetted_surface_or_a_dipole(mesh):
    # the mesh is optional when the body is added; the rules want a wetted surface or a dipole
    plate = surface(flat_plate(up=False, z=-0.5))
    m = model({"name": "plate", "modes": ["heave"]}, [1.0], (), options={"diffraction": False})
    with pytest.raises(ValueError, match="bodies/plate: the body has no wetted surface, no dipoles and no free-"):
        m.check()
    m = model({"name": "plate", "modes": ["heave"], "dipoles": [{"mesh": plate}]}, [1.0], (),
              options={"diffraction": False})  # fmt: skip
    assert_valid(m)
    assert set(m.panels(include_reflections=False).role) == {"dipole"}
    # a body of dipole dampers only is refused: a porous sheet at rest is a damper of the water
    dampers = {"name": "annulus", "dipole_dampers": [{"mesh": plate, "damping_factor": 1.0}]}
    with pytest.raises(ValueError, match="bodies/annulus: the body has no wetted surface, no dipoles and no free-"):
        model([{"mesh": (mesh, HULL), "interior_surface": (mesh, LID)}, dampers], [1.0], ()).check()


def test_empty_wetted_surface(hull):
    # a hull trimmed out of the water leaves no wetted element
    with pytest.raises(ValueError, match="trim: the trimmed mesh has no submerged elements"):
        validate(problem({"mesh": hull, "trim": [1.0, 0.0, 0.0]}))


def test_trimmed_body_rules(mesh, hull):
    trim = {"trim": [0.0, 2.0, 0.0]}
    # the interior surface of a trimmed body is generated from its new waterline
    with pytest.raises(ValueError, match="bodies/body: a trimmed body has a generated interior"):
        validate(problem(body(mesh, **trim)))
    assert_valid(problem(body(mesh, **flat_surface("interior_surface", {"method": "waterline"}), **trim)))
    # a lid from the elements of a trimmed body is refused, a lid of its own is not
    lid = {"name": "l", "mesh": part(mesh, 33, 40), "placed_with": "body"}
    with pytest.raises(ValueError, match="lids/l: a lid from the elements of the trimmed body 'body'"):
        validate(problem(body(mesh, interior_surface="none", **trim), lids=[lid]))
    assert_valid(problem({"mesh": hull, **trim}, lids=[{"name": "l", "mesh": surface(flat_plate(center=(5, 0)))}]))
    ps = [{"mesh": surface(flat_plate(up=False, z=-0.2)), "pressure_modes": [{"name": "p"}]}]
    with pytest.raises(ValueError, match="pressure_surfaces/1: a pressure surface of a trimmed body is not available"):
        validate(problem({"mesh": hull, "pressure_surfaces": ps, **trim}))


def test_lid_rules(mesh, hull):
    total = {"diffraction_formulation": "total"}
    lid_mesh = surface(flat_plate(center=(5, 0)))
    with pytest.raises(ValueError, match="duplicate lid name"):
        validate(problem({"mesh": hull}, lids=[{"name": "a", "mesh": lid_mesh}] * 2, options=total))
    with pytest.raises(ValueError, match="lids/a: placed_with: unknown body 'other'"):
        validate(problem({"mesh": hull}, lids=[{"name": "a", "mesh": lid_mesh, "placed_with": "other"}], options=total))


def test_additional_matrices_are_6x6_and_couplings_name_another_body(hull):
    with pytest.raises(ValueError, match=r"additional_stiffness must have shape \(6, 6\), got \(6, 12\)"):
        problem({"mesh": hull, "additional_stiffness": np.zeros((6, 12))})
    two = [{"mesh": hull, "name": "a", "couplings": [{"body": "b", "stiffness": np.eye(6)}]},
           {"mesh": hull, "name": "b", "position": [5, 0, 0]}]  # fmt: skip
    assert_valid(problem(two))
    for coupling, message in (({"body": "a", "stiffness": np.eye(6)}, "names another body"),
                              ({"body": "b"}, "needs a stiffness or a damping"),
                              ({"stiffness": np.eye(6)}, "needs its body")):  # fmt: skip
        bad = [{**two[0], "couplings": [coupling]}, two[1]]
        with pytest.raises(ValueError, match=r"bodies/a/couplings/1.*" + message):
            validate(problem(bad))
    twice = [{**two[0], "couplings": [{"body": "b", "stiffness": np.eye(6)}, {"body": "b", "damping": np.eye(6)}]},
             two[1]]  # fmt: skip
    with pytest.raises(ValueError, match=r"bodies/a/couplings/2 \(with 'b'\): a second coupling with the body 'b'"):
        validate(problem(twice))
    with pytest.raises(ValueError, match="body: unknown body 'c'"):
        problem([{**two[0], "couplings": [{"body": "c", "stiffness": np.eye(6)}]}, two[1]])


# ------------------------------------------------------------- rule 3


def test_lid_geometry(hull):
    total = {"diffraction_formulation": "total"}
    low = surface(flat_plate(z=-0.1, center=(5, 0)))
    with pytest.raises(ValueError, match="not at z = 0"):
        validate(problem({"mesh": hull}, lids=[{"name": "a", "mesh": low}], options=total))
    # lid elements may be oriented either way (damping-lid panels of a GDF file usually point into the fluid)
    down = surface(flat_plate(up=False, center=(5, 0)))
    validate(problem({"mesh": hull}, lids=[{"name": "a", "mesh": down}], options=total))


def test_interior_geometry(mesh):
    b = body(mesh, mesh=part(mesh, 9, 32), interior_surface=part(mesh, 1, 8))  # the bottom
    with pytest.raises(ValueError, match="interior surface: a panel is not at z = 0"):
        validate(problem(b))
    raised = body(mesh, position=(0, 0, 0.1))
    with pytest.raises(ValueError, match="interior surface: a panel is not at z = 0"):
        validate(problem(raised))


def _pressure_body(hull, surface_vertices, **kw):
    return {"mesh": hull, "pressure_surfaces": [{"mesh": surface(surface_vertices),
                                                 "pressure_modes": [{"name": "p", **kw}]}]}


def test_pressure_surface_geometry(hull):
    tilted = flat_plate(up=False, z=-0.2)
    tilted[0, :, 2] = [-0.2, -0.25, -0.25, -0.2]
    with pytest.raises(ValueError, match="not horizontal"):
        validate(problem(_pressure_body(hull, tilted)))
    with pytest.raises(ValueError, match="above z = 0"):
        validate(problem(_pressure_body(hull, flat_plate(up=False, z=0.1))))
    # either way up: the elements pointing up are turned over (as for lids)
    assert_valid(problem(_pressure_body(hull, flat_plate(up=True, z=-0.2))))
    assert_valid(problem(_pressure_body(hull, flat_plate(up=False, z=-0.2))))


# ------------------------------------------------------------- rules 4-6


def test_nothing_to_solve(mesh):
    # radiation and diffraction both off is the hydrostatics-only problem, not an error
    assert validate(problem(body(mesh), options={"radiation": False, "diffraction": False})) == "hydrostatics"
    with pytest.raises(ValueError, match="nothing to solve"):
        validate(problem(body(mesh, modes=[]), options={"diffraction": False}))


def test_rao_rules(mesh):
    inertia = dict(inertia_input="components", Ixx=1.0, Ixy=0.0, Ixz=0.0, Iyy=1.0, Iyz=0.0, Izz=1.0)
    rao = {"motion": True}
    with pytest.raises(ValueError, match="requires radiation = true and diffraction = true"):
        validate(problem(body(mesh, mass="displaced", center_of_gravity=(0, 0, 0), **inertia),
                         options={"motion": True, "radiation": False}))
    with pytest.raises(ValueError, match="center_of_gravity"):
        validate(problem(body(mesh), options=rao))
    with pytest.raises(ValueError, match="requires the mass of every moving body"):
        validate(problem(body(mesh, center_of_gravity=(0, 0, 0), **inertia), options=rao))
    # a zero inertia (the radii of gyration by default zero): the rotations take the added inertia
    assert_valid(problem(body(mesh, mass="displaced", center_of_gravity=(0, 0, 0)), options=rao))
    with pytest.raises(ValueError, match="ordinary frequency"):
        validate(problem(body(mesh, mass="displaced", center_of_gravity=(0, 0, 0), **inertia), omegas=(0.0,),
                         options=rao))
    # a fixed body needs no mass properties
    assert_valid(
        problem(
            [
                body(mesh, mass="displaced", center_of_gravity=(0, 0, 0), **inertia),
                body(mesh, name="fixed", modes=[], position=(5, 0, 0)),
            ],
            options=rao,
        )
    )


def test_frequency_rules(mesh):
    with pytest.raises(ValueError, match="empty"):
        validate(problem(body(mesh), omegas=()))
    with pytest.raises(ValueError, match="duplicate"):
        validate(problem(body(mesh), omegas=(1.0, 1.0)))
    with pytest.raises(ValueError, match=">= 0"):
        validate(problem(body(mesh), omegas=(-1.0,)))
    with pytest.raises(ValueError, match="duplicate"):
        validate(problem(body(mesh), headings=(0.0, 0.0)))
    with pytest.raises(ValueError, match="limits"):
        validate(problem(body(mesh), omegas=(0.0, 1.0), options={"radiation": False}))
    with pytest.raises(ValueError, match="limits"):
        validate(problem(body(mesh, modes=[]), omegas=(math.inf, 1.0)))
    with pytest.raises(ValueError, match="diffraction requires at least one ordinary"):
        validate(problem(body(mesh), omegas=(0.0, math.inf)))
    with pytest.raises(ValueError, match="heading"):
        validate(problem(body(mesh), headings=()))
    fp = np.array([[5.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="field points require"):
        validate(problem(body(mesh), omegas=(0.0,), field_points=fp, options={"diffraction": False}))
    assert_valid(problem(body(mesh), omegas=(0.0, math.inf), options={"diffraction": False}))


# ------------------------------------------------------------- rule 7


def test_domain_rules(mesh):
    with pytest.raises(ValueError, match="channel_width"):
        validate(problem(body(mesh), options={"domain": "channel"}))
    with pytest.raises(ValueError, match="only allowed with domain 'channel'"):
        validate(problem(body(mesh), environment={"channel_width": 10.0}))
    ch = {"domain": "channel", "channel_width": 10.0}
    with pytest.raises(ValueError, match="finite depth"):
        validate(problem(body(mesh), options=ch))
    with pytest.raises(ValueError, match="headings 0 and 180"):
        validate(problem(body(mesh), headings=(0.0, 90.0), options=ch, environment={"depth": 5.0}))
    assert_valid(problem(body(mesh), headings=(0.0, 180.0, -180.0), options=ch, environment={"depth": 5.0}))
    narrow = {"domain": "channel", "channel_width": 0.9}
    with pytest.raises(ValueError, match="fluid side"):
        validate(problem(body(mesh), options=narrow, environment={"depth": 5.0}))


@pytest.mark.parametrize(
    "domain, position, ok",
    [
        ("x_wall", (1.5, 0, 0), True),
        ("x_wall", (1.0, 0, 0), False),  # touches x = 0
        ("y_wall", (0, 0.6, 0), True),
        ("y_wall", (0, 0.4, 0), False),
        ("orthogonal_walls", (1.5, 0.6, 0), True),
        ("orthogonal_walls", (1.5, -0.6, 0), False),
    ],
)
def test_bodies_on_the_fluid_side_of_walls(mesh, domain, position, ok):
    p = problem(body(mesh, position=position), options={"domain": domain})
    if ok:
        assert_valid(p)
    else:
        with pytest.raises(ValueError, match="fluid side"):
            validate(p)


def test_yawed_body_against_wall(mesh):
    # the 2 x 1 box yawed by 90 degrees spans x in [1.5 - 0.5, 1.5 + 0.5]
    assert_valid(problem(body(mesh, position=(1.5, 0, 0), yaw=90.0), options={"domain": "x_wall"}))
    with pytest.raises(ValueError, match="fluid side"):
        validate(problem(body(mesh, position=(0.45, 0, 0), yaw=90.0), options={"domain": "x_wall"}))


@pytest.mark.parametrize(
    "point, message",
    [
        ((3.0, 0.0, 0.1), "above the free surface"),
        ((3.0, 0.0, -0.4), None),  # submerged: the pressure (theory §4.1)
        ((0.2, 0.1, 0.0), "inside the water plane"),
        ((1.0, 0.2, 0.0), "lies on bodies/"),
        ((5.0, 0.0, 0.0), None),
    ],
)
def test_field_point_rules(mesh, point, message):
    p = problem(body(mesh), field_points=np.array([point]))
    if message is None:
        assert_valid(p)
    else:
        with pytest.raises(ValueError, match=message):
            validate(p)


def test_field_point_symmetric_and_placed_body():
    quarter = surface(box_panels(quarter=True), symmetry="xz_and_yz")
    b = {"mesh": quarter, "position": [10.0, 0.0, 0.0], "yaw": 30.0}
    with pytest.raises(ValueError, match="inside the water plane"):
        validate(problem(b, field_points=np.array([[9.5, -0.1, 0.0]])))
    assert_valid(problem(b, field_points=np.array([[0.0, 0.0, 0.0]])))


def test_field_point_on_lid(hull):
    lid = {"name": "moon", "mesh": surface(flat_plate(center=(5, 0)))}
    p = problem(
        {"mesh": hull},
        lids=[lid],
        field_points=np.array([[5.1, 0.1, 0.0]]),
        options={"diffraction_formulation": "total"},
    )
    with pytest.raises(ValueError, match="lies on lids/moon"):
        validate(p)


def test_field_point_outside_wall_domain(mesh):
    p = problem(body(mesh, position=(2, 0, 0)), field_points=np.array([[-1.0, 0, 0]]), options={"domain": "x_wall"})
    with pytest.raises(ValueError, match="outside the fluid domain"):
        validate(p)


# ------------------------------------------------------------- rule 8


def test_integration_rules(mesh, hull):
    lid = [{"name": "a", "mesh": surface(flat_plate(center=(5, 0)))}]
    # the scattered potential with a lid: the low order (the lid's boundary value), not the higher order
    assert_valid(problem({"mesh": hull}, lids=lid, options={"diffraction_formulation": "scattered"}))
    ho = PROBLEMS["ho_moonpool_lid"]()
    ho.first_order.diffraction_formulation = "scattered"
    with pytest.raises(ValueError, match="damping lids require first_order.diffraction_formulation='total'"):
        validate(ho)
    assert_valid(problem({"mesh": hull}, lids=lid))  # auto: the total formulation with a lid
    with pytest.raises(ValueError, match="interior surfaces require"):
        validate(problem(body(mesh), options={"logarithmic_integration": False}))
    with pytest.raises(ValueError, match="lids require options.logarithmic"):
        validate(
            problem(
                {"mesh": hull},
                lids=lid,
                options={"logarithmic_integration": False, "diffraction_formulation": "total"},
            )
        )
    with pytest.raises(ValueError, match="pressure surfaces at z = 0"):
        validate(problem(_pressure_body(hull, flat_plate(up=False, z=0.0)), options={"logarithmic_integration": False}))
    with pytest.raises(ValueError, match="channel' requires options.logarithmic"):
        validate(
            problem(
                {"mesh": hull},
                environment={"depth": 4.0},
                options={"domain": "channel", "channel_width": 10.0, "logarithmic_integration": False},
            )
        )
    assert_valid(problem({"mesh": hull}, options={"logarithmic_integration": False}))


# ------------------------------------------------------------- rule 9


def test_finite_depth_zero_frequency_flux(mesh):
    finite = {"depth": 3.0}
    rad = {"diffraction": False}
    with pytest.raises(ValueError, match="zero-frequency heave mode has a net volume flux"):
        validate(problem(body(mesh), omegas=(0.0, 1.0), environment=finite, options=rad))
    # surge, sway and yaw have no net flux; roll and pitch about the waterplane centroid neither
    assert_valid(
        problem(
            body(mesh, modes=["surge", "sway", "roll", "pitch", "yaw"]),
            omegas=(0.0,),
            environment=finite,
            options=rad,
        )
    )
    with pytest.raises(ValueError, match="pitch"):
        validate(
            problem(
                body(mesh, modes=["pitch"], rotation_center=(0.3, 0, 0)),
                omegas=(0.0,),
                environment=finite,
                options=rad,
            )
        )
    # infinite depth and omega > 0 are fine
    assert_valid(problem(body(mesh), omegas=(0.0, 1.0), options=rad))
    assert_valid(problem(body(mesh), omegas=(1.0,), environment=finite, options=rad))


def test_pressure_mode_flux(hull):
    finite, rad = {"depth": 3.0}, {"diffraction": False}
    plate = flat_plate(up=False, z=-0.2, n=(2, 2))

    def surge_body(shape):
        surfaces = _pressure_body(hull, plate, shape=shape)["pressure_surfaces"]
        return {"mesh": hull, "modes": ["surge"], "pressure_surfaces": surfaces}

    with pytest.raises(ValueError, match="pressure mode 'p'"):
        validate(problem(surge_body("uniform"), omegas=(0.0,), environment=finite, options=rad))
    assert_valid(problem(surge_body("sign_x"), omegas=(0.0,), environment=finite, options=rad))


# ------------------------------------------------------------- rule 10 and order


def test_inertia_positive_semi_definite(mesh):
    bad = dict(inertia_input="components", Ixx=1.0, Ixy=2.0, Ixz=0.0, Iyy=1.0, Iyz=0.0, Izz=1.0)
    with pytest.raises(ValueError, match=r"not positive semi-definite \(a negative kinetic energy in the direction "
                                         r"of the combination of the free coordinates roll, pitch\)"):
        validate(problem(body(mesh, mass="displaced", center_of_gravity=(0, 0, 0), **bad), options={"motion": True}))
    # restricted to heave only, the same tensor is fine
    assert_valid(problem(body(mesh, modes=["heave"], mass="displaced", center_of_gravity=(0, 0, 0), **bad),
                         options={"motion": True}))
    # a massless body or a zero inertia: the inertia of the motions is the added mass
    assert_valid(problem(body(mesh, mass=0.0, center_of_gravity=(0, 0, 0)), options={"motion": True}))
    # a generalized mode: massless when it moves water, refused when it moves none (only the interior
    # surface), a negative mode mass refused naming the mode
    def gen(ranges, mass):
        mode = {"name": "g", "parity": "heave", "pieces": [{"w": [[1.0, 0, 0, 0]], "ranges": [ranges]}]}
        return problem(body(mesh, modes=["heave"], mass="displaced", center_of_gravity=(0, 0, 0),
                            generalized_modes=[mode], mode_mass=np.diag([0.0] * 6 + [mass])),
                       options={"motion": True})  # fmt: skip
    assert_valid(gen(HULL, 0.0))
    assert_valid(gen(LID, 1.0))
    with pytest.raises(ValueError, match=r"the free coordinate generalized mode 'g' has no mass and moves no water"):
        validate(gen(LID, 0.0))
    with pytest.raises(ValueError, match=r"not positive semi-definite \(a negative mass of generalized mode 'g'\)"):
        validate(gen(HULL, -1.0))


def test_inertia_is_checked_with_rao_only(mesh):
    # rule 10 holds with motion = True only: without the motions the inertia is not used
    bad = dict(inertia_input="components", Ixx=1.0, Ixy=2.0, Ixz=0.0, Iyy=1.0, Iyz=0.0, Izz=1.0)
    assert_valid(problem(body(mesh, mass="displaced", center_of_gravity=(0, 0, 0), **bad), options={"motion": False}))


def test_drift_of_a_body_moving_in_generalized_modes_needs_motion(mesh):
    # a body with modes = [] and a generalized mode still moves: its drift needs the motions
    bend = {"name": "bend", "pieces": [{"w": [[1.0, 0, 0, 0]]}]}
    with pytest.raises(ValueError, match="needs its motions"):
        validate(problem(body(mesh, modes=[], generalized_modes=[bend]), headings=[0.0],
                         options={"drift_momentum_conservation": True}))
    assert_valid(problem(body(mesh, modes=[]), headings=[0.0], options={"drift_momentum_conservation": True}))


def test_rule_order(mesh):
    # rule 4 (nothing to solve) is reported before rule 7 (channel width)
    p = problem(body(mesh, modes=[]), options={"diffraction": False, "domain": "channel"})
    with pytest.raises(ValueError, match="nothing to solve"):
        validate(p)


# ------------------------------------------------------------- counts and queries


def test_counts_with_symmetry_lids_and_pressure():
    v = box_panels(quarter=True, lid=True)
    quarter = surface(v, symmetry="xz_and_yz")
    nlid = 8  # 4 x 2 waterplane elements at the end
    nhull = v.shape[0] - nlid
    plate = surface(flat_plate(a=0.5, up=False, z=-0.3, center=(0.25, 0)), symmetry="yz")
    b = {
        "mesh": part(quarter, 1, nhull),
        "interior_surface": part(quarter, nhull + 1, nhull + nlid),
        "modes": ["surge", "heave"],
        "pressure_surfaces": [{"mesh": plate, "pressure_modes": [{"name": "a"}, {"name": "b", "shape": "sign_x"}]}],
    }
    b2 = {"mesh": surface(box_panels()), "name": "second", "position": [6, 0, 0], "modes": ["yaw"]}
    lid = {"name": "far", "mesh": surface(flat_plate(center=(-8, 1.0)), symmetry="xz")}
    p = problem(
        [b, b2],
        lids=[lid],
        field_points=np.array([[0, 5.0, 0], [0, 6.0, 0]]),
        options={"diffraction_formulation": "total"},
    )
    c = p.counts()
    assert c.coordinates == 6 + 2 + 6
    assert c.radiation_mode_ids == (1, 3, 7, 8, 14)
    assert c.radiation_columns == 5
    assert c.field_points == 2
    assert c.per_lid_elements == (8,)
    assert c.pressure_elements == 8
    assert c.per_body[0] == hb.BodyCounts(
        stored=nhull + nlid, wetted=nhull, wetted_with_reflections=4 * nhull, dipole=0, interior_surface=nlid,
        pressure_modes=2,
        symmetry_copies=4, generalized_modes=0,
    )
    assert c.per_body[1].wetted_with_reflections == 32


def test_interior_panels_of_a_body():
    v = box_panels(quarter=True, lid=True)
    n = len(v)
    quarter = surface(v, symmetry="xz_and_yz")
    b = {
        "mesh": part(quarter, 1, n - 8),
        "position": [3.0, 0.0, 0.0],
        "yaw": 90.0,
        "interior_surface": part(quarter, n - 7, n),
    }
    p = problem(b)
    im = interior_vertices(p, "body")
    assert im.shape == (32, 4, 3)
    np.testing.assert_allclose(im[..., 2], 0.0)
    # yawed by 90 degrees: x extent becomes the body's y extent
    assert np.isclose(im[..., 0].min(), 2.5) and np.isclose(im[..., 0].max(), 3.5)
    for q in im:
        assert np.cross(q[2] - q[0], q[3] - q[1])[2] > 0  # upward after reflection
    gen = problem({"mesh": surface(box_panels()), "interior_surface": {"method": "waterline"}})
    assert gen.counts().per_body[0].interior_surface == -1
    # generated interior meshes cover the waterplane 2 x 1 at z = 0 with upward normals
    for method in ("waterline", "projected", "graded"):
        g = problem({"mesh": surface(box_panels()), "interior_surface": method})
        q = interior_vertices(g)
        np.testing.assert_allclose(q[..., 2], 0.0)
        area = 0.5 * np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])[:, 2]
        assert np.all(area > 0) and np.isclose(area.sum(), 2.0, rtol=1e-12), method


def test_hydrostatics_box_discrete_closed_form():
    # box 2 x 1 x 0.5, 4 x 2 waterplane panels of size 0.5: one-point rule gives
    # int x^2 = I - A h^2 / 12 (test-plan A7)
    p = problem({"mesh": surface(box_panels()), "mass": "displaced", "center_of_gravity": [0, 0, -0.1]})
    h = hydrostatics(p)[0]
    rg = 1000 * 9.80665
    assert np.allclose(h.volumes, 1.0, rtol=1e-14)
    assert np.isclose(h.center_of_buoyancy[2], -0.25, rtol=1e-14)
    ixx = 2.0 * 1.0**3 / 12 - 2.0 * 0.5**2 / 12
    iyy = 2.0**3 * 1.0 / 12 - 2.0 * 0.5**2 / 12
    c = h.stiffness
    assert np.isclose(c[2, 2], rg * 2.0, rtol=1e-14)
    assert np.isclose(c[3, 3], rg * (ixx - 0.25) + 1000 * 9.80665 * 0.1, rtol=1e-12)
    assert np.isclose(c[4, 4], rg * (iyy - 0.25) + 1000 * 9.80665 * 0.1, rtol=1e-12)


def test_wavenumber():
    assert hb.wavenumber(0.0) == 0.0
    assert hb.wavenumber(2.0, gravity=9.81) == 4.0 / 9.81
    with pytest.raises(ValueError, match="omega must be a finite number >= 0"):
        hb.wavenumber(-1.0)
    # finite depth: the dispersion relation is test_finite_depth_and_limits.py's A3
    assert hb.wavenumber(0.0, depth=10.0) == 0.0


def test_complex_input_is_rejected(hull):
    m = problem({"mesh": hull}, [1.0], ())
    with pytest.raises(TypeError, match="headings must be real"):
        m.environment.headings = [1.0 + 1.0j]
    with pytest.raises(TypeError, match="points must be real"):
        m.first_order.field_points = [[3.0 + 0j, 0.0, 0.0]]
    with pytest.raises(TypeError, match="radii_of_gyration must be real"):
        m.bodies["body"].update(mass="displaced", center_of_gravity=[0, 0, 0], radii_of_gyration=[1j, 1, 1])
    with pytest.raises(TypeError, match="body_points must be real"):
        m.bodies["body"].body_points = [[1j, 0, 0]]


@pytest.mark.parametrize("omega", [1e-170, 1e200])
def test_frequencies_whose_wavenumber_underflows_or_overflows_are_refused(omega):
    with pytest.raises(ValueError, match="outside the frequencies the engine resolves"):
        model(lo_free_cylinder(), [omega], [0.0]).check()


def test_a_wetted_record_in_the_free_surface_is_refused():
    # a box whose mesh also holds its waterplane (a lid in z = 0) among the hull's elements: the lid's
    # source and image coincide; the rule names the element instead of a singular system
    body = {"mesh": surface(box_panels(0.5, 0.5, 0.5, (2, 2, 2), lid=True)), "modes": [], "interior_surface": None}
    with pytest.raises(ValueError, match="lies in the free surface z = 0"):
        model(body, [1.0], [0.0]).route()
    # the default interior surface (detect) takes them
    model({**body, "interior_surface": {"method": "detect"}}, [1.0], [0.0]).route()


# ------------------------------------------------------------- the refusals of the problem's rules
# (the rules the document cannot check alone: between objects, of the geometry, of the outputs)


def _modes(name="g"):
    """A generalized mode of the box: a uniform heave of every element."""
    return {"name": name, "parity": "heave", "pieces": [{"w": [[1.0, 0, 0, 0]]}]}


def test_a_mode_name_is_unique_among_the_generalized_and_pressure_modes(mesh):
    b = body(mesh, interior_surface="none", generalized_modes=[_modes("u")],
             pressure_surfaces=[{"mesh": part(mesh, 33, 40), "pressure_modes": [{"name": "u"}]}])  # fmt: skip
    with pytest.raises(ValueError, match="bodies/body: the mode name 'u' is used twice"):
        problem(b).check()
    # the pressure modes of two surfaces of a body share their names too
    b = body(mesh, interior_surface="none",
             pressure_surfaces=[{"mesh": part(mesh, 33, 36), "pressure_modes": [{"name": "p"}]},
                                {"mesh": part(mesh, 37, 40), "pressure_modes": [{"name": "p"}]}])  # fmt: skip
    with pytest.raises(ValueError, match="bodies/body: duplicate pressure mode name 'p'"):
        problem(b).check()


def test_mode_matrices_need_generalized_modes(mesh):
    with pytest.raises(ValueError, match="bodies/body: mode_mass needs generalized modes"):
        problem(body(mesh, mode_mass=np.zeros((6, 6)))).check()
    assert_valid(problem(body(mesh, generalized_modes=[_modes()], mode_mass=np.zeros((7, 7)))))


def _flat_patch(z):
    """One flat patch of 2 x 1 at z (upward normal), a higher-order mesh."""
    return flat([[[-1, -0.5, z], [1, -0.5, z], [1, 0.5, z], [-1, 0.5, z]]], [1, 1, 2, 2, 2, 2])


def test_the_surfaces_of_other_meshes_have_the_order_and_symmetry_of_the_body_mesh(mesh):
    with pytest.raises(ValueError, match="bodies/body: the interior mesh must be of the same order as the body mesh"):
        problem(body(mesh, **flat_surface("interior_surface", (_flat_patch(0.0), (1, 1))))).check()
    with pytest.raises(ValueError, match="bodies/body: dipoles: the mesh of the dipoles must be of the same order"):
        problem(body(mesh, dipoles=[{"mesh": _flat_patch(-0.2)}])).check()
    plate = surface(flat_plate(z=-0.3, n=(2, 2)), symmetry="yz")
    with pytest.raises(ValueError, match="bodies/body: dipoles: the mesh of the dipoles must have the symmetry"):
        problem(body(mesh, dipoles=[{"mesh": plate}])).check()


def _disc(z, up):
    """The quarter disc of radius 1 at z as a spline patch (higher order, symmetry xy): from the rim to
    the axis its normal points down, from the axis to the rim up."""
    order, _, t, tv, c = spline_cylinder_patches(1.0, 0.5, spans=4, order=4)[1]
    c = c.copy()
    c[:, :, 2] = z
    return spline([(order, 2, t, tv, c[::-1] if up else c)], (4, 2, 3, 3, 3, 4), symmetry="xz_and_yz")


def test_the_normals_of_curved_interior_and_pressure_surfaces():
    # elements given the other way are turned over, curved patches too (their u reversed): the
    # interior surface pointing down gives the coefficients of the one pointing up to round-off,
    # and a pressure surface may point up or down
    ho = {"higher_order": True}
    r = [problem({"mesh": spline_cylinder(), "interior_surface": (_disc(0.0, up), (1, 1))}, options=ho).solve()
         for up in (True, False)]  # fmt: skip
    for q in ("added_mass", "radiation_damping"):
        a, b = r[0].quantity(q), r[1].quantity(q)
        assert np.abs(a - b).max() <= 1e-12 * np.abs(a).max(), q
    for up in (True, False):
        ps = [{"mesh": _disc(-0.2, up), "pressure_modes": [{"name": "p"}]}]
        assert_valid(problem({"mesh": spline_cylinder(), "pressure_surfaces": ps}, options=ho))


def test_a_tank_cut_away_by_the_trim_is_refused():
    # the trim cuts a tank at its own free surface: below the walls, nothing is left of it
    b = tank_vessel()
    b["tanks"][0]["surface_level"] = TANK_ZT - TANK_H - 0.5
    with pytest.raises(ValueError, match="bodies/body: tank 1: a tank element lies above its surface level"):
        problem(b).check()
    b["trim"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="bodies/body: tank 1 has no panels"):
        problem(b).check()


_QTF = {"qtf_potential_direct": True}


def _qtf(b, **kw):
    """A model of the potential QTF of a fixed body at two frequencies."""
    return problem({**b, "modes": []}, (1.0, 1.2), outputs=_QTF, **kw)


# the bodies and the water the potential QTF is not available with: test_second_order_refusals.py


def test_the_free_surface_mesh_of_the_potential_qtf_is_of_panels(mesh):
    m = _qtf(body(mesh))
    add_mesh(m, "fs", _flat_patch(0.0))
    m.second_order.update(**flat_surface("free_surface", {"method": "mesh", "mesh": "fs"}))
    with pytest.raises(ValueError, match="second_order: the mesh must be of panels"):
        m.check()
