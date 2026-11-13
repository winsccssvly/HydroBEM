"""Problems of the regression baselines (tests/data/regression, test_regression.py).

Small procedural problems that together reach the main routes of the engine
(low and higher order, finite depth and the frequency limits, walls and the
channel, interior surfaces, damping lids, pressure surfaces, thin sheets,
several bodies, field and body outputs, both drift methods). The problems
of problems.FEATURES (generalized modes, dampers, an analytic shape, a trimmed patch mesh) are
baselines too, and those below add internal tanks with the Haskind excitation, the
control-surface and momentum drifts of every heading pair, a partly reflecting channel, a
wavemaker in the walls and a higher-order body with an interior patch in a channel; the second
order: the quadratic QTFs (pressure integration, control surface) with Newman's approximation of
the three drifts, the potential QTF of both methods (with a quadrature zone; by a wall) and the
second-order quantities of generalized modes; free-surface plates in both methods (a piston of
its own body in a moonpool with motions, field and body outputs; the cover of a barge's moonpool
as a plate of the barge at omega = 0 and ordinary frequencies; the momentum drift with plates in
both methods, with Newman's approximation of it; mats beside a vessel with a tank in a channel; the
barge's cover at both limits, omega = inf included; the pressure drift and the quadratic and
potential QTFs of a floating hull with a piston body on its moonpool; a tank half covered by a
plate of its free surface). The baselines
are HydroBEM's own results: they detect a change of any output, they are no
reference of accuracy (that is the job of the other tests). Regenerate them with
scripts/gen_regression_baselines.py after an intended change, and record why.
"""

import math

import numpy as np

import hydrobem as hb

from bodies import (
    acv_body,
    flat,
    ho_acv,
    lo_cosine_cylinder,
    lo_flexible_cylinder,
    lo_free_cylinder,
    moonpool_barge,
    moonpool_lid_mesh,
    moonpool_mesh,
    part,
    plate_column,
    spline_cylinder,
    surface,
    tank_vessel,
)
from meshes import MOONPOOL_FACES, PLATE_FACES, box_panels, cylinder_panels, face_panels, spheroid_panels
from problems import FEATURES, model

G = 9.80665
W = np.sqrt(np.array([0.5, 1.5]) * G)
POINTS = np.array([[3.0, 0.5, 0.0], [2.0, 2.0, -0.4], [-2.5, 1.0, -0.2]])


def _problem(bodies, omegas=W, headings=(0.0, 30.0), depth=math.inf, **kw):
    return model(bodies, omegas, headings, environment={"depth": depth, "gravity": G}, **kw)


def _cylinder(n=8, **kw):
    pan = cylinder_panels(1.0, 0.5, n_phi=n, n_z=4, n_r=4, quarter=True)
    return {"mesh": surface(pan, symmetry="xz_and_yz"), **kw}


def _box(**kw):
    return {"mesh": surface(box_panels(1.0, 0.5, 0.5, n=(6, 3, 3), quarter=True), symmetry="xz_and_yz"), **kw}


def lo_cylinder_drift():
    return _problem(lo_free_cylinder(), options={"motion": True, "drift_pressure_integration": True,
                                                 "drift_momentum_conservation": True})  # fmt: skip


def lo_cylinder_limits():
    return _problem(_cylinder(), omegas=[0.0, W[0], math.inf], headings=(), options={"diffraction": False})


def lo_cylinder_field_body_points():
    pts = np.array([[1.0, 0.0, -0.25], [0.6, 0.6, -0.5]])
    return _problem(_cylinder(body_points=pts), field_points=POINTS,
                    options={"field_pressure": True, "field_velocity": True, "body_pressure": True,
                             "body_velocity": True})  # fmt: skip


def lo_box_walls_finite():
    return _problem(_box(position=[2.0, 1.5, 0.0], modes=[]), depth=3.0, field_points=POINTS + [3.0, 3.0, 0.0],
                    options={"domain": "orthogonal_walls", "field_pressure": True, "field_velocity": True,
                             "drift_pressure_integration": True})  # fmt: skip


def lo_box_channel():
    return _problem(_box(), headings=(0.0, 180.0), depth=3.0, field_points=POINTS * [1.0, 0.5, 1.0],
                    options={"domain": "channel", "channel_width": 6.0, "field_pressure": True,
                             "field_velocity": True})  # fmt: skip


def lo_cylinder_interior():
    return _problem(_cylinder(interior_surface={"method": "graded"}), omegas=np.sqrt([1.0, 2.5]) * math.sqrt(G),
                    options={"field_velocity": True}, field_points=POINTS)  # fmt: skip


def lo_moonpool_lid():
    lid = {"name": "moon", "mesh": moonpool_lid_mesh(), "damping_factor": 0.1}
    return _problem({"mesh": moonpool_mesh()}, headings=(0.0,), lids=[lid],
                    options={"diffraction_formulation": "total"})  # fmt: skip


def lo_acv_two_modes():
    body = acv_body(modes=("sign_x", "uniform"), mass="displaced", center_of_gravity=[0.0, 0.0, 0.0],
                    radii_of_gyration=[2.5, 5.0, 5.0])
    return _problem(body, omegas=2 * np.pi / np.array([4.0, 6.0]), headings=(150.0,), depth=20.0,
                    options={"motion": True})  # fmt: skip


def lo_plate_sheet():
    hull, sheet = [], []
    for k, (o, du, dv, nn) in enumerate(PLATE_FACES):
        nu, nv = (max(1, round(np.linalg.norm(e) * 6)) for e in (du, dv))
        pan = face_panels(o, du, dv, nu, nv, nn)
        (hull if k < 3 else sheet).extend(pan)
    m = surface(np.array(hull + sheet), symmetry="xz_and_yz")
    body = {"mesh": part(m, 1, len(hull)), "dipoles": [{"mesh": part(m, len(hull) + 1, len(hull) + len(sheet))}]}
    return _problem(body, depth=4.0, options={"diffraction_formulation": "scattered"})


def lo_two_bodies():
    cyl = {"name": "cyl", "mesh": surface(cylinder_panels(1.0, 0.5, n_phi=16, n_z=3, n_r=3, quarter=False)),
           "mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.1],
           "radii_of_gyration": [0.6, 0.6, 0.7]}  # fmt: skip
    # the quadrant of the spheroid with its symmetry (a closed surface, not an open quarter)
    sph = {"name": "sph", "mesh": surface(spheroid_panels(2.0, 0.25, n_x=8, n_t=8), symmetry="xz_and_yz"),
           "position": [0.0, 3.0, 0.0], "mass": "displaced", "center_of_gravity": [0.0, 0.0, 0.0],
           "radii_of_gyration": [0.3, 1.0, 1.0]}  # fmt: skip
    return _problem([cyl, sph], options={"motion": True, "drift_momentum_conservation": True})


def ho_spline_cylinder_drift():
    m = 1000.0 * math.pi * 0.5
    body = {"mesh": spline_cylinder(n=4), "center_of_gravity": [0.0, 0.0, -0.2], "mass": m,
            "inertia_input": "components", "Ixx": m * 0.36,
            "Ixy": 0.0, "Ixz": 0.0, "Iyy": m * 0.36, "Iyz": 0.0, "Izz": m * 0.49,
            "inertia_origin": "center_of_gravity"}  # fmt: skip
    return _problem(body, field_points=POINTS,
                    options={"higher_order": True, "motion": True,
                             "drift_pressure_integration": True, "drift_momentum_conservation": True,
                             "field_pressure": True, "field_velocity": True,
                             "body_pressure": True, "body_velocity": True})  # fmt: skip


def ho_moonpool_lid():
    q = np.array([face_panels(o, du, dv, 1, 1, n)[0] for o, du, dv, n in MOONPOOL_FACES])
    nuv = [(max(1, round(np.linalg.norm(x[1] - x[0]) * 2)), max(1, round(np.linalg.norm(x[3] - x[0]) * 2))) for x in q]
    m = flat(q, [(a, b, 3, 3, 3, 4) for a, b in nuv], symmetry="xz_and_yz")
    # the lid from the last element of the body's mesh, placed with the body
    return _problem({"name": "b", "mesh": part(m, 1, 6)}, headings=(0.0,),
                    lids=[{"name": "l", "mesh": part(m, 7, 7), "placed_with": "b", "damping_factor": 0.5}],
                    options={"higher_order": True, "diffraction_formulation": "total"})  # fmt: skip


def ho_plate_sheet_finite():
    return _problem(plate_column(), omegas=2 * np.pi / np.array([2.0, 5.0]), depth=3.0,
                    options={"higher_order": True, "diffraction_formulation": "total"})  # fmt: skip


def ho_acv_pressure_drift():
    body = {**ho_acv(size=2.0), "mass": "displaced", "center_of_gravity": [0.0, 0.0, 0.0],
            "radii_of_gyration": [2.5, 5.0, 5.0]}
    return _problem(body, omegas=2 * np.pi / np.array([6.0]), headings=(150.0,),
                    options={"higher_order": True, "motion": True,
                             "drift_pressure_integration": True})  # fmt: skip


def lo_tank_haskind_pressure_drift():
    body = {**tank_vessel(1.0), "center_of_gravity": [0.0, 0.0, -0.2], "radii_of_gyration": [0.4, 0.8, 0.8],
            "excitation_method": "haskind"}  # fmt: skip
    return _problem(body, options={"motion": True, "drift_pressure_integration": True})


def lo_cylinder_control_surface_bidirectional():
    cs = {"method": "automatic", "radius": 2.0, "depth": 1.0, "panel_size": 0.5}
    return _problem(lo_free_cylinder(control_surface=cs), headings=(0.0, 30.0, 60.0),
                    options={"motion": True, "drift_control_surface": True, "drift_momentum_conservation": True,
                             "drift_bidirectional": True})


def lo_box_channel_reflect():
    m = lo_box_channel()
    m.environment.channel_reflect = 0.5
    return m


def ho_box_channel_interior():
    # the box with its interior patch off the centreline of a partly reflecting channel: motions,
    # the fields and the pressure drift
    q = box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True, lid=True)
    box = flat(q, [(*nv, 3, 3, 3, 4) for nv in ((4, 2), (2, 2), (4, 2), (4, 2))], symmetry="xz_and_yz")
    body = {"mesh": part(box, 1, 3), "interior_surface": {"method": "mesh", "mesh": part(box, 4, 4)},
            "position": [0.0, 0.6, 0.0],
            "mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.1], "radii_of_gyration": [0.4, 0.8, 0.8]}
    return _problem(body, headings=(0.0, 180.0), depth=3.0, field_points=POINTS * [1.0, 0.5, 1.0],
                    options={"higher_order": True, "domain": "channel", "channel_width": 5.5, "channel_reflect": 0.8,
                             "motion": True, "drift_pressure_integration": True, "field_pressure": True,
                             "field_velocity": True})  # fmt: skip


def lo_wavemaker():
    # a hinged paddle in the wall x = 0: radiation of its generalized mode, the field
    v = face_panels((0.0, 2.0, -2.0), (0.0, 1.0, 0.0), (0.0, 0.0, 2.0), 4, 8, (1.0, 0.0, 0.0))
    paddle = {"name": "paddle", "mesh": surface(np.array(v)), "modes": [],
              "generalized_modes": [{"name": "paddle", "pieces": [{"u": [[1.0, 0, 0, 1], [2.0, 0, 0, 0]]}]}]}
    return _problem(paddle, headings=(), depth=4.0, field_points=[[3.0, 2.5, 0.0], [5.0, 1.0, -1.0]],
                    options={"diffraction": False, "field_pressure": True, "domain": "orthogonal_walls"})  # fmt: skip


# the second order: two frequencies (their pairs), the quadratic QTFs and the potential QTF
W2 = np.array([1.2, 1.6])


def lo_cylinder_quadratic_qtf_newman():
    # the quadratic QTFs by pressure integration and over the control surface, Newman's
    # approximation of the three drifts, two headings and their pairs
    cs = {"method": "automatic", "radius": 2.0, "depth": 1.0, "panel_size": 0.5}
    drifts = {"drift_pressure_integration": True, "drift_control_surface": True, "drift_momentum_conservation": True}
    return _problem(lo_free_cylinder(control_surface=cs), omegas=W2, headings=(0.0, 30.0),
                    options={"motion": True, **drifts, "drift_bidirectional": True,
                             "qtf_quadratic_pressure_integration": True, "qtf_quadratic_control_surface": True,
                             "qtf_newman": True, "qtf_bidirectional": True})  # fmt: skip


def lo_cylinder_potential_qtf():
    # the potential QTF of the floating cylinder by both methods, the free surface to R = 3 in
    # panels of 0.4 and one quadrature annulus beyond it
    m = _problem(lo_cosine_cylinder(), omegas=W2, headings=(0.0,), depth=3.0,
                 options={"motion": True, "qtf_potential_direct": True, "qtf_potential_indirect": True})  # fmt: skip
    m.second_order.update(free_surface_inner_radius=3.0, free_surface_panel_size=0.4, quadrature_annuli=1)
    return m


def lo_cylinder_potential_qtf_wall():
    # the fixed cylinder by an x wall: the potential QTF of both methods with the wall's images
    m = _problem(lo_cosine_cylinder(fixed=True, position=[2.0, 0.0, 0.0]), omegas=W2, headings=(30.0,), depth=3.0,
                 options={"domain": "x_wall", "qtf_potential_direct": True,
                          "qtf_potential_indirect": True})  # fmt: skip
    m.second_order.update(free_surface_inner_radius=6.5, free_surface_panel_size=0.6)
    return m


def lo_flexible_cylinder_second_order():
    # the generalized-mode second order: the drift, the quadratic QTF, Newman's approximation and
    # the potential QTF of both methods of the cylinder with a bulging bottom and bending sides
    m = _problem(lo_flexible_cylinder(8), omegas=W2, headings=(0.0,), depth=3.0,
                 options={"motion": True, "drift_pressure_integration": True,
                          "qtf_quadratic_pressure_integration": True,
                          "qtf_newman": True,
                          "qtf_potential_direct": True, "qtf_potential_indirect": True})  # fmt: skip
    m.second_order.update(free_surface_inner_radius=3.0, free_surface_panel_size=0.4)
    return m


def lo_moonpool_plate():
    # the moonpool's free surface as a light piston, a body of a plate only, beside the hull
    hull = {"name": "hull", "mesh": moonpool_mesh(), "modes": ["heave", "pitch"], "mass": "displaced",
            "center_of_gravity": [0.0, 0.0, -0.2], "radii_of_gyration": [1.2, 1.2, 1.4]}  # fmt: skip
    piston = {"name": "piston", "modes": ["heave"], "free_surface_plates": [{"mesh": moonpool_lid_mesh()}],
              "mass": 50.0, "center_of_gravity": [0.0, 0.0, 0.0], "radii_of_gyration": [0.5, 0.5, 0.5]}  # fmt: skip
    return _problem([hull, piston], field_points=POINTS,
                    options={"motion": True, "field_velocity": True, "body_pressure": True,
                             "body_velocity": True})  # fmt: skip


def ho_barge_plate():
    # the higher order: the moonpool's free surface (element 7) as a plate of the barge
    mb = moonpool_barge()
    body = {"name": "barge", "mesh": part(mb, 1, 6), "modes": ["heave", "pitch"],
            "free_surface_plates": [{"mesh": part(mb, 7, 7)}]}  # fmt: skip
    return _problem(body, omegas=np.r_[0.0, W], headings=(0.0,), options={"higher_order": True, "body_pressure": True})


def lo_plate_drift():
    # the momentum drift with a plate and Newman's approximation of it: the cover of the moonpool
    # hull, moving with it
    hull = {"name": "hull", "mesh": moonpool_mesh(), "modes": ["surge", "heave", "pitch"], "mass": "displaced",
            "center_of_gravity": [0.0, 0.0, -0.2], "radii_of_gyration": [1.2, 1.2, 1.4],
            "free_surface_plates": [{"mesh": moonpool_lid_mesh()}]}  # fmt: skip
    return _problem(hull, options={"motion": True, "drift_momentum_conservation": True, "qtf_newman": True})


def lo_plate_channel_tank():
    # plates (mats of a mesh of their own) beside a vessel with an internal tank, in a channel
    mat = surface(face_panels((1.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.5, 0.0), 4, 2, (0, 0, -1)),
                  symmetry="xz_and_yz")  # fmt: skip
    vessel = {**tank_vessel(), "name": "vessel", "position": [0.0, 1.0, 0.0], "center_of_gravity": [0.0, 0.0, -0.2],
              "radii_of_gyration": [0.4, 0.5, 0.5], "free_surface_plates": [{"mesh": mat}]}  # fmt: skip
    return _problem(vessel, headings=(0.0, 180.0), depth=6.0,
                    options={"motion": True, "domain": "channel", "channel_width": 8.0})


def ho_barge_plate_drift():
    # the higher-order Kochin function of a plate: the barge with its cover, floating, momentum drift
    mb = moonpool_barge()
    body = {"name": "barge", "mesh": part(mb, 1, 6), "modes": ["surge", "heave", "pitch"], "mass": "displaced",
            "center_of_gravity": [0.0, 0.0, -0.3], "radii_of_gyration": [1.2, 1.2, 1.4],
            "free_surface_plates": [{"mesh": part(mb, 7, 7)}]}  # fmt: skip
    return _problem(body, options={"higher_order": True, "motion": True, "drift_momentum_conservation": True})


def ho_barge_plate_infinite():
    # omega = inf with plates: the barge's cover as a plate of the barge (at omega = inf a sheet of the
    # odd-image problem), both limits and ordinary frequencies
    mb = moonpool_barge()
    barge = {"name": "barge", "mesh": part(mb, 1, 6), "modes": ["heave", "pitch"],
             "free_surface_plates": [{"mesh": part(mb, 7, 7)}]}  # fmt: skip
    return _problem(barge, omegas=np.r_[0.0, W, math.inf], headings=(0.0,), options={"higher_order": True})


def lo_plate_pressure_qtf():
    # the pressure drift and the quadratic QTFs with plates whose edges lie on walls: the floating
    # moonpool hull with a massless piston body on its moonpool (the band under a sliding plate)
    hull = {"name": "hull", "mesh": moonpool_mesh(), "modes": ["surge", "heave", "pitch"], "mass": "displaced",
            "center_of_gravity": [0.0, 0.0, -0.2], "radii_of_gyration": [1.2, 1.2, 1.4]}  # fmt: skip
    piston = {"name": "piston", "modes": ["heave"], "mass": 0.0, "center_of_gravity": [0.0, 0.0, 0.0],
              "radii_of_gyration": [0.3, 0.3, 0.3], "free_surface_plates": [{"mesh": moonpool_lid_mesh()}]}  # fmt: skip
    return _problem([hull, piston], options={"motion": True, "drift_pressure_integration": True,
                                             "qtf_quadratic_pressure_integration": True})  # fmt: skip


def lo_plate_potential_qtf():
    # the potential QTF of both methods with plates whose edges lie on walls: the floating moonpool
    # hull with a massless piston body on its moonpool (the free surface ending at the hull's outer
    # waterline, the line integral of the moonpool's waterline with the piston's sliding)
    hull = {"name": "hull", "mesh": moonpool_mesh(), "modes": ["surge", "heave", "pitch"], "mass": "displaced",
            "center_of_gravity": [0.0, 0.0, -0.2], "radii_of_gyration": [1.2, 1.2, 1.4]}  # fmt: skip
    piston = {"name": "piston", "modes": ["heave"], "mass": 0.0, "center_of_gravity": [0.0, 0.0, 0.0],
              "radii_of_gyration": [0.3, 0.3, 0.3], "free_surface_plates": [{"mesh": moonpool_lid_mesh()}]}  # fmt: skip
    m = _problem([hull, piston], omegas=W2, headings=(0.0,), depth=3.0,
                 options={"motion": True, "qtf_potential_direct": True, "qtf_potential_indirect": True})  # fmt: skip
    m.second_order.update(free_surface_inner_radius=6.0, free_surface_panel_size=0.5)
    return m


def lo_plate_control_surface():
    # the control-surface drift and its quadratic QTFs with a plate of the body: the floating
    # moonpool hull with its cover (its edge on the moonpool wall), a cylinder about the hull
    hull = {"name": "hull", "mesh": moonpool_mesh(), "modes": ["surge", "heave", "pitch"], "mass": "displaced",
            "center_of_gravity": [0.0, 0.0, -0.2], "radii_of_gyration": [1.2, 1.2, 1.4],
            "free_surface_plates": [{"mesh": moonpool_lid_mesh()}],
            "control_surface": {"method": "cylinder", "radius": 3.0, "depth": 1.5, "inner_radius": 2.0,
                                "panel_size": 0.5}}  # fmt: skip
    return _problem(hull, options={"motion": True, "drift_control_surface": True,
                                   "qtf_quadratic_pressure_integration": True,
                                   "qtf_quadratic_control_surface": True})  # fmt: skip


def lo_tank_plate():
    # a plate of a tank's free surface: the tank vessel's tank half covered (elements of the body's
    # mesh, the central half of its free surface), floating, ordinary frequencies and omega = inf
    b = tank_vessel()
    mesh, (first, last) = b["tanks"][0]["mesh"]
    zt = mesh["vertices"][first - 1 : last, :, 2].max()
    lx, ly = mesh["vertices"][first - 1 : last, :, 0].max(), mesh["vertices"][first - 1 : last, :, 1].max()
    cover = np.array(face_panels((0.0, 0.0, zt), (0.5 * lx, 0.0, 0.0), (0.0, ly, 0.0), 4, 4, (0, 0, -1)))
    mesh = {**mesh, "vertices": np.concatenate([mesh["vertices"], cover])}
    tank = {**b["tanks"][0], "mesh": (mesh, (first, last)),
            "free_surface_plates": [{"mesh": (mesh, (last + 1, last + len(cover)))}]}  # fmt: skip
    vessel = {**b, "name": "vessel", "mesh": (mesh, b["mesh"][1]), "tanks": [tank],
              "center_of_gravity": [0.0, 0.0, -0.2], "radii_of_gyration": [0.4, 0.5, 0.5]}  # fmt: skip
    return _problem(vessel, omegas=np.r_[W, math.inf], headings=(0.0,), options={"motion": True})


PROBLEMS = {f.__name__: f for f in (
    lo_cylinder_drift, lo_cylinder_limits, lo_cylinder_field_body_points, lo_box_walls_finite, lo_box_channel,
    lo_cylinder_interior, lo_moonpool_lid, lo_acv_two_modes, lo_plate_sheet, lo_two_bodies, ho_spline_cylinder_drift,
    ho_moonpool_lid, ho_plate_sheet_finite, ho_acv_pressure_drift, lo_tank_haskind_pressure_drift,
    lo_cylinder_control_surface_bidirectional, lo_box_channel_reflect, lo_wavemaker, ho_box_channel_interior,
    lo_cylinder_quadratic_qtf_newman, lo_cylinder_potential_qtf, lo_cylinder_potential_qtf_wall,
    lo_flexible_cylinder_second_order, lo_moonpool_plate, ho_barge_plate, lo_plate_drift, lo_plate_channel_tank,
    ho_barge_plate_drift, ho_barge_plate_infinite, lo_plate_pressure_qtf, lo_plate_potential_qtf,
    lo_plate_control_surface, lo_tank_plate,
)} | {f"feature_{name}": build for name, build in FEATURES.items()}  # fmt: skip


def outputs(r: hb.Results) -> dict:
    """Every quantity of a result by its engine name, the hydrostatics included."""
    out = {}
    for name in hb.quantities():
        v = r.quantity(name)
        if v is not None:
            out[name] = np.asarray(v)
    return out


def same_outputs(a, b):
    """Asserts that two results (hb.Results, or their ``outputs``) have the same quantities, each
    with the same dtype, shape and bytes."""
    oa, ob = (x if isinstance(x, dict) else outputs(x) for x in (a, b))
    assert sorted(oa) == sorted(ob)
    for k in oa:
        x, y = np.asarray(oa[k]), np.asarray(ob[k])
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), k


def solve(name: str) -> dict:
    import warnings

    with warnings.catch_warnings():
        # the one note a baseline problem has: the pressure drift over the wetted surface only;
        # any other warning still fails
        warnings.filterwarnings("ignore", message=".*integrated over the wetted surface only")
        return outputs(PROBLEMS[name]().solve())
