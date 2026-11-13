"""The classification of points by the field-point rules and the geometry of a mesh
without a problem.

Every verdict appears for a point of a known kind: in the fluid, above the free surface, below
the sea bed, behind a wall, outside a channel, on and inside a waterline, on a lid, in a tank that
does not exist, not finite, inside a body below the free surface (the solid angle of its
closed wetted surface, for a cylinder, the wall of a moonpool, the pontoon of a TLP, a submerged
box, a B-spline body; a thin sheet encloses nothing), near a low-order panel (a warning). A
point with a failing verdict, given as the problem's field point, is rejected with the message of
that rule, and the problems with valid field points classify them as fluid. The geometry of
a mesh alone is the volume and waterplane of the hydrostatics of that body at the origin."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import TANK_POINTS, plate_column, spline_cylinder, surface
from meshes import cylinder_panels, face_panels, moonpool_panels, tlp_panels
from problems import model, two_tank_vessels
from regression_problems import PROBLEMS

MESSAGE = {
    "above_free_surface": "lies above the free surface",
    "below_sea_bed": "lies on or below the sea bed",
    "outside_domain": "is outside the fluid domain",
    "on_body": "lies on bodies/",
    "inside_waterplane": "lies inside the water plane",
    "on_lid": "lies on lids/",
    "no_tank": "does not exist",
    "inside_body": "lies inside bodies/",
}


def _cylinder_problem(**kw):
    m = surface(cylinder_panels(1.0, 0.5, n_phi=8, n_z=2, n_r=2, quarter=True), symmetry="xz_and_yz")
    return model({"name": "cyl", "mesh": m}, [1.5], [0.0], **kw)


def _check(m, points, expected):
    c = m.classify_points(points)
    assert list(c.status) == expected
    for i, s in enumerate(expected):
        if s not in MESSAGE:
            continue
        p = m.copy()
        p.first_order.field_points = np.asarray(points[i:i + 1], dtype=float)
        p.first_order.field_pressure = True
        with pytest.raises(ValueError, match=MESSAGE[s]):
            p.route()
    return c


def test_open_sea_and_finite_depth():
    pts = [[2.0, 0.0, -0.3], [3.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.5], [math.nan, 0.0, 0.0]]
    c = _check(_cylinder_problem(), pts, ["fluid", "fluid", "inside_waterplane", "on_body", "above_free_surface",
                                          "not_finite"])  # fmt: skip
    assert c.body.tolist() == [-1, -1, 0, 0, -1, -1] and c.lid.tolist() == [-1] * 6 and c.tank.tolist() == [-1] * 6
    deep = _cylinder_problem(environment={"depth": 2.0})
    _check(deep, [[3.0, 0.0, -1.0], [3.0, 0.0, -2.5]], ["fluid", "below_sea_bed"])
    # tank: a tank of the model (not another object)
    with pytest.raises(TypeError, match="a tank of this model"):
        deep.classify_points([[3.0, 0.0, -1.0]], deep.bodies["cyl"])


def test_walls_channel_and_lids():
    walls = PROBLEMS["lo_box_walls_finite"]()
    _check(walls, [[3.0, 3.0, -0.5], [-1.0, 3.0, -0.5], [3.0, -1.0, -0.5]],
           ["fluid", "outside_domain", "outside_domain"])
    channel = PROBLEMS["lo_box_channel"]()
    _check(channel, [[0.0, 2.0, -0.5], [0.0, 3.5, -0.5]], ["fluid", "outside_domain"])
    lid = PROBLEMS["lo_moonpool_lid"]()
    c = _check(lid, [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], ["on_lid", "fluid"])
    assert c.lid.tolist() == [0, -1] and c.body.tolist() == [-1, -1]


@pytest.mark.parametrize("name", ["lo_cylinder_field_body_points", "lo_box_walls_finite", "lo_box_channel"])
def test_the_field_points_of_valid_problems_are_fluid(name):
    m = PROBLEMS[name]()
    assert set(m.classify_points(m.first_order.field_points).status) == {"fluid"}


@pytest.mark.parametrize("higher_order", [False, True])
def test_mesh_geometry_equals_the_hydrostatics_of_a_body_at_the_origin(higher_order):
    # the geometry of a mesh is the volume and waterplane of the hydrostatics of a body at the origin
    # (R = 0, no trim), the same quadrature, bit for bit; its volumes are those of the quality queries
    mesh = (spline_cylinder(n=4) if higher_order else
            surface(cylinder_panels(1.0, 0.5, n_phi=8, n_z=2, n_r=2, quarter=True), symmetry="xz_and_yz"))
    m = model({"mesh": mesh, "mass": "displaced"}, [], (), environment={"density": 1025.0, "gravity": 9.81},
              options={"radiation": False, "diffraction": False, "higher_order": higher_order})  # fmt: skip
    mo = m.meshes[m.bodies[0].mesh]
    g, h = mo.geometry(), m.bodies[0].hydrostatics()
    assert g.volumes.tobytes() == h.volumes.tobytes() and g.volume == h.volume
    assert g.center_of_volume.tobytes() == h.center_of_buoyancy.tobytes()
    assert g.waterplane_area == h.waterplane_area and g.waterplane_centroid.tobytes() == h.waterplane_centroid.tobytes()
    assert h.volume == pytest.approx(h.volumes.mean(), rel=1e-15)
    assert h.displaced_mass == 1025.0 * h.volumes[0] and h.mass == h.displaced_mass
    # C33 = rho g A, C44 = rho g (I_yy of the waterplane + V z_B): the moments of the geometry
    c = h.stiffness_rho_g / (1025.0 * 9.81)
    assert c[2, 2] == pytest.approx(g.waterplane_area, rel=1e-14)
    assert c[3, 3] == pytest.approx(g.waterplane_moments[1] + g.volumes[2] * g.center_of_volume[2], rel=1e-12)
    for q in (mo.quality(), m.bodies[0].mesh_quality(), m.mesh_quality()):
        assert q.volumes[0].tobytes() == g.volumes.tobytes()


def test_points_inside_bodies():
    # the solid angle of the wetted surface closed by its mirror image in z = 0
    _check(_cylinder_problem(), [[0.3, 0.2, -0.25], [-0.5, -0.5, -0.1], [2.0, 0.0, -0.25], [0.0, 0.0, -1.5]],
           ["inside_body", "inside_body", "fluid", "fluid"])
    moon = model({"mesh": surface(moonpool_panels(), symmetry="xz_and_yz")}, [1.5], [0.0])
    c = _check(moon, [[0.0, 0.0, -0.5], [1.5, 0.0, -0.5], [0.0, -1.5, -0.9], [2.5, 0.0, -0.5], [0.5, 0.5, -1.5]],
               ["fluid", "inside_body", "inside_body", "fluid", "fluid"])
    assert c.body.tolist() == [-1, 0, 0, -1, -1]
    # a TLP: inside a pontoon and above it (the concave space between the columns)
    tlp = model({"mesh": surface(tlp_panels(), symmetry="xz_and_yz")}, [0.5], [0.0])
    _check(tlp, [[20.0, 43.125, -30.0], [20.0, 43.125, -3.0], [43.125, 43.125, -20.0], [20.0, 20.0, -30.0]],
           ["inside_body", "fluid", "inside_body", "fluid"])
    # a closed box below the free surface, which needs no mirror image
    faces = [((0, 0, -3), (1, 0, 0), (0, 1, 0), (0, 0, -1)), ((0, 0, -2), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
             ((0, 0, -3), (1, 0, 0), (0, 0, 1), (0, -1, 0)), ((0, 1, -3), (1, 0, 0), (0, 0, 1), (0, 1, 0)),
             ((0, 0, -3), (0, 1, 0), (0, 0, 1), (-1, 0, 0)), ((1, 0, -3), (0, 1, 0), (0, 0, 1), (1, 0, 0))]
    box = np.array([q for f in faces for q in face_panels(*f[:3], 2, 2, f[3])])
    sub = model({"mesh": surface(box), "position": [3.0, 0.0, 0.0]}, [1.5], [0.0])
    _check(sub, [[3.5, 0.5, -2.5], [3.5, 0.5, -1.0], [3.5, 0.5, -4.0]], ["inside_body", "fluid", "fluid"])


def test_inside_higher_order_bodies_and_sheets():
    cyl = model({"mesh": spline_cylinder(n=4)}, [1.5], [0.0], options={"higher_order": True})
    _check(cyl, [[0.3, 0.3, -0.25], [1.5, 0.0, -0.25]], ["inside_body", "fluid"])
    # the heave plate at z = -1 beside the column is a thin sheet: it encloses nothing
    plate = model(plate_column(), [1.5], [0.0], options={"higher_order": True})
    _check(plate, [[0.5, 0.5, -0.5], [1.3, 0.5, -0.5], [1.3, 0.5, -1.2]], ["inside_body", "fluid", "fluid"])


def test_points_near_low_order_panels_warn():
    # a point closer to a panel than the panel's diameter (the side panels 0.2 wide and 0.25 high,
    # diameter 0.32) is near_body and warns (code 6, the error of constant panels there); 0.05 off the
    # side is within it, 2 off is not
    p = _cylinder_problem()
    c = p.classify_points([[1.05, 0.0, -0.25], [3.0, 0.0, -0.25]])
    assert c.status == ("near_body", "fluid") and c.body.tolist() == [0, -1]
    q = p.copy()
    q.first_order.field_points = [[1.05, 0.0, -0.25]]
    q.first_order.field_pressure = True
    with pytest.warns(hb.HydroBEMWarning, match="within a panel diameter") as w:
        q.solve()
    assert any(getattr(x.message, "code", 0) == 6 for x in w)


def test_points_must_have_three_columns():
    m = model({"mesh": surface(cylinder_panels(1.0, 0.5), symmetry="xz_and_yz")}, [1.0], [0.0])
    assert len(m.classify_points([3.0, 0.0, -0.1]).status) == 1  # one point
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        m.classify_points(np.zeros((3, 2)))


def test_a_point_in_a_tank_names_its_body_and_the_tank_of_that_body():
    # the zero-based tank of its body, as the results' bodies[b].tanks[t] (not a number over the
    # tanks of every body): the tank of the second body is its tank 0
    m = two_tank_vessels()
    c = m.classify_points(TANK_POINTS[[1]] - [6.0, 0.0, 0.0], m.bodies["b"].tanks[0])
    assert c.status == ("tank",) and c.tank.tolist() == [0] and c.body.tolist() == [1]
    r = m.solve()
    assert r.bodies["b"].tanks[0].points.shape == (1, 3)
