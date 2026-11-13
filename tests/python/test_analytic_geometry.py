"""Analytic higher-order geometry (reference/inputs-meshes §2).

The engine evaluates the exact maps: the water-plane restoring of every shape equals
its closed form to rounding (the volumes: test_queries_by_owner.py); the patch
points give the corners of the shapes' definitions; a GDF file
with IGDEF < 0 reads into the named shape of its code, also by panel_size; the pressure drift with a
vanishing mapping slope at the waterline is rejected (input rules). The convergence of the uniform
and nonuniform mappings to the same added mass is validation, run when the method changes, not a
test."""

import math

import numpy as np
import pytest

import hydrobem as hb

from problems import RHO, G, analytic, gdf, model

DISC = (4, 4, 3, 3, 3, 4)


def hydro(mesh):
    m = model({"mesh": mesh}, [], (), options={"higher_order": True, "radiation": False, "diffraction": False})
    return m.bodies[0].hydrostatics()


def analytic_mesh(shape, rows=DISC, **kw):
    """The mesh object of a named analytic shape in a model of its own."""
    return hb.Model().meshes.analytic("analytic", shape, rows, **kw)


NONUNIFORM = {"mapping": "nonuniform"}


@pytest.mark.parametrize(
    "shape, dims, volume, c33, c44",
    [
        ("circular_cylinder", {"radius": 1.0, "draft": 0.5}, math.pi * 0.5, math.pi,
         math.pi / 4 - math.pi * 0.5 * 0.25),
        ("circular_cylinder", {"radius": 1.0, "draft": 0.5, **NONUNIFORM}, math.pi * 0.5, math.pi,
         math.pi / 4 - math.pi * 0.5 * 0.25),
        ("elliptic_cylinder", {"a": 2.0, "b": 1.0, "draft": 0.5}, 2 * math.pi * 0.5, 2 * math.pi,
         math.pi * 2 / 4 - 2 * math.pi * 0.5 * 0.25),
        ("sphere", {"radius": 1.0, **NONUNIFORM}, 2 * math.pi / 3, math.pi, math.pi / 4 - 2 * math.pi / 3 * 3 / 8),
        ("ellipsoid", {"a": 2.0, "b": 0.5, "c": 0.25}, math.pi / 6, math.pi,
         math.pi * 2 * 0.125 / 4 - math.pi / 6 * 0.25 * 3 / 8),
        ("barge", {"half_length": 2.0, "half_beam": 1.0, "draft": 0.5}, 4.0, 8.0, 8.0 / 3 - 4.0 * 0.25),
        ("moonpool_cylinder", {"radius": 6.0, "draft": 8.0, "moonpool_radius": 2.5, **NONUNIFORM},
         math.pi * (36 - 6.25) * 8, math.pi * (36 - 6.25),
         math.pi / 4 * (6**4 - 2.5**4) - math.pi * (36 - 6.25) * 8 * 4),
    ],
)  # fmt: skip
def test_volume_and_restoring_are_exact(shape, dims, volume, c33, c44):
    # c44 / (rho g) = I_x of the water plane + V z_B (z_B = -T/2, -3c/8 for the half ellipsoid)
    # (the volume column is the closed form in the c44 terms; the volumes are checked with the geometry)
    h = hydro(analytic(shape, DISC, **dims))
    c = np.asarray(h.stiffness_rho_g) / (RHO * G)
    np.testing.assert_allclose(c[2, 2], c33, rtol=1e-12)
    np.testing.assert_allclose(c[3, 3], c44, rtol=1e-11, atol=1e-10 * c33)  # 0 for the hemisphere


def test_patch_corners_follow_the_definitions():
    mesh = analytic_mesh("circular_cylinder", radius=1.0, draft=0.5, parts=["side", "bottom", "waterplane"],
                         **NONUNIFORM)  # elements 1, 2, 3: side, bottom, interior  # fmt: skip

    def point(record, u, v):
        return mesh.patch_points(record, [[u, v]]).points[0]

    np.testing.assert_allclose(point(1, -1, -1), (1, 0, 0), atol=1e-15)  # waterline at theta = 0
    np.testing.assert_allclose(point(1, 1, 1), (0, 1, -0.5), atol=1e-15)
    np.testing.assert_allclose(point(1, -1, 0), (1, 0, -0.25), atol=1e-15)  # 3t^2 - 2t^3 at t = 1/2
    np.testing.assert_allclose(point(2, -1, 0), (0.75, 0, -0.5), atol=1e-15)  # 1 - t^2
    np.testing.assert_allclose(point(3, -1, 0), (0.75, 0, 0), atol=1e-15)  # 2t - t^2


def test_gdf_reader_and_panel_size(tmp_path):
    path = tmp_path / "cyl.gdf"
    path.write_text("cylinder\n1.0 9.80665\n1 1\n2 -1\n2  NLINES\n1.0 0.5  radius draft\n0\n", encoding="ascii")
    m = hb.Model().meshes.load("cyl", path, higher_order=True, discretization=[DISC, DISC])
    assert m.kind == "analytic_patches" and len(m.patches) == 2
    np.testing.assert_allclose(hydro(gdf(path, higher_order=True, discretization=[DISC, DISC])).volumes,
                               math.pi * 0.5, rtol=1e-12)  # fmt: skip
    ms = hb.Model().meshes.load("cyl", path, higher_order=True, panel_size=0.2)
    # side: quarter circle 1.5708 (8 panels), depth 0.5 (3); bottom: arc 1.5708, radius 1 (6)
    assert [(int(d[0]), int(d[1])) for d in ms.discretization] == [(8, 3), (8, 6)]


def test_inputs_are_checked():
    for shape, kw, match in (
        ("cone", {"radius": 1.0}, "shape must be one of circular_cylinder, elliptic_cylinder"),
        ("circular_cylinder", {"radius": 1.0}, "circular_cylinder needs radius, draft"),
        ("circular_cylinder", {"radius": 1.0, "draft": -0.5}, "draft must be positive and finite"),
        ("circular_cylinder", {"radius": 1.0, "draft": 0.5, "depth": 1.0}, "unknown dimension 'depth'"),
        ("circular_cylinder", {"radius": 1.0, "draft": 0.5, "parts": ["side", "top"]}, "unknown part 'top'"),
        ("circular_cylinder", {"radius": 1.0, "draft": 0.5, "parts": ["side", "side"]}, "the part side is given twice"),
        ("circular_cylinder", {"radius": 1.0, "draft": 0.5, "mapping": "cosine"}, "mapping must be one of"),
        ("barge", {"half_length": 1.0, "half_beam": 1.0, "draft": 1.0, **NONUNIFORM}, "nonuniform mapping is not"),
        ("moonpool_cylinder", {"radius": 1.0, "draft": 1.0, "moonpool_radius": 1.0}, "smaller than radius"),
    ):  # fmt: skip
        with pytest.raises(ValueError, match=match):
            analytic_mesh(shape, **kw)
    # the parts in the order given
    ps = analytic_mesh("circular_cylinder", radius=1.0, draft=0.5, parts=["bottom", "side"]).patches
    assert [list(p.p) for p in ps] == [[1.0, -0.5, 0.0, -0.5], [1.0, 0.0, 1.0, -0.5]]
    body = {"mesh": analytic("circular_cylinder", DISC, radius=1.0, draft=0.5, **NONUNIFORM), "mass": "displaced",
            "center_of_gravity": [0, 0, -0.2],
            "radii_of_gyration": [0.6, 0.6, 0.7]}  # fmt: skip
    with pytest.raises(hb.UnsupportedProblemError, match="nonuniform analytic mapping"):
        opts = {"higher_order": True, "motion": True, "drift_pressure_integration": True}
        model(body, [1.0], [0.0], options=opts).route()
    # either end of the profile, and the waterline of the placed body: the side patch
    # turned round (its waterline at v = 1, dP = 0 there), and the body raised by 0.1 m with
    # its mesh lowered as much
    for side, position in (({"p": [1.0, -0.5, 1.0, 0.0], "profile": "quadratic_end", "theta0": math.pi / 2,
                             "theta1": 0.0}, 0.0),
                           ({"p": [1.0, 0.1, 1.0, -0.4]}, -0.1)):  # fmt: skip
        m = hb.Model()
        mesh = m.meshes.analytic("c", "circular_cylinder", DISC, radius=1.0, draft=0.5, **NONUNIFORM)
        mesh.patches[0].update(**side)
        if position:
            mesh.patches[1].p = [1.0, -0.4, 0.0, -0.4]
        m.bodies.add("c", mesh="c", position=[0.0, 0.0, position], modes=[])
        m.environment.omegas = [1.0]
        m.environment.headings = [0.0]
        m.options.higher_order = True
        m.second_order.drift_pressure_integration = True
        with pytest.raises(hb.UnsupportedProblemError, match="nonuniform analytic mapping"):
            m.route()


def test_cylinder_with_dipole_discs_and_the_planar_rule():
    """The cylinder with dipole discs, the cylinder (with or without its moonpool) and one annular
    disc per entry; a dipole patch must be planar (a curved sheet is rejected)."""
    discs = [(0.625, 0.5, 1.0), (0.25, 0.125, 1.0)]
    ps = analytic_mesh("cylinder_with_dipole_discs", radius=0.5, draft=1.0, moonpool_radius=0.25, discs=discs).patches
    assert len(ps) == 5 and list(ps[3].p) == [0.625, -1.0, 0.5, -1.0] and list(ps[4].p) == [0.25, -1.0, 0.125, -1.0]
    assert len(analytic_mesh("cylinder_with_dipole_discs", radius=1.0, draft=1.0, discs=[(1.3, 0.2, 1.5)]).patches) == 3
    with pytest.raises(ValueError, match="the parts are fixed"):
        analytic_mesh("cylinder_with_dipole_discs", radius=1.0, draft=1.0, parts=["side"])
    with pytest.raises(ValueError, match="the outer radius must exceed the inner radius"):
        analytic_mesh("cylinder_with_dipole_discs", radius=1.0, draft=1.0, discs=[(0.2, 1.3, 1.5)])
    # a disc not below the bottom lies in the water: outside the hull (a heave plate at the bottom)
    # or inside the moonpool; across the wall, in the hull or on the bottom it is refused
    assert len(analytic_mesh("cylinder_with_dipole_discs", radius=1.0, draft=1.0, discs=[(2.0, 1.0, 1.0)]).patches) == 3
    for disc, moon in (((1.3, 0.5, 0.5), {}), ((0.8, 0.2, 1.0), {}), ((0.5, 0.2, 0.5), {"moonpool_radius": 0.25})):
        with pytest.raises(ValueError, match="it cuts the hull"):
            analytic_mesh("cylinder_with_dipole_discs", radius=1.0, draft=1.0, discs=[disc], **moon)
    # the moonpool inside the hull, as for moonpool_cylinder
    with pytest.raises(ValueError, match="moonpool_radius must be smaller than radius"):
        analytic_mesh("cylinder_with_dipole_discs", radius=1.0, draft=1.0, moonpool_radius=1.0, discs=discs)
    d = (2, 2, 3, 3, 3, 4)
    side = analytic("circular_cylinder", [d, d], radius=1.0, draft=1.0)
    body = {"mesh": (side, (2, 2)), "dipoles": [{"mesh": (side, (1, 1))}]}  # the curved side as a sheet
    with pytest.raises(hb.UnsupportedProblemError, match="planar"):
        model(body, [1.0], [0.0], options={"higher_order": True}).route()


# every analytic code of a GDF file and its NPATCH: the data lines, the named shape and its keywords
GDF_CODES = [
    (-1, 2, ["1.0 0.5", "1"], "circular_cylinder", {"radius": 1.0, "draft": 0.5, **NONUNIFORM}),
    (-1, 3, ["1.0 0.5", "0"], "circular_cylinder",
     {"radius": 1.0, "draft": 0.5, "parts": ["side", "bottom", "waterplane"]}),
    (-2, 1, ["2.0 1.0 0.5", "0"], "elliptic_cylinder", {"a": 2.0, "b": 1.0, "draft": 0.5, "parts": ["side"]}),
    (-3, 2, ["1.0", "1"], "sphere", {"radius": 1.0, "parts": ["hull", "waterplane"], **NONUNIFORM}),
    (-4, 1, ["2.0 0.5 0.25"], "ellipsoid", {"a": 2.0, "b": 0.5, "c": 0.25}),
    (-5, 1, ["2.0 1.0 0.5"], "barge", {"half_length": 2.0, "half_beam": 1.0, "draft": 0.5, "parts": ["bottom"]}),
    (-5, 4, ["2.0 1.0 0.5"], "barge", {"half_length": 2.0, "half_beam": 1.0, "draft": 0.5,
                                       "parts": ["end", "side", "bottom", "waterplane"]}),
    (-7, 4, ["6.0 8.0 2.5", "1"], "moonpool_cylinder",
     {"radius": 6.0, "draft": 8.0, "moonpool_radius": 2.5,
      "parts": ["side", "bottom", "moonpool_wall", "moonpool_surface"], **NONUNIFORM}),
    (-34, 5, ["1 2", "0.5 1.0 0.25", "0.625 0.5 1.0", "0.25 0.125 1.0"], "cylinder_with_dipole_discs",
     {"radius": 0.5, "draft": 1.0, "moonpool_radius": 0.25, "discs": [(0.625, 0.5, 1.0), (0.25, 0.125, 1.0)]}),
]  # fmt: skip


@pytest.mark.parametrize("igdef, npatch, lines, shape, kw", GDF_CODES)
def test_gdf_codes_are_the_named_shapes(tmp_path, igdef, npatch, lines, shape, kw):
    # the reader translates IGDEF, NPATCH and the data lines into the named shape: the same patches
    path = tmp_path / "shape.gdf"
    text = ["shape", "1.0 9.80665", "1 1", f"{npatch} {igdef}", f"{len(lines)}  NLINES", *lines]
    path.write_text("\n".join(text) + "\n", encoding="ascii")
    read = hb.Model().meshes.load("s", path, higher_order=True, discretization=[DISC])
    named = analytic_mesh(shape, **kw)
    items = ("map", "profile", "a", "b", "theta0", "theta1")
    assert len(read.patches) == len(named.patches) == npatch
    for a, b in zip(read.patches, named.patches, strict=True):
        assert all(getattr(a, k) == getattr(b, k) for k in items)
        assert list(a.p) == list(b.p)
        if a.map == "bilinear":
            np.testing.assert_array_equal(a.vertices, b.vertices)
