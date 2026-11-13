"""Rational B-spline patches (NURBS; reference/inputs-meshes §2).

The rational quarter circle (Piegl & Tiller 1997, §7.3) makes the cylinder and the hemisphere
exact: their points lie on the surface to rounding, the volume and the restoring are the closed
forms; the hemisphere's added mass and damping agree with Hulme (1982) to the precision of the
tables, and with the analytic sphere. The weights are kept with the mesh; a GDF file,
which has none, is not written for a rational patch; invalid weights are rejected (input rules); a
rational mesh as elements or as meshes of their own gives the same bits (the same equations). The
convergence of the rational cylinder to the analytic cylinder as the potential discretization
refines is validation, run when the method changes, not a test. The evaluation
itself (the derivatives, the two evaluators, a common factor, weights 1) is tested in
crates/hydrobem/tests/problem.rs."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import spline
from meshes import QUARTER_CIRCLE, QUARTER_WEIGHTS, nurbs_cylinder_patches, nurbs_hemisphere_patches
from problems import RHO, G, add_mesh, analytic, hulme, model, solve, view

QS = ("added_mass", "radiation_damping", "excitation_diffraction")


def _mesh(patches, rows):
    m = hb.Model()
    return m, add_mesh(m, "n", spline(patches, rows, symmetry="xz_and_yz"))


def test_cylinder_and_hemisphere_are_exact():
    uv = np.random.default_rng(1).uniform(-1, 1, (40, 2))
    _, cyl = _mesh(nurbs_cylinder_patches(1.0, 0.5), (4, 2, 3, 3, 3, 4))
    x = np.asarray(cyl.patch_points(1, uv).points)
    assert np.abs(np.hypot(x[:, 0], x[:, 1]) - 1.0).max() <= 1e-15
    x = np.asarray(cyl.patch_points(2, uv).points)
    assert np.abs(x[:, 2] + 0.5).max() <= 1e-15 and np.hypot(x[:, 0], x[:, 1]).max() <= 1.0
    np.testing.assert_allclose(cyl.geometry().volumes, math.pi * 0.5, rtol=1e-12)  # the 12-point rule per span
    _, hemi = _mesh(nurbs_hemisphere_patches(1.0), (4, 4, 3, 3, 3, 4))
    x = np.asarray(hemi.patch_points(1, uv).points)
    assert np.abs(np.linalg.norm(x, axis=1) - 1.0).max() <= 1e-15 and x[:, 2].max() <= 1e-15
    np.testing.assert_allclose(hemi.geometry().volumes, 2 * math.pi / 3, rtol=1e-12)  # 1.5e-13
    m = model({"mesh": spline(nurbs_hemisphere_patches(1.0), (4, 4, 3, 3, 3, 4), symmetry="xz_and_yz")}, [], (),
              options={"higher_order": True, "radiation": False, "diffraction": False})  # fmt: skip
    c = np.asarray(m.bodies[0].hydrostatics().stiffness_rho_g) / (RHO * G)
    np.testing.assert_allclose(c[2, 2], math.pi, rtol=1e-12)  # the water plane pi a^2


def test_hemisphere_against_hulme_and_the_analytic_sphere():
    # heave A(0), B(0) and surge A(1), B(1) of Hulme (1982) at Ka = 0.5, 1, 2 (tests/data from
    # scripts/extract_hulme.py): within 3e-4 of the largest tabulated value (the tables have four
    # digits; 1.4e-4 found), as the analytic sphere of the same discretization (within 1e-4)
    ref = hulme()
    w = np.sqrt(ref[:, 0] * G)
    m0 = RHO * 2 / 3 * math.pi
    rows = (8, 8, 3, 3, 3, 4)
    vals = []
    for mesh in (spline(nurbs_hemisphere_patches(1.0), rows, symmetry="xz_and_yz"), analytic("sphere", rows,
                                                                                             radius=1.0)):
        r = solve({"mesh": mesh, "modes": ["surge", "heave"]}, w, options={"higher_order": True, "diffraction": False})
        A, B = r.added_mass, r.radiation_damping
        vals.append(np.c_[A[:, 2, 1] / m0, B[:, 2, 1] / (w * m0), A[:, 0, 0] / m0, B[:, 0, 0] / (w * m0)])
    err = np.abs(vals[0] - ref[:, 1:]).max(axis=0) / np.abs(ref[:, 1:]).max(axis=0)
    assert err.max() <= 3e-4, err
    assert np.abs(vals[0] - vals[1]).max() <= 1e-4 * np.abs(vals[1]).max()


def test_weights_are_kept_and_checked(tmp_path):
    m, cyl = _mesh(nurbs_cylinder_patches(1.0, 0.5), (4, 2, 3, 3, 3, 4))
    np.testing.assert_array_equal(cyl.patches[0].weights, [[1.0, math.sqrt(0.5), 1.0]] * 2)
    m.save(tmp_path / "n.hbm")
    back = hb.Model.load(tmp_path / "n.hbm").meshes["n"]
    np.testing.assert_array_equal(back.patches[1].weights, cyl.patches[1].weights)
    with pytest.raises(ValueError, match="a GDF file has no weights"):
        cyl.write(tmp_path / "n.gdf")
    # weights 1 are a non-rational patch: written
    for p in cyl.patches:
        p.weights = np.ones((2, 3))
    cyl.write(tmp_path / "one.gdf")
    with pytest.raises(ValueError, match="weights must be a positive finite number"):
        cyl.patches[0].weights = [[1.0, 0.0, 1.0]] * 2
    cyl.patches[0].weights = np.ones((3, 2))
    with pytest.raises(ValueError, match=r"weights need the shape \(M_v, M_u\)"):
        cyl.geometry()


def _plate(z=-0.8, a=1.5):
    """A non-rational bilinear plate quadrant 0 <= x, y <= a at depth z, normal up."""
    t = (0.0, 0.0, 1.0, 1.0)
    return (2, 2, t, t, np.array([[(0, 0, z), (a, 0, z)], [(0, a, z), (a, a, z)]]))


def _disc(r0, r1):
    """The ring r1 <= r <= r0 of the quarter disc in z = 0, rational."""
    tu, tv = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    c = np.array([[(r * a, r * b, 0.0) for a, b in QUARTER_CIRCLE] for r in (r1, r0)])  # normal up
    return (3, 2, tu, tv, c, np.array([QUARTER_WEIGHTS] * 2))


def test_records_of_a_rational_mesh():
    # a body on elements 2..3 of a mesh whose first element is another rational patch (other weights
    # and counts before them): the same bits as the mesh of those two patches alone
    w = np.sqrt(np.array([0.5, 1.5]) * G)
    t = (0.0, 0.0, 1.0, 1.0)
    c = np.array([[(3, 0, -1), (4, 0, -1)], [(3, 1, -1), (4, 1, -1)]])
    other = (2, 2, t, t, c, np.array([[1.0, 3.0], [2.0, 1.0]]))
    rows = (4, 2, 3, 3, 3, 4)
    whole = spline([other, *nurbs_cylinder_patches(1.0, 0.5)], rows, symmetry="xz_and_yz")
    a = solve({"mesh": (whole, (2, 3))}, w, [0.0], options={"higher_order": True})
    b = solve({"mesh": spline(nurbs_cylinder_patches(1.0, 0.5), rows, symmetry="xz_and_yz")}, w, [0.0],
              options={"higher_order": True})  # fmt: skip
    for q in QS:
        np.testing.assert_array_equal(view(a, q), view(b, q), err_msg=q)


def test_rational_and_other_patches_of_meshes_of_their_own():
    # the rational cylinder with a non-rational plate (dipoles) and a rational interior surface of
    # two patches as meshes of their own: the same bits as the elements of one mesh (weights 1 for
    # the plate)
    w = np.sqrt(np.array([0.8, 2.5, 3.5]) * G)
    rows = (4, 2, 3, 3, 3, 4)
    cyl, plate, disc = nurbs_cylinder_patches(1.0, 0.5), [_plate()], [_disc(1.0, 0.5), _disc(0.5, 0.0)]
    ho = {"higher_order": True}
    sym = {"symmetry": "xz_and_yz"}
    a = solve({"mesh": spline(cyl, rows, **sym), "dipoles": [{"mesh": spline(plate, rows, **sym)}],
               "interior_surface": {"method": "mesh", "mesh": spline(disc, rows, **sym)}}, w, [0.0],
              options=ho)  # fmt: skip
    whole = spline([*cyl, *plate, *disc], rows, symmetry="xz_and_yz")
    b = solve({"mesh": (whole, (1, 2)), "dipoles": [{"mesh": (whole, (3, 3))}],
               "interior_surface": {"method": "mesh", "mesh": (whole, (4, 5))}}, w, [0.0], options=ho)  # fmt: skip
    for q in QS:
        np.testing.assert_array_equal(view(a, q), view(b, q), err_msg=q)
