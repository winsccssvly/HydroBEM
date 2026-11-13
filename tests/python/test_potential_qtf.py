"""The force of the second-order potential (theory §12.4-§12.15, test-plan B).

Small problems, each solved once where it can be: the symmetries of the coefficients and both
methods in one solve; a floating body held by a stiff spring has the fixed body's coefficients; two
bodies mirror each other's; a body by a wall is the open sea with its mirror image; the auto radius
and size; the rules of the inputs; the panels of an FDF file; the semi-analytic solution of Kim &
Yue (1990); generalized modes (theory §12.15): the rigid modes given as generalized modes (fixed,
floating with the translations, by a wall; with the rotations only the second-order rotation
apart), and the two methods for the generalized forces of flexible modes; the outputs of the model
(body points, body and field outputs) leave the coefficients as they are, and so do the thread
counts. The convergence in R, the discretizations against each other and the zones beyond R are
validation, run when the method changes, not tests. The refusals of the free surface are in
test_second_order_refusals.py. Meshes are procedural (tests/python/meshes.py)."""

from pathlib import Path

import numpy as np
import pytest

import hydrobem as hb

from bodies import lo_cosine_cylinder, lo_flexible_cylinder, surface
from meshes import cylinder_panels, disc_panels
from problems import flat_surface, model, rigid_modes_as_generalized
from regression_problems import outputs, same_outputs

G = 9.80665
W = np.array([1.2, 1.6])
KIM_YUE = Path(__file__).resolve().parents[1] / "data" / "kim_yue1990.txt"
ENV = {"density": 1000.0, "gravity": G, "depth": 3.0}
# the items of the potential QTF's free surface, as the dicts of the tests name them (problems.flat_surface)
FS = {"method", "panel_size", "inner_radius", "mesh", "ranges", "quadrature_points", "quadrature_annuli",
      "quadrature_radius_step", "quadrature_radial_nodes", "quadrature_azimuthal_nodes", "outer_circle_segments",
      "expansion_order"}


def potential(body=None, methods=("indirect",), **out):
    """The potential QTFs of the test cylinder (fixed by default) to R = 3 with panels of 0.3."""
    body = body or lo_cosine_cylinder(fixed=True)
    moving = any(bool(b.get("modes", True)) for b in ([body] if isinstance(body, dict) else body))
    fs = {"inner_radius": 3.0, "panel_size": 0.3, **{k: out.pop(k) for k in list(out) if k in FS}}
    o = {"diffraction": True, "radiation": moving, "motion": moving, **{f"qtf_potential_{x}": True for x in methods},
         **out}
    m = model(body, W, [0.0], environment=ENV, outputs=o)
    m.second_order.update(**flat_surface("free_surface", {**fs}))
    return m.solve()


def test_the_symmetries_and_both_methods_in_one_solve():
    # both methods in one solve: the difference coefficients zero at equal frequencies and Hermitian,
    # the sums symmetric (exact identities of the definitions, theory §12.2); the direct and the
    # indirect method solve one continuous problem (theory §12.6) and differ by the
    # discretization only, a wrong term in either by O(1) (here 4.6e-4 differences, 2.1e-2 sums)
    r = potential(methods=("direct", "indirect"))
    assert r.qtf.heading_pairs.tolist() == [[0.0, 0.0]]
    for meth in ("direct", "indirect"):
        d, s = getattr(r.qtf.potential, meth).difference, getattr(r.qtf.potential, meth).sum
        assert np.all(np.isfinite(d)) and np.all(np.isfinite(s))
        scale = np.abs(s).max()
        for f in range(len(W)):
            assert np.abs(d[f, f]).max() == 0.0
        np.testing.assert_allclose(d, np.conj(np.swapaxes(d, 0, 1)), rtol=0, atol=1e-12 * scale)
        np.testing.assert_allclose(s, np.swapaxes(s, 0, 1), rtol=0, atol=1e-12 * scale)
    for kind, tol in (("difference", 2e-2), ("sum", 5e-2)):
        x = getattr(r.qtf.potential.indirect, kind)[..., [0, 2, 4]]
        y = getattr(r.qtf.potential.direct, kind)[..., [0, 2, 4]]
        assert np.abs(x - y).max() <= tol * np.abs(x).max(), (kind, np.abs(x - y).max() / np.abs(x).max())
    assert r.quantity("qtf_quadratic_pressure_integration_sum") is None


def test_a_body_held_by_a_stiff_spring_has_the_fixed_body_potential_qtf():
    # the motions about 1e-9 of those of the free body: the radiation in the disturbance and the
    # motion terms of the body forcing vanish with them
    held = potential(lo_cosine_cylinder(additional_stiffness=(1e12 * np.eye(6)).tolist()))
    alone = potential()
    assert np.abs(held.motion).max() < 1e-6
    for q in ("qtf_potential_indirect_sum", "qtf_potential_indirect_difference"):
        x, y = held.quantity(q), alone.quantity(q)
        assert np.abs(x - y).max() <= 1e-5 * np.abs(y).max(), (q, np.abs(x - y).max() / np.abs(y).max())


def test_two_fixed_bodies_mirror_each_other():
    # two fixed cylinders at y = ±2, the wave along x: body b's coefficients are body a's mirrored
    # about y = 0 (sway, roll and yaw change sign), an exact identity of the problem; a bookkeeping
    # error (the bodies swapped, a sign) gives O(1), the free surface's triangles next to the
    # waterlines (not mirrored) 2e-3
    bodies = [{**lo_cosine_cylinder(fixed=True), "name": "a", "position": [0.0, 2.0, 0.0]},
              {**lo_cosine_cylinder(fixed=True), "name": "b", "position": [0.0, -2.0, 0.0]}]  # fmt: skip
    r = potential(bodies, inner_radius=6.5)
    sign = np.array([1, -1, 1, -1, 1, -1])
    for q in ("qtf_potential_indirect_sum", "qtf_potential_indirect_difference"):
        x = r.quantity(q)
        a, b = x[..., :6], x[..., 6:]
        scale = np.abs(a).max()
        assert scale > 0 and np.all(np.isfinite(x))
        assert np.abs(b - sign * a).max() <= 1e-2 * scale, (q, np.abs(b - sign * a).max() / scale)


def test_two_floating_bodies_mirror_each_other():
    # two floating cylinders at y = ±2, the wave along x: as for the fixed ones, an exact identity of
    # the problem, now with each body's motions in the other's free-surface forcing (the radiation
    # of the motions of both) and in its own body condition; the motions mirror too
    bodies = [{**lo_cosine_cylinder(), "name": "a", "position": [0.0, 2.0, 0.0]},
              {**lo_cosine_cylinder(), "name": "b", "position": [0.0, -2.0, 0.0]}]  # fmt: skip
    r = potential(bodies, inner_radius=6.5)
    sign = np.array([1, -1, 1, -1, 1, -1])
    mot = r.motion
    assert np.abs(mot[..., 6:] - sign * mot[..., :6]).max() <= 1e-5 * np.abs(mot).max()
    for q in ("qtf_potential_indirect_sum", "qtf_potential_indirect_difference"):
        x = r.quantity(q)
        a, b = x[..., :6], x[..., 6:]
        scale = np.abs(a).max()
        assert scale > 0 and np.all(np.isfinite(x))
        assert np.abs(b - sign * a).max() <= 1e-2 * scale, (q, np.abs(b - sign * a).max() / scale)


def pair_qtf(r, quantity, p, q):
    """[F, F, 6 B] coefficients of the heading pair (p, q) (deg) of a bidirectional result: the
    stored pair, or the other order from [b, a] of (q, p) (the conjugate at the difference
    frequency, the same at the sum; reference/results)."""
    x, hp = r.quantity(quantity), r.qtf.heading_pairs.tolist()
    if [p, q] in hp:
        return x[:, :, hp.index([p, q])]
    y = np.swapaxes(x[:, :, hp.index([q, p])], 0, 1)
    return np.conj(y) if quantity.endswith("difference") else y


@pytest.mark.parametrize(
    "domain, heading, place, free",
    [("x_wall", 30.0, [2.0, 0.0, 0.0], False), ("y_wall", 60.0, [0.0, 2.0, 0.0], True)],
)
def test_a_wall_is_the_open_sea_with_the_mirror_images(domain, heading, place, free):
    # a body 2 m from a wall (1 m of water between them) is the open-sea problem of the body and its
    # mirror image (a second body) under the wave and its mirror image, beta' = 180 - beta (x = 0)
    # or -beta (y = 0), every field even about the wall (theory §12.14). The QTF is bilinear in
    # the two first-order waves: the open-sea coefficients of the four bidirectional pairs
    # (beta or beta') x (beta or beta') summed are those of that problem, A for the body, B for its
    # image. Both solves have the same panels (the image panels of the wall's first-order solve are
    # the second body's mirrored; the free surface is generated for both waterlines about the same
    # centre), so the wall's coefficients, the even part of that problem, are exactly (A + S B) / 2,
    # S the mirror image of a force and a moment: 1.3e-6 observed (the float32 first-order fields;
    # a missing halving or image wave gives O(1), the direct method's right-hand side not doubled
    # for the wall's even projection 37 %). Against A alone they differ by half of B's departure
    # from S A, the generated triangles next to the waterlines not being mirror images (as for two
    # bodies mirroring each other): 2e-3 observed. Fixed and floating (the motions of the wall the
    # sum of the open sea's of both headings), both methods; coarse panels (the identity does not
    # depend on them).
    image = [-place[0], place[1], 0.0] if domain == "x_wall" else [place[0], -place[1], 0.0]
    mirror = 180.0 - heading if domain == "x_wall" else -heading
    sign = np.array([-1, 1, 1, 1, -1, -1] if domain == "x_wall" else [1, -1, 1, -1, 1, -1])

    def make(**kw):
        return lo_cosine_cylinder(fixed=not free, **kw)

    def solved(bodies, domain, headings, **out):
        o = {"diffraction": True, "radiation": free, "motion": free, "qtf_potential_indirect": True,
             "qtf_potential_direct": True,
             **out}  # fmt: skip
        m = model(bodies, W, headings, environment={**ENV, "domain": domain}, outputs=o)
        m.second_order.update(**flat_surface("free_surface", {"inner_radius": 6.5, "panel_size": 0.5}))
        return m.solve()

    wall = solved({**make(), "position": place}, domain, [heading])
    sea = solved([{**make(), "name": "a", "position": place}, {**make(), "name": "b", "position": image}],
                 "open_sea", [heading, mirror], qtf_bidirectional=True)  # fmt: skip
    if free:
        mot = wall.motion[:, 0]
        assert np.abs(mot - sea.motion[:, 0, :6] - sea.motion[:, 1, :6]).max() <= 1e-5 * np.abs(mot).max()
    for meth in ("indirect", "direct"):
        for kind in ("difference", "sum"):
            q = f"qtf_potential_{meth}_{kind}"
            x = wall.quantity(q)[:, :, 0]
            y = sum(pair_qtf(sea, q, p, s) for p in (heading, mirror) for s in (heading, mirror))
            a, b = y[..., :6], y[..., 6:]
            scale = np.abs(a).max()
            assert scale > 0 and np.all(np.isfinite(x))
            even = np.abs(x - 0.5 * (a + sign * b)).max() / scale
            assert even <= 1e-5 and np.abs(x - a).max() <= 1e-2 * scale, (q, even, np.abs(x - a).max() / scale)


def test_the_rules_of_the_walls():
    # one wall: the generated panels only (the cells are built about each body, not its image); two
    # walls and a channel refused (their images are not one mirror image)
    base = {"diffraction": True, "qtf_potential_indirect": True}
    away = {**lo_cosine_cylinder(fixed=True), "position": [2.0, 2.0, 0.0]}
    for domain in ("orthogonal_walls", "channel"):
        env = {**ENV, "domain": domain, **({"channel_width": 10.0} if domain == "channel" else {})}
        m = model({**away, "position": [2.0, 2.0 if domain != "channel" else 0.0, 0.0]}, W[:1], [0.0],
                  environment=env, outputs=base)  # fmt: skip
        with pytest.raises(ValueError, match="not with orthogonal_walls or a channel"):
            m.route()
    m = model(away, W[:1], [0.0], environment={**ENV, "domain": "x_wall"}, outputs=base)
    assert m.route() == "low_order_finite_x_wall"
    m.second_order.free_surface_method = "cells"
    with pytest.raises(ValueError, match="with a wall the free surface is the generated panels"):
        m.route()


def test_the_outputs_of_the_model_leave_the_coefficients_as_they_are():
    # body points with the body outputs and field points with the field outputs of a floating
    # body: the sub-problems take the body panels and the free-surface points of their own, so both
    # methods give the same numbers as without them (the body points once replaced the panels in
    # the sub-problems: no area for the body integrals, an index beyond the direct method's
    # columns); the model's own outputs are there
    def solved(**kw):
        out = {"diffraction": True, "radiation": True, "motion": True, "qtf_potential_direct": True,
               "qtf_potential_indirect": True}
        body, pts = lo_cosine_cylinder(**kw.pop("body", {})), kw.pop("field_points", None)
        m = model(body, W, [0.0], environment=ENV, outputs={**out, **kw}, field_points=pts)
        m.second_order.update(**flat_surface("free_surface", {"inner_radius": 3.0, "panel_size": 0.3}))
        return m.solve()

    plain = solved()
    pts = [[1.0, 0.0, -0.25], [0.6, 0.6, -0.4]]
    r = solved(body={"body_points": pts}, body_pressure=True, body_velocity=True, field_pressure=True,
               field_velocity=True, field_points=[[3.0, 0.5, -0.5], [-2.0, 2.5, 0.0]])  # fmt: skip
    names = [q for q in hb.quantities() if q.startswith("qtf_potential")]
    assert sum(plain.quantity(q) is not None for q in names) == 4
    for q in names:
        x, y = r.quantity(q), plain.quantity(q)
        assert (x is None) == (y is None) and (x is None or np.array_equal(x, y, equal_nan=True)), q
    np.testing.assert_allclose(r.body_points.points, pts, rtol=0, atol=1e-12)
    assert r.body_points.pressure_diffraction.shape == (len(W), 2, 1)
    assert r.field_points.velocity_diffraction.shape == (len(W), 2, 1, 3)


@pytest.mark.slow  # seconds: deselected by scripts/test.* fast
def test_the_second_order_loads_do_not_depend_on_the_thread_counts():
    # the potential QTF (both methods, a quadrature zone), the quadratic QTF, the drift and Newman's
    # approximation of a floating body, two headings and their pairs: every sum of the solve runs
    # in a fixed order (the outer integral over the radii, the parallel pairs collected in order),
    # so one thread and several give the same bytes (theory §13.3)
    def solved(f, i):
        out = {"diffraction": True, "radiation": True, "motion": True, "qtf_potential_direct": True,
               "qtf_potential_indirect": True,
               "drift_pressure_integration": True, "qtf_quadratic_pressure_integration": True,
               "qtf_newman": True, "qtf_bidirectional": True,
               "drift_bidirectional": True}  # fmt: skip
        m = model(lo_cosine_cylinder(), W, [0.0, 30.0], environment=ENV, outputs=out,
                  options={"frequency_threads": f, "inner_threads": i})  # fmt: skip
        m.second_order.update(free_surface_inner_radius=3.0, free_surface_panel_size=0.4, quadrature_annuli=1)
        return outputs(m.solve())

    a, b = solved(1, 1), solved(2, 3)
    assert "qtf_potential_direct_sum" in a and "qtf_quadratic_pressure_integration_sum" in a
    same_outputs(a, b)


def test_the_auto_radius_and_size():
    # auto: R = 4 a and a size of a third of the shortest combined wavelength, at most a/4 (a, the
    # waterline radius 1: R = 4, size 0.25): the coefficients of those given explicitly; the
    # quadratic part beside it
    out = {"diffraction": True, "qtf_quadratic_pressure_integration": True,
           "qtf_potential_indirect": True}  # fmt: skip
    auto = model(lo_cosine_cylinder(fixed=True), W, [0.0], environment=ENV, outputs=out).solve()
    given = potential(inner_radius=4.0, panel_size=0.25)
    for kind in ("difference", "sum"):
        x, y = auto.quantity("qtf_potential_indirect_" + kind), given.quantity("qtf_potential_indirect_" + kind)
        assert np.abs(x - y).max() <= 1e-9 * np.abs(y).max()
        assert np.all(np.isfinite(auto.quantity("qtf_quadratic_pressure_integration_" + kind)))
        assert auto.quantity("qtf_potential_direct_" + kind) is None
    assert auto.quantity("drift_pressure_integration") is None  # computed for the QTF, not asked for


def test_the_rules_of_the_inputs():
    base = {"diffraction": True, "qtf_potential_indirect": True}
    with pytest.raises(ValueError, match="needs first_order.motion"):
        model(lo_cosine_cylinder(), W[:1], [0.0], environment=ENV, outputs=base).route()
    sized = {"inner_radius": 3.0, "panel_size": 0.3}

    def built(bodies, **fs):
        m = model(bodies, W[:1], [0.0], environment=ENV, outputs=base)
        m.second_order.update(**flat_surface("free_surface", {**{**sized, **fs}}))
        return m

    m = built(lo_cosine_cylinder(fixed=True))
    m.options.higher_order = True
    with pytest.raises(ValueError, match="not yet in the higher order"):
        m.route()
    # the generated panels need R beyond twice the waterlines' distance from the centre; the cells,
    # a circle of R holding each body's box
    with pytest.raises(ValueError, match="must exceed twice"):
        built(lo_cosine_cylinder(fixed=True), inner_radius=1.5).solve()
    two = [{**lo_cosine_cylinder(fixed=True), "name": "a"},
           {**lo_cosine_cylinder(fixed=True), "name": "b", "position": [5.0, 0.0, 0.0]}]  # fmt: skip
    with pytest.raises(ValueError, match="does not contain"):
        built(two, method="cells").solve()
    # the free surface given as a mesh: the potential QTF asked for, z = 0; the items of
    # the generated zone kept unused
    m = model(lo_cosine_cylinder(fixed=True), W[:1], [0.0], environment=ENV, outputs={"diffraction": True})
    m.meshes.add("fs", kind="panels", vertices=[[[1.0, 0, 0], [2.0, 0, 0], [2.0, 1.0, 0], [1.0, 1.0, 0]]])
    m.second_order.free_surface_method = "mesh"
    with pytest.raises(ValueError, match="second_order: free_surface_method = 'mesh' needs free_surface_mesh"):
        m.route()
    m.second_order.free_surface_mesh = "fs"
    with pytest.raises(ValueError, match="free_surface_method = 'mesh' needs qtf_potential_direct or "
                                         "qtf_potential_indirect"):  # fmt: skip
        m.route()
    m.second_order.update(qtf_potential_indirect=True)
    m.second_order.free_surface_inner_radius = 4.0  # kept, unused
    m.route()
    m.meshes.add("bad", kind="panels", vertices=[[[1.0, 0, 0], [2.0, 0, -0.1], [2.0, 1.0, 0], [1.0, 1.0, 0]]])
    m.second_order.free_surface_mesh = "bad"
    with pytest.raises(ValueError, match="every vertex must lie in z = 0"):
        m.route()


def test_the_zones_of_the_free_surface():
    # the zones as the solve integrates them (no solve): exact identities of the quadratures — the
    # generated panels tile the disk of the inner radius less the waterline polygon, the quadrature
    # zone's weights sum to the area of its annulus (Gauss-Legendre exact for r dr, equal nodes in
    # theta), the outer circle's to its circumference; the panels' centroids in the panelled points
    out = {"qtf_potential_indirect": True}
    m = model(lo_cosine_cylinder(fixed=True), W, [0.0], environment=ENV, outputs=out)
    m.second_order.update(free_surface_inner_radius=3.0, free_surface_panel_size=0.3, quadrature_annuli=2,
                          quadrature_radius_step=1.0, outer_circle_segments=16)  # fmt: skip
    p = m.panels(role="free_surface")
    assert set(p.role) == {"free_surface"} and np.all(p.body == -1) and np.all(p.areas > 0)
    fs = m.free_surface_points("panelled")
    x, w = fs.points, fs.weights
    np.testing.assert_allclose(w.sum(), p.areas.sum(), rtol=1e-12)
    np.testing.assert_allclose(np.sort(x[:, 0]), np.sort(p.centroids[:, 0]), atol=1e-12)
    rim = np.hypot(p.vertices[..., 0], p.vertices[..., 1]).max()
    assert abs(rim - 3.0) <= 1e-12
    fs = m.free_surface_points("quadrature")
    x, w = fs.points, fs.weights
    np.testing.assert_allclose(w.sum(), np.pi * (5.0**2 - 3.0**2), rtol=1e-12)
    r = np.hypot(x[:, 0], x[:, 1])
    assert r.min() > 3.0 and r.max() < 5.0
    fs = m.free_surface_points("outer_circle")
    x, w = fs.points, fs.weights
    assert len(x) == 4 * 16  # 16 segments a quadrant (the cylinder's two symmetry planes)
    np.testing.assert_allclose(np.hypot(x[:, 0], x[:, 1]), 5.0, rtol=1e-12)
    np.testing.assert_allclose(w.sum(), 2 * np.pi * 5.0, rtol=1e-12)
    m.second_order.free_surface_method = "cells"
    with pytest.raises(ValueError, match="cells have no panels"):
        m.panels(role="free_surface")
    with pytest.raises(ValueError, match="zone must be"):
        m.free_surface_points("asymptotic")


def test_the_keys_of_a_given_free_surface_mesh():
    # the rows of a given free-surface mesh are keyed as every input mesh (reference/queries
    # Table 4): its stored elements, the copy about its symmetry planes (only (1, 1) without
    # include_reflections), the reflected copies' quads keeping the orientation
    out = {"qtf_potential_indirect": True}
    m = model(lo_cosine_cylinder(fixed=True), W, [0.0], environment=ENV, outputs=out)
    ring = disc_panels(3.0, n_r=2, n_phi=4, inner=1.0)
    m.meshes.add("fs", kind="panels", vertices=ring, symmetry="xz_and_yz")
    m.second_order.update(free_surface_method="mesh", free_surface_mesh="fs")
    one = m.panels(role="free_surface", include_reflections=False)
    assert np.asarray(one.element).tolist() == list(range(1, len(ring) + 1))
    assert np.all(np.asarray(one.reflection) == 1) and np.all(np.asarray(one.body) == -1)
    np.testing.assert_array_equal(one.vertices, ring)
    p = m.panels(role="free_surface")
    assert len(p.element) == 4 * len(ring)
    assert {tuple(r) for r in np.asarray(p.reflection)} == {(1, 1), (-1, 1), (1, -1), (-1, -1)}
    assert len(set(np.sign(np.asarray(p.normals)[:, 2]))) == 1
    t = m.tessellation(role="free_surface")
    keys = {(e, *r) for e, r in zip(np.asarray(p.element), np.asarray(p.reflection), strict=True)}
    assert {(e, *r) for e, r in zip(np.asarray(t.element), np.asarray(t.reflection), strict=True)} == keys
    assert len(set(np.sign(np.asarray(t.normals)[..., 2]).ravel())) == 1
    # no frequency between 0 and inf, the panels and the quadrature zone given but its auto step: the
    # panels need neither
    m.environment.omegas = [0.0]
    m.second_order.update(quadrature_annuli=2)
    assert np.asarray(m.panels(role="free_surface").vertices).tobytes() == np.asarray(p.vertices).tobytes()


def test_the_free_surface_queries_check_the_free_surface_alone():
    # the free-surface queries apply the rules of the free surface (its items, the geometry and
    # waterlines of the bodies, the environment and its domain), not those of the frequencies, the
    # orders' switches or the other options; a zone whose auto values the frequencies decide needs
    # them. The answers are those of the model's queries (the free-surface rows of m.panels with
    # another role, which checks every rule)
    out = {"qtf_potential_indirect": True}
    m = model(lo_cosine_cylinder(fixed=True), W, [0.0], environment=ENV, outputs=out)
    m.second_order.update(free_surface_inner_radius=3.0, free_surface_panel_size=0.3, quadrature_annuli=2,
                          quadrature_radius_step=1.0, outer_circle_segments=16)  # fmt: skip
    m.check()
    zones = ("panelled", "quadrature", "outer_circle")
    full = m.panels(role=["wetted", "free_surface"])
    rows = np.asarray(full.role) == "free_surface"
    p, t = m.panels(role="free_surface"), m.tessellation(role="free_surface")
    for key in ("vertices", "centroids", "normals", "areas", "body", "tank", "owner", "element", "part", "reflection"):
        assert np.asarray(getattr(p, key)).tobytes() == np.asarray(getattr(full, key))[rows].tobytes(), key
    full_t = m.tessellation(role=["wetted", "free_surface"])
    quads = np.asarray(full_t.role) == "free_surface"
    assert np.asarray(t.vertices).tobytes() == np.asarray(full_t.vertices)[quads].tobytes()
    assert np.asarray(t.element).tolist() == np.asarray(full_t.element)[quads].tolist()
    points = {z: m.free_surface_points(z) for z in zones}

    def same(*of):
        assert np.asarray(m.panels(role="free_surface").vertices).tobytes() == np.asarray(p.vertices).tobytes()
        assert np.asarray(m.tessellation(role="free_surface").vertices).tobytes() == np.asarray(t.vertices).tobytes()
        for z in of:
            got = m.free_surface_points(z)
            assert got.points.tobytes() == points[z].points.tobytes(), z
            assert got.weights.tobytes() == points[z].weights.tobytes(), z

    # the frequencies: none, with the diffraction on; the quadrature zone's auto azimuthal nodes take
    # them, the other zones' items are given
    m.environment.omegas = []
    with pytest.raises(ValueError, match="omegas is empty"):
        m.check()
    same("panelled", "outer_circle")
    with pytest.raises(ValueError, match=r"omegas has no frequency between 0 and inf.*quadrature_azimuthal_nodes"):
        m.free_surface_points("quadrature")
    m.second_order.quadrature_azimuthal_nodes = 8
    m.free_surface_points("quadrature")
    m.second_order.reset("quadrature_azimuthal_nodes")
    # the auto panel size and segments take them (refused, naming the items); invalid frequencies
    # are refused where they are used
    m.second_order.reset("free_surface_panel_size")
    for query in (lambda: m.panels(role="free_surface"), lambda: m.free_surface_points("panelled")):
        with pytest.raises(ValueError, match=r"auto second_order.free_surface_panel_size of the panelled zone"):
            query()
    m.second_order.free_surface_panel_size = 0.3
    m.second_order.reset("outer_circle_segments")
    with pytest.raises(ValueError, match=r"auto second_order.outer_circle_segments of the outer_circle zone"):
        m.free_surface_points("outer_circle")
    m.second_order.outer_circle_segments = 16
    m.environment.omegas = [1.2, 1.2]
    with pytest.raises(ValueError, match="omegas has the duplicate value 1.2"):
        m.free_surface_points("quadrature")
    same("panelled", "outer_circle")
    m.environment.omegas = W
    same(*zones)
    # the switches and options of the orders: the QTF without the diffraction, Newman's approximation
    # without a drift
    m.first_order.diffraction = False
    with pytest.raises(ValueError, match="require the diffraction solution"):
        m.check()
    same(*zones)
    m.first_order.diffraction = True
    m.second_order.qtf_newman = True
    with pytest.raises(ValueError, match="qtf_newman makes Newman's approximation"):
        m.check()
    same(*zones)
    m.second_order.qtf_newman = False
    # the invalid input of the free surface, its bodies and the domain is still refused
    m.second_order.free_surface_method = "mesh"
    for query in (lambda: m.panels(role="free_surface"), lambda: m.free_surface_points("panelled")):
        with pytest.raises(ValueError, match="free_surface_method = 'mesh' needs free_surface_mesh"):
            query()
    m.second_order.free_surface_method = "panels"
    m.environment.domain = "orthogonal_walls"
    with pytest.raises(ValueError, match="available in the open sea and with one wall"):
        m.free_surface_points("outer_circle")
    m.environment.domain = "open_sea"
    m.bodies["body"].ranges = [(1, 10_000)]
    with pytest.raises(ValueError, match="beyond the mesh") as e:
        m.panels(role="free_surface")
    assert e.value.location.body == 0
    m.bodies["body"].reset("ranges")
    same(*zones)


def test_the_queries_of_a_body_and_of_the_free_surface_follow_every_change():
    # the problem of a body's queries and that of the free surface's are kept for the next query of
    # the same scope: a query of the other scope between them, or a change of any input, builds the
    # problem again (the answers of the inputs of now)
    out = {"qtf_potential_indirect": True}
    m = model(lo_cosine_cylinder(fixed=True), W, [0.0], environment=ENV, outputs=out)
    m.second_order.update(free_surface_inner_radius=3.0, free_surface_panel_size=0.3)
    body = m.bodies["body"]
    h, fs = body.hydrostatics(), m.free_surface_points("panelled")
    for _ in range(2):
        assert body.hydrostatics().displaced_mass == h.displaced_mass
        assert m.free_surface_points("panelled").points.tobytes() == fs.points.tobytes()
    m.environment.density = 1025.0
    assert body.hydrostatics().displaced_mass == pytest.approx(1.025 * h.displaced_mass, rel=1e-14)
    assert m.free_surface_points("panelled").points.tobytes() == fs.points.tobytes()
    m.second_order.free_surface_panel_size = 0.2
    assert len(m.free_surface_points("panelled").points) > len(fs.points)
    assert body.hydrostatics().displaced_mass == pytest.approx(1.025 * h.displaced_mass, rel=1e-14)
    m.second_order.free_surface_panel_size = 0.3
    m.environment.density = 1000.0
    assert m.free_surface_points("panelled").points.tobytes() == fs.points.tobytes()
    assert body.hydrostatics().displaced_mass == h.displaced_mass


def test_the_panels_of_a_free_surface_file(tmp_path):
    # load_fdf: the panels of an FDF file (four x, then four y per panel, the numbers over one or
    # two lines with a comment after them), the symmetry given; only the panels are taken
    lines = ["a free surface", "3.0 RINNER", "2 4 NPF, NTCL", "0 0.0 0 0 NAL, DELR, NCIRE, NGSP",
             "1.0 2.0 2.0 1.0", " 0.0 0.0 1.0 1.0 end of panel 1", "2.0 3.0 3.0 2.0 0.0 0.0 1.0 1.0"]  # fmt: skip
    text = "".join(x + "\n" for x in lines)
    (tmp_path / "fs.fdf").write_bytes(text.encode())
    m = hb.Model()
    mesh = m.meshes.load_fdf("fs", tmp_path / "fs.fdf", symmetry="xz_and_yz")
    v = np.asarray(mesh.vertices)
    assert v.shape == (2, 4, 3) and mesh.symmetry == "xz_and_yz"
    np.testing.assert_array_equal(v[1, :, 0], [2.0, 3.0, 3.0, 2.0])
    np.testing.assert_array_equal(v[0, :, 1], [0.0, 0.0, 1.0, 1.0])
    assert np.all(v[..., 2] == 0.0)
    (tmp_path / "short.fdf").write_bytes(text.replace("2 4 NPF", "3 4 NPF").encode())
    with pytest.raises(ValueError, match="the file ends while reading"):
        m.meshes.load_fdf("short", tmp_path / "short.fdf")


@pytest.mark.slow  # a minute: deselected by scripts/test.* fast
def test_a_bottom_mounted_cylinder_against_the_semi_analytic_solution():
    # Kim & Yue (1990), table 3 (tests/data/kim_yue1990.txt from scripts/extract_kim_yue.py): the
    # second-order potential force on a bottom-mounted cylinder, h/a = 1, |f_p| / (rho g a A_j A_l)
    # at nu a = w^2 a / g = (1.0, 2.0) and (1.4, 1.6), difference and sum; 12 panels a quadrant in
    # each direction, R = 4 a (auto); both methods in one solve: within 1.5 % of each value (8.1e-3
    # found, the direct sum at (1.0, 2.0); the four printed digits round 1.1e-3 of the smallest)
    ref = [line.split() for line in KIM_YUE.read_text().splitlines() if line and not line.startswith("#")]
    nus = [1.0, 1.4, 1.6, 2.0]
    pan = cylinder_panels(1.0, 1.0, n_phi=12, n_z=12, n_r=0, quarter=True, cosine=True)
    body = {"mesh": surface(pan, symmetry="xz_and_yz"), "modes": []}
    out = {"diffraction": True, "qtf_potential_indirect": True, "qtf_potential_direct": True}
    r = model(body, np.sqrt(np.array(nus) * G), [0.0], environment={"density": 1000.0, "gravity": G, "depth": 1.0},
              outputs=out).solve()  # fmt: skip
    for method in ("indirect", "direct"):
        for kind, a, b, v in ref:
            i, j = nus.index(float(a)), nus.index(float(b))
            part = getattr(r.qtf.potential, method)
            q = part.difference[j, i] if kind == "difference" else part.sum[i, j]
            x = abs(q[0, 0]) / (1000.0 * G)
            assert abs(x - float(v)) <= 1.5e-2 * float(v), (method, kind, a, b, x, v)


def both_methods(body, env=ENV, heads=(0.0,), moving=True, **fs):
    """The potential QTFs of both methods in one solve, to R = 3 with panels of 0.3 by default."""
    out = {"diffraction": True, "radiation": moving, "motion": moving, "qtf_potential_direct": True,
           "qtf_potential_indirect": True}
    m = model(body, W, list(heads), environment=env, outputs=out)
    m.second_order.update(**flat_surface("free_surface", {**{"inner_radius": 3.0, "panel_size": 0.3, **fs}}))
    return m.solve()


@pytest.mark.parametrize("case", ["fixed", "translations", "wall"])
def test_rigid_modes_as_generalized_modes_have_the_rigid_potential_qtf(case):
    # the generalized coordinates with the rigid body's mass and restoring: the same first-order
    # solution, the same forcing and body condition (the displacement, the turned normal), and the
    # radiation of a generalized mode at the pair frequency is that of its rigid mode; both methods,
    # the six rigid rows and the generalized rows against the rigid rows of their modes (R = O).
    # Fixed: the rigid body without modes against the generalized one held by 1e12 (4e-7 observed,
    # its motions); floating with the translations as generalized modes, in the open sea and by an
    # x wall (2e-15 observed)
    which = range(6) if case == "fixed" else (0, 1, 2)
    env, fs, place = (ENV, {}, [0.0, 0.0, 0.0]) if case != "wall" else (
        {**ENV, "domain": "x_wall"}, {"inner_radius": 7.0, "panel_size": 0.5}, [2.0, 0.0, 0.0])
    heads = (0.0,) if case != "wall" else (30.0,)
    rigid = lo_cosine_cylinder(fixed=case == "fixed", position=place)
    gen = rigid_modes_as_generalized(lo_cosine_cylinder(position=place), which, env, fixed=case == "fixed")
    a = both_methods(rigid, env, heads, moving=case != "fixed", **fs)
    b = both_methods(gen, env, heads, **fs)
    tol = 1e-5 if case == "fixed" else 1e-12
    for meth in ("direct", "indirect"):
        for kind in ("difference", "sum"):
            x = a.quantity(f"qtf_potential_{meth}_{kind}")
            g = b.quantity(f"qtf_potential_{meth}_{kind}")  # the six rigid rows, then the generalized ones
            scale = np.abs(x).max()
            assert g.shape == x.shape[:3] + (6 + len(which),) and np.all(np.isfinite(g))
            assert np.abs(g[..., :6] - x).max() <= tol * scale, (meth, kind, np.abs(g[..., :6] - x).max() / scale)
            for k, c in enumerate(which):
                assert np.abs(g[..., 6 + k] - x[..., c]).max() <= tol * scale


def test_rigid_rotations_as_generalized_modes_differ_by_the_second_order_rotation():
    # every rigid mode a generalized mode, floating, head seas: the linear map of a rotation has no
    # second-order displacement, the rigid rotation H x (theory §12.7), whose velocity on the
    # body condition is the only difference: in head seas it is even in x and drives heave alone
    # (8e-4 of the largest differences, 3 % sums observed); the other rows the same numbers (4e-16
    # observed), the rotations' generalized rows the moments (their radiation at Ω the rigid one)
    a = both_methods(lo_cosine_cylinder())
    b = both_methods(rigid_modes_as_generalized(lo_cosine_cylinder(), range(6), ENV))
    for meth in ("direct", "indirect"):
        for kind in ("difference", "sum"):
            x = a.quantity(f"qtf_potential_{meth}_{kind}")
            g = b.quantity(f"qtf_potential_{meth}_{kind}")[..., 6:]  # the generalized rows
            scale = np.abs(x).max()
            same = [0, 1, 3, 4, 5]
            assert np.abs(b.quantity(f"qtf_potential_{meth}_{kind}")[..., same] - x[..., same]).max() <= 1e-12 * scale
            assert np.abs(g[..., same] - x[..., same]).max() <= 1e-12 * scale
            assert np.abs(g[..., 2] - x[..., 2]).max() > 1e-4 * scale


def test_the_generalized_forces_of_flexible_modes_by_both_methods():
    # a floating cylinder with a bulging bottom and bending sides: the generalized forces of the
    # modes by the two methods, one continuous problem (a wrong term in either gives O(1)): 1.4e-3
    # (differences) and 9e-3 (sums) of the largest observed, halved on 16 panels a quadrant; the
    # identities of the definitions as for the rigid rows; the rigid rows of the flexible body as
    # for a rigid one (3e-3, 1.2e-2)
    r = both_methods(lo_flexible_cylinder(8))
    for kind, tol in (("difference", 5e-3), ("sum", 2e-2)):
        x = getattr(r.qtf.potential.indirect, kind)[..., 6:]  # the generalized rows
        y = getattr(r.qtf.potential.direct, kind)[..., 6:]
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        scale = np.abs(x).max()
        assert np.abs(x - y).max() <= tol * scale, (kind, np.abs(x - y).max() / scale)
        if kind == "difference":
            for f in range(len(W)):
                assert np.abs(x[f, f]).max() == 0.0 and np.abs(y[f, f]).max() == 0.0
            np.testing.assert_allclose(x, np.conj(np.swapaxes(x, 0, 1)), rtol=0, atol=1e-12 * scale)
        else:
            np.testing.assert_allclose(x, np.swapaxes(x, 0, 1), rtol=0, atol=1e-12 * scale)
        x6 = getattr(r.qtf.potential.indirect, kind)[..., [0, 2, 4]]
        y6 = getattr(r.qtf.potential.direct, kind)[..., [0, 2, 4]]
        assert np.abs(x6 - y6).max() <= 2.5 * tol * np.abs(x6).max()
