"""References between bodies by name: the couplings of a body (its forces against another body's
motions). They follow a rename, block the removal of what they name, and keep their meaning when
the bodies are reordered (with the tanks' field points, which belong to their tank)."""

import numpy as np
import pytest

import hydrobem as hb

from problems import two_tank_vessels, view

K_OWN = np.diag([1e3, 2e3, 3e3, 4e3, 5e3, 6e3])
K_AB = np.arange(36.0).reshape(6, 6) * 10.0
D_AB = np.arange(36.0).reshape(6, 6)[::-1] + np.eye(6)


def two_vessels(**outputs):
    """Two barges with a tank each: a coupled to b, field points in both tanks and outside."""
    a = {"additional_stiffness": K_OWN, "stiffness_origin": [0.1, 0.0, 0.0],
         "couplings": [{"body": "b", "stiffness": K_AB, "damping": D_AB}]}  # fmt: skip
    b = {"stiffness_origin": [-0.2, 0.3, 0.0], "rotation_center": [0.1, 0.0, -0.1]}
    return two_tank_vessels(a, b, **outputs)


def _blocks(m):
    """The stiffness and damping of the motion equation by body name, 6 x 6 blocks."""
    _, c, d, _ = m._doc.motion_equation()
    names = m.bodies.names()
    k = {n: i for i, n in enumerate(names)}
    return lambda a, i, j: np.asarray(a)[6 * k[i]:6 * k[i] + 6, 6 * k[j]:6 * k[j] + 6], c, d


def test_a_coupling_is_the_block_of_the_named_body_in_any_order():
    m = two_vessels()
    block, c, d = _blocks(m)
    before = {(i, j): (block(c, i, j).copy(), block(d, i, j).copy()) for i in "ab" for j in "ab"}
    assert np.abs(before["b", "a"][0]).max() == 0.0 and np.abs(before["a", "b"][0]).max() > 0.0
    m.bodies.move("b", 0)
    assert m.bodies.names() == ["b", "a"]
    block, c, d = _blocks(m)
    for (i, j), (cs, ds) in before.items():
        np.testing.assert_array_equal(block(c, i, j), cs)
        np.testing.assert_array_equal(block(d, i, j), ds)


def test_reordered_bodies_give_the_same_results():
    # the order of the bodies changes the order of the float64 sums (round-off near 1e-15); the
    # results are stored in float32, whose rounding then differs by an ulp or two of the largest value
    # (measured 1.6e-7 of it): the bound is 4 float32 ulps of the largest value
    eps = 4 * np.finfo(np.float32).eps
    m = two_vessels()
    r0 = m.solve()
    m.bodies.move("b", 0)
    r1 = m.solve()
    for name in "ab":
        x0, x1 = r0.bodies[name], r1.bodies[name]
        for q in ("added_mass", "radiation_damping", "excitation_diffraction", "motion"):
            a0, a1 = view(x0, q), view(x1, q)
            np.testing.assert_allclose(a1, a0, rtol=0, atol=eps * np.abs(a0).max())
    # the points of a tank move with its body: the same pressures at the same points, by the body's
    # name; the sea's point stays
    k0 = [r0.coordinate(b, j)[1] for b in "ab" for j in range(1, 7)]
    k1 = [r1.coordinate(b, j)[1] for b in "ab" for j in range(1, 7)]
    for name in "ab":
        t0, t1 = r0.bodies[name].tanks["t"], r1.bodies[name].tanks["t"]
        np.testing.assert_array_equal(t1.points, t0.points)
        np.testing.assert_allclose(t1.pressure_radiation[..., k1], t0.pressure_radiation[..., k0], rtol=0,
                                   atol=eps * np.abs(t0.pressure_radiation).max())  # fmt: skip
    p0, p1 = r0.field_points.pressure_radiation, r1.field_points.pressure_radiation
    np.testing.assert_array_equal(r1.field_points.points, r0.field_points.points)
    np.testing.assert_allclose(p1[..., k1], p0[..., k0], rtol=0, atol=eps * np.abs(p0).max())


def test_renames_follow_and_references_keep_what_they_name(tmp_path):
    m = two_vessels()
    m.bodies.rename("b", "barge")
    assert m.bodies["a"].couplings[0].body == "barge"
    # a model file keeps the names
    m.save(tmp_path / "two.hbm")
    n = hb.Model.load(tmp_path / "two.hbm")
    assert n.bodies["a"].couplings[0].body == "barge"
    # removing what is named is refused
    with pytest.raises(ValueError, match="bodies/barge: is used by bodies/a/couplings/1"):
        del m.bodies["barge"]
    del m.bodies["a"].couplings[0]
    del m.bodies["barge"]
    assert m.bodies.names() == ["a"]


def test_bodies_added_again_couple_in_any_order():
    # a coupling names a body later in the list; clear() is refused while a body names a mesh
    m = two_vessels()
    a, b = (dict(m.bodies[n].items()) for n in "ab")
    tank = dict(m.bodies["b"].tanks[0].items())
    with pytest.raises(ValueError, match="refers to meshes/"):
        m.meshes.clear()
    m.bodies.clear()
    nb = m.bodies.add("b", **b)
    nb.tanks.add("t", **tank)
    m.bodies.add("a", **a)
    nb.couplings.add(body="a", stiffness=K_AB)  # a body later in the list
    assert m.bodies.names() == ["b", "a"] and m.bodies["b"].couplings[0].body == "a"
    m.check()
