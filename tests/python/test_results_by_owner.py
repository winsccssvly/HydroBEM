"""The results by owner and by name: the views of a body, its pressure surfaces and tanks, the lids
and the field points are the slices of the global arrays; ``coefficient`` is the indexed element; the
quantities are the results of a solve alone; a results file keeps every quantity."""

import warnings

import numpy as np
import pytest

import hydrobem as hb

from bodies import TANK_POINTS, acv_body, lo_free_cylinder, moonpool_lid_mesh, moonpool_mesh, part
from problems import model, two_tank_vessels, view

W = np.array([0.8, 1.4])


def two_acvs():
    """Two air-cushion bodies with two and one pressure modes, the second free in heave and pitch
    alone, with the body pressure, the motions and the pressure drift (coarse meshes: the views are
    checked, not the values)."""
    weight = {"mass": "displaced", "center_of_gravity": [0.0, 0.0, 0.0], "radii_of_gyration": [2.5, 5.0, 5.0]}
    a = acv_body(modes=("sign_x", "uniform"), quarter=False, n=1, **weight)
    b = {**acv_body(modes=("uniform",), quarter=False, n=1, name="aft", position=[30.0, 0.0, 0.0], **weight),
         "modes": ["heave", "pitch"]}  # fmt: skip
    return model([a, b], W[:1], (0.0, 45.0),
                 options={"body_pressure": True, "motion": True,
                          "drift_pressure_integration": True})  # fmt: skip


@pytest.fixture(scope="module")
def acvs():
    with warnings.catch_warnings():  # the note of the pressure drift over the wetted surface only
        warnings.filterwarnings("ignore", message=".*integrated over the wetted surface only")
        return two_acvs().solve()


def test_body_views_are_the_slices_of_the_global_arrays(acvs):
    r = acvs
    assert r.bodies.names() == ["acv", "aft"] and len(r.bodies) == 2
    assert isinstance(r.bodies[0], hb.BodyResults) and isinstance(r.excitation, hb.ExcitationResults)
    assert isinstance(r.body_points, hb.BodyPointResults) and isinstance(r.bodies[0].body_points, hb.BodyPointResults)
    assert r.bodies[1].name == "aft" and r.bodies[-1].index == 1 and r.bodies["acv"].index == 0
    # a name is in the list (not its view), as in the lists of the model
    assert "aft" in r.bodies and "bow" not in r.bodies and "" not in r.bodies and r.bodies[0] not in r.bodies
    assert [b.name for b in r.bodies] == ["acv", "aft"] and [b.name for b in r.bodies[::-1]] == ["aft", "acv"]
    ids = np.asarray(r.radiation_mode_ids) - 1
    coords = np.asarray(r.quantity("coordinates"))
    for b, bv in enumerate(r.bodies):
        d = np.flatnonzero(coords[:, 0] == b)
        k = np.flatnonzero(np.isin(ids, d))
        assert len(bv.coordinates) == len(d) == 6 + (2 if b == 0 else 1)
        assert [c.body for c in bv.coordinates] == [b] * len(d)
        for name in ("added_mass", "radiation_damping"):
            g = r.quantity(name)
            v = view(bv, name)
            assert np.array_equal(v, g[:, d][:, :, k]) and np.shares_memory(v, g) and not v.flags.writeable
        for name in ("excitation_diffraction", "excitation_froude_krylov", "excitation_scattering", "motion"):
            assert np.array_equal(view(bv, name), r.quantity(name)[:, :, d]), name
        assert np.array_equal(bv.excitation.haskind, r.excitation.haskind[:, :, k])
        # the drift on its coordinates (NaN on the pressure coordinates), its moving moment by columns
        g = r.drift.pressure_integration
        assert np.array_equal(bv.drift.pressure_integration, g[:, :, d], equal_nan=True)
        assert np.all(np.isfinite(g[:, :, d[:6]])) and np.all(np.isnan(g[:, :, d[6:]]))
        g = r.drift.pressure_integration_body_axes
        assert np.array_equal(bv.drift.pressure_integration_body_axes, g[:, :, 3 * b:3 * b + 3])
        assert bv.drift.control_surface is None and bv.added_mass_zero_frequency is None
        for name in ("mode_restoring", "pressure_mode_damping"):
            assert np.array_equal(view(bv, name), r.quantity(name)[d][:, d]), name
        rows = np.flatnonzero(np.asarray(r.body_points.ids)[:, 0] == b)
        assert rows.size > 0
        for name, axis in (("body_points", 0), ("body_points_normals", 0), ("body_points_areas", 0),
                           ("body_points_ids", 0), ("body_points_pressure_radiation", 1),
                           ("body_points_pressure_diffraction", 1)):  # fmt: skip
            assert np.array_equal(view(bv, name), np.take(r.quantity(name), rows, axis=axis)), name
        assert bv.body_points.velocity_radiation is None
    # the second body's columns: heave, pitch and its pressure mode
    assert [c.name for c in r.bodies["aft"].coordinates][6:] == ["p0"]
    assert r.bodies["aft"].added_mass.shape == (1, 7, 3)


def test_the_hydrostatics_view_of_a_body_is_its_query_field_by_field(acvs):
    m = two_acvs()
    for b in range(2):
        h, ref = acvs.bodies[b].hydrostatics, m.bodies[b].hydrostatics()
        for f in ("volumes", "center_of_buoyancy", "waterplane_centroid", "stiffness_rho_g", "stiffness",
                  "metacentric_height", "center_of_gravity"):  # fmt: skip
            a, e = getattr(h, f), getattr(ref, f)
            assert (a is None and e is None) or np.array_equal(a, e), (b, f)
        for f in ("volume", "displaced_mass", "waterplane_area", "has_weight", "mass"):
            assert getattr(h, f) == getattr(ref, f), (b, f)
    assert acvs.bodies[0].hydrostatics.has_weight and acvs.bodies[1].hydrostatics.metacentric_height is not None


def test_pressure_surface_views_partition_the_global_arrays(acvs):
    r = acvs
    parts = [ps for b in r.bodies for ps in b.pressure_surfaces]
    assert len(parts) == 2 and [len(b.pressure_surfaces) for b in r.bodies] == [1, 1]
    order = np.concatenate([np.flatnonzero(np.asarray(r.quantity("pressure_surfaces_ids")) == s + 1) for s in range(2)])
    assert np.array_equal(np.sort(order), np.arange(len(r.quantity("pressure_surfaces_ids"))))
    for name, axis in (("points", 0), ("elevation_radiation", 1), ("elevation_diffraction", 1)):
        joined = np.concatenate([getattr(ps, name) for ps in parts], axis=axis)
        g = r.quantity("pressure_surfaces" if name == "points" else f"pressure_surfaces_{name}")
        assert np.array_equal(joined, np.take(g, order, axis=axis)), name
    # the surfaces of the second body lie around its position
    assert np.abs(r.bodies["aft"].pressure_surfaces[0].points[:, 0].mean() - 30.0) < 1.0


def test_coefficient_and_coordinate_by_name(acvs):
    r = acvs
    D = len(r.coordinates)
    assert r.coordinate("acv", "heave") == (2, 2)
    assert r.coordinate(0, 3) == (2, 2)  # the one-based position within the body
    d_p1, k_p1 = r.coordinate("acv", "p1")
    assert d_p1 == 7 and r.coordinates[d_p1].name == "p1" and r.radiation_mode_ids[k_p1] == d_p1 + 1
    d_pitch, k_pitch = r.coordinate("aft", "pitch")
    assert r.coordinates[d_pitch].body == 1 and r.coordinates[d_pitch].name == "pitch"
    assert r.coordinate(1, "surge") == (8, None)  # not a radiation column of the second body
    for name in ("added_mass", "radiation_damping"):
        for row in (("acv", "heave"), ("aft", "p0"), ("aft", "surge")):
            for col in (("acv", "p1"), ("aft", "pitch"), ("acv", "roll")):
                d, (_, k) = r.coordinate(*row)[0], r.coordinate(*col)
                c = r.coefficient(name, row, col)
                assert c.shape == (1,) and np.array_equal(c, r.quantity(name)[:, d, k])
    for name in ("excitation_diffraction", "motion", "excitation_froude_krylov"):
        assert np.array_equal(r.coefficient(name, ("aft", "p0")), r.quantity(name)[:, :, D - 1])
    assert np.array_equal(r.coefficient("excitation_haskind", ("aft", "pitch")), r.excitation.haskind[:, :, k_pitch])
    c = r.coefficient("mode_restoring", ("acv", "p0"), ("acv", "p0"))  # a 0-d array, read-only
    assert isinstance(c, np.ndarray) and c.shape == () and not c.flags.writeable and c == r.mode_restoring[6, 6]
    with pytest.raises(KeyError, match="no body 'bow'"):
        r.coefficient("added_mass", ("bow", "heave"), ("acv", "heave"))
    with pytest.raises(KeyError, match="has no mode 'p7'"):
        r.coordinate("acv", "p7")
    with pytest.raises(IndexError, match="modes 1..7, not 10"):
        r.coordinate("aft", 10)
    with pytest.raises(IndexError, match="body 2 does not exist"):
        r.coordinate(2, "heave")
    with pytest.raises(ValueError, match="no result quantity 'added_masses'"):
        r.coefficient("added_masses", ("acv", "heave"), ("acv", "heave"))
    with pytest.raises(ValueError, match="not a radiation column"):
        r.coefficient("added_mass", ("acv", "heave"), ("aft", "surge"))
    with pytest.raises(ValueError, match="not a radiation column"):
        r.coefficient("excitation_haskind", ("aft", "sway"))
    # the second-order loads on the coordinates: a row by its coordinate
    d_heave = r.coordinate("aft", "heave")[0]
    assert np.array_equal(r.coefficient("drift_pressure_integration", ("aft", "heave")),
                          r.drift.pressure_integration[:, :, d_heave])  # fmt: skip
    with pytest.raises(ValueError, match="not indexed by coordinates"):
        r.coefficient("body_points_pressure_diffraction", ("acv", "heave"))
    with pytest.raises(ValueError, match="not available"):
        r.coefficient("added_mass_zero_frequency", ("acv", "heave"), ("acv", "heave"))
    with pytest.raises(TypeError, match="a row and a column"):
        r.coefficient("added_mass", ("acv", "heave"))


def test_generalized_mode_by_name():
    from regression_problems import lo_wavemaker

    r = lo_wavemaker().solve()
    assert r.coordinate("paddle", "paddle") == (6, 0) and r.coordinate("paddle", "heave") == (2, None)
    a = r.coefficient("added_mass", ("paddle", "paddle"), ("paddle", "paddle"))
    assert np.array_equal(a, r.added_mass[:, 6, 0])
    assert np.array_equal(r.bodies["paddle"].added_mass, r.added_mass)
    h, ref = r.bodies[0].hydrostatics, lo_wavemaker().bodies[0].hydrostatics()  # no weight
    assert not h.has_weight and h.mass is None and h.metacentric_height is None and h.center_of_gravity is None
    assert np.array_equal(h.stiffness, ref.stiffness) and h.displaced_mass == ref.displaced_mass


def test_lid_and_field_point_views_partition_the_global_arrays():
    lid = moonpool_lid_mesh()
    n = len(lid["vertices"])
    pts = np.array([[3.0, 0.5, 0.0], [2.5, 2.0, -0.4], [-2.5, 1.0, -0.2]])
    m = model({"name": "moonpool", "mesh": moonpool_mesh()}, W, (0.0,), field_points=pts,
              lids=[{"name": "inner", "mesh": part(lid, 1, n // 2), "damping_factor": 0.1},
                    {"name": "outer", "mesh": part(lid, n // 2 + 1, n), "damping_factor": 0.2}],
              options={"field_velocity": True, "field_pressure": True})  # fmt: skip
    r = m.solve()
    lids = r.lids
    assert lids.names() == ["inner", "outer"] and lids[1].name == "outer"
    rows = [np.flatnonzero(np.asarray(r.quantity("lids_ids")) == k + 1) for k in range(2)]
    assert all(x.size for x in rows) and sum(x.size for x in rows) == len(r.quantity("lids_ids"))
    for name, axis in (("points", 0), ("elevation_radiation", 1), ("elevation_diffraction", 1)):
        g = r.quantity("lids" if name == "points" else f"lids_{name}")
        joined = np.concatenate([getattr(lids[k], name) for k in ("inner", "outer")], axis=axis)
        assert np.array_equal(joined, np.take(g, np.concatenate(rows), axis=axis)), name
        assert all(np.shares_memory(getattr(lids[k], name), g) for k in range(2)), name  # each lid a slice
    f = r.field_points
    assert np.array_equal(f.points, pts) and f.points is r.quantity("field_points")
    for kind in ("radiation", "incident", "scattering", "diffraction"):
        assert getattr(f, f"pressure_{kind}") is r.quantity(f"field_points_pressure_{kind}")
        assert getattr(f, f"velocity_{kind}") is r.quantity(f"field_points_velocity_{kind}")
    with pytest.raises(KeyError, match="no lids 'gap'"):
        lids["gap"]


def test_tank_views_partition_the_tank_arrays():
    # the points of each tank under its body (body axes, as given), its radiation fields the rows of
    # the tanks_* arrays whose tanks_ids is it; the sea's points apart, in field_points
    m = two_tank_vessels(field_velocity=True)
    r = m.solve()
    ids = np.asarray(r.quantity("tanks_ids"))
    assert np.array_equal(ids, [1, 2]) and [len(b.tanks) for b in r.bodies] == [1, 1]
    for k, b in enumerate(r.bodies):
        t = b.tanks["t"]
        assert t.name == "t" and t.index == 0 and isinstance(t, hb.TankResults)
        assert np.array_equal(t.points, np.asarray(m.bodies[k].tanks[0].field_points))
        assert np.array_equal(t.points, r.quantity("tanks")[[k]])
        for name in ("pressure_radiation", "velocity_radiation"):
            assert np.array_equal(getattr(t, name), r.quantity(f"tanks_{name}")[:, [k]]), name
        # the radiation of a tank's points: the columns of its body alone
        _, cols = r._body_columns(k)
        other = np.setdiff1d(np.arange(len(r.radiation_mode_ids)), cols)
        assert np.all(t.pressure_radiation[..., other] == 0) and np.any(t.pressure_radiation[..., cols] != 0)
    assert np.array_equal(r.field_points.points, TANK_POINTS[[2]])
    assert r.field_points.pressure_radiation.shape[1] == 1 and r.field_points.velocity_radiation.shape[1] == 1
    with pytest.raises(KeyError, match="no tanks of a 'bilge'"):
        r.bodies[0].tanks["bilge"]
    c = m.counts()
    assert (c.field_points, c.tank_points) == (1, 2)


def test_the_centres_are_in_body_coordinates():
    # the centres of gravity and buoyancy in body coordinates, as the input: neither moves with the
    # rotation centre R; the stiffness about R does
    def solved(rc):
        return model({**lo_free_cylinder(), "rotation_center": rc}, W[:1], (0.0,)).solve()

    a, b = solved([0.0, 0.0, 0.0]), solved([0.3, -0.2, -0.4])
    for r in (a, b):
        h = r.bodies[0].hydrostatics
        assert np.array_equal(h.center_of_gravity, np.asarray(r.model.bodies[0].center_of_gravity, dtype=float))
        assert np.array_equal(r.hydrostatics.center_of_gravity[0], h.center_of_gravity)
        assert h.volume == pytest.approx(h.volumes.mean(), rel=1e-15)
        assert h.displaced_mass == r.hydrostatics.displaced_mass[0] == r.model.environment.density * h.volumes[0]
    np.testing.assert_allclose(a.hydrostatics.center_of_buoyancy, b.hydrostatics.center_of_buoyancy, rtol=0, atol=1e-12)
    assert not np.allclose(a.hydrostatics.stiffness, b.hydrostatics.stiffness)


def test_the_quantities_are_the_results_of_a_solve():
    names = hb.quantities()
    assert not [n for n in names if n.startswith(("panel_", "tessellation_", "quality_"))]
    assert "hydrostatics_stiffness" in names and "field_points_pressure_radiation" in names
    assert not {"field_radiation", "lid_centroids", "pressure_centroids"} & set(names)
    assert [q.code for q in names.values()] == list(range(len(names)))


def test_a_results_file_keeps_every_quantity(tmp_path, acvs):
    r = model(lo_free_cylinder(), W, (0.0, 30.0), field_points=[[3.0, 0.0, 0.0]],
              options={"motion": True, "drift_pressure_integration": True, "drift_momentum_conservation": True,
                       "field_velocity": True, "body_pressure": True, "body_velocity": True})  # fmt: skip
    for res in (acvs, r.solve()):
        res.save(tmp_path / "run.hbr", overwrite=True)
        back = hb.load_results(tmp_path / "run.hbr")
        have = [n for n in hb.quantities() if res.quantity(n) is not None]
        assert len(have) > 20
        assert [n for n in hb.quantities() if back.quantity(n) is not None] == have
        for n in have:
            a, b = res.quantity(n), back.quantity(n)
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), n
        assert np.array_equal(back.bodies[0].added_mass, res.bodies[0].added_mass)
