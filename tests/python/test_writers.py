"""The numeric files of the engine (reference/file-formats §2-§4) driven by results given as
arrays (``Results.of``): layout, normalization, availability rules, collisions, overwrite and
atomic writes. The Fortran fields themselves are unit-tested in crates/hydrobem/tests/io.rs; here a
reference formatter of this file states what each field must read."""

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest

import hydrobem as hb

from bodies import moonpool_lid_mesh, moonpool_mesh, surface
from meshes import box_panels, flat_plate
from problems import model, written

L, RHO, G = 2.0, 1025.0, 9.81
STAMP = dt.datetime(2026, 10, 24, 12, 30, 5)


def e_1p(x, width=14):
    """1PE<width>.6: one digit before the point (the reference of the tests)."""
    x = float(x) or 0.0
    return f"{x:.6E}".rjust(width)


def e_0p(x, width=14):
    """E<width>.6: 0.dddddd and the exponent."""
    x = float(x) or 0.0
    if x == 0.0:
        return "0.000000E+00".rjust(width)
    m, e = f"{abs(x):.5E}".split("E")
    text = f"0.{m.replace('.', '')}E{int(e) + 1:+03d}"
    return (("-" if x < 0 else "") + text).rjust(width)


def i_w(n, width):
    return str(n).rjust(width)


def test_reference_formatter():
    assert e_0p(-123.4567) == " -0.123457E+03" and e_0p(0.99999951) == "  0.100000E+01"
    assert e_1p(9.9999996) == "  1.000000E+01" and e_1p(-0.0) == "  0.000000E+00"


# ------------------------------------------------------------- synthetic results


def _bodies(cg=True):
    mesh = surface(box_panels(), reference_length=L)
    inertia = dict(inertia_input="components", Ixx=1.0, Ixy=0.0, Ixz=0.0, Iyy=2.0, Iyz=0.0, Izz=3.0)
    b1 = {"name": "a", "mesh": mesh, **({"mass": "displaced", "center_of_gravity": [0, 0, -0.1],
                                         **inertia} if cg else {})}
    b2 = {"name": "b", "mesh": mesh, "position": [6, 0, 0], "modes": ["heave"], "mass": "displaced",
          "center_of_gravity": [0, 0, 0],
          **inertia}  # fmt: skip
    return [b1, b2]


def make_problem(omegas=(0.0, 0.5, 1.25, math.inf), rao=True, field=True, cg=True, **opt):
    fp = np.array([[0.0, 4.0, 0.0], [10.0, 0.0, 0.0], [3.0, -2.0, 0.0]]) if field else None
    headings = (0.0, 45.0) if opt.get("diffraction", True) else ()  # a valid model: headings with diffraction
    m = model(_bodies(cg), omegas, headings, field_points=fp, environment={"density": RHO, "gravity": G},
              options={"motion": rao, **opt})  # fmt: skip
    return m


def make_arrays(m, haskind=True, diffraction=True, seed=1):
    """Made-up arrays of every quantity the files take, by engine name, on the axes of the model."""
    rng = np.random.default_rng(seed)
    omegas = np.asarray(m.environment.omegas)
    F = int(np.sum(np.isfinite(omegas) & (omegas > 0)))
    H, D, K, B = len(m.environment.headings), 12, 7, 2

    def cplx(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    a = {"added_mass": rng.normal(size=(F, D, K)), "radiation_damping": np.abs(rng.normal(size=(F, D, K))),
         "hydrostatics_stiffness": rng.normal(size=(B, 6, 6)) * 1e4,
         "hydrostatics_stiffness_rho_g": np.zeros((B, 6, 6)),
         "hydrostatics_volumes": np.ones((B, 3)), "hydrostatics_volume": np.ones(B),
         "hydrostatics_displaced_mass": np.full(B, RHO),
         "hydrostatics_center_of_buoyancy": np.tile([0.0, 0.0, -0.25], (B, 1)),
         "hydrostatics_waterplane_area": np.full(B, 2.0), "hydrostatics_waterplane_centroid": np.zeros((B, 2)),
         "hydrostatics_mass": np.full(B, 1000.0),
         "hydrostatics_center_of_gravity": np.tile([0.0, 0.0, -0.1], (B, 1)),
         "hydrostatics_metacentric_height": np.ones((B, 2))}  # fmt: skip
    for k, w in (("zero", 0.0), ("infinite", math.inf)):
        if np.any(omegas == w):
            a[f"added_mass_{k}_frequency"] = rng.normal(size=(D, K))
    if diffraction:
        a["excitation_froude_krylov"], a["excitation_scattering"] = cplx(F, H, D), cplx(F, H, D)
        a["excitation_diffraction"] = (a["excitation_froude_krylov"].astype(np.complex64)
                                       + a["excitation_scattering"].astype(np.complex64))  # fmt: skip
        if haskind:
            a["excitation_haskind"] = cplx(F, H, K)
        if m.first_order.motion:
            a["motion"] = cplx(F, H, D)
    if m.first_order.field_points is not None:
        P = m.first_order.field_points.shape[0]
        a["field_points_pressure_radiation"] = cplx(F, P, K)
        a["field_points_pressure_diffraction"], a["field_points_pressure_incident"] = cplx(F, P, H), cplx(F, P, H)
    return a


def make_results(m, **kw):
    return hb.Results.of(m, **make_arrays(m, **kw))  # results made up for the writers: a model not solved


@pytest.fixture
def results():
    return make_results(make_problem())


def test_quantities_read_together_are_given_together(tmp_path):
    # a drift with its moment about the moving origin; body point ids in range
    m = make_problem()
    a = make_arrays(m)
    F = int(np.sum(np.isfinite(m.environment.omegas) & (m.environment.omegas > 0)))  # the ordinary ones
    H, B = len(m.environment.headings), len(m.bodies)
    with pytest.raises(ValueError, match=r"drift_pressure_integration and \w+_body_axes are given together"):
        hb.Results.of(m, **a, drift_pressure_integration=np.zeros((F, H, 6 * B), complex)).write_wamit(tmp_path / "c")
    with pytest.raises(ValueError, match="body_points_ids: row 1"):
        hb.Results.of(m, **a, body_points_ids=np.array([[7, 1, 1, 0]], np.int32)).write_wamit(tmp_path / "d")


def test_given_results_have_no_meta(results):
    # results not from a solve: no meta data
    assert results.meta is None and results.route == "given"


def read_lines(path):
    data = path.read_bytes()
    assert data.endswith(b"\r\n")
    assert b"\n" not in data.replace(b"\r\n", b"")
    return data.decode("ascii").split("\r\n")[:-1]


# ------------------------------------------------------------- files and layout


def test_written_files_and_headers(tmp_path, results):
    results.write_wamit(tmp_path / "case", timestamp=STAMP)
    paths = written(tmp_path / "case")
    names = sorted(p.name for p in paths)
    assert names == sorted(
        f"case{e}" for e in (".1", ".2", ".2fk", ".2sc", ".3", ".3fk", ".3sc", ".4", ".6p", ".fpt", ".hst", ".out")
    )
    for p in paths:
        lines = read_lines(p)
        if p.suffix != ".out":
            assert "HydroBEM" in lines[0] and p.name in lines[0] and "2026-10-24 12:30:05" in lines[0]
    assert not list(tmp_path.glob(".hydrobem-*.tmp"))


def test_dot1_layout_and_normalization(tmp_path, results):
    results.write_wamit(tmp_path / "c", timestamp=STAMP)
    lines = read_lines(tmp_path / "c.1")[1:]
    ids = results.radiation_mode_ids.tolist()
    assert ids == [1, 2, 3, 4, 5, 6, 9]
    D, K = 12, len(ids)
    # input order: omega = 0, 0.5, 1.25, inf
    assert len(lines) == 4 * D * K
    zero, f05, f125, inf = (lines[i * D * K : (i + 1) * D * K] for i in range(4))
    assert all(len(x) == 40 for x in zero + inf) and all(len(x) == 54 for x in f05 + f125)
    assert zero[0][:14] == e_0p(-1.0) and inf[0][:14] == e_0p(0.0)
    assert f05[0][:14] == e_0p(2 * math.pi / 0.5)
    # row I = 4 (roll), column J = 9 (heave of body b): r_i = 1, r_j = 0
    row = f125[3 * K + 6]
    assert int(row[14:20]) == 4 and int(row[20:26]) == 9
    a = float(results.added_mass[1, 3, 6]) / (RHO * L ** (3 + 1))
    b = float(results.radiation_damping[1, 3, 6]) / (1.25 * RHO * L ** (3 + 1))
    assert row[26:40] == e_1p(a) and row[40:54] == e_1p(b)
    z = float(results.added_mass_zero_frequency[4, 4]) / (RHO * L**5)
    assert zero[4 * K + 4][26:40] == e_1p(z)


def test_force_files(tmp_path, results):
    results.write_wamit(tmp_path / "c", timestamp=STAMP)
    F, H, D, K = 2, 2, 12, 7
    l3 = read_lines(tmp_path / "c.3")[1:]
    assert len(l3) == F * H * D and all(len(x) == 90 for x in l3)
    row = l3[(1 * H + 1) * D + 4]  # f = 1, heading 45, coordinate 5 (pitch)
    assert row[14:28] == e_0p(45.0) and int(row[28:34]) == 5
    x = complex(results.excitation.diffraction[1, 1, 4]) / (RHO * G * L**3)
    assert row[34:48] == e_1p(abs(x)) and row[48:62] == e_1p(math.degrees(math.atan2(x.imag, x.real)))
    assert row[62:76] == e_1p(x.real) and row[76:90] == e_1p(x.imag)
    l2 = read_lines(tmp_path / "c.2")[1:]
    assert len(l2) == F * H * K
    assert [int(x[28:34]) for x in l2[:K]] == [1, 2, 3, 4, 5, 6, 9]
    hk = complex(results.excitation.haskind[0, 1, 6]) / (RHO * G * L**2)
    assert l2[1 * K + 6][76:90] == e_1p(hk.imag)
    sc = hk - complex(results.excitation.froude_krylov[0, 1, 8]) / (RHO * G * L**2)
    assert read_lines(tmp_path / "c.2sc")[1:][1 * K + 6][62:76] == e_1p(sc.real)
    l4 = read_lines(tmp_path / "c.4")[1:]
    xi = complex(results.motion[0, 0, 3]) * L
    assert l4[3][62:76] == e_1p(xi.real)


def test_field_files(tmp_path, results):
    results.write_wamit(tmp_path / "c", timestamp=STAMP)
    F, H, P, K = 2, 2, 3, 7
    lines = read_lines(tmp_path / "c.6p")[1:]
    assert len(lines) == F * (P + H * P)
    rad = lines[0]
    assert len(rad) == 14 + 10 + 2 * K * 14 and int(rad[14:24]) == 1
    z = complex(results.field_points.pressure_radiation[0, 0, 3]) / L  # roll column
    assert rad[24 + 6 * 14 : 24 + 7 * 14] == e_1p(z.real)
    dif = lines[P + P + 1]  # second heading, second point
    assert dif[14:28] == e_0p(45.0) and int(dif[28:38]) == 2
    zt = complex(results.field_points.pressure_diffraction[0, 1, 1])
    assert dif[38:52] == e_1p(zt.real) and dif[52:66] == e_1p(zt.imag)
    fpt = read_lines(tmp_path / "c.fpt")[1:]
    assert fpt[2] == i_w(3, 10) + e_1p(3.0) + e_1p(-2.0) + e_1p(0.0)


def test_hst(tmp_path, results):
    results.write_wamit(tmp_path / "c", timestamp=STAMP)
    lines = read_lines(tmp_path / "c.hst")[1:]
    assert len(lines) == 144 and all(len(x) == 27 for x in lines)
    c45 = results.hydrostatics.stiffness[1][3, 4] / (RHO * G * L**4)
    assert lines[(6 + 3) * 12 + 6 + 4] == i_w(10, 6) + i_w(11, 6) + e_1p(c45, width=15)
    assert float(lines[0 * 12 + 6][12:]) == 0.0  # no coupling between bodies


def test_omega_convention(tmp_path, results):
    results.write_wamit(tmp_path / "c", frequency="omega", timestamp=STAMP)
    lines = read_lines(tmp_path / "c.3")[1:]
    assert lines[0][:14] == e_0p(0.5)
    with pytest.raises(ValueError, match="frequency must be 'period' or 'omega'"):
        results.write_wamit(tmp_path / "d", frequency="hertz")


def test_out_summary(tmp_path, results):
    results.write_wamit(tmp_path / "c", timestamp=STAMP)
    text = "\n".join(read_lines(tmp_path / "c.out"))
    for key in (
        "HydroBEM",
        "Gravity:",
        "Length scale:",
        "Water depth:",
        "Water density:",
        "Volumes (VOLX,VOLY,VOLZ):",
        "Center of Buoyancy (Xb,Yb,Zb):",
        "Center of Gravity  (Xg,Yg,Zg):",
        "XBODY =",
        "Wave period (sec) =",
        "ADDED-MASS AND DAMPING COEFFICIENTS",
        "HASKIND EXCITING FORCES AND MOMENTS",
        "DIFFRACTION EXCITING FORCES AND MOMENTS",
        "RESPONSE AMPLITUDE OPERATORS",
        "Wave Heading (deg) :",
        "ZERO-FREQUENCY ADDED MASS",
        "INFINITE-FREQUENCY ADDED MASS",
        "Radiation mode order",
    ):
        assert key in text, key
    assert text.count("Wave period (sec) =") == 2


@pytest.mark.parametrize("damper", [False, True])
def test_out_summary_of_lids_and_formulation(tmp_path, damper):
    # the formulation the solve used (auto: total with lids, scattered with the lids and a thin sheet
    # of the low order) and one line per lid, every line ended by CRLF (read_lines)
    lids = [{"name": "moon", "mesh": moonpool_lid_mesh(), "damping_factor": 0.2}]
    sheet = [{"mesh": surface(flat_plate(1.0, 1.0, -1.0, center=(4.0, 0.0)), symmetry="xz_and_yz"),
              "damping_factor": 1.0}]
    m = model({"mesh": moonpool_mesh(), "modes": ["heave"]}, [1.5], [0.0], lids=lids,
              dampers=sheet if damper else ())  # fmt: skip
    m.solve().write_wamit(tmp_path / "c", timestamp=STAMP)
    text = read_lines(tmp_path / "c.out")
    assert f" Diffraction formulation: {'scattered' if damper else 'total'}" in text
    assert " Physical damping lids: 1" in text
    assert any(line.startswith("   lid 'moon': damping factor") for line in text)
    assert text[1].strip().endswith("summary") and text[5] in (" Low-order panel method", " Higher-order method")


# ------------------------------------------------------------- availability rules


def test_availability(tmp_path):
    r = make_results(make_problem(omegas=(0.5, 1.0), rao=False, field=False, cg=False), haskind=False)
    r.write_wamit(tmp_path / "c", timestamp=STAMP)
    names = {p.suffix for p in written(tmp_path / "c")}
    assert names == {".1", ".3", ".3fk", ".3sc", ".out"}  # no Haskind, RAO, fields; .hst needs every CG
    rad_only = make_results(
        make_problem(omegas=(0.0, 0.5), rao=False, field=False, diffraction=False), diffraction=False
    )
    rad_only.write_wamit(tmp_path / "r", timestamp=STAMP)
    names = {p.suffix for p in written(tmp_path / "r")}
    assert names == {".1", ".hst", ".out"}


def test_rao_rule_and_shapes(tmp_path):
    p = make_problem(rao=False)
    with pytest.raises(ValueError, match="first_order.motion is false"):
        hb.Results.of(p, **make_arrays(p), motion=np.zeros((2, 2, 12))).write_wamit(tmp_path / "c")
    p = make_problem()
    a = make_arrays(p)
    with pytest.raises(ValueError, match="shape"):
        short = {**a, "excitation_diffraction": a["excitation_diffraction"][:, :1]}
        hb.Results.of(p, **short).write_wamit(tmp_path / "c")
    del a["added_mass_infinite_frequency"]
    with pytest.raises(ValueError, match="endpoint"):
        hb.Results.of(p, **a).write_wamit(tmp_path / "c")
    assert not list(tmp_path.iterdir())


def test_collisions(tmp_path):
    r = make_results(make_problem(omegas=(1.0, 1.00000001), rao=False, field=False))
    with pytest.raises(ValueError, match="collide"):
        r.write_wamit(tmp_path / "c")
    m = model(_bodies(), [1.0], [10.0, 10.0000001])
    with pytest.raises(ValueError, match="headings collide"):
        make_results(m).write_wamit(tmp_path / "c")
    assert not list(tmp_path.iterdir())


def test_overflow_writes_nothing(tmp_path):
    p = make_problem()
    a = make_arrays(p)
    a["added_mass"][0, 0, 0] = 1e38
    hb.Results.of(p, **a).write_wamit(tmp_path / "ok", timestamp=STAMP)  # 1e38 / (rho L^3) fits
    a["hydrostatics_stiffness"][0, 2, 2] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        hb.Results.of(p, **a).write_wamit(tmp_path / "bad")
    a["hydrostatics_stiffness"][0, 2, 2] = 1e300
    with pytest.raises(ValueError, match="three-digit exponent"):
        hb.Results.of(p, **a).write_wamit(tmp_path / "bad")
    assert not list(tmp_path.glob("bad*")) and not list(tmp_path.glob(".hydrobem-*"))


def test_overwrite_rules(tmp_path, results):
    stem = tmp_path / "c"
    results.write_wamit(stem, timestamp=STAMP)
    with pytest.raises(FileExistsError, match="overwrite"):
        results.write_wamit(stem, timestamp=STAMP)
    results.write_wamit(stem, timestamp="later", overwrite=True)
    assert "later" in read_lines(tmp_path / "c.3")[0]
    no_rao = make_results(make_problem(rao=False))
    with pytest.raises(FileExistsError, match=r"c\.4 exists but its quantity is not computed"):
        no_rao.write_wamit(stem, overwrite=True)
    # a missing folder is made, once the files are made
    results.write_wamit(tmp_path / "new" / "c", timestamp=STAMP)
    assert (tmp_path / "new" / "c.out").is_file()


def test_normalized(results):
    assert np.isclose(results.normalized("added_mass")[0, 4, 4], results.added_mass[0, 4, 4] / (RHO * L**5))
    assert np.isclose(results.normalized("excitation_diffraction")[0, 0, 2],
                      results.excitation.diffraction[0, 0, 2] / (RHO * G * L**2))  # fmt: skip
    assert np.isclose(results.normalized("motion")[0, 0, 5], results.motion[0, 0, 5] * L)
    assert "HydroBEM results" in results.summary()


def test_results_are_read_only(results):
    with pytest.raises(ValueError, match="read-only"):
        results.added_mass[0, 0, 0] = 1.0
    with pytest.raises(AttributeError, match="added_mass"):
        results.added_mass = None


def test_pnl_normals_and_moments_are_in_body_axes(tmp_path):
    # the centroid in global coordinates, the normal and r x n in the
    # body's axes, so a body moved and yawed lists the same normals and moments
    def pnl(position, yaw, name):
        body = {"mesh": surface(box_panels(1.0, 0.5, 0.5, n=(2, 1, 1), quarter=True), symmetry="xz_and_yz"),
                "modes": [],
                "position": position, "yaw": yaw}  # fmt: skip
        r = model(body, [1.0], [0.0], options={"body_pressure": True}).solve()
        r.write_wamit(tmp_path / name)
        return np.array([[float(v) for v in x.split()[2:]] for x in read_lines(tmp_path / f"{name}.pnl")[1:]])  # header

    a, b = pnl([0.0, 0.0, 0.0], 0.0, "a"), pnl([10.0, -3.0, 0.0], 30.0, "b")
    # area, normal, r x n: the same to the 7 significant digits of the file (values below 10 to 1e-6)
    np.testing.assert_allclose(b[:, 3:], a[:, 3:], rtol=0, atol=1e-6)
    assert np.abs(b[:, :3] - a[:, :3]).max() > 1.0  # the centroids are global


def test_point_and_velocity_files_from_a_solve(tmp_path):
    # .5p, .5vx-z, .pnl, .6vx-z and .7 from a solve, with the rows of
    # reference/file-formats §2
    body = {"mesh": surface(box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True), symmetry="xz_and_yz"), "modes": [],
            "control_surface": {"method": "automatic", "radius": 2.0, "depth": 1.0, "panel_size": 0.5}}
    m = model(body, [1.0, 2.0], [0.0], field_points=[[3.0, 0.0, 0.0], [2.0, 2.0, -0.5]],
              options={"body_pressure": True, "body_velocity": True, "field_pressure": True, "field_velocity": True,
                       "drift_control_surface": True})  # fmt: skip
    r = m.solve()
    r.write_wamit(tmp_path / "c")
    names = {p.suffix for p in written(tmp_path / "c")}
    assert {".5p", ".5vx", ".5vy", ".5vz", ".pnl", ".6p", ".6vx", ".6vy", ".6vz", ".fpt", ".7"} <= names
    # no radiation columns (a fixed body): no .1, .2 files and no radiation tables in .out
    assert not names & {".1", ".2", ".2fk", ".2sc"}
    out = (tmp_path / "c.out").read_text(encoding="utf-8")
    assert "ADDED-MASS" not in out and "HASKIND" not in out and "Radiation (per unit" not in out
    n = len(r.body_points.points)  # every panel of the quadrant and its reflections
    H, F, P = 1, 2, 2
    assert len(read_lines(tmp_path / "c.pnl")) - 1 == n
    assert len(read_lines(tmp_path / "c.5p")) - 1 == F * H * n  # the diffraction rows (no radiation: fixed)
    for e in (".6vx", ".6vy", ".6vz"):
        assert len(read_lines(tmp_path / f"c{e}")) - 1 == F * (H * P), e
    rows7 = read_lines(tmp_path / "c.7")[1:]
    assert len(rows7) == F * H * 9 and {int(x.split()[3]) for x in rows7} == {1, 2, 3, 4, 5, 6, -4, -5, -6}


def test_given_results_have_the_shapes_of_the_model():
    # results given as arrays are checked against the model's axes, so
    # that the normalization and the files never read outside an array
    p = make_problem()
    a = make_arrays(p)
    with pytest.raises(ValueError, match=r"added_mass has the shape \(1, 40, 7\), not that of its axes"):
        hb.Results.of(p, added_mass=np.ones((1, 40, 7)))
    with pytest.raises(ValueError, match="drift_pressure_integration has the shape"):
        hb.Results.of(p, drift_pressure_integration=np.ones((3, 2, 12), complex))
    for axis in ("omegas", "headings", "radiation_mode_ids", "coordinates"):
        with pytest.raises(ValueError, match=f"{axis} is an axis of the model, not given"):
            hb.Results.of(p, **{axis: [1.0, 2.0]})
    with pytest.raises(TypeError, match="added_mass is real, got complex values"):
        hb.Results.of(p, added_mass=a["added_mass"] * 1j)
    del a["hydrostatics_center_of_gravity"]
    with pytest.raises(ValueError, match="the hydrostatics are given in part"):
        hb.Results.of(p, **a).write_wamit(Path(".") / "never")
    r = hb.Results.of(p, **{k: v for k, v in a.items() if not k.startswith("hydrostatics_")})
    assert np.isfinite(r.normalized("added_mass")).all()


def test_field_point_file_radiation_block(tmp_path):
    # the .6p radiation rows of a moving body, one per frequency and point with the real and
    # imaginary parts of every column, hold the normalized field values (reference/file-formats §2)
    body = {"mesh": surface(box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True), symmetry="xz_and_yz"),
            "modes": ["surge", "heave"]}
    m = model(body, [1.0, 2.0], [0.0], field_points=[[3.0, 0.0, 0.0], [2.0, 2.0, -0.5]],
              options={"field_pressure": True})
    r = m.solve()
    r.write_wamit(tmp_path / "c")
    rows = [x.split() for x in read_lines(tmp_path / "c.6p")[1:]]
    F, P, K, H = 2, 2, 2, 1
    assert len(rows) == F * (P + H * P)
    rad = r.normalized("field_points_pressure_radiation")
    for f in range(F):
        for p in range(P):
            row = rows[f * (P + H * P) + p]
            assert int(row[1]) == p + 1 and len(row) == 2 + 2 * K
            got = np.array([float(row[2 + 2 * k]) + 1j * float(row[3 + 2 * k]) for k in range(K)])
            # the 7 significant digits of the file: half a unit of the last, 5e-7 of each part
            # (measured 4.1e-7), at most 1e-6 of the modulus
            np.testing.assert_allclose(got, rad[f, p], rtol=1e-6, atol=0)
