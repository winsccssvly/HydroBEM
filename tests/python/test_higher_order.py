"""The higher-order method (one body in open sea, flat and
B-spline patches, any depth, limits, field points, RAO, interior free-surface
and prescribed-pressure patches, the item panel_size, walls, several bodies).

theory §7 / test-plan B, each test on the basis it rests on: identities of the problem (symmetry
sectors and the symmetry of a system equal the full solve, a wall equals the mirror body, the total
and scattered formulations, Haskind and pressure exciting forces, also for the pressure modes and
thin sheets, the axisymmetry of the heave flow next to the pole of a cylinder's bottom patch, a
closed sheet side split like any side, the thread counts); the closed forms of the pressure-mode
restoring; the analytic irregular frequency of a box, removed by an interior patch; the input rules
(panel_size, routes that are not available). The higher order against the extrapolated low order
of the same geometry (the B-spline cylinder, the air-cushion vessel, the damping lid, thin sheets)
and the quadrature orders against each other are validation, run when the method changes, not
tests. Meshes are procedural (tests/python/meshes.py)."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import W_ACV, flat, full_cylinder, ho_acv, part, plate_column, spline, spline_cylinder, surface
from meshes import (
    ACV_FACES,
    box_panels,
    cylinder_panels,
    face_panels,
    spline_cylinder_patches,
)
from problems import RHO, add_mesh, model, peak, view

G = 9.80665  # the default environment: RHO, G, infinite depth
HO = {"higher_order": True, "diffraction_formulation": "total"}


def solve(mesh, w, heads=None, opts=HO, body=None, **kw):
    body = body if body is not None else {"mesh": mesh, **kw}
    heads = heads if heads is not None else ((0.0,) if opts.get("diffraction", True) else ())
    return model(body, w, list(heads), options=opts).solve()


def test_symmetry_sectors_equal_full_solve():
    # the sector decomposition is exact for the discrete equations (theory §6.3); 1e-5: the float32
    # matrices and the character sums in another order (3.6e-9 measured)
    quarter = box_panels(2.0, 1.0, 1.0, n=(1, 1, 1), quarter=True)
    full = []
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        for q in quarter:
            r = q * [sx, sy, 1]
            full.append(r[[0, 3, 2, 1]] if sx * sy < 0 else r)
    disc = (3, 3, 3, 3, 3, 4)
    w = np.sqrt(np.array([0.6, 1.5]) * G)
    a = solve(flat(quarter, disc, symmetry="xz_and_yz"), w, heads=(0.0, 30.0))
    b = solve(flat(np.array(full), disc), w, heads=(0.0, 30.0))
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        x, y = view(a, q), view(b, q)
        assert peak(x - y) <= 1e-5 * peak(y), q


def test_total_and_scattered_formulations_agree():
    # one continuous diffraction problem in two unknowns (the total and the scattered potential): they
    # differ by the discretization, 2e-3 (1.6e-4 measured; a wrong term gives O(1)); the radiation
    # problem is the same equations, bit-identical
    w =np.sqrt(np.array([0.5, 1.2, 2.0]) * G)
    tot = solve(spline_cylinder(), w, heads=(0.0, 45.0))
    sca = solve(spline_cylinder(), w, heads=(0.0, 45.0), opts={"higher_order": True})
    assert peak(tot.excitation.diffraction - sca.excitation.diffraction) <= 2e-3 * peak(tot.excitation.diffraction)
    np.testing.assert_array_equal(tot.added_mass, sca.added_mass)


def test_haskind_equals_pressure_force():
    # the Haskind relation and the reciprocity of the radiation potentials (Newman 1977) hold in the
    # continuous problem; 2e-3: the patch quadrature of the two forms at the waterline (Haskind 1.0e-4,
    # the symmetry of A 1.3e-4 measured); B >= 0 on the diagonal (no energy created) to rounding
    w =np.sqrt(np.array([0.5, 1.2, 2.0]) * G)
    r = solve(spline_cylinder(n=8), w)
    hk = r.excitation.haskind
    assert peak(hk - r.excitation.diffraction) <= 2e-3 * peak(r.excitation.diffraction)
    A, B = r.added_mass, r.radiation_damping
    assert peak(A - A.transpose(0, 2, 1)) <= 2e-3 * peak(A)
    assert np.all(np.diagonal(B, axis1=1, axis2=2) >= -1e-6 * peak(B))


def test_pressure_patches_haskind_and_restoring():
    # the Haskind relation for the pressure modes (theory §9.3): 2e-3, the patch quadrature at the
    # waterline (2.0e-4 measured); the restoring of the modes against its closed forms over the
    # cushion of 4 x 9 x 4 m (theory §9.3), exact but for rounding (1e-10)
    r = solve(None, W_ACV, heads=(150.0,), body=ho_acv())
    hk = r.excitation.haskind  # [F, H, K]
    for i in range(8):
        x = r.excitation.diffraction[:, 0, i]
        assert peak(hk[:, 0, i] - x) <= 2e-3 * peak(x), i
    assert np.all(np.diagonal(r.radiation_damping, axis1=1, axis2=2)[:, 6:] >= 0)
    rg, area = RHO * G, 4 * 9.0 * 4.0  # the whole cushion
    C = r.mode_restoring
    assert C[7, 7] == pytest.approx(-rg * area, rel=1e-10)  # uniform mode, variational
    assert C[6, 6] == pytest.approx(-rg * area, rel=1e-10)  # (sign x)^2 = 1
    assert C[2, 7] == pytest.approx(rg * area, rel=1e-10) and C[7, 2] == pytest.approx(rg * area, rel=1e-10)
    assert abs(C[2, 6]) <= 1e-9 * rg * area  # the antisymmetric mode has no heave force
    assert C[4, 6] == pytest.approx(-rg * 4 * 4.0 * 0.5 * 9.0**2, rel=1e-10)  # pitch: -rho g int |x|
    n = r.quantity("pressure_surfaces").shape[0]
    assert n == 4 * 10 * 5 * 16 and r.quantity("pressure_surfaces_elevation_diffraction").shape == (3, n, 1)
    numeric = solve(None, W_ACV[:1], heads=(150.0,), body=ho_acv(convention="numeric_compatibility"))
    Cn = numeric.mode_restoring
    assert Cn[7, 7] == pytest.approx(rg * area, rel=1e-10) and Cn[7, 2] == 0.0


def test_pressure_patch_with_an_interior_patch():
    # a moonpool cylinder (radius 1, draft 1, moonpool 0.5) whose inner free surface is a pressure
    # patch with a uniform mode, with and without the annulus between the walls as its interior
    # patch: away from the irregular frequencies the interior patch leaves the solution (an identity
    # of the continuous problem), 2e-3 of the peaks, the discretization of the interior patch (3.0e-4
    # measured; 2.1e-5 at twice the spans)
    def solve_owc(interior):
        m = hb.Model()
        parts = ["side", "bottom", "moonpool_wall", "moonpool_surface"]
        spans = {"side": (2, 3), "bottom": (2, 1), "moonpool_wall": (1, 3), "moonpool_surface": (1, 1)}
        m.meshes.analytic("owc", "moonpool_cylinder", [[*spans[p], 3, 3, 3, 4] for p in parts], parts=parts,
                          radius=1.0, draft=1.0, moonpool_radius=0.5)  # fmt: skip
        kw = {}
        if interior:
            wp = m.meshes.add("annulus", kind="analytic_patches", discretization=[[2, 1, 3, 3, 3, 4]],
                              symmetry="xz_and_yz")  # fmt: skip
            wp.patches.add(type="AnalyticPatch", map="line", profile="linear", a=1.0, b=1.0, theta0=0.0,
                           theta1=0.5 * math.pi, p=[0.5, 0.0, 1.0, 0.0])  # fmt: skip
            kw = {"interior_surface": "mesh", "interior_surface_mesh": "annulus"}
        body = m.bodies.add("owc", mesh="owc", ranges=[(1, 3)], modes=["surge", "heave", "pitch"], **kw)
        body.pressure_surfaces.add("inner", mesh="owc", ranges=[(4, 4)]).pressure_modes.add("p0", shape="uniform")
        m.environment.update(frequency_input="omegas", omegas=np.sqrt(np.array([0.8, 1.5]) * G), headings=[0.0])
        m.options.update(higher_order=True)
        return m.solve()

    plain, irr = solve_owc(False), solve_owc(True)
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        x, y = view(irr, q), view(plain, q)
        assert peak(x - y) <= 2e-3 * peak(y), q


def test_interior_patch_removes_the_irregular_frequency():
    # the box 2 x 1 x 0.5 m: first symmetric irregular frequency, the analytic eigenvalue of the
    # interior Dirichlet problem K = k coth(k d), k = pi (1/4 + 1)^(1/2), K = 3.729. Away from it the
    # interior patch leaves the exterior solution (an identity), 3e-4: the discretization of the
    # interior patch (5.4e-5 measured); across it the second difference of B33 over K = 3.72, 3.73,
    # 3.74 is that of a smooth curve, 1e-3 (1.2e-4 measured), against 8.9e-2 of the spike without it
    q = box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True, lid=True)
    disc = [(*nv, 3, 3, 3, 4) for nv in ((4, 2), (2, 2), (4, 2), (4, 2))]
    K = np.array([2.0, 3.72, 3.73, 3.74])
    w = np.sqrt(K * G)
    plain = solve(flat(q[:3], disc[:3], symmetry="xz_and_yz"), w)
    box = flat(q, disc, symmetry="xz_and_yz")  # the interior patch: element 4
    lid = solve(part(box, 1, 3), w, interior_surface=part(box, 4, 4))
    b, bl = plain.radiation_damping[:, 2, 2], lid.radiation_damping[:, 2, 2]
    assert abs(bl[0] - b[0]) <= 3e-4 * abs(b[0])  # away from it the interior patch changes nothing
    assert abs(bl[1] - 2 * bl[2] + bl[3]) <= 1e-3 * bl[2]  # smooth across it
    assert abs(b[1] - 2 * b[2] + b[3]) >= 2e-2 * b[2]  # the spike without it


def test_interior_patch_with_shared_gauss_nodes():
    # a guard against a defect: IQUO = IQUI (or both odd) puts outer Gauss points on inner ones of
    # the z = 0 patch, where the wave kernel must take its regular limit without psi (and so the
    # subtraction); the logarithm at r = 0 gave an O(1) error. The rule with shared nodes (4, 4)
    # equals one without them (4, 6) to the inner rule's quadrature error, 1e-5 (2.9e-6 measured)
    q = box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True, lid=True)
    w = np.sqrt(np.array([2.0]) * G)
    a = {}
    for iq in ((4, 4), (4, 6)):
        disc = [(*nv, 3, 3, *iq) for nv in ((4, 2), (2, 2), (4, 2), (4, 2))]
        box = flat(q, disc, symmetry="xz_and_yz")
        r = solve(part(box, 1, 3), w, interior_surface=part(box, 4, 4))
        a[iq] = r.added_mass[0, 2, 2] + 1j * r.radiation_damping[0, 2, 2]
    assert abs(a[(4, 4)] - a[(4, 6)]) <= 1e-5 * abs(a[(4, 6)])


def test_panel_size_discretization(tmp_path):
    # panel_size: nu = floor(L / size) + 1 with the longer patch edge in each direction
    gdf = tmp_path / "acv.gdf"
    v = np.array([face_panels(o, du, dv, 1, 1, n)[0] for o, du, dv, n in ACV_FACES])
    lines = ["acv", "1.0 9.80665", "1 1", f"{len(v)} 0"] + [" ".join(f"{c:.6f}" for c in p) for q in v for p in q]
    gdf.write_text("\n".join(lines) + "\n")
    m = hb.Model()
    d = m.meshes.load("acv", gdf, higher_order=True, panel_size=1.0).discretization  # rows nu nv ku kv iquo iqui
    # edges (u, v): 5 x 2, 2 x 10, 5 x 1, 1 x 9, 1 x 4, 9 x 1, 4 x 9 m
    assert [tuple(r[:2]) for r in d.tolist()] == [(6, 3), (3, 11), (6, 2), (2, 10), (2, 5), (10, 2), (5, 10)]
    assert {tuple(r[2:]) for r in d.tolist()} == {(3, 3, 3, 4)}
    with pytest.raises(ValueError, match="one of"):
        m.meshes.load("acv_2", gdf, higher_order=True, panel_size=1.0, discretization=d)
    with pytest.raises(ValueError, match=r"more than 10\^6 panels"):  # not an overflow
        m.meshes.load("acv_3", gdf, higher_order=True, panel_size=1e-9)
    # B-spline patches: the arc lengths of the boundary curves (the spline cylinder of
    # spline_cylinder_patches: quarter arc pi/2, height 0.5, radius 1; an edge of an exact
    # multiple of the size counts that many panels)
    add_mesh(m, "spline", spline(spline_cylinder_patches(1.0, 0.5, spans=4, order=4), (1, 1, 3, 3, 3, 4)))
    m.meshes["spline"].write(tmp_path / "spline.gdf")
    d = m.meshes.load("read", tmp_path / "spline.gdf", higher_order=True, panel_size=0.2).discretization
    assert [tuple(r[:2]) for r in d.tolist()] == [(8, 3), (8, 6)]


def test_wall_equals_the_mirror_body():
    # a body beside the wall x = 0 is the body and its mirror image in open sea (the method of
    # images, exact for the discrete equations) moving as the mirror image: heave and pitch columns
    # of the pair (pitch mirrors to minus pitch), the excitation of the incident wave and its
    # reflection; 1e-5: the float32 matrices summed in another order (8e-8 measured)
    w = np.sqrt(np.array([0.6, 1.4]) * G)
    beta = 30.0
    wall = model({"mesh": spline_cylinder(), "position": [2.0, 0.0, 0.0]}, w, [beta],
                 options={**HO, "domain": "x_wall"}).solve()  # fmt: skip
    pair = [{"name": "a", "mesh": spline_cylinder(), "position": [2.0, 0.0, 0.0]},
            {"name": "b", "mesh": spline_cylinder(), "position": [-2.0, 0.0, 0.0]}]  # fmt: skip
    pair = model(pair, w, [beta, 180.0 - beta], options=HO).solve()
    for q in ("added_mass", "radiation_damping"):
        a, b = view(wall, q), view(pair, q)
        for i, j, sgn in ((2, 2, 1.0), (4, 4, -1.0), (0, 0, -1.0)):
            ref = b[:, i, j] + sgn * b[:, i, 6 + j]
            assert peak(a[:, i, j] - ref) <= 1e-5 * peak(ref), (q, i)
    ref = pair.excitation.diffraction[:, 0, :6] + pair.excitation.diffraction[:, 1, :6]
    assert peak(wall.excitation.diffraction[:, 0] - ref) <= 1e-5 * peak(ref)


def test_symmetry_of_a_system_equals_the_full_solve():
    # two bodies on y = 0, one yawed by 90 deg: the y reflection of the system against the same
    # problem on meshes without symmetry flags, exact for the discrete equations (theory §6.3);
    # 1e-5: the float32 matrices and the character sums in another order (9e-10 measured)
    def bodies(sym):
        quarter = box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True)
        full = []
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            for q in quarter:
                r = q * [sx, sy, 1]
                full.append(r[[0, 3, 2, 1]] if sx * sy < 0 else r)
        disc = (3, 3, 3, 3, 3, 4)  # square: the reflected copies keep it
        box = flat(quarter, disc, symmetry="xz_and_yz") if sym else flat(np.array(full), disc)
        return [{"name": "c", "mesh": spline_cylinder() if sym else full_cylinder(), "position": [2.0, 0.0, 0.0]},
                {"name": "b", "mesh": box, "position": [-1.5, 0.0, 0.0], "yaw": 90.0}]  # fmt: skip

    w = np.sqrt(np.array([0.8, 1.6]) * G)
    a, b = (model(bodies(sym), w, [20.0], options=HO).solve() for sym in (True, False))
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        x, y = view(a, q), view(b, q)
        assert peak(x - y) <= 1e-5 * peak(y), q


def test_unavailable_higher_order_routes_are_rejected():
    # input rules: no omega = inf with pressure patches, no generated interior surface
    m = spline_cylinder()
    for problem, what in (
        (model(ho_acv(), [1.0, math.inf], [0.0], options=HO), "omega = inf"),
        (model({"mesh": m, "interior_surface": {"method": "waterline"}}, [1.0], [0.0],
               options={"higher_order": True}), "interior"),
    ):  # fmt: skip
        with pytest.raises(hb.UnsupportedProblemError, match=what):
            problem.route()


def test_thin_sheet_patches():
    # flat dipole patches, the hypersingular rows in Galerkin form (the direct Rankine term by its
    # finite part and the edge integral of 1/r plus the integrable remainder). Identities of the
    # problem: the normal direction of a sheet does not matter and sectors equal the full solve,
    # exact for the discrete equations, 1e-6: float32 round-off (1.1e-14 measured); the Haskind force
    # equals the pressure force, 5e-3: the pressure jump over the sheet, singular at its edges, by
    # the patch quadrature (6.2e-4 measured); the scattered and the total formulation, 2e-3: two
    # discretized unknowns (1.8e-5 measured)
    w = 2 * np.pi / np.array([2.0, 5.0])
    opts = {"higher_order": True, "diffraction_formulation": "total"}

    def run(body, **kw):
        o = {**opts, **kw}
        return model(body, w, [0.0, 30.0] if o.get("diffraction", True) else [], options=o).solve()

    a = run(plate_column())
    for other in (run(plate_column(flip=True)), run(plate_column(quarter=False))):
        for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
            x, y = view(other, q), view(a, q)
            assert peak(x - y) <= 1e-6 * peak(y), q
    hk = a.excitation.haskind
    for i in range(6):
        x = a.excitation.diffraction[:, 0, i]
        assert peak(hk[:, 0, i] - x) <= 5e-3 * peak(a.excitation.diffraction[:, 0]), i
    s = run(plate_column(), diffraction_formulation="scattered")  # also the scattered potential
    assert peak(s.excitation.diffraction - a.excitation.diffraction) <= 2e-3 * peak(a.excitation.diffraction)


def test_thin_sheet_patch_rules():
    # input rules: no pressure drift with thin sheets; the low order keeps the scattered formulation
    body = plate_column()
    def check(problem, what):
        with pytest.raises(hb.UnsupportedProblemError, match=what):
            problem.route()
    o = {"higher_order": True, "diffraction_formulation": "total"}
    check(model({**body, "modes": []}, [1.0], [0.0], options={**o, "drift_pressure_integration": True}),
          "drift with thin sheets")  # fmt: skip
    # the low order still rejects the total formulation with sheets (a plate under a cylinder)
    hull = cylinder_panels(0.5, 0.5, n_phi=4, n_z=2, n_r=2, quarter=True)
    pan = np.array(face_panels((0, 0, -1), (1, 0, 0), (0, 1, 0), 2, 2, (0, 0, 1)))
    mesh = surface(np.concatenate([hull, pan]), symmetry="xz_and_yz")
    lo = {"mesh": part(mesh, 1, len(hull)), "dipoles": [{"mesh": part(mesh, len(hull) + 1, len(hull) + len(pan))}]}
    with pytest.raises(ValueError, match="low-order"):
        model(lo, [1.0], [0.0], options={"diffraction_formulation": "total"}).route()


def test_closed_sheet_side_is_split_by_its_length():
    # the edge integral of a planar sheet patch splits each side by its length; a side that closes on
    # itself (the outer circle of a full annulus in one patch) has a chord of 0 and was left in one
    # piece (an O(1) error). The same annulus as one patch and as four quarter patches: the same heave
    # added mass to their discretization difference, 1e-4 (1.6e-5 measured); the axisymmetric annulus
    # has no heave-pitch or heave-roll coupling (an identity), 1e-4 of the heave added mass: the
    # spline space of one patch is not symmetric about its seam (1.3e-5 measured)
    def run(splits, nu):
        ts = np.linspace(0.0, 2 * np.pi, splits + 1)
        patches = [{"type": "AnalyticPatch", "map": "line", "profile": "linear", "a": 1.0, "b": 1.0,
                    "theta0": ts[i], "theta1": ts[i + 1], "p": [0.5, -0.3, 1.0, -0.3]} for i in range(splits)]
        mesh = {"kind": "analytic_patches", "discretization": np.array([[nu, 2, 3, 3, 3, 4]] * splits),
                "symmetry": "none", "patches": patches}
        body = {"dipoles": [{"mesh": mesh}]}
        return model(body, [1.5], (), options={"higher_order": True, "diffraction": False}).solve().added_mass[0]

    one, four = run(1, 8), run(4, 2)
    assert abs(one[2, 2] - four[2, 2]) <= 1e-4 * four[2, 2], (one[2, 2], four[2, 2])
    assert abs(one[2, 4]) <= 1e-4 * one[2, 2] and abs(one[2, 3]) <= 1e-4 * one[2, 2]


def test_sheet_edge_integral_near_a_side_between_its_samples():
    # the edge integral of a planar sheet patch splits each side by the distance of the target from
    # each piece; a split by the smallest of 17 samples of the whole side took a Gauss point close to
    # the side between two samples to be far from it (pieces far longer than its distance, the peak
    # of the kernel missed: 0.6 of the term). A circular disc of radius 1 at depth 10, heave, its
    # radial spans graded towards the free edge ("quadratic_end", Gauss points next to the radial
    # sides near the centre) or towards both ends ("smooth"): the mean of A(0) and A(inf) against the
    # disc in unbounded fluid, (8/3) rho a^3 (Lamb 1932, §108), 3e-3: four radial spans and the free
    # surface ten radii above (1.7e-3 and 2.2e-3 measured; 1.1e-2 for "quadratic_end" with the
    # sampled split)
    def a33(profile):
        patch = {"type": "AnalyticPatch", "map": "line", "profile": profile, "a": 1.0, "b": 1.0,
                 "theta0": 0.0, "theta1": 0.5 * np.pi, "p": [0.0, -10.0, 1.0, -10.0]}  # fmt: skip
        mesh = {"kind": "analytic_patches", "discretization": np.array([[7, 4, 3, 3, 3, 4]]),
                "symmetry": "xz_and_yz", "patches": [patch]}  # fmt: skip
        body = {"modes": ["heave"], "dipoles": [{"mesh": mesh}]}
        r = model(body, [0.0, math.inf], (), options={"higher_order": True, "diffraction": False}).solve()
        return 0.5 * (r.added_mass_zero_frequency[2, 0] + r.added_mass_infinite_frequency[2, 0])  # heave force, heave

    lamb = 8.0 / 3.0 * RHO
    for profile in ("quadratic_end", "smooth"):
        assert abs(a33(profile) / lamb - 1.0) <= 3e-3, profile


def test_sheet_rows_converge_next_to_a_free_edge_of_zero_profile_slope():
    # a Gauss point next to a side where the profile has zero slope lies at a distance from it that
    # goes as the square of its parameter (1e-4 of the patch at 12 spans): the Duffy rectangle of
    # its own cell towards that side is flat and graded in q, and the cells of the neighbouring
    # spans are subdivided until they are 2.5 of their sizes away from it. The disc of radius 1 at
    # depth 10 with its radial spans graded towards the free edge ("quadratic_end", the jump
    # sqrt(1 - r^2) smooth in the parameter): the mean of A(0) and A(inf) against (8/3) rho a^3
    # (Lamb 1932, §108) converges with the spans, 1e-4 at four and 3e-5 at six (5.4e-5 and 1.1e-5
    # measured; one 10 x 10 rule per triangle and at most 7 levels left 1.7e-3 at both, rising
    # with refinement)
    def error(n):
        patch = {"type": "AnalyticPatch", "map": "line", "profile": "quadratic_end", "a": 1.0, "b": 1.0,
                 "theta0": 0.0, "theta1": 0.5 * np.pi, "p": [0.0, -10.0, 1.0, -10.0]}  # fmt: skip
        nu = math.ceil(0.5 * np.pi * n)  # spans around about as long as the radial ones
        mesh = {"kind": "analytic_patches", "discretization": np.array([[nu, n, 3, 3, 3, 4]]),
                "symmetry": "xz_and_yz", "patches": [patch]}  # fmt: skip
        body = {"modes": ["heave"], "dipoles": [{"mesh": mesh}]}
        r = model(body, [0.0, math.inf], (), options={"higher_order": True, "diffraction": False}).solve()
        a = 0.5 * (r.added_mass_zero_frequency[2, 0] + r.added_mass_infinite_frequency[2, 0])  # heave force, heave
        return abs(a / (8.0 / 3.0 * RHO) - 1.0)

    e4, e6 = error(4), error(6)
    assert e4 <= 1e-4 and e6 <= 3e-5 and e6 < e4, (e4, e6)


def test_graded_duffy_rule_keeps_the_axisymmetry_next_to_the_pole():
    # the bottom patch of the B-spline cylinder narrows to its pole, so its panels next to
    # the axis are elongated (azimuthal length -> 0). In heave the flow is axisymmetric: at the
    # quadrature points next to the pole the azimuthal velocity is zero and the radial one the
    # same at every azimuth (an identity of the problem). With one 8 x 8 Duffy rule per triangle the
    # azimuthal component was 0.2 of the radial one; graded, both within 1e-3 of the mean radial
    # velocity: the non-rational spline is not quite a circle (4.9e-4 and 6.4e-5 measured).
    w = np.sqrt(np.array([0.6]) * 9.80665)
    opts = {"higher_order": True, "body_velocity": True, "diffraction": False}
    r = model({"mesh": spline_cylinder(n=4)}, w, (), options=opts).solve()
    pts = np.asarray(r.body_points.points)
    v = np.asarray(r.body_points.velocity_radiation)[0, :, 2, :2]  # heave, horizontal components
    rad = np.hypot(pts[:, 0], pts[:, 1])
    bottom = np.abs(pts[:, 2] + 0.5) < 1e-9
    near = bottom & (rad < 1.5 * rad[bottom].min())
    er = pts[near, :2] / rad[near, None]
    vr = np.einsum("qd,qd->q", v[near], er)
    vt = np.einsum("qd,qd->q", v[near], np.stack([-er[:, 1], er[:, 0]], axis=1))
    assert near.sum() >= 8
    assert np.abs(vt).max() <= 1e-3 * np.abs(vr).mean()
    assert np.ptp(np.abs(vr)) <= 1e-3 * np.abs(vr).mean()


@pytest.mark.parametrize("threads", [(2, 1), (1, 3), (2, 2)])
def test_threads_byte_identical(threads):
    # the higher order as the low order: the results do not depend on the thread counts (the
    # deterministic order of the sums, theory §13.3): bit-identical
    body = {"mesh": spline_cylinder(n=4), "mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.2],
            "radii_of_gyration": [0.6, 0.6, 0.7]}  # fmt: skip
    w = [1.0, 2.0, 3.0]
    opts = {"higher_order": True, "motion": True, "field_pressure": True}
    kw = dict(field_points=[[3.0, 0.0, 0.0], [0.0, 2.5, -0.3]])
    r1 = model(body, w, [0.0, 45.0], options=opts, **kw).solve()
    rn = model(body, w, [0.0, 45.0], options={**opts, "frequency_threads": threads[0], "inner_threads": threads[1]},
               **kw).solve()  # fmt: skip
    for q in ("added_mass", "radiation_damping", "excitation_diffraction", "motion",
              "field_points_pressure_diffraction", "field_points_pressure_radiation"):
        assert r1.quantity(q).tobytes() == rn.quantity(q).tobytes(), q
