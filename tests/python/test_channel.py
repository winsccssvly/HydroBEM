"""Bodies in a channel of finite depth, low and higher order (theory §1.2, test-plan B).

Each check on the basis it rests on. Identities of the problem: the field satisfies the wall
condition dphi/dy = 0 on y = +-W/2; a body and its mirror image about the centreline give mirrored
results; symmetry sectors equal the full solve; no reflection (channel_reflect = 0) is the open sea;
the Haskind and the pressure exciting force agree and A, B are symmetric as for the same body in
open water; the limits with and without an interior surface; the control-surface drift does not
depend on the surface. Asymptotics: the omega = inf images decay like exp(-pi R / 2h); A(omega) -
A(0) falls like omega^2. Input rules: routes that are not available are rejected. Damping lids,
thin sheets and dampers (both methods): no reflection is the open sea, the mirror body, a lid or a
damper without damping is none, a rigid damper is the sheet. The comparisons of discretizations
(the wall effect of the higher order against the low order, the slabs that tend to the sheet, the
widening channel) are validation, run when the method changes, not tests. The kernel itself is
tested in crates/hydrobem/tests (Faddeeva function, lattice sum against mpmath, tail tiles). Meshes
are procedural."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

import hydrobem as hb

from bodies import (
    flat,
    full_cylinder,
    graded_cylinder,
    ho_acv,
    ho_submerged_plate,
    lo_free_cylinder,
    moonpool_barge,
    part,
    spline_cylinder,
    submerged_plate,
    surface,
)
from meshes import box_panels, cylinder_panels, face_panels, moonpool_lid_panels, moonpool_panels
from problems import RHO, analytic, model, peak, view

G = 9.80665
H, W = 20.0, 40.0
CYL = surface(cylinder_panels(2.0, 3.0, n_phi=6, n_z=3, n_r=2, quarter=True), symmetry="xz_and_yz")


def channel(**kw):
    return {"domain": "channel", "channel_width": W, **kw}


def env():
    return {"depth": H}


def test_wall_condition():
    # the wall images make dphi/dy = 0 on the walls exactly: the slope at the wall of the quadratic
    # fit in the distance d, of k times the peak. 1e-4: the truncated far-image sums (the tabulated
    # tail, converged to about 1e-6 of the kernel), not the fit: 6.1e-5 found, the same with the
    # points at half and at twice the distances
    w = np.sqrt(np.array([0.05, 0.2]) * G)
    d = np.array([0.0625, 0.125, 0.1875, 0.25])
    pts = [(x0, sgn * (W / 2 - di), 0.0) for x0 in (0.0, 15.0, 60.0) for sgn in (1, -1) for di in d]
    r = model({"mesh": CYL, "position": [0.0, 5.0, 0.0]}, w, [0.0], field_points=np.array(pts), environment=env(),
              options=channel(field_pressure=True)).solve()  # fmt: skip
    for f in range(len(w)):
        k = brentq(lambda kk, K=w[f] ** 2 / G: kk * np.tanh(kk * H) - K, 1e-9, 10.0)
        fr, fd = r.field_points.pressure_radiation, r.field_points.pressure_diffraction
        for v in (fr[f, :, 0], fr[f, :, 2], fd[f, :, 0]):
            for b in range(6):
                c = np.linalg.lstsq(np.c_[np.ones(4), d, d**2], v[4 * b : 4 * b + 4], rcond=None)[0]
                assert abs(c[1]) <= 1e-4 * k * peak(v), (f, b)


def test_mirror_body_gives_identical_results():
    w = np.sqrt(np.array([0.05, 0.12]) * G)
    res = [model({"mesh": CYL, "position": [0.0, y, 0.0]}, w, [0.0, 180.0], environment=env(),
                 options=channel()).solve() for y in (5.0, -5.0)]  # fmt: skip
    # mirror about y = 0: sway, roll and yaw change sign (an identity of the problem). 1e-5: the
    # float32 influence matrices and LU, the image sums in another order (6.1e-7 found)
    s = np.array([1, -1, 1, -1, 1, -1])
    for q in ("added_mass", "radiation_damping"):
        a, b = view(res[0], q), view(res[1], q) * s[None, :, None] * s[None, None, :]
        assert peak(a - b) <= 1e-5 * peak(a), q
    x0, x1 = res[0].excitation.diffraction, res[1].excitation.diffraction
    assert peak(x0 - x1 * s) <= 1e-5 * peak(x0)


def test_symmetry_sectors_equal_full_solve():
    # the sectors solve the same equations; 1e-5: float32 matrices and LU of other sizes (1.3e-6
    # found)
    quarter = box_panels(4.0, 2.0, 2.0, n=(4, 2, 2), quarter=True)
    full = box_panels(4.0, 2.0, 2.0, n=(8, 4, 2), quarter=False)
    w = np.sqrt(np.array([0.06, 0.15]) * G)
    out = [model({"mesh": surface(pan, symmetry=sym)}, w, [0.0], environment=env(), options=channel()).solve()
           for pan, sym in ((quarter, "xz_and_yz"), (full, "none"))]  # fmt: skip
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        a, b = view(out[0], q), view(out[1], q)
        assert peak(a - b) <= 1e-5 * peak(b), q


def test_haskind_and_reciprocity():
    # the Haskind relation and the symmetry of A and B are identities of the problem (the incident
    # wave with its wall reflections in the Haskind relation); they hold to the discretization of
    # the coarse sharp-edged body, which the channel does not change: 1.5 times the level of the
    # same body in open water plus 1e-4 (the Haskind force 1.9e-2 in both, the asymmetry of A 4.3e-3
    # against 3.4e-3); B >= 0 (energy), -1e-6: round-off
    w = np.sqrt(np.array([0.05, 0.12, 0.3]) * G)
    err = []
    for opts, y in ((channel(diffraction_formulation="total"), 6.0),
                    ({"diffraction_formulation": "total"}, 0.0)):  # fmt: skip
        r = model({"mesh": CYL, "position": [0.0, y, 0.0]}, w, [0.0], environment=env(), options=opts).solve()
        hk = r.excitation.haskind  # [F, H, K]
        A, B = r.added_mass, r.radiation_damping
        X = r.excitation.diffraction
        err.append([peak(hk - X) / peak(X), peak(A - A.transpose(0, 2, 1)) / peak(A),
                    peak(B - B.transpose(0, 2, 1)) / peak(B)])  # fmt: skip
        assert np.all(np.diagonal(B, axis1=1, axis2=2) >= -1e-6 * peak(B))
    err = np.array(err)
    assert np.all(err[0] <= 1.5 * err[1] + 1e-4), err


def test_infinite_frequency_limit_in_a_wide_channel_is_open_water():
    # at omega = inf the images decay like exp(-pi R / 2h): a channel of width 400 m in 20 m depth
    # is open water to round-off; 1e-5: float32 matrices (3e-17 found, the same sums)
    w = np.array([math.inf])
    opts = [{"domain": "channel", "channel_width": 400.0, "diffraction": False}, {"diffraction": False}]
    a = [model({"mesh": CYL}, w, (), environment=env(), options=o).solve().added_mass_infinite_frequency
         for o in opts]  # fmt: skip
    assert peak(a[0] - a[1]) <= 1e-5 * peak(a[1])


def test_unavailable_channel_routes_are_rejected():
    def check(m, what):
        with pytest.raises(hb.UnsupportedProblemError, match=what):
            m.route()

    narrow = {"domain": "channel", "channel_width": 5.0}
    small = surface(cylinder_panels(1.0, 1.0, n_phi=4, n_z=2, n_r=2, quarter=True), symmetry="xz_and_yz")
    check(model({"mesh": small}, [1.0], [0.0], environment=env(), options=narrow), "narrower")


# ------------------------------------------------------------- interior surfaces

# the cylinder a = d = 1 off the centreline of a channel 6 m wide in 3 m depth; the interior
# problem is that of the open sea, its first irregular frequency at K a = j01 coth(j01 d / a)
# = 2.4446, between the channel cut-offs k W = 4 pi and 5 pi (k = 2.09 and 2.62)
IRR_CHANNEL = {"domain": "channel", "channel_width": 6.0}


def _cylinder(n, irr=None, **kw):
    b = {"mesh": graded_cylinder(n), "position": [0.0, 0.5, 0.0], **kw}
    return {**b, "interior_surface": irr} if irr else b


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_interior_surface_in_a_channel_removes_the_irregular_frequency():
    ka = np.array([0.8, 1.3, 2.40, 2.45, 2.50])
    w = np.sqrt(ka * G)
    plain, irr = (model(_cylinder(16, s), w, [0.0], environment={"depth": 3.0}, options=IRR_CHANNEL).solve()
                  for s in (None, "waterline"))  # fmt: skip
    # away from the irregular frequency the results agree (test-plan B: 1e-3; a discretization
    # error, O(h^2): 1.2e-3, 4.3e-4 at 12, 20 panels per quarter, as in the open sea)
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        a, b = view(plain, q)[:2], view(irr, q)[:2]
        assert peak(a - b) <= 1e-3 * peak(a), q
    # near it the plain heave added mass is polluted (1.3e-2 of the peak), the extended one smooth
    A33, P33 = irr.added_mass[:, 2, 2], plain.added_mass[:, 2, 2]
    assert abs(P33[3] - A33[3]) > 5e-3 * peak(A33)
    assert abs(A33[3] - 0.5 * (A33[2] + A33[4])) <= 1e-3 * peak(A33)


def test_wall_condition_with_an_interior_surface():
    # the source formulation with the interior sources: v_y extrapolated to the wall y = W/2
    # vanishes (the wall images); 2e-4 of the peak: the truncated far-image sums as without the
    # interior surface (5.4e-5 found; 1e-13 for the modes even in y, whose images cancel exactly)
    d = np.array([0.02, 0.04, 0.06, 0.08])
    pts = np.array([[0.0, 3.0 - x, -0.5] for x in d])
    r = model(_cylinder(12, "waterline"), np.sqrt(np.array([1.3]) * G), [0.0], environment={"depth": 3.0},
              field_points=pts, options={**IRR_CHANNEL, "field_velocity": True}).solve()  # fmt: skip
    vr, vd = r.field_points.velocity_radiation[0], r.field_points.velocity_diffraction[0]
    for v in [vr[:, k] for k in range(6)] + [vd[:, 0]]:
        assert abs(np.polyfit(d, v[:, 1], 1)[1]) <= 2e-4 * peak(v)


def test_limits_with_an_interior_surface_in_a_channel():
    # the limits have no irregular frequencies and are solved without the interior unknowns
    # (identity rows): the same equations; 1e-6: float32 LU of the larger system (3e-8 found).
    # Modes without net flux (rule 9)
    w = [0.0, math.inf]
    opts = {**IRR_CHANNEL, "diffraction": False}
    plain, irr = (model(_cylinder(12, s, modes=["surge", "pitch"]), w, (), environment={"depth": 3.0},
                        options=opts).solve() for s in (None, "waterline"))  # fmt: skip
    for key in ("zero", "infinite"):
        a, b = plain.quantity("added_mass_" + key + "_frequency"), irr.quantity("added_mass_" + key + "_frequency")
        assert peak(a - b) <= 1e-6 * peak(a), key


BOX = box_panels(1.0, 0.5, 0.6, n=(8, 4, 4), quarter=True)
ZERO_FLUX = ("surge", "sway", "roll", "pitch", "yaw")  # no net flux: symmetric waterplane (rule 9)
IDX = [0, 1, 3, 4, 5]


def _zero(width, omegas, mesh=None):
    opts = {"domain": "channel", "channel_width": width, "diffraction": False} if width else {"diffraction": False}
    mesh = mesh or surface(BOX, symmetry="xz_and_yz")
    return model({"mesh": mesh, "modes": list(ZERO_FLUX)}, omegas, (), environment={"depth": 3.0},
                 options=opts).solve()  # fmt: skip


def test_zero_frequency_is_the_low_frequency_limit():
    # the low-frequency expansion of the finite-depth problem: A(omega) - A(0) and B fall like
    # omega^2 for these modes (Abar = Abar(0)), so doubling omega multiplies the difference by 4
    # (3.5 to 4.5: the omega^4 term); at omega = 0.05 both within 2e-4 and 1e-4 of the peak of A(0),
    # the order of K h = 7.7e-4
    r = _zero(8.0, [0.0, 0.05, 0.1])
    A0 = r.added_mass_zero_frequency[IDX]
    d = [peak(r.added_mass[f][IDX] - A0) / peak(A0) for f in range(2)]
    assert d[0] <= 2e-4 and 3.5 <= d[1] / d[0] <= 4.5, d
    assert peak(r.radiation_damping[0][IDX]) <= 1e-4 * peak(A0)


def test_zero_frequency_sectors_equal_full_solve():
    # the same equations; 1e-5: float32 matrices and LU of other sizes (2.1e-7 found)
    full = surface(box_panels(1.0, 0.5, 0.6, n=(16, 8, 4), quarter=False))
    a, b = _zero(6.0, [0.0]), _zero(6.0, [0.0], mesh=full)
    za, zb = a.added_mass_zero_frequency, b.added_mass_zero_frequency
    assert peak(za - zb) <= 1e-5 * peak(zb)


def test_channel_reflect_zero_is_the_open_sea_and_one_the_channel():
    # channel_reflect scales the wall images: 0 leaves the body alone in the open sea of the same
    # depth (also with its interior surface: the channel rows, free term -4 pi on the interior
    # surface, equal the open ones), 1 is the default; 1e-5: float32 matrices, the channel rows
    # summed with the zero images in another order (0 found for the box, 3.2e-7 with the interior
    # surface). 0.5 differs from both (3 % to 22 % of the peaks here); the factor is checked
    body = {"mesh": surface(box_panels(1.0, 0.5, 0.5, n=(4, 2, 2), quarter=True), symmetry="xz_and_yz")}
    w = np.array([1.2, 2.5])
    env = {"depth": 3.0, "density": RHO, "gravity": G}

    def run(**kw):
        return model(body, w, [0.0], environment=env, options=kw).solve()

    sea = run()
    ch = {r: run(domain="channel", channel_width=6.0, channel_reflect=r) for r in (0.0, 0.5, 1.0)}
    default = run(domain="channel", channel_width=6.0)
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        assert peak(view(ch[0.0], q) - view(sea, q)) <= 1e-5 * peak(view(sea, q)), q
        np.testing.assert_array_equal(view(ch[1.0], q), view(default, q))
        assert peak(view(ch[0.5], q) - view(sea, q)) > 1e-2 * peak(view(sea, q))
    wi = np.sqrt(np.array([1.3, 2.4]) * G)
    a, b = (model(_cylinder(12, "waterline"), wi, [0.0], environment={"depth": 3.0}, options=o).solve()
            for o in ({**IRR_CHANNEL, "channel_reflect": 0.0}, {}))  # fmt: skip
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        assert peak(view(a, q) - view(b, q)) <= 1e-5 * peak(view(b, q)), q
    # at the channel resonance W = lambda / 2 (k W = pi) the lattice sum of the wall images diverges:
    # a channel with reflecting walls is refused there, while without reflection no wall image is
    # summed and the open sea is solved (the same 1e-5 as above)
    k = math.pi / 6.0
    wr = np.array([math.sqrt(G * k * math.tanh(3.0 * k))])
    with pytest.raises(np.linalg.LinAlgError, match="half wavelength"):
        model(body, wr, [0.0], environment=env, options={"domain": "channel", "channel_width": 6.0}).solve()
    for o in (HO, {}):  # higher order (the spline cylinder) and low order (the box)
        opts = {**o, "domain": "channel", "channel_width": 6.0, "channel_reflect": 0.0}
        mesh = {"mesh": spline_cylinder()} if o else body
        r0, rs = (model(mesh, wr, [0.0], environment=env, options=q).solve() for q in (opts, o))
        for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
            assert peak(view(r0, q) - view(rs, q)) <= 1e-5 * peak(view(rs, q)), q
    m = model(body, w, [0.0], environment=env)
    with pytest.raises(ValueError, match="channel_reflect"):
        m.environment.channel_reflect = 1.5
    m.environment.channel_reflect = 0.5
    with pytest.raises(ValueError, match="only allowed with domain 'channel'"):
        m.counts()


# ------------------------------------------------------------- higher order

# the spline cylinder a = 1, d = 0.5 in a channel 6 m wide in 3 m depth
HO = {"higher_order": True, "diffraction_formulation": "total"}
HO_CHANNEL = {**HO, "domain": "channel", "channel_width": 6.0}
HO_W = np.sqrt(np.array([0.5, 1.2, 2.0]) * G)
QS = ("added_mass", "radiation_damping", "excitation_diffraction")


def _ho(mesh, y, opts, w=HO_W, headings=(0.0,), **kw):
    body = {"mesh": mesh, "position": [0.0, y, 0.0], **kw}
    return model(body, w, list(headings), environment={"depth": 3.0}, options=opts).solve()


def test_higher_order_channel_without_reflection_is_the_open_sea():
    # channel_reflect = 0: the images (instances and the far sums) vanish; the global frame of the
    # channel against the body frame of the open sea. 1e-10: the same equations in double precision
    sea, ch = (_ho(spline_cylinder(), 0.5, o) for o in (HO, {**HO_CHANNEL, "channel_reflect": 0.0}))
    for q in QS:
        assert peak(view(ch, q) - view(sea, q)) <= 1e-10 * peak(view(sea, q)), q


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_higher_order_pressure_surfaces_in_a_channel():
    # the air-cushion vehicle (its coarser mesh: the identities do not need the finer, at a sixth of
    # its cost) in a channel 60 m wide in 20 m depth: without reflection the open sea (1e-10: the
    # same equations in double precision)
    w = 2 * np.pi / np.array([4.0, 6.0])
    acv = {**ho_acv(size=1.5), "position": [0.0, 3.0, 0.0]}
    sea, r0, ch = (model(acv, w, [0.0], environment={"depth": 20.0}, options=o).solve()
                   for o in (HO, {**HO, "domain": "channel", "channel_width": 60.0, "channel_reflect": 0.0},
                             {**HO, "domain": "channel", "channel_width": 60.0}))  # fmt: skip
    for q in QS:
        assert peak(view(r0, q) - view(sea, q)) <= 1e-10 * peak(view(sea, q)), q
    # the Haskind relation, an identity, to the discretization of the vessel, which the channel does
    # not change: 1.5 times its level in open water plus 1e-5 (1.2e-4, 1.2e-4)
    err = [peak(r.excitation.haskind - r.excitation.diffraction) / peak(r.excitation.diffraction) for r in (ch, sea)]
    assert err[0] <= 1.5 * err[1] + 1e-5, err


def test_higher_order_mirror_body_and_sectors():
    # identities of the problem: the mirror body (sway, roll and yaw change sign), the sectors on the
    # centreline (both reflections) and off it (x only) against the mesh without symmetry flags. The
    # higher order solves in double precision and stores the coefficients in float32: 1e-6, the
    # float32 rounding of the stored values (1e-16 found for the mirror, 7.5e-12 for the sectors)
    s = np.array([1, -1, 1, -1, 1, -1])
    a, b = (_ho(spline_cylinder(), y, HO_CHANNEL, headings=(0.0, 180.0)) for y in (0.5, -0.5))
    for q in ("added_mass", "radiation_damping"):
        assert peak(view(a, q) - view(b, q) * s[None, :, None] * s[None, None, :]) <= 1e-6 * peak(view(a, q))
    assert peak(a.excitation.diffraction - b.excitation.diffraction * s) <= 1e-6 * peak(a.excitation.diffraction)
    for y in (0.0, 0.7):
        a, b = (_ho(m, y, HO_CHANNEL, w=HO_W[1:]) for m in (spline_cylinder(), full_cylinder()))
        for q in QS:
            assert peak(view(a, q) - view(b, q)) <= 1e-6 * peak(view(b, q)), (y, q)


def test_higher_order_wall_condition():
    # the wall images: v_y (by differences) and dp/dy extrapolated to the wall y = W/2 vanish; 2e-4
    # of the peak: the truncated far-image sums as in the low order (5.1e-5 and 3.9e-5 found)
    d = np.array([0.02, 0.04, 0.06, 0.08])
    pts = np.array([[x0, 3.0 - di, z] for x0 in (0.0, 4.0) for z in (-0.3, 0.0) for di in d])
    body = {"mesh": spline_cylinder(), "position": [0.0, 0.5, 0.0]}
    r = model(body, HO_W[:2], [0.0], environment={"depth": 3.0}, field_points=pts,
              options={**HO_CHANNEL, "field_pressure": True, "field_velocity": True}).solve()  # fmt: skip
    fp = r.field_points
    for f in range(2):
        pp, pv = peak(fp.pressure_radiation[f]), peak(fp.velocity_radiation[f])
        for b in range(4):
            sl = slice(4 * b, 4 * b + 4)
            for k in range(7):
                p = fp.pressure_radiation[f, sl, k] if k < 6 else fp.pressure_diffraction[f, sl, 0]
                v = fp.velocity_radiation[f, sl, k, 1] if k < 6 else fp.velocity_diffraction[f, sl, 0, 1]
                assert abs(np.polyfit(d, p, 2)[1]) <= 2e-4 * pp, (f, b, k)
                assert abs(np.polyfit(d, v, 1)[1]) <= 2e-4 * pv, (f, b, k)


def test_higher_order_reciprocity_haskind_and_formulations():
    # identities of the problem, each to the discretization of the higher order: the Haskind
    # relation as in open water (1.5 times its level there plus 1e-4: 3.3e-4 against 3.7e-4); A and
    # B symmetric within the bound of the higher order, 2e-3 (the sway-heave coupling of the walls,
    # O(h^2): 4.8e-4, 2.2e-4, 1.3e-4 of the peak of B at 4, 6, 8 spans; 1.0e-4 in open water, which
    # has no such coupling); B >= 0 (energy; -1e-6: round-off); the scattered and the total
    # diffraction potential, two formulations of one problem that discretize different unknowns,
    # within 1e-3 (1.4e-4 found, 1.5e-4 in open water)
    err = []
    t, sea = (_ho(spline_cylinder(), 0.5, opts) for opts in (HO_CHANNEL, HO))
    for r in (t, sea):
        A, B = r.added_mass, r.radiation_damping
        X = r.excitation.diffraction
        err.append([peak(r.excitation.haskind - X) / peak(X), peak(A - A.transpose(0, 2, 1)) / peak(A),
                    peak(B - B.transpose(0, 2, 1)) / peak(B)])  # fmt: skip
        assert np.all(np.diagonal(B, axis1=1, axis2=2) >= -1e-6 * peak(B))
    err = np.array(err)
    assert err[0, 0] <= 1.5 * err[1, 0] + 1e-4 and np.all(err[0, 1:] <= 2e-3), err
    s = _ho(spline_cylinder(), 0.5, {**HO_CHANNEL, "diffraction_formulation": "scattered"})
    assert peak(s.excitation.diffraction - t.excitation.diffraction) <= 1e-3 * peak(t.excitation.diffraction)


def test_higher_order_limits_in_a_channel():
    # omega = inf: the images decay like exp(-pi R / 2h), a channel 400 m wide is open water to
    # round-off (1e-6: the float32 storage; 1.4e-14 found); omega = 0 is the limit of the low
    # frequencies (A - A(0) like omega^2, the factor 4 for twice omega, 3.5 to 4.5: the omega^4
    # term) for the modes without net flux (rule 9)
    modes = ["surge", "sway", "roll", "pitch", "yaw"]
    opts = {**HO_CHANNEL, "diffraction": False}
    a, b = (_ho(spline_cylinder(), 0.5, o, w=[math.inf], headings=(), modes=modes).added_mass_infinite_frequency
            for o in ({**opts, "channel_width": 400.0}, {**HO, "diffraction": False}))  # fmt: skip
    assert peak(a - b) <= 1e-6 * peak(b)
    r = _ho(spline_cylinder(), 0.5, opts, w=[0.0, 0.05, 0.1], headings=(), modes=modes)
    A0 = r.added_mass_zero_frequency
    d = [peak(r.added_mass[f] - A0) / peak(A0) for f in range(2)]
    assert d[0] <= 3e-4 and 3.5 <= d[1] / d[0] <= 4.5, d


def _ho_box(n=(4, 2)):
    """The box 2 x 1 x 0.5 m (higher order, flat patches) and its interior patch: the mesh with
    the interior patch as its element 4, and the box without it."""
    q = box_panels(1.0, 0.5, 0.5, n=(1, 1, 1), quarter=True, lid=True)
    disc = [(*nv, 3, 3, 3, 4) for nv in (n, (2, 2), n, n)]
    return flat(q, disc, symmetry="xz_and_yz"), flat(q[:3], disc[:3], symmetry="xz_and_yz")


def test_higher_order_interior_patch_in_a_channel():
    # the box with its interior patch at y = 0.6 in a channel 5.5 m wide, away from its first
    # symmetric irregular frequency (K = 3.729, between the cut-offs k W = 6 pi and 7 pi): the
    # interior patch leaves the solution of the continuous problem; 1e-3, the discretization of the
    # interior patch (2.2e-4 found). The removal of the irregular frequency is validation
    box, plain_box = _ho_box()
    w = np.sqrt(np.array([2.0]) * G)
    opts = {**HO, "domain": "channel", "channel_width": 5.5}
    plain = _ho(plain_box, 0.6, opts, w=w)
    irr = _ho(part(box, 1, 3), 0.6, opts, w=w, interior_surface=part(box, 4, 4))
    for name in QS:
        assert peak(view(plain, name) - view(irr, name)) <= 1e-3 * peak(view(plain, name)), name


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_higher_order_control_surface_drift_in_a_channel():
    # the momentum flux through a closed surface around the body, an identity of momentum
    # conservation: the same for the cylinders of radius 1.5 and 2 m (depths 1 and 1.5 m) in a channel
    # 8 m wide; 1e-5: the quadrature of the two surfaces (4.9e-6 of the peak found; the cost is the
    # channel Green function at the surfaces' points, so the surfaces are kept small)
    def drift(radius, depth):
        cs = {"method": "cylinder", "radius": radius, "depth": depth, "inner_radius": 1.0, "panel_size": 0.5}
        b = {**lo_free_cylinder(), "mesh": analytic("circular_cylinder", (4, 4, 3, 3, 3, 4), radius=1.0, draft=0.5),
             "position": [0.0, 0.8, 0.0], "control_surface": cs}  # fmt: skip
        opts = {"higher_order": True, "motion": True, "drift_control_surface": True, "domain": "channel",
                "channel_width": 8.0}  # fmt: skip
        return model(b, [2.5, 3.5], [0.0], environment={"depth": 3.0}, options=opts).solve().drift.control_surface.real

    a, b = drift(1.5, 1.0), drift(2.0, 1.5)
    pk = np.abs(a[..., :2]).max()
    assert np.abs(a[..., :2] - b[..., :2]).max() <= 1e-5 * pk
    assert np.abs(a[..., 5] - b[..., 5]).max() <= 1e-5 * pk
    assert np.abs(a[..., 1]).max() >= 0.05 * pk  # the walls of the off-centre body give a sway drift


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_higher_order_channel_threads():
    two = [{"name": "a", "mesh": spline_cylinder(), "position": [2.0, 0.5, 0.0]},
           {"name": "b", "mesh": spline_cylinder(), "position": [-2.0, -0.7, 0.0], "yaw": 30.0}]
    out = [model(two, HO_W[:2], [0.0], environment={"depth": 3.0},
                 options={**HO_CHANNEL, "frequency_threads": ft, "inner_threads": it}).solve()
           for ft, it in ((1, 1), (2, 1), (1, 3))]  # fmt: skip
    for r in out[1:]:
        for q in QS:
            np.testing.assert_array_equal(view(r, q), view(out[0], q))


# ------------------------------------------------------------- damping lids and thin sheets

# the moonpool cylinder of the lid tests (its lid elements of its mesh, so that it moves with the
# body) and a fixed plate under it at z = -1.6, 6 m wide, in a channel 8 m wide in 10 m depth
LS_CHANNEL = {"domain": "channel", "channel_width": 8.0}
LS_W = np.sqrt(np.array([0.6, 1.5]) * G)
_HULL, _LID = moonpool_panels(n=12), moonpool_lid_panels(n=12)
_MOON = surface(np.concatenate([_HULL, _LID]), symmetry="xz_and_yz")
SIGNS = np.array([1, -1, 1, -1, 1, -1])  # mirror about y = 0: sway, roll and yaw change sign
W_D = np.sqrt(np.array([0.5, 1.0]) * G)


def _first(r, q):
    """The rows (and the excitation) of the first body's six coordinates."""
    a = view(r, q)
    return a[..., :6] if q == "excitation_diffraction" else a[:, :6, :]


def _mirrored(a, b, q):
    """The largest difference of the quantity q of a body and of its mirror image, of its peak."""
    s = SIGNS if q == "excitation_diffraction" else SIGNS[None, :, None] * SIGNS[None, None, :]
    return peak(_first(a, q) - _first(b, q) * s) / peak(_first(a, q))


def _solve(bodies, lids, opts, reflect, w, fp=None):
    env = {"depth": 10.0, **({"channel_reflect": reflect} if reflect is not None else {})}
    limit = not np.all(np.isfinite(w) & (np.asarray(w) > 0))
    return model(bodies, w, [] if limit else [0.0, 180.0], environment=env, lids=lids, field_points=fp,
                 options={**opts, **({"diffraction": False} if limit else {})}).solve()  # fmt: skip


def _moonpool(y=0.0, eps=0.5, plate=None, opts=LS_CHANNEL, reflect=None, w=LS_W, fp=None, **kw):
    """The moonpool cylinder y off the centreline with its lid (damping eps, None: no lid) and
    the plate under it (a sheet, a slab of the thickness plate, None: no plate)."""
    lid = part(_MOON, len(_HULL) + 1, len(_HULL) + len(_LID))
    lids = [{"name": "moon", "mesh": lid, "placed_with": "cyl", "damping_factor": eps}] if eps is not None else []
    bodies = [{"name": "cyl", "mesh": part(_MOON, 1, len(_HULL)), "position": [0.0, y, 0.0], **kw}]
    if plate is not None:
        p = {"dipoles": [{"mesh": submerged_plate(-1.6, 3.0, 3.0)}]} if plate == "sheet" else \
            {"mesh": submerged_plate(-1.6, 3.0, 3.0, t=plate)}  # fmt: skip
        bodies.append({"name": "plate", "modes": [], "position": [0.0, y, 0.0], **p})
    return _solve(bodies, lids, opts, reflect, w, fp)


def _wall_points(d):
    return np.array([(x0, sg * (4.0 - di), 0.0) for x0 in (0.0, 6.0) for sg in (1, -1) for di in d])


def _wall_slope(r, d):
    """The slope at the walls of the field pressure at the points of _wall_points (the quadratic
    fit in d), of k times its peak: the wall condition dphi/dy = 0."""
    worst = 0.0
    for f in range(len(LS_W)):
        k = brentq(lambda kk, K=LS_W[f] ** 2 / G: kk * np.tanh(kk * 10.0) - K, 1e-9, 10.0)
        fr, fd = r.field_points.pressure_radiation, r.field_points.pressure_diffraction
        for v in (fr[f, :, 1], fr[f, :, 2], fd[f, :, 0]):
            for b in range(len(v) // len(d)):
                c = np.linalg.lstsq(np.c_[np.ones(len(d)), d, d**2], v[len(d) * b : len(d) * (b + 1)], rcond=None)[0]
                worst = max(worst, abs(c[1]) / (k * peak(v)))
    return worst


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_damping_lid_in_a_channel():
    # the moonpool lid 1.5 m off the centreline, each check an identity of the problem: the wall
    # condition at the field points (1e-4 of k times the peak: the truncated far-image sums; 1.8e-5,
    # the field near the wall about 1e-6 of its peak); no reflection is the open sea and the mirror
    # body is mirrored (1e-5: float32 matrices, the image sums in another order); the lid without
    # damping is no lid (1e-6: float32, the same equations); the lid changes the coefficients by 60
    # to 80 % (so that these checks see it); the two formulations of one problem differ by their
    # discretization, which the lid does not change (1.1 times it without the lid plus 1e-4: 4.3e-3
    # against 7.6e-3 of the exciting force), the Haskind relation likewise (1.5 times: 5.1e-3
    # against 1.0e-2)
    d = np.array([0.01, 0.02, 0.03, 0.04])
    total = {**LS_CHANNEL, "diffraction_formulation": "total"}
    scattered = {**LS_CHANNEL, "diffraction_formulation": "scattered"}
    ch = _moonpool(1.5, opts=total, fp=_wall_points(d))
    assert _wall_slope(ch, d) <= 1e-4
    sea = _moonpool(1.5, opts={"diffraction_formulation": "total"})
    ch0, mirror = _moonpool(1.5, opts=total, reflect=0.0), _moonpool(-1.5, opts=total)
    zero, none = _moonpool(1.5, 0.0, opts=total), _moonpool(1.5, None, opts=total)
    for q in QS:
        assert peak(view(ch0, q) - view(sea, q)) <= 1e-5 * peak(view(sea, q)), q
        assert _mirrored(ch, mirror, q) <= 1e-5, q
        assert peak(view(zero, q) - view(none, q)) <= 1e-6 * peak(view(none, q)), q
        assert peak(view(ch, q) - view(none, q)) >= 0.3 * peak(view(none, q)), q
    sc, sc_none = _moonpool(1.5, opts=scattered), _moonpool(1.5, None, opts=scattered)
    np.testing.assert_array_equal(sc.added_mass, ch.added_mass)
    dx = [peak(a.excitation.diffraction - b.excitation.diffraction) / peak(b.excitation.diffraction)
          for a, b in ((sc, ch), (sc_none, none))]  # fmt: skip
    assert dx[0] <= 1.1 * dx[1] + 1e-4, dx
    hk = [peak(r.excitation.haskind - r.excitation.diffraction) / peak(r.excitation.diffraction) for r in (ch, none)]
    assert hk[0] <= 1.5 * hk[1], hk


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_thin_sheet_in_a_channel():
    # the plate 1 m from the walls under the moonpool with its lid (the scattered formulation), each
    # check an identity of the problem: the wall condition at the field points (2e-4: the truncated
    # far-image sums, the sheet 1 m from the wall; 8.5e-5 found); no reflection is the open sea and
    # the mirror body (0.5 m off the centreline) is mirrored (1e-5: float32 matrices, the image sums
    # in another order); the walls change the coefficients by 20 to 60 % (so that these checks see
    # them); the Haskind relation with the jump term of the sheet to the discretization of the open
    # sea (1.5 times its level there plus 1e-3). The slabs that tend to the sheet are validation
    d = np.array([0.01, 0.02, 0.03, 0.04])
    sheet = _moonpool(plate="sheet", fp=_wall_points(d))
    assert _wall_slope(sheet, d) <= 2e-4
    sea, ch0 = _moonpool(plate="sheet", opts={}), _moonpool(plate="sheet", reflect=0.0)
    for q in QS:
        assert peak(_first(ch0, q) - _first(sea, q)) <= 1e-5 * peak(_first(sea, q)), q
        assert peak(_first(sheet, q) - _first(sea, q)) >= 0.1 * peak(_first(sea, q)), q
    pos, neg = _moonpool(0.5, plate="sheet"), _moonpool(-0.5, plate="sheet")
    assert all(_mirrored(pos, neg, q) <= 1e-5 for q in QS)
    hk = [peak(r.excitation.haskind[..., :6] - X) / peak(X)
          for r in (sheet, sea) for X in [r.excitation.diffraction[..., :6]]]  # fmt: skip
    assert hk[0] <= 1.5 * hk[1] + 1e-3, hk


def _barge(y=0.0, eps=0.5, plate=None, opts=LS_CHANNEL, reflect=None, w=LS_W, modes=None):
    """The moonpool barge (higher order) y off the centreline with its lid patch (damping eps,
    None: no lid) over the plate (a sheet, a slab of the thickness plate, None: no plate)."""
    barge = moonpool_barge()
    lid = {"name": "l", "mesh": part(barge, 7, 7), "placed_with": "b", "damping_factor": eps}
    lids = [lid] if eps is not None else []
    body = {"name": "b", "mesh": part(barge, 1, 6), "position": [0.0, y, 0.0]}
    bodies = [{**body, "modes": modes} if modes else body]
    if plate is not None:
        p = {"dipoles": [{"mesh": ho_submerged_plate(-1.6, 3.0, 3.0, n=(6, 6))}]} if plate == "sheet" else \
            {"mesh": ho_submerged_plate(-1.6, 3.0, 3.0, n=(6, 6), t=plate)}  # fmt: skip
        bodies.append({"name": "plate", "modes": [], "position": [0.0, y, 0.0], **p})
    return _solve(bodies, lids, {**HO, **opts}, reflect, w)


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_higher_order_damping_lid_in_a_channel():
    # the moonpool barge with its lid patch, identities of the problem: no reflection is the open
    # sea and the lid without damping is no lid (1e-10: the same equations in double precision); A
    # and B symmetric and the Haskind relation within 2e-3, the bound of the higher order (the patch
    # quadrature at the waterline). The wall effect against the low order is validation
    ch, sea, ch0 = _barge(), _barge(opts={}), _barge(reflect=0.0)
    zero, none = _barge(eps=0.0), _barge(eps=None)
    for q in QS:
        assert peak(view(ch0, q) - view(sea, q)) <= 1e-10 * peak(view(sea, q)), q
        assert peak(view(zero, q) - view(none, q)) <= 1e-10 * peak(view(none, q)), q
    A, B = ch.added_mass, ch.radiation_damping
    assert peak(A - A.transpose(0, 2, 1)) <= 2e-3 * peak(A) and peak(B - B.transpose(0, 2, 1)) <= 2e-3 * peak(B)
    assert peak(ch.excitation.haskind - ch.excitation.diffraction) <= 2e-3 * peak(ch.excitation.diffraction)


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_higher_order_thin_sheet_in_a_channel():
    # the barge with its lid over the plate: no reflection is the open sea (1e-10: the same
    # equations in double precision). The slabs that tend to the sheet are validation
    sea, ch0 = _barge(plate="sheet", opts={}), _barge(plate="sheet", reflect=0.0)
    for q in QS:
        assert peak(_first(ch0, q) - _first(sea, q)) <= 1e-10 * peak(_first(sea, q)), q


def test_dipole_dampers_in_a_channel():
    # a porous plate under a cylinder 1.2 m from the walls, a damper of the body or of the water
    # (theory §9.5): zero damping is no damper (1e-6: float32, the same equations; 2e-8 found);
    # large damping the sheet (the body's) or the fixed sheet (the water's), the jump leaving a
    # normal velocity O(omega / eps) through the sheet: 1e-3 (3.3e-4 of the damping found at
    # eps = 1e5, ten times that at 1e4); A and B symmetric to the discretization of the coarse hull
    # (5e-3: 2.8e-3 and 4.1e-3 for the hull alone in the channel), B positive semidefinite (energy;
    # -1e-6: round-off)
    hull = cylinder_panels(1.0, 1.0, n_phi=6, n_z=4, n_r=3, quarter=True)
    plate = np.array(face_panels((0.0, 0.0, -1.5), (1.4, 0, 0), (0, 1.4, 0), 4, 4, (0, 0, 1)))
    both = surface(np.concatenate([hull, plate]), symmetry="xz_and_yz")
    wetted, sheet = part(both, 1, len(hull)), part(both, len(hull) + 1, len(hull) + len(plate))
    opts = {"domain": "channel", "channel_width": 5.0, "diffraction_formulation": "scattered"}

    def run(bodies, dampers=(), **kw):
        heads = [0.0, 180.0] if kw.get("diffraction", True) else []
        return model(bodies, W_D, heads, environment={"depth": 6.0}, options={**opts, **kw},
                     dampers=list(dampers)).solve()  # fmt: skip

    def damped(eps, owner, **kw):
        d = {"mesh": sheet, "damping_factor": eps}
        if owner == "body":
            return run({"mesh": wetted, "dipole_dampers": [d]}, **kw)
        return run({"mesh": wetted}, [d], **kw)

    plain = run({"mesh": surface(hull, symmetry="xz_and_yz")})
    fixed = {"name": "plate", "modes": [], "dipoles": [{"mesh": surface(plate, symmetry="xz_and_yz")}]}
    rigid = {"body": run({"mesh": wetted, "dipoles": [{"mesh": sheet}]}),
             "water": run([{"name": "hull", "mesh": surface(hull, symmetry="xz_and_yz")}, fixed])}  # fmt: skip
    for owner in ("water", "body"):
        zero, large = damped(0.0, owner), damped(1e5, owner)
        for q in QS:
            assert peak(view(zero, q) - view(plain, q)) <= 1e-6 * peak(view(plain, q)), (owner, q)
            assert peak(view(large, q) - _first(rigid[owner], q)) <= 1e-3 * peak(view(large, q)), (owner, q)
        r = damped(2.0, owner, diffraction=False)
        for a in (r.added_mass, r.radiation_damping):
            assert all(peak(a[f] - a[f].T) <= 5e-3 * peak(a[f]) for f in range(len(W_D))), owner
        b = r.radiation_damping
        assert all(np.linalg.eigvalsh(0.5 * (b[f] + b[f].T)).min() >= -1e-6 * peak(b[f]) for f in range(len(W_D)))
