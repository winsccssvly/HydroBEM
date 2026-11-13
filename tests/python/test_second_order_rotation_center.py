"""The second-order loads of a body about its rotation centre R (theory §1.1, §11.2, §11.3;
theory §12.2, §12.7, §12.13): the mean drifts by pressure integration and by the control surface,
their quadratic QTFs, the potential QTF and Newman's approximation have their moments about R and
are the loads with the second-order displacement of R zero (the second-order rotation H applied
about R, H(x − R)). Against the same body with R = 0, turned by M_R = M_O − R × F, they differ by
the load of the rigid second-order translation δ = −H R of the pair (H of the pair coefficients
of the rotations): in the quadratic part the hydrostatic load of the translated body,
ΔF = −ρg A_wp δ_z e_z and ΔM_O = ρg [δ_z (−A_wp y_f, A_wp x_f, 0) + V (δ_y, −δ_x, 0)], in the
potential part the radiation load (Ω² A(Ω) − iΩ B(Ω)) δ. The momentum drift (global, about the
z axis) does not depend on R. The body is in equilibrium (the displaced mass, G above the centre of
buoyancy), so that its motions about R and about O describe the same motion."""

import math
import warnings

import numpy as np
import pytest

from bodies import lo_cosine_cylinder, lo_free_cylinder
from problems import model

G = 9.80665
RHO = 1000.0
W = np.array([1.2, 1.6])
R = np.array([0.3, -0.2, -0.1])
CS = {"method": "automatic", "radius": 2.0, "depth": 1.0, "panel_size": 0.5}
DRIFTS = ("drift_pressure_integration", "drift_control_surface", "drift_momentum_conservation")


def _floating(rc):
    body = lo_free_cylinder(quarter=False, control_surface=CS, rotation_center=list(rc), mass="displaced")
    options = {"motion": True, **{d: True for d in DRIFTS}, "drift_bidirectional": True,
               "qtf_quadratic_pressure_integration": True, "qtf_quadratic_control_surface": True,
               "qtf_newman": True, "qtf_bidirectional": True}  # fmt: skip
    return model([body], W, (0.0, 30.0), environment={"depth": math.inf, "gravity": G}, options=options)


def _fixed_potential(rc):
    body = lo_cosine_cylinder(fixed=True, rotation_center=list(rc))
    m = model([body], W, (30.0,), environment={"depth": 3.0, "gravity": G},
              options={"qtf_potential_direct": True, "qtf_potential_indirect": True})  # fmt: skip
    m.second_order.update(free_surface_inner_radius=3.0, free_surface_panel_size=0.4, quadrature_annuli=1)
    return m


def _floating_potential(rc):
    body = lo_cosine_cylinder(rotation_center=list(rc), mass="displaced")
    m = model([body], W, (30.0,), environment={"depth": 3.0, "gravity": G},
              options={"motion": True, "qtf_potential_direct": True, "qtf_potential_indirect": True,
                       "qtf_quadratic_pressure_integration": True})  # fmt: skip
    m.second_order.update(free_surface_inner_radius=3.0, free_surface_panel_size=0.4, quadrature_annuli=1)
    return m


def _solve(build, rc):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build(rc).solve()


def _transfer(v, r):
    """The six rigid rows (the last axis) with the moment about O turned to r: M − r × F."""
    v = np.array(v, dtype=np.complex128)
    v[..., 3:6] -= np.cross(r, v[..., 0:3])
    return v


def _delta(aa, ab, total):
    """δ = −H R of the rotations aa (wave a) and ab (wave b), [..., 3]: H of theory §11.2 with each
    mean replaced by the difference (total false) or the sum (total true) pair coefficient."""
    if total:
        p = lambda i, k: 0.25 * (aa[..., i] * ab[..., k] + ab[..., i] * aa[..., k])  # noqa: E731
    else:
        p = lambda i, k: 0.25 * (aa[..., i] * np.conj(ab[..., k]) + np.conj(ab[..., i]) * aa[..., k])  # noqa: E731
    h00, h11, h22 = -0.5 * (p(1, 1) + p(2, 2)), -0.5 * (p(0, 0) + p(2, 2)), -0.5 * (p(0, 0) + p(1, 1))
    hr = [h00 * R[0], p(0, 1) * R[0] + h11 * R[1], p(0, 2) * R[0] + p(1, 2) * R[1] + h22 * R[2]]
    return -np.stack(hr, axis=-1)


def _hydrostatic_load(result, d):
    """The hydrostatic load about R of the body translated by d, [..., 6] (theory §11.2)."""
    a = float(np.asarray(result.quantity("hydrostatics_waterplane_area"))[0])
    xf, yf = np.asarray(result.quantity("hydrostatics_waterplane_centroid"))[0]
    v = float(np.asarray(result.quantity("hydrostatics_volumes"))[0, 2])  # ∫ z n_z dS, as the closed forms
    rg = RHO * G
    f = np.zeros(d.shape[:-1] + (6,), dtype=np.complex128)
    f[..., 2] = -rg * a * d[..., 2]
    f[..., 3] = rg * (-d[..., 2] * a * yf + v * d[..., 1])
    f[..., 4] = rg * (d[..., 2] * a * xf - v * d[..., 0])
    f[..., 3:6] -= np.cross(R, f[..., 0:3])  # about R
    return f


def _pairs(result, name, headings):
    """The heading indices (a, b) of the heading pairs `name` [P, 2]."""
    p = np.asarray(result.quantity(name))
    idx = [[int(np.argmin(np.abs(np.asarray(headings) - x))) for x in row] for row in p]
    return np.array(idx)[:, 0], np.array(idx)[:, 1]


def _qtf_delta(result, total, headings=(0.0, 30.0)):
    """δ of every pair of the QTFs [F, F, P, 3] from the motions [F, H, 6] about R."""
    al = np.asarray(result.quantity("motion"))[..., 3:6].astype(np.complex128)
    ha, hb = _pairs(result, "qtf_heading_pairs", headings)
    return _delta(al[:, None, ha, :], al[None, :, hb, :], total)


def _drift_delta(result, headings=(0.0, 30.0)):
    """δ of the drift heading pairs [F, P, 3]."""
    al = np.asarray(result.quantity("motion"))[..., 3:6].astype(np.complex128)
    ha, hb = _pairs(result, "drift_heading_pairs", headings)
    return _delta(al[:, ha], al[:, hb], False)


@pytest.fixture(scope="module")
def floating():
    return _solve(_floating, (0.0, 0.0, 0.0)), _solve(_floating, R)


def _close(a, b, rel=2e-6):
    # the first-order outputs the loads are formed from are float32: 2e-6 of the peak
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    np.testing.assert_allclose(a, b, rtol=0.0, atol=rel * np.abs(a).max())


QUANTITIES = (
    "drift_pressure_integration",
    "drift_control_surface",
    "qtf_quadratic_pressure_integration_difference",
    "qtf_quadratic_pressure_integration_sum",
    "qtf_quadratic_control_surface_difference",
    "qtf_quadratic_control_surface_sum",
)


@pytest.mark.parametrize("name", QUANTITIES)
def test_the_quadratic_loads_are_about_the_rotation_center(floating, name):
    # R ≠ 0 against R = 0 turned to R plus the hydrostatic load of the translation δ = −H R
    r0, r1 = floating
    d = _drift_delta(r1) if name.startswith("drift") else _qtf_delta(r1, name.endswith("_sum"))
    extra = _hydrostatic_load(r1, d)
    assert np.abs(extra).max() > 1e-3 * np.abs(np.nan_to_num(r1.quantity(name))).max()  # it is there
    assert not extra[..., [0, 1, 5]].any()  # the horizontal force and the yaw moment unchanged
    _close(np.asarray(r1.quantity(name)), _transfer(r0.quantity(name), R) + extra)


def test_the_moment_about_the_moving_rotation_center(floating):
    # M_R − <ξ_R × F1> = (M_O − <ξ_O × F1>) − R × F − <(α × R) × F1> (+ the moment of δ): the
    # moving point R less the transfer of the moving O, from the motions and the first-order force
    # of a freely floating body without other forces, F1 = −ω² m ξ_G (ξ_G = ξ_R + α × (G − R))
    r0, r1 = floating
    xi = np.asarray(r1.quantity("motion"))  # [F, H, 6] about R
    al = xi[..., 3:6]
    mass = float(np.asarray(r1.quantity("hydrostatics_mass"))[0])
    f1 = -(W**2)[:, None, None] * mass * (xi[..., 0:3] + np.cross(al, np.array([0.0, 0.0, -0.2]) - R))
    term = -0.5 * np.cross(np.cross(al, R), np.conj(f1)).real  # −<(α × R) × F1>, [F, H, 3]
    term += _hydrostatic_load(r1, _drift_delta(r1)[:, [0, 2]]).real[..., 3:6]
    for drift in ("drift_pressure_integration", "drift_control_surface"):
        m0 = np.asarray(r0.quantity(drift + "_body_axes"))[:, [0, 2], :].real  # the equal headings
        m1 = np.asarray(r1.quantity(drift + "_body_axes"))[:, [0, 2], :].real
        f = np.asarray(r0.quantity(drift))[:, [0, 2], 0:3].real
        diff = m1 - (m0 - np.cross(R, f))
        assert np.abs(term).max() > 1e-3 * np.abs(m1).max()  # the term is there
        np.testing.assert_allclose(diff, term, rtol=0, atol=1e-3 * np.abs(m1).max())
        assert np.isnan(np.asarray(r1.quantity(drift + "_body_axes"))[:, 1]).all()  # (0, 30) has none


def test_newman_is_built_from_the_drifts_about_the_rotation_center(floating):
    r0, r1 = floating
    for drift, newman in (("drift_pressure_integration", "qtf_newman_pressure_integration_difference"),
                          ("drift_control_surface", "qtf_newman_control_surface_difference")):  # fmt: skip
        n1 = np.asarray(r1.quantity(newman))  # [F, F, H, D], real
        d1 = np.asarray(r1.quantity(drift)).real  # [F, P, D]; P = 3 pairs: (0, 0), (0, 30), (30, 30)
        diag = d1[:, [0, 2], :]  # the equal headings
        for i in range(2):
            _close(n1[i, i], diag[i], 1e-12)
        a, b = diag[0], diag[1]
        geo = np.where(a * b > 0, np.sign(a) * np.sqrt(np.abs(a * b)), 0.0)
        _close(n1[0, 1], geo, 1e-12)
        # the horizontal force rows are those of R = 0 (the translation δ has no horizontal load;
        # small at these frequencies: 2e-6 of the peak of all rows, the float32 motions)
        n0 = np.asarray(r0.quantity(newman))
        np.testing.assert_allclose(n1[..., 0:2], n0[..., 0:2], rtol=0, atol=2e-6 * np.abs(n1).max())


def test_the_momentum_drift_does_not_depend_on_the_rotation_center(floating):
    # global axes, about the z axis, for any R: the same as with R = 0 but for the float32 motions
    # and Kochin functions, which the far-field sums at these low frequencies (drifts of 1e-2 of
    # the peak, from cancellations) amplify to 5e-3 of the peak
    r0, r1 = floating
    for name in ("drift_momentum_conservation", "qtf_newman_momentum_conservation_difference"):
        _close(r1.quantity(name), r0.quantity(name), 1e-2)


def test_the_pressure_and_control_surface_drifts_agree_about_the_rotation_center():
    # two methods of one load, both about R (ω = 4.43 rad/s, kR = 2.0, above the heave resonance,
    # where the 32 panels around the cylinder give the pressure drift to a few per cent, 2.8 %
    # observed; nearer the resonance it converges slowly with the panels): the horizontal
    # force and the yaw moment, which includes the transfer −R × F (R_x F_y − R_y F_x)
    body = lo_free_cylinder(quarter=False, control_surface=CS, rotation_center=list(R), mass="displaced")
    m = model([body], np.array([4.43]), (0.0, 30.0), environment={"depth": math.inf, "gravity": G},
              options={"motion": True, "drift_pressure_integration": True, "drift_control_surface": True})  # fmt: skip
    r = _solve(lambda _: m, R)
    p = np.asarray(r.quantity("drift_pressure_integration")).real
    c = np.asarray(r.quantity("drift_control_surface")).real
    for k in (0, 1, 5):
        np.testing.assert_allclose(c[..., k], p[..., k], rtol=0, atol=0.08 * np.abs(p[..., k]).max())


@pytest.mark.parametrize("method", ("direct", "indirect"))
def test_the_potential_qtf_of_a_fixed_body_has_its_moments_about_the_rotation_center(method):
    r0, r1 = _solve(_fixed_potential, (0.0, 0.0, 0.0)), _solve(_fixed_potential, R)
    for part in ("difference", "sum"):
        name = f"qtf_potential_{method}_{part}"
        _close(np.asarray(r1.quantity(name)), _transfer(r0.quantity(name), R))


@pytest.mark.slow  # seconds: deselected by scripts/test.* fast
def test_the_two_parts_of_the_qtf_of_a_floating_body_take_one_rotation_center():
    # the floating cylinder, R ≠ 0 against R = 0 turned: the quadratic part differs by the
    # hydrostatic load of δ = −H R, the potential part by the radiation load of δ,
    # (Ω² A(Ω) − iΩ B(Ω)) δ, with A and B of a first-order solve at |Ω| about R (Ω = ω_a − ω_b < 0
    # for a < b), so that the total differs by the load of one translation. The direct method
    # integrates φ⁽²⁾ against n_j, as A_jk; the indirect one takes ∫ ψ_j Q_B, which for
    # Q_B = iΩ δ·n is A_kj, the transpose (equal by reciprocity to the discretization: 3e-3 of
    # the δ term here); each agrees with its own to the float32 coefficients
    r0, r1 = _solve(_floating_potential, (0.0, 0.0, 0.0)), _solve(_floating_potential, R)
    omegas = np.array([0.4, 2.4, 2.8, 3.2])  # |ω_a − ω_b| ≠ 0 and ω_a + ω_b
    body = lo_cosine_cylinder(rotation_center=list(R), mass="displaced")
    rad = model([body], omegas, (), environment={"depth": 3.0, "gravity": G},
                options={"diffraction": False, "radiation": True})  # fmt: skip
    rad = _solve(lambda _: rad, R)
    am, bd = np.asarray(rad.quantity("added_mass")), np.asarray(rad.quantity("radiation_damping"))
    for part, total in (("difference", False), ("sum", True)):
        d = _qtf_delta(r1, total, (30.0,))[:, :, 0]  # [F, F, 3], the one heading pair
        quad = f"qtf_quadratic_pressure_integration_{part}"
        q0, q1 = (np.asarray(r.quantity(quad))[:, :, 0] for r in (r0, r1))
        hydro = _hydrostatic_load(r1, d)
        _close(q1, _transfer(q0, R) + hydro)
        for method, transpose in (("direct", False), ("indirect", True)):
            radiation = np.zeros_like(hydro)
            for a in range(2):
                for b in range(2):
                    om = W[a] + W[b] if total else W[a] - W[b]
                    if om == 0.0:
                        continue  # the potential force is zero at equal frequencies
                    k = int(np.argmin(np.abs(omegas - abs(om))))
                    z = om**2 * am[k] - 1j * om * bd[k]
                    radiation[a, b] = (z.T if transpose else z)[:, 0:3] @ d[a, b]
            name = f"qtf_potential_{method}_{part}"
            p0, p1 = (np.asarray(r.quantity(name))[:, :, 0] for r in (r0, r1))
            assert np.abs(p1 - _transfer(p0, R)).max() > 4e-5 * np.abs(p1).max()  # 20 times the bound
            _close(p1, _transfer(p0, R) + radiation)
            _close(p1 + q1, _transfer(p0 + q0, R) + hydro + radiation)
