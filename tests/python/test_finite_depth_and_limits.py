"""Finite depth, the omega = 0 and omega = inf limits at all
depths, the diffraction-only option.

A5 (truncated cylinder in finite depth vs the eigenfunction-matching
reference of scripts/gen_cylinder_reference.py), a cylinder standing on the sea bed (the exciting
force of MacCamy & Fuchs 1954 and the mean drift of the same solution), A6 (limits of the floating
hemisphere vs the exact sphere series), the approach of the limits by small
and large frequencies, and the identities of test-plan B on the finite-depth route, each on one
mesh with the cause of its bound. The convergence with the mesh (A5's Richardson extrapolation
with the observed order, A6's and the formulations' decrease with refinement) is validation, run
when the method changes, not tests. Meshes are procedural (tests/python/meshes.py)."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import special as sp

import hydrobem as hb

from bodies import graded_cylinder, hemi, part, surface
from meshes import box_panels, cylinder_panels
from problems import analytic, bottom_cylinder, model, peak, solve, view

RHO, G = 1000.0, 9.80665
DATA = Path(__file__).resolve().parents[1] / "data"


# ------------------------------------------------------------- dispersion


def test_wavenumber_finite_depth():
    # A3: the dispersion relation to the double round-off
    for w, h in ((0.3, 450.0), (3.0, 450.0), (1.0, 10.0), (1.0, 2.0), (0.05, 3.0), (5.0, 1.0), (1e-4, 10.0)):
        k = hb.wavenumber(w, depth=h, gravity=G)
        assert abs(k * math.tanh(k * h) - w * w / G) <= 4e-16 * w * w / G
    assert hb.wavenumber(30.0, depth=1.0, gravity=G) == 900.0 / G  # tanh kh = 1 in double


# ------------------------------------------------------------- A5


def cylinder_values(n):
    """The heave and surge added mass and damping of the floating cylinder a = 1, draft 1, at the
    depths 2 and 4 and Ka = 0.4, 0.8, 1.5 on the graded mesh of n panels a quarter circle, and the
    eigenfunction-matching reference of scripts/gen_cylinder_reference.py (both [rows, 4])."""
    ref = np.loadtxt(DATA / "cylinder_finite_depth.txt")
    rows = []
    for h in (2.0, 4.0):
        sel = ref[ref[:, 0] == h]
        w = np.sqrt(sel[:, 2] * G)
        r = solve({"mesh": graded_cylinder(n), "modes": ["surge", "heave"]}, w, depth=h,
                  options={"diffraction": False})  # fmt: skip
        A, B = r.added_mass / RHO, r.radiation_damping / RHO
        rows.append(np.c_[A[:, 2, 1], B[:, 2, 1] / w, A[:, 0, 0], B[:, 0, 0] / w])
    return np.vstack(rows), ref[:, 3:]


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_A5_truncated_cylinder_finite_depth():
    """Floating cylinder a = 1, draft 1, depths 2 and 4, Ka = 0.4, 0.8, 1.5: heave and surge added
    mass and damping on 24 and 32 panels a quarter circle against the eigenfunction-matching
    reference (converged in its truncation to 1e-5): each quantity within 3e-3 of the largest
    reference value at 32 (2.8e-3 to 7.7e-4 found) and converging to it at the order of the
    discretization O(h^p), p about 1.8 (the bottom edge; 1.74 to 1.91 found from 24 to 32), so
    between 1.5 and 2.2. The finest mesh (48, within 2e-3), the convergence of three meshes and
    the Richardson extrapolation with the observed order are validation."""
    errs = {}
    for n in (24, 32):
        vals, exact = cylinder_values(n)
        errs[n] = np.abs(vals - exact).max(axis=0) / np.abs(exact).max(axis=0)
    order = np.log(errs[24] / errs[32]) / np.log(32 / 24)
    assert np.all(errs[32] <= 3e-3), errs[32]
    assert np.all((order >= 1.5) & (order <= 2.2)), order


# ------------------------------------------------------------- a body on the sea bed


def test_bottom_mounted_cylinder_force_and_drift():
    """The fixed cylinder a = 1 standing on the sea bed in the depth 2 (the wall from z = 0 down to
    the bed, no bottom), head seas at ka = 0.5, 1.5, 2.8 (the last above the first irregular
    frequency, ka = 2.405, removed by the waterplane as interior surface): the surge exciting force
    against MacCamy & Fuchs (1954) and the mean surge drift by pressure integration and by momentum
    conservation against the pressure integration of the same analytic solution (exact to the round-
    off). The bound is the discretization of 6 x 6 spans a quadrant of the wall: within 1e-4 of the
    largest value for the force (2.8e-5 found), 2e-4 for the two drifts (4.7e-5 and 1.0e-4 found;
    8 x 8 spans: 6e-6, 2.9e-5, 2.2e-5)."""
    ka = np.array([0.5, 1.5, 2.8])
    ref = np.array([bottom_cylinder(x, 2.0) for x in ka])
    cyl = analytic("circular_cylinder", [(6, 6, 3, 3, 3, 4), (6, 2, 3, 3, 3, 4)], parts=["side", "waterplane"],
                   mapping="uniform", radius=1.0, draft=2.0)  # fmt: skip
    body = {"mesh": part(cyl, 1, 1), "interior_surface": part(cyl, 2, 2), "modes": []}
    w = np.sqrt(G * ka * np.tanh(2.0 * ka))
    r = solve(body, w, [0.0], depth=2.0, options={"higher_order": True, "radiation": False,
              "drift_pressure_integration": True, "drift_momentum_conservation": True})  # fmt: skip
    force = np.abs(r.coefficient("excitation_diffraction", ("body", "surge"))[:, 0]) / (RHO * G)
    pressure = r.bodies["body"].drift.pressure_integration[:, 0, 0].real / (RHO * G)
    momentum = r.drift.momentum_conservation[:, 0, 0].real / (RHO * G)
    for got, k, bound in ((force, 0, 1e-4), (pressure, 1, 2e-4), (momentum, 1, 2e-4)):
        err = peak(got - ref[:, k]) / peak(ref[:, k])
        assert err <= bound, (k, err)


@pytest.mark.parametrize(("cosine", "bound"), ((False, 0.1), (True, 0.05)))
def test_low_order_pressure_drift_takes_the_potential_at_the_waterline(cosine, bound):
    """The low-order pressure drift of the cylinder on the sea bed (8 x 8 panels a quadrant, ka =
    0.5 and 1.5) against the analytic drift: the waterline term takes the potential of its panel
    expanded from the centroid to the waterline (theory §11.4). The centroid value alone, lower by
    about K z_c, gave 6.1e-1 (uniform panels) and 2.3e-1 (cosine-spaced) of the largest value;
    with the expansion 7.8e-2 and 3.5e-2, about as close as the momentum drift on these meshes
    (2.7e-2 and 3.0e-2)."""
    ka = np.array([0.5, 1.5])
    ref = np.array([bottom_cylinder(x, 2.0) for x in ka])[:, 1]
    pan = cylinder_panels(1.0, 2.0, n_phi=8, n_z=8, n_r=0, quarter=True, cosine=cosine)
    body = {"mesh": surface(pan, symmetry="xz_and_yz"), "modes": []}
    w = np.sqrt(G * ka * np.tanh(2.0 * ka))
    r = solve(body, w, [0.0], depth=2.0, options={"radiation": False, "drift_pressure_integration": True})
    pressure = r.bodies["body"].drift.pressure_integration[:, 0, 0].real / (RHO * G)
    assert peak(pressure - ref) / peak(ref) <= bound


_WATERLINE_CENTROID = """
import json, sys
import numpy as np
sys.path[:0] = {paths!r}
from bodies import surface
from meshes import cylinder_panels
from problems import solve
ka = np.array([0.5, 1.5])
pan = cylinder_panels(1.0, 2.0, n_phi=8, n_z=8, n_r=0, quarter=True, cosine=False)
w = np.sqrt(9.80665 * ka * np.tanh(2.0 * ka))
r = solve({{"mesh": surface(pan, symmetry="xz_and_yz"), "modes": []}}, w, [0.0], depth=2.0,
          options={{"radiation": False, "drift_pressure_integration": True}})
print(json.dumps({{"drift": r.bodies["body"].drift.pressure_integration[:, 0, 0].real.tolist(),
                  "codes": sorted({{d.code for d in r.diagnostics}}), "hooks": list(r.meta.environment_hooks)}}))
"""


def test_the_waterline_centroid_hook_takes_the_centroid_value():
    """The diagnostic hook HB_WATERLINE_CENTROID (read once per process, so in a process of its
    own): the low-order waterline term takes the centroid value of the potential again, the drift
    of the test above before the expansion (6.1e-1 of the largest value on the uniform mesh, the
    expansion 7.8e-2); the hook is listed in the meta data and is the warning of code 9."""
    here = Path(__file__).resolve().parent
    paths = [str(here.parents[1] / "python"), str(here)]
    env = {k: v for k, v in os.environ.items() if k != "HB_WATERLINE_CENTROID"}
    out = {}
    for on in (False, True):
        run = subprocess.run([sys.executable, "-c", _WATERLINE_CENTROID.format(paths=paths)],
                             env={**env, "HB_WATERLINE_CENTROID": "1"} if on else env, capture_output=True,
                             text=True, check=True)  # fmt: skip
        out[on] = json.loads(run.stdout.strip().splitlines()[-1])
    assert out[False]["hooks"] == [] and 9 not in out[False]["codes"]
    assert out[True]["hooks"] == ["HB_WATERLINE_CENTROID"] and 9 in out[True]["codes"]
    ref = np.array([bottom_cylinder(x, 2.0) for x in (0.5, 1.5)])[:, 1]
    err = {on: peak(np.array(out[on]["drift"]) / (RHO * G) - ref) / peak(ref) for on in out}
    assert err[False] <= 0.1 and 0.55 <= err[True] <= 0.7, err


# ------------------------------------------------------------- A6


def _gl(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def sphere_series(mode, limit, nmax=240):
    """Added mass / m0 (m0 = 2 pi rho a^3 / 3) of the floating hemisphere a = 1
    at omega = 0 (rigid lid: the even extension) or inf (phi = 0 on z = 0: the
    odd extension): the exact exterior Neumann series of the whole sphere with
    the extended normal velocity (heave: g(mu) Legendre P_n, surge: cos(phi)
    g(mu) with P_n^1), mu = cos(theta), phi = sum_n -g_n / (n + 1) r^-(n+1) P_n^m."""
    m = 0 if mode == "heave" else 1
    mu, wq = (
        np.concatenate([_gl(400, -1, 0)[0], _gl(400, 0, 1)[0]]),
        np.concatenate([_gl(400, -1, 0)[1], _gl(400, 0, 1)[1]]),
    )
    nlow = mu < 0
    s = np.sqrt(1 - mu * mu)
    n_i = mu if m == 0 else s  # n_3 = mu, n_1 = sin(theta) cos(phi)
    # image velocity: even extension keeps n_3 / n_1 at the mirror point mapped
    # through the mirror normal; heave: -|mu| (even), mu (odd); surge: s (even), -sgn(mu) s (odd)
    if m == 0:
        g = -np.abs(mu) if limit == "zero" else mu
    else:
        g = s if limit == "zero" else -np.sign(mu) * s
    phi = np.zeros_like(mu)
    for n in range(m, nmax):
        P = sp.lpmv(m, n, mu)
        norm = 2 / (2 * n + 1) * (1 if m == 0 else n * (n + 1))
        gn = np.sum(wq * g * P) / norm
        phi += -gn / (n + 1) * P
    # A = -rho int_lower n_i phi dS; heave: 2 pi int mu phi dmu, surge: pi int s phi dmu (cos^2 phi)
    integral = np.sum((wq * n_i * phi)[nlow]) * (2 * math.pi if m == 0 else math.pi)
    return -integral / (2 * math.pi / 3)


def test_sphere_series_closed_forms():
    # heave at inf and surge at 0 are the translating sphere: half its added mass, 0.5 m0 (the
    # series of one term, n = 1; the Gauss quadrature to the double round-off: 1.3e-14 observed)
    assert abs(sphere_series("heave", "infinite") - 0.5) < 1e-12
    assert abs(sphere_series("surge", "zero") - 0.5) < 1e-12


def hemisphere_limit_errors(n):
    """The surge and heave added mass of the floating hemisphere at omega = 0 and inf on n panels a
    quadrant edge against the exact sphere series, relative to the largest."""
    exact = np.array([sphere_series(m, lim) for lim in ("zero", "infinite") for m in ("surge", "heave")])
    m0 = RHO * 2 / 3 * math.pi
    r = solve({"mesh": hemi(n), "modes": ["surge", "heave"]}, [0.0, math.inf], options={"diffraction": False})
    a0, ai = r.added_mass_zero_frequency / m0, r.added_mass_infinite_frequency / m0
    got = np.array([a0[0, 0], a0[2, 1], ai[0, 0], ai[2, 1]])
    return np.abs(got - exact) / np.abs(exact).max()


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_A6_hemisphere_limits_infinite_depth():
    # the exact series (converged: 240 terms); the O(h^2) discretization of 36 panels a quadrant
    # edge leaves 5.3e-4 (1.1e-3 at 24, 3.7e-3 at 12: the convergence is validation)
    errs = hemisphere_limit_errors(36)
    assert np.all(errs <= 1e-3), errs


def test_limits_finite_depth_surge_and_rejection_of_heave():
    m = hemi(12)
    r = solve(
        {"mesh": m, "modes": ["surge"]}, [0.0, math.inf, 1e-3, 1e-2], depth=3.0, options={"diffraction": False}
    )
    a0 = r.added_mass_zero_frequency[0, 0]
    # the limit of the theory approached (not substituted): A(w) - A(0) = O((kh)^2) in finite depth,
    # 3e-7 at w = 1e-3 (kh = 5.5e-4; 1.1e-7 observed, 8.3e-6 at w = 1e-2); heave at omega = 0 is an
    # input rule (no solution with a net volume flux)
    d = np.abs(r.added_mass[:, 0, 0] - a0)
    assert d[0] < d[1] and d[0] < 1e-6 * a0
    with pytest.raises(ValueError, match="net volume flux"):
        solve({"mesh": m, "modes": ["heave"]}, [0.0], depth=3.0, options={"diffraction": False})


def test_limits_approached_infinite_depth():
    m = hemi(12)
    w_small = np.sqrt(np.array([1e-4, 1e-3]) * G)
    r = solve({"mesh": m, "modes": ["surge", "heave"]}, [0.0, math.inf, *w_small], options={"diffraction": False})
    a0 = r.added_mass_zero_frequency
    # the limit of the theory approached: the heave A(K) - A(0) = O(Ka ln Ka) in infinite depth,
    # Ka |ln Ka| = 6.9e-3 at Ka = 1e-3 (4.9e-3 observed, 7e-4 at 1e-4)
    d = np.abs(r.added_mass - a0[None]).max(axis=(1, 2)) / peak(a0)
    assert d[0] < d[1] < 1e-2


def test_finite_depth_tends_to_infinite_depth():
    m = hemi(10)
    w = np.sqrt(np.array([0.5, 1.0]) * G)
    ri = solve({"mesh": m}, w, headings=[0.0, 30.0])
    rf = solve({"mesh": m}, w, depth=400.0, headings=[0.0, 30.0])
    # the depth enters as e^{-2kh} <= e^{-400}: nothing of it is left; the finite-depth Green
    # function evaluated against the infinite-depth one and the float32 storage leave 1.2e-7
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        a, b = view(ri, q), view(rf, q)
        assert peak(a - b) <= 1e-6 * peak(a), q


# ------------------------------------------------------------- test-plan B identities (finite depth)


# the finite-depth Green function in the identities that test_low_order checks in infinite depth;
# the hemisphere: the discrete A and B of shapes with sharp edges (the cylinder) are symmetric only
# to about 4e-4 at practical meshes, at any depth

DEPTH = 2.5
HEADS = [0.0, 45.0, 120.0]


@pytest.fixture(scope="module")
def fd_results():
    w = np.sqrt(np.array([0.4, 1.0, 1.6]) * G)
    return solve({"mesh": hemi(12)}, w, depth=DEPTH, headings=HEADS)


def test_symmetry_and_positive_damping_finite_depth(fd_results):
    # Green's reciprocity and the radiated energy (exact) with the finite-depth Green function: the
    # discretization and float32 leave 6.5e-7
    A, B = fd_results.added_mass, fd_results.radiation_damping
    assert peak(A - A.transpose(0, 2, 1)) <= 1e-5 * peak(A)
    assert peak(B - B.transpose(0, 2, 1)) <= 1e-5 * peak(B)
    assert np.all(np.diagonal(B, axis1=1, axis2=2) >= -1e-6 * peak(B))


def test_haskind_equals_pressure_excitation_finite_depth(fd_results):
    # the Haskind relation (exact): the two forces from different unknowns differ by the O(h^2)
    # discretization, 9.6e-4 at 12 panels a quadrant edge
    X, Hk = fd_results.excitation.diffraction, fd_results.excitation.haskind
    assert peak(X - Hk) <= 1e-3 * peak(X)


def test_diffraction_only_equals_full_solve(fd_results):
    # input rule: radiation off leaves the diffraction solve and the fields as they are (the same
    # equations): bit-identical
    w = fd_results.omegas
    fp = [[3.0, 0.5, 0.0], [-2.0, 4.0, 0.0]]
    full = solve({"mesh": hemi(12)}, w, depth=DEPTH, headings=HEADS, field_points=fp)
    dif = solve({"mesh": hemi(12)}, w, depth=DEPTH, headings=HEADS, field_points=fp, options={"radiation": False})
    for q in ("excitation_diffraction", "excitation_froude_krylov", "excitation_scattering"):
        assert np.array_equal(view(full, q), view(dif, q)), q
    for q in ("pressure_incident", "pressure_scattering", "pressure_diffraction"):
        assert np.array_equal(full.quantity("field_points_" + q), dif.quantity("field_points_" + q)), q
    assert dif.added_mass is None


def total_scattered_difference(n):
    """The exciting force of the total and the scattered formulation on n panels a quadrant edge in
    finite depth: their largest difference relative to the peak; the added mass of both."""
    w = np.sqrt(np.array([0.4, 1.0, 1.6]) * G)
    tot = solve({"mesh": hemi(n)}, w, depth=DEPTH, headings=HEADS, options={"diffraction_formulation": "total"})
    sca = solve({"mesh": hemi(n)}, w, depth=DEPTH, headings=HEADS)
    d = peak(tot.excitation.diffraction - sca.excitation.diffraction) / peak(sca.excitation.diffraction)
    return d, tot.added_mass, sca.added_mass


def test_total_and_scattered_agree_finite_depth():
    # one continuous problem (exact): the two formulations discretize different unknowns and differ
    # by O(h^2), 1.5e-3 at 16 panels a quadrant edge (6.1e-3 at 8); the radiation the same solve
    d, a_tot, a_sca = total_scattered_difference(16)
    assert d <= 2e-3, d
    np.testing.assert_array_equal(a_tot, a_sca)


def test_incident_elevation_finite_depth():
    # the incident wave of the finite-depth dispersion relation at a field point (analytic); the
    # float32 storage of the fields leaves 2.6e-8
    w = np.array([0.8, 2.0])
    r = solve({"mesh": graded_cylinder(6)}, w, depth=2.5, headings=[30.0], field_points=[[3.0, 1.0, 0.0]])
    k = np.array([hb.wavenumber(x, depth=2.5, gravity=G) for x in w])
    phase = -k * (3.0 * math.cos(math.radians(30)) + 1.0 * math.sin(math.radians(30)))
    np.testing.assert_allclose(r.field_points.pressure_incident[:, 0, 0], np.exp(1j * phase), rtol=1e-6)


def test_sector_solve_equals_full_finite_depth():
    # the quarter mesh with the xy symmetry and the full mesh of the same panels: the same discrete
    # problem (exact); the float32 storage and the sums of the finite-depth Green function over the
    # mirror images in another order leave 2e-7 (1e-5 the finite-depth bound of test-plan B)
    w = np.sqrt(np.array([0.7]) * G)
    q = solve({"mesh": graded_cylinder(8)}, w, depth=2.5, headings=[20.0])
    full = surface(cylinder_panels(1.0, 1.0, n_phi=32, n_z=8, n_r=4, quarter=False, cosine=True))
    f = solve({"mesh": full}, w, depth=2.5, headings=[20.0])
    for name in ("added_mass", "radiation_damping", "excitation_diffraction"):
        a, b = view(q, name), view(f, name)
        assert peak(a - b) <= 1e-5 * peak(a), name


@pytest.mark.parametrize("threads", [(2, 1), (3, 1), (1, 3), (2, 2)])
def test_threads_byte_identical_with_limits(threads):
    # the results do not depend on the thread counts, also at the limits: bit-identical
    w = [0.0, math.inf, 0.5, 1.0, 1.5, 2.0]
    m = {"mesh": graded_cylinder(6), "modes": ["surge", "pitch"]}
    a = solve(m, w, depth=2.5, options={"diffraction": False})
    b = solve(m, w, depth=2.5, options={"diffraction": False, "frequency_threads": threads[0],
                                        "inner_threads": threads[1]})  # fmt: skip
    assert np.array_equal(a.added_mass, b.added_mass)
    for key in ("zero", "infinite"):
        name = f"added_mass_{key}_frequency"
        assert np.array_equal(a.quantity(name), b.quantity(name))


def test_large_frequencies_tend_to_the_infinite_frequency_limit():
    # the limit of the theory approached: A(w) -> A(inf) as w grows (a submerged closed box: no
    # irregular frequencies), the free-surface effect decaying with K = w^2 / g: 12 % at w = 2, 1.4 %
    # at 8, 0.4 % at 16 rad/s; beyond, the mesh resolves no shorter wave (0.5 % at 32)
    sub = {"mesh": surface(box_panels(1.0, 0.5, 0.5, n=(8, 4, 4), quarter=True, lid=True), symmetry="xz_and_yz"),
           "position": [0, 0, -1.0]}  # fmt: skip
    ws = [2.0, 4.0, 8.0, 16.0]
    with pytest.warns(hb.HydroBEMWarning, match="larger than a sixth of the shortest wavelength"):
        r = model(sub, ws + [math.inf], (), options={"diffraction": False}).solve()  # the mesh is coarse at 16
    a_inf = r.added_mass_infinite_frequency
    err = [peak(r.added_mass[f] - a_inf) / peak(a_inf) for f in range(len(ws))]
    assert all(e1 < e0 for e0, e1 in zip(err, err[1:], strict=False)) and err[-1] <= 1e-2, err
