"""Regression baselines (tests/data/regression): every output of the problems of
regression_problems.py equals the stored HydroBEM result to 1e-5 of the largest value of
its entry over the first axis (the frequencies, or the points), at least 1e-3 of the peak of
the array: a small coefficient is held to its own size (single-precision
outputs; a change of method moves results far more), plus ATOL in SI units for an entry whose
exact value is zero and whose computed value is round-off (the waterplane centroid of a
symmetric body, about 1e-16: its own size is its round-off; a value above 1e-9 is still held to
RTOL). A failure after an intended change: regenerate with
scripts/gen_regression_baselines.py and give the reason in the commit.

The baselines are the results of the machine that wrote them. Another processor sums in another
order (its linear-algebra kernels and their cache blocking), and cancellation in the second-order
outputs and the body-point velocities amplifies that round-off to about 6e-4 of an array's peak (a
continuous-integration runner). There (CI=true, set by the runner) each entry is held to
OTHER_MACHINE of the peak of its array instead."""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from problems import threads_per_test
from regression_problems import PROBLEMS, outputs, same_outputs, solve

DATA = Path(__file__).resolve().parents[1] / "data" / "regression"
RTOL = 1e-5
ATOL = 1e-14
OTHER_MACHINE = 2e-3 if os.environ.get("CI") == "true" else None


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_regression_baseline(name):
    ref = np.load(DATA / f"{name}.npz")
    out = solve(name)
    assert sorted(out) == sorted(ref.files), "outputs added or removed"
    bad = []
    for key in ref.files:
        a, b = np.asarray(out[key]), ref[key]
        assert a.shape == b.shape, (key, a.shape, b.shape)
        assert a.dtype == b.dtype, (key, a.dtype, b.dtype)  # the precision of each output
        if b.size == 0:
            continue
        # NaN where the quantity is not defined (a centre of gravity not given): at the same places
        nan = np.isnan(b) if b.dtype.kind in "fc" else np.zeros(b.shape, bool)
        if not np.array_equal(nan, np.isnan(a) if a.dtype.kind in "fc" else nan):
            bad.append(f"{key}: NaN elsewhere")
            continue
        if nan.all():
            continue
        a, b = np.where(nan, 0, a), np.where(nan, 0, b)
        peak = np.abs(b).max()
        # each entry against its own largest value over the first axis, at least 1e-3 of the peak
        scale = np.maximum(np.abs(b).max(axis=0, keepdims=True), 1e-3 * peak) if b.ndim > 1 else peak
        tol = RTOL * scale
        if OTHER_MACHINE is not None:
            scale, tol = peak, OTHER_MACHINE * peak
        d = np.abs(a.astype(np.complex128) - b.astype(np.complex128))
        worst = np.unravel_index(np.argmax(d / (scale + 1e-300)), d.shape)
        if not np.all(d <= tol + ATOL):
            of = np.broadcast_to(scale, d.shape)[worst]
            bad.append(f"{key}: {d[worst]:.3g} at {tuple(int(i) for i in worst)} of {of:.3g}")
    assert not bad, "; ".join(bad)


# the problems whose paths the thread tests of the low and higher order do not cover: a channel,
# the drift of each formulation, tanks; the second order: the quadratic QTFs of both methods with
# Newman's approximation, the potential QTF by a wall and of generalized modes (the open sea with a
# quadrature zone: test_potential_qtf)
THREADED = ("lo_box_channel", "lo_cylinder_drift", "lo_tank_haskind_pressure_drift",
            "lo_cylinder_control_surface_bidirectional", "ho_acv_pressure_drift", "lo_cylinder_quadratic_qtf_newman",
            "lo_cylinder_potential_qtf_wall", "lo_flexible_cylinder_second_order")


# the second-order problems on one thread take seconds (the potential QTF by a wall 19 s here)
SLOW = ("lo_cylinder_quadratic_qtf_newman", "lo_cylinder_potential_qtf_wall", "lo_flexible_cylinder_second_order")


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n for n in THREADED])
def test_threads_byte_identical(name):
    def run(f, i):
        m = PROBLEMS[name]()
        m.options.update(frequency_threads=f, inner_threads=i)
        with warnings.catch_warnings():  # the note of the pressure drift, as in solve
            warnings.filterwarnings("ignore", message=".*integrated over the wetted surface only")
            return outputs(m.solve())

    # one thread, two of each, and the threads of a test (when more) at once over the frequencies
    a = run(1, 1)
    same_outputs(a, run(2, 2))
    if threads_per_test() > 2:
        same_outputs(a, run(threads_per_test(), 1))
