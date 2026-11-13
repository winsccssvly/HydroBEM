"""The memory estimate of a solve.

Model.memory_estimate counts the arrays of a solve from the sizes of the problem: the resident peak is
the statics plus frequency_threads times the workspace of one frequency plus the outputs; the
workspace grows with the inner threads by the characters solved at once and the parts of the packing
buffers of the matrix products they write; the commit charge adds the rest of those buffers, which
every thread running a product keeps; a hydrostatics problem needs none; the potential QTF's
sub-problems raise the peak beyond the model's own solve; a solve whose peak exceeds the available memory
is refused before it starts. These are exact identities of the count. The hooks of the environment
(the test hook HB_AVAILABLE_MEMORY of the memory check, the validation hooks HB_VALIDATE_<name>)
are refused when their value does not parse, and one that changes the numbers is a warning.
How close the count comes to the measured peaks of a solve depends on the machine (the processor,
its memory system, the thread count): it is calibrated there. Five cases are measured here by
the process's peak working set (Windows): the second order of the indirect method of the
potential QTF, whose steps between the sub-problems hold the peak, at f = 1 and 3; the field
outputs of both methods; and the direct method of the potential QTF. Meshes are procedural (tests/python/meshes.py)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import hydrobem as hb

from bodies import surface
from meshes import cylinder_panels
from problems import flat_surface, model


def _problem(n=32, quarter=True, f=1, i=1, nf=1, fixed=False, **kw):
    pan = cylinder_panels(1.0, 1.0, n_phi=n, n_z=n // 4, n_r=n // 4, quarter=quarter)
    body = {"mesh": surface(pan, symmetry="xz_and_yz" if quarter else "none"), **({"modes": []} if fixed else {})}
    return model(body, [1.5 + 0.1 * k for k in range(nf)], [0.0, 45.0],
                 options={"frequency_threads": f, "inner_threads": i, **kw})  # fmt: skip


def test_the_parts_of_the_estimate():
    e = _problem().memory_estimate()
    assert set(vars(e)) == {"statics", "per_frequency", "outputs", "peak", "commit"}
    assert e.peak == e.statics + e.per_frequency + e.outputs > 0
    assert e.commit >= e.peak
    # frequency threads multiply the workspace; inner threads add the characters solved at once
    two = _problem(f=2).memory_estimate()
    assert two.peak == e.statics + 2 * e.per_frequency + e.outputs
    assert _problem(i=4).memory_estimate().per_frequency > e.per_frequency
    # more frequencies: more outputs, the same workspace
    many = _problem(nf=5).memory_estimate()
    assert many.outputs > e.outputs and many.per_frequency == e.per_frequency
    # hydrostatics only: no solve
    p = _problem(radiation=False, diffraction=False)
    p.environment.omegas, p.environment.headings = [], []
    assert p.memory_estimate() == hb.MemoryEstimate(statics=0, per_frequency=0, outputs=0, peak=0, commit=0)


def test_the_sub_problems_of_the_potential_qtf():
    # the potential QTF's two sub-problems run after the model's solve with its results held, their
    # field points the free surface: the peak and the commit charge the larger of the model's own
    # and theirs (here theirs: thousands of field points), the parts those of the model's own
    # solve; the direct method adds its extra columns to the second
    def estimate(methods, **fs):
        p = _problem(n=16, depth=3.0, **{f"qtf_potential_{x}": True for x in methods}, fixed=True, radiation=False)
        p.second_order.update(**flat_surface("free_surface", {**{"inner_radius": 3.0, "panel_size": 0.3, **fs}}))
        return p.memory_estimate()

    alone = _problem(n=16, depth=3.0, fixed=True, radiation=False).memory_estimate()
    e = estimate(["indirect"])
    assert (e.statics, e.per_frequency, e.outputs) == (alone.statics, alone.per_frequency, alone.outputs)
    assert e.peak > e.statics + e.per_frequency + e.outputs and e.commit >= e.peak
    both = estimate(["indirect", "direct"])
    assert both.peak > e.peak
    assert estimate(["indirect"], panel_size=0.15).peak > e.peak  # more points


_PEAK = """
import ctypes, ctypes.wintypes as wt, sys
sys.path[:0] = {paths!r}
import numpy as np
from meshes import cylinder_panels
from bodies import spline_cylinder, surface
from problems import flat_surface, model

class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS of the Windows API
    _fields_ = [("cb", wt.DWORD), ("faults", wt.DWORD), ("peak", ctypes.c_size_t), ("ws", ctypes.c_size_t)] + [
        (f"x{{k}}", ctypes.c_size_t) for k in range(6)]

psapi = ctypes.WinDLL("psapi")
psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
ctypes.windll.kernel32.GetCurrentProcess.restype = wt.HANDLE

def counters():
    c = Counters(cb=ctypes.sizeof(Counters))
    psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c

method, points = {method!r}, {points!r}
if points == 0:  # the potential QTF of a fixed cylinder in a flat free surface
    pan = cylinder_panels(1.0, 1.0, n_phi=8, n_z=2, n_r=2, quarter=True)
    m = model({{"mesh": surface(pan, symmetry="xz_and_yz"), "modes": []}}, {omegas!r}, [0.0],
              environment={{"depth": 3.0}}, outputs={{"qtf_potential_" + method: True}},
              options={{"frequency_threads": {f}, "inner_threads": 1}})
    m.second_order.update(**flat_surface("free_surface", {{"inner_radius": 3.0, "panel_size": {size}}}))
else:  # the field pressure and velocity of a floating cylinder at the points of a circle, four headings
    t = np.linspace(0.0, 2 * np.pi, points, endpoint=False)
    pts = np.stack([3 * np.cos(t), 3 * np.sin(t), np.full(points, -0.3)], axis=1)
    mesh = (spline_cylinder(n=4) if method == "higher_order" else
            surface(cylinder_panels(1.0, 0.5, n_phi=8, n_z=2, n_r=2, quarter=True), symmetry="xz_and_yz"))
    m = model({{"mesh": mesh}}, {omegas!r}, [0.0, 30.0, 60.0, 90.0], field_points=pts,
              options={{"frequency_threads": {f}, "inner_threads": 1, "higher_order": method == "higher_order",
                        "field_pressure": True, "field_velocity": True}})
e = m.memory_estimate()
before = counters()
m.solve()
print(e.peak, counters().peak - before.ws, before.peak - before.ws)
"""


def _measured_peak(method, f, size, omegas, points=0):
    """The script above in a process of its own (the potential QTF of a method, or with points the field
    outputs of the low or higher order): the estimated peak, the process's peak working set above its
    working set before the solve, and the peak before the solve above it."""
    here = Path(__file__).resolve().parent
    script = _PEAK.format(paths=[str(here.parents[1] / "python"), str(here)], method=method, f=f, size=size,
                          omegas=omegas, points=points)  # fmt: skip
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    return [int(x) for x in out.stdout.split()]


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "win32", reason="the counters of a Windows process")
@pytest.mark.parametrize("f", [1, 3])
def test_the_estimate_holds_the_measured_peak_of_the_second_order(f):
    # a coarse body in a fine free surface with the indirect method: the steps between the
    # sub-problems' solves are the solve's peak (the fits: the fields of both sub-problems read from
    # their results with the radiation's results still held; then the outer integral and the pairs,
    # min(f, pair frequencies) at once). The estimate counts the arrays, not the allocator's pages
    # and the fixed costs: the measured peak within 0.85-1.2 times it and 8 MiB (on the development
    # machine 1.05 for both f with the zones and field points counted; the sub-problems' solves
    # alone, the estimate before the steps were
    # counted, gave 0.57 times the measured peak)
    est, measured, earlier = _measured_peak("indirect", f, 0.15, [0.8 + 0.1 * k for k in range(6)])
    assert measured > earlier  # the solve set the process's peak
    assert 0.85 * est <= measured <= 1.2 * est + (8 << 20), (est, measured)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "win32", reason="the counters of a Windows process")
@pytest.mark.parametrize("method", ["low_order", "higher_order"])
def test_the_estimate_holds_the_measured_peak_of_field_outputs(method):
    # a coarse floating body with its field pressure and velocity at 3000 points, 20 frequencies and
    # four headings: the outputs are most of the solve's memory. The results take the outputs' arrays
    # over (no copy, also not at the Python boundary), so the peak is the outputs once and the
    # largest copy, not twice; the higher order's static rows count their vectors' headers too (one
    # row per field point and stencil point). The measured peak within 0.95-1.05 times the estimate
    # and 6 MiB, the fixed costs the estimate leaves out (on the development machine 1.06 and 1.05;
    # before, when every quantity was copied into the results and the results copied for Python, the
    # low order measured 74 MiB against 81 estimated, and the higher order 125 against 112 before
    # the rows' headers were counted)
    est, measured, earlier = _measured_peak(method, 1, 0, [0.5 + 0.05 * k for k in range(20)], points=3000)
    assert measured > earlier
    assert 0.95 * est <= measured <= 1.05 * est + (6 << 20), (est, measured)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "win32", reason="the counters of a Windows process")
def test_the_estimate_holds_the_measured_peak_of_the_direct_method():
    # the case of the indirect method with the direct one: its second sub-problem's solve, whose
    # field outputs (the free surface) dominate, holds the peak, with the zones and the field points
    # of the free surface held under it: the measured peak within 0.95-1.05 times the estimate and
    # 6 MiB (on the development machine 1.10, inside the bound by the 6 MiB term: 4 MiB of fixed
    # costs; before, 61 MiB estimated against 80 measured with the outputs counted twice, then 43
    # against 49 without the zones and points)
    est, measured, earlier = _measured_peak("direct", 1, 0.15, [0.8 + 0.1 * k for k in range(6)])
    assert measured > earlier
    assert 0.95 * est <= measured <= 1.05 * est + (6 << 20), (est, measured)


def test_a_solve_beyond_the_available_memory_is_refused(monkeypatch):
    # the solve compares the resident peak of the estimate with the available physical memory
    # before it starts (HB_AVAILABLE_MEMORY, bytes, replaces the reading): refused with a value
    # error giving both and the remedies, solved with options.memory_check = false, and with enough
    # memory; the refusal leaves the model as before the solve
    p = _problem(n=8)
    need = p.memory_estimate().peak
    monkeypatch.setenv("HB_AVAILABLE_MEMORY", str(need - 1))
    try:
        p.solve()
    except ValueError as e:
        text = str(e)
    else:
        raise AssertionError("not refused")
    assert "available" in text and "frequency_threads" in text and "memory_check = false" in text
    assert p.state == "reset"
    p.options.memory_check = False
    assert "added_mass" in p.solve().available
    p.options.memory_check = True
    monkeypatch.setenv("HB_AVAILABLE_MEMORY", str(need))
    assert "added_mass" in p.solve().available


def test_an_available_memory_that_is_not_a_number_is_refused(monkeypatch):
    # a test hook set but not a number is refused with a value error naming it, not passed over
    p = _problem(n=8)
    monkeypatch.setenv("HB_AVAILABLE_MEMORY", "plenty")
    with pytest.raises(ValueError, match="HB_AVAILABLE_MEMORY"):
        p.solve()
    assert p.state == "reset"


_HOOKS = """
import sys, warnings
sys.path[:0] = {paths!r}
from bodies import coarse_cylinder
from problems import model
m = model({{"mesh": coarse_cylinder()}}, [1.5], [0.0])
try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = m.solve()
except ValueError as e:
    print("refused:", e)
else:
    print("codes:", sorted({{d.code for d in r.diagnostics}}), len(w))
"""


def _hooks(env):
    """The script above in a process of its own with the variables env (the validation hooks are
    read once per process): its last line."""
    here = Path(__file__).resolve().parent
    full = {k: v for k, v in os.environ.items() if not k.startswith("HB_VALIDATE_")}
    out = subprocess.run([sys.executable, "-c", _HOOKS.format(paths=[str(here.parents[1] / "python"), str(here)])],
                         env={**full, **env}, capture_output=True, text=True, check=True)  # fmt: skip
    return out.stdout.strip().splitlines()[-1]


def test_the_validation_hooks_are_a_warning_or_refused():
    # a validation hook of a known constant (here its own value): solved, with the warning of code 9
    # (an environment hook that changes the numbers); a hook of no known constant and a value that
    # does not parse: refused with a value error naming the variable
    assert _hooks({"HB_VALIDATE_OUTER_TAPER": "0.3"}) == "codes: [9] 1"
    no_constant = _hooks({"HB_VALIDATE_NOT_A_CONSTANT": "1"})
    assert no_constant.startswith("refused:") and "no validation constant NOT_A_CONSTANT" in no_constant
    not_a_number = _hooks({"HB_VALIDATE_OUTER_TAPER": "0,3"})
    assert not_a_number.startswith("refused:") and "HB_VALIDATE_OUTER_TAPER" in not_a_number
