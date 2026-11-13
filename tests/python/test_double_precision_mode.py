"""Diagnostic double-precision mode.

The production path (float32 Gauss tiers of the Rankine integrals, direct
wave-kernel tiles, float32 matrices and LU) must agree with the diagnostic
double-precision solve to the accuracy budget of theory §6 and §13.1. The mode
is selected when the library is loaded, so it runs in a subprocess."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SCRIPT = """
import json, sys
import numpy as np
sys.path[:0] = {paths!r}
from bodies import surface
from meshes import cylinder_panels
from problems import model
mesh = surface(cylinder_panels(1.0, 1.0, n_phi=8, n_z=6, n_r=4, quarter=True, cosine=True), symmetry="xz_and_yz")
m = model({{"mesh": mesh}}, np.sqrt(np.array([0.5, 1.2, 2.0]) * 9.80665), [0.0, 45.0], field_points=[(3.0, 0.5, 0.0)],
          environment={{"density": 1000.0, "gravity": 9.80665}}, options={{"field_pressure": True}})
r = m.solve()
X = np.asarray(r.excitation.diffraction)
F = np.asarray(r.field_points.pressure_diffraction)
print(json.dumps({{"A": np.asarray(r.added_mass).tolist(), "B": np.asarray(r.radiation_damping).tolist(),
                  "Xr": X.real.tolist(), "Xi": X.imag.tolist(), "Fr": F.real.tolist(), "Fi": F.imag.tolist()}}))
"""


def _run(diag):
    env = dict(os.environ)
    env.pop("HB_DIAG_DOUBLE", None)
    if diag:
        env["HB_DIAG_DOUBLE"] = "1"
    paths = [str(HERE.parents[1] / "python"), str(HERE)]
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(paths=paths)], env=env, capture_output=True, text=True, check=True
    )
    return {k: np.array(v) for k, v in json.loads(out.stdout.strip().splitlines()[-1]).items()}


def test_production_equals_double_precision():
    prod, diag = _run(False), _run(True)
    for key in ("A", "B", "Xr", "Xi", "Fr", "Fi"):
        ref = np.abs(diag[key]).max()
        assert np.abs(prod[key] - diag[key]).max() <= 1e-5 * ref, key
    # the modes really differ (otherwise the test would be vacuous)
    assert np.abs(prod["A"] - diag["A"]).max() > 0.0
