"""Several bodies in one solve against semi-analytic references of the interaction: two submerged
spheres in deep water (the multipole method of Wu 1995, scripts/gen_sphere_array_reference.py,
tests/data/two_spheres.txt), two and three truncated cylinders in finite depth (eigenfunction
matching with the multiple scattering of Kagemoto & Yue 1986, scripts/gen_cylinder_array_reference.py,
tests/data/cylinder_array.txt). Each body is placed by its position, its mesh and rotation centre in
body axes; the higher-order method on exact geometry, so that the bound is the reference's
truncation and the patch discretization."""

import math
from pathlib import Path

import numpy as np

from bodies import ho_sphere
from problems import RHO, G, analytic, solve

DATA = Path(__file__).resolve().parents[1] / "data"


def test_two_submerged_spheres_deep_water():
    """Two spheres a = 1 at (-2, 0, -1.5) and (2, 0, -1.5), waves at 45 deg, nu a = 0.5, 1.4, 2.5:
    on sphere 2 with both spheres moving in surge, sway and heave, A21 + A22 and B21 + B22, and its
    exciting force |E2|, against the multipole reference (converged in its truncation to 1e-6 of
    the largest value). The bound is the discretization of 4 x 8 spans a quadrant: within 3e-5 of
    the largest value of each quantity for A and B (1.2e-5 found), 7e-5 for E (3.4e-5 found; 6 x 12
    spans: 2e-6)."""
    ref = np.loadtxt(DATA / "two_spheres.txt")
    nu = ref[:, 0]
    w = np.sqrt(G * nu)
    bodies = [{"name": name, "mesh": ho_sphere(), "position": [x, 0.0, -1.5], "rotation_center": [0.0, 0.0, 0.0],
               "modes": ["surge", "sway", "heave"], "interior_surface": "none"}
              for name, x in (("s1", -2.0), ("s2", 2.0))]  # fmt: skip
    r = solve(bodies, w, [45.0], options={"higher_order": True})
    v = RHO * 4 * math.pi / 3
    got = []
    for mode in ("surge", "sway", "heave"):
        a = sum(r.coefficient("added_mass", ("s2", mode), (b, mode)) for b in ("s1", "s2"))
        b = sum(r.coefficient("radiation_damping", ("s2", mode), (s, mode)) for s in ("s1", "s2"))
        e = np.abs(r.coefficient("excitation_diffraction", ("s2", mode))[:, 0])
        got += [a / v, b / (v * w), e / (v * w * w * np.exp(-1.5 * nu))]
    err = np.abs(np.array(got).T - ref[:, 1:]).max(axis=0) / np.abs(ref[:, 1:]).max(axis=0)
    bound = np.tile([3e-5, 3e-5, 7e-5], 3)
    assert np.all(err <= bound), err


def cylinder_array(case):
    """The rows (ka, quantities) of a case of tests/data/cylinder_array.txt."""
    rows = [line.split() for line in (DATA / "cylinder_array.txt").read_text(encoding="ascii").splitlines()
            if line and not line.startswith("#")]  # fmt: skip
    return np.array([r[1:] for r in rows if r[0] == case], dtype=np.float64)


def cylinders(centers, draft, modes):
    """Truncated cylinders a = 1 of the draft at the centres: the exact side and bottom as analytic
    patches, the nonuniform mapping (the spans clustered at the waterline and at the bottom edge,
    where the potential varies fastest), 3 x 4 spans a quadrant on the side, 3 x 2 on the bottom."""
    mesh = analytic("circular_cylinder", [(3, 4, 3, 3, 3, 4), (3, 2, 3, 3, 3, 4)], parts=["side", "bottom"],
                    mapping="nonuniform", radius=1.0, draft=draft)  # fmt: skip
    return [{"name": f"c{i + 1}", "mesh": mesh, "position": [x, y, 0.0], "rotation_center": [0.0, 0.0, 0.0],
             "modes": modes, "interior_surface": "none"} for i, (x, y) in enumerate(centers)]  # fmt: skip


def test_two_truncated_cylinders_finite_depth():
    """Two cylinders a = 1, draft 2, in the depth 4 at (-1.3, 0) and (1.3, 0) (Kagemoto & Yue 1986,
    geometry 2), head seas at ka = 0.5, 1.0, 1.6 (below the first irregular frequency, ka = 2.4):
    with both moving together, the surge and heave added mass and damping summed over the two, and
    the surge and heave exciting forces summed, against eigenfunction matching with the multiple
    scattering (extrapolated in its truncation, the two extrapolations agree to 2e-6). The bound is
    the discretization: 1e-4 of the largest value of each quantity (4.9e-5 found, heave A; 4 x 6
    spans on the side: 4.2e-5; the uniform mapping of 4 x 6 spans: 4.3e-3)."""
    ref = cylinder_array("two")
    ka = ref[:, 0]
    w = np.sqrt(G * ka * np.tanh(4.0 * ka))
    r = solve(cylinders([(-1.3, 0.0), (1.3, 0.0)], 2.0, ["surge", "heave"]), w, [0.0], depth=4.0,
              options={"higher_order": True})  # fmt: skip
    names = ("c1", "c2")
    got = []
    for mode in ("surge", "heave"):
        got.append(sum(r.coefficient("added_mass", (p, mode), (q, mode)) for p in names for q in names) / RHO)
        got.append(sum(r.coefficient("radiation_damping", (p, mode), (q, mode)) for p in names for q in names)
                   / (RHO * w))  # fmt: skip
    for mode in ("surge", "heave"):
        got.append(np.abs(sum(r.coefficient("excitation_diffraction", (p, mode))[:, 0] for p in names)) / (RHO * G))
    err = np.abs(np.array(got).T - ref[:, 1:]).max(axis=0) / np.abs(ref[:, 1:]).max(axis=0)
    assert np.all(err <= 1e-4), err


def test_three_truncated_cylinders_finite_depth():
    """Three cylinders a = 1, draft 3.65, in the depth 40 at the corners of the equilateral triangle
    of side 5 (Mavrakos 1991), head seas at ka = 0.5, 1.0, 1.6: cylinder 1 at (5/sqrt 3, 0) moving
    alone in surge, its surge and pitch-surge added mass and damping (pitch about its own axis in
    z = 0), the surge force on cylinder 2 (the interaction), and the surge exciting forces on
    cylinders 1 and 2, against eigenfunction matching with the multiple scattering (its two
    extrapolations agree to 4e-5). The bound is the discretization: 6e-4 of the largest value of
    each quantity (3.3e-4 found, B51; 4 x 6 spans on the side: 1.6e-5, the reference's
    truncation)."""
    ref = cylinder_array("three")
    ka = ref[:, 0]
    w = np.sqrt(G * ka * np.tanh(40.0 * ka))
    s = math.sqrt(3.0)
    centers = [(5.0 / s, 0.0), (-2.5 / s, 2.5), (-2.5 / s, -2.5)]
    r = solve(cylinders(centers, 3.65, ["surge", "pitch"]), w, [0.0], depth=40.0, options={"higher_order": True})
    got = []
    for row in (("c1", "surge"), ("c1", "pitch"), ("c2", "surge")):
        got.append(r.coefficient("added_mass", row, ("c1", "surge")) / RHO)
        got.append(r.coefficient("radiation_damping", row, ("c1", "surge")) / (RHO * w))
    for p in ("c1", "c2"):
        got.append(np.abs(r.coefficient("excitation_diffraction", (p, "surge"))[:, 0]) / (RHO * G))
    err = np.abs(np.array(got).T - ref[:, 1:]).max(axis=0) / np.abs(ref[:, 1:]).max(axis=0)
    assert np.all(err <= 6e-4), err
