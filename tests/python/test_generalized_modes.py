"""Generalized modes, low order (theory §10.1).

A generalized mode with the displacement of a rigid mode (heave: w = 1;
pitch about the origin: u = z, w = -x) reproduces that mode's radiation,
exciting force and, in its own row, the hydrostatic restoring (the restoring of theory §10.1 is
exact for these divergence-free fields); symmetry sectors equal the full mesh for a bending
mode; a mode held by a large stiffness leaves the rigid motions of the body
unchanged; a mode on a panel selection acts only there; the inputs are
checked. The rigid displacements give the rigid mode's normal velocity on every panel, so the
same right-hand side of the same equations: their results agree to the float32 level (0 observed,
1e-6 the bound), the restoring to the double round-off. The second-order loads of generalized modes
are tested with each load (test_drift, test_quadratic_qtf, test_potential_qtf); here their rules. Meshes are procedural
(tests/python/meshes.py)."""

import numpy as np
import pytest

from bodies import free_cylinder, surface
from meshes import box_panels
from problems import close, model, peak, solve, view

W = np.array([0.6, 1.0, 1.6])
HEAVE = {"name": "g-heave", "pieces": [{"w": [[1.0, 0, 0, 0]]}], "parity": "heave"}
PITCH = {"name": "g-pitch", "pieces": [{"u": [[1.0, 0, 0, 1]], "w": [[-1.0, 1, 0, 0]]}], "parity": "pitch"}
# vertical bending w = P2(2x/L) of a barge of length L = 4 (Legendre)
BEND = {"name": "bend", "pieces": [{"w": [[1.5 * 0.25, 2, 0, 0], [-0.5, 0, 0, 0]]}], "parity": "heave"}


def barge(quarter=True, **kw):
    n = (8, 4, 2) if quarter else (16, 8, 2)  # the same panels
    mesh = surface(box_panels(2.0, 1.0, 0.5, n=n, quarter=quarter), symmetry="xz_and_yz" if quarter else "none")
    return {"mesh": mesh, "mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.1],
            "radii_of_gyration": [0.6, 1.2, 1.2], **kw}


@pytest.mark.parametrize("mode, rigid", [(HEAVE, 2), (PITCH, 4)])
def test_rigid_displacement_reproduces_the_rigid_mode(mode, rigid):
    # exact identity: the radiation, the exciting forces (also by Haskind) and the restoring row of
    # the rigid mode; the restoring in double precision (6e-16 observed)
    r = solve(barge(generalized_modes=[mode]), W, headings=[0.0, 30.0])
    g = 6  # the generalized coordinate (zero-based)
    close(r.added_mass[:, g, :], r.added_mass[:, rigid, :])
    close(r.radiation_damping[:, g, :], r.radiation_damping[:, rigid, :])
    close(r.excitation.diffraction[..., g], r.excitation.diffraction[..., rigid])
    close(r.excitation.haskind[..., g], r.excitation.haskind[..., rigid])
    c = r.mode_restoring
    crg = np.asarray(r.hydrostatics.stiffness_rho_g[0])
    np.testing.assert_allclose(c[g, g], crg[rigid, rigid], rtol=1e-12)
    for j in (2, 3, 4):
        np.testing.assert_allclose(c[g, j], crg[rigid, j], rtol=1e-12, atol=1e-12 * abs(crg[rigid, rigid]))
        assert c[j, g] == 0.0  # the rigid rows keep their own restoring


def test_sectors_equal_the_full_mesh():
    # the quarter barge with the xy symmetry and the full barge of the same panels with a bending
    # mode: the same discrete problem (exact); the float32 storage and the sums over the sectors in
    # another order leave 8.7e-7 in the added mass (the general low-order bound 1e-6 of test-plan B
    # is too close to it: 2e-6), the restoring 1e-15 (double)
    res = [solve(barge(quarter=q, generalized_modes=[BEND]), W, headings=[30.0]) for q in (True, False)]
    for q in ("added_mass", "radiation_damping", "excitation_diffraction", "excitation_haskind"):
        close(view(res[0], q), view(res[1], q), 2e-6)
    np.testing.assert_allclose(res[0].mode_restoring, res[1].mode_restoring, rtol=0,
                               atol=1e-12 * np.abs(res[1].mode_restoring).max())  # fmt: skip


def test_stiff_mode_leaves_the_rigid_motions_unchanged():
    # a mode held by a stiffness of 1e12, eight orders above the restoring of the barge (1.5e4): its
    # motion at most 1e-8 of the rigid ones (1.1e-9 observed), the rigid motions change by that
    # (0 observed in float32; 1e-6 the bound)
    k = np.zeros((7, 7))
    k[6, 6] = 1e12
    m = np.zeros((7, 7))
    m[6, 6] = 100.0
    opts = {"motion": True}
    r = solve(barge(generalized_modes=[BEND], mode_stiffness=k, mode_mass=m), W, headings=[30.0], options=opts)
    r0 = solve(barge(), W, headings=[30.0], options=opts)
    close(r.motion[..., :6], r0.motion)
    assert peak(r.motion[..., 6]) <= 1e-8 * peak(r0.motion)


def test_piece_on_a_selection():
    # heave of the bottom panels only: the generalized normal velocity vanishes on the sides,
    # and the bottom elements are the first 8 x 4 of the quarter box (meshes.box_panels); a box has
    # n_z = 0 on the sides, so the same right-hand side as heave on every panel (0 observed)
    body = barge()
    piece = {"w": [[1.0, 0, 0, 0]], "mesh": (body["mesh"], (1, 32))}
    r = solve({**body, "generalized_modes": [{"name": "bottom", "pieces": [piece], "parity": "heave"}]}, W,
              headings=[0.0])  # fmt: skip
    full = solve(barge(generalized_modes=(HEAVE,)), W, headings=[0.0])
    close(r.added_mass, full.added_mass)
    close(r.excitation.diffraction, full.excitation.diffraction)


def test_generalized_inputs_are_checked():
    # input rules
    def counts(body, headings=(), **kw):
        return model(body, W, headings, **kw).counts()

    with pytest.raises(ValueError, match="parity"):
        counts(barge(generalized_modes=[{**BEND, "name": "odd", "parity": "surge"}]))
    k = np.zeros((7, 7))
    k[2, 2] = 1.0
    with pytest.raises(ValueError, match="rigid block"):
        counts(barge(generalized_modes=[BEND], mode_stiffness=k))
    with pytest.raises(ValueError, match=r"mode_mass must be 7 x 7 \(6 \+ G, G = 1 generalized modes\), not 6 x 6"):
        counts(barge(generalized_modes=[BEND], mode_mass=np.zeros((6, 6))))
    with pytest.raises(ValueError, match="powers"):
        counts(barge(generalized_modes=[{"name": "g", "pieces": [{"w": [[1.0, 9, 0, 0]]}]}]))
    with pytest.raises(ValueError, match="duplicate"):
        counts(barge(generalized_modes=[BEND, BEND]))


def test_the_rules_of_the_second_order_loads():
    # the pressure drift and the QTFs of generalized modes (theory §12.15): a mode that tears the
    # wetted surface open (heave of the bottom alone: the sides stay) has no exact surface and is
    # refused, a kink (the bottom and the sides continuous at their edges) is not; the higher
    # order is refused
    body = barge()
    drift = {"motion": True, "drift_pressure_integration": True}
    torn = {"name": "bottom", "pieces": [{"w": [[1.0, 0, 0, 0]], "mesh": (body["mesh"], (1, 32))}], "parity": "heave"}
    kink = {"name": "kink", "parity": "heave",
            "pieces": [{"w": [[1.0, 0, 0, 0]], "mesh": (body["mesh"], (1, 32))},
                       {"w": [[2.0, 0, 0, 0], [2.0, 0, 0, 1]], "mesh": (body["mesh"], (33, 56))}]}  # fmt: skip
    with pytest.raises(ValueError, match="jumps between its pieces"):
        model({**body, "generalized_modes": [torn]}, W, [0.0], outputs=drift).counts()
    with pytest.raises(ValueError, match="jumps between its pieces"):
        model({**body, "generalized_modes": [torn]}, W, [0.0],
              outputs={"motion": True, "qtf_potential_indirect": True}).counts()  # fmt: skip
    model({**body, "generalized_modes": [torn]}, W, [0.0], outputs={"motion": True}).counts()  # first order: allowed
    model({**body, "generalized_modes": [kink]}, W, [0.0], outputs=drift).counts()

    with pytest.raises(ValueError, match="low order only"):
        model(free_cylinder(generalized_modes=[HEAVE]), W, [0.0], outputs=drift,
              options={"higher_order": True}).counts()  # fmt: skip


@pytest.mark.parametrize("mode, rigid", [(HEAVE, 2), (PITCH, 4)])
def test_higher_order_rigid_displacement(mode, rigid):
    # exact identity as in the low order: the same right-hand side (0 observed); the restoring of the
    # mode by the patch quadrature against the rigid closed form, 1.9e-10 for pitch (double)

    r = solve(free_cylinder(generalized_modes=[mode]), W, headings=[0.0, 30.0], options={"higher_order": True})
    g = 6
    close(r.added_mass[:, g, :], r.added_mass[:, rigid, :])
    close(r.radiation_damping[:, g, :], r.radiation_damping[:, rigid, :])
    close(r.excitation.diffraction[..., g], r.excitation.diffraction[..., rigid])
    crg = np.asarray(r.hydrostatics.stiffness_rho_g[0])
    np.testing.assert_allclose(r.mode_restoring[g, g], crg[rigid, rigid], rtol=1e-8)
