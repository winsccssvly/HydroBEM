"""Dipole damping surfaces, low order.

The jump (i eps / omega) psi_n across a damper (theory §9.5) has two limits that
are problems of their own: eps = 0 removes the damper, and eps -> inf makes it rigid (psi_n -> 0): a damper
of a body (moving with it, in its forces) becomes a sheet of the body, a damper of the water (at
rest) a sheet held fixed in space. Between them the problem stays reciprocal (A and B symmetric)
and energy is not created (B positive semidefinite); the Haskind relation holds for a damper of a
body. Routes that are not defined are rejected. Meshes are procedural (tests/python/meshes.py)."""

import numpy as np
import pytest

from bodies import part, surface
from meshes import cylinder_panels, face_panels
from problems import RHO, G, analytic, close, model, peak, solve, view

W = np.sqrt(np.array([0.5, 1.0, 2.0]) * G)
HULL = cylinder_panels(1.0, 1.0, n_phi=6, n_z=4, n_r=3, quarter=True)
# a horizontal plate under the cylinder (quarter, upward normal)
PLATE = np.array(face_panels((0.0, 0.0, -1.5), (1.4, 0, 0), (0, 1.4, 0), 4, 4, (0, 0, 1)))
HULL_AND_PLATE = surface(np.concatenate([HULL, PLATE]), symmetry="xz_and_yz")
WETTED = part(HULL_AND_PLATE, 1, len(HULL))  # the hull elements
SHEET = part(HULL_AND_PLATE, len(HULL) + 1, len(HULL) + len(PLATE))  # the plate elements


def damped(eps, owner, **kw):
    """The hull with the plate as a damper of the body ("body": it moves with the hull and its
    pressure is in the forces) or of the water ("water": at rest): the model keywords."""
    damper = {"mesh": SHEET, "damping_factor": eps}
    if owner == "body":
        return {"bodies": {"mesh": WETTED, "dipole_dampers": [damper], **kw}}
    return {"bodies": {"mesh": WETTED, **kw}, "dampers": [damper]}


def scattered(**kw):
    return {"diffraction_formulation": "scattered", **kw}


def run(case, **kw):
    case = case if "bodies" in case else {"bodies": case}
    return solve(omegas=W, headings=[0.0, 45.0], options=scattered(**kw), **case)


@pytest.mark.parametrize("owner", ["water", "body"])
def test_zero_damping_removes_the_damper(owner):
    # eps = 0: no jump, the problem of the hull alone; 1e-6: float32, the damper rows solved with
    # the hull (9e-8 found)
    ref = run({"mesh": surface(HULL, symmetry="xz_and_yz")})
    r = run(damped(0.0, owner))
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        close(view(r, q), view(ref, q), 1e-6)


# eps -> inf: the jump mu = -(i eps / omega) psi_n stays finite only with psi_n = O(omega / eps), the
# rigid sheet's psi_n = 0 to that order: at eps = 1e5 the coefficients are within 4.4e-4 of the
# rigid sheet for the body's damper (the damping; the added mass 2.9e-6, the forces 2.1e-5) and
# 4.1e-5 for the water's, ten times less a decade higher: 1e-3
LARGE = 1e5


def test_large_damping_body_damper_is_a_body_sheet():
    ref = run({"mesh": WETTED, "dipoles": [{"mesh": SHEET}]})
    r = run(damped(LARGE, "body"))
    for q in ("added_mass", "radiation_damping", "excitation_diffraction", "excitation_haskind"):
        close(view(r, q), view(ref, q), 1e-3)


def test_large_damping_water_damper_is_a_fixed_sheet():
    # the rigid sheet held in space: a second, fixed body of thin sheets only
    hull = {"name": "hull", "mesh": surface(HULL, symmetry="xz_and_yz")}
    plate = {"name": "plate", "modes": [], "dipoles": [{"mesh": surface(PLATE, symmetry="xz_and_yz")}]}
    ref = run([hull, plate])
    r = run(damped(LARGE, "water"))
    for q in ("added_mass", "radiation_damping"):
        close(view(r, q), view(ref, q)[:, :6], 1e-3)
    close(r.excitation.diffraction, ref.excitation.diffraction[:, :, :6], 1e-3)
    assert r.excitation.haskind is None  # not valid with a damper of the water


def test_water_damper_of_its_own_mesh_is_built_once():
    # the problem is rebuilt from the model for every query and solve: a damper of the water that is
    # a mesh of its own appears once, and not after it is removed
    def water_model():
        damper = {"mesh": surface(PLATE, symmetry="xz_and_yz"), "damping_factor": 2.0}
        return model({"mesh": surface(HULL, symmetry="xz_and_yz")}, W, [0.0], options=scattered(),
                     environment={"density": RHO, "gravity": G}, dampers=[damper])  # fmt: skip

    ref = water_model().solve()
    m = water_model()
    m.route()
    m.check()
    assert len(m.panels(role="fixed_damper", include_reflections=False).role) == len(PLATE)
    m.environment.density = 1025.0  # a change: the next query builds the problem again
    m.route()
    m.environment.density = RHO
    r = m.solve()
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        assert view(r, q).tobytes() == view(ref, q).tobytes(), q
    del m.fixed_dampers[0]
    assert len(m.panels(role="fixed_damper").role) == 0
    hull = solve({"mesh": surface(HULL, symmetry="xz_and_yz")}, W, [0.0], options=scattered())
    r = m.solve()
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        assert view(r, q).tobytes() == view(hull, q).tobytes(), q


@pytest.mark.parametrize("owner", ["water", "body"])
def test_damper_is_reciprocal_and_absorbs_energy(owner):
    # Green's identity with the jump c psi_n: the damper terms cancel for a damper at rest and for
    # one moving with its body (A and B symmetric), and the damper dissipates (the symmetric part of
    # B positive semidefinite; -1e-6: round-off, -1.4e-16 found). 5e-3: the discretization of the
    # coarse hull, as asymmetric without the damper (2.0e-3 and 4.1e-3; 2.2e-3 and 4.4e-3 found)
    for eps in (0.5, 2.0, 10.0):
        r = solve(omegas=W, options=scattered(diffraction=False), **damped(eps, owner))
        a, b = r.added_mass, r.radiation_damping
        for f in range(len(W)):
            close(a[f], a[f].T, 5e-3)
            close(b[f], b[f].T, 5e-3)
            assert np.linalg.eigvalsh(0.5 * (b[f] + b[f].T)).min() >= -1e-6 * peak(b[f])


def test_owners_share_the_diffraction_potential():
    # in diffraction a damper of the body and a damper of the water are both at rest: the same
    # equations, the same pressure on the hull, bit for bit
    opts = scattered(body_pressure=True)
    r = {o: solve(omegas=W, headings=[30.0], options=opts, **damped(2.0, o)) for o in ("water", "body")}
    np.testing.assert_array_equal(r["body"].body_points.pressure_diffraction,
                                  r["water"].body_points.pressure_diffraction)  # fmt: skip


def test_body_damper_haskind_equals_pressure():
    # the Haskind relation with the jump term of a damper of the body (theory §9.5), an identity of
    # the problem; 1e-2: the discretization of the coarse hull (4.4e-3 for the hull alone, 3.0e-3
    # found)
    r = run(damped(2.0, "body"))
    close(r.excitation.haskind, r.excitation.diffraction, 1e-2)


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(omegas=[0.0, 1.0]), "frequency limits"),
        (dict(options=scattered(drift_momentum_conservation=True, motion=False)), "momentum_conservation"),
    ],
)
def test_undefined_damper_problems_are_rejected(change, message):
    case = damped(1.0, "water", **(dict(modes=[]) if "options" in change else {}))
    kw = dict(omegas=W, headings=[0.0], options=scattered())
    kw.update(change)
    if "options" not in change:
        kw["options"] = scattered(diffraction=False)
        kw["headings"] = []
    with pytest.raises(ValueError, match=message):
        model(omegas=kw["omegas"], headings=kw["headings"], environment={"density": RHO, "gravity": G},
              options=kw["options"], **case).solve()  # fmt: skip


def test_haskind_excitation_with_water_damper_is_rejected():
    case = damped(1.0, "water", excitation_method="haskind", mass="displaced", center_of_gravity=[0, 0, -0.3])
    with pytest.raises(ValueError, match="haskind"):
        model(omegas=W, headings=[0.0], options=scattered(), **case).counts()
    # a damper of the water removes the Haskind relation of every body
    other = {"name": "b", "mesh": surface(HULL, symmetry="xz_and_yz"), "position": [6.0, 0.0, 0.0],
             "excitation_method": "haskind"}  # fmt: skip
    message = r"bodies/b: excitation_method='haskind' is not valid with fixed dampers"
    with pytest.raises(ValueError, match=message):
        model([{"name": "a", "mesh": WETTED, "modes": []}, other], W, [0.0], options=scattered(),
              dampers=[{"mesh": SHEET, "damping_factor": 1.0}]).counts()  # fmt: skip


def test_water_dampers_at_rest_are_reciprocal_and_absorb_energy():
    # a damper of the water given as a mesh of its own in global coordinates, and as the plate
    # elements of the mesh of a body at the origin: the same sheet at rest beside a placed hull, the
    # same results; A and B symmetric, B positive semidefinite
    hull = surface(HULL, symmetry="xz_and_yz")
    moved = {"name": "b", "mesh": hull, "position": [4.0, 0.0, 0.0]}
    own = solve([{"name": "a", "mesh": hull}, moved], W, options=scattered(diffraction=False),
                dampers=[{"mesh": surface(PLATE, symmetry="xz_and_yz"), "damping_factor": 1.5}])  # fmt: skip
    rec = solve([{"name": "a", "mesh": WETTED}, moved], W, options=scattered(diffraction=False),
                dampers=[{"mesh": SHEET, "damping_factor": 1.5}])  # fmt: skip
    for q in ("added_mass", "radiation_damping"):
        close(view(rec, q), view(own, q), 1e-5)  # the same equations; 1e-5: float32
    # symmetric to the discretization of the two coarse hulls (B of the pair without the damper is
    # symmetric to 1e-2); B positive semidefinite to -1e-4 of its peak, the same discretization
    # perturbing the eigenvalues near zero of the pair (-2.9e-6 found)
    a, b = own.added_mass, own.radiation_damping
    for f in range(len(W)):
        close(a[f], a[f].T, 5e-3)
        close(b[f], b[f].T, 2e-2)
        assert np.linalg.eigvalsh(0.5 * (b[f] + b[f].T)).min() >= -1e-4 * peak(b[f])
    # a water damper of a mesh of its own is an element of no body (body -1)
    q = model({"mesh": hull}, W, (), options={"diffraction": False},
              dampers=[{"mesh": surface(PLATE, symmetry="xz_and_yz")}]).panels(role="fixed_damper",
                                                                               include_reflections=False)  # fmt: skip
    assert q.role.tolist() == ["fixed_damper"] * len(PLATE) and np.all(q.body == -1)


def test_damper_inputs_are_checked():
    m = model({"mesh": WETTED, "dipole_dampers": [{"mesh": SHEET, "damping_factor": 1.0}]}, W, ())
    damper = m.bodies["body"].dipole_dampers[0]
    with pytest.raises(ValueError, match="damping_factor"):
        damper.damping_factor = -1.0
    with pytest.raises(AttributeError, match="condition"):
        damper.condition = "porous"  # no condition: the owner sets it
    m.bodies["body"].dipoles.add(mesh=damper.mesh)
    with pytest.raises(ValueError, match="an element has one role in a body"):
        m.counts()


# the higher order: the cylinder R = 1, T = 1 and a planar annular disc below it (the
# cylinder with a dipole disc), a dipole patch with curved sides
HO_ROWS = [(4, 4, 3, 3, 3, 4), (4, 2, 3, 3, 3, 4), (5, 3, 3, 3, 3, 4)]
HO_DISC = analytic("cylinder_with_dipole_discs", HO_ROWS, radius=1.0, draft=1.0, discs=[(1.3, 0.2, 1.5)])
HO_HULL = analytic("circular_cylinder", HO_ROWS[:2], radius=1.0, draft=1.0)


def run_ho(case, **kw):
    case = case if "bodies" in case else {"bodies": case}
    return solve(omegas=W, headings=[0.0, 45.0], options={"higher_order": True, **kw}, **case)


def ho_damped(eps, owner):
    """The cylinder (elements 1, 2 of the mesh) with the disc (element 3) as a damper of the body or
    of the water: the model keywords."""
    damper = {"mesh": (HO_DISC, (3, 3)), "damping_factor": eps}
    if owner == "body":
        return {"bodies": {"mesh": (HO_DISC, (1, 2)), "dipole_dampers": [damper]}}
    return {"bodies": {"mesh": (HO_DISC, (1, 2))}, "dampers": [damper]}


@pytest.mark.slow
def test_higher_order_damper_limits_reciprocity_and_energy():
    ref0 = run_ho({"mesh": HO_HULL})
    r0 = run_ho(ho_damped(0.0, "body"))
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        close(view(r0, q), view(ref0, q), 1e-5)
    rigid = run_ho({"mesh": (HO_DISC, (1, 2)), "dipoles": [{"mesh": (HO_DISC, (3, 3))}]})
    big = run_ho(ho_damped(1e5, "body"))
    for q in ("added_mass", "radiation_damping", "excitation_diffraction", "excitation_haskind"):
        close(view(big, q), view(rigid, q), 1e-3)
    for owner in ("water", "body"):
        r = solve(omegas=W, options={"higher_order": True, "diffraction": False}, **ho_damped(2.0, owner))
        for f in range(len(W)):
            a, b = r.added_mass[f], r.radiation_damping[f]
            close(a, a.T, 5e-3)
            close(b, b.T, 5e-3)
            assert np.linalg.eigvalsh(0.5 * (b + b.T)).min() >= -1e-4 * peak(b)
    free = run_ho(ho_damped(2.0, "water"))
    assert free.excitation.haskind is None
