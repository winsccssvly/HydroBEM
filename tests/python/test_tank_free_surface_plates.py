"""Free-surface plates of a tank's free surface (theory §10.2, §9.7).

A plate of a tank's free surface z = Z_T (a deck over part of a tank, a full cover) is a plate of
the tank problem, moving with every mode of the hull. A fully covered tank is a closed container
full of liquid: its potential for a translation is the translation itself, so the tank's added
mass is the fluid mass rho_T V_T at every frequency, its damping zero, and heave is defined at
omega = 0; its restoring is that of a solid of the fluid's weight (the plate's area leaves the
free-surface term rho_T g A_T). At omega = 0 the free surface is rigid, so a plate with no
normal velocity (surge, sway, yaw) changes nothing there; a cover of vanishing area tends to the
open tank. A plate of the sea over a tank's fluid (a floating roof as a body of its own) is
refused. Meshes are procedural (tests/python/meshes.py)."""

import math
import warnings

import numpy as np
import pytest

import hydrobem as hb

from bodies import TANK_B as B
from bodies import TANK_H as H
from bodies import TANK_L as L
from bodies import TANK_ZT as ZT
from bodies import flat, part, surface
from meshes import box_panels, face_panels
from problems import RHO, G, endpoints, model, peak, solve

QUARTER = "xz_and_yz"
MASS = RHO * L * B * H  # the tank fluid


def cover_panels(x0=0.0, x1=L / 2, y1=B / 2, n=(8, 4), z=ZT):
    """The quadrant of a cover of the tank x0 <= x <= x1, 0 <= y <= y1 at z (normals turned by the
    engine)."""
    return np.array(face_panels((x0, 0.0, z), (x1 - x0, 0.0, 0.0), (0.0, y1, 0.0), n[0], n[1], (0, 0, -1)))


def vessel(density=1.0, plate=None, own_mesh=False, n=(8, 4, 3), **items):
    """The box barge 2 x 1 x 1 of tests/python/bodies.py with its central tank (quadrant) and the
    plate panels ``plate`` of its free surface: elements of the body's mesh after the walls, or a
    mesh of their own."""
    hull = box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True)
    walls = box_panels(L / 2, B / 2, H, n=n, quarter=True)[:, ::-1] + [0.0, 0.0, ZT]
    extra = [] if plate is None or own_mesh else [plate]
    mesh = surface(np.concatenate([hull, walls, *extra]), symmetry=QUARTER)
    nh, nw = len(hull), len(walls)
    tank = {"mesh": part(mesh, nh + 1, nh + nw), "density_ratio": density}
    if plate is not None:
        own = {"mesh": surface(plate, symmetry=QUARTER)}
        tank["free_surface_plates"] = [own if own_mesh else {"mesh": part(mesh, nh + nw + 1, nh + nw + len(plate))}]
    return {"mesh": part(mesh, 1, nh), "mass": "displaced", "tanks": [tank], **items}


def radiation(body, omegas, higher_order=False, **opts):
    env = {"density": RHO, "gravity": G}
    return model(body, omegas, (), environment=env, options={"diffraction": False, "higher_order": higher_order,
                                                               **opts}).solve()  # fmt: skip


def tank_part(make, omegas, **kw):
    """The tank's part of A (at the limits and the frequencies) and of B: the vessel with its tank
    less the vessel with the tank empty."""
    r1, r0 = radiation(make(1.0), omegas, **kw), radiation(make(0.0), omegas, **kw)
    e1, e0 = endpoints(r1), endpoints(r0)
    lim = {k: np.asarray(e1[k]) - np.asarray(e0[k]) for k in e1}
    a = np.asarray(r1.added_mass) - np.asarray(r0.added_mass)
    b = np.asarray(r1.radiation_damping) - np.asarray(r0.radiation_damping)
    return lim, a, b, np.asarray(r1.added_mass)


def test_full_cover_is_a_closed_container():
    """[A] The fully covered tank: A_jj = rho_T V_T for surge, sway and heave at omega = 0, 1, 2,
    inf (the fluid moves rigidly with the walls, exact in the continuous problem), B = 0 and no
    warning at omega = 0 (no net flux through walls and cover). The closed container's potential is
    defined up to a constant at every frequency, and the solve pins it (theory §10.2): a stretch of
    the cover by 1e-9 changes A by float32 round-off, where the unpinned solve moved heave by
    several per cent. Constant panels on these 8 x 4 x 3 walls and 8 x 4 cover panels a quadrant:
    surge and sway 0.7 to 0.9 % at omega = 0, 1, 2 (the open tank's at omega = 0 to the bit: the
    plate has no normal velocity there), 4.4 % at omega = inf (the cover a sheet of the odd-image
    problem), heave 3.1 to 3.4 % and 5.4 % at omega = inf; on twice as many panels a side 0.3 %,
    2.0 %, heave 1.0 % and 2.3 %, O(h^1.6) and O(h), validation. The damping is that of the
    discretization, 9e-5 of the peak (3e-5 on twice as many). The higher order on flat patches
    represents the rigid flow exactly (the float32 storage)."""
    w = [1.0, 2.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", hb.HydroBEMWarning)
        lim, a, b, a_full = tank_part(lambda d: vessel(d, cover_panels()), [0.0, *w, math.inf])
    for k in ("zero", "infinite"):
        err = np.abs(np.diag(lim[k])[:3] / MASS - 1.0)
        assert np.all(err < 0.06), (k, err)
    err = np.abs(np.array([np.diag(a[f])[:3] for f in range(len(w))]) / MASS - 1.0)
    assert np.all(err < 0.04), err
    assert np.all(err[:, :2] < 0.012), err
    # no wave in a closed container: the damping of the tank's part is that of the discretization
    assert peak(b) <= 3e-4 * peak(a_full), peak(b) / peak(a_full)
    # the constant pinned: a negligible change of the geometry changes A by round-off
    stretched, *_ = tank_part(lambda d: vessel(d, cover_panels(x1=L / 2 * (1.0 + 1e-9))), [0.0, math.inf])
    for k in ("zero", "infinite"):
        change = np.abs(np.diag(stretched[k])[:3] - np.diag(lim[k])[:3])
        assert np.all(change <= 1e-5 * MASS), (k, change / MASS)
    # the low and the higher order agree on the full cover of flat patches
    hull = box_panels(2.0, 1.0, 1.0, n=(1, 1, 1), quarter=True)
    walls = box_panels(L / 2, B / 2, H, n=(1, 1, 1), quarter=True)[:, ::-1] + [0.0, 0.0, ZT]
    top = cover_panels(n=(1, 1))
    mesh = flat(np.concatenate([hull, walls, top]), (4, 4, 3, 3, 3, 4), symmetry=QUARTER)

    def ho(d):
        return {"mesh": part(mesh, 1, 3), "tanks": [{"mesh": part(mesh, 4, 6), "density_ratio": d,
                                                     "free_surface_plates": [{"mesh": part(mesh, 7, 7)}]}]}  # fmt: skip

    lim, a, b, a_full = tank_part(ho, [0.0, *w, math.inf], higher_order=True)
    for k in ("zero", "infinite"):
        assert np.all(np.abs(np.diag(lim[k])[:3] / MASS - 1.0) < 1e-5), (k, np.diag(lim[k]))
    assert np.all(np.abs(np.array([np.diag(a[f])[:3] for f in range(len(w))]) / MASS - 1.0) < 1e-5)
    assert peak(b) <= 1e-5 * peak(a_full)


def test_half_cover_at_zero_frequency_is_the_open_tank():
    """[A] At omega = 0 the free surface is a rigid lid: a plate there with no normal velocity
    (surge, sway, yaw) changes nothing, its column and right-hand side vanish on z = 0 with the
    rigid-lid Green function, so the half covered tank's surge, sway and yaw rows equal the open
    tank's (the plate's unknowns kept: to round-off, float32); roll and pitch move the plate and
    differ; heave is not defined in either (a net flux through the uncovered part). Both methods
    (the higher order on flat patches, the cover one patch). The plate as elements of the body's
    mesh equals the plate as a mesh of its own (the same elements) to round-off at omega = 1, 2."""
    half = cover_panels(x1=L / 4, n=(4, 4))
    with pytest.warns(hb.HydroBEMWarning, match="heave"):
        a_half = endpoints(radiation(vessel(1.0, half), [0.0]))["zero"]
    with pytest.warns(hb.HydroBEMWarning, match="heave"):
        a_open = endpoints(radiation(vessel(1.0), [0.0]))["zero"]
    keep = [0, 1, 5]
    d = np.asarray(a_half)[np.ix_(keep, keep)] - np.asarray(a_open)[np.ix_(keep, keep)]
    assert peak(d) <= 1e-6 * peak(np.asarray(a_open)[np.ix_(keep, keep)]), d
    assert abs(a_half[4, 4] - a_open[4, 4]) > 1e-4 * abs(a_open[4, 4])  # the plate moves in pitch
    assert np.isnan(a_half[2, 2])
    # the higher order
    hull = box_panels(2.0, 1.0, 1.0, n=(1, 1, 1), quarter=True)
    walls = box_panels(L / 2, B / 2, H, n=(1, 1, 1), quarter=True)[:, ::-1] + [0.0, 0.0, ZT]
    mesh = flat(np.concatenate([hull, walls, cover_panels(x1=L / 4, n=(1, 1))]), (4, 4, 3, 3, 3, 4),
                symmetry=QUARTER)  # fmt: skip

    def ho(plate):
        t = {"mesh": part(mesh, 4, 6), **({"free_surface_plates": [{"mesh": part(mesh, 7, 7)}]} if plate else {})}
        with pytest.warns(hb.HydroBEMWarning, match="heave"):
            return np.asarray(endpoints(radiation({"mesh": part(mesh, 1, 3), "tanks": [t]}, [0.0],
                                                  higher_order=True))["zero"])[np.ix_(keep, keep)]  # fmt: skip

    assert peak(ho(True) - ho(False)) <= 1e-6 * peak(ho(False))
    # a mesh of its own: the same equations in another order, equal to float32 round-off (to the
    # bit on some processors, not on others: their kernels sum in another order)
    r1, r2 = radiation(vessel(1.0, half), [1.0, 2.0]), radiation(vessel(1.0, half, own_mesh=True), [1.0, 2.0])
    assert peak(np.asarray(r1.added_mass) - np.asarray(r2.added_mass)) <= 1e-6 * peak(np.asarray(r1.added_mass))
    assert peak(np.asarray(r1.radiation_damping) - np.asarray(r2.radiation_damping)) <= 1e-6 * peak(
        np.asarray(r1.added_mass))


def test_a_vanishing_cover_tends_to_the_open_tank():
    """[B] A square cover of side s at the centre of the tank's free surface (a mesh of its own),
    s -> 0: the tank's part of A at omega = 1 and 2 tends to the open tank's, the difference falling
    with the area: 13 %, 3.3 %, 0.83 % of the peak for s = 0.4, 0.2, 0.1, the ratio 4.0 per
    halving (0.21 % at s = 0.05)."""
    w = [1.0, 2.0]
    _, a_open, _, _ = tank_part(lambda d: vessel(d), w)
    diffs = []
    for s in (0.4, 0.2, 0.1):
        plate = cover_panels(x1=s / 2, y1=s / 2, n=(1, 1))
        _, a, _, _ = tank_part(lambda d, p=plate: vessel(d, p, own_mesh=True), w)
        diffs.append(peak(a - a_open) / peak(a_open))
    assert diffs[0] > diffs[1] > diffs[2], diffs
    assert diffs[2] < 2e-2, diffs
    assert diffs[0] / diffs[1] > 3.5 and diffs[1] / diffs[2] > 3.5, diffs


def test_tank_with_a_plate_is_reciprocal():
    """[B] A and B of the half covered tank symmetric (the vessel's six modes): the asymmetry of the
    discretization, 3.8e-3 and 3.3e-3 of the peak of A, the open tank's to 1 % of itself (the
    asymmetry is the hull's surge-pitch coupling, the same on twice as many panels)."""
    def asymmetry(p):
        r = solve(vessel(1.0, p), [1.0, 2.0], options={"diffraction": False})
        return [peak(np.asarray(q) - np.swapaxes(q, 1, 2)) / peak(np.asarray(r.added_mass))
                for q in (r.added_mass, r.radiation_damping)]  # fmt: skip

    half, open_ = asymmetry(cover_panels(x1=L / 4, n=(4, 4))), asymmetry(None)
    assert max(half) <= 5e-3, half
    assert np.allclose(half, open_, rtol=2e-2), (half, open_)


def test_the_plate_area_leaves_the_tank_free_surface_term():
    """[A] The tank's restoring with plates (theory §10.2): the plate's area and moments leave the
    free-surface terms, exact for flat panels; the full cover removes rho_T g A_T from C33 and the
    free-surface moments of inertia from C44 and C55 (the fluid then restores as a solid of its
    weight), the fluid mass unchanged."""
    def hydro(b):
        return model(b, [1.0], (), environment={"density": RHO, "gravity": G},
                     options={"diffraction": False}).solve().hydrostatics  # fmt: skip

    h_open, h_full, h_empty = hydro(vessel(1.0)), hydro(vessel(1.0, cover_panels())), hydro(vessel(0.0))
    h_half = hydro(vessel(1.0, cover_panels(x1=L / 4, n=(4, 4)), own_mesh=True))
    c = {k: np.asarray(h.stiffness_rho_g[0]) for k, h in
         (("open", h_open), ("full", h_full), ("empty", h_empty), ("half", h_half))}  # fmt: skip
    rg = RHO * G
    np.testing.assert_allclose(c["empty"][2, 2] - c["open"][2, 2], rg * L * B, rtol=1e-12)
    np.testing.assert_allclose(c["full"][2, 2], c["empty"][2, 2], rtol=1e-12)
    np.testing.assert_allclose(c["half"][2, 2] - c["open"][2, 2], 0.5 * rg * L * B, rtol=1e-12)
    # roll and pitch: the full cover leaves the weight term of the tank fluid only (centroid ZT - H/2)
    weight = rg * L * B * H * (ZT - H / 2)
    for i in (3, 4):
        np.testing.assert_allclose(c["empty"][i, i] - c["full"][i, i], weight, rtol=1e-10)
    for h in (h_full, h_half):
        assert h.mass[0] == pytest.approx(h_open.mass[0], rel=1e-14)


def test_tank_plates_in_the_queries():
    """The elements of a tank's plate: role free_surface_plate with the tank's key (on the body's
    mesh and on a mesh of their own), the walls role tank."""
    plate = cover_panels(x1=L / 4, n=(4, 4))
    for own in (False, True):
        m = model(vessel(1.0, plate, own_mesh=own), [1.0], (), options={"diffraction": False})
        q = m.bodies["body"].panels(role="free_surface_plate", include_reflections=False)
        assert len(q.element) == len(plate) and set(q.tank) == {0} and set(q.owner) == {0}
        assert np.allclose(q.centroids[:, 2], ZT)
        walls = m.bodies["body"].panels(role="tank", include_reflections=False)
        assert set(walls.tank) == {0} and len(walls.element) > 0
        assert "free_surface_plate" not in set(walls.role)


def test_a_massless_mode_of_a_tank_plate_of_its_own_moves_water():
    # rule 10: a massless generalized mode that moves only the plate of a tank's free surface given
    # as a mesh of its own (a piece over every element less one over the body mesh's: the plate's
    # elements are in no piece with ranges) moves the tank's water, so it is accepted; without the
    # plate it moves none and is refused
    walls = vessel()["tanks"][0]["mesh"][1]
    mode = {"name": "lid", "parity": "heave",
            "pieces": [{"w": [[1.0, 0, 0, 0]]}, {"w": [[-1.0, 0, 0, 0]], "ranges": [(1, walls[1])]}]}  # fmt: skip
    items = {"modes": ["heave"], "center_of_gravity": [0.0, 0.0, -0.2], "generalized_modes": [mode],
             "mode_mass": np.zeros((7, 7))}  # fmt: skip
    model(vessel(plate=cover_panels(), own_mesh=True, **items), [1.0], options={"motion": True}).check()
    with pytest.raises(ValueError, match=r"generalized mode 'lid' has no mass and moves no water"):
        model(vessel(**items), [1.0], options={"motion": True}).check()


def test_tank_plate_refusals():
    """The rules and refusals of a tank's plates: at its surface level, no overlap (the rules of the
    tank problem, at the model's check), not with the tank fluid's pressure drift, not on a trimmed
    body; a plate of the sea over a tank's fluid (a floating roof) refused."""
    low = cover_panels(z=ZT - 0.05, n=(2, 2))
    with pytest.raises(ValueError, match="tank 1 free-surface plate 1: an element does not lie in the tank's free"):
        model(vessel(1.0, low), [1.0], (), options={"diffraction": False}).check()
    # two plates of the tank over the same water
    b = vessel(1.0, cover_panels(n=(2, 2)), own_mesh=True)
    b["tanks"][0]["free_surface_plates"].append({"mesh": surface(cover_panels(n=(1, 1)), symmetry=QUARTER)})
    with pytest.raises(ValueError, match="overlaps"):
        model(b, [1.0], (), options={"diffraction": False}).check()
    with pytest.raises(hb.UnsupportedProblemError, match="pressure integration .* plates of a tank's free surface"):
        model(vessel(1.0, cover_panels(), center_of_gravity=[0.0, 0.0, -0.6], radii_of_gyration=[0.4, 0.5, 0.5]),
              [1.0], [0.0], options={"motion": True, "drift_pressure_integration": True}).route()
    with pytest.raises(ValueError, match="free-surface plates of a tank of a trimmed body are not available"):
        model(vessel(1.0, cover_panels(), trim=[0.0, 1.0, 0.0]), [1.0], ()).check()
    # a generated plate (inside) is a plate of the sea on a trimmed hull, not of a tank
    b = vessel(1.0, cover_panels(), own_mesh=True)
    b["tanks"][0]["free_surface_plates"] = [{"inside": [0.1, 0.1]}]
    with pytest.raises(ValueError, match="a plate of a tank's free surface is given by its mesh"):
        model(b, [1.0], ()).check()
    # a tank whose surface lies in z = 0 and a plate body of its own on its fluid
    hull = box_panels(2.0, 1.0, 1.0, n=(8, 4, 4), quarter=True)
    walls = box_panels(L / 2, B / 2, H, n=(8, 4, 3), quarter=True)[:, ::-1]
    mesh = surface(np.concatenate([hull, walls]), symmetry=QUARTER)
    tanked = {"name": "hull", "mesh": part(mesh, 1, len(hull)), "modes": [],
              "tanks": [{"mesh": part(mesh, len(hull) + 1, len(hull) + len(walls))}]}  # fmt: skip
    disc = surface(cover_panels(x1=0.3, y1=0.2, n=(1, 1), z=0.0), symmetry=QUARTER)
    roof = {"name": "roof", "modes": ["heave"], "free_surface_plates": [{"mesh": disc}]}
    match = "bodies/roof free-surface plate 1 lies on the fluid of bodies/hull tank 1, a floating roof"
    with pytest.raises(hb.UnsupportedProblemError, match=match):
        model([tanked, roof], [1.0], (), options={"diffraction": False}).route()
