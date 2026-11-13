"""Free-surface plates: massless rigid plates in z = 0 moving with their body (theory §9.7,
test-plan A and B).

Each check on the basis it rests on. Analytic: at omega = 0 a plate in z = 0 is a piston in the
rigid plane z = 0, A33(0) = rho/(2 pi) int int dS dS'/|x - x'| (Rayleigh's piston: (8/3) rho a^3
for a disc of radius a, (2/pi)(ln(1 + sqrt 2) + (1 - sqrt 2)/3) rho s^3 for a square of side s).
Semi-analytic: the disc in finite depth is the truncated cylinder of zero draft of the eigenfunction
matching of scripts/gen_cylinder_reference.py (the dock). Identities of the problem: the Haskind
relation with the plate as part of the body, the energy of the radiated wave (B33 = omega^3 |X3|^2 /
(2 rho g^3) of an axisymmetric body in deep water), A and B symmetric with B >= 0, the symmetry
sectors, both formulations, a plate of the hull's body against the same plate as a body of its own
(the rigid connection), a generalized mode of the plate's elements against the plate's heave, the
mirror image of a wall, the limit of a wetted disc of small draft, the body velocity against the
field velocity beside it and the field velocity against differences of the field pressure, a skirt
hanging from a plate whose elements end on it (converging; reference/inputs-rules §1 rule 3), the two formulations
under the piston of a deep water column (converging to one force, the total one the closer; the
choice of "auto"). A channel: no reflection is the open sea, the mirror body, the wall condition,
the rigid connection, Haskind and reciprocity, and at omega = 0 a wide channel tending to the open
sea as 1/W^2 (both methods). A body with tanks: the tank and the plates independent. The momentum
drift (theory §11.1): the far-field energy against the damping (both methods, O(h^2), O(h^3)), the
optical theorem of a fixed and a floating disc, the formulations, the sectors and head seas, its
convergence, the limits of a wetted disc of small draft and of a moonpool plugged by a wetted disc
just below the cover (the ordinary body's Kochin function without the plate's column), a
generalized piston mode against a piston body, Newman's diagonal, the low against the higher
order. The pressure drift and its quadratic QTFs (theory §11.2, §12.3) of plates whose edges lie
on walls: against the momentum drift for a cover and a piston (the change between them, which the
band under the plate carries) in both methods and for the piston of a deep water column, a
generalized piston mode against a piston body and a cover of the hull against a cover body (to
rounding), the QTF's diagonal and symmetries, a non-conforming cover. The control-surface drift
and its quadratic QTFs (theory §11.3) of a cover inside its body's control surface: against the
momentum drift (fixed and floating, both methods), the independence of the surface, the moonpool
plugged by a wetted disc just below the cover (an ordinary body's control-surface drift, every
component and the QTF), the two forms of the squared elevation, the QTF's diagonal and symmetries,
the automatic surface leaving the covered waterline out. At omega = inf (the plate a
sheet of the odd-image problem): half of Lamb's disc in heave
and pitch (both methods, converging), half of the same disc as a deep sheet (the mean of its two
limits), the square in both methods extrapolated, the cover of a hull against the cover body and the
generalized piston, A symmetric, the wall's mirror pair, finite depth tending to deep water, a channel
without reflection, and A(omega) of the dock by eigenfunction matching approaching A(inf).
The potential QTF (theory §12.4-§12.10) of plates whose edges lie on walls: the free surface ending
at a cover, a cover of the hull against a cover body, a generalized piston mode against a piston
body (the jump of the displacement at the wall), both methods converging to each other with the
panels, fixed and floating, the inner radius, the limit of the plugged moonpool, fixed and
floating, and a wall's mirror image.
Refusals: the features not available (a plate edge bordered by free surface in the pressure
drift and the potential QTF among them; a piston body inside its hull's control surface), the thin
sheets in z = 0, a plate overlapping another
surface of z = 0, a sheet meeting a plate or a lid inside an element and a sheet hanging from a
plate at omega = inf. Meshes are procedural
(tests/python/meshes.py)."""

import math

import numpy as np
import pytest

import hydrobem as hb

from bodies import flat, moonpool_barge, part, spline, surface, tank_vessel
from meshes import (
    MOONPOOL_FACES,
    box_panels,
    cylinder_panels,
    disc_panels,
    face_panels,
    moonpool_lid_panels,
    moonpool_panels,
    spline_cylinder_patches,
)
from problems import analytic, drift_constants, load_script, model, peak, solve, view

RHO, G = 1000.0, 9.80665
KA = np.array([0.5, 1.0, 2.0])
W = np.sqrt(KA * G)
QUARTER = "xz_and_yz"
# the self-integral of the unit square, int int dS dS' / |x - x'| = 4 (ln(1 + sqrt 2) + (1 - sqrt 2) / 3)
SQUARE = 4.0 * (math.log(1.0 + math.sqrt(2.0)) + (1.0 - math.sqrt(2.0)) / 3.0)


def disc(n, quarter=True, graded=False, radius=1.0):
    """A disc of n rings and 2 n divisions a quadrant."""
    return surface(disc_panels(radius, n_r=n, n_phi=2 * n if quarter else 8 * n, quarter=quarter, graded=graded),
                   symmetry=QUARTER if quarter else "none")  # fmt: skip


def plate_body(mesh, name="plate", modes=("heave",), **items):
    """A body of a plate only."""
    return {"name": name, "modes": list(modes), "free_surface_plates": [{"mesh": mesh}], **items}


def heave_a0(r):
    return r.quantity("added_mass_zero_frequency")[2, 0]


def test_a_disc_at_zero_frequency_is_a_piston_in_a_rigid_plane():
    """At omega = 0 the free surface is the rigid plane z = 0 and the heaving disc a piston in it:
    A33(0) = (8/3) rho a^3 (the double integral of 1/R over the disc, 16 pi a^3 / 3, times rho / 2 pi).
    Uniform rings, 8 and 16 a quadrant: 6.0e-4 and 6.4e-5 (O(h^3)); 1e-3 and 2e-4. Near omega = 0
    (K a = 2.5e-4) the added mass is that of the limit to 1.1e-3 (2e-3)."""
    errs = []
    for n in (8, 16):
        r = solve(plate_body(disc(n)), [0.0, 0.05], [], options={"diffraction": False})
        errs.append(abs(heave_a0(r) / (8.0 / 3.0 * RHO) - 1.0))
        assert abs(r.quantity("added_mass")[0, 2, 0] / heave_a0(r) - 1.0) <= 2e-3
    assert errs[0] <= 1e-3 and errs[1] <= 2e-4 and errs[1] < errs[0] / 4, errs


@pytest.mark.parametrize("higher_order, sizes, bounds", [(False, (8, 16), (3e-3, 1e-3)), (True, (2, 4), (6e-4, 2e-4))])
def test_a_square_plate_at_zero_frequency_in_both_methods(higher_order, sizes, bounds):
    """The square piston of side 2 at omega = 0: A33(0) = rho SQUARE s^3 / (2 pi), the self-integral
    of the square in closed form (checked against its numerical quadrature to 1e-15). Low order 8, 16
    panels a side of the quadrant: 1.9e-3, 5.5e-4; higher order 2, 4 cubic panels: 3.1e-4, 7.8e-5
    (O(h^2)); about twice those."""
    ref = RHO * SQUARE * 8.0 / (2.0 * math.pi)
    q = face_panels((0, 0, 0), (1, 0, 0), (0, 1, 0), 1, 1, (0, 0, -1))
    for n, bound in zip(sizes, bounds, strict=True):
        if higher_order:
            mesh = flat(np.array(q), [(n, n, 3, 3, 3, 4)], symmetry=QUARTER)
        else:
            mesh = surface(face_panels((0, 0, 0), (1, 0, 0), (0, 1, 0), n, n, (0, 0, -1)), symmetry=QUARTER)
        r = solve(plate_body(mesh), [0.0], [], options={"diffraction": False, "higher_order": higher_order})
        assert abs(heave_a0(r) / ref - 1.0) <= bound, (n, heave_a0(r) / ref - 1.0)


def test_the_dock_in_finite_depth_against_eigenfunction_matching():
    """A disc of radius 1 in water of depth 2: the truncated cylinder of zero draft of the
    eigenfunction matching (scripts/gen_cylinder_reference.py; with d = 0 the inner region is the
    whole depth under the plate), its truncation N = 200, 400 extrapolated with the observed order
    N^-2 (the step 200 -> 400 is 5e-6). 16 rings a quadrant: A 4e-5 to 1e-4, B 1.0e-3 to 2.0e-3 (the
    edge of the plate bordering the free surface, where the velocity is singular; the error falls as
    h^2, by 3.8 for each halving of the panels); 2e-4 and 4e-3."""
    gen = load_script("gen_cylinder_reference")
    ka = (0.4, 1.5)
    ref = []
    for k in ka:
        a, b = (np.array(gen.solve(2.0, 0.0, k, n, "heave")) for n in (200, 400))
        ref.append(b + (b - a) / 3.0)
    ref = np.array(ref)
    w = np.sqrt(np.array(ka) * G)
    r = solve(plate_body(disc(16)), w, [], depth=2.0, options={"diffraction": False})
    A = r.quantity("added_mass")[:, 2, 0] / RHO
    B = r.quantity("radiation_damping")[:, 2, 0] / (RHO * w)
    assert np.all(np.abs(A / ref[:, 0] - 1.0) <= 2e-4), A / ref[:, 0] - 1.0
    assert np.all(np.abs(B / ref[:, 1] - 1.0) <= 4e-3), B / ref[:, 1] - 1.0


@pytest.fixture(scope="module")
def discs():
    """The floating disc (heave, roll, pitch), graded towards its edge: quadrant and full, both
    formulations."""
    out = {}
    for quarter in (True, False):
        for form in ("scattered", "total"):
            b = plate_body(disc(8, quarter=quarter, graded=True), modes=("heave", "roll", "pitch"))
            out[quarter, form] = solve(b, W, [0.0, 30.0], options={"diffraction_formulation": form})
    return out


def test_haskind_energy_sectors_and_formulations_of_a_disc(discs):
    r = discs[True, "scattered"]
    X = r.excitation.diffraction
    # the Haskind relation over the wetted surface and the plate: 4.0e-5 found (roll and pitch at
    # 30 degrees); 2e-4
    assert np.abs(r.excitation.haskind - X[:, :, [2, 3, 4]]).max() <= 2e-4 * np.abs(X).max()
    # the energy of the radiated wave of an axisymmetric body in deep water, B33 = w^3 |X3|^2 /
    # (2 rho g^3) (Newman 1977, the Haskind relation for the damping): 8.6e-5 found at Ka = 2; 3e-4
    B = r.quantity("radiation_damping")[:, 2, 0]
    pred = W**3 * np.abs(X[:, 0, 2]) ** 2 / (2.0 * RHO * G**3)
    assert np.all(np.abs(B / pred - 1.0) <= 3e-4), B / pred - 1.0
    assert np.all(B > 0.0)
    # the symmetry sectors and the full mesh of the same panels, and the two formulations: the same
    # problem (2.1e-7 and 7.1e-8 found, float32); 1e-5
    for other in (discs[False, "scattered"], discs[True, "total"], discs[False, "total"]):
        for q in ("added_mass", "radiation_damping"):
            assert np.abs(other.quantity(q) - r.quantity(q)).max() <= 1e-5 * np.abs(r.quantity(q)).max(), q
        assert np.abs(other.excitation.diffraction - X).max() <= 1e-5 * np.abs(X).max()


def moonpool(n=8):
    """The quadrant moonpool hull (outer radius 2, moonpool 1, draft 1) and its free surface r < 1
    (the same azimuthal division: the plate's edge lies on the moonpool wall) as elements of one mesh."""
    hull, lid = moonpool_panels(n=n), moonpool_lid_panels(n=n)
    return surface(np.concatenate([hull, lid]), symmetry=QUARTER), len(hull), len(lid)


def test_a_plate_of_the_hull_moves_with_it_one_of_its_own_body_does_not():
    """The moonpool cover as a plate of the hull's body, as a body of its own (the two moving
    together are the one rigid body: the sum of the 2 x 2 blocks of their heave and pitch), and as
    a generalized mode of the hull whose piece names the plate's elements with w = 1 (the plate's
    heave). The same equations, solved in another order: 6e-7 found (float32); 1e-5. The generalized
    mode's restoring is rho g times the plate's area (theory §10.1); the waterplane of the hull's body
    holds the plate's area."""
    mesh, nh, nl = moonpool(8)
    hull = {"name": "hull", "mesh": part(mesh, 1, nh), "modes": ["heave", "pitch"]}
    plate = {"mesh": part(mesh, nh + 1, nh + nl)}
    one = solve({**hull, "free_surface_plates": [plate]}, W, [0.0])
    two = solve([hull, plate_body(plate["mesh"], name="lid", modes=("heave", "pitch"))], W, [0.0])
    for q in ("added_mass", "radiation_damping"):
        a, b = one.quantity(q), two.quantity(q)  # two: coordinates (hull 0..5, lid 6..11), columns (h3, h5, l3, l5)
        for row, (i, j) in ((2, (2, 8)), (4, (4, 10))):
            for col, (k, m) in ((0, (0, 2)), (1, (1, 3))):
                s = b[:, i, k] + b[:, i, m] + b[:, j, k] + b[:, j, m]
                assert np.abs(a[:, row, col] - s).max() <= 1e-5 * np.abs(a).max(), (q, row, col)
    x1, x2 = one.excitation.diffraction, two.excitation.diffraction
    for i, j in ((2, 8), (4, 10)):
        assert np.abs(x1[:, :, i] - x2[:, :, i] - x2[:, :, j]).max() <= 1e-5 * np.abs(x1).max()
    # the plate of its own body has no volume: its waterplane is the plate's (polygon) area
    area = one.quantity("hydrostatics_waterplane_area")[0]
    assert np.allclose(two.quantity("hydrostatics_waterplane_area").sum(), area, rtol=1e-12)
    assert two.quantity("hydrostatics_volume")[1] == 0.0
    # the generalized mode of the plate's elements: the plate's heave
    piston = {"name": "piston", "parity": "heave", "pieces": [{"w": [[1.0, 0, 0, 0]], "ranges": [(nh + 1, nh + nl)]}]}
    gen = solve({**hull, "modes": [], "free_surface_plates": [plate], "generalized_modes": [piston]}, W, [0.0])
    for q in ("added_mass", "radiation_damping"):
        a, b = gen.quantity(q)[:, 6, 0], two.quantity(q)[:, 8, 2]
        assert np.abs(a - b).max() <= 1e-5 * np.abs(b).max(), q
    lid_area = two.quantity("hydrostatics_waterplane_area")[1]
    # the restoring of the piston mode in the convention of theory §10.1: c_66 = c_63 = rho g area in
    # its row, the rigid heave row without it (its coupling belongs to mode_stiffness)
    c = gen.quantity("mode_restoring")
    assert np.isclose(c[6, 6], RHO * G * lid_area, rtol=1e-12)
    assert np.isclose(c[6, 2], RHO * G * lid_area, rtol=1e-12) and c[2, 6] == 0.0


def test_massless_plate_modes_and_bodies():
    """A plate has no mass of its own, and its mode need not have one (theory §3.1): the inertia of a
    massless direction is its added mass, and the motions are continuous as the mass tends to 0.
    The moonpool's free surface as a generalized piston mode of the hull with a mode mass of 1, 1e-3
    and 0 kg (A of the mode about 1e3 kg): 7.3e-4 and 7.8e-7 of the motions' peak found, linear in the
    mass. The same of the moonpool's cover as a body of its own (heave and pitch, radii of gyration
    0.3 m), mass 1, 1e-3 and 0 kg: 1.5e-2 and 1.6e-5 (its pitch inertia 0.09 kg m^2 at 1 kg). Bounds
    3e-2, 3e-5 and a fall of at least 500 times from 1 to 1e-3 kg."""
    mesh, nh, nl = moonpool(8)
    piston = {"name": "piston", "parity": "heave", "pieces": [{"w": [[1.0, 0, 0, 0]], "ranges": [(nh + 1, nh + nl)]}]}
    hull = {"name": "hull", "mesh": part(mesh, 1, nh), "modes": ["heave", "pitch"], "mass": 2000.0,
            "center_of_gravity": [0.0, 0.0, -0.3], "radii_of_gyration": [0.8, 0.8, 0.8]}  # fmt: skip
    plate = {"mesh": part(mesh, nh + 1, nh + nl)}

    def gen(mass):
        mm = np.zeros((7, 7))
        mm[6, 6] = mass
        b = {**hull, "free_surface_plates": [plate], "generalized_modes": [piston], "mode_mass": mm}
        return solve(b, W, [0.0], options={"motion": True}).quantity("motion")

    def lid(mass):
        cover = plate_body(plate["mesh"], name="lid", modes=("heave", "pitch"), mass=mass,
                           center_of_gravity=[0.0, 0.0, 0.0], radii_of_gyration=[0.3, 0.3, 0.3])  # fmt: skip
        return solve([hull, cover], W, [0.0], options={"motion": True}).quantity("motion")

    for motions in (gen, lid):
        x1, x3, x0 = motions(1.0), motions(1e-3), motions(0.0)
        d1, d3 = np.abs(x1 - x0).max() / peak(x0), np.abs(x3 - x0).max() / peak(x0)
        assert d1 <= 3e-2 and d3 <= 3e-5 and d3 <= d1 / 500, (motions.__name__, d1, d3)
        assert np.isfinite(x0).all() and peak(x0[..., 6:]) > 0.1 * peak(x0)


def test_reciprocity_of_a_hull_and_its_cover():
    """A and B of the hull and the cover (bodies of their own, heave and pitch) are symmetric and
    B >= 0. The plate meets the moonpool wall at a right angle of the fluid; the discrete matrices
    are symmetric only to the discretization of that corner: at Ka = 0.5, 1 the asymmetry of A is
    3.7e-3 and 1.7e-3 on 8 panels a quadrant and 1.9e-3, 8.9e-4 on 12 (O(h^1.7)), of B 3.5e-4, 2.8e-3
    and 1.4e-4, 1.2e-3; the eigenvalues of B above -2e-6 of its largest entry. Bounds 1.5 times those
    of 12 panels and a decrease from 8 to 12."""
    rows = [2, 4, 8, 10]
    asym = {}
    for n in (8, 12):
        mesh, nh, nl = moonpool(n)
        r = solve([{"name": "hull", "mesh": part(mesh, 1, nh), "modes": ["heave", "pitch"]},
                   plate_body(part(mesh, nh + 1, nh + nl), name="lid", modes=("heave", "pitch"))],
                  W[:2], [], options={"diffraction": False})  # fmt: skip
        a, b = r.quantity("added_mass")[:, rows], r.quantity("radiation_damping")[:, rows]
        asym[n] = [np.abs(x - np.swapaxes(x, 1, 2)).max(axis=(1, 2)) / np.abs(x).max(axis=(1, 2)) for x in (a, b)]
        for k in range(2):
            e = np.linalg.eigvalsh(0.5 * (b[k] + b[k].T))
            assert e.min() >= -1e-5 * np.abs(b[k]).max()
    for q in range(2):
        assert np.all(asym[12][q] < asym[8][q]), asym
    assert np.all(asym[12][0] <= np.array([3e-3, 1.5e-3])) and np.all(asym[12][1] <= np.array([3e-4, 2e-3])), asym


def test_the_edge_of_a_cover_on_the_wall():
    """The cover's edge on the moonpool wall (two bodies touching along the waterline: the rules
    accept it) and the cover 1e-7 smaller (a gap of 1e-7 of free surface): the same results to
    5e-7 found (float32 and the gap); 1e-5."""
    hull = {"name": "hull", "mesh": surface(moonpool_panels(n=8), symmetry=QUARTER), "modes": ["heave"]}
    out = []
    for r in (1.0, 1.0 - 1e-7):
        lid = plate_body(surface(moonpool_lid_panels(r=r, n=8), symmetry=QUARTER), name="lid")
        out.append(solve([hull, lid], W, [], options={"diffraction": False}).quantity("added_mass"))
    assert np.abs(out[0] - out[1]).max() <= 1e-5 * np.abs(out[0]).max()


def test_a_plate_by_a_wall_is_its_mirror_pair():
    """A disc at x = 3 by the wall x = 0 is the pair of discs at x = +-3 in the open sea moving as
    mirror images (surge, heave, pitch; the mirror image moves with -surge, heave, -pitch; the
    incident wave and its mirror, headings 30 and 150): every entry of the 3 x 3 blocks of A and B
    is the sum of the pair's direct and mirrored columns, and every exciting force the sum of the
    two headings' forces; 3e-7 found (float32); 1e-5."""
    full = disc(8, quarter=False)
    d = plate_body(full, name="d", modes=("surge", "heave", "pitch"), position=[3.0, 0.0, 0.0])
    wall = solve(d, W, [30.0], options={"domain": "x_wall"})
    pair = solve([d, {**d, "name": "e", "position": [-3.0, 0.0, 0.0]}], W, [30.0, 150.0])
    rows, mirror = [0, 2, 4], [-1.0, 1.0, -1.0]  # coordinates of d; columns d 0..2, e 3..5
    for q in ("added_mass", "radiation_damping"):
        a, b = wall.quantity(q)[:, rows], pair.quantity(q)[:, rows]
        s = b[:, :, :3] + b[:, :, 3:] * np.array(mirror)
        assert np.abs(a - s).max() <= 1e-5 * np.abs(a).max(), q
    x, y = wall.excitation.diffraction[:, 0, rows], pair.excitation.diffraction[:, :, rows]
    assert np.abs(x - y[:, 0] - y[:, 1]).max() <= 1e-5 * np.abs(x).max()


def test_a_wetted_disc_of_small_draft_tends_to_the_plate():
    """A truncated cylinder of radius 1 and draft d (the interior surface graded; the same 16 x 16
    panels a quadrant as the plate) against the plate: the difference falls as d (the volume and the
    side wall): at Ka = 0.5 1.6 %, 0.84 %, 0.47 % of A33 for d = 0.02, 0.01, 0.005 (B33 and |X3|
    alike), a ratio of 1.8 to 2.5 for each halving (1.5 to 2.7); at most 1.0 % at d = 0.005 at the
    three frequencies (1.2 %)."""
    n = 16
    quantities = ("added_mass", "radiation_damping")

    def heave(r):
        return np.r_[[r.quantity(q)[:, 2, 0] for q in quantities], [np.abs(r.excitation.diffraction[:, 0, 2])]]

    p = heave(solve(plate_body(disc(n)), W, [0.0]))
    errs = []
    for d in (0.02, 0.01, 0.005):
        c = surface(cylinder_panels(1.0, d, n_phi=2 * n, n_z=1, n_r=n), symmetry=QUARTER)
        r = solve({"mesh": c, "modes": ["heave"], "interior_surface": "graded"}, W, [0.0])
        errs.append(np.abs(heave(r) / p - 1.0))
    ratio = np.array(errs[:-1]) / np.array(errs[1:])
    assert np.all(errs[-1] <= 1.2e-2), errs[-1]
    assert np.all((ratio[:, :, 0] >= 1.5) & (ratio[:, :, 0] <= 2.7)), ratio


def test_body_outputs_on_a_plate():
    """The vertical velocity at the plate panels is that of its condition: i w per unit heave,
    -i w x per unit pitch (z = -x theta), 0 for the total diffraction potential (the plate at rest);
    3.4e-8 found; 1e-6. The plate panels are body points of their body (ids: body 0, element)."""
    b = plate_body(disc(4), modes=("heave", "pitch"))
    r = solve(b, W, [0.0], outputs={"body_pressure": True, "body_velocity": True})
    v, x = r.quantity("body_points_velocity_radiation"), r.quantity("body_points")
    assert len(x) == 4 * 32 and np.all(r.quantity("body_points_ids")[:, 0] == 0)
    iw = 1j * W[:, None]
    assert np.abs(v[:, :, 0, 2] / iw - 1.0).max() <= 1e-6
    assert np.abs(v[:, :, 1, 2] / iw + x[None, :, 0]).max() <= 1e-6
    assert np.abs(r.quantity("body_points_velocity_diffraction")[:, :, 0, 2]).max() <= 1e-6 * W.max()


def test_higher_order_cover_of_a_barge():
    """The higher order, the moonpool barge of flat patches with its moonpool free surface (element 7)
    as a plate of the barge and as a body of its own, both formulations: the rigid connection holds
    to 1e-7 found (1e-5); A and B do not depend on the formulation (1e-5) and the exciting force
    differs by 1.6e-4 between them (the discretization of the two right-hand sides; 1e-3); A and B
    symmetric to 2.6e-4 found at Ka = 2 (1e-3); the Haskind relation 1.1e-3 found at Ka = 2 (5e-3)."""
    mb = moonpool_barge()
    opt = {"higher_order": True}
    hull = {"name": "barge", "mesh": part(mb, 1, 6), "modes": ["heave", "pitch"]}
    res = {}
    for form in ("total", "scattered"):
        o = {**opt, "diffraction_formulation": form}
        res[form] = (solve({**hull, "free_surface_plates": [{"mesh": part(mb, 7, 7)}]}, [0.0, *W], [0.0], options=o),
                     solve([hull, plate_body(part(mb, 7, 7), name="lid", modes=("heave", "pitch"))], [0.0, *W],
                           [0.0], options=o))  # fmt: skip
    one, two = res["total"]
    a, b = one.quantity("added_mass")[:, 2, 0], two.quantity("added_mass")
    assert np.abs(a - (b[:, 2, 0] + b[:, 2, 2] + b[:, 8, 0] + b[:, 8, 2])).max() <= 1e-5 * np.abs(a).max()
    a0, b0 = one.quantity("added_mass_zero_frequency")[2, 0], two.quantity("added_mass_zero_frequency")
    assert abs(a0 - (b0[2, 0] + b0[2, 2] + b0[8, 0] + b0[8, 2])) <= 1e-5 * abs(a0)
    for q in ("added_mass", "radiation_damping"):
        x = two.quantity(q)[:, [2, 4, 8, 10]]
        assert np.abs(x - np.swapaxes(x, 1, 2)).max() <= 1e-3 * np.abs(x).max(), q
        y = res["scattered"][1].quantity(q)
        assert np.abs(y - two.quantity(q)).max() <= 1e-5 * np.abs(y).max(), q
    x = one.excitation.diffraction
    assert np.abs(res["scattered"][0].excitation.diffraction - x).max() <= 1e-3 * np.abs(x).max()
    assert np.abs(one.excitation.haskind[:, 0, 0] - x[:, 0, 2]).max() <= 5e-3 * np.abs(x[:, 0, 2]).max()


def _quarter_disc(p, profile):
    """The quadrant of the unit disc in z = 0 as one analytic patch, a radial line revolved (map
    "line", the radius and z of its ends in p), 4 x 4 spans, B-splines of order 3."""
    patch = {"type": "AnalyticPatch", "map": "line", "profile": profile, "a": 1.0, "b": 1.0, "theta0": 0.0,
             "theta1": 0.5 * math.pi, "p": p}  # fmt: skip
    return {"kind": "analytic_patches", "discretization": np.array([[4, 4, 3, 3, 3, 4]]), "symmetry": QUARTER,
            "patches": [patch]}  # fmt: skip


def spline_disc(up):
    """The quadrant of the unit disc in z = 0 as a B-spline patch (the bottom of the spline cylinder
    lifted): from the rim to the axis its normal points down, from the axis to the rim up."""
    order, _, t, tv, c = spline_cylinder_patches(1.0, 0.5, spans=4, order=4)[1]
    c = c.copy()
    c[:, :, 2] = 0.0
    return spline([(order, 2, t, tv, c[::-1] if up else c)], (4, 2, 3, 3, 3, 4), symmetry=QUARTER)


def _same_results(a, b, tol):
    """The largest difference of A, B and X (pressure and Haskind) relative to each one's peak."""
    worst = 0.0
    for q in ("added_mass", "radiation_damping", "excitation_diffraction", "excitation_haskind"):
        x, y = view(a, q), view(b, q)
        worst = max(worst, np.abs(x - y).max() / np.abs(x).max())
    assert worst <= tol, worst
    return worst


def test_a_plate_given_upwards_is_turned_over():
    """A plate's elements may point either way, in both methods (reference/inputs-bodies §2.4): those pointing up
    are turned over. The higher order turns a curved patch by its parameter u reversed (the same
    surface on the same spans): the waterplane part of the analytic circular cylinder (a radial line
    from the centre to the edge, normal up, nonuniform mapping) as a plate equals the line from the
    edge to the centre with the mirrored profile (its v reversed, normal down) to round-off (6e-16
    found); a linear line from the centre to the edge equals the one from the edge to the centre; a
    B-spline quarter disc from the axis to the rim equals the one from the rim to the axis. The
    waterplane part as a plate of the cylinder's body is accepted. Low order: the disc of panels in
    reversed vertex order equals the disc (7e-8 found, float32 matrices)."""
    ho = {"higher_order": True}
    modes = ("heave", "roll", "pitch")
    cyl = analytic("circular_cylinder", (4, 4, 3, 3, 3, 4), parts=["waterplane"], mapping="nonuniform",
                   radius=1.0, draft=0.5)  # fmt: skip
    pairs = [(cyl, _quarter_disc([1.0, 0.0, 0.0, 0.0], "quadratic_start")),
             (_quarter_disc([0.0, 0.0, 1.0, 0.0], "linear"), _quarter_disc([1.0, 0.0, 0.0, 0.0], "linear")),
             (spline_disc(True), spline_disc(False))]  # fmt: skip
    for up, down in pairs:
        r = [solve(plate_body(x, modes=modes), W, [0.0, 30.0], options=ho) for x in (up, down)]
        _same_results(*r, 1e-12)
    # the waterplane part as a plate of the cylinder's body
    hull = analytic("circular_cylinder", (4, 4, 3, 3, 3, 4), parts=["side", "bottom", "waterplane"], radius=1.0,
                    draft=0.5)  # fmt: skip
    m = model({"name": "c", "mesh": (hull, (1, 2)), "free_surface_plates": [{"mesh": (hull, (3, 3))}]}, [1.0],
              options=ho)  # fmt: skip
    m.check()
    # low order: the panels of a disc given the other way
    v = disc_panels(1.0, n_r=4, n_phi=8, quarter=True)
    r = [solve(plate_body(surface(x, symmetry=QUARTER), modes=modes), W, [0.0, 30.0]) for x in (v, v[:, ::-1])]
    _same_results(*r, 1e-5)


def test_the_panels_of_a_plate():
    """The panels query names the plate's elements with the role "free_surface_plate"."""
    mesh, nh, nl = moonpool(4)
    cover = {"mesh": part(mesh, nh + 1, nh + nl)}
    m = model({"name": "hull", "mesh": part(mesh, 1, nh), "free_surface_plates": [cover]}, [1.0])
    p = m.panels(include_reflections=False)
    assert list(p.role).count("free_surface_plate") == nl and list(p.role).count("wetted") == nh
    assert len(m.panels(role="free_surface_plate", include_reflections=False).role) == nl


PLATE = disc(2)
MOVING = {"motion": True}
CYLINDER = {"method": "cylinder", "radius": 2.0, "depth": 1.0, "inner_radius": 1.0, "panel_size": 0.5}
INERTIA = {"mass": 100.0, "center_of_gravity": [0.0, 0.0, 0.0], "radii_of_gyration": [0.5, 0.5, 0.5]}


FREE_EDGE = (r"free-surface plates with the mean drift by pressure integration \(and its QTFs\) are not "
             r"available for a plate with a free edge, one bordered by free surface \(bodies/plate free-surface "
             r"plate 1, element ")


@pytest.mark.parametrize("case, match", [
    # the disc's edge borders the free surface (theory §11.2: no finite line term)
    (dict(options={"drift_pressure_integration": True, **MOVING}), FREE_EDGE),
    (dict(omegas=[1.0, 1.2], options={"qtf_quadratic_pressure_integration": True, **MOVING}), FREE_EDGE),
    (dict(omegas=[1.0, 1.2], options={"qtf_newman": True, "drift_pressure_integration": True, **MOVING}), FREE_EDGE),
    (dict(options={"drift_control_surface": True, **MOVING}, control_surface=CYLINDER),
     r"free-surface plates with the mean drift by the control surface \(and its QTFs\) are not available for a "
     r"plate with a free edge, one bordered by free surface \(bodies/plate free-surface plate 1, element "),
])  # fmt: skip
def test_the_features_not_available_with_plates_are_refused(case, match):
    case = dict(case)
    cs = case.pop("control_surface", None)
    c = {"bodies": plate_body(PLATE, **INERTIA, **({"control_surface": cs} if cs else {})), "omegas": [1.0], **case}
    m = model(**c)
    with pytest.raises(hb.UnsupportedProblemError, match=match):
        m.route()


def test_a_plate_edge_against_a_plate_of_another_body_is_free():
    """The moonpool's cover as two plate bodies (an inner disc and the ring around it, the ring's
    outer edge on the hull's wall): across their common edge the vertical velocity jumps from one
    plate's motion to the other's, as at an edge bordered by free surface (theory §9.7), so the
    pressure drift and the potential QTF refuse it as a free edge; as two plates of one body the
    edge lies between plate elements and the pressure drift is routed."""
    hull = {"name": "hull", "mesh": surface(moonpool_panels(n=8), symmetry=QUARTER), "modes": []}
    inner = surface(disc_panels(0.5, n_r=2, n_phi=8), symmetry=QUARTER)
    ring = surface(disc_panels(1.0, n_r=2, n_phi=8, inner=0.5), symmetry=QUARTER)
    two = [hull, plate_body(inner, name="inner", **MASSLESS), plate_body(ring, name="ring", **MASSLESS)]
    edge = r"plate with a free edge, one bordered by free surface \(bodies/inner free-surface plate 1, element \d+\)"
    for o in ({"drift_pressure_integration": True}, {"qtf_potential_indirect": True}):
        with pytest.raises(hb.UnsupportedProblemError, match=edge):
            model(two, [1.0, 1.2], options={**o, "motion": True}).route()
    one = [hull, {"name": "piston", "modes": ["heave"], **MASSLESS,
                  "free_surface_plates": [{"mesh": inner}, {"mesh": ring}]}]  # fmt: skip
    assert model(one, [1.0], options={**PRESSURE, "motion": True}).route().startswith("low_order")


def test_a_long_wall_does_not_hold_the_edges_of_a_mat_beside_it():
    """A plate edge lies on a wall when its midpoint lies within a tenth of the plate edge's own
    length of the wall (the chords of two meshes of one curve), however long the wall's edge: a mat
    of 2 x 2 m floating 1 m beside the 40 m side of a barge (one panel along it) has edges bordered
    by free surface, refused with the pressure drift."""
    barge = {"name": "barge", "mesh": surface(box_panels(20.0, 5.0, 2.0, n=(1, 2, 2))), "modes": []}
    mat = surface(np.array(face_panels((-1.0, 6.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), 2, 2, (0, 0, -1))))
    m = model([barge, plate_body(mat, name="mat", **MASSLESS)], [1.0], options={**PRESSURE, "motion": True})
    with pytest.raises(hb.UnsupportedProblemError, match=FREE_EDGE.replace("bodies/plate", "bodies/mat")):
        m.route()


def test_the_rules_of_plates():
    # a plate not in z = 0 (the body placed below it)
    with pytest.raises(ValueError, match="free-surface plate 1: an element does not lie in the free surface z = 0"):
        model(plate_body(PLATE, position=[0.0, 0.0, -0.1]), [1.0]).check()
    # a trimmed body, the logarithmic integration
    with pytest.raises(ValueError, match="a body of free-surface plates only cannot be trimmed"):
        model(plate_body(PLATE, trim=[0.0, 1.0, 0.0]), [1.0]).check()
    with pytest.raises(ValueError, match="free-surface plates require options.logarithmic_integration = true"):
        model(plate_body(PLATE), [1.0], options={"logarithmic_integration": False}).check()
    # a heaving plate in finite depth moves water at omega = 0: a net volume flux (rule 9)
    with pytest.raises(ValueError, match="the zero-frequency heave mode has a net volume flux"):
        model(plate_body(PLATE), [0.0, 1.0], options={"depth": 3.0, "diffraction": False}).check()
    # a wetted element in z = 0 names the plate as one of its meanings
    with pytest.raises(ValueError, match="give it as the interior surface, a lid or a free-surface plate"):
        model({"name": "w", "mesh": PLATE, "interior_surface": "none"}, [1.0]).check()


def test_plates_with_wavemakers_are_refused():
    wavemaker = {"name": "wm", "mesh": surface(face_panels((0, 0, -1.0), (0, 1, 0), (0, 0, 1.0), 2, 2, (1, 0, 0))),
                 "modes": ["surge"], "interior_surface": "none"}  # fmt: skip
    with pytest.raises(hb.UnsupportedProblemError, match="free-surface plates with wavemakers in the walls"):
        model([wavemaker, plate_body(PLATE, position=[3.0, 0.0, 0.0])], [1.0],
              options={"domain": "x_wall", "diffraction": False}).route()  # fmt: skip


def test_thin_sheets_in_the_free_surface_are_refused():
    """A dipole, a dipole damper or a fixed damper in z = 0 coincides with its free-surface image
    (its matrix is not finite): refused with the reason before the solve."""
    sheet = "a thin sheet .*lies in the free surface z = 0"
    with pytest.raises(ValueError, match=f"bodies/s: element 1, {sheet}"):
        model({"name": "s", "modes": ["heave"], "dipoles": [{"mesh": PLATE}]}, [1.0]).check()
    cyl, ring = cylinder_panels(1.0, 0.5, 4, 2, 2), disc_panels(3.0, 2, 4, inner=2.0)
    mesh = surface(np.concatenate([cyl, ring]), symmetry=QUARTER)
    body = {"name": "c", "mesh": part(mesh, 1, len(cyl))}
    damper = {"mesh": part(mesh, len(cyl) + 1, len(cyl) + len(ring)), "damping_factor": 1.0}
    with pytest.raises(ValueError, match=f"bodies/c: element {len(cyl) + 1}, {sheet}"):
        model({**body, "dipole_dampers": [damper]}, [1.0]).check()
    fixed = {"mesh": surface(disc_panels(3.0, 2, 4, inner=2.0, quarter=False))}
    with pytest.raises(ValueError, match="fixed_dampers 1: element 1 lies in the free surface z = 0"):
        model({"name": "c", "mesh": surface(cyl, symmetry=QUARTER)}, [1.0], dampers=[fixed]).check()


# ---------------------------------------------------------------- plates of meshes of their own

POINTS = np.array([[0.3, 0.2, 0.0], [0.3, 0.2, -0.05], [3.1, 0.05, 0.0]])  # on, just under, beside the cover
BY_A_COVER = ("inside_waterplane", "near_body", "fluid")


def test_points_on_a_plate_of_a_mesh_of_its_own():
    """The classification of points takes the plates of meshes of their own as those of the body's
    mesh: a point on the cover is inside the waterplane of its body (and refused as a field point),
    a point just under it near the body (its value has the error of the low order near a panel), a
    point beside it in the fluid. Before, a plate of its own mesh was invisible to it (the point on
    it was in the fluid, and solved), and a point just under any plate was not near the body."""
    hull, lid = moonpool_panels(n=8), moonpool_lid_panels(n=8)
    own = {"name": "h", "mesh": surface(hull, symmetry=QUARTER),
           "free_surface_plates": [{"mesh": surface(lid, symmetry=QUARTER)}]}  # fmt: skip
    mesh, nh, nl = moonpool(8)
    records = {"name": "h", "mesh": part(mesh, 1, nh), "free_surface_plates": [{"mesh": part(mesh, nh + 1, nh + nl)}]}
    # a body of plates only, its second plate (beside the first) on a mesh of its own
    second = surface(disc_panels(0.5, 2, 8, quarter=False) + [3.0, 0.0, 0.0], symmetry="none")
    two = {"name": "p", "free_surface_plates": [{"mesh": disc(4)}, {"mesh": second}]}
    for b in (own, records, plate_body(disc(4))):
        assert model(b, [1.0]).classify_points(POINTS).status == BY_A_COVER, b["name"]
    assert model(two, [1.0]).classify_points(POINTS).status == ("inside_waterplane", "near_body", "inside_waterplane")
    with pytest.raises(ValueError, match="field point 1 lies inside the water plane of bodies/h"):
        model({**own, "modes": ["heave"]}, [1.0], field_points=POINTS[:1]).check()


def test_body_points_of_plates_and_pressure_surfaces_of_meshes_of_their_own(tmp_path):
    """Low order: the panels of a plate or a pressure surface on a mesh of its own are numbered on
    after the elements of the body's mesh (reference/results), so that every row of
    body_points_ids names an element >= 1, each once per reflection, and the files are written; the
    numbers of the body's mesh are those of before. Before, they had element 0 and write_wamit
    refused them."""
    hull, lid = moonpool_panels(n=4), moonpool_lid_panels(n=4)
    nh, nl = len(hull), len(lid)
    cover = surface(lid, symmetry=QUARTER)
    for surf in ({"free_surface_plates": [{"mesh": cover}]},
                 {"pressure_surfaces": [{"mesh": cover, "pressure_modes": [{"name": "p"}]}]}):  # fmt: skip
        b = {"name": "h", "mesh": surface(hull, symmetry=QUARTER), "modes": ["heave"], **surf}
        r = solve(b, W[:1], [0.0], outputs={"body_pressure": True})
        ids = r.quantity("body_points_ids")
        assert ids.shape == (4 * (nh + nl), 4)
        for s in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            rec = ids[(ids[:, 2] == s[0]) & (ids[:, 3] == s[1]), 1]
            assert sorted(rec) == list(range(1, nh + nl + 1)), (surf, s)
        r.write_wamit(tmp_path / ("plate" if "free_surface_plates" in surf else "pressure"))
    # the summary lists the plate
    assert f"Free-surface plate 1: {nl} panels of mesh" in (tmp_path / "plate.out").read_text()


def test_a_plate_of_a_mesh_of_its_own_is_the_plate_of_the_body_mesh():
    """Low order: the moonpool cover on a mesh of its own and as elements of the hull's mesh are the
    same problem (the same panels in another order): A, B, X and the body pressure at the cover's
    panels to 1e-5 (float32)."""
    hull, lid = moonpool_panels(n=8), moonpool_lid_panels(n=8)
    mesh, nh, nl = moonpool(8)
    out = []
    for plate, body_mesh in (({"mesh": surface(lid, symmetry=QUARTER)}, surface(hull, symmetry=QUARTER)),
                             ({"mesh": part(mesh, nh + 1, nh + nl)}, part(mesh, 1, nh))):  # fmt: skip
        b = {"name": "h", "mesh": body_mesh, "modes": ["heave", "pitch"], "free_surface_plates": [plate]}
        out.append(solve(b, W, [30.0], outputs={"body_pressure": True}))
    for q in ("added_mass", "radiation_damping", "body_points_pressure_radiation", "body_points_pressure_diffraction"):
        a, b = out[0].quantity(q), out[1].quantity(q)
        assert np.abs(a - b).max() <= 1e-5 * np.abs(a).max(), q
    x = out[0].excitation.diffraction
    assert np.abs(out[1].excitation.diffraction - x).max() <= 1e-5 * np.abs(x).max()


# ---------------------------------------------------------------- velocities by a plate

def test_body_velocity_on_a_plate_and_on_the_hull_beside_it():
    """The fluid velocity at the panels of the moonpool cover (a plate of the hull's body) and of
    the hull's wall below its edge, every component of the radiation (heave, pitch) and diffraction
    potentials, against the field velocity at points a distance d into the water from them: the
    difference falls as d (the gradient of the velocity times d): at d = 0.01, 0.005 of the largest
    velocity 3.7e-2, 1.8e-2 on the cover (its horizontal components 5.4e-3, 2.0e-3) and 2.0e-2,
    1.0e-2 on the wall; the bounds 1.5 times those of 0.005 and a fall by 1.6 at least. (Before only
    the vertical velocity on the plate, that of its boundary condition, was checked.)"""
    mesh, nh, nl = moonpool(8)
    b = {"name": "hull", "mesh": part(mesh, 1, nh), "modes": ["heave", "pitch"],
         "free_surface_plates": [{"mesh": part(mesh, nh + 1, nh + nl)}]}  # fmt: skip
    w = [1.5]
    r = solve(b, w, [30.0], outputs={"body_velocity": True})
    x, n, ids = (r.quantity(q) for q in ("body_points", "body_points_normals", "body_points_ids"))
    v = np.concatenate([r.quantity("body_points_velocity_radiation")[0],
                        r.quantity("body_points_velocity_diffraction")[0]], axis=1)  # fmt: skip
    first = (ids[:, 2] == 1) & (ids[:, 3] == 1)
    cover = np.where(first & (ids[:, 1] > nh))[0]
    wall = np.where(first & (np.hypot(x[:, 0], x[:, 1]) < 1.0) & (np.abs(n[:, 2]) < 1e-6) & (x[:, 2] > -0.2))[0]
    assert len(cover) == nl and len(wall) == 8
    sel, scale = np.r_[cover, wall], np.abs(v).max()
    errs = []
    for d in (0.01, 0.005):
        with pytest.warns(hb.HydroBEMWarning, match="within a panel diameter"):
            f = solve(b, w, [30.0], field_points=x[sel] + d * n[sel], outputs={"field_velocity": True})
        fv = np.concatenate([f.quantity("field_points_velocity_radiation")[0],
                             f.quantity("field_points_velocity_diffraction")[0]], axis=1)  # fmt: skip
        e = np.abs(fv - v[sel]) / scale
        errs.append([e[:nl].max(), e[:nl, :, :2].max(), e[nl:].max()])
    errs = np.array(errs)
    assert np.all(errs[1] <= 1.5 * np.array([1.8e-2, 2.0e-3, 1.0e-2])), errs
    assert np.all(errs[0] >= 1.6 * errs[1]), errs


@pytest.mark.filterwarnings("ignore:field point .* lies within a panel diameter")
def test_field_velocity_under_and_beside_a_plate():
    """The field velocity under the disc (at depths 0.3 and 0.05) and beside it (r = 1.6) against
    the central differences (step 1e-3) of the field pressure p/(rho g) = -(i w / g) phi: the same
    gradient, to 2e-3 of its largest component (2.8e-4 found, the differences of float32 values);
    the radiation (heave, pitch) and the diffraction potentials."""
    b = plate_body(disc(16, graded=True), modes=("heave", "pitch"))
    h, w = 1e-3, 1.0
    for x0 in ([0.3, 0.2, -0.3], [1.6, 0.4, -0.3], [0.5, 0.0, -0.05]):
        pts = [x0] + [list(np.add(x0, s * h * np.eye(3)[k])) for k in range(3) for s in (1, -1)]
        r = solve(b, [w], [30.0], field_points=np.array(pts), outputs={"field_pressure": True, "field_velocity": True})
        for comp in ("radiation", "diffraction"):
            p = r.quantity(f"field_points_pressure_{comp}")[0]
            v = r.quantity(f"field_points_velocity_{comp}")[0][0]  # [K, 3] at x0
            fd = np.array([(p[1 + 2 * k] - p[2 + 2 * k]) / (2 * h) for k in range(3)]).T * (1j * G / w)
            assert np.abs(fd - v).max() <= 2e-3 * np.abs(v).max(), (x0, comp)


def test_body_points_of_a_body_of_plates_only():
    """A body of plates only takes body_points (its plate panels are body panels): at a panel
    centroid the value of that panel."""
    b = plate_body(disc(4))
    r = solve(b, W[:1], [0.0], outputs={"body_pressure": True})
    x, p = r.quantity("body_points"), r.quantity("body_points_pressure_radiation")
    s = solve({**b, "body_points": x[[0, 5]]}, W[:1], [0.0], outputs={"body_pressure": True})
    assert np.abs(s.quantity("body_points_pressure_radiation") - p[:, [0, 5]]).max() <= 1e-6 * np.abs(p).max()


# ---------------------------------------------------------------- plates with lids and pressure surfaces

def test_plates_with_a_lid_and_with_a_pressure_surface():
    """A disc plate in an annulus of z = 0 (1 < r < 2) that is a damping lid or a pressure surface:
    a lid of damping 0 is the free surface (the same results as without it, to 1e-6); a lid of
    damping 1 and a pressure surface (its pressure mode a coordinate) keep A and B symmetric (the
    damping lid is reciprocal) and the Haskind relation of the plate's modes, to the
    discretization of the coarse mesh (Haskind 2.3e-3 with the lid, 4.0e-3 with the pressure
    surface, its A asymmetric by 3.8e-3; 1e-2); a pressure surface at rest leaves the plate's
    columns as without it (1e-6)."""
    d, ring = disc_panels(1.0, 4, 8), disc_panels(2.0, 2, 8, inner=1.0)
    mesh = surface(np.concatenate([d, ring]), symmetry=QUARTER)
    nd, nr = len(d), len(ring)
    base = {"name": "p", "modes": ["heave", "pitch"], "free_surface_plates": [{"mesh": part(mesh, 1, nd)}]}
    o, w2 = {"diffraction_formulation": "total"}, W[:2]  # Ka 0.5, 1: six panels a wavelength
    a = solve(base, w2, [0.0, 30.0], options=o)

    def close(r, rows=slice(None), cols=slice(None)):
        for q in ("added_mass", "radiation_damping"):
            x, y = r.quantity(q)[:, rows][:, :, cols], a.quantity(q)
            assert np.abs(x - y).max() <= 1e-6 * np.abs(y).max(), q

    lid = {"name": "L", "mesh": part(mesh, nd + 1, nd + nr)}
    close(solve(base, w2, [0.0, 30.0], options=o, lids=[{**lid, "damping_factor": 0.0}]))
    damped = solve(base, w2, [0.0, 30.0], options=o, lids=[{**lid, "damping_factor": 1.0}])
    surfaces = [{"mesh": part(mesh, nd + 1, nd + nr), "pressure_modes": [{"name": "q"}]}]
    ps = solve({**base, "pressure_surfaces": surfaces},
               w2, [0.0, 30.0], options=o)  # fmt: skip
    close(ps, rows=[0, 1, 2, 3, 4, 5], cols=[0, 1])
    for r, coords in ((damped, [2, 4]), (ps, [2, 4, 6])):
        for q in ("added_mass", "radiation_damping"):
            x = r.quantity(q)[:, coords]
            assert np.abs(x - np.swapaxes(x, 1, 2)).max() <= 1e-2 * np.abs(x).max(), q
        X = r.excitation.diffraction[:, :, coords]
        assert np.abs(r.excitation.haskind - X).max() <= 1e-2 * np.abs(X).max()


# ---------------------------------------------------------------- the surfaces of z = 0 and the sheets

def test_overlaps_of_plates_are_refused():
    """A plate overlaps no other surface of z = 0 (reference/inputs-rules §1 rule 3): a plate over the generated
    interior surface of its hull, over a lid, two overlapping plate bodies, two overlapping plates of
    one body, a plate over a pressure surface in z = 0. Before, each solved with numbers of neither
    surface. Touching along edges is accepted (the cover in the moonpool, the tests above), and two
    lids may still overlap."""
    cyl = surface(cylinder_panels(1.0, 0.5, n_phi=8, n_z=2, n_r=4), symmetry=QUARTER)
    big, small = disc(4), surface(disc_panels(0.5, 2, 8), symmetry=QUARTER)
    over = "free-surface plate 1 overlaps {} in the free surface z = 0"
    cases = [
        (dict(bodies={"name": "c", "mesh": cyl, "interior_surface": "graded", "free_surface_plates": [{"mesh": big}]}),
         over.format("bodies/c interior surface")),
        (dict(bodies=plate_body(big, name="p"), lids=[{"name": "L", "mesh": small, "damping_factor": 1.0}]),
         over.format("lids/L")),
        (dict(bodies=[plate_body(big, name="p"), plate_body(small, name="q")]),
         over.format("bodies/[pq] free-surface plate 1")),
        (dict(bodies={"name": "p", "modes": ["heave"], "free_surface_plates": [{"mesh": big}, {"mesh": small}]}),
         "free-surface plate [12] overlaps bodies/p free-surface plate [12]"),
        (dict(bodies={**plate_body(big, name="p"),
                      "pressure_surfaces": [{"mesh": small, "pressure_modes": [{"name": "q"}]}]}),
         over.format("bodies/p pressure surface 1")),
    ]  # fmt: skip
    for c, match in cases:
        with pytest.raises(ValueError, match=match):
            model(omegas=[1.0], options={"diffraction": False}, **c).check()
    ring = surface(cylinder_panels(3.0, 0.5, n_phi=8, n_z=2, n_r=4), symmetry=QUARTER)
    lids = [{"name": "L", "mesh": disc(4), "damping_factor": 1.0}, {"name": "M", "mesh": small, "damping_factor": 1.0}]
    model({"name": "c", "mesh": ring, "modes": ["heave"], "interior_surface": "none"}, [1.0], lids=lids,
          options={"diffraction": False}).check()  # fmt: skip


def skirt_panels(radius, depth, n_phi, n_z):
    """A vertical cylindrical sheet (a skirt) of `radius` from z = 0 down to -depth, a quadrant."""
    ph, z = np.linspace(0, np.pi / 2, n_phi + 1), np.linspace(0.0, -depth, n_z + 1)

    def p(k, j):
        return [radius * np.cos(ph[k]), radius * np.sin(ph[k]), z[j]]

    return np.array([[p(i, j), p(i, j + 1), p(i + 1, j + 1), p(i + 1, j)] for i in range(n_phi) for j in range(n_z)])


def bell(n, conforming, lid=False):
    """A disc of radius 1 with a skirt of radius 0.7 and depth 0.5 hanging from it: the plate's
    elements ending on the skirt (a disc r < 0.7 and an annulus), or rings graded towards the edge
    that straddle it; or the disc a rigid lid (damping 1e9) with the skirt a body of its own."""
    s = surface(skirt_panels(0.7, 0.5, 16 * n, 6 * n), symmetry=QUARTER)
    if conforming:
        pl = np.concatenate([disc_panels(0.7, 6 * n, 16 * n), disc_panels(1.0, 4 * n, 16 * n, graded=True, inner=0.7)])
    else:
        pl = disc_panels(1.0, 8 * n, 16 * n, graded=True)
    dm = surface(pl, symmetry=QUARTER)
    if lid:
        return {"bodies": {"name": "bell", "modes": ["surge"], "dipoles": [{"mesh": s}]},
                "lids": [{"name": "L", "mesh": dm, "damping_factor": 1e9}]}  # fmt: skip
    return {"bodies": {"name": "bell", "modes": ["heave"], "dipoles": [{"mesh": s}],
                       "free_surface_plates": [{"mesh": dm}]}}  # fmt: skip


def test_a_sheet_hanging_from_a_plate_or_a_lid():
    """A skirt hanging from a plate (an oscillating water column's chamber under its piston): the
    potential of the plate jumps across the junction (theory §9.6), which an element straddling it
    cannot hold: the Haskind relation and the energy of the radiated wave (B33 = w^3 |X3|^2 /
    (2 rho g^3)) then stall at 1.1 % at Ka = 2 for every mesh. With the plate's elements ending on
    the skirt both hold and converge as h^2: 2.1e-4, 4.7e-5 at Ka = 2 (8.8e-5, 2.1e-5 at Ka = 1)
    for n = 1, 2. The straddling plate and the same with a rigid damping lid are refused, naming
    the element."""
    errs = []
    for n in (1, 2):
        r = solve(**bell(n, True), omegas=W, headings=[0.0])
        X, H = r.excitation.diffraction[:, 0, 2], r.excitation.haskind[:, 0, 0]
        B = r.quantity("radiation_damping")[:, 2, 0]
        errs.append([np.abs(H / X - 1.0).max(), np.abs(B / (W**3 * np.abs(X) ** 2 / (2 * RHO * G**3)) - 1.0).max()])
    errs = np.array(errs)
    assert np.all(errs[0] <= 4e-4) and np.all(errs[1] <= 1e-4) and np.all(errs[0] >= 3.0 * errs[1]), errs
    junction = "a thin sheet, meets {} inside its element .*: the potential jumps across the sheet"
    with pytest.raises(ValueError, match=junction.format("bodies/bell free-surface plate 1")):
        model(**bell(1, False), omegas=[1.0]).check()
    with pytest.raises(ValueError, match=junction.format("lids/L")):
        model(**bell(1, False, lid=True), omegas=[1.0]).check()
    model(**bell(1, True, lid=True), omegas=[1.0]).check()


def test_a_sheet_hanging_from_a_plate_patch():
    """The higher order: a vertical flat sheet x = 0.5 (depth 0.5) under the square plate of side 2
    (a quadrant): the plate as two patches meeting on the sheet's edge converges (the Haskind
    relation 2.5e-4, 6.4e-5, 1.7e-5 at Ka = 2 on 1, 2, 4 divisions), the plate as one patch across
    it is refused."""
    def body(conforming, n):
        if conforming:
            plate = [*face_panels((0, 0, 0), (0.5, 0, 0), (0, 1, 0), 1, 1, (0, 0, -1)),
                     *face_panels((0.5, 0, 0), (0.5, 0, 0), (0, 1, 0), 1, 1, (0, 0, -1))]  # fmt: skip
            rows = [(n, 2 * n, 3, 3, 3, 4)] * 2
        else:
            plate, rows = face_panels((0, 0, 0), (1, 0, 0), (0, 1, 0), 1, 1, (0, 0, -1)), [(2 * n, 2 * n, 3, 3, 3, 4)]
        skirt = face_panels((0.5, 0, -0.5), (0, 1, 0), (0, 0, 0.5), 1, 1, (1, 0, 0))
        mesh = flat(np.array(plate + skirt), rows + [(2 * n, n, 3, 3, 3, 4)], symmetry=QUARTER)
        k = len(plate)
        return {"name": "bell", "modes": ["heave"], "free_surface_plates": [{"mesh": part(mesh, 1, k)}],
                "dipoles": [{"mesh": part(mesh, k + 1, k + 1)}]}  # fmt: skip

    errs = []
    for n in (1, 2):
        r = solve(body(True, n), W, [0.0], options={"higher_order": True})
        errs.append(np.abs(r.excitation.haskind[:, 0, 0] / r.excitation.diffraction[:, 0, 2] - 1.0).max())
    assert errs[0] <= 5e-4 and errs[1] <= 1.5e-4 and errs[1] < errs[0] / 2, errs
    sheet = "bodies/bell: element 2, a thin sheet, meets bodies/bell free-surface plate 1 inside"
    with pytest.raises(ValueError, match=sheet):
        model(body(False, 1), [1.0], options={"higher_order": True}).check()


def owc(n):
    """The quadrant of a deep oscillating water column: a tube of inner radius 2, outer radius 3 and
    draft 10 (4 n panels around, 20 n down the outer wall, 2 n across the bottom, 18 n up the inner
    wall), its interior surface the annulus of its waterplane, and the light piston on the water
    inside it, a body of the disc r < 2 in z = 0 (6 n uniform rings)."""
    phi = np.linspace(0.0, 0.5 * np.pi, 4 * n + 1)
    r = np.r_[np.full(20 * n + 1, 3.0), np.linspace(3.0, 2.0, 2 * n + 1)[1:], np.full(18 * n, 2.0)]
    z = np.r_[np.linspace(0.0, -10.0, 20 * n + 1), np.full(2 * n, -10.0), np.linspace(-10.0, 0.0, 18 * n + 1)[1:]]

    def p(i, j):
        return [r[i] * np.cos(phi[j]), r[i] * np.sin(phi[j]), z[i]]

    tube = [[p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)] for i in range(len(r) - 1) for j in range(4 * n)]
    hull = {"name": "hull", "mesh": surface(np.array(tube), symmetry=QUARTER), "modes": ["heave"],
            "interior_surface": "projected"}  # fmt: skip
    cover = surface(disc_panels(2.0, n_r=6 * n, n_phi=4 * n), symmetry=QUARTER)
    return [hull, plate_body(cover, name="piston")]


def test_the_formulation_under_a_plate_on_a_deep_water_column():
    """The piston of a deep water column at omega = 1.4 (KH = 2): the column shields the plate from
    the incident wave, the total potential on it is the wave at the open bottom, the exciting force
    a tenth of its Froude-Krylov force (0.093, 0.085 found at n = 1, 2; 0.12), and the scattered potential there nearly
    -phi_I. Both formulations converge as h^2 to the same force (the Richardson extrapolations of
    n = 1, 2: 0.9 % apart found; 2 %), but the scattered one's error, carried by the incident wave's
    variation down the inner wall, is about four times the total one's (0.26, 0.25 of it found at
    n = 1, 2, each against the extrapolation of the total formulation; 0.4). So "auto" solves a
    model with a plate in the total formulation (theory §9.7): identical to "total"."""
    w, x = [1.4], {}
    for n in (1, 2):
        for form in ("scattered", "total"):
            r = solve(owc(n), w, [0.0], options={"diffraction_formulation": form})
            x[n, form] = r.coefficient("excitation_diffraction", ("piston", "heave"))[0, 0]
            if form == "total":
                fk = r.coefficient("excitation_froude_krylov", ("piston", "heave"))[0, 0]
                assert abs(x[n, form]) <= 0.12 * abs(fk), abs(x[n, form] / fk)
    ext = {f: x[2, f] + (x[2, f] - x[1, f]) / 3.0 for f in ("scattered", "total")}
    assert abs(ext["scattered"] / ext["total"] - 1.0) <= 2e-2, ext
    for n in (1, 2):
        ratio = abs(x[n, "total"] - ext["total"]) / abs(x[n, "scattered"] - ext["total"])
        assert ratio <= 0.4, (n, ratio)
    auto = solve(owc(1), w, [0.0])
    total = solve(owc(1), w, [0.0], options={"diffraction_formulation": "total"})
    for q in ("added_mass", "radiation_damping", "excitation_diffraction"):
        assert view(auto, q).tobytes() == view(total, q).tobytes(), q


# ---------------------------------------------------------------- a channel and tanks

CHANNEL = {"domain": "channel", "channel_width": 8.0, "diffraction_formulation": "total"}
W_CH = np.sqrt(np.array([0.6, 1.5]) * G)
COEFFICIENTS = ("added_mass", "radiation_damping", "excitation_diffraction")
SIGNS = np.array([1, -1, 1, -1, 1, -1])  # mirror about y = 0: sway, roll and yaw change sign
RIGID = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def _mirror(r, q):
    """The quantity q of the mirror image about y = 0 of a body of six coordinates."""
    s = SIGNS if q == "excitation_diffraction" else SIGNS[None, :, None] * SIGNS[None, None, :]
    return view(r, q) * s


def _covered(y, options, reflect=None, own_body=False, points=None):
    """The moonpool hull with its cover (a plate of its body, or a body of its own moving in the
    same modes) y off the centreline in water of depth 10, headings 0 and 180."""
    mesh, nh, nl = moonpool(8)
    hull = {"name": "hull", "mesh": part(mesh, 1, nh), "position": [0.0, y, 0.0]}
    cover = part(mesh, nh + 1, nh + nl)
    if own_body:
        bodies = [hull, plate_body(cover, name="lid", modes=RIGID, position=[0.0, y, 0.0])]
    else:
        bodies = {**hull, "free_surface_plates": [{"mesh": cover}]}
    env = {"depth": 10.0, **({"channel_reflect": reflect} if reflect is not None else {})}
    return solve(bodies, W_CH, [0.0, 180.0], environment=env, field_points=points, options=options)


def _wall_condition(r, d):
    """The wall condition dphi/dy = 0 at y = +-4 from each group of points at the distances d: the
    slope at the wall of the field pressure (a quadratic fit in d), of k times its peak, and v_y
    at the wall (the value of the fit), of the peak of the velocity. Returns the largest of each."""
    fp, worst = r.field_points, [0.0, 0.0]
    for f, w in enumerate(W_CH):
        k = w * w / G  # k h >= 3.7: tanh(k h) = 1 to 1.3e-3
        for v in (fp.pressure_radiation[f, :, 1], fp.pressure_radiation[f, :, 2], fp.pressure_diffraction[f, :, 0]):
            for b in range(len(v) // len(d)):
                c = np.polyfit(d, v[len(d) * b : len(d) * (b + 1)], 2)
                worst[0] = max(worst[0], abs(c[1]) / (k * peak(v)))
        for v in (fp.velocity_radiation[f, :, 1], fp.velocity_radiation[f, :, 2], fp.velocity_diffraction[f, :, 0]):
            for b in range(len(v) // len(d)):
                c = np.polyfit(d, v[len(d) * b : len(d) * (b + 1), 1], 2)
                worst[1] = max(worst[1], abs(c[2]) / peak(v))
    return worst


def test_plates_in_a_channel():
    """The moonpool cover 1.5 m off the centreline of a channel 8 m wide in 10 m depth (low order),
    each check an identity of the problem: no reflection (channel_reflect = 0) is the open sea and
    the mirror body is mirrored (1.8e-7 and 1.0e-6 found; 1e-5: float32 matrices, the image sums in
    another order); the walls change the coefficients by 14 to 90 % (so that the checks see them);
    the cover as a body of its own moving with the hull is the cover of the hull's body (the sums
    of the blocks; 1e-5); the asymmetry of A and B and the Haskind relation are those of the open
    sea (4.1e-3, 9.9e-3, 9.9e-3 against 2.8e-3, 9.3e-3, 9.9e-3: the corner of the cover on the
    moonpool wall on 8 panels a quadrant; 1.5 times plus 1e-4); the wall condition dphi/dy = 0 of
    the field pressure and of the velocity by the source formulation at points by the walls (the
    pressure's slope at the wall of a fit in the distance, of k times its peak, 4.8e-5 found; v_y at
    the wall, of the velocity's peak, 9.5e-6; 1e-4 and 2e-4: the truncated far-image sums)."""
    d = np.array([0.01, 0.02, 0.03, 0.04])
    pts = np.array([(x0, sg * (4.0 - di), z) for x0, z in ((0.0, 0.0), (5.0, -0.5)) for sg in (1, -1) for di in d])
    ch = _covered(1.5, {**CHANNEL, "field_pressure": True, "field_velocity": True}, points=pts)
    slope, vy = _wall_condition(ch, d)
    assert slope <= 1e-4 and vy <= 2e-4, (slope, vy)
    sea, ch0 = _covered(1.5, {"diffraction_formulation": "total"}), _covered(1.5, CHANNEL, reflect=0.0)
    mirror = _covered(-1.5, CHANNEL)
    for q in COEFFICIENTS:
        assert peak(view(ch0, q) - view(sea, q)) <= 1e-5 * peak(view(sea, q)), q
        assert peak(view(ch, q) - _mirror(mirror, q)) <= 1e-5 * peak(view(ch, q)), q
        assert peak(view(ch, q) - view(sea, q)) >= 0.1 * peak(view(sea, q)), q
    two = _covered(1.5, CHANNEL, own_body=True)
    A1, A2 = ch.added_mass, two.added_mass  # two: coordinates hull 0..5, lid 6..11, columns likewise
    assert peak(A1 - (A2[:, :6, :6] + A2[:, :6, 6:] + A2[:, 6:, :6] + A2[:, 6:, 6:])) <= 1e-5 * peak(A1)
    x2 = two.excitation.diffraction
    assert peak(ch.excitation.diffraction - x2[..., :6] - x2[..., 6:]) <= 1e-5 * peak(x2)
    level = []
    for r in (ch, sea):
        A, B, X = r.added_mass, r.radiation_damping, r.excitation.diffraction
        level.append([peak(A - A.transpose(0, 2, 1)) / peak(A), peak(B - B.transpose(0, 2, 1)) / peak(B),
                      peak(r.excitation.haskind - X) / peak(X)])  # fmt: skip
    level = np.array(level)
    assert np.all(level[0] <= 1.5 * level[1] + 1e-4), level


def test_a_wide_channel_at_zero_frequency_tends_to_the_open_sea():
    """At omega = 0 the free surface is the rigid plane z = 0, and a rolling and pitching disc in a
    channel of depth 3 sees its wall images as dipoles of the layer between two rigid planes: the
    channel's added mass tends to the open sea's as 1/W^2 (the field of the dipole images at the
    distances n W falls as 1/(n W)^2). Widths 6, 12, 24, 48: 6.2e-3, 1.4e-3, 3.4e-4, 8.6e-5 of the
    peak in roll, twice that in pitch found, the ratio 4.0 to 4.5 for each doubling (3.5 to 4.6)."""
    body = plate_body(disc(8), name="d", modes=("roll", "pitch"))
    opts = {"diffraction": False}
    sea = solve(body, [0.0], [], depth=3.0, options=opts).quantity("added_mass_zero_frequency")
    diff = []
    for width in (6.0, 12.0, 24.0, 48.0):
        ch = solve(body, [0.0], [], depth=3.0, options={**opts, "domain": "channel", "channel_width": width})
        a = ch.quantity("added_mass_zero_frequency")
        diff.append([abs(a[3, 0] - sea[3, 0]), abs(a[4, 1] - sea[4, 1])])
    ratio = np.array(diff[:-1]) / np.array(diff[1:])
    assert np.all((ratio >= 3.5) & (ratio <= 4.6)), ratio
    assert np.all(np.array(diff[-1]) <= 3e-4 * peak(sea)), diff


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_higher_order_plates_in_a_channel():
    """The moonpool barge of flat patches with its cover (element 7) 0.5 m off the centreline of the
    channel 8 m wide in 10 m depth: no reflection is the open sea (2.9e-11 found; 1e-9: the same
    equations in double precision) and the mirror body is mirrored (1e-9); the walls change the
    coefficients by 30 to 47 %; A and B symmetric (1.2e-4, 4.5e-5 found) and the Haskind relation
    (6.9e-5) within 1e-3."""
    mb = moonpool_barge()

    def barge(y, opts, reflect=None):
        env = {"depth": 10.0, **({"channel_reflect": reflect} if reflect is not None else {})}
        body = {"name": "b", "mesh": part(mb, 1, 6), "position": [0.0, y, 0.0],
                "free_surface_plates": [{"mesh": part(mb, 7, 7)}]}  # fmt: skip
        return solve(body, W_CH, [0.0, 180.0], environment=env, options={"higher_order": True, **opts})

    ch, sea = barge(0.5, CHANNEL), barge(0.5, {})
    ch0, mirror = barge(0.5, CHANNEL, 0.0), barge(-0.5, CHANNEL)
    for q in COEFFICIENTS:
        assert peak(view(ch0, q) - view(sea, q)) <= 1e-9 * peak(view(sea, q)), q
        assert peak(view(ch, q) - _mirror(mirror, q)) <= 1e-9 * peak(view(ch, q)), q
        assert peak(view(ch, q) - view(sea, q)) >= 0.2 * peak(view(sea, q)), q
    A, B, X = ch.added_mass, ch.radiation_damping, ch.excitation.diffraction
    assert peak(A - A.transpose(0, 2, 1)) <= 1e-3 * peak(A) and peak(B - B.transpose(0, 2, 1)) <= 1e-3 * peak(B)
    assert peak(ch.excitation.haskind - X) <= 1e-3 * peak(X)


MAT = surface(face_panels((1.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.5, 0.0), 4, 2, (0, 0, -1)), symmetry=QUARTER)
TANK_FIELD = np.array([[0.0, 0.0, -0.3], [0.2, 0.1, -0.5]])  # in the tank of tank_vessel (body axes)


def _vessel(tank, plates, sea_points):
    """The box barge with its internal tank (tank: its density ratio 1, else 0, which removes it
    from the problem) and with or without two mats beside it (plates of a mesh of their own,
    1.5 <= |x| <= 2.5, |y| <= 0.5), free in its six modes."""
    b = tank_vessel(density=1.0 if tank else 0.0)
    b = {**b, "name": "v", "center_of_gravity": [0.0, 0.0, -0.2], "radii_of_gyration": [0.4, 0.5, 0.5],
         "tanks": [{**b["tanks"][0], **({"field_points": TANK_FIELD} if tank else {})}]}  # fmt: skip
    if plates:
        b["free_surface_plates"] = [{"mesh": MAT}]
    return solve(b, np.array([1.0, 2.0, 3.0]), [0.0, 45.0], field_points=sea_points,
                 options={"motion": True, "field_pressure": True})  # fmt: skip


def test_a_plate_on_a_body_with_tanks():
    """The tank problem does not see the sea's plates, nor the sea problem the tank (theory §10.2):
    the tank's part of A and B (the vessel with its tank less the vessel with the tank removed) is
    the same with and without the mats (8.8e-8 of A found; 1e-6), the exciting force and the sea's
    field pressure with the tank are those without it (bit-identical), the pressure in the tank
    does not depend on the mats (1e-6), and the motions take both (the mats change them by 34 %)."""
    sea_points = np.array([[4.0, 1.0, 0.0], [0.0, 3.0, -0.4]])
    r = {(t, p): _vessel(t, p, sea_points) for t in (0, 1) for p in (0, 1)}
    for q in ("added_mass", "radiation_damping"):
        with_mats = view(r[1, 1], q) - view(r[0, 1], q)
        without = view(r[1, 0], q) - view(r[0, 0], q)
        assert peak(with_mats - without) <= 1e-6 * peak(view(r[1, 1], q)), q
    np.testing.assert_array_equal(r[1, 1].excitation.diffraction, r[0, 1].excitation.diffraction)
    for key in ("pressure_radiation", "pressure_diffraction"):
        np.testing.assert_array_equal(r[1, 1].quantity("field_points_" + key), r[0, 1].quantity("field_points_" + key))
    t1, t0 = (r[1, p].quantity("tanks_pressure_radiation") for p in (1, 0))
    assert peak(t1 - t0) <= 1e-6 * peak(t0)
    assert np.isfinite(r[1, 1].motion).all() and peak(r[1, 1].motion - r[1, 0].motion) >= 0.1 * peak(r[1, 0].motion)


# ---------------------------------------------------------------- the mean drift by momentum conservation

DRIFT = {"drift_momentum_conservation": True}
MASSLESS = {"mass": 0.0, "center_of_gravity": [0.0, 0.0, 0.0], "radii_of_gyration": [0.3, 0.3, 0.3]}
W_MP = np.sqrt(np.array([0.5, 1.0, 1.5]) * G / 2.0)  # K a = 0.25, 0.5, 0.75 of a hull of half width 2


def _b2(r, f, values):
    """|b|^2 of Kochin values (theory §11.1)."""
    k, C, _ = drift_constants(float(r.omegas[f]), G, r.model.environment.depth)
    return (C / (4 * math.pi)) ** 2 * 2 / (math.pi * k) * np.abs(values) ** 2


def _energy(r, cols):
    """The flux of the radiated far field of each (coordinate, column) against the damping,
    |rho omega K_e / (2 k) int |b|^2 dtheta / B - 1| at every frequency."""
    hr, out = r.kochin.radiation, []
    for f, w in enumerate(r.omegas):
        k, _, Ke = drift_constants(float(w), G, r.model.environment.depth)
        for j, c in cols:
            flux = RHO * w * Ke / (2 * k) * np.sum(_b2(r, f, hr[f, :, c])) * 2 * math.pi / hr.shape[1]
            out.append(abs(flux / r.radiation_damping[f, j, c] - 1.0))
    return np.array(out)


def _energy_form(r, coords=None):
    """The largest difference, of rho g (the unit radius), between the returned drift force and
    the flux of the total disturbance alone, rho K_e / 4 int |b'|^2 (k_beta - (cos, sin) theta)
    dtheta with b' of phi_S + i omega sum_j xi_j phi_j (the optical theorem of a body without
    dissipation, theory §11.1); coords: the coordinates of the radiation columns."""
    th, hd, hr, worst = r.kochin.angles, r.kochin.diffraction, r.kochin.radiation, 0.0
    for f, w in enumerate(r.omegas):
        _, _, Ke = drift_constants(float(w), G, r.model.environment.depth)
        for h, beta in enumerate(np.radians(r.headings)):
            H = hd[f, :, h]
            if coords:
                H = H + (1j * w * hr[f] * r.motion[f, h, coords][None, :]).sum(axis=1)
            e = _b2(r, f, H) * 2 * math.pi / th.size
            F = RHO * Ke / 4 * np.array([np.sum(e * (math.cos(beta) - np.cos(th))),
                                         np.sum(e * (math.sin(beta) - np.sin(th)))])  # fmt: skip
            worst = max(worst, np.abs(r.drift.momentum_conservation.real[f, h, :2] - F).max() / (RHO * G))
    return worst


@pytest.mark.parametrize("depth", [math.inf, 3.0])
def test_the_far_field_of_a_radiating_plate_carries_its_damping(depth):
    """The energy balance (Newman 1977): the flux of the radiated far field is the damping, here
    through the plate's term -K S of the Kochin function (theory §11.1) on a body of a plate only.
    Low order, the disc graded towards its edge, 8 and 16 rings a quadrant: heave and pitch 1.6e-4
    and 4.1e-5 found at K a = 2 (O(h^2)); higher order, the square plate of side 2 on 2 and 4 cubic
    panels a side: 4.3e-5 and 4.8e-6 (O(h^3)); the same in depth 3. Bounds twice those."""
    for n, bound in ((8, 3.2e-4), (16, 8e-5)):
        b = plate_body(disc(n, graded=True), modes=("heave", "pitch"), **MASSLESS)
        r = solve(b, W, [0.0], depth=depth, options={**DRIFT, "motion": True})
        assert np.all(_energy(r, [(2, 0), (4, 1)]) <= bound), (n, _energy(r, [(2, 0), (4, 1)]))
    q = face_panels((0, 0, 0), (1, 0, 0), (0, 1, 0), 1, 1, (0, 0, -1))
    for n, bound in ((2, 9e-5), (4, 1e-5)):
        b = plate_body(flat(np.array(q), [(n, n, 3, 3, 3, 4)], symmetry=QUARTER), modes=("heave", "pitch"), **MASSLESS)
        r = solve(b, W, [0.0], depth=depth, options={**DRIFT, "motion": True, "higher_order": True})
        assert np.all(_energy(r, [(2, 0), (4, 1)]) <= bound), (n, _energy(r, [(2, 0), (4, 1)]))


def test_the_drift_of_a_disc_conserves_energy():
    """A disc without dissipation, fixed and floating (massless in heave, roll and pitch), headings
    0 and 30: the drift's general form (11.5) equals the flux of the total disturbance alone (the
    optical theorem; 1.5e-6 rho g found, the float32 Kochin samples; 1e-5 rho g); the scattered
    and the total formulation give the same drift (1.7e-6 of its peak found; 1e-5), and the
    quadrant the full mesh (7.8e-7; 1e-5); no sway or yaw drift in head seas (1e-6). The floating
    disc follows the wave: its drift is 6e-7, 3e-4 and 6 % of the fixed one's at K a = 0.5, 1, 2.
    The fixed disc's drift converges as h^2: 1040.4, 1048.4, 1050.5 N/m^2 at K a = 0.5 on 4, 8, 16
    rings (the ratio of the steps 3.9 to 4.0; 3 to 5)."""
    fixed = []
    for n in (4, 8, 16):
        r = solve(plate_body(disc(n, graded=True), modes=()), W, [0.0, 30.0], options=DRIFT)
        fixed.append(r.drift.momentum_conservation.real[:, 0, 0])
        if n == 8:
            assert _energy_form(r) <= 1e-5
    steps = np.abs(np.diff(np.array(fixed), axis=0))
    assert np.all((steps[0] / steps[1] >= 3.0) & (steps[0] / steps[1] <= 5.0)), steps
    out = {}
    for quarter, form in ((True, "total"), (True, "scattered"), (False, "total")):
        b = plate_body(disc(8, quarter=quarter, graded=True), modes=("heave", "roll", "pitch"), **MASSLESS)
        opts = {**DRIFT, "motion": True, "diffraction_formulation": form}
        out[quarter, form] = solve(b, W, [0.0, 30.0], options=opts)
    r = out[True, "total"]
    d = r.drift.momentum_conservation.real
    assert _energy_form(r, [2, 3, 4]) <= 1e-5
    for other in (out[True, "scattered"], out[False, "total"]):
        assert peak(other.drift.momentum_conservation.real - d) <= 1e-5 * peak(d)
    assert np.abs(d[:, 0, 1:]).max() <= 1e-6 * np.abs(d[:, 0, 0]).max()
    assert np.all(d[:, 0, 0] < 0.1 * fixed[1])


def test_the_drift_of_a_wetted_disc_of_small_draft_tends_to_the_plate():
    """A fixed truncated cylinder of radius 1 and draft d (16 x 16 panels a quadrant, its interior
    surface graded) against the fixed plate: the momentum drift of the ordinary body (without the
    plate's column) tends to the plate's. At K a = 0.5 and 1: 0.49 %, 0.22 %, 0.08 % and 0.82 %,
    0.32 %, 0.05 % for d = 0.02, 0.01, 0.005 found, falling with d; at K a = 2 0.14 %, 0.67 %,
    1.2 % (the side wall, one row of panels of height d beside the bottom's of 1/16, loses accuracy
    as d falls, as A33 of the same bodies does). Bounds 1.5 % at d = 0.005 and the fall at K a =
    0.5 and 1."""
    n = 16
    p = solve(plate_body(disc(n), modes=()), W, [0.0], options=DRIFT).drift.momentum_conservation.real[:, 0, 0]
    errs = []
    for d in (0.02, 0.01, 0.005):
        c = surface(cylinder_panels(1.0, d, n_phi=2 * n, n_z=1, n_r=n), symmetry=QUARTER)
        r = solve({"mesh": c, "modes": [], "interior_surface": "graded"}, W, [0.0], options=DRIFT)
        errs.append(np.abs(r.drift.momentum_conservation.real[:, 0, 0] / p - 1.0))
    errs = np.array(errs)
    assert np.all(errs[-1] <= 1.5e-2), errs
    assert np.all(np.diff(errs[:, :2], axis=0) < 0), errs


def _plugged(d, n=8):
    """The moonpool hull of moonpool(n) closed by a wetted disc at z = -d (its normal down), its
    inner wall above the disc dry: an ordinary body."""
    keep = [q for q in moonpool_panels(n=n) if not np.allclose(np.hypot(q[:, 0], q[:, 1]), 1.0)]
    ph, zs = np.linspace(0.0, 0.5 * math.pi, n + 1), np.linspace(-d, -1.0, n // 2 + 1)

    def p(k, z):
        return [math.cos(ph[k]), math.sin(ph[k]), z]

    wall = [[p(i, zs[j]), p(i + 1, zs[j]), p(i + 1, zs[j + 1]), p(i, zs[j + 1])]
            for i in range(n) for j in range(n // 2)]  # fmt: skip
    plug = moonpool_lid_panels(n=n)[:, ::-1] + [0.0, 0.0, -d]
    return surface(np.concatenate([np.array(keep), np.array(wall), plug]), symmetry=QUARTER)


def test_the_drift_of_a_plugged_moonpool_tends_to_the_covered_one():
    """The fixed moonpool hull with its cover (a plate whose edge lies on the wall) against the same
    hull plugged by a wetted disc at z = -d (an ordinary body, without the plate's column): the
    plugged drift tends to the cover's, 1.2e-3, 1.0e-3, 8.4e-4 at K a = 0.25 for d = 0.1, 0.05,
    0.025 found (2.8e-3 to 1.9e-3 and 2.3e-3 to 9.0e-4 at K a = 0.5, 0.75: falling with d, to the
    difference of the two meshes of 8 panels a quadrant). Bounds 3e-3 and the fall with d."""
    mesh, nh, nl = moonpool(8)
    cover = {"name": "hull", "mesh": part(mesh, 1, nh), "modes": [],
             "free_surface_plates": [{"mesh": part(mesh, nh + 1, nh + nl)}]}  # fmt: skip
    c = solve(cover, W_MP, [0.0], options=DRIFT).drift.momentum_conservation.real[:, 0, 0]
    errs = []
    for d in (0.1, 0.05, 0.025):
        plug = {"name": "plug", "mesh": _plugged(d), "modes": [], "interior_surface": "none"}
        r = solve(plug, W_MP, [0.0], options=DRIFT)
        errs.append(np.abs(r.drift.momentum_conservation.real[:, 0, 0] / c - 1.0))
    errs = np.array(errs)
    assert np.all(errs <= 3e-3) and np.all(np.diff(errs, axis=0) < 0), errs


def test_the_drift_of_a_piston_mode_is_that_of_a_piston_body_and_newman():
    """The restrained moonpool hull with its cover moved by a massless generalized piston mode
    against the hull with the cover as a massless body of its own in heave: the same displacement
    of every surface, so the same far field and drift (7.6e-16 found; 1e-10). Newman's
    approximation of the momentum drift (theory §12.13) has the drift on its diagonal (exactly)."""
    mesh, nh, nl = moonpool(8)
    piston = {"name": "piston", "parity": "heave", "pieces": [{"w": [[1.0, 0, 0, 0]], "ranges": [(nh + 1, nh + nl)]}]}
    hull = {"name": "hull", "mesh": part(mesh, 1, nh), "modes": [], **MASSLESS}
    gen = {**hull, "free_surface_plates": [{"mesh": part(mesh, nh + 1, nh + nl)}], "generalized_modes": [piston],
           "mode_mass": np.zeros((7, 7))}  # fmt: skip
    gen = solve(gen, W_MP, [0.0, 30.0], options={**DRIFT, "motion": True, "qtf_newman": True})
    two = solve([hull, plate_body(part(mesh, nh + 1, nh + nl), name="lid", **MASSLESS)], W_MP, [0.0, 30.0],
                options={**DRIFT, "motion": True})  # fmt: skip
    dg, dt = gen.drift.momentum_conservation.real, two.drift.momentum_conservation.real
    assert peak(dg - dt) <= 1e-10 * peak(dt)
    nm = gen.quantity("qtf_newman_momentum_conservation_difference")  # [F, F, H, 3]
    for h in range(2):
        np.testing.assert_array_equal(np.diagonal(nm[:, :, h, :], axis1=0, axis2=1).T, dg[:, h, :])


def test_the_drift_of_a_covered_barge_in_both_methods():
    """The fixed moonpool barge with its cover, flat patches of the higher order against the same
    faces of 2 and 4 low-order panels a metre: 1.6e-2 and 5.8e-3 of the peak found (converging, the
    ratio 2.8); 2.5e-2, 1e-2."""
    mb = moonpool_barge()
    body = {"name": "b", "mesh": part(mb, 1, 6), "modes": [], "free_surface_plates": [{"mesh": part(mb, 7, 7)}]}
    ho = solve(body, W_MP, [0.0, 30.0], options={**DRIFT, "higher_order": True}).drift.momentum_conservation.real
    for f, bound in ((2, 2.5e-2), (4, 1e-2)):
        faces = [face_panels(o, du, dv, round(np.linalg.norm(du) * 2 * f), round(np.linalg.norm(dv) * 2 * f), nrm)
                 for o, du, dv, nrm in MOONPOOL_FACES]  # fmt: skip
        nwet = sum(len(x) for x in faces[:6])
        m = surface(np.concatenate(faces), symmetry=QUARTER)
        cover = part(m, nwet + 1, nwet + len(faces[6]))
        lo = {**body, "mesh": part(m, 1, nwet), "free_surface_plates": [{"mesh": cover}]}
        lo = solve(lo, W_MP, [0.0, 30.0], options=DRIFT).drift.momentum_conservation.real
        assert peak(lo - ho) <= bound * peak(ho), (f, peak(lo - ho) / peak(ho))


# ---------------------------------- the pressure drift and the quadratic QTFs (theory §11.2, §12.3)

PRESSURE = {"drift_momentum_conservation": True, "drift_pressure_integration": True}


def _surge(r, bodies=1, width=6):
    """The pressure drift's surge of all bodies and the momentum drift's, real [F, pair]."""
    p = r.drift.pressure_integration.real
    return sum(p[:, :, b * width] for b in range(bodies)), r.drift.momentum_conservation.real[:, :, 0]


def _cover_and_piston(n):
    """The fixed moonpool hull with its cover (a plate of the hull: edge kind a) and with the same
    plate as a massless piston body in heave (a plate sliding along the hull's wall: kind b)."""
    mesh, nh, nl = moonpool(n)
    hull = {"name": "hull", "mesh": part(mesh, 1, nh), "modes": []}
    cover = {"mesh": part(mesh, nh + 1, nh + nl)}
    a = solve({**hull, "free_surface_plates": [cover]}, W_MP, [0.0], options=PRESSURE)
    b = solve([hull, plate_body(cover["mesh"], name="piston", **MASSLESS)], W_MP, [0.0],
              options={**PRESSURE, "motion": True})  # fmt: skip
    return _surge(a), _surge(b, bodies=2)


def test_the_pressure_drift_of_a_cover_and_of_a_piston_conserves_momentum():
    """The horizontal pressure drift (theory §11.2) against the momentum drift (theory §11.1) of the
    fixed moonpool hull with its cover (kind a, no band at the moonpool wall) and with a massless
    piston in heave (kind b, the band under the plate), 8 and 16 panels a quadrant: kind a 5.4e-2,
    3.7e-2 and kind b 7.2e-2, 5.1e-2 of the momentum drift's peak found (the hull's own discretization,
    as an ordinary body's pressure drift: its sharp edges, uniform panels; 2.9e-2 and 3.7e-2 at 32);
    the change from the cover to the piston, which the band carries (the piston's own horizontal
    drift is zero), 2.6e-2, 1.9e-2 (1.1e-2 at 32) of the peak from the momentum drift's change. Its
    band was 86, 442, 211 N/m^2 of surge at K a = 0.25, 0.5, 0.75 (16 panels; the change's error
    without it 136, 638, 332 against 50, 196, 121). Bounds: the falls and 6e-2, 2.5e-2 at 16."""
    errs = []
    for n in (8, 16):
        (pa, ma), (pb, mb) = _cover_and_piston(n)
        s = np.abs(ma).max()
        errs.append([np.abs(pa - ma).max() / s, np.abs(pb - mb).max() / s, np.abs((pb - pa) - (mb - ma)).max() / s])
    errs = np.array(errs)
    assert np.all(errs[1] < errs[0]) and np.all(errs[1, :2] <= 6e-2) and errs[1, 2] <= 2.5e-2, errs


def test_the_pressure_drift_of_a_piston_in_a_deep_water_column():
    """The deep water column (theory §9.7, the tube of draft 10) fixed with its massless piston
    (kind b), K = 0.3, 0.5, 0.8: the pressure drift against the momentum drift 8.4e-2, 3.7e-2 of the
    latter's peak found at n = 1, 2 (O(h)); bounds 1.2e-1, 5e-2 and the fall by 1.8."""
    w, errs = np.sqrt(np.array([0.3, 0.5, 0.8]) * G), []
    for n in (1, 2):
        hull, piston = owc(n)
        r = solve([{**hull, "modes": []}, {**piston, **MASSLESS}], w, [0.0], options={**PRESSURE, "motion": True})
        p, m = _surge(r, bodies=2)
        errs.append(np.abs(p - m).max() / np.abs(m).max())
    assert errs[0] <= 1.2e-1 and errs[1] <= 5e-2 and errs[0] >= 1.8 * errs[1], errs


def test_the_pressure_drift_and_qtfs_of_a_piston_mode_are_those_of_a_piston_body():
    """The restrained moonpool hull with its cover moved by a massless generalized piston mode (the
    plate's elements, w = 1: a vertical jump at the wall, the plate sliding along it) against the
    hull with the cover as a massless body of its own in heave (headings 0 and 30): the same
    displacement of every surface, so the hull's rigid rows of the one are the sums of the two
    bodies' rows of the other and the mode's generalized force is the piston's heave, for the
    pressure drift and the quadratic QTFs (7.8e-16 found; 1e-10). The difference diagonal is the
    drift, Q-(b, a) = Q-(a, b)* and Q+(b, a) = Q+(a, b) (1e-12)."""
    mesh, nh, nl = moonpool(8)
    piston = {"name": "piston", "parity": "heave", "pieces": [{"w": [[1.0, 0, 0, 0]], "ranges": [(nh + 1, nh + nl)]}]}
    hull = {"name": "hull", "mesh": part(mesh, 1, nh), "modes": [], **MASSLESS}
    cover = part(mesh, nh + 1, nh + nl)
    o = {**PRESSURE, "motion": True, "qtf_quadratic_pressure_integration": True}
    gen = solve({**hull, "free_surface_plates": [{"mesh": cover}], "generalized_modes": [piston],
                 "mode_mass": np.zeros((7, 7))}, W_MP, [0.0, 30.0], options=o)  # fmt: skip
    two = solve([hull, plate_body(cover, name="lid", **MASSLESS)], W_MP, [0.0, 30.0], options=o)
    name = "qtf_quadratic_pressure_integration_"
    for q in ("drift_pressure_integration", name + "difference", name + "sum"):
        g, t = gen.quantity(q), two.quantity(q)
        assert peak(g[..., :6] - t[..., :6] - t[..., 6:12]) <= 1e-10 * peak(g), q
        assert peak(g[..., 6] - t[..., 8]) <= 1e-10 * peak(g), q
    for r in (gen, two):
        dif, sm, pd = r.quantity(name + "difference"), r.quantity(name + "sum"), r.drift.pressure_integration
        scale = peak(pd)
        assert peak(np.stack([dif[f, f] for f in range(dif.shape[0])]) - pd) <= 1e-12 * scale
        assert peak(dif - np.conj(np.swapaxes(dif, 0, 1))) <= 1e-12 * scale
        assert peak(sm - np.swapaxes(sm, 0, 1)) <= 1e-12 * scale
        assert np.all(np.isfinite(dif)) and np.all(np.isfinite(sm))


def test_the_pressure_drift_of_a_cover_of_the_hull_or_of_its_own():
    """The fixed moonpool hull with its cover as a plate of the hull and as a body of its own
    (headings 0 and 30): the hull's drift is the sum of the two bodies' (8.8e-16 found; 1e-10); in
    head seas no sway, roll or yaw (7.5e-16; 1e-12); the scattered formulation 9.5e-3 from the
    total (the shielded water under the cover, theory §9.7; 2e-2). A cover of 8 panels a quadrant on
    the hull of 16 (the chords of the two meshes of the moonpool's circle apart) is routed and its
    drift 4.4e-4 from the conforming cover's (2e-3; as a free waterline its band would
    count)."""
    mesh, nh, nl = moonpool(8)
    hull = {"name": "hull", "mesh": part(mesh, 1, nh), "modes": []}
    cover = part(mesh, nh + 1, nh + nl)
    one = solve({**hull, "free_surface_plates": [{"mesh": cover}]}, W_MP, [0.0, 30.0], options=PRESSURE)
    own = solve([hull, plate_body(cover, name="lid", modes=())], W_MP, [0.0, 30.0], options=PRESSURE)
    a, b = one.drift.pressure_integration, own.drift.pressure_integration
    assert peak(a - b[..., :6] - b[..., 6:]) <= 1e-10 * peak(a)
    assert peak(a.real[:, 0, [1, 3, 5]]) <= 1e-12 * peak(a)
    sc = solve({**hull, "free_surface_plates": [{"mesh": cover}]}, W_MP, [0.0, 30.0],
               options={**PRESSURE, "diffraction_formulation": "scattered"})  # fmt: skip
    assert peak(sc.drift.pressure_integration - a) <= 2e-2 * peak(a)
    fine, nf, nfl = moonpool(16)
    lid = surface(moonpool_lid_panels(n=8), symmetry=QUARTER)
    c = solve({"name": "hull", "mesh": part(fine, 1, nf), "modes": [], "free_surface_plates": [{"mesh": lid}]},
              W_MP, [0.0], options=PRESSURE)  # fmt: skip
    d = solve({"name": "hull", "mesh": part(fine, 1, nf), "modes": [],
               "free_surface_plates": [{"mesh": part(fine, nf + 1, nf + nfl)}]}, W_MP, [0.0],
              options=PRESSURE)  # fmt: skip
    x, y = c.drift.pressure_integration, d.drift.pressure_integration
    assert peak(x - y) <= 2e-3 * peak(y), peak(x - y) / peak(y)


def test_the_higher_order_pressure_drift_of_a_cover_and_of_a_piston():
    """The higher order, the fixed moonpool barge (flat patches, 2 and 4 panels a metre) with its
    cover (kind a) and with the cover a massless piston body in heave (kind b): the pressure drift's
    surge against the momentum drift 1.8e-1, 1.4e-1 of the peak found for the cover (the barge's
    sharp edges, as for an ordinary body), falling; the change from the cover to the piston, which
    the band under the plate carries, 1.0e-2 and 8.1e-3 of the peak from the momentum drift's change
    (the piston moving 0.8 to 1.8 times the wave). Bounds the falls, 1.6e-1 and 1.2e-2."""
    errs = []
    for f in (1, 2):
        q = np.array([face_panels(o, du, dv, 1, 1, nrm)[0] for o, du, dv, nrm in MOONPOOL_FACES])
        nuv = [(max(1, round(np.linalg.norm(x[1] - x[0]) * 2 * f)), max(1, round(np.linalg.norm(x[3] - x[0]) * 2 * f)))
               for x in q]  # fmt: skip
        mb = flat(q, [(u, v, 3, 3, 3, 4) for u, v in nuv], symmetry=QUARTER)
        o = {**PRESSURE, "higher_order": True}
        hull = {"name": "b", "mesh": part(mb, 1, 6), "modes": []}
        pa, ma = _surge(solve({**hull, "free_surface_plates": [{"mesh": part(mb, 7, 7)}]}, W_MP, [0.0], options=o))
        r = solve([hull, plate_body(part(mb, 7, 7), name="p", **MASSLESS)], W_MP, [0.0], options={**o, "motion": True})
        pb, mbm = _surge(r, bodies=2)
        s = np.abs(ma).max()
        errs.append([np.abs(pa - ma).max() / s, np.abs((pb - pa) - (mbm - ma)).max() / s])
    errs = np.array(errs)
    assert np.all(errs[1] < errs[0]) and errs[1, 0] <= 1.6e-1 and errs[1, 1] <= 1.2e-2, errs


def test_the_pressure_drift_with_plates_and_pressure_surfaces_is_refused():
    """A plate whose edges lie on walls is routed with the pressure drift and its QTFs; with a
    pressure surface in the model it is refused (the closed forms of a body with pressure surfaces
    have no plate)."""
    mesh, nh, nl = moonpool(4)
    body = {"name": "hull", "mesh": part(mesh, 1, nh), "modes": [],
            "free_surface_plates": [{"mesh": part(mesh, nh + 1, nh + nl)}]}  # fmt: skip
    o = {**PRESSURE, "qtf_quadratic_pressure_integration": True, "qtf_newman": True}
    assert model(body, [1.0, 1.2], options=o).route().startswith("low_order")
    ring = surface(disc_panels(1.0, 2, 4, inner=0.5, z=-0.5), symmetry=QUARTER)
    tank = {**body, **MASSLESS, "pressure_surfaces": [{"mesh": ring, "pressure_modes": [{"name": "q"}]}]}
    match = "with the mean drift by pressure integration .and its QTFs. and pressure surfaces"
    with pytest.raises(hb.UnsupportedProblemError, match=match):
        model(tank, [1.0], options={**PRESSURE, "motion": True}).route()


# ---------------------------------- the control-surface drift and its quadratic QTFs (theory §11.3)

CONTROL = {"drift_momentum_conservation": True, "drift_control_surface": True}
CS_FLOATING = {"modes": ["surge", "heave", "pitch"], "mass": "displaced", "center_of_gravity": [0.0, 0.0, -0.2],
            "radii_of_gyration": [1.2, 1.2, 1.4]}  # fmt: skip
CS_QTF = {"qtf_quadratic_pressure_integration": True, "qtf_quadratic_control_surface": True}


def _cs(radius, size=0.25, **items):
    """The cylinder control surface about the moonpool hull (outer radius 2), depth 1.5."""
    return {"method": "cylinder", "radius": radius, "depth": 1.5, "inner_radius": 2.0, "panel_size": size, **items}


def _covered_cs(n, cs, **items):
    """The moonpool hull of moonpool(n) with its cover (a plate of the hull: edge kind a)."""
    mesh, nh, nl = moonpool(n)
    return {"name": "hull", "mesh": part(mesh, 1, nh), "modes": [], "control_surface": cs,
            "free_surface_plates": [{"mesh": part(mesh, nh + 1, nh + nl)}], **items}  # fmt: skip


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_the_control_surface_drift_of_a_cover_conserves_momentum():
    """The fixed moonpool hull with its cover, 8 and 16 panels a quadrant, cylinders of radius 3 and
    4: the horizontal control-surface drift against the momentum drift 5.6e-3, 2.8e-3 of its peak
    found (O(h); the pressure drift 4.7e-2, 3.3e-2 on the same meshes, the hull's sharp edges); the
    radius changes the drift by 8.1e-4 and 2.0e-4 of its peak (the identity of the balance; the
    pitch moment). Bounds 8e-3, 4e-3 and the fall; 1.5e-3."""
    errs = []
    for n in (8, 16):
        r3, r4 = (solve(_covered_cs(n, _cs(c)), W_MP, [0.0, 30.0], options=CONTROL) for c in (3.0, 4.0))
        c3, c4, m = r3.drift.control_surface.real, r4.drift.control_surface.real, r3.drift.momentum_conservation.real
        errs.append(peak(c3[..., :2] - m[..., :2]) / peak(m))
        assert peak(c3 - c4) <= 1.5e-3 * peak(c3), (n, peak(c3 - c4) / peak(c3))
    assert errs[0] <= 8e-3 and errs[1] <= 4e-3 and errs[1] < 0.7 * errs[0], errs


def test_the_control_surface_drift_of_a_cover_is_that_of_the_plugged_moonpool():
    """The fixed moonpool hull with its cover against the same hull plugged by a wetted disc at
    z = -d (an ordinary body, its control surface without plates; _plugged), every component:
    the plugged drift tends to the cover's, 1.0e-2, 1.0e-2, 7.9e-3 of the peak for d = 0.1, 0.05,
    0.025 found (the pitch moment the largest part; surge 3e-3, heave 4e-3 of their own peaks), the
    difference of the two meshes. Both differ from the pressure drift by 10 % in heave (the
    pressure drift's own error at the sharp edges, the same for the ordinary plugged body). Bound
    1.5e-2."""
    c = solve(_covered_cs(8, _cs(3.0)), W_MP, [0.0], options=CONTROL).drift.control_surface.real
    for d in (0.1, 0.05, 0.025):
        plug = {"name": "plug", "mesh": _plugged(d), "modes": [], "interior_surface": "none",
                "control_surface": _cs(3.0)}  # fmt: skip
        p = solve(plug, W_MP, [0.0], options=CONTROL).drift.control_surface.real
        assert peak(p - c) <= 1.5e-2 * peak(c), (d, peak(p - c) / peak(c))


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_the_control_surface_drift_and_qtfs_of_a_floating_cover():
    """The floating moonpool hull (surge, heave, pitch, its displaced mass) with its cover, headings 0
    and 30: the horizontal control-surface drift against the momentum drift 3.8e-2, 2.5e-2 of the
    peak on 8 and 16 panels a quadrant found, through the pitch resonance where the pressure drift
    misses by a factor 1.6 to 1.8 (the hull's sharp edges); the form with the squared elevation on
    the body waterline ("body", the free-surface integral beside the hull) 1.6e-1, 7.7e-2,
    converging, its vertical terms equal (exact). The plugged moonpool floating likewise (the
    hydrostatic part with the plate's area in the waterplane, the hull term of the QTFs over the
    plate; K a = 0.15 to 0.5): the difference QTF 1.0e-1, 4.5e-2, 1.5e-2 from the cover's for
    d = 0.05, 0.025, 0.0125 (the volume of the plug shifts the resonance), O(d). The difference
    diagonal is the drift, Q-(b, a) = Q-(a, b)* and Q+(b, a) = Q+(a, b)
    (exact), head seas no sway, roll or yaw (1e-15). Bounds 5e-2, 3e-2; 1.2e-1 and 0.5 the fall;
    1e-12."""
    o = {**CONTROL, **CS_QTF, "motion": True}
    errs = []
    for n in (8, 16):
        r, b = (solve(_covered_cs(n, _cs(3.0, waterline_term=t), **CS_FLOATING), W_MP, [0.0, 30.0], options=o)
                for t in ("control_surface", "body"))  # fmt: skip
        c, cb, m = r.drift.control_surface, b.drift.control_surface, r.drift.momentum_conservation.real
        errs.append([peak(c.real[..., :2] - m[..., :2]) / peak(m), peak(cb.real[..., :2] - m[..., :2]) / peak(m)])
        assert peak(c[..., 2:5] - cb[..., 2:5]) <= 1e-12 * peak(c)
        name = "qtf_quadratic_control_surface_"
        dif, sm = r.quantity(name + "difference"), r.quantity(name + "sum")
        assert peak(np.stack([dif[f, f] for f in range(dif.shape[0])]) - c) <= 1e-12 * peak(c)
        assert peak(dif - np.conj(np.swapaxes(dif, 0, 1))) <= 1e-12 * peak(c)
        assert peak(sm - np.swapaxes(sm, 0, 1)) <= 1e-12 * peak(c)
        assert peak(c.real[:, 0, [1, 3, 5]]) <= 1e-12 * peak(c)
    errs = np.array(errs)
    assert errs[0, 0] <= 5e-2 and errs[1, 0] <= 3e-2 and np.all(errs[1] < errs[0]), errs
    w = np.sqrt(np.array([0.3, 0.5, 0.75, 1.0]) * G / 2.0)
    cover = solve(_covered_cs(8, _cs(3.0), **CS_FLOATING), w, [0.0], options=o)
    q0 = cover.quantity("qtf_quadratic_control_surface_difference")
    e = []
    for d in (0.05, 0.025, 0.0125):
        plug = {"name": "plug", "mesh": _plugged(d), **CS_FLOATING, "interior_surface": "none",
                "control_surface": _cs(3.0)}  # fmt: skip
        q = solve(plug, w, [0.0], options=o).quantity("qtf_quadratic_control_surface_difference")
        e.append(peak(q - q0) / peak(q0))
    assert e[0] <= 1.2e-1 and e[1] <= 0.5 * e[0] and e[2] <= 0.5 * e[1], e


def _box_cs(c, size, depth=1.5, inner=2.0):
    """A box control surface |x|, |y| <= c about the moonpool barge (half width 2), quadrant of
    low-order panels of the size, its free surface between the barge's waterline and the box."""
    k = lambda length: max(1, round(length / size))  # noqa: E731
    faces = [((c, 0, -depth), (0, c, 0), (0, 0, depth), (1, 0, 0)),
             ((0, c, -depth), (c, 0, 0), (0, 0, depth), (0, 1, 0)),
             ((0, 0, -depth), (c, 0, 0), (0, c, 0), (0, 0, -1)),
             ((inner, 0, 0), (c - inner, 0, 0), (0, inner, 0), (0, 0, 1)),
             ((0, inner, 0), (c, 0, 0), (0, c - inner, 0), (0, 0, 1))]  # fmt: skip
    p = [q for o, du, dv, nrm in faces
         for q in face_panels(o, du, dv, k(np.linalg.norm(du)), k(np.linalg.norm(dv)), nrm)]  # fmt: skip
    return surface(np.array(p), symmetry=QUARTER)


def _ho_barge(f):
    """The moonpool barge of flat patches, 2 f panels a metre: elements 1-6 the hull, 7 its cover."""
    q = np.array([face_panels(o, du, dv, 1, 1, nrm)[0] for o, du, dv, nrm in MOONPOOL_FACES])
    nuv = [(max(1, round(np.linalg.norm(x[1] - x[0]) * 2 * f)), max(1, round(np.linalg.norm(x[3] - x[0]) * 2 * f)))
           for x in q]  # fmt: skip
    return flat(q, [(u, v, 3, 3, 3, 4) for u, v in nuv], symmetry=QUARTER)


@pytest.mark.slow  # seconds to minutes: deselected by scripts/test.* fast
def test_the_higher_order_control_surface_drift_of_a_covered_barge():
    """The higher order, the fixed moonpool barge with its cover (2 panels a metre), automatic
    control surfaces (outer squares of half width 3 and 4, panels of 0.25; the covered moonpool's
    waterline left out): the horizontal drift against the momentum drift 1.0e-7 and 7.0e-7 of its
    peak found (the pressure drift 1.8e-1, the barge's sharp edges; 3.4e-8 and 6.7e-7 with 4 panels
    a metre), the size of the square 2.6e-3 of the peak (the pitch moment; 1.6e-3 with 4 panels a
    metre). A box of low-order panels of 0.125 (a given mesh) against the automatic surface 1.7e-2
    (its constant panels of the free surface beside the barge's corners; its horizontal drift 1.1e-3
    from the momentum drift). Bounds 1e-5, 5e-3, 3e-2, 2e-3."""
    o = {**CONTROL, "higher_order": True}
    mb = _ho_barge(1)
    body = {"name": "b", "mesh": part(mb, 1, 6), "modes": [], "free_surface_plates": [{"mesh": part(mb, 7, 7)}]}
    out = []
    for c in (3.0, 4.0):
        auto = {"method": "automatic", "outer": "polygon", "boundary": [(c, 0), (c, c), (0, c)], "depth": 1.5,
                "panel_size": 0.25}  # fmt: skip
        r = solve({**body, "control_surface": auto}, W_MP, [0.0, 30.0], options=o)
        a, m = r.drift.control_surface.real, r.drift.momentum_conservation.real
        assert peak(a[..., :2] - m[..., :2]) <= 1e-5 * peak(m), (c, peak(a[..., :2] - m[..., :2]) / peak(m))
        out.append(a)
    assert peak(out[0] - out[1]) <= 5e-3 * peak(out[0]), peak(out[0] - out[1]) / peak(out[0])
    box = {**body, "control_surface": {"method": "mesh", "mesh": _box_cs(3.0, 0.125)}}
    b = solve(box, W_MP, [0.0, 30.0], options=o).drift.control_surface.real
    assert peak(b - out[0]) <= 3e-2 * peak(out[0]), peak(b - out[0]) / peak(out[0])
    assert peak(b[..., :2] - m[..., :2]) <= 2e-3 * peak(m)


def test_the_automatic_control_surface_of_a_covered_moonpool():
    """The automatic control surface about the moonpool hull with its cover (outer circle of radius
    3): the waterline under the cover is left out (no free surface inside), its drift against the
    cylinder of the same radius 2.0e-2 of the peak found (the pitch moment: the free surface of the
    automatic surface begins at the chords of the hull's waterline, the cylinder's at the circle
    of radius 2, outside them; surge 1e-6, heave 1.2e-3 of the cylinder's; 3e-2) and its
    horizontal drift against the momentum drift 5.6e-3 (8e-3). The same
    hull with a cover of a mesh of its own: the same surface and drift (1e-5, float32). Without the cover
    the automatic surface of the moonpool hull is refused (two waterlines about the origin)."""
    auto = {"method": "automatic", "outer": "circle", "radius": 3.0, "depth": 1.5, "panel_size": 0.25}
    a = solve(_covered_cs(8, auto), W_MP, [0.0], options=CONTROL)
    c = solve(_covered_cs(8, _cs(3.0)), W_MP, [0.0], options=CONTROL).drift.control_surface.real
    ad, m = a.drift.control_surface.real, a.drift.momentum_conservation.real
    assert peak(ad - c) <= 3e-2 * peak(c), peak(ad - c) / peak(c)
    assert peak(ad[..., :2] - m[..., :2]) <= 8e-3 * peak(m), peak(ad[..., :2] - m[..., :2]) / peak(m)
    mesh, nh, nl = moonpool(8)
    lid = surface(moonpool_lid_panels(n=8), symmetry=QUARTER)
    own = {**_covered_cs(8, auto), "free_surface_plates": [{"mesh": lid}]}
    assert peak(solve(own, W_MP, [0.0], options=CONTROL).drift.control_surface.real - ad) <= 1e-5 * peak(ad)
    with pytest.raises(ValueError, match="automatic control surface"):
        model({"name": "hull", "mesh": part(mesh, 1, nh), "modes": [], "control_surface": auto}, W_MP,
              options=CONTROL).counts()  # fmt: skip


def test_a_piston_body_inside_the_control_surface_of_its_hull_is_refused():
    """The moonpool hull's control surface holds its cover as a body of its own (a piston): the
    balance would give the force on both (decision of one body per control surface); refused by
    the model rules, naming the piston, also for a plate of a mesh of its own."""
    mesh, nh, nl = moonpool(4)
    hull = {"name": "hull", "mesh": part(mesh, 1, nh), "modes": [], "control_surface": _cs(3.0)}
    for lid in (part(mesh, nh + 1, nh + nl), surface(moonpool_lid_panels(n=4), symmetry=QUARTER)):
        piston = plate_body(lid, name="piston", control_surface=_cs(3.0), **MASSLESS)
        with pytest.raises(ValueError, match="its control surface reaches .*piston"):
            model([hull, piston], [1.0], options={**CONTROL, "motion": True}).route()


# ------------------------------------------------------------------ omega = inf (theory §9.7, §9.8)

INF = math.inf
NO_DIFFRACTION = {"diffraction": False}
# the disc in unbounded fluid, (8/3) rho a^3 in translation and (16/45) rho a^5 in rotation (Newman
# 1977, figure 4.8: the spheroid's m11'' = 2/pi and m55'' = 2/(3 pi) for a/b -> 0): the plate at
# omega = inf has half of each (the odd image of phi = 0 on the free surface)
DISC_INF = np.array([4.0 / 3.0, 8.0 / 45.0]) * RHO


def _heave_pitch(a):
    return np.array([a[2, 0], a[4, 1]])


def _disc_sheet(n, z):
    """The disc of `disc(n, graded=True)` as a thin sheet of a body of its own at depth -z."""
    v = disc_panels(1.0, n_r=n, n_phi=2 * n, quarter=True, graded=True) + np.array([0.0, 0.0, z])
    return {"modes": ["heave", "pitch"], "dipoles": [{"mesh": surface(v, symmetry=QUARTER)}]}


def test_a_disc_at_infinite_frequency_is_half_of_lambs_disc():
    """At omega = inf the free surface is phi = 0 and a disc in z = 0 is, by the odd image, a rigid
    disc in unbounded fluid with water on both faces: its added mass is half of Lamb's disc, (4/3) rho
    a^3 in heave and (8/45) rho a^5 in pitch. Low order, rings graded to the edge, 8, 16, 32 a
    quadrant: heave 8.1e-2, 4.2e-2, 2.1e-2, pitch 1.0e-1, 5.2e-2, 2.7e-2 found, falling as h (the jump
    of the potential has the square-root edge of a sheet, theory §9.6; as the submerged sheet of
    test_thin_sheets); extrapolated in h from 16 and 32, 4.5e-4 and 1.0e-3 (3e-3). The same disc as
    a sheet ten radii deep, the mean of A(0) and A(inf) (its free-surface images cancel to
    O((a/d)^6)), is twice the plate on the same panels to 1.8e-7 found (1e-6): the plate's coincident
    image doubling its kernels is the separated image of the sheet."""
    errs = []
    for n in (8, 16, 32):
        r = solve(plate_body(disc(n, graded=True), modes=("heave", "pitch")), [INF], [], options=NO_DIFFRACTION)
        a = _heave_pitch(r.added_mass_infinite_frequency)
        errs.append(a / DISC_INF - 1.0)
        if n <= 16:
            s = solve(_disc_sheet(n, -10.0), [0.0, INF], [], options=NO_DIFFRACTION)
            mean = 0.5 * (s.added_mass_zero_frequency + s.added_mass_infinite_frequency)
            assert np.all(np.abs(a / (0.5 * _heave_pitch(mean)) - 1.0) <= 1e-6), (n, a, mean)
    errs = np.array(errs)
    extrap = 2.0 * errs[2] - errs[1]
    assert np.all(errs[2] <= 0.035) and np.all(errs[2] < errs[1]) and np.all(np.abs(extrap) <= 3e-3), (errs, extrap)


def _ho_disc(n, z=0.0):
    """The quadrant of the unit disc at depth -z as one analytic patch, its radial spans graded to the
    edge ("quadratic_end": the jump sqrt(1 - r^2) smooth in the parameter), n of them and about as
    many around."""
    patch = {"type": "AnalyticPatch", "map": "line", "profile": "quadratic_end", "a": 1.0, "b": 1.0, "theta0": 0.0,
             "theta1": 0.5 * math.pi, "p": [0.0, z, 1.0, z]}  # fmt: skip
    return {"kind": "analytic_patches", "discretization": np.array([[math.ceil(0.5 * math.pi * n), n, 3, 3, 3, 4]]),
            "symmetry": QUARTER, "patches": [patch]}  # fmt: skip


def test_a_higher_order_disc_at_infinite_frequency():
    """The higher order: the disc of 2, 4, 6 radial spans graded to its edge against (4/3) rho a^3 and
    (8/45) rho a^5: heave 7.5e-4, 5.4e-5, 1.1e-5, pitch 3.6e-3, 1.2e-4, 1.0e-5 found (the own patch's
    closed forms and singular subtraction doubled, its free-surface image left out); 1e-4, 2e-4 at
    4 and 3e-5 at 6. Equal to half of the same disc as a sheet ten radii deep (the mean of its two
    limits) to float32 (0 found; 1e-6)."""
    ho = {**NO_DIFFRACTION, "higher_order": True}
    errs = []
    for n in (2, 4, 6):
        r = solve(plate_body(_ho_disc(n), modes=("heave", "pitch")), [INF], [], options=ho)
        a = _heave_pitch(r.added_mass_infinite_frequency)
        errs.append(np.abs(a / DISC_INF - 1.0))
        if n == 4:
            s = solve({"modes": ["heave", "pitch"], "dipoles": [{"mesh": _ho_disc(n, -10.0)}]}, [0.0, INF], [],
                      options=ho)  # fmt: skip
            mean = 0.5 * (s.added_mass_zero_frequency + s.added_mass_infinite_frequency)
            assert np.all(np.abs(a / (0.5 * _heave_pitch(mean)) - 1.0) <= 1e-6)
    assert np.all(errs[1] <= [1e-4, 2e-4]) and np.all(errs[2] <= 3e-5), errs


def test_a_square_plate_at_infinite_frequency_in_both_methods():
    """The square of side 2 at omega = inf, half of the square sheet in unbounded fluid (no published
    value checked locally): both methods converge as h (the square-root edge of the jump on uniform
    elements), A33 / rho = 1.8637, 1.8410 (higher order, 4 and 8 cubic panels a side of the quadrant)
    and 1.9040, 1.8616 (low order, 16 and 32 panels), the differences falling by 2.1 and 2.0 for each
    halving; extrapolated in h, 1.8183 and 1.8191 (the higher order from 8 and 12 panels: 1.8188),
    which agree to 4.4e-4 found (1e-3): 0.2274 rho s^3 for the plate."""
    q = face_panels((0, 0, 0), (1, 0, 0), (0, 1, 0), 1, 1, (0, 0, -1))
    ext = []
    for ho, sizes in ((True, (2, 4, 8)), (False, (8, 16, 32))):
        a = []
        for n in sizes:
            if ho:
                mesh = flat(np.array(q), [(n, n, 3, 3, 3, 4)], symmetry=QUARTER)
            else:
                mesh = surface(face_panels((0, 0, 0), (1, 0, 0), (0, 1, 0), n, n, (0, 0, -1)), symmetry=QUARTER)
            r = solve(plate_body(mesh), [INF], [], options={**NO_DIFFRACTION, "higher_order": ho})
            a.append(r.added_mass_infinite_frequency[2, 0] / RHO)
        d = np.diff(a)
        assert np.all(d < 0) and 1.6 <= d[0] / d[1] <= 2.6, (ho, a)
        ext.append(2.0 * a[2] - a[1])
    assert abs(ext[0] / ext[1] - 1.0) <= 1e-3, ext


def test_a_cover_at_infinite_frequency_of_the_hull_or_of_its_own():
    """The moonpool cover at omega = inf as a plate of the hull, as a body of its own (the sums of
    the 2 x 2 blocks of heave and pitch) and as a generalized piston mode of the hull (against the
    cover body's heave): 2.3e-7 and 1.8e-7 found (1e-5). A symmetric to the discretization of the
    cover's edge on the moonpool wall (a sheet ending on a wetted surface and its image): 1.0e-2 and
    6.4e-3 of the peak on 8 and 12 panels a quadrant (1e-2 at 12, falling). The higher order, the
    barge's cover of one flat patch: the sums 8.4e-8 (1e-5), A symmetric to 2.8e-4 (1e-3)."""
    asym = []
    for n in (8, 12):
        mesh, nh, nl = moonpool(n)
        hull = {"name": "hull", "mesh": part(mesh, 1, nh), "modes": ["heave", "pitch"]}
        plate = {"mesh": part(mesh, nh + 1, nh + nl)}
        one = solve({**hull, "free_surface_plates": [plate]}, [INF], [], options=NO_DIFFRACTION)
        two = solve([hull, plate_body(plate["mesh"], name="lid", modes=("heave", "pitch"))], [INF], [],
                    options=NO_DIFFRACTION)  # fmt: skip
        a, b = one.added_mass_infinite_frequency, two.added_mass_infinite_frequency
        for row, (i, j) in ((2, (2, 8)), (4, (4, 10))):
            for col, (k, m) in ((0, (0, 2)), (1, (1, 3))):
                s = b[i, k] + b[i, m] + b[j, k] + b[j, m]
                assert abs(a[row, col] - s) <= 1e-5 * np.abs(a).max(), (n, row, col)
        x = b[[2, 4, 8, 10]]
        asym.append(np.abs(x - x.T).max() / np.abs(x).max())
        if n == 8:
            piston = {"name": "piston", "parity": "heave",
                      "pieces": [{"w": [[1.0, 0, 0, 0]], "ranges": [(nh + 1, nh + nl)]}]}  # fmt: skip
            gen = solve({**hull, "modes": [], "free_surface_plates": [plate], "generalized_modes": [piston]}, [INF],
                        [], options=NO_DIFFRACTION).added_mass_infinite_frequency  # fmt: skip
            assert abs(gen[6, 0] - b[8, 2]) <= 1e-5 * abs(b[8, 2])
    assert asym[1] <= 1e-2 and asym[1] < asym[0], asym
    mb = moonpool_barge()
    ho = {**NO_DIFFRACTION, "higher_order": True}
    barge = {"name": "barge", "mesh": part(mb, 1, 6), "modes": ["heave", "pitch"]}
    one = solve({**barge, "free_surface_plates": [{"mesh": part(mb, 7, 7)}]}, [INF], [], options=ho)
    two = solve([barge, plate_body(part(mb, 7, 7), name="lid", modes=("heave", "pitch"))], [INF], [], options=ho)
    a, b = one.added_mass_infinite_frequency, two.added_mass_infinite_frequency
    assert abs(a[2, 0] - (b[2, 0] + b[2, 2] + b[8, 0] + b[8, 2])) <= 1e-5 * abs(a[2, 0])
    x = b[[2, 4, 8, 10]]
    assert np.abs(x - x.T).max() <= 1e-3 * np.abs(x).max()


def test_a_plate_at_infinite_frequency_by_a_wall_in_finite_depth_and_in_a_channel():
    """The disc at x = 3 by the wall x = 0 at omega = inf is its mirror pair in the open sea (surge,
    heave, pitch): 3.8e-8 found (1e-5). Finite depth tends to deep water: the disc's heave in depth
    2, 5, 20 differs from deep water by 1.1e-2, 8.0e-4, 1.3e-5 found (falling; 1e-4 at 20). A channel
    without reflection is the open sea of its depth, both methods (1.7e-7 and 5e-15 found; 1e-6)."""
    d = plate_body(disc(8, quarter=False), name="d", modes=("surge", "heave", "pitch"), position=[3.0, 0.0, 0.0])
    wall = solve(d, [INF], [], options={**NO_DIFFRACTION, "domain": "x_wall"}).added_mass_infinite_frequency
    pair = solve([d, {**d, "name": "e", "position": [-3.0, 0.0, 0.0]}], [INF], [],
                 options=NO_DIFFRACTION).added_mass_infinite_frequency  # fmt: skip
    rows = [0, 2, 4]
    s = pair[rows][:, :3] + pair[rows][:, 3:] * np.array([-1.0, 1.0, -1.0])
    assert np.abs(wall[rows] - s).max() <= 1e-5 * np.abs(wall).max()
    body = plate_body(disc(8), modes=("heave", "pitch"))
    deep = solve(body, [INF], [], options=NO_DIFFRACTION).added_mass_infinite_frequency[2, 0]
    gaps = [abs(solve(body, [INF], [], depth=h, options=NO_DIFFRACTION).added_mass_infinite_frequency[2, 0] / deep - 1)
            for h in (2.0, 5.0, 20.0)]  # fmt: skip
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] <= 1e-4, gaps
    channel = {"domain": "channel", "channel_width": 8.0, "channel_reflect": 0.0}
    for mesh, ho in ((disc(8), False), (_ho_disc(2), True)):
        b = plate_body(mesh, modes=("heave", "pitch"), position=[0.0, 1.0, 0.0])
        o = {**NO_DIFFRACTION, "higher_order": ho}
        a, c = (solve(b, [INF], [], depth=3.0, options={**o, **x}).added_mass_infinite_frequency for x in ({}, channel))
        assert np.abs(a - c).max() <= 1e-6 * np.abs(a).max(), ho


@pytest.mark.slow
def test_the_added_mass_tends_to_its_infinite_frequency_limit():
    """A(omega) of the disc toward A(inf) (a trend, not a bound): the dock of zero draft in depth 5
    by eigenfunction matching (scripts/gen_cylinder_reference.py, N = 200 and 400 extrapolated)
    against the higher order's A33(inf) in the same depth (6 radial spans, 1.3343 rho a^3): the gap
    is 8.2 %, 6.3 %, 4.2 % at Ka = 16, 32, 64 (approaching from below, slowly; no bound on the rate),
    where the higher order at Ka = 2 and 4 agrees with the matching to 3.3e-6 and 5.6e-5 found
    (5e-4; its spans resolve those waves)."""
    gen = load_script("gen_cylinder_reference")

    def dock(ka):
        a, b = (gen.solve(5.0, 0.0, ka, n, "heave")[0] for n in (200, 400))
        return b + (b - a) / 3.0

    ho = {**NO_DIFFRACTION, "higher_order": True}
    w = np.sqrt(np.array([2.0, 4.0]) * G)
    r = solve(plate_body(_ho_disc(6)), [*w, INF], [], depth=5.0, options=ho)
    a, a_inf = r.added_mass[:, 2, 0] / RHO, r.added_mass_infinite_frequency[2, 0] / RHO
    for k, ka in enumerate((2.0, 4.0)):
        assert abs(a[k] / dock(ka) - 1.0) <= 5e-4, (ka, a[k], dock(ka))
    gaps = [1.0 - dock(ka) / a_inf for ka in (16.0, 32.0, 64.0)]
    assert gaps[0] > gaps[1] > gaps[2] > 0.0 and gaps[2] <= 0.05, gaps


def test_a_sheet_hanging_from_a_plate_is_refused_at_infinite_frequency():
    """At omega = inf the plate is a sheet in z = 0, and a sheet hanging from it makes a junction of
    three sheets (the sheet, its image and the plate) without a rule: refused, naming the sheet and
    the plate. The same sheet ending on a damping lid (unused at omega = inf) and a plate beside a
    submerged sheet are solved."""
    match = r"the omega = inf limit with a thin sheet hanging from a free-surface plate is not available"
    with pytest.raises(hb.UnsupportedProblemError, match=match + r" \(bodies/bell: element 1 .*meets bodies/bell "
                                                           r"free-surface plate 1 in z = 0\)"):  # fmt: skip
        model(**bell(1, True), omegas=[1.0, INF]).route()
    model(**bell(1, True), omegas=[1.0]).route()
    model(**bell(1, True, lid=True), omegas=[1.0, INF]).route()
    deep = {"name": "s", **_disc_sheet(4, -2.0)}
    model([plate_body(disc(2), position=[4.0, 0.0, 0.0]), deep], [1.0, INF]).route()


# ------------------------------------------------- the potential QTF (theory §12.4-§12.10, §12.7)

QTF_W = np.array([1.2, 1.6])
QTF_ENV = {"depth": 3.0}
FLOATING = {"modes": ["surge", "heave", "pitch"], "mass": 9424.8, "center_of_gravity": [0.0, 0.0, -0.3],
            "radii_of_gyration": [1.0, 1.0, 1.0]}  # fmt: skip


def _potential(bodies, moving=False, radius=6.0, size=0.4, headings=(0.0,), domain=None, **out):
    """The potential QTFs of both methods to the inner radius with generated panels of the size."""
    o = {"diffraction": True, "radiation": moving, "motion": moving, "qtf_potential_direct": True,
         "qtf_potential_indirect": True, **out}  # fmt: skip
    env = {**QTF_ENV, **({"domain": domain} if domain else {})}
    m = model(bodies, QTF_W, list(headings), environment=env, outputs=o)
    m.second_order.update(free_surface_inner_radius=radius, free_surface_panel_size=size)
    return m.solve()


def _qtfs(r, meth):
    p = getattr(r.qtf.potential, meth)
    return p.difference, p.sum


def _rel(a, b):
    return peak(a - b) / peak(b)


def _covered_hull(n=8, **body):
    mesh, nh, nl = moonpool(n)
    return {"name": "hull", "mesh": part(mesh, 1, nh), "modes": [], **body,
            "free_surface_plates": [{"mesh": part(mesh, nh + 1, nh + nl)}]}  # fmt: skip


def test_the_free_surface_of_the_potential_qtf_ends_at_a_cover():
    """The free surface ends at the plates (theory §12.8): the moonpool's waterline under its cover
    borders no free surface, so the generated panels of the covered moonpool hull tile the disk of
    the inner radius less the hull's outer waterline polygon only (to round-off), none inside it.
    A given free-surface mesh that covers the plate is refused, and so are the cells."""
    m = model(_covered_hull(), QTF_W, [0.0], environment=QTF_ENV, outputs={"qtf_potential_indirect": True})
    m.second_order.update(free_surface_inner_radius=6.0, free_surface_panel_size=0.4)
    p = m.panels(role="free_surface")
    def polygon(n, r):  # the area of a regular polygon of n corners on the circle r
        return 0.5 * n * r * r * math.sin(2 * math.pi / n)

    v = p.vertices.reshape(-1, 3)
    rim = len(np.unique(np.round(np.arctan2(v[:, 1], v[:, 0])[np.hypot(v[:, 0], v[:, 1]) > 6.0 - 1e-9], 9)))
    # the rim of the spokes less the outer waterline (8 chords a quadrant of r = 2)
    np.testing.assert_allclose(p.areas.sum(), polygon(rim, 6.0) - polygon(32, 2.0), rtol=1e-12)
    assert np.hypot(p.centroids[:, 0], p.centroids[:, 1]).min() > 1.9
    m.second_order.free_surface_method = "cells"
    with pytest.raises(hb.UnsupportedProblemError, match="with the potential QTF take the free surface of the "
                                                         "generated panels or of a given mesh"):  # fmt: skip
        m.route()
    m.meshes.add("fs", kind="panels", vertices=[[[0.1, 0.1, 0], [0.5, 0.1, 0], [0.5, 0.5, 0], [0.1, 0.5, 0]],
                                                [[3.0, 0, 0], [4.0, 0, 0], [4.0, 1.0, 0], [3.0, 1.0, 0]]])  # fmt: skip
    m.second_order.update(free_surface_method="mesh", free_surface_mesh="fs")
    with pytest.raises(ValueError, match="the free surface of the potential QTF covers bodies/hull free-surface "
                                         "plate element"):  # fmt: skip
        m.solve()


def test_the_potential_qtf_with_a_free_plate_edge_is_refused():
    """A plate with an edge bordered by free surface (a disc, a mat): its second-order free-surface
    forcing is not integrable at the edge (theory §9.7, §12.7), refused for good, naming the plate's
    element. The covered moonpool (its cover's edges on the wall) is routed."""
    match = (r"free-surface plates with the potential QTF are not available for a plate with a free edge, one "
             r"bordered by free surface \(bodies/plate free-surface plate 1, element \d+\): the second-order "
             r"free-surface forcing is not integrable")  # fmt: skip
    for o in ({"qtf_potential_indirect": True}, {"qtf_potential_direct": True}):
        with pytest.raises(hb.UnsupportedProblemError, match=match):
            model(plate_body(PLATE, **INERTIA), [1.0, 1.2], options={**o, **MOVING}).route()
        assert model(_covered_hull(), [1.0, 1.2], options=o).route().startswith("low_order")


def test_the_potential_qtf_of_a_cover_of_the_hull_or_of_its_own():
    """The fixed moonpool hull with its cover (edge kind a, theory §12.7) as a plate of the hull and as a
    body of its own (headings 0 and 30): the hull's coefficients are the sums of the two bodies' in
    both methods (direct 9e-16, indirect 7e-7 found: the sums of two bodies' float32 radiation
    potentials against one; 1e-5); the difference coefficients Hermitian and the sums symmetric in
    the frequencies (exact); the direct and the indirect method agree, 7.2e-4 of the peak at the
    differences and 1.3e-2 at the sums on 8 panels a quadrant (3.2e-4, 3.8e-3 on 16; the slow test);
    head seas without sway, roll and yaw to 8.6e-5 (differences) and 3.3e-4 (sums) of the peak (the
    generated triangles beside the waterline not symmetric; 1e-3)."""
    covered = _covered_hull()
    lid = plate_body(covered["free_surface_plates"][0]["mesh"], name="lid", modes=())
    one = _potential(covered, headings=(0.0, 30.0))
    own = _potential([{**covered, "free_surface_plates": []}, lid], headings=(0.0, 30.0))
    for meth in ("direct", "indirect"):
        for x, y in zip(_qtfs(one, meth), _qtfs(own, meth), strict=True):
            assert _rel(y[..., :6] + y[..., 6:], x) <= 1e-5, meth
    d, s = _qtfs(one, "indirect")
    assert peak(d - np.conj(np.swapaxes(d, 0, 1))) <= 1e-12 * peak(d)
    assert peak(s - np.swapaxes(s, 0, 1)) <= 1e-12 * peak(s)
    dd, sd = _qtfs(one, "direct")
    assert _rel(dd, d) <= 2e-3 and _rel(sd, s) <= 3e-2, (_rel(dd, d), _rel(sd, s))
    head = one.qtf.heading_pairs.tolist().index([0.0, 0.0])
    for x in (d, s):
        assert peak(x[:, :, head][..., [1, 3, 5]]) <= 1e-3 * peak(x)


def test_the_potential_qtf_of_a_piston_mode_is_that_of_a_piston_body():
    """The restrained moonpool hull with its cover moved by a massless generalized piston mode (the
    plate sliding along the wall: edge kind b, the line integral of the wall's waterline with the
    jump of the displacement, theory §12.7) against the hull with the cover a massless body of its
    own in heave (the hull fixed, its waterline under the moving plate of another body): the same
    second-order problem, so the hull's rigid rows of the one are the sums of the two bodies' rows of
    the other and the mode's generalized force is the piston's heave. Direct 1.7e-15 found (one
    discretization), indirect 9e-7 (the float32 radiation potentials of two bodies against one);
    bounds 1e-12, 1e-5. The two methods 1.0e-3 (differences) and 1.3e-2 (sums) apart (3e-3, 3e-2)."""
    mesh, nh, nl = moonpool(8)
    piston = {"name": "piston", "parity": "heave", "pieces": [{"w": [[1.0, 0, 0, 0]], "ranges": [(nh + 1, nh + nl)]}]}
    hull = {"name": "hull", "mesh": part(mesh, 1, nh), "modes": [], **MASSLESS}
    cover = part(mesh, nh + 1, nh + nl)
    gen = _potential({**hull, "free_surface_plates": [{"mesh": cover}], "generalized_modes": [piston],
                      "mode_mass": np.zeros((7, 7))}, moving=True)  # fmt: skip
    two = _potential([hull, plate_body(cover, name="lid", **MASSLESS)], moving=True)
    for meth, bound in (("direct", 1e-12), ("indirect", 1e-5)):
        for g, t in zip(_qtfs(gen, meth), _qtfs(two, meth), strict=True):
            assert _rel(g[..., :6], t[..., :6] + t[..., 6:12]) <= bound, meth
            assert _rel(g[..., 6], t[..., 8]) <= bound, meth
            assert peak(g[..., 6]) > 1e-2 * peak(g)
    for x, y, bound in zip(_qtfs(gen, "direct"), _qtfs(gen, "indirect"), (3e-3, 3e-2), strict=True):
        assert _rel(x, y) <= bound


@pytest.mark.slow  # seconds: deselected by scripts/test.* fast
def test_the_potential_qtf_of_a_cover_converges_with_the_panels_and_the_radius():
    """The covered moonpool hull, fixed and floating (surge, heave, pitch), on 8 and 16 panels a
    quadrant: the direct and the indirect method converge to each other, fixed 7.2e-4, 3.2e-4 of the
    peak (differences) and 1.3e-2, 3.8e-3 (sums), floating 3.0e-3, 2.0e-3 and 3.1e-2, 2.5e-2 found
    (the floating sums, as for the open-sea cylinder of theory §12.6, Table 3, the slowest); bounds
    the falls and 1e-3, 1e-2, 4e-3, 4e-2 on 16. The inner radius 5, 6.5, 8 (fixed, 8 panels): the
    coefficients change by 1.1e-4, 6.7e-5 of the peak (differences) and 1.0e-2, 4.9e-3 (sums) from 5
    to 6.5 and from 6.5 to 8; bounds the falls and 2e-3, 1.5e-2."""
    errs = []
    for n in (8, 16):
        row = []
        for body, moving in ((_covered_hull(n), False), (_covered_hull(n, **FLOATING), True)):
            r = _potential(body, moving=moving)
            row += [_rel(x, y) for x, y in zip(_qtfs(r, "direct"), _qtfs(r, "indirect"), strict=True)]
        errs.append(row)
    errs = np.array(errs)
    assert np.all(errs[1] < errs[0]) and np.all(errs[1] <= [1e-3, 1e-2, 4e-3, 4e-2]), errs
    rs = [_qtfs(_potential(_covered_hull(), radius=r), "indirect") for r in (5.0, 6.5, 8.0)]
    ch = np.array([[_rel(a[k], b[k]) for k in (0, 1)] for a, b in zip(rs, rs[1:], strict=False)])
    assert np.all(ch[1] < ch[0]) and np.all(ch[0] <= [2e-3, 1.5e-2]), ch


@pytest.mark.slow  # seconds: deselected by scripts/test.* fast
def test_the_potential_qtf_of_a_plugged_moonpool_tends_to_the_covered_one():
    """The covered moonpool hull against the same hull plugged by a wetted disc at z = -d (an
    ordinary body: the existing potential QTF, its inner wall above the disc dry, one waterline), d =
    0.1, 0.05, 0.025, both methods. Fixed: the differences 5.0e-4, 2.5e-4, 1.3e-4 of the peak found
    (falling as d), the sums 2.5e-3 to 2.7e-3 (the two meshes' difference). Floating (surge, heave,
    pitch, the plate's terms of the body condition): the differences 1.2e-3, 5.6e-4, 4.5e-4, the sums
    1.1e-1, 7.7e-2, 4.7e-2 (indirect; direct 6.9e-2, 4.6e-2, 2.7e-2), falling with d. Bounds: the
    falls of the differences and of the floating sums, and the values at the smallest d."""
    for extra, moving, bounds in (({}, False, (3e-4, 4e-3)), (FLOATING, True, (1e-3, 7e-2))):
        c = _potential(_covered_hull(**extra), moving=moving)
        errs = []
        for d in (0.1, 0.05, 0.025):
            p = _potential({"name": "plug", "mesh": _plugged(d), "interior_surface": "none", "modes": [], **extra},
                           moving=moving)  # fmt: skip
            pairs = [zip(_qtfs(p, m), _qtfs(c, m), strict=True) for m in ("indirect", "direct")]
            errs.append([_rel(x, y) for z in pairs for x, y in z])
        errs = np.array(errs)  # [d][indirect dif, sum, direct dif, sum]
        falls = [0, 2, 1, 3] if moving else [0, 2]
        assert np.all(np.diff(errs[:, falls], axis=0) < 0), errs
        assert np.all(errs[-1, [0, 2]] <= bounds[0]) and np.all(errs[-1, [1, 3]] <= bounds[1]), errs


def test_the_potential_qtf_of_a_cover_by_a_wall_is_the_open_sea_with_its_mirror_image():
    """The fixed covered moonpool hull 3 m from the wall x = 0 (heading 30) against the open-sea
    problem of it and its mirror image under the wave and its mirror image (theory §12.14): the
    wall's coefficients are the even part (A + S B)/2 of the four bidirectional pairs summed, both
    methods (5.8e-8 to 5.5e-7 found, the float32 fields; 1e-5)."""
    place, image, heading = [3.0, 0.0, 0.0], [-3.0, 0.0, 0.0], 30.0
    sign = np.array([-1, 1, 1, 1, -1, -1])
    wall = _potential({**_covered_hull(), "position": place}, radius=12.0, size=0.8, headings=(heading,),
                      domain="x_wall")
    sea = _potential([{**_covered_hull(), "position": place}, {**_covered_hull(), "name": "image", "position": image}],
                     radius=12.0, size=0.8, headings=(heading, 180.0 - heading), qtf_bidirectional=True)  # fmt: skip
    hp = sea.qtf.heading_pairs.tolist()
    for meth in ("indirect", "direct"):
        for kind in ("difference", "sum"):
            q = f"qtf_potential_{meth}_{kind}"
            x, y = wall.quantity(q)[:, :, 0], sea.quantity(q)

            def pair(p, s, y=y, kind=kind):
                if [p, s] in hp:
                    return y[:, :, hp.index([p, s])]
                z = np.swapaxes(y[:, :, hp.index([s, p])], 0, 1)
                return np.conj(z) if kind == "difference" else z

            t = sum(pair(p, s) for p in (heading, 180.0 - heading) for s in (heading, 180.0 - heading))
            a, b = t[..., :6], t[..., 6:]
            even = peak(x - 0.5 * (a + sign * b)) / peak(a)
            assert even <= 1e-5, (q, even)
