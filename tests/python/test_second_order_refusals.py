"""The refusals of the potential QTF (reference/inputs-rules §1, theory §12.2, §12.8-§12.9), each with a model that
breaks one rule: the elements it is not available with (damping lids and dampers of the water,
pressure surfaces and tanks, thin sheets and dipole dampers of a body), and the free surface it
cannot build (the inner radius within the bodies' radius, a coarse waterline, an open waterline, no
waterline, a waterline inside another (a moonpool), the cells of bodies too close or a submerged
body, a free-surface mesh that does not reach beyond the waterlines). A refusal names the input to
change. Meshes are procedural (tests/python/meshes.py). The refusals of the potential QTF are tested
here alone."""

import numpy as np
import pytest

from bodies import acv_body, coarse_cylinder, moonpool_mesh, part, surface
from meshes import cylinder_panels, face_panels, spheroid_panels
from problems import flat_surface, model

G = 9.80665
ENV = {"density": 1000.0, "gravity": G, "depth": 3.0}
OUT = {"diffraction": True, "qtf_potential_indirect": True}


def cylinder(**kw):
    """A fixed cylinder of radius 1 and draft 0.5 (the local test body)."""
    return {"mesh": coarse_cylinder(), "modes": [], **kw}


def built(bodies, *, lids=(), dampers=(), **fs):
    """The model of the bodies with the potential QTF and the free surface items fs (R = 3 and
    panels of 0.3 unless given)."""
    m = model(bodies, [1.2], [0.0], environment=ENV, outputs=OUT, lids=lids, dampers=dampers)
    m.second_order.update(**flat_surface("free_surface", {**{"inner_radius": 3.0, "panel_size": 0.3, **fs}}))
    return m


def test_the_elements_the_potential_qtf_is_not_available_with():
    # a damping lid and a fixed damper on the free surface the potential QTF integrates
    lid = {"name": "lid", "mesh": surface(np.array(face_panels((1.5, 0.0, 0.0), (0.5, 0, 0), (0, 0.5, 0), 2, 2,
                                                               (0, 0, 1))), symmetry="none")}  # fmt: skip
    with pytest.raises(ValueError, match="is not available with damping lids or fixed dampers"):
        built(cylinder(), lids=[lid]).route()
    plate = surface(np.array(face_panels((0.0, 0.0, -1.5), (1.4, 0, 0), (0, 1.4, 0), 4, 4, (0, 0, 1))),
                    symmetry="xz_and_yz")
    with pytest.raises(ValueError, match="is not available with damping lids or fixed dampers"):
        built(cylinder(), dampers=[{"mesh": plate, "damping_factor": 1.0}]).route()
    # pressure surfaces (an air cushion) and internal tanks: the body condition has no term for them
    acv = acv_body(modes=("uniform",), mass="displaced", center_of_gravity=[0.0, 0.0, 0.0],
                   radii_of_gyration=[2.5, 5.0, 5.0])  # fmt: skip
    m = model(acv, [1.2], [0.0], environment=ENV, outputs={**OUT, "radiation": True, "motion": True})
    with pytest.raises(ValueError, match="is available for bodies without pressure surfaces or tanks"):
        m.route()
    # a damper of the body: the sub-problems take the velocities of the source formulation, which a
    # dipole sheet has not; refused before the solve, naming the QTF
    hull = cylinder_panels(1.0, 1.0, n_phi=6, n_z=4, n_r=3, quarter=True)
    sheet = np.array(face_panels((0.0, 0.0, -1.5), (1.4, 0, 0), (0, 1.4, 0), 4, 4, (0, 0, 1)))
    mesh = surface(np.concatenate([hull, sheet]), symmetry="xz_and_yz")
    body = {"mesh": part(mesh, 1, len(hull)), "modes": [],
            "dipole_dampers": [{"mesh": part(mesh, len(hull) + 1, len(hull) + len(sheet)), "damping_factor": 1.0}]}
    msg = r"qtf_potential_indirect\) is not available with thin sheets \(dipoles\) or dipole dampers"
    with pytest.raises(ValueError, match=msg):
        built(body).route()


def widened(k):
    """A fixed cylinder of waterline radius 1 and draft 0.5 whose side widens downwards, the bottom
    of radius 1 + k (a base wider than twice the waterline: beyond the generated panels' least
    inner radius)."""
    p = cylinder_panels(1.0, 0.5, n_phi=8, n_z=2, n_r=2, quarter=True).copy()
    s = 1.0 - k * p[..., 2] / 0.5
    p[..., 0] *= s
    p[..., 1] *= s
    return {"mesh": surface(p, symmetry="xz_and_yz"), "modes": []}


def test_the_free_surface_the_potential_qtf_cannot_build():
    # the inner radius within the radius of the bodies about the centre (a base wider than the
    # waterline): beyond R the fields are outgoing series, which converge only outside the bodies
    with pytest.raises(ValueError, match="must exceed the radius of the bodies about the centre"):
        built(widened(2.5), inner_radius=2.5).solve()
    # a waterline too coarse for the generated panels: a triangle, whose corners lie at twice the
    # distance of its sides' middles, beyond the square of the panels
    triangle = surface(cylinder_panels(1.0, 0.5, n_phi=3, n_z=2, n_r=2, quarter=False))
    with pytest.raises(ValueError, match="too coarse for the generated panels"):
        built(cylinder(mesh=triangle)).solve()
    # an open waterline: a quarter of the cylinder without its symmetry
    with pytest.raises(ValueError, match="a waterline is not closed"):
        built({**cylinder(), "mesh": surface(cylinder_panels(1.0, 0.5, n_phi=8, n_z=2, n_r=2), symmetry="none")}
              ).solve()  # fmt: skip
    # a waterline inside another (an annular cylinder: its moonpool of radius 1 in the hull of
    # radius 2): the generated panels cover the water outside the waterlines only, so the free
    # surface of the moonpool would be left out of the integral; refused, naming the input
    with pytest.raises(ValueError,
                       match=r"second_order: a waterline lies inside another \(a moonpool\)"):
        built(cylinder(mesh=moonpool_mesh()), inner_radius=5.0, panel_size=0.5).solve()
    # no waterline: a submerged body alone, the radius and size auto and given
    sub = {"mesh": surface(spheroid_panels(2.0, 0.25) + [0.0, 0.0, -1.0], symmetry="xz_and_yz"), "modes": []}
    m = model(sub, [1.2], [0.0], environment=ENV, outputs=OUT)
    with pytest.raises(ValueError, match="the bodies have no waterline"):
        m.solve()
    with pytest.raises(ValueError, match="the bodies have no waterline"):
        m.check()  # the check makes the zones as the solve does
    m.route()  # the route does not
    with pytest.raises(ValueError, match="the bodies have no waterline"):
        built(sub).solve()
    # the cells: a submerged body beside a floating one, and two bodies whose waterline boxes meet
    two = [cylinder(name="a"), {**sub, "name": "b", "position": [4.0, 0.0, 0.0]}]
    with pytest.raises(ValueError, match="the cells of the free surface need a waterline of every body"):
        built(two, method="cells", inner_radius=8.0).solve()
    close = [cylinder(name="a"), cylinder(name="b", position=[1.5, 0.0, 0.0])]
    with pytest.raises(ValueError, match="the box of the body's waterline meets another body's"):
        built(close, method="cells", inner_radius=6.0).solve()
    # a free-surface mesh: its farthest vertex sets R, which must lie beyond the waterlines
    m = model(cylinder(), [1.2], [0.0], environment=ENV, outputs=OUT)
    m.meshes.add("fs", kind="panels", vertices=[[[0.2, 0, 0], [0.8, 0, 0], [0.8, 0.5, 0], [0.2, 0.5, 0]]])
    m.second_order.update(**flat_surface("free_surface", {"method": "mesh", "mesh": "fs"}))
    with pytest.raises(ValueError, match="is not beyond the waterlines"):
        m.solve()


def test_a_free_surface_that_cannot_be_built_is_refused_before_the_solve():
    # the checks before a solve are those of the model check, the zones of the free surface among
    # them, also without the memory check: refused before any frequency is solved, with the
    # message of the check
    sub = {"mesh": surface(spheroid_panels(2.0, 0.25) + [0.0, 0.0, -1.0], symmetry="xz_and_yz"), "modes": []}
    m = model(sub, [1.2], [0.0], environment=ENV, outputs=OUT)
    with pytest.raises(ValueError, match="the bodies have no waterline") as checked:
        m.check()
    for memory_check in (True, False):
        m.options.memory_check = memory_check
        calls = []
        with pytest.raises(ValueError, match="the bodies have no waterline") as solved:
            m.solve(progress=calls.append)
        assert str(solved.value) == str(checked.value)
        assert calls == [] and m.progress().frequencies == () and m.progress().done == 0


def test_the_sizes_of_the_free_surface_are_checked():
    # the radius, the panel size and the zones' counts: positive (or auto) and in their ranges, as
    # they are set; the model keeps its values
    m = built(cylinder())
    fs = m.second_order
    bad = (("free_surface_inner_radius", -1.0), ("free_surface_panel_size", 0.0), ("quadrature_annuli", -1),
           ("quadrature_radius_step", -2.0),
           ("quadrature_radial_nodes", 0), ("quadrature_azimuthal_nodes", 0), ("outer_circle_segments", 0),
           ("expansion_order", 200))  # fmt: skip
    before = {item: repr(getattr(fs, item)) for item, _ in bad}
    for item, value in bad:
        with pytest.raises(ValueError, match=item):
            setattr(fs, item, value)
    assert {item: repr(getattr(fs, item)) for item, _ in bad} == before
    assert fs.free_surface_inner_radius == 3.0 and m.route().startswith("low_order")
