"""A floating vertical cylinder in deep water, every input of the model written out.

run.py: the cylinder, out/run/cylinder.hbr (the model with its results) and the WAMIT-format numeric
files out/run/cylinder.1, .2, ... and .out (the example of every input and of every output).

Every item of the objects this case uses is given (the mesh items that only patch meshes and
trims have, and the list `tool`, are left out): the values of this case, the defaults as they
are, and None for what is not used (the lists of objects are empty here; the items of their
objects are in the comment beside). Change a value and run it again. The cylinder: radius 1 m,
draft 0.5 m, one quadrant of panels (the symmetry planes x = 0 and y = 0 give the rest).

How the model is edited:
- A single object (m.environment, m.options, m.first_order, m.second_order, a body): an item by
  assignment (m.options.inner_threads = 4), several at once by update(name=value, ...) (all or
  nothing), back to the default by reset("name") or by assigning None; read by attribute,
  items() (every own item and value, not its name nor its lists:
  m.bodies.add(name="b2", **b1.items()) copies them), info("name") (kind, unit, default, switch,
  description); print(obj) shows the items given that its switches choose and a line naming the unused ones. A name the
  object does not have is refused at once, and nothing changes.
- Where one of two inputs is used, a switch names it, both have defaults, and only the one named
  is read; the items of a choice not made are kept unused and not checked, and nothing is
  refused or reset when one is set: environment.frequency_input (omegas or periods), a body's
  inertia_input (Ixx ... Izz or radii_of_gyration), interior_surface and control_surface (the
  method of each surface, its items prefixed by its name; control_surface_outer chooses
  control_surface_radius or _boundary), second_order.free_surface_method (the free surface of the
  potential QTF, free_surface_*). A needed item of the choice made that is missing (a radius, a
  depth, a mesh ...) is refused by check() and the solve.
- A list of objects (m.bodies, m.meshes, m.lids, m.fixed_dampers, m.tool; of a body: dipoles,
  dipole_dampers, tanks, pressure_surfaces, free_surface_plates, generalized_modes, couplings; of a
  tank: dipoles,
  dipole_dampers; of a pressure surface: pressure_modes; of a generalized mode: pieces; of a mesh:
  patches): add(name, **items) returns the new object, insert(index, name, **items), lst[name] or
  lst[index] finds one, del lst[name] removes it (refused while another object refers to it),
  rename(old, new) (the references follow), move(name, index), names(), len(), iteration,
  clear() (empties it, all or nothing). Its objects are made only by add() and insert(), each
  with its own items; an object's own lists are filled once it is made
  (m.bodies["cylinder"].tanks.add(mesh=..., ranges=...)). A list is not assigned, nor given to
  add() or update().
- m.tool: a list of named texts that other programs (a graphical interface, a script) keep with
  the model; the solve does not read them (m.tool.add(name="my_gui", text="...")).
"""

import math
from pathlib import Path

import numpy as np

import hydrobem as hb

OUT = Path(__file__).resolve().parent / "out" / "run"  # the outputs of this script
RADIUS, DRAFT = 1.0, 0.5
N_PHI, N_Z, N_R = 16, 6, 6  # panels around the quadrant, down the side, across the bottom
FREQUENCY_THREADS = 1  # frequencies solved at once (more give the same results, faster)


def quadrant_panels():
    """Panels of the wetted surface, vertices ordered so that the normals point out of the body
    into the water."""
    phi = np.linspace(0.0, np.pi / 2, N_PHI + 1)
    z = np.linspace(0.0, -DRAFT, N_Z + 1)
    r = np.linspace(RADIUS, 0.0, N_R + 1)

    def p(radius, angle, depth):
        return [radius * np.cos(angle), radius * np.sin(angle), depth]

    side = [[p(RADIUS, phi[j], z[i]), p(RADIUS, phi[j], z[i + 1]), p(RADIUS, phi[j + 1], z[i + 1]),
             p(RADIUS, phi[j + 1], z[i])] for i in range(N_Z) for j in range(N_PHI)]  # fmt: skip
    bottom = [[p(r[i], phi[j], -DRAFT), p(r[i + 1], phi[j], -DRAFT), p(r[i + 1], phi[j + 1], -DRAFT),
               p(r[i], phi[j + 1], -DRAFT)] for i in range(N_R) for j in range(N_PHI)]  # fmt: skip
    return np.array(side + bottom)


def build() -> hb.Model:
    m = hb.Model()

    # ---- the mesh (kind "panels": low-order panels)
    m.meshes.add(
        name="cylinder",
        kind="panels",                 # panels, flat_patches, spline_patches, analytic_patches, trimmed_patches
        vertices=quadrant_panels(),    # (P, 4, 3) m, normals into the water
        reference_length=1.0,          # m, L of the normalized results (the GDF field ULEN)
        symmetry="xz_and_yz",          # the symmetry planes: "none", "yz" (the plane x = 0: x -> -x,
                                       # the GDF field ISX), "xz" (the plane y = 0: y -> -y, ISY),
                                       # "xz_and_yz" (both: one quadrant stored)
        label="cylinder R = 1 m, T = 0.5 m, one quadrant",
        file_gravity=None,             # m/s^2, the gravity (GDF field GRAV) of a file it was read from (not used)
        source_file="",                # the absolute path of the GDF it was read from (set by meshes.load)
        source_elements=None,           # a mesh made by a trim: the source element of each element
    )

    # ---- the body
    m.bodies.add(
        name="cylinder",
        mesh="cylinder",               # the wetted surface: a mesh name
        ranges=None,                   # its panels (patches): [(first, last), ...] one-based, inclusive;
                                       # None: every one; hb.ranges(mask) makes them
        position=[0.0, 0.0, 0.0],      # m, global position of the body origin
        yaw=0.0,                       # deg, about the global z axis
        trim=None,                     # [elevation m, pitch deg, roll deg]: the mesh is the hull, cut at z = 0
        rotation_center=[0.0, 0.0, 0.0],  # m, body-local: every result is about this point
        modes=["surge", "sway", "heave", "roll", "pitch", "yaw"],  # [] a fixed body
        center_of_gravity=[0.0, 0.0, -0.2],  # m, body-local
        mass="displaced",              # kg, or "displaced": density x displaced volume
        inertia_input="radii_of_gyration",  # which inertia is used: "components" (Ixx ... Izz)
                                       #   or "radii_of_gyration" (the default); the other not read
        Ixx=0.0,                       # kg m^2 (the defaults)
        Ixy=0.0,
        Ixz=0.0,
        Iyy=0.0,
        Iyz=0.0,
        Izz=0.0,
        radii_of_gyration=[0.6, 0.6, 0.7],  # m, the diagonal (or a 3 x 3 matrix)
        inertia_origin="center_of_gravity",  # center_of_gravity, rotation_center or a point
        additional_stiffness=None,     # 6 x 6, SI
        additional_damping=None,       # 6 x 6, SI
        stiffness_origin="rotation_center",
        damping_origin="rotation_center",
        excitation_method="diffraction",  # diffraction or haskind
        mode_mass=None,                # (6 + G) x (6 + G), with generalized modes; the rigid block zero
        mode_damping=None,             # likewise, the rigid block zero
        mode_stiffness=None,           # likewise, the rigid block zero
        body_points=None,              # (N, 3) m, body-local points of the body outputs
        body_point_neighbors=4,        # low order with body_points
        enabled=True,                  # False: kept in the model, solved as if absent
        # the interior surface inside the waterline (irregular frequencies): "none", "detect" (the
        # default: the body mesh's panels lying in z = 0, none if it has none), "waterline",
        # "graded", "projected" (generated) or "mesh" (given)
        interior_surface="detect",
        interior_surface_mesh=None,    # (mesh) a mesh name
        interior_surface_ranges=None,  # (mesh) its panels (None: every one)
        # the control surface of the control-surface drift (needed by drift_control_surface, refused
        # without it): "none" (the default), "mesh" (given, e.g. a mesh of m.meshes.load_csf), or
        # generated with control_surface_panel_size: "cylinder", "ellipsoid", "automatic" (around
        # the body's waterline, with its free-surface part)
        control_surface="none",
        control_surface_mesh=None,     # (mesh) a mesh name
        control_surface_ranges=None,   # (mesh) its panels (None: every one)
        control_surface_waterline_term="control_surface",  # (every method) the zeta^2 term on the
                                       #   control surface's waterline, or "body" (on the body
                                       #   waterline, with a free-surface integral)
        control_surface_radius=None,   # m (cylinder; automatic with control_surface_outer "circle")
        control_surface_depth=None,    # m (cylinder, automatic)
        control_surface_inner_radius=None,  # m (cylinder) the circular waterline inside
        control_surface_semi_axes=None,     # m (ellipsoid) a, b, c of the lower half-ellipsoid
        control_surface_inner_semi_axes=None,  # m (ellipsoid) the elliptic waterline inside
        control_surface_outer="circle",     # (automatic) "circle" (radius) or "polygon" (boundary)
        control_surface_boundary=None,      # (automatic, polygon) (x, y) vertices, body coordinates
        control_surface_partitions=None,    # (automatic) further (x, y) polylines, each ended by NaN
        control_surface_panel_size=None,    # m (cylinder, ellipsoid, automatic)
    )
    # what moves with the body: its lists, empty here, filled once it is made (body.tanks.add(...));
    # every "mesh" a mesh name with its "ranges" (None: the whole mesh)
    #   dipoles            name, mesh, ranges: rigid thin sheets (the names optional)
    #   dipole_dampers     mesh, ranges, damping_factor (0.0): porous sheets
    #   tanks              name, mesh, ranges, density_ratio (1.0), surface_level ("walls_top"),
    #                      field_points ((N, 3) m, body axes, points inside the tank: its sloshing);
    #                      its own lists dipoles and dipole_dampers
    #   couplings          body, stiffness, damping: 6 x 6 with another body
    #   pressure_surfaces  mesh, ranges, chamber_coupling (False), response_convention
    #                      ("variational"); its list pressure_modes: name, shape ("uniform"),
    #                      additional_stiffness (0.0), additional_damping (0.0)
    #   free_surface_plates mesh, ranges: massless rigid plates in z = 0 moving with the body
    #   generalized_modes  name, parity ("surge": how the mode reflects with a symmetric mesh, as
    #                      the rigid mode named, "heave", "surge", "sway" or "yaw" class); its list
    #                      pieces: u, v, w, ranges (of the body's mesh; None: all of it)

    # ---- what is at rest in the water (global coordinates): lists of the model, empty here
    #   m.lids             name, mesh, ranges, placed_with (None), damping_factor (0.0)
    #   m.fixed_dampers    mesh, ranges, damping_factor (0.0): porous sheets fixed in space, in no
    #                      force

    # ---- the environment: the water and the waves
    m.environment.update(
        depth=math.inf,                # m
        gravity=9.80665,               # m/s^2
        density=1000.0,                # kg/m^3
        domain="open_sea",             # open_sea, x_wall, y_wall, orthogonal_walls, channel
        channel_width=None,            # m, for the channel
        channel_reflect=1.0,           # 0..1, for the channel
        frequency_input="periods",     # which is used: "omegas" (the default) or "periods";
                                       #   the other not read
        omegas=[],                     # rad/s (the default)
        periods=[2.0, 3.0, 4.0, 6.0, 8.0],  # s
        headings=[0.0],                # deg, counter-clockwise from +x
    )

    # ---- how it is solved
    m.options.update(
        higher_order=False,            # the higher-order method (patch meshes)
        logarithmic_integration=True,
        log_tolerance=0.0001,
        frequency_threads=FREQUENCY_THREADS,  # the results do not depend on the thread counts
        inner_threads=1,
        memory_check=True,             # refuse a solve whose memory estimate exceeds the free memory
    )

    # ---- what the first-order solve computes
    m.first_order.update(
        radiation=True,
        diffraction=True,
        diffraction_formulation="auto",  # auto (total with a lid), scattered or total
        motion=True,                   # the motions (RAOs)
        field_pressure=False,          # with field points
        field_velocity=False,
        body_pressure=False,
        body_velocity=False,
        field_points=None,             # (N, 3) m, global: the points of the field outputs in the sea
                                       #   (the points in a tank: the tank's field_points); used when
                                       #   field_pressure or field_velocity is on
    )

    # ---- the second-order loads
    m.second_order.update(
        # the mean drift, by method
        drift_pressure_integration=False,   # over the body (theory §11.2)
        drift_control_surface=False,        # through the bodies' control_surface (theory §11.3)
        drift_momentum_conservation=False,  # in the far field, all bodies (theory §11.1)
        drift_bidirectional=False,          # True: every pair of headings
        drift_momentum_conservation_angles="auto",  # a count 1..65536, or "auto": chosen by the engine
        # the QTFs, by part and method
        qtf_quadratic_pressure_integration=False,  # results.qtf.quadratic.pressure_integration
        qtf_quadratic_control_surface=False,       # results.qtf.quadratic.control_surface
        qtf_potential_direct=False,                # results.qtf.potential.direct
        qtf_potential_indirect=False,              # results.qtf.potential.indirect
        qtf_newman=False,                   # True: Newman's approximation from every drift switched on
                                            # above (none: refused)
        qtf_newman_average="geometric",     # or "arithmetic"
        qtf_bidirectional=False,            # True: every pair of headings
        qtf_difference=True,                # the difference frequencies
        qtf_sum=True,                       # the sum frequencies
        # the free surface of the potential QTF: its panelled zone "panels" (generated, each
        # integrated by its centroid; the default), "cells" (generated, 4 x 4 points a cell) or
        # "mesh" (given)
        free_surface_method="panels",
        free_surface_panel_size="auto",     # m, or auto (panels, cells)
        free_surface_inner_radius="auto",   # m, or auto: 4 x the largest waterline distance (panels, cells)
        free_surface_mesh=None,             # (mesh) a panels mesh in z = 0, global (e.g. of load_fdf)
        free_surface_ranges=None,           # (mesh) its panels (None: every one)
        free_surface_quadrature_points=1,   # (mesh) Gauss points per side of its panels
        quadrature_annuli=0,                # annuli beyond the inner radius before the far field
        quadrature_radius_step="auto",
        quadrature_radial_nodes=12,
        quadrature_azimuthal_nodes="auto",
        outer_circle_segments="auto",
        expansion_order="auto",             # the far-field series' highest order
    )
    return m


def main():
    m = build()
    print(m, "| route:", m.route())
    # the queries of the owners: the mesh its geometry, the body its hydrostatics (no solve)
    h = m.bodies["cylinder"].hydrostatics()
    print(
        "volume [m^3]:",
        round(m.meshes["cylinder"].geometry().volume, 4),
        "| GM roll, pitch [m]:",
        np.round(h.metacentric_height, 3),
    )
    # the free surface of the potential QTF, as the solve would integrate it:
    #   m.panels(role="free_surface"), m.free_surface_points("quadrature")
    results = m.solve(lambda done, total: print(f"  frequency {done} of {total}"))
    m.save_results(  # the model with its results
        OUT / "cylinder.hbr",
        overwrite=True,
    )
    print(results.summary())
    print("period [s]:           ", np.round(2 * np.pi / results.omegas, 2))
    print("heave added mass [kg]:", np.round(results.added_mass[:, 2, 2], 1))
    print("heave RAO [m/m]:      ", np.round(np.abs(results.motion[:, 0, 2]), 3))
    print("heave exciting force [N/m]:", np.round(np.abs(results.excitation.diffraction[:, 0, 2]), 1))
    print("GM roll, pitch [m] of the solve:", np.round(results.hydrostatics.metacentric_height[0], 3))
    # the result tree (reference/results §1): the groups excitation, body_points, field_points,
    # kochin, drift, qtf and hydrostatics, the owners lids[name] and bodies[name] (with their
    # pressure_surfaces[name] and tanks[name]), e.g. results.drift.pressure_integration,
    # results.qtf.potential.direct.difference, results.lids["gap"].elevation_diffraction,
    # results.bodies["cylinder"].tanks["aft"].pressure_radiation; a quantity by its name:
    # results.quantity("excitation_diffraction") (hb.quantities() lists them)
    print("saved:", OUT / "cylinder.hbr", "(read it with hb.load_results or hb.Model.load_results)")
    # the WAMIT-format numeric files of the results
    results.write_wamit(
        OUT / "cylinder",
        overwrite=True,
    )
    print("written:", " ".join(sorted(f.name for f in OUT.glob("cylinder.*") if f.suffix != ".hbr")))


if __name__ == "__main__":
    main()
