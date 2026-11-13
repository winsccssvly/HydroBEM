# Changelog

The user-visible changes of each version. The methods are described in `docs/theory/`.

## 0.3.0 (2026-11-13)

Rust API version 56, C ABI version 8, model record version 13, results file format 5, geometry
file (.hbg) version 3 (files of the earlier versions are refused). The engine is built for x86-64
with AVX2 and FMA. The C runtime is linked dynamically: the extension module uses the
`vcruntime140.dll` that Python for Windows installs, and the C ABI DLL needs the Visual C++
runtime (Visual Studio or the Visual C++ Redistributable).

**Changes of behaviour of a model that 0.2.0 also read** (beyond the renamed inputs below):

- `interior_surface` is `"detect"` by default: the wetted elements of the body's mesh lying in
  $z = 0$ (a waterplane cover) are taken as the interior surface; in 0.2.0 they were part of the
  wetted surface. Set `interior_surface = "none"` to have no interior surface;
- the centres of buoyancy and gravity and the waterplane centroid of `body.hydrostatics()` and
  `results.hydrostatics` are in body coordinates (they were relative to the rotation centre R);
- the second-order loads of a body have their moments about its rotation centre R (body axes), as
  the first-order results: the mean drifts by pressure integration and by the control surface
  (and their moments about the moving R, `_body_axes`), the quadratic QTFs of both methods, the
  potential QTF of both methods and Newman's approximation of those drifts; in 0.2.0 they were
  about the body origin $O$. $M_R = M_O - R \times F$. They are also the loads with the second-order
  displacement of R zero: the second-order rotation of a moving body is applied about R in every
  part (the pressure integration and the control surface, their QTFs and the body condition of
  the potential QTF), so that a QTF is the right-hand side of the second-order motion equation
  about R; for $R \neq 0$ the loads of a moving body therefore hold, against 0.2.0, the load of the
  rigid second-order translation $-H R$ (the hydrostatic load in $F_z$, $M_x$, $M_y$ of the quadratic
  part, the radiation load in the potential part); the horizontal forces and the yaw moment of
  the drifts are unchanged. With $R = 0$ (the default) every value is the same. The momentum drift
  stays in global axes, its moment about the global z axis (the total of all bodies has no single
  R) ([theory §1.1, §11.2](docs/theory/01-boundary-value-problem.md#11-coordinates-and-conventions));
- the low-order pressure drift (and its quadratic QTFs, and the drift of the tank fluid) takes the
  potential of a waterline panel expanded from its centroid to the waterline with the panel's
  velocity; in 0.2.0 the centroid value, half a panel height below, made the waterline term too
  small by a difference of first order in the panel height (on the bottom-mounted cylinder of the
  validation, 3.7e-1 to 2.1e-2 of the largest drift on 8 to 48 panels a quarter circle, now 3.9e-2
  to 9.0e-3) ([theory §11.4](docs/theory/11-mean-drift.md#114-the-numerical-methods-of-the-drifts));
- the low-order `"waterline"` interior surface is refined by Delaunay refinement (angles of at
  least 25 degrees but at sharper waterline corners, a conforming mesh, triangles graded from the
  waterline segments to twice their mean): its slivers are gone and it has two to four times
  fewer panels on smooth waterlines; results change within the discretization of the interior
  surface (on a cylinder 4e-5 of the peak from the `"graded"` surface)
  ([theory §14](docs/theory/14-geometry-preprocessing.md)).

The second-order potential force is compared with an independent solution for a fixed
bottom-mounted cylinder only; floating and several bodies, one wall and generalized modes rest on
identities, the agreement of two methods and convergence
([validation: not verified](docs/validation/not_verified.md)).

- **The geometry queries** ([queries](docs/reference/queries.md)). One word for each kind of
  surface, the same keys in the panels and the tessellation, "element" for a panel or patch of a
  mesh, the C functions of the model with the prefix `hb_model_`:
  - the roles are the singular of the input that makes the surface, in the argument `role` and in
    the column `role` alike (the codes unchanged): `wetted`, `interior_surface`, `dipole`,
    `dipole_damper`, `pressure_surface`, `lid`, `tank`, `fixed_damper`, `control_surface`,
    `free_surface`, `free_surface_plate`; `role` takes a name or a list of names (C: names
    separated by commas), nothing else (the list names `body`, `dipoles`, ... and the row words
    `wet`, `sheet`, `damper`, `pressure`, `plate`, `interior` are refused);
    `body.trimmed_mesh(name, role="wetted")` takes the roles of a body's mesh;
  - `Panels` and `Tessellation` carry the seven key columns `body`, `role`, `tank`, `owner`,
    `element` (one-based, as the ranges; before `record`), `part` (new: the piece of an element a
    trim cut) and `reflection`: within one call they identify one panels row, and every quad of
    the tessellation has the keys of the panels row of its element. A dipole has its place in the
    dipoles of its body or tank as owner and tank walls their tank (before -1); an element the
    engine generates is numbered from 1 within its surface (before 0); a trim that breaks a
    symmetry keeps the copy of a cut element in `reflection` (before (1, 1));
  - the tessellation covers every element the panels give for the same `role` and
    `include_reflections` (before: a body's own mesh, the lids and fixed dampers of their own):
    the sheets, pressure surfaces, plates, tanks, interior surfaces given and generated, the
    control surface (on request; the generated ones of a low-order model on their exact patches)
    and the free surface of the potential QTF; it takes `role`, and `divisions` (an integer, a
    pair, or `"spans"`: one quad per span of the potential discretization; Rust `[i32; 2]`, C
    `divisions_u`, `divisions_v`, both 0 the spans) instead of `nu`, `nv`;
  - `expand` is `include_reflections`; `vertex_counts` is `distinct_vertices`; `mesh.records()` is
    `mesh.elements()` (`hb.MeshElements`); `Location.record` is `Location.element` (C
    `HbLocation.element`, the location columns of the check and the diagnostics); `MeshQuality`
    has `element` and `part`; `mode_displacements(points, bodies, elements)`;
    `PointClasses.index` is split into `body`, `lid` and `tank`; `Counts.lid_panels`,
    `pressure_panels`, `per_lid_panels` are `lid_elements`, `pressure_elements`,
    `per_lid_elements`; `BodyCounts.wet`, `wet_expanded`, `sheet`, `interior` are `wetted`,
    `wetted_with_reflections`, `dipole`, `interior_surface`; `m.motion_matrices()` is
    `m.motion_equation()` (`hb.MotionEquation` with `ready`); `m.free_surface_points` and
    `mesh.patch_points` return `hb.FreeSurfacePoints` and `hb.PatchPoints`;
  - the mesh items `trim_records` and `source_panels` are `trim_elements` and `source_elements`
    (a model or results file of an earlier model record is refused);
  - the messages and documents say "element" for a panel or patch of a mesh ("element 12 of the
    mesh hull is taken by both ...");
  - C: `hb_panels`, `hb_tessellation`, `hb_mesh_quality`, `hb_counts`,
    `hb_motion_equation`, `hb_mode_displacements`, `hb_memory_estimate`, `hb_classify_points`,
    `hb_free_surface_points`, `hb_route`, `hb_solve`, `hb_new_results`, `hb_load_mesh`,
    `hb_load_csf`, `hb_load_fdf`, `hb_analytic_mesh` are `hb_model_` and the rest of the name,
    `hb_hbg_mesh` is `hb_model_mesh_from_hbg`, `hb_trimmed_body_mesh` is `hb_body_trimmed_mesh`,
    `hb_mesh_records` is `hb_mesh_elements`, `hb_patch_points` is `hb_mesh_patch_points`,
    `hb_write_mesh` is `hb_mesh_write`; `hb_model_tessellation` and `hb_body_tessellation` take
    `role` before the divisions; read the columns `element`, `part`, `distinct_vertices` and of
    `hb_model_classify_points` `body`, `lid`, `tank`; check `hb_abi_version() == 8`. Rust: as
    Python, with `Model::body_trimmed_mesh`, `mesh_elements`, `mesh_patch_points`, `mesh_write`,
    `lid_elements`, the `PanelRole` variants of the role names, `roles: &[&str]` and
    `FreeSurfacePoints`.
- **The input structure.** The inputs are grouped by what owns them and named as the results
  they give ([inputs](docs/reference/inputs-model.md); the table of the names below):
  - `m.water` is gone: `m.lids` and `m.fixed_dampers` (porous sheets fixed in space, global
    coordinates) are lists of the model; the field points in the sea are
    `m.first_order.field_points` ((N, 3), global), the points inside a tank the tank's
    `field_points` ((N, 3), body axes), each with its outputs in the results of its owner (below,
    the result structure);
  - `m.outputs` is split into `m.first_order` (radiation, diffraction, `diffraction_formulation`
    (before `options.diffraction_formulation`: the method of the first-order diffraction problem
    beside its switch, as the second order keeps its methods), motion, the field and body
    outputs, the field points) and `m.second_order` (flat switches named as the result paths:
    `drift_pressure_integration`, `drift_control_surface`, `drift_momentum_conservation`,
    `drift_bidirectional`, `qtf_quadratic_pressure_integration`, `qtf_quadratic_control_surface`,
    `qtf_potential_direct`, `qtf_potential_indirect`, `qtf_newman` (from every drift switched on;
    refused with none), `qtf_bidirectional`, `qtf_difference` and `qtf_sum` (both true by
    default; both false with a QTF on is refused), and the items of the potential QTF's free
    surface);
  - every surface names its mesh by the item `mesh` and its elements by the item `ranges`
    (`[(first, last), ...]`, one-based, inclusive; None every element) instead of a tuple; C
    `hb_set_mesh` is removed (`hb_set_text`, `hb_set_ints`) and `hb_get` has no mesh kind;
  - the surfaces are flat items, no child objects and no dicts: a body's switch
    `interior_surface` (`"detect"`, the default: the wetted elements lying in $z = 0$; `"waterline"`,
    `"graded"`, `"projected"`, `"mesh"` with `interior_surface_mesh`, `interior_surface_ranges`;
    `"none"`), its switch `control_surface` (`"none"`, the default, `"mesh"`, `"cylinder"`,
    `"ellipsoid"`, `"automatic"`) with the items `control_surface_mesh`, `_ranges`,
    `_waterline_term`, `_radius`, `_depth`, `_inner_radius`, `_semi_axes`, `_inner_semi_axes`,
    `_outer` (a switch, `"circle"` with `_radius` or `"polygon"` with `_boundary`), `_partitions`,
    `_panel_size`, and the items of `second_order` `free_surface_method` (`"panels"` (default),
    `"cells"`, `"mesh"`), `free_surface_panel_size`, `free_surface_inner_radius`,
    `free_surface_mesh`, `free_surface_ranges`, `free_surface_quadrature_points` and the six
    quadrature items (the object types `InteriorSurface` and `ControlSurface` of 0.2.0 are
    gone);
  - switches: where one of several inputs is used, a switch names it and the items of the choice
    not made are kept unused, neither read nor checked, nor refused or reset when set (six
    switches: `environment.frequency_input` (`"omegas"` default, `"periods"`: setting periods no
    longer replaces the omegas), a body's `inertia_input` (`"components"`,
    `"radii_of_gyration"` default), `interior_surface`, `control_surface`,
    `control_surface_outer`, `second_order.free_surface_method`); a needed item of the choice
    made, left unset, is refused by the check and before the solve, naming it
    (`bodies/b: control_surface = 'cylinder' needs control_surface_radius`); the schema gives per
    switched item its `switch` and `switch_values` (Python schema and `info(item)`, C
    `hb_schema_items`); `print(obj)` shows the items the switches choose and a last line
    `unused: ...` naming the others, while `items()` stays complete; the inertia and the
    frequencies: both sides have defaults (`periods` `[]` as `omegas`, `Ixx` ... `Izz` 0,
    `radii_of_gyration` zeros), only the chosen side is read, and no rule refuses the motions for
    a missing inertia (a zero inertia is computable; the inertia restricted to the free
    coordinates must still be positive semi-definite); `inertia_origin = "center_of_gravity"` needs
    `center_of_gravity` only when the motions are solved and the body has rigid modes;
  - the symmetry of a mesh names its planes (`"none"`, `"yz"`, `"xz"`, `"xz_and_yz"`; the GDF and
    SPL files are unchanged), also of `meshes.load_fdf`, `meshes.analytic` (default
    `"xz_and_yz"`), C `HbAnalyticShape.symmetry` and `hb_model_load_fdf`;
  - the objects of a list are made only by `add()` and `insert()` (as in the C ABI): a list is not
    assigned (`m.bodies = [...]`, `= []`: refused, naming `add()` and `clear()`), `add`, `insert`
    and `update` take the object's own items only (`dipoles=[...]`, `tanks=[...]`,
    `pressure_modes=[...]`, `pieces=[...]` refused: a nested object is added to its parent's list
    once the parent exists), `items()` gives the own items only (`add("b2", **b1.items())` copies
    them), and `clear()` empties a list (C `hb_clear`, Rust `Model::clear_list`; C
    `hb_replace_begin` / `hb_replace_end` and Rust `Model::replace_begin` / `replace_end` are
    removed; `update` stays a Python convenience, all or nothing in Python);
  - the queries follow: `m.panels(role="lid")` and `m.panels(role="fixed_damper")` (C
    `hb_model_panels`) replace `m.water.panels()` (C `hb_water_panels`); the panels of the
    potential QTF's free surface are `m.panels(role="free_surface")` (C `hb_model_panels`;
    `hb_free_surface_panels` is removed; role None gives every role but the control surface and
    the free surface) and its points `m.free_surface_points(zone)` (C
    `hb_model_free_surface_points`); the panel roles are named after the inputs that make them
    (the geometry queries above); `m.classify_points(points, tank=None)` (C
    `hb_model_classify_points` with a tank object, 0 the sea), `BodyCounts.symmetry_copies`;
  - the rule messages name the new paths (`second_order.drift_control_surface needs a
    control_surface for every body`, `lids/<name>: ...`, `second_order: free_surface_method =
    'mesh' needs free_surface_mesh`, `bodies/b: mode_mass must be 7 x 7 (6 + G, G = 1 generalized
    modes), not 6 x 6`).

  The automatic control surface always has its free-surface part (its switch `free_surface` is
  removed): the results change only for a model that switched it off.
- **The result structure.** The results follow the rules of the inputs
  ([reference/results](docs/reference/results.md); the whole tree in reference/results §1, generated
  from the engine's tree, and the table of the names below):
  - groups where one thing has variants or points with values: `results.excitation`
    (`diffraction`, `froude_krylov`, `scattering`, `haskind`), `results.body_points` (`points`,
    `ids`, `normals`, `areas`, `pressure_radiation`, `pressure_diffraction`, `velocity_radiation`,
    `velocity_diffraction`), also of a body; every group of points has its `points`
    (`results.field_points.points` is a quantity of the results, the points in the sea);
  - an object's results are under its owner: `results.lids[name]` (was `results.water.lids`,
    `WaterResults` is gone), the points of a tank in `results.bodies[b].tanks[t]` (`points` in body
    axes as given, `pressure_radiation`, `velocity_radiation`: the tank problem has radiation
    only), `results.field_points` the points in the sea only; a field point's message and
    location name it in its owner's list (`field point 2 of bodies/hull/tanks/aft lies above the
    free surface of its tank`); `Counts.field_points` counts the points in the sea and
    `Counts.tank_points` (C `hb_model_counts` `tank_points`) those in the tanks; the numeric files are
    unchanged (the points of the sea, then those of the tanks);
  - `drift.pressure_integration_moving` and `control_surface_moving` are
    `drift.pressure_integration_body_axes` and `control_surface_body_axes`;
  - the quantity names (`hb.quantities()`, `results.quantity(name)`, the results file, C
    `hb_results_quantity`, `Results.of`) are the result paths with "_" for ".", the points of a
    group the group's name (`excitation_diffraction`, `body_points_pressure_radiation`,
    `field_points`, `lids`, `lids_elevation_radiation`, `tanks`, `tanks_pressure_radiation`), the
    numbers of the owners of points by the same rule (`lids_ids`, `pressure_surfaces_ids`,
    `tanks_ids`); `Results.of` and `hb_results_put` take the points of the field outputs
    (`field_points`, `tanks`, `tanks_ids`) from the model, as the axes;
  - the hydrostatics of the results take the names and items of the query `body.hydrostatics()`,
    one name one meaning: `volumes` [B, 3] (the three estimates of the divergence theorem; was
    `volume`) and `volume` [B] (their mean, as `mesh.geometry().volume`), `displaced_mass` [B],
    `center_of_buoyancy`, `waterplane_area` [B] and `waterplane_centroid` [B, 2] (was `waterplane`
    [B, 3]), `stiffness_rho_g`, `mass`, `center_of_gravity`, `stiffness`, `metacentric_height`
    [B, 2] (NaN without the weight); the query's `volume` is the mean too (`volumes` the
    estimates); the centres of buoyancy and gravity and the waterplane centroid of the query and of
    the results are in body coordinates, as the input `center_of_gravity` (they were relative to
    the rotation centre R: their values change for a body with R not 0); the stiffness stays
    about R;
  - the results file (.hbr) is format 5, its quantities under the new names; format 4 files are
    refused, as any other version (no conversion: 0.3.0 is the first release of these names);
  - C: the result paths of `hb_results_select`, `hb_result_views`, `hb_result_groups` (`"lids.points"`,
    `"tanks.pressure_radiation"`, `"excitation.diffraction"`; the owner rule `tank`), the table of
    `hb_body_hydrostatics` (`volumes`, `volume`, ...); Rust `QuantityId` variants named as the
    quantities (`QuantityId::BodyPointsIds`, `Tanks`, `HydrostaticsVolumes`, ...), `Owner::Tank`,
    `Hydrostatics::volumes` and `volume`, `Counts::tank_points`.
  - The clear of a list (`clear()`, C `hb_clear`, Rust `Model::clear_list`) is refused while
    another object refers to one of its objects, with the message `bodies/ship refers to
    meshes/m2: the meshes are not cleared while an object refers to them`. The geometry file
    (.hbg) takes the new names of `symmetry`; model, results and geometry files of the earlier
    versions are refused, without conversion (the versions are given at the head of this section).
- **Second-order loads (QTFs).** The second-order force and moment at the difference and the sum
  frequency of every pair of the frequencies (`qtf_difference`, `qtf_sum`), of the heading pairs
  `qtf_heading_pairs` (`qtf_bidirectional`),
  [F, F, P, D] complex on the coordinates (the force and moment of each body on its six rigid
  coordinates, the generalized forces on the coordinates of its generalized modes), each part by
  its methods ([second-order](docs/theory/12-second-order-loads.md)):
  - the quadratic part (`second_order.qtf_quadratic_pressure_integration`,
    `qtf_quadratic_control_surface`; the quantities `qtf_quadratic_pressure_integration_*`,
    `qtf_quadratic_control_surface_*`); the difference
    coefficient of a wave with itself is the drift;
  - the force of the second-order potential (`qtf_potential_direct`, the second-order potential
    solved, and `qtf_potential_indirect`; the quantities `qtf_potential_direct_*`,
    `qtf_potential_indirect_*`), low order, for bodies fixed or floating with `first_order.motion` (the radiation of their motions in the
    free-surface forcing and their terms in the body forcing), one or several (one free surface
    about all), in the open sea or by one wall (`x_wall`, `y_wall`: the mirror problem of the
    bodies and their images, the free surface generated for both and integrated over the whole
    plane, halved). Its free surface is integrated in zones: the panelled zone to the inner
    radius, a quadrature zone of annuli whose fields come from a few circles (the outgoing wave
    plus a remainder interpolated in 1/r), the outer circle and the far field by series. The
    results do not depend on the thread counts, and the other outputs of the model (body points,
    field points, ...) are left as they are. Refused with a reason: the higher order,
    `orthogonal_walls` and a channel, damping lids and fixed dampers, pressure surfaces,
    tanks, thin sheets and dipole dampers of a body (before the solve), an inner radius not beyond
    the bodies, a waterline too coarse for the generated panels, a waterline inside another (a
    moonpool) with the generated panels (a free-surface mesh can cover such water), and with a
    wall the cells or a given free-surface mesh. Checked against the semi-analytic solution of
    Kim & Yue (1990);
  - Newman's approximation (`qtf_newman`, from every drift switched on: by pressure integration,
    control surface or momentum conservation; and `qtf_newman_average`, "geometric" (default) or
    "arithmetic"): `qtf_newman_<method>_difference` [F, F, H, D] (momentum conservation
    [F, F, H, 3]), real, of the equal headings; Python `results.qtf.newman.<method>.difference`.

  The total is the sum of a quadratic and a potential part (without the force of the second-order
  motions). The memory estimate holds the two sub-problems of the potential QTF.
- **The free surface of the potential QTF.** Items of `second_order` give its zones: the panelled
  zone (`free_surface_method` "panels" (default) or "cells", generated, with
  `free_surface_panel_size`, `free_surface_inner_radius` (auto 4a); or "mesh", any panels mesh in
  $z = 0$, e.g. the panels of an FDF file read by `m.meshes.load_fdf`, with `free_surface_mesh`,
  `free_surface_ranges`, `free_surface_quadrature_points`), the quadrature zone
  (`quadrature_annuli`, `quadrature_radius_step`, `quadrature_radial_nodes`,
  `quadrature_azimuthal_nodes`), the outer circle (`outer_circle_segments`) and the far field
  (`expansion_order`). Its queries, without a solve: `m.panels(role="free_surface")` (the
  panelled zone) and `m.free_surface_points(zone)` ("panelled", "quadrature", "outer_circle");
  Rust `Model::load_fdf`, `panels(&["free_surface"], _)`, `free_surface_points`; C
  `hb_model_load_fdf`, `hb_model_panels`, `hb_model_free_surface_points`.
- **Second-order loads with generalized modes.** The pressure drift
  (`drift_pressure_integration`), its quadratic QTFs and the potential QTF (both methods, the open sea and
  one wall) of bodies with generalized modes, low order: their displacement superposed on the rigid
  motion, its gradient turning and stretching the surface element, their motions in the forcing
  and the body condition ([theory §12.15](docs/theory/12-second-order-loads.md#1215-generalized-modes)). The second-order
  loads are on the coordinate axis as the first-order exciting forces: the rigid rows as for rigid
  bodies, the generalized forces of the modes on their coordinates in the same quantities
  (`drift_pressure_integration`, `qtf_quadratic_pressure_integration_*`, `qtf_potential_*`), NaN
  where a method has none (the pressure coordinates; the generalized coordinates of the control
  surface); a body's view (`results.bodies[b].drift.pressure_integration`) has its six rigid rows
  and its own modes. Refused with a reason: the higher order, generalized modes with pressure
  surfaces or tanks, and a mode whose pieces tear the wetted surface open (a jump of the
  displacement between two elements).
- **Free-surface plates.** `body.free_surface_plates` (`FreeSurfacePlate`, `mesh` and `ranges`):
  massless rigid plates lying in $z = 0$ with the water below them only, moving with the body's
  rigid modes and with the generalized modes whose pieces name their elements (an oscillating
  water column's piston, a mat of zero draft, a moonpool cover); a body may consist of plates only
  ([theory §9.7](docs/theory/09-special-surfaces.md#97-rigid-plates-in-the-free-surface), [reference/inputs-bodies §2.4](docs/reference/inputs-bodies.md#24-free-surface-plates)). Both methods
  and both diffraction formulations, radiation, diffraction, motions, field and body outputs,
  $\omega = 0$, the open sea, walls and a channel, bodies with internal tanks, the mean drift by
  momentum conservation (the plate's term $-K$ times its source integral in the Kochin function,
  [theory §11.1](docs/theory/11-mean-drift.md#111-mean-drift-force-and-moment-momentum-conservation))
  and Newman's approximation of it, and the added mass at $\omega = \infty$ (the plate a rigid thin
  sheet in $z = 0$ of the odd-image problem, its coincident image doubling its kernels: a disc has
  $(4/3)\rho a^3$, half of the disc in unbounded fluid; [theory §9.7](docs/theory/09-special-surfaces.md#97-rigid-plates-in-the-free-surface));
  for plates whose every edge lies on a wall (a cover on its own body's wall, a piston in a tube
  of another body or moved by a generalized mode) or between plate elements, also the mean drift
  by pressure integration, its quadratic QTFs and Newman's approximation of it (the plate part of
  the body's surface with no hydrostatic pressure, the band of a wall's waterline under a plate
  with the plate's pressure at its top; [theory §11.2](docs/theory/11-mean-drift.md#112-mean-drift-force-and-moment-pressure-integration))
  and the potential QTF of both methods (the plate part of the body in every body integral, the
  free surface ending at the plates, the line integral of a wall's waterline under a plate with the
  plate's sliding along the wall; the generated panels or a given mesh; [theory §12.7](docs/theory/12-second-order-loads.md#127-the-body-condition-of-a-floating-body)),
  a generalized mode then free to jump vertically between a plate and a wetted element (the plate
  sliding along the wall); their area is part of the body's waterplane. Under the same rule of
  the edges, the mean drift by the control surface, its quadratic QTFs and Newman's approximation
  of it for the plates of the body inside its own control surface (no waterline term under a
  plate, the plate's area in the waterplane of the hydrostatic part; the automatic control surface
  leaves the waterline under the plates out, so a covered moonpool has one;
  [theory §11.3](docs/theory/11-mean-drift.md#113-mean-drift-force-and-moment-control-surface)).
  Refused with a reason: $\omega = \infty$ with a thin sheet hanging from a plate, the pressure and
  control-surface drifts and their QTFs with a plate edge bordered by free surface (a mat, a dock),
  the pressure drift with pressure surfaces, a plate body inside another body's control surface
  (a second body there), the potential QTF with a plate edge bordered by free surface (its forcing
  is not integrable there: refused for good) or with the cells of its free surface, wavemakers, a
  plate given as a mesh on a trimmed body and a trimmed body of plates only (any angle tilts them
  out of $z = 0$). The plate of a trimmed body is generated instead (the items `inside` and `body`
  of `FreeSurfacePlate`): the water of $z = 0$ inside the waterline loop
  around a point, triangulated as the interior surface of the trimmed hull (low order the refined
  `"waterline"` triangulation, higher order the patches of the trimmed patch mesh), its edges on
  the waterline; with `body` a plate body of its own over another body's trimmed loop (the piston of
  a trimmed tube); its elements numbered from 1 in the queries, role `free_surface_plate`
  ([theory §14](docs/theory/14-geometry-preprocessing.md#free-surface-plates-of-a-trimmed-body),
  [reference/inputs-bodies §2.4](docs/reference/inputs-bodies.md#24-free-surface-plates)).
  The plates of a tank's free surface (a deck over part of a tank, a full cover):
  `tank.free_surface_plates`, elements of the body's mesh or a mesh of their own at the tank's
  `surface_level`, plates of the tank problem moving with the body at every frequency and both
  limits, both methods; their area leaves the tank's free-surface restoring, a fully covered tank is
  a closed container (its added mass $\rho_T V_T$ in every translation, heave defined at
  $\omega = 0$; the low order pins the constant of its potential, defined only up to it, at every
  frequency, and that of every tank at $\omega = 0$, so the results no longer take an arbitrary
  amount of it from round-off); the rules of the plates apply in the tank's free surface at the model's check, and
  the queries give them the role `free_surface_plate` with the tank's key; refused with the
  pressure-integration drift and on a trimmed body, and a plate of the sea or a body of its own on
  a tank's fluid (a floating roof) is refused
  ([theory §10.2](docs/theory/10-generalized-modes-and-tanks.md#102-internal-tanks), [reference/inputs-bodies §2.2](docs/reference/inputs-bodies.md#22-tanks)).
  The panels of a plate have the role `"free_surface_plate"` (`PanelRole::FreeSurfacePlate`). A
  plate overlaps no other surface of $z = 0$
  (another plate, a lid, an interior surface, a pressure surface in $z = 0$), and a thin sheet that
  hangs from a plate or a damping lid must meet it along the edges of its elements (refused
  otherwise: an element across the sheet cannot hold the jump of the potential, and the results did
  not converge; [reference/inputs-rules §1](docs/reference/inputs-rules.md#1-model-rules-validation) rule 3). Low order: the body points of a
  pressure surface or plate on a mesh of its own are numbered on after the elements of the body's
  mesh (they had element 0, which the file writer refused). Field points on a plate of a mesh of
  its own are refused as inside the waterplane, and a body of plates only takes `body_points`.
  The metacentric height of a body without a volume (sheets or plates only) is NaN (it was not
  finite). With a plate, `diffraction_formulation = "auto"` solves the total potential (the
  scattered one with low-order thin sheets): under a plate on a deep water column the incident
  and the scattered potentials nearly cancel, and the low-order scattered formulation's error
  there was about four times the total one's (theory §9.7). The elements of a plate, a pressure
  surface or a given interior surface may point either way in both methods, curved patches too:
  a B-spline or analytic patch given the other way is turned over by its parameter u reversed (the
  same surface on the same spans), as panels and flat patches were already; so the `waterplane`
  part of an analytic shape is a plate as it is, and a radial line patch may run from the centre
  to the edge (before, a curved patch facing the other way was refused).
- **Massless coordinates.** The motions need the inertia of a body's free coordinates (the rigid
  inertia and `mode_mass`) to be positive semi-definite only, no longer positive definite: a
  `mass` of 0 (a massless piston or lid) and a zero mode mass (a free-surface mode of a moonpool)
  are accepted, their inertia the added mass, the motions continuous as the mass tends to 0
  ([theory §3.1](docs/theory/03-motions-and-hydrostatics.md#31-rigid-body-motions)). Refused with the coordinates named: a negative kinetic
  energy (now also of `mode_mass`), and a massless direction that moves no water. `mass` is an
  item of rule `>= 0` (its message "must be a finite number >= 0").
- **Thin sheets in the free surface refused.** A dipole, a dipole damper or a fixed damper lying
  in $z = 0$ is refused with the reason (it coincides with its free-surface image); a rigid plate
  there is a free-surface plate. Before, the solve failed with a non-finite system matrix.
- **Bidirectional drifts.** The pressure and control-surface drifts of every pair of headings
  (`drift_bidirectional`, for every drift method), the diagonal of the difference
  quadratic QTFs of the heading pairs.
- **The result tree.** The views of the results by group and owner are the engine's: each result
  quantity names its view (`drift.pressure_integration`, `qtf.quadratic.pressure_integration.sum`,
  `lids.elevation_radiation`, `added_mass` of a body, ...) and the rule of an owner's part
  (the columns of a body, its coordinates, its body points, the points of a lid, of a pressure
  surface or of a tank). Rust: `Results::select`, `Results::selection`, `Results::body_columns`,
  `QuantityId::group`, `QuantityId::owner`, `result_views`, `result_groups`; C:
  `hb_results_select` (a copy), `hb_results_body_columns`, `hb_result_views`, `hb_result_groups`.
  The Python classes of the results (`Results`, `BodyResults`, `results.excitation`,
  `results.body_points`, `results.drift`, `results.qtf`, `results.lids[name]`,
  `results.field_points`, `results.kochin`, `results.hydrostatics`, the pressure surfaces and the
  tanks of a body) are generated from the tree (`scripts/gen_result_views.py`) and read the
  engine's selection.
- **One attribute path of the results (Python).** In 0.2.0 every quantity was an attribute of
  `Results` named as the quantity; now the tree is the only attribute path: the axes and the
  quantities read by body keep their names (`results.omegas`, `results.added_mass`,
  `results.motion`, ...), the others are read at their view (`results.field_pressure_radiation` is
  `results.field_points.pressure_radiation`, `results.excitation` `results.excitation.diffraction`,
  `results.hydrostatics_stiffness` `results.hydrostatics.stiffness`, `results.kochin_angles`
  `results.kochin.angles`, the drifts and QTFs under `results.drift` and `results.qtf`; the tree of
  reference/results §1, the table of the names below), and every quantity by its name with
  `results.quantity(name)` (also `lids_ids`, `pressure_surfaces_ids`, `tanks_ids` and the whole
  arrays of the lids, pressure surfaces and tanks, which have no view of the results).
- **Names by kind and method.** The inputs and results are named kind first, then the method or
  part, written out (the Rust convention, the Python attributes the same):
  - the object `waves` is part of `environment` (`frequency_input`, `omegas`, `periods`,
    `headings`);
  - `outputs.rao` is `first_order.motion`;
  - the drifts: the switches `second_order.drift_pressure_integration`, `drift_control_surface`,
    `drift_momentum_conservation`, with `drift_bidirectional` and
    `drift_momentum_conservation_angles`; the
    results `drift_pressure_integration`, `drift_control_surface` [F, P, D] (with their
    `_body_axes`), `drift_momentum_conservation` [F, P, 3], complex, of the heading pairs
    `drift_heading_pairs` [P, 2] (the unidirectional drifts of 0.2.0 are their pairs $(\beta, \beta)$;
    `momentum_drift_pairs` is gone);
  - the exciting forces: `excitation.diffraction`, `.froude_krylov`, `.scattering`, `.haskind`
    (the quantities `excitation_diffraction`, ...); the body item `excitation_method` takes
    "diffraction" or "haskind";
  - `added_mass_zero_frequency`, `added_mass_infinite_frequency`;
  - Rust `QuantityId::HydrostaticsCenterOfBuoyancy` and `HydrostaticsCenterOfGravity` (the
    quantities `hydrostatics_center_of_buoyancy`, `hydrostatics_center_of_gravity`).
- **One set of functions in Rust, C and Python.** The C ABI reads an FDF file
  (`hb_model_load_fdf`), gives the panels and points of the free surface (`hb_model_panels` with
  the role "free_surface", `hb_model_free_surface_points`) and the columns of a body (`hb_results_body_columns`). Python moves an
  object in its list (`m.bodies.move("b", 0)`, any list), reads the warnings of the last check
  (`m.warnings`), every diagnostic of results (`results.diagnostics`, errors included) and the
  quantities they have (`results.available`); `ItemInfo.dtype` is "real", "int" or "text" as in C.
  The `role` of `panels` and `body.panels` (Rust, Python, C) takes the role names of the geometry
  queries above. `Results.of` checks the quantities given together at once (the drift pairs,
  `drift_heading_pairs`, the body-point companions), with a message naming the one missing;
  `m.free_surface_points` returns read-only arrays; the import of the package asks the system
  for AVX2 and FMA. Rust: `Model::solved_body_counts`, `Value::as_texts`, `Results::check_given`,
  `ItemValue` exported; the fields of `Results` are private (its methods read them, `Results::put`
  changes them); `Document::obj` and `obj_mut` are not public (`get_obj` reads), and a change of a
  `Document` with the id of another document or of a removed object is `ErrorKind::Argument`. A
  missing quantity's message names it; an array of more than 8 (items) or 4 (quantities)
  dimensions, a negative extent or more than $2^{40}$ elements is `ErrorKind::Value` naming the item
  or quantity.
- **Rust: the item values and the route as types** (Rust API 56). `ItemValue` is an enum of its
  kinds (`Unset`, `Bool`, `Int`, `Real`, `Text`, `Name` for an int or real item given as one of
  its names, `Reals`, `Ints`, `Texts` with their shapes), read by `is_set`, `is_array`, `is_name`,
  `dtype`, `boolean`, `integer`, `real`, `text`, `shape`, `reals`, `ints`, `texts`; it was a struct
  of every field. `Document::set` takes a scalar in the variant of its item's kind (an `Int` for a
  real item is taken as a `Real`, as integers in a real array) and refuses another variant
  (`ErrorKind::Type`). `Model::route` returns a `Route` (`Hydrostatics`, or `Solve` with its `Method`,
  the depth and the `Domain`, all three exported and `#[non_exhaustive]`), its name `Route::name`
  (also its `Display`; a route equals its name as a `str`, `&str` or `String`); it returned the
  name. C `hb_model_route`, Python `m.route()`, the
  results' meta data and the files are unchanged.
- **The C ABI checks its arguments.** Every function checks its arguments, the out-pointers first, before it changes
  the model or allocates (a refused call changes nothing and leaks nothing). `hb_move` takes an
  `int64_t` place and refuses a negative one or one past the end (`HB_ERR_ARGUMENT`); `hb_add`
  refuses a negative place other than -1; `hb_results_body_columns` takes an `int64_t` body and
  refuses one out of range; an array argument beyond $2^{40}$ elements (or whose shape overflows) is
  refused before it is read; an unknown zone of `hb_model_free_surface_points` is `HB_ERR_VALUE`;
  `hb_last_error_location(NULL)` returns `HB_ERR_ARGUMENT` and keeps the last error;
  `hb_model_free` must not run concurrently with another call on the model. A processor without
  AVX2 and FMA is checked at the first call (a best effort: reliable on a processor with AVX but
  without AVX2 or FMA): every function that returns a status then returns `HB_ERR_UNSUPPORTED`
  with a message.
- **Progress by worker, phase and second-order step.** Every frequency runs on
  a frequency worker of a fixed number 0 to `frequency_threads - 1`; `m.progress()` gives it per
  frequency and limit (`FrequencyProgress.worker`), the solve's `phase` ("preparation",
  "limits", "first_order", "second_order", "finishing", then "done", "cancelled" or "failed"),
  each worker's state ("running", "idle", "done") with the item it is on (`workers`,
  `hb.WorkerProgress`), and the steps of the second order with their items (`second_order`,
  `hb.SecondOrderProgress`: the rows of the quadratic QTFs, the frequencies of the potential
  QTF's sub-problems, its fits, the batches of its outer integral and its pair frequencies). The
  second-order steps run by item on the same workers. The callback's `hb.Progress` names the
  worker, the phase and a second-order item (`worker`, `phase`, `step`, `item`), with the new
  stages "second_order", "phase" (always reported) and "worker_idle"; the sub-problems of the
  potential QTF now report their stages as items of their steps. A cancel is checked inside the
  second-order steps (before every pair of the quadratic QTFs, every radius of the outer integral
  and every pair of the potential QTF). C: `HbProgress` gains `worker`, `phase`, `step` and
  `item` (`frequency` -3 a second-order item, -4 the solve), the constants `HB_STAGE_SECOND_ORDER`,
  `HB_STAGE_PHASE`, `HB_STAGE_WORKER_IDLE`, `HB_PHASE_*` and `HB_STEP_*`; `hb_model_progress`
  gives `phase`, `limit_worker`, `worker`, the `worker_*` columns per worker and the `step*` and
  `item_*` columns of the second order.
- **The source patch of `trim_elements` is one-based** (the column 1 of the item of a
  `trimmed_patches` mesh), as the other element numbers of the model; model files (and the
  models in results files) of the earlier model records are refused.
- **The interior surface of a trimmed higher-order body** (`"waterline"`) from the waterline loops
  closed along the symmetry planes: a fan for one star-shaped loop, otherwise a triangulation of
  the regions with their holes (moonpools, several hulls in one mesh, a loop not star-shaped); the
  surface of a trimmed flat-patch body is made (it was refused). A waterline that does not close
  or loops that cross or touch are refused.
- **One polygon triangulation.** The `"waterline"` interior surface, the interior surface of a
  trimmed higher-order body and the generated panels of the potential QTF's free surface
  triangulate a region with holes the same way (holes bridged, ear clipping, Delaunay flips;
  [theory §14](docs/theory/14-geometry-preprocessing.md)). The low-order `"waterline"` surface of a waterplane
  with holes or of a non-convex waterline may change (better-shaped triangles); waterline contours
  that touch are refused like crossing ones.
- **A generated plate at a point on a symmetry plane.** `free_surface_plates.add(inside=(x, y))`
  with the point on a plane of the kept symmetry (y = 0 of a half hull `"xz"`, x = 0 of `"yz"`, or
  both) was refused as "the point lies on the waterline" when the body's reference length was at
  least the waterline's extent (the edges that close the stored part's loop along the plane were
  taken as waterline). Such a point now gives the plate of the whole loop, that of a nearby point
  off the plane; other plates do not change.
- **A control surface of flat patches.** A body whose control surface is a mesh of flat patches
  (a CSF file of the higher-order form with `ICDEF = 0`, or a `"flat_patches"` mesh) aborted the
  check of the model with an engine panic (index out of bounds): the rule that a control surface
  holds its body alone evaluated the patches as spline patches, which flat patches have no data
  for. The flat patches are now taken by the bilinear map of their vertices; the drifts are those
  of before, and another body inside such a surface is refused as by the other surfaces.
- **Plate edges against another body's plate, and beside a long wall.** An edge of a free-surface
  plate against a plate of another body (two plate bodies touching in $z = 0$) counted as lying
  between plate elements, so the pressure and control-surface drifts, their QTFs and the potential
  QTF were solved without a term for the jump of the vertical velocity there; such an edge is now
  refused as an edge bordered by free surface. An edge counted as lying on a wall within a tenth of
  the longer of the plate edge and the wall edge, so a long flat side (one panel or patch along a
  barge) held every edge of a small mat floating beside it; the bound is now a tenth of the plate
  edge's own length ([theory §11.2](docs/theory/11-mean-drift.md)). Covers whose edges lie on their
  walls are routed as before. The refusals are located at the body of the plate they name.
- **The higher-order waterline under a plate** finds the plate within a tenth of the plate's
  longest side, as the low order and the potential QTF do (it took a fortieth of the waterline
  panel's side only): a cover of chords over a curved wall whose sag exceeds that takes the plate's
  band in the pressure drift instead of the free surface's.
- **The queries of a body leave its disabled children out.** `body.panels()`, `tessellation()`,
  `mesh_quality()`, `hydrostatics()` and `trimmed_mesh()` of a body took the body's own disabled
  tanks, pressure surfaces, sheets, plates and couplings as enabled; they are left out, as the
  solve leaves them, so that `body.panels()` is the body's rows of `m.panels()` (also for a disabled
  body, which is taken as enabled itself).
- **Rule messages name the stored element.** The rules of the geometry (an element in $z = 0$, a
  thin sheet in $z = 0$, the parity of a generalized mode) named an element by its place in the
  engine's mesh of the body, which differs from the stored element when the body's ranges leave
  elements out; the message and `Location.element` now give the stored element (the numbering of
  `ranges` and `Panels.element`).
- **The keys of a given free-surface mesh.** `m.panels(role="free_surface")` and
  `m.tessellation(...)` numbered the panels of a given `free_surface_mesh` with its symmetry copies
  as one running list with reflection (1, 1), and `include_reflections=False` kept every copy; the
  rows are now keyed as every input mesh: the stored element, the copy, only (1, 1) without the
  reflections, the reflected quads with the orientation kept.
- **A massless mode of a tank's plate.** A massless generalized mode that moves only a plate of a
  tank's free surface given as a mesh of its own was refused as moving no water; it is accepted.
- **Smaller fixes.** The footprint of a patch control surface is sampled 16 points a side (a
  cylinder of four patches of 90 degrees lay up to 7.6 % of its radius outside the polygon of the three
  samples a patch side, so a body there was not refused); `m.panels(role=[])` is refused (it gave the default roles);
  `hb_model_classify_points` takes `HB_NO_OBJECT` for the sea as it takes 0.
- **Tanks in a channel.** A body with internal tanks in a channel solves; the tank problem kept the
  channel's width without its domain, and the solve was refused ("channel_width is only allowed
  with domain 'channel'"). The tank problem is that of the open sea, as by walls.
- **Channel without reflection.** `channel_reflect = 0` is the open sea of the same depth: no
  wall image is summed, so the frequencies at which the channel width is a multiple of the half
  wavelength solve (they were refused). The factor scales every wall image uniformly
  ([theory §1.2](docs/theory/01-boundary-value-problem.md#12-field-equations-and-boundary-conditions)).
- **Velocities with an interior surface.** The low order's fluid velocities (and with them the mean
  drift by pressure integration and the field velocities) next to the waterline of a body with an
  interior surface were off by a few per cent: the interior free surface now carries the vertical
  velocity of the exterior flow at the waterline, which keeps the source strength regular there.
  Results without an interior surface do not change.
- **Pressure and interior surfaces either way up.** Their elements may point up or down, also mixed
  within one surface, as for lids: low-order panels and flat patches are turned to the convention
  (pressure surfaces downward, interior surfaces upward); the results do not change.
- **Less memory in the low order.** The system matrix of each symmetry character is factored in
  place, without a copy: the memory estimate and the peak drop by one matrix per character solved
  at once; the results are unchanged.
- **Less memory for the results.** The results of a solve take the engine's output arrays over
  instead of copying them, hold each quantity once (the NumPy arrays and the C pointers borrow it,
  the results returned by `Model.solve` and `Model.results` share it with the model): a solve whose
  field outputs dominate needs about half the memory at its peak (e.g. 332 MiB instead of 574 for
  12 000 field points at 40 frequencies), and `memory_estimate` counts the outputs once (theory
  §13.3). The results are unchanged.
- **A closer memory estimate.** `memory_estimate` also counts the higher order's per-row records
  of the static rows (one per field point and velocity stencil point), the field points of the
  geometry, which are now allocated at their length, and the free-surface points of the potential
  QTF held through its second order: within 10 % of the measured peak of field-output and
  potential-QTF solves, the rest fixed costs.
- **The memory check before a solve.** A solve whose memory estimate (its resident peak)
  exceeds the physical memory available is refused before it starts, with a `ValueError` (Rust
  `ErrorKind::Value`, C `HB_ERR_VALUE`) that gives both and the remedies, instead of ending the
  process on a failed allocation; `options.memory_check = false` solves without the check.
- **The checks before a solve are those of the model check.** A solve applies the checks of
  `check()` before it starts, the zones of the potential QTF's free surface among them: a free
  surface that cannot be built is refused before any frequency is solved also with
  `options.memory_check = false` (it was refused after the first-order solve), with the same
  message. What is refused and the results are unchanged. With `options.memory_check = true` an
  unbuildable free surface is now refused before the size refusals (a model too large for the
  solve with one is refused for its free surface), the refusals coming in the order of `check()`.
- **Faster assemblies.** The wave kernels of the low- and higher-order assemblies, in infinite and
  finite depth, are evaluated four pairs at a time with AVX2: a low-order assembly in infinite
  depth about 1.5-1.7 times faster on one thread (1.3-1.4 in finite depth), a higher-order solve in
  infinite depth about 1.35-1.5 times; the results are unchanged to the bit.
- **The numeric files and the `.out`.** They are written all or none: a failure (a target held
  open) restores the old files and leaves no temporary file. The `.out` names HydroBEM and its
  version in its header line (`HydroBEM <version>  summary`), the method as `Higher-order method`
  or `Low-order panel method`, the diffraction formulation solved (`"auto"` resolved, also with
  thin sheets) and one line per lid.
- **Geometry.** GDF and CSF files: text after the numbers of a line is a comment (as `end of panel
  1` after a panel's last vertex); a line that begins with text is still refused.
  `mesh.geometry()`: the centre of volume and the waterplane centroid are 0 when their divisor is 0
  (they were NaN). The cylinder with dipole discs refuses a moonpool as wide as the hull. The
  low-order `"waterline"` interior surface refuses more than 200 000 panels. A wetted element
  lying wholly in the free surface $z = 0$ (a waterplane cover among the hull's elements) is
  refused: it is given as the interior surface (the method "detect", the default, takes such
  elements) or a lid, or left out of the body's elements.
- **Control surfaces hold their body alone.** A control surface whose footprint (the convex hull
  of its vertices in the plane) reaches another body or holds a damping lid is refused: the
  momentum balance gives the force on everything inside it, and the free surface inside it is
  taken as free ([theory §11.3](docs/theory/11-mean-drift.md#113-mean-drift-force-and-moment-control-surface), reference/inputs-bodies §2.7).
  The footprint of a patch mesh (every generated control surface, higher-order bodies and lids)
  is taken from points of its patches, and the free-surface plates of meshes of their own are
  part of their body (an earlier build saw no vertex of a patch mesh, so the rule refused nothing
  with a generated control surface).
- **The check covers the routes.** `m.check()` (Rust `Model::check`, C `hb_model_check`) also
  refuses a valid model that has no solver route, with `UnsupportedProblemError`
  (`ErrorKind::Unsupported`, `HB_ERR_UNSUPPORTED`) and the message of `m.route()` and the solve,
  for example the mean drift by momentum conservation in a channel; in 0.2.0 it passed such a
  model. A model that passes `m.check()` has a route. With the potential QTF it also makes the
  zones of the free surface, so their refusals (`ValueError`) no longer wait for the solve
  ([reference/inputs-rules §1](docs/reference/inputs-rules.md#1-model-rules-validation)).
- **The queries of a body check that body alone.** `body.hydrostatics()`, `panels()`,
  `tessellation()`, `mesh_quality()` and the generated interior surface of `trimmed_mesh()` apply
  the rules of the environment and of the body's geometry and surfaces, as
  [reference/queries](docs/reference/queries.md) says: not those of the frequencies, the orders,
  the options or the other bodies (in 0.2.0 a body's hydrostatics without frequencies, or with the
  motions on and another body without a centre of gravity, were refused). Their rows are numbered
  as before, as those of the body in the model's queries.
- **The queries of the free surface check the free surface alone.** `m.panels(role="free_surface")`,
  `m.tessellation(role="free_surface")` and `m.free_surface_points(zone)` (C `hb_model_panels`,
  `hb_model_tessellation` with the role `"free_surface"` alone, `hb_model_free_surface_points`)
  apply the rules of the free surface's items, of the bodies' geometry and surfaces and of the
  environment with its walls, not those of the frequencies, the orders' switches or the other
  options (in 0.2.0 they applied every rule of the model). A zone whose auto values the
  frequencies decide (the auto panel size of the generated panelled zone, the auto step and
  azimuthal nodes of the quadrature zone, the auto segments of the outer circle) needs a
  frequency between 0 and $\infty$ and is refused without one, naming the items
  ([reference/inputs-orders §3](docs/reference/inputs-orders.md#3-free-surface-of-the-potential-qtf-items-of-second_order)).
  The answers of a valid model are unchanged.
- **Messages.** "second_order: free_surface_method = 'mesh' needs qtf_potential_direct or
  qtf_potential_indirect";
  the order and the symmetry of the dipoles' and the interior meshes name the dipoles; the
  generated control surfaces are labelled "cylinder control surface" and "ellipsoid control
  surface"; the analytic control surfaces of a CSF file are named by their ICDEF and their
  parameters; a low-order pressure surface in $z = 0$ is judged by all its panels, as in the higher
  order.
- **Environment hooks.** The environment variables for the validation studies and the tests
  (`HB_DIAG_DOUBLE`, `HB_VALIDATE_<name>`, `HB_WATERLINE_CENTROID`, `HB_AVAILABLE_MEMORY`,
  `HB_FIELD_STATICS_LIMIT`, `HB_PROFILE`) that are set are listed in the meta data of the results (`environment_hooks`);
  one that changes the results is a warning of the solve (code 9). An unknown `HB_VALIDATE_`
  name, or a hook whose value is not a number, refuses the solve.
- **Build.** A build without AVX2 and FMA is a compile error (the flags of `.cargo/config.toml`; a
  `RUSTFLAGS` of one's own must name them too).
- **The finite-depth Green function tables regenerated.** Their generator
  (`scripts/gen_green_finite_tables.py`) now reproduces them bit for bit whatever the BLAS library
  of NumPy and its SIMD dispatch (no least-squares fit or eigenvalue routine; the complex
  arithmetic of the quadrature in real operations) and has a `--check`. The coefficients changed
  in their last bits only (at most 8.6e-15, the accuracy against mpmath unchanged); finite-depth
  results change by at most 4.3e-13 of their peaks.
- **The validation report by case.** [docs/validation/README.md](docs/validation/README.md) indexes
  one page per case (`docs/validation/cases/`): the problem, the reference, the meshes of both
  methods (drawn by `validation/mesh_figure.py`), the low and the higher order against the reference,
  and a reading; with the groups of cylinders and spheres, the OWC chamber and the two-body OWC, the
  heave plates, the fish tanks and the free-surface plates. How the comparisons are made:
  `method.md`; what is not verified: `not_verified.md`. The single `report.md` is gone.
- **The documentation by reader and topic.** [docs/README.md](docs/README.md) gives the reading
  paths and the folders, which replace the single files of before: the user guide by topic
  (`docs/guide/`), the theory in numbered chapters with the physics, the method and the accuracy
  of each topic together (`docs/theory/01-...md` to `14-...md`; the former physics, numerics and
  second-order documents), the reference with the inputs by object (`docs/reference/inputs-*.md`,
  `queries.md`), the results (`results.md`) and the output files (`file-formats.md`), and the
  error maps in `docs/development/error-maps/`. The code cites `theory §11.2`, `guide/<file> §N`
  and `reference/<file> §N`; `scripts/rewrite_doc_refs.py` rewrites the citations of the former
  documents and checks the links of the documents. The error map of the finite-depth tables
  (`green_fin_error_map.md`) is written, as the others, in formulas in the notation of theory §8.2
  with captioned tables (its tables unchanged).
- **The user guide rewritten for new users.** Each page of `docs/guide/` begins with what its
  topic is for, then shows what to do, with examples that run in order from
  [getting-started.md](docs/guide/getting-started.md) (a first problem step by step) to the
  second-order loads. The organization of the model, changing and checking it, and the model and
  results files have a page of their own,
  [working-with-the-model.md](docs/guide/working-with-the-model.md); the other pages keep their
  names, and their sections are renumbered.
- **The examples of the free-surface plates extended.** `examples/floating_owc_cover` computes the
  added mass at $\omega = \infty$, the mean drift by momentum conservation and by pressure
  integration side by side, and the difference-frequency QTFs (quadratic and potential);
  `examples/dock` the added mass at $\omega = \infty$ and the momentum drift (the one drift of a plate
  with a free edge); `examples/floating_owc_piston` its piston's added mass at $\omega = \infty$. New:
  `examples/trimmed_moonpool_cover` (a pitched moonpool hull with its cover generated by
  `free_surface_plates.add(inside=(x, y))`, the drift by its three methods and the QTFs) and
  `examples/fish_tank_cylinders/run_covered_tank.py` (a tank open, half covered and covered by a
  plate of its free surface, the closed tank against its fluid mass).

*Table 1. The names of 0.2.0 and their names in 0.3.0 (the inputs; the result paths are listed above).*

| 0.2.0 | 0.3.0 |
|---|---|
| `m.waves.omegas`, `.periods`, `.headings` | `m.environment.omegas`, `.periods` (with `frequency_input = "periods"`), `.headings` |
| `m.outputs.radiation`, `diffraction`, `rao`, `field_pressure`, `field_velocity`, `body_pressure`, `body_velocity` | `m.first_order.radiation`, `diffraction`, `motion`, `field_pressure`, `field_velocity`, `body_pressure`, `body_velocity` |
| `outputs.pressure_drift`, `control_surface_drift` | `m.second_order.drift_pressure_integration`, `drift_control_surface` |
| `outputs.momentum_drift` (`"none"`, `"unidirectional"`, `"bidirectional"`) | `second_order.drift_momentum_conservation` (bool) with `drift_bidirectional` (bool) |
| `outputs.momentum_drift_angles` | `second_order.drift_momentum_conservation_angles` |
| `m.water.lids` | `m.lids` |
| `m.water.dipole_dampers` (`DipoleDamper`) | `m.fixed_dampers` (`FixedDamper`) |
| `m.water.field_points.points` | `m.first_order.field_points` (the points in the sea) |
| `water.field_points.tanks` (`"<body>/<tank>"` per point) | the tank's item `field_points` (body axes) |
| `water.field_points.enabled` | removed (`field_pressure`, `field_velocity` switch the field outputs) |
| `m.water.panels()`, C `hb_water_panels` | `m.panels(role="lid")`, `m.panels(role="fixed_damper")`, C `hb_model_panels` |
| `mesh=("hull", [(1, 96)])` (a mesh reference) | `mesh="hull", ranges=[(1, 96)]` |
| C `hb_set_mesh` | `hb_set_text` (`mesh`) and `hb_set_ints` (`ranges`, shape (n, 2)) |
| a mode piece's `mesh` | its `ranges` (of the body's mesh) |
| `symmetry` `"none"`, `"x"`, `"y"`, `"xy"` | `"none"`, `"yz"` (the plane $x = 0$), `"xz"` (the plane $y = 0$), `"xz_and_yz"` |
| a `trimmed_patches` mesh's `ranges` | `parameter_ranges` |
| `Tank.free_surface` | `Tank.surface_level` |
| `PressureSurface.modes` | `PressureSurface.pressure_modes` |
| `PressureMode.stiffness`, `.damping` | `PressureMode.additional_stiffness`, `.additional_damping` (`Coupling.stiffness`, `.damping` unchanged) |
| `GeneralizedMode.symmetry` | `GeneralizedMode.parity` |
| `body.interior_surface = "waterline"` | `body.interior_surface = "waterline"` (default `"detect"`) |
| `body.interior_surface = ("hull", (257, 320))` | `interior_surface="mesh", interior_surface_mesh="hull", interior_surface_ranges=[(257, 320)]` |
| `body.interior_surface = None` | `body.interior_surface = "none"` |
| `body.control_surface = {"method": "cylinder", "radius": 40.0, ...}` | `control_surface="cylinder", control_surface_radius=40.0, ...` |
| `body.control_surface = {"mesh": "cs"}` | `control_surface="mesh", control_surface_mesh="cs"` |
| `body.control_surface = None` | `body.control_surface = "none"` (the default) |
| `control_surface.<item>` (`depth`, `inner_radius`, `semi_axes`, `panel_size`, ...) | `control_surface_<item>` |
| `control_surface.alternative` | `control_surface_waterline_term` |
| the automatic control surface's `radius` or `boundary` | `control_surface_outer` `"circle"` (`control_surface_radius`) or `"polygon"` (`control_surface_boundary`) |
| the automatic control surface's `free_surface` | removed (the free-surface part always built) |
| body `Ixx` ... `Izz` or `radii_of_gyration` | the same (defaults 0 and zeros), chosen by `inertia_input` (`"components"`, `"radii_of_gyration"` the default) |
| `hull.tanks = [{...}]`, `add(..., dipoles=[{...}])` (lists of dicts) | `hull.tanks.clear()`, `hull.tanks.add(...)`; `tank.dipoles.add(...)` |
| `excitation_method = "pressure"` | `excitation_method = "diffraction"` |
| `body.trimmed_mesh(name, "interior")`, `panels(role="interior")` | role `"interior_surface"` |
| `Panels.role` `"water_damper"` | `"fixed_damper"` |
| `m.classify_points(points, tanks)` (names per point) | `m.classify_points(points, tank=None)` (C `hb_model_classify_points(..., tank, ...)`, 0 the sea) |
| `BodyCounts.symmetry` | `BodyCounts.symmetry_copies` |
| `hb.engine.WaterObject` | removed |
| Rust `Value::Mesh`, `Value::mesh`, `ItemKind::MeshRef`, `Model::water_panels` | `Value::Text` and `Value::ranges`, `ItemKind::Reference`, `Model::panels` |

The result quantities of 0.2.0 (each also an attribute of `Results` then) and their names and
Python paths in 0.3.0 (the others unchanged: `omegas`, `headings`, `radiation_mode_ids`,
`coordinates`, `added_mass`, `radiation_damping`, `motion`, `mode_restoring`,
`pressure_mode_damping`, `hydrostatics_stiffness_rho_g`, `hydrostatics_mass`,
`hydrostatics_stiffness`, the centres below at new values for R not 0; Table 2).

*Table 2. The result quantities of 0.2.0 and their names and Python paths in 0.3.0.*

| 0.2.0 | 0.3.0 quantity | 0.3.0 Python path |
|---|---|---|
| `excitation`, `froude_krylov`, `scattering`, `haskind` | `excitation_diffraction`, `excitation_froude_krylov`, `excitation_scattering`, `excitation_haskind` | `results.excitation.diffraction`, `.froude_krylov`, `.scattering`, `.haskind` |
| `added_mass_zero`, `added_mass_infinite` | `added_mass_zero_frequency`, `added_mass_infinite_frequency` | the same names |
| `body_points`, `body_point_ids`, `body_normals`, `body_areas` | `body_points`, `body_points_ids`, `body_points_normals`, `body_points_areas` | `results.body_points.points`, `.ids`, `.normals`, `.areas` |
| `body_pressure_radiation`, `body_pressure_diffraction`, `body_velocity_radiation`, `body_velocity_diffraction` | `body_points_pressure_radiation`, ... | `results.body_points.pressure_radiation`, ... |
| the field points (`water.field_points.points`, the sea's and the tanks') | `field_points` (the sea's), `tanks` (a tank's, body axes), `tanks_ids` | `results.field_points.points`, `results.bodies[b].tanks[t].points` |
| `field_pressure_*`, `field_velocity_*` (every field point) | `field_points_pressure_*`, `field_points_velocity_*` (the sea's points); `tanks_pressure_radiation`, `tanks_velocity_radiation` (the tanks' points) | `results.field_points.pressure_radiation`, ...; `results.bodies[b].tanks[t].pressure_radiation`, `.velocity_radiation` |
| `lid_points`, `lid_ids`, `lid_elevation_radiation`, `lid_elevation_diffraction` | `lids`, `lids_ids`, `lids_elevation_radiation`, `lids_elevation_diffraction` | `results.lids[l].points`, `.elevation_radiation`, `.elevation_diffraction` |
| `pressure_surface_points`, `pressure_surface_ids`, `pressure_surface_elevation_*` | `pressure_surfaces`, `pressure_surfaces_ids`, `pressure_surfaces_elevation_*` | `results.bodies[b].pressure_surfaces[s].points`, `.elevation_*` |
| `pressure_drift`, `pressure_drift_moving` | `drift_pressure_integration`, `drift_pressure_integration_body_axes` | `results.drift.pressure_integration`, `.pressure_integration_body_axes` |
| `control_surface_drift`, `control_surface_drift_moving` | `drift_control_surface`, `drift_control_surface_body_axes` | `results.drift.control_surface`, `.control_surface_body_axes` |
| `momentum_drift`, `momentum_drift_pairs` | `drift_momentum_conservation`, `drift_heading_pairs` | `results.drift.momentum_conservation`, `.heading_pairs` |
| `kochin_*` | `kochin_*` | `results.kochin.angles`, `.radiation`, ... |
| `hydrostatics_volume` [B, 3] | `hydrostatics_volumes` [B, 3], and `hydrostatics_volume` [B] (their mean) | `results.hydrostatics.volumes`, `.volume` |
| `hydrostatics_waterplane` [B, 3] | `hydrostatics_waterplane_area` [B], `hydrostatics_waterplane_centroid` [B, 2] | `results.hydrostatics.waterplane_area`, `.waterplane_centroid` |
| `hydrostatics_center_of_buoyancy`, `hydrostatics_center_of_gravity` (relative to R) | the same names (body coordinates) | `results.hydrostatics.center_of_buoyancy`, `.center_of_gravity` |
| (none) | `hydrostatics_displaced_mass` [B], `hydrostatics_metacentric_height` [B, 2] | `results.hydrostatics.displaced_mass`, `.metacentric_height` |
| `body.hydrostatics().volume` [3] | `volumes` [3], and `volume` (their mean) | `body.hydrostatics().volumes`, `.volume` |
| `results.water.lids[name]` | | `results.lids[name]` |

## 0.2.0 (2026-11-04)

This version changes the Python API: `hb.Model` replaces the problem classes of 0.1.0. Its
versions: model file record 7, results file format 3, geometry file 2, Rust API version 42 and C ABI version 3 (`hb.build_info()`, `hb_abi_version`); files of other
versions are refused.

- **Mesh kinds named by what they hold.** `kind="panels"` (low order), `"flat_patches"`,
  `"spline_patches"`, `"analytic_patches"` and `"trimmed_patches"` replace `"surface"`, `"flat"`,
  `"spline"`, `"analytic"` and `"trimmed"`; the map of an analytic patch is its item `map` (was
  `kind`). Geometry files (`.hbg`) are version 2.
- **Fixes.** Generated interior surfaces (`projected`, `waterline`, `graded`)
  of a body placed with `position` $z \neq 0$ find its waterline (they took body $z = 0$); a thin sheet
  in a symmetry plane of its mesh is refused in the higher order as in the low order (it
  coincided with its own image); a flat-patch geometry file with `panel_size` reads (it failed);
  frequencies whose $\omega^2/g$ underflows or overflows are refused; the low-order body velocity on a
  pressure surface in $z = 0$ is computed (it was NaN: the panel coincides with its free-surface
  image); field-point tanks of another length than the points, a rename while a replacement holds a reference, and
  a save while a replacement is open are refused with their messages; a results file is re-read
  by its absolute path; the low-order mesh quality takes panels that do not meet at their
  vertices (a T-junction) as closed, as for patches.
- **The mesh quality of patch meshes.** The sides of the patches are checked as the edges of
  low-order panels: free, non-manifold and inconsistently oriented sides and open waterline ends,
  in `mesh_quality()`, `mesh.quality()` and the warnings of the rules; patches may meet a side of
  several smaller ones.
- **One argument order and unambiguous element types.** The
  tessellations take `(nu, nv, expand)` and the mode displacements `(points, bodies, records)` in
  Rust and C as in Python; `ElementType::ComplexF32`, `ComplexF64` and the `QuantityData` variants
  of the same names replace `Complex64`, `Complex128`, which read as the Rust complex types of the
  other precision.
- **The progress of a solve at any instant.** `m.progress()` (Rust `Model::progress_monitor`, C
  `hb_model_progress`) gives the preparation, each limit (zero, infinite) and each ordinary
  frequency with its state (waiting, running, done, failed, cancelled), stage and fraction done,
  from any thread while the solve runs, without slowing it.
- **The C ABI.** `hydrobem.dll` with its import library and `hydrobem.h` (`crates/hydrobem-c`,
  `docs/reference/c-api.md`): the model, its objects and items, the meshes, the queries, the solve
  with progress and cancellation, and the results, for any program that calls C, built from the
  sources (`cargo build --release -p hydrobem-c`). The Rust API gives an
  `ObjectId` as one number (`to_bits`, `from_bits`).
- **One model in the engine.** The engine holds the model: its inputs as objects with named
  items (the schema gives the unit, default and description of each), its meshes among them,
  and the results of its last solve; any change of an input discards them. `hb.Model` is the
  Python API: items as attributes (`m.bodies["hull"].mass = 1.2e6`), lists of objects
  (`m.bodies.add(...)`, children set from dicts, replaced all or nothing), the meshes read, made,
  and written by the engine (`m.meshes.load`, `add`, `analytic`, `load_csf`, `mesh.write`), the
  queries as
  methods, `m.solve(progress)`, `m.state`, `m.results`, `reset`, `clear`, `close`. The Python
  class of every object type is generated from the schema, so that editors and `help()` show the
  items with their descriptions.
- **Removed.** `hb.Problem`, `hb.Body`, `hb.Options`, `hb.Environment` and the other problem
  classes, the mesh classes and functions (`SurfaceMesh`, `PatchMesh`, `load_mesh`,
  `load_patch_mesh`, `trim_mesh`, ...), `BEMSolver`, `load_problem`, the module functions of the
  queries, the TOML case files, the HDF5 results file and the `hydrobem[hdf5]` extra. The C ABI
  of 0.1.0 (its header and the engine DLL) and its CMake build with oneMKL are gone: the engine
  is built by cargo alone (below).
- **Files.** The model file `.hbm` (one binary file, the meshes inside: `m.save`,
  `hb.Model.load`) and the results file `.hbr` (the model, the meta data and every quantity:
  `m.save_results`, `Results.save`, `hb.load_results`, `hb.Model.load_results`). A mesh read
  from a file keeps the path of its file (`source_file`, `source_spl`). Every writer makes the
  folder of its file and refuses an existing file unless `overwrite=True` (`FileExistsError`).
- **Results.** The engine writes the WAMIT-format numeric files and the `.out` summary and
  normalizes the quantities; every quantity is an attribute of `Results` named as in the engine
  (`results.pressure_drift`, `results.added_mass_zero`, `results.hydrostatics_stiffness`, ...),
  with `results.normalized(name)`, `results.warnings` and `hb.Results.of(m, **arrays)` for
  results computed elsewhere. The drift quantities are named as their outputs; the pressure and
  control-surface drifts are [F, H, 6B] with the moment about the moving origin [F, H, 3B]; the
  result `mode_damping` is `pressure_mode_damping`. Without a solved radiation column the files
  `.1`, `.2*`, the radiation blocks of `.5*`, `.6*` and the radiation tables of `.out` are left out.
- **Results by owner and by name.** The quantities are what a solve computes (`hb.quantities()`
  holds no answers of the queries; they return their own objects). The field, lid and
  pressure-surface quantities are named as their outputs: `field_pressure_radiation`,
  `field_pressure_incident`, `field_pressure_scattering`, `field_pressure_diffraction`;
  `lid_points`, `lid_elevation_radiation`, `lid_elevation_diffraction`;
  `pressure_surface_points`, `pressure_surface_elevation_radiation`,
  `pressure_surface_elevation_diffraction`. Views of the same arrays by owner:
  `results.bodies["spar"]` (its coordinates, the diagonal blocks of the coefficients, its forces,
  motions, drifts, hydrostatics, body points and `pressure_surfaces[i]`),
  `results.water.lids["gap"]`, `results.field_points`; by name,
  `results.coefficient("added_mass", ("spar", "heave"), ("barge", "pitch"))` and
  `results.coordinate(body, mode)`.
- **References by name between bodies.** `additional_stiffness` and `additional_damping` are $6 \times 6$
  (the body's own block); the stiffness and damping against another body's motions are
  `body.couplings` (`Coupling`: `body` by name, `stiffness`, `damping`, each $6 \times 6$). The tanks of the
  field points are names, `water.field_points.tanks = ["", "hull/aft"]` (a named tank), not
  numbers. Both follow a rename, block the removal of what they name and keep their meaning when
  the bodies are reordered.
- **Suppressed objects.** Bodies, tanks, dipoles, dipole dampers, pressure surfaces, lids,
  couplings and the field points have `enabled` (default true): a disabled object stays in the
  model and its file and is solved as if absent (its children, the couplings to a disabled body
  and the field points in a disabled tank with it). The results and the numeric files number the
  enabled bodies; the queries of a body answer for a disabled one; `m.check()` warns of a disabled
  object that would fail enabled.
- **`m.waves`.** The frequencies and headings of the solve are the object `m.waves` (type `Waves`;
  `omegas` or `periods`, `headings`), formerly `m.frequencies`.
- **Higher order: surfaces of meshes of their own.** Damping lids in global coordinates, dipoles,
  pressure surfaces and interior surfaces given as patch meshes of their own solve in the higher
  order as in the low order (formerly records of the body's mesh only).
- **The geometry file .hbg.** One mesh of kind `panels`, `flat_patches`, `spline_patches` (also
  rational) or `analytic_patches` (not `trimmed_patches`) with its discretization and patch names, as JSON: `m.meshes.load(name, "x.hbg")`,
  `m.meshes.from_hbg(name, data)` for the object a program passes, `mesh.write("x.hbg")` and
  `mesh.to_hbg()`. The patches of a mesh may have names.
- **Rational B-spline patches (NURBS).** A `SplinePatch` takes the item `weights` (positive, the
  shape of its control coefficients without the last axis): the rational patch, exact for circles,
  cylinders and spheres. Given from arrays or a model file; a GDF file is not written for it.
- **Named analytic shapes.** `m.meshes.analytic(name, shape, discretization, **dimensions)`
  takes the shape by name (`"circular_cylinder"`, `"sphere"`, `"barge"`, ...), its dimensions as
  keywords, `parts` (the patches in their order) and `mapping`, formerly the numeric shape code
  with `npatch` and a list of values. The analytic GDF files are read as before, into the
  named shapes.
- **Lids and thin sheets in a channel.** Damping lids and thin sheets (also dipole dampers) solve
  in a channel, in both methods, formerly rejected.
- **Thin sheets, pressure surfaces and lids together.** Thin sheets combine with damping lids and
  pressure surfaces, and pressure surfaces with lids, in both methods. The low order solves the
  scattered potential with lids (formerly the total one only); `"auto"` takes it with thin sheets.
- **Channel, higher order.** The higher-order method solves bodies in a channel (also with interior
  patches, pressure surfaces, the field and body outputs and the drifts by pressure integration and
  control surfaces).
- **Interior surfaces in a channel.** A body in a channel can have an interior free surface
  (irregular-frequency removal), formerly rejected.
- **Query answers as objects.** The queries return read-only objects with their fields as
  attributes (`hb.Hydrostatics`, `hb.MeshGeometry`, `hb.Counts`, ...), also `obj.info(item)`,
  `hb.build_info` and `hb.quantities`.
- **Queries by owner.** Each query is a method of the object that holds its data. A mesh
  answers for its geometry in mesh coordinates: `mesh.geometry()` (the volume closed by $z = 0$, its three
  estimates and their mean, the centre of volume, the area, the bounds, the waterplane with its
  second moments), `mesh.records()` (per stored record its centroid, normal, area and vertices),
  `mesh.tessellation()`, `mesh.quality()` (the geometric findings) and `mesh.patch_points`. A body
  answers for what needs its placement, trim, mass or the environment: `body.hydrostatics()` (with
  the displaced mass and, with the weight terms, the metacentric heights), `body.panels(role)`,
  `body.mesh_quality()`, `body.tessellation()`; the water `m.water.panels()`; the model
  `m.panels(role)`, `m.tessellation()`, `m.mesh_quality()` of everything, `counts`,
  `memory_estimate`, `classify_points`, `motion_matrices`, `mode_displacements`. `hb.ranges(mask)`
  turns a mask over the records (or record numbers) into the ranges of a mesh reference. The
  records of the answers (`Panels.record`, `Tessellation.record`, `MeshQuality.record`) and of
  `mode_displacements` are the stored records of the meshes, the numbering of the mesh
  references, also after a trim; the queries of a body and of the water check only the rules of
  the environment, the geometry and the surfaces.
  Removed: `m.hydrostatics(body)`, `m.interior_mesh(body)` (`body.panels(role="interior")`), the
  body arguments of `m.panels`, `m.tessellation` and `m.mesh_quality`, `mesh.hydrostatics`,
  `mesh.waterplane_panels` (a mask over `mesh.records()`), `mesh.trimmed` and `mesh.select` (the
  trim is the body's), the text selections (`hb.normalize_panels`, `hb.format_panels`,
  `hb.PanelRange`). The item `records` of a trimmed mesh is `trim_records`; `interior_panels` is
  removed.
- **Optional names.** The dipoles, dipole dampers (of bodies, tanks and the water), tanks and
  pressure surfaces may have names, unique within their list (`spar.dipoles.add("plate",
  mesh=...)`): an object with a name is found by it, its path holds it and the messages name it.
- **The model by ownership.** The model is `waves`, `environment` (depth, gravity,
  density and the domain: `domain`, `channel_width`, `channel_reflect`), `options` (the method:
  `higher_order`, `diffraction_formulation`, the logarithmic integration, the threads),
  `outputs` (what is computed: `radiation`, `diffraction`, `rao`, the drifts, the field and body
  outputs), `water` (what is at rest: `lids`, `dipole_dampers`, `field_points`), `bodies` and
  `meshes`. What moves with a body is a child of it: `dipoles`, `dipole_dampers`, `tanks`,
  `pressure_surfaces`, `generalized_modes`, and the single children `interior_surface` and
  `control_surface`, each with a `method` the engine generates (interior `"waterline"`,
  `"graded"`, `"projected"`; control surface `"cylinder"`, `"ellipsoid"`, `"automatic"` with
  its parameters as items, among them `alternative`) or a `mesh`; `body_point_neighbors` is an
  item of the body. The sheets inside a tank are children of the tank (`tank.dipoles`,
  `tank.dipole_dampers`), in the tank fluid. A body needs a wetted surface (`mesh`) or a dipole:
  a body of sheets only has no `mesh`.
- **Mesh references.** Every surface has one item `mesh`: a mesh name (all its records), or the
  name with one-based ranges of its stored records, `("hull", (1, 96))` or
  `("hull", [(1, 96), (129, 140)])`; a record has one role within a body. They replace the
  panel selections (`interior_panels`, `dipole_panels`, the `panels` of tanks, dampers,
  pressure surfaces and lids) and the separate meshes (`interior_mesh`, `dipole_mesh`).
- **Trim on the body.** `body.trim = [elevation, pitch, roll]` makes the referenced mesh the
  hull: the engine places it and cuts it and every surface of the body at the free surface when
  the problem is built, the ranges staying those of the hull mesh. `body.trimmed_mesh(name,
  role)` adds the trimmed surface of a role to the meshes (also a generated interior surface);
  `body.panels(role=...)` gives the records as the engine uses them, the generated interior and
  control surfaces among them (the role `"control_surface"`), and the dampers of the water as
  elements of the water.
- **Memory estimate.** `m.memory_estimate()` gives the resident peak (`peak`) and the commit
  charge (`commit`), which adds the reserved, mostly untouched packing buffers of the threads.
- **Dampers by owner.** A damper of a body is porous and moves with it, in its forces; a damper
  of the water is at rest, in no force. The condition item and the condition of a damper driven
  by the body but left out of its forces are removed (it has no physical counterpart: the added
  mass and damping are not symmetric and need not absorb energy).
- **Defaults.** `diffraction_formulation = "auto"`: the total potential with a lid or a
  free-surface plate (the scattered one with low-order thin sheets), the scattered one otherwise. A lid placed with a body (`placed_with`) takes records of that
  body's mesh.
- **Names for choices; no assumed mass.** `symmetry` is `"none"`, `"x"`, `"y"` or `"xy"`;
  `momentum_drift_angles` a count or `"auto"`; `Tank.free_surface` a z or `"walls_top"`. `mass`
  is a number or `"displaced"`; a body without it has no mass (the weight, the motions and the
  radii of gyration need one). Renamed: `options.high_order` $\to$ `higher_order`, the mesh item
  `source_records` $\to$ `source_panels` (the mesh item `interior_records` is removed). Removed:
  `m.meshes.ccyl_cs`, `ellipsoid_cs`, `automatic_control_surface` (now the methods of
  `control_surface`).
- **Messages.** Messages name the items of the model; errors carry a location (`exc.location`);
  an argument error is a `ValueError`; an object that another one refers to cannot be removed.
- **Corrections of results.** The zero-frequency added mass of a tank mode without a solution is
  NaN, with a warning; the computed damping of a tank is kept in its body's radiation damping.
  The second derivatives of the wave kernel near the vertical axis, the logarithmic terms and the
  static integrals of the higher order are more accurate, which changes higher-order results of
  thin sheets, lids and pressure surfaces in their last digits; closed sides of planar sheet
  patches are integrated by their length.
- **Speed.** The momentum drift, the logarithmic corrections of the higher order, the
  projections, the $\omega = \infty$ rows and the field loops run faster.
- **The engine in Rust.** The engine is a Rust crate (`crates/hydrobem`) with a public Rust API
  (`docs/reference/rust-api.md`) and the Python extension module `hydrobem._core`
  (`crates/hydrobem-python`, PyO3) in one cargo workspace. Its linear algebra is the faer crate:
  building needs only Rust and the MSVC linker, and the wheel contains only the Rust crates
  listed in `licenses/rust-crates.txt`. Single-precision results change at the rounding of the
  float32 LU (the regression baselines regenerated); `hb.build_info().linear_algebra` names the
  library and `.compiler` the `rustc` version.

## 0.1.0 (2026-10-27)

First public version.

- Low-order and higher-order (flat, B-spline and analytic patches) panel methods; infinite and
  finite depth with the limits $\omega = 0$ and $\omega = \infty$; open sea, walls and a channel (with a reflection
  factor); several placed and yawed bodies; symmetry sectors.
- Interior free surfaces (irregular frequencies), damping lids, thin sheets, dipole dampers,
  prescribed-pressure surfaces, internal tanks (with field points inside), generalized modes,
  wavemakers in the walls; combinations of interior surfaces with thin sheets and pressure
  surfaces.
- Radiation, diffraction (scattered and total), Haskind and pressure forces, motions,
  hydrostatics; field and body pressure and velocity, free-surface elevation.
- Mean drift by momentum conservation, by pressure integration (also with tanks) and by the
  momentum flux through a control surface (both line-term alternatives); control surfaces of
  panels, patches, analytic shapes or traced automatically from the waterline.
- Trimmed waterline of low-order and higher-order meshes, with the generated interior free
  surface and tank walls cut at their own free surfaces.
- GDF, SPL and control-surface readers, GDF and SPL writers; WAMIT-format numeric files and an
  `.out` summary.
- Model inspection for viewers (panels, tessellation, patch points), mesh quality measures and
  warnings, located errors, progress reports and cancellation, a memory estimate, engine thread
  counts as options (results identical for any counts).
- Windows x86-64 only: `pip install .` builds the engine; a built wheel on each release; the
  import refuses a processor without AVX2 (`HYDROBEM_SKIP_CPU_CHECK=1` overrides).
