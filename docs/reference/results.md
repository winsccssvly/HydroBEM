# Results

`m.solve()` returns the results of a model and keeps them in it: `m.results` gives them while `m.state` is
`"calculated"`. Any change of an input of the model discards them (`m.state`
is then `"reset"`, and `m.results` raises `RuntimeError`); `m.reset()` does so
explicitly. A `Results` object holds its own copy of the values and of the
model solved, so it is independent of later changes of the model.

The result quantities of the engine (`hb.quantities()` lists them with their type, axes, unit and
description; they are the quantities a solve computes, the answers of the queries are not among
them) are read through the result tree (§1), the one attribute path, which follows the inputs: a
group where one thing has variants or points with values, an array where it is one value, and
what belongs to an owner under the owner. The axes and the quantities read by body are
attributes of `Results` named as the quantity (`results.omegas`, `results.added_mass`,
`results.motion`); the others are in its groups and owners (`results.excitation.diffraction`,
`results.body_points.pressure_radiation`, `results.field_points.pressure_radiation`,
`results.drift.pressure_integration`, `results.qtf.potential.direct.sum`, `results.kochin.angles`,
`results.hydrostatics.stiffness`, `results.lids["moonpool"].points`,
`results.bodies["fpso"].tanks["aft"].pressure_radiation`). The name of a quantity is its path in
the tree with "_" for "." (`excitation_diffraction`, `drift_pressure_integration_body_axes`,
`hydrostatics_volumes`), and the points of a group of points take the group's name
(`body_points`, `field_points`, `lids`, `pressure_surfaces`, `tanks`; `body_points_ids`); the
numbers of the owners of points, which have no path of their own, follow the same rule
(`lids_ids`, `pressure_surfaces_ids`, `tanks_ids`). `results.quantity(name)` gives the whole
array of any quantity by its name, for a name taken from a list and for the quantities without a
view of the results (`coordinates`, the owner numbers, the whole arrays of the lids, pressure
surfaces and tanks). Each is a read-only NumPy array, dimensional (SI, unit wave amplitude; the
angles in degrees but `kochin_angles`, in radians, the axis of the Kochin functions as computed),
in the engine's type, or None when it was not computed; `results.coordinates` is a tuple of
`hb.Coordinate` (the id, body, kind, local index, name and whether it is rotational, of each
coordinate). The quantities, the main ones (§1 has the whole tree):

- `added_mass`, `radiation_damping` [F, D, K]; the exciting forces of the group `excitation`:
  `diffraction` (by pressure integration of the diffraction potential) with its parts
  `froude_krylov`, `scattering` [F, H, D], and `haskind` [F, H, K] (by the Haskind relation, of the
  radiation columns); `motion` [F, H, D] (with `first_order.motion`);
- `added_mass_zero_frequency`, `added_mass_infinite_frequency` [D, K]: the added mass at the
  limits $\omega = 0$ and $\infty$ when they are requested;
- the group `body_points` (with `first_order.body_pressure`, `body_velocity`): `points` [N, 3]
  (the panel centres, or a body's `body_points`, global), `ids` [N, 4] (body, element or point, sx,
  sy: low order the element of the body's mesh, the elements of the pressure surfaces and
  free-surface plates on meshes of their own numbered on after it, those of the pressure
  surfaces first, each mesh in its body's order; higher order the point's number), `normals`,
  `areas`, `pressure_radiation`, `pressure_diffraction`, `velocity_radiation`,
  `velocity_diffraction` (the pressure $p/(\rho g)$ is the dynamic pressure at the mean position,
  without a hydrostatic term, theory §4.1);
- the group `field_points` at the points of `first_order.field_points` (the points in the sea,
  global; reference/inputs-model-objects §3), named as the outputs `field_pressure` and `field_velocity`: `points`
  [J, 3] (the input's), `pressure_radiation` [F, J, K], `pressure_incident`,
  `pressure_scattering`, `pressure_diffraction` [F, J, H] (the elevation, or $p/(\rho g)$ below the free
  surface), and `velocity_*` of the same components [..., 3] (global axes);
- the points in a tank (a tank's `field_points`, body axes) have their outputs in the tank,
  `results.bodies[b].tanks[t]`: `points` [M_t, 3] (the input's, body axes),
  `pressure_radiation` [F, M_t, K] and `velocity_radiation` [F, M_t, K, 3] (the radiation of the
  tank problem, on the columns of the tank's body; the fluid of a tank has no incident or
  scattered wave); the quantities `tanks`, `tanks_ids` (the one-based tank of each point,
  numbered over the bodies in order), `tanks_pressure_radiation`, `tanks_velocity_radiation`;
- the damping lids, `results.lids[l]`: `points` [L_l, 3], `elevation_radiation`,
  `elevation_diffraction` (the quantities `lids`, `lids_ids`, `lids_elevation_*`); the
  pressure surfaces of a body, `results.bodies[b].pressure_surfaces[s]`: `points` [Q_s, 3],
  `elevation_radiation`, `elevation_diffraction` (`pressure_surfaces`, `pressure_surfaces_ids`
  numbered over the bodies in order, `pressure_surfaces_elevation_*`);
- `mode_restoring`, `pressure_mode_damping` [D, D]: the computed restoring of the
  generalized and pressure coordinates (hydrostatic and pressure-surface) with
  the stiffness of the pressure modes, and the damping of the pressure modes
  (their rows and columns), None without such coordinates; the body's own
  `mode_stiffness` and `mode_damping` are inputs and are not repeated here;
- the group `hydrostatics` (every body, theory §3.2; the names and items of the query
  `body.hydrostatics()`, reference/queries): `volumes` [B, 3] (the three estimates of the divergence
  theorem) and `volume` [B] (their mean, as `mesh.geometry().volume`), `displaced_mass` [B] ($\rho$
  times `volumes[:, 0]`), `center_of_buoyancy` [B, 3] (body coordinates), `waterplane_area` [B],
  `waterplane_centroid` [B, 2] (body coordinates), `stiffness_rho_g` [B, 6, 6], `mass` [B],
  `center_of_gravity` [B, 3] (body coordinates, as the input), `stiffness` [B, 6, 6] (about R),
  `metacentric_height` [B, 2] (roll, pitch: C44, C55 over $\rho g$ `volumes[:, 0]`); NaN where a body
  has no mass or centre of gravity (`mass`, `center_of_gravity`, `metacentric_height`);
- the mean drifts switched on (`second_order.drift_pressure_integration`,
  `drift_control_surface`, `drift_momentum_conservation`), each of the heading pairs
  `drift.heading_pairs` [P, 2] (deg; with `drift_bidirectional = false` the pair $(\beta, \beta)$ of every
  heading, $P = H$, with `true` every pair $\beta_a \le \beta_b$ in heading order), complex (a pair of two headings has a
  complex coefficient, of $A_a A_b^*$ of the waves $(\omega, \beta_a)$ and $(\omega, \beta_b)$; equal headings a real one):
  `drift.pressure_integration` [F, P, D] on the coordinates (body axes, theory §11.2: the force
  and the moment about the body's rotation centre R on the six rigid coordinates of each body, as
  the first-order results (theory §1.1), the generalized force on its
  generalized coordinates, NaN on its pressure coordinates) and
  `drift.pressure_integration_body_axes` [F, P, 3 B] (the moment about the moving R in body
  axes, of equal headings only, NaN for the other pairs; B bodies concatenated);
  `drift.control_surface` [F, P, D] (theory §11.3; the rigid coordinates, NaN on the others) and
  `drift.control_surface_body_axes` [F, P, 3 B]; `drift.momentum_conservation` [F, P, 3] (surge
  and sway force and yaw moment of all bodies in global axes, the moment about the global z axis:
  the total of all bodies has no single R, theory §11.1);
- the QTFs switched on (`second_order.qtf_quadratic_pressure_integration`,
  `qtf_quadratic_control_surface`, `qtf_potential_direct`, `qtf_potential_indirect`;
  [theory §12](../theory/12-second-order-loads.md)), each of the heading pairs `qtf.heading_pairs`
  [P, 2] (as the drifts', by `qtf_bidirectional`), complex [F, F, P, D] on the coordinates as the
  drifts: [a, b, p, i] the coefficient
  of the waves $(\omega_a, \beta_{p,a})$ and $(\omega_b, \beta_{p,b})$, of $A_a A_b^* e^{i(\omega_a - \omega_b)t}$ at the difference
  frequency (`difference`), of $A_a A_b e^{i(\omega_a + \omega_b)t}$ at the sum (`sum`); body axes, the moment
  about the body's rotation centre R (every method, as the drifts). The other order of a pair of
  headings is [b, a] of the pair: the conjugate at the
  difference frequency, the same at the sum. By method: `qtf.quadratic.pressure_integration`
  ([a, a] of the difference is the pressure drift), `qtf.quadratic.control_surface` (theory
  §11.3), `qtf.potential.direct` and `qtf.potential.indirect` (the force of the second-order
  potential, zero at equal frequencies; for a floating body without the force of the second-order
  motions). The total is the sum of a quadratic part and a potential part. With `qtf_difference`
  or `qtf_sum` false the sums or the differences only; in the results file, not in the numeric files;
- with generalized modes (low order, [theory §12.15](../theory/12-second-order-loads.md#1215-generalized-modes)) the rows of
  their coordinates in `drift.pressure_integration`, `qtf.quadratic.pressure_integration`,
  `qtf.potential.direct` and `qtf.potential.indirect` are their generalized forces (the work
  of the pressure on the exact surface against the mode shape), the rigid rows as for rigid
  bodies; the control-surface quantities have none (NaN); the numeric files hold the rigid rows;
- Newman's approximation (`second_order.qtf_newman`, theory §12.13) of every drift switched
  on, real [F, F, H, D] on the coordinates of its drift (NaN where the drift is NaN;
  `qtf.newman.momentum_conservation.difference` [F, F, H, 3]) of the equal headings:
  `qtf.newman.pressure_integration.difference`, `qtf.newman.control_surface.difference`,
  `qtf.newman.momentum_conservation.difference`; the diagonal the drift; it stands for the whole
  difference-frequency QTF, not a part to add; in the results file;
- with the drift by momentum conservation the Kochin functions of the group `kochin`: `angles`
  [A], `radiation`, `radiation_derivative` [F, A, K], `diffraction`, `diffraction_derivative`
  [F, A, H].

Axes: F the ordinary frequencies (`omegas`), H the headings (`headings`), D
the coordinates (`coordinates`: `hb.Coordinate` objects with the body, kind,
local index, name and whether it is rotational), K the radiation columns
(`radiation_mode_ids`). `model` is a copy of the model solved (an `hb.Model`,
without results), `route` the solver route, `warnings` the warnings of the
model and the solve (`hb.Diagnostic` objects), `diagnostics` every diagnostic of every severity
(the warnings and the errors a results file holds), `available` the names of the quantities the
results have (in the order of `hb.quantities()`), `meta` the meta data of the
solve (§2). Methods: `normalized(name)` (a quantity normalized as in the
files, reference/file-formats §1), `summary()`, `save(path, overwrite=False)` (§2),
`write_wamit(stem, frequency="period", timestamp=None, overwrite=False)` (reference/file-formats §2).
`hb.Results.of(m, **arrays)` makes results of the present inputs of a model
from arrays computed elsewhere, by quantity name, to write them with
`write_wamit` or `save` (their route is `"given"`, their `meta` None). The axes
(`omegas`, `headings`, `radiation_mode_ids`, `coordinates`) and the points of the field outputs
(`field_points`, `tanks`, `tanks_ids`: the inputs' points) are those of the model and are not
given; every array must have the shape of its quantity on them, the extent of each other axis
(lid points, body points, angles) the same in every quantity, and a real quantity real values.
The quantities read together are given together, checked when `Results.of` makes the results (a
`ValueError` names the quantity missing; `write_wamit` checks the same): the eleven
`hydrostatics_*` all or none; `drift_pressure_integration` with
`drift_pressure_integration_body_axes`, and `drift_control_surface` with
`drift_control_surface_body_axes` (each needs the other); any drift (`drift_pressure_integration`,
`drift_control_surface`, `drift_momentum_conservation`) with `drift_heading_pairs`;
`body_points_ids` with a body in range, an element >= 1 and sx, sy in -1..1 in each row, and when a
row is of a panel (sx or sy not 0) with `body_points`, `body_points_normals` and
`body_points_areas`. The Rust API checks them with `Results::check_given` once every quantity is
put.

Missing quantities are None, never zero-filled. The engine computes no totals
after a solve; a quantity such as the
pressure of the moving body, the sum of the diffraction and of the radiation
components weighted by the motion (theory §4.1), is formed by the user from these arrays.

## 1. Results by owner and by name

The owners are those of the model as it was solved: its enabled objects (reference/inputs-model §5). A disabled
body is no owner (`results.bodies["x"]` and `results.coordinate("x", ...)` raise a `KeyError` that
says it was disabled); `results.model` is the model with every object, as it was at the solve.

The same arrays are read by their owner, as views (slices where the rows are contiguous,
read-only like every answer; None where the quantity was not computed). The engine's result tree
defines them: every quantity with a view has a path in it (`drift.pressure_integration`,
`lids.elevation_radiation`, ...) and a rule that selects the part of an owner:

- `shared`: the whole array, also for a body (the heading pairs; Newman's approximation of the
  drift by momentum conservation, of all bodies);
- `global`: the whole array of the results, none (None) for a body;
- `columns` (n): a body's $n$ columns of the last axis (`3*body`), $b n$ to $(b + 1) n$;
- `coordinates`: a body's rows of the coordinate axes D (its coordinates in `coordinates`) and of
  the radiation-column axes K (the columns whose `radiation_mode_ids` is one of them): D_b, K_b;
- `body_points`: a body's rows of the body-point axis (those of `body_points_ids` whose body is
  it): N_b;
- `lid`: the rows of the lid-point axis of one lid (`lids_ids`): L_l;
- `pressure_surface`: the rows of the pressure-point axis of one pressure surface of a body
  (`pressure_surfaces_ids`, numbered over the bodies in order): Q_s;
- `tank`: the rows of the tank-point axis of one tank of a body (`tanks_ids`, numbered over the
  bodies in order): M_t.

The Rust API selects them with `Results::select(path, body, owner)` (`Results::selection` gives
the rows; `result_views()` and `result_groups()` list the tree), the C ABI with
`hb_results_select` (`hb_result_views`, `hb_result_groups`), and the Python classes of the results
(`Results`, `BodyResults` and the classes of the groups: `ExcitationResults`, `BodyPointResults`,
`FieldPointResults`, `LidResults`, `PressureSurfaceResults`, `TankResults`, ...) are generated
from the tree. A path of one segment is an attribute of the results (the axes, and the quantities
read by body, also of a body); the others are groups:

- `results.bodies[name or index]`, a `BodyResults` (`len`, iteration, `names()`): `name`,
  `index`, `coordinates` (its `hb.Coordinate`s), `hydrostatics` (an `hb.Hydrostatics`, of the
  `hydrostatics_*` quantities), its views below (the diagonal blocks of the coupled
  coefficients; at its body points every radiation column of the model), `excitation`,
  `body_points`, `drift`, `qtf`, `pressure_surfaces[name or index]` and `tanks[name or index]`.
- `results.lids[name or index]`: the views of one lid.
- `results.field_points`: the points in the sea of the model solved and the views of the
  `field_points_*` quantities; the points of a tank are its own (`bodies[b].tanks[t]`), and a
  tank disabled (reference/inputs-model §5) has none.
- `results.excitation`, `results.body_points`, `results.drift` and `results.qtf`
  (`quadratic.pressure_integration`, `quadratic.control_surface`, `potential.direct`,
  `potential.indirect`, `newman.<drift method>`), also of a body.
- `results.kochin` (the Kochin functions) and `results.hydrostatics` (the arrays of the
  hydrostatics of every body, [B, ...]; a body's `hydrostatics` is its `hb.Hydrostatics`).
- What belongs to no single body stays on the results: the axes, the coupled coefficients
  between bodies, the drift by momentum conservation, the field points in the sea and the Kochin
  functions.

The result tree (the views of the results and of a body `b`, with the shape, an owner's part on
an owner; the quantity of each at the right; a body's `coordinates` and `hydrostatics` besides):

<!-- result tree: generated by scripts/gen_result_views.py, do not edit -->
```
results
├─ omegas [F]                                        omegas
├─ headings [H]                                      headings
├─ radiation_mode_ids [K]                            radiation_mode_ids
├─ added_mass [F, D, K]                              added_mass
├─ radiation_damping [F, D, K]                       radiation_damping
├─ added_mass_zero_frequency [D, K]                  added_mass_zero_frequency
├─ added_mass_infinite_frequency [D, K]              added_mass_infinite_frequency
├─ excitation
│  ├─ diffraction [F, H, D]                          excitation_diffraction
│  ├─ froude_krylov [F, H, D]                        excitation_froude_krylov
│  ├─ scattering [F, H, D]                           excitation_scattering
│  └─ haskind [F, H, K]                              excitation_haskind
├─ motion [F, H, D]                                  motion
├─ mode_restoring [D, D]                             mode_restoring
├─ pressure_mode_damping [D, D]                      pressure_mode_damping
├─ body_points
│  ├─ points [N, 3]                                  body_points
│  ├─ ids [N, 4]                                     body_points_ids
│  ├─ normals [N, 3]                                 body_points_normals
│  ├─ areas [N]                                      body_points_areas
│  ├─ pressure_radiation [F, N, K]                   body_points_pressure_radiation
│  ├─ pressure_diffraction [F, N, H]                 body_points_pressure_diffraction
│  ├─ velocity_radiation [F, N, K, 3]                body_points_velocity_radiation
│  └─ velocity_diffraction [F, N, H, 3]              body_points_velocity_diffraction
├─ field_points
│  ├─ points [J, 3]                                  field_points
│  ├─ pressure_radiation [F, J, K]                   field_points_pressure_radiation
│  ├─ pressure_incident [F, J, H]                    field_points_pressure_incident
│  ├─ pressure_scattering [F, J, H]                  field_points_pressure_scattering
│  ├─ pressure_diffraction [F, J, H]                 field_points_pressure_diffraction
│  ├─ velocity_radiation [F, J, K, 3]                field_points_velocity_radiation
│  ├─ velocity_incident [F, J, H, 3]                 field_points_velocity_incident
│  ├─ velocity_scattering [F, J, H, 3]               field_points_velocity_scattering
│  └─ velocity_diffraction [F, J, H, 3]              field_points_velocity_diffraction
├─ lids[l]
│  ├─ points [L_l, 3]                                lids
│  ├─ elevation_radiation [F, L_l, K]                lids_elevation_radiation
│  └─ elevation_diffraction [F, L_l, H]              lids_elevation_diffraction
├─ bodies[b]
│  ├─ added_mass [F, D_b, K_b]                       added_mass
│  ├─ radiation_damping [F, D_b, K_b]                radiation_damping
│  ├─ added_mass_zero_frequency [D_b, K_b]           added_mass_zero_frequency
│  ├─ added_mass_infinite_frequency [D_b, K_b]       added_mass_infinite_frequency
│  ├─ excitation
│  │  ├─ diffraction [F, H, D_b]                     excitation_diffraction
│  │  ├─ froude_krylov [F, H, D_b]                   excitation_froude_krylov
│  │  ├─ scattering [F, H, D_b]                      excitation_scattering
│  │  └─ haskind [F, H, K_b]                         excitation_haskind
│  ├─ motion [F, H, D_b]                             motion
│  ├─ mode_restoring [D_b, D_b]                      mode_restoring
│  ├─ pressure_mode_damping [D_b, D_b]               pressure_mode_damping
│  ├─ body_points
│  │  ├─ points [N_b, 3]                             body_points
│  │  ├─ ids [N_b, 4]                                body_points_ids
│  │  ├─ normals [N_b, 3]                            body_points_normals
│  │  ├─ areas [N_b]                                 body_points_areas
│  │  ├─ pressure_radiation [F, N_b, K]              body_points_pressure_radiation
│  │  ├─ pressure_diffraction [F, N_b, H]            body_points_pressure_diffraction
│  │  ├─ velocity_radiation [F, N_b, K, 3]           body_points_velocity_radiation
│  │  └─ velocity_diffraction [F, N_b, H, 3]         body_points_velocity_diffraction
│  ├─ pressure_surfaces[s]
│  │  ├─ points [Q_s, 3]                             pressure_surfaces
│  │  ├─ elevation_radiation [F, Q_s, K]             pressure_surfaces_elevation_radiation
│  │  └─ elevation_diffraction [F, Q_s, H]           pressure_surfaces_elevation_diffraction
│  ├─ tanks[t]
│  │  ├─ points [M_t, 3]                             tanks
│  │  ├─ pressure_radiation [F, M_t, K]              tanks_pressure_radiation
│  │  └─ velocity_radiation [F, M_t, K, 3]           tanks_velocity_radiation
│  ├─ drift
│  │  ├─ pressure_integration [F, P, D_b]            drift_pressure_integration
│  │  ├─ pressure_integration_body_axes [F, P, 3]    drift_pressure_integration_body_axes
│  │  ├─ control_surface [F, P, D_b]                 drift_control_surface
│  │  ├─ control_surface_body_axes [F, P, 3]         drift_control_surface_body_axes
│  │  └─ heading_pairs [P, 2]                        drift_heading_pairs
│  └─ qtf
│     ├─ quadratic
│     │  ├─ pressure_integration
│     │  │  ├─ difference [F, F, P, D_b]             qtf_quadratic_pressure_integration_difference
│     │  │  └─ sum [F, F, P, D_b]                    qtf_quadratic_pressure_integration_sum
│     │  └─ control_surface
│     │     ├─ difference [F, F, P, D_b]             qtf_quadratic_control_surface_difference
│     │     └─ sum [F, F, P, D_b]                    qtf_quadratic_control_surface_sum
│     ├─ potential
│     │  ├─ direct
│     │  │  ├─ difference [F, F, P, D_b]             qtf_potential_direct_difference
│     │  │  └─ sum [F, F, P, D_b]                    qtf_potential_direct_sum
│     │  └─ indirect
│     │     ├─ difference [F, F, P, D_b]             qtf_potential_indirect_difference
│     │     └─ sum [F, F, P, D_b]                    qtf_potential_indirect_sum
│     ├─ newman
│     │  ├─ pressure_integration
│     │  │  └─ difference [F, F, H, D_b]             qtf_newman_pressure_integration_difference
│     │  ├─ control_surface
│     │  │  └─ difference [F, F, H, D_b]             qtf_newman_control_surface_difference
│     │  └─ momentum_conservation
│     │     └─ difference [F, F, H, 3]               qtf_newman_momentum_conservation_difference
│     └─ heading_pairs [P, 2]                        qtf_heading_pairs
├─ kochin
│  ├─ angles [A]                                     kochin_angles
│  ├─ radiation [F, A, K]                            kochin_radiation
│  ├─ radiation_derivative [F, A, K]                 kochin_radiation_derivative
│  ├─ diffraction [F, A, H]                          kochin_diffraction
│  └─ diffraction_derivative [F, A, H]               kochin_diffraction_derivative
├─ drift
│  ├─ pressure_integration [F, P, D]                 drift_pressure_integration
│  ├─ pressure_integration_body_axes [F, P, 3 B]     drift_pressure_integration_body_axes
│  ├─ control_surface [F, P, D]                      drift_control_surface
│  ├─ control_surface_body_axes [F, P, 3 B]          drift_control_surface_body_axes
│  ├─ momentum_conservation [F, P, 3]                drift_momentum_conservation
│  └─ heading_pairs [P, 2]                           drift_heading_pairs
├─ qtf
│  ├─ quadratic
│  │  ├─ pressure_integration
│  │  │  ├─ difference [F, F, P, D]                  qtf_quadratic_pressure_integration_difference
│  │  │  └─ sum [F, F, P, D]                         qtf_quadratic_pressure_integration_sum
│  │  └─ control_surface
│  │     ├─ difference [F, F, P, D]                  qtf_quadratic_control_surface_difference
│  │     └─ sum [F, F, P, D]                         qtf_quadratic_control_surface_sum
│  ├─ potential
│  │  ├─ direct
│  │  │  ├─ difference [F, F, P, D]                  qtf_potential_direct_difference
│  │  │  └─ sum [F, F, P, D]                         qtf_potential_direct_sum
│  │  └─ indirect
│  │     ├─ difference [F, F, P, D]                  qtf_potential_indirect_difference
│  │     └─ sum [F, F, P, D]                         qtf_potential_indirect_sum
│  ├─ newman
│  │  ├─ pressure_integration
│  │  │  └─ difference [F, F, H, D]                  qtf_newman_pressure_integration_difference
│  │  ├─ control_surface
│  │  │  └─ difference [F, F, H, D]                  qtf_newman_control_surface_difference
│  │  └─ momentum_conservation
│  │     └─ difference [F, F, H, 3]                  qtf_newman_momentum_conservation_difference
│  └─ heading_pairs [P, 2]                           qtf_heading_pairs
└─ hydrostatics
   ├─ volumes [B, 3]                                 hydrostatics_volumes
   ├─ volume [B]                                     hydrostatics_volume
   ├─ displaced_mass [B]                             hydrostatics_displaced_mass
   ├─ center_of_buoyancy [B, 3]                      hydrostatics_center_of_buoyancy
   ├─ waterplane_area [B]                            hydrostatics_waterplane_area
   ├─ waterplane_centroid [B, 2]                     hydrostatics_waterplane_centroid
   ├─ stiffness_rho_g [B, 6, 6]                      hydrostatics_stiffness_rho_g
   ├─ mass [B]                                       hydrostatics_mass
   ├─ center_of_gravity [B, 3]                       hydrostatics_center_of_gravity
   ├─ stiffness [B, 6, 6]                            hydrostatics_stiffness
   └─ metacentric_height [B, 2]                      hydrostatics_metacentric_height
```
<!-- end of the result tree -->

*Table 1. The views (Results: the attribute on the results, `[...]` the shape; An owner: on a
body `b`, a lid `l`, or a pressure surface `s` or a tank `t` of a body).*

<!-- result views: generated by scripts/gen_result_views.py, do not edit -->
| View | Quantity | Rule | Results | An owner |
|---|---|---|---|---|
| `omegas` | `omegas` | global | `omegas` [F] |  |
| `headings` | `headings` | global | `headings` [H] |  |
| `radiation_mode_ids` | `radiation_mode_ids` | global | `radiation_mode_ids` [K] |  |
| `added_mass` | `added_mass` | coordinates | `added_mass` [F, D, K] | `bodies[b].added_mass` [F, D_b, K_b] |
| `radiation_damping` | `radiation_damping` | coordinates | `radiation_damping` [F, D, K] | `bodies[b].radiation_damping` [F, D_b, K_b] |
| `added_mass_zero_frequency` | `added_mass_zero_frequency` | coordinates | `added_mass_zero_frequency` [D, K] | `bodies[b].added_mass_zero_frequency` [D_b, K_b] |
| `added_mass_infinite_frequency` | `added_mass_infinite_frequency` | coordinates | `added_mass_infinite_frequency` [D, K] | `bodies[b].added_mass_infinite_frequency` [D_b, K_b] |
| `excitation.diffraction` | `excitation_diffraction` | coordinates | `excitation.diffraction` [F, H, D] | `bodies[b].excitation.diffraction` [F, H, D_b] |
| `excitation.froude_krylov` | `excitation_froude_krylov` | coordinates | `excitation.froude_krylov` [F, H, D] | `bodies[b].excitation.froude_krylov` [F, H, D_b] |
| `excitation.scattering` | `excitation_scattering` | coordinates | `excitation.scattering` [F, H, D] | `bodies[b].excitation.scattering` [F, H, D_b] |
| `excitation.haskind` | `excitation_haskind` | coordinates | `excitation.haskind` [F, H, K] | `bodies[b].excitation.haskind` [F, H, K_b] |
| `motion` | `motion` | coordinates | `motion` [F, H, D] | `bodies[b].motion` [F, H, D_b] |
| `mode_restoring` | `mode_restoring` | coordinates | `mode_restoring` [D, D] | `bodies[b].mode_restoring` [D_b, D_b] |
| `pressure_mode_damping` | `pressure_mode_damping` | coordinates | `pressure_mode_damping` [D, D] | `bodies[b].pressure_mode_damping` [D_b, D_b] |
| `body_points.points` | `body_points` | body_points | `body_points.points` [N, 3] | `bodies[b].body_points.points` [N_b, 3] |
| `body_points.ids` | `body_points_ids` | body_points | `body_points.ids` [N, 4] | `bodies[b].body_points.ids` [N_b, 4] |
| `body_points.normals` | `body_points_normals` | body_points | `body_points.normals` [N, 3] | `bodies[b].body_points.normals` [N_b, 3] |
| `body_points.areas` | `body_points_areas` | body_points | `body_points.areas` [N] | `bodies[b].body_points.areas` [N_b] |
| `body_points.pressure_radiation` | `body_points_pressure_radiation` | body_points | `body_points.pressure_radiation` [F, N, K] | `bodies[b].body_points.pressure_radiation` [F, N_b, K] |
| `body_points.pressure_diffraction` | `body_points_pressure_diffraction` | body_points | `body_points.pressure_diffraction` [F, N, H] | `bodies[b].body_points.pressure_diffraction` [F, N_b, H] |
| `body_points.velocity_radiation` | `body_points_velocity_radiation` | body_points | `body_points.velocity_radiation` [F, N, K, 3] | `bodies[b].body_points.velocity_radiation` [F, N_b, K, 3] |
| `body_points.velocity_diffraction` | `body_points_velocity_diffraction` | body_points | `body_points.velocity_diffraction` [F, N, H, 3] | `bodies[b].body_points.velocity_diffraction` [F, N_b, H, 3] |
| `field_points.points` | `field_points` | global | `field_points.points` [J, 3] |  |
| `field_points.pressure_radiation` | `field_points_pressure_radiation` | global | `field_points.pressure_radiation` [F, J, K] |  |
| `field_points.pressure_incident` | `field_points_pressure_incident` | global | `field_points.pressure_incident` [F, J, H] |  |
| `field_points.pressure_scattering` | `field_points_pressure_scattering` | global | `field_points.pressure_scattering` [F, J, H] |  |
| `field_points.pressure_diffraction` | `field_points_pressure_diffraction` | global | `field_points.pressure_diffraction` [F, J, H] |  |
| `field_points.velocity_radiation` | `field_points_velocity_radiation` | global | `field_points.velocity_radiation` [F, J, K, 3] |  |
| `field_points.velocity_incident` | `field_points_velocity_incident` | global | `field_points.velocity_incident` [F, J, H, 3] |  |
| `field_points.velocity_scattering` | `field_points_velocity_scattering` | global | `field_points.velocity_scattering` [F, J, H, 3] |  |
| `field_points.velocity_diffraction` | `field_points_velocity_diffraction` | global | `field_points.velocity_diffraction` [F, J, H, 3] |  |
| `lids.points` | `lids` | lid |  | `lids[l].points` [L_l, 3] |
| `lids.elevation_radiation` | `lids_elevation_radiation` | lid |  | `lids[l].elevation_radiation` [F, L_l, K] |
| `lids.elevation_diffraction` | `lids_elevation_diffraction` | lid |  | `lids[l].elevation_diffraction` [F, L_l, H] |
| `pressure_surfaces.points` | `pressure_surfaces` | pressure_surface |  | `bodies[b].pressure_surfaces[s].points` [Q_s, 3] |
| `pressure_surfaces.elevation_radiation` | `pressure_surfaces_elevation_radiation` | pressure_surface |  | `bodies[b].pressure_surfaces[s].elevation_radiation` [F, Q_s, K] |
| `pressure_surfaces.elevation_diffraction` | `pressure_surfaces_elevation_diffraction` | pressure_surface |  | `bodies[b].pressure_surfaces[s].elevation_diffraction` [F, Q_s, H] |
| `tanks.points` | `tanks` | tank |  | `bodies[b].tanks[t].points` [M_t, 3] |
| `tanks.pressure_radiation` | `tanks_pressure_radiation` | tank |  | `bodies[b].tanks[t].pressure_radiation` [F, M_t, K] |
| `tanks.velocity_radiation` | `tanks_velocity_radiation` | tank |  | `bodies[b].tanks[t].velocity_radiation` [F, M_t, K, 3] |
| `kochin.angles` | `kochin_angles` | global | `kochin.angles` [A] |  |
| `kochin.radiation` | `kochin_radiation` | global | `kochin.radiation` [F, A, K] |  |
| `kochin.radiation_derivative` | `kochin_radiation_derivative` | global | `kochin.radiation_derivative` [F, A, K] |  |
| `kochin.diffraction` | `kochin_diffraction` | global | `kochin.diffraction` [F, A, H] |  |
| `kochin.diffraction_derivative` | `kochin_diffraction_derivative` | global | `kochin.diffraction_derivative` [F, A, H] |  |
| `drift.pressure_integration` | `drift_pressure_integration` | coordinates | `drift.pressure_integration` [F, P, D] | `bodies[b].drift.pressure_integration` [F, P, D_b] |
| `drift.pressure_integration_body_axes` | `drift_pressure_integration_body_axes` | columns (3) | `drift.pressure_integration_body_axes` [F, P, 3 B] | `bodies[b].drift.pressure_integration_body_axes` [F, P, 3] |
| `drift.control_surface` | `drift_control_surface` | coordinates | `drift.control_surface` [F, P, D] | `bodies[b].drift.control_surface` [F, P, D_b] |
| `drift.control_surface_body_axes` | `drift_control_surface_body_axes` | columns (3) | `drift.control_surface_body_axes` [F, P, 3 B] | `bodies[b].drift.control_surface_body_axes` [F, P, 3] |
| `drift.momentum_conservation` | `drift_momentum_conservation` | global | `drift.momentum_conservation` [F, P, 3] | `bodies[b].drift.momentum_conservation`: None |
| `drift.heading_pairs` | `drift_heading_pairs` | shared | `drift.heading_pairs` [P, 2] | `bodies[b].drift.heading_pairs` [P, 2] |
| `qtf.quadratic.pressure_integration.difference` | `qtf_quadratic_pressure_integration_difference` | coordinates | `qtf.quadratic.pressure_integration.difference` [F, F, P, D] | `bodies[b].qtf.quadratic.pressure_integration.difference` [F, F, P, D_b] |
| `qtf.quadratic.pressure_integration.sum` | `qtf_quadratic_pressure_integration_sum` | coordinates | `qtf.quadratic.pressure_integration.sum` [F, F, P, D] | `bodies[b].qtf.quadratic.pressure_integration.sum` [F, F, P, D_b] |
| `qtf.quadratic.control_surface.difference` | `qtf_quadratic_control_surface_difference` | coordinates | `qtf.quadratic.control_surface.difference` [F, F, P, D] | `bodies[b].qtf.quadratic.control_surface.difference` [F, F, P, D_b] |
| `qtf.quadratic.control_surface.sum` | `qtf_quadratic_control_surface_sum` | coordinates | `qtf.quadratic.control_surface.sum` [F, F, P, D] | `bodies[b].qtf.quadratic.control_surface.sum` [F, F, P, D_b] |
| `qtf.potential.direct.difference` | `qtf_potential_direct_difference` | coordinates | `qtf.potential.direct.difference` [F, F, P, D] | `bodies[b].qtf.potential.direct.difference` [F, F, P, D_b] |
| `qtf.potential.direct.sum` | `qtf_potential_direct_sum` | coordinates | `qtf.potential.direct.sum` [F, F, P, D] | `bodies[b].qtf.potential.direct.sum` [F, F, P, D_b] |
| `qtf.potential.indirect.difference` | `qtf_potential_indirect_difference` | coordinates | `qtf.potential.indirect.difference` [F, F, P, D] | `bodies[b].qtf.potential.indirect.difference` [F, F, P, D_b] |
| `qtf.potential.indirect.sum` | `qtf_potential_indirect_sum` | coordinates | `qtf.potential.indirect.sum` [F, F, P, D] | `bodies[b].qtf.potential.indirect.sum` [F, F, P, D_b] |
| `qtf.newman.pressure_integration.difference` | `qtf_newman_pressure_integration_difference` | coordinates | `qtf.newman.pressure_integration.difference` [F, F, H, D] | `bodies[b].qtf.newman.pressure_integration.difference` [F, F, H, D_b] |
| `qtf.newman.control_surface.difference` | `qtf_newman_control_surface_difference` | coordinates | `qtf.newman.control_surface.difference` [F, F, H, D] | `bodies[b].qtf.newman.control_surface.difference` [F, F, H, D_b] |
| `qtf.newman.momentum_conservation.difference` | `qtf_newman_momentum_conservation_difference` | shared | `qtf.newman.momentum_conservation.difference` [F, F, H, 3] | `bodies[b].qtf.newman.momentum_conservation.difference` [F, F, H, 3] |
| `qtf.heading_pairs` | `qtf_heading_pairs` | shared | `qtf.heading_pairs` [P, 2] | `bodies[b].qtf.heading_pairs` [P, 2] |
| `hydrostatics.volumes` | `hydrostatics_volumes` | global | `hydrostatics.volumes` [B, 3] |  |
| `hydrostatics.volume` | `hydrostatics_volume` | global | `hydrostatics.volume` [B] |  |
| `hydrostatics.displaced_mass` | `hydrostatics_displaced_mass` | global | `hydrostatics.displaced_mass` [B] |  |
| `hydrostatics.center_of_buoyancy` | `hydrostatics_center_of_buoyancy` | global | `hydrostatics.center_of_buoyancy` [B, 3] |  |
| `hydrostatics.waterplane_area` | `hydrostatics_waterplane_area` | global | `hydrostatics.waterplane_area` [B] |  |
| `hydrostatics.waterplane_centroid` | `hydrostatics_waterplane_centroid` | global | `hydrostatics.waterplane_centroid` [B, 2] |  |
| `hydrostatics.stiffness_rho_g` | `hydrostatics_stiffness_rho_g` | global | `hydrostatics.stiffness_rho_g` [B, 6, 6] |  |
| `hydrostatics.mass` | `hydrostatics_mass` | global | `hydrostatics.mass` [B] |  |
| `hydrostatics.center_of_gravity` | `hydrostatics_center_of_gravity` | global | `hydrostatics.center_of_gravity` [B, 3] |  |
| `hydrostatics.stiffness` | `hydrostatics_stiffness` | global | `hydrostatics.stiffness` [B, 6, 6] |  |
| `hydrostatics.metacentric_height` | `hydrostatics_metacentric_height` | global | `hydrostatics.metacentric_height` [B, 2] |  |
<!-- end of the result views -->

By name: `results.coordinate(body, mode)` gives the zero-based index of a coordinate on the D
axis and on the K axis (None when it is no radiation column), the body by name or index and the
mode by name (`"heave"`, a generalized or a pressure mode name) or by its one-based position
among the coordinates of the body. `results.coefficient(name, row, column=None)` gives one
series of a quantity, each coordinate a `(body, mode)` pair: of an [F, D, K] quantity the [F]
series of a row and a radiation column (`results.coefficient("added_mass", ("spar", "heave"),
("barge", "pitch"))`), of an [F, H, D] or [F, H, K] quantity the [F, H] series of a row
(`column` None), of a second-order quantity on the coordinates (`drift_pressure_integration`
[F, P, D], the QTFs [F, F, P, D], ...) the series of a row, of a [D, K] or [D, D] quantity the
element. An unknown body or mode is a
`KeyError` (an index beyond them an `IndexError`), an unknown or missing quantity, a quantity
without coordinate axes and a column that is no radiation column a `ValueError`.

## 2. The results file of the engine (.hbr)

`results.save("run.hbr")` or `m.save_results("run.hbr")` writes it, and
`hb.load_results("run.hbr")` reads it back with its model (`results.model`, an `hb.Model`)
and its meta data (`results.meta`, an `hb.Meta`).
`hb.Model.load_results("run.hbr")` gives the model itself, calculated: its inputs and, as
`m.results`, the results of the file. The engine writes and reads the file (Rust API
`Results::save`, `Model::save_results`, `Model::load_results`). The model file `.hbm`
(reference/inputs-model §4) is the same container without the quantities. A results file holds, each entry a binary
record:

- `model`: the model that was solved, every object and item as given (the meshes included; the
  model record, version 13, reference/inputs-model §4);
- `meta`: the HydroBEM version and build, the time of the solve, the route, the threads, the
  engine seconds and `environment_hooks` (the hooks of the environment set for the solve, for
  validation studies and tests only, as `NAME` or `NAME=value`; empty without);
- `diagnostics`: the warnings of the solve;
- every computed quantity under its name (`q/added_mass`), in the engine's type and layout (the
  quantities of the Rust API), with its axes and unit, and the hydrostatics of every body.

Results loaded from a file behave like the results of a solve. Each quantity is read when it is
first copied, and every entry is checked by its CRC-32: the file must stay as it was while its
results are in use (a file overwritten in between reads as damaged). A file that is cut short or damaged is
refused. The file format is version 5 (the quantities under the names of the result tree, reference/results); a file of
another version is refused (version 4, of the names before, too), and entries this version does not
know are ignored. The write is atomic: a reader sees the old file or the
complete new one.
