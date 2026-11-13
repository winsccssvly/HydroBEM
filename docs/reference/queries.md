# Queries

Each query is a method of the object that holds its data: a mesh answers for its geometry in
mesh coordinates (reference/inputs-meshes §4), a body for anything that needs placement, trim, mass or the
environment, the model for everything together (the free surface of the potential QTF among
it). Each query checks the rules it needs (reference/inputs-rules §1), and only those:

- A query of a body checks the rules of that body alone: the values of the water (depth, gravity
  and density), the body's geometry and its surfaces. It does not check the options, the first
  and second order, the frequencies, the field points or the other bodies.
- A query of the free surface of the potential QTF (`m.panels(role="free_surface")` with that
  role alone, `m.free_surface_points`) checks the items of the free surface in `second_order`, the
  geometry and surfaces of every body (their waterlines, and the plates at which the free surface
  ends), and the environment with its walls. A zone whose auto values the frequencies decide
  (reference/inputs-orders §3) also needs a frequency between 0 and $\infty$, and the query
  refuses without one, naming the items to give.
- Every other query of the model checks every rule, as `m.check()` does.

Every query answers without a solve (Table 1).

*Table 1. The queries by owner and what they return.*

| Owner | Query | Returns |
|---|---|---|
| mesh | `geometry()` | `hb.MeshGeometry`: volume, centre of volume, area, bounds, waterplane (reference/inputs-meshes §4) |
| mesh | `elements()`, `patch_points(element, uv)` | `hb.MeshElements` per stored element; `hb.PatchPoints`, points and normals of a patch (reference/inputs-meshes §4) |
| mesh | `quality()`, `tessellation(divisions, include_reflections)` | `hb.MeshQuality` of the mesh alone; `hb.Tessellation` for display (reference/inputs-meshes §4) |
| body | `hydrostatics()` | `hb.Hydrostatics` after trim and placement, body axes (the stiffness about R) |
| body | `panels(role, include_reflections)`, `tessellation(role, divisions, include_reflections)` | `hb.Panels` of the elements as the engine uses them; `hb.Tessellation` of the same elements |
| body | `mesh_quality()` | `hb.MeshQuality` of the body as the engine uses it, with `hb.MeshEdges` |
| body | `trimmed_mesh(name, role)` | a new mesh of one role of a trimmed body (reference/inputs-bodies §1.2) |
| model | `m.panels(role, include_reflections)`, `m.tessellation(...)`, `m.mesh_quality()` | the same of every body, the lids and the fixed dampers (`m.panels(role="lid")`, `m.panels(role="fixed_damper")`); `m.panels(role="free_surface")` the panelled zone of the potential QTF's free surface (the cells have none) |
| model | `m.free_surface_points(zone)` | `hb.FreeSurfacePoints`: the points and weights of a zone of the potential QTF's free surface, `"panelled"`, `"quadrature"` or `"outer_circle"` (reference/inputs-orders §3) |
| model | `m.counts()` | `hb.Counts` (with `hb.BodyCounts` per body): coordinates, radiation columns, elements, the field points in the sea (`field_points`) and in the tanks (`tank_points`), ... |
| model | `m.memory_estimate()` | `hb.MemoryEstimate`: the memory of a solve in bytes |
| model | `m.motion_equation()`, `m.mode_displacements(points, bodies, elements)` | `hb.MotionEquation`, the matrices of the motion equation; the displacement of every coordinate at points |
| model | `m.classify_points(points, tank=None)` | `hb.PointClasses`: the verdict of the field-point rules on points in the sea, or inside the tank `tank` (reference/inputs-model-objects §3) |
| model | `m.route()`, `m.check()`, `m.warnings` | the name of the solver route; the warnings (reference/inputs-rules §1); the warnings of the last check, each an `hb.Diagnostic` with its location |
| package | `hb.wavenumber(omega, depth, gravity)`, `hb.build_info()`, `hb.quantities()` | the wave number; `hb.BuildInfo` (version, commit, compiler, linear algebra); `hb.QuantityInfo` per quantity |

The queries of the model describe the model as it is solved: its enabled
objects (reference/inputs-model §5), with the bodies numbered over the enabled ones (`Panels.body`, the bodies of
`mode_displacements`, `Counts.per_body`). The queries of a body (`hydrostatics`, `panels`,
`mesh_quality`, `tessellation`, `trimmed_mesh`) also answer for a disabled body: as if it were
enabled and every other object as it is, its own disabled children left out. Its body index is
then its place among the enabled bodies with itself counted.

**Numbering, units and frames.** Every index of the queries is 0-based (bodies, tanks, owners,
parts, lids, field points, frequencies, coordinates, radiation columns, a place in a list) except
two: the mesh element, 1-based as in `ranges`, `hb.ranges` and the files (reference/inputs-meshes
§3), and the coordinate ids of `Counts.radiation_mode_ids`, 1-based as the mode numbers of the
files (Table 12); -1 means "none" in every integer column and is never a valid number. Lengths
are in m, areas in $\mathrm{m}^2$, volumes in $\mathrm{m}^3$, normals are unit vectors; positions are global for the
queries of the model and of a body (body coordinates for `Hydrostatics`), mesh coordinates for a
mesh alone; integer keys and counts have no unit. The tables below give each class with its
columns (N rows, T quads, B bodies, n points).

## 1. Roles

A role is the kind of surface an element is in the problem (Table 2): the singular of the input
that makes it. The argument `role` of `panels` and `tessellation` (model and body) takes these
words and nothing else, and the column `role` of `Panels` and `Tessellation` holds them (C: the
code, `hb_panel_roles`; Rust: `PanelRole`).

*Table 2. The roles.*

| code | role | the elements | made by | of |
| --- | --- | --- | --- | --- |
| 0 | `"wetted"` | the wetted surface | the body's `mesh`, `ranges` | body, model |
| 1 | `"interior_surface"` | the interior surface (given, detected or generated) | the body's `interior_surface` | body, model |
| 2 | `"dipole"` | the dipoles, with those inside the tanks | `body.dipoles`, `tank.dipoles` | body, model |
| 3 | `"dipole_damper"` | the dipole dampers, with those inside the tanks | `body.dipole_dampers`, `tank.dipole_dampers` | body, model |
| 4 | `"pressure_surface"` | the prescribed-pressure surfaces | `body.pressure_surfaces` | body, model |
| 5 | `"lid"` | the lids: placed with a body (under that body) or meshes of their own (body $-1$) | `m.lids` | body (placed), model |
| 6 | `"tank"` | the walls of the tanks | `body.tanks` | body, model |
| 7 | `"fixed_damper"` | the fixed dampers (body $-1$) | `m.fixed_dampers` | model |
| 8 | `"control_surface"` | the control surface (given or generated) | the body's `control_surface` | body, model |
| 9 | `"free_surface"` | the panelled zone of the potential QTF's free surface (generated or given; reference/inputs-orders §3) | `second_order` | model |
| 10 | `"free_surface_plate"` | the free-surface plates, with those of the tanks' free surfaces | `body.free_surface_plates`, `tank.free_surface_plates` | body, model |

The argument `role` is None (every role but `control_surface` and `free_surface`, the generated
auxiliary surfaces, which are no boundary of the problem), one role (`role="wetted"`) or a list of
roles (`role=["wetted", "lid"]`); Rust takes a slice of names (empty: the default roles), C one
name or names separated by commas (`"wetted,lid"`; NULL: the default roles). A role a scope never
has is refused: the query of a body refuses `"fixed_damper"` and `"free_surface"`, which belong to
no body (`ValueError`, its message naming the model's query), and an unknown word is a
`ValueError` naming the roles.

## 2. Panels and tessellation

- `body.panels(role=None, include_reflections=True)` (`hb.Panels`) returns the elements of the
  body as the engine uses them (after its trim): global vertices, the key columns (§3), centroid,
  unit normal into the fluid, area; `include_reflections` also the reflected copies of the
  elements. A body gives the elements of its mesh (with the lids placed with it), then its
  surfaces of their own, in this order: the dipoles of another mesh, the interior surface of
  another mesh, the pressure surfaces of another mesh, the free-surface plates of another mesh
  (the body's, then those of its tanks), the control surface (given or generated) and last the
  generated interior surface (low order). `body.panels(role="interior_surface")` is the interior
  surface as the engine uses it (given or generated), placed globally.
- `m.panels(role=None, include_reflections=True)`: the panels of every body in order, then the
  elements of no body (body $-1$): the lids that are meshes of their own, the fixed dampers on the
  elements of a body's mesh, the fixed dampers that are meshes of their own and the free surface.
  A lid placed with a body is an element of that body (`body.panels`, role `"lid"`).
- `body.tessellation(role=None, divisions=8, include_reflections=True)` and
  `m.tessellation(...)` give a quadrilateral mesh for display of exactly the elements `panels`
  returns for the same `role` and `include_reflections`, placed, with the normals at the corners;
  `mesh.tessellation(divisions=8, include_reflections=True)` that of every element of a mesh alone
  (reference/inputs-meshes §4). A low-order panel is one quad (its projected planar panel, the
  corners as `panels` gives them, the panel normal at each corner; a triangle repeats a corner); a
  patch is $d_u \times d_v$ quads on its exact surface, the points and normals of `patch_points`
  at their corners. The normals point into the fluid (tank walls into the tank's fluid), as in
  `panels`; the quads of a reflection with $s_x s_y = -1$ are reversed to keep the orientation.
  `divisions` is an integer (both directions), a pair `(du, dv)`, each 1..256, or `"spans"`: one
  quad per span of the potential discretization of each patch (reference/inputs-meshes §2; the
  spans are uniform in the patch parameters). Rust takes `[du, dv]`, C `divisions_u`,
  `divisions_v`; `[0, 0]` (both 0) is `"spans"`.

*Table 3. The quads of the tessellation per surface.*

| surface | role | the quads |
|---|---|---|
| a body's wetted surface, low order | `wetted` | one per panel |
| a body's wetted surface, flat, spline or analytic patches | `wetted` | $d_u \times d_v$ per patch, exact |
| a trimmed patch body | `wetted` (and the others on its mesh) | $d_u \times d_v$ per part, on the part's parameter range |
| an interior surface given (`mesh`) or detected | `interior_surface` | by the kind of its mesh |
| an interior surface generated, low order (`waterline`, `graded`, `projected`) | `interior_surface` | one per generated panel |
| the interior surface the trim of a patch mesh generates | `interior_surface` | $d_u \times d_v$ per generated patch |
| dipoles, dipole dampers (of a body or a tank), tank walls, pressure surfaces, plates, lids (placed or of their own), fixed dampers (own mesh or a body's) | Table 2 | by the kind of their mesh, wherever it is (the body's mesh, a mesh of their own, a mesh joined to the body's by the higher order) |
| a control surface given (`mesh`) | `control_surface` | by the kind of its mesh |
| a control surface generated (`cylinder`, `ellipsoid`, `automatic`) | `control_surface` | $d_u \times d_v$ per generated patch, exact, also in a low-order model (they are patch meshes) |
| the potential QTF's free surface, given mesh or generated panels | `free_surface` | one per panel of the panelled zone; the cells have no panels (their points are `free_surface_points`) |
| a mesh alone (`mesh.tessellation`) | `""` | by its kind, mesh coordinates |

## 3. The key columns

`Panels` and `Tessellation` carry the seven key columns of Table 4, with the same values for a
quad as for the panels row of the element it lies on. Within one call the seven values identify
one panels row, and every quad of `tessellation` has the values of exactly one row of `panels` of
the same scope, `role` and `include_reflections`: a quad is joined to its panel by the seven keys.

*Table 4. The key columns.*

| column | type | numbering | definition |
|---|---|---|---|
| `body` | int | 0-based, -1 none | the body whose solve geometry the element is part of, its index among the enabled bodies (a disabled body queried by itself: its index as if it were enabled); -1 an element of no body, fixed in space: a lid not placed with a body, every fixed damper (also one on the elements of a body's mesh), the free surface of the potential QTF; every row of a mesh alone |
| `role` | text (C: int code) | Table 2 | the role of the element; of a mesh alone `""` (C -1) |
| `tank` | int | 0-based in its body's enabled tanks, -1 none | the tank whose fluid the element bounds or lies in: the tank's walls, the dipoles, dipole dampers and free-surface plates of a tank |
| `owner` | int | 0-based in its list, -1 none | the input object that makes the element, by its place in the list that holds it (enabled objects only): a dipole in `body.dipoles`, or in `tank.dipoles` when `tank` $\ge 0$; a dipole damper in `body.dipole_dampers` or `tank.dipole_dampers`; a pressure surface in `body.pressure_surfaces`; a plate in `body.free_surface_plates`, or in `tank.free_surface_plates` when `tank` $\ge 0$; a lid in `m.lids` (also one placed with a body); a fixed damper in `m.fixed_dampers`; the walls of a tank: the tank (= `tank`). -1 for the roles of which a body has one surface (`wetted`, `interior_surface`, `control_surface`) and for `free_surface` |
| `element` | int | 1-based | the number of the element in the numbering of the `ranges` of the mesh its input names (the stored elements, before the symmetry; reference/inputs-meshes §3); of an element a trim cut, the element it was cut from; of a surface the engine generates (the low-order interior surface, a generated control surface, the free-surface panels, the interior surface the trim of a patch mesh adds), its own numbering from 1 in the order of generation |
| `part` | int | 0-based | the part of the element after a trim cut it (a low-order panel cut into a quadrilateral and a triangle; a patch cut into parts by the waterline), in the order of the trim; 0 for an element not cut |
| `reflection` | int [2] | $s_x$, $s_y$ each 1 or -1 | the copy of the stored element about the symmetry planes of its mesh, (1, 1) the stored one; of a trimmed body the copy the cut element came from. With `include_reflections=False` only the rows of (1, 1) and, of a trimmed body without symmetry, the copies it holds as its own |

*Table 5. `hb.Panels`: N elements or reflected copies.*

| column | shape | unit, frame |
|---|---|---|
| `vertices` | [N, 4, 3] | m, global (a triangle repeats a vertex; a patch its corners) |
| `distinct_vertices` | [N] | the distinct vertices among the four, 3 or 4 |
| `body`, `role`, `tank`, `owner`, `element`, `part` | [N] | the keys (Table 4) |
| `reflection` | [N, 2] | the key (Table 4) |
| `centroids` | [N, 3] | m, global (a patch: its point at the centre of its parameters) |
| `normals` | [N, 3] | unit, global, into the fluid (tank walls into the tank's fluid) |
| `areas` | [N] | $\mathrm{m}^2$ |

*Table 6. `hb.Tessellation`: T quads.*

| column | shape | unit, frame |
|---|---|---|
| `vertices` | [T, 4, 3] | m, global (of a mesh alone: mesh coordinates) |
| `normals` | [T, 4, 3] | unit, into the fluid, at the corners |
| `body`, `role`, `tank`, `owner`, `element`, `part` | [T] | the keys of the element the quad lies on (Table 4); a mesh alone: role `""`, body, tank and owner -1, part 0 |
| `reflection` | [T, 2] | the key (Table 4) |

## 4. Mesh quality

`body.mesh_quality()` (`hb.MeshQuality`) measures the mesh of the body as the engine uses it: per
element (and part, after a trim) the panel or span size $d$ (the panel diameter, or the diagonal of
a B-spline span), the aspect $d^2/2A$, the panels or spans per shortest wavelength and flags
(inconsistent orientation, free edge below the waterline, non-manifold edge, fewer than 6
low-order panels per wavelength); the volumes of the divergence theorem and the edge counts (patch
sides) of its wetted surface. `m.mesh_quality()` gives every body, `mesh.quality()` a mesh alone
(reference/inputs-meshes §4). The same findings are warnings (`HydroBEMWarning`, codes 2–5) when
the model is checked or solved.

*Table 7. `hb.MeshQuality`: N elements, B bodies (a mesh alone: one row of the per-body columns).*

| column | shape | unit, frame |
|---|---|---|
| `body` | [N] | 0-based; -1 for a mesh alone |
| `element`, `part` | [N] | the keys as in `Panels` (Table 4), without the copy: of a trimmed body whose trim breaks a symmetry, the copies the body holds as its own are rows of their own with the same `element` and `part`, in the order of `panels(include_reflections=False)` |
| `size` | [N] | m: low order the panel diameter $d$, higher order the diagonal of a span |
| `aspect` | [N] | $d^2/2A$, 1 for a square |
| `per_wavelength` | [N] | panels or spans per shortest wavelength; inf without frequencies (and of a mesh alone) |
| `flags` | N tuples | `"orientation"`, `"free_edge"`, `"non_manifold"`, `"coarse"` |
| `volumes` | [B, 3] | $\mathrm{m}^3$, the divergence theorem in x, y, z |
| `min_per_wavelength` | [B] | the fewest panels or spans per wavelength |
| `edges` | `hb.MeshEdges` | per body [B] the edges `inconsistent` (two panels or patches that disagree in orientation), `free` (of one panel or patch, off the waterline), `non_manifold` (of more than two), `open_waterline` (free edges on the waterline of an open surface) |

## 5. Hydrostatics, the motion equation and the points

- `body.hydrostatics()` (`hb.Hydrostatics`, Table 8): the hydrostatics of the body after its trim
  and placement (theory §3.2), body axes: the `volumes` (the three estimates of the divergence
  theorem, as `mesh.geometry().volumes`) and the `volume` (their mean), the `displaced_mass` $\rho V$ ($V$
  the x estimate `volumes[0]`, the volume of a mass `"displaced"`), the `center_of_buoyancy`, the
  `waterplane_area` and `waterplane_centroid`, the stiffness about R without (`stiffness_rho_g`,
  the geometry alone) and with the weight terms (`stiffness`), `has_weight`, the mass in use
  (`mass`: $\rho$ times `volumes[0]` less the fluid of its tanks when it is `"displaced"`) and the
  `center_of_gravity` (as the input); with the weight terms (a centre of gravity and a mass) the
  `metacentric_height` (roll, pitch), $GM = C_{44} / (\rho g V)$ and $C_{55} / (\rho g V)$, else
  (and without a volume, a body of sheets or plates only) None. The results of a solve hold the
  same items for every body (`results.hydrostatics`, reference/results).

  *Table 8. `hb.Hydrostatics`.*

  | column | shape | unit, frame |
  |---|---|---|
  | `volumes`, `volume` | [3], scalar | $\mathrm{m}^3$ |
  | `displaced_mass`, `mass` | scalar | kg (`mass` None without one) |
  | `center_of_buoyancy`, `center_of_gravity` | [3] | m, body coordinates (`center_of_gravity` None without one) |
  | `waterplane_area` | scalar | $\mathrm{m}^2$ |
  | `waterplane_centroid` | [2] | x, y, m, body coordinates |
  | `stiffness_rho_g`, `stiffness` | [6, 6] | N/m, N, N m by the coordinates, about R, body axes |
  | `has_weight` | bool | |
  | `metacentric_height` | [2] | m, roll and pitch (None without the weight) |

- `m.motion_equation()` (`hb.MotionEquation`): the inertia, stiffness (hydrostatic, weight and
  additional) and additional damping of the motion equation as the engine uses them,
  $6N \times 6N$ about each $R$ in body axes (theory §3.1); `ready` false (and the three matrices
  None) while a body lacks a centre of gravity, a mass or inertia.

  *Table 9. `hb.MotionEquation`.*

  | column | shape | unit, frame |
  |---|---|---|
  | `inertia` | [6N, 6N] | kg, kg m, kg $\mathrm{m}^2$ by the coordinates, about each R, body axes |
  | `stiffness` | [6N, 6N] | N/m, N, N m by the coordinates |
  | `damping` | [6N, 6N] | N s/m, N s, N m s by the coordinates |
  | `ready` | bool | the matrices are formed |

- `m.mode_displacements(points, bodies, elements)`: the displacement of every coordinate at
  points of the bodies, for animations: `points` [n, 3] global (e.g. the centroids of `panels`),
  `bodies` [n] 0-based and `elements` [n] the 1-based element of the body's mesh of each (as
  `Panels.element`; it selects the pieces of the generalized modes; an element the body does not
  use is refused). It returns [n, D, 3] in global axes: the unit rigid modes (rotations about R,
  in body axes) and the generalized modes; zero for the pressure coordinates and other bodies.
- `m.classify_points(points, tank=None)` (`hb.PointClasses`, Table 10): the status of each point
  and the body, lid or tank of the status. `tank` None classifies points in the sea (global
  coordinates, as `first_order.field_points`), a `Tank` object (`m.bodies[b].tanks[t]`) points
  inside it (body axes, as its `field_points`; a tank the solve does not take: `"no_tank"`). The
  statuses: `"fluid"`, `"near_body"` (valid, closer to a low-order panel's centroid than its
  diameter: a warning), `"tank"`, `"no_tank"`, `"empty_tank"`, `"above_tank_surface"`,
  `"above_free_surface"`, `"below_sea_bed"`, `"outside_domain"`, `"on_body"`,
  `"inside_waterplane"`, `"on_lid"`, `"inside_body"`, `"not_finite"`.

  *Table 10. `hb.PointClasses`: n points.*

  | column | shape | definition |
  |---|---|---|
  | `status` | n texts | the status of the point |
  | `body` | [n] | 0-based: the body a point lies on, near or inside (its waterplane), or the body of the tank it lies in; else -1 |
  | `lid` | [n] | 0-based in `m.lids`: the lid a point lies on; else -1 |
  | `tank` | [n] | 0-based: the tank of its body a point lies in (as `results.bodies[b].tanks[t]`); else -1 |

- `m.free_surface_points(zone="panelled")` (`hb.FreeSurfacePoints`): the points and weights of
  one zone of the potential QTF's free surface as the solve integrates it
  (`second_order.free_surface_*`); the frequencies decide the auto sizes and the order.

  *Table 11. `hb.FreeSurfacePoints`: n points.*

  | column | shape | unit, frame |
  |---|---|---|
  | `points` | [n, 3] | m, global |
  | `weights` | [n] | $\mathrm{m}^2$ for `"panelled"` (the panels' or cells' Gauss points) and `"quadrature"`; m for `"outer_circle"` (the line integral around the outer circle) |

- `mesh.patch_points(element, uv)` (`hb.PatchPoints`): the points and unit normals (into the
  fluid) of the patch `element` (1-based) at the parameters `uv` [n, 2] in $[-1, 1]^2$, mesh
  coordinates: `points` [n, 3] (m) and `normals` [n, 3] (unit).

## 6. Counts, memory and the wave number

- `m.counts()` (`hb.Counts`) gives the sizes after the model rules; the counts of elements count
  them with their reflections (patches in a higher-order model).

  *Table 12. `hb.Counts` and `hb.BodyCounts` (no units).*

  | column | definition |
  |---|---|
  | `bodies`, `meshes`, `lids` | the bodies, the meshes the bodies and lids use, the lids |
  | `ordinary_frequencies`, `headings` | F, the frequencies other than 0 and inf; H |
  | `coordinates`, `radiation_columns` | D: six per body, the generalized and the pressure coordinates; K |
  | `field_points`, `tank_points` | J, the field points in the sea; the field points in the tanks |
  | `lid_elements`, `pressure_elements` | the elements of all lids, of all prescribed-pressure surfaces |
  | `has_zero_frequency`, `has_infinite_frequency` | $\omega = 0$, $\omega = \infty$ requested |
  | `radiation_mode_ids` | the coordinate (1-based) of each radiation column |
  | `per_lid_elements` | the elements of each lid |
  | `per_body` | a `BodyCounts` per body: `stored` (the stored elements of its mesh, the numbering of `ranges`), `wetted` (the wetted elements of the mesh as the engine uses it, after a trim, pressure surfaces on it included), `wetted_with_reflections` (the same with the reflections), `dipole` (the dipoles and dipole dampers on its mesh), `interior_surface` (the interior-surface elements on its mesh; -1 when the surface is generated), `pressure_modes`, `symmetry_copies` (1, 2 or 4), `generalized_modes` |

- `m.memory_estimate()` (`hb.MemoryEstimate`) the memory of a solve in bytes without solving it:
  `statics` (shared), `per_frequency` (one frequency thread with its inner threads), `outputs`,
  `peak` = `statics` + `frequency_threads` $\times$ `per_frequency` + `outputs`, the resident peak
  (the working set), and `commit`, the commit charge: the peak and the rest of the packing buffers
  of the matrix products, reserved whole on every thread that runs one (theory §13.3). With
  a potential QTF (`qtf_potential_direct`, `qtf_potential_indirect`) the peak and the commit are the larger of the model's solve and of the
  potential QTF's two sub-problems, solved one after the other with the results held. A solve
  compares this peak with the available physical memory first (`options.memory_check`, reference/inputs-model §3).
- `hb.wavenumber(omega, depth=inf, gravity=9.80665)` is the root $k$ of
  $\omega^2 = g k \tanh(k h)$ (theory §1.1), 0 for $\omega = 0$.
- Every query returns a read-only object with its fields as attributes (the classes of the
  tables above); its arrays are not writable. `obj.info(item)` (`hb.ItemInfo`) describes an item
  of the schema: its kind, unit, default, choices, description and the element type `dtype` of an
  array, `"real"`, `"int"` or `"text"`.
