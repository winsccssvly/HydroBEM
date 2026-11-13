# Inputs: bodies and their surfaces

## 1. Body

A body is an object of the list `m.bodies`, named: `m.bodies.add(name, mesh=..., **items)`. A
body needs a wetted surface (`mesh`) or at least one dipole, a rule of `m.check()` and the solve:
a body of sheets only (a submerged plate, a thin annulus) has no `mesh`, and its dipoles name
their meshes; a body of dipole dampers only is refused (a porous sheet at rest is a fixed damper,
reference/inputs-model-objects §2). A body that a lid is placed with cannot be removed. What moves with the body, or has no meaning
without it, is a child of the body (§2); its surfaces are in body axes and follow its
`position`, `yaw` and `trim`. The order of the list is the order of the bodies in the results and
the files; `m.bodies.move(key, index)` moves a body, by name or position, to a place of the list
(negative from the end, as a Python index), and every list of the model has the same `move`.

### 1.1 Items

*Table 1. The items of a body.*

| Item | Meaning | Default |
|---|---|---|
| `mesh` | the wetted surface: the name of a mesh (reference/inputs-meshes §3), in body axes; a mesh of kind `panels`, or a patch mesh with `options.higher_order = true`; with `trim` the hull (§1.2); no element of it may lie wholly in $z = 0$ (a waterplane cover is the interior surface, found by `interior_surface = "detect"`, or a lid, or left out; §2.6, reference/inputs-rules §1, rule 2); unset: a body of sheets only (the mesh of its first dipole is the body's mesh) | unset |
| `ranges` | the elements of `mesh` that are the wetted surface (reference/inputs-meshes §3); unset: every element | unset |
| `name` | unique non-blank name | `"body"` |
| `position` | global position of the body origin (m) | (0, 0, 0) |
| `yaw` | rotation about global z (degrees), applied before the translation | 0 |
| `trim` | `[elevation, pitch, roll]`: the elevation (m), pitch and roll (degrees) of the hull (§1.2) | unset |
| `rotation_center` | motion reference R, in body axes (m); the first-order results and the second-order loads of the body (the mean drifts by pressure integration and by the control surface, the QTFs, Newman's approximation of those drifts) have their moments about R and are the loads with the second-order displacement of R zero (the second-order rotation about R); the momentum drift is global, about the z axis (theory §1.1) | (0, 0, 0) |
| `modes` | ordered subset of surge, sway, heave, roll, pitch, yaw (a list of names): radiation columns (with `first_order.radiation`) and free motions; `[]` = fixed body that still interacts | all six |
| `center_of_gravity` | G, in body axes (m); never inferred | unset |
| `mass` | kg ($\ge 0$; 0 a massless body, such as a light piston or lid: its inertia the added mass, theory §3.1), or `"displaced"`: $\rho V_x$ (the displaced mass from the x-divergence volume, less the fluid of the tanks); needed with `center_of_gravity` for the weight and for the motions; unset: the body has no mass | unset |
| `inertia_input` | which inertia is used: `"components"` (`Ixx` ... `Izz`) or `"radii_of_gyration"`; only that side is read (an either-or input, see the introduction) | `"radii_of_gyration"` |
| `Ixx, Ixy, Ixz, Iyy, Iyz, Izz` | with `inertia_input = "components"`: the inertia tensor components about `inertia_origin` (kg $\mathrm{m}^2$), body axes, entered without sign change | 0 |
| `radii_of_gyration` | with `inertia_input = "radii_of_gyration"`: radii of gyration $r_{ij}$ (m) about `inertia_origin`, three values (diagonal) or a symmetric $3 \times 3$ matrix; the engine forms $I_{ij} = m\, r_{ij} \lvert r_{ij} \rvert$ with the mass in use (`mass`, a number or $\rho V_x$) | zeros |
| `inertia_origin` | `"center_of_gravity"`, `"rotation_center"` or a point in body axes | `"center_of_gravity"` |
| `additional_stiffness`, `additional_damping` | $6 \times 6$, SI: the external matrices of the body's own rigid modes; with another body a coupling (§1.3) | unset (zero) |
| `stiffness_origin`, `damping_origin` | point of the additional matrices, as for `inertia_origin` | `"rotation_center"` |
| `excitation_method` | `"diffraction"` or `"haskind"`: the exciting force used in this body's motion equation, `excitation_diffraction` or `excitation_haskind` | `"diffraction"` |
| `mode_mass`, `mode_damping`, `mode_stiffness` | $(6 + G) \times (6 + G)$ matrices of the body's rigid (about R) and generalized coordinates, SI; the rigid block zero | unset |
| `body_points` | (N, 3) points in body axes where the body outputs are evaluated (low order the inverse-distance mean of the `body_point_neighbors` nearest centroids, a point on a centroid taking its value; higher order the closest point of the wetted and pressure patches; theory §4.2); for every body or none, with `first_order.body_pressure` or `body_velocity` | unset |
| `body_point_neighbors` | low order with `body_points`: the number of nearest panel centroids averaged by inverse distance; $\ge 1$ | 4 |

Rules:
- Names unique across bodies; with `first_order.motion = true` every moving body needs a
  centre of gravity and a mass; its inertia is the side `inertia_input` names (a zero inertia is
  computable; the inertia of the free coordinates must still be positive semi-definite, below); the weight terms of the
  hydrostatics need the centre of gravity and the mass; an origin `"center_of_gravity"` that is
  used requires `center_of_gravity` (`inertia_origin` when the motions are solved and the body has rigid modes,
  `stiffness_origin` and `damping_origin` with their matrices or couplings). Nothing is assumed: a body without a mass has none
  (its hydrostatic mass is NaN).
- The off-diagonal inertia terms are never assumed zero.
- The engine moves the inertia and the additional matrices to R (theory §3.1).
- Products of inertia, matrices and coordinates must be finite real numbers;
  matrices must fit float32.
- The inertia of the free coordinates (the rigid inertia on the free rigid modes and `mode_mass`
  on the generalized coordinates, with their couplings) must be positive semi-definite, and a
  massless direction of it must move water (theory §3.1): a mass of 0 (a light piston or lid) or a
  zero mode mass (a free-surface mode of a moonpool) is accepted, its inertia the added mass.
- A structure fixed in space whose forces are wanted is a body with `modes = []`.

### 1.2 Trim

Unset, the body's mesh is used as given and is the wetted surface (the rules of reference/inputs-meshes). Set,
also to `[0, 0, 0]`, the body's mesh is the hull in its mesh coordinates $\xi$, which may extend
above the water: when the problem is built the engine places it in body axes at
$x = R \xi + (0, 0, \mathrm{elevation})$, $R$ the pitch (about the $\eta$ axis) followed by the roll (about the $\xi$
axis), and cuts it at $z = 0$, and with it every surface of the body:

- low order, dry elements are removed and elements across $z = 0$ are cut there (a pentagon becomes
  a quadrilateral and a triangle); higher order, a patch the waterline cuts is remapped onto the
  parameter square (every line of constant u, or of constant v, from its submerged end to the
  waterline) and split where its lines change from wholly submerged to cut or dry, as where a
  side crosses the waterline twice; a patch whose lines cross it twice in both directions is
  rejected (theory §14);
- the symmetry planes a nonzero angle breaks are removed after reflecting the stored part
  (pitch: $x = 0$, roll: $y = 0$);
- dipoles and dipole dampers keep their submerged part; a tank is cut at its `surface_level`
  when that is a number (low order); a generated interior surface or automatic control surface
  is made from the new waterline.

With a trim the interior surface must be generated (`interior_surface` `"waterline"`,
`"graded"`, `"projected"`; `"mesh"` is refused, as given elements lie on the waterline of the
untrimmed hull: `set interior_surface to a generated one`; `"detect"` finds none), the dipoles take elements of the body's mesh, a pressure surface is not
available, a free-surface plate is generated from the new waterline (`inside`, §2.4), and a lid
cannot take elements of the body (give it as a mesh of its own). The
`ranges` keep counting the elements of the hull mesh. `body.trimmed_mesh(name, role="wetted")` adds the
surface of one role as the engine uses it to the meshes of the model (low order a `panels`
mesh, higher order a `trimmed_patches` mesh; `role` one of `"wetted"`, `"dipole"`,
`"dipole_damper"`, `"tank"`, `"pressure_surface"`, `"free_surface_plate"`, `"interior_surface"`; the dipoles and dampers with
those inside the tanks; `"pressure_surface"` is accepted, but a trimmed body never has one), its `source_elements` the stored elements of the hull they come from, for
writing or reuse. A generated interior surface of the low order (also of a body without trim) is
the surface the solve generates, a `panels` mesh in the stored symmetry form, its
`source_elements` 0.

```python
m.meshes.load("hull", "hull_full.gdf")         # the whole hull, above the water too
m.bodies.add("hull", mesh="hull", trim=[-0.5, 2.0, 0.0])   # (m, deg, deg)
m.bodies["hull"].interior_surface = "waterline"
```

### 1.3 Couplings between bodies

`couplings`: a list of `Coupling` objects (optional names, as the lists of §2), each the external
stiffness and damping between this body's forces and the motions of another body (Table 2).

*Table 2. The items of a coupling.*

| Item | Meaning | Default |
|---|---|---|
| `body` | the other body, by name (required) | unset |
| `stiffness` | $6 \times 6$, SI: the rows the six forces of this body about its `stiffness_origin`, the columns the six motions of the other body about that body's `stiffness_origin` | unset |
| `damping` | $6 \times 6$, SI: as `stiffness`, about the `damping_origin` of each body | unset |

```python
a.couplings.add(body="b", stiffness=K_ab)      # a's forces from b's motions
b.couplings.add(body="a", stiffness=K_ab.T)    # and b's from a's, when the coupling is symmetric
```

Rules: a coupling names another body, not the body itself; at most one coupling per other body;
at least one of `stiffness`, `damping`; an origin `"center_of_gravity"` of either body requires
that body's `center_of_gravity`. A coupling is named by its body, not by a position: it keeps its
meaning when the bodies are reordered, follows a rename of the other body, and the other body
cannot be removed while a coupling names it (as a lid's `placed_with`, reference/inputs-model-objects §1). Both blocks are
moved to the rotation centres of their bodies as in theory §3.1; a coupling adds only to this
body's rows (give the other direction on the other body).

## 2. The surfaces of a body

The children of a body, each with the items `mesh` and `ranges` (reference/inputs-meshes §3) in body axes. The dipoles,
dipole dampers, tanks with their sheets and mode pieces take elements of the body's mesh (the
dipoles also of one other mesh), with the roles of reference/inputs-meshes §3.

In the higher order a surface of a mesh of its own (the dipoles, a pressure surface, a
free-surface plate, the interior surface) and a damping lid in global coordinates (reference/inputs-model-objects §1) are
appended to the elements of the body's
patch mesh and solved as elements of it, the same results as the same elements of one mesh; the
panels, the tessellation and the mesh quality list them as in the low order (by the elements of
their own mesh, a lid in global coordinates with the lids and fixed dampers). The
other mesh is a flat, B-spline or analytic patch mesh with the symmetry of the body's mesh;
analytic patches join only analytic ones, and flat patches with B-spline ones become bilinear
B-spline patches of the same surface; a trimmed body takes no surfaces of other meshes.

The lists `dipoles`, `dipole_dampers`, `tanks`, `pressure_surfaces` and `free_surface_plates` of
a body, the sheets of a
tank and `m.fixed_dampers` take optional names: `spar.dipoles.add("plate", mesh=...)`, or
`spar.dipoles.add(mesh=...)` without one; set from dicts, `{"name": "plate", "mesh": ...}` or
without `"name"`. An object with a name is found by its name or its position
(`spar.dipoles["plate"]`, `spar.dipoles[0]`), one without by its position. The names are unique
within their list and not digits alone (the position in a path); `rename(old, new)` renames, and
`""` removes the name. The path of a named object, which the messages name, holds its name
(`bodies/spar/dipoles/plate`), that of an object without one its one-based position
(`bodies/spar/dipoles/2`); a message of the model rules names a part by its name (`dipoles
'plate'`) or position (`dipoles 2`). The names are kept in the model file.

### 2.1 Dipoles and dipole dampers

- `dipoles`: a list of `Dipole` objects with the items `mesh` and `ranges`: zero-thickness rigid sheets
  (heave plates, strakes, baffles) moving with the body, both faces at its normal velocity
  (theory §9.5); elements of the body's mesh, or of one other mesh appended to the body with the
  same symmetry.
- `dipole_dampers`: a list of `DipoleDamper` objects with the items `mesh` and `ranges` (elements
  of the body's mesh) and `damping_factor` $\varepsilon \ge 0$ (m/s): porous sheets moving with the body, the jump
  $-(i\varepsilon/\omega)(\phi_n - v_n)$, part of the body forces (theory §9.5; $\varepsilon = 0$: no jump, as no sheet; $\varepsilon \to \infty$: the
  rigid sheet, a dipole). Sheets at rest are fixed dampers (reference/inputs-model-objects §2).

### 2.2 Tanks

`tanks`: a list of `Tank` objects with the items `mesh` and `ranges` (elements of the body's
mesh: the tank walls, normals into the tank fluid), `density_ratio` (1.0: the density relative to
`environment.density`; 0 an empty tank), `surface_level` (the z of the tank's free surface in
body axes (m), or `"walls_top"`, the default: the highest vertex of the walls) and `field_points`
(the points of the field outputs inside the tank, reference/inputs-model-objects §3). Theory §10.2; both methods; the mass
`"displaced"` is then $\rho V_x$ less the tank fluid.

The sheets inside a tank (baffles) are children of the tank: `tank.dipoles` (rigid sheets) and
`tank.dipole_dampers` (porous, `damping_factor` $\varepsilon \ge 0$, the jump $-(i\varepsilon/\omega)(\phi_n - v_n)$), each with `mesh` and `ranges`
naming elements of the body's mesh, each element with no other role (reference/inputs-meshes §3). They lie in the
tank fluid and move with the body through the tank: they belong to the tank's problem, not to the
exterior. With a trim they are cut with the tank, at its `surface_level` when that is a number.

The plates of a tank's free surface (a deck over part of the tank, a full cover) are its list
`tank.free_surface_plates` of `FreeSurfacePlate` objects (§2.4: `mesh` and `ranges`), elements of
the body's mesh with no other role or a mesh of their own, body coordinates, every element at the
tank's surface level (normals turned to $-z$, into the tank fluid). They are massless and rigid,
move with every mode of the body (and with the generalized modes whose pieces name their elements),
and are free-surface plates of the tank's problem (theory §10.2): at every frequency and both
limits, both methods; their area leaves the tank's free-surface restoring, and a fully covered
tank is a closed container (its heave defined at $\omega = 0$). The model's check applies the
rules of the plates (inputs-rules §1) to them in the tank's free surface. Not available: a trimmed
body with tank plates, and the pressure-integration drift of a body whose tank has plates; a plate
of the sea or a body of its own on a tank's fluid (a floating roof) is refused.

```python
aft = m.bodies["fpso"].tanks.add(mesh="fpso", ranges=[(401, 448)])
aft.dipoles.add(mesh="fpso", ranges=[(449, 460)])        # a baffle, added to the tank once it exists
aft.free_surface_plates.add(mesh="fpso", ranges=[(461, 476)])   # a deck over part of the tank
```

### 2.3 Prescribed-pressure surfaces

- `pressure_surfaces`: a list of `PressureSurface` objects: a horizontal surface at $z \le 0$, `mesh` and
  `ranges` the elements of the body's mesh or another mesh in body axes; its elements may point either way, also
  mixed (each is turned to a downward normal, towards the water, §2.4).
- `pressure_modes`: a named list of one or more `PressureMode` objects with the items
  `shape`, `additional_stiffness` (N/m, 0) and `additional_damping` (N s/m, 0);
  shapes `uniform` (1), `sign_x` (the sign of x in body axes) or `sign_y` (the sign of y).
  Each mode adds one generalized coordinate (a pressure head $\xi$ in metres,
  $p = -\rho g \xi\, n(x, y)$) after the body's six rigid coordinates.
- `chamber_coupling` (bool, default `False`): include the restoring coupling
  of an incompressible constant-volume chamber with the rigid modes.
- `response_convention` (default `"variational"`): `variational`
  (theory §9.3) or `numeric_compatibility` (**[B]**, theory §9.3).
- Not available on a trimmed body.

### 2.4 Free-surface plates

`free_surface_plates`: a list of `FreeSurfacePlate` objects with the items `mesh` and `ranges`:
massless rigid plates lying in the free surface $z = 0$ (with the body placed), the water below
them only (theory §9.7): the light piston on the water column of an oscillating water column, a
floating mat of zero draft, a cover of a moonpool. Elements of the body's mesh or of another mesh
in body axes, every one in $z = 0$ (on a trimmed hull generated by the engine, below); they may point either way, also mixed: each is turned to a
downward normal, into the water. A plane surface has an unambiguous orientation, so the turn is
exact and changes no result: a low-order panel and a flat patch by their vertex order reversed, a
B-spline or analytic patch by its parameter u reversed (the same surface on the same spans). The
`waterplane` part of an analytic shape (pointing up, as an interior surface) is a plate as it is,
and a radial `"line"` patch may run from the centre to the edge or from the edge to the centre.
(The same holds for pressure surfaces, §2.3, and given interior surfaces, §2.6, turned to their
own convention; a damping lid, reference/inputs-model-objects §1, takes either orientation as it is.)

- A plate moves with every rigid mode of its body and with the generalized modes whose pieces
  name its elements (a piece without `ranges` moves the plates of other meshes too; a piece with
  `ranges` names elements of the body's mesh). A plate floating apart from a hull is a body of its
  own: a body may consist of plates only (no `mesh`; the body's mesh is then that of its first
  plate), whose rigid heave, roll and pitch are the plate's motions. A plate attached to a hull is
  a plate of the hull's body.
- No mass of its own: the inertia is the body's `mass` and inertia and, for generalized modes,
  `mode_mass`; the motions of a body of plates only need a mass (the piston's, 0 for a massless
  piston: its inertia is its added mass; reference/inputs-rules §1 rules 5, 10).
  No volume: its area is part of the body's waterplane and restoring (theory §3.2, §10.1; the
  restoring of a generalized mode moving the plate in the convention of theory §10.1).
- A plate overlaps no other surface of $z = 0$ (another plate, a lid, an interior surface, a
  pressure surface in $z = 0$), and a thin sheet hanging from it meets it along the edges of its
  elements (split the plate's elements along the sheet): reference/inputs-rules §1 rule 3, theory §9.7.
- Its pressure enters the forces, the added mass and damping and the Haskind relation of its
  body's modes like the wetted surface's; the body outputs (`first_order.body_pressure`,
  `body_velocity`) include its panels or points. No output of its own: the elevation of a plate is
  its displacement.
- Elements of a body's mesh in $z = 0$ that no plate names keep their meaning (the interior surface
  of `interior_surface = "detect"`, or refused as a wetted element).
- Available with plates:
  - both methods and both diffraction formulations, radiation, diffraction, motions, field and
    body outputs at ordinary frequencies and $\omega = 0$;
  - the added mass at $\omega = \infty$ (the plate a rigid thin sheet of the odd-image problem
    there, theory §9.7);
  - the open sea, walls and a channel, and bodies with internal tanks (the tank problems do not
    see the plates);
  - the mean drift by momentum conservation and Newman's approximation of it (theory §11.1);
  - when every plate edge lies on a wall (a cover on its own body's wall, a piston in a tube of
    another body or moved by a generalized mode) or between plate elements: the mean drift by
    pressure integration, its quadratic QTFs and Newman's approximation of it (theory §11.2; the
    higher order without generalized modes, as for every body; not with pressure surfaces);
  - under the same rule of the edges, the potential QTF by both methods in the low order (theory
    §12.7, §12.8): the free surface ends at the plates, so a covered moonpool has one outer
    waterline; it takes the generated panels or a given mesh that covers no plate, not the cells;
  - under the same rule of the edges, the mean drift by the control surface, its quadratic QTFs
    and Newman's approximation of it for the plates of the body whose control surface it is
    (theory §11.3), such as a cover on its own body's wall: the free surface of the control
    surface ends at the outer waterline of hull and plates, and an automatic control surface
    leaves the covered waterline out. The control surface holds its body alone, so a piston body
    inside the control surface of its hull is refused as any second body, and generalized modes
    are not available with the control surface.
- With a plate, `diffraction_formulation = "auto"` solves the total potential (the scattered one
  with low-order thin sheets): under a plate on a deep water column the incident and the
  scattered potentials nearly cancel, and the scattered formulation's error is several times
  larger there (theory §9.7).
- Not available with plates (refused, naming the feature):
  - the $\omega = \infty$ limit with a thin sheet hanging from a plate (the junction of the sheet,
    its image and the plate);
  - the mean drifts by pressure integration and by the control surface and their QTFs for a plate
    with an edge bordered by free surface (a mat, a dock): the line term at such an edge has no
    finite value without a model of the edge; likewise an edge against a plate of another body
    (the vertical velocity jumps there as beside free surface);
  - the potential QTF for a plate with such an edge: its second-order free-surface forcing is not
    integrable at the edge, so it is refused for good;
  - wavemakers in the walls;
  - a plate given as a mesh on a trimmed body (it is generated, below);
  - a plate lying on the fluid of a tank whose free surface is at $z = 0$ (a floating roof: a
    plate of a tank's free surface is the tank's, §2.2);
  - in finite depth, a plate that moves water at $\omega = 0$ (heave), as every mode with a net
    volume flux (rule 9).

```python
owc = m.bodies.add(name="piston", modes=["heave"])          # a body of a plate only
owc.free_surface_plates.add(mesh="water_column")             # the free surface inside the column
```

**Generated plates (a trimmed hull).** A plate given as a mesh lies in $z = 0$ of the untrimmed
hull; any pitch or roll tilts it out of the plane, and the waterline it must fit is known only
after the cut. A plate of a trimmed body (§1.2) is therefore generated by the engine: the item
`inside`, a point $(x, y)$ in the body axes of the trimmed hull, chooses the waterline loop around
it, and the plate is the water of $z = 0$ inside that loop (a moonpool, the water column of a tube)
less the waterplanes of the hulls within it, triangulated as the interior surface of the trimmed
hull is (theory §14): low order by Delaunay refinement to edges of at most twice the mean waterline
segment (the `"waterline"` interior surface), higher order the constrained Delaunay triangulation
of the loop as patches of the hull's kind; its edges lie on the waterline (rule 3 of
reference/inputs-rules §1, the plate's edges on its wall), its normals point down. With the
symmetry the trim keeps the plate is made in the stored part, so a point and its mirror images
choose the same plate. A generated plate takes no `mesh` and no `ranges`. The item `body` names
another body whose trimmed waterline the plate covers (unset: the plate's own body, which must have
a trim): a plate body of its own over a trimmed hull, the piston of a trimmed tube; that body
itself has no trim. Its mesh is the generated plate, in the stored symmetry form of the hull when
the two bodies have the same `position` and `yaw`, else whole.

```python
hull = m.bodies.add("hull", mesh="hull_full", trim=[0.0, 1.5, 0.0], modes=["heave", "pitch"])
hull.free_surface_plates.add(inside=(0.0, 0.0))              # the cover of the moonpool
# or, instead of the cover, a piston body of its own in the moonpool:
# piston = m.bodies.add("piston", modes=["heave"], mass=0.0, center_of_gravity=[0.0, 0.0, 0.0])
# piston.free_surface_plates.add(inside=(0.0, 0.0), body="hull")
```

Here `hull_full` stands for the mesh of a whole hull with a moonpool, its walls above the water
too.

The generated elements are numbered from 1 within the plate in the queries (`element`, role
`free_surface_plate`); `body.trimmed_mesh(name, role="free_surface_plate")` adds them as a mesh.
Refused, naming the reason: a plate given as a mesh on a trimmed body, a trimmed body of plates
only (any angle tilts them out of $z = 0$), a point in the hull's waterplane, in the open sea or on
the waterline, a generated plate without a trimmed hull, `body` naming a body without a trim or a
disabled one, and a generated plate of a trimmed body naming another body.

### 2.5 Generalized modes

`generalized_modes`: a named list of `GeneralizedMode` objects with the item `parity` (the
rigid mode whose reflection parity the mode shares about the symmetry planes of the mesh;
checked) and the list `pieces` of `ModePiece` objects with the items `u`, `v`, `w` and `ranges`:
displacement polynomials, rows (coefficient, px, py, pz) of the terms $c\, x^{px} y^{py} z^{pz}$ in body
axes, on the elements of the body's mesh that `ranges` names (unset: every element; a piece is not
a role, its elements keep theirs). Coordinates
after the six rigid ones, always free (theory §10.1). With the second-order loads (the pressure
drift and the quadratic and potential QTFs, low order) the displacement must be continuous on the
wetted surface: a mode that jumps between two elements of different pieces at a shared vertex is
refused (a kink is allowed, and a vertical jump between a free-surface plate's element and a
wetted one: the plate sliding along the wall, a piston in its tube, whose sliding the potential
QTF's line integral at the wall holds); their generalized forces are
the rows of their coordinates in the
results of these loads ([theory §12.15](../theory/12-second-order-loads.md#1215-generalized-modes)).
The mean drift by the control surface and its QTFs are not available with generalized modes (nor,
therefore, for a free-surface plate sliding along its body's wall by a mode of its own: its
pressure drift is).

### 2.6 Interior surface (irregular frequencies)

The body item `interior_surface` (a switch, §introduction) chooses the surface inside the
waterline: `"detect"` (default), `"waterline"`, `"graded"`, `"projected"`, `"mesh"` or `"none"`
(no interior surface; then a wetted element in $z = 0$ is refused, reference/inputs-rules §1 rule 2). With `"mesh"` the
items `interior_surface_mesh` (a mesh name; no default, needed) and `interior_surface_ranges`
(unset: every element) give it; with the other values they are kept unused:

```python
hull.interior_surface = "waterline"
hull.update(interior_surface="mesh", interior_surface_mesh="hull",
            interior_surface_ranges=[(257, 320)])        # elements of the body's mesh
hull.update(interior_surface="mesh", interior_surface_mesh="lid")   # another mesh, used whole
hull.interior_surface = "none"
```

*Table 3. The values of `interior_surface` and their surfaces.*

| `interior_surface` | Surface |
|---|---|
| `"detect"` (default) | the elements of the wetted surface (the body's `mesh` and `ranges`) lying in $z = 0$ become the interior surface (those another role takes, a pressure surface at $z = 0$, are left to it); none if there are none; if every wetted element lies there they are left, and reference/inputs-rules §1 rule 2 refuses them; not for a trimmed body (it finds none) |
| `"mesh"` | `interior_surface_mesh`, `interior_surface_ranges`: elements at $z = 0$: of the body's mesh, or of another mesh in body axes; either way up, also mixed (each element is turned upward, as the elements of a plate are turned downward, §2.4) |
| `"projected"` | the vertical projection of the downward-facing wetted elements onto $z = 0$ (an upward-facing element is a multi-valued projection and rejected) |
| `"waterline"` | the regions enclosed by the waterline (holes such as moonpools kept) triangulated, edges at most twice the mean waterline segment, angles at least 25 degrees (theory §14); higher order: the interior surface of a trimmed body |
| `"graded"` | per waterplane region a fan from its centroid to the waterline segments, the rays divided at $\sin(\pi k/2N)$, graded towards the waterline **[C]**; regions without openings, star-shaped about the centroid |
| `"none"` | no interior surface |

The generated surfaces are made in the stored symmetry form of the body mesh (theory §14),
from the waterline of the trimmed hull when the body has a trim; generation rejects geometry it
cannot handle (multi-valued projections, open contours, contours that cross or touch) instead of falling back to
another method. The elements of a mesh on its waterplane are a mask over `mesh.elements()` (the
normals upward, the vertices at $z = 0$) and their ranges `hb.ranges(mask)` (reference/inputs-meshes §3).

### 2.7 Control surface

The body item `control_surface` (a switch) chooses the surface of the drift
`second_order.drift_control_surface` (theory §11.3), which needs one on every body: `"none"`
(default), `"mesh"`, `"cylinder"`, `"ellipsoid"` or `"automatic"`; a control surface given with
that drift off is refused. Its items are body items prefixed `control_surface_`; each value takes
those of the table, the others are kept unused. A needed item without a default left unset is
refused by the check, naming it. It must hold its body alone: a control surface whose footprint
(the convex hull of its vertices in the plane) reaches a vertex of another body (of its mesh or
of a free-surface plate of a mesh of its own), or holds a damping lid, is refused (the momentum
balance gives the force on everything inside it, and the free surface inside it is taken as
free). The free-surface plates of the body itself belong inside it (§2.4): the free-surface part
of a given or generated surface lies outside the body's outer waterline. Table 4 lists the values.

*Table 4. The values of `control_surface`, their items and surfaces.*

| `control_surface` | Items and surface |
|---|---|
| `"mesh"` | `control_surface_mesh` (needed), `control_surface_ranges` (unset: the whole mesh): a surface given, in body axes around the body, normals away from it (the vertex order of the CSF format), the parts in the free surface ($z = -z_{body}$ in body axes) for the free surface inside it; a low-order mesh (panel centroids) or a patch mesh (`nu` $\times$ `nv` panels per patch, `iquo` $\times$ `iquo` Gauss points each); `m.meshes.load_csf(name, csf, body=...)` reads CSF files (the automatic surface of a file traced around `body`) |
| `"cylinder"` | `control_surface_radius`, `_depth`, `_inner_radius`, `_panel_size` (each needed): a quadrant of a truncated cylinder with its free-surface part, from the circular waterline of radius `inner_radius` inside, generated when the problem is built |
| `"ellipsoid"` | `control_surface_semi_axes` (a, b, c), `_inner_semi_axes`, `_panel_size` (needed): a quadrant of the lower half-ellipsoid with its free-surface part, from the elliptic waterline of `inner_semi_axes` inside |
| `"automatic"` | `control_surface_depth`, `_panel_size` (needed), `_outer` (a switch: `"circle"`, the default, with `_radius`, or `"polygon"` with `_boundary`; needed), `_partitions` (unset: none): around the body's waterline (theory §14), always with its free-surface part, between the body waterline and the outer surface (a waterline covered by a free-surface plate of the body, a covered moonpool, left out); `radius` of a circle about the body origin, `boundary` the (x, y) vertices of an outer polygon in the stored sector, counter-clockwise; `partitions`, further (x, y) polylines of the free surface, each ended by a row of NaN |
| `"none"` | no control surface |

Every control surface takes `control_surface_waterline_term`: `"control_surface"` (default) or
`"body"`, where the squared-elevation line term is taken, on the waterline of the control
surface, or on the body waterline with $\zeta \nabla\zeta$ over the free-surface part (for a control surface
close to the body; theory §11.3); `"body"` needs `second_order.drift_control_surface`.

```python
hull.update(control_surface="automatic", control_surface_radius=30.0,   # m; outer "circle"
            control_surface_depth=20.0, control_surface_panel_size=2.0)  # m
hull.update(control_surface="mesh", control_surface_mesh="cs",
            control_surface_waterline_term="body")  # radius, depth, ... kept unused
```
