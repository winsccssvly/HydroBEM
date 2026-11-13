# The Rust API of the engine

The engine is the Rust crate `hydrobem_core` (`crates/hydrobem`). Its public API is at the crate
root; the Python extension module `hydrobem._core` (`crates/hydrobem-python`, PyO3) and the C ABI
`hydrobem.dll` (`crates/hydrobem-c`, [c-api.md](c-api.md)) are bindings of it, and a Rust program
uses it directly. The complete reference with
every type and method is the crate documentation:

```bat
cargo doc -p hydrobem_core --no-deps --open
```

The API carries a version number (`BuildInfo::api_version`, 56 for HydroBEM 0.3.0), raised with
every change of the public API (new result quantities included: the enums are exhaustive).

## Types

*Table 1. The types of the public API.*

| Type | What it is |
|---|---|
| `Model` | the model: its document (the inputs as objects), the problem built from it, the `Results` of its last solve and its `ModelState` |
| `ObjectId`, `Obj`, `ObjectInfo`, `ItemInfo` | the handle of an object of a model's document (a typed index, not a pointer; `to_bits` and `from_bits` as one `u64` for a client outside Rust) and its descriptions |
| `Value`, `DType`, `ItemKind` | the value of an item: bool, int, real, text, list of texts, an array of reals or ints with its shape, read by `as_bool`, `as_int`, `as_real`, `as_text`, `as_texts`, `as_reals`, `as_ints`; the mesh of a surface is a `Value::Text` (the item `mesh`, a reference to `meshes`), its elements the item `ranges`, an (n, 2) integer array of the one-based first and last stored elements of each range, made by `Value::ranges(&[[first, last], ...])` (unset: every element); the kinds of items (`ItemKind`) |
| `ItemValue` | the value of an item as the document holds it (`Obj::get`, `Document::set`), an enum of its kinds: `Unset` (no value), `Bool`, `Int`, `Real`, `Text` (a text, a choice, a reference, an origin by name), `Name` (an int or real item given as one of its names, such as `"auto"`), `Reals`, `Ints`, `Texts` (arrays with their shape); read by `is_set`, `is_array`, `is_name`, `dtype`, `boolean`, `integer`, `real`, `text`, `shape`, `reals`, `ints`, `texts`, each the variant's part or the empty value; a scalar is given in the variant of its item's kind (`Document::set` takes an `Int` for a real item as a `Real`, as integers in a real array, and refuses another variant with `ErrorKind::Type`) |
| `Document`, `schema()`, `TypeDef`, `ItemDef`, `ListDef` | the document of a model (the model file: reference/inputs-model §4) and the schema of its object types (reference/inputs-*; `ItemDef::switch`, the switch items that choose the item, and `ItemDef::switch_values`, the values that take it, `"v/w"` for two switches, both empty for an item no switch chooses); `get_obj` reads an object, and a change with the id of another document or of a removed object is refused (`ErrorKind::Argument`) |
| `Results`, `QuantityId`, `QuantityData`, `QuantityShape`, `ElementType`, `RunMeta`, `WamitOptions` | the results of a solve, made by a client or of a results file: the quantities by id (reference/results) and their meta data, read by the methods of `Results` and changed only by `Results::put`; `Results::write_wamit` writes the WAMIT-format numeric files |
| `Owner`, `ResultView`, `ResultGroup`, `Selection`, `Rows`, `Selected`, `result_views`, `result_groups`, `result_view` | the result tree: the views of the quantities by group and owner and the part of an owner (`Results::select`, reference/results §1) |
| `Counts`, `BodyCounts`, `Hydrostatics`, `MeshGeometry`, `MeshElements`, `Panels`, `Tessellation`, `MeshQuality`, `MotionEquation`, `MemoryEstimate`, `PointClass`, `PointStatus`, `PatchPoints`, `FreeSurfacePoints`, `PanelRole`, `QualityFlags` | the answers of the queries of a model, each a struct of its rows, with the columns, units and frames of reference/queries (`Hydrostatics`: of a body, with the displaced mass and the metacentric heights; `MeshGeometry`, `MeshElements`: of a mesh alone; `Panels`: per element its corners, distinct vertices, the seven keys `body`, `role`, `tank`, `owner`, `element`, `part`, `reflection`, centroid, normal and area; `Tessellation`: per quadrilateral its corners and normals and the same seven keys as the panels row of its element; `MeshQuality`: per element (and part) its size, aspect, panels or spans per wavelength and flags, per body its volumes, fewest panels or spans per wavelength and edge counts; `PatchPoints`: points and normals of a patch; `FreeSurfacePoints`: points and weights of a zone of the potential QTF's free surface) (`PanelRole`, the roles of reference/queries Table 2 by `PanelRole::name`: `Wetted`, `InteriorSurface`, `Dipole`, `DipoleDamper`, `PressureSurface`, `Lid`, `Tank`, `FixedDamper`, `ControlSurface`, `FreeSurface` (the panelled zone of the potential QTF's free surface), `FreeSurfacePlate`; `BodyCounts`: `stored`, `wetted`, `wetted_with_reflections`, `dipole`, `interior_surface`, `pressure_modes`, `symmetry_copies` (the copies 1, 2 or 4 of the symmetry planes), `generalized_modes`; `PointClass`: per point its `PointStatus` and its `body`, `lid` and `tank` (zero-based `Option<usize>`, `None` none, which Python and C give as -1; the tank of its body, as the results' `bodies[b].tanks[t]`); `MemoryEstimate`: `peak` resident, `commit` the commit charge) |
| `Route`, `Method`, `Domain` | the solver route of a model (`Model::route`): `Route::Hydrostatics` (no boundary-value problem) or `Route::Solve` with its `Method` (`LowOrder`, `HigherOrder`), the depth (`finite_depth`) and the `Domain` (`OpenSea`, `XWall`, `YWall`, `OrthogonalWalls`, `Channel`); `Route::name` (and its `Display`) the name the results' meta data hold, e.g. `"low_order_infinite_open_sea"`, and a route equals its name as any string type, on either side (`route == "hydrostatics"`, a `str` or a `String`); the three enums are `#[non_exhaustive]`, so that a later method or domain is not a breaking change: a `match` on them has a wildcard arm |
| `MeshFile`, `AnalyticShape`, `ControlSurfaceKind` | the meshes a model reads or makes |
| `Error`, `ErrorKind`, `Location`, `Result` | failures |
| `Diagnostic`, `Severity`, `Warning` | the warnings of the rules and of a solve |
| `Progress`, `ProgressAt`, `ProgressFn`, `Stage`, `CancelToken`, `SolveJob` | the progress and the cancellation of a solve |
| `ProgressMonitor`, `SolveProgress`, `FrequencyProgress`, `LimitProgress`, `FrequencyState`, `Limit`, `Phase`, `WorkerProgress`, `WorkerState`, `Task`, `SecondOrderProgress`, `SecondOrderStep` | the state of a solve at one instant, read from any thread (`Model::progress_monitor`, `Model::progress`) |
| `BuildInfo`, `build_info()`, `wavenumber()` | the build, the dispersion relation |
| `Complex32`, `Complex64` | the complex types of the quantities (re-exported from num-complex) |

## The methods of a `Model`

- Files: `Model::load`, `save` (the model file `.hbm`, reference/inputs-model §4), `Model::load_results`,
  `save_results` (`.hbr`, reference/results §2), `clone_inputs`.
- Objects: `document`, `root`, `object(path)`, `object_info`, `path`, `parent`, `children`,
  `child_at`, `add`, `remove`, `move_to`, `rename`, `clear_list(id, list)` (a list emptied, all
  or nothing; as C `hb_clear`).
- Items: `item_info`, `get`, `set` (the value checked where it is set), `reset_item`.
- Meshes: `load_mesh` (GDF, GDF with SPL, or a geometry file `.hbg`), `hbg_mesh` (a mesh from the
  text of a geometry file), `mesh_hbg` (the text of a mesh), `load_csf`, `load_fdf` (the panels of
  an FDF file as a panels mesh in $z = 0$, for `second_order` `free_surface_mesh` with `free_surface_method = "mesh"`), `analytic_mesh`,
  `body_trimmed_mesh(body, name, role)` (the surface of one role of a trimmed body as a mesh),
  `mesh_write`. The
  generated control surfaces are the items of a body's `control_surface`, made when the problem
  is built.
- The queries of a mesh (its geometry alone, in its own coordinates; reference/inputs-meshes §4):
  `mesh_geometry` (a `MeshGeometry`), `mesh_elements(mesh)` (a `MeshElements`, the stored
  elements), `mesh_tessellation(mesh, divisions, include_reflections)` (a `Tessellation`),
  `mesh_geometry_quality` (a `MeshQuality` without wavelengths), `mesh_patch_points(mesh, element,
  uv)` (a `PatchPoints`).
- The queries of a body, by its `ObjectId` (after the rules of the environment, the geometry and
  the surfaces, reference/queries): `body_hydrostatics`, `body_panels(body, roles,
  include_reflections)` (`Panels`), `body_mesh_quality` (`MeshQuality`), `body_tessellation(body,
  roles, divisions, include_reflections)` (`Tessellation`). `roles: &[&str]` names roles of
  reference/queries Table 2 (empty: every role but the control surface and the free surface; a
  body refuses `"fixed_damper"` and `"free_surface"`, `ErrorKind::Value`), `divisions: [i32; 2]`
  the quads per patch in u and v (1..256; `[0, 0]` one quad per span of the potential
  discretization), `include_reflections: bool` also the reflected copies. Their elements are
  numbered as the stored elements of the meshes (`Panels::element`, one-based).
  `Hydrostatics` holds `volumes` (the three estimates) and `volume` (their mean), `cb`, `cg`
  and `wp_centroid` in body coordinates and the stiffness matrices about R.
  The answers of the queries are not result quantities: `QuantityId` lists what a solve computes.
  The views by owner (reference/results §1) are the result tree of the engine: each quantity
  names its view (`QuantityId::group`, e.g. `drift.pressure_integration`) and the rule of an
  owner's part (`QuantityId::owner`, an `Owner`); `result_views` and `result_groups` list the tree.
  `Results::select(path, body, owner)` gives the part of a body, a lid, or a pressure surface or a
  tank of a body (a `Selected`: dimensions and elements, copied; None when not computed or of no
  part),
  `Results::selection` the rows it takes of each axis (a `Selection` of `Rows`), and
  `Results::body_columns(body)` the coordinates and radiation columns of a body.
- Rules and the queries of the model: `check` (the model rules, reference/inputs-rules §1, and the refusals of
  `route`), `diagnostics` (the
  warnings of the last check), `route` (a `Route`), `counts`, `body_counts(body)` (by its `ObjectId`),
  `solved_body_counts` (the `BodyCounts` of every body solved, in order), `lid_elements`,
  `radiation_mode_ids`, `motion_equation`, `panels(roles, include_reflections)` (the elements
  after a trim, of every body, the lids and the fixed dampers, of the roles named or, with `&[]`,
  of every role but the control surface and the free surface; `panels(&["free_surface"], true)`
  the panelled zone of the potential QTF's free surface as the solve integrates it),
  `free_surface_points(zone)` (a `FreeSurfacePoints`, the points and weights of a zone of that
  free surface: `"panelled"`, `"quadrature"`, `"outer_circle"`; theory §12.8–§12.9; an unknown
  zone is `ErrorKind::Value`; these free-surface queries check the rules of the free surface,
  reference/queries), `tessellation(roles, divisions, include_reflections)`,
  `mesh_quality()`, `mode_displacements(points, bodies, elements)`, `memory_estimate`,
  `classify_points(points, tank: Option<ObjectId>)` (None: points in the sea, global coordinates;
  the `ObjectId` of a tank: points inside it, body coordinates), `progress` (the state of the
  running or last solve).
- The lists with optional names (`ListDef::optional_names`: the dipoles, dipole dampers, tanks,
  pressure surfaces and couplings): `add` with a name or `""`, `rename` (`""` removes the name);
  `object(path)` finds such an object by its name or its one-based position.
- References name objects, never positions: an item of kind `ItemKind::Reference` (a body, a
  mesh by its name; `ItemDef::target` the list it names; `is_reference_item`).
  They follow `rename`, and `remove`, `rename(id, "")` and a `clear_list` that would
  leave one unresolved fail with `ErrorKind::Value`.
- Suppressed objects (reference/inputs-model §5): an object whose item `enabled` is false is left out of the
  problem the queries of the model and the solve build (a copy of the enabled objects of the
  document; the engine core never sees the others). The locations of the errors and of
  `diagnostics()` are those of the document; `check` adds a
  `Warning::DisabledObject` per disabled object that would fail enabled. The queries of a body
  take a disabled body as enabled; `body_counts` refuses one. `Results::document` is the model
  with every object, `Results::solved_document` its enabled objects, which the quantities and the
  numeric files number.
- Solve: `solve(progress)`, or in three steps `begin_solve`, `SolveJob::run`, `finish_solve`;
  `results`, `reset`, `clear`, `state`, `cancel_token`, `new_results` (results computed
  elsewhere, filled with `Results::put`; with the axes and the points of the field outputs of the
  model, which `put` refuses). `Results::check_given(&self) -> Result<()>` checks the
  rules of the quantities given together (reference/results) for results made by
  `Model::new_results` once every quantity is put: `ErrorKind::Value` naming the quantity missing,
  `ErrorKind::State` without a model. `RunMeta::environment_hooks` lists the hooks of the
  environment set for the solve (`HB_DIAG_DOUBLE`, `HB_VALIDATE_<name>=<value>`,
  `HB_WATERLINE_CENTROID`, `HB_AVAILABLE_MEMORY=<n>`, `HB_FIELD_STATICS_LIMIT=<x>`, `HB_PROFILE`; for validation studies
  and tests only), empty without; one that changes the results is `Warning::EnvironmentHook` (code 9).
- The bindings name a few of these by their own idiom: Python `body.panels`, `body.trimmed_mesh`,
  `mesh.elements()`, `m.motion_equation()` (the matrices None while not `ready`) and
  `write_wamit(frequency=..., timestamp=...)` are `Model::body_panels`, `body_trimmed_mesh`,
  `mesh_elements`, `motion_equation` (`MotionEquation::ready`) and `WamitOptions { omega, stamp }`;
  the C names are in [c-api.md](c-api.md).

## Errors

Every fallible function returns `Result<T, Error>`. An `Error` has a kind (`ErrorKind`: value,
type, unsupported, state, cancelled, internal, argument, numerical, io, exists), a message and a
`Location`: the zero-based body, mesh, field point, lid or frequency and the one-based mesh
element it concerns (`body`, `mesh`, `element`, `field_point`, `lid`, `frequency`), each `None` where it does not apply. The Python exceptions are chosen by the
kind (design.md, Errors). A model refuses a change of its inputs while it is calculating
(`ErrorKind::State`), and every text the model keeps is checked to be valid UTF-8. A panic is a
bug of the engine.

## Threads, progress and cancellation

A solve runs on `rayon` pools sized by the options `frequency_threads` and `inner_threads`;
its results do not depend on the thread counts (they are bit-identical). The progress
callback of `solve` receives `Progress` reports (the stage, what it is about, `ProgressAt`: a
frequency, a limit, an item of a second-order step or the solve as a whole; its steps, the
frequencies done, the frequency worker that reports and the `Phase` of the solve), from one
thread at a time; the end of every frequency and every change of the phase always arrive, the
other stages at most every 0.1 s. A `CancelToken` (`Model::cancel_token`, `Clone + Send + Sync`) cancels the
running solve from any thread, or the next one when none runs; the solve then fails with
`ErrorKind::Cancelled`. A `ProgressMonitor` (`Model::progress_monitor`, `Clone + Send + Sync`)
gives a `SolveProgress` of the running solve, or of the last one, at any instant (`snapshot`;
it never waits for the solve): the preparation done, per limit (zero, then infinite) and per
ordinary frequency (in the order of the results) its `FrequencyState`, stage, fraction done
(the stages weighted by their cost in the solve; never decreasing, 1 once done) and worker, the
frequencies done of all, the `Phase`, per frequency worker (its fixed number 0 to
`frequency_threads - 1` its index) a `WorkerProgress` (its `WorkerState` and the `Task` it is
on), and per step of the second order a `SecondOrderProgress` (its `SecondOrderStep`, state,
fraction and items) ([user guide, performance](../guide/performance.md#5-the-progress-of-a-running-solve)). A `SolveJob` takes the built problem out of the model, so that the model
can be read on another thread while the job runs.
