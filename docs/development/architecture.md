# Architecture

How HydroBEM is put together: the modules of the engine and of the Python
package, the solve flow, tables, tests and the conventions used everywhere.
What the program does is described in the other documents of `docs/`; the methods, their
parameters and accuracy in [the chapters of the theory](../theory/README.md) (cited in the code as
`theory §N.M`; the folders of `docs/` and the citation forms in [rules.md](rules.md)).

## Overview

```
Python package (hydrobem)                      engine (hydrobem_core, Rust API)
  engine.py ── hb.Model: items, objects, ─────▶ model.rs     Model: the public API (via hydrobem._core)
               meshes, solve, queries            document*.rs the model as objects, its record, its meshes
                                                 problem.rs   the problem built from them; rules/: the model rules (reference/inputs-rules §1)
                                                 route.rs     route of the problem (or the reason for none)
                                                 solver.rs    low order: lo_*.rs; higher order: ho_*.rs
  results.py ◀── quantities (QuantityId) ◀───── results.rs, results_file.rs (the results in the model; .hbm, .hbr)
                                                 wamit_files.rs .1 .2 .3 .4 .5 .6 .7 .8 .9 .hst .pnl .out
```

- The engine holds the model: its inputs as objects with named items, its meshes among
  them, and the results of its last solve. It computes **dimensional** results (the
  first-order quantities, the mean drifts and the second-order loads) and the geometry
  preprocessing (trim, generated surfaces, control surfaces, the free surface of the potential
  QTF). It reads and writes the geometry files, the model file and the results file, and writes
  the WAMIT-format numeric files with their normalization; Python computes and writes nothing
  of its own ([design.md](design.md)). A solve's results hold what its model asked for.
- Python talks to the engine only through the extension module `hydrobem._core`
  (`crates/hydrobem-python`, PyO3), a binding of the engine's public Rust API: its classes
  `Model` (the engine model; the solve runs with the GIL released, the progress callback and other
  threads read the model and cancel the solve meanwhile), `Object` (an object of a model: its
  items, its lists, the functions on a mesh) and `Results` (the quantities as read-only NumPy
  arrays over the engine's data). The engine's errors raise the exceptions of `errors.py`.
- Other programs use the C ABI `hydrobem.dll` (`crates/hydrobem-c`, [c-api.md](../reference/c-api.md)),
  a second binding of the same API: the model behind an opaque handle with its lock, objects by
  `uint64_t` ids, items set by kind, answers as tables of named columns, result quantities as
  views, statuses with the last error of the thread; `include/hydrobem.h` is written from its
  sources by `scripts/gen_c_header.py`.
- Numbers: geometry and kernels in double; the low-order influence matrices and LU in
  complex64 (the blocked LU of `linalg.rs` on faer), the higher order and the diagnostic mode in complex128; the
  coefficients stored as float32/complex64, the hydrostatics, the mode matrices and the drifts as
  float64.

## Engine modules (`src/`)

The engine is the Rust crate `crates/hydrobem` (library `hydrobem_core`, an `rlib`, in the cargo
workspace of `Cargo.toml` at the root); the Python extension module
`hydrobem._core` (`crates/hydrobem-python`, PyO3) and the C ABI (`crates/hydrobem-c`) link it. Its public API is at the crate root
([reference/rust-api.md](../reference/rust-api.md)); the other modules are internal. The engine is
pure Rust: its dependencies are crates (faer, rayon, num-complex), and
`build.rs` only writes the build info; `.cargo/config.toml` compiles for AVX2 and FMA (the C runtime
is linked dynamically, the `vcruntime140.dll` of Python or of the Visual C++ Redistributable) (a build without them is a compile error of `lib.rs`: the kernels call
their intrinsics from ordinary functions; a `RUSTFLAGS` of one's own replaces the flags of
`.cargo/config.toml` and must name them too). `src/lib.rs` lists the modules, one file per
subject (the potential QTF a directory of its own; Table 1).

*Table 1. The modules of the engine.*

| File | Role |
|---|---|
| `model.rs` | `Model`, the public API: its construction, files (.hbm, .hbr), state and results, the problem built from its document, its objects and items; it re-exports the public types of the files below |
| `model_types.rs` | the small public types of the model: `Value`, `Progress`, `ProgressAt`, the answers of the queries (`ItemInfo`, `ObjectInfo`, `Counts`, `BodyCounts`, `MotionEquation`, `PointClass`, `PatchPoints`, `FreeSurfacePoints`), `MeshFile`, `AnalyticShape`, the internal control surface and trim of a mesh, `wavenumber` |
| `model_queries.rs` | the queries of `Model` by owner (of the model the rule check with the route's refusals, the route, counts and `solved_body_counts`, panels and tessellation (by role, reference/queries §1), mesh quality, motion equation, memory, ...; of a body its hydrostatics, panels, mesh quality and tessellation on the problem of that body alone (its rules only), a disabled one as if enabled; the panels (role `free_surface`, private `free_surface_panels`) and the points of the potential QTF's free surface), the status of points (in the sea, or in a tank), empty results |
| `model_solve.rs` | the solve of `Model` (`solve`, `begin_solve`, `finish_solve`), the checks before a solve (`pre_solve`, also of `Model::check`) and `SolveJob`, with the hydrostatics and coordinates a solve adds to its results and its field outputs split by owner (the sea's points, each tank's) |
| `model_meshes.rs` | the meshes of `Model` as objects (read, made, written; the panels of an FDF file), the queries of a mesh alone, the control surfaces the engine makes |
| `model_trim.rs` | the trimmed surfaces of a body (`body_trimmed_mesh`, on the internal trim of a mesh `trim_mesh` that the build uses too) |
| `codes.rs` | the codes of the public API as enums: `QuantityId`, `ElementType`, `Stage`, `ModelState`, `Severity`, `Warning`, `PanelRole`, `QualityFlags`, `PointStatus`, `ControlSurfaceKind` |
| `build_info.rs`, `quantities.rs` | the build of the library (`build.rs` writes the commit and the compiler) and the table of the result quantities (with each one's view in the result tree) |
| `text.rs` | file text: UTF-8 lines and tokens as Python's `str` methods, free-format numbers, Python's `repr` of a double, file access |
| `hbg.rs`, `json.rs` | the geometry file .hbg: one mesh object read from and written to JSON through the item checks of the model; a small JSON reader and writer (RFC 8259) |
| `free_surface_panels.rs` | the generated panels of the potential QTF's free surface (a square grid about the waterlines with triangles to them, blended layers and rings to R), their centroids and areas |
| `gdf.rs` | GDF, SPL, CSF and FDF (its panels) readers, GDF and SPL writers, the panel-size rule, the analytic codes of a GDF file as named shapes, the analytic control surfaces |
| `polygon.rs` | polygons in the plane: the nesting of closed loops into regions with holes, the test for loops that cross or touch, the triangulation of a region with holes (bridges, ear clipping, Delaunay edge flips; theory §14) and its Delaunay refinement (the `waterline` interior surface), for the `waterline` interior surface (`interior.rs`), the interior surface of a trimmed higher-order body and the generated free-surface panels (`free_surface_panels.rs`); a tree of boxes (a bounding-volume hierarchy) for the searches of the plate edges and of the plate elements beside a waterline (`plate_rules.rs`, `plate_geometry.rs`); the convex hull and the test of a point in a convex polygon |
| `binary.rs` | the little-endian records of the model and results files |
| `model_codes.rs`, `mesh_desc.rs` | the choices of the problem as enums (`Domain`, `Formulation`, `Origin`, the QTFs' `QtfHeadings` and `QtfFrequencies`, `WaterlineTerm`, `Mass`, ...), the drifts of Newman's approximation (`NewmanDrifts`) and `QuantitySet` (the quantities a solve computes for others, `Options::hidden`); the internal mesh descriptors the readers, the trim and the geometry work on, and `MeshKind` (the kind of a mesh object) |
| `document.rs`, `document_schema.rs`, `document_rules.rs` | the model as objects with named data items: values (`ItemValue`, an enum of the kinds of the items), the typed reads by which the build reads them (`Obj::boolean`, `integer`, `real`, `choice`, `origin`, `reals`, `ints`, `texts`, `shape`, `mesh`, `body`, ...: the item's kind checked, an item without a value an internal error, never a zero), the checks of a value alone, the object tree and paths, the references by name to an object of a list of the root (`meshes`, `bodies`; followed by a rename, blocking a removal); the schema |
| `wamit_files.rs` | the WAMIT-format numeric files and the `.out` summary: the normalization (reference/file-formats §1), the Fortran fields, the availability rules, the collision checks, the write of all files or none |
| `results_file.rs` | the model file `.hbm` and the results file `.hbr`: their layout, CRC-32, atomic write, reading on demand |
| `document_build.rs` | the problem of the model: the rules between items and objects (one role per element of a body's mesh, the ranges within the mesh), then the engine's problem filled from the objects: the items `mesh` and `ranges` resolved, per body one engine mesh from the elements its parts take (the interior surface detected, the trim made here when the body has one; a body of sheets only on the mesh of its first dipole, one of plates only on that of its first plate), the sheets of a tank with its elements, the couplings as the $6 \times 6N$ rows of a body, the field points (the sea's, then the tanks' placed with their bodies), the generated control surfaces, the lids and fixed dampers |
| `document_enabled.rs` | the model as it is solved: the copy of the document without its disabled objects (`enabled = false`, with what goes with them: the couplings to a disabled body, the field points in a disabled tank), one per problem built from the model (`Scope`: the model, a body's queries, a body's mesh, the free surface's queries; `scoped_document`, made by one walk of the tree, `suppressed` with its `Walk`), and the numbering of its bodies, lids and field points in the model, by which the errors and warnings are located |
| `document_record.rs`, `document_mesh.rs` | the model as a binary record, read by setting every item again; the mesh objects as mesh descriptors, and filled from geometry files |
| `error.rs` | `Error` with its `ErrorKind` and `Location`, returned in a `Result` and propagated with `?`; the location attached where it is known (`Error::at`, `ResultExt::located`); `Diagnostic` |
| `constants.rs` | $\pi$, $\gamma$, the standard gravity, the table hash, the number format of messages |
| `results.rs` | `Results`: the quantities by `QuantityId` (type, dimensions, data) and the diagnostics of a solve, its fields private; `put`; `parts`, the access of the component tests |
| `result_tree.rs` | the result tree: the views of the quantities by group and owner (the `group` and `owner` of each row of `quantities.rs`, the `Owner` rules, the groups), `Results::selection` and `Results::select` (an owner's part of a quantity: its rows, or a copy), `Results::body_columns`; `result_views`, `result_groups` for the generators and the bindings |
| `problem.rs` | `Problem`: the engine's problem (meshes, bodies, lids, environment, frequencies, options), its types, the geometry derived from them (the roles of the elements, `Role`, with their owners; the origins of the meshes) and the counts of its coordinates |
| `rules/` | the model rules (reference/inputs-rules §1) as `impl Problem`, by area: `options.rs` (the options and their combinations, the potential QTF's free surface, the footprint of the control surfaces, something to solve, the motions, the integration), `geometry.rs` (the bodies, the lids, the fixed dampers, then the roles of the elements), `surfaces.rs`, `frequencies.rs`, `environment.rs` (the domain and the walls), `coordinates.rs` (the flux at $\omega = 0$, the inertia, the torn generalized modes); `mod.rs`: the one ordered table of the rules (`RULES`) with the scopes each belongs to (`RuleScope`: the model, a body's queries, the free surface's queries, the classification of points), `check_rules` and the scoped checks, the warnings |
| `hydrostatics.rs` | `Hydrostatics` and `Problem::hydrostatics` (theory §3.2, the internal tanks of theory §10.2), the rigid inertia about the rotation centre (`rigid_inertia`, theory §3.1), the symmetric eigen-decomposition of the inertia rule |
| `field_points.rs` | the field points: the verdict on each point (`PointVerdict`, `classify_field_points`) from the closed wetted surface of each body and its solid angle, and `check_field_points`, the first point that breaks a rule |
| `mesh.rs`, `geometry.rs` | the engine's mesh (`Mesh`, `RecordShape`, `Range`), panel and patch construction and validation (projection to the midpoint plane), reflections, placement |
| `bspline.rs`, `quadrature.rs` | B-spline basis evaluation (one evaluator, orders $\le 8$; the rational patches weight it in `mesh.rs::patch_eval` and `PatchGeo::eval`), Gauss rules (one cache of the orders 1–16, `gauss_nodes`) and element quadrature for hydrostatics and patch meshes |
| `panel_integrals.rs` | Rankine source/dipole integrals (exact near, float32 AVX2 Gauss tiers far), log-singularity integrals, in-plane log integral |
| `green_inf.rs` | infinite-depth wave kernel (tables + asymptotics), Bessel J/Y |
| `green_channel.rs` | channel far-image tail: Faddeeva function, periodic lattice sum (strip modes / Ewald), per-frequency tiles (values, first and second derivatives) |
| `green_fin.rs`, `dispersion.rs` | finite-depth Green function (Newman decomposition, eigenfunction series, per-frequency tiles), dispersion roots |
| `interior.rs` | interior free-surface panels: selection, external mesh, `projected`/`waterline`/`graded` generators |
| `plate_rules.rs` | the rules of the free-surface plates in $z = 0$: no overlap of a plate with another plate, a lid, an interior or a pressure surface (convex clipping of the elements), and a thin sheet meeting a plate or a lid only along the edges of their elements (reference/inputs-rules §1 rule 3); the edge kinds of the plates (an edge on a wall, between plate elements, or bordered by free surface: the pressure drift and the potential QTF refuse the last, theory §11.2, §12.7); a plate of the sea over a tank's fluid in $z = 0$ |
| `plate_geometry.rs` | the geometry of the surfaces in $z = 0$ that the plate rules and the solvers share: an element's horizontal polygon (convex pieces, boundary, box), the plane vertices of a mesh (the footprint of a control surface, theory §11.3), the plate elements of a problem in the numbering of the body outputs, the one search of the plate beside a waterline side (`PlateIndex`: the pressure drifts of both orders and the potential QTF's sides of a waterline under a plate), and the winding of a point about a waterline (`Winding`) |
| `linalg.rs` | the dense linear algebra on faer: the blocked complex64 and complex128 LU with partial pivoting, its blocks on the threads of the current pool and the progress and cancel check per panel, the real LU of the projections, the products C = A B, the per-thread work buffers of faer's products and the parts the LU writes, for the memory estimate |
| `progress.rs` | `SolveContext` (the `CancelToken`, the `Reporter` and the board of a solve) and `Work`, its view per step (with its frequency worker); `StepWork` and `ItemWork`, a second-order step and its items, and `SubProblem`, a sub-problem of the potential QTF reported as a step: the progress reports, at most every 0.1 s (the phases and the end of every frequency always), and the cancel checks inside the steps; the board (the phase, each frequency's and second-order item's state, stage, fraction and worker, each frequency worker's state and task as atomics, written by the steps without a lock) and the `ProgressMonitor` that reads it as a `SolveProgress` |
| `parallel.rs` | the rayon pools of a solve sized by the options (the frequency workers, each with its fixed number and its inner pool, for the frequencies and the second-order steps that run by item: `FrequencyPools::for_each_item`, `StepRows`; the pools of the statics and of the shared second-order work); the items write disjoint data, so the results do not depend on the thread counts |
| `memory.rs` | `memory_estimate`: the memory of a solve from the sizes of the problem, resident and committed, with the sub-problems of the potential QTF; `check_memory`: the estimate (alone) against the available physical memory before a solve (`options.memory_check`) |
| `route.rs` | `select_route` and `ho_unsupported`: the route of a valid problem (`Route`, public: the hydrostatics, or a `Method` with the depth and the domain; `Model::route` returns it, `Route::name` is the name of the results' meta data, the `.hbr`, C `hb_model_route` and Python `m.route()`), or an `ErrorKind::Unsupported` error with the reason |
| `solver.rs` | `solve`: the hydrostatics route, the dispatch to the higher order, the low-order solve (geometry, statics, the point outputs of the panels) |
| `solve_driver.rs` | the parts of a solve shared by both methods: `allocate_outputs`, `finish_plan`, `run_frequencies` (the phases: limits on worker 0, then the frequencies on the workers; progress, cancel), `common_results` (with the quadratic QTFs as second-order steps), the lid, pressure-surface and body outputs |
| `motion.rs` | motion matrices (theory §3.1) and the motion solve shared by both methods |
| `solve_common.rs` | the plan of a solve shared by both methods: coordinates, pressure-coordinate restoring, motion matrices of the plan |
| `pressure_drift.rs` | the pressure-integration drift of one body (theory §11.2; surface, waterline, closed forms, the moments about the rotation centre and the moving one: `about_rotation_center`, shared with the control surface), both methods; `ClosedForms`, the hydrostatic closed forms shared with the quadratic QTFs; with generalized modes their displacement, Nanson's surface element and their generalized forces (theory §12.15) |
| `heading_pairs.rs` | the heading pairs of the drifts and the QTFs: the list $(h, h)$ or $h_a \le h_b$, the selection of a list from the coefficients of every heading pair, the `*_heading_pairs` quantities; the second-order loads placed on the coordinate axis (`CoordinateRows`: the rigid rows and the generalized forces at their coordinates, NaN on the others) |
| `quadratic_qtf.rs` | the quadratic QTFs (theory §12.1): the first-order values each frequency's pressure drift records, and the pairs of frequencies and headings formed after the solve (with the generalized rows of generalized modes) |
| `potential_qtf/mod.rs` | the force of the second-order potential (theory §12.1; theory §12.4–§12.15) by the indirect or the direct method: its two sub-problems (made quiet of the model's other outputs, checked by the rules), the field points, the phases and the output |
| `potential_qtf/zones.rs` | the zones of its free surface (theory §12.8–§12.9): the panelled zone (the generated panels, the cells, or a given mesh), the quadrature zone of annuli, the outer circle with the orders of the far-field series; the waterline sides of every body with the plate over each (the free surface ends at the plates) |
| `potential_qtf/far_field.rs` | the fields beyond the panelled zone (theory §12.9): the series fitted on two circles, the quadrature zone's fields interpolated in 1/r through the circles |
| `potential_qtf/outer.rs` | the free surface beyond the outer circle (theory §12.9): the forcing of every pair on a radial Gauss rule with a taper, against $\psi_j$ (indirect) and the outgoing radial functions (direct), summed over the radii in order |
| `potential_qtf/items.rs` | the inner free surface and the bodies of every pair (theory §12.5–§12.9): the indirect forces and the direct method's extra columns with their solve, both in one pass |
| `potential_qtf/forcing.rs` | the free-surface forcing of a pair (theory §12.4), its second derivatives moved by the divergence theorem (§12.8), the incident plane waves and the one wall (§12.14) |
| `potential_qtf/body.rs` | the bodies in the potential QTF: their points, coordinates and radiation columns in the sub-problems, the waterline values (§6) and the body condition of the motions (§5, §12), with the line integral of a waterline side under a plate (the jump of the displacement, theory §12.7) |
| `potential_qtf/bessel.rs` | the Bessel functions of the far field: $J_n$, $Y_n$, the Hankel functions $H^{(2)}_m$ with their derivatives, $e^x K_n$ |
| `control_drift.rs` | the mean drift of one body by the momentum flux through its control surface (theory §11.3; both forms of the waterline term) |
| `tanks.rs` | internal tanks: the sub-model of each tank and its terms in the body's equations (theory §10.2); field points inside tanks; the tank-wall pressure drift |
| `analytic.rs` | closed-form patch maps of the analytic geometry (lines, arcs, bilinear; profiles) |
| `analytic_shapes.rs` | the named analytic shapes (cylinders, sphere, ellipsoid, barge, moonpool cylinder, dipole discs) as analytic patches by their dimensions and parts |
| `low_order.rs` | internal types of the low order: `Geometry`, `Static<R>`, `FreqCtx`, `Plan`, `Output`, pair kernels, the channel influence and its far images (body and sheet rows), incident wave, far kernel, thread blocks, profiling |
| `lo_geometry.rs` | `build_geometry` by the stages of `Builder` (lid layout, dampers and pressure surfaces, body owners, external lids, frame and panels, generalized modes, characters, points) |
| `lo_static.rs` | static influence data: Rankine integrals of the image terms, log-singularity corrections by the tolerance rule |
| `lo_frequency.rs` | `solve_frequency` by the steps of `Frequency<R>`: boundary values, character matrices, sheet rows, LU per character, forces, Haskind, lid and pressure-surface elevations, body pressure, field points (`field_values`), Kochin functions; then the motions |
| `preprocess.rs` | geometry preprocessing: the trimmed waterline of low-order panels (clipped, tank cuts) and of patches (remapped, split where the fibres change class; the interior surface from the waterline loops, a fan or a triangulation with holes; the free-surface plates generated in a loop of the water around a point, `plate_triangles`), the body waterline, the automatic control surface |
| `quality.rs` | mesh quality: panel or span sizes, aspect, panels or spans per wavelength, the edges of the closed surface (of a patch mesh, the patch sides), the volumes, of a body or of a mesh alone (`geometry_quality`), answered as a `MeshQuality`; the mesh warnings of the rules |
| `response.rs` | the momentum drift from the Kochin functions and the motions, the Kochin angle count, the mode displacements |
| `inspect.rs` | model inspection: the panels (`Panels`) and the tessellation (`Tessellation`) of a `Scope` by role, from one walk over its elements so that both carry the same seven keys (every body with the lids and fixed dampers, or one body; with the surfaces of a body of their own, generated ones included), `Model::mesh_patch_points`; of a mesh alone its elements (`MeshElements`), tessellation and `MeshGeometry` (volume, centre, area, bounds, waterplane); the answers are structs of their rows, no result quantities |
| `lo_velocity.rs` | the source formulation by the steps of `Sources<R>`: source strengths, field velocities, panel and body velocities, the pressure drift, the control-surface drift |
| `patch_geometry.rs` | `PatchGeo`, the geometry of one stored patch (flat, B-spline, analytic, trimmed element), used by the higher order and by the low-order control surfaces of patch meshes |
| `ho_internal.rs` | internal types of the higher order: patch discretization, instances, quadrature points, `HoGeom`; the spline values of a solution at a point (`spline_values`) and the L2 projection of point values onto the basis (`project`) |
| `ho_geometry.rs` | `build` by stages: patches and kinds, frame and instances, wall and channel images, layout, inner/outer Gauss points, waterline, body output and specified points, field points, control-surface stencils; the compact point arrays of the wave rows (`build_wave_arrays`) and the factored mass matrices of the projection (`build_projections`) |
| `ho_static.rs` | tiered static integrals ($3 \times 3$, inner, $6 \times 6$, adaptive, Duffy graded on elongated panels; tiers per image) |
| `ho_merge.rs` | the higher order: the surfaces of meshes of their own and the lids in global coordinates appended to the elements of the body's patch mesh before the rules (kept as `MeshOrigin.joined`, so that the inspections list them as the low order) |
| `ho_sheets.rs` | rows of the flat dipole patches (with the channel terms) |
| `ho_frequency.rs` | one frequency by the steps of `Frequency`: incident wave, body and sheet rows (the wave rows with the channel terms), LU per character, forces, Haskind, elevations, field points and velocities, Kochin functions, body outputs, pressure and control-surface drift |
| `ho_solver.rs` | `solve_higher_order`: patch geometry, restoring, statics and sheet rows; the shared driver for the rest |
| `env_hooks.rs` | the environment hooks of the engine: the double-precision diagnostic mode `HB_DIAG_DOUBLE`, the waterline diagnostic `HB_WATERLINE_CENTROID` (the low-order waterline potential at the panel centroid, unexpanded, `lo_velocity::at_waterline`), the validation hooks `HB_VALIDATE_<name>` of the sensitivity studies (`validation/constants.py`; `VALIDATION_CONSTANTS` lists every hooked constant with its value, `validation_constant(name)` returns it unless the variable is set; validation only, a hook never changes a default, a unit test holds every value to the theory documents), `environment_hooks` (the hooks set, recorded in `RunMeta::environment_hooks`, a warning when they change the numbers; an unknown `HB_VALIDATE_` name or a value that does not parse refuses the solve), `env_number` |
| `tables/*.rs` | generated tables (never edit by hand; see below) |

### Where a rule goes

An input is refused at the first layer that can see what is wrong (Table 2): a value alone when
it is set, the objects and their names when the problem is built, the geometry and the built
problem by the model rules, and a valid problem that no solver takes by the route. A new rule
goes into the first layer that has what it needs, so that it is refused as early and with as
little built as possible; each rule is written once, and its message names the items it refuses
(`first_order.motion = true requires ...`; `tests/python/test_rule_messages.py` checks that every
item named so exists in the schema). The rules themselves are listed in
[reference/inputs-rules.md](../reference/inputs-rules.md).

*Table 2. The layers of the rules, in the order they apply.*

| Layer | Where | What it sees |
|---|---|---|
| a value alone | `document::check_value`, the rules of `document_rules.rs` (`ItemRule`) | one item when it is set: its kind, shape, choices and range (a finite number, a name, element ranges) |
| the objects and their names | `document_build::check_objects` (the model), `check_body_objects` (the bodies, lids and fixed dampers), with `check_body`, `check_plate`, `check_couplings` | the items of an object together and the objects between them, before the problem is built: switches that need others, a mesh of the right kind, references by name, the parts of a body's mesh |
| the geometry and the built problem | the `rule_*` functions of `rules/` by area (the table `rules::RULES` gives their order: the options, the geometry, the surfaces, the two rows below, then the outputs, the motions, the frequencies, the environment and its domain with the field points (`field_points.rs`), the integration, the flux at $\omega = 0$, the inertia) | the problem in the engine's terms: elements in z = 0, the roles of the elements, the surfaces of a body, the coordinates |
| the plates in z = 0 | `plate_rules::check` | the plates against the other surfaces of z = 0 and the sheets meeting them |
| the tanks' plates | `tanks::check_tank_plates` | the rules of the problem of each tank with plates, in the tank's frame |
| valid but without a solver | `route::select_route`, with `plates_unsupported` (route.rs), `tanks::tank_plates_unsupported` and `ho_unsupported` (the higher order) | a combination the engine has no method for: refused as `Unsupported`, not as a wrong input |
| the mesh warnings | `quality.rs` (`quality_warnings`, through `rule_warnings` at the end of `check_rules`) | coarse, open, inverted or volume-less meshes: a warning, never a refusal |

The rules apply to four problems (scopes), each a subset of the layers; of the model rules each
scope takes the rules of `rules::RULES` that name its `RuleScope`, in the order of the table (a
unit test of `rules/mod.rs` writes out the order of each). Each problem is built from its own
copy of the document (`document_enabled::Scope`, `scoped_document`) by
`document_build::build_scoped`, and checked by `Problem::check_scope`; the problem of the last
query of a body or of the free surface is kept in the model (`ScopedProblem`) until an input
changes or a query of another scope:

- **the model** (`Model::check`, `Model::route`, a solve): every layer: `check_objects`, the build,
  `check_rules` (with the plates, the tanks' plates and the mesh warnings), the route, and the
  zones of the potential QTF's free surface (`potential_qtf::zones`): the checks before a solve
  (`Model::pre_solve`), which `Model::check` and a solve (`begin_solve`) both apply. `Model::check`
  then checks each disabled object as if it alone were enabled (a warning); a solve adds the
  memory check (`memory::check_memory`, the estimate alone, `options.memory_check`).
- **a body's queries** (hydrostatics, panels, tessellation, mesh quality, the trimmed mesh;
  `model_queries::body_query`) run on the problem of that body alone (`Scope::Body`):
  `check_body_objects` without the second order, the build, then `Problem::check_body_rules`: the
  values of the environment, the geometry and the surfaces; not the plates, the options, the
  outputs, the frequencies, the field points, the warnings or the route.
- **the free surface's queries** (the panels and tessellation of the role `free_surface`, the
  points of the potential QTF's free surface; `model_queries::free_surface_query`) run on the
  problem of the free surface (`Scope::FreeSurface`, the motions off, no field
  points): the free-surface items of `second_order` and `check_body_objects`, the build, then
  `Problem::check_free_surface_rules`: the free-surface items, the geometry, the surfaces, the
  plates in z = 0 and the environment with its domain; `rule_frequency_values` where a zone's auto
  values take the frequencies; then `potential_qtf::zones`.
- **the classification of points** (`Model::classify_points`) runs on the model's problem (built
  with `check_objects`) with `Problem::check_geometry_rules`: the options, the geometry and the
  surfaces; not the frequencies, the outputs or the model's own field points.

The warnings of the last model check are kept through the checks of the other scopes.

### Solve flow (`solver::solve`, `lo_*.rs`)

0. The problem is built from the enabled objects of the model (document_enabled.rs: a copy of the
   document without the disabled ones; the locations of its errors mapped back to the model), by
   document_build.rs: the rules between the objects,
   the items `mesh` and `ranges` resolved onto the elements of the meshes, the trim of each trimmed body and
   its surfaces, the generated control surfaces.
1. The checks before a solve (`Model::pre_solve`, model_solve.rs, the same as `Model::check`):
   the model rules, `select_route` (route.rs) — reject combinations without a solver, with
   the reason — and with the potential QTF the zones of its free surface; then the memory check
   (`options.memory_check`).
2. `build_geometry` — panels in one frame with kinds (wet / interior / lid /
   pressure / sheet / plate), the reflection group, sector representatives and image
   tables, wall image copies, character list.
3. `compute_static` — frequency-independent Rankine and log-singularity
   integrals for every sector row against every panel (parallel over rows).
4. Per frequency (`run_frequencies`, solve_driver.rs: the limits, then the
   ordinary frequencies on the threads; `solve_frequency`): `FreqCtx` ($K$, $k$, $\kappa$ of lids, finite-depth
   data) $\to$ pair kernels (`pair_kernel`, `apply_kernel`) $\to$ character matrices $\to$
   right-hand sides $\to$ LU per character $\to$ potentials $\to$ forces, Haskind, field
   points, lid elevations, motions; the source-formulation outputs
   (lo_velocity.rs).
5. `common_results` and the point outputs put the quantities into `Results`; the solve (`SolveJob::run`,
   model_solve.rs) then adds the momentum drift from the Kochin functions and the motions
   (`add_momentum_drift`, response.rs), drops the quantities computed only for others
   (`Options::hidden`, `solver::drop_hidden`: the drifts and QTFs the quadratic QTFs rest on, the
   QTFs of the frequencies not asked for) and adds the warnings of the rules.

The higher order follows the same steps in `ho_solver.rs` with the patch
geometry of `ho_geometry.rs` and its own frequency solve (`ho_frequency::solve_frequency`); the
loop over the frequencies and the common results are the same code (`run_frequencies`,
`common_results`, solve_driver.rs).

After the frequencies (the phase "second_order" begins only when every frequency is done) the
second-order loads (`quadratic_qtf.rs`, `potential_qtf/`) are formed from the values the
frequencies recorded, as steps of items on the same frequency workers: the rows of the quadratic
QTFs, the frequencies of the potential QTF's two sub-problems (theory §12.10, solved through
the same steps 1–5 with a board of their own whose workers are the model's) and its pair
frequencies; its fits and outer integral run on the pool of `frequency_threads × inner_threads`,
shared by all workers.

Profiling: set `HB_PROFILE` (any value) to print phase times on stderr. `HB_FIELD_STATICS_LIMIT`
(bytes, a test hook) replaces the 4 GiB above which the static rows of the field points are made
at evaluation instead of stored; read once per solve. `HB_AVAILABLE_MEMORY` (bytes, a test hook)
replaces the available physical memory that the memory check before a solve compares the estimate
with (`memory.rs`, `check_memory`). A value that is not a number refuses the solve; every hook set
is listed in the results' meta data (`environment_hooks`).

## The bindings (`crates/hydrobem-python/src/`, `crates/hydrobem-c/src/`)

Both bind the engine's public Rust API and hold no physics of their own (Table 3).

*Table 3. The modules of the bindings.*

| File | Role |
|---|---|
| `hydrobem-python/src/lib.rs` | the module `hydrobem._core` and its classes and functions |
| `model.rs`, `object.rs`, `results.rs` | `Model` (life cycle, queries, solve with progress and cancellation, the GIL released while waiting), `Object` (items, children), `Results` (quantities read when first asked) |
| `convert.rs` | NumPy arrays in and out: read-only views of the engine's vectors, the answers of the queries as dicts of arrays |
| `errors.rs` | an engine `Error` as the exception of its kind with its location; the diagnostics |
| `tables.rs` | the schema, the quantity table, the build, `wavenumber` |
| `hydrobem-c/src/lib.rs` | the C ABI `hydrobem.dll`; `include/hydrobem.h` is generated from it (`scripts/gen_c_header.py`) |
| `status.rs` | the status of a call, the error of the last failed call of a thread, the guard that keeps panics from crossing the boundary, `elements` (the checked element count of an array argument, at most $2^{40}$) |
| `model.rs`, `object.rs`, `mesh.rs`, `query.rs`, `results.rs` | the model and its solve, the objects and items, the meshes read, made and written, the queries, the results (`HbResults`, views valid until the results change) |
| `table.rs` | `HbTable`: the answer of a query as named typed columns |
| `info.rs` | what needs no model: the ABI version, the build, the schema, the quantity table, the codes |

## Python package (`python/hydrobem/`)

*Table 4. The modules of the Python package.*

| File | Role |
|---|---|
| `__init__.py` | the package: the names of `hb` (`Model`, `Results`, `load_results`, the queries, `ranges`, the errors) and `__version__` |
| `engine.py` | `hb.Model`: the engine's model, the queries of the model (everything at once), `warnings` (of the last check), the solve and its progress (`Progress`), the results file |
| `objects.py` | the objects of a model: `ModelObject` (items as attributes, lists of children, `update`, `info`, `items`, the printed form with its `unused:` line), `ObjectList` (named lists and lists with optional names: `add`, `insert`, `move`, `rename`), `BodyObject` (`hydrostatics`, `panels`, `mesh_quality`, `tessellation`, `trimmed_mesh`), the meshes `MeshListBase` and `MeshObject` (read, made, written by the engine, `load_fdf` among them; `geometry`, `elements`, `tessellation`, `quality`, `patch_points`) |
| `_classes.py` | generated by `scripts/gen_model_classes.py` from the schema (do not edit): the class of every object type (a property with the description of every item and list, an `update` with a keyword per item), the class of every list (`BodyList`, `MeshList`, ...: an `add` with a keyword per item, each described in its Args), and the typed objects and lists of `hb.Model` |
| `_core` (extension module, `crates/hydrobem-python`) | the engine's `Model`, `Object` (its items) and `Results` (with `selection` and `body_columns`); the answers of the panel, tessellation and quality queries as dicts of arrays (`convert.rs`); the schema, the quantity table, the result tree (`result_views`, `result_groups`), the build, `wavenumber` |
| `_cpu.py` | the check that the processor and the system run AVX2 and FMA, asking the system (Windows: `IsProcessorFeaturePresent`, AVX2; elsewhere the `avx2` and `fma` flags of `/proc/cpuinfo`), before the extension module loads |
| `_result_views.py` | generated by `scripts/gen_result_views.py` from the engine's result tree (do not edit): every public class of the results, each with a property per view (the engine's selection of an owner's part, `Results._select`) and per child group: a class per group (`DriftResults`, `QtfResults` and its parts, `KochinResults`, `HydrostaticsResults`, `ExcitationResults`, `BodyPointResults`, `FieldPointResults`, `LidResults`, `PressureSurfaceResults`, `TankResults`), `BodyResults` and `Results` (the axes, the quantities read by body, the groups, `bodies`), each on its hand-written base of `_result_extras.py` (the map `EXTRAS` of the script) or on `_View` |
| `_result_extras.py` | what the classes of the results have beyond the tree (it does not import `_result_views`): the bases `_View` and `_OwnerView`, `_OwnerList`, `Coordinate`, `_ResultsExtras` (the engine's results and the cached arrays, `quantity(name)`, `_select`, the owner lists, the model solved, `meta`, `route`, `warnings`, `diagnostics`, `available`, `coordinates`, `coordinate` and `coefficient` by name, `normalized`, `summary`, `save` (.hbr), `write_wamit`, `Results.of`), `_BodyExtras` (`index`, `name`, `coordinates`, `hydrostatics`) |
| `results.py` | the public names of the results (`Results`, `BodyResults`, the views of the groups, `Coordinate`) and `load_results` |
| `queries.py` | the answers of the queries as read-only classes (`Hydrostatics`, `MeshGeometry`, `MeshElements`, `PatchPoints`, `Counts`, `MemoryEstimate`, `MotionEquation`, `PointClasses`, `FreeSurfacePoints`, `Panels`, `Tessellation`, `MeshQuality`, `BuildInfo`, `QuantityInfo`) and the queries without a model: `wavenumber`, `quantities`, `build_info` |
| `selection.py` | `hb.ranges`: the item `ranges` of a surface from a mask or element numbers |
| `errors.py` | the exceptions of the engine status codes (with `location`, an `hb.Location`), `hb.Diagnostic` and `HydroBEMWarning` |

The engine modules carry Rust documentation comments (`//!`, `///`); `cargo doc -p hydrobem_core`
writes `target/doc`.

Every public class and function has a docstring (NumPy style; the generated `add` and `update`
of `_classes.py` describe their keywords in a Google-style Args section, which editors show on a
keyword) for IDE hovers; the items of the model carry theirs in the schema
(`ModelObject.info`). Hydrostatics are a solve with `first_order.radiation` and
`first_order.diffraction` false (route `hydrostatics`, no frequencies), or
`body.hydrostatics()`; the item `radii_of_gyration` forms the inertia from the mass, a
number or `"displaced"`, which it needs.
The objects of a list are made by `add` / `insert` only (a list is not assigned); `clear()` empties
a list in the engine, all or nothing (`Model::clear_list`).

## Tables and reference data

Every table and reference file is produced by a script; regenerate instead of
editing, and give the reason in the commit and in `CHANGELOG.md` (Table 5).

*Table 5. The generated tables and reference data and their scripts.*

| Output | Script | Notes |
|---|---|---|
| `src/tables/green_fin_tables.rs` | `scripts/gen_green_finite_tables.py` | error map `docs/development/error-maps/green_fin_error_map.md`; bit-reproducible, `--check` by hand (about 7 min, not in the tests) |
| `src/tables/green_inf_tables.rs` | `scripts/gen_green_tables.py` | error map `docs/development/error-maps/green_inf_error_map.md`; checksum verified at run time |
| the Rust form of both tables | `scripts/green_tables_rust.py` | the writer shared by both generators |
| `tests/data/green_inf_reference.txt` | `scripts/gen_green_reference_data.py` | mpmath, `green_reference.py` |
| `tests/data/green_fin_reference.txt` | `scripts/gen_green_finite_reference_data.py` | mpmath, `green_finite_reference.py` |
| `tests/data/channel_reference.txt` | `scripts/gen_channel_reference.py` | mpmath, Faddeeva and lattice sums |
| `tests/data/dispersion_reference.txt` | `scripts/gen_dispersion_reference.py` | 40-digit roots |
| `tests/data/panel_reference.txt` | `scripts/gen_panel_reference.py` | panel integrals |
| `tests/data/cylinder_finite_depth.txt` | `scripts/gen_cylinder_reference.py` | eigenfunction matching (A5) |
| `tests/data/cylinder_array.txt` | `scripts/gen_cylinder_array_reference.py` | two and three truncated cylinders in finite depth: eigenfunction matching with multiple scattering (35 s; not in the checks) |
| `tests/data/two_spheres.txt` | `scripts/gen_sphere_array_reference.py` | two submerged spheres in deep water: multipoles (`--check` in the tests) |
| `tests/data/hulme1982_full.txt`, `tests/data/hulme1982_limits.txt` | `scripts/extract_hulme.py` | tables 1, 2 and the limits of the added masses at $\omega = 0$ and $\infty$; needs a local copy of the paper (not in the repository) |
| `tests/data/kim_yue1990.txt` | `scripts/extract_kim_yue.py` | table 3 of Kim & Yue (1990); needs a local text conversion of the paper (not in the repository) |
| `src/tables/log_gauss_table.rs` | `scripts/gen_log_gauss.py` | the 12-point rule exact for $p^k$ and $p^k \log p$ (theory §7.1; `--check` in the tests) |
| the Gauss tiers of the panel integrals | `scripts/panel_rule_error_map.py` | error map `docs/development/error-maps/panel_rule_error_map.md` (about 2 min) |
| the log-correction bound 0.6 | `scripts/log_correction_bound.py` | the bound of the analytic log correction against the panel size |
| the quadrature of the finite-depth tables | `scripts/green_finite_quad.py` | the verified double-precision quadrature of the tabulated functions |
| `tests/data/regression/*.npz` | `scripts/gen_regression_baselines.py` | HydroBEM's own results of `regression_problems.py` |
| `python/hydrobem/_classes.py`, the tree of the inputs in `docs/reference/inputs-model.md` | `scripts/gen_model_classes.py` | the classes of the model's objects from the schema (`--check` in the tests) |
| `python/hydrobem/_result_views.py`, the tree and the table of the views in `docs/reference/results.md` | `scripts/gen_result_views.py` | the views of the results from the engine's result tree (`--check` in the tests) |
| `crates/hydrobem-c/include/hydrobem.h` | `scripts/gen_c_header.py` | the C header from the sources of `crates/hydrobem-c` (`--check` in the tests) |
| the expected strings of `REPR_CASES` in `crates/hydrobem/tests/io.rs` | `scripts/gen_repr_cases.py` | Python's repr and the Fortran fields of the bit patterns (`--check` in the tests) |
| `licenses/rust-crates.txt` | `scripts/gen_crate_licenses.py` | the licences of the crates of `Cargo.lock` (`--check` in the tests) |
| `docs/theory/figures/*.png` | `scripts/theory_figures.py` | the sketches of the theory chapters, one function per figure (matplotlib; not in the checks) |

## Documents (`docs/`)

The documentation by reader (the folders and the citation forms: [rules.md](rules.md); the reading
paths: [docs/README.md](../README.md)): `guide/` the user guide by topic, `theory/` the numbered
chapters (`01-boundary-value-problem.md` ... `14-geometry-preprocessing.md`, the index
`README.md`), `reference/` the inputs by object (`inputs-model.md`, `inputs-meshes.md`,
`inputs-bodies.md`, `inputs-model-objects.md`, `inputs-orders.md`, `inputs-rules.md`), the
queries, the results, the output files and the APIs, `validation/` the test plan and the report,
`development/` these documents and the error maps (`error-maps/`), `references.md` the
bibliography. Generated parts: the tree of the inputs (`reference/inputs-model.md`), the result
tree and views (`reference/results.md`), the error maps (Table 5). `scripts/rewrite_doc_refs.py`
rewrites the citations of the documentation of before 2026-11-12 and, with `--check` (in
`scripts/test.bat`), refuses them and checks every relative link of the published documents.
The sketches of the theory chapters (`theory/figures/*.png`) are drawn by
`scripts/theory_figures.py`, one function per figure named as its file and prefixed by its chapter
(`python scripts/theory_figures.py [prefix ...]`).

## Tests

- `crates/hydrobem/tests/core.rs` with its modules `io.rs`, `document.rs`, `problem.rs`, `prep.rs`,
  `kernels.rs`, `lo_a.rs`, `lo_b.rs`, `ho_a.rs`, `ho_b.rs`, and `model_api.rs` (the public API)
  (`cargo test --release`): accuracy of the components (test-plan A1, A2, A3, tiles, sheet and
  channel integrals; `kernels.rs` the Chebyshev bases, the direct finite-depth kernel at
  coincidence, the panel images and the lattice forms), the readers and
  writers, the model, the preprocessing and both methods; the environment variable `HB_TIMING`
  adds the kernel timings. `crates/hydrobem/tests/pools.rs`, a test binary of its own (the
  global rayon pool is process-wide): the LU runs on the pool of its caller, its solution
  independent of the thread count. Unit tests (`#[cfg(test)]`) in the modules, among them the LU
  of `linalg.rs`.
- `crates/hydrobem-c/tests/c_api.rs`: the C ABI through its exported functions (a model through it
  solves to the bits of the Rust API; tables, errors, the progress callback and its cancellation,
  files, results of arrays); `tests/python/test_c_abi.py` compiles `crates/hydrobem-c/tests/c_api.c`
  and the header (as C and as C++, MSVC) and runs the program on the DLL.
- `tests/python/test_<feature>.py`: identities, input rules and published
  references per feature (test-plan A, B): `test_low_order.py`,
  `test_finite_depth_and_limits.py`, `test_walls_and_interior_surfaces.py`,
  `test_damping_lids.py`, `test_thin_sheets.py`, `test_pressure_surfaces.py`, `test_free_surface_plates.py`,
  `test_generated_plates.py`, `test_tank_free_surface_plates.py`,
  `test_channel.py`, `test_higher_order.py`, `test_drift.py`, `test_field.py`,
  `test_dipole_dampers.py`, `test_trimmed_waterline.py`, `test_generalized_modes.py`,
  `test_analytic_geometry.py`, `test_internal_tanks.py`, `test_multiple_bodies.py`, `test_control_surface_drift.py`,
  `test_wavemakers.py`, `test_mesh_quality.py`, `test_mode_displacements.py`, `test_motions.py`,
  `test_rational_patches.py`, `test_higher_order_other_meshes.py`, `test_c_abi.py`; the
  second-order loads in `test_quadratic_qtf.py`, `test_potential_qtf.py`,
  `test_second_order_refusals.py`, `test_second_order_rotation_center.py`;
  inputs, files, queries and the extension module in `test_rules.py`, `test_mesh_references.py`,
  `test_readers.py`, `test_writers.py`, `test_geometry_files.py`, `test_extension_module.py`,
  `test_inspection.py`,
  `test_diagnostics.py`, `test_double_precision_mode.py`,
  `test_model_files.py`, `test_progress.py`, `test_memory_estimate.py`, `test_engine_info.py`,
  `test_classify_points.py`, `test_model_document.py`, `test_results_file.py`, `test_model_api.py`,
  `test_queries_by_owner.py`, `test_results_by_owner.py`, `test_body_references.py`, `test_enabled.py`,
  `test_model_meshes.py`, `test_model_classes.py`, `test_result_views.py`, `test_example_references.py`; the examples run in
  `test_examples.py`.
- `tests/python/test_regression.py`: every output of the small problems of
  `regression_problems.py` against HydroBEM's stored results
  (`tests/data/regression/`, 1e-5 of the largest value of each entry over the first axis, at
  least 1e-3 of the peak, 1e-14 absolute for an entry zero up to round-off, with the same dtype), and the thread counts byte-identical for the
  problems with a channel, drift and tanks and three of the second order.
- Helpers: `meshes.py` (vertex generators), `bodies.py` (meshes and bodies as dicts of the
  model's items: `surface`, `flat`, `spline`, `part`, and the shared bodies `hemi`,
  `coarse_cylinder`, `slow_cylinder`, `lo_cosine_cylinder`, `named_tank_vessel` with
  `TANK_POINTS`, `heave_pieces`), `problems.py` (`model(...)` builds an `hb.Model` from them,
  `solve`, `peak`, `view` (a view of results by its quantity name), `hydrostatics`, `two_tank_vessels`, `load_script`, `close`, `hulme` (rows of
  `hulme1982_full.txt`)), `regression_problems.py` (the regression problems and `same_outputs`);
  test files do not import each other. Tests of private functions are in-module
  (`#[cfg(test)]`).
- Tests that take seconds to minutes are marked `slow`:
  `scripts/test.* fast` leaves them out (about a minute), `full` runs all.

## Validation report (`validation/`)

The scripts of the validation report ([validation/README.md](../validation/README.md), one page per
case in `docs/validation/cases/`), each a group, run by
`python validation/make_report.py [--quick | --pages] [group ...]` (the figures, the mesh figures and
the numbers of the tables into `docs/validation/figures/`, `numbers.json`, then every page and the
index from the numbers); they use the helpers of `tests/python/`. Each script of a case defines its
page (`page`, `summary`, `INDEX`; `qtf.py` two pages in `PAGES`; Table 6).

*Table 6. The scripts of the validation report and what they compute.*

| Script | What it computes (page) |
|---|---|
| `common.py` | paths, figure style, the store of the numbers, the helpers of the pages, the observed order and Richardson's extrapolation, a function in a process of its own with validation hooks |
| `mesh_figure.py` | the mesh figures: the surfaces of a model by role from `Model.panels` and `Model.tessellation` joined by their keys, with the reflections, the spans of the higher order |
| `hemisphere.py` | the hemisphere against Hulme (1982), low and higher order; their mesh convergence; the irregular frequencies with and without interior surfaces (`floating_hemisphere`) |
| `cylinder.py` | the truncated cylinder in finite depth against the eigenfunction-matching reference of `scripts/gen_cylinder_reference.py`, low and higher order; mesh convergence and the reference's truncation (`truncated_cylinder_finite_depth`) |
| `bottom_cylinder.py` | the bottom-mounted cylinder: the exciting force against MacCamy & Fuchs (1954), the mean drift against the pressure integration of the analytic series (Havelock 1940), low and higher order (`bottom_mounted_cylinder`) |
| `kernels.py` | the error maps of the Green functions at the A2 reference points (the example `crates/hydrobem/examples/green_error_map.rs`; `green_functions`) |
| `qtf.py` | the potential QTF against Kim & Yue (1990) (`second_order_bottom_cylinder`), Newman's approximation against the full difference QTF, the convergence of the potential QTF in its free-surface parameters, the Kochin angle count (`cylinder`) |
| `constants.py` | the sensitivity of the fixed constants (inputs, and the validation hooks of `env_hooks.rs`; `constants`) |
| `two_truncated_cylinders.py`, `three_truncated_cylinders.py` | groups of cylinders against the multiple-scattering reference of `scripts/gen_cylinder_array_reference.py` (over the whole range) and the values read from Kagemoto & Yue (1986) and Mavrakos (1991) |
| `two_submerged_spheres.py` | two submerged spheres against the multipoles of `scripts/gen_sphere_array_reference.py` and the values read from Wu (1995) |
| `floating_owc_piston.py` | the two-body OWC: the paper's panels, fine panels and the higher order against each other |
| `floating_owc_chamber.py` | the OWC chamber with a turbine against Tables 1, 2, 5 of Mavrakos & Konispoliatis (2012) and the capture width of Evans (1982) |
| `cylinder_with_heave_plate.py` | the cylinder with a thick and a thin heave plate against the values read from Han et al. (2024) |
| `fish_tank_cylinders.py` | the fluid in a circular tank against the interior series of Faltinsen & Timokha (2009) |
| `dock.py`, `floating_owc_cover.py` | free-surface plates: the disc against eigenfunction matching and Rayleigh's piston; a cover of a moonpool by identities and the methods against each other |
| `make_report.py` | the driver: every group, the run times, the pages and the index |

## Conventions (used everywhere)

- Time factor $e^{+i\omega t}$; $z$ up, free surface $z = 0$, bottom $z = -h$; $K = \omega^2/g$.
- Panel normals point **out of the body into the fluid**; interior surfaces
  and lids have upward normals and lie exactly in $z = 0$.
- Body equation $2\pi\phi - \int \phi \, \partial G/\partial n = -\int G \, \partial\phi/\partial n$; free terms: $+2\pi$ body, $-4\pi$
  interior surface, $+4\pi$ lid.
- $A - iB/\omega = -\rho \int n_i \phi_j \,\mathrm{d}S$; $X = i\omega\rho \int n_i \phi_D \,\mathrm{d}S$.
- First-order results in body axes about each body's rotation centre, SI units, unit wave
  amplitude; the mean drift moments as theory §1.1.
- In the engine's problem (`problem.rs`) the codes of the options are enums of `model_codes.rs`,
  the mesh of a surface or of the free surface that may be absent is `Option<usize>` (None: the
  elements of the body's mesh, or none), and the values left to the engine ("auto": the zones of
  the potential QTF, a tank's surface level, a body's displaced mass) are `None` or an enum.

## Adding a feature

1. Extend the inputs only where the reference of the inputs (`reference/inputs-*.md`) requires it: the schema of the model
   (`document_schema.rs`, which Python reads and the model file keeps; every item with a description),
   the problem built from it (`document_build.rs`, `problem.rs`, `Problem`), and the public API
   (`model.rs`, the extension module) with a new API version if it changes; then `scripts/gen_model_classes.py` for the Python classes.
2. Implement the physics in the engine; keep `select_route` rejecting what is
   not available. New surface kinds follow the pattern of interior surfaces and
   lids (a `kind` in `build_geometry`, a free term and a column rule in the
   assembly, outputs in `solve`).
3. Describe each method choice in its chapter of `docs/theory/` (the usage in `docs/guide/`, the inputs in
   `docs/reference/`): the method,
   its parameters, its derivation and accuracy, with the sources in `docs/references.md`;
   cite the section in the code.
4. Add tests: component accuracy in `crates/hydrobem/tests/`, identities and published results in
   `tests/python/`, and a problem in `tests/python/regression_problems.py`
   with its baseline (`scripts/gen_regression_baselines.py`).
5. Record user-visible changes in `CHANGELOG.md`.
