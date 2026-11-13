# The C ABI of the engine

The engine is also a C library: `hydrobem.dll` with its import library `hydrobem.dll.lib` and the
header `hydrobem.h`, for any program that calls C functions. It binds the public Rust API of the
engine ([rust-api.md](rust-api.md)) as the Python package does: the same model, objects, items,
meshes (also the panels of an FDF file), queries (also the free surface of the potential QTF),
solve and results, the same checks and messages. The header declares every
function with its contract; this page gives the conventions.

## Building it

The C ABI is the crate `crates/hydrobem-c`; HydroBEM distributes the Python wheel, and the DLL is
built from the sources with Rust and the Visual Studio Build Tools (the requirements of the
[user guide, Installation](../guide/getting-started.md#1-installation); Python is not needed for the DLL):

```bat
cargo build --release -p hydrobem-c
```

*Table 1. The files of the build.*

| File | What it is |
|---|---|
| `target\release\hydrobem.dll` | the engine; x86-64 with AVX2 and FMA; it needs the Visual C++ runtime (`vcruntime140.dll`, installed with Visual Studio or the Visual C++ Redistributable, and by Python) |
| `target\release\hydrobem.dll.lib` | the import library to link |
| `crates\hydrobem-c\include\hydrobem.h` | the header (C and C++) |

A program includes the header, links the import library and finds the DLL beside it (or on the
search path); a program that ships the DLL ships the licences with it (`LICENSE`, `NOTICE`,
`THIRD_PARTY_NOTICES.md`, `licenses/`). `scripts\build.bat` builds the DLL with the rest of the
workspace. The header is written from the sources of `crates/hydrobem-c` by
`scripts/gen_c_header.py` (never edited by hand; `scripts\test.bat` checks it).

The engine is compiled for x86-64 with AVX2 and FMA (`-C target-feature=+avx2,+fma` in
`.cargo/config.toml`): a build without them stops with a compile error, and a `RUSTFLAGS` of its
own replaces the flags of `.cargo/config.toml` and must name them too. On a processor (or a
system) without AVX2 and FMA every function that returns a status, but `hb_last_error_location`,
returns `HB_ERR_UNSUPPORTED` with a message that says so, instead of stopping at an illegal
instruction (the processor is read once, at the start of the first such call). The check is a best
effort: it is compiled with the same instructions as the rest of the library, so it is reliable on
a processor with AVX but without AVX2 or FMA, while on a processor without AVX at all the check
itself may stop at an illegal instruction. (The Python package checks for AVX2 and FMA at import,
asking the system.)

`hb_abi_version()` returns the version of the ABI of the DLL, raised with every change of a
function, a struct or a constant; a program compares it with `HB_ABI_VERSION` of the header it was
built with. It is 8 for HydroBEM 0.3.0.

## Conventions

- **Status.** Every function that can fail returns an `HbStatus`: `HB_OK` (0) or an `HB_ERR_`
  code, one per kind of error (`HB_ERR_VALUE`, `HB_ERR_TYPE`, `HB_ERR_UNSUPPORTED`,
  `HB_ERR_STATE`, `HB_ERR_CANCELLED`, `HB_ERR_INTERNAL`, `HB_ERR_ARGUMENT`, `HB_ERR_NUMERICAL`,
  `HB_ERR_IO`, `HB_ERR_EXISTS`; `HB_ERR_UNSUPPORTED` also for a processor without AVX2 and FMA).
  The message of the last failed call of a thread is
  `hb_last_error_message()` (UTF-8, valid until the next failure on that thread), its place
  `hb_last_error_location()` (an `HbLocation`: body, mesh, element, field point, lid, frequency;
  -1 where it names none). An argument the library itself refuses (a null pointer, a text that is
  not UTF-8, a column that does not exist) is `HB_ERR_ARGUMENT`; a panic of the engine is caught
  at the boundary and returned as `HB_ERR_INTERNAL`. `hb_last_error_location(NULL)` returns
  `HB_ERR_ARGUMENT` and keeps the last error. A function checks its arguments, its out-pointers
  first, before it changes the model or allocates: a refused call changes nothing and hands out
  nothing. The out-pointer of `hb_model_check` and `hb_model_solve` may be null (no table, no results).
  `hb_model_check` applies the refusals of the solver routes too: a valid model without a route is
  `HB_ERR_UNSUPPORTED`, as from `hb_model_route` and `hb_model_solve`, invalid input `HB_ERR_VALUE`.
- **Texts and paths** are NUL-terminated UTF-8, in and out.
- **Ownership.** What the library hands out the caller frees: a model with `hb_model_free`,
  results with `hb_results_free`, a table with `hb_table_free`, a text with `hb_string_free`.
  Everything passed in is read during the call and not kept.
- **Objects** of a model are named by `uint64_t` ids (`hb_object(model, "bodies/hull", &id)`;
  `""` the model itself); an id stays valid while its object is in its model, and the model
  refuses the id of a removed object. `HB_NO_OBJECT` names none. `hb_add` puts a new object at a
  zero-based place of its list or at its end (-1), `hb_move` moves one to a zero-based place
  (`int64_t`); a negative place or one past the last object is `HB_ERR_ARGUMENT`, its message
  naming the place and the number of objects of the list. The order of a list is the order of
  its objects in the results and the files.
- **Items** are set by their kind (`hb_set_bool`, `hb_set_int`, `hb_set_real`, `hb_set_text`,
  `hb_set_texts`, `hb_set_reals`, `hb_set_ints`) and reset by `hb_reset`; the
  engine checks every value where it is set. The mesh of a surface is the text item `mesh`
  (`hb_set_text`), its elements the integer item `ranges` of shape (n, 2), the one-based first and
  last stored elements of each range (`hb_set_ints`; reset: every element). `hb_get` returns the
  value as a table: its kind and the column `value`. The object types, their items (kind,
  unit, default, choices, description, and for a switched item `switch`, the switch items, and
  `switch_values`, the values that take it: inputs, introduction) and lists are the tables of
  `hb_schema_items` and `hb_schema_lists`, the same schema as [inputs-model.md](inputs-model.md). The
  surfaces chosen by a switch (a body's `interior_surface` and `control_surface`, the free surface
  of `second_order`) are flat items set by `hb_set_*` as any other; the items of a value not
  chosen are kept unused. `hb_clear(model, object, list)` empties a list of an object, all or
  nothing: it is refused (`HB_ERR_VALUE`, nothing removed) while another object refers to one of
  its objects, and a list that always holds its object (`environment`, ...) is not emptied. An array argument of more than
  $2^{40}$ elements (the product of its shape), or whose shape overflows, is `HB_ERR_ARGUMENT` before
  it is read.
- **Tables.** An answer is an `HbTable` of named columns; `hb_table_count`, `hb_table_column`
  and `hb_table_find` give an `HbColumn`: name, element type (`HB_F64`, `HB_F32`, `HB_I32`,
  `HB_I64`, `HB_U64`, `HB_C64` (complex of two float32), `HB_C128` (of two float64), `HB_TEXT`),
  dimensions, number of elements and the
  elements in C order (a complex element is its real and imaginary parts; a text column an array
  of strings). Its pointers are valid until the table is freed.
- **Result quantities** (`hb_results_quantity`, by the names of `hb_quantities` and
  [results.md](results.md)) are views of the engine's elements, not copies:
  valid until the results are freed or changed (`hb_results_put`); a quantity's name is its path
  in the result tree with "_" for "." (`excitation_diffraction`, `lids_elevation_radiation`; the
  points of a group the group's name, `body_points`, `field_points`, `lids`, `tanks`). The part of
  an owner (reference/results §1: a body, a lid, a pressure surface or a tank of a body) is a
  copy in a table, `hb_results_select(results, path, body, owner, &table)` (the path of the view,
  `"excitation.diffraction"`, `"lids.points"`, `"tanks.pressure_radiation"`; -1 for no body or
  owner; the column `values`, or none when it was not computed); `hb_result_views` and
  `hb_result_groups` list the paths of the result tree. `hb_results_select` is the copying form of the Rust
  `Results::selection`; `hb_results_selection` (the same arguments) says where that part lies in
  the whole quantity: the column `quantity`, and one row per index taken the columns `axis` and
  `index` (zero-based; the axes not listed taken whole), as Python's `Results.selection`;
  `hb_results_body_columns(results, body,
  &table)` (`int64_t body`, the zero-based body of the model solved; a negative body or one beyond
  the bodies solved is `HB_ERR_ARGUMENT`, as in `hb_results_select`) gives the zero-based columns
  of a body (its coordinates on the D axis, its radiation columns on the K axis), the rows the
  selection takes of it. `hb_results_meta` gives the meta data of the solve (`present` 0 for
  results not from a solve): the version and build, the time, the route, the threads, the engine
  seconds and `environment_hooks`, the texts of the environment hooks set for the solve (for
  validation studies and tests only; none without).
- **Results made from arrays** (`hb_model_new_results`, then `hb_results_put` per quantity; the axes and
  the points of the field outputs, `field_points`, `tanks`, `tanks_ids`, are the model's and are
  not put): `hb_results_write_wamit` checks the quantities given together (reference/results)
  before it writes; `hb_results_put` checks each quantity's shape, and `hb_results_check(results)`,
  called once all are put, the quantities that are given together (`HB_ERR_VALUE` naming the one
  missing), as Python's `Model.new_results` does after its quantities.
- **Threads.** A model may be used from several threads: each call locks it. `hb_model_solve` runs on
  the calling thread and releases the lock meanwhile: other threads may read the model and cancel
  the solve (`hb_model_cancel`, which never waits); changes, queries and `hb_model_free` are
  refused with `HB_ERR_STATE` until the solve ends. `hb_model_free` must not run concurrently
  with any other call on the model, nor may a call begin before it returns. The progress callback receives an
  `HbProgress` and the caller's pointer, one call at a time, maybe on another thread than the
  solve's; returning 0 cancels the solve (`HB_ERR_CANCELLED`, the model reset). A report gives
  its stage (`HB_STAGE_`: the stages of a frequency, `HB_STAGE_SECOND_ORDER` for the parts of a
  second-order item, `HB_STAGE_PHASE` for a change of the phase, always reported,
  `HB_STAGE_WORKER_IDLE` for a worker with no item left in its phase), the frequency (-1 a limit,
  -2 the statics, -3 a second-order item, -4 the solve as a whole), the frequency worker that
  reports (`worker`, 0 to frequency_threads - 1, fixed for the solve; -1 the work all threads
  share), the phase (`HB_PHASE_`) and for a second-order item its step (`HB_STEP_`) and index
  (`item`; -1 none).
  `hb_model_progress` gives the state of the running solve, or of the last one, at any instant,
  from any thread and without waiting: the phase, the preparation, per limit and per ordinary
  frequency (in the order of the results) its state, stage, fraction done and worker, per
  frequency worker its state (`running`, `idle`, `done`) and the item it is on, and per step of
  the second order its state, fraction and items, as the user guide describes it.
  An `HbResults` may be read from several threads at once, but not while `hb_results_put` changes
  it; the views of its quantities stay valid until the results are freed or a put succeeds.

## Functions

*Table 2. The functions of the C ABI by group.*

| Group | Functions |
|---|---|
| Library | `hb_abi_version`, `hb_build_info`, `hb_schema_items`, `hb_schema_lists`, `hb_quantities`, `hb_result_views`, `hb_result_groups`, `hb_panel_roles`, `hb_wavenumber`, `hb_last_error_message`, `hb_last_error_location`, `hb_string_free` |
| Tables | `hb_table_count`, `hb_table_column`, `hb_table_find`, `hb_table_free` |
| Model | `hb_model_new`, `hb_model_load`, `hb_model_load_results`, `hb_model_save`, `hb_model_save_results`, `hb_model_copy`, `hb_model_free`, `hb_model_state`, `hb_model_cancel`, `hb_model_progress`, `hb_model_reset`, `hb_model_clear`, `hb_model_check`, `hb_model_diagnostics`, `hb_model_route`, `hb_model_solve`, `hb_model_results`, `hb_model_new_results` |
| Objects and items | `hb_object`, `hb_object_info`, `hb_children`, `hb_add`, `hb_remove`, `hb_rename`, `hb_move`, `hb_clear`, `hb_item_info`, `hb_get`, `hb_set_*`, `hb_reset` |
| Meshes | `hb_model_load_mesh` (GDF, GDF with SPL, the geometry file `.hbg`; `HbMeshOptions`), `hb_model_load_csf`, `hb_model_load_fdf`, `hb_model_mesh_from_hbg`, `hb_mesh_hbg`, `hb_model_analytic_mesh` (`HbAnalyticShape`), `hb_body_trimmed_mesh`, `hb_mesh_write` |
| Queries | of the model `hb_model_counts`, `hb_model_motion_equation`, `hb_model_panels`, `hb_model_tessellation`, `hb_model_mesh_quality`, `hb_model_mode_displacements`, `hb_model_memory_estimate`, `hb_model_classify_points`, `hb_model_free_surface_points`; of a body `hb_body_hydrostatics`, `hb_body_panels`, `hb_body_mesh_quality`, `hb_body_tessellation`; of a mesh alone `hb_mesh_geometry`, `hb_mesh_elements`, `hb_mesh_tessellation`, `hb_mesh_geometry_quality`, `hb_mesh_patch_points` |
| Results | `hb_results_quantity`, `hb_results_select`, `hb_results_selection`, `hb_results_body_columns`, `hb_results_has`, `hb_results_available`, `hb_results_route`, `hb_results_diagnostics`, `hb_results_meta`, `hb_results_model`, `hb_results_solved_model`, `hb_results_normalized`, `hb_results_save`, `hb_results_write_wamit`, `hb_results_put`, `hb_results_check`, `hb_results_free` |

The header declares 95 functions. A function's prefix names what it answers for or acts on:
`hb_model_` the model, `hb_body_` a body (by its object id), `hb_mesh_` a mesh, `hb_results_`
results; the functions of the library, of the tables and of objects and items (`hb_object`,
`hb_add`, `hb_get`, `hb_set_*`, ...: they act on any object named by its id) keep their own. The
names follow the Rust API, with a few binding idioms: the motion matrices are the table of
`hb_model_motion_equation` with its column `ready` (0 when the model cannot form them; Rust
`MotionEquation::ready`, Python `m.motion_equation().ready`), the trimmed surface of a role is
`hb_body_trimmed_mesh` (Python `body.trimmed_mesh`; its `role` one of `"wetted"`, `"dipole"`,
`"dipole_damper"`, `"tank"`, `"pressure_surface"`, `"free_surface_plate"`,
`"interior_surface"`), and the date of the numeric files is the argument `stamp` of
`hb_results_write_wamit` (Python `timestamp=`).

The geometry queries (reference/queries §1–§3):

- `hb_model_panels(model, role, include_reflections, &table)`,
  `hb_body_panels(model, body, role, include_reflections, &table)`,
  `hb_model_tessellation(model, role, divisions_u, divisions_v, include_reflections, &table)`,
  `hb_body_tessellation(model, body, role, divisions_u, divisions_v, include_reflections, &table)`
  and `hb_mesh_tessellation(model, mesh, divisions_u, divisions_v, include_reflections, &table)`.
  `role` is one role of reference/queries Table 2 (`"wetted"`, `"dipole"`, `"lid"`,
  `"fixed_damper"`, `"free_surface"`, ...), several separated by commas (`"wetted,lid"`), or NULL
  (every role but the control surface and the free surface); an unknown role, or one a body never
  has (`"fixed_damper"`, `"free_surface"`), is `HB_ERR_VALUE`. `include_reflections` is 0 or 1.
  `divisions_u`, `divisions_v` are the quads per patch in u and v (1..256); both 0 is one quad per
  span of the potential discretization of each patch. The tables of the panels and the
  tessellation have the seven key columns `body`, `role` (the code of `hb_panel_roles`; -1 for a
  mesh alone), `tank`, `owner`, `element` (1-based), `part` and `reflection` [n, 2], with the same
  values for a quad as for the panels row of its element: a quad is joined to its panel by these
  columns. The panels have also `vertices`, `distinct_vertices`, `centroids`, `normals` and
  `areas`, the tessellation `vertices` and `normals` [T, 4, 3].
- `hb_mesh_elements` gives per stored element `element`, `centroids`, `normals`, `areas`,
  `vertices` and `distinct_vertices`; `hb_mesh_patch_points(model, mesh, element, uv, n, &table)`
  the `points` and `normals` of the patch `element` (1-based) at `uv` [n, 2];
  `hb_model_mesh_quality`, `hb_body_mesh_quality` and `hb_mesh_geometry_quality` the columns of
  reference/queries Table 7 (`body`, `element`, `part`, `size`, ...).
- `hb_model_free_surface_points(model, zone, &table)`: the columns `points` and `weights` of a
  zone of the free surface of the potential QTF (Python `m.free_surface_points(zone)`); an unknown
  `zone` is `HB_ERR_VALUE`. It and `hb_model_panels`, `hb_model_tessellation` with the role
  `"free_surface"` alone check the rules of the free surface (reference/queries).
- `hb_model_classify_points(model, points, n, tank, &table)` classifies points in the sea (global
  coordinates) with `tank` 0 or `HB_NO_OBJECT`, or points inside a tank (body coordinates) with `tank` the object id
  of that tank: the columns `status`, `body`, `lid` and `tank` (0-based, -1 none; `tank` the tank
  of its body, as the results' tanks of a body, the `owner` of `hb_results_select`).
- `hb_model_mode_displacements(model, points, bodies, elements, n, &table)`: `bodies` 0-based,
  `elements` 1-based.
- The table of `hb_model_counts` has the columns of `hb.Counts` (`lid_elements`,
  `pressure_elements`, `per_lid_elements`, ...) and per body those of `hb.BodyCounts` (`stored`,
  `wetted`, `wetted_with_reflections`, `dipole`, `interior_surface`, `pressure_modes`,
  `symmetry_copies` (1, 2 or 4), `generalized_modes`; reference/queries Table 12).

The location columns of the tables of `hb_model_check` and `hb_model_diagnostics` are those of
`HbLocation`: `body`, `mesh`, `element`, `field_point`, `lid`, `frequency`. The symmetry of a mesh
(`HbAnalyticShape.symmetry`, `hb_model_load_fdf`) is `"none"`, `"yz"`, `"xz"` or `"xz_and_yz"`.

## From ABI 7

A program written against ABI 7 changes:

- the names: `hb_panels` becomes `hb_model_panels`, `hb_tessellation` becomes `hb_model_tessellation`,
  `hb_mesh_quality` becomes `hb_model_mesh_quality`, `hb_counts` becomes `hb_model_counts`,
  `hb_motion_equation` becomes `hb_model_motion_equation`, `hb_mode_displacements` becomes
  `hb_model_mode_displacements`, `hb_memory_estimate` becomes `hb_model_memory_estimate`,
  `hb_classify_points` becomes `hb_model_classify_points`, `hb_free_surface_points` becomes
  `hb_model_free_surface_points`, `hb_route` becomes `hb_model_route`, `hb_solve` becomes `hb_model_solve`,
  `hb_new_results` becomes `hb_model_new_results`, `hb_load_mesh` becomes `hb_model_load_mesh`,
  `hb_load_csf` becomes `hb_model_load_csf`, `hb_load_fdf` becomes `hb_model_load_fdf`, `hb_hbg_mesh` becomes
  `hb_model_mesh_from_hbg`, `hb_analytic_mesh` becomes `hb_model_analytic_mesh`,
  `hb_trimmed_body_mesh` becomes `hb_body_trimmed_mesh`, `hb_mesh_records` becomes `hb_mesh_elements`,
  `hb_patch_points` becomes `hb_mesh_patch_points`, `hb_write_mesh` becomes `hb_mesh_write`;
- the arguments: the tessellations of the model and of a body take a `role` (NULL: the default
  roles) before the divisions; `nu`, `nv` are `divisions_u`, `divisions_v` (both 0: the spans);
  `expand` is `include_reflections`; `records` of the mode displacements is `elements`;
- the roles: the same codes, their names those of reference/queries Table 2 (`"body"` or `"wet"`
  becomes `"wetted"`, `"dipoles"` or `"sheet"` becomes `"dipole"`, `"lids"` becomes `"lid"`, ...), the only names
  accepted; a dipole now has its owner and a tank's walls their tank as owner;
- the columns: `element` for `record` (also in `HbLocation` and the location columns of the check
  and diagnostics), `distinct_vertices` for `vertex_counts`, `body`, `lid` and `tank` of the point
  classes for `index`, the counts as above, and the new key columns (`role`, `tank`, `owner`,
  `part` of the tessellation, `part` of the panels and the mesh quality);
- the model check: `hb_model_check` may return `HB_ERR_UNSUPPORTED` (a valid model without a solver
  route), which it did not before; a program that took every status but `HB_OK` for invalid input
  tells the two apart;
- the check: `hb_abi_version() == 8`.

## Example

```c
#include <stdio.h>
#include "hydrobem.h"

int main(void) {
    HbModel *m = NULL;
    uint64_t environment;
    HbResults *r = NULL;
    HbColumn a;
    if (hb_model_load("barge.hbm", &m) != HB_OK) goto fail;
    const double periods[] = {5.0, 8.0, 12.0};
    size_t n = 3;
    if (hb_object(m, "environment", &environment) != HB_OK) goto fail;
    if (hb_set_reals(m, environment, "periods", periods, &n, 1) != HB_OK) goto fail;
    /* the periods are read, not the omegas (both are kept) */
    if (hb_set_text(m, environment, "frequency_input", "periods") != HB_OK) goto fail;
    if (hb_model_solve(m, NULL, NULL, &r) != HB_OK) goto fail;
    if (hb_results_quantity(r, "added_mass", &a) != HB_OK) goto fail;   /* float [F, D, K] */
    /* (2, 2): heave and heave when all six rigid modes are radiation columns */
    printf("A33 at 5 s: %g\n", ((const float *)a.data)[2 * a.shape[2] + 2]);
    hb_results_free(r);
    hb_model_free(m);
    return 0;
fail:
    fprintf(stderr, "%s\n", hb_last_error_message());
    hb_results_free(r);
    hb_model_free(m);
    return 1;
}
```
