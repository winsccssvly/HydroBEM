# Design of the program

## Components

1. **Engine library** (`hydrobem_core`, a Rust library with a safe public API,
   [rust-api.md](../reference/rust-api.md)). It holds the model: the inputs as objects with named items, checked when they
   are set and by the model rules, its meshes among them, and the results of its last solve. It solves and gives
   **dimensional results**, first and second order (the mean drifts and the QTFs; SI units, per
   unit incident wave amplitude of 1 m). It
   reads and writes the model file (`.hbm`) and the results file (`.hbr`), reads GDF, SPL, CSF,
   FDF (its panels) and `.hbg` mesh files and writes GDF, SPL and `.hbg`, and writes the WAMIT-format numeric files and the `.out` summary with the
   normalization of the files.
2. **Python package** (`hydrobem`), a thin client. `hb.Model` reads and sets the engine's items
   (reference/inputs-model), calls the engine through its extension module `hydrobem._core` (PyO3), and gives the results (`Results`, the
   engine's quantities through its result tree). It computes nothing and writes no file of its own:
   every rule, quantity, normalization and file is the engine's.
3. **C ABI** (`hydrobem.dll` and `hydrobem.h`, `crates/hydrobem-c`,
   [c-api.md](../reference/c-api.md)), the same binding for programs that call C: the model
   behind a handle, its objects by number, every answer a table of named columns.

The library has no other front end and no fallback implementation. A valid
problem for which the library has no solver is rejected with a dedicated error kind
(`ErrorKind::Unsupported`, Python `UnsupportedProblemError`, a subclass of
`NotImplementedError`), never approximated.

## Layers of the Python package

*Table 1. The layers of the Python package.*

| Layer | Contents |
|---|---|
| model (`engine.py`) | `Model` (load, save, copy, check, warnings, route, solve, progress, cancel, state, results, reset, clear, close, the results file); `Progress`, `SolveProgress`, `FrequencyProgress`, `LimitProgress` |
| objects (`objects.py`, `_classes.py`) | `ModelObject` (items as attributes, lists of children, `update`, `info`), `ObjectList` (`add`, `insert`, `move`, `rename`), `BodyObject` (`trimmed_mesh`), `MeshObject` and `MeshListBase` (the meshes read, made, trimmed and written by the engine); generated from the schema: the class of every object type and list, with the descriptions and keywords of the items |
| queries (methods of the owner of what they need; their answers the read-only classes of `queries.py`) | of a mesh `geometry`, `elements`, `tessellation`, `quality`, `patch_points`; of a body `hydrostatics`, `panels`, `tessellation`, `mesh_quality`; of the `Model` `counts`, `memory_estimate`, `classify_points`, `motion_equation`, `panels` and `tessellation` (by role: the lids, the fixed dampers and the free surface of the potential QTF among them; one set of role words and the same seven keys in both), `free_surface_points(zone)`, `mode_displacements`, `mesh_quality`; `hb.ranges(selection)` (`selection.py`), `wavenumber(omega, depth, gravity)`, `quantities()`, `build_info()` need no model |
| results (`results.py`, `_result_views.py`, `_result_extras.py`) | `Results`: an engine results handle, its quantities by one path, the views by owner and group (every class generated from the result tree on a hand-written base), a quantity by name (`quantity`), `warnings`, `diagnostics`, `available`, their axes, the model solved, `normalized`, `save`, `write_wamit`; `Results.of`; `load_results`; the results hold what the model asked for |
| low level (`_core`, the extension module) | the engine's model, its objects and items, its results (`Model`, `Object`, `Results`) |

Output writers never run solves; solves never write files.

## Precision

- Geometry and input data: float64.
- Influence matrices, linear systems and their solutions: complex64 (single
  precision) in the low order; complex128 in the higher order and in the diagnostic
  double-precision mode (`HB_DIAG_DOUBLE`); the real mass matrices of the projections
  float64. Stored coefficients float32, complex results complex64.
- Results meet the accuracy targets of theory §13.2 and test-plan A, B in this precision.
- Why two precisions: the low-order systems are well conditioned, so complex64 moves their
  solution by about 1e-6; the Galerkin systems of the higher order have condition numbers two to
  three orders of magnitude larger and its body velocities differentiate the
  solution, so they stay complex128, at under 1 % of the solve time (theory §7, §13.1).
- Every kernel of the linear algebra runs sequentially inside its item. Dense linear algebra
  uses a blocked LU with partial pivoting on faer's kernels (pure Rust; the blocks on the
  engine's threads).

## Parallelism

- **Engine threads** (the items `options.frequency_threads`, `options.inner_threads`): the limits first, then the ordinary
  frequencies on the frequency threads; inside each frequency the matrix
  assembly, the symmetry characters and the blocked LU on the inner threads; the statics on both
  counts multiplied. The results must be **byte-identical** for any counts. Two counts, because
  frequency threads scale almost linearly but each holds a frequency's matrices, while inner
  threads share them but scale less well; byte identity makes the counts a choice of speed only
  and lets every change be checked bit for bit (theory §13.3).
- Progress callbacks are serialized: the stage reports at most every 0.1 s, and every finished
  frequency. The state of every frequency (`Model.progress`, `ProgressMonitor`) is kept in atomics
  the steps write without a lock, read at any instant from any thread.

## Errors

Every fallible function of the engine returns a `Result`; its `Error` has a kind, a message and a
location, and is propagated with `?` (a panic is a bug of the engine). Python maps the
kinds as Table 2 shows.

*Table 2. The error kinds of the engine and the Python exceptions they map to.*

| Kind | Python |
|---|---|
| value | `ValueError` |
| type | `TypeError` |
| valid but no solver | `UnsupportedProblemError` |
| cancelled | `SolveCancelledError` |
| singular / non-finite system | `numpy.linalg.LinAlgError` |
| argument (an unknown object, a bad index or argument) | `ValueError` |
| a file that exists (`ErrorKind::Exists`) | `FileExistsError` ("... exists (pass overwrite=True)") |
| a file that cannot be read or written | `OSError` |
| state (calculating, no results), internal | `RuntimeError` |

Validation is complete before any assembly: invalid or unsupported inputs fail
before work starts. Nothing is silently skipped, substituted or zero-filled.

An engine error also names where it applies: the
exception's `location` is an `hb.Location` of the zero-based body, mesh, field point, lid or
ordinary frequency and the one-based mesh element that apply (`body`, `mesh`, `field_point`, `lid`, `frequency`, `element`; None where one does not). The engine's warnings (valid input whose result
may be unexpected, e.g. the pressure drift with pressure surfaces) are issued as
`HydroBEMWarning` (a `UserWarning`) with `code` and `location`, and listed in
`Results.warnings` (`hb.Diagnostic` objects; `Results.diagnostics` every severity, `Model.warnings`
those of the last check). In Rust: `Model::diagnostics` (the warnings of the
last rule check) and `Results::diagnostics`.

## Data needed at run time

The library needs no data files: any tables it uses are generated by scripts in
this repository and compiled into it, with a checksum verified before first
use.
