//! hydrobem._core: the Python extension module of the engine (PyO3), a binding of the public Rust
//! API of hydrobem_core. The Python package (python/hydrobem) builds its public API on it.
//!
//! - `Model` owns an engine `Model` (its document, problem, results and state) behind a mutex; the
//!   solve runs with the Python thread state detached (the GIL released), the model's lock not held,
//!   so that the progress callback and other threads read the model and cancel the solve meanwhile.
//! - `Object` is an object of a model (the model and its `ObjectId`): its items, its lists of
//!   children, the functions of the engine on a mesh and the queries of a body.
//! - `Results` holds the results of a solve, of a results file or made from arrays;
//!   its quantities are NumPy arrays over the engine's data (read-only, not copied).
//! - The engine's errors raise the exceptions of hydrobem.errors (by their kind, with the location);
//!   its diagnostics are `hydrobem.Diagnostic`s.

mod convert;
mod errors;
mod model;
mod object;
mod results;
mod tables;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_core")]
fn core_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<model::PyModel>()?;
    m.add_class::<object::PyObj>()?;
    m.add_class::<results::PyResults>()?;
    tables::register(m)?;
    Ok(())
}
