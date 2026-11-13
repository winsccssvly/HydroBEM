//! The engine's errors and diagnostics in Python: an `Error` raises the exception that
//! hydrobem.errors.exception_for chooses by its kind (ValueError, UnsupportedProblemError, ...), with
//! its message and its location; a `Diagnostic` is a hydrobem.errors.Diagnostic.

use hydrobem_core::{Diagnostic, Error, ErrorKind, Location, Severity};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn errors_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("hydrobem.errors")
}

/// The name of an error kind for exception_for.
fn kind_name(k: ErrorKind) -> &'static str {
    match k {
        ErrorKind::Value => "value",
        ErrorKind::Type => "type",
        ErrorKind::Unsupported => "unsupported",
        ErrorKind::State => "state",
        ErrorKind::Cancelled => "cancelled",
        ErrorKind::Internal => "internal",
        ErrorKind::Argument => "argument",
        ErrorKind::Numerical => "numerical",
        ErrorKind::Io => "io",
        ErrorKind::Exists => "exists",
    }
}

/// A hydrobem.Location: the fields that apply (zero-based; the mesh element one-based).
pub fn location<'py>(py: Python<'py>, l: &Location) -> PyResult<Bound<'py, PyAny>> {
    let fields = PyDict::new(py);
    for (name, v) in [
        ("body", l.body),
        ("mesh", l.mesh),
        ("element", l.element),
        ("field_point", l.field_point),
        ("lid", l.lid),
        ("frequency", l.frequency),
    ] {
        if let Some(v) = v {
            fields.set_item(name, v)?;
        }
    }
    errors_module(py)?.getattr("Location")?.call((), Some(&fields))
}

/// A hydrobem.Diagnostic of an engine diagnostic.
pub fn diagnostic<'py>(py: Python<'py>, d: &Diagnostic) -> PyResult<Bound<'py, PyAny>> {
    let severity = match d.severity {
        Severity::Warning => "warning",
        Severity::Error => "error",
    };
    errors_module(py)?.getattr("Diagnostic")?.call1((severity, d.code, location(py, &d.location)?, &d.message))
}

/// The diagnostics as a list of hydrobem.Diagnostic.
pub fn diagnostics<'py>(py: Python<'py>, ds: &[Diagnostic]) -> PyResult<Vec<Bound<'py, PyAny>>> {
    ds.iter().map(|d| diagnostic(py, d)).collect()
}

/// The Python exception of an engine error.
pub fn engine_error(e: Error) -> PyErr {
    Python::attach(|py| {
        let made = || -> PyResult<PyErr> {
            let exc = errors_module(py)?.getattr("exception_for")?.call1((
                kind_name(e.kind()),
                e.message(),
                location(py, e.location())?,
            ))?;
            Ok(PyErr::from_value(exc))
        };
        made().unwrap_or_else(|failure| failure)
    })
}

/// `?` for the engine's results in the functions of the module.
pub trait ToPy<T> {
    fn py(self) -> PyResult<T>;
}

impl<T> ToPy<T> for hydrobem_core::Result<T> {
    fn py(self) -> PyResult<T> {
        self.map_err(engine_error)
    }
}
