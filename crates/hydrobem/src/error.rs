//! Engine error type: every failure carries a kind, a message and where it occurred;
//! warnings are reported as diagnostics.
//!
//! A fallible function returns `Result`; the location of an error is attached where it is known
//! (`Error::at`) and completed on the way out of a scope that knows more (`ResultExt::located`:
//! the body of a rule loop, the frequency of a solve). A panic is a bug.

use std::fmt;

use crate::codes::{Severity, Warning};

/// The result of a fallible engine function.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Where a diagnostic or an error applies: zero-based indices, the mesh element one-based; `None` where
/// the field does not apply.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Location {
    pub body: Option<usize>,
    pub mesh: Option<usize>,
    pub element: Option<usize>,
    pub field_point: Option<usize>,
    pub lid: Option<usize>,
    pub frequency: Option<usize>,
}

/// An engine index as a location field: a negative index (none) is `None`.
pub(crate) fn index(i: i32) -> Option<usize> {
    usize::try_from(i).ok()
}

impl Location {
    /// The fields of this location, those not set taken from o.
    pub fn merged(&self, o: &Location) -> Location {
        Location {
            body: self.body.or(o.body),
            mesh: self.mesh.or(o.mesh),
            element: self.element.or(o.element),
            field_point: self.field_point.or(o.field_point),
            lid: self.lid.or(o.lid),
            frequency: self.frequency.or(o.frequency),
        }
    }
}

pub(crate) fn at_body(b: i32) -> Location {
    Location { body: index(b), ..Location::default() }
}
pub(crate) fn at_element(element: i32) -> Location {
    Location { element: index(element), ..Location::default() }
}
pub(crate) fn at_body_element(body: i32, element: i32) -> Location {
    Location { body: index(body), element: index(element), ..Location::default() }
}

/// The kind of a failure; the Python exceptions are chosen by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Value,
    Type,
    Unsupported,
    State,
    Cancelled,
    Internal,
    Argument,
    Numerical,
    Io,
    Exists,
}

/// A failure: its kind, its message and where it occurred. The fields are boxed, so that a
/// `Result` stays as small as its value.
#[derive(Clone, Debug)]
pub struct Error(Box<ErrorData>);

#[derive(Clone, Debug)]
struct ErrorData {
    kind: ErrorKind,
    message: String,
    location: Location,
}

impl Error {
    /// An error of `kind` with no location.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Error {
        Error(Box::new(ErrorData { kind, message: message.into(), location: Location::default() }))
    }
    pub fn value(message: impl Into<String>) -> Error {
        Error::new(ErrorKind::Value, message)
    }
    pub fn type_(message: impl Into<String>) -> Error {
        Error::new(ErrorKind::Type, message)
    }
    pub fn unsupported(message: impl Into<String>) -> Error {
        Error::new(ErrorKind::Unsupported, message)
    }
    pub fn state(message: impl Into<String>) -> Error {
        Error::new(ErrorKind::State, message)
    }
    pub fn cancelled(message: impl Into<String>) -> Error {
        Error::new(ErrorKind::Cancelled, message)
    }
    pub fn internal(message: impl Into<String>) -> Error {
        Error::new(ErrorKind::Internal, message)
    }
    pub fn argument(message: impl Into<String>) -> Error {
        Error::new(ErrorKind::Argument, message)
    }
    pub fn numerical(message: impl Into<String>) -> Error {
        Error::new(ErrorKind::Numerical, message)
    }
    pub fn io(message: impl Into<String>) -> Error {
        Error::new(ErrorKind::Io, message)
    }
    pub fn exists(message: impl Into<String>) -> Error {
        Error::new(ErrorKind::Exists, message)
    }
    /// This error at `location` (its fields replace those of the error).
    pub fn at(mut self, location: Location) -> Error {
        self.0.location = location;
        self
    }
    /// This error with the fields of its location not set taken from `location`.
    pub fn located(mut self, location: Location) -> Error {
        self.0.location = self.0.location.merged(&location);
        self
    }
    /// This error with another message (the kind and location kept).
    pub fn with_message(mut self, message: impl Into<String>) -> Error {
        self.0.message = message.into();
        self
    }
    pub fn kind(&self) -> ErrorKind {
        self.0.kind
    }
    pub fn message(&self) -> &str {
        &self.0.message
    }
    pub fn location(&self) -> &Location {
        &self.0.location
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.message)
    }
}

impl std::error::Error for Error {}

/// Completes the location of the error of a Result (the fields the error sets win).
pub trait ResultExt<T> {
    fn located(self, location: Location) -> Result<T>;
}

impl<T> ResultExt<T> for Result<T> {
    fn located(self, location: Location) -> Result<T> {
        self.map_err(|e| e.located(location))
    }
}

/// A warning of the model rules or of a solve, or a diagnostic of severity error read from a
/// results file.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub severity: Severity,
    /// the code of the warning (`Warning::code`)
    pub code: i32,
    pub location: Location,
    pub message: String,
}

impl Diagnostic {
    /// A warning at a location.
    pub fn warning_at(w: Warning, location: Location, message: impl Into<String>) -> Diagnostic {
        Diagnostic { severity: Severity::Warning, code: w.code(), location, message: message.into() }
    }
    /// The warning (None for an error or an unknown code).
    pub fn warning(&self) -> Option<Warning> {
        if self.severity == Severity::Error { None } else { Warning::from_code(self.code) }
    }
}

/// An integer item value as int32: values outside its range are an error, not wrapped.
pub(crate) fn int32_of(x: i64, what: &str) -> Result<i32> {
    i32::try_from(x).map_err(|_| Error::value(format!("{what}: the value {x} exceeds int32")))
}
