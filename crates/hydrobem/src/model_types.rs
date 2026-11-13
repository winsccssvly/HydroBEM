//! The small types of the public API of the model (model.rs): the progress reports of a solve, the
//! values of the items, the answers of the queries (`ItemInfo`, `ObjectInfo`, `Counts`,
//! `BodyCounts`, `MotionEquation`, `PointClass`, `PatchPoints`), the descriptions of the meshes to
//! read or make (`MeshFile`, `AnalyticShape`), the internal control surface and trim of a mesh, and
//! `wavenumber`.

use crate::codes::{ControlSurfaceKind, PointStatus, Stage};
use crate::dispersion::dispersion_x0;
use crate::document::{DType, ItemKind};
use crate::error::{Error, Result};
use crate::progress::{Phase, SecondOrderStep};

/// What a progress report is about: the statics (no frequency), a limit (zero or infinite
/// frequency), an ordinary frequency (zero-based), an item of a second-order step (none: the work
/// of the step all workers share, e.g. the statics of a sub-problem of the potential QTF) or the
/// solve as a whole (a change of the phase, a worker that has become idle).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgressAt {
    Statics,
    Limit,
    Frequency(usize),
    SecondOrder { step: SecondOrderStep, item: Option<usize> },
    Solve,
}

/// A progress report of a solve: the stage, its steps done of total, the frequencies done
/// of total, the seconds since the solve began, the frequency worker that reports (0 to
/// frequency_threads − 1; none for the work all threads share: the statics, a phase) and the
/// phase of the solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Progress {
    pub stage: Stage,
    pub at: ProgressAt,
    pub stage_done: usize,
    pub stage_total: usize,
    pub frequencies_done: usize,
    pub frequencies_total: usize,
    pub seconds: f64,
    pub worker: Option<usize>,
    pub phase: Phase,
}

impl Progress {
    /// A report of the solver (progress.rs).
    #[allow(clippy::too_many_arguments)] // the fields of the report
    pub(crate) fn new(
        stage: Stage,
        at: ProgressAt,
        stage_done: i32,
        stage_total: i32,
        frequencies_done: i32,
        frequencies_total: i32,
        seconds: f64,
        worker: Option<usize>,
        phase: Phase,
    ) -> Progress {
        let count = |n: i32| usize::try_from(n).unwrap_or(0);
        Progress {
            stage,
            at,
            stage_done: count(stage_done),
            stage_total: count(stage_total),
            frequencies_done: count(frequencies_done),
            frequencies_total: count(frequencies_total),
            seconds,
            worker,
            phase,
        }
    }
}

/// A progress callback of a solve; called from one thread at a time, maybe not the solve's.
pub type ProgressFn<'a> = dyn FnMut(&Progress) + Send + 'a;

/// The value of an item. Text holds a text, choice, reference, an origin given by name, an integer
/// or real item given as one of its choices; the arrays are in C order (element ranges an (n, 2)
/// integer array, `Value::ranges`).
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Real(f64),
    Text(String),
    Reals { shape: Vec<usize>, data: Vec<f64> },
    Ints { shape: Vec<usize>, data: Vec<i64> },
    Texts(Vec<String>),
}

impl Value {
    /// A real array of a shape.
    pub fn reals(shape: &[usize], data: Vec<f64>) -> Value {
        Value::Reals { shape: shape.to_vec(), data }
    }
    /// An integer array of a shape.
    pub fn ints(shape: &[usize], data: Vec<i64>) -> Value {
        Value::Ints { shape: shape.to_vec(), data }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Value::Bool(b) = self { Some(*b) } else { None }
    }
    pub fn as_int(&self) -> Option<i64> {
        if let Value::Int(x) = self { Some(*x) } else { None }
    }
    pub fn as_real(&self) -> Option<f64> {
        if let Value::Real(x) = self { Some(*x) } else { None }
    }
    pub fn as_text(&self) -> Option<&str> {
        if let Value::Text(t) = self { Some(t) } else { None }
    }
    pub fn as_reals(&self) -> Option<&[f64]> {
        if let Value::Reals { data, .. } = self { Some(data) } else { None }
    }
    pub fn as_ints(&self) -> Option<&[i64]> {
        if let Value::Ints { data, .. } = self { Some(data) } else { None }
    }
    pub fn as_texts(&self) -> Option<&[String]> {
        if let Value::Texts(t) = self { Some(t) } else { None }
    }
    /// Element ranges (the item ranges of a surface): (first, last) pairs of one-based stored
    /// elements, an (n, 2) integer array.
    pub fn ranges(ranges: &[[i64; 2]]) -> Value {
        Value::Ints { shape: vec![ranges.len(), 2], data: ranges.iter().flatten().copied().collect() }
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Bool(b)
    }
}
impl From<i64> for Value {
    fn from(x: i64) -> Value {
        Value::Int(x)
    }
}
impl From<i32> for Value {
    fn from(x: i32) -> Value {
        Value::Int(x as i64)
    }
}
impl From<f64> for Value {
    fn from(x: f64) -> Value {
        Value::Real(x)
    }
}
impl From<&str> for Value {
    fn from(t: &str) -> Value {
        Value::Text(t.to_string())
    }
}
impl From<String> for Value {
    fn from(t: String) -> Value {
        Value::Text(t)
    }
}
impl From<Vec<f64>> for Value {
    /// A one-dimensional real array.
    fn from(v: Vec<f64>) -> Value {
        Value::Reals { shape: vec![v.len()], data: v }
    }
}
impl<const N: usize> From<[f64; N]> for Value {
    fn from(v: [f64; N]) -> Value {
        Value::from(v.to_vec())
    }
}
impl From<Vec<String>> for Value {
    fn from(v: Vec<String>) -> Value {
        Value::Texts(v)
    }
}
impl From<&[&str]> for Value {
    fn from(v: &[&str]) -> Value {
        Value::Texts(v.iter().map(|s| s.to_string()).collect())
    }
}

/// The kind and state of an item of an object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItemInfo {
    pub kind: ItemKind,
    /// set (not its default)
    pub is_set: bool,
    /// it has a value (set, or a default)
    pub has_value: bool,
    pub is_array: bool,
    /// the element type of an array item, Int for a mesh reference (its ranges), Real for the others
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// an integer or real item given as one of its choices
    pub is_choice: bool,
}

/// The type, name and place of an object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectInfo {
    pub type_name: &'static str,
    pub name: String,
    /// the list it is in ("" for the model)
    pub list: String,
    /// its zero-based place in the list
    pub index: usize,
}

/// The sizes of the problem after the model rules.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub bodies: usize,
    pub meshes: usize,
    pub lids: usize,
    pub ordinary_frequencies: usize,
    pub headings: usize,
    pub coordinates: usize,
    pub radiation_columns: usize,
    /// the field points in the sea (first_order.field_points)
    pub field_points: usize,
    /// the field points in the tanks (the field_points of every tank)
    pub tank_points: usize,
    /// the elements of every lid, with their reflections (patches in a higher-order model)
    pub lid_elements: usize,
    /// the elements of every pressure surface, with their reflections
    pub pressure_elements: usize,
    pub has_zero_frequency: bool,
    pub has_infinite_frequency: bool,
}

/// The elements of one body after the model rules (`Model::body_counts`): the stored elements of
/// its mesh, and the elements of the mesh as the engine uses it (after its trim, the elements no
/// role takes left out) by their roles.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BodyCounts {
    /// the stored elements of the body's mesh (the numbering of the mesh references)
    pub stored: usize,
    /// the elements of the wetted surface (the pressure surfaces on the body's mesh included)
    pub wetted: usize,
    /// the wetted elements with their reflections (wetted x symmetry_copies)
    pub wetted_with_reflections: usize,
    /// the elements of the thin sheets: the dipoles and the dipole dampers on the body's mesh
    pub dipole: usize,
    /// the elements of the interior free surface on the body's mesh; None: the surface is
    /// generated by the engine (its elements counted in the solve)
    pub interior_surface: Option<usize>,
    /// the pressure modes of the body
    pub pressure_modes: usize,
    /// the generalized modes of the body
    pub generalized_modes: usize,
    /// the copies of the symmetry: 1, 2 or 4
    pub symmetry_copies: usize,
}

/// The inertia, stiffness (hydrostatic, weight and additional) and additional damping of the
/// motion equation, ndof x ndof (ndof = 6 N) about each R (theory §3.1).
#[derive(Clone, Debug, PartialEq)]
pub struct MotionEquation {
    pub ndof: usize,
    pub inertia: Vec<f64>,
    pub stiffness: Vec<f64>,
    pub damping: Vec<f64>,
    /// false when a body lacks a centre of gravity, a mass or inertia
    pub ready: bool,
}

/// The status of a point under the field-point rules, with the body, lid or tank it refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointClass {
    pub status: PointStatus,
    /// the zero-based body of the status (a point on or near a body or inside its waterplane), or
    /// the body of the tank of a point given to a tank; None for none
    pub body: Option<usize>,
    /// the zero-based lid of a point on a lid; None for none
    pub lid: Option<usize>,
    /// of a point given to a tank (`PointStatus::Tank`, `EmptyTank`, `AboveTankSurface`) the
    /// zero-based tank of its body (as `bodies[b].tanks[t]` of the results); None for none
    pub tank: Option<usize>,
}

/// A GDF file (and its SPL file) to read as a mesh (reference/inputs-meshes).
#[derive(Clone, Debug, Default)]
pub struct MeshFile {
    pub path: std::path::PathBuf,
    /// a higher-order (patch) mesh
    pub higher_order: bool,
    /// the file ends with a line of zeros
    pub zero_trailer: bool,
    pub spl_path: Option<std::path::PathBuf>,
    /// the discretization (nu, nv, ku, kv, iquo, iqui): one row for all patches or one per patch
    pub discretization: Option<Vec<[i32; 6]>>,
    pub panel_size: Option<f64>,
    /// None: the engine's rule
    pub spline_order: Option<i32>,
    pub outer_quadrature: Option<i32>,
    pub inner_quadrature: Option<i32>,
}

/// A named analytic shape (reference/inputs-meshes §2): circular_cylinder, elliptic_cylinder, sphere, ellipsoid,
/// barge, moonpool_cylinder or cylinder_with_dipole_discs, with its dimensions by name.
#[derive(Clone, Debug, Default)]
pub struct AnalyticShape {
    pub shape: String,
    /// the dimensions by name (radius, draft, a, b, c, half_length, half_beam, moonpool_radius)
    pub dimensions: Vec<(String, f64)>,
    /// cylinder_with_dipole_discs: per disc its outer radius, inner radius and depth
    pub discs: Vec<[f64; 3]>,
    /// the parts (patches) in their order; empty: the default parts of the shape
    pub parts: Vec<String>,
    /// "uniform" (also empty) or "nonuniform"
    pub mapping: String,
    /// one row (nu, nv, ku, kv, iquo, iqui) for all patches, or one per patch
    pub discretization: Vec<[i32; 6]>,
    pub reference_length: f64,
    /// the item of the mesh: "none", "yz", "xz" or "xz_and_yz"
    pub symmetry: String,
    pub label: String,
}

/// A control surface made by the engine (the items of a body's control_surface with a method).
#[derive(Clone, Debug)]
pub(crate) struct ControlSurfaceInput {
    pub kind: ControlSurfaceKind,
    /// the numbers of the analytic cylinder or ellipsoid of a control-surface file
    pub values: Vec<f64>,
    pub panel_size: f64,
    /// automatic: > 0 a circle of this radius about the body origin; <= 0 the polygon
    pub radius: f64,
    pub depth: f64,
    /// automatic: also the free surface between the body waterline and the outer surface
    pub free_surface: bool,
    /// automatic, radius <= 0: (x, y) vertices in the stored sector, counter-clockwise
    pub polygon: Vec<[f64; 2]>,
    /// further (x, y) polylines
    pub partitions: Vec<Vec<[f64; 2]>>,
    pub label: String,
}

/// A trim of a mesh (Model::trim_mesh).
#[derive(Clone, Debug, Default)]
pub(crate) struct Trim {
    /// z of the mesh origin in body coordinates
    pub elevation: f64,
    /// degrees: the pitch about the eta axis first, then the roll about the xi axis
    pub pitch: f64,
    pub roll: f64,
    /// higher order: also the interior free surface inside the trimmed waterline
    pub interior_surface: bool,
    /// low order: the one-based elements of each tank and the body z of its free surface
    pub tanks: Vec<(Vec<i32>, f64)>,
    /// the free-surface plates generated in the trimmed waterline: per plate a point (x, y), body
    /// coordinates, inside the loop it covers (TrimDesc::plates)
    pub plates: Vec<[f64; 2]>,
}

/// Points and unit normals of a patch (mesh coordinates, m; the normals unit, into the fluid).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PatchPoints {
    pub points: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
}

/// The points (global, m) and weights (m^2, or m along the outer circle) of one zone of the
/// potential QTF's free surface as the solve integrates it (`Model::free_surface_points`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FreeSurfacePoints {
    pub points: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
}

/// Wave number of omega^2 = g k tanh(k h): omega >= 0, depth h > 0 or +inf, gravity g > 0; 0 for
/// omega = 0.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
pub fn wavenumber(omega: f64, depth: f64, gravity: f64) -> Result<f64> {
    if !(omega.is_finite() && omega >= 0.0) {
        return Err(Error::value("omega must be a finite number >= 0"));
    }
    if depth.is_nan() || !(depth > 0.0) {
        return Err(Error::value("depth must be > 0 or inf"));
    }
    if !(gravity.is_finite() && gravity > 0.0) {
        return Err(Error::value("gravity must be a positive finite number"));
    }
    if omega == 0.0 {
        return Ok(0.0);
    }
    if depth.is_infinite() {
        return Ok(omega * omega / gravity);
    }
    // k tanh(k h) = w^2 / g
    let x = omega * omega / gravity * depth;
    Ok(dispersion_x0(x) / depth)
}
