//! Results object of a solve: dimensional quantities keyed by QuantityId.
//!
//! A quantity holds its elements as their type (`Quantity::elements`, a QuantityData), the one copy
//! of them: a solve moves its arrays in (`put_vec`), the typed slices of the public API
//! (`Results::f64s`, `complex32s`, ...) and the arrays of the Python and C interfaces borrow them, and
//! `Quantity::data()` views their bytes (native order); a copy of the results (`Clone`, as the
//! Python and C interfaces return the results of a solve) shares them. A quantity of a results file is read when
//! first asked for: its elements go into the same OnceLock (results_file::load_quantity); `pending`
//! stays true once loaded (byte_count is its size).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use num_complex::{Complex32, Complex64};

use crate::codes::{ElementType, QuantityId};
use crate::document::Document;
use crate::document_enabled::{Scope, scoped_document};
use crate::error::{Diagnostic, Error, Result};
use crate::model::path_text;
use crate::quantities::quantity_info;
use crate::results_file::{load_quantity, save_results};
use crate::wamit_files::{WamitWrite, check_quantities, normalized, write_wamit};

#[derive(Clone, Debug)]
pub struct Quantity {
    pub dtype: ElementType,
    pub dims: Vec<i64>,
    // a quantity of a results file not read yet: its place in the file and its CRC-32
    pub pending: bool,
    pub offset: i64,
    pub size: i64,
    pub crc: u32,
    /// the elements: those put (set when the quantity is made), or those of a pending quantity once
    /// read (load_quantity); shared by the copies of the results (Results::clone), since a quantity
    /// is replaced, never changed in place
    pub elements: OnceLock<Arc<QuantityData>>,
}

impl Default for Quantity {
    fn default() -> Self {
        Quantity {
            dtype: ElementType::Float64,
            dims: Vec::new(),
            pending: false,
            offset: 0,
            size: 0,
            crc: 0,
            elements: OnceLock::new(),
        }
    }
}

impl Quantity {
    /// A quantity of these elements and dimensions (its type that of the elements).
    pub fn of(dims: Vec<i64>, data: QuantityData) -> Quantity {
        Quantity { dtype: data.element_type(), dims, elements: OnceLock::from(Arc::new(data)), ..Quantity::default() }
    }
    pub fn byte_count(&self) -> i64 {
        if self.pending { self.size } else { self.data().len() as i64 }
    }
    /// The bytes of the elements (native order): those put, or those of a pending quantity read by
    /// load_quantity (empty before).
    pub fn data(&self) -> &[u8] {
        self.elements.get().map_or(&[], |d| d.as_bytes())
    }
}

/// The elements of a quantity as their type.
#[derive(Clone, Debug, PartialEq)]
pub enum QuantityData {
    Float64(Vec<f64>),
    Float32(Vec<f32>),
    ComplexF32(Vec<Complex32>),
    Int32(Vec<i32>),
    ComplexF64(Vec<Complex64>),
    Int64(Vec<i64>),
}

impl QuantityData {
    pub fn element_type(&self) -> ElementType {
        match self {
            QuantityData::Float64(_) => ElementType::Float64,
            QuantityData::Float32(_) => ElementType::Float32,
            QuantityData::ComplexF32(_) => ElementType::ComplexF32,
            QuantityData::Int32(_) => ElementType::Int32,
            QuantityData::ComplexF64(_) => ElementType::ComplexF64,
            QuantityData::Int64(_) => ElementType::Int64,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            QuantityData::Float64(v) => v.len(),
            QuantityData::Float32(v) => v.len(),
            QuantityData::ComplexF32(v) => v.len(),
            QuantityData::Int32(v) => v.len(),
            QuantityData::ComplexF64(v) => v.len(),
            QuantityData::Int64(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    // the elements of the bytes of a quantity of type t (native order, as put stores them)
    pub(crate) fn of_bytes(t: ElementType, b: &[u8]) -> Result<QuantityData> {
        Ok(match t {
            ElementType::Float64 => QuantityData::Float64(elements_of(b)),
            ElementType::Float32 => QuantityData::Float32(elements_of(b)),
            ElementType::Int32 => QuantityData::Int32(elements_of(b)),
            ElementType::Int64 => QuantityData::Int64(elements_of(b)),
            ElementType::ComplexF32 => QuantityData::ComplexF32(elements_of(b)),
            ElementType::ComplexF64 => QuantityData::ComplexF64(elements_of(b)),
            ElementType::Text => return Err(Error::internal("a quantity of text elements")),
        })
    }
    /// The bytes of the elements (native order): a view of them, not a copy.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            QuantityData::Float64(v) => plain_bytes(v),
            QuantityData::Float32(v) => plain_bytes(v),
            QuantityData::Int32(v) => plain_bytes(v),
            QuantityData::Int64(v) => plain_bytes(v),
            QuantityData::ComplexF32(v) => plain_bytes(v),
            QuantityData::ComplexF64(v) => plain_bytes(v),
        }
    }
}

// The element types of a quantity: numbers of fixed size without padding, every bit pattern a value
// (Complex<T> is repr(C), its two parts in order), the bytes in memory those of to_ne_bytes.
trait Plain: Copy {}
impl Plain for f32 {}
impl Plain for f64 {}
impl Plain for i32 {}
impl Plain for i64 {}
impl Plain for Complex32 {}
impl Plain for Complex64 {}

// the bytes of elements in memory (native order), borrowed
fn plain_bytes<T: Plain>(v: &[T]) -> &[u8] {
    // SAFETY: T is one of the Plain types above, which have no padding, so all size_of_val(v) bytes
    // from v's start are initialized; u8 has alignment 1; the slice borrows v for its lifetime
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// The meta data of a solve: the build, the time (UTC, ISO 8601), the route, the
/// threads, the engine seconds and the hooks of the environment that were set (`NAME=value`, empty
/// without; internal studies and tests only); present is false for results not from a solve and
/// for a model file (whose build and time of the save it holds).
#[derive(Clone, Debug, Default)]
pub struct RunMeta {
    pub present: bool,
    pub hydrobem_version: String,
    pub commit: String,
    pub compiler: String,
    pub linear_algebra: String,
    pub build_type: String,
    pub time: String,
    pub route: String,
    pub api_version: i32,
    pub frequency_threads: i32,
    pub inner_threads: i32,
    pub engine_seconds: Vec<f64>,
    pub environment_hooks: Vec<String>,
}

/// The results of a solve, of a results file or made by a client (`Model::new_results`): the
/// quantities by QuantityId, the diagnostics, the meta data and, for a
/// solve, a copy of the model solved.
#[derive(Clone, Debug, Default)]
pub struct Results {
    pub(crate) route: String,
    pub(crate) quantities: BTreeMap<QuantityId, Quantity>, // absent key: not available
    pub(crate) diagnostics: Vec<Diagnostic>,               // the warnings of the model and of the solve
    pub(crate) model: Option<Arc<Document>>,               // the inputs of the solve, when given as objects
    pub(crate) meta: RunMeta,                              // the meta data of the solve
    pub(crate) file: String,                               // the file it was opened from, else empty
    pub(crate) model_file: bool,                           // opened from a model file (.hbm): no quantities
    pub(crate) solved: OnceLock<Option<Arc<Document>>>,    // the enabled objects of model (made when first used)
}

/// The parts of results for the component tests of the results files (tests/io.rs; reached
/// through the hidden `internals` module only, not an API).
pub mod parts {
    use super::*;

    /// Empty results of a route.
    pub fn with_route(route: &str) -> Results {
        Results { route: route.to_string(), ..Results::default() }
    }
    pub fn quantities(r: &Results) -> &BTreeMap<QuantityId, Quantity> {
        &r.quantities
    }
    pub fn meta_mut(r: &mut Results) -> &mut RunMeta {
        &mut r.meta
    }
    pub fn diagnostics_mut(r: &mut Results) -> &mut Vec<Diagnostic> {
        &mut r.diagnostics
    }
}

/// An element of a quantity: its bytes in native order, as the quantities store them.
pub trait Element: Copy {
    /// the bytes of one element
    const SIZE: usize;
    /// The element of its SIZE bytes.
    fn from_bytes(b: &[u8]) -> Self;
    /// The elements as the data of a quantity (moved, not copied).
    fn data(v: Vec<Self>) -> QuantityData;
}

macro_rules! real_element {
    ($($t:ty, $variant:ident);*) => {
        $(impl Element for $t {
            const SIZE: usize = std::mem::size_of::<$t>();
            fn from_bytes(b: &[u8]) -> Self {
                <$t>::from_ne_bytes(b.try_into().expect("the bytes of one element"))
            }
            fn data(v: Vec<Self>) -> QuantityData {
                QuantityData::$variant(v)
            }
        })*
    };
}
real_element!(f32, Float32; f64, Float64; i32, Int32; i64, Int64);

macro_rules! complex_element {
    ($($t:ty, $part:ty, $variant:ident);*) => {
        $(impl Element for $t {
            const SIZE: usize = 2 * std::mem::size_of::<$part>();
            fn from_bytes(b: &[u8]) -> Self {
                let half = std::mem::size_of::<$part>();
                <$t>::new(<$part>::from_bytes(&b[..half]), <$part>::from_bytes(&b[half..]))
            }
            fn data(v: Vec<Self>) -> QuantityData {
                QuantityData::$variant(v)
            }
        })*
    };
}
complex_element!(Complex32, f32, ComplexF32; Complex64, f64, ComplexF64);

/// The elements of bytes (native order); a partial element at the end is ignored.
pub(crate) fn elements_of<T: Element>(b: &[u8]) -> Vec<T> {
    b.chunks_exact(T::SIZE).map(T::from_bytes).collect()
}

/// Store a copy of v as quantity q of r with its dimensions (put_vec moves them); the element
/// type is that of T, which must be the quantity's (QuantityId::element_type).
pub fn put<T: Element>(r: &mut Results, q: QuantityId, dims: Vec<i64>, v: &[T]) {
    put_vec(r, q, dims, v.to_vec());
}

/// Store v as quantity q of r with its dimensions: the elements moved into the results, not
/// copied; their type T must be the quantity's (QuantityId::element_type).
pub fn put_vec<T: Element>(r: &mut Results, q: QuantityId, dims: Vec<i64>, v: Vec<T>) {
    let qq = Quantity::of(dims, T::data(v));
    assert_eq!(qq.dtype, q.element_type(), "{} put with another element type", q.name());
    r.quantities.insert(q, qq);
}

/// The availability, type and dimensions of a quantity of results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantityShape {
    pub element_type: ElementType,
    pub dims: Vec<usize>,
    /// the size of its data
    pub bytes: usize,
}

/// What `Results::write_wamit` writes (reference/file-formats §2-§4).
#[derive(Clone, Debug, Default)]
pub struct WamitOptions {
    /// replace files that exist
    pub overwrite: bool,
    /// the frequencies as radian frequencies (else periods)
    pub omega: bool,
    /// the time stamp of the headers (one line); None: the local time now
    pub stamp: Option<String>,
}

macro_rules! typed_slice {
    ($(#[$doc:meta])* $name:ident, $variant:ident, $t:ty) => {
        $(#[$doc])*
        pub fn $name(&self, q: QuantityId) -> Result<&[$t]> {
            match self.typed(q)? {
                QuantityData::$variant(v) => Ok(v),
                d => Err(Error::type_(format!("{} holds {:?} elements", q.name(), d.element_type()))),
            }
        }
    };
}

impl Results {
    /// The route of the solve ("given" for results made by a client).
    pub fn route(&self) -> &str {
        &self.route
    }
    /// The warnings of the model, then of the solve.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
    pub fn meta(&self) -> &RunMeta {
        &self.meta
    }
    /// The inputs of the solve (None for results without a model): every object, the disabled ones
    /// with their item `enabled` false.
    pub fn document(&self) -> Option<&Document> {
        self.model.as_deref()
    }
    /// The model as it was solved: the copy of the enabled objects of the inputs, whose bodies,
    /// lids and field points the quantities number (None for results without a model).
    pub fn solved_document(&self) -> Option<Arc<Document>> {
        self.solved
            .get_or_init(|| {
                let m = self.model.as_deref()?;
                match scoped_document(m, Scope::Model) {
                    Ok(e) => Some(Arc::new(e.doc)),
                    Err(_) => self.model.clone(), // no body enabled: nothing was solved of it
                }
            })
            .clone()
    }
    /// Whether the results have quantity q.
    pub fn has(&self, q: QuantityId) -> bool {
        self.quantities.contains_key(&q)
    }
    /// The quantities the results have, in the order of the table.
    pub fn available(&self) -> Vec<QuantityId> {
        self.quantities.keys().copied().collect()
    }
    /// The type and dimensions of quantity q (None when the results do not have it).
    pub fn shape(&self, q: QuantityId) -> Option<QuantityShape> {
        let x = self.quantities.get(&q)?;
        Some(QuantityShape {
            element_type: x.dtype,
            dims: x.dims.iter().map(|&d| d.max(0) as usize).collect(),
            bytes: x.byte_count().max(0) as usize,
        })
    }
    // the quantity, its bytes read from the file first: ErrorKind::State when it is not there
    fn loaded(&self, code: QuantityId) -> Result<&Quantity> {
        let Some(q) = self.quantities.get(&code) else {
            return Err(Error::state(format!("quantity {} is not available", code.name())));
        };
        load_quantity(self, q, code.name())?;
        Ok(q)
    }
    /// The bytes of quantity q (C order, native element layout), read from its file the first time.
    pub fn bytes(&self, q: QuantityId) -> Result<&[u8]> {
        Ok(self.loaded(q)?.data())
    }
    /// The elements of quantity q as their type.
    pub fn typed(&self, q: QuantityId) -> Result<&QuantityData> {
        self.loaded(q)?
            .elements
            .get()
            .map(|d| &**d)
            .ok_or_else(|| Error::internal(format!("quantity {} has no elements", q.name())))
    }
    typed_slice!(
        /// The elements of a float64 quantity.
        f64s,
        Float64,
        f64
    );
    typed_slice!(
        /// The elements of a float32 quantity.
        f32s,
        Float32,
        f32
    );
    typed_slice!(
        /// The elements of a complex64 quantity (complex float32).
        complex32s,
        ComplexF32,
        Complex32
    );
    typed_slice!(
        /// The elements of a complex128 quantity (complex float64).
        complex64s,
        ComplexF64,
        Complex64
    );
    typed_slice!(
        /// The elements of an int32 quantity.
        i32s,
        Int32,
        i32
    );
    typed_slice!(
        /// The elements of an int64 quantity.
        i64s,
        Int64,
        i64
    );

    /// Quantity q normalized as in the numeric files (reference/file-formats §1), with its
    /// dimensions: the coefficients, forces, motions, mean drifts and the field and body components
    /// have one (ErrorKind::Argument for the others).
    pub fn normalized(&self, q: QuantityId) -> Result<(Vec<i64>, Vec<Complex64>)> {
        normalized(self, q)
    }
    /// Writes a results file (.hbr): the model, the meta data, the diagnostics and every
    /// quantity; atomic (a temporary file renamed).
    pub fn save(&self, path: &Path, overwrite: bool) -> Result<()> {
        save_results(self, path_text(path)?, overwrite)
    }
    /// Writes the WAMIT-format numeric files (.1 ... .9, .hst, .fpt, .pnl) and the .out summary of
    /// results with a model (reference/file-formats §2-§4): stem plus each extension, atomically,
    /// the folder made; returns the files written.
    pub fn write_wamit(&self, stem: &Path, options: &WamitOptions) -> Result<Vec<String>> {
        let w = WamitWrite {
            overwrite: options.overwrite,
            omega: options.omega,
            stamp: options.stamp.clone().unwrap_or_default(),
        };
        write_wamit(self, path_text(stem)?, &w)
    }
    /// Stores quantity q in results made by `Model::new_results` (results computed elsewhere, to
    /// write them): its data of the quantity's element type, the dimensions of its axes.
    pub fn put(&mut self, q: QuantityId, dims: &[usize], data: QuantityData) -> Result<()> {
        if data.element_type() != q.element_type() {
            return Err(Error::type_(format!("{} holds {:?} elements", q.name(), q.element_type())));
        }
        let dims: Vec<i64> = dims.iter().map(|&d| i64::try_from(d).unwrap_or(i64::MAX)).collect();
        self.put_data(q, &dims, data)
    }
    /// Checks the quantities that are given together, once every quantity of results made by
    /// `Model::new_results` is put (each `put` checks the shapes alone): the eleven
    /// `hydrostatics_*` all or none; `drift_pressure_integration` with
    /// `drift_pressure_integration_body_axes` and `drift_control_surface` with
    /// `drift_control_surface_body_axes` (each needs the other); any drift with
    /// `drift_heading_pairs`; `body_points_ids` in range, and with a row of panels (sx or sy not 0)
    /// with `body_points`, `body_points_normals` and `body_points_areas`. `ErrorKind::Value` naming the quantity
    /// missing. `write_wamit` checks the same before it writes.
    pub fn check_given(&self) -> Result<()> {
        let Some(model) = self.solved_document() else {
            return Err(Error::state("these results hold no model"));
        };
        check_quantities(self, &model, true)
    }
    // put of the elements (of the quantity's type), moved into the results
    fn put_data(&mut self, quantity: QuantityId, dims: &[i64], data: QuantityData) -> Result<()> {
        let info = quantity_info(quantity);
        if dims.len() > 4 {
            return Err(Error::value(format!("{}: a quantity has 0 to 4 dimensions, got {}", info.name, dims.len())));
        }
        let mut q = Quantity { dtype: info.dtype, ..Quantity::default() };
        let mut n: i64 = 1;
        for &d in dims {
            if d < 0 {
                return Err(Error::value(format!("{}: a negative dimension {d}", info.name)));
            }
            if d > 0 && n > (1i64 << 40) / d {
                return Err(Error::value(format!("{}: more than 2^40 elements", info.name)));
            }
            q.dims.push(d);
            n *= d;
        }
        let bytes = data.as_bytes().len();
        if data.len() as i64 != n {
            return Err(Error::argument(format!("{}: {bytes} bytes for {n} elements of its type", info.name)));
        }
        // the axes are the model's (Model::new_results), and every quantity has their shape
        if quantity == QuantityId::Omegas
            || quantity == QuantityId::Headings
            || quantity == QuantityId::RadiationModeIds
            || quantity == QuantityId::Coordinates
        {
            return Err(Error::value(format!("{} is an axis of the model, not given", info.name)));
        }
        // so are the points of the field outputs: the inputs' points
        if quantity == QuantityId::FieldPoints || quantity == QuantityId::Tanks || quantity == QuantityId::TanksIds {
            return Err(Error::value(format!("{} are the points of the model, not given", info.name)));
        }
        let Some(model) = self.solved_document() else {
            return Err(Error::state("these results hold no model"));
        };
        q.elements = OnceLock::from(Arc::new(data));
        let before = self.quantities.insert(quantity, q);
        // the shapes; the set when it is used
        if let Err(e) = check_quantities(self, &model, false) {
            // the results as they were
            match before {
                Some(b) => {
                    self.quantities.insert(quantity, b);
                }
                None => {
                    self.quantities.remove(&quantity);
                }
            }
            return Err(e);
        }
        Ok(())
    }
}
