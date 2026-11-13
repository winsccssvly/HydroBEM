//! `HbResults`: the results of a solve, of a results file, or made from arrays. A quantity is
//! read as a view of the engine's elements (no copy), valid until the results are freed or changed.

use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::sync::Mutex;

use hydrobem_core::{Complex32, Complex64, ElementType, Error, Model, QuantityData, QuantityId, Results, WamitOptions};

use crate::model::{HbModel, diagnostics_table, give_model, model};
use crate::status::{HbStatus, Out, argument, elements, guard, opt_text, owned_text, path, put, slice, text, writable};
use crate::table::{Data, HB_C64, HB_C128, HB_F32, HB_F64, HB_I32, HB_I64, HbColumn, HbTable, give};

// the name and the dimensions of a quantity viewed, kept while the results live
struct Kept {
    name: CString,
    shape: Box<[usize]>,
}

/// The results of a solve (opaque).
pub struct HbResults {
    r: Results,
    kept: Mutex<HashMap<usize, Kept>>,
}

/// Hands results to the caller.
///
/// # Safety
/// `out` is null or a writable pointer.
pub unsafe fn give_results(out: *mut *mut HbResults, r: Results) -> Out<()> {
    // refused before the box is made: nothing leaks
    writable(out, "out")?;
    let h = HbResults { r, kept: Mutex::new(HashMap::new()) };
    unsafe { put(out, Box::into_raw(Box::new(h)), "out") }
}

// the results for reading: shared, so that several threads may read them at once
unsafe fn results<'a>(r: *const HbResults) -> Out<&'a HbResults> {
    if r.is_null() {
        return argument("results is a null pointer");
    }
    // SAFETY: results of this library (the caller's contract)
    Ok(unsafe { &*r })
}

// the results for a change (hb_results_put): no other call on them meanwhile (the caller's contract)
unsafe fn results_mut<'a>(r: *mut HbResults) -> Out<&'a mut HbResults> {
    if r.is_null() {
        return argument("results is a null pointer");
    }
    // SAFETY: results of this library, not shared during the call (the caller's contract)
    Ok(unsafe { &mut *r })
}

fn quantity(name: &str) -> Out<QuantityId> {
    QuantityId::from_name(name).ok_or_else(|| Error::value(format!("no result quantity {name:?}")).into())
}

fn code(q: QuantityId) -> usize {
    QuantityId::ALL.iter().position(|x| *x == q).unwrap_or(usize::MAX)
}

/// The column type of an element type (none for text).
pub fn dtype_of(t: ElementType) -> Option<i32> {
    match t {
        ElementType::Float64 => Some(HB_F64),
        ElementType::Float32 => Some(HB_F32),
        ElementType::ComplexF32 => Some(HB_C64),
        ElementType::Int32 => Some(HB_I32),
        ElementType::ComplexF64 => Some(HB_C128),
        ElementType::Int64 => Some(HB_I64),
        ElementType::Text => None,
    }
}

/// Frees results (null: nothing).
///
/// # Safety
/// `results` is null or results of this library not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_free(results: *mut HbResults) {
    if !results.is_null() {
        // SAFETY: made by Box::into_raw in give_results
        drop(unsafe { Box::from_raw(results) });
    }
}

/// Whether the results have a quantity (1) or not (0); an unknown name is `HB_ERR_VALUE`.
///
/// # Safety
/// `results` is results of this library; `name` a UTF-8 string; `has` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_has(results: *mut HbResults, name: *const c_char, has: *mut i32) -> HbStatus {
    guard(|| {
        writable(has, "has")?;
        let r = unsafe { self::results(results) }?;
        let q = quantity(unsafe { text(name, "name") }?)?;
        unsafe { put(has, i32::from(r.r.has(q)), "has") }
    })
}

/// The view of a quantity (reference/results): its name, element type, dimensions and the
/// engine's elements in C order, valid until the results are freed or changed; `HB_ERR_VALUE`
/// when the results do not have it.
///
/// # Safety
/// `results` is results of this library; `name` a UTF-8 string; `column` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_quantity(
    results: *mut HbResults,
    name: *const c_char,
    column: *mut HbColumn,
) -> HbStatus {
    guard(|| {
        writable(column, "column")?;
        let h = unsafe { self::results(results) }?;
        let q = quantity(unsafe { text(name, "name") }?)?;
        let Some(shape) = h.r.shape(q) else {
            return Err(Error::value(format!("the results do not have {}", q.name())).into());
        };
        let (dtype, count, data): (i32, usize, *const c_void) = match h.r.typed(q)? {
            QuantityData::Float64(v) => (HB_F64, v.len(), v.as_ptr().cast()),
            QuantityData::Float32(v) => (HB_F32, v.len(), v.as_ptr().cast()),
            QuantityData::ComplexF32(v) => (HB_C64, v.len(), v.as_ptr().cast()),
            QuantityData::Int32(v) => (HB_I32, v.len(), v.as_ptr().cast()),
            QuantityData::ComplexF64(v) => (HB_C128, v.len(), v.as_ptr().cast()),
            QuantityData::Int64(v) => (HB_I64, v.len(), v.as_ptr().cast()),
        };
        let mut kept = h.kept.lock().unwrap_or_else(|p| p.into_inner());
        // kept until the results are changed (hb_results_put clears them) or freed
        let k = kept.entry(code(q)).or_insert_with(|| Kept {
            name: CString::new(q.name()).unwrap_or_default(),
            shape: shape.dims.clone().into_boxed_slice(),
        });
        let c = HbColumn { name: k.name.as_ptr(), dtype, ndim: k.shape.len(), shape: k.shape.as_ptr(), count, data };
        unsafe { put(column, c, "column") }
    })
}

/// The part of a quantity the view of a path selects for an owner (reference/results §1, the
/// paths of `hb_result_views`): `body` the zero-based body of the model solved (-1: the results),
/// `owner` the zero-based lid of a `lids` path, pressure surface of the body of a
/// `pressure_surfaces` path or tank of the body of a `tanks` path (-1 for the others). The table
/// holds one column, `values`, of the quantity's element type and the dimensions of the part (a
/// copy); no column when the results do not have the quantity or the body has no part of it.
/// `HB_ERR_VALUE` for a path of no view,
/// `HB_ERR_ARGUMENT` for a body or owner out of range, missing or not taken.
///
/// # Safety
/// `results` is results of this library; `path` a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_select(
    results: *mut HbResults,
    path: *const c_char,
    body: i64,
    owner: i64,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::results(results) }?;
        let path = unsafe { text(path, "path") }?;
        let index = |v: i64, what: &str| -> Out<Option<usize>> {
            match v {
                -1 => Ok(None),
                v if v >= 0 => Ok(Some(v as usize)),
                _ => argument(format!("{what} must be -1 or a zero-based index, not {v}")),
            }
        };
        let (body, owner) = (index(body, "body")?, index(owner, "owner")?);
        let t = match r.r.select(path, body, owner)? {
            None => HbTable::new(),
            Some(s) => {
                let data = match s.data {
                    QuantityData::Float64(v) => Data::F64(v),
                    QuantityData::Float32(v) => Data::F32(v),
                    QuantityData::ComplexF32(v) => Data::C64(v),
                    QuantityData::Int32(v) => Data::I32(v),
                    QuantityData::ComplexF64(v) => Data::C128(v),
                    QuantityData::Int64(v) => Data::I64(v),
                };
                HbTable::new().col("values", &s.dims, data)
            }
        };
        unsafe { give(out, t) }
    })
}

/// Where the part of `hb_results_select` lies in the whole quantity (the same arguments and
/// errors): the column `quantity` (text, its name) and, one row per index taken, the columns `axis`
/// (`HB_I64`, the zero-based axis of the quantity) and `index` (`HB_I64`, the zero-based index on
/// that axis); the axes not listed are taken whole. No column when `hb_results_select` gives none.
/// Python's `Results.selection` gives the same.
///
/// # Safety
/// `results` is results of this library; `path` a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_selection(
    results: *mut HbResults,
    path: *const c_char,
    body: i64,
    owner: i64,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::results(results) }?;
        let path = unsafe { text(path, "path") }?;
        let index = |v: i64, what: &str| -> Out<Option<usize>> {
            match v {
                -1 => Ok(None),
                v if v >= 0 => Ok(Some(v as usize)),
                _ => argument(format!("{what} must be -1 or a zero-based index, not {v}")),
            }
        };
        let (body, owner) = (index(body, "body")?, index(owner, "owner")?);
        let t = match r.r.selection(path, body, owner)? {
            None => HbTable::new(),
            Some(s) => {
                let (mut axes, mut rows) = (Vec::new(), Vec::new());
                for (axis, taken) in &s.axes {
                    for i in taken.indices() {
                        axes.push(*axis as i64);
                        rows.push(i as i64);
                    }
                }
                let n = axes.len();
                HbTable::new().text("quantity", s.quantity.name()).i64s("axis", &[n], axes).i64s("index", &[n], rows)
            }
        };
        unsafe { give(out, t) }
    })
}

/// Checks the quantities put into results made by `hb_model_new_results` that are given together, once
/// all are put (each `hb_results_put` checks its own shape alone): the eleven `hydrostatics_*` all
/// or none, each drift with its `_body_axes` part and with `drift_heading_pairs`, the body points
/// with their normals and areas. `HB_ERR_VALUE` naming the quantity missing, `HB_ERR_STATE` for
/// results without a model. `hb_results_write_wamit` checks the same before it writes; Python's
/// `Model.new_results` makes this check after its quantities are put.
///
/// # Safety
/// `results` is results of this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_check(results: *mut HbResults) -> HbStatus {
    guard(|| {
        let r = unsafe { self::results(results) }?;
        r.r.check_given()?;
        Ok(())
    })
}

/// The zero-based indices of the columns of a body of the model solved: coordinates (the D axis)
/// and radiation_columns (the K axis), `HB_I64` (the rows `hb_results_select` takes of a body).
/// `body` the zero-based body of the model solved; `HB_ERR_ARGUMENT` for a negative body or one
/// beyond the bodies solved, `HB_ERR_STATE` for results without a model.
///
/// # Safety
/// `results` is results of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_body_columns(
    results: *mut HbResults,
    body: i64,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::results(results) }?;
        let Some(doc) = r.r.solved_document() else {
            return Err(Error::state("these results hold no model").into());
        };
        let bodies = doc.root().children("bodies").len();
        let b = match usize::try_from(body) {
            Ok(b) if b < bodies => b,
            _ => return argument(format!("body {body} does not exist ({bodies} in these results)")),
        };
        let (d, k) = r.r.body_columns(b)?;
        let t = HbTable::new().i64s("coordinates", &[d.len()], d.into_iter().map(|i| i as i64).collect()).i64s(
            "radiation_columns",
            &[k.len()],
            k.into_iter().map(|i| i as i64).collect(),
        );
        unsafe { give(out, t) }
    })
}

/// The quantities the results have: the column `names`, their names (texts) in the order of the
/// quantity table.
///
/// # Safety
/// `results` is results of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_available(results: *mut HbResults, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::results(results) }?;
        let have = r.r.available();
        let names: Vec<&str> = QuantityId::ALL.iter().filter(|q| have.contains(q)).map(|q| q.name()).collect();
        unsafe { give(out, HbTable::new().texts("names", &names)) }
    })
}

/// The solver route of the results ("given" for results made from arrays), freed by the caller
/// with `hb_string_free`.
///
/// # Safety
/// `results` is results of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_route(results: *mut HbResults, out: *mut *mut c_char) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::results(results) }?;
        unsafe { put(out, owned_text(r.r.route()), "out") }
    })
}

/// The diagnostics of the results (the warnings of the model, then of the solve, and the errors a
/// results file holds, severity 1), as `hb_model_diagnostics`.
///
/// # Safety
/// `results` is results of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_diagnostics(results: *mut HbResults, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::results(results) }?;
        unsafe { give(out, diagnostics_table(r.r.diagnostics())) }
    })
}

/// The meta data of the solve: present (0 for results not from a solve), hydrobem_version,
/// api_version, commit, compiler, linear_algebra, build_type, time, route, frequency_threads,
/// inner_threads, engine_seconds, environment_hooks (the texts of the environment hooks set for the
/// solve, `NAME` or `NAME=value`, for internal studies and tests only; none without).
///
/// # Safety
/// `results` is results of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_meta(results: *mut HbResults, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let m = unsafe { self::results(results) }?.r.meta();
        let n = m.engine_seconds.len();
        let t = HbTable::new()
            .bool("present", m.present)
            .text("hydrobem_version", &m.hydrobem_version)
            .i32("api_version", m.api_version)
            .text("commit", &m.commit)
            .text("compiler", &m.compiler)
            .text("linear_algebra", &m.linear_algebra)
            .text("build_type", &m.build_type)
            .text("time", &m.time)
            .text("route", &m.route)
            .i32("frequency_threads", m.frequency_threads)
            .i32("inner_threads", m.inner_threads)
            .f64s("engine_seconds", &[n], m.engine_seconds.clone())
            .texts("environment_hooks", &m.environment_hooks);
        unsafe { give(out, t) }
    })
}

/// A new model of the inputs of the results (`HB_ERR_STATE` for results without a model).
///
/// # Safety
/// `results` is results of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_model(results: *mut HbResults, out: *mut *mut HbModel) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::results(results) }?;
        let Some(doc) = r.r.document() else {
            return Err(Error::state("these results hold no model").into());
        };
        unsafe { give_model(out, Model::with_document(doc.clone())) }
    })
}

/// A new model of the model as it was solved: the enabled objects of the inputs, whose bodies,
/// lids and field points the quantities number.
///
/// # Safety
/// `results` is results of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_solved_model(results: *mut HbResults, out: *mut *mut HbModel) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::results(results) }?;
        let Some(doc) = r.r.solved_document() else {
            return Err(Error::state("these results hold no model").into());
        };
        unsafe { give_model(out, Model::with_document(doc.as_ref().clone())) }
    })
}

/// A quantity normalized as in the numeric files (reference/file-formats §1): the column values,
/// `HB_C128` for a complex quantity, else `HB_F64`.
///
/// # Safety
/// `results` is results of this library; `name` a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_normalized(
    results: *mut HbResults,
    name: *const c_char,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::results(results) }?;
        let q = quantity(unsafe { text(name, "name") }?)?;
        let (dims, v) = r.r.normalized(q)?;
        let dims: Vec<usize> = dims.iter().map(|&d| d.max(0) as usize).collect();
        let t = if q.element_type().is_complex() {
            HbTable::new().col("values", &dims, Data::C128(v))
        } else {
            HbTable::new().f64s("values", &dims, v.iter().map(|z| z.re).collect())
        };
        unsafe { give(out, t) }
    })
}

/// Writes the results file (.hbr).
///
/// # Safety
/// `results` is results of this library; `path` a UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_save(results: *mut HbResults, path: *const c_char, overwrite: i32) -> HbStatus {
    guard(|| {
        let r = unsafe { self::results(results) }?;
        let p = unsafe { self::path(path, "path") }?;
        Ok(r.r.save(&p, overwrite != 0)?)
    })
}

/// Writes the WAMIT-format numeric files and the .out summary of the stem (reference/file-formats
/// §2-§4): periods, or the frequencies when `omega` is nonzero; `stamp` the date of the headers
/// (null: now).
///
/// # Safety
/// `results` is results of this library; `stem` a UTF-8 string; `stamp` null or one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_write_wamit(
    results: *mut HbResults,
    stem: *const c_char,
    overwrite: i32,
    omega: i32,
    stamp: *const c_char,
) -> HbStatus {
    guard(|| {
        let r = unsafe { self::results(results) }?;
        let stem = unsafe { self::path(stem, "stem") }?;
        let stamp = unsafe { opt_text(stamp, "stamp") }?.map(String::from);
        let options = WamitOptions { overwrite: overwrite != 0, omega: omega != 0, stamp };
        r.r.write_wamit(&stem, &options)?;
        Ok(())
    })
}

/// New empty results of the present inputs of a model, to be filled by `hb_results_put` (results
/// a client computed).
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_new_results(model: *mut HbModel, out: *mut *mut HbResults) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::model(model) }?.with(|m| m.new_results())?;
        unsafe { give_results(out, r) }
    })
}

/// Puts a quantity into results: its elements in C order, of the quantity's element type (the
/// dtype of `hb_quantities`), `ndim` dimensions at `shape`. The views of these results taken before
/// are no longer valid.
///
/// # Safety
/// `results` is results of this library; `name` a UTF-8 string; `shape` ndim sizes; `data` their
/// product of elements of the quantity's type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_results_put(
    results: *mut HbResults,
    name: *const c_char,
    data: *const c_void,
    shape: *const usize,
    ndim: usize,
) -> HbStatus {
    guard(|| {
        let h = unsafe { self::results_mut(results) }?;
        let q = quantity(unsafe { text(name, "name") }?)?;
        let dims = unsafe { slice(shape, ndim, "shape") }?.to_vec();
        let n = elements(&dims, "shape")?;
        let d = match q.element_type() {
            ElementType::Float64 => QuantityData::Float64(unsafe { slice(data.cast::<f64>(), n, "data") }?.to_vec()),
            ElementType::Float32 => QuantityData::Float32(unsafe { slice(data.cast::<f32>(), n, "data") }?.to_vec()),
            ElementType::ComplexF32 => {
                QuantityData::ComplexF32(unsafe { slice(data.cast::<Complex32>(), n, "data") }?.to_vec())
            }
            ElementType::Int32 => QuantityData::Int32(unsafe { slice(data.cast::<i32>(), n, "data") }?.to_vec()),
            ElementType::ComplexF64 => {
                QuantityData::ComplexF64(unsafe { slice(data.cast::<Complex64>(), n, "data") }?.to_vec())
            }
            ElementType::Int64 => QuantityData::Int64(unsafe { slice(data.cast::<i64>(), n, "data") }?.to_vec()),
            ElementType::Text => return argument(format!("{} holds text", q.name())),
        };
        // the views of earlier calls stay valid until the results change (a refused put changes nothing)
        h.r.put(q, &dims, d)?;
        h.kept.lock().unwrap_or_else(|p| p.into_inner()).clear();
        Ok(())
    })
}
