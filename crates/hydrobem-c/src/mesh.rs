//! The meshes the engine reads, makes and writes (reference/inputs-meshes): each a new mesh object of the model,
//! named by its id.

use std::ffi::c_char;

use hydrobem_core::{AnalyticShape, MeshFile};

use crate::model::{HbModel, model};
use crate::object::{HB_NO_OBJECT, id};
use crate::status::{
    HbStatus, Out, argument, elements, guard, opt_text, owned_text, path, put, slice, text, texts, writable,
};

/// The options of a mesh file read by `hb_model_load_mesh` (reference/inputs-meshes); a zero or null field is not
/// given. higher_order and zero_trailer: nonzero for true; spl_path: the SPL file of a B-spline
/// GDF; discretization: `discretization_rows` rows of 6 integers (nu, nv, the spline orders and the
/// quadrature orders); panel_size: the panel size of the discretization (> 0) with the orders
/// spline_order, outer_quadrature and inner_quadrature.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HbMeshOptions {
    pub higher_order: i32,
    pub zero_trailer: i32,
    pub spl_path: *const c_char,
    pub discretization: *const i32,
    pub discretization_rows: usize,
    pub panel_size: f64,
    pub spline_order: i32,
    pub outer_quadrature: i32,
    pub inner_quadrature: i32,
}

/// A named analytic shape (reference/inputs-meshes §2): the shape's name; `dimensions` pairs of a name and its
/// value (meters); `disc_count` discs (x, y, radius) of the dipole shapes at `discs`; `part_count`
/// part names (none: the shape's parts); the mapping (`uniform` or `nonuniform`);
/// `discretization_rows` rows of 6 integers; the reference length; the symmetry (`none`, `yz`,
/// `xz`, `xz_and_yz`); the label (null: none).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HbAnalyticShape {
    pub shape: *const c_char,
    pub dimension_names: *const *const c_char,
    pub dimension_values: *const f64,
    pub dimensions: usize,
    pub discs: *const f64,
    pub disc_count: usize,
    pub parts: *const *const c_char,
    pub part_count: usize,
    pub mapping: *const c_char,
    pub discretization: *const i32,
    pub discretization_rows: usize,
    pub reference_length: f64,
    pub symmetry: *const c_char,
    pub label: *const c_char,
}

unsafe fn rows6(p: *const i32, n: usize, what: &str) -> Out<Vec<[i32; 6]>> {
    let v = unsafe { slice(p, elements(&[n, 6], what)?, what) }?;
    Ok(v.chunks_exact(6).map(|r| [r[0], r[1], r[2], r[3], r[4], r[5]]).collect())
}

fn given(v: i32) -> Option<i32> {
    if v == 0 { None } else { Some(v) }
}

unsafe fn mesh_file(p: std::path::PathBuf, o: *const HbMeshOptions) -> Out<MeshFile> {
    let mut f = MeshFile { path: p, ..MeshFile::default() };
    if o.is_null() {
        return Ok(f);
    }
    // SAFETY: an options struct (the caller's contract)
    let o = unsafe { &*o };
    f.higher_order = o.higher_order != 0;
    f.zero_trailer = o.zero_trailer != 0;
    f.spl_path = unsafe { opt_text(o.spl_path, "spl_path") }?.map(Into::into);
    if o.discretization_rows > 0 {
        f.discretization = Some(unsafe { rows6(o.discretization, o.discretization_rows, "discretization") }?);
    }
    if o.panel_size != 0.0 {
        f.panel_size = Some(o.panel_size);
    }
    f.spline_order = given(o.spline_order);
    f.outer_quadrature = given(o.outer_quadrature);
    f.inner_quadrature = given(o.inner_quadrature);
    Ok(f)
}

/// A mesh read from a file: a GDF file (with its SPL file), or a geometry file `.hbg`; `options`
/// null for none.
///
/// # Safety
/// `model` is a model of this library; `name` and `path` UTF-8 strings; `options` null or an
/// options struct; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_load_mesh(
    model: *mut HbModel,
    name: *const c_char,
    path: *const c_char,
    options: *const HbMeshOptions,
    out: *mut u64,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let name = unsafe { text(name, "name") }?;
        let file = unsafe { mesh_file(self::path(path, "path")?, options) }?;
        let o = unsafe { self::model(model) }?.with(|m| m.load_mesh(name, &file))?;
        unsafe { put(out, o.to_bits(), "out") }
    })
}

/// The panels of a free-surface file (FDF, reference/inputs-meshes) as a mesh of panels in z = 0, for
/// `second_order.free_surface_mesh` (free_surface_method "mesh"): `symmetry` that of the model the file
/// belongs to ("none", "yz", "xz", "xz_and_yz"; null: "none"; the file holds none). Only the panels
/// are read.
///
/// # Safety
/// `model` is a model of this library; `name` and `path` UTF-8 strings; `symmetry` null or one;
/// `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_load_fdf(
    model: *mut HbModel,
    name: *const c_char,
    path: *const c_char,
    symmetry: *const c_char,
    out: *mut u64,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let name = unsafe { text(name, "name") }?;
        let p = unsafe { self::path(path, "path") }?;
        let symmetry = unsafe { opt_text(symmetry, "symmetry") }?.unwrap_or("none");
        let o = unsafe { self::model(model) }?.with(|m| m.load_fdf(name, &p, symmetry))?;
        unsafe { put(out, o.to_bits(), "out") }
    })
}

/// A control surface read from a CSF file; `body` the body that places the automatic surface, or
/// `HB_NO_OBJECT`.
///
/// # Safety
/// `model` is a model of this library; `name` and `path` UTF-8 strings; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_load_csf(
    model: *mut HbModel,
    name: *const c_char,
    path: *const c_char,
    zero_trailer: i32,
    body: u64,
    out: *mut u64,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let name = unsafe { text(name, "name") }?;
        let p = unsafe { self::path(path, "path") }?;
        let body = if body == HB_NO_OBJECT { None } else { Some(id(body)) };
        let o = unsafe { self::model(model) }?.with(|m| m.load_csf(name, &p, zero_trailer != 0, body))?;
        unsafe { put(out, o.to_bits(), "out") }
    })
}

/// A mesh from the text of a geometry file (.hbg, reference/inputs-meshes §5).
///
/// # Safety
/// `model` is a model of this library; `name` and `content` UTF-8 strings; `out` a writable
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_mesh_from_hbg(
    model: *mut HbModel,
    name: *const c_char,
    content: *const c_char,
    out: *mut u64,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let name = unsafe { text(name, "name") }?;
        let content = unsafe { text(content, "content") }?;
        let o = unsafe { self::model(model) }?.with(|m| m.hbg_mesh(name, content))?;
        unsafe { put(out, o.to_bits(), "out") }
    })
}

/// The text of the geometry file (.hbg) of a mesh, freed by the caller with `hb_string_free`.
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_mesh_hbg(model: *mut HbModel, mesh: u64, out: *mut *mut c_char) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let s = unsafe { self::model(model) }?.with(|m| m.mesh_hbg(id(mesh)))?;
        unsafe { put(out, owned_text(&s), "out") }
    })
}

/// A mesh of a named analytic shape.
///
/// # Safety
/// `model` is a model of this library; `name` a UTF-8 string; `shape` a shape struct whose arrays
/// hold their counts; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_analytic_mesh(
    model: *mut HbModel,
    name: *const c_char,
    shape: *const HbAnalyticShape,
    out: *mut u64,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let name = unsafe { text(name, "name") }?;
        if shape.is_null() {
            return argument("shape is a null pointer");
        }
        // SAFETY: a shape struct (the caller's contract)
        let s = unsafe { &*shape };
        let names = unsafe { texts(s.dimension_names, s.dimensions, "dimension_names") }?;
        let values = unsafe { slice(s.dimension_values, s.dimensions, "dimension_values") }?;
        let discs = unsafe { slice(s.discs, elements(&[s.disc_count, 3], "discs")?, "discs") }?;
        let a = AnalyticShape {
            shape: unsafe { text(s.shape, "shape") }?.to_string(),
            dimensions: names.into_iter().zip(values.iter().copied()).collect(),
            discs: discs.chunks_exact(3).map(|d| [d[0], d[1], d[2]]).collect(),
            parts: unsafe { texts(s.parts, s.part_count, "parts") }?,
            mapping: unsafe { text(s.mapping, "mapping") }?.to_string(),
            discretization: unsafe { rows6(s.discretization, s.discretization_rows, "discretization") }?,
            reference_length: s.reference_length,
            symmetry: unsafe { text(s.symmetry, "symmetry") }?.to_string(),
            label: unsafe { opt_text(s.label, "label") }?.unwrap_or("").to_string(),
        };
        let o = unsafe { self::model(model) }?.with(|m| m.analytic_mesh(name, &a))?;
        unsafe { put(out, o.to_bits(), "out") }
    })
}

/// A new mesh of the surface of one role of a body as the engine uses it (after its trim).
///
/// # Safety
/// `model` is a model of this library; `name` and `role` UTF-8 strings; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_body_trimmed_mesh(
    model: *mut HbModel,
    body: u64,
    name: *const c_char,
    role: *const c_char,
    out: *mut u64,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let name = unsafe { text(name, "name") }?;
        let role = unsafe { text(role, "role") }?;
        let o = unsafe { self::model(model) }?.with(|m| m.body_trimmed_mesh(id(body), name, role))?;
        unsafe { put(out, o.to_bits(), "out") }
    })
}

/// Writes a mesh: a path ending in `.hbg` the geometry file, otherwise GDF (with the SPL file at
/// `spl_path`, null for none); an existing file is kept unless `overwrite` is nonzero.
///
/// # Safety
/// `model` is a model of this library; `path` a UTF-8 string; `spl_path` null or one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_mesh_write(
    model: *mut HbModel,
    mesh: u64,
    path: *const c_char,
    spl_path: *const c_char,
    overwrite: i32,
) -> HbStatus {
    guard(|| {
        let p = unsafe { self::path(path, "path") }?;
        let spl = unsafe { opt_text(spl_path, "spl_path") }?.map(std::path::PathBuf::from);
        unsafe { self::model(model) }?.with(|m| m.mesh_write(id(mesh), &p, spl.as_deref(), overwrite != 0))
    })
}
