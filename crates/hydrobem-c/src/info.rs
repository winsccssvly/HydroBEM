//! The functions that need no model: the version of the ABI, the build, the schema of the model
//! objects, the result quantities, the names of the engine's codes, the wave number; the errors of
//! the last failed call; the texts the library hands out.

use std::ffi::{CString, c_char};

use hydrobem_core::{DType, ItemKind, PanelRole, QuantityId, schema};

use crate::results::dtype_of;
use crate::status::{HbLocation, HbStatus, guard, last_location, last_message, put, writable};
use crate::table::{HbTable, give};

/// The version of this ABI: raised with every change of a function, a struct or a constant of
/// the header (`HB_ABI_VERSION` of the header the caller was built with).
pub const HB_ABI_VERSION: u32 = 8;

/// An element with the wrong orientation (`flags` of the mesh quality).
pub const HB_QUALITY_ORIENTATION: i32 = 1;
/// An element with a free edge.
pub const HB_QUALITY_FREE_EDGE: i32 = 2;
/// An element with a non-manifold edge.
pub const HB_QUALITY_NON_MANIFOLD: i32 = 4;
/// An element coarse for the shortest wavelength.
pub const HB_QUALITY_COARSE: i32 = 8;

/// The version of the ABI of the library (compare with `HB_ABI_VERSION`).
#[unsafe(no_mangle)]
pub extern "C" fn hb_abi_version() -> u32 {
    HB_ABI_VERSION
}

/// Frees a text the library handed out (null: nothing).
///
/// # Safety
/// `text` is null or a text of this library not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_string_free(text: *mut c_char) {
    if !text.is_null() {
        // SAFETY: made by CString::into_raw (status::owned_text)
        drop(unsafe { CString::from_raw(text) });
    }
}

/// The message of the last failed call of this thread (UTF-8; empty when none failed), owned by
/// the library and valid until the next call of this thread fails.
#[unsafe(no_mangle)]
pub extern "C" fn hb_last_error_message() -> *const c_char {
    last_message()
}

/// The location of the error of the last failed call of this thread. A null `location` is
/// `HB_ERR_ARGUMENT` and keeps the last error as it was (this call sets no error of its own).
///
/// # Safety
/// `location` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_last_error_location(location: *mut HbLocation) -> HbStatus {
    let l = last_location();
    if location.is_null() {
        return crate::status::HB_ERR_ARGUMENT;
    }
    // SAFETY: a writable location (the caller's contract)
    unsafe { location.write(l) };
    crate::status::HB_OK
}

/// The build of the engine: version, commit, compiler, linear_algebra, build_type (texts),
/// api_version (of the Rust API) and avx2 (0 or 1).
///
/// # Safety
/// `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_build_info(out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let b = hydrobem_core::build_info();
        let t = HbTable::new()
            .text("version", &b.version)
            .text("commit", &b.commit)
            .text("compiler", &b.compiler)
            .text("linear_algebra", &b.linear_algebra)
            .text("build_type", &b.build_type)
            .i32("api_version", b.api_version)
            .bool("avx2", b.avx2);
        unsafe { give(out, t) }
    })
}

/// The items of the object types of the model (reference/inputs-*), one row per item: type, item, kind
/// (as `hb_item_info`), dtype, shape (the rule, e.g. `N,3`), unit, default (a schema literal, ""
/// none), choices (separated by spaces), target, kinds (the mesh kinds, separated by spaces),
/// switch (the switch items that choose the item, separated by spaces; empty: none), switch_values
/// (the values of the switches that take it, separated by spaces, each "v" of the first switch or
/// "v/w" of the first and the second; empty: always), description.
///
/// # Safety
/// `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_schema_items(out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let mut col: [Vec<String>; 13] = Default::default();
        for t in schema() {
            for d in &t.items {
                let dtype = if d.kind == ItemKind::Array { d.dtype } else { DType::Real };
                let row = [
                    t.name,
                    d.name,
                    d.kind.name(),
                    dtype.name(),
                    d.shape,
                    d.unit,
                    d.fallback.unwrap_or(""),
                    &d.choices.join(" "),
                    d.target,
                    d.kinds,
                    d.switch,
                    d.switch_values,
                    d.description,
                ]
                .map(String::from);
                for (c, v) in col.iter_mut().zip(row) {
                    c.push(v);
                }
            }
        }
        let names = [
            "type",
            "item",
            "kind",
            "dtype",
            "shape",
            "unit",
            "default",
            "choices",
            "target",
            "kinds",
            "switch",
            "switch_values",
            "description",
        ];
        let t = names.iter().zip(&col).fold(HbTable::new(), |t, (n, c)| t.texts(n, c));
        unsafe { give(out, t) }
    })
}

/// The lists of the object types, one row per list: type, list, types (the object types it holds,
/// separated by spaces), description (texts), named, optional_names, single, always (0 or 1).
///
/// # Safety
/// `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_schema_lists(out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let (mut ty, mut list, mut types, mut desc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut named, mut optional, mut single, mut always) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for t in schema() {
            for l in &t.lists {
                ty.push(t.name);
                list.push(l.name);
                types.push(l.types.join(" "));
                desc.push(l.description);
                named.push(i32::from(l.named));
                optional.push(i32::from(l.optional_names));
                single.push(i32::from(l.single));
                always.push(i32::from(l.always));
            }
        }
        let n = ty.len();
        let t = HbTable::new()
            .texts("type", &ty)
            .texts("list", &list)
            .texts("types", &types)
            .texts("description", &desc)
            .i32s("named", &[n], named)
            .i32s("optional_names", &[n], optional)
            .i32s("single", &[n], single)
            .i32s("always", &[n], always);
        unsafe { give(out, t) }
    })
}

/// The result quantities in the order of their codes: name, dims (the axes, separated by commas),
/// unit, description (texts) and dtype (the column type of its elements, 0 for text).
///
/// # Safety
/// `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_quantities(out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let q = QuantityId::ALL;
        let n = q.len();
        let t = HbTable::new()
            .texts("name", &q.iter().map(|q| q.name()).collect::<Vec<_>>())
            .texts("dims", &q.iter().map(|q| q.dims()).collect::<Vec<_>>())
            .texts("unit", &q.iter().map(|q| q.unit()).collect::<Vec<_>>())
            .texts("description", &q.iter().map(|q| q.description()).collect::<Vec<_>>())
            .i32s("dtype", &[n], q.iter().map(|q| dtype_of(q.element_type()).unwrap_or(0)).collect());
        unsafe { give(out, t) }
    })
}

/// The views of the result tree (reference/results §1), one row per view in the order of the
/// quantity table: path (e.g. `drift.pressure_integration`), quantity (its name), owner (the rule of
/// an owner's part: shared, global, columns, coordinates, body_points, lid, pressure_surface, tank;
/// texts)
/// and width (the columns of a body of the rule columns, else 0).
///
/// # Safety
/// `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_result_views(out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let v = hydrobem_core::result_views();
        let n = v.len();
        let t = HbTable::new()
            .texts("path", &v.iter().map(|v| v.path).collect::<Vec<_>>())
            .texts("quantity", &v.iter().map(|v| v.quantity.name()).collect::<Vec<_>>())
            .texts("owner", &v.iter().map(|v| v.owner.name()).collect::<Vec<_>>())
            .i32s("width", &[n], v.iter().map(|v| v.owner.width() as i32).collect());
        unsafe { give(out, t) }
    })
}

/// The groups of the result tree (its inner nodes, parents first): path, description.
///
/// # Safety
/// `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_result_groups(out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let g = hydrobem_core::result_groups();
        let t = HbTable::new()
            .texts("path", &g.iter().map(|g| g.path).collect::<Vec<_>>())
            .texts("description", &g.iter().map(|g| g.description).collect::<Vec<_>>());
        unsafe { give(out, t) }
    })
}

/// The names of the panel roles by their values (the role column of the panels): names.
///
/// # Safety
/// `out` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_panel_roles(out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let names: Vec<&str> = (0..).map_while(PanelRole::from_value).map(PanelRole::name).collect();
        unsafe { give(out, HbTable::new().texts("names", &names)) }
    })
}

/// The wave number k of omega² = g k tanh(k h) (h = depth, infinite allowed).
///
/// # Safety
/// `k` is a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_wavenumber(omega: f64, depth: f64, gravity: f64, k: *mut f64) -> HbStatus {
    guard(|| {
        writable(k, "k")?;
        let v = hydrobem_core::wavenumber(omega, depth, gravity)?;
        unsafe { put(k, v, "k") }
    })
}
