//! The objects of a model and their items. An object is named by its id (`uint64_t`), which stays
//! valid while the object is in its model; the model refuses the id of a removed object. The items
//! are read as a table and set by their kind; the engine checks every value.

use std::ffi::c_char;

use hydrobem_core::{Error, ObjectId, Value};

use crate::model::{HbModel, model};
use crate::status::{
    HbStatus, Out, argument, elements, guard, opt_text, owned_text, put, slice, text, texts, writable,
};
use crate::table::{HbTable, give};

/// No object: the `body` of `hb_model_load_csf` when it names none, the parent of the model's root.
pub const HB_NO_OBJECT: u64 = u64::MAX;

pub fn id(bits: u64) -> ObjectId {
    ObjectId::from_bits(bits)
}

/// The id of the object of a path ("" the model, "options", "bodies/hull",
/// "bodies/hull/tanks/1").
///
/// # Safety
/// `model` is a model of this library; `path` a UTF-8 string; `object` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_object(model: *mut HbModel, path: *const c_char, object: *mut u64) -> HbStatus {
    guard(|| {
        writable(object, "object")?;
        let p = unsafe { text(path, "path") }?;
        let o = unsafe { self::model(model) }?.with(|m| m.object(p))?;
        unsafe { put(object, o.to_bits(), "object") }
    })
}

/// The object as a table: type, name, list (of its parent), index (zero-based, in that list), path
/// and parent (`HB_NO_OBJECT` for the model).
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_object_info(model: *mut HbModel, object: u64, out: *mut *mut HbTable) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let o = id(object);
        let (i, path, parent) = unsafe { self::model(model) }?.with(|m| {
            let i = m.object_info(o)?;
            Ok((i, m.path(o)?, m.parent(o)?))
        })?;
        let t = HbTable::new()
            .text("type", i.type_name)
            .text("name", &i.name)
            .text("list", &i.list)
            .i64("index", i.index as i64)
            .text("path", &path)
            .u64s("parent", &[], vec![parent.map_or(HB_NO_OBJECT, |p| p.to_bits())]);
        unsafe { give(out, t) }
    })
}

/// The objects of a list of an object: a table with the column ids (`HB_U64`, in list order).
///
/// # Safety
/// `model` is a model of this library; `list` a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_children(
    model: *mut HbModel,
    object: u64,
    list: *const c_char,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let list = unsafe { text(list, "list") }?;
        let ids = unsafe { self::model(model) }?.with(|m| Ok(m.children(id(object), list)?.to_vec()))?;
        let n = ids.len();
        unsafe { give(out, HbTable::new().u64s("ids", &[n], ids.into_iter().map(|i| i.to_bits()).collect())) }
    })
}

/// A new object in a list of an object: of a type (null: the list's first type), with a name
/// (null or "": none, in a list whose objects may be unnamed), at a zero-based place (-1: the end;
/// another negative index is refused).
///
/// # Safety
/// `model` is a model of this library; `list` a UTF-8 string; `type_name` and `name` null or UTF-8
/// strings; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_add(
    model: *mut HbModel,
    parent: u64,
    list: *const c_char,
    type_name: *const c_char,
    name: *const c_char,
    index: i64,
    out: *mut u64,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let list = unsafe { text(list, "list") }?;
        let ty = unsafe { opt_text(type_name, "type_name") }?;
        let name = unsafe { opt_text(name, "name") }?.unwrap_or("");
        let at = match index {
            -1 => None,
            i if i < 0 => return argument(format!("index must be -1 (the end) or a zero-based place, not {i}")),
            i => Some(usize::try_from(i).unwrap_or(usize::MAX)),
        };
        let o = unsafe { self::model(model) }?.with(|m| m.add(id(parent), list, ty, name, at))?;
        unsafe { put(out, o.to_bits(), "out") }
    })
}

/// Removes an object and its children.
///
/// # Safety
/// `model` is a model of this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_remove(model: *mut HbModel, object: u64) -> HbStatus {
    guard(|| unsafe { self::model(model) }?.with(|m| m.remove(id(object))))
}

/// Renames an object ("" removes the name of an object of a list with optional names).
///
/// # Safety
/// `model` is a model of this library; `name` a UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_rename(model: *mut HbModel, object: u64, name: *const c_char) -> HbStatus {
    guard(|| {
        let name = unsafe { text(name, "name") }?;
        unsafe { self::model(model) }?.with(|m| m.rename(id(object), name))
    })
}

/// Moves an object to a zero-based place of its list; the order of a list is the order of its
/// objects in the results and the files. A negative index or one past the last object is
/// `HB_ERR_ARGUMENT`, its message naming the index and the number of objects of the list.
///
/// # Safety
/// `model` is a model of this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_move(model: *mut HbModel, object: u64, index: i64) -> HbStatus {
    guard(|| {
        let o = id(object);
        unsafe { self::model(model) }?.with(|m| {
            let list = m.object_info(o)?.list;
            if let Some(parent) = m.parent(o)? {
                let count = m.children(parent, &list)?.len();
                if usize::try_from(index).map_or(true, |i| i >= count) {
                    return Err(Error::argument(format!(
                        "{list}: index {index} out of range (the list has {count} objects)"
                    )));
                }
            }
            m.move_to(o, usize::try_from(index).unwrap_or(usize::MAX))
        })
    })
}

/// Removes every object of a list of an object, as `clear()` of the package: all or nothing,
/// refused (nothing removed) while another object of the model refers to one of them.
///
/// # Safety
/// `model` is a model of this library; `list` a UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_clear(model: *mut HbModel, object: u64, list: *const c_char) -> HbStatus {
    guard(|| {
        let list = unsafe { text(list, "list") }?;
        unsafe { self::model(model) }?.with(|m| m.clear_list(id(object), list))
    })
}

/// The kind and state of an item as a table: kind (`bool`, `int`, `real`, `text`, `choice`,
/// `array`, `reference`, `origin`, `terms`), is_set, has_value, is_array (0 or 1), dtype (`real`,
/// `int`, `text`: the element type of an array value, `real` for the scalars), shape (the
/// dimensions of its value) and is_choice.
///
/// # Safety
/// `model` is a model of this library; `item` a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_item_info(
    model: *mut HbModel,
    object: u64,
    item: *const c_char,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let item = unsafe { text(item, "item") }?;
        let i = unsafe { self::model(model) }?.with(|m| m.item_info(id(object), item))?;
        let n = i.shape.len();
        let t = HbTable::new()
            .text("kind", i.kind.name())
            .bool("is_set", i.is_set)
            .bool("has_value", i.has_value)
            .bool("is_array", i.is_array)
            .text("dtype", i.dtype.name())
            .i64s("shape", &[n], i.shape.iter().map(|&d| d as i64).collect())
            .bool("is_choice", i.is_choice);
        unsafe { give(out, t) }
    })
}

/// The value of an item (its default when it is not set) as a table: the text column kind, then
/// by kind the column value: `none` (no value: no column), `bool` (`HB_I32` 0 or 1), `int`
/// (`HB_I64`), `real` (`HB_F64`), `text` (a text scalar: a text, choice, reference, an origin or
/// number given by name), `texts` (a text array), `reals` (`HB_F64` of the item's shape), `ints`
/// (`HB_I64`; the ranges of a surface [n, 2], its one-based first and last stored elements).
///
/// # Safety
/// `model` is a model of this library; `item` a UTF-8 string; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_get(
    model: *mut HbModel,
    object: u64,
    item: *const c_char,
    out: *mut *mut HbTable,
) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let item = unsafe { text(item, "item") }?;
        let v = unsafe { self::model(model) }?.with(|m| m.get(id(object), item))?;
        let t = match v {
            None => HbTable::new().text("kind", "none"),
            Some(Value::Bool(b)) => HbTable::new().text("kind", "bool").bool("value", b),
            Some(Value::Int(x)) => HbTable::new().text("kind", "int").i64("value", x),
            Some(Value::Real(x)) => HbTable::new().text("kind", "real").f64("value", x),
            Some(Value::Text(s)) => HbTable::new().text("kind", "text").text("value", &s),
            Some(Value::Texts(v)) => HbTable::new().text("kind", "texts").texts("value", &v),
            Some(Value::Reals { shape, data }) => HbTable::new().text("kind", "reals").f64s("value", &shape, data),
            Some(Value::Ints { shape, data }) => HbTable::new().text("kind", "ints").i64s("value", &shape, data),
        };
        unsafe { give(out, t) }
    })
}

unsafe fn set(model: *mut HbModel, object: u64, item: *const c_char, v: impl FnOnce() -> Out<Value>) -> HbStatus {
    guard(|| {
        let item = unsafe { text(item, "item") }?;
        let value = v()?;
        unsafe { self::model(model) }?.with(|m| m.set(id(object), item, value))
    })
}

// the shape of an array value and its number of elements (refused beyond 2^40)
fn shape_of(shape: &[usize]) -> Out<(Vec<usize>, usize)> {
    if shape.is_empty() {
        return argument("an array needs at least one dimension");
    }
    Ok((shape.to_vec(), elements(shape, "shape")?))
}

/// Sets a bool item (0 false, else true).
///
/// # Safety
/// `model` is a model of this library; `item` a UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_set_bool(model: *mut HbModel, object: u64, item: *const c_char, value: i32) -> HbStatus {
    unsafe { set(model, object, item, || Ok(Value::Bool(value != 0))) }
}

/// Sets an int item.
///
/// # Safety
/// `model` is a model of this library; `item` a UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_set_int(model: *mut HbModel, object: u64, item: *const c_char, value: i64) -> HbStatus {
    unsafe { set(model, object, item, || Ok(Value::Int(value))) }
}

/// Sets a real item.
///
/// # Safety
/// `model` is a model of this library; `item` a UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_set_real(model: *mut HbModel, object: u64, item: *const c_char, value: f64) -> HbStatus {
    unsafe { set(model, object, item, || Ok(Value::Real(value))) }
}

/// Sets an item of text: a text, choice or reference item, an origin or a number item by one of
/// its names, a mesh reference to every element of a mesh.
///
/// # Safety
/// `model` is a model of this library; `item` and `value` UTF-8 strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_set_text(
    model: *mut HbModel,
    object: u64,
    item: *const c_char,
    value: *const c_char,
) -> HbStatus {
    unsafe { set(model, object, item, || Ok(Value::Text(text(value, "value")?.to_string()))) }
}

/// Sets a text array item of n strings.
///
/// # Safety
/// `model` is a model of this library; `item` a UTF-8 string; `values` n UTF-8 strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_set_texts(
    model: *mut HbModel,
    object: u64,
    item: *const c_char,
    values: *const *const c_char,
    n: usize,
) -> HbStatus {
    unsafe { set(model, object, item, || Ok(Value::Texts(texts(values, n, "values")?))) }
}

/// Sets a real array item: the elements in C order, `ndim` dimensions at `shape`.
///
/// # Safety
/// `model` is a model of this library; `item` a UTF-8 string; `shape` ndim sizes; `data` their
/// product of elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_set_reals(
    model: *mut HbModel,
    object: u64,
    item: *const c_char,
    data: *const f64,
    shape: *const usize,
    ndim: usize,
) -> HbStatus {
    unsafe {
        set(model, object, item, || {
            let (shape, n) = shape_of(slice(shape, ndim, "shape")?)?;
            Ok(Value::Reals { shape, data: slice(data, n, "data")?.to_vec() })
        })
    }
}

/// Sets an integer array item: the elements in C order, `ndim` dimensions at `shape`.
///
/// # Safety
/// As `hb_set_reals`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_set_ints(
    model: *mut HbModel,
    object: u64,
    item: *const c_char,
    data: *const i64,
    shape: *const usize,
    ndim: usize,
) -> HbStatus {
    unsafe {
        set(model, object, item, || {
            let (shape, n) = shape_of(slice(shape, ndim, "shape")?)?;
            Ok(Value::Ints { shape, data: slice(data, n, "data")?.to_vec() })
        })
    }
}

/// Returns an item to its default (or unsets it).
///
/// # Safety
/// `model` is a model of this library; `item` a UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_reset(model: *mut HbModel, object: u64, item: *const c_char) -> HbStatus {
    guard(|| {
        let item = unsafe { text(item, "item") }?;
        unsafe { self::model(model) }?.with(|m| m.reset_item(id(object), item))
    })
}

/// The route the solver takes for the model (after the rules), as a text the caller frees with
/// `hb_string_free`.
///
/// # Safety
/// `model` is a model of this library; `out` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_model_route(model: *mut HbModel, out: *mut *mut c_char) -> HbStatus {
    guard(|| {
        writable(out, "out")?;
        let r = unsafe { self::model(model) }?.with(|m| m.route())?.name();
        unsafe { put(out, owned_text(&r), "out") }
    })
}
