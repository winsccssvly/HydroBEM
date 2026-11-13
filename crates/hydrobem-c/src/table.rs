//! `HbTable`: the answer of a query as named columns, each an array of one element type with its
//! shape (a scalar: no dimensions), in C order. A table owns its elements; they stay valid until
//! `hb_table_free`.

use std::ffi::{CString, c_char, c_void};

use hydrobem_core::{Complex32, Complex64};

use crate::status::{HbStatus, Out, argument, guard, put, text, writable};

/// The element type of a column (`HbColumn::dtype`; codes of this ABI, not the element codes of
/// the results file): 64-bit float.
pub const HB_F64: i32 = 1;
/// 32-bit float.
pub const HB_F32: i32 = 2;
/// 32-bit signed integer.
pub const HB_I32: i32 = 3;
/// 64-bit signed integer.
pub const HB_I64: i32 = 4;
/// 64-bit unsigned integer (object ids).
pub const HB_U64: i32 = 5;
/// Complex of two 32-bit floats (real, imaginary).
pub const HB_C64: i32 = 6;
/// Complex of two 64-bit floats (real, imaginary).
pub const HB_C128: i32 = 7;
/// Text: `data` is an array of `count` NUL-terminated UTF-8 strings (`const char *const *`).
pub const HB_TEXT: i32 = 8;

/// A column of a table, or the view of a result quantity: its name, element type (`HB_F64`, ...),
/// dimensions (`ndim` of them at `shape`; none for a scalar), the number of elements (the product
/// of the dimensions) and the elements in C order.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HbColumn {
    pub name: *const c_char,
    pub dtype: i32,
    pub ndim: usize,
    pub shape: *const usize,
    pub count: usize,
    pub data: *const c_void,
}

pub enum Data {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    C64(Vec<Complex32>),
    C128(Vec<Complex64>),
    Text(Vec<CString>, Vec<*const c_char>),
}

struct Col {
    name: CString,
    shape: Vec<usize>,
    data: Data,
}

/// The answer of a query: named columns.
#[derive(Default)]
pub struct HbTable {
    cols: Vec<Col>,
}

fn cstring(s: &str) -> CString {
    CString::new(s.split('\0').next().unwrap_or("")).unwrap_or_default()
}

impl HbTable {
    pub fn new() -> HbTable {
        HbTable::default()
    }

    /// A column of the shape (a scalar for none) with its elements.
    pub fn col(mut self, name: &str, shape: &[usize], data: Data) -> HbTable {
        self.cols.push(Col { name: cstring(name), shape: shape.to_vec(), data });
        self
    }
    pub fn f64s(self, name: &str, shape: &[usize], v: Vec<f64>) -> HbTable {
        self.col(name, shape, Data::F64(v))
    }
    pub fn f64(self, name: &str, v: f64) -> HbTable {
        self.col(name, &[], Data::F64(vec![v]))
    }
    pub fn i32s(self, name: &str, shape: &[usize], v: Vec<i32>) -> HbTable {
        self.col(name, shape, Data::I32(v))
    }
    pub fn i32(self, name: &str, v: i32) -> HbTable {
        self.col(name, &[], Data::I32(vec![v]))
    }
    pub fn i64s(self, name: &str, shape: &[usize], v: Vec<i64>) -> HbTable {
        self.col(name, shape, Data::I64(v))
    }
    pub fn i64(self, name: &str, v: i64) -> HbTable {
        self.col(name, &[], Data::I64(vec![v]))
    }
    pub fn u64s(self, name: &str, shape: &[usize], v: Vec<u64>) -> HbTable {
        self.col(name, shape, Data::U64(v))
    }
    /// A text column of the strings (one dimension; `text` a scalar).
    pub fn texts(self, name: &str, v: &[impl AsRef<str>]) -> HbTable {
        let n = v.len();
        self.text_col(name, &[n], v)
    }
    pub fn text(self, name: &str, v: &str) -> HbTable {
        self.text_col(name, &[], &[v])
    }
    fn text_col(self, name: &str, shape: &[usize], v: &[impl AsRef<str>]) -> HbTable {
        let owned: Vec<CString> = v.iter().map(|s| cstring(s.as_ref())).collect();
        let ptrs = owned.iter().map(|s| s.as_ptr()).collect();
        self.col(name, shape, Data::Text(owned, ptrs))
    }
    pub fn bool(self, name: &str, v: bool) -> HbTable {
        self.i32(name, i32::from(v))
    }
}

/// The column view of elements: dtype, count and the pointer to them.
pub fn view(data: &Data) -> (i32, usize, *const c_void) {
    match data {
        Data::F64(v) => (HB_F64, v.len(), v.as_ptr().cast()),
        Data::F32(v) => (HB_F32, v.len(), v.as_ptr().cast()),
        Data::I32(v) => (HB_I32, v.len(), v.as_ptr().cast()),
        Data::I64(v) => (HB_I64, v.len(), v.as_ptr().cast()),
        Data::U64(v) => (HB_U64, v.len(), v.as_ptr().cast()),
        Data::C64(v) => (HB_C64, v.len(), v.as_ptr().cast()),
        Data::C128(v) => (HB_C128, v.len(), v.as_ptr().cast()),
        Data::Text(_, p) => (HB_TEXT, p.len(), p.as_ptr().cast()),
    }
}

fn column_of(c: &Col) -> HbColumn {
    let (dtype, count, data) = view(&c.data);
    HbColumn { name: c.name.as_ptr(), dtype, ndim: c.shape.len(), shape: c.shape.as_ptr(), count, data }
}

/// Hands a table to the caller.
///
/// # Safety
/// `out` is null or a writable pointer.
pub unsafe fn give(out: *mut *mut HbTable, t: HbTable) -> Out<()> {
    // refused before the box is made: nothing leaks
    writable(out, "out")?;
    unsafe { put(out, Box::into_raw(Box::new(t)), "out") }
}

unsafe fn table<'a>(t: *const HbTable) -> Out<&'a HbTable> {
    if t.is_null() {
        return argument("table is a null pointer");
    }
    // SAFETY: a table of this library, not freed (the caller's contract)
    Ok(unsafe { &*t })
}

/// The number of columns of a table.
///
/// # Safety
/// `table` is a table of this library; `count` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_table_count(table: *const HbTable, count: *mut usize) -> HbStatus {
    guard(|| {
        writable(count, "count")?;
        let t = unsafe { self::table(table) }?;
        unsafe { put(count, t.cols.len(), "count") }
    })
}

/// The column of a zero-based index; its pointers are valid until the table is freed.
///
/// # Safety
/// `table` is a table of this library; `column` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_table_column(table: *const HbTable, index: usize, column: *mut HbColumn) -> HbStatus {
    guard(|| {
        writable(column, "column")?;
        let t = unsafe { self::table(table) }?;
        let Some(c) = t.cols.get(index) else {
            return argument(format!("column {index} does not exist (the table has {})", t.cols.len()));
        };
        unsafe { put(column, column_of(c), "column") }
    })
}

/// The column of a name (`HB_ERR_ARGUMENT` when the table has none of it).
///
/// # Safety
/// `table` is a table of this library; `name` a string; `column` a writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_table_find(table: *const HbTable, name: *const c_char, column: *mut HbColumn) -> HbStatus {
    guard(|| {
        writable(column, "column")?;
        let t = unsafe { self::table(table) }?;
        let name = unsafe { text(name, "name") }?;
        let Some(c) = t.cols.iter().find(|c| c.name.to_bytes() == name.as_bytes()) else {
            return argument(format!("the table has no column {name:?}"));
        };
        unsafe { put(column, column_of(c), "column") }
    })
}

/// Frees a table (null: nothing).
///
/// # Safety
/// `table` is null or a table of this library not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hb_table_free(table: *mut HbTable) {
    if !table.is_null() {
        // SAFETY: made by Box::into_raw in give
        drop(unsafe { Box::from_raw(table) });
    }
}
