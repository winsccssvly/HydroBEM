//! The status of a call and the error of the last failed call of a thread: every function of the
//! ABI returns a status; a failure keeps its message and location for `hb_last_error_message` and
//! `hb_last_error_location` on the calling thread. A panic of the engine (a bug) is caught at the
//! boundary and returned as `HB_ERR_INTERNAL`; it never unwinds into the caller.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::OnceLock;

use hydrobem_core::{Error, ErrorKind, Location};

/// The status of a call: `HB_OK` or one of the `HB_ERR_` codes.
pub type HbStatus = i32;

/// The call succeeded.
pub const HB_OK: HbStatus = 0;
/// A value is invalid (an item, an argument the engine checks, a file's content).
pub const HB_ERR_VALUE: HbStatus = 1;
/// A value has the wrong kind for its item.
pub const HB_ERR_TYPE: HbStatus = 2;
/// The problem or the combination is not available.
pub const HB_ERR_UNSUPPORTED: HbStatus = 3;
/// The model is in a state that refuses the call (calculating, not calculated).
pub const HB_ERR_STATE: HbStatus = 4;
/// The solve was cancelled.
pub const HB_ERR_CANCELLED: HbStatus = 5;
/// An internal error of the engine (a bug; a panic caught at the boundary).
pub const HB_ERR_INTERNAL: HbStatus = 6;
/// An argument of the call is invalid: a null pointer, a text that is not UTF-8, an index out of
/// range.
pub const HB_ERR_ARGUMENT: HbStatus = 7;
/// A numerical failure (a singular system).
pub const HB_ERR_NUMERICAL: HbStatus = 8;
/// A file could not be read or written.
pub const HB_ERR_IO: HbStatus = 9;
/// A file exists and overwrite was not given.
pub const HB_ERR_EXISTS: HbStatus = 10;

/// The place an error or a diagnostic names: zero-based indices, the mesh element one-based; -1 where
/// it names none.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HbLocation {
    pub body: i64,
    pub mesh: i64,
    pub element: i64,
    pub field_point: i64,
    pub lid: i64,
    pub frequency: i64,
}

impl HbLocation {
    pub fn of(l: &Location) -> HbLocation {
        let f = |v: Option<usize>| v.map_or(-1, |v| v as i64);
        HbLocation {
            body: f(l.body),
            mesh: f(l.mesh),
            element: f(l.element),
            field_point: f(l.field_point),
            lid: f(l.lid),
            frequency: f(l.frequency),
        }
    }
}

const NOWHERE: HbLocation = HbLocation { body: -1, mesh: -1, element: -1, field_point: -1, lid: -1, frequency: -1 };

thread_local! {
    static LAST: RefCell<(CString, HbLocation)> = RefCell::new((CString::default(), NOWHERE));
}

/// The status of an error kind.
pub fn status_of(k: ErrorKind) -> HbStatus {
    match k {
        ErrorKind::Value => HB_ERR_VALUE,
        ErrorKind::Type => HB_ERR_TYPE,
        ErrorKind::Unsupported => HB_ERR_UNSUPPORTED,
        ErrorKind::State => HB_ERR_STATE,
        ErrorKind::Cancelled => HB_ERR_CANCELLED,
        ErrorKind::Internal => HB_ERR_INTERNAL,
        ErrorKind::Argument => HB_ERR_ARGUMENT,
        ErrorKind::Numerical => HB_ERR_NUMERICAL,
        ErrorKind::Io => HB_ERR_IO,
        ErrorKind::Exists => HB_ERR_EXISTS,
    }
}

fn keep(message: &str, location: HbLocation) {
    // a NUL inside the message ends it there
    let text = CString::new(message.split('\0').next().unwrap_or("")).unwrap_or_default();
    LAST.with(|l| *l.borrow_mut() = (text, location));
}

/// The failure of a call: an engine error, or an argument refused at the boundary.
pub enum Fail {
    Engine(Error),
    Argument(String),
}

impl From<Error> for Fail {
    fn from(e: Error) -> Fail {
        Fail::Engine(e)
    }
}

pub type Out<T> = std::result::Result<T, Fail>;

/// A refused argument.
pub fn argument<T>(message: impl Into<String>) -> Out<T> {
    Err(Fail::Argument(message.into()))
}

/// Whether the processor and the system run the instructions the engine is built with (AVX2 and
/// FMA, `.cargo/config.toml`), read once. CPUID itself is asked: the run-time detection of std
/// answers yes for every feature the build enables. A best effort: this function, `guard` and the
/// code around them are compiled with the same features (Rust cannot compile one function without
/// them), so the compiler may use VEX (AVX) encodings in them too. The check is reliable on a
/// processor with AVX but without AVX2 or FMA; on one without AVX at all the check itself may stop
/// at an illegal instruction before it can answer.
pub fn cpu_supported() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        use std::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};
        let bit = |r: u32, b: u32| (r >> b) & 1 == 1;
        // SAFETY: CPUID exists on every x86-64 processor; XGETBV exists when CPUID reports OSXSAVE
        // (checked before it is called)
        #[allow(unused_unsafe)] // __cpuid is a safe function in newer toolchains
        unsafe {
            let l1 = __cpuid(1);
            // FMA (ECX 12), OSXSAVE (ECX 27), AVX (ECX 28)
            if !(bit(l1.ecx, 12) && bit(l1.ecx, 27) && bit(l1.ecx, 28)) || __cpuid(0).eax < 7 {
                return false;
            }
            // the system saves the SSE and AVX registers (XCR0 bits 1 and 2), AVX2 (leaf 7, EBX 5)
            (_xgetbv(0) & 6) == 6 && bit(__cpuid_count(7, 0).ebx, 5)
        }
    })
}

/// The message of a call on a processor without AVX2 and FMA.
const CPU_UNSUPPORTED: &str = "this processor (or its system) lacks the AVX2 and FMA instructions the engine is built \
                                   for; the library runs on x86-64 processors with AVX2 and FMA";

/// Runs the body of a call: its status, the error kept on failure, a panic caught. On a processor
/// without AVX2 and FMA the body is not run: `HB_ERR_UNSUPPORTED` (the engine's code would stop at
/// an illegal instruction). A best effort (`cpu_supported`): on a processor without AVX at all the
/// check itself may stop at an illegal instruction.
pub fn guard(f: impl FnOnce() -> Out<()>) -> HbStatus {
    if !cpu_supported() {
        keep(CPU_UNSUPPORTED, NOWHERE);
        return HB_ERR_UNSUPPORTED;
    }
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => HB_OK,
        Ok(Err(Fail::Engine(e))) => {
            keep(e.message(), HbLocation::of(e.location()));
            status_of(e.kind())
        }
        Ok(Err(Fail::Argument(m))) => {
            keep(&m, NOWHERE);
            HB_ERR_ARGUMENT
        }
        Err(p) => {
            let what = p
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            keep(&format!("internal error of the engine (please report it): {what}"), NOWHERE);
            HB_ERR_INTERNAL
        }
    }
}

/// The message of the last failed call of this thread (UTF-8, empty when none failed); valid until
/// the next call of this thread fails.
pub fn last_message() -> *const c_char {
    LAST.with(|l| l.borrow().0.as_ptr())
}

pub fn last_location() -> HbLocation {
    LAST.with(|l| l.borrow().1)
}

/// A text argument: UTF-8, not null.
///
/// # Safety
/// `p` is null or a NUL-terminated string that lives for the call.
pub unsafe fn text<'a>(p: *const c_char, what: &str) -> Out<&'a str> {
    if p.is_null() {
        return argument(format!("{what} is a null pointer"));
    }
    // SAFETY: a NUL-terminated string (the caller's contract)
    let s = unsafe { CStr::from_ptr(p) };
    s.to_str().or_else(|_| argument(format!("{what} is not UTF-8")))
}

/// An optional text argument: null for none.
///
/// # Safety
/// As `text`.
pub unsafe fn opt_text<'a>(p: *const c_char, what: &str) -> Out<Option<&'a str>> {
    if p.is_null() { Ok(None) } else { unsafe { text(p, what) }.map(Some) }
}

/// A path argument (UTF-8).
///
/// # Safety
/// As `text`.
pub unsafe fn path(p: *const c_char, what: &str) -> Out<PathBuf> {
    unsafe { text(p, what) }.map(PathBuf::from)
}

/// The texts of an array of n strings.
///
/// # Safety
/// `p` points to n strings as `text` takes them (or is null with n = 0).
pub unsafe fn texts(p: *const *const c_char, n: usize, what: &str) -> Out<Vec<String>> {
    let ptrs = unsafe { slice(p, n, what) }?;
    ptrs.iter().enumerate().map(|(k, &s)| unsafe { text(s, &format!("{what}[{k}]")) }.map(String::from)).collect()
}

/// The most elements of an array argument (2^40, as the engine's arrays): the byte size of the
/// elements stays far below `isize::MAX`.
pub const MAX_ELEMENTS: usize = 1 << 40;

/// The number of elements of a shape (its product), refused beyond `MAX_ELEMENTS`.
pub fn elements(shape: &[usize], what: &str) -> Out<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    let mut n: usize = 1;
    for &d in shape {
        match n.checked_mul(d) {
            Some(v) if v <= MAX_ELEMENTS => n = v,
            _ => return argument(format!("{what}: more than 2^40 elements")),
        }
    }
    Ok(n)
}

/// The n elements at p (none when n = 0, p then may be null; more than `MAX_ELEMENTS` refused).
///
/// # Safety
/// `p` points to n elements of T that live for the call.
pub unsafe fn slice<'a, T>(p: *const T, n: usize, what: &str) -> Out<&'a [T]> {
    if n == 0 {
        return Ok(&[]);
    }
    if n > MAX_ELEMENTS {
        return argument(format!("{what}: more than 2^40 elements"));
    }
    if p.is_null() {
        return argument(format!("{what} is a null pointer"));
    }
    // SAFETY: n elements at p (the caller's contract)
    Ok(unsafe { std::slice::from_raw_parts(p, n) })
}

/// Refuses a null out-pointer. Every function checks its out-pointers with it first, before it
/// changes a model or allocates, so that a refused call changes nothing and leaks nothing.
pub fn writable<T>(p: *mut T, what: &str) -> Out<()> {
    if p.is_null() { argument(format!("{what} is a null pointer")) } else { Ok(()) }
}

/// Writes a value through an out-pointer.
///
/// # Safety
/// `p` is null or points to a T the call may write.
pub unsafe fn put<T>(p: *mut T, v: T, what: &str) -> Out<()> {
    if p.is_null() {
        return argument(format!("{what} is a null pointer"));
    }
    // SAFETY: a writable T (the caller's contract)
    unsafe { p.write(v) };
    Ok(())
}

/// A text the caller frees with `hb_string_free`.
pub fn owned_text(s: &str) -> *mut c_char {
    CString::new(s.split('\0').next().unwrap_or("")).unwrap_or_default().into_raw()
}
