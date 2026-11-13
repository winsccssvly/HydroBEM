//! The model and results files of the engine, .hbm and .hbr: the model, the meta
//! data, the diagnostics and (.hbr) every quantity of a solve, in a header, 64-byte aligned entries
//! and a directory with CRC-32.
//!
//! Little-endian: a 64-byte header (the signature, the format version, the offset, size and CRC-32
//! of the directory), the entries each on a 64-byte boundary, and the directory last (written after
//! the entries, whose sizes it holds). The entries "meta", "diagnostics" and "model" are binary
//! records (binary.rs); a results file adds the quantities "q/<name>".

use std::borrow::Cow;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::binary::{Reader, put_f64, put_i64, put_text, put_u};
use crate::build_info::build_info;
use crate::codes::{ElementType, QuantityId, Severity};
use crate::document::Document;
use crate::error::{Diagnostic, Error, Location, Result, index, int32_of};
use crate::quantities::quantity_info;
use crate::results::{Quantity, QuantityData, Results, RunMeta};
use crate::text::{absolute_path, file_exists, make_parent_folder, write_file};

const SIGNATURE: [u8; 8] = [0x89, b'H', b'B', b'R', b'\r', b'\n', 0x1a, b'\n'];
const MODEL_SIGNATURE: [u8; 8] = [0x89, b'H', b'B', b'M', b'\r', b'\n', 0x1a, b'\n']; // the model file (.hbm)
// 2: binary records of the model, meta and diagnostics; 3: the quantities of a solve alone, named as the
// output requests; 4: model record 10 and later; 5: model record 11 and later and the quantities named by
// their paths of the result tree
const VERSION: u32 = 5;
const HEADER: usize = 64;
const ALIGN: usize = 64;

// the element types of the entries
const T_TEXT: u8 = 0;
const T_INT32: u8 = 1;
const T_INT64: u8 = 2;
const T_FLOAT32: u8 = 3;
const T_FLOAT64: u8 = 4;
const T_COMPLEX64: u8 = 5;
const T_COMPLEX128: u8 = 6;

fn file_type(dtype: ElementType) -> Result<u8> {
    match dtype {
        ElementType::Int32 => Ok(T_INT32),
        ElementType::Int64 => Ok(T_INT64),
        ElementType::Float32 => Ok(T_FLOAT32),
        ElementType::Float64 => Ok(T_FLOAT64),
        ElementType::ComplexF32 => Ok(T_COMPLEX64),
        ElementType::ComplexF64 => Ok(T_COMPLEX128),
        ElementType::Text => Err(Error::internal(format!("results file: no entry type of dtype {dtype:?}"))),
    }
}

fn dtype_of(t: u8, where_: &str) -> Result<ElementType> {
    match t {
        T_INT32 => Ok(ElementType::Int32),
        T_INT64 => Ok(ElementType::Int64),
        T_FLOAT32 => Ok(ElementType::Float32),
        T_FLOAT64 => Ok(ElementType::Float64),
        T_COMPLEX64 => Ok(ElementType::ComplexF32),
        T_COMPLEX128 => Ok(ElementType::ComplexF64),
        _ => Err(Error::value(format!("{where_}: unknown entry type {t}"))),
    }
}

#[derive(Clone, Debug, Default)]
struct Entry {
    name: String,
    type_: u8,
    shape: Vec<i64>,
    offset: u64,
    size: u64,
    crc: u32,
    dims: String,
    unit: String,
}

// ------------------------------------------------------------------ the records of the meta data and diagnostics

fn diagnostics_record(diags: &[Diagnostic]) -> Vec<u8> {
    let mut out = Vec::new();
    put_u(&mut out, diags.len() as u64, 4);
    for d in diags {
        // the location fields that do not apply as -1
        let w = &d.location;
        let field = |f: Option<usize>| f.map_or(-1, |v| v as i64);
        put_i64(&mut out, d.severity.code() as i64);
        put_i64(&mut out, d.code as i64);
        for x in [w.body, w.mesh, w.element, w.field_point, w.lid, w.frequency] {
            put_i64(&mut out, field(x));
        }
        put_text(&mut out, &d.message);
    }
    out
}

fn diagnostics_of(bytes: &[u8], where_: &str) -> Result<Vec<Diagnostic>> {
    let mut r = Reader::new(bytes, where_.to_string());
    let mut out = Vec::new();
    let n = r.u(4)?;
    for _ in 0..n {
        let mut f = [0i32; 8];
        for x in f.iter_mut() {
            *x = int32_of(r.i64()?, where_)?;
        }
        let location = Location {
            body: index(f[2]),
            mesh: index(f[3]),
            element: index(f[4]),
            field_point: index(f[5]),
            lid: index(f[6]),
            frequency: index(f[7]),
        };
        out.push(Diagnostic { severity: Severity::from_code(f[0]), code: f[1], location, message: r.text()? });
    }
    Ok(out)
}

fn meta_record(m: &RunMeta) -> Vec<u8> {
    let mut out = Vec::new();
    put_u(&mut out, if m.present { 1 } else { 0 }, 1);
    for t in [&m.hydrobem_version, &m.commit, &m.compiler, &m.linear_algebra, &m.build_type, &m.time, &m.route] {
        put_text(&mut out, t);
    }
    for x in [m.api_version, m.frequency_threads, m.inner_threads] {
        put_i64(&mut out, x as i64);
    }
    put_u(&mut out, m.engine_seconds.len() as u64, 4);
    for &x in &m.engine_seconds {
        put_f64(&mut out, x);
    }
    put_u(&mut out, m.environment_hooks.len() as u64, 4);
    for t in &m.environment_hooks {
        put_text(&mut out, t);
    }
    out
}

fn meta_of(bytes: &[u8], where_: &str) -> Result<RunMeta> {
    let mut r = Reader::new(bytes, where_.to_string());
    let mut m = RunMeta { present: r.u(1)? != 0, ..RunMeta::default() };
    m.hydrobem_version = r.text()?;
    m.commit = r.text()?;
    m.compiler = r.text()?;
    m.linear_algebra = r.text()?;
    m.build_type = r.text()?;
    m.time = r.text()?;
    m.route = r.text()?;
    m.api_version = int32_of(r.i64()?, where_)?;
    m.frequency_threads = int32_of(r.i64()?, where_)?;
    m.inner_threads = int32_of(r.i64()?, where_)?;
    let n = r.u(4)?;
    for _ in 0..n {
        m.engine_seconds.push(r.f64()?);
    }
    let n = r.u(4)?;
    for _ in 0..n {
        m.environment_hooks.push(r.text()?);
    }
    Ok(m)
}

// The civil date of a day count from 1970-01-01 (proleptic Gregorian; H. Hinnant's days_from_civil inverse).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// "%Y-%m-%dT%H:%M:%SZ" of a time in seconds since 1970 (UTC)
fn utc_text(secs: i64) -> String {
    let days = secs.div_euclid(86400);
    let s = secs.rem_euclid(86400);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", s / 3600, s / 60 % 60, s % 60)
}

fn utc_now() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    utc_text(secs)
}

fn temporary_name(path: &str) -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let tick = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    format!("{path}.tmp{tick}-{}", N.fetch_add(1, Ordering::Relaxed) + 1)
}

/// The text of an I/O error as std::error_code::message gives it (without Rust's "(os error N)").
pub(crate) fn io_message(e: &std::io::Error) -> String {
    let s = e.to_string();
    match s.rfind(" (os error ") {
        Some(k) if s.ends_with(')') => s[..k].to_string(),
        _ => s,
    }
}

/// CRC-32 of ISO 3309 / ITU-T V.42 (the zlib polynomial, reflected 0xEDB88320); crc: the CRC so
/// far (0 to start).
pub fn crc32(data: &[u8], crc: u32) -> u32 {
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for i in 0..256u32 {
            let mut c = i;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB88320u32 ^ (c >> 1) } else { c >> 1 };
            }
            t[i as usize] = c;
        }
        t
    });
    let mut crc = !crc;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

/// The meta data of a solve: the build, the time, the route, the thread counts, the engine seconds
/// (the hooks of the environment empty: the solve sets them).
pub fn results_meta(route: &str, frequency_threads: i32, inner_threads: i32, engine_seconds: &[f64]) -> RunMeta {
    let b = build_info();
    RunMeta {
        present: true,
        hydrobem_version: b.version.clone(),
        api_version: crate::build_info::API_VERSION,
        commit: b.commit.clone(),
        compiler: b.compiler.clone(),
        linear_algebra: b.linear_algebra.clone(),
        build_type: b.build_type.clone(),
        time: utc_now(),
        route: route.to_string(),
        frequency_threads,
        inner_threads,
        engine_seconds: engine_seconds.to_vec(),
        environment_hooks: Vec::new(),
    }
}

// The bytes of a pending quantity read from the file of r, its CRC checked.
fn read_entry(r: &Results, q: &Quantity, name: &str) -> Result<Vec<u8>> {
    let mut f = match fs::File::open(&r.file) {
        Ok(f) => f,
        Err(_) => return Err(Error::io(format!("{}: cannot open the file", r.file))),
    };
    let mut bytes = vec![0u8; q.size as usize];
    let ok = f.seek(SeekFrom::Start(q.offset as u64)).is_ok() && (q.size == 0 || f.read_exact(&mut bytes).is_ok());
    if !ok {
        return Err(Error::io(format!("{}: cannot read the entry q/{name}", r.file)));
    }
    if crc32(&bytes, 0) != q.crc {
        return Err(Error::value(format!("{}: the entry q/{name} is damaged (CRC-32)", r.file)));
    }
    Ok(bytes)
}

/// The elements of a quantity, read from the file of r the first time (its CRC checked): into
/// q.elements (results.rs). Nothing for a quantity that is not pending.
pub fn load_quantity(r: &Results, q: &Quantity, name: &str) -> Result<()> {
    if !q.pending || q.elements.get().is_some() {
        return Ok(());
    }
    let data = QuantityData::of_bytes(q.dtype, &read_entry(r, q, name)?)?;
    let _ = q.elements.set(Arc::new(data)); // another thread may have read it first: the same elements
    Ok(())
}

// The bytes of a quantity for a write: those held, or read from the file (into a copy, not kept).
fn quantity_bytes<'a>(r: &Results, q: &'a Quantity, name: &str) -> Result<Cow<'a, [u8]>> {
    match q.elements.get() {
        Some(d) => Ok(Cow::Borrowed(d.as_bytes())),
        None if q.pending => Ok(Cow::Owned(read_entry(r, q, name)?)),
        None => Ok(Cow::Borrowed(&[])),
    }
}

// the file of r: a results file, or a model file (its model and meta data only); model:
// the model written (that of r, or the one a model file is saved from)
fn write_file_of(r: &Results, model: Option<&Document>, path: &str, overwrite: bool, model_file: bool) -> Result<()> {
    if !overwrite && file_exists(path) {
        return Err(Error::exists(format!("{path} exists")));
    }
    make_parent_folder(path);
    let mut body: Vec<u8> = Vec::new(); // the entries after the header
    let mut entries: Vec<Entry> = Vec::new();
    fn align(body: &mut Vec<u8>) {
        while (HEADER + body.len()) % ALIGN != 0 {
            body.push(0);
        }
    }
    fn add(body: &mut Vec<u8>, entries: &mut Vec<Entry>, mut e: Entry, data: &[u8]) {
        align(body);
        e.offset = (HEADER + body.len()) as u64;
        e.size = data.len() as u64;
        e.crc = crc32(data, 0);
        body.extend_from_slice(data);
        entries.push(e);
    }
    let text_entry = |name: &str| Entry { name: name.to_string(), type_: T_TEXT, ..Entry::default() };
    let mut meta = r.meta.clone();
    if !meta.present && meta.hydrobem_version.is_empty() {
        // given results: the build and the time of the save
        meta = results_meta(&r.route, 0, 0, &[]);
        meta.present = false;
    }
    add(&mut body, &mut entries, text_entry("meta"), &meta_record(&meta));
    add(&mut body, &mut entries, text_entry("diagnostics"), &diagnostics_record(&r.diagnostics));
    if let Some(m) = model {
        add(&mut body, &mut entries, text_entry("model"), &m.record());
    }
    if !model_file {
        for (&q, qq) in &r.quantities {
            let info = quantity_info(q);
            let bytes = quantity_bytes(r, qq, info.name)?;
            let e = Entry {
                name: format!("q/{}", info.name),
                type_: file_type(qq.dtype)?,
                shape: qq.dims.clone(),
                dims: info.dims.to_string(),
                unit: info.unit.to_string(),
                ..Entry::default()
            };
            add(&mut body, &mut entries, e, &bytes);
        }
    }
    align(&mut body);
    let mut dir = Vec::new();
    put_u(&mut dir, entries.len() as u64, 4);
    for e in &entries {
        put_text(&mut dir, &e.name);
        put_u(&mut dir, e.type_ as u64, 1);
        put_u(&mut dir, e.shape.len() as u64, 1);
        for &s in &e.shape {
            put_u(&mut dir, s as u64, 8);
        }
        put_u(&mut dir, e.offset, 8);
        put_u(&mut dir, e.size, 8);
        put_u(&mut dir, e.crc as u64, 4);
        put_text(&mut dir, &e.dims);
        put_text(&mut dir, &e.unit);
    }
    let mut header: Vec<u8> = (if model_file { MODEL_SIGNATURE } else { SIGNATURE }).to_vec();
    put_u(&mut header, VERSION as u64, 4);
    put_u(&mut header, 0, 4);
    put_u(&mut header, (HEADER + body.len()) as u64, 8);
    put_u(&mut header, dir.len() as u64, 8);
    put_u(&mut header, crc32(&dir, 0) as u64, 4);
    header.resize(HEADER, 0);
    let tmp = temporary_name(path);
    let write = || -> Result<()> {
        let mut all = header;
        all.extend_from_slice(&body);
        all.extend_from_slice(&dir);
        write_file(&tmp, &all)?;
        if let Err(e) = fs::rename(&tmp, path) {
            return Err(Error::io(format!("{path}: cannot write the file ({})", io_message(&e))));
        }
        Ok(())
    };
    if let Err(e) = write() {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

/// Writes a results file: to a temporary file in the folder of path, renamed to path at the end.
pub fn save_results(r: &Results, path: &str, overwrite: bool) -> Result<()> {
    write_file_of(r, r.model.as_deref(), path, overwrite, false)?;
    Ok(())
}

/// Writes a model file (.hbm): the model and the build and time of the save, no results.
pub fn save_model(doc: &Document, path: &str, overwrite: bool) -> Result<()> {
    let r = Results::default();
    write_file_of(&r, Some(doc), path, overwrite, true)?; // the model borrowed for the write
    Ok(())
}

// bytes [offset, offset + size) of a file
fn read_range(path: &str, offset: u64, size: u64) -> Result<Vec<u8>> {
    let mut f = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Err(Error::io(format!("{path}: cannot open the file"))),
    };
    let mut out = vec![0u8; size as usize];
    let ok = f.seek(SeekFrom::Start(offset)).is_ok() && (size == 0 || f.read_exact(&mut out).is_ok());
    if !ok {
        return Err(Error::io(format!("{path}: cannot read the file")));
    }
    Ok(out)
}

// the bytes of one element of a quantity's type
fn element_size(dtype: ElementType) -> u64 {
    match dtype {
        ElementType::Float32 | ElementType::Int32 => 4,
        ElementType::ComplexF64 => 16,
        _ => 8, // float64, int64, complex64
    }
}

/// Opens a results or a model file: the header and the directory (their CRC checked); the
/// quantities are read when first asked for (load_quantity).
pub fn open_results(path: &str) -> Result<Results> {
    let file_size = match fs::metadata(path) {
        Ok(m) if m.is_file() => m.len(),
        _ => return Err(Error::io(format!("{path}: cannot open the file"))),
    };
    if file_size < HEADER as u64 {
        return Err(Error::value(format!("{path}: not a HydroBEM model or results file")));
    }
    let head = read_range(path, 0, HEADER as u64)?;
    let model_file = head[..8] == MODEL_SIGNATURE;
    if !model_file && head[..8] != SIGNATURE {
        return Err(Error::value(format!("{path}: not a HydroBEM model or results file")));
    }
    let mut h = Reader::new(&head[8..], path.to_string());
    let version = h.u(4)?;
    let _ = h.u(4)?; // flags
    let dir_offset = h.u(8)?;
    let dir_size = h.u(8)?;
    let dir_crc = h.u(4)? as u32;
    if version != VERSION as u64 {
        return Err(Error::value(format!("{path}: file format version {version}; this HydroBEM reads {VERSION}")));
    }
    if dir_offset < HEADER as u64 || dir_offset > file_size || dir_size != file_size - dir_offset {
        return Err(Error::value(format!("{path}: the file is cut short or damaged (no valid directory)")));
    }
    let dir = read_range(path, dir_offset, dir_size)?;
    if crc32(&dir, 0) != dir_crc {
        return Err(Error::value(format!("{path}: the directory is damaged (CRC-32)")));
    }
    let mut d = Reader::new(&dir, path.to_string());
    let n = d.u(4)?;
    let mut entries = Vec::new();
    for _ in 0..n {
        let mut e = Entry { name: d.text()?, type_: d.u(1)? as u8, ..Entry::default() };
        let rank = d.u(1)?;
        for _ in 0..rank {
            e.shape.push(d.u(8)? as i64);
        }
        if rank > 4 {
            return Err(Error::value(format!("{path}: the entry {} has more than four dimensions", e.name)));
        }
        e.offset = d.u(8)?;
        e.size = d.u(8)?;
        e.crc = d.u(4)? as u32;
        e.dims = d.text()?;
        e.unit = d.text()?;
        if e.offset < HEADER as u64 || e.offset > dir_offset || e.size > dir_offset - e.offset {
            return Err(Error::value(format!("{path}: the entry {} lies outside the data", e.name)));
        }
        entries.push(e);
    }
    // the quantities are read when first used: by the absolute path (the working folder may change)
    let mut r = Results { file: absolute_path(path), model_file, ..Results::default() };
    let text_of = |e: &Entry| {
        let t = read_range(path, e.offset, e.size)?;
        if crc32(&t, 0) != e.crc {
            return Err(Error::value(format!("{path}: the entry {} is damaged (CRC-32)", e.name)));
        }
        Ok(t)
    };
    let mut has_meta = false;
    for e in &entries {
        if e.name == "meta" {
            r.meta = meta_of(&text_of(e)?, &format!("{path}: meta"))?;
            has_meta = true;
        } else if e.name == "diagnostics" {
            r.diagnostics = diagnostics_of(&text_of(e)?, &format!("{path}: diagnostics"))?;
        } else if e.name == "model" {
            let mut doc = Document::new();
            doc.load_record(&text_of(e)?, &format!("{path}: model"))?;
            r.model = Some(Arc::new(doc));
        } else if e.name.starts_with("q/") && !model_file {
            let name = &e.name[2..];
            for q in QuantityId::ALL {
                let info = quantity_info(q);
                if name != info.name {
                    continue;
                }
                let mut qq = Quantity { dtype: dtype_of(e.type_, path)?, dims: e.shape.clone(), ..Quantity::default() };
                // the entry as the engine defines the quantity: its type, and its size the product of
                // its extents
                if qq.dtype != info.dtype {
                    return Err(Error::value(format!(
                        "{path}: the entry {} has another element type than {name}",
                        e.name
                    )));
                }
                let mut count: u64 = 1;
                for &x in &e.shape {
                    if x < 0 || (x > 0 && count > (1u64 << 40) / x as u64) {
                        return Err(Error::value(format!("{path}: the entry {} has an invalid shape", e.name)));
                    }
                    count *= x as u64;
                }
                if count * element_size(qq.dtype) != e.size {
                    return Err(Error::value(format!("{path}: the entry {} does not match its shape", e.name)));
                }
                qq.pending = true;
                qq.offset = e.offset as i64;
                qq.size = e.size as i64;
                qq.crc = e.crc;
                r.quantities.insert(q, qq);
            }
        } // an unknown entry is ignored (a later version of the format)
    }
    if !has_meta {
        return Err(Error::value(format!("{path}: no meta data")));
    }
    r.route = r.meta.route.clone();
    if model_file && r.model.is_none() {
        return Err(Error::value(format!("{path}: a model file without its model")));
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::Warning;

    #[test]
    fn crc32_check_values() {
        assert_eq!(crc32(b"", 0), 0);
        assert_eq!(crc32(b"123456789", 0), 0xCBF43926); // the check value of CRC-32/ISO-HDLC
        assert_eq!(crc32(b"The quick brown fox jumps over the lazy dog", 0), 0x414FA339);
        // in parts: the CRC so far continues
        assert_eq!(crc32(b"6789", crc32(b"12345", 0)), 0xCBF43926);
    }

    #[test]
    fn utc_dates() {
        assert_eq!(utc_text(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc_text(951782400), "2000-02-29T00:00:00Z");
        assert_eq!(utc_text(1793491199), "2026-10-31T23:59:59Z");
        assert_eq!(utc_text(4107542400), "2100-03-01T00:00:00Z");
    }

    #[test]
    fn meta_and_diagnostics_records() -> Result<()> {
        let m = RunMeta {
            present: true,
            hydrobem_version: "1.2".into(),
            route: "lo".into(),
            api_version: 28,
            frequency_threads: 3,
            engine_seconds: vec![0.5, 1.25],
            environment_hooks: vec!["HB_DIAG_DOUBLE".into(), "HB_VALIDATE_OUTER_TAPER=0.4".into()],
            ..RunMeta::default()
        };
        let b = meta_of(&meta_record(&m), "m")?;
        assert_eq!(b.environment_hooks, m.environment_hooks);
        assert!(b.present && b.hydrobem_version == "1.2" && b.route == "lo" && b.api_version == 28);
        assert_eq!(b.frequency_threads, 3);
        assert_eq!(b.engine_seconds, vec![0.5, 1.25]);
        let d =
            Diagnostic::warning_at(Warning::TankZeroFrequency, Location { body: Some(2), ..Location::default() }, "w");
        let back = diagnostics_of(&diagnostics_record(std::slice::from_ref(&d)), "d")?;
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].code, 7);
        assert_eq!(back[0].location, d.location);
        assert_eq!(back[0].severity, Severity::Warning);
        assert_eq!(back[0].message, "w");
        let e = meta_of(&[1, 0], "x").unwrap_err();
        assert_eq!(e.message(), "x: the record is damaged (cut short)");
        Ok(())
    }
}
