//! Readers and writers of the geometry files (reference/inputs-meshes): low-order and higher-order GDF
//! files (Fortran list-directed records), the SPL file, the panel-size rule, the analytic
//! shapes (a negative geometry kind in the GDF header), and the control-surface file (CSF).

use crate::analytic::AnalyticPatch;
use crate::analytic_shapes::{ShapeSpec, named_shape};
use crate::constants::{PI, STANDARD_GRAVITY};
use crate::error::{Error, Result};
use crate::geometry::{Vec3, max_of, norm};
use crate::mesh::{Mesh, check_knots, record_eval, spline_layout};
use crate::mesh_desc::*;
use crate::mesh_desc::{AnalyticKind, PatchKind, Profile};
use crate::preprocess::automatic_control_surface;
use crate::text::{is_number, parse_number, py_repr, py_str_repr, read_file, split_lines, strip, tokens};

/// A GDF or CSF file as read, with the storage the mesh descriptors are made from.
#[derive(Clone, Debug)]
pub struct GdfFile {
    pub label: String,
    pub ulen: f64,
    pub gravity: f64,
    pub isx: i32,
    pub isy: i32,
    pub higher_order: bool,
    pub geometry: PatchKind, // higher order
    pub count: i32,          // the panel or patch count of the header
    pub vertices: Vec<f64>,  // panels or flat patches, count x 4 x 3
    pub orders: Vec<i32>,    // spline: per patch the orders (u, v) and (M_u, M_v)
    pub counts: Vec<i32>,
    pub knots: Vec<f64>, // spline: concatenated per patch
    pub coefficients: Vec<f64>,
    pub analytic: Vec<AnalyticPatchDesc>,         // an analytic shape: the patches
    pub discretization: Vec<PatchDiscretization>, // higher order
}

impl Default for GdfFile {
    fn default() -> Self {
        GdfFile {
            label: String::new(),
            ulen: 1.0,
            gravity: STANDARD_GRAVITY,
            isx: 0,
            isy: 0,
            higher_order: false,
            geometry: PatchKind::Flat,
            count: 0,
            vertices: Vec::new(),
            orders: Vec::new(),
            counts: Vec::new(),
            knots: Vec::new(),
            coefficients: Vec::new(),
            analytic: Vec::new(),
            discretization: Vec::new(),
        }
    }
}

/// How the discretization of a higher-order file is given (exactly one of spl, discretization,
/// panel size; the keyword values 0 when not given).
#[derive(Clone, Debug, Default)]
pub struct PatchOptions {
    pub spl: String, // empty: none
    pub has_discretization: bool,
    pub discretization: Vec<PatchDiscretization>,
    pub has_panel_size: bool,
    pub panel_size: f64,
    pub spline_order: i32,
    pub outer_quadrature: i32,
    pub inner_quadrature: i32,
}

fn num_repr(v: f64) -> String {
    py_repr(v)
}

// An integer field: the value must be integral (and fit an int32).
fn integer(value: f64, name: &str, path: &str) -> Result<i32> {
    if !(value.is_finite() && value == value.floor() && value.abs() < 2147483647.0) {
        return Err(Error::value(format!("{path}: {name} must be an integer, got {}", num_repr(value))));
    }
    Ok(value as i32)
}

// The first n numbers of a line (a comment may follow them).
fn leading_numbers(line: &str, n: usize, what: &str, path: &str) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    for t in tokens(line).iter().take(n) {
        if !is_number(t) {
            break;
        }
        values.push(parse_number(t)?);
    }
    if values.len() < n {
        return Err(Error::value(format!("{path}: expected {what} on the line {}", py_str_repr(&strip(line)))));
    }
    Ok(values)
}

// Fortran list-directed records: every read starts on a new line, may continue over further lines,
// and the rest of its last line must be text (a comment), never further numbers. Blank lines are
// skipped.
struct Records {
    path: String,
    lines: Vec<String>,
    pos: usize,
    first: i32,
}

impl Records {
    fn new(path: &str, lines: Vec<String>, first: i32) -> Records {
        Records { path: path.to_string(), lines, pos: 0, first }
    }

    fn lineno(&self) -> i32 {
        self.first + self.pos as i32
    }

    fn take(&mut self, n: usize, what: &str) -> Result<Vec<f64>> {
        let mut values = Vec::new();
        while values.len() < n {
            if self.pos >= self.lines.len() {
                return Err(Error::value(format!("{}: the file ends while reading {what}", self.path)));
            }
            let t = tokens(&self.lines[self.pos]);
            let ln = self.lineno();
            self.pos += 1;
            for tok in &t {
                if values.len() == n {
                    if is_number(tok) {
                        return Err(Error::value(format!("{}: line {ln}: more numbers than {what} needs", self.path)));
                    }
                    break; // trailing comment
                }
                if !is_number(tok) {
                    return Err(Error::value(format!(
                        "{}: line {ln}: expected {what}, found {}",
                        self.path,
                        py_str_repr(tok)
                    )));
                }
                values.push(parse_number(tok)?);
            }
        }
        Ok(values)
    }

    // All remaining numbers: on a line that begins with numbers, text after them is a comment
    // (as "end of panel 1" after the last vertex of a panel); a line that begins with text is not
    // a data line and refused.
    fn rest(&mut self) -> Result<Vec<f64>> {
        let mut out = Vec::new();
        while self.pos < self.lines.len() {
            let ln = self.lineno();
            for (k, tok) in tokens(&self.lines[self.pos]).into_iter().enumerate() {
                if !is_number(&tok) && k > 0 {
                    break; // a trailing comment
                }
                if !is_number(&tok) {
                    return Err(Error::value(format!(
                        "{}: line {ln}: invalid number {}",
                        self.path,
                        py_str_repr(&tok)
                    )));
                }
                out.push(parse_number(&tok)?);
            }
            self.pos += 1;
        }
        Ok(out)
    }

    // The leading numbers of each of the next n lines (the data lines of an analytic shape).
    fn data_lines(&mut self, n: i32) -> Result<Vec<f64>> {
        let mut values = Vec::new();
        for _ in 0..n {
            if self.pos >= self.lines.len() {
                return Err(Error::value(format!("{}: the file ends in the NLINES data lines", self.path)));
            }
            for tok in tokens(&self.lines[self.pos]) {
                if !is_number(&tok) {
                    break;
                }
                values.push(parse_number(&tok)?);
            }
            self.pos += 1;
        }
        Ok(values)
    }
}

struct Header {
    label: String,
    ulen: f64,
    grav: f64,
    isx: i32,
    isy: i32,
    text: Vec<String>,
}

// The lines of a file (its bytes as UTF-8; an invalid sequence is replaced).
fn file_lines(path: &str) -> Result<Vec<String>> {
    Ok(split_lines(&String::from_utf8_lossy(&read_file(path)?)))
}

// A symmetry flag of a GDF or CSF header: 1 a plane of symmetry (x = 0 or y = 0), 0 none
fn symmetry_flag(v: f64, name: &str, path: &str) -> Result<i32> {
    let f = integer(v, name, path)?;
    if f != 0 && f != 1 {
        return Err(Error::value(format!("{path}: {name} must be 0 or 1, not {f}")));
    }
    Ok(f)
}

fn read_header(path: &str, line4: &str) -> Result<Header> {
    let text = file_lines(path)?;
    if text.len() < 4 {
        return Err(Error::value(format!("{path}: a GDF file needs a header, ULEN GRAV, ISX ISY and {line4} lines")));
    }
    let label = strip(&text[0]);
    let ug = leading_numbers(&text[1], 2, "ULEN GRAV", path)?;
    let (ulen, grav) = (ug[0], ug[1]);
    if !(grav.is_finite() && grav > 0.0) {
        return Err(Error::value(format!("{path}: GRAV must be a positive finite number")));
    }
    let s = leading_numbers(&text[2], 2, "ISX ISY", path)?;
    let isx = symmetry_flag(s[0], "ISX", path)?;
    let isy = symmetry_flag(s[1], "ISY", path)?;
    Ok(Header { label, ulen, grav, isx, isy, text })
}

fn records_after_header(path: &str, h: &Header) -> Records {
    Records::new(path, h.text[4..].to_vec(), 5)
}

// A trailing record after the data: all zeros only when accepted; reports whether there was one.
fn check_trailer(path: &str, extra: &[f64], zero_trailer: bool, count_name: &str, count: i32) -> Result<bool> {
    if extra.is_empty() {
        return Ok(false);
    }
    if extra.iter().all(|&v| v == 0.0) {
        if zero_trailer {
            return Ok(true);
        }
        return Err(Error::value(format!(
            "{path}: trailing record of zeros after {count_name} = {count} (pass zero_trailer = true)"
        )));
    }
    Err(Error::value(format!("{path}: unexpected data after {count_name} = {count}")))
}

// ---------------------------------------------------------------- the panel-size rule

/// The panel count of an edge of length L for a panel size: floor(L / size) + 1, an edge within
/// 1e-5 sizes of a multiple of the size counting as the multiple (rounding of the arc length).
/// More than 10^6 panels (also a NaN length) is an ErrorKind::Value error, so the count never
/// overflows. The rule of panel_size and of the control surfaces.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
pub(crate) fn panel_count(length: f64, size: f64) -> Result<i32> {
    let n = length / size + 1e-5;
    if !(n < 1e6) {
        return Err(Error::value("panel_size gives more than 10^6 panels along an edge"));
    }
    Ok(n as i32 + 1)
}

/// The arc lengths (lu, lv) of a patch map on [-1, 1]^2: of the longer of its two u edges
/// (v = -1, 1) and of its two v edges, each summed over the chords of 257 equally spaced points.
/// The lengths of the panel-size rule, for any map (flat, B-spline, rational, analytic, trimmed).
pub(crate) fn edge_lengths(eval: impl Fn(f64, f64) -> Vec3) -> (f64, f64) {
    let s = equally_spaced(-1.0, 1.0);
    let edge = |along_u: bool, fixed: f64| {
        let pts: Vec<Vec3> = s.iter().map(|&si| if along_u { eval(si, fixed) } else { eval(fixed, si) }).collect();
        arc_length(&pts)
    };
    (max_of(edge(true, -1.0), edge(true, 1.0)), max_of(edge(false, -1.0), edge(false, 1.0)))
}

fn size_discretization(lu: f64, lv: f64, size: f64, o: &PatchOptions) -> Result<PatchDiscretization> {
    let k = if o.spline_order != 0 { o.spline_order } else { 3 };
    Ok(PatchDiscretization {
        nu: panel_count(lu, size)?,
        nv: panel_count(lv, size)?,
        ku: k,
        kv: k,
        iquo: if o.outer_quadrature != 0 { o.outer_quadrature } else { 3 },
        iqui: if o.inner_quadrature != 0 { o.inner_quadrature } else { 4 },
    })
}

// 257 equally spaced points from a to b, the last exactly b
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn equally_spaced(a: f64, b: f64) -> Vec<f64> {
    let mut x = vec![0.0; 257];
    let step = (b - a) / 256.0;
    for i in 0..257 {
        x[i] = a + i as f64 * step;
    }
    x[256] = b;
    x
}

fn arc_length(pts: &[Vec3]) -> f64 {
    let mut s = 0.0;
    for i in 1..pts.len() {
        s += norm(pts[i] - pts[i - 1]);
    }
    s
}

// the B-spline patches of a file as a mesh (for the edge lengths of the panel-size rule), the knots
// of every patch checked first (the mesh checks them again with the rest)
fn spline_mesh(g: &GdfFile) -> Result<Mesh> {
    let mut m = Mesh {
        patch: true,
        geometry: PatchKind::Spline,
        count: g.count,
        orders: g.orders.clone(),
        ncoef: g.counts.clone(),
        knots: g.knots.clone(),
        coefficients: g.coefficients.clone(),
        ..Mesh::default()
    };
    m.layout = spline_layout(&m);
    for p in 0..g.count as usize {
        let (ku, kv) = (g.orders[2 * p], g.orders[2 * p + 1]);
        let (mu, mv) = (g.counts[2 * p], g.counts[2 * p + 1]);
        let k0 = m.layout.knot_offset[p] as usize;
        let tu = &g.knots[k0..k0 + (mu + ku) as usize];
        let tv = &g.knots[k0 + (mu + ku) as usize..k0 + (mu + ku + mv + kv) as usize];
        let mut why = check_knots(tu, mu, ku, "u");
        if why.is_empty() {
            why = check_knots(tv, mv, kv, "v");
        }
        if !why.is_empty() {
            return Err(Error::value(format!("patch {}: {why}", p + 1)));
        }
    }
    Ok(m)
}

fn size_analytic(d: &AnalyticPatchDesc, o: &PatchOptions) -> Result<PatchDiscretization> {
    let p = AnalyticPatch {
        kind: d.kind,
        profile: d.profile,
        a: d.a,
        b: d.b,
        theta0: d.theta0,
        theta1: d.theta1,
        p: d.p,
        vt: d.vertices,
    };
    let (lu, lv) = edge_lengths(|u, v| {
        let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        p.eval(u, v, &mut x, &mut xu, &mut xv);
        x
    });
    size_discretization(lu, lv, o.panel_size, o)
}

/// The discretization of every patch of a mesh from a panel size: the rule of size_discretization
/// with the arc lengths of the patch edges measured on the mesh (edge_lengths; any patch kind, also
/// rational), for the geometry files (hbg.rs) and the B-spline patches of a GDF file.
pub fn panel_size_discretization(mesh: &Mesh, o: &PatchOptions) -> Result<Vec<PatchDiscretization>> {
    let mut out = Vec::new();
    for p in 0..mesh.count {
        let (lu, lv) = edge_lengths(|u, v| {
            let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
            record_eval(mesh, p, u, v, &mut x, &mut xu, &mut xv);
            x
        });
        out.push(size_discretization(lu, lv, o.panel_size, o)?);
    }
    Ok(out)
}

#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn size_flat(v: &[f64], o: &PatchOptions) -> Result<PatchDiscretization> {
    let P = |k: usize| Vec3([v[3 * k], v[3 * k + 1], v[3 * k + 2]]);
    let lu = max_of(norm(P(1) - P(0)), norm(P(2) - P(3)));
    let lv = max_of(norm(P(2) - P(1)), norm(P(3) - P(0)));
    size_discretization(lu, lv, o.panel_size, o)
}

// ---------------------------------------------------------------- analytic shapes

const QUADRANT: f64 = 0.5 * PI;

fn shape_name(igdef: i32) -> Option<&'static str> {
    match igdef {
        -1 => Some("circular_cylinder"),
        -2 => Some("elliptic_cylinder"),
        -3 => Some("sphere"),
        -4 => Some("ellipsoid"),
        -5 => Some("barge"),
        -7 => Some("moonpool_cylinder"),
        -34 => Some("cylinder_with_dipole_discs"),
        _ => None,
    }
}

fn shape_npatch(igdef: i32) -> Vec<i32> {
    match igdef {
        -1 | -2 => vec![1, 2, 3],
        -3 | -4 => vec![1, 2],
        -5 => vec![1, 2, 3, 4],
        -7 => vec![3, 4],
        _ => (2..20).collect(),
    }
}

fn tuple_text(v: &[i32]) -> String {
    let mut s = "(".to_string();
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s += ", ";
        }
        s += &x.to_string();
    }
    s + if v.len() == 1 { ",)" } else { ")" }
}

// ---------------------------------------------------------------- control surfaces

const GAUSS: i32 = 3; // Gauss points per panel side of a control surface

// the division of a control-surface patch by the panel size: spline order 3, 3 x 3 Gauss points
fn csf_options(size: f64) -> PatchOptions {
    PatchOptions {
        panel_size: size,
        spline_order: 3,
        outer_quadrature: GAUSS,
        inner_quadrature: GAUSS,
        ..PatchOptions::default()
    }
}

fn positive(v: &[f64], names: &[&str]) -> Result<()> {
    for (i, &x) in v.iter().enumerate() {
        if !(x.is_finite() && x > 0.0) {
            return Err(Error::value(format!("control surface: {} must be positive and finite", names[i])));
        }
    }
    Ok(())
}

fn revolved(kind: AnalyticKind, a: f64, b: f64, p0: f64, p1: f64, p2: f64, p3: f64) -> AnalyticPatchDesc {
    AnalyticPatchDesc {
        kind,
        profile: Profile::Linear,
        a,
        b,
        theta0: 0.0,
        theta1: QUADRANT,
        p: [p0, p1, p2, p3],
        ..AnalyticPatchDesc::default()
    }
}

fn csf_discretization(patches: &[AnalyticPatchDesc], size: f64) -> Result<Vec<PatchDiscretization>> {
    if !(size.is_finite() && size > 0.0) {
        return Err(Error::value("the control-surface panel size (PSZCSF) must be a positive finite length"));
    }
    patches.iter().map(|p| size_analytic(p, &csf_options(size))).collect()
}

// ---------------------------------------------------------------- writing

fn vertex_rows(v: &[f64], n: i64) -> String {
    let mut out = String::new();
    for i in 0..n as usize {
        out += &py_repr(v[3 * i]);
        out += " ";
        out += &py_repr(v[3 * i + 1]);
        out += " ";
        out += &py_repr(v[3 * i + 2]);
        out += "\n";
    }
    out
}

fn first_line(label: &str) -> String {
    let lines = split_lines(label);
    if lines.is_empty() || label.is_empty() { "HydroBEM mesh".to_string() } else { lines[0].clone() }
}

fn head(label: &str, ulen: f64, gravity: f64, isx: i32, isy: i32) -> String {
    format!(
        "{}\n{} {}   ULEN GRAV\n{isx} {isy}   ISX ISY\n",
        first_line(label),
        py_repr(ulen),
        py_repr(if gravity.is_finite() { gravity } else { STANDARD_GRAVITY })
    )
}

// ---------------------------------------------------------------- readers

/// The descriptor of an analytic patch.
pub fn analytic_desc(q: &AnalyticPatch) -> AnalyticPatchDesc {
    AnalyticPatchDesc {
        kind: q.kind,
        profile: q.profile,
        a: q.a,
        b: q.b,
        theta0: q.theta0,
        theta1: q.theta1,
        p: q.p,
        vertices: q.vt,
    }
}

pub fn read_low_order_gdf(path: &str, zero_trailer: bool) -> Result<GdfFile> {
    let h = read_header(path, "NPAN")?;
    let mut g =
        GdfFile { label: h.label.clone(), ulen: h.ulen, gravity: h.grav, isx: h.isx, isy: h.isy, ..GdfFile::default() };
    let npan = integer(leading_numbers(&h.text[3], 1, "NPAN", path)?[0], "NPAN", path)?;
    let mut r = records_after_header(path, &h);
    let mut values = r.rest()?;
    if npan < 1 {
        return Err(Error::value(format!("{path}: NPAN must be >= 1")));
    }
    let need = 12 * npan as usize;
    if values.len() < need {
        return Err(Error::value(format!(
            "{path}: the file ends after {} of NPAN = {npan} records",
            values.len() / 12
        )));
    }
    let _ = check_trailer(path, &values[need..], zero_trailer, "NPAN", npan)?;
    values.truncate(need);
    g.count = npan;
    g.vertices = values;
    Ok(g)
}

/// The SPL file of npatch patches (keyword values 0 when not given).
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn read_spl(
    path: &str,
    npatch: i32,
    spline_order: i32,
    outer_quadrature: i32,
    inner_quadrature: i32,
) -> Result<Vec<PatchDiscretization>> {
    let text = file_lines(path)?;
    if text.is_empty() {
        return Err(Error::value(format!("{path}: an SPL file needs a header line")));
    }
    // data lines start with a number; lines of text only (comments) are skipped
    let mut data: Vec<(usize, &str)> = Vec::new();
    for i in 1..text.len() {
        if strip(&text[i]).is_empty() {
            continue;
        }
        let t = tokens(&text[i]);
        if !t.is_empty() && is_number(&t[0]) {
            data.push((i + 1, &text[i]));
        }
    }
    let keys: [(&str, i32); 3] =
        [("KU KV", spline_order), ("IQUO IQVO", outer_quadrature), ("IQUI IQVI", inner_quadrature)];
    let mut pairs: Vec<&str> = vec!["NU NV"];
    for k in &keys {
        if k.1 == 0 {
            pairs.push(k.0);
        }
    }
    let expected = npatch.max(0) as usize * pairs.len();
    if data.len() != expected {
        let mut given = String::new();
        for k in &keys {
            if k.1 != 0 {
                if !given.is_empty() {
                    given += ", ";
                }
                given += k.0;
            }
        }
        if given.is_empty() {
            given = "none".to_string();
        }
        return Err(Error::value(format!(
            "{path}: expected {expected} data lines ({} pairs for each of {npatch} patches; pairs given by keywords: \
             {given}), found {}; a pair must be given either in the file or by its keyword, never both",
            pairs.len(),
            data.len()
        )));
    }
    let mut out = Vec::new();
    let mut it = 0usize;
    for p in 1..=npatch {
        let mut val = [[0i32; 2]; 4]; // panels, spline orders, outer and inner quadrature (u, v)
        for &pair in &pairs {
            let (lineno, line) = data[it];
            it += 1;
            let mut nums: Vec<String> = Vec::new();
            for t in tokens(line) {
                if !is_number(&t) {
                    break;
                }
                nums.push(t);
            }
            if nums.len() != 2 {
                return Err(Error::value(format!("{path}: line {lineno}: expected the pair {pair} of patch {p}")));
            }
            let slot = match pair {
                "NU NV" => 0,
                "KU KV" => 1,
                "IQUO IQVO" => 2,
                _ => 3,
            };
            val[slot][0] = integer(parse_number(&nums[0])?, pair, path)?;
            val[slot][1] = integer(parse_number(&nums[1])?, pair, path)?;
        }
        for k in 0..3 {
            if keys[k].1 != 0 {
                val[k + 1][0] = keys[k].1;
                val[k + 1][1] = keys[k].1;
            }
        }
        for k in 2..4 {
            if val[k][0] != val[k][1] {
                return Err(Error::value(format!(
                    "{path}: patch {p}: the directional quadrature orders {} must be equal, got {} {}",
                    if k == 2 { "IQUO IQVO" } else { "IQUI IQVI" },
                    val[k][0],
                    val[k][1]
                )));
            }
        }
        out.push(PatchDiscretization {
            nu: val[0][0],
            nv: val[0][1],
            ku: val[1][0],
            kv: val[1][1],
            iquo: val[2][0],
            iqui: val[3][0],
        });
    }
    Ok(out)
}

/// The patches of an analytic shape from the numbers of its GDF data lines: the geometry code
/// IGDEF < 0, NPATCH (the layout of its patches) and the numbers of its data lines, translated into
/// the shape's name, dimensions, parts and mapping (the mapping flag INONUMAP of the cylinders, the
/// sphere and the moonpool).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
pub fn analytic_shape(igdef: i32, npatch: i32, vals: &[f64]) -> Result<Vec<AnalyticPatchDesc>> {
    let Some(name) = shape_name(igdef) else {
        return Err(Error::value(format!(
            "IGDEF = {igdef} is not an available analytic shape (available: [-34, -7, -5, -4, -3, -2, -1])"
        )));
    };
    let allowed = shape_npatch(igdef);
    if !allowed.contains(&npatch) {
        return Err(Error::value(format!(
            "IGDEF = {igdef} ({name}) takes NPATCH in {}, got {npatch}",
            tuple_text(&allowed)
        )));
    }
    let at = format!("IGDEF = {igdef} ({name})");
    let need = |n: usize, names: &str| -> Result<Vec<f64>> {
        if vals.len() < n {
            return Err(Error::value(format!("{at} needs {names}")));
        }
        Ok(vals[..n].to_vec())
    };
    let flag = |i: usize| -> Result<String> {
        let v = if vals.len() > i { vals[i] } else { 0.0 };
        if !(v.abs() < 2147483647.0) {
            return Err(Error::value(format!("{at}: INONUMAP = {} is not available (0, 1)", py_repr(v))));
        }
        let f = v as i32; // truncated toward zero
        if f != 0 && f != 1 {
            return Err(Error::value(format!("{at}: INONUMAP = {f} is not available (0, 1)")));
        }
        Ok(if f == 1 { "nonuniform" } else { "uniform" }.to_string())
    };
    let named = |keys: &[&str], d: &[f64]| -> Vec<(String, f64)> {
        keys.iter().zip(d).map(|(k, v)| (k.to_string(), *v)).collect()
    };
    // the parts of the first NPATCH patches of the layout of the file
    let first = |parts: &[&str]| -> Vec<String> { parts[..npatch as usize].iter().map(|p| p.to_string()).collect() };
    let mut spec = ShapeSpec { shape: name.to_string(), ..ShapeSpec::default() };
    match igdef {
        -1 => {
            spec.dimensions = named(&["radius", "draft"], &need(2, "RADIUS DRAFT")?);
            spec.parts = first(&["side", "bottom", "waterplane"]);
            spec.mapping = flag(2)?;
        }
        -2 => {
            spec.dimensions = named(&["a", "b", "draft"], &need(3, "A B DRAFT")?);
            spec.parts = first(&["side", "bottom", "waterplane"]);
            spec.mapping = flag(3)?;
        }
        -3 => {
            spec.dimensions = named(&["radius"], &need(1, "RADIUS")?);
            spec.parts = first(&["hull", "waterplane"]);
            spec.mapping = flag(1)?;
        }
        -4 => {
            spec.dimensions = named(&["a", "b", "c"], &need(3, "A B C")?);
            spec.parts = first(&["hull", "waterplane"]);
        }
        -5 => {
            spec.dimensions = named(&["half_length", "half_beam", "draft"], &need(3, "HALFLEN HALFBEAM DRAFT")?);
            // NPATCH = 1: the bottom alone
            spec.parts =
                if npatch == 1 { vec!["bottom".into()] } else { first(&["end", "side", "bottom", "waterplane"]) };
        }
        -7 => {
            spec.dimensions = named(&["radius", "draft", "moonpool_radius"], &need(3, "RADIUS DRAFT RADMP")?);
            spec.parts = first(&["side", "bottom", "moonpool_wall", "moonpool_surface"]);
            spec.mapping = flag(3)?;
        }
        _ => {
            // -34: NMP NDIP, RADIUS DRAFT (RADMP), and RDOUT RDIN DDIP per dipole disc
            if vals.len() < 2 {
                return Err(Error::value("IGDEF = -34 (cylinder with dipole discs) needs NMP NDIP"));
            }
            let int_ok = vals[0].abs() < 2147483647.0 && vals[1].abs() < 2147483647.0;
            let nmp = if int_ok { vals[0] as i32 } else { -1 };
            let ndip = if int_ok { vals[1] as i32 } else { -1 };
            if (nmp != 0 && nmp != 1) || ndip < 0 || vals[0] != nmp as f64 || vals[1] != ndip as f64 {
                return Err(Error::value("IGDEF = -34: NMP must be 0 or 1 and NDIP a count"));
            }
            let k = 4 + nmp as usize;
            if vals.len() < k + 3 * ndip as usize {
                return Err(Error::value("IGDEF = -34 needs RADIUS DRAFT (RADMP) and RDOUT RDIN DDIP per dipole disc"));
            }
            spec.dimensions = named(&["radius", "draft"], &vals[2..4]);
            if nmp != 0 {
                spec.dimensions.push(("moonpool_radius".into(), vals[4]));
            }
            spec.discs =
                (0..ndip as usize).map(|i| [vals[k + 3 * i], vals[k + 3 * i + 1], vals[k + 3 * i + 2]]).collect();
            let base = 2 + nmp as usize;
            if npatch as usize != base + ndip as usize {
                return Err(Error::value(format!(
                    "IGDEF = -34: NPATCH = {npatch} for {base} body and {ndip} dipole patches (the damping lid on the \
                     moonpool is not available)"
                )));
            }
        }
    }
    named_shape(&spec).map_err(|e| Error::value(format!("IGDEF = {igdef}: {}", e.message())))
}

// B-spline patches: the panel counts nu nv, the orders ku kv, the u and v knot vectors
// (nu + 2 ku - 1 and nv + 2 kv - 1 values) and (nu + ku - 1)(nv + kv - 1) vector
// coefficients, the u index varying fastest
fn read_spline_patches(path: &str, r: &mut Records, npatch: i32, g: &mut GdfFile) -> Result<()> {
    for p in 1..=npatch {
        let at = format!("patch {p}");
        let nn = r.take(2, &format!("NUG NVG of {at}"))?;
        let nug = integer(nn[0], "NUG/NVG", path)?;
        let nvg = integer(nn[1], "NUG/NVG", path)?;
        let kk = r.take(2, &format!("KUG KVG of {at}"))?;
        let kug = integer(kk[0], "KUG/KVG", path)?;
        let kvg = integer(kk[1], "KUG/KVG", path)?;
        if nug < 1 || nvg < 1 {
            return Err(Error::value(format!("{path}: {at}: NUG and NVG must be >= 1")));
        }
        if kug < 2 || kvg < 2 {
            return Err(Error::value(format!("{path}: {at}: KUG and KVG must be >= 2")));
        }
        let u = r.take(nug as usize + 2 * kug as usize - 1, &format!("the u knot vector of {at}"))?;
        let v = r.take(nvg as usize + 2 * kvg as usize - 1, &format!("the v knot vector of {at}"))?;
        let mu = nug + kug - 1;
        let mv = nvg + kvg - 1;
        g.orders.extend_from_slice(&[kug, kvg]);
        g.counts.extend_from_slice(&[mu, mv]);
        g.knots.extend_from_slice(&u);
        g.knots.extend_from_slice(&v);
        for k in 0..mu as i64 * mv as i64 {
            let c = r.take(3, &format!("coefficient {} of {at}", k + 1))?;
            g.coefficients.extend_from_slice(&c);
        }
    }
    Ok(())
}

// the discretization of the B-spline patches of a file by the panel size, the edges measured on
// the patches (panel_size_discretization)
fn spline_sizes(g: &mut GdfFile, opt: &PatchOptions) -> Result<()> {
    g.discretization = panel_size_discretization(&spline_mesh(g)?, opt)?;
    Ok(())
}

pub fn read_patch_gdf(path: &str, opt: &PatchOptions, zero_trailer: bool) -> Result<GdfFile> {
    let h = read_header(path, "NPATCH IGDEF")?;
    let mut g = GdfFile {
        higher_order: true,
        label: h.label.clone(),
        ulen: h.ulen,
        gravity: h.grav,
        isx: h.isx,
        isy: h.isy,
        ..GdfFile::default()
    };
    let ni = leading_numbers(&h.text[3], 2, "NPATCH IGDEF", path)?;
    let npatch = integer(ni[0], "NPATCH/IGDEF", path)?;
    let igdef = integer(ni[1], "NPATCH/IGDEF", path)?;
    if npatch < 1 {
        return Err(Error::value(format!("{path}: NPATCH must be >= 1")));
    }
    if igdef != 0 && igdef != 1 && igdef >= 0 {
        return Err(Error::value(format!(
            "{path}: IGDEF = {igdef} is not supported (0: flat patches, 1: B-spline patches, < 0: analytic shapes)"
        )));
    }
    g.count = npatch;
    let mut by_size = false;
    if opt.has_panel_size {
        if !opt.spl.is_empty() || opt.has_discretization {
            return Err(Error::value(
                "give the discretization by one of an SPL file, `discretization` or `panel_size`",
            ));
        }
        if !(opt.panel_size.is_finite() && opt.panel_size > 0.0) {
            return Err(Error::value("panel_size must be a positive finite length"));
        }
        by_size = true;
    } else if !opt.spl.is_empty() {
        if opt.has_discretization {
            return Err(Error::value("give the discretization either by an SPL file or by `discretization`, not both"));
        }
        g.discretization = read_spl(&opt.spl, npatch, opt.spline_order, opt.outer_quadrature, opt.inner_quadrature)?;
    } else {
        let mut given = String::new();
        let kw: [(&str, i32); 3] = [
            ("spline_order", opt.spline_order),
            ("outer_quadrature", opt.outer_quadrature),
            ("inner_quadrature", opt.inner_quadrature),
        ];
        for k in &kw {
            if k.1 != 0 {
                if !given.is_empty() {
                    given += ", ";
                }
                given += k.0;
            }
        }
        if !given.is_empty() {
            return Err(Error::value(format!("{given} only complete an SPL file; without one give `discretization`")));
        }
        if !opt.has_discretization {
            return Err(Error::value(format!(
                "without an SPL file, give `discretization`: one row for every patch, or one per patch ({npatch})"
            )));
        }
        if opt.discretization.len() != 1 && opt.discretization.len() != npatch as usize {
            return Err(Error::value(format!(
                "`discretization` has {} rows for NPATCH = {npatch} patches (one row for all, or one per patch)",
                opt.discretization.len()
            )));
        }
        // one row: the discretization of every patch
        g.discretization = if opt.discretization.len() == 1 {
            vec![opt.discretization[0]; npatch as usize]
        } else {
            opt.discretization.clone()
        };
    }
    let mut r = records_after_header(path, &h);
    if igdef < 0 {
        // analytic shape: the number of data lines, then that many data lines
        // whose leading numbers are the dimensions and flags (a comment may follow)
        let nlines = integer(r.take(1, "NLINES")?[0], "NLINES", path)?;
        let values = r.data_lines(nlines)?;
        g.analytic = analytic_shape(igdef, npatch, &values)?;
        g.geometry = PatchKind::Analytic;
        if by_size {
            for p in &g.analytic {
                g.discretization.push(size_analytic(p, opt)?);
            }
        }
        return Ok(g);
    }
    if igdef == 0 {
        // one list-directed read of 12 numbers per patch: text after the twelfth number of a record
        // is a comment
        for p in 0..npatch {
            let v = r.take(12, &format!("the vertices of patch {}", p + 1))?;
            g.vertices.extend_from_slice(&v);
        }
        let _ = check_trailer(path, &r.rest()?, zero_trailer, "NPATCH", npatch)?;
        g.geometry = PatchKind::Flat;
        if by_size {
            for p in 0..npatch as usize {
                let d = size_flat(&g.vertices[12 * p..], opt)?;
                g.discretization.push(d);
            }
        }
        return Ok(g);
    }
    read_spline_patches(path, &mut r, npatch, &mut g)?;
    let _ = check_trailer(path, &r.rest()?, zero_trailer, "NPATCH", npatch)?;
    g.geometry = PatchKind::Spline;
    if by_size {
        spline_sizes(&mut g, opt)?;
    }
    Ok(g)
}

/// The analytic control surfaces of kind -1001 (a circular cylinder: radius, draft, inner
/// radius) and -1003 (an ellipsoid: semi-axes a, b, c, inner semi-axes a, b), one quadrant, every
/// patch divided by the panel size.
pub fn analytic_control_surface(
    icdef: i32,
    v: &[f64],
    panel_size: f64,
    patches: &mut Vec<AnalyticPatchDesc>,
    discretization: &mut Vec<PatchDiscretization>,
) -> Result<()> {
    if icdef == -1001 {
        if v.len() < 3 {
            return Err(Error::value(
                "control surface: the analytic cylinder (ICDEF = -1001) needs its radius, draft and inner radius",
            ));
        }
        positive(&[v[0], v[1], v[2]], &["the radius", "the draft", "the inner radius"])?;
        let (r, d, ri) = (v[0], v[1], v[2]);
        if ri >= r {
            return Err(Error::value(
                "control surface: the inner radius of the analytic cylinder (ICDEF = -1001) must be smaller than its radius",
            ));
        }
        *patches = vec![
            revolved(AnalyticKind::Line, r, r, 1.0, 0.0, 1.0, -d),
            revolved(AnalyticKind::Line, r, r, 1.0, -d, 0.0, -d),
            revolved(AnalyticKind::Line, r, r, ri / r, 0.0, 1.0, 0.0),
        ];
    } else if icdef == -1003 {
        if v.len() < 5 {
            return Err(Error::value(
                "control surface: the analytic ellipsoid (ICDEF = -1003) needs its semi-axes a, b, c and inner semi-axes a, b",
            ));
        }
        positive(
            &[v[0], v[1], v[2], v[3], v[4]],
            &[
                "the semi-axis a",
                "the semi-axis b",
                "the semi-axis c",
                "the inner semi-axis a",
                "the inner semi-axis b",
            ],
        )?;
        if v[3] >= v[0] || v[4] >= v[1] {
            return Err(Error::value(
                "control surface: the inner semi-axes of the analytic ellipsoid (ICDEF = -1003) must be smaller than the outer ones",
            ));
        }
        *patches = vec![
            revolved(AnalyticKind::Arc, v[0], v[1], v[2], 0.0, QUADRANT, 0.0),
            revolved(AnalyticKind::Ellipses, v[0], v[1], v[3], v[4], 0.0, 0.0),
        ];
    } else {
        return Err(Error::value(format!(
            "control surface: ICDEF = {icdef} is not an analytic control surface (-1001 the cylinder, -1003 the ellipsoid)"
        )));
    }
    *discretization = csf_discretization(patches, panel_size)?;
    Ok(())
}

/// Reads the panels of a free-surface file (FDF: a header, RINNER, NPF NTCL, NAL DELR NCIRE
/// NGSP, then NPF panels of their four x and their four y in z = 0) as a panels mesh in z = 0 with
/// the symmetry flags given (the file holds none: that of the model it belongs to). The lines of
/// the format are checked (RINNER a positive length); only the panels are taken: the radius and
/// the parameters of the zones beyond (NAL, DELR, NCIRE, NGSP) are the engine's own
/// (theory §12.9).
pub fn read_fdf(path: &str, isx: i32, isy: i32) -> Result<GdfFile> {
    let text = file_lines(path)?;
    if text.len() < 4 {
        return Err(Error::value(format!(
            "{path}: an FDF file needs a header, RINNER, NPF NTCL and NAL DELR NCIRE NGSP"
        )));
    }
    let rinner = leading_numbers(&text[1], 1, "RINNER", path)?[0];
    if !(rinner > 0.0 && rinner.is_finite()) {
        return Err(Error::value(format!("{path}: RINNER must be positive and finite")));
    }
    let npf = integer(leading_numbers(&text[2], 2, "NPF NTCL", path)?[0], "NPF", path)?;
    if npf < 1 {
        return Err(Error::value(format!("{path}: NPF must be >= 1")));
    }
    let _ = leading_numbers(&text[3], 4, "NAL DELR NCIRE NGSP", path)?;
    let mut r = Records::new(path, text[4..].to_vec(), 5);
    let mut g = GdfFile { label: strip(&text[0]), ulen: 1.0, isx, isy, count: npf, ..GdfFile::default() };
    for k in 0..npf {
        let v = r.take(8, &format!("the four x and four y of panel {}", k + 1))?;
        for i in 0..4 {
            g.vertices.extend_from_slice(&[v[i], v[4 + i], 0.0]);
        }
    }
    Ok(g)
}

/// A control-surface file: panels (method flag 0) or patches (1), each divided by the panel size
/// of the file; the automatic surface (a patch count <= 0) needs the body mesh (else None) and the
/// free surface in body axes.
pub fn read_csf(path: &str, zero_trailer: bool, body: Option<&Mesh>, waterline_z: f64) -> Result<GdfFile> {
    let text = file_lines(path)?;
    if text.len() < 2 {
        return Err(Error::value(format!("{path}: a CSF file needs a header and ILOWHICSF")));
    }
    let mut g = GdfFile { label: strip(&text[0]), ..GdfFile::default() };
    let low = integer(leading_numbers(&text[1], 1, "ILOWHICSF", path)?[0], "ILOWHICSF", path)?;
    if low != 0 && low != 1 {
        return Err(Error::value(format!(
            "{path}: ILOWHICSF = {low} is not available (0: panels, 1: patches; two combined files, ILOWHICSF = 2, \
             are not available)"
        )));
    }
    if text.len() < 4 {
        return Err(Error::value(format!(
            "{path}: a CSF file needs a header, ILOWHICSF, ISXCSF ISYCSF and a count line"
        )));
    }
    let s = leading_numbers(&text[2], 2, "ISXCSF ISYCSF", path)?;
    g.isx = symmetry_flag(s[0], "ISXCSF", path)?;
    g.isy = symmetry_flag(s[1], "ISYCSF", path)?;
    let mut r = Records::new(path, text[4..].to_vec(), 5);
    if low == 0 {
        let npan = integer(leading_numbers(&text[3], 1, "NPANCSF", path)?[0], "NPANCSF", path)?;
        let mut values = r.rest()?;
        if npan < 1 {
            return Err(Error::value(format!("{path}: NPANCSF must be >= 1")));
        }
        let need = 12 * npan as usize;
        if values.len() < need {
            return Err(Error::value(format!(
                "{path}: the file ends after {} of NPANCSF = {npan} records",
                values.len() / 12
            )));
        }
        let _ = check_trailer(path, &values[need..], zero_trailer, "NPANCSF", npan)?;
        values.truncate(need);
        g.count = npan;
        g.vertices = values;
        return Ok(g);
    }
    g.higher_order = true;
    let nis = leading_numbers(&text[3], 3, "NPATCSF ICDEF PSZCSF", path)?;
    let npatch = integer(nis[0], "NPATCSF", path)?;
    let icdef = integer(nis[1], "ICDEF", path)?;
    let size = nis[2];
    if !(size > 0.0 && size.is_finite()) {
        return Err(Error::value(format!(
            "{path}: PSZCSF must be positive and finite (the .csp subdivision, PSZCSF < 0, is not available)"
        )));
    }
    g.count = npatch;
    if npatch <= 0 {
        // the automatic control surface
        if icdef != 0 {
            return Err(Error::value(format!("{path}: an automatic control surface (NPATCSF <= 0) needs ICDEF = 0")));
        }
        let Some(body) = body else {
            return Err(Error::value(format!("{path}: an automatic control surface needs the body mesh")));
        };
        let rd = r.take(2, "RADIUS DEPTH")?;
        let npart = integer(r.take(1, "NPART")?[0], "NPART", path)?;
        let mut parts: Vec<Vec<f64>> = Vec::new();
        for k in 1..=npart {
            let nv_name = format!("NV({k})");
            let nv = integer(r.take(1, &nv_name)?[0], &nv_name, path)?;
            let mut xy = Vec::new();
            for _ in 0..nv {
                let p = r.take(2, &format!("X Y of partition {k}"))?;
                xy.extend_from_slice(&p);
            }
            parts.push(xy);
        }
        let (radius, depth) = (rd[0], rd[1]);
        if radius <= 0.0 && parts.is_empty() {
            return Err(Error::value(format!(
                "{path}: RADIUS <= 0 needs the outer boundary as the first partition (NPART >= 1)"
            )));
        }
        if g.isx != body.isx || g.isy != body.isy {
            return Err(Error::value(format!(
                "{path}: ISXCSF ISYCSF must equal the symmetry of the body (({}, {}))",
                body.isx, body.isy
            )));
        }
        positive(&[depth], &["DEPTH"])?;
        let mut d = ControlSurfaceDesc {
            radius: if radius > 0.0 { radius } else { -1.0 },
            depth,
            panel_size: size,
            waterline_z,
            free_surface: if npatch == 0 { 1 } else { 0 },
            ..ControlSurfaceDesc::default()
        };
        let mut xy: Vec<f64> = Vec::new();
        let mut counts: Vec<i32> = Vec::new();
        let mut first = 0usize;
        if radius <= 0.0 {
            d.polygon = parts[0].clone();
            d.polygon_count = (parts[0].len() / 2) as i32;
            first = 1;
        }
        for part in &parts[first..] {
            xy.extend_from_slice(part);
            counts.push((part.len() / 2) as i32);
        }
        d.partition_count = counts.len() as i32;
        d.partitions = xy;
        d.partition_counts = counts;
        let c = automatic_control_surface(body, &d)?;
        g.geometry = PatchKind::Analytic;
        g.count = c.patches.len() as i32;
        g.isx = c.symmetry[0];
        g.isy = c.symmetry[1];
        for q in &c.patches {
            g.analytic.push(analytic_desc(q));
        }
        g.discretization = c.discretization;
        return Ok(g);
    }
    if icdef < 0 {
        let nlines = integer(r.take(1, "NLINES")?[0], "NLINES", path)?;
        let values = r.data_lines(nlines)?;
        g.geometry = PatchKind::Analytic;
        if icdef == -1001 || icdef == -1003 {
            if g.isx != 1 || g.isy != 1 {
                return Err(Error::value(format!(
                    "{path}: ICDEF = {icdef} defines one quadrant (ISXCSF = ISYCSF = 1)"
                )));
            }
            let need = if icdef == -1001 { 3 } else { 5 };
            if values.len() < need {
                return Err(Error::value(format!("{path}: ICDEF = {icdef} needs {need} numbers")));
            }
            let mut patches = Vec::new();
            let mut disc = Vec::new();
            analytic_control_surface(icdef, &values[..need], size, &mut patches, &mut disc)?;
            if npatch as usize > patches.len() {
                return Err(Error::value(format!("{path}: ICDEF = {icdef} has at most {} patches", patches.len())));
            }
            g.analytic = patches[..npatch as usize].to_vec();
            g.discretization = disc[..npatch as usize].to_vec();
            return Ok(g);
        }
        g.analytic = analytic_shape(icdef, npatch, &values)?;
        g.discretization = csf_discretization(&g.analytic, size)?;
        return Ok(g);
    }
    let o = csf_options(size);
    if icdef == 0 {
        for p in 0..npatch {
            let v = r.take(12, &format!("the vertices of patch {}", p + 1))?;
            g.vertices.extend_from_slice(&v);
        }
        let _ = check_trailer(path, &r.rest()?, zero_trailer, "NPATCSF", npatch)?;
        g.geometry = PatchKind::Flat;
        for p in 0..npatch as usize {
            let d = size_flat(&g.vertices[12 * p..], &o)?;
            g.discretization.push(d);
        }
        return Ok(g);
    }
    if icdef == 1 {
        read_spline_patches(path, &mut r, npatch, &mut g)?;
        let _ = check_trailer(path, &r.rest()?, zero_trailer, "NPATCSF", npatch)?;
        g.geometry = PatchKind::Spline;
        spline_sizes(&mut g, &o)?;
        return Ok(g);
    }
    Err(Error::value(format!(
        "{path}: ICDEF = {icdef} is not available (0 flat, 1 B-spline, < 0 analytic; MultiSurf, ICDEF = 2, is not \
         available)"
    )))
}

// ---------------------------------------------------------------- writers

/// GDF text of a low-order mesh with every number at full precision (py_repr), as the readers read
/// it back.
pub fn gdf_text(m: &LowOrderMeshDesc, label: &str) -> String {
    head(label, m.reference_length, m.gravity, m.symmetry_x, m.symmetry_y)
        + &format!("{}   NPAN\n", m.panel_count)
        + &vertex_rows(&m.vertices, 4 * m.panel_count as i64)
}

/// GDF text of a patch mesh, as gdf_text.
pub fn gdf_text_patch(m: &PatchMeshDesc, label: &str) -> Result<String> {
    let mut out = head(label, m.reference_length, m.gravity, m.symmetry_x, m.symmetry_y);
    if m.geometry == PatchKind::Flat {
        out += &format!("{} 0   NPATCH IGDEF\n", m.patch_count);
        return Ok(out + &vertex_rows(&m.vertices, 4 * m.patch_count as i64));
    }
    if m.geometry != PatchKind::Spline {
        return Err(Error::value("a GDF file is written for flat or B-spline patches only"));
    }
    if m.weights.iter().any(|&w| w != 1.0) {
        return Err(Error::value(
            "a GDF file has no weights: a mesh with rational patches (weights other than 1) is not written",
        ));
    }
    out += &format!("{} 1   NPATCH IGDEF\n", m.patch_count);
    let (mut knot, mut coef) = (0usize, 0usize);
    for p in 0..m.patch_count as usize {
        let (ku, kv) = (m.spline_orders[2 * p], m.spline_orders[2 * p + 1]);
        let (mu, mv) = (m.spline_counts[2 * p], m.spline_counts[2 * p + 1]);
        out += &format!("{} {}\n", mu - ku + 1, mv - kv + 1);
        out += &format!("{ku} {kv}\n");
        for n in [mu + ku, mv + kv] {
            for i in 0..n as usize {
                if i > 0 {
                    out += " ";
                }
                out += &py_repr(m.knots[knot + i]);
            }
            out += "\n";
            knot += n as usize;
        }
        out += &vertex_rows(&m.coefficients[3 * coef..], mu as i64 * mv as i64);
        coef += mu as usize * mv as usize;
    }
    Ok(out)
}

/// SPL text of the discretization of a patch mesh.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn spl_text(m: &PatchMeshDesc, label: &str) -> String {
    let mut out = first_line(label) + "\n";
    for p in 0..m.patch_count as usize {
        let d = &m.discretization[p];
        out += &format!("{} {}\n{} {}\n{} {}\n{} {}\n", d.nu, d.nv, d.ku, d.kv, d.iquo, d.iquo, d.iqui, d.iqui);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_count_rule() -> Result<()> {
        assert_eq!(panel_count(1.0, 0.25)?, 5); // an exact multiple: floor(4 + 1e-5) + 1
        assert_eq!(panel_count(0.99999, 0.25)?, 4);
        assert_eq!(panel_count(1.1, 0.25)?, 5);
        assert!(panel_count(1e7, 1.0).is_err());
        Ok(())
    }

    #[test]
    fn shapes() -> Result<()> {
        let b = analytic_shape(-5, 1, &[2.0, 1.0, 0.5])?;
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].vertices[2], -0.5);
        assert_eq!(analytic_shape(-1, 3, &[1.0, 2.0, 1.0])?[0].profile, Profile::Smooth);
        let e = analytic_shape(-3, 3, &[1.0]).unwrap_err();
        assert_eq!(e.message(), "IGDEF = -3 (sphere) takes NPATCH in (1, 2), got 3");
        let e = analytic_shape(-1, 1, &[1.0, 2.0, 2.0]).unwrap_err();
        assert_eq!(e.message(), "IGDEF = -1 (circular_cylinder): INONUMAP = 2 is not available (0, 1)");
        let e = analytic_shape(-9, 1, &[]).unwrap_err();
        assert_eq!(
            e.message(),
            "IGDEF = -9 is not an available analytic shape (available: [-34, -7, -5, -4, -3, -2, -1])"
        );
        let p = analytic_shape(-34, 3, &[0.0, 1.0, 2.0, 1.0, 1.5, 0.5, 2.5])?; // the disc below the bottom
        assert_eq!(p.len(), 3);
        assert_eq!(p[2].p, [1.5, -2.5, 0.5, -2.5]);
        Ok(())
    }

    #[test]
    fn writers() -> Result<()> {
        let m = LowOrderMeshDesc {
            panel_count: 1,
            vertices: vec![0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, -0.1],
            reference_length: 1.0,
            gravity: f64::NAN,
            symmetry_x: 1,
            symmetry_y: 0,
        };
        assert_eq!(
            gdf_text(&m, "hull\nsecond"),
            "hull\n1.0 9.80665   ULEN GRAV\n1 0   ISX ISY\n1   NPAN\n0.0 0.0 -1.0\n1.0 0.0 -1.0\n1.0 1.0 -1.0\n0.0 1.0 -0.1\n"
        );
        assert!(gdf_text(&m, "").starts_with("HydroBEM mesh\n"));
        let pm = PatchMeshDesc {
            geometry: PatchKind::Spline,
            patch_count: 1,
            spline_orders: vec![2, 2],
            spline_counts: vec![2, 2],
            knots: vec![-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            coefficients: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            discretization: vec![PatchDiscretization { nu: 2, nv: 3, ku: 3, kv: 3, iquo: 2, iqui: 4 }],
            reference_length: 2.5,
            gravity: 9.81,
            ..PatchMeshDesc::default()
        };
        assert_eq!(
            gdf_text_patch(&pm, "p")?,
            "p\n2.5 9.81   ULEN GRAV\n0 0   ISX ISY\n1 1   NPATCH IGDEF\n1 1\n2 2\n-1.0 -1.0 1.0 1.0\n-1.0 -1.0 1.0 1.0\n\
             0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1.0 1.0 0.0\n"
        );
        assert_eq!(spl_text(&pm, "s"), "s\n2 3\n3 3\n2 2\n4 4\n");
        Ok(())
    }
}
