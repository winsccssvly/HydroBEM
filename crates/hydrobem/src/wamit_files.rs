//! The WAMIT-format numeric files and the .out summary of a results object (reference/file-formats
//! §1-§4): the nondimensional normalization of the format, the layouts of its numeric files,
//! Fortran edit descriptors E14.6 (0P and 1P) and Iw.
//!
//! From the quantities of a results object and its model, each numeric file with a header line.

use crate::codes::{ElementType, QuantityId};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use num_complex::Complex64;

use crate::build_info::build_info;
use crate::constants::PI;
use crate::document::{Document, Obj, symmetry_flags};
use crate::document_build::{body_mesh_name, build_model};
use crate::error::{Error, Result};
use crate::mesh_desc::MeshKind;
use crate::model_codes::Formulation;
use crate::problem::Problem;
use crate::quantities::quantity_info;
use crate::results::Results;
use crate::results_file::{io_message, load_quantity};
use crate::text::{file_exists, parent_folder, py_repr, py_str_repr, write_file};

// ---------------------------------------------------------------- the edit descriptors

fn exponent(e: i32, x: f64) -> Result<String> {
    if e.abs() > 99 {
        return Err(Error::value(format!("value {} needs a three-digit exponent", py_repr(x))));
    }
    Ok(format!("E{}{:02}", if e < 0 { '-' } else { '+' }, e.abs()))
}

fn pad(s: &str, width: i32) -> String {
    if s.len() as i32 >= width { s.to_string() } else { " ".repeat((width as usize) - s.len()) + s }
}

// the mantissa of `digits` decimals (correctly rounded, ties to even), E, the sign of the exponent
// and at least two exponent digits, right-aligned in width
fn sci_e(x: f64, width: usize, digits: usize) -> String {
    let s = format!("{x:.digits$e}");
    let k = s.find('e').expect("the exponent of a scientific format");
    let e: i32 = s[k + 1..].parse().expect("an integer exponent");
    let t = format!("{}E{}{:02}", &s[..k], if e < 0 { '-' } else { '+' }, e.abs());
    format!("{t:>width$}")
}

// x with `digits` decimals right-aligned in width (correctly rounded, ties to even), in the E form
// of sci_e for |x| >= 1e7; a negative zero is written as zero (as fortran_e)
fn fixed(x: f64, width: i32, digits: i32) -> String {
    // a quantity without a value (NaN: a body without mass) as in the limit tables
    if !x.is_finite() {
        return " ".repeat(0.max(width - 9) as usize) + "undefined";
    }
    let x = if x == 0.0 { 0.0 } else { x };
    let (w, d) = (width as usize, digits as usize);
    if x.abs() < 1e7 { format!("{x:>w$.d$}") } else { sci_e(x, w, d) }
}

// the diffraction formulation of the solve: the one given, or for auto that of the problem built
// from the solved model (the rule of the build, document_build.rs); only auto builds, so that a
// formulation given is written without the cost of a build and whatever a later build would say
fn solved_formulation(m: &Document) -> Result<&'static str> {
    let given = single(m, "first_order").choice("diffraction_formulation")?;
    match given {
        "total" => return Ok("total"),
        "scattered" => return Ok("scattered"),
        _ => {}
    }
    let mut p = Problem::default();
    build_model(m, &mut p).map_err(|e| {
        let message = format!("write_wamit: the formulation of the solved model: {}", e.message());
        e.with_message(message)
    })?;
    Ok(match p.opt.formulation {
        Formulation::Total => "total",
        Formulation::Scattered => "scattered",
    })
}

fn boolean(b: bool) -> &'static str {
    if b { "True" } else { "False" }
}

fn degrees(z: Complex64) -> f64 {
    if z == Complex64::new(0.0, 0.0) { 0.0 } else { z.im.atan2(z.re) * (180.0 / PI) }
}

fn complex_cols(z: Complex64) -> Result<String> {
    Ok(fortran_e(z.re.hypot(z.im), true, 14, 6)?
        + &fortran_e(degrees(z), true, 14, 6)?
        + &fortran_e(z.re, true, 14, 6)?
        + &fortran_e(z.im, true, 14, 6)?)
}

fn e1p(x: f64) -> Result<String> {
    fortran_e(x, true, 14, 6)
}

fn e0p(x: f64) -> Result<String> {
    fortran_e(x, false, 14, 6)
}

// ---------------------------------------------------------------- the quantities

// a quantity as complex values (real ones with a zero imaginary part), its dimensions
#[derive(Clone, Default)]
struct Q {
    ok: bool,
    dims: Vec<i64>,
    v: Vec<Complex64>,
}

impl Q {
    fn at(&self, idx: &[i64]) -> usize {
        let mut o = 0usize;
        for (k, &i) in idx.iter().enumerate() {
            o = o * self.dims[k] as usize + i as usize;
        }
        o
    }
    fn re(&self, idx: &[i64]) -> f64 {
        self.v[self.at(idx)].re
    }
    fn c(&self, idx: &[i64]) -> Complex64 {
        self.v[self.at(idx)]
    }
}

fn fetch(r: &Results, id: QuantityId) -> Result<Q> {
    let mut out = Q::default();
    let Some(q) = r.quantities.get(&id) else { return Ok(out) };
    load_quantity(r, q, id.name())?;
    out.ok = true;
    out.dims = q.dims.clone();
    let mut n = 1usize;
    for &d in &q.dims {
        n *= d as usize;
    }
    let b = q.data();
    let f32_at = |k: usize| f32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap());
    let f64_at = |k: usize| f64::from_le_bytes(b[8 * k..8 * k + 8].try_into().unwrap());
    out.v = Vec::with_capacity(n);
    for k in 0..n {
        let z = match q.dtype {
            ElementType::Float32 => Complex64::new(f32_at(k) as f64, 0.0),
            ElementType::Float64 => Complex64::new(f64_at(k), 0.0),
            ElementType::Int32 => {
                Complex64::new(i32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap()) as f64, 0.0)
            }
            ElementType::ComplexF32 => Complex64::new(f32_at(2 * k) as f64, f32_at(2 * k + 1) as f64),
            ElementType::ComplexF64 => Complex64::new(f64_at(2 * k), f64_at(2 * k + 1)),
            _ => return Err(Error::internal("write_wamit: a quantity of an unknown element type")),
        };
        out.v.push(z);
    }
    Ok(out)
}

// ---------------------------------------------------------------- the model of the results

fn body_of(d: &Document, b: usize) -> Obj<'_> {
    d.root().children("bodies")[b]
}

fn single<'a>(d: &'a Document, list: &str) -> Obj<'a> {
    d.root().children(list)[0]
}

// the field points of the files, global: those of the sea, then those of the tanks placed with their
// bodies (the points of the solve; the results hold the two apart)
fn field_points(d: &Document) -> Result<Vec<f64>> {
    Ok(crate::document_build::field_points(d)?.0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect())
}

// the counts of the field points in the sea and in the tanks of a model
fn field_point_counts(d: &Document) -> (i64, i64) {
    (crate::document_build::sea_points(d).len() as i64, crate::document_build::tank_points(d).0.len() as i64)
}

// a field quantity of the files at the field points of the files: the rows of the sea's quantity,
// then those of the tanks' (`tank`, the radiation; zero for the incident, scattered and total waves,
// which the fluid of a tank does not have); not ok when neither is available
fn joined(sea: Q, tank: Q, j: i64, n: i64) -> Q {
    if !sea.ok && !tank.ok {
        return sea;
    }
    let shape = if sea.ok { &sea.dims } else { &tank.dims };
    let row: usize = shape[2..].iter().product::<i64>() as usize;
    let f = shape[0] as usize;
    let mut dims = shape.clone();
    dims[1] = j + n;
    let mut v = Vec::with_capacity(f * (j + n) as usize * row);
    let zero = Complex64::new(0.0, 0.0);
    for fi in 0..f {
        if sea.ok {
            let a = fi * j as usize * row;
            v.extend_from_slice(&sea.v[a..a + j as usize * row]);
        }
        if tank.ok {
            let a = fi * n as usize * row;
            v.extend_from_slice(&tank.v[a..a + n as usize * row]);
        } else {
            v.extend(std::iter::repeat_n(zero, n as usize * row));
        }
    }
    Q { ok: true, dims, v }
}

// the mesh object of a name, or for a trimmed mesh its source (which has the header items)
fn mesh_head<'a>(d: &'a Document, name: &str) -> Result<Obj<'a>> {
    let m = d.root().child("meshes", name);
    let Some(m) = m else {
        return Err(Error::value(format!("write_wamit: the model has no mesh {}", py_str_repr(name))));
    };
    if m.mesh_kind() == MeshKind::TrimmedPatches {
        return mesh_head(d, m.get("source").text());
    }
    Ok(m)
}

fn records_of(mesh: Obj) -> i64 {
    let kind = mesh.mesh_kind();
    if matches!(kind, MeshKind::Panels | MeshKind::FlatPatches) {
        let v = mesh.get("vertices");
        return if v.shape().is_empty() { 0 } else { v.shape()[0] };
    }
    if kind == MeshKind::TrimmedPatches {
        let v = mesh.get("trim_elements");
        return if v.shape().is_empty() { 0 } else { v.shape()[0] };
    }
    mesh.children("patches").len() as i64
}

// every frequency of the model in rad/s, the limits 0 and inf with them (periods as 2 pi / T)
fn model_omegas(d: &Document) -> Vec<f64> {
    let f = single(d, "environment");
    if f.get("frequency_input").text() == "periods" {
        f.get("periods").reals().iter().map(|&t| if t != 0.0 { 2.0 * PI / t } else { f64::INFINITY }).collect()
    } else {
        f.get("omegas").reals().to_vec()
    }
}

#[derive(Clone)]
struct Coord {
    body: i32,
    rotational: bool,
    name: String,
}

const MODE_NAMES: [&str; 6] = ["surge", "sway", "heave", "roll", "pitch", "yaw"];

fn pressure_mode_names(body: Obj) -> Vec<String> {
    let mut out = Vec::new();
    for ps in body.children("pressure_surfaces") {
        for m in ps.children("pressure_modes") {
            out.push(m.name().to_string());
        }
    }
    out
}

fn coordinates(d: &Document) -> Vec<Coord> {
    let mut out = Vec::new();
    let bodies = d.root().children("bodies");
    for (b, body) in bodies.iter().enumerate() {
        for (j, name) in MODE_NAMES.iter().enumerate() {
            out.push(Coord { body: b as i32, rotational: j >= 3, name: name.to_string() });
        }
        for g in body.children("generalized_modes") {
            out.push(Coord { body: b as i32, rotational: false, name: g.name().to_string() });
        }
        for n in pressure_mode_names(*body) {
            out.push(Coord { body: b as i32, rotational: false, name: n });
        }
    }
    out
}

// ---------------------------------------------------------------- the files

type Files = BTreeMap<String, Vec<String>>;

#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
struct Writer<'a> {
    r: &'a Results,
    m: &'a Document,
    omega: bool,
    // the axes
    om: Vec<f64>,
    beta: Vec<f64>,
    ids: Vec<i32>,
    coords: Vec<Coord>,
    F: i64,
    H: i64,
    D: i64,
    K: i64,
    P: i64,
    J: i64,
    M: i64,
    rot: Vec<i32>, // r_i of every coordinate, rk of every column
    rk: Vec<i32>,
    L: f64,
    rho: f64,
    g: f64,
    per: Vec<String>,
    bcol: Vec<String>,
    fp: Vec<f64>, // field points [P][3]: the sea's J, then the tanks' M
    has_fp: bool,
}

impl<'a> Writer<'a> {
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn new(r: &'a Results, m: &'a Document, omega: bool) -> Result<Writer<'a>> {
        let qo = fetch(r, QuantityId::Omegas)?;
        let qh = fetch(r, QuantityId::Headings)?;
        let qi = fetch(r, QuantityId::RadiationModeIds)?;
        let om: Vec<f64> = if qo.ok {
            qo.v.iter().map(|z| z.re).collect()
        } else {
            model_omegas(m).into_iter().filter(|&x| x > 0.0 && x.is_finite()).collect()
        };
        let beta: Vec<f64> = if qh.ok {
            qh.v.iter().map(|z| z.re).collect()
        } else {
            single(m, "environment").get("headings").reals().to_vec()
        };
        let ids: Vec<i32> = qi.v.iter().map(|z| z.re as i32).collect();
        let coords = coordinates(m);
        let rot: Vec<i32> = coords.iter().map(|c| if c.rotational { 1 } else { 0 }).collect();
        let rk: Vec<i32> = ids.iter().map(|&j| rot[(j - 1) as usize]).collect();
        let L = mesh_head(m, &body_mesh_name(body_of(m, 0)).unwrap_or_default())?.get("reference_length").real();
        let env = single(m, "environment");
        let rho = env.get("density").real();
        let g = env.get("gravity").real();
        let fp = field_points(m)?;
        let has_fp = !fp.is_empty();
        let P = (fp.len() / 3) as i64;
        let (J, M) = field_point_counts(m);
        Ok(Writer {
            r,
            m,
            omega,
            F: om.len() as i64,
            H: beta.len() as i64,
            D: coords.len() as i64,
            K: ids.len() as i64,
            P,
            J,
            M,
            om,
            beta,
            ids,
            coords,
            rot,
            rk,
            L,
            rho,
            g,
            per: Vec::new(),
            bcol: Vec::new(),
            fp,
            has_fp,
        })
    }

    // the first columns of the numeric files, distinct at their printed precision
    fn columns(&mut self) -> Result<()> {
        for &x in &self.om {
            self.per.push(e0p(if self.omega { x } else { 2.0 * PI / x })?);
        }
        if self.per.iter().collect::<BTreeSet<_>>().len() != self.per.len() {
            return Err(Error::value("distinct frequencies collide at the printed precision of the numeric files"));
        }
        for &b in &self.beta {
            self.bcol.push(e0p(b)?);
        }
        if self.bcol.iter().collect::<BTreeSet<_>>().len() != self.bcol.len() {
            return Err(Error::value("distinct headings collide at the printed precision of the numeric files"));
        }
        Ok(())
    }

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn Lp(&self, e: i32) -> f64 {
        self.L.powf(e as f64)
    }

    // a field quantity of the sea normalized at the field points of the files (`joined`)
    fn field(&self, sea: QuantityId, tank: Option<QuantityId>) -> Result<Q> {
        let t = match tank {
            Some(t) => self.norm(t)?,
            None => Q::default(),
        };
        Ok(joined(self.norm(sea)?, t, self.J, self.M))
    }

    // the quantity q normalized as reference/file-formats §1 (A = 1 m)
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    fn norm(&self, id: QuantityId) -> Result<Q> {
        let mut q = fetch(self.r, id)?;
        if !q.ok {
            return Ok(q);
        }
        let dm = q.dims.clone();
        let (rho, g, L) = (self.rho, self.g, self.L);
        let (rot, rk, om) = (&self.rot, &self.rk, &self.om);
        match id {
            QuantityId::AddedMass | QuantityId::RadiationDamping => {
                for f in 0..dm[0] {
                    for i in 0..dm[1] {
                        for k in 0..dm[2] {
                            let e = 3 + rot[i as usize] + rk[k as usize];
                            let s = rho * self.Lp(e);
                            let at = q.at(&[f, i, k]);
                            let z = q.v[at];
                            q.v[at] = if id == QuantityId::AddedMass {
                                Complex64::new(z.re / s, 0.0)
                            } else {
                                Complex64::new(z.re / (om[f as usize] * rho * self.Lp(e)), 0.0)
                            };
                        }
                    }
                }
            }
            QuantityId::AddedMassZeroFrequency | QuantityId::AddedMassInfiniteFrequency => {
                for i in 0..dm[0] {
                    for k in 0..dm[1] {
                        let at = q.at(&[i, k]);
                        q.v[at] =
                            Complex64::new(q.v[at].re / (rho * self.Lp(3 + rot[i as usize] + rk[k as usize])), 0.0);
                    }
                }
            }
            QuantityId::ExcitationDiffraction
            | QuantityId::ExcitationFroudeKrylov
            | QuantityId::ExcitationScattering
            | QuantityId::ExcitationHaskind => {
                for f in 0..dm[0] {
                    for h in 0..dm[1] {
                        for i in 0..dm[2] {
                            let ri = if id == QuantityId::ExcitationHaskind { rk[i as usize] } else { rot[i as usize] };
                            let at = q.at(&[f, h, i]);
                            q.v[at] /= rho * g * self.Lp(2 + ri);
                        }
                    }
                }
            }
            QuantityId::Motion => {
                for f in 0..dm[0] {
                    for h in 0..dm[1] {
                        for i in 0..dm[2] {
                            let at = q.at(&[f, h, i]);
                            q.v[at] *= self.Lp(rot[i as usize]);
                        }
                    }
                }
            }
            QuantityId::DriftMomentumConservation => {
                for k in 0..q.v.len() {
                    q.v[k] /= rho * g * (if k % 3 == 2 { self.Lp(2) } else { L });
                }
            }
            QuantityId::DriftPressureIntegration | QuantityId::DriftControlSurface => {
                // on the coordinates: a force per m^2 (rho g L), a moment (rho g L^2)
                let d = dm[2] as usize;
                for k in 0..q.v.len() {
                    q.v[k] /= rho * g * self.Lp(1 + rot[k % d]);
                }
            }
            QuantityId::DriftPressureIntegrationBodyAxes | QuantityId::DriftControlSurfaceBodyAxes => {
                let s = rho * g * self.Lp(2);
                for z in q.v.iter_mut() {
                    *z /= s;
                }
            }
            QuantityId::FieldPointsPressureRadiation
            | QuantityId::TanksPressureRadiation
            | QuantityId::BodyPointsPressureRadiation => {
                for f in 0..dm[0] {
                    for p in 0..dm[1] {
                        for k in 0..dm[2] {
                            let at = q.at(&[f, p, k]);
                            q.v[at] /= self.Lp(rk[k as usize]);
                        }
                    }
                }
            }
            QuantityId::FieldPointsVelocityRadiation
            | QuantityId::TanksVelocityRadiation
            | QuantityId::BodyPointsVelocityRadiation => {
                // K L grad phi_j / L^n with grad phi_j = V / (i w) per unit displacement
                for f in 0..dm[0] {
                    for p in 0..dm[1] {
                        for k in 0..dm[2] {
                            for e in 0..3 {
                                let at = q.at(&[f, p, k, e]);
                                let z = q.v[at];
                                let w = om[f as usize];
                                let s = w * w / g * L;
                                let t = Complex64::new(s * z.re, s * z.im);
                                q.v[at] = Complex64::new(t.im / w, -t.re / w) / self.Lp(rk[k as usize]);
                            }
                        }
                    }
                }
            }
            QuantityId::FieldPointsVelocityIncident
            | QuantityId::FieldPointsVelocityScattering
            | QuantityId::FieldPointsVelocityDiffraction
            | QuantityId::BodyPointsVelocityDiffraction => {
                // V w L / (i g)
                for f in 0..dm[0] {
                    for p in 0..dm[1] {
                        for h in 0..dm[2] {
                            for e in 0..3 {
                                let at = q.at(&[f, p, h, e]);
                                let t = q.v[at] * om[f as usize] * L;
                                q.v[at] = Complex64::new(t.im / g, -t.re / g);
                            }
                        }
                    }
                }
            }
            _ => {} // the elevations and p / (rho g) of the incident and diffracted waves
        }
        Ok(q)
    }

    fn force_file(&self, a: &Q, rows: &[i32], by_column: bool) -> Result<Vec<String>> {
        let mut lines = Vec::new();
        for f in 0..self.F {
            for h in 0..self.H {
                for (n, &i) in rows.iter().enumerate() {
                    let z = if by_column { a.c(&[f, h, n as i64]) } else { a.c(&[f, h, i as i64 - 1]) };
                    lines.push(
                        self.per[f as usize].clone()
                            + &self.bcol[h as usize]
                            + &fortran_i(i as i64, 6)?
                            + &complex_cols(z)?,
                    );
                }
            }
        }
        Ok(lines)
    }
}

fn shape_text(d: &[i64]) -> String {
    let mut t = "(".to_string();
    for (k, x) in d.iter().enumerate() {
        if k > 0 {
            t += ", ";
        }
        t += &x.to_string();
    }
    t + if d.len() == 1 { ",)" } else { ")" }
}

/// Every quantity of results r has the shape of its axes (quantities.rs) on the axes of the
/// model the results are of: F, H, D, K from the axis quantities (which must match the model), P the
/// field points, B the bodies, and each other axis (lid points, body points, angles, elements) the same
/// extent in every quantity that has it; with complete, the hydrostatics all given or none (the
/// writers check with complete = true).
/// Fails with ErrorKind::Value naming the quantity and its shape.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn check_quantities(r: &Results, m: &Document, complete: bool) -> Result<()> {
    let axis = |q: QuantityId| -> Result<Vec<f64>> { Ok(fetch(r, q)?.v.iter().map(|z| z.re).collect()) };
    // the axis quantities: those of the model (Model::new_results puts them, Results::put refuses them)
    let model_om: Vec<f64> = model_omegas(m).into_iter().filter(|&x| x > 0.0 && x.is_finite()).collect();
    let om = axis(QuantityId::Omegas)?;
    let heads = axis(QuantityId::Headings)?;
    let ids = axis(QuantityId::RadiationModeIds)?;
    let model_heads = single(m, "environment").get("headings").reals().to_vec();
    if r.quantities.contains_key(&QuantityId::Omegas) && om != model_om {
        return Err(Error::value("omegas are not the ordinary frequencies of the model"));
    }
    if r.quantities.contains_key(&QuantityId::Headings) && heads != model_heads {
        return Err(Error::value("headings are not the headings of the model"));
    }
    let F = model_om.len() as i64;
    let H = model_heads.len() as i64;
    let D = coordinates(m).len() as i64;
    let K = ids.len() as i64;
    for &j in &ids {
        if !(j >= 1.0 && j <= D as f64 && j.fract() == 0.0) {
            return Err(Error::value(format!("radiation_mode_ids must be coordinates 1..{D} (integers)")));
        }
    }
    let B = m.root().children("bodies").len() as i64;
    let (J, M) = field_point_counts(m);
    let mut free_axes: BTreeMap<String, i64> = BTreeMap::new(); // lid points, body points, angles, elements, quads: one extent each
    for (&id, q) in &r.quantities {
        let info = quantity_info(id);
        let names: Vec<&str> = info.dims.split(',').collect();
        let mut ok = q.dims.len() == names.len();
        let mut k = 0;
        while ok && k < names.len() {
            let n = names[k];
            let e = q.dims[k];
            let want: i64 = match n {
                "frequency" => F,
                "heading" => H,
                "coordinate" => D,
                "radiation_column" => K,
                "field_point" => J,
                "tank_point" => M,
                "body" => B,
                "3*body" => 3 * B,
                _ if n.as_bytes().first().is_some_and(|c| c.is_ascii_digit()) => {
                    n.parse().expect("a numeric axis of the quantity table is an integer")
                }
                _ => -1,
            };
            if want >= 0 {
                ok = e == want;
            } else {
                match free_axes.get(n) {
                    Some(&x) => ok = x == e,
                    None => {
                        free_axes.insert(n.to_string(), e);
                    }
                }
            }
            k += 1;
        }
        if !ok {
            return Err(Error::value(format!(
                "{} has the shape {}, not that of its axes ({}) on the model",
                info.name,
                shape_text(&q.dims),
                info.dims
            )));
        }
    }
    let hq: Vec<QuantityId> =
        QuantityId::ALL.iter().copied().filter(|q| q.name().starts_with("hydrostatics_")).collect();
    let hydro = hq.iter().filter(|q| r.quantities.contains_key(q)).count();
    if complete && hydro != 0 && hydro != hq.len() {
        return Err(Error::value(format!(
            "the hydrostatics are given in part: give all {} hydrostatics_* or none",
            hq.len()
        )));
    }
    if !complete {
        return Ok(());
    }
    // the quantities the writers read together: a drift with its moment about the moving rotation centre
    // (body axes); the body point ids with their points, normals and areas, the ids in range
    let has = |q: QuantityId| r.quantities.contains_key(&q);
    let pair = |a: QuantityId, b: QuantityId| {
        if has(a) != has(b) {
            let missing = if has(a) { b } else { a };
            return Err(Error::value(format!(
                "{} and {} are given together (each needs the other): {} is missing",
                a.name(),
                b.name(),
                missing.name()
            )));
        }
        Ok(())
    };
    pair(QuantityId::DriftPressureIntegration, QuantityId::DriftPressureIntegrationBodyAxes)?;
    pair(QuantityId::DriftControlSurface, QuantityId::DriftControlSurfaceBodyAxes)?;
    let drifts =
        [QuantityId::DriftPressureIntegration, QuantityId::DriftControlSurface, QuantityId::DriftMomentumConservation];
    if let Some(d) = drifts.iter().find(|&&q| has(q)) {
        if !has(QuantityId::DriftHeadingPairs) {
            return Err(Error::value(format!(
                "{} needs drift_heading_pairs (the heading pairs of the drifts): drift_heading_pairs is missing",
                d.name()
            )));
        }
    }
    if has(QuantityId::BodyPointsIds) {
        let bids = fetch(r, QuantityId::BodyPointsIds)?;
        let mut specified = true;
        for q in 0..bids.dims[0] {
            let (b, sx, sy) = (bids.re(&[q, 0]), bids.re(&[q, 2]), bids.re(&[q, 3]));
            if !(b >= 0.0 && b < B as f64) || bids.re(&[q, 1]) < 1.0 || sx.abs() > 1.0 || sy.abs() > 1.0 {
                return Err(Error::value(format!(
                    "body_points_ids: row {} is not (body 0..{}, element >= 1, sx, sy in -1..1)",
                    q + 1,
                    B - 1
                )));
            }
            specified = specified && sx == 0.0 && sy == 0.0;
        }
        if !specified {
            for q in [QuantityId::BodyPoints, QuantityId::BodyPointsNormals, QuantityId::BodyPointsAreas] {
                if !has(q) {
                    return Err(Error::value(format!(
                        "body_points_ids of panels (a row with sx or sy not 0) need body_points, \
                         body_points_normals and body_points_areas: {} is missing",
                        q.name()
                    )));
                }
            }
        }
    }
    Ok(())
}

// shapes and availability rules checked before any file is written
fn preflight(w: &Writer) -> Result<()> {
    let r = w.r;
    check_quantities(r, w.m, true)?;
    let shape = |id: QuantityId, want: &[i64]| {
        if let Some(q) = r.quantities.get(&id) {
            if q.dims != want {
                return Err(Error::value(format!("{} does not have the shape of the model's axes", id.name())));
            }
        }
        Ok(())
    };
    for id in [QuantityId::AddedMass, QuantityId::RadiationDamping] {
        shape(id, &[w.F, w.D, w.K])?;
    }
    for id in [
        QuantityId::ExcitationDiffraction,
        QuantityId::ExcitationFroudeKrylov,
        QuantityId::ExcitationScattering,
        QuantityId::Motion,
    ] {
        shape(id, &[w.F, w.H, w.D])?;
    }
    shape(QuantityId::ExcitationHaskind, &[w.F, w.H, w.K])?;
    for id in [QuantityId::AddedMassZeroFrequency, QuantityId::AddedMassInfiniteFrequency] {
        shape(id, &[w.D, w.K])?;
    }
    shape(QuantityId::FieldPointsPressureRadiation, &[w.F, w.J, w.K])?;
    shape(QuantityId::TanksPressureRadiation, &[w.F, w.M, w.K])?;
    for id in [
        QuantityId::FieldPointsPressureIncident,
        QuantityId::FieldPointsPressureScattering,
        QuantityId::FieldPointsPressureDiffraction,
    ] {
        shape(id, &[w.F, w.J, w.H])?;
    }
    let rao = single(w.m, "first_order").get("motion").boolean();
    let motion = r.quantities.contains_key(&QuantityId::Motion);
    if motion && !rao {
        return Err(Error::value("motion is present but first_order.motion is false (.4 is written only with motion)"));
    }
    if rao && !motion {
        return Err(Error::value("first_order.motion is true but the motion is not available"));
    }
    if r.quantities.contains_key(&QuantityId::AddedMass) != r.quantities.contains_key(&QuantityId::RadiationDamping) {
        return Err(Error::value("added mass and radiation damping must be available together"));
    }
    if r.quantities.contains_key(&QuantityId::ExcitationHaskind)
        && !r.quantities.contains_key(&QuantityId::ExcitationFroudeKrylov)
    {
        return Err(Error::value("the Haskind force needs the Froude-Krylov force for .2fk and .2sc"));
    }
    let all = model_omegas(w.m);
    for (id, limit) in
        [(QuantityId::AddedMassZeroFrequency, 0.0), (QuantityId::AddedMassInfiniteFrequency, f64::INFINITY)]
    {
        let requested = all.contains(&limit);
        if requested != r.quantities.contains_key(&id) {
            return Err(Error::value(format!(
                "the endpoint added mass '{}' does not match the requested frequencies",
                if limit == 0.0 { "zero" } else { "infinite" }
            )));
        }
    }
    if let Some(hv) = r.quantities.get(&QuantityId::HydrostaticsVolumes) {
        if hv.dims[0] != w.m.root().children("bodies").len() as i64 {
            return Err(Error::value("hydrostatics must be given for every body"));
        }
    }
    Ok(())
}

impl<'a> Writer<'a> {
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn build(&self) -> Result<Files> {
        preflight(self)?;
        let (F, H, D, K, P) = (self.F, self.H, self.D, self.K, self.P);
        let (per, bcol, ids, m, r) = (&self.per, &self.bcol, &self.ids, self.m, self.r);
        let mut files = Files::new();
        let A = self.norm(QuantityId::AddedMass)?;
        let B = self.norm(QuantityId::RadiationDamping)?;
        let A0 = self.norm(QuantityId::AddedMassZeroFrequency)?;
        let Ai = self.norm(QuantityId::AddedMassInfiniteFrequency)?;
        // no radiation columns (K = 0): no .1, .2, .2fk, .2sc, as no radiation block of .5*, .6*
        // (reference/file-formats §2)
        if (A.ok || A0.ok || Ai.ok) && K > 0 {
            let mut lines = Vec::new();
            let mut f = 0i64; // the ordinary frequencies in the order of the model
            for w in model_omegas(m) {
                if w == 0.0 || w.is_infinite() {
                    let a = if w == 0.0 { &A0 } else { &Ai };
                    if !a.ok {
                        continue;
                    }
                    let code = e0p(if w == 0.0 { -1.0 } else { 0.0 })?;
                    for i in 0..D {
                        for k in 0..K {
                            // an entry without a solution (a tank) is not listed
                            if a.re(&[i, k]).is_finite() {
                                lines.push(
                                    code.clone()
                                        + &fortran_i(i + 1, 6)?
                                        + &fortran_i(ids[k as usize] as i64, 6)?
                                        + &e1p(a.re(&[i, k]))?,
                                );
                            }
                        }
                    }
                } else if w > 0.0 {
                    if A.ok {
                        for i in 0..D {
                            for k in 0..K {
                                lines.push(
                                    per[f as usize].clone()
                                        + &fortran_i(i + 1, 6)?
                                        + &fortran_i(ids[k as usize] as i64, 6)?
                                        + &e1p(A.re(&[f, i, k]))?
                                        + &e1p(B.re(&[f, i, k]))?,
                                );
                            }
                        }
                    }
                    f += 1;
                }
            }
            files.insert(".1".into(), lines);
        }
        let hk = self.norm(QuantityId::ExcitationHaskind)?;
        let fk = self.norm(QuantityId::ExcitationFroudeKrylov)?;
        if hk.ok && K > 0 {
            files.insert(".2".into(), self.force_file(&hk, ids, true)?);
            let mut fkc = hk.clone();
            let mut sc = hk.clone();
            for f in 0..F {
                for h in 0..H {
                    for n in 0..K {
                        let j = ids[n as usize] as i64 - 1;
                        let a = fkc.at(&[f, h, n]);
                        fkc.v[a] = fk.c(&[f, h, j]);
                        let a = sc.at(&[f, h, n]);
                        sc.v[a] = hk.c(&[f, h, n]) - fk.c(&[f, h, j]);
                    }
                }
            }
            files.insert(".2fk".into(), self.force_file(&fkc, ids, true)?);
            files.insert(".2sc".into(), self.force_file(&sc, ids, true)?);
        }
        let all_rows: Vec<i32> = (1..=D as i32).collect();
        for (ext, id) in [
            (".3", QuantityId::ExcitationDiffraction),
            (".3fk", QuantityId::ExcitationFroudeKrylov),
            (".3sc", QuantityId::ExcitationScattering),
            (".4", QuantityId::Motion),
        ] {
            let a = self.norm(id)?;
            if a.ok {
                files.insert(ext.into(), self.force_file(&a, &all_rows, false)?);
            }
        }
        // the drift heading pairs: (h, h) of every heading (unidirectional) or every pair
        // h_a <= h_b in heading order (bidirectional), told apart by their number
        let pairs_of = |q: &Q| crate::heading_pairs::heading_pairs(H as usize, q.dims[1] != H);
        let dmo = self.norm(QuantityId::DriftMomentumConservation)?;
        if dmo.ok {
            // mean drift by momentum: PER, beta_1, beta_2, I (1, 2, 6) of the heading pairs
            let mut lines = Vec::new();
            let rows = [1i64, 2, 6];
            let hp = pairs_of(&dmo);
            for f in 0..F {
                for (p, &(h, h2)) in hp.iter().enumerate() {
                    for n in 0..3 {
                        lines.push(
                            per[f as usize].clone()
                                + &bcol[h]
                                + &bcol[h2]
                                + &fortran_i(rows[n as usize], 6)?
                                + &complex_cols(dmo.c(&[f, p as i64, n]))?,
                        );
                    }
                }
            }
            files.insert(".8".into(), lines);
        }
        for (id, moving, ext) in [
            (QuantityId::DriftPressureIntegration, QuantityId::DriftPressureIntegrationBodyAxes, ".9"),
            (QuantityId::DriftControlSurface, QuantityId::DriftControlSurfaceBodyAxes, ".7"),
        ] {
            let dp = self.norm(id)?;
            let dm = self.norm(moving)?;
            if !dp.ok {
                continue;
            }
            // per body rows 6b + 1..6 about its rotation centre (its rigid coordinates), -(6b + 4..6) the
            // moment about the moving rotation centre (equal headings only)
            let mut lines = Vec::new();
            let nb = self.coords.iter().map(|c| c.body + 1).max().unwrap_or(0) as i64;
            let first: Vec<i64> =
                (0..nb).map(|b| self.coords.iter().position(|c| c.body as i64 == b).unwrap_or(0) as i64).collect();
            let hp = pairs_of(&dp);
            for f in 0..F {
                for (p, &(h, h2)) in hp.iter().enumerate() {
                    let p = p as i64;
                    for b in 0..nb {
                        let head = per[f as usize].clone() + &bcol[h] + &bcol[h2];
                        for i in 0..6 {
                            lines.push(
                                head.clone()
                                    + &fortran_i(6 * b + i + 1, 6)?
                                    + &complex_cols(dp.c(&[f, p, first[b as usize] + i]))?,
                            );
                        }
                        if h != h2 {
                            continue;
                        }
                        for i in 0..3 {
                            lines.push(
                                head.clone()
                                    + &fortran_i(-(6 * b + i + 4), 6)?
                                    + &complex_cols(dm.c(&[f, p, 3 * b + i]))?,
                            );
                        }
                    }
                }
            }
            files.insert(ext.into(), lines);
        }
        if r.quantities.contains_key(&QuantityId::BodyPointsIds) {
            self.body_files(&mut files)?;
        }
        let frad = self.field(QuantityId::FieldPointsPressureRadiation, Some(QuantityId::TanksPressureRadiation))?;
        let fdif = self.field(QuantityId::FieldPointsPressureDiffraction, None)?;
        let field_file = |rad: &Q, dif: &Q, e: i64| -> Result<Vec<String>> {
            let mut lines = Vec::new();
            let val = |q: &Q, f: i64, p: i64, k: i64| if e < 0 { q.c(&[f, p, k]) } else { q.c(&[f, p, k, e]) };
            for f in 0..F {
                if rad.ok && K > 0 {
                    // no radiation columns: no radiation block (reference/file-formats §2)
                    for p in 0..P {
                        let mut row = per[f as usize].clone() + &fortran_i(p + 1, 10)?;
                        for k in 0..K {
                            let z = val(rad, f, p, k);
                            row += &e1p(z.re)?;
                            row += &e1p(z.im)?;
                        }
                        lines.push(row);
                    }
                }
                if dif.ok {
                    for h in 0..H {
                        for p in 0..P {
                            let z = val(dif, f, p, h);
                            lines.push(
                                per[f as usize].clone()
                                    + &bcol[h as usize]
                                    + &fortran_i(p + 1, 10)?
                                    + &e1p(z.re)?
                                    + &e1p(z.im)?,
                            );
                        }
                    }
                }
            }
            Ok(lines)
        };
        if self.has_fp && (frad.ok || fdif.ok) {
            files.insert(".6p".into(), field_file(&frad, &fdif, -1)?);
        }
        if self.has_fp {
            let fp = &self.fp;
            let mut lines = Vec::new();
            for p in 0..P as usize {
                lines.push(
                    fortran_i(p as i64 + 1, 10)? + &e1p(fp[3 * p])? + &e1p(fp[3 * p + 1])? + &e1p(fp[3 * p + 2])?,
                );
            }
            files.insert(".fpt".into(), lines);
        }
        let fvr = self.field(QuantityId::FieldPointsVelocityRadiation, Some(QuantityId::TanksVelocityRadiation))?;
        let fvd = self.field(QuantityId::FieldPointsVelocityDiffraction, None)?;
        if self.has_fp && (fvr.ok || fvd.ok) {
            // the velocity components in the layout of .6p
            let ext = [".6vx", ".6vy", ".6vz"];
            for e in 0..3 {
                files.insert(ext[e].into(), field_file(&fvr, &fvd, e as i64)?);
            }
        }
        // .hst: the restoring with weight of every body and of the pressure and generalized coordinates,
        // when every body has its centre of gravity and its mass
        let stiff = fetch(r, QuantityId::HydrostaticsStiffness)?;
        let mut weight = stiff.ok;
        for b in m.root().children("bodies") {
            weight = weight && b.is_set("center_of_gravity") && b.is_set("mass");
        }
        if weight {
            let Du = D as usize;
            let mut C = vec![0.0f64; Du * Du];
            let mut off = 0usize;
            let bodies = m.root().children("bodies");
            for (b, body) in bodies.iter().enumerate() {
                for i in 0..6usize {
                    for j in 0..6usize {
                        C[(off + i) * Du + off + j] = stiff.re(&[b as i64, i as i64, j as i64]);
                    }
                }
                off += 6 + body.children("generalized_modes").len() + pressure_mode_names(*body).len();
            }
            let mr = fetch(r, QuantityId::ModeRestoring)?;
            if mr.ok {
                for k in 0..Du * Du {
                    C[k] += mr.v[k].re;
                }
            }
            let (rho, g, rot) = (self.rho, self.g, &self.rot);
            let mut lines = Vec::new();
            for i in 0..Du {
                for j in 0..Du {
                    lines.push(
                        fortran_i(i as i64 + 1, 6)?
                            + &fortran_i(j as i64 + 1, 6)?
                            + &fortran_e(C[i * Du + j] / (rho * g * self.Lp(2 + rot[i] + rot[j])), true, 15, 6)?,
                    );
                }
            }
            files.insert(".hst".into(), lines);
        }
        Ok(files)
    }

    // .5p, .5vx, .5vy, .5vz (radiation and diffraction separately) and .pnl (the panel
    // data, normals into the body, the format's convention)
    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn body_files(&self, files: &mut Files) -> Result<()> {
        let (F, H, K, m, r) = (self.F, self.H, self.K, self.m, self.r);
        let (per, bcol) = (&self.per, &self.bcol);
        let qid = fetch(r, QuantityId::BodyPointsIds)?;
        let n = qid.dims[0];
        let id = |q: i64, c: i64| qid.re(&[q, c]) as i32;
        let mut specified = true;
        for q in 0..n {
            specified = specified && id(q, 2) == 0 && id(q, 3) == 0;
        }
        let mut order: Vec<i64> = (0..n).collect();
        let mut labels: Vec<(i32, i32)> = Vec::new();
        if specified && n > 0 {
            // specified points (the item body_points): body and point numbers in input order
            for q in 0..n {
                labels.push((id(q, 0) + 1, id(q, 1)));
            }
        } else {
            // the format's order of the reflected panels: one body in open sea identity, -x, -x -y, -y (the
            // quadrants in turn); several bodies or walls identity, -x, -y, -x -y
            let domain = single(m, "environment").get("domain").text().to_string();
            let one = m.root().children("bodies").len() == 1
                && (domain != "x_wall" && domain != "y_wall" && domain != "orthogonal_walls");
            let ro = |sx: i32, sy: i32| -> i32 {
                if one {
                    if sx == 1 && sy == 1 {
                        0
                    } else if sx == -1 && sy == 1 {
                        1
                    } else if sx == -1 && sy == -1 {
                        2
                    } else {
                        3
                    }
                } else if sx == 1 && sy == 1 {
                    0
                } else if sx == -1 && sy == 1 {
                    1
                } else if sx == 1 && sy == -1 {
                    2
                } else {
                    3
                }
            };
            let key = |q: i64| (id(q, 0), ro(id(q, 2), id(q, 3)), id(q, 1));
            order.sort_by_key(|&q| key(q)); // stable
            if one {
                let present: BTreeSet<i32> = (0..n).map(|q| key(q).1).collect();
                let pv: Vec<i32> = present.into_iter().collect();
                for &q in &order {
                    let k = key(q);
                    labels.push((pv.iter().position(|&x| x == k.1).unwrap() as i32 + 1, k.2));
                }
            } else {
                for k in 0..n {
                    labels.push((1, k as i32 + 1));
                }
            }
        }
        let lab: Vec<String> = labels
            .iter()
            .map(|&(a, b)| Ok(fortran_i(a as i64, 6)? + &fortran_i(b as i64, 10)?))
            .collect::<Result<_>>()?;
        let pr = self.norm(QuantityId::BodyPointsPressureRadiation)?;
        let pd = self.norm(QuantityId::BodyPointsPressureDiffraction)?;
        let vr = self.norm(QuantityId::BodyPointsVelocityRadiation)?;
        let vd = self.norm(QuantityId::BodyPointsVelocityDiffraction)?;
        let file = |rad: &Q, dif: &Q, e: i64| -> Result<Vec<String>> {
            let mut lines = Vec::new();
            let val = |q: &Q, f: i64, p: i64, k: i64| if e < 0 { q.c(&[f, p, k]) } else { q.c(&[f, p, k, e]) };
            for f in 0..F {
                if rad.ok && K > 0 {
                    // no radiation columns: no radiation block (reference/file-formats §2)
                    for k2 in 0..n as usize {
                        let mut row = per[f as usize].clone() + &lab[k2];
                        for k in 0..K {
                            let z = val(rad, f, order[k2], k);
                            row += &e1p(z.re)?;
                            row += &e1p(z.im)?;
                        }
                        lines.push(row);
                    }
                }
                if dif.ok {
                    for h in 0..H {
                        for k2 in 0..n as usize {
                            let z = val(dif, f, order[k2], h);
                            lines.push(
                                per[f as usize].clone() + &bcol[h as usize] + &lab[k2] + &e1p(z.re)? + &e1p(z.im)?,
                            );
                        }
                    }
                }
            }
            Ok(lines)
        };
        if pr.ok || pd.ok {
            files.insert(".5p".into(), file(&pr, &pd, -1)?);
        }
        if vr.ok || vd.ok {
            let ext = [".5vx", ".5vy", ".5vz"];
            for e in 0..3 {
                files.insert(ext[e].into(), file(&vr, &vd, e as i64)?);
            }
        }
        if specified {
            return Ok(()); // specified points: no .pnl
        }
        // the centroid in global axes, the normal and r x n in the axes of its body about the body origin
        let pts = fetch(r, QuantityId::BodyPoints)?;
        let nrm = fetch(r, QuantityId::BodyPointsNormals)?;
        let area = fetch(r, QuantityId::BodyPointsAreas)?;
        let mut lines = Vec::new();
        for k2 in 0..n as usize {
            let q = order[k2];
            let body = body_of(m, id(q, 0) as usize);
            let yaw = body.get("yaw").real() * PI / 180.0;
            let (c, s) = (yaw.cos(), yaw.sin());
            let pos = body.get("position").reals().to_vec();
            let x = [pts.re(&[q, 0]), pts.re(&[q, 1]), pts.re(&[q, 2])];
            let nn = [-nrm.re(&[q, 0]), -nrm.re(&[q, 1]), -nrm.re(&[q, 2])];
            let nl = [c * nn[0] + s * nn[1], -s * nn[0] + c * nn[1], nn[2]];
            let d = [x[0] - pos[0], x[1] - pos[1], x[2] - pos[2]];
            let xl = [c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]];
            let rn = [xl[1] * nl[2] - xl[2] * nl[1], xl[2] * nl[0] - xl[0] * nl[2], xl[0] * nl[1] - xl[1] * nl[0]];
            let mut row = lab[k2].clone();
            for v in [x[0], x[1], x[2], area.re(&[q]), nl[0], nl[1], nl[2], rn[0], rn[1], rn[2]] {
                row += &e1p(v)?;
            }
            lines.push(row);
        }
        files.insert(".pnl".into(), lines);
        Ok(())
    }

    // ---------------------------------------------------------------- the .out summary

    #[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn out_file(&self, files: &Files, stamp: &str) -> Result<Vec<String>> {
        let (F, H, D, K, P, m, r) = (self.F, self.H, self.D, self.K, self.P, self.m, self.r);
        let (om, beta, ids, coords) = (&self.om, &self.beta, &self.ids, &self.coords);
        let (g, L, rho) = (self.g, self.L, self.rho);
        let rule = format!(" {}", "-".repeat(72));
        let stars = format!(" {}", "*".repeat(72));
        let opt = single(m, "options");
        let env = single(m, "environment");
        let hi = opt.get("higher_order").boolean();
        let lids = m.root().children("lids");
        let formulation = solved_formulation(m)?;
        let depth = env.get("depth").real();
        let version = &build_info().version;
        let mut lines: Vec<String> = vec![
            rule.clone(),
            format!("                      HydroBEM {version}  summary"),
            rule.clone(),
            format!(" Produced by HydroBEM on {stamp}"),
            String::new(),
            if hi { " Higher-order method".into() } else { " Low-order panel method".into() },
            String::new(),
            format!(" Gravity: {}                Length scale: {}", fixed(g, 12, 5), fixed(L, 12, 5)),
            format!(
                " Water depth: {}        Water density: {}",
                if depth.is_infinite() { "    infinite".to_string() } else { fixed(depth, 12, 5) },
                fixed(rho, 12, 5)
            ),
            format!(" Logarithmic singularity integration: {}", boolean(opt.get("logarithmic_integration").boolean())),
            format!(" Diffraction formulation: {}", formulation),
        ];
        {
            let mut s = format!(" Boundary: {}", env.get("domain").text());
            let cw = env.get("channel_width");
            if cw.is_set() && cw.real() != 0.0 {
                s += &format!("  (channel width {})", fixed(cw.real(), 12, 5));
            }
            lines.push(s);
        }
        lines.push(format!(" Physical damping lids: {}", lids.len()));
        for lid in &lids {
            // one line per lid (the writer ends every line)
            lines.push(format!(
                "   lid '{}': damping factor {}",
                lid.name(),
                fixed(lid.get("damping_factor").real(), 12, 5)
            ));
        }
        lines.push(String::new());
        let vol = fetch(r, QuantityId::HydrostaticsVolumes)?;
        let cbq = fetch(r, QuantityId::HydrostaticsCenterOfBuoyancy)?;
        let wpa = fetch(r, QuantityId::HydrostaticsWaterplaneArea)?;
        let wpc = fetch(r, QuantityId::HydrostaticsWaterplaneCentroid)?;
        let stq = fetch(r, QuantityId::HydrostaticsStiffness)?;
        let massq = fetch(r, QuantityId::HydrostaticsMass)?;
        let cgq = fetch(r, QuantityId::HydrostaticsCenterOfGravity)?;
        let bodies = m.root().children("bodies");
        for (b, body) in bodies.iter().enumerate() {
            let mesh_name = body_mesh_name(*body).unwrap_or_default();
            let head = mesh_head(m, &mesh_name)?;
            let mesh = m.root().child("meshes", &mesh_name).expect("the mesh of a body");
            let label = head.get("label").text().to_string();
            lines.push(format!(
                " Body number: N={:4}   Name: {}   Geometry: {}",
                b + 1,
                body.name(),
                if label.is_empty() { "(vertex arrays)" } else { &label }
            ));
            let rec = format!("{:6}", records_of(mesh));
            if hi {
                lines.push(format!(" Total patches: {rec}   Geometry kind: {}", mesh.mesh_kind().name()));
            } else {
                lines.push(format!(" Total panels: {rec}"));
            }
            let sym = symmetry_flags(&mesh.get("symmetry"));
            lines.push(format!(" Symmetries: ISX={}  ISY={}", sym[0], sym[1]));
            // the elements of the dipoles on the body's mesh (with those inside its tanks), and whether a
            // dipole is a mesh of its own
            let (mut n_sheets, mut external) = (0i64, false);
            let mut dipoles = body.children("dipoles");
            for t in body.children("tanks") {
                dipoles.extend(t.children("dipoles"));
            }
            for d in dipoles {
                let r = d.get("ranges");
                if d.get("mesh").text() != mesh_name {
                    external = true;
                } else if !r.is_set() {
                    n_sheets += records_of(mesh);
                } else {
                    n_sheets += r.ints().chunks_exact(2).map(|c| c[1] - c[0] + 1).sum::<i64>();
                }
            }
            let sheets = format!("{n_sheets:6}");
            lines.push(format!(
                " Thin-sheet (dipole) panels: {sheets}{}",
                if external { "   plus external dipole mesh" } else { "" }
            ));
            let pos = body.get("position").reals().to_vec();
            let rc = body.get("rotation_center").reals().to_vec();
            lines.push(format!(
                " XBODY ={} YBODY ={} ZBODY ={} PHIBODY ={}",
                fixed(pos[0], 12, 5),
                fixed(pos[1], 12, 5),
                fixed(pos[2], 12, 5),
                fixed(body.get("yaw").real(), 8, 2)
            ));
            lines.push(format!(
                " Motion reference (rotation_center, body axes): {}{}{}",
                fixed(rc[0], 12, 5),
                fixed(rc[1], 12, 5),
                fixed(rc[2], 12, 5)
            ));
            let interior = match body.get("interior_surface").text() {
                "mesh" if body.get("interior_surface_mesh").text() == mesh_name => "panels".to_string(),
                m => m.to_string(),
            };
            lines.push(format!(" Irregular-frequency interior surface: {interior}"));
            lines
                .push(format!(" Exciting force used in the motion equation: {}", body.get("excitation_method").text()));
            let mut s = 0;
            for ps in body.children("pressure_surfaces") {
                let names: Vec<&str> = ps.children("pressure_modes").iter().map(|mode| mode.name()).collect();
                s += 1;
                lines.push(format!(
                    " Pressure surface {s}: modes {}; chamber coupling {}; convention {}",
                    names.join(", "),
                    boolean(ps.get("chamber_coupling").boolean()),
                    ps.get("response_convention").text()
                ));
            }
            // the free-surface plates: their elements, on the body's mesh or a mesh of their own
            for (k, fp) in body.children("free_surface_plates").iter().enumerate() {
                let pm_name = fp.get("mesh").text().to_string();
                let pm = m.root().child("meshes", &pm_name).expect("the mesh of a plate");
                let r = fp.get("ranges");
                let n = if r.is_set() {
                    r.ints().chunks_exact(2).map(|c| c[1] - c[0] + 1).sum::<i64>()
                } else {
                    records_of(pm)
                };
                let name = if fp.name().is_empty() { String::new() } else { format!(" '{}'", fp.name()) };
                lines.push(format!(
                    " Free-surface plate {}{name}: {n} {} of mesh '{pm_name}'",
                    k + 1,
                    if hi { "patches" } else { "panels" }
                ));
            }
            let h = vol.ok;
            let bi = b as i64;
            let mut mass_in_use = 0.0;
            if h {
                lines.push(format!(
                    " Volumes (VOLX,VOLY,VOLZ):     {}  {}  {}",
                    fixed(vol.re(&[bi, 0]), 14, 6),
                    fixed(vol.re(&[bi, 1]), 14, 6),
                    fixed(vol.re(&[bi, 2]), 14, 6)
                ));
                lines.push(format!(
                    " Center of Buoyancy (Xb,Yb,Zb): {} {} {}",
                    fixed(cbq.re(&[bi, 0]), 12, 6),
                    fixed(cbq.re(&[bi, 1]), 12, 6),
                    fixed(cbq.re(&[bi, 2]), 12, 6)
                ));
                lines.push(format!(
                    " Water-plane area and centroid: {} {} {}",
                    fixed(wpa.re(&[bi]), 14, 6),
                    fixed(wpc.re(&[bi, 0]), 12, 6),
                    fixed(wpc.re(&[bi, 1]), 12, 6)
                ));
                lines.push(" Hydrostatic and gravitational restoring coefficients (dimensional):".into());
                let c = |i: i64, j: i64| e1p(stq.re(&[bi, i, j]));
                lines.push(format!(" C(3,3),C(3,4),C(3,5): {} {} {}", c(2, 2)?, c(2, 3)?, c(2, 4)?));
                lines.push(format!(" C(4,4),C(4,5),C(4,6): {} {} {}", c(3, 3)?, c(3, 4)?, c(3, 5)?));
                lines.push(format!("        C(5,5),C(5,6): {} {}", c(4, 4)?, c(4, 5)?));
                if !cgq.re(&[bi, 0]).is_nan() {
                    lines.push(format!(
                        " Center of Gravity  (Xg,Yg,Zg): {} {} {}",
                        fixed(cgq.re(&[bi, 0]), 12, 6),
                        fixed(cgq.re(&[bi, 1]), 12, 6),
                        fixed(cgq.re(&[bi, 2]), 12, 6)
                    ));
                }
                mass_in_use = massq.re(&[bi]);
                if !mass_in_use.is_nan() {
                    lines.push(format!(" Mass: {}", e1p(mass_in_use)?)); // none given: no line
                }
            }
            // the inertia of inertia_input (the other kept unused)
            let components = body.get("inertia_input").text() == "components";
            let given = components && body.is_set("Ixx");
            let radii = !components && body.is_set("radii_of_gyration");
            if (given || radii) && (given || h) {
                // as given, or m r_ij |r_ij| from the radii of gyration (three values: the diagonal) with the mass in use
                let mut I = [0.0f64; 6];
                let names = ["Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"];
                if given {
                    for k in 0..6 {
                        I[k] = body.get(names[k]).real();
                    }
                } else {
                    let rg = body.get("radii_of_gyration");
                    let mut R = [[0.0f64; 3]; 3];
                    for i in 0..3 {
                        for j in 0..3 {
                            R[i][j] = if rg.reals().len() == 3 {
                                if i == j { rg.reals()[i] } else { 0.0 }
                            } else {
                                rg.reals()[3 * i + j]
                            };
                        }
                    }
                    let ij = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]];
                    for k in 0..6 {
                        let x = R[ij[k][0]][ij[k][1]];
                        I[k] = mass_in_use * x * x.abs();
                    }
                }
                let io = body.get("inertia_origin");
                let where_ = if io.is_array() {
                    format!("({}, {}, {})", py_repr(io.reals()[0]), py_repr(io.reals()[1]), py_repr(io.reals()[2]))
                } else {
                    io.text().to_string()
                };
                lines.push(format!(" Inertia (Ixx,Ixy,Ixz,Iyy,Iyz,Izz) about {where_}:"));
                let mut row = " ".to_string();
                for v in I {
                    row += &e1p(v)?;
                }
                lines.push(row);
            }
            lines.push(String::new());
        }
        {
            // the files in the sorted order of their names (the keys of a BTreeMap)
            let mut s = " Numeric output files:".to_string();
            for e in files.keys() {
                if e != ".out" {
                    s += " ";
                    s += e;
                }
            }
            lines.push(s);
        }
        lines.push(format!(
            " First column of the numeric files: {}",
            if self.omega { "frequency (rad/s)" } else { "wave period (s)" }
        ));
        {
            let s: Vec<String> = ids.iter().map(|x| x.to_string()).collect();
            lines.push(format!(" Radiation mode order (global coordinate ids): {}", s.join(" ")));
            let c: Vec<String> = coords
                .iter()
                .enumerate()
                .map(|(k, co)| format!("{}={}:{}", k + 1, body_of(m, co.body as usize).name(), co.name))
                .collect();
            lines.push(format!(" Coordinates: {}", c.join(", ")));
        }
        for (id, title) in [
            (QuantityId::ModeRestoring, "Restoring matrix of the generalized and pressure coordinates"),
            (QuantityId::PressureModeDamping, "Damping matrix of the pressure modes"),
        ] {
            let q = fetch(r, id)?;
            if !q.ok {
                continue;
            }
            lines.push(format!(" {title}:"));
            for i in 0..q.dims[0] {
                let mut row = " ".to_string();
                for j in 0..q.dims[1] {
                    row += &e1p(q.re(&[i, j]))?;
                }
                lines.push(row);
            }
        }
        lines.push(String::new());
        lines.push(rule.clone());
        lines.push(" Output from HydroBEM".into());
        lines.push(rule.clone());
        let table_header = |lines: &mut Vec<String>, title: &str| {
            lines.push(String::new());
            lines.push(format!("    {title}"));
            lines.push(String::new());
        };
        let A = self.norm(QuantityId::AddedMass)?;
        let B = self.norm(QuantityId::RadiationDamping)?;
        let hk = self.norm(QuantityId::ExcitationHaskind)?;
        let X = self.norm(QuantityId::ExcitationDiffraction)?;
        let mo = self.norm(QuantityId::Motion)?;
        let fd = self.field(QuantityId::FieldPointsPressureDiffraction, None)?;
        let fr = self.field(QuantityId::FieldPointsPressureRadiation, Some(QuantityId::TanksPressureRadiation))?;
        let lr = fetch(r, QuantityId::LidsElevationRadiation)?;
        let ld = fetch(r, QuantityId::LidsElevationDiffraction)?;
        let pr = fetch(r, QuantityId::PressureSurfacesElevationRadiation)?;
        let pdq = fetch(r, QuantityId::PressureSurfacesElevationDiffraction)?;
        let all: Vec<i32> = (1..=D as i32).collect();
        let fmt_i6 = |v: i64| format!("{v:6}");
        let fmt_i10 = |v: i64| format!("{v:10}");
        for f in 0..F {
            lines.extend([
                String::new(),
                stars.clone(),
                String::new(),
                format!(" Wave period (sec) = {}", e1p(2.0 * PI / om[f as usize])?),
                stars.clone(),
            ]);
            if A.ok && K > 0 {
                // no radiation columns: no table
                table_header(&mut lines, "ADDED-MASS AND DAMPING COEFFICIENTS");
                lines.push("     I     J         A(I,J)         B(I,J)".into());
                lines.push(String::new());
                for i in 0..D {
                    for k in 0..K {
                        lines.push(
                            fmt_i6(i + 1)
                                + &fmt_i6(ids[k as usize] as i64)
                                + " "
                                + &e1p(A.re(&[f, i, k]))?
                                + " "
                                + &e1p(B.re(&[f, i, k]))?,
                        );
                    }
                }
            }
            for (q, title, rows, by_column) in [
                (&hk, "HASKIND EXCITING FORCES AND MOMENTS", ids, true),
                (&X, "DIFFRACTION EXCITING FORCES AND MOMENTS", &all, false),
                (&mo, "RESPONSE AMPLITUDE OPERATORS", &all, false),
            ] {
                if !q.ok || rows.is_empty() {
                    continue;
                }
                table_header(&mut lines, title);
                for h in 0..H {
                    lines.push(format!("  Wave Heading (deg) : {}", fixed(beta[h as usize], 14, 6)));
                    lines.push(String::new());
                    lines.push("     I     Mod[X(I)]     Pha[X(I)]".into());
                    lines.push(String::new());
                    for (n, &i) in rows.iter().enumerate() {
                        let z = if by_column { q.c(&[f, h, n as i64]) } else { q.c(&[f, h, i as i64 - 1]) };
                        lines
                            .push(fmt_i6(i as i64) + "   " + &e1p(z.re.hypot(z.im))? + " " + &fixed(degrees(z), 13, 6));
                    }
                    lines.push(String::new());
                }
            }
            if fd.ok || (fr.ok && K > 0) {
                table_header(&mut lines, "FREE-SURFACE ELEVATION AT FIELD POINTS");
                if fr.ok && K > 0 {
                    lines.push("   Radiation (per unit displacement), columns by radiation mode".into());
                    for p in 0..P {
                        let mut row = fmt_i10(p + 1);
                        for k in 0..K {
                            let z = fr.c(&[f, p, k]);
                            row += &e1p(z.re.hypot(z.im))?;
                        }
                        lines.push(row);
                    }
                }
                if fd.ok {
                    for h in 0..H {
                        lines.push(format!(
                            "  Wave Heading (deg) : {}   (incident + scattered)",
                            fixed(beta[h as usize], 14, 6)
                        ));
                        for p in 0..P {
                            let z = fd.c(&[f, p, h]);
                            lines.push(
                                fmt_i10(p + 1) + "   " + &e1p(z.re.hypot(z.im))? + " " + &fixed(degrees(z), 13, 6),
                            );
                        }
                    }
                }
            }
            for (rad, dif, title) in [(&lr, &ld, "DAMPING-LID ELEVATION"), (&pr, &pdq, "PRESSURE-SURFACE ELEVATION")] {
                let with_rad = rad.ok && rad.dims[2] > 0; // no radiation columns: no radiation rows
                if !with_rad && !dif.ok {
                    continue;
                }
                table_header(&mut lines, title);
                for (q, name, is_rad) in [(rad, "radiation", true), (dif, "diffraction", false)] {
                    if !q.ok || (is_rad && !with_rad) {
                        continue;
                    }
                    lines.push(format!("   {name}"));
                    for p in 0..q.dims[1] {
                        let mut row = fmt_i10(p + 1);
                        for k in 0..q.dims[2] {
                            let z = q.c(&[f, p, k]);
                            row += &e1p(z.re.hypot(z.im))?;
                        }
                        lines.push(row);
                    }
                }
            }
        }
        for (id, title) in [
            (QuantityId::AddedMassZeroFrequency, "ZERO-FREQUENCY ADDED MASS"),
            (QuantityId::AddedMassInfiniteFrequency, "INFINITE-FREQUENCY ADDED MASS"),
        ] {
            let a = self.norm(id)?;
            if !a.ok || K == 0 {
                continue;
            }
            lines.extend([
                String::new(),
                stars.clone(),
                String::new(),
                format!("    {title}"),
                String::new(),
                "     I     J         A(I,J)".to_string(),
                String::new(),
            ]);
            for i in 0..D {
                for k in 0..K {
                    let v = a.re(&[i, k]);
                    lines.push(
                        fmt_i6(i + 1)
                            + &fmt_i6(ids[k as usize] as i64)
                            + " "
                            + &(if v.is_finite() { e1p(v)? } else { pad("undefined", 14) }),
                    );
                }
            }
        }
        lines.push(String::new());
        Ok(lines)
    }
}

// ---------------------------------------------------------------- the local time

#[cfg(windows)]
fn local_now() -> String {
    #[repr(C)]
    #[derive(Default)]
    // the SYSTEMTIME of the Windows API
    struct SystemTime {
        year: u16,
        month: u16,
        day_of_week: u16,
        day: u16,
        hour: u16,
        minute: u16,
        second: u16,
        milliseconds: u16,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLocalTime(t: *mut SystemTime);
    }
    let mut t = SystemTime::default();
    // SAFETY: GetLocalTime writes one SYSTEMTIME (eight WORDs, the layout of the struct above,
    // repr(C)) through a pointer to a live, writable value; it has no failure mode.
    unsafe { GetLocalTime(&mut t) };
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", t.year, t.month, t.day, t.hour, t.minute, t.second)
}

// the UTC date and time of a Unix time (s): the civil date of the day count (Hinnant, "chrono-
// compatible low-level date algorithms", days_from_civil inverted)
#[cfg(any(not(windows), test))]
fn utc_text(t: i64) -> String {
    let (days, s) = (t.div_euclid(86400), t.rem_euclid(86400));
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

#[cfg(not(windows))]
fn local_now() -> String {
    use std::os::raw::{c_char, c_int, c_long};
    #[repr(C)]
    struct Tm {
        sec: c_int,
        min: c_int,
        hour: c_int,
        mday: c_int,
        mon: c_int,
        year: c_int,
        wday: c_int,
        yday: c_int,
        isdst: c_int,
        gmtoff: c_long,
        zone: *const c_char,
    }
    unsafe extern "C" {
        fn localtime_r(t: *const i64, tm: *mut Tm) -> *mut Tm;
    }
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let mut tm = Tm {
        sec: 0,
        min: 0,
        hour: 0,
        mday: 0,
        mon: 0,
        year: 0,
        wday: 0,
        yday: 0,
        isdst: 0,
        gmtoff: 0,
        zone: std::ptr::null(),
    };
    // SAFETY: localtime_r reads one time_t and writes one struct tm through pointers to live
    // values; the binding assumes a 64-bit time_t and the glibc/BSD struct tm above (this target is
    // not tested). A null return (a time it cannot convert) leaves tm unset: the UTC time then.
    let ok = !unsafe { localtime_r(&now, &mut tm) }.is_null();
    if !ok {
        return utc_text(now);
    }
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", tm.year + 1900, tm.mon + 1, tm.mday, tm.hour, tm.min, tm.sec)
}

const EXTENSIONS: [&str; 23] = [
    ".1", ".2", ".2fk", ".2sc", ".3", ".3fk", ".3sc", ".4", ".5p", ".5vx", ".5vy", ".5vz", ".pnl", ".6p", ".6vx",
    ".6vy", ".6vz", ".fpt", ".hst", ".7", ".8", ".9", ".out",
];

// path::filename: the last element (after the root name and the last separator)
fn file_name(p: &str) -> &str {
    let b = p.as_bytes();
    let mut start = if cfg!(windows) && b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' { 2 } else { 0 };
    for (k, &c) in b.iter().enumerate() {
        if c == b'/' || (cfg!(windows) && c == b'\\') {
            start = k + 1;
        }
    }
    &p[start..]
}

/// E14.6 in 0P form (0.dddddd E+xx) or 1P form (d.dddddd E+xx) of x, right-aligned in width;
/// an exponent of three digits or a non-finite value is an ErrorKind::Value error (E14.6:
/// width 14, digits 6).
pub fn fortran_e(x: f64, one_p: bool, width: i32, digits: i32) -> Result<String> {
    if !x.is_finite() {
        return Err(Error::value(format!("non-finite value {} cannot be written", py_repr(x))));
    }
    let x = if x == 0.0 { 0.0 } else { x }; // no negative zero
    let body = if x == 0.0 {
        format!("0.{}E+00", "0".repeat(digits as usize))
    } else {
        // the correctly rounded digits (scientific, of that precision, ties to even)
        let p = if one_p { digits } else { digits - 1 } as usize;
        let s = format!("{:.p$e}", x.abs());
        let epos = s.find('e').expect("the exponent of a scientific format");
        let mant = &s[..epos];
        let e: i32 = s[epos + 1..].parse().expect("an integer exponent");
        if one_p {
            mant.to_string() + &exponent(e, x)?
        } else {
            let m: String = mant.chars().filter(|&c| c != '.').collect();
            format!("0.{m}{}", exponent(e + 1, x)?)
        }
    };
    let body = if x < 0.0 { format!("-{body}") } else { body };
    if body.len() as i32 > width {
        return Err(Error::value(format!(
            "value {} does not fit {}E{width}.{digits}",
            py_repr(x),
            if one_p { "1P" } else { "" }
        )));
    }
    Ok(pad(&body, width))
}

/// Iw of n; a wider integer is an ErrorKind::Value error.
pub fn fortran_i(n: i64, width: i32) -> Result<String> {
    let s = n.to_string();
    if s.len() as i32 > width {
        return Err(Error::value(format!("integer {s} does not fit I{width}")));
    }
    Ok(pad(&s, width))
}

/// The quantity q of r normalized as in the numeric files (reference/file-formats §1), every element as a
/// complex value (a real quantity with zero imaginary parts), with its dimensions (those of the
/// quantity); ErrorKind::Argument for a quantity without a normalized form.
pub fn normalized(r: &Results, q: QuantityId) -> Result<(Vec<i64>, Vec<Complex64>)> {
    const KINDS: [QuantityId; 28] = [
        QuantityId::AddedMass,
        QuantityId::RadiationDamping,
        QuantityId::AddedMassZeroFrequency,
        QuantityId::AddedMassInfiniteFrequency,
        QuantityId::ExcitationDiffraction,
        QuantityId::ExcitationFroudeKrylov,
        QuantityId::ExcitationScattering,
        QuantityId::ExcitationHaskind,
        QuantityId::Motion,
        QuantityId::DriftMomentumConservation,
        QuantityId::DriftPressureIntegration,
        QuantityId::DriftPressureIntegrationBodyAxes,
        QuantityId::DriftControlSurface,
        QuantityId::DriftControlSurfaceBodyAxes,
        QuantityId::FieldPointsPressureRadiation,
        QuantityId::FieldPointsPressureIncident,
        QuantityId::FieldPointsPressureScattering,
        QuantityId::FieldPointsPressureDiffraction,
        QuantityId::FieldPointsVelocityRadiation,
        QuantityId::FieldPointsVelocityIncident,
        QuantityId::FieldPointsVelocityScattering,
        QuantityId::FieldPointsVelocityDiffraction,
        QuantityId::TanksPressureRadiation,
        QuantityId::TanksVelocityRadiation,
        QuantityId::BodyPointsPressureRadiation,
        QuantityId::BodyPointsPressureDiffraction,
        QuantityId::BodyPointsVelocityRadiation,
        QuantityId::BodyPointsVelocityDiffraction,
    ];
    if !KINDS.contains(&q) {
        return Err(Error::argument(format!("{} has no normalized form", q.name())));
    }
    let Some(model) = r.solved_document() else {
        return Err(Error::state("these results hold no model"));
    };
    if !r.quantities.contains_key(&q) {
        return Err(Error::state(format!("{} is not available", q.name())));
    }
    let model = model.as_ref();
    check_quantities(r, model, false)?; // the shapes the normalization reads
    let w = Writer::new(r, model, false)?;
    let out = w.norm(q)?;
    Ok((out.dims, out.v))
}

#[derive(Clone, Debug, Default)]
pub struct WamitWrite {
    pub omega: bool,     // the first column: omega (rad/s) instead of the period (s)
    pub overwrite: bool, // replace existing files
    pub stamp: String,   // the header time stamp; empty: the local time now
}

/// Writes every computed quantity of r (the model of the results gives the layout) as the numeric
/// files stem.1, .2, ... and the summary stem.out, the folder made; all or none (temporary files
/// renamed once all are made; when a file cannot be replaced the earlier files are restored and
/// ErrorKind::Io returned). An existing file is ErrorKind::Exists unless overwrite; a file of the stem
/// whose quantity these results do not have is ErrorKind::Exists also with overwrite (it would be left
/// stale beside them). Returns the names of the files written.
pub fn write_wamit(r: &Results, stem: &str, w: &WamitWrite) -> Result<Vec<String>> {
    // the model as it was solved: its enabled objects, numbered as the quantities
    let Some(solved) = r.solved_document() else {
        return Err(Error::state("these results hold no model: no numeric files"));
    };
    let m = solved.as_ref();
    if m.root().children("bodies").is_empty() {
        return Err(Error::value("write_wamit: the model has no bodies"));
    }
    let stamp = if w.stamp.is_empty() { local_now() } else { w.stamp.clone() };
    if stamp.contains(['\r', '\n']) {
        return Err(Error::value("the time stamp must be a single line"));
    }
    check_quantities(r, m, true)?; // before the writer indexes by them
    let mut wr = Writer::new(r, m, w.omega)?;
    wr.columns()?;
    let mut files = wr.build()?;
    let out = wr.out_file(&files, &stamp)?;
    files.insert(".out".into(), out);
    let parent = parent_folder(stem);
    if !parent.is_empty() {
        let _ = fs::create_dir_all(&parent); // the folder, once the files are made
    }
    for ext in EXTENSIONS {
        if !files.contains_key(ext) && file_exists(&format!("{stem}{ext}")) {
            return Err(Error::exists(format!("{stem}{ext} exists but its quantity is not computed by these results")));
        }
    }
    if !w.overwrite {
        for ext in files.keys() {
            if file_exists(&format!("{stem}{ext}")) {
                return Err(Error::exists(format!("{stem}{ext} exists")));
            }
        }
    }
    let version = &build_info().version;
    let base = file_name(stem);
    let mut texts: Vec<(String, String)> = Vec::new(); // (target, text)
    for (ext, lines) in &files {
        let mut text = String::new();
        if ext != ".out" {
            text = format!(" HydroBEM {version}   {base}{ext}   {stamp}\r\n");
        }
        for l in lines {
            text += l;
            text += "\r\n";
        }
        texts.push((format!("{stem}{ext}"), text));
    }
    replace_files(stem, &texts)?;
    Ok(files.keys().map(|ext| format!("{base}{ext}")).collect())
}

// a temporary name beside the stem, unique in the process and in time
fn temporary_name(stem: &str) -> String {
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    let tick = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    format!("{stem}.hydrobem-{tick}-{}.tmp", SERIAL.fetch_add(1, Ordering::Relaxed) + 1)
}

// Writes the (target, text) files all or none: every text to a temporary file, the existing
// targets moved to temporary names, then the new files renamed to the targets and the old ones
// removed. A failure (a target held open, e.g. by a spreadsheet) undoes the steps made: the old
// files are back under their names and no temporary file is left (as far as the file system lets
// the undo itself proceed).
fn replace_files(stem: &str, texts: &[(String, String)]) -> Result<()> {
    let mut temps: Vec<String> = Vec::new(); // the new files, one per target
    let mut olds: Vec<(String, String)> = Vec::new(); // (old file moved away, its target)
    let mut placed: Vec<String> = Vec::new(); // the targets that hold their new file
    let mut steps = || -> Result<()> {
        for (_, text) in texts {
            let tmp = temporary_name(stem);
            temps.push(tmp.clone());
            write_file(&tmp, text.as_bytes())?;
        }
        for (target, _) in texts {
            if file_exists(target) {
                let old = temporary_name(stem);
                fs::rename(target, &old)
                    .map_err(|e| Error::io(format!("{target}: cannot be replaced: {}", io_message(&e))))?;
                olds.push((old, target.clone()));
            }
        }
        for ((target, _), tmp) in texts.iter().zip(&temps) {
            fs::rename(tmp, target)
                .map_err(|e| Error::io(format!("{target}: cannot be written: {}", io_message(&e))))?;
            placed.push(target.clone());
        }
        Ok(())
    };
    let result = steps();
    if result.is_err() {
        for target in &placed {
            let _ = fs::remove_file(target);
        }
        for (old, target) in &olds {
            let _ = fs::rename(old, target);
        }
        for tmp in &temps {
            let _ = fs::remove_file(tmp);
        }
        return result;
    }
    for (old, _) in &olds {
        let _ = fs::remove_file(old);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::ItemValue;

    #[test]
    fn fixed_fields() {
        assert_eq!(fixed(9.80665, 12, 5), "     9.80665");
        assert_eq!(fixed(-1.5, 8, 2), "   -1.50");
        assert_eq!(fixed(0.125, 8, 2), "    0.12"); // an exact tie rounds to even
        assert_eq!(fixed(1e7, 12, 5), " 1.00000E+07");
        assert_eq!(fixed(-2.5e12, 14, 6), " -2.500000E+12");
        assert_eq!(fixed(1e300, 12, 5), "1.00000E+300");
        assert_eq!(fixed(f64::NAN, 12, 5), "   undefined");
        assert_eq!(fixed(f64::INFINITY, 8, 2), "undefined");
        assert_eq!(fixed(-0.0, 8, 2), "    0.00"); // no negative zero
        assert_eq!(fixed(-0.001, 8, 2), "   -0.00"); // a negative value rounded to zero keeps its sign
    }

    // the formulation of the .out: a given one is read without a build (a model without bodies
    // does not build), auto builds and says so on failure
    #[test]
    fn a_given_formulation_needs_no_build() {
        let mut d = Document::new();
        let first = d.root().child("first_order", "").expect("the model has its first_order").id();
        for f in ["total", "scattered"] {
            d.set(first, "diffraction_formulation", ItemValue::Text(f.into())).unwrap();
            assert_eq!(solved_formulation(&d).unwrap(), f);
        }
        d.set(first, "diffraction_formulation", ItemValue::Text("auto".into())).unwrap();
        let e = solved_formulation(&d).unwrap_err();
        assert!(e.message().starts_with("write_wamit: the formulation of the solved model: "), "{}", e.message());
    }

    // the civil calendar of Unix times: the epoch, a leap day, the day before the epoch
    #[test]
    fn utc_dates() {
        assert_eq!(utc_text(0), "1970-01-01 00:00:00");
        assert_eq!(utc_text(951782400 + 3661), "2000-02-29 01:01:01");
        assert_eq!(utc_text(-1), "1969-12-31 23:59:59");
    }

    // all or none: with one target held open without sharing the others keep their old text and
    // no temporary file is left; without it every target has its new text
    #[cfg(windows)]
    #[test]
    fn replace_all_or_none() -> Result<()> {
        use std::os::windows::fs::OpenOptionsExt;
        let dir = std::env::temp_dir().join(format!("hydrobem-replace-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("a scratch folder");
        let stem = dir.join("run").to_string_lossy().into_owned();
        let (a, b) = (format!("{stem}.1"), format!("{stem}.2"));
        fs::write(&a, "old 1").expect("a file");
        fs::write(&b, "old 2").expect("a file");
        let texts = vec![(a.clone(), "new 1".to_string()), (b.clone(), "new 2".to_string())];
        {
            let _held = fs::OpenOptions::new().read(true).share_mode(0).open(&b).expect("the file held");
            let e = replace_files(&stem, &texts).unwrap_err();
            assert!(e.message().contains("cannot be replaced"), "{}", e.message());
        }
        assert_eq!(fs::read_to_string(&a).expect("a"), "old 1");
        assert_eq!(fs::read_to_string(&b).expect("b"), "old 2");
        assert_eq!(fs::read_dir(&dir).expect("the folder").count(), 2);
        replace_files(&stem, &texts)?;
        assert_eq!(fs::read_to_string(&a).expect("a"), "new 1");
        assert_eq!(fs::read_to_string(&b).expect("b"), "new 2");
        assert_eq!(fs::read_dir(&dir).expect("the folder").count(), 2);
        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn fortran_ties_and_extremes() -> Result<()> {
        // exactly representable ties round to even
        assert_eq!(fortran_e(1234567.5, true, 14, 6)?, "  1.234568E+06");
        assert_eq!(fortran_e(1234568.5, true, 14, 6)?, "  1.234568E+06");
        assert_eq!(fortran_e(0.1234565, false, 14, 6)?, "  0.123456E+00"); // below the tie in binary
        assert_eq!(fortran_e(9.9999996e-100, true, 14, 6)?, "  1.000000E-99"); // rounded into two digits
        assert_eq!(fortran_e(1e-99, false, 14, 6)?, "  0.100000E-98");
        assert_eq!(fortran_e(9.99999949e98, true, 14, 6)?, "  9.999999E+98");
        let e = fortran_e(5e-324, true, 14, 6).unwrap_err();
        assert_eq!(e.message(), "value 5e-324 needs a three-digit exponent");
        let e = fortran_e(1e99, false, 14, 6).unwrap_err();
        assert_eq!(e.message(), "value 1e+99 needs a three-digit exponent");
        let e = fortran_e(f64::NEG_INFINITY, true, 14, 6).unwrap_err();
        assert_eq!(e.message(), "non-finite value -inf cannot be written");
        let e = fortran_e(-1.0, true, 12, 6).unwrap_err();
        assert_eq!(e.message(), "value -1.0 does not fit 1PE12.6");
        assert_eq!(fortran_i(-12345, 6)?, "-12345");
        let e = fortran_i(-123456, 6).unwrap_err();
        assert_eq!(e.message(), "integer -123456 does not fit I6");
        Ok(())
    }

    #[test]
    fn names_and_shapes() {
        assert_eq!(file_name("out/run1"), "run1");
        assert_eq!(file_name("run1"), "run1");
        if cfg!(windows) {
            assert_eq!(file_name("C:\\x\\run1"), "run1");
            assert_eq!(file_name("C:run1"), "run1");
        }
        assert_eq!(shape_text(&[3]), "(3,)");
        assert_eq!(shape_text(&[2, 3]), "(2, 3)");
        assert_eq!(degrees(Complex64::new(0.0, 1.0)), 90.0);
        assert_eq!(degrees(Complex64::new(-0.0, 0.0)), 0.0);
    }
}
