//! The meshes of the model as the descriptors of the mesh builders, readers and writers.
//!
//! The descriptors own their arrays (mesh_desc.rs), so MeshDesc holds no separate storage. A
//! trimmed mesh: `ho` is a copy of the source's
//! descriptor with its own trim elements, `source` the descriptor of the source.

use crate::constants::STANDARD_GRAVITY;
use crate::document::{ItemValue, Obj, ObjMut, symmetry_flags, symmetry_value};
use crate::error::{Error, Result, int32_of};
use crate::gdf::GdfFile;
use crate::mesh_desc::*;
use crate::mesh_desc::{AnalyticKind, PatchKind};

/// The descriptor of a mesh object.
#[derive(Clone, Debug)]
pub struct MeshDesc {
    pub low_order: bool,
    pub lo: LowOrderMeshDesc,
    pub ho: PatchMeshDesc,
    pub label: String,
    pub source: Option<Box<MeshDesc>>, // a trimmed mesh: the descriptor of its source
}

impl Default for MeshDesc {
    fn default() -> Self {
        MeshDesc {
            low_order: true,
            lo: LowOrderMeshDesc::default(),
            ho: PatchMeshDesc::default(),
            label: String::new(),
            source: None,
        }
    }
}

const ANALYTIC_KINDS: [(&str, AnalyticKind); 5] = [
    ("line", AnalyticKind::Line),
    ("arc", AnalyticKind::Arc),
    ("bilinear", AnalyticKind::Bilinear),
    ("ellipses", AnalyticKind::Ellipses),
    ("segment_arc", AnalyticKind::SegmentArc),
];
const PROFILES: [(&str, Profile); 4] = [
    ("linear", Profile::Linear),
    ("smooth", Profile::Smooth),
    ("quadratic_start", Profile::QuadraticStart),
    ("quadratic_end", Profile::QuadraticEnd),
];

// the name of a choice of a table
fn name_of<T: Copy + PartialEq>(table: &[(&'static str, T)], v: T) -> &'static str {
    table.iter().find(|(_, x)| *x == v).map(|(n, _)| *n).expect("every value has a name")
}

// the value of a choice the schema checked (a choice it does not list is a bug)
fn choice<T: Copy>(table: &[(&str, T)], s: &str) -> T {
    match table.iter().find(|(n, _)| *n == s) {
        Some(&(_, v)) => v,
        None => unreachable!("the schema checks the choices of a table, not '{s}'"),
    }
}

fn real_array(shape: Vec<i64>, data: Vec<f64>) -> ItemValue {
    ItemValue::Reals { shape, data }
}

fn int_array(shape: Vec<i64>, data: Vec<i64>) -> ItemValue {
    ItemValue::Ints { shape, data }
}

fn scalar(x: f64) -> ItemValue {
    ItemValue::Real(x)
}

fn text(s: &str) -> ItemValue {
    ItemValue::Text(s.to_string())
}

fn integer(x: i64) -> ItemValue {
    ItemValue::Int(x)
}

fn discretization_value(d: &[PatchDiscretization]) -> ItemValue {
    let mut x: Vec<i64> = Vec::new();
    for q in d {
        x.extend_from_slice(&[q.nu as i64, q.nv as i64, q.ku as i64, q.kv as i64, q.iquo as i64, q.iqui as i64]);
    }
    int_array(vec![d.len() as i64, 6], x)
}

fn discretization_of(m: Obj) -> Result<Vec<PatchDiscretization>> {
    let v = m.get("discretization");
    let mut out = Vec::new();
    let at = format!("{}discretization", m.prefix());
    let mut k = 0usize;
    while k + 6 <= v.ints().len() {
        let q = PatchDiscretization {
            nu: int32_of(v.ints()[k], &at)?,
            nv: int32_of(v.ints()[k + 1], &at)?,
            ku: int32_of(v.ints()[k + 2], &at)?,
            kv: int32_of(v.ints()[k + 3], &at)?,
            iquo: int32_of(v.ints()[k + 4], &at)?,
            iqui: int32_of(v.ints()[k + 5], &at)?,
        };
        out.push(q);
        k += 6;
    }
    Ok(out)
}

/// Fills a mesh object of the model from a geometry file as read; a mesh made by the
/// engine (from_file false) has no file gravity.
pub fn fill_mesh(m: &mut ObjMut, g: &GdfFile, from_file: bool) -> Result<()> {
    let kind = MeshKind::of_geometry(g.higher_order, g.geometry);
    m.set("kind", text(kind.name()))?;
    m.set("reference_length", scalar(g.ulen))?;
    m.set("symmetry", symmetry_value(g.isx, g.isy))?;
    m.set("label", text(&g.label))?;
    if from_file {
        m.set("file_gravity", scalar(g.gravity))?;
    }
    if matches!(kind, MeshKind::Panels | MeshKind::FlatPatches) {
        m.set("vertices", real_array(vec![g.count as i64, 4, 3], g.vertices.clone()))?;
    }
    if kind.patches() {
        m.set("discretization", discretization_value(&g.discretization))?;
    }
    if kind == MeshKind::SplinePatches {
        let mut knot = 0usize;
        let mut coef = 0usize;
        for p in 0..g.count as usize {
            let mut s = m.add("patches", "SplinePatch", "", -1)?;
            let ku = g.orders[2 * p];
            let kv = g.orders[2 * p + 1];
            let mu = g.counts[2 * p];
            let mv = g.counts[2 * p + 1];
            s.set("u_order", integer(ku as i64))?;
            s.set("v_order", integer(kv as i64))?;
            let nu = (mu + ku) as usize;
            s.set("u_knots", real_array(vec![(mu + ku) as i64], g.knots[knot..knot + nu].to_vec()))?;
            knot += nu;
            let nv = (mv + kv) as usize;
            s.set("v_knots", real_array(vec![(mv + kv) as i64], g.knots[knot..knot + nv].to_vec()))?;
            knot += nv;
            let n = 3 * mu as usize * mv as usize;
            // (a geometry file holds no weights: its B-spline patches are not rational)
            s.set("coefficients", real_array(vec![mv as i64, mu as i64, 3], g.coefficients[coef..coef + n].to_vec()))?;
            coef += n;
        }
    }
    if kind == MeshKind::AnalyticPatches {
        for d in &g.analytic {
            let mut a = m.add("patches", "AnalyticPatch", "", -1)?;
            a.set("map", text(name_of(&ANALYTIC_KINDS, d.kind)))?;
            a.set("profile", text(name_of(&PROFILES, d.profile)))?;
            a.set("a", scalar(d.a))?;
            a.set("b", scalar(d.b))?;
            a.set("theta0", scalar(d.theta0))?;
            a.set("theta1", scalar(d.theta1))?;
            a.set("p", real_array(vec![4], d.p.to_vec()))?;
            if d.kind == AnalyticKind::Bilinear {
                a.set("vertices", real_array(vec![4, 3], d.vertices[..12].to_vec()))?;
            }
            if d.kind == AnalyticKind::SegmentArc {
                a.set("vertices", real_array(vec![2, 3], d.vertices[..6].to_vec()))?;
            }
        }
    }
    Ok(())
}

/// The descriptor of a mesh object (a trimmed mesh with its source); gravity is the file's gravity,
/// or the standard gravity.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn mesh_desc(m: Obj) -> Result<MeshDesc> {
    let mut d = MeshDesc::default();
    let kind = m.mesh_kind();
    if kind == MeshKind::TrimmedPatches {
        let src = m.get("source");
        let s = if src.is_set() { m.document().root().child("meshes", src.text()) } else { None };
        let Some(s) = s else {
            return Err(Error::value(format!("{}a trimmed mesh needs its source", m.prefix())));
        };
        if matches!(s.mesh_kind(), MeshKind::Panels | MeshKind::TrimmedPatches) {
            return Err(Error::value(format!("{}the source of a trimmed mesh is a patch mesh", m.prefix())));
        }
        let source = mesh_desc(s)?;
        d.low_order = false;
        d.ho = source.ho.clone();
        let rec = m.get("trim_elements");
        let rng = m.get("parameter_ranges");
        let q = discretization_of(m)?;
        let n = if rec.shape().is_empty() { 0 } else { rec.shape()[0] as usize };
        if n == 0 {
            // (without elements the trim would be its source, untrimmed)
            return Err(Error::value(format!("{}trim_elements: a trimmed mesh has at least one element", m.prefix())));
        }
        if rng.shape().is_empty() || rng.shape()[0] as usize != n || q.len() != n {
            return Err(Error::value(format!(
                "{}trim_elements, ranges and discretization must have one row per element",
                m.prefix()
            )));
        }
        let mut trim: Vec<PatchTrim> = Vec::new();
        for k in 0..n {
            let at = format!("{}trim_elements", m.prefix());
            // the source patch one-based in the item (as every element number), zero-based in the engine
            let mut t = PatchTrim {
                patch: int32_of(rec.ints()[4 * k], &at)?.saturating_sub(1),
                sx: int32_of(rec.ints()[4 * k + 1], &at)?,
                sy: int32_of(rec.ints()[4 * k + 2], &at)?,
                cut: int32_of(rec.ints()[4 * k + 3], &at)?,
                discretization: q[k],
                ..PatchTrim::default()
            };
            t.range.copy_from_slice(&rng.reals()[4 * k..4 * k + 4]);
            trim.push(t);
        }
        let tr = m.get("trim");
        d.ho.trim_count = trim.len() as i32;
        d.ho.trim = trim;
        d.ho.trim_elevation = tr.reals()[0];
        d.ho.trim_pitch = tr.reals()[1];
        d.ho.trim_roll = tr.reals()[2];
        d.label = source.label.clone();
        d.source = Some(Box::new(source));
        return Ok(d);
    }
    let grav = m.get("file_gravity");
    let gravity = if grav.is_set() { grav.real() } else { STANDARD_GRAVITY };
    let sym = symmetry_flags(&m.get("symmetry"));
    d.label = m.get("label").text().to_string();
    if kind == MeshKind::Panels {
        let vertices = m.get("vertices").reals().to_vec();
        d.lo.panel_count = (vertices.len() / 12) as i32;
        d.lo.vertices = vertices;
        d.lo.reference_length = m.get("reference_length").real();
        d.lo.gravity = gravity;
        d.lo.symmetry_x = sym[0];
        d.lo.symmetry_y = sym[1];
        return Ok(d);
    }
    d.low_order = false;
    let h = &mut d.ho;
    h.discretization = discretization_of(m)?;
    h.reference_length = m.get("reference_length").real();
    h.gravity = gravity;
    h.symmetry_x = sym[0];
    h.symmetry_y = sym[1];
    if kind == MeshKind::FlatPatches {
        h.geometry = PatchKind::Flat;
        let vertices = m.get("vertices").reals().to_vec();
        h.patch_count = (vertices.len() / 12) as i32;
        h.vertices = vertices;
    } else if kind == MeshKind::SplinePatches {
        h.geometry = PatchKind::Spline;
        let mut weights: Vec<Option<Vec<f64>>> = Vec::new();
        for p in m.children("patches") {
            let c = p.get("coefficients");
            let ku = p.get("u_knots");
            let kv = p.get("v_knots");
            if !c.is_set() || !ku.is_set() || !kv.is_set() || !p.get("u_order").is_set() || !p.get("v_order").is_set() {
                return Err(Error::value(format!(
                    "{}a SplinePatch needs u_order, v_order, u_knots, v_knots and coefficients",
                    p.prefix()
                )));
            }
            // each patch's knots of its own length (a wrong pair with the right sum would be sliced
            // across the patches)
            if c.shape().len() != 3
                || ku.reals().len() as i64 != p.get("u_order").integer() + c.shape()[1]
                || kv.reals().len() as i64 != p.get("v_order").integer() + c.shape()[0]
            {
                return Err(Error::value(format!(
                    "{}u_knots and v_knots need u_order + coefficients.shape[1] and v_order + coefficients.shape[0] values",
                    p.prefix()
                )));
            }
            h.spline_orders.push(int32_of(p.get("u_order").integer(), &format!("{}u_order", p.prefix()))?);
            h.spline_orders.push(int32_of(p.get("v_order").integer(), &format!("{}v_order", p.prefix()))?);
            h.spline_counts.push(c.shape()[1] as i32);
            h.spline_counts.push(c.shape()[0] as i32);
            h.knots.extend_from_slice(ku.reals());
            h.knots.extend_from_slice(kv.reals());
            h.coefficients.extend_from_slice(c.reals());
            let w = p.get("weights");
            if w.is_set() && (w.shape().len() != 2 || w.shape()[0] != c.shape()[0] || w.shape()[1] != c.shape()[1]) {
                return Err(Error::value(format!(
                    "{}weights need the shape (M_v, M_u) of the coefficients ({}, {})",
                    p.prefix(),
                    c.shape()[0],
                    c.shape()[1]
                )));
            }
            weights.push(if w.is_set() { Some(w.reals().to_vec()) } else { None });
        }
        if weights.iter().any(|w| w.is_some()) {
            for (k, w) in weights.iter().enumerate() {
                let n = (h.spline_counts[2 * k] * h.spline_counts[2 * k + 1]) as usize;
                match w {
                    Some(w) => h.weights.extend_from_slice(w),
                    None => h.weights.extend(std::iter::repeat_n(1.0, n)),
                }
            }
        }
        h.patch_count = m.children("patches").len() as i32;
        h.knot_count = h.knots.len() as i64;
        h.coefficient_count = (h.coefficients.len() / 3) as i64;
    } else {
        h.geometry = PatchKind::Analytic;
        for p in m.children("patches") {
            let mut a = AnalyticPatchDesc::default();
            if !p.get("map").is_set() {
                return Err(Error::value(format!("{}an AnalyticPatch needs its map", p.prefix())));
            }
            a.kind = choice(&ANALYTIC_KINDS, p.get("map").text());
            a.profile = choice(&PROFILES, p.get("profile").text());
            a.a = p.get("a").real();
            a.b = p.get("b").real();
            a.theta0 = p.get("theta0").real();
            a.theta1 = p.get("theta1").real();
            let pp = p.get("p");
            a.p[..pp.reals().len()].copy_from_slice(pp.reals());
            let v = p.get("vertices");
            let need: usize = if a.kind == AnalyticKind::Bilinear {
                12
            } else if a.kind == AnalyticKind::SegmentArc {
                6
            } else {
                0
            };
            if need != 0 && (!v.is_set() || v.reals().len() != need) {
                return Err(Error::value(format!(
                    "{}a {} AnalyticPatch needs vertices of shape ({}, 3)",
                    p.prefix(),
                    p.get("map").text(),
                    need / 3
                )));
            }
            if need != 0 {
                a.vertices[..need].copy_from_slice(v.reals());
            }
            h.analytic_patches.push(a);
        }
        h.patch_count = h.analytic_patches.len() as i32;
    }
    if h.discretization.len() != h.patch_count as usize {
        return Err(Error::value(format!(
            "{}discretization has {} values for {} patches",
            m.prefix(),
            h.discretization.len(),
            h.patch_count
        )));
    }
    Ok(d)
}
