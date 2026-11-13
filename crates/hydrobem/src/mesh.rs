//! The meshes of the engine (an element list with its shapes, low or higher order) and their
//! construction and validation from the descriptors (reference/inputs-meshes).

use crate::mesh_desc::PatchKind;
use std::sync::Arc;

use crate::analytic::AnalyticPatch;
use crate::bspline::{MAX_BASIS_ORDER, basis_and_derivative, find_span};
use crate::constants::PI;
use crate::error::{Error, Result, at_element};
use crate::geometry::{COINCIDENCE_TOL, PLANE_TOL, Panel, Vec3, cross, make_panel, norm, v3};
use crate::mesh_desc::*;
use crate::patch_geometry::bilinear_map;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Range {
    pub first: i32, // one-based inclusive
    pub last: i32,
}

/// Geometry of one stored element used by the rules: a representative normal
/// (into the fluid) and the points bounding the element (vertices of a panel or
/// flat patch, control points of a spline patch, which contain the surface).
#[derive(Clone, Debug, Default)]
pub struct RecordShape {
    pub normal: Vec3,
    pub hull: Vec<Vec3>,
    pub centre: Vec3, // area centroid (panel) or parametric centre (patch)
    pub area: f64,    // low-order panels and flat patches
    pub size: f64,    // diameter of the hull
}

#[derive(Clone, Debug)]
pub struct Mesh {
    pub patch: bool, // a higher-order mesh of patches (flat, spline, analytic, trimmed)
    pub geometry: PatchKind,
    pub count: i32,
    pub reference_length: f64,
    pub gravity: f64,
    pub isx: i32,
    pub isy: i32,
    pub panels: Vec<Panel>,       // low order
    pub shapes: Vec<RecordShape>, // every element (low and higher order)
    // higher-order data (patch vertices, B-spline geometry, discretization)
    pub vertices: Vec<f64>,
    pub orders: Vec<i32>,
    pub ncoef: Vec<i32>,
    pub knots: Vec<f64>,
    pub coefficients: Vec<f64>,
    pub weights: Vec<f64>, // rational patches: one per coefficient; empty: none rational
    // B-spline meshes: the offsets of each patch in knots and coefficients, set with them by
    // the constructors (build_patch_mesh, tanks::subset_mesh); empty for the other geometries
    pub layout: SplineLayout,
    pub discretization: Vec<PatchDiscretization>,
    pub analytic: Vec<AnalyticPatch>, // PatchKind::Analytic
    // a trimmed patch mesh: the elements are trimmed parts of the patches of `source`;
    // geometry is PatchKind::Analytic (smooth maps with x_u x x_v into the body) and every
    // evaluation goes through patch_eval
    pub source: Option<Arc<Mesh>>,
    pub trim: Vec<PatchTrim>,
    pub trim_r: [f64; 9],
    pub trim_z: f64,
    pub trim_root: Vec<Vec<f64>>, // per element: the waterline roots at 129 fibres (NaN: none)
}

impl Default for Mesh {
    fn default() -> Self {
        Mesh {
            patch: false,
            geometry: PatchKind::Flat,
            count: 0,
            reference_length: 1.0,
            gravity: 0.0,
            isx: 0,
            isy: 0,
            panels: Vec::new(),
            shapes: Vec::new(),
            vertices: Vec::new(),
            orders: Vec::new(),
            ncoef: Vec::new(),
            knots: Vec::new(),
            coefficients: Vec::new(),
            weights: Vec::new(),
            layout: SplineLayout::default(),
            discretization: Vec::new(),
            analytic: Vec::new(),
            source: None,
            trim: Vec::new(),
            trim_r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            trim_z: 0.0,
            trim_root: Vec::new(),
        }
    }
}

impl Mesh {
    /// Turns element rec over, so that its normal points the other way: a low-order panel or a flat
    /// patch by its vertex order reversed (0, 3, 2, 1; a flat patch's discretization transposed
    /// with it), a B-spline or analytic patch by its parameter u reversed (x(u, v) becomes
    /// x(-u, v): the same surface on the same spans, x_u and with it the normal negated). False for
    /// an element of a trimmed mesh (its maps are those of the source mesh) and a degenerate panel.
    pub fn turn_over(&mut self, rec: usize) -> bool {
        if self.trimmed() {
            return false;
        }
        let flat = self.patch && self.geometry == PatchKind::Flat;
        if self.patch && !flat {
            self.reverse_u(rec);
            // the centre (u = 0) and the hull (a symmetric sample of the map, or the control net) stay
            let s = &mut self.shapes[rec];
            s.normal = -1.0 * s.normal;
            return true;
        }
        let raw = if flat {
            let q = &self.vertices[12 * rec..12 * rec + 12];
            [v3(q[0], q[1], q[2]), v3(q[3], q[4], q[5]), v3(q[6], q[7], q[8]), v3(q[9], q[10], q[11])]
        } else {
            self.panels[rec].raw
        };
        let order = [0, 3, 2, 1];
        let mut v12 = [0.0f64; 12];
        for (k, &i) in order.iter().enumerate() {
            for d in 0..3 {
                v12[3 * k + d] = raw[i][d];
            }
        }
        let mut panel = Panel::default();
        if !make_panel(&v12, &mut panel).is_empty() {
            return false;
        }
        if flat {
            self.vertices[12 * rec..12 * rec + 12].copy_from_slice(&v12);
            if let Some(d) = self.discretization.get_mut(rec) {
                std::mem::swap(&mut d.nu, &mut d.nv);
                std::mem::swap(&mut d.ku, &mut d.kv);
            }
        } else {
            self.panels[rec] = panel;
        }
        let s = &mut self.shapes[rec];
        s.normal = panel.normal;
        s.hull = panel.raw.to_vec();
        s.centre = panel.centroid;
        if !flat {
            s.area = panel.area;
        }
        true
    }

    pub fn trimmed(&self) -> bool {
        self.source.is_some()
    }

    // The map of B-spline or analytic patch rec with u reversed: a B-spline patch's u knots
    // reflected about the middle of their range and its coefficients (and weights) reversed along
    // u; an analytic patch's azimuths theta0 and theta1 swapped (with them the segment ends of a
    // segment-arc patch), a bilinear one's corners exchanged along u (0 <-> 1, 3 <-> 2).
    fn reverse_u(&mut self, rec: usize) {
        if self.geometry == PatchKind::Analytic {
            let a = &mut self.analytic[rec];
            let v = a.vt;
            if a.kind == AnalyticKind::Bilinear {
                for (k, i) in [1usize, 0, 3, 2].into_iter().enumerate() {
                    a.vt[3 * k..3 * k + 3].copy_from_slice(&v[3 * i..3 * i + 3]);
                }
                return;
            }
            std::mem::swap(&mut a.theta0, &mut a.theta1);
            if a.kind == AnalyticKind::SegmentArc {
                a.vt[..3].copy_from_slice(&v[3..6]);
                a.vt[3..6].copy_from_slice(&v[..3]);
            }
            return;
        }
        let (ko, co) = (self.layout.knot_offset[rec] as usize, self.layout.coef_offset[rec] as usize);
        let (nu, nv) = (self.ncoef[2 * rec] as usize, self.ncoef[2 * rec + 1] as usize);
        let lu = nu + self.orders[2 * rec] as usize;
        let tu = &mut self.knots[ko..ko + lu];
        let ends = tu[0] + tu[lu - 1];
        tu.reverse();
        for t in tu.iter_mut() {
            *t = ends - *t;
        }
        for j in 0..nv {
            let row = co + j * nu;
            for i in 0..nu / 2 {
                let (a, b) = (row + i, row + nu - 1 - i);
                for d in 0..3 {
                    self.coefficients.swap(3 * a + d, 3 * b + d);
                }
                if !self.weights.is_empty() {
                    self.weights.swap(a, b);
                }
            }
        }
    }
}

/// Largest B-spline geometry order (ku, kv) of a patch: the evaluators use
/// fixed arrays of this size.
pub const MAX_SPLINE_ORDER: i32 = MAX_BASIS_ORDER;

// Meshes from the descriptors (mesh_desc.rs): low-order panels, flat, spline and analytic patch
// meshes, and trimmed patch meshes (reference/inputs-meshes).

fn check_header(reference_length: f64, gravity: f64, isx: i32, isy: i32) -> Result<()> {
    if !(reference_length.is_finite() && reference_length > 0.0) {
        return Err(Error::value("mesh reference_length (ULEN) must be a positive finite number"));
    }
    if !(gravity.is_finite() && gravity > 0.0) {
        return Err(Error::value("mesh gravity must be a positive finite number"));
    }
    if (isx != 0 && isx != 1) || (isy != 0 && isy != 1) {
        return Err(Error::value("mesh symmetry flags (ISX, ISY) must be 0 or 1"));
    }
    Ok(())
}

// the discretization of a patch or a trimmed element: the potential spans, orders and Gauss orders
fn check_discretization(d: &PatchDiscretization, at: &str) -> std::result::Result<(), String> {
    if d.nu < 1 || d.nv < 1 {
        return Err(format!("{at}the potential spans nu, nv (the columns 1, 2 of discretization) must be >= 1"));
    }
    if d.ku < 2 || d.ku > 6 || d.kv < 2 || d.kv > 6 {
        return Err(format!("{at}the potential orders ku, kv (the columns 3, 4 of discretization) must be in 2..6"));
    }
    if d.iquo < 1 || d.iqui < 1 {
        return Err(format!("{at}the Gauss orders iquo, iqui (the columns 5, 6 of discretization) must be >= 1"));
    }
    Ok(())
}

fn hull_size(h: &[Vec3]) -> f64 {
    let mut d: f64 = 0.0;
    for i in 0..h.len() {
        for j in i + 1..h.len() {
            d = d.max(norm(h[i] - h[j]));
        }
    }
    d
}

/// The offsets of the patches of a B-spline mesh in its concatenated knots and coefficients
/// (count + 1 entries each, the last the totals).
#[derive(Clone, Debug, Default)]
pub struct SplineLayout {
    pub knot_offset: Vec<i64>,
    pub coef_offset: Vec<i64>,
}

/// The layout of the spline data of m (orders and ncoef of its m.count patches).
pub(crate) fn spline_layout(m: &Mesh) -> SplineLayout {
    let mut l = SplineLayout { knot_offset: Vec::new(), coef_offset: Vec::new() };
    let (mut ko, mut co) = (0i64, 0i64);
    for p in 0..m.count as usize {
        l.knot_offset.push(ko);
        l.coef_offset.push(co);
        let (kug, kvg) = (m.orders[2 * p], m.orders[2 * p + 1]);
        let (nug, nvg) = (m.ncoef[2 * p], m.ncoef[2 * p + 1]);
        ko += nug as i64 + kug as i64 + nvg as i64 + kvg as i64;
        co += nug as i64 * nvg as i64;
    }
    l.knot_offset.push(ko);
    l.coef_offset.push(co);
    l
}

/// "" when t[0 .. n+k) is a complete (clamped) knot vector of order k with n coefficients and
/// C1 inside, else the reason (dir names the direction, "u" or "v").
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
pub fn check_knots(t: &[f64], n: i32, k: i32, dir: &str) -> String {
    // Complete (clamped) knot vector of order k with n coefficients, C1 inside.
    let d = dir;
    let len = (n + k) as usize;
    let k = k as usize;
    let n = n as usize;
    if !t[..len].iter().all(|x| x.is_finite()) {
        return format!("{d} knot vector has a non-finite value");
    }
    for i in 1..len {
        if t[i] < t[i - 1] {
            return format!("{d} knot vector is not non-decreasing");
        }
    }
    if !(t[0] < t[len - 1]) {
        return format!("{d} knot vector has zero length");
    }
    for i in 1..k {
        if t[i] != t[0] || t[len - 1 - i] != t[len - 1] {
            return format!("{d} knot vector is not complete (end knots must have multiplicity equal to the order)");
        }
    }
    // Interior knots: multiplicity m gives continuity C^(k-1-m); C1 needs m <= k-2.
    let mut i = k;
    while i < n {
        let mut j = i;
        while j + 1 < n && t[j + 1] == t[i] {
            j += 1;
        }
        let mult = (j - i + 1) as i64;
        if mult > k as i64 - 2 {
            return format!("{d} knot vector has a C0 (or discontinuous) interior knot");
        }
        i = j + 1;
    }
    String::new()
}

/// A validated low-order mesh (panel projection rule).
/// Fails with ErrorKind::Value naming the one-based element of an invalid panel.
pub fn build_mesh(desc: &LowOrderMeshDesc) -> Result<Mesh> {
    check_header(desc.reference_length, desc.gravity, desc.symmetry_x, desc.symmetry_y)?;
    if desc.panel_count < 1 {
        return Err(Error::value("mesh must have at least one panel"));
    }
    if desc.vertices.len() < 12 * desc.panel_count as usize {
        return Err(Error::value(format!(
            "the mesh has {} panels and the vertices of {} (4 x 3 numbers per panel)",
            desc.panel_count,
            desc.vertices.len() / 12
        )));
    }
    let mut m = Mesh {
        count: desc.panel_count,
        reference_length: desc.reference_length,
        gravity: desc.gravity,
        isx: desc.symmetry_x,
        isy: desc.symmetry_y,
        ..Mesh::default()
    };
    let n = m.count as usize;
    m.panels = vec![Panel::default(); n];
    m.shapes = vec![RecordShape::default(); n];
    for p in 0..n {
        let why = make_panel(&desc.vertices[12 * p..], &mut m.panels[p]);
        if !why.is_empty() {
            return Err(Error::value(format!("mesh element {}: {}", p + 1, why)).at(at_element(p as i32 + 1)));
        }
        let panel = &m.panels[p];
        let s = &mut m.shapes[p];
        s.normal = panel.normal;
        s.hull = panel.raw.to_vec();
        s.centre = panel.centroid;
        s.area = panel.area;
        s.size = panel.diameter;
    }
    Ok(m)
}

/// The rotation of a trim: x = R xi + (0, 0, elevation), the pitch
/// (degrees, about the eta axis) first, then the roll (about the xi axis); row-major.
pub fn trim_rotation(pitch: f64, roll: f64, r: &mut [f64; 9]) {
    let (c2, s2) = ((pitch * PI / 180.0).cos(), (pitch * PI / 180.0).sin());
    let (c3, s3) = ((roll * PI / 180.0).cos(), (roll * PI / 180.0).sin());
    let q = [c2, s2 * s3, s2 * c3, 0.0, c3, -s3, -s2, c2 * s3, c2 * c3];
    r.copy_from_slice(&q);
}

// a patch of an untrimmed mesh, any geometry (flat: the bilinear map of its vertices)
fn source_eval(m: &Mesh, p: i32, u: f64, v: f64, x: &mut Vec3, xu: &mut Vec3, xv: &mut Vec3) {
    if m.geometry == PatchKind::Flat {
        let q = &m.vertices[12 * p as usize..];
        let mut vt = [Vec3::ZERO; 4];
        for i in 0..4 {
            vt[i] = v3(q[3 * i], q[3 * i + 1], q[3 * i + 2]);
        }
        bilinear_map(&vt, u, v, x, xu, xv);
        return;
    }
    patch_eval(m, p, u, v, x, xu, xv);
}

/// Point and parametric derivatives of an element of any patch mesh at (u, v) in [-1, 1]^2 (an
/// untrimmed flat patch by the bilinear map of its vertices, the others by patch_eval).
pub fn record_eval(m: &Mesh, p: i32, u: f64, v: f64, x: &mut Vec3, xu: &mut Vec3, xv: &mut Vec3) {
    if m.trimmed() {
        patch_eval(m, p, u, v, x, xu, xv);
    } else {
        source_eval(m, p, u, v, x, xu, xv);
    }
}

// the element's patch reflected and moved into the trimmed position, u reversed where that
// keeps x_u x x_v pointing into the body (a reflection of odd order, or a flat source); the
// element's square is the range [u0, u1] x [v0, v1] of the source parameters
fn trim_base(m: &Mesh, t: &PatchTrim, u: f64, v: f64, x: &mut Vec3, xu: &mut Vec3, xv: &mut Vec3) {
    let s: &Mesh = m.source.as_ref().expect("a trimmed mesh");
    let su = if (t.sx * t.sy < 0) != (s.geometry == PatchKind::Flat) { -1.0 } else { 1.0 };
    let whole = t.range[0] == 0.0 && t.range[1] == 0.0 && t.range[2] == 0.0 && t.range[3] == 0.0;
    let u0 = if whole { -1.0 } else { t.range[0] };
    let v0 = if whole { -1.0 } else { t.range[2] };
    let hu = if whole { 1.0 } else { 0.5 * (t.range[1] - t.range[0]) };
    let hv = if whole { 1.0 } else { 0.5 * (t.range[3] - t.range[2]) };
    let (mut y, mut yu, mut yv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    source_eval(s, t.patch, u0 + (su * u + 1.0) * hu, v0 + (v + 1.0) * hv, &mut y, &mut yu, &mut yv);
    yu = hu * yu;
    yv = hv * yv;
    let (tsx, tsy) = (t.sx as f64, t.sy as f64);
    let r = &m.trim_r;
    let rot = |a: Vec3| -> Vec3 {
        let b = v3(tsx * a[0], tsy * a[1], a[2]);
        v3(
            r[0] * b[0] + r[1] * b[1] + r[2] * b[2],
            r[3] * b[0] + r[4] * b[1] + r[5] * b[2],
            r[6] * b[0] + r[7] * b[1] + r[8] * b[2],
        )
    };
    *x = rot(y);
    x[2] += m.trim_z;
    *xu = su * rot(yu);
    *xv = rot(yv);
}

// the fibre of a cut element at the fixed parameter a: its point, derivative along a and along
// the fibre at the fibre parameter sv
fn fibre_at(m: &Mesh, t: &PatchTrim, a: f64, sv: f64, y: &mut Vec3, ya: &mut Vec3, ys: &mut Vec3) {
    let (mut bu, mut bv) = (Vec3::ZERO, Vec3::ZERO);
    let along_v = t.cut <= 2;
    if along_v {
        trim_base(m, t, a, sv, y, &mut bu, &mut bv);
    } else {
        trim_base(m, t, sv, a, y, &mut bu, &mut bv);
    }
    *ya = if along_v { bu } else { bv };
    *ys = if along_v { bv } else { bu };
}

// the waterline root r of the fibre a by sampling from its wet end and bisection, and dr/da =
// -z_a / z_s (implicit); false for a fibre wholly submerged. A fibre of a cut element crosses the
// waterline once (the trim classifies the fibres on 129 samples and refuses a patch whose fibres
// cross twice), so the first sign change bracketed on 32 samples is its root; a fibre that dips
// above and below z = 0 between two samples is taken at its first bracket
fn fibre_search(m: &Mesh, t: &PatchTrim, a: f64, r: &mut f64, dr: &mut f64) -> bool {
    let e = if t.cut % 2 == 1 { -1.0 } else { 1.0 };
    const SAMPLES: i32 = 32; // the first crossing is bracketed on 32 samples, then bisected
    let (mut y, mut ya, mut ys) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    let mut s0 = e;
    fibre_at(m, t, a, s0, &mut y, &mut ya, &mut ys);
    let mut z0 = y[2];
    for k in 1..=SAMPLES {
        let s1 = e - e * 2.0 * k as f64 / SAMPLES as f64;
        fibre_at(m, t, a, s1, &mut y, &mut ya, &mut ys);
        if y[2] > 0.0 && z0 <= 0.0 {
            let (mut lo, mut hi) = (s0, s1); // z(lo) <= 0 < z(hi)
            for _it in 0..60 {
                let mid = 0.5 * (lo + hi);
                fibre_at(m, t, a, mid, &mut y, &mut ya, &mut ys);
                if y[2] <= 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            *r = 0.5 * (lo + hi);
            fibre_at(m, t, a, *r, &mut y, &mut ya, &mut ys);
            *dr = if ys[2].abs() > 0.0 { -ya[2] / ys[2] } else { 0.0 };
            return true;
        }
        s0 = s1;
        z0 = y[2];
    }
    false
}

// Newton steps on z(a, s) = 0 from s = r, kept where they stay in the fibre: every path ends
// at the same fixed point, so the map is continuous from fibre to fibre; dr/da = -z_a / z_s
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
fn fibre_polish(m: &Mesh, t: &PatchTrim, a: f64, r: &mut f64, dr: &mut f64) {
    let (mut y, mut ya, mut ys) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    for _it in 0..4 {
        fibre_at(m, t, a, *r, &mut y, &mut ya, &mut ys);
        if !(ys[2].abs() > 0.0) {
            break;
        }
        let rr = *r - y[2] / ys[2];
        if !(-1.0..=1.0).contains(&rr) || rr == *r {
            break;
        }
        *r = rr;
    }
    fibre_at(m, t, a, *r, &mut y, &mut ya, &mut ys);
    *dr = if ys[2].abs() > 0.0 { -ya[2] / ys[2] } else { 0.0 };
}

// the root of element p at a: from the element's table where both neighbours have a root (the
// interpolated root polished), else the search
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
fn fibre_root(m: &Mesh, p: i32, a: f64, r: &mut f64, dr: &mut f64) -> bool {
    let t = &m.trim[p as usize];
    if (p as usize) < m.trim_root.len() && !m.trim_root[p as usize].is_empty() {
        let tab = &m.trim_root[p as usize];
        let n = tab.len() as i32 - 1;
        let fi = 0.5 * (a + 1.0) * n as f64;
        let i0 = (n - 1).min(0.max(fi as i32));
        let i0u = i0 as usize;
        if !tab[i0u].is_nan() && !tab[i0u + 1].is_nan() {
            *r = tab[i0u] + (fi - i0 as f64) * (tab[i0u + 1] - tab[i0u]);
            // Newton steps from the interpolated root until they vanish (two or three): the same
            // fixed point from every start; the search where they leave the fibre or stall
            let (mut y, mut ya, mut ys) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
            for _it in 0..8 {
                fibre_at(m, t, a, *r, &mut y, &mut ya, &mut ys);
                if !(ys[2].abs() > 0.0) {
                    break;
                }
                let step = y[2] / ys[2];
                *r -= step;
                if !(*r >= -1.0 && *r <= 1.0) {
                    break;
                }
                if step.abs() <= 1e-15 {
                    *dr = -ya[2] / ys[2];
                    return true;
                }
            }
        }
    }
    if !fibre_search(m, t, a, r, dr) {
        *r = if t.cut % 2 == 1 { 1.0 } else { -1.0 }; // a fibre wholly submerged: to its far end
        *dr = 0.0;
        return false;
    }
    fibre_polish(m, t, a, r, dr);
    true
}

// an element of a trimmed mesh: along every fibre (a line of constant u for cut 1, 2, of
// constant v for 3, 4) the submerged part from its wet end e to the waterline root r, mapped
// linearly onto [-1, 1]
fn trimmed_eval(m: &Mesh, p: i32, u: f64, v: f64, x: &mut Vec3, xu: &mut Vec3, xv: &mut Vec3) {
    let t = &m.trim[p as usize];
    if t.cut == 0 {
        trim_base(m, t, u, v, x, xu, xv);
        return;
    }
    let along_v = t.cut <= 2;
    let e = if t.cut % 2 == 1 { -1.0 } else { 1.0 };
    let a = if along_v { u } else { v };
    let tt = if along_v { v } else { u };
    let (mut r, mut dr) = (-e, 0.0); // a fibre wholly submerged: the whole fibre
    let cut = fibre_root(m, p, a, &mut r, &mut dr);
    let w = 0.5 * (tt + 1.0);
    let sv = if e < 0.0 { -1.0 + w * (r + 1.0) } else { r + w * (1.0 - r) };
    let ds_dt = if e < 0.0 { 0.5 * (r + 1.0) } else { 0.5 * (1.0 - r) };
    let ds_da = (if e < 0.0 { w } else { 1.0 - w }) * dr;
    let (mut y, mut ya, mut ys) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    fibre_at(m, t, a, sv, &mut y, &mut ya, &mut ys);
    *x = y;
    if along_v {
        *xu = ya + ds_da * ys;
        *xv = ds_dt * ys;
    } else {
        *xu = ds_dt * ys;
        *xv = ya + ds_da * ys;
    }
    if cut && tt == (if e < 0.0 { 1.0 } else { -1.0 }) {
        x[2] = 0.0; // the waterline end exactly in z = 0
    }
}

/// Point and parametric derivatives of a patch of a trimmed, spline or analytic mesh at (u, v) in
/// [-1, 1]^2 (a trimmed mesh by its trim map over the source mesh, which may be flat).
/// patch: zero-based patch; x point, xu dx/du, xv dx/dv. An untrimmed flat mesh has no spline
/// data and is not evaluated here (it would index the empty orders): its bilinear map is
/// source_eval's.
pub fn patch_eval(mesh: &Mesh, patch: i32, u: f64, v: f64, x: &mut Vec3, xu: &mut Vec3, xv: &mut Vec3) {
    let m = mesh;
    let p = patch as usize;
    if m.trimmed() {
        trimmed_eval(m, patch, u, v, x, xu, xv);
        return;
    }
    if m.geometry == PatchKind::Analytic {
        m.analytic[p].eval(u, v, x, xu, xv);
        return;
    }
    // the offsets of the patch in the concatenated knots and coefficients
    let (ko, co) = (m.layout.knot_offset[p] as usize, m.layout.coef_offset[p] as usize);
    let (kug, kvg) = (m.orders[2 * p], m.orders[2 * p + 1]);
    let (nug, nvg) = (m.ncoef[2 * p], m.ncoef[2 * p + 1]);
    let lu = (nug + kug) as usize;
    let lv = (nvg + kvg) as usize;
    let tu = &m.knots[ko..ko + lu];
    let tv = &m.knots[ko + lu..ko + lu + lv];
    let (tu0, tu1) = (tu[0], tu[lu - 1]);
    let (tv0, tv1) = (tv[0], tv[lv - 1]);
    // parameters are given on [-1, 1]^2 and mapped affinely to the knot range
    let uu = tu0 + 0.5 * (u + 1.0) * (tu1 - tu0);
    let vv = tv0 + 0.5 * (v + 1.0) * (tv1 - tv0);
    let su = 0.5 * (tu1 - tu0);
    let sv = 0.5 * (tv1 - tv0);
    let iu = find_span(tu, nug, kug, uu);
    let iv = find_span(tv, nvg, kvg, vv);
    let mut bu = [0.0f64; 2 * MAX_BASIS_ORDER as usize];
    let mut bv = [0.0f64; 2 * MAX_BASIS_ORDER as usize];
    basis_and_derivative(tu, kug, iu, uu, &mut bu[..2 * kug as usize]);
    basis_and_derivative(tv, kvg, iv, vv, &mut bv[..2 * kvg as usize]);
    *x = Vec3::ZERO;
    *xu = Vec3::ZERO;
    *xv = Vec3::ZERO;
    let c = &m.coefficients[3 * co..];
    let (ku, kv) = (kug as usize, kvg as usize);
    if !m.weights.is_empty() {
        // rational: x = A / W with A = sum w N P, W = sum w N; x_u = (A_u - x W_u) / W
        let w = &m.weights[co..];
        let (mut w0, mut wu, mut wv) = (0.0, 0.0, 0.0);
        for b in 0..kv {
            let jv = (iv - (kvg - 1)) as usize + b;
            for a in 0..ku {
                let ju = (iu - (kug - 1)) as usize + a;
                let k = jv * nug as usize + ju;
                let q = &c[3 * k..];
                let cp = w[k] * v3(q[0], q[1], q[2]);
                let (n, nu, nv) = (bu[a] * bv[b], bu[ku + a] * bv[b] * su, bu[a] * bv[kv + b] * sv);
                *x += n * cp;
                *xu += nu * cp;
                *xv += nv * cp;
                w0 += n * w[k];
                wu += nu * w[k];
                wv += nv * w[k];
            }
        }
        *x = (1.0 / w0) * *x;
        *xu = (1.0 / w0) * (*xu - wu * *x);
        *xv = (1.0 / w0) * (*xv - wv * *x);
        return;
    }
    for b in 0..kv {
        let jv = (iv - (kvg - 1)) as usize + b;
        for a in 0..ku {
            let ju = (iu - (kug - 1)) as usize + a;
            let q = &c[3 * (jv * nug as usize + ju)..];
            let cp = v3(q[0], q[1], q[2]);
            *x += (bu[a] * bv[b]) * cp;
            *xu += (bu[ku + a] * bv[b] * su) * cp;
            *xv += (bu[a] * bv[kv + b] * sv) * cp;
        }
    }
}

/// A validated higher-order mesh (flat or B-spline patches).
/// Fails with ErrorKind::Value naming the patch.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn build_patch_mesh(desc: &PatchMeshDesc) -> Result<Mesh> {
    let src = build_source_mesh(desc)?;
    if desc.trim_count == 0 {
        return Ok(src);
    }
    // a trimmed mesh: its elements are the trimmed parts of the source patches
    if desc.trim_count < 0 || desc.trim.len() != desc.trim_count as usize {
        return Err(Error::value(format!(
            "the trimmed mesh has {} elements and {} trim elements",
            desc.trim_count,
            desc.trim.len()
        )));
    }
    if !desc.trim_elevation.is_finite() || !desc.trim_pitch.is_finite() || !desc.trim_roll.is_finite() {
        return Err(Error::value("trim elevation, pitch and roll must be finite"));
    }
    let mut m = Mesh {
        patch: true,
        geometry: PatchKind::Analytic,
        count: desc.trim_count,
        reference_length: src.reference_length,
        gravity: src.gravity,
        // a pitch removes the symmetry about x = 0, a roll that about y = 0
        isx: if src.isx != 0 && desc.trim_pitch == 0.0 { 1 } else { 0 },
        isy: if src.isy != 0 && desc.trim_roll == 0.0 { 1 } else { 0 },
        ..Mesh::default()
    };
    let n = m.count as usize;
    m.trim = desc.trim[..n].to_vec();
    trim_rotation(desc.trim_pitch, desc.trim_roll, &mut m.trim_r);
    m.trim_z = desc.trim_elevation;
    let source_count = src.count;
    m.source = Some(Arc::new(src));
    m.shapes = vec![RecordShape::default(); n];
    // the waterline roots of 129 fibres of every cut element, by the search (NaN: a submerged fibre)
    let mut roots: Vec<Vec<f64>> = vec![Vec::new(); n];
    for p in 0..n {
        let t = &m.trim[p];
        if t.cut < 1 || t.cut > 4 || t.patch < 0 || t.patch >= source_count {
            continue;
        }
        let mut ok = true; // a valid range (checked below)
        for k in 0..4 {
            ok = ok && t.range[k] >= -1.0 && t.range[k] <= 1.0;
        }
        if !ok {
            continue;
        }
        let mut tab = vec![0.0; 129];
        for k in 0..=128 {
            let (mut r, mut dr) = (0.0, 0.0);
            tab[k] = if fibre_search(&m, t, -1.0 + 2.0 * k as f64 / 128.0, &mut r, &mut dr) { r } else { f64::NAN };
        }
        roots[p] = tab;
    }
    m.trim_root = roots;
    let (src_isx, src_isy) = {
        let s = m.source.as_ref().unwrap();
        (s.isx, s.isy)
    };
    for p in 0..n {
        let t = m.trim[p];
        let at = format!("trim element {}: ", p + 1);
        let here = at_element(p as i32 + 1);
        if t.patch < 0 || t.patch >= source_count {
            return Err(Error::value(format!(
                "{at}the source patch (the column 1 of trim_elements) must be in 1..{source_count}"
            ))
            .at(here));
        }
        if (t.sx != 1 && t.sx != -1) || (t.sy != 1 && t.sy != -1) {
            return Err(Error::value(format!("{at}sx and sy must be +1 or -1")).at(here));
        }
        if (t.sx < 0 && src_isx == 0) || (t.sy < 0 && src_isy == 0) {
            return Err(Error::value(format!("{at}a reflection the source mesh does not declare")).at(here));
        }
        if t.sx < 0 && m.isx != 0 {
            return Err(
                Error::value(format!("{at}a reflected copy about x = 0 of a mesh that keeps that symmetry")).at(here)
            );
        }
        if t.sy < 0 && m.isy != 0 {
            return Err(
                Error::value(format!("{at}a reflected copy about y = 0 of a mesh that keeps that symmetry")).at(here)
            );
        }
        if t.cut < 0 || t.cut > 4 {
            return Err(Error::value(format!("{at}cut must be 0..4")).at(here));
        }
        let whole = t.range[0] == 0.0 && t.range[1] == 0.0 && t.range[2] == 0.0 && t.range[3] == 0.0;
        if !whole
            && !(t.range[0] >= -1.0
                && t.range[0] < t.range[1]
                && t.range[1] <= 1.0
                && t.range[2] >= -1.0
                && t.range[2] < t.range[3]
                && t.range[3] <= 1.0)
        {
            return Err(
                Error::value(format!("{at}range must be -1 <= u0 < u1 <= 1, -1 <= v0 < v1 <= 1 (or all 0)")).at(here)
            );
        }
        let d = t.discretization;
        check_discretization(&d, &at).map_err(|e| Error::value(e).at(here))?;
        m.discretization.push(d);
        let mut sh = RecordShape::default();
        let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        for i in 0..=8 {
            for j in 0..=8 {
                patch_eval(&m, p as i32, -1.0 + 0.25 * i as f64, -1.0 + 0.25 * j as f64, &mut x, &mut xu, &mut xv);
                sh.hull.push(x);
            }
        }
        sh.size = hull_size(&sh.hull);
        if !(sh.size > 0.0) {
            return Err(Error::value(format!("{at}degenerate trimmed patch")).at(here));
        }
        for h in &sh.hull {
            if h[2] > COINCIDENCE_TOL * sh.size {
                return Err(Error::value(format!(
                    "{at}a part of the element lies above z = 0 (a cut the fibres do not describe)"
                ))
                .at(here));
            }
        }
        patch_eval(&m, p as i32, 0.0, 0.0, &mut x, &mut xu, &mut xv);
        let nrm = cross(xu, xv);
        let nn = norm(nrm);
        if !(nn > 0.0) {
            return Err(Error::value(format!("{at}degenerate patch (zero Jacobian at the parametric centre)")).at(here));
        }
        sh.normal = (-1.0 / nn) * nrm;
        sh.centre = x;
        m.shapes[p] = sh;
    }
    Ok(m)
}

#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn build_source_mesh(desc: &PatchMeshDesc) -> Result<Mesh> {
    check_header(desc.reference_length, desc.gravity, desc.symmetry_x, desc.symmetry_y)?;
    if desc.geometry != PatchKind::Flat && desc.geometry != PatchKind::Spline && desc.geometry != PatchKind::Analytic {
        return Err(Error::value("patch mesh geometry must be flat (0), spline (1) or analytic (2)"));
    }
    if desc.patch_count < 1 {
        return Err(Error::value("patch mesh must have at least one patch"));
    }
    if desc.discretization.len() < desc.patch_count as usize {
        return Err(Error::value(format!(
            "the patch mesh has {} patches and the discretization of {}",
            desc.patch_count,
            desc.discretization.len()
        )));
    }
    let mut m = Mesh {
        patch: true,
        geometry: desc.geometry,
        count: desc.patch_count,
        reference_length: desc.reference_length,
        gravity: desc.gravity,
        isx: desc.symmetry_x,
        isy: desc.symmetry_y,
        ..Mesh::default()
    };
    let n = m.count as usize;
    m.discretization = desc.discretization[..n].to_vec();
    for p in 0..n {
        let d = &m.discretization[p];
        let at = format!("patch {}: ", p + 1);
        check_discretization(d, &at).map_err(|e| Error::value(e).at(at_element(p as i32 + 1)))?;
    }
    m.shapes = vec![RecordShape::default(); n];
    if m.geometry == PatchKind::Flat {
        if desc.vertices.len() < 12 * n {
            return Err(Error::value(format!(
                "the flat patch mesh has {n} patches and the vertices of {} (4 x 3 numbers per patch)",
                desc.vertices.len() / 12
            )));
        }
        m.vertices = desc.vertices[..12 * n].to_vec();
        for p in 0..n {
            let at = format!("patch {}: ", p + 1);
            let here = at_element(p as i32 + 1);
            let mut panel = Panel::default();
            let why = make_panel(&desc.vertices[12 * p..], &mut panel);
            if !why.is_empty() {
                return Err(Error::value(format!("{at}{why}")).at(here));
            }
            if panel.distinct != 4 {
                return Err(Error::value(format!("{at}a flat patch needs four distinct vertices")).at(here));
            }
            for i in 0..4 {
                if norm(panel.raw[i] - panel.proj[i]) > PLANE_TOL * panel.diameter {
                    return Err(Error::value(format!("{at}flat patch vertices are not coplanar")).at(here));
                }
            }
            let s = &mut m.shapes[p];
            s.normal = panel.normal;
            s.hull = panel.raw.to_vec();
            s.centre = panel.centroid;
            s.area = panel.area;
            s.size = panel.diameter;
        }
        return Ok(m);
    }
    if m.geometry == PatchKind::Analytic {
        if desc.analytic_patches.len() < n {
            return Err(Error::value(format!(
                "the analytic patch mesh has {n} patches and the shapes of {}",
                desc.analytic_patches.len()
            )));
        }
        for p in 0..n {
            let at = format!("patch {}: ", p + 1);
            let here = at_element(p as i32 + 1);
            let d = &desc.analytic_patches[p];
            let mut a = AnalyticPatch {
                kind: d.kind,
                profile: d.profile,
                a: d.a,
                b: d.b,
                theta0: d.theta0,
                theta1: d.theta1,
                ..AnalyticPatch::default()
            };
            let mut finite = a.a.is_finite() && a.b.is_finite() && a.theta0.is_finite() && a.theta1.is_finite();
            for k in 0..4 {
                a.p[k] = d.p[k];
                finite = finite && a.p[k].is_finite();
            }
            for k in 0..12 {
                a.vt[k] = d.vertices[k];
                finite = finite && a.vt[k].is_finite();
            }
            if !finite {
                return Err(Error::value(format!("{at}non-finite analytic patch data")).at(here));
            }
            m.analytic.push(a);
            // hull: a 9 x 9 sample of the map (the size and the bounding geometry)
            let s = &mut m.shapes[p];
            let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
            for i in 0..=8 {
                for j in 0..=8 {
                    a.eval(-1.0 + 0.25 * i as f64, -1.0 + 0.25 * j as f64, &mut x, &mut xu, &mut xv);
                    s.hull.push(x);
                }
            }
            s.size = hull_size(&s.hull);
            if !(s.size > 0.0) {
                return Err(Error::value(format!("{at}degenerate analytic patch")).at(here));
            }
            a.eval(0.0, 0.0, &mut x, &mut xu, &mut xv);
            let nrm = cross(xu, xv);
            let nn = norm(nrm);
            if !(nn > 0.0) {
                return Err(
                    Error::value(format!("{at}degenerate patch (zero Jacobian at the parametric centre)")).at(here)
                );
            }
            s.normal = (-1.0 / nn) * nrm; // x_u x x_v points into the body (reference/inputs-meshes §2)
            s.centre = x;
        }
        return Ok(m);
    }
    if desc.spline_orders.len() < 2 * n || desc.spline_counts.len() < 2 * n {
        return Err(Error::value(format!(
            "the B-spline patch mesh has {n} patches and the orders and coefficient counts of {}",
            desc.spline_orders.len().min(desc.spline_counts.len()) / 2
        )));
    }
    m.orders = desc.spline_orders[..2 * n].to_vec();
    m.ncoef = desc.spline_counts[..2 * n].to_vec();
    let (mut need_knots, mut need_coef) = (0i64, 0i64);
    for p in 0..n {
        let at = format!("patch {}: ", p + 1);
        let here = at_element(p as i32 + 1);
        let (kug, kvg) = (m.orders[2 * p], m.orders[2 * p + 1]);
        let (nug, nvg) = (m.ncoef[2 * p], m.ncoef[2 * p + 1]);
        if kug < 2 || kvg < 2 || kug > MAX_SPLINE_ORDER || kvg > MAX_SPLINE_ORDER {
            return Err(Error::value(format!("{at}the geometry orders (u_order, v_order) must be in 2..8")).at(here));
        }
        if nug < kug || nvg < kvg {
            return Err(Error::value(format!("{at}number of coefficients must be at least the order")).at(here));
        }
        need_knots += nug as i64 + kug as i64 + nvg as i64 + kvg as i64;
        need_coef += nug as i64 * nvg as i64;
    }
    if desc.knot_count != need_knots {
        return Err(Error::value("spline knot count does not match the patch orders and counts"));
    }
    if desc.coefficient_count != need_coef {
        return Err(Error::value("spline coefficient count does not match the patch counts"));
    }
    if desc.knots.len() < need_knots as usize || desc.coefficients.len() < 3 * need_coef as usize {
        return Err(Error::value(format!(
            "the B-spline patch mesh has {} knots and {} coefficients of the {need_knots} and {need_coef} its patches need",
            desc.knots.len(),
            desc.coefficients.len() / 3
        )));
    }
    m.knots = desc.knots[..need_knots as usize].to_vec();
    m.coefficients = desc.coefficients[..3 * need_coef as usize].to_vec();
    if !desc.weights.is_empty() {
        // rational patches: a positive finite weight per coefficient
        if desc.weights.len() != need_coef as usize {
            return Err(Error::value(format!(
                "the B-spline patch mesh has {} weights for its {need_coef} coefficients",
                desc.weights.len()
            )));
        }
        if let Some(w) = desc.weights.iter().find(|w| !(w.is_finite() && **w > 0.0)) {
            return Err(Error::value(format!("the weights of rational patches must be positive and finite, got {w}")));
        }
        m.weights = desc.weights.clone();
    }
    m.layout = spline_layout(&m);
    let l = m.layout.clone();
    for p in 0..n {
        let at = format!("patch {}: ", p + 1);
        let here = at_element(p as i32 + 1);
        let (kug, kvg) = (m.orders[2 * p], m.orders[2 * p + 1]);
        let (nug, nvg) = (m.ncoef[2 * p], m.ncoef[2 * p + 1]);
        let t = &m.knots[l.knot_offset[p] as usize..];
        let mut why = check_knots(t, nug, kug, "u");
        if why.is_empty() {
            why = check_knots(&t[(nug + kug) as usize..], nvg, kvg, "v");
        }
        if !why.is_empty() {
            return Err(Error::value(format!("{at}{why}")).at(here));
        }
        let mut s = RecordShape::default();
        let c = &m.coefficients[3 * l.coef_offset[p] as usize..];
        for i in 0..(nug as usize * nvg as usize) {
            if !c[3 * i].is_finite() || !c[3 * i + 1].is_finite() || !c[3 * i + 2].is_finite() {
                return Err(Error::value(format!("{at}non-finite control coefficient")).at(here));
            }
            s.hull.push(v3(c[3 * i], c[3 * i + 1], c[3 * i + 2]));
        }
        s.size = hull_size(&s.hull);
        let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        patch_eval(&m, p as i32, 0.0, 0.0, &mut x, &mut xu, &mut xv);
        let nrm = cross(xu, xv);
        let nn = norm(nrm);
        if !(nn > 0.0) {
            return Err(Error::value(format!("{at}degenerate patch (zero Jacobian at the parametric centre)")).at(here));
        }
        s.normal = (-1.0 / nn) * nrm; // x_u x x_v points into the body (reference/inputs-meshes §2)
        s.centre = x;
        m.shapes[p] = s;
    }
    Ok(m)
}
