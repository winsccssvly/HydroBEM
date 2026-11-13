//! Geometry preprocessing: the trimmed waterline (`Model::trim_mesh`), the body
//! waterline and the automatic control surface (`Model::control_surface`).
//!
//! The trimmed waterline of a mesh (low order, higher order, tank free surfaces), the body
//! waterline and the automatic control surface (theory §11.3). Arrays in, arrays out: no model and
//! no files. The tolerances are those of theory §14, with their grounds there (features below
//! 1e-6 of the length scale cannot be told from the rounding of mesh files).

use crate::analytic::AnalyticPatch;
use crate::constants::PI;
use crate::error::{Error, Result, at_element};
use crate::gdf::{edge_lengths, panel_count};
use crate::geometry::{COINCIDENCE_TOL, Vec3, cross, length_tol, norm, v3};
use crate::mesh::{Mesh, build_patch_mesh, record_eval, trim_rotation};
use crate::mesh_desc::*;
use crate::mesh_desc::{AnalyticKind, PatchKind, Profile};
use crate::polygon::{self, P2, Region};

/// Waterline chords (a, b), body-local.
pub type Segments = Vec<[Vec3; 2]>;

#[derive(Clone, Debug, Default)]
pub struct LoTrim {
    pub panels: Vec<[Vec3; 4]>, // body coordinates
    pub source: Vec<i32>,       // one-based element of the input per panel
    pub copy: Vec<[i32; 2]>,    // per panel the copy (sx, sy) of its input element
    pub symmetry: [i32; 2],     // the flags the trim keeps
    pub plate: Vec<i32>,        // per panel the generated plate (TrimDesc::plates) it is part of, -1 none
}

#[derive(Clone, Debug, Default)]
pub struct HoTrim {
    pub records: Vec<PatchTrim>, // the trimmed elements (the interior ones last)
    pub source: Vec<i32>,        // one-based source patch per element
    pub symmetry: [i32; 2],
    pub interior: Vec<[Vec3; 4]>, // interior patches to append to the source (source coordinates)
    pub interior_disc: Vec<PatchDiscretization>,
    pub interior_plate: Vec<i32>, // per interior patch the generated plate it is part of, -1 the interior surface
}

#[derive(Clone, Debug, Default)]
pub struct ControlSurfacePatches {
    pub patches: Vec<AnalyticPatch>,
    pub discretization: Vec<PatchDiscretization>,
    pub symmetry: [i32; 2],
}

// ---------------------------------------------------------------- trim, low order

fn rotated(r: &[f64; 9], x: &Vec3) -> Vec3 {
    v3(
        r[0] * x[0] + r[1] * x[1] + r[2] * x[2],
        r[3] * x[0] + r[4] * x[1] + r[5] * x[2],
        r[6] * x[0] + r[7] * x[1] + r[8] * x[2],
    )
}

// the part z <= 0 of a planar loop (Sutherland-Hodgman against one plane)
fn clip(poly: &[Vec3], tol: f64) -> Vec<Vec3> {
    let mut out: Vec<Vec3> = Vec::new();
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let ina = a[2] <= tol;
        let inb = b[2] <= tol;
        if ina {
            out.push(if a[2] <= 0.0 { a } else { v3(a[0], a[1], 0.0) });
        }
        if ina != inb {
            let t = a[2] / (a[2] - b[2]);
            let mut p = a + t * (b - a);
            p[2] = 0.0;
            out.push(p);
        }
    }
    // a vertex on the waterline appears twice
    let mut keep: Vec<Vec3> = Vec::new();
    for p in &out {
        if keep.is_empty() || norm(*p - *keep.last().unwrap()) > tol {
            keep.push(*p);
        }
    }
    if keep.len() > 1 && norm(keep[0] - *keep.last().unwrap()) <= tol {
        keep.pop();
    }
    keep
}

// the panels of a clipped loop: a triangle (repeated vertex), a quadrilateral, or a pentagon
// split into a quadrilateral and a triangle
fn quads(p: &[Vec3]) -> Result<Vec<[Vec3; 4]>> {
    Ok(match p.len() {
        0..=2 => Vec::new(),
        3 => vec![[p[0], p[1], p[2], p[2]]],
        4 => vec![[p[0], p[1], p[2], p[3]]],
        5 => vec![[p[0], p[1], p[2], p[3]], [p[0], p[3], p[4], p[4]]],
        _ => return Err(Error::value("a trimmed panel has more than five vertices (a non-convex element)")),
    })
}

#[derive(Clone, Copy)]
struct Copy_ {
    sx: f64,
    sy: f64,
}

// the reflected copies of the stored part where the trim breaks a symmetry (a pitch that of the
// plane x = 0, a roll that of y = 0)
fn trim_copies(isx: i32, isy: i32, pitch: f64, roll: f64, keep: &mut [i32; 2]) -> Vec<Copy_> {
    let kx = isx != 0 && pitch == 0.0;
    let ky = isy != 0 && roll == 0.0;
    let ex = isx != 0 && !kx;
    let ey = isy != 0 && !ky;
    keep[0] = if kx { 1 } else { 0 };
    keep[1] = if ky { 1 } else { 0 };
    let mut c = vec![Copy_ { sx: 1.0, sy: 1.0 }];
    if ex {
        c.push(Copy_ { sx: -1.0, sy: 1.0 });
    }
    if ey {
        c.push(Copy_ { sx: 1.0, sy: -1.0 });
    }
    if ex && ey {
        c.push(Copy_ { sx: -1.0, sy: -1.0 });
    }
    c
}

fn check_trim(t: &TrimDesc) -> Result<()> {
    if !(t.elevation.is_finite() && t.pitch.is_finite() && t.roll.is_finite()) {
        return Err(Error::value("elevation, pitch and roll must be finite"));
    }
    Ok(())
}

// ---------------------------------------------------------------- trim, higher order

// a point of an element of a patch mesh (record_eval)
fn point(m: &Mesh, p: i32, u: f64, v: f64) -> Vec3 {
    let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    record_eval(m, p, u, v, &mut x, &mut xu, &mut xv);
    x
}

struct TrimCtx<'a> {
    m: &'a Mesh,
    r: [f64; 9],
    z0: f64,
}

const WHOLE: [f64; 4] = [-1.0, 1.0, -1.0, 1.0];

// the source patch i reflected and moved into the trimmed position, u reversed where that keeps
// x_u x x_v into the body (as the engine), on the range (u0, u1, v0, v1) of the source
// (the whole square: WHOLE)
fn base(t: &TrimCtx, i: i32, sx: f64, sy: f64, u: f64, v: f64, rng: &[f64; 4]) -> Vec3 {
    let su = if (sx * sy < 0.0) != (t.m.geometry == PatchKind::Flat) { -1.0 } else { 1.0 };
    let us = rng[0] + (su * u + 1.0) * 0.5 * (rng[1] - rng[0]);
    let vs = rng[2] + (v + 1.0) * 0.5 * (rng[3] - rng[2]);
    let x = point(t.m, i, us, vs);
    let mut y = rotated(&t.r, &v3(sx * x[0], sy * x[1], x[2]));
    y[2] += t.z0;
    y
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fiber {
    W,
    D,
    C1,
    C2,
    None,
}

// the class of a fibre from its z samples: wholly submerged, wholly dry, cut once from its
// first end, from its last end, or none (crossing twice)
fn fibre_class(row: &[f64], tol: f64) -> Fiber {
    let mut anydry = false;
    let mut allwet_or_on = true;
    for &z in row {
        anydry = anydry || z > tol;
        allwet_or_on = allwet_or_on && z >= -tol;
    }
    if !anydry {
        return Fiber::W;
    }
    if allwet_or_on {
        return Fiber::D;
    }
    let cut = |reverse: bool| {
        let n = row.len();
        let dry = |k: usize| row[if reverse { n - 1 - k } else { k }] > tol;
        if dry(0) {
            return false;
        }
        let mut k = 0;
        while k < n && !dry(k) {
            k += 1;
        }
        while k < n {
            if !dry(k) {
                return false;
            }
            k += 1;
        }
        true
    };
    if cut(false) {
        return Fiber::C1;
    }
    if cut(true) {
        return Fiber::C2;
    }
    Fiber::None
}

#[derive(Clone, Copy, Debug)]
struct Piece {
    cut: i32,
    rng: [f64; 4],
}

// the elements of a cut patch: for the first fibre direction (lines of constant u, then of
// constant v) in which every fibre has a class, the runs of fibres of one class, their boundaries
// by bisection on the class of the fibre (129 samples); dry runs left out
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn pieces(t: &TrimCtx, i: i32, sx: f64, sy: f64, zz: &[Vec<f64>], tol: f64, out: &mut Vec<Piece>) -> bool {
    let su = if (sx * sy < 0.0) != (t.m.geometry == PatchKind::Flat) { -1.0 } else { 1.0 };
    const G: usize = 33;
    let mut g = vec![0.0f64; G];
    let mut s = vec![0.0f64; 129];
    for k in 0..G {
        g[k] = -1.0 + 2.0 * k as f64 / (G - 1) as f64;
    }
    for k in 0..129 {
        s[k] = -1.0 + 2.0 * k as f64 / 128.0;
    }
    for along_v in [true, false] {
        let mut cls = [Fiber::W; G];
        let mut ok = true;
        let mut a = 0;
        while a < G && ok {
            let mut row = vec![0.0f64; G];
            for b in 0..G {
                row[b] = if along_v { zz[a][b] } else { zz[b][a] };
            }
            cls[a] = fibre_class(&row, tol);
            ok = cls[a] != Fiber::None;
            a += 1;
        }
        if !ok {
            continue;
        }
        let fibre = |a: f64| {
            let mut row = vec![0.0f64; s.len()];
            for k in 0..s.len() {
                row[k] =
                    (if along_v { base(t, i, sx, sy, a, s[k], &WHOLE) } else { base(t, i, sx, sy, s[k], a, &WHOLE) })
                        [2];
            }
            fibre_class(&row, tol)
        };
        let mut bounds = vec![-1.0f64];
        let mut kinds = vec![cls[0]];
        for k in 0..G - 1 {
            if cls[k + 1] == cls[k] {
                continue;
            }
            let mut lo = g[k];
            let mut hi = g[k + 1];
            for _ in 0..50 {
                let mid = 0.5 * (lo + hi);
                if fibre(mid) == cls[k] {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            bounds.push(0.5 * (lo + hi));
            kinds.push(cls[k + 1]);
        }
        bounds.push(1.0);
        out.clear();
        for k in 0..kinds.len() {
            let a0 = bounds[k];
            let a1 = bounds[k + 1];
            if kinds[k] == Fiber::D || a1 - a0 <= 1e-9 {
                continue;
            }
            let cut = if kinds[k] == Fiber::W {
                0
            } else if kinds[k] == Fiber::C1 {
                if along_v { 1 } else { 3 }
            } else if along_v {
                2
            } else {
                4
            };
            let rng = if along_v {
                // the element's u range in the source parameters (u reversed by su)
                [if su > 0.0 { a0 } else { -a1 }, if su > 0.0 { a1 } else { -a0 }, -1.0, 1.0]
            } else {
                [-1.0, 1.0, a0, a1]
            };
            out.push(Piece { cut, rng });
        }
        return true;
    }
    false
}

// ---------------------------------------------------------------- waterline

#[derive(Clone, Copy, Debug)]
struct CircleSide {
    patch: i32,
    radius: f64,
    theta0: f64,
    theta1: f64,
}

// the waterline sides of revolved analytic patches with a circular section (line and arc with
// a = b): (patch, radius, theta0, theta1) at a v side in the free surface
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn circle_sides(m: &Mesh, zw: f64, tol: f64) -> Vec<CircleSide> {
    let mut out = Vec::new();
    if !m.patch || m.geometry != PatchKind::Analytic || m.trimmed() {
        return out;
    }
    for i in 0..m.count {
        let p = &m.analytic[i as usize];
        if (p.kind != AnalyticKind::Line && p.kind != AnalyticKind::Arc) || (p.a - p.b).abs() > tol * 1.0f64.max(p.a) {
            continue;
        }
        let mut ends = [[0.0f64; 2]; 2];
        if p.kind == AnalyticKind::Line {
            ends[0][0] = p.p[0];
            ends[0][1] = p.p[1];
            ends[1][0] = p.p[2];
            ends[1][1] = p.p[3];
        } else {
            for e in 0..2 {
                let phi = p.p[1 + e];
                ends[e][0] = phi.cos();
                ends[e][1] = -p.p[0] * phi.sin();
            }
        }
        for e in &ends {
            if (e[1] - zw).abs() <= tol * 1.0f64.max(p.a) && p.a * e[0] > tol {
                out.push(CircleSide { patch: i, radius: p.a * e[0], theta0: p.theta0, theta1: p.theta1 });
            }
        }
    }
    out
}

/// The whole waterline of the stored part of a mesh in z = zw as segments: those of
/// waterline_segments and 16 chords of every exact circular side of a revolved analytic patch.
pub fn waterline_chords(m: &Mesh, zw: f64, tol: f64) -> Segments {
    let mut segs = waterline_segments(m, zw, tol);
    for c in circle_sides(m, zw, tol) {
        for k in 0..16 {
            let t0 = c.theta0 + (c.theta1 - c.theta0) * k as f64 / 16.0;
            let t1 = c.theta0 + (c.theta1 - c.theta0) * (k + 1) as f64 / 16.0;
            segs.push([
                v3(c.radius * t0.cos(), c.radius * t0.sin(), zw),
                v3(c.radius * t1.cos(), c.radius * t1.sin(), zw),
            ]);
        }
    }
    segs
}

/// The waterline of the stored part of a mesh in z = zw: panel edges (low order) or 16
/// chords of the patch sides there; elements wholly in z = zw and the exact circular sides of
/// revolved analytic patches left out.
pub fn waterline_segments(m: &Mesh, zw: f64, tol: f64) -> Segments {
    let t = tol * 1.0f64.max(m.reference_length);
    let mut segs: Segments = Vec::new();
    let in_fs = |pts: &[Vec3]| {
        for x in pts {
            if (x[2] - zw).abs() > t {
                return false;
            }
        }
        true
    };
    if !m.patch {
        for p in &m.panels {
            let q = &p.raw;
            if in_fs(q) {
                continue;
            }
            for e in 0..4 {
                let a = q[e];
                let b = q[(e + 1) % 4];
                if (a[2] - zw).abs() <= t && (b[2] - zw).abs() <= t && norm(b - a) > t {
                    segs.push([a, b]);
                }
            }
        }
    } else {
        let mut exact = vec![false; m.count as usize];
        for c in circle_sides(m, zw, tol) {
            exact[c.patch as usize] = true;
        }
        for i in 0..m.count {
            let mut sides: [Vec<Vec3>; 4] = Default::default();
            for k in 0..=16 {
                let s = -1.0 + k as f64 / 8.0;
                sides[0].push(point(m, i, s, -1.0));
                sides[1].push(point(m, i, s, 1.0));
                sides[2].push(point(m, i, -1.0, s));
                sides[3].push(point(m, i, 1.0, s));
            }
            let mut all = true;
            for sd in &sides {
                all = all && in_fs(sd);
            }
            if exact[i as usize] || all {
                continue;
            }
            for sd in &sides {
                if in_fs(sd) {
                    for k in 0..sd.len() - 1 {
                        if norm(sd[k + 1] - sd[k]) > t {
                            segs.push([sd[k], sd[k + 1]]);
                        }
                    }
                }
            }
        }
    }
    for s in &mut segs {
        s[0][2] = zw;
        s[1][2] = zw;
    }
    segs
}

/// The submerged part of a low-order mesh in the trimmed position.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn trim_low_order(mesh: &Mesh, trim: &TrimDesc) -> Result<LoTrim> {
    check_trim(trim)?;
    if mesh.patch {
        return Err(Error::value("the low-order trim takes a low-order mesh"));
    }
    let mut out = LoTrim::default();
    let mut r = [0.0f64; 9];
    trim_rotation(trim.pitch, trim.roll, &mut r);
    let copies = trim_copies(mesh.isx, mesh.isy, trim.pitch, trim.roll, &mut out.symmetry);
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in &mesh.panels {
        for x in &p.raw {
            for c in 0..3 {
                lo[c] = lo[c].min(x[c]);
                hi[c] = hi[c].max(x[c]);
            }
        }
    }
    let mut span: f64 = 0.0;
    for c in 0..3 {
        span = span.max(hi[c] - lo[c]);
    }
    if span == 0.0 {
        span = 1.0;
    }
    let tol = 1e-9 * span;
    for cp in &copies {
        let flip = cp.sx * cp.sy < 0.0;
        for k in 0..mesh.count as usize {
            let mut v = [Vec3::ZERO; 4];
            for i in 0..4 {
                let q = mesh.panels[k].raw[if flip { 3 - i } else { i }];
                v[i] = rotated(&r, &v3(cp.sx * q[0], cp.sy * q[1], q[2]));
                v[i][2] += trim.elevation;
            }
            let mut lp = vec![v[0]];
            for i in 1..4 {
                if norm(v[i] - v[i - 1]) > tol {
                    lp.push(v[i]);
                }
            }
            if lp.len() > 1 && norm(*lp.last().unwrap() - lp[0]) <= tol {
                lp.pop();
            }
            // a tank element: cut at its free surface
            let zc = match &trim.record_cut {
                Some(rc) if rc[k].is_finite() => rc[k],
                _ => 0.0,
            };
            for x in &mut lp {
                x[2] -= zc;
            }
            let mut part = clip(&lp, tol);
            for x in &mut part {
                x[2] += zc;
            }
            for a in quads(&part)? {
                // a sliver at the waterline (zero area by the panel rule, geometry.rs) is dropped
                let area = 0.5 * norm(cross(a[2] - a[0], a[3] - a[1]));
                let mut diam: f64 = 0.0;
                for i in 0..4 {
                    for j in 0..i {
                        diam = diam.max(norm(a[i] - a[j]));
                    }
                }
                if area > COINCIDENCE_TOL * diam * diam {
                    out.panels.push(a);
                    out.source.push(k as i32 + 1);
                    out.copy.push([cp.sx as i32, cp.sy as i32]);
                }
            }
        }
    }
    if out.panels.is_empty() {
        return Err(Error::value("the trimmed mesh has no submerged elements"));
    }
    out.plate = vec![-1; out.panels.len()];
    if !trim.plates.is_empty() {
        // the free-surface plates generated in the new waterline (theory §14), the waterline as
        // mesh_plate_triangles takes it from the trimmed mesh; each triangle a panel with its
        // normal down, into the water
        let segs = panel_waterline(&out.panels, 1e-6 * mesh.reference_length.max(1.0));
        let ltol = length_tol(mesh.reference_length);
        for (k, p) in trim.plates.iter().enumerate() {
            for t in plate_triangles(&segs, out.symmetry, ltol, *p, true)? {
                out.panels.push([t[0], t[2], t[1], t[1]]);
                out.source.push(0);
                out.copy.push([1, 1]);
                out.plate.push(k as i32);
            }
        }
    }
    Ok(out)
}

/// The trimmed elements of an untrimmed patch mesh, with the interior free surface
/// when asked.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn trim_higher_order(desc: &PatchMeshDesc, trim: &TrimDesc) -> Result<HoTrim> {
    check_trim(trim)?;
    if desc.trim_count > 0 {
        return Err(Error::value("the trim takes an untrimmed patch mesh"));
    }
    let mesh = build_patch_mesh(desc)?;
    let mut out = HoTrim::default();
    let mut t = TrimCtx { m: &mesh, r: [0.0; 9], z0: trim.elevation };
    trim_rotation(trim.pitch, trim.roll, &mut t.r);
    let copies = trim_copies(mesh.isx, mesh.isy, trim.pitch, trim.roll, &mut out.symmetry);
    const G: usize = 33;
    let mut g = vec![0.0f64; G];
    for k in 0..G {
        g[k] = -1.0 + 2.0 * k as f64 / (G - 1) as f64;
    }
    for cp in &copies {
        for i in 0..mesh.count {
            let mut zz = vec![vec![0.0f64; G]; G];
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            let mut zmax = f64::NEG_INFINITY;
            let mut zmin = f64::INFINITY;
            for a in 0..G {
                for b in 0..G {
                    let x = base(&t, i, cp.sx, cp.sy, g[a], g[b], &WHOLE);
                    zz[a][b] = x[2];
                    zmax = zmax.max(x[2]);
                    zmin = zmin.min(x[2]);
                    for c in 0..3 {
                        lo[c] = lo[c].min(x[c]);
                        hi[c] = hi[c].max(x[c]);
                    }
                }
            }
            let mut span: f64 = 0.0;
            for c in 0..3 {
                span = span.max(hi[c] - lo[c]);
            }
            if span == 0.0 {
                span = 1.0;
            }
            let tol = 1e-9 * span.max(mesh.reference_length);
            let mut ps: Vec<Piece> = Vec::new();
            if zmax <= tol {
                ps.push(Piece { cut: 0, rng: [-1.0, 1.0, -1.0, 1.0] });
            } else if zmin >= -tol {
                continue; // dry
            } else if !pieces(&t, i, cp.sx, cp.sy, &zz, tol, &mut ps) {
                return Err(Error::value(format!(
                    "trim: patch {} is cut by the waterline where neither its lines of constant u nor of \
                         constant v cross it at most once",
                    i + 1
                ))
                .at(at_element(i + 1)));
            }
            let q = mesh.discretization[i as usize];
            for p in &ps {
                // the spans of the source in proportion to the part of its square
                let mut rec = PatchTrim {
                    patch: i,
                    sx: cp.sx as i32,
                    sy: cp.sy as i32,
                    cut: p.cut,
                    range: p.rng,
                    discretization: q,
                };
                rec.discretization.nu = 1.max((q.nu as f64 * (p.rng[1] - p.rng[0]) / 2.0 - 1e-9).ceil() as i32);
                rec.discretization.nv = 1.max((q.nv as f64 * (p.rng[3] - p.rng[2]) / 2.0 - 1e-9).ceil() as i32);
                out.records.push(rec);
                out.source.push(i + 1);
            }
        }
    }
    if out.records.is_empty() {
        return Err(Error::value("the trimmed mesh has no submerged elements"));
    }
    if trim.interior_surface == 0 && trim.plates.is_empty() {
        return Ok(out);
    }

    // the interior free surface (theory §14): the waterline of the trimmed elements traced (16
    // chords per element side in z = 0) and closed along the planes of the kept symmetry into
    // loops. One loop star-shaped about the mean of the waterline points (with their mirror copies;
    // on the planes of the kept symmetry) is fanned from that centre, to every waterline chord;
    // every other waterline (several loops, holes such as moonpools, a loop that is not
    // star-shaped) is triangulated with its holes (polygon.rs). Each triangle is a bilinear patch
    // with a degenerate edge at one corner, normals up, moved back into the source coordinates, to
    // be appended to the source patches. The generated free-surface plates follow, each the
    // triangulated water inside the loop around its point (plate_triangles), normals down
    let mut td = desc.clone();
    td.trim = out.records.clone();
    td.trim_count = out.records.len() as i32;
    td.trim_elevation = trim.elevation;
    td.trim_pitch = trim.pitch;
    td.trim_roll = trim.roll;
    let trimmed = build_patch_mesh(&td)?; // (its symmetry: the flags of the source the trim keeps)
    let segs = waterline_segments(&trimmed, 0.0, 1e-6);
    if segs.is_empty() && trim.interior_surface != 0 {
        return Err(Error::value("trim: the trimmed body has no waterline for the interior surface"));
    }
    let tr = t.r;
    let back = |x: &Vec3| {
        // body coordinates into the source (GDF) coordinates: R^T (x - z0)
        let y = v3(x[0], x[1], x[2] - trim.elevation);
        v3(
            tr[0] * y[0] + tr[3] * y[1] + tr[6] * y[2],
            tr[1] * y[0] + tr[4] * y[1] + tr[7] * y[2],
            tr[2] * y[0] + tr[5] * y[1] + tr[8] * y[2],
        )
    };
    let seg = longest_chord(&segs);
    // the triangles (corner c, then a, b counter-clockwise about c seen from above) with their
    // spans along ab (nu) and from c to ab (nv), and the plate they belong to (-1: the interior
    // surface)
    let mut tris: Vec<([Vec3; 3], i32, i32, i32)> = Vec::new();
    if trim.interior_surface != 0 {
        interior_triangles(&segs, out.symmetry, length_tol(trimmed.reference_length), seg, &mut tris)?;
    }
    for (k, p) in trim.plates.iter().enumerate() {
        for q in plate_triangles(&segs, out.symmetry, length_tol(trimmed.reference_length), *p, false)? {
            let (c, nu, nv) = cornered(q, seg);
            tris.push((c, nu, nv, k as i32));
        }
    }
    let quads = triangle_patches(tris, mesh.geometry == PatchKind::Flat, seg);
    let d0 = mesh.discretization[0];
    let n0 = mesh.count;
    for (mut quad, nu, nv, plate) in quads {
        for x in &mut quad {
            *x = back(x);
        }
        out.interior.push(quad);
        out.interior_plate.push(plate);
        let mut dq = d0;
        dq.nu = nu;
        dq.nv = nv;
        out.interior_disc.push(dq);
        let rec = PatchTrim {
            patch: n0 + out.interior.len() as i32 - 1,
            sx: 1,
            sy: 1,
            cut: 0,
            range: WHOLE,
            discretization: dq,
        };
        out.records.push(rec);
        out.source.push(rec.patch + 1);
    }
    Ok(out)
}

// the longest waterline chord
fn longest_chord(segs: &Segments) -> f64 {
    segs.iter().map(|s| norm(s[1] - s[0])).fold(0.0, f64::max)
}

// the spans of about two longest chords (seg) each in a length l, least..8
fn spans(l: f64, seg: f64, least: i32) -> i32 {
    least.max(8.min((l / seg / 2.0).round() as i32))
}

// a triangle counter-clockwise from above as (corner, a, b) from the corner opposite its longest
// edge (the degenerate edge there), with its spans: 1..8 along ab, 2..8 from the corner
fn cornered(p: [Vec3; 3], seg: f64) -> ([Vec3; 3], i32, i32) {
    let k = (0..3)
        .max_by(|&i, &j| {
            let (li, lj) = (norm(p[(i + 2) % 3] - p[(i + 1) % 3]), norm(p[(j + 2) % 3] - p[(j + 1) % 3]));
            li.partial_cmp(&lj).unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("three corners");
    let (cc, a, b) = (p[k], p[(k + 1) % 3], p[(k + 2) % 3]);
    ([cc, a, b], spans(norm(b - a), seg, 1), spans(norm(a - cc).max(norm(b - cc)), seg, 2))
}

// The patches of the triangles (cornered, with the plate of each, -1 the interior surface): a
// triangle of an analytic or B-spline source as one bilinear patch with the corners (c, c, b, a)
// (-(x_u x x_v) points up; a plate's (c, c, a, b), down); a flat patch has four distinct corners,
// so a triangle of a flat source is three quadrilaterals from its corners to the midpoints of its
// sides and its centroid (counter-clockwise: x_u x x_v points up; a plate's clockwise), their spans
// by their sides.
type Cornered = ([Vec3; 3], i32, i32, i32);
fn triangle_patches(tris: Vec<Cornered>, flat: bool, seg: f64) -> Vec<([Vec3; 4], i32, i32, i32)> {
    let mut quads: Vec<([Vec3; 4], i32, i32, i32)> = Vec::new();
    for ([cc, a, b], nu, nv, plate) in tris {
        let down = plate >= 0;
        if !flat {
            quads.push((if down { [cc, cc, a, b] } else { [cc, cc, b, a] }, nu, nv, plate));
            continue;
        }
        let g = (1.0 / 3.0) * (cc + a + b);
        let p = [cc, a, b];
        for k in 0..3 {
            let (x, y, z) = (p[k], p[(k + 1) % 3], p[(k + 2) % 3]);
            let q = if down { [x, 0.5 * (x + z), g, 0.5 * (x + y)] } else { [x, 0.5 * (x + y), g, 0.5 * (x + z)] };
            let lu = norm(q[1] - q[0]).max(norm(q[2] - q[3]));
            let lv = norm(q[3] - q[0]).max(norm(q[2] - q[1]));
            quads.push((q, spans(lu, seg, 1), spans(lv, seg, 1), plate));
        }
    }
    quads
}

// The triangles of the interior surface of a trimmed patch mesh (trim_higher_order), cornered,
// with the plate -1: the fan of a star-shaped loop, else the triangulation of the regions.
fn interior_triangles(
    segs: &Segments,
    symmetry: [i32; 2],
    ltol: f64,
    seg: f64,
    tris: &mut Vec<Cornered>,
) -> Result<()> {
    let mut full: Vec<Vec3> = Vec::new();
    for s in segs {
        for x in s {
            full.push(*x);
            if symmetry[0] != 0 {
                full.push(v3(-x[0], x[1], x[2]));
            }
            if symmetry[1] != 0 {
                full.push(v3(x[0], -x[1], x[2]));
            }
            if symmetry[0] != 0 && symmetry[1] != 0 {
                full.push(v3(-x[0], -x[1], x[2]));
            }
        }
    }
    let mut c = v3(0.0, 0.0, 0.0);
    for x in &full {
        c += *x;
    }
    c = (1.0 / full.len() as f64) * c;
    c[2] = 0.0;
    if symmetry[0] != 0 {
        c[0] = 0.0;
    }
    if symmetry[1] != 0 {
        c[1] = 0.0;
    }
    let wl = waterline_loops(segs, symmetry, ltol)?;
    if wl.star_shaped(&c) {
        let mut radius: f64 = 1e-9;
        for s in segs {
            for x in s {
                radius = radius.max((x[0] - c[0]).hypot(x[1] - c[1]));
            }
        }
        // the radial spans: about two longest chords each, 2..8
        let nv = 2.max(8.min((radius / seg / 2.0).round() as i32));
        for s in segs {
            let (mut a, mut b) = (s[0], s[1]);
            if (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]) < 0.0 {
                std::mem::swap(&mut a, &mut b);
            }
            tris.push(([c, a, b], 1, nv, -1));
        }
    } else {
        // straight runs of chords merged (their triangles would have no area)
        let loops = wl.loops.iter().map(|l| polygon::merge_collinear(l, wl.tol)).collect();
        for r in polygon::regions(loops) {
            for q in polygon::triangulate(&r, wl.tol)? {
                // back to the stored side (a reflection of odd order reverses the orientation)
                let mut p: [Vec3; 3] = [0, 1, 2].map(|k| v3(wl.sx * q[k][0], wl.sy * q[k][1], 0.0));
                if wl.sx * wl.sy < 0.0 {
                    p.swap(1, 2);
                }
                let (t, nu, nv) = cornered(p, seg);
                tris.push((t, nu, nv, -1));
            }
        }
    }
    Ok(())
}

// the closed loops of a trimmed waterline in the stored part, mapped to x >= 0 (a kept symmetry
// about x = 0) and y >= 0 (about y = 0) by (sx, sy)
struct WaterLoops {
    loops: Vec<Vec<P2>>,
    sx: f64,
    sy: f64,
    sym: [i32; 2],
    tol: f64,
}

impl WaterLoops {
    // on a plane of the kept symmetry (mapped): y = 0, x = 0
    fn on_axis(&self, p: P2) -> bool {
        (self.sym[1] != 0 && p[1].abs() <= self.tol) || (self.sym[0] != 0 && p[0].abs() <= self.tol)
    }

    // one loop, every waterline chord of it seen from the centre c (body coordinates) turning
    // counter-clockwise by more than a sine of 1e-9 (the edges closing it on the planes of the
    // symmetry pass through c): the loop is star-shaped about c, its fan covers it once
    fn star_shaped(&self, c: &Vec3) -> bool {
        if self.loops.len() != 1 {
            return false;
        }
        let mut l = self.loops[0].clone();
        if polygon::signed_area(&l) < 0.0 {
            l.reverse();
        }
        let cm = [self.sx * c[0], self.sy * c[1]];
        let n = l.len();
        (0..n).all(|k| {
            let (p, q) = (l[k], l[(k + 1) % n]);
            let far = |x: P2| (x[0] - cm[0]).hypot(x[1] - cm[1]);
            (self.on_axis(p) && self.on_axis(q)) || polygon::orient(cm, p, q) > 1e-9 * far(p) * far(q)
        })
    }
}

// The waterline chords of the stored part joined into closed loops (mapped to the stored side,
// WaterLoops): the chains that end on the planes of the kept symmetry closed along them. Ordered
// along the boundary of the stored part (y = 0 from x = +inf to 0, then x = 0 to y = +inf), the
// ends of the chains alternate between open water and the waterplane, which lies between the
// first and the second end, the third and the fourth, and so on; such a pair is joined along the
// boundary (through the corner x = y = 0 where it passes there). Refused: a chain ending off those
// planes (an open waterline) and loops that cross or touch.
fn waterline_loops(segs: &Segments, sym: [i32; 2], tol: f64) -> Result<WaterLoops> {
    boundary_loops(segs, sym, tol, false)
}

// The loops of waterline_loops (water false: the waterplane's), or with water true those of the
// water the waterline encloses: the ends of the chains paired the other way, the second with the
// third, the fourth with the fifth, ...; the first and the last end bound the open sea, which
// reaches infinity along the boundary and is no loop. The closed chains are loops of both.
fn boundary_loops(segs: &Segments, sym: [i32; 2], tol: f64, water: bool) -> Result<WaterLoops> {
    let (mut mx, mut my) = (0.0, 0.0);
    for s in segs {
        for x in s {
            mx += x[0];
            my += x[1];
        }
    }
    let sx = if sym[0] != 0 && mx < 0.0 { -1.0 } else { 1.0 };
    let sy = if sym[1] != 0 && my < 0.0 { -1.0 } else { 1.0 };
    let mapped: Segments =
        segs.iter().map(|s| [v3(sx * s[0][0], sy * s[0][1], 0.0), v3(sx * s[1][0], sy * s[1][1], 0.0)]).collect();
    let not_closed = || {
        Error::value(
            "trim: the waterline of the trimmed body does not close (it ends off the planes of its symmetry): no \
             interior surface",
        )
    };
    let mut loops: Vec<Vec<P2>> = Vec::new();
    let mut open: Vec<Vec<P2>> = Vec::new();
    for ch in chains(&mapped, tol) {
        let mut pts: Vec<P2> = vec![ch.chords[0][0]];
        pts.extend(ch.chords.iter().map(|c| c[1]));
        if ch.closed {
            pts.pop();
            loops.push(pts);
        } else {
            open.push(pts);
        }
    }
    // the place of a chain end along the boundary of the stored part
    let place = |p: P2| -> Option<f64> {
        if sym[1] != 0 && p[1].abs() <= tol && (sym[0] == 0 || p[0] >= -tol) {
            Some(if sym[0] != 0 { -p[0] } else { p[0] })
        } else if sym[0] != 0 && p[0].abs() <= tol && (sym[1] == 0 || p[1] >= -tol) {
            Some(p[1])
        } else {
            None
        }
    };
    let mut ends: Vec<(f64, usize, bool)> = Vec::new(); // (place, chain, its last point)
    for (k, c) in open.iter().enumerate() {
        let (Some(a), Some(b)) = (place(c[0]), place(c[c.len() - 1])) else {
            return Err(not_closed());
        };
        ends.push((a, k, false));
        ends.push((b, k, true));
    }
    ends.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let at = |k: usize, last: bool| ends.iter().position(|e| e.1 == k && e.2 == last).expect("every end placed");
    let n_ends = ends.len();
    // the partner of an end along the boundary: (0, 1), (2, 3), ... for the waterplane; (1, 2),
    // (3, 4), ... for the water, the first and the last end towards infinity (None)
    let partner = |i: usize| -> Option<usize> {
        if !water {
            Some(i ^ 1)
        } else if i == 0 || i + 1 == n_ends {
            None
        } else {
            Some(((i - 1) ^ 1) + 1)
        }
    };
    let mut used = vec![false; open.len()];
    if water && n_ends > 0 {
        // the open sea: the chains from the first end to the last, joined along the boundary
        let (mut k, mut last) = (ends[0].1, ends[0].2);
        loop {
            used[k] = true;
            let Some(j) = partner(at(k, !last)) else { break };
            (k, last) = (ends[j].1, ends[j].2);
            if used[k] {
                break;
            }
        }
    }
    for k0 in 0..open.len() {
        if used[k0] {
            continue;
        }
        let mut pts: Vec<P2> = Vec::new();
        let (mut k, mut forward) = (k0, true);
        loop {
            used[k] = true;
            if forward {
                pts.extend_from_slice(&open[k]);
            } else {
                pts.extend(open[k].iter().rev());
            }
            let i = at(k, forward);
            let Some(j) = partner(i) else {
                return Err(Error::internal(
                    "trim: a loop of the water enclosed by the waterline reaches the open sea",
                ));
            };
            if sym[0] != 0
                && sym[1] != 0
                && (ends[i].0 < 0.0) != (ends[j].0 < 0.0)
                && ends[i].0 != 0.0
                && ends[j].0 != 0.0
            {
                pts.push([0.0, 0.0]); // the corner of the stored quadrant
            }
            let (kn, last) = (ends[j].1, ends[j].2);
            if used[kn] {
                break; // back at the start of k0
            }
            (k, forward) = (kn, !last);
        }
        loops.push(pts);
    }
    // repeated points dropped (two ends at one place)
    for l in loops.iter_mut() {
        let mut c: Vec<P2> = Vec::new();
        for &p in l.iter() {
            if c.last().is_none_or(|q| (p[0] - q[0]).hypot(p[1] - q[1]) > tol) {
                c.push(p);
            }
        }
        while c.len() > 1 && (c[0][0] - c[c.len() - 1][0]).hypot(c[0][1] - c[c.len() - 1][1]) <= tol {
            c.pop();
        }
        *l = c;
    }
    if loops.iter().any(|l| l.len() < 3) || polygon::loops_meet(&loops, tol) {
        return Err(Error::value(
            "trim: the waterline of the trimmed body crosses or touches itself (contours of the waterplane meet): no \
             interior surface",
        ));
    }
    Ok(WaterLoops { loops, sx, sy, sym, tol })
}

// ---------------------------------------------------------------- generated free-surface plates

/// The most triangles of a generated plate (as the interior surface's cap, interior.rs: a dense
/// system of more unknowns is impractical).
const MAX_PLATE_TRIANGLES: usize = 200000;

/// The waterline of panels (body coordinates) in z = 0: the panel edges with both ends within t
/// of the plane, of the panels not wholly in it (the rule of waterline_segments for a low-order
/// mesh).
fn panel_waterline(panels: &[[Vec3; 4]], t: f64) -> Segments {
    let mut segs: Segments = Vec::new();
    for q in panels {
        if q.iter().all(|x| x[2].abs() <= t) {
            continue;
        }
        for e in 0..4 {
            let (a, b) = (q[e], q[(e + 1) % 4]);
            if a[2].abs() <= t && b[2].abs() <= t && norm(b - a) > t {
                segs.push([v3(a[0], a[1], 0.0), v3(b[0], b[1], 0.0)]);
            }
        }
    }
    segs
}

/// The region of a generated free-surface plate (theory §14): the water of z = 0 inside the
/// waterline loop around the point `inside` (body coordinates), less the waterplanes of the hulls
/// within it, in the stored part of the kept symmetry (a point beyond a plane of it is taken by its
/// mirror image: the plate covers the mirror images of its region too), mapped as WaterLoops: the
/// region and the map back (sx, sy).
fn plate_region(segs: &Segments, sym: [i32; 2], tol: f64, inside: [f64; 2]) -> Result<(Region, f64, f64)> {
    let point = format!("({}, {})", inside[0], inside[1]);
    if segs.is_empty() {
        return Err(Error::value(format!(
            "the generated plate at {point}: the trimmed body has no waterline (no loop to cover)"
        )));
    }
    let plane = boundary_loops(segs, sym, tol, false)?;
    let water = boundary_loops(segs, sym, tol, true)?;
    let (sx, sy) = (plane.sx, plane.sy);
    // the point in the stored part, off its planes (the corner of a quadrant: the centre of a
    // moonpool on both planes) by more than the tolerance of the loops' ends on them, so that the
    // crossing number sees it inside the loop that a plane closes
    let mut extent: f64 = 0.0;
    for s in segs {
        for x in s {
            extent = extent.max(x[0].abs()).max(x[1].abs());
        }
    }
    let off = (1e-6 * extent).max(4.0 * tol);
    let mut p = [sx * inside[0], sy * inside[1]];
    for (k, &s) in [sym[0], sym[1]].iter().enumerate() {
        if s != 0 {
            p[k] = p[k].abs().max(off);
        }
    }
    // on the waterline: within tol of a chord; the edges that close a loop along a plane of the
    // kept symmetry are no waterline (a point on the plane lies in the water of the whole loop)
    let on_plane = |a: P2, b: P2| {
        (sym[1] != 0 && a[1].abs() <= tol && b[1].abs() <= tol)
            || (sym[0] != 0 && a[0].abs() <= tol && b[0].abs() <= tol)
    };
    let on_line = water.loops.iter().chain(&plane.loops).any(|l| {
        (0..l.len()).any(|k| {
            let (a, b) = (l[k], l[(k + 1) % l.len()]);
            !on_plane(a, b) && polygon::segment_distance(p, a, b) <= tol
        })
    });
    if on_line {
        return Err(Error::value(format!(
            "the generated plate at {point}: the point lies on the waterline (give a point inside the loop)"
        )));
    }
    if plane.loops.iter().filter(|l| polygon::inside(l, p)).count() % 2 == 1 {
        return Err(Error::value(format!(
            "the generated plate at {point}: the point lies in the waterplane of the hull, not in the water inside a \
             waterline loop (a moonpool, the water column of a tube)"
        )));
    }
    let area = |l: &Vec<P2>| polygon::signed_area(l).abs();
    let outer = (0..water.loops.len())
        .filter(|&k| polygon::inside(&water.loops[k], p))
        .min_by(|&a, &b| area(&water.loops[a]).total_cmp(&area(&water.loops[b])));
    let Some(o) = outer else {
        return Err(Error::value(format!(
            "the generated plate at {point}: the point lies in the open sea, outside every waterline loop of the \
             trimmed body"
        )));
    };
    // the loops inside it: the outermost are the waterlines of the hulls within (islands)
    let within: Vec<usize> =
        (0..water.loops.len()).filter(|&k| k != o && polygon::inside(&water.loops[o], water.loops[k][0])).collect();
    let mut loops = vec![polygon::merge_collinear(&water.loops[o], tol)];
    for &k in &within {
        if !within.iter().any(|&j| j != k && polygon::inside(&water.loops[j], water.loops[k][0])) {
            loops.push(polygon::merge_collinear(&water.loops[k], tol));
        }
    }
    let r = polygon::regions(loops).into_iter().next().expect("the loop around the point bounds a region");
    Ok((r, sx, sy))
}

/// The triangles of a generated free-surface plate (theory §14) on the waterline `segs` of a
/// trimmed body (body coordinates, z = 0, the stored part of the kept symmetry `sym`), in the
/// water loop around the point `inside`: refined (the low order: Delaunay refinement to edges of
/// at most twice the mean waterline segment and angles of at least polygon::REFINE_MIN_ANGLE, as
/// the `waterline` interior surface) or not (the higher order: the constrained Delaunay
/// triangulation of the loop's vertices, as the interior surface of a trimmed patch mesh). The
/// triangles in body coordinates of the stored part, counter-clockwise seen from above; their
/// boundary edges lie on the waterline chords (new vertices of the refinement on their midpoints).
pub fn plate_triangles(
    segs: &Segments,
    sym: [i32; 2],
    tol: f64,
    inside: [f64; 2],
    refine: bool,
) -> Result<Vec<[Vec3; 3]>> {
    let (r, sx, sy) = plate_region(segs, sym, tol, inside)?;
    let tris = if refine {
        let mean = segs.iter().map(|s| norm(s[1] - s[0])).sum::<f64>() / segs.len() as f64;
        let t = polygon::triangulate_refined(&r, tol, 2.0 * mean, MAX_PLATE_TRIANGLES)?;
        if t.len() > MAX_PLATE_TRIANGLES {
            return Err(Error::value(format!(
                "the generated plate at ({}, {}) needs more than {MAX_PLATE_TRIANGLES} panels of twice the mean \
                 waterline segment length",
                inside[0], inside[1]
            )));
        }
        t
    } else {
        polygon::triangulate(&r, tol)?
    };
    Ok(tris
        .iter()
        .map(|t| {
            let mut q: [Vec3; 3] = [0, 1, 2].map(|k| v3(sx * t[k][0], sy * t[k][1], 0.0));
            if sx * sy < 0.0 {
                q.swap(1, 2); // a reflection of odd order reverses the orientation
            }
            q
        })
        .collect())
}

/// The triangles of a generated free-surface plate on the waterline of a trimmed mesh (the mesh of
/// another body, whose loop a plate body of its own covers): `plate_triangles` on the waterline of
/// the mesh in z = 0 (the elements wholly in the plane, its generated surfaces, left out), its
/// symmetry flags and its length tolerance; refined for a low-order mesh.
pub fn mesh_plate_triangles(m: &Mesh, inside: [f64; 2]) -> Result<Vec<[Vec3; 3]>> {
    let segs = waterline_segments(m, 0.0, 1e-6);
    plate_triangles(&segs, [m.isx, m.isy], length_tol(m.reference_length), inside, !m.patch)
}

/// The flat patches (corners, nu, nv) of a generated free-surface plate on the waterline of a
/// trimmed patch mesh, as the trim of a flat patch mesh appends them (three quadrilaterals per
/// triangle of mesh_plate_triangles, normals down, their spans by the longest waterline chord).
pub fn mesh_plate_patches(m: &Mesh, inside: [f64; 2]) -> Result<Vec<([Vec3; 4], i32, i32)>> {
    let segs = waterline_segments(m, 0.0, 1e-6);
    let seg = longest_chord(&segs);
    let tris = plate_triangles(&segs, [m.isx, m.isy], length_tol(m.reference_length), inside, false)?;
    let tris: Vec<Cornered> = tris
        .into_iter()
        .map(|t| {
            let (c, nu, nv) = cornered(t, seg);
            (c, nu, nv, 0)
        })
        .collect();
    Ok(triangle_patches(tris, true, seg).into_iter().map(|(q, nu, nv, _)| (q, nu, nv)).collect())
}

// ---------------------------------------------------------------- automatic control surface

fn cross2(a: &P2, b: &P2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn sub(a: &P2, b: &P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}
fn add(a: &P2, b: &P2) -> P2 {
    [a[0] + b[0], a[1] + b[1]]
}
fn scale(s: f64, a: &P2) -> P2 {
    [s * a[0], s * a[1]]
}
fn dot2(a: &P2, b: &P2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn len2(a: &P2) -> f64 {
    a[0].hypot(a[1])
}
fn unit(a: &P2) -> P2 {
    scale(1.0 / len2(a), a)
}

fn analytic(kind: AnalyticKind, a: f64, b: f64, th0: f64, th1: f64, p: [f64; 4]) -> AnalyticPatch {
    AnalyticPatch { kind, profile: Profile::Linear, a, b, theta0: th0, theta1: th1, p, ..AnalyticPatch::default() }
}

#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn bilinear(a: &Vec3, b: &Vec3, c: &Vec3, d: &Vec3) -> AnalyticPatch {
    let mut q = analytic(AnalyticKind::Bilinear, 1.0, 1.0, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0]);
    let v = [a, b, c, d];
    for k in 0..4 {
        for i in 0..3 {
            q.vt[3 * k + i] = v[k][i];
        }
    }
    q
}

fn segment_arc(r: f64, th0: f64, th1: f64, zw: f64, a: &Vec3, b: &Vec3) -> AnalyticPatch {
    let mut q = analytic(AnalyticKind::SegmentArc, r, r, th0, th1, [zw, 0.0, 0.0, 0.0]);
    for i in 0..3 {
        q.vt[i] = a[i];
        q.vt[3 + i] = b[i];
    }
    q
}

// the side (v down from z0) and the bottom (v to the axis) of a circular cylinder
fn cylinder(r: f64, depth: f64, th0: f64, th1: f64, z0: f64, out: &mut Vec<AnalyticPatch>) {
    out.push(analytic(AnalyticKind::Line, r, r, th0, th1, [1.0, z0, 1.0, z0 - depth]));
    out.push(analytic(AnalyticKind::Line, r, r, th0, th1, [1.0, z0 - depth, 0.0, z0 - depth]));
}

// the walls (v down) along a polygonal outer boundary and the bottom as a fan from the axis
fn polygon_outer(poly: &[P2], depth: f64, zw: f64, out: &mut Vec<AnalyticPatch>) {
    let bottom = zw - depth;
    for k in 0..poly.len().saturating_sub(1) {
        let p = &poly[k];
        let q = &poly[k + 1];
        out.push(bilinear(&v3(p[0], p[1], zw), &v3(q[0], q[1], zw), &v3(q[0], q[1], bottom), &v3(p[0], p[1], bottom)));
        out.push(bilinear(
            &v3(p[0], p[1], bottom),
            &v3(q[0], q[1], bottom),
            &v3(0.0, 0.0, bottom),
            &v3(0.0, 0.0, bottom),
        ));
    }
}

// the outer boundary of the free surface: straight pieces (partitions, a polygon) and the circle
struct Boundary {
    radius: f64, // < 0: none
    segments: Vec<[P2; 2]>,
    corners: Vec<P2>,
}

impl Boundary {
    fn extend(&self, p: &P2, q: &P2) -> P2 {
        // the ray from p through q continued to the circle (q itself when on it or beyond)
        let d0 = sub(q, p);
        if len2(&d0) == 0.0 || len2(q) >= self.radius {
            return *q;
        }
        let d = unit(&d0);
        let bq = dot2(q, &d);
        let t = -bq + (bq * bq - (dot2(q, q) - self.radius * self.radius)).sqrt();
        add(q, &scale(t, &d))
    }

    fn new(r: f64, poly: Option<&Vec<P2>>, parts: &[Vec<P2>]) -> Boundary {
        let mut bnd = Boundary { radius: r, segments: Vec::new(), corners: Vec::new() };
        let mut lines: Vec<Vec<P2>> = Vec::new();
        if let Some(p) = poly {
            lines.push(p.clone());
        }
        for p in parts {
            lines.push(p.clone());
        }
        for mut line in lines {
            if bnd.radius > 0.0 {
                // a partition ends on the outer boundary: extended to the circle (theory §14)
                let n = line.len();
                let first = bnd.extend(&line[1], &line[0]);
                let last = bnd.extend(&line[n - 2], &line[n - 1]);
                line[0] = first;
                line[n - 1] = last;
            }
            for k in 0..line.len() - 1 {
                if len2(&sub(&line[k + 1], &line[k])) > 0.0 {
                    bnd.segments.push([line[k], line[k + 1]]);
                }
            }
            bnd.corners.extend_from_slice(&line);
        }
        bnd
    }

    // the first point of the boundary on the ray q + t d, t > 0, and its piece (-1: the circle)
    #[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
    fn hit(&self, q: &P2, d: &P2, piece: &mut i32) -> Result<P2> {
        let mut best = f64::INFINITY;
        *piece = -2;
        for k in 0..self.segments.len() {
            let e = sub(&self.segments[k][1], &self.segments[k][0]);
            let den = cross2(d, &e);
            if den.abs() < 1e-14 {
                continue;
            }
            let w = sub(&self.segments[k][0], q);
            let t = cross2(&w, &e) / den;
            let s = cross2(&w, d) / den;
            if t > 1e-12 && (-1e-9..=1.0 + 1e-9).contains(&s) && t < best {
                best = t;
                *piece = k as i32;
            }
        }
        if self.radius > 0.0 {
            let bq = dot2(q, d);
            let disc = bq * bq - (dot2(q, q) - self.radius * self.radius);
            if disc >= 0.0 {
                let t = -bq + disc.sqrt();
                if t > 1e-12 && t < best {
                    best = t;
                    *piece = -1;
                }
            }
        }
        if *piece == -2 {
            return Err(Error::value(
                "automatic control surface: a ray from the waterline meets no boundary (the waterline must be \
                 enclosed by the outer boundary and the partitions)",
            ));
        }
        Ok(add(q, &scale(best, d)))
    }
}

#[derive(Clone, Debug, Default)]
struct Chain {
    chords: Vec<[P2; 2]>,
    closed: bool,
}

fn near2(a: &P2, b: &P2, tol: f64) -> bool {
    (a[0] - b[0]).abs() <= tol && (a[1] - b[1]).abs() <= tol
}

// the waterline chords joined at their common ends
fn chains(segs: &Segments, tol: f64) -> Vec<Chain> {
    let mut left: Vec<[P2; 2]> = Vec::new();
    for s in segs {
        left.push([[s[0][0], s[0][1]], [s[1][0], s[1][1]]]);
    }
    let mut out = Vec::new();
    while !left.is_empty() {
        let mut c = Chain::default();
        c.chords.push(left[0]);
        left.remove(0);
        let mut grown = true;
        while grown {
            grown = false;
            for k in 0..left.len() {
                let a = left[k][0];
                let b = left[k][1];
                let head = c.chords[0][0];
                let tail = c.chords[c.chords.len() - 1][1];
                if near2(&a, &tail, tol) {
                    c.chords.push([a, b]);
                } else if near2(&b, &tail, tol) {
                    c.chords.push([b, a]);
                } else if near2(&b, &head, tol) {
                    c.chords.insert(0, [a, b]);
                } else if near2(&a, &head, tol) {
                    c.chords.insert(0, [b, a]);
                } else {
                    continue;
                }
                left.remove(k);
                grown = true;
                break;
            }
        }
        c.closed = c.chords.len() > 2 && near2(&c.chords[0][0], &c.chords[c.chords.len() - 1][1], tol);
        out.push(c);
    }
    out
}

// the free surface between the waterline chords and the boundary
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn free_surface_general(
    segs: &Segments,
    sym: &[i32; 2],
    zw: f64,
    bnd: &Boundary,
    tol: f64,
    out: &mut Vec<AnalyticPatch>,
) -> Result<()> {
    for mut ch in chains(segs, tol) {
        let mut c: P2 = [0.0, 0.0];
        let mut total = 0.0;
        for s in &ch.chords {
            let l = len2(&sub(&s[1], &s[0]));
            c = add(&c, &scale(0.5 * l, &add(&s[0], &s[1])));
            total += l;
        }
        c = scale(1.0 / total, &c);
        if !ch.closed {
            for p in [ch.chords[0][0], ch.chords[ch.chords.len() - 1][1]] {
                if sym[1] != 0 && p[1].abs() <= tol {
                    c[1] = 0.0;
                }
                if sym[0] != 0 && p[0].abs() <= tol {
                    c[0] = 0.0;
                }
            }
        }
        if cross2(&sub(&ch.chords[0][0], &c), &sub(&ch.chords[0][1], &c)) < 0.0 {
            // counter-clockwise about c
            let r: Vec<[P2; 2]> = ch.chords.iter().rev().map(|s| [s[1], s[0]]).collect();
            ch.chords = r;
        }
        let mut corners: Vec<P2> = Vec::new();
        for k in &bnd.corners {
            if len2(&sub(k, &c)) > tol {
                corners.push(*k);
            }
        }
        for s in &ch.chords {
            let a = s[0];
            let b = s[1];
            let span = cross2(&sub(&a, &c), &sub(&b, &c)).atan2(dot2(&sub(&a, &c), &sub(&b, &c)));
            let mut cuts: Vec<(f64, P2)> = Vec::new();
            for k in &corners {
                let dk = cross2(&sub(&a, &c), &sub(k, &c)).atan2(dot2(&sub(&a, &c), &sub(k, &c)));
                if dk > 1e-9 && dk < span - 1e-9 {
                    // the point of the chord on the ray from c to the corner
                    let e = sub(&b, &a);
                    let d = unit(&sub(k, &c));
                    let sp = cross2(&sub(&c, &a), &d) / cross2(&e, &d);
                    cuts.push((dk, add(&a, &scale(sp, &e))));
                }
            }
            cuts.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut pts = vec![a];
            for q in &cuts {
                pts.push(q.1);
            }
            pts.push(b);
            for k in 0..pts.len() - 1 {
                let p = pts[k];
                let q = pts[k + 1];
                let m = scale(0.5, &add(&p, &q));
                let (mut kp, mut kq, mut km) = (0, 0, 0);
                let Pp = bnd.hit(&p, &unit(&sub(&p, &c)), &mut kp)?;
                let Qq = bnd.hit(&q, &unit(&sub(&q, &c)), &mut kq)?;
                bnd.hit(&m, &unit(&sub(&m, &c)), &mut km)?;
                let p3 = v3(p[0], p[1], zw);
                let q3 = v3(q[0], q[1], zw);
                let P3 = v3(Pp[0], Pp[1], zw);
                let Q3 = v3(Qq[0], Qq[1], zw);
                if km == -1 {
                    let t0 = Pp[1].atan2(Pp[0]);
                    let mut t1 = Qq[1].atan2(Qq[0]);
                    if t1 - t0 > PI {
                        t1 -= 2.0 * PI;
                    } else if t0 - t1 > PI {
                        t1 += 2.0 * PI;
                    }
                    out.push(segment_arc(bnd.radius, t0, t1, zw, &p3, &q3));
                } else {
                    out.push(bilinear(&p3, &q3, &Q3, &P3));
                }
            }
        }
    }
    Ok(())
}

// the free surface between the exact circular waterline sides and the boundary
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn free_surface_arcs(circles: &[CircleSide], zw: f64, bnd: &Boundary, out: &mut Vec<AnalyticPatch>) -> Result<()> {
    let mut corners: Vec<f64> = Vec::new();
    for k in &bnd.corners {
        if len2(k) > 0.0 {
            corners.push(k[1].atan2(k[0]));
        }
    }
    for &cs0 in circles {
        let mut cs = cs0;
        if cs.theta1 < cs.theta0 {
            std::mem::swap(&mut cs.theta0, &mut cs.theta1);
        }
        let mut ts = vec![cs.theta0];
        let mut cut: Vec<f64> = Vec::new();
        for &t in &corners {
            if t > cs.theta0 + 1e-9 && t < cs.theta1 - 1e-9 {
                cut.push(t);
            }
        }
        cut.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        ts.extend_from_slice(&cut);
        ts.push(cs.theta1);
        let rc = cs.radius;
        for k in 0..ts.len() - 1 {
            let t0 = ts[k];
            let t1 = ts[k + 1];
            let d0 = [t0.cos(), t0.sin()];
            let d1 = [t1.cos(), t1.sin()];
            let dm = [(0.5 * (t0 + t1)).cos(), (0.5 * (t0 + t1)).sin()];
            let (mut kp, mut kq, mut km) = (0, 0, 0);
            let Pp = bnd.hit(&scale(rc, &d0), &d0, &mut kp)?;
            let Qq = bnd.hit(&scale(rc, &d1), &d1, &mut kq)?;
            bnd.hit(&scale(rc, &dm), &dm, &mut km)?;
            if km == -1 {
                out.push(analytic(AnalyticKind::Ellipses, bnd.radius, bnd.radius, t0, t1, [rc, rc, zw, zw]));
            } else {
                // a segment-arc patch run backwards: the same normal
                out.push(segment_arc(rc, t1, t0, zw, &v3(Qq[0], Qq[1], zw), &v3(Pp[0], Pp[1], zw)));
            }
        }
    }
    Ok(())
}

// the free surface of one star-shaped waterline inside a circular outer surface
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
fn free_surface_circle(
    segs: &Segments,
    circles: &[CircleSide],
    r: f64,
    sym: &[i32; 2],
    th0: f64,
    th1: f64,
    zw: f64,
    out: &mut Vec<AnalyticPatch>,
) -> Result<()> {
    struct Row {
        ta: f64,
        tb: f64,
        circle: bool,
        rc: f64,
        a: Vec3,
        b: Vec3,
    }
    let mut rows: Vec<Row> = Vec::new();
    for &cs0 in circles {
        let mut cs = cs0;
        if cs.radius >= r {
            return Err(Error::value("automatic control surface: the waterline reaches the outer radius"));
        }
        if cs.theta1 < cs.theta0 {
            std::mem::swap(&mut cs.theta0, &mut cs.theta1);
        }
        rows.push(Row { ta: cs.theta0, tb: cs.theta1, circle: true, rc: cs.radius, a: Vec3::ZERO, b: Vec3::ZERO });
    }
    for s in segs {
        let mut a = s[0];
        let mut b = s[1];
        let mut ta = a[1].atan2(a[0]);
        let mut tb = b[1].atan2(b[0]);
        if sym[0] == 0 && sym[1] == 0 {
            ta = (ta + 2.0 * PI) % (2.0 * PI);
            tb = (tb + 2.0 * PI) % (2.0 * PI);
        }
        if a[0].hypot(a[1]).max(b[0].hypot(b[1])) >= r {
            return Err(Error::value("automatic control surface: the waterline reaches the outer radius"));
        }
        if (tb - ta).abs() <= 1e-9 {
            return Err(Error::value(
                "automatic control surface: a waterline segment points at the origin (the waterline must be \
                 star-shaped about the body origin)",
            ));
        }
        if (tb - ta).abs() > PI {
            tb += if tb < ta { 2.0 * PI } else { -2.0 * PI }; // across the cut of the full circle
        }
        if tb < ta {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut ta, &mut tb);
        }
        rows.push(Row { ta, tb, circle: false, rc: 0.0, a, b });
    }
    rows.sort_by(|x, y| x.ta.partial_cmp(&y.ta).unwrap_or(std::cmp::Ordering::Equal));
    // the rows cover the azimuths of the stored part once: their spans sum to the sector and no
    // two overlap (modulo 2 pi; an overlap and an equal gap would sum to the sector), to 1e-6 of it
    let gap = 1e-6 * (th1 - th0);
    let mut span = 0.0;
    for w in &rows {
        span += w.tb - w.ta;
    }
    let overlap = |p: &Row, q: &Row| {
        let d = (q.ta - p.ta + PI).rem_euclid(2.0 * PI) - PI; // q's start near p's
        let (a, b) = (p.ta + d, p.ta + d + (q.tb - q.ta));
        p.tb.min(b) - p.ta.max(a) > gap
    };
    let mut once = (span - (th1 - th0)).abs() <= gap;
    for i in 0..rows.len() {
        for j in i + 1..rows.len() {
            once = once && !overlap(&rows[i], &rows[j]);
        }
    }
    if !once {
        return Err(Error::value(
            "automatic control surface: the waterline segments do not cover the azimuths of the control surface \
             once (one star-shaped waterline about the origin)",
        ));
    }
    for w in &rows {
        if w.circle {
            out.push(analytic(AnalyticKind::Ellipses, r, r, w.ta, w.tb, [w.rc, w.rc, zw, zw]));
        } else {
            out.push(segment_arc(r, w.ta, w.tb, zw, &w.a, &w.b));
        }
    }
    Ok(())
}

// nu, nv of a patch by the panel size: the panel-size rule of the geometry files (gdf.rs) on the
// arc lengths of its edges
fn discretization_of(p: &AnalyticPatch, size: f64) -> Result<PatchDiscretization> {
    let (lu, lv) = edge_lengths(|u, v| {
        let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        p.eval(u, v, &mut x, &mut xu, &mut xv);
        x
    });
    // 3 x 3 Gauss points per panel
    Ok(PatchDiscretization { nu: panel_count(lu, size)?, nv: panel_count(lv, size)?, ku: 3, kv: 3, iquo: 3, iqui: 3 })
}

/// The automatic control surface of a body mesh (theory §11.3).
pub fn automatic_control_surface(body: &Mesh, d: &ControlSurfaceDesc) -> Result<ControlSurfacePatches> {
    let circle = d.radius > 0.0;
    if !circle && d.polygon_count == 0 {
        return Err(Error::value(
            "automatic control surface: give the radius of a circular outer surface or a boundary polygon",
        ));
    }
    if circle && d.polygon_count > 0 {
        return Err(Error::value(
            "automatic control surface: give the radius of a circular outer surface or a boundary polygon (not both)",
        ));
    }
    if !(d.depth.is_finite() && d.depth > 0.0) {
        return Err(Error::value("control surface: DEPTH must be positive and finite"));
    }
    if circle && !d.radius.is_finite() {
        return Err(Error::value("control surface: RADIUS must be positive and finite"));
    }
    if !(d.panel_size.is_finite() && d.panel_size > 0.0) {
        return Err(Error::value("the control-surface panel size (PSZCSF) must be a positive finite length"));
    }
    if !d.waterline_z.is_finite() {
        return Err(Error::value("waterline_z must be finite"));
    }
    let mut poly: Vec<P2> = Vec::new();
    for k in 0..d.polygon_count as usize {
        poly.push([d.polygon[2 * k], d.polygon[2 * k + 1]]);
    }
    if !circle && poly.len() < 2 {
        return Err(Error::value("automatic control surface: the boundary polygon needs at least two vertices"));
    }
    let mut parts: Vec<Vec<P2>> = Vec::new();
    let mut at: usize = 0;
    for k in 0..d.partition_count as usize {
        let mut p: Vec<P2> = Vec::new();
        for _ in 0..d.partition_counts[k] {
            p.push([d.partitions[2 * at], d.partitions[2 * at + 1]]);
            at += 1;
        }
        if p.len() < 2 {
            return Err(Error::value("automatic control surface: a partition needs at least two vertices"));
        }
        parts.push(p);
    }
    let mut out = ControlSurfacePatches::default();
    out.symmetry[0] = body.isx;
    out.symmetry[1] = body.isy;
    let sym = [body.isx, body.isy];
    // the azimuth range of the stored part
    let mut th0 = 0.0;
    let mut th1 = 2.0 * PI;
    if sym[0] != 0 && sym[1] != 0 {
        th1 = 0.5 * PI;
    } else if sym[1] != 0 {
        th1 = PI;
    } else if sym[0] != 0 {
        th0 = -0.5 * PI;
        th1 = 0.5 * PI;
    }
    let zw = d.waterline_z;
    let r = d.radius;
    if circle {
        cylinder(r, d.depth, th0, th1, zw, &mut out.patches);
    } else {
        polygon_outer(&poly, d.depth, zw, &mut out.patches);
    }
    if d.free_surface != 0 {
        let mut segs = waterline_segments(body, zw, 1e-6);
        let mut circles = circle_sides(body, zw, 1e-6);
        // a waterline under a free-surface plate of the body (its edge on the wall: a covered
        // moonpool) borders no free surface: its midpoint within a tenth of its length of a plate
        // element (the chords of two meshes of one curve)
        if !d.covers.is_empty() {
            let covered = |x: P2, len: f64| d.covers.iter().any(|c| polygon::region_distance(c, x) <= 0.1 * len);
            segs.retain(|s| {
                let mid = [0.5 * (s[0][0] + s[1][0]), 0.5 * (s[0][1] + s[1][1])];
                !covered(mid, norm(s[1] - s[0]))
            });
            circles.retain(|c| {
                let t = 0.5 * (c.theta0 + c.theta1);
                !covered([c.radius * t.cos(), c.radius * t.sin()], c.radius * (c.theta1 - c.theta0).abs())
            });
        }
        if segs.is_empty() && circles.is_empty() {
            return Err(Error::value("automatic control surface: the body has no waterline"));
        }
        if circle && parts.is_empty() {
            free_surface_circle(&segs, &circles, r, &sym, th0, th1, zw, &mut out.patches)?;
        } else {
            let tol = length_tol(body.reference_length);
            let bnd = Boundary::new(if circle { r } else { -1.0 }, if circle { None } else { Some(&poly) }, &parts);
            if !segs.is_empty() {
                free_surface_general(&segs, &sym, zw, &bnd, tol, &mut out.patches)?;
            }
            free_surface_arcs(&circles, zw, &bnd, &mut out.patches)?;
        }
    }
    for p in &out.patches {
        out.discretization.push(discretization_of(p, d.panel_size)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // the waterline clipping of one panel: a square across z = 0 keeps its submerged half,
    // the vertices on the waterline put in z = 0
    #[test]
    fn clip_square_across_waterline() -> Result<()> {
        let sq = [v3(0.0, 0.0, -1.0), v3(1.0, 0.0, -1.0), v3(1.0, 0.0, 1.0), v3(0.0, 0.0, 1.0)];
        let part = clip(&sq, 1e-9);
        assert_eq!(part.len(), 4);
        assert_eq!(part[0], v3(0.0, 0.0, -1.0));
        assert_eq!(part[1], v3(1.0, 0.0, -1.0));
        assert_eq!(part[2], v3(1.0, 0.0, 0.0));
        assert_eq!(part[3], v3(0.0, 0.0, 0.0));
        let q = quads(&part)?;
        assert_eq!(q.len(), 1);
        Ok(())
    }

    // a triangle tip above the water: the clipped loop is a quadrilateral; a corner below the
    // water of a square: a triangle (repeated last vertex); a square with one dry corner: a pentagon
    // split into a quadrilateral and a triangle
    #[test]
    fn clip_shapes() -> Result<()> {
        let tri = [v3(0.0, 0.0, -1.0), v3(2.0, 0.0, -1.0), v3(1.0, 0.0, 1.0)];
        let p = clip(&tri, 1e-9);
        assert_eq!(p.len(), 4);
        assert!((p[2][0] - 1.5).abs() < 1e-15 && p[2][2] == 0.0);
        assert!((p[3][0] - 0.5).abs() < 1e-15 && p[3][2] == 0.0);

        let sq = [v3(0.0, 0.0, -1.0), v3(2.0, 0.0, 1.0), v3(2.0, 0.0, 3.0), v3(0.0, 0.0, 1.0)];
        let p = clip(&sq, 1e-9);
        assert_eq!(p.len(), 3);
        let q = quads(&p)?;
        assert_eq!(q.len(), 1);
        assert_eq!(q[0][2], q[0][3]);

        let sq = [v3(0.0, 0.0, -2.0), v3(2.0, 0.0, -2.0), v3(2.0, 0.0, 1.0), v3(0.0, 0.0, -1.0)];
        let p = clip(&sq, 1e-9);
        assert_eq!(p.len(), 5);
        let q = quads(&p)?;
        assert_eq!(q.len(), 2);
        assert_eq!(q[1][2], q[1][3]);
        Ok(())
    }

    // a vertex exactly on the waterline appears once; a wholly dry panel leaves nothing
    #[test]
    fn clip_vertex_on_waterline_and_dry() -> Result<()> {
        let tri = [v3(0.0, 0.0, -1.0), v3(1.0, 0.0, 0.0), v3(2.0, 0.0, -1.0)];
        let p = clip(&tri, 1e-9);
        assert_eq!(p.len(), 3);
        let dry = [v3(0.0, 0.0, 1.0), v3(1.0, 0.0, 1.0), v3(1.0, 0.0, 2.0)];
        assert!(clip(&dry, 1e-9).is_empty());
        assert!(quads(&clip(&dry, 1e-9))?.is_empty());
        Ok(())
    }

    // the chords of a polyline in z = 0
    fn polyline(p: &[[f64; 2]]) -> Segments {
        p.windows(2).map(|w| [v3(w[0][0], w[0][1], 0.0), v3(w[1][0], w[1][1], 0.0)]).collect()
    }

    // the quadrant (xy symmetry) of a square annulus: one loop closed along both axes, around the
    // corner of the quadrant, not star-shaped about it; the full annulus (no symmetry): a loop and
    // its hole; two squares: two loops; a chain off the planes of the symmetry: refused
    #[test]
    fn waterline_loops_close_along_the_symmetry_planes() -> Result<()> {
        let mut s = polyline(&[[2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]);
        s.extend(polyline(&[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]));
        let w = waterline_loops(&s, [1, 1], 1e-9)?;
        assert_eq!(w.loops.len(), 1);
        assert_eq!(w.loops[0].len(), 6);
        assert!((polygon::signed_area(&w.loops[0]).abs() - 3.0).abs() < 1e-14);
        assert!(!w.star_shaped(&v3(0.0, 0.0, 0.0)));
        let sq = |a: f64, x0: f64| polyline(&[[x0 - a, -a], [x0 + a, -a], [x0 + a, a], [x0 - a, a], [x0 - a, -a]]);
        let mut s = sq(2.0, 0.0);
        s.extend(sq(1.0, 0.0));
        let w = waterline_loops(&s, [0, 0], 1e-9)?;
        let r = polygon::regions(w.loops.clone());
        assert_eq!((r.len(), r[0].holes.len()), (1, 1));
        let mut s = sq(1.0, -2.0);
        s.extend(sq(1.0, 2.0));
        let w = waterline_loops(&s, [0, 0], 1e-9)?;
        assert_eq!(w.loops.len(), 2);
        assert!(!w.star_shaped(&v3(0.0, 0.0, 0.0)));
        // one square about its centre: star-shaped
        assert!(waterline_loops(&sq(1.0, 0.0), [0, 0], 1e-9)?.star_shaped(&v3(0.0, 0.0, 0.0)));
        let e = waterline_loops(&polyline(&[[2.0, 0.5], [2.0, 2.0], [0.0, 2.0]]), [1, 1], 1e-9);
        assert!(e.is_err_and(|e| e.message().contains("does not close")));
        Ok(())
    }

    #[test]
    fn fibre_classes() {
        assert_eq!(fibre_class(&[-1.0, -0.5, 0.0], 1e-9), Fiber::W);
        assert_eq!(fibre_class(&[0.0, 0.5, 1.0], 1e-9), Fiber::D);
        assert_eq!(fibre_class(&[-1.0, 0.5, 1.0], 1e-9), Fiber::C1);
        assert_eq!(fibre_class(&[1.0, 0.5, -1.0], 1e-9), Fiber::C2);
        assert_eq!(fibre_class(&[1.0, -0.5, 1.0], 1e-9), Fiber::None);
    }
}
