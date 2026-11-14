//! Mesh quality: `Model::mesh_quality`, the quality of a mesh alone (`Model::mesh_geometry_quality`)
//! and the mesh warnings of the rules.
//!
//! Per element the element size, its aspect ratio and the elements per shortest wavelength; per
//! body (or mesh) the volumes of the divergence theorem and the edges (of a patch mesh, the patch
//! sides) of the closed surface (orientation, gaps, non-manifold edges, open waterline).

use crate::codes::{PanelRole, QualityFlags, Warning};
use crate::mesh_desc::PatchKind;
use crate::model_codes::Domain;
use std::collections::{BTreeMap, BTreeSet};

use crate::constants::{PI, fmt, fmt_g};
use crate::dispersion::dispersion_x0;
use crate::error::{Diagnostic, Error, Result, at_body_element};
use crate::geometry::{
    COINCIDENCE_TOL, Placement, Reflection, Vec3, cross, dot, max_of, min_of, norm, reflect_vertices, reflections, v3,
};
use crate::inspect::{Scope, inspect_tessellation, mesh_geometry};
use crate::mesh::{Mesh, patch_eval, record_eval};
use crate::problem::{Problem, Role, body_label};

/// Low-order panels per shortest wavelength below which a mesh is coarse: the panel average of a
/// wave e^{ikx} over a diameter d errs by (k d)^2 / 24, 4.6 % at 6 **[C]**.
pub const QUALITY_PER_WAVELENGTH: f64 = 6.0;
/// The relative spread of the three volumes of the divergence theorem above which a closed
/// surface is suspect **[C]**.
pub const QUALITY_VOLUME_SPREAD: f64 = 1e-2;
/// The largest gap between the edges of two panels or the sides of two patches that do not meet at
/// their vertices, relative to the edge length, at which they still meet **[C]**: a gap of a
/// thousandth of an element lies far below the error of the discretization, a larger one is a
/// defect of the geometry worth reporting.
pub const QUALITY_EDGE_GAP: f64 = 1e-3;
/// Two wetted elements with opposite normals lie on each other when the centroid of one lies on the
/// other within this fraction of the larger element's diameter **[C]**: two meshes of one circular
/// surface, of N chords a circle each, lie apart by up to the sagitta, pi/4N of a chord (0.011 at
/// N = 72, 0.020 at 40), while a thin solid part (a heave plate 0.1 m thick under elements of 1 to
/// 1.6 m, 0.06 to 0.1) lies above it.
pub const QUALITY_FACING_GAP: f64 = 0.02;
// the cosine of the angle between the normals above which two elements face each other
const FACING_COS: f64 = -0.9;
// the points of a patch side at which its neighbours are sought (the quarter points of halves,
// none at an end or the middle, where the corners of the neighbours of a T-junction lie), and the
// points of its outline for the search
const SIDE_SAMPLES: usize = 4;
const SIDE_OUTLINE: usize = 16;

#[derive(Clone, Debug, Default)]
pub struct BodyQuality {
    pub size: Vec<f64>, // per stored element
    pub aspect: Vec<f64>,
    pub per_wavelength: Vec<f64>,
    pub flags: Vec<QualityFlags>, // per stored element
    pub volume: [f64; 3],
    pub edges_checked: bool,
    pub open_boundary: bool, // the surface ends on the sea bed or a wall plane
    pub inconsistent_edges: i32,
    pub free_edges: i32,
    pub non_manifold_edges: i32,
    pub open_waterline: i32,
}

#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn shortest_wavelength(m: &Problem) -> f64 {
    let mut k = 0.0;
    for &w in &m.omegas {
        if w > 0.0 && w.is_finite() {
            let K = w * w / m.env.gravity;
            k = max_of(k, if m.env.depth.is_finite() { dispersion_x0(K * m.env.depth) / m.env.depth } else { K });
        }
    }
    if k > 0.0 { 2.0 * PI / k } else { f64::INFINITY }
}

// a curve of a patch: its length from 16 chords
fn patch_length(mesh: &Mesh, rec: i32, along_u: bool) -> f64 {
    let mut len = 0.0;
    let mut prev = Vec3::ZERO;
    let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    for i in 0..=16 {
        let t = -1.0 + i as f64 / 8.0;
        if along_u {
            patch_eval(mesh, rec, t, 0.0, &mut x, &mut xu, &mut xv);
        } else {
            patch_eval(mesh, rec, 0.0, t, &mut x, &mut xu, &mut xv);
        }
        if i > 0 {
            len += norm(x - prev);
        }
        prev = x;
    }
    len
}

// a flat patch: the bilinear map of its corners
fn flat_length(mesh: &Mesh, rec: i32, along_u: bool) -> f64 {
    let q = &mesh.vertices[12 * rec as usize..];
    let a = v3(q[0], q[1], q[2]);
    let b = v3(q[3], q[4], q[5]);
    let c = v3(q[6], q[7], q[8]);
    let d = v3(q[9], q[10], q[11]);
    // the centre lines: u from the mid-edge (a, d) to (b, c), v from (a, b) to (d, c)
    if along_u { norm(0.5 * (b + c) - 0.5 * (a + d)) } else { norm(0.5 * (d + c) - 0.5 * (a + b)) }
}

#[derive(Clone, Copy)]
struct Use {
    rec: i32,
    forward: bool,
}

// the element of every element of a mesh: a low-order panel (diameter d, aspect d^2 / 2A: 1 for a
// square), or a span of a patch (hu x hv: its diagonal and (hu^2 + hv^2) / (2 hu hv)); the elements
// per wavelength lambda, coarse below QUALITY_PER_WAVELENGTH for a low-order element of `coarse`
fn record_sizes(mesh: &Mesh, lambda: f64, coarse: &dyn Fn(usize) -> bool) -> BodyQuality {
    let n = mesh.count.max(0) as usize;
    let mut q = BodyQuality {
        size: vec![0.0; n],
        aspect: vec![0.0; n],
        per_wavelength: vec![0.0; n],
        flags: vec![QualityFlags::default(); n],
        ..BodyQuality::default()
    };
    for rec in 0..n {
        let (size, aspect);
        if !mesh.patch {
            let p = &mesh.panels[rec];
            size = p.diameter;
            aspect = if p.area > 0.0 { p.diameter * p.diameter / (2.0 * p.area) } else { f64::INFINITY };
        } else {
            let flat = mesh.geometry == PatchKind::Flat && !mesh.trimmed();
            let r = rec as i32;
            let lu = if flat { flat_length(mesh, r, true) } else { patch_length(mesh, r, true) };
            let lv = if flat { flat_length(mesh, r, false) } else { patch_length(mesh, r, false) };
            let ds = &mesh.discretization[rec];
            let hu = lu / 1.max(ds.nu) as f64;
            let hv = lv / 1.max(ds.nv) as f64;
            size = hu.hypot(hv);
            aspect = if hu > 0.0 && hv > 0.0 { (hu * hu + hv * hv) / (2.0 * hu * hv) } else { f64::INFINITY };
        }
        q.size[rec] = size;
        q.aspect[rec] = aspect;
        q.per_wavelength[rec] = if size > 0.0 { lambda / size } else { f64::INFINITY };
        // the error (k d)^2 / 24 of the panel average of a wave exceeds 4.6 % below 6 per wavelength
        if !mesh.patch && coarse(rec) && q.per_wavelength[rec] < QUALITY_PER_WAVELENGTH {
            q.flags[rec] |= QualityFlags::COARSE;
        }
    }
    q
}

// the tolerance of the vertex and plane tests: from the extent of the mesh
fn tolerance(mesh: &Mesh) -> f64 {
    let mut extent = mesh.reference_length;
    for s in &mesh.shapes {
        for x in &s.hull {
            extent = max_of(max_of(max_of(extent, x[0].abs()), x[1].abs()), x[2].abs());
        }
    }
    COINCIDENCE_TOL * max_of(1.0, extent)
}

// an edge of a panel that no other panel shares by its vertices: from a to c in the sense of the
// panel's vertices
struct Loose {
    a: Vec3,
    c: Vec3,
    rec: usize,
}

// the edges of the closed surface of a low-order mesh placed by place: the elements of `keep` with
// their reflections; a free edge on the waterline z = 0 is the waterline, one on `on_boundary` (the
// sea bed, a wall plane) opens the surface there. An edge that no other panel shares by its
// vertices may lie along the edges of smaller panels (a T-junction: panels need not meet at their
// vertices); it is followed as a patch side (check_patch_edges): each of its points must lie on
// such an edge of exactly one other panel, within QUALITY_EDGE_GAP of its length, traversed in the
// opposite sense
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
fn check_edges(
    mesh: &Mesh,
    place: &Placement,
    keep: &dyn Fn(usize) -> bool,
    tol: f64,
    on_boundary: &dyn Fn(Vec3) -> bool,
    q: &mut BodyQuality,
) {
    let n = mesh.count.max(0) as usize;
    q.edges_checked = true;
    // the closed surface: the elements kept with their reflections, placed
    let mut ids: BTreeMap<[i64; 3], usize> = BTreeMap::new();
    let mut verts: Vec<Vec3> = Vec::new();
    // vertices merged on a grid of cells of size tol: two within tol of each other on two sides of
    // a cell boundary stay apart, and their edges are then followed as loose edges, which match
    // within QUALITY_EDGE_GAP (the counts are the same)
    let mut vid = |x: Vec3| -> usize {
        let key = [(x[0] / tol).round() as i64, (x[1] / tol).round() as i64, (x[2] / tol).round() as i64];
        if let Some(&id) = ids.get(&key) {
            return id;
        }
        let n = verts.len();
        ids.insert(key, n);
        verts.push(x);
        n
    };
    let mut edges: BTreeMap<(usize, usize), Vec<Use>> = BTreeMap::new();
    for r in reflections(mesh.isx, mesh.isy) {
        for rec in 0..n {
            if !keep(rec) {
                continue;
            }
            let v = reflect_vertices(&mesh.panels[rec].raw, &r);
            let mut lp: Vec<usize> = Vec::new();
            for x in &v {
                let id = vid(place.apply(*x));
                if lp.is_empty() || *lp.last().unwrap() != id {
                    lp.push(id);
                }
            }
            while lp.len() > 1 && lp[0] == lp[lp.len() - 1] {
                lp.pop();
            }
            for k in 0..lp.len() {
                let a = lp[k];
                let c = lp[(k + 1) % lp.len()];
                if a == c {
                    continue;
                }
                edges.entry((a.min(c), a.max(c))).or_default().push(Use { rec: rec as i32, forward: a < c });
            }
        }
    }
    let mut waterline_degree: BTreeMap<usize, i32> = BTreeMap::new();
    let mut loose: Vec<Loose> = Vec::new();
    for (key, uses) in &edges {
        let a = verts[key.0];
        let c = verts[key.1];
        if uses.len() == 1 {
            if a[2].abs() <= tol && c[2].abs() <= tol {
                *waterline_degree.entry(key.0).or_insert(0) += 1;
                *waterline_degree.entry(key.1).or_insert(0) += 1;
            } else if on_boundary(a) && on_boundary(c) {
                q.open_boundary = true;
            } else {
                let (a, c) = if uses[0].forward { (a, c) } else { (c, a) };
                loose.push(Loose { a, c, rec: uses[0].rec as usize });
            }
        } else if uses.len() == 2 {
            if uses[0].forward == uses[1].forward {
                q.inconsistent_edges += 1;
                q.flags[uses[0].rec as usize] |= QualityFlags::ORIENTATION;
                q.flags[uses[1].rec as usize] |= QualityFlags::ORIENTATION;
            }
        } else {
            q.non_manifold_edges += 1;
            for u in uses {
                q.flags[u.rec as usize] |= QualityFlags::NON_MANIFOLD;
            }
        }
    }
    for (&v, &n) in &waterline_degree {
        if n % 2 != 0 && !on_boundary(verts[v]) {
            q.open_waterline += 1;
        }
    }
    check_loose_edges(&loose, tol, q);
}

// the edges no other panel shares by its vertices (check_edges): met where they lie on such edges
// of other panels (a T-junction), otherwise free
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
fn check_loose_edges(loose: &[Loose], tol: f64, q: &mut BodyQuality) {
    if loose.is_empty() {
        return;
    }
    let length = |e: &Loose| norm(e.c - e.a);
    // a grid of cells of the median edge length: every edge in the cells of its box
    let mut lengths: Vec<f64> = loose.iter().map(length).collect();
    lengths.sort_by(f64::total_cmp);
    let h = max_of(lengths[lengths.len() / 2], tol);
    let cell = |x: Vec3| [(x[0] / h).floor() as i64, (x[1] / h).floor() as i64, (x[2] / h).floor() as i64];
    let mut grid: BTreeMap<[i64; 3], Vec<usize>> = BTreeMap::new();
    for (i, e) in loose.iter().enumerate() {
        let margin = QUALITY_EDGE_GAP * length(e) + tol;
        let (mut lo, mut hi) = (e.a, e.a);
        for c in 0..3 {
            lo[c] = min_of(e.a[c], e.c[c]) - margin;
            hi[c] = max_of(e.a[c], e.c[c]) + margin;
        }
        let (a, b) = (cell(lo), cell(hi));
        for ix in a[0]..=b[0] {
            for iy in a[1]..=b[1] {
                for iz in a[2]..=b[2] {
                    grid.entry([ix, iy, iz]).or_default().push(i);
                }
            }
        }
    }
    for (i, e) in loose.iter().enumerate() {
        let d = e.c - e.a;
        let gap = QUALITY_EDGE_GAP * length(e) + tol;
        let (mut free, mut many, mut flipped) = (false, false, false);
        for j in 0..SIDE_SAMPLES {
            let x = e.a + ((2 * j + 1) as f64 / (2 * SIDE_SAMPLES) as f64) * d;
            // the edges within the gap of x: (edge, parameter t in [0, 1] along it)
            let mut found: Vec<(usize, f64)> = Vec::new();
            for &k in grid.get(&cell(x)).map_or(&[][..], |v| v.as_slice()) {
                let o = &loose[k];
                let od = o.c - o.a;
                let oo = dot(od, od);
                if k == i || !(oo > 0.0) {
                    continue;
                }
                let t = (dot(x - o.a, od) / oo).clamp(0.0, 1.0);
                if norm(o.a + t * od - x) <= gap {
                    found.push((k, t));
                }
            }
            // a point at the corner of neighbours lies at the end of each of their edges: it is met
            let at_corner = found.len() > 1;
            if at_corner {
                found.retain(|&(_, t)| t > 1e-9 && t < 1.0 - 1e-9);
            }
            match found.as_slice() {
                [] if at_corner => {}
                [] => free = true,
                [(k, _)] => {
                    if dot(d, loose[*k].c - loose[*k].a) > 0.0 {
                        flipped = true;
                    }
                }
                _ => many = true,
            }
        }
        if free {
            q.free_edges += 1;
            q.flags[e.rec] |= QualityFlags::FREE_EDGE;
        } else if many {
            q.non_manifold_edges += 1;
            q.flags[e.rec] |= QualityFlags::NON_MANIFOLD;
        } else if flipped {
            q.inconsistent_edges += 1;
            q.flags[e.rec] |= QualityFlags::ORIENTATION;
        }
    }
}

// a side k of an element in a reflection: s in [-1, 1] runs around the square (u, v) in the sense
// that keeps the parametric normal on its left (side 0 v = -1, 1 u = 1, 2 v = 1, 3 u = -1)
struct Side {
    rec: usize,
    refl: Reflection,
    k: usize,
    outline: Vec<Vec3>,
    length: f64,
    lo: Vec3,
    hi: Vec3,
}

// the point of a side at s and its derivative in s, reflected (an odd reflection reverses the
// sense of the traversal: `sense`)
fn side_eval(mesh: &Mesh, rec: usize, r: &Reflection, k: usize, s: f64) -> (Vec3, Vec3) {
    let (u, v, du, dv) = match k {
        0 => (s, -1.0, 1.0, 0.0),
        1 => (1.0, s, 0.0, 1.0),
        2 => (-s, 1.0, -1.0, 0.0),
        _ => (-1.0, -s, 0.0, -1.0),
    };
    let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    record_eval(mesh, rec as i32, u, v, &mut x, &mut xu, &mut xv);
    let f = |p: Vec3| v3(r.sx * p[0], r.sy * p[1], p[2]);
    (f(x), f(du * xu + dv * xv))
}
fn sense(r: &Reflection) -> f64 {
    r.sx * r.sy
}

// the parameter of the point of side q nearest to x (from the nearest point of its outline by
// Newton's method on the distance) and the distance
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
fn project(mesh: &Mesh, q: &Side, x: Vec3) -> (f64, f64) {
    let n = q.outline.len() - 1;
    let mut best = 0;
    for i in 1..=n {
        if norm(q.outline[i] - x) < norm(q.outline[best] - x) {
            best = i;
        }
    }
    let mut s = -1.0 + 2.0 * best as f64 / n as f64;
    for _ in 0..16 {
        let (y, t) = side_eval(mesh, q.rec, &q.refl, q.k, s);
        let tt = dot(t, t);
        if !(tt > 0.0) {
            break;
        }
        let ds = -dot(y - x, t) / tt;
        s = (s + ds).clamp(-1.0, 1.0);
        if ds.abs() < 1e-13 {
            break;
        }
    }
    (s, norm(side_eval(mesh, q.rec, &q.refl, q.k, s).0 - x))
}

// the sides of the closed surface of a patch mesh placed by place: the elements of `keep` with
// their reflections. Patches need not meet at their corners (a side may border several smaller
// ones), so a side is followed by points along it: each must lie on the side of exactly one other
// patch (within QUALITY_EDGE_GAP of the side length), traversed in the opposite sense; a side in
// the waterline z = 0 or on `on_boundary` is no edge of the closed surface, and an end of a
// waterline side must lie on another one. The counts are of patch sides.
fn check_patch_edges(
    mesh: &Mesh,
    place: &Placement,
    keep: &dyn Fn(usize) -> bool,
    tol: f64,
    on_boundary: &dyn Fn(Vec3) -> bool,
    q: &mut BodyQuality,
) {
    let n = mesh.count.max(0) as usize;
    q.edges_checked = true;
    let mut sides: Vec<Side> = Vec::new();
    for r in reflections(mesh.isx, mesh.isy) {
        for rec in 0..n {
            if !keep(rec) {
                continue;
            }
            for k in 0..4 {
                let outline: Vec<Vec3> = (0..=SIDE_OUTLINE)
                    .map(|i| side_eval(mesh, rec, &r, k, -1.0 + 2.0 * i as f64 / SIDE_OUTLINE as f64).0)
                    .collect();
                let length: f64 = outline.windows(2).map(|w| norm(w[1] - w[0])).sum();
                if length <= tol {
                    continue; // a side collapsed to a point (an axis, a pole)
                }
                let (mut lo, mut hi) = (outline[0], outline[0]);
                for x in &outline {
                    for c in 0..3 {
                        lo[c] = min_of(lo[c], x[c]);
                        hi[c] = max_of(hi[c], x[c]);
                    }
                }
                // the side between two outline points stays within a fraction of a segment
                let margin = length / SIDE_OUTLINE as f64 + QUALITY_EDGE_GAP * length + tol;
                for c in 0..3 {
                    lo[c] -= margin;
                    hi[c] += margin;
                }
                sides.push(Side { rec, refl: r, k, outline, length, lo, hi });
            }
        }
    }
    if sides.is_empty() {
        return;
    }
    // a grid of cells of the median side length: every side in the cells of the boxes of its
    // outline segments (the side between two outline points stays within a segment of them), so
    // that a long side does not fill the cells of its whole box
    let mut lengths: Vec<f64> = sides.iter().map(|x| x.length).collect();
    lengths.sort_by(f64::total_cmp);
    let h = lengths[lengths.len() / 2];
    let cell = |x: Vec3| [(x[0] / h).floor() as i64, (x[1] / h).floor() as i64, (x[2] / h).floor() as i64];
    let mut grid: BTreeMap<[i64; 3], Vec<usize>> = BTreeMap::new();
    for (i, sd) in sides.iter().enumerate() {
        let mut cells: BTreeSet<[i64; 3]> = BTreeSet::new();
        for w in sd.outline.windows(2) {
            let margin = norm(w[1] - w[0]) + QUALITY_EDGE_GAP * sd.length + tol;
            let (mut lo, mut hi) = (w[0], w[0]);
            for c in 0..3 {
                lo[c] = min_of(w[0][c], w[1][c]) - margin;
                hi[c] = max_of(w[0][c], w[1][c]) + margin;
            }
            let (a, b) = (cell(lo), cell(hi));
            for ix in a[0]..=b[0] {
                for iy in a[1]..=b[1] {
                    for iz in a[2]..=b[2] {
                        cells.insert([ix, iy, iz]);
                    }
                }
            }
        }
        for c in cells {
            grid.entry(c).or_default().push(i);
        }
    }
    let inside = |x: Vec3, sd: &Side| (0..3).all(|c| sd.lo[c] <= x[c] && x[c] <= sd.hi[c]);
    // the sides within the gap of x: (side, parameter)
    let near = |x: Vec3, except: usize, gap: f64, only: &dyn Fn(usize) -> bool| -> Vec<(usize, f64)> {
        let mut found = Vec::new();
        for &j in grid.get(&cell(x)).map_or(&[][..], |v| v.as_slice()) {
            if j == except || !only(j) || !inside(x, &sides[j]) {
                continue;
            }
            let (s, d) = project(mesh, &sides[j], x);
            if d <= gap {
                found.push((j, s));
            }
        }
        found
    };
    let mut waterline: Vec<usize> = Vec::new();
    for (i, p) in sides.iter().enumerate() {
        let gap = QUALITY_EDGE_GAP * p.length + tol;
        let points: Vec<(Vec3, Vec3)> = (0..SIDE_SAMPLES)
            .map(|j| side_eval(mesh, p.rec, &p.refl, p.k, -1.0 + (2 * j + 1) as f64 / SIDE_SAMPLES as f64))
            .collect();
        if points.iter().all(|(x, _)| place.apply(*x)[2].abs() <= gap) {
            waterline.push(i);
            continue;
        }
        if points.iter().all(|(x, _)| on_boundary(place.apply(*x))) {
            q.open_boundary = true;
            continue;
        }
        let (mut free, mut many, mut flipped) = (false, false, false);
        for &(x, t) in &points {
            let mut found = near(x, i, gap, &|_| true);
            // a point at the corner of neighbours lies at the end of each of their sides: it is
            // met, and the sense is tested at the other points (and by the neighbours' own points)
            let at_corner = found.len() > 1;
            if at_corner {
                found.retain(|&(_, s)| s.abs() < 1.0 - 1e-9);
            }
            match found.as_slice() {
                [] if at_corner => {}
                [] => free = true,
                [(j, s)] => {
                    let o = &sides[*j];
                    let to = side_eval(mesh, o.rec, &o.refl, o.k, *s).1;
                    if dot(sense(&p.refl) * t, sense(&o.refl) * to) > 0.0 {
                        flipped = true;
                    }
                }
                _ => many = true,
            }
        }
        if free {
            q.free_edges += 1;
            q.flags[p.rec] |= QualityFlags::FREE_EDGE;
        } else if many {
            q.non_manifold_edges += 1;
            q.flags[p.rec] |= QualityFlags::NON_MANIFOLD;
        } else if flipped {
            q.inconsistent_edges += 1;
            q.flags[p.rec] |= QualityFlags::ORIENTATION;
        }
    }
    // the open ends of the waterline: an end of a waterline side on no other waterline side
    let in_waterline = |j: usize| waterline.contains(&j);
    for &i in &waterline {
        let p = &sides[i];
        let gap = QUALITY_EDGE_GAP * p.length + tol;
        for s in [-1.0, 1.0] {
            let x = side_eval(mesh, p.rec, &p.refl, p.k, s).0;
            if !on_boundary(place.apply(x)) && near(x, i, gap, &in_waterline).is_empty() {
                q.open_waterline += 1;
            }
        }
    }
}

/// The quality of one body after the rules.
pub fn analyze_quality(m: &Problem, body: i32) -> BodyQuality {
    let bi = body as usize;
    let b = &m.bodies[bi];
    let mesh = &m.meshes[b.mesh as usize];
    let d = &m.derived()[bi];
    let lambda = shortest_wavelength(m);
    // the wetted and pressure elements of the body's mesh (not those of other meshes the higher
    // order appended): the coarse elements and the closed surface of a low-order mesh
    let wet = |rec: usize| {
        (d.role[rec] == Role::Wet || d.role[rec] == Role::Pressure) && m.joined_at(b.mesh, rec as i32).is_none()
    };
    let mut q = record_sizes(mesh, lambda, &wet);
    let h = m.hydrostatics(body);
    for k in 0..3 {
        q.volume[k] = h.volumes[k];
    }

    // the extent of the body: the tolerances of the vertex and plane tests
    let tol = tolerance(mesh);
    let place = Placement { position: b.position, yaw_deg: b.yaw };
    // an element on the sea bed or a wall plane opens the surface there (a column on the bed, a wavemaker)
    let on_boundary = |x: Vec3| -> bool {
        if m.env.depth.is_finite() && (x[2] + m.env.depth).abs() <= tol {
            return true;
        }
        let wx = m.opt.domain == Domain::XWall || m.opt.domain == Domain::OrthogonalWalls;
        let wy = m.opt.domain == Domain::YWall || m.opt.domain == Domain::OrthogonalWalls;
        if wx && x[0].abs() <= tol {
            return true;
        }
        if wy && x[1].abs() <= tol {
            return true;
        }
        if m.opt.domain == Domain::Channel && (x[1].abs() - 0.5 * m.opt.channel_width).abs() <= tol {
            return true;
        }
        false
    };
    if mesh.patch {
        // the closed surface takes the wetted and pressure elements of the meshes appended to the
        // body's mesh too (a cushion given as a mesh of its own closes the hull)
        let closing = |rec: usize| d.role[rec] == Role::Wet || d.role[rec] == Role::Pressure;
        check_patch_edges(mesh, &place, &closing, tol, &on_boundary, &mut q);
    } else {
        check_edges(mesh, &place, &wet, tol, &on_boundary, &mut q);
    }
    q
}

/// The geometric quality of a mesh alone, in its own coordinates: per element the element size and
/// aspect ratio (no wavelength: the elements per wavelength infinite), the volumes of the
/// divergence theorem and the edges (of a patch mesh the patch sides) of the surface of all its
/// elements closed by the waterline z = 0.
pub fn mesh_geometry_quality(mesh: &Mesh) -> BodyQuality {
    let mut q = record_sizes(mesh, f64::INFINITY, &|_| false);
    q.volume = mesh_geometry(mesh).volumes;
    if mesh.patch {
        check_patch_edges(mesh, &Placement::default(), &|_| true, tolerance(mesh), &|_| false, &mut q);
    } else {
        check_edges(mesh, &Placement::default(), &|_| true, tolerance(mesh), &|_| false, &mut q);
    }
    q
}

/// The quality of the body meshes (`Model::mesh_quality`, `Model::body_mesh_quality`; N stored
/// elements, B bodies), or of a mesh alone (`Model::mesh_geometry_quality`: one row per body, body
/// -1, no wavelength).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshQuality {
    /// per element: the zero-based body; -1 for a mesh alone
    pub body: Vec<i32>,
    /// per element: the one-based element of the body's mesh (as `Panels::element`: of an element a
    /// trim cut, the element it was cut from; an element the engine generated numbered from 1 among
    /// the generated ones), of a mesh alone its stored element
    pub element: Vec<i32>,
    /// per element: its part after a trim (as `Panels::part`), 0 for an element not cut
    pub part: Vec<i32>,
    /// per element: low order the panel diameter d, higher order the diagonal of a span, m
    pub size: Vec<f64>,
    /// per element: d^2 / 2A, 1 for a square
    pub aspect: Vec<f64>,
    /// per element: the elements per shortest wavelength; infinite without frequencies (and of a mesh)
    pub per_wavelength: Vec<f64>,
    /// per element: the findings
    pub flags: Vec<QualityFlags>,
    /// per body: the volumes of the divergence theorem in x, y, z, m^3
    pub volumes: Vec<[f64; 3]>,
    /// per body: the fewest elements per wavelength of its closed surface
    pub min_per_wavelength: Vec<f64>,
    /// per body: the inconsistently oriented, free and non-manifold edges (of a patch mesh, patch
    /// sides) and the open ends of the waterline
    pub edges: Vec<[i32; 4]>,
}

impl MeshQuality {
    // the rows of one body or mesh (body -1): its elements (wet: those of the closed surface; number:
    // the one-based stored element of a zero-based element)
    fn push(&mut self, bi: i32, q: &BodyQuality, wet: &dyn Fn(usize) -> bool, number: &dyn Fn(usize) -> (i32, i32)) {
        let mut worst = f64::INFINITY;
        for r in 0..q.size.len() {
            self.body.push(bi);
            let (element, part) = number(r);
            self.element.push(element);
            self.part.push(part);
            self.size.push(q.size[r]);
            self.aspect.push(q.aspect[r]);
            self.per_wavelength.push(q.per_wavelength[r]);
            self.flags.push(q.flags[r]);
            if wet(r) {
                worst = min_of(worst, q.per_wavelength[r]);
            }
        }
        self.volumes.push(q.volume);
        self.min_per_wavelength.push(worst);
        self.edges.push(if q.edges_checked {
            [q.inconsistent_edges, q.free_edges, q.non_manifold_edges, q.open_waterline]
        } else {
            [-1, -1, -1, -1]
        });
    }
}

/// The quality of a mesh alone (one row of the bodies, body -1).
pub fn geometry_quality(mesh: &Mesh) -> MeshQuality {
    let mut rows = MeshQuality::default();
    rows.push(-1, &mesh_geometry_quality(mesh), &|_| true, &|r| (r as i32 + 1, 0));
    rows
}

/// The quality of a body (or of every body: -1).
pub fn mesh_quality(m: &Problem, body: i32) -> Result<MeshQuality> {
    if body < -1 || body >= m.bodies.len() as i32 {
        return Err(Error::argument(format!("body index {body} does not exist")));
    }
    let mut rows = MeshQuality::default();
    for bi in 0..m.bodies.len() as i32 {
        if body >= 0 && bi != body {
            continue;
        }
        let d = &m.derived()[bi as usize];
        let mi = m.bodies[bi as usize].mesh;
        let wet =
            |r: usize| (d.role[r] == Role::Wet || d.role[r] == Role::Pressure) && m.joined_at(mi, r as i32).is_none();
        let number = |r: usize| {
            let (element, part, _) = m.element_key(mi, r as i32);
            (element, part)
        };
        rows.push(bi, &analyze_quality(m, bi), &wet, &number);
    }
    Ok(rows)
}

// a wetted element placed, with its reflections (one quadrilateral; of a patch one per span)
struct Face {
    body: i32,
    element: i32,
    corners: [Vec3; 4],
    centroid: Vec3,
    normal: Vec3, // unit, into the fluid
    diameter: f64,
    area: f64,
}

// whether the point x lies on the face f within gap along its normal (and within the outline,
// within 1e-6 of its diameter)
fn lies_on(f: &Face, x: Vec3, gap: f64) -> bool {
    if dot(x - f.centroid, f.normal).abs() > gap {
        return false;
    }
    // the sense of the outline: its own normal, from the diagonals
    let s = cross(f.corners[2] - f.corners[0], f.corners[3] - f.corners[1]);
    let tol = COINCIDENCE_TOL * f.diameter;
    let mut inside = true;
    for k in 0..4 {
        let a = f.corners[k];
        let e = f.corners[(k + 1) % 4] - a;
        let l = norm(e);
        if l <= tol {
            continue; // the repeated vertex of a triangle
        }
        inside &= dot(cross(e, x - a), s) / (l * norm(s)) >= -tol;
    }
    inside
}

/// The wetted elements of the bodies that lie on other wetted elements with opposite normals
/// (Warning::MeshFacing): of two bodies (two hulls in contact, both meshed across the contact) or
/// of one (a surface meshed on both sides with no thickness). A centroid lying on another element
/// within QUALITY_FACING_GAP of the larger diameter, the normals within 26 degrees of opposite (cos -0.9)
/// (theory §14).
#[allow(clippy::needless_range_loop)] // index notation over the faces and the bodies
fn facing_warnings(m: &Problem, out: &mut Vec<Diagnostic>) {
    let Ok(t) = inspect_tessellation(m, Scope::All, &[PanelRole::Wetted], [0, 0], true) else {
        return;
    };
    let mut faces: Vec<Face> = Vec::new();
    for i in 0..t.len() {
        let bi = t.body[i];
        if bi < 0 || m.derived()[bi as usize].wavemaker {
            continue; // a wavemaker lies on its own wall image
        }
        let c: [Vec3; 4] = std::array::from_fn(|k| Vec3(t.vertices[i][k]));
        let mut n = v3(0.0, 0.0, 0.0);
        for k in 0..4 {
            n += Vec3(t.normals[i][k]);
        }
        let area = 0.5 * norm(cross(c[2] - c[0], c[3] - c[1]));
        let mut diameter: f64 = 0.0;
        for a in 0..4 {
            for b in a + 1..4 {
                diameter = diameter.max(norm(c[b] - c[a]));
            }
        }
        if norm(n) == 0.0 || area <= 0.0 || diameter <= 0.0 {
            continue;
        }
        let centroid = 0.25 * (c[0] + c[1] + c[2] + c[3]);
        faces.push(Face {
            body: bi,
            element: t.element[i],
            corners: c,
            centroid,
            normal: (1.0 / norm(n)) * n,
            diameter,
            area,
        });
    }
    if faces.len() < 2 {
        return;
    }
    // a grid of cells of the median diameter: every face in the cells of its box, widened by its gap
    let mut d: Vec<f64> = faces.iter().map(|f| f.diameter).collect();
    d.sort_by(|a, b| a.total_cmp(b));
    let h = d[d.len() / 2];
    let cell = |x: f64| (x / h).floor() as i64;
    let mut grid: BTreeMap<[i64; 3], Vec<usize>> = BTreeMap::new();
    for (j, f) in faces.iter().enumerate() {
        let w = QUALITY_FACING_GAP * f.diameter;
        let lo: [i64; 3] =
            std::array::from_fn(|k| cell(f.corners.iter().map(|c| c[k]).fold(f64::INFINITY, f64::min) - w));
        let hi: [i64; 3] =
            std::array::from_fn(|k| cell(f.corners.iter().map(|c| c[k]).fold(f64::NEG_INFINITY, f64::max) + w));
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    grid.entry([x, y, z]).or_default().push(j);
                }
            }
        }
    }
    // per pair of bodies (a <= b) and side: the faces of that side lying on the other's, their area
    // and the first element
    let nb = m.bodies.len();
    let mut count = vec![vec![[0usize; 2]; nb]; nb];
    let mut area = vec![vec![[0.0f64; 2]; nb]; nb];
    let mut first = vec![vec![[i32::MAX; 2]; nb]; nb];
    for f in &faces {
        let x = f.centroid;
        let key = [cell(x[0]), cell(x[1]), cell(x[2])];
        let Some(near) = grid.get(&key) else { continue };
        let hit = near.iter().map(|&j| &faces[j]).find(|g| {
            !std::ptr::eq(*g, f)
                && dot(f.normal, g.normal) < FACING_COS
                && lies_on(g, x, QUALITY_FACING_GAP * max_of(f.diameter, g.diameter))
        });
        if let Some(g) = hit {
            let (a, b) = (f.body.min(g.body) as usize, f.body.max(g.body) as usize);
            let side = usize::from(f.body as usize != a);
            count[a][b][side] += 1;
            area[a][b][side] += f.area;
            first[a][b][side] = first[a][b][side].min(f.element);
        }
    }
    for a in 0..nb {
        for b in a..nb {
            let [na, nbb] = count[a][b];
            if na + nbb == 0 {
                continue;
            }
            let la = body_label(&m.bodies[a]);
            let gap = fmt_g(QUALITY_FACING_GAP, 3);
            let message = if a == b {
                format!(
                    "{la}: {na} wetted elements ({} m^2, the first element {}) lie on other wetted elements of the body with opposite normals, closer than {gap} of the element diameter: a surface meshed on both of its sides with almost no thickness between them",
                    fmt_g(area[a][b][0], 4),
                    first[a][b][0]
                )
            } else {
                let lb = body_label(&m.bodies[b]);
                let fa = if na > 0 { format!(", the first element {}", first[a][b][0]) } else { String::new() };
                let fb = if nbb > 0 { format!(", the first element {}", first[a][b][1]) } else { String::new() };
                format!(
                    "{la} and {lb}: {na} wetted elements of {la} ({} m^2{fa}) and {nbb} of {lb} ({} m^2{fb}) lie on each other with opposite normals, closer than {gap} of the element diameter: no fluid lies between them that the mesh resolves, yet both are wetted surface (where the bodies touch, leave those elements out of the wetted surfaces: the body item ranges)",
                    fmt_g(area[a][b][0], 4),
                    fmt_g(area[a][b][1], 4),
                )
            };
            let (body, element) = if na > 0 { (a as i32, first[a][b][0]) } else { (b as i32, first[a][b][1]) };
            out.push(Diagnostic::warning_at(Warning::MeshFacing, at_body_element(body, element), message));
        }
    }
}

/// The mesh warnings (Warning::CoarseMesh, MeshOrientation, MeshOpen, MeshVolume of every body;
/// MeshFacing of the bodies together).
pub fn quality_warnings(m: &Problem, out: &mut Vec<Diagnostic>) {
    for bi in 0..m.bodies.len() as i32 {
        if m.derived()[bi as usize].wavemaker {
            continue; // a wavemaker in a wall plane has no volume
        }
        let q = analyze_quality(m, bi);
        let at = body_label(&m.bodies[bi as usize]);
        let mi = m.bodies[bi as usize].mesh;
        // the location of a one-based stored element (none: -1, or 0 an element the engine generated)
        let mut warn = |w: Warning, record: i32, msg: String| {
            let record = if record > 0 { record } else { -1 };
            out.push(Diagnostic::warning_at(w, at_body_element(bi, record), format!("{at}: {msg}")));
        };
        // the stored element of the first element of the body's own mesh with a flag of `flag`, -1
        // none (an element appended from another mesh is not an element of this one)
        let first = |flag: QualityFlags| -> i32 {
            (0..q.flags.len())
                .find(|&r| q.flags[r].intersects(flag) && m.joined_at(mi, r as i32).is_none())
                .map_or(-1, |r| m.stored_record(mi, r as i32))
        };
        let mut coarse = 0;
        let mut worst: i32 = -1;
        for r in 0..q.flags.len() {
            if q.flags[r].intersects(QualityFlags::COARSE) {
                coarse += 1;
                if worst < 0 || q.per_wavelength[r] < q.per_wavelength[worst as usize] {
                    worst = r as i32;
                }
            }
        }
        if coarse > 0 {
            let record = m.stored_record(mi, worst);
            let place =
                if record > 0 { format!("element {record}") } else { "an element the trim generated".to_string() };
            warn(
                Warning::CoarseMesh,
                record,
                format!(
                    "{coarse} panels are larger than a sixth of the shortest wavelength ({} per wavelength at {place}; the panel average of the wave errs by (k d)^2 / 24)",
                    fmt(q.per_wavelength[worst as usize]),
                ),
            );
        }
        let patch = m.meshes[mi as usize].patch;
        if q.inconsistent_edges > 0 {
            warn(
                Warning::MeshOrientation,
                first(QualityFlags::ORIENTATION),
                if patch {
                    format!(
                        "{} patch sides are traversed in the same direction as the side of their neighbour: the normals of the wetted surface are not consistently oriented",
                        q.inconsistent_edges
                    )
                } else {
                    format!(
                        "{} edges are traversed in the same direction by both of their panels: the normals of the wetted surface are not consistently oriented",
                        q.inconsistent_edges
                    )
                },
            );
        }
        if q.free_edges > 0 || q.non_manifold_edges > 0 || q.open_waterline > 0 {
            warn(
                Warning::MeshOpen,
                first(QualityFlags::FREE_EDGE | QualityFlags::NON_MANIFOLD),
                if patch {
                    format!(
                        "the wetted surface is not closed: {} patch sides below the waterline meet no other patch (a gap of more than {} of the side), {} meet more than one, {} open ends of the waterline",
                        q.free_edges,
                        fmt(QUALITY_EDGE_GAP),
                        q.non_manifold_edges,
                        q.open_waterline
                    )
                } else {
                    format!(
                        "the wetted surface is not closed: {} free edges below the waterline (a gap of more than {} of the edge), {} edges of more than two panels, {} open ends of the waterline",
                        q.free_edges,
                        fmt(QUALITY_EDGE_GAP),
                        q.non_manifold_edges,
                        q.open_waterline
                    )
                },
            );
        }
        let vmax = max_of(max_of(q.volume[0].abs(), q.volume[1].abs()), q.volume[2].abs());
        let vmin = min_of(min_of(q.volume[0], q.volume[1]), q.volume[2]);
        let spread = max_of(max_of(q.volume[0], q.volume[1]), q.volume[2]) - vmin;
        if !q.open_boundary && vmax > 0.0 && (vmin <= 0.0 || spread > QUALITY_VOLUME_SPREAD * vmax) {
            warn(
                Warning::MeshVolume,
                -1,
                format!(
                    "the volumes of the divergence theorem are ({}, {}, {}): {}",
                    fmt(q.volume[0]),
                    fmt(q.volume[1]),
                    fmt(q.volume[2]),
                    if vmin <= 0.0 {
                        "a volume is not positive (normals into the body?)"
                    } else {
                        "they differ by more than 1 % (an open, inconsistently oriented or strongly non-planar surface: the projected panels leave gaps)"
                    }
                ),
            );
        }
    }
    facing_warnings(m, out);
}
