//! The zones of the free surface of the potential QTF (theory §12.8-§12.9): the panelled zone
//! inside the inner radius (the generated panels, the cells of an automatic control surface, or a
//! given mesh), the quadrature zone of annuli beyond it and the outer circle with the orders of
//! the far-field series.

use super::forcing::wall_axis;
use crate::codes::ControlSurfaceKind;
use crate::constants::PI;
use crate::document_build::{engine_mesh, gdf_desc};
use crate::error::{Error, Result, at_body};
use crate::geometry::{Vec3, reflections, v3};
use crate::lo_geometry::control_surface_points;
use crate::mesh::Mesh;
use crate::model_codes::QtfFrequencies;
use crate::model_meshes::control_surface_gdf;
use crate::model_types::{ControlSurfaceInput, wavenumber};
use crate::plate_geometry::{PlateElement, PlateIndex, Tie};
use crate::polygon::inside_convex;
use crate::preprocess::waterline_chords;
use crate::problem::{Problem, body_label};

/// A side of a body's waterline in the plane z = 0 (global): its body and its two ends.
#[derive(Clone, Copy, Debug)]
pub(super) struct WlSeg {
    pub(super) body: usize,
    pub(super) p: [[f64; 2]; 2],
}

/// A side of a body's waterline: its ends in body axes (the mesh's reflection applied) and in the
/// plane (global), and the plate element over the water beside it (its body and one-based element,
/// PlateElement), if any. A side under a plate borders no free surface: it is no waterline of the
/// free surface (theory §12.8) and its line integral of the body condition holds the plate's edge
/// (theory §12.7).
#[derive(Clone, Debug)]
pub(super) struct BodySide {
    pub(super) local: [Vec3; 2],
    pub(super) global: [[f64; 2]; 2],
    pub(super) plate: Option<(usize, i32)>,
}

/// The horizontal normal (into the water, mesh axes) of the element of each side of
/// waterline_chords(mesh, zw, tol), in its order: the loop of preprocess::waterline_segments over
/// the panels of a low-order mesh; none (zero) for patches, whose sides the plates do not take
/// (the potential QTF is a low-order method). A low-order mesh whose loop finds another number of
/// sides is an internal error: with zero normals no side would lie under a plate, and a covered
/// waterline would join the free surface's.
fn side_normals(mesh: &Mesh, zw: f64, tol: f64, n: usize) -> Result<Vec<[f64; 2]>> {
    let mut out = Vec::with_capacity(n);
    if !mesh.patch {
        let t = tol * 1.0f64.max(mesh.reference_length);
        for p in &mesh.panels {
            let q = &p.raw;
            if q.iter().all(|x| (x[2] - zw).abs() <= t) {
                continue;
            }
            for e in 0..4 {
                let (a, b) = (q[e], q[(e + 1) % 4]);
                let d = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
                if (a[2] - zw).abs() <= t && (b[2] - zw).abs() <= t && d > t {
                    out.push([p.normal[0], p.normal[1]]);
                }
            }
        }
    }
    if mesh.patch {
        out = vec![[0.0, 0.0]; n];
    } else if out.len() != n {
        return Err(Error::internal(format!(
            "potential QTF: the waterline of a mesh has {} sides and {} side normals",
            n,
            out.len()
        )));
    }
    Ok(out)
}

/// The waterline sides of body bi (its mesh's sides in z = 0, reflected with the mesh's symmetry
/// in the order (1, 1), (−1, 1), (1, −1), (−1, −1)) with the plate element over the water beside
/// each (PlateIndex::beside, the first of equal distances).
pub(super) fn body_sides(m: &Problem, bi: usize, plates: &PlateIndex<PlateElement>) -> Result<Vec<BodySide>> {
    let b = &m.bodies[bi];
    let mesh = &m.meshes[b.mesh as usize];
    let (cy, sy) = ((b.yaw * PI / 180.0).cos(), (b.yaw * PI / 180.0).sin());
    let place = |x: f64, y: f64| [b.position[0] + cy * x - sy * y, b.position[1] + sy * x + cy * y];
    let chords = waterline_chords(mesh, -b.position[2], 1e-6);
    let normals =
        if plates.elements.is_empty() { Vec::new() } else { side_normals(mesh, -b.position[2], 1e-6, chords.len())? };
    let mut out = Vec::new();
    for (fx, fy) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
        if (fx < 0.0 && mesh.isx == 0) || (fy < 0.0 && mesh.isy == 0) {
            continue;
        }
        for (k, s) in chords.iter().enumerate() {
            let local = [v3(fx * s[0][0], fy * s[0][1], s[0][2]), v3(fx * s[1][0], fy * s[1][1], s[1][2])];
            let global = [place(local[0][0], local[0][1]), place(local[1][0], local[1][1])];
            let mut plate = None;
            if let Some(nl) = normals.get(k) {
                let (nx, ny) = (fx * nl[0], fy * nl[1]);
                let (gx, gy) = (cy * nx - sy * ny, sy * nx + cy * ny);
                let h = gx.hypot(gy);
                let len = (global[1][0] - global[0][0]).hypot(global[1][1] - global[0][1]);
                if h > 1e-6 && len > 0.0 {
                    let mid = [0.5 * (global[0][0] + global[1][0]), 0.5 * (global[0][1] + global[1][1])];
                    plate = plates.beside(mid, [gx, gy], len, Tie::First).map(|i| {
                        let e = &plates.elements[i];
                        (e.body, e.element)
                    });
                }
            }
            out.push(BodySide { local, global, plate });
        }
    }
    Ok(out)
}

/// The waterline sides of every body in the plane that border the free surface (body_sides, the
/// sides under a plate left out), body after body: the one list the free surface's waterline
/// points and line integrals are made from.
fn waterline_segments(sides: &[Vec<BodySide>]) -> Vec<WlSeg> {
    let mut out = Vec::new();
    for (bi, s) in sides.iter().enumerate() {
        out.extend(s.iter().filter(|x| x.plate.is_none()).map(|x| WlSeg { body: bi, p: x.global }));
    }
    out
}

/// A convex polygon of the plane z = 0 (counter-clockwise vertices).
type Poly = Vec<[f64; 2]>;

/// The arcs of the circle |x − c0| = r outside the polygons: (θ0, θ1), θ0 < θ1, in [0, 2π).
fn free_arcs(c0: [f64; 2], r: f64, polys: &[Poly]) -> Vec<(f64, f64)> {
    let tau = 2.0 * PI;
    let mut cuts: Vec<f64> = Vec::new();
    for poly in polys {
        let n = poly.len();
        for k in 0..n {
            let (p, q) = (poly[k], poly[(k + 1) % n]);
            // |p + t (q − p) − c0| = r, t in [0, 1]
            let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
            let (fx, fy) = (p[0] - c0[0], p[1] - c0[1]);
            let (aa, bb, cc) = (dx * dx + dy * dy, 2.0 * (fx * dx + fy * dy), fx * fx + fy * fy - r * r);
            let disc = bb * bb - 4.0 * aa * cc;
            if disc < 0.0 {
                continue;
            }
            for t in [(-bb - disc.sqrt()) / (2.0 * aa), (-bb + disc.sqrt()) / (2.0 * aa)] {
                if (0.0..=1.0).contains(&t) {
                    cuts.push((fy + t * dy).atan2(fx + t * dx).rem_euclid(tau));
                }
            }
        }
    }
    let at = |t: f64| [c0[0] + r * t.cos(), c0[1] + r * t.sin()];
    let free = |t: f64| polys.iter().all(|p| !inside_convex(p, at(t)));
    if cuts.is_empty() {
        return if free(0.0) { vec![(0.0, tau)] } else { Vec::new() };
    }
    cuts.sort_by(|x, y| x.partial_cmp(y).unwrap());
    cuts.dedup_by(|x, y| (*x - *y).abs() <= 1e-14);
    let mut out: Vec<(f64, f64)> = Vec::new();
    let n = cuts.len();
    for k in 0..n {
        let (t0, t1) = (cuts[k], if k + 1 < n { cuts[k + 1] } else { cuts[0] + tau });
        if t1 - t0 > 1e-14 && free(0.5 * (t0 + t1)) {
            match out.last_mut() {
                Some(last) if (last.1 - t0).abs() <= 1e-14 => last.1 = t1,
                _ => out.push((t0, t1)),
            }
        }
    }
    out
}

/// Points and weights of the free surface |x − c0| < R outside the convex polygons: Gauss panels
/// of about the panel size in r (broken at the radii of the polygons' vertices and of the feet
/// on their sides, where arcs begin or end) and along the free arcs.
#[allow(non_snake_case)] // the notation of the theory documents
fn cut_points(c0: [f64; 2], R: f64, polys: &[Poly], size: f64) -> Vec<(Vec3, f64)> {
    let (gx, gw) = crate::quadrature::gauss_legendre(4);
    let mut br = vec![0.0, R];
    let d0 = |x: [f64; 2]| ((x[0] - c0[0]).powi(2) + (x[1] - c0[1]).powi(2)).sqrt();
    for poly in polys {
        let n = poly.len();
        for k in 0..n {
            let (p, q) = (poly[k], poly[(k + 1) % n]);
            br.push(d0(p));
            let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
            let t = ((c0[0] - p[0]) * dx + (c0[1] - p[1]) * dy) / (dx * dx + dy * dy);
            if t > 0.0 && t < 1.0 {
                br.push(d0([p[0] + t * dx, p[1] + t * dy]));
            }
        }
    }
    br.retain(|e| *e >= 0.0 && *e <= R);
    br.sort_by(|x, y| x.partial_cmp(y).unwrap());
    br.dedup_by(|x, y| (*x - *y).abs() <= 1e-12 * R);
    let mut out = Vec::new();
    for w in br.windows(2) {
        let np = ((w[1] - w[0]) / size).ceil().max(1.0) as usize;
        for p in 0..np {
            let (a, b) =
                (w[0] + (w[1] - w[0]) * p as f64 / np as f64, w[0] + (w[1] - w[0]) * (p + 1) as f64 / np as f64);
            for (x, wx) in gx.iter().zip(&gw) {
                let r = 0.5 * (a + b) + 0.5 * (b - a) * x;
                let wr = 0.5 * (b - a) * wx * r;
                for (t0, t1) in free_arcs(c0, r, polys) {
                    let na = ((r * (t1 - t0)) / size).ceil().max(1.0) as usize;
                    for k in 0..na {
                        let (s0, s1) =
                            (t0 + (t1 - t0) * k as f64 / na as f64, t0 + (t1 - t0) * (k + 1) as f64 / na as f64);
                        for (y, wy) in gx.iter().zip(&gw) {
                            let t = 0.5 * (s0 + s1) + 0.5 * (s1 - s0) * y;
                            out.push((v3(c0[0] + r * t.cos(), c0[1] + r * t.sin(), 0.0), wr * 0.5 * (s1 - s0) * wy));
                        }
                    }
                }
            }
        }
    }
    out
}

/// The distance between two convex polygons that do not overlap (0 when they do).
fn poly_gap(a: &Poly, b: &Poly) -> f64 {
    if a.iter().any(|x| inside_convex(b, *x)) || b.iter().any(|x| inside_convex(a, *x)) {
        return 0.0;
    }
    let seg = |x: [f64; 2], p: [f64; 2], q: [f64; 2]| {
        let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
        let t = (((x[0] - p[0]) * dx + (x[1] - p[1]) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
        ((x[0] - p[0] - t * dx).powi(2) + (x[1] - p[1] - t * dy).powi(2)).sqrt()
    };
    let one = |a: &Poly, b: &Poly| {
        let n = b.len();
        a.iter()
            .map(|x| (0..n).map(|k| seg(*x, b[k], b[(k + 1) % n])).fold(f64::INFINITY, f64::min))
            .fold(f64::INFINITY, f64::min)
    };
    // (edges that cross with no vertex inside the other: a vertex-edge distance of zero is not
    // guaranteed; the boxes here are rectangles of parallel or rotated sides, checked both ways)
    one(a, b).min(one(b, a))
}

/// The free surface inside the circle of radius R about c0 by cells (points, weights): one body,
/// the free-surface part of its automatic control surface of radius R; several, that of each body
/// to a rectangle of its own (the box of its waterline in its axes, widened on every side by 0.45
/// of the narrowest gap to another body's box) and the rest by cut_points.
#[allow(non_snake_case)] // the notation of the theory documents
fn free_surface(m: &Problem, c0: [f64; 2], R: f64, size: f64) -> Result<Vec<(Vec3, f64)>> {
    let nb = m.bodies.len();
    // the waterline box of each body in its axes, and as a polygon in the plane
    let mut boxes = Vec::with_capacity(nb);
    for (bi, b) in m.bodies.iter().enumerate() {
        let mesh = &m.meshes[b.mesh as usize];
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for s in waterline_chords(mesh, -b.position[2], 1e-6) {
            for x in s {
                for (fx, fy) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
                    if (fx < 0.0 && mesh.isx == 0) || (fy < 0.0 && mesh.isy == 0) {
                        continue;
                    }
                    let y = [fx * x[0], fy * x[1]];
                    for k in 0..2 {
                        lo[k] = lo[k].min(y[k]);
                        hi[k] = hi[k].max(y[k]);
                    }
                }
            }
        }
        if !(lo[0].is_finite() && hi[0].is_finite()) {
            // (a submerged body: the cells are built about each body's waterline)
            return Err(Error::value(format!(
                "{}: second_order: the cells of the free surface need a waterline of every body (free_surface_method = 'panels' takes submerged bodies)",
                body_label(b)
            ))
            .at(at_body(bi as i32)));
        }
        boxes.push((lo, hi));
    }
    let place = |bi: usize, x: [f64; 2]| -> [f64; 2] {
        let b = &m.bodies[bi];
        let (c, s) = ((b.yaw * PI / 180.0).cos(), (b.yaw * PI / 180.0).sin());
        [b.position[0] + c * x[0] - s * x[1], b.position[1] + s * x[0] + c * x[1]]
    };
    let rect = |bi: usize, e: f64| -> Poly {
        let (lo, hi) = boxes[bi];
        [[lo[0] - e, lo[1] - e], [hi[0] + e, lo[1] - e], [hi[0] + e, hi[1] + e], [lo[0] - e, hi[1] + e]]
            .iter()
            .map(|x| place(bi, *x))
            .collect()
    };
    let mut fs = super::quiet(m);
    let mut polys: Vec<Poly> = Vec::new();
    for (bi, b) in m.bodies.iter().enumerate() {
        let mut spec = ControlSurfaceInput {
            kind: ControlSurfaceKind::Automatic,
            values: Vec::new(),
            panel_size: size,
            radius: R,
            depth: 0.25 * R,
            free_surface: true,
            polygon: Vec::new(),
            partitions: Vec::new(),
            label: "free surface".to_string(),
        };
        if nb > 1 {
            let mut gap = f64::INFINITY;
            for c in 0..nb {
                if c != bi {
                    gap = gap.min(poly_gap(&rect(bi, 0.0), &rect(c, 0.0)));
                }
            }
            if gap.is_nan() || gap <= 0.0 {
                return Err(Error::value(format!(
                    "{}: the potential QTF (second_order.qtf_potential_direct, qtf_potential_indirect): the box of the body's waterline meets another body's",
                    body_label(b)
                ))
                .at(at_body(bi as i32)));
            }
            let e = crate::env_hooks::validation_constant("FS_GAP_FACTOR") * gap;
            let poly = rect(bi, e);
            let far = poly.iter().map(|x| ((x[0] - c0[0]).powi(2) + (x[1] - c0[1]).powi(2)).sqrt()).fold(0.0, f64::max);
            if far >= R {
                return Err(Error::value(format!(
                    "second_order.free_surface_inner_radius: {} m does not contain {} with the free surface about it (the centre of the bodies to {:.3} m)",
                    R,
                    body_label(b),
                    far
                ))
                .at(at_body(bi as i32)));
            }
            polys.push(poly);
            // the box in the stored sector of the mesh, counter-clockwise (closed when nothing is reflected)
            let (lo, hi) = boxes[bi];
            let (x0, x1, y0, y1) = (lo[0] - e, hi[0] + e, lo[1] - e, hi[1] + e);
            let mesh = &m.meshes[b.mesh as usize];
            spec.radius = -1.0;
            spec.depth = 0.25 * (x1 - x0).max(y1 - y0);
            spec.polygon = match (mesh.isx != 0, mesh.isy != 0) {
                (true, true) => vec![[x1, 0.0], [x1, y1], [0.0, y1]],
                (true, false) => vec![[0.0, y0], [x1, y0], [x1, y1], [0.0, y1]],
                (false, true) => vec![[x1, 0.0], [x1, y1], [x0, y1], [x0, 0.0]],
                (false, false) => vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
            };
        }
        let gdf = control_surface_gdf(&spec, Some((&m.meshes[b.mesh as usize], 0.0 - b.position[2]))).map_err(|e| {
            Error::value(format!(
                "the potential QTF (second_order.qtf_potential_direct, qtf_potential_indirect): the free surface: {}",
                e.message()
            ))
        })?;
        let k = fs.meshes.len();
        fs.meshes.push(engine_mesh(gdf_desc(&gdf)?, k)?);
        fs.bodies[bi].control_surface = Some(k);
    }
    let mut pts: Vec<(Vec3, f64)> =
        control_surface_points(&fs, 0.0, 0.0, 0.0).into_iter().filter(|p| p.part == 1).map(|p| (p.x, p.w)).collect();
    if nb > 1 {
        pts.extend(cut_points(c0, R, &polys, size));
    }
    Ok(pts)
}

/// The free surface inside R from a mesh of panels in z = 0 (second_order.free_surface_*; global axes,
/// its symmetry expanded): n x n Gauss points of each panel's bilinear map with their weights
/// (n = 1: the centroid with the area), and R, the largest distance of a vertex from the origin.
fn mesh_free_surface(mesh: &Mesh, n: usize) -> (Vec<(Vec3, f64)>, f64) {
    let (gx, gw) = crate::quadrature::gauss_legendre(n.max(1) as i32);
    let mut pts = Vec::new();
    let mut r = 0.0f64;
    for rf in reflections(mesh.isx, mesh.isy) {
        for p in &mesh.panels {
            let v: [[f64; 2]; 4] = std::array::from_fn(|k| [rf.sx * p.raw[k][0], rf.sy * p.raw[k][1]]);
            for x in &v {
                r = r.max(x[0].hypot(x[1]));
            }
            if n <= 1 {
                pts.push((v3(rf.sx * p.centroid[0], rf.sy * p.centroid[1], 0.0), p.area));
                continue;
            }
            for (a, wa) in gx.iter().zip(&gw) {
                for (b, wb) in gx.iter().zip(&gw) {
                    // the bilinear map x(u, v) of the four vertices and its Jacobian
                    let sh =
                        [(1.0 - a) * (1.0 - b), (1.0 + a) * (1.0 - b), (1.0 + a) * (1.0 + b), (1.0 - a) * (1.0 + b)];
                    let du = [-(1.0 - b), 1.0 - b, 1.0 + b, -(1.0 + b)];
                    let dv = [-(1.0 - a), -(1.0 + a), 1.0 + a, 1.0 - a];
                    let (mut x, mut xu, mut xv) = ([0.0; 2], [0.0; 2], [0.0; 2]);
                    for k in 0..4 {
                        for d in 0..2 {
                            x[d] += 0.25 * sh[k] * v[k][d];
                            xu[d] += 0.25 * du[k] * v[k][d];
                            xv[d] += 0.25 * dv[k] * v[k][d];
                        }
                    }
                    let j = (xu[0] * xv[1] - xu[1] * xv[0]).abs();
                    pts.push((v3(x[0], x[1], 0.0), j * wa * wb));
                }
            }
        }
    }
    (pts, r)
}

/// The sectors the azimuthal counts of the outer zones are given for: 1 (the full circle), 2 or 4
/// with one or two symmetry planes of the whole model (every body's mesh symmetric about the plane,
/// its origin on it, its yaw a multiple of 180°).
fn sectors(m: &Problem) -> usize {
    let plane = |axis: usize| {
        m.bodies.iter().all(|b| {
            let ms = &m.meshes[b.mesh as usize];
            let flag = if axis == 0 { ms.isx } else { ms.isy };
            flag != 0 && b.position[axis].abs() <= 1e-9 && b.yaw.rem_euclid(180.0) <= 1e-9
        })
    };
    [plane(0), plane(1)].iter().map(|&p| if p { 2 } else { 1 }).product()
}

/// The quadrature zone (theory §12.9): n annuli of width step from the inner radius R, each with
/// Gauss-Legendre nodes across it by equally spaced azimuthal nodes (na a sector of the model's
/// symmetry, sectors(); the first half a spacing from θ = 0), weights r dr dθ, appended to the
/// free-surface points circle by circle. Returns the outer radius, the radii of the circles and
/// the nodes of each. kc: the shortest combined wavenumber 2k + k_Ω (the auto azimuthal nodes), ks:
/// the largest wavenumber (the auto step: two of its wavelengths).
#[allow(non_snake_case)] // the notation of the theory documents
fn quadrature_zone(
    m: &Problem,
    c0: [f64; 2],
    R: f64,
    pts: &mut Vec<(Vec3, f64)>,
    ks: f64,
    kc: f64,
) -> (f64, Vec<f64>, usize) {
    let n = m.opt.potential_qtf_annuli.max(0) as usize;
    if n == 0 {
        return (R, Vec::new(), 0);
    }
    let step = m.opt.potential_qtf_annulus_step.unwrap_or(2.0 * 2.0 * PI / ks);
    if !step.is_finite() {
        // the auto step without a frequency between 0 and inf (ks = 0): only a query of another
        // zone gets here (the solve and the quadrature zone's query need the frequencies), so the
        // zone is left out rather than put at infinite radii
        return (R, Vec::new(), 0);
    }
    let Ro = R + n as f64 * step;
    let ns = sectors(m);
    let nt = match m.opt.potential_qtf_azimuthal_nodes {
        Some(n) => ns * n as usize,
        None => ((1.2 * kc * Ro).ceil() as usize + 16).div_ceil(ns) * ns,
    };
    let (gx, gw) = crate::quadrature::gauss_legendre(m.opt.potential_qtf_radial_nodes);
    let mut radii = Vec::with_capacity(n * gx.len());
    for a in 0..n {
        let (r0, r1) = (R + a as f64 * step, R + (a + 1) as f64 * step);
        for (x, w) in gx.iter().zip(&gw) {
            let r = 0.5 * (r0 + r1) + 0.5 * (r1 - r0) * x;
            let wr = 0.5 * (r1 - r0) * w * r * 2.0 * PI / nt as f64;
            radii.push(r);
            for t in 0..nt {
                let th = 2.0 * PI * (t as f64 + 0.5) / nt as f64;
                pts.push((v3(c0[0] + r * th.cos(), c0[1] + r * th.sin(), 0.0), wr));
            }
        }
    }
    (Ro, radii, nt)
}

/// The radius about c0 of the bodies' wetted meshes (their vertices below the free surface,
/// reflected with each mesh's symmetry and placed; with a wall the images lie as far from c0 on the
/// wall): the outgoing series of the far field (theory §12.9) converge outside it.
fn body_radius(m: &Problem, c0: [f64; 2]) -> f64 {
    let mut r: f64 = 0.0;
    for b in &m.bodies {
        let mesh = &m.meshes[b.mesh as usize];
        let (cy, sy) = ((b.yaw * PI / 180.0).cos(), (b.yaw * PI / 180.0).sin());
        for rf in reflections(mesh.isx, mesh.isy) {
            for q in &mesh.panels {
                for v in q.raw.iter().filter(|v| v[2] + b.position[2] <= 1e-9) {
                    let (x, y) = (rf.sx * v[0], rf.sy * v[1]);
                    let (dx, dy) = (b.position[0] + cy * x - sy * y - c0[0], b.position[1] + sy * x + cy * y - c0[1]);
                    r = r.max(dx.hypot(dy));
                }
            }
        }
    }
    r
}

/// The free surface of the potential QTF as the solve integrates it (theory §12.8-§12.9).
#[allow(non_snake_case)] // the notation of the theory documents
pub(crate) struct Zones {
    pub c0: [f64; 2],                                // the centre of the circles
    pub(super) want_sum: bool,                       // the sum frequencies asked for
    pub(super) want_dif: bool,                       // the difference frequencies asked for
    pub(super) kf: Vec<f64>,                         // the wavenumbers of the model's frequencies
    pub(super) oms: Vec<f64>,                        // the distinct pair frequencies
    pub(super) ko: Vec<f64>,                         // their wavenumbers
    pub(super) ks_max: f64,                          // the largest wavenumber of all
    pub(super) wsegs: Vec<WlSeg>,                    // the waterline sides of the bodies by the free surface
    pub(super) sides: Vec<Vec<BodySide>>,            // every waterline side of each body (body_sides)
    pub(super) wsegs_all: Vec<([f64; 2], [f64; 2])>, // with a wall also those of their mirror images
    pub(super) wall: Option<usize>,                  // the axis normal to the one wall (wall_axis)
    pub panels: Vec<[[f64; 3]; 4]>,                  // the panels of the panelled zone (none for the cells)
    pub pts: Vec<(Vec3, f64)>,                       // the points and weights of the panelled and quadrature zones
    pub n_inner: usize,                              // the points of the panelled zone (the first ones)
    pub(super) inner_radius: f64,                    // the panelled zone's circle, the first of the fit
    pub(super) circles: usize,                       // the circles of the fit to the outer circle (Circles)
    pub(super) quad_radii: Vec<f64>,                 // the circles of the quadrature zone's points, in order
    pub(super) quad_nodes: usize,                    // the points of each (θ_t = 2π(t + ½)/n)
    pub outer_radius: f64,                           // the outer circle
    pub(super) order: usize,                         // the order M of the series beyond the outer circle
    pub(super) given_order: bool,                    // M given (the incident waves truncated at M)
    pub(super) fit_order: usize,                     // the order of the fitted series (every radius beyond R)
    pub(super) ring_points: usize,                   // the points of each ring of the far-field fit
    segments: bool,                                  // the outer circle's segments given
    pub line_points: usize,                          // the points of the line integral around the outer circle
}

impl Zones {
    /// The angle of point t of the line integral around the outer circle.
    pub fn line_angle(&self, t: usize) -> f64 {
        if self.segments {
            2.0 * PI * (t as f64 + 0.5) / self.line_points as f64
        } else {
            2.0 * PI * t as f64 / self.ring_points as f64
        }
    }
}

/// The zones of the free surface of the model m (theory §12.8-§12.9): the frequencies decide the
/// auto sizes and the order.
#[allow(non_snake_case)] // the notation of the theory documents
pub(crate) fn zones(m: &Problem) -> Result<Zones> {
    // the validation hooks of the zones' constants refused (also for the queries of the zones)
    crate::env_hooks::environment_hooks()?;
    let g = m.env.gravity;
    let h = m.env.depth;
    let NBd = m.bodies.len();
    let omegas: Vec<f64> = m.omegas.iter().copied().filter(|w| *w > 0.0 && w.is_finite()).collect();
    let F = omegas.len();
    // the centre of the circle: the body origin, the mean of the origins of several bodies; the
    // global origin with a free-surface mesh (second_order.free_surface_*); with a wall that mean projected
    // onto the wall (the centre of the bodies and their mirror images)
    let fsm = m.opt.free_surface_mesh;
    let wall = wall_axis(m);
    if wall.is_some() && (fsm.is_some() || !m.opt.potential_qtf_panels) {
        return Err(Error::value(
            "second_order: with a wall the free surface is the generated panels (free_surface_method = 'panels')",
        ));
    }
    let mut c0 = if fsm.is_some() {
        [0.0, 0.0]
    } else {
        [
            m.bodies.iter().map(|b| b.position[0]).sum::<f64>() / NBd as f64,
            m.bodies.iter().map(|b| b.position[1]).sum::<f64>() / NBd as f64,
        ]
    };
    if let Some(ax) = wall {
        c0[ax] = 0.0;
    }
    // the frequencies asked for: 0 both, 1 the differences, 2 the sums
    let (want_sum, want_dif) =
        (m.opt.qtf_frequencies != QtfFrequencies::Difference, m.opt.qtf_frequencies != QtfFrequencies::Sum);
    // the wavenumbers of the first order and of the pair frequencies
    let kf = omegas.iter().map(|&w| wavenumber(w, h, g)).collect::<Result<Vec<f64>>>()?;
    let k1 = kf.iter().copied().fold(0.0, f64::max);
    let mut oms: Vec<f64> = Vec::new();
    for a in 0..F {
        for bb in 0..F {
            let s = omegas[a] + omegas[bb];
            let d = omegas[a] - omegas[bb];
            for (o, want) in [(s, want_sum), (d, want_dif)] {
                if want && o > 0.0 && !oms.iter().any(|x| (x - o).abs() <= 1e-9 * o) {
                    oms.push(o);
                }
            }
        }
    }
    oms.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let ko: Vec<f64> = oms.iter().map(|&w| wavenumber(w, h, g)).collect::<Result<Vec<f64>>>()?;
    let ks_max = ko.iter().copied().fold(k1, f64::max);
    // the radius and the cell size, given or auto (theory §12.12): a, the largest distance from the
    // centre to a waterline; R = 4 a (at 3 a a small difference frequency was 1.5 % off the
    // semi-analytic solution, from 4 a within 0.6 %); the cell a third of the shortest combined wavelength
    // 2 pi/(2 k + k_Omega), at most a/4
    let plates = PlateIndex::new(crate::plate_geometry::plate_elements(m));
    let sides: Vec<Vec<BodySide>> = (0..NBd).map(|bi| body_sides(m, bi, &plates)).collect::<Result<_>>()?;
    let wsegs = waterline_segments(&sides);
    let mut wsegs_all: Vec<([f64; 2], [f64; 2])> = wsegs.iter().map(|s| (s.p[0], s.p[1])).collect();
    if let Some(ax) = wall {
        let mirror = |x: [f64; 2]| if ax == 0 { [-x[0], x[1]] } else { [x[0], -x[1]] };
        wsegs_all.extend(wsegs.iter().map(|s| (mirror(s.p[0]), mirror(s.p[1]))));
    }
    let a_wl =
        wsegs_all.iter().flat_map(|&(p, q)| [p, q]).map(|x| (x[0] - c0[0]).hypot(x[1] - c0[1])).fold(0.0, f64::max);
    let ko_max = ko.iter().copied().fold(0.0, f64::max);
    let mesh_fs = fsm.map(|fi| mesh_free_surface(&m.meshes[fi], m.opt.free_surface_points as usize));
    let R = match &mesh_fs {
        Some((_, r)) => *r,
        None => m.opt.potential_qtf_radius.unwrap_or(4.0 * a_wl),
    };
    if mesh_fs.is_some() && (R.is_nan() || R <= a_wl) {
        return Err(Error::value(format!(
            "second_order: the farthest vertex ({R} m from the origin) is not beyond the waterlines ({a_wl} m)"
        )));
    }
    let size = match m.opt.potential_qtf_panel_size {
        Some(size) => size,
        None => (2.0 * PI / (3.0 * (2.0 * k1 + ko_max))).min(0.25 * a_wl),
    };
    if !(R.is_finite() && R > 0.0 && size.is_finite() && size > 0.0) {
        return Err(Error::value(
            "the potential QTF (second_order.qtf_potential_direct, qtf_potential_indirect): the bodies have no waterline (auto radius and cell size)",
        ));
    }
    // the panelled zone: the given mesh's panels (its symmetry expanded), the generated panels or
    // the cells (no panels to show: points only)
    let mut panels: Vec<[[f64; 3]; 4]> = Vec::new();
    let mut pts = match mesh_fs {
        Some((p, _)) => {
            let mesh = &m.meshes[fsm.expect("the mesh of the free surface")];
            for rf in reflections(mesh.isx, mesh.isy) {
                for q in &mesh.panels {
                    panels.push(std::array::from_fn(|k| [rf.sx * q.raw[k][0], rf.sy * q.raw[k][1], 0.0]));
                }
            }
            p
        }
        None if m.opt.potential_qtf_panels => {
            let made = crate::free_surface_panels::panels(&wsegs_all, c0, R, size)?;
            panels = made.iter().map(|q| q.map(|x| [x[0], x[1], 0.0])).collect();
            made.iter()
                .map(|q| {
                    let (c, a) = crate::free_surface_panels::centroid_area(q);
                    (v3(c[0], c[1], 0.0), a)
                })
                .collect()
        }
        None => free_surface(m, c0, R, size)?,
    };
    let n_inner = pts.len();
    // the free surface ends at the plates: a point of the given mesh (or of the cells) on a plate
    // would integrate the forcing over water the plate covers
    if let Some((x, e)) = pts.iter().find_map(|(x, _)| plates.holding([x[0], x[1]]).map(|i| (x, &plates.elements[i]))) {
        return Err(Error::value(format!(
            "second_order: the free surface of the potential QTF covers {} free-surface plate element {} at ({}, {}) m: it must end at the plates",
            body_label(&m.bodies[e.body]),
            e.element,
            x[0],
            x[1]
        )));
    }
    // beyond the panelled zone the fields come from their values on circles (theory §12.9), as
    // outgoing series that converge only outside the bodies' circle about c0
    let rb = body_radius(m, c0);
    if R <= rb {
        return Err(Error::value(format!(
            "second_order: the inner radius ({R} m) must exceed the radius of the bodies about the centre ({rb} m): beyond it the fields are outgoing series"
        )));
    }
    let Rin = R;
    // the quadrature zone beyond the inner free surface; from here on R is the outer circle
    let (R, quad_radii, quad_nodes) = quadrature_zone(m, c0, R, &mut pts, ks_max, 2.0 * k1 + ko_max);
    // the order of the series beyond the outer circle: given, or kR + 15; the incident waves to M
    // given, else 2M + 1 (so that their products with the orders <= M are exact trigonometric
    // polynomials, theory §12.9). The fitted series (the quadrature zone, the outer circle) to the
    // larger of M and kR + 15, well above the fields' bandwidth at the circles (about k a plus a
    // few orders, a the bodies' radius)
    let given_order = m.opt.potential_qtf_order.is_some();
    let auto = ((ks_max * R).ceil() as usize + 15).min(160);
    let M = m.opt.potential_qtf_order.map_or(auto, |n| n as usize);
    let Mf = M.max(auto);
    // the circles of the fit across the quadrature zone: their number for an interpolation in 1/r
    // to the rounding of the fields (the remainder of the wave smooth in 1/r; the interpolation
    // converges geometrically in n at a rate set by R_OC/R_I: n = 8 reached 1e-7 at 5/3, 10-12 at
    // 3, 12-14 at 6, theory §12.9)
    let circles = if R > Rin { 6 + (4.0 * (R / Rin).log2()).ceil() as usize } else { 1 };
    let NR = 4 * Mf + 8;
    // the line integral around the outer circle: NR points, or the middles of NL given segments
    let segs = m.opt.potential_qtf_segments.is_some();
    let NL = m.opt.potential_qtf_segments.map_or(NR, |n| sectors(m) * n as usize);
    Ok(Zones {
        c0,
        want_sum,
        want_dif,
        kf,
        oms,
        ko,
        ks_max,
        wsegs,
        sides,
        wsegs_all,
        wall,
        panels,
        pts,
        n_inner,
        inner_radius: Rin,
        circles,
        quad_radii,
        quad_nodes,
        outer_radius: R,
        order: M,
        given_order,
        fit_order: Mf,
        ring_points: NR,
        segments: segs,
        line_points: NL,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_free_surface_between_rectangles() {
        // the area and the second moment of a disk of radius 10 less three rectangles: one about
        // the centre, one cut by the circles about the centre, one turned by 30°
        let rect = |x0: f64, x1: f64, y0: f64, y1: f64| -> Poly { vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]] };
        let (c, s) = ((PI / 6.0).cos(), (PI / 6.0).sin());
        let turned: Poly = rect(-1.0, 1.0, -0.5, 0.5)
            .iter()
            .map(|x| [-5.0 + c * x[0] - s * x[1], -3.0 + s * x[0] + c * x[1]])
            .collect();
        let polys = vec![rect(-1.0, 1.0, -0.5, 0.5), rect(3.0, 5.0, -1.0, 2.0), turned];
        let pts = cut_points([0.0, 0.0], 10.0, &polys, 0.3);
        let area: f64 = pts.iter().map(|p| p.1).sum();
        let x2: f64 = pts.iter().map(|p| p.0[0] * p.0[0] * p.1).sum();
        // ∫ x² over [x0, x1] × [y0, y1]; over the turned 2 × 1 rectangle about (−5, −3):
        // its area 2 times (25 + its own ∫x²/area in the turned frame)
        let rx2 = |x0: f64, x1: f64, y0: f64, y1: f64| (x1.powi(3) - x0.powi(3)) / 3.0 * (y1 - y0);
        let ixx = 2.0 * (c * c * (2.0 * 2.0) / 12.0 + s * s * (1.0 * 1.0) / 12.0);
        let a0 = PI * 100.0 - 2.0 - 6.0 - 2.0;
        let x0 = PI * 1e4 / 4.0 - rx2(-1.0, 1.0, -0.5, 0.5) - rx2(3.0, 5.0, -1.0, 2.0) - (2.0 * 25.0 + ixx);
        assert!((area - a0).abs() <= 1e-5 * a0, "{area} {a0}"); // (the square root at a tangent radius)
        assert!((x2 - x0).abs() <= 1e-5 * x0, "{x2} {x0}");
        assert!(pts.iter().all(|p| polys.iter().all(|q| !inside_convex(q, [p.0[0], p.0[1]]))));
        assert!((poly_gap(&polys[0], &polys[1]) - 2.0).abs() < 1e-12);
    }
}
