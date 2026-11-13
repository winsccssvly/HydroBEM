//! Frequency-independent panel integrals of the low-order method:
//!  - Rankine source and normal-dipole integrals of a constant-strength plane
//!    polygon (Newman 1986, eqs. 2.1, 3.10; Hess & Smith 1964): rankine() exact
//!    within 12 radii and by a 3 x 3 Gauss rule beyond; the production path
//!    (rankine_any) exact within 2 radii and by float32 AVX2 Gauss tiers of
//!    orders 6, 5, 4, 3 from 2, 3, 4, 8 radii;
//!  - integrals of the logarithmic singularity psi = log(|xi - p| + p_z - xi_z)
//!    of the wave kernel and of its normal derivative.
//!
//! Why float32 Gauss tiers beyond 2 radii and the closed forms within: theory §6.1 (grounds);
//! why the log term is split off with a constant coefficient and when its correction is applied:
//! theory §6.2 (grounds).

use std::arch::x86_64::*;

use crate::env_hooks::diag_double;
use crate::geometry::{COINCIDENCE_TOL, Vec3, cross, dot, norm, v3};
use crate::patch_geometry::bilinear_map;
use crate::quadrature::gauss_nodes;

#[derive(Clone, Debug, Default)]
pub struct PanelGeom {
    pub nv: i32,      // distinct vertices (3 or 4)
    pub v: [Vec3; 4], // distinct projected vertices, counter-clockwise about n
    pub c: Vec3,      // area centroid
    pub n: Vec3,      // unit normal (into the fluid)
    pub area: f64,
    pub radius: f64,    // max |v - c|
    pub elen: [f64; 4], // edge lengths
    pub em: [Vec3; 4],  // outward in-plane unit normals of the edges
    pub gx: [Vec3; 9],  // 3 x 3 Gauss points of the bilinear map
    pub gw: [f64; 9],   // weights (area elements)
    // Gauss rules of orders 3, 4, 5, 6 for the far and intermediate
    // Rankine tiers: points relative to c and weights, float32, structure of
    // arrays, each rule padded to a multiple of 8 points with zero weight
    pub tq: Vec<f32>, // [x | y | z | w] blocks of all rules
    pub toff: [i32; 4],
    pub tlen: [i32; 4],
}

// The bilinear map of a quadrilateral, corners at (-1,-1), (1,-1), (1,1), (-1,1).
struct Bilinear([Vec3; 4]);

impl Bilinear {
    // the point and the Jacobian at (u, v)
    fn eval(&self, u: f64, v: f64) -> (Vec3, f64) {
        let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        bilinear_map(&self.0, u, v, &mut x, &mut xu, &mut xv);
        (x, norm(cross(xu, xv)))
    }
}

// The bilinear map of a panel's distinct vertices (a triangle with its last vertex repeated).
fn panel_map(p: &PanelGeom) -> Bilinear {
    let q3 = if p.nv == 4 { p.v[3] } else { p.v[2] };
    Bilinear([p.v[0], p.v[1], p.v[2], q3])
}

// the sub-cells of an adaptive split, in a fixed order (the sum is that of this order)
fn quarters(u0: f64, u1: f64, v0: f64, v1: f64) -> ([[f64; 2]; 4], [[f64; 2]; 4]) {
    let um = 0.5 * (u0 + u1);
    let vm = 0.5 * (v0 + v1);
    ([[u0, um], [um, u1], [um, u1], [u0, um]], [[v0, vm], [v0, vm], [vm, v1], [vm, v1]])
}

// Adaptive quadrisection of the 4 x 4 Gauss rule of the bilinear map over the parameter square,
// for N integrals of f at once: a cell is split into its quarters until their sum agrees with
// the cell's own value (done(sum, cell)), at most 14 levels deep.
struct Adaptive<'a, F, D> {
    map: &'a Bilinear,
    f: F,
    done: D,
}

impl<F, D, const N: usize> Adaptive<'_, F, D>
where
    F: Fn(&Vec3) -> [f64; N],
    D: Fn(&[f64; N], &[f64; N]) -> bool,
{
    // the 4 x 4 Gauss rule on the cell [u0, u1] x [v0, v1]
    fn cell(&self, u0: f64, u1: f64, v0: f64, v1: f64) -> [f64; N] {
        let x = gauss_nodes(4, false);
        let w = gauss_nodes(4, true);
        let mut s = [0.0f64; N];
        let hu = 0.5 * (u1 - u0);
        let hv = 0.5 * (v1 - v0);
        for i in 0..4 {
            for j in 0..4 {
                let (xi, jac) = self.map.eval(u0 + hu * (x[i] + 1.0), v0 + hv * (x[j] + 1.0));
                let ww = w[i] * w[j] * hu * hv * jac;
                let fv = (self.f)(&xi);
                for k in 0..N {
                    s[k] += ww * fv[k];
                }
            }
        }
        s
    }

    // adds to acc the integrals over the cell, whose own rule gave prev
    #[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
    fn adapt(&self, u0: f64, u1: f64, v0: f64, v1: f64, prev: &[f64; N], depth: i32, acc: &mut [f64; N]) {
        let um = 0.5 * (u0 + u1);
        let vm = 0.5 * (v0 + v1);
        let c = [
            self.cell(u0, um, v0, vm),
            self.cell(um, u1, v0, vm),
            self.cell(um, u1, vm, v1),
            self.cell(u0, um, vm, v1),
        ];
        let sum: [f64; N] = std::array::from_fn(|k| c[0][k] + c[1][k] + c[2][k] + c[3][k]);
        if depth >= 14 || (self.done)(&sum, prev) {
            for k in 0..N {
                acc[k] += sum[k];
            }
            return;
        }
        let (us, vs) = quarters(u0, u1, v0, v1);
        for q in 0..4 {
            self.adapt(us[q][0], us[q][1], vs[q][0], vs[q][1], &c[q], depth + 1, acc);
        }
    }

    // the rule on the whole square
    fn whole(&self) -> [f64; N] {
        self.cell(-1.0, 1.0, -1.0, 1.0)
    }

    // the adaptive rule over the whole square from its own value whole
    fn refine(&self, whole: &[f64; N]) -> [f64; N] {
        let mut acc = [0.0f64; N];
        self.adapt(-1.0, 1.0, -1.0, 1.0, whole, 0, &mut acc);
        acc
    }

    // the integrals over the whole square
    fn integrate(&self) -> [f64; N] {
        self.refine(&self.whole())
    }
}

/// Builds the panel geometry from projected vertices (4 entries, possibly a
/// repeated vertex) and the validated panel data.
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn make_panel_geom(proj4: &[Vec3; 4], normal: &Vec3, centroid: &Vec3, area: f64) -> PanelGeom {
    let q = proj4;
    let mut g = PanelGeom { n: *normal, c: *centroid, area, ..Default::default() };
    let mut diam: f64 = 0.0;
    for i in 0..4 {
        for j in i + 1..4 {
            diam = diam.max(norm(q[i] - q[j]));
        }
    }
    for i in 0..4 {
        if norm(q[i] - q[(i + 1) % 4]) > COINCIDENCE_TOL * diam {
            g.v[g.nv as usize] = q[i];
            g.nv += 1;
        }
    }
    let nv = g.nv as usize;
    for i in 0..nv {
        g.radius = g.radius.max(norm(g.v[i] - *centroid));
    }
    for e in 0..nv {
        let edge = g.v[(e + 1) % nv] - g.v[e];
        g.elen[e] = norm(edge);
        g.em[e] = (1.0 / g.elen[e]) * cross(edge, *normal);
    }
    let map = Bilinear(*q);
    let x = gauss_nodes(3, false);
    let w = gauss_nodes(3, true);
    let mut sum = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let jac;
            (g.gx[3 * i + j], jac) = map.eval(x[i], x[j]);
            g.gw[3 * i + j] = w[i] * w[j] * jac;
            sum += g.gw[3 * i + j];
        }
    }
    // the bilinear map of a plane quadrilateral covers exactly the panel; the
    // weights are rescaled to the exact area to remove round-off
    for v in g.gw.iter_mut() {
        *v *= area / sum;
    }
    const ORDERS: [i32; 4] = [3, 4, 5, 6];
    let mut off: usize = 0;
    for t in 0..4 {
        let n = ORDERS[t] as usize;
        let cnt = n * n;
        let padded = cnt.div_ceil(8) * 8;
        g.toff[t] = off as i32;
        g.tlen[t] = padded as i32;
        g.tq.resize(off + 4 * padded, 0.0f32);
        let xn = gauss_nodes(n as i32, false);
        let wn = gauss_nodes(n as i32, true);
        let mut wsum = 0.0;
        let mut ww = vec![0.0f64; cnt];
        let mut pts = vec![Vec3::ZERO; cnt];
        for i in 0..n {
            for j in 0..n {
                let jac;
                (pts[i * n + j], jac) = map.eval(xn[i], xn[j]);
                ww[i * n + j] = wn[i] * wn[j] * jac;
                wsum += ww[i * n + j];
            }
        }
        for k in 0..cnt {
            g.tq[off + k] = (pts[k][0] - centroid[0]) as f32;
            g.tq[off + padded + k] = (pts[k][1] - centroid[1]) as f32;
            g.tq[off + 2 * padded + k] = (pts[k][2] - centroid[2]) as f32;
            g.tq[off + 3 * padded + k] = (ww[k] * area / wsum) as f32;
        }
        off += 4 * padded;
    }
    g
}

/// Source  S = int dS / |x - xi|  and dipole  D = int n.(x - xi) / |x - xi|^3 dS
/// (the solid angle of the panel seen from x, positive on the side of n; the
/// principal value 0 for x on the panel).
/// p: panel geometry from make_panel_geom(); x: field point. Returns (S, D): the source and the
/// dipole integral.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
pub fn rankine(p: &PanelGeom, x: &Vec3) -> (f64, f64) {
    let rc = *x - p.c;
    let dist = norm(rc);
    // the diagnostic mode keeps the closed forms to 1000 radii: farther their edge terms cancel to
    // about eps (r/a)^2 of the result, while the 3 x 3 rule errs by (a/r)^6
    if dist > (if diag_double() { 1000.0 } else { FAR_RATIO }) * p.radius {
        let (mut S, mut D) = (0.0, 0.0);
        for k in 0..9 {
            let r = *x - p.gx[k];
            let ir = 1.0 / norm(r);
            S += p.gw[k] * ir;
            D += p.gw[k] * dot(p.n, r) * ir * ir * ir;
        }
        return (S, D);
    }
    let w = dot(rc, p.n);
    let nv = p.nv as usize;
    let mut av = [Vec3::ZERO; 4];
    let mut r = [0.0f64; 4];
    for i in 0..nv {
        av[i] = p.v[i] - *x;
        r[i] = norm(av[i]);
    }
    // solid angle (Van Oosterom & Strackee 1983 for each fan triangle); with
    // the vertices counter-clockwise about n it is positive for w > 0
    let mut omega = 0.0;
    if w.abs() > 1e-12 * p.radius {
        for i in 1..nv - 1 {
            let (a, b, c) = (av[0], av[i], av[i + 1]);
            let num = dot(a, cross(b, c));
            let den = r[0] * r[i] * r[i + 1] + dot(a, b) * r[i + 1] + dot(a, c) * r[i] + dot(b, c) * r[0];
            omega -= 2.0 * num.atan2(den);
        }
    }
    let mut s = 0.0;
    for e in 0..nv {
        let f = (e + 1) % nv;
        let d = dot(av[e], p.em[e]); // in-plane distance to the edge line
        if d.abs() <= 1e-14 * p.radius {
            continue;
        }
        let q = r[e] * r[f] + dot(av[e], av[f]); // (ra + rb)^2 - len^2 = 2 q
        if !(q > 0.0) {
            continue;
        }
        let sum = r[e] + r[f] + p.elen[e];
        s += d * (sum * sum / (2.0 * q)).ln();
    }
    (s - w * omega, omega)
}

/// (psi, d psi / dn): psi = log(|xi - p| + p_z - xi_z) at xi for the image point p, and its
/// derivative along n.
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn psi_terms(xi: &Vec3, p: &Vec3, n: &Vec3) -> (f64, f64) {
    let d = *xi - *p;
    let R = norm(d);
    let den = R - d[2]; // = R + |d_z|, d_z <= 0 in the fluid
    (den.ln(), (dot(d, *n) / R - n[2]) / den)
}

// Gauss tier of a field point at r panel radii, or -1 for the exact closed
// forms; 1e-6 relative (measured 9.7e-7 by scripts/panel_rule_error_map.py),
// ten times inside the 1e-5 of the single-precision production path (theory §6.1).
#[inline]
fn rankine_tier(r: f64) -> i32 {
    if r >= 8.0 {
        0
    } else if r >= 4.0 {
        1
    } else if r >= 3.0 {
        2
    } else if r >= EXACT_RATIO {
        3
    } else {
        -1
    }
}

#[repr(align(32))]
struct Lanes([f32; 8]);

// Gauss rule of tier t at the point x, float32 AVX2 (theory §6.1: <= 1e-5 in
// the single-precision path). The point is taken relative to the centroid in
// double first, so the float differences carry no cancellation; the
// reciprocal square root is refined by one Newton step (about 1e-7).
#[inline]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn rankine_gauss(p: &PanelGeom, x: &Vec3, t: usize) -> (f64, f64) {
    let len = p.tlen[t] as usize;
    let q = &p.tq[p.toff[t] as usize..p.toff[t] as usize + 4 * len];
    let mut sv = Lanes([0.0; 8]);
    let mut dv = Lanes([0.0; 8]);
    // SAFETY: the crate is built with AVX2 and FMA (enforced in lib.rs); the loads read 8 floats
    // inside the 4 len blocks of q (len is a multiple of 8), the stores the aligned lanes.
    unsafe {
        let px = _mm256_set1_ps((x[0] - p.c[0]) as f32);
        let py = _mm256_set1_ps((x[1] - p.c[1]) as f32);
        let pz = _mm256_set1_ps((x[2] - p.c[2]) as f32);
        let nx = _mm256_set1_ps(p.n[0] as f32);
        let ny = _mm256_set1_ps(p.n[1] as f32);
        let nz = _mm256_set1_ps(p.n[2] as f32);
        let half = _mm256_set1_ps(0.5f32);
        let three = _mm256_set1_ps(3.0f32);
        let mut sacc = _mm256_setzero_ps();
        let mut dacc = _mm256_setzero_ps();
        let qp = q.as_ptr();
        let mut k = 0;
        while k < len {
            let rx = _mm256_sub_ps(px, _mm256_loadu_ps(qp.add(k)));
            let ry = _mm256_sub_ps(py, _mm256_loadu_ps(qp.add(len + k)));
            let rz = _mm256_sub_ps(pz, _mm256_loadu_ps(qp.add(2 * len + k)));
            let w = _mm256_loadu_ps(qp.add(3 * len + k));
            let r2 = _mm256_fmadd_ps(rz, rz, _mm256_fmadd_ps(rx, rx, _mm256_mul_ps(ry, ry)));
            let mut ir = _mm256_rsqrt_ps(r2);
            ir = _mm256_mul_ps(_mm256_mul_ps(half, ir), _mm256_fnmadd_ps(_mm256_mul_ps(r2, ir), ir, three));
            let nr = _mm256_fmadd_ps(nz, rz, _mm256_fmadd_ps(nx, rx, _mm256_mul_ps(ny, ry)));
            sacc = _mm256_fmadd_ps(w, ir, sacc);
            dacc = _mm256_fmadd_ps(_mm256_mul_ps(w, nr), _mm256_mul_ps(ir, _mm256_mul_ps(ir, ir)), dacc);
            k += 8;
        }
        _mm256_store_ps(sv.0.as_mut_ptr(), sacc);
        _mm256_store_ps(dv.0.as_mut_ptr(), dacc);
    }
    let (mut S, mut D) = (0.0, 0.0);
    for k in 0..8 {
        S += sv.0[k] as f64;
        D += dv.0[k] as f64;
    }
    (S, D)
}

#[inline]
fn rankine_any(p: &PanelGeom, x: &Vec3) -> (f64, f64) {
    let t = if diag_double() { -1 } else { rankine_tier(norm(*x - p.c) / p.radius) };
    if t < 0 { rankine(p, x) } else { rankine_gauss(p, x, t as usize) }
}

/// The Rankine terms at x (S, D) and at its image x' = (x, y, -z) (Sb, Db); returns
/// (S, D, Sb, Db).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn rankine_pair(p: &PanelGeom, x: &Vec3) -> (f64, f64, f64, f64) {
    let (S, D) = rankine_any(p, x);
    let (Sb, Db) = rankine_any(p, &v3(x[0], x[1], -x[2]));
    (S, D, Sb, Db)
}

/// L = int psi dS and Ld = int d psi / d n_xi dS for the image point p
/// (p_z >= 0) of a field point; requires p off the panel.
/// p: panel geometry; image: the image point (x, y, -z) of the field point;
/// returns (L, Ld): the source integral of psi and its normal-derivative integral.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn log_singularity(p: &PanelGeom, image: &Vec3) -> (f64, f64) {
    let dist = norm(*image - p.c);
    if !diag_double() && dist > FAR_RATIO * p.radius {
        let (mut L, mut Ld) = (0.0, 0.0);
        for k in 0..9 {
            let (psi, dpsi) = psi_terms(&p.gx[k], image, &p.n);
            L += p.gw[k] * psi;
            Ld += p.gw[k] * dpsi;
        }
        return (L, Ld);
    }
    let map = panel_map(p);
    // psi is analytic on the panel (its singularity is the image point above
    // the free surface); the n x n Gauss order by distance r (in panel radii), measured by
    // scripts/panel_rule_error_map.py (L <= 5.2e-9, Ld <= 1.2e-7 on their scales);
    // adaptive for r < 1, and for 1 <= r < 2 when the image point is less than half a radius
    // above the panel's highest vertex
    let mut top = p.v[0][2];
    for i in 1..p.nv as usize {
        top = top.max(p.v[i][2]);
    }
    let r = dist / p.radius;
    let clearance = (image[2] - top) / p.radius;
    let mut order = 0;
    if clearance >= 0.5 || r >= 2.0 {
        order = if r >= 6.0 {
            4
        } else if r >= 3.0 {
            5
        } else if r >= 2.0 {
            6
        } else if r >= 1.5 {
            8
        } else if r >= 1.0 {
            10
        } else {
            0
        };
    }
    if order > 0 {
        let x = gauss_nodes(order, false);
        let w = gauss_nodes(order, true);
        let (mut L, mut Ld) = (0.0, 0.0);
        for i in 0..order as usize {
            for j in 0..order as usize {
                let (xi, jac) = map.eval(x[i], x[j]);
                let (psi, dpsi) = psi_terms(&xi, image, &p.n);
                let ww = w[i] * w[j] * jac;
                L += ww * psi;
                Ld += ww * dpsi;
            }
        }
        return (L, Ld);
    }
    // adaptive 4 x 4 Gauss on the bilinear map of the polygon, to 1e-13 of the integrals' scale
    // (the area times |log| and over the clearance); the clearance at least 1e-3 radii keeps the
    // tolerance finite for an image point on the panel's plane
    let dmin = (dist - p.radius).max(1e-3 * p.radius);
    let scale = (dist + p.radius).ln().abs() + 1.0;
    let (tol_l, tol_d) = (1e-13 * p.area * scale, 1e-13 * p.area / dmin);
    let lq = Adaptive {
        map: &map,
        f: |xi: &Vec3| {
            let (psi, dpsi) = psi_terms(xi, image, &p.n);
            [psi, dpsi]
        },
        done: |s: &[f64; 2], c: &[f64; 2]| (s[0] - c[0]).abs() <= tol_l && (s[1] - c[1]).abs() <= tol_d,
    };
    let [L, Ld] = lq.integrate();
    (L, Ld)
}

/// Thin sheets: gradients with respect to the field point
/// x of rankine(): gS = grad S (principal value 0 of the normal component on
/// the panel itself) and gD = grad D (the velocity of the vortex ring around
/// the panel edges: the Hadamard finite part on the panel itself); exact closed
/// forms within FAR_RATIO radii, the 3 x 3 Gauss rule beyond. x off the edges. Returns (gS, gD).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
pub fn rankine_grad(p: &PanelGeom, x: &Vec3) -> (Vec3, Vec3) {
    let rc = *x - p.c;
    let dist = norm(rc);
    let mut gS = Vec3::ZERO;
    let mut gD = Vec3::ZERO;
    if dist > FAR_RATIO * p.radius {
        for k in 0..9 {
            let r = *x - p.gx[k];
            let r2 = dot(r, r);
            let ir = 1.0 / r2.sqrt();
            let ir3 = ir / r2;
            let nr = dot(p.n, r);
            gS -= (p.gw[k] * ir3) * r;
            gD += (p.gw[k] * ir3) * (p.n - (3.0 * nr / r2) * r);
        }
        return (gS, gD);
    }
    let w = dot(rc, p.n);
    let nv = p.nv as usize;
    let mut av = [Vec3::ZERO; 4];
    let mut r = [0.0f64; 4];
    for i in 0..nv {
        av[i] = p.v[i] - *x;
        r[i] = norm(av[i]);
    }
    let mut omega = 0.0;
    if w.abs() > 1e-12 * p.radius {
        for i in 1..nv - 1 {
            let (a, b, c) = (av[0], av[i], av[i + 1]);
            let num = dot(a, cross(b, c));
            let den = r[0] * r[i] * r[i + 1] + dot(a, b) * r[i + 1] + dot(a, c) * r[i] + dot(b, c) * r[0];
            omega -= 2.0 * num.atan2(den);
        }
    }
    // grad S = -sum_e m_e L_e - n Omega (the terms d_e grad L_e - w grad Omega cancel);
    // grad Omega = -sum_e (a x b)(|a| + |b|) / (|a||b|(|a||b| + a.b)), a, b the edge ends
    // relative to x (Biot-Savart of the edges)
    for e in 0..nv {
        let f = (e + 1) % nv;
        let q = r[e] * r[f] + dot(av[e], av[f]);
        if !(q > 1e-14 * p.radius * p.radius) {
            continue; // x on the edge line segment
        }
        let sum = r[e] + r[f] + p.elen[e];
        gS -= (sum * sum / (2.0 * q)).ln() * p.em[e];
        gD -= ((r[e] + r[f]) / (r[e] * r[f] * q)) * cross(av[e], av[f]);
    }
    (gS - omega * p.n, gD)
}

/// Derivatives of a function f(R, t) of a point pair, R = |x_h - xi_h|,
/// t = -(z + zeta): k = n_x . grad_x f, h = n_x^T grad_x grad_xi f n_xi from
/// f_R, f_t, f_RR, f_R / R, f_Rt, f_tt (grad_x f = (f_R e, -f_t), grad_xi f =
/// (-f_R e, -f_t), e the horizontal unit vector from xi to x). Returns (k, h).
#[inline]
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn pair_derivatives(
    x: &Vec3,
    xi: &Vec3,
    nx: &Vec3,
    nxi: &Vec3,
    fR: f64,
    ft: f64,
    fRR: f64,
    fR_R: f64,
    fRt: f64,
    ftt: f64,
) -> (f64, f64) {
    let dx = x[0] - xi[0];
    let dy = x[1] - xi[1];
    let R = (dx * dx + dy * dy).sqrt();
    let e0 = if R > 0.0 { dx / R } else { 0.0 };
    let e1 = if R > 0.0 { dy / R } else { 0.0 };
    let ax = nx[0] * e0 + nx[1] * e1;
    let ab = nxi[0] * e0 + nxi[1] * e1;
    let k = fR * ax - ft * nx[2];
    // M_hh = -(f_RR e e^T + f_R/R (I - e e^T)), M_hz = -f_Rt e, M_zh = f_Rt e^T, M_zz = f_tt
    let h = -(fRR * ax * ab + fR_R * (nx[0] * nxi[0] + nx[1] * nxi[1] - ax * ab)) - fRt * ax * nxi[2]
        + fRt * nx[2] * ab
        + ftt * nx[2] * nxi[2];
    (k, h)
}

/// psi(x, xi) = log(rho + t), t = -(z + zeta) >= 0, rho = sqrt(R^2 + t^2): the
/// derivatives (k, h): n_x . grad_x psi (k) and n_x^T grad_x grad_xi psi n_xi (h).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn psi_derivatives(x: &Vec3, xi: &Vec3, nx: &Vec3, nxi: &Vec3) -> (f64, f64) {
    let dx = x[0] - xi[0];
    let dy = x[1] - xi[1];
    let R = (dx * dx + dy * dy).sqrt();
    let t = -(x[2] + xi[2]);
    let rho = (R * R + t * t).sqrt();
    let rt = rho + t;
    let fR = R / (rho * rt);
    let ft = 1.0 / rho;
    let ir3 = 1.0 / (rho * rho * rho);
    let ftt = -t * ir3;
    let fRt = -R * ir3;
    let fR_R = 1.0 / (rho * rt); // psi_R / R, also at R = 0
    let fRR = t * ir3 - fR_R; // psi is harmonic: psi_RR + psi_R/R + psi_tt = 0
    pair_derivatives(x, xi, nx, nxi, fR, ft, fRR, fR_R, fRt, ftt)
}

/// Lk = int n_x . grad_x psi dS, Lh = int n_x^T grad_x grad_xi psi n dS over the
/// panel (adaptive Gauss rule), for x below the free surface. Returns (Lk, Lh).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn log_singularity_grad(p: &PanelGeom, x: &Vec3, nx: &Vec3) -> (f64, f64) {
    let map = panel_map(p);
    // the singularity of psi is the image point (x, y, -z) above the free surface
    let xb = v3(x[0], x[1], -x[2]);
    let dmin = (norm(xb - p.c) - p.radius).max(1e-3 * p.radius);
    let (tol_k, tol_h) = (1e-11 * p.area / dmin, 1e-11 * p.area / (dmin * dmin));
    let gq = Adaptive {
        map: &map,
        f: |xi: &Vec3| {
            let (k, h) = psi_derivatives(x, xi, nx, &p.n);
            [k, h]
        },
        done: |s: &[f64; 2], c: &[f64; 2]| (s[0] - c[0]).abs() <= tol_k && (s[1] - c[1]).abs() <= tol_h,
    };
    let whole = gq.whole();
    // beyond 3 radii from the panel (4 from its centroid) the single cell is converged, as in
    // log_singularity_grad3 (the same singularity sets the Gauss convergence)
    let [Lk, Lh] = if dmin > 3.0 * p.radius { whole } else { gq.refine(&whole) };
    (Lk, Lh)
}

/// grad_x of int_panel psi dS, psi = log(rho + t), t = -(z + zeta) (the source
/// formulation): one adaptive pass for the three components; and the
/// gradient of psi at a point xi.
pub fn log_singularity_grad3(p: &PanelGeom, x: &Vec3) -> Vec3 {
    let map = panel_map(p);
    let xb = v3(x[0], x[1], -x[2]);
    let dmin = (norm(xb - p.c) - p.radius).max(1e-3 * p.radius);
    let tol = 1e-11 * p.area / dmin;
    let gq = Adaptive {
        map: &map,
        f: |xi: &Vec3| {
            let g = psi_gradient(x, xi);
            [g[0], g[1], g[2]]
        },
        done: |s: &[f64; 3], c: &[f64; 3]| norm(v3(s[0] - c[0], s[1] - c[1], s[2] - c[2])) <= tol,
    };
    let whole = gq.whole();
    // beyond 3 radii from the panel (4 from its centroid) the 4 x 4 Gauss rule is converged
    // (the Bernstein ellipse of the panel: an error of about 7e-8 of the term)
    let g = if dmin > 3.0 * p.radius { whole } else { gq.refine(&whole) };
    v3(g[0], g[1], g[2])
}

#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn psi_gradient(x: &Vec3, xi: &Vec3) -> Vec3 {
    let dx = x[0] - xi[0];
    let dy = x[1] - xi[1];
    let R2 = dx * dx + dy * dy;
    let t = -(x[2] + xi[2]);
    let rho = (R2 + t * t).sqrt();
    let f = 1.0 / (rho * (rho + t));
    v3(dx * f, dy * f, -1.0 / rho)
}

/// int log|xi - x| dS for x in the plane of the panel (exact: the polygon is
/// split into the signed triangles (x, edge), each integrated in polar
/// coordinates).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub fn log_in_plane(p: &PanelGeom, x: &Vec3) -> f64 {
    // triangle (x, a, b) at signed distance h from the edge line, tangential
    // coordinates t1, t2 of a and b from the foot of the perpendicular:
    //   (h/2) [F(t2) - F(t1)],  F(t) = (t log(h^2 + t^2) - 2t + 2h atan(t/h))/2 - t/2
    let nv = p.nv as usize;
    let mut sum = 0.0;
    for i in 0..nv {
        let a = p.v[i];
        let b = p.v[(i + 1) % nv];
        let h = dot(a - *x, p.em[i]); // > 0: x on the inner side of the edge
        if h == 0.0 {
            continue;
        }
        let e = (1.0 / p.elen[i]) * (b - a);
        let t1 = dot(a - *x, e);
        let t2 = dot(b - *x, e);
        let F = |t: f64| {
            let q = h * h + t * t;
            0.5 * (t * q.ln() - 2.0 * t + 2.0 * h * (t / h).atan()) - 0.5 * t
        };
        sum += 0.5 * h * (F(t2) - F(t1));
    }
    sum
}

/// Far-field switch: |x - c| > FAR_RATIO * radius; the
/// Rankine integrals use Gauss rules down to EXACT_RATIO radii. Why these distances
/// (theory §6.1, its grounds): each tier is the cheapest rule that keeps 1e-6 at its inner
/// boundary, and the 6 x 6 rule reaches that limit at 2 radii, so the closed forms take the
/// nearer pairs.
pub const FAR_RATIO: f64 = 12.0;
pub const EXACT_RATIO: f64 = 2.0;
