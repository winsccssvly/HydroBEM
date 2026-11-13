//! Surface quadrature of mesh elements (body-local coordinates).
//!
//! Gauss-Legendre rules (Abramowitz & Stegun 25.4.29; Newton iteration from x_i =
//! cos(pi (i + 3/4) / (n + 1/2)), the leading term of A&S 22.16.6) and the quadrature of the
//! elements for the hydrostatics.

use crate::mesh_desc::PatchKind;
use std::borrow::Cow;
use std::sync::OnceLock;

use crate::constants::PI;
use crate::geometry::{Reflection, Vec3, cross, v3};
use crate::mesh::{Mesh, patch_eval};
use crate::patch_geometry::bilinear_map;

/// Gauss-Legendre (nodes, weights) on [-1, 1] (Newton iteration on P_n).
pub fn gauss_legendre(n: i32) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n.max(0) as usize];
    let mut w = vec![0.0; n.max(0) as usize];
    let pi = PI;
    let nf = n as f64;
    for i in 0..((n + 1) / 2) as usize {
        let mut z = (pi * (i as f64 + 0.75) / (nf + 0.5)).cos();
        let mut dp;
        for _it in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let kf = k as f64;
                let p2 = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            dp = nf * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() <= 1e-16 {
                break;
            }
        }
        // recompute the derivative at the converged node
        let (mut p0, mut p1) = (1.0, z);
        for k in 2..=n {
            let kf = k as f64;
            let p2 = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf;
            p0 = p1;
            p1 = p2;
        }
        dp = nf * (z * p1 - p0) / (z * z - 1.0);
        let nn = n as usize;
        x[i] = -z;
        x[nn - 1 - i] = z;
        let wi = 2.0 / ((1.0 - z * z) * dp * dp);
        w[i] = wi;
        w[nn - 1 - i] = wi;
    }
    if n % 2 == 1 {
        x[(n / 2) as usize] = 0.0;
    }
    (x, w)
}

/// The largest Gauss-Legendre order that gauss_nodes holds.
pub const GAUSS_CACHED: i32 = 16;

/// The Gauss-Legendre nodes (weights: true, the weights) of order n, 1 <= n <= GAUSS_CACHED,
/// computed once and shared by the panel integrals of both methods.
pub fn gauss_nodes(n: i32, weights: bool) -> &'static [f64] {
    static TABLE: OnceLock<Vec<(Vec<f64>, Vec<f64>)>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = vec![(Vec::new(), Vec::new()); (GAUSS_CACHED + 1) as usize];
        for k in 1..=GAUSS_CACHED {
            t[k as usize] = gauss_legendre(k);
        }
        t
    });
    assert!((1..=GAUSS_CACHED).contains(&n), "gauss_nodes: order {n} outside 1..16");
    let e = &table[n as usize];
    if weights { &e.1 } else { &e.0 }
}

/// The Gauss-Legendre rule of order n >= 1: the shared one of gauss_nodes up to GAUSS_CACHED (the
/// same numbers), else computed. Returns (nodes, weights).
pub fn gauss_rule(n: i32) -> (Cow<'static, [f64]>, Cow<'static, [f64]>) {
    if (1..=GAUSS_CACHED).contains(&n) {
        return (Cow::Borrowed(gauss_nodes(n, false)), Cow::Borrowed(gauss_nodes(n, true)));
    }
    let (x, w) = gauss_legendre(n);
    (Cow::Owned(x), Cow::Owned(w))
}

/// The first-kind Chebyshev nodes t_i = cos(pi (i + 1/2) / N), i = 0..N-1, of the interpolation of
/// the per-frequency tiles.
#[allow(clippy::needless_range_loop)] // index notation of the formula
pub fn chebyshev_nodes<const N: usize>() -> [f64; N] {
    let nf = N as f64;
    let mut t = [0.0f64; N];
    for i in 0..N {
        t[i] = (PI * (i as f64 + 0.5) / nf).cos();
    }
    t
}

/// The cosines cos(pi j (i + 1/2) / N) of the discrete cosine transform from the values at the
/// first-kind nodes to the Chebyshev coefficients: [j][i].
#[allow(clippy::needless_range_loop)] // index notation of the formula
pub fn chebyshev_cosines<const N: usize>() -> [[f64; N]; N] {
    let nf = N as f64;
    let mut cs = [[0.0f64; N]; N];
    for j in 0..N {
        for i in 0..N {
            cs[j][i] = (PI * j as f64 * (i as f64 + 0.5) / nf).cos();
        }
    }
    cs
}

/// The power-basis coefficients of the Chebyshev polynomials, T_j(t) = sum_k a[j][k] t^k, from
/// T_{j+1} = 2 t T_j - T_{j-1} (exact integers), N >= 2.
#[allow(clippy::needless_range_loop)] // index notation of the formula
pub fn chebyshev_power_basis<const N: usize>() -> [[f64; N]; N] {
    let mut a = [[0.0f64; N]; N];
    a[0][0] = 1.0;
    a[1][1] = 1.0;
    for j in 2..N {
        for k in 0..N {
            a[j][k] = (if k > 0 { 2.0 * a[j - 1][k - 1] } else { 0.0 }) - a[j - 2][k];
        }
    }
    a
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QPoint {
    pub x: Vec3,   // point
    pub nds: Vec3, // normal (into the fluid) times surface element
}

fn reflect_point(p: Vec3, r: &Reflection) -> Vec3 {
    v3(r.sx * p[0], r.sy * p[1], p[2])
}

/// Quadrature of one stored element reflected by r:
///
/// - low-order panel: the one-point area-centroid rule (theory §3.2);
/// - flat patch: 3 x 3 Gauss rule of the bilinear map;
/// - spline patch: 2k-point Gauss rule per knot span and direction.
///
/// The flat and spline rules integrate the hydrostatic integrands exactly.
/// mesh: low-order or patch mesh; element: zero-based stored element; r: reflection copy
/// (reflections()). Returns points with position and n dS (normal into the fluid).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > a)` also rejects NaN, unlike `x <= a`
pub fn record_quadrature(mesh: &Mesh, record: i32, r: &Reflection) -> Vec<QPoint> {
    let m = mesh;
    let rec = record;
    let mut q = Vec::new();
    if !m.patch {
        let p = &m.panels[rec as usize];
        q.push(QPoint { x: reflect_point(p.centroid, r), nds: reflect_point(p.area * p.normal, r) });
        return q;
    }
    if m.geometry == PatchKind::Flat {
        let v = &m.vertices[12 * rec as usize..];
        let a = v3(v[0], v[1], v[2]);
        let b = v3(v[3], v[4], v[5]);
        let c = v3(v[6], v[7], v[8]);
        let d = v3(v[9], v[10], v[11]);
        let (gx, gw) = gauss_rule(3);
        for i in 0..3 {
            for j in 0..3 {
                // x(u,s) = sum N_i v_i with corners (-1,-1),(1,-1),(1,1),(-1,1); the vertices of a flat
                // patch follow the panel rule (reference/inputs-meshes §1: the right-hand rule points into the fluid),
                // so x_u x x_s is n dS
                let (mut x, mut xu, mut xs) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                bilinear_map(&[a, b, c, d], gx[i], gx[j], &mut x, &mut xu, &mut xs);
                q.push(QPoint { x: reflect_point(x, r), nds: reflect_point((gw[i] * gw[j]) * cross(xu, xs), r) });
            }
        }
        return q;
    }
    if m.geometry == PatchKind::Analytic {
        // smooth maps: a 12 x 12 Gauss rule on the parameter square; x_u x x_v of an analytic or a
        // spline patch points into the body (reference/inputs-meshes §2), so n dS is its negative
        let (gx, gw) = gauss_rule(12);
        for i in 0..gx.len() {
            for j in 0..gx.len() {
                let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                patch_eval(m, rec, gx[i], gx[j], &mut x, &mut xu, &mut xv);
                q.push(QPoint { x: reflect_point(x, r), nds: reflect_point((-gw[i] * gw[j]) * cross(xu, xv), r) });
            }
        }
        return q;
    }
    let ri = rec as usize;
    let (kug, kvg) = (m.orders[2 * ri], m.orders[2 * ri + 1]);
    let (nug, nvg) = (m.ncoef[2 * ri], m.ncoef[2 * ri + 1]);
    let ko = m.layout.knot_offset[ri];
    let tu = &m.knots[ko as usize..];
    let tv = &tu[(nug + kug) as usize..];
    let (u0, u1) = (tu[0], tu[(nug + kug - 1) as usize]);
    let (v0, v1) = (tv[0], tv[(nvg + kvg - 1) as usize]);
    // the integrands of a polynomial patch are polynomials in each span (the hydrostatic ones of
    // degree up to 4k - 5), which the 2k-point rule integrates exactly; a rational patch is
    // smooth, 12 points per span (as the analytic maps per patch)
    let rational = !m.weights.is_empty();
    let (gu, wu) = gauss_rule(if rational { 12.max(2 * kug) } else { 2 * kug });
    let (gv, wv) = gauss_rule(if rational { 12.max(2 * kvg) } else { 2 * kvg });
    for a in (kug - 1) as usize..nug as usize {
        if !(tu[a + 1] > tu[a]) {
            continue;
        }
        for b in (kvg - 1) as usize..nvg as usize {
            if !(tv[b + 1] > tv[b]) {
                continue;
            }
            for i in 0..gu.len() {
                for j in 0..gv.len() {
                    let uu = 0.5 * (tu[a] + tu[a + 1]) + 0.5 * (tu[a + 1] - tu[a]) * gu[i];
                    let vv = 0.5 * (tv[b] + tv[b + 1]) + 0.5 * (tv[b + 1] - tv[b]) * gv[j];
                    // patch_eval takes [-1, 1] parameters
                    let pu = 2.0 * (uu - u0) / (u1 - u0) - 1.0;
                    let pv = 2.0 * (vv - v0) / (v1 - v0) - 1.0;
                    let (mut x, mut xu, mut xv) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                    patch_eval(m, rec, pu, pv, &mut x, &mut xu, &mut xv);
                    // derivatives are per [-1,1] parameter; the weights carry
                    // the span lengths in [-1,1] units
                    let ju = (tu[a + 1] - tu[a]) / (u1 - u0);
                    let jv = (tv[b + 1] - tv[b]) / (v1 - v0);
                    let wgt = wu[i] * wv[j] * ju * jv;
                    q.push(QPoint { x: reflect_point(x, r), nds: reflect_point((-wgt) * cross(xu, xv), r) });
                }
            }
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_rules_integrate_polynomials() {
        for n in 1..=GAUSS_CACHED {
            let (x, w) = (gauss_nodes(n, false), gauss_nodes(n, true));
            for p in 0..2 * n {
                let s: f64 = x.iter().zip(w).map(|(xi, wi)| wi * xi.powi(p)).sum();
                let exact = if p % 2 == 1 { 0.0 } else { 2.0 / (p as f64 + 1.0) };
                assert!((s - exact).abs() < 1e-13, "n {n} p {p}: {s} vs {exact}");
            }
        }
    }
}
