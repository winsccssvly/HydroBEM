//! The panel integrals of the low-order method (test-plan A1): the Rankine and log-singularity integrals against the adaptive
//! quadrature reference, the float32 Gauss tiers of the production path, the in-plane log
//! integral, the thin-sheet gradients; with HB_TIMING the informational panel timings.

use std::time::Instant;

use hydrobem_core::internals::geometry::{Panel, Vec3, cross, dot, make_panel, norm, v3};
use hydrobem_core::internals::panel_integrals::{
    PanelGeom, log_in_plane, log_singularity, log_singularity_grad, make_panel_geom, psi_derivatives, rankine,
    rankine_grad, rankine_pair,
};
use hydrobem_core::internals::quadrature::gauss_legendre;

use super::{Mt64, data_lines, num, read_data};

fn geom_from(v: &[f64; 12]) -> PanelGeom {
    let mut p = Panel::default();
    let _ = make_panel(v, &mut p);
    make_panel_geom(&p.proj, &p.normal, &p.centroid, p.area)
}

/// Standard normal values by the Marsaglia polar method (the check is a bound over 12000 random
/// points, so any normal generator serves).
struct Normal {
    saved: Option<f64>,
}

impl Normal {
    fn next(&mut self, rng: &mut Mt64) -> f64 {
        if let Some(v) = self.saved.take() {
            return v;
        }
        loop {
            let x = rng.uniform(-1.0, 1.0);
            let y = rng.uniform(-1.0, 1.0);
            let r2 = x * x + y * y;
            if r2 > 1.0 || r2 == 0.0 {
                continue;
            }
            let m = (-2.0 * r2.ln() / r2).sqrt();
            self.saved = Some(x * m);
            return y * m;
        }
    }
}

// A1: Rankine and log-singularity panel integrals against adaptive quadrature.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn panel_reference() {
    let text = read_data("panel_reference.txt");
    let (mut worstR, mut worstL) = (0.0f64, 0.0f64);
    let mut n = 0;
    for (_, f) in data_lines(&text) {
        let kind = f[0];
        let mut v = [0.0f64; 12];
        for i in 0..12 {
            v[i] = num(f[1 + i]);
        }
        let p = v3(num(f[13]), num(f[14]), num(f[15]));
        let (a, b) = (num(f[16]), num(f[17]));
        let g = geom_from(&v);
        let dist = norm(p - g.c);
        let (S, D);
        if kind == "R" {
            (S, D) = rankine(&g, &p);
            let sS = g.area / (dist + g.radius);
            let sD = g.area / (dist * dist + g.radius * g.radius);
            worstR = worstR.max((S - a).abs() / a.abs().max(sS)).max((D - b).abs() / b.abs().max(sD));
        } else {
            (S, D) = log_singularity(&g, &p);
            let sL = g.area * ((dist + g.radius).ln().abs() + 1.0);
            let sD = g.area / (dist + g.radius);
            worstL = worstL.max((S - a).abs() / a.abs().max(sL)).max((D - b).abs() / b.abs().max(sD));
        }
        n += 1;
    }
    println!(
        "A1 panel integrals: {n} cases, max relative error Rankine {worstR:.2e}, log singularity {worstL:.2e} \
         (target 1e-6)"
    );
    assert!(n > 50, "A1 reference has the expected size");
    assert!(worstR <= 1e-6, "A1 Rankine integrals within 1e-6");
    assert!(worstL <= 1e-6, "A1 log-singularity integrals within 1e-6");
}

// Rankine source and dipole (S, D) of 1/r + 1/r' at x: the sums of the production rule at x and at
// its image x' = (x, y, -z).
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn with_image(p: &PanelGeom, x: &Vec3) -> (f64, f64) {
    let (S, D, Sb, Db) = rankine_pair(p, x);
    (S + Sb, D + Db)
}

// A1, single-precision production path: rankine_pair summed (Gauss tiers in
// float32) against the double-precision closed forms at the point and its
// image, over random points at 0.5 to 80 panel radii (target 1e-5).
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn rankine_production_path() {
    let mut rng = Mt64::new(11);
    let mut nd = Normal { saved: None };
    let panels: [[f64; 12]; 3] = [
        [0.0, 0.0, -0.4, 0.2, 0.0, -0.4, 0.2, 0.15, -0.4, 0.0, 0.12, -0.4],
        [1.0, 0.0, -0.1, 1.0, 0.3, -0.1, 1.0, 0.3, 0.0, 1.0, 0.0, 0.0], // vertical, at the surface
        [0.0, 0.0, -1.0, 0.5, 0.02, -1.1, 0.4, 0.06, -1.05, 0.05, 0.04, -1.0],
    ];
    let mut worst = 0.0f64;
    for v in &panels {
        let g = geom_from(v);
        for _ in 0..4000 {
            let d = v3(nd.next(&mut rng), nd.next(&mut rng), nd.next(&mut rng));
            let r = (0.5 + 80.0 * rng.uniform(0.0, 1.0) * rng.uniform(0.0, 1.0)) * g.radius / norm(d);
            let mut x = g.c + r * d;
            x[2] = -x[2].abs(); // field points in the fluid
            let (S, D) = with_image(&g, &x);
            let (s1, d1) = rankine(&g, &x);
            let (s2, d2) = rankine(&g, &v3(x[0], x[1], -x[2]));
            let dist = norm(x - g.c);
            let sS = g.area / (dist + g.radius);
            let sD = g.area / (dist * dist + g.radius * g.radius);
            worst = worst
                .max((S - (s1 + s2)).abs() / (s1 + s2).abs().max(sS))
                .max((D - (d1 + d2)).abs() / (d1 + d2).abs().max(sD));
        }
    }
    println!("A1 production path (float32 Gauss tiers): max relative error {worst:.2e} (target 1e-5)");
    assert!(worst <= 1e-5, "A1 production path within 1e-5");
}

// The tiers one by one: each float32 Gauss rule at the near end of its range against the
// closed forms (the 1e-6 relative error of the rules, with the float32 margin of 1e-5).
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn rankine_gauss_tiers() {
    let g = geom_from(&[0.0, 0.0, -1.0, 0.5, 0.02, -1.1, 0.4, 0.06, -1.05, 0.05, 0.04, -1.0]);
    let mut worst = 0.0f64;
    for r in [2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0, 11.0, 13.0, 40.0] {
        for i in 0..64 {
            let t = i as f64 * 0.37;
            let d = v3(t.cos() * (0.3 * t).sin(), t.sin() * (0.3 * t).sin(), -(0.3 * t).cos());
            let x = g.c + (r * g.radius / norm(d)) * d;
            // only the term at x: its image is at least 2 |z| away, z <= -0.9
            let (S, D) = with_image(&g, &x);
            let (s1, d1) = rankine(&g, &x);
            let (s2, d2) = rankine(&g, &v3(x[0], x[1], -x[2]));
            let dist = norm(x - g.c);
            let sS = g.area / (dist + g.radius);
            let sD = g.area / (dist * dist + g.radius * g.radius);
            worst = worst
                .max((S - (s1 + s2)).abs() / (s1 + s2).abs().max(sS))
                .max((D - (d1 + d2)).abs() / (d1 + d2).abs().max(sD));
        }
    }
    println!("Gauss tiers of the production path from 2 to 40 radii: max relative error {worst:.2e} (target 1e-5)");
    assert!(worst <= 1e-5, "Gauss tiers within 1e-5");
}

// A1, in-plane log integral: int log|xi - x| dS for x in
// the plane of the panel, against signed triangles (x, edge) integrated with
// the Duffy map (the singular vertex at x) and composite 24-point Gauss rules
// (4 x 64 sub-intervals, for points close to an edge line).
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn log_in_plane_reference() {
    let (gx, gw) = gauss_legendre(24);
    let tri = |x: Vec3, a: Vec3, b: Vec3| {
        // xi = x + u (a - x) + u v (b - a), u, v in [0, 1], dS = u |(a - x) x (b - a)| du dv, signed
        let e1 = a - x;
        let e2 = b - a;
        let J = cross(e1, e2)[2];
        let mut s = 0.0;
        let (nu, nv) = (4, 64);
        for iu in 0..nu {
            for iv in 0..nv {
                for i in 0..24 {
                    for j in 0..24 {
                        let u = (iu as f64 + 0.5 * (gx[i] + 1.0)) / nu as f64;
                        let v = (iv as f64 + 0.5 * (gx[j] + 1.0)) / nv as f64;
                        let xi = x + u * e1 + (u * v) * e2;
                        let r = norm(xi - x);
                        s += 0.25 * gw[i] * gw[j] / (nu * nv) as f64 * u * J * r.ln();
                    }
                }
            }
        }
        s
    };
    let panels: [[f64; 12]; 3] = [
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 2.0, 0.1, 0.0, 2.0, 0.1, 0.0, 0.3, 0.9, 0.0],
        [-0.2, -0.1, 0.0, 0.7, -0.3, 0.0, 0.9, 0.6, 0.0, 0.1, 0.5, 0.0],
    ];
    let mut worst = 0.0f64;
    for v in &panels {
        let g = geom_from(v);
        let pts = [
            g.c,
            v3(g.c[0] + 0.3 * g.radius, g.c[1] - 0.2 * g.radius, 0.0),
            0.5 * (g.v[0] + g.v[1]),
            v3(g.c[0] + 3.0 * g.radius, g.c[1] + 0.5 * g.radius, 0.0),
            g.v[0] + v3(0.01, 0.0, 0.0),
        ];
        for x in pts {
            let mut rf = 0.0;
            for i in 0..g.nv as usize {
                rf += tri(x, g.v[i], g.v[(i + 1) % g.nv as usize]);
            }
            let got = log_in_plane(&g, &x);
            worst = worst.max((got - rf).abs() / rf.abs().max(g.area));
        }
    }
    println!("A1 in-plane log integral: max relative error {worst:.2e} (target 1e-6)");
    assert!(worst <= 1e-6, "A1 in-plane log integral");
}

// Thin sheets: gradients of the Rankine integrals and the psi
// derivatives against central differences; the finite part on the panel.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn sheet_integrals() {
    let p = geom_from(&[0.1, -0.2, -1.0, 1.3, 0.0, -1.4, 1.1, 0.9, -1.2, -0.1, 0.7, -0.8]);
    let pts = [
        v3(0.6, 0.4, -0.3),
        v3(0.2, 0.3, -2.5),
        v3(3.0, -1.0, -1.1),
        v3(30.0, 5.0, -4.0),
        p.c + 0.3 * p.n,
        p.c + (-0.05) * p.n,
    ];
    let (mut worstS, mut worstD) = (0.0f64, 0.0f64);
    for x in pts {
        let (gS, gD) = rankine_grad(&p, &x);
        let hstep = 1e-5;
        for a in 0..3 {
            let (mut xp, mut xm) = (x, x);
            xp[a] += hstep;
            xm[a] -= hstep;
            let (sp, dp) = rankine(&p, &xp);
            let (sm, dm) = rankine(&p, &xm);
            worstS = worstS.max(((sp - sm) / (2.0 * hstep) - gS[a]).abs() / (gS[a].abs() + 1e-3));
            worstD = worstD.max(((dp - dm) / (2.0 * hstep) - gD[a]).abs() / (gD[a].abs() + 1e-3));
        }
    }
    // the finite part at the centroid equals the limit from both sides
    let (_, gD0) = rankine_grad(&p, &p.c);
    let (_, gDp) = rankine_grad(&p, &(p.c + 1e-7 * p.n));
    let (_, gDm) = rankine_grad(&p, &(p.c + (-1e-7) * p.n));
    let fp = (dot(gD0, p.n) - dot(gDp, p.n)).abs().max((dot(gD0, p.n) - dot(gDm, p.n)).abs());
    // psi derivatives
    let (mut worstK, mut worstH) = (0.0f64, 0.0f64);
    let nx = (1.0 / 3.0f64.sqrt()) * v3(1.0, -1.0, 1.0);
    for x in [v3(0.6, 0.4, -0.3), v3(2.0, -1.0, -0.05), v3(0.3, 0.2, -2.0)] {
        let xi = v3(0.4, 0.1, -0.7);
        let psi = |a: Vec3, b: Vec3| {
            let R = (a[0] - b[0]).hypot(a[1] - b[1]);
            let t = -(a[2] + b[2]);
            ((R * R + t * t).sqrt() + t).ln()
        };
        let (k, h) = psi_derivatives(&x, &xi, &nx, &p.n);
        let e = 1e-4;
        let kf = (psi(x + e * nx, xi) - psi(x + (-e) * nx, xi)) / (2.0 * e);
        let hf = (psi(x + e * nx, xi + e * p.n) - psi(x + e * nx, xi + (-e) * p.n) - psi(x + (-e) * nx, xi + e * p.n)
            + psi(x + (-e) * nx, xi + (-e) * p.n))
            / (4.0 * e * e);
        worstK = worstK.max((kf - k).abs() / (k.abs() + 1e-3));
        worstH = worstH.max((hf - h).abs() / (h.abs() + 1e-3));
    }
    // panel integrals of the psi derivatives against a fine tensor Gauss rule
    let mut worstL = 0.0f64;
    for x in [v3(0.6, 0.4, -0.3), v3(3.0, 1.0, -0.4)] {
        let (Lk, Lh) = log_singularity_grad(&p, &x, &nx);
        let (g, w) = gauss_legendre(40);
        let (mut rk, mut rh) = (0.0, 0.0);
        for i in 0..40 {
            for j in 0..40 {
                let (u, vv) = (g[i], g[j]);
                let n0 = 0.25 * (1.0 - u) * (1.0 - vv);
                let n1 = 0.25 * (1.0 + u) * (1.0 - vv);
                let n2 = 0.25 * (1.0 + u) * (1.0 + vv);
                let n3 = 0.25 * (1.0 - u) * (1.0 + vv);
                let xi = n0 * p.v[0] + n1 * p.v[1] + n2 * p.v[2] + n3 * p.v[3];
                let xu = (0.25 * (1.0 - vv)) * (p.v[1] - p.v[0]) + (0.25 * (1.0 + vv)) * (p.v[2] - p.v[3]);
                let xv = (0.25 * (1.0 - u)) * (p.v[3] - p.v[0]) + (0.25 * (1.0 + u)) * (p.v[2] - p.v[1]);
                let (k, h) = psi_derivatives(&x, &xi, &nx, &p.n);
                let jac = norm(cross(xu, xv));
                rk += w[i] * w[j] * jac * k;
                rh += w[i] * w[j] * jac * h;
            }
        }
        worstL = worstL.max((Lk - rk).abs() / rk.abs()).max((Lh - rh).abs() / rh.abs());
    }
    println!(
        "sheet integrals: grad S {worstS:.2e}, grad D {worstD:.2e}, finite part {fp:.2e}, psi' {worstK:.2e}, \
         psi'' {worstH:.2e}, int {worstL:.2e}"
    );
    assert!(worstS <= 1e-6 && worstD <= 1e-6, "gradients of the Rankine integrals");
    assert!(fp <= 1e-6 * dot(gD0, p.n).abs(), "finite part of the hypersingular integral");
    assert!(worstK <= 1e-6 && worstH <= 1e-5, "psi derivatives");
    assert!(worstL <= 1e-8, "integrals of the psi derivatives");
}

// Informational timing of the panel integrals (exact and far-field forms), HB_TIMING only.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn panel_timings() {
    if std::env::var_os("HB_TIMING").is_none() {
        return;
    }
    let g = geom_from(&[0.0, 0.0, -1.0, 0.2, 0.0, -1.0, 0.2, 0.15, -1.0, 0.0, 0.12, -1.0]);
    let n = 1 << 18;
    let mut sink = 0.0;
    for dist in [0.5, 5.0] {
        // panel radius about 0.12: near and far tier
        let t0 = Instant::now();
        for i in 0..n {
            let fi = i as f64;
            let (S, D) =
                rankine(&g, &v3(0.1 + dist * (fi * 0.001).cos(), 0.07, -1.0 - dist * (fi * 0.001).sin() * 0.5));
            sink += S + D;
        }
        let ns = t0.elapsed().as_nanos() as f64 / n as f64;
        println!("rankine at {dist:.1}: {ns:.1} ns");
        let t0 = Instant::now();
        for i in 0..n / 16 {
            let fi = i as f64;
            let (L, Ld) = log_singularity(&g, &v3(0.1 + dist * (fi * 0.01).cos(), 0.07, 0.3 + 0.1 * (fi * 0.01).sin()));
            sink += L + Ld;
        }
        let ns2 = t0.elapsed().as_nanos() as f64 / (n / 16) as f64;
        println!("log singularity at {dist:.1}: {ns2:.1} ns (checksum {sink:.3})");
    }
}

// The characters of the reflection groups and the local reflection a global one induces.
#[test]
fn characters_and_induced_reflections() {
    use hydrobem_core::internals::geometry::{Reflection, reflections};
    use hydrobem_core::internals::lo_geometry::{characters, induced_reflection};
    use hydrobem_core::internals::problem::Body;
    assert_eq!(characters(&reflections(0, 0)), vec![vec![1.0]]);
    assert_eq!(characters(&reflections(1, 0)), vec![vec![1.0, 1.0], vec![1.0, -1.0]]);
    let c4 = characters(&reflections(1, 1));
    assert_eq!(c4.len(), 4);
    for a in 0..4 {
        for b in 0..4 {
            // orthogonality of the characters of the Klein four-group
            let s: f64 = (0..4).map(|g| c4[a][g] * c4[b][g]).sum();
            assert_eq!(s, if a == b { 4.0 } else { 0.0 });
        }
    }
    let sx = Reflection { sx: -1.0, sy: 1.0 };
    let sy = Reflection { sx: 1.0, sy: -1.0 };
    let mut l = Reflection { sx: 0.0, sy: 0.0 };
    let mut b = Body::default();
    // straight: the global x reflection is the local x one
    assert!(induced_reflection(&b, 1, 0, &sx, &mut l) && l == Reflection { sx: -1.0, sy: 1.0 });
    assert!(!induced_reflection(&b, 0, 1, &sx, &mut l));
    // turned by 90 degrees (also -270): the global x reflection is the local y one
    b.yaw = -270.0;
    assert!(induced_reflection(&b, 0, 1, &sx, &mut l) && l == Reflection { sx: 1.0, sy: -1.0 });
    assert!(induced_reflection(&b, 1, 0, &sy, &mut l) && l == Reflection { sx: -1.0, sy: 1.0 });
    // off the symmetry plane, or at another angle: none
    b.position[0] = 1.0;
    assert!(!induced_reflection(&b, 1, 1, &sx, &mut l));
    assert!(induced_reflection(&b, 1, 1, &sy, &mut l));
    b.position[0] = 0.0;
    b.yaw = 45.0;
    assert!(!induced_reflection(&b, 1, 1, &sx, &mut l));
    assert!(induced_reflection(&b, 1, 1, &Reflection { sx: 1.0, sy: 1.0 }, &mut l));
}
