//! Group ho_a (ho_geometry, ho_static, ho_sheets): the thin-sheet integrals and the static integrals of a hand-built higher-order geometry.

use hydrobem_core::internals::geometry::{Panel, Vec3, cross, dot, make_panel, norm, v3};
use hydrobem_core::internals::ho_geometry::{build_blocks, fill_point, open_knots};
use hydrobem_core::internals::ho_internal::{HoGeom, Inst, PatchDisc, Pt, StaticRow, block_of};
use hydrobem_core::internals::ho_static::static_row;
use hydrobem_core::internals::low_order::SurfaceKind;
use hydrobem_core::internals::panel_integrals::{
    PanelGeom, log_singularity_grad, make_panel_geom, psi_derivatives, rankine, rankine_grad,
};
use hydrobem_core::internals::patch_geometry::PatchGeo;
use hydrobem_core::internals::quadrature::gauss_legendre;

fn geom_from(v: &[f64; 12]) -> PanelGeom {
    let mut p = Panel::default();
    make_panel(v, &mut p);
    make_panel_geom(&p.proj, &p.normal, &p.centroid, p.area)
}

/// Thin sheets: gradients of the Rankine integrals and the psi
/// derivatives against central differences; the finite part on the panel.
#[test]
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn sheet_integrals() {
    let v = [0.1, -0.2, -1.0, 1.3, 0.0, -1.4, 1.1, 0.9, -1.2, -0.1, 0.7, -0.8];
    let p = geom_from(&v);
    let pts = [
        v3(0.6, 0.4, -0.3),
        v3(0.2, 0.3, -2.5),
        v3(3.0, -1.0, -1.1),
        v3(30.0, 5.0, -4.0),
        p.c + 0.3 * p.n,
        p.c + (-0.05) * p.n,
    ];
    let (mut worstS, mut worstD) = (0.0f64, 0.0f64);
    for x in &pts {
        let (gS, gD) = rankine_grad(&p, x);
        let hstep = 1e-5;
        for a in 0..3 {
            let (mut xp, mut xm) = (*x, *x);
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

/// A hand-built higher-order geometry: one flat square patch (side 2, centre (0, 0, -3), its
/// vertices in the GDF order), 2 x 2 panels of quadratic B-splines.
fn square_geom<'a>() -> HoGeom<'a> {
    let mut h = HoGeom::default();
    let g = PatchGeo {
        flat: true,
        vt: [v3(-1.0, -1.0, -3.0), v3(1.0, -1.0, -3.0), v3(1.0, 1.0, -3.0), v3(-1.0, 1.0, -3.0)],
        ..PatchGeo::default()
    };
    h.geo.push(g);
    h.disc.push(PatchDisc { nu: 2, nv: 2, ku: 3, kv: 3, iquo: 3, iqui: 4, Tu: open_knots(2, 3), Tv: open_knots(2, 3) });
    h.Np = 1;
    h.NB = 1;
    h.pbody = vec![0];
    h.prec = vec![1];
    h.rc = vec![Vec3::ZERO];
    h.kind = vec![SurfaceKind::Wet];
    h.pss = vec![-1];
    h.pdmp = vec![-1];
    h.lidx = vec![-1];
    h.inst.push(Inst::default());
    h.rows = vec![0];
    h.dof0 = vec![0];
    h.col0 = vec![0];
    h.blk0 = vec![0];
    h.Ndof = h.disc[0].nb();
    h.Ncol = h.Ndof;
    h.depth = f64::INFINITY;
    let (gx, gw) = gauss_legendre(4);
    for iv in 0..2 {
        for iu in 0..2 {
            h.inner0.push(h.inner.len() as i32);
            for a in 0..4 {
                for c in 0..4 {
                    let u = -1.0 + 0.5 * ((2 * iu + 1) as f64 + gx[a]);
                    let v = -1.0 + 0.5 * ((2 * iv + 1) as f64 + gx[c]);
                    let mut pt = Pt::default();
                    fill_point(&h, 0, iu, iv, u, v, gw[a] * gw[c] * 0.25, true, &mut pt);
                    h.inner.push(pt);
                }
            }
        }
    }
    h.inner0.push(h.inner.len() as i32);
    h.bc = vec![Vec3::ZERO; 4];
    h.bs = vec![0.0; 4];
    build_blocks(&mut h);
    h
}

/// The static row of a target far from the square and of one on it: the source sums (partition of
/// unity of the basis) against the point-source and the analytic values.
#[test]
fn static_row_of_a_square() {
    let h = square_geom();
    assert_eq!(block_of(&h, 0, 1, 1), 3);
    assert!((h.bs[0] - 0.5f64.sqrt()).abs() < 1e-14);
    // far: the source of the patch and its free-surface image (+ ordinary, - at omega = inf)
    let xt = v3(40.0, 10.0, -5.0);
    let mut row = StaticRow::default();
    static_row(&h, &xt, -1, 0.0, 0.0, false, true, &mut row);
    let rs: f64 = row.rs.iter().sum();
    let rsi: f64 = row.rsi.iter().sum();
    let (gx, gw) = gauss_legendre(20);
    let (mut fs, mut fsi) = (0.0, 0.0);
    for i in 0..20 {
        for j in 0..20 {
            let (a, b) = (1.0 / norm(xt - v3(gx[i], gx[j], -3.0)), 1.0 / norm(xt - v3(gx[i], gx[j], 3.0)));
            fs += gw[i] * gw[j] * (a + b);
            fsi += gw[i] * gw[j] * (a - b);
        }
    }
    assert!((rs - fs).abs() < 1e-8 * fs, "{rs} {fs}");
    assert!((rsi - fsi).abs() < 1e-6 * fsi.abs(), "{rsi} {fsi}");
    // the heave mode: ms[2] = int n_z / r over the patch
    assert!((row.ms[2] - rs * h.inner[0].nm[2]).abs() < 1e-10 * rs);
    assert!(row.lcol.is_empty() && row.ls.is_empty());
    // on the patch at its centre: int 1/r over the square of side 2 = 8 asinh(1), plus the image
    let mut row = StaticRow::default();
    static_row(&h, &v3(0.0, 0.0, -3.0), 0, 0.0, 0.0, false, false, &mut row);
    let rs: f64 = row.rs.iter().sum();
    let mut img = 0.0;
    for q in &h.inner {
        img += q.w / norm(q.x - v3(0.0, 0.0, 3.0));
    }
    let exact = 8.0 * 1.0f64.asinh() + img;
    assert!((rs - exact).abs() < 1e-6 * exact, "{rs} {exact}");
}
