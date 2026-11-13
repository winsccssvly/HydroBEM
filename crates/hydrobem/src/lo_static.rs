//! The static (frequency-independent) influence data of the low-order method: the Rankine
//! integrals of the field points and their image terms against all panels and the corrections
//! of the log singularity, for f32 (production) and f64 (diagnostic mode).

use crate::codes::Stage;
use std::sync::atomic::{AtomicI32, Ordering};

use rayon::prelude::*;

use crate::error::Result;
use crate::geometry::{Vec3, dot, norm, v3};
use crate::low_order::{Geometry, Real, Static, StaticEntry, SurfaceKind};
use crate::panel_integrals::{
    PanelGeom, log_in_plane, log_singularity, log_singularity_grad, log_singularity_grad3, psi_derivatives,
    psi_gradient, psi_terms, rankine, rankine_grad, rankine_pair,
};
use crate::parallel::{par_rows, par_rows_opt};
use crate::progress::Work;

#[inline]
fn image(x: &Vec3) -> Vec3 {
    v3(x[0], x[1], -x[2])
}

// Field points of the Rankine terms: |x_p - xi| = r_p for the source xi
// (green_fin.rs): x, its bottom image, its free-surface image, and the
// shifts and double image of the bottom-surface reflection chain.
#[inline]
fn image_points(x: &Vec3, h: f64) -> [Vec3; 6] {
    [
        v3(x[0], x[1], x[2]),
        v3(x[0], x[1], -x[2] - 2.0 * h),
        v3(x[0], x[1], -x[2]),
        v3(x[0], x[1], x[2] + 2.0 * h),
        v3(x[0], x[1], x[2] - 2.0 * h),
        v3(x[0], x[1], -x[2] - 4.0 * h),
    ]
}

// Per-pair choice of the analytic log-singularity integral:
// the correction delta = int_j psi dS - A_j psi(c_j) it adds to the centroid value
// is bounded by |delta| / A_j <= LOG_BOUND (a_j / d)^2 (the second-order term of
// psi about c_j; LOG_BOUND = 0.6 a margin over the largest constant 0.56 that
// scripts/log_correction_bound.py measures), d = r' + |z + zeta| to the centroid, a_j the panel
// radius; the correction is applied where that bound exceeds the tolerance eps,
// d < a_j sqrt(LOG_BOUND / eps); eps = 0: every pair.
const LOG_BOUND: f64 = 0.6;
#[inline]
fn log_pair(p: &PanelGeom, x: &Vec3, tol: f64) -> bool {
    let dx = x[0] - p.c[0];
    let dy = x[1] - p.c[1];
    let zz = (x[2] + p.c[2]).abs();
    let d = (dx * dx + dy * dy + zz * zz).sqrt() + zz;
    d * d * tol < LOG_BOUND * p.radius * p.radius
}

#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub(crate) fn static_row<R: Real>(
    g: &Geometry,
    x: &Vec3,
    logint: bool,
    tol: f64,
    depth: f64,
    want_inf: bool,
    row: &mut [StaticEntry<R>],
    mut inf: Option<&mut [[R; 2]]>,
) {
    let xb = image(x);
    let finite = depth.is_finite();
    let xp = if finite { image_points(x, depth) } else { [Vec3::ZERO; 6] };
    for j in 0..g.Nf as usize {
        let p = &g.pg[j];
        // the Rankine terms of x and its images once, for the regular row and the omega = inf row
        // at x and its free-surface image x' (xp[2] of image_points), and in finite depth at
        // xp[1] with its image xp[3] and the bottom images xp[4], xp[5] (at least h away); the
        // terms s2, s4, s5, s6 are those of r_p = |xp[p - 1] - xi| (the one-based r_p below)
        let (s0, d0, sb, db) = rankine_pair(p, x);
        let mut sr = s0 + sb;
        let mut dr = d0 + db;
        let (mut s2, mut d2, mut s4, mut d4, mut s5, mut d5, mut s6, mut d6) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        if finite {
            (s2, d2, s4, d4) = rankine_pair(p, &xp[1]);
            (s5, d5) = rankine(p, &xp[4]);
            (s6, d6) = rankine(p, &xp[5]);
            sr += (s2 + s4) + s5 + s6;
            dr += (d2 + d4) + d5 + d6;
        }
        if want_inf {
            // 1/r1 + 1/r2 - (1/r3 + ... + 1/r6) (infinite depth: 1/r - 1/r')
            let si = s0 + s2 - sb - s4 - s5 - s6;
            let di = d0 + d2 - db - d4 - d5 - d6;
            if let Some(inf) = inf.as_deref_mut() {
                inf[j] = [R::from_f64(si), R::from_f64(di)];
            }
        }
        row[j].rs = R::from_f64(sr);
        row[j].rd = R::from_f64(dr);
        let use_log = logint && log_pair(p, x, tol);
        if use_log && g.on_surface(j as i32) && x[2] == 0.0 {
            // both on z = 0: psi = log |xi - x| integrated exactly in the plane; at
            // the panel's own centroid the centroid value is part of the kernel
            // limit (pair_kernel); the dipole of interior panels is K times the
            // source (G_zeta = K G on zeta = 0), so dd is not used
            let L = log_in_plane(p, x);
            let Rr = norm(p.c - *x);
            row[j].dl = R::from_f64(if Rr > 0.0 { L - p.area * Rr.ln() } else { L });
            row[j].dd = R::from_f64(0.0);
        } else if use_log {
            let (L, Ld) = log_singularity(p, &xb);
            let (psi, dpsi) = psi_terms(&p.c, &xb, &p.n);
            row[j].dl = R::from_f64(L - p.area * psi);
            row[j].dd = R::from_f64(Ld - p.area * dpsi);
        } else {
            row[j].dl = R::from_f64(0.0);
            row[j].dd = R::from_f64(0.0);
        }
    }
}

// Sheet row of the sector panel at x with normal nx: n_x . grad_x of
// the Rankine dipole (h) and source (k) integrals of every panel with the
// free-surface image x' = (x, y, -z) added (and subtracted for omega = inf),
// and the corrections of the log singularity relative to its centroid value.
// Finite depth: the six Rankine terms sum_p sign_p / r_p of green_fin.rs,
// image_points(), the bottom images added, at omega = inf with the signs
// (+, +, -, -, -, -).
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
fn sheet_row<R: Real>(
    g: &Geometry,
    x: &Vec3,
    nx: &Vec3,
    logint: bool,
    tol: f64,
    depth: f64,
    row: &mut [[R; 4]],
    mut inf: Option<&mut [[R; 2]]>,
) {
    let xb = image(x);
    let down = |n: &Vec3, v: &Vec3| n[0] * v[0] + n[1] * v[1] - n[2] * v[2];
    let finite = depth.is_finite();
    let xp = if finite { image_points(x, depth) } else { [Vec3::ZERO; 6] };
    for j in 0..g.Nf as usize {
        let p = &g.pg[j];
        let (gS, gD) = rankine_grad(p, x);
        let (gSb, gDb) = rankine_grad(p, &xb); // d/dz of a function of the image point is -d/dz'
        let mut h1 = dot(*nx, gD);
        let mut k1 = dot(*nx, gS);
        let h2 = down(nx, &gDb);
        let k2 = down(nx, &gSb);
        if finite {
            // bottom images: (z + zeta + 2h) and (z + zeta + 4h) are z-reflections,
            // (z - zeta + 2h), (-z + zeta + 2h) translations of x; v2 belongs to the
            // first family (sign + at omega = inf), v4..v6 to the second (sign -)
            let (mut hb, mut kb, mut hm, mut km) = (0.0, 0.0, 0.0, 0.0);
            for q in [1usize, 3, 4, 5] {
                let (s, d) = rankine_grad(p, &xp[q]);
                let flip = q == 1 || q == 5;
                let hq = if flip { down(nx, &d) } else { dot(*nx, d) };
                let kq = if flip { down(nx, &s) } else { dot(*nx, s) };
                if q == 1 {
                    hb += hq;
                    kb += kq;
                } else {
                    hm += hq;
                    km += kq;
                }
            }
            if let Some(inf) = inf.as_deref_mut() {
                inf[j] = [R::from_f64(h1 + hb - h2 - hm), R::from_f64(k1 + kb - k2 - km)];
            }
            h1 += hb + hm;
            k1 += kb + km;
        }
        let (mut hl, mut kl) = (0.0, 0.0);
        // a row in z = 0 is a free-surface plate's, used at omega = inf only (K = 0: no log
        // term); there the log singularity psi = log(r' + |z + zeta|) of a panel in z = 0 is singular
        if logint && x[2] != 0.0 && log_pair(p, x, tol) {
            let (Lk, Lh) = log_singularity_grad(p, x, nx);
            let (kc, hc) = psi_derivatives(x, &p.c, nx, &p.n);
            hl = Lh - p.area * hc;
            kl = Lk - p.area * kc;
        }
        row[j] = [R::from_f64(h1 + h2), R::from_f64(k1 + k2), R::from_f64(hl), R::from_f64(kl)];
        if !finite {
            if let Some(inf) = inf.as_deref_mut() {
                inf[j] = [R::from_f64(h1 - h2), R::from_f64(k1 - k2)];
            }
        }
    }
}

// Source-formulation row of the sector panel at x: grad_x of the
// Rankine source integral of every panel with all its images (z-reflected
// images change the sign of the z derivative), principal value on the panel
// itself, and the gradient of the log-singularity correction.
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
pub(crate) fn source_row<R: Real>(
    g: &Geometry,
    x: &Vec3,
    logint: bool,
    tol: f64,
    depth: f64,
    vg: &mut [[R; 3]],
    vl: &mut [[R; 3]],
) {
    let finite = depth.is_finite();
    let mut xp = [Vec3::ZERO; 6];
    let flip: [bool; 6];
    let mut np = 2;
    if finite {
        xp = image_points(x, depth);
        flip = [false, true, true, false, false, true];
        np = 6;
    } else {
        xp[0] = *x;
        xp[1] = image(x);
        flip = [false, true, false, false, false, false];
    }
    for j in 0..g.Nf as usize {
        let p = &g.pg[j];
        let mut sum = Vec3::ZERO;
        for q in 0..np {
            let (mut gS, _) = rankine_grad(p, &xp[q]);
            if flip[q] {
                gS[2] = -gS[2];
            }
            sum += gS;
        }
        vg[j] = [R::from_f64(sum[0]), R::from_f64(sum[1]), R::from_f64(sum[2])];
        vl[j] = [R::from_f64(0.0); 3];
        if logint && log_pair(p, x, tol) {
            let L = log_singularity_grad3(p, x);
            // the one-point term of the log part is subtracted where the kernel's one-point value
            // holds it; at a panel's own centroid in z = 0 the kernel takes only its regular part
            // (pair_kernel), and psi is singular there: the integral alone
            let dz = x[2] + p.c[2];
            let own_in_z0 = x[0] == p.c[0] && x[1] == p.c[1] && dz == 0.0;
            let c = if own_in_z0 { L } else { L - p.area * psi_gradient(x, &p.c) };
            vl[j] = [R::from_f64(c[0]), R::from_f64(c[1]), R::from_f64(c[2])];
        }
    }
}

/// Static influence data of the field points against all panels. velocity_rows: also the
/// velocity rows of the source formulation at the points, for the field velocity or the
/// control-surface points (the body rows have them whenever the source formulation is used).
/// The rows on the threads of the current pool, each row its own entries; work: the progress
/// of the rows and the cancel check before each.
#[allow(clippy::too_many_arguments)] // the operands of the formula, passed as they are
#[allow(non_snake_case)] // the notation of the theory documents (matrices, operators, counts)
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
pub fn compute_static<R: Real>(
    g: &Geometry,
    points: &[Vec3],
    logint: bool,
    tol: f64,
    depth: f64,
    want_inf: bool,
    body_rows: bool,
    velocity_rows: bool,
    velocity_from: usize,
    work: Work<'_>,
) -> Result<Static<R>> {
    let mut s = Static::<R>::default();
    let Nf = g.Nf as usize;
    let n = points.len() * Nf;
    s.e.resize(n, StaticEntry::default());
    if want_inf {
        s.inf.resize(n, [R::default(); 2]);
    }
    // the progress: the rows of every part below
    // (the source rows at the sector centroids and the sheet rows at the sheet sectors)
    let np = points.len() as i32;
    // the sheet rows: the sheets, and with omega = inf the free-surface plates (their rows at
    // omega = inf are sheet rows, theory §9.8)
    let sheet_row_of = |k: usize| {
        let kd = g.kind[g.sec[k] as usize];
        kd == SurfaceKind::Sheet || (want_inf && kd == SurfaceKind::Plate)
    };
    let sheet_rows = body_rows && (g.sheets || (want_inf && g.plates));
    let mut nsheet = 0;
    if sheet_rows {
        for k in 0..g.Ns as usize {
            nsheet += sheet_row_of(k) as i32;
        }
    }
    let total = np
        + (if g.W > 0.0 { np } else { 0 })
        + (if g.source && (body_rows || velocity_rows) {
            if body_rows { g.Ns } else { np - (velocity_from as i32).min(np) }
        } else {
            0
        })
        + nsheet * (if g.W > 0.0 { 2 } else { 1 });
    let done = AtomicI32::new(0);
    // a cancel check before every row
    let row_start = || work.check_cancel();
    let row_done = || work.report(Stage::Statics, done.fetch_add(1, Ordering::SeqCst) + 1, total);
    let np = np as usize;
    // point k writes its own row [k Nf, (k + 1) Nf)
    par_rows(&mut s.e, Nf, np).zip(par_rows_opt(&mut s.inf, Nf, np)).enumerate().try_for_each(|(k, (row, inf))| {
        row_start()?;
        static_row(g, &points[k], logint, tol, depth, want_inf, row, inf);
        row_done();
        Ok(())
    })?;
    if g.W > 0.0 {
        // channel: the static entries at the field points reflected in the walls
        s.e1.resize(n, StaticEntry::default());
        s.e2.resize(n, StaticEntry::default());
        if want_inf {
            s.inf1.resize(n, [R::default(); 2]);
            s.inf2.resize(n, [R::default(); 2]);
        }
        // point k writes its own rows
        let rows1 = par_rows(&mut s.e1, Nf, np).zip(par_rows_opt(&mut s.inf1, Nf, np));
        let rows2 = par_rows(&mut s.e2, Nf, np).zip(par_rows_opt(&mut s.inf2, Nf, np));
        rows1.zip(rows2).enumerate().try_for_each(|(k, ((e1, i1), (e2, i2)))| {
            row_start()?;
            let x = &points[k];
            static_row(g, &v3(x[0], g.W - x[1], x[2]), logint, tol, depth, want_inf, e1, i1);
            static_row(g, &v3(x[0], -g.W - x[1], x[2]), logint, tol, depth, want_inf, e2, i2);
            row_done();
            Ok(())
        })?;
    }
    // the source formulation at the sector centroids (body rows) or the field
    // points; in a channel also at these points reflected in the walls
    if g.source && (body_rows || velocity_rows) {
        let mut xs: Vec<Vec3> = Vec::new();
        if body_rows {
            for k in 0..g.Ns as usize {
                xs.push(g.pg[g.sec[k] as usize].c);
            }
        } else {
            // the field points from velocity_from on
            s.vfrom = velocity_from.min(points.len());
            xs = points[s.vfrom..].to_vec();
        }
        let nr = xs.len() * Nf;
        let z3 = [R::default(); 3];
        s.vg.resize(nr, z3);
        s.vl.resize(nr, z3);
        if g.W > 0.0 {
            s.vg1.resize(nr, z3);
            s.vl1.resize(nr, z3);
            s.vg2.resize(nr, z3);
            s.vl2.resize(nr, z3);
        }
        // point k writes its own rows
        let nx = xs.len();
        let rows = par_rows(&mut s.vg, Nf, nx).zip(par_rows(&mut s.vl, Nf, nx));
        let rows1 = par_rows_opt(&mut s.vg1, Nf, nx).zip(par_rows_opt(&mut s.vl1, Nf, nx));
        let rows2 = par_rows_opt(&mut s.vg2, Nf, nx).zip(par_rows_opt(&mut s.vl2, Nf, nx));
        rows.zip(rows1.zip(rows2)).enumerate().try_for_each(|(k, ((vg, vl), ((vg1, vl1), (vg2, vl2))))| {
            row_start()?;
            let x = &xs[k];
            source_row(g, x, logint, tol, depth, vg, vl);
            if let (Some(vg1), Some(vl1), Some(vg2), Some(vl2)) = (vg1, vl1, vg2, vl2) {
                // a channel (g.W > 0)
                source_row(g, &v3(x[0], g.W - x[1], x[2]), logint, tol, depth, vg1, vl1);
                source_row(g, &v3(x[0], -g.W - x[1], x[2]), logint, tol, depth, vg2, vl2);
            }
            row_done();
            Ok(())
        })?;
    }
    if sheet_rows {
        s.sheet = vec![-1; g.Ns as usize];
        let mut rows: Vec<i32> = Vec::new();
        for k in 0..g.Ns as usize {
            if sheet_row_of(k) {
                s.sheet[k] = rows.len() as i32;
                rows.push(k as i32);
            }
        }
        s.sh.resize(rows.len() * Nf, [R::default(); 4]);
        if want_inf {
            s.shinf.resize(s.sh.len(), [R::default(); 2]);
        }
        // row q writes its own entries
        let nr = rows.len();
        par_rows(&mut s.sh, Nf, nr).zip(par_rows_opt(&mut s.shinf, Nf, nr)).enumerate().try_for_each(
            |(q, (sh, shinf))| {
                row_start()?;
                let p = &g.pg[g.sec[rows[q] as usize] as usize];
                sheet_row(g, &p.c, &p.n, logint, tol, depth, sh, shinf);
                row_done();
                Ok(())
            },
        )?;
        if g.W > 0.0 {
            // channel: at the sheet points reflected in the walls, (x, W - y, z) and (x, -W - y, z),
            // n_x . grad_x of G(x_m, xi) being (R n_x) . grad at x_m
            s.sh1.resize(s.sh.len(), [R::default(); 4]);
            s.sh2.resize(s.sh.len(), [R::default(); 4]);
            if want_inf {
                s.shinf1.resize(s.sh.len(), [R::default(); 2]);
                s.shinf2.resize(s.sh.len(), [R::default(); 2]);
            }
            let rows1 = par_rows(&mut s.sh1, Nf, nr).zip(par_rows_opt(&mut s.shinf1, Nf, nr));
            let rows2 = par_rows(&mut s.sh2, Nf, nr).zip(par_rows_opt(&mut s.shinf2, Nf, nr));
            rows1.zip(rows2).enumerate().try_for_each(|(q, ((sh1, i1), (sh2, i2)))| {
                row_start()?;
                let p = &g.pg[g.sec[rows[q] as usize] as usize];
                let nr_ = v3(p.n[0], -p.n[1], p.n[2]);
                sheet_row(g, &v3(p.c[0], g.W - p.c[1], p.c[2]), &nr_, logint, tol, depth, sh1, i1);
                sheet_row(g, &v3(p.c[0], -g.W - p.c[1], p.c[2]), &nr_, logint, tol, depth, sh2, i2);
                row_done();
                Ok(())
            })?;
        }
    }
    Ok(s)
}
