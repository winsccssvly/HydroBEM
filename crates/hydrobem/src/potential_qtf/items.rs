//! The inner free surface and the bodies of every pair (theory §12.5-§12.9): the indirect method's
//! forces, and the direct method's extra columns of the sub-problem of the pair frequencies with
//! their solve (§12.6). Both methods asked for take one pass over the points.

use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;

use super::body::Motion;
use super::far_field::{ZoneRadial, cdiv, fit_map, ring_weights};
use super::forcing::{Lap, forcing};
use super::{CZ, First, I, Radiation, Setup};
use crate::constants::PI;
use crate::error::{Error, Result};
use crate::parallel::FrequencyPools;
use crate::problem::{DirectColumns, DirectFreq, Problem};
use crate::progress::{SecondOrderStep, SolveContext};
use crate::solver::solve;

/// The coefficients [F][F][H][H][n6] of no pair yet: NaN, zero at equal frequencies (Ω = 0) of the
/// differences.
#[allow(non_snake_case)] // the notation of the theory documents
pub(super) fn empty(s: &Setup) -> (Vec<Complex64>, Vec<Complex64>) {
    let (F, H, n6) = (s.nf, s.nh, s.n6);
    let nan = Complex64::new(f64::NAN, f64::NAN);
    let mut dif = vec![nan; F * F * H * H * n6];
    let sm = dif.clone();
    for f in 0..F {
        for ha in 0..H {
            for hb in 0..H {
                for c in 0..n6 {
                    dif[(((f * F + f) * H + ha) * H + hb) * n6 + c] = CZ; // Ω = 0
                }
            }
        }
    }
    (dif, sm)
}

/// The forces f6 of work item wi into both orders of its pair: F⁻_ba = F⁻_ab*, F⁺_ba = F⁺_ab.
#[allow(non_snake_case)] // the notation of the theory documents
fn store(s: &Setup, dif: &mut [Complex64], sm: &mut [Complex64], wi: usize, f6: &[Complex64]) {
    let (F, H, n6) = (s.nf, s.nh, s.n6);
    let (a, bb, ha, hb, sum) = s.work[wi];
    let at = |x: usize, y: usize, hx: usize, hy: usize, c: usize| (((x * F + y) * H + hx) * H + hy) * n6 + c;
    for j in 0..n6 {
        if sum {
            sm[at(a, bb, ha, hb, j)] = f6[j];
            sm[at(bb, a, hb, ha, j)] = f6[j];
        } else {
            dif[at(a, bb, ha, hb, j)] = f6[j];
            dif[at(bb, a, hb, ha, j)] = f6[j].conj();
        }
    }
}

/// The direct method's extra column of a work item: the scalar and gradient weights of its
/// free-surface term on the field points, its body condition u and Stokes vectors bt (global axes),
/// and the force of the second-order incident wave.
struct DirectItem {
    ws: Vec<Complex64>,
    wg: Vec<[Complex64; 2]>,
    u: Vec<Complex64>,
    bt: Vec<[Complex64; 3]>,
    fi2: Vec<Complex64>,
}

/// The coefficients of a work item: its forces by the indirect method (empty without it) and the
/// direct method's column.
struct Item {
    f6: Vec<Complex64>,
    direct: Option<DirectItem>,
}

/// What the items of one pair frequency o share: ψ_j at the quadrature zone's points
/// [j nqr + circle][t] and on the outer circle [j][t] (the indirect method); the radial functions
/// of the zone at Ω at the quadrature zone's circles, its wave denominators and G's far-field fit
/// (the direct method).
struct AtOmega {
    o: usize,
    psq: Vec<Vec<(Complex64, [Complex64; 2])>>,
    psl: Vec<Vec<(Complex64, [Complex64; 2])>>,
    gq: Vec<ZoneRadial>,
    den: Vec<Complex64>,
    gmap: Vec<[Complex64; 4]>,
}

/// The direct method's columns of every pair frequency (into the sub-problem of the pair
/// frequencies) and the forces of the second-order incident wave of every work item.
pub(super) struct DirectWork {
    dc: DirectColumns,
    fi2: Vec<Vec<Complex64>>,
}

/// What the pairs give: the indirect method's coefficients (difference, sum) and the direct
/// method's columns.
pub(super) struct Done {
    pub(super) indirect: Option<(Vec<Complex64>, Vec<Complex64>)>,
    pub(super) direct: Option<DirectWork>,
}

/// The work item wi at the pair frequency of `at`: the inner free surface (the panelled and the
/// quadrature zone), the line integrals of the divergence theorem (the outer circle and the
/// waterlines), the second-order incident wave and the motion terms of the bodies, for the
/// methods asked for (outer: its outer integral; mwl: the field point of every waterline side by
/// the free surface of each moving body with its place among the body's sides, the direct method).
#[allow(non_snake_case)] // the notation of the theory documents
#[allow(clippy::needless_range_loop)] // index notation of the formula over several arrays
#[allow(clippy::too_many_arguments)] // the phases' state, the item and its frequency
fn item(
    s: &Setup,
    first: &First,
    motions: &[Motion],
    rad: Option<&Radiation>,
    outer: &[Complex64],
    mwl: &[Vec<(usize, usize)>],
    at: &AtOmega,
    wi: usize,
) -> Item {
    let (m, z, fp, g, h) = (s.m, s.z, &s.fp, s.g, s.h);
    let (ind, dir) = (s.ind, s.dir);
    let (n6, H, hf, o) = (s.n6, s.nh, s.hf, at.o);
    let (Ni, Nq, P, vf0, st0) = (fp.ni, z.pts.len(), fp.len(), fp.vf0, fp.st0);
    let (M, Mf, nm, NR, NL) = (z.order, z.fit_order, s.nm, z.ring_points, z.line_points);
    let (nz, nqr, nci) = (z.quad_nodes, z.quad_radii.len(), s.ci.rho.len());
    let (c0, R) = (z.c0, z.outer_radius);
    let lay = &first.lay;
    let Nb = lay.len();
    let (a, bb, ha, hb, sum) = s.work[wi];
    let (wa, wb) = (s.wave(a, ha), s.wave(bb, hb));
    let om = if sum { wa.w + wb.w } else { wa.w - wb.w };
    let rp = rad.filter(|_| ind);
    let mut ws = if dir { vec![CZ; P] } else { Vec::new() };
    let mut wgv = if dir { vec![[CZ; 2]; P - vf0] } else { Vec::new() };
    // the direct method's free surface beyond the panelled zone as a functional of G's values on
    // the circles of the fit: per order (|m| <= Mf) its moments against the coefficients of the
    // wave and of the local term of G's far-field series (mh, ml: the asymptotic zone, outer,
    // |m| <= M) and against G's coefficients on the circles of the zone (mc: the quadrature
    // zone, Zone::modes, and the outer circle)
    let (mut mh, mut ml) = (vec![CZ; nm], vec![CZ; nm]);
    let mut mc = if dir { vec![vec![CZ; nm]; nci] } else { Vec::new() };
    if dir {
        for i in 0..2 * M + 1 {
            mh[Mf - M + i] = outer[n6 + i];
            ml[Mf - M + i] = outer[n6 + 2 * M + 1 + i];
        }
    }
    // the quadrature zone's weights on G and on the radial and azimuthal components of ∇_h G
    let mut qa = if dir { vec![[CZ; 3]; Nq - Ni] } else { Vec::new() };
    // the inner free surface: the panelled zone and the quadrature zone
    let mut fsi = vec![CZ; if ind { n6 } else { 0 }];
    let one = Complex64::new(1.0, 0.0);
    for q in 0..Nq {
        let (x, wq) = z.pts[q];
        let wq = wq * hf;
        let (pia, gia, gmia) = wa.incident(g, x[0], x[1]);
        let (pib, gib, gmib) = wb.incident(g, x[0], x[1]);
        // the disturbances: at a panelled point from the field points, beyond from the series
        let (cq, tq) = if q < Ni { (0, 0) } else { ((q - Ni) / nz, (q - Ni) % nz) };
        let (psa, gsa, psb, gsb) = if q < Ni {
            let (pa, ga) = first.sc(s, a, q, ha);
            let (pb, gb) = first.sc(s, bb, q, hb);
            (pa, ga, pb, gb)
        } else {
            let (pa, ga) = first.dq[(a * H + ha) * nqr + cq][tq];
            let (pb, gb) = first.dq[(bb * H + hb) * nqr + cq][tq];
            (pa, [ga[0], ga[1], wa.kk * pa], pb, [gb[0], gb[1], wb.kk * pb])
        };
        // γ of the disturbances without −g ∇_h²φ: that part by the divergence theorem below
        let gmsa = -wa.w * wa.w * wa.kk * psa;
        let gmsb = -wb.w * wb.w * wb.kk * psb;
        let fs_ = forcing(wa.w, wb.w, g, sum, pia, &gia, gmia, psb, &gsb, gmsb)
            + forcing(wa.w, wb.w, g, sum, psa, &gsa, gmsa, pib, &gib, gmib)
            + forcing(wa.w, wb.w, g, sum, psa, &gsa, gmsa, psb, &gsb, gmsb);
        let lp = Lap::new(sum, wa.w, wb.w, (pia, gia), (psa, gsa), (pib, gib), (psb, gsb));
        if let Some(rp) = rp {
            for j in 0..n6 {
                let (ps, gp) =
                    if q < Ni { (rp.psi_f(s, o, q, j), rp.gpsi_f(s, o, q, j)) } else { at.psq[j * nqr + cq][tq] };
                fsi[j] += (ps * fs_ + lp.area(ps, gp)) * wq;
            }
        }
        if !dir {
            continue;
        }
        // area(ψ, ∇ψ) is linear: ψ α + ∇ψ·β; ψ at q, ∇ψ by the stencil copies (or the velocity
        // copy next to a waterline); beyond the panelled zone against G's series
        let alpha = (fs_ + lp.area(one, [CZ, CZ])) * wq;
        let beta = [lp.area(CZ, [one, CZ]) * wq, lp.area(CZ, [CZ, one]) * wq];
        if q >= Ni {
            let (c, sn) = s.qcs[tq];
            qa[q - Ni] = [alpha, c * beta[0] + sn * beta[1], -sn * beta[0] + c * beta[1]];
            continue;
        }
        ws[q] += alpha;
        if fp.near[q] {
            // against ∇G at the point's mirror image beyond a wall: the weight mirrored
            let mut beta = beta;
            if let (Some(ax), true) = (z.wall, fp.flip[fp.vat[q]]) {
                beta[ax] = -beta[ax];
            }
            let v = &mut wgv[fp.vat[q] - vf0];
            v[0] += beta[0];
            v[1] += beta[1];
        } else {
            let st2 = 2.0 * fp.step;
            ws[st0 + q] += beta[0] / st2;
            ws[st0 + Ni + q] -= beta[0] / st2;
            ws[st0 + 2 * Ni + q] += beta[1] / st2;
            ws[st0 + 3 * Ni + q] -= beta[1] / st2;
        }
    }
    if dir {
        // per circle of the quadrature zone the Fourier sums of its weights, then the moments:
        // α f e^{imθ} + β·∇(f e^{imθ}) = e^{imθ} [α f + β_r f' + β_θ (im/r) f], f the radial
        // function of each coefficient (Zone::modes)
        for (c, &r) in z.quad_radii.iter().enumerate() {
            let mut sums = vec![[CZ; 3]; nm];
            for (t, w) in qa[c * nz..(c + 1) * nz].iter().enumerate() {
                for (sm_, e) in sums.iter_mut().zip(&s.qe[t * nm..(t + 1) * nm]) {
                    sm_[0] += w[0] * e;
                    sm_[1] += w[1] * e;
                    sm_[2] += w[2] * e;
                }
            }
            // v = Σ_j c_j [ℓ_j + dd_j g0 / den], v' = Σ_j c_j [ℓ_j' + dd_j g1 / den]
            let (g0, g1, l, d) = &at.gq[c];
            for (i, sm_) in sums.iter().enumerate() {
                let mo = i as f64 - Mf as f64;
                let (s0, s1) = (sm_[0] + I * (mo / r) * sm_[2], sm_[1]);
                let wave = if at.den[i] == CZ { CZ } else { cdiv(s0 * g0[i] + s1 * g1[i], at.den[i]) };
                for j in 0..nci {
                    mc[j][i] += l[j] * s0 + d[j] * s1 + s.ci.dd[j] * wave;
                }
            }
        }
    }
    // the line integrals of the divergence theorem: the outer circle (ν outward; the
    // disturbances and ψ from their series) and the waterlines (ν into the bodies)
    let mut la = if dir { vec![CZ; NL] } else { Vec::new() };
    for t in 0..NL {
        let (c, sn) = s.lcs[t];
        let x = [c0[0] + R * c, c0[1] + R * sn];
        let nu = [c, sn];
        let (pia, gia, _) = wa.incident(g, x[0], x[1]);
        let (pib, gib, _) = wb.incident(g, x[0], x[1]);
        let (psa, ga) = first.dl[a * H + ha][t];
        let (psb, gb) = first.dl[bb * H + hb][t];
        let lp =
            Lap::new(sum, wa.w, wb.w, (pia, gia), (psa, [ga[0], ga[1], CZ]), (pib, gib), (psb, [gb[0], gb[1], CZ]));
        let dlen = hf * 2.0 * PI * R / NL as f64;
        if dir {
            la[t] = lp.line(one, nu) * dlen;
        }
        if rp.is_some() {
            for j in 0..n6 {
                fsi[j] += lp.line(at.psl[j][t].0, nu) * dlen;
            }
        }
    }
    if dir {
        // G on the outer circle: its coefficients there (the last circle of the zone)
        for i in 0..nm {
            let mut sv = CZ;
            for (t, w) in la.iter().enumerate() {
                sv += w * s.le[t * nm + i];
            }
            mc[nci - 1][i] += sv;
        }
    }
    for (iw, w) in first.wlf.iter().enumerate() {
        let x = w.x;
        let (pia, gia, _) = wa.incident(g, x[0], x[1]);
        let (pib, gib, _) = wb.incident(g, x[0], x[1]);
        let (psa, gsa) = w.disturbance(a, ha, &pia, &gia);
        let (psb, gsb) = w.disturbance(bb, hb, &pib, &gib);
        let lp = Lap::new(sum, wa.w, wb.w, (pia, gia), (psa, gsa), (pib, gib), (psb, gsb));
        if let Some(rp) = rp {
            for j in 0..n6 {
                fsi[j] += lp.line(w.psi(|q| rp.psi_b(s, Nb, o, q, j)), w.nu) * w.dl;
            }
        }
        if dir {
            ws[fp.wl0 + iw] += lp.line(one, w.nu) * w.dl;
        }
    }
    // the second-order incident wave: the plane wave of κ = k_a ∓ k_b, with a wall the four of
    // the wave or its image of a with the wave or its image of b (each pair's forcing a plane
    // wave; their sum even about the wall)
    let sgn = if sum { 1.0 } else { -1.0 };
    let mut planes: Vec<(Complex64, [f64; 2], f64)> = Vec::new(); // amplitude / den, κ, |κ|
    for &ba in &wa.headings() {
        for &bh in &wb.headings() {
            let (p0a, g0a, m0a) = wa.plane(ba, g, 0.0, 0.0);
            let (p0b, g0b, m0b) = wb.plane(bh, g, 0.0, 0.0);
            let fi0 = forcing(wa.w, wb.w, g, sum, p0a, &g0a, m0a, p0b, &g0b, m0b);
            let kap = [wa.k * ba.cos() + sgn * wb.k * bh.cos(), wa.k * ba.sin() + sgn * wb.k * bh.sin()];
            let ak = (kap[0] * kap[0] + kap[1] * kap[1]).sqrt();
            let den = -om * om + if h.is_finite() { g * ak * (ak * h).tanh() } else { g * ak };
            planes.push((fi0 / den, kap, ak));
        }
    }
    let th_ = |ak: f64, zz: f64| -> (f64, f64) {
        if h.is_finite() {
            ((ak * (zz + h)).cosh() / (ak * h).cosh(), ak * (ak * (zz + h)).sinh() / (ak * h).cosh())
        } else {
            ((ak * zz).exp(), ak * (ak * zz).exp())
        }
    };
    let mut fb = vec![CZ; if ind { n6 } else { 0 }];
    let mut uq = if dir { vec![CZ; Nb] } else { Vec::new() };
    let mut fi2 = vec![CZ; if dir { n6 } else { 0 }]; // the force of the second-order incident wave
    for q in 0..Nb {
        let x = [lay.pts[3 * q], lay.pts[3 * q + 1], lay.pts[3 * q + 2]];
        let n = [lay.nrm[3 * q], lay.nrm[3 * q + 1], lay.nrm[3 * q + 2]];
        let (mut ph2, mut dn) = (CZ, CZ);
        for &(amp, kap, ak) in &planes {
            let (z0, z1) = th_(ak, x[2]);
            let e = Complex64::from_polar(1.0, -(kap[0] * x[0] + kap[1] * x[1]));
            let p = amp * z0 * e;
            ph2 += p;
            dn += p * (-I) * (kap[0] * n[0] + kap[1] * n[1]) + amp * z1 * e * n[2];
        }
        let k = lay.off[lay.owner[q]];
        for j in 0..lay.width(m, lay.owner[q]) {
            let v = ph2 * first.njs[q][j] * lay.area[q];
            if ind {
                fb[k + j] += v;
            }
            if dir {
                fi2[k + j] += v;
            }
        }
        if dir {
            uq[q] = -dn; // ∂φ_S⁽²⁾/∂n = −∂φ_I⁽²⁾/∂n on a fixed body
        }
        if let Some(rp) = rp {
            for j in 0..n6 {
                fb[j] -= rp.psi_b(s, Nb, o, q, j) * dn * lay.area[q];
            }
        }
    }
    // the motion terms of the body condition, the terms of each body once for both methods
    let mut btq = if dir && !motions.is_empty() { vec![[CZ; 3]; Nb] } else { Vec::new() };
    for (mi, mo) in motions.iter().enumerate() {
        let pt = mo.pair_terms(a, ha, bb, hb, sum);
        if rp.is_some() {
            for j in 0..n6 {
                fb[j] += mo.body_term(&pt, o, j);
            }
        }
        if dir {
            // the scalar part to u, the Stokes vector to bt (global axes), the waterline factor to
            // the scalar weight of its field point (−g l: E is divided by g, and the right-hand
            // side is −∫G Q_B)
            let b = &m.bodies[mo.bi];
            let (cy, sy) = ((b.yaw * PI / 180.0).cos(), (b.yaw * PI / 180.0).sin());
            for (k, &q) in mo.idx.iter().enumerate() {
                uq[q] += pt.u[k];
                let v = pt.t[k];
                btq[q] = [cy * v[0] - sy * v[1], sy * v[0] + cy * v[1], v[2]];
            }
            for &(k, i) in &mwl[mi] {
                ws[i] -= g * pt.l[k];
            }
            for (q, v) in mo.under_plates(&pt) {
                uq[q] += v;
            }
        }
    }
    let direct = dir.then(|| {
        // the free surface beyond the panelled zone: the moments through G's far-field fit on
        // the outer circle and 1.1 R (a_m = A1 c1 + A2 c2, b_m = B1 c1 + B2 c2), then as
        // weights on the points of every circle of the fit
        let mut mu = mc;
        mu.push(vec![CZ; nm]);
        for (i, p) in at.gmap.iter().enumerate() {
            if *p != [CZ; 4] {
                mu[nci - 1][i] += mh[i] * p[0] + ml[i] * p[2];
                mu[nci][i] += mh[i] * p[1] + ml[i] * p[3];
            }
        }
        for (j, mj) in mu.iter().enumerate() {
            for (t, w) in ring_weights(mj, NR).into_iter().enumerate() {
                ws[Ni + j * NR + t] += w;
            }
        }
        DirectItem { ws, wg: wgv, u: uq, bt: btq, fi2 }
    });
    let mut f6 = Vec::new();
    if ind {
        let rho = m.env.density;
        f6 = (0..n6).map(|j| I * rho * om * (fb[j] - (fsi[j] + outer[j]) / g)).collect();
    }
    Item { f6, direct }
}

/// Every work item, the pair frequencies on the frequency workers of the solve (each an item of its
/// second-order step; the work items of one in parallel on its worker's inner threads, a cancel
/// request checked before each): the indirect method's coefficients and the direct method's
/// columns (every pair frequency writes its own entries, so the order they end in does not change
/// them).
#[allow(non_snake_case)] // the notation of the theory documents
pub(super) fn pairs(
    s: &Setup,
    first: &First,
    motions: &[Motion],
    rad: Option<&Radiation>,
    outer: &[Vec<Complex64>],
    ctx: &SolveContext,
) -> Result<Done> {
    let z = s.z;
    let (n6, Nb, P, vf0) = (s.n6, first.lay.len(), s.fp.len(), s.fp.vf0);
    let (Mf, R, NO, nqr) = (z.fit_order, z.outer_radius, z.oms.len(), z.quad_radii.len());
    // the direct method: the field point of every waterline side of a moving body by the free
    // surface (the sides of the bodies in the order of the zones' list, whose midpoints follow wl0),
    // with the side's place among the body's sides (a side under a plate has none: Motion::under_plates)
    let mut mwl: Vec<Vec<(usize, usize)>> = Vec::new();
    if s.dir {
        for mo in motions {
            let v: Vec<usize> =
                (0..z.wsegs.len()).filter(|&i| z.wsegs[i].body == mo.bi).map(|i| s.fp.wl0 + i).collect();
            let k: Vec<usize> = (0..mo.wl.len()).filter(|&k| mo.free[k]).collect();
            if v.len() != k.len() {
                return Err(Error::internal("potential QTF: a waterline side without its field point"));
            }
            mwl.push(k.into_iter().zip(v).collect());
        }
    }
    let ind = s.ind.then(|| empty(s));
    let dw = s.dir.then(|| DirectWork {
        dc: DirectColumns {
            freqs: s
                .cols
                .iter()
                .map(|cl| {
                    let n = cl.len();
                    DirectFreq {
                        ncols: n,
                        u: vec![CZ; n * Nb],
                        ws: vec![CZ; P * n],
                        wg: vec![[CZ; 2]; (P - vf0) * n],
                        bt: if motions.is_empty() { Vec::new() } else { vec![[CZ; 3]; Nb * n] },
                        ..Default::default()
                    }
                })
                .collect(),
        },
        fi2: vec![Vec::new(); s.work.len()],
    });
    let acc = std::sync::Mutex::new((ind, dw));
    let pools = FrequencyPools::new(s.m.opt.frequency_threads, s.m.opt.inner_threads, NO)?;
    let step = ctx.step(SecondOrderStep::PotentialPairs, NO);
    pools.for_each_item(NO, &step, |it| {
        let o = it.index();
        if s.cols[o].is_empty() {
            return Ok(());
        }
        it.check_cancel()?;
        let (ko, kk) = (z.ko[o], z.oms[o] * z.oms[o] / s.g);
        let gq = s.zone_radial(ko);
        let (psq, psl) = match rad.filter(|_| s.ind) {
            Some(rp) => (
                (0..n6 * nqr)
                    .into_par_iter()
                    .map(|i| s.in_zone(&rp.zones[o * n6 + i / nqr], &gq[i % nqr], i % nqr))
                    .collect(),
                (0..n6).into_par_iter().map(|j| s.on_outer(&rp.zones[o * n6 + j])).collect(),
            ),
            None => (Vec::new(), Vec::new()),
        };
        let at = AtOmega {
            o,
            psq,
            psl,
            den: s.ci.wave_denominators(ko, Mf),
            gmap: if s.dir { fit_map(Mf, ko, kk, R, super::far_field::fit_ratio() * R, s.h).1 } else { Vec::new() },
            gq,
        };
        let n = s.cols[o].len();
        let done = std::sync::atomic::AtomicUsize::new(0);
        let items: Vec<Item> = s.cols[o]
            .par_iter()
            .map(|&wi| {
                it.check_cancel()?;
                let x = item(s, first, motions, rad, &outer[wi], &mwl, &at, wi);
                it.report(done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1, n);
                Ok(x)
            })
            .collect::<Result<_>>()?;
        let mut g = acc.lock().unwrap_or_else(|p| p.into_inner());
        let (ind, dw) = &mut *g;
        for (c, (&wi, it)) in s.cols[o].iter().zip(items).enumerate() {
            if let Some((dif, sm)) = ind {
                store(s, dif, sm, wi, &it.f6);
            }
            if let (Some(dw), Some(di)) = (&mut *dw, it.direct) {
                // with a wall the solve projects its right-hand side on the characters even about
                // the wall, the mean of every panel and its wall image: the body terms (given on
                // the bodies, none on their images) and the free surface (half of the whole plane)
                // twice
                let w = if z.wall.is_some() { 2.0 } else { 1.0 };
                let d = &mut dw.dc.freqs[o];
                let n = d.ncols;
                for (q, v) in di.u.iter().enumerate() {
                    d.u[c * Nb + q] = w * v;
                }
                for (q, v) in di.bt.iter().enumerate() {
                    d.bt[q * n + c] = v.map(|x| w * x);
                }
                for (q, v) in di.ws.iter().enumerate() {
                    d.ws[q * n + c] = w * v;
                }
                for (q, v) in di.wg.iter().enumerate() {
                    d.wg[q * n + c] = v.map(|x| w * x);
                }
                dw.fi2[wi] = di.fi2;
            }
        }
        Ok(())
    })?;
    let (ind, dw) = acc.into_inner().unwrap_or_else(|p| p.into_inner());
    Ok(Done { indirect: ind, direct: dw })
}

/// The direct method (theory §12.6): the second-order disturbance potential of every pair
/// solved in the sub-problem of its pair frequency sb, with the same matrix as the radiation there;
/// its force on the body points with that of the second-order incident wave. The coefficients
/// (difference, sum).
#[allow(non_snake_case)] // the notation of the theory documents
pub(super) fn direct_solve(
    s: &Setup,
    first: &First,
    dw: DirectWork,
    sb: &mut Problem,
    ctx: &SolveContext,
) -> Result<(Vec<Complex64>, Vec<Complex64>)> {
    let (m, lay) = (s.m, &first.lay);
    let Nb = lay.len();
    let rho = m.env.density;
    let dc = Arc::new(dw.dc);
    sb.direct = Some(dc.clone());
    // the sub-problem of the pair frequencies once more, as a step of the second order
    let (limits, n) = crate::model_solve::board_frequencies(sb);
    let sp = ctx.sub_problem(SecondOrderStep::PotentialDirect, limits, n);
    let r = solve(sb, &sp.ctx);
    sp.end(r.is_ok());
    let _ = r?;
    let (mut dif, mut sm) = empty(s);
    for (o, cl) in s.cols.iter().enumerate() {
        let phis = dc.freqs[o].out.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if phis.len() != cl.len() * Nb {
            return Err(Error::internal("potential QTF: the direct method's solution is missing"));
        }
        for (c, &wi) in cl.iter().enumerate() {
            let om = s.omega(wi);
            let mut f6 = dw.fi2[wi].clone();
            for q in 0..Nb {
                let k = lay.off[lay.owner[q]];
                for j in 0..lay.width(m, lay.owner[q]) {
                    f6[k + j] += phis[c * Nb + q] * first.njs[q][j] * lay.area[q];
                }
            }
            for v in f6.iter_mut() {
                *v *= I * rho * om;
            }
            store(s, &mut dif, &mut sm, wi, &f6);
        }
    }
    Ok((dif, sm))
}
